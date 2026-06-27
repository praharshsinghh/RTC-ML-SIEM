"""
src/pipeline/pipeline.py
========================
End-to-end network threat detection pipeline (Phase 6).

Execution flow:
  Input (CSV / Parquet / DataFrame)
    → detect_preprocessing_needed()        # skip if already preprocessed
    → DataCleaner.transform()              # only on raw CSV input
    → EnsembleDetector.predict()           # RF + XGB + IF + AE
    → AttackEnricher.enrich_batch()        # MITRE ATT&CK enrichment
    → report_generator                     # JSON incident reports

Key implementation notes
------------------------
Models are loaded from disk, never retrained here. Saving and loading
serialised models (joblib for sklearn, torch.save for PyTorch) guarantees
byte-for-byte reproducibility and enables cold-start in seconds.

The pipeline auto-detects whether the input data is raw or already
preprocessed by inspecting the dtype of the categorical columns (proto,
state, service). Raw data contains strings; preprocessed Parquet files
contain integer-encoded values. This prevents double preprocessing when
the pipeline is called with processed files from data/processed/.

Severity is computed inside EnsembleDetector using a two-dimensional
(confidence × agreement) matrix, which prevents a single high-weight
model from escalating an alert to CRITICAL when other models disagree.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from src.preprocessing.cleaner import DataCleaner, CAT_FEATURES
from src.models.random_forest import RandomForestDetector
from src.models.xgboost_model import XGBoostDetector
from src.models.isolation_forest import IsolationForestDetector
from src.models.autoencoder import DenseAutoencoderDetector
from src.ensemble.detector import EnsembleDetector
from src.ensemble.schemas import EnsembleResult
from src.attack_mapping.enricher import AttackEnricher
from src.attack_mapping.schemas import EnrichedIncident
from src.pipeline.report_generator import (
    build_incident_report,
    save_incident_reports,
    generate_summary,
)
from src.pipeline.utils import (
    load_input_data,
    read_feature_list,
    align_features,
    now_utc_iso,
)

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).parent.parent.parent
_DEFAULT_CLEANER_PATH       = _ROOT / "data" / "processed" / "cleaner.joblib"
_DEFAULT_FEATURE_LIST_PATH  = _ROOT / "data" / "processed" / "feature_list.txt"
_DEFAULT_RF_PATH            = _ROOT / "data" / "models"    / "rf_detector.joblib"
_DEFAULT_XGB_PATH           = _ROOT / "data" / "models"    / "xgb_detector.joblib"
_DEFAULT_ISO_PATH           = _ROOT / "models"             / "isolation_forest.joblib"
_DEFAULT_AE_PATH            = _ROOT / "models"             / "autoencoder.pt"
_DEFAULT_INCIDENTS_DIR      = _ROOT / "reports"            / "incidents"


def _needs_preprocessing(df: pd.DataFrame) -> bool:
    """
    Detect whether df contains raw (unprocessed) data.

    Inspects the dtype of the categorical columns (proto, state, service).
    Preprocessed Parquet files contain integer-encoded values produced by
    DataCleaner.fit_transform(); raw CSVs contain string values.

    Returns True  → raw data, DataCleaner.transform() must be applied.
    Returns False → already preprocessed, skip directly to inference.
    """
    for col in CAT_FEATURES:
        if col not in df.columns:
            continue
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            return True
    return False


class ThreatDetectionPipeline:
    """
    End-to-end network threat detection pipeline.

    Orchestrates preprocessing → inference → ensemble → MITRE enrichment →
    incident reporting in a single call to run().

    Parameters
    ----------
    cleaner_path : Path, optional
        Path to saved DataCleaner (cleaner.joblib).
    feature_list_path : Path, optional
        Path to feature_list.txt.
    rf_path : Path, optional
        Path to rf_detector.joblib.
    xgb_path : Path, optional
        Path to xgb_detector.joblib.
    iso_path : Path, optional
        Path to isolation_forest.joblib.
    ae_path : Path, optional
        Path to autoencoder.pt.
    output_dir : Path, optional
        Directory for output reports. Defaults to reports/incidents/.
    stix_preload : bool
        Load MITRE STIX data eagerly on init. Set False to defer until
        first enrichment call (faster startup in tests).
    batch_size : int
        Ensemble inference batch size.

    Example
    -------
    >>> pipeline = ThreatDetectionPipeline()
    >>> result = pipeline.run("data/processed/test.parquet")
    >>> print(result["summary"]["attack_count"])
    """

    def __init__(
        self,
        cleaner_path: Optional[Path] = None,
        feature_list_path: Optional[Path] = None,
        rf_path: Optional[Path] = None,
        xgb_path: Optional[Path] = None,
        iso_path: Optional[Path] = None,
        ae_path: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        stix_preload: bool = True,
        batch_size: int = 2048,
    ) -> None:
        self._cleaner_path      = Path(cleaner_path)      if cleaner_path      else _DEFAULT_CLEANER_PATH
        self._feature_list_path = Path(feature_list_path) if feature_list_path else _DEFAULT_FEATURE_LIST_PATH
        self._rf_path           = Path(rf_path)           if rf_path           else _DEFAULT_RF_PATH
        self._xgb_path          = Path(xgb_path)          if xgb_path          else _DEFAULT_XGB_PATH
        self._iso_path          = Path(iso_path)          if iso_path          else _DEFAULT_ISO_PATH
        self._ae_path           = Path(ae_path)           if ae_path           else _DEFAULT_AE_PATH
        self._output_dir        = Path(output_dir)        if output_dir        else _DEFAULT_INCIDENTS_DIR
        self._batch_size        = batch_size
        self._stix_preload      = stix_preload

        self._cleaner:  Optional[DataCleaner]        = None
        self._features: Optional[list[str]]          = None
        self._ensemble: Optional[EnsembleDetector]   = None
        self._enricher: Optional[AttackEnricher]     = None

    def load_preprocessor(self) -> None:
        """
        Load the fitted DataCleaner and feature list from disk.

        The DataCleaner was persisted by Phase 1 and contains all fitted
        state: LabelEncoders, IQR clip bounds, StandardScaler. Loading it
        (not re-fitting) prevents data leakage and guarantees the same
        numerical transformations as during training.
        """
        logger.info("[Pipeline] Loading preprocessing artifacts")
        self._cleaner = DataCleaner.load(self._cleaner_path)
        self._features = read_feature_list(self._feature_list_path)
        logger.info(f"[Pipeline] Feature list: {len(self._features)} features")

    def load_models(self) -> None:
        """
        Load all four trained models from disk and construct the EnsembleDetector.

        Model paths
        -----------
        RF          → data/models/rf_detector.joblib
        XGBoost     → data/models/xgb_detector.joblib
        Isolation F → models/isolation_forest.joblib
        Autoencoder → models/autoencoder.pt
        """
        logger.info("[Pipeline] Loading models")

        rf  = RandomForestDetector.load_model(self._rf_path)
        xgb = XGBoostDetector.load_model(self._xgb_path)
        iso = IsolationForestDetector.load_model(self._iso_path)
        ae  = DenseAutoencoderDetector.load_model(self._ae_path)

        self._ensemble = EnsembleDetector(
            rf=rf, xgb=xgb, iso=iso, ae=ae, batch_size=self._batch_size
        )
        logger.info("[Pipeline] All models loaded")

    def preprocess(self, df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        """
        Prepare the feature matrix for inference.

        Auto-detects whether the input is raw or already preprocessed by
        inspecting the dtype of categorical columns (proto, state, service).
        Raw data (string dtypes) is passed through DataCleaner.transform()
        exactly once. Pre-processed Parquet files (integer dtypes) skip
        the transform step and proceed directly to feature extraction.

        Calling DataCleaner.transform() on already-scaled data would apply
        a second round of z-scoring using a scaler whose mean/std were
        fitted on original raw-scale values, producing completely wrong
        feature values and collapsing all model predictions to ATTACK.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame. Must contain all 43 feature columns from
            feature_list.txt, either as raw strings or encoded integers.

        Returns
        -------
        df : pd.DataFrame
            The (possibly transformed) DataFrame, unchanged beyond preprocessing.
        X : np.ndarray, shape (n_samples, n_features), dtype float32
            Feature matrix ready for model inference.
        """
        if self._features is None:
            raise RuntimeError(
                "Feature list not loaded. Call load_preprocessor() first."
            )
        if self._cleaner is None:
            raise RuntimeError(
                "DataCleaner not loaded. Call load_preprocessor() first."
            )

        if _needs_preprocessing(df):
            logger.info(
                f"[Pipeline] Raw input detected — applying DataCleaner.transform() "
                f"on {len(df):,} rows"
            )
            df = self._cleaner.transform(df)
        else:
            logger.info(
                f"[Pipeline] Preprocessed input detected — skipping DataCleaner "
                f"({len(df):,} rows)"
            )

        missing_cols = [c for c in self._features if c not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Input is missing {len(missing_cols)} required feature column(s): "
                f"{missing_cols[:10]}. "
                "Ensure the input was produced by DataCleaner.fit_transform() "
                "or is a raw UNSW-NB15 CSV."
            )

        X = df[self._features].values.astype(np.float32)
        logger.info(
            f"[Pipeline] Feature matrix: {X.shape[0]:,} × {X.shape[1]}"
        )
        return df, X

    def run_supervised(self, X: np.ndarray) -> list:
        """
        Run Random Forest and XGBoost predictions independently.

        Exposed for transparency and unit testing. In the standard run()
        flow, all four models are invoked together by ensemble_prediction().

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        list[list[PredictionResult]] : [rf_preds, xgb_preds]
        """
        if self._ensemble is None:
            raise RuntimeError("Models not loaded. Call load_models() first.")

        rf_preds  = self._ensemble.rf.predict(X)
        xgb_preds = self._ensemble.xgb.predict(X)
        return [rf_preds, xgb_preds]

    def run_unsupervised(self, X: np.ndarray) -> list:
        """
        Run Isolation Forest and Dense Autoencoder predictions independently.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        list[list[PredictionResult]] : [iso_preds, ae_preds]
        """
        if self._ensemble is None:
            raise RuntimeError("Models not loaded. Call load_models() first.")

        iso_preds = self._ensemble.iso.predict(X)
        ae_preds  = self._ensemble.ae.predict(X)
        return [iso_preds, ae_preds]

    def ensemble_prediction(
        self,
        X: np.ndarray,
        timestamps: Optional[list[str]] = None,
    ) -> list[EnsembleResult]:
        """
        Run the full ensemble on X and return one EnsembleResult per row.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        timestamps : list[str], optional
            ISO-8601 timestamps for each row. Defaults to current UTC.

        Returns
        -------
        list[EnsembleResult]
        """
        if self._ensemble is None:
            raise RuntimeError("Models not loaded. Call load_models() first.")

        logger.info("[Pipeline] Running ensemble inference")
        results = self._ensemble.predict(X, timestamps=timestamps)
        attack_count = sum(1 for r in results if r.final_verdict == "ATTACK")
        logger.info(
            f"[Pipeline] Ensemble complete: {attack_count:,} attacks "
            f"in {len(results):,} records"
        )
        return results

    def mitre_mapping(
        self,
        ensemble_results: list[EnsembleResult],
    ) -> list[EnrichedIncident]:
        """
        Enrich EnsembleResult objects with MITRE ATT&CK threat intelligence.

        Enrichment is applied post-detection so that ATT&CK labels cannot
        influence model behaviour. NORMAL verdicts receive a minimal
        EnrichedIncident with is_mapped=False.

        Parameters
        ----------
        ensemble_results : list[EnsembleResult]

        Returns
        -------
        list[EnrichedIncident] — same length as input
        """
        if self._enricher is None:
            logger.info("[Pipeline] Initialising MITRE ATT&CK enricher")
            self._enricher = AttackEnricher(preload=self._stix_preload)

        logger.info(
            f"[Pipeline] Running MITRE enrichment on {len(ensemble_results):,} results"
        )
        incidents = self._enricher.enrich_batch(ensemble_results)
        logger.info("[Pipeline] MITRE enrichment complete")
        return incidents

    def assign_severity(self, incidents: list[EnrichedIncident]) -> list[EnrichedIncident]:
        """
        Pass-through: severity is already assigned by EnsembleDetector.

        Exposed to make the pipeline flow explicit and to allow future
        override of the severity logic without modifying the ensemble module.

        Severity matrix (confidence × agreement):
            CRITICAL : conf >= 0.85 AND agreement >= 0.85
            HIGH     : conf >= 0.70 AND agreement >= 0.70
            MEDIUM   : conf >= 0.50 AND agreement >= 0.50
            LOW      : any other ATTACK verdict

        Parameters
        ----------
        incidents : list[EnrichedIncident]

        Returns
        -------
        list[EnrichedIncident] — unchanged
        """
        return incidents

    def generate_reports(
        self,
        incidents: list[EnrichedIncident],
        run_id: str,
        run_timestamp: str,
        input_source: str,
        row_indices: Optional[list[int]] = None,
    ) -> tuple[list[dict], dict]:
        """
        Build per-attack incident reports and a session-level summary.

        Only rows where verdict == 'ATTACK' produce a report dict.
        All rows contribute to the summary statistics.

        Parameters
        ----------
        incidents : list[EnrichedIncident]
        run_id : str
            Unique run identifier.
        run_timestamp : str
            Timestamp string for output filename (YYYYMMDD_HHMMSS).
        input_source : str
            Human-readable source label (filename or 'DataFrame').
        row_indices : list[int], optional
            Original DataFrame row indices. Defaults to 0, 1, 2, …

        Returns
        -------
        reports : list[dict]  — one report dict per attack row
        summary : dict        — session-level summary
        """
        if row_indices is None:
            row_indices = list(range(len(incidents)))

        logger.info("[Pipeline] Generating incident reports")
        reports: list[dict] = [
            build_incident_report(incident, idx, run_id)
            for idx, incident in zip(row_indices, incidents)
            if incident.verdict == "ATTACK"
        ]
        logger.info(
            f"[Pipeline] {len(reports)} attack report(s) from {len(incidents):,} records"
        )

        save_incident_reports(
            reports,
            output_dir=self._output_dir,
            run_timestamp=run_timestamp,
        )

        summary = generate_summary(
            all_incidents=incidents,
            run_id=run_id,
            run_timestamp=run_timestamp,
            input_source=input_source,
            output_dir=self._output_dir,
        )

        logger.info("[Pipeline] Reports saved")
        return reports, summary

    def run(
        self,
        source: Union[str, Path, pd.DataFrame],
        max_rows: Optional[int] = None,
    ) -> dict:
        """
        Execute the full end-to-end detection pipeline.

        Steps
        -----
        1. Load preprocessor (DataCleaner + feature_list.txt)
        2. Load all models (RF, XGB, IF, AE → EnsembleDetector)
        3. Load input data
        4. Detect and apply preprocessing if needed
        5. Run ensemble prediction
        6. MITRE ATT&CK enrichment
        7. Severity assignment (pass-through)
        8. Generate and save incident reports

        Parameters
        ----------
        source : str | Path | pd.DataFrame
            Input network traffic data (raw CSV or preprocessed Parquet).
        max_rows : int, optional
            Process only the first N rows.

        Returns
        -------
        dict with keys:
            run_id       : str
            run_timestamp: str  — YYYYMMDD_HHMMSS
            reports      : list[dict]  — one per attack row
            summary      : dict  — session-level statistics
            incidents    : list[EnrichedIncident]
        """
        run_id      = str(uuid.uuid4())
        run_ts_iso  = now_utc_iso()
        run_ts_file = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")

        logger.info("=" * 66)
        logger.info("[Pipeline] ThreatDetectionPipeline starting")
        logger.info(f"[Pipeline] Run ID    : {run_id}")
        logger.info(f"[Pipeline] Timestamp : {run_ts_iso}")
        logger.info("=" * 66)

        self.load_preprocessor()
        self.load_models()

        logger.info("[Pipeline] Loading input data")
        df = load_input_data(source)
        input_source = (
            str(source) if not isinstance(source, pd.DataFrame) else "DataFrame"
        )
        if max_rows is not None:
            df = df.head(max_rows)
            logger.info(f"[Pipeline] Truncated to {max_rows:,} rows (max_rows)")

        _, X = self.preprocess(df)
        row_indices = list(df.index)

        ensemble_results = self.ensemble_prediction(X)
        incidents        = self.mitre_mapping(ensemble_results)
        incidents        = self.assign_severity(incidents)

        reports, summary = self.generate_reports(
            incidents=incidents,
            run_id=run_id,
            run_timestamp=run_ts_file,
            input_source=input_source,
            row_indices=row_indices,
        )

        logger.info("=" * 66)
        logger.info("[Pipeline] Pipeline completed")
        logger.info(
            f"[Pipeline] Records: {summary['total_records']:,} total | "
            f"{summary['attack_count']:,} attacks | "
            f"{summary['normal_count']:,} normal"
        )
        logger.info(
            f"[Pipeline] Severity: "
            f"CRITICAL={summary['severity_distribution']['CRITICAL']}, "
            f"HIGH={summary['severity_distribution']['HIGH']}, "
            f"MEDIUM={summary['severity_distribution']['MEDIUM']}, "
            f"LOW={summary['severity_distribution']['LOW']}"
        )
        logger.info("=" * 66)

        return {
            "run_id":        run_id,
            "run_timestamp": run_ts_file,
            "reports":       reports,
            "summary":       summary,
            "incidents":     incidents,
        }
