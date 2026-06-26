"""
src/pipeline/pipeline.py — ThreatDetectionPipeline: End-to-End Orchestrator
=============================================================================
Phase 6: End-to-End Detection Pipeline.

The ThreatDetectionPipeline is the top-level orchestrator that connects
all previous phases into a single, callable workflow:

    Input CSV / Parquet / DataFrame
      ↓
    Preprocess (Phase 1: DataCleaner saved state)
      ↓
    Load models (Phase 2: RF, XGB  |  Phase 3: IsolationForest, DenseAutoencoder)
      ↓
    Run predictions (all four models via EnsembleDetector — Phase 4)
      ↓
    Ensemble voting + confidence + severity (Phase 4)
      ↓
    MITRE ATT&CK enrichment (Phase 5: AttackEnricher)
      ↓
    Assign severity (already embedded in EnsembleDetector)
      ↓
    Generate incident reports (Phase 6: report_generator)
      ↓
    Save JSON to reports/incidents/

Why models are loaded, not retrained
--------------------------------------
Training RF on 2M+ samples takes ~45 minutes; XGBoost ~10 minutes; the
autoencoder ~1 hour. Saving/loading serialised models (joblib for sklearn,
torch.save for PyTorch) enables instant cold-start in seconds. More
importantly, loading ensures byte-for-byte reproducibility: the inference
model is IDENTICAL to the evaluated model — re-training would produce a
statistically-equivalent but slightly different model due to random seeds.

How input data is prepared for inference
-----------------------------------------
The processed parquet files (train.parquet, test.parquet,
train_normal_only.parquet) were saved by Phase 1's DataCleaner.fit_transform().
They already contain fully preprocessed, z-score-scaled feature values. The
DataCleaner is still loaded here to provide two things:
  1. The canonical list of 43 feature columns (via feature_list.txt)
  2. Availability for future use if raw (un-processed) CSV inputs are added

CRITICAL: DataCleaner.transform() must NOT be called on data that already came
through fit_transform() (i.e., the parquet files). Doing so applies a second
round of z-scoring using a scaler whose mean/std were learned on the original
raw-scale data. This double-scaling shifts every feature value far outside the
range the models were trained on and collapses all model predictions to ATTACK.

The correct pattern — identical to run_supervised.py and run_unsupervised.py —
is to select the 43 feature columns directly from the DataFrame and cast to
float32. No additional scaling is applied.

Why the ensemble improves robustness
--------------------------------------
RF and XGB are high-precision supervised classifiers but may miss zero-day
attacks outside their training distribution. IsolationForest and the Dense
Autoencoder detect statistical anomalies without labels — they are the
\"unknown threat\" detectors. The weighted vote (RF=0.35, XGB=0.35, IF=0.15,
AE=0.15) fuses these complementary signals: supervised precision + anomaly
breadth. When all four agree, the alert is near-certain; when only
unsupervised models fire, the alert is lower priority but still captured.

Why MITRE enrichment is applied after detection
-------------------------------------------------
Enrichment adds contextual threat intelligence (technique ID, tactic, links)
to an already-established ML verdict. It does not influence the verdict —
enriching before detection would risk selection bias (using threat-intel
labels as implicit features). The strictly sequential design keeps detection
(statistical) and contextualisation (knowledge-based) separate and auditable.

How severity is computed
-------------------------
Severity = f(confidence, agreement_score):
  CRITICAL : conf ≥ 0.85 AND agreement ≥ 0.85
  HIGH     : conf ≥ 0.70 AND agreement ≥ 0.70
  MEDIUM   : conf ≥ 0.50 AND agreement ≥ 0.50
  LOW      : any ATTACK not meeting the above thresholds
This two-dimensional matrix prevents a single high-weight model from
escalating to CRITICAL if the other models disagree.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

# ── Phase 1: Preprocessing ────────────────────────────────────────────────────
from src.preprocessing.cleaner import DataCleaner

# ── Phase 2 & 3: Individual models ───────────────────────────────────────────
from src.models.random_forest import RandomForestDetector
from src.models.xgboost_model import XGBoostDetector
from src.models.isolation_forest import IsolationForestDetector
from src.models.autoencoder import DenseAutoencoderDetector

# ── Phase 4: Ensemble ─────────────────────────────────────────────────────────
from src.ensemble.detector import EnsembleDetector
from src.ensemble.schemas import EnsembleResult

# ── Phase 5: MITRE ATT&CK ─────────────────────────────────────────────────────
from src.attack_mapping.enricher import AttackEnricher
from src.attack_mapping.schemas import EnrichedIncident

# ── Phase 6: Report generation ────────────────────────────────────────────────
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

# ── Default file paths (relative to project root) ────────────────────────────
_ROOT = Path(__file__).parent.parent.parent
_DEFAULT_CLEANER_PATH       = _ROOT / "data" / "processed" / "cleaner.joblib"
_DEFAULT_FEATURE_LIST_PATH  = _ROOT / "data" / "processed" / "feature_list.txt"
_DEFAULT_RF_PATH            = _ROOT / "data" / "models"    / "rf_detector.joblib"
_DEFAULT_XGB_PATH           = _ROOT / "data" / "models"    / "xgb_detector.joblib"
_DEFAULT_ISO_PATH           = _ROOT / "models"             / "isolation_forest.joblib"
_DEFAULT_AE_PATH            = _ROOT / "models"             / "autoencoder.pt"
_DEFAULT_INCIDENTS_DIR      = _ROOT / "reports"            / "incidents"


class ThreatDetectionPipeline:
    """
    End-to-end network threat detection pipeline (Phase 6).

    Orchestrates preprocessing → inference → ensemble → enrichment →
    reporting in a single call to run().

    Parameters
    ----------
    cleaner_path : Path, optional
        Path to saved DataCleaner (cleaner.joblib).
    feature_list_path : Path, optional
        Path to feature_list.txt.
    rf_path : Path, optional
        Path to rf_detector.joblib (Phase 2).
    xgb_path : Path, optional
        Path to xgb_detector.joblib (Phase 2).
    iso_path : Path, optional
        Path to isolation_forest.joblib (Phase 3).
    ae_path : Path, optional
        Path to autoencoder.pt (Phase 3).
    output_dir : Path, optional
        Where to write reports. Defaults to reports/incidents/.
    stix_preload : bool
        If True, MITRE STIX data is loaded eagerly on init. Set False to
        speed up __init__ when running tests that don't call enrich.
    batch_size : int
        Ensemble prediction batch size. Reduce on memory-constrained hardware.

    Usage
    -----
    >>> pipeline = ThreatDetectionPipeline()
    >>> result = pipeline.run("data/test.parquet")
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

        # Lazy-loaded state
        self._cleaner:  Optional[DataCleaner]        = None
        self._features: Optional[list[str]]          = None
        self._ensemble: Optional[EnsembleDetector]   = None
        self._enricher: Optional[AttackEnricher]     = None
        self._stix_preload = stix_preload

    # ── Step 1: Load preprocessor ────────────────────────────────────────────

    def load_preprocessor(self) -> None:
        """
        Load the fitted DataCleaner and feature list from disk.

        The cleaner was saved by Phase 1 (run_preprocessing.py) and contains
        all fitted state: LabelEncoders, IQR clip bounds, StandardScaler.
        Loading it here (not re-fitting) is critical to avoid data leakage
        and to guarantee the same numerical transformation as during training.
        """
        logger.info("[Pipeline] Loading preprocessor (DataCleaner) …")
        self._cleaner = DataCleaner.load(self._cleaner_path)
        logger.info("[Pipeline] DataCleaner loaded ✓")

        logger.info("[Pipeline] Loading feature list …")
        self._features = read_feature_list(self._feature_list_path)
        logger.info(f"[Pipeline] {len(self._features)} features loaded ✓")

    # ── Step 2: Load all models ──────────────────────────────────────────────

    def load_models(self) -> None:
        """
        Load all four trained models from disk and construct the EnsembleDetector.

        Models are NEVER retrained here. The saved weights from Phases 2 & 3
        are loaded directly, guaranteeing inference is performed with the exact
        models that were evaluated and reported.

        Model file locations
        --------------------
        RF          → data/models/rf_detector.joblib
        XGBoost     → data/models/xgb_detector.joblib
        Isolation F → models/isolation_forest.joblib
        Autoencoder → models/autoencoder.pt
        """
        logger.info("[Pipeline] Loading models …")

        logger.info("[Pipeline]   → Loading Random Forest …")
        rf = RandomForestDetector.load_model(self._rf_path)

        logger.info("[Pipeline]   → Loading XGBoost …")
        xgb = XGBoostDetector.load_model(self._xgb_path)

        logger.info("[Pipeline]   → Loading Isolation Forest …")
        iso = IsolationForestDetector.load_model(self._iso_path)

        logger.info("[Pipeline]   → Loading Dense Autoencoder …")
        ae = DenseAutoencoderDetector.load_model(self._ae_path)

        self._ensemble = EnsembleDetector(
            rf=rf, xgb=xgb, iso=iso, ae=ae, batch_size=self._batch_size
        )
        logger.info("[Pipeline] All models loaded ✓")

    # ── Step 3: Preprocess input data ────────────────────────────────────────

    def preprocess(self, df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        """
        Select and align the 43 model-ready feature columns from df.

        WHY NO DataCleaner.transform() IS CALLED HERE
        -----------------------------------------------
        The processed parquet files used as pipeline input were produced by
        DataCleaner.fit_transform() during Phase 1. They already contain
        z-score-scaled, label-encoded values. All four models (RF, XGBoost,
        IsolationForest, DenseAutoencoder) were trained on these values via
        the pattern `df[feature_names].values.astype(float32)`, as implemented
        in run_supervised.py (line 104) and run_unsupervised.py (line 103).

        Calling DataCleaner.transform() on already-scaled data applies a SECOND
        round of z-scoring. Because the scaler's mean_ and scale_ were fitted
        on the original raw values (e.g., sport mean ~30534), applying them to
        already-normalised values (sport z-score ~-0.5 to +1.5) produces
        completely wrong outputs — e.g., sport becomes -1.49 for every row
        regardless of actual value. This caused 100% ATTACK predictions for
        XGBoost and DenseAutoencoder (Bug confirmed in diagnostic report).

        The fix mirrors the training scripts exactly: extract the feature
        columns by name from the loaded DataFrame and cast to float32.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame loaded from a processed parquet/CSV file.
            Must contain all 43 feature columns from feature_list.txt.

        Returns
        -------
        df : pd.DataFrame — the input DataFrame, unchanged (returned for
             downstream access to non-feature columns like label/attack_cat)
        X  : np.ndarray (n_samples, n_features), dtype float32
        """
        if self._features is None:
            raise RuntimeError(
                "Feature list not loaded. Call load_preprocessor() first."
            )

        logger.info(f"[Pipeline] Preparing feature matrix from {len(df):,} rows …")

        missing_cols = [c for c in self._features if c not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Input DataFrame is missing {len(missing_cols)} required feature "
                f"column(s): {missing_cols[:10]}. "
                "Ensure the input was produced by DataCleaner.fit_transform() "
                "(i.e., is a processed parquet file from data/processed/)."
            )

        X = df[self._features].values.astype(np.float32)
        logger.info(
            f"[Pipeline] Feature matrix ready: "
            f"{X.shape[0]:,} samples × {X.shape[1]} features ✓"
        )
        return df, X

    # ── Step 4: Run individual model predictions ─────────────────────────────

    def run_supervised(self, X: np.ndarray) -> list:
        """
        Run Random Forest and XGBoost predictions.

        This method is exposed for transparency and testing. In the main
        run() flow, predictions from all four models are gathered together
        by ensemble_prediction(). Calling run_supervised() separately allows
        inspection of individual model outputs.

        Parameters
        ----------
        X : np.ndarray (n_samples, n_features)

        Returns
        -------
        list[list[PredictionResult]] — [rf_preds, xgb_preds]
        """
        if self._ensemble is None:
            raise RuntimeError("Models not loaded. Call load_models() first.")

        logger.info("[Pipeline] Running Random Forest …")
        rf_preds = self._ensemble.rf.predict(X)

        logger.info("[Pipeline] Running XGBoost …")
        xgb_preds = self._ensemble.xgb.predict(X)

        return [rf_preds, xgb_preds]

    def run_unsupervised(self, X: np.ndarray) -> list:
        """
        Run Isolation Forest and Dense Autoencoder predictions.

        Parameters
        ----------
        X : np.ndarray (n_samples, n_features)

        Returns
        -------
        list[list[PredictionResult]] — [iso_preds, ae_preds]
        """
        if self._ensemble is None:
            raise RuntimeError("Models not loaded. Call load_models() first.")

        logger.info("[Pipeline] Running Isolation Forest …")
        iso_preds = self._ensemble.iso.predict(X)

        logger.info("[Pipeline] Running Dense Autoencoder (LSTM Autoencoder) …")
        ae_preds = self._ensemble.ae.predict(X)

        return [iso_preds, ae_preds]

    # ── Step 5: Ensemble voting ───────────────────────────────────────────────

    def ensemble_prediction(
        self,
        X: np.ndarray,
        timestamps: Optional[list[str]] = None,
    ) -> list[EnsembleResult]:
        """
        Run the full ensemble on X and return one EnsembleResult per row.

        The EnsembleDetector internally runs all four models, performs
        weighted voting, calculates confidence and severity, and resolves
        the attack category. All logic is inherited from Phase 4.

        Parameters
        ----------
        X : np.ndarray (n_samples, n_features)
        timestamps : list[str], optional
            ISO-8601 timestamps for each row. Defaults to current UTC.

        Returns
        -------
        list[EnsembleResult]
        """
        if self._ensemble is None:
            raise RuntimeError("Models not loaded. Call load_models() first.")

        logger.info("[Pipeline] Running ensemble prediction …")
        results = self._ensemble.predict(X, timestamps=timestamps)
        logger.info(
            f"[Pipeline] Ensemble complete: "
            f"{sum(1 for r in results if r.final_verdict == 'ATTACK'):,} attacks detected ✓"
        )
        return results

    # ── Step 6: MITRE ATT&CK enrichment ─────────────────────────────────────

    def mitre_mapping(
        self,
        ensemble_results: list[EnsembleResult],
    ) -> list[EnrichedIncident]:
        """
        Enrich EnsembleResult objects with MITRE ATT&CK threat intelligence.

        Enrichment is applied post-detection (after the verdict is final) to
        avoid any possibility of the threat-intel mapping influencing model
        behaviour. This mirrors real SOC workflows: detect first, then look
        up threat context.

        For each result:
        - NORMAL verdict → EnrichedIncident with is_mapped=False, no technique
        - ATTACK verdict → technique ID, name, tactic, description from STIX

        Parameters
        ----------
        ensemble_results : list[EnsembleResult]

        Returns
        -------
        list[EnrichedIncident] — same length as input
        """
        if self._enricher is None:
            logger.info("[Pipeline] Initialising MITRE ATT&CK enricher …")
            self._enricher = AttackEnricher(preload=self._stix_preload)
            logger.info("[Pipeline] Enricher ready ✓")

        logger.info(
            f"[Pipeline] Running MITRE ATT&CK enrichment on "
            f"{len(ensemble_results):,} results …"
        )
        incidents = self._enricher.enrich_batch(ensemble_results)
        logger.info("[Pipeline] MITRE enrichment complete ✓")
        return incidents

    # ── Step 7: Severity is embedded in ensemble — exposed for clarity ────────

    def assign_severity(self, incidents: list[EnrichedIncident]) -> list[EnrichedIncident]:
        """
        No-op pass-through: severity is already computed by the EnsembleDetector.

        Included to make the pipeline flow explicit and to allow future override
        of the severity logic (e.g., domain-specific SOC playbooks) without
        touching the ensemble module.

        The severity formula is:
            CRITICAL : confidence ≥ 0.85 AND agreement ≥ 0.85
            HIGH     : confidence ≥ 0.70 AND agreement ≥ 0.70
            MEDIUM   : confidence ≥ 0.50 AND agreement ≥ 0.50
            LOW      : any other ATTACK

        Parameters
        ----------
        incidents : list[EnrichedIncident]

        Returns
        -------
        list[EnrichedIncident] — unchanged
        """
        logger.debug(
            "[Pipeline] Severity already assigned by EnsembleDetector.scoring.py "
            "(confidence × agreement matrix). No additional computation needed."
        )
        return incidents

    # ── Step 8: Report generation ─────────────────────────────────────────────

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
        All rows (ATTACK + NORMAL) contribute to the summary statistics.

        Parameters
        ----------
        incidents : list[EnrichedIncident]
            Full list of enriched incidents (both NORMAL and ATTACK).
        run_id : str
            Unique run identifier for the pipeline_run_id field.
        run_timestamp : str
            ISO-8601 or YYYYMMDD_HHMMSS timestamp for filename.
        input_source : str
            Human-readable source name (filename or 'DataFrame').
        row_indices : list[int], optional
            Original DataFrame row indices. Defaults to 0, 1, 2, …

        Returns
        -------
        reports : list[dict]  — one report dict per attack row
        summary : dict        — session-level summary
        """
        if row_indices is None:
            row_indices = list(range(len(incidents)))

        logger.info("[Pipeline] Generating incident reports …")
        reports: list[dict] = []
        for idx, incident in zip(row_indices, incidents):
            if incident.verdict == "ATTACK":
                report = build_incident_report(incident, idx, run_id)
                reports.append(report)

        logger.info(
            f"[Pipeline] {len(reports)} attack report(s) generated "
            f"(out of {len(incidents):,} total records)"
        )

        # Save incident reports
        save_incident_reports(
            reports,
            output_dir=self._output_dir,
            run_timestamp=run_timestamp,
        )

        # Build and save summary
        summary = generate_summary(
            all_incidents=incidents,
            run_id=run_id,
            run_timestamp=run_timestamp,
            input_source=input_source,
            output_dir=self._output_dir,
        )

        logger.info("[Pipeline] Reports saved ✓")
        return reports, summary

    # ── Main orchestrator ────────────────────────────────────────────────────

    def run(
        self,
        source: Union[str, Path, pd.DataFrame],
        max_rows: Optional[int] = None,
    ) -> dict:
        """
        Execute the full end-to-end detection pipeline.

        Execution flow
        --------------
        1. Load preprocessor (DataCleaner + feature_list.txt)
        2. Load all models (RF, XGB, IF, AE → EnsembleDetector)
        3. Load input data (CSV / Parquet / DataFrame)
        4. Preprocess input (DataCleaner.transform)
        5. Run ensemble prediction (all four models + voting + confidence)
        6. MITRE ATT&CK enrichment (AttackEnricher)
        7. Severity assignment (already embedded — pass-through)
        8. Generate incident reports and summary
        9. Return aggregated result dict

        Parameters
        ----------
        source : str | Path | pd.DataFrame
            Input network traffic data.
        max_rows : int, optional
            If set, only process the first N rows. Useful for quick tests.

        Returns
        -------
        dict with keys:
            "run_id"       : str  — unique run UUID
            "run_timestamp": str  — YYYYMMDD_HHMMSS
            "reports"      : list[dict] — one per attack row
            "summary"      : dict — session-level statistics
            "incidents"    : list[EnrichedIncident] — raw enriched objects
        """
        run_id        = str(uuid.uuid4())
        run_ts_iso    = now_utc_iso()
        run_ts_file   = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")

        logger.info("=" * 66)
        logger.info(f"[Pipeline] Phase 6 — ThreatDetectionPipeline starting")
        logger.info(f"[Pipeline] Run ID: {run_id}")
        logger.info(f"[Pipeline] Timestamp: {run_ts_iso}")
        logger.info("=" * 66)

        # ── 1. Preprocessor ───────────────────────────────────────────────────
        self.load_preprocessor()

        # ── 2. Models ─────────────────────────────────────────────────────────
        self.load_models()

        # ── 3. Load input data ────────────────────────────────────────────────
        logger.info("[Pipeline] Loading input data …")
        df = load_input_data(source)
        input_source = (
            str(source) if not isinstance(source, pd.DataFrame) else "DataFrame"
        )
        if max_rows is not None:
            df = df.head(max_rows)
            logger.info(f"[Pipeline] Limiting to first {max_rows:,} rows (max_rows)")

        # ── 4. Extract feature matrix (data already scaled from Phase 1) ────────
        _, X = self.preprocess(df)
        row_indices = list(df.index)

        # ── 5. Ensemble prediction ────────────────────────────────────────────
        ensemble_results = self.ensemble_prediction(X)

        # ── 6. MITRE enrichment ────────────────────────────────────────────────
        incidents = self.mitre_mapping(ensemble_results)

        # ── 7. Severity (pass-through) ────────────────────────────────────────
        incidents = self.assign_severity(incidents)

        # ── 8. Reports ────────────────────────────────────────────────────────
        logger.info("[Pipeline] Generating reports …")
        reports, summary = self.generate_reports(
            incidents=incidents,
            run_id=run_id,
            run_timestamp=run_ts_file,
            input_source=input_source,
            row_indices=row_indices,
        )

        logger.info("=" * 66)
        logger.info("[Pipeline] Pipeline completed successfully ✓")
        logger.info(
            f"[Pipeline] Processed {summary['total_records']:,} records → "
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
