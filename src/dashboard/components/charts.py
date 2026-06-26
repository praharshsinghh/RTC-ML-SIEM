"""
src/dashboard/components/charts.py — Plotly Chart Library
===========================================================
All charts return go.Figure objects with a consistent dark SOC theme.
Pages call these functions and render with st.plotly_chart(fig, use_container_width=True).

No Streamlit calls here — charts are pure Plotly so they are independently testable.
"""
from __future__ import annotations

from typing import Optional
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Shared palette ────────────────────────────────────────────────────────────
BG       = "#0a0e1a"
CARD_BG  = "#0d1b2a"
BORDER   = "#1e3a5f"
ACCENT   = "#00d4ff"
TEXT     = "#e0e0e0"
MUTED    = "#5a7a9a"

SEV_COLORS = {
    "CRITICAL": "#ff2244",
    "HIGH":     "#ff6b00",
    "MEDIUM":   "#ffd700",
    "LOW":      "#00cc66",
    "N/A":      "#5a7a9a",
}

CAT_PALETTE = [
    "#00d4ff", "#ff6b00", "#ff2244", "#ffd700",
    "#00cc66", "#bf5fff", "#ff99bb", "#44ddff",
    "#ff9944", "#aaffcc",
]

_DARK_TEMPLATE = dict(
    layout=go.Layout(
        paper_bgcolor=CARD_BG,
        plot_bgcolor=CARD_BG,
        font=dict(color=TEXT, family="Inter, sans-serif", size=12),
        xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER),
        yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER),
        margin=dict(l=16, r=16, t=40, b=16),
        legend=dict(bgcolor=CARD_BG, bordercolor=BORDER, borderwidth=1),
    )
)


def _apply(fig: go.Figure) -> go.Figure:
    fig.update_layout(**_DARK_TEMPLATE["layout"].to_plotly_json())
    return fig


# ── 1. Attack vs Normal donut ─────────────────────────────────────────────────

def attack_vs_normal_donut(summary: dict) -> go.Figure:
    labels = ["Normal", "Attack"]
    values = [summary.get("normal_count", 0), summary.get("attack_count", 0)]
    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.62,
        marker=dict(colors=[ACCENT, "#ff2244"],
                    line=dict(color=BG, width=2)),
        textinfo="label+percent",
        hovertemplate="%{label}: %{value:,}<extra></extra>",
    ))
    total = sum(values)
    fig.add_annotation(text=f"<b>{total:,}</b><br>records",
                       x=0.5, y=0.5, showarrow=False,
                       font=dict(size=16, color=TEXT))
    fig.update_layout(title="Traffic Distribution", showlegend=True)
    return _apply(fig)


# ── 2. Severity horizontal bar ───────────────────────────────────────────────

def severity_bar(summary: dict) -> go.Figure:
    sev = summary.get("severity_distribution", {})
    order = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    labels = [s for s in order if s in sev]
    values = [sev[s] for s in labels]
    colors = [SEV_COLORS[s] for s in labels]
    fig = go.Figure(go.Bar(
        x=values, y=labels,
        orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=values, textposition="outside",
        textfont=dict(color=TEXT),
        hovertemplate="%{y}: %{x} incidents<extra></extra>",
    ))
    fig.update_layout(title="Severity Distribution", xaxis_title="Count",
                      yaxis=dict(autorange="reversed"))
    return _apply(fig)


# ── 3. Attack category treemap ────────────────────────────────────────────────

def category_treemap(summary: dict) -> go.Figure:
    cat_dist = summary.get("attack_category_distribution", {})
    if not cat_dist:
        return _empty("No attack categories to display")
    labels = list(cat_dist.keys())
    values = list(cat_dist.values())
    fig = go.Figure(go.Treemap(
        labels=labels,
        parents=[""] * len(labels),
        values=values,
        marker=dict(colors=CAT_PALETTE[:len(labels)],
                    line=dict(width=2, color=BG)),
        textinfo="label+value+percent root",
        hovertemplate="%{label}: %{value}<extra></extra>",
    ))
    fig.update_layout(title="Attack Category Distribution")
    return _apply(fig)


# ── 4. Confidence histogram ───────────────────────────────────────────────────

def confidence_histogram(df: pd.DataFrame, col: str = "Confidence") -> go.Figure:
    if df.empty or col not in df.columns:
        return _empty("No confidence data")
    fig = go.Figure(go.Histogram(
        x=df[col], nbinsx=30,
        marker=dict(color=ACCENT, opacity=0.8, line=dict(width=0)),
        hovertemplate="Confidence %{x:.2f}: %{y} rows<extra></extra>",
    ))
    fig.update_layout(title="Confidence Distribution",
                      xaxis_title="Confidence", yaxis_title="Count")
    return _apply(fig)


# ── 5. Agreement distribution ─────────────────────────────────────────────────

def agreement_histogram(df: pd.DataFrame, col: str = "Agreement") -> go.Figure:
    if df.empty or col not in df.columns:
        return _empty("No agreement data")
    # Agreement is a string like '3/4'; convert to fraction
    def _frac(s):
        try:
            a, b = str(s).split("/")
            return int(a) / int(b)
        except Exception:
            return None
    vals = df[col].apply(_frac).dropna()
    fig = go.Figure(go.Histogram(
        x=vals, nbinsx=10,
        marker=dict(color="#bf5fff", opacity=0.8, line=dict(width=0)),
        hovertemplate="Agreement %{x:.2f}: %{y} rows<extra></extra>",
    ))
    fig.update_layout(title="Model Agreement Distribution",
                      xaxis_title="Agreement (fraction)", yaxis_title="Count",
                      xaxis=dict(range=[0, 1]))
    return _apply(fig)


# ── 6. Confidence by severity box plot ───────────────────────────────────────

def confidence_by_severity(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return _empty("No data")
    order = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    fig = go.Figure()
    for sev in order:
        sub = df[df["Severity"] == sev]["Confidence"] if "Severity" in df.columns else pd.Series(dtype=float)
        if sub.empty:
            continue
        fig.add_trace(go.Box(
            y=sub, name=sev,
            marker_color=SEV_COLORS.get(sev, MUTED),
            line_color=SEV_COLORS.get(sev, MUTED),
            boxmean=True,
        ))
    fig.update_layout(title="Confidence by Severity",
                      yaxis_title="Confidence", showlegend=False)
    return _apply(fig)


# ── 7. MITRE tactic bar ───────────────────────────────────────────────────────

def mitre_tactic_bar(df: pd.DataFrame) -> go.Figure:
    if df.empty or "Tactic" not in df.columns:
        return _empty("No MITRE data")
    counts = df["Tactic"].value_counts().sort_values()
    fig = go.Figure(go.Bar(
        x=counts.values, y=counts.index,
        orientation="h",
        marker=dict(color=ACCENT, opacity=0.85),
        hovertemplate="%{y}: %{x}<extra></extra>",
        text=counts.values, textposition="outside",
        textfont=dict(color=TEXT),
    ))
    fig.update_layout(title="ATT&CK Tactic Distribution",
                      xaxis_title="Incidents")
    return _apply(fig)


# ── 8. MITRE technique bar (top N) ────────────────────────────────────────────

def mitre_technique_bar(df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    if df.empty or "Technique ID" not in df.columns:
        return _empty("No technique data")
    counts = df["Technique ID"].value_counts().head(top_n).sort_values()
    fig = go.Figure(go.Bar(
        x=counts.values, y=counts.index,
        orientation="h",
        marker=dict(color="#bf5fff", opacity=0.85),
        hovertemplate="%{y}: %{x} incidents<extra></extra>",
        text=counts.values, textposition="outside",
        textfont=dict(color=TEXT),
    ))
    fig.update_layout(title=f"Top {top_n} ATT&CK Techniques",
                      xaxis_title="Incidents", height=max(300, top_n * 28))
    return _apply(fig)


# ── 9. Per-model vote comparison ──────────────────────────────────────────────

def model_vote_bar(df: pd.DataFrame) -> go.Figure:
    """Grouped bar showing ATTACK/NORMAL counts per model."""
    if df.empty:
        return _empty("No data")
    model_cols = {"RF": "Random Forest", "XGB": "XGBoost",
                  "IForest": "Isolation Forest", "AE": "Autoencoder"}
    attack_counts, normal_counts, names = [], [], []
    for col, label in model_cols.items():
        if col not in df.columns:
            continue
        names.append(label)
        attack_counts.append(int((df[col] == "ATTACK").sum()))
        normal_counts.append(int((df[col] == "NORMAL").sum()))
    fig = go.Figure([
        go.Bar(name="Attack", x=names, y=attack_counts,
               marker_color="#ff2244", hovertemplate="%{x}<br>Attack: %{y}<extra></extra>"),
        go.Bar(name="Normal", x=names, y=normal_counts,
               marker_color=ACCENT, hovertemplate="%{x}<br>Normal: %{y}<extra></extra>"),
    ])
    fig.update_layout(barmode="group", title="Per-Model Vote Breakdown",
                      yaxis_title="Incident Count")
    return _apply(fig)


# ── 10. Severity pie ──────────────────────────────────────────────────────────

def severity_pie(summary: dict) -> go.Figure:
    sev = summary.get("severity_distribution", {})
    order = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    labels = [s for s in order if sev.get(s, 0) > 0]
    values = [sev[s] for s in labels]
    colors = [SEV_COLORS[s] for s in labels]
    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        marker=dict(colors=colors, line=dict(color=BG, width=2)),
        textinfo="label+percent",
        hovertemplate="%{label}: %{value}<extra></extra>",
    ))
    fig.update_layout(title="Severity Breakdown")
    return _apply(fig)


# ── Utility ───────────────────────────────────────────────────────────────────

def _empty(msg: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, x=0.5, y=0.5, showarrow=False,
                       font=dict(size=14, color=MUTED), xref="paper", yref="paper")
    fig.update_layout(title=msg)
    return _apply(fig)
