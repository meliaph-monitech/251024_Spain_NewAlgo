import io
import os
import zipfile
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(layout="wide")

# ============================================================
# CORE (MUST MATCH NOTEBOOK LOGIC)
# ============================================================

def resolve_colname(df: pd.DataFrame, idx: int) -> str:
    return df.columns[idx]

def segment_beads(df: pd.DataFrame, column: str, threshold: float):
    start_indices, end_indices = [], []
    signal = df[column].to_numpy()
    i = 0
    while i < len(signal):
        if signal[i] > threshold:
            start = i
            while i < len(signal) and signal[i] > threshold:
                i += 1
            end = i - 1
            start_indices.append(start)
            end_indices.append(end)
        else:
            i += 1
    return list(zip(start_indices, end_indices))

def aggregate_steps(y, step: int, method: str) -> np.ndarray:
    y = np.asarray(y, float)
    if len(y) == 0:
        return np.array([])
    if step <= 0:
        step = 1
    chunks = [y[i:i+step] for i in range(0, len(y), step)]
    if method == "median":
        return np.array([np.median(c) for c in chunks if len(c)])
    return np.array([np.mean(c) for c in chunks if len(c)])

def compute_ok_reference(buffer_steps: List[dict], signal_indices: List[int]) -> dict:
    """
    Returns: ref[(bead_no, sig_idx)] = {mu, sigma, min, denom, L}
    All arrays are length L.
    """
    ref = {}
    beads = sorted({b for fs in buffer_steps for b in fs})
    for bead in beads:
        for sig in signal_indices:
            vecs = [fs[bead][sig] for fs in buffer_steps if bead in fs and sig in fs[bead]]
            if len(vecs) < 2:
                continue

            L = min(len(v) for v in vecs)
            if L <= 0:
                continue

            M = np.vstack([v[:L] for v in vecs])
            mu = np.median(M, axis=0)
            sigma = M.std(axis=0, ddof=1)
            sigma[sigma < 1e-12] = 1e-12

            mn = M.min(axis=0)
            mx = M.max(axis=0)
            denom = mx - mn
            denom[denom < 1e-12] = 1e-12

            ref[(bead, sig)] = dict(mu=mu, sigma=sigma, min=mn, denom=denom, L=L)

    return ref

def safe_exceed_metrics(
    step_y: np.ndarray,
    ref: dict,
    norm_lower: float,
    norm_upper: float,
    z_lower: float,
    z_upper: float,
):
    """
    Prevent shape mismatch by truncating both signal and reference arrays to common length.
    Exactly matches the notebook logic.
    """
    L = int(ref["L"])
    L_eff = min(len(step_y), L)
    if L_eff <= 0:
        return "LOW", dict(Norm_Low_Exceed=np.nan, Norm_High_Exceed=np.nan, Z_Low_Exceed=np.nan, Z_High_Exceed=np.nan)

    y = np.asarray(step_y[:L_eff], float)
    ref_min = ref["min"][:L_eff]
    ref_denom = ref["denom"][:L_eff]
    ref_mu = ref["mu"][:L_eff]
    ref_sigma = ref["sigma"][:L_eff]

    norm = (y - ref_min) / ref_denom
    z = (y - ref_mu) / ref_sigma

    nl = np.min(norm[norm < norm_lower]) if np.any(norm < norm_lower) else np.nan
    nh = np.max(norm[norm > norm_upper]) if np.any(norm > norm_upper) else np.nan
    zl = np.min(z[z < -z_lower]) if np.any(z < -z_lower) else np.nan
    zh = np.max(z[z > z_upper]) if np.any(z > z_upper) else np.nan

    if not np.isnan(nl) or not np.isnan(zl):
        status = "LOW"
    elif not np.isnan(nh) or not np.isnan(zh):
        status = "HIGH"
    else:
        status = "OK"

    return status, dict(
        Norm_Low_Exceed=nl,
        Norm_High_Exceed=nh,
        Z_Low_Exceed=zl,
        Z_High_Exceed=zh
    )

# ============================================================
# STREAMLIT HELPERS
# ============================================================

@dataclass(frozen=True)
class ZipCsv:
    name: str
    df: pd.DataFrame

def read_zip_csvs(uploaded_zip) -> List[ZipCsv]:
    if uploaded_zip is None:
        return []
    out: List[ZipCsv] = []
    with zipfile.ZipFile(uploaded_zip, "r") as z:
        names = [n for n in z.namelist() if n.lower().endswith(".csv")]
        # sort by filename (for hhmmss prefix ordering)
        names = sorted(names, key=lambda x: os.path.basename(x).lower())
        for n in names:
            with z.open(n) as f:
                df = pd.read_csv(f)
            out.append(ZipCsv(name=os.path.basename(n), df=df))
    return out

def extract_steps_for_df(
    df: pd.DataFrame,
    segment_col_idx: int,
    segment_threshold: float,
    min_bead_length: int,
    signal_indices: List[int],
    step_interval: int,
    step_method: str,
) -> Dict[int, Dict[int, np.ndarray]]:
    seg_col_name = resolve_colname(df, segment_col_idx)
    ranges = segment_beads(df, seg_col_name, segment_threshold)
    ranges = [(a, b) for a, b in ranges if (b - a + 1) >= min_bead_length]

    file_steps: Dict[int, Dict[int, np.ndarray]] = {}
    for bead_no, (a, b) in enumerate(ranges, start=1):
        bead_df = df.iloc[a:b+1]
        file_steps[bead_no] = {}
        for sig in signal_indices:
            col_name = resolve_colname(bead_df, sig)
            file_steps[bead_no][sig] = aggregate_steps(bead_df[col_name], step_interval, step_method)
    return file_steps

def short_label(csv_name: str) -> str:
    base = os.path.splitext(os.path.basename(csv_name))[0]
    return base[:6] if len(base) > 6 else base

# ============================================================
# UI
# ============================================================

st.title("OKREF Verifier (Streamlit) — Notebook-Consistent Metrics (Option A)")

st.sidebar.header("1) Upload OKREF ZIP (Reference)")
ok_zip = st.sidebar.file_uploader("OKREF ZIP (must contain the exact OKREF_RANGE files)", type=["zip"], key="ok_zip")

st.sidebar.header("2) Upload TEST ZIP (Files to inspect)")
test_zip = st.sidebar.file_uploader("TEST ZIP", type=["zip"], key="test_zip")

# Load once to infer columns
ok_csvs = read_zip_csvs(ok_zip) if ok_zip else []
test_csvs = read_zip_csvs(test_zip) if test_zip else []

if not ok_csvs:
    st.info("Upload OKREF ZIP to begin.")
    st.stop()

sample_df = ok_csvs[0].df
n_cols = sample_df.shape[1]

st.sidebar.header("3) Parameters (INDEX ONLY)")
segment_col_idx = st.sidebar.number_input("SEGMENT_COLUMN_INDEX", min_value=0, max_value=max(0, n_cols-1), value=0, step=1)
segment_threshold = st.sidebar.number_input("SEGMENT_THRESHOLD", value=0.2, step=0.01, format="%.4f")
min_bead_length = st.sidebar.number_input("MIN_BEAD_LENGTH", min_value=1, value=5000, step=100)

# Signal indices selection as indices only (but display names)
default_signal = [0, 1] if n_cols >= 2 else [0]
signal_choices = list(range(n_cols))
signal_indices = st.sidebar.multiselect(
    "SIGNAL_COLUMN_INDICES (indices only)",
    options=signal_choices,
    default=[i for i in default_signal if i in signal_choices]
)

step_interval = st.sidebar.number_input("GLOBAL_STEP_INTERVAL", min_value=1, value=20, step=1)
step_method = st.sidebar.selectbox("STEP_AGG_METHOD", options=["median", "mean"], index=0)

st.sidebar.header("4) Thresholds (same as notebook)")
norm_lower = st.sidebar.number_input("GLOBAL_NORM_LOWER", value=-4.0, step=0.1)
norm_upper = st.sidebar.number_input("GLOBAL_NORM_UPPER", value=10.0, step=0.1)
z_lower = st.sidebar.number_input("GLOBAL_Z_LOWER", value=6.0, step=0.1)
z_upper = st.sidebar.number_input("GLOBAL_Z_UPPER", value=20.0, step=0.1)

if len(signal_indices) == 0:
    st.error("Select at least one signal index.")
    st.stop()

# Display column mapping (index -> name)
with st.expander("Column index mapping (for your verification)", expanded=True):
    mapping_df = pd.DataFrame({"Index": list(range(n_cols)), "ColumnName": list(sample_df.columns)})
    st.dataframe(mapping_df, use_container_width=True)

st.markdown("### OKREF ZIP Summary")
st.write(f"- OKREF CSV files loaded: **{len(ok_csvs)}**")
st.write(f"- First file: `{ok_csvs[0].name}`")
st.write(f"- Last file : `{ok_csvs[-1].name}`")

build_btn = st.button("Build OKREF (compute_ok_reference)")

if "ok_ref" not in st.session_state:
    st.session_state.ok_ref = None
    st.session_state.ok_steps_by_file = None

if build_btn:
    with st.spinner("Extracting steps from OKREF files and building OK reference..."):
        ok_steps_by_file = {}
        buffer_steps = []
        for zc in ok_csvs:
            fs = extract_steps_for_df(
                zc.df,
                segment_col_idx=segment_col_idx,
                segment_threshold=segment_threshold,
                min_bead_length=int(min_bead_length),
                signal_indices=signal_indices,
                step_interval=int(step_interval),
                step_method=step_method
            )
            ok_steps_by_file[zc.name] = fs
            buffer_steps.append(fs)

        ok_ref = compute_ok_reference(buffer_steps, signal_indices=signal_indices)

    st.session_state.ok_ref = ok_ref
    st.session_state.ok_steps_by_file = ok_steps_by_file
    st.success(f"OKREF built. Reference keys: {len(ok_ref)} (bead, signal) pairs")

if st.session_state.ok_ref is None:
    st.info("Click **Build OKREF** to compute reference statistics.")
    st.stop()

ok_ref = st.session_state.ok_ref
ok_steps_by_file = st.session_state.ok_steps_by_file

# ------------------------------------------------------------
# TEST inspection section
# ------------------------------------------------------------
st.markdown("### TEST Inspection (Notebook-consistent metrics)")

if not test_csvs:
    st.info("Upload TEST ZIP to inspect files.")
    st.stop()

test_names = [c.name for c in test_csvs]
sel_test = st.selectbox("Select a TEST CSV to inspect", options=test_names)

test_df = next(c.df for c in test_csvs if c.name == sel_test)

inspect_btn = st.button("Inspect selected TEST CSV")

if inspect_btn:
    with st.spinner("Inspecting TEST file against OKREF..."):
        test_steps = extract_steps_for_df(
            test_df,
            segment_col_idx=segment_col_idx,
            segment_threshold=segment_threshold,
            min_bead_length=int(min_bead_length),
            signal_indices=signal_indices,
            step_interval=int(step_interval),
            step_method=step_method
        )

        rows = []
        for bead_no in sorted(test_steps.keys()):
            for sig in signal_indices:
                # if missing in test, skip
                if sig not in test_steps[bead_no]:
                    continue

                key = (bead_no, sig)
                if key not in ok_ref:
                    status = "MISSING_REF"
                    metrics = dict(
                        Norm_Low_Exceed=np.nan,
                        Norm_High_Exceed=np.nan,
                        Z_Low_Exceed=np.nan,
                        Z_High_Exceed=np.nan
                    )
                else:
                    status, metrics = safe_exceed_metrics(
                        test_steps[bead_no][sig],
                        ok_ref[key],
                        norm_lower=norm_lower,
                        norm_upper=norm_upper,
                        z_lower=z_lower,
                        z_upper=z_upper
                    )

                # REQUIRED table columns only (your format)
                col_name = resolve_colname(test_df, sig)
                rows.append(dict(
                    CSV_File=sel_test,
                    Bead=bead_no,
                    SignalColumn=f"{sig} ({col_name})",   # index only input; show real name in display
                    Status=status,
                    Norm_Low_Exceed=metrics["Norm_Low_Exceed"],
                    Norm_High_Exceed=metrics["Norm_High_Exceed"],
                    Z_Low_Exceed=metrics["Z_Low_Exceed"],
                    Z_High_Exceed=metrics["Z_High_Exceed"],
                ))

        result_df = pd.DataFrame(rows)

    st.subheader("Inspection Result Table (Exact metrics logic as notebook)")
    st.dataframe(result_df, use_container_width=True)

    # Filtered NOK-only view
    nok_df = result_df[result_df["Status"].isin(["LOW", "HIGH", "MISSING_REF"])].copy()
    st.subheader("NOK Rows Only")
    st.dataframe(nok_df, use_container_width=True)

    # Quick plots for user verification: pick bead + signal
    st.subheader("Quick Verification Plot (Raw overlay + Step Z/Norm)")
    bead_options = sorted(test_steps.keys())
    if len(bead_options) == 0:
        st.warning("No beads found in TEST after segmentation/min-length filter.")
        st.stop()

    sel_bead = st.selectbox("Bead", options=bead_options, index=0)
    sel_sig = st.selectbox("Signal index", options=signal_indices, index=0)

    # Build raw overlay for that bead/sig:
    # - OKREF raw lines: use min raw length across OK for bead
    # - TEST raw line: truncated to same min length
    seg_col_name = resolve_colname(test_df, segment_col_idx)
    test_ranges = segment_beads(test_df, seg_col_name, segment_threshold)
    test_ranges = [(a, b) for a, b in test_ranges if (b - a + 1) >= min_bead_length]

    if len(test_ranges) < sel_bead:
        st.warning("Selected bead not present in TEST after filtering.")
        st.stop()

    ta, tb = test_ranges[sel_bead - 1]
    test_bead_df = test_df.iloc[ta:tb+1]
    test_col = resolve_colname(test_bead_df, sel_sig)
    test_raw = np.asarray(test_bead_df[test_col], float)

    # OK raw lines for same bead/sig
    ok_raw_lines = []
    ok_min_len = None
    for zc in ok_csvs:
        df_ok = zc.df
        seg_ok = resolve_colname(df_ok, segment_col_idx)
        ok_ranges = segment_beads(df_ok, seg_ok, segment_threshold)
        ok_ranges = [(a, b) for a, b in ok_ranges if (b - a + 1) >= min_bead_length]
        if len(ok_ranges) < sel_bead:
            continue
        a, b = ok_ranges[sel_bead - 1]
        bead_ok = df_ok.iloc[a:b+1]
        col_ok = resolve_colname(bead_ok, sel_sig)
        y_ok = np.asarray(bead_ok[col_ok], float)
        ok_raw_lines.append((short_label(zc.name), y_ok))
        if ok_min_len is None:
            ok_min_len = len(y_ok)
        else:
            ok_min_len = min(ok_min_len, len(y_ok))

    if not ok_raw_lines or ok_min_len is None or ok_min_len <= 0:
        st.warning("No OK raw lines available for this bead/signal. Check segmentation/min length.")
        st.stop()

    Lraw = min(ok_min_len, len(test_raw))
    test_raw = test_raw[:Lraw]
    ok_raw_lines = [(lbl, y[:Lraw]) for (lbl, y) in ok_raw_lines]

    # Step arrays (exact same processing as notebook)
    test_step = aggregate_steps(test_raw, int(step_interval), step_method)
    key = (sel_bead, sel_sig)

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=False,
        vertical_spacing=0.18,
        subplot_titles=("Raw overlay (OKREF gray + TEST color)", "Per-step Norm & Z (aligned by safe_exceed_metrics truncation)")
    )

    # Raw overlay
    for lbl, y in ok_raw_lines:
        fig.add_trace(
            go.Scatter(x=np.arange(len(y)), y=y, mode="lines", name=f"{lbl} OK", opacity=0.20, line=dict(width=1)),
            row=1, col=1
        )
    fig.add_trace(
        go.Scatter(x=np.arange(len(test_raw)), y=test_raw, mode="lines", name=f"{short_label(sel_test)} TEST", opacity=0.90, line=dict(width=2)),
        row=1, col=1
    )

    if key in ok_ref:
        ref = ok_ref[key]
        L_eff = min(len(test_step), int(ref["L"]))
        if L_eff > 0:
            y = test_step[:L_eff]
            norm = (y - ref["min"][:L_eff]) / ref["denom"][:L_eff]
            z = (y - ref["mu"][:L_eff]) / ref["sigma"][:L_eff]

            fig.add_trace(
                go.Scatter(x=np.arange(L_eff), y=norm, mode="lines", name="TEST Norm", line=dict(width=2)),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(x=np.arange(L_eff), y=z, mode="lines", name="TEST Z", line=dict(width=2)),
                row=2, col=1
            )

            # Threshold lines
            fig.add_hline(y=norm_lower, row=2, col=1, line=dict(dash="dash"))
            fig.add_hline(y=norm_upper, row=2, col=1, line=dict(dash="dash"))
            fig.add_hline(y=-z_lower, row=2, col=1, line=dict(dash="dot"))
            fig.add_hline(y=z_upper, row=2, col=1, line=dict(dash="dot"))
        else:
            st.warning("Step vector length is zero after aggregation.")
    else:
        st.warning("Reference missing for selected bead/signal (MISSING_REF).")

    fig.update_layout(height=700, legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Notes")
    st.write(
        "- This app uses the same reference builder and metric logic as your notebook "
        "(compute_ok_reference + safe_exceed_metrics, including length truncation).\n"
        "- Inputs are index-only; displays show real column names.\n"
        "- If you still see mismatch after using the exact same OKREF_RANGE files, "
        "the cause is upstream (file content differences, ordering, or segmentation inputs)."
    )
