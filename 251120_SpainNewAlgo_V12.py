import os
import io
import zipfile

import numpy as np
import pandas as pd

import streamlit as st
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from scipy.signal import savgol_filter, butter, filtfilt

# For clustering
from sklearn.cluster import KMeans

# ============================================================
# Basic page config
# ============================================================
st.set_page_config(
    page_title="Spain New Algo V12 - Weld NOK Detection & Clustering",
    layout="wide"
)

st.title("Spain New Algo V12 – Weld NOK Detection & Clustering")
st.markdown(
    """
    This app:
    1. Segments weld beads from OK and TEST CSV files.  
    2. Uses OK beads as reference to compute per-step 0–1 normalization & Z-scores.  
    3. Flags TEST beads as LOW / HIGH / OK based on global thresholds.  
    4. Summarizes suspected NOK beads.  
    5. **(NEW in V12)** Clusters NOK features and visualizes them in upper/lower exceed scatter plots.
    """
)

# ============================================================
# Session State Init
# ============================================================
if "segmented_ok" not in st.session_state:
    st.session_state.segmented_ok = {}
if "segmented_test" not in st.session_state:
    st.session_state.segmented_test = {}
if "seg_col" not in st.session_state:
    st.session_state.seg_col = None
if "seg_thresh" not in st.session_state:
    st.session_state.seg_thresh = None

# ============================================================
# Helper: Short label from file name
# ============================================================
def short_label(path: str) -> str:
    base = os.path.basename(path)
    name, _ = os.path.splitext(base)
    return name

# ============================================================
# Helper: Bead segmentation
# ============================================================
def segment_beads(df: pd.DataFrame, column: str, threshold: float):
    """
    Simple threshold-based segmentation:
    A bead is a contiguous block where df[column] > threshold.
    Returns list of (start_idx, end_idx).
    """
    if column not in df.columns:
        return []

    signal = df[column].to_numpy()
    ranges = []
    in_bead = False
    start_idx = None

    for i, val in enumerate(signal):
        if not in_bead and val > threshold:
            in_bead = True
            start_idx = i
        elif in_bead and val <= threshold:
            end_idx = i - 1
            if end_idx >= start_idx:
                ranges.append((start_idx, end_idx))
            in_bead = False
            start_idx = None

    if in_bead and start_idx is not None:
        ranges.append((start_idx, len(signal) - 1))

    return ranges

# ============================================================
# Helper: Aggregate for step (window)
# ============================================================
def aggregate_for_step(x: np.ndarray, y: np.ndarray, interval: int):
    """
    Aggregates y over non-overlapping windows of length = interval.
    Returns (agg_x, agg_y) where agg_y[k] is mean of y in that window.
    """
    if interval <= 0:
        raise ValueError("interval must be > 0")

    n = len(y)
    if n == 0:
        return np.array([]), np.array([])

    num_steps = int(np.ceil(n / interval))
    agg_x = []
    agg_y = []

    for k in range(num_steps):
        start = k * interval
        end = min((k + 1) * interval, n)
        if start >= end:
            continue
        window_vals = y[start:end]
        agg_x.append(x[start])
        agg_y.append(window_vals.mean())

    return np.asarray(agg_x), np.asarray(agg_y)

# ============================================================
# Helper: Transform Signals
# ============================================================
def compute_transformed_signals(observations, mode, **params):
    """
    observations: list of dicts with keys: csv, data (pd.Series or 1D array)
    mode: 'raw', 'savgol', 'lowpass', 'poly'
    """
    transformed_obs = []
    for obs in observations:
        y = np.asarray(obs["data"]).astype(float)
        if mode == "raw":
            transformed = y
        elif mode == "savgol":
            window = int(params.get("window", 51))
            poly = int(params.get("poly", 2))
            # Ensure window is valid
            if window >= len(y):
                window = len(y) - 1 if len(y) % 2 == 0 else len(y)
            if window < 3:
                transformed = y
            else:
                if window % 2 == 0:
                    window -= 1
                transformed = savgol_filter(y, window_length=window, polyorder=poly)
        elif mode == "lowpass":
            cutoff = float(params.get("cutoff", 0.1))
            order = int(params.get("order", 2))
            b, a = butter(order, cutoff, btype='low', analog=False)
            # filtfilt needs enough points
            if len(y) > 3 * max(len(a), len(b)):
                transformed = filtfilt(b, a, y)
            else:
                transformed = y
        elif mode == "poly":
            deg = int(params.get("deg", 25))
            x = np.arange(len(y))
            max_deg = max(1, len(y) - 1)
            deg = min(deg, max_deg)
            try:
                coeffs = np.polyfit(x, y, deg)
                transformed = np.polyval(coeffs, x)
            except np.linalg.LinAlgError:
                transformed = y
        else:
            transformed = y

        transformed_obs.append({
            **obs,
            "transformed": transformed
        })
    return transformed_obs

# ============================================================
# Helper: Compute per-step normalization & flags + metrics
# ============================================================
def compute_step_normalization_and_flags(
    ref_obs,
    test_obs,
    step_interval,
    norm_lower,
    norm_upper,
    z_lower,
    z_upper,
    title_suffix
):
    """
    ref_obs: list of dicts (OK reference), with 'transformed'
    test_obs: list of dicts (TEST), with 'transformed'
    Returns: (fig, status_map, metrics_map)
        status_map: {csv_name: 'low' | 'high' | 'ok'}
        metrics_map: {csv_name: dict with exceed values}
    """
    if len(ref_obs) == 0:
        st.warning("No OK reference signals available for normalization.")
        return None, {}, {}

    # --- Build OK step arrays ---
    ok_step_arrays = []
    ok_step_meta = []
    for obs in ref_obs:
        y = np.asarray(obs["transformed"])
        x = np.arange(len(y))
        _, step_y = aggregate_for_step(x, y, step_interval)
        ok_step_arrays.append(step_y)
        ok_step_meta.append({
            "csv": obs["csv"],
            "step_y": step_y
        })

    if len(ok_step_arrays) == 0:
        st.warning("No valid OK step data for normalization.")
        return None, {}, {}

    min_steps = min(arr.shape[0] for arr in ok_step_arrays)
    if min_steps == 0:
        st.warning("OK step data are empty after aggregation.")
        return None, {}, {}

    ok_matrix = np.vstack([arr[:min_steps] for arr in ok_step_arrays])

    # Reference stats from OK
    mu = ok_matrix.mean(axis=0)
    sigma = ok_matrix.std(axis=0, ddof=1)
    sigma[sigma < 1e-12] = 1e-12
    min_ok = ok_matrix.min(axis=0)
    max_ok = ok_matrix.max(axis=0)
    denom = max_ok - min_ok
    denom[denom < 1e-12] = 1e-12

    step_indices = np.arange(min_steps)

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("0–1 Normalization (OK-based)", "Z-score per Step")
    )

    status_map = {}
    metrics_map = {}

    # --- Plot OK reference signals in gray ---
    for meta in ok_step_meta:
        step_y_ok = meta["step_y"][:min_steps]
        norm_ok = (step_y_ok - min_ok) / denom
        z_ok = (step_y_ok - mu) / sigma

        fig.add_trace(
            go.Scatter(
                x=step_indices,
                y=norm_ok,
                mode="lines",
                name=f"{short_label(meta['csv'])} (OK ref)",
                line=dict(color="#aaaaaa", width=1),
                legendgroup="OK_REF",
                showlegend=True
            ),
            row=1,
            col=1
        )

        fig.add_trace(
            go.Scatter(
                x=step_indices,
                y=z_ok,
                mode="lines",
                line=dict(color="#aaaaaa", width=1),
                legendgroup="OK_REF",
                showlegend=False
            ),
            row=1,
            col=2
        )

    # --- Plot TEST signals and compute status + metrics ---
    for obs in test_obs:
        y = np.asarray(obs["transformed"])
        x = np.arange(len(y))
        _, step_y = aggregate_for_step(x, y, step_interval)
        step_y = step_y[:min_steps]

        norm_vals = (step_y - min_ok) / denom
        z_vals = (step_y - mu) / sigma

        # Masks for threshold violations
        mask_norm_low = norm_vals < norm_lower
        mask_norm_high = norm_vals > norm_upper
        mask_z_low = z_vals < -z_lower
        mask_z_high = z_vals > z_upper

        # Exceeding values (NaN if no violation)
        norm_low_exceed = norm_vals[mask_norm_low].min() if mask_norm_low.any() else np.nan
        norm_high_exceed = norm_vals[mask_norm_high].max() if mask_norm_high.any() else np.nan
        z_low_exceed = z_vals[mask_z_low].min() if mask_z_low.any() else np.nan
        z_high_exceed = z_vals[mask_z_high].max() if mask_z_high.any() else np.nan

        # Low deviation (drop) condition
        low_flag = mask_norm_low.any() or mask_z_low.any()
        # High deviation (too high) condition
        high_flag = mask_norm_high.any() or mask_z_high.any()

        if low_flag:
            status = "low"      # red
        elif high_flag:
            status = "high"     # orange
        else:
            status = "ok"       # green

        status_map[obs["csv"]] = status
        metrics_map[obs["csv"]] = {
            "Norm_Low_Exceed": norm_low_exceed,
            "Norm_High_Exceed": norm_high_exceed,
            "Z_Low_Exceed": z_low_exceed,
            "Z_High_Exceed": z_high_exceed,
        }

        if status == "low":
            color = "red"
            width = 2
        elif status == "high":
            color = "orange"
            width = 2
        else:
            color = "green"
            width = 1

        name = short_label(obs["csv"])

        # Left: 0–1 normalization (TEST)
        fig.add_trace(
            go.Scatter(
                x=step_indices,
                y=norm_vals,
                mode="lines",
                name=f"{name} (TEST)",
                line=dict(color=color, width=width),
                legendgroup=name,
                showlegend=True
            ),
            row=1,
            col=1
        )

        # Right: z-score (TEST)
        fig.add_trace(
            go.Scatter(
                x=step_indices,
                y=z_vals,
                mode="lines",
                line=dict(color=color, width=width),
                legendgroup=name,
                showlegend=False
            ),
            row=1,
            col=2
        )

    # Threshold lines
    fig.add_hline(
        y=norm_lower,
        line=dict(color="gray", dash="dash"),
        row=1,
        col=1
    )
    fig.add_hline(
        y=norm_upper,
        line=dict(color="gray", dash="dash"),
        row=1,
        col=1
    )
    fig.add_hline(
        y=-z_lower,
        line=dict(color="gray", dash="dash"),
        row=1,
        col=2
    )
    fig.add_hline(
        y=z_upper,
        line=dict(color="gray", dash="dash"),
        row=1,
        col=2
    )

    fig.update_layout(
        title=dict(
            text=f"Per-step Normalization {title_suffix}",
            font=dict(size=22)
        ),
        legend=dict(orientation="h")
    )

    return fig, status_map, metrics_map

# ============================================================
# Helper: Plot Top (Transformed TEST + OK in gray)
# ============================================================
def plot_top_signals(ref_transformed, test_transformed, status_map, title, y_label):
    fig = go.Figure()

    # OK reference signals in gray
    for obs in ref_transformed:
        y = obs["transformed"]
        x = np.arange(len(y))
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name=f"{short_label(obs['csv'])} (OK ref)",
                line=dict(color="#aaaaaa", width=1),
                legendgroup="OK_REF"
            )
        )

    # TEST signals, colored by status
    for obs in test_transformed:
        y = obs["transformed"]
        x = np.arange(len(y))
        csv_name = obs["csv"]
        status = status_map.get(csv_name, "ok")

        if status == "low":
            color = "red"
            width = 2
            label = "LOW"
        elif status == "high":
            color = "orange"
            width = 2
            label = "HIGH"
        else:
            color = "green"
            width = 1
            label = "OK-like"

        name = f"{short_label(csv_name)} (TEST, {label})"

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name=name,
                line=dict(color=color, width=width),
                legendgroup=short_label(csv_name)
            )
        )

    fig.update_layout(
        title_text=title,
        title_font=dict(size=22),
        xaxis_title="Index",
        yaxis_title=y_label,
        legend=dict(orientation="h")
    )
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# Helper: Build global summary (used in Summary tab + Clustering)
# ============================================================
def build_global_summary_for_signal(
    segmented_ok,
    segmented_test,
    bead_options,
    ok_files,
    test_files,
    signal_col,
    step_interval,
    norm_lower,
    norm_upper,
    z_lower,
    z_upper,
    window=51,
    poly=2
):
    """
    Build df_summary for a given signal column (like V11 global summary).
    """
    rows = []

    for bead in bead_options:
        # Build OK & TEST obs for this bead
        ref_obs_bead = []
        for fname in ok_files:
            beads = segmented_ok.get(fname, {})
            if bead in beads:
                bead_df = beads[bead]
                if signal_col in bead_df.columns:
                    data = bead_df[signal_col].reset_index(drop=True)
                    ref_obs_bead.append({"csv": fname, "data": data})

        test_obs_bead = []
        for fname in test_files:
            beads = segmented_test.get(fname, {})
            if bead in beads:
                bead_df = beads[bead]
                if signal_col in bead_df.columns:
                    data = bead_df[signal_col].reset_index(drop=True)
                    test_obs_bead.append({"csv": fname, "data": data})

        if not ref_obs_bead or not test_obs_bead:
            continue

        ref_transformed_bead = compute_transformed_signals(
            ref_obs_bead,
            mode="savgol",
            window=window,
            poly=poly
        )
        test_transformed_bead = compute_transformed_signals(
            test_obs_bead,
            mode="savgol",
            window=window,
            poly=poly
        )

        _, status_bead, metrics_bead = compute_step_normalization_and_flags(
            ref_transformed_bead,
            test_transformed_bead,
            step_interval=step_interval,
            norm_lower=norm_lower,
            norm_upper=norm_upper,
            z_lower=z_lower,
            z_upper=z_upper,
            title_suffix=f"(Bead #{bead}, Smoothed, Signal={signal_col})"
        )

        for csv_name, status in status_bead.items():
            if status == "ok":
                continue
            m = metrics_bead.get(csv_name, {})
            rows.append({
                "CSV_File": csv_name,              # full name
                "Bead": bead,
                "Status": "LOW" if status == "low" else "HIGH",
                "Norm_Low_Exceed": m.get("Norm_Low_Exceed", np.nan),
                "Norm_High_Exceed": m.get("Norm_High_Exceed", np.nan),
                "Z_Low_Exceed": m.get("Z_Low_Exceed", np.nan),
                "Z_High_Exceed": m.get("Z_High_Exceed", np.nan),
            })

    if not rows:
        return None

    df_summary = pd.DataFrame(rows)
    df_summary = df_summary.sort_values(["CSV_File", "Bead"]).reset_index(drop=True)
    # Mark which rows are true NOKs based on file name containing "NG"
    df_summary["Is_NG"] = df_summary["CSV_File"].str.contains("NG", case=False)

    return df_summary

# ============================================================
# STEP 1: Upload & Segment OK REFERENCE SET
# ============================================================
st.sidebar.header("Step 1: Upload OK Reference Set (All Beads Known-OK)")
uploaded_ok_zip = st.sidebar.file_uploader(
    "Upload ZIP of OK reference CSV files",
    type="zip",
    key="ok_zip"
)

if uploaded_ok_zip:
    with zipfile.ZipFile(uploaded_ok_zip, 'r') as zip_ref:
        csv_names_ok = [name for name in zip_ref.namelist() if name.endswith('.csv')]
        if not csv_names_ok:
            st.sidebar.error("No CSV files found in the OK ZIP.")
        else:
            first_ok_csv = csv_names_ok[0]
            with zip_ref.open(first_ok_csv) as f:
                sample_df_ok = pd.read_csv(f)
            ok_columns = sample_df_ok.columns.tolist()

            st.session_state.seg_col = st.sidebar.selectbox(
                "Column for Segmentation (OK set)",
                ok_columns,
                key="seg_col_ok"
            )
            st.session_state.seg_thresh = st.sidebar.number_input(
                "Segmentation Threshold (OK & TEST share this)",
                value=0.5
            )
            segment_ok_btn = st.sidebar.button("Segment OK Files")

    # Perform segmentation for OK set
    if 'segment_ok_btn' in locals() and segment_ok_btn:
        segmented_ok = {}
        with zipfile.ZipFile(uploaded_ok_zip, 'r') as zip_ref:
            for file_name in zip_ref.namelist():
                if file_name.endswith('.csv'):
                    with zip_ref.open(file_name) as f:
                        df = pd.read_csv(f)
                    bead_ranges = segment_beads(
                        df,
                        st.session_state.seg_col,
                        st.session_state.seg_thresh
                    )
                    bead_dict = {}
                    for idx, (start, end) in enumerate(bead_ranges, start=1):
                        bead_dict[idx] = df.iloc[start:end+1].reset_index(drop=True)
                    segmented_ok[os.path.basename(file_name)] = bead_dict
        st.session_state.segmented_ok = segmented_ok
        st.success("✅ OK reference beads segmented and stored.")

# ============================================================
# STEP 2: Upload & Segment TEST SET
# ============================================================
st.sidebar.header("Step 2: Upload TEST Set (Unknown OK / NOK)")
uploaded_test_zip = st.sidebar.file_uploader(
    "Upload ZIP of TEST CSV files",
    type="zip",
    key="test_zip"
)

if uploaded_test_zip:
    if st.session_state.seg_col is None or st.session_state.seg_thresh is None:
        st.sidebar.warning("Please segment the OK reference set first to define segmentation settings.")
    else:
        segment_test_btn = st.sidebar.button("Segment TEST Files")
        if 'segment_test_btn' in locals() and segment_test_btn:
            segmented_test = {}
            with zipfile.ZipFile(uploaded_test_zip, 'r') as zip_ref:
                for file_name in zip_ref.namelist():
                    if file_name.endswith('.csv'):
                        with zip_ref.open(file_name) as f:
                            df = pd.read_csv(f)
                        bead_ranges = segment_beads(
                            df,
                            st.session_state.seg_col,
                            st.session_state.seg_thresh
                        )
                        bead_dict = {}
                        for idx, (start, end) in enumerate(bead_ranges, start=1):
                            bead_dict[idx] = df.iloc[start:end+1].reset_index(drop=True)
                        segmented_test[os.path.basename(file_name)] = bead_dict
            st.session_state.segmented_test = segmented_test
            st.success("✅ TEST beads segmented and stored.")

# ============================================================
# STEP 3: Analysis (Requires both OK & TEST segmented)
# ============================================================
if st.session_state.segmented_ok and st.session_state.segmented_test:
    segmented_ok = st.session_state.segmented_ok
    segmented_test = st.session_state.segmented_test

    st.sidebar.header("Step 3: Global Thresholds & Step Interval")

    # Global thresholds for all tabs
    global_norm_lower = st.sidebar.number_input(
        "Global Lower Threshold for 0–1 Normalization (flag LOW if below)",
        min_value=-5.0,
        max_value=5.0,
        value=-0.7,
        step=0.05
    )
    global_norm_upper = st.sidebar.number_input(
        "Global Upper Threshold for 0–1 Normalization (flag HIGH if above)",
        min_value=-5.0,
        max_value=5.0,
        value=2.0,
        step=0.05
    )
    global_z_lower = st.sidebar.number_input(
        "Global Z-score Threshold (flag LOW if below -T)",
        min_value=0.5,
        max_value=10.0,
        value=3.0,
        step=0.5
    )
    global_z_upper = st.sidebar.number_input(
        "Global Z-score Threshold (flag HIGH if above +T)",
        min_value=0.5,
        max_value=10.0,
        value=3.25,
        step=0.25
    )

    # Global step interval
    global_step_interval = st.sidebar.slider(
        "Global Step Interval (points)",
        min_value=10,
        max_value=500,
        value=40,
        step=10
    )

    st.sidebar.header("Step 4: Bead & Signal for Analysis")

    ok_files = sorted(segmented_ok.keys())
    test_files = sorted(segmented_test.keys())

    # Bead options = intersection of beads present in OK and TEST
    bead_ok = set()
    for fname, beads in segmented_ok.items():
        bead_ok.update(beads.keys())
    bead_test = set()
    for fname, beads in segmented_test.items():
        bead_test.update(beads.keys())
    bead_options = sorted(bead_ok.intersection(bead_test))

    if not bead_options:
        st.warning("No common bead numbers found in both OK and TEST sets.")
    else:
        selected_bead = st.sidebar.selectbox("Select Bead Number", bead_options)

        # Example OK bead df to list signal columns
        example_bead_df = None
        for fname in ok_files:
            beads = segmented_ok[fname]
            if selected_bead in beads:
                example_bead_df = beads[selected_bead]
                break

        if example_bead_df is None:
            st.error("Selected bead not found in OK reference set.")
        else:
            signal_col = st.sidebar.selectbox(
                "Select Signal Column",
                example_bead_df.columns.tolist()
            )

            st.markdown(
                f"### Analysis for Signal **{signal_col}**"
            )
            st.markdown(
                f"- OK reference files: {len(ok_files)}  \n"
                f"- TEST files: {len(test_files)}"
            )

            # Tabs:
            tabs = st.tabs([
                "Summary",
                "Raw Signal",
                "Smoothed (Savitzky)",
                "Low-pass Filter",
                "Curve Fit",
                "Clustering (NIR / VIS)"
            ])

            # ---------- Build df_summary for current signal (for Summary + Clustering) ----------
            df_summary_current = build_global_summary_for_signal(
                segmented_ok,
                segmented_test,
                bead_options,
                ok_files,
                test_files,
                signal_col=signal_col,
                step_interval=global_step_interval,
                norm_lower=global_norm_lower,
                norm_upper=global_norm_upper,
                z_lower=global_z_lower,
                z_upper=global_z_upper,
                window=51,
                poly=2
            )

            # ------------ Tab 0: Global Summary (All Beads) ------------
            with tabs[0]:
                st.subheader("Global Summary of Suspected NOK (All Beads, Smoothed Basis)")

                if df_summary_current is None or df_summary_current.empty:
                    st.info("No suspected NOK beads found with current thresholds.")
                else:
                    df_summary = df_summary_current.copy()
                    st.dataframe(df_summary, use_container_width=True)

                    # Color rule for summary scatter:
                    # - Is_NG True  -> red (Correct NOK)
                    # - Is_NG False -> black (Over-detected NOK)

                    # 1) Norm_Low_Exceed vs Bead
                    df1 = df_summary[df_summary["Norm_Low_Exceed"].notna()]
                    if not df1.empty:
                        fig1 = go.Figure()
                        colors1 = ["red" if is_ng else "black" for is_ng in df1["Is_NG"]]
                        fig1.add_trace(
                            go.Scatter(
                                x=df1["Bead"],
                                y=df1["Norm_Low_Exceed"],
                                mode="markers",
                                marker=dict(size=8, color=colors1),
                                text=(
                                    "File: " + df1["CSV_File"].astype(str)
                                    + "<br>Bead: " + df1["Bead"].astype(str)
                                    + "<br>Norm_Low_Exceed: " + df1["Norm_Low_Exceed"].round(4).astype(str)
                                ),
                                hovertemplate="%{text}<extra></extra>"
                            )
                        )
                        fig1.update_layout(
                            title="Norm_Low_Exceed vs Bead",
                            xaxis_title="Bead Number",
                            yaxis_title="Norm_Low_Exceed"
                        )
                        st.plotly_chart(fig1, use_container_width=True)

                    # 2) Norm_High_Exceed vs Bead
                    df2 = df_summary[df_summary["Norm_High_Exceed"].notna()]
                    if not df2.empty:
                        fig2 = go.Figure()
                        colors2 = ["red" if is_ng else "black" for is_ng in df2["Is_NG"]]
                        fig2.add_trace(
                            go.Scatter(
                                x=df2["Bead"],
                                y=df2["Norm_High_Exceed"],
                                mode="markers",
                                marker=dict(size=8, color=colors2),
                                text=(
                                    "File: " + df2["CSV_File"].astype(str)
                                    + "<br>Bead: " + df2["Bead"].astype(str)
                                    + "<br>Norm_High_Exceed: " + df2["Norm_High_Exceed"].round(4).astype(str)
                                ),
                                hovertemplate="%{text}<extra></extra>"
                            )
                        )
                        fig2.update_layout(
                            title="Norm_High_Exceed vs Bead",
                            xaxis_title="Bead Number",
                            yaxis_title="Norm_High_Exceed"
                        )
                        st.plotly_chart(fig2, use_container_width=True)

                    # 3) Z_Low_Exceed vs Bead
                    df3 = df_summary[df_summary["Z_Low_Exceed"].notna()]
                    if not df3.empty:
                        fig3 = go.Figure()
                        colors3 = ["red" if is_ng else "black" for is_ng in df3["Is_NG"]]
                        fig3.add_trace(
                            go.Scatter(
                                x=df3["Bead"],
                                y=df3["Z_Low_Exceed"],
                                mode="markers",
                                marker=dict(size=8, color=colors3),
                                text=(
                                    "File: " + df3["CSV_File"].astype(str)
                                    + "<br>Bead: " + df3["Bead"].astype(str)
                                    + "<br>Z_Low_Exceed: " + df3["Z_Low_Exceed"].round(4).astype(str)
                                ),
                                hovertemplate="%{text}<extra></extra>"
                            )
                        )
                        fig3.update_layout(
                            title="Z_Low_Exceed vs Bead",
                            xaxis_title="Bead Number",
                            yaxis_title="Z_Low_Exceed"
                        )
                        st.plotly_chart(fig3, use_container_width=True)

                    # 4) Z_High_Exceed vs Bead
                    df4 = df_summary[df_summary["Z_High_Exceed"].notna()]
                    if not df4.empty:
                        fig4 = go.Figure()
                        colors4 = ["red" if is_ng else "black" for is_ng in df4["Is_NG"]]
                        fig4.add_trace(
                            go.Scatter(
                                x=df4["Bead"],
                                y=df4["Z_High_Exceed"],
                                mode="markers",
                                marker=dict(size=8, color=colors4),
                                text=(
                                    "File: " + df4["CSV_File"].astype(str)
                                    + "<br>Bead: " + df4["Bead"].astype(str)
                                    + "<br>Z_High_Exceed: " + df4["Z_High_Exceed"].round(4).astype(str)
                                ),
                                hovertemplate="%{text}<extra></extra>"
                            )
                        )
                        fig4.update_layout(
                            title="Z_High_Exceed vs Bead",
                            xaxis_title="Bead Number",
                            yaxis_title="Z_High_Exceed"
                        )
                        st.plotly_chart(fig4, use_container_width=True)

            # ------------ Build per-bead observations for detailed tabs ------------
            # For the selected bead only:
            ref_observations = []
            for fname in ok_files:
                beads = segmented_ok[fname]
                if selected_bead in beads:
                    bead_df = beads[selected_bead]
                    if signal_col in bead_df.columns:
                        data = bead_df[signal_col].reset_index(drop=True)
                        ref_observations.append({
                            "csv": fname,
                            "data": data
                        })

            test_observations = []
            for fname in test_files:
                beads = segmented_test[fname]
                if selected_bead in beads:
                    bead_df = beads[selected_bead]
                    if signal_col in bead_df.columns:
                        data = bead_df[signal_col].reset_index(drop=True)
                        test_observations.append({
                            "csv": fname,
                            "data": data
                        })

            # ------------ Tab 1: Raw Signal ------------
            with tabs[1]:
                st.subheader("Raw Signal (Top) + Per-step Normalization (Bottom)")
                if not ref_observations or not test_observations:
                    st.warning("No data for this bead in OK or TEST set.")
                else:
                    ref_transformed_raw = compute_transformed_signals(
                        ref_observations, mode="raw"
                    )
                    test_transformed_raw = compute_transformed_signals(
                        test_observations, mode="raw"
                    )

                    fig_norm_raw, status_raw, _ = compute_step_normalization_and_flags(
                        ref_transformed_raw,
                        test_transformed_raw,
                        step_interval=global_step_interval,
                        norm_lower=global_norm_lower,
                        norm_upper=global_norm_upper,
                        z_lower=global_z_lower,
                        z_upper=global_z_upper,
                        title_suffix=f"• Raw Signal {signal_col} • Bead #{selected_bead}"
                    )
                    if fig_norm_raw is not None:
                        plot_top_signals(
                            ref_transformed_raw,
                            test_transformed_raw,
                            status_raw,
                            title=f"Raw Signal {signal_col} • Bead #{selected_bead} • Recipe: Norm[{global_norm_lower},{global_norm_upper}] Z-score[-{global_z_lower},{global_z_upper}] Step[{global_step_interval}]",
                            y_label="Signal Value"
                        )
                        st.plotly_chart(fig_norm_raw, use_container_width=True)

            # ------------ Tab 2: Smoothed (Savitzky) ------------
            with tabs[2]:
                st.subheader("Smoothed Signal (Top) + Per-step Normalization (Bottom)")
                if not ref_observations or not test_observations:
                    st.warning("No data for this bead in OK or TEST set.")
                else:
                    window = st.slider(
                        "Savitzky-Golay Window Length",
                        min_value=5,
                        max_value=101,
                        value=51,
                        step=10
                    )
                    poly = st.slider(
                        "Polynomial Order",
                        min_value=2,
                        max_value=5,
                        value=2,
                        step=1
                    )

                    ref_transformed_sg = compute_transformed_signals(
                        ref_observations,
                        mode="savgol",
                        window=window,
                        poly=poly
                    )
                    test_transformed_sg = compute_transformed_signals(
                        test_observations,
                        mode="savgol",
                        window=window,
                        poly=poly
                    )

                    fig_norm_sg, status_sg, _ = compute_step_normalization_and_flags(
                        ref_transformed_sg,
                        test_transformed_sg,
                        step_interval=global_step_interval,
                        norm_lower=global_norm_lower,
                        norm_upper=global_norm_upper,
                        z_lower=global_z_lower,
                        z_upper=global_z_upper,
                        title_suffix=f"• Smoothed Signal {signal_col} • Bead #{selected_bead}"
                    )
                    if fig_norm_sg is not None:
                        plot_top_signals(
                            ref_transformed_sg,
                            test_transformed_sg,
                            status_sg,
                            title=f"Smoothed Signal {signal_col} • Bead #{selected_bead} • Recipe: Norm[{global_norm_lower},{global_norm_upper}] Z-score[-{global_z_lower},{global_z_upper}] Step[{global_step_interval}]",
                            y_label="Signal Value"
                        )
                        st.plotly_chart(fig_norm_sg, use_container_width=True)

            # ------------ Tab 3: Low-pass Filter ------------
            with tabs[3]:
                st.subheader("Low-pass Filtered Signal (Top) + Per-step Normalization (Bottom)")
                if not ref_observations or not test_observations:
                    st.warning("No data for this bead in OK or TEST set.")
                else:
                    cutoff = st.slider(
                        "Low-pass Cutoff Frequency (normalized, 0.01–0.5)",
                        min_value=0.01,
                        max_value=0.5,
                        value=0.1,
                        step=0.01
                    )
                    order = st.slider(
                        "Filter Order",
                        min_value=1,
                        max_value=5,
                        value=2,
                        step=1
                    )

                    ref_transformed_lp = compute_transformed_signals(
                        ref_observations,
                        mode="lowpass",
                        cutoff=cutoff,
                        order=order
                    )
                    test_transformed_lp = compute_transformed_signals(
                        test_observations,
                        mode="lowpass",
                        cutoff=cutoff,
                        order=order
                    )

                    fig_norm_lp, status_lp, _ = compute_step_normalization_and_flags(
                        ref_transformed_lp,
                        test_transformed_lp,
                        step_interval=global_step_interval,
                        norm_lower=global_norm_lower,
                        norm_upper=global_norm_upper,
                        z_lower=global_z_lower,
                        z_upper=global_z_upper,
                        title_suffix=f"• Low-pass Signal {signal_col} • Bead #{selected_bead}"
                    )
                    if fig_norm_lp is not None:
                        plot_top_signals(
                            ref_transformed_lp,
                            test_transformed_lp,
                            status_lp,
                            title=f"Low-pass Filtered Signal {signal_col} • Bead #{selected_bead} • Recipe: Norm[{global_norm_lower},{global_norm_upper}] Z-score[-{global_z_lower},{global_z_upper}] Step[{global_step_interval}]",
                            y_label="Signal Value"
                        )
                        st.plotly_chart(fig_norm_lp, use_container_width=True)

            # ------------ Tab 4: Curve Fit ------------
            with tabs[4]:
                st.subheader("Curve Fit Signal (Top) + Per-step Normalization (Bottom)")
                if not ref_observations or not test_observations:
                    st.warning("No data for this bead in OK or TEST set.")
                else:
                    deg = st.slider(
                        "Curve Fit Polynomial Degree",
                        min_value=1,
                        max_value=100,
                        value=25,
                        step=1
                    )

                    ref_transformed_cf = compute_transformed_signals(
                        ref_observations,
                        mode="poly",
                        deg=deg
                    )
                    test_transformed_cf = compute_transformed_signals(
                        test_observations,
                        mode="poly",
                        deg=deg
                    )

                    fig_norm_cf, status_cf, _ = compute_step_normalization_and_flags(
                        ref_transformed_cf,
                        test_transformed_cf,
                        step_interval=global_step_interval,
                        norm_lower=global_norm_lower,
                        norm_upper=global_norm_upper,
                        z_lower=global_z_lower,
                        z_upper=global_z_upper,
                        title_suffix=f"• Curve Fit Signal {signal_col} • Bead #{selected_bead}"
                    )
                    if fig_norm_cf is not None:
                        plot_top_signals(
                            ref_transformed_cf,
                            test_transformed_cf,
                            status_cf,
                            title=f"Curve-fit Signal {signal_col} • Bead #{selected_bead} • Recipe: Norm[{global_norm_lower},{global_norm_upper}] Z-score[-{global_z_lower},{global_z_upper}] Step[{global_step_interval}]",
                            y_label="Signal Value"
                        )
                        st.plotly_chart(fig_norm_cf, use_container_width=True)

            # ------------ Tab 5: Clustering (NIR / VIS) ------------
            with tabs[5]:
                st.subheader("Clustering of NOK Features (NIR / VIS)")

                # We treat first column as NIR, second as VIS
                all_cols = example_bead_df.columns.tolist()
                if len(all_cols) < 2:
                    st.info("Need at least 2 signal columns to form NIR (1st) and VIS (2nd).")
                else:
                    nir_col = all_cols[0]
                    vis_col = all_cols[1]

                    st.markdown(
                        f"- NIR signal assumed as **1st column**: `{nir_col}`  \n"
                        f"- VIS signal assumed as **2nd column**: `{vis_col}`"
                    )

                    mode = st.radio(
                        "Select which signal to cluster",
                        options=["NIR only", "VIS only", "Both (NIR & VIS)"],
                        index=2,
                        horizontal=True
                    )

                    # Helper to build and cluster for a given signal column
                    def clustering_for_signal(sig_col_name: str, label: str):
                        st.markdown(f"#### {label} – NOK Feature Clustering")

                        df_sig = build_global_summary_for_signal(
                            segmented_ok,
                            segmented_test,
                            bead_options,
                            ok_files,
                            test_files,
                            signal_col=sig_col_name,
                            step_interval=global_step_interval,
                            norm_lower=global_norm_lower,
                            norm_upper=global_norm_upper,
                            z_lower=global_z_lower,
                            z_upper=global_z_upper,
                            window=51,
                            poly=2
                        )

                        if df_sig is None or df_sig.empty:
                            st.info(f"No suspected NOK for {label} with current thresholds.")
                            return

                        st.dataframe(df_sig, use_container_width=True)

                        # Upper exceed features (Norm_High_Exceed vs Z_High_Exceed)
                        df_u = df_sig.dropna(subset=["Norm_High_Exceed", "Z_High_Exceed"]).copy()
                        if len(df_u) >= 2:
                            n_clusters_u = st.slider(
                                f"{label} Upper: Number of clusters (K)",
                                min_value=2,
                                max_value=min(6, len(df_u)),
                                value=min(3, len(df_u)),
                                step=1
                            )
                            km_u = KMeans(n_clusters=n_clusters_u, random_state=0, n_init=10)
                            X_u = df_u[["Norm_High_Exceed", "Z_High_Exceed"]].to_numpy()
                            df_u["Cluster_Upper"] = km_u.fit_predict(X_u)
                        else:
                            if not df_u.empty:
                                df_u["Cluster_Upper"] = np.nan
                            n_clusters_u = 0

                        # Lower exceed features (distance below threshold)
                        df_l = df_sig.dropna(subset=["Norm_Low_Exceed", "Z_Low_Exceed"]).copy()
                        if len(df_l) >= 2:
                            df_l["X_Lower"] = np.maximum(
                                0.0,
                                global_norm_lower - df_l["Norm_Low_Exceed"]
                            )
                            df_l["Y_Lower"] = np.maximum(
                                0.0,
                                -df_l["Z_Low_Exceed"] - global_z_lower
                            )
                            n_clusters_l = st.slider(
                                f"{label} Lower: Number of clusters (K)",
                                min_value=2,
                                max_value=min(6, len(df_l)),
                                value=min(3, len(df_l)),
                                step=1
                            )
                            km_l = KMeans(n_clusters=n_clusters_l, random_state=0, n_init=10)
                            X_l = df_l[["X_Lower", "Y_Lower"]].to_numpy()
                            df_l["Cluster_Lower"] = km_l.fit_predict(X_l)
                        else:
                            if not df_l.empty:
                                df_l["X_Lower"] = np.maximum(
                                    0.0,
                                    global_norm_lower - df_l["Norm_Low_Exceed"]
                                )
                                df_l["Y_Lower"] = np.maximum(
                                    0.0,
                                    -df_l["Z_Low_Exceed"] - global_z_lower
                                )
                                df_l["Cluster_Lower"] = np.nan
                            n_clusters_l = 0

                        # --- Visualization: one figure with 2 subplots (Upper, Lower) ---
                        fig_sig = make_subplots(
                            rows=1,
                            cols=2,
                            subplot_titles=(
                                f"{label}: Upper Exceed (Norm_High vs Z_High)",
                                f"{label}: Lower Exceed (distance below thresholds)"
                            )
                        )

                        # Color mapping by Is_NG (true NOK vs over-detected)
                        def ng_color(is_ng):
                            return "red" if is_ng else "black"

                        # Marker symbols by cluster id
                        symbol_list = ["circle", "square", "diamond", "triangle-up", "cross", "x"]

                        # ----- Upper subplot -----
                        if not df_u.empty:
                            for _, row in df_u.iterrows():
                                x_val = row["Norm_High_Exceed"]
                                y_val = row["Z_High_Exceed"]
                                color = ng_color(row["Is_NG"])
                                cid = row.get("Cluster_Upper", np.nan)
                                if not pd.isna(cid):
                                    cid_int = int(cid)
                                    symbol = symbol_list[cid_int % len(symbol_list)]
                                    cluster_text = f"Cluster {cid_int}"
                                else:
                                    symbol = "circle"
                                    cluster_text = "Cluster: N/A"

                                fig_sig.add_trace(
                                    go.Scatter(
                                        x=[x_val],
                                        y=[y_val],
                                        mode="markers",
                                        marker=dict(
                                            size=9,
                                            color=color,
                                            symbol=symbol
                                        ),
                                        text=(
                                            f"File: {row['CSV_File']}<br>"
                                            f"Bead: {row['Bead']}<br>"
                                            f"Is_NG: {row['Is_NG']}<br>"
                                            f"Status(Algo): {row['Status']}<br>"
                                            f"Norm_High_Exceed: {row['Norm_High_Exceed']:.4f}<br>"
                                            f"Z_High_Exceed: {row['Z_High_Exceed']:.4f}<br>"
                                            f"{cluster_text}"
                                        ),
                                        hovertemplate="%{text}<extra></extra>"
                                    ),
                                    row=1,
                                    col=1
                                )

                        fig_sig.update_xaxes(
                            title_text="Norm_High_Exceed",
                            row=1,
                            col=1
                        )
                        fig_sig.update_yaxes(
                            title_text="Z_High_Exceed",
                            row=1,
                            col=1
                        )

                        # ----- Lower subplot -----
                        if not df_l.empty:
                            for _, row in df_l.iterrows():
                                x_val = row["X_Lower"]
                                y_val = row["Y_Lower"]
                                color = ng_color(row["Is_NG"])
                                cid = row.get("Cluster_Lower", np.nan)
                                if not pd.isna(cid):
                                    cid_int = int(cid)
                                    symbol = symbol_list[cid_int % len(symbol_list)]
                                    cluster_text = f"Cluster {cid_int}"
                                else:
                                    symbol = "circle"
                                    cluster_text = "Cluster: N/A"

                                fig_sig.add_trace(
                                    go.Scatter(
                                        x=[x_val],
                                        y=[y_val],
                                        mode="markers",
                                        marker=dict(
                                            size=9,
                                            color=color,
                                            symbol=symbol
                                        ),
                                        text=(
                                            f"File: {row['CSV_File']}<br>"
                                            f"Bead: {row['Bead']}<br>"
                                            f"Is_NG: {row['Is_NG']}<br>"
                                            f"Status(Algo): {row['Status']}<br>"
                                            f"Norm_Low_Exceed: {row['Norm_Low_Exceed']:.4f}<br>"
                                            f"Z_Low_Exceed: {row['Z_Low_Exceed']:.4f}<br>"
                                            f"{cluster_text}"
                                        ),
                                        hovertemplate="%{text}<extra></extra>"
                                    ),
                                    row=1,
                                    col=2
                                )

                        fig_sig.update_xaxes(
                            title_text="Distance below Norm lower limit",
                            row=1,
                            col=2
                        )
                        fig_sig.update_yaxes(
                            title_text="Distance below Z lower limit",
                            row=1,
                            col=2
                        )

                        fig_sig.update_layout(
                            title=dict(
                                text=f"{label} – Clustering in Upper/Lower Exceed Space",
                                font=dict(size=20)
                            ),
                            showlegend=False
                        )
                        st.plotly_chart(fig_sig, use_container_width=True)

                    # Execute depending on user choice
                    if mode in ["NIR only", "Both (NIR & VIS)"]:
                        clustering_for_signal(nir_col, label="NIR (1st Column)")
                    if mode in ["VIS only", "Both (NIR & VIS)"]:
                        clustering_for_signal(vis_col, label="VIS (2nd Column)")

else:
    st.info("Please upload and segment both OK and TEST ZIP files to start the analysis.")
