import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import savgol_filter, butter, filtfilt

st.set_page_config(layout="wide")

# --- Utility: Bead Segmentation ---
def segment_beads(df, column, threshold):
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

def aggregate_for_step(x, y, interval):
    """Aggregate x and y into buckets of given interval for step plotting."""
    if interval <= 0:
        interval = 1
    x = np.asarray(x)
    y = np.asarray(y)
    agg_x = x[::interval]
    agg_y = [np.mean(y[i:i+interval]) for i in range(0, len(y), interval)]
    agg_x = agg_x[:len(agg_y)]
    return agg_x, np.array(agg_y)

# --- Session State ---
if "segmented_ok" not in st.session_state:
    st.session_state.segmented_ok = None
if "segmented_test" not in st.session_state:
    st.session_state.segmented_test = None
if "seg_col" not in st.session_state:
    st.session_state.seg_col = None
if "seg_thresh" not in st.session_state:
    st.session_state.seg_thresh = None

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
                value=1.0
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
# Helper: Compute per-step normalization & flags
# ============================================================
def compute_step_normalization_and_flags(
    ref_obs,
    test_obs,
    step_interval,
    norm_thresh,
    z_thresh,
    title_suffix
):
    """
    ref_obs: list of dicts (OK reference), with 'transformed'
    test_obs: list of dicts (TEST), with 'transformed'
    Returns: (fig, flagged_map)
        - fig: plotly Figure for bottom chart
        - flagged_map: {csv_name: bool} True if flagged (test only)
    """
    if len(ref_obs) == 0:
        st.warning("No OK reference signals available for normalization.")
        return None, {}

    # Build OK step arrays
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
        return None, {}

    min_steps = min(arr.shape[0] for arr in ok_step_arrays)
    if min_steps == 0:
        st.warning("OK step data are empty after aggregation.")
        return None, {}

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

    flagged_map = {}

    # --- Plot OK reference signals in gray (bottom plots) ---
    for meta in ok_step_meta:
        step_y_ok = meta["step_y"][:min_steps]
        norm_ok = (step_y_ok - min_ok) / denom
        z_ok = (step_y_ok - mu) / sigma

        fig.add_trace(
            go.Scatter(
                x=step_indices,
                y=norm_ok,
                mode="lines",
                name=f"{meta['csv']} (OK ref)",
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

    # --- Plot TEST signals and compute flags ---
    for obs in test_obs:
        y = np.asarray(obs["transformed"])
        x = np.arange(len(y))
        _, step_y = aggregate_for_step(x, y, step_interval)
        step_y = step_y[:min_steps]

        norm_vals = (step_y - min_ok) / denom
        z_vals = (step_y - mu) / sigma

        flagged = np.any(norm_vals < norm_thresh) or np.any(z_vals < -z_thresh)
        flagged_map[obs["csv"]] = bool(flagged)

        color = "red" if flagged else "green"
        width = 2 if flagged else 1
        name = obs["csv"]

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
        y=norm_thresh,
        line=dict(color="black", dash="dash"),
        row=1,
        col=1
    )
    fig.add_hline(
        y=-z_thresh,
        line=dict(color="black", dash="dash"),
        row=1,
        col=2
    )

    fig.update_xaxes(title_text="Step Index", row=1, col=1)
    fig.update_yaxes(title_text="Normalized Value", row=1, col=1)
    fig.update_xaxes(title_text="Step Index", row=1, col=2)
    fig.update_yaxes(title_text="Z-score", row=1, col=2)

    fig.update_layout(
        title=f"Per-step Normalization {title_suffix}",
        # Let Plotly auto-place legend inside the figure to avoid overlap
        legend=dict(orientation="h")
    )

    return fig, flagged_map

# ============================================================
# Helper: Plot Top (Transformed TEST + OK in gray)
# ============================================================
def plot_top_signals(ref_transformed, test_transformed, flagged_map, title, y_label):
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
                name=f"{obs['csv']} (OK ref)",
                line=dict(color="#aaaaaa", width=1),
                legendgroup="OK_REF"
            )
        )

    # TEST signals, colored by flag
    for obs in test_transformed:
        y = obs["transformed"]
        x = np.arange(len(y))
        csv_name = obs["csv"]
        flagged = flagged_map.get(csv_name, False)
        color = "red" if flagged else "green"
        width = 2 if flagged else 1
        label = "FLAGGED" if flagged else "OK-like"
        name = f"{csv_name} (TEST, {label})"

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name=name,
                line=dict(color=color, width=width),
                legendgroup=csv_name
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Index",
        yaxis_title=y_label,
        legend=dict(orientation="h")  # let Plotly position this inside the figure
    )
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# STEP 3: Analysis (Requires both OK & TEST segmented)
# ============================================================
if st.session_state.segmented_ok and st.session_state.segmented_test:
    st.sidebar.header("Step 3: Global Thresholds")

    # Global thresholds for all tabs
    global_norm_thresh = st.sidebar.number_input(
        "Global Lower Threshold for 0–1 Normalization (flag if below)",
        min_value=-1.0,
        max_value=1.0,
        value=0.0,
        step=0.05
    )
    global_z_thresh = st.sidebar.number_input(
        "Global Z-score Threshold (flag if below -T)",
        min_value=0.5,
        max_value=6.0,
        value=3.0,
        step=0.5
    )

    st.sidebar.header("Step 4: Bead & Signal for Analysis")

    segmented_ok = st.session_state.segmented_ok
    segmented_test = st.session_state.segmented_test

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

            # Build OK reference observations for this bead & signal
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

            # Build TEST observations for this bead & signal
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

            if len(ref_observations) == 0:
                st.warning("No OK reference signals found for this bead and signal.")
            elif len(test_observations) == 0:
                st.warning("No TEST signals found for this bead and signal.")
            else:
                st.markdown(
                    f"### Analysis for Bead **{selected_bead}**, Signal **{signal_col}**"
                )
                st.markdown(
                    f"- OK reference files: {len(ref_observations)}  \n"
                    f"- TEST files: {len(test_observations)}"
                )

                tabs = st.tabs([
                    "Raw Signal",
                    "Smoothed (Savitzky)",
                    "Low-pass Filter",
                    "Curve Fit"
                ])

                # ------------ Tab 1: Raw Signal ------------
                with tabs[0]:
                    st.subheader("Raw Signal (Top) + Per-step Normalization (Bottom)")

                    step_interval_raw = st.slider(
                        "Step Interval (points) - Raw",
                        min_value=10,
                        max_value=500,
                        value=70,
                        step=10
                    )

                    ref_transformed_raw = compute_transformed_signals(
                        ref_observations, mode="raw"
                    )
                    test_transformed_raw = compute_transformed_signals(
                        test_observations, mode="raw"
                    )

                    fig_norm_raw, flagged_raw = compute_step_normalization_and_flags(
                        ref_transformed_raw,
                        test_transformed_raw,
                        step_interval=step_interval_raw,
                        norm_thresh=global_norm_thresh,
                        z_thresh=global_z_thresh,
                        title_suffix="(Raw Signal)"
                    )
                    if fig_norm_raw is not None:
                        plot_top_signals(
                            ref_transformed_raw,
                            test_transformed_raw,
                            flagged_raw,
                            title="Raw Signal (TEST, colored by global thresholds)",
                            y_label="Signal Value"
                        )
                        st.plotly_chart(fig_norm_raw, use_container_width=True)

                # ------------ Tab 2: Smoothed (Savitzky) ------------
                with tabs[1]:
                    st.subheader("Smoothed Signal (Top) + Per-step Normalization (Bottom)")

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
                    step_interval_sg = st.slider(
                        "Step Interval (points) - Smoothed",
                        min_value=10,
                        max_value=500,
                        value=70,
                        step=10
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

                    fig_norm_sg, flagged_sg = compute_step_normalization_and_flags(
                        ref_transformed_sg,
                        test_transformed_sg,
                        step_interval=step_interval_sg,
                        norm_thresh=global_norm_thresh,
                        z_thresh=global_z_thresh,
                        title_suffix="(Smoothed)"
                    )
                    if fig_norm_sg is not None:
                        plot_top_signals(
                            ref_transformed_sg,
                            test_transformed_sg,
                            flagged_sg,
                            title="Smoothed Signal (TEST, colored by global thresholds)",
                            y_label="Signal Value"
                        )
                        st.plotly_chart(fig_norm_sg, use_container_width=True)

                # ------------ Tab 3: Low-pass Filter ------------
                with tabs[2]:
                    st.subheader("Low-pass Filtered Signal (Top) + Per-step Normalization (Bottom)")

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
                    step_interval_lp = st.slider(
                        "Step Interval (points) - Low-pass",
                        min_value=10,
                        max_value=500,
                        value=70,
                        step=10
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

                    fig_norm_lp, flagged_lp = compute_step_normalization_and_flags(
                        ref_transformed_lp,
                        test_transformed_lp,
                        step_interval=step_interval_lp,
                        norm_thresh=global_norm_thresh,
                        z_thresh=global_z_thresh,
                        title_suffix="(Low-pass)"
                    )
                    if fig_norm_lp is not None:
                        plot_top_signals(
                            ref_transformed_lp,
                            test_transformed_lp,
                            flagged_lp,
                            title="Low-pass Filtered Signal (TEST, colored by global thresholds)",
                            y_label="Signal Value"
                        )
                        st.plotly_chart(fig_norm_lp, use_container_width=True)

                # ------------ Tab 4: Curve Fit ------------
                with tabs[3]:
                    st.subheader("Curve Fit Signal (Top) + Per-step Normalization (Bottom)")

                    deg = st.slider(
                        "Curve Fit Polynomial Degree",
                        min_value=1,
                        max_value=100,
                        value=25,
                        step=1
                    )
                    step_interval_cf = st.slider(
                        "Step Interval (points) - Curve Fit",
                        min_value=10,
                        max_value=500,
                        value=70,
                        step=10
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

                    fig_norm_cf, flagged_cf = compute_step_normalization_and_flags(
                        ref_transformed_cf,
                        test_transformed_cf,
                        step_interval=step_interval_cf,
                        norm_thresh=global_norm_thresh,
                        z_thresh=global_z_thresh,
                        title_suffix="(Curve Fit)"
                    )
                    if fig_norm_cf is not None:
                        plot_top_signals(
                            ref_transformed_cf,
                            test_transformed_cf,
                            flagged_cf,
                            title="Curve-fit Signal (TEST, colored by global thresholds)",
                            y_label="Signal Value"
                        )
                        st.plotly_chart(fig_norm_cf, use_container_width=True)
