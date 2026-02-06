import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(layout="wide")

# Notebook parity: step aggregation method ("median" or "mean")
STEP_AGG_METHOD = "median"

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
    """Aggregate x and y into buckets of given interval for step plotting.
    Uses STEP_AGG_METHOD (median/mean) to match notebook logic.
    """
    if interval <= 0:
        interval = 1
    x = np.asarray(x)
    y = np.asarray(y, dtype=float)

    agg_x = x[::interval]
    chunks = [y[i:i+interval] for i in range(0, len(y), interval)]
    if STEP_AGG_METHOD == "median":
        agg_y = [np.median(c) for c in chunks if len(c)]
    else:
        agg_y = [np.mean(c) for c in chunks if len(c)]
    agg_x = agg_x[:len(agg_y)]
    return agg_x, np.array(agg_y, dtype=float)

def short_label(csv_name: str) -> str:
    base = os.path.splitext(os.path.basename(csv_name))[0]
    return base[:6] if len(base) > 6 else base

def get_channel_columns(bead_df: pd.DataFrame):
    """
    No hard-coding: Channel 1 and Channel 2 are interpreted as
    the first and second columns in the bead dataframe.
    """
    cols = bead_df.columns.tolist()
    if len(cols) < 2:
        return []
    return [cols[0], cols[1]]

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
                value=0.2
            )
            segment_ok_btn = st.sidebar.button("Segment OK Files")

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
# Helper: Transform Signals (RAW ONLY)
# ============================================================
def compute_transformed_signals(observations, mode="raw", **params):
    transformed_obs = []
    for obs in observations:
        y = np.asarray(obs["data"]).astype(float)
        transformed_obs.append({
            **obs,
            "transformed": y
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
    """Notebook-consistent per-step reference:
    - Build reference length L from OK only.
    - For each TEST, evaluate on L_eff = min(len(TEST), L) (truncate safely).
    - Plot arrays are padded with NaN to keep a stable x-axis of length L.
    """
    if len(ref_obs) == 0:
        st.warning("No OK reference signals available for normalization.")
        return None, {}, {}

    ok_step_arrays, ok_step_meta = [], []
    for obs in ref_obs:
        y = np.asarray(obs["transformed"], dtype=float)
        x = np.arange(len(y))
        _, step_y = aggregate_for_step(x, y, step_interval)
        if step_y.size == 0:
            continue
        ok_step_arrays.append(step_y)
        ok_step_meta.append({"csv": obs["csv"], "step_y": step_y})

    if len(ok_step_arrays) == 0:
        st.warning("No valid OK step data for normalization.")
        return None, {}, {}

    test_step_arrays, test_step_meta = [], []
    for obs in test_obs:
        y = np.asarray(obs["transformed"], dtype=float)
        x = np.arange(len(y))
        _, step_y = aggregate_for_step(x, y, step_interval)
        if step_y.size == 0:
            continue
        test_step_arrays.append(step_y)
        test_step_meta.append({"csv": obs["csv"], "step_y": step_y})

    if len(test_step_arrays) == 0:
        st.warning("No valid TEST step data for normalization.")
        return None, {}, {}

    # Reference length comes from OK only (not min(OK, TEST)).
    L_ok = min(arr.shape[0] for arr in ok_step_arrays)
    if L_ok <= 0:
        st.warning("OK step data are empty after aggregation.")
        return None, {}, {}

    ok_matrix = np.vstack([arr[:L_ok] for arr in ok_step_arrays])

    mu = np.median(ok_matrix, axis=0)
    sigma = ok_matrix.std(axis=0, ddof=1)
    sigma[sigma < 1e-12] = 1e-12

    min_ok = ok_matrix.min(axis=0)
    max_ok = ok_matrix.max(axis=0)
    denom = max_ok - min_ok
    denom[denom < 1e-12] = 1e-12

    step_indices = np.arange(L_ok)

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("0–1 Normalization (OK-based)", "Z-score per Step")
    )

    status_map = {}
    metrics_map = {}

    # OK refs
    for meta in ok_step_meta:
        step_y_ok = meta["step_y"][:L_ok]
        norm_ok = (step_y_ok - min_ok) / denom
        z_ok = (step_y_ok - mu) / sigma

        fig.add_trace(
            go.Scatter(
                x=step_indices, y=norm_ok, mode="lines",
                name=f"{short_label(meta['csv'])} (OK ref)",
                line=dict(color="#aaaaaa", width=1),
                legendgroup="OK_REF", showlegend=True
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=step_indices, y=z_ok, mode="lines",
                line=dict(color="#aaaaaa", width=1),
                legendgroup="OK_REF", showlegend=False
            ),
            row=1, col=2
        )

    # TEST
    for meta in test_step_meta:
        step_y_full = np.asarray(meta["step_y"], dtype=float)
        csv_name = meta["csv"]

        L_eff = min(len(step_y_full), L_ok)
        if L_eff <= 0:
            status_map[csv_name] = "low"
            metrics_map[csv_name] = {
                "Norm_Low_Exceed": np.nan,
                "Norm_High_Exceed": np.nan,
                "Z_Low_Exceed": np.nan,
                "Z_High_Exceed": np.nan,
            }
            continue

        y = step_y_full[:L_eff]
        norm_vals_eff = (y - min_ok[:L_eff]) / denom[:L_eff]
        z_vals_eff = (y - mu[:L_eff]) / sigma[:L_eff]

        mask_norm_low = norm_vals_eff < norm_lower
        mask_norm_high = norm_vals_eff > norm_upper
        mask_z_low = z_vals_eff < -z_lower
        mask_z_high = z_vals_eff > z_upper

        norm_low_exceed = norm_vals_eff[mask_norm_low].min() if mask_norm_low.any() else np.nan
        norm_high_exceed = norm_vals_eff[mask_norm_high].max() if mask_norm_high.any() else np.nan
        z_low_exceed = z_vals_eff[mask_z_low].min() if mask_z_low.any() else np.nan
        z_high_exceed = z_vals_eff[mask_z_high].max() if mask_z_high.any() else np.nan

        low_flag = mask_norm_low.any() or mask_z_low.any()
        high_flag = mask_norm_high.any() or mask_z_high.any()

        if low_flag:
            status = "low"
        elif high_flag:
            status = "high"
        else:
            status = "ok"

        status_map[csv_name] = status
        metrics_map[csv_name] = {
            "Norm_Low_Exceed": norm_low_exceed,
            "Norm_High_Exceed": norm_high_exceed,
            "Z_Low_Exceed": z_low_exceed,
            "Z_High_Exceed": z_high_exceed,
        }

        if status == "low":
            color, width = "red", 2
        elif status == "high":
            color, width = "orange", 2
        else:
            color, width = "green", 1

        name = short_label(csv_name)

        # pad to L_ok for stable x-axis
        norm_plot = np.full(L_ok, np.nan, dtype=float)
        z_plot = np.full(L_ok, np.nan, dtype=float)
        norm_plot[:L_eff] = norm_vals_eff
        z_plot[:L_eff] = z_vals_eff

        fig.add_trace(
            go.Scatter(
                x=step_indices, y=norm_plot, mode="lines",
                name=f"{name} (TEST)",
                line=dict(color=color, width=width),
                legendgroup=name, showlegend=True
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=step_indices, y=z_plot, mode="lines",
                line=dict(color=color, width=width),
                legendgroup=name, showlegend=False
            ),
            row=1, col=2
        )

    fig.add_hline(y=norm_lower, line=dict(color="gray", dash="dash"), row=1, col=1)
    fig.add_hline(y=norm_upper, line=dict(color="gray", dash="dash"), row=1, col=1)
    fig.add_hline(y=-z_lower, line=dict(color="gray", dash="dash"), row=1, col=2)
    fig.add_hline(y=z_upper, line=dict(color="gray", dash="dash"), row=1, col=2)

    fig.update_xaxes(title_text="Step Index", row=1, col=1)
    fig.update_yaxes(title_text="Normalized Value", row=1, col=1)
    fig.update_xaxes(title_text="Step Index", row=1, col=2)
    fig.update_yaxes(title_text="Z-score", row=1, col=2)

    fig.update_layout(
        title=dict(text=f"Per-step Normalization {title_suffix}", font=dict(size=22)),
        legend=dict(orientation="h")
    )

    return fig, status_map, metrics_map

# ============================================================
# Helper: Plot Top (RAW TEST + OK in gray)
# ============================================================
def plot_top_signals(ref_transformed, test_transformed, status_map, title, y_label):
    fig = go.Figure()

    for obs in ref_transformed:
        y = obs["transformed"]
        x = np.arange(len(y))
        fig.add_trace(
            go.Scatter(
                x=x, y=y, mode="lines",
                name=f"{short_label(obs['csv'])} (OK ref)",
                line=dict(color="#aaaaaa", width=1),
                legendgroup="OK_REF"
            )
        )

    for obs in test_transformed:
        y = obs["transformed"]
        x = np.arange(len(y))
        csv_name = obs["csv"]
        status = status_map.get(csv_name, "ok")

        if status == "low":
            color, width, label = "red", 2, "LOW"
        elif status == "high":
            color, width, label = "orange", 2, "HIGH"
        else:
            color, width, label = "green", 1, "OK-like"

        name = f"{short_label(csv_name)} (TEST, {label})"
        fig.add_trace(
            go.Scatter(
                x=x, y=y, mode="lines",
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
# STEP 3: Analysis (Requires both OK & TEST segmented)
# ============================================================
if st.session_state.segmented_ok and st.session_state.segmented_test:
    segmented_ok = st.session_state.segmented_ok
    segmented_test = st.session_state.segmented_test

    st.sidebar.header("Step 3: Global Thresholds & Step Interval")

    global_norm_lower = st.sidebar.number_input(
        "Global Lower Threshold for 0–1 Normalization (flag LOW if below)",
        min_value=-5.0, max_value=5.0, value=-4.0, step=0.05
    )
    global_norm_upper = st.sidebar.number_input(
        "Global Upper Threshold for 0–1 Normalization (flag HIGH if above)",
        min_value=-5.0, max_value=5.0, value=10.0, step=0.05
    )
    global_z_lower = st.sidebar.number_input(
        "Global Z-score Threshold (flag LOW if below -T)",
        min_value=0.5, max_value=10.0, value=6.0, step=0.5
    )
    global_z_upper = st.sidebar.number_input(
        "Global Z-score Threshold (flag HIGH if above +T)",
        min_value=0.5, max_value=20.0, value=20.0, step=0.25
    )

    global_step_interval = st.sidebar.slider(
        "Global Step Interval (points)",
        min_value=10, max_value=500, value=20, step=10
    )

    st.sidebar.header("Step 4: Bead & Signal for Analysis")

    ok_files = sorted(segmented_ok.keys())
    test_files = sorted(segmented_test.keys())

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

            st.markdown(f"### Analysis for Signal **{signal_col}** (RAW ONLY)")
            st.markdown(f"- OK reference files: {len(ok_files)}  \n- TEST files: {len(test_files)}")

            tabs = st.tabs([
                "Summary",
                "Raw Signal",
            ])

            # ------------ Tab 0: Global Summary (All Beads, BOTH CHANNELS, RAW ONLY) ------------
            with tabs[0]:
                st.subheader("Global Summary of Suspected NOK (All Beads, Both Channels, Raw Only)")

                rows = []

                with st.spinner("Running global summary across both channels (raw only)..."):
                    for bead in bead_options:
                        bead_df_for_cols = None
                        for fname in ok_files:
                            beads = segmented_ok[fname]
                            if bead in beads:
                                bead_df_for_cols = beads[bead]
                                break
                        if bead_df_for_cols is None:
                            continue

                        channel_cols = get_channel_columns(bead_df_for_cols)
                        if len(channel_cols) < 2:
                            continue

                        for ch_idx, ch_col in enumerate(channel_cols):  # 0, 1
                            ref_obs_bead = []
                            for fname in ok_files:
                                beads = segmented_ok[fname]
                                if bead in beads:
                                    bead_df = beads[bead]
                                    if ch_col in bead_df.columns:
                                        data = bead_df[ch_col].reset_index(drop=True)
                                        ref_obs_bead.append({"csv": fname, "data": data})

                            test_obs_bead = []
                            for fname in test_files:
                                beads = segmented_test[fname]
                                if bead in beads:
                                    bead_df = beads[bead]
                                    if ch_col in bead_df.columns:
                                        data = bead_df[ch_col].reset_index(drop=True)
                                        test_obs_bead.append({"csv": fname, "data": data})

                            if not ref_obs_bead or not test_obs_bead:
                                continue

                            ref_t = compute_transformed_signals(ref_obs_bead, mode="raw")
                            test_t = compute_transformed_signals(test_obs_bead, mode="raw")

                            _, status_map, metrics_map = compute_step_normalization_and_flags(
                                ref_t,
                                test_t,
                                step_interval=global_step_interval,
                                norm_lower=global_norm_lower,
                                norm_upper=global_norm_upper,
                                z_lower=global_z_lower,
                                z_upper=global_z_upper,
                                title_suffix="(Summary • Raw Signal)"
                            )

                            for csv_name, status in status_map.items():
                                if status == "ok":
                                    continue
                                m = metrics_map.get(csv_name, {})
                                rows.append({
                                    "CSV_File": csv_name,
                                    "Bead": bead,
                                    "Channel": ch_idx,  # int: 0 or 1
                                    "Status": "LOW" if status == "low" else "HIGH",
                                    "SignalTransform": "Raw Signal",
                                    "Norm_Low_Exceed": m.get("Norm_Low_Exceed", np.nan),
                                    "Norm_High_Exceed": m.get("Norm_High_Exceed", np.nan),
                                    "Z_Low_Exceed": m.get("Z_Low_Exceed", np.nan),
                                    "Z_High_Exceed": m.get("Z_High_Exceed", np.nan),
                                })

                if rows:
                    df_summary = pd.DataFrame(rows)
                    df_summary = df_summary.sort_values(
                        ["CSV_File", "Bead", "Channel", "SignalTransform"]
                    ).reset_index(drop=True)
                    df_summary["Is_NG"] = df_summary["CSV_File"].str.contains("NG", case=False)
                    st.dataframe(df_summary, use_container_width=True)
                else:
                    st.info("No suspected NOK beads found with current thresholds (both channels, raw only).")

            # ------------ Build per-bead observations for detailed tab (selected signal_col) ------------
            ref_observations = []
            for fname in ok_files:
                beads = segmented_ok[fname]
                if selected_bead in beads:
                    bead_df = beads[selected_bead]
                    if signal_col in bead_df.columns:
                        data = bead_df[signal_col].reset_index(drop=True)
                        ref_observations.append({"csv": fname, "data": data})

            test_observations = []
            for fname in test_files:
                beads = segmented_test[fname]
                if selected_bead in beads:
                    bead_df = beads[selected_bead]
                    if signal_col in bead_df.columns:
                        data = bead_df[signal_col].reset_index(drop=True)
                        test_observations.append({"csv": fname, "data": data})

            # ------------ Tab 1: Raw Signal ------------
            with tabs[1]:
                st.subheader("Raw Signal (Top) + Per-step Normalization (Bottom)")
                if not ref_observations or not test_observations:
                    st.warning("No data for this bead in OK or TEST set.")
                else:
                    ref_transformed_raw = compute_transformed_signals(ref_observations, mode="raw")
                    test_transformed_raw = compute_transformed_signals(test_observations, mode="raw")

                    fig_norm_raw, status_raw, _ = compute_step_normalization_and_flags(
                        ref_transformed_raw, test_transformed_raw,
                        step_interval=global_step_interval,
                        norm_lower=global_norm_lower, norm_upper=global_norm_upper,
                        z_lower=global_z_lower, z_upper=global_z_upper,
                        title_suffix=f"• Raw Signal {signal_col} • Bead #{selected_bead}"
                    )
                    if fig_norm_raw is not None:
                        plot_top_signals(
                            ref_transformed_raw, test_transformed_raw, status_raw,
                            title=f"Raw Signal {signal_col} • Bead #{selected_bead} • Recipe: Norm[{global_norm_lower},{global_norm_upper}] Z-score[-{global_z_lower},{global_z_upper}] Step[{global_step_interval}]",
                            y_label="Signal Value"
                        )
                        st.plotly_chart(fig_norm_raw, use_container_width=True)
