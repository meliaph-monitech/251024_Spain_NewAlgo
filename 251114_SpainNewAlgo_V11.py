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

def short_label(csv_name: str) -> str:
    """Use first 6 characters of basename (without extension) as label."""
    base = os.path.splitext(os.path.basename(csv_name))[0]
    return base[:6] if len(base) > 6 else base

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

    fig.update_xaxes(title_text="Step Index", row=1, col=1)
    fig.update_yaxes(title_text="Normalized Value", row=1, col=1)
    fig.update_xaxes(title_text="Step Index", row=1, col=2)
    fig.update_yaxes(title_text="Z-score", row=1, col=2)

    # fig.update_layout(
    #     title=f"Per-step Normalization {title_suffix}",
    #     legend=dict(orientation="h")
    # )

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

    # fig.update_layout(
    #     title=title,
    #     xaxis_title="Index",
    #     yaxis_title=y_label,
    #     legend=dict(orientation="h")
    # )
    # st.plotly_chart(fig, use_container_width=True)

    fig.update_layout(
        title_text=title,
        title_font=dict(size=22),   # ← title font size
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

    # Global thresholds for all tabs (using your tuned defaults)
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
        value=4.0,
        step=0.05
    )
    global_z_lower = st.sidebar.number_input(
        "Global Z-score Threshold (flag LOW if below -T)",
        min_value=0.5,
        max_value=10.0,
        value=4.5,
        step=0.5
    )
    global_z_upper = st.sidebar.number_input(
        "Global Z-score Threshold (flag HIGH if above +T)",
        min_value=0.5,
        max_value=20.0,
        value=10.0,
        step=0.25
    )

    # Global step interval
    global_step_interval = st.sidebar.slider(
        "Global Step Interval (points)",
        min_value=10,
        max_value=500,
        value=20,
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

            # Build OK reference observations for this signal (all beads)
            # (We still keep per-bead construction later where needed.)
            st.markdown(
                f"### Analysis for Signal **{signal_col}**"
            )
            st.markdown(
                f"- OK reference files: {len(ok_files)}  \n"
                f"- TEST files: {len(test_files)}"
            )

            tabs = st.tabs([
                "Summary",
                "Raw Signal",
                "Smoothed (Savitzky)",
                "Low-pass Filter",
                "Curve Fit"
            ])

            # ------------ Tab 0: Global Summary (All Beads) ------------
            with tabs[0]:
                st.subheader("Global Summary of Suspected NOK (All Beads, Smoothed Basis)")

                rows = []

                # We summarize using Smoothed transform with fixed defaults
                summary_window = 51
                summary_poly = 2

                for bead in bead_options:
                    # Build OK & TEST obs for this bead
                    ref_obs_bead = []
                    for fname in ok_files:
                        beads = segmented_ok[fname]
                        if bead in beads:
                            bead_df = beads[bead]
                            if signal_col in bead_df.columns:
                                data = bead_df[signal_col].reset_index(drop=True)
                                ref_obs_bead.append({"csv": fname, "data": data})

                    test_obs_bead = []
                    for fname in test_files:
                        beads = segmented_test[fname]
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
                        window=summary_window,
                        poly=summary_poly
                    )
                    test_transformed_bead = compute_transformed_signals(
                        test_obs_bead,
                        mode="savgol",
                        window=summary_window,
                        poly=summary_poly
                    )

                    _, status_bead, metrics_bead = compute_step_normalization_and_flags(
                        ref_transformed_bead,
                        test_transformed_bead,
                        step_interval=global_step_interval,
                        norm_lower=global_norm_lower,
                        norm_upper=global_norm_upper,
                        z_lower=global_z_lower,
                        z_upper=global_z_upper,
                        title_suffix=f"(Bead #{bead}, Smoothed)"
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

                if rows:
                    df_summary = pd.DataFrame(rows)
                    df_summary = df_summary.sort_values(["CSV_File", "Bead"]).reset_index(drop=True)
                
                    # Mark which rows are true NOKs based on file name containing "NG"
                    df_summary["Is_NG"] = df_summary["CSV_File"].str.contains("NG", case=False)
                
                    st.dataframe(df_summary, use_container_width=True)
                
                    # ---- Scatter summaries under the table ----
                    # Color rule:
                    # - CSV_File name contains "NG"  -> Correct NOK detection  -> red
                    # - CSV_File name does NOT have "NG" -> Over-detected NOK  -> black

                    # 1) Norm_Low_Exceed vs Bead
                    df1 = df_summary[df_summary["Norm_Low_Exceed"].notna()]
                    if not df1.empty:
                        fig1 = go.Figure()
                        colors1 = ["red" if is_ng else "grey" for is_ng in df1["Is_NG"]]
                        fig1.add_trace(
                            go.Scatter(
                                x=df1["Bead"],
                                y=df1["Norm_Low_Exceed"],
                                mode="markers",
                                marker=dict(size=8, color=colors1),
                                text=df1["CSV_File"],
                                hovertemplate="CSV: %{text}<br>Bead: %{x}<br>Norm_Low_Exceed: %{y}<extra></extra>"
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
                        colors2 = ["red" if is_ng else "grey" for is_ng in df2["Is_NG"]]
                        fig2.add_trace(
                            go.Scatter(
                                x=df2["Bead"],
                                y=df2["Norm_High_Exceed"],
                                mode="markers",
                                marker=dict(size=8, color=colors2),
                                text=df2["CSV_File"],
                                hovertemplate="CSV: %{text}<br>Bead: %{x}<br>Norm_High_Exceed: %{y}<extra></extra>"
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
                        colors3 = ["red" if is_ng else "grey" for is_ng in df3["Is_NG"]]
                        fig3.add_trace(
                            go.Scatter(
                                x=df3["Bead"],
                                y=df3["Z_Low_Exceed"],
                                mode="markers",
                                marker=dict(size=8, color=colors3),
                                text=df3["CSV_File"],
                                hovertemplate="CSV: %{text}<br>Bead: %{x}<br>Z_Low_Exceed: %{y}<extra></extra>"
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
                        colors4 = ["red" if is_ng else "grey" for is_ng in df4["Is_NG"]]
                        fig4.add_trace(
                            go.Scatter(
                                x=df4["Bead"],
                                y=df4["Z_High_Exceed"],
                                mode="markers",
                                marker=dict(size=8, color=colors4),
                                text=df4["CSV_File"],
                                hovertemplate="CSV: %{text}<br>Bead: %{x}<br>Z_High_Exceed: %{y}<extra></extra>"
                            )
                        )

                        fig4.update_layout(
                            title="Z_High_Exceed vs Bead",
                            xaxis_title="Bead Number",
                            yaxis_title="Z_High_Exceed"
                        )
                        st.plotly_chart(fig4, use_container_width=True)

                else:
                    st.info("No suspected NOK beads found with current thresholds.")

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

            # If there is no data for this bead in detailed tabs, just warn in each tab.
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
                            title=f"Raw Signal {signal_col} • Bead #{selected_bead} • Recipe: Norm[{global_norm_lower},{global_norm_upper}] Z-score[-{global_z_lower},{global_z_upper}] Step[{global_step_interval}]", #f"Analysis for Signal {signal_col} • Raw Signal (Bead {selected_bead}, TEST, colored by global thresholds)",
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
                        title_suffix=f"• Smoothed Signal {signal_col} • Bead #{selected_bead}" #f"(Smoothed, Bead {selected_bead})"
                    )
                    if fig_norm_sg is not None:
                        plot_top_signals(
                            ref_transformed_sg,
                            test_transformed_sg,
                            status_sg,
                            title=f"Smoothed Signal {signal_col} • Bead #{selected_bead} • Recipe: Norm[{global_norm_lower},{global_norm_upper}] Z-score[-{global_z_lower},{global_z_upper}] Step[{global_step_interval}]", #f"Analysis for Signal {signal_col} • Smoothed Signal (Bead {selected_bead}, TEST, colored by global thresholds)",
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
                        title_suffix=f"• Low-pass Signal {signal_col} • Bead #{selected_bead}" #f"(Low-pass, Bead {selected_bead})"
                    )
                    if fig_norm_lp is not None:
                        plot_top_signals(
                            ref_transformed_lp,
                            test_transformed_lp,
                            status_lp,
                            title=f"Low-pass Filtered Signal {signal_col} • Bead #{selected_bead} • Recipe: Norm[{global_norm_lower},{global_norm_upper}] Z-score[-{global_z_lower},{global_z_upper}] Step[{global_step_interval}]", #f"Analysis for Signal {signal_col} • Low-pass Filtered Signal (Bead {selected_bead}, TEST, colored by global thresholds)",
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
                        title_suffix=f"• Curve Fit Signal {signal_col} • Bead #{selected_bead}" #f"(Curve Fit, Bead {selected_bead})"
                    )
                    if fig_norm_cf is not None:
                        plot_top_signals(
                            ref_transformed_cf,
                            test_transformed_cf,
                            status_cf,
                            title=f"Curve-fit Signal {signal_col} • Bead #{selected_bead} • Recipe: Norm[{global_norm_lower},{global_norm_upper}] Z-score[-{global_z_lower},{global_z_upper}] Step[{global_step_interval}]", #f"Analysis for Signal {signal_col} • Curve-fit Signal (Bead {selected_bead}, TEST, colored by global thresholds)",
                            y_label="Signal Value"
                        )
                        st.plotly_chart(fig_norm_cf, use_container_width=True)
