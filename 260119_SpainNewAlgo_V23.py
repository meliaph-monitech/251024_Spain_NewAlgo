import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
            confirmed = (start, end)
            start_indices.append(confirmed[0])
            end_indices.append(confirmed[1])
        else:
            i += 1
    return list(zip(start_indices, end_indices))

def aggregate_for_step(x, y, interval):
    if interval <= 0:
        interval = 1
    x = np.asarray(x)
    y = np.asarray(y)
    agg_x = x[::interval]
    agg_y = [np.mean(y[i:i+interval]) for i in range(0, len(y), interval)]
    agg_x = agg_x[:len(agg_y)]
    return agg_x, np.array(agg_y)

def short_label(csv_name: str) -> str:
    base = os.path.splitext(os.path.basename(csv_name))[0]
    return base[:6] if len(base) > 6 else base

def get_channel_columns(bead_df: pd.DataFrame):
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

# Versioning to invalidate caches when data changes
if "data_version" not in st.session_state:
    st.session_state.data_version = 0

# Cache for summary results
if "summary_cache" not in st.session_state:
    st.session_state.summary_cache = {
        "key": None,
        "df_summary": None,
    }

# Persisted "applied" thresholds so bead select doesn't change them until user submits
if "applied_params" not in st.session_state:
    st.session_state.applied_params = {
        "global_norm_lower": -1.0,
        "global_norm_upper": 4.0,
        "global_z_lower": 6.0,
        "global_z_upper": 40.0,
        "global_step_interval": 20,
    }

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

        # Invalidate summary cache when data changes
        st.session_state.data_version += 1
        st.session_state.summary_cache["key"] = None
        st.session_state.summary_cache["df_summary"] = None

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

            # Invalidate summary cache when data changes
            st.session_state.data_version += 1
            st.session_state.summary_cache["key"] = None
            st.session_state.summary_cache["df_summary"] = None

            st.success("✅ TEST beads segmented and stored.")

# ============================================================
# Helper: Transform Signals (RAW ONLY)
# ============================================================
def compute_transformed_signals(observations, mode="raw", **params):
    transformed_obs = []
    for obs in observations:
        y = np.asarray(obs["data"]).astype(float)
        transformed_obs.append({**obs, "transformed": y})
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

    min_steps = min(
        min(arr.shape[0] for arr in ok_step_arrays),
        min(arr.shape[0] for arr in test_step_arrays),
    )

    if min_steps == 0:
        st.warning("Step data are empty after aggregation.")
        return None, {}, {}

    ok_matrix = np.vstack([arr[:min_steps] for arr in ok_step_arrays])

    mu = np.median(ok_matrix, axis=0)
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

    # plot OK refs
    for meta in ok_step_meta:
        step_y_ok = meta["step_y"][:min_steps]
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

    # plot TEST and compute metrics
    for meta in test_step_meta:
        step_y = meta["step_y"][:min_steps]
        csv_name = meta["csv"]

        norm_vals = (step_y - min_ok) / denom
        z_vals = (step_y - mu) / sigma

        mask_norm_low = norm_vals < norm_lower
        mask_norm_high = norm_vals > norm_upper
        mask_z_low = z_vals < -z_lower
        mask_z_high = z_vals > z_upper

        norm_low_exceed = norm_vals[mask_norm_low].min() if mask_norm_low.any() else np.nan
        norm_high_exceed = norm_vals[mask_norm_high].max() if mask_norm_high.any() else np.nan
        z_low_exceed = z_vals[mask_z_low].min() if mask_z_low.any() else np.nan
        z_high_exceed = z_vals[mask_z_high].max() if mask_z_high.any() else np.nan

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

        fig.add_trace(
            go.Scatter(
                x=step_indices, y=norm_vals, mode="lines",
                name=f"{name} (TEST)",
                line=dict(color=color, width=width),
                legendgroup=name, showlegend=True
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=step_indices, y=z_vals, mode="lines",
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
# Helper: Global scatter plots (Summary Tab)
# ============================================================
def plot_global_metric_scatter(df_summary: pd.DataFrame, metric_col: str, title: str):
    if df_summary is None or df_summary.empty or metric_col not in df_summary.columns:
        st.info(f"No data to plot for {metric_col}.")
        return

    dfp = df_summary.copy()
    dfp[metric_col] = pd.to_numeric(dfp[metric_col], errors="coerce")
    dfp["Bead"] = pd.to_numeric(dfp["Bead"], errors="coerce")

    dfp = dfp[np.isfinite(dfp["Bead"].to_numpy(dtype=float, na_value=np.nan))]
    dfp = dfp[np.isfinite(dfp[metric_col].to_numpy(dtype=float, na_value=np.nan))]

    if dfp.empty:
        st.info(f"No valid values to plot for {metric_col}.")
        return

    col_names = sorted(dfp["SignalColumn"].astype(str).unique())
    palette = ["#1f77b4", "#ff7f0e"]
    color_map = {name: palette[i % len(palette)] for i, name in enumerate(col_names)}

    fig = go.Figure()

    for col_name in col_names:
        dcol = dfp[dfp["SignalColumn"].astype(str) == col_name]

        custom = np.stack(
            [
                dcol["CSV_File"].astype(str).to_numpy(),
                dcol["SignalColumn"].astype(str).to_numpy(),
                dcol["Status"].astype(str).to_numpy(),
            ],
            axis=1
        )

        hovertemplate = (
            "CSV: %{customdata[0]}<br>"
            "Bead: %{x}<br>"
            "Column: %{customdata[1]}<br>"
            "Status: %{customdata[2]}<br>"
            + metric_col + ": %{y}<extra></extra>"
        )

        fig.add_trace(
            go.Scatter(
                x=dcol["Bead"],
                y=dcol[metric_col],
                mode="markers",
                name=str(col_name),
                marker=dict(size=8, color=color_map.get(col_name, "#888888"), opacity=0.85),
                customdata=custom,
                hovertemplate=hovertemplate,
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Bead Number",
        yaxis_title=metric_col,
        legend=dict(orientation="h"),
        margin=dict(l=40, r=20, t=60, b=40),
    )

    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# STEP 3: Analysis (Requires both OK & TEST segmented)
# ============================================================
if st.session_state.segmented_ok and st.session_state.segmented_test:
    segmented_ok = st.session_state.segmented_ok
    segmented_test = st.session_state.segmented_test

    ok_files = sorted(segmented_ok.keys())
    test_files = sorted(segmented_test.keys())

    # Bead options
    bead_ok = set()
    for _, beads in segmented_ok.items():
        bead_ok.update(beads.keys())
    bead_test = set()
    for _, beads in segmented_test.items():
        bead_test.update(beads.keys())
    bead_options = sorted(bead_ok.intersection(bead_test))

    # ---- Step 3 uses a FORM so bead selection won't "apply" changes unless submitted
    st.sidebar.header("Step 3: Global Thresholds & Step Interval")

    with st.sidebar.form("global_params_form", clear_on_submit=False):
        p = st.session_state.applied_params

        global_norm_lower = st.number_input(
            "Global Lower Threshold for 0–1 Normalization (flag LOW if below)",
            min_value=-5.0, max_value=5.0, value=float(p["global_norm_lower"]), step=0.05
        )
        global_norm_upper = st.number_input(
            "Global Upper Threshold for 0–1 Normalization (flag HIGH if above)",
            min_value=-5.0, max_value=10.0, value=float(p["global_norm_upper"]), step=0.05
        )
        global_z_lower = st.number_input(
            "Global Z-score Threshold (flag LOW if below -T)",
            min_value=0.5, max_value=10.0, value=float(p["global_z_lower"]), step=0.5
        )
        global_z_upper = st.number_input(
            "Global Z-score Threshold (flag HIGH if above +T)",
            min_value=0.5, max_value=50.0, value=float(p["global_z_upper"]), step=0.25
        )
        global_step_interval = st.slider(
            "Global Step Interval (points)",
            min_value=10, max_value=500, value=int(p["global_step_interval"]), step=10
        )

        apply_btn = st.form_submit_button("Apply / Recompute Summary")

    # Only update applied params & invalidate cache when user presses the button
    if apply_btn:
        st.session_state.applied_params = {
            "global_norm_lower": float(global_norm_lower),
            "global_norm_upper": float(global_norm_upper),
            "global_z_lower": float(global_z_lower),
            "global_z_upper": float(global_z_upper),
            "global_step_interval": int(global_step_interval),
        }
        st.session_state.summary_cache["key"] = None
        st.session_state.summary_cache["df_summary"] = None
        st.sidebar.success("Applied. Summary will use these parameters.")

    # Use applied params for all computations
    p = st.session_state.applied_params
    global_norm_lower = p["global_norm_lower"]
    global_norm_upper = p["global_norm_upper"]
    global_z_lower = p["global_z_lower"]
    global_z_upper = p["global_z_upper"]
    global_step_interval = p["global_step_interval"]

    st.sidebar.header("Step 4: Bead for Analysis")
    if not bead_options:
        st.warning("No common bead numbers found in both OK and TEST sets.")
    else:
        selected_bead = st.sidebar.selectbox("Select Bead Number", bead_options)

        # Infer columns from an example bead df
        example_bead_df = None
        for fname in ok_files:
            beads = segmented_ok[fname]
            if selected_bead in beads:
                example_bead_df = beads[selected_bead]
                break

        if example_bead_df is None:
            st.error("Selected bead not found in OK reference set.")
        else:
            signal_cols = get_channel_columns(example_bead_df)
            if len(signal_cols) < 2:
                st.error("Bead dataframe has fewer than 2 columns, cannot visualize two columns.")
            else:
                tabs = st.tabs(["Summary", "DataViz"])

                # ============================================================
                # Tab 0: Summary (cached)
                # ============================================================
                with tabs[0]:
                    st.subheader("Global Summary of Suspected NOK (All Beads, First Two Columns, Raw Only)")

                    summary_key = (
                        st.session_state.data_version,
                        float(global_norm_lower),
                        float(global_norm_upper),
                        float(global_z_lower),
                        float(global_z_upper),
                        int(global_step_interval),
                    )

                    if st.session_state.summary_cache["key"] == summary_key and st.session_state.summary_cache["df_summary"] is not None:
                        df_summary = st.session_state.summary_cache["df_summary"]
                    else:
                        rows = []
                        with st.spinner("Running global summary across both columns (raw only)..."):
                            for bead in bead_options:
                                bead_df_for_cols = None
                                for fname in ok_files:
                                    beads = segmented_ok[fname]
                                    if bead in beads:
                                        bead_df_for_cols = beads[bead]
                                        break
                                if bead_df_for_cols is None:
                                    continue

                                cols_local = get_channel_columns(bead_df_for_cols)
                                if len(cols_local) < 2:
                                    continue

                                for col_name in cols_local:
                                    ref_obs_bead = []
                                    for fname in ok_files:
                                        beads = segmented_ok[fname]
                                        if bead in beads:
                                            bead_df = beads[bead]
                                            if col_name in bead_df.columns:
                                                data = bead_df[col_name].reset_index(drop=True)
                                                ref_obs_bead.append({"csv": fname, "data": data})

                                    test_obs_bead = []
                                    for fname in test_files:
                                        beads = segmented_test[fname]
                                        if bead in beads:
                                            bead_df = beads[bead]
                                            if col_name in bead_df.columns:
                                                data = bead_df[col_name].reset_index(drop=True)
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
                                        title_suffix=f"(Summary • {col_name})"
                                    )

                                    for csv_name, status in status_map.items():
                                        if status == "ok":
                                            continue
                                        m = metrics_map.get(csv_name, {})
                                        rows.append({
                                            "CSV_File": csv_name,
                                            "Bead": bead,
                                            "SignalColumn": str(col_name),
                                            "Status": "LOW" if status == "low" else "HIGH",
                                            "SignalTransform": "Raw Signal",
                                            "Norm_Low_Exceed": m.get("Norm_Low_Exceed", np.nan),
                                            "Norm_High_Exceed": m.get("Norm_High_Exceed", np.nan),
                                            "Z_Low_Exceed": m.get("Z_Low_Exceed", np.nan),
                                            "Z_High_Exceed": m.get("Z_High_Exceed", np.nan),
                                        })

                        if rows:
                            df_summary = pd.DataFrame(rows).sort_values(
                                ["CSV_File", "Bead", "SignalColumn", "SignalTransform"]
                            ).reset_index(drop=True)
                            df_summary["Is_NG"] = df_summary["CSV_File"].str.contains("NG", case=False)
                        else:
                            df_summary = pd.DataFrame(columns=[
                                "CSV_File","Bead","SignalColumn","Status","SignalTransform",
                                "Norm_Low_Exceed","Norm_High_Exceed","Z_Low_Exceed","Z_High_Exceed","Is_NG"
                            ])

                        st.session_state.summary_cache["key"] = summary_key
                        st.session_state.summary_cache["df_summary"] = df_summary

                    if df_summary is None or df_summary.empty:
                        st.info("No suspected NOK beads found with current thresholds (both columns, raw only).")
                    else:
                        st.markdown("### Suspected NOK Table")
                        st.dataframe(df_summary, use_container_width=True)

                        st.markdown("### Global Severity Scatter (NOK-only dots)")
                        plot_global_metric_scatter(df_summary, "Norm_Low_Exceed", "Norm_Low_Exceed vs Bead (color = Column)")
                        plot_global_metric_scatter(df_summary, "Z_Low_Exceed", "Z_Low_Exceed vs Bead (color = Column)")
                        plot_global_metric_scatter(df_summary, "Norm_High_Exceed", "Norm_High_Exceed vs Bead (color = Column)")
                        plot_global_metric_scatter(df_summary, "Z_High_Exceed", "Z_High_Exceed vs Bead (color = Column)")

                # ============================================================
                # Tab 1: DataViz (recompute only for selected bead; fast)
                # ============================================================
                with tabs[1]:
                    st.subheader("Data Visualization (Selected Bead, Raw Only)")
                    st.caption(f"Selected Bead: #{selected_bead} | Columns: {signal_cols[0]} / {signal_cols[1]}")

                    def build_observations_for_column(col_name: str):
                        ref_obs = []
                        for fname in ok_files:
                            beads = segmented_ok[fname]
                            if selected_bead in beads:
                                bead_df = beads[selected_bead]
                                if col_name in bead_df.columns:
                                    data = bead_df[col_name].reset_index(drop=True)
                                    ref_obs.append({"csv": fname, "data": data})

                        test_obs = []
                        for fname in test_files:
                            beads = segmented_test[fname]
                            if selected_bead in beads:
                                bead_df = beads[selected_bead]
                                if col_name in bead_df.columns:
                                    data = bead_df[col_name].reset_index(drop=True)
                                    test_obs.append({"csv": fname, "data": data})

                        return ref_obs, test_obs

                    col0, col1 = signal_cols[0], signal_cols[1]

                    with st.expander(f"{col0}", expanded=True):
                        ref_obs, test_obs = build_observations_for_column(col0)
                        if not ref_obs or not test_obs:
                            st.warning(f"No data for this bead in OK or TEST set ({col0}).")
                        else:
                            ref_t = compute_transformed_signals(ref_obs, mode="raw")
                            test_t = compute_transformed_signals(test_obs, mode="raw")

                            fig_norm, status_map, _ = compute_step_normalization_and_flags(
                                ref_t, test_t,
                                step_interval=global_step_interval,
                                norm_lower=global_norm_lower, norm_upper=global_norm_upper,
                                z_lower=global_z_lower, z_upper=global_z_upper,
                                title_suffix=f"• {col0} • Bead #{selected_bead}"
                            )

                            if fig_norm is not None:
                                plot_top_signals(
                                    ref_t, test_t, status_map,
                                    title=(
                                        f"{col0} • Bead #{selected_bead} • "
                                        f"Recipe: Norm[{global_norm_lower},{global_norm_upper}] "
                                        f"Z-score[-{global_z_lower},{global_z_upper}] "
                                        f"Step[{global_step_interval}]"
                                    ),
                                    y_label="Signal Value"
                                )
                                st.plotly_chart(fig_norm, use_container_width=True)

                    with st.expander(f"{col1}", expanded=False):
                        ref_obs, test_obs = build_observations_for_column(col1)
                        if not ref_obs or not test_obs:
                            st.warning(f"No data for this bead in OK or TEST set ({col1}).")
                        else:
                            ref_t = compute_transformed_signals(ref_obs, mode="raw")
                            test_t = compute_transformed_signals(test_obs, mode="raw")

                            fig_norm, status_map, _ = compute_step_normalization_and_flags(
                                ref_t, test_t,
                                step_interval=global_step_interval,
                                norm_lower=global_norm_lower, norm_upper=global_norm_upper,
                                z_lower=global_z_lower, z_upper=global_z_upper,
                                title_suffix=f"• {col1} • Bead #{selected_bead}"
                            )

                            if fig_norm is not None:
                                plot_top_signals(
                                    ref_t, test_t, status_map,
                                    title=(
                                        f"{col1} • Bead #{selected_bead} • "
                                        f"Recipe: Norm[{global_norm_lower},{global_norm_upper}] "
                                        f"Z-score[-{global_z_lower},{global_z_upper}] "
                                        f"Step[{global_step_interval}]"
                                    ),
                                    y_label="Signal Value"
                                )
                                st.plotly_chart(fig_norm, use_container_width=True)
