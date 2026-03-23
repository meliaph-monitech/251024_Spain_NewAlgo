import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import os
import tempfile
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(layout="wide")

# ============================================================
# --- Utility: Bead Segmentation ---
# ============================================================
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


# ============================================================
# --- Helper: label / file discovery ---
# ============================================================
def discover_relative_subfolders(root_dir: str):
    rel_dirs = []
    for cur_root, dirnames, _ in os.walk(root_dir):
        for d in dirnames:
            abs_path = os.path.join(cur_root, d)
            rel_path = os.path.relpath(abs_path, root_dir)
            rel_dirs.append(rel_path)
    rel_dirs = sorted(set(rel_dirs))
    return rel_dirs


def discover_relative_csv_files(root_dir: str):
    rel_files = []
    for cur_root, _, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith(".csv"):
                abs_path = os.path.join(cur_root, f)
                rel_path = os.path.relpath(abs_path, root_dir)
                rel_files.append(rel_path)
    rel_files = sorted(set(rel_files))
    return rel_files


def normalize_label_value(v):
    if pd.isna(v):
        return ""
    return str(v).strip().upper()


def build_label_map(label_df: pd.DataFrame):
    """
    Returns:
    {
        "file.csv": {
            1: {"orig_idx": 2, "label": "OK"},
            2: {"orig_idx": 3, "label": "NOK"},
            ...
        }
    }

    Rule:
    - original columns 1..6 are original bead indices
    - keep only cells marked OK or NOK
    - ignore 0 / Ignore / blank
    - after filtering, reindex remaining valid beads from 1..N
    """
    label_map = {}

    bead_cols = [str(i) for i in range(1, 7) if str(i) in label_df.columns]

    for _, row in label_df.iterrows():
        fname = os.path.basename(str(row["FileName"]).strip())
        kept = []

        for col in bead_cols:
            cell = normalize_label_value(row[col])
            orig_idx = int(col)

            if cell in {"OK", "NOK"}:
                kept.append((orig_idx, cell))
            elif cell in {"IGNORE", "0", "0.0", ""}:
                continue
            else:
                continue

        bead_info = {}
        for new_idx, (orig_idx, label) in enumerate(kept, start=1):
            bead_info[new_idx] = {
                "orig_idx": orig_idx,
                "label": label
            }

        label_map[fname] = bead_info

    return label_map


# ============================================================
# --- Session State ---
# ============================================================
if "segmented_ok" not in st.session_state:
    st.session_state.segmented_ok = None
if "segmented_test" not in st.session_state:
    st.session_state.segmented_test = None
if "seg_col" not in st.session_state:
    st.session_state.seg_col = None
if "seg_thresh" not in st.session_state:
    st.session_state.seg_thresh = None

if "analysis_mode" not in st.session_state:
    st.session_state.analysis_mode = "Per-Bead"

if "data_root_dir" not in st.session_state:
    st.session_state.data_root_dir = None
if "data_dir" not in st.session_state:
    st.session_state.data_dir = None
if "label_path" not in st.session_state:
    st.session_state.label_path = None
if "label_map" not in st.session_state:
    st.session_state.label_map = None
if "source_applied" not in st.session_state:
    st.session_state.source_applied = False

if "selected_bead_applied" not in st.session_state:
    st.session_state.selected_bead_applied = None

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
# STEP 0: Upload Data ZIP + choose folder / label file / mode
# ============================================================
st.sidebar.header("Step 0: Upload Data Source")

uploaded_data_zip = st.sidebar.file_uploader(
    "Upload ZIP containing data folder and label CSV",
    type="zip",
    key="data_zip_special"
)

if uploaded_data_zip is not None:
    if "extracted_zip_name" not in st.session_state or st.session_state.get("extracted_zip_name") != uploaded_data_zip.name:
        temp_dir = tempfile.mkdtemp(prefix="step_based_app_")
        with zipfile.ZipFile(uploaded_data_zip, "r") as zip_ref:
            zip_ref.extractall(temp_dir)
        st.session_state.extracted_zip_dir = temp_dir
        st.session_state.extracted_zip_name = uploaded_data_zip.name

        # reset previously applied source if a new zip is uploaded
        st.session_state.source_applied = False
        st.session_state.data_root_dir = None
        st.session_state.data_dir = None
        st.session_state.label_path = None
        st.session_state.label_map = None
        st.session_state.segmented_ok = None
        st.session_state.segmented_test = None
        st.session_state.selected_bead_applied = None
        st.session_state.summary_cache["key"] = None
        st.session_state.summary_cache["df_summary"] = None

    extracted_root = st.session_state.extracted_zip_dir

    folder_options = discover_relative_subfolders(extracted_root)
    label_options = discover_relative_csv_files(extracted_root)

    with st.sidebar.form("step0_source_form", clear_on_submit=False):
        selected_folder_rel = st.selectbox(
            "Folder that contains the CSV files to analyze",
            folder_options if folder_options else ["<no folder found>"]
        )

        selected_label_rel = st.selectbox(
            "CSV file that contains FileName / bead label table",
            label_options if label_options else ["<no csv found>"]
        )

        selected_analysis_mode = st.radio(
            "Analysis Mode",
            ["Per-Bead", "Global"],
            index=0 if st.session_state.analysis_mode == "Per-Bead" else 1,
            help="This changes only the calculation behind the analysis. Summary table and all visualizations remain presented per bead."
        )

        apply_source_btn = st.form_submit_button("Apply Data Source")

    if apply_source_btn:
        if selected_folder_rel == "<no folder found>" or selected_label_rel == "<no csv found>":
            st.sidebar.error("Could not find a valid data folder or label CSV inside the uploaded ZIP.")
        else:
            label_path_abs = os.path.join(extracted_root, selected_label_rel)
            data_dir_abs = os.path.join(extracted_root, selected_folder_rel)

            try:
                label_df = pd.read_csv(label_path_abs)
            except Exception as e:
                st.sidebar.error(f"Failed to read label CSV: {e}")
                label_df = None

            required_cols = {"FileName"}
            missing_required = []
            if label_df is not None:
                missing_required = sorted(list(required_cols - set(label_df.columns)))

            if label_df is None:
                pass
            elif missing_required:
                st.sidebar.error(f"Label CSV is missing required column(s): {missing_required}")
            else:
                st.session_state.data_root_dir = extracted_root
                st.session_state.data_dir = data_dir_abs
                st.session_state.label_path = label_path_abs
                st.session_state.label_map = build_label_map(label_df)
                st.session_state.analysis_mode = selected_analysis_mode
                st.session_state.source_applied = True

                # reset downstream checkpoints
                st.session_state.segmented_ok = None
                st.session_state.segmented_test = None
                st.session_state.selected_bead_applied = None
                st.session_state.summary_cache["key"] = None
                st.session_state.summary_cache["df_summary"] = None

                st.sidebar.success("Applied. Data source and analysis mode are locked.")


# ============================================================
# STEP 1: Segment Files
# ============================================================
if st.session_state.source_applied and st.session_state.data_dir is not None:
    st.sidebar.header("Step 1: Segment Files")

    sample_csvs = sorted([
        f for f in os.listdir(st.session_state.data_dir)
        if f.lower().endswith(".csv")
    ]) if os.path.isdir(st.session_state.data_dir) else []

    if not sample_csvs:
        st.sidebar.error("No CSV files found in selected data folder.")
    else:
        sample_file = sample_csvs[0]
        sample_df = pd.read_csv(os.path.join(st.session_state.data_dir, sample_file))

        seg_cols = sample_df.columns.tolist()
        seg_default_idx = 0
        if st.session_state.seg_col in seg_cols:
            seg_default_idx = seg_cols.index(st.session_state.seg_col)

        st.session_state.seg_col = st.sidebar.selectbox(
            "Column for Segmentation",
            seg_cols,
            index=seg_default_idx,
            key="seg_col_special"
        )
        st.session_state.seg_thresh = st.sidebar.number_input(
            "Segmentation Threshold",
            value=1.0 if st.session_state.seg_thresh is None else float(st.session_state.seg_thresh),
            key="seg_thresh_special"
        )

        segment_btn = st.sidebar.button("Segment Files")

        if segment_btn:
            segmented_ok = {}
            segmented_test = {}

            missing_files = []
            insufficient_bead_rows = []

            for fname, bead_info in st.session_state.label_map.items():
                path = os.path.join(st.session_state.data_dir, fname)

                if not os.path.exists(path):
                    missing_files.append(fname)
                    continue

                try:
                    df = pd.read_csv(path)
                except Exception:
                    continue

                if st.session_state.seg_col not in df.columns:
                    continue

                bead_ranges = segment_beads(
                    df,
                    st.session_state.seg_col,
                    st.session_state.seg_thresh
                )

                ok_dict = {}
                test_dict = {}

                for new_idx, meta in bead_info.items():
                    orig_idx = meta["orig_idx"]

                    if orig_idx > len(bead_ranges):
                        insufficient_bead_rows.append((fname, orig_idx, len(bead_ranges)))
                        continue

                    start, end = bead_ranges[orig_idx - 1]
                    bead_df = df.iloc[start:end+1].reset_index(drop=True)

                    if meta["label"] == "OK":
                        ok_dict[new_idx] = bead_df
                    elif meta["label"] == "NOK":
                        test_dict[new_idx] = bead_df

                if ok_dict:
                    segmented_ok[fname] = ok_dict
                if test_dict:
                    segmented_test[fname] = test_dict

            st.session_state.segmented_ok = segmented_ok
            st.session_state.segmented_test = segmented_test

            # invalidate summary cache when data changes
            st.session_state.data_version += 1
            st.session_state.summary_cache["key"] = None
            st.session_state.summary_cache["df_summary"] = None
            st.session_state.selected_bead_applied = None

            st.success("✅ Files segmented and auto-sorted into OK / TEST.")

            if missing_files:
                st.warning(f"{len(missing_files)} file(s) listed in label CSV were not found in the selected data folder.")
            if insufficient_bead_rows:
                st.warning(f"{len(insufficient_bead_rows)} bead mapping(s) could not be extracted because segmented bead count was smaller than expected.")


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
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
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
    analysis_mode = st.session_state.analysis_mode

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
        with st.sidebar.form("bead_selection_form", clear_on_submit=False):
            default_bead = bead_options[0] if st.session_state.selected_bead_applied not in bead_options else st.session_state.selected_bead_applied
            selected_bead_form = st.selectbox("Select Bead Number", bead_options, index=bead_options.index(default_bead))
            bead_apply_btn = st.form_submit_button("Apply Bead Selection")

        if bead_apply_btn or st.session_state.selected_bead_applied is None:
            st.session_state.selected_bead_applied = selected_bead_form

        selected_bead = st.session_state.selected_bead_applied

        # Infer columns from an example bead df
        example_bead_df = None
        for fname in ok_files:
            beads = segmented_ok[fname]
            if selected_bead in beads:
                example_bead_df = beads[selected_bead]
                break

        if example_bead_df is None:
            for fname in test_files:
                beads = segmented_test[fname]
                if selected_bead in beads:
                    example_bead_df = beads[selected_bead]
                    break

        if example_bead_df is None:
            st.error("Selected bead not found in segmented data.")
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
                    st.caption(f"Analysis Mode: {analysis_mode}")

                    summary_key = (
                        st.session_state.data_version,
                        str(analysis_mode),
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
                            # prebuild global ref cache if mode = Global
                            global_ref_cache = {}

                            # determine two columns from any available bead
                            bead_df_for_global_cols = None
                            for fname in ok_files:
                                if segmented_ok[fname]:
                                    bead_df_for_global_cols = next(iter(segmented_ok[fname].values()))
                                    break
                            if bead_df_for_global_cols is None:
                                for fname in test_files:
                                    if segmented_test[fname]:
                                        bead_df_for_global_cols = next(iter(segmented_test[fname].values()))
                                        break

                            cols_global = get_channel_columns(bead_df_for_global_cols) if bead_df_for_global_cols is not None else []

                            if analysis_mode == "Global":
                                for col_name in cols_global:
                                    ref_obs_global = []
                                    for fname in ok_files:
                                        beads = segmented_ok[fname]
                                        for bead_no, bead_df in beads.items():
                                            if col_name in bead_df.columns:
                                                data = bead_df[col_name].reset_index(drop=True)
                                                ref_obs_global.append({"csv": fname, "data": data})
                                    global_ref_cache[col_name] = ref_obs_global

                            for bead in bead_options:
                                bead_df_for_cols = None
                                for fname in ok_files:
                                    beads = segmented_ok[fname]
                                    if bead in beads:
                                        bead_df_for_cols = beads[bead]
                                        break
                                if bead_df_for_cols is None:
                                    for fname in test_files:
                                        beads = segmented_test[fname]
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

                                    if analysis_mode == "Per-Bead":
                                        ref_obs = ref_obs_bead
                                    else:
                                        ref_obs = global_ref_cache.get(col_name, [])

                                    test_obs_bead = []
                                    for fname in test_files:
                                        beads = segmented_test[fname]
                                        if bead in beads:
                                            bead_df = beads[bead]
                                            if col_name in bead_df.columns:
                                                data = bead_df[col_name].reset_index(drop=True)
                                                test_obs_bead.append({"csv": fname, "data": data})

                                    if not ref_obs or not test_obs_bead:
                                        continue

                                    ref_t = compute_transformed_signals(ref_obs, mode="raw")
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
                            df_summary["Is_NG"] = df_summary["CSV_File"].str.contains("NG", case=False, na=False)
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
                    st.caption(
                        f"Selected Bead: #{selected_bead} | Columns: {signal_cols[0]} / {signal_cols[1]} | Analysis Mode: {analysis_mode}"
                    )

                    # prebuild global refs for dataviz if needed
                    global_ref_dv_cache = {}
                    if analysis_mode == "Global":
                        for col_name in signal_cols:
                            ref_obs_global = []
                            for fname in ok_files:
                                beads = segmented_ok[fname]
                                for bead_no, bead_df in beads.items():
                                    if col_name in bead_df.columns:
                                        data = bead_df[col_name].reset_index(drop=True)
                                        ref_obs_global.append({"csv": fname, "data": data})
                            global_ref_dv_cache[col_name] = ref_obs_global

                    def build_observations_for_column(col_name: str):
                        if analysis_mode == "Per-Bead":
                            ref_obs = []
                            for fname in ok_files:
                                beads = segmented_ok[fname]
                                if selected_bead in beads:
                                    bead_df = beads[selected_bead]
                                    if col_name in bead_df.columns:
                                        data = bead_df[col_name].reset_index(drop=True)
                                        ref_obs.append({"csv": fname, "data": data})
                        else:
                            ref_obs = global_ref_dv_cache.get(col_name, [])

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
