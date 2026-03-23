import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(layout="wide")

# ============================================================
# Utility
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
            start_indices.append(start)
            end_indices.append(end)
        else:
            i += 1
    return list(zip(start_indices, end_indices))


def aggregate_for_step(x, y, interval):
    if interval <= 0:
        interval = 1
    x = np.asarray(x)
    y = np.asarray(y)
    agg_x = x[::interval]
    agg_y = [np.mean(y[i:i + interval]) for i in range(0, len(y), interval)]
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


def normalize_cell_value(v):
    if pd.isna(v):
        return ""
    s = str(v).strip()
    return s


def is_zero_like(v: str) -> bool:
    return v in {"0", "0.0", ""}


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
    - keep only rows/cells marked OK or NOK
    - ignore 0 and Ignore
    - after filtering, reindex remaining valid beads from 1..N
    """
    label_map = {}

    bead_cols = [str(i) for i in range(1, 7) if str(i) in label_df.columns]

    for _, row in label_df.iterrows():
        fname = os.path.basename(str(row["FileName"]).strip())
        kept = []

        for col in bead_cols:
            cell = normalize_cell_value(row[col]).upper()
            orig_idx = int(col)

            if cell in {"OK", "NOK"}:
                kept.append((orig_idx, cell))
            elif cell == "IGNORE":
                continue
            elif is_zero_like(cell):
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


def compute_transformed_signals(observations, mode="raw", **params):
    transformed_obs = []
    for obs in observations:
        y = np.asarray(obs["data"]).astype(float)
        transformed_obs.append({**obs, "transformed": y})
    return transformed_obs


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


def build_obs_from_segmented(segmented_dict, col_name, bead=None):
    obs = []
    for fname in sorted(segmented_dict.keys()):
        beads = segmented_dict[fname]
        if bead is None:
            bead_keys = sorted(beads.keys())
        else:
            bead_keys = [bead] if bead in beads else []

        for bead_no in bead_keys:
            bead_df = beads[bead_no]
            if col_name in bead_df.columns:
                obs.append({
                    "csv": f"{fname} | B{bead_no}",
                    "data": bead_df[col_name].reset_index(drop=True)
                })
    return obs


def infer_example_bead_df(segmented_ok, segmented_test, selected_bead=None):
    for source in [segmented_ok, segmented_test]:
        for fname in sorted(source.keys()):
            beads = source[fname]
            if selected_bead is None:
                if len(beads) > 0:
                    return beads[sorted(beads.keys())[0]]
            else:
                if selected_bead in beads:
                    return beads[selected_bead]
    return None


# ============================================================
# Session State
# ============================================================
defaults = {
    "data_source_config": None,
    "label_df": None,
    "label_map": None,
    "source_applied": False,
    "segmented_ok": None,
    "segmented_test": None,
    "seg_col": None,
    "seg_thresh": None,
    "analysis_mode_applied": "Per-Bead",
    "data_version": 0,
    "summary_cache": {"key": None, "df_summary": None},
    "applied_params": {
        "global_norm_lower": -1.0,
        "global_norm_upper": 4.0,
        "global_z_lower": 6.0,
        "global_z_upper": 40.0,
        "global_step_interval": 20,
    },
    "viz_selection": {
        "selected_bead": None,
    }
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ============================================================
# STEP 0: Data Source Selection (checkpoint)
# ============================================================
st.sidebar.header("Step 0: Select Data Source")

default_main_dir = r"C:\Users\user\Downloads\MELIA\HYUNDAI MOBIS SPAIN\260303 MOBIS GAP Image\260323 Step-based Confirm\DATA_TO_CHECK"

with st.sidebar.form("data_source_form", clear_on_submit=False):
    main_dir_input = st.text_input("Main Folder Directory", value=default_main_dir)

    subfolder_options = []
    label_csv_options = []

    if os.path.isdir(main_dir_input):
        try:
            subfolder_options = sorted([
                x for x in os.listdir(main_dir_input)
                if os.path.isdir(os.path.join(main_dir_input, x))
            ])
            label_csv_options = sorted([
                x for x in os.listdir(main_dir_input)
                if x.lower().endswith(".csv")
            ])
        except Exception:
            subfolder_options = []
            label_csv_options = []

    selected_subfolder = st.selectbox(
        "Folder that contains the CSV files to analyze",
        subfolder_options if subfolder_options else ["<No subfolder found>"],
        index=0
    )

    selected_label_csv = st.selectbox(
        "CSV file that contains FileName / bead labels",
        label_csv_options if label_csv_options else ["<No CSV found>"],
        index=0
    )

    analysis_mode_choice = st.radio(
        "Analysis Mode",
        ["Per-Bead", "Global"],
        index=0,
        help="This changes only the calculation behind the analysis. Summary and visualization presentation remain the same."
    )

    apply_source_btn = st.form_submit_button("Apply Data Source / Analysis Mode")

if apply_source_btn:
    valid_main = os.path.isdir(main_dir_input)
    valid_sub = valid_main and selected_subfolder != "<No subfolder found>"
    valid_lbl = valid_main and selected_label_csv != "<No CSV found>"

    if not valid_main:
        st.sidebar.error("Main folder directory does not exist.")
    elif not valid_sub:
        st.sidebar.error("No valid subfolder selected.")
    elif not valid_lbl:
        st.sidebar.error("No valid label CSV selected.")
    else:
        label_path = os.path.join(main_dir_input, selected_label_csv)
        try:
            label_df = pd.read_csv(label_path)
        except Exception as e:
            st.sidebar.error(f"Failed to read label CSV: {e}")
            label_df = None

        required_cols = {"FileName"}
        if label_df is None:
            pass
        elif not required_cols.issubset(set(label_df.columns)):
            st.sidebar.error("Label CSV must contain at least the 'FileName' column.")
        else:
            st.session_state.data_source_config = {
                "main_dir": main_dir_input,
                "data_subfolder": selected_subfolder,
                "data_dir": os.path.join(main_dir_input, selected_subfolder),
                "label_csv": selected_label_csv,
                "label_path": label_path,
            }
            st.session_state.label_df = label_df.copy()
            st.session_state.label_map = build_label_map(label_df)
            st.session_state.analysis_mode_applied = analysis_mode_choice
            st.session_state.source_applied = True

            # invalidate segmentation + summary when source/mode changes
            st.session_state.segmented_ok = None
            st.session_state.segmented_test = None
            st.session_state.summary_cache["key"] = None
            st.session_state.summary_cache["df_summary"] = None
            st.session_state.viz_selection = {"selected_bead": None}

            st.sidebar.success("Applied. Data source and analysis mode are locked.")

# ============================================================
# STEP 1: Segment Files (checkpoint)
# ============================================================
if st.session_state.source_applied and st.session_state.data_source_config is not None:
    cfg = st.session_state.data_source_config
    data_dir = cfg["data_dir"]
    label_df = st.session_state.label_df
    label_map = st.session_state.label_map

    st.sidebar.header("Step 1: Segment Files")

    sample_csvs = []
    if os.path.isdir(data_dir):
        try:
            sample_csvs = sorted([x for x in os.listdir(data_dir) if x.lower().endswith(".csv")])
        except Exception:
            sample_csvs = []

    sample_df = None
    sample_err = None
    if sample_csvs:
        try:
            sample_df = pd.read_csv(os.path.join(data_dir, sample_csvs[0]))
        except Exception as e:
            sample_err = str(e)

    if sample_df is None:
        if sample_err:
            st.sidebar.error(f"Could not read sample CSV: {sample_err}")
        else:
            st.sidebar.error("No CSV files found in selected data folder.")
    else:
        seg_columns = sample_df.columns.tolist()

        st.session_state.seg_col = st.sidebar.selectbox(
            "Column for Segmentation",
            seg_columns,
            index=0 if st.session_state.seg_col is None or st.session_state.seg_col not in seg_columns else seg_columns.index(st.session_state.seg_col),
            key="seg_col_special_app"
        )
        st.session_state.seg_thresh = st.sidebar.number_input(
            "Segmentation Threshold",
            value=1.0 if st.session_state.seg_thresh is None else float(st.session_state.seg_thresh),
            key="seg_thresh_special_app"
        )

        segment_btn = st.sidebar.button("Segment Files")

        if segment_btn:
            segmented_ok = {}
            segmented_test = {}
            missing_files = []
            files_with_no_valid_bead = []
            files_with_short_segmentation = []

            for file_name, bead_info in label_map.items():
                csv_path = os.path.join(data_dir, file_name)

                if not os.path.isfile(csv_path):
                    missing_files.append(file_name)
                    continue

                try:
                    df = pd.read_csv(csv_path)
                except Exception:
                    continue

                if st.session_state.seg_col not in df.columns:
                    continue

                bead_ranges = segment_beads(
                    df,
                    st.session_state.seg_col,
                    st.session_state.seg_thresh
                )

                ok_beads = {}
                test_beads = {}

                valid_found = False
                short_found = False

                for new_idx, meta in bead_info.items():
                    orig_idx = meta["orig_idx"]
                    label = meta["label"]

                    if orig_idx > len(bead_ranges):
                        short_found = True
                        continue

                    start, end = bead_ranges[orig_idx - 1]
                    bead_df = df.iloc[start:end + 1].reset_index(drop=True)

                    if label == "OK":
                        ok_beads[new_idx] = bead_df
                        valid_found = True
                    elif label == "NOK":
                        test_beads[new_idx] = bead_df
                        valid_found = True

                if short_found:
                    files_with_short_segmentation.append(file_name)

                if not valid_found:
                    files_with_no_valid_bead.append(file_name)

                if ok_beads:
                    segmented_ok[file_name] = ok_beads
                if test_beads:
                    segmented_test[file_name] = test_beads

            st.session_state.segmented_ok = segmented_ok
            st.session_state.segmented_test = segmented_test

            st.session_state.data_version += 1
            st.session_state.summary_cache["key"] = None
            st.session_state.summary_cache["df_summary"] = None
            st.session_state.viz_selection = {"selected_bead": None}

            st.sidebar.success("✅ Files segmented and auto-sorted into OK / TEST.")

            if missing_files:
                st.warning(f"{len(missing_files)} file(s) listed in label CSV were not found in the selected data folder.")
            if files_with_short_segmentation:
                st.warning(f"{len(files_with_short_segmentation)} file(s) had fewer segmented beads than expected by the label CSV.")
            if files_with_no_valid_bead:
                st.warning(f"{len(files_with_no_valid_bead)} file(s) had no usable OK/NOK bead after filtering.")

# ============================================================
# STEP 3: Analysis
# ============================================================
if st.session_state.segmented_ok and st.session_state.segmented_test:
    segmented_ok = st.session_state.segmented_ok
    segmented_test = st.session_state.segmented_test
    analysis_mode = st.session_state.analysis_mode_applied

    ok_files = sorted(segmented_ok.keys())
    test_files = sorted(segmented_test.keys())

    bead_ok = set()
    for _, beads in segmented_ok.items():
        bead_ok.update(beads.keys())

    bead_test = set()
    for _, beads in segmented_test.items():
        bead_test.update(beads.keys())

    bead_options = sorted(bead_ok.intersection(bead_test))

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

    p = st.session_state.applied_params
    global_norm_lower = p["global_norm_lower"]
    global_norm_upper = p["global_norm_upper"]
    global_z_lower = p["global_z_lower"]
    global_z_upper = p["global_z_upper"]
    global_step_interval = p["global_step_interval"]

    st.sidebar.header("Step 4: Bead for Analysis / Visualization")

    if not bead_options:
        st.warning("No common bead numbers found in both OK and TEST sets.")
    else:
        with st.sidebar.form("viz_selection_form", clear_on_submit=False):
            viz_bead_default = (
                bead_options[0]
                if st.session_state.viz_selection["selected_bead"] not in bead_options
                else st.session_state.viz_selection["selected_bead"]
            )
            selected_bead_form = st.selectbox(
                "Select Bead Number",
                bead_options,
                index=bead_options.index(viz_bead_default)
            )
            apply_viz_btn = st.form_submit_button("Apply Bead Selection")

        if apply_viz_btn or st.session_state.viz_selection["selected_bead"] is None:
            st.session_state.viz_selection["selected_bead"] = selected_bead_form

        selected_bead = st.session_state.viz_selection["selected_bead"]

        example_bead_df = infer_example_bead_df(segmented_ok, segmented_test, selected_bead=selected_bead)

        if example_bead_df is None:
            st.error("Selected bead not found in segmented data.")
        else:
            signal_cols = get_channel_columns(example_bead_df)
            if len(signal_cols) < 2:
                st.error("Bead dataframe has fewer than 2 columns, cannot visualize two columns.")
            else:
                tabs = st.tabs(["Summary", "DataViz"])

                # ========================================================
                # Tab 0: Summary
                # ========================================================
                with tabs[0]:
                    st.subheader("Global Summary of Suspected NOK (All Beads, First Two Columns, Raw Only)")
                    st.caption(f"Applied Analysis Mode: {analysis_mode}")

                    summary_key = (
                        st.session_state.data_version,
                        analysis_mode,
                        float(global_norm_lower),
                        float(global_norm_upper),
                        float(global_z_lower),
                        float(global_z_upper),
                        int(global_step_interval),
                    )

                    if (
                        st.session_state.summary_cache["key"] == summary_key
                        and st.session_state.summary_cache["df_summary"] is not None
                    ):
                        df_summary = st.session_state.summary_cache["df_summary"]
                    else:
                        rows = []

                        with st.spinner("Running global summary across both columns (raw only)..."):
                            # global reference pool per column if needed
                            example_df_any = infer_example_bead_df(segmented_ok, segmented_test, selected_bead=None)
                            cols_global = get_channel_columns(example_df_any) if example_df_any is not None else []

                            global_ref_cache = {}
                            if len(cols_global) >= 2 and analysis_mode == "Global":
                                for col_name in cols_global:
                                    ref_obs_global = build_obs_from_segmented(segmented_ok, col_name, bead=None)
                                    global_ref_cache[col_name] = compute_transformed_signals(ref_obs_global, mode="raw")

                            for bead in bead_options:
                                bead_df_for_cols = infer_example_bead_df(segmented_ok, segmented_test, selected_bead=bead)
                                if bead_df_for_cols is None:
                                    continue

                                cols_local = get_channel_columns(bead_df_for_cols)
                                if len(cols_local) < 2:
                                    continue

                                for col_name in cols_local:
                                    if analysis_mode == "Per-Bead":
                                        ref_obs = build_obs_from_segmented(segmented_ok, col_name, bead=bead)
                                        ref_t = compute_transformed_signals(ref_obs, mode="raw")
                                    else:
                                        ref_t = global_ref_cache.get(col_name, [])

                                    test_obs = build_obs_from_segmented(segmented_test, col_name, bead=bead)
                                    test_t = compute_transformed_signals(test_obs, mode="raw")

                                    if not ref_t or not test_t:
                                        continue

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
                                        pure_csv_name = csv_name.split(" | B")[0] if " | B" in csv_name else csv_name
                                        rows.append({
                                            "CSV_File": pure_csv_name,
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
                                "CSV_File", "Bead", "SignalColumn", "Status", "SignalTransform",
                                "Norm_Low_Exceed", "Norm_High_Exceed", "Z_Low_Exceed", "Z_High_Exceed", "Is_NG"
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

                # ========================================================
                # Tab 1: DataViz
                # ========================================================
                with tabs[1]:
                    st.subheader("Data Visualization (Selected Bead, Raw Only)")
                    st.caption(
                        f"Selected Bead: #{selected_bead} | Columns: {signal_cols[0]} / {signal_cols[1]} | "
                        f"Applied Analysis Mode: {analysis_mode}"
                    )

                    def build_ref_and_test_for_column(col_name: str, bead_no: int):
                        if analysis_mode == "Per-Bead":
                            ref_obs = build_obs_from_segmented(segmented_ok, col_name, bead=bead_no)
                        else:
                            ref_obs = build_obs_from_segmented(segmented_ok, col_name, bead=None)

                        test_obs = build_obs_from_segmented(segmented_test, col_name, bead=bead_no)
                        return ref_obs, test_obs

                    col0, col1 = signal_cols[0], signal_cols[1]

                    with st.expander(f"{col0}", expanded=True):
                        ref_obs, test_obs = build_ref_and_test_for_column(col0, selected_bead)
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
                        ref_obs, test_obs = build_ref_and_test_for_column(col1, selected_bead)
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
