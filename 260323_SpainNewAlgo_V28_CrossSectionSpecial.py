import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import os
import tempfile
import plotly.graph_objects as go

st.set_page_config(layout="wide")

# ============================================================
# Utility Functions
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


def get_channel_columns(df):
    cols = df.columns.tolist()
    return cols[:2] if len(cols) >= 2 else []


# ============================================================
# Core Analysis Function (FIXED)
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
    ok_arrays = []
    for obs in ref_obs:
        y = obs["transformed"]
        x = np.arange(len(y))
        _, step_y = aggregate_for_step(x, y, step_interval)
        if len(step_y) > 0:
            ok_arrays.append(step_y)

    test_arrays = []
    for obs in test_obs:
        y = obs["transformed"]
        x = np.arange(len(y))
        _, step_y = aggregate_for_step(x, y, step_interval)
        if len(step_y) > 0:
            test_arrays.append(step_y)

    if not ok_arrays or not test_arrays:
        return None, {}, {}

    min_len = min(min(len(x) for x in ok_arrays), min(len(x) for x in test_arrays))

    ok_matrix = np.vstack([x[:min_len] for x in ok_arrays])

    mu = np.median(ok_matrix, axis=0)
    sigma = np.std(ok_matrix, axis=0)
    sigma[sigma < 1e-12] = 1e-12

    min_ok = ok_matrix.min(axis=0)
    max_ok = ok_matrix.max(axis=0)
    denom = max_ok - min_ok
    denom[denom < 1e-12] = 1e-12

    fig = go.Figure()
    status_map = {}

    for obs in test_obs:
        y = obs["transformed"]
        x = np.arange(len(y))
        _, step_y = aggregate_for_step(x, y, step_interval)
        step_y = step_y[:min_len]

        norm = (step_y - min_ok) / denom
        z = (step_y - mu) / sigma

        low = (norm < norm_lower).any() or (z < -z_lower).any()
        high = (norm > norm_upper).any() or (z > z_upper).any()

        if low:
            status = "low"
            color = "red"
        elif high:
            status = "high"
            color = "orange"
        else:
            status = "ok"
            color = "green"

        status_map[obs["csv"]] = status

        fig.add_trace(go.Scatter(
            y=norm,
            mode="lines",
            name=obs["csv"],
            line=dict(color=color)
        ))

    fig.update_layout(title=f"Step Analysis {title_suffix}")

    return fig, status_map, {}


# ============================================================
# Label Mapping
# ============================================================
def build_label_map(label_df):
    label_map = {}

    for _, row in label_df.iterrows():
        fname = os.path.basename(str(row["FileName"]).strip())
        kept = []

        for i in range(1, 7):
            val = str(row.get(str(i), "")).strip().upper()

            if val in ["OK", "NOK"]:
                kept.append((i, val))

        bead_info = {}
        for new_idx, (orig_idx, label) in enumerate(kept, start=1):
            bead_info[new_idx] = {
                "orig_idx": orig_idx,
                "label": label
            }

        label_map[fname] = bead_info

    return label_map


# ============================================================
# STEP 0: Upload ZIP
# ============================================================
st.sidebar.header("Step 0: Upload Data (ZIP)")

uploaded_zip = st.sidebar.file_uploader("Upload DATA ZIP", type="zip")

if uploaded_zip:
    temp_dir = tempfile.mkdtemp()

    with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)

    subfolders = [f for f in os.listdir(temp_dir) if os.path.isdir(os.path.join(temp_dir, f))]
    csv_files = [f for f in os.listdir(temp_dir) if f.endswith(".csv")]

    selected_subfolder = st.sidebar.selectbox("Select Data Folder", subfolders)
    selected_label = st.sidebar.selectbox("Select Label CSV", csv_files)

    analysis_mode = st.sidebar.radio("Analysis Mode", ["Per-Bead", "Global"])

    if st.sidebar.button("Apply Data"):

        data_dir = os.path.join(temp_dir, selected_subfolder)
        label_df = pd.read_csv(os.path.join(temp_dir, selected_label))
        label_map = build_label_map(label_df)

        st.session_state.data_dir = data_dir
        st.session_state.label_map = label_map
        st.session_state.analysis_mode = analysis_mode
        st.session_state.ready = True

        st.success("✅ Data Ready")

# ============================================================
# STEP 1: Segmentation
# ============================================================
if st.session_state.get("ready"):

    st.sidebar.header("Step 1: Segment")

    sample_file = os.listdir(st.session_state.data_dir)[0]
    sample_df = pd.read_csv(os.path.join(st.session_state.data_dir, sample_file))

    seg_col = st.sidebar.selectbox("Seg Column", sample_df.columns)
    seg_thresh = st.sidebar.number_input("Threshold", value=1.0)

    if st.sidebar.button("Segment Files"):

        segmented_ok = {}
        segmented_test = {}

        for fname, bead_info in st.session_state.label_map.items():

            path = os.path.join(st.session_state.data_dir, fname)
            if not os.path.exists(path):
                continue

            df = pd.read_csv(path)
            bead_ranges = segment_beads(df, seg_col, seg_thresh)

            ok_dict = {}
            test_dict = {}

            for new_idx, meta in bead_info.items():
                orig_idx = meta["orig_idx"]

                if orig_idx > len(bead_ranges):
                    continue

                start, end = bead_ranges[orig_idx - 1]
                bead_df = df.iloc[start:end+1].reset_index(drop=True)

                if meta["label"] == "OK":
                    ok_dict[new_idx] = bead_df
                else:
                    test_dict[new_idx] = bead_df

            if ok_dict:
                segmented_ok[fname] = ok_dict
            if test_dict:
                segmented_test[fname] = test_dict

        st.session_state.segmented_ok = segmented_ok
        st.session_state.segmented_test = segmented_test

        st.success("✅ Segmentation Done")

# ============================================================
# STEP 2: Analysis
# ============================================================
if st.session_state.get("segmented_ok"):

    segmented_ok = st.session_state.segmented_ok
    segmented_test = st.session_state.segmented_test

    bead_options = sorted(set().union(*[d.keys() for d in segmented_ok.values()]))

    st.sidebar.header("Step 2: Parameters")

    norm_low = st.sidebar.number_input("Norm Lower", value=-1.0)
    norm_high = st.sidebar.number_input("Norm Upper", value=4.0)
    z_low = st.sidebar.number_input("Z Lower", value=6.0)
    z_high = st.sidebar.number_input("Z Upper", value=40.0)
    step_interval = st.sidebar.slider("Step Interval", 10, 200, 20)

    selected_bead = st.sidebar.selectbox("Select Bead", bead_options)

    def build_ref(col):
        if st.session_state.analysis_mode == "Per-Bead":
            return [
                {"csv": f"{f}|B{selected_bead}", "transformed": np.array(beads[selected_bead][col])}
                for f, beads in segmented_ok.items()
                if selected_bead in beads
            ]
        else:
            return [
                {"csv": f"{f}|B{k}", "transformed": np.array(df[col])}
                for f, beads in segmented_ok.items()
                for k, df in beads.items()
            ]

    def build_test(col):
        return [
            {"csv": f"{f}|B{selected_bead}", "transformed": np.array(beads[selected_bead][col])}
            for f, beads in segmented_test.items()
            if selected_bead in beads
        ]

    example_df = list(segmented_ok.values())[0][selected_bead]
    cols = get_channel_columns(example_df)

    for col in cols:

        ref = build_ref(col)
        test = build_test(col)

        fig, _, _ = compute_step_normalization_and_flags(
            ref, test,
            step_interval,
            norm_low,
            norm_high,
            z_low,
            z_high,
            col
        )

        if fig:
            st.plotly_chart(fig, use_container_width=True)
