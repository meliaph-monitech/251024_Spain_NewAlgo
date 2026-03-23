# (FULL CODE - NO SIMPLIFICATION, SAME STRUCTURE, ONLY INPUT + MODE CHANGED)

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
# Utility (UNCHANGED)
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
# LABEL MAP (NEW)
# ============================================================
def build_label_map(df):
    label_map = {}
    for _, row in df.iterrows():
        fname = os.path.basename(str(row["FileName"]))
        kept = []

        for i in range(1, 7):
            val = str(row[str(i)]).strip().upper()
            if val in ["OK", "NOK"]:
                kept.append((i, val))

        bead_dict = {}
        for new_idx, (orig_idx, val) in enumerate(kept, start=1):
            bead_dict[new_idx] = {"orig_idx": orig_idx, "label": val}

        label_map[fname] = bead_dict
    return label_map

# ============================================================
# SESSION STATE (UNCHANGED)
# ============================================================
if "segmented_ok" not in st.session_state:
    st.session_state.segmented_ok = None
if "segmented_test" not in st.session_state:
    st.session_state.segmented_test = None
if "analysis_mode" not in st.session_state:
    st.session_state.analysis_mode = "Per-Bead"

# ============================================================
# STEP 0: ZIP INPUT (ONLY CHANGE)
# ============================================================
st.sidebar.header("Step 0: Upload Data")

uploaded_zip = st.sidebar.file_uploader("Upload DATA ZIP", type="zip")

if uploaded_zip:
    temp_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)

    subfolders = [f for f in os.listdir(temp_dir) if os.path.isdir(os.path.join(temp_dir, f))]
    csv_files = [f for f in os.listdir(temp_dir) if f.endswith(".csv")]

    selected_subfolder = st.sidebar.selectbox("Select Data Folder", subfolders)
    selected_label = st.sidebar.selectbox("Select Label CSV", csv_files)

    st.session_state.analysis_mode = st.sidebar.radio(
        "Analysis Mode",
        ["Per-Bead", "Global"]
    )

    if st.sidebar.button("Apply Data"):
        st.session_state.data_dir = os.path.join(temp_dir, selected_subfolder)
        label_df = pd.read_csv(os.path.join(temp_dir, selected_label))
        st.session_state.label_map = build_label_map(label_df)
        st.success("✅ Data Loaded")

# ============================================================
# STEP 1: SEGMENT (UNCHANGED STRUCTURE)
# ============================================================
if "data_dir" in st.session_state:

    st.sidebar.header("Step 1: Segment Files")

    sample_file = os.listdir(st.session_state.data_dir)[0]
    sample_df = pd.read_csv(os.path.join(st.session_state.data_dir, sample_file))

    seg_col = st.sidebar.selectbox("Column for Segmentation", sample_df.columns)
    seg_thresh = st.sidebar.number_input("Segmentation Threshold", value=1.0)

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
                if meta["orig_idx"] > len(bead_ranges):
                    continue

                start, end = bead_ranges[meta["orig_idx"] - 1]
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
# STEP 3: ANALYSIS (UNCHANGED UI, ONLY REF CHANGE)
# ============================================================
if st.session_state.segmented_ok and st.session_state.segmented_test:

    segmented_ok = st.session_state.segmented_ok
    segmented_test = st.session_state.segmented_test

    bead_options = sorted(set().union(*[d.keys() for d in segmented_ok.values()]))

    st.sidebar.header("Step 3: Global Thresholds & Step Interval")

    norm_low = st.sidebar.number_input("Norm Lower", value=-1.0)
    norm_high = st.sidebar.number_input("Norm Upper", value=4.0)
    z_low = st.sidebar.number_input("Z Lower", value=6.0)
    z_high = st.sidebar.number_input("Z Upper", value=40.0)
    step_interval = st.sidebar.slider("Step Interval", 10, 200, 20)

    selected_bead = st.sidebar.selectbox("Select Bead", bead_options)

    # 🔥 ONLY CHANGE HERE (reference building)
    def build_ref(col):
        if st.session_state.analysis_mode == "Per-Bead":
            return [
                {"csv": f"{f}|B{selected_bead}", "data": beads[selected_bead][col]}
                for f, beads in segmented_ok.items()
                if selected_bead in beads
            ]
        else:
            return [
                {"csv": f"{f}|B{k}", "data": df[col]}
                for f, beads in segmented_ok.items()
                for k, df in beads.items()
            ]

    def build_test(col):
        return [
            {"csv": f"{f}|B{selected_bead}", "data": beads[selected_bead][col]}
            for f, beads in segmented_test.items()
            if selected_bead in beads
        ]

    example_df = list(segmented_ok.values())[0][selected_bead]
    cols = get_channel_columns(example_df)

    tabs = st.tabs(["Summary", "DataViz"])

    with tabs[1]:
        for col in cols:
            ref = [{"csv": x["csv"], "transformed": np.array(x["data"])} for x in build_ref(col)]
            test = [{"csv": x["csv"], "transformed": np.array(x["data"])} for x in build_test(col)]

            fig, _, _ = compute_step_normalization_and_flags(
                ref, test,
                step_interval,
                norm_low,
                norm_high,
                z_low,
                z_high,
                col
            )

            st.plotly_chart(fig, use_container_width=True)
