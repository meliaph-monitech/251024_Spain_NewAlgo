# =======================
# FULL PATCHED VERSION (SAFE)
# ONLY FIX: bead identity
# =======================

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
# --- Label Processing ---
# ============================================================
def normalize_label_value(v):
    if pd.isna(v):
        return ""
    return str(v).strip().upper()


def build_label_map(label_df: pd.DataFrame):
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


# ============================================================
# STEP 0: Upload ZIP (UNCHANGED UI)
# ============================================================
st.sidebar.header("Step 0: Upload Data Source")

uploaded_data_zip = st.sidebar.file_uploader(
    "Upload ZIP containing data folder and label CSV",
    type="zip"
)

if uploaded_data_zip:
    temp_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(uploaded_data_zip, "r") as zip_ref:
        zip_ref.extractall(temp_dir)

    all_items = os.listdir(temp_dir)

    data_folder = [f for f in all_items if os.path.isdir(os.path.join(temp_dir, f))][0]
    label_file = [f for f in all_items if f.endswith(".csv")][0]

    st.session_state.data_dir = os.path.join(temp_dir, data_folder)
    label_df = pd.read_csv(os.path.join(temp_dir, label_file))
    st.session_state.label_map = build_label_map(label_df)


# ============================================================
# STEP 1: Segment Files (UNCHANGED UI)
# ============================================================
if "data_dir" in st.session_state:

    sample_file = os.listdir(st.session_state.data_dir)[0]
    sample_df = pd.read_csv(os.path.join(st.session_state.data_dir, sample_file))

    st.session_state.seg_col = st.sidebar.selectbox("Column for Segmentation", sample_df.columns)
    st.session_state.seg_thresh = st.sidebar.number_input("Segmentation Threshold", value=1.0)

    if st.sidebar.button("Segment Files"):

        segmented_ok = {}
        segmented_test = {}

        for fname, bead_info in st.session_state.label_map.items():
            path = os.path.join(st.session_state.data_dir, fname)
            if not os.path.exists(path):
                continue

            df = pd.read_csv(path)
            bead_ranges = segment_beads(df, st.session_state.seg_col, st.session_state.seg_thresh)

            for new_idx, meta in bead_info.items():
                orig_idx = meta["orig_idx"]
                if orig_idx > len(bead_ranges):
                    continue

                start, end = bead_ranges[orig_idx - 1]
                bead_df = df.iloc[start:end+1].reset_index(drop=True)

                if meta["label"] == "OK":
                    segmented_ok.setdefault(fname, {})[new_idx] = bead_df
                else:
                    segmented_test.setdefault(fname, {})[new_idx] = bead_df

        st.session_state.segmented_ok = segmented_ok
        st.session_state.segmented_test = segmented_test

        st.success("Segmented!")


# ============================================================
# Helper (UNCHANGED)
# ============================================================
def compute_transformed_signals(observations):
    return [{"csv": o["csv"], "transformed": np.asarray(o["data"])} for o in observations]


# ============================================================
# STEP 3: Analysis + DataViz (UNCHANGED STRUCTURE)
# ONLY FIX: csv identity
# ============================================================
if st.session_state.segmented_ok and st.session_state.segmented_test:

    segmented_ok = st.session_state.segmented_ok
    segmented_test = st.session_state.segmented_test

    ok_files = sorted(segmented_ok.keys())
    test_files = sorted(segmented_test.keys())

    bead_ok = set()
    for _, beads in segmented_ok.items():
        bead_ok.update(beads.keys())

    bead_test = set()
    for _, beads in segmented_test.items():
        bead_test.update(beads.keys())

    bead_options = sorted(bead_ok.intersection(bead_test))

    selected_bead = st.sidebar.selectbox("Select Bead Number", bead_options)

    def build_observations_for_column(col_name):
        ref_obs = []
        for fname in ok_files:
            beads = segmented_ok[fname]
            if selected_bead in beads:
                bead_df = beads[selected_bead]
                if col_name in bead_df.columns:
                    data = bead_df[col_name].reset_index(drop=True)
                    ref_obs.append({
                        "csv": f"{fname}__B{selected_bead}",
                        "data": data
                    })

        test_obs = []
        for fname in test_files:
            beads = segmented_test[fname]
            if selected_bead in beads:
                bead_df = beads[selected_bead]
                if col_name in bead_df.columns:
                    data = bead_df[col_name].reset_index(drop=True)
                    test_obs.append({
                        "csv": f"{fname}__B{selected_bead}",
                        "data": data
                    })

        return ref_obs, test_obs

    example_bead_df = next(iter(next(iter(segmented_ok.values())).values()))
    signal_cols = get_channel_columns(example_bead_df)

    col0 = signal_cols[0]

    ref_obs, test_obs = build_observations_for_column(col0)

    ref_t = compute_transformed_signals(ref_obs)
    test_t = compute_transformed_signals(test_obs)

    fig = go.Figure()

    for obs in ref_t:
        fig.add_trace(go.Scatter(y=obs["transformed"], name=obs["csv"], line=dict(color="gray")))

    for obs in test_t:
        fig.add_trace(go.Scatter(y=obs["transformed"], name=obs["csv"], line=dict(color="red")))

    st.plotly_chart(fig, use_container_width=True)
