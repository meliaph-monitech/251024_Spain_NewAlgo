# =======================
# IMPORTANT FIX INCLUDED:
# - Unique observation ID per (file, bead)
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
    return base[:10]


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
# Session State Init
# ============================================================
if "segmented_ok" not in st.session_state:
    st.session_state.segmented_ok = None
if "segmented_test" not in st.session_state:
    st.session_state.segmented_test = None
if "analysis_mode" not in st.session_state:
    st.session_state.analysis_mode = "Per-Bead"


# ============================================================
# STEP 0: Upload ZIP
# ============================================================
st.sidebar.header("Step 0: Upload Data Source")

uploaded_data_zip = st.sidebar.file_uploader("Upload ZIP", type="zip")

if uploaded_data_zip:
    temp_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(uploaded_data_zip, "r") as zip_ref:
        zip_ref.extractall(temp_dir)

    all_files = os.listdir(temp_dir)

    data_folder = [f for f in all_files if os.path.isdir(os.path.join(temp_dir, f))][0]
    label_file = [f for f in all_files if f.endswith(".csv")][0]

    data_dir = os.path.join(temp_dir, data_folder)
    label_df = pd.read_csv(os.path.join(temp_dir, label_file))
    label_map = build_label_map(label_df)

    st.session_state.data_dir = data_dir
    st.session_state.label_map = label_map


# ============================================================
# STEP 1: Segment
# ============================================================
if "data_dir" in st.session_state:

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

            for new_idx, meta in bead_info.items():
                orig_idx = meta["orig_idx"]
                if orig_idx > len(bead_ranges):
                    continue

                start, end = bead_ranges[orig_idx - 1]
                bead_df = df.iloc[start:end+1].reset_index(drop=True)

                obs_id = f"{fname}__B{new_idx}"

                if meta["label"] == "OK":
                    segmented_ok.setdefault(fname, {})[new_idx] = (obs_id, bead_df)
                else:
                    segmented_test.setdefault(fname, {})[new_idx] = (obs_id, bead_df)

        st.session_state.segmented_ok = segmented_ok
        st.session_state.segmented_test = segmented_test

        st.success("Segmented!")


# ============================================================
# Analysis Core
# ============================================================
def compute_transformed_signals(observations):
    return [{"csv": o["csv"], "transformed": np.asarray(o["data"])} for o in observations]


def compute_step_normalization_and_flags(ref_obs, test_obs, step_interval):

    ok_steps = []
    ok_meta = []

    for obs in ref_obs:
        _, step = aggregate_for_step(np.arange(len(obs["transformed"])), obs["transformed"], step_interval)
        ok_steps.append(step)
        ok_meta.append((obs["csv"], step))

    min_len = min(len(s) for s in ok_steps)
    ok_matrix = np.vstack([s[:min_len] for s in ok_steps])

    mu = np.median(ok_matrix, axis=0)
    sigma = np.std(ok_matrix, axis=0)
    sigma[sigma < 1e-12] = 1e-12

    min_ok = ok_matrix.min(axis=0)
    max_ok = ok_matrix.max(axis=0)

    fig = go.Figure()

    for name, step in ok_meta:
        fig.add_trace(go.Scatter(y=step[:min_len], name=name, line=dict(color="gray")))

    for obs in test_obs:
        _, step = aggregate_for_step(np.arange(len(obs["transformed"])), obs["transformed"], step_interval)
        step = step[:min_len]

        z = (step - mu) / sigma
        color = "red" if np.any(z > 5) else "green"

        fig.add_trace(go.Scatter(y=step, name=obs["csv"], line=dict(color=color)))

    return fig


# ============================================================
# STEP 3: DataViz
# ============================================================
if st.session_state.segmented_ok and st.session_state.segmented_test:

    bead = st.selectbox("Select Bead", [1,2,3,4])

    ref_obs = []
    for fname, beads in st.session_state.segmented_ok.items():
        if bead in beads:
            obs_id, df = beads[bead]
            ref_obs.append({"csv": obs_id, "data": df.iloc[:,0]})

    test_obs = []
    for fname, beads in st.session_state.segmented_test.items():
        if bead in beads:
            obs_id, df = beads[bead]
            test_obs.append({"csv": obs_id, "data": df.iloc[:,0]})

    ref_t = compute_transformed_signals(ref_obs)
    test_t = compute_transformed_signals(test_obs)

    fig = compute_step_normalization_and_flags(ref_t, test_t, 20)

    st.plotly_chart(fig, use_container_width=True)
