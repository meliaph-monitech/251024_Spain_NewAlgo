import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import os
import tempfile
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import Counter

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


# ============================================================
# --- Label Handling ---
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
        renumbered = []

        for col in bead_cols:
            cell = normalize_label_value(row[col])
            orig_idx = int(col)

            if cell == "IGNORE":
                continue

            if cell in {"OK", "NOK"}:
                final_label = cell
            else:
                final_label = "0"

            renumbered.append((orig_idx, final_label))

        bead_info = {}
        for new_idx, (orig_idx, label) in enumerate(renumbered, start=1):
            bead_info[new_idx] = {
                "orig_idx": orig_idx,
                "label": label
            }

        label_map[fname] = bead_info

    return label_map


# ============================================================
# 🔥 FIXED: Bead Options Logic
# ============================================================
def get_all_bead_options_from_label_map(label_map):
    """
    Use MOST COMMON bead count after Ignore-remapping,
    but never expose more than 4 bead numbers.
    """
    bead_counts = [len(v) for v in label_map.values() if len(v) > 0]

    if not bead_counts:
        return []

    most_common_count = Counter(bead_counts).most_common(1)[0][0]
    most_common_count = min(most_common_count, 4)

    return list(range(1, most_common_count + 1))


# ============================================================
# --- Session State ---
# ============================================================
if "label_map" not in st.session_state:
    st.session_state.label_map = None

if "all_bead_options" not in st.session_state:
    st.session_state.all_bead_options = []


# ============================================================
# STEP 0: Upload ZIP
# ============================================================
st.sidebar.header("Step 0: Upload Data Source")

uploaded_zip = st.sidebar.file_uploader("Upload ZIP", type="zip")

if uploaded_zip:
    temp_dir = tempfile.mkdtemp()

    with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)

    files = os.listdir(temp_dir)

    label_file = [f for f in files if f.endswith(".csv")][0]
    data_folder = [f for f in files if os.path.isdir(os.path.join(temp_dir, f))][0]

    label_df = pd.read_csv(os.path.join(temp_dir, label_file))

    label_map = build_label_map(label_df)

    st.session_state.label_map = label_map
    st.session_state.all_bead_options = get_all_bead_options_from_label_map(label_map)

    st.sidebar.success("Data Loaded")


# ============================================================
# STEP 4: Bead Selection (UNCHANGED UI)
# ============================================================
if st.session_state.label_map:

    st.sidebar.header("Step 4: Bead for Analysis")

    bead_options = st.session_state.all_bead_options

    selected_bead = st.sidebar.selectbox(
        "Select Bead Number",
        bead_options
    )

    st.write("Selected Bead:", selected_bead)
