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
# --- Utility ---
# ============================================================
def make_obs_id(fname, bead):
    return f"{fname}__B{bead}"

def parse_obs_id(obs_id):
    fname, bead = obs_id.split("__B")
    return fname, int(bead)

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
    x = np.asarray(x)
    y = np.asarray(y)
    agg_x = x[::interval]
    agg_y = [np.mean(y[i:i+interval]) for i in range(0, len(y), interval)]
    agg_x = agg_x[:len(agg_y)]
    return agg_x, np.array(agg_y)

def short_label(name):
    return os.path.splitext(os.path.basename(name))[0][:6]

def get_channel_columns(df):
    return df.columns[:2]

# ============================================================
# --- Label Map (FIXED) ---
# ============================================================
def normalize_label(v):
    if pd.isna(v):
        return ""
    return str(v).strip().upper()

def build_label_map(label_df):
    label_map = {}
    bead_cols = [str(i) for i in range(1,7) if str(i) in label_df.columns]

    for _, row in label_df.iterrows():
        fname = os.path.basename(row["FileName"])
        renumbered = []

        for col in bead_cols:
            val = normalize_label(row[col])
            orig_idx = int(col)

            if val == "IGNORE":
                continue

            if val in {"OK","NOK"}:
                label = val
            else:
                label = "0"

            renumbered.append((orig_idx, label))

        bead_info = {}
        for new_idx, (orig_idx, label) in enumerate(renumbered, start=1):
            bead_info[new_idx] = {"orig_idx": orig_idx, "label": label}

        label_map[fname] = bead_info

    return label_map

# ============================================================
# --- Bead Options (FIXED) ---
# ============================================================
def get_bead_options(label_map):
    counts = [len(v) for v in label_map.values() if len(v) > 0]
    if not counts:
        return []
    most_common = Counter(counts).most_common(1)[0][0]
    return list(range(1, most_common+1))

# ============================================================
# --- Session ---
# ============================================================
if "label_map" not in st.session_state:
    st.session_state.label_map = None
if "seg_ok" not in st.session_state:
    st.session_state.seg_ok = None
if "seg_test" not in st.session_state:
    st.session_state.seg_test = None
if "beads" not in st.session_state:
    st.session_state.beads = []

# ============================================================
# STEP 0
# ============================================================
st.sidebar.header("Step 0: Upload")

uploaded = st.sidebar.file_uploader("ZIP", type="zip")

if uploaded:
    tmp = tempfile.mkdtemp()
    with zipfile.ZipFile(uploaded) as z:
        z.extractall(tmp)

    files = os.listdir(tmp)
    label_file = [f for f in files if f.endswith(".csv")][0]
    folder = [f for f in files if os.path.isdir(os.path.join(tmp,f))][0]

    label_df = pd.read_csv(os.path.join(tmp,label_file))
    label_map = build_label_map(label_df)

    st.session_state.label_map = label_map
    st.session_state.beads = get_bead_options(label_map)

    st.sidebar.success("Loaded")

# ============================================================
# STEP 1
# ============================================================
if st.session_state.label_map:
    st.sidebar.header("Step 1: Segment")

    col = st.sidebar.text_input("Column", "Signal")
    thresh = st.sidebar.number_input("Threshold", 1.0)

    if st.sidebar.button("Segment"):
        ok = {}
        test = {}

        for fname, info in st.session_state.label_map.items():
            path = os.path.join(tmp, folder, fname)
            if not os.path.exists(path):
                continue

            df = pd.read_csv(path)
            ranges = segment_beads(df, col, thresh)

            ok[fname] = {}
            test[fname] = {}

            for new_idx, meta in info.items():
                orig = meta["orig_idx"]
                if orig > len(ranges):
                    continue

                s,e = ranges[orig-1]
                bead_df = df.iloc[s:e+1].reset_index(drop=True)

                obs_id = make_obs_id(fname, new_idx)

                if meta["label"] == "OK":
                    ok[fname][new_idx] = (obs_id, bead_df)
                elif meta["label"] == "NOK":
                    test[fname][new_idx] = (obs_id, bead_df)

        st.session_state.seg_ok = ok
        st.session_state.seg_test = test

        st.success("Segmented")

# ============================================================
# STEP 4 + TABS (RESTORED)
# ============================================================
if st.session_state.seg_ok and st.session_state.seg_test:

    st.sidebar.header("Step 4: Bead")
    bead = st.sidebar.selectbox("Bead", st.session_state.beads)

    tabs = st.tabs(["Summary","DataViz"])

    # =========================
    # Summary
    # =========================
    with tabs[0]:
        rows = []

        for fname, beads in st.session_state.seg_test.items():
            if bead in beads:
                obs_id, df = beads[bead]
                val = df.iloc[:,0].mean()

                f,b = parse_obs_id(obs_id)

                rows.append({
                    "CSV": f,
                    "Bead": b,
                    "Value": val
                })

        if rows:
            st.dataframe(pd.DataFrame(rows))
        else:
            st.info("No data")

    # =========================
    # DataViz
    # =========================
    with tabs[1]:
        fig = go.Figure()

        # OK
        for fname, beads in st.session_state.seg_ok.items():
            if bead in beads:
                obs_id, df = beads[bead]
                fig.add_trace(go.Scatter(
                    y=df.iloc[:,0],
                    mode="lines",
                    name=f"{short_label(obs_id)} OK"
                ))

        # TEST
        for fname, beads in st.session_state.seg_test.items():
            if bead in beads:
                obs_id, df = beads[bead]
                fig.add_trace(go.Scatter(
                    y=df.iloc[:,0],
                    mode="lines",
                    name=f"{short_label(obs_id)} TEST"
                ))

        st.plotly_chart(fig, use_container_width=True)
