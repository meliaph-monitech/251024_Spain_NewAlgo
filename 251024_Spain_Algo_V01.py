import streamlit as st
import zipfile
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from collections import defaultdict
from scipy.signal import savgol_filter
from sklearn.ensemble import IsolationForest
import shutil

# --- Utility: File Extraction ---
def extract_zip(uploaded_file, extract_dir):
    if os.path.exists(extract_dir):
        for file in os.listdir(extract_dir):
            path = os.path.join(extract_dir, file)
            if os.path.isfile(path):
                os.remove(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)
    else:
        os.makedirs(extract_dir)

    with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    return [os.path.join(extract_dir, f) for f in os.listdir(extract_dir) if f.endswith('.csv')]

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

# --- Utility: Resampling to a fixed length ---
def resample_to_length(y, target_len):
    """Resample a 1D array y to target_len using linear interpolation over normalized index [0,1]."""
    if len(y) == 0:
        return np.zeros(target_len)
    if len(y) == target_len:
        return y.astype(float)
    x_old = np.linspace(0.0, 1.0, num=len(y))
    x_new = np.linspace(0.0, 1.0, num=target_len)
    return np.interp(x_new, x_old, y).astype(float)

# --- Isolation Forest (position-wise) ---
def position_wise_if_matrix(matrix, contamination=0.05, n_estimators=200, random_state=42, max_samples='auto'):
    """
    matrix: shape (num_files, N) where N is resampled length.
    For each position j, fit IsolationForest on matrix[:, j] (1D feature) across files.
    Returns:
      labels: shape (num_files, N), with 1 for inlier, -1 for anomaly
      scores: shape (num_files, N), decision_function (higher is more normal)
    """
    num_files, N = matrix.shape
    labels = np.ones((num_files, N), dtype=int)
    scores = np.zeros((num_files, N), dtype=float)

    # Fit a tiny IF per position. Works well because each fit is 1D & small.
    for j in range(N):
        X = matrix[:, j].reshape(-1, 1)
        # If all values are identical, every point is equally normal → set labels=1, score=0
        if np.allclose(X, X[0]):
            labels[:, j] = 1
            scores[:, j] = 0.0
            continue

        clf = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            max_samples=max_samples,
            n_jobs=-1,
        )
        clf.fit(X)
        labels[:, j] = clf.predict(X)        # 1 (inlier) or -1 (outlier)
        scores[:, j] = clf.decision_function(X)  # higher => more normal

    return labels, scores

# --- Streamlit App ---
st.set_page_config(layout="wide")
st.title("Anomaly-Based NOK Detector (Isolation Forest) with Bead Segmentation")

st.sidebar.header("Upload and Segmentation")
uploaded_zip = st.sidebar.file_uploader("Upload ZIP of CSVs", type="zip")

if uploaded_zip:
    # Peek first CSV for column names
    with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
        first_csv = [name for name in zip_ref.namelist() if name.endswith('.csv')][0]
        with zip_ref.open(first_csv) as f:
            sample_df = pd.read_csv(f)
    columns = sample_df.columns.tolist()

    seg_col = st.sidebar.selectbox("Column for Segmentation", columns)
    seg_thresh = st.sidebar.number_input("Segmentation Threshold", value=3.0)
    signal_col = st.sidebar.selectbox("Signal Column for Analysis", columns)
    analysis_percent = st.sidebar.slider("% of Signal Length to Consider for NOK Decision", 10, 100, 50, 10)

    if st.sidebar.button("Segment Beads"):
        with open("uploaded.zip", "wb") as f:
            f.write(uploaded_zip.getbuffer())
        files = extract_zip("uploaded.zip", "data")

        raw_beads = defaultdict(list)  # bead_num -> list of (filename, series)
        for file in files:
            df = pd.read_csv(file)
            segments = segment_beads(df, seg_col, seg_thresh)
            for bead_num, (start, end) in enumerate(segments, 1):
                sig = df.iloc[start:end+1][signal_col].reset_index(drop=True)
                raw_beads[bead_num].append((os.path.basename(file), sig))

        st.session_state["raw_beads"] = raw_beads
        st.session_state["analysis_ready"] = True
        st.session_state["analysis_percent"] = analysis_percent
        st.success(f"✅ Bead segmentation completed. Found {len(raw_beads)} unique bead numbers across files.")

if "raw_beads" in st.session_state and st.session_state.get("analysis_ready", False):
    raw_beads = st.session_state["raw_beads"]
    analysis_percent = st.session_state["analysis_percent"]

    # --- Smoothing (optional) ---
    st.sidebar.header("Smoothing (Optional)")
    use_smooth = st.sidebar.checkbox("Apply Savitzky–Golay Smoothing", value=False)
    if use_smooth:
        win_len = st.sidebar.number_input("Smoothing Window Length (odd)", 3, 499, 199, step=2)
        polyorder = st.sidebar.number_input("Polynomial Order", 1, 5, 3)

    # --- Isolation Forest Parameters ---
    st.sidebar.header("Isolation Forest")
    resample_len = st.sidebar.number_input("Resampled Length per Bead (N)", 50, 2000, 300, step=50)
    contamination = st.sidebar.number_input("Contamination (expected anomaly proportion)", 0.001, 0.5, 0.05, step=0.01)
    n_estimators = st.sidebar.number_input("n_estimators", 50, 1000, 200, step=50)
    max_samples_opt = st.sidebar.selectbox("max_samples", ["auto", "all"])  # keep simple
    max_samples = 'auto' if max_samples_opt == "auto" else None
    random_state = st.sidebar.number_input("random_state", 0, 9999, 42, step=1)

    bead_options = sorted(raw_beads.keys())
    selected_bead = st.selectbox("Select Bead Number to Display", bead_options)

    st.subheader(f"Raw and (Optional) Smoothed Signal for Bead {selected_bead}")
    raw_fig = go.Figure()
    score_fig = go.Figure()

    # We’ll compute global per-file flags and per-bead tables.
    table_data = []
    global_summary = defaultdict(list)

    # --- Build per-bead matrices and run IF ---
    for bead_num in bead_options:
        file_names = []
        processed_signals = []  # resampled arrays (length = resample_len)
        raw_signals = []        # original Series (for plotting)
        smoothed_signals = []   # smoothed Series (same original length)

        for fname, raw_sig in raw_beads[bead_num]:
            sig = raw_sig.copy().astype(float)
            raw_signals.append((fname, sig))

            if use_smooth and len(sig) >= (win_len if use_smooth else 3):
                try:
                    smoothed = pd.Series(savgol_filter(sig, win_len, polyorder))
                except ValueError:
                    # If parameters invalid for this small signal, fallback to raw
                    smoothed = sig.copy()
            else:
                smoothed = sig.copy()

            smoothed_signals.append((fname, smoothed))

            # Resample (we run IF on resampled to align positions across files)
            rs = resample_to_length(smoothed.values, resample_len)
            processed_signals.append(rs)
            file_names.append(fname)

        if len(processed_signals) == 0:
            continue

        matrix = np.vstack(processed_signals)  # shape (num_files, resample_len)

        # Position-wise IF: compare each position across files for this bead number
        labels, scores = position_wise_if_matrix(
            matrix,
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
            max_samples=max_samples
        )
        # labels: (num_files, N) with -1 anomalies at positions; scores: (num_files, N), higher => more normal
        early_limit = int(resample_len * analysis_percent / 100)

        # Aggregate per file: decide OK / NOK / NOK_Check using early region positions
        for i, fname in enumerate(file_names):
            early_labels = labels[i, :early_limit]
            anomalous_positions = np.where(early_labels == -1)[0]

            if len(anomalous_positions) == 0:
                flag = "OK"
            elif len(anomalous_positions) == 1:
                flag = "NOK"
                global_summary[fname].append(f"{bead_num}")
            else:
                flag = "NOK_Check"
                global_summary[fname].append(f"{bead_num}")

            # Count total anomalous positions over the full bead (for summary info)
            total_anoms = int(np.sum(labels[i, :] == -1))

            table_data.append({
                "File": fname,
                "Bead": bead_num,
                "Change Points / Anom. Positions": total_anoms,
                "Flag": flag
            })

            # If this bead is the one we’re visualizing, add plots
            if bead_num == selected_bead:
                # Highlight anomalous contiguous spans on the resampled axis for readability
                # Build contiguous intervals from labels[i, :]
                anom_spans = []
                in_span = False
                start = 0
                for j in range(resample_len):
                    if labels[i, j] == -1 and not in_span:
                        in_span = True
                        start = j
                    elif labels[i, j] == 1 and in_span:
                        in_span = False
                        anom_spans.append((start, j-1))
                if in_span:
                    anom_spans.append((start, resample_len-1))

                # Shade anomalous spans
                for a0, a1 in anom_spans:
                    raw_fig.add_vrect(x0=a0, x1=a1, fillcolor="red", opacity=0.08, layer="below", line_width=0)

                # Draw the raw series (original index), and smoothed+resampled for alignment
                # For consistent x-axes in combined plot, we’ll plot resampled signals (length=N)
                raw_fig.add_trace(go.Scatter(
                    x=np.arange(resample_len),
                    y=matrix[i, :],
                    mode='lines',
                    name=f"{fname} (resampled{' + smooth' if use_smooth else ''})",
                    line=dict(width=1, color='red' if np.any(labels[i, :] == -1) else 'black')
                ))

                # Score trace (decision_function) for this file → higher is more normal; 0 is a natural reference line
                score_fig.add_trace(go.Scatter(
                    x=np.arange(resample_len),
                    y=scores[i, :],
                    mode='lines',
                    name=f"{fname} Score"
                ))

        # Add a vertical line to indicate the NOK analysis region limit on both plots
        if bead_num == selected_bead:
            raw_fig.add_vline(x=early_limit, line=dict(color="orange", dash="dash"), annotation_text="NOK region limit", annotation_position="top right")
            score_fig.add_vline(x=early_limit, line=dict(color="orange", dash="dash"), annotation_text="NOK region limit", annotation_position="top right")

    # --- Display Plots & Tables ---
    st.plotly_chart(raw_fig, use_container_width=True)
    st.subheader("Isolation Forest Decision Function per Position (higher = more normal)")
    # Add a reference threshold line at y=0
    score_fig.add_hline(y=0.0, line=dict(color="gray", dash="dot"))
    st.plotly_chart(score_fig, use_container_width=True)

    st.subheader("Anomaly Summary Table")
    st.dataframe(pd.DataFrame(table_data))

    st.subheader("Global NOK and NOK_Check Beads Summary")
    global_table = pd.DataFrame([{ "File": k, "NOK/NOK_Check Beads": ", ".join(v) } for k, v in global_summary.items()])
    st.dataframe(global_table)
