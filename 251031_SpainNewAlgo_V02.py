import streamlit as st
import pandas as pd
import numpy as np
import zipfile, os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import savgol_filter, butter, filtfilt

st.set_page_config(layout="wide")

# --- Utility: Bead Segmentation (unchanged) ---
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
    """Aggregate x and y into buckets of given interval for step plotting."""
    if interval <= 0:
        interval = 1
    x = np.asarray(x)
    y = np.asarray(y)
    agg_x = x[::interval]
    agg_y = [np.mean(y[i:i+interval]) for i in range(0, len(y), interval)]
    # Align lengths
    agg_x = agg_x[:len(agg_y)]
    return agg_x, np.array(agg_y)

# --- Session State ---
if "segmented_data" not in st.session_state:
    st.session_state.segmented_data = None

# --- Sidebar: Upload & Segmentation (Stage 1 - keep intact) ---
st.sidebar.header("Step 1: Upload & Segmentation")
uploaded_zip = st.sidebar.file_uploader("Upload ZIP of CSV files", type="zip")

if uploaded_zip:
    with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
        csv_names = [name for name in zip_ref.namelist() if name.endswith('.csv')]
        if not csv_names:
            st.sidebar.error("No CSV files found in the ZIP.")
        else:
            first_csv = csv_names[0]
            with zip_ref.open(first_csv) as f:
                sample_df = pd.read_csv(f)
            columns = sample_df.columns.tolist()
            seg_col = st.sidebar.selectbox("Column for Segmentation", columns)
            seg_thresh = st.sidebar.number_input("Segmentation Threshold", value=1.0)
            segment_btn = st.sidebar.button("Bead Segmentation")

# --- Perform Segmentation (unchanged logic) ---
if uploaded_zip and 'segment_btn' in locals() and segment_btn:
    segmented_data = {}
    with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
        for file_name in zip_ref.namelist():
            if file_name.endswith('.csv'):
                with zip_ref.open(file_name) as f:
                    df = pd.read_csv(f)
                bead_ranges = segment_beads(df, seg_col, seg_thresh)
                bead_dict = {}
                for idx, (start, end) in enumerate(bead_ranges, start=1):
                    bead_dict[idx] = df.iloc[start:end+1].reset_index(drop=True)
                segmented_data[os.path.basename(file_name)] = bead_dict
    st.session_state.segmented_data = segmented_data
    st.success("✅ Bead segmentation complete and locked!")

# --- Helper: Transform Signals ---
def compute_transformed_signals(observations, mode, **params):
    """
    observations: list of dicts with keys: csv, status, data (pd.Series or 1D array)
    mode: 'raw', 'savgol', 'lowpass', 'poly'
    params: mode-specific parameters
    """
    transformed_obs = []
    for obs in observations:
        y = np.asarray(obs["data"]).astype(float)
        if mode == "raw":
            transformed = y
        elif mode == "savgol":
            window = int(params.get("window", 51))
            poly = int(params.get("poly", 2))
            # Make sure window is valid for this signal
            if window >= len(y):
                window = len(y) - 1 if len(y) % 2 == 0 else len(y)
            if window < 3:
                transformed = y  # fallback
            else:
                if window % 2 == 0:
                    window -= 1
                transformed = savgol_filter(y, window_length=window, polyorder=poly)
        elif mode == "lowpass":
            cutoff = float(params.get("cutoff", 0.1))
            order = int(params.get("order", 2))
            b, a = butter(order, cutoff, btype='low', analog=False)
            if len(y) > 3 * max(len(a), len(b)):
                transformed = filtfilt(b, a, y)
            else:
                transformed = y  # fallback if too short
        elif mode == "poly":
            deg = int(params.get("deg", 25))
            x = np.arange(len(y))
            # Degree must be < number of points
            max_deg = max(1, len(y) - 1)
            deg = min(deg, max_deg)
            try:
                coeffs = np.polyfit(x, y, deg)
                transformed = np.polyval(coeffs, x)
            except np.linalg.LinAlgError:
                transformed = y  # fallback
        else:
            transformed = y

        transformed_obs.append({
            **obs,
            "transformed": transformed
        })
    return transformed_obs

# --- Helper: Plot Top (Transformed Signals) ---
def plot_top_signals(transformed_obs, title, y_label):
    fig = go.Figure()
    for obs in transformed_obs:
        y = obs["transformed"]
        x = np.arange(len(y))
        color = "green" if obs["status"] == "OK" else "red"
        name = f"{obs['csv']} ({obs['status']})"
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode="lines",
            name=name,
            line=dict(color=color, width=2 if obs["status"] == "NOK" else 1)
        ))
    fig.update_layout(
        title=title,
        xaxis_title="Index",
        yaxis_title=y_label,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )
    st.plotly_chart(fig, use_container_width=True)

# --- Helper: Per-Step Normalization (0–1 + z-score) ---
def plot_step_normalization(transformed_obs, step_interval, title_suffix):
    # Build step-wise data
    step_meta = []  # list of dict: csv, status, step_x, step_y
    ok_step_arrays = []

    for obs in transformed_obs:
        y = np.asarray(obs["transformed"])
        x = np.arange(len(y))
        step_x, step_y = aggregate_for_step(x, y, step_interval)
        step_meta.append({
            "csv": obs["csv"],
            "status": obs["status"],
            "step_x": np.asarray(step_x),
            "step_y": np.asarray(step_y)
        })
        if obs["status"] == "OK":
            ok_step_arrays.append(np.asarray(step_y))

    if len(ok_step_arrays) == 0:
        st.warning("No OK signals available for normalization reference.")
        return

    # Align by minimum number of steps across OK signals
    min_steps = min(arr.shape[0] for arr in ok_step_arrays)
    ok_matrix = np.vstack([arr[:min_steps] for arr in ok_step_arrays])  # shape: (n_ok, min_steps)

    # Reference stats from OK
    mu = ok_matrix.mean(axis=0)
    sigma = ok_matrix.std(axis=0, ddof=1)
    sigma[sigma < 1e-12] = 1e-12  # avoid division by zero
    min_ok = ok_matrix.min(axis=0)
    max_ok = ok_matrix.max(axis=0)
    denom = max_ok - min_ok
    denom[denom < 1e-12] = 1e-12  # avoid division by zero

    step_indices = np.arange(min_steps)

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("0–1 Normalization (OK-based)", "Z-score per Step")
    )

    for meta in step_meta:
        step_y = meta["step_y"][:min_steps]
        norm_vals = (step_y - min_ok) / denom
        z_vals = (step_y - mu) / sigma

        color = "green" if meta["status"] == "OK" else "red"
        name = f"{meta['csv']} ({meta['status']})"
        width = 2 if meta["status"] == "NOK" else 1

        # Left: 0–1 normalization
        fig.add_trace(
            go.Scatter(
                x=step_indices,
                y=norm_vals,
                mode="lines",
                name=name,
                line=dict(color=color, width=width),
                legendgroup=meta["csv"],
                showlegend=True
            ),
            row=1, col=1
        )

        # Right: z-score
        fig.add_trace(
            go.Scatter(
                x=step_indices,
                y=z_vals,
                mode="lines",
                line=dict(color=color, width=width),
                legendgroup=meta["csv"],
                showlegend=False  # keep legend only on left subplot
            ),
            row=1, col=2
        )

    fig.update_xaxes(title_text="Step Index", row=1, col=1)
    fig.update_yaxes(title_text="Normalized Value", row=1, col=1)
    fig.update_xaxes(title_text="Step Index", row=1, col=2)
    fig.update_yaxes(title_text="Z-score", row=1, col=2)

    fig.update_layout(
        title=f"Per-step Normalization {title_suffix}",
        legend=dict(orientation="h", yanchor="bottom", y=1.15, xanchor="left", x=0)
    )

    st.plotly_chart(fig, use_container_width=True)

# --- Stage 2: Bead & File Selection + Analysis ---
if st.session_state.segmented_data:
    st.sidebar.header("Step 2: Bead & File Selection")

    all_files = sorted(st.session_state.segmented_data.keys())

    # Determine available bead numbers (union across files)
    bead_set = set()
    for fname, beads in st.session_state.segmented_data.items():
        bead_set.update(beads.keys())
    bead_options = sorted(bead_set)

    if not bead_options:
        st.warning("No beads found after segmentation.")
    else:
        selected_bead = st.sidebar.selectbox("Select Bead Number", bead_options)

        # Find an example bead df to list signal columns
        example_bead_df = None
        for fname in all_files:
            beads = st.session_state.segmented_data[fname]
            if selected_bead in beads:
                example_bead_df = beads[selected_bead]
                break

        if example_bead_df is None:
            st.error("Selected bead not found in any file.")
        else:
            signal_col = st.sidebar.selectbox(
                "Select Signal Column",
                example_bead_df.columns.tolist()
            )

            # Files that actually contain this bead
            files_with_bead = [
                f for f in all_files if selected_bead in st.session_state.segmented_data[f]
            ]

            # NOK selection (rest will be OK)
            nok_files = st.sidebar.multiselect(
                "Select NOK Files (others with this bead are treated as OK)",
                options=files_with_bead,
                default=[]
            )
            ok_files = [f for f in files_with_bead if f not in nok_files]

            if len(ok_files) == 0:
                st.warning("At least one OK file is required to build normalization reference.")
            else:
                # Build observations list
                observations = []
                for fname in ok_files:
                    bead_df = st.session_state.segmented_data[fname][selected_bead]
                    data = bead_df[signal_col].reset_index(drop=True)
                    observations.append({
                        "csv": fname,
                        "status": "OK",
                        "data": data
                    })
                for fname in nok_files:
                    bead_df = st.session_state.segmented_data[fname][selected_bead]
                    data = bead_df[signal_col].reset_index(drop=True)
                    observations.append({
                        "csv": fname,
                        "status": "NOK",
                        "data": data
                    })

                if not observations:
                    st.info("No signals found for selected bead and signal column.")
                else:
                    st.markdown(
                        f"### Analysis for Bead **{selected_bead}**, Signal **{signal_col}**"
                    )
                    st.markdown(
                        f"- OK files: {len(ok_files)}  \n"
                        f"- NOK files: {len(nok_files)}"
                    )

                    # Create tabs
                    tabs = st.tabs([
                        "Raw Signal",
                        "Smoothed (Savitzky)",
                        "Low-pass Filter",
                        "Curve Fit"
                    ])

                    # --- Tab 1: Raw Signal ---
                    with tabs[0]:
                        st.subheader("Raw Signal (Top) + Per-step Normalization (Bottom)")
                        step_interval_raw = st.slider(
                            "Step Interval (points) - Raw",
                            min_value=10,
                            max_value=500,
                            value=70,
                            step=10
                        )

                        transformed_raw = compute_transformed_signals(
                            observations, mode="raw"
                        )
                        plot_top_signals(
                            transformed_raw,
                            title="Raw Signal (OK vs NOK)",
                            y_label="Signal Value"
                        )
                        plot_step_normalization(
                            transformed_raw,
                            step_interval=step_interval_raw,
                            title_suffix="(Raw Signal)"
                        )

                    # --- Tab 2: Smoothed (Savitzky) ---
                    with tabs[1]:
                        st.subheader("Smoothed Signal (Top) + Per-step Normalization (Bottom)")

                        window = st.slider(
                            "Savitzky-Golay Window Length",
                            min_value=5,
                            max_value=101,
                            value=51,
                            step=10
                        )
                        poly = st.slider(
                            "Polynomial Order",
                            min_value=2,
                            max_value=5,
                            value=2,
                            step=1
                        )
                        step_interval_sg = st.slider(
                            "Step Interval (points) - Smoothed",
                            min_value=10,
                            max_value=500,
                            value=70,
                            step=10
                        )

                        transformed_sg = compute_transformed_signals(
                            observations,
                            mode="savgol",
                            window=window,
                            poly=poly
                        )
                        plot_top_signals(
                            transformed_sg,
                            title="Smoothed Signal (Savitzky-Golay)",
                            y_label="Signal Value"
                        )
                        plot_step_normalization(
                            transformed_sg,
                            step_interval=step_interval_sg,
                            title_suffix="(Smoothed)"
                        )

                    # --- Tab 3: Low-pass Filter ---
                    with tabs[2]:
                        st.subheader("Low-pass Filtered Signal (Top) + Per-step Normalization (Bottom)")

                        cutoff = st.slider(
                            "Low-pass Cutoff Frequency (normalized, 0.01–0.5)",
                            min_value=0.01,
                            max_value=0.5,
                            value=0.1,
                            step=0.01
                        )
                        order = st.slider(
                            "Filter Order",
                            min_value=1,
                            max_value=5,
                            value=2,
                            step=1
                        )
                        step_interval_lp = st.slider(
                            "Step Interval (points) - Low-pass",
                            min_value=10,
                            max_value=500,
                            value=70,
                            step=10
                        )

                        transformed_lp = compute_transformed_signals(
                            observations,
                            mode="lowpass",
                            cutoff=cutoff,
                            order=order
                        )
                        plot_top_signals(
                            transformed_lp,
                            title="Low-pass Filtered Signal",
                            y_label="Signal Value"
                        )
                        plot_step_normalization(
                            transformed_lp,
                            step_interval=step_interval_lp,
                            title_suffix="(Low-pass)"
                        )

                    # --- Tab 4: Curve Fit ---
                    with tabs[3]:
                        st.subheader("Curve Fit Signal (Top) + Per-step Normalization (Bottom)")

                        deg = st.slider(
                            "Curve Fit Polynomial Degree",
                            min_value=1,
                            max_value=100,
                            value=25,
                            step=1
                        )
                        step_interval_cf = st.slider(
                            "Step Interval (points) - Curve Fit",
                            min_value=10,
                            max_value=500,
                            value=70,
                            step=10
                        )

                        transformed_cf = compute_transformed_signals(
                            observations,
                            mode="poly",
                            deg=deg
                        )
                        plot_top_signals(
                            transformed_cf,
                            title="Curve-fit Signal (Polynomial)",
                            y_label="Signal Value"
                        )
                        plot_step_normalization(
                            transformed_cf,
                            step_interval=step_interval_cf,
                            title_suffix="(Curve Fit)"
                        )
