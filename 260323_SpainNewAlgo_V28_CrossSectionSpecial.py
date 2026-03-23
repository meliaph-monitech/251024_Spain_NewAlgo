# ============================
# ADD THIS ABOVE STEP 1
# ============================

st.sidebar.header("Step 0: Upload Data")

uploaded_zip = st.sidebar.file_uploader("Upload DATA ZIP", type="zip")

if uploaded_zip:
    import tempfile
    temp_dir = tempfile.mkdtemp()

    with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)

    subfolders = [f for f in os.listdir(temp_dir) if os.path.isdir(os.path.join(temp_dir, f))]
    csv_files = [f for f in os.listdir(temp_dir) if f.endswith(".csv")]

    selected_subfolder = st.sidebar.selectbox("Select Data Folder", subfolders)
    selected_label = st.sidebar.selectbox("Select Label CSV", csv_files)

    analysis_mode = st.sidebar.radio("Analysis Mode", ["Per-Bead", "Global"])

    if st.sidebar.button("Apply Data"):

        st.session_state.analysis_mode = analysis_mode
        st.session_state.data_dir = os.path.join(temp_dir, selected_subfolder)

        label_df = pd.read_csv(os.path.join(temp_dir, selected_label))

        # Build label map
        label_map = {}
        for _, row in label_df.iterrows():
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

        st.session_state.label_map = label_map

        st.success("✅ Data Loaded")
