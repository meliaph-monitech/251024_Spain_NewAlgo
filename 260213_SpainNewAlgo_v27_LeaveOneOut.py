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
    if len(ref_obs) == 0 or len(test_obs) == 0:
        st.warning("Reference or test signals missing.")
        return None, {}, {}

    status_map = {}
    metrics_map = {}

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("0–1 Normalization (OK-based)", "Z-score per Step")
    )

    # --------------------------------------------------------
    # PRECOMPUTE step arrays for ALL observations
    # --------------------------------------------------------
    def build_step_array(obs):
        y = np.asarray(obs["transformed"], dtype=float)
        x = np.arange(len(y))
        _, step_y = aggregate_for_step(x, y, step_interval)
        return step_y

    ref_step_arrays = {
        obs["csv"]: build_step_array(obs)
        for obs in ref_obs
    }

    test_step_arrays = {
        obs["csv"]: build_step_array(obs)
        for obs in test_obs
    }

    # --------------------------------------------------------
    # LOOP over each TEST file independently (Leave-One-Out)
    # --------------------------------------------------------
    for test_csv, test_steps in test_step_arrays.items():

        # Build reference matrix:
        # All OK refs + ALL OTHER test files except itself
        ref_matrix_list = []

        # Add OK references
        for arr in ref_step_arrays.values():
            if arr.size > 0:
                ref_matrix_list.append(arr)

        # Add other test files (leave-one-out)
        for other_csv, other_arr in test_step_arrays.items():
            if other_csv == test_csv:
                continue
            if other_arr.size > 0:
                ref_matrix_list.append(other_arr)

        if not ref_matrix_list:
            continue

        # Align step length
        min_steps = min(arr.shape[0] for arr in ref_matrix_list + [test_steps])
        if min_steps == 0:
            continue

        ok_matrix = np.vstack([arr[:min_steps] for arr in ref_matrix_list])
        step_y = test_steps[:min_steps]

        # ----------------------------------------------------
        # IDENTICAL to Notebook math
        # ----------------------------------------------------
        mu = np.median(ok_matrix, axis=0)
        sigma = ok_matrix.std(axis=0, ddof=1)
        sigma[sigma < 1e-12] = 1e-12

        min_ok = ok_matrix.min(axis=0)
        max_ok = ok_matrix.max(axis=0)
        denom = max_ok - min_ok
        denom[denom < 1e-12] = 1e-12

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

        status_map[test_csv] = status
        metrics_map[test_csv] = {
            "Norm_Low_Exceed": norm_low_exceed,
            "Norm_High_Exceed": norm_high_exceed,
            "Z_Low_Exceed": z_low_exceed,
            "Z_High_Exceed": z_high_exceed,
        }

    # --------------------------------------------------------
    # PLOT SECTION (UNCHANGED STYLE)
    # --------------------------------------------------------
    step_indices = np.arange(min_steps)

    # Plot reference signals (gray)
    for csv_name, arr in ref_step_arrays.items():
        if arr.size < min_steps:
            continue
        arr = arr[:min_steps]
        fig.add_trace(
            go.Scatter(
                x=step_indices,
                y=(arr - min_ok) / denom,
                mode="lines",
                line=dict(color="#aaaaaa", width=1),
                showlegend=False
            ),
            row=1, col=1
        )

    # Plot test signals
    for csv_name, arr in test_step_arrays.items():
        if arr.size < min_steps:
            continue

        arr = arr[:min_steps]
        norm_vals = (arr - min_ok) / denom
        z_vals = (arr - mu) / sigma

        status = status_map.get(csv_name, "ok")

        if status == "low":
            color, width = "red", 2
        elif status == "high":
            color, width = "orange", 2
        else:
            color, width = "green", 1

        fig.add_trace(
            go.Scatter(
                x=step_indices,
                y=norm_vals,
                mode="lines",
                name=f"{short_label(csv_name)} (TEST)",
                line=dict(color=color, width=width),
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=step_indices,
                y=z_vals,
                mode="lines",
                line=dict(color=color, width=width),
                showlegend=False
            ),
            row=1, col=2
        )

    fig.add_hline(y=norm_lower, line=dict(color="gray", dash="dash"), row=1, col=1)
    fig.add_hline(y=norm_upper, line=dict(color="gray", dash="dash"), row=1, col=1)
    fig.add_hline(y=-z_lower, line=dict(color="gray", dash="dash"), row=1, col=2)
    fig.add_hline(y=z_upper, line=dict(color="gray", dash="dash"), row=1, col=2)

    fig.update_layout(
        title=dict(text=f"Per-step Normalization {title_suffix}", font=dict(size=22)),
        legend=dict(orientation="h")
    )

    return fig, status_map, metrics_map
