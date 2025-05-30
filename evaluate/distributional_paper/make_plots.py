import matplotlib.pyplot as plt
import scienceplots
plt.style.use("science")

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from scipy.interpolate import interp1d
from matplotlib.colors import LogNorm, PowerNorm


pd.set_option("display.max_colwidth", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.expand_frame_repr", False)

from src.io_utils import get_filtered_and_grouped_paths, collect_results, num_model_params


def generate_sample_sizes(total_samples: int) -> tuple[int, ...]:
    if total_samples < 1:
        raise ValueError("total_samples must be ≥ 1")
    bases = (1, 2, 5)          # 1-2-5 pattern for each power of ten
    result = []
    power = 0
    while True:
        scale = 10 ** power
        for b in bases:
            value = b * scale
            if value > total_samples:
                # Stop once the next milestone exceeds the target
                result.append(total_samples) if result[-1] != total_samples else None
                return tuple(result)
            result.append(value)
            if value == total_samples:
                return tuple(result)
        power += 1

def _dominance_frontier(xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the non-dominated (Pareto-optimal) points, ordered by cost.
    The frontier is defined as points for which no other point has
    *both* lower cost (x) and lower mean p_harmful (y).

    Parameters
    ----------
    xs, ys : 1-D arrays of equal length
        Coordinates of the candidate points.

    Returns
    -------
    frontier_xs, frontier_ys : 1-D arrays
        Coordinates of the Pareto frontier, sorted by xs ascending.
    """
    order = np.argsort(xs)              # sort by cost
    xs_sorted, ys_sorted = xs[order], ys[order]

    frontier_x, frontier_y = [0], [0]
    best_y_so_far = 0
    for x_val, y_val in zip(xs_sorted, ys_sorted):
        if y_val > best_y_so_far:       # strictly better in y
            frontier_x.append(x_val)
            frontier_y.append(y_val)
            best_y_so_far = y_val
    frontier_x.append(xs_sorted[-1])
    frontier_y.append(frontier_y[-1])
    return np.asarray(frontier_x), np.asarray(frontier_y)


# ------------------------------------------------------------------
# 1. Empirical‑copula Pareto frontier (no Archimedean fit required)
# ------------------------------------------------------------------
def _copula_frontier(xs: np.ndarray,
                     ys: np.ndarray,
                     eps: float = 1e-9) -> tuple[np.ndarray, np.ndarray]:
    """
    Pareto frontier estimator based on the empirical copula level
    set at alpha* = min_i C_n(U_i).

    Parameters
    ----------
    xs, ys : 1-D arrays
        Coordinates of the candidate points.
    eps : float
        Numerical tolerance when selecting the boundary.

    Returns
    -------
    frontier_xs, frontier_ys : 1-D arrays
        Estimated frontier, ordered by xs ascending.
    """
    n = xs.size
    # pseudo‑observations U_k  (Eq. 7 in the paper)
    u = rankdata(-xs, method="ordinal") / (n + 1.0)
    v = rankdata(ys, method="ordinal") / (n + 1.0)
    U = np.column_stack((u, v))

    # empirical copula values at the sample points
    #   C_n(U_i) = 1/n * number of points dominated by U_i
    dom_matrix = (U[:, None, :] <= U[None, :, :]).all(axis=2)
    C_vals = dom_matrix.mean(axis=1)

    alpha_star = C_vals.min()          # Lemma 2.1
    on_boundary = np.abs(C_vals - alpha_star) < eps

    fx, fy = xs[on_boundary], ys[on_boundary]
    order = np.argsort(fx)
    return -fx[order], fy[order]


# ------------------------------------------------------------------
# 2. Thin wrapper so you can switch methods with one argument
# ------------------------------------------------------------------
def _pareto_frontier(xs: np.ndarray,
                     ys: np.ndarray,
                     method: str = "empirical_copula",
                     **kwargs):
    if method == "empirical_copula":
        return _copula_frontier(xs, ys, **kwargs)
    elif method == "basic":
        # your original dominance‑based frontier
        return _dominance_frontier(xs, ys)   # rename old function
    else:
        raise ValueError(f"Unknown frontier method '{method}'")


def pareto_plot(
    results: dict[str,np.ndarray],
    baseline: dict[str,np.ndarray] | None = None,
    title: str = "Pareto Frontier",
    sample_levels_to_plot: tuple[int, ...]|None = None,
    frontier_method: str = "basic",
    metric: tuple[str, ...] = ('scores', 'strong_reject', 'p_harmful'),
    plot_points: bool = True,
    plot_frontiers: bool = True,
    plot_envelope: bool = False,
    verbose: bool = True,
    cumulative: bool = False,
    flops_per_step: int | None = None,
    n_x_points: int = 10000,
    x_scale="log",
    threshold: float|None = None,
    color_scale: str = "linear",
):
    """
    Scatter the full design-space AND overlay Pareto frontiers
    for selected sampling counts.

    Parameters
    ----------
    data : (x, y) where x is ignored and y has shape (n_steps, total_samples)
        Your original score tensor.
    sampling_cost_factor : float, optional
        Multiplies the sampling cost term j.
    frontier_samples : tuple[int, ...], optional
        Which n_sample values to draw Pareto lines for.

    Returns
    -------
    None
    """
    y = np.array(results[metric])  # (B, n_steps, n_samples)
    if threshold is not None:
        y = y > threshold
    n_runs, n_steps, n_total_samples = y.shape
    if sample_levels_to_plot is None:
        sample_levels_to_plot = generate_sample_sizes(n_total_samples)

    flops_sampling = np.array(results["flops_sampling"]) # (B, n_steps)
    if "flops" in results:
        flops_optimization = np.array(results["flops"]) # (B, n_steps)
    else:
        flops_optimization = np.zeros_like(flops_sampling) # (B, n_steps)
        if flops_per_step is not None:
            flops_optimization += flops_per_step(np.arange(flops_optimization.shape[1]))


    def subsample_and_aggregate(step_idx, sample_idx, cumulative, y, opt_flops, sampling_flops, rng):
        n_runs, n_steps, n_total_samples = y.shape
        opt_flop = np.mean(opt_flops[:, :step_idx+1].sum(axis=1))
        sampling_flop = np.mean(sampling_flops[:, step_idx]) * sample_idx
        if cumulative and step_idx > 0:
            samples_at_end = y[:, step_idx, rng.choice(n_total_samples, size=sample_idx, replace=False)].max(axis=-1)
            samples_up_to_now = y[:, :step_idx, rng.choice(n_total_samples, size=1, replace=False)].max(axis=1)[:, 0]
            values = np.stack([samples_up_to_now, samples_at_end], axis=1).max(axis=1)
            return (opt_flop + sampling_flop, step_idx, sample_idx, values.mean(0))
        return (opt_flop + sampling_flop, step_idx, sample_idx, y[:, step_idx, rng.choice(n_total_samples, size=sample_idx, replace=False)].max(axis=-1).mean(axis=0))


    def get_pts(y, opt_flops, sampling_flops):
        n_runs, n_steps, total_samples = y.shape
        rng = np.random.default_rng()
        pts = []  # (cost, step, n_samples, mean_p)
        for j in range(1, total_samples + 1, 1):
            for i in range(0, n_steps, 1):
                pts.append(subsample_and_aggregate(i, j, cumulative, y, opt_flops, sampling_flops, rng))
        pts = np.asarray(pts)
        return pts

    pts = get_pts(y, flops_optimization, flops_sampling)
    cost, step_idx, n_samp, mean_p = pts.T
    max_cost = max(cost)
    if x_scale == "log":
        x_interp = np.logspace(11, np.log10(max_cost+1), n_x_points)
    else:
        x_interp = np.linspace(0, max_cost+1, n_x_points)

    # Create figure with subplots: main plot + 2 bar charts
    fig = plt.figure(figsize=(18, 6))

    # Main Pareto plot (left half, spanning both rows)
    ax1 = plt.subplot2grid((1, 4), (0, 0), colspan=2)

    # ---------- scatter all points ----------
    if plot_points:
        if color_scale == "log":
            color_norm = LogNorm()
        elif color_scale == "sqrt":
            color_norm = PowerNorm(gamma=0.5)  # Square root normalization
        else:
            color_norm = None
        sc = plt.scatter(cost, mean_p, c=n_samp, cmap="viridis", alpha=0.15, s=3, norm=color_norm)
    plt.xlabel("Cost (FLOPS (optimization + sampling))", fontsize=14)
    if threshold is None:
        plt.ylabel("Mean p_harmful", fontsize=14)
    else:
        plt.ylabel(f"Max ASR (threshold: {threshold})", fontsize=12)

    # ---------- overlay Pareto frontiers ----------
    cmap = plt.get_cmap("viridis")
    if color_scale == "log":
        norm = LogNorm(n_samp.min(), n_samp.max())
    elif color_scale == "sqrt":
        norm = PowerNorm(gamma=0.5, vmin=n_samp.min(), vmax=n_samp.max())
    else:
        norm = plt.Normalize(n_samp.min(), n_samp.max())
    rng = np.random.default_rng()

    n_smoothing = 50
    frontier_data = {}  # Store frontier data for bar charts

    if plot_frontiers:
        for j in sample_levels_to_plot:
            xs = []
            ys = []
            if j == n_total_samples:
                n_smoothing = 1
            for _ in range(n_smoothing):
                pts = []
                for i in range(0, n_steps, 1):
                    pts.append(subsample_and_aggregate(i, j, cumulative, y, flops_optimization, flops_sampling, rng))

                pts = np.asarray(pts)
                cost, _, _, mean_p = pts.T

                fx, fy = _pareto_frontier(cost, mean_p, method=frontier_method)
                xs.append(fx)
                ys.append(fy)
            y_interp = [interp1d(x_, y_, kind="previous", bounds_error=False, fill_value=(0, max(y_)))(x_interp) for x_, y_ in zip(xs, ys)]

            color = cmap(norm(j))
            y_mean = np.mean(y_interp, axis=0)
            # Filter out leading zeros
            nonzero_mask = y_mean > 0

            # Store data for bar charts
            frontier_data[j] = {
                'x': x_interp[nonzero_mask],
                'y': y_mean[nonzero_mask],
                'color': color,
                'max_asr': np.max(y_mean[nonzero_mask]) if np.any(nonzero_mask) else 0
            }

            plt.plot(
                x_interp[nonzero_mask],
                y_mean[nonzero_mask],
                marker="o",
                linewidth=1.8,
                markersize=2,
                label=f"{j} samples",
                color=color,
            )

    if plot_envelope:
        n_smoothing = n_total_samples
        y_interps = []
        for j in range(1, n_total_samples+1):
            xs = []
            ys = []
            for n in range(n_smoothing):
                pts = []
                for i in range(0, n_steps, 1):
                    pts.append(subsample_and_aggregate(i, j, cumulative, y, flops_optimization, flops_sampling, rng))

                pts = np.asarray(pts)
                cost, step_idx, n_samp, mean_p = pts.T

                fx, fy = _pareto_frontier(cost, mean_p, method=frontier_method)
                xs.append(fx)
                ys.append(fy)

            y_interp = [interp1d(x_, y_, kind="previous", bounds_error=False, fill_value=(0, max(y_)))(x_interp) for x_, y_ in zip(xs, ys)]
            y_interps.append(np.mean(y_interp, axis=0))
        y_interps = np.array(y_interps)
        argmax = np.argmax(y_interps, axis=0)
        argmax = np.maximum.accumulate(argmax)
        y_envelope = np.max(y_interps, axis=0)

        # Filter out leading zeros
        nonzero_mask = y_envelope > 0
        color = [cmap(norm(argmax[i])) for i in range(len(argmax)) if nonzero_mask[i]]
        plt.scatter(x_interp[nonzero_mask], y_envelope[nonzero_mask], c=color, s=2)

    title_suffix = ""

    # Handle baseline data
    baseline_max_asr = 0
    baseline_frontier_data = None

    if baseline is not None:
        y_baseline = np.array(baseline[metric]) # (B, n_steps, n_samples)
        if threshold is not None:
            y_baseline = y_baseline > threshold

        baseline_flops_sampling = np.array(baseline["flops_sampling"])
        if "flops" in baseline:
            baseline_flops_optimization = np.array(baseline["flops"]) # (B, n_steps)
        else:
            baseline_flops_optimization = np.zeros_like(baseline_flops_sampling) # (B, n_steps)
            if flops_per_step is not None:
                baseline_flops_optimization += flops_per_step(np.arange(baseline_flops_optimization.shape[1]))

        if y_baseline is not None:
            title_suffix = f" ({n_runs}, {y_baseline.shape[0]})"
            if verbose:
                print(n_runs, "for main")
                print(y_baseline.shape[0], "for baseline")
            n_runs_baseline, n_steps_baseline, n_total_samples_baseline = y_baseline.shape
            assert n_total_samples_baseline == 1

            rng = np.random.default_rng()
            pts = []  # (cost, step, n_samples, mean_p)
            for i in range(0, n_steps_baseline, 1):
                for j in range(1, n_total_samples_baseline + 1, 1):
                    pts.append(subsample_and_aggregate(i, j, cumulative, y_baseline, baseline_flops_optimization, baseline_flops_sampling, rng))

            pts = np.asarray(pts)
            cost_baseline, step_idx_baseline, n_samp_baseline, mean_p_baseline = pts.T

            # ---------- overlay Pareto frontiers ----------
            if plot_frontiers or plot_envelope:
                mask = n_samp_baseline == 1
                fx, fy = _pareto_frontier(cost_baseline[mask], mean_p_baseline[mask], method=frontier_method)
                y_interp_baseline = interp1d(fx, fy, kind="previous", bounds_error=False, fill_value=(0, max(fy)))(x_interp)
                nonzero_mask_baseline = y_interp_baseline > 0

                # Store baseline data for bar charts
                baseline_max_asr = np.max(y_interp_baseline[nonzero_mask_baseline]) if np.any(nonzero_mask_baseline) else 0
                baseline_frontier_data = {
                    'x': x_interp[nonzero_mask_baseline],
                    'y': y_interp_baseline[nonzero_mask_baseline],
                    'max_asr': baseline_max_asr
                }

                plt.plot(
                    x_interp[nonzero_mask_baseline],
                    y_interp_baseline[nonzero_mask_baseline],
                    marker="o",
                    linewidth=1.8,
                    markersize=2,
                    label=f"greedy",
                    color="r",
                )

    plt.title(title + title_suffix)
    plt.grid(True, alpha=0.3)
    plt.ylim(bottom=0)
    plt.xscale(x_scale)
    plt.legend(title="Frontiers", loc="upper left" if x_scale == "log" else "lower right")

    # ---------- Bar Chart 1: Max ASR Comparison (Vertical Slice) ----------
    ax2 = plt.subplot2grid((1, 4), (0, 2))

    methods = []
    max_asrs = []
    colors = []

    # Add baseline (delta = 0 for baseline)
    if baseline_frontier_data is not None:
        methods.append("Greedy")
        max_asrs.append(0.0)  # Delta from itself is 0
        colors.append("red")

    # Add sampling methods (calculate delta from baseline)
    for j in sample_levels_to_plot:
        if j in frontier_data:
            methods.append(f"{j} samples")
            delta_asr = frontier_data[j]['max_asr'] - baseline_max_asr if baseline_frontier_data is not None else 0
            max_asrs.append(delta_asr)
            colors.append(frontier_data[j]['color'])

    if methods:
        bars = plt.bar(methods, max_asrs, color=colors, alpha=0.7, edgecolor='black')
        plt.xlabel("Method", fontsize=12)
        if threshold is None:
            plt.ylabel(r"$\Delta$ $p_{harmful}$ - greedy", fontsize=12)
        else:
            plt.ylabel(r"Max $p_{harmful}$" + f" (threshold: {threshold})", fontsize=12)
        plt.title(r"Max $p_{harmful}$ Comparison", fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        # Increase ylim by 2% on top and bottom
        ymin, ymax = plt.ylim()
        margin = (ymax - ymin) * 0.03
        plt.ylim(ymin - margin, ymax + margin)

        # ----- add labels with a 4-point gap -----
        for bar, value in zip(bars, max_asrs):
            # choose label position: above for positive, below for negative
            offset_pt = 4      # visual gap in points
            y = bar.get_height()
            va = 'bottom' if y >= 0 else 'top'
            offset = (0, offset_pt if y >= 0 else -offset_pt)

            ax2.annotate(f'{value:.3f}',
                        xy=(bar.get_x() + bar.get_width()/2, y),
                        xytext=offset,
                        textcoords='offset points',
                        ha='center', va=va, fontsize=10)

    # ---------- Bar Chart 2: FLOPS Efficiency to Reach Greedy ASR (Horizontal Slice) ----------
    ax3 = plt.subplot2grid((1, 4), (0, 3))

    if baseline_frontier_data is not None and baseline_max_asr > 0:
        methods_flops = []
        flops_required = []
        colors_flops = []

        # Find FLOPS required to reach baseline ASR for each sampling method
        target_asr = baseline_max_asr

        for j in sample_levels_to_plot:
            if j in frontier_data:
                # Find the minimum FLOPS where ASR >= target_asr
                y_vals = frontier_data[j]['y']
                x_vals = frontier_data[j]['x']

                # Find points where ASR >= target_asr
                valid_indices = y_vals >= target_asr
                if np.any(valid_indices):
                    min_flops = np.min(x_vals[valid_indices])
                    methods_flops.append(f"{j} samples")
                    flops_required.append(min_flops)
                    colors_flops.append(frontier_data[j]['color'])

        # Add baseline (find minimum FLOPS where it reaches target ASR)
        if baseline_frontier_data['x'].size > 0:
            # Find the minimum FLOPS where baseline ASR >= target_asr
            baseline_y_vals = baseline_frontier_data['y']
            baseline_x_vals = baseline_frontier_data['x']
            baseline_valid_indices = baseline_y_vals >= target_asr
            if np.any(baseline_valid_indices):
                baseline_flops = np.min(baseline_x_vals[baseline_valid_indices])
            else:
                # Fallback to minimum FLOPS if no point reaches target ASR
                baseline_flops = np.min(baseline_x_vals)
            methods_flops.insert(0, "Greedy")
            flops_required.insert(0, baseline_flops)
            colors_flops.insert(0, "red")

        if methods_flops:
            bars = plt.bar(methods_flops, flops_required, color=colors_flops, alpha=0.7, edgecolor='black')
            plt.xlabel("Method", fontsize=12)
            plt.ylabel("FLOPS Required", fontsize=12)
            plt.title(r"FLOPS to Reach Greedy $p_{harmful}$" + f" ( = {target_asr:.3f})", fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.yscale('log')
            plt.grid(True, alpha=0.3, axis='y')
            # Increase ylim by 2% on top and bottom
            ymin, ymax = plt.ylim()
            plt.ylim(ymin, ymax * 1.1)

            # Add value labels on bars
            # for bar, value in zip(bars, flops_required):
            #     plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1,
            #             f'{value:.2e}', ha='center', va='bottom', fontsize=9, rotation=45)
            # --- constant 5-point vertical gap ---
            for bar, value in zip(bars, flops_required):
                ax3.annotate(f'{value:.2e}',
                            xy=(bar.get_x() + bar.get_width()/2, value),   # anchor at top of bar
                            xytext=(0, 5),                                  # 5 points straight up
                            textcoords='offset points',
                            ha='center', va='bottom', rotation=45, fontsize=9)

    plt.tight_layout()
    plt.savefig(f"evaluate/distributional_paper/pareto_plots/{title}.pdf")
    plt.close()


def flops_ratio_plot(
    results: dict[str,np.ndarray],
    baseline: dict[str,np.ndarray] | None = None,
    title: str = "FLOPS Ratio Analysis",
    sample_levels_to_plot: tuple[int, ...]|None = None,
    metric: tuple[str, ...] = ('scores', 'strong_reject', 'p_harmful'),
    cumulative: bool = False,
    flops_per_step: int | None = None,
    threshold: float|None = None,
    color_scale: str = "linear",
    color_by: str = "samples",
    verbose: bool = True,
):
    """
    Plot p_harmful vs the ratio of optimization FLOPS to sampling FLOPS.

    Parameters
    ----------
    results : dict
        Results dictionary containing metric data and FLOPS information
    baseline : dict, optional
        Baseline results for comparison
    title : str
        Plot title
    sample_levels_to_plot : tuple[int, ...], optional
        Which sample levels to highlight
    metric : tuple[str, ...]
        Metric to plot on y-axis
    cumulative : bool
        Whether to use cumulative aggregation
    flops_per_step : callable, optional
        Function to compute optimization FLOPS per step
    threshold : float, optional
        Threshold for binary classification
    color_scale : str
        Color scale type ("linear" or "log")
    color_by : str
        What to color points by: "samples" (number of samples) or "total_flops" (total FLOP usage)
    verbose : bool
        Whether to print debug information
    """
    y = np.array(results[metric])  # (B, n_steps, n_samples)
    if threshold is not None:
        y = y > threshold
    n_runs, n_steps, n_total_samples = y.shape
    if sample_levels_to_plot is None:
        sample_levels_to_plot = generate_sample_sizes(n_total_samples)

    flops_sampling = np.array(results["flops_sampling"]) # (B, n_steps)
    if "flops" in results:
        flops_optimization = np.array(results["flops"]) # (B, n_steps)
    else:
        flops_optimization = np.zeros_like(flops_sampling) # (B, n_steps)
        if flops_per_step is not None:
            flops_optimization += flops_per_step(np.arange(flops_optimization.shape[1]))

    def subsample_and_aggregate_ratio(step_idx, sample_idx, cumulative, y, opt_flops, sampling_flops, rng):
        n_runs, n_steps, n_total_samples = y.shape
        opt_flop = np.mean(opt_flops[:, :step_idx+1].sum(axis=1))
        sampling_flop = np.mean(sampling_flops[:, step_idx]) * sample_idx

        # Calculate ratio (sampling / total), handle division by zero
        ratio = sampling_flop / (sampling_flop + opt_flop + 1e-9)

        if cumulative and step_idx > 0:
            samples_at_end = y[:, step_idx, rng.choice(n_total_samples, size=sample_idx, replace=False)].max(axis=-1)
            samples_up_to_now = y[:, :step_idx, rng.choice(n_total_samples, size=1, replace=False)].max(axis=1)[:, 0]
            values = np.stack([samples_up_to_now, samples_at_end], axis=1).max(axis=1)
            return (ratio, step_idx, sample_idx, values.mean(0), opt_flop, sampling_flop)
        return (ratio, step_idx, sample_idx, y[:, step_idx, rng.choice(n_total_samples, size=sample_idx, replace=False)].max(axis=-1).mean(axis=0), opt_flop, sampling_flop)

    def get_ratio_pts(y, opt_flops, sampling_flops):
        n_runs, n_steps, n_total_samples = y.shape
        rng = np.random.default_rng()
        pts = []  # (ratio, step, n_samples, mean_p, opt_flop, sampling_flop)
        for j in range(1, n_total_samples + 1, 1):
            for i in range(0, n_steps, 1):
                pts.append(subsample_and_aggregate_ratio(i, j, cumulative, y, opt_flops, sampling_flops, rng))
        pts = np.asarray(pts)
        return pts

    pts = get_ratio_pts(y, flops_optimization, flops_sampling)
    ratio, step_idx, n_samp, mean_p, opt_flop, sampling_flop = pts.T

    # Calculate total FLOPS for coloring option
    total_flop = opt_flop + sampling_flop

    # Filter out infinite ratios for plotting
    finite_mask = np.isfinite(ratio)
    ratio_finite = ratio[finite_mask]
    mean_p_finite = mean_p[finite_mask]
    n_samp_finite = n_samp[finite_mask]
    total_flop_finite = total_flop[finite_mask]

    plt.figure(figsize=(10, 6))

    # Choose color values based on color_by parameter
    if color_by == "samples":
        color_values = n_samp_finite
        color_label = "Number of Samples"
    elif color_by == "total_flops":
        color_values = total_flop_finite
        color_label = "Total FLOPS"
    else:
        raise ValueError(f"color_by must be 'samples' or 'total_flops', got '{color_by}'")

    # Scatter plot
    if color_scale == "log":
        color_norm = LogNorm()
    elif color_scale == "sqrt":
        color_norm = PowerNorm(gamma=0.5)  # Square root normalization
    else:
        color_norm = None

    sc = plt.scatter(ratio_finite, mean_p_finite, c=color_values, cmap="viridis", alpha=0.3, s=15, norm=color_norm)
    if color_by != "samples":
        plt.colorbar(sc, label=color_label)

    # Highlight specific sample levels (only when coloring by samples)
    if color_by == "samples":
        cmap = plt.get_cmap("viridis")
        if color_scale == "log":
            norm = LogNorm(n_samp.min(), n_samp.max())
        elif color_scale == "sqrt":
            norm = PowerNorm(gamma=0.5, vmin=n_samp.min(), vmax=n_samp.max())
        else:
            norm = plt.Normalize(n_samp.min(), n_samp.max())

        for j in sample_levels_to_plot:
            mask = (n_samp == j) & finite_mask
            if np.any(mask):
                color = cmap(norm(j))
                plt.scatter(ratio[mask], mean_p[mask],
                           color=color, s=50, alpha=0.8,
                           edgecolors='black', linewidth=0.5,
                           label=f"{j} samples")

    plt.xlabel("Sampling FLOPS / Total FLOPS", fontsize=14)
    if threshold is None:
        plt.ylabel("Mean p_harmful", fontsize=14)
    else:
        plt.ylabel(f"Max ASR (threshold: {threshold})", fontsize=12)

    plt.grid(True, alpha=0.3)
    plt.title(title, fontsize=16)

    # Add baseline if provided
    if baseline is not None:
        y_baseline = np.array(baseline[metric])
        if threshold is not None:
            y_baseline = y_baseline > threshold

        baseline_flops_sampling = np.array(baseline["flops_sampling"])
        if "flops" in baseline:
            baseline_flops_optimization = np.array(baseline["flops"])
        else:
            baseline_flops_optimization = np.zeros_like(baseline_flops_sampling)
            if flops_per_step is not None:
                baseline_flops_optimization += flops_per_step(np.arange(baseline_flops_optimization.shape[1]))

        baseline_pts = get_ratio_pts(y_baseline, baseline_flops_optimization, baseline_flops_sampling)
        baseline_ratio, _, baseline_n_samp, baseline_mean_p, _, _ = baseline_pts.T

        baseline_finite_mask = np.isfinite(baseline_ratio)
        plt.scatter(baseline_ratio[baseline_finite_mask], baseline_mean_p[baseline_finite_mask],
                   color="red", s=50, alpha=0.8, marker="^",
                   edgecolors='black', linewidth=0.5, label="Greedy")

    plt.xscale("log")
    plt.xlim(1e-5, 1)
    plt.ylim(bottom=0)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(f"evaluate/distributional_paper/flops_ratio_plots/{title}.pdf", bbox_inches='tight')
    plt.close()

    if verbose:
        print(f"FLOPS ratio range: {ratio_finite.min():.2e} to {ratio_finite.max():.2e}")
        print(f"Mean p_harmful range: {mean_p_finite.min():.4f} to {mean_p_finite.max():.4f}")


# ----------------------------------------------------------------------------------
# Pareto plots – simplified
# ----------------------------------------------------------------------------------
import numpy as np

MODELS = {
    "meta-llama/Meta-Llama-3.1-8B-Instruct": "Meta Llama 3.1 8B",
    "google/gemma-3-1b-it": "Gemma 3.1 1B",
    "GraySwanAI/Llama-3-8B-Instruct-RR": "Llama 3 CB",
}

FLOPS_PER_STEP = {
    "autodan": lambda s, c: 69845248149248 // num_model_params("Qwen/Qwen2.5-0.5B-Instruct") * c,
    "gcg":     lambda s, c: 14958709489152 // num_model_params("Qwen/Qwen2.5-0.5B-Instruct") * c,
    "beast":   lambda s, c: 10447045889280 // num_model_params("Qwen/Qwen2.5-0.5B-Instruct") * c,
    "pair":    lambda s, c: 83795198566400 + 78737584640 // num_model_params("Qwen/Qwen2.5-0.5B-Instruct") * c,
} # for 0.5B model

# Attack-specific configuration -----------------------------------------------------
ATTACKS = [
    ("pair", dict(
        title_suffix="PAIR",
        cumulative=True,
        sample_params=lambda: {
            "generation_config": {"num_return_sequences": 50, "temperature": 0.7},
        },
        baseline_params=lambda: {
            "generation_config": {"num_return_sequences": 1, "temperature": 0.0},
        },
    )),
    ("autodan", dict(
        title_suffix="AutoDAN",
        cumulative=False,
        sample_params=lambda: {
            "generation_config": {"num_return_sequences": 50, "temperature": 0.7},
            "early_stopping_threshold": 0,
        },
        baseline_params=lambda: {
            "generation_config": {"num_return_sequences": 1, "temperature": 0.0},
        },
    )),
    ("gcg", dict(
        title_suffix="GCG",
        cumulative=False,
        sample_params=lambda: {
            "generation_config": {"num_return_sequences": 50, "temperature": 0.7},
            "num_steps": 250,
            "loss": "ce",
            "token_selection": "default",
            "use_prefix_cache": True,
        },
        baseline_params=lambda: {
            "generation_config": {"num_return_sequences": 1, "temperature": 0.0},
            "num_steps": 250,
            "loss": "ce",
            "token_selection": "default",
            "use_prefix_cache": True,
        },
    )),
    ("gcg", dict(
        title_suffix="GCG 500",
        cumulative=False,
        sample_params=lambda: {
            "generation_config": {"num_return_sequences": 500, "temperature": 0.7},
            "num_steps": 250,
            "loss": "ce",
            "token_selection": "default",
            "use_prefix_cache": True,
        },
        baseline_params=lambda: {
            "generation_config": {"num_return_sequences": 1, "temperature": 0.0},
            "num_steps": 250,
            "loss": "ce",
            "token_selection": "default",
            "use_prefix_cache": True,
        },
    )),
    ("gcg", dict(
        title_suffix="GCG Entropy Loss",
        cumulative=False,
        sample_params=lambda: {
            "generation_config": {"num_return_sequences": 50, "temperature": 0.7},
            "num_steps": 250,
            "loss": "entropy_adaptive",
            "token_selection": "default",
            "use_prefix_cache": True,
        },
        baseline_params=lambda: {
            "generation_config": {"num_return_sequences": 1, "temperature": 0.0},
            "num_steps": 250,
            "loss": "ce",
            "token_selection": "default",
            "use_prefix_cache": True,
        },
    )),
    ("bon", dict(
        title_suffix="BoN",
        cumulative=False,
        sample_params=lambda: {"num_steps": 1000, "generation_config": {"temperature": 0.7}},
        baseline_params=lambda: {
            # BoN's baseline is *Direct* with one deterministic sample
            "generation_config": {"num_return_sequences": 1, "temperature": 0.0},
        },
        baseline_attack="direct",
        postprocess=lambda data, metric: data.__setitem__(
            metric, np.array(data[metric]).transpose(0, 2, 1)
        ),
    )),
    ("bon", dict(
        title_suffix="BoN Repro",
        cumulative=False,
        sample_params=lambda: {"num_steps": 1000, "generation_config": {"temperature": 1.0}},
        baseline_params=lambda: {
            # BoN's baseline is *Direct* with one deterministic sample
            "generation_config": {"num_return_sequences": 1, "temperature": 0.0},
        },
        baseline_attack="direct",
        postprocess=lambda data, metric: data.__setitem__(
            metric, np.array(data[metric]).transpose(0, 2, 1)
        ),
    )),
    ("direct", dict(
        title_suffix="Direct",
        cumulative=True,
        sample_params=lambda: {
            "generation_config": {"num_return_sequences": 1000, "temperature": 0.7},
        },
        baseline_params=lambda: {
            "generation_config": {"num_return_sequences": 1, "temperature": 0.0},
        },
        skip_if_empty=True,  # gracefully continue if no paths were found
    )),
    ("direct", dict(
        title_suffix="Direct temp 1.0",
        cumulative=True,
        sample_params=lambda: {
            "generation_config": {"num_return_sequences": 1000, "temperature": 1.0},
        },
        baseline_params=lambda: {
            "generation_config": {"num_return_sequences": 1, "temperature": 0.0},
        },
        skip_if_empty=True,  # gracefully continue if no paths were found
    )),
]

METRIC = ("scores", "strong_reject", "p_harmful")
GROUP_BY = {"model", "attack_params"}
DATASET_IDX = list(range(50))


# Helper ---------------------------------------------------------------------------
def run_attack(
    model: str,
    model_title: str,
    atk_name: str,
    cfg: dict,
):
    print("Attack:", atk_name, cfg["title_suffix"])

    # ---------- helper to fetch data ----------
    def fetch(attack: str, attack_params: dict):
        filter_by = dict(
            model=model,
            attack=attack,
            attack_params=attack_params,
            dataset_params={"idx": DATASET_IDX},
        )
        paths = get_filtered_and_grouped_paths(filter_by, GROUP_BY)
        results = collect_results(paths, infer_sampling_flops=True)
        assert len(results) == 1, len(results)
        return list(results.values())[0]

    # ---------- sampled run ----------
    sampled_data = fetch(cfg.get("attack_override", atk_name), cfg["sample_params"]())

    # Attack-specific post-processing
    if post := cfg.get("postprocess"):
        post(sampled_data, METRIC)

    # ---------- baseline run ----------
    baseline_attack = cfg.get("baseline_attack", atk_name)
    baseline_data = fetch(baseline_attack, cfg["baseline_params"]())

    # ---------- plot ----------
    pareto_plot(
        sampled_data,
        baseline_data,
        title=f"{model_title} {cfg['title_suffix']}",
        cumulative=cfg["cumulative"],
        metric=METRIC,
        flops_per_step=lambda x: FLOPS_PER_STEP.get(atk_name, lambda x, c: 0)(x, num_model_params(model)),
        threshold=None,
        color_scale="sqrt",
    )


def run_attack_flops_ratio(
    model: str,
    model_title: str,
    atk_name: str,
    cfg: dict,
):
    print("FLOPS Ratio Attack:", atk_name)

    # ---------- helper to fetch data ----------
    def fetch(attack: str, attack_params: dict):
        filter_by = dict(
            model=model,
            attack=attack,
            attack_params=attack_params,
            dataset_params={"idx": DATASET_IDX},
        )
        paths = get_filtered_and_grouped_paths(filter_by, GROUP_BY)
        results = collect_results(paths, infer_sampling_flops=True)
        assert len(results) == 1, len(results)
        return list(results.values())[0]

    # ---------- sampled run ----------
    sampled_data = fetch(cfg.get("attack_override", atk_name), cfg["sample_params"]())

    # Attack-specific post-processing
    if post := cfg.get("postprocess"):
        post(sampled_data, METRIC)

    # ---------- baseline run ----------
    baseline_attack = cfg.get("baseline_attack", atk_name)
    baseline_data = fetch(baseline_attack, cfg["baseline_params"]())

    # ---------- plot ----------
    flops_ratio_plot(
        sampled_data,
        baseline_data,
        title=f"{model_title} {cfg['title_suffix']} FLOPS Ratio",
        cumulative=cfg["cumulative"],
        metric=METRIC,
        flops_per_step=lambda x: FLOPS_PER_STEP.get(atk_name, lambda x, c: 0)(x, num_model_params(model)),
        threshold=None,
        # color_by="total_flops",
        color_scale="sqrt",
    )


# # Main loop ------------------------------------------------------------------------
for model_key, model_title in MODELS.items():
    print("Model:", model_key)
    for atk_name, atk_cfg in ATTACKS:
        try:
            run_attack(model_key, model_title, atk_name, atk_cfg)
        except Exception as e:
            print(f"Error running attack {atk_name}, atk_cfg: {atk_cfg['title_suffix']}: {e}")


# FLOPS Ratio plots main loop ------------------------------------------------------
print("\n" + "="*80)
print("GENERATING FLOPS RATIO PLOTS")
print("="*80)

for model_key, model_title in MODELS.items():
    print("Model:", model_key)
    for atk_name, atk_cfg in ATTACKS:
        try:
            run_attack_flops_ratio(model_key, model_title, atk_name, atk_cfg)
        except Exception as e:
            print(f"Error running FLOPS ratio attack {atk_name}, atk_cfg: {atk_cfg['title_suffix']}: {e}")


# Helper ---------------------------------------------------------------------------
def run_attack_2(
    model: str,
    model_title: str,
    atk_name: str,
    cfg: dict,
):
    print("Attack:", atk_name)

    # ---------- helper to fetch data ----------
    def fetch(attack: str, attack_params: dict):
        filter_by = dict(
            model=model,
            attack=attack,
            attack_params=attack_params,
            dataset_params={"idx": DATASET_IDX},
        )
        paths = get_filtered_and_grouped_paths(filter_by, GROUP_BY)
        results = collect_results(paths, infer_sampling_flops=True)
        assert len(results) == 1, len(results)
        return list(results.values())[0]

    # ---------- sampled run ----------
    sampled_data = fetch(cfg.get("attack_override", atk_name), cfg["sample_params"]())

    # Attack-specific post-processing
    if post := cfg.get("postprocess"):
        post(sampled_data, METRIC)

    data = np.array(sampled_data[("scores", "strong_reject", "p_harmful")])[:, 0]
    # Create histogram plot
    plt.figure(figsize=(10, 6))
    plt.hist(data.flatten(), bins=100, alpha=0.7, edgecolor='black')
    plt.xlabel('p_harmful', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.title(f'{model_title} - {atk_name} - p_harmful Distribution', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save the plot
    filename = f"evaluate/distributional_paper/histograms/{model_title}_{cfg['title_suffix']}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

for model_key, model_title in MODELS.items():
    print("Model:", model_key)
    for atk_name, atk_cfg in ATTACKS:
        if atk_name != "direct": continue
        try:
            run_attack_2(model_key, model_title, atk_name, atk_cfg)
        except Exception as e:
            print(f"Error running attack {atk_name}, atk_cfg: {atk_cfg['title_suffix']}: {e}")


# Helper ---------------------------------------------------------------------------
def run_attack_2(
    model: str,
    model_title: str,
    atk_name: str,
    cfg: dict,
):
    print("Attack:", atk_name)

    # ---------- helper to fetch data ----------
    def fetch(attack: str, attack_params: dict):
        filter_by = dict(
            model=model,
            attack=attack,
            attack_params=attack_params,
            dataset_params={"idx": DATASET_IDX},
        )
        paths = get_filtered_and_grouped_paths(filter_by, GROUP_BY)
        results = collect_results(paths, infer_sampling_flops=True)
        assert len(results) == 1, f"Should only have exactly one type of result, got {len(results)}, {list(results.keys())}"
        return list(results.values())[0]

    # ---------- sampled run ----------
    sampled_data = fetch(cfg.get("attack_override", atk_name), cfg["sample_params"]())

    # Attack-specific post-processing
    if post := cfg.get("postprocess"):
        post(sampled_data, METRIC)

    plt.figure(figsize=(10, 6))
    data_list = []
    positions = []
    for i in range(0, 250, 25):
        data = np.array(sampled_data[("scores", "strong_reject", "p_harmful")])[:, i]
        data_list.append(data.flatten())
        positions.append(i)

    # Create violin plot
    plt.violinplot(data_list, positions=positions, widths=20, showmeans=True, showmedians=True)
    plt.xlabel('Step', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.title(f'{model_title} - {atk_name} - p_harmful Distribution', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save the plot
    filename = f"evaluate/distributional_paper/histograms/{model_title}_{cfg['title_suffix']}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

for model_key, model_title in MODELS.items():
    print("Model:", model_key)
    for atk_name, atk_cfg in ATTACKS:
        if atk_name != "gcg": continue
        try:
            run_attack_2(model_key, model_title, atk_name, atk_cfg)
        except Exception as e:
            print(f"Error running attack {atk_name}, atk_cfg: {atk_cfg['title_suffix']}: {e}")
