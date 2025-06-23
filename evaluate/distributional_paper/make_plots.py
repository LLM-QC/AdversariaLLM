import matplotlib.pyplot as plt
import scienceplots
plt.style.use("science")

import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import rankdata
from scipy.interpolate import interp1d
from matplotlib.colors import LogNorm, PowerNorm
from scipy.interpolate import griddata
import logging
logging.basicConfig(level=logging.INFO)

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


# ------------------------------------------------------------------
# Common helper functions
# ------------------------------------------------------------------
def fetch_data(model: str, attack: str, attack_params: dict, dataset_idx: list[int], group_by: set[str]):
    """Common data fetching logic used across all plotting functions."""
    filter_by = dict(
        model=model,
        attack=attack,
        attack_params=attack_params,
        dataset_params={"idx": dataset_idx},
    )
    paths = get_filtered_and_grouped_paths(filter_by, group_by)

    results = collect_results(paths, infer_sampling_flops=True)
    assert len(results) == 1, f"Should only have exactly one type of result, got {len(results)}, {list(results.keys())}"
    return list(results.values())[0]


def preprocess_data(results: dict[str, np.ndarray], metric: tuple[str, ...], threshold: float|None, flops_per_step_fn):
    """Common data preprocessing logic."""
    y = np.array(results[metric])  # (B, n_steps, n_samples)
    if threshold is not None:
        y = y > threshold

    flops_sampling_prefill_cache = np.array(results["flops_sampling_prefill_cache"]) # (B, n_steps)
    flops_sampling_generation = np.array(results["flops_sampling_generation"]) # (B, n_steps)

    if "flops" in results:
        flops_optimization = np.array(results["flops"]) # (B, n_steps)
    else:
        flops_optimization = np.zeros_like(flops_sampling_generation) # (B, n_steps)
        if flops_per_step_fn is not None:
            flops_optimization += flops_per_step_fn(np.arange(flops_optimization.shape[1]))

    return y, flops_optimization, flops_sampling_prefill_cache, flops_sampling_generation


def subsample_and_aggregate(step_idx: int, sample_idx: int, cumulative: bool, y: np.ndarray,
                            opt_flops: np.ndarray, sampling_prefill_flops: np.ndarray,
                            sampling_generation_flops: np.ndarray, rng: np.random.Generator,
                            return_ratio: bool = False, n_smoothing: int = 1):
    """
    Unified subsampling and aggregation function.

    Parameters
    ----------
    return_ratio : bool
        If True, returns sampling ratio instead of total cost
    n_smoothing : int
        Number of smoothing iterations for variance reduction
    """
    n_runs, n_steps, n_total_samples = y.shape
    opt_flop = np.mean(opt_flops[:, :step_idx+1].sum(axis=1))
    sampling_flop = np.mean(sampling_generation_flops[:, step_idx]) * sample_idx + np.mean(sampling_prefill_flops[:, step_idx])
    total_flop = opt_flop + sampling_flop

    # Calculate value with smoothing
    values = []
    for _ in range(n_smoothing):
        #
        rng = np.random.default_rng(sample_idx+n_smoothing)
        if cumulative and step_idx > 0:
            samples_up_to_now = y[:, :step_idx, rng.choice(n_total_samples, size=1, replace=False)].max(axis=1)[:, 0]
            samples_at_end = y[:, step_idx, rng.choice(n_total_samples, size=sample_idx, replace=False)].max(axis=-1)
            values.append(np.stack([samples_up_to_now, samples_at_end], axis=1).max(axis=1).mean(axis=0))
        else:
            values.append(y[:, step_idx, rng.choice(n_total_samples, size=sample_idx, replace=False)].max(axis=-1).mean(axis=0))

    mean_value = np.mean(values)

    if return_ratio:
        ratio = sampling_flop / (total_flop + 1e-9)
        return (ratio, step_idx, sample_idx, mean_value, opt_flop, sampling_flop)
    else:
        return (total_flop, step_idx, sample_idx, mean_value)


def get_points(y: np.ndarray, opt_flops: np.ndarray, sampling_prefill_flops: np.ndarray,
               sampling_generation_flops: np.ndarray, return_ratio: bool = False,
               n_smoothing: int = 1, cumulative: bool = False):
    """Generate points for plotting with optional ratio calculation."""
    n_runs, n_steps, total_samples = y.shape
    rng = np.random.default_rng(42)  # Fixed seed for reproducibility
    pts = []

    for j in range(1, total_samples + 1, 1):
        for i in range(0, n_steps, 1):
            pts.append(subsample_and_aggregate(
                i, j, cumulative, y, opt_flops, sampling_prefill_flops,
                sampling_generation_flops, rng, return_ratio, n_smoothing
            ))

    return np.asarray(pts)


def setup_color_normalization(color_scale: str, values: np.ndarray):
    """Setup color normalization based on scale type."""
    if color_scale == "log":
        return LogNorm(values.min(), values.max())
    elif color_scale == "sqrt":
        return PowerNorm(gamma=0.5, vmin=values.min(), vmax=values.max())
    else:
        return plt.Normalize(values.min(), values.max())


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
    """
    y, flops_optimization, flops_sampling_prefill_cache, flops_sampling_generation = preprocess_data(
        results, metric, threshold, flops_per_step
    )
    n_runs, n_steps, n_total_samples = y.shape
    if sample_levels_to_plot is None:
        sample_levels_to_plot = generate_sample_sizes(n_total_samples)

    pts = get_points(y, flops_optimization, flops_sampling_prefill_cache, flops_sampling_generation,
                     return_ratio=False, cumulative=cumulative)
    cost, step_idx, n_samp, mean_p = pts.T
    max_cost = max(cost)
    if x_scale == "log":
        x_interp = np.logspace(11, np.log10(max_cost)+0.001, n_x_points)
    else:
        x_interp = np.linspace(0, max_cost+1, n_x_points)

    # Create figure with subplots: main plot + 2x2 grid on the right
    fig = plt.figure(figsize=(18, 8))

    # Main Pareto plot (left half, spanning both rows)
    ax1 = plt.subplot2grid((2, 4), (0, 0), colspan=2, rowspan=2)

    # ---------- scatter all points ----------
    color_norm = setup_color_normalization(color_scale, n_samp)
    if plot_points:
        sc = plt.scatter(cost, mean_p, c=n_samp, cmap="viridis", alpha=0.15, s=3, norm=color_norm)
    plt.xlabel("Cost (FLOPS (optimization + sampling))", fontsize=14)
    if threshold is None:
        plt.ylabel(r"$\overline{p_{harmful}}$", fontsize=14)
    else:
        plt.ylabel(r"$\overline{{ASR}}\quad (p_{{harmful}} \geq {threshold})$".format(threshold=threshold), fontsize=14)

    # ---------- overlay Pareto frontiers ----------
    cmap = plt.get_cmap("viridis")
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
                    pts.append(subsample_and_aggregate(i, j, cumulative, y, flops_optimization, flops_sampling_prefill_cache, flops_sampling_generation, rng))

                pts = np.asarray(pts)
                cost, _, _, mean_p = pts.T

                fx, fy = _pareto_frontier(cost, mean_p, method=frontier_method)
                xs.append(fx)
                ys.append(fy)
            y_interp = [interp1d(x_, y_, kind="previous", bounds_error=False, fill_value=(0, max(y_)))(x_interp) for x_, y_ in zip(xs, ys)]

            color = cmap(color_norm(j))
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
                    pts.append(subsample_and_aggregate(i, j, cumulative, y, flops_optimization, flops_sampling_prefill_cache, flops_sampling_generation, rng))

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
        color = [cmap(color_norm(argmax[i])) for i in range(len(argmax)) if nonzero_mask[i]]
        plt.scatter(x_interp[nonzero_mask], y_envelope[nonzero_mask], c=color, s=2)

    title_suffix = ""

    # Handle baseline data
    baseline_max_asr = 0
    baseline_frontier_data = None

    if baseline is not None:
        y_baseline, baseline_flops_optimization, baseline_flops_sampling_prefill_cache, baseline_flops_sampling_generation = preprocess_data(
            baseline, metric, threshold, flops_per_step
        )

        if y_baseline is not None:
            title_suffix = f" ({n_runs}, {y_baseline.shape[0]})"
            if verbose:
                logging.info(f"{n_runs} for main")
                logging.info(f"{y_baseline.shape[0]} for baseline")
            n_runs_baseline, n_steps_baseline, n_total_samples_baseline = y_baseline.shape
            assert n_total_samples_baseline == 1

            pts = get_points(y_baseline, baseline_flops_optimization, baseline_flops_sampling_prefill_cache,
                           baseline_flops_sampling_generation, return_ratio=False, cumulative=cumulative)
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
                    label=f"Baseline",
                    color="r",
                )

    plt.title(title + title_suffix)
    plt.grid(True, alpha=0.3)
    plt.ylim(bottom=0)
    plt.xscale(x_scale)
    plt.legend(title="Frontiers", loc="upper left" if x_scale == "log" else "lower right")

    # ---------- Bar Chart 1: Max ASR Comparison (Vertical Slice) ----------
    ax2 = plt.subplot2grid((2, 4), (0, 2))

    methods = []
    max_asrs = []
    colors = []

    # Add baseline (delta = 0 for baseline)
    if baseline_frontier_data is not None:
        methods.append("Baseline")
        max_asrs.append(0.0)  # Delta from itself is 0
        colors.append("red")

    # Add sampling methods (calculate delta from baseline)
    for j in sample_levels_to_plot:
        if j in frontier_data:
            methods.append(f"{j} samples" if j != 1 else "1 sample")
            delta_asr = frontier_data[j]['max_asr'] - baseline_max_asr if baseline_frontier_data is not None else 0
            max_asrs.append(delta_asr)
            colors.append(frontier_data[j]['color'])

    if methods:
        bars = plt.bar(methods, max_asrs, color=colors, alpha=0.7, edgecolor='black')
        if threshold is None:
            plt.ylabel(r"$\Delta$ $p_{harmful}$", fontsize=14)
            # plt.title(r"$p_{harmful}$ vs. \#samples", fontsize=14)
        else:
            # plt.title(r"$\overline{{ASR}}$ vs. \#samples".format(threshold=threshold), fontsize=14)
            plt.ylabel(r"$\Delta$ $\overline{{ASR}}\quad (p_{{harmful}} \geq {threshold})$".format(threshold=threshold), fontsize=14)
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
            va = 'bottom' if value >= 0 else 'top'
            offset = (0, offset_pt if value >= 0 else -offset_pt)

            ax2.annotate(f'{value:.3f}',
                        xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=offset,
                        textcoords='offset points',
                        ha='center', va=va, fontsize=10)

    # ---------- Bar Chart 2: FLOPS Efficiency to Reach Baseline ASR (Horizontal Slice) ----------
    ax3 = plt.subplot2grid((2, 4), (0, 3))

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
            methods_flops.insert(0, "Baseline")
            flops_required.insert(0, baseline_flops)
            colors_flops.insert(0, "red")

        if methods_flops:
            bars = plt.bar(methods_flops, flops_required, color=colors_flops, alpha=0.7, edgecolor='black')
            plt.ylabel(r"FLOPS for Baseline $p_{harmful}$" + f" ( = {target_asr:.3f})", fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.yscale('log')
            plt.grid(True, alpha=0.3, axis='y')
            # Increase ylim by
            ymin, ymax = plt.ylim()
            import math
            margin = ((math.log10(ymax) - math.log10(ymin)) * 0.2)
            plt.ylim(ymin, ymax * (1+margin))

            # --- constant 5-point vertical gap ---
            for bar, value in zip(bars, flops_required):
                ax3.annotate(f'{value:.2e}',
                            xy=(bar.get_x() + bar.get_width()/2, value),   # anchor at top of bar
                            xytext=(0, 5),                                  # 5 points straight up
                            textcoords='offset points',
                            ha='center', va='bottom', rotation=45, fontsize=9)

    # ---------- Bar Chart 3: Speedup vs Baseline (Bottom Right) ----------
    ax4 = plt.subplot2grid((2, 4), (1, 3))

    # Create speedup plot
    speedup_methods = []
    speedups = []
    speedup_colors = []

    # Calculate speedup for each method (baseline_flops / method_flops)
    baseline_flops = flops_required[0] if methods_flops[0] == "Baseline" else None

    if baseline_flops is not None:
        for i, (method, flops, color) in enumerate(zip(methods_flops, flops_required, colors_flops)):
            if method != "Baseline":  # Skip baseline itself
                speedup = baseline_flops / flops if flops > 0 else 0
                speedup_methods.append(method)
                speedups.append(speedup)
                speedup_colors.append(color)

        if speedup_methods:
            bars = plt.bar(speedup_methods, speedups, color=speedup_colors, alpha=0.7, edgecolor='black')
            plt.ylabel("Speedup (FLOPS) vs Baseline", fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, alpha=0.3, axis='y')

            # Add horizontal line at y=1 for reference
            plt.axhline(y=1, color='red', linestyle='--', alpha=0.7, linewidth=1)

            # Increase ylim by small margin
            ymin, ymax = plt.ylim()
            margin = (ymax - ymin) * 0.05
            plt.ylim(max(0, ymin - margin), ymax + margin)

            # Add value labels on bars
            for bar, value in zip(bars, speedups):
                ax4.annotate(f'{value:.2f}x',
                            xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                            xytext=(0, 5),
                            textcoords='offset points',
                            ha='center', va='bottom', fontsize=10)

    # ---------- Line Plot 4: Continuous FLOPS to Reach Baseline ASR (Bottom Left) ----------
    ax5 = plt.subplot2grid((2, 4), (1, 2))

    if baseline_frontier_data is not None and baseline_max_asr > 0:
        target_asr = baseline_max_asr

        # Generate continuous range of sample counts
        sample_range = range(1, n_total_samples + 1)
        continuous_flops = []
        continuous_samples = []

        # Calculate frontier data for all sample counts (not just sample_levels_to_plot)
        rng_continuous = np.random.default_rng()
        n_smoothing_continuous = 10  # Reduced for performance

        for j in sample_range:
            xs = []
            ys = []
            for _ in range(n_smoothing_continuous):
                pts = []
                for i in range(0, n_steps, 1):
                    pts.append(subsample_and_aggregate(i, j, cumulative, y, flops_optimization,
                                                     flops_sampling_prefill_cache, flops_sampling_generation, rng_continuous))

                pts = np.asarray(pts)
                cost, _, _, mean_p = pts.T

                fx, fy = _pareto_frontier(cost, mean_p, method=frontier_method)
                xs.append(fx)
                ys.append(fy)

            # Interpolate and average
            y_interp = [interp1d(x_, y_, kind="previous", bounds_error=False, fill_value=(0, max(y_)))(x_interp)
                       for x_, y_ in zip(xs, ys)]
            y_mean = np.mean(y_interp, axis=0)

            # Find minimum FLOPS where ASR >= target_asr
            nonzero_mask = y_mean > 0
            if np.any(nonzero_mask):
                y_vals = y_mean[nonzero_mask]
                x_vals = x_interp[nonzero_mask]

                valid_indices = y_vals >= target_asr
                if np.any(valid_indices):
                    min_flops = np.min(x_vals[valid_indices])
                    continuous_flops.append(min_flops)
                    continuous_samples.append(j)

        if continuous_flops:
            # Plot the continuous line
            plt.plot(continuous_samples, continuous_flops, 'b-', linewidth=2, alpha=0.8, label='All Samples')

            # Highlight the baseline point
            if baseline_frontier_data['x'].size > 0:
                baseline_y_vals = baseline_frontier_data['y']
                baseline_x_vals = baseline_frontier_data['x']
                baseline_valid_indices = baseline_y_vals >= target_asr
                if np.any(baseline_valid_indices):
                    baseline_flops = np.min(baseline_x_vals[baseline_valid_indices])
                    plt.axhline(y=baseline_flops, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Baseline')

            # Highlight the discrete sample levels from the bar chart
            for j in sample_levels_to_plot:
                if j in [s for s in continuous_samples]:
                    idx = continuous_samples.index(j)
                    color = cmap(color_norm(j))
                    plt.scatter(j, continuous_flops[idx], color=color, s=60, alpha=0.9,
                              edgecolors='black', linewidth=0.5, zorder=5)

            plt.xlabel("Number of Samples", fontsize=12)
            plt.ylabel("FLOPS to Reach Baseline ASR", fontsize=12)
            plt.xscale('log')
            plt.yscale('log')
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=10)

            # Set reasonable x-axis limits
            plt.xlim(1, n_total_samples)

            # Increase ylim by small margin
            ymin, ymax = plt.ylim()
            import math
            margin = ((math.log10(ymax) - math.log10(ymin)) * 0.1)
            plt.ylim(ymin / (1+margin), ymax * (1+margin))

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
    """
    y, flops_optimization, flops_sampling_prefill_cache, flops_sampling_generation = preprocess_data(
        results, metric, threshold, flops_per_step
    )
    n_runs, n_steps, n_total_samples = y.shape
    if sample_levels_to_plot is None:
        sample_levels_to_plot = generate_sample_sizes(n_total_samples)

    pts = get_points(y, flops_optimization, flops_sampling_prefill_cache, flops_sampling_generation,
                     return_ratio=True, cumulative=cumulative)
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

    # Create dual color encoding: hue based on samples, strength based on total FLOPS
    # Normalize sample counts for hue
    sample_norm = setup_color_normalization("linear", n_samp_finite)
    # Normalize total FLOPS for alpha/strength
    flops_norm = setup_color_normalization(color_scale, total_flop_finite)

    # Get base colors from viridis colormap based on sample count
    cmap = plt.get_cmap("viridis")
    base_colors = cmap(sample_norm(n_samp_finite))

    # Scatter plot with dual color encoding
    sc = plt.scatter(ratio_finite, mean_p_finite, c=base_colors, s=15, alpha=0.05)

    # Create custom colorbar for samples (hue)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=sample_norm)
    sm.set_array([])

    # Highlight specific sample levels
    for j in sample_levels_to_plot:
        mask = (n_samp == j) & finite_mask
        if np.any(mask):
            # Use the same dual coloring for highlighted points
            highlight_base_color = cmap(sample_norm(j))
            # Create colors for this sample level

            plt.scatter(ratio[mask], mean_p[mask],
                       c=highlight_base_color, s=50, alpha=0.9,
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
        y_baseline, baseline_flops_optimization, baseline_flops_sampling_prefill_cache, baseline_flops_sampling_generation = preprocess_data(
            baseline, metric, threshold, flops_per_step
        )

        baseline_pts = get_points(y_baseline, baseline_flops_optimization, baseline_flops_sampling_prefill_cache,
                                baseline_flops_sampling_generation, return_ratio=True, cumulative=cumulative)
        baseline_ratio, _, baseline_n_samp, baseline_mean_p, _, _ = baseline_pts.T

        baseline_finite_mask = np.isfinite(baseline_ratio) & np.isfinite(baseline_mean_p)
        if np.any(baseline_finite_mask):
            # For baseline, just plot the raw ratios
            plt.scatter(baseline_ratio[baseline_finite_mask], baseline_mean_p[baseline_finite_mask],
                       color="red", s=60, alpha=0.9, marker="^",
                       edgecolors='black', linewidth=0.5, label="Baseline", zorder=6)

    # Add subtle iso-FLOP lines (fitted quadratics)
    if n_total_samples == 500:  # Only if we have enough data points
        # Select 5 FLOP levels spanning the range
        flop_min, flop_max = np.min(total_flop_finite), np.max(total_flop_finite)
        iso_flop_levels = np.logspace(np.log10(flop_min), np.log10(flop_max), 5)

        for i, flop_level in enumerate(iso_flop_levels):
            # Find points near this FLOP level (within 20% tolerance)
            tolerance = 0.15
            near_flop_mask = np.abs(total_flop_finite - flop_level) / flop_level < tolerance

            if np.sum(near_flop_mask) >= 3:  # Need at least 3 points for quadratic fit
                x_iso = ratio_finite[near_flop_mask]
                y_iso = mean_p_finite[near_flop_mask]

                # Sort by x for smooth curve
                sort_idx = np.argsort(x_iso)
                x_iso_sorted = x_iso[sort_idx]
                y_iso_sorted = y_iso[sort_idx]

                # Fit quadratic in log-space for x
                try:
                    log_x = np.log10(x_iso_sorted)
                    coeffs = np.polyfit(log_x, y_iso_sorted, 2)

                    # Generate smooth curve
                    x_smooth = np.logspace(np.log10(x_iso_sorted.min()) - 0.25,
                                         np.log10(x_iso_sorted.max()) + 0.25, 50)
                    log_x_smooth = np.log10(x_smooth)
                    y_smooth = np.polyval(coeffs, log_x_smooth)

                    # Plot the iso-FLOP line with label for first one only
                    label = "Iso-FLOP lines" if i == 0 else None
                    plt.plot(x_smooth, y_smooth, '--', color='gray', alpha=0.8,
                            linewidth=1, zorder=1, label=label)

                    # Add text annotation for FLOP level at the end of the curve
                    if len(x_smooth) > 0 and len(y_smooth) > 0:
                        # Find a good position for the text (middle of the curve)
                        mid_idx = 0
                        text_x = x_smooth[mid_idx]
                        text_y = y_smooth[mid_idx]

                        # Format FLOP level in scientific notation
                        flop_text = f"{flop_level:.1e}"
                        plt.text(text_x, text_y, flop_text, fontsize=8, alpha=0.8,
                                ha='center', va='top', color='black',
                                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                         alpha=0.7, edgecolor='none'))

                except (np.linalg.LinAlgError, ValueError) as e:
                    raise e
    plt.xscale("log")
    plt.xlim(1e-5, 1)
    plt.ylim(bottom=0)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(f"evaluate/distributional_paper/flops_ratio_plots/{title}.pdf", bbox_inches='tight')
    plt.close()

    if verbose:
        logging.info(f"FLOPS ratio range: {ratio_finite.min():.2e} to {ratio_finite.max():.2e}")
        logging.info(f"Mean p_harmful range: {mean_p_finite.min():.4f} to {mean_p_finite.max():.4f}")
        logging.info(f"Total FLOPS range: {total_flop_finite.min():.2e} to {total_flop_finite.max():.2e}")


def ideal_ratio_plot(
    results: dict[str,np.ndarray],
    baseline: dict[str,np.ndarray] | None = None,
    title: str = "Ideal Sampling FLOPS Ratio",
    sample_levels_to_plot: tuple[int, ...]|None = None,
    metric: tuple[str, ...] = ('scores', 'strong_reject', 'p_harmful'),
    cumulative: bool = False,
    flops_per_step: int | None = None,
    threshold: float|None = None,
    n_p_harmful_points: int = 100,
    verbose: bool = True,
):
    """
    Plot the ideal sampling FLOPS ratio for achieving different levels of harmfulness.
    For each p_harmful level, finds the point that achieves that level with minimum total FLOPS
    and plots the corresponding sampling ratio.
    """
    y, flops_optimization, flops_sampling_prefill_cache, flops_sampling_generation = preprocess_data(
        results, metric, threshold, flops_per_step
    )

    n_smoothing = 50
    pts = get_points(y, flops_optimization, flops_sampling_prefill_cache, flops_sampling_generation,
                     return_ratio=True, n_smoothing=n_smoothing, cumulative=cumulative)
    ratio, step_idx, n_samp, mean_p, total_flop = pts.T[:5]

    # Filter out infinite ratios and invalid points
    finite_mask = np.isfinite(ratio) & np.isfinite(mean_p) & np.isfinite(total_flop)
    ratio_finite = ratio[finite_mask]
    mean_p_finite = mean_p[finite_mask]
    total_flop_finite = total_flop[finite_mask]

    # Create p_harmful levels to evaluate
    p_harmful_min = np.min(mean_p_finite)
    p_harmful_max = np.max(mean_p_finite)
    p_harmful_levels = np.linspace(p_harmful_min, p_harmful_max, n_p_harmful_points)

    # Find ideal ratio for each p_harmful level
    ideal_ratios = []
    max_ratios = []  # Track maximum ratios explored at each level
    min_ratios = []  # Track minimum ratios explored at each level
    achieved_p_levels = []

    for p_level in p_harmful_levels:
        # Find all points that achieve at least this p_harmful level
        achieving_mask = mean_p_finite >= p_level

        if np.any(achieving_mask):
            # Among achieving points, find the one with minimum total FLOPS
            achieving_flops = total_flop_finite[achieving_mask]
            achieving_ratios = ratio_finite[achieving_mask]

            min_flops_idx = np.argmin(achieving_flops)
            ideal_ratio = achieving_ratios[min_flops_idx]

            # Find the maximum and minimum ratios explored at this level
            max_ratio = np.max(achieving_ratios)
            min_ratio = np.min(achieving_ratios)

            ideal_ratios.append(ideal_ratio)
            max_ratios.append(max_ratio)
            min_ratios.append(min_ratio)
            achieved_p_levels.append(p_level)

    ideal_ratios = np.array(ideal_ratios)
    max_ratios = np.array(max_ratios)
    min_ratios = np.array(min_ratios)
    achieved_p_levels = np.array(achieved_p_levels)

    plt.figure(figsize=(12, 8))

    # Create the FLOP landscape: interpolated surface with color indicating total FLOPS
    # Use raw ratios instead of normalized ones
    landscape_p_harmful = []
    landscape_ratios = []
    landscape_total_flops = []

    for i, (p_val, ratio_val, flop_val) in enumerate(zip(mean_p_finite, ratio_finite, total_flop_finite)):
        landscape_p_harmful.append(p_val)
        landscape_ratios.append(ratio_val)
        landscape_total_flops.append(flop_val)

    landscape_p_harmful = np.array(landscape_p_harmful)
    landscape_ratios = np.array(landscape_ratios)
    landscape_total_flops = np.array(landscape_total_flops)

    # Create interpolated surface
    # Define grid for interpolation
    p_grid = np.linspace(np.min(landscape_p_harmful), np.max(landscape_p_harmful), 100)
    ratio_grid = np.logspace(np.log10(1e-5), np.log10(1.0), 100)
    P_grid, Ratio_grid = np.meshgrid(p_grid, ratio_grid)

    # Interpolate FLOPS values onto the grid
    try:
        flops_grid = griddata(
            (landscape_p_harmful, landscape_ratios),
            landscape_total_flops,
            (P_grid, Ratio_grid),
            method='linear',
            fill_value=np.nan
        )
    except Exception as e:
        raise ValueError(f"Error interpolating FLOPS values.")

    # Create mask to only show values within the explored bounds
    mask = np.ones_like(flops_grid, dtype=bool)
    for i, p_val in enumerate(p_grid):
        # Find the closest p_harmful level to get min/max bounds
        closest_idx = np.argmin(np.abs(achieved_p_levels - p_val))
        if closest_idx < len(min_ratios) and closest_idx < len(max_ratios):
            min_bound = min_ratios[closest_idx]
            max_bound = max_ratios[closest_idx]

            # Mask out values outside the bounds
            ratio_col = Ratio_grid[:, i]
            outside_bounds = (ratio_col < min_bound) | (ratio_col > max_bound)
            mask[outside_bounds, i] = False

    # Apply mask
    flops_grid_masked = np.where(mask, flops_grid, np.nan)

    # Create contour plot of the FLOP landscape
    contour = plt.contourf(P_grid, Ratio_grid, flops_grid_masked, levels=50,
                          cmap='plasma', alpha=0.8, extend='both')

    # Add colorbar for total FLOPS
    cbar = plt.colorbar(contour, label='Total FLOPS')
    cbar.formatter.set_powerlimits((0, 0))  # Use scientific notation

    # Plot the ideal ratio curve (raw ratios) - this traces the minimum through the landscape
    plt.plot(achieved_p_levels, ideal_ratios, 'b-', linewidth=3, label='Ideal Ratio (Min FLOPS)', marker='o', markersize=4, zorder=5)

    # Plot the maximum ratio explored curve (raw ratios)
    # plt.plot(achieved_p_levels, max_ratios, 'g--', linewidth=2, label='Max Ratio Explored', marker='s', markersize=3, zorder=5)

    # Plot the minimum ratio explored curve (raw ratios)
    # plt.plot(achieved_p_levels, min_ratios, 'r:', linewidth=2, label='Min Ratio Explored', marker='^', markersize=3, zorder=5)

    # Handle baseline if provided
    if baseline is not None:
        y_baseline, baseline_flops_optimization, baseline_flops_sampling_prefill_cache, baseline_flops_sampling_generation = preprocess_data(
            baseline, metric, threshold, flops_per_step
        )

        baseline_pts = get_points(y_baseline, baseline_flops_optimization, baseline_flops_sampling_prefill_cache,
                                baseline_flops_sampling_generation, return_ratio=True, n_smoothing=n_smoothing, cumulative=cumulative)
        baseline_ratio, _, baseline_n_samp, baseline_mean_p, baseline_total_flop = baseline_pts.T[:5]

        baseline_finite_mask = np.isfinite(baseline_ratio) & np.isfinite(baseline_mean_p)
        if np.any(baseline_finite_mask):
            # For baseline, just plot the raw ratios
            plt.scatter(baseline_mean_p[baseline_finite_mask], baseline_ratio[baseline_finite_mask],
                       color="red", s=80, alpha=0.9, marker="^",
                       edgecolors='black', linewidth=0.5, label="Baseline", zorder=6)

    plt.xlabel("p_harmful", fontsize=14)
    plt.ylabel("Sampling FLOPS / Total FLOPS", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.title(title, fontsize=16)
    plt.xlim(0, 1)
    plt.yscale('log')
    plt.ylim(1e-5, 1.0)
    plt.xlim(left=0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"evaluate/distributional_paper/ideal_ratio_plots/{title}.pdf", bbox_inches='tight')
    plt.close()

    if verbose:
        logging.info(f"p_harmful range: {p_harmful_min:.4f} to {p_harmful_max:.4f}")
        logging.info(f"Ideal ratio range: {ideal_ratios.min():.4f} to {ideal_ratios.max():.4f}")
        logging.info(f"Max ratio range: {max_ratios.min():.4f} to {max_ratios.max():.4f}")
        logging.info(f"Min ratio range: {min_ratios.min():.4f} to {min_ratios.max():.4f}")
        logging.info(f"Total FLOPS landscape range: {landscape_total_flops.min():.2e} to {landscape_total_flops.max():.2e}")
        logging.info(f"Number of points in landscape: {len(landscape_total_flops)}")
        logging.info(f"Number of p_harmful levels with solutions: {len(achieved_p_levels)}")


def flops_breakdown_plot(
    results: dict[str,np.ndarray],
    baseline: dict[str,np.ndarray] | None = None,
    title: str = "FLOPS Breakdown Analysis",
    sample_levels_to_plot: tuple[int, ...]|None = None,
    metric: tuple[str, ...] = ('scores', 'strong_reject', 'p_harmful'),
    cumulative: bool = False,
    flops_per_step: int | None = None,
    threshold: float|None = None,
    color_scale: str = "linear",
    verbose: bool = True,
):
    """
    Plot optimization FLOPS vs sampling FLOPS with p_harmful as a 2D surface.
    """
    y, flops_optimization, flops_sampling_prefill_cache, flops_sampling_generation = preprocess_data(
        results, metric, threshold, flops_per_step
    )
    n_runs, n_steps, n_total_samples = y.shape
    if sample_levels_to_plot is None:
        sample_levels_to_plot = generate_sample_sizes(n_total_samples)

    pts = get_points(y, flops_optimization, flops_sampling_prefill_cache, flops_sampling_generation,
                     return_ratio=False, cumulative=cumulative)
    cost, step_idx, n_samp, mean_p = pts.T

    # Calculate individual FLOP components
    opt_flops = []
    sampling_flops = []
    p_harmful_vals = []
    n_samples_vals = []

    rng = np.random.default_rng(42)  # Fixed seed for reproducibility

    for j in range(1, n_total_samples + 1, 1):
        for i in range(0, n_steps, 1):
            opt_flop = np.mean(flops_optimization[:, :i+1].sum(axis=1))
            sampling_flop = np.mean(flops_sampling_generation[:, i]) * j + np.mean(flops_sampling_prefill_cache[:, i])
            p_vals = []
            for n in range(10):
                # Calculate p_harmful value with same logic as other functions
                if cumulative and i > 0:
                    samples_up_to_now = y[:, :i, rng.choice(n_total_samples, size=1, replace=False)].max(axis=1)[:, 0]
                    samples_at_end = y[:, i, rng.choice(n_total_samples, size=j, replace=False)].max(axis=-1)
                    p_val = np.stack([samples_up_to_now, samples_at_end], axis=1).max(axis=1).mean(axis=0)
                else:
                    p_val = y[:, i, rng.choice(n_total_samples, size=j, replace=False)].max(axis=-1).mean(axis=0)
                p_vals.append(p_val)

            opt_flops.append(opt_flop+sampling_flop)
            sampling_flops.append(sampling_flop)
            p_harmful_vals.append(np.mean(p_vals))
            n_samples_vals.append(j)

    opt_flops = np.array(opt_flops)
    sampling_flops = np.array(sampling_flops)
    p_harmful_vals = np.array(p_harmful_vals)
    n_samples_vals = np.array(n_samples_vals)

    plt.figure(figsize=(12, 8))

    # Create 2D surface plot using griddata interpolation
    # Define grid for interpolation
    sampling_min, sampling_max = sampling_flops.min(), sampling_flops.max()
    opt_min, opt_max = opt_flops.min(), opt_flops.max()

    # Use log space for sampling FLOPS if range is large
    if sampling_max / sampling_min > 100:
        sampling_grid = np.logspace(np.log10(sampling_min), np.log10(sampling_max), 100)
    else:
        sampling_grid = np.linspace(sampling_min, sampling_max, 100)

    # Use log space for optimization FLOPS if range is large
    if opt_max / opt_min > 100:
        opt_grid = np.logspace(np.log10(opt_min), np.log10(opt_max), 100)
    else:
        opt_grid = np.linspace(opt_min, opt_max, 100)

    Sampling_grid, Opt_grid = np.meshgrid(sampling_grid, opt_grid)

    # Interpolate p_harmful values onto the grid
    try:
        p_harmful_grid = griddata(
            (sampling_flops, opt_flops),
            p_harmful_vals,
            (Sampling_grid, Opt_grid),
            method='linear',
            fill_value=np.nan
        )
    except Exception as e:
        if verbose:
            logging.info(f"Linear interpolation failed: {e}, trying nearest neighbor")
        p_harmful_grid = griddata(
            (sampling_flops, opt_flops),
            p_harmful_vals,
            (Sampling_grid, Opt_grid),
            method='nearest',
            fill_value=0
        )

    # Create contour plot
    levels = np.linspace(np.nanmin(p_harmful_vals), np.nanmax(p_harmful_vals), 50)
    contour = plt.contourf(Sampling_grid, Opt_grid, p_harmful_grid, levels=levels,
                          cmap='plasma', extend='both')


    # Add colorbar
    cbar = plt.colorbar(contour)
    if threshold is None:
        cbar.set_label(r"$p_{harmful}$", fontsize=14)
    else:
        cbar.set_label(f"ASR (threshold: {threshold})", fontsize=14)


    # Add baseline if provided
    if baseline is not None:
        y_baseline, baseline_flops_optimization, baseline_flops_sampling_prefill_cache, baseline_flops_sampling_generation = preprocess_data(
            baseline, metric, threshold, flops_per_step
        )

        baseline_pts = get_points(y_baseline, baseline_flops_optimization, baseline_flops_sampling_prefill_cache,
                                baseline_flops_sampling_generation, return_ratio=False, cumulative=cumulative)
        baseline_cost, baseline_step_idx, baseline_n_samp, baseline_mean_p = baseline_pts.T

        # Calculate baseline FLOP components
        baseline_opt_flops = []
        baseline_sampling_flops = []

        for i in range(0, y_baseline.shape[1], 1):
            opt_flop = np.mean(baseline_flops_optimization[:, :i+1].sum(axis=1))
            sampling_flop = np.mean(baseline_flops_sampling_generation[:, i]) * 1 + np.mean(baseline_flops_sampling_prefill_cache[:, i])

            baseline_opt_flops.append(opt_flop+sampling_flop)
            baseline_sampling_flops.append(sampling_flop)

        baseline_opt_flops = np.array(baseline_opt_flops)
        baseline_sampling_flops = np.array(baseline_sampling_flops)

        plt.scatter(baseline_sampling_flops, baseline_opt_flops,
                   s=60, alpha=0.9, marker="^",
                   edgecolors='red', linewidth=2,
                   color='white', label="Baseline")

    plt.xlabel("Sampling FLOPS", fontsize=14)
    plt.ylabel("Total FLOPS", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.title(title, fontsize=16)

    # Use log scale for both axes if the range is large
    if sampling_max / sampling_min > 100:
        plt.xscale('log')
    if opt_max / opt_min > 100:
        plt.yscale('log')

    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(f"evaluate/distributional_paper/flops_breakdown/{title}.pdf", bbox_inches='tight')
    plt.close()

    if verbose:
        logging.info(f"Sampling FLOPS range: {sampling_flops.min():.2e} to {sampling_flops.max():.2e}")
        logging.info(f"Optimization FLOPS range: {opt_flops.min():.2e} to {opt_flops.max():.2e}")
        logging.info(f"p_harmful range: {p_harmful_vals.min():.4f} to {p_harmful_vals.max():.4f}")
        logging.info(f"Surface grid shape: {p_harmful_grid.shape}")
        logging.info(f"Valid surface points: {np.sum(~np.isnan(p_harmful_grid))}/{p_harmful_grid.size}")

def ridge_plot(
    sampled_data: dict[str,np.ndarray],
    model_title: str,
    cfg: dict,
):
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

    # Create ridge plot for p_harmful distributions across steps
    data = np.array(sampled_data[("scores", "strong_reject", "p_harmful")])

    # Prepare data for ridge plot
    ridge_data = []
    step_idxs = [0] + list(generate_sample_sizes(data.shape[1]-1))
    if data.shape[1]-1 not in step_idxs:
        step_idxs.append(data.shape[1]-1)
    for step_idx in step_idxs:
        step_data = data[:, step_idx, :].flatten()  # Get p_harmful values for this step
        # Round/bucketize the data into five values: 0, 0.25, 0.5, 0.75, 1.0
        # step_data = np.round(step_data * 4) / 4
        for value in step_data:
            ridge_data.append({'step': f'Step {step_idx}', 'p_harmful': value})
    df = pd.DataFrame(ridge_data)

    # Create ridge plot for p_harmful distributions across steps
    unique_steps = sorted(df['step'].unique(), key=lambda x: int(x.split()[1]))
    n_steps = len(unique_steps)
    pal = sns.cubehelix_palette(n_steps, rot=-.25, light=.7)

    # Initialize the FacetGrid object
    g = sns.FacetGrid(df, row="step", hue="step", aspect=15, height=.5, palette=pal,
                        row_order=unique_steps)

    # Draw the densities
    g.map(sns.kdeplot, "p_harmful", bw_adjust=0.5, clip_on=True, fill=True, alpha=1, linewidth=1.5)
    g.map(sns.kdeplot, "p_harmful", bw_adjust=0.5, clip_on=True, color="w", lw=2)

    # Add vertical lines for mean and median
    def add_mean_lines(x, **kwargs):
        ax = plt.gca()
        mean_val = np.mean(x)
        ax.axvline(mean_val, color='red', linestyle='-', alpha=0.7, linewidth=1, ymax=0.8)

    g.map(add_mean_lines, "p_harmful")

    # Add reference line at y=0
    g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)

    # Set the subplots to overlap
    g.figure.subplots_adjust(hspace=-.4)

    # Remove axes details that don't play well with overlap
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.despine(bottom=True, left=True)
    g.set(xlim=(0, 1))

    # Add title to ridge plot
    g.figure.suptitle(f'{model_title} - {cfg["title_suffix"]} - p_harmful Ridge Plot',
                        fontsize=14, y=0.95)

    # Save the ridge plot
    filename = f"evaluate/distributional_paper/ridge_plots/{model_title}_{cfg['title_suffix']}.pdf"
    g.figure.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(g.figure)



# Helper ---------------------------------------------------------------------------
def run_analysis(
    model: str,
    model_title: str,
    atk_name: str,
    cfg: dict,
    analysis_type: str = "pareto",
):
    """
    Unified function to run different types of analysis.

    Parameters
    ----------
    analysis_type : str
        Type of analysis: "pareto", "flops_ratio", "ideal_ratio", "histogram", "histogram_2"
    """
    logging.info(f"{analysis_type.title()} Analysis: {atk_name} {cfg.get('title_suffix', '')}")

    # ---------- sampled run ----------
    sampled_data = fetch_data(model, cfg.get("attack_override", atk_name), cfg["sample_params"](),
                             DATASET_IDX, GROUP_BY)

    # Attack-specific post-processing
    if post := cfg.get("postprocess"):
        post(sampled_data, METRIC)

    # ---------- baseline run (not needed for histograms) ----------
    baseline_data = None
    if analysis_type not in ["histogram", "histogram_2"]:
        baseline_attack = cfg.get("baseline_attack", atk_name)
        baseline_data = fetch_data(model, baseline_attack, cfg["baseline_params"](),
                                  DATASET_IDX, GROUP_BY)

    # ---------- generate plot based on analysis type ----------
    flops_per_step_fn = lambda x: FLOPS_PER_STEP.get(atk_name, lambda x, c: 0)(x, num_model_params(model))

    if analysis_type == "pareto":
        pareto_plot(
            sampled_data,
            baseline_data,
            title=f"{model_title} {cfg['title_suffix']}",
            cumulative=cfg["cumulative"],
            metric=METRIC,
            flops_per_step=flops_per_step_fn,
            threshold=None,
            color_scale="sqrt",
        )
    elif analysis_type == "flops_ratio":
        flops_ratio_plot(
            sampled_data,
            baseline_data,
            title=f"{model_title} {cfg['title_suffix']} FLOPS Ratio",
            cumulative=cfg["cumulative"],
            metric=METRIC,
            flops_per_step=flops_per_step_fn,
            threshold=None,
            color_scale="sqrt",
        )
    elif analysis_type == "ideal_ratio":
        ideal_ratio_plot(
            sampled_data,
            baseline_data,
            title=f"{model_title} {cfg['title_suffix']} Ideal Ratio",
            cumulative=cfg["cumulative"],
            metric=METRIC,
            flops_per_step=flops_per_step_fn,
            threshold=None,
        )
    elif analysis_type == "histogram":

        plt.figure(figsize=(10, 6))
        data_list = []
        positions = []
        data = np.array(sampled_data[("scores", "strong_reject", "p_harmful")])
        for i in np.arange(data.shape[1]):
            data_list.append(data[:, i].flatten())
            positions.append(i)

        # Create 2D heatmap
        # Define bins for p_harmful values (y-axis)
        p_harmful_bins = np.linspace(0, 1, 101)  # 50 bins from 0 to 1

        # Create 2D histogram matrix
        heatmap_data = np.zeros((len(p_harmful_bins)-1, len(positions)))

        for i, (pos, d) in enumerate(zip(positions, data_list)):
            # Calculate histogram for this position
            counts, _ = np.histogram(d, bins=p_harmful_bins)
            heatmap_data[:, i] = counts / len(d)

        # Create the heatmap
        im = plt.imshow(heatmap_data,
                       aspect='auto',
                       origin='lower',
                       extent=[positions[0], positions[-1], 0, 1],
                       cmap='viridis',
                       norm=LogNorm(vmin=1/len(d), vmax=heatmap_data.max())
        )
        # Add colorbar
        cbar = plt.colorbar(im, label='Density')

        # Calculate and plot median and mean lines
        medians = []
        means = []
        for data_at_pos in data_list:
            medians.append(np.median(data_at_pos))
            means.append(np.mean(data_at_pos))

        # Plot mean line
        plt.plot(positions, means, color='orange', linewidth=2, label='Mean', alpha=0.8)
        plt.plot(positions, np.maximum.accumulate(means), color='red', linewidth=2, label='Max', alpha=0.8)

        # Add legend
        plt.legend(loc='upper right', framealpha=0.8)

        plt.xscale("log")
        plt.xlabel('Step Index', fontsize=14)
        plt.ylabel('p_harmful', fontsize=14)
        plt.title(f'{model_title} - {atk_name} - p_harmful Distribution', fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save the plot
        filename = f"evaluate/distributional_paper/histograms/{model_title}_{cfg['title_suffix']}.pdf"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
    elif analysis_type == "ridge":
        ridge_plot(
            sampled_data,
            model_title,
            cfg,
        )
    elif analysis_type == "histogram_2":
        # Create histogram plot
        plt.figure(figsize=(10, 6))
        threshold = 0.0
        show_top = False  # Set to False to hide the top subplot
        bins = np.linspace(0, 1, 21)


        data_low = np.array(sampled_data[("scores","strong_reject","p_harmful")])[:, 0].flatten()
        data_low = data_low[data_low > threshold]

        data_high = np.array(sampled_data[("scores","strong_reject","p_harmful")])[:, -1].flatten()
        data_high = data_high[data_high > threshold]

        # Create CDF plot
        fig, ax = plt.subplots(figsize=(10, 6))

        # Calculate survival functions (1 - CDF)
        data_low_sorted = np.sort(data_low)
        data_high_sorted = np.sort(data_high)

        # Calculate proportion of items with value <= x, then invert to get survival function
        cdf_low = np.arange(1, len(data_low_sorted) + 1) / len(data_low_sorted)
        cdf_high = np.arange(1, len(data_high_sorted) + 1) / len(data_high_sorted)

        survival_low = 1 - cdf_low
        survival_high = 1 - cdf_high

        # Plot survival functions
        ax.plot(data_low_sorted, survival_low, label="First Step", linewidth=2, alpha=0.8)
        ax.plot(data_high_sorted, survival_high, label="Last Step", linewidth=2, alpha=0.8)

        ax.set_xlabel("p_harmful", fontsize=14)
        ax.set_ylabel("Survival Probability (P(X $>$ x))", fontsize=14)
        ax.set_xlim(threshold, 1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)

        ax.set_title(f"{model_title} - {cfg['title_suffix']} - p_harmful Survival Function",
                     fontsize=16)

        # Save the plot
        filename = f"evaluate/distributional_paper/cdf_plots/{model_title}_{cfg['title_suffix']}.pdf"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

        if show_top:
            fig, (ax_top, ax_bottom) = plt.subplots(
                2, 1, sharex=True,
                figsize=(10, 6),
                gridspec_kw={"height_ratios": [1, 3]}   # top is shorter
            )
            axes = (ax_top, ax_bottom)
        else:
            fig, ax_bottom = plt.subplots(figsize=(10, 4))
            axes = (ax_bottom,)

        # draw the same two histograms on both axes
        for ax in axes:
            ax.hist(data_high, bins=bins, alpha=0.7, label="Last Step")
            ax.hist(data_low,  bins=bins, alpha=0.7, label="First Step")

        if show_top:
            # ------------- tune y limits so the top shows only the tall bar(s) -----
            counts_start = np.sort(np.histogram(data_low.flatten(), bins=bins)[0])
            counts_end = np.sort(np.histogram(data_high.flatten(), bins=bins)[0])

            ylims = (
                (0, max(counts_start[-2], counts_end[-2]) * 1.1),
                (min(counts_start[-1], counts_end[-1]) * 0.9, max(counts_start[-1], counts_end[-1]) * 1.1),
            )
            ax_top.set_ylim(*ylims[1])
            ax_bottom.set_ylim(*ylims[0])

            # ---------------- cosmetic fixes -----------------
            ax_top.spines.bottom.set_visible(False)
            ax_bottom.spines.top.set_visible(False)
            ax_top.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

            # little diagonal "break" marks
            d = .5  # size of diagonal lines
            kwargs = dict(marker=[(-1, -1), (1, 1)], markersize=12,
                        linestyle="none", color="k", clip_on=False)
            ax_top.plot([0, 1], [0, 0], transform=ax_top.transAxes, **kwargs)
            ax_bottom.plot([0, 1], [1, 1], transform=ax_bottom.transAxes, **kwargs)

        # --------- the usual labels ------------
        ax_bottom.set_xlabel("p_harmful", fontsize=14)
        ax_bottom.set_ylabel("Frequency (\\%)", fontsize=14)
        ax_bottom.set_xlim(threshold, 1)

        if show_top:
            ax_top.set_title(f"{model_title} - {cfg['title_suffix']} - p_harmful distribution",
                            fontsize=16)
        else:
            ax_bottom.set_title(f"{model_title} - {cfg['title_suffix']} - p_harmful distribution",
                               fontsize=16)

        # Convert y-axis tick labels to percentages
        if not show_top:
            # Get current y-tick locations
            yticks = ax_bottom.get_yticks()
            # Convert to percentages by dividing by total count and multiplying by 100
            ytick_labels = [f"{100*tick / len(data_low.flatten()):.0f}%" for tick in yticks]
            ax_bottom.set_yticklabels(ytick_labels)
        else:
            for ax in (ax_bottom,):
                # Get current y-tick locations
                yticks = ax.get_yticks()
                # Convert to percentages by dividing by total count and multiplying by 100
                ytick_labels = [f"{100*tick / len(data_low.flatten()):.0f}%" for tick in yticks]
                ax.set_yticklabels(ytick_labels)
            # Convert y-axis tick labels to percentages
            for ax in (ax_top,):
                # Get current y-tick locations
                yticks = ax.get_yticks()
                # Convert to percentages by dividing by total count and multiplying by 100
                ytick_labels = [f"{100*tick / len(data_low.flatten()):.1f}%" for tick in yticks]
                ax.set_yticklabels(ytick_labels)
            ax_top.legend()

        if not show_top:
            ax_bottom.legend()

        for ax in axes:
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save the plot
        filename = f"evaluate/distributional_paper/histograms_2/{model_title}_{cfg['title_suffix']}.pdf"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
    elif analysis_type == "flops_breakdown":
        flops_breakdown_plot(
            sampled_data,
            baseline_data,
            title=f"{model_title} {cfg['title_suffix']} FLOPS Breakdown",
            cumulative=cfg["cumulative"],
            metric=METRIC,
            flops_per_step=flops_per_step_fn,
            threshold=None,
            color_scale="sqrt",
        )
    else:
        raise ValueError(f"Unknown analysis type: {analysis_type}")



# ----------------------------------------------------------------------------------
# Configuration and Constants
# ----------------------------------------------------------------------------------

MODELS = {
    "meta-llama/Meta-Llama-3.1-8B-Instruct": "Meta Llama 3.1 8B",
    "google/gemma-3-1b-it": "Gemma 3 1B",
    "GraySwanAI/Llama-3-8B-Instruct-RR": "Llama 3 CB",
    "Unispac/Llama2-7B-Chat-Augmented": "Llama 2 DeepAlign",
}

FLOPS_PER_STEP = {
    "autodan": lambda s, c: 69845248149248 // num_model_params("Qwen/Qwen2.5-0.5B-Instruct") * c,
    "gcg":     lambda s, c: int(1e14) + 14958709489152 // num_model_params("Qwen/Qwen2.5-0.5B-Instruct") * c,
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
    ("beast", dict(
        title_suffix="BEAST",
        cumulative=False,
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
DATASET_IDX = list(range(75))

def main(fail: bool = False):
    for analysis_type in ["pareto", "flops_ratio", "ideal_ratio", "histogram", "histogram_2", "ridge", "flops_breakdown"]:
    # for analysis_type in [ "ridge"]:
        logging.info("\n" + "="*80)
        logging.info(f"GENERATING {analysis_type.upper().replace('_', ' ')} PLOTS")
        logging.info("="*80)

        for model_key, model_title in MODELS.items():
            logging.info(f"Model: {model_key}")
            for atk_name, atk_cfg in ATTACKS:
                try:
                    run_analysis(model_key, model_title, atk_name, atk_cfg, analysis_type)
                except Exception as e:
                    if fail:
                        raise e
                    logging.info(f"Error running {analysis_type} analysis for {atk_name}, "
                        f"cfg: {atk_cfg.get('title_suffix', 'unknown')}: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Generate plots for distributional paper')
    parser.add_argument('--fail', action='store_true', help='Override flag to fail')
    args = parser.parse_args()

    main(args.fail)
