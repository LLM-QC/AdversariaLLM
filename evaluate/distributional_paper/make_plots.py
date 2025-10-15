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
from matplotlib.ticker import MaxNLocator
logging.basicConfig(level=logging.INFO)

pd.set_option("display.max_colwidth", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.expand_frame_repr", False)

from src.io_utils import get_filtered_and_grouped_paths, collect_results, num_model_params

s_harm_tex = r"$s_{harm}$"
def generate_sample_sizes(total_samples: int) -> tuple[int, ...]:
    if total_samples < 1:
        return tuple()
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


def _non_cumulative_dominance_frontier(xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Return all points ordered by cost, without dominance filtering.
    This creates a non-cumulative frontier that includes all points.

    Parameters
    ----------
    xs, ys : 1-D arrays of equal length
        Coordinates of the candidate points.

    Returns
    -------
    frontier_xs, frontier_ys : 1-D arrays
        All points, sorted by xs ascending.
    """
    order = np.argsort(xs)              # sort by cost
    xs_sorted, ys_sorted = xs[order], ys[order]

    frontier_x, frontier_y = [0, *xs_sorted], [0, *ys_sorted]

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
    elif method == "non_cumulative":
        # non-cumulative frontier that includes all points
        return _non_cumulative_dominance_frontier(xs, ys)
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

    # Create figure with subplots: legend + main plot + 2x2 grid on the right
    # fig = plt.figure(figsize=(5.4, 2.4))  # hero when slicing at bottom=0.1
    fig = plt.figure(figsize=(5.4, 2.8))  # hero when slicing at bottom=0.1

    # Main Pareto plot (left half, spanning both rows)
    ax1 = plt.subplot2grid((2, 3), (0, 1), colspan=2, rowspan=2)

    # ---------- scatter all points ----------
    color_norm = setup_color_normalization(color_scale, n_samp)
    if plot_points:
        # Subsample points for plotting, considering logarithmic cost spacing
        if len(cost) > 1000:
            # Sample uniformly in log space
            log_cost = np.log10(cost + 1e-10)
            log_indices = np.argsort(log_cost)
            step = len(log_indices) // 1000
            subsample_indices = log_indices[::step][:1000]

            cost_sub = cost[subsample_indices]
            mean_p_sub = mean_p[subsample_indices]
            n_samp_sub = n_samp[subsample_indices]
        else:
            cost_sub = cost
            mean_p_sub = mean_p
            n_samp_sub = n_samp

        sc = plt.scatter(cost_sub, mean_p_sub, c=n_samp_sub, cmap="viridis", alpha=0.15, s=3, norm=color_norm)


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
            y_interp = [interp1d(x_, y_, kind="previous", bounds_error=False, fill_value=np.nan)(x_interp) for x_, y_ in zip(xs, ys)]

            color = cmap(color_norm(j))
            y_mean = np.nanmean(y_interp, axis=0)
            # Filter out NaN values and zeros
            valid_mask = ~np.isnan(y_mean) & (y_mean > 0)

            x_pts = x_interp[valid_mask]
            y_pts = y_mean[valid_mask]
            # Store data for bar charts
            frontier_data[j] = {
                'x': x_pts,
                'y': y_pts,
                'color': color,
                'max_asr': np.max(y_pts) if np.any(valid_mask) else 0
            }

            plt.plot(
                x_pts,
                y_pts,
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

            y_interp = [interp1d(x_, y_, kind="previous", bounds_error=False, fill_value=np.nan)(x_interp) for x_, y_ in zip(xs, ys)]
            y_interps.append(np.nanmean(y_interp, axis=0))
        y_interps = np.array(y_interps)
        argmax = np.nanargmax(y_interps, axis=0)
        argmax = np.maximum.accumulate(argmax)
        y_envelope = np.nanmax(y_interps, axis=0)

        # Filter out NaN values and zeros
        valid_mask = ~np.isnan(y_envelope) & (y_envelope > 0)
        color = [cmap(color_norm(argmax[i])) for i in range(len(argmax)) if valid_mask[i]]
        plt.scatter(x_interp[valid_mask], y_envelope[valid_mask], c=color, s=2)

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
            max_cost_baseline = max(cost_baseline)

            # ---------- overlay Pareto frontiers ----------
            if plot_frontiers or plot_envelope:
                mask = n_samp_baseline == 1
                fx, fy = _pareto_frontier(cost_baseline[mask], mean_p_baseline[mask], method=frontier_method)
                # if len(fx) > 1:
                #     fx[0] = min(f["x"][0] for f in frontier_data.values())
                #     print(fx[0], fx[1])
                y_interp_baseline = interp1d(fx, fy, kind="previous", bounds_error=False, fill_value=np.nan)(x_interp)
                if max_cost_baseline / max_cost < 0.7:
                    max_cost_baseline = max_cost
                valid_mask_baseline = ~np.isnan(y_interp_baseline) & (y_interp_baseline > 0) & (x_interp < max_cost_baseline)
                # Store baseline data for bar charts
                baseline_max_asr = np.max(y_interp_baseline[valid_mask_baseline]) if np.any(valid_mask_baseline) else 0
                baseline_frontier_data = {
                    'x': x_interp[valid_mask_baseline],
                    'y': y_interp_baseline[valid_mask_baseline],
                    'max_asr': baseline_max_asr
                }
                plt.plot(
                    x_interp[valid_mask_baseline],
                    y_interp_baseline[valid_mask_baseline],
                    marker="o",
                    linewidth=1.8,
                    markersize=2,
                    label=f"Baseline (greedy)",
                    color="r",
                )

    plt.xlabel("Total FLOPs", fontsize=13)
    if threshold is None:
        plt.ylabel(r"$s_{harm@n}$", fontsize=18)
    else:
        plt.ylabel(r"${ASR}@n$", fontsize=18)
    plt.grid(True, alpha=0.3)
    # plt.ylim(bottom=0.1)
    plt.xscale(x_scale)
    if "autodan" in title.lower():
        loc = "lower right"
    elif x_scale == "log":
        loc = "upper left"
    else:
        loc = "lower right"

    handles, labels = plt.gca().get_legend_handles_labels()
    # Create legend subplot and move legend there
    ax0 = plt.subplot2grid((2, 3), (0, 0), colspan=1, rowspan=2)
    ax0.axis('off')  # Remove all axes
    # Get legend from current plot and move to ax0
    handles = [*handles[:-1][::-1], handles[-1]]
    labels = [*labels[:-1][::-1], labels[-1]]
    ax0.legend(handles, labels, loc='center', fontsize=12)
    plt.tight_layout()
    if threshold is None:
        plt.savefig(f"evaluate/distributional_paper/pareto_plots/{title.replace(' ', '_')}.pdf")
    else:
        plt.savefig(f"evaluate/distributional_paper/pareto_plots/{title.replace(' ', '_')}_t={threshold}.pdf")
    plt.close()

    fig = plt.figure(figsize=(6, 5))


    bar_chart_margin_multiplier = 5
    if baseline_frontier_data is not None and baseline_max_asr > 0:
        methods_flops = []
        flops_required = []
        colors_flops = []

        # Find FLOPs required to reach baseline ASR for each sampling method
        target_asr = baseline_max_asr

        for j in sample_levels_to_plot:
            if j in frontier_data:
                # Find the minimum FLOPs where ASR >= target_asr
                y_vals = frontier_data[j]['y']
                x_vals = frontier_data[j]['x']

                # Find points where ASR >= target_asr
                valid_indices = y_vals >= target_asr
                if np.any(valid_indices):
                    min_flops = np.min(x_vals[valid_indices])
                    methods_flops.append(f"{j} samples")
                    flops_required.append(min_flops)
                    colors_flops.append(frontier_data[j]['color'])

        # Add baseline (find minimum FLOPs where it reaches target ASR)
        if baseline_frontier_data['x'].size > 0:
            # Find the minimum FLOPs where baseline ASR >= target_asr
            baseline_y_vals = baseline_frontier_data['y']
            baseline_x_vals = baseline_frontier_data['x']
            baseline_valid_indices = baseline_y_vals >= target_asr
            if np.any(baseline_valid_indices):
                baseline_flops = np.min(baseline_x_vals[baseline_valid_indices])
            else:
                # Fallback to minimum FLOPs if no point reaches target ASR
                baseline_flops = np.min(baseline_x_vals)
            methods_flops.insert(0, "Baseline")
            flops_required.insert(0, baseline_flops)
            colors_flops.insert(0, "red")
    else:
        methods_flops = []
        flops_required = []
        colors_flops = []

    # ---------- Bar Chart 1: Max ASR Comparison (Vertical Slice) ----------
    def add_asr_bar_chart():
        ax2 = plt.subplot2grid((2, 2), (0, 0))

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
                plt.ylabel(r"$\Delta$ $s_{harm@n}$" , fontsize=17)
            else:
                plt.ylabel(r"$\Delta$ ${ASR}@n$", fontsize=17)
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, alpha=0.3, axis='y')
            # Increase ylim by 2% on top and bottom
            ymin, ymax = plt.ylim()
            margin = (ymax - ymin) * 0.03 * bar_chart_margin_multiplier
            plt.ylim(ymin - margin, ymax + margin)

            # ----- add labels with a 4-point gap -----
            for bar, value in zip(bars, max_asrs):
                # choose label position: above for positive, below for negative
                offset_pt = 4      # visual gap in points
                va = 'bottom' if value >= 0 else 'top'
                offset = (0, offset_pt if value >= 0 else -offset_pt)

                ax2.annotate(f'{value:.2f}',
                            xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                            xytext=offset,
                            textcoords='offset points',
                            ha='center', va=va, fontsize=10)
    add_asr_bar_chart()


    # ---------- Bar Chart 2: FLOPs Efficiency to Reach Baseline ASR (Horizontal Slice) ----------
    def add_flops_bar_chart():
        ax3 = plt.subplot2grid((2, 2), (0, 1))

        if methods_flops:
            bars = plt.bar(methods_flops, flops_required, color=colors_flops, alpha=0.7, edgecolor='black')
            plt.ylabel("FLOPs to match baseline", fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.yscale('log')
            plt.grid(True, alpha=0.3, axis='y')
            # Increase ylim by
            ymin, ymax = plt.ylim()
            import math
            margin = ((math.log10(ymax) - math.log10(ymin)) * 0.2)
            plt.ylim(ymin, ymax * (1+margin))

            # --- constant 5-point vertical gap ---
            # for bar, value in zip(bars, flops_required):
            #     ax3.annotate(f'{value:.2e}',
            #                 xy=(bar.get_x() + bar.get_width()/2, value),   # anchor at top of bar
            #                 xytext=(0, 5),                                  # 5 points straight up
            #                 textcoords='offset points',
            #                 ha='center', va='bottom', rotation=45, fontsize=9)

    add_flops_bar_chart()

    # ---------- Bar Chart 3: Speedup vs Baseline (Bottom Left) ----------
    def add_speedup_bar_chart():
        ax4 = plt.subplot2grid((2, 2), (1, 0))

        # Create speedup plot
        speedup_methods = []
        speedups = []
        speedup_colors = []

        # Calculate speedup for each method (baseline_flops / method_flops)
        baseline_flops = flops_required[0] if methods_flops and methods_flops[0] == "Baseline" else None

        if baseline_flops is not None:
            for i, (method, flops, color) in enumerate(zip(methods_flops, flops_required, colors_flops)):
                if method != "Baseline":  # Skip baseline itself
                    speedup = baseline_flops / flops if flops > 0 else 0
                    speedup_methods.append(method)
                    speedups.append(speedup)
                    speedup_colors.append(color)

            if speedup_methods:
                bars = plt.bar(speedup_methods, speedups, color=speedup_colors, alpha=0.7, edgecolor='black')
                plt.ylabel("Speedup (FLOPs)", fontsize=12)
                plt.xticks(rotation=45, ha='right')
                plt.grid(True, alpha=0.3, axis='y')

                # Add horizontal line at y=1 for reference
                plt.axhline(y=1, color='red', linestyle='--', alpha=0.7, linewidth=1)

                # Increase ylim by small margin
                ymin, ymax = plt.ylim()
                margin = (ymax - ymin) * 0.05 * bar_chart_margin_multiplier
                plt.ylim(max(0, ymin - margin), ymax + margin)

                # Add value labels on bars
                for bar, value in zip(bars, speedups):
                    ax4.annotate(f'{value:.1f}x',
                                xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                                xytext=(0, 5),
                                textcoords='offset points',
                                ha='center', va='bottom', fontsize=10)

    add_speedup_bar_chart()

    # --------- Base Plot 4: ASR @ max greedy FLOPs (vertical slice), bottom right ----------
    def add_asr_at_max_greedy_flops_bar_chart():
        ax5 = plt.subplot2grid((2, 2), (1, 1))

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
                if baseline_frontier_data["x"].size == 0:
                    continue
                baseline_max_flops = baseline_frontier_data['x'][-1]
                x_idx_of_same_flops_as_baseline = np.argmax(frontier_data[j]['x'] > baseline_max_flops) - 1
                delta_asr = frontier_data[j]['y'][x_idx_of_same_flops_as_baseline] - baseline_max_asr if baseline_frontier_data is not None else 0
                max_asrs.append(delta_asr)
                colors.append(frontier_data[j]['color'])

        if methods:
            bars = plt.bar(methods, max_asrs, color=colors, alpha=0.7, edgecolor='black')
            if threshold is None:
                plt.ylabel(r"$\Delta$ $s_{harm@n}$" , fontsize=14)
            else:
                plt.ylabel(r"$\Delta$ ${ASR}@n$", fontsize=14)
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, alpha=0.3, axis='y')
            # Increase ylim by 2% on top and bottom
            ymin, ymax = plt.ylim()
            margin = (ymax - ymin) * 0.03 * bar_chart_margin_multiplier
            plt.ylim(ymin - margin, ymax + margin)

            # ----- add labels with a 4-point gap -----
            for bar, value in zip(bars, max_asrs):
                # choose label position: above for positive, below for negative
                offset_pt = 4      # visual gap in points
                va = 'bottom' if value >= 0 else 'top'
                offset = (0, offset_pt if value >= 0 else -offset_pt)

                ax5.annotate(f'{value:.2f}',
                            xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                            xytext=offset,
                            textcoords='offset points',
                            ha='center', va=va, fontsize=10)
    add_asr_at_max_greedy_flops_bar_chart()
    # # ---------- Line Plot 4: Continuous FLOPs to Reach Baseline ASR (Bottom Left) ----------
    # ax5 = plt.subplot2grid((2, 5), (1, 3))

    # if baseline_frontier_data is not None and baseline_max_asr > 0:
    #     target_asr = baseline_max_asr

    #     # Generate continuous range of sample counts
    #     sample_range = range(1, n_total_samples + 1)
    #     continuous_flops = []
    #     continuous_samples = []

    #     # Calculate frontier data for all sample counts (not just sample_levels_to_plot)
    #     rng_continuous = np.random.default_rng()
    #     n_smoothing_continuous = 10  # Reduced for performance

    #     for j in sample_range:
    #         xs = []
    #         ys = []
    #         for _ in range(n_smoothing_continuous):
    #             pts = []
    #             for i in range(0, n_steps, 1):
    #                 pts.append(subsample_and_aggregate(i, j, cumulative, y, flops_optimization,
    #                                                  flops_sampling_prefill_cache, flops_sampling_generation, rng_continuous))

    #             pts = np.asarray(pts)
    #             cost, _, _, mean_p = pts.T

    #             fx, fy = _pareto_frontier(cost, mean_p, method=frontier_method)
    #             xs.append(fx)
    #             ys.append(fy)

    #         # Interpolate and average
    #         y_interp = [interp1d(x_, y_, kind="previous", bounds_error=False, fill_value=(0, max(y_)))(x_interp)
    #                    for x_, y_ in zip(xs, ys)]
    #         y_mean = np.mean(y_interp, axis=0)

    #         # Find minimum FLOPs where ASR >= target_asr
    #         nonzero_mask = y_mean > 0
    #         if np.any(nonzero_mask):
    #             y_vals = y_mean[nonzero_mask]
    #             x_vals = x_interp[nonzero_mask]

    #             valid_indices = y_vals >= target_asr
    #             if np.any(valid_indices):
    #                 min_flops = np.min(x_vals[valid_indices])
    #                 continuous_flops.append(min_flops)
    #                 continuous_samples.append(j)

    #     if continuous_flops:
    #         # Plot the continuous line
    #         plt.plot(continuous_samples, continuous_flops, 'b-', linewidth=2, alpha=0.8, label='All Samples')

    #         # Highlight the baseline point
    #         if baseline_frontier_data['x'].size > 0:
    #             baseline_y_vals = baseline_frontier_data['y']
    #             baseline_x_vals = baseline_frontier_data['x']
    #             baseline_valid_indices = baseline_y_vals >= target_asr
    #             if np.any(baseline_valid_indices):
    #                 baseline_flops = np.min(baseline_x_vals[baseline_valid_indices])
    #                 plt.axhline(y=baseline_flops, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Baseline')

    #         # Highlight the discrete sample levels from the bar chart
    #         for j in sample_levels_to_plot:
    #             if j in [s for s in continuous_samples]:
    #                 idx = continuous_samples.index(j)
    #                 color = cmap(color_norm(j))
    #                 plt.scatter(j, continuous_flops[idx], color=color, s=60, alpha=0.9,
    #                           edgecolors='black', linewidth=0.5, zorder=5)

    #         plt.xlabel("Number of Samples", fontsize=12)
    #         plt.ylabel("FLOPs to Reach Baseline ASR", fontsize=12)
    #         plt.xscale('log')
    #         plt.yscale('log')
    #         plt.grid(True, alpha=0.3)
    #         plt.legend(fontsize=10)

    #         # Set reasonable x-axis limits
    #         plt.xlim(1, n_total_samples)

    #         # Increase ylim by small margin
    #         ymin, ymax = plt.ylim()
    #         import math
    #         margin = ((math.log10(ymax) - math.log10(ymin)) * 0.1)
    #         plt.ylim(ymin / (1+margin), ymax * (1+margin))

    plt.tight_layout()
    if threshold is None:
        plt.savefig(f"evaluate/distributional_paper/bar_charts/{title.replace(' ', '_')}.pdf")
    else:
        plt.savefig(f"evaluate/distributional_paper/bar_charts/{title.replace(' ', '_')}_t={threshold}.pdf")
    plt.close()
    # Create a separate figure for just the legend
    fig_legend = plt.figure(figsize=(4, 1))
    ax_legend = fig_legend.add_subplot(111)
    ax_legend.axis('off')

    # Create legend elements for sample levels
    legend_elements = []
    # Add baseline if it exists
    if baseline is not None:
        legend_elements.append(plt.Line2D([0], [0], color="red", linewidth=2,
                                        label="Baseline (Greedy)"))
    cmap = plt.get_cmap("viridis")
    color_norm = setup_color_normalization("linear", np.array(sample_levels_to_plot))

    for j in sample_levels_to_plot:
        if j in frontier_data:
            color = cmap(color_norm(j))
            legend_elements.append(plt.Line2D([0], [0], color=color, linewidth=2,
                                            label=f"{j} samples"))


    # Create horizontal legend
    ax_legend.legend(handles=legend_elements, loc='center', ncol=len(legend_elements),
        fontsize=10, frameon=False, columnspacing=1.0, handletextpad=0.5)


    plt.tight_layout()
    plt.savefig(f"evaluate/distributional_paper/pareto_plots/legend_{n_total_samples}.pdf", bbox_inches='tight')
    plt.close()


def non_cumulative_pareto_plot(
    results: dict[str,np.ndarray],
    baseline: dict[str,np.ndarray] | None = None,
    title: str = "Non-Cumulative Pareto Frontier",
    sample_levels_to_plot: tuple[int, ...]|None = None,
    metric: tuple[str, ...] = ('scores', 'strong_reject', 'p_harmful'),
    plot_points: bool = False,
    plot_frontiers: bool = True,
    plot_envelope: bool = False,
    plot_baseline: bool = False,
    verbose: bool = True,
    flops_per_step: int | None = None,
    n_x_points: int = 10000,
    x_scale="linear",
    threshold: float|None = None,
    color_scale: str = "linear",
):
    """
    Scatter the full design-space AND overlay non-cumulative Pareto frontiers
    for selected sampling counts. Uses the non_cumulative frontier method
    which includes all points without dominance filtering.
    """
    y, flops_optimization, flops_sampling_prefill_cache, flops_sampling_generation = preprocess_data(
        results, metric, threshold, flops_per_step
    )
    n_runs, n_steps, n_total_samples = y.shape
    if sample_levels_to_plot is None:
        sample_levels_to_plot = generate_sample_sizes(n_total_samples)

    pts = get_points(y, flops_optimization, flops_sampling_prefill_cache, flops_sampling_generation,
                     return_ratio=False, cumulative=False)
    cost, step_idx, n_samp, mean_p = pts.T
    max_step = max(step_idx)
    if x_scale == "log":
        x_interp = np.logspace(0, np.log10(max_step+1), n_x_points)
    else:
        x_interp = np.linspace(0, max_step+1, n_x_points)

    # Create figure with subplots: legend + main plot + 2x2 grid on the right
    # fig = plt.figure(figsize=(5.4, 2.4))  # hero when slicing at bottom=0.1
    fig = plt.figure(figsize=(5.4, 2.8))  # hero when slicing at bottom=0.1

    # Main Pareto plot (left half, spanning both rows)
    ax1 = plt.subplot2grid((2, 3), (0, 1), colspan=2, rowspan=2)

    # ---------- scatter all points ----------
    color_norm = setup_color_normalization(color_scale, n_samp)
    if plot_points:
        # Subsample points for plotting, considering step spacing
        if len(step_idx) > 1000:
            # Sample uniformly in step space
            step_indices = np.argsort(step_idx)
            step = len(step_indices) // 1000
            subsample_indices = step_indices[::step][:1000]

            step_idx_sub = step_idx[subsample_indices]
            mean_p_sub = mean_p[subsample_indices]
            n_samp_sub = n_samp[subsample_indices]
        else:
            step_idx_sub = step_idx
            mean_p_sub = mean_p
            n_samp_sub = n_samp

        sc = plt.scatter(step_idx_sub, mean_p_sub, c=n_samp_sub, cmap="viridis", alpha=0.15, s=3, norm=color_norm)


    # ---------- overlay non-cumulative Pareto frontiers ----------
    cmap = plt.get_cmap("viridis")
    rng = np.random.default_rng()

    n_smoothing = 50
    frontier_data = {}  # Store frontier data for bar charts

    if plot_frontiers:
        # Only plot the maximum number of samples frontier
        j = n_total_samples
        xs = []
        ys = []
        n_smoothing = 1  # Use single smoothing for max samples
        for _ in range(n_smoothing):
            pts = []
            for i in range(0, n_steps, 1):
                pts.append(subsample_and_aggregate(i, j, False, y, flops_optimization, flops_sampling_prefill_cache, flops_sampling_generation, rng))

            pts = np.asarray(pts)
            cost, step_idx_pts, _, mean_p = pts.T

            fx, fy = _pareto_frontier(step_idx_pts, mean_p, method="non_cumulative")
            xs.append(fx)
            ys.append(fy)
        y_interp = [interp1d(x_, y_, kind="previous", bounds_error=False, fill_value=np.nan)(x_interp) for x_, y_ in zip(xs, ys)]

        color = cmap(color_norm(j))
        y_mean = np.nanmean(y_interp, axis=0)
        # Filter out NaN values and zeros
        valid_mask = ~np.isnan(y_mean) & (y_mean > 0)

        x_pts = x_interp[valid_mask]
        y_pts = y_mean[valid_mask]
        # Store data for bar charts
        frontier_data[j] = {
            'x': x_pts,
            'y': y_pts,
            'color': color,
            'max_asr': np.max(y_pts) if np.any(valid_mask) else 0
        }

        plt.plot(
            x_pts,
            y_pts,
            linewidth=1.2,
            label="Steps",
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
                    pts.append(subsample_and_aggregate(i, j, False, y, flops_optimization, flops_sampling_prefill_cache, flops_sampling_generation, rng))

                pts = np.asarray(pts)
                cost, step_idx_pts, n_samp, mean_p = pts.T

                fx, fy = _pareto_frontier(step_idx_pts, mean_p, method="non_cumulative")
                xs.append(fx)
                ys.append(fy)

            y_interp = [interp1d(x_, y_, kind="previous", bounds_error=False, fill_value=np.nan)(x_interp) for x_, y_ in zip(xs, ys)]
            y_interps.append(np.nanmean(y_interp, axis=0))
        y_interps = np.array(y_interps)
        argmax = np.nanargmax(y_interps, axis=0)
        argmax = np.maximum.accumulate(argmax)
        y_envelope = np.nanmax(y_interps, axis=0)

        # Filter out NaN values and zeros
        valid_mask = ~np.isnan(y_envelope) & (y_envelope > 0)
        color = [cmap(color_norm(argmax[i])) for i in range(len(argmax)) if valid_mask[i]]
        plt.scatter(x_interp[valid_mask], y_envelope[valid_mask], c=color, s=2)

    title_suffix = ""

    # Handle baseline data
    baseline_max_asr = 0
    baseline_frontier_data = None

    if baseline is not None and plot_baseline:
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
                           baseline_flops_sampling_generation, return_ratio=False, cumulative=False)
            cost_baseline, step_idx_baseline, n_samp_baseline, mean_p_baseline = pts.T
            max_step_baseline = max(step_idx_baseline)

            # ---------- overlay Pareto frontiers ----------
            if plot_frontiers or plot_envelope:
                mask = n_samp_baseline == 1
                fx, fy = _pareto_frontier(step_idx_baseline[mask], mean_p_baseline[mask], method="non_cumulative")
                y_interp_baseline = interp1d(fx, fy, kind="previous", bounds_error=False, fill_value=np.nan)(x_interp)
                if max_step_baseline / max_step < 0.7:
                    max_step_baseline = max_step
                valid_mask_baseline = ~np.isnan(y_interp_baseline) & (y_interp_baseline > 0) & (x_interp < max_step_baseline)
                # Store baseline data for bar charts
                baseline_max_asr = np.max(y_interp_baseline[valid_mask_baseline]) if np.any(valid_mask_baseline) else 0
                baseline_frontier_data = {
                    'x': x_interp[valid_mask_baseline],
                    'y': y_interp_baseline[valid_mask_baseline],
                    'max_asr': baseline_max_asr
                }
                plt.plot(
                    x_interp[valid_mask_baseline],
                    y_interp_baseline[valid_mask_baseline],
                    linewidth=1.2,
                    label=f"Baseline",
                    color="r",
                )

    plt.xlabel("Optimization Steps", fontsize=13)
    if threshold is None:
        plt.ylabel(r"$s_{harm@n}$", fontsize=18)
    else:
        plt.ylabel(r"${ASR}@n$", fontsize=18)
    plt.grid(True, alpha=0.3)
    # plt.ylim(bottom=0.1)
    plt.xscale(x_scale)
    if "autodan" in title.lower():
        loc = "lower right"
    elif x_scale == "log":
        loc = "upper left"
    else:
        loc = "lower right"

    handles, labels = plt.gca().get_legend_handles_labels()
    # Create legend subplot and move legend there
    ax0 = plt.subplot2grid((2, 3), (0, 0), colspan=1, rowspan=2)
    ax0.axis('off')  # Remove all axes
    # Get legend from current plot and move to ax0
    if len(handles) > 1:
        # If we have baseline, put it last
        if plot_baseline and len(handles) > 1:
            handles = [*handles[:-1][::-1], handles[-1]]
            labels = [*labels[:-1][::-1], labels[-1]]
        else:
            handles = handles[::-1]
            labels = labels[::-1]
    ax0.legend(handles, labels, loc='center', fontsize=12)
    plt.tight_layout()
    if threshold is None:
        plt.savefig(f"evaluate/distributional_paper/non_cumulative_pareto_plots/{title.replace(' ', '_')}.pdf")
    else:
        plt.savefig(f"evaluate/distributional_paper/non_cumulative_pareto_plots/{title.replace(' ', '_')}_t={threshold}.pdf")
    plt.close()

    # Note: Skipping the bar charts for non-cumulative version to keep it simple
    # The non-cumulative version is primarily for visualization of all points

    if verbose:
        logging.info(f"Non-cumulative Pareto plot saved for {title}")


def multi_attack_non_cumulative_pareto_plot(
    attacks_data: dict,  # {attack_name: (results_dict, config)}
    model_title: str,
    title: str = "Multi-Attack Non-Cumulative Pareto",
    metric: tuple[str, ...] = ('scores', 'strong_reject', 'p_harmful'),
    threshold: float|None = None,
    n_x_points: int = 10000,
    x_scale: str = "linear",
    verbose: bool = True,
):
    """
    Create a non-cumulative Pareto plot showing multiple attacks on the same axes.
    Each attack shows its frontier with 50 samples.
    """

    # Color scheme for attacks
    attack_colors = {
        "gcg": "#1f77b4",      # blue
        "autodan": "#ff7f0e",  # orange
        "beast": "#2ca02c",    # green
        "pair": "#d62728",     # red
        "bon": "#9467bd",      # purple
        "direct": "#8c564b",   # brown
    }

    plt.figure(figsize=(4, 3))

    # Filter to only show specific attacks
    desired_attacks = {"PAIR", "BEAST", "AutoDAN", "GCG"}
    filtered_attacks_data = {}

    for config_key, (results, config) in attacks_data.items():
        if config_key in desired_attacks:
            filtered_attacks_data[config_key] = (results, config)

    attacks_data = filtered_attacks_data

    if not attacks_data:
        logging.warning("No desired attacks found in data")
        return

    # Use percentage of optimization steps (0-100%) as x-axis
    x_interp = np.linspace(0, 100, n_x_points)

    # Process each attack
    rng = np.random.default_rng(42)

    for config_key, (results, config) in attacks_data.items():
        y, flops_optimization, flops_sampling_prefill_cache, flops_sampling_generation = preprocess_data(
            results, metric, threshold, None
        )
        n_runs, n_steps, n_total_samples = y.shape

        # Extract original attack name from config for color mapping
        original_attack_name = None
        for atk_name, atk_cfg in ATTACKS:
            if atk_cfg.get('title_suffix') == config_key:
                original_attack_name = atk_name
                break

        color = attack_colors.get(original_attack_name, "black")

        # Use 50 samples for each attack
        target_samples = 1#min(50, n_total_samples)
        n_smoothing = 1  # Single smoothing for cleaner lines
        xs = []
        ys = []

        for _ in range(n_smoothing):
            pts = []
            for i in range(0, n_steps, 1):
                pts.append(subsample_and_aggregate(i, target_samples, False, y,
                                                 flops_optimization, flops_sampling_prefill_cache,
                                                 flops_sampling_generation, rng))

            pts = np.asarray(pts)
            cost, step_idx_pts, _, mean_p = pts.T

            # Convert step indices to percentages (0-100%)
            step_percentages = (step_idx_pts / (n_steps - 1)) * 100

            fx, fy = _pareto_frontier(step_percentages, mean_p, method="non_cumulative")
            xs.append(fx)
            ys.append(fy)

        y_interp = [interp1d(x_, y_, kind="previous", bounds_error=False,
                           fill_value=np.nan)(x_interp) for x_, y_ in zip(xs, ys)]
        y_mean = np.nanmean(y_interp, axis=0)

        # Calculate delta from first step's value
        # Find the first valid (non-NaN) value as baseline
        valid_indices = ~np.isnan(y_mean)
        if np.any(valid_indices):
            first_valid_value = y_mean[valid_indices][0]
            y_delta = y_mean - first_valid_value

            # Filter out NaN values
            valid_mask = ~np.isnan(y_delta)
            if np.any(valid_mask):
                label = f"{config_key}"

                plt.plot(x_interp[valid_mask], y_delta[valid_mask],
                        linewidth=1.2,
                        label=label, color=color)

    plt.xlabel(r"Optimization Progress (\%)", fontsize=15)
    if threshold is None:
        plt.ylabel(r"$\Delta$ $s_{harm@1}$", fontsize=16)
    else:
        plt.ylabel(r"$\Delta$ ${ASR}@1$", fontsize=16)

    plt.grid(True, alpha=0.3)
    plt.xlim(0, 100)  # Set x-axis limits to 0-100%
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)  # Reference line at delta=0
    plt.title(f"{model_title}", fontsize=15)

    # Get legend handles and labels before saving main plot
    handles, labels = plt.gca().get_legend_handles_labels()
    # Sort handles and labels alphabetically by labels
    sorted_pairs = sorted(zip(handles, labels), key=lambda x: x[1])
    handles, labels = zip(*sorted_pairs) if sorted_pairs else ([], [])

    plt.tight_layout()
    if threshold is None:
        plt.savefig(f"evaluate/distributional_paper/multi_attack_non_cumulative_pareto_plots/{title.replace(' ', '_')}.pdf")
    else:
        plt.savefig(f"evaluate/distributional_paper/multi_attack_non_cumulative_pareto_plots/{title.replace(' ', '_')}_t={threshold}.pdf")
    plt.close()

    # Create a separate figure for just the legend
    fig_legend = plt.figure(figsize=(4, 1))
    ax_legend = fig_legend.add_subplot(111)
    ax_legend.axis('off')

    # Create horizontal legend
    ax_legend.legend(handles=handles, loc='center', ncol=2,
        fontsize=12, frameon=False, columnspacing=1.0, handletextpad=0.5)

    plt.tight_layout()
    if threshold is None:
        plt.savefig(f"evaluate/distributional_paper/multi_attack_non_cumulative_pareto_plots/legend_{title.replace(' ', '_')}.pdf", bbox_inches='tight')
    else:
        plt.savefig(f"evaluate/distributional_paper/multi_attack_non_cumulative_pareto_plots/legend_{title.replace(' ', '_')}_t={threshold}.pdf", bbox_inches='tight')
    plt.close()

    if verbose:
        logging.info(f"Multi-attack non-cumulative Pareto plot saved for {model_title}")
        logging.info(f"Legend saved separately")


def comparative_pareto_plot(
    model: str,
    model_title: str,
    attacks_data: dict,  # {attack_name: (results_dict, config)}
    title: str = "Comparative Pareto Analysis",
    metric: tuple[str, ...] = ('scores', 'strong_reject', 'p_harmful'),
    threshold: float|None = None,
    n_x_points: int = 10000,
    x_scale: str = "log",
    flops_per_step_fns: dict = None,  # {attack_name: flops_per_step_fn}
    frontier_method: str = "basic",
    verbose: bool = True,
    baseline_attacks: set = {"gcg", "beast", "pair", "autodan"},  # Attacks to show baseline points for
):
    """
    Create a comparative Pareto plot showing multiple attacks on the same axes.
    - For gcg, autodan, beast, pair: use 50-sample frontier
    - For bon, direct: use envelope curve
    """

    # Define which attacks use which approach
    envelope_attacks = {"bon", "direct"}

    # Color scheme for attacks
    attack_colors = {
        "gcg": "#1f77b4",      # blue
        "autodan": "#ff7f0e",  # orange
        "beast": "#2ca02c",    # green
        "pair": "#d62728",     # red
        "bon": "#9467bd",      # purple
        "direct": "#8c564b",   # brown
    }

    plt.figure(figsize=(8, 5))

    # Calculate overall x-axis range from all attacks
    all_costs = []
    for config_key, (results, config) in attacks_data.items():
        flops_per_step_fn = flops_per_step_fns.get(config_key) if flops_per_step_fns else None
        y, flops_optimization, flops_sampling_prefill_cache, flops_sampling_generation = preprocess_data(
            results, metric, threshold, flops_per_step_fn
        )
        pts = get_points(y, flops_optimization, flops_sampling_prefill_cache, flops_sampling_generation,
                        return_ratio=False, cumulative=config.get("cumulative", False))
        cost, _, _, _ = pts.T
        all_costs.extend(cost)

    max_cost = max(all_costs)
    if x_scale == "log":
        x_interp = np.logspace(11, np.log10(max_cost)+0.001, n_x_points)
    else:
        x_interp = np.linspace(0, max_cost+1, n_x_points)

    # Process each attack
    rng = np.random.default_rng(42)

    for config_key, (results, config) in attacks_data.items():
        flops_per_step_fn = flops_per_step_fns.get(config_key) if flops_per_step_fns else None
        y, flops_optimization, flops_sampling_prefill_cache, flops_sampling_generation = preprocess_data(
            results, metric, threshold, flops_per_step_fn
        )
        n_runs, n_steps, n_total_samples = y.shape

        # Extract original attack name from config for color mapping
        # We need to find the original attack name by looking at the ATTACKS list
        original_attack_name = None
        for atk_name, atk_cfg in ATTACKS:
            if atk_cfg.get('title_suffix') == config_key:
                original_attack_name = atk_name
                break

        color = attack_colors.get(original_attack_name, "black")

        if original_attack_name in envelope_attacks:
            # Use envelope approach with more permissive extrapolation
            n_smoothing = 10  # Limit smoothing for performance
            y_interps = []
            all_xs = []  # Collect all x values to determine meaningful range

            for j in range(1, n_total_samples+1):
                xs = []
                ys = []
                for n in range(n_smoothing):
                    pts = []
                    for i in range(0, n_steps, 1):
                        pts.append(subsample_and_aggregate(i, j, config.get("cumulative", False), y,
                                                         flops_optimization, flops_sampling_prefill_cache,
                                                         flops_sampling_generation, rng))

                    pts = np.asarray(pts)
                    cost, _, _, mean_p = pts.T

                    fx, fy = _pareto_frontier(cost, mean_p, method=frontier_method)
                    xs.append(fx)
                    ys.append(fy)

                # Collect all x values for range determination
                for x_ in xs:
                    if len(x_) > 0:
                        all_xs.extend(x_)

                # For envelope attacks, use 0 fill for left side and last value for right side to avoid gaps
                y_interp = []
                for x_, y_ in zip(xs, ys):
                    if len(x_) > 0 and len(y_) > 0:
                        interp_func = interp1d(x_, y_, kind="previous", bounds_error=False,
                                             fill_value=(0, y_[-1]))
                        y_interp.append(interp_func(x_interp))
                    else:
                        y_interp.append(np.zeros_like(x_interp))

                y_interps.append(np.mean(y_interp, axis=0) if y_interp else np.zeros_like(x_interp))

            y_interps = np.array(y_interps)
            y_envelope = np.max(y_interps, axis=0)

            # For envelope, only filter out leading zeros, but cap at reasonable x-range
            max_meaningful_x = np.max(all_xs) if all_xs else x_interp[-1]
            valid_mask = (y_envelope > 0) & (x_interp <= max_meaningful_x * 1.1)  # Allow 10% extension

            if np.any(valid_mask):
                plt.plot(x_interp[valid_mask], y_envelope[valid_mask],
                        marker="o", linewidth=2.5, markersize=3,
                        label=config_key, color=color)

        else:
            # Use 50-sample frontier approach
            target_samples = n_total_samples
            n_smoothing = 50
            xs = []
            ys = []

            for _ in range(n_smoothing):
                pts = []
                for i in range(0, n_steps, 1):
                    pts.append(subsample_and_aggregate(i, target_samples, config.get("cumulative", False), y,
                                                     flops_optimization, flops_sampling_prefill_cache,
                                                     flops_sampling_generation, rng))

                pts = np.asarray(pts)
                cost, _, _, mean_p = pts.T

                fx, fy = _pareto_frontier(cost, mean_p, method=frontier_method)
                xs.append(fx)
                ys.append(fy)

            y_interp = [interp1d(x_, y_, kind="previous", bounds_error=False,
                               fill_value=np.nan)(x_interp) for x_, y_ in zip(xs, ys)]
            y_mean = np.nanmean(y_interp, axis=0)

            # Filter out NaN values and zeros
            valid_mask = ~np.isnan(y_mean) & (y_mean > 0)
            if np.any(valid_mask):
                label = f"{config_key}"
                if target_samples < n_total_samples:
                    label += f" ({target_samples} samples)"

                plt.plot(x_interp[valid_mask], y_mean[valid_mask],
                        marker="o", linewidth=2.5, markersize=3,
                        label=label, color=color)

    # Add baseline points for specified attacks
    for baseline_attack_name in baseline_attacks:
        # Find the config key for this attack (could be multiple configs for same attack)
        matching_configs = []
        for config_key, (results, config) in attacks_data.items():
            # Find the original attack name for this config
            for atk_name, atk_cfg in ATTACKS:
                if atk_cfg.get('title_suffix') == config_key and atk_name == baseline_attack_name:
                    matching_configs.append((config_key, results, config))
                    break

        # Use the first matching config (or could choose a specific one)
        if matching_configs:
            config_key, results, config = matching_configs[0]
            try:
                # Fetch baseline data for this attack
                baseline_params = config.get("baseline_params", lambda: {
                    "generation_config": {"num_return_sequences": 1, "temperature": 0.0}
                })()
                baseline_attack = config.get("baseline_attack", baseline_attack_name)

                baseline_data = fetch_data(model, baseline_attack, baseline_params,
                                         list(range(100)), {"model", "attack_params"})

                # Process baseline data
                flops_per_step_fn = flops_per_step_fns.get(config_key) if flops_per_step_fns else None
                y_baseline, baseline_flops_optimization, baseline_flops_sampling_prefill_cache, baseline_flops_sampling_generation = preprocess_data(
                    baseline_data, metric, threshold, flops_per_step_fn
                )

                if y_baseline is not None:
                    n_runs_baseline, n_steps_baseline, n_total_samples_baseline = y_baseline.shape

                    # Get the point at max step count (last step)
                    pts = get_points(y_baseline, baseline_flops_optimization, baseline_flops_sampling_prefill_cache,
                                   baseline_flops_sampling_generation, return_ratio=False, cumulative=config.get("cumulative", False))
                    cost_baseline, step_idx_baseline, n_samp_baseline, mean_p_baseline = pts.T

                    # Find the point at max step count
                    max_step_mask = step_idx_baseline == (n_steps_baseline - 1)
                    if np.any(max_step_mask):
                        baseline_cost = cost_baseline[max_step_mask][0]
                        baseline_mean_p = mean_p_baseline[max_step_mask][0]

                        # Plot baseline point
                        color = attack_colors.get(baseline_attack_name, "black")
                        plt.scatter(baseline_cost, baseline_mean_p,
                                  s=100, marker="^", color=color,
                                  edgecolors='black', linewidth=1.5, alpha=0.9,
                                  label=f"{config_key} Baseline", zorder=10)

                        if verbose:
                            logging.info(f"Added baseline point for {baseline_attack_name}: cost={baseline_cost:.2e}, p_harmful={baseline_mean_p:.3f}")

            except Exception as e:
                if verbose:
                    logging.warning(f"Could not add baseline point for {baseline_attack_name}: {e}")

    plt.xlabel("Total FLOPs", fontsize=14)
    if threshold is None:
        plt.ylabel(r"$s_{harm@n}$", fontsize=18)
    else:
        plt.ylabel(r"${ASR}@n$", fontsize=18)

    plt.grid(True, alpha=0.3)
    plt.xscale(x_scale)
    plt.legend(loc='upper left', fontsize=12)
    plt.title(f"{model_title}", fontsize=16)

    plt.tight_layout()
    if threshold is None:
        plt.savefig(f"evaluate/distributional_paper/comparative_pareto_plots/{title.replace(' ', '_')}.pdf")
    else:
        plt.savefig(f"evaluate/distributional_paper/comparative_pareto_plots/{title.replace(' ', '_')}_t={threshold}.pdf")
    plt.close()

    if verbose:
        logging.info(f"Comparative Pareto plot saved for {model_title}")


def flops_ratio_plot(
    results: dict[str,np.ndarray],
    baseline: dict[str,np.ndarray] | None = None,
    title: str = "FLOPs Ratio Analysis",
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
    Plot p_harmful vs the ratio of optimization FLOPs to sampling FLOPs.
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

    # Calculate total FLOPs for coloring option
    total_flop = opt_flop + sampling_flop

    # Filter out infinite ratios for plotting
    finite_mask = np.isfinite(ratio)
    ratio_finite = ratio[finite_mask]
    mean_p_finite = mean_p[finite_mask]
    n_samp_finite = n_samp[finite_mask]
    total_flop_finite = total_flop[finite_mask]

    plt.figure(figsize=(10, 6))

    # Create dual color encoding: hue based on samples, strength based on total FLOPs
    # Normalize sample counts for hue
    sample_norm = setup_color_normalization("linear", n_samp_finite)
    # Normalize total FLOPs for alpha/strength
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

    plt.xlabel("Sampling FLOPs / Total FLOPs", fontsize=14)
    if threshold is None:
        plt.ylabel(r"$s_{harm@n}$", fontsize=14)
    else:
        plt.ylabel(r"${ASR}@n$".format(threshold=threshold), fontsize=14)

    plt.grid(True, alpha=0.3)
    # plt.title(title, fontsize=16)

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
    # plt.legend(loc='upper left')
    plt.tight_layout()
    if threshold is None:
        plt.savefig(f"evaluate/distributional_paper/flops_ratio_plots/{title.replace(' ', '_')}.pdf", bbox_inches='tight')
    else:
        plt.savefig(f"evaluate/distributional_paper/flops_ratio_plots/{title.replace(' ', '_')}_t={threshold}.pdf", bbox_inches='tight')
    plt.close()

    if verbose:
        logging.info(f"FLOPs ratio range: {ratio_finite.min():.2e} to {ratio_finite.max():.2e}")
        logging.info(f"Mean s_harm range: {mean_p_finite.min():.4f} to {mean_p_finite.max():.4f}")
        logging.info(f"Total FLOPs range: {total_flop_finite.min():.2e} to {total_flop_finite.max():.2e}")


def ideal_ratio_plot(
    results: dict[str,np.ndarray],
    baseline: dict[str,np.ndarray] | None = None,
    title: str = "Ideal Sampling FLOPs Ratio",
    sample_levels_to_plot: tuple[int, ...]|None = None,
    metric: tuple[str, ...] = ('scores', 'strong_reject', 'p_harmful'),
    cumulative: bool = False,
    flops_per_step: int | None = None,
    threshold: float|None = None,
    n_p_harmful_points: int = 100,
    verbose: bool = True,
):
    """
    Plot the ideal sampling FLOPs ratio for achieving different levels of harmfulness.
    For each p_harmful level, finds the point that achieves that level with minimum total FLOPs
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
            # Among achieving points, find the one with minimum total FLOPs
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

    plt.figure(figsize=(6, 4))

    # Create the FLOP landscape: interpolated surface with color indicating total FLOPs
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

    # Interpolate FLOPs values onto the grid
    try:
        flops_grid = griddata(
            (landscape_p_harmful, landscape_ratios),
            landscape_total_flops,
            (P_grid, Ratio_grid),
            method='linear',
            fill_value=np.nan
        )
    except Exception as e:
        raise ValueError(f"Error interpolating FLOPs values.")

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
                          cmap='viridis', alpha=0.8, extend='both')

    # Add colorbar for total FLOPs
    cbar = plt.colorbar(contour, label='Total FLOPs')
    cbar.formatter.set_powerlimits((0, 0))  # Use scientific notation

    # Plot the ideal ratio curve (raw ratios) - this traces the minimum through the landscape
    plt.plot(achieved_p_levels, ideal_ratios, 'k', linewidth=3, label='Ideal Ratio (Min FLOPs)', zorder=5)

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
                       color="red", s=60, alpha=0.9, marker="^",
                       edgecolors='black', linewidth=0.5, label="Baseline", zorder=6)

    plt.xlabel(r"$s_{harm}$", fontsize=16)
    plt.ylabel("Sampling FLOPs / Total FLOPs", fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.xlim(0, 1)
    plt.yscale('log')
    plt.ylim(1e-5, 1.0)
    plt.legend(loc='lower right')
    plt.tight_layout()
    if threshold is None:
        plt.savefig(f"evaluate/distributional_paper/ideal_ratio_plots/{title.replace(' ', '_')}.pdf", bbox_inches='tight')
    else:
        plt.savefig(f"evaluate/distributional_paper/ideal_ratio_plots/{title.replace(' ', '_')}_t={threshold}.pdf", bbox_inches='tight')
    plt.close()

    if verbose:
        logging.info(f"p_harmful range: {p_harmful_min:.4f} to {p_harmful_max:.4f}")
        logging.info(f"Ideal ratio range: {ideal_ratios.min():.4f} to {ideal_ratios.max():.4f}")
        logging.info(f"Max ratio range: {max_ratios.min():.4f} to {max_ratios.max():.4f}")
        logging.info(f"Min ratio range: {min_ratios.min():.4f} to {min_ratios.max():.4f}")
        logging.info(f"Total FLOPs landscape range: {landscape_total_flops.min():.2e} to {landscape_total_flops.max():.2e}")
        logging.info(f"Number of points in landscape: {len(landscape_total_flops)}")
        logging.info(f"Number of p_harmful levels with solutions: {len(achieved_p_levels)}")


def flops_breakdown_plot(
    results: dict[str,np.ndarray],
    baseline: dict[str,np.ndarray] | None = None,
    title: str = "FLOPs Breakdown Analysis",
    sample_levels_to_plot: tuple[int, ...]|None = None,
    metric: tuple[str, ...] = ('scores', 'strong_reject', 'p_harmful'),
    cumulative: bool = False,
    flops_per_step: int | None = None,
    threshold: float|None = None,
    color_scale: str = "linear",
    verbose: bool = True,
):
    """
    Plot optimization FLOPs vs sampling FLOPs with p_harmful as a 2D surface.
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

            opt_flops.append(opt_flop + sampling_flop)
            sampling_flops.append(sampling_flop)
            p_harmful_vals.append(np.mean(p_vals))
            n_samples_vals.append(j)

    opt_flops = np.array(opt_flops)
    sampling_flops = np.array(sampling_flops)
    p_harmful_vals = np.array(p_harmful_vals)
    n_samples_vals = np.array(n_samples_vals)

    plt.figure(figsize=(4, 2.8))

    # Create 2D surface plot using griddata interpolation
    # Define grid for interpolation
    sampling_min, sampling_max = sampling_flops.min(), sampling_flops.max()
    opt_min, opt_max = opt_flops.min(), opt_flops.max()

    # Use log space for sampling FLOPs if range is large
    if sampling_max / sampling_min > 100:
        sampling_grid = np.logspace(np.log10(sampling_min), np.log10(sampling_max), 100)
    else:
        sampling_grid = np.linspace(sampling_min, sampling_max, 100)

    # Use log space for optimization FLOPs if range is large
    if opt_max / opt_min > 100:
        opt_grid = np.logspace(np.log10(opt_min), np.log10(opt_max), 100)
    else:
        opt_grid = np.linspace(opt_min, opt_max, 100)

    Sampling_grid, Opt_grid = np.meshgrid(sampling_grid, opt_grid)

    # Interpolate p_harmful values onto the grid
    # use anisotropic interpolation
    try:
        p_harmful_grid = griddata(
            (sampling_flops, opt_flops),
            p_harmful_vals,
            (Sampling_grid, Opt_grid),
            method='linear',
            rescale=True
        )
        if np.isnan(p_harmful_grid).sum() > 0:
            p_harmful_grid_nearest = griddata(
                (sampling_flops, opt_flops),
                p_harmful_vals,
                (Sampling_grid, Opt_grid),
                method='nearest',
                fill_value=0,
                rescale=True
            )
            fill_mask = np.isnan(p_harmful_grid)
            impossible_mask = ((Sampling_grid + opt_min) > Opt_grid) | ((Opt_grid-Sampling_grid) > opt_max-Sampling_grid)
            p_harmful_grid[fill_mask] = p_harmful_grid_nearest[fill_mask]
            p_harmful_grid[impossible_mask] = np.nan
    except Exception as e:
        if verbose:
            logging.info(f"Linear interpolation failed: {e}, trying nearest neighbor")
        p_harmful_grid = griddata(
            (sampling_flops, opt_flops),
            p_harmful_vals,
            (Sampling_grid, Opt_grid),
            method='nearest',
            fill_value=0,
            rescale=True
        )

    # Create contour plot (transpose the grids)
    levels = np.linspace(np.nanmin(p_harmful_vals), np.nanmax(p_harmful_vals), 50)

    contour = plt.contourf(Opt_grid, Sampling_grid, p_harmful_grid, levels=levels,
                          cmap='viridis', extend='both')

    # Add colorbar
    cbar = plt.colorbar(contour, ticks=np.linspace(np.nanmin(p_harmful_vals), np.nanmax(p_harmful_vals), 5))
    cbar.ax.set_yticklabels([f'{tick:.2f}' for tick in cbar.get_ticks()])
    if threshold is None:
        cbar.set_label(r"$s_{harm@n}$", fontsize=17)
    else:
        cbar.set_label(r"ASR@$n$", fontsize=17)

    # Find maximum ASR at each total FLOP level, ignoring higher FLOP levels with lower ASR
    total_flops = sampling_flops + opt_flops

    # Sort by total FLOPs to process in order
    sort_idx = np.argsort(total_flops)
    sorted_total_flops = total_flops[sort_idx]
    sorted_sampling_flops = sampling_flops[sort_idx]
    sorted_opt_flops = opt_flops[sort_idx]
    sorted_p_harmful = p_harmful_vals[sort_idx]

    max_asr_points = []
    max_asr_seen = -np.inf

    for i in range(len(sorted_total_flops)):
        current_asr = sorted_p_harmful[i]

        # Only add this point if it achieves a higher ASR than we've seen before
        if current_asr > max_asr_seen:
            max_asr_seen = current_asr
            max_asr_points.append((sorted_opt_flops[i], sorted_sampling_flops[i]))

    if max_asr_points:
        max_asr_points = np.array(max_asr_points)

        plt.plot(max_asr_points[:, 0], max_asr_points[:, 1],
                color='black', linewidth=2, linestyle="--",alpha=0.8, label="Compute Optimal Frontier")

    plt.xlabel("Total FLOPs", fontsize=14)
    plt.ylabel("Sampling FLOPs", fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.grid(True, alpha=0.3)
    # plt.title(title, fontsize=16)

    # Use log scale for both axes if the range is large
    if sampling_max / sampling_min > 100:
        plt.xscale('log')
    if opt_max / opt_min > 100:
        plt.yscale('log')

    plt.legend(loc='lower left', fontsize=13, bbox_to_anchor=(-0.1, 0.95))
    plt.tight_layout()
    if threshold is None:
        plt.savefig(f"evaluate/distributional_paper/flops_breakdown/{title.replace(' ', '_')}.pdf", bbox_inches='tight')
    else:
        plt.savefig(f"evaluate/distributional_paper/flops_breakdown/{title.replace(' ', '_')}_t={threshold}.pdf", bbox_inches='tight')
    plt.close()

    if verbose:
        logging.info(f"Sampling FLOPs range: {sampling_flops.min():.2e} to {sampling_flops.max():.2e}")
        logging.info(f"Optimization FLOPs range: {opt_flops.min():.2e} to {opt_flops.max():.2e}")
        logging.info(f"p_harmful range: {p_harmful_vals.min():.4f} to {p_harmful_vals.max():.4f}")
        logging.info(f"Surface grid shape: {p_harmful_grid.shape}")
        logging.info(f"Valid surface points: {np.sum(~np.isnan(p_harmful_grid))}/{p_harmful_grid.size}")

def histogram_plot(
    sampled_data: dict[str,np.ndarray],
    model_title: str,
    atk_name: str,
    cfg: dict,
    threshold: float|None = None,
):
    plt.figure(figsize=(5, 6))
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
    plt.xlabel('Step Index', fontsize=18)
    plt.ylabel('p_harmful', fontsize=18)
    # plt.title(f'{model_title} - {atk_name}', fontsize=18)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save the plot
    if threshold is None:
        filename = f"evaluate/distributional_paper/histograms/{model_title}_{cfg['title_suffix']}.pdf"
    else:
        filename = f"evaluate/distributional_paper/histograms/{model_title}_{cfg['title_suffix']}_t={threshold}.pdf"
    plt.savefig(filename.replace(' ', '_'), dpi=300, bbox_inches='tight')
    plt.close()

def histogram_2_plot(
    sampled_data: dict[str,np.ndarray],
    model_title: str,
    cfg: dict,
    threshold: float|None = None,
):
    # Create histogram plot
    plt.figure(figsize=(10, 6))
    threshold = 0.0
    show_top = False  # Set to False to hide the top subplot
    bins = np.linspace(0, 1, 21)


    data_low = np.array(sampled_data[("scores","strong_reject","p_harmful")])[:, 0].flatten()
    data_low = data_low[data_low > threshold]

    data_high = np.array(sampled_data[("scores","strong_reject","p_harmful")])[:, -1].flatten()
    data_high = data_high[data_high > threshold]

    # Calculate Fisher-Pearson skewness coefficient for both datasets
    def calculate_skewness(data):
        n = len(data)
        mean = np.mean(data)
        std = np.std(data, ddof=0)  # Population standard deviation
        m3 = np.sum((data - mean)**3) / n
        return m3 / (std**3) if std > 0 else 0

    skew_low = calculate_skewness(data_low)
    skew_high = calculate_skewness(data_high)

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
    ax.plot(data_low_sorted, survival_low, label=r"First Step ($\gamma_1$" + f"={skew_low:.2f})", linewidth=2, alpha=0.8)
    ax.plot(data_high_sorted, survival_high, label=r"Last Step ($\gamma_1$" + f"={skew_high:.2f})", linewidth=2, alpha=0.8)

    ax.set_xlabel(r"$s_{harm}$", fontsize=14)
    ax.set_ylabel("Survival Probability (P(X $>$ x))", fontsize=14)
    ax.set_xlim(threshold, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    ax.set_title(f"{model_title} - {cfg['title_suffix']} - p_harmful Survival Function",
                    fontsize=16)

    # Save the plot
    if threshold is None:
        filename = f"evaluate/distributional_paper/cdf_plots/{model_title}_{cfg['title_suffix']}.pdf"
    else:
        filename = f"evaluate/distributional_paper/cdf_plots/{model_title}_{cfg['title_suffix']}_t={threshold}.pdf"
    plt.savefig(filename.replace(' ', '_'), dpi=300, bbox_inches='tight')
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
        ax.hist(data_high, bins=bins, alpha=0.7, label=r"Last Step ($\gamma_1$" + f"={skew_high:.2f})")
        ax.hist(data_low,  bins=bins, alpha=0.7, label=r"First Step ($\gamma_1$" + f"={skew_low:.2f})")

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
    ax_bottom.set_xlabel(r"$s_{harm}$", fontsize=14)
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
    if threshold is None:
        filename = f"evaluate/distributional_paper/histograms_2/{model_title}_{cfg['title_suffix']}.pdf"
    else:
        filename = f"evaluate/distributional_paper/histograms_2/{model_title}_{cfg['title_suffix']}_t={threshold}.pdf"
    plt.savefig(filename.replace(' ', '_'), dpi=300, bbox_inches='tight')
    plt.close()

def ridge_plot(
    sampled_data: dict[str,np.ndarray],
    model_title: str,
    cfg: dict,
    threshold: float|None = None,
):
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0), 'figure.figsize': (3, 3)})

    # Create ridge plot for p_harmful distributions across steps
    data = np.array(sampled_data[("scores", "strong_reject", "p_harmful")])

    # Prepare data for ridge plot
    ridge_data = []
    def log_spaced_indices(n_cols: int, k: int = 4) -> list[int]:
        """
        Return k log-spaced column indices in [0, n_cols-1] inclusive.
        Guarantees 0 and n_cols-1 are present; deduplicates if n_cols is small.
        """
        # corner cases: 0 or 1 column → just [0]; 2–3 cols → all of them
        if n_cols <= k:
            return list(range(n_cols))

        max_idx = n_cols - 1
        # make (k) points geometrically spaced in (1 … max_idx)
        inner = np.geomspace(1, max_idx, num=k, dtype=int)

        # build the final list and drop duplicates, then sort
        idx = np.unique(np.concatenate(([0], inner, [max_idx])))
        # if de-duplication left us with fewer than k values, pad with lin-spaced ones
        if idx.size < k:
            extra = np.linspace(0, max_idx, num=k, dtype=int)
            idx = np.unique(np.concatenate((idx, extra)))[:k]

        return idx.tolist()
    step_idxs = log_spaced_indices(data.shape[1], 4)

    for step_idx in step_idxs:
        step_data = data[:, step_idx, :].flatten()  # Get p_harmful values for this step
        # Round/bucketize the data into five values: 0, 0.25, 0.5, 0.75, 1.0
        # step_data = np.round(step_data * 4) / 4
        for value in step_data:
            ridge_data.append({'step': f'Step {step_idx}', r"$h(Y)$": value})
    df = pd.DataFrame(ridge_data)

    # Create ridge plot for p_harmful distributions across steps
    unique_steps = sorted(df['step'].unique(), key=lambda x: int(x.split()[1]))
    n_steps = len(unique_steps)
    pal = sns.cubehelix_palette(n_steps, rot=-.25, light=.7)

    # Initialize the FacetGrid object
    g = sns.FacetGrid(df, row="step", hue="step", aspect=5, height=.4, palette=pal,
                        row_order=unique_steps)

    # Draw the densities
    g.map(sns.kdeplot, r"$h(Y)$", bw_adjust=0.5, clip=(0, 1), fill=True, alpha=1, linewidth=0, zorder=1)
    g.map(sns.kdeplot, r"$h(Y)$", bw_adjust=0.5, clip=(0, 1), color="w", lw=3, zorder=0)

    # Add vertical lines for mean and median
    def add_mean_lines(x, **kwargs):
        ax = plt.gca()
        mean_val = np.mean(x)
        median_val = np.median(x)
        percentile_95 = np.percentile(x, 95)
        # ax.axhline(0, color='black', linestyle='-', alpha=0.7, linewidth=0.5, ymax=0.5)
        ax.axvline(median_val, color='black', linestyle='--', alpha=0.7, linewidth=1, ymax=0.5)
        ax.axvline(percentile_95, color='blue', linestyle='--', alpha=0.7, linewidth=1, ymax=0.5)
        ax.axvline(mean_val, color='red', linestyle='-', alpha=0.7, linewidth=1, ymax=0.5)

    g.map(add_mean_lines, r"$h(Y)$")

    # Add reference line at y=0
    g.refline(y=0, linewidth=1, linestyle="-", color=None, clip_on=False)

    # Set the subplots to overlap
    g.figure.subplots_adjust(hspace=-.4)
    # g.figure.subplots_adjust(top=)

    # Remove axes details that don't play well with overlap
    g.set_titles(f"")
    g.set(yticks=[], ylabel="")
    g.despine(bottom=True, left=True)
    g.set_xlabels(r"$h(Y)$", fontsize=14)
    g.set(xlim=(0, 1))
    plt.style.use("science")
    # Add legend for the mean line
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color='red', lw=1, alpha=0.7, label=r'$\text{Mean}$'),
                       Line2D([0], [0], color='black', linestyle='--', lw=1, alpha=0.7, label=r'$\text{Median}$'),
                       Line2D([0], [0], color='blue', linestyle='--', lw=1, alpha=0.7, label=r'$\text{95th Percentile}$')]
    def put_legend_on_top(fig, handles, **legend_kw):
        """
        Add a single figure-level legend centred above *fig* and
        tighten the subplot area so only the legend's real height is reserved.
        """
        # 1 — draw the legend (temporarily anywhere)
        lg = fig.legend(handles=handles,
                        loc="upper center",
                        bbox_to_anchor=(0.5, 1),      # top centre of the figure
                        frameon=False, **legend_kw)

        # 2 — draw the canvas *once* so we get the correct bbox
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        legend_bbox = lg.get_window_extent(renderer=renderer)

        # convert pixel height to figure fraction
        legend_h_px = legend_bbox.height
        fig_h_px     = fig.get_size_inches()[1] * fig.dpi
        frac = legend_h_px / fig_h_px

        # 3 — shrink the subplot area so it sits just below the legend
        pad = 0.01           # a tiny bit of breathing room
        fig.subplots_adjust(top=1-frac-pad)

        return lg

    # put_legend_on_top(g.figure, legend_elements, ncol=1)   # <─ that's it
    # g.figure.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1), ncol=1,
    #                         frameon=False, columnspacing=1.0, handletextpad=0.5)
    g.figure.suptitle(f"{model_title}", fontsize=14, y=0.9, va="top")

    # Save the ridge plot
    if threshold is None:
        filename = f"evaluate/distributional_paper/ridge_plots/{model_title}_{cfg['title_suffix']}.pdf"
    else:
        filename = f"evaluate/distributional_paper/ridge_plots/{model_title}_{cfg['title_suffix']}_t={threshold}.pdf"
    g.figure.savefig(filename.replace(' ', '_'), bbox_inches='tight')
    plt.close(g.figure)
    n_steps_to_show = 4

    # ----------  basic theming ----------
    sns.set_theme(
        style="white",
        rc={
            "axes.facecolor": (0, 0, 0, 0),
            "figure.figsize": (1.5 * n_steps_to_show, 1.5),   # widen for columns
        },
    )

    # ----------  collect the data ----------
    data = np.array(sampled_data[("scores", "strong_reject", "p_harmful")])

    # choose exactly n_steps_to_show equally-spaced indices
    all_step_idxs = [0] + list(generate_sample_sizes(data.shape[1]-1))
    if data.shape[1]-1 not in all_step_idxs:
        all_step_idxs.append(data.shape[1])

    # Select exactly n_steps_to_show equally-spaced indices
    if len(all_step_idxs) > n_steps_to_show:
        # Use numpy to select evenly spaced indices
        indices = np.linspace(0, len(all_step_idxs) - 1, n_steps_to_show + 1, dtype=int)
        indices = [indices[0], *indices[2:]]
        step_idxs = [all_step_idxs[i] for i in indices]
    else:
        step_idxs = all_step_idxs

    ridge_rows = []
    for idx in step_idxs:
        ridge_rows.extend(
            {
                "step": f"Step {idx}",
                r"$h(Y)$": val,
            }
            for val in data[:, idx, :].ravel()
        )

    df = pd.DataFrame(ridge_rows)

    # ----------  build the faceted plot ----------
    unique_steps = sorted(df["step"].unique(), key=lambda x: int(x.split()[1]))
    pal = sns.cubehelix_palette(int(len(unique_steps)*1.5), rot=-0.25, light=0.7)

    g = sns.FacetGrid(
        df,
        col="step",
        hue="step",
        palette=pal,
        col_order=unique_steps,
        sharey=False,          # independent y-axis per column
        aspect=1,
        height=2.5,
    )

    # densities
    g.map(sns.kdeplot, r"$h(Y)$", bw_adjust=0.5, fill=True, alpha=1, linewidth=0, zorder=1)
    g.map(sns.kdeplot, r"$h(Y)$", bw_adjust=0.5, color="w", lw=3, zorder=0)
    # # Store the y-limits from the first plot to apply to all plots
    # first_ax = g.axes.flat[0]
    # first_ylim = first_ax.get_ylim()

    # # Apply the same y-limits to all subplots
    # for ax in g.axes.flat:
    #     ax.set_ylim(first_ylim)
    # Add a single y-axis tick at the density value of h(Y)=0 for each subplot

    g.set(yticks=[], ylabel="")      # hide y-ticks

    for ax in g.axes.flat:
        # Get the KDE line data
        kde_line = ax.lines[0]  # The first line should be the KDE plot
        x_data = kde_line.get_xdata()
        y_data = kde_line.get_ydata()

        # Find the density value at h(Y)=0 by interpolating
        if len(x_data) > 0 and len(y_data) > 0:
            # Find the closest x value to 0 or interpolate
            if 0 in x_data:
                density_at_zero = y_data[x_data == 0][0]
            else:
                # Interpolate to find density at x=0
                from scipy.interpolate import interp1d
                if x_data.min() <= 0 <= x_data.max():
                    interp_func = interp1d(x_data, y_data, kind='linear', bounds_error=False, fill_value=0)
                    density_at_zero = interp_func(0)
                else:
                    density_at_zero = 0

            # Set a single y-tick at this density value
            ax.set_yticks([density_at_zero/2, density_at_zero])
            ax.set_yticklabels([f'{density_at_zero/2:.1f}', f'{density_at_zero:.1f}'])
            ax.tick_params(axis='y', labelsize=12, pad=-2)

    # central-tendency & cut-off lines
    def add_mean_lines(x, **kwargs):
        ax = plt.gca()
        mean_val = np.mean(x)
        median_val = np.median(x)
        p95 = np.percentile(x, 95)
        ax.axvline(median_val, ls="--", lw=1, color="black", ymax=0.5, alpha=0.7)
        ax.axvline(p95,       ls="--", lw=1, color="blue",  ymax=0.5, alpha=0.7)
        ax.axvline(mean_val,  ls="-",  lw=1, color="red",   ymax=0.5, alpha=0.7)

    g.map(add_mean_lines, r"$h(Y)$")

    # Reduce horizontal spacing between subplots
    # g.figure.subplots_adjust(wspace=-0.0)

    # aesthetics
    g.set_titles("")                 # no subplot headers
    tick_vals = np.linspace(0, 1, 6)  # [0. , 0.2, 0.4, 0.6, 0.8, 1.]
    g.set(xticks=tick_vals)
    for ax in g.axes.flat:
        ax.tick_params(axis='x', pad=0, labelsize=12)
    g.set_xlabels("Harmfulness", fontsize=14)
    g.set(xlim=(0, 1))
    # g.despine(left=True)
    plt.style.use("science")
    # -------------  y-axis label on first facet -------------
    first_ax = g.axes.flat[0]
    first_ax.set_ylabel("Density", fontsize=13)#, labelpad=-14)

    # optional: make sure the other facets stay unlabeled
    for ax in g.axes.flat[1:]:
        ax.set_ylabel("")
    # -------------  add "Step x" labels -------------
    for ax, step in zip(g.axes.flat, unique_steps):
        if step[-1] == "9":
            step = "Step " + str(int(step.split()[1])+1)
        ax.text(
            0.5, 0.95, step,                       # centered just above each panel
            ha="center", va="bottom",
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold"                      # optional, adjust to taste
        )

    # build a single, vertical legend that mimics the example image
    legend_elements = [
        Line2D([0], [0], color="black",  lw=1, label="Median", ls="--"),
        Line2D([0], [0], color="red",   lw=1, label="Greedy"),
        Line2D([0], [0], color="blue",  lw=1, label="95th percentile", ls="--"),
    ]

    # Determine if we have a single subplot
    if len(g.axes.flat) == 1:
        bbox_anchor = (0.2, 0.8)
    else:
        bbox_anchor = (0.055, 0.8)

    g.figure.legend(
        handles=legend_elements,
        loc="upper left",              # anchor to top-left of the figure
        bbox_to_anchor=bbox_anchor,   # fine-tune position (x, y in fig-coords)
        frameon=False,
        ncol=1,                        # vertical stack
        handletextpad=0.4,
        labelspacing=0.3,
        borderaxespad=0.0,
    )

        # Add horizontal time arrow above the plots
    if len(g.axes.flat) > 1:
        from matplotlib.patches import FancyArrowPatch

        # Get the positions of the first and last subplots
        first_ax = g.axes.flat[0]
        last_ax = g.axes.flat[-1]

        # Get the positions in figure coordinates
        first_pos = first_ax.get_position()
        last_pos = last_ax.get_position()

        # Calculate arrow position (slightly above the plots)
        arrow_y = first_pos.y1 + 0.08  # 8% above the top of the plots
        arrow_start_x = first_pos.x0 + 0.1 * first_pos.width  # 10% into first subplot
        arrow_end_x = last_pos.x1 - 0.1 * last_pos.width     # 90% into last subplot

        # Create and add the arrow patch
        arrow = FancyArrowPatch((arrow_start_x, arrow_y), (arrow_end_x, arrow_y),
                               connectionstyle="arc3",
                               arrowstyle='-|>',
                               mutation_scale=10,
                               linewidth=0.75,
                               color='black',
                               alpha=1.0,
                               transform=g.figure.transFigure)
        g.figure.patches.append(arrow)

    # --------- save / close ----------
    if threshold is None:
        filename = (
            f"evaluate/distributional_paper/ridge_plots/"
            f"{model_title}_{cfg['title_suffix']}_side_by_side.pdf"
        )
    else:
        filename = (
            f"evaluate/distributional_paper/ridge_plots/"
            f"{model_title}_{cfg['title_suffix']}_side_by_side_t={threshold}.pdf"
        )
    g.figure.savefig(filename.replace(' ', '_'), bbox_inches="tight")
    plt.close(g.figure)
    # ---------- NEW: ratio line plot ----------
    num_steps = data.shape[1]
    ratios_1 = []
    ratios_2 = []

    for step_idx in range(num_steps):
        vals = data[:, step_idx, :].flatten()
        # Original ratio: [0.1,0.5] vs [0.5,1.0]
        n_low_1  = np.sum((vals > 0.50) & (vals <= 1.0))
        n_high_1 = np.sum((vals > 0.10) & (vals <= 1.0))
        ratio_1 = n_low_1 / n_high_1 if n_high_1 else np.nan
        ratios_1.append(ratio_1)

        # New ratio: [0.0,0.1] vs [0.1,1.0]
        n_low_2  = np.sum((vals >= 0.10) & (vals <= 1.0))
        n_high_2 = np.sum((vals >= 0.00) & (vals <= 1.0))
        ratio_2 = n_low_2 / n_high_2 if n_high_2 else np.nan
        ratios_2.append(ratio_2)

    # Create figure with two subfigures
    fig, (ax1) = plt.subplots(1, 1, figsize=(6.5, 2.65))
    plt.style.use("science")

    # First subfigure: original ratio plots
    sns.lineplot(x=np.arange(num_steps), y=ratios_2, label=r"$P(\text{¬refusal})$", ax=ax1, marker="o" if num_steps == 1 else None)
    sns.lineplot(x=np.arange(num_steps), y=ratios_1, linestyle="--", label=r"$P(\text{harmful} \mid \text{¬refusal})$", ax=ax1, marker="x" if num_steps == 1 else None)
    ax1.yaxis.set_major_locator(MaxNLocator(nbins="auto", integer=False))
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Frequency")
    ax1.set_title(f"{model_title}")
    # ax1.set_ylim(bottom=0.4)

    # Place legend to the left of the first subplot
    ax1.legend(bbox_to_anchor=(-0.3, 0.5), loc='center right')

    plt.tight_layout()
    if threshold is None:
        plt.savefig(
            f"evaluate/distributional_paper/ratio_plots/{model_title.replace(' ', '_')}_{cfg['title_suffix'].replace(' ', '_')}.pdf",
            bbox_inches="tight"
        )
    else:
        plt.savefig(
            f"evaluate/distributional_paper/ratio_plots/{model_title.replace(' ', '_')}_{cfg['title_suffix'].replace(' ', '_')}_t={threshold}.pdf",
            bbox_inches="tight"
        )
    plt.close()


def optimization_progress_plot(
    results: dict[str,np.ndarray],
    title: str = "Optimization Progress Analysis",
    metric: tuple[str, ...] = ('scores', 'strong_reject', 'p_harmful'),
    cumulative: bool = False,
    threshold: float|None = None,
    step_subset: tuple[int, ...] = None,
    verbose: bool = True,
):
    """
    Plot optimization progress by showing harmfulness level vs number of samples
    for different optimization steps. This reveals whether attacks actually improve
    during optimization.

    Parameters
    ----------
    results : dict
        Attack results data
    title : str
        Plot title
    metric : tuple
        Metric to analyze
    cumulative : bool
        Whether to use cumulative sampling
    threshold : float, optional
        Threshold for binary classification
    step_subset : tuple, optional
        Specific steps to show. If None, shows evenly spaced steps
    verbose : bool
        Whether to print verbose output
    """
    y, _, _, _ = preprocess_data(results, metric, threshold, None)
    n_runs, n_steps, n_total_samples = y.shape

    # Select steps to show
    if step_subset is None:
        # Show specific steps: 0, 1, 2, 5, 10, 20, 50, 100, 200, 500 if they exist, plus the last one
        candidate_steps = [0, 1, 2, 5, 10, 20, 50, 100, 200, 500]
        steps_to_show = [s for s in candidate_steps if s < n_steps]
        # Always include the last step if it's not already included
        if n_steps > 0 and (n_steps - 1) not in steps_to_show:
            steps_to_show.append(n_steps - 1)
    else:
        steps_to_show = [s for s in step_subset if s < n_steps]

    plt.figure(figsize=(8, 5))

    # Color map for different steps
    colors = plt.cm.viridis(np.linspace(0, 1, len(steps_to_show)))

    rng = np.random.default_rng(42)  # Fixed seed for reproducibility

    # For each optimization step, create a curve
    for i, step_idx in enumerate(steps_to_show):
        sample_counts = range(1, n_total_samples + 1)
        harm_levels = []

        # Calculate harmfulness level for different sample counts at this step
        for n_samples in sample_counts:
            step_values = []

            # Multiple random samples for smoother curves
            for _ in range(50):  # 50 random samples for averaging
                if cumulative and step_idx > 0:
                    # Cumulative: max of (best from previous steps, best from current step)
                    samples_up_to_now = y[:, :step_idx, rng.choice(n_total_samples, size=1, replace=False)].max(axis=1)[:, 0]
                    samples_at_step = y[:, step_idx, rng.choice(n_total_samples, size=n_samples, replace=False)].max(axis=-1)
                    step_value = np.stack([samples_up_to_now, samples_at_step], axis=1).max(axis=1).mean(axis=0)
                else:
                    # Non-cumulative: just the best from current step
                    step_value = y[:, step_idx, rng.choice(n_total_samples, size=n_samples, replace=False)].max(axis=-1).mean(axis=0)

                step_values.append(step_value)

            harm_levels.append(np.mean(step_values))

        # Plot the curve for this optimization step
        label = f"Step {step_idx}" if step_idx < n_steps - 1 else f"Step {step_idx + 1} (Final)"
        plt.plot(sample_counts, harm_levels,
                color=colors[i], linewidth=2.5, marker='o', markersize=4,
                label=label, alpha=0.9)

    # Styling
    plt.xlabel("Number of Samples", fontsize=14)
    if threshold is None:
        plt.ylabel(r"$s_{harm@n}$", fontsize=16)
    else:
        plt.ylabel(r"${ASR}@n$", fontsize=14)

    plt.grid(True, alpha=0.3)
    plt.xlim(1, n_total_samples)
    plt.ylim(bottom=0)

    # Add legend
    plt.legend(loc='lower right', fontsize=11)

    plt.tight_layout()

    # Save plot
    if threshold is None:
        plt.savefig(f"evaluate/distributional_paper/optimization_progress/{title.replace(' ', '_')}.pdf")
    else:
        plt.savefig(f"evaluate/distributional_paper/optimization_progress/{title.replace(' ', '_')}_t={threshold}.pdf")
    plt.close()

    if verbose:
        logging.info(f"Optimization progress plot saved for {title}")
        logging.info(f"Steps analyzed: {steps_to_show}")
        logging.info(f"Sample range: 1 to {n_total_samples}")

        # Calculate improvement metrics
        if len(steps_to_show) >= 2:
            first_step = steps_to_show[0]
            last_step = steps_to_show[-1]

            # Compare harm levels at max samples
            first_harm = []
            last_harm = []

            for _ in range(50):  # More samples for accurate comparison
                if cumulative and first_step > 0:
                    first_val = np.stack([
                        y[:, :first_step, rng.choice(n_total_samples, size=1, replace=False)].max(axis=1)[:, 0],
                        y[:, first_step, rng.choice(n_total_samples, size=n_total_samples, replace=False)].max(axis=-1)
                    ], axis=1).max(axis=1).mean(axis=0)
                else:
                    first_val = y[:, first_step, rng.choice(n_total_samples, size=n_total_samples, replace=False)].max(axis=-1).mean(axis=0)

                if cumulative and last_step > 0:
                    last_val = np.stack([
                        y[:, :last_step, rng.choice(n_total_samples, size=1, replace=False)].max(axis=1)[:, 0],
                        y[:, last_step, rng.choice(n_total_samples, size=n_total_samples, replace=False)].max(axis=-1)
                    ], axis=1).max(axis=1).mean(axis=0)
                else:
                    last_val = y[:, last_step, rng.choice(n_total_samples, size=n_total_samples, replace=False)].max(axis=-1).mean(axis=0)

                first_harm.append(first_val)
                last_harm.append(last_val)

            improvement = np.mean(last_harm) - np.mean(first_harm)
            relative_improvement = improvement / np.mean(first_harm) if np.mean(first_harm) > 0 else 0

            logging.info(f"Absolute improvement (first to last): {improvement:.4f}")
            logging.info(f"Relative improvement: {relative_improvement:.2%}")


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
    flops_per_step_fn = lambda x: FLOPs_PER_STEP[atk_name](x, num_model_params(model))

    if analysis_type == "pareto":
        pareto_plot(
            sampled_data,
            baseline_data,
            title=f"{model_title} {cfg['title_suffix']}",
            cumulative=cfg["cumulative"],
            metric=METRIC,
            flops_per_step=flops_per_step_fn,
            threshold=0.5,
            color_scale="sqrt",
        )
    elif analysis_type == "non_cumulative_pareto":
        non_cumulative_pareto_plot(
            sampled_data,
            baseline_data,
            title=f"{model_title} {cfg['title_suffix']}",
            metric=METRIC,
            flops_per_step=flops_per_step_fn,
            threshold=0.5,
            color_scale="sqrt",
        )
    elif analysis_type == "flops_ratio":
        flops_ratio_plot(
            sampled_data,
            baseline_data,
            title=f"{model_title} {cfg['title_suffix']} FLOPs Ratio",
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
        histogram_plot(
            sampled_data,
            model_title,
            atk_name,
            cfg,
            threshold=None,
        )
    elif analysis_type == "ridge":
        ridge_plot(
            sampled_data,
            model_title,
            cfg,
            threshold=None,
        )
    elif analysis_type == "histogram_2":
        histogram_2_plot(
            sampled_data,
            model_title,
            cfg,
            threshold=None,
        )
    elif analysis_type == "flops_breakdown":
        flops_breakdown_plot(
            sampled_data,
            baseline_data,
            title=f"{model_title} {cfg['title_suffix']} FLOPs Breakdown",
            cumulative=cfg["cumulative"],
            metric=METRIC,
            flops_per_step=flops_per_step_fn,
            threshold=None,
            color_scale="sqrt",
        )
    elif analysis_type == "optimization_progress":
        optimization_progress_plot(
            sampled_data,
            title=f"{model_title} {cfg['title_suffix']} Optimization Progress",
            cumulative=False,  # each step is considered independent
            metric=METRIC,
            threshold=None,
        )
    else:
        raise ValueError(f"Unknown analysis type: {analysis_type}")


def run_comparative_analysis(
    model: str,
    model_title: str,
    analysis_type: str = "comparative_pareto",
    threshold: float|None = None,
):
    """
    Run comparative analysis across multiple attacks for a single model.
    """
    logging.info(f"{analysis_type.title()} Analysis: {model_title}")

    # Collect data from all attacks for this model
    attacks_data = {}
    flops_per_step_fns = {}

    for atk_name, cfg in ATTACKS:
        try:
            # Fetch attack data
            sampled_data = fetch_data(model, cfg.get("attack_override", atk_name), cfg["sample_params"](),
                                     DATASET_IDX, GROUP_BY)

            # Apply post-processing if needed
            if post := cfg.get("postprocess"):
                post(sampled_data, METRIC)

            # Use title_suffix as key to distinguish between different configs of the same attack
            config_key = cfg['title_suffix']

            # Store the data and config
            attacks_data[config_key] = (sampled_data, cfg)

            # Store flops function (using default parameter to capture current value)
            flops_per_step_fns[config_key] = lambda x, attack=atk_name: FLOPs_PER_STEP[attack](x, num_model_params(model))

        except Exception as e:
            logging.warning(f"Could not load data for {atk_name} ({cfg.get('title_suffix', 'unknown config')}): {e}")
            continue

    # Generate comparative plot
    if analysis_type == "comparative_pareto":
        comparative_pareto_plot(
            model=model,
            model_title=model_title,
            attacks_data=attacks_data,
            title=f"{model_title}",
            metric=METRIC,
            flops_per_step_fns=flops_per_step_fns,
            threshold=None,
            baseline_attacks={"gcg", "beast", "pair", "autodan"},
        )
    elif analysis_type == "multi_attack_non_cumulative_pareto":
        multi_attack_non_cumulative_pareto_plot(
            attacks_data=attacks_data,
            model_title=model_title,
            title=f"{model_title}",
            metric=METRIC,
            threshold=None,
        )
    else:
        raise ValueError(f"Unknown comparative analysis type: {analysis_type}")


# ----------------------------------------------------------------------------------
# Configuration and Constants
# ----------------------------------------------------------------------------------

MODELS = {
    "meta-llama/Meta-Llama-3.1-8B-Instruct": "Llama 3.1 8B",
    "google/gemma-3-1b-it": "Gemma 3 1B",
    "GraySwanAI/Llama-3-8B-Instruct-RR": "Llama 3 8B CB",
    "Unispac/Llama2-7B-Chat-Augmented": "Llama 2 7B DA",
}

FLOPs_PER_STEP = {
    "autodan": lambda s, c: 69845248149248 // num_model_params("Qwen/Qwen2.5-0.5B-Instruct") * c,
    "gcg":     lambda s, c: int(1e14) + 14958709489152 // num_model_params("Qwen/Qwen2.5-0.5B-Instruct") * c,
    "beast":   lambda s, c: 10447045889280 // num_model_params("Qwen/Qwen2.5-0.5B-Instruct") * c,
    "pair":    lambda s, c: 83795198566400 + 78737584640 // num_model_params("Qwen/Qwen2.5-0.5B-Instruct") * c,
    "direct":  lambda s, c: 0,
    "bon":     lambda s, c: 0,
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
            "mask_undecided_tokens": False,
            "generation_config": {"num_return_sequences": 50, "temperature": 0.7},
        },
        baseline_params=lambda: {
            "mask_undecided_tokens": False,
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
        title_suffix="BoN temp 1.0",
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
DATASET_IDX = list(range(100))

def main(fail: bool = False, analysis_types=None):
    if analysis_types is None:
        analysis_types = ["pareto", "non_cumulative_pareto", "flops_ratio", "ideal_ratio", "histogram", "histogram_2", "ridge", "flops_breakdown", "optimization_progress", "comparative_pareto", "multi_attack_non_cumulative_pareto"]
    for analysis_type in analysis_types:
        logging.info("\n" + "="*80)
        logging.info(f"GENERATING {analysis_type.upper().replace('_', ' ')} PLOTS")
        logging.info("="*80)

        if analysis_type in ["comparative_pareto", "multi_attack_non_cumulative_pareto"]:
            # For comparative analysis, iterate over models only
            for model_key, model_title in MODELS.items():
                logging.info(f"Model: {model_key}")
                try:
                    run_comparative_analysis(model_key, model_title, analysis_type)
                except Exception as e:
                    if fail:
                        raise e
                    logging.info(f"Error running {analysis_type} analysis for {model_title}: {e}")
        else:
            # For individual attack analysis, iterate over both models and attacks
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

def make_hero_plot():
    asr_labels = ["PAIR", "AutoDAN", "GCG"]
    asr_delta = [0.16, 0.21, 0.37]
    speedup_labels = asr_labels
    speedups = [2.7, 8.9, 137.5]
    # Create side-by-side subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5, 2.65))
    cmap = plt.get_cmap("viridis")

    # Left subplot: Speedup
    bars1 = ax1.bar(speedup_labels, speedups, color=cmap(np.linspace(0, 1, len(speedup_labels)+1)[1:]), alpha=0.8, edgecolor='black')
    ax1.set_ylabel("Speedup (FLOPs)", fontsize=16)
    ax1.set_ylim(0, 170)
    # ax1.set_title("Computational Efficiency", fontsize=14)
    ax1.grid(True, alpha=0.3, axis='y')
    # ax1.set_xticks(["PAIR", "AutoDAN", "GCG"], rotation=45, ha='right')

    # Add horizontal line at y=1 for reference
    # ax1.axhline(y=1, color='red', linestyle='--', alpha=0.7, linewidth=1)

    # Add value labels on bars
    for bar, value in zip(bars1, speedups):
        ax1.annotate(f'{value:.1f}x',
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 5),
                    textcoords='offset points',
                    ha='center', va='bottom', fontsize=14)

    # Right subplot: ASR Delta
    bars2 = ax2.bar(asr_labels, asr_delta, color=cmap(np.linspace(0, 1, len(asr_labels)+1)[1:]), alpha=0.8, edgecolor='black')
    ax2.set_ylabel(r"$\Delta$ ASR", fontsize=16)
    ax2.set_ylim(0, 0.45)
    # ax2.set_title("Attack Success Rate Improvement", fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')
    # ax2.set_xticks(labels=["PAIR", "AutoDAN", "GCG"], rotation=45, ha='right')
    # Make ticks bigger
    ax1.tick_params(axis='both', which='major', labelsize=14)
    ax2.tick_params(axis='both', which='major', labelsize=14)
    ax1.tick_params(axis='x', which='major', labelrotation=45)
    ax2.tick_params(axis='x', which='major', labelrotation=45)

    # Add value labels on bars
    for bar, value in zip(bars2, asr_delta):
        ax2.annotate(f'+{value:.2f}' if value > 0 else f'{value:.2f}',
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 5),
                    textcoords='offset points',
                    ha='center', va='bottom', fontsize=14)

    plt.tight_layout()
    plt.savefig("evaluate/distributional_paper/mini_hero_plot.pdf", dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    make_hero_plot()
    import argparse
    parser = argparse.ArgumentParser(description='Generate plots for distributional paper')
    parser.add_argument('--fail', action='store_true', help='Override flag to fail')
    parser.add_argument('--analysis_types', "-p", nargs='+', help='Analysis types to run')
    args = parser.parse_args()

    main(args.fail, args.analysis_types)
