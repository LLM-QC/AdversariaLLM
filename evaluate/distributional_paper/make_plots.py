import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from scipy.interpolate import interp1d
from matplotlib.colors import LogNorm


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
        x_interp = np.logspace(13, np.log10(max_cost+1), n_x_points)
    else:
        x_interp = np.linspace(0, max_cost+1, n_x_points)


    # ---------- scatter all points ----------
    plt.figure(figsize=(9, 6))
    if plot_points:
        if color_scale == "log":
            color_norm = LogNorm()
        else:
            color_norm = None
        sc = plt.scatter(cost, mean_p, c=n_samp, cmap="viridis", alpha=0.15, s=3, norm=color_norm)
    plt.xlabel("FLOPS")
    if threshold is None:
        plt.ylabel("Mean p_harmful")
    else:
        plt.ylabel(f"Mean ASR (threshold: {threshold})")

    # ---------- overlay Pareto frontiers ----------
    cmap = plt.get_cmap("viridis")
    if color_scale == "log":
        norm = LogNorm(n_samp.min(), n_samp.max())
    else:
        norm = plt.Normalize(n_samp.min(), n_samp.max())
    rng = np.random.default_rng()

    n_smoothing = 50
    if plot_frontiers:
        for j in sample_levels_to_plot:
            xs = []
            ys = []
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
            plt.plot(
                x_interp,
                np.mean(y_interp, axis=0),
                marker="o",
                linewidth=1.8,
                markersize=2,
                label=f"{j} samples",
                color=color,
            )

    if plot_envelope:
        # ---------- overlay Pareto frontiers ----------
        n_smoothing = 50
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

        # color = "g" #[
        # plt.plot(
        #     np.arange(max_cost+1),
        #     y_envelope,
        #     marker="o",
        #     linewidth=1.8,
        #     markersize=4,
        #     label=f"envelope",
        #     color=color,
        # )
        color = [cmap(norm(argmax[i])) for i in range(len(argmax))]
        plt.scatter(x_interp, y_envelope, c=color, s=2)

    title_suffix = ""

    y = np.array(baseline[metric]) # (B, n_steps, n_samples)
    if threshold is not None:
        y = y > threshold

    baseline_flops_sampling = np.array(baseline["flops_sampling"])
    if "flops" in baseline:
        baseline_flops_optimization = np.array(baseline["flops"]) # (B, n_steps)
    else:
        baseline_flops_optimization = np.zeros_like(baseline_flops_sampling) # (B, n_steps)
        if flops_per_step is not None:
            baseline_flops_optimization += flops_per_step(np.arange(baseline_flops_optimization.shape[1]))

    if y is not None:
        title_suffix = f" ({n_runs}, {y.shape[0]})"
        if verbose:
            print(n_runs, "for main")
            print(y.shape[0], "for baseline")
        n_runs, n_steps, n_total_samples = y.shape
        assert n_total_samples == 1

        rng = np.random.default_rng()
        pts = []  # (cost, step, n_samples, mean_p)
        for i in range(0, n_steps, 1):
            for j in range(1, n_total_samples + 1, 1):
                pts.append(subsample_and_aggregate(i, j, cumulative, y, baseline_flops_optimization, baseline_flops_sampling, rng))

        pts = np.asarray(pts)
        cost, step_idx, n_samp, mean_p = pts.T

        # ---------- scatter all points ----------
        # sc = plt.scatter(cost, mean_p, c="r", alpha=0.35, s=4)

        # ---------- overlay Pareto frontiers ----------
        if plot_frontiers or plot_envelope:
            mask = n_samp == 1
            fx, fy = _pareto_frontier(cost[mask], mean_p[mask], method=frontier_method)
            y_interp = interp1d(fx, fy, kind="previous", bounds_error=False, fill_value=(0, max(fy)))(x_interp)
            plt.plot(
                x_interp,
                y_interp,
                marker="o",
                linewidth=1.8,
                markersize=2,
                label=f"greedy",
                color="r",
            )
    plt.title(title + title_suffix)
    plt.grid(True, linewidth=0.3)

    plt.xscale(x_scale)
    plt.legend(title="Frontiers", loc="upper left" if x_scale == "log" else "lower right")
    plt.tight_layout()
    plt.savefig(f"evaluate/distributional_paper/pareto_plots/{title}.pdf")



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
]

METRIC = ("scores", "strong_reject", "p_harmful")
GROUP_BY = {("model",)}
DATASET_IDX = list(range(50))


# Helper ---------------------------------------------------------------------------
def run_attack(
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
    )

# Main loop ------------------------------------------------------------------------
for model_key, model_title in MODELS.items():
    print("Model:", model_key)
    for atk_name, atk_cfg in ATTACKS:
        try:
            run_attack(model_key, model_title, atk_name, atk_cfg)
        except Exception as e:
            print(f"Error running attack {atk_name}, atk_cfg: {atk_cfg['title_suffix']}: {e}")
