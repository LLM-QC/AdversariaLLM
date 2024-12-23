import json
import pandas as pd
import numpy as np


def filter_runs(
    runs: pd.DataFrame,
    model_name=None,
    algorithm=None,
    algorithm_params=None,
):
    def matches(value, target):
        if pd.isna(value):  # Handle NaN values gracefully
            return False
        if target is None:
            return True

        if isinstance(target, str):
            return value.lower() == target.lower()
        defaults = {
            'normalize_alpha': False,
        }
        if isinstance(target, dict):
            return all(value.get(k, defaults.get(k, None)) == v for k, v in target.items())
        return any(value.lower() == t.lower() for t in target)

    # Apply the matches function to filter runs
    filtered_runs = runs[
        runs["model"].apply(lambda model: matches(model, model_name))
        & runs["algorithm"].apply(lambda algo: matches(algo, algorithm))
        & runs["algorithm_params"].apply(lambda algorithm_p: matches(json.loads(algorithm_p), algorithm_params))
    ].copy()
    return filtered_runs

def run_has_metric(run, metric):
    return metric in run and run[metric] is not None and not any(m is None for m in run[metric])


def make_metrics_cumulative(runs: pd.DataFrame):
    # Methods which generate anyways can use accumulate here
    # Methods which only generate once at the very end have to use lambda x: x
    aggregator_functions = {
        "ample_gcg": np.maximum.accumulate,
        "autodan": np.maximum.accumulate,
        "direct": np.maximum.accumulate,
        "gcg": lambda x: x,
        "human_jailbreaks": np.maximum.accumulate,
        "pair": np.maximum.accumulate,
        "pgd": lambda x: x,
        "pgd_one_hot": lambda x: x,
    }

    metric_types = {
        "asr_cais": 1,
        "asr_llama_guard_3_8b": 1,
        "asr_total": 1,
        "sr_cais": -1,
        "loss": -1,
        "asr_prefix": 1,
        "sr_prefix": -1,
    }

    for idx, run in runs.iterrows():
        for metric in metric_types:
            if not run_has_metric(run, metric):
                continue
            m = np.array(run[metric]) * metric_types[metric]
            runs.at[idx, metric] = (aggregator_functions[run["algorithm"]](m) * metric_types[metric]).tolist()
    return runs


def pad_to_max_length(runs: pd.DataFrame):
    # Repeats the last value of each metric to make all runs the same length
    metric_types = ["asr_cais", "asr_llama_guard_3_8b", "asr_total", "sr_cais", "loss", "asr_prefix", "sr_prefix"]
    for metric in metric_types:
        if metric in runs:
            # Determine the maximum length for the current metric
            max_length = runs[metric].apply(len).max()

            # Pad each row of the column to the maximum length
            runs.loc[:, metric] = runs[metric].apply(
                lambda x: x + [x[-1]] * (max_length - len(x)) if len(x) < max_length else x
            )
    return runs