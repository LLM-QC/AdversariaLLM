import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # determinism
import copy
import json
import logging
import sys

import filelock
import hydra
import torch
from judgezoo import Judge
from omegaconf import DictConfig, ListConfig, OmegaConf
from tqdm import tqdm

from src.errors import print_exceptions
from src.io_utils import CompactJSONEncoder, get_mongodb_connection, delete_orphaned_runs, get_filtered_and_grouped_paths

torch.use_deterministic_algorithms(True, warn_only=True)  # determinism
torch.backends.cuda.matmul.allow_tf32 = True


def collect_run_paths(suffixes: list[str]|str, classifier: str, filter_by: dict|None) -> list[str]:
    """
    Collect paths to run files that have not been scored by the specified classifier.

    Args:
        suffixes: List of suffixes that must be in the path
        classifier: Name of the classifier to check in scored_by

    Returns:
        List of paths to run files
    """

    if not isinstance(suffixes, (list, ListConfig)):
        suffixes = [str(suffixes)]
    delete_orphaned_runs()
    db = get_mongodb_connection()
    collection = db.runs

    # Use MongoDB's find() method to get all documents
    all_results = list(collection.find())
    paths = []
    for item in all_results:
        log_file = item["log_file"]
        date_time_string = log_file.split("/")[-3]
        if classifier in item["scored_by"]:
            continue
        if any(date_time_string.endswith(suffix) for suffix in suffixes):
            paths.append(log_file)
    # remove duplicates
    paths = list(set(paths))
    if filter_by:
        filtered_paths = set(get_filtered_and_grouped_paths(OmegaConf.to_container(filter_by, resolve=True))[("all",)])
        paths = [p for p in paths if p in filtered_paths]
    return sorted(paths, reverse=True)


@hydra.main(config_path="./conf", config_name="judge", version_base="1.3")
@print_exceptions
def run_judges(cfg: DictConfig) -> None:
    logging.info("-------------------")
    logging.info("Commencing judge run")
    logging.info("-------------------")
    logging.info(cfg)

    paths = collect_run_paths(cfg.suffixes, cfg.classifier, cfg.filter_by)
    if not paths:
        logging.info("No unjudged paths found")
        return
    logging.info(f"Found {len(paths)} paths")
    logging.info("Loading judge...")
    judge = None
    n = 0
    pbar = tqdm(paths, file=sys.stdout)
    for path in pbar:
        with filelock.FileLock(path + ".lock") as lock:
            try:
                attack_run = json.load(open(path))
                if (cfg.classifier == "overrefusal") != (attack_run["config"]["dataset"] in ("or_bench", "xs_test")):
                    continue
                for subrun in attack_run["runs"]:
                    original_conversation = subrun["original_prompt"]
                    modified_prompts = []
                    if cfg.classifier in subrun["steps"][0]["scores"]:
                        continue
                    # Late init to avoid loading the judge if not needed
                    if judge is None:
                        judge = Judge.from_name(cfg.classifier)
                    for step in subrun["steps"]:
                        model_input = step["model_input"]
                        completions: list = step["model_completions"]
                        for completion in completions:
                            modified_prompt = copy.deepcopy(original_conversation)
                            if modified_prompt[-1]["role"] == "assistant":
                                modified_prompt[-1]["content"] = model_input[-1]["content"] + completion
                            else:
                                modified_prompt.append({"role": "assistant", "content": completion})
                            modified_prompts.append(modified_prompt)
                    pbar.set_description(f"{len(modified_prompts)} | {n} total")
                    results = judge(modified_prompts)
                    if all(r is None for r in results):
                        continue
                    i = 0
                    for step in subrun["steps"]:
                        n_completions = len(step["model_completions"])
                        step["scores"][cfg.classifier] = {k: v[i:i+n_completions] for k, v in results.items()}
                        i += n_completions
                        n += n_completions
                json.dump(attack_run, open(path, "w"), indent=2, cls=CompactJSONEncoder)
                db = get_mongodb_connection()
                collection = db.runs
                collection.update_many({"log_file": path}, {"$addToSet": {"scored_by": cfg.classifier}})
            except Exception as e:
                print(path, str(e))
                os.remove(path + ".lock")
                raise Exception(f"Error in {path}. Original exception: {e}") from e
        if os.path.exists(path + ".lock"):
            os.remove(path + ".lock")
    print(f"Judged {n} completions")


if __name__ == "__main__":
    run_judges()
