import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # determinism
import copy
import json
import logging
import sys

import filelock
import hydra
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from src.errors import print_exceptions
from src.io_utils import get_mongodb_connection
from src.judges import Judge

torch.use_deterministic_algorithms(True, warn_only=True)  # determinism
torch.backends.cuda.matmul.allow_tf32 = True


def collect_run_paths(db, suffixes: list[str]|str, classifier: str) -> list[str]:
    """
    Collect paths to run files that have not been scored by the specified classifier.

    Args:
        save_dir: Directory containing the database
        suffixes: List of suffixes that must be in the path
        classifier: Name of the classifier to check in scored_by

    Returns:
        List of paths to run files
    """
    if not isinstance(suffixes, list):
        suffixes = [str(suffixes)]
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
    return sorted(paths, reverse=True)


@hydra.main(config_path="./conf", config_name="judge", version_base="1.3")
@print_exceptions
def run_judge(cfg: DictConfig) -> None:
    logging.info("-------------------")
    logging.info("Commencing judge run")
    logging.info("-------------------")
    logging.info(cfg)

    paths = collect_run_paths(cfg.save_dir, cfg.suffixes, cfg.classifier)
    logging.info(f"Found {len(paths)} paths")
    logging.info("Loading judge...")
    judge = Judge.from_name(cfg.classifier)
    n = 0
    pbar = tqdm(paths, file=sys.stdout)
    for path in pbar:
        with filelock.FileLock(path + ".lock") as lock:
            try:
                run = json.load(open(path))
                if (cfg.classifier == "overrefusal") != (run["config"]["dataset"] in ("or_bench", "xs_test")):
                    continue
                for subrun in run["runs"]:
                    prompt = subrun["original_prompt"]
                    modified_prompts = []
                    if cfg.classifier in subrun["steps"][0]["scores"]:
                        continue
                    for step in subrun["steps"]:
                        completions: list = step["model_completions"]
                        for completion in completions:
                            modfied_prompt = copy.deepcopy(prompt)
                            modfied_prompt[-1]["content"] += completion
                            modified_prompts.append(modfied_prompt)
                    pbar.set_description(f"{len(modified_prompts)} | {n} total")
                    results = judge(modified_prompts)
                    if results is None:
                        continue
                    i = 0
                    for step in subrun["steps"]:
                        n_completions = len(step["model_completions"])
                        step["scores"][cfg.classifier] = {k: v[i:i+n_completions] for k, v in results.items()}
                        i += n_completions
                        n += n_completions
                json.dump(run, open(path, "w"), indent=4)
                db = get_mongodb_connection()
                collection = db.runs
                collection.update_one({"log_file": path}, {"$addToSet": {"scored_by": cfg.classifier}})
            except Exception as e:
                print(path, str(e))
                os.remove(path + ".lock")
                raise Exception(f"Error in {path}. Original exception: {e}") from e
        if os.path.exists(path + ".lock"):
            os.remove(path + ".lock")
    print(f"Judged {n} completions")


if __name__ == "__main__":
    run_judge()
