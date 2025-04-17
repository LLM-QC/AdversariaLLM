import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # determinism
import copy
import json
import logging
import sys

import filelock
import hydra
import torch
from omegaconf import DictConfig, ListConfig
from tqdm import tqdm

from src.errors import print_exceptions
from src.judges import Judge

torch.use_deterministic_algorithms(True, warn_only=True)  # determinism
torch.backends.cuda.matmul.allow_tf32 = True


def collect_run_paths(save_dir, suffixes):
    paths = []
    save_dir = "/ceph/ssd/staff/beyer/llm-quick-check/outputs/2025-04-17"
    for root, _, files in os.walk(save_dir):
        for file in files:
            if not file.endswith("run.json"):
                continue
            if isinstance(suffixes, ListConfig):
                if not any(root.endswith(s) for s in suffixes):
                    continue
            elif not root.endswith(str(suffixes)):
                continue
            paths.append(os.path.join(root, file))
    paths.sort(reverse=True)
    return paths


@hydra.main(config_path="./conf", config_name="judge", version_base="1.3")
@print_exceptions
def run_judge(cfg: DictConfig) -> None:
    logging.info("-------------------")
    logging.info("Commencing judge run")
    logging.info("-------------------")
    print(cfg)
    # Run model loading and path collection in parallel
    paths = collect_run_paths(cfg.save_dir, cfg.suffixes)
    judge = Judge.from_name(cfg.classifier)()
    n = 0
    for path in tqdm(paths, file=sys.stdout):
        with filelock.FileLock(path + ".lock") as lock:
            try:
                run_list = json.load(open(path))
                for run in run_list:
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
                        logging.info(f"Judging {len(modified_prompts)} prompts")
                        results = judge(modified_prompts)
                        if results is None:
                            continue
                        i = 0
                        for step in subrun["steps"]:
                            n_completions = len(step["model_completions"])
                            step["scores"][cfg.classifier] = {k: v[i:i+n_completions] for k, v in results.items()}
                            i += n_completions
                            n += n_completions
                json.dump(run_list, open(path, "w"), indent=4)
            except Exception as e:
                print(path, str(e))
                raise Exception(f"Error in {path}. Original exception: {e}") from e
        if os.path.exists(path + ".lock"):
            os.remove(path + ".lock")
    print(f"Judged {n} completions")


if __name__ == "__main__":
    run_judge()
