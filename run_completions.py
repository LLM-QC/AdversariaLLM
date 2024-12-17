import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # determinism
import logging

import hydra
import torch
import json
from omegaconf import DictConfig

from src.errors import print_exceptions
from accelerate.utils import find_executable_batch_size
from src.io_utils import load_model_and_tokenizer
from src.lm_utils import prepare_tokens, get_batched_completions


torch.use_deterministic_algorithms(True, warn_only=True)  # determinism
torch.backends.cuda.matmul.allow_tf32 = True


def batch_completions(batch_size, model, tokenizer, token_list):
    completions = []
    for i in range(0, len(token_list), batch_size):
        batch = token_list[i:i + batch_size]
        completions.extend(get_batched_completions(model, tokenizer, token_list=batch, use_cache=True))
    return completions


@hydra.main(config_path="./conf", config_name="completions", version_base="1.3")
@print_exceptions
def main(cfg: DictConfig) -> None:
    logging.info("Creating completions with cfg:")
    logging.info(cfg)

    paths = []
    for root, dirs, files in os.walk(cfg.save_dir):
        for file in files:
            if file.endswith("run.json"):
                if any(root.endswith(s) for s in cfg.suffixes):
                    paths.append(os.path.join(root, file))
    paths.sort(reverse=True)


    id = cfg.model_name
    model_params = [v for k, v in cfg.models.items() if k == id][0]
    model, tokenizer = load_model_and_tokenizer(model_params)
    for path in paths:
        runs = json.load(open(path))
        updated_runs = [r.copy() for r in runs]
        for i, run in enumerate(runs):
            if run["config"]["attack"] in ("pair", "pgd", "pgd_one_hot"):
                continue
            if run["config"]["model"] != id:
                continue
            if not all([len(c) == len(a) for c, a in zip(run["completions"], run["attacks"])]):
                print(path)
                for j, (attacks, prompt, completions) in enumerate(zip(run["attacks"], run["prompts"], run["completions"])):
                    if run["config"]["attack_params"]["placement"] == "suffix":
                        token_list = [prepare_tokens(tokenizer=tokenizer, prompt=prompt["content"], target="ZZZZZ", attack=attack) for attack in attacks]
                    else:
                        token_list = [prepare_tokens(tokenizer=tokenizer, prompt=attack, target="ZZZZZ", attack=prompt["content"]) for attack in attacks]
                    token_list = [torch.cat(t[:4]) for t in token_list]
                    new_completions = find_executable_batch_size(batch_completions, 128)(model, tokenizer, token_list)
                    print("original ------")
                    print(completions[-1])
                    print("new ------")
                    print(new_completions[-1])
                    print(new_completions)
                    updated_runs[i]["completions"][j] = new_completions
                    updated_runs[i]["completions"][j][-1] = completions[-1]
                if "successes_cais" in updated_runs[i]:
                    del updated_runs[i]["successes_cais"]
                if "p_harmful" in updated_runs[i]:
                    del updated_runs[i]["p_harmful"]

                try:
                    json.dump(updated_runs, open(path, "w"), indent=2)
                except KeyboardInterrupt as e:
                    json.dump(runs, open(path, "w"), indent=2)
                    break


if __name__ == "__main__":
    main()
