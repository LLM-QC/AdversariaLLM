"""Tool to schedule runs for all attacks on all models for all prompts in the adv_behaviors dataset.

Usage:
- Run `python schedule_runs.py` to print the commands that would be run.
- Run `python schedule_runs.py --run` to schedule all runs.
- Run `python schedule_runs.py --prompt-idx 10 --run` to schedule all missing runs for the first 10 prompts.
- Run `python schedule_runs.py --model=llama --attack=pgd --run` to schedule all missing runs for the llama model and the pgd attack.

"""
import os
import argparse
import pandas as pd
from joblib import Parallel, delayed

from colorama import Fore, init

import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.datasets.dataset import AdvBehaviorsConfig, Dataset

def main(args):
    d = Dataset.from_name("adv_behaviors")(
        AdvBehaviorsConfig(
            "adv_behaviors",
            messages_path="/nfs/staff-ssd/beyer/llm-quick-check/data/behavior_datasets/harmbench_behaviors_text_all.csv",
            targets_path="/nfs/staff-ssd/beyer/llm-quick-check/data/optimizer_targets/harmbench_targets_text.json",
            seed=0,
            categories=[
                "chemical_biological",
                "illegal",
                "misinformation_disinformation",
                "harmful",
                "harassment_bullying",
                "cybercrime_intrusion",
            ],
        )
    )
    prompt_to_idx = {prompt[0]['content']: idx for idx, prompt in enumerate(d)}

    timeouts = {
        'autodan': 480,
        'gcg': 240,
        'pgd': 30,  # 2 min IRL
        'human_jailbreaks': 120,
        'pgd_one_hot': 30,
        'ample_gcg': 120,
        'direct': 10,
        'pair': 30
    }

    # read existing runs
    def process_file(path):
        try:
            data = pd.read_json(path)
            return [data.iloc[i] for i in range(len(data))]
        except ValueError as e:
            raise ValueError(f'Error in {path} with {e}')

    data_path = 'outputs'

    paths = []
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith('run.json'):
                paths.append(os.path.join(root, file))

    paths = sorted(paths, reverse=True)

    runs = Parallel(n_jobs=16, verbose=10)(delayed(process_file)(path) for path in paths)
    attack_runs = [r for run in runs for r in run]
    existing = set()
    existing_judged = set()
    n_duplicates = 0

    attack_runs_new = []
    for run in attack_runs:
        config = run['config']
        model = config['model']
        dataset = config['dataset']

        for prompt in run['prompts']:
            if prompt['content'] not in prompt_to_idx:
                continue
            key = (prompt_to_idx[prompt['content']], config['attack'], model)
            if key not in existing_judged and 'successes_cais' in run:
                existing_judged.add(key)
            if key not in existing:
                existing.add(key)
                attack_runs_new.append(run)
            else:
                n_duplicates += 1

    print(f'Found {len(attack_runs_new)} unique runs out of {len(attack_runs)} ({n_duplicates} duplicates).')
    attack_runs = attack_runs_new

    # models = sorted(list(set(key[2] for key in existing if 'vicuna' not in key[2])))
    models = [
        "cais/zephyr_7b_r2d2",
        "ContinuousAT/Llama-2-7B-CAT",
        "ContinuousAT/Phi-CAT",
        "ContinuousAT/Zephyr-CAT",
        "google/gemma-2-2b-it",
        "GraySwanAI/Llama-3-8B-Instruct-RR",
        "GraySwanAI/Mistral-7B-Instruct-RR",
        "HuggingFaceH4/zephyr-7b-beta",
        "meta-llama/Llama-2-7b-chat-hf",
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "microsoft/Phi-3-mini-4k-instruct",
        "mistralai/Mistral-7B-Instruct-v0.3",
        "qwen/Qwen2-7B-Instruct",
    ]
    attacks = sorted(list(set(key[1] for key in existing)))
    attacks = [
        "ample_gcg",
        "autodan",
        "direct",
        "gcg",
        "human_jailbreaks",
        "pair",
        "pgd",
        "pgd_one_hot",
    ]

    tgt_models = args.model
    tgt_attacks = args.attack
    def get_runs(d, tgt_models, tgt_attacks):
        to_run = []
        for idx in range(len(d)):
            for tgt_attack in attacks if tgt_attacks == 'all' else [tgt_attacks]:
                for tgt_model in models if tgt_models == 'all' else [tgt_models]:
                    if (idx, tgt_attack, tgt_model) not in existing:
                        if args.prompt_idx is None:
                            if len(to_run) >= args.n_runs:
                                return to_run
                        elif idx > args.prompt_idx:
                            return to_run
                        to_run.append((idx, tgt_attack, tgt_model))
        return to_run


    to_run = get_runs(d, tgt_models, tgt_attacks)

    max_idx = max((idx for idx, _, _ in to_run), default=args.prompt_idx or 0)

    def merge_tuples(tuples, idx=0):
        if isinstance(idx, list):
            if len(idx) == 1:
                idx = idx[0]
            else:
                return merge_tuples(merge_tuples(tuples, idx[0]), idx[1:])

        # Merge runs with equal attack and model
        merged_runs = {}
        for t in tuples:
            key = tuple(t for i, t in enumerate(t) if i != idx)
            if key not in merged_runs:
                merged_runs[key] = []
            merged_runs[key].append(t[idx])
        merged_runs = [(*key[:idx], tuple(id), *key[idx:]) for key, id in merged_runs.items()]
        return merged_runs

    # We don't merge attacks as that would schedule some runs on H100 that could also run on A100
    merged_runs = merge_tuples(to_run, [0, 2])


    # Create commands for merged runs
    commands = []
    for indices, tgt_attacks, tgt_models in merged_runs:
        if isinstance(indices, int):
            indices = [indices]
        if isinstance(tgt_attacks, str):
            tgt_attacks = [tgt_attacks]
        if isinstance(tgt_models, str):
            tgt_models = [tgt_models]

        timeout = max(timeouts[tgt_attack] for tgt_attack in tgt_attacks)
        template = "python run_attacks.py -m ++model_name={tgt_models} ++attack_name={tgt_attacks} ++datasets.adv_behaviors.idx={indices} ++dataset_name=adv_behaviors ++hydra.launcher.timeout_min={timeout}"
        if any(tgt_attack in ("pair", "ample_gcg") for tgt_attack in tgt_attacks):
            template = "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True " + template + " ++hydra.launcher.partition=gpu_h100"

        indices = ','.join(str(i) for i in indices)
        tgt_attacks = ','.join(tgt_attacks)
        tgt_models = ','.join(tgt_models)

        commands.append(template.format(tgt_models=tgt_models, indices=indices, tgt_attacks=tgt_attacks, timeout=timeout))


    print(f'Scheduling {len(to_run)} runs.')
    print(f'When completed, runs for the first {args.prompt_idx or max_idx-1} prompts are done.')


    # Initialize colorama
    init(autoreset=True)

    n_scheduled = [sum(e[0]==i for e in to_run if 'vicuna' not in e[2]) for i in range(len(d))]
    n_judged = [sum(e[0]==i for e in existing_judged if 'vicuna' not in e[2]) for i in range(len(d))]
    n_completed = [sum(e[0]==i for e in existing if 'vicuna' not in e[2])-n_judged[i] for i in range(len(d))]


    # Define bar characters and colors
    scheduled_bar_char = "█"  # Bar for scheduled runs
    completed_bar_char = "▓"  # Bar for completed runs
    max_bar_width = 1 # Maximum bar width for visual simplicity

    # Function to plot bars for scheduled and completed runs
    def ascii_bar_chart(scheduled_data, completed_data, judged_data):
        print(f"{Fore.WHITE}            {Fore.GREEN}C O M P L E T E D                             {Fore.CYAN}S C H E D U L E D                            {Fore.RED}M I S S I N G")
        n_judged_total = 0
        n_completed_total = 0
        n_scheduled_total = 0
        n_remaining_total = 0
        for idx, (scheduled_runs, completed_runs, judged_runs) in enumerate(zip(scheduled_data, completed_data, judged_data)):
            # Scale the bar widths
            completed_width = int(completed_runs * max_bar_width)
            scheduled_width = int(scheduled_runs * max_bar_width)
            judged_width = int(judged_runs * max_bar_width)
            max_runs_per_prompt = len(attacks) * len(models)
            remaining_runs = max_runs_per_prompt - scheduled_runs - completed_runs - judged_runs
            remaining_width = int(remaining_runs * max_bar_width)

            # Create colored bars
            judged_bar = Fore.GREEN + completed_bar_char * judged_width
            completed_bar = Fore.LIGHTGREEN_EX + completed_bar_char * completed_width
            scheduled_bar = Fore.CYAN + scheduled_bar_char * scheduled_width
            remaining_bar = Fore.RED + scheduled_bar_char * remaining_width

            print(f"{Fore.WHITE}Prompt {idx:>03}: {judged_bar}{completed_bar}{scheduled_bar}{remaining_bar} {Fore.GREEN}{judged_runs}/{Fore.LIGHTGREEN_EX}{completed_runs}/{Fore.CYAN}{scheduled_runs}/{Fore.RED}{remaining_runs}")
            n_judged_total += judged_runs
            n_completed_total += completed_runs
            n_scheduled_total += scheduled_runs
            n_remaining_total += remaining_runs
        print(f"     {Fore.WHITE}TOTAL: " + " " * max_runs_per_prompt + f" {Fore.GREEN}{n_judged_total}/{Fore.LIGHTGREEN_EX}{n_completed_total}/{Fore.CYAN}{n_scheduled_total}/{Fore.RED}{n_remaining_total}")
    ascii_bar_chart(n_scheduled, n_completed, n_judged)

    # Run commands in parallel
    if args.run:
        print('Running commands')
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(subprocess.run, c, shell=True) for c in commands]
            for future in as_completed(futures):
                try:
                    future.result()  # Raises an exception if the command failed
                except Exception as e:
                    if isinstance(e, KeyboardInterrupt):
                        break
                    print(e)
    else:
        for c in commands:
            print(c)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--model', type=str, default='all')
    argparser.add_argument('--attack', type=str, default='all')
    argparser.add_argument('--n-runs', type=int, default=33600, help='Number of SLURM runs to schedule')
    argparser.add_argument('--prompt-idx', type=int, default=None, help='Run all attacks for prompts up to this index')
    argparser.add_argument('--run', action='store_true', help='Run the commands')
    args = argparser.parse_args()
    main(args)
