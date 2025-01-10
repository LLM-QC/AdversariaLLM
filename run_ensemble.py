import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

args = argparse.ArgumentParser()
args.add_argument("--model", type=str, default=None)
args.add_argument("--min-idx", type=int, default=0)
args.add_argument("--max-idx", type=int, default=100)
args.add_argument("--pgd", action="store_true")
args.add_argument("--skip_a100", action="store_true")
args.add_argument("--skip_h100", action="store_true")
args = args.parse_args()

model_name = args.model
attacks_a100 = [
    "gcg",
    "autodan",
    "human_jailbreaks",
    "beast",
    "direct",
]
attacks_h100 = [
    "ample_gcg",
    "pair"
]

# Base command template
template_a100 = (
    "python run_attacks.py -m ++model_name={model_name} ++attack_name={attack_name} "
    "++datasets.adv_behaviors.idx={indices} ++dataset_name=adv_behaviors "
    "++hydra.launcher.timeout_min=300 "
)

template_h100 = (
    "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True " + template_a100 +
    "++hydra.launcher.partition=gpu_h100"
)
# Base command template
template_pgd = (
    "python run_attacks.py ++model_name={model_name} ++attack_name=pgd "
    '++datasets.adv_behaviors.idx="{indices}" ++dataset_name=adv_behaviors '
    "++hydra.launcher.timeout_min=300 "
    "++attacks.pgd.epsilon=1 ++attacks.pgd.normalize_alpha=true "
    "-m "
)

indices = f"\"range({args.min_idx},{args.max_idx})\""

commands = []
if not args.skip_a100:
    commands.append(
        template_a100.format(
            model_name=model_name,
            attack_name=",".join(attacks_a100),
            indices=indices,
        )
    )
if not args.skip_h100:
    commands.append(
        template_h100.format(
            model_name=model_name,
            attack_name=",".join(attacks_h100),
            indices=indices,
        )
    )
if args.pgd:
    commands.append(
        template_pgd.format(
            model_name=model_name,
            indices=list(range(args.min_idx, args.max_idx)),
        )
    )

print("Commands")
print("\n".join(commands))
with ThreadPoolExecutor() as executor:
    futures = [executor.submit(subprocess.run, c, shell=True) for c in commands]
    for future in as_completed(futures):
        try:
            future.result()  # Raises an exception if the command failed
        except Exception as e:
            if isinstance(e, KeyboardInterrupt):
                break
            print(e)