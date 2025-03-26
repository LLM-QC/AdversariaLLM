import argparse
import logging
import os
import subprocess
import multiprocessing
from typing import List
from pathlib import Path

# Setup logger

def hf_id_to_cache_dir(repo_id: str, repo_type: str = "model") -> str:
    assert repo_type in {"model", "dataset", "space"}, f"Unknown repo_type: {repo_type}"
    prefix_map = {
        "model": "models--",
        "dataset": "datasets--",
        "space": "spaces--"
    }
    return prefix_map[repo_type] + repo_id.replace("/", "--")


def _rsync_dir(src_dst_tuple):
    src_path, dst_root = src_dst_tuple
    if not src_path.exists():
        logging.warning(f"Skipping non-existent: {src_path}")
        return
    logging.info(f"Copying {src_path.name} to fast storage...")
    try:
        subprocess.run([
            # "rsync", "-a", "--exclude", ".lock/", str(src_path), str(dst_root)
            "rsync", "-avzh", "--progress", "--stats", "--exclude", ".lock/", str(src_path), str(dst_root)
        ], check=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"rsync failed for {src_path.name}: {e}")


def setup_hf_cache_on_slurm_tmpdir(
    source_hf_home: str,
    model_names: List[str] = None,
    dataset_names: List[str] = None,
    set_env_vars: bool = True,
    num_workers: int = 4,
    tmpdir: str = None
) -> Path:
    if model_names is None and dataset_names is None:
        logging.warning("No models or datasets were copied.")
        return None
    model_names = [] if model_names is None else model_names
    dataset_names = [] if dataset_names is None else dataset_names

    if not tmpdir:
        tmpdir = os.environ.get("SLURM_TMPDIR")
        logging.warning("SLURM_TMPDIR is not set. Are you running inside a Slurm job?")

    fast_hf_home = Path(tmpdir) / "hf_cache"
    hub_cache_dir = fast_hf_home / "hub"
    hub_cache_dir.mkdir(parents=True, exist_ok=True)

    repo_names = []
    for name in model_names:
        repo_names.append(hf_id_to_cache_dir(name, repo_type="model"))
    for name in dataset_names:
        repo_names.append(hf_id_to_cache_dir(name, repo_type="dataset"))
    
    src_dst_tuples = [
        (Path(source_hf_home) / "hub" / name, hub_cache_dir)
        for name in repo_names
    ]

    logging.info(f"Starting rsync for {len(src_dst_tuples)} repos using {num_workers} workers.")
    with multiprocessing.Pool(processes=num_workers) as pool:
        pool.map(_rsync_dir, src_dst_tuples)

    if set_env_vars:
        os.environ["HF_HOME"] = str(fast_hf_home)
        os.environ["TRANSFORMERS_CACHE"] = str(fast_hf_home)
        os.environ["HF_DATASETS_CACHE"] = str(fast_hf_home)
        logging.info(f"Set HF cache env vars to: {fast_hf_home}")

    return fast_hf_home


def main():
    parser = argparse.ArgumentParser(description="Copy selected Hugging Face models or datasets into SLURM_TMPDIR cache.")
    parser.add_argument(
        "--model-names",
        type=str,
        nargs="+",
        default=None,
        help="List of Hugging Face model repo IDs (e.g., cais/HarmBench-Llama-2-13b-cls)"
    )
    parser.add_argument(
        "--dataset-names",
        type=str,
        choices=["model", "dataset"],
        nargs="+",
        default=None,
        help="List of Hugging Face dataset repo IDs (e.g., cais/HarmBench-Llama-2-13b-cls)"
    )
    parser.add_argument(
        "--source-hf-home",
        type=str,
        required=True,
        help="Source HF_HOME path (e.g., /scratch/.hf_cache)"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of parallel workers for copying"
    )
    parser.add_argument(
        "--no-set-env",
        action="store_true",
        help="If set, do NOT update HF_HOME, TRANSFORMERS_CACHE, etc."
    )

    args = parser.parse_args()

    setup_hf_cache_on_slurm_tmpdir(
        source_hf_home=args.source_hf_home,
        model_or_dataset_names=args.repo_ids,
        repo_type=args.repo_type,
        set_env_vars=not args.no_set_env,
        num_workers=args.num_workers
    )


if __name__ == "__main__":
    main()
