# LLM QuickCheck

Repo to compare continuous and discrete attacks on LLMs.


## Usage

### Step 0:
Change 5 paths in `conf/config.yaml` and `conf/datasets/datasets.yaml` to point to the correct location for your setup.

### Step 1 (Run Attacks):
Specify which attacks your like to run and launch a hydra job.

To evaluate Phi3 with gcg on all of adv_behaviors, for example:
```python3
python run_attacks.py -m ++model_name=microsoft/Phi-3-mini-4k-instruct ++dataset_name=adv_behaviors ++datasets.adv_behaviors.idx="range(0,300)" ++attack_name=gcg ++hydra.launcher.timeout_min=240
```

### Step 2 (Judge Attack Success):
Open `src/judge.ipynb` and run it.

### Step 3 (Inspect Results):
Open `src/viz.ipynb` and have a look.


python run_attacks.py ++model_name=google/gemma-2-2b-it ++dataset_name=adv_behaviors ++datasets.adv_behaviors.idx=0 ++attack_name=my_gcg ++hydra.launcher.timeout_min=60 ++hydra.mode=RUN