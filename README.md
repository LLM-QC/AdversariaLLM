# LLM QuickCheck

Repo to compare continuous and discrete attacks on LLMs.


## Usage

### Step 0:
Change 5 paths in `conf/config.yaml` and `conf/datasets/datasets.yaml` to point to the correct location for your setup.

### Step 1 (Run Attacks):
Specify which attacks your like to run and launch a hydra job.

To evaluate Phi3 with gcg on all of adv_behaviors, for example:
```python3
python run_attacks.py -m ++model=microsoft/Phi-3-mini-4k-instruct ++dataset=adv_behaviors ++datasets.adv_behaviors.idx="range(0,300)" ++attack=gcg ++hydra.launcher.timeout_min=240
```

You can sweep over multiple options like this:
```python3
python run_attacks.py -m ++model=microsoft/Phi-3-mini-4k-instruct ++dataset=adv_behaviors ++datasets.adv_behaviors.idx="range(0,300)" ++attack=gcg,pair,autodan ++hydra.launcher.timeout_min=240
```
will launch 900 jobs and run GCG, PAIR and AutoDAN against Phi on all 300 prompts.

By default, we judge all completions with StrongREJECT - you can change this by adapting the `classifiers` attribute of your config. 
To see which judges are supported, please have a look at `src/judges.py`
