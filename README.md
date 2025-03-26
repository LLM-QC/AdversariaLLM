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


# Notes

For DRAC, run attacks:

```bash
python run_attacks.py --config-name=config_rf_llama32 --multirun root_dir=$PWD save_dir=$SCRATCH/llm-quick-check-outputs/ hydra/launcher=drac_gpu hydra.launcher.cpus_per_task=1  hf_offline_mode=true attack_name=gcg ++attacks.gcg.num_steps=300 datasets.rf_test.idx="range(0,160,step=10)" ++datasets.rf_test.batch=10
python run_attacks.py --config-name=config_rf_phi35 --multirun root_dir=$PWD save_dir=$SCRATCH/llm-quick-check-outputs/ hydra/launcher=drac_gpu hydra.launcher.cpus_per_task=1  hf_offline_mode=true attack_name=gcg ++attacks.gcg.num_steps=300 datasets.rf_test.idx="range(0,160,step=10)" ++datasets.rf_test.batch=10
python run_attacks.py --config-name=config_rf_mistral --multirun root_dir=$PWD save_dir=$SCRATCH/llm-quick-check-outputs/ hydra/launcher=drac_gpu hydra.launcher.cpus_per_task=1  hf_offline_mode=true attack_name=gcg ++attacks.gcg.num_steps=300 datasets.rf_test.idx="range(0,160,step=10)" ++datasets.rf_test.batch=10
```

and for PAIR:

```bash
python run_attacks.py --config-name=config_rf_llama32 --multirun root_dir=$PWD save_dir=$SCRATCH/llm-quick-check-outputs/ hydra/launcher=drac_gpu hydra.launcher.cpus_per_task=2  hf_offline_mode=true ++datasets.rf_test.path=${HOME}/harmful-harmless-eval/ ++datasets.rf_test.batch=20 attack_name=pair
python run_attacks.py --config-name=config_rf_phi35 --multirun root_dir=$PWD save_dir=$SCRATCH/llm-quick-check-outputs/ hydra/launcher=drac_gpu hydra.launcher.cpus_per_task=2  hf_offline_mode=true ++datasets.rf_test.path=${HOME}/harmful-harmless-eval/ ++datasets.rf_test.batch=20 attack_name=pair
python run_attacks.py --config-name=config_rf_mistral --multirun root_dir=$PWD save_dir=$SCRATCH/llm-quick-check-outputs/ hydra/launcher=drac_gpu hydra.launcher.cpus_per_task=2  hf_offline_mode=true ++datasets.rf_test.path=${HOME}/harmful-harmless-eval/ ++datasets.rf_test.batch=20 attack_name=pair
```

for some unknown reason trying to do it directly in the config like this:

```yaml
attacks:
  gcg:
    num_steps: 300
datasets:
  rf_test:
    batch: 20
    path: ${oc.env:HOME}/harmful-harmless-eval/
```

breaks the output and dumps the pkls in base `multirun/.../.submitit/` directory, rather than in the indivudal run. 
We need to do `++datasets.rf_test.batch=...`



### Installation

```bash
module load python/3.10 cuda/12.6.0
virtualenv venv

pip install uv
uv pip install -r requirements
uv pip install vllm peft bitsandbytes
uv pip install -e ../jailbreakbench # make sure you have a local jailbreakbench installed with the litellm fix
uv pip install flash-attn --no-build-isolation
```
