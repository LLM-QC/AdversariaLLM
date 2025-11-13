# AdversariaLLM

[![arXiv](https://img.shields.io/badge/arXiv-2511.04316-b31b1b.svg)](https://arxiv.org/abs/2511.04316)


A comprehensive toolkit for evaluating and comparing continuous and discrete adversarial attacks on LLMs.
This repository provides a unified framework for running various attack methods, generating adversarial prompts, and evaluating model safety and robustness.

## 🔧 Installation

1. Clone the repository:
```bash
git clone https://github.com/LLM-QC/AdversariaLLM
cd AdversariaLLM
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install the package in development mode:
```bash
pip install -e .
```

## ⚙️ Configuration

### Step 1: Configure Paths
Update the following configuration files with your environment-specific paths:
- `conf/paths.yaml` - Root directory


## 🚀 Quick Start

### Running Basic Attacks

To evaluate a model with a single attack method:

```bash
python run_attacks.py -m \
    model=microsoft/Phi-3-mini-4k-instruct \
    dataset=adv_behaviors \
    datasets.adv_behaviors.idx="range(0,300)" \
    attack=gcg \
    hydra.launcher.timeout_min=240
```

### Running Multiple Attacks (Sweep)

To compare multiple attack methods:

```bash
python run_attacks.py -m \
    model=microsoft/Phi-3-mini-4k-instruct \
    dataset=adv_behaviors \
    datasets.adv_behaviors.idx="range(0,300)" \
    attack=gcg,pair,autodan \
    hydra.launcher.timeout_min=240
```

This will launch 900 jobs (3 attacks × 300 prompts) and run GCG, PAIR, and AutoDAN against Phi-3 on all 300 prompts.

## 🎯 Supported Attack Methods

The framework supports various adversarial attack algorithms:

- **GCG** - Greedy Coordinate Gradient attack (with various objectives, including REINFORCE)
- **PAIR** - Prompt Automatic Iterative Refinement
- **AutoDAN** - Automatic prompt generation
- **PGD** - Projected Gradient Descent (continuous in embedding and indicator-space, with & without discretization)
- **Random Search** - Baseline random optimization
- **Human Jailbreaks** - Curated human-written prompts
- **Direct** - Direct prompt testing without optimization
- **BEAST** - Gradient-based discrete optimization
- **Best-of-N** - Jailbreaking with simple string perturbations


## 📊 Evaluation and Judging

For a complete list of supported judges, see: [JudgeZoo](https://github.com/LLM-QC/judgezoo)

### Default Judge
By default, all completions are evaluated using **StrongREJECT**. You can change this by modifying the `classifiers` attribute in your config:

```yaml
classifiers: ["strong_reject", "harmbench", "custom_judge"]
```

### Running Judges Separately
```bash
python run_judges.py \
    judge=strong_reject
```
will judge all files with strong_reject which haven not been judged yet.


## 🔧 Advanced Usage

### Custom Attack Parameters
You can override specific attack parameters:

```bash
python run_attacks.py -m \
    attack=gcg \
    attacks.gcg.num_steps=500 \
    attacks.gcg.search_width=512
```

### Distributional Evaluation

Distributional evaluation allows you to assess the behavior of attacks across multiple sampled responses rather than a single deterministic output.
This is particularly useful for measuring the robustness of safety mechanisms and understanding the distribution of model behaviors under adversarial conditions.
Inspired by [arxiv:2410.03523](https://arxiv.org/abs/2410.03523) and [arxiv:2507.04446](https://arxiv.org/abs/2507.04446).


#### Specify Generation Parameters
```yaml
generation_config:
  temperature: 0.7
  top_p: 1.0
  top_k: 0
  max_new_tokens: 256
  num_return_sequences: 50
```

#### Example: Basic Distributional Evaluation

To evaluate a model with multiple sampled responses:

```bash
python run_attacks.py -m \
    model=microsoft/Phi-3-mini-4k-instruct \
    dataset=adv_behaviors \
    datasets.adv_behaviors.idx="range(0,50)" \
    attack=gcg \
    attacks.gcg.generation_config.temperature=0.7 \
    attacks.gcg.generation_config.num_return_sequences=50 \
    attacks.gcg.generation_config.max_new_tokens=256
```

This will generate 50 diverse responses per prompt at temperature 0.7, allowing you to compute metrics like:
- Expected harmfulness: E[h(Y)]
- Success rate across samples
- Distribution of refusal vs. compliance behaviors

#### Example: Comparing Baseline vs. Distributional Attacks

Compare deterministic baseline (temperature=0.0) with distributional sampling:

```bash
# Baseline: deterministic evaluation
python run_attacks.py -m \
    model=meta-llama/Meta-Llama-3.1-8B-Instruct \
    dataset=adv_behaviors \
    attack=pair \
    attacks.pair.generation_config.temperature=0.0 \
    attacks.pair.generation_config.num_return_sequences=1

# Distributional: sample-based evaluation
python run_attacks.py -m \
    model=meta-llama/Meta-Llama-3.1-8B-Instruct \
    dataset=adv_behaviors \
    attack=pair \
    attacks.pair.generation_config.temperature=0.7 \
    attacks.pair.generation_config.num_return_sequences=50
```

## 📈 Results and Analysis

Results are saved in the configured output directory with the following structure:
```
outputs/
├── YYYY-MM-DD/HH-MM-SS/{i}/run.json
...
└── YYYY-MM-DD/HH-MM-SS/{i}/run.json
```

### Visualization & Evaluation (WIP)
Generate plots and analysis with `visualize_results.ipynb` in `evaluations/`


## Used in
[1] Beyer, Tim, et al. ["Fast Proxies for LLM Robustness Evaluation."](https://arxiv.org/abs/2502.10487) arXiv preprint arXiv:2502.10487 (2025).\
[2] Xhonneux, Sophie, et al. ["A generative approach to LLM harmfulness detection with special red flag tokens."](https://arxiv.org/abs/2502.16366) arXiv preprint arXiv:2502.16366 (2025).\
[3] Beyer, Tim, et al. ["LLM-safety Evaluations Lack Robustness."](https://arxiv.org/abs/2503.02574) arXiv preprint arXiv:2503.02574 (2025).\
[4] Beyer, Tim, et al. ["Sampling-aware adversarial attacks against large language models."](https://arxiv.org/abs/2507.04446) arXiv preprint arXiv:2507.04446 (2025).\
[5] Lüdke, David, et al. ["Diffusion LLMs are Natural Adversaries for any LLM."](https://arxiv.org/abs/2511.00203) arXiv preprint arXiv:2511.00203 (2025).

## 🤝 Contributing

Contributions welcome!

## 📁 Project Structure

```
llm-quick-check/
├── src/
│   ├── attacks/           # Attack implementations
│   │   ├── gcg.py        # GCG attack
│   │   ├── pair.py       # PAIR attack
│   │   ├── autodan.py    # AutoDAN attack
│   │   └── ...
│   ├── dataset/          # Dataset handling (modular)
│   │   ├── prompt_dataset.py      # Base dataset class
│   │   ├── adv_behaviors.py       # AdvBench behaviors
│   │   ├── jbb_behaviors.py       # JailbreakBench
│   │   ├── strong_reject.py       # StrongREJECT
│   │   ├── or_bench.py            # ORBench
│   │   ├── refusal_direction.py   # RefusalDirection
│   │   ├── xs_test.py             # XSTest
│   │   ├── alpaca.py              # Alpaca
│   │   ├── mmlu.py                # MMLU
│   │   └── ...
│   ├── io_utils/         # I/O utilities
│   ├── lm_utils/         # Language model utilities
│   └── types.py          # Type definitions
├── conf/                 # Configuration files
│   ├── config.yaml       # Main config
│   ├── attacks/          # Attack-specific configs
│   ├── datasets/         # Dataset configs
│   └── models/           # Model configs
├── run_attacks.py        # Main attack runner
├── run_judges.py         # Judge evaluation
├── run_sampling.py       # Sampling utilities
└── requirements.txt      # Dependencies
```

## 🙏 Acknowledgments

Please be sure to cite the underlying work if you build on it.

Datasets
- [Alpaca](https://github.com/tatsu-lab/stanford_alpaca)
- [JailbreakBench](https://github.com/JailbreakBench/jailbreakbench)
- [HarmBench](https://github.com/centerforaisafety/HarmBench) for reference attacks & data
- [ORBench](https://arxiv.org/abs/2405.20947)
- [RefusalDirection](https://proceedings.neurips.cc/paper_files/paper/2024/hash/f545448535dfde4f9786555403ab7c49-Abstract-Conference.html)
- [StrongREJECT](https://github.com/dsbowen/strong_reject)
- [XSTest](https://arxiv.org/abs/2308.01263)
- [MMLU](https://arxiv.org/abs/2009.03300)

Attacks
- [ActorBreaker](https://arxiv.org/abs/2410.10700)
- [AmpleGCG](https://arxiv.org/abs/2404.07921)
- [AutoDAN](https://arxiv.org/abs/2310.04451)
- [BEAST](https://arxiv.org/abs/2402.15570)
- [Best-of-N Jailbreaking](https://arxiv.org/abs/2412.03556)
- [Crescendo](https://www.usenix.org/system/files/usenixsecurity25-russinovich.pdf)
- [GCG](https://arxiv.org/abs/2307.15043)
- [GCG (REINFORCE)](https://arxiv.org/abs/2502.17254)
- [PAIR](https://arxiv.org/abs/2310.08419)
- [PGD (embedding space)](https://arxiv.org/abs/2402.09063)
- [PGD (discrete relaxation)](https://arxiv.org/abs/2402.09154)
- [Human Jailbreaks](https://github.com/centerforaisafety/HarmBench/blob/main/baselines/human_jailbreaks/jailbreaks.py)

Other
- [JudgeZoo](https://github.com/LLM-QC/judgezoo) for judge implementations


## Citation
If you use this repo in your work or found it useful, please consider citing
```
@article{beyer2025adversariallm,
  title={AdversariaLLM: A Unified and Modular Toolbox for LLM Robustness Research},
  author={Beyer, Tim and Dornbusch, Jonas and Steimle, Jakob and Ladenburger, Moritz and Schwinn, Leo and G{\"u}nnemann, Stephan},
  journal={arXiv preprint arXiv:2511.04316},
  year={2025}
}
```