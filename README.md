# LLM QuickCheck

A comprehensive toolkit for evaluating and comparing continuous and discrete adversarial attacks on LLMs.
This repository provides a unified framework for running various attack methods, generating adversarial prompts, and evaluating model safety and robustness.

## 🔧 Installation

1. Clone the repository:
```bash
git clone https://github.com/LLM-QC/llm-quick-check
cd llm-quick-check
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

### Default Judge
By default, all completions are evaluated using **StrongREJECT**. You can change this by modifying the `classifiers` attribute in your config:

```yaml
classifiers: ["strong_reject", "harmbench", "custom_judge"]
```

### Supported Judges
For a complete list of supported judges, see: [JudgeZoo](https://github.com/LLM-QC/judgezoo)

### Running Judges Separately
```bash
python run_judges.py \
    judge=strong_reject
```
will judge all files with strong_reject which havent been judged yet.

## 📁 Project Structure

```
llm-quick-check/
├── src/
│   ├── attacks/           # Attack implementations
│   │   ├── gcg.py        # GCG attack
│   │   ├── pair.py       # PAIR attack
│   │   ├── autodan.py    # AutoDAN attack
│   │   └── ...
│   ├── dataset.py        # Dataset handling
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

## 🔧 Advanced Usage

### Custom Attack Parameters
You can override specific attack parameters:

```bash
python run_attacks.py -m \
    attack=gcg \
    attacks.gcg.num_steps=500 \
    attacks.gcg.search_width=512
```

### Custom Generation Parameters
Modify text generation settings - this is useful e.g. for results in https://arxiv.org/abs/2507.04446:

```yaml
generation_config:
  temperature: 0.0
  top_p: 1.0
  top_k: 0
  max_new_tokens: 256
  num_return_sequences: 1
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

## 🤝 Contributing

Contributions welcome!

## 🙏 Acknowledgments

- [JudgeZoo](https://github.com/LLM-QC/judgezoo) for judge implementations
- [JailbreakBench](https://github.com/JailbreakBench/jailbreakbench) for baseline datasets
- [HarmBench](https://github.com/centerforaisafety/HarmBench) for reference attacks &  data
- [Hydra](https://hydra.cc/) for configuration management
