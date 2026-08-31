# IHO attack

IHO trains a masked-diffusion model (LLaDA-8B-Base) to write jailbreak prompts. It inpaints a
prompt in front of an affirmative target (`Sure, here is how...`), the target model answers, a
judge scores the answer, and the diffusion model is DPO-fine-tuned on its own best/worst samples
over several cycles. `attack=iho` runs the whole thing and writes the usual per-behavior `run.json`.

## Setup

`pixi install`. The only added dependency is `wandb`, and it is optional — if wandb isn't
configured, logging falls back to a local `metrics.jsonl` and everything still runs
(`ADVERSARIALLM_WANDB=off` forces this).

## The four ways to run it

IHO's optimization *is* the DPO training, so `attack=iho` **trains by default** and then evaluates,
just like every other attack runs its own search.

**Train + evaluate (adaptive — the default).** Trains the attacker against the target, then samples
the winning adapter fresh and logs those attacks to `run.json`:

```bash
pixi run python run_attacks.py attack=iho \
  model=meta-llama/Meta-Llama-3-8B-Instruct dataset=jbb_behaviors \
  datasets.jbb_behaviors.shuffle=false \
  'attacks.iho.training.train_idx=[0,1,2,3,4]' \
  'attacks.iho.training.eval_idx=[0,1,2,3,4]' \
  attacks.iho.training.n_cycles=8 attacks.iho.num_samples_per_behavior=1024
```

**Transfer with a trained adapter (no training).** A checkpoint implies transfer, so training is
skipped:

```bash
pixi run python run_attacks.py attack=iho model=... dataset=jbb_behaviors \
  'datasets.jbb_behaviors.idx=[0,1,2]' \
  attacks.iho.attacker.lora_checkpoint=<attacker_dir>/<attacker_id>/best
```

**Transfer with the untrained base attacker (baseline).** `attacks.iho.train=false` with no
checkpoint.

**Training only (produce an adapter, no fresh eval).** Either `attacks.iho.eval_after_train=false`,
or the standalone CLI, which calls the same engine and writes just the adapter:

```bash
pixi run python train_attacker.py model=... dataset=jbb_behaviors \
  datasets.jbb_behaviors.shuffle=false \
  training.n_cycles=8 'training.train_idx=[0,1,2,3,4]' \
  training.train_judge=strong_reject
```

## Configuring training

Every training knob lives under `attacks.iho.training.*` (the same block `train_attacker.py` uses as
`training.*`) and is overridable directly on the CLI — no overlay, no `+`:

- `n_cycles`, `dpo.epochs`, `dpo.patience` — training length
- `dpo.learning_rate`, `dpo.beta` — DPO objective
- `pairing.min_score`, `pairing.percent_chosen`, `pairing.expanding` — preference construction
- `train_judge` (drives pairing + early stopping), `val_judges` (logged only)
- `attacker.*` — diffusion knobs (`prompt_len`, `num_denoise_steps`, `temperature`, LoRA `r`/`alpha`)

## Where results go

- **Per-behavior `run.json`**, one file per behavior, under the usual `save_dir/<date>/<time>/<idx>/`
  — same location and shape as every other attack.
- **Trained attacker artifacts** under `attacker_dir` (set in `conf/paths.yaml`, default
  `${root_dir}/attackers/`; on a cluster point it at large shared storage, not an NFS home):

  ```
  <attacker_dir>/<attacker_id>/
    checkpoints/best_cycle_<i>/            # LoRA adapters
    best -> checkpoints/best_cycle_<k>     # winning cycle
    samples/cycle_<i>.parquet              # per-cycle attack attempts (pre-DPO)
    dpo_samples/cycle_<i>_epoch_<e>.parquet
    metrics.jsonl   flops.json   train_config.yaml
  ```

  `attacker_id=auto` is a content hash of the training config, so rerunning an identical config
  resumes the same directory.

## How the final attack data is logged

- The **fresh post-training evaluation** is logged inline as normal `run.json` steps (prompt,
  completion, judge score, FLOPs). Read it like any other attack; score post-hoc with the usual
  `run_judges.py` / `classifiers=[...]` path.
- The **training attempts are not thrown away.** Every cycle's samples are real attacks against the
  target, persisted as parquet under `attacker_dir` and *linked* from each `run.json` via
  `resume_metadata.iho_training.training_samples` (a compact judge→column + per-phase parquet map —
  far too many rows to inline). Read them back with:

  ```python
  from adversariallm.io_utils.iho_training_samples import load_training_score_sequences
  seqs = load_training_score_sequences(run_json, judge="strong_reject", phases=["dpo_eval"])
  # -> {behavior_id: [scores]}
  ```

  So the clean "paper" number uses the inline eval steps; a compute-honest number can additionally
  union the linked training attempts.

## Gotchas worth knowing

- **Behavior indexing.** Datasets default to `shuffle=true`, so `idx` / `train_idx` / `eval_idx` are
  positions in a seed-0 permutation, *not* raw dataset rows. Pin `datasets.jbb_behaviors.shuffle=false`
  for reproducible or comparable splits; `train_attacker.py` logs the resolved original rows at startup.
- **LLaDA + transformers 5.** Handled in code (three compat shims in `attackers/diffusion.py`). Always
  load LLaDA via `LLaDADiffusionAttacker`, never a bare `AutoModel.from_pretrained`.
- **`strong_reject` judge offline.** It's a LoRA repo with no `config.json`, so it can't resolve under
  `HF_HUB_OFFLINE=1`. Run it online, or use `harmbench` for offline jobs.
- **Memory.** The 8B attacker, the target, and the judge ping-pong on/off GPU each cycle. Budget one
  H100/H200 and roughly an hour for a small multi-cycle run.

## Validate the install

```bash
pixi run python -c "from adversariallm.attacks import Attack; print(Attack.from_name('iho').__name__)"
pixi run python -m pytest tests/test_training/ tests/test_attackers/ tests/test_attacks/test_iho.py -q
```
