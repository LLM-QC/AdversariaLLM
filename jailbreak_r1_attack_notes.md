# Jailbreak-R1 Integration Notes

## Objective
Integrate Jailbreak-R1 (arXiv:2506.00782) as a benchmark attack in this repo while matching existing architecture/patterns:
- one file per attack
- use repo model loading (`load_model_and_tokenizer`)
- use repo generation path (`generate_ragged_batched`)
- preserve reproducibility and explicit configuration

## Sources inspected
- Paper/code references provided by user:
  - https://arxiv.org/pdf/2506.00782
  - local clone: `/nfs/homedirs/dornb/gits/Jailbreak-R1`
- Key upstream files examined:
  - `README.md`
  - `jailbreak_generate.py`
  - `src/utils/build_datasets.py`
  - `src/algorithm/grpo_warm_jailbreak.py`

## Upstream inference behavior to replicate
From upstream `jailbreak_generate.py` and dataset builders:
- Attack model prompt template:
  - user message asks model to produce:
    - `<think> ... </think><attack> ... </attack>`
  - includes `Attack target: {goal}`
- Generation is seeded with assistant prefix condition:
  - `"<think> Sure, let's think step by step, "`
- Sampling settings for attack generation:
  - temperature=1.0
  - top_p=0.95
  - top_k=0
  - do_sample=True
  - max_new_tokens=512
- Parse with regex for `<think>` and `<attack>` blocks.
- Retry loop when parsing fails.
- Minor cleanup heuristics in upstream:
  - strip leading `"\nAttack Prompt:"` and `"Question"` variants from parsed attack text.

## Integration design (repo-aligned)
- New attack file: `adversariallm/attacks/jailbreak_r1.py`.
- New config entry in `conf/attacks/attacks.yaml`.
- Attack registry wiring in `adversariallm/attacks/attack.py`.
- Use nested `attack_model` config and load it via `load_model_and_tokenizer`.
  - Reuse target model/tokenizer if same model id.
- Use `generate_ragged_batched` for:
  - attack prompt generation from attack model
  - target completion generation on generated attack prompts
- Represent each generated attack as one attack step (`AttackStepResult`).

## Open implementation choices
- Whether to expose both "strict upstream-like" and "repo-default" generation controls.
  - Current plan: explicit `attack_model_generation_config` and separate target `generation_config`.
- Whether to compute target loss per step.
  - Current plan: keep minimal and omit loss unless needed for parity with existing reporting.

## Validation plan
1. Unit/smoke test with monkeypatched generation for deterministic parser + step assembly checks.
2. Config wiring test via `Attack.from_name("jailbreak_r1")`.
3. Real run on GPU with small hparams:
   - small target model
   - small dataset slice
   - reduced number of generated prompts / retries
4. Iterate until end-to-end run is stable.

## Work log
- 2026-04-10: Created feature branch `feature/jailbreak-r1-benchmark` from `jonas` in worktree `.worktrees/jailbreak-r1`.
- 2026-04-10: Mapped repo attack architecture + upstream Jailbreak-R1 inference behavior.
