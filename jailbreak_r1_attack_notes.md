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

## Runtime findings (2026-04-10)
- Attempted strict default run with:
  - target: `google/gemma-3-1b-it`
  - attack model: `yukiyounai/Jailbreak-R1`
  - Result: failed with HF gated-model 403 (`GatedRepoError`).
- Verified end-to-end GPU execution by overriding attack model to accessible checkpoints.
  - `attack_model.id=google/gemma-3-1b-it` (same as target): run completed and wrote `run.json`.
  - `attack_model.id=qwen/Qwen2-7B-Instruct`: run completed and wrote `run.json`.
- Practical caveat:
  - With substitute (non-R1) attack models, strict upstream parser often fails to find `<think>/<attack>` tags, causing fallback prompt `"a"`.

## Practical compatibility tweak
- Added optional config flag:
  - `allow_untagged_fallback` (default: `false`)
- Rationale:
  - Keep strict upstream behavior by default for faithful benchmarking.
  - Allow local smoke runs with substitute attack models to still generate non-trivial prompts when strict tags are missing.

## Command log (executed)
- Unit/smoke tests:
  - `pixi run pytest -q tests/test_attacks/test_jailbreak_r1.py tests/test_attacks/test_attacks.py`
- Real runs:
  - Strict default (expected to match paper model id):
    - `pixi run python run_attacks.py model=google/gemma-3-1b-it attack=jailbreak_r1 dataset=adv_behaviors datasets.adv_behaviors.idx=0 datasets.adv_behaviors.shuffle=false attacks.jailbreak_r1.num_steps=2 attacks.jailbreak_r1.parse_retries=2 generation_config.max_new_tokens=64 attacks.jailbreak_r1.attack_model_generation_config.max_new_tokens=128 classifiers=null`
  - Substitute attack model = target model:
    - `pixi run python run_attacks.py model=google/gemma-3-1b-it attack=jailbreak_r1 dataset=adv_behaviors datasets.adv_behaviors.idx=0 datasets.adv_behaviors.shuffle=false attacks.jailbreak_r1.num_steps=2 attacks.jailbreak_r1.parse_retries=2 generation_config.max_new_tokens=64 attacks.jailbreak_r1.attack_model.id=google/gemma-3-1b-it attacks.jailbreak_r1.attack_model.tokenizer_id=google/gemma-3-1b-it attacks.jailbreak_r1.attack_model.chat_template=null attacks.jailbreak_r1.attack_model_generation_config.max_new_tokens=128 classifiers=null`
  - Substitute attack model = Qwen2:
    - `pixi run python run_attacks.py model=google/gemma-3-1b-it attack=jailbreak_r1 dataset=adv_behaviors datasets.adv_behaviors.idx=0 datasets.adv_behaviors.shuffle=false attacks.jailbreak_r1.num_steps=1 attacks.jailbreak_r1.parse_retries=2 generation_config.max_new_tokens=64 attacks.jailbreak_r1.attack_model.id=qwen/Qwen2-7B-Instruct attacks.jailbreak_r1.attack_model.tokenizer_id=qwen/Qwen2-7B-Instruct attacks.jailbreak_r1.attack_model.chat_template=null attacks.jailbreak_r1.attack_model.short_name=Qwen2 attacks.jailbreak_r1.attack_model.developer_name=Alibaba attacks.jailbreak_r1.attack_model_generation_config.max_new_tokens=128 classifiers=null`
- Final practical smoke run with `allow_untagged_fallback=true` and Qwen2 attack model succeeded and produced a non-trivial generated attack prompt (`attack_prompt_length=289`) while still recording `parse_success=0.0` (strict tag parser miss).

## Scalability update (2026-04-10)
- Refactored `jailbreak_r1` to parallelize per-behavior attempts across `num_steps`:
  - attack generation now runs in batched retry rounds over pending steps
  - target completion generation now runs in a single batched call over all steps
- This preserves method behavior (independent sampled attempts) while improving throughput.

## Prompt cache feature (2026-04-10)
- Added cache support to avoid attacker-model inference at scale:
  - `prompt_cache_mode`: `off|read|write|read_write`
  - `prompt_cache_path`: JSON file path
  - `prompt_cache_num_steps`: prompts per behavior to store (e.g., 1024)
  - `prompt_cache_subset_strategy`: `seeded_random|first_n` for selecting `num_steps` at runtime
  - `prompt_cache_strict_match`: enforce fingerprint+dataset signature equality on read
- In `read` mode, attacker model is not loaded.
- Verified smoke path:
  - write cache with `prompt_cache_num_steps=4`
  - read cache with `num_steps=2` and deterministic subsetting
