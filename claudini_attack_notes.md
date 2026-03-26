# Claudini Attack Integration Notes (v63 + v82 + oss_v53)

Date: 2026-03-26
Repos compared:
- Local: `/nfs/homedirs/dornb/gits/llm-quick-check`
- Upstream reference: `/tmp/claudini` (cloned from `https://github.com/romovpa/claudini`)

## Goal
Implement selected Claudini attack variants in this repo while preserving this repo's reproducible execution stack (model loading, tokenizer handling, generation, logging/results shape).

## Implemented versions in local repo (single-file attack)
- `claude_v63`
  - ADC + LSGM decoupled loss implementation.
- `claude_v82`
  - Same ADC/LSGM loop as `v63` with v82 defaults when using baseline config values:
  - `lr=12.0`, `momentum=0.99`, `ema_alpha=0.01`, `num_starts=8`, `lsgm_gamma=0.70`.
- `claude_oss_v53` (alias: `claude_v53-oss`)
  - DPTO candidate sampling + MAC momentum gradient + coarse-to-fine replacement (`n_replace=2 -> 1` at `switch_fraction`).
  - Uses local reproducibility stack (conversation prep, generation, run logging).
  - Default knobs wired in config: `num_candidates=80`, `topk_per_position=300`, `dpto_temperature=0.4`, `n_replace=2`, `switch_fraction=0.8`.

## What "v63" means in Claudini
There are multiple `v63` names in Claudini:
- `claudini/methods/claude_random/v63/optimizer.py` (`method_name = "claude_v63"`)
- `claudini/methods/claude_safeguard/v63/optimizer.py` (`method_name = "claude_oss_v63"`)
- `claudini/methods/claude_unrolled/claude_v63/optimizer.py` (`method_name = "claude_v63_unrolled"`)

For random-target track (`configs/random_valid.yaml`), the active method is `claude_v63` from `claude_random/v63`.

Important: `claude_random/v63` is only a thin hyperparameter override on top of an inheritance chain:
- `v63 -> v26 -> v19 -> original/adc -> TokenOptimizer`

So, copying only `v63/optimizer.py` is insufficient. For a single-file implementation in this repo, use the *unrolled* v63 logic as the main reference (already flattened and readable).

## Core algorithmic pieces to preserve from Claudini v63
1. ADC-style soft optimization over token distributions (`z` in `[K, L, V]`).
2. Decoupled K/lr behavior (sum over restarts, not mean).
3. LSGM backward hooks on LayerNorm modules (`grad_input *= gamma`).
4. Adaptive sparsification schedule using EMA of per-restart wrong target-token counts.
5. Discrete evaluation of argmax suffix candidates and running global-best suffix tracking.

## Local repo intersection points

### Keep from local repo (for reproducibility + comparability)
- Model/tokenizer loading: `adversariallm/io_utils/load_model_and_tokenizer` path through `run_attacks.py`.
- Prompt/tokenization handling for conversations: `prepare_conversation` in `adversariallm/lm_utils/tokenization.py`.
- Generation path for final completions: `generate_ragged_batched` + local generation config (`AttackStepResult.model_completions`).
- Result schema and run logging: `AttackStepResult`, `SingleAttackRunResult`, `AttackResult` in `adversariallm/attacks/attack.py`.
- Attack registration and config flow: `Attack.from_name` + `conf/attacks/attacks.yaml`.

### Borrow/adapt from Claudini
- v63 optimization loop mechanics and state transitions (prefer `claude_unrolled/claude_v63/optimizer.py` for porting).
- Norm-module hook matching patterns and cleanup semantics.
- Batched `K` restart compute layout.

### Required adaptation (cannot be copied 1:1)
- Claudini uses `TokenOptimizer` base abstractions not present here (e.g., `_prepare_prompt`, `compute_discrete_loss_batch`, custom flop counter). These must be reimplemented in local attack style.
- Local repo is conversation-centric and supports multi-turn datasets; Claudini benchmark is single prompt/target layout. Need a deterministic policy for which turns are optimized and which target token(s) define loss.
- Local repo has explicit disallowed-token filtering and model-specific tokenization edge handling; retain local behavior.

## Proposed implementation shape (single-file + versioned modes)
Create one new file:
- `adversariallm/attacks/claudini.py`

Inside file:
1. `ClaudiniConfig` dataclass with:
   - `name: claudini`
   - `version` selector (e.g., `"random_v63"`, future: `"random_v82"`)
   - shared knobs (`num_steps`, `num_starts`, `lr`, `momentum`, `ema_alpha`, `lsgm_gamma`, `optim_str_init`, token filters, generation config, etc.)
2. Version dispatch table in-file:
   - `VERSION_IMPLS = {"random_v63": _run_random_v63, ...}`
3. `ClaudiniAttack(Attack)`:
   - run() uses local conversation preparation and local result objects.
   - delegates optimization loop to version-specific in-file function.
4. Version implementation function(s):
   - Start with faithful `random_v63`.
   - Keep code isolated and clearly documented where behavior is exact-vs-adapted.

Rationale: one file stays close to your preference while making future Claudini versions selectable via config only.

## Fidelity strategy
Target: "as close as possible" without bypassing local reproducibility scaffolding.

- Exact/fidelity-critical:
  - ADC + decoupled loss + adaptive sparsity + LSGM update rules.
  - Random seeding behavior controlled by local attack seed.
- Local consistency-first:
  - Prompt assembly/token boundaries via local `prepare_conversation`.
  - Completion generation and logging via local generation pipeline.
  - Disallowed token filtering consistent with other local attacks.

## Validation plan (scoped first)
1. Static integration checks:
   - attack registry includes `claudini`.
   - config resolves from `conf/attacks/attacks.yaml`.
2. Minimal runtime smoke test (single sample, small steps, tiny K):
   - verify no hook leak, no shape mismatch, deterministic rerun for fixed seed.
3. Behavior sanity:
   - objective decreases/improves over steps for at least some prompts.
   - outputs and metadata logged in standard local format.
4. Comparative sanity:
   - compare a short run against Claudini reference on same model/prompt to ensure trajectory is qualitatively aligned (not necessarily identical due to different wrappers/templates).

## Risks / gotchas to remember
- LSGM hooks must always be removed even on exceptions.
- Chat template and tokenization differences can dominate outcomes; use local tokenization path intentionally.
- Exact equivalence with Claudini numbers is unlikely unless all formatting/model/runtime assumptions match their benchmark stack.
- `v63` name ambiguity must be explicit in config and docs.

## Phase plan for implementation
1. Land skeleton attack + config + registry wiring (`claudini`, `version=random_v63`) with no silent default changes to existing attacks.
2. Port `random_v63` optimizer loop faithfully in that file, adapting only base/plumbing calls.
3. Add targeted tests for registry/config + one deterministic smoke test.
4. Run small Pixi-based validation command and record expected effect and limitations.
5. (Optional extension) add second version (e.g., `random_v82`) in same file to prove version dispatch pattern.

## Command notes used for research
- Local repo exploration: `rg`, `sed`, `ls` under `adversariallm/attacks`, `conf/attacks`, `run_attacks.py`, `adversariallm/lm_utils`.
- External reference cloned to: `/tmp/claudini` via `git clone --depth 1 https://github.com/romovpa/claudini /tmp/claudini`.
