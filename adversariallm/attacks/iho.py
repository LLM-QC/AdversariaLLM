"""IHO (Inpainting + Human-preference Optimization) attack.

This module holds two things:

* :class:`IHOSampler` -- the single source of truth for "what is a sample". It owns a diffusion
  attacker, turns behaviors into adversarial prompts by inpainting an affirmative target, and
  scores those prompts against a :class:`~adversariallm.defenses.TargetSystem`. The phase-3
  trainer (`train_attacker.py`) calls this same class, so it is deliberately independent of the
  :class:`Attack` lifecycle.
* :class:`IHOAttack` -- the transfer-mode attack. It runs the untrained (or LoRA-loaded) attacker
  over a dataset, one :class:`SingleAttackRunResult` per behavior and one
  :class:`AttackStepResult` per sample, and leaves judging to `run_judges.py` (footgun below).

FLOPs are attributed per step as ``attacker_denoise + target_generate + target_loss`` for that
sample (plan §5.3); training FLOPs stay in a separate phase in ``flops.json`` and are never folded
into per-step numbers.

``IHOSampler`` exposes two ways to spend a sample budget over a list of behaviors:

* :meth:`IHOSampler.generate` -- ``n_per_behavior`` samples for *every* behavior (used by the
  transfer-mode :class:`IHOAttack`, where "how many attacks per behavior" is the actual knob).
* :meth:`IHOSampler.generate_pooled` -- a single TOTAL budget spread across the behaviors,
  reproducing AAPL's ``num_sampled_attacks`` semantics (used by ``train_attacker.py``'s
  ``training.n_samples.{train,eval,val}``). AAPL draws behavior indices in ``batch_size``-sized
  chunks -- with replacement when the split has fewer behaviors than ``batch_size`` (the common
  case for a handful of training behaviors), else as repeated full-dataset shuffles -- until the
  running count reaches the budget (``aapl/utils/general_utils.py:108-133`` build_attack_dataloader
  + ``aapl/AAPLPipeline.py:386-421`` sample()). See :meth:`IHOSampler._draw_pooled_counts`.

@article{lüdke2025diffusionllmsnaturaladversaries,
      title={Diffusion LLMs are Natural Adversaries for any LLM},
      author={David Lüdke and Tom Wollschläger and Paul Ungermann and Stephan Günnemann and Leo Schwinn},
      year={2025},
}
"""

import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pandas as pd
import torch
from beartype import BeartypeConf, beartype
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import RandomSampler

from ..attackers import DiffusionAttackerConfig, build_attacker
from ..dataset import PromptDataset
from ..defenses import TargetSystem
from ..io_utils import free_vram
from ..io_utils.iho_training_samples import build_training_samples_manifest
from ..lm_utils import prepare_conversation
from ..training.flops import FlopsLedger, pass_flops
from ..types import Conversation
from .attack import Attack, AttackResult, AttackStepResult, GenerationConfig, SingleAttackRunResult

if TYPE_CHECKING:  # judgezoo is heavy; only needed for type hints here
    from judgezoo import Judge

logger = logging.getLogger(__name__)

#: beartype with the PEP 484 numeric tower enabled, so an ``int`` is accepted where a ``float`` is
#: annotated (a yaml ``0`` for a ``0.0`` default must not raise). Matches ``training/flops.py``.
_typed = beartype(conf=BeartypeConf(is_pep484_tower=True))

#: Exact column set produced by :meth:`IHOSampler.to_dataframe` (plan §4.6). ``jb_index`` and
#: ``judge_score_training`` are load-bearing for
#: ``evaluate/expected_asr_multiphase.py::build_iho_phase_table``; the rest keep AAPL's analysis
#: notebooks portable. The ``flops_*`` glob in the plan is expanded to the concrete components.
IHO_DATAFRAME_COLUMNS: tuple[str, ...] = (
    "jb_index",
    "goal_text",
    "original_prompt_text",
    "attacking_prompt_text",
    "attacking_prompt_ids",
    "inpainted_prompt_text_full",
    "attack_loglikelihood",
    "attacked_output",
    "attacked_output_raw",
    "judge_score_training",
    "judge_score_validation",
    "cycle_id",
    "phase_id",
    "flops_attacker_denoise",
    "flops_target_generate",
    "flops_target_loss",
    "flops_total",
)

_CYCLE_RE = re.compile(r"cycle(\d+)")


@dataclass
class IHOSample:
    """One adversarial prompt and everything scored about it.

    Fields above ``completion`` are set by :meth:`IHOSampler.generate`; ``completion`` onward are
    filled by :meth:`IHOSampler.score`. ``flops`` accumulates per-sample FLOPs by component
    (``attacker_denoise`` from generation, ``target_generate`` / ``target_loss`` from scoring).
    """

    behavior_idx: int  # index into the dataset passed in
    goal: str
    target: str
    prompt_text: str  # the adversarial prompt (decoded prompt span)
    prompt_token_ids: list[int]
    full_text: str  # decoded full canvas -- this is what DPO trains on
    attacker_loglikelihood: float
    phase: str

    completion: str | None = None
    completion_raw: str | None = None
    target_loss: float | None = None
    judge_scores: dict[str, float] = field(default_factory=dict)  # role -> p_harmful
    defense_metadata: list[dict[str, Any]] | None = None

    # Per-sample FLOPs attribution and the target's tokenized generation input, kept so the attack
    # can build an AttackStepResult without re-tokenizing (not part of the plan's core schema).
    flops: dict[str, int] = field(default_factory=dict)
    input_token_ids: list[int] | None = None

    def total_flops(self) -> int:
        return int(sum(self.flops.values()))


class IHOSampler:
    """Owns a diffusion attacker; generates and scores samples against a ``TargetSystem``.

    ``attacker`` is a test-injection point (mirroring ``LLaDADiffusionAttacker``'s ``model`` /
    ``tokenizer`` kwargs): when passed, :meth:`load` uses it as-is and :meth:`unload` never frees
    it, since no small masked-diffusion LM exists to load for real in a unit test. Production code
    passes ``attacker=None`` and lets :meth:`load` build one from ``attacker_cfg``.
    """

    def __init__(
        self,
        attacker_cfg: Any,
        *,
        ledger: FlopsLedger,
        seed: int,
        attacker: Any = None,
    ) -> None:
        self.attacker_cfg = attacker_cfg
        self.ledger = ledger
        self.seed = seed
        self.attacker = attacker
        self._owns_attacker = attacker is None
        # Set by the caller before generate() so the attacker can take the GPU while the target
        # model is parked on CPU (the ample_gcg pattern). None means "no target to move".
        self.target: TargetSystem | None = None
        # Generation settings for score(); the attack/trainer overwrites this before scoring.
        self.gen_cfg: GenerationConfig = GenerationConfig()
        # A dedicated (non-global) RNG for generate_pooled()'s behavior-index draws, seeded from
        # `seed` and consumed across the whole sampler's lifetime -- mirrors AAPL's single
        # evolving RNG stream across sample() calls, without perturbing torch's global RNG (which
        # the diffusion attacker's own denoising/remasking randomness relies on).
        self._budget_generator = torch.Generator().manual_seed(int(seed))

    # -- lifecycle ---------------------------------------------------------------------

    def load(self) -> None:
        """Instantiate the attacker on its device (no-op if one was injected)."""
        if self.attacker is None:
            logger.info("Building diffusion attacker for IHO sampling...")
            self.attacker = build_attacker(self.attacker_cfg)

    def unload(self) -> None:
        """Free the attacker's VRAM. An injected attacker is kept (its owner frees it)."""
        if self._owns_attacker:
            self.attacker = None
        free_vram()

    # -- generation --------------------------------------------------------------------

    def generate(
        self,
        behaviors: list[tuple[int, str, str]],
        n_per_behavior: int,
        *,
        phase: str,
        batch_size: int,
    ) -> list[IHOSample]:
        """Inpaint ``n_per_behavior`` adversarial prompts for EACH ``(idx, goal, target)`` behavior.

        Used by the transfer-mode :class:`IHOAttack`, where the budget genuinely is per behavior.
        For a pooled TOTAL budget across ``behaviors`` (AAPL's ``num_sampled_attacks`` semantics),
        use :meth:`generate_pooled` instead.
        """
        counts = [int(n_per_behavior)] * len(behaviors)
        return self._generate_with_counts(behaviors, counts, phase=phase, batch_size=batch_size)

    def generate_pooled(
        self,
        behaviors: list[tuple[int, str, str]],
        total_budget: int,
        *,
        phase: str,
        batch_size: int,
    ) -> list[IHOSample]:
        """Inpaint ``total_budget`` adversarial prompts POOLED across ``behaviors``.

        Reproduces AAPL's ``num_sampled_attacks`` semantics (``aapl/utils/general_utils.py:108-133``
        + ``aapl/AAPLPipeline.py:386-421``): ``total_budget`` is a total sample count for the split,
        not a per-behavior count, so behaviors receive uneven shares -- see
        :meth:`_draw_pooled_counts` for the exact draw algorithm. Used by ``train_attacker.py``'s
        ``training.n_samples.{train,eval,val}``.
        """
        counts = self._draw_pooled_counts(len(behaviors), int(total_budget), int(batch_size))
        return self._generate_with_counts(behaviors, counts, phase=phase, batch_size=batch_size)

    def _draw_pooled_counts(self, n_behaviors: int, total_budget: int, batch_size: int) -> list[int]:
        """AAPL's total-budget-to-per-behavior draw (see the module docstring for the file:line refs).

        Draws behavior indices in ``batch_size``-sized chunks until the running count reaches
        ``total_budget``, then returns how many times each behavior (by position in the ``behaviors``
        list passed to :meth:`generate_pooled`) was drawn:

        * ``n_behaviors < batch_size`` (the common case -- a handful of training behaviors against a
          much larger generation batch size): each chunk is ``batch_size`` i.i.d. draws *with*
          replacement (AAPL's ``RandomSampler(replacement=True, num_samples=batch_size)``).
        * otherwise: each chunk is one random permutation ("epoch") of all behaviors, i.e. sampling
          *without* replacement until the epoch is exhausted, then reshuffling (AAPL's
          ``RandomSampler(replacement=False, num_samples=len(dataset))``).

        Like AAPL, the last chunk is kept whole rather than trimmed, so the actual total drawn can
        exceed ``total_budget`` by up to ``batch_size - 1``. Draws consume ``self._budget_generator``,
        a dedicated ``torch.Generator`` seeded from ``self.seed`` and never reseeded, so repeated
        calls across a run (e.g. once per cycle) are reproducible top-to-bottom under a fixed seed
        but do not repeat the same draw.
        """
        if n_behaviors <= 0 or total_budget <= 0:
            return [0] * max(n_behaviors, 0)
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        use_replacement = n_behaviors < batch_size
        num_per_epoch = batch_size if use_replacement else n_behaviors
        dataset_indices = list(range(n_behaviors))

        drawn: list[int] = []
        while len(drawn) < total_budget:
            epoch = list(
                RandomSampler(
                    dataset_indices,
                    replacement=use_replacement,
                    num_samples=num_per_epoch,
                    generator=self._budget_generator,
                )
            )
            for start in range(0, len(epoch), batch_size):
                if len(drawn) >= total_budget:
                    break
                drawn.extend(epoch[start : start + batch_size])

        counts = [0] * n_behaviors
        for idx in drawn:
            counts[idx] += 1
        return counts

    def _generate_with_counts(
        self,
        behaviors: list[tuple[int, str, str]],
        counts: list[int],
        *,
        phase: str,
        batch_size: int,
    ) -> list[IHOSample]:
        """Shared body of :meth:`generate` / :meth:`generate_pooled`: inpaint ``counts[i]`` prompts
        for ``behaviors[i]``, for each ``i``.

        The target model is moved to CPU while the attacker holds the GPU (footgun §6.3: the target
        is owned by ``run_attacks.py`` and must never be freed, only parked), then restored.
        """
        target_device = None
        if self.target is not None:
            target_device = self.target.model.device
            self.target.model.cpu()
            free_vram()

        self.load()
        attacker = self.attacker
        n_params = int(attacker.n_params_no_embed)
        try:
            samples: list[IHOSample] = []
            for (behavior_idx, goal, target_text), n_per_behavior in zip(behaviors, counts):
                remaining = n_per_behavior
                while remaining > 0:
                    b = min(batch_size, remaining)
                    canvas = attacker.build_attack_canvas([target_text] * b)
                    result = attacker.denoise(canvas.masked_ids)
                    self.ledger.add_denoise(
                        phase=phase, attacker=attacker, batch_size=b, denoise_result=result
                    )

                    prompts = attacker.decode_prompt(result.token_ids, canvas.prompt_positions)
                    fulls = attacker.decode_full(result.token_ids)
                    prompt_ids = attacker.extract_prompt_ids(result.token_ids, canvas.prompt_positions)
                    # Diffusion cost is num_denoise_steps full-canvas forwards, shared across the
                    # batch's rows (identical canvas width), so per sample it is steps * 2*N*L.
                    denoise_flops = int(
                        result.n_forward_passes * pass_flops(n_params, result.n_tokens_per_pass, "forward")
                    )

                    for j in range(b):
                        samples.append(
                            IHOSample(
                                behavior_idx=behavior_idx,
                                goal=goal,
                                target=target_text,
                                prompt_text=prompts[j],
                                prompt_token_ids=[int(t) for t in prompt_ids[j].tolist()],
                                full_text=fulls[j],
                                attacker_loglikelihood=float(result.loglikelihood[j].item()),
                                phase=phase,
                                flops={"attacker_denoise": denoise_flops},
                            )
                        )
                    remaining -= b
        finally:
            self.unload()
            if self.target is not None and target_device is not None:
                self.target.model.to(target_device)

        return samples

    # -- scoring -----------------------------------------------------------------------

    def score(
        self,
        samples: list[IHOSample],
        target: TargetSystem,
        *,
        judges: "dict[str, Judge] | None",
        phase: str,
        compute_target_loss: bool = True,
    ) -> list[IHOSample]:
        """Query ``target`` with each sample's prompt, filling completions, loss, and judge scores.

        Mirrors ``inpainting.py``: the adversarial prompt is the user turn, the affirmative target
        is the assistant turn used for the teacher-forced loss. ``judges`` maps a *role*
        (``"training"`` / ``"validation"``) to a loaded judgezoo judge; scores land in
        ``IHOSample.judge_scores`` keyed by role. In the transfer path ``judges`` is ``None`` and
        post-hoc scoring (``run_judges.py``) does the judging instead (footgun §6.1).
        """
        if not samples:
            return samples

        gen_convs: list[Conversation] = []
        gen_prompt_tensors: list[torch.Tensor] = []
        full_tensors: list[torch.Tensor] = []
        for sample in samples:
            gen_conv: Conversation = [
                {"role": "user", "content": sample.prompt_text},
                {"role": "assistant", "content": ""},
            ]
            gen_convs.append(gen_conv)
            gen_prompt_tensors.append(_flatten_conversation(target.tokenizer, gen_conv))

            loss_conv: Conversation = [
                {"role": "user", "content": sample.prompt_text},
                {"role": "assistant", "content": sample.target},
            ]
            full_tensors.append(_flatten_conversation(target.tokenizer, loss_conv))

        B = len(samples)

        losses: list[float | None] = [None] * B
        if compute_target_loss:
            losses = target.loss(full_tensors, gen_prompt_tensors, initial_batch_size=B, verbose=True)

        generation_result = target.generate(
            gen_convs,
            max_new_tokens=self.gen_cfg.max_new_tokens,
            temperature=self.gen_cfg.temperature,
            top_p=self.gen_cfg.top_p,
            top_k=self.gen_cfg.top_k,
            num_return_sequences=self.gen_cfg.num_return_sequences,
            initial_batch_size=B,
            verbose=True,
        )
        completions = generation_result.gen
        input_ids = generation_result.require_input_ids("IHO", expected_len=B)

        n_params_tgt = int(target.model.num_parameters(exclude_embeddings=True))
        model_id_tgt = _target_model_id(target)

        prompt_token_counts: list[int] = []
        new_token_counts: list[int] = []
        loss_token_counts: list[int] = []
        for i, sample in enumerate(samples):
            first = completions[i][0] if completions[i] else ""
            raw = generation_result.raw_for(i)
            sample.completion = first
            sample.completion_raw = raw[0] if raw else None
            sample.target_loss = losses[i]
            sample.defense_metadata = generation_result.defense_metadata_for(i)
            sample.input_token_ids = list(input_ids[i])

            p_tokens = len(input_ids[i])
            g_tokens = _count_tokens(target.tokenizer, first)
            t_tokens = int(full_tensors[i].size(0))
            prompt_token_counts.append(p_tokens)
            new_token_counts.append(g_tokens)
            loss_token_counts.append(t_tokens)

            sample.flops["target_generate"] = int(pass_flops(n_params_tgt, p_tokens + g_tokens, "forward"))
            if compute_target_loss:
                sample.flops["target_loss"] = int(pass_flops(n_params_tgt, t_tokens, "forward"))

        # One batch-level entry each, so flops.json aggregates match the per-step attribution above.
        self.ledger.add_ar_generation(
            phase=phase,
            model_id=model_id_tgt,
            n_params=n_params_tgt,
            prompt_tokens=prompt_token_counts,
            new_tokens=new_token_counts,
            component="target_generate",
        )
        if compute_target_loss:
            self.ledger.add_ar_generation(
                phase=phase,
                model_id=model_id_tgt,
                n_params=n_params_tgt,
                prompt_tokens=loss_token_counts,
                new_tokens=[0] * B,
                component="target_loss",
            )

        if judges:
            self._score_with_judges(samples, judges, phase=phase)

        return samples

    def _score_with_judges(self, samples: list[IHOSample], judges: "dict[str, Judge]", *, phase: str) -> None:
        for role, judge in judges.items():
            chats = [
                [
                    {"role": "user", "content": sample.goal},
                    {"role": "assistant", "content": sample.completion or ""},
                ]
                for sample in samples
            ]
            p_harmful = judge(chats)["p_harmful"]
            for sample, score in zip(samples, p_harmful):
                sample.judge_scores[role] = score

    # -- dataframe ---------------------------------------------------------------------

    def to_dataframe(self, samples: list[IHOSample]) -> pd.DataFrame:
        """Serialize samples to the plan §4.6 schema (exact columns in :data:`IHO_DATAFRAME_COLUMNS`)."""
        rows = []
        for sample in samples:
            match = _CYCLE_RE.search(sample.phase)
            rows.append(
                {
                    "jb_index": sample.behavior_idx,
                    "goal_text": sample.goal,
                    "original_prompt_text": sample.goal,
                    "attacking_prompt_text": sample.prompt_text,
                    "attacking_prompt_ids": sample.prompt_token_ids,
                    "inpainted_prompt_text_full": sample.full_text,
                    "attack_loglikelihood": sample.attacker_loglikelihood,
                    "attacked_output": sample.completion,
                    "attacked_output_raw": sample.completion_raw,
                    "judge_score_training": sample.judge_scores.get("training"),
                    "judge_score_validation": sample.judge_scores.get("validation"),
                    "cycle_id": int(match.group(1)) if match else None,
                    "phase_id": sample.phase,
                    "flops_attacker_denoise": sample.flops.get("attacker_denoise", 0),
                    "flops_target_generate": sample.flops.get("target_generate", 0),
                    "flops_target_loss": sample.flops.get("target_loss", 0),
                    "flops_total": sample.total_flops(),
                }
            )
        return pd.DataFrame(rows, columns=list(IHO_DATAFRAME_COLUMNS))


# --------------------------------------------------------------------------------------
# Attack
# --------------------------------------------------------------------------------------


@_typed
@dataclass
class IHOConfig:
    name: str = "iho"
    type: str = "discrete"
    version: str = "0.1.0"
    generation_config: GenerationConfig = field(default_factory=GenerationConfig)
    seed: int = 0

    attacker: DiffusionAttackerConfig = field(default_factory=DiffusionAttackerConfig)
    num_samples_per_behavior: int = 64
    attacker_batch_size: int = 64
    compute_target_loss: bool = True

    # Adaptive mode. IHO's optimization *is* DPO-training the diffusion attacker, so `attack=iho`
    # trains by default (like every other attack runs its own search). Two ways to get transfer
    # instead: set a `lora_checkpoint` on `attacker` (evaluate/transfer a trained adapter -- forces
    # skip-training), or `train=false` with no checkpoint (evaluate the untrained base attacker,
    # i.e. the transfer baseline). `training` carries the full training block (see
    # conf/attacks/attacks.yaml) and is required whenever training actually runs.
    train: bool = True
    training: Any = None
    attacker_dir: str | None = None  # where a trained adapter is written
    # After training, also sample the winning checkpoint fresh and log those as inline run.json
    # steps (the repo-normal transfer eval). Set False for a training-only run: run.json then holds
    # no inline steps, only the linked training-sample manifest in resume_metadata.
    eval_after_train: bool = True


class IHOAttack(Attack):
    """Transfer-mode IHO: run the (untrained or LoRA-loaded) diffusion attacker over a dataset.

    One :class:`SingleAttackRunResult` per behavior, one :class:`AttackStepResult` per sample. The
    default path passes ``judges=None`` so ``run_judges.py`` does post-hoc scoring and
    ``AttackStepResult.scores`` stays empty (footgun §6.1).
    """

    def __init__(self, config: Any):
        super().__init__(config)
        # config.attacker arrives as an OmegaConf DictConfig from run_attacks.py; build_attacker and
        # LLaDADiffusionAttacker both accept that, so it is passed through untouched (never asdict'd).

    @torch.no_grad
    def run(self, target: TargetSystem, dataset: PromptDataset) -> AttackResult:
        cfg = self.config

        # Adaptive mode (plan §4.1 step 1): train a LoRA adapter against *this* target first, point
        # the evaluation attacker at the winning adapter, and keep the training provenance for the
        # per-run resume_metadata. Training FLOPs stay out of the per-step flops (plan §5.3).
        # `attack=iho` trains by default; a `lora_checkpoint` forces transfer (evaluate the given
        # adapter without retraining), and `train=false` skips training entirely.
        has_checkpoint = getattr(getattr(cfg, "attacker", None), "lora_checkpoint", None) is not None
        do_train = bool(getattr(cfg, "train", True)) and not has_checkpoint
        train_provenance: dict[str, Any] | None = None
        if do_train:
            train_provenance = self._train_before_attack(target, dataset)

        conversations: list[Conversation] = list(dataset)

        # Training-only run: skip the fresh eval sampling. Each behavior's run.json still carries the
        # training-sample manifest (resume_metadata) linking every training attempt against it.
        if do_train and not bool(getattr(cfg, "eval_after_train", True)):
            logger.info(
                "IHO training-only (eval_after_train=False): logged training-sample manifest for %d behaviors.",
                len(conversations),
            )
            return AttackResult(runs=[
                SingleAttackRunResult(original_prompt=conv, steps=[], total_time=0.0,
                                      resume_metadata=train_provenance)
                for conv in conversations
            ])

        t0 = time.time()
        behaviors: list[tuple[int, str, str]] = []
        for i, conversation in enumerate(conversations):
            assert len(conversation) == 2, "IHO currently assumes single-turn conversations."
            behaviors.append((i, conversation[0]["content"], conversation[1]["content"]))

        ledger = FlopsLedger()
        sampler = IHOSampler(cfg.attacker, ledger=ledger, seed=cfg.seed)
        sampler.gen_cfg = cfg.generation_config
        sampler.target = target

        samples = sampler.generate(
            behaviors,
            int(cfg.num_samples_per_behavior),
            phase="eval",
            batch_size=int(cfg.attacker_batch_size),
        )
        # Transfer path: no in-loop judges (footgun §6.1) -- run_judges.py scores post hoc.
        sampler.score(
            samples,
            target,
            judges=None,
            phase="eval",
            compute_target_loss=bool(cfg.compute_target_loss),
        )
        t1 = time.time()

        # Group samples back by behavior, preserving generation order (step j = 0..n-1).
        by_behavior: dict[int, list[IHOSample]] = {}
        for sample in samples:
            by_behavior.setdefault(sample.behavior_idx, []).append(sample)

        per_sample_time = (t1 - t0) / max(len(samples), 1)
        runs: list[SingleAttackRunResult] = []
        for i, conversation in enumerate(conversations):
            step_results = []
            for j, sample in enumerate(by_behavior.get(i, [])):
                model_input: Conversation = [
                    {"role": "user", "content": sample.prompt_text},
                    {"role": "assistant", "content": ""},
                ]
                step_results.append(
                    AttackStepResult(
                        step=j,
                        model_completions=[sample.completion] if sample.completion is not None else [],
                        model_completions_raw=[sample.completion_raw] if sample.completion_raw is not None else None,
                        time_taken=per_sample_time,
                        loss=sample.target_loss,
                        flops=sample.total_flops(),
                        model_input=model_input,
                        model_input_tokens=sample.input_token_ids,
                        defense_metadata=sample.defense_metadata,
                    )
                )
            runs.append(
                SingleAttackRunResult(
                    original_prompt=conversation,
                    steps=step_results,
                    total_time=t1 - t0,
                    resume_metadata=train_provenance,
                )
            )

        logger.info(
            "IHO attack completed: %d behaviors x %d samples in %.2fs (%.3e total FLOPs).",
            len(conversations),
            int(cfg.num_samples_per_behavior),
            t1 - t0,
            ledger.total(),
        )
        return AttackResult(runs=runs)

    def _train_before_attack(self, target: TargetSystem, dataset: PromptDataset) -> dict[str, Any]:
        """Run the phase-3 training loop against ``target``; set ``lora_checkpoint`` and return provenance.

        The loop lives in ``adversariallm.training.attacker_loop`` (imported lazily to break the
        genuine ``iho`` <-> ``attacker_loop`` cycle: that module imports :class:`IHOSampler` from
        here). The trained adapter is amortised over the training behaviors, so its total FLOPs and
        behavior count are returned separately for the per-run ``resume_metadata`` (plan §5.3),
        never folded into per-step evaluation FLOPs.
        """
        from ..training.attacker_loop import train_attacker_run  # noqa: PLC0415 - lazy: breaks the import cycle

        cfg = self.config
        if getattr(cfg, "training", None) is None:
            raise ValueError(
                "IHO training is enabled (train=true, no lora_checkpoint) but no `training` block "
                "was supplied. Provide attacks.iho.training (see conf/attacks/attacks.yaml), pass a "
                "lora_checkpoint to transfer a trained adapter, or set attacks.iho.train=false."
            )

        model_id = _target_model_id(target)
        dataset_name = str(getattr(getattr(dataset, "config", None), "name", "dataset"))
        defense_name = str(getattr(target, "NAME", None) or getattr(type(target), "NAME", "none"))
        attacker_dir = getattr(cfg, "attacker_dir", None)
        if not attacker_dir:
            attacker_dir = os.path.join(os.getcwd(), "attackers")
            logger.warning("IHOConfig.attacker_dir is unset; writing trained adapters under %s.", attacker_dir)

        # run() is decorated @torch.no_grad for the transfer/eval path, but DPO training needs
        # gradients, so re-enable them just around the training loop.
        with torch.enable_grad():
            result = train_attacker_run(
                target=target,
                dataset=dataset,
                training=cfg.training,
                model_id=model_id,
                dataset_name=dataset_name,
                defense_name=defense_name,
                defense_params=None,  # the attack does not see raw defense params; hash on name only
                attacker_dir=str(attacker_dir),
                generation_config=cfg.generation_config,
            )

        if result.best_checkpoint is None:
            raise RuntimeError("Attacker training produced no checkpoint; cannot run the adaptive attack.")

        # Point the evaluation attacker at the winning adapter (lora / lora_checkpoint are exclusive).
        attacker_cfg = cfg.attacker
        if isinstance(attacker_cfg, DictConfig):
            OmegaConf.set_struct(attacker_cfg, False)
            attacker_cfg.lora = None
            attacker_cfg.lora_checkpoint = result.best_checkpoint
        else:
            attacker_cfg.lora = None
            attacker_cfg.lora_checkpoint = result.best_checkpoint

        logger.info(
            "IHO adaptive: trained %s (winning cycle %s, metric %s); evaluating adapter %s.",
            result.attacker_id, result.winning_cycle, result.best_metric, result.best_checkpoint,
        )

        # Link every training attempt (all cycles, pre-DPO + mid-training eval) into run.json via a
        # compact manifest of the parquets under run_dir, tagged by phase/cycle/epoch with a
        # judge->score-column map. The heavy sample rows stay in the parquets (size); run.json links
        # them. See adversariallm/io_utils/iho_training_samples.py.
        training_samples = build_training_samples_manifest(
            result.run_dir,
            train_judge=_training_get(cfg.training, "train_judge"),
            val_judges=_training_get(cfg.training, "val_judges"),
        )

        return {
            "iho_training": {
                "attacker_id": result.attacker_id,
                "run_dir": result.run_dir,
                "checkpoint": result.best_checkpoint,
                "best_metric": result.best_metric,
                "winning_cycle": result.winning_cycle,
                "n_cycles": result.n_cycles_completed,
                "total_training_flops": result.total_training_flops,
                "n_behaviors_amortised": result.n_behaviors_amortised,
                "train_config": result.train_config,
                "training_samples": training_samples,
            }
        }


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def _flatten_conversation(tokenizer: Any, conversation: Conversation) -> torch.Tensor:
    """Tokenize a conversation into a single flat id tensor, exactly like ``inpainting.py``."""
    token_tensors = prepare_conversation(tokenizer, conversation)
    flat = [t for turn_tokens in token_tensors for t in turn_tokens]
    return torch.cat(flat, dim=0)


def _count_tokens(tokenizer: Any, text: str) -> int:
    """Number of tokens in ``text`` (no special tokens), for the generation FLOPs count."""
    if not text:
        return 0
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def _target_model_id(target: TargetSystem) -> str:
    """Best-effort identifier for the target model, for FLOPs-ledger provenance."""
    name = getattr(target.model, "name_or_path", None)
    if isinstance(name, str) and name:
        return name
    return type(target).__name__


def _training_get(training: Any, key: str) -> Any:
    """Read ``key`` from the training config, tolerating an OmegaConf DictConfig or a plain dict."""
    if training is None:
        return None
    getter = getattr(training, "get", None)
    if callable(getter):
        try:
            return training.get(key)
        except Exception:
            pass
    return getattr(training, key, None)
