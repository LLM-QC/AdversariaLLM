"""LLaDA diffusion attacker.

Port of ``aapl/model_wrapper/LLaDAWrapper.py`` (AAPL commit ``7a449cf``) onto the
:class:`~adversariallm.attackers.base.Attacker` interface. The sampling numerics are
deliberately kept bit-for-bit identical to AAPL: same Gumbel noise, same special-token
suppression, same transfer schedule, same re-noising, same global remasking. Deviations from
the original are limited to genuine bugs and are marked with an ``AAPL:`` comment.

LLaDA is a masked *diffusion* LM, so it is loaded with ``AutoModel`` + ``trust_remote_code``
(``AutoModelForCausalLM`` has no entry in its ``auto_map``) and the repo-wide
``load_model_and_tokenizer`` helper cannot be used.
"""

import inspect
import logging
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Iterator, Literal

import torch
import torch.nn.functional as F
from beartype import beartype
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModel, AutoTokenizer
from transformers.dynamic_module_utils import get_class_from_dynamic_module

# The plan (§3.2) mandates reusing this helper rather than reimplementing the shared-cache
# lock fallback; it is private only because nothing outside model loading needed it before.
from ..io_utils.model_loading import _with_shared_snapshot_fallback
from .base import Attacker, DenoiseResult, MaskingMode, MaskingResult

logger = logging.getLogger(__name__)

#: The literal LLaDA mask token; its id is ``DiffusionAttackerConfig.mask_token_id`` (126336).
LLADA_MASK_TOKEN = "<|mdm_mask|>"

#: Dotted reference into LLaDA's remote code, as declared in its ``config.json`` ``auto_map``.
LLADA_MODEL_CLASS_REF = "modeling_llada.LLaDAModelLM"


# --------------------------------------------------------------------------------------
# transformers 5.x compatibility shim -- delete once LLaDA's remote code is fixed upstream
# --------------------------------------------------------------------------------------


def _apply_tied_keys_patch(cls: type) -> bool:
    """Make ``cls.__init__`` set ``all_tied_weights_keys`` if the base class did not.

    Split out from :func:`_patch_llada_for_transformers_v5` so it can be unit tested without
    downloading remote code. Returns ``True`` if the patch was applied, ``False`` if ``cls``
    was already patched.
    """
    if getattr(cls, "_adversariallm_tied_keys_patched", False):
        return False

    orig_init = cls.__init__

    def __init__(self, config, *args, **kwargs):  # noqa: N807 - replaces cls.__init__
        orig_init(self, config, *args, **kwargs)
        if not hasattr(self, "all_tied_weights_keys"):
            # A fresh dict per instance: transformers' tie_weights() mutates this in place, so a
            # shared class-level dict would leak tied keys between models.
            self.all_tied_weights_keys = {}

    cls.__init__ = __init__
    cls._adversariallm_tied_keys_patched = True
    return True


def _apply_tie_weights_patch(cls: type) -> bool:
    """Let ``cls.tie_weights`` swallow the keyword arguments transformers 5.x passes to it.

    ``modeling_utils._finalize_model_loading`` calls
    ``model.tie_weights(missing_keys=..., recompute_mapping=False)``; LLaDA's override is
    ``def tie_weights(self)``, so ``from_pretrained`` dies with a ``TypeError``. The override is
    a no-op under the shipped ``"weight_tying": false`` config, so dropping the kwargs is safe.

    Self-guarding: after wrapping, the signature accepts ``**kwargs`` and this returns ``False``.
    """
    original = getattr(cls, "tie_weights", None)
    if original is None:
        return False
    params = inspect.signature(original).parameters
    if "missing_keys" in params or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return False

    def tie_weights(self, *args, **kwargs):  # noqa: ARG001 - kwargs are deliberately dropped
        return original(self)

    cls.tie_weights = tie_weights
    return True


def _patch_llada_for_transformers_v5(model_id: str, revision: str | None = None) -> None:
    """LLaDA's remote code never calls ``post_init()``, so ``all_tied_weights_keys`` is unset.

    ``LLaDAModelLM.__init__`` calls ``super().__init__(config)`` and stops there. In
    transformers 5.x it is ``post_init()`` that assigns ``self.all_tied_weights_keys``, and
    ``modeling_utils._finalize_model_loading`` -> ``_move_missing_keys_from_meta_to_device``
    does ``missing_keys - self.all_tied_weights_keys.keys()`` with no ``getattr`` default, so
    ``from_pretrained`` dies with::

        AttributeError: 'LLaDAModelLM' object has no attribute 'all_tied_weights_keys'

    This is not fixed by a transformers bump (5.10.4 and 5.12.1 are identical here). LLaDA's
    ``config.json`` sets ``"weight_tying": false`` and the class sets ``_tied_weights_keys =
    None``, i.e. nothing is tied, so the correct value is an empty dict.

    A second, independent incompatibility in the same class is fixed alongside it, see
    :func:`_apply_tie_weights_patch`.

    Idempotent, and a no-op (with a warning) if the remote class cannot be resolved, so a
    future fixed checkpoint keeps loading.
    """
    try:
        cls = get_class_from_dynamic_module(LLADA_MODEL_CLASS_REF, model_id, revision=revision)
    except Exception as exc:  # noqa: BLE001 - never block loading on a compat shim
        logger.warning(
            "Could not resolve %s for %s to apply the transformers-5 tied-keys shim (%s); "
            "continuing unpatched.",
            LLADA_MODEL_CLASS_REF,
            model_id,
            exc,
        )
        return

    if _apply_tied_keys_patch(cls):
        logger.info("Patched %s.__init__ to set all_tied_weights_keys (transformers 5.x compat).", cls.__name__)
    if _apply_tie_weights_patch(cls):
        logger.info("Patched %s.tie_weights to accept transformers 5.x keyword arguments.", cls.__name__)


# --------------------------------------------------------------------------------------
# Configs
# --------------------------------------------------------------------------------------


@beartype
@dataclass
class LoRASpec:
    """Serializable mirror of ``peft.LoraConfig``, so the adapter is part of the run config."""

    r: int = 8
    lora_alpha: int = 16
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    use_rslora: bool = False


@beartype
@dataclass
class DiffusionAttackerConfig:
    backend: Literal["llada"] = "llada"
    id: str = "GSAI-ML/LLaDA-8B-Base"
    tokenizer_id: str | None = None  # defaults to `id`
    dtype: str = "bfloat16"
    trust_remote_code: bool = True
    mask_token_id: int = 126336
    prompt_len: int = 32  # AAPL `attacker_input_size`
    num_denoise_steps: int = 32  # AAPL `attacker_step_number`
    temperature: float = 0.0
    remasking: Literal["low_confidence", "random"] = "low_confidence"
    mask_padding_tokens: bool = True
    global_remask_every: int = 8  # AAPL `number_global_remask`; 0 disables
    global_remasking: Literal["random", "low_confidence"] = "random"
    answer_prefix: str = "\nAnswer: "
    lora: LoRASpec | None = None
    lora_checkpoint: str | None = None

    def __post_init__(self) -> None:
        _validate_lora_fields(self)


def _validate_lora_fields(cfg: Any) -> None:
    """``lora`` (fresh adapter) and ``lora_checkpoint`` (trained adapter) are exclusive.

    Both ``None`` is legal and means "untrained base attacker", i.e. the transfer baseline.
    AAPL raised in that case; the plan (§3.2) explicitly overrides that.
    """
    if getattr(cfg, "lora", None) is not None and getattr(cfg, "lora_checkpoint", None) is not None:
        raise ValueError(
            "DiffusionAttackerConfig: `lora` and `lora_checkpoint` are mutually exclusive "
            "(pass `lora` to initialise a fresh adapter, `lora_checkpoint` to load a trained one)."
        )


def _lora_config_from_spec(spec: Any) -> LoraConfig:
    """Build a ``peft.LoraConfig`` from a :class:`LoRASpec`, a dict, or an OmegaConf node."""
    if isinstance(spec, LoraConfig):
        return spec
    if is_dataclass(spec):
        kwargs = asdict(spec)
    elif isinstance(spec, Mapping):
        kwargs = dict(spec)
    else:
        kwargs = {f: getattr(spec, f) for f in LoRASpec.__dataclass_fields__}
    kwargs["target_modules"] = list(kwargs["target_modules"])  # ListConfig -> list
    return LoraConfig(**kwargs)


# --------------------------------------------------------------------------------------
# Attacker
# --------------------------------------------------------------------------------------


class LLaDADiffusionAttacker(Attacker):
    """Masked-diffusion attacker backed by ``GSAI-ML/LLaDA-8B-*``."""

    def __init__(
        self,
        cfg: DiffusionAttackerConfig,
        *,
        device: str | torch.device | None = None,
        model: Any = None,
        tokenizer: Any = None,
    ) -> None:
        """``model``/``tokenizer`` are injection points for tests; production passes neither.

        No small masked-diffusion LM exists (LLaDA-8B is the floor, plan §6.6), so unit tests
        substitute a tiny fake module instead of loading real weights.
        """
        _validate_lora_fields(cfg)
        self.cfg = cfg
        self.model_id = cfg.id
        self.mask_token_id = int(cfg.mask_token_id)
        self.prompt_len = int(cfg.prompt_len)
        self.device = torch.device(device) if device is not None else _default_device()
        self._n_params_no_embed: int | None = None

        if tokenizer is None:
            tokenizer = _with_shared_snapshot_fallback(
                AutoTokenizer.from_pretrained,
                cfg.tokenizer_id or cfg.id,
                trust_remote_code=cfg.trust_remote_code,
            )
        try:
            tokenizer.mask_token = LLADA_MASK_TOKEN
        except Exception as exc:  # noqa: BLE001 - cosmetic only; masking goes through mask_token_id
            logger.debug("Could not set tokenizer.mask_token: %s", exc)
        self.tokenizer = tokenizer
        self.padding_token_id = self.tokenizer.pad_token_id

        if model is None:
            _patch_llada_for_transformers_v5(cfg.id)
            logger.info("Loading diffusion attacker %s...", cfg.id)
            model = _with_shared_snapshot_fallback(
                AutoModel.from_pretrained,
                cfg.id,
                trust_remote_code=cfg.trust_remote_code,
                dtype=_resolve_dtype(cfg.dtype),
            )
        self.model = model.to(self.device)
        # LLaDA's remote `forward` falls back to `self.config.use_cache`, which transformers 5.x
        # no longer defines on `PretrainedConfig` (AttributeError on every forward). Diffusion
        # sampling is a full-canvas bidirectional pass with no KV cache, so pinning it to False
        # is both the fix and the semantically correct value.
        if getattr(self.model, "config", None) is not None:
            self.model.config.use_cache = False

        self.has_lora = False
        if cfg.lora_checkpoint is not None:
            logger.info("Loading LoRA checkpoint from %s...", cfg.lora_checkpoint)
            # AAPL: default `is_trainable=False`, which silently freezes the adapter and makes a
            # resumed DPO cycle a no-op. Inference is unaffected (denoise runs under no_grad).
            self.model = PeftModel.from_pretrained(self.model, cfg.lora_checkpoint, is_trainable=True)
            self.has_lora = True
        elif cfg.lora is not None:
            logger.info("Initializing LoRA layers from configuration...")
            self.model = get_peft_model(self.model, _lora_config_from_spec(cfg.lora))
            self.has_lora = True
            trainable, total = self.model.get_nb_trainable_parameters()
            logger.info(
                "trainable params: %d || all params: %d || trainable%%: %.4f", trainable, total, 100 * trainable / total
            )
        else:
            logger.info("No LoRA adapter configured: using the untrained base attacker (transfer baseline).")

        self.model.eval()

    # -- lifecycle ---------------------------------------------------------------------

    @property
    def n_params_no_embed(self) -> int:
        if self._n_params_no_embed is None:
            # Counted on the *base* model: the adapter adds <0.1% of N, and using the base count
            # keeps the FLOPs ledger comparable between the untrained baseline and a trained attacker.
            base = self.model.get_base_model() if self.has_lora else self.model
            self._n_params_no_embed = int(base.num_parameters(exclude_embeddings=True))
        return self._n_params_no_embed

    def train(self) -> "LLaDADiffusionAttacker":
        self.model.train()
        return self

    def eval(self) -> "LLaDADiffusionAttacker":
        self.model.eval()
        return self

    def to(self, device: str | torch.device) -> "LLaDADiffusionAttacker":
        self.device = torch.device(device)
        self.model = self.model.to(self.device)
        return self

    @contextmanager
    def disable_adapter(self) -> Iterator[None]:
        if not self.has_lora:
            yield
            return
        self.model.disable_adapter_layers()
        try:
            yield
        finally:
            self.model.enable_adapter_layers()

    def save_lora(self, path: str) -> None:
        if not self.has_lora:
            raise RuntimeError("Model does not have LoRA adapters to save")
        logger.info("Saving LoRA adapter to %s...", path)
        self.model.save_pretrained(path)

    def load_lora(self, path: str) -> None:
        """Overwrite the active adapter's weights in-place from ``path`` (plan: cycle reset).

        AAPL reloads the attacker from the previous cycle's *best* checkpoint before each new
        cycle by rebuilding the whole wrapper. We keep the same PEFT module (so the sampler and a
        freshly built trainer keep valid references) and only replace the adapter tensors in
        place, which is equivalent for the "default" adapter and far cheaper than a reload.
        """
        if not self.has_lora:
            raise RuntimeError("Model does not have LoRA adapters to load into")
        from peft import set_peft_model_state_dict
        from peft.utils import load_peft_weights

        logger.info("Reloading LoRA adapter weights from %s...", path)
        state_dict = load_peft_weights(path, device=str(self.device))
        load_result = set_peft_model_state_dict(self.model, state_dict)
        missing = getattr(load_result, "missing_keys", None)
        unexpected = getattr(load_result, "unexpected_keys", None)
        if unexpected:
            raise RuntimeError(f"Unexpected keys reloading LoRA from {path}: {unexpected}")
        if missing:
            # LoRA state dicts only cover adapter tensors; base-model keys are legitimately absent.
            adapter_missing = [k for k in missing if "lora_" in k]
            if adapter_missing:
                raise RuntimeError(f"Missing adapter keys reloading LoRA from {path}: {adapter_missing}")

    # -- tokenisation and canvases -----------------------------------------------------

    def encode(self, texts: list[str]) -> torch.Tensor:
        encoded = self.tokenizer(texts, padding=True, return_tensors="pt", add_special_tokens=True)
        return encoded["input_ids"].to(self.device)

    def build_attack_canvas(self, target_texts: list[str]) -> MaskingResult:
        """Canvas = ``prompt_len`` mask tokens followed by ``answer_prefix + target``.

        The masked span is exactly the leading block, so denoising inpaints an adversarial
        prompt conditioned on the affirmative answer that follows it.
        """
        target_ids = self.encode([self.cfg.answer_prefix + t for t in target_texts])
        return self.mask_tokens(target_ids, masking_mode="attack", mask_all=True)

    def mask_tokens(
        self,
        token_ids: torch.Tensor,
        masking_mode: MaskingMode,
        mask_all: bool = False,
    ) -> MaskingResult:
        B, L = token_ids.shape
        device = token_ids.device
        K = self.prompt_len

        if masking_mode == "all":
            eligible_mask = torch.ones((B, L), dtype=torch.bool, device=device)
        elif masking_mode == "prompt":
            eligible_mask = torch.zeros((B, L), dtype=torch.bool, device=device)
            eligible_mask[:, :K] = True
        elif masking_mode == "attack":
            mask_block = token_ids.new_full((B, K), self.mask_token_id)
            token_ids = torch.cat([mask_block, token_ids], dim=1)
            eligible_mask = torch.zeros(token_ids.shape, dtype=torch.bool, device=device)
            eligible_mask[:, :K] = True
        else:
            raise ValueError(f"Unknown masking_mode: {masking_mode}")

        if mask_all:
            mask_positions = eligible_mask.clone()
        else:
            # AAPL drew `rand` with the pre-concat width, which cannot broadcast against the
            # widened canvas in "attack" mode. Identical to AAPL for "all"/"prompt".
            p = torch.rand(B, device=device).view(B, 1)
            rand = torch.rand(eligible_mask.shape, device=device)
            mask_positions = (rand < p) & eligible_mask

        masked_ids = token_ids.clone()
        masked_ids[mask_positions] = self.mask_token_id

        # The adversarial prompt always occupies the leading `prompt_len` columns of the canvas.
        prompt_positions = torch.zeros(eligible_mask.shape, dtype=torch.bool, device=device)
        prompt_positions[:, :K] = True

        return MaskingResult(masked_ids=masked_ids, mask_positions=mask_positions, prompt_positions=prompt_positions)

    def extract_prompt_ids(self, denoised_ids: torch.Tensor, prompt_positions: torch.Tensor) -> torch.Tensor:
        return denoised_ids[prompt_positions].view(denoised_ids.size(0), -1)

    def decode_prompt(self, denoised_ids: torch.Tensor, prompt_positions: torch.Tensor) -> list[str]:
        return self.tokenizer.batch_decode(
            self.extract_prompt_ids(denoised_ids, prompt_positions), skip_special_tokens=False
        )

    def decode_full(self, denoised_ids: torch.Tensor) -> list[str]:
        return self.tokenizer.batch_decode(denoised_ids, skip_special_tokens=False)

    # -- likelihoods and sampling ------------------------------------------------------

    def compute_log_likelihood(
        self,
        masked_ids: torch.Tensor,
        mask_positions: torch.Tensor,
        target_ids: torch.Tensor,
        use_base_model: bool = False,
    ) -> torch.Tensor:
        """Sum of ``target_ids`` logprobs over ``mask_positions``, per sequence."""
        if use_base_model:
            # Plan §3.7: the DPO reference forward must be under no_grad *and* disable_adapter.
            # AAPL only disabled the adapter, so the reference branch built a graph it never used.
            with torch.no_grad(), self.disable_adapter():
                logits = self.model(masked_ids).logits
        elif not self.model.training:
            with torch.no_grad():
                logits = self.model(masked_ids).logits
        else:
            logits = self.model(masked_ids).logits

        log_probs = F.log_softmax(logits, dim=-1)
        target_log_probs = torch.gather(log_probs, dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
        return (target_log_probs * mask_positions).sum(dim=1)

    @torch.no_grad()
    def _add_gumbel_noise(self, logits: torch.Tensor, temperature: float) -> torch.Tensor:
        if temperature == 0:
            return logits
        eps = 1e-20
        # bfloat16 noise is a deliberate memory tradeoff in AAPL; keep the cast and the
        # divide-by-temperature exactly as-is, they change the sampled tokens.
        return (logits - torch.log(-torch.log(torch.rand_like(logits, dtype=torch.bfloat16) + eps) + eps)) / temperature

    @torch.no_grad()
    def _forward_process_batched(
        self, batch: torch.Tensor, fixed_mask: torch.Tensor, mask_id: int = 126336
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Re-noise: mask a random prefix-count of positions, shuffled, restricted to unknowns."""
        b, l = batch.shape
        device = batch.device

        target_len = l

        x = torch.randint(1, target_len + 1, (b,), device=device)

        indices = torch.arange(target_len, device=device).unsqueeze(0).expand(b, -1)

        is_mask = indices < x.unsqueeze(1)

        randperm = torch.argsort(torch.rand(b, target_len, device=device), dim=1)
        is_mask = torch.gather(is_mask, 1, randperm)

        is_mask = is_mask & fixed_mask

        noisy_batch = torch.where(is_mask, mask_id, batch)

        mask_ratio = (x / target_len).unsqueeze(1).expand(-1, l)

        return noisy_batch, mask_ratio

    def _get_num_transfer_tokens(self, mask_index: torch.Tensor, steps: int) -> torch.Tensor:
        """Spread the masked positions as evenly as possible over ``steps`` transfers."""
        mask_num = mask_index.sum(dim=1, keepdim=True)
        base = mask_num // steps
        remainder = mask_num % steps

        num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

        for i in range(mask_num.size(0)):
            num_transfer_tokens[i, : remainder[i]] += 1

        return num_transfer_tokens

    @torch.no_grad()
    def denoise(self, masked_ids: torch.Tensor) -> DenoiseResult:
        """AAPL's ``predict_masked`` with the config baked in.

        The per-sequence Python loop is kept: global remasking selects a different number of
        positions per row, so vectorising it is not obviously bit-identical (plan §3.2).
        """
        if self.model.training:
            raise RuntimeError("denoise must be run in evaluation mode.")

        steps = int(self.cfg.num_denoise_steps)
        temperature = float(self.cfg.temperature)
        remasking = self.cfg.remasking
        mask_padding = bool(self.cfg.mask_padding_tokens)
        number_global_remask = int(self.cfg.global_remask_every)
        global_remasking = self.cfg.global_remasking

        device = self.device
        # AAPL wrote into `masked_ids` in place when it was already on the target device; clone
        # so the caller's canvas survives denoising.
        x = masked_ids.to(device).clone()
        batch_size, seq_len = x.shape

        known_mask = x != self.mask_token_id
        known_tokens = x.clone()
        global_conf = torch.zeros((batch_size, seq_len), dtype=torch.float32, device=device)
        total_loglikelihood = torch.zeros(batch_size, device=device)

        special_token_ids = [
            self.tokenizer.bos_token_id,
            self.tokenizer.eos_token_id,
            self.tokenizer.pad_token_id,
        ]
        special_token_ids = list(set(int(s) for s in special_token_ids if s is not None))

        mask_index_initial = x == self.mask_token_id
        num_transfer_tokens = self._get_num_transfer_tokens(mask_index_initial, steps)

        for s in range(steps):
            mask_index = x == self.mask_token_id

            x_l, _ = self._forward_process_batched(x, known_mask, mask_id=self.mask_token_id)

            attention_mask = None
            if mask_padding:
                attention_mask = (x_l != self.padding_token_id).long()

            # AAPL passed this positionally; on the bare LLaDA class the second positional is
            # `input_embeddings`, so it only worked because PEFT always wrapped the model.
            logits = self.model(x_l, attention_mask=attention_mask).logits
            logits_with_noise = self._add_gumbel_noise(logits, temperature=temperature)

            if len(special_token_ids) > 0:
                logits_with_noise[:, :, special_token_ids] = -float("inf")

            x0 = torch.argmax(logits_with_noise, dim=-1)
            log_probs = F.log_softmax(logits, dim=-1)
            chosen_logp = torch.gather(log_probs, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)

            if remasking == "low_confidence":
                p = F.softmax(logits, dim=-1)
                idx = x0.unsqueeze(-1)
                x0_p = torch.gather(p, dim=-1, index=idx).squeeze(-1).to(device)
            elif remasking == "random":
                x0_p = torch.rand((batch_size, seq_len), device=device)
            else:
                raise NotImplementedError(f"Remasking strategy '{remasking}' not implemented")

            x0 = torch.where(known_mask, known_tokens, x0)

            neg_inf = torch.tensor(-float("inf"), device=device)
            confidence = torch.where(mask_index, x0_p, neg_inf)
            confidence = torch.where(known_mask, neg_inf, confidence)

            transfer_index = torch.zeros_like(x, dtype=torch.bool, device=device)
            new_mask_index = torch.zeros_like(x, dtype=torch.bool, device=device)

            for b in range(batch_size):
                k = int(num_transfer_tokens[b, s].item())

                # Global remasking only fires once this row has nothing left to transfer, which
                # is what turns the schedule into a refinement loop over already-filled tokens.
                if (
                    k == 0
                    and s < steps - 1 - number_global_remask
                    and number_global_remask > 0
                    and (s % number_global_remask == 0)
                ):
                    unknown_indices = (~known_mask[b]).nonzero(as_tuple=True)[0]

                    if global_remasking == "random":
                        if len(unknown_indices) >= number_global_remask:
                            rnd = torch.randperm(len(unknown_indices), device=device)[:number_global_remask]
                            random_index = unknown_indices[rnd]
                            new_mask_index[b, random_index] = True
                    elif global_remasking == "low_confidence":
                        if len(unknown_indices) >= number_global_remask:
                            unknown_confidence = global_conf[b][unknown_indices]
                            _, local_indices = unknown_confidence.topk(number_global_remask, largest=False)
                            selected_indices = unknown_indices[local_indices]
                            new_mask_index[b, selected_indices] = True

                k = max(k, 1)
                _, select_index = torch.topk(confidence[b], k=k)
                transfer_index[b, select_index] = True

            x[transfer_index] = x0[transfer_index]
            step_loglik = (chosen_logp * transfer_index).sum(dim=1)
            total_loglikelihood += step_loglik

            global_conf[transfer_index] = confidence[transfer_index].float()
            x[new_mask_index] = self.mask_token_id

        return DenoiseResult(
            token_ids=x,
            loglikelihood=total_loglikelihood,
            n_forward_passes=steps,
            n_tokens_per_pass=seq_len,
        )


def _default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_dtype(dtype: str) -> torch.dtype:
    resolved = getattr(torch, dtype, None)
    if not isinstance(resolved, torch.dtype):
        raise ValueError(f"Unknown torch dtype: {dtype!r}")
    return resolved
