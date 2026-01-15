"""
Model and tokenizer loading utilities.

This module provides functions for loading and configuring language models
and tokenizers with various optimizations and model-specific configurations.
"""

import gc
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from transformers.utils.logging import disable_progress_bar

disable_progress_bar()  # disable progress bar for model loading

_UNSET = object()


def load_model_and_tokenizer(
    model_params: DictConfig | dict | None = None,
    *,
    id: str | object = _UNSET,
    tokenizer_id: str | object = _UNSET,
    short_name: str | object = _UNSET,
    developer_name: str | object = _UNSET,
    compile: bool | object = _UNSET,
    dtype: str | object = _UNSET,
    chat_template: str | object = _UNSET,
    trust_remote_code: bool | object = _UNSET,
    attn_implementation: str | object = _UNSET,
    peft_path: str | object = _UNSET,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load a model and tokenizer.

    Why do we need this?
    Technically, AutoModelForCausalLM.from_pretrained() and AutoTokenizer.from_pretrained()
    should handle what we're doing here, but there are many models which do not work
    correctly out of the box. Common issues include:
    - model_max_length is not set correctly
    - eos_token_id/pad_token_id/unk_token_id/bos_token_id are not set correctly
    - chat_template is not set correctly
    - etc.

    Usage:
    - If you already have the Hydra config (run_attacks path), pass the DictConfig/dict via ``model_params``.
    - Otherwise, call with keyword args like ``id=...``. If the id is in ``conf/models/models.yaml``, that
      preset supplies defaults and only the fields you set are overridden. If the id is not in the presets,
      we log an info message and fall back to safe defaults (tokenizer_id=id, dtype=None, compile=False,
      trust_remote_code=True, chat_template=None → HF default).

      Example usage:
        model, tokenizer = load_model_and_tokenizer(
            id="meta-llama/Meta-Llama-3.1-8B-Instruct",
            dtype="bfloat16",
            trust_remote_code=True,
        )

        model, tokenizer = load_model_and_tokenizer(
            model_params=hydra_cfg.model_params
        )

    Args:
        model_params: Optional DictConfig/dict with model configuration.
        id: Model identifier (HuggingFace model ID or local path).
        tokenizer_id: Tokenizer identifier (HuggingFace model ID or local path).
        short_name: Short name for the model.
        developer_name: Developer name for the model.
        compile: Whether to compile the model with torch.compile().
        dtype: Data type for model weights (e.g., "float16", "bfloat16", "int8", "int4").
        chat_template: Name of the chat template to use.
        trust_remote_code: Whether to trust remote code when loading the model/tokenizer.
        attn_implementation: Attention implementation to use (model-specific).
        peft_path: Path to the peft weights. Should be None in regular loading.

    Returns:
        A tuple containing the loaded model and tokenizer.
    """
    model_params = load_model_config(
        model_params,
        id=id,
        tokenizer_id=tokenizer_id,
        short_name=short_name,
        developer_name=developer_name,
        compile=compile,
        dtype=dtype,
        chat_template=chat_template,
        trust_remote_code=trust_remote_code,
        attn_implementation=attn_implementation,
        peft_path=peft_path,
    )

    gc.collect()
    torch.cuda.empty_cache()
    if model_params.dtype is None:
        model = AutoModelForCausalLM.from_pretrained(
            model_params.id,
            trust_remote_code=model_params.trust_remote_code,
            low_cpu_mem_usage=True,
            device_map="auto",
        ).eval()
    elif "float" not in model_params.dtype:
        if model_params.dtype == "int4":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        elif model_params.dtype == "int8":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            raise ValueError(f"Unknown dtype {model_params.dtype}")

        model = AutoModelForCausalLM.from_pretrained(
            model_params.id,
            trust_remote_code=model_params.trust_remote_code,
            quantization_config=quantization_config,
        ).eval()
    else:
        if "gemma-3" in model_params.id:
            model = AutoModelForCausalLM.from_pretrained(
                model_params.id,
                dtype=getattr(torch, model_params.dtype),
                trust_remote_code=model_params.trust_remote_code,
                low_cpu_mem_usage=True,
                attn_implementation="eager",
                device_map="auto",
            ).eval()
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_params.id,
                dtype=getattr(torch, model_params.dtype),
                trust_remote_code=model_params.trust_remote_code,
                low_cpu_mem_usage=True,
                device_map="auto",
            ).eval()
    model: PreTrainedModel
    if model_params.peft_path:
        model = PeftModel.from_pretrained(model, model_params.peft_path)
        model = model.merge_and_unload()
        model.eval()
    if model_params.compile:
        model = torch.compile(model)  # type: ignore


    model.config.short_name = model_params.short_name
    model.config.developer_name = model_params.developer_name

    tokenizer = AutoTokenizer.from_pretrained(
        model_params.tokenizer_id,
        trust_remote_code=model_params.trust_remote_code,
        truncation_side="left",
        padding_side="left",
    )
    # Model-specific tokenizer fixes
    match model_params.tokenizer_id.lower():
        case path if "oasst-sft-6-llama-30b" in path:
            tokenizer.bos_token_id = 1
            tokenizer.unk_token_id = 0
        case path if "guanaco" in path:
            tokenizer.eos_token_id = 2
            tokenizer.unk_token_id = 0
        case path if "vicuna" in path:
            tokenizer.pad_token = tokenizer.eos_token
        case path if "llama-2" in path:
            tokenizer.pad_token = tokenizer.unk_token
            tokenizer.model_max_length = 4096
        case path if "llama2" in path:
            tokenizer.pad_token = tokenizer.unk_token
            tokenizer.model_max_length = 4096
        case path if "meta-llama/meta-llama-3-8b-instruct" in path:
            tokenizer.model_max_length = 8192
            tokenizer.eos_token_id = 128009  # want to use <|eot_id|> instead of <|eos_id|>  (https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct/discussions/4)  # fmt: off
        case path if "grayswanai/llama-3-8b-instruct-rr" in path:
            tokenizer.eos_token_id = 128009  # want to use <|eot_id|> instead of <|eos_id|>  # fmt: off
        case path if "nousresearch/hermes-2-pro-llama-3-8b" in path:
            tokenizer.model_max_length = 8192
        case path if "llm-lat/robust-llama3-8b-instruct" in path:
            tokenizer.model_max_length = 8192
        case path if "openchat/openchat_3.5" in path:
            tokenizer.model_max_length = 8192
        case path if "mistralai/mistral-7b-instruct-v0.3" in path:
            tokenizer.model_max_length = 32768
        case path if "mistralai/ministral-8b-instruct-2410" in path:
            tokenizer.model_max_length = 32768
        case path if "mistralai/mistral-nemo-instruct-2407" in path:
            tokenizer.model_max_length = 32768
        case path if "gemma-2" in path:
            tokenizer.model_max_length = 8192
        case path if "gemma-3" in path:
            # true ctx is 128k but we don't have that much memory
            tokenizer.model_max_length = 32768
        case path if "zephyr" in path:
            tokenizer.model_max_length = 32768
        case path if "openai/gpt-oss" in path:
            tokenizer.model_max_length = 128000
    if tokenizer.model_max_length > 262144:
        raise ValueError(
            f"Model max length {tokenizer.model_max_length} is probably too large."
        )

    if model_params.chat_template is not None:
        tokenizer.chat_template = load_chat_template(model_params.chat_template)

    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    assert tokenizer.pad_token is not None, "pad_token is not set"
    return model, tokenizer


def load_model_config(
    model_params: DictConfig | dict | None = None,
    *,
    id: str | object = _UNSET,
    tokenizer_id: str | object = _UNSET,
    short_name: str | object = _UNSET,
    developer_name: str | object = _UNSET,
    compile: bool | object = _UNSET,
    dtype: str | object = _UNSET,
    chat_template: str | object = _UNSET,
    trust_remote_code: bool | object = _UNSET,
    attn_implementation: str | object = _UNSET,
    peft_path: str | object = _UNSET,
) -> DictConfig:
    """Normalize model configuration from either model_params or explicit kwargs.

    If model_params is provided (DictConfig/dict from Hydra), other kwargs must
    stay unset. Otherwise, kwargs are used; presets are pulled by id when present
    and then overridden by any explicitly provided kwargs.
    """

    # Branch 1: model_params provided (Hydra path)
    if model_params is not None:
        if any(x is not _UNSET for x in [id, tokenizer_id, short_name, developer_name, compile, dtype, chat_template, trust_remote_code, attn_implementation, peft_path]):
            raise ValueError("Pass either model_params or explicit kwargs, not both.")
        if isinstance(model_params, dict):
            model_params = DictConfig(model_params)

        if "peft_path" not in model_params or model_params.peft_path is None:
            model_params.peft_path = None

        return model_params

    # Branch 2: explicit kwargs path
    lookup_name = id if id is not _UNSET else None
    if lookup_name is None:
        raise ValueError("Provide model_params or id= to select a model.")

    # load presets (models.yaml) and lookup id
    presets = _load_presets()
    base = presets.get(lookup_name, {})
    if not base:
        logging.info(
            "Preset '%s' not found in models.yaml; using defaults and HF chat template.",
            lookup_name,
        )

    final_config: dict[str, Any] = {}
    final_config.update(base)

    # user overrides
    overrides = {
        "id": id,
        "tokenizer_id": tokenizer_id,
        "short_name": short_name,
        "developer_name": developer_name,
        "compile": compile,
        "dtype": dtype,
        "chat_template": chat_template,
        "trust_remote_code": trust_remote_code,
        "attn_implementation": attn_implementation,
        "peft_path": peft_path,
    }
    for key, value in overrides.items():
        if value is not _UNSET:
            final_config[key] = value

    # general defaults to fall back on
    defaults = {
        "id": lookup_name,
        "tokenizer_id": lookup_name,
        "short_name": None,
        "developer_name": None,
        "compile": False,
        "dtype": None,
        "chat_template": None,
        "trust_remote_code": True,
        "attn_implementation": None,
        "peft_path": None,
    }
    for key, default in defaults.items():
        if key not in final_config or final_config[key] is None:
            final_config[key] = default
            # DEBUG: check if peft_path is actually set here, it should

    return DictConfig(final_config)


def list_model_presets() -> list[str]:
    """List available model preset names from models.yaml."""

    return sorted(_load_presets().keys())


def _load_presets() -> dict[str, dict[str, Any]]:
    """Load model presets from models.yaml, returning a plain dictionary."""

    path = _default_models_file()
    if not path.exists():
        logging.info("Preset file %s not found; skipping preset lookup.", path)
        return {}
    presets = OmegaConf.load(path)
    data = OmegaConf.to_container(presets, resolve=True)
    if not isinstance(data, dict):
        return {}
    return data or {}


def _default_models_file() -> Path:
    return Path(__file__).parent.parent.parent / "conf" / "models" / "models.yaml"


def load_chat_template(template_name: str) -> str:
    """Load a chat template from the chat_templates directory.

    Also removes indentation and newlines from the template to get the correct format.

    Args:
        template_name: The name of the template to load.

    Returns:
        The chat template as a string.
    """
    # Get project root by going up from current file
    project_root = Path(__file__).parent.parent.parent
    template_path = (
        project_root / "chat_templates" / "chat_templates" / f"{template_name}.jinja"
    )
    return template_path.read_text().replace("    ", "").replace("\n", "")


@lru_cache(maxsize=None)
def num_model_params(id: str) -> int:
    """Get the number of parameters in a model (excluding embeddings)."""
    model = AutoModelForCausalLM.from_pretrained(id, device_map="cpu")
    return model.num_parameters(exclude_embeddings=True)
