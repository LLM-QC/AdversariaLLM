"""
Model and tokenizer loading utilities.

This module provides functions for loading and configuring language models
and tokenizers with various optimizations and model-specific configurations.
"""

import gc
from functools import lru_cache
from pathlib import Path

import torch
from omegaconf import DictConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from transformers.utils.logging import disable_progress_bar

disable_progress_bar()  # disable progress bar for model loading


def load_model_and_tokenizer(
    model_params: DictConfig | dict,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load a model and tokenizer from a model parameters configuration.

    Why do we need this?
    Technically, AutoModelForCausalLM.from_pretrained() and AutoTokenizer.from_pretrained()
    should handle what we're doing here, but there are many models which do not work
    correctly out of the box. Common issues include:
    - model_max_length is not set correctly
    - eos_token_id/pad_token_id/unk_token_id/bos_token_id are not set correctly
    - chat_template is not set correctly
    - etc.

    Args:
        model_params: A configuration object or dictionary containing model parameters.

    Returns:
        A tuple containing the loaded model and tokenizer.
    """
    if not isinstance(model_params, DictConfig):
        model_params = DictConfig(model_params)

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
