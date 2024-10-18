import gc
import json
import logging
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import torch
from dataclasses import asdict
from omegaconf import OmegaConf

from src.attacks import AttackResult

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_and_tokenizer(model_path, config, model_params):
    gc.collect()
    torch.cuda.empty_cache()
    if model_params.dtype != "float16":
        if model_params.dtype == "int4":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16
            )
        elif model_params.dtype == "int8":
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        else:
            raise ValueError(f"Unknown dtype {model_params.dtype}")

        model = AutoModelForCausalLM.from_pretrained(model_path,
                                                     trust_remote_code=True,
                                                     quantization_config=quantization_config,)
    else:
        model = (
            AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=getattr(torch, model_params.dtype),
                trust_remote_code=True,
                attn_implementation=model_params.attn_implementation,
            )
            .to(device)
            .eval()
        )
    if model_params.compile:
        model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)

    model.config.short_name = model_params.short_name
    model.config.developer_name = model_params.developer_name
    tokenizer = AutoTokenizer.from_pretrained(model_params.tokenizer_id, trust_remote_code=True)

    if "oasst-sft-6-llama-30b" in model_path:
        tokenizer.bos_token_id = 1
        tokenizer.unk_token_id = 0
    if "guanaco" in model_path:
        tokenizer.eos_token_id = 2
        tokenizer.unk_token_id = 0
    if "llama-2" in model_path:
        tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = "left"
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def log_attack(run_config, result: AttackResult, log_file: str):
    # Create a structured log message as a JSON object
    log_message = {
        "config": OmegaConf.to_container(
            OmegaConf.structured(run_config), resolve=True
        )
    }
    log_message.update(asdict(result))
    # merge into log file if it exists already
    try:
        with open(log_file, "r") as f:
            log_data = json.load(f)
    except FileNotFoundError:
        log_data = []
    log_data.append(log_message)

    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, "w") as f:
        json.dump(log_data, f, indent=2)
    logging.info(f"Attack logged to {log_file}")
