import gc
import json
import logging
import os
import time
from dataclasses import asdict

import torch
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftConfig, AutoPeftModelForCausalLM, get_peft_model, PeftModel

from src.attacks import AttackResult


def _load_merge_peft(model_name, peft_name, manual_untie_embeddings=False, dtype=None):
    config = PeftConfig.from_pretrained(peft_name)
    if manual_untie_embeddings:
        # Load base model with untied embeddings to avoid conflicts
        ref_model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, tie_word_embeddings=False, device_map="auto"
        ).eval()

        # Manually tie the embeddings for the base model
        ref_model.lm_head.weight.data = ref_model.model.embed_tokens.weight.data.clone()

        # Load PEFT config and apply to base model
        model_lora = get_peft_model(ref_model, config)
        
        # Load PEFT weights WITHOUT forcing tie_word_embeddings=True
        model = model_lora.from_pretrained(
            ref_model, peft_name, is_trainable=False  # Removed tie_word_embeddings=True
        )

        # Check if LoRA has broken the tie (expected behavior)
        embeddings_tied = torch.equal(
            model.model.lm_head.weight.data, 
            model.model.model.embed_tokens.weight.data
        )
        
        if embeddings_tied:
            logging.warning("Embeddings are still tied after LoRA loading - this might indicate LoRA is not modifying both layers")
        else:
            logging.info("LoRA has successfully broken the tie between embeddings and lm_head (as expected)")
        
        # Merge and unload - this will preserve the LoRA modifications
        model = model.merge_and_unload()  
        return model

    model = AutoPeftModelForCausalLM.from_pretrained(
        peft_name, torch_dtype=dtype, device_map="auto"
    ).eval()
    model = model.merge_and_unload(safe_merge=True)
    return model


def load_model_and_tokenizer(model_params):
    gc.collect()
    torch.cuda.empty_cache()
    if "float" not in model_params.dtype:
        if model_params.dtype == "int4":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        elif model_params.dtype == "int8":
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        else:
            raise ValueError(f"Unknown dtype {model_params.dtype}")

        if "lora_cfg" in model_params.keys():
            if model_params.lora_cfg.manual_untie_embeddings:
                raise NotImplementedError("LoRA + quantization is not supported for Llama3.2 models")
            base_model = AutoModelForCausalLM.from_pretrained(
                model_params.lora_cfg.base_name,
                trust_remote_code=model_params.trust_remote_code,
                quantization_config=quantization_config,
            ).eval()
   
            model = PeftModel.from_pretrained(base_model, model_params.id)
            if model_params.lora_cfg.merge_lora:
                model = model.merge_and_unload()
            
            gc.collect(); torch.cuda.empty_cache()

        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_params.id,
                trust_remote_code=model_params.trust_remote_code,
                quantization_config=quantization_config,
            ).eval()

    else:
        if "lora_cfg" in model_params.keys() and model_params.lora_cfg.merge_lora:
            model = _load_merge_peft(
                model_params.lora_cfg.base_name,
                model_params.id,
                manual_untie_embeddings=model_params.lora_cfg.manual_untie_embeddings,
                dtype=getattr(torch, model_params.dtype)
            ).eval()
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_params.id,
                torch_dtype=getattr(torch, model_params.dtype),
                trust_remote_code=model_params.trust_remote_code,
                low_cpu_mem_usage=True,
                device_map="auto",
            ).eval()


    model.config.short_name = model_params.short_name
    model.config.developer_name = model_params.developer_name
    tokenizer = AutoTokenizer.from_pretrained(
        model_params.tokenizer_id,
        trust_remote_code=model_params.trust_remote_code,
        truncation_side="left",
        padding_side="left"
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
        case path if "meta-llama/meta-llama-3-8b-instruct" in path:
            tokenizer.model_max_length = 8192
        case path if "meta-llama/Llama-3.1-8B-Instruct" in path:
            # tokenizer.model_max_length = 8192
            pass
        case path if "nousresearch/hermes-2-pro-llama-3-8b" in path:
            tokenizer.model_max_length = 8192
        case path if "llm-lat/robust-llama3-8b-instruct" in path:
            tokenizer.model_max_length = 8192
        case path if "openchat/openchat_3.5" in path:
            tokenizer.model_max_length = 8192
        case path if 'mistralai/mistral-7b-instruct-v0.3' in path:
            tokenizer.model_max_length = 32768
        case path if "mistralai/ministral-8b-instruct-2410" in path:
            tokenizer.model_max_length = 32768
        case path if "mistralai/mistral-nemo-instruct-2407" in path:
            tokenizer.model_max_length = 32768
        case path if "gemma-2" in path:
            tokenizer.model_max_length = 8192
        case path if 'zephyr' in path:
            tokenizer.model_max_length = 32768
    if tokenizer.model_max_length > 262144:
        raise ValueError(f"Model max length {tokenizer.model_max_length} is too high")

    if model_params.chat_template is not None:
        tokenizer.chat_template = load_chat_template(model_params.chat_template)

    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def load_chat_template(template_name):
    chat_template_dir = os.path.dirname(os.path.dirname(__file__))
    chat_template = open(os.path.join(chat_template_dir, f"chat_templates/chat_templates/{template_name}.jinja")).read()
    chat_template = chat_template.replace("    ", "").replace("\n", "")
    return chat_template


def free_vram():
    for _ in range(3):
        gc.collect()
        torch.cuda.empty_cache()


def log_attack(run_config, result: AttackResult, log_file: str):
    # Create a structured log message as a JSON object
    OmegaConf.resolve(run_config.config)
    log_message = {
        "config": OmegaConf.to_container(OmegaConf.structured(run_config), resolve=True)
    }
    log_message.update(asdict(result))
    # merge into log file if it exists already
    # try a few times to make sure we dont get contention issues
    for _ in range(3):
        try:
            with open(log_file, "r") as f:
                log_data = json.load(f)
            break
        except FileNotFoundError:
            log_data = []
            break
        except json.decoder.JSONDecodeError:
            time.sleep(1)

    log_data.append(log_message)

    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, "w") as f:
        json.dump(log_data, f, indent=2)
    logging.info(f"Attack logged to {log_file}")
