import gc
import json
import logging
import os
from dataclasses import asdict, dataclass

import torch
from omegaconf import OmegaConf
from pymongo import MongoClient
from pymongo.synchronous.database import Database
from transformers.utils.logging import disable_progress_bar
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.attacks import AttackResult

disable_progress_bar()  # disable progress bar for model loading


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
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            raise ValueError(f"Unknown dtype {model_params.dtype}")

        model = AutoModelForCausalLM.from_pretrained(
            model_params.id,
            trust_remote_code=model_params.trust_remote_code,
            quantization_config=quantization_config,
        ).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_params.id,
            torch_dtype=getattr(torch, model_params.dtype),
            trust_remote_code=model_params.trust_remote_code,
            low_cpu_mem_usage=True,
            device_map="auto",
        ).eval()
    if model_params.compile:
        model = torch.compile(model)

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
            tokenizer.eos_token_id = 128009  # want to use <|eot_id|> instead of <|eos_id|>  (https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct/discussions/4)
        case path if "grayswanai/llama-3-8b-instruct-rr" in path:
            tokenizer.eos_token_id = 128009  # want to use <|eot_id|> instead of <|eos_id|>
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
        case path if "gemma-3" in path:
            tokenizer.model_max_length = 32768  # true ctx is 128k but we dont have that much memory
        case path if 'zephyr' in path:
            tokenizer.model_max_length = 32768
    if tokenizer.model_max_length > 262144:
        raise ValueError(f"Model max length {tokenizer.model_max_length} is probably too large.")

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


class CompactJSONEncoder(json.JSONEncoder):
    """A JSON Encoder that puts small containers on single lines."""

    CONTAINER_TYPES = (list, tuple, dict)
    """Container datatypes include primitives or other containers."""

    MAX_WIDTH = 7000
    """Maximum width of a container that might be put on a single line."""

    MAX_ITEMS = 1000
    """Maximum number of items in container that might be put on single line."""

    def __init__(self, *args, **kwargs):
        # using this class without indentation is pointless
        if kwargs.get("indent") is None:
            kwargs["indent"] = 4
        super().__init__(*args, **kwargs)
        self.indentation_level = 0

    def encode(self, o):
        """Encode JSON object *o* with respect to single line lists."""
        if isinstance(o, (list, tuple)):
            return self._encode_list(o)
        if isinstance(o, dict):
            return self._encode_object(o)
        return json.dumps(
            o,
            skipkeys=self.skipkeys,
            ensure_ascii=self.ensure_ascii,
            check_circular=self.check_circular,
            allow_nan=self.allow_nan,
            sort_keys=self.sort_keys,
            indent=self.indent,
            separators=(self.item_separator, self.key_separator),
            default=self.default if hasattr(self, "default") else None,
        )

    def _encode_list(self, o):
        if self._put_on_single_line(o):
            return "[" + ", ".join(self.encode(el) for el in o) + "]"
        self.indentation_level += 1
        output = [self.indent_str + self.encode(el) for el in o]
        self.indentation_level -= 1
        return "[\n" + ",\n".join(output) + "\n" + self.indent_str + "]"

    def _encode_object(self, o):
        if not o:
            return "{}"

        # ensure keys are converted to strings
        o = {str(k) if k is not None else "null": v for k, v in o.items()}

        if self.sort_keys:
            o = dict(sorted(o.items(), key=lambda x: x[0]))

        if self._put_on_single_line(o):
            return (
                "{ "
                + ", ".join(
                    f"{json.dumps(k)}: {self.encode(el)}" for k, el in o.items()
                )
                + " }"
            )

        self.indentation_level += 1
        output = [
            f"{self.indent_str}{json.dumps(k)}: {self.encode(v)}" for k, v in o.items()
        ]
        self.indentation_level -= 1

        return "{\n" + ",\n".join(output) + "\n" + self.indent_str + "}"

    def iterencode(self, o, **kwargs):
        """Required to also work with `json.dump`."""
        return self.encode(o)

    def _put_on_single_line(self, o):
        if isinstance(o, list):
            if all(isinstance(el, str) for el in o) and len(str(o)) > 200:
                return False

        # we allow lists of ints to be printed on a single line, no matter how long,
        # otherwise containers are put on multiple lines if they are too long.
        # Usually ints are mainly used for token ids, which become very long for some prompts.
        return (
            self._primitives_only(o)
            and (
                all(isinstance(el, int) for el in o)
                or (len(o) <= self.MAX_ITEMS and len(str(o)) - 2 <= self.MAX_WIDTH)
            )
        )

    def _primitives_only(self, o: list | tuple | dict):
        if isinstance(o, (list, tuple)):
            return not any(isinstance(el, self.CONTAINER_TYPES) for el in o)
        elif isinstance(o, dict):
            return not any(isinstance(el, self.CONTAINER_TYPES) for el in o.values())

    @property
    def indent_str(self) -> str:
        if isinstance(self.indent, int):
            return " " * (self.indentation_level * self.indent)
        elif isinstance(self.indent, str):
            return self.indentation_level * self.indent
        else:
            raise ValueError(
                f"indent must either be of type int or str (is: {type(self.indent)})"
            )


@dataclass
class RunConfig:
    model: str
    dataset: str
    attack: str
    model_params: dict
    dataset_params: dict
    attack_params: dict


def log_attack(run_config: RunConfig, result: AttackResult, save_dir: str, date_time_string: str):
    # Create a structured log message as a JSON object
    OmegaConf.resolve(run_config.attack_params)
    OmegaConf.resolve(run_config.dataset_params)
    OmegaConf.resolve(run_config.model_params)
    log_message = {
        "config": OmegaConf.to_container(OmegaConf.structured(run_config), resolve=True)
    }
    log_message.update(asdict(result))
    # Find the first available run_i.json file
    i = 0
    log_dir = os.path.join(save_dir, date_time_string)
    while os.path.exists(os.path.join(log_dir, str(i), f"run.json")):
        i += 1
    log_file = os.path.join(log_dir, str(i), f"run.json")

    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, "w") as f:
        json.dump(log_message, f, indent=2, cls=CompactJSONEncoder)
    logging.info(f"Attack logged to {log_file}")
    log_config_to_db(run_config, result, log_file)


def get_mongodb_connection() -> Database:
    """Get a MongoDB connection.

    Connects to MongoDB using connection details from a config file or environment
    variables. Falls back to a default localhost connection if not specified.
    """
    user = os.environ.get("MONGODB_USER")
    password = os.environ.get("MONGODB_PASSWORD")
    host = os.environ.get("MONGODB_HOST")
    mongo_uri = os.environ.get("MONGODB_URI", f"mongodb://{user}:{password}@{host}?authSource={user}")
    client = MongoClient(mongo_uri)
    db_name = os.environ.get("MONGODB_DB", user)
    return client[db_name]


def log_config_to_db(run_config, result, log_file):
    db = get_mongodb_connection()
    collection = db.runs

    idx = run_config.dataset_params.idx
    if idx is None:
        idx = [i for i in range(len(result.runs))]
    elif isinstance(idx, int):
        idx = [idx]

    for i in idx:
        run_config.dataset_params.idx = i
        config_data = {
            "config": OmegaConf.to_container(OmegaConf.structured(run_config), resolve=True),
            "log_file": log_file,
            "scored_by": []
        }
        # If a run with the same config already exists, replace it
        collection.replace_one(
            {"config": config_data["config"]},
            config_data,
            upsert=True
        )


def filter_config(run_config: RunConfig, dset_len: int, overwrite: bool = False) -> bool:
    db = get_mongodb_connection()
    collection = db.runs

    OmegaConf.resolve(run_config.attack_params)
    OmegaConf.resolve(run_config.dataset_params)
    OmegaConf.resolve(run_config.model_params)
    original_idx = run_config.dataset_params.idx

    if original_idx is None:
        idx = list(range(dset_len))
    elif isinstance(original_idx, int):
        idx = [original_idx]
    else:
        idx = original_idx

    filtered_idx = []
    for i in idx:
        run_config.dataset_params.idx = i
        config_data = OmegaConf.to_container(OmegaConf.structured(run_config), resolve=True)
        if not overwrite and collection.find_one({"config": config_data}):
            print(f"Skipping {run_config.model} {run_config.dataset} {run_config.attack} idx={i} because it already exists")
            continue
        filtered_idx.append(i)

    if not filtered_idx:
        return None
    run_config.dataset_params.idx = filtered_idx
    return run_config


def delete_orphaned_runs():
    db = get_mongodb_connection()
    items = db.runs.find()
    for item in items:
        log_file = item["log_file"]
        if not os.path.exists(log_file):
            print(f"Log file not found: {log_file}, deleting from database")
            db.runs.delete_one({"_id": item["_id"]})
