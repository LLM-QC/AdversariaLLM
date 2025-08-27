"""
Text generation interface and implementations.

This module provides a unified interface for text generation that supports
both local Hugging Face models and API-based generation services.
"""

import copy
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

import torch
from openai import OpenAI
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from ..types import Conversation, JsonSchema
from .generation import generate_ragged_batched
from .json_utils import forbid_extras
from .tokenization import prepare_conversation


@dataclass
class GenerationResult:
    gen: list[list[str]]  # batch_size x n_choices
    input_ids: Optional[list[int]] = None

    @property
    def gen0(self) -> list[str]:
        """Returns the first choice of generation."""
        return [g[0] for g in self.gen]

    def __getitem__(self, k):
        return getattr(self, k)


@dataclass
class CommonGenerateArgs:
    num_return_sequences: Optional[int] = None
    max_new_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    json_schema: Optional[JsonSchema] = None

    def to_hf_args(self):
        # Convert CommonGenerateArgs to generate_ragged_batched arguments

        hf_args = {
            "num_return_sequences": self.num_return_sequences,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "json_schema": self.json_schema,
        }
        # Remove None values
        hf_args = {k: v for k, v in hf_args.items() if v is not None}

        return hf_args

    def to_api_args(self):
        # Convert CommonGenerateArgs to API arguments

        if self.json_schema:
            self.json_schema = forbid_extras(
                self.json_schema
            )  # ensure no additional properties are allowed, which is required for strict=True
            schema = {
                "type": "json_schema",
                "json_schema": {"name": "json_schema", "strict": True, "schema": self.json_schema},
            }

        api_args = {
            "n": self.num_return_sequences,
            "max_completion_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "response_format": schema if self.json_schema else None,
        }
        # Remove None values
        api_args = {k: v for k, v in api_args.items() if v is not None}

        return api_args


class TextGenerator(ABC):
    """
    Base class for text generation.
    This class defines the interface for text generation backends.
    The idea is to be able to use an API solely via this interface, such that you can just switch the backend to a local model and be able to use it like an API.
    This way, you can easily write code for both local and API-based text generation.

    Subclasses must implement the `generate` method and override the `__init__` method.
    """

    def __init__(
        self,
        num_return_sequences: int = None,
        max_new_tokens: int = None,
        temperature: float = None,
        top_p: float = None,
        json_schema: Optional[JsonSchema] = None,
    ):
        # no default values are given, as the default values are in the backends and should not be overwritten

        self.num_return_sequences = num_return_sequences
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.json_schema = json_schema

    @abstractmethod
    def generate(
        self,
        convs: list[Conversation],
        num_return_sequences: int = None,
        max_new_tokens: int = None,
        temperature: float = None,
        top_p: float = None,
        json_schema: Optional[JsonSchema] = None,
        **kwargs: Any,
    ) -> GenerationResult:
        # no default values are given, as the default values are in the backends and should not be overwritten
        raise NotImplementedError("Subclasses must implement the generate method.")


class LocalTextGenerator(TextGenerator):
    """
    A local backend for text generation using Hugging Face transformers based on the `generate_ragged_batched` function.
    This class is essentially a wrapper around the Hugging Face transformers library to provide a consistent interface for text generation.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        default_generate_kwargs: Optional[dict[str, Any]] = None,
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.default_generate_kwargs = default_generate_kwargs or {}

    def generate(
        self,
        convs: list[Conversation],
        num_return_sequences: int = None,
        max_new_tokens: int = None,
        temperature: float = None,
        top_p: float = None,
        json_schema: Optional[JsonSchema] = None,
        **kwargs: Any,
    ) -> GenerationResult:
        if len(convs) == 0:
            return GenerationResult(gen=[[]])

        common_generate_args = CommonGenerateArgs(
            num_return_sequences=num_return_sequences,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            json_schema=json_schema,
        )

        common_generate_args_dict = common_generate_args.to_hf_args()

        params = {**self.default_generate_kwargs, **common_generate_args_dict, **kwargs}

        token_list = self._batch_tokenize(convs)

        token_list = [token.to(self.model.device.index) for token in token_list]

        output = generate_ragged_batched(
            self.model,
            self.tokenizer,
            token_list=token_list,
            **params,
        )

        return GenerationResult(gen=output, input_ids=[tokens.cpu().squeeze(0).tolist() for tokens in token_list])

    def _batch_tokenize(self, convs: list[Conversation]) -> list[torch.LongTensor]:
        token_list = []
        for conv in convs:
            next_tokens = self._tokenize(conv)
            token_list += next_tokens

        return token_list

    def _tokenize(self, conv: Conversation) -> list[torch.LongTensor]:
        conv = copy.deepcopy(conv)
        if conv[-1]["role"] == "user":
            conv.append({"role": "assistant", "content": ""})
        parts_list = prepare_conversation(self.tokenizer, conv)
        parts_list = [torch.cat(parts) for parts in parts_list]
        token_ids = torch.cat(parts_list, dim=0)

        return [token_ids]


class APITextGenerator(TextGenerator):
    """
    API-based text generation using various models. Note that not all models support all parameters. This class is mainly tailored and tested with OpenAI models.

    TODO: Change this to a cleaner and more general API integration, as this one is a minimal implementation inspired by
    https://github.com/AI45Lab/ActorAttack/tree/master focused on the OpenAI API.
    """


    def __init__(self, model_name: str, default_generate_kwargs: Optional[dict[str, Any]] = None):
        self.CALL_SLEEP = 1
        self.clients = {}
        self._initialize_clients()

        self.model_name = model_name
        self.client = self._get_client(model_name)

        self.default_generate_kwargs = default_generate_kwargs or {}

    def generate(
        self,
        convs: list[Conversation],
        num_return_sequences: int = None,
        max_new_tokens: int = None,
        temperature: float = None,
        top_p: float = None,
        json_schema: Optional[JsonSchema] = None,
        **kwargs: Any,
    ) -> GenerationResult:
        if len(convs) == 0:
            return GenerationResult(gen=[[]])

        common_generate_args = CommonGenerateArgs(
            num_return_sequences=num_return_sequences,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            json_schema=json_schema,
        )

        common_generate_args_dict = common_generate_args.to_api_args()

        params = {**self.default_generate_kwargs, **common_generate_args_dict, **kwargs}
        res = []
        for conv in convs:
            res.append(self._api_call(conv, **params))

        return GenerationResult(gen=res)

    def _initialize_clients(self):
        """Dynamically initialize available clients based on environment variables."""

        gpt_api_key = _get_env_variable("GPT_API_KEY")
        gpt_base_url = _get_env_variable("BASE_URL_GPT")
        if gpt_api_key and gpt_base_url:
            self.clients["gpt"] = OpenAI(base_url=gpt_base_url, api_key=gpt_api_key)

        claude_api_key = _get_env_variable("CLAUDE_API_KEY")
        claude_base_url = _get_env_variable("BASE_URL_CLAUDE")
        if claude_api_key and claude_base_url:
            self.clients["claude"] = OpenAI(base_url=claude_base_url, api_key=claude_api_key)

        deepseek_api_key = _get_env_variable("DEEPSEEK_API_KEY")
        deepseek_base_url = _get_env_variable("BASE_URL_DEEPSEEK")
        if deepseek_api_key and deepseek_base_url:
            self.clients["deepseek"] = OpenAI(base_url=deepseek_base_url, api_key=deepseek_api_key)

        deepinfra_api_key = _get_env_variable("DEEPINFRA_API_KEY")
        deepinfra_base_url = _get_env_variable("BASE_URL_DEEPINFRA")
        if deepinfra_api_key and deepinfra_base_url:
            self.clients["deepinfra"] = OpenAI(base_url=deepinfra_base_url, api_key=deepinfra_api_key)

        if not self.clients:
            logging.info("No valid API credentials found. Exiting.")
            raise RuntimeError("No valid API credentials found. Exiting.")

    def _get_client(self, model_name: str) -> OpenAI:
        """Select appropriate client based on the given model name."""
        if "gpt" in model_name or "o1-" in model_name:
            client = self.clients.get("gpt")
        elif "claude" in model_name:
            client = self.clients.get("claude")
        elif "deepseek" in model_name:
            client = self.clients.get("deepseek")
        elif any(keyword in model_name.lower() for keyword in ["llama", "qwen", "mistral", "microsoft"]):
            client = self.clients.get("deepinfra")
        else:
            raise ValueError(f"Unsupported or unknown model name: {model_name}")

        if not client:
            raise ValueError(f"{model_name} client is not available.")
        return client

    def _api_call(self, messages: list, **kwargs) -> list[str]:
        for _ in range(3):
            try:
                completion = self.client.chat.completions.create(model=self.model_name, messages=messages, **kwargs)
                resp = [completion.choices[i].message.content for i in range(len(completion.choices))]
                return resp
            except Exception as e:
                logging.info(f"GPT_CALL Error: {self.model_name}:{e}")
                time.sleep(self.CALL_SLEEP)
                continue
        return []


def _get_env_variable(var_name: str) -> str | None:
    """Fetch environment variable or return None if not set."""
    return os.getenv(var_name)