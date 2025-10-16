"""
Text generation interface and implementations.

This module provides a unified interface for text generation that supports
both local Hugging Face models and API-based generation services.
"""

import inspect
import copy
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, List, Optional, Callable

import torch
from openai import OpenAI
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from ..types import Conversation
from ..io_utils.json_utils import safe_parse_json_responses
from .filters import forbid_extras
from .generation import generate_ragged_batched
from .utils import select_active_subset, update_masked_subset
from .tokenization import prepare_conversation

# Retry policy definitions are kept here to avoid scattered backbone-specific configuration.


@dataclass
class APIRetryOverrides:
    max_attempts: Optional[int] = None
    sleep_seconds: Optional[float] = None
    backoff: Optional[float] = None


@dataclass
class LocalRetryOverrides:
    max_attempts: Optional[int] = None
    temperature_schedule: Optional[List[Optional[float]]] = None # Nones in the list will be set to the temperature argument


@dataclass
class RetryOverrides:
    api: Optional[APIRetryOverrides] = None
    local: Optional[LocalRetryOverrides] = None


@dataclass
class APIRetryPolicy:
    max_attempts: int = 3
    sleep_seconds: float = 1.0
    backoff: float = 1.5

    def merge(self, overrides: Optional[APIRetryOverrides]) -> "APIRetryPolicy":
        if not overrides:
            return APIRetryPolicy(self.max_attempts, self.sleep_seconds, self.backoff)
        max_attempts = overrides.max_attempts if overrides.max_attempts is not None else self.max_attempts
        if max_attempts <= 0:
            max_attempts = 1
        sleep_seconds = overrides.sleep_seconds if overrides.sleep_seconds is not None else self.sleep_seconds
        if sleep_seconds < 0:
            sleep_seconds = 0.0
        backoff = overrides.backoff if overrides.backoff is not None else self.backoff
        if backoff < 1.0:
            backoff = 1.0
        return APIRetryPolicy(max_attempts=max_attempts, sleep_seconds=sleep_seconds, backoff=backoff)


@dataclass
class LocalRetryPolicy:
    temperature_schedule: List[Optional[float]] = field(default_factory=lambda: [None]) # Nones in the list will be set to the temperature argument
    max_attempts: Optional[int] = 1

    def merge(self, overrides: Optional[LocalRetryOverrides]) -> "LocalRetryPolicy":
        schedule = list(self.temperature_schedule)
        max_attempts = self.max_attempts

        if overrides:
            if overrides.temperature_schedule is not None:
                schedule = overrides.temperature_schedule
            if overrides.max_attempts is not None:
                max_attempts = overrides.max_attempts

        if not schedule:
            schedule = [None]

        attempts = max_attempts if max_attempts is not None else len(schedule)
        if attempts is None or attempts <= 0:
            attempts = 1

        if len(schedule) < attempts:
            schedule = schedule + [schedule[-1]] * (attempts - len(schedule))
        else:
            schedule = schedule[:attempts]

        return LocalRetryPolicy(temperature_schedule=schedule, max_attempts=attempts)


@dataclass
class RetryPolicy:
    api: APIRetryPolicy = APIRetryPolicy()
    local: LocalRetryPolicy = LocalRetryPolicy()

    def merge(self, overrides: Optional[RetryOverrides]) -> "RetryPolicy":
        return RetryPolicy(
            api=self.api.merge(overrides.api if overrides else None),
            local=self.local.merge(overrides.local if overrides else None),
        )


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
    filters: Optional[list[dict]] = None

    def to_hf_args(self):
        # Convert CommonGenerateArgs to generate_ragged_batched arguments

        hf_args = {
            "num_return_sequences": self.num_return_sequences,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "filters": self.filters,
        }
        # Remove None values
        hf_args = {k: v for k, v in hf_args.items() if v is not None}

        return hf_args

    def to_api_args(self):
        # Convert CommonGenerateArgs to API arguments
        # Extract JSON filters only, silently ignore other filter types

        schema = None
        if self.filters:
            # Find the first JSON filter in the list
            for filter_config in self.filters:
                if filter_config.get("type") == "json":
                    json_schema = filter_config.get("schema")
                    if json_schema:
                        json_schema = forbid_extras(
                            json_schema
                        )  # ensure no additional properties are allowed, which is required for strict=True
                        schema = {
                            "type": "json_schema",
                            "json_schema": {"name": "json_schema", "strict": True, "schema": json_schema},
                        }
                        break  # Use first JSON filter found

        api_args = {
            "n": self.num_return_sequences,
            "max_completion_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "response_format": schema,
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
        filters: Optional[list[dict]] = None,
    ):
        # no default values are given, as the default values are in the backends and should not be overwritten

        self.num_return_sequences = num_return_sequences
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.filters = filters

    @abstractmethod
    def generate(
        self,
        convs: list[Conversation],
        num_return_sequences: int = None,
        max_new_tokens: int = None,
        temperature: float = None,
        top_p: float = None,
        filters: Optional[list[dict]] = None,
        retry_overrides: Optional[RetryOverrides] = None,
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
        self.retry_policy = RetryPolicy()

    def generate(
        self,
        convs: list[Conversation],
        num_return_sequences: int = None,
        max_new_tokens: int = None,
        temperature: float = None,
        top_p: float = None,
        filters: Optional[list[dict]] = None,
        retry_overrides: Optional[RetryOverrides] = None,
        **kwargs: Any,
    ) -> GenerationResult:
        if len(convs) == 0:
            return GenerationResult(gen=[], input_ids=[])

        effective_policy = self.retry_policy.merge(retry_overrides)
        local_policy = effective_policy.local
        schedule = list(local_policy.temperature_schedule)
        attempts = local_policy.max_attempts if local_policy.max_attempts is not None else len(schedule)

        last_exception: Exception | None = None
        token_list_cache: list[torch.LongTensor] | None = None
        result: GenerationResult | None = None

        for attempt_idx in range(attempts):
            attempt_temperature = schedule[attempt_idx]
            if attempt_temperature is None:
                attempt_temperature = temperature

            common_generate_args = CommonGenerateArgs(
                num_return_sequences=num_return_sequences,
                max_new_tokens=max_new_tokens,
                temperature=attempt_temperature,
                top_p=top_p,
                filters=filters,
            )

            common_generate_args_dict = common_generate_args.to_hf_args()

            params = {**self.default_generate_kwargs, **common_generate_args_dict, **kwargs}

            if token_list_cache is None:
                token_list_cache = self._batch_tokenize(convs)
                token_list_cache = [token.to(self.model.device.index) for token in token_list_cache]

            try:
                output = generate_ragged_batched(
                    self.model,
                    self.tokenizer,
                    token_list=token_list_cache,
                    **params,
                )
                result = GenerationResult(
                    gen=output,
                    input_ids=[tokens.cpu().squeeze(0).tolist() for tokens in token_list_cache],
                )
                break
            except Exception as exc:
                logging.error(
                    "LocalTextGenerator error on attempt %d/%d: %s",
                    attempt_idx + 1,
                    attempts,
                    exc,
                )
                last_exception = exc
                continue
        if result is None:
            raise RuntimeError("LocalTextGenerator failed after retries") from last_exception

        return result

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


    def __init__(
        self,
        model_name: str,
        default_generate_kwargs: Optional[dict[str, Any]] = None,
        *,
        max_retries: int = 3,
        retry_sleep: float = 1.0,
        retry_backoff: float = 1.5,
    ):
        self.max_retries = max(1, max_retries)
        self.retry_sleep = max(retry_sleep, 0.0)
        self.retry_backoff = max(retry_backoff, 1.0)
        self.clients = {}
        self._initialize_clients()

        self.model_name = model_name
        self.client = self._get_client(model_name)

        self.default_generate_kwargs = default_generate_kwargs or {}
        self.retry_policy = RetryPolicy()

    def generate(
        self,
        convs: list[Conversation],
        num_return_sequences: int = None,
        max_new_tokens: int = None,
        temperature: float = None,
        top_p: float = None,
        filters: Optional[list[dict]] = None,
        retry_overrides: Optional[RetryOverrides] = None,
        **kwargs: Any,
    ) -> GenerationResult:
        if len(convs) == 0:
            return GenerationResult(gen=[], input_ids=[])

        effective_policy = self.retry_policy.merge(retry_overrides)
        api_policy = effective_policy.api
        max_retries = api_policy.max_attempts
        retry_sleep = api_policy.sleep_seconds
        retry_backoff = api_policy.backoff

        function_args = {
            "num_return_sequences": num_return_sequences,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "filters": filters,
        }

        combined_args = {k: (func_val if func_val is not None else self.default_generate_kwargs.get(k, None)) for k, func_val in function_args.items()}

        common_generate_args = CommonGenerateArgs(**combined_args)

        common_generate_args_dict = common_generate_args.to_api_args()

        params = {**common_generate_args_dict, **kwargs}
        responses: list[list[str]] = []
        for conv in convs:
            responses.append(
                self._api_call(
                    conv,
                    max_retries=max_retries,
                    retry_sleep=retry_sleep,
                    retry_backoff=retry_backoff,
                    **params,
                )
            )

        return GenerationResult(gen=responses)

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

    def _api_call(
        self,
        messages: list,
        *,
        max_retries: Optional[int] = None,
        retry_sleep: Optional[float] = None,
        retry_backoff: Optional[float] = None,
        **kwargs,
    ) -> list[str]:
        attempts = max_retries if max_retries is not None else self.max_retries
        sleep_seconds = retry_sleep if retry_sleep is not None else self.retry_sleep
        backoff = retry_backoff if retry_backoff is not None else self.retry_backoff

        for attempt in range(1, attempts + 1):
            try:
                completion = self.client.chat.completions.create(model=self.model_name, messages=messages, **kwargs)
                resp = [completion.choices[i].message.content for i in range(len(completion.choices))]
                return resp
            except Exception as e:
                logging.warning(
                    "API_CALL Error on attempt %d/%d for model %s: %s", attempt, attempts, self.model_name, e
                )
                if attempt == attempts:
                    logging.error("Exhausted retries for model %s; returning empty response.", self.model_name)
                    break
                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)
                    sleep_seconds *= backoff
        return []


def generate_from_prompts(model: TextGenerator, prompts: list[str], **kwargs) -> list[str]:
    """Convert prompts to messages and generate responses."""
    messages = [[{"role": "user", "content": prompt}] for prompt in prompts]
    return model.generate(messages, **kwargs).gen0


def generate_with_conv(
    model: TextGenerator, convs: list[Conversation], queries: list[str], **kwargs
) -> tuple[list[str], list[Conversation]]:
    """Generate text and append to conversations for both local and API generators."""
    convs = copy.deepcopy(convs)

    # Add user queries to converstations
    for conv, query in zip(convs, queries):
        conv.append({"role": "user", "content": query})

    # Generate responses
    res = model.generate(convs, **kwargs)

    # Append generated responses to converstations
    for conv, gen, input_ids in zip(convs, res.gen0, res.input_ids):
        conv.append({"role": "assistant", "content": gen, "input_ids": input_ids})

    return res.gen0, convs



def parse_json_response_with_regenerate(
    model: TextGenerator, convs: list[Conversation], dummy_result: dict, filters: list[dict] | None = None
):
    """Parse JSON responses with error handling and retry logic."""
    JSON_REGENERATE_PROMPT = """Your previous output could not be parsed as valid JSON. The json5.loads() error message was: {error_message}.
Please reformat your last response into a strict, correct JSON object with the following rules:
- Output only a JSON object, with no extra commentary before or after.
- All keys and string values must be enclosed in double quotes ("), not single quotes.
- Use commas correctly between items in arrays and objects.
- Fix any missing quotes, commas, or structural errors.
- Do not forget any curly or square brackets especially the \"}}\" at the very end!
- Ensure the output is fully valid and can be parsed by standard JSON parsers (e.g., json.loads() in Python).
Only return the corrected JSON."""

    convs = copy.deepcopy(convs)
    jsons, parse_unsuccessful_map, error_messages = safe_parse_json_responses(convs, attempt=1)

    # Try to fix parsing errors by regenerating with error messages
    if any(parse_unsuccessful_map):
        logging.info(f"Regenerating {sum(parse_unsuccessful_map)}/{len(parse_unsuccessful_map)} conversations with JSON parse errors")
        unsuccessful_convs = select_active_subset(convs, parse_unsuccessful_map)
        json_regenerate_prompts = [JSON_REGENERATE_PROMPT.format(error_message=error_message) for error_message in error_messages]

        # generate new responses
        _, new_convs = safe_generate_with_conv(model, unsuccessful_convs, json_regenerate_prompts, filters=filters)

        # try to parse again
        new_jsons, _, _ = safe_parse_json_responses(new_convs, attempt=2, dummy_result=dummy_result)

        # apply results to the full set
        jsons = update_masked_subset(jsons, parse_unsuccessful_map, new_jsons)
        convs = update_masked_subset(convs, parse_unsuccessful_map, new_convs)

    return jsons, convs


def check_json_content(jsons: list[dict], check_json_func: callable, dummy_result: None | dict = None) -> list[bool]:
    check_unsuccessful_map = [False] * len(jsons)
    for i, item in enumerate(jsons):
        try:
            check_json_func(item)
        except Exception as e:
            logging.error(f"Error in checking JSON content: {e} JSON: {item}")
            check_unsuccessful_map[i] = True
            if dummy_result is not None:
                jsons[i] = dummy_result

    return jsons, check_unsuccessful_map

def check_json_content_with_regenerate(
    model: TextGenerator, jsons: list[dict], check_json_func: callable, convs: list[Conversation], regenerate=True, filters: list[dict] | None = None, dummy_result: dict = None
):
    """Check JSON content against validation function with retry logic."""
    JSON_CONTENT_PROMPT = f"""The format of the previous JSON response is incorrect. Please ensure that the JSON passes this check:\n{inspect.getsource(check_json_func)}"""

    convs = copy.deepcopy(convs)

    _, check_unsuccessful_map = check_json_content(jsons, check_json_func)

    if regenerate and any(check_unsuccessful_map):
        logging.info(f"Regenerating {sum(check_unsuccessful_map)}/{len(check_unsuccessful_map)} conversations with JSON content errors")
        unsuccessful_convs = select_active_subset(convs, check_unsuccessful_map)

        # regenerate new responses
        _, new_convs = safe_generate_with_conv(
            model, unsuccessful_convs, [JSON_CONTENT_PROMPT] * sum(check_unsuccessful_map), filters=filters
        )

        # check parse them again
        new_jsons, _, _ = safe_parse_json_responses(new_convs, attempt=1, dummy_result=dummy_result)

        # check the content again, if still not successful, replace with dummy result
        new_jsons, _ = check_json_content(new_jsons, check_json_func, dummy_result=dummy_result)

        # update the full set
        jsons = update_masked_subset(jsons, check_unsuccessful_map, new_jsons)
        convs = update_masked_subset(convs, check_unsuccessful_map, new_convs)

    return jsons, convs


def handle_json_response(
    model: TextGenerator, convs: list[Conversation], check_json_func: callable, dummy_result: dict, filters: list[dict] | None = None
) -> tuple[list[dict], list[Conversation]]:
    """Handle JSON response parsing with validation and error recovery."""

    # try to parse the json and regenerate if it fails
    parsed_jsons, convs = parse_json_response_with_regenerate(model, convs, dummy_result, filters=filters)

    # check the content of the parsed jsons and regenerate if not all content is present
    parsed_jsons, convs = check_json_content_with_regenerate(model, parsed_jsons, check_json_func, convs, regenerate=True, filters=filters, dummy_result=dummy_result)

    return parsed_jsons, convs




def safe_generate_with_conv(
    model: TextGenerator,
    convs: list[Conversation],
    prompts: Optional[list[str]] = None,
    filters: Optional[list[dict]] = None,
    context: str = "text_generation",
    fallback_response: Optional[str] = "",
    generate_kwargs: Optional[dict[str, Any]] = None,
    retry_overrides: Optional[RetryOverrides] = None,
) -> tuple[list[str], list[Conversation]]:
    """Generate text robustly, retrying on errors and falling back to a default response."""

    base_kwargs = dict(generate_kwargs or {})
    overrides = retry_overrides or RetryOverrides()

    attempt_kwargs = dict(base_kwargs)
    if overrides.api or overrides.local:
        attempt_kwargs["retry_overrides"] = overrides

    convs = copy.deepcopy(convs)

    if prompts is not None:
        # Add user queries to conversations
        for conv, query in zip(convs, prompts):
            conv.append({"role": "user", "content": query})
        batch_size = len(prompts)
    else:
        batch_size = len(convs)
        for idx, conv in enumerate(convs):
            if conv and conv[-1].get("role") == "assistant" and conv[-1].get("content"):
                logging.warning(
                    f"No prompt provided for conversation index {idx} in context '{context}', "
                    "last message is a non-empty assistant response."
                )

    # Generate responses
    try:
        res = model.generate(convs, filters=filters, **attempt_kwargs)
        responses = res.gen0
        input_ids = res.input_ids or [None] * len(responses)
    except Exception as exc:
        logging.error(f"Error in {context} generation: {exc}. Prompts: {prompts}, filters: {filters}, kwargs: {attempt_kwargs}")
        logging.info("Falling back to empty responses.")
        responses = [fallback_response] * batch_size
        input_ids = [None] * batch_size

    # Append generated responses to converstations
    for conv, gen, input_ids in zip(convs, responses, input_ids):
        conv.append({"role": "assistant", "content": gen, "input_ids": input_ids})

    return responses, convs



def safe_generate_from_prompts(
    model: TextGenerator,
    prompts: list[str],
    filters: Optional[list[dict]] = None,
    context: str = "text_generation",
    fallback_response: Optional[str] = "",
    generate_kwargs: Optional[dict[str, Any]] = None,
    retry_overrides: Optional[RetryOverrides] = None,
) -> list[str]:
    """Generate text robustly, retrying on errors and falling back to a default response."""

    responses, _ = safe_generate_with_conv(
        model,
        convs=[[] for _ in prompts],  # empty conversations
        prompts=prompts,
        filters=filters,
        context=context,
        fallback_response=fallback_response,
        generate_kwargs=generate_kwargs,
        retry_overrides=retry_overrides,
    )

    return responses


def generate_json(
    model: TextGenerator,
    prompts: Optional[list[str]] = None,
    *,
    convs: Optional[list[Conversation]] = None,
    filters: Optional[list[dict]] = None,
    dummy_result: dict,
    check_json_func: Callable[[dict], None],
    context: str = "json_generation",
    generate_kwargs: Optional[dict[str, Any]] = {},
    retry_overrides: Optional[RetryOverrides] = RetryOverrides(),
) -> tuple[list[dict], list[Conversation]]:
    """Generate JSON outputs robustly, retrying on errors and falling back to dummy data."""

    if prompts is None and convs is None:
        raise ValueError("generate_json requires at least one of prompts or convs.")

    convs_to_use = convs if convs is not None else [[] for _ in prompts]

    # generate responses, which might or might not be valid JSON (the json enforcement is not perfect)
    # the json filter is expected to contain raise_on_error=False, so the generation function should not raise
    # on invalid JSON generation, as we have a custom handling for that, but to avoid crashing a batch, we also catch other exceptions
    _, convs = safe_generate_with_conv(
        model,
        convs_to_use,
        prompts,
        filters=filters,
        context=context,
        fallback_response="",
        generate_kwargs=generate_kwargs,
        retry_overrides=retry_overrides,
    )

    parsed_jsons, convs = handle_json_response(
        model,
        convs,
        check_json_func,
        dummy_result,
        filters=filters,
    )
    return parsed_jsons, convs



def _get_env_variable(var_name: str) -> str | None:
    """Fetch environment variable or return None if not set."""
    return os.getenv(var_name)
