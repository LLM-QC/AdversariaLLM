"""
JSON schema validation and filtering utilities.

This module provides utilities for JSON schema validation and filtering
during text generation, including logits filtering and validation.
"""

import copy
import logging
from typing import Any, Iterator, Mapping, Optional, Sequence

import json5
import torch
from transformers import PreTrainedTokenizerBase

from ..types import JsonSchema


class NullFilter:  # <- used when json_schema=None
    def step(
        self,
        token_histories: list[list[int]],  # token IDs so far, per row
        logits: torch.Tensor,  # (B, vocab)
    ) -> torch.Tensor:
        return logits  # no masking


class JSONFilter:
    """
    Logits-filter helper:
      1. enforces JSON schema with lm-format-enforcer
      2. suppresses leading whitespace
      3. limits runs of whitespace-only tokens to `max_ws_run`
    """

    def __init__(
        self,
        schema: dict,
        tokenizer: PreTrainedTokenizerBase,
        batch_size: int,
        max_ws_run: int = 2,
    ):
        from lmformatenforcer import JsonSchemaParser
        from lmformatenforcer.integrations.transformers import \
            build_token_enforcer_tokenizer_data
        from lmformatenforcer.tokenenforcer import TokenEnforcer

        tok_data = build_token_enforcer_tokenizer_data(tokenizer)
        self.enforcers: list[TokenEnforcer] = [
            TokenEnforcer(tok_data, JsonSchemaParser(schema)) for _ in range(batch_size)
        ]
        self.token_histories = [[] for _ in range(batch_size)]

        # ── whitespace handling ──────────────────────────────────────────
        self.ws_ids: set[int] = {tid for tid in range(tokenizer.vocab_size) if tokenizer.decode([tid]).strip() == ""}
        self.seen_real = [False] * batch_size  # any non-WS token emitted yet?
        self.streak = [0] * batch_size  # current run length of WS tokens
        self.max_ws_run = max_ws_run

        # misc
        self.PAD = tokenizer.pad_token_id
        self.tokenizer = tokenizer

    # ------------------------------------------------------------------ #
    def _is_ws(self, tid: int) -> bool:
        return tid in self.ws_ids

    # ------------------------------------------------------------------ #
    def step(self, prev_tokens: torch.LongTensor, logits: torch.Tensor) -> torch.Tensor:
        B, V = logits.shape

        # 1 update histories + whitespace state from *previous* step
        for i in range(B):
            tid = prev_tokens[i].item()
            if tid == self.PAD:
                continue

            self.token_histories[i].append(tid)

            if self._is_ws(tid):
                self.streak[i] += 1
            else:
                self.streak[i] = 0
                self.seen_real[i] = True

        # 2 build masked logits
        masked = torch.full_like(logits, float("-inf"))

        for i, enf in enumerate(self.enforcers):
            allowed = [tid for tid in enf.get_allowed_tokens(self.token_histories[i]) if 0 <= tid < V]

            # ── strip disallowed whitespace tokens ─────────────────────
            cand = []
            for tid in allowed:
                if not self._is_ws(tid):
                    cand.append(tid)
                    continue

                # pure whitespace token:
                if not self.seen_real[i]:
                    # leading WS -> skip
                    continue
                if self.streak[i] >= self.max_ws_run:
                    # run too long -> skip
                    continue
                cand.append(tid)

            # never leave the model with no options
            if not cand:
                cand = allowed

            masked[i, cand] = logits[i, cand]

        return masked


def validate_json_strings(json_likes: list[str], schema: JsonSchema) -> None:
    """
    Validate a list of JSON-like strings against a schema.

    Parameters:
    - json_likes: List of JSON-like strings to validate.
    - schema: JsonSchema object defining the validation rules.

    Raises:
    - SchemaValidationError: If any string does not conform to the schema.
    """
    for json_like in json_likes:
        _validate_json_string(json_like, schema)


def _validate_json_string(json_like: str, schema: JsonSchema) -> None:
    json_obj = _parse_json(json_like)
    if json_obj is None:
        raise SchemaValidationError(f"Invalid JSON string: {json_like}")
    _validate_json(schema, json_obj)


class SchemaValidationError(RuntimeError):
    """Raised when a value violates the (trimmed-down) schema."""


def _validate_json(schema: Mapping[str, Any], value: Any) -> None:
    """
    Minimal validator for LM Format Enforcer-style schemas.

    Checks:
      • required keys present
      • no unexpected keys unless "additionalProperties": true
      • primitive types: string / integer  (bools rejected as ints)
      • integer "maximum" keyword
      • full recursion into nested objects/arrays

    Raises SchemaValidationError on the first mismatch.
    Returns None on success.
    """

    def _rec(sch: Mapping[str, Any], val: Any, path: str = "$") -> None:
        node_type = sch.get("type")
        is_obj = node_type == "object" or "properties" in sch
        is_arr = node_type == "array"

        # ── objects ──────────────────────────────────────────────────────
        if is_obj:
            if not isinstance(val, dict):
                raise SchemaValidationError(f"{path}: expected object")

            props: Mapping[str, Any] = sch.get("properties", {})
            required: Sequence[str] = sch.get("required", [])
            addl_ok = bool(sch.get("additionalProperties", False))

            for key in required:
                if key not in val:
                    raise SchemaValidationError(f"{path}: missing key '{key}'")

            for key in val:
                if key not in props and not addl_ok:
                    raise SchemaValidationError(f"{path}: unexpected key '{key}'")

            for key in props:
                if key in val:
                    _rec(props[key], val[key], f"{path}.{key}")

        # ── arrays ───────────────────────────────────────────────────────
        elif is_arr:
            if not isinstance(val, list):
                raise SchemaValidationError(f"{path}: expected array")
            item_schema = sch.get("items")
            if item_schema:
                for idx, item in enumerate(val):
                    _rec(item_schema, item, f"{path}[{idx}]")

        # ── primitives ───────────────────────────────────────────────────
        elif node_type == "string":
            if not isinstance(val, str):
                raise SchemaValidationError(f"{path}: expected string")

        elif node_type == "integer":
            if type(val) is not int:  # bools fail
                raise SchemaValidationError(f"{path}: expected integer")
            if "maximum" in sch and val > sch["maximum"]:
                raise SchemaValidationError(f"{path}: {val} > maximum {sch['maximum']}")

        # ── anything else = treated as 'no constraints' ────────────────

    _rec(schema, value)


def _get_json_candidates(text: str) -> Iterator[str]:
    """
    Yield every balanced  {...}  *or*  [...]  block that occurs in `text`,
    while ignoring braces/brackets that live inside quoted strings.

    Works in a single left-to-right pass (O(n)), no regex recursion.
    """
    in_string: str | None = None  # current quote char or None
    escape = False
    stack: list[str] = []  # expected closing symbols
    start = None  # index of the first opening symbol
    match_closer = {"{": "}", "[": "]"}  # open → close

    for i, ch in enumerate(text):
        # ── string handling ────────────────────────────────────────────
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == in_string:
                in_string = None  # string closes
            continue

        if ch in ('"', "'"):
            in_string = ch
            continue

        # ── brace / bracket tracking ──────────────────────────────────
        if ch in match_closer:  # opening { or [
            if not stack:
                start = i
            stack.append(match_closer[ch])

        elif ch in ("]", "}"):  # closing ] or }
            if not stack or ch != stack[-1]:
                # mismatched closer – reset tracking
                stack.clear()
                start = None
                continue
            stack.pop()
            if not stack and start is not None:
                yield text[start : i + 1]  # balanced block found
                start = None  # keep scanning


def _parse_json(output: str) -> Optional[Any]:
    """
    Extract and parse the first valid JSON/JSON5 object embedded in *output*.
    Returns the parsed value, or None if nothing parses.
    """
    for block in _get_json_candidates(output):
        try:
            return json5.loads(block)
        except Exception as e:
            logging.info("Failed to parse JSON candidate:", block)
            logging.info("Error:", e)
            logging.info("Continuing to search for valid JSON candidates...")
            #     # skip to next candidate
            continue

    logging.info("No valid JSON object found in the output.")

    return None  # no valid JSON found


def forbid_extras(schema: dict) -> dict:
    """
    Return a *new* schema in which every object-level node
    has `"additionalProperties": false` unless the user
    already set it explicitly.
    """

    def _walk(node: dict):
        if not isinstance(node, dict):
            return

        # object → add flag + walk its children
        if node.get("type") == "object":
            node.setdefault("additionalProperties", False)
            for key in ("properties", "patternProperties"):
                for child in node.get(key, {}).values():
                    _walk(child)

        # array → walk its item schemas
        if node.get("type") == "array":
            items = node.get("items")
            if isinstance(items, list):
                for child in items:
                    _walk(child)
            else:
                _walk(items)

        # composition keywords (oneOf / anyOf / allOf)
        for key in ("oneOf", "anyOf", "allOf"):
            for child in node.get(key, []):
                _walk(child)

    new_schema = copy.deepcopy(schema)
    _walk(new_schema)
    return new_schema
