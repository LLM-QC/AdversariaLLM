"""
JSON schema validation and filtering utilities.

This module provides utilities for JSON schema validation and filtering
during text generation, including logits filtering and validation.
"""

import copy
import logging
import threading
from typing import Any, Iterator, Mapping, Optional, Sequence, Protocol
from collections import Counter, deque

import json5
import torch
from transformers import PreTrainedTokenizerBase

from ..types import JsonSchema

# Guard to avoid repeatedly patching the lm-format-enforcer decode hook
_DECODE_PATCHED = False

# Lightweight in-memory cache for tokenizer-derived enforcer data
_TOK_DATA_CACHE: dict[tuple, object] = {}
_TOK_DATA_CACHE_LOCK = threading.Lock()


def _tokenizer_signature(tokenizer: PreTrainedTokenizerBase) -> tuple:
    """Build a lightweight signature for a tokenizer to cache enforcer data.
    Avoids hashing full vocabs; focuses on stable identifiers and sizes.
    """
    try:
        name = getattr(tokenizer, "name_or_path", None) or tokenizer.__class__.__name__
    except Exception:
        name = tokenizer.__class__.__name__
    try:
        vocab_size = int(getattr(tokenizer, "vocab_size", 0))
    except Exception:
        vocab_size = 0

    num_added = 0
    try:
        if hasattr(tokenizer, "get_added_vocab"):
            added = tokenizer.get_added_vocab()
            if isinstance(added, dict):
                num_added = len(added)
        else:
            added = getattr(tokenizer, "added_tokens_encoder", None)
            if isinstance(added, dict):
                num_added = len(added)
            elif isinstance(added, (list, tuple, set)):
                num_added = len(added)
    except Exception:
        pass

    try:
        specials_repr = str(getattr(tokenizer, "special_tokens_map", None))
    except Exception:
        specials_repr = None

    return (name, vocab_size, num_added, specials_repr)


class FilterProtocol(Protocol):
    """Protocol defining the interface that all filters must implement.

    This ensures consistent behavior across all filter implementations and
    provides clear documentation of the filter contract.
    """

    def step(self, prev_tokens: torch.LongTensor, logits: torch.Tensor) -> torch.Tensor:
        """Apply filter logic to the current logits.

        Args:
            prev_tokens: Previous tokens for each batch element. Shape: (B,)
                        May contain pad_token_id for batch elements that haven't
                        generated real tokens yet.
            logits: Current logits for all vocabulary tokens. Shape: (B, vocab_size)

        Returns:
            torch.Tensor: Modified logits with unwanted tokens set to -inf.
                         Shape: (B, vocab_size)
        """
        ...



class NullFilter:
    """No-op filter that passes through all tokens unchanged.

    Implements FilterProtocol. Used as default when no filtering is required.
    """
    def __init__(self, **kwargs):
        # Accept any parameters for consistency but ignore them
        pass

    def step(
        self,
        prev_tokens: torch.LongTensor,  # previous tokens for each batch element
        logits: torch.Tensor,  # (B, vocab)
        active_mask: torch.Tensor  # (B, ) bool mask for active batch elements
    ) -> torch.Tensor:
        return logits  # no masking


class RepetitionFilter:
    """Filter that prevents repetitive token sequences.

    Implements FilterProtocol. Forbids generating any set of tokens of size 'repeat_set_size_max'
    for more than `max_repeats` times in a row. It forbids these tokens for the next 'penalty_duration' tokens.
    So for repeat_set_size_max=2 and max_repeats=3,
    "Rainbow Butter Rainbow Butter Rainbow Butter" would forbid 'Butter' and 'Rainbow' for the next 'penalty_duration' tokens.
    This is achieved by saving the last 'max_repeats * repeat_set_size_max' tokens in a buffer and checking if in the last
    n * max_repeats tokens there are n non-unique tokens that appear at least 'max_repeats' times.
    """
    def __init__(
        self,
        batch_size: int,
        tokenizer: PreTrainedTokenizerBase,
        repeat_set_size_max: int = 3,
        max_repeats: int = 5,
        penalty_duration: int = 20,
        force_eos: bool = False,
        **kwargs
    ):
        self.repeat_set_size_max = repeat_set_size_max
        self.max_repeats = max_repeats
        self.penalty_duration = penalty_duration
        self.force_eos = force_eos
        self.eos_id = tokenizer.eos_token_id
        self.PAD = tokenizer.pad_token_id

        # Keep a rolling buffer per batch with bounded length
        self.buffer_maxlen = self.repeat_set_size_max * self.max_repeats
        self.token_buffer = [deque(maxlen=self.buffer_maxlen) for _ in range(batch_size)]
        # Map: token_id -> remaining penalty steps
        self.penalty_map = [dict() for _ in range(batch_size)]

        if self.force_eos and self.eos_id is None:
            raise ValueError("If force_eos is True, tokenizer must have a valid eos_token_id.")

    def step(self, prev_tokens: torch.LongTensor, logits: torch.Tensor, active_mask: torch.Tensor) -> torch.Tensor:
        B, V = logits.shape

        # 1 update histories + decrement penalties from *previous* step
        for i in range(B):
            if not active_mask[i]:
                continue
            tid = prev_tokens[i].item()
            # if tid == self.PAD:
            #     continue

            # add token to buffer (deque auto-discards when full)
            self.token_buffer[i].append(tid)

            # decrement penalties and remove expired ones
            if self.penalty_map[i]:
                pm = self.penalty_map[i]
                # keep entries with counter > 1, decrementing by 1
                self.penalty_map[i] = {tok: cnt - 1 for tok, cnt in pm.items() if cnt > 1}

        # 2 build masked logits
        for i in range(B):
            # find all tokens that have been repeated too often in the buffer
            new_forbidden_tokens = set()
            buf = self.token_buffer[i]
            buf_len = len(buf)

            # look at different scopes
            if buf_len:
                # Convert once for slicing semantics
                buf_list = list(buf)
                for num_repeat_candidates in range(1, self.repeat_set_size_max + 1):
                    range_end = num_repeat_candidates * self.repeat_set_size_max
                    if buf_len < range_end:
                        continue
                    window = buf_list[-range_end:]
                    candidates = set(window)
                    if len(candidates) == num_repeat_candidates:
                        new_forbidden_tokens.update(candidates)

            # add to penalty list
            if new_forbidden_tokens:
                pm = self.penalty_map[i]
                for token_id in new_forbidden_tokens:
                    # If already penalized, keep the max remaining duration
                    prev = pm.get(token_id, 0)
                    if prev < self.penalty_duration:
                        pm[token_id] = self.penalty_duration

            currently_forbidden_tokens = list(self.penalty_map[i].keys())

            if currently_forbidden_tokens and self.force_eos:
                logits[i, :] = float("-inf")
                logits[i, self.eos_id] = 1.0 # allow only eos
            else:
                if currently_forbidden_tokens:
                    logits[i, currently_forbidden_tokens] = float("-inf")

        return logits

    def get_duplicates_over_max(self, xs, max):
        cnt = Counter(xs)                 # O(n)
        dupes = {k: v for k, v in cnt.items() if v > max}
        return dupes                 # dupes: only duplicates; cnt: all counts


class JSONFilter:
    """JSON schema enforcement filter.

    Implements FilterProtocol. Logits-filter helper that:
      1. enforces JSON schema with lm-format-enforcer
      2. suppresses leading whitespace
      3. limits runs of whitespace-only tokens to `max_ws_run`
      4. optionally filters tokens to only allow safe characters (when white_list_chars=True)

    How to create schemata:
    # =============================================================================
    # -----------------------------------------------------------------------------
    # Think of a Schema as “JSON that describes another JSON.”  The **bare minimum**
    # looks like this:
    #     {
    #       "type": "object",
    #       "properties": {
    #         ... one entry per field in your real JSON ...
    #       }
    #     }
    # For *each* nested level you add, include a `"type"` so the validator knows
    # what to expect (`"object"` for dicts, `"array"` for lists, `"string"`, `"number"`,
    # `"integer"`, `"boolean"`, `"null"`).  Below are the options you’ll reach for
    # 90 % of the time.
    #
    # ┌──────────────────────────────  TOP-LEVEL OBJECT  ─────────────────────────┐
    # | "type": "object"                                                           |
    # | "properties": { <field-name> : <schema>, … }                               |
    # | "required":   ["fieldA", "fieldB"]            # ← list mandatory keys     |
    # | "additionalProperties": false                 # ← forbid unknown keys     |
    # └────────────────────────────────────────────────────────────────────────────┘
    # ┌───────────────────────────────  ARRAYS / LISTS  ──────────────────────────┐
    # | "type": "array",                                                          |
    # | "items":   <schema of ONE element>,                                       |
    # | "minItems": 1,    "maxItems": 10                                          |
    # |                                                                           |
    # | • For a TUPEL of fixed length use "items": [schema1, schema2, …]          |
    # └────────────────────────────────────────────────────────────────────────────┘
    # ┌────────────────────────────────  STRINGS  ────────────────────────────────┐
    # | { "type": "string", "minLength": 1, "maxLength": 200,                    |
    # |   "pattern": "^[A-Z][a-z]+$",     # regex                                |
    # |   "enum": ["red","green","blue"]  # fixed vocab                          |
    # | }                                                                        |
    # └────────────────────────────────────────────────────────────────────────────┘
    # ┌───────────────────────────────  NUMBERS  ───────────────────────────────┐  # IMPORTANT: minimum and maximum are unfortunately not enforced by lmformatenforcer!
    # | { "type": "number", "minimum": 0, "maximum": 1 }                        |
    # | Use "integer" instead of "number" when you need whole numbers.          |
    # └──────────────────────────────────────────────────────────────────────────┘
    # ┌────────────────────────────  NULLABLE FIELDS  ───────────────────────────┐
    # | { "type": ["string", "null"] }            # string *or* null             |
    # └──────────────────────────────────────────────────────────────────────────┘
    # ┌─────────────────────────────  EITHER/OR (UNIONS)  ───────────────────────┐
    # | { "oneOf": [ schemaA, schemaB ] }                                        |
    # | …or "anyOf"/"allOf" for other logic.                                     |
    # └──────────────────────────────────────────────────────────────────────────┘
    #
    # EXAMPLE – convert real JSON → schema
    # ------------------------------------
    # Real JSON we expect:
    #     {"name":"Ada", "skills":["math","coding"], "age":38}
    #
    # Schema:
    #     {
    #       "type": "object",
    #       "additionalProperties": false,
    #       "properties": {
    #         "name":   { "type":"string",  "minLength":1 },
    #         "skills": {
    #             "type":"array",
    #             "items": { "type":"string" },
    #             "minItems":1
    #         },
    #         "age":    { "type":"integer", "minimum":0 } # minimum is not enforced!
    #       },
    #       "required": ["name","skills","age"]
    #     }
    #
    # Feed this dict to forbid_extras() → wrap for OpenAI → done.
    # =============================================================================
    """

    def __init__(
        self,
        batch_size: int,
        tokenizer: PreTrainedTokenizerBase,
        stop_token_ids: torch.Tensor,
        schema: dict,
        max_ws_run: int = 2,
        white_list_chars: bool = True,
        **kwargs
    ):

        # ── patch the decode fn just-in-time (once per process) ──────
        global _DECODE_PATCHED
        if not _DECODE_PATCHED:
            import lmformatenforcer.integrations.transformers as _tr

            def _decode_override(tokenizer, tokens):
                # preserve every raw space/punct exactly
                # this prevents trailing spaces from being stripped causing parsing errors in the package
                # issue was reported here: https://github.com/noamgat/lm-format-enforcer/issues/166
                decoded = tokenizer.decode(tokens, clean_up_tokenization_spaces=False)
                return decoded  # originally: decoded.rstrip("�"), but this causes the same issue when this character is actually generated

            _tr._decode_function = _decode_override
            _DECODE_PATCHED = True

        from lmformatenforcer import JsonSchemaParser, CharacterLevelParserConfig
        from lmformatenforcer.integrations.transformers import \
            build_token_enforcer_tokenizer_data
        from lmformatenforcer.tokenenforcer import TokenEnforcer
        parser_config = CharacterLevelParserConfig(max_consecutive_whitespaces=max_ws_run)
        # Build or fetch cached tokenizer data for the enforcer
        key = _tokenizer_signature(tokenizer)
        with _TOK_DATA_CACHE_LOCK:
            tok_data = _TOK_DATA_CACHE.get(key)
            if tok_data is None:
                tok_data = build_token_enforcer_tokenizer_data(tokenizer)
                _TOK_DATA_CACHE[key] = tok_data
        # Single enforcer instance; stateless w.r.t. history input
        self.enforcer: TokenEnforcer = TokenEnforcer(tok_data, JsonSchemaParser(schema, parser_config))
        self.token_histories = [[] for _ in range(batch_size)]

        # ── store schema and stop tokens for JSON completeness checking ──
        self.schema = schema

        # ── determine true vocab size (includes added tokens) ─────────────
        base_vocab_size = int(getattr(tokenizer, "vocab_size", 0) or 0)
        resolved_vocab_size = base_vocab_size
        max_token_id: int | None = None
        try:
            vocab_dict = tokenizer.get_vocab()
            if vocab_dict:
                max_token_id = max(int(v) for v in vocab_dict.values())
                resolved_vocab_size = max(resolved_vocab_size, max_token_id + 1)
        except Exception:
            # Some tokenizers don't expose get_vocab or return non-iterables; fall back to base size
            pass
        self.vocab_size = resolved_vocab_size

        # Store both set and boolean mask
        V = self.vocab_size
        self.stop_token_ids = {tid for tid in stop_token_ids.tolist() if 0 <= tid < V}
        self.stop_mask: list[bool] = [False] * V
        for tid in self.stop_token_ids:
            self.stop_mask[tid] = True

        # ── whitespace handling ──────────────────────────────────────────
        # Build whitespace mask via one batch decode over the vocab for speed
        vocab_size = self.vocab_size
        # Use batch_decode to avoid O(V) python decode calls
        try:
            token_texts = tokenizer.batch_decode([[tid] for tid in range(vocab_size)],
                                                 skip_special_tokens=False,
                                                 clean_up_tokenization_spaces=False)
        except TypeError:
            # Some tokenizers may not accept keyword args; fall back safely
            token_texts = tokenizer.batch_decode([[tid] for tid in range(vocab_size)])
        # Keep token texts for cheap heuristics in step
        self.token_texts = token_texts

        self.ws_mask: list[bool] = [txt.strip() == "" for txt in token_texts]
        self.seen_real = [False] * batch_size  # any non-WS token emitted yet?
        self.streak = [0] * batch_size  # current run length of WS tokens
        self.max_ws_run = max_ws_run

        # ── character whitelist handling ─────────────────────────────────
        self.white_list_chars = white_list_chars
        if white_list_chars:
            # Define a comprehensive but safe character whitelist for JSON
            # Includes: letters, digits, punctuation, common symbols, whitespace
            self.allowed_chars = set(
                # Basic ASCII letters and digits
                'abcdefghijklmnopqrstuvwxyz'
                'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
                '0123456789'
                # JSON structural characters
                '{}[]":,\''
                # Common punctuation and symbols
                '!@#$%^&*()_+-=<>?/\\|`~;.'
                '–—…•'
                # Whitespace characters
                ' \t\n\r\f\v'
                # Accented and international characters (common ones)
                'àáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿ'
                'ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞŸ'
                # Common mathematical and currency symbols
                '±×÷°£€¥¢¤§¶©®™'
            )

            # Problematic quote characters that can break JSON parsing
            problematic_quotes = ["“", "”", "‘", "’"]  # curly quotes

            # Get special token IDs that should always be allowed
            special_token_ids = self._get_special_token_ids(tokenizer)

            # Build a boolean mask of allowed-by-characters tokens (or special tokens)
            self.char_ok_mask: list[bool] = [False] * vocab_size
            for tid, token_text in enumerate(token_texts):
                if tid in special_token_ids:
                    self.char_ok_mask[tid] = True
                    continue
                # Check if all characters in the token are in our whitelist
                if all((c in self.allowed_chars) for c in token_text):
                    # Additional filter: exclude tokens containing problematic quote chars
                    if not any(quote in token_text for quote in problematic_quotes) and token_text.count('"') < 2:
                        self.char_ok_mask[tid] = True
        else:
            self.char_ok_mask = [True] * vocab_size

        # Prebuild CPU mask tensors; moved to device on demand in step
        self._ws_mask_t = torch.tensor(self.ws_mask, dtype=torch.bool)
        self._char_ok_mask_t = torch.tensor(self.char_ok_mask, dtype=torch.bool)
        self._stop_mask_t = torch.tensor(self.stop_mask, dtype=torch.bool)

        # Cached device-bound masks (lazily created on first use per device)
        self._mask_device = None  # type: Optional[torch.device]
        self._ws_mask_t_dev = None  # type: Optional[torch.Tensor]
        self._char_ok_mask_t_dev = None  # type: Optional[torch.Tensor]
        self._stop_mask_t_dev = None  # type: Optional[torch.Tensor]

        # misc
        self.tokenizer = tokenizer

    # ------------------------------------------------------------------ #
    def _is_ws(self, tid: int) -> bool:
        # Fast O(1) boolean lookup
        return self.ws_mask[tid] if 0 <= tid < len(self.ws_mask) else False

    # ------------------------------------------------------------------ #
    def _is_json_complete(self, token_history: list[int]) -> bool:
        """
        Check if the current token history forms a complete, parseable JSON.
        Returns True if JSON is complete and parseable, False otherwise.
        """
        if not token_history:
            return False

        try:
            # Decode the token history to text
            text = self.tokenizer.decode(token_history, skip_special_tokens=True)
            # Try to parse as JSON - if successful, JSON is complete
            parsed = _parse_json(text)
            return parsed is not None
        except Exception:
            return False

    # ------------------------------------------------------------------ #
    def _ensure_masks_on_device(self, device: torch.device) -> None:
        """Ensure precomputed masks are resident on the given device.
        Copies once per device and reuses thereafter to avoid per-step .to() overhead.
        """
        # Normalize device representation for comparison
        dev = torch.device(device)
        if self._mask_device is not None and self._mask_device == dev:
            return
        # Move masks to target device; for CPU, these are typically views/no-ops
        self._ws_mask_t_dev = self._ws_mask_t.to(dev, non_blocking=True)
        self._char_ok_mask_t_dev = self._char_ok_mask_t.to(dev, non_blocking=True)
        self._stop_mask_t_dev = self._stop_mask_t.to(dev, non_blocking=True)
        self._mask_device = dev

    # ------------------------------------------------------------------ #
    def _recent_has_closer(self, token_history: list[int], max_lookback: int = 8) -> bool:
        """Cheap heuristic: look back a few tokens for a closing '}' or ']'.
        Handles cases where the last token is whitespace by scanning up to `max_lookback` tokens.
        """
        if not token_history:
            return False
        # Scan recent tokens for any closer character in their decoded text
        for tid in reversed(token_history[-max_lookback:]):
            if 0 <= tid < len(self.token_texts):
                t = self.token_texts[tid]
            else:
                # Fallback decode if out of range
                try:
                    t = self.tokenizer.decode([tid], clean_up_tokenization_spaces=False)
                except Exception:
                    t = ""
            if ("}" in t) or ("]" in t):
                return True
        return False

    # ------------------------------------------------------------------ #
    def step(self, prev_tokens: torch.LongTensor, logits: torch.Tensor, active_mask: torch.Tensor) -> torch.Tensor:
        B, V = logits.shape

        # 1 update histories + whitespace state from *previous* step
        for i in range(B):
            if not active_mask[i]:
                continue
            tid = prev_tokens[i].item()

            self.token_histories[i].append(tid)

            if self._is_ws(tid):
                self.streak[i] += 1
            else:
                self.streak[i] = 0
                self.seen_real[i] = True

        # 2 build masked logits
        masked = torch.full_like(logits, float("-inf"))

        # Ensure mask tensors are on the same device as logits (cached per device)
        self._ensure_masks_on_device(logits.device)
        ws_mask_t = self._ws_mask_t_dev  # type: ignore[assignment]
        char_ok_mask_t = self._char_ok_mask_t_dev  # type: ignore[assignment]
        stop_mask_t = self._stop_mask_t_dev  # type: ignore[assignment]

        for i in range(B):
            if not active_mask[i]:
                # pass through unchanged for inactive rows
                masked[i] = logits[i]
                continue

            allowed = [tid for tid in self.enforcer.get_allowed_tokens(self.token_histories[i]) if 0 <= tid < V]
            if not allowed:
                # No allowed tokens from enforcer; copy original row
                masked[i] = logits[i]
                continue

            allowed_t = torch.as_tensor(allowed, device=logits.device)

            # ── strip disallowed whitespace tokens ─────────────────────
            # Start with all allowed; drop whitespace based on state via vectorized mask
            keep = torch.ones_like(allowed_t, dtype=torch.bool, device=logits.device)
            ws_sel = ws_mask_t[allowed_t]
            if (not self.seen_real[i]) or (self.streak[i] >= self.max_ws_run):
                # Drop whitespace entirely for leading WS or too-long runs
                keep = keep & (~ws_sel)

            # ── apply character whitelist filtering (fast mask) ────────
            if self.white_list_chars:
                ok_sel = char_ok_mask_t[allowed_t] | stop_mask_t[allowed_t]
                keep = keep & ok_sel

            # Materialize candidates
            idx = allowed_t[keep]

            # ── apply JSON completeness check for stop/EOS tokens ──────
            if idx.numel() == 0:
                logging.info("Warning: no candidates remain after JSON/WS/char filtering!")
                idx = allowed_t
            else:
                # ── apply JSON completeness check for stop/EOS tokens ──
                if any(self.stop_mask):
                    stop_sel = stop_mask_t[idx]
                    if torch.any(stop_sel):
                        # Cheap heuristic: only run full check if recent tokens contain a closer
                        likely_ready = self._recent_has_closer(self.token_histories[i])
                        is_complete = False
                        if likely_ready:
                            is_complete = self._is_json_complete(self.token_histories[i])
                        if not is_complete:
                            idx_no_stop = idx[~stop_sel]
                            if idx_no_stop.numel() == 0:
                                logging.warning(
                                    f"JSON incomplete but no valid tokens available for batch {i}. "
                                    f"Token history length: {len(self.token_histories[i])}, "
                                    f"Last 10 tokens: {self.token_histories[i][-10:] if len(self.token_histories[i]) >= 10 else self.token_histories[i]}, "
                                    f"Decoded text preview: {repr(self.tokenizer.decode(self.token_histories[i][-50:], skip_special_tokens=True)[-100:])}, "
                                    f"Allowing stop tokens as fallback."
                                )
                                idx = idx[stop_sel]
                            else:
                                idx = idx_no_stop


            masked[i, idx] = logits[i, idx]

        return masked

    # ------------------------------------------------------------------ #
    def _get_special_token_ids(self, tokenizer: PreTrainedTokenizerBase) -> set[int]:
        """
        Get set of special token IDs that should always be allowed regardless of character content.
        This includes EOS, BOS, UNK, PAD and other important control tokens.

        Important: EOS tokens are especially critical as they allow the model to properly
        terminate generation when the JSON is complete, even if the lm-format-enforcer
        hasn't explicitly marked them as allowed.
        """
        special_ids = set()

        # Common special tokens that most tokenizers have
        special_token_attrs = [
            'eos_token_id', 'bos_token_id', 'unk_token_id', 'pad_token_id',
            'cls_token_id', 'sep_token_id', 'mask_token_id'
        ]

        for attr in special_token_attrs:
            token_id = getattr(tokenizer, attr, None)
            if token_id is not None:
                special_ids.add(token_id)

        # Some tokenizers have additional special tokens in special_tokens_map
        if hasattr(tokenizer, 'special_tokens_map'):
            for token_name, token_value in tokenizer.special_tokens_map.items():
                if isinstance(token_value, str):
                    # Convert token string to ID
                    try:
                        token_ids = tokenizer.encode(token_value, add_special_tokens=False)
                        if len(token_ids) == 1:  # Only single-token special tokens
                            special_ids.add(token_ids[0])
                    except Exception:
                        pass  # Skip if encoding fails

        # Some tokenizers have all_special_ids attribute
        if hasattr(tokenizer, 'all_special_ids'):
            special_ids.update(tokenizer.all_special_ids)

        # Some tokenizers store special tokens in added_tokens_encoder
        if hasattr(tokenizer, 'added_tokens_encoder'):
            added_enc = tokenizer.added_tokens_encoder
            try:
                if isinstance(added_enc, dict):
                    special_ids.update(added_enc.values())
                elif isinstance(added_enc, (list, tuple, set)):
                    special_ids.update(list(added_enc))
            except Exception:
                pass

        # Some tokenizers provide get_added_vocab()
        if hasattr(tokenizer, 'get_added_vocab'):
            try:
                added_vocab = tokenizer.get_added_vocab()
                if isinstance(added_vocab, dict):
                    special_ids.update(added_vocab.values())
            except Exception:
                pass

        return special_ids


def validate_json_strings(json_likes: list[str], schema: JsonSchema, raise_on_error: bool) -> None:
    """
    Validate a list of JSON-like strings against a schema.

    Parameters:
    - json_likes: List of JSON-like strings to validate.
    - schema: JsonSchema object defining the validation rules.

    Raises:
    - SchemaValidationError: If any string does not conform to the schema.
    """
    for json_like in json_likes:
        _validate_json_string(json_like, schema, raise_on_error)


def _validate_json_string(json_like: str, schema: JsonSchema, raise_on_error: bool) -> None:
    json_obj = _parse_json(json_like)
    if json_obj is None:
        logging.info(f"Failed to parse generated text as JSON: {json_like}")
        if raise_on_error:
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
            # if "maximum" in sch and val > sch["maximum"]:
            #     raise SchemaValidationError(f"{path}: {val} > maximum {sch['maximum']}")
            # the enforcer can not enforce max and min

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


class FilterPipeline:
    """Pipeline that applies multiple filters sequentially.

    This class composes multiple filters into a single filter that applies
    them in the specified order. Each filter receives the output of the
    previous filter, allowing for complex filtering logic through composition.
    """

    def __init__(self, filters: list[FilterProtocol]):
        """Initialize the filter pipeline.

        Args:
            filters: List of filter instances that implement FilterProtocol
        """
        self.filters = filters

    def step(self, prev_tokens: torch.LongTensor, logits: torch.Tensor, active_mask: torch.Tensor) -> torch.Tensor:
        """Apply all filters sequentially.

        Args:
            prev_tokens: Previous tokens for each batch element. Shape: (B,)
            logits: Current logits. Shape: (B, vocab_size)


        Returns:
            torch.Tensor: Logits after applying all filters. Shape: (B, vocab_size)
        """
        for filter_instance in self.filters:
            logits = filter_instance.step(prev_tokens, logits, active_mask)
        return logits


# Factory functions for clean filter configuration
def json_filter(schema: dict, validate_output: bool = True, raise_on_error: bool = True) -> dict:
    """Create a JSON schema filter configuration."""
    return {
        "type": "json",
        "schema": schema,
        "validate_output": validate_output,
        "raise_on_error": raise_on_error
    }


def repetition_filter(
    repeat_set_size_max: int = 3,
    max_repeats: int = 5,
    penalty_duration: int = 20,
    force_eos: bool = False
) -> dict:
    """Create a repetition filter configuration."""
    return {
        "type": "repetition",
        "repeat_set_size_max": repeat_set_size_max,
        "max_repeats": max_repeats,
        "penalty_duration": penalty_duration,
        "force_eos": force_eos
    }


def null_filter() -> dict:
    """Create a null filter configuration (no filtering)."""
    return {"type": "null"}


FILTER_REGISTRY = {
    "json": JSONFilter,
    "repetition": RepetitionFilter,
    "null": NullFilter,
}
