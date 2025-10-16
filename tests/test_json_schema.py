"""
Full JSON-schema test-suite for generate_ragged + JSONFilter.

• 3 schema shapes  (nested object / array-of-objects / freetext)
• 3 padding/cache modes (left/-cache, right/-cache, right/+cache)
• 3 representative models  (Llama-3-8B, Mistral-7B, Gemma-2B)

The suite checks that:
  - output contains *balanced* JSON,
  - every row parses,
  - keys match the schema exactly,
  - primitive types & simple numeric bounds are respected.
"""

import json
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from src.lm_utils import generate_ragged, json_filter, validate_json_strings
from src.lm_utils.filters import _parse_json, _validate_json_string

# ═════════════════════════════════════════════════════════════════════
#  Project paths
# ═════════════════════════════════════════════════════════════════════
ROOT = Path(__file__).resolve().parents[1]  # project root
MODELS_YAML = ROOT / "conf" / "models" / "models.yaml"

# ═════════════════════════════════════════════════════════════════════
#  Model matrix   – Pick any subset your hardware can handle
# ═════════════════════════════════════════════════════════════════════
MODEL_IDS = [
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "google/gemma-2-2b-it",
]

# ═════════════════════════════════════════════════════════════════════
#  Schemas under test
# ═════════════════════════════════════════════════════════════════════
SCHEMAS = {
    "nested": {
        "type": "object",
        "properties": {
            "person": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer", "maximum": 120},
                },
                "required": ["name", "age"],
            }
        },
        "required": ["person"],
    },
    "array": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer", "maximum": 120},
            },
            "required": ["name", "age"],
        },
        "minItems": 1,
        "maxItems": 3,
    },
    "freetext": {
        "type": "object",
        "properties": {
            "description": {"type": "string"},
            "tags": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["description"],
    },
}

# ═════════════════════════════════════════════════════════════════════
#  Padding / cache matrix  (left + cache not implemented)
# ═════════════════════════════════════════════════════════════════════
PADDING_CASES = [
    ("right", False),
    ("right", True),
]


# ═════════════════════════════════════════════════════════════════════
#  Model loader util  (loads & caches per session)
# ═════════════════════════════════════════════════════════════════════
_model_cache: dict[str, tuple] = {}


def _get_model_tok(model_id: str):
    if model_id not in _model_cache:
        from src.io_utils import load_model_and_tokenizer

        cfg = OmegaConf.load(MODELS_YAML)[model_id]
        model, tok = load_model_and_tokenizer(cfg)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        _model_cache[model_id] = (model, tok)
    return _model_cache[model_id]


# ═════════════════════════════════════════════════════════════════════
#  Core generation helper
# ═════════════════════════════════════════════════════════════════════
def _generate(model, tok, schema, padding_side, use_cache, prompt_suffix=""):
    prompt = f"Return ONLY JSON that matches this schema:\n{json.dumps(schema)}\n{prompt_suffix}"
    ids = tok.encode(prompt, add_special_tokens=False)
    out = generate_ragged(
        model=model,
        tokenizer=tok,
        token_list=[torch.tensor(ids)],
        max_new_tokens=128,
        temperature=0.0,
        padding_side=padding_side,
        use_cache=use_cache,
        filters=[json_filter(schema)],
    )[0][0]
    print(f"\n--- JSON output ---\n{out.strip()}")
    _validate_json_string(out, schema, raise_on_error=True)
    frag = _parse_json(out)
    return frag, out


def test_parse_json():
    from src.lm_utils.filters import _parse_json

    assert _parse_json('prefix {"a":1} suffix')['a'] == 1
    assert _parse_json('{"b": 2, "note": "brace } in string"}')['b'] == 2
    assert _parse_json('junk {"x": {"y": 3}} stuff')['x']['y'] == 3
    assert _parse_json('first {"ok":1} second {broken') == {'ok': 1}
    assert _parse_json('[{"name":"Ada","age":33}]')[0]['age'] == 33
    assert _parse_json('pre [ {"n":1} , {"n":2} ] post')[1]['n'] == 2
    assert _parse_json('no objects here') is None


@pytest.mark.parametrize(
    "sample",
    [
        '{"person":{"name":"Ada","age":33}}',
        'noise {"person":{"name":"Bob","age":50,"skills":["C","Rust"]}} trailing',
        '{"person":{"name":"Eve","age":120}}',
        '```json\n{ "person": { "name":"Li", "age":22 } }\n```',
    ],
)
def test_valid_samples(sample):
    from src.lm_utils.filters import _parse_json, _validate_json

    SCHEMA = {
        "type": "object",
        "properties": {
            "person": {
                "type": "object",  # explicit, but could be omitted
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer", "maximum": 120},
                    "skills": {  # optional array of strings
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["name", "age"],
            }
        },
        "required": ["person"],
    }
    obj = _parse_json(sample)
    _validate_json(SCHEMA, obj)  # should NOT raise


# ═════════════════════════════════════════════════════════════════════
#  1) Single-row tests (schema × model × padding/cache)
# ═════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("model_id", MODEL_IDS)
@pytest.mark.parametrize("schema_name", SCHEMAS.keys())
@pytest.mark.parametrize("padding_side,use_cache", PADDING_CASES)
def test_single_row(model_id, schema_name, padding_side, use_cache):
    if padding_side == "left" and use_cache:
        pytest.skip("left+cache branch not implemented")
    model, tok = _get_model_tok(model_id)

    _, preview = _generate(model, tok, SCHEMAS[schema_name], padding_side, use_cache)
    print(f"\n{model_id.split('/')[-1]:<40} | {schema_name:<8} | {padding_side:<5} cache={use_cache}: {preview} ...")


# ═════════════════════════════════════════════════════════════════════
#  2) Mixed-batch test (one per model, right-padding with cache)
# ═════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_mixed_batch(model_id):
    model, tok = _get_model_tok(model_id)
    schema = SCHEMAS["nested"]

    prompts = [
        f"Generate a JSON object according to this schema: {schema}",
    ]
    batch = [torch.tensor(tok.encode(p, add_special_tokens=False)) for p in prompts]


    outs = generate_ragged(
        model=model,
        tokenizer=tok,
        token_list=batch,
        max_new_tokens=96,
        padding_side="right",
        use_cache=True,
        filters=[json_filter(schema)],
    )[0]

    for i, out in enumerate(outs):
        print(f"\nRow {i}: {out.strip()} ...\n")

    validate_json_strings(outs, schema, raise_on_error=True)

    print(f"\n--- Mixed batch for {model_id} ---")
    for i, out in enumerate(outs):
        frag = _parse_json(out) or "<NO JSON>"
        print(f"row {i}: {frag}")


# Test string limit

@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_string_max_length_limit(model_id):
    # Question whose natural answer exceeds 10 characters,
    # but schema restricts the field to maxLength=10
    model, tok = _get_model_tok(model_id)
    short_schema = {
        "type": "object",
        "properties": {
            "answer1": {"type": "string", "maxLength": 10},
            "answer2": {"type": "string", "maxLength": 10},
        },
        "required": ["answer1", "answer2"],
        "additionalProperties": False,
    }
    # prompt asking for a verbose explanation
    frag, out = _generate(
        model,
        tok,
        short_schema,
        padding_side="right",
        use_cache=True,
        # prompt_suffix="Explain photosynthesis in detail. Answer with a json object containing a single string field 'answer'.",
        prompt_suffix="Explain photosynthesis in detail. Answer with two options in a json object with fields 'answer1' and 'answer2'.",
    )

    assert "answer1" in frag
    assert isinstance(frag["answer1"], str)
    assert len(frag["answer1"]) <= 10, (
        f"Expected <=10 chars, got {len(frag['answer1'])}: {frag['answer1']!r}"
    )

    assert "answer2" in frag
    assert isinstance(frag["answer2"], str)
    assert len(frag["answer2"]) <= 10, (
        f"Expected <=10 chars, got {len(frag['answer2'])}: {frag['answer2']!r}"
    )
