import json

from src.lm_utils import APITextGen, HFLocalTextGen, GenerationResult
import pytest
from tests.test_json_schema import _get_model_tok


def test_text_generation_interface():
    model, tokenizer = _get_model_tok("mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated")

    hf_gen = HFLocalTextGen(model, tokenizer)
    api_gen = APITextGen("gpt-4o")

    prompts = [
        "What is the capital of France? Answer with a json.",
        "Give me a random number between 1 and 100. Answer with a json: '{\n'answer': '<random_number>'\n}'.",
    ]

    convs = [[{"role": "user", "content": prompt}, {"role": "assistant", "content": ""}] for prompt in prompts]

    json_schema = {
        "type": "object",
        "properties": {"answer": {"type": "string", "description": "The answer to the question."}},
        "required": ["answer"],
    }

    hf_results = hf_gen.generate(convs, max_new_tokens=20, num_return_sequences=2, json_schema=json_schema).gen
    api_results = api_gen.generate(convs, max_new_tokens=20, num_return_sequences=2, json_schema=json_schema).gen

    # check if results are lists of length 2
    assert isinstance(hf_results, list)
    assert len(hf_results) == 2
    assert isinstance(api_results, list)
    assert len(api_results) == 2

    # check if it can be parsed as JSON
    for result in hf_results + api_results:
        for completion in result:
            assert isinstance(completion, str)
            assert completion.strip() != ""
            try:
                parsed = json.loads(completion)
                assert isinstance(parsed, dict)
                assert "answer" in parsed
            except json.JSONDecodeError:
                assert False, f"HF result is not valid JSON: {completion}"


def test_generationresult_accessors():
    res = GenerationResult(gen=[["a0", "a1"], ["b0", "b1"]], input_ids=[[1, 2], [3, 4]])
    assert res.gen0 == ["a0", "b0"]
    assert res["gen"] == res.gen
    with pytest.raises(KeyError):
        _ = res["doesnt_exist"]
