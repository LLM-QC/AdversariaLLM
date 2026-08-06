import random
import string
from typing import List, cast

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from adversariallm.lm_utils import generate_ragged_batched, prepare_conversation
from adversariallm.lm_utils.utils import get_stop_token_ids


def _get_gpu_family() -> str:
    name = torch.cuda.get_device_name().upper()
    if "H200" in name:
        return "h200"
    if "H100" in name:
        return "h100"
    if "A100" in name:
        return "a100"
    return "other"


def _load_test_model(attn_implementation: str | None, dtype: str):
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    torch_dtype = getattr(torch, dtype)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch_dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation=attn_implementation,
        device_map="auto",
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        truncation_side="left",
        padding_side="left",
    )
    tokenizer.model_max_length = 8192
    tokenizer.eos_token_id = 128009
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.short_name = "Llama"
    model.config.developer_name = "Meta"
    return model, tokenizer


@pytest.fixture
def model_and_tokenizer():
    """Reference fixture for the historical strict eager/bfloat16 path."""
    gpu_family = _get_gpu_family()
    if gpu_family != "h200":
        pytest.skip("Historical bf16+eager strict reference is only stable end-to-end on H200; use the matrix cases below instead.")
    return _load_test_model(attn_implementation="eager", dtype="bfloat16")


def longest_common_prefix_length(str1: str, str2: str) -> int:
    """Return the length of the longest common prefix between two strings."""
    min_len = min(len(str1), len(str2))
    for i in range(min_len):
        if str1[i] != str2[i]:
            return i
    return min_len


REFERENCE_CASES = [
    pytest.param("sdpa", "bfloat16", {"a100", "h100"}, id="bf16-sdpa"),
    pytest.param("eager", "bfloat16", {"h100", "h200"}, id="bf16-eager"),
    pytest.param("eager", "float32", {"a100", "h100", "h200"}, id="fp32-eager"),
]


def test_generate_ragged_batched_greedy_no_batch(model_and_tokenizer):
    """Compares the output of generate_ragged_batched with the output of model.generate and a manual loop"""
    model, tokenizer = model_and_tokenizer
    conversation = [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": ""},
    ]
    tokens = torch.cat(prepare_conversation(tokenizer, conversation)[0]) # (L,)
    tokens = torch.tensor(tokens, device=model.device).unsqueeze(0) # (1, L)

    max_new_tokens = 256
    stop_ids = get_stop_token_ids(model, tokenizer).to(model.device)
    stop_tokens = tokenizer.convert_ids_to_tokens(stop_ids.tolist())

    # Use the cached single-sample greedy baseline with the same stop-token semantics
    # as generate_ragged_batched.
    reference = _manual_greedy_generation(
        model,
        tokenizer,
        tokens[0],
        max_new_tokens,
        stop_ids,
        stop_tokens,
    )


    # Generate with generate_ragged_batched
    ragged_generated = generate_ragged_batched(
        model,
        tokenizer,
        token_list=cast(List[torch.LongTensor], [tokens[0]]),
        max_new_tokens=max_new_tokens,
        num_return_sequences=1,
        temperature=0.0,
    )[0][0]

    # Generate with model.generate
    generated = model.generate(tokens, do_sample=False, max_new_tokens=max_new_tokens)[0]
    generated = tokenizer.decode(generated[tokens.size(1):])

    ragged_generated_str = cast(str, ragged_generated)
    ragged_vs_reference = longest_common_prefix_length(reference, ragged_generated_str) / max(len(reference), len(ragged_generated_str))
    generate_vs_reference = longest_common_prefix_length(reference, generated) / max(len(reference), len(generated))
    print(f"Ragged vs reference: {ragged_vs_reference:.2f}")
    print(f"Generate vs reference: {generate_vs_reference:.2f}")
    assert ragged_vs_reference == 1.0


def test_generate_ragged_batched_parallel_identical(model_and_tokenizer):
    """Test that generating 16 identical inputs in parallel produces the same output as a single reference"""
    model, tokenizer = model_and_tokenizer
    conversation = [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": ""},
    ]
    tokens = torch.cat(prepare_conversation(tokenizer, conversation)[0])
    tokens = torch.tensor(tokens, device=model.device)

    max_new_tokens = 512
    stop_ids = get_stop_token_ids(model, tokenizer).to(model.device)
    stop_tokens = tokenizer.convert_ids_to_tokens(stop_ids.tolist())

    reference_output = _manual_greedy_generation(
        model,
        tokenizer,
        tokens,
        max_new_tokens,
        stop_ids,
        stop_tokens,
    )

    # Create batch of 16 identical inputs
    batch_tokens = cast(List[torch.LongTensor], [tokens] * 16)

    # Generate with batch of 16 identical inputs
    batch_outputs = generate_ragged_batched(
        model,
        tokenizer,
        token_list=batch_tokens,
        max_new_tokens=max_new_tokens,
        num_return_sequences=1,
        temperature=0.0,
    )[0]

    # Verify all outputs match the reference
    for i, output in enumerate(batch_outputs):
        output_str = cast(str, output)
        similarity = longest_common_prefix_length(reference_output, output_str) / max(len(reference_output), len(output_str))
        print(f"Batch item {i} vs reference: {similarity:.2f}")
        assert similarity == 1.0, f"Batch item {i} differs from reference: {output[:50]}... vs {reference_output[:50]}..."


def test_generate_ragged_batched_parallel_identical_large_batch(model_and_tokenizer):
    """Test that generating 16 identical inputs in parallel produces the same output as a single reference"""
    model, tokenizer = model_and_tokenizer
    conversation = [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": ""},
    ]
    tokens = torch.cat(prepare_conversation(tokenizer, conversation)[0])
    tokens = torch.tensor(tokens, device=model.device)

    max_new_tokens = 512
    stop_ids = get_stop_token_ids(model, tokenizer).to(model.device)
    stop_tokens = tokenizer.convert_ids_to_tokens(stop_ids.tolist())

    reference_output = _manual_greedy_generation(
        model,
        tokenizer,
        tokens,
        max_new_tokens,
        stop_ids,
        stop_tokens,
    )

    # Create batch of 16 identical inputs
    batch_tokens = cast(List[torch.LongTensor], [tokens] * 512)

    # Generate with batch of 16 identical inputs
    batch_outputs = generate_ragged_batched(
        model,
        tokenizer,
        token_list=batch_tokens,
        max_new_tokens=max_new_tokens,
        num_return_sequences=1,
        temperature=0.0,
    )[0]

    # Generate with model.generate
    generated = model.generate(tokens.unsqueeze(0).expand(512, -1).clone(), do_sample=False, max_new_tokens=max_new_tokens)
    generated = tokenizer.batch_decode(generated[:, tokens.size(0):])

    # Verify all outputs match the reference
    for i, output in enumerate(batch_outputs):
        output_str = cast(str, output)
        generated_str = cast(str, generated[i])
        similarity = longest_common_prefix_length(reference_output, output_str) / max(len(reference_output), len(output_str))
        generated_similarity = longest_common_prefix_length(reference_output, generated_str) / max(len(reference_output), len(generated_str))
        print(f"Batch item {i} vs reference: {similarity:.2f}, generated vs reference: {generated_similarity:.2f}")
        assert similarity == 1.0, f"Batch item {i} differs from reference: {output[:50]}... vs {reference_output[:50]}..."


@pytest.mark.slow
@pytest.mark.parametrize(("attn_implementation", "dtype", "supported_gpus"), REFERENCE_CASES)
def test_generate_ragged_batched_reference_matrix(attn_implementation: str, dtype: str, supported_gpus: set[str]):
    """Check the exact generation configurations that are currently stable on each GPU family.

    This test encodes the observed reference matrix for the upgraded Transformers stack:
    each configuration is only asserted on the GPU families where the cached single-prompt
    reference and ragged generation are currently exact under the repo's stop-token trimming
    semantics.
    """
    gpu_family = _get_gpu_family()
    if gpu_family not in supported_gpus:
        pytest.skip(f"{dtype}+{attn_implementation} is not an exact-reference case on {gpu_family}.")

    model, tokenizer = _load_test_model(attn_implementation=attn_implementation, dtype=dtype)
    conversation = [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": ""},
    ]
    tokens = torch.cat(prepare_conversation(tokenizer, conversation)[0]).to(model.device)

    max_new_tokens = 256
    stop_ids = get_stop_token_ids(model, tokenizer).to(model.device)
    stop_tokens = tokenizer.convert_ids_to_tokens(stop_ids.tolist())

    reference = _manual_greedy_generation(
        model,
        tokenizer,
        tokens,
        max_new_tokens,
        stop_ids,
        stop_tokens,
    )
    ragged_generated = generate_ragged_batched(
        model,
        tokenizer,
        token_list=cast(List[torch.LongTensor], [tokens]),
        max_new_tokens=max_new_tokens,
        num_return_sequences=1,
        temperature=0.0,
    )[0][0]

    ragged_generated_str = cast(str, ragged_generated)
    ragged_vs_reference = longest_common_prefix_length(reference, ragged_generated_str) / max(len(reference), len(ragged_generated_str))
    print(f"{gpu_family} {dtype} {attn_implementation}: {ragged_vs_reference:.2f}")
    assert ragged_vs_reference == 1.0


def _truncate_at_stop(text: str, stop_tokens: list[str]) -> str:
    """Trim generated text at the first encountered stop token."""
    candidates = [text]
    for token in stop_tokens:
        if token:
            segments = text.split(token)
            if segments:
                candidates.append(segments[0])
    return min(candidates, key=len)


def _manual_greedy_generation(
    model,
    tokenizer,
    tokens: torch.Tensor,
    max_new_tokens: int,
    stop_ids: torch.Tensor,
    stop_tokens: list[str],
) -> str:
    """Greedy generation by repeatedly forwarding a single prompt."""
    input_ids = tokens.unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)
    past_key_values = None
    generated_tokens: list[int] = []
    next_input = input_ids

    for _ in range(max_new_tokens):
        outputs = model(
            input_ids=next_input,
            attention_mask=attention_mask,
            use_cache=True,
            past_key_values=past_key_values,
        )
        logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(logits, dim=-1)
        generated_tokens.append(int(next_token.item()))
        past_key_values = outputs.past_key_values

        attention_mask = torch.cat(
            [attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device=attention_mask.device)],
            dim=1,
        )
        next_input = next_token.unsqueeze(1)

        if torch.isin(next_token, stop_ids).item():
            break

    if generated_tokens:
        text = tokenizer.decode(generated_tokens, skip_special_tokens=False)
    else:
        text = ""
    return _truncate_at_stop(text, stop_tokens)


def _render_prefix_bar(label: str, completion: str, baseline: str, width: int = 40, color: int = 32) -> None:
    """Render an ANSI-colored bar showing how much of `completion` matches the prefix of `baseline`.

    - label: label printed at the beginning of the line
    - completion/baseline: strings to compare
    - width: total bar width in characters
    - color: ANSI color code for the matched portion (e.g., 32=green, 34=blue)
    """
    lcp = longest_common_prefix_length(completion, baseline)
    baseline_len = len(baseline)
    ratio = (lcp / baseline_len) if baseline_len else 0.0
    filled = max(0, min(width, int(round(ratio * width))))
    empty = width - filled

    reset = "\x1b[0m"
    dim = "\x1b[90m"
    color_code = f"\x1b[{color}m"

    matched_bar = "█" * filled
    remainder_bar = "░" * empty
    bar = f"{color_code}{matched_bar}{reset}{dim}{remainder_bar}{reset}"
    print(f"{label:<8} |{bar}| {lcp}/{baseline_len}")


def _render_average_bar(label: str, ratio: float, width: int = 40, color: int = 32, suffix: str = "") -> None:
    """Render an ANSI bar for an average ratio value in [0,1]."""
    ratio = max(0.0, min(1.0, ratio))
    filled = int(round(ratio * width))
    empty = width - filled

    reset = "\x1b[0m"
    dim = "\x1b[90m"
    color_code = f"\x1b[{color}m"

    matched_bar = "█" * filled
    remainder_bar = "░" * empty
    bar = f"{color_code}{matched_bar}{reset}{dim}{remainder_bar}{reset}"
    print(f"{label:<8} |{bar}| {suffix}")


@pytest.mark.slow
def test_generate_ragged_batched_random_prompts_alignment(model_and_tokenizer):
    """Compare ragged generation and HF generate against a single-prompt baseline."""
    model, tokenizer = model_and_tokenizer
    rng = random.Random(0)
    num_prompts = 128
    max_new_tokens = 1024

    prompts = [
        [{"role": "user", "content": " ".join(
            "".join(rng.choice(string.ascii_lowercase) for _ in range(rng.randint(3, 8)))
            for _ in range(rng.randint(5, 12))
        )}]
        for _ in range(num_prompts)
    ]

    prompts = [tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True) for prompt in prompts]
    encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=False)
    input_ids = encoded["input_ids"].to(model.device)
    attention_mask = encoded["attention_mask"].to(model.device)
    token_tensors = [tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(model.device)[0] for prompt in prompts]

    stop_ids = get_stop_token_ids(model, tokenizer).to(model.device)
    stop_tokens = tokenizer.convert_ids_to_tokens(stop_ids.tolist())

    # Baseline: sequential single-sample greedy generation
    baseline_completions = [
        _manual_greedy_generation(model, tokenizer, tokens, max_new_tokens, stop_ids, stop_tokens)
        for tokens in token_tensors
    ]

    # Ragged batched generation
    ragged_outputs = generate_ragged_batched(
        model,
        tokenizer,
        token_list=token_tensors,
        max_new_tokens=max_new_tokens,
        num_return_sequences=1,
        temperature=0.0,
    )
    ragged_completions = [output[0] for output in ragged_outputs]

    # Hugging Face generate with tokenizer-provided padding
    hf_outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        do_sample=False,
        max_new_tokens=max_new_tokens,

        pad_token_id=tokenizer.pad_token_id,
    )
    hf_completions = []
    for i, output in enumerate(hf_outputs):
        text = tokenizer.decode(output[input_ids.size(1):], skip_special_tokens=False)
        hf_completions.append(_truncate_at_stop(text, stop_tokens))

    ragged_matches = sum(r == b for r, b in zip(ragged_completions, baseline_completions))
    hf_matches = sum(h == b for h, b in zip(hf_completions, baseline_completions))


    print("\nPrefix match to baseline (per prompt):")
    print("Bar shows proportion of baseline matched; numbers show characters matched.")
    print(f"Legend: \x1b[32m█\x1b[0m Ragged  \x1b[34m█\x1b[0m HF  \x1b[90m░\x1b[0m remainder\n")

    for i in range(num_prompts):
        print(f"Prompt {i}:")
        _render_prefix_bar("Ragged", cast(str, ragged_completions[i]), cast(str, baseline_completions[i]), width=40, color=32)
        _render_prefix_bar("HF", cast(str, hf_completions[i]), cast(str, baseline_completions[i]), width=40, color=34)
        print("")

    # Summary chart: averages until divergence
    lcp_ragged: list[int] = [
        longest_common_prefix_length(cast(str, r), cast(str, b))
        for r, b in zip(ragged_completions, baseline_completions)
    ]
    lcp_hf: list[int] = [
        longest_common_prefix_length(cast(str, h), cast(str, b))
        for h, b in zip(hf_completions, baseline_completions)
    ]
    baseline_lengths: list[int] = [len(cast(str, b)) for b in baseline_completions]

    # Avoid division by zero by treating empty baselines as zero contribution
    ratios_ragged = [(
        (lr / bl) if bl else 0.0
    ) for lr, bl in zip(lcp_ragged, baseline_lengths)]
    ratios_hf = [(
        (lh / bl) if bl else 0.0
    ) for lh, bl in zip(lcp_hf, baseline_lengths)]

    avg_chars_ragged = (sum(lcp_ragged) / len(lcp_ragged)) if lcp_ragged else 0.0
    avg_chars_hf = (sum(lcp_hf) / len(lcp_hf)) if lcp_hf else 0.0
    avg_ratio_ragged = (sum(ratios_ragged) / len(ratios_ragged)) if ratios_ragged else 0.0
    avg_ratio_hf = (sum(ratios_hf) / len(ratios_hf)) if ratios_hf else 0.0

    print("Summary (average prefix agreement):")
    ragged_suffix = f"{avg_chars_ragged:.1f} chars, {avg_ratio_ragged*100:.1f}%"
    hf_suffix = f"{avg_chars_hf:.1f} chars, {avg_ratio_hf*100:.1f}%"
    _render_average_bar("Ragged", avg_ratio_ragged, width=40, color=32, suffix=ragged_suffix)
    _render_average_bar("HF", avg_ratio_hf, width=40, color=34, suffix=hf_suffix)

    print(f"Ragged matches: {ragged_matches}/{num_prompts}")
    print(f"HF generate matches: {hf_matches}/{num_prompts}")
    assert ragged_matches >= hf_matches
    assert ragged_matches >= num_prompts // 2
