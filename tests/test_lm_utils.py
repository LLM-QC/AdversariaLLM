import datetime
import time
import pytest
import torch
import transformers
from transformers import AutoTokenizer
from src.lm_utils import prepare_conversation


@pytest.fixture(autouse=True)
def mock_datetime():
    transformers.utils.chat_template_utils._compile_jinja_template.strftime_now = lambda x: datetime.datetime(2025, 4, 3, 12, 30, 0).strftime(x)


@pytest.fixture
def tokenizer():
    """Fixture providing a tokenizer for testing."""
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    return tokenizer


@pytest.fixture
def basic_conversation():
    """Fixture providing a basic conversation for testing."""
    return [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm good, thank you!"},
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "Paris."},
    ]


def conversation_with_attack(add_spaces: bool, add_prefix: bool = True, add_suffix: bool = True):
    """Function providing a conversation with attack strings for testing."""
    if add_prefix:
        prefix_attack = "x x x x x " if add_spaces else "x x x x x"
    else:
        prefix_attack = ""
    if add_suffix:
        suffix_attack = " x x x x x" if add_spaces else "x x x x x"
    else:
        suffix_attack = ""
    return [
        {"role": "user", "content": prefix_attack + "Hello, how are you?" + suffix_attack},
        {"role": "assistant", "content": "I'm good, thank you!"},
        {"role": "user", "content": prefix_attack + "What is the capital of France?" + suffix_attack},
        {"role": "assistant", "content": "Paris."},
    ]


def test_prepare_conversation_basic(tokenizer, basic_conversation):
    """Test basic conversation preparation without attack."""
    tokens = prepare_conversation(tokenizer, basic_conversation)

    # Should return a list of tuples, one for each user-assistant pair
    assert isinstance(tokens, list)
    assert len(tokens) == 2

    # Each tuple should contain 6 elements (pre, attack, prompt, suffix, post, target)
    for tup in tokens:
        assert len(tup) == 6
        assert all(isinstance(x, torch.Tensor) for x in tup)

    # First conversation pair
    first_pair = tokens[0]
    # Attack tensors should be empty since no attack was provided
    assert first_pair[1].numel() == 0  # attack prefix
    assert first_pair[3].numel() == 0  # attack suffix

    # Second conversation pair
    second_pair = tokens[1]
    assert second_pair[1].numel() == 0  # attack prefix
    assert second_pair[3].numel() == 0  # attack suffix


def test_prepare_conversation_invalid_input(tokenizer):
    """Test that invalid inputs raise appropriate errors."""
    # Test with non-assistant last message
    invalid_conversation = [
        {"role": "user", "content": "Hello"},
        {"role": "user", "content": "Hi"},  # Last message should be assistant
    ]
    with pytest.raises(AssertionError, match="Last message must be assistant message"):
        prepare_conversation(tokenizer, invalid_conversation)

    # Test with empty conversation
    empty_conversation = []
    with pytest.raises(IndexError):
        prepare_conversation(tokenizer, empty_conversation)


# Ground truth token sets for each model
MODEL_GROUND_TRUTH = {
    "mistralai/Mistral-7B-Instruct-v0.3": [
        {
            "pre": torch.tensor([1, 3]),
            "attack_prefix": torch.tensor([2086, 2086, 2086, 2086, 2086]),
            "prompt": torch.tensor([23325, 29493, 1678, 1228, 1136, 29572]),
            "attack_suffix": torch.tensor([2086, 2086, 2086, 2086, 2086]),
            "post": torch.tensor([4]),
            "target": torch.tensor([1083, 29510, 29487, 1947, 29493, 7747, 1136, 29576]),
        },
        {
            "pre": torch.tensor([2, 3]),
            "attack_prefix": torch.tensor([2086, 2086, 2086, 2086, 2086]),
            "prompt": torch.tensor([2592, 1117, 1040, 6333, 1070, 5611, 29572]),
            "attack_suffix": torch.tensor([2086, 2086, 2086, 2086, 2086]),
            "post": torch.tensor([4]),
            "target": torch.tensor([6233, 29491]),
        },
    ],
    "openchat/openchat_3.5": [
        {
            "pre": torch.tensor([1, 420, 6316, 28781, 3198, 3123, 1247, 28747]),
            "attack_prefix": torch.tensor([1318, 1318, 1318, 1318, 1318]),
            "prompt": torch.tensor([22557, 28725, 910, 460, 368, 28804]),
            "attack_suffix": torch.tensor([1318, 1318, 1318, 1318, 1318]),
            "post": torch.tensor([32000, 420, 6316, 28781, 3198, 3123, 21631, 28747]),
            "target": torch.tensor([315, 28742, 28719, 1179, 28725, 6979, 368, 28808]),
        },
        {
            "pre": torch.tensor([32000, 420, 6316, 28781, 3198, 3123, 1247, 28747]),
            "attack_prefix": torch.tensor([1318, 1318, 1318, 1318, 1318]),
            "prompt": torch.tensor([1824, 349, 272, 5565, 302, 4843, 28804]),
            "attack_suffix": torch.tensor([1318, 1318, 1318, 1318, 1318]),
            "post": torch.tensor([32000, 420, 6316, 28781, 3198, 3123, 21631, 28747]),
            "target": torch.tensor([5465, 28723]),
        },
    ],
    "meta-llama/Llama-2-7b-chat-hf": [
        {
            "pre": torch.tensor([1, 518, 25580, 29962]),
            "attack_prefix": torch.tensor([921, 921, 921, 921, 921]),
            "prompt": torch.tensor([15043, 29892, 920, 526, 366, 29973]),
            "attack_suffix": torch.tensor([921, 921, 921, 921, 921]),
            "post": torch.tensor([518, 29914, 25580, 29962, 29871]),
            "target": torch.tensor([306, 29915, 29885, 1781, 29892, 6452, 366, 29991]),
        },
        {
            "pre": torch.tensor([2, 1, 518, 25580, 29962]),
            "attack_prefix": torch.tensor([921, 921, 921, 921, 921]),
            "prompt": torch.tensor([1724, 338, 278, 7483, 310, 3444, 29973]),
            "attack_suffix": torch.tensor([921, 921, 921, 921, 921]),
            "post": torch.tensor([518, 29914, 25580, 29962, 29871]),
            "target": torch.tensor([3681, 29889]),
        },
    ],
    "microsoft/Phi-3-mini-4k-instruct": [
        {
            "pre": torch.tensor([32010]),
            "attack_prefix": torch.tensor([921, 921, 921, 921, 921]),
            "prompt": torch.tensor([15043, 29892, 920, 526, 366, 29973]),
            "attack_suffix": torch.tensor([921, 921, 921, 921, 921]),
            "post": torch.tensor([32007, 32001]),
            "target": torch.tensor([306, 29915, 29885, 1781, 29892, 6452, 366, 29991]),
        },
        {
            "pre": torch.tensor([32007, 32010]),
            "attack_prefix": torch.tensor([921, 921, 921, 921, 921]),
            "prompt": torch.tensor([1724, 338, 278, 7483, 310, 3444, 29973]),
            "attack_suffix": torch.tensor([921, 921, 921, 921, 921]),
            "post": torch.tensor([32007, 32001]),
            "target": torch.tensor([3681, 29889]),
        },
    ],
    "GraySwanAI/Mistral-7B-Instruct-RR": [
        {
            "pre": torch.tensor([1, 733, 16289, 28793]),
            "attack_prefix": torch.tensor([1318, 1318, 1318, 1318, 1318]),
            "prompt": torch.tensor([22557, 28725, 910, 460, 368, 28804]),
            "attack_suffix": torch.tensor([1318, 1318, 1318, 1318, 1318]),
            "post": torch.tensor([733, 28748, 16289, 28793]),
            "target": torch.tensor([315, 28742, 28719, 1179, 28725, 6979, 368, 28808]),
        },
        {
            "pre": torch.tensor([2, 733, 16289, 28793]),
            "attack_prefix": torch.tensor([1318, 1318, 1318, 1318, 1318]),
            "prompt": torch.tensor([1824, 349, 272, 5565, 302, 4843, 28804]),
            "attack_suffix": torch.tensor([1318, 1318, 1318, 1318, 1318]),
            "post": torch.tensor([733, 28748, 16289, 28793]),
            "target": torch.tensor([5465, 28723]),
        },
    ],
    "google/gemma-2-2b-it": [
        {
            "pre": torch.tensor([2, 106, 1645, 108]),
            "attack_prefix": torch.tensor([235297, 1141, 1141, 1141, 1141]),
            "prompt": torch.tensor([4521, 235269, 1368, 708, 692, 235336]),
            "attack_suffix": torch.tensor([235297, 1141, 1141, 1141, 1141]),
            "post": torch.tensor([107, 108, 106, 2516, 108]),
            "target": torch.tensor([235285, 235303, 235262, 1426, 235269, 7593, 692, 235341]),
        },
        {
            "pre": torch.tensor([235248, 108, 107, 108, 106, 1645, 108]),
            "attack_prefix": torch.tensor([235297, 1141, 1141, 1141, 1141]),
            "prompt": torch.tensor([1841, 603, 573, 6037, 576, 6081, 235336]),
            "attack_suffix": torch.tensor([235297, 1141, 1141, 1141, 1141]),
            "post": torch.tensor([107, 108, 106, 2516, 108]),
            "target": torch.tensor([29437, 235265]),
        },
    ],
    "google/gemma-3-1b-it": [
        {
            "pre": torch.tensor([2, 105, 2364, 107]),
            "attack_prefix": torch.tensor([236781, 1123, 1123, 1123, 1123]),
            "prompt": torch.tensor([9259, 236764, 1217, 659, 611, 236881]),
            "attack_suffix": torch.tensor([236781, 1123, 1123, 1123, 1123]),
            "post": torch.tensor([106, 107, 105, 4368, 107]),
            "target": torch.tensor([236777, 236789, 236757, 1535, 236764, 7806, 611, 236888]),
        },
        {
            "pre": torch.tensor([106, 107, 105, 2364, 107]),
            "attack_prefix": torch.tensor([236781, 1123, 1123, 1123, 1123]),
            "prompt": torch.tensor([3689, 563, 506, 5279, 529, 7001, 236881]),
            "attack_suffix": torch.tensor([236781, 1123, 1123, 1123, 1123]),
            "post": torch.tensor([106, 107, 105, 4368, 107]),
            "target": torch.tensor([50429, 236761]),
        },
    ],
    "google/gemma-3-4b-it": [
        {
            "pre": torch.tensor([2, 105, 2364, 107]),
            "attack_prefix": torch.tensor([236781, 1123, 1123, 1123, 1123]),
            "prompt": torch.tensor([9259, 236764, 1217, 659, 611, 236881]),
            "attack_suffix": torch.tensor([236781, 1123, 1123, 1123, 1123]),
            "post": torch.tensor([106, 107, 105, 4368, 107]),
            "target": torch.tensor([236777, 236789, 236757, 1535, 236764, 7806, 611, 236888]),
        },
        {
            "pre": torch.tensor([106, 107, 105, 2364, 107]),
            "attack_prefix": torch.tensor([236781, 1123, 1123, 1123, 1123]),
            "prompt": torch.tensor([3689, 563, 506, 5279, 529, 7001, 236881]),
            "attack_suffix": torch.tensor([236781, 1123, 1123, 1123, 1123]),
            "post": torch.tensor([106, 107, 105, 4368, 107]),
            "target": torch.tensor([50429, 236761]),
        },
    ],
    "google/gemma-3-12b-it": [
        {
            "pre": torch.tensor([2, 105, 2364, 107]),
            "attack_prefix": torch.tensor([236781, 1123, 1123, 1123, 1123]),
            "prompt": torch.tensor([9259, 236764, 1217, 659, 611, 236881]),
            "attack_suffix": torch.tensor([236781, 1123, 1123, 1123, 1123]),
            "post": torch.tensor([106, 107, 105, 4368, 107]),
            "target": torch.tensor([236777, 236789, 236757, 1535, 236764, 7806, 611, 236888]),
        },
        {
            "pre": torch.tensor([106, 107, 105, 2364, 107]),
            "attack_prefix": torch.tensor([236781, 1123, 1123, 1123, 1123]),
            "prompt": torch.tensor([3689, 563, 506, 5279, 529, 7001, 236881]),
            "attack_suffix": torch.tensor([236781, 1123, 1123, 1123, 1123]),
            "post": torch.tensor([106, 107, 105, 4368, 107]),
            "target": torch.tensor([50429, 236761]),
        },
    ],
    "meta-llama/Meta-Llama-3.1-8B-Instruct": [
        {
            "pre": torch.tensor([128000, 128006, 9125, 128007, 271, 38766, 1303, 33025, 2696, 25, 6790, 220, 2366, 18, 198, 15724, 2696, 25, 220, 1627, 10263, 220, 2366, 19, 271, 128009, 128006, 882, 128007, 271]),
            "attack_prefix": torch.tensor([87, 865, 865, 865, 865]),
            "prompt": torch.tensor([9906, 11, 1268, 527, 499, 30]),
            "attack_suffix": torch.tensor([87, 865, 865, 865, 865]),
            "post": torch.tensor([128009, 128006, 78191, 128007, 271]),
            "target": torch.tensor([40, 2846, 1695, 11, 9901, 499, 0]),
        },
        {
            "pre": torch.tensor([128009, 128006, 882, 128007, 271]),
            "attack_prefix": torch.tensor([87, 865, 865, 865, 865]),
            "prompt": torch.tensor([3923, 374, 279, 6864, 315, 9822, 30]),
            "attack_suffix": torch.tensor([87, 865, 865, 865, 865]),
            "post": torch.tensor([128009, 128006, 78191, 128007, 271]),
            "target": torch.tensor([60704, 13]),
        },
    ],
    "meta-llama/Llama-3.2-1B-Instruct": [
        {
            "pre": torch.tensor([128000, 128006, 9125, 128007, 271, 38766, 1303, 33025, 2696, 25, 6790, 220, 2366, 18, 198, 15724, 2696, 25, 220, 2437, 5186, 220, 2366, 20, 271, 128009, 128006, 882, 128007, 271]),
            "attack_prefix": torch.tensor([87, 865, 865, 865, 865]),
            "prompt": torch.tensor([9906, 11, 1268, 527, 499, 30]),
            "attack_suffix": torch.tensor([87, 865, 865, 865, 865]),
            "post": torch.tensor([128009, 128006, 78191, 128007, 271]),
            "target": torch.tensor([40, 2846, 1695, 11, 9901, 499, 0]),
        },
        {
            "pre": torch.tensor([128009, 128006, 882, 128007, 271]),
            "attack_prefix": torch.tensor([87, 865, 865, 865, 865]),
            "prompt": torch.tensor([3923, 374, 279, 6864, 315, 9822, 30]),
            "attack_suffix": torch.tensor([87, 865, 865, 865, 865]),
            "post": torch.tensor([128009, 128006, 78191, 128007, 271]),
            "target": torch.tensor([60704, 13]),
        },
    ],
    "meta-llama/Llama-3.2-3B-Instruct": [
        {
            "pre": torch.tensor([128000, 128006, 9125, 128007, 271, 38766, 1303, 33025, 2696, 25, 6790, 220, 2366, 18, 198, 15724, 2696, 25, 220, 2437, 5186, 220, 2366, 20, 271, 128009, 128006, 882, 128007, 271]),
            "attack_prefix": torch.tensor([87, 865, 865, 865, 865]),
            "prompt": torch.tensor([9906, 11, 1268, 527, 499, 30]),
            "attack_suffix": torch.tensor([87, 865, 865, 865, 865]),
            "post": torch.tensor([128009, 128006, 78191, 128007, 271]),
            "target": torch.tensor([40, 2846, 1695, 11, 9901, 499, 0]),
        },
        {
            "pre": torch.tensor([128009, 128006, 882, 128007, 271]),
            "attack_prefix": torch.tensor([87, 865, 865, 865, 865]),
            "prompt": torch.tensor([3923, 374, 279, 6864, 315, 9822, 30]),
            "attack_suffix": torch.tensor([87, 865, 865, 865, 865]),
            "post": torch.tensor([128009, 128006, 78191, 128007, 271]),
            "target": torch.tensor([60704, 13]),
        },
    ],
    "NousResearch/Hermes-2-Pro-Llama-3-8B": [
        {
            "pre": torch.tensor([128000, 128002, 882, 198]),
            "attack_prefix": torch.tensor([87, 865, 865, 865, 865]),
            "prompt": torch.tensor([9906, 11, 1268, 527, 499, 30]),
            "attack_suffix": torch.tensor([87, 865, 865, 865, 865]),
            "post": torch.tensor([128003, 198, 128002, 78191, 198]),
            "target": torch.tensor([40, 2846, 1695, 11, 9901, 499, 0]),
        },
        {
            "pre": torch.tensor([128003, 198, 128002, 882, 198]),
            "attack_prefix": torch.tensor([87, 865, 865, 865, 865]),
            "prompt": torch.tensor([3923, 374, 279, 6864, 315, 9822, 30]),
            "attack_suffix": torch.tensor([87, 865, 865, 865, 865]),
            "post": torch.tensor([128003, 198, 128002, 78191, 198]),
            "target": torch.tensor([60704, 13]),
        },
    ],
    "meta-llama/Meta-Llama-3-8B-Instruct": [
        {
            "pre": torch.tensor([128000, 128006, 882, 128007, 271]),
            "attack_prefix": torch.tensor([87, 865, 865, 865, 865]),
            "prompt": torch.tensor([9906, 11, 1268, 527, 499, 30]),
            "attack_suffix": torch.tensor([87, 865, 865, 865, 865]),
            "post": torch.tensor([128009, 128006, 78191, 128007, 271]),
            "target": torch.tensor([40, 2846, 1695, 11, 9901, 499, 0]),
        },
        {
            "pre": torch.tensor([128009, 128006, 882, 128007, 271]),
            "attack_prefix": torch.tensor([87, 865, 865, 865, 865]),
            "prompt": torch.tensor([3923, 374, 279, 6864, 315, 9822, 30]),
            "attack_suffix": torch.tensor([87, 865, 865, 865, 865]),
            "post": torch.tensor([128009, 128006, 78191, 128007, 271]),
            "target": torch.tensor([60704, 13]),
        },
    ],
    "LLM-LAT/robust-llama3-8b-instruct": [
        {
            "pre": torch.tensor([128000, 128006, 882, 128007, 271]),
            "attack_prefix": torch.tensor([87, 865, 865, 865, 865]),
            "prompt": torch.tensor([9906, 11, 1268, 527, 499, 30]),
            "attack_suffix": torch.tensor([87, 865, 865, 865, 865]),
            "post": torch.tensor([128009, 128006, 78191, 128007, 271]),
            "target": torch.tensor([40, 2846, 1695, 11, 9901, 499, 0]),
        },
        {
            "pre": torch.tensor([128009, 128006, 882, 128007, 271]),
            "attack_prefix": torch.tensor([87, 865, 865, 865, 865]),
            "prompt": torch.tensor([3923, 374, 279, 6864, 315, 9822, 30]),
            "attack_suffix": torch.tensor([87, 865, 865, 865, 865]),
            "post": torch.tensor([128009, 128006, 78191, 128007, 271]),
            "target": torch.tensor([60704, 13]),
        },
    ],
    "qwen/Qwen2-7B-Instruct": [
        {
            "pre": torch.tensor([151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198]),
            "attack_prefix": torch.tensor([87, 856, 856, 856, 856]),
            "prompt": torch.tensor([9707, 11, 1246, 525, 498, 30]),
            "attack_suffix": torch.tensor([87, 856, 856, 856, 856]),
            "post": torch.tensor([151645, 198, 151644, 77091, 198]),
            "target": torch.tensor([40, 2776, 1661, 11, 9702, 498, 0]),
        },
        {
            "pre": torch.tensor([151645, 198, 151644, 872, 198]),
            "attack_prefix": torch.tensor([87, 856, 856, 856, 856]),
            "prompt": torch.tensor([3838, 374, 279, 6722, 315, 9625, 30]),
            "attack_suffix": torch.tensor([87, 856, 856, 856, 856]),
            "post": torch.tensor([151645, 198, 151644, 77091, 198]),
            "target": torch.tensor([59604, 13]),
        },
    ],
    "HuggingFaceH4/zephyr-7b-beta": [
        {
            "pre": torch.tensor([1, 523, 28766, 1838, 28766, 28767, 13]),
            "attack_prefix": torch.tensor([28744, 1318, 1318, 1318, 1318]),
            "prompt": torch.tensor([16230, 28725, 910, 460, 368, 28804]),
            "attack_suffix": torch.tensor([28744, 1318, 1318, 1318, 1318]),
            "post": torch.tensor([2, 28705, 13, 28789, 28766, 489, 11143, 28766, 28767, 13]),
            "target": torch.tensor([28737, 28742, 28719, 1179, 28725, 6979, 368, 28808]),
        },
        {
            "pre": torch.tensor([2, 28705, 13, 28789, 28766, 1838, 28766, 28767, 13]),
            "attack_prefix": torch.tensor([28744, 1318, 1318, 1318, 1318]),
            "prompt": torch.tensor([3195, 349, 272, 5565, 302, 4843, 28804]),
            "attack_suffix": torch.tensor([28744, 1318, 1318, 1318, 1318]),
            "post": torch.tensor([2, 28705, 13, 28789, 28766, 489, 11143, 28766, 28767, 13]),
            "target": torch.tensor([3916, 278, 28723]),
        },
    ],
    "microsoft/phi-4": [
        {
            "pre": torch.tensor([100264, 882, 100266]),
            "attack_prefix": torch.tensor([87, 865, 865, 865, 865]),
            "prompt": torch.tensor([9906, 11, 1268, 527, 499, 30]),
            "attack_suffix": torch.tensor([87, 865, 865, 865, 865]),
            "post": torch.tensor([100265, 100264, 78191, 100266]),
            "target": torch.tensor([40, 2846, 1695, 11, 9901, 499, 0]),
        },
        {
            "pre": torch.tensor([100265, 100264, 882, 100266]),
            "attack_prefix": torch.tensor([87, 865, 865, 865, 865]),
            "prompt": torch.tensor([3923, 374, 279, 6864, 315, 9822, 30]),
            "attack_suffix": torch.tensor([87, 865, 865, 865, 865]),
            "post": torch.tensor([100265, 100264, 78191, 100266]),
            "target": torch.tensor([60704, 13]),
        },
    ],
    "cais/zephyr_7b_r2d2": [
        {
            "pre": torch.tensor([1, 523, 28766, 1838, 28766, 28767, 13]),
            "attack_prefix": torch.tensor([28744, 1318, 1318, 1318, 1318]),
            "prompt": torch.tensor([16230, 28725, 910, 460, 368, 28804]),
            "attack_suffix": torch.tensor([28744, 1318, 1318, 1318, 1318]),
            "post": torch.tensor([2, 28705, 13, 28789, 28766, 489, 11143, 28766, 28767, 13]),
            "target": torch.tensor([28737, 28742, 28719, 1179, 28725, 6979, 368, 28808]),
        },
        {
            "pre": torch.tensor([2, 28705, 13, 28789, 28766, 1838, 28766, 28767, 13]),
            "attack_prefix": torch.tensor([28744, 1318, 1318, 1318, 1318]),
            "prompt": torch.tensor([3195, 349, 272, 5565, 302, 4843, 28804]),
            "attack_suffix": torch.tensor([28744, 1318, 1318, 1318, 1318]),
            "post": torch.tensor([2, 28705, 13, 28789, 28766, 489, 11143, 28766, 28767, 13]),
            "target": torch.tensor([3916, 278, 28723]),
        },
    ],
    "GraySwanAI/Llama-3-8B-Instruct-RR": [
        {
            "pre": torch.tensor([128000, 128006, 882, 128007, 271]),
            "attack_prefix": torch.tensor([87, 865, 865, 865, 865]),
            "prompt": torch.tensor([9906, 11, 1268, 527, 499, 30]),
            "attack_suffix": torch.tensor([87, 865, 865, 865, 865]),
            "post": torch.tensor([128009, 128006, 78191, 128007, 271]),
            "target": torch.tensor([40, 2846, 1695, 11, 9901, 499, 0]),
        },
        {
            "pre": torch.tensor([128009, 128006, 882, 128007, 271]),
            "attack_prefix": torch.tensor([87, 865, 865, 865, 865]),
            "prompt": torch.tensor([3923, 374, 279, 6864, 315, 9822, 30]),
            "attack_suffix": torch.tensor([87, 865, 865, 865, 865]),
            "post": torch.tensor([128009, 128006, 78191, 128007, 271]),
            "target": torch.tensor([60704, 13]),
        },
    ],
    "mistralai/Mistral-Nemo-Instruct-2407": [
        {
            "pre": torch.tensor([1, 3]),
            "attack_prefix": torch.tensor([1120, 2460, 2460, 2460, 2460]),
            "prompt": torch.tensor([22177, 1044, 2606, 1584, 1636, 1063]),
            "attack_suffix": torch.tensor([1120, 2460, 2460, 2460, 2460]),
            "post": torch.tensor([4]),
            "target": torch.tensor([1073, 4525, 3683, 1044, 15412, 1636, 1033]),
        },
        {
            "pre": torch.tensor([2, 3]),
            "attack_prefix": torch.tensor([1120, 2460, 2460, 2460, 2460]),
            "prompt": torch.tensor([7493, 1395, 1278, 8961, 1307, 5498, 1063]),
            "attack_suffix": torch.tensor([1120, 2460, 2460, 2460, 2460]),
            "post": torch.tensor([4]),
            "target": torch.tensor([42572, 1046]),
        },
    ],
    "mistralai/Ministral-8B-Instruct-2410": [
        {
            "pre": torch.tensor([1, 3]),
            "attack_prefix": torch.tensor([1120, 2460, 2460, 2460, 2460]),
            "prompt": torch.tensor([22177, 1044, 2606, 1584, 1636, 1063]),
            "attack_suffix": torch.tensor([1120, 2460, 2460, 2460, 2460]),
            "post": torch.tensor([4]),
            "target": torch.tensor([1073, 4525, 3683, 1044, 15412, 1636, 1033]),
        },
        {
            "pre": torch.tensor([2, 3]),
            "attack_prefix": torch.tensor([1120, 2460, 2460, 2460, 2460]),
            "prompt": torch.tensor([7493, 1395, 1278, 8961, 1307, 5498, 1063]),
            "attack_suffix": torch.tensor([1120, 2460, 2460, 2460, 2460]),
            "post": torch.tensor([4]),
            "target": torch.tensor([42572, 1046]),
        },
    ],
    "GSAI-ML/LLaDA-8B-Instruct": [
        {
            "pre": torch.tensor([126346, 3840, 126347, 198, 198]),
            "attack_prefix": torch.tensor([87, 1529, 1529, 1529, 1529]),
            "prompt": torch.tensor([14455, 11, 1099, 449, 362, 30]),
            "attack_suffix": torch.tensor([87, 1529, 1529, 1529, 1529]),
            "post": torch.tensor([126348, 126346, 598, 10450, 126347, 198, 198]),
            "target": torch.tensor([40, 3180, 1505, 11, 7949, 362, 0]),
        },
        {
            "pre": torch.tensor([126348, 126346, 3840, 126347, 198, 198]),
            "attack_prefix": torch.tensor([87, 1529, 1529, 1529, 1529]),
            "prompt": torch.tensor([2372, 341, 268, 7706, 300, 11406, 30]),
            "attack_suffix": torch.tensor([87, 1529, 1529, 1529, 1529]),
            "post": torch.tensor([126348, 126346, 598, 10450, 126347, 198, 198]),
            "target": torch.tensor([65926, 13]),
        },
    ],
    "Qwen/Qwen2.5-0.5B-Instruct": [
        {
            "pre": torch.tensor([151644, 8948, 198, 2610, 525, 1207, 16948, 11, 3465, 553, 54364, 14817, 13, 1446, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198]),
            "attack_prefix": torch.tensor([87, 856, 856, 856, 856]),
            "prompt": torch.tensor([9707, 11, 1246, 525, 498, 30]),
            "attack_suffix": torch.tensor([87, 856, 856, 856, 856]),
            "post": torch.tensor([151645, 198, 151644, 77091, 198]),
            "target": torch.tensor([40, 2776, 1661, 11, 9702, 498, 0]),
        },
        {
            "pre": torch.tensor([151645, 198, 151644, 872, 198]),
            "attack_prefix": torch.tensor([87, 856, 856, 856, 856]),
            "prompt": torch.tensor([3838, 374, 279, 6722, 315, 9625, 30]),
            "attack_suffix": torch.tensor([87, 856, 856, 856, 856]),
            "post": torch.tensor([151645, 198, 151644, 77091, 198]),
            "target": torch.tensor([59604, 13]),
        },
    ],
}

# List of models to test
MODELS_WITH_SPACE = [
    "berkeley-nest/Starling-LM-7B-alpha",
    "meta-llama/Llama-2-7b-chat-hf",
    "GraySwanAI/Mistral-7B-Instruct-RR",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "microsoft/Phi-3-mini-4k-instruct",
    "openchat/openchat_3.5",
    "allenai/Llama-3.1-Tulu-3-8B-DPO",   # super weird tokenizer
]

MODELS_NO_SPACE = [
    "cais/zephyr_7b_r2d2",
    "LLM-LAT/robust-llama3-8b-instruct",
    "google/gemma-2-2b-it",
    "google/gemma-3-1b-it",
    "google/gemma-3-12b-it",
    "google/gemma-3-4b-it",
    "GraySwanAI/Llama-3-8B-Instruct-RR",
    "GSAI-ML/LLaDA-8B-Instruct",
    "HuggingFaceH4/zephyr-7b-beta",
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "microsoft/phi-4",
    "mistralai/Ministral-8B-Instruct-2410",
    "mistralai/Mistral-Nemo-Instruct-2407",
    "NousResearch/Hermes-2-Pro-Llama-3-8B",
    "qwen/Qwen2-7B-Instruct",
    "Qwen/Qwen2.5-0.5B-Instruct",
]


@pytest.mark.parametrize("model_id,ground_truth", MODEL_GROUND_TRUTH.items())
def test_prepare_conversation_ground_truth_with_both(model_id, ground_truth, basic_conversation):
    """Test conversation preparation against ground truth token sets."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    conversation_opt = conversation_with_attack(add_spaces=model_id in MODELS_WITH_SPACE, add_prefix=True, add_suffix=True)
    tokens = prepare_conversation(tokenizer, basic_conversation, conversation_opt)

    # Compare each part with ground truth
    for i, (pre, attack_prefix, prompt, attack_suffix, post, target) in enumerate(tokens):
        gt = ground_truth[i]
        assert torch.equal(pre, gt["pre"]), f"PRE mismatch for {model_id}, {pre}, {gt['pre']}"
        assert torch.equal(attack_prefix, gt["attack_prefix"]), f"ATT prefix mismatch for {model_id}, {attack_prefix}, {gt['attack_prefix']}"
        assert torch.equal(prompt, gt["prompt"]), f"PROMPT mismatch for {model_id}, {prompt}, {gt['prompt']}"
        assert torch.equal(attack_suffix, gt["attack_suffix"]), f"ATT suffix mismatch for {model_id}, {attack_suffix}, {gt['attack_suffix']}"
        assert torch.equal(post, gt["post"]), f"POST mismatch for {model_id}, {post}, {gt['post']}"
        assert torch.equal(target, gt["target"]), f"TARGET mismatch for {model_id}, {target}, {gt['target']}"


@pytest.mark.parametrize("model_id,ground_truth", MODEL_GROUND_TRUTH.items())
def test_prepare_conversation_ground_truth_with_suffix(model_id, ground_truth, basic_conversation):
    """Test conversation preparation against ground truth token sets."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    conversation_opt = conversation_with_attack(add_spaces=model_id in MODELS_WITH_SPACE, add_prefix=False, add_suffix=True)
    tokens = prepare_conversation(tokenizer, basic_conversation, conversation_opt)

    # Compare each part with ground truth
    for i, (pre, attack_prefix, prompt, attack_suffix, post, target) in enumerate(tokens):
        gt = ground_truth[i]
        assert torch.equal(pre, gt["pre"]), f"PRE mismatch for {model_id}, {pre}, {gt['pre']}"
        assert torch.equal(attack_prefix, torch.tensor([], dtype=torch.long)), f"ATT prefix mismatch for {model_id}, {attack_prefix}, {torch.tensor([], dtype=torch.long)}"
        assert torch.equal(prompt, gt["prompt"]), f"PROMPT mismatch for {model_id}, {prompt}, {gt['prompt']}"
        assert torch.equal(attack_suffix, gt["attack_suffix"]), f"ATT suffix mismatch for {model_id}, {attack_suffix}, {gt['attack_suffix']}"
        assert torch.equal(post, gt["post"]), f"POST mismatch for {model_id}, {post}, {gt['post']}"
        assert torch.equal(target, gt["target"]), f"TARGET mismatch for {model_id}, {target}, {gt['target']}"


@pytest.mark.parametrize("model_id,ground_truth", MODEL_GROUND_TRUTH.items())
def test_prepare_conversation_ground_truth_with_prefix(model_id, ground_truth, basic_conversation):
    """Test conversation preparation against ground truth token sets."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    conversation_opt = conversation_with_attack(add_spaces=model_id in MODELS_WITH_SPACE, add_prefix=True, add_suffix=False)
    tokens = prepare_conversation(tokenizer, basic_conversation, conversation_opt)

    # Compare each part with ground truth
    for i, (pre, attack_prefix, prompt, attack_suffix, post, target) in enumerate(tokens):
        gt = ground_truth[i]
        assert torch.equal(pre, gt["pre"]), f"PRE mismatch for {model_id}, {pre}, {gt['pre']}"
        assert torch.equal(attack_prefix, gt["attack_prefix"]), f"ATT prefix mismatch for {model_id}, {attack_prefix}, {gt['attack_prefix']}"
        assert torch.equal(prompt, gt["prompt"]), f"PROMPT mismatch for {model_id}, {prompt}, {gt['prompt']}"
        assert torch.equal(attack_suffix, torch.tensor([], dtype=torch.long)), f"ATT suffix mismatch for {model_id}, {attack_suffix}, {torch.tensor([], dtype=torch.long)}"
        assert torch.equal(post, gt["post"]), f"POST mismatch for {model_id}, {post}, {gt['post']}"
        assert torch.equal(target, gt["target"]), f"TARGET mismatch for {model_id}, {target}, {gt['target']}"


@pytest.mark.parametrize("model_id,ground_truth", MODEL_GROUND_TRUTH.items())
def test_prepare_conversation_ground_truth_with_none(model_id, ground_truth, basic_conversation):
    """Test conversation preparation against ground truth token sets."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    conversation_opt = conversation_with_attack(add_spaces=model_id in MODELS_WITH_SPACE, add_prefix=False, add_suffix=False)
    tokens = prepare_conversation(tokenizer, basic_conversation, conversation_opt)

    # Compare each part with ground truth
    for i, (pre, attack_prefix, prompt, attack_suffix, post, target) in enumerate(tokens):
        gt = ground_truth[i]
        assert torch.equal(pre, gt["pre"]), f"PRE mismatch for {model_id}, {pre}, {gt['pre']}"
        assert torch.equal(attack_prefix, torch.tensor([], dtype=torch.long)), f"ATT prefix mismatch for {model_id}, {attack_prefix}, {torch.tensor([], dtype=torch.long)}"
        assert torch.equal(prompt, gt["prompt"]), f"PROMPT mismatch for {model_id}, {prompt}, {gt['prompt']}"
        assert torch.equal(attack_suffix, torch.tensor([], dtype=torch.long)), f"ATT suffix mismatch for {model_id}, {attack_suffix}, {torch.tensor([], dtype=torch.long)}"
        assert torch.equal(post, gt["post"]), f"POST mismatch for {model_id}, {post}, {gt['post']}"
        assert torch.equal(target, gt["target"]), f"TARGET mismatch for {model_id}, {target}, {gt['target']}"


def generate_test_conversations(n: int, add_spaces: bool = False, add_prefix: bool = True, add_suffix: bool = True) -> list[list[dict[str, str]]]:
    """Generate n test conversations with random content."""
    conversations = []
    conversations_with_attack = []
    import random
    import string

    def get_random_string():
        return ''.join(random.choice(string.ascii_letters) for _ in range(random.randint(1, 10)))

    for _ in range(n):
        a, b, c, d = get_random_string(), get_random_string(), get_random_string(), get_random_string()
        conv = [
            {"role": "user", "content": f"Hello, how {a} are you?"},
            {"role": "assistant", "content": f"I'm good, {b} thank you!"},
            {"role": "user", "content": f"What is the capital of {c} ?"},
            {"role": "assistant", "content": f"Par {d} is."},
        ]
        conversations.append(conv)
        prefix = "" if not add_prefix else ("x x x x x " if add_spaces else "x x x x x")
        suffix = "" if not add_suffix else (" x x x x x" if add_spaces else "x x x x x")
        conv_attack = [
            {"role": "user", "content": f"{prefix}Hello, how {a} are you?{suffix}"},
            {"role": "assistant", "content": f"I'm good, {b} thank you!"},
            {"role": "user", "content": f"{prefix}What is the capital of {c} ?{suffix}"},
            {"role": "assistant", "content": f"Par {d} is."},
        ]
        conversations_with_attack.append(conv_attack)
    return conversations, conversations_with_attack


def test_prepare_conversation_performance():
    """Test the performance of prepare_conversation function."""
    # Number of conversations to process
    n_conversations = 1000
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")

    # Generate test conversations
    conversations, conversations_with_attack = generate_test_conversations(n_conversations)

    # Warm up the tokenizer
    for conv, conv_attack in zip(conversations[:10], conversations_with_attack[:10]):
        prepare_conversation(tokenizer, conv, conv_attack)

    # Measure performance
    start_time = time.time()

    for conv, conv_attack in zip(conversations, conversations_with_attack):
        prepare_conversation(tokenizer, conv, conv_attack)

    end_time = time.time()
    total_time = end_time - start_time

    # Calculate metrics
    conversations_per_second = n_conversations / total_time
    avg_time_per_conversation = total_time / n_conversations

    print("\nPerformance Results:")
    print(f"Total conversations processed: {n_conversations}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Conversations per second: {conversations_per_second:.2f}")
    print(f"Average time per conversation: {avg_time_per_conversation*1000:.2f} ms")
    # Add assertions to ensure reasonable performance
    assert conversations_per_second > 50.0, "Should process at least 50 conversation per second"
