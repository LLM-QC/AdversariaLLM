"""
Tokenization and conversation handling utilities.

This module provides functions for processing tokens and conversations,
including chat template handling and token preparation for attacks.
"""

import copy
import random
import string
from functools import lru_cache
from typing import Literal

import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase

from ..types import Conversation


class TokenMergeError(Exception):
    """
    Exception raised when a merge error occurs.
    """
    pass


def prepare_tokens(
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    target: str = "",
    attack: str | None = None,
    placement: Literal["prompt", "suffix"] = "suffix",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """For many attacks, we need to figure out how exactly to tokenize the input.
    Since only some models add a space or various control tokens, we have to figure
    out the exact format. We want to make sure that the first generated token is
    exactly 'Sure', and not a space or control token.

    We thus chunk the sequence into the following 5 parts (some of which may be empty):

    [PRE] + [Prompt] + [Attack] + [POST] + [Target]

    Treating prompt and attack separately is important for optimization, as we only
    want to optimize the attack part.
    Tested with:
        cais/zephyr_7b_r2d2
        google/gemma-2-2b-it
        HuggingFaceH4/zephyr-7b-beta
        meta-llama/Llama-2-7b-chat-hf
        meta-llama/Meta-Llama-3-8B-Instruct
        meta-llama/Meta-Llama-3.1-8B-Instruct
        microsoft/Phi-3-mini-4k-instruct
        mistralai/Mistral-7B-Instruct-v0.3
        qwen/Qwen2-7B-Instruct


    Parameters:
    - tokenizer: The tokenizer to use.
    - prompt: The prompt string to use.
    - target: The target string to use.
    - attack: The attack string to use.
    - placement: Where to place the attack. Can be either "prompt" or "suffix".

    Returns:
    - pre_tokens: The tokens before the prompt.
    - prompt_tokens: The tokens of the prompt.
    - attack_tokens: The tokens of the attack. <- optimize these
    - post_tokens: The tokens after the attack.
    - target_tokens: The tokens of the target string. <- apply loss here
    """
    if placement == "prompt":
        attack, prompt = prompt, ""
    elif attack is None:
        raise ValueError("If placement is 'suffix', attack must be provided.")

    # Some tokenizers and templates (e.g., allenai/Llama-3.1-Tulu-3-8B-DPO) need more
    # messages because their tokenization is more likely to have weird splits.
    for num_messages in [100, 1000, 10000]:
        pre_tokens, post_tokens, suffix_tokens = get_pre_post_suffix_tokens(tokenizer, num_messages)
        # Now we look at the actual chat by the user
        chat = [
            {"role": "user", "content": prompt + attack},
            {"role": "assistant", "content": target},
        ]
        tokenized_together = tokenize_chats([chat], tokenizer)[0]
        # We now cut the tokenized sequence into parts step-by-step.
        # First, we remove the prefix and suffix tokens, as we already know the prefix and
        # don't neeed the suffix.
        prompt_attack_post_target = tokenized_together[len(pre_tokens) : -len(suffix_tokens)]
        # We now look for the post tokens. These are between [prompt + attack] and [target].

        # Now, we cut out sliding views from the remaining tokens and check if they match the post tokens.
        sliding_windows = torch.stack([
            prompt_attack_post_target[i:i+len(post_tokens)]
            for i in range(len(prompt_attack_post_target) - len(post_tokens) + 1)
        ])

        # Compare each window with post_tokens
        matches = torch.all(sliding_windows == post_tokens, dim=1)
        # Find the first match index
        match_indices = torch.where(matches)[0]

        if len(match_indices) > 0:
            # Get the first match position
            i = match_indices[0].item()
            prompt_attack_tokens = prompt_attack_post_target[:i]
            target_tokens = prompt_attack_post_target[i + len(post_tokens):]
            break
    else:
        raise ValueError(
            f"Unable to find consistent tokenizer patterns for {tokenizer.name_or_path}"
        )

    tokenized_together_no_attack = get_tokenized_no_attack(prompt, target, tokenizer)

    attack_length = len(tokenized_together) - len(tokenized_together_no_attack)

    # OPTIMIZATION: Use direct indexing instead of tensor_split if possible
    prompt_tokens, attack_tokens = torch.tensor_split(
        prompt_attack_tokens, [prompt_attack_tokens.size(0)-attack_length]
    )
    if "llama-2" in tokenizer.name_or_path.lower():
        # LLama 2 models have incorrect templating and need to be fixed manually
        post_tokens = torch.cat([post_tokens, torch.tensor([29871])])

    return pre_tokens, prompt_tokens, attack_tokens, post_tokens, target_tokens


TOKENIZER_CACHE = {}


def prepare_conversation(
    tokenizer: PreTrainedTokenizerBase,
    conversation: Conversation,
    conversation_opt: Conversation | None = None,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """For many attacks, we need to figure out how exactly to tokenize the input.
    Since only some models add a space or various control tokens, we have to figure
    out the exact format. We want to make sure that the first generated token is
    exactly 'Sure', and not a space or control token.

    We thus chunk each back-and-forth message pair into the following parts
    (some of which may be empty):

    [PRE] + [Attack_Prefix0] + [Prompt0] + [Attack_Suffix0] + [POST0] + [Target0] +
    [SEP] + [Attack_Prefix1] + [Prompt1] + [Attack_Suffix1] + [POST1] + [Target1] +
    [SEP] + [Attack_Prefix2] + [Prompt2] + [Attack_Suffix2] + [POST2] + [Target2] ...

    Treating prompt and attack separately is important for optimization, as we only
    want to optimize the attack part.
    TODO: should probably move to a more general api which uses something like:
    [
        {"role": "user", "content": ["prefix", "Hello, how are you?", "suffix"]},
        {"role": "assistant", "content": ["I'm doing well, thank you!", "suffix"]},
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."}
    ]
    this would allow more granular control over the prompt segments.

    Parameters:
    - tokenizer: The tokenizer to use.
    - conversation: The conversation to use. Last message must be assistant message.
    - conversation_opt: The conversation to use for the attack. Parts of the string in
                        this conversation that are not in `conversation` are used as attack.

    Returns:
    - pre_tokens: The tokens before the prompt.
    - attack_prefix_tokens: The tokens of the attack prefix. <- optimize these
    - prompt_tokens: The tokens of the prompt.
    - attack_suffix_tokens: The tokens of the attack suffix. <- optimize these
    - post_tokens: The tokens after the attack.
    - target_tokens: The tokens of the target string. <- apply loss here
    """
    assert conversation[-1]["role"] == "assistant", "Last message must be assistant message."
    if conversation_opt is None:
        conversation_opt = copy.deepcopy(conversation)

    def get_common_prefix_len(tokens: list[torch.Tensor]) -> int:
        max_length = max(t.size(0) for t in tokens)
        padded_tokens = [F.pad(t, (0, max_length - t.size(0)), value=-1) for t in tokens]
        stacked_tokens = torch.stack(padded_tokens, dim=0)
        common_tokens = torch.all(stacked_tokens == stacked_tokens[:1, :], dim=0)
        common_prefix_len = 0
        while common_prefix_len < common_tokens.size(0) and common_tokens[common_prefix_len]:
            common_prefix_len += 1
        return common_prefix_len

    def get_common_suffix_len(tokens: list[torch.Tensor]) -> int:
        max_length = max(t.size(0) for t in tokens)
        padded_tokens = [F.pad(t, (max_length - t.size(0), 0), value=-1) for t in tokens]
        stacked_tokens = torch.stack(padded_tokens, dim=0)
        common_tokens = torch.all(stacked_tokens == stacked_tokens[:1, :], dim=0)
        common_suffix_len = 0
        while common_suffix_len < common_tokens.size(0) and common_tokens[common_tokens.size(0) - common_suffix_len - 1]:
            common_suffix_len += 1
        return common_suffix_len

    n_random_strings = 8  # Lower numbers are faster, but yield incorrect results. P_incorrect~(1/24)^n_random_strings
    out_tokens = []
    n_tokenized_clean = 0
    n_tokenized_attack = 0
    n_turns = len(conversation)

    if conversation[0]["role"] == "system":
        start_idx = 2
    else:
        start_idx = 1

    for i in range(start_idx, n_turns, 2):
        # We work our way through the conversation, section by section.

        # First, lets get the tokens before the user message.
        # For this, we replace the user message with random strings and find the common pre- and suffix.

        # sadly this cannot be cached as the suffix and sep length depends on the position in the conversation
        empty_convs = [copy.deepcopy(conversation[:i]) for _ in range(n_random_strings)]
        for conv in empty_convs:
            conv[-1]["content"] = generate_random_string(5)
        tokenized_empty = tokenize_chats(empty_convs, tokenizer)
        sep_len = get_common_prefix_len(tokenized_empty)
        common_suffix_len = get_common_suffix_len(tokenized_empty)

        sep = tokenized_empty[0][n_tokenized_clean:sep_len]
        n_tokenized_clean += sep.size(0)
        n_tokenized_attack += sep.size(0)

        # Now the user message itself.
        # Here we have to also take into account the prefix and suffix attack tokens.
        tokenized_clean = tokenize_chats([conversation[:i]], tokenizer)[0][n_tokenized_clean:]
        tokenized_attack = tokenize_chats([conversation_opt[:i]], tokenizer)[0][n_tokenized_attack:]
        if common_suffix_len > 0:
            tokenized_clean = tokenized_clean[:-common_suffix_len]
            tokenized_attack = tokenized_attack[:-common_suffix_len]
        for j in range(len(tokenized_attack)-len(tokenized_clean)+1):
            if torch.equal(tokenized_attack[j:j+len(tokenized_clean)], tokenized_clean):
                prompt = tokenized_attack[j:j+len(tokenized_clean)]
                break
        else:
            raise TokenMergeError(
                "There are tokenizer merges across prompt and attack, cannot split.\n"
                + f"Prompt: {conversation[:i]}\n"
                + f"Attack: {conversation_opt[:i]}\n"
                + f"{tokenized_clean}\n"
                + f"{tokenized_attack}"
            )
        pre_attack = tokenized_attack[:j]
        suf_attack = tokenized_attack[j+len(tokenized_clean):]
        n_tokenized_clean += prompt.size(0)
        n_tokenized_attack += pre_attack.size(0) + prompt.size(0) + suf_attack.size(0)

        # Done with user message, now time for assistant message
        if tokenizer not in TOKENIZER_CACHE:
            empty_convs = [copy.deepcopy(conversation[:i+1]) for _ in range(n_random_strings)]
            for conv in empty_convs:
                conv[-1]["content"] = generate_random_string(5)
            tokenized_empty = tokenize_chats(empty_convs, tokenizer)
            tokenized_empty = [t[n_tokenized_clean:] for t in tokenized_empty if t.size(0) > 0]
            post_len = get_common_prefix_len(tokenized_empty)
            suffix_len = get_common_suffix_len(tokenized_empty)
            TOKENIZER_CACHE[tokenizer] = post_len, suffix_len
        else:
            post_len, suffix_len = TOKENIZER_CACHE[tokenizer]

        tokenized_clean = tokenize_chats([conversation[:i+1]], tokenizer)[0]
        post = tokenized_clean[n_tokenized_clean:n_tokenized_clean+post_len]
        n_tokenized_clean += post.size(0)
        n_tokenized_attack += post.size(0)
        if "llama-2" in tokenizer.name_or_path.lower():
            # LLama 2 models have incorrect templating and need to be fixed manually
            post = torch.cat([post, torch.tensor([29871])])
            if sep[0] == 29871:
                sep = sep[1:]
        elif "gemma-2" in tokenizer.name_or_path.lower():
            if i != start_idx:
                t = torch.tensor([235248,    108])
                sep = torch.cat([t, sep])
        target = tokenized_clean[n_tokenized_clean:-suffix_len]
        n_tokenized_clean += target.size(0)
        n_tokenized_attack += target.size(0)
        out_tokens.append([sep, pre_attack, prompt, suf_attack, post, target])
    return out_tokens


def generate_random_string(k: int = 5) -> str:
    chars = string.ascii_letters + string.digits + " "

    return "".join(random.choices(chars, k=k))


def _make_random_chats(n: int, k: int = 5) -> list[Conversation]:
    """Generate n random chat conversations with k-length messages.

    Returns:
        List of chat conversations, where each conversation is a list of
        user/assistant message dictionaries with random content.
    """
    chats = []
    for _ in range(n):
        chat = [
            {"role": "user", "content": generate_random_string(k)},
            {"role": "assistant", "content": generate_random_string(k)},
        ]
        chats.append(chat)

    return chats


def _extract_prefix_middle_suffix(vectors: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    def longest_common_prefix(sequences):
        if not sequences:
            return []
        prefix = sequences[0]
        for seq in sequences[1:]:
            min_len = min(len(prefix), len(seq))
            i = 0
            while i < min_len and prefix[i] == seq[i]:
                i += 1
            prefix = prefix[:i]
            if not prefix:
                return []
        return prefix

    def longest_common_suffix(sequences):
        if not sequences:
            return []
        suffix = sequences[0]
        for seq in sequences[1:]:
            min_len = min(len(suffix), len(seq))
            i = 1
            while i <= min_len and suffix[-i] == seq[-i]:
                i += 1
            if i > 1:
                suffix = suffix[-(i - 1) :]
            else:
                return []
        return suffix

    def longest_common_subsequence(sequences):
        if not sequences:
            return []
        reference = sequences[0]
        n = len(reference)
        # Start with the longest possible substrings and decrease length
        for length in range(n, 0, -1):
            for start in range(n - length + 1):
                candidate = reference[start : start + length]
                if all(
                    any(
                        candidate == seq[i : i + length]
                        for i in range(len(seq) - length + 1)
                    )
                    for seq in sequences[1:]
                ):
                    return candidate
        return []

    sequences = [vec.tolist() for vec in vectors]
    prefix = longest_common_prefix(sequences)
    suffix = longest_common_suffix(sequences)
    # Trim the prefix and suffix from sequences
    sequences_trimmed = [
        seq[len(prefix) : len(seq) - len(suffix) if len(suffix) > 0 else None]
        for seq in sequences
    ]
    middle = longest_common_subsequence(sequences_trimmed)
    return torch.tensor(prefix), torch.tensor(middle), torch.tensor(suffix)


def tokenize_chats(chats: list[Conversation], tokenizer) -> list[torch.Tensor]:
    templates = tokenizer.apply_chat_template(
        chats, tokenize=False, add_generation_prompt=False
    )
    # Sometimes, the chat template adds the BOS token to the beginning of the template.
    # The tokenizer adds it again later, so we need to remove it to avoid duplication.
    if tokenizer.bos_token:
        for i, template in enumerate(templates):
            templates[i] = template.removeprefix(tokenizer.bos_token)

    # have to torchify individually because results may be different lengths
    return [torch.tensor(t) for t in tokenizer(templates, add_special_tokens=True).input_ids]


@lru_cache()
def get_tokenized_no_attack(prompt, target, tokenizer):
    # Cache the tokenization of the chat without attack, cause it changes rarely for
    # most attacks.
    chat_no_attack = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": target},
    ]
    return tokenize_chats([chat_no_attack], tokenizer)[0]


@lru_cache()
def get_pre_post_suffix_tokens(tokenizer, num_messages):
    # Generate random messages to find tokenizer patterns, this is ugly but fast
    test_chats = _make_random_chats(num_messages)
    test_tokenized = tokenize_chats(test_chats, tokenizer)
    return _extract_prefix_middle_suffix(test_tokenized)


def filter_suffix(
    tokenizer: PreTrainedTokenizerBase,
    clean_conversation: Conversation,
    ids: list[list[torch.Tensor | None]]
) -> list[int]:
    """
    Filters out sequences of token ids that are not invariant under decode-encode round trip.

    Example usage:
    >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    >>> clean_conversation = [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you!"},
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."}
    ]
    >>> prefix_ids_turn0 = torch.randint(1000, 2000, (512, 10))
    >>> suffix_ids_turn0 = torch.randint(1000, 2000, (512, 10))
    >>> prefix_ids_turn1 = None
    >>> suffix_ids_turn1 = torch.empty((512, 0))
    >>> ids = [[prefix_ids_turn0, suffix_ids_turn0], [prefix_ids_turn1, suffix_ids_turn1]]
    >>> filter_suffix(tokenizer, clean_conversation, ids)

    Parameters
    ----------
    tokenizer : PreTrainedTokenizerBase
        The tokenizer to use.
    clean_conversation : list of dicts
        Each dict contains {"role": ..., "content": ...}.
    ids : list of list of tensors
        Outer list indexed by conversation turn index.
        Each inner list should contain [prefix_tensor, suffix_tensor],
        both shaped (search_width, n_optim_ids).
        If the turn has no prefix or suffix, the corresponding item can also be None.
    Returns
    -------
    retain_idx : List[int]
        Indices into the search dimension where token ids are stable under decode/encode.
    """
    # Structural assertions
    assert all(len(turn_ids) == 2 for turn_ids in ids), "Each conversation turn must contain [prefix, suffix]."
    search_width = max(
        max(t.size(0) if t is not None else 0 for t, _ in ids),
        max(t.size(0) if t is not None else 0 for _, t in ids)
    )
    n_turns = len(clean_conversation)
    # Decode all ids
    decoded_tokens: list[tuple[list[str], list[str]]] = []
    for turn_prefix, turn_suffix in ids:
        prefix_decoded = tokenizer.batch_decode(turn_prefix) if turn_prefix is not None else [""] * search_width
        suffix_decoded = tokenizer.batch_decode(turn_suffix) if turn_suffix is not None else [""] * search_width
        decoded_tokens.append((prefix_decoded, suffix_decoded))

    retain_idx = []
    recon_ids = None
    for i in range(search_width):
        conversation = []
        for j in range(n_turns):
            content = clean_conversation[j]["content"]
            if j % 2 == 0:
                conversation.append({"role": "user", "content": decoded_tokens[j//2][0][i] + content + decoded_tokens[j//2][1][i]})
            else:
                conversation.append({"role": "assistant", "content": content})
        try:
            recon_ids = prepare_conversation(tokenizer, clean_conversation, conversation)
        except TokenMergeError:
            continue

        prefix_match = all([torch.equal(ids[j][0][i] if ids[j][0] is not None else torch.empty(0), recon_ids[j][1]) for j in range(len(recon_ids))])  # type: ignore
        suffix_match = all([torch.equal(ids[j][1][i] if ids[j][1] is not None else torch.empty(0), recon_ids[j][3]) for j in range(len(recon_ids))])  # type: ignore
        if prefix_match and suffix_match:
            retain_idx.append(i)

    if not retain_idx:
        # This occurs in some cases, e.g. using the Llama-3 tokenizer with a bad initialization
        raise RuntimeError(
            "No token sequences are the same after decoding and re-encoding. "
            "Consider setting `filter_ids=False` or trying a different `optim_str_init`.\n"
            "Here's an example of the token sequence that failed:\n"
            f"{ids[-1]}"
            "\n->\n"
            f"{decoded_tokens[-1]}"
            "\n->\n"
            f"{(recon_ids[-1][1], recon_ids[-1][3]) if recon_ids is not None else 'No tokenizations worked'}"
        )
    return retain_idx
