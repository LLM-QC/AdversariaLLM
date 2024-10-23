import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import random
from typing import Literal


@torch.no_grad
def get_batched_completions(
    model,
    tokenizer,
    embedding_list=None,
    token_list=None,
    max_new_tokens: int = 256,
    return_tokens=False,
    padding_side='right',
) -> list[str] | torch.Tensor:
    """
    Generate completions for multiple prompts in a single batch.
    No KV-cache for now.
    Heavily tested across models to be close to individual generations.
    This is far from trivial due to various padding (left/right) and masking issues.
    The final function is still not identical to individual generations, but it is close.
    The reason for this is that attention masks typically do not use -inf for masked tokens,
    but instead use values like -65504 for float16. This can lead to small differences in the
    final logits and thus the generated tokens.
    We are much closer to individual generations than HF model.generate, which often
    fails in mysterious ways for LLama & Qwen models.
    Number of generations that are the same as single-batch:
        Model Name                         This function    HF generate
        cais/zephyr_7b_r2d2                      100/100        100/100
        ContinuousAT/Llama-2-7B-CAT              100/100         39/100
        ContinuousAT/Phi-CAT                      90/100         95/100
        ContinuousAT/Zephyr-CAT                  100/100        100/100
        google/gemma-2-2b-it                      55/100         56/100
        meta-llama/Meta-Llama-3.1-8B-Instruct     62/100         11/100
        meta-llama/Llama-2-7b-chat-hf             88/100         30/100
        microsoft/Phi-3-mini-4k-instruct          53/100         50/100
        mistralai/Mistral-7B-Instruct-v0.3        83/100         79/100
        qwen/Qwen2-7B-Instruct                    78/100         19/100
        ---------------------------------------------------------------
        Total                                    809/1000      579/1000

    Args:
        model: A pretrained model.
        tokenizer: A pretrained tokenizer.
        embedding_list: list[torch.Tensor], optional
            A list of embeddings for each prompt. Should not be padded and can be of different lengths.
        token_list: list[torch.Tensor], optional
            A list of tokens for each prompt. Should not be padded and can be of different lengths.
        max_new_tokens: The maximum number of tokens to generate for each prompt.
    Returns:
        A list of completions for each prompt.
    """
    if embedding_list is  None and token_list is None:
        raise ValueError("Either embedding_list or token_list must be provided.")
    if embedding_list is not None:
        assert all(e.ndim == 2 for e in embedding_list), "Embeddings must be 2D."
        embedding_list = [e.to(model.device) for e in embedding_list]
    if token_list is not None:
        assert all(t.ndim == 1 for t in token_list), "Tokens must be 1D."
        token_list = [t.to(model.device) for t in token_list]
        embedding_list = [
            model.get_input_embeddings()(t.unsqueeze(0))[0] for t in token_list
        ]

    B = len(embedding_list)
    tokens = []
    if padding_side == 'left':
        # Add left padding
        embeddings = pad_sequence(
            [e.flip(0) for e in embedding_list], batch_first=True, padding_value=0
        ).flip(1)
        padded_embeddings = F.pad(embeddings, (0, 0, 0, max_new_tokens))
        # Create attention mask and position ids
        lengths = [
            {
                "padding": embeddings.size(1) - e.size(0),
                "generation": max_new_tokens - e.size(0),
            }
            for e in embedding_list
        ]
        attention_mask = torch.stack(
            [
                torch.cat([torch.zeros(pl["padding"]), torch.ones([pl["generation"]])])
                for pl in lengths
            ]
        ).to(model.device)
        position_ids = torch.stack(
            [
                torch.cat([torch.zeros(pl["padding"]), torch.arange(pl["generation"])])
                for pl in lengths
            ]
        ).long().to(model.device)
        next_token_idx = embeddings.size(1)
        for i in range(max_new_tokens):
            outputs = model(
                inputs_embeds=padded_embeddings[:, :next_token_idx],
                attention_mask=attention_mask[:, :next_token_idx],
                position_ids=position_ids[:, :next_token_idx],
            )
            next_tokens = outputs.logits.argmax(dim=-1)[torch.arange(B), -1]
            padded_embeddings[torch.arange(B), next_token_idx] = (
                model.get_input_embeddings()(next_tokens).detach()
            )
            tokens.append(next_tokens)
            next_token_idx += 1
    elif padding_side == 'right':
        # Add right padding
        embeddings = pad_sequence(
            [e for e in embedding_list], batch_first=True, padding_value=0
        )
        padded_embeddings = F.pad(embeddings, (0, 0, 0, max_new_tokens))
        next_token_idx = torch.tensor([e.size(0) for e in embedding_list])
        for i in range(max_new_tokens):
            outputs = model(inputs_embeds=padded_embeddings[:, :next_token_idx.max()])
            next_tokens = outputs.logits.argmax(dim=-1)[torch.arange(B), next_token_idx-1]
            padded_embeddings[torch.arange(B), next_token_idx] = (
                model.get_input_embeddings()(next_tokens).detach()
            )
            tokens.append(next_tokens)
            next_token_idx += 1
    else:
        raise ValueError(f"Unknown padding_side: {padding_side}")

    tokens = torch.stack(tokens, dim=0).T
    if return_tokens:
        return tokens
    completion = tokenizer.batch_decode(tokens, skip_special_tokens=False)
    completion = [c.split(tokenizer.eos_token)[0] for c in completion]
    return completion


@torch.no_grad
def get_batched_losses(
    model,
    targets,
    embedding_list=None,
    token_list=None,
    padding_side='right',
) -> torch.Tensor:
    """
    Get per-timestep losses for multiple ragged prompts in a single batch.
    No KV-cache for now.

    Args:
        model: A pretrained model.
        targets: A list of 1D tensors containing the target tokens for each prompt.
        embedding_list: list[torch.Tensor], optional
            A list of embeddings for each prompt. Should not be padded and can be of different lengths.
        token_list: list[torch.Tensor], optional
            A list of tokens for each prompt. Should not be padded and can be of different lengths.
        max_new_tokens: The maximum number of tokens to generate for each prompt.
    Returns:
        A list of completions for each prompt.
    """
    if embedding_list is None and token_list is None:
        raise ValueError("Either embedding_list or token_list must be provided.")
    if embedding_list is not None:
        assert all(e.ndim == 2 for e in embedding_list), "Embeddings must be 2D."
        embedding_list = [e.to(model.device) for e in embedding_list]
    if token_list is not None:
        assert all(t.ndim == 1 for t in token_list), "Tokens must be 1D."
        token_list = [t.to(model.device) for t in token_list]
        embedding_list = [
            model.get_input_embeddings()(t.unsqueeze(0))[0] for t in token_list
        ]
    assert all(t.ndim == 1 for t in targets), "Targets must be 1D."
    targets = [t.to(model.device) for t in targets]

    # We first pad the embeddings to the maximum context length of the model.
    B = len(embedding_list)
    if padding_side == 'left':
        print("Warning: Padding side 'left' is not recommended for get_batched_losses as it may yield nans.")
        # Add left padding
        embeddings = pad_sequence(
            [e.flip(0) for e in embedding_list], batch_first=True, padding_value=0
        ).flip(1)
        targets_padded = pad_sequence(
            [t.flip(0) for t in targets], batch_first=True, padding_value=0
        ).flip(1)
        # Create attention mask and position ids
        lengths = [
            {
                "padding": embeddings.size(1) - e.size(0),
                "generation": e.size(0),
            }
            for e in embedding_list
        ]
        attention_mask = torch.stack(
            [
                torch.cat([torch.zeros(pl["padding"]), torch.ones([pl["generation"]])])
                for pl in lengths
            ]
        ).to(model.device)
        position_ids = torch.stack(
            [
                torch.cat([torch.zeros(pl["padding"]), torch.arange(pl["generation"])])
                for pl in lengths
            ]
        ).long().to(model.device)
        outputs = model(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
        ).logits
        losses = F.cross_entropy(
            outputs.reshape(-1, outputs.size(-1)),
            targets_padded.view(-1),
            reduction="none",
        )
        losses = losses.view(B, -1)
        losses = [losses[i, -t.size(0):-1] for i, t in enumerate(targets)]
    elif padding_side == 'right':
        # Add right padding
        embeddings = pad_sequence(
            [e for e in embedding_list], batch_first=True, padding_value=0
        )
        targets_padded = pad_sequence(
            [t for t in targets], batch_first=True, padding_value=0
        )
        outputs = model(inputs_embeds=embeddings).logits
        losses = F.cross_entropy(
            outputs.reshape(-1, outputs.size(-1)),
            targets_padded.view(-1),
            reduction="none",
        )
        losses = losses.view(B, -1)
        losses = [losses[i, :t.size(0)-1] for i, t in enumerate(targets)]
    else:
        raise ValueError(f"Unknown padding_side: {padding_side}")

    return losses




def prepare_tokens(tokenizer, prompt: str, target: str, attack: str|None = None, placement: Literal['prompt']|Literal['suffix']="suffix") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        HuggingFaceH4/zephyr-7b-beta
        meta-llama/Llama-2-7b-chat-hf
        meta-llama/Meta-Llama-3-8B-Instruct
        meta-llama/Meta-Llama-3.1-8B-Instruct
        microsoft/Phi-3-mini-4k-instruct
        microsoft/Phi-3-mini-4k-instruct
        mistralai/Mistral-7B-Instruct-v0.3
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
    else:
        assert attack is not None

    def make_random_chats(n, k=5):
        generate_random_string = lambda: "".join(random.choices(" ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789", k=k))

        return [
            [
                {"role": "user", "content": generate_random_string()},
                {"role": "assistant", "content": generate_random_string()},
            ]
            for _ in range(n)
        ]

    def extract_prefix_middle_suffix(vectors):
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
                    suffix = suffix[-(i-1):]
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
                    candidate = reference[start:start+length]
                    if all(
                        any(candidate == seq[i:i+length] for i in range(len(seq) - length + 1))
                        for seq in sequences[1:]
                    ):
                        return candidate
            return []
        # Convert tensors to lists
        sequences = [vec.tolist() for vec in vectors]
        # Find the longest common prefix
        prefix = longest_common_prefix(sequences)
        # Find the longest common suffix
        suffix = longest_common_suffix(sequences)
        # Trim the prefix and suffix from sequences
        sequences_trimmed = [
            seq[len(prefix):len(seq)-len(suffix) if len(suffix) > 0 else None]
            for seq in sequences
        ]
        # Find the longest common subsequence in the trimmed sequences
        middle = longest_common_subsequence(sequences_trimmed)
        return torch.tensor(prefix), torch.tensor(middle), torch.tensor(suffix)
    # create random messages, this is ugly but fast
    num_messages = 100
    test_chats = make_random_chats(num_messages)
    templates = tokenizer.apply_chat_template(
        test_chats, tokenize=False, add_generation_prompt=False
    )
    # Sometimes, the chat template adds the BOS token to the beginning of the template.
    # The tokenizer adds it again later, so we need to remove it to avoid duplication.
    for i, template in enumerate(templates):
        if tokenizer.bos_token and template.startswith(tokenizer.bos_token):
            templates[i] = template.replace(tokenizer.bos_token, "", 1)
    tokenized = [tokenizer(
        template, return_tensors="pt", add_special_tokens=True
    ).input_ids for template in templates]

    pre_tokens, post_tokens, suffix_tokens = extract_prefix_middle_suffix([tok[0] for tok in tokenized])

    chat = [
        {"role": "user", "content": prompt + attack},
        {"role": "assistant", "content": target},
    ]
    template = tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=False
    )
    if tokenizer.bos_token and template.startswith(tokenizer.bos_token):
        template = template.replace(tokenizer.bos_token, "", 1)

    tokenized_together = tokenizer(
        template, return_tensors="pt", add_special_tokens=True
    ).input_ids

    prompt_attack_post_target = tokenized_together[0, len(pre_tokens): -len(suffix_tokens)]
    prompt_attack_tokens = None
    for i in range(max(len(prompt_attack_post_target)-len(post_tokens), 0)):
        if torch.all(prompt_attack_post_target[i:i+len(post_tokens)] == post_tokens):
            prompt_attack_tokens, target_tokens = prompt_attack_post_target[:i], prompt_attack_post_target[i+len(post_tokens):]
            break
    else:
        raise ValueError(f"Unable to find consistent tokenizer patterns for {tokenizer.name_or_path}")

    chat_no_attack = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": target},
    ]
    template_no_attack = tokenizer.apply_chat_template(
        chat_no_attack, tokenize=False, add_generation_prompt=False
    )
    if tokenizer.bos_token and template_no_attack.startswith(tokenizer.bos_token):
        template_no_attack = template_no_attack.replace(tokenizer.bos_token, "", 1)

    tokenized_together_no_attack = tokenizer(
        template_no_attack, return_tensors="pt", add_special_tokens=True
    ).input_ids
    attack_length = len(tokenized_together[0]) - len(tokenized_together_no_attack[0])

    prompt_tokens, attack_tokens = torch.tensor_split(prompt_attack_tokens, [-attack_length])
    if 'llama-2' in tokenizer.name_or_path.lower():
        # LLama 2 models have incorrect templating and need to be fixed manually
        post_tokens = torch.cat([post_tokens, torch.tensor([29871])])
    return pre_tokens, prompt_tokens, attack_tokens, post_tokens, target_tokens