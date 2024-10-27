"""Baseline, just prompts the original prompt."""
import torch
import transformers

from src.attacks import AttackResult, Attack
from src.lm_utils import prepare_tokens, get_batched_completions, get_batched_losses


class DirectAttack(Attack):
    def __init__(self, config):
        super().__init__(config)

    @torch.no_grad
    def run(
        self,
        model: transformers.AutoModelForCausalLM,
        tokenizer: transformers.AutoTokenizer,
        dataset: torch.utils.data.Dataset,
    ) -> AttackResult:
        result = AttackResult([], [], [], [])
        token_lists = []
        for msg, target in dataset:
            result.prompts.append(msg)
            result.attacks.append([""])
            token_lists.append(prepare_tokens(
                tokenizer,
                msg["content"],
                target,
                attack="",
                placement="suffix",
            ))
        def chunks(lst, n):
            """Yield successive n-sized chunks from lst."""
            for i in range(0, len(lst), n):
                yield lst[i:i + n]

        for chunk in chunks(token_lists, self.config.batch_size):
            completions = get_batched_completions(
                model,
                tokenizer,
                token_list=[torch.cat([a, b, c, d], dim=0) for a,b,c,d,e in chunk],
                max_new_tokens=self.config.max_new_tokens,
            )
            token_list = [torch.cat([a, b, c, d, e], dim=0) for a,b,c,d,e in chunk]
            targets = [t.roll(-1, 0) for t in token_list]
            losses = get_batched_losses(model, targets, token_list=token_list)
            losses = [l[-tl:].mean().item() for l, tl in zip(losses, [e.size(0) for a,b,c,d,e in chunk])]
            for l, c in zip(losses, completions):
                result.losses.append([l])
                result.completions.append([c])
        return result