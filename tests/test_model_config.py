import hydra
import torch
from omegaconf import DictConfig

from adversariallm.io_utils import load_model_and_tokenizer
from adversariallm.lm_utils import prepare_tokens


@hydra.main(config_path="conf/models", config_name="models")
def main(cfg: DictConfig) -> None:
    print(cfg)
    for model, model_params in cfg.models.items():
        model, tokenizer = load_model_and_tokenizer(model_params)
        assert tokenizer.model_max_length <= 65536, f"Model {model} has model_max_length {tokenizer.model_max_length}, likely invalid."

        sample_convo = [
            {'role': 'user', 'content': 'What is the capital of France?'},
        ]
        text = tokenizer.apply_chat_template(sample_convo, add_generation_prompt=True, tokenize=False)
        tokens = tokenizer(text, return_tensors='pt').input_ids
        reconstructed_text = tokenizer.decode(tokens[0], skip_special_tokens=True)
        reconstructed_tokens = tokenizer(reconstructed_text, return_tensors='pt').input_ids
        assert torch.equal(tokens, reconstructed_tokens), f"Tokenization mismatch for tokenizer {tokenizer}."

        try:
            tokens = prepare_tokens(tokenizer, sample_convo[0]['content'], "Sure, here's a sample target.", attack='x x x x x x x x x x x x x x x', placement="suffix")
        except ValueError as e:
            print(f"Failed to prepare tokens for model {model}: {e}")
            continue
        print(tokens)

        print(model)
        print(tokenizer)

if __name__ == "__main__":
    main()