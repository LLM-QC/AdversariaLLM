"""
@article{hendrycks2020measuring,
  title={Measuring massive multitask language understanding},
  author={Hendrycks, Dan and Burns, Collin and Basart, Steven and Zou, Andy and Mazeika, Mantas and Song, Dawn and Steinhardt, Jacob},
  journal={arXiv preprint arXiv:2009.03300},
  year={2020}
}
"""

from dataclasses import dataclass

from datasets import load_dataset

from ..types import Conversation

from .prompt_dataset import PromptDataset


@dataclass
class MMLUConfig:
    name: str = "mmlu"
    seed: int = 0
    idx: list[int] | int | str | None = None
    shuffle: bool = True


@PromptDataset.register("mmlu")
class MMLUDataset(PromptDataset):

    TEMPLATE = "The following are multiple choice questions (with answers) about {}.\n\n"
    LETTERS = ["A", "B", "C", "D", "E", "F"]

    def __init__(self, config: MMLUConfig):
        super().__init__(config)
        dataset = load_dataset("cais/mmlu", "all")["test"].to_pandas()

        self.idx, self.config_idx = self._select_idx(config, len(dataset))
        self.data = [dataset.iloc[idx] for idx in self.idx.tolist()]


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Conversation:
        row = self.data[idx]
        print(row)

        # Resolve subject/category for the header
        subject = row["subject"]
        formatted_subject = str(subject).replace("_", " ")

        question = row["question"]
        choices = row["choices"]

        # Build prompt text
        prompt_lines = [self.TEMPLATE.format(formatted_subject).rstrip(), str(question)]
        for j, opt in enumerate(choices):
            prompt_lines.append(f"{self.LETTERS[j]}. {opt}")
        prompt_lines.append("Answer:")
        user_prompt = "\n".join(prompt_lines)

        target_letter = self.LETTERS[int(row["answer"])]
        conversation = [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": target_letter},
        ]
        return conversation