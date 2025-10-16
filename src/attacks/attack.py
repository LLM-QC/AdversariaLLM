from dataclasses import dataclass, field
from abc import abstractmethod
from typing import Union

from torch import Tensor
import transformers
from beartype import beartype
from beartype.typing import Literal, Optional, Generic, TypeVar

from ..dataset import PromptDataset
from ..types import Conversation


@dataclass
class GenerationConfig:
    generate_completions: Literal["all", "best", "last"] = "all"
    max_new_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    num_return_sequences: int = 1


@beartype
@dataclass(kw_only=True)
class AttackStepResult:
    """Stores results for a single step of an attack algorithm."""
    step: int  # The step number (e.g., iteration)

    # The model's completion(s) in response to model_input. We use a list to store
    # multiple completions in case we are using distributional evaluation.
    model_completions: list[str]

    # Judge scores - should be a dict of judge name -> dict[str, list] mapping type to list of scores
    scores: dict[str, dict[str, list[float]]] = field(default_factory=dict)
    # Time taken specifically for this step, per prompt (i.e. if we're running a batched
    # attack, this is step_time / batch_size).
    time_taken: float = 0.0

    # FLOPS, excluding all sampling from the model which is not necessary for the algorithm.
    # For example, GCG only samples from the model once after the optimization, so we do
    # not include sampling FLOPS. PAIR on the other hand relies on generations from the
    # model during the optimization process, so we do include sampling FLOPS for that.
    flops: Optional[int] = None

    # Optional fields, depending on the attack type
    # ---
    # Loss computed at this step (e.g., target loss for GCG)
    loss: Optional[float] = None

    # Sometimes we cannot provide these: (e.g., for embedding attacks)
    # The full input presented to the model at this step (e.g., original_prompt + optimized_suffix)
    model_input: Optional[Conversation] = None
    # Actual unrolled input tokens (i.e. including the system prompt, if present)
    model_input_tokens: Optional[list[int]] = None

    model_input_embeddings: Optional[Union[Tensor, str]] = None


@beartype
@dataclass
class SingleAttackRunResult:
    """Stores the results of running a single attack on a single conversation."""
    # The original multi-turn conversation from the dataset
    # We include the target response (usually `Sure, here's how to...`) as the
    # assistant message in the original conversation to make it easier to reproduce
    # results in the future.
    original_prompt: Conversation

    # Results for each step of the attack
    steps: list[AttackStepResult] = field(default_factory=list)

    # Total time taken for this entire attack run on a **single instance**
    total_time: float = 0.0


@beartype
@dataclass
class AttackResult:
    """Stores the results for all attack runs across a dataset.

    Contains a list of SingleAttackRunResult objects, one for each instance
    in the dataset processed.
    """
    runs: list[SingleAttackRunResult] = field(default_factory=list)

AttRes = TypeVar("AttRes", bound=AttackResult)

class Attack(Generic[AttRes]):
    def __init__(self, config):
        self.config = config
        transformers.set_seed(config.seed)

    @classmethod
    def from_name(cls, name: str) -> type["Attack"]:
        match name:
            case "actor":
                from .actor import ActorAttack

                return ActorAttack
            case "autodan":
                from .autodan import AutoDANAttack

                return AutoDANAttack
            case "ample_gcg":
                from .ample_gcg import AmpleGCGAttack

                return AmpleGCGAttack
            case "beast":
                from .beast import BEASTAttack

                return BEASTAttack
            case "bon":
                from .bon import BonAttack

                return BonAttack
            case "crescendo":
                from .crescendo import CrescendoAttack

                return CrescendoAttack
            case "direct":
                from .direct import DirectAttack

                return DirectAttack
            case "gcg":
                from .gcg import GCGAttack

                return GCGAttack
            case "gcg_reinforce":
                from .gcg_reinforce import GCGReinforceAttack

                return GCGReinforceAttack
            case "gcg_refusal":
                from .gcg_refusal import GCGRefusalAttack

                return GCGRefusalAttack
            case "human_jailbreaks":
                from .human_jailbreaks import HumanJailbreaksAttack

                return HumanJailbreaksAttack
            case "pair":
                from .pair import PAIRAttack

                return PAIRAttack
            case "pgd":
                from .pgd import PGDAttack

                return PGDAttack
            case "pgd_discrete":
                from .pgd_discrete import PGDDiscreteAttack

                return PGDDiscreteAttack
            case "prefilling":
                from .prefilling import PrefillingAttack

                return PrefillingAttack

            case "random_search":
                from .random_search import RandomSearchAttack

                return RandomSearchAttack
            case _:
                raise ValueError(f"Unknown attack: {name}")

    @abstractmethod
    def run(
        self,
        model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizerBase,
        dataset: PromptDataset,
    ) -> AttRes:
        raise NotImplementedError
