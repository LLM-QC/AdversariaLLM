from functools import lru_cache
MODEL_FAMILIES = {
    "gemma": {
        "ids": [
            "google/gemma-2-2b-it",
        ],
        "color": (60, 136, 240),
        "markers": ['o'],

    },
    "mistral": {
        "ids": [
            "mistralai/Mistral-7B-Instruct-v0.3",
            "HuggingFaceH4/zephyr-7b-beta",
            "cais/zephyr_7b_r2d2",
            "ContinuousAT/Zephyr-CAT",
            "GraySwanAI/Mistral-7B-Instruct-RR",
            "berkeley-nest/Starling-LM-7B-alpha",
        ],
        "color": (255, 111, 32),
        "markers": ['o', 'p', 'h', '^', 'x', 's'],
    },
    "llama3": {
        "ids": [
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "meta-llama/Meta-Llama-3-8B-Instruct",
            "meta-llama/Llama-3.2-1B-Instruct",
            "meta-llama/Llama-3.2-3B-Instruct",
            "GraySwanAI/Llama-3-8B-Instruct-RR",
            "NousResearch/Hermes-2-Pro-Llama-3-8B",
            "allenai/Llama-3.1-Tulu-3-8B-DPO",
        ],
        "color": (0, 103, 219),
        "markers": ['o', 'x', 'p', 'h', 's', '^', 'd'],
    },
    "llama2": {
        "ids": [
            "meta-llama/Llama-2-7b-chat-hf",
            "lmsys/vicuna-13b-v1.5",
            "lmsys/vicuna-7b-v1.5",
            "ContinuousAT/Llama-2-7B-CAT",
        ],
        "color": (0, 120, 255),
        "markers": ['o', 'p', '^', 'x'],
    },
    "qwen": {
        "ids": [
            "qwen/Qwen2-7B-Instruct",
        ],
        "color": (91, 65, 225),
        "markers": ['o'],
    },
    "phi3": {
        "ids": [
            "microsoft/Phi-3-mini-4k-instruct",
            "ContinuousAT/Phi-CAT",
        ],
        "color": (84, 228, 136),
        "markers": ['o', '^'],
    },
}
@lru_cache
def get_model_style(model_id):
    """
    Get the color and marker associated with a given model ID.

    Args:
        model_id (str): The model ID to look up.

    Returns:
        dict: A dictionary with 'color' and 'marker', or None if not found.
    """

    for family in MODEL_FAMILIES.values():
        if model_id in family['ids']:
            # Get the index of the model in the family ID list
            index = family['ids'].index(model_id)
            # Return color and corresponding marker (cycled if needed)
            color = tuple([c / 255 for c in family['color']])
            marker = family['markers'][index % len(family['markers'])]
            return {'color': color, 'marker': marker}
    raise ValueError(f"Model ID '{model_id}' not found in the model families.")




def step_to_wall_time(step, method):
    pass