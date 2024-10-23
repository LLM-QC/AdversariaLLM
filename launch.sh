#!/bin/bash

python run_attacks.py --config-name=config_dd -m \
    ++model_name=google/gemma-2-2b-it,mistralai/Mistral-7B-Instruct-v0.3,meta-llama/Meta-Llama-3.1-8B-Instruct,qwen/Qwen2-7B-Instruct,HuggingFaceH4/zephyr-7b-beta,meta-llama/Llama-2-7b-chat-hf,ContinuousAT/Llama-2-7B-CAT,ContinuousAT/Zephyr-CAT,ContinuousAT/Phi-CAT,microsoft/Phi-3-mini-4k-instruct,cais/zephyr_7b_r2d2,GraySwanAI/Llama-3-8B-Instruct-RR,GraySwanAI/Mistral-7B-Instruct-RR \
    ++dataset_name=adv_behaviors \
    ++datasets.adv_behaviors.idx="range(0,300,25)" \
    ++datasets.adv_behaviors.batch=25 \
    ++attack_name=ample_gcg \
    ++hydra.launcher.timeout_min=500

# python run_attacks.py -m ++model_name=google/gemma-2-2b-it ++dataset_name=adv_behaviors ++datasets.adv_behaviors.idx="range(0,300, 20)" ++datasets.adv_behaviors.batch=20 ++attack_name=ample_gcg ++hydra.launcher.timeout_min=500