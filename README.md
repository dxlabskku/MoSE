# MoSE: A Mixture of Strategies Framework for Emotional Support Conversation

![overall_framework](./images/overall_framework.png)

This directory contains the supplementary implementation of **MoSE**, our single-count strategy-mixture inference pipeline for ESConv.

## Files

- `main.py`: entry point for running inference
- `MoSE.py`: strategy counting, strategy-mixture selection, response generation, and refinement
- `prompt.py`: prompts used by the pipeline
- `OAI_CONFIG_LIST`: AutoGen config for an OpenAI-compatible endpoint

## Requirements

Required Python packages:

```bash
pip install pyautogen sentence-transformers torch transformers tqdm numpy
```

The default LLM name in the code is `qwen2.5:32b`.

## Setup

1. Place the dataset here:

```text
dataset/ESConv.json
```

2. Prepare `OAI_CONFIG_LIST` so it matches your actual endpoint, model name, and credentials.

## Run

Run from this directory:

```bash
python main.py --num_samples 100
```

If you want to restrict execution to a specific GPU, you can use `CUDA_VISIBLE_DEVICES`:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py --num_samples 100
```

Example with a few useful options:

```bash
python main.py \
  --llm_name qwen2.5:32b \
  --num_samples 100 \
  --retrieval_top_k 10 \
  --mixture_agent_num 3 \
  --max_mixture_strategies 3 \
  --save_path results_single_count
```

## Output

Results are written under:

```text
results_single_count/<timestamp>/results.json
```

If `embeddings.txt` does not exist, it is created automatically from `dataset/ESConv.json`.

## Notes

- `OAI_CONFIG_LIST` must exist and must match your serving setup.
- This code assumes an OpenAI-compatible endpoint configured through AutoGen.
- Runtime and outputs can vary depending on endpoint behavior, cache state, and hardware.
