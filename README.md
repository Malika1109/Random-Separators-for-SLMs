
# Random Separators for Small Language Models

This repository contains the code for my MSc dissertation experiments on **prompt optimisation for Small Language Models (SLMs)** using **random separator sampling**.

It builds on and extends the methodology from:
> **Strings from the Library of Babel: Random Sampling as a Strong Baseline for Prompt Optimisation**  
> Yao Lu, Jiayi Wang, Raphael Tang, Sebastian Riedel, Pontus Stenetorp  
> [NAACL 2024](https://aclanthology.org/2024.naacl-long.122)

---

## Overview

The original Lu et al. work evaluated random separator strategies **only on Large Language Models (LLMs)** and **classification tasks**.

This code adapts their framework to:
- **Small Language Models** (< 1B parameters), e.g., `gpt2`, `EleutherAI/gpt-neo-125M`, `Qwen2.5-0.5B`, etc.
- **Both classification and generation tasks**  
  - Classification: SST-2, SST-5, MR, Subj, AG News, TREC, MPQA  
  - Generation: SAMSum (summarisation) and ASSET (text simplification)
- **Multiple separator generation strategies**, including:
  - `random_vocab` — random token IDs from model's vocabulary
  - `random_wo_context` — model-generated separator without context
  - `random_with_context` — model-generated separator with few-shot context
  - Baselines (e.g., `"Answer:"`)

---

## Repository Structure

```
├── main.py          # Entry point for experiments
├── run.sh           # Example execution script
├── datasets/        # Data loading utilities
├── utils/           # Helper functions
├── separator_accuracy_distribution_[model_name]/ # Saved summary statistics for  experiments
├── separator_logs_[model_name]/         # Top 5 separators for an experiment, with training and testing score
└── README.md        # This file
```

## Run an experiment

An example execution is provided in `run.sh`:

```bash
python3 main.py \
  --model gpt2 \
  --num_random_draw 160 \
  --context_shot_size 1 \
  --corpus_size 64 \
  --optimization_mode random_vocab \
  --seed 1 \
  --dataset sst2
```

## Command-Line Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `--model` | Model name from Hugging Face: `gpt2`, `EleutherAI/gpt-neo-125M`, `HuggingFaceTB/SmolLM2-360M`, `EleutherAI/pythia-410m`, `Qwen/Qwen2.5-0.5B` | `gpt2` |
| `--num_random_draw` | Number of random separators to sample | `160` |
| `--context_shot_size` | Number of in-context examples in prompt (e.g., 1-shot) | `1` |
| `--corpus_size` | Number of training examples to use for separator evaluation | `64` |
| `--optimization_mode` | Separator generation method: `random_vocab`, `random_wo_context`, `random_with_context`, `human_baseline`, `cot`, `gen_cot`, `simplify` | `random_vocab` |
| `--seed` | Random seed for reproducibility | `1` |
| `--dataset` | Dataset name: `sst2`, `sst5`, `mr`, `subj`, `agnews`, `trec`, `mpqa`, `samsum`, `asset` | `sst2` |

## Implementation Notes

This repository builds from the original implementation by Lu et al.

* Code is extended to **SLM-specific evaluation**
* **Generation tasks** are added
* **Cross-validation** and **robustness testing** are included
* All metrics for classification are **exact match accuracy**
* Generation tasks use:
   * **ROUGE-L F1** for SAMSum
   * **SARI** for ASSET

## Citation

If you use this code, please cite the original work by Lu et al.:

```bibtex
@inproceedings{lu-etal-2024-strings,
    title = "Strings from the Library of Babel: Random Sampling as a Strong Baseline for Prompt Optimisation",
    author = "Lu, Yao and
      Wang, Jiayi and
      Tang, Raphael and
      Riedel, Sebastian and
      Stenetorp, Pontus",
    booktitle = "Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)",
    month = jun,
    year = "2024",
    address = "Mexico City, Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.naacl-long.122",
    pages = "2221--2231",
}
```