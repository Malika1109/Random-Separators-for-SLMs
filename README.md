
# Random Separators for Small Language Models

This repository contains the code for my MSc Individual Project experiments on **prompt optimisation for Small Language Models (SLMs)** using **random separator sampling**.

It builds on and extends the methodology from:
> **Strings from the Library of Babel: Random Sampling as a Strong Baseline for Prompt Optimisation**  
> Yao Lu, Jiayi Wang, Raphael Tang, Sebastian Riedel, Pontus Stenetorp  
> [NAACL 2024](https://aclanthology.org/2024.naacl-long.122)

---

## Overview

This project investigates whether random separators, previously shown to work on LLMs, can also improve performance in resource-constrained Small Language Models (SLMs).


This code adapts the framework of Lu et al. (2024) to:
- **Small Language Models** (< 1B parameters), e.g., `gpt2`, `EleutherAI/gpt-neo-125M`, `Qwen2.5-0.5B`, etc.
- **Both classification and generation tasks**  
  - Classification: SST-2, SST-5, MR, Subj, AG News, TREC, MPQA  
  - Generation: SAMSum (summarisation) and ASSET (text simplification)
- **Multiple separator generation strategies**, including:
  - `random_vocab` — random token IDs from model's vocabulary
  - `random_wo_context` — model-generated separator without context
  - `random_with_context` — model-generated separator with few-shot context
  - Baselines (e.g., `"Answer:"`)
- **Robustness testing** through perturbation analysis (replacement, insertion deletion, shuffling) to evaluate stability of high-performing separators.

This work advances the study of prompt optimisation in three ways:

1. Extending to SLMs — providing the first systematic analysis of random separators below 1B parameters.
2. Broadening scope — applying separators beyond classification to generative tasks like summarisation and simplification.
3. Introducing robustness analysis — systematically probing separator stability under perturbations to assess reliability.



---

## Repository Structure

```
├── main.py                                       # Entry point for experiments
├── run.sh                                        # Example execution script
├── datasets/                                     # Data loading utilities
├── utils/                                        # Helper functions
├── analyze_robustness.py                         # Analyze separator robustness across seeds for SST-5 (random with context), ASSET, TREC (random without context) and MPQA (random vocab)
├── requirements.txt                              # Python dependencies
├── separator_accuracy_distribution_[model_name]/ # Saved summary statistics for  experiments and plots showing the distribution of accuracy on training set across separators. The plots indicate the accuracies of baselines separators, the mean accuracy of all seperators, and the accuracy of the top 5% of separators on the training set. 
├── separator_logs_[model_name]/                  # Top 5 separators for an experiment, with training and testing score
├──robustness_results_[model_name]_[dataset]_[optimisation-mode]_[seed]_[model].pkl   # Saved robustness results on SST-5 (random with context), ASSET, TREC (random without context) and MPQA (random vocab), which can be plugged into analyze_robustness.py
└── README.md        # This file
```

When you run an experiment, the output logs will depend on the chosen model, dataset, and optimisation mode specified in the command-line arguments. The output reports statistics such as the mean training score across all sampled separators, the top 5 separators and their training and test scores. Additional output logs show how these top separators respond to perturbations (replacement, insertion, deletion, shuffling), with side-by-side comparisons of original versus perturbed accuracies.

Saved results are organised into folders:  

- **`separator_accuracy_distribution_[model_name]/`**  
  - Summary statistics for experiments are saved as `.csv` files, e.g.:
    `separator_accuracy_distribution_[model_name]/summary_stats_[optimisation-mode]_[seed]_[dataset].csv`
    For Classification tasks, summary statistics include    
    - `seed`, `optimization_mode`, `dataset`, `average_train_accuracy`,  
      `baseline_accuracy`, `percent_better_than_baseline` (% of separators that performed better than baseline in training set),  
      `num_better_than_baseline`, `total_separators`,  
      `average_test_accuracy` (average score of top 5 separators on test set), `baseline_test_accuracy`.  
  - For Generative tasks, summary statistics include log the `seed`, `optimisation mode`, `dataset`, `total separators` and the `average training and test scores of the separators`
  - Distribution plots (`.png`) showing separator performance: baseline accuracies, mean of all separators, and top 5% threshold.  

- **`separator_logs_[model_name]/`**  
  - Logs of the top 5 separators for each experiment with detailed training and test scores for each one.  

- **`robustness_results_[model_name]_[dataset]_[optimisation-mode]_[seed]_[model].pkl`**  
  - Pickled robustness results that can be loaded into `analyze_robustness.py` for aggregation and analysis.  


## Installation

Clone the repository and navigate into it:
```bash
git clone https://github.com/Malika1109/Random-Separators-for-SLMs.git
cd Random-Separators-for-SLMs
```

Create a virtual environment and install dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
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

## Robustness Analysis

The script `analyze_robustness.py` evaluates separator robustness across seeds and perturbations (e.g., synonym substitution, insertion, deletion) with the command ```python3 analyze_robustness.py```.

Robustness results can be produced for the following configurations: SST-5 (random with context), ASSET, TREC (random without context), MPQA (random vocab)

It loads saved `pkl` results, aggregates them across multiple seeds, and reports key statistics: Average, worst-case, best-case performance drops and most common vulnerabilities across perturbation modes.

## Implementation Notes

* Code is extended to **SLM-specific evaluation**
* **Generation tasks** are added
* **Cross-validation** for classification and **robustness testing** are included
* All metrics for classification are **exact match accuracy**
* Generation tasks use:
   * **ROUGE-L** for SAMSum
   * **SARI** for ASSET

## Reproducability

All experiments were carried out on NVIDIA A16 GPU. Reported results are averaged across seeds 1-5 for robust evaluation. The results were reported using context_shot_size = 1, num_random_draw = 160 and corpus_size = 64. 

## Citation

If you use this code, please cite this repository and the original work by Lu et al.:

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