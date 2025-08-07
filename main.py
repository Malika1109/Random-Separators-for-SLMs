import torch
import random
import argparse
import transformers
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Callable, Optional, Set
import os
import string
from math import log

# NLP and ML libraries
import nltk
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.tree import Tree
from nltk.corpus import wordnet
import pandas as pd
from sklearn.model_selection import KFold
from evaluate import load
from rouge import Rouge
from sacremoses import MosesTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM

# Local imports
import utils # local utility file for loading datasets and setting seed

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('maxent_ne_chunker', quiet=True)
nltk.download('words', quiet=True)
nltk.download('maxent_ne_chunker_tab', quiet=True)

# Suppress transformers warnings for cleaner output
transformers.logging.set_verbosity_error()
print("Starting the script...")

from rouge import Rouge
from sacremoses import MosesTokenizer

def cal_rouge(output_texts: List[str], ref_texts: List[str], 
              tokenizer: MosesTokenizer, rouge: Rouge) -> tuple:
    """
    Calculate ROUGE scores for text generation evaluation.
    
    Args:
        output_texts: List of generated/predicted texts
        ref_texts: List of reference/ground truth texts
        tokenizer: Moses tokenizer for text preprocessing
        rouge: Rouge scorer instance
        
    Returns:
        Tuple of (ROUGE-1 F1, ROUGE-2 F1, ROUGE-L F1) scores
        
    Raises:
        ValueError: If all predictions are empty
    """
    print("Calculating ROUGE scores...")

    # Filter out empty predictions to avoid ROUGE computation errors
    filtered_pairs = [
        (pred, ref)
        for pred, ref in zip(output_texts, ref_texts)
        if pred.strip()  # Keep non-empty predictions only
    ]

    if not filtered_pairs:
        raise ValueError("All predictions were empty. Cannot compute ROUGE.")

    output_texts, ref_texts = zip(*filtered_pairs)

    # Tokenize texts using Moses tokenizer for consistent preprocessing
    output_texts = [" ".join(tokenizer.tokenize(sent)) for sent in output_texts]
    ref_texts = [" ".join(tokenizer.tokenize(sent)) for sent in ref_texts]

    scores = rouge.get_scores(output_texts, ref_texts, avg=True)
    return scores['rouge-1']['f'], scores['rouge-2']['f'], scores['rouge-l']['f']


def cal_sari(predictions: List[str], sources: List[str], references: List[str]) -> Dict:
    """
    Calculate SARI (System output Against References and Input) score.
    
    SARI is specifically designed for text simplification tasks and measures
    the quality of edits made to transform source sentences into simpler versions.
    
    Args:
        predictions: List of model-generated simplified texts
        sources: List of original complex sentences
        references: List of human-simplified reference texts
        
    Returns:
        Dictionary containing SARI score
    """
    sari_metric = load("sari")
    sari_score = sari_metric.compute(predictions=predictions, references=references, sources=sources)
    return sari_score


def evaluate_separator(separator_text: str, model, context: str, 
                      dataset: List[Dict], tokenizer) -> float:
    """
    Evaluate a separator's classification accuracy on a given dataset.

    This function is used specifically to evaluate the performance of a
    baseline separator (e.g., "Answer:") on the training and test sets
    for classification tasks, for experiments where I wanted to see how many
    random separators out of 160 were outperforming the baseline accuracy.

    It wraps model inference and returns exact match accuracy, using
    `compute_accuracy()` as the metric. All other separators (e.g., generated according to 
    the random separator strategy mentioned) are evaluated directly using `compute_accuracy()` 
    in K-Fold loops.

    Args:
        separator_text (str): The separator string to insert in prompts.
        model: The causal language model.
        context (str): Few-shot context prepended to each prompt.
        dataset (List[Dict]): List of input-output pairs.
        tokenizer: Tokenizer corresponding to the model.

    Returns:
        float: Exact match accuracy.
    """
    # Construct full prompts with context and separator
    text_sequences = [context + "\n\n" + instance["prompt"] for instance in dataset]
    text_sequences = [seq.replace("{separator}", separator_text) for seq in text_sequences]
    labels = [instance["output"] for instance in dataset]
    
    predictions = []
    batch_size = 4 
    max_new_tokens = 1 # For classification, we only need the next token

    for j in range(0, len(text_sequences), batch_size):
        batch = text_sequences[j:j + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=768).to(model.device)
        
        with torch.no_grad():
            # For newer versions of transformers:
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=False,
                return_dict_in_generate=True  # Important for structured output
            )
            generated_tokens = outputs.sequences[:, inputs['input_ids'].shape[1]:]
        
        batch_predictions = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        predictions.extend([p.strip() for p in batch_predictions])
    
    return compute_accuracy(predictions, labels)


def compute_accuracy(predictions: List[str], labels: List[str]) -> float:
    """
    Calculate accuracy as the proportion of exact matches.
    
    Args:
        predictions: List of predicted outputs
        labels: List of ground truth labels
        
    Returns:
        Accuracy score between 0 and 1
    """
    correct = 0
    for pred, label in zip(predictions, labels):
        if pred == label:
            correct += 1
    return correct / len(labels)


def generate_random_vocab_separator(min_separator_length: int, 
                                   max_separator_length: int, tokenizer, 
                                   seed: Optional[int] = None) -> str:
    """
    Generate a random separator by sampling tokens from the selected model's vocabulary.
    
    This method creates separators by randomly selecting tokens from the model's
    vocabulary, which may result in nonsensical but tokenizer-valid sequences.
    
    Args:
        min_separator_length: Minimum number of tokens in separator
        max_separator_length: Maximum number of tokens in separator
        tokenizer: Model tokenizer
        seed: Random seed for reproducibility
        
    Returns:
        Random separator string
    """
    if seed is not None:
        random.seed(seed)
    vocabulary_size = tokenizer.vocab_size
    separator_length = random.randint(min_separator_length, max_separator_length)
    separator_ids = random.sample(range(vocabulary_size), separator_length)
    separator_text = tokenizer.decode(separator_ids)
    return separator_text

def generate_random_wo_context_separator(model, tokenizer, min_separator_length: int, 
                                        max_separator_length: int) -> str:
    """
    Generate a random separator using the model without context.
    
    This method uses the language model to generate text from an empty prompt,
    potentially creating more linguistically coherent separators.
    
    Args:
        model: Language model for generation
        tokenizer: Model tokenizer
        min_separator_length: Minimum number of tokens to generate
        max_separator_length: Maximum number of tokens to generate
        
    Returns:
        Model-generated separator string
    """
    from transformers import pipeline
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    length = random.randint(min_separator_length, max_separator_length)
    return generator("", max_new_tokens=length, do_sample=True)[0]['generated_text'].strip()


def generate_random_with_context_separator(model, tokenizer, context: str,
                                          min_separator_length: int, 
                                          max_separator_length: int) -> str:
    """
    Generate a random separator using context examples as model input.
    
    This method conditions the language model on the provided context examples
    to generate potentially more task-relevant separators.
    
    Args:
        model: Language model for generation
        tokenizer: Model tokenizer  
        context: Context examples to condition generation
        min_separator_length: Minimum number of tokens to generate
        max_separator_length: Maximum number of tokens to generate
        
    Returns:
        Context-conditioned separator string
    """
    length = random.randint(min_separator_length, max_separator_length)
    
    # Encode context as input to the model
    context_input_ids = tokenizer.encode(context, return_tensors='pt').to(model.device)
    
    with torch.no_grad():
        random_separator_ids = model.generate(
            context_input_ids, 
            do_sample=True, 
            max_new_tokens=length,
            pad_token_id=tokenizer.pad_token_id
        )[0]
    
    # Decode only the newly generated tokens (exclude the input context)
    new_tokens = random_separator_ids[context_input_ids.shape[1]:]
    random_separator_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    
    return random_separator_text

def get_unique_random_separators(generate_fn: Callable, num_draws: int, 
                                *args, **kwargs) -> List[str]:
    """
    Generate a specified number of unique random separators.
    
    This function ensures that all generated separators are unique by maintaining
    a set of previously seen separators and continuing generation until the
    required number of unique separators is obtained.
    
    Args:
        generate_fn: Function to call for generating separators
        num_draws: Number of unique separators to generate
        *args, **kwargs: Arguments to pass to the generation function
        
    Returns:
        List of unique separator strings
    """
    seen = set()
    unique_separators = []
    while len(unique_separators) < num_draws:
        sep = generate_fn(*args, **kwargs).strip()
        if sep not in seen:
            seen.add(sep)
            unique_separators.append(sep)
    return unique_separators


def get_kfold_splits(data: List, k: int = 5, seed: int = 42) -> List[tuple]:
    """
    Generate K-fold cross-validation splits for robust evaluation.
    
    Args:
        data: Dataset to split
        k: Number of folds
        seed: Random seed for reproducible splits
        
    Returns:
        List of (train_indices, validation_indices) tuples
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    return [(train_idx, val_idx) for train_idx, val_idx in kf.split(data)]


def get_synonym(word: str) -> str:
    """
    Get a random synonym for a word using WordNet.
    
    Args:
        word: Input word to find synonyms for
        
    Returns:
        A random synonym if available, otherwise the original word
    """
    synsets = wordnet.synsets(word)
    lemmas = set()
    for syn in synsets:
        for lemma in syn.lemmas():
            if lemma.name().lower() != word.lower():
                lemmas.add(lemma.name().replace('_', ' '))
    return random.choice(list(lemmas)) if lemmas else word

def _char_noise(word: str) -> str:
    """
    Add lightweight character-level noise to a word.
    
    For short words (≤3 chars), performs single character replacement.
    For longer words, either swaps two internal letters or replaces one.
    
    Args:
        word: Input word to add noise to
        
    Returns:
        Word with character-level noise applied
    """
    if len(word) < 4:  # Short words: single character replacement
        idx = random.randrange(len(word))
        new_char = random.choice(string.ascii_letters)
        return word[:idx] + new_char + word[idx+1:]

    # Longer words: swap or replace internal characters
    if random.random() < 0.5:         # swap two internal letters
        i, j = random.sample(range(1, len(word)-1), 2)
        word_as_list = list(word)
        word_as_list[i], word_as_list[j] = word_as_list[j], word_as_list[i]
        return "".join(word_as_list)
    else:   # Single character replacement
        idx = random.randrange(1, len(word)-1)
        new_char = random.choice(string.ascii_letters)
        return word[:idx] + new_char + word[idx+1:]


def perturb_separator(separator: str, vocab: List[str], mode: str = "synonym") -> str:
    """
    Apply perturbations to a separator for stability testing.
    
    This function implements various perturbation strategies to test how
    sensitive model performance is to small changes in the separator text.
    
    Args:
        separator: Original separator text
        vocab: Vocabulary for word replacement
        mode: Perturbation mode ('synonym', 'replace', 'delete', 'insert', 'shuffle')
        
    Returns:
        Perturbed separator string
    """
    words = separator.strip().split()
    if not words:
        return separator

    idx = random.randint(0, len(words) - 1)
    perturbed = words.copy()

    if mode == "synonym":
        candidate = get_synonym(perturbed[idx])
        # fall back to character noise if no synonym (or identical)
        if candidate == perturbed[idx]:
            candidate = _char_noise(perturbed[idx])
        perturbed[idx] = candidate


    elif mode == "replace":
        # Replace with words of similar length from vocabulary
        candidates = [w for w in vocab if abs(len(w) - len(words[idx])) <= 2]
        if candidates:
            perturbed[idx] = random.choice(candidates)

    elif mode == "delete" and len(words) > 1:
        perturbed.pop(idx)

    elif mode == "insert":
        extra = random.choice(vocab)
        perturbed.insert(idx, extra)

    elif mode == "shuffle":
        random.shuffle(perturbed)

    return " ".join(perturbed)


def test_separator_stability(top_separators: List[Dict], model, tokenizer, 
                           context: str, test_data: List[Dict], 
                           optimization_mode: str, *, is_generation_task: bool,
                           dataset_name: str, batch_size: int = 8,
                           max_new_tokens: int = 1, max_length: int = 768,
                           cal_sari=None, cal_rouge=None, tokenizer_moses=None, 
                           rouge=None):
    """
    Test separator stability through perturbation analysis.
    
    This function evaluates how robust the top-performing separators are to
    small modifications, which helps assess whether good performance is due
    to the separator content or random chance.
    
    Args:
        top_separators: List of best-performing separators
        model: Language model for evaluation
        tokenizer: Model tokenizer
        context: Context examples for prompts
        test_data: Test dataset
        optimization_mode: Current optimization strategy
        is_generation_task: Whether this is a generation (vs classification) task
        dataset_name: Name of the dataset
        batch_size: Batch size for model inference
        max_new_tokens: Maximum tokens to generate
        max_length: Maximum sequence length
        cal_sari: SARI calculation function (for ASSET dataset)
        cal_rouge: ROUGE calculation function (for generation tasks)
        tokenizer_moses: Moses tokenizer (for ROUGE calculation)
        rouge: Rouge scorer instance
    """

    print("\n=== Separator Stability Test ===")

    # Build vocabulary for perturbations 
    vocab = [tok for tok in tokenizer.get_vocab() if tok.isalpha() and len(tok) > 2]

    def _generate(batch_prompts: List[str]) -> List[str]:
        """Helper function to generate model outputs for a batch of prompts."""
        inputs = tokenizer(batch_prompts,
                           return_tensors="pt",
                           padding=True,
                           truncation=True,
                           max_length=max_length).to(model.device)

        with torch.no_grad():
            outs = model.generate(**inputs,
                                  max_new_tokens=max_new_tokens,
                                  do_sample=False,
                                  pad_token_id=tokenizer.pad_token_id,
                                  return_dict_in_generate=True)

        gen_tokens = outs.sequences[:, inputs["input_ids"].shape[1]:]
        return tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)

    # Test each top separator with different perturbation modes
    for entry in top_separators:
        base_sep    = entry["separator"]
        base_acc    = entry.get("test_score")     # Classification
        base_sari   = entry.get("sari")           # ASSET SARI Score
        base_rouge  = entry.get("rougeL_f1")      # Generation ROUGE-L F1

        for mode in ["synonym", "replace", "insert", "delete", "shuffle"]:
            perturbed_sep = perturb_separator(base_sep, vocab, mode=mode)

            # Build full prompts with perturbed separator
            prompts = [context + "\n\n" + item["prompt"] for item in test_data]
            prompts = [p.replace("{separator}", perturbed_sep) for p in prompts]

            # Generate predictions in mini-batches
            predictions = []
            for i in range(0, len(prompts), batch_size):
                predictions.extend(_generate(prompts[i:i+batch_size]))
            predictions = [p.strip() for p in predictions]

            # Calculate appropriate metric based on task type
            if not is_generation_task:
                labels      = [item["output"] for item in test_data]
                accuracy    = compute_accuracy(predictions, labels)

                print(f"[{mode.upper()}] {repr(base_sep)} → {repr(perturbed_sep)}")
                print(f"Test Accuracy: {base_acc:.3f} → Perturbed Accuracy: {accuracy:.3f}\n")

            elif dataset_name == "asset":
                references  = [item["output"] for item in test_data]
                sources     = [item["prompt"].split("{separator}")[0].strip()
                               for item in test_data]
                sari_score  = cal_sari(predictions, sources, references)["sari"]

                print(f"[{mode.upper()}] {repr(base_sep)} → {repr(perturbed_sep)}")
                print(f"Test SARI: {base_sari:.3f} → Perturbed SARI: {sari_score:.3f}\n")

            else:
                labels      = [item["output"] for item in test_data]
                _, _, rl_f1 = cal_rouge(predictions, labels,
                                        tokenizer=tokenizer_moses,
                                        rouge=rouge)

                print(f"[{mode.upper()}] {repr(base_sep)} → {repr(perturbed_sep)}")
                print(f"Test ROUGE-L F1: {base_rouge:.3f} → Perturbed ROUGE-L F1: {rl_f1:.3f}\n")



def main(args):
    """
    Main execution function for the separator evaluation framework.
    
    This function orchestrates the entire evaluation pipeline:
    1. Load data and initialize model
    2. Generate separators using specified strategy
    3. Evaluate separators on training data (with cross-validation for classification)
    4. Test top-performing separators on test data
    5. Save results and generate visualizations
    
    Args:
        args: Parsed command-line arguments containing experiment configuration
    """
    # Extract configuration from arguments

    seed = args.seed
    dataset_name = args.dataset
    model_name = args.model
    corpus_size = args.corpus_size
    context_shot_size = args.context_shot_size
    min_separator_length = args.min_separator_length
    max_separator_length = args.max_separator_length
    num_random_draw = args.num_random_draw
    optimization_mode = args.optimization_mode

    # Load datasets
    train_corpus = utils.load_data(dataset_name, path=f"data/{dataset_name}/train.jsonl")
    test_corpus = utils.load_data(dataset_name, path=f"data/{dataset_name}/dev_subsample.jsonl")
    context_corpus = utils.load_data(dataset_name, path=f"data/{dataset_name}/train_full.jsonl")
    is_generation_task = train_corpus.task_type == "generation"

    # Initialize model and tokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    # Load model with appropriate precision
    if device == "cuda":
        try:
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to("cuda")
        except:
            model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    print(next(model.parameters()).dtype)  # Should be torch.float16


    # Prepare context examples
    if is_generation_task:
        context_examples = random.sample(list(context_corpus), context_shot_size)
    else:
        context_examples = utils.balanced_sample(context_corpus, n_shot=context_shot_size)

    random.shuffle(context_examples)


    # Prepare training and test data
    train_data = [train_corpus.__getitem__(i, include_output=False) for i in range(len(train_corpus))][:corpus_size]
    test_data = [test_corpus.__getitem__(i, include_output=False) for i in range(len(test_corpus))]
    context = "\n\n".join([elem["prompt"] for elem in context_examples])

    # Generate separators based on optimization mode
    if optimization_mode == "random_vocab":
        random_separator_texts = get_unique_random_separators(
            generate_random_vocab_separator,
            num_draws=num_random_draw,
            min_separator_length=min_separator_length,
            max_separator_length=max_separator_length,
            tokenizer=tokenizer
        )

    elif optimization_mode == "random_wo_context":
        random_separator_texts = get_unique_random_separators(
            generate_random_wo_context_separator,
            num_draws=num_random_draw,
            model=model,
            tokenizer=tokenizer,
            min_separator_length=min_separator_length,
            max_separator_length=max_separator_length
        )
    elif optimization_mode == "random_with_context":
        random_separator_texts = get_unique_random_separators(
        generate_random_with_context_separator,
        num_draws=num_random_draw,
        model=model,
        tokenizer=tokenizer,
        context=context,  # Pass the context here
        min_separator_length=min_separator_length,
        max_separator_length=max_separator_length
    )

    # Human-designed baselines
    elif optimization_mode == "human_baseline":
        random_separator_texts = ["Answer:"]

    elif optimization_mode == "cot":
        random_separator_texts = ["Let's think step by step"]

    elif optimization_mode == "gen_cot":
        random_separator_texts = ["In summary"]

    elif optimization_mode == "simplify":
        random_separator_texts = ["Simplification:"]

    else:
        raise ValueError("Unknown optimization mode")

    if optimization_mode != "guided_evolutionary" and optimization_mode != "latin_hypercube" and optimization_mode != "experiment":
        if model_name in ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"


    # Configure generation parameters based on task type
    print("running random separator search over train set...")
    separator_search_result = []
    batch_size = 4 if is_generation_task else 16 
    max_new_tokens = 45 if is_generation_task else 1

    # Evaluate baseline for classification tasks for experiments to see how many random separators outperform the baseline
    if not is_generation_task:
        baseline_separator = "Answer:"
        baseline_accuracy = evaluate_separator(baseline_separator, model, context, train_data, tokenizer)
        print(f"\n[Baseline: '{baseline_separator}'] Accuracy = {baseline_accuracy:.3f}")
        baseline_better_count = 0

        baseline_test_accuracy = evaluate_separator(baseline_separator, model, context, test_data, tokenizer)
        print(f"[Baseline: '{baseline_separator}'] Test Accuracy = {baseline_test_accuracy:.3f}")

    # Evaluation with cross-validation for classification tasks
    if not is_generation_task:
        folds = get_kfold_splits(train_data, k=5, seed=seed)
        kfold_results = defaultdict(list)

        print(f"Evaluating {len(random_separator_texts)} separators with 5-fold cross-validation...")

        for fold_num, (train_idx, val_idx) in enumerate(folds):
            print(f"\n--- Fold {fold_num+1}/5 ---")
            fold_val = [train_data[i] for i in val_idx]

            for i, separator_text in enumerate(random_separator_texts):
                # Prepare sequences for this fold
                text_sequences = [context + "\n\n" + instance["prompt"] for instance in fold_val]
                text_sequences = [seq.replace("{separator}", separator_text) for seq in text_sequences]
                labels = [instance["output"] for instance in fold_val]
                predictions = []

                # Process in batches
                for j in range(0, len(text_sequences), batch_size):
                    batch = text_sequences[j:j + batch_size]
                    inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=768).to(model.device)
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            return_dict_in_generate=True,
                            output_scores=False,
                            do_sample=False,
                            pad_token_id=tokenizer.pad_token_id
                        )
                    generated_tokens = outputs.sequences[:, inputs['input_ids'].shape[1]:]
                    batch_predictions = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                    predictions.extend([p.strip() for p in batch_predictions])

                acc = compute_accuracy(predictions, labels)
                kfold_results[separator_text].append(acc)

        # Aggregate cross-validation results
        for sep, scores in kfold_results.items():
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            separator_search_result.append({
                "separator": sep,
                "accuracy": mean_score,
                "std": std_score
            })
            # Track performance relative to baseline for classification tasks
            if mean_score > baseline_accuracy:
                baseline_better_count += 1

    # Initialize ROUGE components for generation tasks
    rouge = Rouge()
    tokenizer_moses = MosesTokenizer(lang='en')

    # Evaluation without cross-validation for generation tasks
    if is_generation_task: 
        print(f"Evaluating {len(random_separator_texts)} separators on full training set...")

        num_skipped = 0

        for i, separator_text in enumerate(random_separator_texts):
            text_sequences = [context + "\n\n" + instance["prompt"] for instance in train_data]
            text_sequences = [seq.replace("{separator}", separator_text) for seq in text_sequences]
            # Prepare sequences
            if dataset_name == "asset":
                references = [instance["output"] for instance in train_data]
                sources = [instance["prompt"].split("{separator}")[0].strip() for instance in train_data]
            else:
                labels = [instance["output"] for instance in train_data]
            predictions = []

            # Generate predictions in batches
            for j in range(0, len(text_sequences), batch_size):
                batch = text_sequences[j:j + batch_size]
                inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=768).to(model.device)
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        return_dict_in_generate=True,
                        output_scores=False,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id
                    )
                generated_tokens = outputs.sequences[:, inputs['input_ids'].shape[1]:]
                batch_predictions = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                predictions.extend([p.strip() for p in batch_predictions])
            if all(not p for p in predictions):  # all predictions empty
                print(f"Skipping {i + 1}/{len(random_separator_texts)} - EMPTY predictions for separator: {repr(separator_text)}")
                num_skipped += 1
                continue

             # Calculate metrics based on dataset type
            try:
                if dataset_name == "asset":
                    sari_score = cal_sari(predictions, sources, references)
                    sari_score = sari_score['sari']
                    print(f"{i + 1}/{len(random_separator_texts)} - SARI: {sari_score:.3f}, separator: {repr(separator_text)}")
                    separator_search_result.append({
                        "separator": separator_text,
                        "sari": sari_score
                    })

                else:
                    rouge1_f1, rouge2_f1, rougel_f1 = cal_rouge(predictions, labels, tokenizer=tokenizer_moses, rouge=rouge)
                    print(f"{i + 1}/{len(random_separator_texts)} - ROUGE-L F1: {rougel_f1:.3f}, ROUGE-1: {rouge1_f1:.3f}, ROUGE-2: {rouge2_f1:.3f}, separator: {repr(separator_text)}")
                    separator_search_result.append({
                        "separator": separator_text,
                        "rougeL_f1": rougel_f1,
                        "rouge1_f1": rouge1_f1,
                        "rouge2_f1": rouge2_f1
                    })
            except ValueError as e:
                print(f"Skipping {i + 1}/{len(random_separator_texts)} - ROUGE error: {e} - separator: {repr(separator_text)}")
                num_skipped += 1
                continue

        print(f"\nSkipped {num_skipped} separators due to empty predictions or ROUGE errors.")


    # --- Sort and summarize results on training data ---
    if is_generation_task:
        if dataset_name == "asset":
            # For ASSET (simplification): sort by SARI score
            separator_search_result.sort(key=lambda x: x["sari"], reverse=True)
            average_score = sum(x["sari"] for x in separator_search_result) / len(separator_search_result)
            print(f"average train SARI: {average_score:.3f}")
        else:
            # For SAMSum generation datasets: sort by ROUGE-L F1
            separator_search_result.sort(key=lambda x: x["rougeL_f1"], reverse=True)
            average_score = sum(x["rougeL_f1"] for x in separator_search_result) / len(separator_search_result)
            print(f"average train ROUGE-L F1: {average_score:.3f}")
    else:
        # For classification tasks: sort by accuracy
        separator_search_result.sort(key=lambda x: x["accuracy"], reverse=True)
        average_score = sum(x["accuracy"] for x in separator_search_result) / len(separator_search_result)
        print(f"average train accuracy: {average_score:.3f}")

        # Calculate and print how many separators outperform the baseline
        percent_better = 100 * baseline_better_count / len(random_separator_texts)
        print(f"\n{baseline_better_count}/{len(random_separator_texts)} random separators outperformed the baseline '{baseline_separator}'.")
        print(f"→ {percent_better:.1f}% of random separators were better than the baseline.")

    
# --- Generate histogram and save results for classification ---
    if not is_generation_task:
        separator_texts = [entry["separator"] for entry in separator_search_result]
        accuracies = [entry["accuracy"] for entry in separator_search_result]

        # Create DataFrame
        df = pd.DataFrame({
            "separator": separator_texts,
            "accuracy": accuracies
        })

        save_dir = f"separator_accuracy_distribution_{model_name}"
        os.makedirs(save_dir, exist_ok=True)

        # Compute statistics
        mean_acc = df["accuracy"].mean()
        top_5_threshold = np.percentile(df["accuracy"], 95)
        df["top_5_percent"] = df["accuracy"] >= top_5_threshold

        # Summary statistics
        print(f"\nMean Training Accuracy: {mean_acc:.3f}")
        print(f"Top 5% Threshold: {top_5_threshold:.3f}")
        print(f"Number of Top 5% Separators: {df['top_5_percent'].sum()} / {len(df)}")

        # Plot histogram of accuracy distribution
        plt.figure(figsize=(10, 6))
        plt.hist(df["accuracy"], bins=20, color="skyblue", edgecolor="black")
        plt.axvline(mean_acc, color='red', linestyle='--', label=f'Mean: {mean_acc:.2f}')
        plt.axvline(top_5_threshold, color='green', linestyle='--', label=f'Top 5% Threshold: {top_5_threshold:.2f}')
        plt.axvline(baseline_accuracy, color='orange', linestyle='--', label=f'Answer: {baseline_accuracy:.2f}')
        plt.title("Accuracy Distribution of Random Separators")
        plt.xlabel("Accuracy")
        plt.ylabel("Number of Separators")
        plt.legend()
        plt.tight_layout()

        # Save plot and results
        plot_path = os.path.join(save_dir, f"separator_accuracy_distribution_{optimization_mode}_{seed}_{dataset_name}.png")
        plt.savefig(plot_path)
        plt.show()
        print(f"Saved histogram to {plot_path}")

        # --- Save to CSV ---
        csv_path = os.path.join(save_dir, "separator_accuracy_distribution.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved results to {csv_path}")


    # --- Evaluate top 5 separators on the test set ---
    top_k_entries = []
    top_test_score = []
    top_separators = separator_search_result[:5]

    for i, separator in enumerate(top_separators):
        separator_text = separator["separator"]
        if is_generation_task:
            train_score = separator["rougeL_f1"] if "rougeL_f1" in separator else separator["sari"]
        else:
            train_score = separator["accuracy"]

        text_sequences = [context + "\n\n" + test_instance["prompt"] for test_instance in test_data]
        text_sequences = [seq.replace("{separator}", separator_text) for seq in text_sequences]
        labels = [test_instance["output"] for test_instance in test_data]
        predictions = []

        for j in range(0, len(text_sequences), batch_size):
            batch = text_sequences[j:j + batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=768).to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    return_dict_in_generate=True,
                    output_scores=False,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id
                )
            generated_tokens = outputs.sequences[:, inputs['input_ids'].shape[1]:]
            batch_predictions = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            predictions.extend([p.strip() for p in batch_predictions])

        if is_generation_task:
            if dataset_name == "asset":
                references = [test_instance["output"] for test_instance in test_data]
                sources = [test_instance["prompt"].split("{separator}")[0].strip() for test_instance in test_data]
                sari_score = cal_sari(predictions, sources, references)
                sari_score = sari_score['sari']
                print(f"top {i+1} separator: {repr(separator_text)} - Test SARI: {sari_score:.3f} - train SARI: {train_score:.3f}")
                

            else:
                rouge1_f1, rouge2_f1, rougel_f1 = cal_rouge(predictions, labels, tokenizer=tokenizer_moses, rouge=rouge)
                print(f"top {i+1} separator: {repr(separator_text)} - Test ROUGE-L F1: {rougel_f1:.3f} - train ROUGE-L F1: {train_score:.3f}")
                #separator_search_result.append({"separator": separator_text, "rougeL_f1": rougel_f1})
        else:
            test_accuracy = compute_accuracy(predictions=predictions, labels=labels)
            print(f"top {i+1} separator: {repr(separator_text)} - test accuracy: {test_accuracy:.3f} - train accuracy: {train_score:.3f}")

        if is_generation_task:
            if dataset_name == "asset":
                top_test_score.append(sari_score)
                top_k_entries.append({
                    "separator": separator_text,
                    "sari": sari_score,
                    "train_score": train_score
                })
            
            else:
                top_test_score.append(rougel_f1)
                top_k_entries.append({
                    "separator": separator_text,
                    "rougeL_f1": rougel_f1,
                    "rouge1_f1": rouge1_f1,
                    "rouge2_f1": rouge2_f1,
                    "train_score": train_score
                })

        else:
            top_test_score.append(test_accuracy)
            top_k_entries.append({
                "separator": separator_text,
                "train_score": train_score,
                "test_score": test_accuracy
            })


    # --- Compute and print average test score for top 5 separators ---
    avg_test_score = sum(top_test_score) / len(top_test_score)
    if is_generation_task:
        if dataset_name == "asset":
            print(f"average test SARI: {avg_test_score:.3f}")
        else:
            print(f"average test ROUGE-L F1: {avg_test_score:.3f}")
    else:
        print(f"average test accuracy: {avg_test_score:.3f}")
 

    # --- Save summary statistics ---
    if not is_generation_task:
        summary_stats = {
            "seed": seed,
            "optimization_mode": optimization_mode,
            "dataset": dataset_name,
            "average_train_accuracy": round(average_score, 4),
            "baseline_accuracy": round(baseline_accuracy, 4),
            "percent_better_than_baseline": round(percent_better, 2),
            "num_better_than_baseline": baseline_better_count,
            "total_separators": len(random_separator_texts),
            "average_test_accuracy": round(avg_test_score, 4),
            "baseline_test_accuracy": round(baseline_test_accuracy, 4)
        }
    else:
        summary_stats = {
            "seed": seed,
            "optimization_mode": optimization_mode,
            "dataset": dataset_name,
            "average_train_accuracy": round(average_score, 4),
            "total_separators": len(random_separator_texts),
            "average_test_accuracy": round(avg_test_score, 4),
        }


    # Convert to DataFrame for logging
    summary_df = pd.DataFrame([summary_stats])
    summary_dir = f"separator_accuracy_distribution_{model_name}"
    os.makedirs(summary_dir, exist_ok=True)
    summary_path = os.path.join(summary_dir, f"summary_stats_{optimization_mode}_{seed}_{dataset_name}.csv")
    summary_df.to_csv(summary_path, index=False)

    print(f"\nSaved summary statistics to {summary_path}")

    # --- Save top-k separator log ---
    safe_model_name = model_name.replace("/", "__")
    log_dir = f"separator_logs_{safe_model_name}"
    os.makedirs(log_dir, exist_ok=True)
    separator_log_df = pd.DataFrame(top_k_entries)

    if is_generation_task:
        if dataset_name == "asset":
            fname = f"separator_log_{dataset_name}_{optimization_mode}_{seed}_{safe_model_name}_sari.csv"
        else:
            fname = f"separator_log_{dataset_name}_{optimization_mode}_{seed}_{safe_model_name}_rouge.csv"
    else:
        fname = f"separator_log_{dataset_name}_{optimization_mode}_{seed}_{safe_model_name}_acc.csv"

    file_path = os.path.join(log_dir, fname)
    separator_log_df = pd.DataFrame(top_k_entries)
    separator_log_df.to_csv(file_path, index=False)
    print(f"Separator log saved to: {fname}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", type=str, default="sst2")
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--min_separator_length", type=int, default=1)
    parser.add_argument("--max_separator_length", type=int, default=5)
    parser.add_argument("--num_random_draw", type=int, default=160)
    parser.add_argument("--context_shot_size", type=int, default=1)
    parser.add_argument("--corpus_size", type=int, default=64)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument('--experiment_type', type=str, default='standard',
                       choices=['standard', 'length']),
    parser.add_argument("--optimization_mode", choices=[
        "random_vocab", "random_wo_context", "pos", "experiment", "ape", "evolutionary", "empty",
        "guided_evolutionary", "bandit", "human_baseline", "cot", "random_with_context", "gen_cot",
        "simplify"
    ], default="random_vocab")
    args = parser.parse_args()
    utils.set_random_seed(args.seed)
    main(args)