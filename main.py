import torch
import random
import argparse
import transformers
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Callable, Optional, Set
import utils
import os
import nltk
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.tree import Tree
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('maxent_ne_chunker_tab')
from collections import Counter
from math import log
import string
from nltk.corpus import wordnet
import pandas as pd
from sklearn.model_selection import KFold
from evaluate import load
import optuna

transformers.logging.set_verbosity_error()
print("Starting the script...")

from rouge import Rouge
from sacremoses import MosesTokenizer

def cal_rouge(output_texts, ref_texts, tokenizer, rouge):
    print("calculating rouge score...")

    # Filter out empty predictions
    filtered_pairs = [
        (pred, ref)
        for pred, ref in zip(output_texts, ref_texts)
        if pred.strip()  # non-empty after stripping whitespace
    ]

    if not filtered_pairs:
        raise ValueError("All predictions were empty. Cannot compute ROUGE.")

    output_texts, ref_texts = zip(*filtered_pairs)

    output_texts = [" ".join(tokenizer.tokenize(sent)) for sent in output_texts]
    ref_texts = [" ".join(tokenizer.tokenize(sent)) for sent in ref_texts]

    scores = rouge.get_scores(output_texts, ref_texts, avg=True)
    return scores['rouge-1']['f'], scores['rouge-2']['f'], scores['rouge-l']['f']

def cal_sari(predictions, sources, references):
    sari_metric = load("sari")
    sari_score = sari_metric.compute(predictions=predictions, references=references, sources=sources)
    return sari_score

""" def evaluate_separator(separator_text, model, context, data, tokenizer, mode):
    text_sequences = [context + "\n\n" + instance["prompt"].replace("{separator}", separator_text)
                      for instance in data]
    labels = [instance["output"] for instance in data]

    if mode not in ["guided_evolutionary", "experiment"]:
        r = model(text_sequences, max_new_tokens=1, return_full_text=False, do_sample=False, batch_size=16)
        predictions = [elem[0]['generated_text'].strip() for elem in r]
    else:
        inputs = tokenizer(text_sequences, return_tensors='pt', padding=True, truncation=True).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=1, return_dict_in_generate=True, output_scores=True, do_sample=False,
                pad_token_id=tokenizer.pad_token_id)
        generated_tokens = outputs.sequences[:, inputs['input_ids'].shape[1]:]
        predictions = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        predictions = [p.strip() for p in predictions]

    return compute_accuracy(predictions, labels) """

def evaluate_separator(separator_text, model, context, dataset, tokenizer):
    """Unified evaluation function for both experiments"""
    text_sequences = [context + "\n\n" + instance["prompt"] for instance in dataset]
    text_sequences = [seq.replace("{separator}", separator_text) for seq in text_sequences]
    labels = [instance["output"] for instance in dataset]
    
    predictions = []
    batch_size = 4 
    max_new_tokens = 1

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


def compute_accuracy(predictions, labels):
    correct = 0
    for pred, label in zip(predictions, labels):
        if pred == label:
            correct += 1
    return correct / len(labels)


def generate_random_vocab_separator(model, min_separator_length, max_separator_length, tokenizer, seed=None):
    if seed is not None:
        random.seed(seed)
    vocabulary_size = tokenizer.vocab_size
    separator_length = random.randint(min_separator_length, max_separator_length)
    separator_ids = random.sample(range(vocabulary_size), separator_length)
    separator_text = tokenizer.decode(separator_ids)
    return separator_text

def generate_random_wo_context_separator(model, tokenizer, min_separator_length, max_separator_length):
    from transformers import pipeline
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    length = random.randint(min_separator_length, max_separator_length)
    return generator("", max_new_tokens=length, do_sample=True)[0]['generated_text'].strip()

def generate_random_with_context_separator(model, tokenizer, context, min_separator_length, max_separator_length):
    """
    Generate random separator using context examples as input to the model
    """
    length = random.randint(min_separator_length, max_separator_length)
    
    # Use context as input instead of empty string
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

def get_unique_random_separators(generate_fn, num_draws, *args, **kwargs):
    seen = set()
    unique_separators = []
    while len(unique_separators) < num_draws:
        sep = generate_fn(*args, **kwargs).strip()
        if sep not in seen:
            seen.add(sep)
            unique_separators.append(sep)
    return unique_separators

def get_kfold_splits(data, k=5, seed=42):
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    return [(train_idx, val_idx) for train_idx, val_idx in kf.split(data)]


def get_synonym(word: str) -> str:
    """Returns a random synonym from WordNet if available."""
    synsets = wordnet.synsets(word)
    lemmas = set()
    for syn in synsets:
        for lemma in syn.lemmas():
            if lemma.name().lower() != word.lower():
                lemmas.add(lemma.name().replace('_', ' '))
    return random.choice(list(lemmas)) if lemmas else word

def _char_noise(word: str) -> str:
    """Add lightweight character noise: swap two letters or replace one."""
    if len(word) < 4:                 # ≤ 3 chars → do single-letter replacement
        idx = random.randrange(len(word))
        new_char = random.choice(string.ascii_letters)
        return word[:idx] + new_char + word[idx+1:]

    # At least two internal positions exist → safe to swap
    if random.random() < 0.5:         # swap two internal letters
        i, j = random.sample(range(1, len(word)-1), 2)
        word_as_list = list(word)
        word_as_list[i], word_as_list[j] = word_as_list[j], word_as_list[i]
        return "".join(word_as_list)
    else:                             # single-letter replacement
        idx = random.randrange(1, len(word)-1)
        new_char = random.choice(string.ascii_letters)
        return word[:idx] + new_char + word[idx+1:]


def perturb_separator(separator: str, vocab: List[str], mode: str = "synonym") -> str:
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

  
def test_separator_stability_original(top_k_entries: List[Dict], model, tokenizer, context, test_data, optimization_mode):
    print("\n=== Separator Stability Test ===")
    vocab = [token for token in tokenizer.get_vocab().keys() if token.isalpha() and len(token) > 2]

    for entry in top_k_entries:
        for mode in ["synonym", "replace", "insert", "delete", "shuffle"]:
            original_sep = entry['separator']
            perturbed_sep = perturb_separator(original_sep, vocab, mode=mode)

            # Prepare input with perturbed separator
            text_sequences = [context + "\n\n" + test_instance["prompt"] for test_instance in test_data]
            text_sequences = [elem.replace("{separator}", perturbed_sep) for elem in text_sequences]

            labels = [test_instance["output"] for test_instance in test_data]

            inputs = tokenizer(
            text_sequences,
            return_tensors="pt",
            padding=True,
            truncation=True
            ).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=1,
                    return_dict_in_generate=True,
                    output_scores=True,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id
                )

            generated_tokens = outputs.sequences[:, inputs['input_ids'].shape[1]:]
            predictions = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            predictions = [p.strip() for p in predictions]

            accuracy = compute_accuracy(predictions=predictions, labels=labels)
            print(f"[{mode.upper()}] {repr(original_sep)} → {repr(perturbed_sep)}")
            print(f"Test Accuracy: {entry['test_score']:.3f} → Perturbed Accuracy: {accuracy:.3f}\n")

from typing import List, Dict

def test_separator_stability(
        top_separators: List[Dict],
        model,
        tokenizer,
        context: str,
        test_data: List[Dict],
        optimization_mode: str,
        *,
        is_generation_task: bool,
        dataset_name: str,
        batch_size: int      = 8,
        max_new_tokens: int  = 1,
        max_length: int      = 768,
        # ↓ only needed for generation metrics
        cal_sari=None,
        cal_rouge=None,
        tokenizer_moses=None,
        rouge=None,
):
    """
    Re-evaluate each top separator after perturbations and report how the test
    score changes.  For classification we use *accuracy*.  For generation:
      • ASSET  → SARI
      • else   → ROUGE-L F1
    """

    print("\n=== Separator Stability Test ===")

    # build a simple word-only vocabulary once
    vocab = [tok for tok in tokenizer.get_vocab() if tok.isalpha() and len(tok) > 2]

    # -----------------------------------------------------------------------
    # helper → run the model exactly like in `main`
    # -----------------------------------------------------------------------
    def _generate(batch_prompts: List[str]) -> List[str]:
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

    # -----------------------------------------------------------------------
    # main loop over the top-k separators and perturbation modes
    # -----------------------------------------------------------------------
    for entry in top_separators:
        base_sep    = entry["separator"]

        # whichever score your search phase stored:
        base_acc    = entry.get("test_score")     # classification
        base_sari   = entry.get("sari")           # ASSET
        base_rouge  = entry.get("rougeL_f1")      # other generation

        for mode in ["synonym", "replace", "insert", "delete", "shuffle"]:
            perturbed_sep = perturb_separator(base_sep, vocab, mode=mode)

            # build full prompts once
            prompts = [context + "\n\n" + item["prompt"] for item in test_data]
            prompts = [p.replace("{separator}", perturbed_sep) for p in prompts]

            # run model in mini-batches
            predictions = []
            for i in range(0, len(prompts), batch_size):
                predictions.extend(_generate(prompts[i:i+batch_size]))
            predictions = [p.strip() for p in predictions]

            # ---------------- choose metric exactly as in `main` ------------
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

def run_length_experiment(args, model, tokenizer, context, dataset):
    """Run controlled length experiment with statistical rigor"""
    from scipy import stats
    results = {}
    lengths = range(args.min_separator_length, args.max_separator_length + 1)
    
    for l in lengths:
        scores = []
        examples = []
        for i in range(args.num_random_draw):
            sep = generate_random_wo_context_separator(
                model, tokenizer, l, l
            )
            score = evaluate_separator(sep, model, context, dataset, tokenizer, args)
            scores.append(score)
            if len(examples) < 3:  # Store first 3 examples
                examples.append(sep)
        
        mean, std = np.mean(scores), np.std(scores)
        ci_low, ci_high = stats.t.interval(
            0.95, len(scores)-1, loc=mean, scale=std/np.sqrt(len(scores)))
        
        results[l] = {
            'mean': mean,
            'std': std,
            'ci_low': ci_low,
            'ci_high': ci_high,
            'examples': examples
        }
    
    return results



def main(args):
    seed = args.seed
    dataset_name = args.dataset
    model_name = args.model
    corpus_size = args.corpus_size
    context_shot_size = args.context_shot_size
    min_separator_length = args.min_separator_length
    max_separator_length = args.max_separator_length
    num_random_draw = args.num_random_draw
    optimization_mode = args.optimization_mode

    train_corpus = utils.load_data(dataset_name, path=f"data/{dataset_name}/train.jsonl")
    test_corpus = utils.load_data(dataset_name, path=f"data/{dataset_name}/dev_subsample.jsonl")
    context_corpus = utils.load_data(dataset_name, path=f"data/{dataset_name}/train_full.jsonl")
    is_generation_task = train_corpus.task_type == "generation"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    #model = transformers.pipeline("text-generation", model=model_name, device=device)
    #model.tokenizer.pad_token_id = model.model.config.eos_token_id
    #model.tokenizer.padding_side = "left"
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    # Load model with FP16 if on GPU
    if device == "cuda":
        try:
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to("cuda")
        except:
            model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    print(next(model.parameters()).dtype)  # Should be torch.float16


    if is_generation_task:
        context_examples = random.sample(list(context_corpus), context_shot_size)
    else:
        context_examples = utils.balanced_sample(context_corpus, n_shot=context_shot_size)

    random.shuffle(context_examples)

    train_data = [train_corpus.__getitem__(i, include_output=False) for i in range(len(train_corpus))][:corpus_size]
    test_data = [test_corpus.__getitem__(i, include_output=False) for i in range(len(test_corpus))]
    context = "\n\n".join([elem["prompt"] for elem in context_examples])

    # --- Experiment Selection --- #
    if args.experiment_type == "length":
        print("Running length experiment...")
        length_results = run_length_experiment(
            args, model, tokenizer, context, train_data
        )
        
        # Print and save results
        print("\nLength Experiment Results:")
        for length, result in length_results.items():
            print(f"Length {length}: Mean = {result['mean']:.4f} ± {result['std']:.4f}")
            print(f"  95% CI: [{result['ci_low']:.4f}, {result['ci_high']:.4f}]")
            print(f"  Example separators: {result['examples']}")
        
        # Optional: Plot results
        #plot_length_results(length_results)
        return

    # --- Separator generation --- #
    if optimization_mode == "random_vocab":
        random_separator_texts = get_unique_random_separators(
            generate_random_vocab_separator,
            num_draws=num_random_draw,
            model=model,
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

    elif optimization_mode == "evolutionary":
        # instead of random sampling, do evolutionary search
        from transformers import AutoTokenizer
        from evolutionary import evolutionary_search
        vocab = [token for token in tokenizer.get_vocab().keys() if token.isalpha()]  # simple vocab filtering
        
        def objective(trial):
            mutation_rate = trial.suggest_float("mutation_rate", 0.1, 0.7)
            pop_size = trial.suggest_int("pop_size", 8, 60)
            insert_prob = trial.suggest_float("insert_prob", 0.1, 0.4)
            delete_prob = trial.suggest_float("delete_prob", 0.1, 0.4)

            best_separator = evolutionary_search(
                model=model,
                evaluate_fn=lambda sep: evaluate_separator(sep, model, context, train_data, tokenizer),
                vocab=vocab,
                tokenizer=tokenizer,
                generations=10,
                pop_size=pop_size,
                min_sep_len=args.min_separator_length,
                max_sep_len=args.max_separator_length,
                mutation_rate=mutation_rate,
                insert_prob=insert_prob,
                delete_prob=delete_prob
            )
            val_acc = evaluate_separator(best_separator, model, context, test_data, tokenizer)  # use test set to evaluate fitness
            return val_acc  # maximize accuracy!

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=5)  # try 15 different hyperparameter combos

        print("Best hyperparameters:", study.best_params)
        
        # Now retrain evolutionary search with best hyperparameters
        best_params = study.best_params
        best_separator = evolutionary_search(
            model=model,
            evaluate_fn=lambda sep: evaluate_separator(sep, model, context, train_data, tokenizer),
            vocab=vocab,
            tokenizer=tokenizer,
            generations=10,
            pop_size=best_params["pop_size"],
            min_sep_len=args.min_separator_length,
            max_sep_len=args.max_separator_length,
            mutation_rate=best_params["mutation_rate"],
            insert_prob=best_params["insert_prob"],
            delete_prob=best_params["delete_prob"]
        )
        random_separator_texts = [best_separator]  # wrap into list



    elif optimization_mode == "pos":
        from pos import generate_multiple_separators
        random_separator_texts = generate_multiple_separators(
            num_samples=num_random_draw, 
            template="noun_verb"  # can also try "noun_verb" or "the_noun_verbing"
        )

    elif optimization_mode == "experiment":
        from experiment import metropolis_hastings_separators
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"

        model = AutoModelForCausalLM.from_pretrained(model_name)
        evaluate_fn = lambda sep: evaluate_separator(sep, model, tokenizer, context, train_data)

        random_separator_texts = metropolis_hastings_separators(
        tokenizer=tokenizer,
        evaluate_fn=evaluate_fn,
        n_samples=num_random_draw,
        )


    elif optimization_mode == "guided_evolutionary":
        from guided_evolutionary import POSAwareSeparatorEvolution
        from transformers import AutoTokenizer, AutoModelForCausalLM
        #tokenizer = AutoTokenizer.from_pretrained(args.model)
        #model_raw = AutoModelForCausalLM.from_pretrained(args.model).to(model.device)
        eval_fn = lambda sep: evaluate_separator(sep, model, context, train_data, tokenizer)
        evolver = POSAwareSeparatorEvolution(model, tokenizer, eval_fn, pos_structure= ["ADJ", "NOUN", "VERB", "PUNCT"])
        best_sep, _ = evolver.evolve(generations=10, pop_size=30)
        random_separator_texts = [best_sep]

    elif optimization_mode == "bandit":
        from bandit import SeparatorBandit

        #tokenizer = model.tokenizer
        vocab = sorted([t for t in tokenizer.get_vocab().keys() if t.replace("Ġ", "").isalpha()])

        bandit = SeparatorBandit(
            evaluate_fn=evaluate_separator,
            model=model,
            context=context,
            train_data=train_data,
            tokenizer=tokenizer,
            vocab=vocab,
            num_draws=num_random_draw,
            min_len=args.min_separator_length,
            max_len=args.max_separator_length,
            epsilon=0.2,
            seed=args.seed
        )

        best_separator, reward_history = bandit.run()
        random_separator_texts = [best_separator]


    elif optimization_mode == "human_baseline":
        random_separator_texts = ["Answer:"]

    elif optimization_mode == "empty":
        random_separator_texts = [""]

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


    print("running random separator search over train set...")
    separator_search_result = []
    batch_size = 4 if is_generation_task else 16 #changed 16 to 8
    max_new_tokens = 45 if is_generation_task else 1

    if not is_generation_task:
        baseline_separator = "Answer:"
        baseline_accuracy = evaluate_separator(baseline_separator, model, context, train_data, tokenizer)
        print(f"\n[Baseline: '{baseline_separator}'] Accuracy = {baseline_accuracy:.3f}")
        baseline_better_count = 0

        baseline_test_accuracy = evaluate_separator(baseline_separator, model, context, test_data, tokenizer)
        print(f"[Baseline: '{baseline_separator}'] Test Accuracy = {baseline_test_accuracy:.3f}")


    if not is_generation_task:
        # Prepare KFold
        folds = get_kfold_splits(train_data, k=5, seed=seed)
        kfold_results = defaultdict(list)

        print(f"Evaluating {len(random_separator_texts)} separators with 5-fold cross-validation...")

        for fold_num, (train_idx, val_idx) in enumerate(folds):
            print(f"\n--- Fold {fold_num+1}/5 ---")
            fold_val = [train_data[i] for i in val_idx]

            for i, separator_text in enumerate(random_separator_texts):
                text_sequences = [context + "\n\n" + instance["prompt"] for instance in fold_val]
                text_sequences = [seq.replace("{separator}", separator_text) for seq in text_sequences]
                labels = [instance["output"] for instance in fold_val]
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

                acc = compute_accuracy(predictions, labels)
                kfold_results[separator_text].append(acc)

        for sep, scores in kfold_results.items():
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            separator_search_result.append({
                "separator": sep,
                "accuracy": mean_score,
                "std": std_score
            })
            # Compare to baseline
            if mean_score > baseline_accuracy:
                baseline_better_count += 1

    rouge = Rouge()
    tokenizer_moses = MosesTokenizer(lang='en')

    if is_generation_task: # without cross validation
        import time
        print(f"Evaluating {len(random_separator_texts)} separators on full training set...")

        num_skipped = 0

        for i, separator_text in enumerate(random_separator_texts):
            text_sequences = [context + "\n\n" + instance["prompt"] for instance in train_data]
            text_sequences = [seq.replace("{separator}", separator_text) for seq in text_sequences]
            if dataset_name == "asset":
                references = [instance["output"] for instance in train_data]
                sources = [instance["prompt"].split("{separator}")[0].strip() for instance in train_data]
            else:
                labels = [instance["output"] for instance in train_data]
            predictions = []

            for j in range(0, len(text_sequences), batch_size):
                batch = text_sequences[j:j + batch_size]
                #t0 = time.time()
                inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=768).to(model.device)
                #print(f"Tokenizer: {time.time() - t0:.2f}s")
                t1 = time.time()
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        return_dict_in_generate=True,
                        output_scores=False,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id
                    )
                #print(f"Generation: {time.time() - t1:.2f}s")
                #t2 = time.time()
                generated_tokens = outputs.sequences[:, inputs['input_ids'].shape[1]:]
                batch_predictions = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                predictions.extend([p.strip() for p in batch_predictions])
                #print(f"Decoding: {time.time() - t2:.2f}s")
            if all(not p for p in predictions):  # all predictions empty
                print(f"Skipping {i + 1}/{len(random_separator_texts)} - EMPTY predictions for separator: {repr(separator_text)}")
                num_skipped += 1
                continue

            #import time
            #start = time.time()
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


    # Final sorting and averaging
    if is_generation_task:
        if dataset_name == "asset":
            separator_search_result.sort(key=lambda x: x["sari"], reverse=True)
            average_score = sum(x["sari"] for x in separator_search_result) / len(separator_search_result)
            print(f"average train SARI: {average_score:.3f}")
        else:
            separator_search_result.sort(key=lambda x: x["rougeL_f1"], reverse=True)
            average_score = sum(x["rougeL_f1"] for x in separator_search_result) / len(separator_search_result)
            print(f"average train ROUGE-L F1: {average_score:.3f}")
    else:
        separator_search_result.sort(key=lambda x: x["accuracy"], reverse=True)
        average_score = sum(x["accuracy"] for x in separator_search_result) / len(separator_search_result)
        print(f"average train accuracy: {average_score:.3f}")
        percent_better = 100 * baseline_better_count / len(random_separator_texts)
        print(f"\n{baseline_better_count}/{len(random_separator_texts)} random separators outperformed the baseline '{baseline_separator}'.")
        print(f"→ {percent_better:.1f}% of random separators were better than the baseline.")

    

    
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

        # Print summary
        print(f"\nMean Training Accuracy: {mean_acc:.3f}")
        print(f"Top 5% Threshold: {top_5_threshold:.3f}")
        print(f"Number of Top 5% Separators: {df['top_5_percent'].sum()} / {len(df)}")

        # Plot distribution
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
        plot_path = os.path.join(save_dir, f"separator_accuracy_distribution_{optimization_mode}_{seed}_{dataset_name}.png")
        plt.savefig(plot_path)
        plt.show()
        print(f"Saved histogram to {plot_path}")

        # --- Save to CSV ---
        csv_path = os.path.join(save_dir, "separator_accuracy_distribution.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved results to {csv_path}")


    # Evaluate top separators on test set
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


    #top_separator_texts = [entry['separator'] for entry in top_separators]
    avg_test_score = sum(top_test_score) / len(top_test_score)
    if is_generation_task:
        if dataset_name == "asset":
            print(f"average test SARI: {avg_test_score:.3f}")
        else:
            print(f"average test ROUGE-L F1: {avg_test_score:.3f}")
    else:
        print(f"average test accuracy: {avg_test_score:.3f}")
 

    # --- Summary stats ---

    # START OF COMMENTING OUT ON 6 JULY
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

    # Save to CSV
    summary_dir = f"separator_accuracy_distribution_{model_name}"
    os.makedirs(summary_dir, exist_ok=True)
    summary_path = os.path.join(summary_dir, f"summary_stats_{optimization_mode}_{seed}_{dataset_name}.csv")
    summary_df.to_csv(summary_path, index=False)

    print(f"\nSaved summary statistics to {summary_path}")


    
    
    safe_model_name = model_name.replace("/", "__")
    log_dir = f"separator_logs_{safe_model_name}"
    os.makedirs(log_dir, exist_ok=True)

    separator_log_df = pd.DataFrame(top_k_entries)
    if is_generation_task:
        fname = f"separator_log_{dataset_name}_{optimization_mode}_{seed}_{safe_model_name}_rouge.csv"
    else:
        fname = f"separator_log_{dataset_name}_{optimization_mode}_{seed}_{safe_model_name}_acc.csv"

    file_path = os.path.join(log_dir, fname)
    separator_log_df = pd.DataFrame(top_k_entries)
    separator_log_df.to_csv(file_path, index=False)
    print(f"Separator log saved to: {fname}")

    # END OF COMMENTING OUT ON 6 JULY
    

    test_separator_stability(top_k_entries, model, tokenizer, context, test_data, optimization_mode=optimization_mode, is_generation_task=is_generation_task, dataset_name=dataset_name, batch_size=batch_size, max_new_tokens=max_new_tokens, max_length=768, cal_sari=cal_sari, cal_rouge=cal_rouge, tokenizer_moses=tokenizer_moses, rouge=rouge)


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