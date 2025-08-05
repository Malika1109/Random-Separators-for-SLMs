import torch
import random
import argparse
import transformers
from evolutionary import evolutionary_search
from codecarbon import EmissionsTracker
from evolutionary_pos import evolutionary_search as evolutionary_search_pos, split_vocab
from pos import generate_multiple_separators
#from greater_hybrid_ea import greater_hybrid_search
from clustered_lhs import clustered_lhs_separators
from lhs_with_and_without_context import generate_lhs_vocab_separator, generate_lhs_wo_context_separator
from sampling_poisson_disk import poissondisk_separators
from sampling_halton import halton_separators
from sampling_sobol import sobol_separators
from bayesian import BayesianSeparatorOptimizer

import optuna
import utils

transformers.logging.set_verbosity_error()


generator = transformers.pipeline('text-generation', model='gpt2')

def hf_generate(prompt: str, num_return: int = 3):
    outputs = generator(
        prompt,
        num_return_sequences=num_return,
        max_new_tokens=15
    )
    completions = [out['generated_text'][len(prompt):].strip() for out in outputs]
    return [c.split('\n')[0].strip() for c in completions if c.strip()]


def evaluate_separator(separator_text, model, context, train_data):
    text_sequences = [context + "\n\n" + train_instance["prompt"] for train_instance in train_data]
    text_sequences = [elem.replace("{separator}", separator_text) for elem in text_sequences]
    labels = [train_instance["output"] for train_instance in train_data]
    r = model(text_sequences, max_new_tokens=1, return_full_text=False, do_sample=False, batch_size=16)
    predictions = [elem[0]['generated_text'].strip() for elem in r]
    accuracy = compute_accuracy(predictions=predictions, labels=labels)
    return accuracy


def evaluate_special_separator(separator_text, model, tokenizer, context, train_data):
    # Construct full input sequences by inserting the separator
    text_sequences = [context + "\n\n" + train_instance["prompt"] for train_instance in train_data]
    text_sequences = [elem.replace("{separator}", separator_text) for elem in text_sequences]
    labels = [train_instance["output"] for train_instance in train_data]

    # Tokenize input for generation
    inputs = tokenizer(
        text_sequences,
        return_tensors='pt',
        padding=True,
        truncation=True
    ).to(model.device)

    # Generate predictions
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1,
            return_dict_in_generate=True,
            output_scores=True,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id  # avoids warning
        )

    # Decode generated tokens
    generated_tokens = outputs.sequences[:, inputs['input_ids'].shape[1]:]  # only new tokens
    predictions = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    predictions = [p.strip() for p in predictions]

    # Evaluate accuracy
    accuracy = compute_accuracy(predictions=predictions, labels=labels)
    return accuracy

def debug_evaluate_separator(separator_text, model, tokenizer, context, train_data, num_examples=3):
    print("="*60)
    print(f"🔍 Evaluating separator: {repr(separator_text)}")
    
    # Inject separator into prompts
    text_sequences = [
        context + "\n\n" + instance["prompt"].replace("{separator}", separator_text)
        for instance in train_data
    ]
    labels = [instance["output"] for instance in train_data]

    print("\n🧾 Sample Prompt After Insertion:")
    print(repr(text_sequences[0]))

    # Tokenize input
    enc = tokenizer(
        text_sequences,
        return_tensors='pt',
        padding=True,
        truncation=True
    ).to(model.device)

    print("\n🧱 Tokenized input shape:", enc['input_ids'].shape)

    # Generate output
    with torch.no_grad():
        outputs = model.generate(
            **enc,
            max_new_tokens=1,
            return_dict_in_generate=True,
            output_scores=True,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode generated tokens (just the new token)
    gen_tokens = outputs.sequences[:, enc['input_ids'].shape[1]:]
    predictions = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
    predictions = [p.strip() for p in predictions]

    print("\n🧠 Predictions vs Labels:")
    for i in range(min(num_examples, len(predictions))):
        print(f"[{i}] Pred: '{predictions[i]}'  |  Label: '{labels[i]}'")

    # Simple accuracy
    correct = sum(p.lower() == l.lower() for p, l in zip(predictions, labels))
    acc = correct / len(predictions)
    
    print(f"\n📈 Accuracy = {acc:.4f}")
    print("="*60 + "\n")

    return acc



def compute_accuracy(predictions, labels):
    correct = 0
    for pred, label in zip(predictions, labels):
        if pred == label:
            correct += 1
    return correct / len(labels)


def generate_random_vocab_separator(model, min_separator_length, max_separator_length):
    vocabulary_size = model.tokenizer.vocab_size
    separator_length = random.randint(min_separator_length, max_separator_length)
    separator_ids = random.sample(range(vocabulary_size), separator_length)
    separator_text = model.tokenizer.decode(separator_ids)
    return separator_text


def generate_random_wo_context_separator(model, min_separator_length, max_separator_length):
    separator_text = \
    model(text_inputs="", do_sample=True, max_new_tokens=random.randint(min_separator_length, max_separator_length))[0][
        'generated_text']
    return separator_text


def main(args):
    dataset_name = args.dataset
    model_name = args.model
    corpus_size = args.corpus_size
    context_shot_size = args.context_shot_size
    min_separator_length = args.min_separator_length
    max_separator_length = args.max_separator_length
    num_random_draw = args.num_random_draw
    optimization_mode = args.optimization_mode

    train_corpus = utils.load_data(
        dataset_name, path=f"data/{dataset_name}/train.jsonl"
    )  # we will use train corpus to optimize the separator

    test_corpus = utils.load_data(
        dataset_name, path=f"data/{dataset_name}/dev_subsample.jsonl"
    )

    context_corpus = utils.load_data(
        dataset_name, path=f"data/{dataset_name}/train_full.jsonl"  #INTERCHANGING TRAIN AND TRAIN_FULL
    )  # a separate context corpus to construct the context

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = transformers.pipeline("text-generation", model=model_name, device=device)

    model.tokenizer.pad_token_id = model.model.config.eos_token_id
    model.tokenizer.padding_side = "left"

    context_examples = utils.balanced_sample(
        context_corpus, n_shot=context_shot_size
    )

    random.shuffle(context_examples)

    train_data = [
        train_corpus.__getitem__(i, include_output=False)
        for i in range(len(train_corpus))
    ]
    train_data = train_data[:corpus_size]

    test_data = [
        test_corpus.__getitem__(i, include_output=False)
        for i in range(len(test_corpus))
    ]

    context = "\n\n".join([elem["prompt"] for elem in context_examples])

    #tracker = EmissionsTracker(project_name=f"{optimization_mode}_separator_search")
    #tracker.start()

    if optimization_mode == "random_vocab":
        random_separator_texts = [generate_random_vocab_separator(model, min_separator_length, max_separator_length) for _ in range(num_random_draw)]
    elif optimization_mode == "random_wo_context":
        random_separator_texts = [generate_random_wo_context_separator(model, min_separator_length, max_separator_length) for _ in range(num_random_draw)]
    elif optimization_mode == "evolutionary":
        # instead of random sampling, do evolutionary search
        from transformers import AutoTokenizer
        tokenizer = model.tokenizer
        vocab = [token for token in tokenizer.get_vocab().keys() if token.isalpha()]  # simple vocab filtering
        
        def objective(trial):
            mutation_rate = trial.suggest_float("mutation_rate", 0.1, 0.7)
            pop_size = trial.suggest_int("pop_size", 8, 20)
            insert_prob = trial.suggest_float("insert_prob", 0.05, 0.2)
            delete_prob = trial.suggest_float("delete_prob", 0.05, 0.2)

            best_separator = evolutionary_search(
                model=model,
                evaluate_fn=lambda sep: evaluate_separator(sep, model, context, train_data),
                vocab=vocab,
                generations=5,
                pop_size=pop_size,
                min_sep_len=args.min_separator_length,
                max_sep_len=args.max_separator_length,
                mutation_rate=mutation_rate,
                insert_prob=insert_prob,
                delete_prob=delete_prob
            )
            val_acc = evaluate_separator(best_separator, model, context, test_data)  # use test set to evaluate fitness
            return val_acc  # maximize accuracy!

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=15)  # try 15 different hyperparameter combos

        print("Best hyperparameters:", study.best_params)
        
        # Now retrain evolutionary search with best hyperparameters
        best_params = study.best_params
        best_separator = evolutionary_search(
            model=model,
            evaluate_fn=lambda sep: evaluate_separator(sep, model, context, train_data),
            vocab=vocab,
            generations=5,
            pop_size=best_params["pop_size"],
            min_sep_len=args.min_separator_length,
            max_sep_len=args.max_separator_length,
            mutation_rate=best_params["mutation_rate"],
            insert_prob=best_params["insert_prob"],
            delete_prob=best_params["delete_prob"]
        )
        random_separator_texts = [best_separator]  # wrap into list

    elif optimization_mode == "bayesian":
        optimizer = BayesianSeparatorOptimizer(model, model.tokenizer, min_separator_length, 3)
        best_separator = optimizer.optimize(context, train_data[:corpus_size], n_iter=num_random_draw)
        random_separator_texts = [best_separator["separator"]]  # Wrap in list
    elif optimization_mode == "latin_hypercube":
        from latin_hypercube import lhs_embedding_separators
        from transformers import AutoTokenizer, AutoModelForCausalLM

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"

        model = AutoModelForCausalLM.from_pretrained(model_name)
        vocab = [token for token in tokenizer.get_vocab().keys() if token.isalpha()]
        random_separator_texts = lhs_embedding_separators(
            tokenizer=tokenizer,
            model=model,
            sep_len=3,
            n_samples=num_random_draw
        )
    elif optimization_mode == "evolutionary_pos":
        # Evolutionary search with structured separators
        tokenizer = model.tokenizer
        full_vocab = [token for token in tokenizer.get_vocab().keys()]
        words, punctuations = split_vocab(full_vocab)

        best_separator = evolutionary_search_pos(
            model=model,
            evaluate_fn=lambda sep: evaluate_separator(sep, model, context, train_data),
            words=words,
            punctuations=punctuations,
            generations=5,
            pop_size=10,
            min_sep_len=args.min_separator_length,
            max_sep_len=args.max_separator_length,
            mutation_rate=0.3,
            template_type="word_word"  # can switch to "word_punct" if you want
        )
        random_separator_texts = [best_separator]  # wrap into list
    elif optimization_mode == "pos":
    # POS/Structure-Constrained Separator Generation
        random_separator_texts = generate_multiple_separators(
            num_samples=num_random_draw, 
            template="adj_noun_punct"  # can also try "noun_verb" or "the_noun_verbing"
        )
    elif optimization_mode == "guided_evolutionary":
        from guided_evolutionary import POSAwareSeparatorEvolution
        from transformers import AutoTokenizer, AutoModelForCausalLM

        # Load tokenizer + model properly
        #tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"

        model = AutoModelForCausalLM.from_pretrained(model_name)
        #model_raw.eval()
        #model_raw.to(device)
        evaluate_fn = lambda sep: evaluate_special_separator(sep, model, tokenizer, context, train_data)

        # Initialize with your model and tokenizer
        evolver = POSAwareSeparatorEvolution(
            model=model,
            tokenizer=tokenizer,
            evaluate_fn=evaluate_fn,
            pos_structure=["ADJ", "NOUN", "PUNCT"]  
            #llm_generate_fn=hf_generate  # Optional LLM interface
        )

        # Run evolution
        best_sep, best_score = evolver.evolve(
            generations=10,
            pop_size=20
        )
        random_separator_texts = [best_sep]  # wrap into list

    elif optimization_mode == "clustered_lhs":
        tokenizer = model.tokenizer
        random_separator_texts = clustered_lhs_separators(
            tokenizer=tokenizer,
            model=model,
            sep_len=3,
            n_samples=num_random_draw,
            n_clusters=50
        )
    elif optimization_mode == "halton":
        tokenizer = model.tokenizer
        vocab = [token for token in tokenizer.get_vocab().keys() if token.isalpha()]
        random_separator_texts = halton_separators(
            model=model,
            min_separator_length=1,
            max_separator_length=3,
            n_samples=num_random_draw
        )
    elif optimization_mode == "poisson_disk":
        tokenizer = model.tokenizer
        vocab = [token for token in tokenizer.get_vocab().keys() if token.isalpha()]
        random_separator_texts = poissondisk_separators(
            vocab=vocab,
            sep_len=3,
            n_samples=num_random_draw
        )
    elif optimization_mode == "sobol":
        tokenizer = model.tokenizer
        vocab = [token for token in tokenizer.get_vocab().keys() if token.isalpha()]
        random_separator_texts = sobol_separators(
            model=model,
            min_separator_length=1,
            max_separator_length=3,
            n_samples=num_random_draw
        )
        
    elif optimization_mode == "lhs_wo_context":
        random_separator_texts = generate_lhs_wo_context_separator(
            model, 1,5, num_random_draw
        )
    elif optimization_mode == "experiment":
        from experiment import metropolis_hastings_separators
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"

        model = AutoModelForCausalLM.from_pretrained(model_name)
        evaluate_fn = lambda sep: evaluate_special_separator(sep, model, tokenizer, context, train_data)

        random_separator_texts = metropolis_hastings_separators(
        tokenizer=tokenizer,
        evaluate_fn=evaluate_fn,
        n_samples=num_random_draw,
        )

    else:
        raise NotImplementedError
    
    #emissions: float = tracker.stop()
    #print(f"Carbon emissions for {optimization_mode}: {emissions:.6f} kg CO₂")

    if optimization_mode != "guided_evolutionary" and optimization_mode != "latin_hypercube" and optimization_mode != "experiment":
        if model_name in ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]:
            model.tokenizer.pad_token_id = model.tokenizer.eos_token_id
        model.tokenizer.padding_side = "left"

    print("running random separator search over train set...")
    separator_search_result = []
    for i, separator_text in enumerate(random_separator_texts):
        text_sequences = [context + "\n\n" + train_instance["prompt"] for train_instance in train_data]
        text_sequences = [elem.replace("{separator}", separator_text) for elem in text_sequences]
        labels = [train_instance["output"] for train_instance in train_data]
        if optimization_mode != "guided_evolutionary" and optimization_mode != "latin_hypercube" and optimization_mode != "experiment":
            r = model(text_sequences, max_new_tokens=1, return_full_text=False, do_sample=False, batch_size=16)
            predictions = [elem[0]['generated_text'].strip() for elem in r]
        else:
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
        print(f"{i + 1}/{num_random_draw} - accuracy: {accuracy}, separator: {repr(separator_text)}")
        separator_search_result.append({"separator": separator_text,
                                        "accuracy": accuracy})

    separator_search_result = sorted(separator_search_result, key=lambda x: x["accuracy"], reverse=True)

    average_accuracy = sum([elem['accuracy'] for elem in separator_search_result]) / len(separator_search_result)
    print(f"average train accuracy: {average_accuracy}")

    # select top 4 separators
    top_separators = separator_search_result[:4]
    top_accuracy = []
    for i, separator in enumerate(top_separators):
        train_accuracy = separator['accuracy']
        separator_text = separator['separator']
        text_sequences = [context + "\n\n" + test_instance["prompt"] for test_instance in test_data]
        text_sequences = [elem.replace("{separator}", separator_text) for elem in text_sequences]
        labels = [test_instance["output"] for test_instance in test_data]
        if optimization_mode != "guided_evolutionary" and optimization_mode != "latin_hypercube" and optimization_mode != "experiment":
            r = model(text_sequences, max_new_tokens=1, return_full_text=False, do_sample=False, batch_size=16)
            predictions = [elem[0]['generated_text'].strip() for elem in r]
        else:
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
        print(
            f"top {i + 1} separator: {repr(separator_text)} - accuracy: {accuracy} - train accuracy: {train_accuracy}")
        top_accuracy.append(accuracy)
    print(f"average test accuracy: {sum(top_accuracy) / len(top_accuracy)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--dataset", type=str, default="sst2")
    parser.add_argument("--model", type=str, default="gpt2-large")
    parser.add_argument(
        "--min_separator_length", type=int, default=1
    )
    parser.add_argument(
        "--max_separator_length", type=int, default=5
    )
    parser.add_argument("--num_random_draw", type=int, default=160)
    parser.add_argument(
        "--context_shot_size", type=int, default=4
    )  # ICL demonstration, context_shot_size is the number of shots we use to construct the balanced context

    parser.add_argument("--corpus_size", type=int, default=64)

    parser.add_argument(
        "--optimization_mode",
        choices=[
            "random_vocab",
            "random_wo_context",
            "evolutionary",
            "evolutionary_pos",
            "bayesian",
            "pos",
            "latin_hypercube",
            "greater_hybrid",
            "clustered_lhs",
            "lhs_wo_context",
            "poisson_disk",
            "halton",
            "sobol",
            "guided_evolutionary",
            "experiment"
        ],
        default="random_wo_context",
    )

    args = parser.parse_args()

    utils.set_random_seed(args.seed)
    main(args) how to integrate SmolLM2