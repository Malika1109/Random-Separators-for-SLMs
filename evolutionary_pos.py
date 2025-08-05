# evolutionary_pos.py
#TODO: Hyperparameter tuning

import random

def split_vocab(vocab):
    words = [token for token in vocab if token.isalpha()]
    punctuations = [token for token in vocab if token in {".", ",", "!", "?", ";", ":"}]
    return words, punctuations

def generate_structured_separator(words, punctuations, template_type="word_word"):
    if template_type == "word_punct":
        word = random.choice(words)
        punct = random.choice(punctuations)
        return f"{word} {punct}"
    elif template_type == "word_word":
        word1 = random.choice(words)
        word2 = random.choice(words)
        return f"{word1} {word2}"
    else:
        raise ValueError(f"Unknown template type: {template_type}")

def generate_initial_population(pop_size, words, punctuations, template_type="word_word"):
    population = []
    for _ in range(pop_size):
        separator = generate_structured_separator(words, punctuations, template_type)
        population.append(separator)
    return population

def mutate(separator, words, punctuations, mutation_rate=0.3, insert_prob=0.1, delete_prob=0.1, min_sep_len=1, max_sep_len=5, template_type="word_word"):
    new_sep = []
    tokens = separator.split()

    for token in tokens:
        if random.random() < mutation_rate:
            if template_type == "word_punct" and random.random() < 0.5:
                new_sep.append(random.choice(punctuations))
            else:
                new_sep.append(random.choice(words))
        else:
            new_sep.append(token)

    # Random insert
    if random.random() < insert_prob and len(new_sep) < max_sep_len:
        insert_position = random.randint(0, len(new_sep))
        if template_type == "word_punct" and random.random() < 0.5:
            new_token = random.choice(punctuations)
        else:
            new_token = random.choice(words)
        new_sep = new_sep[:insert_position] + [new_token] + new_sep[insert_position:]

    # Random delete
    if random.random() < delete_prob and len(new_sep) > min_sep_len:
        delete_position = random.randint(0, len(new_sep) - 1)
        new_sep = new_sep[:delete_position] + new_sep[delete_position + 1:]

    return " ".join(new_sep)

def crossover(parent1, parent2):
    parent1_tokens = parent1.split()
    parent2_tokens = parent2.split()
    min_len = min(len(parent1_tokens), len(parent2_tokens))
    if min_len <= 1:
        return random.choice([parent1, parent2])
    point = random.randint(1, min_len - 1)
    return " ".join(parent1_tokens[:point] + parent2_tokens[point:])

def evolutionary_search(model, evaluate_fn, words, punctuations, generations=10, pop_size=10, min_sep_len=1, max_sep_len=3, mutation_rate=0.3, template_type="word_word"):
    population = generate_initial_population(pop_size, words, punctuations, template_type)

    for gen in range(generations):
        scores = []
        for separator in population:
            acc = evaluate_fn(separator)
            scores.append((acc, separator))
        scores.sort(reverse=True)
        survivors = [sep for acc, sep in scores[:pop_size//2]]

        new_population = []
        for _ in range(pop_size):
            parent1 = random.choice(survivors)
            parent2 = random.choice(survivors)
            child = crossover(parent1, parent2)
            child = mutate(child, words, punctuations, mutation_rate, min_sep_len=min_sep_len, max_sep_len=max_sep_len, template_type=template_type)
            new_population.append(child)

        population = new_population

    best_separator = max(population, key=lambda sep: evaluate_fn(sep))
    return best_separator