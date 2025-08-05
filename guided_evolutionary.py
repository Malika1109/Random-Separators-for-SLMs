# token_aware_evolution.py
# SeparatorEvolutionV2

import torch
import random
import numpy as np
from typing import List, Callable, Tuple
from transformers import PreTrainedTokenizer, PreTrainedModel

# guided_separator_evolution.py (full EvoPrompt-style for separators)

import os

# Set random seeds for reproducibility
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)


import nltk
from nltk.corpus import wordnet


#nltk.download('wordnet')
#nltk.download('omw-1.4')
#nltk.download('averaged_perceptron_tagger_eng')

import random
from typing import List, Tuple, Callable
from transformers import PreTrainedModel, PreTrainedTokenizer
import nltk
from nltk.corpus import wordnet
from nltk import pos_tag, word_tokenize

#nltk.download('wordnet')
#nltk.download('omw-1.4')
#nltk.download('averaged_perceptron_tagger')




class POSAwareSeparatorEvolution:
    def __init__(self, 
                 model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizer,
                 evaluate_fn: Callable[[str], float],
                 pos_structure):
        self.model = model
        self.tokenizer = tokenizer
        self.evaluate_fn = evaluate_fn
        self.embedding_matrix = model.get_input_embeddings().weight.detach()
        self.evaluated_cache = {}
        self.pos_structure = pos_structure
        self.pos_vocab = self.build_pos_vocab()

    def build_pos_vocab(self) -> dict:
        print("here")
        words = [w.replace('_', ' ') for w in list(set(wordnet.words()))[:50000]]
        tagged = pos_tag(words)
        pos_vocab = {
            "ADJ": [], "NOUN": [], "VERB": [], "ADV": [],
            "PUNCT": [".", "!", ",", ":", ";"]
        }
        for word, tag in tagged:
            if tag.startswith("JJ"):
                pos_vocab["ADJ"].append(word)
            elif tag.startswith("NN"):
                pos_vocab["NOUN"].append(word)
            elif tag.startswith("VB"):
                pos_vocab["VERB"].append(word)
            elif tag.startswith("RB"):
                pos_vocab["ADV"].append(word)
        return pos_vocab

    def is_valid_pos_sequence(self, text: str, target_pos: List[str]) -> bool:
        print("here2")
        tagged = pos_tag(word_tokenize(text))
        tag_map = {"JJ": "ADJ", "NN": "NOUN", "NNS": "NOUN", "VB": "VERB", "RB": "ADV"}
        simplified = []
        for _, tag in tagged:
            for prefix, simple in tag_map.items():
                if tag.startswith(prefix):
                    simplified.append(simple)
                    break
            else:
                simplified.append("PUNCT" if tag in [".", ",", ":", ";", "!"] else "OTHER")
        return simplified == target_pos

    def initialize_population(self, pop_size: int) -> List[List[int]]:
        print("here3")
        population = []
        while len(population) < pop_size:
            tokens = []
            for pos in self.pos_structure:
                tokens.append(random.choice(self.pos_vocab.get(pos, ["UNK"])))
            text = " ".join(tokens)
            if self.is_valid_pos_sequence(text, self.pos_structure):
                ids = self.tokenizer.encode(text, add_special_tokens=False)
                population.append(ids)
        return population

    def mutate(self, indiv: List[int]) -> List[int]:
        print("here4")
        words = self.tokenizer.convert_ids_to_tokens(indiv)
        mutated = []
        for i, pos in enumerate(self.pos_structure):
            if i >= len(words):
                replacement = random.choice(self.pos_vocab.get(pos, ["UNK"]))
            elif random.random() < 0.3:
                replacement = random.choice(self.pos_vocab.get(pos, [words[i]]))
            else:
                replacement = words[i]
            mutated.append(replacement)
        mutated_text = " ".join(mutated)
        if not self.is_valid_pos_sequence(mutated_text, self.pos_structure):
            return indiv  # fallback to original
        return self.tokenizer.encode(mutated_text, add_special_tokens=False)

    def crossover(self, p1: List[int], p2: List[int]) -> List[int]:
        print("here5")
        if not p1 or not p2:
            return p1.copy()
        child_tokens = []
        tokens1 = self.tokenizer.convert_ids_to_tokens(p1)
        tokens2 = self.tokenizer.convert_ids_to_tokens(p2)
        for i in range(len(self.pos_structure)):
            word1 = tokens1[i] if i < len(tokens1) else ""
            word2 = tokens2[i] if i < len(tokens2) else ""
            choice = random.choice([word1, word2])
            if choice == "":
                choice = random.choice(self.pos_vocab[self.pos_structure[i]])
            child_tokens.append(choice)
        child_text = " ".join(child_tokens)
        if not self.is_valid_pos_sequence(child_text, self.pos_structure):
            return p1
        return self.tokenizer.encode(child_text, add_special_tokens=False)

    def evaluate_separator(self, separator_ids: List[int]) -> float:
        print("here6")
        key = tuple(separator_ids)
        if key in self.evaluated_cache:
            return self.evaluated_cache[key]

        text = self.tokenizer.decode(separator_ids)
        if not self.is_valid_pos_sequence(text, self.pos_structure):
            return 0.0

        base_score = self.evaluate_fn(text)

        # Perturbation robustness: change one token per copy
        perturbed_scores = []
        words = text.split()
        for i, pos in enumerate(self.pos_structure):
            if pos not in self.pos_vocab or not self.pos_vocab[pos]:
                continue
            perturbed = words.copy()
            perturbed[i] = random.choice(self.pos_vocab[pos])
            perturbed_text = " ".join(perturbed)
            if self.is_valid_pos_sequence(perturbed_text, self.pos_structure):
                score = self.evaluate_fn(perturbed_text)
                perturbed_scores.append(score)

        avg_perturbed_score = sum(perturbed_scores) / len(perturbed_scores) if perturbed_scores else base_score

        final_score = 0.6 * base_score + 0.6 * avg_perturbed_score
        self.evaluated_cache[key] = final_score
        return final_score

    def evolve(self, generations: int = 10, pop_size: int = 20, elite_size: int = 4) -> Tuple[str, float]:
        print("here7")
        population = self.initialize_population(pop_size)
        for gen in range(generations):
            scored = [(self.evaluate_separator(indiv), indiv) for indiv in population]
            scored.sort(key=lambda x: x[0], reverse=True)
            best_score, best_indiv = scored[0]
            print(f"Gen {gen+1}: Best={best_score:.3f}, Separator='{self.tokenizer.decode(best_indiv)}'")
            elites = [indiv for _, indiv in scored[:elite_size]]
            new_population = elites.copy()
            while len(new_population) < pop_size:
                p1, p2 = random.sample(elites, 2)
                child = self.crossover(p1, p2)
                child = self.mutate(child)
                new_population.append(child)
            population = new_population
        final_best_key = max(self.evaluated_cache.items(), key=lambda x: x[1])[0]
        return self.tokenizer.decode(final_best_key), self.evaluated_cache[final_best_key]



# this one is very good, above one is same, just minor code changes for different separators across
# gen which didnt work
""" class POSAwareSeparatorEvolution:
    def __init__(self, 
                 model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizer,
                 evaluate_fn: Callable[[str], float]):
        self.model = model
        self.tokenizer = tokenizer
        self.evaluate_fn = evaluate_fn
        self.embedding_matrix = model.get_input_embeddings().weight.detach()
        self.evaluated_cache = {}
        self.pos_structure = ["NOUN", "VERB"]
        self.pos_vocab = self.build_pos_vocab()

    def build_pos_vocab(self) -> dict:
        print("Building POS vocabulary...")
        words = list(wordnet.words())[:10000]  # Limit for speed
        pos_vocab = {"ADJ": [], "NOUN": [], "VERB": [], "ADV": [], "PUNCT": [".", "!", ",", ":", ";"]}
        tagged = nltk.pos_tag(words)

        for word, tag in tagged:
            if tag.startswith('JJ'):
                pos_vocab["ADJ"].append(word)
            elif tag.startswith('NN'):
                pos_vocab["NOUN"].append(word)
            elif tag.startswith('VB'):
                pos_vocab["VERB"].append(word)
            elif tag.startswith('RB'):
                pos_vocab["ADV"].append(word)

        return pos_vocab

    def initialize_population(self, pop_size: int) -> List[List[int]]:
        print("Initializing population...")
        population = []
        for _ in range(pop_size):
            tokens = ["The"]
            for pos in self.pos_structure:
                candidates = self.pos_vocab.get(pos, ["-"])
                tokens.append(random.choice(candidates))
            text = " ".join(tokens)
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            population.append(ids)
        return population

    def mutate(self, indiv: List[int]) -> List[int]:
        print("Mutating individual...")
        tokens = self.tokenizer.convert_ids_to_tokens(indiv)
        mutated_tokens = [tokens[0]]  # Preserve "The"
        for i, token in enumerate(tokens[1:], start=1):
            if random.random() < 0.3:
                pos = self.pos_structure[(i - 1) % len(self.pos_structure)]
                replacement = random.choice(self.pos_vocab.get(pos, [token]))
                mutated_tokens.append(replacement)
            else:
                mutated_tokens.append(token)
        mutated_text = " ".join(mutated_tokens)
        return self.tokenizer.encode(mutated_text, add_special_tokens=False)

    def crossover(self, p1: List[int], p2: List[int]) -> List[int]:
        print("Crossover between two individuals...")
        if len(p1) < 2 or len(p2) < 2:
            return p1.copy()
        cut1 = random.randint(1, len(p1) - 1)
        cut2 = random.randint(1, len(p2) - 1)
        return p1[:cut1] + p2[cut2:]

    def evaluate_separator(self, separator_ids: List[int]) -> float:
        print("Evaluating separator...")
        key = tuple(separator_ids)
        if key in self.evaluated_cache:
            return self.evaluated_cache[key]
        text = self.tokenizer.decode(separator_ids)
        score = self.evaluate_fn(text)
        self.evaluated_cache[key] = score
        return score

    def evolve(self, generations: int = 10, pop_size: int = 20, elite_size: int = 4) -> Tuple[str, float]:
        print("Evolving population...")
        population = self.initialize_population(pop_size)
        for gen in range(generations):
            scored = [(self.evaluate_separator(indiv), indiv) for indiv in population]
            scored.sort(key=lambda x: x[0], reverse=True)
            elites = [indiv for _, indiv in scored[:elite_size]]

            new_population = elites.copy()
            while len(new_population) < pop_size:
                p1 = random.choice(elites)
                p2 = random.choice(elites)
                child = self.crossover(p1, p2)
                child = self.mutate(child)
                new_population.append(child)
            population = new_population

            best_score, best_indiv = scored[0]
            print(f"Gen {gen+1}: Best={best_score:.3f}, Separator='{self.tokenizer.decode(best_indiv)}'")

        final_best_key = max(self.evaluated_cache.items(), key=lambda x: x[1])[0]
        return self.tokenizer.decode(final_best_key), self.evaluated_cache[final_best_key]
 """