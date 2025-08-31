import random
import nltk
from nltk.corpus import wordnet as wn

# Ensure these are downloaded before use
# nltk.download('wordnet')
# nltk.download('omw-1.4')

"""
This script generates random separators from simple part-of-speech–based templates 
using WordNet synsets (nouns, verbs, adjectives, adverbs, punctuation).  

Each separator is created by sampling random words from the specified POS category 
and assembling them according to a template (e.g., "adj_noun_punct" → "happy dog,").  

This can be integrated into the main separator search framework (`main.py`) 
as an additional optimization mode (e.g., `--optimization_mode template_based`) 
to evaluate template-driven random separators alongside other strategies.  
"""

def set_seed(seed: int = 42):
    random.seed(seed)

def get_random_noun(rng):
    nouns = list(wn.all_synsets('n'))
    synset = rng.choice(nouns)
    return rng.choice(synset.lemma_names()).replace('_', ' ')

def get_random_verb(rng):
    verbs = list(wn.all_synsets('v'))
    synset = rng.choice(verbs)
    return rng.choice(synset.lemma_names()).replace('_', ' ')

def get_random_adjective(rng):
    adjectives = list(wn.all_synsets('a'))
    synset = rng.choice(adjectives)
    return rng.choice(synset.lemma_names()).replace('_', ' ')

def get_random_adverb(rng):
    adverbs = list(wn.all_synsets('r'))
    synset = rng.choice(adverbs)
    return rng.choice(synset.lemma_names()).replace('_', ' ')

def get_random_punctuation(rng):
    return rng.choice([",", ":"])

def generate_separator_from_template(template="adj_noun_punct", rng=None):
    if rng is None:
        rng = random

    if template == "adj_noun_punct":
        separator = f"{get_random_adjective(rng)} {get_random_noun(rng)}{get_random_punctuation(rng)}"
    elif template == "noun_noun_punct":
        separator = f"{get_random_noun(rng)} {get_random_noun(rng)}{get_random_punctuation(rng)}"
    elif template == "noun_noun":
        separator = f"{get_random_noun(rng)} {get_random_noun(rng)}"
    elif template == "verb_verb":
        separator = f"{get_random_verb(rng)} {get_random_verb(rng)}"
    elif template == "verb":
        separator = f"{get_random_verb(rng)}"
    elif template == "noun":
        separator = f"{get_random_noun(rng)}"
    elif template == "adj":
        separator = f"{get_random_adjective(rng)}"
    elif template == "noun_punct":
        separator = f"{get_random_noun(rng)}{get_random_punctuation(rng)}"
    elif template == "adv_verb":
        separator = f"{get_random_adverb(rng)} {get_random_verb(rng)}"
    elif template == "verb_punct":
        separator = f"{get_random_verb(rng)}{get_random_punctuation(rng)}"
    elif template == "noun_verb":
        separator = f"{get_random_noun(rng)} {get_random_verb(rng)}"
    elif template == "verb_verb_punct":
        separator = f"{get_random_adverb(rng)} {get_random_verb(rng)} {get_random_punctuation(rng)}"
    elif template == "adj_adv_verb_punct":
        separator = f"{get_random_adjective(rng)} {get_random_adverb(rng)} {get_random_verb(rng)}{get_random_punctuation(rng)}"
    elif template == "the_noun_verbing":
        noun = get_random_noun(rng)
        verb = get_random_verb(rng)
        separator = f"The {noun} {verb}ing"  # still naive 'ing'
    elif template == "noun_and_noun_verb_punct":
        separator = f"{get_random_noun(rng)} and {get_random_noun(rng)} {get_random_verb(rng)}{get_random_punctuation(rng)}"
    elif template == "verb_the_adj_noun_punct":
        separator = f"{get_random_verb(rng)} the {get_random_adjective(rng)} {get_random_noun(rng)}{get_random_punctuation(rng)}"
    elif template == "noun_verb_adj_noun_punct":
        separator = f"{get_random_noun(rng)} {get_random_verb(rng)} {get_random_adjective(rng)} {get_random_noun(rng)}{get_random_punctuation(rng)}"
    else:
        raise ValueError(f"Unknown template: {template}")
    return separator

def generate_multiple_separators(num_samples=10, template="adj_noun_punct", seed=42):
    rng = random.Random(seed)
    return [generate_separator_from_template(template, rng=rng) for _ in range(num_samples)]
