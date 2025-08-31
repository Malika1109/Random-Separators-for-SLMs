import json

# ====================================
# Base class for generation datasets
# ====================================
class TextGenerativeDataset:
    """
    Generic dataset class for generation tasks (e.g., summarization, simplification).

    Each line in the input JSONL file should contain fields for:
        - input_text (e.g., dialogue or original sentence)
        - output_text (e.g., summary or simplification)

    Args:
        path (str): path to the dataset (JSONL file).
        prompt_template (str): template used to construct prompts. 
                               Should contain placeholders: {input_text}, {separator}, {output_text}.
    """
    def __init__(self, path, prompt_template="{input_text} {separator} {output_text}"):
        self.task_type = "generation"
        self.path = path
        self.data = self.load_data(path)
        self.prompt_template = prompt_template

    def load_data(self, path):
        with open(path, "r") as f:
            data = [json.loads(line) for line in f]
        return data

    def __len__(self):
        return len(self.data)

    def make_prompt(self, input_text, output_text):
        return self.prompt_template.format(
            input_text=input_text, separator="{separator}", output_text=output_text
        )

    def __getitem__(self, idx, include_output=False):
        """
        Retrieve a dataset instance.

        Args:
            idx (int): index of the instance
            include_output (bool): whether to include the gold output in the prompt

        Returns:
            dict with keys:
                - "prompt": formatted text with {separator} placeholder
                - "output": gold reference output
        """
        instance = self.data[idx]
        input_text = instance['dialogue'].strip()
        output_text = instance['summary'].strip()

        if include_output:
            prompt = self.make_prompt(input_text, output_text).strip()
        else:
            prompt = self.make_prompt(input_text, "").strip()
        return {"prompt": prompt, "output": output_text}


class SAMSumDataset(TextGenerativeDataset):
    def __init__(self, path, prompt_template="{input_text} {separator} {output_text}"):
        super().__init__(path, prompt_template)
        # Currently, placeholder


class ASSETDataset(TextGenerativeDataset):
    def __init__(self, path, prompt_template="{input_text} {separator} {output_text}"):
        super().__init__(path, prompt_template)

    def __getitem__(self, idx, include_output=False):
        instance = self.data[idx]
        input_text = instance['original'].strip()
        reference_list = [ref.strip() for ref in instance['references']]

        if include_output:
            # Use one reference for prompt construction — first one by default: can also change to random
            prompt = self.make_prompt(input_text, reference_list[0]).strip()
        else:
            prompt = self.make_prompt(input_text, "").strip()

        return {
            "prompt": prompt,
            "output": reference_list  # <-- now a list, not just one string
        }


# ====================================
# Base class for classification datasets
# ====================================
class TextClassificationDataset:
    """
    Generic dataset class for classification tasks.

    Args:
        path (str): path to JSONL file with {"sentence": ..., "label": ...}
        prompt_template (str): formatting template
    """
    def __init__(self, path, prompt_template="{input_text} {separator} {output_text}"):
        self.task_type = "classification"
        self.path = path
        self.data = self.load_data(path)
        self.prompt_template = prompt_template
        self.label_mapping = {}

    def load_data(self, path):
        with open(path, "r") as f:
            data = [json.loads(line) for line in f]
        return data

    def __len__(self):
        return len(self.data)

    def make_prompt(self, input_text, output_text):
        return self.prompt_template.format(
            input_text=input_text, separator="{separator}", output_text=output_text
        )

    def __getitem__(self, idx, include_output=False):
        """
        Retrieve a dataset instance.

        Args:
            idx (int): index of the instance
            include_output (bool): whether to include the gold output in the prompt

        Returns:
            dict with keys:
                - "prompt": formatted text with {separator} placeholder
                - "output": gold reference output
        """
        instance = self.data[idx]
        input_text = instance["sentence"]
        output_text = self.label_mapping[instance["label"]]

        if include_output:
            prompt = self.make_prompt(input_text, output_text).strip()
        else:
            prompt = self.make_prompt(input_text, "").strip()
        return {"prompt": prompt, "output": output_text}

# ====================================
# Specific classification datasets
# ====================================
class RTEDataset(TextClassificationDataset):
    """RTE: Recognizing Textual Entailment (binary: entailment vs not_entailment)."""
    def __init__(self, path, prompt_template="{input_text} {separator} {output_text}"):
        super().__init__(path, prompt_template)
        self.label_mapping = {"not_entailment": "False", "entailment": "True"}

    def __getitem__(self, idx, include_output=False):
        instance = self.data[idx]
        input_text_a = instance["sentence_1"]
        input_text_b = instance["sentence_2"]
        input_text = f"{input_text_a} {input_text_b}"  # TODO: add a separator if necessary, whitespace for now
        output_text = self.label_mapping[instance["label"]]

        if include_output:
            prompt = self.make_prompt(input_text, output_text).strip()
        else:
            prompt = self.make_prompt(input_text, "").strip()
        return {"prompt": prompt, "output": output_text}


class CBDataset(TextClassificationDataset):
    """CB: CommitmentBank natural language inference dataset."""
    def __init__(self, path, prompt_template="{input_text} {separator} {output_text}"):
        super().__init__(path, prompt_template)
        self.label_mapping = {
            "contradiction": "false",
            "entailment": "true",
            "neutral": "neither",
        }

    def __getitem__(self, idx, include_output=False):
        instance = self.data[idx]
        input_text_a = instance["premise"]
        input_text_b = instance["hypothesis"]
        input_text = f"{input_text_a} {input_text_b}"  # TODO: add a separator if necessary, whitespace for now
        output_text = self.label_mapping[instance["label"]]

        if include_output:
            prompt = self.make_prompt(input_text, output_text).strip()
        else:
            prompt = self.make_prompt(input_text, "").strip()
        return {"prompt": prompt, "output": output_text}


class SST2Dataset(TextClassificationDataset):
    """SST-2: Sentiment analysis (binary)."""
    def __init__(self, path, prompt_template="{input_text} {separator} {output_text}"):
        super().__init__(path, prompt_template)
        self.label_mapping = {"0": "negative", "1": "positive"}


class TRECDataset(TextClassificationDataset):
    """TREC: Question classification (6-way)."""
    def __init__(self, path, prompt_template="{input_text} {separator} {output_text}"):
        super().__init__(path, prompt_template)
        self.label_mapping = {
            "0": "description",
            "1": "entity",
            "2": "expression",
            "3": "human",
            "4": "location",
            "5": "number",
        }


class AGNewsDataset(TextClassificationDataset):
    """AGNews: Topic classification (4-way)."""
    def __init__(self, path, prompt_template="{input_text} {separator} {output_text}"):
        super().__init__(path, prompt_template)
        self.label_mapping = {
            "1": "world",
            "2": "sports",
            "3": "business",
            "4": "technology",
        }


class DBPediaDataset(TextClassificationDataset):
    """DBPedia: Ontology classification (14-way)."""
    def __init__(self, path, prompt_template="{input_text} {separator} {output_text}"):
        super().__init__(path, prompt_template)
        self.label_mapping = {
            "1": "company",
            "2": "school",
            "3": "artist",
            "4": "athlete",
            "5": "politics",
            "6": "transportation",
            "7": "building",
            "8": "nature",
            "9": "village",
            "10": "animal",
            "11": "plant",
            "12": "album",
            "13": "film",
            "14": "book",
        }


class SubjDataset(TextClassificationDataset):
    """Subjectivity dataset: subjective vs objective sentences."""
    def __init__(self, path, prompt_template="{input_text} {separator} {output_text}"):
        super().__init__(path, prompt_template)
        self.label_mapping = {
            "0": "subjective",
            "1": "objective",
        }


class MRDataset(TextClassificationDataset):
    """MR: Movie review sentiment classification (binary)."""
    def __init__(self, path, prompt_template="{input_text} {separator} {output_text}"):
        super().__init__(path, prompt_template)
        self.label_mapping = {
            "0": "negative",
            "1": "positive",
        }


class SST5Dataset(TextClassificationDataset):
    """SST-5: Fine-grained sentiment classification (5-way)."""
    def __init__(self, path, prompt_template="{input_text} {separator} {output_text}"):
        super().__init__(path, prompt_template)
        self.label_mapping = {
            "0": "terrible",
            "1": "bad",
            "2": "okay",
            "3": "good",
            "4": "great",
        }


class MPQADataset(TextClassificationDataset):
    """MPQA: Opinion polarity dataset (binary)."""
    def __init__(self, path, prompt_template="{input_text} {separator} {output_text}"):
        super().__init__(path, prompt_template)
        self.label_mapping = {
            "0": "negative",
            "1": "positive",
        }


class CRDataset(TextClassificationDataset):
    """CR: Customer review sentiment dataset (binary)."""
    def __init__(self, path, prompt_template="{input_text} {separator} {output_text}"):
        super().__init__(path, prompt_template)
        self.label_mapping = {
            "0": "negative",
            "1": "positive",
        }




