import json
from typing import List
from tqdm import tqdm
from transformers import BartTokenizer

def load_generated_summaries():
    # TODO: load generated summaries from disk
    return []

def load_valid_examples(loc = "data/paper/train.jsonl") -> List[dict]:
    """
        Loads training data with negative examples
        from provided filepath
    """

    docs_with_negative_examples = []
    with open(loc, "r") as f:
        for line in f:
            obj = json.loads(line)
            if len(obj["negative_examples"]) > 0:
                docs_with_negative_examples.append(obj)

    return docs_with_negative_examples

def tokenize_data(tokenizer, data):
    """
        Returns a tensor with pairs of summaries
        [positive example, negative example]
    """
    tokenized = []
    for doc in data:
        positive = doc["positive_examples"][0]
        for negative in doc["negative_examples"]:
            tokenized.append(tokenizer(
                [positive, negative],
                text_pair=[doc["source_text"]] * 2,
                return_tensors="pt",
                truncation='only_second',
                padding=True
            ))

    return tokenized

def tokenize_data_batch(tokenizer: BartTokenizer, data: List[dict]) -> List[List[int]]:
    """
    Tokenize a set of 1 positive and many negative examples provided via `data`.
    For each example, return a list of lists, where the inner list is the list of tokens
    for the concat(sumamry, document).

    List instead of Tensor as we write the tokenized data to a file.
    """
    tokenized = []
    for doc in tqdm(data):
        positive_example = doc["positive_examples"]
        negative_examples = doc["negative_examples"]
        all_examples =  positive_example + negative_examples
        tokenized.append(tokenizer(
            all_examples,
            text_pair=[doc["source_text"]] * len(all_examples),
            truncation='only_second',
            padding='max_length',
            max_length=tokenizer.model_max_length,
        )['input_ids'])

    return tokenized
