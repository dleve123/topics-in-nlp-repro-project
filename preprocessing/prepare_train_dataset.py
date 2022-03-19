import json

def load_generated_summaries():
    # TODO: load generated summaries from disk
    return []

def load_paper_summaries(loc = "data/paper/train.jsonl"):
    """
        Loads training data (summaries with negative examples)
        provided by the paper from local storage
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

def tokenize_data_batch(tokenizer, data):
    """
        Returns a tensor with pairs of summaries
        [positive example, negative example]
    """
    tokenized = []
    for doc in data:
        positive_example = doc["positive_examples"]
        negative_examples = doc["negative_examples"]
        all_examples =  positive_example + negative_examples
        tokenized.append(tokenizer(
            all_examples,
            text_pair=[doc["source_text"]] * len(all_examples),
            return_tensors="pt",
            truncation='only_second',
            padding='max_length',
            max_length=tokenizer.model_max_length,
        ))

    return tokenized
