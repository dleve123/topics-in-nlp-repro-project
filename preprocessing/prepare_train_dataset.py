import json

def load_generated_summaries():
    # TODO: load generated summaries from disk
    return []

def load_paper_summaries(loc = "data/paper/train.jsonl"):
    """
        Loads training data (summaries with negative examples)
        provided by the paper from local storage
    """

    papers_with_negative_examples = []
    with open(loc, "r") as f:
        for line in f:
            obj = json.loads(line)
            if len(obj["negative_examples"]) > 0:
                papers_with_negative_examples.append(obj)
            
    return papers_with_negative_examples

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