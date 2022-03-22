# Contrastive BART Correction Reproduction

Paper: https://arxiv.org/pdf/2104.09061.pdf

## Introduction

The goal of this project is to reproduce Table 3 of Chen et al's paper
on building a correction model.

At a high-level, the paper:

1. Generates summaries of Xsum train using BART-Large.
2. Generates artificial "negative" examples of these generated summaries using NER swapping.
3. Fine-tunes BART-Base to discriminate between correct and incorrect summaries.
4. Against the Xsum test set:
    1. Evaluates BERT score and ROUGE-L score between the summary selected as most correct by the
   discriminative model and the ground-truth human-created summary.
   2. Evaluates the FEQA score (measure of semantic correctness) of the selected summary given the source document.

## Methods

In order to replicate Table 3, we complete the following tasks:

1. Generate summaries and compute ROUGE-L and BERT score for Xsum test using a BART-Large model fine-tuned on Xsum.
2. Train a discrimative Correction model (from BART-Base) on Xsum train from generated negative and given positive examples.
4. Evaluate trained Correction model on Xsum test.

## Replication


### Results
- We replicate the trends in the paper (higher FEQA score from correction model)
- We compute FEQA for a subset of the changed summaries due to computational constraints
## Data Preprocessing

For efficiency, we tokenize the datasets in advance of training.

```bash
$ python bart_tokenize.py data/paper/val.jsonl data/tokenized/val.tokenized.jsonl
```

## Running [FEQA](https://github.com/esdurmus/feqa)
See [this notebook](https://colab.research.google.com/drive/1ie9oz20mt6RRm6KsGLM9Mwxn9LQJAWKr?authuser=1#scrollTo=NOP0jqxdKiCZ).

Trained models for question generation and question answering systems are under [this drive](https://drive.google.com/drive/u/1/folders/1O3kjSIhjDULw1RPJZTQ002GK3XNo2Vxl).

1. Download squad1.0 from Google Drive and place it under evaluation/qa_models directory.
2. Download checkpoints folder and place it under evaluation/bart_qg directory.
3. Run `python -m spacy download en_core_web_sm`
