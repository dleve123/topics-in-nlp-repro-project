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

### Original table
![Screenshot 2022-03-22 at 13 01 55](https://user-images.githubusercontent.com/1349225/159535087-48116051-f951-41ac-92fb-ef1f1c12c6d1.png)


## Replication
![Screenshot 2022-03-22 at 12 59 22](https://user-images.githubusercontent.com/1349225/159534666-a8a6dbe2-dc15-4d43-93f6-e794aca8819f.png)

We replicate the trends in the paper successfully. There are some differences due to:
- we compute FEQA for a subset of the changed summaries due to computational constraints
- BERT fine-tuning instability

## Training logs
![Screenshot 2022-03-22 at 13 14 47](https://user-images.githubusercontent.com/1349225/159537740-f2ce17eb-69fe-46d9-8ae6-575dfc5fdf1c.png)
![Screenshot 2022-03-22 at 13 14 08](https://user-images.githubusercontent.com/1349225/159537741-d8265e37-6b44-4fcb-95c3-9e4d8ec43f09.png)
![Screenshot 2022-03-22 at 13 13 42](https://user-images.githubusercontent.com/1349225/159537744-26741168-2ac4-490e-8d52-d5a23bd8820a.png)


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
