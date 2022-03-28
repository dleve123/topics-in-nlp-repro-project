# Replication Project - Contrast Candidate Generation and Selection
The goal of this project is to reproduce Table 3 of [Chen et al's paper](https://arxiv.org/pdf/2104.09061.pdf) which introduces a model for correcting summary hallucinations.

![Screenshot 2022-03-22 at 13 01 55](https://user-images.githubusercontent.com/1349225/159535087-48116051-f951-41ac-92fb-ef1f1c12c6d1.png)

- [Paper Presentation for CS6741](https://docs.google.com/presentation/d/1O3qgO7NvnJ1jZbE_-lZDYVkpRgSA8NszTTgdVkUKy3U/edit#slide=id.p)

## Introduction

The authors present a method for correcting hallucinations in generated text summaries. They leverage a pre-trained BART-Base model which is fine-tuned to discriminate between "faithful" and "unfaithful" summaries. The training data for fine-tuning is created by artificially corrupting ground truth (human created) summaries. 

At a high-level, the process can be broken down into two steps:

1. **Candidate generation:** candidate summaries are created by replacing entities and quantities in summaries with entities of compatible semantic types from the source document. 
   - At training time, entities are replaced in the ground truth summary to create negative examples
   - At inference time, entities are replaced in the generated summary to create candidate summaries
2. **Candidate selection:** a fine-tuned BART “faithfulness” classifier ranks the candidate summaries according to how faithful they are to the source document. Within the set of candidate summaries, the most faithful summary as predicted by this classifier is the final output of the system. 

![Screenshot 2022-03-22 at 20 54 23](https://user-images.githubusercontent.com/1349225/159600671-a1cc97b3-61ea-4a9a-8215-7ae88ac5aa71.png)

## Methods
In order to replicate Table 3, we set out to:

1. Generate summaries using the baseline model (BART-Large fine-tuned on XSUM)
2. Implement and fine-tune the BART-Base faithfulness classifier from positive and artificially generated negative examples of XSUM train
3. Run evaluation metrics (Rouge-, BERT-, and FEQA-score) for the baseline and correction model on XSUM test

### Baseline
We begin by generating summaries for XSum Test using a pre-trained BART-Large that's fine-tuned on XSum.
- Code: [generate_bart_baseline.py](https://github.com/dleve123/topics-in-nlp-repro-project/blob/main/scripts/generate_bart_baseline.py) and [eval_bert_score.py](https://github.com/dleve123/topics-in-nlp-repro-project/blob/main/eval_bert_score.py)
- Stored summaries, rougeL & BERT-score: [data/xsum/facebook-bart-large-xsum-metrics](https://github.com/dleve123/topics-in-nlp-repro-project/blob/main/data/xsum/)

### Data Generation 
We make use of [code published by the author's papers](https://github.com/CogComp/faithful_summarization) to run the candidate generation process, and write our own scripts for tokenizing and batching candidate summaries ([bart_tokenize.py](https://github.com/dleve123/topics-in-nlp-repro-project/blob/main/bart_tokenize.py), [prepare_train_dataset.py](https://github.com/dleve123/topics-in-nlp-repro-project/blob/main/preprocessing/prepare_train_dataset.py)). 

### Correction Model
We implement the correction model with a combined cross-entropy and contrastive max-margin loss from scratch ([code](https://github.com/dleve123/topics-in-nlp-repro-project/blob/main/model/correction_model.py)).

![Screenshot 2022-03-22 at 21 43 55](https://user-images.githubusercontent.com/1349225/159605242-82cc20da-fdfd-4713-ac33-8b7886e172db.png)

The model is fine-tuned on Colab using training data that was published by the authors.

### Evaluation
We compute BERT-score, RougeL and FEQA scores for two sets of summaries:
- baseline generated summaries 
- corrected baseline generated summaries 

**Code:**
- [Evaluation notebook for FEQA-score (with caching)](evaluate_correction_model_feqa.ipynb)
- [Evaluation notebook for BERT & rouge-scores](evaluate_rouge_bert_score.ipynb)

## Results

### Original Table
![Screenshot 2022-03-22 at 13 01 55](https://user-images.githubusercontent.com/1349225/159535087-48116051-f951-41ac-92fb-ef1f1c12c6d1.png)

### Our Replication
![Screenshot 2022-03-22 at 22 25 54](https://user-images.githubusercontent.com/1349225/159611781-3d596d9f-6ae1-4c97-b67d-be1ccc385e97.png)


We are able to replicate the evaluation trends from the paper successfully. Most notably, our trained correction model successfully improves the faithfulness of the baseline generated summaries as measured by the FEQA scores.

There are some subtle differences which we attribute to:
- Transformer fine-tuning instability
- Variance in the loss due to the small batch sizes (one contrastive pair) of shuffled training data.

### Training logs
![Screenshot 2022-03-22 at 13 14 47](https://user-images.githubusercontent.com/1349225/159537740-f2ce17eb-69fe-46d9-8ae6-575dfc5fdf1c.png)
![Screenshot 2022-03-22 at 13 14 08](https://user-images.githubusercontent.com/1349225/159537741-d8265e37-6b44-4fcb-95c3-9e4d8ec43f09.png)
![Screenshot 2022-03-22 at 13 13 42](https://user-images.githubusercontent.com/1349225/159537744-26741168-2ac4-490e-8d52-d5a23bd8820a.png)

A key challenge in this replication was computational constraints, given that the full model requires 18 hours of GPU fine-tuning. 
Furthermore, evaluation using FEQA takes up to 2 hours for the corrected test set (on a GPU), which makes it challenging to evaluate the correction model during training. To overcome this we implemented saved model checkpoints and cached FEQA scores for a sample of the candidate summaries. This enabled us to iteratively evaluate our model during training.

___

# Appendix
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
