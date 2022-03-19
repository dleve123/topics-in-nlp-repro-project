# Contrastive BART Correction Reproduction

Paper: https://arxiv.org/pdf/2104.09061.pdf

The goal of this project is to reproduce Table 3 of Chen et al's paper
on building a correction model.


Python version: 3.9

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
