import torch
from evaluation.feqa import FEQA
import benepar
import nltk
from datasets import load_dataset
from sumtool.storage import get_summaries

def download_models():
    benepar.download('benepar_en3')
    nltk.download('stopwords')
    nltk.download('punkt')

def load_data(dataset: str, split: str, model_summaries: str):
    data = load_dataset(dataset)[split]

    summaries_by_id = get_summaries(dataset, model_summaries)
    source_docs_by_id = {doc["id"]: doc["document"] for doc in data}

    source_docs = []
    summaries = []
    for doc_id, summary in summaries_by_id.items():
        source_docs.append(source_docs_by_id[doc_id])
        summaries.append(summary["summary"])

    return source_docs, summaries


def evaluate(docs, summaries, squad_dir, bart_qa_dir):
    scorer = FEQA(
        squad_dir=squad_dir,
        bart_qa_dir=bart_qa_dir,
        use_gpu=torch.cuda.is_available()
    )
    score = scorer.compute_score(docs, summaries, aggregate=False)

    return [float(s) for s in score]


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='Run FEQA on summaries stored in sumtool')
    parser.add_argument('dataset_split', type=str, help="Huggingface dataset with split, i.e. xsum/test")
    parser.add_argument('model_summaries', type=str, help="Path summaries in Sumtool storage, i.e. facebook-bart-large-xsum")
    parser.add_argument('--squad_dir', type=str, default="./evaluation/qa_models/squad1.0", help="Path to squad")
    parser.add_argument('--bart_qa_dir', type=str, default="./evaluation/bart_qg/checkpoints/", help="Path to bart-qa checkpoint")
    args = parser.parse_args()

    download_models()
    dataset, split = args.dataset_split.split("/")
    docs, summaries = load_data(dataset, split, args.model_summaries)
    print(len(docs), len(summaries))
    scores = evaluate(
        docs[:10], 
        summaries[:10], 
        args.squad_dir, 
        args.bart_qa_dir
    )
    print(scores)