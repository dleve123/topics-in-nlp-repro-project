from bert_score import score
from datasets import load_dataset
from sumtool.storage import get_summaries, store_summary_metrics


def load_data(dataset: str, split: str, model_summaries: str):
    data = load_dataset(dataset)[split]

    gen_summaries_by_id = get_summaries(dataset, model_summaries)
    gt_summaries_by_id = {doc["id"]: doc["summary"] for doc in data}

    gt_sums = []
    gen_sums = []
    sum_ids = []
    for doc_id, gen_summary in gen_summaries_by_id.items():
        sum_ids.append(doc_id)
        gen_sums.append(gen_summary["summary"])
        gt_sums.append(gt_summaries_by_id[doc_id])

    return gt_sums, gen_sums, sum_ids


def evaluate(gt_summaries, generated_summaries):
    (P, R, F), hashname = score(generated_summaries, gt_summaries, lang="en", return_hash=True)
    print(f"P={P.mean().item():.6f} R={R.mean().item():.6f} F={F.mean().item():.6f}")
    return F.tolist()

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='Calculate BERT scores for summaries stored in sumtool')
    parser.add_argument('dataset_split', type=str, help="Huggingface dataset with split, i.e. xsum/test")
    parser.add_argument('model_summaries', type=str, help="Path summaries in Sumtool storage, i.e. facebook-bart-large-xsum")
    parser.add_argument('--subset', type=int, help="Run on a subset of the data", default=0)
    args = parser.parse_args()

    dataset, split = args.dataset_split.split("/")
    gt_sums, gen_sums, sum_ids = load_data(dataset, split, args.model_summaries)
    print(len(gt_sums), len(gen_sums))
    scores = evaluate(
        gt_sums[:args.subset] if args.subset != 0 else gt_sums, 
        gen_sums[:args.subset] if args.subset != 0 else gen_sums
    )
    store_summary_metrics(
        dataset, 
        args.model_summaries,
        {
            doc_id: { "bert-score": score} for doc_id, score in zip(sum_ids, scores)
        }
    )
