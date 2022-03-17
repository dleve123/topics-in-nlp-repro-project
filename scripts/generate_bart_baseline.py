import torch
from transformers import BartTokenizer, BartForConditionalGeneration
from typing import Tuple
from datasets import load_dataset
from transformers import BartTokenizer, BartForConditionalGeneration
from torch.utils.data import DataLoader
from rouge_score import rouge_scorer
from tqdm import tqdm
from sumtool.storage import store_model_summaries

def load_summarization_model_and_tokenizer(device) -> Tuple[
    BartForConditionalGeneration, BartTokenizer
]:
    """
    Load summary generation model and move to GPU, if possible.
    Returns:
        (model, tokenizer)
    """
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-xsum")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-xsum")
    model.to(device)

    return model, tokenizer

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, tokenizer = load_summarization_model_and_tokenizer(device)

    xsum_test = load_dataset('xsum')['test']
    data_loader = DataLoader(xsum_test, batch_size=8)

    rouge_l_scorer = rouge_scorer.RougeScorer(['rougeL'])

    test_rouge_scores = []
    rouge_score_metadata = {}
    id_to_batch_generated_summaries = {}

    for batch_idx, batch in enumerate(tqdm(data_loader)):
        batch_inputs = tokenizer(
            batch['document'],
            max_length=1024,
            truncation=True,
            padding=True,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            batch_summaries_encoded = model.generate(batch_inputs["input_ids"])
            batch_summaries = tokenizer.batch_decode(batch_summaries_encoded, skip_special_tokens=True, clean_up_tokenization_spaces=False)

            results = zip(batch['id'], batch['summary'], batch_summaries)
            for example_id, gt_summary, generated_summary in results:
                rouge_score_metadata[example_id] = rouge_l_scorer.score(gt_summary, generated_summary)['rougeL'].fmeasure
                id_to_batch_generated_summaries[example_id] = generated_summary


    store_model_summaries(
        'xsum',
        model.config.name_or_path,
        model.config.to_dict(),
        id_to_batch_generated_summaries,
        rouge_score_metadata
    )

    try:
        from google.colab import files
        files.download('/content/data/xsum/facebook-bart-large-xsum-summaries.json')
    except ImportError:
        print("Check your local filesystem for generated summaries")
        exit(0)
