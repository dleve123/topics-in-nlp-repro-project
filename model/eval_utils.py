from typing import List, Optional

import torch
from tqdm import tqdm

from model.correction_model import CorrectionModel


def eval_pair_positive_heigher(
    correction_model: CorrectionModel, dset: List[torch.Tensor], max_num_pairs_per_doc: Optional[int] = None
) -> float:
    """
    Debugging utility to assess how accurate the corrector model is at separating
    faithful summaries from unfaithful summaries.

    Returns fraction of pairs where positive sample has higher faithful prob.
    """
    model = correction_model.model
    model.eval()

    correct_counter = 0
    total_counter = 0

    with torch.no_grad():
        for document_examples in tqdm(dset):

            if max_num_pairs_per_doc and document_examples.size(0) > (
                max_num_pairs_per_doc + 1
            ):
                document_examples = document_examples[: (max_num_pairs_per_doc + 1)]

            for contrastive_pair in model.create_pairs(document_examples):
                logits = model(contrastive_pair.to(model.device)).logits
                more_faithful_idx = logits[:, 1].argmax()

                total_counter += 1
                if more_faithful_idx == 0:
                    correct_counter += 1

    print()
    print(f"{correct_counter} / {total_counter} pairs correctly ranked")
    return correct_counter / total_counter


def eval_set_positive_heigher(
    correction_model, dset: List[torch.Tensor], max_num_pairs_per_doc: Optional[int] = None
) -> float:
    """
    Debugging utility to assess how accurate the discrimator model is at separating
    faithful summaries from unfaithfull summaries.

    Returns fraction of sets where positive sample has higher prob than all negative samples.
    """
    model = correction_model.model
    model.eval()

    correct_counter = 0
    total_counter = 0

    with torch.no_grad():
        for document_examples in tqdm(dset):

            if max_num_pairs_per_doc and document_examples.size(0) > (
                max_num_pairs_per_doc + 1
            ):
                document_examples = document_examples[: (max_num_pairs_per_doc + 1)]

            logits = model(document_examples.to(model.device)).logits
            more_faithful_idx = logits[:, 1].argmax()

            total_counter += 1
            if more_faithful_idx == 0:
                correct_counter += 1

    print()
    print(f"{correct_counter} / {total_counter} sets correctly ranked")

    return correct_counter / total_counter
