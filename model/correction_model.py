import os
import pathlib
from typing import List, Optional
from transformers import BartTokenizer, BartForSequenceClassification
import torch
from torch.utils.data import DataLoader
from torch import Tensor, nn
from torch.optim import Adam
import time
from preprocessing.make_entity_perturbations import make_perturbations
import stanza
from tqdm import tqdm

# From the paper
"""
For our contrast candidate selection model, we use
a pretrained BART base model. We add a linear
layer over the max pooled embedding, and the classification
model is expected to output a label between ["FAITHFUL", "HALLUCINATED"].
"""


class CorrectionModel:
    def __init__(self, model_checkpoint="facebook/bart-base"):
        self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
        self.model = BartForSequenceClassification.from_pretrained(
            model_checkpoint, num_labels=2
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def create_pairs(self, document_examples: Tensor) -> List[Tensor]:
        """
        Split a tensor of 1 positive and N negative pairs into
        a list of N elements, which each element being a Tensor with 2 rows:
        the first row being the tokenized positive input, the second row being
        the tokenized negative input.
        """
        batches = []

        positive = document_examples[0]
        negatives = document_examples[1:]

        for negative in negatives:
            batches.append(torch.vstack((positive, negative)))

        return batches

    def train(
        self,
        dset: List,
        model_save_path: str,
        max_num_pairs_per_doc: Optional[int] = None,
        epochs: int = 3,
        learning_rate: float = 1e-5,
        steps_save_interal: int = 200,
    ) -> None:
        """
        Trains CorrectionModel using a CE loss and MarginRankLoss.
        Snapshots the model every `steps_save_interval`
        """

        self.model.train()

        optim = Adam(self.model.parameters(), lr=learning_rate)

        cross_entropy_loss_function = nn.CrossEntropyLoss()
        margin_loss_function = nn.MarginRankingLoss(margin=0)

        epoch_data_iterator = DataLoader(dset, batch_size=1, shuffle=True)
        total_steps_counter = 0

        for epoch in range(epochs):
            start_time = time.perf_counter()
            epoch_loss = 0
            epoch_steps_counter = 0

            epoch_bar = tqdm(epoch_data_iterator)

            for document_examples in epoch_bar:
                doc_losses = []

                # Remove the batch dim to iterate over number of examples per batch (2)
                document_examples = document_examples.squeeze(0)

                if max_num_pairs_per_doc and document_examples.size(0) > (
                    max_num_pairs_per_doc + 1
                ):
                    document_examples = document_examples[: (max_num_pairs_per_doc + 1)]

                for contrastive_pair in self.create_pairs(document_examples):
                    optim.zero_grad()

                    contrastive_pair = contrastive_pair.to(self.device)

                    preds = self.model(contrastive_pair).logits

                    # should tend towards 1
                    positive_faithful_pred = preds[0, 1].unsqueeze(0)
                    # should tend towards 0
                    negative_faithful_pred = preds[1, 1].unsqueeze(0)

                    good_then_bad_labels = torch.LongTensor([1, 0]).to(self.device)
                    ce_loss = cross_entropy_loss_function(preds, good_then_bad_labels)

                    # positive_faithful_pred - negative_faithful_pred should be > 0
                    margin_target = torch.LongTensor([1]).to(self.device)
                    margin_loss = margin_loss_function(
                        positive_faithful_pred, negative_faithful_pred, margin_target
                    )

                    loss = ce_loss + margin_loss
                    loss.backward()

                    optim.step()

                    loss_item = loss.item()
                    epoch_loss += loss_item
                    doc_losses.append(loss_item)

                    epoch_steps_counter += 1
                    total_steps_counter += 1

                    current_loss = sum(doc_losses) / len(doc_losses)
                    epoch_bar.set_description(f"Current Loss {current_loss:.3f}")

            # save model after every steps_save_interval steps
            if (total_steps_counter % steps_save_interal) == 0:
                model_save_dir_path = os.path.join(
                    model_save_path, f"epoch-{epoch}_totalsteps-{total_steps_counter}"
                )
                pathlib.Path(model_save_dir_path).mkdir(parents=True, exist_ok=True)
                self.model.save_pretrained(model_save_dir_path)

            print(f"Epoch {epoch}")
            print(f"Epoch time {time.perf_counter() - start_time}")
            print(f"Train epoch loss {epoch_loss / epoch_steps_counter}")

        print("saving one last snapshot")
        model_save_dir_path = os.path.join(
            model_save_path, f"final-epoch-{epoch}_totalsteps-{total_steps_counter}"
        )
        pathlib.Path(model_save_dir_path).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(model_save_dir_path)

        return None

    def correct_summary(self, source, generated_summary):
        """
        Given a source doc and generated summary,
        generate candidate summaries and rank the candidates.

        Return the candidate summaries ranked according to faithfulness
        """

        stanza.download('en')
        nlp = stanza.Pipeline("en")

        src_doc = nlp(source)
        src_doc.build_ents()

        tgt_doc = nlp(generated_summary)
        tgt_doc.build_ents()

        candidate_summaries, changed_list = make_perturbations(
            target_text=tgt_doc._text,
            target_ents=tgt_doc.ents,
            source_ents=src_doc.ents,
            is_training_mode=False,
            max_perturbation_per_example=10,
        )
        summaries = [generated_summary] + candidate_summaries
        inputs = self.tokenizer(
            summaries,
            text_pair=[src_doc._text] * len(summaries),
            truncation="only_second",
            return_tensors="pt",
            padding=True,
        )
        outputs = self.model(**inputs)
        loss = outputs.loss
        logits = outputs.logits

        # TODO: order summary according to their likelihood of being faithful

        return summaries, loss, logits

    def batch_inference(
        self, 
        dset: List
    ):
        data_iterator = DataLoader(dset, batch_size=1, shuffle=False)
        prediction_logits = []

        with torch.no_grad():
            for document_examples in tqdm(data_iterator):
                document_examples = document_examples.squeeze(0)
                logits = self.model(document_examples.to(self.device)).logits
                prediction_logits.append(logits.cpu())
        
        return prediction_logits

if __name__ == "__main__":
    model = CorrectionModel()
    source = """\
He was re-elected for a second term by the UN General Assembly, \
unopposed and unanimously, on 21 \
June 2011, with effect from 1 January 2012. Mr. Ban \
describes his priorities as mobilising world leaders to deal \
with climate change, economic upheaval, pandemics and \
increasing pressures involving food, energy and water"""
    summary = """\
The United Nations Secretary-General Ban Ki-moon was elected \
for a second term in 21 June 2011."""
    summaries, loss, logits = model.correct_summary(source, summary)
