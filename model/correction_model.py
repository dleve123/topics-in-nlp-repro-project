from typing import List
from transformers import BartTokenizer, BartForSequenceClassification
import torch
from torch import nn
from torch.optim import Adam
import time
from preprocessing.make_entity_perturbations import make_perturbations
import stanza
from tqdm import tqdm

#stanza.download('en')
nlp = stanza.Pipeline('en')

# From the paper
"""
For our contrast candidate selection model, we use
a pretrained BART base model. We add a linear
layer over the max pooled embedding, and the classification 
model is expected to output a label between ["FAITHFUL", "HALLUCINATED"].
"""

class CorrectionModel:
    def __init__(self, model_checkpoint="facebook/bart-base"):
        self.tokenizer = BartTokenizer.from_pretrained(model_checkpoint)
        self.model = BartForSequenceClassification.from_pretrained(
            model_checkpoint,
            # label 0: faithful, label 1: hallucinated
            num_labels=2
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(
        self,
        tokenized_dataset: List,
        epochs=3, 
        learning_rate=1e-5,
        batch_size=32 # For now hard-coded batch size of 1
    ):
        """
            Fine-tunes a correction model for a given tokenized dataset 
            which contains pairs of positive and negative examples
        """
        optim = Adam(
            self.model.parameters(), 
            lr=learning_rate
        )

        self.model.train()
        print("Training...")
        for epoch in range(epochs):
            start_time = time.time()
            epoch_loss = 0
            for pairs in tqdm(tokenized_dataset):
                optim.zero_grad()
                input_ids = pairs['input_ids'].to(self.device)
                attention_mask = pairs['attention_mask'].to(self.device)
                labels = torch.tensor([0, 1]).to(self.device)
                outputs = self.model(
                    input_ids,
                    attention_mask=attention_mask, 
                    labels=labels
                )
                loss = outputs[0]
                loss.backward()
                optim.step()
                epoch_loss += loss.item()
            print(f"Epoch {epoch}")
            print(f"Epoch time {time.time() - start_time}")
            print(f"Train epoch loss {epoch_loss / (len(tokenized_dataset))}")
        self.model.eval()

    def correct_summary(self, source, generated_summary):
        """
            Given a source doc and generated summary,
            generate candidate summaries and rank the candidates.

            Return the candidate summaries ranked according to faithulness
        """
        
        src_doc = nlp(source)
        src_doc.build_ents()

        tgt_doc = nlp(generated_summary)
        tgt_doc.build_ents()

        candidate_summaries, changed_list = make_perturbations(
            target_text=tgt_doc._text,
            target_ents=tgt_doc.ents,
            source_ents=src_doc.ents,
            is_training_mode=False,
            max_perturbation_per_example=10
        )
        summaries = [generated_summary] + candidate_summaries
        inputs = self.tokenizer(
            summaries, 
            text_pair=[src_doc._text] * len(summaries),
            truncation='only_second',
            return_tensors="pt", 
            padding=True
        )
        outputs = self.model(**inputs)
        loss = outputs.loss
        logits = outputs.logits

        # TODO: order summary according to their likelihood of being faithful

        return summaries, loss, logits

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
    summaries, loss, logits = model.correct_summary(
        source,
        summary
    )