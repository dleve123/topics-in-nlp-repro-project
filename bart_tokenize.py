from preprocessing.prepare_train_dataset import load_valid_examples, tokenize_data, tokenize_data_batch
from model.correction_model import CorrectionModel
import argparse
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Filter input data to those that only include examples, BART tokenize input dataset and write to file'
    )
    parser.add_argument('input_filepath', type=str, help="Filepath to input file")
    parser.add_argument('output_filepath', type=str, help="Filepath to output file")
    parser.add_argument('--num_examples', type=int, help="For debugging: number of examples to tokenize")
    parser.add_argument('--paired', action="store_true", default=False, help="Automatically create sets with 1 positive and 1 negative example")
    args = parser.parse_args()

    corrector_tokenizer = CorrectionModel().tokenizer

    examples = load_valid_examples(args.input_filepath)
    if args.num_examples:
        examples = examples[:args.num_examples]

    if args.paired:
        tokenized_example_sets = tokenize_data(corrector_tokenizer, examples)
    else:
        tokenized_example_sets = tokenize_data_batch(corrector_tokenizer, examples)

    # write tokenized to jsonl format for consistency
    with open(args.output_filepath, 'w') as out_file:
        for tokenized_example_set in tokenized_example_sets:
            out_file.write(json.dumps(tokenized_example_set))
            out_file.write("\n")
