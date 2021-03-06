import argparse
import wandb
from time import perf_counter
from model.correction_model import CorrectionModel

from preprocessing.prepare_train_dataset import tensors_from_jsonl_filepath


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a corrector model")
    parser.add_argument(
        "train_data_filepath", type=str, help="Filepath to training data"
    )
    parser.add_argument(
        "model_save_path",
        type=str,
        help="Filepath to directory to save model snapshots",
    )
    parser.add_argument(
        "--epochs", type=int, help="Number of epochs to run training", default=3
    )
    parser.add_argument(
        "--learning_rate", type=float, help="training learning rate", default=1e-5
    )
    parser.add_argument(
        "--warmup", type=float, help="warmup steps as a proportion of training set size", default=0.1
    )
    parser.add_argument(
        "--negative_examples_per_batch", type=int, help="number of negative examples per batch", default=1
    )
    parser.add_argument(
        "--max_num_pairs_per_doc",
        type=int,
        help="Limit of the number of contrastive pairs to train on per doc."
        "Avoid over-fitting against certain examples",
        default=None,
    )
    parser.add_argument(
        "--steps_save_interval",
        type=int,
        help="gradient step interval to trigger saving",
        default=200,
    )
    parser.add_argument(
        "--run_name",
        type=str,
        help="Name of run for weights and biases",
    )
    args = parser.parse_args()

    run = wandb.init(project="correction-repro", entity="danton-nlp")
    run.name = args.run_name
    config = wandb.config
    config.update(args)

    train_dataset = tensors_from_jsonl_filepath(args.train_data_filepath)
    model = CorrectionModel()

    print("-- Training Starting --")
    train_start_time = perf_counter()

    model.train(
        dset=train_dataset,
        model_save_path=args.model_save_path,
        max_num_pairs_per_doc=args.max_num_pairs_per_doc,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_save_interval=args.steps_save_interval,
        negative_examples_per_batch=args.negative_examples_per_batch,
        warmup=args.warmup
    )

    print("-- Training Ending --")
    train_end_time = perf_counter()
    print("Total Training Time (mins):", (train_end_time - train_start_time) / 60)
