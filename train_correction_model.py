import argparse
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
    args = parser.parse_args()

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
        steps_save_interal=args.steps_save_interval,
    )

    print("-- Training Ending --")
    train_end_time = perf_counter()
    print("Total Training Time (mins):", (train_end_time - train_start_time) / 60)
