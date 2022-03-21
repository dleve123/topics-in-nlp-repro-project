import argparse
from time import perf_counter
from model.correction_model import CorrectionModel
from preprocessing.prepare_train_dataset import tensors_from_jsonl_filepath


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run batch inference for a corrector model")
    parser.add_argument(
        "test_data_filepath", type=str, help="Filepath to test data"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="Filepath to model snapshot",
    )
    args = parser.parse_args()

    test_dataset = tensors_from_jsonl_filepath(args.test_data_filepath)
    model = CorrectionModel()

    print("-- Inference Starting --")
    test_start_time = perf_counter()

    if args.model_path:
        model.model.load_pretrained(args.model_path)
    print(model.batch_inference(test_dataset))

    print("-- Inference Ending --")
    test_end_time = perf_counter()
    print("Total Inference Time (mins):", (test_end_time - test_start_time) / 60)
