import argparse
import os
import subprocess
import sys
from typing import Iterable, List, Tuple


def levenshtein(a: str, b: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr.append(
                min(
                    prev[j] + 1,      # deletion
                    curr[j - 1] + 1,  # insertion
                    prev[j - 1] + cost,  # substitution
                )
            )
        prev = curr
    return prev[-1]


def collect_samples(full_dir: str, train_dir: str) -> List[str]:
    """Return base names that exist in full_dir but not in train_dir."""
    full = {
        os.path.splitext(name)[0]
        for name in os.listdir(full_dir)
        if name.lower().endswith(".tif")
    }
    train = {
        os.path.splitext(name)[0]
        for name in os.listdir(train_dir)
        if name.lower().endswith(".tif")
    }
    return sorted(full - train)


def run_tesseract(image_path: str, tessdata_dir: str, model_name: str) -> str:
    """Run Tesseract on image_path and return recognized text."""
    cmd = [
        "tesseract",
        image_path,
        "stdout",
        "-l",
        model_name,
        "--tessdata-dir",
        tessdata_dir,
        "--psm",
        "13",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def evaluate(
    base_names: Iterable[str],
    dataset_dir: str,
    tessdata_dir: str,
    model_name: str,
) -> Tuple[int, int, int, int]:
    """Evaluate model on provided samples.

    Returns (sample_count, char_errors, char_total, word_errors).
    """
    sample_count = 0
    total_char_errors = 0
    total_char_count = 0
    total_word_errors = 0

    for base in base_names:
        tif_path = os.path.join(dataset_dir, f"{base}.tif")
        gt_path = os.path.join(dataset_dir, f"{base}.gt.txt")
        if not os.path.isfile(tif_path) or not os.path.isfile(gt_path):
            continue
        try:
            prediction = run_tesseract(tif_path, tessdata_dir, model_name)
        except subprocess.CalledProcessError as exc:
            print(f"warning: tesseract failed on {base}: {exc}", file=sys.stderr)
            continue
        with open(gt_path, "r", encoding="utf-8") as f:
            ground_truth = f.read().strip()

        dist = levenshtein(prediction, ground_truth)
        total_char_errors += dist
        total_char_count += max(len(ground_truth), 1)
        total_word_errors += 0 if prediction == ground_truth else 1
        sample_count += 1

    return sample_count, total_char_errors, total_char_count, total_word_errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Baybayin model on holdout images."
    )
    parser.add_argument(
        "--full-set",
        required=True,
        help="Directory with the full Baybayin dataset (e.g., kaggle_dataset).",
    )
    parser.add_argument(
        "--train-set",
        required=True,
        help="Directory used for training (e.g., kaggle_dataset_dummy_corrected_full).",
    )
    parser.add_argument(
        "--tessdata-dir",
        required=True,
        help="Directory containing the traineddata file (e.g., tesseract_training/data).",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model name (traineddata filename without extension).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum number of holdout samples to evaluate (default: 100).",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Starting index within the holdout list (default: 0).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.isdir(args.full_set):
        sys.exit(f"error: full dataset '{args.full_set}' does not exist")
    if not os.path.isdir(args.train_set):
        sys.exit(f"error: train dataset '{args.train_set}' does not exist")

    holdouts = collect_samples(args.full_set, args.train_set)
    if not holdouts:
        sys.exit("error: no holdout samples found")

    start = max(args.offset, 0)
    end = len(holdouts) if args.limit <= 0 else min(start + args.limit, len(holdouts))
    selected = holdouts[start:end]

    (
        sample_count,
        char_errors,
        char_total,
        word_errors,
    ) = evaluate(selected, args.full_set, args.tessdata_dir, args.model)

    if sample_count == 0:
        sys.exit("error: no samples evaluated (missing tif/gt pairs?)")

    char_error_rate = (char_errors / char_total) * 100.0
    word_error_rate = (word_errors / sample_count) * 100.0

    print(f"Evaluated samples: {sample_count}")
    print(f"Character errors: {char_errors} / {char_total} ({char_error_rate:.2f}%)")
    print(f"Word errors: {word_errors} / {sample_count} ({word_error_rate:.2f}%)")


if __name__ == "__main__":
    main()
