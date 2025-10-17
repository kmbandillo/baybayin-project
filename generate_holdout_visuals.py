import argparse
import os
from typing import Iterable, List

from PIL import Image, ImageDraw

from evaluate_holdout import collect_samples, run_tesseract
from visualize_boxes import get_font, visualize_boxes


def add_caption(image_path: str, gt: str, pred: str) -> None:
    """Append prediction and ground truth text to the visualization image."""
    with Image.open(image_path).convert("RGBA") as base_img:
        font = get_font(size=28)
        padding = 12
        lines = [f"GT: {gt}", f"PRED: {pred}"]

        draw = ImageDraw.Draw(base_img)
        text_widths = []
        text_heights = []
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            text_widths.append(bbox[2] - bbox[0])
            text_heights.append(bbox[3] - bbox[1])

        caption_height = sum(text_heights) + padding * (len(lines) + 1)
        caption_width = base_img.width

        canvas_height = base_img.height + caption_height
        canvas = Image.new("RGBA", (caption_width, canvas_height), (30, 30, 30, 255))
        canvas.paste(base_img, (0, 0))

        draw_canvas = ImageDraw.Draw(canvas)
        y = base_img.height + padding
        for idx, line in enumerate(lines):
            draw_canvas.text(
                (padding, y),
                line,
                fill="yellow" if idx == 1 else "white",
                font=font,
            )
            y += text_heights[idx] + padding

        canvas.convert("RGB").save(image_path)


def visualize_samples(
    samples: Iterable[str],
    dataset_dir: str,
    tessdata_dir: str,
    model: str,
    output_dir: str,
) -> List[str]:
    os.makedirs(output_dir, exist_ok=True)
    outputs = []

    for base in samples:
        tif_path = os.path.join(dataset_dir, f"{base}.tif")
        box_path = os.path.join(dataset_dir, f"{base}.box")
        gt_path = os.path.join(dataset_dir, f"{base}.gt.txt")
        if not (os.path.isfile(tif_path) and os.path.isfile(box_path) and os.path.isfile(gt_path)):
            continue

        with open(gt_path, "r", encoding="utf-8") as f:
            ground_truth = f.read().strip()

        prediction = run_tesseract(tif_path, tessdata_dir, model)

        output_path = os.path.join(output_dir, f"{base}_visualization.png")
        visualize_boxes(tif_path, box_path, output_path)
        add_caption(output_path, ground_truth, prediction)
        outputs.append(output_path)

    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate visualization images for holdout samples with predictions."
    )
    parser.add_argument("--full-set", required=True, help="Directory with full dataset.")
    parser.add_argument(
        "--train-set", required=True, help="Directory used during training (dummy corrected set)."
    )
    parser.add_argument(
        "--tessdata-dir", required=True, help="Directory containing the traineddata file."
    )
    parser.add_argument("--model", required=True, help="Model name to evaluate.")
    parser.add_argument(
        "--output-dir",
        default="visualizations/holdout_predictions",
        help="Directory where visualization images will be stored.",
    )
    parser.add_argument("--limit", type=int, default=10, help="Number of samples to visualize.")
    parser.add_argument("--offset", type=int, default=0, help="Offset into holdout list.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    holdouts = collect_samples(args.full_set, args.train_set)
    if not holdouts:
        raise SystemExit("error: no holdout samples available.")

    start = max(args.offset, 0)
    end = len(holdouts) if args.limit <= 0 else min(start + args.limit, len(holdouts))
    samples = holdouts[start:end]

    outputs = visualize_samples(
        samples,
        args.full_set,
        args.tessdata_dir,
        args.model,
        args.output_dir,
    )

    if not outputs:
        raise SystemExit("error: no visualizations generated (files missing?).")

    print("Generated visualizations:")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
