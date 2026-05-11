#!/usr/bin/env python3
"""
Detect all text with EAST, run a lightweight Latin OCR pass, and mask Latin boxes.

Typical usage:

    python3 tools/mask_latin_regions.py dataset/words/withnoise/kapit.png \
        --east /path/to/frozen_east_text_detection.pb \
        --output dataset/words/withnoise/kapit_masked.png \
        --overlay dataset/words/withnoise/kapit_overlay.png
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class DetectedBox:
    x0: int
    y0: int
    x1: int
    y1: int
    score: float
    latin_confidence: float = -1.0
    latin_text: str = ""
    masked: bool = False

    def expanded(self, margin: int, width: int, height: int) -> Tuple[int, int, int, int]:
        """Return a margin-expanded bounding box clipped to image bounds."""
        if margin <= 0:
            return self.x0, self.y0, self.x1, self.y1
        x0 = max(0, self.x0 - margin)
        y0 = max(0, self.y0 - margin)
        x1 = min(width, self.x1 + margin)
        y1 = min(height, self.y1 + margin)
        return x0, y0, x1, y1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mask Latin regions by combining EAST detection with a Latin OCR filter."
    )
    parser.add_argument("image", type=Path, help="Path to the input image.")
    parser.add_argument(
        "--east",
        type=Path,
        required=True,
        help="Path to frozen_east_text_detection.pb (download from OpenCV's model zoo).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Destination image with masked Latin boxes. Defaults to <image>_masked.<ext>.",
    )
    parser.add_argument(
        "--overlay",
        type=Path,
        help="Optional debug overlay path that visualizes which boxes were masked vs kept.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Resized width fed into EAST (must be divisible by 32).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=640,
        help="Resized height fed into EAST (must be divisible by 32).",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.5,
        help="Minimum EAST score for keeping a detection before NMS.",
    )
    parser.add_argument(
        "--nms-threshold",
        type=float,
        default=0.4,
        help="IoU threshold passed to cv2.dnn.NMSBoxes.",
    )
    parser.add_argument(
        "--expand",
        type=int,
        default=4,
        help="Extra pixels added on each side before running the Latin OCR step.",
    )
    parser.add_argument(
        "--tesseract-lang",
        default="eng",
        help="Language passed to Tesseract for the Latin classifier.",
    )
    parser.add_argument(
        "--tesseract-psm",
        type=int,
        default=7,
        help="Page segmentation mode for Tesseract (7 = single text line).",
    )
    parser.add_argument(
        "--latin-threshold",
        type=float,
        default=70.0,
        help="Average Tesseract confidence (0-100) required to consider a box Latin.",
    )
    parser.add_argument(
        "--min-ascii-letters",
        type=int,
        default=2,
        help="Minimum number of ASCII alphabetic characters returned by Tesseract.",
    )
    parser.add_argument(
        "--mask-color",
        default="0,0,0",
        help="RGB color used for masking (e.g. 0,0,0 for black or 255,255,255 for white).",
    )
    return parser.parse_args()


def ensure_inputs(args: argparse.Namespace) -> None:
    if not args.image.exists():
        raise FileNotFoundError(f"Missing input image: {args.image}")
    if not args.east.exists():
        raise FileNotFoundError(f"Missing EAST model: {args.east}")
    if args.width % 32 != 0 or args.height % 32 != 0:
        raise ValueError("EAST width/height must be divisible by 32.")
    if shutil.which("tesseract") is None:
        raise RuntimeError("Tesseract binary not found in PATH.")


def load_image(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")
    return image


def decode_predictions(
    scores: np.ndarray,
    geometry: np.ndarray,
    min_score: float,
) -> Tuple[List[Tuple[int, int, int, int]], List[float]]:
    num_rows, num_cols = scores.shape[2], scores.shape[3]
    boxes: List[Tuple[int, int, int, int]] = []
    confidences: List[float] = []

    for y in range(num_rows):
        scores_data = scores[0, 0, y]
        x_data0 = geometry[0, 0, y]
        x_data1 = geometry[0, 1, y]
        x_data2 = geometry[0, 2, y]
        x_data3 = geometry[0, 3, y]
        angles_data = geometry[0, 4, y]

        for x in range(num_cols):
            score = scores_data[x]
            if score < min_score:
                continue

            offset_x = x * 4.0
            offset_y = y * 4.0
            angle = angles_data[x]
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)

            h = x_data0[x] + x_data2[x]
            w = x_data1[x] + x_data3[x]

            end_x = int(offset_x + (cos_a * x_data1[x]) + (sin_a * x_data2[x]))
            end_y = int(offset_y - (sin_a * x_data1[x]) + (cos_a * x_data2[x]))
            start_x = int(end_x - w)
            start_y = int(end_y - h)

            boxes.append((start_x, start_y, end_x, end_y))
            confidences.append(float(score))
    return boxes, confidences


def detect_text_regions(
    image: np.ndarray,
    net: cv2.dnn_Net,
    width: int,
    height: int,
    min_score: float,
    nms_threshold: float,
) -> List[DetectedBox]:
    orig_h, orig_w = image.shape[:2]
    resized = cv2.resize(image, (width, height))

    blob = cv2.dnn.blobFromImage(
        resized,
        1.0,
        (width, height),
        (123.68, 116.78, 103.94),
        swapRB=True,
        crop=False,
    )
    net.setInput(blob)
    layer_names = ("feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3")
    scores, geometry = net.forward(layer_names)

    decoded_boxes, confidences = decode_predictions(scores, geometry, min_score)
    rects_for_nms = []
    for start_x, start_y, end_x, end_y in decoded_boxes:
        rects_for_nms.append(
            (
                start_x,
                start_y,
                max(1, end_x - start_x),
                max(1, end_y - start_y),
            )
        )

    indices = cv2.dnn.NMSBoxes(rects_for_nms, confidences, min_score, nms_threshold)
    detected: List[DetectedBox] = []
    if len(indices) == 0:
        return detected

    scale_x = orig_w / float(width)
    scale_y = orig_h / float(height)
    for idx in np.array(indices).flatten():
        start_x, start_y, end_x, end_y = decoded_boxes[idx]
        x0 = int(max(0, start_x * scale_x))
        y0 = int(max(0, start_y * scale_y))
        x1 = int(min(orig_w, end_x * scale_x))
        y1 = int(min(orig_h, end_y * scale_y))
        if x1 <= x0 or y1 <= y0:
            continue
        detected.append(
            DetectedBox(
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
                score=confidences[idx],
            )
        )
    return detected


def save_patch(tmp_dir: Path, roi: np.ndarray, idx: int) -> Path:
    patch_path = tmp_dir / f"roi_{idx}.png"
    ok = cv2.imwrite(str(patch_path), roi)
    if not ok:
        raise RuntimeError("Failed to serialize ROI for OCR.")
    return patch_path


def parse_tesseract_tsv(tsv_text: str) -> Tuple[float, str]:
    rows = tsv_text.strip().splitlines()
    if not rows:
        return -1.0, ""
    reader = csv.DictReader(rows, delimiter="\t")
    confidences: List[float] = []
    texts: List[str] = []

    for row in reader:
        if not row:
            continue
        text = row.get("text", "").strip()
        if text:
            texts.append(text)
        conf_str = row.get("conf", "-1")
        try:
            conf = float(conf_str)
        except ValueError:
            conf = -1.0
        if conf >= 0:
            confidences.append(conf)
    avg_conf = sum(confidences) / len(confidences) if confidences else -1.0
    combined_text = " ".join(texts)
    return avg_conf, combined_text


def latin_confidence(
    roi: np.ndarray,
    tmp_dir: Path,
    idx: int,
    lang: str,
    psm: int,
) -> Tuple[float, str]:
    patch_path = save_patch(tmp_dir, roi, idx)
    cmd = [
        "tesseract",
        str(patch_path),
        "stdout",
        "--psm",
        str(psm),
        "-l",
        lang,
        "tsv",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Tesseract failed on ROI {idx}: {result.stderr.strip()}")
    return parse_tesseract_tsv(result.stdout)


def ascii_letter_count(text: str) -> int:
    count = 0
    for char in text:
        if char.isascii() and char.isalpha():
            count += 1
    return count


def parse_color(color_spec: str) -> Tuple[int, int, int]:
    if "," in color_spec:
        parts = color_spec.split(",")
        if len(parts) != 3:
            raise ValueError(f"Invalid color spec: {color_spec}")
        try:
            rgb = tuple(int(p.strip()) for p in parts)
        except ValueError as exc:
            raise ValueError(f"Invalid color component in {color_spec}") from exc
        if any(not 0 <= value <= 255 for value in rgb):
            raise ValueError("Mask color components must be within 0-255.")
        r, g, b = rgb
        return b, g, r  # Convert to BGR for OpenCV.
    raise ValueError(f"Unsupported color format: {color_spec}")


def draw_overlay(image: np.ndarray, detections: Sequence[DetectedBox], overlay_path: Path) -> None:
    overlay = image.copy()
    for box in detections:
        color = (0, 0, 255) if box.masked else (0, 180, 0)
        cv2.rectangle(overlay, (box.x0, box.y0), (box.x1, box.y1), color, 2)
        label_parts = []
        if box.latin_confidence >= 0:
            label_parts.append(f"{box.latin_confidence:.0f}")
        if box.latin_text:
            label_parts.append(box.latin_text[:12])
        label = " ".join(label_parts)
        if label:
            cv2.putText(
                overlay,
                label,
                (box.x0, max(12, box.y0 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
                lineType=cv2.LINE_AA,
            )
    cv2.imwrite(str(overlay_path), overlay)


def mask_latin_boxes(
    image: np.ndarray,
    detections: List[DetectedBox],
    args: argparse.Namespace,
) -> Tuple[np.ndarray, List[DetectedBox]]:
    height, width = image.shape[:2]
    mask_color = parse_color(args.mask_color)
    masked = image.copy()
    with tempfile.TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        for idx, box in enumerate(detections):
            x0, y0, x1, y1 = box.expanded(args.expand, width, height)
            roi = image[y0:y1, x0:x1]
            if roi.size == 0:
                continue
            try:
                conf, text = latin_confidence(roi, tmp_dir, idx, args.tesseract_lang, args.tesseract_psm)
            except RuntimeError as exc:
                print(str(exc), file=sys.stderr)
                continue
            box.latin_confidence = conf
            box.latin_text = text
            ascii_letters = ascii_letter_count(text)
            if conf >= args.latin_threshold and ascii_letters >= args.min_ascii_letters:
                cv2.rectangle(masked, (x0, y0), (x1, y1), mask_color, thickness=-1)
                box.masked = True
    return masked, detections


def main() -> None:
    args = parse_args()
    ensure_inputs(args)

    image = load_image(args.image)
    net = cv2.dnn.readNet(str(args.east))

    detections = detect_text_regions(
        image=image,
        net=net,
        width=args.width,
        height=args.height,
        min_score=args.min_score,
        nms_threshold=args.nms_threshold,
    )
    if not detections:
        print("No detections found. Copying input image to output.", file=sys.stderr)
        output_path = args.output or args.image.with_name(f"{args.image.stem}_masked{args.image.suffix}")
        cv2.imwrite(str(output_path), image)
        if args.overlay:
            draw_overlay(image, detections, args.overlay)
        return

    masked_image, processed = mask_latin_boxes(image, detections, args)
    output_path = args.output or args.image.with_name(f"{args.image.stem}_masked{args.image.suffix}")
    cv2.imwrite(str(output_path), masked_image)

    masked_count = sum(1 for box in processed if box.masked)
    print(f"Masked {masked_count} / {len(processed)} detected boxes.")
    if args.overlay:
        draw_overlay(image, processed, args.overlay)


if __name__ == "__main__":
    main()
