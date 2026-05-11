#!/usr/bin/env python3
"""
Evaluate a traineddata file against the image counterparts of an lstmf list.

For every .lstmf path in the provided listfile this script looks for the image
(`.png`, `.tif`, `.tiff`, `.jpg`) and `.gt.txt` that share the same stem.
It then runs Tesseract on the image with the supplied traineddata, compares the
OCR output against the GT text, and prints a short summary.

Usage:
  python scripts/eval_images_with_gt.py \
      --list releases/.../test.list \
      --traineddata releases/.../model.traineddata \
      --tessdata_prefix /usr/share/tesseract-ocr/4.00/tessdata \
      --psm 13
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


IMAGE_EXTS = [".png", ".tif", ".tiff", ".jpg", ".jpeg"]


def find_image_and_gt(stem: Path) -> Tuple[Path, Path]:
    for ext in IMAGE_EXTS:
        candidate = stem.with_suffix(ext)
        if candidate.exists():
            gt = stem.with_suffix(".gt.txt")
            if not gt.exists():
                raise FileNotFoundError(f"Missing GT file for {candidate}")
            return candidate, gt
    raise FileNotFoundError(f"No image found for {stem}")


def run_tesseract(
    image_path: Path,
    traineddata: Path,
    tessdata_dir: Path,
    psm: Optional[int],
) -> str:
    cmd: List[str] = [
        "tesseract",
        str(image_path),
        "stdout",
        "--tessdata-dir",
        str(tessdata_dir),
        "-l",
        traineddata.stem,
    ]
    if psm is not None:
        cmd.extend(["--psm", str(psm)])
    env = dict(os.environ)
    result = subprocess.run(
        cmd,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def normalize_text(text: str) -> str:
    return text.strip()


def build_mismatch_log_path(list_path: Path, traineddata: Path, psm: Optional[int]) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    psm_token = f"psm{psm}" if psm is not None else "psmdefault"
    log_name = (
        f"{list_path.name}."
        f"tesseract_mismatches."
        f"{traineddata.stem}."
        f"{psm_token}."
        f"{timestamp}.log"
    )
    return list_path.parent / log_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", required=True, type=Path, help="List of .lstmf paths.")
    parser.add_argument("--traineddata", required=True, type=Path, help="Traineddata file to use.")
    parser.add_argument(
        "--tessdata_dir",
        required=True,
        type=Path,
        help="Directory that contains the traineddata file.",
    )
    parser.add_argument(
        "--psm",
        type=int,
        default=None,
        help="Tesseract page segmentation mode (optional).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.list.exists():
        raise SystemExit(f"List file not found: {args.list}")
    if not args.traineddata.exists():
        raise SystemExit(f"Traineddata not found: {args.traineddata}")
    if not args.tessdata_dir.exists():
        raise SystemExit(f"Tessdata dir does not exist: {args.tessdata_dir}")

    entries = [line.strip() for line in args.list.read_text().splitlines() if line.strip()]
    mismatches = []
    for idx, lstmf_path in enumerate(entries, start=1):
        lstmf = Path(lstmf_path)
        stem = lstmf.with_suffix("")
        try:
            image_path, gt_path = find_image_and_gt(stem)
        except FileNotFoundError as err:
            print(f"[{idx}] {err}", file=sys.stderr)
            continue
        try:
            ocr = run_tesseract(image_path, args.traineddata, args.tessdata_dir, args.psm)
        except subprocess.CalledProcessError as err:
            print(f"[{idx}] tesseract failed on {image_path}: {err}", file=sys.stderr)
            continue
        gt_text = normalize_text(gt_path.read_text(encoding="utf-8"))
        if normalize_text(ocr) != gt_text:
            mismatches.append((image_path, gt_text, ocr))
            print(f"[{idx}] mismatch {image_path.name}: truth='{gt_text}' ocr='{ocr}'")
        else:
            print(f"[{idx}] ok {image_path.name}")

    print(f"\nTotal samples: {len(entries)}")
    print(f"Mismatches   : {len(mismatches)}")
    if mismatches:
        log_path = build_mismatch_log_path(args.list, args.traineddata, args.psm)
        with log_path.open("w", encoding="utf-8") as handle:
            for image_path, truth, ocr in mismatches:
                handle.write(f"{image_path}\nTruth:{truth}\nOCR  :{ocr}\n\n")
        print(f"Mismatch log written to {log_path}")


if __name__ == "__main__":
    main()
