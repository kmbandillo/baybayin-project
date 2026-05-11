#!/usr/bin/env python3
"""Run Tesseract OCR on every image inside a folder."""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable, List


def collect_images(root: Path, patterns: Iterable[str]) -> List[Path]:
    files: List[Path] = []
    for pattern in patterns:
        files.extend(root.glob(pattern))
    return sorted(files)


def run_tesseract(
    image_path: Path,
    traineddata: str,
    tessdata_dir: Path,
    psm: int,
    oem: int,
) -> subprocess.CompletedProcess[str]:
    cmd = [
        "tesseract",
        str(image_path),
        "stdout",
        "--psm",
        str(psm),
        "--oem",
        str(oem),
        "-l",
        traineddata,
    ]
    env = os.environ.copy()
    env["TESSDATA_PREFIX"] = str(tessdata_dir.resolve())
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def export_checkpoint(checkpoint: Path, base_traineddata: Path, output: Path) -> None:
    cmd = [
        "lstmtraining",
        "--stop_training",
        "--continue_from",
        str(checkpoint),
        "--traineddata",
        str(base_traineddata),
        "--model_output",
        str(output),
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OCR on every image inside a folder.")
    parser.add_argument("folder", type=Path, help="Folder containing images.")
    parser.add_argument(
        "--traineddata",
        default="dataset_stage3_words_phrases_best",
        help="Model name (looked up inside --tessdata-dir) or an explicit .traineddata/.checkpoint path.",
    )
    parser.add_argument(
        "--tessdata-dir",
        type=Path,
        default=Path("releases"),
        help="Directory containing traineddata files (default: releases).",
    )
    parser.add_argument(
        "--base-traineddata",
        type=Path,
        help="Base traineddata used during training. Required when --traineddata points to a .checkpoint file.",
    )
    parser.add_argument("--psm", type=int, default=6, help="Tesseract PSM mode (default: 6).")
    parser.add_argument("--oem", type=int, default=1, help="Tesseract OEM mode (default: 1).")
    parser.add_argument(
        "--patterns",
        nargs="*",
        default=["*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"],
        help="Glob patterns for images relative to the folder.",
    )
    args = parser.parse_args()

    folder = args.folder.resolve()
    if not folder.is_dir():
        parser.error(f"{folder} is not a directory.")

    images = collect_images(folder, args.patterns)
    if not images:
        parser.error(f"No images matching {args.patterns} found in {folder}")

    traineddata_name = args.traineddata
    tessdata_dir = args.tessdata_dir
    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    traineddata_path = Path(args.traineddata)
    try:
        if traineddata_path.suffix == ".traineddata":
            if not traineddata_path.exists():
                parser.error(f"{traineddata_path} does not exist.")
            tessdata_dir = traineddata_path.parent
            traineddata_name = traineddata_path.stem
        elif traineddata_path.suffix == ".checkpoint":
            if not traineddata_path.exists():
                parser.error(f"{traineddata_path} does not exist.")
            if args.base_traineddata is None:
                parser.error("--base-traineddata is required when using a .checkpoint file.")
            if not args.base_traineddata.exists():
                parser.error(f"{args.base_traineddata} does not exist.")
            temp_dir = tempfile.TemporaryDirectory()
            exported_traineddata = Path(temp_dir.name) / f"{traineddata_path.stem}.traineddata"
            print(f"[INFO] Exporting {traineddata_path} to {exported_traineddata}")
            export_checkpoint(traineddata_path, args.base_traineddata, exported_traineddata)
            tessdata_dir = exported_traineddata.parent
            traineddata_name = exported_traineddata.stem

        for image in images:
            result = run_tesseract(image, traineddata_name, tessdata_dir, args.psm, args.oem)
            if result.returncode != 0:
                print(f"[ERROR] {image.name}: {result.stderr.strip()}")
                continue
            print(f"== {image.name} ==")
            print(result.stdout.strip())
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


if __name__ == "__main__":
    main()
