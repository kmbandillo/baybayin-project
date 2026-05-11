#!/usr/bin/env python3
"""Generate .lstmf files for augmented character samples."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
import os
from typing import Iterable


IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create .lstmf files for augmented character images."
    )
    parser.add_argument(
        "--aug-dir",
        type=Path,
        default=Path("dataset/characters/augmented_characters"),
        help="Directory that stores augmented character assets.",
    )
    parser.add_argument(
        "--tool",
        type=Path,
        default=Path("tesseract_training_v2/tools/run_lstmf_with_auto_psm.sh"),
        help="Helper script used to invoke tesseract with consistent PSM settings.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recreate .lstmf files even if they already exist.",
    )
    parser.add_argument(
        "--tessdata-prefix",
        type=Path,
        default=Path("/usr/share/tesseract-ocr/4.00/tessdata"),
        help="Path passed via TESSDATA_PREFIX for tesseract executions.",
    )
    return parser.parse_args()


def iter_images(root: Path) -> Iterable[Path]:
    for path in sorted(root.iterdir()):
        if path.is_dir():
            yield from iter_images(path)
        elif path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def run_lstm_training(tool: Path, image_path: Path, tessdata_prefix: Path) -> None:
    base = image_path.with_suffix("")
    cmd = [
        str(tool),
        str(image_path),
        str(base),
    ]
    env = os.environ.copy()
    env["TESSDATA_PREFIX"] = str(tessdata_prefix)
    subprocess.run(cmd, check=True, env=env)


def main() -> None:
    args = parse_args()
    tool = args.tool.resolve()
    aug_dir = args.aug_dir.resolve()

    if not tool.exists():
        raise SystemExit(f"Helper script not found: {tool}")
    if not aug_dir.exists():
        raise SystemExit(f"Augmented directory not found: {aug_dir}")

    created = 0
    skipped = 0
    tessdata_prefix = args.tessdata_prefix.resolve()

    if not tessdata_prefix.exists():
        raise SystemExit(f"TESSDATA_PREFIX path does not exist: {tessdata_prefix}")

    for image_path in iter_images(aug_dir):
        lstmf_path = image_path.with_suffix(".lstmf")
        if lstmf_path.exists() and not args.force:
            skipped += 1
            continue
        run_lstm_training(tool, image_path, tessdata_prefix)
        created += 1
        print(f"Created {lstmf_path.relative_to(aug_dir)}")

    print(
        f"Completed .lstmf generation. Created={created}, skipped existing={skipped}."
    )


if __name__ == "__main__":
    main()
