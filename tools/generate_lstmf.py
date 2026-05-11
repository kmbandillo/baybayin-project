#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Iterable, List


def iter_images(root: Path, patterns: List[str]) -> Iterable[Path]:
    for pattern in patterns:
        yield from root.glob(pattern)


def run_tesseract(image_path: Path, lang: str, psm: int, tessdata_dir: Path, config_path: Path) -> None:
    output_base = image_path.with_suffix("")
    lstmf_path = output_base.with_suffix(".lstmf")
    if lstmf_path.exists():
        return

    cmd = [
        "tesseract",
        image_path.name,
        output_base.name,
        "-l",
        lang,
        "--psm",
        str(psm),
        "--oem",
        "1",
        "--tessdata-dir",
        str(tessdata_dir.resolve()),
        str(config_path.resolve()),
    ]

    subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=image_path.parent,
    )

    if not lstmf_path.exists():
        raise RuntimeError(f"Tesseract did not produce {lstmf_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate .lstmf files for a dataset with existing boxes and gt.")
    parser.add_argument("root", type=Path, help="Dataset directory containing images")
    parser.add_argument("--psm", type=int, required=True, help="Tesseract PSM to use")
    parser.add_argument("--lang", default="baybayin_full_best", help="language code / traineddata to use")
    parser.add_argument("--tessdata-dir", type=Path, default=Path("releases/baybayin_full_current"))
    parser.add_argument("--config", type=Path, default=Path("releases/tesseract_training/data/configs/lstm.train"))
    parser.add_argument("--patterns", nargs="*", default=["*.tif", "*.png", "*.jpg", "*.jpeg"], help="Glob patterns to match images")
    args = parser.parse_args()

    root = args.root.resolve()
    images = sorted(iter_images(root, args.patterns))
    if not images:
        raise SystemExit(f"No images matching {args.patterns} found in {root}")

    print(f"Generating lstmf for {len(images)} images under {root}")

    failures = []
    for idx, image_path in enumerate(images, 1):
        try:
            run_tesseract(image_path, args.lang, args.psm, args.tessdata_dir, args.config)
        except subprocess.CalledProcessError as exc:
            failures.append((image_path, exc.stderr.decode("utf-8", "ignore")))
        except Exception as exc:  # noqa: BLE001
            failures.append((image_path, str(exc)))
        if idx % 100 == 0:
            print(f"  processed {idx} images...")

    if failures:
        log_path = root / "lstmf_generation_errors.log"
        with log_path.open("w", encoding="utf-8") as fh:
            for path, message in failures:
                fh.write(f"{path}: {message}\n")
        print(f"Completed with {len(failures)} failures; see {log_path}")
    else:
        print("Completed without failures.")


if __name__ == "__main__":
    main()
