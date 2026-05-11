#!/usr/bin/env python3
"""Create blurred/noisy character augmentations for Stage 1 training."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageFilter


IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
DEFAULT_SKIP = {"augmented_characters", "hw_characters", "__pycache__"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate two augmented copies (blur + noise) for every character "
            "image located in dataset/characters/* directories."
        )
    )
    parser.add_argument(
        "--characters-root",
        type=Path,
        default=Path("dataset/characters"),
        help="Root folder that contains character font folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Destination folder for augmented files. Defaults to "
            "<characters-root>/augmented_characters."
        ),
    )
    parser.add_argument(
        "--blur-blend",
        type=float,
        default=0.30,
        help="Blend factor applied between the original and blurred image.",
    )
    parser.add_argument(
        "--blur-radius",
        type=float,
        default=1.4,
        help="GaussianBlur radius used when producing blurred copies.",
    )
    parser.add_argument(
        "--noise-std-frac",
        type=float,
        default=0.12,
        help="Noise std-dev expressed as a fraction of the 0-255 range.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260329,
        help="Seed used for deterministic noise generation.",
    )
    parser.add_argument(
        "--only-fonts",
        nargs="*",
        default=None,
        help="Optional subset of font directories to process.",
    )
    parser.add_argument(
        "--extra-skip-fonts",
        nargs="*",
        default=None,
        help="Additional font directories to skip.",
    )
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Removes existing augmented files before generating new data.",
    )
    return parser.parse_args()


def collect_fonts(char_root: Path, only_fonts: Iterable[str] | None, skip: set[str]) -> list[Path]:
    fonts = []
    for subdir in sorted(char_root.iterdir()):
        if not subdir.is_dir():
            continue
        if subdir.name in skip:
            continue
        if only_fonts and subdir.name not in only_fonts:
            continue
        fonts.append(subdir)
    return fonts


def ensure_output_dir(path: Path, clean_output: bool) -> None:
    if clean_output and path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def apply_blur(image: Image.Image, blend_factor: float, radius: float) -> Image.Image:
    blurred = image.filter(ImageFilter.GaussianBlur(radius=radius))
    # Blend keeps most of the original structure while injecting blur.
    return Image.blend(image, blurred, alpha=blend_factor)


def apply_noise(
    image: Image.Image, rng: np.random.Generator, std_fraction: float
) -> Image.Image:
    arr = np.asarray(image).astype(np.float32)
    std = std_fraction * 255.0
    noise = rng.normal(0.0, std, size=arr.shape)
    noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy, mode=image.mode)


def copy_annotations(font_dir: Path, base_name: str, augmented_base: Path) -> None:
    box_path = font_dir / f"{base_name}.box"
    gt_path = font_dir / f"{base_name}.gt.txt"
    if box_path.exists():
        shutil.copy2(box_path, augmented_base.with_suffix(".box"))
    if gt_path.exists():
        shutil.copy2(gt_path, augmented_base.with_suffix(".gt.txt"))


def process_font_dir(
    font_dir: Path,
    output_dir: Path,
    blur_blend: float,
    blur_radius: float,
    noise_std_frac: float,
    rng: np.random.Generator,
) -> int:
    generated = 0
    for image_path in sorted(font_dir.iterdir()):
        if not image_path.is_file():
            continue
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        base_name = image_path.stem
        with Image.open(image_path) as img:
            original_mode = img.mode
            work_img = img.convert("RGB")

            blur_aug = apply_blur(work_img, blur_blend, blur_radius).convert(original_mode)
            blur_name = (
                output_dir
                / f"{font_dir.name}_augmentedchar_1_{base_name}{image_path.suffix.lower()}"
            )
            blur_aug.save(blur_name)
            copy_annotations(font_dir, base_name, blur_name.with_suffix(""))

            noise_aug = apply_noise(work_img, rng, noise_std_frac).convert(original_mode)
            noise_name = (
                output_dir
                / f"{font_dir.name}_augmentedchar_2_{base_name}{image_path.suffix.lower()}"
            )
            noise_aug.save(noise_name)
            copy_annotations(font_dir, base_name, noise_name.with_suffix(""))

            generated += 2
    return generated


def main() -> None:
    args = parse_args()
    characters_root = args.characters_root
    output_dir = args.output_dir or characters_root / "augmented_characters"

    skip_fonts = set(DEFAULT_SKIP)
    if args.extra_skip_fonts:
        skip_fonts.update(args.extra_skip_fonts)

    ensure_output_dir(output_dir, clean_output=args.clean_output)
    fonts = collect_fonts(characters_root, args.only_fonts, skip_fonts)

    rng = np.random.default_rng(args.seed)
    total = 0
    for font_dir in fonts:
        generated = process_font_dir(
            font_dir,
            output_dir,
            args.blur_blend,
            args.blur_radius,
            args.noise_std_frac,
            rng,
        )
        total += generated
        print(f"{font_dir.name}: generated {generated} augmented samples.")

    print(f"Completed augmentation. Total augmented samples: {total}")


if __name__ == "__main__":
    main()
