#!/usr/bin/env python3
"""
Preprocess handwritten Baybayin word images to improve OCR quality.
Pipeline matches the Dart implementation provided by the user:
  1. Auto-crop light borders starting from the image centre.
  2. Convert to grayscale.
  3. Apply Gaussian blur (radius=1) to reduce noise.
  4. Perform Otsu binarization.
  5. Remove tiny connected components (noise).
Outputs are saved as PNG files in the target directory without touching originals.
"""

from __future__ import annotations

import argparse
import shutil
from collections import deque
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
from PIL import Image, ImageFilter


def iter_images(root: Path) -> Iterable[Path]:
    for path in root.iterdir():
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}:
            yield path


def load_rgb(image_path: Path) -> Image.Image:
    return Image.open(image_path).convert("RGB")


def auto_crop_white_borders(image: Image.Image, white_threshold: int = 200, min_margin: float = 0.05) -> Image.Image:
    """Crop light margins while mirroring the Dart logic that expands from the centre."""
    rgb = np.array(image)
    if rgb.size == 0:
        return image
    height, width, _ = rgb.shape
    center_x = width // 2
    center_y = height // 2

    content_mask = np.any(rgb < white_threshold, axis=2)

    def row_has_content(y: int) -> bool:
        return bool(content_mask[y, :].any())

    def col_has_content(x: int) -> bool:
        return bool(content_mask[:, x].any())

    top_bound = None
    for y in range(center_y, -1, -1):
        has_content = row_has_content(y)
        if not has_content and top_bound is None:
            top_bound = min(height - 1, y + 1)
            break
        elif has_content:
            top_bound = y
    if top_bound is None:
        top_bound = 0

    bottom_bound = None
    for y in range(center_y, height):
        has_content = row_has_content(y)
        if not has_content and bottom_bound is None:
            bottom_bound = max(0, y - 1)
            break
        elif has_content:
            bottom_bound = y
    if bottom_bound is None:
        bottom_bound = height - 1

    left_bound = None
    for x in range(center_x, -1, -1):
        has_content = col_has_content(x)
        if not has_content and left_bound is None:
            left_bound = min(width - 1, x + 1)
            break
        elif has_content:
            left_bound = x
    if left_bound is None:
        left_bound = 0

    right_bound = None
    for x in range(center_x, width):
        has_content = col_has_content(x)
        if not has_content and right_bound is None:
            right_bound = max(0, x - 1)
            break
        elif has_content:
            right_bound = x
    if right_bound is None:
        right_bound = width - 1

    if right_bound <= left_bound or bottom_bound <= top_bound:
        return image

    crop_width = right_bound - left_bound + 1
    crop_height = bottom_bound - top_bound + 1

    horizontal_margin = (left_bound + (width - right_bound)) / width
    vertical_margin = (top_bound + (height - bottom_bound)) / height

    if horizontal_margin < min_margin and vertical_margin < min_margin:
        return image

    return image.crop((left_bound, top_bound, left_bound + crop_width, top_bound + crop_height))


def reduce_noise(image: Image.Image) -> Image.Image:
    return image.filter(ImageFilter.GaussianBlur(radius=1))


def calculate_otsu_threshold(arr: np.ndarray) -> int:
    histogram, _ = np.histogram(arr, bins=256, range=(0, 256))
    total = arr.size
    sum_total = np.dot(np.arange(256), histogram)

    sum_background = 0.0
    weight_background = 0
    max_variance = 0.0
    threshold = 0

    for t in range(256):
        weight_background += histogram[t]
        if weight_background == 0:
            continue

        weight_foreground = total - weight_background
        if weight_foreground == 0:
            break

        sum_background += t * histogram[t]
        mean_background = sum_background / weight_background
        mean_foreground = (sum_total - sum_background) / weight_foreground
        variance = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2

        if variance > max_variance:
            max_variance = variance
            threshold = t

    return threshold


def apply_otsu_binarization(image: Image.Image) -> Image.Image:
    arr = np.array(image)
    threshold = calculate_otsu_threshold(arr)
    binary = np.where(arr > threshold, 255, 0).astype(np.uint8)
    return Image.fromarray(binary, mode="L")


def remove_small_components(image: Image.Image, fraction: float = 0.0001) -> Image.Image:
    arr = np.array(image)
    height, width = arr.shape
    min_component_size = max(1, int(height * width * fraction))

    visited = np.zeros((height, width), dtype=bool)
    black = arr == 0

    neighbors: Tuple[Tuple[int, int], ...] = ((1, 0), (-1, 0), (0, 1), (0, -1))

    for y in range(height):
        for x in range(width):
            if not black[y, x] or visited[y, x]:
                continue
            queue: deque[Tuple[int, int]] = deque([(y, x)])
            component: list[Tuple[int, int]] = []
            visited[y, x] = True

            while queue:
                cy, cx = queue.pop()
                component.append((cy, cx))
                for dy, dx in neighbors:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < height and 0 <= nx < width and black[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        queue.append((ny, nx))

            if len(component) < min_component_size:
                for cy, cx in component:
                    arr[cy, cx] = 255

    return Image.fromarray(arr, mode="L")


def preprocess_image(image_path: Path) -> Image.Image:
    image_rgb = load_rgb(image_path)
    cropped = auto_crop_white_borders(image_rgb)
    grayscale = cropped.convert("L")
    blurred = reduce_noise(grayscale)
    binary = apply_otsu_binarization(blurred)
    cleaned = remove_small_components(binary)
    return cleaned


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess handwritten Baybayin word images.")
    parser.add_argument("--input", type=Path, required=True, help="Source directory containing original images.")
    parser.add_argument("--output", type=Path, required=True, help="Destination directory for preprocessed images.")
    parser.add_argument("--overwrite", action="store_true", help="Re-create outputs even if they already exist.")
    args = parser.parse_args()

    input_dir = args.input.resolve()
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(iter_images(input_dir))
    print(f"Found {len(images)} candidate images in {input_dir}.")

    processed = 0
    for image_path in images:
        stem = image_path.stem
        out_image_path = output_dir / f"{stem}.png"
        out_gt_path = output_dir / f"{stem}.gt.txt"

        if out_image_path.exists() and not args.overwrite:
            print(f"Skipping {stem}: already exists.")
            continue

        cleaned = preprocess_image(image_path)
        cleaned.save(out_image_path)
        processed += 1

        gt_path = image_path.with_suffix(".gt.txt")
        if gt_path.exists():
            shutil.copy2(gt_path, out_gt_path)

        print(f"Processed {stem} -> {out_image_path.name}")

    print(f"Completed preprocessing: {processed} images written to {output_dir}.")


if __name__ == "__main__":
    main()
