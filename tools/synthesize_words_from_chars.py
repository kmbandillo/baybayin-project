#!/usr/bin/env python3
"""Generate synthetic Baybayin word lines from individual character crops."""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from PIL import Image, ImageStat


MARKS = {"ᜒ", "ᜓ", "᜔"}
WHITESPACE = {" ", "\t", "\r", "\n", "\u00a0"}

HEIGHT_BIN_SIZE = 8
WIDTH_BIN_SIZE = 8
DENSITY_BIN_SIZE = 0.08
MAX_DENSITY_BIN = 20
DEFAULT_HEIGHT_TOLERANCE = 0.15


@dataclass
class BoxEntry:
    char: str
    left: int
    bottom: int
    right: int
    top: int


@dataclass
class Sample:
    glyph: str
    tif_path: Path
    width: int
    height: int
    boxes: List[BoxEntry]
    base_height_px: int
    base_width_px: int
    density: float
    height_bin: int
    width_bin: int
    density_bin: int
    group_key: Tuple[int, int, int]


@dataclass
class GlyphInstance:
    glyph: str
    image: Image.Image
    boxes: List[BoxEntry]
    width: int
    height: int
    baseline_tl: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compose Baybayin word lines from char crops.")
    parser.add_argument(
        "--char-dir",
        type=Path,
        default=Path("baybayin_dataset/handwritten/characters"),
        help="Directory containing single-character TIFF/BOX/GT triples.",
    )
    parser.add_argument("--wordlist", type=Path, required=True, help="UTF-8 file with Baybayin words per line.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Destination directory for synthesized words.")
    parser.add_argument("--max-words", type=int, default=0, help="Limit output count (0 = all lines in wordlist).")
    parser.add_argument("--char-spacing", type=int, default=10, help="Pixels between consecutive glyphs.")
    parser.add_argument("--space-width", type=int, default=40, help="Pixels to advance for whitespace tokens.")
    parser.add_argument("--h-margin", type=int, default=20, help="Horizontal margin (pixels) on both sides.")
    parser.add_argument("--v-margin", type=int, default=20, help="Vertical margin (pixels) on top/bottom.")
    parser.add_argument("--crop-pad", type=int, default=2, help="Extra pixels kept around each glyph crop.")
    parser.add_argument("--seed", type=int, help="Seed for deterministic sampling.")
    parser.add_argument(
        "--height-tolerance",
        type=float,
        default=DEFAULT_HEIGHT_TOLERANCE,
        help="Allowable +/- ratio for base glyph height differences (e.g., 0.05 = 5%%).",
    )
    parser.add_argument(
        "--disable-style-groups",
        action="store_true",
        help="Allow mixing glyph styles within a word (default keeps same-height/density group).",
    )
    return parser.parse_args()


def read_box_file(path: Path) -> List[BoxEntry]:
    entries: List[BoxEntry] = []
    if not path.exists():
        return entries
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        ch = parts[0]
        try:
            left, bottom, right, top = map(int, parts[1:5])
        except ValueError:
            continue
        entries.append(BoxEntry(ch, left, bottom, right, top))
    return entries


def style_bounds(boxes: Sequence[BoxEntry], height: int) -> Tuple[int, int, int, int, int, int] | None:
    relevant = [box for box in boxes if box.char not in MARKS]
    if not relevant:
        relevant = list(boxes)
    if not relevant:
        return None
    left = min(box.left for box in relevant)
    right = max(box.right for box in relevant)
    top_tl = min(height - box.top for box in relevant)
    bottom_tl = max(height - box.bottom for box in relevant)
    right = max(left + 1, right)
    bottom_tl = max(top_tl + 1, bottom_tl)
    base_height = bottom_tl - top_tl
    base_width = right - left
    return left, top_tl, right, bottom_tl, base_height, base_width


def compute_density(img: Image.Image, bounds: Tuple[int, int, int, int, int, int]) -> float:
    left, top, right, bottom, _, _ = bounds
    if right <= left or bottom <= top:
        return 0.0
    region = img.crop((left, top, right, bottom))
    try:
        stat = ImageStat.Stat(region)
        mean = stat.mean[0] / 255.0
    finally:
        region.close()
    return max(0.0, min(1.0, 1.0 - mean))


def quantize_size(value: int, step: int) -> int:
    if value <= 0:
        return 0
    return value // step


def quantize_density(density: float) -> int:
    bin_value = int(density / DENSITY_BIN_SIZE)
    return max(0, min(MAX_DENSITY_BIN, bin_value))


def build_group_key(base_height: int, base_width: int, density: float) -> Tuple[int, int, int]:
    height_bin = quantize_size(base_height, HEIGHT_BIN_SIZE)
    width_bin = quantize_size(base_width, WIDTH_BIN_SIZE)
    density_bin = quantize_density(density)
    return height_bin, width_bin, density_bin


def load_samples(char_dir: Path) -> Dict[str, List[Sample]]:
    mapping: Dict[str, List[Sample]] = {}
    for gt_path in sorted(char_dir.glob("*.gt.txt")):
        glyph = gt_path.read_text(encoding="utf-8").strip()
        if not glyph:
            continue
        base = gt_path.with_suffix("").with_suffix("")
        tif_path = base.with_suffix(".tif")
        box_path = base.with_suffix(".box")
        if not tif_path.exists() or not box_path.exists():
            continue
        boxes = read_box_file(box_path)
        if not boxes:
            continue
        bounds: Tuple[int, int, int, int, int, int] | None = None
        density = 0.0
        with Image.open(tif_path) as img:
            gray = img.convert("L")
            try:
                width, height = gray.size
                bounds = style_bounds(boxes, height)
                if bounds is not None:
                    density = compute_density(gray, bounds)
            finally:
                gray.close()
        if bounds is None:
            continue
        _, _, _, _, base_height, base_width = bounds
        if base_height <= 0 or base_width <= 0:
            continue
        group_key = build_group_key(base_height, base_width, density)
        sample = Sample(
            glyph=glyph,
            tif_path=tif_path,
            width=width,
            height=height,
            boxes=boxes,
            base_height_px=base_height,
            base_width_px=base_width,
            density=density,
            height_bin=group_key[0],
            width_bin=group_key[1],
            density_bin=group_key[2],
            group_key=group_key,
        )
        mapping.setdefault(glyph, []).append(sample)
    return mapping


def segment_text(text: str) -> List[str]:
    tokens: List[str] = []
    i = 0
    length = len(text)
    while i < length:
        ch = text[i]
        if ch in WHITESPACE:
            tokens.append(" ")
            i += 1
            continue
        base = ch
        i += 1
        if i < length and text[i] in MARKS:
            base += text[i]
            i += 1
        tokens.append(base)
    return tokens


def crop_bounds(sample: Sample, pad: int) -> Sequence[int]:
    min_left = min(box.left for box in sample.boxes)
    max_right = max(box.right for box in sample.boxes)
    min_top_tl = min(sample.height - box.top for box in sample.boxes)
    max_bottom_tl = max(sample.height - box.bottom for box in sample.boxes)

    left = max(0, min_left - pad)
    right = min(sample.width, max_right + pad)
    top = max(0, min_top_tl - pad)
    bottom = min(sample.height, max_bottom_tl + pad)
    if right <= left:
        right = min(sample.width, left + 1)
    if bottom <= top:
        bottom = min(sample.height, top + 1)
    return left, top, right, bottom


def adjust_boxes(sample: Sample, crop: Sequence[int]) -> List[BoxEntry]:
    crop_left, crop_top, crop_right, crop_bottom = crop
    new_height = crop_bottom - crop_top
    adjusted: List[BoxEntry] = []
    for entry in sample.boxes:
        new_left = entry.left - crop_left
        new_right = entry.right - crop_left

        orig_top_tl = sample.height - entry.top
        orig_bottom_tl = sample.height - entry.bottom
        new_top_tl = orig_top_tl - crop_top
        new_bottom_tl = orig_bottom_tl - crop_top

        new_top = new_height - new_top_tl
        new_bottom = new_height - new_bottom_tl
        adjusted.append(BoxEntry(entry.char, new_left, new_bottom, new_right, new_top))
    return adjusted


def compute_baseline(boxes: Sequence[BoxEntry], height: int) -> int:
    base_boxes = [box for box in boxes if box.char not in MARKS]
    if not base_boxes:
        base_boxes = list(boxes)
    if not base_boxes:
        return height
    return max(height - box.bottom for box in base_boxes)


def make_glyph(sample: Sample, pad: int) -> GlyphInstance:
    crop = crop_bounds(sample, pad)
    with Image.open(sample.tif_path) as img:
        glyph_img = img.convert("L").crop(crop)
    boxes = adjust_boxes(sample, crop)
    width, height = glyph_img.size
    baseline_tl = compute_baseline(boxes, height)
    return GlyphInstance(sample.glyph, glyph_img, boxes, width, height, baseline_tl)


def iter_words(wordlist: Path) -> Iterable[str]:
    for raw_line in wordlist.read_text(encoding="utf-8").splitlines():
        word = raw_line.strip()
        if word:
            yield word


def select_sample(
    options: Sequence[Sample],
    rng: random.Random,
    target_group: Tuple[int, int, int] | None,
    enforce_style: bool,
    target_height_px: float | None,
    height_tol_ratio: float,
) -> Sample:
    def within_height(sample: Sample) -> bool:
        if target_height_px is None:
            return True
        lower = target_height_px * (1 - height_tol_ratio)
        upper = target_height_px * (1 + height_tol_ratio)
        return lower <= sample.base_height_px <= upper

    filtered = [sample for sample in options if within_height(sample)] or list(options)

    if not enforce_style or target_group is None:
        return rng.choice(filtered)

    same_group = [sample for sample in filtered if sample.group_key == target_group]
    if same_group:
        return rng.choice(same_group)
    target_height = target_group[0]
    same_height = [sample for sample in filtered if sample.height_bin == target_height]
    if same_height:
        return rng.choice(same_height)
    return rng.choice(filtered)


def place_boxes(
    glyph: GlyphInstance,
    x_offset: int,
    y_offset: int,
    final_height: int,
) -> List[BoxEntry]:
    placed: List[BoxEntry] = []
    for entry in glyph.boxes:
        left = entry.left + x_offset
        right = entry.right + x_offset

        glyph_top_tl = glyph.height - entry.top
        glyph_bottom_tl = glyph.height - entry.bottom
        top_tl = y_offset + glyph_top_tl
        bottom_tl = y_offset + glyph_bottom_tl

        top = final_height - top_tl
        bottom = final_height - bottom_tl
        placed.append(BoxEntry(entry.char, left, bottom, right, top))
    return placed


def compose_word(
    text: str,
    samples: Dict[str, List[Sample]],
    rng: random.Random,
    pad: int,
    char_spacing: int,
    space_width: int,
    h_margin: int,
    v_margin: int,
    out_prefix: Path,
    enforce_style: bool,
    height_tolerance: float,
) -> bool:
    tokens = segment_text(text)
    pieces: List[GlyphInstance | None] = []
    target_group: Tuple[int, int, int] | None = None
    target_height_px: float | None = None
    for token in tokens:
        if token == " ":
            pieces.append(None)
            continue
        options = samples.get(token)
        if not options:
            return False
        sample = select_sample(
            options,
            rng,
            target_group,
            enforce_style,
            target_height_px,
            height_tolerance,
        )
        glyph = make_glyph(sample, pad)
        pieces.append(glyph)
        if enforce_style:
            target_group = sample.group_key
        if target_height_px is None:
            target_height_px = sample.base_height_px

    glyphs = [p for p in pieces if p is not None]
    if not glyphs:
        return False

    max_ascent = max(glyph.baseline_tl for glyph in glyphs)
    max_descent = max(glyph.height - glyph.baseline_tl for glyph in glyphs)
    content_width = 0
    prev_was_glyph = False
    for piece in pieces:
        if piece is None:
            content_width += space_width
            prev_was_glyph = False
            continue
        if prev_was_glyph:
            content_width += char_spacing
        content_width += piece.width
        prev_was_glyph = True

    width = h_margin * 2 + content_width
    height = v_margin * 2 + max_ascent + max_descent
    canvas = Image.new("L", (width, height), color=255)

    box_entries: List[BoxEntry] = []
    x = h_margin
    prev_was_glyph = False
    for piece in pieces:
        if piece is None:
            x += space_width
            prev_was_glyph = False
            continue
        if prev_was_glyph:
            x += char_spacing
        y_offset = v_margin + (max_ascent - piece.baseline_tl)
        canvas.paste(piece.image, (x, y_offset))
        box_entries.extend(place_boxes(piece, x, y_offset, height))
        piece.image.close()
        x += piece.width
        prev_was_glyph = True

    tif_path = out_prefix.with_suffix(".tif")
    box_path = out_prefix.with_suffix(".box")
    gt_path = out_prefix.with_suffix(".gt.txt")

    canvas.save(tif_path, compression="tiff_deflate")
    box_lines = [f"{entry.char} {entry.left} {entry.bottom} {entry.right} {entry.top} 0" for entry in box_entries]
    box_path.write_text("\n".join(box_lines) + "\n", encoding="utf-8")
    gt_path.write_text(text + "\n", encoding="utf-8")
    return True


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        rng = random.Random(args.seed)
    else:
        rng = random.Random()

    samples = load_samples(args.char_dir)
    if not samples:
        raise SystemExit(f"No samples found under {args.char_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    generated = 0
    skipped_missing = 0

    limit = args.max_words if args.max_words and args.max_words > 0 else None
    enforce_style = not args.disable_style_groups
    height_tolerance = max(0.0, args.height_tolerance)

    for word in iter_words(args.wordlist):
        if limit is not None and generated >= limit:
            break
        total += 1
        out_prefix = args.output_dir / f"word_{generated:05d}"
        ok = compose_word(
            text=word,
            samples=samples,
            rng=rng,
            pad=args.crop_pad,
            char_spacing=args.char_spacing,
            space_width=args.space_width,
            h_margin=args.h_margin,
            v_margin=args.v_margin,
            out_prefix=out_prefix,
            enforce_style=enforce_style,
            height_tolerance=height_tolerance,
        )
        if ok:
            generated += 1
        else:
            skipped_missing += 1

    print(
        f"Generated {generated} words out of {total} candidates. "
        f"Skipped {skipped_missing} lines (missing glyph samples)."
    )


if __name__ == "__main__":
    main()
