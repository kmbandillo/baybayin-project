#!/usr/bin/env python3
"""Compose single-line Baybayin phrases by stitching together rendered word crops."""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterable, List, Sequence, Optional

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from generate_boxes_and_lstmf import BoxEntry, write_box_file  # type: ignore

MARKS = {"ᜒ", "ᜓ", "᜔"}


@dataclass
class SampleRecord:
    """Represents a rendered TIFF/BOX/GT triple (word or punctuation glyph)."""

    name: str
    text: str
    tif_path: Path
    width: int
    height: int
    baseline_tl: int
    boxes: List[BoxEntry]
    kind: str  # "word" or "punct"


@dataclass
class Component:
    kind: str
    text: str
    image: Image.Image
    boxes: List[BoxEntry]
    width: int
    height: int
    baseline_tl: int
    base_left: int
    base_right: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stitch rendered words into training phrases.")
    parser.add_argument("--word-dir", type=Path, required=True, help="Directory of synthesized words.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Destination for phrase TIFF/BOX/GT triples.")
    parser.add_argument("--count", type=int, default=1000, help="Number of phrases to generate.")
    parser.add_argument("--min-words", type=int, default=3, help="Minimum words per phrase.")
    parser.add_argument("--max-words", type=int, default=4, help="Maximum words per phrase.")
    parser.add_argument("--char-spacing", type=int, default=10, help="Pixels between characters inside a word.")
    parser.add_argument("--word-spacing", type=int, default=32, help="Pixels between consecutive words.")
    parser.add_argument("--punct-spacing", type=int, default=0, help="Pixels before punctuation.")
    parser.add_argument(
        "--space-box-pad",
        type=int,
        default=0,
        help="Pixels trimmed from both sides of the space bounding box (legacy behavior, default 0).",
    )
    parser.add_argument(
        "--space-box-offset",
        type=int,
        default=0,
        help="How many pixels to inset the space box from the previous glyph's right edge.",
    )
    parser.add_argument("--h-margin", type=int, default=18, help="Horizontal margin in pixels.")
    parser.add_argument("--v-margin", type=int, default=18, help="Vertical margin in pixels.")
    parser.add_argument("--punct-prob", type=float, default=0.85, help="Chance to append terminal punctuation.")
    parser.add_argument("--seed", type=int, help="Random seed.")
    parser.add_argument(
        "--punct-chars",
        type=str,
        default=".,",
        help="String of punctuation characters to sample from (default: . ,).",
    )
    parser.add_argument(
        "--punct-char-dir",
        type=Path,
        help="Directory containing single-character TIFF/BOX pairs for punctuation glyphs.",
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


def compute_baseline(boxes: Sequence[BoxEntry], height: int) -> int:
    base_boxes = [box for box in boxes if box.char not in MARKS]
    if not base_boxes:
        base_boxes = list(boxes)
    if not base_boxes:
        return height
    return max(height - box.bottom for box in base_boxes)


def load_samples_from_dir(char_dir: Path, target_chars: Sequence[str] | None = None, kind: str = "word") -> List[SampleRecord]:
    needed = set(target_chars) if target_chars else None
    samples: List[SampleRecord] = []
    for gt_path in sorted(char_dir.glob("*.gt.txt")):
        text = gt_path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        if needed is not None and text not in needed:
            continue
        stem = gt_path.with_suffix("").with_suffix("")
        tif_path = stem.with_suffix(".tif")
        box_path = stem.with_suffix(".box")
        if not tif_path.exists() or not box_path.exists():
            continue
        boxes = read_box_file(box_path)
        if not boxes:
            continue
        with Image.open(tif_path) as img:
            width, height = img.size
        baseline = compute_baseline(boxes, height)
        samples.append(SampleRecord(stem.name, text, tif_path, width, height, baseline, boxes, kind))
    return samples


def duplicate_boxes(boxes: Sequence[BoxEntry]) -> List[BoxEntry]:
    return [BoxEntry(entry.char, entry.left, entry.bottom, entry.right, entry.top) for entry in boxes]


def base_bounds_from_boxes(boxes: Sequence[BoxEntry], fallback_width: int) -> tuple[int, int]:
    base_boxes = [box for box in boxes if box.char not in MARKS]
    if not base_boxes:
        base_boxes = list(boxes)
    if not base_boxes:
        return 0, max(1, fallback_width)
    left = min(box.left for box in base_boxes)
    right = max(box.right for box in base_boxes)
    if right <= left:
        right = left + 1
    return left, right


def make_component(sample: SampleRecord) -> Component:
    image = Image.open(sample.tif_path).convert("L")
    boxes = duplicate_boxes(sample.boxes)
    width = sample.width
    height = sample.height
    baseline_tl = sample.baseline_tl

    if sample.kind == "punct":
        crop_left = min(box.left for box in boxes)
        crop_right = max(box.right for box in boxes)
        crop_top_tl = min(height - box.top for box in boxes)
        crop_bottom_tl = max(height - box.bottom for box in boxes)
        crop_left = max(0, min(crop_left, width - 1))
        crop_right = min(width, max(crop_right, crop_left + 1))
        crop_top_tl = max(0, min(crop_top_tl, height - 1))
        crop_bottom_tl = min(height, max(crop_bottom_tl, crop_top_tl + 1))
        image = image.crop((crop_left, crop_top_tl, crop_right, crop_bottom_tl))
        new_width = crop_right - crop_left
        new_height = crop_bottom_tl - crop_top_tl
        for entry in boxes:
            entry.left -= crop_left
            entry.right -= crop_left
            orig_top_tl = height - entry.top
            orig_bottom_tl = height - entry.bottom
            new_top_tl = orig_top_tl - crop_top_tl
            new_bottom_tl = orig_bottom_tl - crop_top_tl
            entry.top = new_height - new_top_tl
            entry.bottom = new_height - new_bottom_tl
        baseline_tl = max(0, min(new_height, baseline_tl - crop_top_tl))
        width = new_width
        height = new_height

    base_left, base_right = base_bounds_from_boxes(boxes, width)

    return Component(
        kind=sample.kind,
        text=sample.text,
        image=image,
        boxes=boxes,
        width=width,
        height=height,
        baseline_tl=baseline_tl,
        base_left=base_left,
        base_right=base_right,
    )


def fallback_punctuation_component(char: str, reference_height: float) -> Component:
    height = max(28, int(round(reference_height * 0.55)))
    width = max(14, int(round(height * 0.35)))
    image = Image.new("L", (width, height), color=255)
    draw = ImageDraw.Draw(image)
    pad_bottom = max(3, height // 12)
    center_x = width // 2
    radius = max(3, height // 10)
    center_y = height - pad_bottom - radius
    bbox = (center_x - radius, center_y - radius, center_x + radius, center_y + radius)
    draw.ellipse(bbox, fill=0)
    if char == ",":
        tail_len = max(4, height // 8)
        draw.line(
            (center_x + radius // 2, center_y + radius, center_x, min(height - 1, center_y + radius + tail_len)),
            fill=0,
            width=max(1, radius // 3),
        )
    image_box = image.getbbox()
    if image_box is None:
        image_box = (width // 4, height - pad_bottom - radius * 2, 3 * width // 4, height - pad_bottom)
    left_px, top_px, right_px, bottom_px = image_box
    top = height - top_px
    bottom = height - bottom_px
    boxes = [BoxEntry(char, left_px, bottom, right_px, top)]
    baseline_tl = height - pad_bottom
    base_left, base_right = base_bounds_from_boxes(boxes, width)
    return Component(
        kind="punct",
        text=char,
        image=image,
        boxes=boxes,
        width=width,
        height=height,
        baseline_tl=baseline_tl,
        base_left=base_left,
        base_right=base_right,
    )


def place_boxes(component: Component, x_offset: int, y_offset: int, final_height: int) -> List[BoxEntry]:
    placed: List[BoxEntry] = []
    for entry in component.boxes:
        left = entry.left + x_offset
        right = entry.right + x_offset
        comp_top_tl = component.height - entry.top
        comp_bottom_tl = component.height - entry.bottom
        top_tl = y_offset + comp_top_tl
        bottom_tl = y_offset + comp_bottom_tl
        top = final_height - top_tl
        bottom = final_height - bottom_tl
        placed.append(BoxEntry(entry.char, left, bottom, right, top))
    return placed


def compose_components(
    components: Sequence[Component],
    word_spacing: int,
    punct_spacing: int,
    h_margin: int,
    v_margin: int,
    space_box_pad: int,
    space_box_offset: int,
) -> tuple[Image.Image, List[BoxEntry]]:
    max_ascent = max(component.baseline_tl for component in components)
    max_descent = max(component.height - component.baseline_tl for component in components)
    content_width = 0
    last_kind: str | None = None
    for component in components:
        spacing = 0
        if last_kind is not None:
            spacing = punct_spacing if component.kind == "punct" else word_spacing
        content_width += spacing + component.width
        last_kind = component.kind
    canvas_width = max(1, content_width + 2 * h_margin)
    canvas_height = max(1, max_ascent + max_descent + 2 * v_margin)
    baseline_y = v_margin + max_ascent
    image = Image.new("L", (canvas_width, canvas_height), color=255)
    placed_boxes: List[BoxEntry] = []
    x = h_margin
    last_kind = None
    last_component_right: Optional[int] = None
    space_top_tl = v_margin
    space_bottom_tl = v_margin + max_ascent + max_descent
    for component in components:
        spacing = 0
        add_space_box = False
        if last_kind is not None:
            if component.kind == "punct":
                spacing = 0
            else:
                spacing = word_spacing
                add_space_box = spacing > 0
        if spacing:
            next_start = x + spacing
            if add_space_box:
                prev_right = last_component_right if last_component_right is not None else x
                offset = max(0, space_box_offset)
                left = prev_right + offset
                next_base_left = next_start + component.base_left
                if left >= next_base_left:
                    left = next_base_left - 1
                width = max(1, next_base_left - left - 2)
                right = left + width
                if space_box_pad > 0:
                    pad = min(space_box_pad, max(0, (right - left - 1) // 2))
                    left += pad
                    right -= pad
                if right <= left:
                    right = min(next_base_left, left + 1)
                top = canvas_height - space_top_tl
                bottom = canvas_height - space_bottom_tl
                placed_boxes.append(BoxEntry(" ", left, bottom, right, top))
            x = next_start
        y = baseline_y - component.baseline_tl
        image.paste(component.image, (x, y))
        placed = place_boxes(component, x, y, canvas_height)
        placed_boxes.extend(placed)
        base_right = next((entry.right for entry in reversed(placed) if entry.char not in MARKS), None)
        if base_right is None:
            base_right = placed[-1].right if placed else x
        component_right = base_right
        x += component.width
        last_component_right = component_right
        last_kind = component.kind
        component.image.close()
    return image, placed_boxes


def build_phrase(
    samples: Sequence[SampleRecord],
    rng: random.Random,
    min_words: int,
    max_words: int,
    punct_prob: float,
    punct_chars: Sequence[str],
    punct_samples: dict[str, SampleRecord] | None,
) -> tuple[str, List[Component]]:
    count = rng.randint(min_words, max(min_words, max_words))
    chosen = [rng.choice(samples) for _ in range(count)]
    components = [make_component(sample) for sample in chosen]
    phrase_text = " ".join(sample.text for sample in chosen)
    if punct_chars and rng.random() < punct_prob:
        avg_height = sum(component.height for component in components) / len(components)
        char = rng.choice(punct_chars)
        if punct_samples and char in punct_samples:
            punct_component = make_component(punct_samples[char])
        else:
            punct_component = fallback_punctuation_component(char, avg_height)
        components.append(punct_component)
        phrase_text = phrase_text.rstrip() + char
    return phrase_text, components


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    word_dir = args.word_dir.resolve()
    output_dir = args.output_dir.resolve()

    word_samples = load_samples_from_dir(word_dir, kind="word")
    if not word_samples:
        raise SystemExit(f"No usable word samples found in {word_dir}")

    punct_chars = [ch for ch in args.punct_chars if ch.strip()]
    punct_samples = None
    if args.punct_char_dir and punct_chars:
        punct_dir = args.punct_char_dir.resolve()
        char_samples = load_samples_from_dir(punct_dir, punct_chars, kind="punct")
        if char_samples:
            punct_samples = {sample.text: sample for sample in char_samples}

    output_dir.mkdir(parents=True, exist_ok=True)

    generated = 0
    attempts = 0
    while generated < args.count:
        attempts += 1
        phrase_text, components = build_phrase(
            word_samples,
            rng,
            args.min_words,
            args.max_words,
            args.punct_prob,
            punct_chars,
            punct_samples,
        )
        if not components:
            continue

        image, boxes = compose_components(
            components,
            word_spacing=args.word_spacing,
            punct_spacing=args.punct_spacing,
            h_margin=args.h_margin,
            v_margin=args.v_margin,
            space_box_pad=args.space_box_pad,
            space_box_offset=args.space_box_offset,
        )
        out_prefix = output_dir / f"phrase_{generated:05d}"
        image.save(out_prefix.with_suffix(".tif"), format="TIFF", compression="tiff_deflate")
        write_box_file(out_prefix.with_suffix(".box"), boxes)
        out_prefix.with_suffix(".gt.txt").write_text(phrase_text + "\n", encoding="utf-8")
        image.close()
        generated += 1

    print(f"Generated {generated} phrases into {output_dir} after {attempts} attempts.")


if __name__ == "__main__":
    main()
