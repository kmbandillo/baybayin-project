#!/usr/bin/env python3
"""
Reorder multipage TIFF/BOX pairs so that pages with identical box coordinates
are grouped next to each other.

Example usage:
    python3 tools/reorder_pages_by_boxes.py \
        --tif final_training_dataset/handwritten/char_bundle/he_hi.tif \
        --box final_training_dataset/handwritten/char_bundle/he_hi.box
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image


@dataclass
class BoxEntry:
    label: str
    left: int
    bottom: int
    right: int
    top: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Group pages that share identical boxes.")
    parser.add_argument("--tif", type=Path, required=True, help="Path to input multipage TIFF.")
    parser.add_argument("--box", type=Path, required=True, help="Path to input BOX file.")
    parser.add_argument(
        "--out-tif",
        type=Path,
        help="Optional output TIFF path (default adds _grouped before extension).",
    )
    parser.add_argument(
        "--out-box",
        type=Path,
        help="Optional output BOX path (default adds _grouped before extension).",
    )
    return parser.parse_args()


def load_boxes(box_path: Path) -> Dict[int, List[BoxEntry]]:
    page_map: Dict[int, List[BoxEntry]] = {}
    for raw_line in box_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 6:
            continue
        label = parts[0]
        left, bottom, right, top = map(int, parts[1:5])
        try:
            page = int(parts[5])
        except ValueError:
            continue
        page_map.setdefault(page, []).append(BoxEntry(label, left, bottom, right, top))
    return page_map


def load_images(tif_path: Path) -> List[Image.Image]:
    images: List[Image.Image] = []
    with Image.open(tif_path) as img:
        try:
            idx = 0
            while True:
                img.seek(idx)
                images.append(img.copy())
                idx += 1
        except EOFError:
            pass
    if not images:
        raise RuntimeError(f"No pages found in {tif_path}")
    return images


def signature_for(entries: List[BoxEntry]) -> Tuple[Tuple[str, int, int, int, int], ...]:
    return tuple(sorted((e.label, e.left, e.bottom, e.right, e.top) for e in entries))


def build_grouped_order(total_pages: int, boxes: Dict[int, List[BoxEntry]]) -> List[int]:
    groups: Dict[Tuple[Tuple[str, int, int, int, int], ...], List[int]] = {}
    for page in range(total_pages):
        entries = boxes.get(page, [])
        sig = signature_for(entries)
        groups.setdefault(sig, []).append(page)
    new_order: List[int] = []
    for pages in groups.values():
        new_order.extend(pages)
    if len(new_order) != total_pages:
        raise RuntimeError("Grouped order does not cover all pages.")
    return new_order


def save_reordered_tif(images: List[Image.Image], order: List[int], out_path: Path) -> None:
    reordered = [images[idx].copy() for idx in order]
    first = reordered[0]
    rest = reordered[1:]
    first.save(out_path, save_all=True, append_images=rest)


def save_reordered_box(boxes: Dict[int, List[BoxEntry]], order: List[int], out_path: Path) -> None:
    lines: List[str] = []
    for new_page, old_page in enumerate(order):
        for entry in boxes.get(old_page, []):
            lines.append(f"{entry.label} {entry.left} {entry.bottom} {entry.right} {entry.top} {new_page}")
    out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def default_output_path(path: Path, suffix: str) -> Path:
    return path.with_name(f"{path.stem}{suffix}{path.suffix}")


def main() -> None:
    args = parse_args()
    images = load_images(args.tif)
    boxes = load_boxes(args.box)
    total_pages = len(images)
    order = build_grouped_order(total_pages, boxes)

    out_tif = args.out_tif or default_output_path(args.tif, "_grouped")
    out_box = args.out_box or default_output_path(args.box, "_grouped")

    save_reordered_tif(images, order, out_tif)
    save_reordered_box(boxes, order, out_box)

    summary = []
    start = 0
    while start < total_pages:
        run_sig = signature_for(boxes.get(order[start], []))
        end = start + 1
        while end < total_pages and signature_for(boxes.get(order[end], [])) == run_sig:
            end += 1
        summary.append(f"{start}-{end - 1} (source pages {order[start]}-{order[end - 1]})")
        start = end
    print(f"Reordered {total_pages} pages.")
    print("Grouped runs:", "; ".join(summary))
    print(f"Output TIFF: {out_tif}")
    print(f"Output BOX:  {out_box}")


if __name__ == "__main__":
    main()
