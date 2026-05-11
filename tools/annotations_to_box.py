#!/usr/bin/env python3
"""
Generate .box files from annotation CSV data for handwritten Baybayin word images.

The CSV is expected to contain Roboflow-style rows:
    filename,width,height,class,xmin,ymin,xmax,ymax
where `class` encodes the ordinal position of the glyph cluster within the word
and optional decimal suffixes denote associated marks:
    .1 – upper kudlit (ᜒ)
    .2 – lower kudlit (ᜓ)
    .3 – pamudpod / virama (᜔)

Ground-truth text files (.gt.txt) are used to map cluster indices to the actual
Baybayin characters so each bounding box is labeled correctly.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

MARK_TOP = "\u1712"  # ᜒ
MARK_BOTTOM = "\u1713"  # ᜓ
MARK_PAM = "\u1714"  # ᜔
MARKS = {MARK_TOP, MARK_BOTTOM, MARK_PAM}
SPACE_MARK_SUFFIX = "space"
MARK_CHAR_MAP = {
    "top": MARK_TOP,
    "bottom": MARK_BOTTOM,
    "pam": MARK_PAM,
    SPACE_MARK_SUFFIX: " ",
}


def csv_name_to_base(filename: str) -> str:
    stem = Path(filename).stem
    if "_png.rf." in stem:
        stem = stem.split("_png.rf.", 1)[0]
    return stem


def load_gt(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Missing GT file: {path}")
    text = path.read_text(encoding="utf-8")
    return "".join(ch for ch in text if ch not in {" ", "\n", "\r", "\t"})


def build_clusters(text: str) -> List[Dict[str, List[str]]]:
    clusters: List[Dict[str, List[str]]] = []
    current: Dict[str, List[str]] | None = None
    for ch in text:
        if ch in MARKS:
            if current is None:
                raise ValueError("Encountered mark without base glyph.")
            current["marks"].append(ch)
        else:
            current = {"base": ch, "marks": []}
            clusters.append(current)
    return clusters


def select_mark(cluster: Dict[str, List[str]], kind: str) -> str:
    for mark in cluster["marks"]:
        if kind == "top" and mark == MARK_TOP:
            return mark
        if kind == "bottom" and mark == MARK_BOTTOM:
            return mark
        if kind == "pam" and mark == MARK_PAM:
            return mark
    raise ValueError(f"Cluster {cluster['base']} missing mark '{kind}'.")


def class_to_kind(class_str: str) -> Tuple[int, str]:
    if "." not in class_str:
        return int(float(class_str)), "base"
    base_str, frac_str = class_str.split(".", 1)
    frac = int(frac_str)
    kind = {1: "top", 2: "bottom", 3: "pam", 4: SPACE_MARK_SUFFIX}.get(frac)
    if kind is None:
        raise ValueError(f"Unsupported class suffix: {class_str}")
    return int(base_str), kind


def write_box(path: Path, lines: List[str]) -> None:
    content = "\n".join(lines)
    if content:
        content += "\n"
    path.write_text(content, encoding="utf-8")


def process_annotations(
    csv_path: Path,
    images_dir: Path,
    gt_dir: Path,
    output_dir: Path,
) -> Tuple[int, int]:
    groups: Dict[str, List[dict]] = defaultdict(list)
    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            groups[row["filename"]].append(row)

    total_images = len(groups)
    written = 0

    for filename, rows in groups.items():
        base_name = csv_name_to_base(filename)
        csv_image_path = images_dir / filename
        if csv_image_path.exists():
            image_path = csv_image_path
        else:
            # Fall back to looking for processed PNG versions with the same stem.
            stem = Path(filename).stem
            image_path = None
            for ext in (".png", ".PNG", ".jpg", ".JPG"):
                candidate = images_dir / f"{stem}{ext}"
                if candidate.exists():
                    image_path = candidate
                    break
            if image_path is None:
                raise FileNotFoundError(f"Image not found for {base_name}")

        gt_path = gt_dir / f"{base_name}.gt.txt"
        if not gt_path.exists():
            gt_path = images_dir / f"{base_name}.gt.txt"
        text = load_gt(gt_path)
        clusters = build_clusters(text)

        grouped: Dict[int, Dict[str, str]] = defaultdict(dict)
        cluster_order: List[int] = []

        for row in rows:
            idx, kind = class_to_kind(row["class"])
            if idx < 1 or idx > len(clusters):
                raise ValueError(f"Cluster index {idx} out of range for {base_name}")
            cluster = clusters[idx - 1]
            if kind == "base":
                label = cluster["base"]
            else:
                label = MARK_CHAR_MAP.get(kind, cluster["base"])
            if idx not in cluster_order:
                cluster_order.append(idx)

            height = int(row["height"])
            left = int(row["xmin"])
            right = int(row["xmax"])
            bottom = height - int(row["ymax"])
            top = height - int(row["ymin"])
            line = f"{label} {left} {bottom} {right} {top} 0"
            grouped[idx][kind] = line
            if kind not in {"base", SPACE_MARK_SUFFIX} and "base" not in grouped[idx]:
                grouped[idx]["base"] = f"{cluster['base']} {left} {bottom} {right} {top} 0"

        # Ensure any clusters that only had mark annotations still get included in sorted order.
        for idx in sorted(grouped):
            if idx not in cluster_order:
                cluster_order.append(idx)

        lines: List[str] = []
        for idx in cluster_order:
            entry = grouped[idx]
            base_line = entry.get("base")
            if base_line is None:
                raise ValueError(f"{base_name}: missing base glyph for cluster {idx}")
            lines.append(base_line)
            for mark_kind in ("top", "bottom", "pam", SPACE_MARK_SUFFIX):
                mark_line = entry.get(mark_kind)
                if mark_line:
                    lines.append(mark_line)

        out_path = output_dir / f"{base_name}.box"
        write_box(out_path, lines)
        written += 1

    return total_images, written


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert annotation CSV to Tesseract .box files.")
    parser.add_argument("--csv", type=Path, required=True, help="Annotation CSV path.")
    parser.add_argument("--images-dir", type=Path, required=True, help="Directory containing images (.png/.jpg).")
    parser.add_argument("--gt-dir", type=Path, required=True, help="Directory containing .gt.txt files.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory where .box files will be written.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    total, written = process_annotations(
        args.csv.resolve(),
        args.images_dir.resolve(),
        args.gt_dir.resolve(),
        args.output_dir.resolve(),
    )
    print(f"Wrote boxes for {written}/{total} images.")


if __name__ == "__main__":
    main()
