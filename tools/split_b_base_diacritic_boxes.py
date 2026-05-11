#!/usr/bin/env python3
"""Split base glyph and diacritic boxes for archive bundles."""

from __future__ import annotations

import argparse
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image


NEIGHBORS = ((-1, 0), (1, 0), (0, -1), (0, 1))
KUDLIT_I = "\u1712"
KUDLIT_O = "\u1713"
VIRAMA = "\u1714"

DIAC_POSITIONS = {
    KUDLIT_I: "top",
    KUDLIT_O: "right",
    VIRAMA: "bottom",
}


@dataclass
class Component:
    index: int
    area: int
    top: int
    bottom: int
    left: int
    right: int


def load_label_map(script_path: Path) -> Dict[str, str]:
    spec = importlib.util.spec_from_file_location(
        "rebox_archive_dataset", script_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.LABEL_MAP  # type: ignore[attr-defined]


def bbox_from_mask(mask: np.ndarray) -> Tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    top = int(ys.min())
    bottom = int(ys.max()) + 1
    left = int(xs.min())
    right = int(xs.max()) + 1
    return left, top, right, bottom


def label_components(mask: np.ndarray, min_area: int) -> Tuple[np.ndarray, np.ndarray, List[Component]]:
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    labels = -np.ones_like(mask, dtype=np.int16)
    components: List[Component] = []
    next_index = 0

    for y in range(height):
        for x in range(width):
            if not mask[y, x] or visited[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = True
            coords: List[Tuple[int, int]] = []
            min_y = max_y = y
            min_x = max_x = x
            area = 0
            while stack:
                cy, cx = stack.pop()
                coords.append((cy, cx))
                area += 1
                if cy < min_y:
                    min_y = cy
                if cy > max_y:
                    max_y = cy
                if cx < min_x:
                    min_x = cx
                if cx > max_x:
                    max_x = cx
                for dy, dx in NEIGHBORS:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            if area < min_area:
                for cy, cx in coords:
                    mask[cy, cx] = False
                continue
            for cy, cx in coords:
                labels[cy, cx] = next_index
            components.append(
                Component(
                    index=next_index,
                    area=area,
                    top=min_y,
                    bottom=max_y + 1,
                    left=min_x,
                    right=max_x + 1,
                )
            )
            next_index += 1

    return mask, labels, components


def clamp(value: int, min_value: int, max_value: int) -> int:
    return max(min_value, min(max_value, value))


def choose_diac_component(
    components: Sequence[Component], base: Component, orientation: str
) -> Component | None:
    base_area = base.area
    limit = max(10, int(base_area * 0.65))
    candidates = [c for c in components if c.index != base.index and c.area <= limit]
    if not candidates:
        return None
    if orientation == "top":
        return min(candidates, key=lambda c: (c.top, c.area))
    if orientation == "bottom":
        return max(candidates, key=lambda c: (c.bottom, -c.area))
    if orientation == "right":
        return max(candidates, key=lambda c: (c.right, -c.area))
    return min(candidates, key=lambda c: c.area)


def projection_split(mask: np.ndarray, orientation: str) -> Tuple[np.ndarray, np.ndarray, str]:
    bbox = bbox_from_mask(mask)
    if bbox is None:
        raise RuntimeError("Empty mask for projection split")
    left, top, right, bottom = bbox
    height = bottom - top
    width = right - left
    projection = mask.astype(bool)
    ratios = (0.28, 0.24, 0.32, 0.2, 0.35)

    for ratio in ratios:
        if orientation in {"top", "bottom"}:
            window = clamp(int(round(height * ratio)), 3, min(10, height))
            if window <= 0:
                continue
            if orientation == "bottom":
                start = max(top + 1, bottom - window)
                diac_mask = np.zeros_like(projection)
                diac_mask[start:bottom, :] = projection[start:bottom, :]
            else:
                end = min(bottom, top + window)
                diac_mask = np.zeros_like(projection)
                diac_mask[top:end, :] = projection[top:end, :]
        elif orientation == "right":
            window = clamp(int(round(width * ratio)), 3, min(10, width))
            if window <= 0:
                continue
            start = max(left + 1, right - window)
            diac_mask = np.zeros_like(projection)
            diac_mask[:, start:right] = projection[:, start:right]
        else:
            raise ValueError(f"Unsupported orientation {orientation}")

        base_mask = projection & ~diac_mask
        if base_mask.any() and diac_mask.any():
            return base_mask, diac_mask, f"projection_{orientation}"

    raise RuntimeError(f"Projection split failed for orientation {orientation}")


def split_masks(mask: np.ndarray, orientation: str) -> Tuple[np.ndarray, np.ndarray, str]:
    working_mask = mask.copy()
    working_mask, labels, components = label_components(working_mask, min_area=3)
    if components:
        base_comp = max(components, key=lambda c: c.area)
        diac_comp = choose_diac_component(components, base_comp, orientation)
        if diac_comp is not None:
            diac_mask = labels == diac_comp.index
            base_mask = working_mask & ~diac_mask
            if base_mask.any() and diac_mask.any():
                return base_mask, diac_mask, "components"
    return projection_split(working_mask, orientation)


def boxes_for_sample(
    path: Path, threshold: int, orientation: str
) -> Tuple[int, Tuple[int, int, int, int], Tuple[int, int, int, int], str]:
    with Image.open(path) as img:
        arr = np.array(img.convert("L"))
    mask = arr < threshold
    base_mask, diac_mask, method = split_masks(mask, orientation)
    base_bbox = bbox_from_mask(base_mask)
    diac_bbox = bbox_from_mask(diac_mask)
    if base_bbox is None or diac_bbox is None:
        raise RuntimeError(f"Failed to derive bounding boxes for {path}")
    return arr.shape[0], base_bbox, diac_bbox, method


def format_box(ch: str, bbox: Tuple[int, int, int, int], height: int, page_idx: int) -> str:
    left, top, right, bottom = bbox
    top = max(0, min(top, height - 1))
    bottom = max(top + 1, min(bottom, height))
    return f"{ch} {left} {height - bottom} {right} {height - top} {page_idx}"


def collect_prefixes(root: Path) -> List[str]:
    prefixes = set()
    for tif_path in root.glob("*.tif"):
        stem = tif_path.stem
        if "_" not in stem:
            continue
        prefix, suffix = stem.rsplit("_", 1)
        if suffix.isdigit():
            prefixes.add(prefix)
    return sorted(prefixes)


def gather_samples(root: Path, prefix: str) -> List[Path]:
    samples: List[Path] = []
    for tif_path in sorted(root.glob(f"{prefix}_*.tif")):
        parts = tif_path.stem.rsplit("_", 1)
        if len(parts) != 2 or not parts[1].isdigit():
            continue
        box_path = tif_path.with_suffix(".box")
        if not box_path.exists() or box_path.stat().st_size == 0:
            continue
        samples.append(tif_path)
    return samples


def determine_targets(
    available: Iterable[str],
    label_map: Dict[str, str],
    explicit: Sequence[str] | None,
    skip: Sequence[str],
    process_all: bool,
) -> List[str]:
    skip_set = set(skip)
    if explicit:
        prefixes = [p for p in explicit if p not in skip_set]
    elif process_all:
        prefixes = [p for p in available if p not in skip_set]
    else:
        prefixes = ["b"] if "b" not in skip_set else []
    output: List[str] = []
    for prefix in prefixes:
        label = label_map.get(prefix)
        if not label or len(label) <= 1:
            continue
        output.append(prefix)
    return sorted(output)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split base glyph and diacritic boxes for archive bundles."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("final_version/handwritten/archive_reboxed"),
        help="Directory containing per-sample TIFF/BOX pairs.",
    )
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=Path("final_version/archive_bundle"),
        help="Directory where bundle BOX files reside.",
    )
    parser.add_argument(
        "--prefixes",
        nargs="*",
        default=None,
        help="Specific prefixes to process (default: b only).",
    )
    parser.add_argument(
        "--skip",
        nargs="*",
        default=[],
        help="Prefixes to skip.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all prefixes found in --input-dir.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=235,
        help="Binary threshold used for mask extraction.",
    )
    args = parser.parse_args()

    label_map = load_label_map(Path(__file__).with_name("rebox_archive_dataset.py"))
    prefixes_available = collect_prefixes(args.input_dir)
    targets = determine_targets(
        prefixes_available, label_map, args.prefixes, args.skip, args.all
    )

    if not targets:
        raise SystemExit("No prefixes selected for processing.")

    total_pages = 0
    total_lines = 0
    for prefix in targets:
        label = label_map[prefix]
        base_char = label[0]
        diac_char = label[1]
        orientation = DIAC_POSITIONS.get(diac_char)
        if orientation is None:
            print(f"Skipping {prefix}: unsupported diacritic {diac_char!r}")
            continue

        samples = gather_samples(args.input_dir, prefix)
        if not samples:
            print(f"Skipping {prefix}: no samples found in {args.input_dir}")
            continue

        lines: List[str] = []
        counts: Dict[str, int] = {}
        for page_idx, path in enumerate(samples):
            height, base_bbox, diac_bbox, method = boxes_for_sample(
                path, args.threshold, orientation
            )
            lines.append(format_box(base_char, base_bbox, height, page_idx))
            lines.append(format_box(diac_char, diac_bbox, height, page_idx))
            counts[method] = counts.get(method, 0) + 1

        out_path = args.bundle_dir / f"{prefix}.box"
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        total_pages += len(samples)
        total_lines += len(lines)
        methods_summary = ", ".join(f"{k}:{v}" for k, v in sorted(counts.items()) if v)
        print(
            f"{prefix}: wrote {len(lines)} lines ({len(samples)} pages) -> {out_path} [{methods_summary}]"
        )

    print(f"Done: {total_lines} box lines written across {total_pages} pages.")


if __name__ == "__main__":
    main()
