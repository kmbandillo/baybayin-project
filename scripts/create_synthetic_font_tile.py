#!/usr/bin/env python3
"""Create a 12x59 tile for synthetic Baybayin character fonts."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

IMAGE_EXTENSIONS = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")
DEFAULT_EXCLUDE = {"augmented_characters", "hw_characters"}

# Canonical 59 Baybayin labels in the desired column order with romanized x-axis labels.
CANONICAL_COLUMNS: list[tuple[str, str]] = [
    ("\u1700", "a"),
    ("\u1701", "i/e"),
    ("\u1702", "o/u"),
    ("\u1703", "ka"),
    ("\u1703\u1712", "ke/ki"),
    ("\u1703\u1713", "ko/ku"),
    ("\u1703\u1714", "k"),
    ("\u1704", "ga"),
    ("\u1704\u1712", "ge/gi"),
    ("\u1704\u1713", "go/gu"),
    ("\u1704\u1714", "g"),
    ("\u1705", "nga"),
    ("\u1705\u1712", "nge/ngi"),
    ("\u1705\u1713", "ngo/ngu"),
    ("\u1705\u1714", "ng"),
    ("\u1706", "ta"),
    ("\u1706\u1712", "te/ti"),
    ("\u1706\u1713", "to/tu"),
    ("\u1706\u1714", "t"),
    ("\u1707", "da/ra"),
    ("\u1707\u1712", "de/di"),
    ("\u1707\u1713", "do/du"),
    ("\u1707\u1714", "d/r"),
    ("\u1708", "na"),
    ("\u1708\u1712", "ne/ni"),
    ("\u1708\u1713", "no/nu"),
    ("\u1708\u1714", "n"),
    ("\u1709", "pa"),
    ("\u1709\u1712", "pe/pi"),
    ("\u1709\u1713", "po/pu"),
    ("\u1709\u1714", "p"),
    ("\u170a", "ba"),
    ("\u170a\u1712", "be/bi"),
    ("\u170a\u1713", "bo/bu"),
    ("\u170a\u1714", "b"),
    ("\u170b", "ma"),
    ("\u170b\u1712", "me/mi"),
    ("\u170b\u1713", "mo/mu"),
    ("\u170b\u1714", "m"),
    ("\u170c", "ya"),
    ("\u170c\u1712", "ye/yi"),
    ("\u170c\u1713", "yo/yu"),
    ("\u170c\u1714", "y"),
    ("\u170e", "la"),
    ("\u170e\u1712", "le/li"),
    ("\u170e\u1713", "lo/lu"),
    ("\u170e\u1714", "l"),
    ("\u170f", "wa"),
    ("\u170f\u1712", "we/wi"),
    ("\u170f\u1713", "wo/wu"),
    ("\u170f\u1714", "w"),
    ("\u1710", "sa"),
    ("\u1710\u1712", "se/si"),
    ("\u1710\u1713", "so/su"),
    ("\u1710\u1714", "s"),
    ("\u1711", "ha"),
    ("\u1711\u1712", "he/hi"),
    ("\u1711\u1713", "ho/hu"),
    ("\u1711\u1714", "h"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a synthetic-font Baybayin character tile.")
    parser.add_argument(
        "--characters-root",
        type=Path,
        default=Path("dataset/characters"),
        help="Root folder that contains *_characters font directories.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("visuals/synthetic_fonts_tile_12x59.png"),
        help="Output image path.",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=sorted(DEFAULT_EXCLUDE),
        help="Font folders to exclude from the tile.",
    )
    parser.add_argument(
        "--cell-size",
        type=int,
        default=72,
        help="Square cell size in pixels.",
    )
    parser.add_argument(
        "--cell-padding",
        type=int,
        default=6,
        help="Padding inside each cell around the glyph image.",
    )
    return parser.parse_args()


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    )
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def collect_font_dirs(characters_root: Path, excluded: set[str]) -> list[Path]:
    font_dirs: list[Path] = []
    for path in sorted(characters_root.iterdir()):
        if not path.is_dir():
            continue
        if not path.name.endswith("_characters"):
            continue
        if path.name in excluded:
            continue
        font_dirs.append(path)
    return font_dirs


def read_label(gt_path: Path) -> str:
    text = gt_path.read_text(encoding="utf-8", errors="ignore")
    return text.replace("\ufeff", "").strip()


def find_image_path(font_dir: Path, stem: str) -> Path | None:
    for extension in IMAGE_EXTENSIONS:
        candidate = font_dir / f"{stem}{extension}"
        if candidate.exists():
            return candidate
    return None


def collect_label_images(
    font_dir: Path,
    canonical_set: set[str],
) -> tuple[dict[str, Path], int]:
    mapping: dict[str, Path] = {}
    skipped_noncanonical = 0

    for gt_path in sorted(font_dir.glob("*.gt.txt")):
        label = read_label(gt_path)
        if not label:
            continue
        if label not in canonical_set:
            skipped_noncanonical += 1
            continue

        stem = gt_path.name[: -len(".gt.txt")]
        image_path = find_image_path(font_dir, stem)
        if image_path is None:
            continue

        if label not in mapping:
            mapping[label] = image_path

    return mapping, skipped_noncanonical


def fit_glyph_image(path: Path, max_side: int) -> Image.Image:
    with Image.open(path) as source:
        gray = ImageOps.grayscale(source)
        gray = ImageOps.autocontrast(gray)

        scale = min(max_side / gray.width, max_side / gray.height)
        width = max(1, int(gray.width * scale))
        height = max(1, int(gray.height * scale))

        resized = gray.resize((width, height), Image.Resampling.LANCZOS)
        return resized.convert("RGB")


def draw_tile(
    font_dirs: list[Path],
    per_font_mapping: dict[str, dict[str, Path]],
    output_path: Path,
    cell_size: int,
    cell_padding: int,
) -> None:
    columns = CANONICAL_COLUMNS
    column_labels = [name for _, name in columns]
    canonical_labels = [label for label, _ in columns]

    rows = len(font_dirs)
    cols = len(columns)

    left_margin = 190
    top_margin = 135
    right_margin = 20
    bottom_margin = 20

    grid_width = cols * cell_size
    grid_height = rows * cell_size

    canvas = Image.new(
        "RGB",
        (left_margin + grid_width + right_margin, top_margin + grid_height + bottom_margin),
        "white",
    )
    draw = ImageDraw.Draw(canvas)

    title_font = load_font(28)
    row_font = load_font(16)
    col_font = load_font(13)

    draw.text(
        (left_margin + grid_width // 2, 34),
        "Synthetic Baybayin Fonts Tile (12x59)",
        fill=(0, 0, 0),
        anchor="ma",
        font=title_font,
    )
    draw.text(
        (left_margin + grid_width // 2, 64),
        "Rows: font folders | Columns: character names",
        fill=(60, 60, 60),
        anchor="ma",
        font=col_font,
    )

    for col_index, label in enumerate(column_labels):
        x = left_margin + (col_index * cell_size) + (cell_size // 2)
        draw.text((x, top_margin - 14), label, fill=(35, 35, 35), anchor="ms", font=col_font)

    for row_index, font_dir in enumerate(font_dirs):
        y = top_margin + (row_index * cell_size) + (cell_size // 2)
        row_label = font_dir.name.replace("_characters", "")
        draw.text((left_margin - 10, y), row_label, fill=(10, 10, 10), anchor="rm", font=row_font)

    # Draw major vertical separators after the first 3 vowels and then every 4 columns.
    for col in range(cols + 1):
        x = left_margin + (col * cell_size)
        major = col == 3 or (col > 3 and (col - 3) % 4 == 0)
        draw.line(
            (x, top_margin, x, top_margin + grid_height),
            fill=(165, 165, 165) if major else (215, 215, 215),
            width=2 if major else 1,
        )

    for row in range(rows + 1):
        y = top_margin + (row * cell_size)
        draw.line((left_margin, y, left_margin + grid_width, y), fill=(215, 215, 215), width=1)

    max_side = max(1, cell_size - (2 * cell_padding))

    for row_index, font_dir in enumerate(font_dirs):
        mapping = per_font_mapping[font_dir.name]
        for col_index, label in enumerate(canonical_labels):
            x0 = left_margin + (col_index * cell_size)
            y0 = top_margin + (row_index * cell_size)
            x1 = x0 + cell_size - 1
            y1 = y0 + cell_size - 1

            image_path = mapping.get(label)
            if image_path is None:
                draw.line((x0 + 8, y0 + 8, x1 - 8, y1 - 8), fill=(220, 70, 70), width=1)
                draw.line((x0 + 8, y1 - 8, x1 - 8, y0 + 8), fill=(220, 70, 70), width=1)
                continue

            glyph = fit_glyph_image(image_path, max_side=max_side)
            paste_x = x0 + ((cell_size - glyph.width) // 2)
            paste_y = y0 + ((cell_size - glyph.height) // 2)
            canvas.paste(glyph, (paste_x, paste_y))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def main() -> None:
    args = parse_args()
    characters_root = args.characters_root
    excluded = set(args.exclude)

    if not characters_root.exists():
        raise FileNotFoundError(f"Characters root not found: {characters_root}")

    font_dirs = collect_font_dirs(characters_root, excluded)
    if not font_dirs:
        raise RuntimeError("No font directories found after exclusions.")

    canonical_labels = [label for label, _ in CANONICAL_COLUMNS]
    canonical_set = set(canonical_labels)

    per_font_mapping: dict[str, dict[str, Path]] = {}
    skipped_counts: dict[str, int] = {}

    for font_dir in font_dirs:
        mapping, skipped_noncanonical = collect_label_images(font_dir, canonical_set)
        per_font_mapping[font_dir.name] = mapping
        skipped_counts[font_dir.name] = skipped_noncanonical

    draw_tile(
        font_dirs=font_dirs,
        per_font_mapping=per_font_mapping,
        output_path=args.output,
        cell_size=args.cell_size,
        cell_padding=args.cell_padding,
    )

    print(f"Saved tile: {args.output}")
    print(f"Rows x Columns: {len(font_dirs)} x {len(CANONICAL_COLUMNS)}")
    if len(font_dirs) != 12:
        print(f"Warning: expected 12 rows but found {len(font_dirs)}.")

    for font_dir in font_dirs:
        mapped = len(per_font_mapping[font_dir.name])
        missing = len(CANONICAL_COLUMNS) - mapped
        skipped = skipped_counts[font_dir.name]
        print(
            f"{font_dir.name}: mapped={mapped}, missing={missing}, "
            f"skipped_noncanonical={skipped}"
        )


if __name__ == "__main__":
    main()
