#!/usr/bin/env python3
"""
Render Baybayin Namin ground-truth text into PNG/TIFF images using HarfBuzz shaping.

The Baybayin Namin font encodes glyphs on Latin codepoints with virama support expressed
by `=` characters.  This script converts Baybayin Unicode text back into that Latin
encoding, feeds it through HarfBuzz + FreeType to obtain accurate glyph placement, and
pipes the rasterised glyphs into Pillow images.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Sequence, Tuple

import freetype  # type: ignore
import uharfbuzz as hb  # type: ignore
from PIL import Image, ImageOps


MARKS = {"ᜒ", "ᜓ", "᜔"}
PUNCT_MAP = {"᜵": ",", "᜶": "."}
CONSONANT_MAP = {
    "ᜊ": "B",
    "ᜃ": "K",
    "ᜄ": "G",
    "ᜅ": "NG",
    "ᜆ": "T",
    "ᜇ": "D",
    "ᜈ": "N",
    "ᜉ": "P",
    "ᜋ": "M",
    "ᜌ": "Y",
    "ᜎ": "L",
    "ᜏ": "W",
    "ᜐ": "S",
    "ᜑ": "H",
    "\u170d": "R",
}
VOWEL_MAP = {"ᜀ": "A", "ᜁ": "I", "ᜂ": "U"}
MARK_TO_SUFFIX = {"": "A", "ᜒ": "I", "ᜓ": "U", "᜔": "="}
WHITESPACE = {" ", "\u00a0", "\t", "\r", "\n"}


def transliterate_to_font_encoding(text: str) -> str:
    """
    Convert Baybayin Unicode text into the synthetic Latin encoding expected by the
    Baybayin Namin font (upper-case letters with '=' marking virama).
    """
    segments: List[str] = []
    i = 0
    length = len(text)
    while i < length:
        ch = text[i]
        if ch in WHITESPACE:
            segments.append(ch)
            i += 1
            continue

        if ch in PUNCT_MAP:
            segments.append(PUNCT_MAP[ch])
            i += 1
            continue

        base = ch
        i += 1
        mark = ""
        if i < length and text[i] in MARKS:
            mark = text[i]
            i += 1

        if base in VOWEL_MAP:
            if mark:
                raise ValueError(f"Unexpected vowel mark {mark!r} after vowel {base!r}.")
            segments.append(VOWEL_MAP[base])
            continue

        cons = CONSONANT_MAP.get(base)
        if cons is None:
            raise ValueError(f"Unsupported Baybayin base glyph {base!r}.")

        suffix = MARK_TO_SUFFIX.get(mark)
        if suffix is None:
            raise ValueError(f"Unsupported Baybayin mark sequence {mark!r} after {base!r}.")

        segments.append(cons + suffix)

    return "".join(segments)


def segment_baybayin(text: str, *, group_marks: bool = True) -> List[str]:
    """
    Split Baybayin Unicode text into logical tokens (base with optional mark, spaces, newlines).
    """
    tokens: List[str] = []
    i = 0
    length = len(text)
    while i < length:
        ch = text[i]
        if ch == "\n":
            tokens.append("\n")
            i += 1
            continue
        if ch in {" ", "\t", "\u00a0"}:
            tokens.append(" ")
            i += 1
            continue
        base = ch
        i += 1
        mark = ""
        if i < length and text[i] in MARKS:
            mark = text[i]
            i += 1
        if group_marks:
            tokens.append(base + mark)
        else:
            tokens.append(base)
            if mark:
                tokens.append(mark)
    return tokens


@dataclass
class ShapedLine:
    glyph_infos: Sequence[hb.GlyphInfo]
    glyph_positions: Sequence[hb.GlyphPosition]
    latin_text: str


class BaybayinTextRenderer:
    def __init__(
        self,
        font_path: Path,
        font_size: int = 120,
        margin: int = 40,
        line_spacing: float = 1.3,
        crop_margin_ratio: float = 0.25,
    ) -> None:
        self.font_path = Path(font_path)
        if font_size <= 0:
            raise ValueError("font_size must be positive.")
        if margin < 0:
            raise ValueError("margin must be non-negative.")
        if line_spacing <= 0:
            raise ValueError("line_spacing must be positive.")

        self.font_size = font_size
        self.margin = margin
        self.line_spacing = line_spacing
        self.crop_margin_ratio = crop_margin_ratio

        self._font_bytes = self.font_path.read_bytes()
        self._hb_face = hb.Face(self._font_bytes)
        self._hb_font = hb.Font(self._hb_face)
        hb.ot_font_set_funcs(self._hb_font)

        self._ft_face = freetype.Face(str(self.font_path))
        self._ft_face.set_char_size(font_size * 64)
        self._hb_font.scale = (
            self._ft_face.size.x_ppem << 6,
            self._ft_face.size.y_ppem << 6,
        )

        self.ascender = self._ft_face.size.ascender / 64
        self.descender = -self._ft_face.size.descender / 64
        self.baseline_step = (self.ascender + self.descender) * self.line_spacing

    def _shape_line(self, latin_text: str) -> ShapedLine:
        buf = hb.Buffer()
        buf.add_str(latin_text)
        buf.guess_segment_properties()
        hb.shape(self._hb_font, buf, {})
        return ShapedLine(
            glyph_infos=list(buf.glyph_infos),
            glyph_positions=list(buf.glyph_positions),
            latin_text=latin_text,
        )

    def _layout_text(
        self, latin_text: str
    ) -> Tuple[List[ShapedLine], List[str], int, int]:
        lines = latin_text.splitlines() or [latin_text]
        shaped_lines = [self._shape_line(line) for line in lines]

        max_advance = 0.0
        for shaped in shaped_lines:
            advance = sum(pos.x_advance for pos in shaped.glyph_positions) / 64.0
            max_advance = max(max_advance, advance)

        width = max(
            1,
            int(math.ceil(max_advance + 2 * self.margin + 0.5 * self.font_size)),
        )
        height = max(
            1,
            int(
                math.ceil(
                    2 * self.margin
                    + self.ascender
                    + self.descender
                    + self.baseline_step * (len(shaped_lines) - 1)
                )
            ),
        )
        return shaped_lines, lines, width, height

    def _render_internal(
        self, latin_text: str, capture_boxes: bool = False
    ) -> Tuple[Image.Image, List[Tuple[float, float, float, float, int]]]:
        shaped_lines, lines, width, height = self._layout_text(latin_text)
        image = Image.new("L", (width, height), 255)
        boxes: List[Tuple[float, float, float, float, int]] = []
        line_offsets: List[int] = []
        byte_offset = 0
        for idx, line in enumerate(lines):
            line_offsets.append(byte_offset)
            byte_offset += len(line.encode("utf-8"))
            if idx < len(lines) - 1:
                byte_offset += len("\n".encode("utf-8"))

        for line_idx, shaped in enumerate(shaped_lines):
            pen_x = float(self.margin)
            baseline = (
                self.margin + self.ascender + self.baseline_step * line_idx
            )
            pen_y = float(baseline)
            line_byte_offset = line_offsets[line_idx] if line_idx < len(line_offsets) else 0

            for info, pos in zip(shaped.glyph_infos, shaped.glyph_positions):
                gid = info.codepoint
                if gid == 0:
                    if capture_boxes:
                        advance_x = pos.x_advance / 64.0
                        if advance_x != 0:
                            left = pen_x + pos.x_offset / 64.0
                            right = left + advance_x
                            if right < left:
                                left, right = right, left
                            top = pen_y - self.ascender
                            bottom = pen_y + self.descender
                            boxes.append(
                                (
                                    int(round(left)),
                                    int(round(top)),
                                    int(round(right)),
                                    int(round(bottom)),
                                    info.cluster + line_byte_offset,
                                )
                            )
                    pen_x += pos.x_advance / 64.0
                    pen_y += pos.y_advance / 64.0
                    continue
                self._ft_face.load_glyph(
                    gid, freetype.FT_LOAD_RENDER | freetype.FT_LOAD_NO_HINTING
                )
                bitmap = self._ft_face.glyph.bitmap
                width_px, height_px = bitmap.width, bitmap.rows
                if width_px == 0 or height_px == 0:
                    if capture_boxes:
                        advance_x = pos.x_advance / 64.0
                        if advance_x != 0:
                            left = pen_x + pos.x_offset / 64.0
                            right = left + advance_x
                            if right < left:
                                left, right = right, left
                            top = pen_y - self.ascender
                            bottom = pen_y + self.descender
                            boxes.append(
                                (
                                    int(round(left)),
                                    int(round(top)),
                                    int(round(right)),
                                    int(round(bottom)),
                                    info.cluster + line_byte_offset,
                                )
                            )
                    pen_x += pos.x_advance / 64.0
                    pen_y += pos.y_advance / 64.0
                    continue

                glyph_image = Image.frombytes(
                    "L", (width_px, height_px), bytes(bitmap.buffer)
                )
                x_offset = pos.x_offset / 64.0
                y_offset = pos.y_offset / 64.0

                x_pos = pen_x + x_offset + self._ft_face.glyph.bitmap_left
                y_pos = pen_y - y_offset - self._ft_face.glyph.bitmap_top

                left = int(round(x_pos))
                top = int(round(y_pos))
                right = left + width_px
                bottom = top + height_px

                image.paste(0, (left, top, right, bottom), glyph_image)

                if capture_boxes:
                    boxes.append((left, top, right, bottom, info.cluster + line_byte_offset))

                pen_x += pos.x_advance / 64.0
                pen_y += pos.y_advance / 64.0

        bbox = ImageOps.invert(image).getbbox()
        if bbox:
            crop_margin = int(self.margin * self.crop_margin_ratio)
            left = max(bbox[0] - crop_margin, 0)
            upper = max(bbox[1] - crop_margin, 0)
            right = min(bbox[2] + crop_margin, image.width)
            lower = min(bbox[3] + crop_margin, image.height)
            image = image.crop((left, upper, right, lower))
            if capture_boxes:
                boxes = [
                    (
                        x0 - left,
                        y0 - upper,
                        x1 - left,
                        y1 - upper,
                        cluster,
                    )
                    for (x0, y0, x1, y1, cluster) in boxes
                ]

        return image, boxes

    def render(self, latin_text: str) -> Image.Image:
        image, _ = self._render_internal(latin_text, capture_boxes=False)
        return image

    def render_with_boxes(
        self, latin_text: str
    ) -> Tuple[Image.Image, List[Tuple[float, float, float, float, int]]]:
        return self._render_internal(latin_text, capture_boxes=True)


def iter_gt_files(gt_dir: Path) -> Iterator[Path]:
    yield from sorted(gt_dir.glob("*.gt.txt"))


def write_image_pair(image: Image.Image, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_base.with_suffix(".png")
    tif_path = output_base.with_suffix(".tif")

    image.save(png_path)
    image.save(tif_path, format="TIFF", compression="tiff_deflate")


def normalise_gt_text(raw_text: str) -> str:
    return raw_text.rstrip("\n")


def process_split(
    split_name: str,
    gt_dir: Path,
    output_dir: Path,
    renderer: BaybayinTextRenderer,
    font_encoding: str,
) -> None:
    print(f"[{split_name}] rendering from {gt_dir} -> {output_dir}")
    count = 0
    for gt_file in iter_gt_files(gt_dir):
        text = normalise_gt_text(gt_file.read_text(encoding="utf-8"))
        if font_encoding == "latin":
            font_text = transliterate_to_font_encoding(text)
            font_suffix = ".latin.txt"
        else:
            font_text = text
            font_suffix = ".input.txt"
        image = renderer.render(font_text)

        base_name = gt_file.name.replace(".gt.txt", "")
        output_base = output_dir / base_name
        write_image_pair(image, output_base)

        (output_base.with_suffix(".txt")).write_text(text, encoding="utf-8")
        (output_base.with_suffix(".gt.txt")).write_text(text, encoding="utf-8")
        (output_base.with_suffix(font_suffix)).write_text(
            font_text, encoding="utf-8"
        )

        count += 1
        if count % 100 == 0:
            print(f"  rendered {count} samples…")

    print(f"[{split_name}] completed {count} samples.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render Baybayin Namin datasets with HarfBuzz + FreeType."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("full_dataset/Baybayin_Namin"),
        help="Root directory containing *_gt folders.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("full_dataset/Baybayin_Namin_rendered"),
        help="Root directory for rendered *_images output.",
    )
    parser.add_argument(
        "--font-path",
        type=Path,
        default=Path("full_dataset/Baybayin_Namin/BaybayinNamin.otf"),
        help="Path to Baybayin Namin OpenType font.",
    )
    parser.add_argument(
        "--font-size",
        type=int,
        default=128,
        help="Font size in points used for rendering.",
    )
    parser.add_argument(
        "--margin",
        type=int,
        default=48,
        help="Outer margin (pixels) to keep around the rendered text.",
    )
    parser.add_argument(
        "--line-spacing",
        type=float,
        default=1.35,
        help="Line spacing multiplier for multi-line renders.",
    )
    parser.add_argument(
        "--splits",
        nargs="*",
        choices=("characters", "words", "pages"),
        default=("characters", "words", "pages"),
        help="Dataset splits to render.",
    )
    parser.add_argument(
        "--font-encoding",
        choices=("latin", "baybayin"),
        default="latin",
        help="Input encoding expected by the font. 'latin' uses the BaybayinNamin mapping; 'baybayin' passes Unicode text through unchanged.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    renderer = BaybayinTextRenderer(
        font_path=args.font_path,
        font_size=args.font_size,
        margin=args.margin,
        line_spacing=args.line_spacing,
    )

    split_map = {
        "characters": (
            args.dataset_root / "baybayin_namin_character_gt",
            args.output_root / "baybayin_namin_character_images",
        ),
        "words": (
            args.dataset_root / "baybayin_namin_words_gt",
            args.output_root / "baybayin_namin_words_images",
        ),
        "pages": (
            args.dataset_root / "baybayin_namin_pages_gt",
            args.output_root / "baybayin_namin_pages_images",
        ),
    }

    for split in args.splits:
        gt_dir, output_dir = split_map[split]
        process_split(split, gt_dir, output_dir, renderer, args.font_encoding)


if __name__ == "__main__":
    main()
