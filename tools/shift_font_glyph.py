#!/usr/bin/env python3
"""Shift specific glyphs vertically inside a TTF/OTF font."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from fontTools.ttLib import TTFont


def iter_glyph_names(font: TTFont, targets: Iterable[str]) -> list[str]:
    order = font.getGlyphOrder()
    result: list[str] = []
    for name in order:
        if any(name == tgt or name.startswith(tgt) for tgt in targets):
            result.append(name)
    cmap = font.getBestCmap()
    reverse = {name: code for code, name in cmap.items()}
    for code in (0x1711,):
        name = cmap.get(code)
        if name and name not in result:
            result.append(name)
    return result


def shift_simple(glyph, dy: int, glyf_table) -> None:
    coordinates, end_pts, flags = glyph.getCoordinates(glyf_table)
    coordinates = coordinates.copy()
    coordinates.translate((0, dy))
    glyph.coordinates = coordinates
    glyph.endPtsOfContours = end_pts
    glyph.flags = flags
    glyph.recalcBounds(glyf_table)


def shift_glyph(glyph, dy: int, glyf_table) -> None:
    if dy == 0:
        return
    if glyph.isComposite():
        for comp in glyph.components:
            comp.yOffset += dy
        glyph.recalcBounds(glyf_table)
    else:
        shift_simple(glyph, dy, glyf_table)


def shift_font(path: Path, glyph_targets: list[str], dy: int) -> None:
    font = TTFont(str(path))
    glyf_table = font["glyf"]
    names = iter_glyph_names(font, glyph_targets)
    for name in names:
        glyph = glyf_table[name]
        shift_glyph(glyph, dy, glyf_table)
        print(f"Shifted {name} by {dy} units")
    font.save(str(path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Shift glyphs vertically inside a font")
    parser.add_argument("font", type=Path, help="Path to the font file (TTF/OTF)")
    parser.add_argument("--glyph", action="append", default=["uni1711"], help="Glyph name prefix to shift")
    parser.add_argument("--dy", type=int, required=True, help="Vertical delta in font units (positive moves up)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    shift_font(args.font, args.glyph, args.dy)


if __name__ == "__main__":
    main()
