#!/usr/bin/env python3
"""
Simple GUI for inspecting Baybayin multipage TIFFs with BOX overlays.

Usage:
    python3 tools/view_boxes_gui.py \
        --tif final_version/handwritten/characters/ge_gi.tif \
        --box final_version/handwritten/characters/ge_gi.box

Controls:
    ← / → keys, or Prev / Next buttons to change pages.
    Mouse wheel / +/- keys to zoom in and out.
    Press R to reset zoom.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageTk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GUI viewer for Baybayin boxes.")
    parser.add_argument("--tif", type=Path, required=True, help="Path to multipage TIFF.")
    parser.add_argument("--box", type=Path, required=True, help="Path to BOX file.")
    return parser.parse_args()


def load_boxes(box_path: Path) -> Dict[int, List[Tuple[str, int, int, int, int]]]:
    page_map: Dict[int, List[Tuple[str, int, int, int, int]]] = {}
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
        page_map.setdefault(page, []).append((label, left, bottom, right, top))
    return page_map


class BoxViewer(tk.Tk):
    def __init__(self, tif_path: Path, box_path: Path) -> None:
        super().__init__()
        self.title(f"Box Viewer - {tif_path.name}")

        self.tif_path = tif_path
        self.page_boxes = load_boxes(box_path)

        self.images: List[Image.Image] = []
        with Image.open(tif_path) as img:
            try:
                index = 0
                while True:
                    img.seek(index)
                    self.images.append(img.copy().convert("RGB"))
                    index += 1
            except EOFError:
                pass

        if not self.images:
            raise RuntimeError("No pages found in TIFF.")

        self.current_page = 0
        self.zoom = 1.0

        self._build_widgets()
        self._bind_events()
        self._render_page()

    def _build_widgets(self) -> None:
        top_frame = ttk.Frame(self)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=8, pady=4)

        self.prev_btn = ttk.Button(top_frame, text="Prev", command=self.prev_page)
        self.prev_btn.pack(side=tk.LEFT)

        self.next_btn = ttk.Button(top_frame, text="Next", command=self.next_page)
        self.next_btn.pack(side=tk.LEFT, padx=(4, 0))

        self.info_var = tk.StringVar()
        ttk.Label(top_frame, textvariable=self.info_var).pack(side=tk.LEFT, padx=12)

        self.canvas = tk.Canvas(self, background="#222222")
        self.canvas.pack(fill=tk.BOTH, expand=True)

    def _bind_events(self) -> None:
        self.bind("<Left>", lambda _: self.prev_page())
        self.bind("<Right>", lambda _: self.next_page())
        self.bind("<minus>", lambda _: self.adjust_zoom(0.9))
        self.bind("<plus>", lambda _: self.adjust_zoom(1.1))
        self.bind("<equal>", lambda _: self.adjust_zoom(1.1))
        self.bind("<r>", lambda _: self.reset_zoom())
        self.bind("<MouseWheel>", self._on_wheel)
        if tk.TkVersion >= 8.6:
            self.bind("<Button-4>", lambda _: self.adjust_zoom(1.1))
            self.bind("<Button-5>", lambda _: self.adjust_zoom(0.9))

    def _on_wheel(self, event: tk.Event) -> None:
        factor = 1.1 if event.delta > 0 else 0.9
        self.adjust_zoom(factor)

    def adjust_zoom(self, factor: float) -> None:
        self.zoom = max(0.1, min(5.0, self.zoom * factor))
        self._render_page()

    def reset_zoom(self) -> None:
        self.zoom = 1.0
        self._render_page()

    def prev_page(self) -> None:
        if self.current_page > 0:
            self.current_page -= 1
            self._render_page()

    def next_page(self) -> None:
        if self.current_page < len(self.images) - 1:
            self.current_page += 1
            self._render_page()

    def _render_page(self) -> None:
        page_image = self.images[self.current_page].copy()
        width, height = page_image.size
        draw = ImageDraw.Draw(page_image)

        for label, left, bottom, right, top in self.page_boxes.get(self.current_page, []):
            x0 = left
            y0 = height - top
            x1 = right
            y1 = height - bottom
            draw.rectangle((x0, y0, x1, y1), outline="red", width=2)
            draw.text((x0 + 2, y0 + 2), label, fill="yellow")

        display_w = int(width * self.zoom)
        display_h = int(height * self.zoom)
        resized = page_image.resize((display_w, display_h), Image.NEAREST)

        self.photo = ImageTk.PhotoImage(resized)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))

        total_pages = len(self.images)
        boxes = len(self.page_boxes.get(self.current_page, []))
        self.info_var.set(
            f"Page {self.current_page + 1}/{total_pages} — {boxes} box(es) — Zoom {self.zoom:.2f}×"
        )


if __name__ == "__main__":
    from PIL import ImageDraw  # local import for GUI mode

    args = parse_args()
    viewer = BoxViewer(args.tif, args.box)
    viewer.mainloop()
