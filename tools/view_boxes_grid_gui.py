#!/usr/bin/env python3
"""
Grid viewer for Baybayin TIFF/BOX pairs.

This displays many pages (default 100 = 10x10 grid) at once inside a Tk window.

Usage example:
    python3 tools/view_boxes_grid_gui.py \
        --tif final_version/handwritten/characters/ge_gi.tif \
        --box final_version/handwritten/characters/ge_gi.box \
        --rows 10 --cols 10 --scale 0.6
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

import tkinter as tk
from tkinter import messagebox, ttk

from PIL import Image, ImageDraw, ImageTk


SUPPORTED_IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}


@dataclass
class BoxEntry:
    label: str
    left: int
    bottom: int
    right: int
    top: int

    def as_line(self, page: int) -> str:
        return f"{self.label} {self.left} {self.bottom} {self.right} {self.top} {page}"

    def clone(self) -> "BoxEntry":
        return BoxEntry(self.label, self.left, self.bottom, self.right, self.top)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tk grid viewer for Baybayin box overlays.")
    parser.add_argument("--tif", type=Path, help="Path to multipage TIFF.")
    parser.add_argument("--box", type=Path, help="Path to BOX file.")
    parser.add_argument(
        "--folder",
        type=Path,
        help="Folder containing individual image files (each treated as a page).",
    )
    parser.add_argument(
        "--box-dir",
        type=Path,
        help="Folder containing per-image BOX files (defaults to the image folder).",
    )
    parser.add_argument("--rows", type=int, default=10, help="Rows per batch (default 10).")
    parser.add_argument("--cols", type=int, default=10, help="Columns per batch (default 10).")
    parser.add_argument(
        "--scale",
        type=float,
        default=0.6,
        help="Scaling factor per tile (1.0 keeps original size, <1.0 shrinks).",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=12,
        help="Padding between tiles (in pixels).",
    )
    args = parser.parse_args()
    using_folder = args.folder is not None
    if using_folder:
        if args.tif or args.box:
            parser.error("Use either --folder (with optional --box-dir) or --tif/--box, not both.")
        if args.box_dir is None:
            args.box_dir = args.folder
    else:
        if not args.tif or not args.box:
            parser.error("--tif and --box are required unless --folder is provided.")
        if args.box_dir is not None:
            parser.error("--box-dir is only valid with --folder.")
    return args


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


def load_single_image_boxes(box_path: Path) -> List[BoxEntry]:
    entries: List[BoxEntry] = []
    for raw_line in box_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        label = parts[0]
        try:
            left, bottom, right, top = map(int, parts[1:5])
        except ValueError:
            continue
        entries.append(BoxEntry(label, left, bottom, right, top))
    return entries


def draw_boxes(image: Image.Image, boxes: List[BoxEntry], outline_width: float = 0.5) -> Image.Image:
    rgb = image.convert("RGB")
    draw = ImageDraw.Draw(rgb)
    width, height = rgb.size
    outline_w = max(1, outline_width)
    for entry in boxes:
        x0, x1 = sorted((entry.left, entry.right))
        # Some auto-generated boxes (e.g. blanks/spaces) can end up with inverted Y
        # coordinates; sort to keep Pillow happy and still visualize them.
        y0, y1 = sorted((height - entry.top, height - entry.bottom))
        draw.rectangle([(x0, y0), (x1, y1)], outline="red", width=outline_w)
        # draw.text((x0 + 2, y0 + 2), entry.label, fill="yellow")
    return rgb


def save_boxes(box_path: Path, page_map: Dict[int, List[BoxEntry]]) -> None:
    lines: List[str] = []
    for page in sorted(page_map.keys()):
        for entry in page_map[page]:
            lines.append(entry.as_line(page))
    if not lines:
        box_path.write_text("", encoding="utf-8")
    else:
        box_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class GridViewer(tk.Tk):
    def __init__(
        self,
        tif_path: Optional[Path],
        box_path: Optional[Path],
        folder_path: Optional[Path],
        box_dir: Optional[Path],
        rows: int,
        cols: int,
        scale: float,
        padding: int,
    ) -> None:
        super().__init__()

        self.mode = "folder" if folder_path is not None else "tif"
        self.tif_path = tif_path if self.mode == "tif" else None
        self.box_path = box_path if self.mode == "tif" else None
        self.folder_path = folder_path
        self.box_dir = box_dir if self.mode == "folder" else None

        self.rows = max(1, rows)
        self.cols = max(1, cols)
        self.scale = max(0.1, min(3.0, scale))
        self.padding = max(0, padding)
        self.batch_size = self.rows * self.cols
        self.batch_index = 0
        self.dirty = False  # BOX dirty flag
        self.tif_dirty = False
        self.selection_mode = tk.BooleanVar(value=False)
        self.selected_pages: Set[int] = set()

        self.page_box_paths: Dict[int, Path] = {}
        self.page_names: List[str] = []
        self.page_paths: List[Path] = []
        self.images: List[Image.Image] = []

        if self.mode == "folder":
            (
                self.images,
                self.page_boxes,
                self.page_box_paths,
                self.page_names,
                self.page_paths,
            ) = self._load_folder_images(folder_path, self.box_dir)
            self.source_label = folder_path.name
        else:
            self.page_boxes = load_boxes(box_path)
            with Image.open(tif_path) as img:
                try:
                    idx = 0
                    while True:
                        img.seek(idx)
                        self.images.append(img.copy())
                        idx += 1
                except EOFError:
                    pass
            self.page_names = [f"Page {idx}" for idx in range(len(self.images))]
            self.source_label = tif_path.name
            self.page_paths = []

        if not self.images:
            raise RuntimeError("No pages found in the TIFF file.")
        self._update_title()

        self._build_ui()
        self._bind_events()
        self._render_batch()
        self.protocol("WM_DELETE_WINDOW", self._on_request_exit)

    def _build_ui(self) -> None:
        top = ttk.Frame(self)
        top.pack(side=tk.TOP, fill=tk.X, padx=8, pady=4)

        self.prev_btn = ttk.Button(top, text="◀ Prev 100", command=self.prev_batch)
        self.prev_btn.pack(side=tk.LEFT)

        self.next_btn = ttk.Button(top, text="Next 100 ▶", command=self.next_batch)
        self.next_btn.pack(side=tk.LEFT, padx=(6, 0))

        self.save_btn = ttk.Button(top, text="💾 Save BOX", command=self.save_box_file)
        self.save_btn.pack(side=tk.LEFT, padx=(6, 0))

        self.save_tif_btn = ttk.Button(top, text="💾 Save TIFF", command=self.save_tif_file)
        self.save_tif_btn.pack(side=tk.LEFT, padx=(6, 0))
        if self.mode == "folder":
            self.save_tif_btn.state(["disabled"])

        self.info_var = tk.StringVar()
        ttk.Label(top, textvariable=self.info_var).pack(side=tk.LEFT, padx=12)
        self.tip_var = tk.StringVar(value="Tip: Click an image to edit its boxes.")
        ttk.Label(top, textvariable=self.tip_var).pack(side=tk.RIGHT)

        sel_frame = ttk.Frame(self)
        sel_frame.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(0, 4))
        ttk.Checkbutton(
            sel_frame,
            text="Selection mode",
            variable=self.selection_mode,
            command=self._on_selection_mode_changed,
        ).pack(side=tk.LEFT)
        self.selection_info = tk.StringVar(value="Selected: 0")
        ttk.Label(sel_frame, textvariable=self.selection_info).pack(side=tk.LEFT, padx=8)
        ttk.Button(sel_frame, text="Clear selection", command=self.clear_page_selection).pack(side=tk.LEFT)
        ttk.Button(sel_frame, text="Delete selected files", command=self.delete_selected_files).pack(side=tk.LEFT, padx=8)
        self._update_selection_info()

        body = ttk.Frame(self)
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.paned = ttk.Panedwindow(body, orient=tk.HORIZONTAL)
        self.paned.pack(fill=tk.BOTH, expand=True)

        self.grid_container = ttk.Frame(self.paned)
        self.editor_container = ttk.Frame(self.paned)
        self.paned.add(self.grid_container, weight=4)
        self.paned.add(self.editor_container, weight=1)

        self.canvas = tk.Canvas(self.grid_container, background="#202020", highlightthickness=0)
        self.scroll_x = ttk.Scrollbar(self.grid_container, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.scroll_y = ttk.Scrollbar(self.grid_container, orient=tk.VERTICAL, command=self.canvas.yview)
        self.scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.configure(yscrollcommand=self.scroll_y.set, xscrollcommand=self.scroll_x.set)

        self.inner = ttk.Frame(self.canvas)
        self.canvas_window = self.canvas.create_window(0, 0, anchor="nw", window=self.inner)
        self.inner.bind("<Configure>", self._on_inner_configure)

        self._split_locked = False
        self.editor = PageEditor(self.editor_container, viewer=self)
        self.editor.pack(fill=tk.BOTH, expand=True)
        self.after(100, self._init_pane_split)

    def _bind_events(self) -> None:
        self.bind("<Left>", lambda _: self.prev_batch())
        self.bind("<Right>", lambda _: self.next_batch())
        self.bind("<Prior>", lambda _: self.prev_batch())
        self.bind("<Next>", lambda _: self.next_batch())
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.bind("<Configure>", lambda _evt: self._maybe_update_pane_split())

    def _on_inner_configure(self, event: tk.Event) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        req_width = self.inner.winfo_reqwidth()
        canvas_width = self.canvas.winfo_width()
        target_width = max(req_width, canvas_width)
        self.canvas.itemconfig(self.canvas_window, width=target_width)

    def _on_canvas_configure(self, event: tk.Event) -> None:
        req_width = self.inner.winfo_reqwidth()
        target_width = max(event.width, req_width)
        self.canvas.itemconfig(self.canvas_window, width=target_width)

    def _init_pane_split(self) -> None:
        width = self.paned.winfo_width()
        if width <= 0:
            self.after(100, self._init_pane_split)
            return
        editor_width = max(260, width // 5)
        self.paned.sashpos(0, width - editor_width)

    def _maybe_update_pane_split(self) -> None:
        if getattr(self, "_split_locked", False):
            return
        self._split_locked = True
        self._init_pane_split()
        self.after(200, self._unlock_split)

    def _unlock_split(self) -> None:
        self._split_locked = False

    def _load_folder_images(
        self,
        folder: Path,
        box_dir: Path,
    ) -> Tuple[List[Image.Image], Dict[int, List[BoxEntry]], Dict[int, Path], List[str], List[Path]]:
        image_files = sorted(
            [p for p in folder.iterdir() if p.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES],
            key=lambda p: p.name.lower(),
        )
        if not image_files:
            raise RuntimeError(f"No supported image files found in {folder}")
        images: List[Image.Image] = []
        page_boxes: Dict[int, List[BoxEntry]] = {}
        box_paths: Dict[int, Path] = {}
        page_names: List[str] = []
        page_paths: List[Path] = []
        for idx, img_path in enumerate(image_files):
            with Image.open(img_path) as img:
                images.append(img.copy())
            page_names.append(img_path.name)
            page_paths.append(img_path)
            page_boxes[idx] = []
            if box_dir is not None:
                box_path = box_dir / f"{img_path.stem}.box"
            else:
                box_path = img_path.with_suffix(".box")
            box_paths[idx] = box_path
            if box_path.exists():
                page_boxes[idx] = load_single_image_boxes(box_path)
        return images, page_boxes, box_paths, page_names, page_paths

    def prev_batch(self) -> None:
        if self.batch_index > 0:
            self.batch_index -= 1
            self._render_batch()

    def next_batch(self) -> None:
        max_index = (len(self.images) - 1) // self.batch_size
        if self.batch_index < max_index:
            self.batch_index += 1
            self._render_batch()

    def _render_batch(self) -> None:
        for widget in self.inner.winfo_children():
            widget.destroy()

        self.photo_refs: List[ImageTk.PhotoImage] = []
        start = self.batch_index * self.batch_size
        end = min(len(self.images), start + self.batch_size)

        tile_w, tile_h = None, None
        for idx in range(start, end):
            page = idx
            row = (idx - start) // self.cols
            col = (idx - start) % self.cols

            img = draw_boxes(self.images[page], self.page_boxes.get(page, []))
            if self.scale != 1.0:
                img = img.resize(
                    (
                        max(1, int(img.width * self.scale)),
                        max(1, int(img.height * self.scale)),
                    ),
                    Image.NEAREST,
                )

            tk_img = ImageTk.PhotoImage(img)
            self.photo_refs.append(tk_img)

            cell = ttk.Frame(self.inner)
            cell.grid(row=row, column=col, padx=self.padding // 2, pady=self.padding // 2, sticky="n")
            if page in self.selected_pages:
                cell.configure(borderwidth=2, relief=tk.SOLID)
            else:
                cell.configure(borderwidth=0, relief=tk.FLAT)
            title = self.page_names[page] if page < len(self.page_names) else f"Page {page}"
            ttk.Label(cell, text=title, anchor="center").pack(side=tk.TOP)
            image_label = ttk.Label(cell, image=tk_img)
            image_label.pack(side=tk.TOP)
            image_label.bind("<Button-1>", lambda _evt, p=page: self._handle_tile_click(p))

            if tile_w is None:
                tile_w, tile_h = img.width, img.height

        total = len(self.images)
        displayed = end - start
        self.info_var.set(
            f"Batch {self.batch_index + 1}/{max(1, (total - 1) // self.batch_size + 1)} "
            f"— Showing pages {start}-{end - 1} of {total}"
        )

    def _handle_tile_click(self, page: int) -> None:
        if self.selection_mode.get():
            if page in self.selected_pages:
                self.selected_pages.remove(page)
            else:
                self.selected_pages.add(page)
            self._update_selection_info()
            self._render_batch()
            self.editor.update_targets_from_grid()
        else:
            self.editor.load_page(page)

    def request_grid_refresh(self) -> None:
        self._render_batch()

    def get_selected_pages(self) -> List[int]:
        return sorted(self.selected_pages)

    def clear_page_selection(self) -> None:
        if self.selected_pages:
            self.selected_pages.clear()
            self._render_batch()
        self._update_selection_info()
        self.editor.update_targets_from_grid()
    def delete_selected_files(self) -> None:
        if self.mode != "folder":
            messagebox.showinfo("Unavailable", "Deleting files is only supported when loading from a folder.")
            return
        targets = sorted(self.selected_pages)
        if not targets:
            messagebox.showinfo("Select pages", "Use selection mode to choose thumbnails to delete.")
            return
        valid_targets = [p for p in targets if p < len(self.page_paths)]
        if not valid_targets:
            messagebox.showinfo("Nothing to delete", "Selected entries no longer exist.")
            return
        names = [self.page_paths[p].name for p in valid_targets]
        confirm = messagebox.askyesno(
            "Delete files?",
            f"Delete {len(valid_targets)} image/box pair(s)?\n" + "\n".join(names[:10]),
        )
        if not confirm:
            return
        for page in sorted(valid_targets, reverse=True):
            img_path = self.page_paths[page]
            box_path = self.page_box_paths.get(page)
            try:
                img_path.unlink()
            except FileNotFoundError:
                pass
            if box_path and box_path.exists():
                try:
                    box_path.unlink()
                except FileNotFoundError:
                    pass
        self.selected_pages.clear()
        self._update_selection_info()
        self._reload_folder_data()

    def _on_selection_mode_changed(self) -> None:
        if self.selection_mode.get():
            self.tip_var.set("Selection mode: click tiles to toggle pages.")
        else:
            self.tip_var.set("Tip: Click an image to edit its boxes.")

    def _update_selection_info(self) -> None:
        self.selection_info.set(f"Selected: {len(self.selected_pages)}")

    def mark_dirty(self) -> None:
        if not self.dirty:
            self.dirty = True
            self._update_title()

    def clear_dirty(self) -> None:
        if self.dirty:
            self.dirty = False
            self._update_title()

    def mark_tif_dirty(self) -> None:
        if not self.tif_dirty:
            self.tif_dirty = True
            self._update_title()

    def clear_tif_dirty(self) -> None:
        if self.tif_dirty:
            self.tif_dirty = False
            self._update_title()

    def _update_title(self) -> None:
        mark = "*" if (self.dirty or self.tif_dirty) else ""
        self.title(f"Box Grid Viewer{mark} - {self.source_label}")
    def _reload_folder_data(self, focus_page: Optional[int] = None) -> None:
        if self.mode != "folder" or not self.folder_path:
            return
        try:
            (
                self.images,
                self.page_boxes,
                self.page_box_paths,
                self.page_names,
                self.page_paths,
            ) = self._load_folder_images(self.folder_path, self.box_dir)
        except RuntimeError as exc:
            messagebox.showinfo("Folder reload", str(exc))
            self.images = []
            self.page_boxes = {}
            self.page_box_paths = {}
            self.page_names = []
            self.page_paths = []
            return
        self.selected_pages.clear()
        self._update_selection_info()
        if not self.images:
            self.editor.current_page = None
            self._render_batch()
            return
        if focus_page is None or focus_page >= len(self.images):
            focus_page = 0
        self.editor.load_page(focus_page)
        self._render_batch()

    def save_box_file(self) -> None:
        if self.mode == "folder":
            self._save_folder_boxes()
            return
        try:
            save_boxes(self.box_path, self.page_boxes)
        except OSError as exc:
            messagebox.showerror("Save failed", f"Could not write BOX file:\n{exc}")
            return
        self.clear_dirty()
        messagebox.showinfo("Saved", f"BOX file updated:\n{self.box_path}")

    def save_tif_file(self) -> None:
        if self.mode == "folder":
            messagebox.showinfo("Not available", "Saving TIFF bundles is disabled in folder mode.")
            return
        if not self.images:
            return
        try:
            base = self.images[0].copy()
            append = [img.copy() for img in self.images[1:]]
            base.save(self.tif_path, save_all=True, append_images=append)
        except OSError as exc:
            messagebox.showerror("Save failed", f"Could not write TIFF file:\n{exc}")
            return
        self.clear_tif_dirty()
        messagebox.showinfo("Saved", f"TIFF file updated:\n{self.tif_path}")

    def _save_folder_boxes(self) -> None:
        if not self.page_box_paths:
            messagebox.showinfo("Nothing to save", "No BOX paths available for folder mode.")
            return
        try:
            for idx in range(len(self.images)):
                box_path = self.page_box_paths.get(idx)
                if box_path is None:
                    continue
                entries = self.page_boxes.get(idx, [])
                if entries:
                    text = "\n".join(entry.as_line(0) for entry in entries) + "\n"
                else:
                    text = ""
                box_path.parent.mkdir(parents=True, exist_ok=True)
                box_path.write_text(text, encoding="utf-8")
        except OSError as exc:
            messagebox.showerror("Save failed", f"Could not write BOX files:\n{exc}")
            return
        self.clear_dirty()
        messagebox.showinfo("Saved", f"BOX files updated in {self.box_dir}")

    def _on_request_exit(self) -> None:
        if self.dirty:
            if not messagebox.askyesno(
                "Discard unsaved changes?",
                "There are unsaved BOX edits. Quit anyway?",
                icon="warning",
            ):
                return
        self.destroy()


class PageEditor(ttk.Frame):
    def __init__(self, parent: tk.Widget, viewer: GridViewer) -> None:
        super().__init__(parent)
        self.viewer = viewer
        self.current_page: Optional[int] = None
        self.boxes: List[BoxEntry] = []
        self.selected_index: Optional[int] = None
        self._key_actions: Dict[str, Callable[[], None]] = {}
        self.target_pages: List[int] = []
        self.paste_dialog: Optional["BoxPasteDialog"] = None

        self.label_var = tk.StringVar()
        self.left_var = tk.StringVar()
        self.bottom_var = tk.StringVar()
        self.right_var = tk.StringVar()
        self.top_var = tk.StringVar()
        self.apply_pages_var = tk.StringVar(value="(none)")
        self.apply_include_image = tk.BooleanVar(value=False)
        self.preview_zoom = tk.DoubleVar(value=2.0)

        self.preview_image: Optional[ImageTk.PhotoImage] = None
        self.header_var = tk.StringVar(value="No page selected")

        self._build_ui()
        self._init_hotkeys()
        self._update_selected_pages_display()
        self._refresh_preview()

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)

        header = ttk.Frame(self)
        header.pack(fill=tk.X, padx=10, pady=(10, 4))
        ttk.Label(header, textvariable=self.header_var, font=("TkDefaultFont", 11, "bold")).pack(anchor="w")

        preview_frame = ttk.LabelFrame(self, text="Preview")
        preview_frame.pack(fill=tk.BOTH, expand=False, padx=10, pady=(0, 6))

        self.preview_canvas = tk.Canvas(preview_frame, background="#101010", height=260)
        self.preview_canvas.pack(fill=tk.BOTH, expand=True)
        self.preview_canvas.bind("<Button-1>", lambda _: self.preview_canvas.focus_set())

        zoom_bar = ttk.Frame(preview_frame)
        zoom_bar.pack(fill=tk.X, padx=4, pady=(4, 0))
        ttk.Label(zoom_bar, text="Zoom").pack(side=tk.LEFT)
        ttk.Button(zoom_bar, text="−", width=3, command=lambda: self._step_zoom(-0.1)).pack(side=tk.LEFT, padx=(6, 2))
        self.zoom_slider = ttk.Scale(
            zoom_bar,
            from_=0.25,
            to=3.0,
            orient=tk.HORIZONTAL,
            variable=self.preview_zoom,
            command=lambda _val: self._on_zoom_change(),
        )
        self.zoom_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        ttk.Button(zoom_bar, text="+", width=3, command=lambda: self._step_zoom(0.1)).pack(side=tk.LEFT, padx=(2, 6))
        self.zoom_label = ttk.Label(zoom_bar, text="100%")
        self.zoom_label.pack(side=tk.LEFT)

        list_frame = ttk.LabelFrame(self, text="Boxes on page")
        list_frame.pack(fill=tk.BOTH, expand=False, padx=10, pady=(4, 0))

        self.box_list = tk.Listbox(list_frame, height=8, exportselection=False)
        self.box_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.box_list.bind("<<ListboxSelect>>", self._on_select_box)

        list_scroll = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.box_list.yview)
        list_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.box_list.configure(yscrollcommand=list_scroll.set)

        fields = ttk.LabelFrame(self, text="Selected box")
        fields.pack(fill=tk.X, padx=10, pady=(8, 0))

        self._add_field(fields, "Label", self.label_var)
        self._add_field(fields, "Left", self.left_var)
        self._add_field(fields, "Bottom", self.bottom_var)
        self._add_field(fields, "Right", self.right_var)
        self._add_field(fields, "Top", self.top_var)

        ttk.Button(self, text="Apply changes", command=self._apply_changes).pack(fill=tk.X, padx=10, pady=(10, 0))
        ttk.Button(self, text="Delete selected box", command=self._delete_selected).pack(fill=tk.X, padx=10, pady=(6, 0))
        ttk.Button(self, text="Paste BOX lines…", command=self._open_paste_dialog).pack(
            fill=tk.X, padx=10, pady=(6, 0)
        )

        clone_frame = ttk.LabelFrame(self, text="Apply to other pages")
        clone_frame.pack(fill=tk.X, padx=10, pady=(10, 10))
        ttk.Label(clone_frame, text="Grid selection target pages").pack(anchor="w")
        ttk.Label(clone_frame, textvariable=self.apply_pages_var).pack(fill=tk.X, pady=(2, 4))
        ttk.Button(
            clone_frame,
            text="Use grid selection",
            command=self._use_grid_selection,
        ).pack(fill=tk.X, pady=(0, 4))
        ttk.Checkbutton(
            clone_frame,
            text="Copy TIFF image content too",
            variable=self.apply_include_image,
        ).pack(anchor="w")
        ttk.Button(
            clone_frame,
            text="Use current boxes on pages",
            command=self._apply_to_pages,
        ).pack(fill=tk.X, pady=(6, 0))

    def _add_field(self, parent: ttk.LabelFrame, label: str, var: tk.StringVar) -> None:
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=2)
        ttk.Label(frame, text=label, width=7).pack(side=tk.LEFT)
        entry = ttk.Entry(frame, textvariable=var)
        entry.pack(side=tk.RIGHT, fill=tk.X, expand=True)

    def load_page(self, page: int) -> None:
        self.current_page = page
        self.header_var.set(f"Page {page}")
        self.boxes = self.viewer.page_boxes.setdefault(page, [])
        self.selected_index = None
        self._populate_box_list()
        self.update_targets_from_grid()
        self._refresh_preview()

    def refresh_current_page(self) -> None:
        if self.current_page is None:
            return
        prev_index = self.selected_index
        self.boxes = self.viewer.page_boxes.setdefault(self.current_page, [])
        self._populate_box_list(preferred_index=prev_index)
        self.update_targets_from_grid()
        self._refresh_preview()

    def handle_external_update(self, page: int) -> None:
        if self.current_page == page:
            self.refresh_current_page()
        else:
            self.update_targets_from_grid()

    def update_targets_from_grid(self) -> None:
        if self.current_page is None:
            self.target_pages = []
        else:
            self.target_pages = [p for p in self.viewer.get_selected_pages() if p != self.current_page]
        self._update_selected_pages_display()

    def _populate_box_list(self, preferred_index: Optional[int] = None) -> None:
        self.box_list.delete(0, tk.END)
        for idx, entry in enumerate(self.boxes):
            self.box_list.insert(tk.END, self._format_entry(idx, entry))
        if self.boxes:
            selection = 0
            if preferred_index is not None and 0 <= preferred_index < len(self.boxes):
                selection = preferred_index
            self.box_list.selection_clear(0, tk.END)
            self.box_list.selection_set(selection)
            self.box_list.see(selection)
            self._on_select_box(None)
        else:
            self._clear_selection()

    def _clear_selection(self) -> None:
        self.selected_index = None
        for var in (self.label_var, self.left_var, self.bottom_var, self.right_var, self.top_var):
            var.set("")

    def _on_select_box(self, _event: Optional[tk.Event]) -> None:
        try:
            selection = int(self.box_list.curselection()[0])
        except (IndexError, ValueError):
            self._clear_selection()
            return
        self.selected_index = selection
        entry = self.boxes[selection]
        self.label_var.set(entry.label)
        self.left_var.set(str(entry.left))
        self.bottom_var.set(str(entry.bottom))
        self.right_var.set(str(entry.right))
        self.top_var.set(str(entry.top))

    def _apply_changes(self) -> None:
        if self.current_page is None:
            messagebox.showinfo("Select page", "Click a page tile in the grid first.")
            return
        if self.selected_index is None:
            messagebox.showinfo("Select box", "Pick a box from the list first.")
            return
        label = self.label_var.get().strip()
        if not label:
            messagebox.showerror("Invalid label", "Label cannot be empty.")
            return
        try:
            left = int(self.left_var.get())
            bottom = int(self.bottom_var.get())
            right = int(self.right_var.get())
            top = int(self.top_var.get())
        except ValueError:
            messagebox.showerror("Invalid coordinates", "Coordinates must be integers.")
            return
        if left >= right or bottom >= top:
            messagebox.showerror("Invalid rectangle", "Left must be < Right and Bottom must be < Top.")
            return
        entry = self.boxes[self.selected_index]
        entry.label = label
        entry.left = left
        entry.bottom = bottom
        entry.right = right
        entry.top = top
        self._after_box_change()

    def _refresh_preview(self) -> None:
        self.preview_canvas.delete("all")
        if self.current_page is None:
            self.preview_canvas.create_text(
                10,
                10,
                anchor="nw",
                fill="#aaaaaa",
                text="Select a page from the grid to begin editing.",
            )
            self.preview_canvas.configure(scrollregion=(0, 0, 300, 200))
            return
        base = self.viewer.images[self.current_page]
        preview = draw_boxes(base, self.boxes, outline_width=1)
        scale = max(0.25, min(3.0, float(self.preview_zoom.get())))
        if scale != 1.0:
            preview = preview.resize(
                (
                    max(1, int(preview.width * scale)),
                    max(1, int(preview.height * scale)),
                ),
                Image.NEAREST,
            )
        self.preview_image = ImageTk.PhotoImage(preview)
        self.preview_canvas.create_image(0, 0, anchor="nw", image=self.preview_image)
        self.preview_canvas.configure(scrollregion=(0, 0, preview.width, preview.height))

    def _init_hotkeys(self) -> None:
        self._key_actions = {
            "w": lambda: self._translate_box(0, 1),
            "s": lambda: self._translate_box(0, -1),
            "a": lambda: self._translate_box(-1, 0),
            "d": lambda: self._translate_box(1, 0),
            "q": lambda: self._resize_box(d_width=-1),
            "e": lambda: self._resize_box(d_width=1),
            "f": lambda: self._adjust_bottom(-1),
            "r": lambda: self._adjust_bottom(1),
        }
        self.preview_canvas.bind("<Key>", self._handle_keypress, add="+")

    def _handle_keypress(self, event: tk.Event) -> Optional[str]:
        if self.current_page is None:
            return None
        keysym = event.keysym.lower()
        action = self._key_actions.get(keysym)
        if not action:
            return None
        if self._is_entry_focused() or self.selected_index is None:
            return None
        action()
        return "break"

    def _is_entry_focused(self) -> bool:
        widget = self.focus_get()
        return isinstance(widget, (tk.Entry, ttk.Entry))

    def _format_entry(self, idx: int, entry: BoxEntry) -> str:
        return f"{idx:03d}: {entry.label} [{entry.left},{entry.bottom},{entry.right},{entry.top}]"

    def _after_box_change(self) -> None:
        if not self.boxes:
            self._clear_selection()
            self._refresh_preview()
            return
        preferred = self.selected_index
        self.viewer.mark_dirty()
        self._populate_box_list(preferred_index=preferred)
        self._refresh_preview()
        self.viewer.request_grid_refresh()

    def _translate_box(self, dx: int, dy: int) -> None:
        if self.selected_index is None:
            return
        entry = self.boxes[self.selected_index]
        entry.left += dx
        entry.right += dx
        entry.bottom += dy
        entry.top += dy
        self._after_box_change()

    def _resize_box(
        self,
        d_width: int = 0,
        d_height: int = 0,
        anchor_x: str = "center",
        anchor_y: str = "center",
    ) -> None:
        if self.selected_index is None:
            return
        entry = self.boxes[self.selected_index]
        changed = False
        if d_width:
            current_w = entry.right - entry.left
            new_w = max(1, current_w + d_width)
            if new_w != current_w:
                if anchor_x == "center":
                    center_x = (entry.left + entry.right) / 2.0
                    entry.left = int(round(center_x - new_w / 2.0))
                    entry.right = entry.left + new_w
                elif anchor_x == "left":
                    entry.right = entry.left + new_w
                elif anchor_x == "right":
                    entry.left = entry.right - new_w
                else:
                    raise ValueError(f"Unsupported anchor_x: {anchor_x}")
                changed = True
        if d_height:
            current_h = entry.top - entry.bottom
            new_h = max(1, current_h + d_height)
            if new_h != current_h:
                if anchor_y == "center":
                    center_y = (entry.bottom + entry.top) / 2.0
                    entry.bottom = int(round(center_y - new_h / 2.0))
                    entry.top = entry.bottom + new_h
                elif anchor_y == "top":
                    entry.bottom = entry.top - new_h
                elif anchor_y == "bottom":
                    entry.top = entry.bottom + new_h
                else:
                    raise ValueError(f"Unsupported anchor_y: {anchor_y}")
                changed = True
        if changed:
            self._after_box_change()

    def _adjust_bottom(self, delta: int) -> None:
        if self.selected_index is None or delta == 0:
            return
        entry = self.boxes[self.selected_index]
        new_bottom = entry.bottom + delta
        if new_bottom >= entry.top:
            new_bottom = entry.top - 1
        if new_bottom == entry.bottom:
            return
        entry.bottom = new_bottom
        self._after_box_change()

    def _apply_to_pages(self) -> None:
        if self.current_page is None:
            messagebox.showinfo("Select page", "Click a page tile in the grid first.")
            return
        if not self.boxes:
            messagebox.showinfo("No boxes", "There are no boxes on this page to replicate.")
            return
        targets = sorted(p for p in self.target_pages if p != self.current_page)
        if not targets:
            messagebox.showinfo(
                "No pages updated",
                "Use selection mode to pick destination pages in the grid (excluding the current page).",
            )
            return
        total_pages = len(self.viewer.images)
        invalid = [p for p in targets if p < 0 or p >= total_pages]
        if invalid:
            messagebox.showerror(
                "Invalid page number",
                f"Pages out of range (0-{total_pages - 1}): {', '.join(map(str, invalid))}",
            )
            return
        updated = 0
        include_image = self.apply_include_image.get()
        source_image = self.viewer.images[self.current_page] if include_image else None
        for page in targets:
            clones = [entry.clone() for entry in self.boxes]
            self.viewer.page_boxes[page] = clones
            if include_image and source_image is not None:
                self.viewer.images[page] = source_image.copy()
            self.viewer.editor.handle_external_update(page)
            updated += 1
        if updated:
            self.viewer.mark_dirty()
            if include_image:
                self.viewer.mark_tif_dirty()
            self.viewer.request_grid_refresh()
            desc = f"Copied {len(self.boxes)} box(es)"
            if include_image:
                desc += " and page image"
            desc += f" to {updated} page(s): {', '.join(map(str, targets))}"
            messagebox.showinfo("Applied", desc)

    def _on_zoom_change(self) -> None:
        orig = float(self.preview_zoom.get())
        clamped = max(0.25, min(3.0, orig))
        if abs(clamped - orig) > 1e-6:
            self.preview_zoom.set(clamped)
            return
        self.zoom_label.config(text=f"{int(clamped * 100)}%")
        self._refresh_preview()

    def _step_zoom(self, delta: float) -> None:
        new_value = float(self.preview_zoom.get()) + delta
        self.preview_zoom.set(max(0.25, min(3.0, new_value)))
        self._on_zoom_change()

    def _update_selected_pages_display(self) -> None:
        if self.target_pages:
            preview = ", ".join(str(p) for p in self.target_pages[:10])
            if len(self.target_pages) > 10:
                preview += ", …"
            self.apply_pages_var.set(preview)
        else:
            self.apply_pages_var.set("(none)")

    def _use_grid_selection(self) -> None:
        if self.current_page is None:
            messagebox.showinfo("Select page", "Click a page tile in the grid first.")
            return
        self.update_targets_from_grid()
        if not self.target_pages:
            messagebox.showinfo(
                "No pages",
                "Turn on selection mode, click destination pages in the grid, then try again.",
            )
            return

    def _open_paste_dialog(self) -> None:
        if self.current_page is None:
            messagebox.showinfo("Select page", "Click a page tile in the grid first.")
            return
        if self.paste_dialog and self.paste_dialog.winfo_exists():
            self.paste_dialog.lift()
            self.paste_dialog.focus_force()
            return
        self.paste_dialog = BoxPasteDialog(
            parent=self,
            page=self.current_page,
            on_submit=self._handle_pasted_boxes,
            on_close=lambda: setattr(self, "paste_dialog", None),
        )

    def _handle_pasted_boxes(self, raw_text: str, page: int) -> None:
        try:
            new_entries = self._parse_box_lines(raw_text, page)
        except ValueError as exc:
            messagebox.showerror("Invalid BOX lines", str(exc))
            return
        if not new_entries:
            messagebox.showinfo("No boxes", "Paste text must contain at least one valid BOX line.")
            return
        start_index = len(self.boxes)
        self.boxes.extend(new_entries)
        self.viewer.mark_dirty()
        self._populate_box_list(preferred_index=start_index)
        self._refresh_preview()
        self.viewer.request_grid_refresh()
        messagebox.showinfo("Boxes added", f"Successfully added {len(new_entries)} box(es).")

    def _parse_box_lines(self, raw_text: str, expected_page: int) -> List[BoxEntry]:
        parsed: List[BoxEntry] = []
        for lineno, raw_line in enumerate(raw_text.splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 6:
                raise ValueError(f"Line {lineno}: Expected 6 columns (label left bottom right top page).")
            label = parts[0]
            try:
                left, bottom, right, top = map(int, parts[1:5])
            except ValueError:
                raise ValueError(f"Line {lineno}: Left/bottom/right/top must be integers.")
            try:
                page = int(parts[5])
            except ValueError:
                raise ValueError(f"Line {lineno}: Page must be an integer.")
            if page != expected_page:
                raise ValueError(
                    f"Line {lineno}: Page {page} does not match current page {expected_page}."
                )
            if left >= right or bottom >= top:
                raise ValueError(f"Line {lineno}: Box must have left < right and bottom < top.")
            parsed.append(BoxEntry(label, left, bottom, right, top))
        return parsed

    def _delete_selected(self) -> None:
        if self.current_page is None:
            messagebox.showinfo("Select page", "Click a page tile in the grid first.")
            return
        if self.selected_index is None or not self.boxes:
            messagebox.showinfo("Select box", "Pick a box from the list first.")
            return
        deleted = self.boxes.pop(self.selected_index)
        self.selected_index = None
        self.viewer.mark_dirty()
        self._populate_box_list()
        self._refresh_preview()
        self.viewer.request_grid_refresh()
        messagebox.showinfo("Box removed", f"Deleted box '{deleted.label}'.")


class BoxPasteDialog(tk.Toplevel):
    def __init__(
        self,
        parent: PageEditor,
        page: int,
        on_submit: Callable[[str, int], None],
        on_close: Callable[[], None],
    ) -> None:
        master = parent.winfo_toplevel()
        super().__init__(master)
        self.title("Paste BOX lines")
        self.page = page
        self.on_submit = on_submit
        self.on_close = on_close
        self.transient(master)
        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", self._cancel)

        body = ttk.Frame(self)
        body.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)
        ttk.Label(
            body,
            text=(
                f"Paste BOX lines for page {page}.\n"
                "Format: <label> <left> <bottom> <right> <top> <page>"
            ),
            justify=tk.LEFT,
        ).pack(anchor="w")
        self.text = tk.Text(body, width=52, height=12)
        self.text.pack(fill=tk.BOTH, expand=True, pady=(6, 10))

        btns = ttk.Frame(body)
        btns.pack(fill=tk.X)
        ttk.Button(btns, text="Insert clipboard", command=self._insert_clipboard).pack(side=tk.LEFT)
        ttk.Button(btns, text="Cancel", command=self._cancel).pack(side=tk.RIGHT, padx=(0, 6))
        ttk.Button(btns, text="Add boxes", command=self._apply).pack(side=tk.RIGHT)

    def _insert_clipboard(self) -> None:
        try:
            data = self.clipboard_get()
        except tk.TclError:
            return
        if data:
            self.text.insert(tk.INSERT, data)

    def _apply(self) -> None:
        raw = self.text.get("1.0", tk.END)
        self.on_submit(raw, self.page)
        self._close()

    def _cancel(self) -> None:
        self._close()

    def _close(self) -> None:
        self.on_close()
        self.grab_release()
        self.destroy()


if __name__ == "__main__":
    args = parse_args()
    app = GridViewer(
        tif_path=args.tif,
        box_path=args.box,
        folder_path=args.folder,
        box_dir=args.box_dir,
        rows=args.rows,
        cols=args.cols,
        scale=args.scale,
        padding=args.padding,
    )
    app.mainloop()
