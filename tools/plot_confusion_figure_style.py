#!/usr/bin/env python3
"""Render a figure-style confusion matrix with precision/recall margins.

The plot mimics the format where:
- Inner cells show count (top) and global percent (bottom)
- Last column shows per-class precision and its complement
- Last row shows per-class recall and its complement
- Bottom-right cell shows overall accuracy and error rate
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import font_manager

INSERTION_LABELS = {"Insertion", "<INS>"}
DELETION_LABELS = {"Deletion", "<DEL>"}

LATIN_TO_BAYBAYIN = {
    "A": "ᜀ",
    "E/I": "ᜁ",
    "O/U": "ᜂ",
    "KA": "ᜃ",
    "GA": "ᜄ",
    "NGA": "ᜅ",
    "TA": "ᜆ",
    "DA/RA": "ᜇ",
    "NA": "ᜈ",
    "PA": "ᜉ",
    "BA": "ᜊ",
    "MA": "ᜋ",
    "YA": "ᜌ",
    "LA": "ᜎ",
    "WA": "ᜏ",
    "SA": "ᜐ",
    "HA": "ᜑ",
    "Kudlit E/I": "ᜒ",
    "Kudlit O/U": "ᜓ",
    "Virama": "᜔",
}


def _to_float(token: str) -> float:
    text = token.strip()
    if not text:
        return 0.0
    if text.endswith("%"):
        text = text[:-1]
    return float(text)


def read_20_class_counts(csv_path: Path, class_count: int) -> Tuple[List[str], np.ndarray]:
    """Read a truth-vs-pred counts CSV and return the class-only matrix.

    Returns:
        labels: class labels in matrix order
        matrix: shape (N, N), rows=true class, cols=predicted class
    """
    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        pred_labels_all = [cell.strip() for cell in header[1:]]

        class_labels = [lbl for lbl in pred_labels_all if lbl not in DELETION_LABELS][:class_count]
        if len(class_labels) < class_count:
            raise ValueError(
                f"Requested {class_count} classes, but only found {len(class_labels)} class columns in {csv_path}."
            )

        pred_index = {lbl: idx for idx, lbl in enumerate(pred_labels_all)}
        rows_by_label: Dict[str, List[float]] = {}

        for row in reader:
            if not row:
                continue
            true_label = row[0].strip()
            if true_label in INSERTION_LABELS:
                continue
            if true_label not in class_labels:
                continue

            values = row[1:]
            row_values: List[float] = []
            for pred_label in class_labels:
                idx = pred_index[pred_label]
                token = values[idx] if idx < len(values) else ""
                row_values.append(_to_float(token))
            rows_by_label[true_label] = row_values

    missing = [lbl for lbl in class_labels if lbl not in rows_by_label]
    if missing:
        raise ValueError(f"Missing rows for classes: {', '.join(missing)}")

    matrix = np.array([rows_by_label[lbl] for lbl in class_labels], dtype=float)
    return class_labels, matrix


def build_figure_grid(
    truth_vs_pred: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
    """Build display values and text annotations for figure-style matrix.

    Returns:
        output_vs_target: shape (N, N), rows=predicted(output), cols=true(target)
        display_values: shape (N+1, N+1), percentages for coloring
        annotations: shape (N+1, N+1), cell text strings
        total: total sample count in the class-only matrix
        precision: shape (N,), per-class precision in percent
        recall: shape (N,), per-class recall in percent
    """
    output_vs_target = truth_vs_pred.T
    n = output_vs_target.shape[0]
    total = float(output_vs_target.sum())
    if total <= 0:
        raise ValueError("Total count is zero; cannot compute percentages.")

    diagonal = np.diag(output_vs_target)
    row_sums = output_vs_target.sum(axis=1)
    col_sums = output_vs_target.sum(axis=0)

    precision = np.divide(diagonal, row_sums, out=np.zeros_like(diagonal), where=row_sums > 0) * 100.0
    recall = np.divide(diagonal, col_sums, out=np.zeros_like(diagonal), where=col_sums > 0) * 100.0
    accuracy = float(diagonal.sum() / total * 100.0)

    global_percent = output_vs_target / total * 100.0

    display_values = np.zeros((n + 1, n + 1), dtype=float)
    display_values[:n, :n] = global_percent
    display_values[:n, n] = precision
    display_values[n, :n] = recall
    display_values[n, n] = accuracy

    annotations = np.empty((n + 1, n + 1), dtype=object)
    for r in range(n):
        for c in range(n):
            count = int(round(output_vs_target[r, c]))
            pct = global_percent[r, c]
            annotations[r, c] = f"{count}\n{pct:.1f}%"

    for r in range(n):
        p = precision[r]
        annotations[r, n] = f"{p:.1f}%\n{100.0 - p:.1f}%"

    for c in range(n):
        rec = recall[c]
        annotations[n, c] = f"{rec:.1f}%\n{100.0 - rec:.1f}%"

    annotations[n, n] = f"{accuracy:.1f}%\n{100.0 - accuracy:.1f}%"

    return output_vs_target, display_values, annotations, total, precision, recall


def render_figure_style(
    labels: List[str],
    display_values: np.ndarray,
    annotations: np.ndarray,
    output_path: Path,
    title: str,
    annotate_fontsize: int,
    glyph_fontsize: int,
    font_path: Path | None,
    no_title: bool,
    dpi: int,
) -> None:
    n = len(labels)
    x_labels = labels + ["Precision"]
    y_labels = labels + ["Recall"]

    glyph_font_prop = None
    if font_path is not None:
        font_manager.fontManager.addfont(str(font_path))
        glyph_font_prop = font_manager.FontProperties(fname=str(font_path))

    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(24, 18))

    sns.heatmap(
        display_values,
        cmap="YlGnBu",
        vmin=0.0,
        vmax=100.0,
        cbar=True,
        cbar_kws={"label": "Percent"},
        square=True,
        annot=annotations,
        fmt="",
        annot_kws={"fontsize": annotate_fontsize},
        linewidths=0.6,
        linecolor="#777777",
        xticklabels=x_labels,
        yticklabels=y_labels,
        ax=ax,
    )

    if not no_title:
        ax.set_title(title, fontsize=14, pad=14)

    ax.set_xlabel("TARGET CLASS", fontsize=12, fontweight="bold")
    ax.set_ylabel("OUTPUT CLASS", fontsize=12, fontweight="bold")

    ax.tick_params(axis="x", labelrotation=45, labelsize=9, pad=24)
    for tick in ax.get_xticklabels():
        tick.set_horizontalalignment("right")
    ax.tick_params(axis="y", labelrotation=0, labelsize=9, pad=38)

    # Add Baybayin glyphs above x-axis Latin labels.
    transform_x = ax.get_xaxis_transform()
    for idx, label in enumerate(labels):
        glyph = LATIN_TO_BAYBAYIN.get(label, "")
        if not glyph:
            continue
        ax.text(
            idx + 0.5,
            -0.02,
            glyph,
            ha="center",
            va="top",
            fontsize=glyph_fontsize,
            transform=transform_x,
            fontproperties=glyph_font_prop,
        )

    # Add Baybayin glyphs to the right of y-axis Latin labels.
    transform_y = ax.get_yaxis_transform()
    for idx, label in enumerate(labels):
        glyph = LATIN_TO_BAYBAYIN.get(label, "")
        if not glyph:
            continue
        ax.text(
            -0.015,
            idx + 0.5,
            glyph,
            ha="right",
            va="center",
            fontsize=glyph_fontsize,
            transform=transform_y,
            fontproperties=glyph_font_prop,
        )

    # Draw separators before summary row/column.
    ax.axvline(n, color="black", linewidth=1.6)
    ax.axhline(n, color="black", linewidth=1.6)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_color("black")

    fig.subplots_adjust(left=0.30, right=0.95, bottom=0.30, top=0.92)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--counts_csv",
        type=Path,
        required=True,
        help="Input confusion-matrix counts CSV (truth rows, predicted columns).",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output PNG path.")
    parser.add_argument(
        "--class_count",
        type=int,
        default=20,
        help="Number of classes to keep from the matrix (default: 20).",
    )
    parser.add_argument(
        "--title",
        default="Baybayin 20-Class Confusion Matrix (Figure-Style)",
        help="Figure title.",
    )
    parser.add_argument(
        "--annotate_fontsize",
        type=int,
        default=9,
        help="Cell annotation font size.",
    )
    parser.add_argument(
        "--glyph_fontsize",
        type=int,
        default=13,
        help="Baybayin axis glyph font size.",
    )
    parser.add_argument(
        "--font",
        type=Path,
        default=Path("font/NotoSansTagalog-Regular.ttf"),
        help="Font path used for Baybayin axis glyphs.",
    )
    parser.add_argument("--no_title", action="store_true", help="Disable title rendering.")
    parser.add_argument("--dpi", type=int, default=300, help="Output DPI.")
    args = parser.parse_args()

    font_path = args.font if args.font and args.font.exists() else None

    labels, truth_vs_pred = read_20_class_counts(args.counts_csv, args.class_count)
    _, display_values, annotations, total, _, _ = build_figure_grid(truth_vs_pred)
    render_figure_style(
        labels,
        display_values,
        annotations,
        args.output,
        args.title,
        args.annotate_fontsize,
        args.glyph_fontsize,
        font_path,
        args.no_title,
        args.dpi,
    )

    diagonal = float(np.trace(truth_vs_pred))
    accuracy = (diagonal / total * 100.0) if total else 0.0
    print(f"Wrote {args.output}")
    print(f"Classes: {len(labels)}")
    print(f"Total samples (class-only): {int(total)}")
    print(f"Overall accuracy (class-only): {accuracy:.2f}%")


if __name__ == "__main__":
    main()