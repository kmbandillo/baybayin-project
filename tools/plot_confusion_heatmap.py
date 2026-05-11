#!/usr/bin/env python3
"""Render a heatmap PNG from a confusion-matrix CSV."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import font_manager

DEFAULT_CLASS_ORDER = [
    "ᜀ",
    "ᜊ",
    "ᜃ",
    "ᜇ",
    "ᜁ",
    "ᜄ",
    "ᜑ",
    "ᜎ",
    "ᜋ",
    "ᜈ",
    "ᜅ",
    "ᜂ",
    "ᜉ",
    "ᜐ",
    "ᜆ",
    "ᜏ",
    "ᜌ",
    "ᜒ",
    "ᜓ",
    "᜔",
]

TRANSLITERATION = {
    "ᜀ": "A",
    "ᜁ": "E/I",
    "ᜂ": "O/U",
    "ᜃ": "KA",
    "ᜄ": "GA",
    "ᜅ": "NGA",
    "ᜆ": "TA",
    "ᜇ": "DA/RA",
    "ᜈ": "NA",
    "ᜉ": "PA",
    "ᜊ": "BA",
    "ᜋ": "MA",
    "ᜌ": "YA",
    "ᜎ": "LA",
    "ᜏ": "WA",
    "ᜐ": "SA",
    "ᜑ": "HA",
    "ᜒ": "Kudlit E/I",
    "ᜓ": "Kudlit O/U",
    "᜔": "Virama",
    "<INS>": "Insertion (FP)",
    "<DEL>": "Deletion (FN)",
    "Precision": "Precision",
    "Recall": "Recall",
}


def reorder_matrix(
    labels: List[str],
    columns: List[str],
    data: List[List[float]],
    order: List[str],
) -> Tuple[List[str], List[str], List[List[float]]]:
    label_index = {label: idx for idx, label in enumerate(labels)}
    column_index = {column: idx for idx, column in enumerate(columns)}
    ordered_labels = [label for label in order if label in label_index]
    ordered_labels.extend(label for label in labels if label not in ordered_labels)
    ordered_columns = [column for column in order if column in column_index]
    ordered_columns.extend(column for column in columns if column not in ordered_columns)
    reordered = [
        [data[label_index[label]][column_index[column]] for column in ordered_columns]
        for label in ordered_labels
    ]
    return ordered_labels, ordered_columns, reordered


def compute_precision_recall_margins(
    labels: List[str],
    columns: List[str],
    counts: List[List[float]],
    scale: float,
) -> Tuple[List[float], List[float], float]:
    label_index = {label: idx for idx, label in enumerate(labels)}
    column_index = {column: idx for idx, column in enumerate(columns)}
    class_labels = [
        label for label in labels if label not in {"<INS>", "<DEL>", "Precision", "Recall"}
    ]
    precision: List[float] = []
    recall: List[float] = []

    total_tp = 0.0
    total_fp = 0.0
    total_fn = 0.0
    for label in class_labels:
        row_idx = label_index[label]
        col_idx = column_index[label]
        true_positive = counts[row_idx][col_idx]
        false_positive = sum(
            counts[r][col_idx] for r, row_label in enumerate(labels) if row_label != label
        )
        false_negative = sum(
            counts[row_idx][c] for c, col_label in enumerate(columns) if col_label != label
        )
        precision_denominator = true_positive + false_positive
        recall_denominator = true_positive + false_negative
        precision.append(
            (true_positive / precision_denominator * scale) if precision_denominator else 0.0
        )
        recall.append((true_positive / recall_denominator * scale) if recall_denominator else 0.0)
        total_tp += true_positive
        total_fp += false_positive
        total_fn += false_negative

    overall_denominator = total_tp + total_fp + total_fn
    overall = (total_tp / overall_denominator * scale) if overall_denominator else 0.0
    return precision, recall, overall


def add_precision_recall_margins(
    labels: List[str],
    columns: List[str],
    data: List[List[float]],
    counts: List[List[float]],
    annot: List[List[str]] | None,
    scale: float,
    decimals: int,
    as_decimal: bool,
    truncate_decimals: bool,
) -> Tuple[List[str], List[str], List[List[float]], List[List[str]] | None]:
    precision, recall, overall = compute_precision_recall_margins(labels, columns, counts, scale)

    def truncate(value: float) -> float:
        factor = 10**decimals
        return math.trunc(value * factor) / factor

    def fmt(value: float) -> str:
        value = truncate(value) if truncate_decimals else value
        suffix = "" if as_decimal else "%"
        return f"{value:.{decimals}f}{suffix}"

    class_columns = [column for column in columns if column not in {"<DEL>", "Precision"}]
    precision_by_column = dict(zip(class_columns, precision))
    recall_by_label = {
        label: value
        for label, value in zip(
            [label for label in labels if label not in {"<INS>", "<DEL>", "Precision", "Recall"}],
            recall,
        )
    }

    new_data = [row[:] + [precision_by_column.get(label, 0.0)] for label, row in zip(labels, data)]
    recall_row = [recall_by_label.get(column, 0.0) for column in columns] + [overall]
    new_data.append(recall_row)

    new_annot = None
    if annot is not None:
        new_annot = [
            row[:] + [fmt(precision_by_column.get(label, 0.0))]
            for label, row in zip(labels, annot)
        ]
        new_annot.append([fmt(recall_by_label.get(column, 0.0)) for column in columns] + [fmt(overall)])

    return labels + ["Recall"], columns + ["Precision"], new_data, new_annot


def parse_csv(
    csv_path: Path, as_percent: bool, include_ins_del: bool
) -> Tuple[List[str], List[str], List[List[float]]]:
    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        columns = header[1:] if include_ins_del else header[1:-1]
        labels: List[str] = []
        data: List[List[float]] = []
        for row in reader:
            label = row[0]
            if not include_ins_del and label == "<INS>":
                continue
            labels.append(label)
            values: List[float] = []
            cells = row[1:] if include_ins_del else row[1:-1]
            for cell in cells:
                token = cell.strip()
                if as_percent and token.endswith("%"):
                    token = token[:-1]
                if not token:
                    values.append(0.0)
                else:
                    values.append(float(token))
            data.append(values)
    return labels, columns, data


def render_heatmap(
    labels: List[str],
    columns: List[str],
    data: List[List[float]],
    annot: List[List[str]] | None,
    output_path: Path,
    title: str,
    font_path: Path | None,
    latin_labels: bool,
    latin_with_baybayin: bool,
    predicted_on_y: bool,
    annotate_fontsize: int,
    no_title: bool,
    figure_width: float,
    figure_height: float,
    plain_ins_del_labels: bool,
    keep_ins_del_corner: bool,
) -> None:
    glyph_set = set(labels) | set(columns)
    plot_rows = labels
    plot_cols = columns
    plot_data = data
    plot_annot = annot
    x_axis_title = "Predicted"
    y_axis_title = "True"

    # Optional transpose for the common "Predicted vs True" layout.
    if predicted_on_y:
        plot_rows = columns
        plot_cols = labels
        plot_data = [list(row) for row in zip(*data)]
        if annot is not None:
            plot_annot = [list(row) for row in zip(*annot)]
        x_axis_title = "True"
        y_axis_title = "Predicted"

    if font_path:
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font_path)).get_name()

    sns.set(color_codes=True)
    plt.figure(figsize=(figure_width, figure_height))

    def format_x_tick(glyph: str) -> str:
        latin = TRANSLITERATION.get(glyph, glyph)
        if plain_ins_del_labels and glyph == "<INS>":
            return "Insertion"
        if plain_ins_del_labels and glyph == "<DEL>":
            return "Deletion"
        if glyph in {"<INS>", "<DEL>"}:
            return latin
        if latin_with_baybayin:
            # Latin stays as the tick label (rotated); Baybayin is drawn separately above.
            return latin
        return latin if latin_labels else glyph

    def format_y_tick(glyph: str) -> str:
        latin = TRANSLITERATION.get(glyph, glyph)
        if plain_ins_del_labels and glyph == "<INS>":
            return "Insertion"
        if plain_ins_del_labels and glyph == "<DEL>":
            return "Deletion"
        if glyph in {"<INS>", "<DEL>"}:
            return latin
        if latin_with_baybayin:
            return f"{latin} {glyph}"
        return latin if latin_labels else glyph

    display_columns = [format_x_tick(col) for col in plot_cols]
    display_labels = [format_y_tick(row) for row in plot_rows]
    annot_kws = {"fontsize": annotate_fontsize} if plot_annot is not None else None
    cmap = plt.cm.Blues.copy()
    cmap.set_bad("white")
    plot_array = np.array(plot_data, dtype=float)
    plot_array = np.ma.masked_where(plot_array == 0, plot_array)
    if not keep_ins_del_corner and "<INS>" in plot_rows and "<DEL>" in plot_cols:
        plot_array[plot_rows.index("<INS>"), plot_cols.index("<DEL>")] = np.ma.masked
        if plot_annot is not None:
            plot_annot[plot_rows.index("<INS>")][plot_cols.index("<DEL>")] = ""

    ax = sns.heatmap(
        plot_array,
        annot=plot_annot if plot_annot is not None else False,
        fmt="" if plot_annot is not None else ".2f",
        annot_kws=annot_kws,
        cmap=cmap,
        xticklabels=display_columns,
        yticklabels=display_labels,
        cbar_kws={},
    )

    if not no_title:
        plt.title(title)
    ax.set_xlabel(x_axis_title)
    ax.set_ylabel(y_axis_title)
    font_prop = font_manager.FontProperties(fname=str(font_path)) if font_path else None
    for label in ax.get_xticklabels():
        label.set_rotation(90)

    # When Latin is rotated and Baybayin is drawn above it, add extra pad to
    # push Latin tick labels further down for readability.
    if latin_with_baybayin:
        ax.tick_params(axis="x", pad=22)

    if latin_with_baybayin:
        # Place Baybayin glyphs above rotated Latin tick labels.
        # This avoids rotating the Baybayin glyphs and matches the requested layout.
        transform_x = ax.get_xaxis_transform()
        for idx, glyph in enumerate(plot_cols):
            if glyph in {"<INS>", "<DEL>"}:
                continue
            ax.text(
                idx + 0.5,
                -0.02,
                glyph,
                ha="center",
                va="top",
                fontsize=11,
                rotation=0,
                transform=transform_x,
                fontproperties=font_prop,
            )

    if font_prop:
        for tick in ax.get_xticklabels():
            if any(ch in glyph_set for ch in tick.get_text()):
                tick.set_fontproperties(font_prop)
        for tick in ax.get_yticklabels():
            if any(ch in glyph_set for ch in tick.get_text()):
                tick.set_fontproperties(font_prop)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.gcf().subplots_adjust(left=0.3, bottom=0.42 if latin_with_baybayin else 0.3)
    plt.savefig(output_path, dpi=300)
    plt.close()


def build_count_percent_annotations(
    counts: List[List[float]],
    percents: List[List[float]],
    percent_decimals: int,
    as_decimal: bool,
    truncate_decimals: bool,
) -> List[List[str]]:
    annotations: List[List[str]] = []
    fmt = f"{{:.{percent_decimals}f}}" if as_decimal else f"{{:.{percent_decimals}f}}%"

    def truncate(value: float, decimals: int) -> float:
        factor = 10**decimals
        return math.trunc(value * factor) / factor

    for row_counts, row_percents in zip(counts, percents):
        row_ann: List[str] = []
        for count, pct in zip(row_counts, row_percents):
            count_int = int(round(count))
            value = truncate(pct, percent_decimals) if truncate_decimals else pct
            row_ann.append(f"{count_int}\n{fmt.format(value)}")
        annotations.append(row_ann)
    return annotations


def to_decimal_fraction(matrix: List[List[float]]) -> List[List[float]]:
    return [[value / 100.0 for value in row] for row in matrix]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Confusion-matrix CSV file.")
    parser.add_argument(
        "--counts_csv",
        type=Path,
        default=None,
        help="Optional counts CSV used to annotate each cell as 'count + percent'.",
    )
    parser.add_argument("--output", type=Path, required=True, help="PNG file to write.")
    parser.add_argument("--title", required=True, help="Plot title.")
    parser.add_argument("--font", type=Path, default=None, help="Font for glyph tick labels.")
    parser.add_argument(
        "--percent",
        action="store_true",
        help="Interpret numbers as percentages (strip percent sign).",
    )
    parser.add_argument(
        "--annotate_counts_percent",
        action="store_true",
        help="Annotate each cell with count and percent (requires --counts_csv and --percent).",
    )
    parser.add_argument(
        "--percent_decimals",
        type=int,
        default=1,
        help="Decimal places used when rendering percent annotations.",
    )
    parser.add_argument(
        "--percent_as_decimal",
        action="store_true",
        help="Convert percent values to decimal fractions (0-1) for coloring and annotations.",
    )
    parser.add_argument(
        "--truncate_decimals",
        action="store_true",
        help="Truncate annotation decimals instead of rounding.",
    )
    parser.add_argument(
        "--annotate_fontsize",
        type=int,
        default=7,
        help="Font size for cell annotations.",
    )
    parser.add_argument(
        "--include_ins_del",
        action="store_true",
        help="Include <INS> row and <DEL> column in the heatmap.",
    )
    parser.add_argument(
        "--latin_labels",
        action="store_true",
        help="Replace Baybayin tick labels with their Latin equivalents.",
    )
    parser.add_argument(
        "--latin_with_baybayin",
        action="store_true",
        help="Use Latin labels and add Baybayin (x: above Latin, y: to the right of Latin).",
    )
    parser.add_argument(
        "--predicted_on_y",
        action="store_true",
        help="Transpose matrix and render y-axis as Predicted, x-axis as True.",
    )
    parser.add_argument(
        "--no_title",
        action="store_true",
        help="Do not render a title.",
    )
    parser.add_argument(
        "--figure_width",
        type=float,
        default=15.0,
        help="Figure width in inches (increase to enlarge cell width).",
    )
    parser.add_argument(
        "--figure_height",
        type=float,
        default=12.0,
        help="Figure height in inches (increase to enlarge cell height).",
    )
    parser.add_argument(
        "--baybayin_order",
        action="store_true",
        help="Render labels in A, BA, KA, DA/RA, E/I, GA, HA, LA, MA, NA, NGA, O/U, PA, SA, TA, WA, YA, kudlits, virama order.",
    )
    parser.add_argument(
        "--show_precision_recall",
        action="store_true",
        help="Append a precision column and recall row computed from --counts_csv.",
    )
    parser.add_argument(
        "--plain_ins_del_labels",
        action="store_true",
        help="Render <INS>/<DEL> as plain Insertion/Deletion labels.",
    )
    parser.add_argument(
        "--keep_ins_del_corner",
        action="store_true",
        help="Keep the <INS> row x <DEL> column cell visible instead of masking it.",
    )
    args = parser.parse_args()

    labels, columns, data = parse_csv(args.input, args.percent, args.include_ins_del)
    if args.baybayin_order:
        labels, columns, data = reorder_matrix(labels, columns, data, DEFAULT_CLASS_ORDER)
    if args.percent and args.percent_as_decimal:
        data = to_decimal_fraction(data)
    annot: List[List[str]] | None = None
    if args.annotate_counts_percent:
        if not args.percent:
            raise SystemExit("--annotate_counts_percent requires --percent (input must be percent CSV).")
        if args.counts_csv is None:
            raise SystemExit("--annotate_counts_percent requires --counts_csv.")
        count_labels, count_columns, counts = parse_csv(args.counts_csv, False, args.include_ins_del)
        if args.baybayin_order:
            count_labels, count_columns, counts = reorder_matrix(
                count_labels, count_columns, counts, DEFAULT_CLASS_ORDER
            )
        if count_labels != labels or count_columns != columns:
            raise SystemExit("Counts CSV labels/columns do not match the input CSV.")
        annot = build_count_percent_annotations(
            counts,
            data,
            args.percent_decimals,
            args.percent_as_decimal,
            args.truncate_decimals,
        )
    elif args.show_precision_recall:
        if args.counts_csv is None:
            raise SystemExit("--show_precision_recall requires --counts_csv.")
        count_labels, count_columns, counts = parse_csv(args.counts_csv, False, args.include_ins_del)
        if args.baybayin_order:
            count_labels, count_columns, counts = reorder_matrix(
                count_labels, count_columns, counts, DEFAULT_CLASS_ORDER
            )
        if count_labels != labels or count_columns != columns:
            raise SystemExit("Counts CSV labels/columns do not match the input CSV.")

    if args.show_precision_recall:
        if args.counts_csv is None:
            raise SystemExit("--show_precision_recall requires --counts_csv.")
        if "counts" not in locals():
            count_labels, count_columns, counts = parse_csv(args.counts_csv, False, args.include_ins_del)
            if args.baybayin_order:
                count_labels, count_columns, counts = reorder_matrix(
                    count_labels, count_columns, counts, DEFAULT_CLASS_ORDER
                )
            if count_labels != labels or count_columns != columns:
                raise SystemExit("Counts CSV labels/columns do not match the input CSV.")
        scale = 1.0 if args.percent_as_decimal else 100.0 if args.percent else max(max(row) for row in data)
        data, counts_for_margin = data, counts
        labels, columns, data, annot = add_precision_recall_margins(
            labels,
            columns,
            data,
            counts_for_margin,
            annot,
            scale,
            args.percent_decimals,
            args.percent_as_decimal,
            args.truncate_decimals,
        )

    render_heatmap(
        labels,
        columns,
        data,
        annot,
        args.output,
        args.title,
        args.font,
        args.latin_labels,
        args.latin_with_baybayin,
        args.predicted_on_y,
        args.annotate_fontsize,
        args.no_title,
        args.figure_width,
        args.figure_height,
        args.plain_ins_del_labels,
        args.keep_ins_del_corner,
    )


if __name__ == "__main__":
    main()
