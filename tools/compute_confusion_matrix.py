#!/usr/bin/env python3
"""
Compute a per-glyph confusion matrix from an lstmeval log.

The log must contain repeating pairs of lines that look like:
  Truth:ᜉᜅᜓ
  OCR  :ᜉᜅᜈ

Usage:
  python3 tools/compute_confusion_matrix.py \
      --log releases/tesseract_training/data/baybayin_23char_words_ft_v3/eval_words.log \
      --unicharset releases/tesseract_training/data/baybayin_23char_words_ft_v3/baybayin_23char_words_ft_v3/baybayin_23char_words_ft_v3.unicharset \
      --output releases/tesseract_training/data/baybayin_23char_words_ft_v3/confusion_matrix.csv
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple


def parse_pairs(log_path: Path) -> List[Tuple[str, str]]:
    """Return ordered list of (truth, ocr) strings."""
    pairs: List[Tuple[str, str]] = []
    pending_truth: Optional[str] = None
    with log_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line.startswith("Truth:"):
                pending_truth = line.split("Truth:", 1)[1]
            elif line.startswith("OCR  :") and pending_truth is not None:
                ocr = line.split("OCR  :", 1)[1]
                pairs.append((pending_truth, ocr))
                pending_truth = None
    return pairs


def parse_unicharset(unicharset_path: Path) -> List[str]:
    """Extract glyph order from a unicharset file."""
    glyphs: List[str] = []
    with unicharset_path.open("r", encoding="utf-8") as handle:
        first = handle.readline()  # count, unused
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            token = line.split()[0]
            if token in {"NULL", "Joined"} or token.startswith("|"):
                continue
            glyphs.append(token)
    return glyphs


def parse_glyph_order_csv(order_csv_path: Path) -> List[str]:
    """Extract glyph order from a confusion CSV header (excluding <DEL>)."""
    with order_csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
    # Header shape: truth\pred,<glyph1>,...,<glyphN>,<DEL>
    glyphs = header[1:]
    if glyphs and glyphs[-1] == "<DEL>":
        glyphs = glyphs[:-1]
    return glyphs


def levenshtein_align(truth: str, ocr: str) -> List[Tuple[Optional[str], Optional[str]]]:
    """Align two strings with Levenshtein distance (unit costs)."""
    t = list(truth)
    o = list(ocr)
    m, n = len(t), len(o)
    # dp[i][j] = cost to align t[:i] with o[:j]
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    back = [[None] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        dp[i][0] = i
        back[i][0] = "del"
    for j in range(1, n + 1):
        dp[0][j] = j
        back[0][j] = "ins"

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost_sub = dp[i - 1][j - 1] + (t[i - 1] != o[j - 1])
            cost_del = dp[i - 1][j] + 1
            cost_ins = dp[i][j - 1] + 1
            best = min(cost_sub, cost_del, cost_ins)
            dp[i][j] = best
            if best == cost_sub:
                back[i][j] = "sub"
            elif best == cost_del:
                back[i][j] = "del"
            else:
                back[i][j] = "ins"

    aligned: List[Tuple[Optional[str], Optional[str]]] = []
    i, j = m, n
    while i > 0 or j > 0:
        op = back[i][j]
        if op == "sub":
            aligned.append((t[i - 1], o[j - 1]))
            i -= 1
            j -= 1
        elif op == "del":
            aligned.append((t[i - 1], None))
            i -= 1
        elif op == "ins":
            aligned.append((None, o[j - 1]))
            j -= 1
        else:
            # Should only happen at origin when strings are empty.
            break
    aligned.reverse()
    return aligned


def compute_confusion(
    pairs: Iterable[Tuple[str, str]]
) -> Tuple[defaultdict[str, Counter], Counter, Counter]:
    """Compute per-character counts."""
    matrix: defaultdict[str, Counter] = defaultdict(Counter)
    deletions: Counter = Counter()
    insertions: Counter = Counter()

    for truth, ocr in pairs:
        for t_char, o_char in levenshtein_align(truth, ocr):
            if t_char is None and o_char is not None:
                insertions[o_char] += 1
            elif o_char is None and t_char is not None:
                deletions[t_char] += 1
            elif t_char is not None and o_char is not None:
                matrix[t_char][o_char] += 1
    return matrix, deletions, insertions


def write_csv(
    output_path: Path,
    glyphs: Sequence[str],
    matrix: defaultdict[str, Counter],
    deletions: Counter,
    insertions: Counter,
    as_percent: bool,
) -> None:
    header = ["truth\\pred"] + list(glyphs) + ["<DEL>"]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for truth in glyphs:
            row = [truth]
            row_total = sum(matrix[truth].get(pred, 0) for pred in glyphs) + deletions.get(truth, 0)
            for pred in glyphs:
                value = matrix[truth].get(pred, 0)
                if as_percent:
                    row.append(format_percent(value, row_total))
                else:
                    row.append(value)
            if as_percent:
                row.append(format_percent(deletions.get(truth, 0), row_total))
            else:
                row.append(deletions.get(truth, 0))
            writer.writerow(row)
        ins_row = ["<INS>"]
        total_insertions = sum(insertions.get(pred, 0) for pred in glyphs)
        for pred in glyphs:
            value = insertions.get(pred, 0)
            if as_percent:
                ins_row.append(format_percent(value, total_insertions))
            else:
                ins_row.append(value)
        ins_row.append("100%" if as_percent and total_insertions else "")
        writer.writerow(ins_row)


def format_percent(value: int, total: int) -> str:
    if total <= 0:
        return "0%"
    return f"{(value / total) * 100:.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a glyph confusion matrix from an lstmeval log.")
    parser.add_argument("--log", required=True, type=Path, help="Path to eval log that contains Truth/OCR lines.")
    parser.add_argument("--unicharset", required=True, type=Path, help="Corresponding unicharset file.")
    parser.add_argument("--output", required=True, type=Path, help="CSV file to write.")
    parser.add_argument(
        "--exclude",
        action="append",
        dest="exclude",
        default=[],
        help="Glyph to exclude (pass multiple times for multiple glyphs).",
    )
    parser.add_argument(
        "--percent",
        action="store_true",
        help="Output percentages instead of raw counts.",
    )
    parser.add_argument(
        "--glyph_order_csv",
        type=Path,
        default=None,
        help="Optional confusion CSV whose header glyph order should be reused.",
    )
    args = parser.parse_args()

    if not args.log.exists():
        raise SystemExit(f"Log file not found: {args.log}")
    if not args.unicharset.exists():
        raise SystemExit(f"Unicharset not found: {args.unicharset}")

    pairs = parse_pairs(args.log)
    if not pairs:
        raise SystemExit(f"No Truth/OCR pairs found in {args.log}")
    glyphs = parse_unicharset(args.unicharset)
    exclude_set = set(args.exclude or [])

    if args.glyph_order_csv is not None:
        if not args.glyph_order_csv.exists():
            raise SystemExit(f"Glyph-order CSV not found: {args.glyph_order_csv}")
        ordered = [g for g in parse_glyph_order_csv(args.glyph_order_csv) if g not in exclude_set]
        # Keep any remaining glyphs from unicharset at the end for robustness.
        ordered_set = set(ordered)
        trailing = [g for g in glyphs if g not in ordered_set and g not in exclude_set]
        glyphs = ordered + trailing
    else:
        glyphs = [g for g in glyphs if g not in exclude_set]

    if not glyphs:
        raise SystemExit(f"No glyphs parsed from {args.unicharset}")
    matrix, deletions, insertions = compute_confusion(pairs)
    write_csv(args.output, glyphs, matrix, deletions, insertions, args.percent)

    total_chars = sum(sum(row.values()) for row in matrix.values())
    print(f"Wrote confusion matrix for {len(glyphs)} glyphs covering {total_chars} aligned characters to {args.output}")


if __name__ == "__main__":
    main()
