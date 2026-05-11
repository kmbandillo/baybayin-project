#!/usr/bin/env python3
"""
Reorder rows and columns of a confusion matrix CSV to match a specified glyph order.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List


def reorder_confusion_csv(
    input_csv: Path,
    output_csv: Path,
    target_glyphs: List[str],
    as_percent: bool,
) -> None:
    """Reorder confusion matrix rows/cols to match target_glyphs order."""
    
    # Read input CSV
    rows_dict = {}
    header = None
    with input_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            label = row[0]
            rows_dict[label] = row[1:]
    
    # Extract original column order (skip "truth\pred", exclude last if <DEL>)
    original_cols = header[1:]
    if original_cols and original_cols[-1] == "<DEL>":
        original_cols = original_cols[:-1]
        has_del = True
    else:
        has_del = False
    
    # Build mapping from original column indices
    col_map = {col: idx for idx, col in enumerate(original_cols)}
    
    # Reorder: use target_glyphs + any missing columns at end + <DEL> if present
    reordered_cols = target_glyphs.copy()
    for col in original_cols:
        if col not in reordered_cols:
            reordered_cols.append(col)
    if has_del:
        reordered_cols.append("<DEL>")
    
    # Write reordered CSV
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        new_header = ["truth\\pred"] + reordered_cols
        writer.writerow(new_header)
        
        for glyph in reordered_cols:
            if glyph in {"<DEL>", "<INS>"}:
                continue
            if glyph not in rows_dict:
                continue
            
            row_data = rows_dict[glyph]
            new_row = [glyph]
            
            for col in reordered_cols:
                if col == "<DEL>":
                    # <DEL> is the last column in original
                    if has_del:
                        new_row.append(row_data[-1])
                    else:
                        new_row.append("0")
                elif col in col_map:
                    orig_idx = col_map[col]
                    new_row.append(row_data[orig_idx])
                else:
                    new_row.append("0")
            
            writer.writerow(new_row)
        
        # <INS> row (if present in original)
        if "<INS>" in rows_dict:
            row_data = rows_dict["<INS>"]
            new_row = ["<INS>"]
            
            for col in reordered_cols:
                if col == "<DEL>":
                    # For <INS> row, <DEL> column may be empty or special
                    new_row.append(row_data[-1] if has_del else "")
                elif col in col_map:
                    orig_idx = col_map[col]
                    new_row.append(row_data[orig_idx])
                else:
                    new_row.append("0")
            
            writer.writerow(new_row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Input confusion CSV.")
    parser.add_argument("--output", type=Path, required=True, help="Output reordered CSV.")
    parser.add_argument(
        "--glyphs",
        type=str,
        required=True,
        help="Comma-separated list of glyphs in desired order.",
    )
    parser.add_argument(
        "--percent",
        action="store_true",
        help="Input is in percent format (informational only).",
    )
    args = parser.parse_args()
    
    target_glyphs = [g.strip() for g in args.glyphs.split(",")]
    reorder_confusion_csv(args.input, args.output, target_glyphs, args.percent)
    print(f"Reordered confusion matrix written to {args.output}")


if __name__ == "__main__":
    main()
