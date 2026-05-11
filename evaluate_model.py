#!/usr/bin/env python3
"""
Offline evaluation helper for Baybayin Tesseract models.

This script wraps `lstmeval` and summarises the character / word error rates
for a given traineddata file against its corresponding `list.eval`.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional


def discover_paths(model_name: str, project_root: Path) -> Dict[str, Path]:
    data_dir = project_root / "data"
    model_dir = data_dir / model_name
    defaults = {
        "model": data_dir / f"{model_name}.traineddata",
        "listfile": model_dir / "list.eval",
        "tessdata": data_dir,
    }
    return defaults


def run_lstmeval(
    model_path: Path,
    eval_listfile: Path,
    tessdata_prefix: Optional[Path],
    verbosity: int,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    if tessdata_prefix is not None:
        env["TESSDATA_PREFIX"] = str(tessdata_prefix)
    cmd = [
        "lstmeval",
        "--verbosity",
        str(verbosity),
        "--model",
        str(model_path),
        "--eval_listfile",
        str(eval_listfile),
    ]
    return subprocess.run(
        cmd,
        check=False,
        text=True,
        capture_output=True,
        env=env,
    )


def parse_metrics(output: str) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    pattern = re.compile(
        r"Eval Char error rate=([0-9.]+), Word error rate=([0-9.]+)"
    )
    for line in output.splitlines():
        match = pattern.search(line)
        if match:
            metrics["char_error_rate"] = float(match.group(1))
            metrics["word_error_rate"] = float(match.group(2))
    return metrics


def main() -> None:
    project_root = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Evaluate a trained Tesseract model on its eval list via lstmeval."
    )
    parser.add_argument(
        "--model-name",
        default="baybayin",
        help="Model name (used to locate data/<model-name>.traineddata)",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        help="Path to .traineddata (overrides --model-name discovery)",
    )
    parser.add_argument(
        "--listfile",
        type=Path,
        help="Path to list.eval (defaults to data/<model-name>/list.eval)",
    )
    parser.add_argument(
        "--tessdata-prefix",
        type=Path,
        help="Override TESSDATA_PREFIX for lstmeval",
    )
    parser.add_argument(
        "--verbosity",
        type=int,
        default=0,
        help="lstmeval verbosity (default: 0)",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Optional path to write metrics as JSON",
    )
    parser.add_argument(
        "--keep-stdout",
        action="store_true",
        help="Print raw lstmeval output in addition to the summary.",
    )

    args = parser.parse_args()

    defaults = discover_paths(args.model_name, project_root)
    model_path = Path(args.model_path or defaults["model"])
    listfile = Path(args.listfile or defaults["listfile"])
    tessdata_prefix = args.tessdata_prefix or defaults["tessdata"]

    if not model_path.exists():
        parser.error(f"traineddata not found: {model_path}")
    if not listfile.exists():
        parser.error(f"list.eval not found: {listfile}")

    result = run_lstmeval(
        model_path=model_path,
        eval_listfile=listfile,
        tessdata_prefix=tessdata_prefix,
        verbosity=args.verbosity,
    )

    combined_output = (result.stdout or "") + (result.stderr or "")

    if args.keep_stdout or result.returncode != 0:
        sys.stdout.write(combined_output)
        if not combined_output.endswith("\n"):
            print()

    if result.returncode != 0:
        raise SystemExit(f"lstmeval failed with exit code {result.returncode}")

    metrics = parse_metrics(combined_output)
    if not metrics:
        print("Warning: could not parse error rates from lstmeval output.")
    else:
        print(
            f"Eval CER: {metrics['char_error_rate']:.4f}%, "
            f"Eval WER: {metrics['word_error_rate']:.4f}%"
        )

    if args.json_output:
        payload = {
            "model": str(model_path),
            "listfile": str(listfile),
            "metrics": metrics,
        }
        args.json_output.write_text(json.dumps(payload, indent=2))
        print(f"Metrics written to {args.json_output}")


if __name__ == "__main__":
    main()
