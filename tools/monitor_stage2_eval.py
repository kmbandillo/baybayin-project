#!/usr/bin/env python3
"""
Monitor a Stage 2 training log and export/evaluate checkpoints at milestones.

The script polls the log file, waits until the reported iteration count reaches
each requested milestone, then:
  1. Copies the live checkpoint to a snapshot file.
  2. Runs `lstmtraining --stop_training` to export `.traineddata`.
  3. Runs `lstmeval` on the exported model, saving the log.

Usage example:
  python3 tools/monitor_stage2_eval.py \
      --log releases/.../training_stage2.log \
      --checkpoint releases/.../checkpoints/baybayin_stage2_seq500k_checkpoint \
      --traineddata_template releases/.../baybayin_stage2_seq500k/model_iter{iter}.traineddata \
      --eval_log_template releases/.../baybayin_stage2_seq500k/eval_iter{iter}.log \
      --tessdata_prefix releases/tesseract_training/data \
      --traineddata_base releases/tesseract_training/data/baybayin_stage2_full/baybayin_stage2_full/baybayin_stage2_full.traineddata \
      --eval_list releases/tesseract_training/data/baybayin_stage2_full/list.eval \
      --milestones 50000 100000 ...
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Sequence

ITER_RE = re.compile(r"At iteration (\d+)/")


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True, help="training log to monitor")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="path to the live checkpoint file (will be copied before export)",
    )
    parser.add_argument(
        "--traineddata_base",
        type=Path,
        required=True,
        help="traineddata file used during training (passed to --traineddata).",
    )
    parser.add_argument(
        "--tessdata_prefix",
        type=Path,
        required=True,
        help="TESSDATA_PREFIX to use for lstmtraining/lstmeval calls.",
    )
    parser.add_argument(
        "--traineddata_template",
        type=str,
        required=True,
        help="Template for exported traineddata paths, e.g. '.../model_iter{iter}.traineddata'.",
    )
    parser.add_argument(
        "--eval_log_template",
        type=str,
        required=True,
        help="Template for eval log paths, e.g. '.../eval_iter{iter}.log'.",
    )
    parser.add_argument(
        "--eval_list",
        type=Path,
        required=True,
        help="Eval listfile to feed to lstmeval.",
    )
    parser.add_argument(
        "--milestones",
        type=int,
        nargs="+",
        required=True,
        help="Iteration milestones that should trigger export/eval.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=30.0,
        help="Seconds between log polls.",
    )
    return parser.parse_args(argv)


def read_latest_iteration(log_path: Path) -> int | None:
    """Return the most recent iteration reported in the log, or None if unavailable."""
    if not log_path.exists():
        return None
    latest = None
    with log_path.open("r", errors="ignore") as fh:
        for line in fh:
            match = ITER_RE.search(line)
            if match:
                latest = int(match.group(1))
    return latest


def run_cmd(cmd: list[str], env: dict[str, str], log_path: Path | None = None) -> None:
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w") as fh:
            subprocess.run(cmd, check=True, env=env, stdout=fh, stderr=subprocess.STDOUT)
    else:
        subprocess.run(cmd, check=True, env=env)


def snapshot_checkpoint(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    env = os.environ.copy()
    env["TESSDATA_PREFIX"] = str(args.tessdata_prefix.resolve())
    pending = sorted(args.milestones)
    completed: set[int] = set()

    while pending:
        latest = read_latest_iteration(args.log)
        if latest is not None:
            while pending and latest >= pending[0]:
                milestone = pending.pop(0)
                if milestone in completed:
                    continue
                checkpoint_snapshot = args.checkpoint.parent / f"{args.checkpoint.name}.iter{milestone}"
                snapshot_checkpoint(args.checkpoint, checkpoint_snapshot)
                traineddata_path = Path(
                    args.traineddata_template.format(iter=milestone)
                )
                traineddata_path.parent.mkdir(parents=True, exist_ok=True)
                run_cmd(
                    [
                        "lstmtraining",
                        "--stop_training",
                        "--continue_from",
                        str(checkpoint_snapshot),
                        "--traineddata",
                        str(args.traineddata_base),
                        "--model_output",
                        str(traineddata_path),
                    ],
                    env,
                )
                eval_log_path = Path(args.eval_log_template.format(iter=milestone))
                run_cmd(
                    [
                        "lstmeval",
                        "--model",
                        str(traineddata_path),
                        "--eval_listfile",
                        str(args.eval_list),
                    ],
                    env,
                    log_path=eval_log_path,
                )
                completed.add(milestone)
        time.sleep(args.poll_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(os.sys.argv[1:]))
