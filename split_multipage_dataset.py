import argparse
import os
from typing import List, Tuple

from PIL import Image


def parse_box_file(path: str) -> List[Tuple[str, int, int, int, int, int]]:
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            char = parts[0]
            left, bottom, right, top, page = map(int, parts[1:6])
            entries.append((char, left, bottom, right, top, page))
    return entries


def write_box_file(path: str, char: str, left: int, bottom: int, right: int, top: int) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{char} {left} {bottom} {right} {top} 0\n")


def split_file(base_path: str, dest_dir: str) -> int:
    tif_path = base_path + ".tif"
    box_path = base_path + ".box"
    gt_path = base_path + ".gt.txt"

    if not os.path.exists(tif_path) or not os.path.exists(box_path):
        return 0

    boxes = parse_box_file(box_path)
    if not boxes:
        return 0

    os.makedirs(dest_dir, exist_ok=True)

    with Image.open(tif_path) as img:
        frame_count = getattr(img, "n_frames", 1)
        if frame_count != len(boxes):
            print(f"warning: frame/box mismatch for {tif_path}: {frame_count} vs {len(boxes)}")
        for idx, (char, left, bottom, right, top, page) in enumerate(boxes):
            try:
                img.seek(idx)
            except EOFError:
                print(f"warning: frame {idx} missing for {tif_path}")
                break
            frame = img.copy()
            out_base = os.path.join(dest_dir, f"{os.path.basename(base_path)}_{idx:04d}")
            frame.save(out_base + ".tif")
            with open(out_base + ".gt.txt", "w", encoding="utf-8") as gt_file:
                gt_file.write(char + "\n")
            write_box_file(out_base + ".box", char, left, bottom, right, top)

    return len(boxes)


def main():
    parser = argparse.ArgumentParser(description="Split multi-page TIFF Baybayin dataset into single-page samples.")
    parser.add_argument("--source", required=True, help="Source dataset directory containing multi-page TIFFs.")
    parser.add_argument("--dest", required=True, help="Destination directory for split samples.")
    args = parser.parse_args()

    os.makedirs(args.dest, exist_ok=True)

    total_samples = 0
    for name in os.listdir(args.source):
        if not name.endswith(".tif"):
            continue
        base = os.path.splitext(name)[0]
        base_path = os.path.join(args.source, base)
        created = split_file(base_path, args.dest)
        total_samples += created
        if created:
            print(f"{base}: {created} samples")

    print(f"Total samples created: {total_samples}")


if __name__ == "__main__":
    main()
