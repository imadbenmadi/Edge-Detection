"""Create a smaller, Kaggle-uploadable HED-BSDS dataset in "processed" format.

Why:
- The RCF HED-BSDS tarball is large (~14GB extracted).
- Your training code (`ProcessedDataset`) expects:
    <root>/<split>/images/*.png
    <root>/<split>/edges/*.png

This script converts a subset of the RCF "train_pair.lst" pairs into that format.
It can:
- Filter to the "canonical" augmentation folder (0.0_1_0) to reduce duplicates.
- Subsample counts for train/val/test.
- Resize to a maximum side length to shrink disk usage.
- Emit a `manifest.csv` mapping outputs to source files.

Example:
  python scripts/create_small_rcf_dataset.py \
    --rcf-root "datasets/RCF/extracted/HED-BSDS" \
    --out "datasets/RCF_small/processed_HED_BSDS_small" \
    --canonical-only \
    --max-side 320 \
    --train-max 2000 --val-max 200 --test-max 200 \
    --seed 123

Then in Kaggle (or other device), point DATA_ROOT to the produced folder.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image
from tqdm import tqdm


@dataclass(frozen=True)
class Pair:
    img_rel: str
    gt_rel: str


def _read_pairs(list_path: Path) -> list[Pair]:
    pairs: list[Pair] = []
    with list_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(f"Bad line in {list_path}: {line!r}")
            pairs.append(Pair(parts[0], parts[1]))
    return pairs


def _is_canonical(pair: Pair) -> bool:
    # RCF uses folders like train/aug_data/0.0_1_0/<id>.jpg
    # and train/aug_gt/0.0_1_0/<id>.png
    return (
        "train/aug_data/0.0_1_0/" in pair.img_rel.replace("\\", "/")
        and "train/aug_gt/0.0_1_0/" in pair.gt_rel.replace("\\", "/")
    )


def _safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _resize_keep_aspect(img: Image.Image, max_side: int, *, resample: int) -> Image.Image:
    if max_side <= 0:
        return img
    w, h = img.size
    m = max(w, h)
    if m <= max_side:
        return img
    scale = max_side / float(m)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return img.resize((new_w, new_h), resample=resample)


def _hash_id(*parts: str) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(p.encode("utf-8", errors="ignore"))
        h.update(b"\0")
    return h.hexdigest()[:12]


def _iter_splits(
    pairs: list[Pair],
    *,
    seed: int,
    train_max: int,
    val_max: int,
    test_max: int,
) -> dict[str, list[Pair]]:
    rng = random.Random(seed)
    shuffled = list(pairs)
    rng.shuffle(shuffled)

    def _take(n: int, start: int) -> tuple[list[Pair], int]:
        if n <= 0:
            return [], start
        end = min(len(shuffled), start + n)
        return shuffled[start:end], end

    idx = 0
    train, idx = _take(train_max, idx)
    val, idx = _take(val_max, idx)
    test, idx = _take(test_max, idx)

    return {"train": train, "val": val, "test": test}


def _save_png(img: Image.Image, out_path: Path) -> None:
    # Keep outputs small(ish): PNG with optimize on.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, format="PNG", optimize=True)


def build_processed(
    *,
    rcf_root: Path,
    out_root: Path,
    canonical_only: bool,
    max_side: int,
    train_max: int,
    val_max: int,
    test_max: int,
    seed: int,
) -> None:
    rcf_root = rcf_root.resolve()
    out_root = out_root.resolve()

    train_list = rcf_root / "train_pair.lst"
    if not train_list.exists():
        raise FileNotFoundError(f"Missing train_pair.lst at: {train_list}")

    pairs = _read_pairs(train_list)
    if canonical_only:
        pairs = [p for p in pairs if _is_canonical(p)]

    if not pairs:
        raise RuntimeError("No training pairs found after filtering.")

    splits = _iter_splits(pairs, seed=seed, train_max=train_max, val_max=val_max, test_max=test_max)

    # Ensure folder structure
    for split in ("train", "val", "test"):
        _safe_mkdir(out_root / split / "images")
        _safe_mkdir(out_root / split / "edges")

    manifest_path = out_root / "manifest.csv"
    meta_path = out_root / "meta.json"

    rows: list[dict[str, str]] = []

    for split, split_pairs in splits.items():
        if not split_pairs:
            continue

        pbar = tqdm(split_pairs, desc=f"Building {split}", unit="pair")
        for i, pair in enumerate(pbar):
            img_src = (rcf_root / pair.img_rel).resolve()
            gt_src = (rcf_root / pair.gt_rel).resolve()

            if not img_src.exists():
                raise FileNotFoundError(f"Missing image: {img_src}")
            if not gt_src.exists():
                raise FileNotFoundError(f"Missing GT: {gt_src}")

            uid = _hash_id(split, pair.img_rel, pair.gt_rel)
            out_name = f"{split}_{i:06d}_{uid}.png"
            img_out = out_root / split / "images" / out_name
            gt_out = out_root / split / "edges" / out_name

            # Load + resize
            with Image.open(img_src) as im:
                im = im.convert("RGB")
                im = _resize_keep_aspect(im, max_side, resample=Image.BILINEAR)
                _save_png(im, img_out)

            with Image.open(gt_src) as gt:
                gt = gt.convert("L")
                gt = _resize_keep_aspect(gt, max_side, resample=Image.NEAREST)
                _save_png(gt, gt_out)

            rows.append(
                {
                    "split": split,
                    "out_name": out_name,
                    "src_image_rel": pair.img_rel,
                    "src_gt_rel": pair.gt_rel,
                }
            )

    # Write manifest + meta
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["split", "out_name", "src_image_rel", "src_gt_rel"])
        w.writeheader()
        w.writerows(rows)

    meta = {
        "rcf_root": str(rcf_root),
        "out_root": str(out_root),
        "canonical_only": bool(canonical_only),
        "max_side": int(max_side),
        "seed": int(seed),
        "requested": {"train_max": int(train_max), "val_max": int(val_max), "test_max": int(test_max)},
        "produced": {k: len(v) for k, v in splits.items()},
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description="Create a smaller processed dataset from RCF HED-BSDS.")
    p.add_argument(
        "--rcf-root",
        type=str,
        required=True,
        help='Path to extracted HED-BSDS root (contains train_pair.lst), e.g. "datasets/RCF/extracted/HED-BSDS"',
    )
    p.add_argument(
        "--out",
        type=str,
        required=True,
        help='Output folder for processed dataset, e.g. "datasets/RCF_small/processed_HED_BSDS_small"',
    )
    p.add_argument(
        "--canonical-only",
        action="store_true",
        help="Use only the canonical augmentation folder (train/aug_* /0.0_1_0) to reduce size.",
    )
    p.add_argument(
        "--max-side",
        type=int,
        default=320,
        help="Resize so max(width,height) <= this (0 disables). Default: 320.",
    )
    p.add_argument("--train-max", type=int, default=2000, help="Max train pairs to export. Default: 2000.")
    p.add_argument("--val-max", type=int, default=200, help="Max val pairs to export. Default: 200.")
    p.add_argument("--test-max", type=int, default=200, help="Max test pairs to export. Default: 200.")
    p.add_argument("--seed", type=int, default=123, help="Shuffle seed for reproducible splits.")

    args = p.parse_args(argv)

    build_processed(
        rcf_root=Path(args.rcf_root),
        out_root=Path(args.out),
        canonical_only=bool(args.canonical_only),
        max_side=int(args.max_side),
        train_max=int(args.train_max),
        val_max=int(args.val_max),
        test_max=int(args.test_max),
        seed=int(args.seed),
    )

    print("\nDone.")
    print(f"Processed dataset: {Path(args.out).resolve()}")
    print("Structure:")
    print("  train/images + train/edges")
    print("  val/images   + val/edges")
    print("  test/images  + test/edges")
    print("Also wrote: manifest.csv, meta.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
