#!/usr/bin/env python3
"""Create dataset_examples.pdf/png from TUM Freiburg1 XYZ.

The figure follows the same association and pair construction as the benchmark:
nearest ground-truth timestamp, max pose difference 0.02 s, frame step 10,
and pair stride 20. By default it shows the first, middle, and last pair.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


def parse_rgb(path: Path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        p = line.split()
        if len(p) >= 2:
            rows.append((float(p[0]), p[1]))
    return sorted(rows)


def parse_gt_times(path: Path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        p = line.split()
        if len(p) >= 8:
            rows.append(float(p[0]))
    return np.asarray(sorted(rows), dtype=float)


def nearest_dt(ts: float, gt: np.ndarray) -> float:
    i = int(np.searchsorted(gt, ts))
    candidates = []
    if i < len(gt):
        candidates.append(gt[i])
    if i > 0:
        candidates.append(gt[i - 1])
    return min(abs(ts - x) for x in candidates)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Path to rgbd_dataset_freiburg1_xyz")
    ap.add_argument("--output", default="dataset_examples.pdf")
    ap.add_argument("--frame-step", type=int, default=10)
    ap.add_argument("--pair-stride", type=int, default=20)
    ap.add_argument("--max-pose-dt", type=float, default=0.02)
    ap.add_argument("--pair-indices", default="", help="Optional comma-separated pair indices")
    args = ap.parse_args()

    dataset = Path(args.dataset).expanduser().resolve()
    rgb_file = dataset / "rgb.txt"
    gt_file = dataset / "groundtruth.txt"
    if not rgb_file.exists() or not gt_file.exists():
        raise FileNotFoundError("Dataset must contain rgb.txt and groundtruth.txt")

    gt = parse_gt_times(gt_file)
    records = []
    for ts, rel in parse_rgb(rgb_file):
        image_path = dataset / rel
        if image_path.exists() and nearest_dt(ts, gt) <= args.max_pose_dt:
            records.append((ts, image_path))

    pairs = [
        (i, records[i], records[i + args.frame_step])
        for i in range(0, len(records) - args.frame_step, args.pair_stride)
    ]
    if not pairs:
        raise RuntimeError("No valid pairs were produced")

    if args.pair_indices.strip():
        selected = [int(x) for x in args.pair_indices.split(",")]
    else:
        selected = sorted(set([0, len(pairs) // 2, len(pairs) - 1]))
    selected = selected[0:2]

    fig, axes = plt.subplots(len(selected), 2, figsize=(7, 3.25 * len(selected)))
    if len(selected) == 1:
        axes = np.asarray([axes])


    for row, pair_idx in enumerate(selected):
        start_index, first, second = pairs[pair_idx]
        for col, (ts, path) in enumerate([first, second]):
            image = cv.imread(str(path), cv.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(path)
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            axes[row, col].imshow(image)
            axes[row, col].axis("off")
            frame_index = start_index if col == 0 else start_index + args.frame_step
            axes[row, col].set_title(
                f"Pair {pair_idx}: $I_{{{frame_index}}}$, timestamp {ts:.6f} s",
                fontsize=10,
            )
        dt = second[0] - first[0]
        axes[row, 0].text(
            0.01,
            -0.08,
            f"Frame separation = {args.frame_step}; $\\Delta t={dt:.3f}$ s",
            transform=axes[row, 0].transAxes,
            fontsize=9,
        )

    # fig.suptitle("Representative TUM RGB-D Freiburg1 XYZ evaluation pairs", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output = Path(args.output)
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    print(output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
