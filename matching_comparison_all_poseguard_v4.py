#!/usr/bin/env python3
"""
Unified two-view matching benchmark and two-image visual comparison.

The script compares all methods from the original project with EAGM-Fuse and optional modern matchers. It also supports direct comparison of any two input images.

The original methods include raw nearest-neighbour matching, one-way Lowe ratio, mutual Lowe ratio, ORB cross-check, AdaLAM with SIFT/ORB, and an ORB-SLAM-inspired adaptive homography/fundamental model selector.

EAGM-Fuse is an experimental
edge-aware hybrid pipeline designed for a strong speed/accuracy trade-off:

  edge-aware ORB + edge-corner fusion
  -> spatially balanced feature budget
  -> bidirectional adaptive ratio matching
  -> orientation/scale robust consistency
  -> local grid-motion consensus
  -> forward/backward pyramidal LK refinement
  -> USAC_MAGSAC essential-matrix estimation

EdgeFusion is a research prototype and should be presented as a new hybrid
combination, not as a proven state-of-the-art algorithm until it is validated
on several datasets.

Several efficiency-oriented proposed variants are included:

  edgefusion_simple
      standard ORB keypoints + cheap edge-proximity re-ranking
      -> spatial quota -> one-way Lowe ratio -> orientation/scale check
      -> USAC_MAGSAC

  edgefusion_adaptive
      ORB-Lowe + fast geometry first
      -> accept when geometric confidence is high
      -> otherwise reuse the same ORB features, activate structural
         edge-aware filtering and selective LK refinement, then re-estimate pose

  edgefusion_adaptive_guarded
      preserves the Adaptive fast/fallback cascade but adds three protections:
      -> soft early acceptance for borderline-but-consistently-strong fast poses
      -> a pre-LK structural pose check that can avoid unnecessary optical flow
      -> post-refinement model selection on the SAME ORB correspondences using
         shared epipolar support, spatial coverage, and Sampson residual
      -> refinement replaces the fast pose only when geometric quality improves

The adaptive thresholds are fixed benchmark parameters and should be tuned on
validation data, not on the final evaluation pairs.

  edgefusion_adaptive_poseguard
      PoseGuard-v2 keeps all earlier Adaptive variants fixed and replaces
      epipolar-only model protection with shared cheirality, coverage,
      Sampson residual, and parallax evidence.

  edgefusion_adaptive_poseguard_v3
      keeps PoseGuard-v2 unchanged and adds:
      -> a hard candidate observability gate using native cheirality + parallax
      -> a gain-based cheirality rescue that is not tied to one fast-pose cutoff
      -> conservative early termination before LK for extremely weak fallbacks
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import importlib.util
import importlib.metadata as importlib_metadata
import json
import math
import os
import platform
import statistics
import sys
import threading
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2 as cv
import numpy as np

try:
    from scipy.spatial.transform import Rotation
    print()
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("scipy is required: pip install scipy") from exc

try:
    import psutil
except ImportError:  # optional
    psutil = None

try:
    import pandas as pd
except ImportError:  # optional, CSV fallback is available
    pd = None

try:
    import matplotlib.pyplot as plt
except ImportError:  # optional
    plt = None


EPS = 1e-12
_XFEAT_CACHE: Dict[Tuple[str, str], Any] = {}
_LIGHTGLUE_CACHE: Dict[Tuple[str, int], Tuple[Any, Any]] = {}
_BUNDLED_ADALAM_CACHE: Optional[Tuple[Any, str]] = None


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CameraModel:
    fx: float = 517.3
    fy: float = 516.5
    cx: float = 318.6
    cy: float = 255.3
    # TUM Freiburg 1 RGB calibration.
    distortion: Tuple[float, float, float, float, float] = (
        0.2624,
        -0.9531,
        -0.0054,
        0.0026,
        1.1633,
    )

    @property
    def K(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    @property
    def dist(self) -> np.ndarray:
        return np.asarray(self.distortion, dtype=np.float64).reshape(-1, 1)


# Official TUM RGB-camera intrinsics for the three Freiburg camera groups.
# IMPORTANT: this benchmark loads color images from rgb.txt, so RGB calibration
# must be used here. The IR calibration values are for the infrared camera and
# must not be applied to these RGB images.
TUM_RGB_CAMERA_MODELS: Dict[str, CameraModel] = {
    "freiburg1": CameraModel(
        fx=517.3,
        fy=516.5,
        cx=318.6,
        cy=255.3,
        distortion=(0.2624, -0.9531, -0.0054, 0.0026, 1.1633),
    ),
    "freiburg2": CameraModel(
        fx=520.9,
        fy=521.0,
        cx=325.1,
        cy=249.7,
        distortion=(0.2312, -0.7849, -0.0033, -0.0001, 0.9172),
    ),
    # Freiburg 3 color images are already undistorted in the released dataset.
    "freiburg3": CameraModel(
        fx=535.4,
        fy=539.2,
        cx=320.1,
        cy=247.6,
        distortion=(0.0, 0.0, 0.0, 0.0, 0.0),
    ),
}


def infer_tum_camera_family(dataset_dir: Path) -> str:
    """Infer Freiburg camera family from a standard TUM dataset path/name.

    Examples accepted automatically:
      rgbd_dataset_freiburg1_xyz
      rgbd_dataset_freiburg2_desk
      rgbd_dataset_freiburg3_structure_notexture_near

    If a dataset has been renamed so the Freiburg family is not present in the
    path, use --tum-camera explicitly.
    """
    value = str(Path(dataset_dir).expanduser()).lower().replace("-", "_")
    for family in TUM_RGB_CAMERA_MODELS:
        if family in value:
            return family
    raise ValueError(
        "Could not infer the TUM camera family from dataset path "
        f"'{dataset_dir}'. Keep 'freiburg1', 'freiburg2', or 'freiburg3' "
        "in the folder/path name, or pass --tum-camera explicitly."
    )


def select_tum_rgb_camera(
    dataset_dir: Path,
    requested: str = "auto",
) -> Tuple[str, CameraModel]:
    """Return the official RGB calibration for the selected TUM camera group."""
    value = str(requested or "auto").strip().lower()
    aliases = {
        "fr1": "freiburg1",
        "tum_fr1": "freiburg1",
        "fr2": "freiburg2",
        "tum_fr2": "freiburg2",
        "fr3": "freiburg3",
        "tum_fr3": "freiburg3",
    }
    value = aliases.get(value, value)
    family = infer_tum_camera_family(dataset_dir) if value == "auto" else value
    if family not in TUM_RGB_CAMERA_MODELS:
        raise ValueError(
            f"Unknown TUM camera selection '{requested}'. "
            "Use auto, freiburg1, freiburg2, or freiburg3."
        )
    return family, TUM_RGB_CAMERA_MODELS[family]


@dataclass(frozen=True)
class Pose:
    timestamp: float
    t_w_c: np.ndarray
    R_w_c: np.ndarray


@dataclass(frozen=True)
class ImageRecord:
    timestamp: float
    path: Path
    pose: Pose


@dataclass(frozen=True)
class PairRecord:
    pair_index: int
    first: ImageRecord
    second: ImageRecord


@dataclass
class FeatureSet:
    keypoints: List[cv.KeyPoint]
    descriptors: Optional[np.ndarray]
    points: np.ndarray
    edge_confidence: np.ndarray
    gradient_angle: np.ndarray
    gradient_magnitude: np.ndarray


@dataclass
class PipelineConfig:
    nfeatures: int = 1600
    oversample_factor: float = 1.5
    max_side: int = 960
    clahe_clip: float = 2.0
    canny_sigma: float = 0.33
    edge_dilate: int = 3
    edge_sigma_px: float = 4.0
    grid_rows: int = 6
    grid_cols: int = 8
    ratio: float = 0.80
    adaptive_ratio_min: float = 0.74
    adaptive_ratio_max: float = 0.90
    local_motion_min_px: float = 2.5
    local_motion_mad_scale: float = 3.0
    orientation_min_deg: float = 18.0
    orientation_mad_scale: float = 3.0
    scale_min_log: float = 0.18
    scale_mad_scale: float = 3.0
    lk_window: int = 21
    lk_levels: int = 3
    lk_fb_threshold: float = 1.5
    lk_max_correction: float = 5.0
    magsac_threshold_px: float = 1.25
    confidence: float = 0.999
    max_iters: int = 10000
    min_pose_matches: int = 8
    min_filter_matches: int = 12
    use_clahe: bool = True
    use_edge_corners: bool = True
    use_edge_weighting: bool = True
    use_edge_orientation_filter: bool = True
    use_orientation_scale_filter: bool = True
    use_grid_motion_filter: bool = True
    use_lk_refinement: bool = True
    use_distortion_correction: bool = True

    # Efficiency-oriented proposed variants.
    # edgefusion_simple keeps only cheap structural operations.
    simple_feature_fraction: float = 0.65
    simple_edge_weight: float = 0.35

    # edgefusion_adaptive first attempts ORB-Lowe and escalates only when the
    # fast pose does not satisfy all three confidence conditions.
    adaptive_accept_inliers: int = 80
    adaptive_accept_inlier_ratio: float = 0.65
    adaptive_accept_coverage: float = 0.25
    adaptive_fallback_feature_fraction: float = 0.75
    adaptive_lk_max_matches: int = 240

    # Guarded Adaptive variant. These controls are intentionally expressed as
    # validation-time hyperparameters rather than being fitted to test-pair
    # ground truth.
    adaptive_guarded_soft_score: float = 0.97
    adaptive_guarded_soft_floor_fraction: float = 0.90
    adaptive_guarded_quality_margin: float = 0.02
    adaptive_guarded_prelk_margin: float = 0.04

    # PoseGuard-v2 uses pose observability rather than epipolar residual alone.
    adaptive_poseguard_quality_margin: float = 0.02
    adaptive_poseguard_prelk_margin: float = 0.035
    adaptive_poseguard_parallax_target_deg: float = 1.0
    adaptive_poseguard_rescue_fast_cheirality_max: float = 0.35
    adaptive_poseguard_rescue_candidate_cheirality_min: float = 0.60
    adaptive_poseguard_rescue_cheirality_gain: float = 0.25
    adaptive_poseguard_cheirality_drop_tolerance: float = 0.05

    # PoseGuard-v3 keeps v2 unchanged and adds a candidate-observability gate,
    # a gain-based rescue rule, and an optional early stop before LK. These
    # defaults are development hyperparameters and must be frozen on validation
    # data before the final benchmark.
    adaptive_poseguard_v3_quality_margin: float = 0.02
    adaptive_poseguard_v3_prelk_margin: float = 0.035
    adaptive_poseguard_v3_min_candidate_native_cheirality: float = 0.50
    adaptive_poseguard_v3_min_candidate_parallax_deg: float = 0.50
    adaptive_poseguard_v3_rescue_candidate_native_cheirality_min: float = 0.80
    adaptive_poseguard_v3_rescue_native_cheirality_gain: float = 0.25
    adaptive_poseguard_v3_rescue_candidate_shared_cheirality_min: float = 0.60
    adaptive_poseguard_v3_rescue_shared_cheirality_gain: float = 0.20
    adaptive_poseguard_v3_shared_cheirality_drop_tolerance: float = 0.05
    adaptive_poseguard_v3_native_cheirality_drop_tolerance: float = 0.10
    adaptive_poseguard_v3_early_abort_native_cheirality_max: float = 0.15
    adaptive_poseguard_v3_early_abort_parallax_max_deg: float = 0.25

    # PoseGuard-v4 keeps v3 unchanged and adds a stability gate specifically
    # for LK replacing an already-selected pre-LK structural pose. Cheirality
    # rescue remains available for fast->structural replacement, but not for
    # pre-LK->LK replacement.
    adaptive_poseguard_v4_lk_quality_margin: float = 0.015
    adaptive_poseguard_v4_lk_max_rotation_disagreement_deg: float = 3.0
    adaptive_poseguard_v4_lk_max_translation_disagreement_deg: float = 15.0
    adaptive_poseguard_v4_lk_parallax_ratio_min: float = 0.50
    adaptive_poseguard_v4_lk_parallax_ratio_max: float = 2.00
    adaptive_poseguard_v4_lk_shared_cheirality_drop_tolerance: float = 0.03
    adaptive_poseguard_v4_lk_native_cheirality_drop_tolerance: float = 0.05


@dataclass
class MethodResult:
    method: str
    success: bool
    status: str
    points1: np.ndarray = field(default_factory=lambda: np.empty((0, 2), np.float32))
    points2: np.ndarray = field(default_factory=lambda: np.empty((0, 2), np.float32))
    R: Optional[np.ndarray] = None
    t: Optional[np.ndarray] = None
    geometric_inlier_mask: np.ndarray = field(default_factory=lambda: np.empty(0, bool))
    num_keypoints1: int = 0
    num_keypoints2: int = 0
    num_raw_matches: int = 0
    num_filtered_matches: int = 0
    num_geometric_inliers: int = 0
    timings_ms: Dict[str, float] = field(default_factory=dict)
    peak_rss_delta_mb: float = float("nan")
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    @property
    def inlier_ratio(self) -> float:
        if self.num_filtered_matches <= 0:
            return 0.0
        return self.num_geometric_inliers / self.num_filtered_matches

    @property
    def total_ms(self) -> float:
        return float(sum(self.timings_ms.values()))


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


class StageTimer:
    def __init__(self, timings: Dict[str, float], name: str):
        self.timings = timings
        self.name = name
        self.start_ns = 0

    def __enter__(self) -> "StageTimer":
        self.start_ns = time.perf_counter_ns()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        elapsed_ms = (time.perf_counter_ns() - self.start_ns) / 1e6
        self.timings[self.name] = self.timings.get(self.name, 0.0) + elapsed_ms


class PeakRSSMonitor:
    """Sample process RSS while a method runs.

    This includes native OpenCV allocations, unlike tracemalloc. The sampling
    thread adds a small overhead, so disable it for pure latency measurements.
    """

    def __init__(self, enabled: bool, period_s: float = 0.003):
        self.enabled = bool(enabled and psutil is not None)
        self.period_s = period_s
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._baseline = 0
        self._peak = 0

    def __enter__(self) -> "PeakRSSMonitor":
        if not self.enabled:
            return self
        proc = psutil.Process(os.getpid())
        self._baseline = proc.memory_info().rss
        self._peak = self._baseline

        def worker() -> None:
            while not self._stop.is_set():
                try:
                    self._peak = max(self._peak, proc.memory_info().rss)
                except Exception:
                    return
                self._stop.wait(self.period_s)

        self._thread = threading.Thread(target=worker, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if not self.enabled:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=0.2)
        try:
            rss = psutil.Process(os.getpid()).memory_info().rss
            self._peak = max(self._peak, rss)
        except Exception:
            pass

    @property
    def delta_mb(self) -> float:
        if not self.enabled:
            return float("nan")
        return max(0, self._peak - self._baseline) / (1024.0 * 1024.0)


def ensure_gray(image: np.ndarray) -> np.ndarray:
    if image is None:
        raise ValueError("Image is None")
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 3:
        return cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv.cvtColor(image, cv.COLOR_BGRA2GRAY)
    raise ValueError(f"Unsupported image shape: {image.shape}")


def resize_to_max_side(image: np.ndarray, max_side: int) -> Tuple[np.ndarray, float]:
    if max_side <= 0:
        return image, 1.0
    h, w = image.shape[:2]
    scale = min(1.0, float(max_side) / max(h, w))
    if scale >= 0.999999:
        return image, 1.0
    resized = cv.resize(image, None, fx=scale, fy=scale, interpolation=cv.INTER_AREA)
    return resized, scale


def scale_camera(camera: CameraModel, scale: float) -> CameraModel:
    return CameraModel(
        fx=camera.fx * scale,
        fy=camera.fy * scale,
        cx=camera.cx * scale,
        cy=camera.cy * scale,
        distortion=camera.distortion,
    )


def robust_mad(values: np.ndarray, center: Optional[float] = None) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return 0.0
    c = float(np.median(values)) if center is None else float(center)
    return 1.4826 * float(np.median(np.abs(values - c)))


def wrap_angle_rad(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def circular_median(angles: np.ndarray) -> float:
    angles = np.asarray(angles, dtype=np.float64)
    if angles.size == 0:
        return 0.0
    # The circular mean is used as a stable center; the robust gate uses MAD.
    return float(math.atan2(np.mean(np.sin(angles)), np.mean(np.cos(angles))))


def keypoints_to_points(kps: Sequence[cv.KeyPoint]) -> np.ndarray:
    if not kps:
        return np.empty((0, 2), dtype=np.float32)
    return np.asarray([kp.pt for kp in kps], dtype=np.float32)


def points_to_keypoints(points: np.ndarray, size: float = 5.0) -> List[cv.KeyPoint]:
    return [cv.KeyPoint(float(x), float(y), float(size)) for x, y in np.asarray(points)]


def safe_normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    n = np.linalg.norm(v)
    if n < EPS:
        return v * 0.0
    return v / n


def rotation_error_deg(R_est: np.ndarray, R_gt: np.ndarray) -> float:
    R_delta = np.asarray(R_est) @ np.asarray(R_gt).T
    cos_theta = np.clip((np.trace(R_delta) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def vector_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    a = safe_normalize(a)
    b = safe_normalize(b)
    if np.linalg.norm(a) < EPS or np.linalg.norm(b) < EPS:
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))))


def relative_ground_truth(first: Pose, second: Pose) -> Tuple[np.ndarray, np.ndarray]:
    """Return transform from camera 1 coordinates to camera 2 coordinates.

    TUM provides camera-to-world poses T_w_c. Therefore:
      R_21 = R_w_c2^T R_w_c1
      t_21 = R_w_c2^T (t_w_c1 - t_w_c2)
    """

    R_21 = second.R_w_c.T @ first.R_w_c
    t_21 = second.R_w_c.T @ (first.t_w_c - second.t_w_c)
    return R_21, safe_normalize(t_21)


def write_csv_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                keys.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# TUM dataset loading
# ---------------------------------------------------------------------------


def parse_timestamp_file(path: Path) -> List[Tuple[float, str]]:
    records: List[Tuple[float, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            records.append((float(parts[0]), parts[1]))
    records.sort(key=lambda x: x[0])
    return records


def parse_groundtruth(path: Path) -> List[Pose]:
    poses: List[Pose] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            ts = float(parts[0])
            t = np.asarray([float(v) for v in parts[1:4]], dtype=np.float64)
            q_xyzw = np.asarray([float(v) for v in parts[4:8]], dtype=np.float64)
            R_w_c = Rotation.from_quat(q_xyzw).as_matrix()
            poses.append(Pose(ts, t, R_w_c))
    poses.sort(key=lambda p: p.timestamp)
    return poses


def nearest_pose(timestamp: float, poses: Sequence[Pose], pose_times: np.ndarray) -> Tuple[Pose, float]:
    idx = int(np.searchsorted(pose_times, timestamp))
    candidates = []
    if idx < len(poses):
        candidates.append(poses[idx])
    if idx > 0:
        candidates.append(poses[idx - 1])
    if not candidates:
        raise ValueError("No ground-truth poses are available")
    best = min(candidates, key=lambda p: abs(p.timestamp - timestamp))
    return best, abs(best.timestamp - timestamp)


def load_tum_records(dataset_dir: Path, max_pose_dt: float = 0.02) -> List[ImageRecord]:
    rgb_file = dataset_dir / "rgb.txt"
    gt_file = dataset_dir / "groundtruth.txt"
    if not rgb_file.exists():
        raise FileNotFoundError(f"Missing {rgb_file}")
    if not gt_file.exists():
        raise FileNotFoundError(f"Missing {gt_file}")

    rgb_records = parse_timestamp_file(rgb_file)
    poses = parse_groundtruth(gt_file)
    pose_times = np.asarray([p.timestamp for p in poses], dtype=np.float64)

    output: List[ImageRecord] = []
    skipped = 0
    for ts, rel_path in rgb_records:
        pose, dt = nearest_pose(ts, poses, pose_times)
        if dt > max_pose_dt:
            skipped += 1
            continue
        image_path = dataset_dir / rel_path
        if not image_path.exists():
            # Compatibility with the old project's dot-to-underscore renaming.
            p = Path(rel_path)
            stem_alt = p.stem.replace(".", "_")
            alt = dataset_dir / p.parent / f"{stem_alt}{p.suffix}"
            if alt.exists():
                image_path = alt
            else:
                skipped += 1
                continue
        output.append(ImageRecord(ts, image_path, pose))

    if len(output) < 2:
        raise RuntimeError(
            f"Only {len(output)} associated RGB frames were found. "
            "Do not rename the downloaded TUM files; keep rgb.txt paths intact."
        )
    if skipped:
        print(f"[dataset] skipped {skipped} RGB records without a close pose or image file")
    return output


def build_pairs(
    records: Sequence[ImageRecord],
    frame_step: int,
    pair_stride: int,
    max_pairs: int,
) -> List[PairRecord]:
    if frame_step < 1:
        raise ValueError("frame_step must be >= 1")
    if pair_stride < 1:
        raise ValueError("pair_stride must be >= 1")
    candidates = [
        PairRecord(i, records[i], records[i + frame_step])
        for i in range(0, len(records) - frame_step, pair_stride)
    ]
    if max_pairs > 0:
        candidates = candidates[:max_pairs]
    return [PairRecord(i, p.first, p.second) for i, p in enumerate(candidates)]


# ---------------------------------------------------------------------------
# Image preprocessing and feature extraction
# ---------------------------------------------------------------------------


def preprocess_for_features(gray: np.ndarray, cfg: PipelineConfig) -> np.ndarray:
    if not cfg.use_clahe:
        return gray
    clahe = cv.createCLAHE(clipLimit=cfg.clahe_clip, tileGridSize=(8, 8))
    return clahe.apply(gray)


def edge_fields(gray: np.ndarray, cfg: PipelineConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    med = float(np.median(gray))
    lower = int(max(0, (1.0 - cfg.canny_sigma) * med))
    upper = int(min(255, (1.0 + cfg.canny_sigma) * med))
    if upper <= lower:
        lower, upper = 40, 120
    edges = cv.Canny(gray, lower, upper, L2gradient=True)
    if cfg.edge_dilate > 0:
        k = 2 * cfg.edge_dilate + 1
        edge_mask = cv.dilate(edges, np.ones((k, k), np.uint8), iterations=1)
    else:
        edge_mask = edges

    gx = cv.Scharr(gray, cv.CV_32F, 1, 0)
    gy = cv.Scharr(gray, cv.CV_32F, 0, 1)
    mag = cv.magnitude(gx, gy)
    mag /= float(np.percentile(mag, 99.0) + EPS)
    mag = np.clip(mag, 0.0, 1.0)
    angle = cv.phase(gx, gy, angleInDegrees=False)

    non_edge = (edges == 0).astype(np.uint8)
    distance_to_edge = cv.distanceTransform(non_edge, cv.DIST_L2, 3)
    edge_proximity = np.exp(-distance_to_edge / max(cfg.edge_sigma_px, EPS)).astype(np.float32)
    return edges, edge_mask, mag.astype(np.float32), angle.astype(np.float32), edge_proximity


def deduplicate_keypoints(kps: Sequence[cv.KeyPoint], cell_px: float = 2.5) -> List[cv.KeyPoint]:
    best: Dict[Tuple[int, int], cv.KeyPoint] = {}
    for kp in kps:
        key = (int(round(kp.pt[0] / cell_px)), int(round(kp.pt[1] / cell_px)))
        old = best.get(key)
        if old is None or kp.response > old.response:
            best[key] = kp
    return list(best.values())


def grid_balanced_indices(
    points: np.ndarray,
    scores: np.ndarray,
    image_shape: Tuple[int, int],
    max_features: int,
    rows: int,
    cols: int,
) -> np.ndarray:
    n = len(points)
    if n <= max_features:
        return np.arange(n, dtype=np.int32)
    h, w = image_shape
    quota = max(1, int(math.ceil(max_features / float(rows * cols))))
    buckets: Dict[Tuple[int, int], List[int]] = {}
    for idx, (x, y) in enumerate(points):
        c = min(cols - 1, max(0, int(x * cols / max(w, 1))))
        r = min(rows - 1, max(0, int(y * rows / max(h, 1))))
        buckets.setdefault((r, c), []).append(idx)

    chosen: List[int] = []
    chosen_set = set()
    for idxs in buckets.values():
        idxs_sorted = sorted(idxs, key=lambda i: float(scores[i]), reverse=True)
        for idx in idxs_sorted[:quota]:
            chosen.append(idx)
            chosen_set.add(idx)

    if len(chosen) < max_features:
        remaining = [i for i in np.argsort(-scores) if int(i) not in chosen_set]
        chosen.extend(int(i) for i in remaining[: max_features - len(chosen)])
    elif len(chosen) > max_features:
        chosen = sorted(chosen, key=lambda i: float(scores[i]), reverse=True)[:max_features]

    return np.asarray(chosen, dtype=np.int32)


def sample_field(field: np.ndarray, points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.empty(0, dtype=np.float32)
    h, w = field.shape[:2]
    x = np.clip(np.rint(points[:, 0]).astype(np.int32), 0, w - 1)
    y = np.clip(np.rint(points[:, 1]).astype(np.int32), 0, h - 1)
    return np.asarray(field[y, x])


def extract_edge_orb(gray: np.ndarray, cfg: PipelineConfig) -> FeatureSet:
    processed = preprocess_for_features(gray, cfg)
    _, edge_mask, grad_mag, grad_angle, edge_proximity = edge_fields(processed, cfg)

    oversampled = max(cfg.nfeatures, int(round(cfg.nfeatures * cfg.oversample_factor)))
    orb = cv.ORB_create(
        nfeatures=oversampled,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=19,
        firstLevel=0,
        WTA_K=2,
        scoreType=cv.ORB_HARRIS_SCORE,
        patchSize=31,
        fastThreshold=10,
    )
    kps = list(orb.detect(processed, None))

    if cfg.use_edge_corners:
        corners = cv.goodFeaturesToTrack(
            processed,
            maxCorners=max(200, oversampled // 2),
            qualityLevel=0.006,
            minDistance=4.0,
            mask=edge_mask,
            blockSize=5,
            useHarrisDetector=False,
        )
        if corners is not None:
            for x, y in corners.reshape(-1, 2):
                response = float(grad_mag[int(round(y)), int(round(x))])
                kps.append(cv.KeyPoint(float(x), float(y), 31.0, -1.0, response))

    kps = deduplicate_keypoints(kps)
    if not kps:
        return FeatureSet([], None, np.empty((0, 2), np.float32), *(np.empty(0, np.float32) for _ in range(3)))

    # Rank candidates before descriptor computation. This is significantly
    # cheaper than computing ORB descriptors for every oversampled candidate.
    candidate_pts = keypoints_to_points(kps)
    candidate_edge = sample_field(edge_proximity, candidate_pts).astype(np.float32)
    candidate_grad = sample_field(grad_mag, candidate_pts).astype(np.float32)
    responses = np.asarray([max(0.0, kp.response) for kp in kps], dtype=np.float32)
    if responses.size:
        responses /= float(np.percentile(responses, 95.0) + EPS)
    responses = np.clip(responses, 0.0, 1.0)
    if cfg.use_edge_weighting:
        candidate_score = 0.42 * responses + 0.38 * candidate_edge + 0.20 * candidate_grad
    else:
        candidate_score = responses

    precompute_budget = min(len(kps), max(cfg.nfeatures, int(round(cfg.nfeatures * 1.20))))
    candidate_keep = grid_balanced_indices(
        candidate_pts,
        candidate_score,
        processed.shape[:2],
        precompute_budget,
        cfg.grid_rows,
        cfg.grid_cols,
    )
    selected_kps = [kps[int(i)] for i in candidate_keep]
    selected_kps, desc = orb.compute(processed, selected_kps)
    selected_kps = list(selected_kps or [])
    if desc is None or not selected_kps:
        return FeatureSet([], None, np.empty((0, 2), np.float32), *(np.empty(0, np.float32) for _ in range(3)))

    pts = keypoints_to_points(selected_kps)
    edge_conf = sample_field(edge_proximity, pts).astype(np.float32)
    gmag = sample_field(grad_mag, pts).astype(np.float32)
    gang = sample_field(grad_angle, pts).astype(np.float32)
    responses = np.asarray([max(0.0, kp.response) for kp in selected_kps], dtype=np.float32)
    if responses.size:
        responses /= float(np.percentile(responses, 95.0) + EPS)
    responses = np.clip(responses, 0.0, 1.0)
    if cfg.use_edge_weighting:
        score = 0.42 * responses + 0.38 * edge_conf + 0.20 * gmag
    else:
        score = responses

    keep = grid_balanced_indices(
        pts,
        score,
        processed.shape[:2],
        cfg.nfeatures,
        cfg.grid_rows,
        cfg.grid_cols,
    )
    return FeatureSet(
        keypoints=[selected_kps[int(i)] for i in keep],
        descriptors=np.ascontiguousarray(desc[keep]),
        points=np.ascontiguousarray(pts[keep]),
        edge_confidence=np.ascontiguousarray(edge_conf[keep]),
        gradient_angle=np.ascontiguousarray(gang[keep]),
        gradient_magnitude=np.ascontiguousarray(gmag[keep]),
    )

def extract_standard(gray: np.ndarray, detector_name: str, nfeatures: int) -> FeatureSet:
    name = detector_name.lower()
    if name == "orb":
        detector = cv.ORB_create(nfeatures=nfeatures, fastThreshold=12)
    elif name == "sift":
        detector = cv.SIFT_create(nfeatures=nfeatures, contrastThreshold=0.03)
    elif name == "akaze":
        detector = cv.AKAZE_create(
            descriptor_type=cv.AKAZE_DESCRIPTOR_MLDB,
            threshold=0.001,
            nOctaves=4,
            nOctaveLayers=4,
        )
    else:
        raise ValueError(f"Unknown detector: {detector_name}")

    kps, desc = detector.detectAndCompute(gray, None)
    kps = list(kps or [])
    pts = keypoints_to_points(kps)
    empty = np.zeros(len(kps), dtype=np.float32)
    return FeatureSet(kps, desc, pts, empty.copy(), empty.copy(), empty.copy())




def _simple_edge_proximity(gray: np.ndarray, cfg: PipelineConfig) -> np.ndarray:
    """Compute only the cheap edge-proximity field used by EdgeFusion-Simple.

    Unlike the full EdgeFusion extractor, this helper intentionally omits
    CLAHE, Shi--Tomasi corner generation, Scharr magnitude/orientation, and
    edge-orientation filtering. It uses only Canny + a distance transform.
    """
    med = float(np.median(gray))
    lower = int(max(0, (1.0 - cfg.canny_sigma) * med))
    upper = int(min(255, (1.0 + cfg.canny_sigma) * med))
    if upper <= lower:
        lower, upper = 40, 120
    edges = cv.Canny(gray, lower, upper, L2gradient=True)
    non_edge = (edges == 0).astype(np.uint8)
    distance_to_edge = cv.distanceTransform(non_edge, cv.DIST_L2, 3)
    return np.exp(
        -distance_to_edge / max(float(cfg.edge_sigma_px), EPS)
    ).astype(np.float32)


def _subset_feature_set(features: FeatureSet, indices: np.ndarray) -> FeatureSet:
    """Return a descriptor-aligned subset of a FeatureSet."""
    idx = np.asarray(indices, dtype=np.int32).reshape(-1)
    descriptors = None
    if features.descriptors is not None:
        descriptors = np.ascontiguousarray(features.descriptors[idx])
    return FeatureSet(
        keypoints=[features.keypoints[int(i)] for i in idx],
        descriptors=descriptors,
        points=np.ascontiguousarray(features.points[idx], dtype=np.float32),
        edge_confidence=np.ascontiguousarray(
            features.edge_confidence[idx], dtype=np.float32
        ),
        gradient_angle=np.ascontiguousarray(
            features.gradient_angle[idx], dtype=np.float32
        ),
        gradient_magnitude=np.ascontiguousarray(
            features.gradient_magnitude[idx], dtype=np.float32
        ),
    )


def extract_simple_edge_orb(gray: np.ndarray, cfg: PipelineConfig) -> FeatureSet:
    """Lightweight structurally guided ORB extractor.

    The goal is not to reproduce the full EdgeFusion front end. Instead, the
    method keeps ORB as the only keypoint/descriptor operator and adds two cheap
    operations: Canny edge proximity and a spatial quota.

    Candidates are detected first, ranked by
        (1-w) * normalized_ORB_response + w * edge_proximity,
    and descriptors are computed only for the retained candidates.
    """
    target = max(
        cfg.min_filter_matches,
        int(round(cfg.nfeatures * float(cfg.simple_feature_fraction))),
    )
    target = min(cfg.nfeatures, target)
    detect_budget = max(target, int(round(target * 1.20)))

    orb = cv.ORB_create(
        nfeatures=detect_budget,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=19,
        firstLevel=0,
        WTA_K=2,
        scoreType=cv.ORB_HARRIS_SCORE,
        patchSize=31,
        fastThreshold=12,
    )
    kps = list(orb.detect(gray, None))
    if not kps:
        empty = np.empty(0, np.float32)
        return FeatureSet(
            [], None, np.empty((0, 2), np.float32),
            empty.copy(), empty.copy(), empty.copy()
        )

    pts = keypoints_to_points(kps)
    edge_prox_field = _simple_edge_proximity(gray, cfg)
    edge_conf = sample_field(edge_prox_field, pts).astype(np.float32)

    responses = np.asarray(
        [max(0.0, kp.response) for kp in kps], dtype=np.float32
    )
    if responses.size:
        responses /= float(np.percentile(responses, 95.0) + EPS)
    responses = np.clip(responses, 0.0, 1.0)

    edge_weight = float(np.clip(cfg.simple_edge_weight, 0.0, 1.0))
    scores = (1.0 - edge_weight) * responses + edge_weight * edge_conf
    keep = grid_balanced_indices(
        pts,
        scores,
        gray.shape[:2],
        target,
        cfg.grid_rows,
        cfg.grid_cols,
    )
    selected = [kps[int(i)] for i in keep]
    selected, desc = orb.compute(gray, selected)
    selected = list(selected or [])
    if desc is None or not selected:
        empty = np.empty(0, np.float32)
        return FeatureSet(
            [], None, np.empty((0, 2), np.float32),
            empty.copy(), empty.copy(), empty.copy()
        )

    pts = keypoints_to_points(selected)
    edge_conf = sample_field(edge_prox_field, pts).astype(np.float32)
    zeros = np.zeros(len(selected), dtype=np.float32)
    return FeatureSet(
        keypoints=selected,
        descriptors=np.ascontiguousarray(desc),
        points=np.ascontiguousarray(pts),
        edge_confidence=np.ascontiguousarray(edge_conf),
        gradient_angle=zeros.copy(),
        gradient_magnitude=zeros.copy(),
    )


def _attach_structural_fields(
    gray: np.ndarray,
    features: FeatureSet,
    cfg: PipelineConfig,
) -> FeatureSet:
    """Attach edge/gradient evidence to already-computed ORB features.

    This is the key efficiency feature of EdgeFusion-Adaptive: if the initial
    ORB-Lowe pose is weak, the fallback reuses the ORB keypoints/descriptors
    instead of extracting them a second time.
    """
    if len(features.points) == 0:
        return features
    _, _, grad_mag, grad_angle, edge_proximity = edge_fields(gray, cfg)
    return FeatureSet(
        keypoints=list(features.keypoints),
        descriptors=features.descriptors,
        points=np.ascontiguousarray(features.points, dtype=np.float32),
        edge_confidence=np.ascontiguousarray(
            sample_field(edge_proximity, features.points), dtype=np.float32
        ),
        gradient_angle=np.ascontiguousarray(
            sample_field(grad_angle, features.points), dtype=np.float32
        ),
        gradient_magnitude=np.ascontiguousarray(
            sample_field(grad_mag, features.points), dtype=np.float32
        ),
    )


def _edge_rerank_existing_features(
    features: FeatureSet,
    image_shape: Tuple[int, int],
    max_features: int,
    cfg: PipelineConfig,
) -> FeatureSet:
    """Re-rank existing ORB features using the full EdgeFusion soft score."""
    if len(features.points) <= max_features:
        return features

    responses = np.asarray(
        [max(0.0, kp.response) for kp in features.keypoints],
        dtype=np.float32,
    )
    if responses.size:
        responses /= float(np.percentile(responses, 95.0) + EPS)
    responses = np.clip(responses, 0.0, 1.0)

    score = (
        0.42 * responses
        + 0.38 * features.edge_confidence
        + 0.20 * features.gradient_magnitude
    )
    keep = grid_balanced_indices(
        features.points,
        score,
        image_shape,
        max_features,
        cfg.grid_rows,
        cfg.grid_cols,
    )
    return _subset_feature_set(features, keep)


def _spatial_coverage_ratio(
    points: np.ndarray,
    inlier_mask: np.ndarray,
    image_shape: Tuple[int, int],
    rows: int = 4,
    cols: int = 6,
) -> float:
    """Fraction of coarse image cells containing at least one inlier."""
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    mask = np.asarray(inlier_mask, dtype=bool).reshape(-1)
    if len(points) == 0 or len(mask) != len(points) or not np.any(mask):
        return 0.0
    selected = points[mask]
    h, w = image_shape
    rr = np.clip(
        (selected[:, 1] * rows / max(h, 1)).astype(np.int32),
        0, rows - 1,
    )
    cc = np.clip(
        (selected[:, 0] * cols / max(w, 1)).astype(np.int32),
        0, cols - 1,
    )
    occupied = len(set(zip(rr.tolist(), cc.tolist())))
    return float(occupied / max(rows * cols, 1))


def _adaptive_pose_confidence(
    success: bool,
    points1: np.ndarray,
    points2: np.ndarray,
    inlier_mask: np.ndarray,
    image_shape: Tuple[int, int],
    cfg: PipelineConfig,
) -> Tuple[bool, Dict[str, float]]:
    """Evaluate whether the fast ORB-Lowe pose is strong enough to accept.

    The rule is intentionally simple and interpretable. All three conditions
    must pass:
      1. enough cheirality-consistent inliers,
      2. sufficient inlier ratio,
      3. sufficient spatial coverage.

    These thresholds are benchmark hyperparameters and should be fixed on a
    validation split before final testing.
    """
    mask = np.asarray(inlier_mask, dtype=bool).reshape(-1)
    n_matches = int(len(points1))
    n_inliers = int(np.count_nonzero(mask)) if len(mask) == n_matches else 0
    inlier_ratio = n_inliers / max(n_matches, 1)
    coverage = _spatial_coverage_ratio(
        points1, mask, image_shape, rows=4, cols=6
    )

    median_flow = float("nan")
    if len(mask) == n_matches and np.any(mask):
        flow = np.linalg.norm(
            np.asarray(points2)[mask] - np.asarray(points1)[mask], axis=1
        )
        if len(flow):
            median_flow = float(np.median(flow))

    accepted = bool(
        success
        and n_inliers >= int(cfg.adaptive_accept_inliers)
        and inlier_ratio >= float(cfg.adaptive_accept_inlier_ratio)
        and coverage >= float(cfg.adaptive_accept_coverage)
    )

    # A bounded score is stored for diagnostics only; acceptance uses the
    # explicit conditions above so the decision remains interpretable.
    c_inliers = min(
        1.0,
        n_inliers / max(float(cfg.adaptive_accept_inliers), 1.0),
    )
    c_ratio = min(
        1.0,
        inlier_ratio / max(float(cfg.adaptive_accept_inlier_ratio), EPS),
    )
    c_coverage = min(
        1.0,
        coverage / max(float(cfg.adaptive_accept_coverage), EPS),
    )
    score = 0.35 * c_inliers + 0.35 * c_ratio + 0.30 * c_coverage

    return accepted, {
        "adaptive_fast_inliers": float(n_inliers),
        "adaptive_fast_inlier_ratio": float(inlier_ratio),
        "adaptive_fast_coverage": float(coverage),
        "adaptive_fast_median_flow_px": median_flow,
        "adaptive_fast_confidence_score": float(score),
        "adaptive_fast_accepted": float(accepted),
    }



def _skew_symmetric(v: np.ndarray) -> np.ndarray:
    """Return the 3x3 skew-symmetric matrix [v]_x."""
    x, y, z = np.asarray(v, dtype=np.float64).reshape(3)
    return np.array(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
        dtype=np.float64,
    )


def _pose_shared_quality(
    R_est: Optional[np.ndarray],
    t_est: Optional[np.ndarray],
    points1: np.ndarray,
    points2: np.ndarray,
    camera: CameraModel,
    cfg: PipelineConfig,
    image_shape: Tuple[int, int],
) -> Dict[str, float]:
    """Score one pose on a shared correspondence set.

    This function is used only by EdgeFusion-Adaptive-Guarded. Competing fast,
    pre-LK, and refined poses are evaluated on the SAME Stage-A ORB-Lowe
    correspondences, avoiding a misleading comparison between inlier ratios
    computed from different candidate sets.

    The quality score combines:
      * shared epipolar inlier ratio,
      * spatial coverage of those shared inliers,
      * median Sampson residual of the shared inliers.

    No ground truth is used for model selection.
    """
    p1 = np.asarray(points1, dtype=np.float64).reshape(-1, 2)
    p2 = np.asarray(points2, dtype=np.float64).reshape(-1, 2)
    n = len(p1)
    empty = {
        "shared_quality_score": 0.0,
        "shared_inliers": 0.0,
        "shared_inlier_ratio": 0.0,
        "shared_coverage": 0.0,
        "shared_median_sampson_px": float("inf"),
    }
    if (
        R_est is None
        or t_est is None
        or n < cfg.min_pose_matches
        or len(p2) != n
    ):
        return empty

    try:
        if cfg.use_distortion_correction:
            q1 = cv.undistortPoints(
                p1.reshape(-1, 1, 2), camera.K, camera.dist, P=camera.K
            ).reshape(-1, 2)
            q2 = cv.undistortPoints(
                p2.reshape(-1, 1, 2), camera.K, camera.dist, P=camera.K
            ).reshape(-1, 2)
        else:
            q1, q2 = p1, p2

        E = _skew_symmetric(safe_normalize(t_est)) @ np.asarray(
            R_est, dtype=np.float64
        ).reshape(3, 3)
        K_inv = np.linalg.inv(camera.K)
        F = K_inv.T @ E @ K_inv

        x1 = np.column_stack([q1, np.ones(n, dtype=np.float64)])
        x2 = np.column_stack([q2, np.ones(n, dtype=np.float64)])

        Fx1 = (F @ x1.T).T
        Ftx2 = (F.T @ x2.T).T
        numerator = np.sum(x2 * Fx1, axis=1) ** 2
        denominator = (
            Fx1[:, 0] ** 2
            + Fx1[:, 1] ** 2
            + Ftx2[:, 0] ** 2
            + Ftx2[:, 1] ** 2
        )
        sampson_px = np.sqrt(
            numerator / np.maximum(denominator, EPS)
        )
        valid = np.isfinite(sampson_px)
        threshold = max(float(cfg.magsac_threshold_px), EPS)
        shared_mask = valid & (sampson_px <= threshold)
        n_inliers = int(np.count_nonzero(shared_mask))
        ratio = n_inliers / max(n, 1)
        coverage = _spatial_coverage_ratio(
            p1, shared_mask, image_shape, rows=4, cols=6
        )
        if n_inliers:
            median_residual = float(np.median(sampson_px[shared_mask]))
            residual_quality = float(
                np.exp(-median_residual / threshold)
            )
        else:
            median_residual = float("inf")
            residual_quality = 0.0

        # Because every candidate is scored on the same correspondences, the
        # ratio is directly comparable. Coverage prevents a compact cluster of
        # inliers from dominating, while residual quality breaks close ties.
        score = (
            0.50 * float(ratio)
            + 0.30 * float(coverage)
            + 0.20 * residual_quality
        )
        return {
            "shared_quality_score": float(score),
            "shared_inliers": float(n_inliers),
            "shared_inlier_ratio": float(ratio),
            "shared_coverage": float(coverage),
            "shared_median_sampson_px": float(median_residual),
        }
    except Exception:
        return empty



def _poseguard_shared_quality(
    R_est: Optional[np.ndarray],
    t_est: Optional[np.ndarray],
    points1: np.ndarray,
    points2: np.ndarray,
    camera: CameraModel,
    cfg: PipelineConfig,
    image_shape: Tuple[int, int],
) -> Dict[str, float]:
    """Evaluate a candidate pose on one shared correspondence set.

    PoseGuard-v2 addresses a failure mode of the first guarded variant: a pose
    can fit the epipolar constraint well yet yield a weak or ambiguous
    translation after essential-matrix decomposition. Competing poses are
    therefore judged using four complementary quantities on the SAME Stage-A
    ORB-Lowe correspondences:

      1. epipolar support,
      2. positive-depth (cheirality) support,
      3. spatial coverage of the positive-depth support,
      4. rotation-compensated bearing parallax.

    No benchmark ground truth is used.
    """
    p1 = np.asarray(points1, dtype=np.float64).reshape(-1, 2)
    p2 = np.asarray(points2, dtype=np.float64).reshape(-1, 2)
    n = len(p1)
    empty = {
        "poseguard_quality_score": 0.0,
        "poseguard_epipolar_inliers": 0.0,
        "poseguard_epipolar_ratio": 0.0,
        "poseguard_cheirality_inliers": 0.0,
        "poseguard_cheirality_ratio": 0.0,
        "poseguard_cheirality_coverage": 0.0,
        "poseguard_median_sampson_px": float("inf"),
        "poseguard_median_parallax_deg": 0.0,
        "poseguard_parallax_quality": 0.0,
    }
    if (
        R_est is None
        or t_est is None
        or n < cfg.min_pose_matches
        or len(p2) != n
    ):
        return empty

    try:
        R = np.asarray(R_est, dtype=np.float64).reshape(3, 3)
        t = safe_normalize(t_est).reshape(3)
        if np.linalg.norm(t) < EPS:
            return empty

        # Normalized image coordinates are used for triangulation/bearings.
        if cfg.use_distortion_correction:
            n1 = cv.undistortPoints(
                p1.reshape(-1, 1, 2), camera.K, camera.dist
            ).reshape(-1, 2)
            n2 = cv.undistortPoints(
                p2.reshape(-1, 1, 2), camera.K, camera.dist
            ).reshape(-1, 2)
            q1 = cv.undistortPoints(
                p1.reshape(-1, 1, 2), camera.K, camera.dist, P=camera.K
            ).reshape(-1, 2)
            q2 = cv.undistortPoints(
                p2.reshape(-1, 1, 2), camera.K, camera.dist, P=camera.K
            ).reshape(-1, 2)
        else:
            K_inv = np.linalg.inv(camera.K)
            x1h = np.column_stack([p1, np.ones(n)])
            x2h = np.column_stack([p2, np.ones(n)])
            b1h = (K_inv @ x1h.T).T
            b2h = (K_inv @ x2h.T).T
            n1 = b1h[:, :2] / np.maximum(b1h[:, 2:3], EPS)
            n2 = b2h[:, :2] / np.maximum(b2h[:, 2:3], EPS)
            q1, q2 = p1, p2

        # Epipolar residual in pixel coordinates, directly comparable to the
        # configured MAGSAC threshold.
        E = _skew_symmetric(t) @ R
        K_inv = np.linalg.inv(camera.K)
        F = K_inv.T @ E @ K_inv
        x1 = np.column_stack([q1, np.ones(n, dtype=np.float64)])
        x2 = np.column_stack([q2, np.ones(n, dtype=np.float64)])
        Fx1 = (F @ x1.T).T
        Ftx2 = (F.T @ x2.T).T
        numerator = np.sum(x2 * Fx1, axis=1) ** 2
        denominator = (
            Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2
            + Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2
        )
        sampson_px = np.sqrt(numerator / np.maximum(denominator, EPS))
        threshold = max(float(cfg.magsac_threshold_px), EPS)
        epi_mask = np.isfinite(sampson_px) & (sampson_px <= threshold)
        epi_count = int(np.count_nonzero(epi_mask))
        epi_ratio = epi_count / max(n, 1)
        if epi_count:
            median_sampson = float(np.median(sampson_px[epi_mask]))
            residual_quality = float(np.exp(-median_sampson / threshold))
        else:
            median_sampson = float("inf")
            residual_quality = 0.0

        # Triangulate only epipolar-consistent correspondences, then require
        # positive depth in both cameras. This directly probes whether the
        # candidate essential-matrix decomposition supports a physically
        # plausible relative pose.
        cheirality_mask = np.zeros(n, dtype=bool)
        epi_idx = np.flatnonzero(epi_mask)
        if len(epi_idx) >= cfg.min_pose_matches:
            P1 = np.column_stack([np.eye(3), np.zeros(3)])
            P2 = np.column_stack([R, t])
            Xh = cv.triangulatePoints(
                P1,
                P2,
                n1[epi_idx].T.astype(np.float64),
                n2[epi_idx].T.astype(np.float64),
            )
            valid_w = np.abs(Xh[3]) > EPS
            X = np.zeros((len(epi_idx), 3), dtype=np.float64)
            X[valid_w] = (Xh[:3, valid_w] / Xh[3:4, valid_w]).T
            X2 = (R @ X.T + t.reshape(3, 1)).T
            positive = (
                valid_w
                & np.isfinite(X).all(axis=1)
                & np.isfinite(X2).all(axis=1)
                & (X[:, 2] > 0.0)
                & (X2[:, 2] > 0.0)
            )
            cheirality_mask[epi_idx[positive]] = True

        ch_count = int(np.count_nonzero(cheirality_mask))
        ch_ratio = ch_count / max(n, 1)
        ch_coverage = _spatial_coverage_ratio(
            p1, cheirality_mask, image_shape, rows=4, cols=6
        )

        # Rotation-compensated bearing parallax. With little parallax,
        # translation direction is weakly observable even when epipolar
        # residuals are small.
        parallax_deg = 0.0
        if ch_count:
            b1 = np.column_stack([n1, np.ones(n, dtype=np.float64)])
            b2 = np.column_stack([n2, np.ones(n, dtype=np.float64)])
            b1 /= np.maximum(np.linalg.norm(b1, axis=1, keepdims=True), EPS)
            b2 /= np.maximum(np.linalg.norm(b2, axis=1, keepdims=True), EPS)
            b1_rot = (R @ b1.T).T
            dots = np.sum(b1_rot[cheirality_mask] * b2[cheirality_mask], axis=1)
            dots = np.clip(dots, -1.0, 1.0)
            angles = np.degrees(np.arccos(dots))
            if len(angles):
                parallax_deg = float(np.median(angles))
        parallax_target = max(
            float(cfg.adaptive_poseguard_parallax_target_deg), EPS
        )
        parallax_quality = float(np.clip(parallax_deg / parallax_target, 0.0, 1.0))

        # Cheirality is deliberately the largest term because the first
        # Guarded experiment showed that epipolar fit alone can protect a poor
        # translation hypothesis. Parallax contributes observability evidence.
        score = (
            0.40 * ch_ratio
            + 0.20 * ch_coverage
            + 0.15 * epi_ratio
            + 0.10 * residual_quality
            + 0.15 * parallax_quality
        )
        return {
            "poseguard_quality_score": float(score),
            "poseguard_epipolar_inliers": float(epi_count),
            "poseguard_epipolar_ratio": float(epi_ratio),
            "poseguard_cheirality_inliers": float(ch_count),
            "poseguard_cheirality_ratio": float(ch_ratio),
            "poseguard_cheirality_coverage": float(ch_coverage),
            "poseguard_median_sampson_px": float(median_sampson),
            "poseguard_median_parallax_deg": float(parallax_deg),
            "poseguard_parallax_quality": float(parallax_quality),
        }
    except Exception:
        return empty


def _poseguard_should_replace(
    current: Mapping[str, float],
    candidate: Mapping[str, float],
    cfg: PipelineConfig,
    *,
    prelk: bool = False,
) -> Tuple[bool, str]:
    """Decide whether a new candidate should replace the current pose.

    A strong cheirality rescue can replace a weak fast pose even when the
    aggregate score improvement is modest. Otherwise replacement requires an
    overall quality margin and prohibits a substantial cheirality loss.
    """
    cur_q = float(current.get("poseguard_quality_score", 0.0))
    new_q = float(candidate.get("poseguard_quality_score", 0.0))
    cur_ch = float(current.get("poseguard_cheirality_ratio", 0.0))
    new_ch = float(candidate.get("poseguard_cheirality_ratio", 0.0))
    cur_native = float(current.get("poseguard_native_cheirality_ratio", cur_ch))
    new_native = float(candidate.get("poseguard_native_cheirality_ratio", new_ch))

    shared_rescue = bool(
        cur_ch <= float(cfg.adaptive_poseguard_rescue_fast_cheirality_max)
        and new_ch >= float(cfg.adaptive_poseguard_rescue_candidate_cheirality_min)
        and (new_ch - cur_ch) >= float(cfg.adaptive_poseguard_rescue_cheirality_gain)
    )
    native_rescue = bool(
        cur_native <= float(cfg.adaptive_poseguard_rescue_fast_cheirality_max)
        and new_native >= max(0.80, float(cfg.adaptive_poseguard_rescue_candidate_cheirality_min))
        and (new_native - cur_native) >= max(0.40, float(cfg.adaptive_poseguard_rescue_cheirality_gain))
    )
    rescue = shared_rescue or native_rescue
    if rescue:
        return True, "cheirality_rescue"

    margin = (
        float(cfg.adaptive_poseguard_prelk_margin)
        if prelk
        else float(cfg.adaptive_poseguard_quality_margin)
    )
    cheirality_ok = bool(
        new_ch + float(cfg.adaptive_poseguard_cheirality_drop_tolerance) >= cur_ch
    )
    if cheirality_ok and new_q >= cur_q + margin:
        return True, "quality_margin"
    return False, "keep_current"


def _poseguard_v3_candidate_observable(
    candidate: Mapping[str, float],
    cfg: PipelineConfig,
) -> Tuple[bool, str]:
    """Hard validity gate for a candidate that would replace another pose.

    PoseGuard-v2 could still select a candidate whose aggregate score rose even
    though its own recovered pose had very little positive-depth support and
    almost no rotation-compensated parallax. V3 separates *validity* from
    *ranking*: a replacement candidate must first be sufficiently observable.

    This gate is deliberately applied to replacement candidates, not to the
    benchmark's definition of pose-recovery success. Thus the benchmark success
    criterion remains unchanged and the selector simply refuses to replace a
    current solution with an observably weaker candidate.
    """
    native_ch = float(candidate.get("poseguard_native_cheirality_ratio", 0.0))
    parallax = float(candidate.get("poseguard_median_parallax_deg", 0.0))

    if native_ch < float(cfg.adaptive_poseguard_v3_min_candidate_native_cheirality):
        return False, "native_cheirality_below_gate"
    if parallax < float(cfg.adaptive_poseguard_v3_min_candidate_parallax_deg):
        return False, "parallax_below_gate"
    return True, "observable"


def _poseguard_v3_should_replace(
    current: Mapping[str, float],
    candidate: Mapping[str, float],
    cfg: PipelineConfig,
    *,
    prelk: bool = False,
) -> Tuple[bool, str]:
    """PoseGuard-v3 replacement rule.

    The decision is intentionally two-stage:

      1. the candidate must pass a positive-depth/parallax observability gate;
      2. it must either provide a large cheirality gain (rescue) or exceed the
         current aggregate quality by a fixed margin without materially losing
         positive-depth support.

    The rescue rule depends on *gain* rather than requiring the current pose to
    lie below one fixed failure threshold. This permits recovery when a
    structurally filtered candidate is substantially stronger even if the fast
    pose was only moderately weak.
    """
    observable, obs_reason = _poseguard_v3_candidate_observable(candidate, cfg)
    if not observable:
        return False, f"veto_{obs_reason}"

    cur_q = float(current.get("poseguard_quality_score", 0.0))
    new_q = float(candidate.get("poseguard_quality_score", 0.0))
    cur_shared = float(current.get("poseguard_cheirality_ratio", 0.0))
    new_shared = float(candidate.get("poseguard_cheirality_ratio", 0.0))
    cur_native = float(current.get("poseguard_native_cheirality_ratio", cur_shared))
    new_native = float(candidate.get("poseguard_native_cheirality_ratio", new_shared))

    native_rescue = bool(
        new_native >= float(cfg.adaptive_poseguard_v3_rescue_candidate_native_cheirality_min)
        and (new_native - cur_native)
        >= float(cfg.adaptive_poseguard_v3_rescue_native_cheirality_gain)
    )
    shared_rescue = bool(
        new_shared >= float(cfg.adaptive_poseguard_v3_rescue_candidate_shared_cheirality_min)
        and (new_shared - cur_shared)
        >= float(cfg.adaptive_poseguard_v3_rescue_shared_cheirality_gain)
    )
    if native_rescue or shared_rescue:
        return True, "cheirality_gain_rescue"

    margin = (
        float(cfg.adaptive_poseguard_v3_prelk_margin)
        if prelk
        else float(cfg.adaptive_poseguard_v3_quality_margin)
    )
    shared_ok = bool(
        new_shared
        + float(cfg.adaptive_poseguard_v3_shared_cheirality_drop_tolerance)
        >= cur_shared
    )
    native_ok = bool(
        new_native
        + float(cfg.adaptive_poseguard_v3_native_cheirality_drop_tolerance)
        >= cur_native
    )
    if shared_ok and native_ok and new_q >= cur_q + margin:
        return True, "quality_margin_after_observability"
    return False, "keep_current"


def _poseguard_v3_should_abort_lk(
    fast_success: bool,
    pre_candidate: Mapping[str, float],
    cfg: PipelineConfig,
) -> Tuple[bool, str]:
    """Skip LK only for an extremely weak pre-LK candidate when fast pose exists.

    The early stop is conservative: it never converts a pair with no valid fast
    pose into failure, because LK remains the last opportunity to recover such a
    pair. It is intended only to avoid spending optical-flow time on a fallback
    whose native positive-depth support and parallax are both extremely small.
    """
    if not fast_success:
        return False, "no_fast_pose_keep_refining"
    native_ch = float(pre_candidate.get("poseguard_native_cheirality_ratio", 0.0))
    parallax = float(pre_candidate.get("poseguard_median_parallax_deg", 0.0))
    if (
        native_ch <= float(cfg.adaptive_poseguard_v3_early_abort_native_cheirality_max)
        and parallax <= float(cfg.adaptive_poseguard_v3_early_abort_parallax_max_deg)
    ):
        return True, "very_low_cheirality_and_parallax"
    return False, "continue_to_lk"



def _poseguard_v4_pose_disagreement(
    current_R: Optional[np.ndarray],
    current_t: Optional[np.ndarray],
    candidate_R: Optional[np.ndarray],
    candidate_t: Optional[np.ndarray],
) -> Tuple[float, float]:
    """Return rotation and sign-invariant translation disagreement in degrees.

    LK is only a local correspondence refinement. If a small image-space
    correction causes a large change in recovered pose, the two-view solution
    is numerically unstable. PoseGuard-v4 uses this disagreement only to decide
    whether LK may replace an already-selected pre-LK structural pose.
    """
    if current_R is None or current_t is None or candidate_R is None or candidate_t is None:
        return float("inf"), float("inf")
    dR = rotation_error_deg(np.asarray(candidate_R), np.asarray(current_R))
    dt = min(
        vector_angle_deg(np.asarray(candidate_t), np.asarray(current_t)),
        vector_angle_deg(-np.asarray(candidate_t), np.asarray(current_t)),
    )
    return float(dR), float(dt)


def _poseguard_v4_lk_should_replace_prelk(
    pre_quality: Mapping[str, float],
    ref_quality: Mapping[str, float],
    pre_R: Optional[np.ndarray],
    pre_t: Optional[np.ndarray],
    ref_R: Optional[np.ndarray],
    ref_t: Optional[np.ndarray],
    cfg: PipelineConfig,
) -> Tuple[bool, str, Dict[str, float]]:
    """Conservative LK-over-pre-LK replacement rule used only by v4.

    Cheirality rescue is deliberately *not* used here. A structural pre-LK
    pose that has already beaten the fast ORB solution is considered the
    current geometric hypothesis. LK may replace it only if the refined pose
    remains observable, is pose-consistent with it, does not imply a large
    parallax jump, preserves positive-depth support, and improves the aggregate
    quality by a small fixed margin.
    """
    observable, obs_reason = _poseguard_v3_candidate_observable(ref_quality, cfg)

    dR, dt = _poseguard_v4_pose_disagreement(pre_R, pre_t, ref_R, ref_t)
    pre_parallax = float(pre_quality.get("poseguard_median_parallax_deg", 0.0))
    ref_parallax = float(ref_quality.get("poseguard_median_parallax_deg", 0.0))
    parallax_ratio = ref_parallax / max(pre_parallax, EPS)

    diagnostics = {
        "poseguard_v4_lk_rotation_disagreement_deg": float(dR),
        "poseguard_v4_lk_translation_disagreement_deg": float(dt),
        "poseguard_v4_lk_parallax_ratio": float(parallax_ratio),
    }

    if not observable:
        return False, f"veto_{obs_reason}", diagnostics
    if dR > float(cfg.adaptive_poseguard_v4_lk_max_rotation_disagreement_deg):
        return False, "veto_lk_rotation_instability", diagnostics
    if dt > float(cfg.adaptive_poseguard_v4_lk_max_translation_disagreement_deg):
        return False, "veto_lk_translation_instability", diagnostics
    if parallax_ratio < float(cfg.adaptive_poseguard_v4_lk_parallax_ratio_min):
        return False, "veto_lk_parallax_collapse", diagnostics
    if parallax_ratio > float(cfg.adaptive_poseguard_v4_lk_parallax_ratio_max):
        return False, "veto_lk_parallax_jump", diagnostics

    pre_shared = float(pre_quality.get("poseguard_cheirality_ratio", 0.0))
    ref_shared = float(ref_quality.get("poseguard_cheirality_ratio", 0.0))
    pre_native = float(pre_quality.get("poseguard_native_cheirality_ratio", pre_shared))
    ref_native = float(ref_quality.get("poseguard_native_cheirality_ratio", ref_shared))
    if (
        ref_shared
        + float(cfg.adaptive_poseguard_v4_lk_shared_cheirality_drop_tolerance)
        < pre_shared
    ):
        return False, "veto_lk_shared_cheirality_drop", diagnostics
    if (
        ref_native
        + float(cfg.adaptive_poseguard_v4_lk_native_cheirality_drop_tolerance)
        < pre_native
    ):
        return False, "veto_lk_native_cheirality_drop", diagnostics

    pre_q = float(pre_quality.get("poseguard_quality_score", 0.0))
    ref_q = float(ref_quality.get("poseguard_quality_score", 0.0))
    if ref_q < pre_q + float(cfg.adaptive_poseguard_v4_lk_quality_margin):
        return False, "keep_prelk_insufficient_lk_quality_gain", diagnostics

    return True, "stable_lk_quality_gain", diagnostics

def _adaptive_guarded_soft_accept(
    fast_success: bool,
    conf_diag: Mapping[str, float],
    cfg: PipelineConfig,
) -> bool:
    """Allow a borderline but consistently strong fast pose to exit early.

    The original Adaptive method requires every hard threshold to pass. The
    guarded variant additionally allows early acceptance when the aggregate
    confidence is very high and every individual measure remains within a
    configurable fraction of its hard threshold. This avoids expensive
    fallbacks caused by a tiny miss on only one criterion.
    """
    if not fast_success:
        return False
    floor = float(
        np.clip(cfg.adaptive_guarded_soft_floor_fraction, 0.0, 1.0)
    )
    return bool(
        float(conf_diag.get("adaptive_fast_confidence_score", 0.0))
        >= float(cfg.adaptive_guarded_soft_score)
        and float(conf_diag.get("adaptive_fast_inliers", 0.0))
        >= floor * float(cfg.adaptive_accept_inliers)
        and float(conf_diag.get("adaptive_fast_inlier_ratio", 0.0))
        >= floor * float(cfg.adaptive_accept_inlier_ratio)
        and float(conf_diag.get("adaptive_fast_coverage", 0.0))
        >= floor * float(cfg.adaptive_accept_coverage)
    )


def descriptor_norm(desc: Optional[np.ndarray]) -> int:
    if desc is None:
        return cv.NORM_L2
    return cv.NORM_HAMMING if desc.dtype == np.uint8 else cv.NORM_L2


# ---------------------------------------------------------------------------
# Match generation and filtering
# ---------------------------------------------------------------------------


def _knn2(matcher: cv.DescriptorMatcher, query: np.ndarray, train: np.ndarray) -> List[List[cv.DMatch]]:
    if query is None or train is None or len(query) == 0 or len(train) < 2:
        return []
    return matcher.knnMatch(query, train, k=2)


def mutual_ratio_matches(
    f1: FeatureSet,
    f2: FeatureSet,
    fixed_ratio: float,
    adaptive: bool,
    cfg: PipelineConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if f1.descriptors is None or f2.descriptors is None:
        return np.empty((0, 2), np.int32), np.empty(0, np.float32), np.empty(0, np.float32)
    if len(f1.descriptors) < 2 or len(f2.descriptors) < 2:
        return np.empty((0, 2), np.int32), np.empty(0, np.float32), np.empty(0, np.float32)

    matcher = cv.BFMatcher(descriptor_norm(f1.descriptors), crossCheck=False)
    forward = _knn2(matcher, f1.descriptors, f2.descriptors)
    reverse = _knn2(matcher, f2.descriptors, f1.descriptors)

    reverse_best: Dict[int, int] = {}
    reverse_ratio_ok: Dict[int, bool] = {}
    for pair in reverse:
        if len(pair) < 2:
            continue
        m, n = pair
        reverse_best[m.queryIdx] = m.trainIdx
        reverse_ratio_ok[m.queryIdx] = (m.distance / max(n.distance, EPS)) < fixed_ratio

    pairs: List[Tuple[int, int]] = []
    distances: List[float] = []
    ratios: List[float] = []
    for pair in forward:
        if len(pair) < 2:
            continue
        m, n = pair
        ratio = float(m.distance / max(n.distance, EPS))
        if adaptive:
            conf = float(f1.edge_confidence[m.queryIdx]) if len(f1.edge_confidence) else 0.5
            # High-confidence structural points can tolerate a slightly larger
            # ratio; weak/non-edge points must be more distinctive.
            limit = cfg.ratio + 0.12 * (conf - 0.5)
            limit = float(np.clip(limit, cfg.adaptive_ratio_min, cfg.adaptive_ratio_max))
        else:
            limit = fixed_ratio
        if ratio >= limit:
            continue
        if reverse_best.get(m.trainIdx) != m.queryIdx:
            continue
        if not adaptive and not reverse_ratio_ok.get(m.trainIdx, False):
            continue
        pairs.append((m.queryIdx, m.trainIdx))
        distances.append(float(m.distance))
        ratios.append(ratio)

    return (
        np.asarray(pairs, dtype=np.int32).reshape(-1, 2),
        np.asarray(distances, dtype=np.float32),
        np.asarray(ratios, dtype=np.float32),
    )


def crosscheck_matches(f1: FeatureSet, f2: FeatureSet) -> Tuple[np.ndarray, np.ndarray]:
    if f1.descriptors is None or f2.descriptors is None:
        return np.empty((0, 2), np.int32), np.empty(0, np.float32)
    if len(f1.descriptors) == 0 or len(f2.descriptors) == 0:
        return np.empty((0, 2), np.int32), np.empty(0, np.float32)
    matcher = cv.BFMatcher(descriptor_norm(f1.descriptors), crossCheck=True)
    matches = sorted(matcher.match(f1.descriptors, f2.descriptors), key=lambda m: m.distance)
    pairs = np.asarray([(m.queryIdx, m.trainIdx) for m in matches], dtype=np.int32).reshape(-1, 2)
    distances = np.asarray([m.distance for m in matches], dtype=np.float32)
    return pairs, distances



def nearest_neighbor_matches(f1: FeatureSet, f2: FeatureSet) -> Tuple[np.ndarray, np.ndarray]:
    """Return the single nearest descriptor in image 2 for every descriptor in image 1.

    This reproduces the unfiltered NN baseline from the original notebook, but
    uses OpenCV's native matcher rather than allocating an N1 x N2 x D tensor.
    """
    if f1.descriptors is None or f2.descriptors is None:
        return np.empty((0, 2), np.int32), np.empty(0, np.float32)
    if len(f1.descriptors) == 0 or len(f2.descriptors) == 0:
        return np.empty((0, 2), np.int32), np.empty(0, np.float32)
    matcher = cv.BFMatcher(descriptor_norm(f1.descriptors), crossCheck=False)
    matches = matcher.match(f1.descriptors, f2.descriptors)
    matches = sorted(matches, key=lambda m: m.distance)
    pairs = np.asarray([(m.queryIdx, m.trainIdx) for m in matches], dtype=np.int32).reshape(-1, 2)
    distances = np.asarray([m.distance for m in matches], dtype=np.float32)
    return pairs, distances


def oneway_ratio_matches(
    f1: FeatureSet,
    f2: FeatureSet,
    ratio: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standard one-direction Lowe two-nearest-neighbour ratio test."""
    if f1.descriptors is None or f2.descriptors is None:
        return np.empty((0, 2), np.int32), np.empty(0, np.float32), np.empty(0, np.float32)
    if len(f1.descriptors) == 0 or len(f2.descriptors) < 2:
        return np.empty((0, 2), np.int32), np.empty(0, np.float32), np.empty(0, np.float32)
    matcher = cv.BFMatcher(descriptor_norm(f1.descriptors), crossCheck=False)
    knn = _knn2(matcher, f1.descriptors, f2.descriptors)
    pairs: List[Tuple[int, int]] = []
    distances: List[float] = []
    ratios: List[float] = []
    for candidates in knn:
        if len(candidates) < 2:
            continue
        m, n = candidates
        r = float(m.distance / max(n.distance, EPS))
        if r < ratio:
            pairs.append((m.queryIdx, m.trainIdx))
            distances.append(float(m.distance))
            ratios.append(r)
    return (
        np.asarray(pairs, dtype=np.int32).reshape(-1, 2),
        np.asarray(distances, dtype=np.float32),
        np.asarray(ratios, dtype=np.float32),
    )


def orientation_scale_consistency(
    f1: FeatureSet,
    f2: FeatureSet,
    pairs: np.ndarray,
    cfg: PipelineConfig,
) -> np.ndarray:
    n = len(pairs)
    if n < cfg.min_filter_matches:
        return np.ones(n, dtype=bool)

    angles1 = np.radians(np.asarray([f1.keypoints[i].angle for i in pairs[:, 0]], dtype=np.float64))
    angles2 = np.radians(np.asarray([f2.keypoints[j].angle for j in pairs[:, 1]], dtype=np.float64))
    valid_angle = (angles1 >= 0.0) & (angles2 >= 0.0)

    keep = np.ones(n, dtype=bool)
    if np.count_nonzero(valid_angle) >= cfg.min_filter_matches:
        delta = wrap_angle_rad(angles2[valid_angle] - angles1[valid_angle])
        center = circular_median(delta)
        residual = np.abs(wrap_angle_rad(delta - center))
        mad = robust_mad(residual, center=0.0)
        threshold = max(math.radians(cfg.orientation_min_deg), cfg.orientation_mad_scale * mad)
        local = residual <= threshold
        angle_keep = np.ones(n, dtype=bool)
        angle_keep[np.flatnonzero(valid_angle)] = local
        if np.count_nonzero(angle_keep) >= cfg.min_filter_matches:
            keep &= angle_keep

    size1 = np.asarray([max(f1.keypoints[i].size, EPS) for i in pairs[:, 0]], dtype=np.float64)
    size2 = np.asarray([max(f2.keypoints[j].size, EPS) for j in pairs[:, 1]], dtype=np.float64)
    log_scale = np.log(size2 / size1)
    center_s = float(np.median(log_scale))
    mad_s = robust_mad(log_scale, center_s)
    threshold_s = max(cfg.scale_min_log, cfg.scale_mad_scale * mad_s)
    scale_keep = np.abs(log_scale - center_s) <= threshold_s
    if np.count_nonzero(keep & scale_keep) >= cfg.min_filter_matches:
        keep &= scale_keep
    return keep


def local_grid_motion_consistency(
    points1: np.ndarray,
    points2: np.ndarray,
    image_shape: Tuple[int, int],
    cfg: PipelineConfig,
) -> np.ndarray:
    n = len(points1)
    if n < cfg.min_filter_matches:
        return np.ones(n, dtype=bool)
    h, w = image_shape
    flow = points2 - points1
    rows, cols = cfg.grid_rows, cfg.grid_cols
    cell_r = np.clip((points1[:, 1] * rows / max(h, 1)).astype(int), 0, rows - 1)
    cell_c = np.clip((points1[:, 0] * cols / max(w, 1)).astype(int), 0, cols - 1)

    buckets: Dict[Tuple[int, int], np.ndarray] = {}
    for r in range(rows):
        for c in range(cols):
            idx = np.flatnonzero((cell_r == r) & (cell_c == c))
            if len(idx):
                buckets[(r, c)] = idx

    global_center = np.median(flow, axis=0)
    global_res = np.linalg.norm(flow - global_center, axis=1)
    global_thr = max(cfg.local_motion_min_px, cfg.local_motion_mad_scale * robust_mad(global_res, 0.0))
    keep = np.zeros(n, dtype=bool)

    # Compute one robust local motion model per occupied grid cell, then apply
    # it vectorially to all matches in that cell.
    for (r, c), idx_current in buckets.items():
        neighbor_arrays = []
        for rr in range(max(0, r - 1), min(rows, r + 2)):
            for cc in range(max(0, c - 1), min(cols, c + 2)):
                arr = buckets.get((rr, cc))
                if arr is not None:
                    neighbor_arrays.append(arr)
        neighborhood = np.concatenate(neighbor_arrays) if neighbor_arrays else idx_current
        if len(neighborhood) >= 4:
            local_flow = flow[neighborhood]
            center = np.median(local_flow, axis=0)
            residuals = np.linalg.norm(local_flow - center, axis=1)
            threshold = max(
                cfg.local_motion_min_px,
                cfg.local_motion_mad_scale * robust_mad(residuals, 0.0),
            )
        else:
            center = global_center
            threshold = global_thr
        keep[idx_current] = np.linalg.norm(flow[idx_current] - center, axis=1) <= threshold

    if np.count_nonzero(keep) < cfg.min_filter_matches:
        return np.ones(n, dtype=bool)
    return keep

def edge_orientation_consistency(
    f1: FeatureSet,
    f2: FeatureSet,
    pairs: np.ndarray,
    cfg: PipelineConfig,
) -> np.ndarray:
    n = len(pairs)
    if n < cfg.min_filter_matches:
        return np.ones(n, dtype=bool)
    a1 = f1.gradient_angle[pairs[:, 0]].astype(np.float64)
    a2 = f2.gradient_angle[pairs[:, 1]].astype(np.float64)
    strength = np.minimum(
        f1.gradient_magnitude[pairs[:, 0]],
        f2.gradient_magnitude[pairs[:, 1]],
    )
    reliable = strength > 0.08
    if np.count_nonzero(reliable) < cfg.min_filter_matches:
        return np.ones(n, dtype=bool)
    delta = wrap_angle_rad(a2[reliable] - a1[reliable])
    center = circular_median(delta)
    residual = np.abs(wrap_angle_rad(delta - center))
    threshold = max(math.radians(22.0), 3.0 * robust_mad(residual, 0.0))
    local_keep = residual <= threshold
    keep = np.ones(n, dtype=bool)
    keep[np.flatnonzero(reliable)] = local_keep
    if np.count_nonzero(keep) < cfg.min_filter_matches:
        return np.ones(n, dtype=bool)
    return keep


def refine_matches_lk(
    gray1: np.ndarray,
    gray2: np.ndarray,
    points1: np.ndarray,
    points2: np.ndarray,
    cfg: PipelineConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    n = len(points1)
    if n < cfg.min_filter_matches:
        return points1, points2, np.ones(n, dtype=bool), {"lk_median_fb_px": float("nan")}

    p1 = np.asarray(points1, np.float32).reshape(-1, 1, 2)
    p2_init = np.asarray(points2, np.float32).reshape(-1, 1, 2)
    criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.01)
    p2_ref, st12, _ = cv.calcOpticalFlowPyrLK(
        gray1,
        gray2,
        p1,
        p2_init.copy(),
        winSize=(cfg.lk_window, cfg.lk_window),
        maxLevel=cfg.lk_levels,
        criteria=criteria,
        flags=cv.OPTFLOW_USE_INITIAL_FLOW,
        minEigThreshold=1e-4,
    )
    if p2_ref is None or st12 is None:
        return points1, points2, np.ones(n, dtype=bool), {"lk_median_fb_px": float("nan")}

    p1_back, st21, _ = cv.calcOpticalFlowPyrLK(
        gray2,
        gray1,
        p2_ref,
        p1.copy(),
        winSize=(cfg.lk_window, cfg.lk_window),
        maxLevel=cfg.lk_levels,
        criteria=criteria,
        flags=cv.OPTFLOW_USE_INITIAL_FLOW,
        minEigThreshold=1e-4,
    )
    if p1_back is None or st21 is None:
        return points1, points2, np.ones(n, dtype=bool), {"lk_median_fb_px": float("nan")}

    p1_flat = p1.reshape(-1, 2)
    p2_flat = p2_ref.reshape(-1, 2)
    p2_initial_flat = p2_init.reshape(-1, 2)
    p1_back_flat = p1_back.reshape(-1, 2)
    fb = np.linalg.norm(p1_back_flat - p1_flat, axis=1)
    correction = np.linalg.norm(p2_flat - p2_initial_flat, axis=1)
    keep = (
        st12.reshape(-1).astype(bool)
        & st21.reshape(-1).astype(bool)
        & np.isfinite(fb)
        & np.isfinite(correction)
        & (fb <= cfg.lk_fb_threshold)
        & (correction <= cfg.lk_max_correction)
    )
    if np.count_nonzero(keep) < cfg.min_filter_matches:
        return points1, points2, np.ones(n, dtype=bool), {
            "lk_median_fb_px": float(np.nanmedian(fb)),
            "lk_acceptance": 0.0,
        }
    return p1_flat, p2_flat, keep, {
        "lk_median_fb_px": float(np.median(fb[keep])),
        "lk_acceptance": float(np.mean(keep)),
    }


# ---------------------------------------------------------------------------
# Pose estimation
# ---------------------------------------------------------------------------


def estimate_pose(
    points1: np.ndarray,
    points2: np.ndarray,
    camera: CameraModel,
    cfg: PipelineConfig,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], np.ndarray, Dict[str, Any]]:
    n = len(points1)
    empty = np.zeros(n, dtype=bool)
    if n < cfg.min_pose_matches:
        return None, None, empty, {"pose_status": "insufficient_matches"}

    pts1 = np.asarray(points1, dtype=np.float64).reshape(-1, 1, 2)
    pts2 = np.asarray(points2, dtype=np.float64).reshape(-1, 1, 2)

    if cfg.use_distortion_correction:
        p1n = cv.undistortPoints(pts1, camera.K, camera.dist).reshape(-1, 2)
        p2n = cv.undistortPoints(pts2, camera.K, camera.dist).reshape(-1, 2)
        K_est = np.eye(3, dtype=np.float64)
        threshold = cfg.magsac_threshold_px / max((camera.fx + camera.fy) * 0.5, EPS)
    else:
        p1n = pts1.reshape(-1, 2)
        p2n = pts2.reshape(-1, 2)
        K_est = camera.K
        threshold = cfg.magsac_threshold_px

    method = getattr(cv, "USAC_MAGSAC", cv.RANSAC)
    try:
        E, mask = cv.findEssentialMat(
            p1n,
            p2n,
            K_est,
            method=method,
            prob=cfg.confidence,
            threshold=threshold,
            maxIters=cfg.max_iters,
        )
    except cv.error:
        E, mask = cv.findEssentialMat(
            p1n,
            p2n,
            K_est,
            method=cv.RANSAC,
            prob=cfg.confidence,
            threshold=threshold,
            maxIters=cfg.max_iters,
        )

    if E is None or mask is None:
        return None, None, empty, {"pose_status": "essential_failed"}

    # OpenCV can return stacked essential matrices. Try each and retain the
    # solution with the largest cheirality-consistent inlier set.
    candidates = [E]
    if E.shape[0] > 3 and E.shape[1] == 3:
        candidates = [E[i : i + 3] for i in range(0, E.shape[0], 3)]

    best = None
    for E_i in candidates:
        try:
            count, R_est, t_est, pose_mask = cv.recoverPose(E_i, p1n, p2n, K_est, mask=mask.copy())
        except cv.error:
            continue
        if best is None or int(count) > best[0]:
            best = (int(count), R_est, t_est.reshape(3), pose_mask.reshape(-1).astype(bool))

    if best is None:
        return None, None, empty, {"pose_status": "recover_pose_failed"}
    count, R_est, t_est, inlier_mask = best
    pose_status = "ok" if count >= cfg.min_pose_matches else "low_cheirality_inliers"
    return R_est, safe_normalize(t_est), inlier_mask, {
        "pose_status": pose_status,
        "estimator": "USAC_MAGSAC" if method == getattr(cv, "USAC_MAGSAC", -1) else "RANSAC",
        "recover_pose_inliers": count,
    }


# ---------------------------------------------------------------------------
# Pipelines
# ---------------------------------------------------------------------------


def run_baseline(
    method: str,
    gray1: np.ndarray,
    gray2: np.ndarray,
    camera: CameraModel,
    cfg: PipelineConfig,
) -> MethodResult:
    """Run one of the classical matchers from the original project.

    Naming is explicit:
      *_nn             raw one-way nearest neighbour
      *_lowe           standard one-way Lowe ratio test
      *_mutual_lowe    Lowe ratio in both directions + mutual NN
      orb_crosscheck   OpenCV BF Hamming cross-check
    """
    result = MethodResult(method=method, success=False, status="started")
    specs: Dict[str, Tuple[str, str, float]] = {
        "sift_nn": ("sift", "nn", 1.0),
        "orb_nn": ("orb", "nn", 1.0),
        "sift_lowe": ("sift", "ratio", 0.75),
        "orb_lowe": ("orb", "ratio", 0.80),
        # The original notebook used r=0.95 for the mutual Lowe comparison.
        "sift_mutual_lowe": ("sift", "mutual_ratio", 0.95),
        "orb_mutual_lowe": ("orb", "mutual_ratio", 0.95),
        "orb_crosscheck": ("orb", "crosscheck", 1.0),
        "akaze_lowe": ("akaze", "ratio", 0.80),
    }
    if method not in specs:
        raise ValueError(f"Unsupported baseline method: {method}")
    detector, mode, ratio = specs[method]

    with StageTimer(result.timings_ms, "extract_ms"):
        f1 = extract_standard(gray1, detector, cfg.nfeatures)
        f2 = extract_standard(gray2, detector, cfg.nfeatures)
    result.num_keypoints1 = len(f1.keypoints)
    result.num_keypoints2 = len(f2.keypoints)
    result.diagnostics.update(
        detector=detector.upper(),
        match_policy=mode,
        ratio_threshold=ratio if "ratio" in mode else float("nan"),
    )

    with StageTimer(result.timings_ms, "match_ms"):
        if mode == "nn":
            pairs, distances = nearest_neighbor_matches(f1, f2)
            ratios = np.empty(0, np.float32)
        elif mode == "crosscheck":
            pairs, distances = crosscheck_matches(f1, f2)
            ratios = np.empty(0, np.float32)
        elif mode == "ratio":
            pairs, distances, ratios = oneway_ratio_matches(f1, f2, ratio)
        elif mode == "mutual_ratio":
            pairs, distances, ratios = mutual_ratio_matches(
                f1, f2, fixed_ratio=ratio, adaptive=False, cfg=cfg
            )
        else:  # pragma: no cover
            raise RuntimeError(mode)

    result.num_raw_matches = len(pairs)
    result.num_filtered_matches = len(pairs)
    if len(distances):
        result.diagnostics["median_descriptor_distance"] = float(np.median(distances))
    if len(ratios):
        result.diagnostics["median_ratio"] = float(np.median(ratios))
    if len(pairs) < cfg.min_pose_matches:
        result.status = "insufficient_matches"
        return result

    result.points1 = np.ascontiguousarray(f1.points[pairs[:, 0]], dtype=np.float32)
    result.points2 = np.ascontiguousarray(f2.points[pairs[:, 1]], dtype=np.float32)
    with StageTimer(result.timings_ms, "geometry_ms"):
        R_est, t_est, mask, diag = estimate_pose(result.points1, result.points2, camera, cfg)
    result.diagnostics.update(diag)
    result.R, result.t, result.geometric_inlier_mask = R_est, t_est, mask
    result.num_geometric_inliers = int(np.count_nonzero(mask))
    result.success = R_est is not None and result.num_geometric_inliers >= cfg.min_pose_matches
    result.status = "ok" if result.success else str(diag.get("pose_status", "failed"))
    return result


def run_edgefusion(
    method: str,
    gray1: np.ndarray,
    gray2: np.ndarray,
    camera: CameraModel,
    cfg: PipelineConfig,
) -> MethodResult:
    result = MethodResult(method=method, success=False, status="started")

    with StageTimer(result.timings_ms, "extract_ms"):
        f1 = extract_edge_orb(gray1, cfg)
        f2 = extract_edge_orb(gray2, cfg)
    result.num_keypoints1 = len(f1.keypoints)
    result.num_keypoints2 = len(f2.keypoints)

    with StageTimer(result.timings_ms, "match_ms"):
        pairs, distances, ratios = mutual_ratio_matches(
            f1, f2, fixed_ratio=cfg.ratio, adaptive=True, cfg=cfg
        )
    result.num_raw_matches = len(pairs)
    if len(pairs) < cfg.min_pose_matches:
        result.status = "insufficient_matches"
        return result

    result.diagnostics["median_descriptor_distance"] = float(np.median(distances)) if len(distances) else float("nan")
    result.diagnostics["median_ratio"] = float(np.median(ratios)) if len(ratios) else float("nan")

    with StageTimer(result.timings_ms, "filter_ms"):
        keep = np.ones(len(pairs), dtype=bool)
        if cfg.use_orientation_scale_filter:
            keep &= orientation_scale_consistency(f1, f2, pairs, cfg)
        if np.count_nonzero(keep) >= cfg.min_filter_matches:
            pairs = pairs[keep]

        if cfg.use_edge_orientation_filter:
            edge_keep = edge_orientation_consistency(f1, f2, pairs, cfg)
            if np.count_nonzero(edge_keep) >= cfg.min_filter_matches:
                pairs = pairs[edge_keep]

        p1 = f1.points[pairs[:, 0]]
        p2 = f2.points[pairs[:, 1]]
        if cfg.use_grid_motion_filter:
            motion_keep = local_grid_motion_consistency(p1, p2, gray1.shape[:2], cfg)
            if np.count_nonzero(motion_keep) >= cfg.min_filter_matches:
                pairs = pairs[motion_keep]
                p1, p2 = p1[motion_keep], p2[motion_keep]

    if len(pairs) < cfg.min_pose_matches:
        result.status = "filtered_below_minimum"
        return result

    if cfg.use_lk_refinement:
        with StageTimer(result.timings_ms, "lk_refine_ms"):
            p1_ref, p2_ref, lk_keep, lk_diag = refine_matches_lk(gray1, gray2, p1, p2, cfg)
        result.diagnostics.update(lk_diag)
        if np.count_nonzero(lk_keep) >= cfg.min_filter_matches:
            p1, p2 = p1_ref[lk_keep], p2_ref[lk_keep]
        else:
            p1, p2 = p1_ref, p2_ref

    result.points1 = np.ascontiguousarray(p1, dtype=np.float32)
    result.points2 = np.ascontiguousarray(p2, dtype=np.float32)
    result.num_filtered_matches = len(result.points1)

    with StageTimer(result.timings_ms, "geometry_ms"):
        R_est, t_est, mask, diag = estimate_pose(result.points1, result.points2, camera, cfg)
    result.diagnostics.update(diag)
    result.R, result.t, result.geometric_inlier_mask = R_est, t_est, mask
    result.num_geometric_inliers = int(np.count_nonzero(mask))
    result.success = R_est is not None and result.num_geometric_inliers >= cfg.min_pose_matches
    result.status = "ok" if result.success else str(diag.get("pose_status", "failed"))
    return result



def run_edgefusion_simple(
    gray1: np.ndarray,
    gray2: np.ndarray,
    camera: CameraModel,
    cfg: PipelineConfig,
) -> MethodResult:
    """Run the lightweight EdgeFusion-Simple proposed variant.

    Design goal: stay close to ORB-Lowe latency while retaining a small amount
    of structural guidance. It intentionally uses only simple operations:
    ORB, Canny edge proximity, spatial quotas, one-way Lowe filtering,
    orientation/scale consistency, and USAC-MAGSAC. No Shi--Tomasi fusion,
    bidirectional matching, edge-orientation filter, motion grid, or LK is used.
    """
    method = "edgefusion_simple"
    result = MethodResult(method=method, success=False, status="started")

    with StageTimer(result.timings_ms, "extract_ms"):
        f1 = extract_simple_edge_orb(gray1, cfg)
        f2 = extract_simple_edge_orb(gray2, cfg)

    result.num_keypoints1 = len(f1.keypoints)
    result.num_keypoints2 = len(f2.keypoints)
    result.diagnostics.update(
        profile="simple",
        feature_fraction=float(cfg.simple_feature_fraction),
        edge_weight=float(cfg.simple_edge_weight),
        structural_operations="canny_proximity+spatial_quota",
    )

    with StageTimer(result.timings_ms, "match_ms"):
        pairs, distances, ratios = oneway_ratio_matches(
            f1, f2, float(cfg.ratio)
        )
    result.num_raw_matches = len(pairs)
    if len(distances):
        result.diagnostics["median_descriptor_distance"] = float(
            np.median(distances)
        )
    if len(ratios):
        result.diagnostics["median_ratio"] = float(np.median(ratios))

    if len(pairs) < cfg.min_pose_matches:
        result.status = "insufficient_matches"
        return result

    with StageTimer(result.timings_ms, "filter_ms"):
        keep = orientation_scale_consistency(f1, f2, pairs, cfg)
        if np.count_nonzero(keep) >= cfg.min_filter_matches:
            pairs = pairs[keep]

    if len(pairs) < cfg.min_pose_matches:
        result.status = "filtered_below_minimum"
        return result

    result.points1 = np.ascontiguousarray(
        f1.points[pairs[:, 0]], dtype=np.float32
    )
    result.points2 = np.ascontiguousarray(
        f2.points[pairs[:, 1]], dtype=np.float32
    )
    result.num_filtered_matches = len(result.points1)

    with StageTimer(result.timings_ms, "geometry_ms"):
        R_est, t_est, mask, diag = estimate_pose(
            result.points1, result.points2, camera, cfg
        )
    result.diagnostics.update(diag)
    result.R = R_est
    result.t = t_est
    result.geometric_inlier_mask = mask
    result.num_geometric_inliers = int(np.count_nonzero(mask))
    result.success = (
        R_est is not None
        and result.num_geometric_inliers >= cfg.min_pose_matches
    )
    result.status = (
        "ok" if result.success else str(diag.get("pose_status", "failed"))
    )
    return result


def run_edgefusion_adaptive(
    gray1: np.ndarray,
    gray2: np.ndarray,
    camera: CameraModel,
    cfg: PipelineConfig,
) -> MethodResult:
    """Run the conditional EdgeFusion-Adaptive cascade.

    Stage A is deliberately equivalent in spirit to ORB-Lowe: standard ORB
    extraction, one-way Lowe filtering, and the common pose backend. If that
    pose has enough inliers, purity, and spatial coverage, it is accepted
    immediately.

    Only uncertain pairs enter Stage B. The fallback reuses the Stage-A ORB
    features/descriptors, attaches edge/gradient evidence, performs the
    EdgeFusion soft re-ranking + bidirectional adaptive filtering, applies
    robust orientation/scale and edge-orientation checks, refines at most a
    limited number of spatially balanced matches with LK, and re-estimates the
    pose. The expensive local-motion grid and Shi--Tomasi augmentation are not
    used in this efficiency-oriented variant.
    """
    method = "edgefusion_adaptive"
    result = MethodResult(method=method, success=False, status="started")

    # -------------------- Stage A: fast ORB-Lowe pose --------------------
    with StageTimer(result.timings_ms, "fast_extract_ms"):
        f1 = extract_standard(gray1, "orb", cfg.nfeatures)
        f2 = extract_standard(gray2, "orb", cfg.nfeatures)
    result.num_keypoints1 = len(f1.keypoints)
    result.num_keypoints2 = len(f2.keypoints)

    with StageTimer(result.timings_ms, "fast_match_ms"):
        fast_pairs, fast_distances, fast_ratios = oneway_ratio_matches(
            f1, f2, float(cfg.ratio)
        )

    fast_p1 = np.empty((0, 2), dtype=np.float32)
    fast_p2 = np.empty((0, 2), dtype=np.float32)
    fast_R: Optional[np.ndarray] = None
    fast_t: Optional[np.ndarray] = None
    fast_mask = np.empty(0, dtype=bool)
    fast_diag: Dict[str, Any] = {}
    fast_success = False

    if len(fast_pairs) >= cfg.min_pose_matches:
        fast_p1 = np.ascontiguousarray(
            f1.points[fast_pairs[:, 0]], dtype=np.float32
        )
        fast_p2 = np.ascontiguousarray(
            f2.points[fast_pairs[:, 1]], dtype=np.float32
        )
        with StageTimer(result.timings_ms, "fast_geometry_ms"):
            fast_R, fast_t, fast_mask, fast_diag = estimate_pose(
                fast_p1, fast_p2, camera, cfg
            )
        fast_success = bool(
            fast_R is not None
            and np.count_nonzero(fast_mask) >= cfg.min_pose_matches
        )

    accepted, conf_diag = _adaptive_pose_confidence(
        fast_success,
        fast_p1,
        fast_p2,
        fast_mask,
        gray1.shape[:2],
        cfg,
    )
    result.diagnostics.update(conf_diag)
    result.diagnostics.update(
        adaptive_accept_inliers=int(cfg.adaptive_accept_inliers),
        adaptive_accept_inlier_ratio=float(
            cfg.adaptive_accept_inlier_ratio
        ),
        adaptive_accept_coverage=float(cfg.adaptive_accept_coverage),
        adaptive_fallback_feature_fraction=float(
            cfg.adaptive_fallback_feature_fraction
        ),
        adaptive_lk_max_matches=int(cfg.adaptive_lk_max_matches),
        fast_raw_matches=int(len(fast_pairs)),
    )
    if len(fast_distances):
        result.diagnostics["fast_median_descriptor_distance"] = float(
            np.median(fast_distances)
        )
    if len(fast_ratios):
        result.diagnostics["fast_median_ratio"] = float(
            np.median(fast_ratios)
        )

    if accepted:
        result.points1 = fast_p1
        result.points2 = fast_p2
        result.num_raw_matches = len(fast_pairs)
        result.num_filtered_matches = len(fast_p1)
        result.R = fast_R
        result.t = fast_t
        result.geometric_inlier_mask = fast_mask
        result.num_geometric_inliers = int(np.count_nonzero(fast_mask))
        result.success = True
        result.status = "ok_fast_accept"
        result.diagnostics.update(fast_diag)
        result.diagnostics["adaptive_path"] = "fast_accept"
        return result

    # -------------------- Stage B: conditional refinement ----------------
    # Reuse the standard ORB features/descriptors; only structural evidence is
    # computed now.
    with StageTimer(result.timings_ms, "structural_support_ms"):
        sf1 = _attach_structural_fields(gray1, f1, cfg)
        sf2 = _attach_structural_fields(gray2, f2, cfg)

        fallback_budget = max(
            cfg.min_filter_matches,
            int(
                round(
                    cfg.nfeatures
                    * float(cfg.adaptive_fallback_feature_fraction)
                )
            ),
        )
        fallback_budget = min(cfg.nfeatures, fallback_budget)
        sf1 = _edge_rerank_existing_features(
            sf1, gray1.shape[:2], fallback_budget, cfg
        )
        sf2 = _edge_rerank_existing_features(
            sf2, gray2.shape[:2], fallback_budget, cfg
        )

    with StageTimer(result.timings_ms, "refine_match_ms"):
        pairs, distances, ratios = mutual_ratio_matches(
            sf1,
            sf2,
            fixed_ratio=cfg.ratio,
            adaptive=True,
            cfg=cfg,
        )

    fallback_raw = int(len(pairs))
    if len(pairs) >= cfg.min_pose_matches:
        with StageTimer(result.timings_ms, "refine_filter_ms"):
            keep = orientation_scale_consistency(sf1, sf2, pairs, cfg)
            if np.count_nonzero(keep) >= cfg.min_filter_matches:
                pairs = pairs[keep]

            edge_keep = edge_orientation_consistency(sf1, sf2, pairs, cfg)
            if np.count_nonzero(edge_keep) >= cfg.min_filter_matches:
                pairs = pairs[edge_keep]

        if len(pairs) >= cfg.min_pose_matches:
            # Select a limited, spatially balanced subset for LK. This caps the
            # most expensive fallback stage while preserving image coverage.
            max_lk = int(cfg.adaptive_lk_max_matches)
            if max_lk > 0 and len(pairs) > max_lk:
                support = np.minimum(
                    sf1.edge_confidence[pairs[:, 0]],
                    sf2.edge_confidence[pairs[:, 1]],
                ).astype(np.float32)
                choose = grid_balanced_indices(
                    sf1.points[pairs[:, 0]],
                    support,
                    gray1.shape[:2],
                    max_lk,
                    cfg.grid_rows,
                    cfg.grid_cols,
                )
                pairs = pairs[choose]

            p1 = np.ascontiguousarray(
                sf1.points[pairs[:, 0]], dtype=np.float32
            )
            p2 = np.ascontiguousarray(
                sf2.points[pairs[:, 1]], dtype=np.float32
            )

            if cfg.use_lk_refinement and len(p1) >= cfg.min_filter_matches:
                with StageTimer(result.timings_ms, "lk_refine_ms"):
                    p1_ref, p2_ref, lk_keep, lk_diag = refine_matches_lk(
                        gray1, gray2, p1, p2, cfg
                    )
                result.diagnostics.update(lk_diag)
                if np.count_nonzero(lk_keep) >= cfg.min_filter_matches:
                    p1 = np.ascontiguousarray(
                        p1_ref[lk_keep], dtype=np.float32
                    )
                    p2 = np.ascontiguousarray(
                        p2_ref[lk_keep], dtype=np.float32
                    )
                else:
                    p1 = np.ascontiguousarray(p1_ref, dtype=np.float32)
                    p2 = np.ascontiguousarray(p2_ref, dtype=np.float32)

            if len(p1) >= cfg.min_pose_matches:
                with StageTimer(result.timings_ms, "refine_geometry_ms"):
                    R_est, t_est, mask, diag = estimate_pose(
                        p1, p2, camera, cfg
                    )
                fallback_success = bool(
                    R_est is not None
                    and np.count_nonzero(mask) >= cfg.min_pose_matches
                )
                if fallback_success:
                    result.points1 = p1
                    result.points2 = p2
                    result.num_raw_matches = fallback_raw
                    result.num_filtered_matches = len(p1)
                    result.R = R_est
                    result.t = t_est
                    result.geometric_inlier_mask = mask
                    result.num_geometric_inliers = int(
                        np.count_nonzero(mask)
                    )
                    result.success = True
                    result.status = "ok_refined"
                    result.diagnostics.update(diag)
                    result.diagnostics["adaptive_path"] = "refined"
                    result.diagnostics[
                        "fallback_filtered_matches"
                    ] = int(len(p1))
                    if len(distances):
                        result.diagnostics[
                            "fallback_median_descriptor_distance"
                        ] = float(np.median(distances))
                    if len(ratios):
                        result.diagnostics[
                            "fallback_median_ratio"
                        ] = float(np.median(ratios))
                    return result

    # If refinement fails but the fast stage produced a valid low-confidence
    # pose, retain it rather than converting a recoverable pair into failure.
    if fast_success:
        result.points1 = fast_p1
        result.points2 = fast_p2
        result.num_raw_matches = len(fast_pairs)
        result.num_filtered_matches = len(fast_p1)
        result.R = fast_R
        result.t = fast_t
        result.geometric_inlier_mask = fast_mask
        result.num_geometric_inliers = int(np.count_nonzero(fast_mask))
        result.success = True
        result.status = "ok_fast_fallback"
        result.diagnostics.update(fast_diag)
        result.diagnostics["adaptive_path"] = "fallback_failed_use_fast"
        result.diagnostics["fallback_raw_matches"] = fallback_raw
        return result

    result.num_raw_matches = fallback_raw
    result.num_filtered_matches = 0
    result.status = "adaptive_pose_failed"
    result.diagnostics["adaptive_path"] = "failed"
    return result




def run_edgefusion_adaptive_guarded(
    gray1: np.ndarray,
    gray2: np.ndarray,
    camera: CameraModel,
    cfg: PipelineConfig,
) -> MethodResult:
    """Run EdgeFusion-Adaptive-Guarded.

    This method keeps ``edgefusion_adaptive`` unchanged and adds a conservative
    decision layer around its refinement path.

    1. ORB-Lowe + pose is computed first.
    2. Hard high-confidence poses are accepted as before.
    3. Borderline fast poses can also exit through a strict soft-accept rule.
    4. Uncertain pairs receive structural re-ranking/filtering using the
       already-computed ORB features.
    5. A pre-LK structural pose is estimated. If it clearly improves shared
       geometric quality, LK is skipped.
    6. Otherwise selective LK is applied and the refined pose is estimated.
    7. Fast, pre-LK, and refined candidates are compared on the SAME original
       ORB-Lowe correspondences. A new candidate replaces the current one only
       when its shared quality improves by the configured margin.

    Model selection uses no benchmark ground truth.
    """
    method = "edgefusion_adaptive_guarded"
    result = MethodResult(method=method, success=False, status="started")

    # -------------------- Stage A: fast ORB-Lowe -------------------------
    with StageTimer(result.timings_ms, "fast_extract_ms"):
        f1 = extract_standard(gray1, "orb", cfg.nfeatures)
        f2 = extract_standard(gray2, "orb", cfg.nfeatures)
    result.num_keypoints1 = len(f1.keypoints)
    result.num_keypoints2 = len(f2.keypoints)

    with StageTimer(result.timings_ms, "fast_match_ms"):
        fast_pairs, fast_distances, fast_ratios = oneway_ratio_matches(
            f1, f2, float(cfg.ratio)
        )

    fast_p1 = np.empty((0, 2), dtype=np.float32)
    fast_p2 = np.empty((0, 2), dtype=np.float32)
    fast_R: Optional[np.ndarray] = None
    fast_t: Optional[np.ndarray] = None
    fast_mask = np.empty(0, dtype=bool)
    fast_diag: Dict[str, Any] = {}
    fast_success = False

    if len(fast_pairs) >= cfg.min_pose_matches:
        fast_p1 = np.ascontiguousarray(
            f1.points[fast_pairs[:, 0]], dtype=np.float32
        )
        fast_p2 = np.ascontiguousarray(
            f2.points[fast_pairs[:, 1]], dtype=np.float32
        )
        with StageTimer(result.timings_ms, "fast_geometry_ms"):
            fast_R, fast_t, fast_mask, fast_diag = estimate_pose(
                fast_p1, fast_p2, camera, cfg
            )
        fast_success = bool(
            fast_R is not None
            and np.count_nonzero(fast_mask) >= cfg.min_pose_matches
        )

    hard_accept, conf_diag = _adaptive_pose_confidence(
        fast_success,
        fast_p1,
        fast_p2,
        fast_mask,
        gray1.shape[:2],
        cfg,
    )
    soft_accept = _adaptive_guarded_soft_accept(
        fast_success, conf_diag, cfg
    )

    result.diagnostics.update(conf_diag)
    result.diagnostics.update(
        adaptive_guarded_soft_score=float(
            cfg.adaptive_guarded_soft_score
        ),
        adaptive_guarded_soft_floor_fraction=float(
            cfg.adaptive_guarded_soft_floor_fraction
        ),
        adaptive_guarded_quality_margin=float(
            cfg.adaptive_guarded_quality_margin
        ),
        adaptive_guarded_prelk_margin=float(
            cfg.adaptive_guarded_prelk_margin
        ),
        fast_raw_matches=int(len(fast_pairs)),
    )
    if len(fast_distances):
        result.diagnostics["fast_median_descriptor_distance"] = float(
            np.median(fast_distances)
        )
    if len(fast_ratios):
        result.diagnostics["fast_median_ratio"] = float(
            np.median(fast_ratios)
        )

    def finalize(
        p1: np.ndarray,
        p2: np.ndarray,
        R_est: Optional[np.ndarray],
        t_est: Optional[np.ndarray],
        mask: np.ndarray,
        status: str,
        path: str,
        raw_matches: int,
        diag: Optional[Mapping[str, Any]] = None,
    ) -> MethodResult:
        result.points1 = np.ascontiguousarray(p1, dtype=np.float32)
        result.points2 = np.ascontiguousarray(p2, dtype=np.float32)
        result.num_raw_matches = int(raw_matches)
        result.num_filtered_matches = int(len(result.points1))
        result.R = R_est
        result.t = t_est
        result.geometric_inlier_mask = np.asarray(mask, dtype=bool)
        result.num_geometric_inliers = int(
            np.count_nonzero(result.geometric_inlier_mask)
        )
        result.success = bool(
            R_est is not None
            and t_est is not None
            and result.num_geometric_inliers >= cfg.min_pose_matches
        )
        result.status = status if result.success else "adaptive_guarded_failed"
        result.diagnostics["adaptive_guarded_path"] = path
        if diag:
            result.diagnostics.update(dict(diag))
        return result

    if hard_accept:
        result.diagnostics["adaptive_guarded_fast_accept_type"] = "hard"
        return finalize(
            fast_p1, fast_p2, fast_R, fast_t, fast_mask,
            "ok_fast_hard_accept", "fast_hard_accept",
            len(fast_pairs), fast_diag,
        )

    if soft_accept:
        result.diagnostics["adaptive_guarded_fast_accept_type"] = "soft"
        return finalize(
            fast_p1, fast_p2, fast_R, fast_t, fast_mask,
            "ok_fast_soft_accept", "fast_soft_accept",
            len(fast_pairs), fast_diag,
        )

    # Shared fast quality is the reference for any later replacement.
    fast_shared = _pose_shared_quality(
        fast_R, fast_t, fast_p1, fast_p2, camera, cfg, gray1.shape[:2]
    )
    for key, value in fast_shared.items():
        result.diagnostics[f"guarded_fast_{key}"] = value

    best_name = "fast" if fast_success else "none"
    best_p1, best_p2 = fast_p1, fast_p2
    best_R, best_t, best_mask = fast_R, fast_t, fast_mask
    best_diag: Dict[str, Any] = dict(fast_diag)
    best_score = (
        float(fast_shared["shared_quality_score"])
        if fast_success else -float("inf")
    )
    best_raw_matches = len(fast_pairs)

    # -------------------- Stage B: reused structural support -------------
    with StageTimer(result.timings_ms, "structural_support_ms"):
        sf1 = _attach_structural_fields(gray1, f1, cfg)
        sf2 = _attach_structural_fields(gray2, f2, cfg)
        fallback_budget = max(
            cfg.min_filter_matches,
            int(
                round(
                    cfg.nfeatures
                    * float(cfg.adaptive_fallback_feature_fraction)
                )
            ),
        )
        fallback_budget = min(cfg.nfeatures, fallback_budget)
        sf1 = _edge_rerank_existing_features(
            sf1, gray1.shape[:2], fallback_budget, cfg
        )
        sf2 = _edge_rerank_existing_features(
            sf2, gray2.shape[:2], fallback_budget, cfg
        )

    with StageTimer(result.timings_ms, "refine_match_ms"):
        pairs, distances, ratios = mutual_ratio_matches(
            sf1, sf2, fixed_ratio=cfg.ratio, adaptive=True, cfg=cfg
        )
    fallback_raw = int(len(pairs))
    result.diagnostics["fallback_raw_matches"] = fallback_raw

    if len(pairs) < cfg.min_pose_matches:
        if fast_success:
            return finalize(
                fast_p1, fast_p2, fast_R, fast_t, fast_mask,
                "ok_fast_guarded", "fallback_insufficient_use_fast",
                len(fast_pairs), fast_diag,
            )
        result.status = "adaptive_guarded_pose_failed"
        result.diagnostics["adaptive_guarded_path"] = "failed_no_fallback_matches"
        return result

    with StageTimer(result.timings_ms, "refine_filter_ms"):
        keep = orientation_scale_consistency(sf1, sf2, pairs, cfg)
        if np.count_nonzero(keep) >= cfg.min_filter_matches:
            pairs = pairs[keep]
        edge_keep = edge_orientation_consistency(sf1, sf2, pairs, cfg)
        if np.count_nonzero(edge_keep) >= cfg.min_filter_matches:
            pairs = pairs[edge_keep]

    if len(pairs) < cfg.min_pose_matches:
        if fast_success:
            return finalize(
                fast_p1, fast_p2, fast_R, fast_t, fast_mask,
                "ok_fast_guarded", "fallback_filtered_use_fast",
                len(fast_pairs), fast_diag,
            )
        result.status = "adaptive_guarded_pose_failed"
        result.diagnostics["adaptive_guarded_path"] = "failed_after_filter"
        return result

    # Limit the structural candidate set before geometry/LK.
    max_lk = int(cfg.adaptive_lk_max_matches)
    if max_lk > 0 and len(pairs) > max_lk:
        support = np.minimum(
            sf1.edge_confidence[pairs[:, 0]],
            sf2.edge_confidence[pairs[:, 1]],
        ).astype(np.float32)
        choose = grid_balanced_indices(
            sf1.points[pairs[:, 0]],
            support,
            gray1.shape[:2],
            max_lk,
            cfg.grid_rows,
            cfg.grid_cols,
        )
        pairs = pairs[choose]

    pre_p1 = np.ascontiguousarray(
        sf1.points[pairs[:, 0]], dtype=np.float32
    )
    pre_p2 = np.ascontiguousarray(
        sf2.points[pairs[:, 1]], dtype=np.float32
    )

    # -------------------- Stage B1: pre-LK pose --------------------------
    pre_R: Optional[np.ndarray] = None
    pre_t: Optional[np.ndarray] = None
    pre_mask = np.empty(0, dtype=bool)
    pre_diag: Dict[str, Any] = {}
    pre_success = False
    with StageTimer(result.timings_ms, "prelk_geometry_ms"):
        pre_R, pre_t, pre_mask, pre_diag = estimate_pose(
            pre_p1, pre_p2, camera, cfg
        )
    pre_success = bool(
        pre_R is not None
        and np.count_nonzero(pre_mask) >= cfg.min_pose_matches
    )

    pre_shared = _pose_shared_quality(
        pre_R, pre_t, fast_p1, fast_p2, camera, cfg, gray1.shape[:2]
    )
    for key, value in pre_shared.items():
        result.diagnostics[f"guarded_prelk_{key}"] = value

    pre_score = float(pre_shared["shared_quality_score"])
    if pre_success and (
        not fast_success
        or pre_score
        >= best_score + float(cfg.adaptive_guarded_prelk_margin)
    ):
        best_name = "prelk"
        best_p1, best_p2 = pre_p1, pre_p2
        best_R, best_t, best_mask = pre_R, pre_t, pre_mask
        best_diag = dict(pre_diag)
        best_score = pre_score
        best_raw_matches = fallback_raw

        # A clear pre-LK improvement is an explicit early exit: no optical flow.
        result.diagnostics["guarded_selected_quality_score"] = best_score
        return finalize(
            best_p1, best_p2, best_R, best_t, best_mask,
            "ok_prelk_selected", "prelk_early_accept",
            best_raw_matches, best_diag,
        )

    # Keep pre-LK as a candidate even when its improvement is smaller than the
    # early-exit threshold. Final selection after LK uses the smaller general
    # quality margin.
    candidate_pre = pre_success

    # -------------------- Stage B2: selective LK -------------------------
    lk_p1, lk_p2 = pre_p1, pre_p2
    if cfg.use_lk_refinement and len(pre_p1) >= cfg.min_filter_matches:
        with StageTimer(result.timings_ms, "lk_refine_ms"):
            p1_ref, p2_ref, lk_keep, lk_diag = refine_matches_lk(
                gray1, gray2, pre_p1, pre_p2, cfg
            )
        result.diagnostics.update(lk_diag)
        if np.count_nonzero(lk_keep) >= cfg.min_filter_matches:
            lk_p1 = np.ascontiguousarray(
                p1_ref[lk_keep], dtype=np.float32
            )
            lk_p2 = np.ascontiguousarray(
                p2_ref[lk_keep], dtype=np.float32
            )
        else:
            lk_p1 = np.ascontiguousarray(p1_ref, dtype=np.float32)
            lk_p2 = np.ascontiguousarray(p2_ref, dtype=np.float32)

    ref_R: Optional[np.ndarray] = None
    ref_t: Optional[np.ndarray] = None
    ref_mask = np.empty(0, dtype=bool)
    ref_diag: Dict[str, Any] = {}
    ref_success = False

    if len(lk_p1) >= cfg.min_pose_matches:
        with StageTimer(result.timings_ms, "refine_geometry_ms"):
            ref_R, ref_t, ref_mask, ref_diag = estimate_pose(
                lk_p1, lk_p2, camera, cfg
            )
        ref_success = bool(
            ref_R is not None
            and np.count_nonzero(ref_mask) >= cfg.min_pose_matches
        )

    ref_shared = _pose_shared_quality(
        ref_R, ref_t, fast_p1, fast_p2, camera, cfg, gray1.shape[:2]
    )
    for key, value in ref_shared.items():
        result.diagnostics[f"guarded_refined_{key}"] = value

    margin = float(cfg.adaptive_guarded_quality_margin)

    # Candidate pool is compared without ground truth and on the same fast
    # correspondences. Start from fast (if valid), then permit replacement
    # only by a sufficient geometric-quality improvement.
    if candidate_pre:
        pre_score = float(pre_shared["shared_quality_score"])
        if (
            best_name == "none"
            or pre_score >= best_score + margin
        ):
            best_name = "prelk"
            best_p1, best_p2 = pre_p1, pre_p2
            best_R, best_t, best_mask = pre_R, pre_t, pre_mask
            best_diag = dict(pre_diag)
            best_score = pre_score
            best_raw_matches = fallback_raw

    if ref_success:
        ref_score = float(ref_shared["shared_quality_score"])
        if (
            best_name == "none"
            or ref_score >= best_score + margin
        ):
            best_name = "refined"
            best_p1, best_p2 = lk_p1, lk_p2
            best_R, best_t, best_mask = ref_R, ref_t, ref_mask
            best_diag = dict(ref_diag)
            best_score = ref_score
            best_raw_matches = fallback_raw

    result.diagnostics["guarded_selected_candidate"] = best_name
    result.diagnostics["guarded_selected_quality_score"] = float(
        best_score if math.isfinite(best_score) else -1.0
    )
    result.diagnostics["guarded_refinement_replaced_fast"] = float(
        best_name in {"prelk", "refined"}
    )

    if best_name != "none":
        status = {
            "fast": "ok_fast_guarded",
            "prelk": "ok_prelk_selected",
            "refined": "ok_refined_selected",
        }[best_name]
        path = {
            "fast": "quality_gate_kept_fast",
            "prelk": "quality_gate_selected_prelk",
            "refined": "quality_gate_selected_refined",
        }[best_name]
        return finalize(
            best_p1, best_p2, best_R, best_t, best_mask,
            status, path, best_raw_matches, best_diag,
        )

    result.status = "adaptive_guarded_pose_failed"
    result.diagnostics["adaptive_guarded_path"] = "failed"
    return result




def run_edgefusion_adaptive_poseguard(
    gray1: np.ndarray,
    gray2: np.ndarray,
    camera: CameraModel,
    cfg: PipelineConfig,
) -> MethodResult:
    """Run EdgeFusion-Adaptive-PoseGuard (Guarded-v2).

    This method keeps both earlier Adaptive variants unchanged and replaces
    the first Guarded score with pose-observability evidence: shared
    cheirality support, spatial coverage, epipolar fit, and parallax.

    1. ORB-Lowe + pose is computed first.
    2. Hard high-confidence poses are accepted as before.
    3. Uncertain pairs receive structural re-ranking/filtering using the
       already-computed ORB features.
    4. A pre-LK structural pose is estimated and evaluated for positive-depth
       support and parallax; a clear rescue can skip LK.
    5. Otherwise selective LK is applied and the refined pose is estimated.
    6. Fast, pre-LK, and refined candidates are compared on the SAME original
       ORB-Lowe correspondences using pose observability, not ground truth.

    Model selection uses no benchmark ground truth.
    """
    method = "edgefusion_adaptive_poseguard"
    result = MethodResult(method=method, success=False, status="started")

    # -------------------- Stage A: fast ORB-Lowe -------------------------
    with StageTimer(result.timings_ms, "fast_extract_ms"):
        f1 = extract_standard(gray1, "orb", cfg.nfeatures)
        f2 = extract_standard(gray2, "orb", cfg.nfeatures)
    result.num_keypoints1 = len(f1.keypoints)
    result.num_keypoints2 = len(f2.keypoints)

    with StageTimer(result.timings_ms, "fast_match_ms"):
        fast_pairs, fast_distances, fast_ratios = oneway_ratio_matches(
            f1, f2, float(cfg.ratio)
        )

    fast_p1 = np.empty((0, 2), dtype=np.float32)
    fast_p2 = np.empty((0, 2), dtype=np.float32)
    fast_R: Optional[np.ndarray] = None
    fast_t: Optional[np.ndarray] = None
    fast_mask = np.empty(0, dtype=bool)
    fast_diag: Dict[str, Any] = {}
    fast_success = False

    if len(fast_pairs) >= cfg.min_pose_matches:
        fast_p1 = np.ascontiguousarray(
            f1.points[fast_pairs[:, 0]], dtype=np.float32
        )
        fast_p2 = np.ascontiguousarray(
            f2.points[fast_pairs[:, 1]], dtype=np.float32
        )
        with StageTimer(result.timings_ms, "fast_geometry_ms"):
            fast_R, fast_t, fast_mask, fast_diag = estimate_pose(
                fast_p1, fast_p2, camera, cfg
            )
        fast_success = bool(
            fast_R is not None
            and np.count_nonzero(fast_mask) >= cfg.min_pose_matches
        )

    hard_accept, conf_diag = _adaptive_pose_confidence(
        fast_success,
        fast_p1,
        fast_p2,
        fast_mask,
        gray1.shape[:2],
        cfg,
    )
    result.diagnostics.update(conf_diag)
    result.diagnostics.update(
        adaptive_poseguard_quality_margin=float(
            cfg.adaptive_poseguard_quality_margin
        ),
        adaptive_poseguard_prelk_margin=float(
            cfg.adaptive_poseguard_prelk_margin
        ),
        adaptive_poseguard_parallax_target_deg=float(
            cfg.adaptive_poseguard_parallax_target_deg
        ),
        adaptive_poseguard_rescue_fast_cheirality_max=float(
            cfg.adaptive_poseguard_rescue_fast_cheirality_max
        ),
        adaptive_poseguard_rescue_candidate_cheirality_min=float(
            cfg.adaptive_poseguard_rescue_candidate_cheirality_min
        ),
        adaptive_poseguard_rescue_cheirality_gain=float(
            cfg.adaptive_poseguard_rescue_cheirality_gain
        ),
        fast_raw_matches=int(len(fast_pairs)),
    )
    if len(fast_distances):
        result.diagnostics["fast_median_descriptor_distance"] = float(
            np.median(fast_distances)
        )
    if len(fast_ratios):
        result.diagnostics["fast_median_ratio"] = float(
            np.median(fast_ratios)
        )

    def finalize(
        p1: np.ndarray,
        p2: np.ndarray,
        R_est: Optional[np.ndarray],
        t_est: Optional[np.ndarray],
        mask: np.ndarray,
        status: str,
        path: str,
        raw_matches: int,
        diag: Optional[Mapping[str, Any]] = None,
    ) -> MethodResult:
        result.points1 = np.ascontiguousarray(p1, dtype=np.float32)
        result.points2 = np.ascontiguousarray(p2, dtype=np.float32)
        result.num_raw_matches = int(raw_matches)
        result.num_filtered_matches = int(len(result.points1))
        result.R = R_est
        result.t = t_est
        result.geometric_inlier_mask = np.asarray(mask, dtype=bool)
        result.num_geometric_inliers = int(
            np.count_nonzero(result.geometric_inlier_mask)
        )
        result.success = bool(
            R_est is not None
            and t_est is not None
            and result.num_geometric_inliers >= cfg.min_pose_matches
        )
        result.status = status if result.success else "adaptive_poseguard_failed"
        result.diagnostics["adaptive_poseguard_path"] = path
        if diag:
            result.diagnostics.update(dict(diag))
        return result

    if hard_accept:
        result.diagnostics["adaptive_poseguard_fast_accept_type"] = "hard"
        return finalize(
            fast_p1, fast_p2, fast_R, fast_t, fast_mask,
            "ok_fast_hard_accept", "fast_hard_accept",
            len(fast_pairs), fast_diag,
        )

    # Shared fast quality is the reference for any later replacement.
    fast_shared = _poseguard_shared_quality(
        fast_R, fast_t, fast_p1, fast_p2, camera, cfg, gray1.shape[:2]
    )
    fast_shared["poseguard_native_cheirality_ratio"] = float(
        np.count_nonzero(fast_mask) / max(len(fast_p1), 1)
    ) if fast_success else 0.0
    for key, value in fast_shared.items():
        suffix = key[10:] if key.startswith("poseguard_") else key
        result.diagnostics[f"poseguard_fast_{suffix}"] = value

    best_name = "fast" if fast_success else "none"
    best_p1, best_p2 = fast_p1, fast_p2
    best_R, best_t, best_mask = fast_R, fast_t, fast_mask
    best_diag: Dict[str, Any] = dict(fast_diag)
    best_score = (
        float(fast_shared["poseguard_quality_score"])
        if fast_success else -float("inf")
    )
    best_raw_matches = len(fast_pairs)

    # -------------------- Stage B: reused structural support -------------
    with StageTimer(result.timings_ms, "structural_support_ms"):
        sf1 = _attach_structural_fields(gray1, f1, cfg)
        sf2 = _attach_structural_fields(gray2, f2, cfg)
        fallback_budget = max(
            cfg.min_filter_matches,
            int(
                round(
                    cfg.nfeatures
                    * float(cfg.adaptive_fallback_feature_fraction)
                )
            ),
        )
        fallback_budget = min(cfg.nfeatures, fallback_budget)
        sf1 = _edge_rerank_existing_features(
            sf1, gray1.shape[:2], fallback_budget, cfg
        )
        sf2 = _edge_rerank_existing_features(
            sf2, gray2.shape[:2], fallback_budget, cfg
        )

    with StageTimer(result.timings_ms, "refine_match_ms"):
        pairs, distances, ratios = mutual_ratio_matches(
            sf1, sf2, fixed_ratio=cfg.ratio, adaptive=True, cfg=cfg
        )
    fallback_raw = int(len(pairs))
    result.diagnostics["fallback_raw_matches"] = fallback_raw

    if len(pairs) < cfg.min_pose_matches:
        if fast_success:
            return finalize(
                fast_p1, fast_p2, fast_R, fast_t, fast_mask,
                "ok_fast_poseguard", "fallback_insufficient_use_fast",
                len(fast_pairs), fast_diag,
            )
        result.status = "adaptive_poseguard_pose_failed"
        result.diagnostics["adaptive_poseguard_path"] = "failed_no_fallback_matches"
        return result

    with StageTimer(result.timings_ms, "refine_filter_ms"):
        keep = orientation_scale_consistency(sf1, sf2, pairs, cfg)
        if np.count_nonzero(keep) >= cfg.min_filter_matches:
            pairs = pairs[keep]
        edge_keep = edge_orientation_consistency(sf1, sf2, pairs, cfg)
        if np.count_nonzero(edge_keep) >= cfg.min_filter_matches:
            pairs = pairs[edge_keep]

    if len(pairs) < cfg.min_pose_matches:
        if fast_success:
            return finalize(
                fast_p1, fast_p2, fast_R, fast_t, fast_mask,
                "ok_fast_poseguard", "fallback_filtered_use_fast",
                len(fast_pairs), fast_diag,
            )
        result.status = "adaptive_poseguard_pose_failed"
        result.diagnostics["adaptive_poseguard_path"] = "failed_after_filter"
        return result

    # Limit the structural candidate set before geometry/LK.
    max_lk = int(cfg.adaptive_lk_max_matches)
    if max_lk > 0 and len(pairs) > max_lk:
        support = np.minimum(
            sf1.edge_confidence[pairs[:, 0]],
            sf2.edge_confidence[pairs[:, 1]],
        ).astype(np.float32)
        choose = grid_balanced_indices(
            sf1.points[pairs[:, 0]],
            support,
            gray1.shape[:2],
            max_lk,
            cfg.grid_rows,
            cfg.grid_cols,
        )
        pairs = pairs[choose]

    pre_p1 = np.ascontiguousarray(
        sf1.points[pairs[:, 0]], dtype=np.float32
    )
    pre_p2 = np.ascontiguousarray(
        sf2.points[pairs[:, 1]], dtype=np.float32
    )

    # -------------------- Stage B1: pre-LK pose --------------------------
    pre_R: Optional[np.ndarray] = None
    pre_t: Optional[np.ndarray] = None
    pre_mask = np.empty(0, dtype=bool)
    pre_diag: Dict[str, Any] = {}
    pre_success = False
    with StageTimer(result.timings_ms, "prelk_geometry_ms"):
        pre_R, pre_t, pre_mask, pre_diag = estimate_pose(
            pre_p1, pre_p2, camera, cfg
        )
    pre_success = bool(
        pre_R is not None
        and np.count_nonzero(pre_mask) >= cfg.min_pose_matches
    )

    pre_shared = _poseguard_shared_quality(
        pre_R, pre_t, fast_p1, fast_p2, camera, cfg, gray1.shape[:2]
    )
    pre_shared["poseguard_native_cheirality_ratio"] = float(
        np.count_nonzero(pre_mask) / max(len(pre_p1), 1)
    ) if pre_success else 0.0
    for key, value in pre_shared.items():
        suffix = key[10:] if key.startswith("poseguard_") else key
        result.diagnostics[f"poseguard_prelk_{suffix}"] = value

    pre_score = float(pre_shared["poseguard_quality_score"])
    pre_early = False
    pre_reason = "no_candidate"
    if pre_success:
        if not fast_success:
            pre_early, pre_reason = True, "fast_invalid"
        else:
            pre_early, pre_reason = _poseguard_should_replace(
                fast_shared, pre_shared, cfg, prelk=True
            )
    result.diagnostics["poseguard_prelk_decision"] = pre_reason
    if pre_early:
        best_name = "prelk"
        best_p1, best_p2 = pre_p1, pre_p2
        best_R, best_t, best_mask = pre_R, pre_t, pre_mask
        best_diag = dict(pre_diag)
        best_score = pre_score
        best_raw_matches = fallback_raw
        result.diagnostics["poseguard_selected_quality_score"] = best_score
        return finalize(
            best_p1, best_p2, best_R, best_t, best_mask,
            "ok_prelk_selected", "prelk_early_accept",
            best_raw_matches, best_diag,
        )

    # Keep pre-LK as a candidate even when its improvement is smaller than the
    # early-exit threshold. Final selection after LK uses the smaller general
    # quality margin.
    candidate_pre = pre_success

    # -------------------- Stage B2: selective LK -------------------------
    lk_p1, lk_p2 = pre_p1, pre_p2
    if cfg.use_lk_refinement and len(pre_p1) >= cfg.min_filter_matches:
        with StageTimer(result.timings_ms, "lk_refine_ms"):
            p1_ref, p2_ref, lk_keep, lk_diag = refine_matches_lk(
                gray1, gray2, pre_p1, pre_p2, cfg
            )
        result.diagnostics.update(lk_diag)
        if np.count_nonzero(lk_keep) >= cfg.min_filter_matches:
            lk_p1 = np.ascontiguousarray(
                p1_ref[lk_keep], dtype=np.float32
            )
            lk_p2 = np.ascontiguousarray(
                p2_ref[lk_keep], dtype=np.float32
            )
        else:
            lk_p1 = np.ascontiguousarray(p1_ref, dtype=np.float32)
            lk_p2 = np.ascontiguousarray(p2_ref, dtype=np.float32)

    ref_R: Optional[np.ndarray] = None
    ref_t: Optional[np.ndarray] = None
    ref_mask = np.empty(0, dtype=bool)
    ref_diag: Dict[str, Any] = {}
    ref_success = False

    if len(lk_p1) >= cfg.min_pose_matches:
        with StageTimer(result.timings_ms, "refine_geometry_ms"):
            ref_R, ref_t, ref_mask, ref_diag = estimate_pose(
                lk_p1, lk_p2, camera, cfg
            )
        ref_success = bool(
            ref_R is not None
            and np.count_nonzero(ref_mask) >= cfg.min_pose_matches
        )

    ref_shared = _poseguard_shared_quality(
        ref_R, ref_t, fast_p1, fast_p2, camera, cfg, gray1.shape[:2]
    )
    ref_shared["poseguard_native_cheirality_ratio"] = float(
        np.count_nonzero(ref_mask) / max(len(lk_p1), 1)
    ) if ref_success else 0.0
    for key, value in ref_shared.items():
        suffix = key[10:] if key.startswith("poseguard_") else key
        result.diagnostics[f"poseguard_refined_{suffix}"] = value

    # Final candidate selection uses pose observability. A candidate can replace
    # a weak fast pose through an explicit cheirality-rescue rule; otherwise it
    # must exceed the current quality by the configured margin without a large
    # loss of positive-depth support.
    best_quality = fast_shared if fast_success else {
        "poseguard_quality_score": 0.0,
        "poseguard_cheirality_ratio": 0.0,
    }

    if candidate_pre:
        pre_score = float(pre_shared["poseguard_quality_score"])
        if best_name == "none":
            replace_pre, pre_reason = True, "no_current_candidate"
        else:
            replace_pre, pre_reason = _poseguard_should_replace(
                best_quality, pre_shared, cfg, prelk=False
            )
        result.diagnostics["poseguard_final_prelk_decision"] = pre_reason
        if replace_pre:
            best_name = "prelk"
            best_p1, best_p2 = pre_p1, pre_p2
            best_R, best_t, best_mask = pre_R, pre_t, pre_mask
            best_diag = dict(pre_diag)
            best_score = pre_score
            best_raw_matches = fallback_raw
            best_quality = pre_shared

    if ref_success:
        ref_score = float(ref_shared["poseguard_quality_score"])
        if best_name == "none":
            replace_ref, ref_reason = True, "no_current_candidate"
        else:
            replace_ref, ref_reason = _poseguard_should_replace(
                best_quality, ref_shared, cfg, prelk=False
            )
        result.diagnostics["poseguard_final_refined_decision"] = ref_reason
        if replace_ref:
            best_name = "refined"
            best_p1, best_p2 = lk_p1, lk_p2
            best_R, best_t, best_mask = ref_R, ref_t, ref_mask
            best_diag = dict(ref_diag)
            best_score = ref_score
            best_raw_matches = fallback_raw
            best_quality = ref_shared

    result.diagnostics["poseguard_selected_candidate"] = best_name
    result.diagnostics["poseguard_selected_quality_score"] = float(
        best_score if math.isfinite(best_score) else -1.0
    )
    result.diagnostics["poseguard_refinement_replaced_fast"] = float(
        best_name in {"prelk", "refined"}
    )

    if best_name != "none":
        status = {
            "fast": "ok_fast_poseguard",
            "prelk": "ok_prelk_selected",
            "refined": "ok_refined_selected",
        }[best_name]
        path = {
            "fast": "quality_gate_kept_fast",
            "prelk": "quality_gate_selected_prelk",
            "refined": "quality_gate_selected_refined",
        }[best_name]
        return finalize(
            best_p1, best_p2, best_R, best_t, best_mask,
            status, path, best_raw_matches, best_diag,
        )

    result.status = "adaptive_poseguard_pose_failed"
    result.diagnostics["adaptive_poseguard_path"] = "failed"
    return result




def run_edgefusion_adaptive_poseguard_v3(
    gray1: np.ndarray,
    gray2: np.ndarray,
    camera: CameraModel,
    cfg: PipelineConfig,
) -> MethodResult:
    """Run EdgeFusion-Adaptive-PoseGuard-v3.

    This method keeps Adaptive, Guarded, and PoseGuard-v2 unchanged. It adds
    a separate observability gate before candidate ranking, a gain-based
    cheirality rescue rule, and a conservative early stop before LK.

    1. ORB-Lowe + pose is computed first.
    2. Hard high-confidence poses are accepted as before.
    3. Uncertain pairs receive structural re-ranking/filtering using the
       already-computed ORB features.
    4. A pre-LK structural pose is estimated and evaluated for positive-depth
       support and parallax; a clear rescue can skip LK.
    5. Otherwise selective LK is applied and the refined pose is estimated.
    6. A replacement candidate must first pass native-cheirality and parallax
       validity gates; only then is quality/rescue ranking considered.
    7. LK can be skipped when the pre-LK fallback is extremely weak and a valid
       fast pose already exists.
    8. Fast, pre-LK, and refined candidates are compared without ground truth.

    Model selection uses no benchmark ground truth.
    """
    method = "edgefusion_adaptive_poseguard_v3"
    result = MethodResult(method=method, success=False, status="started")

    # -------------------- Stage A: fast ORB-Lowe -------------------------
    with StageTimer(result.timings_ms, "fast_extract_ms"):
        f1 = extract_standard(gray1, "orb", cfg.nfeatures)
        f2 = extract_standard(gray2, "orb", cfg.nfeatures)
    result.num_keypoints1 = len(f1.keypoints)
    result.num_keypoints2 = len(f2.keypoints)

    with StageTimer(result.timings_ms, "fast_match_ms"):
        fast_pairs, fast_distances, fast_ratios = oneway_ratio_matches(
            f1, f2, float(cfg.ratio)
        )

    fast_p1 = np.empty((0, 2), dtype=np.float32)
    fast_p2 = np.empty((0, 2), dtype=np.float32)
    fast_R: Optional[np.ndarray] = None
    fast_t: Optional[np.ndarray] = None
    fast_mask = np.empty(0, dtype=bool)
    fast_diag: Dict[str, Any] = {}
    fast_success = False

    if len(fast_pairs) >= cfg.min_pose_matches:
        fast_p1 = np.ascontiguousarray(
            f1.points[fast_pairs[:, 0]], dtype=np.float32
        )
        fast_p2 = np.ascontiguousarray(
            f2.points[fast_pairs[:, 1]], dtype=np.float32
        )
        with StageTimer(result.timings_ms, "fast_geometry_ms"):
            fast_R, fast_t, fast_mask, fast_diag = estimate_pose(
                fast_p1, fast_p2, camera, cfg
            )
        fast_success = bool(
            fast_R is not None
            and np.count_nonzero(fast_mask) >= cfg.min_pose_matches
        )

    hard_accept, conf_diag = _adaptive_pose_confidence(
        fast_success,
        fast_p1,
        fast_p2,
        fast_mask,
        gray1.shape[:2],
        cfg,
    )
    result.diagnostics.update(conf_diag)
    result.diagnostics.update(
        adaptive_poseguard_quality_margin=float(
            cfg.adaptive_poseguard_quality_margin
        ),
        adaptive_poseguard_prelk_margin=float(
            cfg.adaptive_poseguard_prelk_margin
        ),
        adaptive_poseguard_parallax_target_deg=float(
            cfg.adaptive_poseguard_parallax_target_deg
        ),
        adaptive_poseguard_rescue_fast_cheirality_max=float(
            cfg.adaptive_poseguard_rescue_fast_cheirality_max
        ),
        adaptive_poseguard_rescue_candidate_cheirality_min=float(
            cfg.adaptive_poseguard_rescue_candidate_cheirality_min
        ),
        adaptive_poseguard_rescue_cheirality_gain=float(
            cfg.adaptive_poseguard_rescue_cheirality_gain
        ),
        adaptive_poseguard_v3_quality_margin=float(
            cfg.adaptive_poseguard_v3_quality_margin
        ),
        adaptive_poseguard_v3_prelk_margin=float(
            cfg.adaptive_poseguard_v3_prelk_margin
        ),
        adaptive_poseguard_v3_min_candidate_native_cheirality=float(
            cfg.adaptive_poseguard_v3_min_candidate_native_cheirality
        ),
        adaptive_poseguard_v3_min_candidate_parallax_deg=float(
            cfg.adaptive_poseguard_v3_min_candidate_parallax_deg
        ),
        adaptive_poseguard_v3_rescue_candidate_native_cheirality_min=float(
            cfg.adaptive_poseguard_v3_rescue_candidate_native_cheirality_min
        ),
        adaptive_poseguard_v3_rescue_native_cheirality_gain=float(
            cfg.adaptive_poseguard_v3_rescue_native_cheirality_gain
        ),
        adaptive_poseguard_v3_rescue_candidate_shared_cheirality_min=float(
            cfg.adaptive_poseguard_v3_rescue_candidate_shared_cheirality_min
        ),
        adaptive_poseguard_v3_rescue_shared_cheirality_gain=float(
            cfg.adaptive_poseguard_v3_rescue_shared_cheirality_gain
        ),
        adaptive_poseguard_v3_early_abort_native_cheirality_max=float(
            cfg.adaptive_poseguard_v3_early_abort_native_cheirality_max
        ),
        adaptive_poseguard_v3_early_abort_parallax_max_deg=float(
            cfg.adaptive_poseguard_v3_early_abort_parallax_max_deg
        ),
        fast_raw_matches=int(len(fast_pairs)),
    )
    if len(fast_distances):
        result.diagnostics["fast_median_descriptor_distance"] = float(
            np.median(fast_distances)
        )
    if len(fast_ratios):
        result.diagnostics["fast_median_ratio"] = float(
            np.median(fast_ratios)
        )

    def finalize(
        p1: np.ndarray,
        p2: np.ndarray,
        R_est: Optional[np.ndarray],
        t_est: Optional[np.ndarray],
        mask: np.ndarray,
        status: str,
        path: str,
        raw_matches: int,
        diag: Optional[Mapping[str, Any]] = None,
    ) -> MethodResult:
        result.points1 = np.ascontiguousarray(p1, dtype=np.float32)
        result.points2 = np.ascontiguousarray(p2, dtype=np.float32)
        result.num_raw_matches = int(raw_matches)
        result.num_filtered_matches = int(len(result.points1))
        result.R = R_est
        result.t = t_est
        result.geometric_inlier_mask = np.asarray(mask, dtype=bool)
        result.num_geometric_inliers = int(
            np.count_nonzero(result.geometric_inlier_mask)
        )
        result.success = bool(
            R_est is not None
            and t_est is not None
            and result.num_geometric_inliers >= cfg.min_pose_matches
        )
        result.status = status if result.success else "adaptive_poseguard_v3_failed"
        result.diagnostics["adaptive_poseguard_v3_path"] = path
        if diag:
            result.diagnostics.update(dict(diag))
        return result

    if hard_accept:
        result.diagnostics["adaptive_poseguard_v3_fast_accept_type"] = "hard"
        return finalize(
            fast_p1, fast_p2, fast_R, fast_t, fast_mask,
            "ok_fast_hard_accept", "fast_hard_accept",
            len(fast_pairs), fast_diag,
        )

    # Shared fast quality is the reference for any later replacement.
    fast_shared = _poseguard_shared_quality(
        fast_R, fast_t, fast_p1, fast_p2, camera, cfg, gray1.shape[:2]
    )
    fast_shared["poseguard_native_cheirality_ratio"] = float(
        np.count_nonzero(fast_mask) / max(len(fast_p1), 1)
    ) if fast_success else 0.0
    for key, value in fast_shared.items():
        suffix = key[10:] if key.startswith("poseguard_") else key
        result.diagnostics[f"poseguard_fast_{suffix}"] = value

    best_name = "fast" if fast_success else "none"
    best_p1, best_p2 = fast_p1, fast_p2
    best_R, best_t, best_mask = fast_R, fast_t, fast_mask
    best_diag: Dict[str, Any] = dict(fast_diag)
    best_score = (
        float(fast_shared["poseguard_quality_score"])
        if fast_success else -float("inf")
    )
    best_raw_matches = len(fast_pairs)

    # -------------------- Stage B: reused structural support -------------
    with StageTimer(result.timings_ms, "structural_support_ms"):
        sf1 = _attach_structural_fields(gray1, f1, cfg)
        sf2 = _attach_structural_fields(gray2, f2, cfg)
        fallback_budget = max(
            cfg.min_filter_matches,
            int(
                round(
                    cfg.nfeatures
                    * float(cfg.adaptive_fallback_feature_fraction)
                )
            ),
        )
        fallback_budget = min(cfg.nfeatures, fallback_budget)
        sf1 = _edge_rerank_existing_features(
            sf1, gray1.shape[:2], fallback_budget, cfg
        )
        sf2 = _edge_rerank_existing_features(
            sf2, gray2.shape[:2], fallback_budget, cfg
        )

    with StageTimer(result.timings_ms, "refine_match_ms"):
        pairs, distances, ratios = mutual_ratio_matches(
            sf1, sf2, fixed_ratio=cfg.ratio, adaptive=True, cfg=cfg
        )
    fallback_raw = int(len(pairs))
    result.diagnostics["fallback_raw_matches"] = fallback_raw

    if len(pairs) < cfg.min_pose_matches:
        if fast_success:
            return finalize(
                fast_p1, fast_p2, fast_R, fast_t, fast_mask,
                "ok_fast_poseguard", "fallback_insufficient_use_fast",
                len(fast_pairs), fast_diag,
            )
        result.status = "adaptive_poseguard_v3_pose_failed"
        result.diagnostics["adaptive_poseguard_v3_path"] = "failed_no_fallback_matches"
        return result

    with StageTimer(result.timings_ms, "refine_filter_ms"):
        keep = orientation_scale_consistency(sf1, sf2, pairs, cfg)
        if np.count_nonzero(keep) >= cfg.min_filter_matches:
            pairs = pairs[keep]
        edge_keep = edge_orientation_consistency(sf1, sf2, pairs, cfg)
        if np.count_nonzero(edge_keep) >= cfg.min_filter_matches:
            pairs = pairs[edge_keep]

    if len(pairs) < cfg.min_pose_matches:
        if fast_success:
            return finalize(
                fast_p1, fast_p2, fast_R, fast_t, fast_mask,
                "ok_fast_poseguard", "fallback_filtered_use_fast",
                len(fast_pairs), fast_diag,
            )
        result.status = "adaptive_poseguard_v3_pose_failed"
        result.diagnostics["adaptive_poseguard_v3_path"] = "failed_after_filter"
        return result

    # Limit the structural candidate set before geometry/LK.
    max_lk = int(cfg.adaptive_lk_max_matches)
    if max_lk > 0 and len(pairs) > max_lk:
        support = np.minimum(
            sf1.edge_confidence[pairs[:, 0]],
            sf2.edge_confidence[pairs[:, 1]],
        ).astype(np.float32)
        choose = grid_balanced_indices(
            sf1.points[pairs[:, 0]],
            support,
            gray1.shape[:2],
            max_lk,
            cfg.grid_rows,
            cfg.grid_cols,
        )
        pairs = pairs[choose]

    pre_p1 = np.ascontiguousarray(
        sf1.points[pairs[:, 0]], dtype=np.float32
    )
    pre_p2 = np.ascontiguousarray(
        sf2.points[pairs[:, 1]], dtype=np.float32
    )

    # -------------------- Stage B1: pre-LK pose --------------------------
    pre_R: Optional[np.ndarray] = None
    pre_t: Optional[np.ndarray] = None
    pre_mask = np.empty(0, dtype=bool)
    pre_diag: Dict[str, Any] = {}
    pre_success = False
    with StageTimer(result.timings_ms, "prelk_geometry_ms"):
        pre_R, pre_t, pre_mask, pre_diag = estimate_pose(
            pre_p1, pre_p2, camera, cfg
        )
    pre_success = bool(
        pre_R is not None
        and np.count_nonzero(pre_mask) >= cfg.min_pose_matches
    )

    pre_shared = _poseguard_shared_quality(
        pre_R, pre_t, fast_p1, fast_p2, camera, cfg, gray1.shape[:2]
    )
    pre_shared["poseguard_native_cheirality_ratio"] = float(
        np.count_nonzero(pre_mask) / max(len(pre_p1), 1)
    ) if pre_success else 0.0
    for key, value in pre_shared.items():
        suffix = key[10:] if key.startswith("poseguard_") else key
        result.diagnostics[f"poseguard_prelk_{suffix}"] = value

    pre_score = float(pre_shared["poseguard_quality_score"])
    pre_observable, pre_obs_reason = _poseguard_v3_candidate_observable(
        pre_shared, cfg
    ) if pre_success else (False, "no_candidate")
    result.diagnostics["poseguard_v3_prelk_observable"] = float(pre_observable)
    result.diagnostics["poseguard_v3_prelk_observability_reason"] = pre_obs_reason

    abort_lk, abort_reason = _poseguard_v3_should_abort_lk(
        fast_success, pre_shared, cfg
    ) if pre_success else (False, "no_candidate")
    result.diagnostics["poseguard_v3_early_abort_lk"] = float(abort_lk)
    result.diagnostics["poseguard_v3_early_abort_reason"] = abort_reason

    pre_early = False
    pre_reason = "no_candidate"
    if pre_success:
        if not fast_success:
            pre_early = bool(pre_observable)
            pre_reason = "fast_invalid_observable_candidate" if pre_early else f"fast_invalid_{pre_obs_reason}"
        else:
            pre_early, pre_reason = _poseguard_v3_should_replace(
                fast_shared, pre_shared, cfg, prelk=True
            )
    result.diagnostics["poseguard_prelk_decision"] = pre_reason
    if pre_early:
        best_name = "prelk"
        best_p1, best_p2 = pre_p1, pre_p2
        best_R, best_t, best_mask = pre_R, pre_t, pre_mask
        best_diag = dict(pre_diag)
        best_score = pre_score
        best_raw_matches = fallback_raw
        result.diagnostics["poseguard_selected_quality_score"] = best_score
        return finalize(
            best_p1, best_p2, best_R, best_t, best_mask,
            "ok_prelk_selected", "prelk_early_accept",
            best_raw_matches, best_diag,
        )

    # Keep the pre-LK pose as a final candidate only when it passes the new
    # observability gate. An unobservable candidate may still be refined by LK.
    candidate_pre = bool(pre_success and pre_observable)

    # Conservative long-tail optimization: if the pre-LK fallback is extremely
    # weak in both positive-depth support and parallax, and a valid fast pose
    # already exists, do not spend additional time on LK.
    if abort_lk and fast_success:
        result.diagnostics["poseguard_selected_candidate"] = "fast"
        result.diagnostics["poseguard_selected_quality_score"] = float(
            fast_shared["poseguard_quality_score"]
        )
        result.diagnostics["poseguard_refinement_replaced_fast"] = 0.0
        return finalize(
            fast_p1, fast_p2, fast_R, fast_t, fast_mask,
            "ok_fast_poseguard_v3", "early_abort_lk_keep_fast",
            len(fast_pairs), fast_diag,
        )

    # -------------------- Stage B2: selective LK -------------------------
    lk_p1, lk_p2 = pre_p1, pre_p2
    if cfg.use_lk_refinement and len(pre_p1) >= cfg.min_filter_matches:
        with StageTimer(result.timings_ms, "lk_refine_ms"):
            p1_ref, p2_ref, lk_keep, lk_diag = refine_matches_lk(
                gray1, gray2, pre_p1, pre_p2, cfg
            )
        result.diagnostics.update(lk_diag)
        if np.count_nonzero(lk_keep) >= cfg.min_filter_matches:
            lk_p1 = np.ascontiguousarray(
                p1_ref[lk_keep], dtype=np.float32
            )
            lk_p2 = np.ascontiguousarray(
                p2_ref[lk_keep], dtype=np.float32
            )
        else:
            lk_p1 = np.ascontiguousarray(p1_ref, dtype=np.float32)
            lk_p2 = np.ascontiguousarray(p2_ref, dtype=np.float32)

    ref_R: Optional[np.ndarray] = None
    ref_t: Optional[np.ndarray] = None
    ref_mask = np.empty(0, dtype=bool)
    ref_diag: Dict[str, Any] = {}
    ref_success = False

    if len(lk_p1) >= cfg.min_pose_matches:
        with StageTimer(result.timings_ms, "refine_geometry_ms"):
            ref_R, ref_t, ref_mask, ref_diag = estimate_pose(
                lk_p1, lk_p2, camera, cfg
            )
        ref_success = bool(
            ref_R is not None
            and np.count_nonzero(ref_mask) >= cfg.min_pose_matches
        )

    ref_shared = _poseguard_shared_quality(
        ref_R, ref_t, fast_p1, fast_p2, camera, cfg, gray1.shape[:2]
    )
    ref_shared["poseguard_native_cheirality_ratio"] = float(
        np.count_nonzero(ref_mask) / max(len(lk_p1), 1)
    ) if ref_success else 0.0
    for key, value in ref_shared.items():
        suffix = key[10:] if key.startswith("poseguard_") else key
        result.diagnostics[f"poseguard_refined_{suffix}"] = value

    ref_observable, ref_obs_reason = _poseguard_v3_candidate_observable(
        ref_shared, cfg
    ) if ref_success else (False, "no_candidate")
    result.diagnostics["poseguard_v3_refined_observable"] = float(ref_observable)
    result.diagnostics["poseguard_v3_refined_observability_reason"] = ref_obs_reason

    # Final candidate selection uses pose observability. A candidate can replace
    # a weak fast pose through an explicit cheirality-rescue rule; otherwise it
    # must exceed the current quality by the configured margin without a large
    # loss of positive-depth support.
    best_quality = fast_shared if fast_success else {
        "poseguard_quality_score": 0.0,
        "poseguard_cheirality_ratio": 0.0,
    }

    if candidate_pre:
        pre_score = float(pre_shared["poseguard_quality_score"])
        if best_name == "none":
            replace_pre, pre_reason = True, "no_current_candidate"
        else:
            replace_pre, pre_reason = _poseguard_v3_should_replace(
                best_quality, pre_shared, cfg, prelk=False
            )
        result.diagnostics["poseguard_final_prelk_decision"] = pre_reason
        if replace_pre:
            best_name = "prelk"
            best_p1, best_p2 = pre_p1, pre_p2
            best_R, best_t, best_mask = pre_R, pre_t, pre_mask
            best_diag = dict(pre_diag)
            best_score = pre_score
            best_raw_matches = fallback_raw
            best_quality = pre_shared

    if ref_success and ref_observable:
        ref_score = float(ref_shared["poseguard_quality_score"])
        if best_name == "none":
            replace_ref, ref_reason = True, "no_current_candidate"
        else:
            replace_ref, ref_reason = _poseguard_v3_should_replace(
                best_quality, ref_shared, cfg, prelk=False
            )
        result.diagnostics["poseguard_final_refined_decision"] = ref_reason
        if replace_ref:
            best_name = "refined"
            best_p1, best_p2 = lk_p1, lk_p2
            best_R, best_t, best_mask = ref_R, ref_t, ref_mask
            best_diag = dict(ref_diag)
            best_score = ref_score
            best_raw_matches = fallback_raw
            best_quality = ref_shared

    result.diagnostics["poseguard_selected_candidate"] = best_name
    result.diagnostics["poseguard_selected_quality_score"] = float(
        best_score if math.isfinite(best_score) else -1.0
    )
    result.diagnostics["poseguard_refinement_replaced_fast"] = float(
        best_name in {"prelk", "refined"}
    )

    if best_name != "none":
        status = {
            "fast": "ok_fast_poseguard_v3",
            "prelk": "ok_prelk_poseguard_v3",
            "refined": "ok_refined_poseguard_v3",
        }[best_name]
        path = {
            "fast": "quality_gate_kept_fast",
            "prelk": "quality_gate_selected_prelk",
            "refined": "quality_gate_selected_refined",
        }[best_name]
        return finalize(
            best_p1, best_p2, best_R, best_t, best_mask,
            status, path, best_raw_matches, best_diag,
        )

    result.status = "adaptive_poseguard_v3_pose_failed"
    result.diagnostics["adaptive_poseguard_v3_path"] = "failed"
    return result



def run_edgefusion_adaptive_poseguard_v4(
    gray1: np.ndarray,
    gray2: np.ndarray,
    camera: CameraModel,
    cfg: PipelineConfig,
) -> MethodResult:
    """Run EdgeFusion-Adaptive-PoseGuard-v4.

    This method keeps Adaptive, Guarded, PoseGuard-v2, and PoseGuard-v3
    unchanged. V4 preserves the v3 observability and rescue logic for
    fast-to-structural replacement, but adds a hierarchical LK stability gate:
    once a pre-LK structural pose has become the current solution, LK cannot
    replace it through cheirality rescue alone.

    1. ORB-Lowe + pose is computed first.
    2. Hard high-confidence poses are accepted as before.
    3. Uncertain pairs receive structural re-ranking/filtering using the
       already-computed ORB features.
    4. A pre-LK structural pose is estimated and evaluated for positive-depth
       support and parallax; a clear rescue can skip LK.
    5. Otherwise selective LK is applied and the refined pose is estimated.
    6. A replacement candidate must first pass native-cheirality and parallax
       validity gates; only then is quality/rescue ranking considered.
    7. LK can be skipped when the pre-LK fallback is extremely weak and a valid
       fast pose already exists.
    8. Cheirality rescue is restricted to fast-to-structural replacement.
    9. If pre-LK is already selected, LK must remain rotation/translation and
       parallax-consistent with it and provide an explicit quality gain.
   10. Fast, pre-LK, and refined candidates are compared without ground truth.

    Model selection uses no benchmark ground truth.
    """
    method = "edgefusion_adaptive_poseguard_v4"
    result = MethodResult(method=method, success=False, status="started")

    # -------------------- Stage A: fast ORB-Lowe -------------------------
    with StageTimer(result.timings_ms, "fast_extract_ms"):
        f1 = extract_standard(gray1, "orb", cfg.nfeatures)
        f2 = extract_standard(gray2, "orb", cfg.nfeatures)
    result.num_keypoints1 = len(f1.keypoints)
    result.num_keypoints2 = len(f2.keypoints)

    with StageTimer(result.timings_ms, "fast_match_ms"):
        fast_pairs, fast_distances, fast_ratios = oneway_ratio_matches(
            f1, f2, float(cfg.ratio)
        )

    fast_p1 = np.empty((0, 2), dtype=np.float32)
    fast_p2 = np.empty((0, 2), dtype=np.float32)
    fast_R: Optional[np.ndarray] = None
    fast_t: Optional[np.ndarray] = None
    fast_mask = np.empty(0, dtype=bool)
    fast_diag: Dict[str, Any] = {}
    fast_success = False

    if len(fast_pairs) >= cfg.min_pose_matches:
        fast_p1 = np.ascontiguousarray(
            f1.points[fast_pairs[:, 0]], dtype=np.float32
        )
        fast_p2 = np.ascontiguousarray(
            f2.points[fast_pairs[:, 1]], dtype=np.float32
        )
        with StageTimer(result.timings_ms, "fast_geometry_ms"):
            fast_R, fast_t, fast_mask, fast_diag = estimate_pose(
                fast_p1, fast_p2, camera, cfg
            )
        fast_success = bool(
            fast_R is not None
            and np.count_nonzero(fast_mask) >= cfg.min_pose_matches
        )

    hard_accept, conf_diag = _adaptive_pose_confidence(
        fast_success,
        fast_p1,
        fast_p2,
        fast_mask,
        gray1.shape[:2],
        cfg,
    )
    result.diagnostics.update(conf_diag)
    result.diagnostics.update(
        adaptive_poseguard_quality_margin=float(
            cfg.adaptive_poseguard_quality_margin
        ),
        adaptive_poseguard_prelk_margin=float(
            cfg.adaptive_poseguard_prelk_margin
        ),
        adaptive_poseguard_parallax_target_deg=float(
            cfg.adaptive_poseguard_parallax_target_deg
        ),
        adaptive_poseguard_rescue_fast_cheirality_max=float(
            cfg.adaptive_poseguard_rescue_fast_cheirality_max
        ),
        adaptive_poseguard_rescue_candidate_cheirality_min=float(
            cfg.adaptive_poseguard_rescue_candidate_cheirality_min
        ),
        adaptive_poseguard_rescue_cheirality_gain=float(
            cfg.adaptive_poseguard_rescue_cheirality_gain
        ),
        adaptive_poseguard_v3_quality_margin=float(
            cfg.adaptive_poseguard_v3_quality_margin
        ),
        adaptive_poseguard_v3_prelk_margin=float(
            cfg.adaptive_poseguard_v3_prelk_margin
        ),
        adaptive_poseguard_v3_min_candidate_native_cheirality=float(
            cfg.adaptive_poseguard_v3_min_candidate_native_cheirality
        ),
        adaptive_poseguard_v3_min_candidate_parallax_deg=float(
            cfg.adaptive_poseguard_v3_min_candidate_parallax_deg
        ),
        adaptive_poseguard_v3_rescue_candidate_native_cheirality_min=float(
            cfg.adaptive_poseguard_v3_rescue_candidate_native_cheirality_min
        ),
        adaptive_poseguard_v3_rescue_native_cheirality_gain=float(
            cfg.adaptive_poseguard_v3_rescue_native_cheirality_gain
        ),
        adaptive_poseguard_v3_rescue_candidate_shared_cheirality_min=float(
            cfg.adaptive_poseguard_v3_rescue_candidate_shared_cheirality_min
        ),
        adaptive_poseguard_v3_rescue_shared_cheirality_gain=float(
            cfg.adaptive_poseguard_v3_rescue_shared_cheirality_gain
        ),
        adaptive_poseguard_v3_early_abort_native_cheirality_max=float(
            cfg.adaptive_poseguard_v3_early_abort_native_cheirality_max
        ),
        adaptive_poseguard_v3_early_abort_parallax_max_deg=float(
            cfg.adaptive_poseguard_v3_early_abort_parallax_max_deg
        ),
        adaptive_poseguard_v4_lk_quality_margin=float(
            cfg.adaptive_poseguard_v4_lk_quality_margin
        ),
        adaptive_poseguard_v4_lk_max_rotation_disagreement_deg=float(
            cfg.adaptive_poseguard_v4_lk_max_rotation_disagreement_deg
        ),
        adaptive_poseguard_v4_lk_max_translation_disagreement_deg=float(
            cfg.adaptive_poseguard_v4_lk_max_translation_disagreement_deg
        ),
        adaptive_poseguard_v4_lk_parallax_ratio_min=float(
            cfg.adaptive_poseguard_v4_lk_parallax_ratio_min
        ),
        adaptive_poseguard_v4_lk_parallax_ratio_max=float(
            cfg.adaptive_poseguard_v4_lk_parallax_ratio_max
        ),
        fast_raw_matches=int(len(fast_pairs)),
    )
    if len(fast_distances):
        result.diagnostics["fast_median_descriptor_distance"] = float(
            np.median(fast_distances)
        )
    if len(fast_ratios):
        result.diagnostics["fast_median_ratio"] = float(
            np.median(fast_ratios)
        )

    def finalize(
        p1: np.ndarray,
        p2: np.ndarray,
        R_est: Optional[np.ndarray],
        t_est: Optional[np.ndarray],
        mask: np.ndarray,
        status: str,
        path: str,
        raw_matches: int,
        diag: Optional[Mapping[str, Any]] = None,
    ) -> MethodResult:
        result.points1 = np.ascontiguousarray(p1, dtype=np.float32)
        result.points2 = np.ascontiguousarray(p2, dtype=np.float32)
        result.num_raw_matches = int(raw_matches)
        result.num_filtered_matches = int(len(result.points1))
        result.R = R_est
        result.t = t_est
        result.geometric_inlier_mask = np.asarray(mask, dtype=bool)
        result.num_geometric_inliers = int(
            np.count_nonzero(result.geometric_inlier_mask)
        )
        result.success = bool(
            R_est is not None
            and t_est is not None
            and result.num_geometric_inliers >= cfg.min_pose_matches
        )
        result.status = status if result.success else "adaptive_poseguard_v4_failed"
        result.diagnostics["adaptive_poseguard_v4_path"] = path
        if diag:
            result.diagnostics.update(dict(diag))
        return result

    if hard_accept:
        result.diagnostics["adaptive_poseguard_v4_fast_accept_type"] = "hard"
        return finalize(
            fast_p1, fast_p2, fast_R, fast_t, fast_mask,
            "ok_fast_hard_accept", "fast_hard_accept",
            len(fast_pairs), fast_diag,
        )

    # Shared fast quality is the reference for any later replacement.
    fast_shared = _poseguard_shared_quality(
        fast_R, fast_t, fast_p1, fast_p2, camera, cfg, gray1.shape[:2]
    )
    fast_shared["poseguard_native_cheirality_ratio"] = float(
        np.count_nonzero(fast_mask) / max(len(fast_p1), 1)
    ) if fast_success else 0.0
    for key, value in fast_shared.items():
        suffix = key[10:] if key.startswith("poseguard_") else key
        result.diagnostics[f"poseguard_fast_{suffix}"] = value

    best_name = "fast" if fast_success else "none"
    best_p1, best_p2 = fast_p1, fast_p2
    best_R, best_t, best_mask = fast_R, fast_t, fast_mask
    best_diag: Dict[str, Any] = dict(fast_diag)
    best_score = (
        float(fast_shared["poseguard_quality_score"])
        if fast_success else -float("inf")
    )
    best_raw_matches = len(fast_pairs)

    # -------------------- Stage B: reused structural support -------------
    with StageTimer(result.timings_ms, "structural_support_ms"):
        sf1 = _attach_structural_fields(gray1, f1, cfg)
        sf2 = _attach_structural_fields(gray2, f2, cfg)
        fallback_budget = max(
            cfg.min_filter_matches,
            int(
                round(
                    cfg.nfeatures
                    * float(cfg.adaptive_fallback_feature_fraction)
                )
            ),
        )
        fallback_budget = min(cfg.nfeatures, fallback_budget)
        sf1 = _edge_rerank_existing_features(
            sf1, gray1.shape[:2], fallback_budget, cfg
        )
        sf2 = _edge_rerank_existing_features(
            sf2, gray2.shape[:2], fallback_budget, cfg
        )

    with StageTimer(result.timings_ms, "refine_match_ms"):
        pairs, distances, ratios = mutual_ratio_matches(
            sf1, sf2, fixed_ratio=cfg.ratio, adaptive=True, cfg=cfg
        )
    fallback_raw = int(len(pairs))
    result.diagnostics["fallback_raw_matches"] = fallback_raw

    if len(pairs) < cfg.min_pose_matches:
        if fast_success:
            return finalize(
                fast_p1, fast_p2, fast_R, fast_t, fast_mask,
                "ok_fast_poseguard", "fallback_insufficient_use_fast",
                len(fast_pairs), fast_diag,
            )
        result.status = "adaptive_poseguard_v4_pose_failed"
        result.diagnostics["adaptive_poseguard_v4_path"] = "failed_no_fallback_matches"
        return result

    with StageTimer(result.timings_ms, "refine_filter_ms"):
        keep = orientation_scale_consistency(sf1, sf2, pairs, cfg)
        if np.count_nonzero(keep) >= cfg.min_filter_matches:
            pairs = pairs[keep]
        edge_keep = edge_orientation_consistency(sf1, sf2, pairs, cfg)
        if np.count_nonzero(edge_keep) >= cfg.min_filter_matches:
            pairs = pairs[edge_keep]

    if len(pairs) < cfg.min_pose_matches:
        if fast_success:
            return finalize(
                fast_p1, fast_p2, fast_R, fast_t, fast_mask,
                "ok_fast_poseguard", "fallback_filtered_use_fast",
                len(fast_pairs), fast_diag,
            )
        result.status = "adaptive_poseguard_v4_pose_failed"
        result.diagnostics["adaptive_poseguard_v4_path"] = "failed_after_filter"
        return result

    # Limit the structural candidate set before geometry/LK.
    max_lk = int(cfg.adaptive_lk_max_matches)
    if max_lk > 0 and len(pairs) > max_lk:
        support = np.minimum(
            sf1.edge_confidence[pairs[:, 0]],
            sf2.edge_confidence[pairs[:, 1]],
        ).astype(np.float32)
        choose = grid_balanced_indices(
            sf1.points[pairs[:, 0]],
            support,
            gray1.shape[:2],
            max_lk,
            cfg.grid_rows,
            cfg.grid_cols,
        )
        pairs = pairs[choose]

    pre_p1 = np.ascontiguousarray(
        sf1.points[pairs[:, 0]], dtype=np.float32
    )
    pre_p2 = np.ascontiguousarray(
        sf2.points[pairs[:, 1]], dtype=np.float32
    )

    # -------------------- Stage B1: pre-LK pose --------------------------
    pre_R: Optional[np.ndarray] = None
    pre_t: Optional[np.ndarray] = None
    pre_mask = np.empty(0, dtype=bool)
    pre_diag: Dict[str, Any] = {}
    pre_success = False
    with StageTimer(result.timings_ms, "prelk_geometry_ms"):
        pre_R, pre_t, pre_mask, pre_diag = estimate_pose(
            pre_p1, pre_p2, camera, cfg
        )
    pre_success = bool(
        pre_R is not None
        and np.count_nonzero(pre_mask) >= cfg.min_pose_matches
    )

    pre_shared = _poseguard_shared_quality(
        pre_R, pre_t, fast_p1, fast_p2, camera, cfg, gray1.shape[:2]
    )
    pre_shared["poseguard_native_cheirality_ratio"] = float(
        np.count_nonzero(pre_mask) / max(len(pre_p1), 1)
    ) if pre_success else 0.0
    for key, value in pre_shared.items():
        suffix = key[10:] if key.startswith("poseguard_") else key
        result.diagnostics[f"poseguard_prelk_{suffix}"] = value

    pre_score = float(pre_shared["poseguard_quality_score"])
    pre_observable, pre_obs_reason = _poseguard_v3_candidate_observable(
        pre_shared, cfg
    ) if pre_success else (False, "no_candidate")
    result.diagnostics["poseguard_v3_prelk_observable"] = float(pre_observable)
    result.diagnostics["poseguard_v3_prelk_observability_reason"] = pre_obs_reason

    abort_lk, abort_reason = _poseguard_v3_should_abort_lk(
        fast_success, pre_shared, cfg
    ) if pre_success else (False, "no_candidate")
    result.diagnostics["poseguard_v3_early_abort_lk"] = float(abort_lk)
    result.diagnostics["poseguard_v3_early_abort_reason"] = abort_reason

    pre_early = False
    pre_reason = "no_candidate"
    if pre_success:
        if not fast_success:
            pre_early = bool(pre_observable)
            pre_reason = "fast_invalid_observable_candidate" if pre_early else f"fast_invalid_{pre_obs_reason}"
        else:
            pre_early, pre_reason = _poseguard_v3_should_replace(
                fast_shared, pre_shared, cfg, prelk=True
            )
    result.diagnostics["poseguard_prelk_decision"] = pre_reason
    if pre_early:
        best_name = "prelk"
        best_p1, best_p2 = pre_p1, pre_p2
        best_R, best_t, best_mask = pre_R, pre_t, pre_mask
        best_diag = dict(pre_diag)
        best_score = pre_score
        best_raw_matches = fallback_raw
        result.diagnostics["poseguard_selected_quality_score"] = best_score
        return finalize(
            best_p1, best_p2, best_R, best_t, best_mask,
            "ok_prelk_selected", "prelk_early_accept",
            best_raw_matches, best_diag,
        )

    # Keep the pre-LK pose as a final candidate only when it passes the new
    # observability gate. An unobservable candidate may still be refined by LK.
    candidate_pre = bool(pre_success and pre_observable)

    # Conservative long-tail optimization: if the pre-LK fallback is extremely
    # weak in both positive-depth support and parallax, and a valid fast pose
    # already exists, do not spend additional time on LK.
    if abort_lk and fast_success:
        result.diagnostics["poseguard_selected_candidate"] = "fast"
        result.diagnostics["poseguard_selected_quality_score"] = float(
            fast_shared["poseguard_quality_score"]
        )
        result.diagnostics["poseguard_refinement_replaced_fast"] = 0.0
        return finalize(
            fast_p1, fast_p2, fast_R, fast_t, fast_mask,
            "ok_fast_poseguard_v4", "early_abort_lk_keep_fast",
            len(fast_pairs), fast_diag,
        )

    # -------------------- Stage B2: selective LK -------------------------
    lk_p1, lk_p2 = pre_p1, pre_p2
    if cfg.use_lk_refinement and len(pre_p1) >= cfg.min_filter_matches:
        with StageTimer(result.timings_ms, "lk_refine_ms"):
            p1_ref, p2_ref, lk_keep, lk_diag = refine_matches_lk(
                gray1, gray2, pre_p1, pre_p2, cfg
            )
        result.diagnostics.update(lk_diag)
        if np.count_nonzero(lk_keep) >= cfg.min_filter_matches:
            lk_p1 = np.ascontiguousarray(
                p1_ref[lk_keep], dtype=np.float32
            )
            lk_p2 = np.ascontiguousarray(
                p2_ref[lk_keep], dtype=np.float32
            )
        else:
            lk_p1 = np.ascontiguousarray(p1_ref, dtype=np.float32)
            lk_p2 = np.ascontiguousarray(p2_ref, dtype=np.float32)

    ref_R: Optional[np.ndarray] = None
    ref_t: Optional[np.ndarray] = None
    ref_mask = np.empty(0, dtype=bool)
    ref_diag: Dict[str, Any] = {}
    ref_success = False

    if len(lk_p1) >= cfg.min_pose_matches:
        with StageTimer(result.timings_ms, "refine_geometry_ms"):
            ref_R, ref_t, ref_mask, ref_diag = estimate_pose(
                lk_p1, lk_p2, camera, cfg
            )
        ref_success = bool(
            ref_R is not None
            and np.count_nonzero(ref_mask) >= cfg.min_pose_matches
        )

    ref_shared = _poseguard_shared_quality(
        ref_R, ref_t, fast_p1, fast_p2, camera, cfg, gray1.shape[:2]
    )
    ref_shared["poseguard_native_cheirality_ratio"] = float(
        np.count_nonzero(ref_mask) / max(len(lk_p1), 1)
    ) if ref_success else 0.0
    for key, value in ref_shared.items():
        suffix = key[10:] if key.startswith("poseguard_") else key
        result.diagnostics[f"poseguard_refined_{suffix}"] = value

    ref_observable, ref_obs_reason = _poseguard_v3_candidate_observable(
        ref_shared, cfg
    ) if ref_success else (False, "no_candidate")
    result.diagnostics["poseguard_v3_refined_observable"] = float(ref_observable)
    result.diagnostics["poseguard_v3_refined_observability_reason"] = ref_obs_reason

    # Final candidate selection uses pose observability. A candidate can replace
    # a weak fast pose through an explicit cheirality-rescue rule; otherwise it
    # must exceed the current quality by the configured margin without a large
    # loss of positive-depth support.
    best_quality = fast_shared if fast_success else {
        "poseguard_quality_score": 0.0,
        "poseguard_cheirality_ratio": 0.0,
    }

    if candidate_pre:
        pre_score = float(pre_shared["poseguard_quality_score"])
        if best_name == "none":
            replace_pre, pre_reason = True, "no_current_candidate"
        else:
            replace_pre, pre_reason = _poseguard_v3_should_replace(
                best_quality, pre_shared, cfg, prelk=False
            )
        result.diagnostics["poseguard_final_prelk_decision"] = pre_reason
        if replace_pre:
            best_name = "prelk"
            best_p1, best_p2 = pre_p1, pre_p2
            best_R, best_t, best_mask = pre_R, pre_t, pre_mask
            best_diag = dict(pre_diag)
            best_score = pre_score
            best_raw_matches = fallback_raw
            best_quality = pre_shared

    if ref_success and ref_observable:
        ref_score = float(ref_shared["poseguard_quality_score"])
        if best_name == "none":
            replace_ref, ref_reason = True, "no_current_candidate"
        elif best_name == "prelk":
            # V4 hierarchy: once the structural pre-LK pose is current, LK is
            # treated strictly as a local refinement. Cheirality rescue is not
            # permitted to override pre-LK. The refined pose must remain
            # geometrically stable relative to pre-LK and improve quality.
            replace_ref, ref_reason, stability_diag = _poseguard_v4_lk_should_replace_prelk(
                best_quality,
                ref_shared,
                best_R,
                best_t,
                ref_R,
                ref_t,
                cfg,
            )
            result.diagnostics.update(stability_diag)
        else:
            # Fast -> refined is still a structural rescue path, so retain the
            # v3 observability + cheirality-gain logic here.
            replace_ref, ref_reason = _poseguard_v3_should_replace(
                best_quality, ref_shared, cfg, prelk=False
            )
        result.diagnostics["poseguard_final_refined_decision"] = ref_reason
        if replace_ref:
            best_name = "refined"
            best_p1, best_p2 = lk_p1, lk_p2
            best_R, best_t, best_mask = ref_R, ref_t, ref_mask
            best_diag = dict(ref_diag)
            best_score = ref_score
            best_raw_matches = fallback_raw
            best_quality = ref_shared

    result.diagnostics["poseguard_selected_candidate"] = best_name
    result.diagnostics["poseguard_selected_quality_score"] = float(
        best_score if math.isfinite(best_score) else -1.0
    )
    result.diagnostics["poseguard_refinement_replaced_fast"] = float(
        best_name in {"prelk", "refined"}
    )

    if best_name != "none":
        status = {
            "fast": "ok_fast_poseguard_v4",
            "prelk": "ok_prelk_poseguard_v4",
            "refined": "ok_refined_poseguard_v4",
        }[best_name]
        path = {
            "fast": "quality_gate_kept_fast",
            "prelk": "quality_gate_selected_prelk",
            "refined": "quality_gate_selected_refined",
        }[best_name]
        return finalize(
            best_p1, best_p2, best_R, best_t, best_mask,
            status, path, best_raw_matches, best_diag,
        )

    result.status = "adaptive_poseguard_v4_pose_failed"
    result.diagnostics["adaptive_poseguard_v4_path"] = "failed"
    return result



def _load_adalam_filter() -> Tuple[Any, str]:
    """Load the exact AdaLAM source bundled with this project.

    The implementation is loaded from ``third_party/adalam`` using a private
    package name. This prevents an installed ``adalam`` or Kornia package from
    silently replacing the user-provided source code.
    """
    global _BUNDLED_ADALAM_CACHE
    if _BUNDLED_ADALAM_CACHE is not None:
        return _BUNDLED_ADALAM_CACHE

    package_dir = Path(__file__).resolve().parent / "third_party" / "adalam"
    init_file = package_dir / "__init__.py"
    required = ["adalam.py", "core.py", "ransac.py", "utils.py", "__init__.py"]
    missing = [name for name in required if not (package_dir / name).is_file()]
    if missing:
        raise ImportError(
            "Bundled AdaLAM source is incomplete. Missing: " + ", ".join(missing)
        )

    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "The bundled AdaLAM source requires PyTorch. Install it with "
            "`python -m pip install torch`, then rerun the benchmark."
        ) from exc

    module_name = "_edgefusion_bundled_adalam"
    module = sys.modules.get(module_name)
    if module is None:
        spec = importlib.util.spec_from_file_location(
            module_name,
            init_file,
            submodule_search_locations=[str(package_dir)],
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not create an import specification for {init_file}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(module_name, None)
            raise

    filter_class = getattr(module, "AdalamFilter", None)
    if filter_class is None:
        raise ImportError(f"AdalamFilter was not found in bundled source: {init_file}")

    backend = f"bundled_source:{package_dir}"
    _BUNDLED_ADALAM_CACHE = (filter_class, backend)
    return _BUNDLED_ADALAM_CACHE


def _resolve_adalam_device(requested: str) -> Any:
    """Resolve a PyTorch device while keeping CPU comparison reproducible."""
    import torch

    value = str(requested or "cpu").strip().lower()
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    if value.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"AdaLAM device '{requested}' was requested, but CUDA is unavailable. "
            "Use --device cpu or install a CUDA-enabled PyTorch build."
        )
    return torch.device(value)


def _synchronize_torch(device: Any) -> None:
    """Synchronize CUDA so the reported AdaLAM time is not under-measured."""
    import torch

    if getattr(device, "type", "cpu") == "cuda":
        torch.cuda.synchronize(device)


def _adalam_descriptors(descriptors: np.ndarray, method: str) -> Tuple[np.ndarray, str]:
    """Prepare descriptors for the supplied AdaLAM implementation.

    ``adalam_sift`` passes OpenCV SIFT descriptors directly, matching the
    provided example. ``adalam_orb`` reproduces the original project notebook
    by passing raw ORB bytes; the supplied AdaLAM code converts them to float.
    ``adalam_orb_bits`` is an additional corrected variant that unpacks ORB
    bytes into bits, making squared Euclidean distance equivalent to Hamming
    distance for binary descriptors.
    """
    if method == "adalam_orb_bits":
        return np.unpackbits(np.asarray(descriptors, dtype=np.uint8), axis=1).astype(np.float32), "orb_unpacked_bits"
    if method == "adalam_orb":
        return np.asarray(descriptors, dtype=np.uint8), "orb_raw_bytes_original_notebook"
    return np.asarray(descriptors, dtype=np.float32), "sift_raw_float"


def run_adalam(
    method: str,
    gray1: np.ndarray,
    gray2: np.ndarray,
    camera: CameraModel,
    cfg: PipelineConfig,
    device: str,
) -> MethodResult:
    detector = "sift" if method == "adalam_sift" else "orb"
    result = MethodResult(method=method, success=False, status="started")
    try:
        AdalamFilter, backend = _load_adalam_filter()
        torch_device = _resolve_adalam_device(device)
    except Exception as exc:
        result.status = "adalam_source_unavailable"
        result.diagnostics["error"] = str(exc)
        return result

    with StageTimer(result.timings_ms, "extract_ms"):
        f1 = extract_standard(gray1, detector, cfg.nfeatures)
        f2 = extract_standard(gray2, detector, cfg.nfeatures)
    result.num_keypoints1 = len(f1.keypoints)
    result.num_keypoints2 = len(f2.keypoints)
    result.diagnostics.update(
        detector=detector.upper(),
        adalam_backend=backend,
        adalam_device=str(torch_device),
        adalam_source="uploaded_adalam_folder",
    )
    if f1.descriptors is None or f2.descriptors is None:
        result.status = "no_descriptors"
        return result
    if len(f1.descriptors) < 1 or len(f2.descriptors) < 2:
        result.status = "insufficient_descriptors_for_adalam_knn"
        return result

    def adalam_data(f: FeatureSet) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pts = np.ascontiguousarray(f.points, dtype=np.float32)
        ori = np.asarray([kp.angle for kp in f.keypoints], np.float32)
        scale = np.asarray([kp.size for kp in f.keypoints], np.float32)
        return pts, ori, scale

    p1, o1, s1 = adalam_data(f1)
    p2, o2, s2 = adalam_data(f2)
    d1, descriptor_mode = _adalam_descriptors(f1.descriptors, method)
    d2, _ = _adalam_descriptors(f2.descriptors, method)
    result.diagnostics["adalam_descriptor_mode"] = descriptor_mode
    # match_and_filter creates one nearest-neighbour putative match for every
    # source descriptor before applying AdaLAM's local-affine filter.
    result.num_raw_matches = int(len(d1))

    try:
        matcher = AdalamFilter(custom_config={"device": torch_device})
        _synchronize_torch(torch_device)
        with StageTimer(result.timings_ms, "match_filter_ms"):
            matches = matcher.match_and_filter(
                k1=p1,
                k2=p2,
                o1=o1,
                o2=o2,
                d1=d1,
                d2=d2,
                s1=s1,
                s2=s2,
                im1shape=gray1.shape[:2],
                im2shape=gray2.shape[:2],
            )
            _synchronize_torch(torch_device)
        if hasattr(matches, "detach"):
            matches = matches.detach()
        if hasattr(matches, "cpu"):
            matches = matches.cpu().numpy()
        pairs = np.asarray(matches, dtype=np.int32).reshape(-1, 2)
    except Exception as exc:
        result.status = "adalam_runtime_error"
        result.diagnostics["error"] = str(exc)
        return result

    # Defensive validation protects subsequent indexing if third-party output
    # is malformed or contains out-of-range indices.
    if len(pairs):
        valid = (
            (pairs[:, 0] >= 0)
            & (pairs[:, 0] < len(f1.points))
            & (pairs[:, 1] >= 0)
            & (pairs[:, 1] < len(f2.points))
        )
        invalid_count = int(len(pairs) - np.count_nonzero(valid))
        if invalid_count:
            result.diagnostics["adalam_invalid_pairs_removed"] = invalid_count
            pairs = pairs[valid]

    result.num_filtered_matches = len(pairs)
    result.diagnostics["adalam_retain_ratio"] = len(pairs) / max(result.num_raw_matches, 1)
    if len(pairs) < cfg.min_pose_matches:
        result.status = "insufficient_matches"
        return result

    result.points1 = np.ascontiguousarray(f1.points[pairs[:, 0]], dtype=np.float32)
    result.points2 = np.ascontiguousarray(f2.points[pairs[:, 1]], dtype=np.float32)
    with StageTimer(result.timings_ms, "geometry_ms"):
        R_est, t_est, mask, diag = estimate_pose(result.points1, result.points2, camera, cfg)
    result.diagnostics.update(diag)
    result.R, result.t, result.geometric_inlier_mask = R_est, t_est, mask
    result.num_geometric_inliers = int(np.count_nonzero(mask))
    result.success = R_est is not None and result.num_geometric_inliers >= cfg.min_pose_matches
    result.status = "ok" if result.success else str(diag.get("pose_status", "failed"))
    return result


def _homography_score(
    H21: Optional[np.ndarray], points1: np.ndarray, points2: np.ndarray, sigma: float = 1.0
) -> Tuple[float, np.ndarray]:
    n = len(points1)
    if H21 is None or n == 0:
        return 0.0, np.zeros(n, dtype=bool)
    try:
        H12 = np.linalg.inv(H21)
    except np.linalg.LinAlgError:
        return 0.0, np.zeros(n, dtype=bool)
    p1 = points1.reshape(-1, 1, 2).astype(np.float64)
    p2 = points2.reshape(-1, 1, 2).astype(np.float64)
    p1_to_2 = cv.perspectiveTransform(p1, H21).reshape(-1, 2)
    p2_to_1 = cv.perspectiveTransform(p2, H12).reshape(-1, 2)
    inv_sigma2 = 1.0 / max(sigma * sigma, EPS)
    chi12 = np.sum((points2 - p1_to_2) ** 2, axis=1) * inv_sigma2
    chi21 = np.sum((points1 - p2_to_1) ** 2, axis=1) * inv_sigma2
    threshold = 5.991
    keep = np.isfinite(chi12) & np.isfinite(chi21) & (chi12 <= threshold) & (chi21 <= threshold)
    score = float(np.sum((threshold - chi12[keep]) + (threshold - chi21[keep])))
    return score, keep


def _fundamental_score(
    F21: Optional[np.ndarray], points1: np.ndarray, points2: np.ndarray, sigma: float = 1.0
) -> Tuple[float, np.ndarray]:
    n = len(points1)
    if F21 is None or n == 0:
        return 0.0, np.zeros(n, dtype=bool)
    x1 = np.column_stack([points1, np.ones(n, dtype=np.float64)])
    x2 = np.column_stack([points2, np.ones(n, dtype=np.float64)])
    l2 = (F21 @ x1.T).T
    l1 = (F21.T @ x2.T).T
    num = np.sum(x2 * l2, axis=1)
    d2 = num * num / np.maximum(l2[:, 0] ** 2 + l2[:, 1] ** 2, EPS)
    d1 = num * num / np.maximum(l1[:, 0] ** 2 + l1[:, 1] ** 2, EPS)
    inv_sigma2 = 1.0 / max(sigma * sigma, EPS)
    chi1 = d1 * inv_sigma2
    chi2 = d2 * inv_sigma2
    inlier_threshold = 3.841
    score_threshold = 5.991
    keep = np.isfinite(chi1) & np.isfinite(chi2) & (chi1 <= inlier_threshold) & (chi2 <= inlier_threshold)
    score = float(np.sum((score_threshold - chi1[keep]) + (score_threshold - chi2[keep])))
    return score, keep


def _recover_pose_from_fundamental(
    F: np.ndarray,
    points1: np.ndarray,
    points2: np.ndarray,
    mask: np.ndarray,
    camera: CameraModel,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
    E = camera.K.T @ F @ camera.K
    cv_mask = mask.astype(np.uint8).reshape(-1, 1)
    try:
        _, R, t, pose_mask = cv.recoverPose(
            E,
            points1.astype(np.float64),
            points2.astype(np.float64),
            camera.K,
            mask=cv_mask,
        )
    except cv.error:
        return None, None, np.zeros(len(points1), dtype=bool)
    return R, safe_normalize(t.reshape(3)), pose_mask.reshape(-1).astype(bool)


def _recover_pose_from_homography(
    H: np.ndarray,
    points1: np.ndarray,
    points2: np.ndarray,
    mask: np.ndarray,
    camera: CameraModel,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
    try:
        count, rotations, translations, _ = cv.decomposeHomographyMat(H, camera.K)
    except cv.error:
        return None, None, np.zeros(len(points1), dtype=bool)
    if count <= 0:
        return None, None, np.zeros(len(points1), dtype=bool)

    active = np.flatnonzero(mask)
    if len(active) < 4:
        return None, None, np.zeros(len(points1), dtype=bool)
    p1 = points1[active].reshape(-1, 1, 2).astype(np.float64)
    p2 = points2[active].reshape(-1, 1, 2).astype(np.float64)
    p1n = cv.undistortPoints(p1, camera.K, None).reshape(-1, 2)
    p2n = cv.undistortPoints(p2, camera.K, None).reshape(-1, 2)
    P1 = np.column_stack([np.eye(3), np.zeros(3)])

    best: Optional[Tuple[int, np.ndarray, np.ndarray, np.ndarray]] = None
    for R, t in zip(rotations, translations):
        t = safe_normalize(np.asarray(t, dtype=np.float64).reshape(3))
        P2 = np.column_stack([np.asarray(R, dtype=np.float64), t])
        try:
            Xh = cv.triangulatePoints(P1, P2, p1n.T, p2n.T)
        except cv.error:
            continue
        valid_w = np.abs(Xh[3]) > EPS
        X = np.zeros((len(active), 3), dtype=np.float64)
        X[valid_w] = (Xh[:3, valid_w] / Xh[3:4, valid_w]).T
        X2 = (np.asarray(R, dtype=np.float64) @ X.T + t.reshape(3, 1)).T
        positive = valid_w & np.isfinite(X).all(axis=1) & (X[:, 2] > 0) & (X2[:, 2] > 0)
        score = int(np.count_nonzero(positive))
        if best is None or score > best[0]:
            full_mask = np.zeros(len(points1), dtype=bool)
            full_mask[active[positive]] = True
            best = (score, np.asarray(R, dtype=np.float64), t, full_mask)
    if best is None:
        return None, None, np.zeros(len(points1), dtype=bool)
    return best[1], best[2], best[3]


def run_orbslam_adaptive_fh(
    gray1: np.ndarray,
    gray2: np.ndarray,
    camera: CameraModel,
    cfg: PipelineConfig,
) -> MethodResult:
    """ORB-SLAM-inspired adaptive homography/fundamental model selection.

    It follows the documented ORB-SLAM initializer principle: estimate H and F,
    score both with symmetric chi-square residuals, and select H when
    SH / (SH + SF) > 0.40. This is a Python research reimplementation, not the
    original GPL C++ initializer.
    """
    method = "orbslam_adaptive_fh"
    result = MethodResult(method=method, success=False, status="started")
    with StageTimer(result.timings_ms, "extract_ms"):
        f1 = extract_standard(gray1, "orb", cfg.nfeatures)
        f2 = extract_standard(gray2, "orb", cfg.nfeatures)
    result.num_keypoints1 = len(f1.keypoints)
    result.num_keypoints2 = len(f2.keypoints)

    with StageTimer(result.timings_ms, "match_ms"):
        pairs, distances, ratios = mutual_ratio_matches(
            f1, f2, fixed_ratio=0.90, adaptive=False, cfg=cfg
        )
    result.num_raw_matches = len(pairs)
    result.num_filtered_matches = len(pairs)
    result.diagnostics.update(
        detector="ORB",
        match_policy="mutual_ratio_then_adaptive_FH",
        ratio_threshold=0.90,
    )
    if len(distances):
        result.diagnostics["median_descriptor_distance"] = float(np.median(distances))
    if len(ratios):
        result.diagnostics["median_ratio"] = float(np.median(ratios))
    if len(pairs) < cfg.min_pose_matches:
        result.status = "insufficient_matches"
        return result

    points1 = np.ascontiguousarray(f1.points[pairs[:, 0]], dtype=np.float64)
    points2 = np.ascontiguousarray(f2.points[pairs[:, 1]], dtype=np.float64)
    # Estimate both models on undistorted pixel coordinates.
    if cfg.use_distortion_correction:
        model_p1 = cv.undistortPoints(points1.reshape(-1, 1, 2), camera.K, camera.dist, P=camera.K).reshape(-1, 2)
        model_p2 = cv.undistortPoints(points2.reshape(-1, 1, 2), camera.K, camera.dist, P=camera.K).reshape(-1, 2)
    else:
        model_p1, model_p2 = points1, points2

    method_flag = getattr(cv, "USAC_MAGSAC", cv.RANSAC)
    with StageTimer(result.timings_ms, "adaptive_fh_ms"):
        try:
            H, _ = cv.findHomography(
                model_p1, model_p2, method_flag, 3.0,
                maxIters=cfg.max_iters, confidence=cfg.confidence,
            )
        except cv.error:
            H, _ = cv.findHomography(model_p1, model_p2, cv.RANSAC, 3.0)
        try:
            F, _ = cv.findFundamentalMat(
                model_p1, model_p2, method_flag, 1.5, cfg.confidence, cfg.max_iters
            )
        except cv.error:
            F, _ = cv.findFundamentalMat(model_p1, model_p2, cv.FM_RANSAC, 1.5, cfg.confidence)
        if F is not None and F.shape != (3, 3):
            F = np.asarray(F).reshape(-1, 3, 3)[0]
        score_h, mask_h = _homography_score(H, model_p1, model_p2)
        score_f, mask_f = _fundamental_score(F, model_p1, model_p2)
        denom = score_h + score_f
        ratio_h = score_h / denom if denom > EPS else 0.0
        selected = "H" if ratio_h > 0.40 and H is not None else "F"

        R_est: Optional[np.ndarray]
        t_est: Optional[np.ndarray]
        pose_mask: np.ndarray
        if selected == "H" and H is not None:
            R_est, t_est, pose_mask = _recover_pose_from_homography(
                H, model_p1, model_p2, mask_h, camera
            )
            if R_est is None and F is not None:
                selected = "F_fallback"
                R_est, t_est, pose_mask = _recover_pose_from_fundamental(
                    F, model_p1, model_p2, mask_f, camera
                )
        elif F is not None:
            R_est, t_est, pose_mask = _recover_pose_from_fundamental(
                F, model_p1, model_p2, mask_f, camera
            )
        else:
            R_est, t_est, pose_mask = None, None, np.zeros(len(points1), dtype=bool)

    result.points1 = points1.astype(np.float32)
    result.points2 = points2.astype(np.float32)
    result.R, result.t = R_est, t_est
    result.geometric_inlier_mask = pose_mask
    result.num_geometric_inliers = int(np.count_nonzero(pose_mask))
    result.diagnostics.update(
        selected_model=selected,
        homography_score=score_h,
        fundamental_score=score_f,
        homography_score_ratio=ratio_h,
        homography_inliers=int(np.count_nonzero(mask_h)),
        fundamental_inliers=int(np.count_nonzero(mask_f)),
        estimator="adaptive_HF",
    )
    result.success = R_est is not None and t_est is not None and result.num_geometric_inliers >= cfg.min_pose_matches
    result.status = "ok" if result.success else "adaptive_fh_pose_failed"
    return result


def _to_torch_image(gray: np.ndarray, device: str, batched: bool = True):
    import torch

    rgb = cv.cvtColor(gray, cv.COLOR_GRAY2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    if batched:
        tensor = tensor[None]
    return tensor.to(device)


def run_xfeat(
    gray1: np.ndarray,
    gray2: np.ndarray,
    camera: CameraModel,
    cfg: PipelineConfig,
    xfeat_repo: Optional[Path],
    device: str,
) -> MethodResult:
    result = MethodResult(method="xfeat", success=False, status="started")
    try:
        import torch
        if xfeat_repo is not None:
            repo_str = str(xfeat_repo.resolve())
            if repo_str not in sys.path:
                sys.path.insert(0, repo_str)
        from modules.xfeat import XFeat
    except Exception as exc:
        result.status = "skipped_xfeat_not_installed"
        result.diagnostics["error"] = str(exc)
        return result

    try:
        cache_key = (str(xfeat_repo.resolve()) if xfeat_repo is not None else "sys.path", device)
        if cache_key not in _XFEAT_CACHE:
            with StageTimer(result.timings_ms, "model_init_ms"):
                _XFEAT_CACHE[cache_key] = XFeat().to(device).eval()
        model = _XFEAT_CACHE[cache_key]
        with StageTimer(result.timings_ms, "model_and_match_ms"):
            x1 = _to_torch_image(gray1, device)
            x2 = _to_torch_image(gray2, device)
            with torch.inference_mode():
                mk1, mk2 = model.match_xfeat(x1, x2, top_k=cfg.nfeatures)
            if hasattr(mk1, "detach"):
                mk1 = mk1.detach().cpu().numpy()
                mk2 = mk2.detach().cpu().numpy()
        result.points1 = np.asarray(mk1, dtype=np.float32).reshape(-1, 2)
        result.points2 = np.asarray(mk2, dtype=np.float32).reshape(-1, 2)
    except TypeError:
        # Compatibility with earlier XFeat versions without top_k in match_xfeat.
        with StageTimer(result.timings_ms, "model_and_match_ms"):
            model = _XFEAT_CACHE[cache_key]
            x1 = _to_torch_image(gray1, device)
            x2 = _to_torch_image(gray2, device)
            with torch.inference_mode():
                mk1, mk2 = model.match_xfeat(x1, x2)
            result.points1 = np.asarray(mk1.detach().cpu(), dtype=np.float32).reshape(-1, 2)
            result.points2 = np.asarray(mk2.detach().cpu(), dtype=np.float32).reshape(-1, 2)
    except Exception as exc:
        result.status = "xfeat_runtime_error"
        result.diagnostics["error"] = str(exc)
        return result

    result.num_keypoints1 = len(result.points1)
    result.num_keypoints2 = len(result.points2)
    result.num_raw_matches = len(result.points1)
    result.num_filtered_matches = len(result.points1)
    if len(result.points1) < cfg.min_pose_matches:
        result.status = "insufficient_matches"
        return result

    with StageTimer(result.timings_ms, "geometry_ms"):
        R_est, t_est, mask, diag = estimate_pose(result.points1, result.points2, camera, cfg)
    result.diagnostics.update(diag)
    result.R, result.t, result.geometric_inlier_mask = R_est, t_est, mask
    result.num_geometric_inliers = int(np.count_nonzero(mask))
    result.success = R_est is not None and result.num_geometric_inliers >= cfg.min_pose_matches
    result.status = "ok" if result.success else str(diag.get("pose_status", "failed"))
    return result


def run_lightglue_sift(
    gray1: np.ndarray,
    gray2: np.ndarray,
    camera: CameraModel,
    cfg: PipelineConfig,
    device: str,
) -> MethodResult:
    """Optional SIFT + LightGlue comparison using the official API."""
    result = MethodResult(method="lightglue_sift", success=False, status="started")
    try:
        import torch
        from lightglue import LightGlue, SIFT
        from lightglue.utils import rbd
    except Exception as exc:
        result.status = "skipped_lightglue_not_installed"
        result.diagnostics["error"] = str(exc)
        return result

    cache_key = (device, cfg.nfeatures)
    try:
        if cache_key not in _LIGHTGLUE_CACHE:
            with StageTimer(result.timings_ms, "model_init_ms"):
                extractor = SIFT(max_num_keypoints=cfg.nfeatures).eval().to(device)
                matcher = LightGlue(
                    features="sift",
                    depth_confidence=0.95,
                    width_confidence=0.99,
                    filter_threshold=0.1,
                ).eval().to(device)
                _LIGHTGLUE_CACHE[cache_key] = (extractor, matcher)
        extractor, matcher = _LIGHTGLUE_CACHE[cache_key]

        image0 = _to_torch_image(gray1, device, batched=False)
        image1 = _to_torch_image(gray2, device, batched=False)
        with StageTimer(result.timings_ms, "extract_ms"):
            with torch.inference_mode():
                feats0 = extractor.extract(image0)
                feats1 = extractor.extract(image1)
        with StageTimer(result.timings_ms, "match_ms"):
            with torch.inference_mode():
                matches01 = matcher({"image0": feats0, "image1": feats1})
                feats0, feats1, matches01 = [rbd(x) for x in (feats0, feats1, matches01)]
                matches = matches01["matches"]
                keypoints0 = feats0["keypoints"]
                keypoints1 = feats1["keypoints"]
                points0 = keypoints0[matches[..., 0]].detach().cpu().numpy()
                points1 = keypoints1[matches[..., 1]].detach().cpu().numpy()
        result.num_keypoints1 = int(len(keypoints0))
        result.num_keypoints2 = int(len(keypoints1))
        result.points1 = np.asarray(points0, dtype=np.float32).reshape(-1, 2)
        result.points2 = np.asarray(points1, dtype=np.float32).reshape(-1, 2)
    except Exception as exc:
        result.status = "lightglue_runtime_error"
        result.diagnostics["error"] = str(exc)
        return result

    result.num_raw_matches = len(result.points1)
    result.num_filtered_matches = len(result.points1)
    if len(result.points1) < cfg.min_pose_matches:
        result.status = "insufficient_matches"
        return result
    with StageTimer(result.timings_ms, "geometry_ms"):
        R_est, t_est, mask, diag = estimate_pose(result.points1, result.points2, camera, cfg)
    result.diagnostics.update(diag)
    result.R, result.t, result.geometric_inlier_mask = R_est, t_est, mask
    result.num_geometric_inliers = int(np.count_nonzero(mask))
    result.success = R_est is not None and result.num_geometric_inliers >= cfg.min_pose_matches
    result.status = "ok" if result.success else str(diag.get("pose_status", "failed"))
    return result


def make_ablation_config(base: PipelineConfig, method: str) -> PipelineConfig:
    values = asdict(base)
    if method == "edgefusion_fast":
        values["oversample_factor"] = 1.15
        values["use_clahe"] = False
        values["use_edge_corners"] = False
        values["use_orientation_scale_filter"] = False
        values["use_edge_orientation_filter"] = False
        values["use_lk_refinement"] = False
    elif method == "edgefusion_no_edges":
        values["use_edge_corners"] = False
        values["use_edge_weighting"] = False
        values["use_edge_orientation_filter"] = False
    elif method == "edgefusion_no_grid":
        values["use_grid_motion_filter"] = False
    elif method == "edgefusion_no_lk":
        values["use_lk_refinement"] = False
    elif method == "edgefusion_lite":
        values["use_lk_refinement"] = False
        values["use_orientation_scale_filter"] = False
        values["use_edge_orientation_filter"] = False
    return PipelineConfig(**values)


def run_method(
    method: str,
    gray1: np.ndarray,
    gray2: np.ndarray,
    camera: CameraModel,
    cfg: PipelineConfig,
    measure_memory: bool,
    xfeat_repo: Optional[Path],
    device: str,
) -> MethodResult:
    local_cfg = make_ablation_config(cfg, method)
    with PeakRSSMonitor(measure_memory) as memory:
        if method in CLASSICAL_METHODS:
            result = run_baseline(method, gray1, gray2, camera, local_cfg)
        elif method == "edgefusion_simple":
            result = run_edgefusion_simple(gray1, gray2, camera, local_cfg)
        elif method == "edgefusion_adaptive":
            result = run_edgefusion_adaptive(gray1, gray2, camera, local_cfg)
        elif method == "edgefusion_adaptive_guarded":
            result = run_edgefusion_adaptive_guarded(
                gray1, gray2, camera, local_cfg
            )
        elif method == "edgefusion_adaptive_poseguard":
            result = run_edgefusion_adaptive_poseguard(
                gray1, gray2, camera, local_cfg
            )
        elif method == "edgefusion_adaptive_poseguard_v3":
            result = run_edgefusion_adaptive_poseguard_v3(
                gray1, gray2, camera, local_cfg
            )
        elif method == "edgefusion_adaptive_poseguard_v4":
            result = run_edgefusion_adaptive_poseguard_v4(
                gray1, gray2, camera, local_cfg
            )
        elif method.startswith("edgefusion"):
            result = run_edgefusion(method, gray1, gray2, camera, local_cfg)
        elif method in {"adalam_sift", "adalam_orb", "adalam_orb_bits"}:
            result = run_adalam(method, gray1, gray2, camera, local_cfg, device)
        elif method == "orbslam_adaptive_fh":
            result = run_orbslam_adaptive_fh(gray1, gray2, camera, local_cfg)
        elif method == "xfeat":
            result = run_xfeat(gray1, gray2, camera, local_cfg, xfeat_repo, device)
        elif method == "lightglue_sift":
            result = run_lightglue_sift(gray1, gray2, camera, local_cfg, device)
        else:
            raise ValueError(f"Unknown method: {method}")
    result.peak_rss_delta_mb = memory.delta_mb
    return result


# ---------------------------------------------------------------------------
# Reporting and visualization
# ---------------------------------------------------------------------------


# Human-readable labels used in the consolidated CSV/Excel tables.
METHOD_METADATA: Dict[str, Tuple[str, str]] = {
    "sift_nn": ("Original baseline", "SIFT + raw nearest neighbour"),
    "orb_nn": ("Original baseline", "ORB + raw nearest neighbour"),
    "sift_lowe": ("Standard baseline", "SIFT + one-way Lowe ratio (0.75)"),
    "orb_lowe": ("Standard baseline", "ORB + one-way Lowe ratio (0.80)"),
    "sift_mutual_lowe": ("Original filter", "SIFT + bidirectional Lowe ratio + mutual NN (0.95)"),
    "orb_mutual_lowe": ("Original filter", "ORB + bidirectional Lowe ratio + mutual NN (0.95)"),
    "orb_crosscheck": ("Original baseline", "ORB + BF Hamming cross-check"),
    "akaze_lowe": ("Standard baseline", "AKAZE + one-way Lowe ratio"),
    "adalam_sift": ("Original filter", "SIFT + bundled original AdaLAM source"),
    "adalam_orb": ("Original filter", "ORB raw bytes + bundled AdaLAM; exact original-notebook descriptor handling"),
    "adalam_orb_bits": ("Corrected variant", "ORB unpacked bits + bundled AdaLAM; Hamming-equivalent descriptor distance"),
    "orbslam_adaptive_fh": ("ORB-SLAM-inspired", "ORB + adaptive homography/fundamental selection"),
    "edgefusion_fast": ("Proposed", "EdgeFusion fast profile"),
    "edgefusion_full": ("Proposed", "EdgeFusion full profile"),
    "edgefusion_simple": (
        "Proposed efficiency variant",
        "EdgeFusion-Simple: ORB + edge-proximity re-ranking + spatial quota + one-way Lowe + orientation/scale",
    ),
    "edgefusion_adaptive": (
        "Proposed efficiency variant",
        "EdgeFusion-Adaptive: ORB-Lowe fast path with conditional structural refinement and selective LK",
    ),
    "edgefusion_adaptive_guarded": (
        "Proposed efficiency variant",
        "EdgeFusion-Adaptive-Guarded: Adaptive cascade with soft early acceptance, pre-LK early exit, and shared-quality model selection",
    ),
    "edgefusion_adaptive_poseguard": (
        "Proposed efficiency variant",
        "EdgeFusion-Adaptive-PoseGuard-v2: Adaptive cascade with cheirality/parallax-aware pose selection",
    ),
    "edgefusion_adaptive_poseguard_v3": (
        "Proposed efficiency variant",
        "EdgeFusion-Adaptive-PoseGuard-v3: observability-gated selection, gain-based cheirality rescue, and conservative LK early stop",
    ),
    "edgefusion_adaptive_poseguard_v4": (
        "Proposed efficiency variant",
        "EdgeFusion-Adaptive-PoseGuard-v4: v3 observability/rescue plus hierarchical LK pose-stability gating",
    ),
    "edgefusion_lite": ("Ablation", "EdgeFusion lite profile"),
    "edgefusion_no_edges": ("Ablation", "EdgeFusion without edge guidance"),
    "edgefusion_no_grid": ("Ablation", "EdgeFusion without grid-motion filter"),
    "edgefusion_no_lk": ("Ablation", "EdgeFusion without LK refinement"),
    "xfeat": ("Modern optional", "XFeat sparse matching"),
    "lightglue_sift": ("Modern optional", "SIFT + LightGlue"),
}


def method_metadata(method: str) -> Tuple[str, str]:
    return METHOD_METADATA.get(method, ("Other", method))


def _array_json(value: Optional[np.ndarray]) -> str:
    if value is None:
        return ""
    return json.dumps(np.asarray(value, dtype=float).round(10).tolist())


def result_to_row(pair: PairRecord, result: MethodResult, R_gt: np.ndarray, t_gt: np.ndarray) -> Dict[str, Any]:
    family, description = method_metadata(result.method)
    row: Dict[str, Any] = {
        "pair_index": pair.pair_index,
        "timestamp1": pair.first.timestamp,
        "timestamp2": pair.second.timestamp,
        "delta_t_s": pair.second.timestamp - pair.first.timestamp,
        "image1": str(pair.first.path),
        "image2": str(pair.second.path),
        "method": result.method,
        "method_family": family,
        "method_description": description,
        "success": int(result.success),
        "status": result.status,
        "keypoints1": result.num_keypoints1,
        "keypoints2": result.num_keypoints2,
        "raw_matches": result.num_raw_matches,
        "filtered_matches": result.num_filtered_matches,
        "geometric_inliers": result.num_geometric_inliers,
        "inlier_ratio": result.inlier_ratio,
        "total_ms": result.total_ms,
        "peak_rss_delta_mb": result.peak_rss_delta_mb,
        "gt_rotation_matrix": _array_json(R_gt),
        "gt_translation_direction": _array_json(t_gt),
        "estimated_rotation_matrix": _array_json(result.R),
        "estimated_translation_direction": _array_json(result.t),
    }
    for key, value in result.timings_ms.items():
        row[key] = value
    if result.success and result.R is not None and result.t is not None:
        row["rotation_error_deg"] = rotation_error_deg(result.R, R_gt)
        row["translation_error_deg"] = vector_angle_deg(result.t, t_gt)
        row["translation_error_sign_invariant_deg"] = min(
            vector_angle_deg(result.t, t_gt),
            vector_angle_deg(-result.t, t_gt),
        )
    else:
        row["rotation_error_deg"] = float("nan")
        row["translation_error_deg"] = float("nan")
        row["translation_error_sign_invariant_deg"] = float("nan")
    for key, value in result.diagnostics.items():
        if isinstance(value, (str, int, float, bool, np.number)):
            row[f"diag_{key}"] = value
    return row


def save_raw_correspondence_result(
    root: Path,
    pair: PairRecord,
    result: MethodResult,
    R_gt: np.ndarray,
    t_gt: np.ndarray,
) -> Path:
    """Save the exact correspondences and poses used by one method/pair.

    The compressed NPZ is the raw source for custom correspondence, residual,
    spatial-coverage, and pose visualisations. It contains both successful and
    failed outputs so failures are auditable rather than silently omitted.
    """
    pair_dir = root / f"pair_{pair.pair_index:04d}"
    pair_dir.mkdir(parents=True, exist_ok=True)
    path = pair_dir / f"{result.method}.npz"
    np.savez_compressed(
        path,
        method=np.asarray(result.method),
        status=np.asarray(result.status),
        success=np.asarray(int(result.success), dtype=np.int8),
        points1=np.asarray(result.points1, dtype=np.float32),
        points2=np.asarray(result.points2, dtype=np.float32),
        geometric_inlier_mask=np.asarray(result.geometric_inlier_mask, dtype=bool),
        R_est=np.asarray(result.R if result.R is not None else np.empty((0, 0)), dtype=np.float64),
        t_est=np.asarray(result.t if result.t is not None else np.empty(0), dtype=np.float64),
        R_gt=np.asarray(R_gt, dtype=np.float64),
        t_gt=np.asarray(t_gt, dtype=np.float64),
        timings_json=np.asarray(json.dumps(result.timings_ms, sort_keys=True)),
        diagnostics_json=np.asarray(json.dumps(result.diagnostics, default=str, sort_keys=True)),
        image1=np.asarray(str(pair.first.path)),
        image2=np.asarray(str(pair.second.path)),
        timestamp1=np.asarray(pair.first.timestamp, dtype=np.float64),
        timestamp2=np.asarray(pair.second.timestamp, dtype=np.float64),
    )
    return path


def write_runtime_environment_snapshot(output_dir: Path) -> None:
    packages = {}
    for name in [
        "numpy", "scipy", "pandas", "matplotlib", "opencv-contrib-python",
        "torch", "torchvision", "kornia", "onnxruntime", "openpyxl", "psutil",
    ]:
        try:
            packages[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            packages[name] = None
    payload = {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "opencv_version": cv.__version__,
        "packages": packages,
    }
    (output_dir / "runtime_environment.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    installed = sorted(
        f"{dist.metadata.get('Name', dist.metadata.get('Summary', 'unknown'))}=={dist.version}"
        for dist in importlib_metadata.distributions()
        if dist.metadata.get("Name")
    )
    (output_dir / "installed_packages.txt").write_text("\n".join(installed) + "\n", encoding="utf-8")


def draw_result(
    image1: np.ndarray,
    image2: np.ndarray,
    result: MethodResult,
    max_draw: int = 180,
    inliers_only: bool = True,
) -> np.ndarray:
    p1 = result.points1
    p2 = result.points2
    if len(p1) == 0:
        h = max(image1.shape[0], image2.shape[0])
        def pad(im: np.ndarray) -> np.ndarray:
            if im.shape[0] == h:
                return im
            return cv.copyMakeBorder(im, 0, h - im.shape[0], 0, 0, cv.BORDER_CONSTANT)
        return np.hstack([pad(image1), pad(image2)])
    if inliers_only and len(result.geometric_inlier_mask) == len(p1):
        idx = np.flatnonzero(result.geometric_inlier_mask)
    else:
        idx = np.arange(len(p1))
    if len(idx) > max_draw:
        idx = idx[np.linspace(0, len(idx) - 1, max_draw).astype(int)]
    kp1 = points_to_keypoints(p1[idx])
    kp2 = points_to_keypoints(p2[idx])
    matches = [cv.DMatch(i, i, 0.0) for i in range(len(idx))]
    return cv.drawMatches(
        image1,
        kp1,
        image2,
        kp2,
        matches,
        None,
        matchColor=(0, 220, 0) if inliers_only else (255, 180, 0),
        singlePointColor=(0, 0, 255),
        flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )


def aggregate_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    methods = sorted({str(r["method"]) for r in rows})
    summary: List[Dict[str, Any]] = []
    for method in methods:
        group = [r for r in rows if r["method"] == method]
        successful = [r for r in group if int(r["success"]) == 1]

        def vals(key: str, source: Sequence[Mapping[str, Any]] = group) -> List[float]:
            out = []
            for r in source:
                try:
                    v = float(r.get(key, float("nan")))
                except (TypeError, ValueError):
                    continue
                if math.isfinite(v):
                    out.append(v)
            return out

        def median_or_nan(key: str, source: Sequence[Mapping[str, Any]] = group) -> float:
            v = vals(key, source)
            return float(statistics.median(v)) if v else float("nan")

        def mean_or_nan(key: str, source: Sequence[Mapping[str, Any]] = group) -> float:
            v = vals(key, source)
            return float(statistics.fmean(v)) if v else float("nan")

        family, description = method_metadata(method)
        summary.append(
            {
                "method": method,
                "method_family": family,
                "method_description": description,
                "pairs": len(group),
                "successful_pairs": len(successful),
                "success_rate": len(successful) / max(len(group), 1),
                "median_total_ms": median_or_nan("total_ms"),
                "mean_total_ms": mean_or_nan("total_ms"),
                "median_peak_rss_delta_mb": median_or_nan("peak_rss_delta_mb"),
                "median_keypoints1": median_or_nan("keypoints1"),
                "median_raw_matches": median_or_nan("raw_matches"),
                "median_filtered_matches": median_or_nan("filtered_matches"),
                "median_geometric_inliers": median_or_nan("geometric_inliers"),
                "median_inlier_ratio": median_or_nan("inlier_ratio"),
                "median_rotation_error_deg": median_or_nan("rotation_error_deg", successful),
                "mean_rotation_error_deg": mean_or_nan("rotation_error_deg", successful),
                "median_translation_error_deg": median_or_nan("translation_error_deg", successful),
                "median_translation_error_sign_invariant_deg": median_or_nan(
                    "translation_error_sign_invariant_deg", successful
                ),
            }
        )
    return summary


def write_markdown_summary(path: Path, summary: Sequence[Mapping[str, Any]], args: argparse.Namespace) -> None:
    columns = [
        "method",
        "success_rate",
        "median_total_ms",
        "median_peak_rss_delta_mb",
        "median_geometric_inliers",
        "median_inlier_ratio",
        "median_rotation_error_deg",
        "median_translation_error_deg",
    ]
    lines = [
        "# Matching benchmark summary",
        "",
        f"Dataset: `{args.dataset}`  ",
        f"Frame step: `{args.frame_step}`; pair stride: `{args.pair_stride}`; pairs: `{args.max_pairs}`  ",
        f"OpenCV: `{cv.__version__}`; Python: `{platform.python_version()}`  ",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in summary:
        cells = []
        for c in columns:
            v = row.get(c, "")
            if isinstance(v, float):
                cells.append("nan" if not math.isfinite(v) else f"{v:.4f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Use the success rate, pose errors, runtime, and memory together. Do not select a method solely because it returns more matches. The strongest method should provide a high geometric-inlier ratio, low pose error, a low failure rate, and acceptable latency/resource use.",
            "",
            "`edgefusion_full` is the full proposed hybrid; `edgefusion_simple`, `edgefusion_adaptive`, `edgefusion_adaptive_guarded`, `edgefusion_adaptive_poseguard`, `edgefusion_adaptive_poseguard_v3`, and `edgefusion_adaptive_poseguard_v4` are efficiency-oriented proposed variants. PoseGuard-v4 adds hierarchical LK pose-stability gating while preserving all earlier variants unchanged. The `edgefusion_no_*` rows are ablations that quantify which components contribute to the result.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_plots(summary: Sequence[Mapping[str, Any]], output_dir: Path) -> None:
    if plt is None or not summary:
        return
    methods = [str(r["method"]) for r in summary]
    plots = [
        ("median_total_ms", "Median end-to-end time (ms)", "time_comparison.png"),
        ("median_rotation_error_deg", "Median rotation error (deg)", "rotation_error_comparison.png"),
        ("median_translation_error_deg", "Median translation-direction error (deg)", "translation_error_comparison.png"),
        ("median_inlier_ratio", "Median geometric inlier ratio", "inlier_ratio_comparison.png"),
    ]
    for key, ylabel, filename in plots:
        values = [float(r.get(key, float("nan"))) for r in summary]
        fig = plt.figure(figsize=(max(8, len(methods) * 1.2), 5))
        x = np.arange(len(methods))
        plt.bar(x, values)
        plt.xticks(x, methods, rotation=35, ha="right")
        plt.ylabel(ylabel)
        plt.tight_layout()
        fig.savefig(output_dir / filename, dpi=160)
        plt.close(fig)



def write_excel_workbook(path: Path, sheets: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
    """Write one workbook with consolidated tables when pandas/openpyxl are available."""
    if pd is None:
        return
    try:
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            for sheet_name, rows in sheets.items():
                pd.DataFrame(list(rows)).to_excel(writer, sheet_name=sheet_name[:31], index=False)
    except Exception as exc:
        print(f"[warning] Excel output was not written: {exc}", file=sys.stderr)


def pair_result_to_row(result: MethodResult) -> Dict[str, Any]:
    family, description = method_metadata(result.method)
    row: Dict[str, Any] = {
        "method": result.method,
        "method_family": family,
        "method_description": description,
        "success": int(result.success),
        "status": result.status,
        "total_ms": result.total_ms,
        "peak_rss_delta_mb": result.peak_rss_delta_mb,
        "keypoints1": result.num_keypoints1,
        "keypoints2": result.num_keypoints2,
        "raw_matches": result.num_raw_matches,
        "filtered_matches": result.num_filtered_matches,
        "geometric_inliers": result.num_geometric_inliers,
        "inlier_ratio": result.inlier_ratio,
        "rotation_matrix": "" if result.R is None else json.dumps(np.asarray(result.R).round(8).tolist()),
        "translation_direction": "" if result.t is None else json.dumps(np.asarray(result.t).round(8).tolist()),
    }
    for key, value in result.timings_ms.items():
        row[key] = value
    for key, value in result.diagnostics.items():
        if isinstance(value, (str, int, float, bool, np.number)):
            row[f"diag_{key}"] = value
    return row


def _annotate_tile(image: np.ndarray, result: MethodResult, subtitle: str) -> np.ndarray:
    canvas = image.copy()
    cv.rectangle(canvas, (0, 0), (canvas.shape[1], 66), (18, 18, 18), -1)
    cv.putText(
        canvas,
        result.method,
        (12, 25),
        cv.FONT_HERSHEY_SIMPLEX,
        0.68,
        (255, 255, 255),
        2,
        cv.LINE_AA,
    )
    cv.putText(
        canvas,
        subtitle,
        (12, 53),
        cv.FONT_HERSHEY_SIMPLEX,
        0.53,
        (0, 230, 255) if result.success else (80, 160, 255),
        1,
        cv.LINE_AA,
    )
    return canvas


def make_contact_sheet(
    tiles: Sequence[np.ndarray],
    columns: int = 2,
    tile_width: int = 1100,
) -> np.ndarray:
    if not tiles:
        return np.zeros((200, 400, 3), dtype=np.uint8)
    resized: List[np.ndarray] = []
    max_h = 0
    for tile in tiles:
        scale = tile_width / max(tile.shape[1], 1)
        h = max(1, int(round(tile.shape[0] * scale)))
        r = cv.resize(tile, (tile_width, h), interpolation=cv.INTER_AREA if scale < 1 else cv.INTER_LINEAR)
        resized.append(r)
        max_h = max(max_h, h)
    padded: List[np.ndarray] = []
    for tile in resized:
        bottom = max_h - tile.shape[0]
        padded.append(cv.copyMakeBorder(tile, 0, bottom, 0, 0, cv.BORDER_CONSTANT, value=(28, 28, 28)))
    rows = int(math.ceil(len(padded) / max(columns, 1)))
    blank = np.full((max_h, tile_width, 3), 28, dtype=np.uint8)
    while len(padded) < rows * columns:
        padded.append(blank.copy())
    row_images = [np.hstack(padded[r * columns : (r + 1) * columns]) for r in range(rows)]
    return np.vstack(row_images)


def _parse_distortion(value: str) -> Tuple[float, float, float, float, float]:
    parts = [x.strip() for x in value.split(",") if x.strip()]
    if not parts:
        return (0.0, 0.0, 0.0, 0.0, 0.0)
    if len(parts) != 5:
        raise ValueError("--distortion must contain five comma-separated values: k1,k2,p1,p2,k3")
    return tuple(float(x) for x in parts)  # type: ignore[return-value]


def _camera_for_pair(args: argparse.Namespace, image: np.ndarray, resize_scale: float) -> CameraModel:
    h, w = image.shape[:2]
    mode = args.pair_camera.lower()
    pair_aliases = {
        "tum_fr1": "freiburg1",
        "tum_fr2": "freiburg2",
        "tum_fr3": "freiburg3",
    }
    if mode in pair_aliases:
        return scale_camera(TUM_RGB_CAMERA_MODELS[pair_aliases[mode]], resize_scale)
    if mode == "custom":
        if args.fx is None or args.fy is None:
            raise ValueError("--pair-camera custom requires --fx and --fy")
        cx = args.cx if args.cx is not None else (w / max(resize_scale, EPS) - 1.0) * 0.5
        cy = args.cy if args.cy is not None else (h / max(resize_scale, EPS) - 1.0) * 0.5
        base = CameraModel(
            fx=float(args.fx),
            fy=float(args.fy),
            cx=float(cx),
            cy=float(cy),
            distortion=_parse_distortion(args.distortion),
        )
        return scale_camera(base, resize_scale)
    # Generic approximate calibration for visualization/relative-pose diagnostics.
    focal = 0.9 * max(w, h)
    return CameraModel(
        fx=float(focal),
        fy=float(focal),
        cx=float((w - 1) * 0.5),
        cy=float((h - 1) * 0.5),
        distortion=(0.0, 0.0, 0.0, 0.0, 0.0),
    )


def compare_two_images(args: argparse.Namespace) -> int:
    """Compare every selected method on two explicit images and save visual outputs."""
    np.random.seed(args.seed)
    cv.setRNGSeed(args.seed)
    cv.setNumThreads(args.cv_threads)

    image1_path = Path(args.image1).expanduser().resolve()
    image2_path = Path(args.image2).expanduser().resolve()
    image1 = cv.imread(str(image1_path), cv.IMREAD_COLOR)
    image2 = cv.imread(str(image2_path), cv.IMREAD_COLOR)
    if image1 is None:
        raise FileNotFoundError(f"Cannot read image 1: {image1_path}")
    if image2 is None:
        raise FileNotFoundError(f"Cannot read image 2: {image2_path}")

    image1, scale1 = resize_to_max_side(image1, args.max_side)
    image2, _ = resize_to_max_side(image2, args.max_side)
    camera = _camera_for_pair(args, image1, scale1)
    gray1, gray2 = ensure_gray(image1), ensure_gray(image2)
    cfg = PipelineConfig(
        nfeatures=args.nfeatures,
        max_side=args.max_side,
        ratio=args.ratio,
        magsac_threshold_px=args.magsac_threshold,
        use_distortion_correction=not args.no_distortion_correction,
        simple_feature_fraction=args.simple_feature_fraction,
        simple_edge_weight=args.simple_edge_weight,
        adaptive_accept_inliers=args.adaptive_accept_inliers,
        adaptive_accept_inlier_ratio=args.adaptive_accept_inlier_ratio,
        adaptive_accept_coverage=args.adaptive_accept_coverage,
        adaptive_fallback_feature_fraction=args.adaptive_fallback_feature_fraction,
        adaptive_lk_max_matches=args.adaptive_lk_max_matches,
        adaptive_guarded_soft_score=args.adaptive_guarded_soft_score,
        adaptive_guarded_soft_floor_fraction=args.adaptive_guarded_soft_floor_fraction,
        adaptive_guarded_quality_margin=args.adaptive_guarded_quality_margin,
        adaptive_guarded_prelk_margin=args.adaptive_guarded_prelk_margin,
        adaptive_poseguard_quality_margin=args.adaptive_poseguard_quality_margin,
        adaptive_poseguard_prelk_margin=args.adaptive_poseguard_prelk_margin,
        adaptive_poseguard_parallax_target_deg=args.adaptive_poseguard_parallax_target_deg,
        adaptive_poseguard_rescue_fast_cheirality_max=args.adaptive_poseguard_rescue_fast_cheirality_max,
        adaptive_poseguard_rescue_candidate_cheirality_min=args.adaptive_poseguard_rescue_candidate_cheirality_min,
        adaptive_poseguard_rescue_cheirality_gain=args.adaptive_poseguard_rescue_cheirality_gain,
        adaptive_poseguard_cheirality_drop_tolerance=args.adaptive_poseguard_cheirality_drop_tolerance,
        adaptive_poseguard_v3_quality_margin=args.adaptive_poseguard_v3_quality_margin,
        adaptive_poseguard_v3_prelk_margin=args.adaptive_poseguard_v3_prelk_margin,
        adaptive_poseguard_v3_min_candidate_native_cheirality=args.adaptive_poseguard_v3_min_candidate_native_cheirality,
        adaptive_poseguard_v3_min_candidate_parallax_deg=args.adaptive_poseguard_v3_min_candidate_parallax_deg,
        adaptive_poseguard_v3_rescue_candidate_native_cheirality_min=args.adaptive_poseguard_v3_rescue_candidate_native_cheirality_min,
        adaptive_poseguard_v3_rescue_native_cheirality_gain=args.adaptive_poseguard_v3_rescue_native_cheirality_gain,
        adaptive_poseguard_v3_rescue_candidate_shared_cheirality_min=args.adaptive_poseguard_v3_rescue_candidate_shared_cheirality_min,
        adaptive_poseguard_v3_rescue_shared_cheirality_gain=args.adaptive_poseguard_v3_rescue_shared_cheirality_gain,
        adaptive_poseguard_v3_shared_cheirality_drop_tolerance=args.adaptive_poseguard_v3_shared_cheirality_drop_tolerance,
        adaptive_poseguard_v3_native_cheirality_drop_tolerance=args.adaptive_poseguard_v3_native_cheirality_drop_tolerance,
        adaptive_poseguard_v3_early_abort_native_cheirality_max=args.adaptive_poseguard_v3_early_abort_native_cheirality_max,
        adaptive_poseguard_v3_early_abort_parallax_max_deg=args.adaptive_poseguard_v3_early_abort_parallax_max_deg,
        adaptive_poseguard_v4_lk_quality_margin=args.adaptive_poseguard_v4_lk_quality_margin,
        adaptive_poseguard_v4_lk_max_rotation_disagreement_deg=args.adaptive_poseguard_v4_lk_max_rotation_disagreement_deg,
        adaptive_poseguard_v4_lk_max_translation_disagreement_deg=args.adaptive_poseguard_v4_lk_max_translation_disagreement_deg,
        adaptive_poseguard_v4_lk_parallax_ratio_min=args.adaptive_poseguard_v4_lk_parallax_ratio_min,
        adaptive_poseguard_v4_lk_parallax_ratio_max=args.adaptive_poseguard_v4_lk_parallax_ratio_max,
        adaptive_poseguard_v4_lk_shared_cheirality_drop_tolerance=args.adaptive_poseguard_v4_lk_shared_cheirality_drop_tolerance,
        adaptive_poseguard_v4_lk_native_cheirality_drop_tolerance=args.adaptive_poseguard_v4_lk_native_cheirality_drop_tolerance,
    )
    methods = parse_methods(args.methods, args.include_ablation)
    output_dir = Path(args.pair_output).expanduser().resolve()
    filtered_dir = output_dir / "filtered_matches"
    inlier_dir = output_dir / "geometric_inliers"
    filtered_dir.mkdir(parents=True, exist_ok=True)
    inlier_dir.mkdir(parents=True, exist_ok=True)
    xfeat_repo = Path(args.xfeat_repo).expanduser() if args.xfeat_repo else None

    rows: List[Dict[str, Any]] = []
    tiles: List[np.ndarray] = []
    print(f"[two-image comparison] {image1_path.name} -> {image2_path.name}")
    print(f"[methods] {methods}")
    for method in methods:
        try:
            result = run_method(
                method,
                gray1,
                gray2,
                camera,
                cfg,
                args.measure_memory,
                xfeat_repo,
                args.device,
            )
        except Exception as exc:
            result = MethodResult(method=method, success=False, status="exception")
            result.diagnostics["error"] = str(exc)
            if args.debug:
                traceback.print_exc()
        rows.append(pair_result_to_row(result))

        filtered = draw_result(image1, image2, result, max_draw=args.max_draw_matches, inliers_only=False)
        inliers = draw_result(image1, image2, result, max_draw=args.max_draw_matches, inliers_only=True)
        subtitle = (
            f"{result.status} | {result.total_ms:.1f} ms | "
            f"matches {result.num_filtered_matches} | inliers {result.num_geometric_inliers}"
        )
        filtered = _annotate_tile(filtered, result, "filtered: " + subtitle)
        inliers = _annotate_tile(inliers, result, "inliers: " + subtitle)
        cv.imwrite(str(filtered_dir / f"{method}.jpg"), filtered)
        cv.imwrite(str(inlier_dir / f"{method}.jpg"), inliers)
        tiles.append(inliers)
        print(
            f"  {method:24s} status={result.status:30s} "
            f"time={result.total_ms:8.2f} ms matches={result.num_filtered_matches:4d} "
            f"inliers={result.num_geometric_inliers:4d}"
        )

    write_csv_rows(output_dir / "two_image_comparison.csv", rows)
    write_excel_workbook(output_dir / "two_image_comparison.xlsx", {"Comparison": rows})
    sheet = make_contact_sheet(tiles, columns=args.contact_columns)
    cv.imwrite(str(output_dir / "all_methods_geometric_inliers.jpg"), sheet)
    (output_dir / "pair_config.json").write_text(
        json.dumps(
            {
                "image1": str(image1_path),
                "image2": str(image2_path),
                "arguments": vars(args),
                "camera": asdict(camera),
                "pipeline_config": asdict(cfg),
                "opencv_version": cv.__version__,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"[done] table: {output_dir / 'two_image_comparison.csv'}")
    print(f"[done] contact sheet: {output_dir / 'all_methods_geometric_inliers.jpg'}")
    return 0


# ---------------------------------------------------------------------------
# Benchmark orchestration
# ---------------------------------------------------------------------------


CLASSICAL_METHODS = [
    "sift_nn",
    "orb_nn",
    "sift_lowe",
    "orb_lowe",
    "sift_mutual_lowe",
    "orb_mutual_lowe",
    "orb_crosscheck",
    "akaze_lowe",
]

ORIGINAL_PROJECT_METHODS = CLASSICAL_METHODS + [
    "adalam_sift",
    "adalam_orb",
    "orbslam_adaptive_fh",
]

CORRECTED_COMPARISON_METHODS = [
    "adalam_orb_bits",
]

PROPOSED_METHODS = [
    "edgefusion_fast",
    "edgefusion_full",
    "edgefusion_simple",
    "edgefusion_adaptive",
    "edgefusion_adaptive_guarded",
    "edgefusion_adaptive_poseguard",
    "edgefusion_adaptive_poseguard_v3",
    "edgefusion_adaptive_poseguard_v4",
]

DEFAULT_METHODS = ORIGINAL_PROJECT_METHODS + CORRECTED_COMPARISON_METHODS + PROPOSED_METHODS

ABLATION_METHODS = [
    "edgefusion_lite",
    "edgefusion_no_edges",
    "edgefusion_no_grid",
    "edgefusion_no_lk",
]

OPTIONAL_METHODS = ["xfeat", "lightglue_sift"]


def parse_methods(value: str, include_ablation: bool) -> List[str]:
    selector = value.strip().lower()
    if selector in {"core", "default", "original+proposed"}:
        methods = list(DEFAULT_METHODS)
    elif selector == "original":
        methods = list(ORIGINAL_PROJECT_METHODS)
    elif selector == "proposed":
        methods = list(PROPOSED_METHODS)
    elif selector == "all":
        methods = DEFAULT_METHODS + OPTIONAL_METHODS
    else:
        methods = [x.strip().lower() for x in value.split(",") if x.strip()]
    if include_ablation:
        methods += ABLATION_METHODS
    valid = set(DEFAULT_METHODS + ABLATION_METHODS + OPTIONAL_METHODS)
    unknown = [m for m in methods if m not in valid]
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}. Valid: {sorted(valid)}")
    return list(dict.fromkeys(methods))



def _benchmark_method_order(
    methods: Sequence[str],
    pair_index: int,
    mode: str,
    seed: int,
) -> List[str]:
    """Return deterministic per-pair method execution order.

    Rotating the order reduces systematic warm-cache bias while preserving
    reproducibility. ``shuffle`` is also deterministic for a fixed seed.
    """
    ordered = list(methods)
    if len(ordered) <= 1 or mode == "fixed":
        return ordered
    if mode == "rotate":
        shift = int(pair_index) % len(ordered)
        return ordered[shift:] + ordered[:shift]
    if mode == "shuffle":
        rng = np.random.default_rng(int(seed) + int(pair_index) * 1009)
        idx = rng.permutation(len(ordered))
        return [ordered[int(i)] for i in idx]
    raise ValueError(f"Unknown method-order mode: {mode}")


def _warmup_selected_methods(
    pair: PairRecord,
    methods: Sequence[str],
    camera_base: CameraModel,
    cfg: PipelineConfig,
    xfeat_repo: Optional[Path],
    device: str,
) -> None:
    """Run one unmeasured warm-up pass for every selected method."""
    image1 = cv.imread(str(pair.first.path), cv.IMREAD_COLOR)
    image2 = cv.imread(str(pair.second.path), cv.IMREAD_COLOR)
    if image1 is None or image2 is None:
        print("[warm-up] skipped: could not read first benchmark pair", file=sys.stderr)
        return
    image1, scale1 = resize_to_max_side(image1, cfg.max_side)
    image2, scale2 = resize_to_max_side(image2, cfg.max_side)
    if abs(scale1 - scale2) > 1e-6:
        return
    camera = scale_camera(camera_base, scale1)
    gray1, gray2 = ensure_gray(image1), ensure_gray(image2)
    print(f"[warm-up] running {len(methods)} selected methods once (unmeasured)")
    for method in methods:
        try:
            run_method(
                method, gray1, gray2, camera, cfg,
                False, xfeat_repo, device,
            )
        except Exception as exc:
            print(f"[warm-up] {method}: {exc}", file=sys.stderr)

def benchmark(args: argparse.Namespace) -> int:
    np.random.seed(args.seed)
    cv.setRNGSeed(args.seed)
    cv.setNumThreads(args.cv_threads)

    dataset_dir = Path(args.dataset).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "matches").mkdir(exist_ok=True)

    cfg = PipelineConfig(
        nfeatures=args.nfeatures,
        max_side=args.max_side,
        ratio=args.ratio,
        magsac_threshold_px=args.magsac_threshold,
        use_distortion_correction=not args.no_distortion_correction,
        simple_feature_fraction=args.simple_feature_fraction,
        simple_edge_weight=args.simple_edge_weight,
        adaptive_accept_inliers=args.adaptive_accept_inliers,
        adaptive_accept_inlier_ratio=args.adaptive_accept_inlier_ratio,
        adaptive_accept_coverage=args.adaptive_accept_coverage,
        adaptive_fallback_feature_fraction=args.adaptive_fallback_feature_fraction,
        adaptive_lk_max_matches=args.adaptive_lk_max_matches,
        adaptive_guarded_soft_score=args.adaptive_guarded_soft_score,
        adaptive_guarded_soft_floor_fraction=args.adaptive_guarded_soft_floor_fraction,
        adaptive_guarded_quality_margin=args.adaptive_guarded_quality_margin,
        adaptive_guarded_prelk_margin=args.adaptive_guarded_prelk_margin,
        adaptive_poseguard_quality_margin=args.adaptive_poseguard_quality_margin,
        adaptive_poseguard_prelk_margin=args.adaptive_poseguard_prelk_margin,
        adaptive_poseguard_parallax_target_deg=args.adaptive_poseguard_parallax_target_deg,
        adaptive_poseguard_rescue_fast_cheirality_max=args.adaptive_poseguard_rescue_fast_cheirality_max,
        adaptive_poseguard_rescue_candidate_cheirality_min=args.adaptive_poseguard_rescue_candidate_cheirality_min,
        adaptive_poseguard_rescue_cheirality_gain=args.adaptive_poseguard_rescue_cheirality_gain,
        adaptive_poseguard_cheirality_drop_tolerance=args.adaptive_poseguard_cheirality_drop_tolerance,
        adaptive_poseguard_v3_quality_margin=args.adaptive_poseguard_v3_quality_margin,
        adaptive_poseguard_v3_prelk_margin=args.adaptive_poseguard_v3_prelk_margin,
        adaptive_poseguard_v3_min_candidate_native_cheirality=args.adaptive_poseguard_v3_min_candidate_native_cheirality,
        adaptive_poseguard_v3_min_candidate_parallax_deg=args.adaptive_poseguard_v3_min_candidate_parallax_deg,
        adaptive_poseguard_v3_rescue_candidate_native_cheirality_min=args.adaptive_poseguard_v3_rescue_candidate_native_cheirality_min,
        adaptive_poseguard_v3_rescue_native_cheirality_gain=args.adaptive_poseguard_v3_rescue_native_cheirality_gain,
        adaptive_poseguard_v3_rescue_candidate_shared_cheirality_min=args.adaptive_poseguard_v3_rescue_candidate_shared_cheirality_min,
        adaptive_poseguard_v3_rescue_shared_cheirality_gain=args.adaptive_poseguard_v3_rescue_shared_cheirality_gain,
        adaptive_poseguard_v3_shared_cheirality_drop_tolerance=args.adaptive_poseguard_v3_shared_cheirality_drop_tolerance,
        adaptive_poseguard_v3_native_cheirality_drop_tolerance=args.adaptive_poseguard_v3_native_cheirality_drop_tolerance,
        adaptive_poseguard_v3_early_abort_native_cheirality_max=args.adaptive_poseguard_v3_early_abort_native_cheirality_max,
        adaptive_poseguard_v3_early_abort_parallax_max_deg=args.adaptive_poseguard_v3_early_abort_parallax_max_deg,
        adaptive_poseguard_v4_lk_quality_margin=args.adaptive_poseguard_v4_lk_quality_margin,
        adaptive_poseguard_v4_lk_max_rotation_disagreement_deg=args.adaptive_poseguard_v4_lk_max_rotation_disagreement_deg,
        adaptive_poseguard_v4_lk_max_translation_disagreement_deg=args.adaptive_poseguard_v4_lk_max_translation_disagreement_deg,
        adaptive_poseguard_v4_lk_parallax_ratio_min=args.adaptive_poseguard_v4_lk_parallax_ratio_min,
        adaptive_poseguard_v4_lk_parallax_ratio_max=args.adaptive_poseguard_v4_lk_parallax_ratio_max,
        adaptive_poseguard_v4_lk_shared_cheirality_drop_tolerance=args.adaptive_poseguard_v4_lk_shared_cheirality_drop_tolerance,
        adaptive_poseguard_v4_lk_native_cheirality_drop_tolerance=args.adaptive_poseguard_v4_lk_native_cheirality_drop_tolerance,
    )
    methods = parse_methods(args.methods, args.include_ablation)
    records = load_tum_records(dataset_dir, args.max_pose_dt)
    pairs = build_pairs(records, args.frame_step, args.pair_stride, args.max_pairs)
    if not pairs:
        raise RuntimeError("No image pairs were produced")

    pair_manifest = [
        {
            "pair_index": p.pair_index,
            "timestamp1": p.first.timestamp,
            "timestamp2": p.second.timestamp,
            "delta_t_s": p.second.timestamp - p.first.timestamp,
            "image1": str(p.first.path),
            "image2": str(p.second.path),
        }
        for p in pairs
    ]
    write_csv_rows(output_dir / "pair_manifest.csv", pair_manifest)
    raw_correspondence_root = output_dir / "raw_correspondences"
    if args.save_raw_correspondences:
        raw_correspondence_root.mkdir(parents=True, exist_ok=True)

    print(f"[benchmark] dataset={dataset_dir}")
    print(f"[benchmark] associated_frames={len(records)} pairs={len(pairs)}")
    print(f"[benchmark] methods={methods}")
    print(f"[benchmark] output={output_dir}")

    all_rows: List[Dict[str, Any]] = []
    camera_family, camera_base = select_tum_rgb_camera(dataset_dir, args.tum_camera)
    print(
        "[benchmark] camera="
        f"{camera_family} RGB "
        f"fx={camera_base.fx:.1f} fy={camera_base.fy:.1f} "
        f"cx={camera_base.cx:.1f} cy={camera_base.cy:.1f} "
        f"dist={camera_base.distortion}"
    )
    xfeat_repo = Path(args.xfeat_repo).expanduser() if args.xfeat_repo else None

    if not args.no_method_warmup:
        _warmup_selected_methods(
            pairs[0], methods, camera_base, cfg, xfeat_repo, args.device
        )

    for pair_idx, pair in enumerate(pairs):
        image1 = cv.imread(str(pair.first.path), cv.IMREAD_COLOR)
        image2 = cv.imread(str(pair.second.path), cv.IMREAD_COLOR)
        if image1 is None or image2 is None:
            print(f"[pair {pair_idx}] failed to read one or both images", file=sys.stderr)
            continue

        image1, scale1 = resize_to_max_side(image1, cfg.max_side)
        image2, scale2 = resize_to_max_side(image2, cfg.max_side)
        if abs(scale1 - scale2) > 1e-6:
            raise RuntimeError("Image scales differ; this benchmark expects equal TUM image sizes")
        camera = scale_camera(camera_base, scale1)
        gray1, gray2 = ensure_gray(image1), ensure_gray(image2)
        R_gt, t_gt = relative_ground_truth(pair.first.pose, pair.second.pose)

        print(
            f"\n[pair {pair_idx + 1}/{len(pairs)}] "
            f"dt={pair.second.timestamp - pair.first.timestamp:.3f}s "
            f"{pair.first.path.name} -> {pair.second.path.name}"
        )

        pair_methods = _benchmark_method_order(
            methods, pair_idx, args.method_order, args.seed
        )
        for execution_position, method in enumerate(pair_methods):
            try:
                # Warm-up is useful for OpenCV dispatch and optional learned models,
                # but do not warm up every pair because it doubles runtime.
                result = run_method(
                    method,
                    gray1,
                    gray2,
                    camera,
                    cfg,
                    args.measure_memory,
                    xfeat_repo,
                    args.device,
                )
            except Exception as exc:
                result = MethodResult(method=method, success=False, status="exception")
                result.diagnostics["error"] = str(exc)
                if args.debug:
                    traceback.print_exc()

            result.diagnostics["benchmark_execution_position"] = int(execution_position)
            result.diagnostics["benchmark_method_order_mode"] = str(args.method_order)
            row = result_to_row(pair, result, R_gt, t_gt)
            all_rows.append(row)
            save_this_pair = (
                args.save_raw_correspondences
                and (args.raw_correspondence_pairs < 0 or pair_idx < args.raw_correspondence_pairs)
            )
            if save_this_pair:
                save_raw_correspondence_result(
                    raw_correspondence_root, pair, result, R_gt, t_gt
                )
            rot = row["rotation_error_deg"]
            trans = row["translation_error_deg"]
            print(
                f"  {method:24s} status={result.status:28s} "
                f"time={result.total_ms:8.2f} ms "
                f"matches={result.num_filtered_matches:4d} "
                f"inliers={result.num_geometric_inliers:4d} "
                f"Rerr={rot:7.2f} Terr={trans:7.2f}"
            )

            if pair_idx < args.visualize_pairs and result.num_filtered_matches > 0:
                vis = draw_result(image1, image2, result)
                cv.putText(
                    vis,
                    f"{method} | inliers {result.num_geometric_inliers}/{result.num_filtered_matches}",
                    (12, 28),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                    cv.LINE_AA,
                )
                cv.imwrite(str(output_dir / "matches" / f"pair_{pair_idx:03d}_{method}.jpg"), vis)

        # Persist after every pair so an interrupted long benchmark is recoverable.
        write_csv_rows(output_dir / "per_pair_results.csv", all_rows)

    summary = aggregate_rows(all_rows)
    write_csv_rows(output_dir / "summary.csv", summary)
    write_excel_workbook(
        output_dir / "all_algorithm_results.xlsx",
        {"Summary": summary, "PerPair": all_rows},
    )
    write_markdown_summary(output_dir / "summary.md", summary, args)
    make_plots(summary, output_dir)
    (output_dir / "config.json").write_text(
        json.dumps(
            {
                "arguments": vars(args),
                "pipeline_config": asdict(cfg),
                "camera_calibration": {
                    "source": "TUM official RGB calibration",
                    "family": camera_family,
                    "fx": camera_base.fx,
                    "fy": camera_base.fy,
                    "cx": camera_base.cx,
                    "cy": camera_base.cy,
                    "distortion": list(camera_base.distortion),
                },
                "opencv_version": cv.__version__,
                "python_version": platform.python_version(),
                "platform": platform.platform(),
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    write_runtime_environment_snapshot(output_dir)

    if not args.skip_statistics:
        try:
            from statistical_analysis import build_arg_parser as build_stats_parser
            from statistical_analysis import run_analysis

            stats_argv = [
                "--input", str(output_dir / "per_pair_results.csv"),
                "--output", str(output_dir / "statistics"),
                "--methods", args.stats_methods,
                "--metrics", args.stats_metrics,
                "--alpha", str(args.stats_alpha),
                "--bootstrap-resamples", str(args.stats_bootstrap_resamples),
                "--min-complete-pairs", str(args.stats_min_complete_pairs),
                "--seed", str(args.seed),
            ]
            run_analysis(build_stats_parser().parse_args(stats_argv))
        except Exception as exc:
            print(f"[warning] statistical analysis failed: {exc}", file=sys.stderr)
            if args.debug:
                traceback.print_exc()

    print("\n[done]")
    print(f"  per-pair results: {output_dir / 'per_pair_results.csv'}")
    print(f"  summary:          {output_dir / 'summary.csv'}")
    print(f"  report:           {output_dir / 'summary.md'}")
    print(f"  pair manifest:    {output_dir / 'pair_manifest.csv'}")
    if not args.skip_statistics:
        print(f"  statistics:       {output_dir / 'statistics'}")
    if args.save_raw_correspondences:
        print(f"  raw matches:      {raw_correspondence_root}")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare all original matching algorithms, the proposed EdgeFusion variants "
            "(Full, Fast, Simple, Adaptive, Adaptive-Guarded, PoseGuard-v2, PoseGuard-v3), and optional modern matchers on a TUM "
            "dataset and/or a supplied pair of images."
        )
    )
    parser.add_argument(
        "--dataset",
        default="",
        help=(
            "Optional path to an extracted TUM RGB-D dataset directory, e.g. "
            "rgbd_dataset_freiburg1_xyz, rgbd_dataset_freiburg2_desk, or "
            "rgbd_dataset_freiburg3_structure_notexture_near."
        ),
    )
    parser.add_argument(
        "--tum-camera",
        choices=["auto", "freiburg1", "freiburg2", "freiburg3"],
        default="auto",
        help=(
            "RGB camera calibration used for TUM dataset benchmarking. "
            "'auto' (default) infers Freiburg 1/2/3 from the dataset path. "
            "Use an explicit family only if the dataset folder was renamed."
        ),
    )
    parser.add_argument("--output", default="results_all_algorithms", help="Dataset benchmark output directory")
    parser.add_argument("--image1", default="", help="First image for direct two-image comparison")
    parser.add_argument("--image2", default="", help="Second image for direct two-image comparison")
    parser.add_argument("--pair-output", default="results_two_images", help="Two-image comparison output directory")
    parser.add_argument(
        "--methods",
        default="core",
        help=(
            "Selector: core/default, original, proposed, all; or a comma-separated list. "
            "'proposed' includes Full, Fast, Simple, Adaptive, Adaptive-Guarded, PoseGuard-v2, and PoseGuard-v3 EdgeFusion; "
            "'all' also attempts XFeat and LightGlue."
        ),
    )
    parser.add_argument("--include-ablation", action="store_true", help="Add EdgeFusion component ablations")
    parser.add_argument("--frame-step", type=int, default=10, help="Frame separation inside each TUM pair")
    parser.add_argument("--pair-stride", type=int, default=10, help="Stride between TUM pair starting frames")
    parser.add_argument("--max-pairs", type=int, default=50, help="Maximum TUM pairs; <=0 means all")
    parser.add_argument("--max-pose-dt", type=float, default=0.02, help="Maximum RGB/ground-truth association error in seconds")
    parser.add_argument("--nfeatures", type=int, default=1600, help="Feature budget per image")
    parser.add_argument("--max-side", type=int, default=960, help="Downscale so max image side does not exceed this; <=0 disables")
    parser.add_argument("--ratio", type=float, default=0.80, help="Base ratio used by EdgeFusion")
    parser.add_argument("--magsac-threshold", type=float, default=1.25, help="MAGSAC reprojection threshold in pixels")
    parser.add_argument(
        "--simple-feature-fraction",
        type=float,
        default=0.65,
        help=(
            "Fraction of --nfeatures retained by EdgeFusion-Simple "
            "(default: 0.65; 1600 -> about 1040 features)."
        ),
    )
    parser.add_argument(
        "--simple-edge-weight",
        type=float,
        default=0.35,
        help=(
            "Edge-proximity weight in the lightweight EdgeFusion-Simple "
            "candidate score; ORB response receives 1-weight."
        ),
    )
    parser.add_argument(
        "--adaptive-accept-inliers",
        type=int,
        default=80,
        help=(
            "Minimum fast-path geometric inliers required by "
            "EdgeFusion-Adaptive before accepting the ORB-Lowe pose."
        ),
    )
    parser.add_argument(
        "--adaptive-accept-inlier-ratio",
        type=float,
        default=0.65,
        help=(
            "Minimum fast-path inlier ratio required by EdgeFusion-Adaptive."
        ),
    )
    parser.add_argument(
        "--adaptive-accept-coverage",
        type=float,
        default=0.25,
        help=(
            "Minimum fraction of occupied cells in a 4x6 coarse grid required "
            "for EdgeFusion-Adaptive fast-path acceptance."
        ),
    )
    parser.add_argument(
        "--adaptive-fallback-feature-fraction",
        type=float,
        default=0.75,
        help=(
            "Fraction of the already-extracted ORB features retained for "
            "conditional structural refinement."
        ),
    )
    parser.add_argument(
        "--adaptive-lk-max-matches",
        type=int,
        default=240,
        help=(
            "Maximum spatially balanced correspondences refined by LK in the "
            "EdgeFusion-Adaptive fallback; <=0 disables this cap."
        ),
    )
    parser.add_argument(
        "--adaptive-guarded-soft-score",
        type=float,
        default=0.97,
        help=(
            "Minimum aggregate fast-path confidence for the guarded variant's "
            "borderline early-accept rule."
        ),
    )
    parser.add_argument(
        "--adaptive-guarded-soft-floor-fraction",
        type=float,
        default=0.90,
        help=(
            "Each fast confidence component must reach at least this fraction "
            "of its original hard threshold before guarded soft acceptance."
        ),
    )
    parser.add_argument(
        "--adaptive-guarded-quality-margin",
        type=float,
        default=0.02,
        help=(
            "Minimum shared geometric-quality improvement required for a "
            "pre-LK/refined pose to replace the current candidate."
        ),
    )
    parser.add_argument(
        "--adaptive-guarded-prelk-margin",
        type=float,
        default=0.04,
        help=(
            "Minimum shared-quality improvement for the structural pre-LK "
            "pose to be accepted immediately, skipping optical flow."
        ),
    )
    parser.add_argument(
        "--adaptive-poseguard-quality-margin",
        type=float,
        default=0.02,
        help="General PoseGuard-v2 quality improvement required for replacement.",
    )
    parser.add_argument(
        "--adaptive-poseguard-prelk-margin",
        type=float,
        default=0.035,
        help="PoseGuard-v2 quality margin for pre-LK early acceptance.",
    )
    parser.add_argument(
        "--adaptive-poseguard-parallax-target-deg",
        type=float,
        default=1.0,
        help="Median rotation-compensated parallax (deg) treated as fully informative.",
    )
    parser.add_argument(
        "--adaptive-poseguard-rescue-fast-cheirality-max",
        type=float,
        default=0.35,
        help="Maximum fast shared cheirality ratio eligible for explicit rescue.",
    )
    parser.add_argument(
        "--adaptive-poseguard-rescue-candidate-cheirality-min",
        type=float,
        default=0.60,
        help="Minimum candidate shared cheirality ratio for explicit rescue.",
    )
    parser.add_argument(
        "--adaptive-poseguard-rescue-cheirality-gain",
        type=float,
        default=0.25,
        help="Minimum cheirality-ratio gain required by the rescue rule.",
    )
    parser.add_argument(
        "--adaptive-poseguard-cheirality-drop-tolerance",
        type=float,
        default=0.05,
        help="Maximum cheirality-ratio loss tolerated by ordinary quality replacement.",
    )
    parser.add_argument(
        "--adaptive-poseguard-v3-quality-margin",
        type=float,
        default=0.02,
        help="PoseGuard-v3 general quality improvement required after observability gating.",
    )
    parser.add_argument(
        "--adaptive-poseguard-v3-prelk-margin",
        type=float,
        default=0.035,
        help="PoseGuard-v3 quality margin for pre-LK early acceptance.",
    )
    parser.add_argument(
        "--adaptive-poseguard-v3-min-candidate-native-cheirality",
        type=float,
        default=0.50,
        help="Minimum candidate native positive-depth ratio before it may replace another pose.",
    )
    parser.add_argument(
        "--adaptive-poseguard-v3-min-candidate-parallax-deg",
        type=float,
        default=0.50,
        help="Minimum candidate median rotation-compensated parallax (deg) before replacement.",
    )
    parser.add_argument(
        "--adaptive-poseguard-v3-rescue-candidate-native-cheirality-min",
        type=float,
        default=0.80,
        help="Minimum candidate native positive-depth ratio for the v3 gain-based rescue rule.",
    )
    parser.add_argument(
        "--adaptive-poseguard-v3-rescue-native-cheirality-gain",
        type=float,
        default=0.25,
        help="Minimum native positive-depth-ratio gain for v3 rescue.",
    )
    parser.add_argument(
        "--adaptive-poseguard-v3-rescue-candidate-shared-cheirality-min",
        type=float,
        default=0.60,
        help="Minimum shared positive-depth ratio for the v3 gain-based rescue rule.",
    )
    parser.add_argument(
        "--adaptive-poseguard-v3-rescue-shared-cheirality-gain",
        type=float,
        default=0.20,
        help="Minimum shared positive-depth-ratio gain for v3 rescue.",
    )
    parser.add_argument(
        "--adaptive-poseguard-v3-shared-cheirality-drop-tolerance",
        type=float,
        default=0.05,
        help="Maximum shared positive-depth-ratio loss tolerated by ordinary v3 replacement.",
    )
    parser.add_argument(
        "--adaptive-poseguard-v3-native-cheirality-drop-tolerance",
        type=float,
        default=0.10,
        help="Maximum native positive-depth-ratio loss tolerated by ordinary v3 replacement.",
    )
    parser.add_argument(
        "--adaptive-poseguard-v3-early-abort-native-cheirality-max",
        type=float,
        default=0.15,
        help="Pre-LK native positive-depth ratio below which LK may be skipped when parallax is also tiny and fast pose exists.",
    )
    parser.add_argument(
        "--adaptive-poseguard-v3-early-abort-parallax-max-deg",
        type=float,
        default=0.25,
        help="Pre-LK median parallax ceiling used together with the early-abort cheirality threshold.",
    )
    parser.add_argument(
        "--adaptive-poseguard-v4-lk-quality-margin",
        type=float,
        default=0.015,
        help="Minimum PoseGuard quality gain required for LK to replace an already-selected pre-LK pose.",
    )
    parser.add_argument(
        "--adaptive-poseguard-v4-lk-max-rotation-disagreement-deg",
        type=float,
        default=3.0,
        help="Maximum rotation disagreement (deg) allowed between pre-LK and LK poses.",
    )
    parser.add_argument(
        "--adaptive-poseguard-v4-lk-max-translation-disagreement-deg",
        type=float,
        default=15.0,
        help="Maximum sign-invariant translation-direction disagreement (deg) allowed between pre-LK and LK poses.",
    )
    parser.add_argument(
        "--adaptive-poseguard-v4-lk-parallax-ratio-min",
        type=float,
        default=0.50,
        help="Minimum LK/pre-LK median parallax ratio for a stable LK replacement.",
    )
    parser.add_argument(
        "--adaptive-poseguard-v4-lk-parallax-ratio-max",
        type=float,
        default=2.00,
        help="Maximum LK/pre-LK median parallax ratio for a stable LK replacement.",
    )
    parser.add_argument(
        "--adaptive-poseguard-v4-lk-shared-cheirality-drop-tolerance",
        type=float,
        default=0.03,
        help="Maximum shared positive-depth-ratio loss tolerated when LK replaces pre-LK.",
    )
    parser.add_argument(
        "--adaptive-poseguard-v4-lk-native-cheirality-drop-tolerance",
        type=float,
        default=0.05,
        help="Maximum native positive-depth-ratio loss tolerated when LK replaces pre-LK.",
    )
    parser.add_argument(
        "--method-order",
        choices=["fixed", "rotate", "shuffle"],
        default="rotate",
        help=(
            "Measured per-pair method execution order. 'rotate' (default) cycles "
            "methods through execution positions to reduce cache/order bias."
        ),
    )
    parser.add_argument(
        "--no-method-warmup",
        action="store_true",
        help="Disable the one-pass unmeasured warm-up performed before dataset timing.",
    )
    parser.add_argument("--no-distortion-correction", action="store_true", help="Disable keypoint distortion correction")
    parser.add_argument("--measure-memory", action="store_true", help="Sample peak RSS; adds small timing overhead")
    parser.add_argument("--visualize-pairs", type=int, default=3, help="Save dataset visualizations for the first N pairs")
    parser.add_argument("--max-draw-matches", type=int, default=180, help="Maximum correspondences drawn per method")
    parser.add_argument("--contact-columns", type=int, default=2, help="Columns in the two-image contact sheet")
    parser.add_argument(
        "--pair-camera",
        choices=["auto", "tum_fr1", "tum_fr2", "tum_fr3", "custom"],
        default="auto",
        help=(
            "Camera model for direct two-image pose estimation. TUM choices "
            "refer to the official RGB-camera calibration."
        ),
    )
    parser.add_argument("--fx", type=float, default=None, help="Custom camera fx before resizing")
    parser.add_argument("--fy", type=float, default=None, help="Custom camera fy before resizing")
    parser.add_argument("--cx", type=float, default=None, help="Custom camera cx before resizing")
    parser.add_argument("--cy", type=float, default=None, help="Custom camera cy before resizing")
    parser.add_argument(
        "--distortion",
        default="0,0,0,0,0",
        help="Custom k1,k2,p1,p2,k3 before resizing",
    )
    parser.add_argument("--cv-threads", type=int, default=1, help="OpenCV CPU thread count for reproducible timing")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--xfeat-repo", default="", help="Path to cloned official accelerated_features repository")
    parser.add_argument("--device", default="cpu", help="PyTorch device for bundled AdaLAM and optional XFeat/LightGlue; use cpu for fair CPU timing or auto/cuda for acceleration")
    parser.add_argument(
        "--save-raw-correspondences",
        action="store_true",
        help="Save per-method/per-pair NPZ files containing points, inlier masks, poses, timings, and diagnostics",
    )
    parser.add_argument(
        "--raw-correspondence-pairs",
        type=int,
        default=-1,
        help="Number of leading pairs for raw NPZ export; -1 saves all pairs",
    )
    parser.add_argument("--skip-statistics", action="store_true", help="Do not run the paired statistical analysis after the benchmark")
    parser.add_argument("--stats-methods", default="all", help="Methods used by the statistical analysis: all or comma-separated names")
    parser.add_argument(
        "--stats-metrics",
        default="total_ms,rotation_error_deg,translation_error_sign_invariant_deg,inlier_ratio,geometric_inliers,peak_rss_delta_mb",
        help="Comma-separated metrics for Friedman/Wilcoxon/CD analysis",
    )
    parser.add_argument("--stats-alpha", type=float, default=0.05)
    parser.add_argument("--stats-bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--stats-min-complete-pairs", type=int, default=10)
    parser.add_argument("--debug", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if bool(args.image1) != bool(args.image2):
        parser.error("--image1 and --image2 must be supplied together")
    if not args.dataset and not (args.image1 and args.image2):
        parser.error("Provide --dataset and/or both --image1 and --image2")
    try:
        exit_code = 0
        if args.dataset:
            exit_code = max(exit_code, benchmark(args))
        if args.image1 and args.image2:
            exit_code = max(exit_code, compare_two_images(args))
        return exit_code
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        if getattr(args, "debug", False):
            traceback.print_exc()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
