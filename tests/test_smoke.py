from pathlib import Path
import sys

import cv2 as cv
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from enhanced_matching_benchmark import CameraModel, PipelineConfig, run_method


def synthetic_pair():
    img = np.zeros((480, 640), np.uint8)
    rng = np.random.default_rng(5)
    for _ in range(120):
        x, y = rng.integers([30, 30], [610, 450])
        r = int(rng.integers(3, 12))
        cv.circle(img, (int(x), int(y)), r, int(rng.integers(80, 255)), 1)
    for i in range(12):
        cv.putText(img, f"E{i}", (30 + (i % 4) * 145, 70 + (i // 4) * 140), cv.FONT_HERSHEY_SIMPLEX, 0.8, 255, 2)
    H = cv.getRotationMatrix2D((320, 240), 2.0, 1.0)
    H[:, 2] += np.array([8.0, 3.0])
    warped = cv.warpAffine(img, H, (640, 480))
    return img, warped


def test_edgefusion_smoke():
    a, b = synthetic_pair()
    cfg = PipelineConfig(nfeatures=800, use_distortion_correction=False)
    result = run_method("edgefusion_full", a, b, CameraModel(), cfg, False, None, "cpu")
    assert result.num_raw_matches >= 8
    assert result.num_filtered_matches >= 8
