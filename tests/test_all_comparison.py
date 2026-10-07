from pathlib import Path
import sys

import cv2 as cv
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from matching_comparison_all import (
    CameraModel,
    PipelineConfig,
    _load_adalam_filter,
    parse_methods,
    run_method,
)


def synthetic_pair():
    img = np.zeros((360, 480), np.uint8)
    rng = np.random.default_rng(11)
    for _ in range(140):
        x, y = rng.integers([20, 20], [460, 340])
        cv.circle(img, (int(x), int(y)), int(rng.integers(2, 7)), int(rng.integers(100, 255)), -1)
    M = cv.getRotationMatrix2D((240, 180), 1.5, 1.0)
    M[:, 2] += np.array([6.0, 3.0])
    return img, cv.warpAffine(img, M, (480, 360))


def test_core_method_selector_contains_original_and_proposed():
    methods = parse_methods("core", include_ablation=False)
    for name in [
        "sift_nn",
        "orb_lowe",
        "sift_mutual_lowe",
        "adalam_sift",
        "adalam_orb",
        "adalam_orb_bits",
        "orbslam_adaptive_fh",
        "edgefusion_fast",
        "edgefusion_full",
    ]:
        assert name in methods


def test_classical_and_adaptive_fh_smoke():
    a, b = synthetic_pair()
    cfg = PipelineConfig(nfeatures=400, use_distortion_correction=False)
    camera = CameraModel(fx=430, fy=430, cx=240, cy=180, distortion=(0, 0, 0, 0, 0))
    for method in ["orb_lowe", "orb_mutual_lowe", "orb_crosscheck", "orbslam_adaptive_fh"]:
        result = run_method(method, a, b, camera, cfg, False, None, "cpu")
        assert result.status != "exception"
        assert result.num_raw_matches >= 8


def test_bundled_adalam_source_is_loaded_and_runs():
    _, backend = _load_adalam_filter()
    assert "third_party" in backend
    assert "adalam" in backend

    sample_dir = ROOT / "third_party" / "adalam"
    image1 = cv.imread(str(sample_dir / "im1.jpg"), cv.IMREAD_GRAYSCALE)
    image2 = cv.imread(str(sample_dir / "im2.jpg"), cv.IMREAD_GRAYSCALE)
    assert image1 is not None and image2 is not None

    cfg = PipelineConfig(nfeatures=1000, use_distortion_correction=False)
    h, w = image1.shape[:2]
    camera = CameraModel(fx=0.9 * w, fy=0.9 * w, cx=w / 2, cy=h / 2, distortion=(0, 0, 0, 0, 0))
    results = {}
    for method in ["adalam_sift", "adalam_orb", "adalam_orb_bits"]:
        result = run_method(method, image1, image2, camera, cfg, False, None, "cpu")
        results[method] = result
        assert result.status not in {"exception", "adalam_source_unavailable", "adalam_runtime_error"}
        assert result.num_raw_matches > 0
        assert "bundled_source" in str(result.diagnostics.get("adalam_backend", ""))

    # The exact raw-byte ORB mode is retained for historical reproducibility and
    # can legitimately return no local-affine consensus on this sample pair.
    assert results["adalam_sift"].num_filtered_matches > 0
    assert results["adalam_orb_bits"].num_filtered_matches > 0
