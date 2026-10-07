from pathlib import Path

import numpy as np
import pandas as pd

from statistical_analysis import build_arg_parser, run_analysis


def test_statistics_and_cd_outputs(tmp_path: Path) -> None:
    rows = []
    methods = ["a", "b", "c"]
    for pair in range(12):
        for idx, method in enumerate(methods):
            rows.append(
                {
                    "pair_index": pair,
                    "method": method,
                    "success": 1,
                    "status": "ok",
                    "total_ms": 10 + idx * 2 + pair * 0.1,
                    "rotation_error_deg": 0.5 + idx * 0.2 + pair * 0.01,
                    "translation_error_sign_invariant_deg": 4 + idx + pair * 0.02,
                    "inlier_ratio": 0.9 - idx * 0.05,
                    "geometric_inliers": 200 - idx * 10,
                    "peak_rss_delta_mb": 4 + idx,
                    "extract_ms": 5 + idx,
                    "match_ms": 3 + idx,
                }
            )
    input_csv = tmp_path / "per_pair_results.csv"
    pd.DataFrame(rows).to_csv(input_csv, index=False)
    output = tmp_path / "stats"
    args = build_arg_parser().parse_args(
        [
            "--input", str(input_csv),
            "--output", str(output),
            "--bootstrap-resamples", "50",
            "--min-complete-pairs", "5",
        ]
    )
    assert run_analysis(args) == 0
    assert (output / "tables/friedman_omnibus.csv").exists()
    assert (output / "tables/pairwise_wilcoxon_holm_total_ms.csv").exists()
    assert (output / "figures/cd_total_ms.png").exists()
    assert (output / "figure_data/total_ms_long.csv").exists()
    omnibus = pd.read_csv(output / "tables/friedman_omnibus.csv")
    assert "total_ms" in set(omnibus["metric"])


def test_skipped_method_is_excluded(tmp_path: Path) -> None:
    rows = []
    for pair in range(10):
        rows.extend(
            [
                {"pair_index": pair, "method": "a", "success": 1, "status": "ok", "total_ms": 10 + pair},
                {"pair_index": pair, "method": "b", "success": 1, "status": "ok", "total_ms": 12 + pair},
                {"pair_index": pair, "method": "missing", "success": 0, "status": "skipped_not_installed", "total_ms": 0},
            ]
        )
    input_csv = tmp_path / "input.csv"
    pd.DataFrame(rows).to_csv(input_csv, index=False)
    output = tmp_path / "stats"
    args = build_arg_parser().parse_args(
        [
            "--input", str(input_csv),
            "--output", str(output),
            "--metrics", "total_ms",
            "--bootstrap-resamples", "10",
            "--min-complete-pairs", "5",
        ]
    )
    assert run_analysis(args) == 0
    ranks = pd.read_csv(output / "tables/average_ranks_total_ms.csv")
    assert "missing" not in set(ranks["method"])
