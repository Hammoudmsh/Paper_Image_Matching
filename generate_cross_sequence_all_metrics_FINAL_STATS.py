#!/usr/bin/env python3
"""
Sequence-level cross-sequence analysis for the EdgeFusion paper.

The statistical unit for cross-sequence inference is the DATASET/SEQUENCE,
not the individual image pair.

For each feature budget independently, this script produces:

DESCRIPTIVE ANALYSIS
--------------------
For every Method x Metric across sequences:
    * number of sequences
    * mean
    * median
    * SD
    * IQR
    * min / max
    * sequence-bootstrap 95% CI for the mean
    * sequence-bootstrap 95% CI for the median

The main paper cell is formatted as:
    mean [95% sequence-bootstrap CI] / median

INFERENTIAL ANALYSIS
--------------------
For every metric, with sequence as the repeated-measures block:
    1. Friedman omnibus test across methods
    2. Kendall's W effect size
    3. Pairwise Wilcoxon signed-rank tests across sequences
    4. Holm family-wise multiplicity correction
    5. Paired rank-biserial effect size (r_rb)

Positive r_rb always favours method A after accounting for whether a metric
is higher-is-better or lower-is-better.

If the input contains several NFeatures values, each feature budget is analyzed
SEPARATELY. Budgets are never pooled into one cross-sequence method comparison.

Input
-----
Expected sequence-level numeric summaries, e.g.:
    results_edgefusion_all/all_ds/TABLE_ALL_DATASETS.csv

Required columns:
    Dataset
and either:
    method
or:
    method_label

Expected metric columns include:
    success_mean_pct
    total_ms_median
    peak_rss_delta_mb_median
    rotation_error_deg_median
    translation_error_sign_invariant_deg_median
    inlier_ratio_median
    geometric_inliers_median

Example
-------
python generate_cross_sequence_all_metrics_FINAL_STATS.py \
    --input results_edgefusion_all/cross_sequence/N1600/TABLE_ALL_DATASETS.csv \
    --output results_edgefusion_all/PAPER_RESULTS/cross_sequence_by_budget/N1600

For a master file containing NFeatures=400,800,1200,1600:
python generate_cross_sequence_all_metrics_FINAL_STATS.py \
    --input results_edgefusion_all/cross_sequence/TABLE_ALL_DATASETS_ALL_BUDGETS.csv \
    --output results_edgefusion_all/PAPER_RESULTS/final_cross_sequence
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib

# Safe for batch/headless execution.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_INPUT = Path("results_edgefusion_all/all_ds/TABLE_ALL_DATASETS.csv")
DEFAULT_OUTPUT = Path("results_edgefusion_all/PAPER_RESULTS/cross_sequence_outputs")

DEFAULT_NBOOT = 10_000
DEFAULT_SEED = 7
DEFAULT_ALPHA = 0.05
DATASET_PREFIX = "freiburg"


# Sequence-level metrics.
# The values in the aggregate CSV are already per-sequence summaries.
METRICS: Dict[str, str] = {
    "Success (%)": "success_mean_pct",
    "Time (ms)": "total_ms_median",
    "Delta RSS (MB)": "peak_rss_delta_mb_median",
    "eR (deg)": "rotation_error_deg_median",
    "et (deg)": "translation_error_sign_invariant_deg_median",
    "Inlier ratio": "inlier_ratio_median",
    "Inliers": "geometric_inliers_median",
}

# Direction controls rankings and orientation of rank-biserial effects.
# For pairwise results, positive r_rb always favours method A.
METRIC_DIRECTION: Dict[str, str] = {
    "Success (%)": "higher",
    "Time (ms)": "lower",
    "Delta RSS (MB)": "lower",
    "eR (deg)": "lower",
    "et (deg)": "lower",
    "Inlier ratio": "higher",
    "Inliers": "higher",
}

METRIC_LATEX: Dict[str, str] = {
    "Success (%)": r"Success (\%)",
    "Time (ms)": r"Time (ms)",
    "Delta RSS (MB)": r"$\Delta$RSS (MB)",
    "eR (deg)": r"$e_R$ ($^\circ$)",
    "et (deg)": r"$e_t$ ($^\circ$)",
    "Inlier ratio": r"Inlier ratio",
    "Inliers": r"Inliers",
}

# Wide rank-table columns.  These are the same average ranks used by the
# sequence-level Friedman/CD-style comparison: rank 1 is always best.
RANK_TABLE_METRICS: Tuple[Tuple[str, str], ...] = tuple(
    (
        metric_name,
        (
            f"{metric_name} rank "
            f"({'higher' if METRIC_DIRECTION[metric_name] == 'higher' else 'lower'} "
            "is better)"
        ),
    )
    for metric_name in METRICS
)


# Canonical mapping from raw benchmark IDs to publication labels.
RAW_TO_PUB: Dict[str, str] = {
    "sift_nn": "SIFT--NN",
    "orb_nn": "ORB--NN",
    "sift_lowe": "SIFT--Lowe",
    "orb_lowe": "ORB--Lowe",
    "sift_mutual_lowe": "SIFT--Mutual-Lowe",
    "orb_mutual_lowe": "ORB--Mutual-Lowe",
    "orb_crosscheck": "ORB cross-check",
    "akaze_lowe": "AKAZE--Lowe",
    "adalam_sift": "AdaLAM--SIFT",
    "adalam_orb": "AdaLAM--ORB",
    # "orbslam_adaptive_fh": "ORB-SLAM-inspired H/F",
    "adalam_orb_bits": "AdaLAM--ORB-bits",
    "xfeat": "XFeat",
    "lightglue_sift": "SIFT--LightGlue",
    "edgefusion_fast": "EdgeFusion--Fast",
    "edgefusion_full": "EdgeFusion--Full",
    "edgefusion_lite": "EdgeFusion--Lite",
    "edgefusion_no_edges": "EdgeFusion--No-Edges",
    "edgefusion_no_grid": "EdgeFusion--No-Grid",
    "edgefusion_no_lk": "EdgeFusion--No-LK",
    "edgefusion_adaptive_poseguard_v4": "EdgeFusion--PoseGuard",
}

# Also normalize labels already written by older scripts.
LABEL_ALIASES: Dict[str, str] = {
    "SIFT--NN": "SIFT--NN",
    "ORB--NN": "ORB--NN",
    "SIFT--Lowe": "SIFT--Lowe",
    "ORB--Lowe": "ORB--Lowe",
    "SIFT--Mutual-Lowe": "SIFT--Mutual-Lowe",
    "ORB--Mutual-Lowe": "ORB--Mutual-Lowe",
    "ORB cross-check": "ORB cross-check",
    "AKAZE--Lowe": "AKAZE--Lowe",
    "AdaLAM--SIFT": "AdaLAM--SIFT",
    "AdaLAM--ORB": "AdaLAM--ORB",
    # "ORB-SLAM-inspired H/F": "ORB-SLAM-inspired H/F",
    "AdaLAM--ORB-bits": "AdaLAM--ORB-bits",
    "XFeat": "XFeat",
    "SIFT--LightGlue": "SIFT--LightGlue",
    "EdgeFusion-Fast": "EdgeFusion--Fast",
    "EdgeFusion--Fast": "EdgeFusion--Fast",
    "EdgeFusion-Full": "EdgeFusion--Full",
    "EdgeFusion--Full": "EdgeFusion--Full",
    "EdgeFusion-Lite": "EdgeFusion--Lite",
    "EdgeFusion--Lite": "EdgeFusion--Lite",
    "EdgeFusion-No-Edges": "EdgeFusion--No-Edges",
    "EdgeFusion--No-Edges": "EdgeFusion--No-Edges",
    "EdgeFusion-No-Grid": "EdgeFusion--No-Grid",
    "EdgeFusion--No-Grid": "EdgeFusion--No-Grid",
    "EdgeFusion-No-LK": "EdgeFusion--No-LK",
    "EdgeFusion--No-LK": "EdgeFusion--No-LK",
    "edgefusion-adaptive-poseguard-v4": "EdgeFusion--PoseGuard",
    "EdgeFusion-PoseGuard": "EdgeFusion--PoseGuard",
    "EdgeFusion--PoseGuard": "EdgeFusion--PoseGuard",
}

METHOD_ORDER: List[str] = [
    "SIFT--NN",
    "ORB--NN",
    "SIFT--Lowe",
    "ORB--Lowe",
    "SIFT--Mutual-Lowe",
    "ORB--Mutual-Lowe",
    "ORB cross-check",
    "AKAZE--Lowe",
    "AdaLAM--SIFT",
    "AdaLAM--ORB",
    # "ORB-SLAM-inspired H/F",
    "AdaLAM--ORB-bits",
    "XFeat",
    "SIFT--LightGlue",
    "EdgeFusion--Fast",
    "EdgeFusion--Full",
    "EdgeFusion--Lite",
    "EdgeFusion--No-Edges",
    "EdgeFusion--No-Grid",
    "EdgeFusion--No-LK",
    "EdgeFusion--PoseGuard",
]


# Plot identity retained from the previous cross-sequence script.
STYLE_MAP = {
    "SIFT--NN": {"color": "#1f77b4", "marker": "o"},
    "ORB--NN": {"color": "#ff7f0e", "marker": "s"},
    "SIFT--Lowe": {"color": "#2ca02c", "marker": "^"},
    "ORB--Lowe": {"color": "#d62728", "marker": "D"},
    "SIFT--Mutual-Lowe": {"color": "#9467bd", "marker": "v"},
    "ORB--Mutual-Lowe": {"color": "#8c564b", "marker": "P"},
    "ORB cross-check": {"color": "#e377c2", "marker": "X"},
    "AKAZE--Lowe": {"color": "#7f7f7f", "marker": "<"},
    "AdaLAM--SIFT": {"color": "#bcbd22", "marker": ">"},
    "AdaLAM--ORB": {"color": "#17becf", "marker": "h"},
    # "ORB-SLAM-inspired H/F": {"color": "#393b79", "marker": "H"},
    "AdaLAM--ORB-bits": {"color": "#637939", "marker": "8"},
    "XFeat": {"color": "#8c6d31", "marker": "p"},
    "SIFT--LightGlue": {"color": "#843c39", "marker": "H"},
    "EdgeFusion--Fast": {"color": "#3182bd", "marker": "*"},
    "EdgeFusion--Full": {"color": "#e6550d", "marker": "o"},
    "EdgeFusion--Lite": {"color": "#31a354", "marker": "s"},
    "EdgeFusion--No-Edges": {"color": "#756bb1", "marker": "^"},
    "EdgeFusion--No-Grid": {"color": "#636363", "marker": "D"},
    "EdgeFusion--No-LK": {"color": "#dd1c77", "marker": "v"},
    "EdgeFusion--PoseGuard": {"color": "#000000", "marker": "P"},
}


# =============================================================================
# BASIC UTILITIES
# =============================================================================

def finite(values: Iterable[object]) -> np.ndarray:
    return (
        pd.to_numeric(pd.Series(list(values)), errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .to_numpy(dtype=float)
    )


def latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
        "$": r"\$",
        "{": r"\{",
        "}": r"\}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def ordered_present_methods(methods: Iterable[str]) -> List[str]:
    present = list(dict.fromkeys(str(x) for x in methods))
    known = [m for m in METHOD_ORDER if m in present]
    unknown = sorted(m for m in present if m not in known)

    # print(unknown)
    # aaaaaaaaaaa
    return known


def canonical_method_label(row: pd.Series) -> str:
    if "method" in row and pd.notna(row["method"]):
        raw = str(row["method"]).strip()
        if raw in RAW_TO_PUB:
            return RAW_TO_PUB[raw]

    for col in ["method_label_pub", "method_label"]:
        if col in row and pd.notna(row[col]):
            label = str(row[col]).strip()
            return LABEL_ALIASES.get(label, label)

    return "Unknown"


def holm_adjust(pvalues: Sequence[float]) -> np.ndarray:
    """Holm step-down family-wise error-rate adjustment."""
    p = np.asarray(pvalues, dtype=float)
    adjusted = np.full(len(p), np.nan, dtype=float)

    finite_idx = np.flatnonzero(np.isfinite(p))
    if len(finite_idx) == 0:
        return adjusted

    order = finite_idx[np.argsort(p[finite_idx])]
    m = len(order)
    running = 0.0

    for rank, idx in enumerate(order):
        candidate = min(1.0, (m - rank) * p[idx])
        running = max(running, candidate)
        adjusted[idx] = running

    return adjusted


def p_text(value: float) -> str:
    if not np.isfinite(value):
        return "--"
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


# =============================================================================
# BOOTSTRAP DESCRIPTIVE STATISTICS
# =============================================================================

def bootstrap_ci(
    values: np.ndarray,
    *,
    statistic: str,
    confidence: float,
    nboot: int,
    rng: np.random.Generator,
) -> Tuple[float, float]:
    """
    Percentile bootstrap CI by resampling SEQUENCES with replacement.

    `values` must contain one sequence-level value per sequence.
    """
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return np.nan, np.nan

    if len(x) == 1 or nboot <= 0:
        return float(x[0]), float(x[0])

    indices = rng.integers(
        0,
        len(x),
        size=(nboot, len(x)),
    )
    samples = x[indices]

    if statistic == "mean":
        estimates = np.mean(samples, axis=1)
    elif statistic == "median":
        estimates = np.median(samples, axis=1)
    else:
        raise ValueError(f"Unknown statistic: {statistic}")

    alpha = 1.0 - confidence
    lo, hi = np.quantile(
        estimates,
        [alpha / 2.0, 1.0 - alpha / 2.0],
    )

    return float(lo), float(hi)


def sequence_descriptive_statistics(
    seq: pd.DataFrame,
    methods: Sequence[str],
    *,
    confidence: float,
    nboot: int,
    seed: int,
) -> pd.DataFrame:
    """
    One row per Method x Metric.

    Each input value represents one sequence, so sequences receive equal weight
    regardless of how many image pairs they contained.
    """
    rows: List[Dict[str, object]] = []

    for metric_index, (metric_name, col) in enumerate(METRICS.items()):
        if col not in seq.columns:
            continue

        for method_index, method in enumerate(methods):
            g = seq.loc[
                seq["Method"] == method,
                ["Dataset", col],
            ].copy()

            g[col] = pd.to_numeric(
                g[col],
                errors="coerce",
            )
            g = g.dropna(subset=[col])

            # Defensive aggregation: one value per sequence even if an input
            # file accidentally contains duplicate rows.
            by_sequence = (
                g.groupby("Dataset", as_index=False)[col]
                .median()
            )

            x = by_sequence[col].to_numpy(dtype=float)

            if len(x) == 0:
                continue

            # Deterministic but different bootstrap stream per metric/method.
            rng = np.random.default_rng(
                seed + metric_index * 10_000 + method_index * 101
            )

            mean_ci = bootstrap_ci(
                x,
                statistic="mean",
                confidence=confidence,
                nboot=nboot,
                rng=rng,
            )
            median_ci = bootstrap_ci(
                x,
                statistic="median",
                confidence=confidence,
                nboot=nboot,
                rng=rng,
            )

            q1, q3 = np.quantile(x, [0.25, 0.75])

            rows.append(
                {
                    "Method": method,
                    "Metric": metric_name,
                    "metric_column": col,
                    "direction": METRIC_DIRECTION[metric_name],
                    "n_sequences": int(len(x)),
                    "mean": float(np.mean(x)),
                    "mean_ci_low": mean_ci[0],
                    "mean_ci_high": mean_ci[1],
                    "median": float(np.median(x)),
                    "median_ci_low": median_ci[0],
                    "median_ci_high": median_ci[1],
                    "std": (
                        float(np.std(x, ddof=1))
                        if len(x) > 1
                        else np.nan
                    ),
                    "q1": float(q1),
                    "q3": float(q3),
                    "iqr": float(q3 - q1),
                    "minimum": float(np.min(x)),
                    "maximum": float(np.max(x)),
                    "confidence_level": confidence,
                    "bootstrap_resamples": nboot,
                }
            )

    return pd.DataFrame(rows)


def format_main_cell(row: pd.Series, digits: int = 3) -> str:
    """Main manuscript format: mean [95% CI] / median."""
    return (
        f"{row['mean']:.{digits}f} "
        f"[{row['mean_ci_low']:.{digits}f}, "
        f"{row['mean_ci_high']:.{digits}f}] / "
        f"{row['median']:.{digits}f}"
    )


def build_formatted_summary(
    descriptive: pd.DataFrame,
    methods: Sequence[str],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    for method in methods:
        row: Dict[str, object] = {"Method": method}

        for metric_name in METRICS:
            subset = descriptive[
                (descriptive["Method"] == method)
                & (descriptive["Metric"] == metric_name)
            ]

            if subset.empty:
                row[metric_name] = "--"
                continue

            row[metric_name] = format_main_cell(
                subset.iloc[0],
                digits=3,
            )

        rows.append(row)

    return pd.DataFrame(rows)


def build_numeric_wide_summary(
    descriptive: pd.DataFrame,
    methods: Sequence[str],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    for method in methods:
        row: Dict[str, object] = {"Method": method}

        for metric_name in METRICS:
            subset = descriptive[
                (descriptive["Method"] == method)
                & (descriptive["Metric"] == metric_name)
            ]
            if subset.empty:
                continue

            r = subset.iloc[0]
            prefix = metric_name.replace(" ", "_").replace("%", "pct")

            for field in [
                "n_sequences",
                "mean",
                "mean_ci_low",
                "mean_ci_high",
                "median",
                "median_ci_low",
                "median_ci_high",
                "std",
                "q1",
                "q3",
                "iqr",
                "minimum",
                "maximum",
            ]:
                row[f"{prefix}__{field}"] = r[field]

        rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# FRIEDMAN / WILCOXON / EFFECT SIZE
# =============================================================================

def metric_pivot(
    seq: pd.DataFrame,
    methods: Sequence[str],
    metric_column: str,
) -> pd.DataFrame:
    """
    Return Dataset x Method sequence-level matrix.

    Duplicate sequence-method rows are collapsed with a median only as a
    defensive safeguard.
    """
    subset = seq[
        seq["Method"].isin(methods)
    ][["Dataset", "Method", metric_column]].copy()

    subset[metric_column] = pd.to_numeric(
        subset[metric_column],
        errors="coerce",
    )

    return subset.pivot_table(
        index="Dataset",
        columns="Method",
        values=metric_column,
        aggfunc="median",
    ).reindex(columns=list(methods))


def rank_matrix(
    complete: pd.DataFrame,
    direction: str,
) -> pd.DataFrame:
    if complete.empty:
        return complete.copy()

    values = complete.to_numpy(dtype=float)

    # rankdata assigns rank 1 to the smallest value.
    if direction == "higher":
        values = -values

    ranks = np.vstack(
        [
            stats.rankdata(row, method="average")
            for row in values
        ]
    )

    return pd.DataFrame(
        ranks,
        index=complete.index,
        columns=complete.columns,
    )


def paired_rank_biserial(
    a: np.ndarray,
    b: np.ndarray,
    direction: str,
) -> float:
    """
    Paired rank-biserial effect size.

    Positive values favour method A.
    Negative values favour method B.

    For lower-is-better metrics:
        improvement = B - A
    For higher-is-better metrics:
        improvement = A - B
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if direction == "lower":
        improvement = b - a
    elif direction == "higher":
        improvement = a - b
    else:
        raise ValueError(direction)

    improvement = improvement[np.isfinite(improvement)]
    improvement = improvement[improvement != 0]

    if len(improvement) == 0:
        return 0.0

    ranks = stats.rankdata(
        np.abs(improvement),
        method="average",
    )

    positive = float(
        ranks[improvement > 0].sum()
    )
    negative = float(
        ranks[improvement < 0].sum()
    )

    denominator = positive + negative

    if denominator == 0:
        return 0.0

    return float(
        (positive - negative) / denominator
    )


def friedman_sequence_level(
    seq: pd.DataFrame,
    methods: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    omnibus_rows: List[Dict[str, object]] = []
    rank_rows: List[Dict[str, object]] = []

    for metric_name, col in METRICS.items():
        if col not in seq.columns:
            continue

        pivot = metric_pivot(
            seq,
            methods,
            col,
        )

        # Methods with no finite values anywhere cannot participate.
        pivot = pivot.dropna(
            axis=1,
            how="all",
        )

        # Friedman requires the same blocks for all participating methods.
        complete = pivot.dropna(
            axis=0,
            how="any",
        )

        n = len(complete)
        k = len(complete.columns)

        if n < 2 or k < 3:
            omnibus_rows.append(
                {
                    "Metric": metric_name,
                    "metric_column": col,
                    "direction": METRIC_DIRECTION[metric_name],
                    "n_sequences_complete": n,
                    "n_methods": k,
                    "friedman_chi2": np.nan,
                    "friedman_p": np.nan,
                    "kendall_w": np.nan,
                    "status": "insufficient_complete_blocks",
                }
            )
            continue

        arrays = [
            complete[m].to_numpy(dtype=float)
            for m in complete.columns
        ]

        test = stats.friedmanchisquare(
            *arrays
        )

        chi2 = float(test.statistic)
        p = float(test.pvalue)

        # Kendall's W for Friedman repeated-measures design.
        w = (
            chi2 / (n * (k - 1.0))
            if n > 0 and k > 1
            else np.nan
        )

        omnibus_rows.append(
            {
                "Metric": metric_name,
                "metric_column": col,
                "direction": METRIC_DIRECTION[metric_name],
                "n_sequences_complete": n,
                "n_methods": k,
                "friedman_chi2": chi2,
                "friedman_p": p,
                "kendall_w": float(w),
                "status": "ok",
            }
        )

        ranks = rank_matrix(
            complete,
            METRIC_DIRECTION[metric_name],
        )

        avg = ranks.mean(axis=0).sort_values()

        for method, average_rank in avg.items():
            rank_rows.append(
                {
                    "Metric": metric_name,
                    "Method": method,
                    "n_sequences_complete": n,
                    "average_rank": float(average_rank),
                }
            )

    return (
        pd.DataFrame(omnibus_rows),
        pd.DataFrame(rank_rows),
    )


def pairwise_wilcoxon_sequence_level(
    seq: pd.DataFrame,
    methods: Sequence[str],
    *,
    alpha: float,
) -> pd.DataFrame:
    """
    Pairwise Wilcoxon tests using SEQUENCES as paired observations.

    Holm correction is performed separately within each metric across all
    method-pair comparisons for that metric.
    """
    all_metric_frames: List[pd.DataFrame] = []

    for metric_name, col in METRICS.items():
        if col not in seq.columns:
            continue

        pivot = metric_pivot(
            seq,
            methods,
            col,
        )

        rows: List[Dict[str, object]] = []

        for method_a, method_b in combinations(
            methods,
            2,
        ):
            if (
                method_a not in pivot.columns
                or method_b not in pivot.columns
            ):
                continue

            pair = (
                pivot[[method_a, method_b]]
                .apply(
                    pd.to_numeric,
                    errors="coerce",
                )
                .dropna()
            )

            a = pair[method_a].to_numpy(dtype=float)
            b = pair[method_b].to_numpy(dtype=float)

            n = len(pair)
            nonzero = int(
                np.count_nonzero(a - b)
            )

            if n < 2:
                statistic = np.nan
                p_raw = np.nan
            elif nonzero == 0:
                statistic = 0.0
                p_raw = 1.0
            else:
                try:
                    test = stats.wilcoxon(
                        a,
                        b,
                        zero_method="pratt",
                        alternative="two-sided",
                        method="auto",
                    )
                    statistic = float(
                        test.statistic
                    )
                    p_raw = float(
                        test.pvalue
                    )
                except ValueError:
                    statistic = np.nan
                    p_raw = np.nan

            effect = paired_rank_biserial(
                a,
                b,
                METRIC_DIRECTION[metric_name],
            )

            rows.append(
                {
                    "Metric": metric_name,
                    "metric_column": col,
                    "direction": METRIC_DIRECTION[metric_name],
                    "method_a": method_a,
                    "method_b": method_b,
                    "n_common_sequences": n,
                    "wilcoxon_statistic": statistic,
                    "p_raw": p_raw,
                    "r_rb_positive_favours_a": effect,
                    "absolute_r_rb": abs(effect),
                    "median_a": (
                        float(np.median(a))
                        if len(a)
                        else np.nan
                    ),
                    "median_b": (
                        float(np.median(b))
                        if len(b)
                        else np.nan
                    ),
                }
            )

        metric_df = pd.DataFrame(rows)

        if not metric_df.empty:
            metric_df["p_holm"] = holm_adjust(
                metric_df["p_raw"].to_numpy(dtype=float)
            )
            metric_df["significant_holm"] = (
                metric_df["p_holm"] < alpha
            )

            metric_df["favours"] = np.where(
                metric_df["r_rb_positive_favours_a"] > 1e-12,
                metric_df["method_a"],
                np.where(
                    metric_df["r_rb_positive_favours_a"] < -1e-12,
                    metric_df["method_b"],
                    "Neither",
                ),
            )

            all_metric_frames.append(
                metric_df
            )

    if not all_metric_frames:
        return pd.DataFrame()

    return pd.concat(
        all_metric_frames,
        ignore_index=True,
    )


def extract_primary_vs_all(
    pairwise: pd.DataFrame,
    primary: str,
) -> pd.DataFrame:
    if pairwise.empty:
        return pd.DataFrame()

    rows: List[Dict[str, object]] = []

    for r in pairwise.itertuples(index=False):
        if (
            r.method_a != primary
            and r.method_b != primary
        ):
            continue

        comparator = (
            r.method_b
            if r.method_a == primary
            else r.method_a
        )

        effect = float(
            r.r_rb_positive_favours_a
        )

        # Reorient so positive ALWAYS favours primary.
        if r.method_b == primary:
            effect = -effect

        rows.append(
            {
                "Primary": primary,
                "Comparator": comparator,
                "Metric": r.Metric,
                "n_common_sequences": r.n_common_sequences,
                "p_raw": r.p_raw,
                "p_holm_full_family": r.p_holm,
                "significant_holm": r.significant_holm,
                "r_rb_positive_favours_primary": effect,
                "absolute_r_rb": abs(effect),
                "favours": (
                    primary
                    if effect > 1e-12
                    else comparator
                    if effect < -1e-12
                    else "Neither"
                ),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# TOP-3 / LATEX TABLES
# =============================================================================

def top3_methods(
    descriptive: pd.DataFrame,
) -> Dict[str, set[str]]:
    result: Dict[str, set[str]] = {}

    for metric_name in METRICS:
        g = descriptive[
            descriptive["Metric"] == metric_name
        ][["Method", "mean"]].dropna()

        ascending = (
            METRIC_DIRECTION[metric_name]
            == "lower"
        )

        g = g.sort_values(
            "mean",
            ascending=ascending,
        )

        result[metric_name] = set(
            g.head(3)["Method"].astype(str)
        )

    return result


def build_all_metric_rank_table(
    average_ranks: pd.DataFrame,
    methods: Sequence[str],
) -> pd.DataFrame:
    """Build a wide table of average ranks for every configured metric."""
    table = pd.DataFrame(
        {
            "Method": list(methods),
        }
    )

    for metric_name, output_column in RANK_TABLE_METRICS:
        table[output_column] = np.nan

        if average_ranks.empty:
            continue

        subset = average_ranks[
            average_ranks["Metric"] == metric_name
        ][["Method", "average_rank"]].drop_duplicates(
            subset=["Method"]
        )

        rank_by_method = subset.set_index(
            "Method"
        )["average_rank"]

        table[output_column] = table["Method"].map(
            rank_by_method
        )

    return table


def render_all_metric_rank_latex(
    rank_table: pd.DataFrame,
    *,
    budget_label: str,
) -> List[str]:
    """Render a copy-ready table containing all metric average ranks."""
    metric_headers = {
        # "Success (%)": r"Success",
        "Time (ms)": r"Time",
        "Delta RSS (MB)": r"$\Delta$RSS",
        "eR (deg)": r"$e_R$",
        "et (deg)": r"$e_t$",
        "Inlier ratio": r"Inlier-ratio",
        # "Inliers": r"Inlier-count",
    }

    # print(rank_table.columns)
    rank_table = rank_table.sort_values('et (deg) rank (lower is better)', ascending=True)

    latex_headers = [
        metric_headers.get(metric_name, f"{metric_name}".replace("rank (lower is better)", "$downarrow$").replace("rank (higher is better)", "$uparrow$"))
        for metric_name, _ in RANK_TABLE_METRICS
    ]
    alignment = "l" + "r" * len(latex_headers)

    lines = [
        r"\begin{table*}[!t]",
        r"\centering",
        (
            rf"\caption{{Average algorithm ranks at {budget_label}. Rank 1 is "
            r"best; ranking direction follows each metric. Ranks are averaged "
            r"across complete sequence blocks.}"
        ),
        r"\label{tab:cross_sequence_all_metric_ranks}",
        r"\tiny",
        r"\setlength{\tabcolsep}{2.5pt}",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{{alignment}}}",
        r"\toprule",
        "Method & " + " & ".join(latex_headers) + r" \\",
        r"\midrule",
    ]

    rank_columns = [
        output_column
        for _, output_column in RANK_TABLE_METRICS
    ]

    for _, row in rank_table.iterrows():
        values = []
        for column in rank_columns:
            value = row[column]
            values.append(
                "--"
                if pd.isna(value)
                else f"{float(value):.2f}"
            )

        lines.append(
            f"""{latex_escape(row['Method'].replace("--","-").replace("EdgeFusion", "EF").replace("Mutual-Lowe", "ML"))} & """
            + " & ".join(values)
            + r" \\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\end{table*}",
    ]

    return lines


def render_descriptive_latex(
    formatted: pd.DataFrame,
    top3: Mapping[str, set[str]],
    *,
    budget_label: str,
    nboot: int,
) -> List[str]:
    lines = [
        r"\begin{table*}[!t]",
        r"\centering",
        (
            rf"\caption{{Cross-sequence performance at {latex_escape(budget_label)}. "
            rf"Each entry is mean [95\% sequence-bootstrap CI] / median across "
            rf"sequences, using {nboot} sequence-level bootstrap resamples. "
            r"Each sequence contributes one summary value regardless of its "
            r"number of image pairs.}}"
        ),
        r"\label{tab:cross_sequence_all_metrics}",
        r"\tiny",
        r"\setlength{\tabcolsep}{2.2pt}",
        r"\renewcommand{\arraystretch}{1.10}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        (
            r"Method & Success (\%) & Time (ms) & $\Delta$RSS (MB) & "
            r"$e_R$ ($^\circ$) & $e_t$ ($^\circ$) & "
            r"Inlier ratio & Inliers \\"
        ),
        r"\midrule",
    ]

    for _, row in formatted.iterrows():
        method = str(row["Method"])
        values = [latex_escape(method)]

        for metric_name in METRICS:
            value = str(row.get(metric_name, "--"))

            if method in top3.get(
                metric_name,
                set(),
            ):
                value = (
                    r"\textbf{"
                    + value
                    + "}"
                )

            values.append(value)

        lines.append(
            " & ".join(values) + r" \\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table*}",
    ]

    return lines


def render_friedman_latex(
    omnibus: pd.DataFrame,
    *,
    budget_label: str,
) -> List[str]:
    lines = [
        r"\begin{table}[!t]",
        r"\centering",
        (
            rf"\caption{{Sequence-level Friedman tests at "
            rf"{latex_escape(budget_label)}. Sequences are the repeated-measures "
            r"blocks.}}"
        ),
        r"\label{tab:cross_sequence_friedman}",
        r"\scriptsize",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Metric & $N$ & $K$ & $\chi_F^2$ & $p$ / Kendall's $W$ \\",
        r"\midrule",
    ]

    for r in omnibus.itertuples(index=False):
        if not np.isfinite(r.friedman_chi2):
            chi = "--"
            pt = "--"
            wt = "--"
        else:
            chi = f"{r.friedman_chi2:.2f}"
            pt = p_text(float(r.friedman_p))
            wt = f"{r.kendall_w:.3f}"

        lines.append(
            f"{latex_escape(r.Metric)} & "
            f"{int(r.n_sequences_complete)} & "
            f"{int(r.n_methods)} & "
            f"{chi} & {pt} / {wt} \\\\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    return lines


def render_primary_vs_all_latex(
    primary_df: pd.DataFrame,
    *,
    primary: str,
    budget_label: str,
) -> List[str]:
    lines = [
        r"\begin{table*}[!t]",
        r"\centering",
        (
            rf"\caption{{Sequence-level Wilcoxon--Holm comparisons of "
            rf"{latex_escape(primary)} with the other methods at "
            rf"{latex_escape(budget_label)}. Positive $r_{{rb}}$ favours "
            r"the primary method. Holm adjustment is inherited from the full "
            r"all-pairs family within each metric.}}"
        ),
        r"\label{tab:cross_sequence_primary_vs_all}",
        r"\tiny",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{lllrrrl}",
        r"\toprule",
        (
            r"Comparator & Metric & $N$ & $p_{\mathrm{Holm}}$ & "
            r"$r_{rb}$ & $|r_{rb}|$ & Favours \\"
        ),
        r"\midrule",
    ]

    if primary_df.empty:
        lines.append(
            r"\multicolumn{7}{c}{No primary-method comparisons available.} \\"
        )
    else:
        for r in primary_df.itertuples(index=False):
            lines.append(
                f"{latex_escape(r.Comparator)} & "
                f"{latex_escape(r.Metric)} & "
                f"{int(r.n_common_sequences)} & "
                f"{p_text(float(r.p_holm_full_family))} & "
                f"{float(r.r_rb_positive_favours_primary):+.3f} & "
                f"{float(r.absolute_r_rb):.3f} & "
                f"{latex_escape(r.favours)} \\\\"
            )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]

    return lines


# =============================================================================
# PARETO PLOT
# =============================================================================

def pareto_frontier(
    df: pd.DataFrame,
    xcol: str,
    ycol: str,
) -> pd.DataFrame:
    pts = df[
        [xcol, ycol]
    ].to_numpy(dtype=float)

    is_pareto = np.ones(
        len(pts),
        dtype=bool,
    )

    for i in range(len(pts)):
        xi, yi = pts[i]

        dominated = (
            (pts[:, 0] <= xi)
            & (pts[:, 1] <= yi)
            & (
                (pts[:, 0] < xi)
                | (pts[:, 1] < yi)
            )
        )

        # Ignore self; the strict condition already prevents self domination.
        if dominated.any():
            is_pareto[i] = False

    out = df.copy()
    out["pareto_optimal"] = is_pareto

    return out


def build_pareto_data(
    descriptive: pd.DataFrame,
    methods: Sequence[str],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    for method in methods:
        def get(metric: str, field: str) -> float:
            s = descriptive[
                (descriptive["Method"] == method)
                & (descriptive["Metric"] == metric)
            ]
            if s.empty:
                return np.nan
            return float(s.iloc[0][field])

        rows.append(
            {
                "Method": method,
                # Use median across sequences for plot coordinates.
                "Time_ms": get(
                    "Time (ms)",
                    "median",
                ),
                "TranslationError_deg": get(
                    "et (deg)",
                    "median",
                ),
                "RotationError_deg": get(
                    "eR (deg)",
                    "median",
                ),
                "RSS_MB": get(
                    "Delta RSS (MB)",
                    "median",
                ),
            }
        )

    plot_df = pd.DataFrame(rows).dropna(
        subset=[
            "Time_ms",
            "TranslationError_deg",
        ]
    )

    if plot_df.empty:
        return plot_df

    plot_df = pareto_frontier(
        plot_df,
        "Time_ms",
        "TranslationError_deg",
    ).reset_index(drop=True)

    plot_df["ID"] = np.arange(
        1,
        len(plot_df) + 1,
    )

    return plot_df


def save_pareto_plot(
    plot_df: pd.DataFrame,
    output_dir: Path,
    *,
    budget_label: str,
) -> None:
    if plot_df.empty:
        return

    fig, ax = plt.subplots(
        figsize=(11, 7)
    )

    for _, row in plot_df.iterrows():
        method = row["Method"]
        style = STYLE_MAP.get(
            method,
            {"color": "tab:blue", "marker": "o"},
        )

        ax.scatter(
            row["Time_ms"],
            row["TranslationError_deg"],
            s=140,
            c=style["color"],
            marker=style["marker"],
            edgecolors="black",
            linewidths=0.8,
            zorder=3,
        )

        ax.annotate(
            str(int(row["ID"])),
            (
                row["Time_ms"],
                row["TranslationError_deg"],
            ),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=11,
            weight="bold",
        )

    front = plot_df[
        plot_df["pareto_optimal"]
    ].sort_values("Time_ms")

    if not front.empty:
        ax.plot(
            front["Time_ms"],
            front["TranslationError_deg"],
            linestyle="--",
            linewidth=1.5,
            color="black",
            alpha=0.8,
            zorder=2,
        )

    ax.set_xlabel(
        "Median runtime (ms)",
        fontsize=15,
    )


    ax.set_ylabel("Median sign-invariant translation error across sequences (deg)", fontsize=15)

    ax.set_title(
        f"Cross-sequence runtime--translation trade-off ({budget_label})",
        fontsize=19,
    )
    ax.tick_params(
        axis="both",
        labelsize=15,
    )
    ax.grid(
        True,
        alpha=0.25,
    )

    handles = []
    labels = []

    for _, row in plot_df.sort_values("ID").iterrows():
        method = row["Method"]
        style = STYLE_MAP.get(
            method,
            {"color": "tab:blue", "marker": "o"},
        )

        handle = plt.Line2D(
            [0],
            [0],
            marker=style["marker"],
            color="w",
            markerfacecolor=style["color"],
            markeredgecolor="black",
            markersize=9,
            linestyle="None",
        )
        handles.append(handle)
        labels.append(
            f"{int(row['ID'])}. {method}"
        )

    handles.append(
        plt.Line2D(
            [0],
            [0],
            color="black",
            linestyle="--",
            linewidth=1.5,
        )
    )
    labels.append(
        "Pareto frontier"
    )
    labels = [e.replace("--", "-").replace("Mutual-Lowe", "MR").replace("EdgeFusion", "EF")  for e in labels]


    # ax.legend(
    #     handles,
    #     labels,
    #     loc="upper center",
    #     bbox_to_anchor=(0.5, -0.16),
    #     ncol=3,
    #     # bbox_to_anchor=(0.88, 0.9),
    #     # ncol=1,
    #     frameon=True,
    #     fontsize=14,
    #     title="Methods",
    #     title_fontsize=10,
    #     columnspacing=1.2,
    #     handletextpad=0.5,
    #     borderaxespad=0.0,
    # )

    ax.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=4,
        # bbox_to_anchor=(0.88, 0.9),
        # ncol=1,
        frameon=True,
        fontsize=14,
        title="Methods",
        title_fontsize=10,
        columnspacing=1.2,
        handletextpad=0.5,
        borderaxespad=0.0,
    )


    fig.tight_layout()
    fig.subplots_adjust(
        bottom=0.32
    )

    fig.savefig(
        output_dir
        / "pareto_cross_sequence_runtime_translation.png",
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        output_dir
        / "pareto_cross_sequence_runtime_translation.pdf",
        bbox_inches="tight",
    )

    plt.close(fig)


# =============================================================================
# INPUT PREPARATION
# =============================================================================

def load_and_prepare(
    input_csv: Path,
    *,
    dataset_prefix: str,
) -> pd.DataFrame:
    df = pd.read_csv(
        input_csv
    )

    if "Dataset" not in df.columns:
        raise ValueError(
            "Input must contain a 'Dataset' column."
        )

    if (
        "method" not in df.columns
        and "method_label" not in df.columns
        and "method_label_pub" not in df.columns
    ):
        raise ValueError(
            "Input must contain 'method', 'method_label', "
            "or 'method_label_pub'."
        )

    missing_metrics = [
        col
        for col in METRICS.values()
        if col not in df.columns
    ]

    if missing_metrics:
        print(
            "[warning] Missing metric columns; they will be skipped: "
            + ", ".join(missing_metrics),
            file=sys.stderr,
        )

    seq = df[
        df["Dataset"]
        .astype(str)
        .str.startswith(dataset_prefix)
    ].copy()

    if seq.empty:
        raise ValueError(
            f"No datasets start with prefix '{dataset_prefix}'."
        )

    seq["Method"] = seq.apply(
        canonical_method_label,
        axis=1,
    )

    # Keep any unknown method too, but canonical methods get stable ordering.
    seq["Dataset"] = seq[
        "Dataset"
    ].astype(str)

    # Normalize NFeatures when present.
    if "NFeatures" in seq.columns:
        seq["NFeatures"] = pd.to_numeric(
            seq["NFeatures"],
            errors="coerce",
        )

    return seq


# =============================================================================
# ONE-BUDGET ANALYSIS
# =============================================================================

def analyze_one_budget(
    seq: pd.DataFrame,
    output_dir: Path,
    *,
    budget_label: str,
    nboot: int,
    confidence: float,
    seed: int,
    alpha: float,
    primary: str,
) -> Dict[str, object]:
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save exact sequence-level input used for this analysis.
    seq.to_csv(
        output_dir / "analysis_input_sequence_level.csv",
        index=False,
    )

    methods = ordered_present_methods(
        seq["Method"].astype(str)
    )

    if len(methods) < 2:
        raise ValueError(
            "At least two methods are required."
        )

    # -------------------------------------------------------------------------
    # 1. DESCRIPTIVE CROSS-SEQUENCE STATISTICS
    # -------------------------------------------------------------------------
    descriptive = sequence_descriptive_statistics(
        seq,
        methods,
        confidence=confidence,
        nboot=nboot,
        seed=seed,
    )

    # descriptive.to_csv(
    #     output_dir
    #     / "cross_sequence_descriptive_long.csv",
    #     index=False,
    # )

    formatted = build_formatted_summary(
        descriptive,
        methods,
    )
    # formatted.to_csv(
    #     output_dir
    #     / "cross_sequence_all_metrics_summary.csv",
    #     index=False,
    # )

    numeric_wide = build_numeric_wide_summary(
        descriptive,
        methods,
    )
    # numeric_wide.to_csv(
    #     output_dir
    #     / "cross_sequence_all_metrics_summary_numeric.csv",
    #     index=False,
    # )

    # Top-3 only affects table formatting, not statistical inference.
    top3 = top3_methods(
        descriptive
    )

    descriptive_tex = render_descriptive_latex(
        formatted,
        top3,
        budget_label=budget_label,
        nboot=nboot,
    )
    (
        output_dir
        / "cross_sequence_all_metrics_summary.tex"
    ).write_text(
        "\n".join(descriptive_tex) + "\n",
        encoding="utf-8",
    )

    # -------------------------------------------------------------------------
    # 2. CONFIRMATORY SEQUENCE-LEVEL TESTS
    # -------------------------------------------------------------------------
    omnibus, average_ranks = friedman_sequence_level(
        seq,
        methods,
    )

    # omnibus.to_csv(
    #     output_dir
    #     / "friedman_sequence_level.csv",
    #     index=False,
    # )

    # Save the long-form ranks used by the Friedman/CD-style analysis.
    average_ranks.to_csv(
        output_dir
        / "average_ranks_sequence_level.csv",
        index=False,
    )

    # Save a wide table containing ranks for every configured metric.
    all_metric_ranks = build_all_metric_rank_table(
        average_ranks,
        methods,
    )

    all_metric_ranks = all_metric_ranks.sort_values('et (deg) rank (lower is better)', ascending=True)

    all_metric_ranks.round(2).to_csv(
        output_dir
        / "average_ranks_all_metrics.csv",
        index=False,
    )

    rank_tex = render_all_metric_rank_latex(
        all_metric_ranks,
        budget_label=budget_label,
    )
    (
        output_dir
        / "average_ranks_all_metrics.tex"
    ).write_text(
        "\n".join(rank_tex) + "\n",
        encoding="utf-8",
    )

    pairwise = pairwise_wilcoxon_sequence_level(
        seq,
        methods,
        alpha=alpha,
    )

    # pairwise.to_csv(
    #     output_dir
    #     / "pairwise_wilcoxon_holm_sequence_level.csv",
    #     index=False,
    # )

    # Canonicalize primary selector.
    primary_pub = RAW_TO_PUB.get(
        primary,
        LABEL_ALIASES.get(primary, primary),
    )

    primary_vs_all = extract_primary_vs_all(
        pairwise,
        primary_pub,
    )

    # primary_vs_all.to_csv(
    #     output_dir
    #     / "primary_vs_all_wilcoxon_holm_rrb.csv",
    #     index=False,
    # )

    friedman_tex = render_friedman_latex(
        omnibus,
        budget_label=budget_label,
    )
    (
        output_dir
        / "friedman_sequence_level.tex"
    ).write_text(
        "\n".join(friedman_tex) + "\n",
        encoding="utf-8",
    )

    primary_tex = render_primary_vs_all_latex(
        primary_vs_all,
        primary=primary_pub,
        budget_label=budget_label,
    )
    (
        output_dir
        / "primary_vs_all_wilcoxon_holm_rrb.tex"
    ).write_text(
        "\n".join(primary_tex) + "\n",
        encoding="utf-8",
    )

    # -------------------------------------------------------------------------
    # PARETO DATA / FIGURE
    # -------------------------------------------------------------------------
    pareto = build_pareto_data(
        descriptive,
        methods,
    )

    # pareto.to_csv(
    #     output_dir
    #     / "pareto_cross_sequence_runtime_translation_data.csv",
    #     index=False,
    # )

    save_pareto_plot(
        pareto,
        output_dir,
        budget_label=budget_label,
    )

    # -------------------------------------------------------------------------
    # CONSOLIDATED LATEX
    # -------------------------------------------------------------------------
    (
        output_dir
        / "PAPER_CROSS_SEQUENCE_TABLES.tex"
    ).write_text(
        "\n\n".join(
            [
                "% Descriptive sequence-level summary",
                "\n".join(descriptive_tex),
                "% Friedman omnibus tests",
                "\n".join(friedman_tex),
                "% Primary-vs-all Wilcoxon-Holm + r_rb",
                "\n".join(primary_tex),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    # -------------------------------------------------------------------------
    # REVIEW WORKBOOK
    # -------------------------------------------------------------------------
    try:
        with pd.ExcelWriter(
            output_dir
            / "cross_sequence_statistical_results.xlsx",
            engine="openpyxl",
        ) as writer:
            formatted.to_excel(
                writer,
                sheet_name="Paper_Summary",
                index=False,
            )
            descriptive.to_excel(
                writer,
                sheet_name="Descriptive_Long",
                index=False,
            )
            numeric_wide.to_excel(
                writer,
                sheet_name="Descriptive_Wide",
                index=False,
            )
            omnibus.to_excel(
                writer,
                sheet_name="Friedman",
                index=False,
            )
            average_ranks.to_excel(
                writer,
                sheet_name="Average_Ranks",
                index=False,
            )
            all_metric_ranks.to_excel(
                writer,
                sheet_name="Ranks_All_Metrics",
                index=False,
            )
            pairwise.to_excel(
                writer,
                sheet_name="Wilcoxon_Holm",
                index=False,
            )
            primary_vs_all.to_excel(
                writer,
                sheet_name="Primary_vs_All",
                index=False,
            )
            pareto.to_excel(
                writer,
                sheet_name="Pareto",
                index=False,
            )
    except Exception as exc:
        print(
            f"[warning] Excel workbook not written: {exc}",
            file=sys.stderr,
        )

    metadata = {
        "budget": budget_label,
        "n_sequences_unique": int(
            seq["Dataset"].nunique()
        ),
        "n_methods": len(methods),
        "methods": methods,
        "metrics": {
            k: v
            for k, v in METRICS.items()
            if v in seq.columns
        },
        "bootstrap_unit": "sequence",
        "bootstrap_resamples": nboot,
        "confidence": confidence,
        "friedman_block": "sequence",
        "wilcoxon_pairing_unit": "sequence",
        "holm_family": (
            "all method-pair comparisons separately within each metric"
        ),
        "rank_biserial_orientation": (
            "positive favours method_a; primary table is re-oriented "
            "so positive favours the primary method"
        ),
        "average_rank_table": {
            "metrics": list(METRICS.keys()),
            "rank_1_is_best": True,
            "ranking_unit": "complete sequence blocks",
            "directions": {
                metric_name: METRIC_DIRECTION[metric_name]
                for metric_name in METRICS
            },
        },
        "alpha": alpha,
        "primary_method": primary_pub,
    }

    (
        output_dir
        / "analysis_metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    # Console report.
    print("\n" + "=" * 100)
    print(f"CROSS-SEQUENCE ANALYSIS: {budget_label}")
    print("=" * 100)

    print(
        f"Sequences: {seq['Dataset'].nunique()} | "
        f"Methods: {len(methods)} | "
        f"Bootstrap resamples: {nboot}"
    )

    print("\nDESCRIPTIVE SUMMARY")
    display_cols = [
        "Method",
        *[
            m
            for m in METRICS
            if m in formatted.columns
        ],
    ]
    print(
        formatted[display_cols]
        .to_string(index=False)
    )

    print("\nFRIEDMAN / KENDALL W")
    if not omnibus.empty:
        print(
            omnibus[
                [
                    "Metric",
                    "n_sequences_complete",
                    "n_methods",
                    "friedman_chi2",
                    "friedman_p",
                    "kendall_w",
                    "status",
                ]
            ].to_string(index=False)
        )

    print("\nAVERAGE RANKS: ALL METRICS (rank 1 = best)")
    print(
        all_metric_ranks.to_string(index=False)
    )

    print("\nPRIMARY VS ALL")
    if primary_vs_all.empty:
        print(
            f"No comparisons found for {primary_pub}."
        )
    else:
        print(
            primary_vs_all[
                [
                    "Comparator",
                    "Metric",
                    "n_common_sequences",
                    "p_holm_full_family",
                    "r_rb_positive_favours_primary",
                    "favours",
                ]
            ].to_string(index=False)
        )

    print(
        f"\nSaved outputs: {output_dir}"
    )

    return metadata


# =============================================================================
# OPTIONAL CROSS-BUDGET SENSITIVITY INFERENCE
# =============================================================================

def analyze_budget_effects(
    seq: pd.DataFrame,
    output_dir: Path,
    *,
    alpha: float,
) -> None:
    """
    When a master input contains multiple feature budgets, test the effect of
    feature budget WITHIN each method, using sequence as the repeated block.

    For each Method x Metric:
        Friedman across budgets
        pairwise Wilcoxon across budgets
        Holm correction within that Method x Metric family
        paired rank-biserial effect (positive favours budget A)
    """
    if "NFeatures" not in seq.columns:
        return

    budgets = sorted(
        int(x)
        for x in seq["NFeatures"].dropna().unique()
    )

    if len(budgets) < 2:
        return

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    omnibus_rows: List[Dict[str, object]] = []
    pairwise_rows: List[Dict[str, object]] = []

    methods = ordered_present_methods(
        seq["Method"].astype(str)
    )

    for method in methods:
        method_df = seq[
            seq["Method"] == method
        ].copy()

        for metric_name, col in METRICS.items():
            if col not in method_df.columns:
                continue

            pivot = method_df.pivot_table(
                index="Dataset",
                columns="NFeatures",
                values=col,
                aggfunc="median",
            )

            present_budgets = [
                b for b in budgets
                if b in pivot.columns
            ]

            pivot = pivot.reindex(
                columns=present_budgets
            )

            complete = pivot.dropna(
                axis=0,
                how="any",
            )

            n = len(complete)
            k = len(complete.columns)

            if n >= 2 and k >= 3:
                arrays = [
                    complete[b].to_numpy(dtype=float)
                    for b in complete.columns
                ]
                test = stats.friedmanchisquare(
                    *arrays
                )
                chi2 = float(test.statistic)
                p = float(test.pvalue)
                w = chi2 / (
                    n * (k - 1.0)
                )
            else:
                chi2 = np.nan
                p = np.nan
                w = np.nan

            omnibus_rows.append(
                {
                    "Method": method,
                    "Metric": metric_name,
                    "n_sequences_complete": n,
                    "n_budgets": k,
                    "friedman_chi2": chi2,
                    "friedman_p": p,
                    "kendall_w": w,
                }
            )

            local_rows = []

            for budget_a, budget_b in combinations(
                present_budgets,
                2,
            ):
                pair = (
                    pivot[[budget_a, budget_b]]
                    .apply(
                        pd.to_numeric,
                        errors="coerce",
                    )
                    .dropna()
                )

                a = pair[
                    budget_a
                ].to_numpy(dtype=float)
                b = pair[
                    budget_b
                ].to_numpy(dtype=float)

                if len(pair) < 2:
                    stat = np.nan
                    p_raw = np.nan
                elif np.count_nonzero(a - b) == 0:
                    stat = 0.0
                    p_raw = 1.0
                else:
                    try:
                        test = stats.wilcoxon(
                            a,
                            b,
                            zero_method="pratt",
                            alternative="two-sided",
                            method="auto",
                        )
                        stat = float(
                            test.statistic
                        )
                        p_raw = float(
                            test.pvalue
                        )
                    except ValueError:
                        stat = np.nan
                        p_raw = np.nan

                effect = paired_rank_biserial(
                    a,
                    b,
                    METRIC_DIRECTION[metric_name],
                )

                local_rows.append(
                    {
                        "Method": method,
                        "Metric": metric_name,
                        "budget_a": int(budget_a),
                        "budget_b": int(budget_b),
                        "n_common_sequences": len(pair),
                        "wilcoxon_statistic": stat,
                        "p_raw": p_raw,
                        "r_rb_positive_favours_budget_a": effect,
                    }
                )

            local_df = pd.DataFrame(
                local_rows
            )

            if not local_df.empty:
                local_df["p_holm"] = holm_adjust(
                    local_df["p_raw"].to_numpy(dtype=float)
                )
                local_df["significant_holm"] = (
                    local_df["p_holm"] < alpha
                )
                pairwise_rows.extend(
                    local_df.to_dict("records")
                )

    pd.DataFrame(
        omnibus_rows
    ).to_csv(
        output_dir
        / "budget_effect_friedman_sequence_level.csv",
        index=False,
    )

    pd.DataFrame(
        pairwise_rows
    ).to_csv(
        output_dir
        / "budget_effect_wilcoxon_holm_rrb_sequence_level.csv",
        index=False,
    )

    print(
        f"\nSaved cross-budget sensitivity tests: "
        f"{output_dir}"
    )


# =============================================================================
# CLI / MAIN
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Cross-sequence descriptive and confirmatory sequence-level "
            "statistics for EdgeFusion benchmark summaries."
        )
    )

    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="Sequence-level aggregate CSV.",
    )

    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Output directory.",
    )

    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=DEFAULT_NBOOT,
        help=(
            "Number of SEQUENCE bootstrap resamples "
            f"(default: {DEFAULT_NBOOT})."
        ),
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Bootstrap confidence level (default: 0.95).",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help="Family-wise significance level (default: 0.05).",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    parser.add_argument(
        "--dataset-prefix",
        default=DATASET_PREFIX,
    )

    parser.add_argument(
        "--primary-method",
        default="edgefusion_no_grid",
        help=(
            "Primary method raw ID or publication label for the compact "
            "primary-vs-all table."
        ),
    )

    return parser


def main() -> int:
    args = build_parser().parse_args()

    input_csv = Path(
        args.input
    ).expanduser().resolve()

    output_root = Path(
        args.output
    ).expanduser().resolve()

    if not input_csv.exists():
        raise FileNotFoundError(
            input_csv
        )

    seq = load_and_prepare(
        input_csv,
        dataset_prefix=args.dataset_prefix,
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # MULTI-BUDGET INPUT
    # -------------------------------------------------------------------------
    if (
        "NFeatures" in seq.columns
        and seq["NFeatures"].notna().any()
    ):
        budgets = sorted(
            int(x)
            for x in seq["NFeatures"].dropna().unique()
        )
    else:
        budgets = []

    if len(budgets) > 1:
        print(
            f"Detected multiple feature budgets: {budgets}. "
            "Each budget will be analyzed separately."
        )

        metadata_all = []

        for budget in budgets:
            budget_seq = seq[
                seq["NFeatures"] == budget
            ].copy()

            budget_output = (
                output_root
                / f"N{budget}"
            )

            metadata = analyze_one_budget(
                budget_seq,
                budget_output,
                budget_label=f"$N_f={budget}$",
                nboot=args.bootstrap_resamples,
                confidence=args.confidence,
                seed=args.seed + budget,
                alpha=args.alpha,
                primary=args.primary_method,
            )

            metadata_all.append(
                metadata
            )

        # Additional feature-budget sensitivity inference.
        analyze_budget_effects(
            seq,
            output_root
            / "budget_effects",
            alpha=args.alpha,
        )

        (
            output_root
            / "analysis_metadata_all_budgets.json"
        ).write_text(
            json.dumps(
                {
                    "input": str(input_csv),
                    "budgets": budgets,
                    "per_budget": metadata_all,
                    "budget_effect_analysis": True,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    else:
        # ---------------------------------------------------------------------
        # SINGLE-BUDGET INPUT
        # ---------------------------------------------------------------------
        if len(budgets) == 1:
            budget = budgets[0]
            budget_label = f"$N_f={budget}$"
        else:
            budget_label = "the evaluated feature budget"

        analyze_one_budget(
            seq,
            output_root,
            budget_label=budget_label,
            nboot=args.bootstrap_resamples,
            confidence=args.confidence,
            seed=args.seed,
            alpha=args.alpha,
            primary=args.primary_method,
        )

    print("\nAnalysis complete.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(
            f"ERROR: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(2)
