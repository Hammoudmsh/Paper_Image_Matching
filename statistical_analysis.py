#!/usr/bin/env python3
"""Statistical analysis and publication-ready figures for matching benchmarks.

The script consumes ``per_pair_results.csv`` produced by
``matching_comparison_all.py`` and writes:

* descriptive statistics and bootstrap confidence intervals;
* Friedman omnibus tests on complete paired blocks;
* pairwise Wilcoxon signed-rank tests with Holm correction;
* paired rank-biserial effect sizes;
* exact McNemar tests for pose-recovery success;
* average-rank tables and Critical Difference (CD) diagrams;
* figure-source CSV files and PNG/PDF/SVG figures.

Image pairs are treated as repeated-measures blocks. Results from one video
sequence can be temporally correlated, so the tests should be described as
paired/exploratory unless independent sequences are used as the statistical
blocks.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy import stats


DEFAULT_METRICS: Dict[str, str] = {
    "total_ms": "lower",
    "rotation_error_deg": "lower",
    "translation_error_sign_invariant_deg": "lower",
    "inlier_ratio": "higher",
    "geometric_inliers": "higher",
    "peak_rss_delta_mb": "lower",
}

POSE_METRICS = {
    "rotation_error_deg",
    "translation_error_deg",
    "translation_error_sign_invariant_deg",
}

DISPLAY_NAMES = {
    "total_ms": "End-to-end runtime (ms)",
    "rotation_error_deg": "Rotation error (deg)",
    "translation_error_sign_invariant_deg": "Translation-direction error (deg)",
    "inlier_ratio": "Geometric inlier ratio",
    "geometric_inliers": "Geometric inliers",
    "peak_rss_delta_mb": "Peak RSS increase (MB)",
}


@dataclass(frozen=True)
class MetricResult:
    metric: str
    direction: str
    n_blocks: int
    n_methods: int
    friedman_statistic: float
    friedman_pvalue: float
    critical_difference: float


def _finite(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def parse_method_selector(value: str, available: Sequence[str]) -> List[str]:
    available_list = list(dict.fromkeys(str(x) for x in available))
    selector = (value or "all").strip()
    if selector.lower() in {"all", "*", ""}:
        return available_list
    requested = [x.strip() for x in selector.split(",") if x.strip()]
    missing = [x for x in requested if x not in available_list]
    if missing:
        raise ValueError(f"Unknown methods in --methods: {missing}")
    return requested


def holm_adjust(pvalues: Sequence[float]) -> np.ndarray:
    """Holm step-down family-wise error-rate adjustment."""
    p = np.asarray(pvalues, dtype=float)
    adjusted = np.full_like(p, np.nan)
    finite_idx = np.flatnonzero(np.isfinite(p))
    if len(finite_idx) == 0:
        return adjusted
    order_local = np.argsort(p[finite_idx])
    order = finite_idx[order_local]
    m = len(order)
    running = 0.0
    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * p[idx])
        running = max(running, value)
        adjusted[idx] = running
    return adjusted


def bootstrap_ci(
    values: np.ndarray,
    statistic: str,
    confidence: float,
    resamples: int,
    rng: np.random.Generator,
) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    if len(values) == 1 or resamples <= 0:
        v = float(values[0])
        return v, v
    n = len(values)
    sampled = rng.choice(values, size=(resamples, n), replace=True)
    if statistic == "mean":
        estimates = sampled.mean(axis=1)
    else:
        estimates = np.median(sampled, axis=1)
    alpha = 1.0 - confidence
    return (
        float(np.quantile(estimates, alpha / 2.0)),
        float(np.quantile(estimates, 1.0 - alpha / 2.0)),
    )


def descriptive_statistics(
    data: pd.DataFrame,
    methods: Sequence[str],
    metrics: Mapping[str, str],
    confidence: float,
    bootstrap_resamples: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, object]] = []
    for method in methods:
        group = data[data["method"] == method]
        successes = pd.to_numeric(group.get("success", 0), errors="coerce").fillna(0).astype(int)
        for metric, direction in metrics.items():
            if metric not in group.columns:
                continue
            values = _finite(group[metric]).dropna().to_numpy(dtype=float)
            median_ci = bootstrap_ci(values, "median", confidence, bootstrap_resamples, rng)
            mean_ci = bootstrap_ci(values, "mean", confidence, bootstrap_resamples, rng)
            q1 = float(np.quantile(values, 0.25)) if len(values) else float("nan")
            q3 = float(np.quantile(values, 0.75)) if len(values) else float("nan")
            rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "direction": direction,
                    "total_pairs": int(len(group)),
                    "successful_pairs": int(successes.sum()),
                    "success_rate": float(successes.mean()) if len(successes) else float("nan"),
                    "n_finite": int(len(values)),
                    "mean": float(np.mean(values)) if len(values) else float("nan"),
                    "std": float(np.std(values, ddof=1)) if len(values) > 1 else float("nan"),
                    "median": float(np.median(values)) if len(values) else float("nan"),
                    "q1": q1,
                    "q3": q3,
                    "iqr": q3 - q1 if len(values) else float("nan"),
                    "minimum": float(np.min(values)) if len(values) else float("nan"),
                    "maximum": float(np.max(values)) if len(values) else float("nan"),
                    "p90": float(np.quantile(values, 0.90)) if len(values) else float("nan"),
                    "p95": float(np.quantile(values, 0.95)) if len(values) else float("nan"),
                    "median_ci_low": median_ci[0],
                    "median_ci_high": median_ci[1],
                    "mean_ci_low": mean_ci[0],
                    "mean_ci_high": mean_ci[1],
                    "confidence_level": confidence,
                    "bootstrap_resamples": bootstrap_resamples,
                }
            )
    return pd.DataFrame(rows)


def pivot_complete_blocks(
    data: pd.DataFrame,
    methods: Sequence[str],
    metric: str,
) -> pd.DataFrame:
    subset = data[data["method"].isin(methods)].copy()
    subset[metric] = _finite(subset[metric])
    # A duplicated method/pair row would invalidate a paired analysis. The
    # median is deterministic and robust, while duplicates are reported.
    pivot = subset.pivot_table(index="pair_index", columns="method", values=metric, aggfunc="median")
    pivot = pivot.reindex(columns=list(methods))
    return pivot.dropna(axis=0, how="any")


def rank_matrix(pivot: pd.DataFrame, direction: str) -> pd.DataFrame:
    if pivot.empty:
        return pivot.copy()
    values = pivot.to_numpy(dtype=float)
    if direction == "higher":
        values = -values
    ranks = np.vstack([stats.rankdata(row, method="average") for row in values])
    return pd.DataFrame(ranks, index=pivot.index, columns=pivot.columns)


def nemenyi_critical_difference(alpha: float, k: int, n: int) -> float:
    if k < 2 or n < 1:
        return float("nan")
    q_alpha = stats.studentized_range.ppf(1.0 - alpha, k, np.inf) / math.sqrt(2.0)
    return float(q_alpha * math.sqrt(k * (k + 1.0) / (6.0 * n)))


def paired_rank_biserial(a: np.ndarray, b: np.ndarray, direction: str) -> float:
    """Rank-biserial effect; positive values favour method A."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    improvement = (b - a) if direction == "lower" else (a - b)
    improvement = improvement[np.isfinite(improvement)]
    improvement = improvement[improvement != 0]
    if len(improvement) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(improvement), method="average")
    positive = float(ranks[improvement > 0].sum())
    negative = float(ranks[improvement < 0].sum())
    denominator = positive + negative
    return (positive - negative) / denominator if denominator > 0 else 0.0


def pairwise_wilcoxon(
    data: pd.DataFrame,
    methods: Sequence[str],
    metric: str,
    direction: str,
    alpha: float,
) -> pd.DataFrame:
    pivot = data[data["method"].isin(methods)].pivot_table(
        index="pair_index", columns="method", values=metric, aggfunc="median"
    )
    rows: List[Dict[str, object]] = []
    for method_a, method_b in combinations(methods, 2):
        if method_a not in pivot.columns or method_b not in pivot.columns:
            continue
        pair = pivot[[method_a, method_b]].apply(pd.to_numeric, errors="coerce").dropna()
        a = pair[method_a].to_numpy(dtype=float)
        b = pair[method_b].to_numpy(dtype=float)
        nonzero = np.count_nonzero(a - b)
        if len(pair) < 2 or nonzero == 0:
            statistic, pvalue = 0.0, 1.0
        else:
            try:
                test = stats.wilcoxon(a, b, zero_method="pratt", alternative="two-sided", method="auto")
                statistic, pvalue = float(test.statistic), float(test.pvalue)
            except ValueError:
                statistic, pvalue = float("nan"), float("nan")
        median_a = float(np.median(a)) if len(a) else float("nan")
        median_b = float(np.median(b)) if len(b) else float("nan")
        rows.append(
            {
                "metric": metric,
                "direction": direction,
                "method_a": method_a,
                "method_b": method_b,
                "n_common_pairs": int(len(pair)),
                "wilcoxon_statistic": statistic,
                "p_raw": pvalue,
                "median_a": median_a,
                "median_b": median_b,
                "median_difference_a_minus_b": median_a - median_b,
                "rank_biserial_positive_favours_a": paired_rank_biserial(a, b, direction),
            }
        )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["p_holm"] = holm_adjust(result["p_raw"].to_numpy(dtype=float))
        result["significant_holm"] = result["p_holm"] < alpha
    return result


def exact_mcnemar_tests(data: pd.DataFrame, methods: Sequence[str], alpha: float) -> pd.DataFrame:
    pivot = data[data["method"].isin(methods)].pivot_table(
        index="pair_index", columns="method", values="success", aggfunc="max"
    )
    rows: List[Dict[str, object]] = []
    for method_a, method_b in combinations(methods, 2):
        if method_a not in pivot.columns or method_b not in pivot.columns:
            continue
        pair = pivot[[method_a, method_b]].dropna().astype(int)
        a = pair[method_a].to_numpy()
        b = pair[method_b].to_numpy()
        a_only = int(np.count_nonzero((a == 1) & (b == 0)))
        b_only = int(np.count_nonzero((a == 0) & (b == 1)))
        discordant = a_only + b_only
        if discordant == 0:
            pvalue = 1.0
        else:
            pvalue = min(1.0, 2.0 * stats.binom.cdf(min(a_only, b_only), discordant, 0.5))
        rows.append(
            {
                "method_a": method_a,
                "method_b": method_b,
                "n_common_pairs": int(len(pair)),
                "a_success_b_failure": a_only,
                "a_failure_b_success": b_only,
                "discordant_pairs": discordant,
                "success_rate_a": float(a.mean()) if len(a) else float("nan"),
                "success_rate_b": float(b.mean()) if len(b) else float("nan"),
                "success_rate_difference_a_minus_b": float(a.mean() - b.mean()) if len(a) else float("nan"),
                "p_raw": pvalue,
            }
        )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["p_holm"] = holm_adjust(result["p_raw"].to_numpy(dtype=float))
        result["significant_holm"] = result["p_holm"] < alpha
    return result


def _maximal_nonsignificant_intervals(sorted_ranks: Sequence[float], cd: float) -> List[Tuple[int, int]]:
    intervals: List[Tuple[int, int]] = []
    n = len(sorted_ranks)
    for i in range(n):
        j = i
        while j + 1 < n and sorted_ranks[j + 1] - sorted_ranks[i] <= cd + 1e-12:
            j += 1
        if j > i:
            intervals.append((i, j))
    # Remove intervals fully contained in a longer interval.
    maximal: List[Tuple[int, int]] = []
    for interval in intervals:
        if not any(
            other != interval and other[0] <= interval[0] and other[1] >= interval[1]
            for other in intervals
        ):
            maximal.append(interval)
    return maximal


def save_figure(fig: plt.Figure, base_path: Path, dpi: int = 220) -> None:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base_path.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(base_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def plot_cd_diagram(
    average_ranks: pd.Series,
    cd: float,
    title: str,
    output_base: Path,
    alpha: float,
    n_blocks: int,
) -> None:
    average_ranks = average_ranks.sort_values()
    methods = list(average_ranks.index)
    ranks = average_ranks.to_numpy(dtype=float)
    k = len(methods)
    width = max(11.0, 0.55 * k + 7.0)
    height = max(5.5, 0.28 * k + 3.8)
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_xlim(0.5, k + 0.5)
    ax.set_ylim(-0.3, max(3.0, 0.18 * k + 1.9))
    ax.axis("off")

    axis_y = 0.35
    ax.hlines(axis_y, 1, k, linewidth=1.4)
    for rank in range(1, k + 1):
        ax.vlines(rank, axis_y - 0.07, axis_y + 0.07, linewidth=1.0)
        ax.text(rank, axis_y - 0.18, str(rank), ha="center", va="top")
    ax.text((k + 1) / 2.0, -0.20, "Average rank (lower is better)", ha="center", va="top")

    left_count = math.ceil(k / 2)
    left_indices = list(range(left_count))
    right_indices = list(range(left_count, k))
    label_step = 0.32
    base_label_y = 1.15

    for pos, idx in enumerate(left_indices):
        y = base_label_y + pos * label_step
        x_rank = ranks[idx]
        ax.plot([x_rank, x_rank, 0.75], [axis_y + 0.08, y, y], linewidth=0.9)
        ax.text(0.70, y, f"{methods[idx]}  ({x_rank:.2f})", ha="right", va="center")

    for pos, idx in enumerate(reversed(right_indices)):
        y = base_label_y + pos * label_step
        x_rank = ranks[idx]
        ax.plot([x_rank, x_rank, k + 0.25], [axis_y + 0.08, y, y], linewidth=0.9)
        ax.text(k + 0.30, y, f"({x_rank:.2f})  {methods[idx]}", ha="left", va="center")

    cd_y = max(base_label_y + max(len(left_indices), len(right_indices)) * label_step + 0.15, 2.0)
    if math.isfinite(cd):
        cd_start = 1.0
        cd_end = min(k, cd_start + cd)
        ax.hlines(cd_y, cd_start, cd_end, linewidth=2.4)
        ax.vlines([cd_start, cd_end], cd_y - 0.07, cd_y + 0.07, linewidth=1.5)
        ax.text((cd_start + cd_end) / 2.0, cd_y + 0.10, f"CD = {cd:.3f}", ha="center", va="bottom")

        intervals = _maximal_nonsignificant_intervals(ranks, cd)
        for level, (i, j) in enumerate(intervals):
            y = axis_y + 0.22 + level * 0.10
            ax.hlines(y, ranks[i], ranks[j], linewidth=3.0)

    ax.set_title(f"{title}\nN={n_blocks} paired blocks, Nemenyi α={alpha:g}")
    save_figure(fig, output_base)


def plot_boxplot(long_data: pd.DataFrame, metric: str, output_base: Path, log_scale: bool = False) -> None:
    methods = list(dict.fromkeys(long_data["method"].astype(str)))
    arrays = [
        _finite(long_data.loc[long_data["method"] == method, metric]).dropna().to_numpy(dtype=float)
        for method in methods
    ]
    keep = [(m, a) for m, a in zip(methods, arrays) if len(a)]
    if not keep:
        return
    methods, arrays = zip(*keep)
    fig, ax = plt.subplots(figsize=(max(10, 0.65 * len(methods)), 6))
    ax.boxplot(arrays, tick_labels=methods, showfliers=True)
    ax.tick_params(axis="x", rotation=35)
    ax.set_ylabel(DISPLAY_NAMES.get(metric, metric))
    if log_scale and all(np.all(a > 0) for a in arrays):
        ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save_figure(fig, output_base)


def plot_success_rate(data: pd.DataFrame, methods: Sequence[str], output_base: Path) -> pd.DataFrame:
    source = (
        data[data["method"].isin(methods)]
        .groupby("method", as_index=False)["success"]
        .agg(success_rate="mean", successful_pairs="sum", pairs="count")
        .sort_values(["success_rate", "method"], ascending=[False, True])
    )
    fig, ax = plt.subplots(figsize=(max(10, 0.62 * len(source)), 5.5))
    ax.bar(source["method"], source["success_rate"] * 100.0)
    ax.set_ylabel("Pose-recovery success rate (%)")
    ax.set_ylim(0, 105)
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save_figure(fig, output_base)
    return source


def plot_pareto(descriptive: pd.DataFrame, output_base: Path) -> pd.DataFrame:
    med = descriptive.pivot(index="method", columns="metric", values="median")
    succ = descriptive.groupby("method", as_index=True)["success_rate"].max()
    required = ["total_ms", "translation_error_sign_invariant_deg"]
    if not all(c in med.columns for c in required):
        return pd.DataFrame()
    source = med[required].join(succ.rename("success_rate")).dropna().reset_index()
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    ax.scatter(source["total_ms"], source["translation_error_sign_invariant_deg"], s=55)
    for row in source.itertuples(index=False):
        ax.annotate(row.method, (row.total_ms, row.translation_error_sign_invariant_deg), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Median end-to-end runtime (ms)")
    ax.set_ylabel("Median sign-invariant translation error (deg)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    save_figure(fig, output_base)
    return source


def plot_stage_timings(data: pd.DataFrame, methods: Sequence[str], output_base: Path) -> pd.DataFrame:
    stage_columns = [
        c for c in data.columns
        if c.endswith("_ms") and c not in {"total_ms"} and not c.startswith("diag_")
    ]
    if not stage_columns:
        return pd.DataFrame()
    source = data[data["method"].isin(methods)][["method"] + stage_columns].copy()
    for col in stage_columns:
        source[col] = _finite(source[col])
    medians = source.groupby("method")[stage_columns].median().fillna(0.0)
    medians = medians.loc[[m for m in methods if m in medians.index]]
    nonzero_columns = [c for c in stage_columns if medians[c].abs().sum() > 0]
    if medians.empty or not nonzero_columns:
        return pd.DataFrame()
    fig, ax = plt.subplots(figsize=(max(10, 0.65 * len(medians)), 6))
    bottom = np.zeros(len(medians), dtype=float)
    x = np.arange(len(medians))
    for col in nonzero_columns:
        values = medians[col].to_numpy(dtype=float)
        ax.bar(x, values, bottom=bottom, label=col)
        bottom += values
    ax.set_xticks(x, medians.index, rotation=35, ha="right")
    ax.set_ylabel("Median stage time (ms)")
    ax.legend(fontsize=8, ncols=2)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save_figure(fig, output_base)
    return medians.reset_index()


def write_markdown_report(
    path: Path,
    omnibus: pd.DataFrame,
    descriptive: pd.DataFrame,
    methods: Sequence[str],
    alpha: float,
) -> None:
    lines = [
        "# Statistical comparison report",
        "",
        f"Methods: {len(methods)}; family-wise α: {alpha:g}.",
        "",
        "## Statistical design",
        "",
        "Each image pair is treated as a repeated-measures block. Friedman tests compare average ranks on complete blocks; pairwise Wilcoxon signed-rank tests use all available common finite pairs and are adjusted with Holm's procedure. Exact McNemar tests compare pose-recovery success. CD diagrams use the Nemenyi critical difference.",
        "",
        "Pairs extracted from one video sequence may be temporally correlated. For confirmatory inference, repeat the experiment on multiple independent sequences and use sequence-level aggregation or a hierarchical analysis.",
        "",
        "## Omnibus tests",
        "",
        "| Metric | Complete blocks | Methods | Friedman statistic | p-value | Critical difference |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in omnibus.itertuples(index=False):
        lines.append(
            f"| {row.metric} | {row.n_blocks} | {row.n_methods} | "
            f"{row.friedman_statistic:.4f} | {row.friedman_pvalue:.6g} | {row.critical_difference:.4f} |"
        )
    lines.extend([
        "",
        "## Output interpretation",
        "",
        "A small Friedman p-value indicates that at least one method has a different rank distribution. It does not identify which pair differs. Use the Holm-adjusted Wilcoxon tables for pairwise conclusions and report rank-biserial effect sizes with the p-values.",
        "",
        "A CD diagram connects methods whose average-rank separation does not exceed the Nemenyi critical difference. Lower average rank is better after the metric direction has been applied.",
        "",
        "Do not rank skipped methods or rows with zero coverage. Pose-error tests are based on pairs with finite pose estimates; success-rate differences are evaluated separately with McNemar tests.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_analysis(args: argparse.Namespace) -> int:
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    figure_data_dir = output_dir / "figure_data"
    rank_dir = output_dir / "rank_matrices"
    for directory in [table_dir, figure_dir, figure_data_dir, rank_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(input_path)
    required = {"pair_index", "method", "success"}
    missing_required = sorted(required - set(data.columns))
    if missing_required:
        raise ValueError(f"Input is missing required columns: {missing_required}")
    data["method"] = data["method"].astype(str)
    data["success"] = pd.to_numeric(data["success"], errors="coerce").fillna(0).astype(int)
    status_text = data.get("status", pd.Series("", index=data.index)).astype(str).str.lower()
    skipped_mask = status_text.str.startswith("skipped") | status_text.str.contains("not_installed", regex=False)
    for metric_name in DEFAULT_METRICS:
        if metric_name in data.columns:
            data.loc[skipped_mask, metric_name] = np.nan
    if "total_ms" in data.columns:
        data.loc[_finite(data["total_ms"]) <= 0, "total_ms"] = np.nan

    available_methods = list(dict.fromkeys(data["method"].tolist()))
    requested_methods = parse_method_selector(args.methods, available_methods)
    active_methods = []
    for method in requested_methods:
        group = data[data["method"] == method]
        all_skipped = bool(len(group)) and bool(skipped_mask.loc[group.index].all())
        if all_skipped:
            print(f"[warning] excluded unavailable method: {method}", file=sys.stderr)
        else:
            active_methods.append(method)
    methods = active_methods
    if len(methods) < 2:
        raise ValueError("Fewer than two available methods remain after excluding skipped methods")
    data = data[data["method"].isin(methods)].copy()

    metrics: Dict[str, str] = {}
    requested_metrics = [x.strip() for x in args.metrics.split(",") if x.strip()]
    for metric in requested_metrics:
        if metric not in data.columns:
            print(f"[warning] metric not present and skipped: {metric}", file=sys.stderr)
            continue
        metrics[metric] = DEFAULT_METRICS.get(metric, "lower")
    if not metrics:
        raise ValueError("No requested metrics are present in the input table")

    # Preserve the complete benchmark table used for the analysis.
    data.to_csv(output_dir / "analysis_input_snapshot.csv", index=False)

    descriptive = descriptive_statistics(
        data,
        methods,
        metrics,
        args.confidence,
        args.bootstrap_resamples,
        args.seed,
    )
    descriptive.to_csv(table_dir / "descriptive_statistics.csv", index=False)

    omnibus_rows: List[Dict[str, object]] = []
    pairwise_tables: List[pd.DataFrame] = []
    rank_summary_rows: List[Dict[str, object]] = []

    for metric, direction in metrics.items():
        complete = pivot_complete_blocks(data, methods, metric)
        if len(complete) < args.min_complete_pairs:
            print(
                f"[warning] {metric}: only {len(complete)} complete pairs; "
                f"minimum is {args.min_complete_pairs}. CD/Friedman skipped.",
                file=sys.stderr,
            )
        else:
            ranks = rank_matrix(complete, direction)
            avg_ranks = ranks.mean(axis=0).sort_values()
            arrays = [complete[c].to_numpy(dtype=float) for c in complete.columns]
            friedman = stats.friedmanchisquare(*arrays) if len(arrays) >= 3 else None
            cd = nemenyi_critical_difference(args.alpha, len(complete.columns), len(complete))
            omnibus_rows.append(
                {
                    "metric": metric,
                    "direction": direction,
                    "n_blocks": int(len(complete)),
                    "n_methods": int(len(complete.columns)),
                    "friedman_statistic": float(friedman.statistic) if friedman else float("nan"),
                    "friedman_pvalue": float(friedman.pvalue) if friedman else float("nan"),
                    "critical_difference": cd,
                }
            )
            complete.to_csv(rank_dir / f"{metric}_complete_values_wide.csv")
            ranks.to_csv(rank_dir / f"{metric}_ranks_wide.csv")
            ranks.reset_index().melt(id_vars="pair_index", var_name="method", value_name="rank").to_csv(
                figure_data_dir / f"cd_{metric}_ranks_long.csv", index=False
            )
            avg_rank_table = avg_ranks.rename("average_rank").reset_index().rename(columns={"index": "method"})
            avg_rank_table["metric"] = metric
            avg_rank_table["direction"] = direction
            avg_rank_table["n_blocks"] = len(complete)
            avg_rank_table["critical_difference"] = cd
            avg_rank_table.to_csv(table_dir / f"average_ranks_{metric}.csv", index=False)
            rank_summary_rows.extend(avg_rank_table.to_dict("records"))
            plot_cd_diagram(
                avg_ranks,
                cd,
                f"Critical Difference diagram: {DISPLAY_NAMES.get(metric, metric)}",
                figure_dir / f"cd_{metric}",
                args.alpha,
                len(complete),
            )

        pairwise = pairwise_wilcoxon(data, methods, metric, direction, args.alpha)
        pairwise.to_csv(table_dir / f"pairwise_wilcoxon_holm_{metric}.csv", index=False)
        if not pairwise.empty:
            pairwise_tables.append(pairwise)

        # Figure-source long table exactly as plotted.
        columns = [c for c in ["pair_index", "method", "success", metric] if c in data.columns]
        long_source = data[columns].copy()
        long_source[metric] = _finite(long_source[metric])
        long_source.to_csv(figure_data_dir / f"{metric}_long.csv", index=False)
        plot_boxplot(
            long_source.dropna(subset=[metric]),
            metric,
            figure_dir / f"boxplot_{metric}",
            log_scale=(metric in {"total_ms", "peak_rss_delta_mb"}),
        )

    omnibus = pd.DataFrame(omnibus_rows)
    omnibus.to_csv(table_dir / "friedman_omnibus.csv", index=False)
    pd.DataFrame(rank_summary_rows).to_csv(table_dir / "average_ranks_all_metrics.csv", index=False)
    if pairwise_tables:
        pd.concat(pairwise_tables, ignore_index=True).to_csv(table_dir / "pairwise_wilcoxon_holm_all_metrics.csv", index=False)

    mcnemar = exact_mcnemar_tests(data, methods, args.alpha)
    mcnemar.to_csv(table_dir / "pairwise_mcnemar_holm_success.csv", index=False)

    success_source = plot_success_rate(data, methods, figure_dir / "success_rate")
    success_source.to_csv(figure_data_dir / "success_rate.csv", index=False)
    pareto_source = plot_pareto(descriptive, figure_dir / "pareto_runtime_translation")
    pareto_source.to_csv(figure_data_dir / "pareto_runtime_translation.csv", index=False)
    stage_source = plot_stage_timings(data, methods, figure_dir / "stage_timing_stacked")
    stage_source.to_csv(figure_data_dir / "stage_timing_medians.csv", index=False)

    # A long table of every stage timing is useful for custom figures.
    stage_cols = [c for c in data.columns if c.endswith("_ms") and c != "total_ms" and not c.startswith("diag_")]
    if stage_cols:
        stage_long = data[["pair_index", "method"] + stage_cols].melt(
            id_vars=["pair_index", "method"], var_name="stage", value_name="time_ms"
        )
        stage_long["time_ms"] = _finite(stage_long["time_ms"])
        stage_long.to_csv(figure_data_dir / "stage_timings_long.csv", index=False)

    write_markdown_report(output_dir / "STATISTICAL_REPORT.md", omnibus, descriptive, methods, args.alpha)

    metadata = {
        "input": str(input_path),
        "methods": methods,
        "metrics": metrics,
        "alpha": args.alpha,
        "confidence": args.confidence,
        "bootstrap_resamples": args.bootstrap_resamples,
        "minimum_complete_pairs": args.min_complete_pairs,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scipy_version": scipy.__version__,
    }
    (output_dir / "analysis_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # Consolidated Excel workbook for convenient review.
    try:
        with pd.ExcelWriter(output_dir / "statistical_results.xlsx", engine="openpyxl") as writer:
            descriptive.to_excel(writer, sheet_name="Descriptive", index=False)
            omnibus.to_excel(writer, sheet_name="Friedman", index=False)
            mcnemar.to_excel(writer, sheet_name="McNemar", index=False)
            if pairwise_tables:
                pd.concat(pairwise_tables, ignore_index=True).to_excel(writer, sheet_name="Wilcoxon_Holm", index=False)
            pd.DataFrame(rank_summary_rows).to_excel(writer, sheet_name="Average_Ranks", index=False)
    except Exception as exc:
        print(f"[warning] Excel workbook not written: {exc}", file=sys.stderr)

    print(f"[statistics] input:   {input_path}")
    print(f"[statistics] output:  {output_dir}")
    print(f"[statistics] methods: {len(methods)}")
    print(f"[statistics] metrics: {list(metrics)}")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Paired statistical comparison and CD plots for matching benchmarks")
    parser.add_argument("--input", required=True, help="Path to per_pair_results.csv")
    parser.add_argument("--output", default="statistical_analysis", help="Output directory")
    parser.add_argument("--methods", default="all", help="all or comma-separated method names")
    parser.add_argument(
        "--metrics",
        default=",".join(DEFAULT_METRICS),
        help="Comma-separated metrics to analyse",
    )
    parser.add_argument("--alpha", type=float, default=0.05, help="Family-wise significance level")
    parser.add_argument("--confidence", type=float, default=0.95, help="Bootstrap confidence level")
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--min-complete-pairs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        return run_analysis(args)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
