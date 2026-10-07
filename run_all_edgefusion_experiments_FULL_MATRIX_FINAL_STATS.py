#!/usr/bin/env python3
"""
Master EdgeFusion full-matrix paper experiment runner.

FINAL EXPERIMENT DESIGN
-----------------------
Runs every configured method on every configured TUM/Freiburg sequence at:

    Nf = 800, 1200, 1600

This single matrix subsumes:
1. Main benchmark
   -> freiburg1_xyz, Nf=1600
2. EdgeFusion component ablation
   -> EdgeFusion variants from freiburg1_xyz, Nf=1600
3. Feature-budget sensitivity
   -> compare Nf=800/1200/1600
4. Cross-sequence generalization
   -> aggregate all sequences separately for each Nf

There is deliberately NO separate RUN_MAIN because that experiment would be
duplicated inside the full matrix.

Expected number of benchmark runs:
    len(DATASETS) * len(FEATURE_BUDGETS)
    = 14 * 3
    = 42

Each benchmark run contains all methods in ALL_METHODS.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


# =============================================================================
# GLOBAL SWITCHES
# =============================================================================

RUN_FULL_MATRIX = True
RUN_REPORTS = True
RUN_CROSS_SEQUENCE_REPORTS = True

# Resume-friendly default.
# Existing completed benchmark/report outputs are reused unless --force is used.
SKIP_EXISTING = True

# For final paper experiments, failing fast is safer.
CONTINUE_ON_ERROR = False

# If False, cross-sequence aggregation requires every configured sequence.
ALLOW_PARTIAL_GENERALIZATION = False


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable

DATA_ROOT = (PROJECT_ROOT / ".." / ".." / "data").resolve()
RESULTS_ROOT = (PROJECT_ROOT / "results_edgefusion_all").resolve()


# =============================================================================
# EXPERIMENT MATRIX
# =============================================================================

FEATURE_BUDGETS = [800, 1200, 1600]

DATASETS = [
    "rgbd_dataset_freiburg1_360",
    "rgbd_dataset_freiburg1_desk",
    "rgbd_dataset_freiburg1_desk2",
    "rgbd_dataset_freiburg1_floor",
    "rgbd_dataset_freiburg1_room",
    "rgbd_dataset_freiburg1_xyz",
    "rgbd_dataset_freiburg2_360_hemisphere",
    "rgbd_dataset_freiburg2_coke",
    "rgbd_dataset_freiburg2_dishes",
    "rgbd_dataset_freiburg2_flowerbouquet",
    "rgbd_dataset_freiburg2_flowerbouquet_brownbackground",
    "rgbd_dataset_freiburg2_metallic_sphere2",
    "rgbd_dataset_freiburg3_nostructure_texture_near_withloop",
    "rgbd_dataset_freiburg3_structure_notexture_near",
]

ALL_METHOD_LIST = [
    "sift_nn",
    "orb_nn",
    "sift_lowe",
    "orb_lowe",
    "sift_mutual_lowe",
    "orb_mutual_lowe",
    "orb_crosscheck",
    "akaze_lowe",
    "adalam_sift",
    "adalam_orb",
    # "orbslam_adaptive_fh",
    "adalam_orb_bits",
    "xfeat",

    # Proposed EdgeFusion family / ablations
    "edgefusion_fast",
    "edgefusion_full",
    "edgefusion_lite",
    "edgefusion_no_edges",
    "edgefusion_no_grid",
    "edgefusion_no_lk",
    "edgefusion_adaptive_poseguard_v4",
]

ALL_METHODS = ",".join(ALL_METHOD_LIST)

# Keep this as No-Grid unless/until the paper's designated primary method changes.
PAPER_PRIMARY_METHOD = "edgefusion_no_grid"

# The canonical main-paper configuration is recovered from the matrix as:
REFERENCE_DATASET = "rgbd_dataset_freiburg1_xyz"
REFERENCE_NFEATURES = 1600


# =============================================================================
# FROZEN BENCHMARK PARAMETERS
# =============================================================================

COMMON_ARGS = [
    "--frame-step", "10",
    "--pair-stride", "20",
    "--max-pairs", "0",
    "--max-pose-dt", "0.02",
    "--max-side", "960",
    "--ratio", "0.80",
    "--magsac-threshold", "1.25",

    "--adaptive-accept-inliers", "80",
    "--adaptive-accept-inlier-ratio", "0.65",
    "--adaptive-accept-coverage", "0.25",
    "--adaptive-fallback-feature-fraction", "0.75",
    "--adaptive-lk-max-matches", "240",

    "--adaptive-poseguard-quality-margin", "0.02",
    "--adaptive-poseguard-prelk-margin", "0.035",
    "--adaptive-poseguard-parallax-target-deg", "1.0",
    "--adaptive-poseguard-rescue-fast-cheirality-max", "0.35",
    "--adaptive-poseguard-rescue-candidate-cheirality-min", "0.60",
    "--adaptive-poseguard-rescue-cheirality-gain", "0.25",
    "--adaptive-poseguard-cheirality-drop-tolerance", "0.05",

    "--adaptive-poseguard-v3-quality-margin", "0.02",
    "--adaptive-poseguard-v3-prelk-margin", "0.035",
    "--adaptive-poseguard-v3-min-candidate-native-cheirality", "0.50",
    "--adaptive-poseguard-v3-min-candidate-parallax-deg", "0.50",
    "--adaptive-poseguard-v3-rescue-candidate-native-cheirality-min", "0.80",
    "--adaptive-poseguard-v3-rescue-native-cheirality-gain", "0.25",
    "--adaptive-poseguard-v3-rescue-candidate-shared-cheirality-min", "0.60",
    "--adaptive-poseguard-v3-rescue-shared-cheirality-gain", "0.20",
    "--adaptive-poseguard-v3-shared-cheirality-drop-tolerance", "0.05",
    "--adaptive-poseguard-v3-native-cheirality-drop-tolerance", "0.10",
    "--adaptive-poseguard-v3-early-abort-native-cheirality-max", "0.15",
    "--adaptive-poseguard-v3-early-abort-parallax-max-deg", "0.25",

    "--adaptive-poseguard-v4-lk-quality-margin", "0.015",
    "--adaptive-poseguard-v4-lk-max-rotation-disagreement-deg", "3.0",
    "--adaptive-poseguard-v4-lk-max-translation-disagreement-deg", "15.0",
    "--adaptive-poseguard-v4-lk-parallax-ratio-min", "0.50",
    "--adaptive-poseguard-v4-lk-parallax-ratio-max", "2.00",
    "--adaptive-poseguard-v4-lk-shared-cheirality-drop-tolerance", "0.03",
    "--adaptive-poseguard-v4-lk-native-cheirality-drop-tolerance", "0.05",

    "--method-order", "rotate",
    "--cv-threads", "1",
    "--seed", "7",
    "--device", "cpu",
    "--visualize-pairs", "0",
    "--stats-bootstrap-resamples", "1000",
    "--measure-memory",
]


# =============================================================================
# SCRIPT DISCOVERY
# =============================================================================

def first_existing(candidates: Iterable[Path], description: str) -> Path:
    checked = []
    for candidate in candidates:
        candidate = candidate.resolve()
        checked.append(str(candidate))
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"{description} was not found.\nChecked:\n  " + "\n  ".join(checked)
    )


def resolve_scripts() -> tuple[Path, Path, Path]:
    benchmark_script = first_existing(
        [
            PROJECT_ROOT / "matching_comparison_all_poseguard_v4.py",
        ],
        "Benchmark script",
    )

    report_script = first_existing(
        [
            PROJECT_ROOT / "reproduce_paper_results_updated.py",
            PROJECT_ROOT / "reproduce_paper_results_updated(1).py",
        ],
        "Per-run paper/statistics report script",
    )

    final_analysis_script = first_existing(
        [
            PROJECT_ROOT / "generate_cross_sequence_all_metrics_FINAL_STATS.py",
            PROJECT_ROOT / "generate_cross_sequence_all_metrics_top3_legend_bottom(2).py",
            PROJECT_ROOT / "generate_cross_sequence_all_metrics_top3_legend_bottom.py",
        ],
        "Final cross-sequence analysis script",
    )

    return benchmark_script, report_script, final_analysis_script


# =============================================================================
# RUN SPECIFICATION
# =============================================================================

@dataclass(frozen=True)
class RunSpec:
    dataset_name: str
    nfeatures: int
    output_dir: Path

    @property
    def dataset_short(self) -> str:
        return self.dataset_name.replace("rgbd_dataset_", "")

    @property
    def workbook(self) -> Path:
        return self.output_dir / "all_algorithm_results.xlsx"

    @property
    def report_dir(self) -> Path:
        return self.output_dir / "paper_results"

    @property
    def report_complete_marker(self) -> Path:
        return self.report_dir / "analysis_config.json"


def build_full_matrix_specs() -> list[RunSpec]:
    specs: list[RunSpec] = []

    for nfeatures in FEATURE_BUDGETS:
        for dataset_name in DATASETS:
            short_name = dataset_name.replace("rgbd_dataset_", "")

            specs.append(
                RunSpec(
                    dataset_name=dataset_name,
                    nfeatures=nfeatures,
                    output_dir=(
                        RESULTS_ROOT
                        / "full_matrix"
                        / f"N{nfeatures}"
                        / short_name
                    ),
                )
            )

    return specs


# =============================================================================
# GENERAL HELPERS
# =============================================================================

def print_header(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def quote_command(cmd: list[str]) -> str:
    return " ".join(
        f'"{x}"' if " " in str(x) else str(x)
        for x in cmd
    )


def run_command(
    cmd: list[str],
    *,
    cwd: Optional[Path] = None,
    env_extra: Optional[dict[str, str]] = None,
) -> None:
    print("\nCOMMAND:")
    print(quote_command(cmd))
    if cwd is not None:
        print(f"CWD: {cwd}")
    print()

    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")

    if env_extra:
        env.update(env_extra)

    subprocess.run(
        cmd,
        check=True,
        cwd=str(cwd) if cwd is not None else None,
        env=env,
    )


def dataset_path(dataset_name: str) -> Path:
    path = (DATA_ROOT / dataset_name).resolve()

    if not path.is_dir():
        raise FileNotFoundError(
            f"Dataset folder not found: {path}"
        )

    return path


def validate_datasets() -> None:
    missing = [
        str((DATA_ROOT / dataset_name).resolve())
        for dataset_name in DATASETS
        if not (DATA_ROOT / dataset_name).is_dir()
    ]

    if missing:
        raise FileNotFoundError(
            "Required dataset folders are missing:\n  "
            + "\n  ".join(missing)
        )


def write_run_manifest(
    spec: RunSpec,
    benchmark_script: Path,
) -> None:
    spec.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": spec.dataset_name,
        "dataset_short": spec.dataset_short,
        "nfeatures": spec.nfeatures,
        "methods": ALL_METHOD_LIST,
        "num_methods": len(ALL_METHOD_LIST),
        "benchmark_script": str(benchmark_script),
        "dataset_path": str(dataset_path(spec.dataset_name)),
        "output_dir": str(spec.output_dir),
        "common_args": COMMON_ARGS,
        "is_reference_main_configuration": (
            spec.dataset_name == REFERENCE_DATASET
            and spec.nfeatures == REFERENCE_NFEATURES
        ),
    }

    (spec.output_dir / "runner_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


# =============================================================================
# PER-RUN BENCHMARK + REPORT
# =============================================================================

def run_report(
    spec: RunSpec,
    report_script: Path,
    *,
    force: bool,
) -> None:
    if not RUN_REPORTS:
        return

    if not spec.workbook.exists():
        raise FileNotFoundError(
            f"Benchmark workbook was not created: {spec.workbook}"
        )

    if (
        SKIP_EXISTING
        and not force
        and spec.report_complete_marker.exists()
    ):
        print(f"SKIP existing report: {spec.report_dir}")
        return

    spec.report_dir.mkdir(parents=True, exist_ok=True)

    # The reporting script writes an auxiliary relative all_ds directory.
    # Keep that side effect isolated within this experiment folder.
    (spec.output_dir / "all_ds").mkdir(
        parents=True,
        exist_ok=True,
    )

    cmd = [
        PYTHON,
        str(report_script),
        "--input", str(spec.workbook.resolve()),
        "--output", str(spec.report_dir.resolve()),
        "--bootstrap-resamples", "1000",
        "--alpha", "0.05",
        "--paper-primary-method", PAPER_PRIMARY_METHOD,
    ]

    run_command(
        cmd,
        cwd=spec.output_dir,
    )


def run_one_experiment(
    spec: RunSpec,
    benchmark_script: Path,
    report_script: Path,
    *,
    force: bool,
) -> None:
    print_header(
        f"Nf={spec.nfeatures} | {spec.dataset_short}"
    )

    spec.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_run_manifest(
        spec,
        benchmark_script,
    )

    if (
        SKIP_EXISTING
        and not force
        and spec.workbook.exists()
    ):
        print(
            f"SKIP existing benchmark: {spec.workbook}"
        )
    else:
        cmd = [
            PYTHON,
            str(benchmark_script),
            "--dataset", str(
                dataset_path(spec.dataset_name)
            ),
            "--output", str(spec.output_dir),
            "--methods", ALL_METHODS,
            "--nfeatures", str(spec.nfeatures),
        ] + COMMON_ARGS

        run_command(
            cmd,
            cwd=PROJECT_ROOT,
        )

    # If the benchmark exists but its report is missing,
    # the report is still generated.
    run_report(
        spec,
        report_script,
        force=force,
    )


# =============================================================================
# CROSS-SEQUENCE AGGREGATION
# =============================================================================

def find_numeric_summary(spec: RunSpec) -> Path:
    tables_dir = (
        spec.report_dir
        / "paper_to_export"
        / "tables"
    )

    candidates = sorted(
        tables_dir.glob(
            "TABLE_ALL_METHODS_MEAN_CI_MEDIAN_NUMERIC_*.csv"
        )
    )

    if len(candidates) == 1:
        return candidates[0]

    if len(candidates) == 0:
        raise FileNotFoundError(
            f"No numeric summary found in {tables_dir}"
        )

    # Prefer a filename matching the current sequence output folder.
    matching = [
        p for p in candidates
        if p.stem.endswith(spec.output_dir.stem)
    ]

    if len(matching) == 1:
        return matching[0]

    raise RuntimeError(
        f"Cannot uniquely determine numeric summary in "
        f"{tables_dir}: {[p.name for p in candidates]}"
    )


def rebuild_budget_aggregate(
    specs: list[RunSpec],
    nfeatures: int,
) -> Path:
    """
    Build one clean aggregate per feature budget.

    Output:
      results_edgefusion_all/
        cross_sequence/
          N800/TABLE_ALL_DATASETS.csv
          N1200/TABLE_ALL_DATASETS.csv
          N1600/TABLE_ALL_DATASETS.csv
    """

    budget_specs = [
        spec for spec in specs
        if spec.nfeatures == nfeatures
    ]

    frames: list[pd.DataFrame] = []
    missing: list[str] = []

    for spec in budget_specs:
        try:
            summary_path = find_numeric_summary(spec)
        except FileNotFoundError:
            missing.append(spec.dataset_short)
            continue

        frame = pd.read_csv(summary_path)

        frame["Dataset"] = spec.dataset_short
        frame["NFeatures"] = nfeatures

        frames.append(frame)

    if missing and not ALLOW_PARTIAL_GENERALIZATION:
        raise FileNotFoundError(
            f"Nf={nfeatures}: cross-sequence aggregation "
            "is incomplete. Missing:\n  "
            + "\n  ".join(missing)
        )

    if not frames:
        raise RuntimeError(
            f"Nf={nfeatures}: no report summaries were found."
        )

    combined = pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )

    if "method" not in combined.columns:
        raise ValueError(
            "Expected 'method' column is missing from "
            "the numeric report summaries."
        )

    combined = combined.drop_duplicates(
        subset=["Dataset", "NFeatures", "method"],
        keep="last",
    )

    combined = combined.sort_values(
        ["Dataset", "method"],
        kind="stable",
    ).reset_index(drop=True)

    aggregate_dir = (
        RESULTS_ROOT
        / "cross_sequence"
        / f"N{nfeatures}"
    )
    aggregate_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    aggregate_path = (
        aggregate_dir
        / "TABLE_ALL_DATASETS.csv"
    )

    combined.to_csv(
        aggregate_path,
        index=False,
    )

    print_header(
        f"CROSS-SEQUENCE AGGREGATE | Nf={nfeatures}"
    )
    print(
        f"Datasets: {combined['Dataset'].nunique()}"
    )
    print(
        f"Methods:  {combined['method'].nunique()}"
    )
    print(
        f"Rows:     {len(combined)}"
    )
    print(
        f"Output:   {aggregate_path}"
    )

    return aggregate_path


def run_final_sequence_analysis(
    final_analysis_script: Path,
    aggregate_path: Path,
    output_dir: Path,
) -> None:
    """
    Run the final sequence-level analysis directly on a clean aggregate.

    The analysis script produces:
      * mean / median / sequence-bootstrap CI
      * Friedman omnibus tests
      * Kendall's W
      * pairwise Wilcoxon signed-rank tests
      * Holm-adjusted p-values
      * paired rank-biserial r_rb
      * primary-vs-all table
      * Pareto plot and source CSV
      * LaTeX / CSV / Excel paper outputs

    If aggregate_path contains multiple NFeatures values, the analysis script
    automatically analyzes each budget separately and additionally performs
    within-method feature-budget sensitivity tests.
    """
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cmd = [
        PYTHON,
        str(final_analysis_script),
        "--input", str(aggregate_path.resolve()),
        "--output", str(output_dir.resolve()),
        "--bootstrap-resamples", "10000",
        "--confidence", "0.95",
        "--alpha", "0.05",
        "--seed", "7",
        "--primary-method", PAPER_PRIMARY_METHOD,
    ]

    run_command(
        cmd,
        cwd=PROJECT_ROOT,
    )


# =============================================================================
# OPTIONAL MASTER AGGREGATE ACROSS ALL BUDGETS
# =============================================================================

def build_all_budget_master_table(
    budget_paths: dict[int, Path],
) -> Path:
    frames = []

    for nfeatures, path in budget_paths.items():
        frame = pd.read_csv(path)

        if "NFeatures" not in frame.columns:
            frame["NFeatures"] = nfeatures

        frames.append(frame)

    master = pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )

    master = master.drop_duplicates(
        subset=["Dataset", "NFeatures", "method"],
        keep="last",
    )

    master = master.sort_values(
        ["NFeatures", "Dataset", "method"],
        kind="stable",
    ).reset_index(drop=True)

    output = (
        RESULTS_ROOT
        / "cross_sequence"
        / "TABLE_ALL_DATASETS_ALL_BUDGETS.csv"
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    master.to_csv(
        output,
        index=False,
    )

    print_header("MASTER ALL-BUDGET TABLE")
    print(f"Rows:    {len(master)}")
    print(
        f"Budgets: {sorted(master['NFeatures'].unique().tolist())}"
    )
    print(
        f"Datasets:{master['Dataset'].nunique()}"
    )
    print(
        f"Methods: {master['method'].nunique()}"
    )
    print(
        f"Output:  {output}"
    )

    return output


# =============================================================================
# CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run all EdgeFusion methods on all datasets "
            "for Nf=800,1200,1600."
        )
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Rerun benchmark and report outputs even "
            "when completed files already exist."
        ),
    )

    parser.add_argument(
        "--only-budget",
        type=int,
        choices=FEATURE_BUDGETS,
        default=None,
        help=(
            "Run only one feature budget "
            "(800, 1200, or 1600)."
        ),
    )

    parser.add_argument(
        "--only-dataset",
        default="",
        help=(
            "Run only one dataset. Accepts either the "
            "full folder name or the short name, e.g. "
            "'rgbd_dataset_freiburg1_xyz' or 'freiburg1_xyz'."
        ),
    )

    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help=(
            "Do not run benchmarks. Rebuild cross-sequence "
            "aggregates/reports from existing per-run reports."
        ),
    )

    return parser


def normalize_dataset_selector(
    value: str,
) -> Optional[str]:
    value = value.strip()

    if not value:
        return None

    if value in DATASETS:
        return value

    full = (
        value
        if value.startswith("rgbd_dataset_")
        else f"rgbd_dataset_{value}"
    )

    if full not in DATASETS:
        raise ValueError(
            f"Dataset '{value}' is not in DATASETS."
        )

    return full


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    args = build_parser().parse_args()

    benchmark_script, report_script, final_analysis_script = (
        resolve_scripts()
    )

    RESULTS_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    selected_dataset = normalize_dataset_selector(
        args.only_dataset
    )

    all_specs = build_full_matrix_specs()

    selected_specs = [
        spec
        for spec in all_specs
        if (
            args.only_budget is None
            or spec.nfeatures == args.only_budget
        )
        and (
            selected_dataset is None
            or spec.dataset_name == selected_dataset
        )
    ]

    print_header(
        "EDGEFUSION FULL-MATRIX PAPER EXPERIMENT RUNNER"
    )

    print(f"Python:              {PYTHON}")
    print(f"Project root:        {PROJECT_ROOT}")
    print(f"Data root:           {DATA_ROOT}")
    print(f"Results root:        {RESULTS_ROOT}")
    print(f"Methods/run:         {len(ALL_METHOD_LIST)}")
    print(f"Datasets:            {len(DATASETS)}")
    print(f"Feature budgets:     {FEATURE_BUDGETS}")
    print(f"Full matrix runs:    {len(all_specs)}")
    print(f"Selected runs:       {len(selected_specs)}")
    print(
        f"Reference main cell: "
        f"{REFERENCE_DATASET}, Nf={REFERENCE_NFEATURES}"
    )

    validate_datasets()

    failures: list[
        tuple[RunSpec, Exception]
    ] = []

    if RUN_FULL_MATRIX and not args.aggregate_only:
        for index, spec in enumerate(
            selected_specs,
            start=1,
        ):
            print_header(
                f"RUN {index}/{len(selected_specs)} | "
                f"Nf={spec.nfeatures} | "
                f"{spec.dataset_short}"
            )

            try:
                run_one_experiment(
                    spec,
                    benchmark_script,
                    report_script,
                    force=args.force,
                )
            except Exception as exc:
                failures.append(
                    (spec, exc)
                )

                print(
                    f"ERROR: Nf={spec.nfeatures} "
                    f"{spec.dataset_short}: {exc}",
                    file=sys.stderr,
                )

                if not CONTINUE_ON_ERROR:
                    raise

    # ---------------------------------------------------------
    # Cross-sequence reports
    # ---------------------------------------------------------
    # Only aggregate budgets for which the user requested all datasets.
    # A single-dataset filtered run should not pretend to be a complete
    # cross-sequence analysis.
    if (
        RUN_CROSS_SEQUENCE_REPORTS
        and selected_dataset is None
    ):
        budgets_to_aggregate = (
            [args.only_budget]
            if args.only_budget is not None
            else FEATURE_BUDGETS
        )

        budget_paths: dict[int, Path] = {}

        for nfeatures in budgets_to_aggregate:
            aggregate_path = rebuild_budget_aggregate(
                all_specs,
                nfeatures,
            )

            budget_paths[nfeatures] = (
                aggregate_path
            )


        # -------------------------------------------------------------
        # FINAL PAPER ANALYSIS
        # -------------------------------------------------------------
        # If all configured budgets are available, build one master file and
        # analyze it once. The final analysis script separates budgets
        # internally and also performs cross-budget sensitivity inference.
        if set(budget_paths) == set(FEATURE_BUDGETS):
            master_path = build_all_budget_master_table(
                budget_paths
            )

            run_final_sequence_analysis(
                final_analysis_script,
                master_path,
                RESULTS_ROOT
                / "PAPER_RESULTS"
                / "FINAL_SEQUENCE_LEVEL_ANALYSIS",
            )

        # If the user requested only one budget, still run the full
        # sequence-level descriptive + inferential analysis for that budget.
        elif len(budget_paths) == 1:
            only_budget = next(iter(budget_paths))
            run_final_sequence_analysis(
                final_analysis_script,
                budget_paths[only_budget],
                RESULTS_ROOT
                / "PAPER_RESULTS"
                / "FINAL_SEQUENCE_LEVEL_ANALYSIS"
                / f"N{only_budget}",
            )

    if failures:
        print_header("COMPLETED WITH FAILURES")

        for spec, exc in failures:
            print(
                f"- Nf={spec.nfeatures} "
                f"{spec.dataset_short}: {exc}"
            )

        return 1

    print_header(
        "ALL REQUESTED EXPERIMENTS COMPLETED"
    )

    print(
        "Full matrix root:\n"
        f"  {RESULTS_ROOT / 'full_matrix'}"
    )

    print(
        "\nFinal sequence-level statistical analysis:\n"
        f"  {RESULTS_ROOT / 'PAPER_RESULTS' / 'FINAL_SEQUENCE_LEVEL_ANALYSIS'}"
    )

    print(
        "\nMaster all-budget table:\n"
        f"  {RESULTS_ROOT / 'cross_sequence' / 'TABLE_ALL_DATASETS_ALL_BUDGETS.csv'}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
