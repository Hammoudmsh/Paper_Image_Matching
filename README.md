# Image Matching Algorithms and EdgeFusion Experiments

Benchmark classical image matchers, AdaLAM, optional learned matchers, and EdgeFusion variants on TUM RGB-D image pairs. The workflow measures matching, relative pose accuracy, runtime, and memory, then generates statistical tables and figures. This is an image-pair benchmark; it does not run a complete SLAM trajectory evaluation.

This guide follows the current `matching_comparison_all_poseguard_v4.py` implementation and the full-matrix experiment runner. Start by creating the environment below. Run all subsequent commands from the repository root with that environment activated, except where a report example explicitly changes directories. Single-line commands work in PowerShell, Windows Command Prompt, and Bash unless a shell is explicitly named.

**Repository contents:** The current `.gitignore` excludes `reproduce_paper_results_updated.py`, notebooks, and batch files. The full-matrix runner requires that report script even for script discovery; obtain your local copy and place it in the repository root before running the matrix or per-run paper report commands below. Direct image-matching benchmarks and cross-sequence analysis from existing numeric summaries can be used independently.

## Workflow

1. [Create the environment](#1-create-the-environment), including the matching-model dependencies.
2. [Download the data](#2-download-the-data) and align the runner's dataset path.
3. [Run all experiments](#3-run-all-experiments) with `run_all_edgefusion_experiments_FULL_MATRIX_FINAL_STATS.py`.
4. [Run specific experiments](#4-run-specific-experiments): one dataset, one budget, a single matrix cell, or selected methods.
5. [Generate cross-sequence reports](#5-generate-cross-sequence-reports) with `generate_cross_sequence_all_metrics_FINAL_STATS.py`.
6. [Generate individual experiment reports](#6-generate-reports-for-an-individual-experiment).
7. [Interpret and preserve results](#7-interpret-and-preserve-results).
8. [Command reference and validation](#8-command-reference-and-validation).

## 1. Create the environment

Install Git and either Conda or Python 3.10. A CPU environment is sufficient and matches the matrix runner's default device. Dataset archives and extracted images require substantial disk space; check the sizes of your selected sequences before downloading all of them.

```text
git clone https://github.com/Hammoudmsh/Paper_Image_Matching.git
cd Paper_Image_Matching
```

If you already have a checkout, open a terminal in its root instead.

### Option A: Conda CPU environment

```text
conda env create -f step0_init_env/environment.yml
conda activate edgefusion-benchmark
```

### Option B: Python virtual environment

Create the environment using Python 3.10:

```text
python -m venv venv
```

Activate it with the command for your shell:

| Shell | Command |
| --- | --- |
| PowerShell | `.\venv\Scripts\Activate.ps1` |
| Windows Command Prompt | `venv\Scripts\activate.bat` |
| Bash on Linux/macOS | `source venv/bin/activate` |

Install the pinned CPU stack on Windows/Linux:

```text
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r step0_init_env/requirements.txt
```

For macOS, use the platform-specific installation command from the [official PyTorch 2.5.1 instructions](https://pytorch.org/get-started/previous-versions/), then install the requirements file. The repository's pinned versions are the reproducibility target, rather than a request to install the newest packages.

### Optional NVIDIA GPU environment

```text
conda env create -f step0_init_env/environment-gpu.yml
conda activate edgefusion-benchmark-gpu
python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

The GPU environment specifies PyTorch 2.5.1 and CUDA 12.1. Direct benchmark commands accept `--device cuda`; the matrix runner still uses `--device cpu` unless its `COMMON_ARGS` are changed. Keep CPU and GPU experiments in separate result directories.

### Check the active environment

```text
python -c "import sys, cv2, numpy, scipy, pandas, torch, openpyxl; print(sys.executable); print('OpenCV:', cv2.__version__, 'NumPy:', numpy.__version__, 'PyTorch:', torch.__version__); print('MAGSAC:', hasattr(cv2, 'USAC_MAGSAC'))"
python -m pip check
python matching_comparison_all_poseguard_v4.py --help
```

Use `opencv-contrib-python` from the requirements file. Avoid installing several different OpenCV wheel variants into the same environment.

### Prepare matching models for the experiments

AdaLAM source is bundled under `third_party/adalam`; its provenance is documented in `third_party/README_ADALAM_SOURCE.md`. PyTorch is needed for AdaLAM. XFeat and LightGlue are optional for direct runs, but **the current full matrix includes XFeat**.

XFeat and LightGlue are Git submodules. After cloning this repository, fetch their recorded revisions with `git submodule update --init --recursive`. Check for `third_party/accelerated_features/modules/xfeat.py` and `third_party/LightGlue/lightglue/__init__.py` before proceeding.

If the repositories are missing, clone them using the following commands. If Git reports that a destination already exists, inspect it first: an existing populated checkout can be used directly; an empty placeholder can be removed or renamed before cloning.

```text
git clone https://github.com/verlab/accelerated_features.git third_party/accelerated_features
git clone https://github.com/cvg/LightGlue.git third_party/LightGlue
```

To match the revisions recorded in this repository's current Git index, use:

```text
git -C third_party/accelerated_features checkout e92685f57f8318b18725c5c8c0bd28c7fe188d9a
git -C third_party/LightGlue checkout eb42fee2d71449efb0aa5c10549752b5d75384d8
python -m pip install -e third_party/LightGlue --no-deps
```

Check model imports and XFeat's local weights:

```text
python -c "import sys; from pathlib import Path; sys.path.insert(0, 'third_party/accelerated_features'); from modules.xfeat import XFeat; assert Path('third_party/accelerated_features/weights/xfeat.pt').is_file(); print('XFeat source and weights found')"
python -c "from lightglue import LightGlue, SIFT; print('LightGlue import OK')"
```

These import checks do not perform matching. LightGlue may download pretrained weights on its first actual run. Complete a small learned-matcher run while internet access is available before starting a long experiment.

For direct runs, pass `--xfeat-repo third_party/accelerated_features`. For the matrix runner, which does not supply that argument, set `PYTHONPATH` in the terminal used to launch it:

PowerShell:

```powershell
$env:PYTHONPATH = (Resolve-Path "third_party/accelerated_features").Path + [IO.Path]::PathSeparator + $env:PYTHONPATH
```

Windows Command Prompt:

```bat
set "PYTHONPATH=%CD%\third_party\accelerated_features;%PYTHONPATH%"
```

Bash:

```bash
export PYTHONPATH="$PWD/third_party/accelerated_features${PYTHONPATH:+:$PYTHONPATH}"
```

Repeat this in a new terminal session. An alternative is to add `"--xfeat-repo", str(PROJECT_ROOT / "third_party" / "accelerated_features")` to the runner's `COMMON_ARGS` list and record that configuration change.

## 2. Download the data

The source is the [official TUM RGB-D dataset download page](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download). Keep the original sequence directory names and timestamps. Consult the dataset page for attribution and usage information.

### Download one sequence first

The downloader uses only the Python standard library:

```text
python download_tum.py --list
python download_tum.py --output data --dataset rgbd_dataset_freiburg1_xyz --debug
python download_tum.py --output data --dataset rgbd_dataset_freiburg1_xyz
```

`--debug` checks URLs without downloading archives. `--keep-archive` retains the `.tgz` after extraction; otherwise the archive is removed after successful extraction.

The expected layout is:

```text
data/
  rgbd_dataset_freiburg1_xyz/
    rgb.txt
    groundtruth.txt
    rgb/
      <timestamp>.png
    depth.txt
    depth/
      <timestamp>.png
```

The matching benchmark uses RGB images, RGB timestamps, and ground-truth poses. The downloaded archives also contain depth data. Do not point `--dataset` at the parent `data` folder or directly at `rgb`.

Check the files needed for the first experiment:

```text
python -c "from pathlib import Path; p=Path('data/rgbd_dataset_freiburg1_xyz'); assert (p/'rgb.txt').is_file(), 'Missing rgb.txt'; assert (p/'groundtruth.txt').is_file(), 'Missing groundtruth.txt'; assert any((p/'rgb').glob('*.png')), 'Missing RGB images'; print('Dataset files found:', p.resolve())"
```

### Download all 14 configured sequences

```text
python download_tum.py --output data
```

| Family | Sequence suffixes after `rgbd_dataset_` |
| --- | --- |
| Freiburg 1 | `freiburg1_360`, `freiburg1_desk`, `freiburg1_desk2`, `freiburg1_floor`, `freiburg1_room`, `freiburg1_xyz` |
| Freiburg 2 | `freiburg2_360_hemisphere`, `freiburg2_coke`, `freiburg2_dishes`, `freiburg2_flowerbouquet`, `freiburg2_flowerbouquet_brownbackground`, `freiburg2_metallic_sphere2` |
| Freiburg 3 | `freiburg3_nostructure_texture_near_withloop`, `freiburg3_structure_notexture_near` |

Existing sequence folders are automatically skipped. The downloader checks folder existence, not completeness: if extraction was interrupted, move the incomplete sequence folder aside before downloading that sequence again. The current `--timeout` option is parsed but not passed through to the download/check functions, so increasing it does not change their timeouts. Also, `--skip-existing` can produce exit code 1 when folders were skipped successfully; inspect the summary and files.

If a URL fails, use the official download page to obtain the archive manually and extract its sequence folder into `data/`.

### Align the matrix runner's dataset path

**Before using the matrix runner**, edit this constant in `run_all_edgefusion_experiments_FULL_MATRIX_FINAL_STATS.py`:

```python
# Original default: two directories above the repository root
DATA_ROOT = (PROJECT_ROOT / ".." / ".." / "data").resolve()

# Use this for the repository-local downloads shown in this guide
DATA_ROOT = (PROJECT_ROOT / "data").resolve()
```

Replace the original assignment with the second assignment. There is no `--data-root` option. Alternatively, keep the original constant and download with `--output ../../data` instead; adjust the direct benchmark paths accordingly.

## 3. Run all experiments

First download all sequences, align `DATA_ROOT`, prepare XFeat, and set its `PYTHONPATH` as described above.

The runner currently defines **14 sequences x 3 feature budgets = 42 benchmark runs**, each containing **19 methods**:

```text
sift_nn, orb_nn, sift_lowe, orb_lowe, sift_mutual_lowe,
orb_mutual_lowe, orb_crosscheck, akaze_lowe,
adalam_sift, adalam_orb, adalam_orb_bits, xfeat,
edgefusion_fast, edgefusion_full, edgefusion_lite,
edgefusion_no_edges, edgefusion_no_grid, edgefusion_no_lk,
edgefusion_adaptive_poseguard_v4
```

Budgets are 800, 1200, and 1600. The reference configuration is Freiburg 1 XYZ at 1600 features. The paper primary method is currently `edgefusion_no_grid`. LightGlue is not in this matrix.

The runner fixes frame step 10, pair stride 20, all eligible pairs, CPU execution, one OpenCV thread, seed 7, memory measurement, and 1,000 per-run bootstrap resamples. It disables match visualizations. Inspect `COMMON_ARGS` for the full set of frozen thresholds.

Run all experiments with the main runner:

```text
python run_all_edgefusion_experiments_FULL_MATRIX_FINAL_STATS.py
```

This command performs the complete pipeline in order:

1. Validate the configured dataset directories.
2. Benchmark every configured method on every sequence at 800, 1200, and 1600 features.
3. Create each run's workbook, basic statistics, and paper report.
4. Build one numeric cross-sequence aggregate for each feature budget.
5. Build the combined aggregate and invoke `generate_cross_sequence_all_metrics_FINAL_STATS.py` for the final reports.

Runtime depends on the computer, methods, and eligible pair counts. Complete a small trial from the next section before starting the entire matrix.

**The runner validates all 14 dataset directories even for a filtered run or `--aggregate-only`.** Download all datasets and align `DATA_ROOT` first. For a trial with only one sequence downloaded, use the direct benchmark in Section 4.

### Resume and rerun behavior

Rerun the same command after an interruption. The runner reuses a benchmark when `all_algorithm_results.xlsx` exists and a report when `paper_results/analysis_config.json` exists. It does not verify that these files match newly edited parameters. A partially written `per_pair_results.csv` is useful for inspection, but the next benchmark invocation starts that run again rather than continuing from its last pair.

To recompute a selected completed run:

```text
python run_all_edgefusion_experiments_FULL_MATRIX_FINAL_STATS.py --only-dataset freiburg1_xyz --only-budget 1600 --force
```

`--force` reruns benchmarks and reports in their existing directories. Preserve previous results or change `RESULTS_ROOT` before starting a new experimental configuration. Rebuild aggregates after selectively recomputing runs.

### Matrix output layout

```text
results_edgefusion_all/
  full_matrix/
    N800/<sequence>/
    N1200/<sequence>/
    N1600/freiburg1_xyz/
      runner_manifest.json
      per_pair_results.csv
      all_algorithm_results.xlsx
      paper_results/
  cross_sequence/
    N800/TABLE_ALL_DATASETS.csv
    N1200/TABLE_ALL_DATASETS.csv
    N1600/TABLE_ALL_DATASETS.csv
    TABLE_ALL_DATASETS_ALL_BUDGETS.csv
  PAPER_RESULTS/
    FINAL_SEQUENCE_LEVEL_ANALYSIS/
```

## 4. Run specific experiments

### Select a dataset, a feature budget, or both

These commands keep the matrix's configured methods and frozen parameters:

| Experiment | Selection | Number of benchmark runs |
| --- | --- | --- |
| One sequence at all budgets | `--only-dataset freiburg1_xyz` | 3 |
| All sequences at one budget | `--only-budget 1600` | 14 |
| One sequence at one budget | Both options | 1 |

Run Freiburg 1 XYZ at all three budgets:

```text
python run_all_edgefusion_experiments_FULL_MATRIX_FINAL_STATS.py --only-dataset freiburg1_xyz
```

Run every sequence at 1600 features:

```text
python run_all_edgefusion_experiments_FULL_MATRIX_FINAL_STATS.py --only-budget 1600
```

Run only the reference experiment, Freiburg 1 XYZ at 1600 features:

```text
python run_all_edgefusion_experiments_FULL_MATRIX_FINAL_STATS.py --only-dataset freiburg1_xyz --only-budget 1600
```

Select a different sequence and budget:

```text
python run_all_edgefusion_experiments_FULL_MATRIX_FINAL_STATS.py --only-dataset rgbd_dataset_freiburg2_coke --only-budget 800
```

`--only-dataset` accepts the short name or full directory name. Budget choices are 800, 1200, and 1600. A single-dataset selection generates per-run reports but skips cross-sequence aggregation. A budget-only selection also produces cross-sequence analysis for that budget, because it includes all configured datasets.

The matrix runner has no `--methods`, `--max-pairs`, `--dataset`, or `--output` options. For a custom subset of methods or pairs, use the direct benchmark below. To change the entire matrix design, edit `DATASETS`, `FEATURE_BUDGETS`, `ALL_METHOD_LIST`, `COMMON_ARGS`, and/or `RESULTS_ROOT` in the runner and record those changes. Use a new results root for a changed design.

### Start with a small trial

Start with five image pairs and three methods:

```text
python matching_comparison_all_poseguard_v4.py --dataset data/rgbd_dataset_freiburg1_xyz --output results_edgefusion_all/smoke_xyz --methods orb_lowe,edgefusion_full,edgefusion_adaptive_poseguard_v4 --nfeatures 800 --frame-step 10 --pair-stride 20 --max-pairs 5 --device cpu --visualize-pairs 2 --skip-statistics
```

Inspect `summary.md`, `summary.csv`, and the `matches/` images in the output directory. Inspect method `status` and `success` in `per_pair_results.csv`, too: a completed process can contain failed or skipped methods. Five pairs are a functional check, not the final statistical experiment.

Then verify the learned matcher used in the matrix:

```text
python matching_comparison_all_poseguard_v4.py --dataset data/rgbd_dataset_freiburg1_xyz --output results_edgefusion_all/smoke_xfeat --methods orb_lowe,xfeat --xfeat-repo third_party/accelerated_features --max-pairs 2 --device cpu --skip-statistics
```

### Compare two individual images

Replace the example image paths with your own existing images:

```text
python matching_comparison_all_poseguard_v4.py --image1 path/to/image1.png --image2 path/to/image2.png --pair-output results_edgefusion_all/two_images --methods orb_lowe,adalam_orb,edgefusion_full,edgefusion_adaptive_poseguard_v4 --include-ablation --pair-camera auto --nfeatures 1600 --measure-memory
```

Outputs include `two_image_comparison.csv`, `two_image_comparison.xlsx`, `all_methods_geometric_inliers.jpg`, and `pair_config.json`. For known TUM images, select `--pair-camera tum_fr1`, `tum_fr2`, or `tum_fr3`. For a calibrated custom camera, use `--pair-camera custom --fx ... --fy ... --cx ... --cy ...` with numeric values and optionally `--distortion k1,k2,p1,p2,k3`.

Two-image mode estimates geometry without TUM ground-truth association; it does not establish ground-truth pose accuracy. Use dataset mode for pose-error experiments.

### Run custom methods and ablations on one sequence

This example runs core methods and component ablations on up to 50 pairs, with automatic statistical analysis:

```text
python matching_comparison_all_poseguard_v4.py --dataset data/rgbd_dataset_freiburg1_xyz --output results_edgefusion_all/xyz_core_N1600 --methods core --include-ablation --nfeatures 1600 --frame-step 10 --pair-stride 20 --max-pairs 50 --max-pose-dt 0.02 --max-side 960 --device cpu --method-order rotate --cv-threads 1 --seed 7 --measure-memory --visualize-pairs 3 --stats-bootstrap-resamples 1000
```

Use `--max-pairs 0` for every eligible pair. This command's method selection differs from the frozen matrix; use the matrix runner for its exact configured experiment.

### Method selection

| Selector | Meaning |
| --- | --- |
| `original` | Classical methods, AdaLAM SIFT/ORB, and `orbslam_adaptive_fh` |
| `core` / `default` | Original methods, `adalam_orb_bits`, and proposed EdgeFusion variants |
| `proposed` | EdgeFusion Fast, Full, Simple, Adaptive, Guarded, and PoseGuard variants including v4 |
| `all` | Core methods plus `xfeat` and `lightglue_sift` |
| Comma-separated IDs | Only the requested methods, e.g. `orb_lowe,edgefusion_full,edgefusion_no_grid` |
| `--include-ablation` | Adds `edgefusion_lite`, `edgefusion_no_edges`, `edgefusion_no_grid`, and `edgefusion_no_lk` |

`all` does not automatically include the four component ablations; add `--include-ablation`. Learned methods can be recorded as skipped if their dependencies are unavailable.

### Parameters to control

| Parameter | Interpretation |
| --- | --- |
| `--frame-step 10` | Separation between the two frames in a pair |
| `--pair-stride 20` | Step between candidate pair starting frames |
| `--max-pairs 50` | Maximum number of pairs; zero or negative means all |
| `--max-pose-dt 0.02` | Maximum RGB-to-ground-truth timestamp difference, in seconds |
| `--nfeatures 1600` | Feature budget per image |
| `--max-side 960` | Maximum image side after downscaling; zero disables downscaling |
| `--tum-camera auto` | Infer Freiburg calibration from the sequence path |
| `--ratio 0.80` | Base EdgeFusion matching ratio |
| `--magsac-threshold 1.25` | MAGSAC pixel threshold |
| `--method-order rotate` | Rotate execution positions to reduce order bias |
| `--cv-threads 1` | OpenCV thread count |
| `--seed 7` | Random seed |
| `--measure-memory` | Sample process RSS; adds timing overhead |
| `--visualize-pairs 3` | Save match illustrations for the first three pairs |

For an ablation, hold images, feature budget, device, and all other parameters fixed while changing the component variant. For feature-budget sensitivity, repeat the same experiment at 800, 1200, and 1600 features, using a different output directory for each. For cross-sequence evaluation, repeat the same settings across sequences and keep results separate until aggregation.

To retain raw correspondences, add `--save-raw-correspondences --raw-correspondence-pairs 3`; use `-1` for all pairs. This saves NPZ files with points, inlier masks, poses, timings, and diagnostics, and can require significant storage.

## 5. Generate cross-sequence reports

Use `generate_cross_sequence_all_metrics_FINAL_STATS.py` to compare methods across sequences and analyze feature-budget effects. It reads existing numeric summaries; it does not run image matching or create the initial benchmark results. A complete matrix run invokes it automatically. The commands below let you regenerate the reports independently.

### Step A: Build or refresh the numeric input tables

After per-run reports exist for the entire matrix:

```text
python run_all_edgefusion_experiments_FULL_MATRIX_FINAL_STATS.py --aggregate-only
```

This reads each run's `paper_results/paper_to_export/tables/TABLE_ALL_METHODS_MEAN_CI_MEDIAN_NUMERIC_*.csv`, rebuilds the aggregate CSV files, and invokes final analysis. It does not create missing per-run paper reports. If those are missing, rerun the matrix runner without `--aggregate-only`; existing completed benchmarks are reused.

To aggregate only the 1600-feature results:

```text
python run_all_edgefusion_experiments_FULL_MATRIX_FINAL_STATS.py --aggregate-only --only-budget 1600
```

Do not combine `--aggregate-only` with `--only-dataset`: a single-dataset selection skips cross-sequence aggregation. All configured sequences are required for each aggregated budget by default.

### Step B: Generate reports for all feature budgets

To regenerate final analysis directly from the combined numeric aggregate, without the runner's dataset-directory validation:

```text
python generate_cross_sequence_all_metrics_FINAL_STATS.py --input results_edgefusion_all/cross_sequence/TABLE_ALL_DATASETS_ALL_BUDGETS.csv --output results_edgefusion_all/PAPER_RESULTS/FINAL_SEQUENCE_LEVEL_ANALYSIS --bootstrap-resamples 2000 --confidence 0.95 --alpha 0.05 --seed 7 --primary-method edgefusion_no_grid
```

### Step C: Generate reports for one feature budget

For just one budget:

```text
python generate_cross_sequence_all_metrics_FINAL_STATS.py --input results_edgefusion_all/cross_sequence/N1600/TABLE_ALL_DATASETS.csv --output results_edgefusion_all/PAPER_RESULTS/N1600_manual --bootstrap-resamples 2000 --seed 7
```

These scripts consume **numeric sequence summaries**, not concatenated raw pair rows or formatted `mean [CI]` strings. Final outputs include descriptive tables, sequence-bootstrap confidence intervals, Friedman/Wilcoxon-Holm tests, effect sizes, figures, and `cross_sequence_statistical_results.xlsx` within the relevant analysis directories. Multi-budget input is analyzed separately by budget and also supports feature-budget sensitivity analysis.

### Analysis options

Always supply `--input` and `--output` explicitly. The script's default input is the older `results_edgefusion_all/all_ds/TABLE_ALL_DATASETS.csv` location, while the current runner writes aggregates under `cross_sequence/`.

| Option | Purpose | Default in the analysis script |
| --- | --- | --- |
| `--input` | Numeric sequence-summary CSV | Legacy path noted above |
| `--output` | Report destination directory | Use an explicit path as shown above |
| `--bootstrap-resamples` | Number of sequence-level bootstrap samples | 10000 |
| `--confidence` | Confidence interval level | 0.95 |
| `--alpha` | Significance level for inference | 0.05 |
| `--seed` | Seed for reproducible resampling | 7 |
| `--primary-method` | Method used in the primary-versus-all table | `edgefusion_no_grid` |
| `--dataset-prefix` | Prefix used when normalizing dataset names | Usually leave unchanged for runner-produced input |

The examples explicitly use 2,000 bootstrap resamples. Use `--bootstrap-resamples 10000` for a larger resampling run, and record the value with the results. Choose a primary method present in the input; changing it affects the primary-versus-all table, not which image-matching methods were originally evaluated.

### Input structure and output locations

The input requires `Dataset` and either `method` or `method_label`. Numeric metric columns include `success_mean_pct`, `total_ms_median`, `peak_rss_delta_mb_median`, `rotation_error_deg_median`, `translation_error_sign_invariant_deg_median`, `inlier_ratio_median`, and `geometric_inliers_median`. Combined input also includes `NFeatures` so budgets remain separate.

For the combined-input command above, open:

```text
results_edgefusion_all/PAPER_RESULTS/FINAL_SEQUENCE_LEVEL_ANALYSIS/
  N800/
    cross_sequence_statistical_results.xlsx
    PAPER_CROSS_SEQUENCE_TABLES.tex
    friedman_sequence_level.tex
    primary_vs_all_wilcoxon_holm_rrb.tex
    pareto_cross_sequence_runtime_translation.pdf
  N1200/
    ...
  N1600/
    ...
  budget_effects/
  analysis_metadata_all_budgets.json
```

Start with each budget's Excel workbook to review the tables, the Pareto PDF to compare runtime and translation error, and `PAPER_CROSS_SEQUENCE_TABLES.tex` for manuscript tables. `budget_effects/` contains within-method feature-budget inference. With a single-budget input, the report files go directly into the requested output directory, without an additional `N1600` subdirectory.

If the input CSV is missing, complete Step A first. If numeric summaries are missing, generate the per-run reports in Section 6 or rerun the matrix runner. Keep the full result folders when sharing or archiving an experiment so the report inputs can be traced back to the original pairs.

## 6. Generate reports for an individual experiment

### Basic paired statistical report

Dataset mode runs `statistical_analysis.py` automatically unless `--skip-statistics` is set. To rerun it independently:

```text
python statistical_analysis.py --input results_edgefusion_all/xyz_core_N1600/per_pair_results.csv --output results_edgefusion_all/xyz_core_N1600/statistics --bootstrap-resamples 1000 --min-complete-pairs 10 --alpha 0.05 --seed 7
```

The analysis includes descriptive statistics, bootstrap intervals, Friedman tests, paired Wilcoxon tests with Holm correction, success comparisons, and rank/CD figures where supported by the data. Figures are exported in PNG, PDF, and SVG formats.

### Paper tables and figures for one run

The paper exporter writes an auxiliary `all_ds/` directory relative to its working directory. Create it before a manual invocation. The following commands isolate that directory within this experiment:

```text
cd results_edgefusion_all/xyz_core_N1600
python -c "from pathlib import Path; Path('all_ds').mkdir(exist_ok=True)"
python ../../reproduce_paper_results_updated.py --input all_algorithm_results.xlsx --output paper_results --bootstrap-resamples 1000 --alpha 0.05 --seed 7 --paper-primary-method edgefusion_no_grid
cd ../..
```

The exporter also accepts `per_pair_results.csv` as input. Select a primary method that was actually included in the benchmark. Typical outputs under `paper_results/` are:

| Output | Purpose |
| --- | --- |
| `STATISTICAL_REPORT.md` | Readable statistical report |
| `statistical_results.xlsx` | Combined tables in Excel format |
| `tables/` | Descriptive statistics, ranks, hypothesis tests, and paper tables |
| Plot files directly in `paper_results/` | Runtime/accuracy plots and other report figures |
| `figure_data/` | Source data behind figures |
| `TABLE_PAPER.tex` | Generated LaTeX table content |
| `paper_to_export/tables/` | Publication tables in CSV and LaTeX, including sequence-specific filenames |
| `paper_to_export/paper_tables.xlsx` | Compact paper table workbook |
| `paper_to_export/qualitative_matches/` | Copied match illustrations, when available |
| `analysis_config.json` | Analysis settings and metadata |

The frozen matrix uses `--visualize-pairs 0`, so its runs do not create qualitative match images. Run a separate direct benchmark with visualizations enabled if those figures are needed. PDF plots are generated directly; LaTeX table files can be included in a manuscript separately.

## 7. Interpret and preserve results

| Metric | Interpretation |
| --- | --- |
| `success`, `status` | Whether a method produced a usable result; inspect failures and skipped backends |
| `total_ms` | Measured runtime; lower is faster |
| `rotation_error_deg` | Relative rotation error against ground truth; lower is better |
| `translation_error_sign_invariant_deg` | Translation-direction error allowing sign ambiguity; lower is better |
| `inlier_ratio` | Fraction retained as geometric inliers; consider alongside accuracy and success |
| `geometric_inliers` | Number of geometric inliers; more alone does not prove better pose accuracy |
| `peak_rss_delta_mb` | Sampled change in process RSS; not GPU VRAM |

Relative translation direction is not metric translation distance. Per-run paired tests use image pairs; cross-sequence inference uses sequences as the statistical units. Do not treat all pairs across all sequences as independent sequence-level evidence. Sparse complete cases or failed methods can make some tests unavailable; the default minimum is 10 complete pairs for per-run tests. Paper reports exclude zero-success/skipped methods by default; `--legacy-paper-mode` restores the older behavior.

Preserve these artifacts with any reported result:

- `per_pair_results.csv`, `summary.csv`, and `all_algorithm_results.xlsx`.
- `pair_manifest.csv`, recording the actual evaluated image pairs.
- `config.json`, `runtime_environment.json`, and `installed_packages.txt`.
- `runner_manifest.json` for matrix runs, and each report's `analysis_config.json`.
- The repository commit, any local code/configuration changes, machine/device details, and optional model revisions.

Record the code revision with `git rev-parse HEAD` and inspect uncommitted changes with `git diff`. Keep sampling, feature budget, image resizing, device, threading, warm-up, and memory measurement consistent when comparing methods. A seed helps reproducibility but does not make wall-clock timing identical between runs.

The repository ignores `data/`, `datasets/`, and `results_edgefusion_all/`. Store experiment outputs in that results directory to avoid accidentally committing large datasets or generated files.

## 8. Command reference and validation

Get the current command-line options without starting an experiment:

```text
python download_tum.py --help
python matching_comparison_all_poseguard_v4.py --help
python run_all_edgefusion_experiments_FULL_MATRIX_FINAL_STATS.py --help
python statistical_analysis.py --help
python reproduce_paper_results_updated.py --help
python generate_cross_sequence_all_metrics_FINAL_STATS.py --help
```

Run the independent statistics test:

```text
python -m pytest tests/test_statistical_analysis.py -q
```

**Legacy entry points:** `enhanced_matching_benchmark.py`, `tests/test_smoke.py`, and `tests/test_all_comparison.py` still depend on `matching_comparison_all.py`, which is absent from the current working tree. Consequently, a blanket `python -m pytest` may fail during collection. Older notebooks and batch files can also reference historical paths. Use the v4 commands in this guide; the legacy imports need updating before those entry points can be treated as validation of v4.

For notebook exploration, launch `python -m jupyterlab`, select the intended environment's kernel, and inspect each notebook's script imports and dataset paths before running its cells.
