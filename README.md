# Evaluation Bias in Cattle Behaviour Classification from Collar Accelerometry

### A Protocol-and-Input Decomposition on a Public Cohort

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0%2B-orange)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Data: Zenodo](https://img.shields.io/badge/data-10.5281%2Fzenodo.4064802-1682D4)](https://doi.org/10.5281/zenodo.4064802)

Reproducibility code for the paper:

> **Bokani, A.** *Evaluation bias in cattle behaviour classification from collar
> accelerometry: a protocol-and-input decomposition on a public cohort.* 2026.

---

## What this repository is (and is not)

Deep-learning models that classify cattle behaviour from collar accelerometers
routinely report **>95 % accuracy**, but those numbers usually come from
training and testing on the **same animals**, and some models are handed extra
label information that inflates them further.

This repository is the reference implementation of an **honest evaluation
template** — a factorial **input-set × protocol decomposition** that separates
genuine architectural gains from two confounding effects:

1. the strong second-to-second **autocorrelation** of behaviour, and
2. **leakage from privileged inputs** (a behaviour-derived channel).

It is **not** a "our model wins" package. Measured on one common cohort, holding
everything else fixed, the two evaluation choices dwarf the architecture:

| What changes (everything else held fixed) | Effect on reported accuracy |
|---|---|
| Shuffling overlapping windows (leaky CV) vs. a leakage-free split | **+4.6 pp** inflation (clean, one-variable contrast) |
| Adding a behaviour-history input channel | jumps to **~0.99** — no better than a trivial "repeat the last label" rule |
| Swapping the architecture (7 models, leakage-free, accel-only) | **3.9 pp** total spread; no deep or attention model beats a plain gradient-boosted tree |

So the deployment-honest, leakage-free, accelerometer-only, cross-animal accuracy
is **~0.76–0.80**, not the 95 %+ routinely reported. The **STA-LSTM-H**
architecture is included as a *testbed*, not as a new accuracy record. Two limits
on interpretation, stated plainly: the halter labels agree with visual
observation only 88–95 % of the time, and the 18 animals come from **one farm
across three trials (2015–2016)**, so leave-one-animal-out measures
generalisation *across animals* but not *across farms or hardware*.

---

## Models evaluated

All neural backbones share **dual heads**: a regression head for pseudo-position
`(x, y)` and a classification head for 3 behaviour classes (`Other`,
`Ruminating`, `Eating`). The two confound-carrying inputs are isolated by which
models receive the behaviour channel.

| Model | Input features | Architecture |
|---|---|---|
| **STA-LSTM-H** | 14 accel + behaviour channel (15) | LSTM-128 → MultiheadAttention(4) → LSTM-64 (+ opt. Kalman) |
| **STA-LSTM** | 14 accel + behaviour channel (15) | STA-LSTM-H without Kalman post-processing |
| **LSTM** | 14 accel only | stacked LSTM 96×2 |
| **GRU** | 14 accel only | stacked GRU 96×2 |
| **Transformer** | 14 accel only | encoder-only, d_model=128, 4 heads, 2 layers |
| **1D-CNN** | 14 accel only | temporal convolutional classifier |
| XGBoost | 14 accel only | gradient-boosted trees (PLF industry baseline) |
| KF / EKF-HMM | position observations (noisy) | classical filters, no training |

The factorial decomposition runs each neural model under both input sets
(accelerometer-only vs. with the behaviour channel) and both protocols
(within-animal vs. leave-one-animal-out), which is what isolates true
architectural gains from autocorrelation and privileged-input leakage.

The **14 accelerometer features** are `{mean, std, min, max}` of each axis plus
`mag_mean, mag_std`, aggregated into **1-second windows** from raw **10 Hz**
triaxial collar data. Pseudo-position `(x, y)` is a **relative-motion proxy**
from double-integrated, demeaned acceleration (scale `k = dt²/1e6`) — it is *not*
GPS, and the paper discloses this explicitly.

---

## Dataset (required — not included in this repo)

The data is **not mirrored here** (8.22 GB across 36 CSVs) and must be downloaded
once from the original source:

- **Download (Zenodo, DOI):** <https://doi.org/10.5281/zenodo.4064802>
- **Data citation:** Pavlovič, D. *et al.* (2021), accelerometer + halter cattle
  behaviour dataset, Zenodo.

After downloading, place the files so the layout is exactly:

```
<repo>/data/raw/
├── accel-01.csv  …  accel-18.csv     # 18 files, ~5 GB  (10 Hz triaxial accel)
└── halter-01.csv …  halter-18.csv    # 18 files, ~3 GB  (behaviour labels)
```

That is all the configuration needed. The code resolves `<repo>/data/raw/`
automatically (`src/data_loader.py`, `src/run_animal.py`); to keep the data
elsewhere, pass `--data-root /path/to/csvs` to any script or set
`DataConfig(zenodo_root=...)`. The directory is git-ignored, so it will never be
committed.

---

## Installation

Requires **Python 3.10+** (tested on 3.12). A GPU is optional — everything runs
on CPU, just slower.

```bash
git clone https://github.com/aubokani/cattle-accel-evaluation-bias.git
cd cattle-accel-evaluation-bias
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Dependencies (`requirements.txt`): `torch`, `numpy`, `pandas`, `scikit-learn`,
`scipy`, `matplotlib`, `imbalanced-learn`.

---

## Quick smoke test (≈ 1 minute)

Confirm the pipeline runs end-to-end on a tiny slice before launching anything
large:

```bash
python src/run_animal.py --animal-id 1 --folds 2 --epochs 2 \
       --max-windows 500 --max-horizon 0 --device cpu
# → writes Results/python_pipeline/animal-01/comparison_report.csv
```

The first run builds a 1-second **feature cache** at
`Results/python_pipeline/feature_cache/` (git-ignored); subsequent runs reuse it.

---

## Reproducing the paper

Each experiment is one script per animal; results are then consolidated into the
CSVs under `Results/` and rendered into figures. Run from the repo root with the
virtual environment active.

### 1. Within-animal 5-fold CV (the optimistic protocol)

```bash
# one animal
python src/run_animal.py --animal-id 1 --folds 5 --epochs 40
# all 18 animals (loop locally, or use the Slurm array — see HPC section)
for a in $(seq 1 18); do python src/run_animal.py --animal-id $a --folds 5 --epochs 40; done
python scripts/consolidate_results.py            # → Results/comparison_report_all_animals.csv
```

### 2. Cross-animal leave-one-animal-out (LOAO — the honest protocol)

```bash
for a in $(seq 1 18); do python scripts/run_loao_animal.py --held-out-animal-id $a --epochs 40; done
python scripts/consolidate_loao.py               # → Results/comparison_report_loao.csv
```

### 3. Input-set × architecture factorial ablation

```bash
for a in $(seq 1 18); do python src/run_animal.py --animal-id $a --folds 5 --epochs 40 --ablation; done
python scripts/consolidate_results.py            # ablation outputs → Results/*_ablation.csv
python scripts/ablation_compare.py
```

### 4. Trivial "previous-second label" baseline

```bash
python scripts/run_trivial_baselines.py --animal-ids $(seq 1 18) --folds 5
python scripts/predict_prev_baseline.py          # → Results/predict_prev_cohort.csv
```

### 5. Seed-stability audit

```bash
python scripts/run_seed_audit.py --animal-id 1   # default seeds 7 42 101 1729 2026 (as reported)
python scripts/consolidate_seed_audit.py         # → Results/aggregate_seed_audit_summary.csv
python scripts/wilcoxon_effects.py               # significance + effect sizes
```

### 6. Protocol-and-input decomposition, statistics, and paper figures

These reproduce the exact numbers and the three figures in the paper:

```bash
# leakage ladder (protocols A–D on the same backbone; Slurm array on HPC)
python scripts/leakage_experiment.py --all --epochs 5   # original low-budget ladder → Results/leakage_protocol_*.csv
# headline input-ablation vs predict-previous ceiling tables
python scripts/build_paper_tables.py             # → Results/paper_table1/2_*.csv
# clean leakage contrasts, full 7-model pairwise Wilcoxon+TOST, param counts
python scripts/revision_stats.py                 # → Results/revision/*.csv
# regenerate the three paper figures
python scripts/make_paper_figures.py             # → Results/figures/fig_*.{png,eps}
```

### 7. Revision analyses (Scientific Reports review, 2026-10)

Re-analysis of existing per-animal outputs (10-s label grid, metric dependence,
daily time budgets, mixed model, TOST sensitivity, LOAO floors, smoothing):

```bash
python scripts/revision2_stats.py                # → Results/revision2/  (SUMMARY.md lists every number)
```

New training controls (stated 40-epoch recipe ladder + one-factor mechanism rungs,
class-weighting / CE-only / seed / leave-one-trial-out LOAO controls, XGBoost controls).
Each family is a Slurm task-file array:

```bash
mkdir -p Results/revision2_hpc/tasks logs
python scripts/revision2_leakage_ladder.py --list-tasks core > Results/revision2_hpc/tasks/ladder_core.txt   # also: sweep, mech
python scripts/revision2_loao_variant.py  --list-tasks weighting > Results/revision2_hpc/tasks/loao_weighting.txt   # also: seeds, loto
python scripts/revision2_xgb.py           --list-tasks > Results/revision2_hpc/tasks/xgb.txt
T=Results/revision2_hpc/tasks/ladder_core.txt
sbatch --array=0-$(( $(wc -l < $T) - 1 )) --export=ALL,TASKFILE=$T scripts/submit_revision2_ladder.sh
# likewise submit_revision2_loao.sh / submit_revision2_xgb.sh with their task files
python scripts/revision2_hpc_analysis.py         # → Results/revision2_hpc/SUMMARY.md
```

Pre-computed outputs for all of the above are committed under `Results/`
(aggregate CSVs, per-animal reports, prediction arrays, and figures), so the
paper's numbers and plots can be re-derived without a full re-run.

---

## Running on HPC (Slurm)

The study was produced on the CQUniversity **Ada Lovelace** cluster as an
18-task Slurm array (one task per animal). Cluster specifics (partitions,
walltime, GPU types) are documented in [`docs/hpc/ada_reference.md`](docs/hpc/ada_reference.md)
and parameterised in [`config/hpc.cfg`](config/hpc.cfg).

```bash
bash scripts/setup_env.sh           # once: load Python module, build .venv, pip install
sbatch scripts/submit_smoke.sh      # 1-task sanity check on a compute node
sbatch scripts/submit_hpc.sh        # within-animal array: --array=1-18, 1 GPU, 32 GB, 4 h/task
sbatch scripts/submit_loao.sh       # leave-one-animal-out array
sbatch scripts/submit_hpc_ablation.sh
```

`scripts/` also contains the other submission variants used (`submit_array_l40.sh`,
`submit_array_h100.sh`, CPU-only `submit_loop_cpu.sh`, seed-audit, block-CV).
To port to another cluster, edit the partition names, account, and module name
in `config/hpc.cfg` and the `#SBATCH` headers in `scripts/submit_*.sh`.

---

## Running purely on a local machine (no Slurm)

Everything runs without a cluster — keep these points in mind:

- **No GPU needed.** Pass `--device cpu` (or leave `--device auto`, which falls
  back to CPU, and to Apple-Silicon `mps` if available). A GPU mainly speeds up
  the neural training; the classical filters and trivial baseline are CPU-only.
- **Memory.** Ingestion streams the 10 Hz CSVs in chunks, so peak RAM is modest
  (~a few GB). The default per-task request on HPC is 32 GB, but local runs with
  the default `--max-windows 86400` (24 h/animal) fit comfortably in 8–16 GB.
- **Runtime.** A full 40-epoch, 5-fold run for one animal is minutes on a GPU
  and longer on CPU. To iterate quickly, cap the data with
  `--max-windows` (e.g. `3600` = first hour) and/or reduce `--epochs`/`--folds`.
- **No `sbatch`.** Replace each array with a shell loop over `seq 1 18`, exactly
  as shown in the "Reproducing the paper" section above.
- **First run is slower.** It builds the per-animal feature cache; later runs
  reuse `Results/python_pipeline/feature_cache/`.

---

## Repository layout

```
STA-LSTM-H/
├── README.md                       # this file
├── LICENSE                         # MIT
├── requirements.txt                # Python dependencies
├── .gitignore
│
├── src/                            # core library (one-way chain: data_loader → model → trainer)
│   ├── data_loader.py              # 10 Hz streaming, 1-s windowing, pseudo-position, sequence builders
│   ├── model.py                    # model registry, neural backbones, behaviour-adaptive Kalman, classical filters
│   ├── trainer.py                  # 5-fold CV, multi-task loss, autoregressive eval, Wilcoxon, ranking
│   └── run_animal.py               # per-animal CLI entry point (within-animal CV; --ablation factorial)
│
├── scripts/
│   ├── run_loao_animal.py          # leave-one-animal-out (cross-animal) runner
│   ├── run_trivial_baselines.py    # "repeat previous second's label" baseline
│   ├── predict_prev_baseline.py    # cohort-level previous-label baseline
│   ├── run_seed_audit.py           # multi-seed stability audit
│   ├── run_block_cv_animal.py      # blocked (contiguous) cross-validation variant
│   ├── consolidate_results.py      # → Results/comparison_report_all_animals.csv
│   ├── consolidate_loao.py         # → Results/comparison_report_loao.csv
│   ├── consolidate_loao_ablation.py
│   ├── consolidate_seed_audit.py
│   ├── aggregate_18animals.py      # cohort-level aggregate summaries
│   ├── ablation_compare.py         # input-set × architecture comparison
│   ├── class_prevalence_audit.py   # behaviour-class balance per animal
│   ├── wilcoxon_effects.py         # significance tests + effect sizes
│   ├── mcu_latency_estimate.py     # on-device latency estimate
│   ├── make_figures.py             # all paper figures → Results/figures/
│   ├── fetch_results.sh            # rsync per-animal outputs back from the cluster
│   ├── setup_env.sh                # build .venv + install deps on the cluster
│   └── submit_*.sh                 # Slurm submission scripts (see "Running on HPC")
│
├── config/
│   └── hpc.cfg                     # cluster paths, Slurm partitions, Python module
│
├── docs/
│   └── hpc/ada_reference.md        # Ada Lovelace cluster operational reference
│
├── Results/                        # committed outputs that back the paper
│   ├── *.csv                       # aggregate summaries, comparison reports, Wilcoxon stats
│   ├── figures/                    # all paper figures (.png + .eps pairs)
│   ├── python_pipeline/            # within-animal: animal-NN/{comparison_report.csv,
│   │                               #   wilcoxon_stats.csv, run_summary.json, last_fold_predictions.npy}
│   └── python_pipeline_ablation/   # same layout, factorial-ablation runs
│
└── data/raw/                       # YOU place the Zenodo CSVs here — git-ignored, not shipped
    ├── accel-01.csv … accel-18.csv
    └── halter-01.csv … halter-18.csv
```

---

## Citation

If you use this code or the evaluation template, please cite:

```bibtex
@article{bokani2026crossanimal,
  title   = {Cross-Animal Evaluation of Cattle Behaviour Classification from
             Collar Accelerometry: A Protocol-and-Input Decomposition},
  author  = {Bokani, Ayub},
  journal = {Smart Agricultural Technology},
  year    = {2026},
  note    = {Submitted}
}
```

**Dataset:** Pavlovič, D. *et al.* (2021), Zenodo,
DOI [10.5281/zenodo.4064802](https://doi.org/10.5281/zenodo.4064802).

## License

Released under the [MIT License](LICENSE). The dataset is distributed separately
by its original authors under the terms stated on Zenodo.
```