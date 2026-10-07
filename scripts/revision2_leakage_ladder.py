#!/usr/bin/env python
"""scripts/revision2_leakage_ladder.py — leakage ladder re-run at the stated
recipe, plus mechanism-isolating rungs (revision of 2026-10-07).

Why (reviewer comments R1.M2, R1.M3, R3.M1, R3.M2)
---------------------------------------------------
The original ladder (scripts/leakage_experiment.py, Slurm 165856) trained the
accelerometer-only STA-LSTM-H backbone for 5 epochs at batch 256 with all-zero
regression targets (classification-only in effect), whereas main.tex Table 8
states 40 epochs / batch 64 / multi-task for "all reported experiments". This
script leaves the original script untouched and re-runs the ladder with the
recipe actually used by the LOAO experiments (scripts/run_loao_animal.py
defaults + TrainConfig): Adam lr 2e-3, weight-decay 1e-4, grad-clip 1.2,
batch 64, 40 epochs, multi-task loss MSE + 0.35·CE with the *real*
pseudo-position target (min–max fitted on the training partition) and
class-frequency-weighted CE (data_loader.audit_class_balance weights fitted on
the training partition). Features are z-scored with training-partition
statistics, exactly as in the original ladder and in LOAO. The training step
is the canonical `model.train_epoch`. The model is the same accel-only
STA-LSTM-H (`build_model("lstm_h", 14)`) as the original ladder; XGBoost
(hyper-parameters of trainer._eval_xgboost, unweighted as in the headline
LOAO) is run on every split as a second, cheap model (R1.M3 "at least one
further model").

Protocols (25-step windows; window i uses rows i..i+24 and predicts the label
of row i+25, so the *support* of window i is rows i..i+25)
-------------------------------------------------------------------------
Original ladder, recipe-matched (definitions identical to leakage_experiment.py):
  A    shuffled stride-1, StratifiedKFold(5, shuffle, seed 42)
  B    stride-1, KFold(5) contiguous ~4.8 h blocks, no guard gap
  C    stride-1, chronological 80/20, 25-window guard gap at the seam
  D    stride-25 (non-overlapping), chronological 80/20
  Deq  D with the epoch count scaled so the number of gradient steps equals
       C's (R3.M1b: D is otherwise confounded with a ~25x smaller budget).
       One factor vs D: optimisation budget.

Training-budget sweep (R3.M1a): A_e{5,10,20}, B_e{5,10,20} — identical to A/B
but with 5/10/20 epochs. One factor vs A/B: epochs.

Mechanism family M (R1.M2, R3.M2): "blocked-shuffled" 5-fold CV. The
timeline of stride-1 windows is cut into contiguous blocks of L windows
(seconds); blocks are assigned to folds with KFold(5, shuffle, seed 42). With
purge gap g = 26, every training window whose support overlaps the support of
any test window (|i − j| <= 25) is removed, so no train/test pair shares a raw
sample and every train target is >= 26 s (> 2 label epochs of 10 s) from every
test target. Training sets of ALL M rungs are randomly subsampled (seed 42 +
fold) to one common size N* per animal (the smallest M training set, which is
L25_g26), so training-set size is held fixed across the family.
  M_L1_g0       window-level shuffle (A without stratification, size-matched)
  M_L25_g0      25-s blocks, overlap retained
  M_L25_g26     25-s blocks, overlapping train windows purged  (same test set
                as M_L25_g0 → M_L25_g0 − M_L25_g26 isolates OVERLAP alone)
  M_L300_g26    5-min blocks, purged
  M_L3600_g26   1-h blocks, purged
  M_L17280_g26  5 contiguous ~4.8-h blocks, purged (= B + guard, size-matched)
Reading the family: L25_g0 → L25_g26 removes shared samples only (overlap);
L25_g26 → L300 → L3600 → L17280 increases temporal distance between test
windows and the nearest training data (proximity), ending at B-like
contiguous-block extrapolation (day-part coverage). A − M_L1_g0 = stratification
+ training-set size; M_L1_g0 − M_L25_g0 = sub-window-scale granularity.

Usage
-----
  python scripts/revision2_leakage_ladder.py --list-tasks core   > tasks.txt
  python scripts/revision2_leakage_ladder.py --task "1 A 0"
  python scripts/revision2_leakage_ladder.py --task "1 A 0" --epochs 1 --max-rows 3000   # pilot
Outputs: Results/revision2_hpc/leakage_ladder/per_fold/a{NN}_{protocol}_f{k}_{model}.{json,npz}
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from revision2_common import (OUT_ROOT, RECIPE, ROOT, ce_class_weights,  # noqa: E402
                              done, env_info, fit_eval_nn, fit_eval_xgb,
                              save_result, setup_torch)

sys.path.insert(0, str(ROOT / "src"))
from data_loader import ACCEL_FEAT_COLS  # noqa: E402

CACHE = ROOT / "Results" / "python_pipeline" / "feature_cache"
OUT = OUT_ROOT / "leakage_ladder" / "per_fold"
SEQ_LEN = 25
SEED = 42
PURGE = SEQ_LEN + 1          # min |i-j| between any train and test window start

CORE = ["A", "B", "C", "D", "Deq"]
SWEEP = [f"{p}_e{e}" for p in ("A", "B") for e in (5, 10, 20)]
MECH = ["M_L1_g0", "M_L25_g0", "M_L25_g26", "M_L300_g26", "M_L3600_g26", "M_L17280_g26"]
N_FOLDS = {**{p: 5 for p in ("A", "B")}, "C": 1, "D": 1, "Deq": 1,
           **{p: 5 for p in SWEEP}, **{p: 5 for p in MECH}}


def windows(df: pd.DataFrame, stride: int):
    """Same content as leakage_experiment.build_windows (vectorised), plus xy."""
    from numpy.lib.stride_tricks import sliding_window_view
    feats = df[ACCEL_FEAT_COLS].to_numpy(np.float32)
    cls = df["behavior"].to_numpy(np.int64)
    xy = df[["x", "y"]].to_numpy(np.float32)
    starts = np.arange(0, len(df) - SEQ_LEN, stride, dtype=np.int64)
    V = sliding_window_view(feats, (SEQ_LEN, feats.shape[1]))[:, 0]
    X = np.ascontiguousarray(V[starts]).astype(np.float32)
    return X, cls[starts + SEQ_LEN], xy[starts + SEQ_LEN], starts


def mech_split(n: int, L: int, g: int, fold: int):
    blocks = np.arange(n) // L
    ub = np.unique(blocks)
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    tr_b, te_b = list(kf.split(ub))[fold]
    te_mask = np.isin(blocks, ub[te_b])
    tr_mask = ~te_mask
    if g > 0:
        # windows within g-1 of any test window are purged from training
        k = np.ones(2 * (g - 1) + 1)
        near = np.convolve(te_mask.astype(float), k, mode="same") > 0
        tr_mask &= ~near
    return np.flatnonzero(tr_mask), np.flatnonzero(te_mask)


def mech_nstar(n: int) -> int:
    sizes = []
    for p in MECH:
        L, g = (int(s[1:]) for s in p.split("_")[1:])
        for f in range(5):
            sizes.append(len(mech_split(n, L, g, f)[0]))
    return int(min(sizes))


def split(protocol: str, fold: int, Yc: np.ndarray, n: int):
    base = protocol.split("_e")[0] if protocol in SWEEP else protocol
    if base == "A":
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        return list(skf.split(np.zeros(n), Yc))[fold]
    if base == "B":
        return list(KFold(n_splits=5, shuffle=False).split(np.zeros(n)))[fold]
    if base == "C":
        cut = int(n * 0.8)
        return np.arange(0, cut - SEQ_LEN), np.arange(cut, n)
    if base in ("D", "Deq"):
        cut = int(n * 0.8)
        return np.arange(0, cut), np.arange(cut, n)
    if base in MECH:
        L, g = (int(s[1:]) for s in base.split("_")[1:])
        tr, te = mech_split(n, L, g, fold)
        nstar = mech_nstar(n)
        rng = np.random.default_rng(SEED + fold)
        tr = np.sort(rng.choice(tr, size=nstar, replace=False))
        return tr, te
    raise ValueError(protocol)


def list_tasks(which: str) -> list[str]:
    protos = {"core": CORE, "sweep": SWEEP, "mech": MECH}[which]
    return [f"{a} {p} {f}" for a in range(1, 19) for p in protos for f in range(N_FOLDS[p])]


def run_task(aid: int, protocol: str, fold: int, epochs_override: int | None,
             max_rows: int | None, out_dir: Path) -> None:
    device = setup_torch()
    df = pd.read_csv(CACHE / f"animal-{aid:02d}_1s_features.csv")
    if max_rows:
        df = df.iloc[:max_rows].reset_index(drop=True)
    stride = SEQ_LEN if protocol in ("D", "Deq") else 1
    X, Yc, Yxy_raw, _ = windows(df, stride)
    n = len(X)
    tr, te = split(protocol, fold, Yc, n)

    epochs = RECIPE["epochs"]
    if protocol in SWEEP:
        epochs = int(protocol.split("_e")[1])
    if protocol == "Deq":
        n_c = int(len(df) - SEQ_LEN)                       # C's stride-1 window count
        steps_c = RECIPE["epochs"] * int(np.ceil((int(n_c * 0.8) - SEQ_LEN) / RECIPE["batch_size"]))
        epochs = int(round(steps_c / np.ceil(len(tr) / RECIPE["batch_size"])))
    if epochs_override:
        epochs = epochs_override

    X_tr, X_te = X[tr], X[te]
    flat = X_tr.reshape(-1, X_tr.shape[-1])
    mu, sd = flat.mean(0), flat.std(0)
    sd[sd < 1e-9] = 1.0
    X_tr = ((X_tr - mu) / sd).astype(np.float32)
    X_te = ((X_te - mu) / sd).astype(np.float32)
    xy_min = Yxy_raw[tr].min(0)
    xy_span = Yxy_raw[tr].max(0) - xy_min
    xy_span[xy_span < 1e-9] = 1.0
    Yxy_tr = ((Yxy_raw[tr] - xy_min) / xy_span).astype(np.float32)
    Yxy_te = ((Yxy_raw[te] - xy_min) / xy_span).astype(np.float32)
    meta_n = {"xy_min": xy_min, "xy_span": xy_span}

    base_meta = {"animal": aid, "protocol": protocol, "fold": fold,
                 "n_windows": n, "n_train": int(len(tr)), "n_test": int(len(te)),
                 "stride": stride, **env_info()}
    stem = f"a{aid:02d}_{protocol}_f{fold}"

    p_nn = out_dir / f"{stem}_lstm_h_accel"
    if not done(p_nn):
        w = ce_class_weights(Yc[tr])
        res = fit_eval_nn("lstm_h", X_tr, Yxy_tr, Yc[tr], X_te, Yxy_te, Yc[te],
                          meta_n, w, objective="multitask", seed=SEED,
                          epochs=epochs, device=device,
                          log_epochs=(1, 10, 20) if protocol in ("A", "B", "C") else ())
        save_result(p_nn, res, {**base_meta, "model": "STA-LSTM-H (accel)",
                                "epochs": epochs, "batch_size": RECIPE["batch_size"],
                                "objective": "multitask", "ce_weighting": "weighted",
                                "seed": SEED})
        print(f"[{stem}] STA-LSTM-H acc={res['accuracy']:.4f} f1={res['f1_macro']:.4f} "
              f"bal={res['balanced_accuracy']:.4f} epochs={epochs} "
              f"n_tr={len(tr)} {res['train_s']:.0f}s", flush=True)

    if protocol not in SWEEP and protocol != "Deq":
        p_x = out_dir / f"{stem}_xgboost"
        if not done(p_x):
            res = fit_eval_xgb(X_tr, Yc[tr], X_te, Yc[te], weighting="none", seed=SEED)
            save_result(p_x, res, {**base_meta, "model": "XGBoost",
                                   "ce_weighting": "none", "seed": SEED})
            print(f"[{stem}] XGBoost acc={res['accuracy']:.4f} f1={res['f1_macro']:.4f} "
                  f"{res['train_s']:.0f}s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list-tasks", choices=["core", "sweep", "mech"])
    ap.add_argument("--task", help='"<animal> <protocol> <fold>"')
    ap.add_argument("--task-file")
    ap.add_argument("--task-index", type=int)
    ap.add_argument("--epochs", type=int, default=None, help="pilot override only")
    ap.add_argument("--max-rows", type=int, default=None, help="pilot only")
    ap.add_argument("--out-dir", type=Path, default=OUT)
    a = ap.parse_args()
    if a.list_tasks:
        print("\n".join(list_tasks(a.list_tasks)))
        return
    if a.task_file is not None:
        lines = Path(a.task_file).read_text().split("\n")
        a.task = lines[a.task_index]
    aid, proto, fold = a.task.split()
    run_task(int(aid), proto, int(fold), a.epochs, a.max_rows, a.out_dir)


if __name__ == "__main__":
    main()
