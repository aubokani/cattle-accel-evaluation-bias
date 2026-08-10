#!/usr/bin/env python
"""
Leakage quantification for the within-animal validation protocol.

Reviewer weakness #6 (SAT / ATECH-D-26-01740): the within-animal protocol uses
shuffled overlapping windows (seq_len=25, stride=1 → 96% overlap between adjacent
windows). Under StratifiedKFold(shuffle=True), near-identical adjacent windows are
split across train and test, so the two sets share most of their raw samples. This
inflates within-animal accuracy toward the trivial predict-previous ceiling.

This script measures the effect directly, on the *real pipeline model*, per animal:

  Protocol A — shuffled_overlap  : stride=1 windows, StratifiedKFold(shuffle=True)
                                   (the leaky protocol the paper originally reported)
  Protocol B — blocked_overlap   : stride=1 windows, contiguous temporal blocks as
                                   folds (no shuffle) — neighbours stay together but
                                   train/test are still adjacent at block seams
  Protocol C — temporal_holdout  : stride=1 windows, a single chronological split
                                   (first train_frac → train, last 1-train_frac →
                                   test) with a GUARD GAP of seq_len windows removed
                                   at the seam so NO test window shares any raw
                                   sample with any train window. This is the honest
                                   within-animal estimate.
  Protocol D — nonoverlap_temporal: stride=seq_len windows (0% overlap) + chronological
                                   split. Fully leakage-free; strictest control.

For each animal and protocol we train the accel-only STA-LSTM-H backbone (the
paper's model, 14 accel features, NO behaviour-history channel) and report test
accuracy / macro-F1. The gap A - C (and A - D) is the leakage-driven inflation.

Also emits the exact overlap fraction and an estimate of the fraction of test
windows whose raw-sample support is (partly) seen in training under Protocol A.

Usage:
    source .venv/bin/activate
    python scripts/leakage_experiment.py --animals 1 2 3 --epochs 5
    python scripts/leakage_experiment.py --all --epochs 5    # all 18
Outputs:
    Results/leakage_protocol_per_animal.csv
    Results/leakage_protocol_summary.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, KFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_loader import ACCEL_FEAT_COLS  # noqa: E402
from model import build_model, train_epoch, evaluate  # noqa: E402

CACHE = ROOT / "Results" / "python_pipeline" / "feature_cache"
OUT = ROOT / "Results"
SEQ_LEN = 25
SEED = 42


# ─────────────────────────────────────────────────────────────────────────────
# Sequence construction with configurable stride (train-set normalisation only)
# ─────────────────────────────────────────────────────────────────────────────
def build_windows(df: pd.DataFrame, feature_cols: list[str], seq_len: int,
                  stride: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X, Yc, start_idx) where start_idx[i] is the first raw row of window i.

    Normalisation is deliberately NOT applied here; each protocol z-scores using
    its own training partition to avoid a second (normalisation) leak.
    """
    feats = df[feature_cols].to_numpy(np.float32)
    cls = df["behavior"].to_numpy(np.int64)
    max_start = len(df) - seq_len
    starts = np.arange(0, max_start, stride, dtype=np.int64)
    X = np.empty((len(starts), seq_len, len(feature_cols)), np.float32)
    Yc = np.empty(len(starts), np.int64)
    for i, s in enumerate(starts):
        X[i] = feats[s:s + seq_len]
        Yc[i] = cls[s + seq_len]           # predict the label at the step after the window
    return X, Yc, starts


def zscore_fit(X_tr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = X_tr.reshape(-1, X_tr.shape[-1])
    mu = flat.mean(0)
    sigma = flat.std(0)
    sigma[sigma < 1e-9] = 1.0
    return mu.astype(np.float32), sigma.astype(np.float32)


def zscore_apply(X: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return ((X - mu) / sigma).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Train one accel-only STA-LSTM-H backbone on a given split; return acc, f1
# ─────────────────────────────────────────────────────────────────────────────
def train_eval(X_tr, Yc_tr, X_te, Yc_te, epochs, device) -> tuple[float, float]:
    mu, sigma = zscore_fit(X_tr)
    X_tr = zscore_apply(X_tr, mu, sigma)
    X_te = zscore_apply(X_te, mu, sigma)

    # dummy xy targets (this experiment only scores classification)
    zeros_tr = np.zeros((len(X_tr), 2), np.float32)
    zeros_te = np.zeros((len(X_te), 2), np.float32)
    norm_meta = {"xy_min": np.zeros(2, np.float32), "xy_span": np.ones(2, np.float32)}

    ds_tr = torch.utils.data.TensorDataset(
        torch.from_numpy(X_tr), torch.from_numpy(zeros_tr), torch.from_numpy(Yc_tr))
    ds_te = torch.utils.data.TensorDataset(
        torch.from_numpy(X_te), torch.from_numpy(zeros_te), torch.from_numpy(Yc_te))
    ld_tr = torch.utils.data.DataLoader(ds_tr, batch_size=256, shuffle=True)
    ld_te = torch.utils.data.DataLoader(ds_te, batch_size=512, shuffle=False)

    # class-weighted CE for imbalance (matches pipeline's weighted_ce branch)
    counts = np.bincount(Yc_tr, minlength=3).astype(np.float64)
    counts[counts == 0] = 1.0
    w = counts.sum() / (len(counts) * counts)
    ce_w = torch.tensor(w, dtype=torch.float32, device=device)

    model = build_model("lstm_h", X_tr.shape[-1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-4)
    for _ in range(epochs):
        train_epoch(model, ld_tr, opt, device, ce_w, 1.2)

    _, _, _, pred_cls, true_cls = evaluate(model, ld_te, norm_meta, device)
    acc = accuracy_score(true_cls, pred_cls)
    f1 = f1_score(true_cls, pred_cls, average="macro", zero_division=0)
    return float(acc), float(f1)


# ─────────────────────────────────────────────────────────────────────────────
# The four protocols
# ─────────────────────────────────────────────────────────────────────────────
def protocol_shuffled_overlap(df, epochs, device):
    X, Yc, _ = build_windows(df, ACCEL_FEAT_COLS, SEQ_LEN, stride=1)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    accs, f1s = [], []
    for tr, te in skf.split(np.zeros(len(Yc)), Yc):
        a, f = train_eval(X[tr], Yc[tr], X[te], Yc[te], epochs, device)
        accs.append(a); f1s.append(f)
    return float(np.mean(accs)), float(np.mean(f1s))


def protocol_blocked_overlap(df, epochs, device):
    X, Yc, _ = build_windows(df, ACCEL_FEAT_COLS, SEQ_LEN, stride=1)
    kf = KFold(n_splits=5, shuffle=False)          # contiguous temporal blocks
    accs, f1s = [], []
    for tr, te in kf.split(X):
        a, f = train_eval(X[tr], Yc[tr], X[te], Yc[te], epochs, device)
        accs.append(a); f1s.append(f)
    return float(np.mean(accs)), float(np.mean(f1s))


def protocol_temporal_holdout(df, epochs, device, train_frac=0.8):
    X, Yc, starts = build_windows(df, ACCEL_FEAT_COLS, SEQ_LEN, stride=1)
    n = len(X)
    cut = int(n * train_frac)
    # guard gap: drop SEQ_LEN windows at the seam so no test window overlaps a train window
    tr_idx = np.arange(0, cut - SEQ_LEN)
    te_idx = np.arange(cut, n)
    return train_eval(X[tr_idx], Yc[tr_idx], X[te_idx], Yc[te_idx], epochs, device)


def protocol_nonoverlap_temporal(df, epochs, device, train_frac=0.8):
    X, Yc, _ = build_windows(df, ACCEL_FEAT_COLS, SEQ_LEN, stride=SEQ_LEN)  # 0% overlap
    n = len(X)
    cut = int(n * train_frac)
    return train_eval(X[:cut], Yc[:cut], X[cut:], Yc[cut:], epochs, device)


def overlap_diagnostics(n_rows: int, seq_len: int) -> dict:
    """Fraction of raw samples shared between two adjacent stride-1 windows, and
    the share of stride-1 test windows (under a random 20% test split) expected to
    have at least one temporal neighbour in the training set."""
    adj_overlap = (seq_len - 1) / seq_len
    # P(a given window has neither immediate neighbour in test) under 20% test:
    # each of its 2 neighbours is in train w.p. 0.8 → expected fraction of test
    # windows with >=1 train neighbour = 1 - 0.2^2 = 0.96
    frac_test_touching_train = 1 - 0.2 ** 2
    return {
        "adjacent_window_overlap": adj_overlap,
        "expected_test_windows_with_train_neighbour": frac_test_touching_train,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--animals", type=int, nargs="*", default=[1, 2, 3])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--epochs", type=int, default=5)
    args = ap.parse_args()

    animals = list(range(1, 19)) if args.all else args.animals
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED); np.random.seed(SEED)
    print(f"device={device}  animals={animals}  epochs={args.epochs}")

    diag = overlap_diagnostics(86400, SEQ_LEN)
    print(f"[diagnostics] adjacent-window raw-sample overlap = "
          f"{diag['adjacent_window_overlap']:.3f}")
    print(f"[diagnostics] expected test windows with a train neighbour = "
          f"{diag['expected_test_windows_with_train_neighbour']:.3f}")

    rows = []
    for aid in animals:
        fp = CACHE / f"animal-{aid:02d}_1s_features.csv"
        if not fp.exists():
            print(f"  !! missing cache {fp}, skipping"); continue
        df = pd.read_csv(fp)
        print(f"\n── animal {aid:02d}  ({len(df)} rows) ──")
        protos = {
            "A_shuffled_overlap":   protocol_shuffled_overlap,
            "B_blocked_overlap":    protocol_blocked_overlap,
            "C_temporal_holdout":   protocol_temporal_holdout,
            "D_nonoverlap_temporal": protocol_nonoverlap_temporal,
        }
        for name, fn in protos.items():
            acc, f1 = fn(df, args.epochs, device)
            print(f"    {name:24s} acc={acc:.4f}  f1={f1:.4f}")
            rows.append({"animal": aid, "protocol": name, "accuracy": acc, "f1": f1})

    per = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    per.to_csv(OUT / "leakage_protocol_per_animal.csv", index=False)

    summ = (per.groupby("protocol")
               .agg(acc_mean=("accuracy", "mean"), acc_std=("accuracy", "std"),
                    f1_mean=("f1", "mean"), f1_std=("f1", "std"),
                    n=("animal", "nunique"))
               .reset_index())
    # inflation gaps vs the honest protocols
    a = summ.set_index("protocol")["acc_mean"]
    for honest in ("C_temporal_holdout", "D_nonoverlap_temporal"):
        if "A_shuffled_overlap" in a and honest in a:
            print(f"\n[INFLATION] A_shuffled_overlap - {honest} = "
                  f"{a['A_shuffled_overlap'] - a[honest]:+.4f} accuracy")
    summ.to_csv(OUT / "leakage_protocol_summary.csv", index=False)
    print(f"\nSaved → {OUT}/leakage_protocol_per_animal.csv")
    print(f"Saved → {OUT}/leakage_protocol_summary.csv")


if __name__ == "__main__":
    main()
