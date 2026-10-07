"""scripts/revision2_common.py — shared helpers for the 2026-10-07 revision HPC runs.

Used by:
    scripts/revision2_leakage_ladder.py   (R1.M2/M3, R3.M1/M2)
    scripts/revision2_loao_variant.py     (R1.M4/M5, R3.M4/M6, R3.m2)
    scripts/revision2_xgb.py              (R1.M4, R1.m8, R2.M7, R3.m1/m2)

Nothing here changes the behaviour of any existing script or `src/` module; the
helpers only *call* the canonical pipeline functions:

  * multi-task training uses `model.train_epoch` (MSE + 0.35·CE, weighted CE,
    grad-clip 1.2) — the exact function every reported neural result used;
  * evaluation uses `model.evaluate`;
  * class weights come from `data_loader.audit_class_balance` (w_norm);
  * the XGBoost hyper-parameters replicate `trainer._eval_xgboost` exactly
    (300 trees, depth 6, lr 0.05, hist, softprob), adding only an optional
    `sample_weight` and an explicit `n_jobs` (= Slurm cpus) so it does not
    oversubscribe a shared node.

Two deliberate, documented additions:

  * CE-only objective (`objective="ce_only"`): identical loop to
    `model.train_epoch` but the loss is `0.35·CE` (the MSE term removed, all
    else unchanged — a one-factor change; Adam is approximately invariant to
    the constant 0.35).
  * loss-component logging: at a few epochs the mean MSE and mean CE on a
    fixed random subsample of training windows are recorded (eval mode), so
    the claim "the MSE term dominates" (main.tex l.385) can be checked
    against measured magnitudes rather than asserted (R1.M4, R3.M4).

Each neural model is seeded explicitly immediately before construction
(R3.m14), so a run is reproducible on its own rather than depending on its
position in a training loop.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data_loader import audit_class_balance  # noqa: E402
from model import build_model, evaluate, train_epoch  # noqa: E402

OUT_ROOT = ROOT / "Results" / "revision2_hpc"
ALPHA = 0.35            # CE weight in model.multitask_loss (unchanged)
TRIALS = {1: tuple(range(1, 7)), 2: tuple(range(7, 11)), 3: tuple(range(11, 19))}

# Recipe stated in main.tex Table 8 and used by run_loao_animal.py defaults.
RECIPE = dict(epochs=40, batch_size=64, lr=2e-3, weight_decay=1e-4, grad_clip=1.2)


def n_threads() -> int:
    """Allocated CPUs minus one. On Ada workq nodes a WekaFS client thread is
    pinned (busy-polling) to one core that can fall inside a job's cgroup;
    using every allocated core then stalls OpenMP barriers ~50-150x
    (measured 2026-10-07: 1D-CNN 2.8 ms/step at 7 threads vs 159 ms at 8)."""
    n = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    return max(1, n - 1)


def setup_torch() -> torch.device:
    torch.set_num_threads(n_threads())
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────
def class_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_cls: int = 3) -> dict:
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 cohen_kappa_score, confusion_matrix, f1_score)
    y_true = np.asarray(y_true, np.int64)
    y_pred = np.asarray(y_pred, np.int64)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_cls)))
    with np.errstate(divide="ignore", invalid="ignore"):
        rec = np.diag(cm) / cm.sum(1)
        prec = np.diag(cm) / cm.sum(0)
    out = {
        "n_test": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "confusion": cm.tolist(),
    }
    for c, name in enumerate(("other", "rum", "eat")):
        out[f"recall_{name}"] = float(rec[c]) if np.isfinite(rec[c]) else float("nan")
        out[f"precision_{name}"] = float(prec[c]) if np.isfinite(prec[c]) else float("nan")
    return out


def ce_class_weights(labels: np.ndarray) -> np.ndarray:
    """Exactly the pipeline's weights: audit_class_balance → w_norm."""
    decision, w = audit_class_balance(pd.DataFrame({"behavior": np.asarray(labels)}))
    if decision != "weighted":
        raise RuntimeError(f"class-balance audit returned {decision!r}; the "
                           "revision runs assume the weighted-CE branch "
                           "(no cohort animal exceeds 80%).")
    return np.asarray(w, np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Neural training
# ─────────────────────────────────────────────────────────────────────────────
def _train_epoch_ce_only(model, loader, optimizer, device, ce_weight, grad_clip):
    """Line-for-line copy of model.train_epoch with the MSE term removed."""
    model.train()
    tot, nb = 0.0, 0
    for xb, _yxy, yc in loader:
        xb = xb.to(device)
        yc = yc.to(device)
        optimizer.zero_grad(set_to_none=True)
        _pxy, pcls = model(xb)
        loss = ALPHA * nn.functional.cross_entropy(pcls, yc, weight=ce_weight)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        tot += loss.item()
        nb += 1
    return tot / max(nb, 1)


@torch.no_grad()
def _loss_components(model, X, Yxy, Yc, device, ce_weight, bs=2048) -> dict:
    model.eval()
    mse_s, ce_s, n = 0.0, 0.0, 0
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i + bs]).to(device)
        yxy = torch.from_numpy(Yxy[i:i + bs]).to(device)
        yc = torch.from_numpy(Yc[i:i + bs]).to(device)
        pxy, pc = model(xb)
        k = len(xb)
        mse_s += nn.functional.mse_loss(pxy, yxy).item() * k
        ce_s += nn.functional.cross_entropy(pc, yc, weight=ce_weight).item() * k
        n += k
    return {"mse": mse_s / n, "ce": ce_s / n, "alpha_ce": ALPHA * ce_s / n}


def fit_eval_nn(kind: str,
                X_tr, Yxy_tr, Yc_tr, X_te, Yxy_te, Yc_te, norm_meta: dict,
                class_weights: np.ndarray | None,
                objective: str = "multitask",
                seed: int = 42,
                epochs: int = RECIPE["epochs"],
                batch_size: int = RECIPE["batch_size"],
                lr: float = RECIPE["lr"],
                weight_decay: float = RECIPE["weight_decay"],
                grad_clip: float = RECIPE["grad_clip"],
                device: torch.device | None = None,
                log_epochs: tuple[int, ...] = (),
                n_log: int = 16384) -> dict:
    """Train one neural backbone (canonical recipe unless overridden) and
    return class metrics + predictions + optional loss-component trace."""
    device = device or setup_torch()
    if objective not in ("multitask", "ce_only"):
        raise ValueError(objective)
    seed_all(seed)
    ds_tr = torch.utils.data.TensorDataset(
        torch.from_numpy(np.ascontiguousarray(X_tr, np.float32)),
        torch.from_numpy(np.ascontiguousarray(Yxy_tr, np.float32)),
        torch.from_numpy(np.ascontiguousarray(Yc_tr, np.int64)))
    ds_te = torch.utils.data.TensorDataset(
        torch.from_numpy(np.ascontiguousarray(X_te, np.float32)),
        torch.from_numpy(np.ascontiguousarray(Yxy_te, np.float32)),
        torch.from_numpy(np.ascontiguousarray(Yc_te, np.int64)))
    ld_tr = torch.utils.data.DataLoader(ds_tr, batch_size=batch_size, shuffle=True)
    ld_te = torch.utils.data.DataLoader(ds_te, batch_size=batch_size, shuffle=False)

    ce_w = (torch.tensor(class_weights, dtype=torch.float32, device=device)
            if class_weights is not None else None)
    model = build_model(kind, X_tr.shape[-1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    rng = np.random.default_rng(seed)
    sub = rng.choice(len(X_tr), size=min(n_log, len(X_tr)), replace=False)
    trace = []
    t0 = time.perf_counter()
    for ep in range(1, epochs + 1):
        if objective == "multitask":
            tl = train_epoch(model, ld_tr, opt, device, ce_w, grad_clip)
        else:
            tl = _train_epoch_ce_only(model, ld_tr, opt, device, ce_w, grad_clip)
        if ep in log_epochs or (log_epochs and ep == epochs):
            comp = _loss_components(model, X_tr[sub], Yxy_tr[sub], Yc_tr[sub],
                                    device, ce_w)
            comp.update(epoch=ep, train_loss=tl)
            trace.append(comp)
    train_s = time.perf_counter() - t0
    _m, _pxy, _txy, pred_cls, true_cls = evaluate(model, ld_te, norm_meta, device)
    out = class_metrics(true_cls, pred_cls)
    out.update(train_s=train_s, n_train=int(len(X_tr)),
               steps=int(epochs * int(np.ceil(len(X_tr) / batch_size))),
               loss_trace=trace)
    out["_pred"] = np.asarray(pred_cls, np.int8)
    out["_true"] = np.asarray(true_cls, np.int8)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# XGBoost (hyper-parameters copied verbatim from trainer._eval_xgboost)
# ─────────────────────────────────────────────────────────────────────────────
def fit_eval_xgb(X_tr, Yc_tr, X_te, Yc_te, weighting: str = "none",
                 seed: int = 42) -> dict:
    from sklearn.utils.class_weight import compute_sample_weight
    from xgboost import XGBClassifier
    Xtr = X_tr.reshape(len(X_tr), -1)
    Xte = X_te.reshape(len(X_te), -1)
    clf = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                        objective="multi:softprob", num_class=3,
                        random_state=seed, n_jobs=n_threads(),
                        tree_method="hist", eval_metric="mlogloss")
    sw = None
    if weighting == "balanced":
        sw = compute_sample_weight("balanced", Yc_tr.astype(int))
    elif weighting != "none":
        raise ValueError(weighting)
    t0 = time.perf_counter()
    clf.fit(Xtr, Yc_tr.astype(int), sample_weight=sw)
    train_s = time.perf_counter() - t0
    pred = clf.predict(Xte)
    out = class_metrics(Yc_te, pred)
    out.update(train_s=train_s, n_train=int(len(Xtr)))
    out["_pred"] = np.asarray(pred, np.int8)
    out["_true"] = np.asarray(Yc_te, np.int8)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Fast (vectorised) window builder that reproduces data_loader.make_sequences_split
# `_build` exactly given its fitted norm_meta (used for per-animal test sets)
# ─────────────────────────────────────────────────────────────────────────────
def build_with_meta(df: pd.DataFrame, fcols: list[str], seq_len: int, meta: dict):
    df = df.copy()
    df["behavior_prev"] = df["behavior"] / 2.0
    df["behavior_prev_lag1"] = df["behavior"].shift(1).fillna(0).astype(float) / 2.0
    feats = (df[fcols].to_numpy(np.float32) - meta["mu"]) / meta["sigma"]
    xy = (df[["x", "y"]].to_numpy(np.float32) - meta["xy_min"]) / meta["xy_span"]
    cls = df["behavior"].to_numpy(np.int64)
    n = len(df) - seq_len
    from numpy.lib.stride_tricks import sliding_window_view
    X = sliding_window_view(feats, (seq_len, feats.shape[1]))[:n, 0].astype(np.float32)
    return np.ascontiguousarray(X), xy[seq_len:seq_len + n].astype(np.float32), cls[seq_len:seq_len + n]


# ─────────────────────────────────────────────────────────────────────────────
# I/O
# ─────────────────────────────────────────────────────────────────────────────
def save_result(path: Path, res: dict, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pred = res.pop("_pred", None)
    true = res.pop("_true", None)
    rec = {**meta, **res}
    path.with_suffix(".json").write_text(json.dumps(rec, indent=1, default=float))
    if pred is not None:
        np.savez_compressed(path.with_suffix(".npz"), pred=pred, true=true)


def done(path: Path) -> bool:
    return path.with_suffix(".json").exists()


def env_info() -> dict:
    import sklearn
    import xgboost
    return {
        "host": os.environ.get("SLURMD_NODENAME", os.uname().nodename),
        "job": f"{os.environ.get('SLURM_ARRAY_JOB_ID', os.environ.get('SLURM_JOB_ID', 'local'))}"
               f"_{os.environ.get('SLURM_ARRAY_TASK_ID', '')}",
        "torch": torch.__version__, "xgboost": xgboost.__version__,
        "sklearn": sklearn.__version__, "cuda": torch.cuda.is_available(),
        "threads": n_threads(),
    }
