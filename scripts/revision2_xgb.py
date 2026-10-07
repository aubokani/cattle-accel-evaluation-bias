#!/usr/bin/env python
"""scripts/revision2_xgb.py — XGBoost controls for the 2026-10-07 revision.

XGBoost hyper-parameters are those of trainer._eval_xgboost (300 trees, depth
6, lr 0.05, hist, multi:softprob, seed 42) on the flattened 25 x F window, as
in the headline LOAO. Modes (one task each):

  loao  <h>   held-out animal h, variants:
                none       unweighted, accel-only (reproduces the headline run)
                balanced   sklearn 'balanced' sample weights (inverse class
                           frequency; same weighting principle as the neural
                           weighted CE)                      (R1.M4a, R3.M4a)
                bp         + behavior_prev channel (label at t-1)   (R1.m8)
                bp_lag1    + behavior_prev_lag1 channel (label at t-2)
  loto  <k>   leave-one-trial-out, test = trial k animals; none / balanced
              (R3.m2, R2.M2d); test windows built per animal
  pooled <f>  pooled shuffled stride-1 5-fold CV over all 18 animals
              (StratifiedKFold, seed 42; windows built per animal), fold f;
              unweighted — the "typical published" leaky design vs LOAO (R2.M7)
  day2  <h>   LOAO on the SECOND 24 h of every animal (seconds 86,400–172,799
              after the first accelerometer timestamp); none / balanced (R3.m1)
  build_day2 <a>  build the day-2 1-s feature cache for animal a with the exact
              data_loader feature/label/pseudo-position code (start offset only)

Outputs: Results/revision2_hpc/xgb_controls/<mode><id>_<variant>.{json,npz}
         Results/revision2_hpc/day2_feature_cache/animal-NN_1s_features.csv
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from revision2_common import (OUT_ROOT, ROOT, TRIALS, build_with_meta,  # noqa: E402
                              class_metrics, done, env_info, fit_eval_xgb,
                              save_result)

sys.path.insert(0, str(ROOT / "src"))
import data_loader as dl  # noqa: E402
from data_loader import (ACCEL_FEAT_COLS, DataConfig, load_animals,  # noqa: E402
                         make_sequences_split)

DATA_ROOT = Path("/home/bokania/projects/2026-STA-LSTM-H/data/raw")
DAY2_CACHE = OUT_ROOT / "day2_feature_cache"
OUT = OUT_ROOT / "xgb_controls"
SEQ_LEN = 25
FCOLS = {"none": ACCEL_FEAT_COLS, "balanced": ACCEL_FEAT_COLS,
         "bp": ACCEL_FEAT_COLS + ["behavior_prev"],
         "bp_lag1": ACCEL_FEAT_COLS + ["behavior_prev_lag1"]}


def list_tasks() -> list[str]:
    t = [f"loao {h}" for h in range(1, 19)]
    t += [f"loto {k}" for k in (1, 2, 3)]
    t += [f"pooled {f}" for f in range(5)]
    return t


def build_day2(aid: int) -> None:
    out = DAY2_CACHE / f"animal-{aid:02d}_1s_features.csv"
    if out.exists():
        print(f"day2 cache exists {out}"); return
    cfg = DataConfig(zenodo_root=DATA_ROOT, max_windows=86400)
    acc, hal = dl._accel_path(DATA_ROOT, aid), dl._halter_path(DATA_ROOT, aid)
    start = dl._timestamp_start(acc, cfg.chunk_size) + pd.to_timedelta(86400, unit="s")
    n_bins = 86400
    frame = dl.stream_accel_features(acc, start, n_bins, cfg.chunk_size)
    frame["timestamp"] = (pd.date_range(start=start, periods=n_bins, freq="1s")
                          + pd.to_timedelta(0.5, unit="s"))
    frame["animal_id"] = aid
    frame["behavior"] = dl.stream_halter_labels(hal, start, n_bins, cfg.chunk_size)
    frame["x"], frame["y"] = dl.pseudo_position_from_accel(
        frame["ax_mean"].to_numpy(), frame["ay_mean"].to_numpy(), dt_sec=1)
    n0 = len(frame)
    frame = frame.dropna().reset_index(drop=True)
    DAY2_CACHE.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    print(f"day2 animal {aid:02d}: start={start} kept {len(frame)}/{n0} s "
          f"prevalence={np.bincount(frame['behavior'].astype(int), minlength=3)/len(frame)}")


def _run(tag, X_tr, y_tr, X_te, y_te, variant, meta, te_animal=None):
    path = OUT / tag
    if done(path):
        print(f"[{tag}] done already"); return
    w = "balanced" if variant == "balanced" else "none"
    res = fit_eval_xgb(X_tr, y_tr, X_te, y_te, weighting=w)
    if te_animal is not None:
        res["per_animal"] = {int(a): class_metrics(res["_true"][te_animal == a],
                                                   res["_pred"][te_animal == a])
                             for a in np.unique(te_animal)}
    save_result(path, res, {**meta, "variant": variant, "sample_weighting": w,
                            "n_features": int(np.prod(X_tr.shape[1:])), **env_info()})
    print(f"[{tag}] acc={res['accuracy']:.4f} bal={res['balanced_accuracy']:.4f} "
          f"f1={res['f1_macro']:.4f} fit {res['train_s']:.0f}s", flush=True)


def run(mode: str, sid: int, max_windows: int) -> None:
    if mode == "build_day2":
        build_day2(sid); return
    cache = DAY2_CACHE if mode == "day2" else dl.CACHE_DIR
    full = load_animals(tuple(range(1, 19)),
                        DataConfig(zenodo_root=DATA_ROOT, cache_dir=cache,
                                   max_windows=max_windows))
    if mode in ("loao", "day2", "loto"):
        test_ids = TRIALS[sid] if mode == "loto" else (sid,)
        is_te = full["animal_id"].isin(test_ids)
        d_tr, d_te = full[~is_te].reset_index(drop=True), full[is_te].reset_index(drop=True)
        variants = ["none", "balanced", "bp", "bp_lag1"] if mode == "loao" else ["none", "balanced"]
        built = {}
        for v in variants:
            fc = FCOLS[v]
            key = tuple(fc)
            if key not in built:
                first = d_te[d_te["animal_id"] == test_ids[0]].reset_index(drop=True)
                (X_tr, _, y_tr), (X_te, _, y_te), nm = make_sequences_split(d_tr, first, fc, SEQ_LEN)
                te_a = np.full(len(y_te), test_ids[0], np.int16)
                for a in test_ids[1:]:
                    Xa, _, ya = build_with_meta(d_te[d_te["animal_id"] == a].reset_index(drop=True),
                                                fc, SEQ_LEN, nm)
                    X_te = np.concatenate([X_te, Xa]); y_te = np.concatenate([y_te, ya])
                    te_a = np.concatenate([te_a, np.full(len(ya), a, np.int16)])
                built = {key: (X_tr, y_tr, X_te, y_te, te_a)}
            X_tr, y_tr, X_te, y_te, te_a = built[key]
            _run(f"{mode}{sid:02d}_{v}", X_tr, y_tr, X_te, y_te, v,
                 {"mode": mode, "split_id": sid, "test_animals": list(test_ids)},
                 te_animal=te_a if len(test_ids) > 1 else None)
    elif mode == "pooled":
        from sklearn.model_selection import StratifiedKFold
        Xs, ys, aa = [], [], []
        meta = {"mu": np.zeros(len(ACCEL_FEAT_COLS), np.float32),
                "sigma": np.ones(len(ACCEL_FEAT_COLS), np.float32),
                "xy_min": np.zeros(2, np.float32), "xy_span": np.ones(2, np.float32)}
        for a in range(1, 19):        # trees are invariant to per-feature affine scaling
            Xa, _, ya = build_with_meta(full[full["animal_id"] == a].reset_index(drop=True),
                                        ACCEL_FEAT_COLS, SEQ_LEN, meta)
            Xs.append(Xa); ys.append(ya); aa.append(np.full(len(ya), a, np.int16))
        X, y, an = np.concatenate(Xs), np.concatenate(ys), np.concatenate(aa)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        tr, te = list(skf.split(np.zeros(len(y)), y))[sid]
        _run(f"pooled{sid:02d}_none", X[tr], y[tr], X[te], y[te], "none",
             {"mode": "pooled", "split_id": sid}, te_animal=an[te])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list-tasks", action="store_true")
    ap.add_argument("--task")
    ap.add_argument("--task-file")
    ap.add_argument("--task-index", type=int)
    ap.add_argument("--max-windows", type=int, default=86400)
    ap.add_argument("--out-dir", type=Path, default=None)
    a = ap.parse_args()
    global OUT
    if a.out_dir:
        OUT = a.out_dir
    if a.list_tasks:
        print("\n".join(list_tasks())); return
    if a.task_file is not None:
        a.task = Path(a.task_file).read_text().split("\n")[a.task_index]
    mode, sid = a.task.split()
    t0 = time.perf_counter()
    run(mode, int(sid), a.max_windows)
    print(f"task {a.task} finished in {time.perf_counter()-t0:.0f}s")


if __name__ == "__main__":
    main()
