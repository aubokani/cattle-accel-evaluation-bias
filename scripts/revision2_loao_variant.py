#!/usr/bin/env python
"""scripts/revision2_loao_variant.py — accelerometer-only LOAO / LOTO neural
controls for the 2026-10-07 revision.

Addresses R1.M4 / R3.M4 (class-weighting and auxiliary-objective confounds of
the XGBoost-vs-neural comparison), R1.M5 / R3.M6 (seed replicates), and
R3.m2 / R2.M2(d) / R1.M8 (leave-one-trial-out).

One task = one (split, model, objective, ce_weighting, seed). Everything not
named by the task is the headline LOAO recipe (scripts/run_loao_animal.py →
trainer.run_loao_split): all 18 animals loaded through data_loader.load_animals
(first 86,400 s), split by animal, sequences from data_loader.make_sequences_split
(z-score + xy min–max fitted on the training animals), 25-step windows, Adam
2e-3 / wd 1e-4 / clip 1.2 / batch 64 / 40 epochs, weighted CE weights from
audit_class_balance(data_train). Factors varied, one at a time:

  ce_weighting : weighted (pipeline default) | none        (R1.M4b, R3.M4a)
  objective    : multitask MSE+0.35·CE (default) | ce_only 0.35·CE  (R3.M4a)
  seed         : explicit torch/numpy seed set immediately before model build

For multitask runs the mean MSE and CE on a fixed 16k-window training
subsample are logged at epochs 1/10/20/40 (R1.M4 last bullet: "log and report
the actual magnitudes of the MSE and CE terms").

Splits:
  --held-out N     LOAO: train 17 animals, test animal N (as headline)
  --loto-trial K   leave-one-trial-out: test = all animals of trial K
                   (1: animals 1–6, Jun 2015; 2: 7–10, Sep 2015; 3: 11–18, 2016);
                   test windows are built per test animal with the training
                   normalisation (no cross-animal windows in test).

Outputs: Results/revision2_hpc/loao_controls/<split>_<model>_<objective>_<w>_s<seed>.{json,npz}
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from revision2_common import (OUT_ROOT, ROOT, TRIALS, build_with_meta,  # noqa: E402
                              class_metrics, done, env_info, fit_eval_nn,
                              save_result, setup_torch)

sys.path.insert(0, str(ROOT / "src"))
from data_loader import (ACCEL_FEAT_COLS, DataConfig, audit_class_balance,  # noqa: E402
                         load_animals, make_sequences_split)

DATA_ROOT = Path("/home/bokania/projects/2026-STA-LSTM-H/data/raw")
SEQ_LEN = 25


def list_tasks(which: str) -> list[str]:
    t = []
    if which == "weighting":      # Exp 2: one seed, variants (priority)
        for h in range(1, 19):
            t += [f"loao {h} cnn1d multitask none 1",
                  f"loao {h} lstm multitask none 1",
                  f"loao {h} cnn1d ce_only weighted 1",
                  f"loao {h} cnn1d ce_only none 1",
                  f"loao {h} cnn1d multitask weighted 1",
                  f"loao {h} lstm multitask weighted 1"]
    elif which == "seeds":        # Exp 3a: replicate seeds of the weighted default
        for h in range(1, 19):
            for s in (2, 3):
                t += [f"loao {h} cnn1d multitask weighted {s}",
                      f"loao {h} lstm multitask weighted {s}",
                      f"loao {h} cnn1d multitask none {s}"]
    elif which == "loto":         # Exp 3b
        for k in (1, 2, 3):
            t += [f"loto {k} cnn1d multitask weighted 1",
                  f"loto {k} cnn1d multitask none 1"]
    return t


def run(split: str, sid: int, kind: str, objective: str, weighting: str, seed: int,
        epochs: int, max_windows: int, out_dir: Path) -> None:
    tag = f"{split}{sid:02d}_{kind}_{objective}_{weighting}_s{seed}"
    path = out_dir / tag
    if done(path):
        print(f"[{tag}] already done"); return
    device = setup_torch()
    t0 = time.perf_counter()
    full = load_animals(tuple(range(1, 19)), DataConfig(zenodo_root=DATA_ROOT,
                                                         max_windows=max_windows))
    test_ids = (sid,) if split == "loao" else TRIALS[sid]
    is_test = full["animal_id"].isin(test_ids)
    d_tr = full[~is_test].reset_index(drop=True)
    d_te = full[is_test].reset_index(drop=True)
    fcols = ACCEL_FEAT_COLS
    first_te = d_te[d_te["animal_id"] == test_ids[0]].reset_index(drop=True)
    (X_tr, Yxy_tr, Yc_tr), (X_te, Yxy_te, Yc_te), nm = make_sequences_split(
        d_tr, first_te, fcols, SEQ_LEN)
    te_animal = np.full(len(Yc_te), test_ids[0], np.int16)
    if len(test_ids) > 1:                       # LOTO: per-animal test windows
        parts = [(X_te, Yxy_te, Yc_te, te_animal)]
        for a in test_ids[1:]:
            Xa, Ya, Ca = build_with_meta(d_te[d_te["animal_id"] == a].reset_index(drop=True),
                                         fcols, SEQ_LEN, nm)
            parts.append((Xa, Ya, Ca, np.full(len(Ca), a, np.int16)))
        X_te = np.concatenate([p[0] for p in parts])
        Yxy_te = np.concatenate([p[1] for p in parts])
        Yc_te = np.concatenate([p[2] for p in parts])
        te_animal = np.concatenate([p[3] for p in parts])
    print(f"[{tag}] data ready {time.perf_counter()-t0:.0f}s  train={len(X_tr):,} "
          f"test={len(X_te):,}  device={device}", flush=True)

    _, w = audit_class_balance(d_tr)
    cw = np.asarray(w, np.float32) if weighting == "weighted" else None
    res = fit_eval_nn(kind, X_tr, Yxy_tr, Yc_tr, X_te, Yxy_te, Yc_te, nm, cw,
                      objective=objective, seed=seed, epochs=epochs, device=device,
                      log_epochs=(1, 10, 20, 40) if objective == "multitask" else (1, 40))
    if len(test_ids) > 1:
        res["per_animal"] = {int(a): class_metrics(res["_true"][te_animal == a],
                                                   res["_pred"][te_animal == a])
                             for a in test_ids}
    meta = {"split": split, "split_id": sid, "test_animals": list(test_ids),
            "model": kind, "objective": objective, "ce_weighting": weighting,
            "class_weights": None if cw is None else cw.tolist(), "seed": seed,
            "epochs": epochs, "batch_size": 64, **env_info()}
    save_result(path, res, meta)
    if len(test_ids) > 1:
        np.save(path.with_name(path.name + "_animal.npy"), te_animal)
    print(f"[{tag}] acc={res['accuracy']:.4f} bal={res['balanced_accuracy']:.4f} "
          f"f1={res['f1_macro']:.4f} train {res['train_s']/60:.1f} min", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list-tasks", choices=["weighting", "seeds", "loto"])
    ap.add_argument("--task", help='"loao|loto <id> <model> <objective> <weighting> <seed>"')
    ap.add_argument("--task-file")
    ap.add_argument("--task-index", type=int)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--max-windows", type=int, default=86400)
    ap.add_argument("--out-dir", type=Path, default=OUT_ROOT / "loao_controls")
    a = ap.parse_args()
    if a.list_tasks:
        print("\n".join(list_tasks(a.list_tasks))); return
    if a.task_file is not None:
        a.task = Path(a.task_file).read_text().split("\n")[a.task_index]
    split, sid, kind, obj, w, seed = a.task.split()
    run(split, int(sid), kind, obj, w, int(seed), a.epochs, a.max_windows, a.out_dir)


if __name__ == "__main__":
    main()
