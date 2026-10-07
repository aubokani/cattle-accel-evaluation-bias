#!/usr/bin/env python
"""Second-round (Scientific Reports, review 2026-10-07) re-analysis from existing per-animal LOAO predictions, the 1-s feature cache and raw halter labels -> Results/revision2/*.csv + SUMMARY.md (deterministic; no neural training; only cheap sklearn no-context floors are fitted)."""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from functools import lru_cache
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import median_filter, uniform_filter1d

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "Results"
OUT = R / "revision2"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = R / "python_pipeline" / "feature_cache"
RAW = ROOT / "data" / "raw"
MS_TEX = ROOT / "manuscript" / "2026_Scientific_Reports" / "main.tex"

ANIMALS = list(range(1, 19))
TRIAL = {a: (1 if a <= 6 else 2 if a <= 10 else 3) for a in ANIMALS}
TRIAL_NAME = {1: "T1 Jun-2015", 2: "T2 Sep-2015", 3: "T3 Aug-Oct-2016"}
SEQ = 25
CLASSES = ["Other", "Ruminating", "Eating"]
FEATS = ["ax_mean", "ay_mean", "az_mean", "ax_std", "ay_std", "az_std",
         "ax_min", "ay_min", "az_min", "ax_max", "ay_max", "az_max",
         "mag_mean", "mag_std"]
NEURAL6 = ["STA-LSTM-H", "STA-LSTM", "LSTM", "GRU", "Transformer", "1D-CNN"]
ACCEL7 = ["XGBoost", "1D-CNN", "Transformer", "STA-LSTM-H", "GRU", "LSTM", "STA-LSTM"]
RNG_SEED = 20261007

SUMMARY: dict[str, list[str]] = {}


def note(key: str, *lines: str) -> None:
    SUMMARY.setdefault(key, []).extend(lines)


def save(df: pd.DataFrame, name: str, index: bool = False) -> str:
    df.to_csv(OUT / name, index=index)
    return f"`Results/revision2/{name}`"


# ════════════════════════════════════════════════════════════════════════════
# Statistics helpers
# ════════════════════════════════════════════════════════════════════════════
@lru_cache(maxsize=None)
def _bits(n: int) -> np.ndarray:
    idx = np.arange(2 ** n, dtype=np.uint32)
    return ((idx[:, None] >> np.arange(n, dtype=np.uint32)) & 1).astype(np.float64)


def signflip_wilcoxon(d, alternative="two-sided", tol=1e-12) -> dict:
    """Exact conditional (permutation) Wilcoxon signed-rank test.

    Zeros are dropped (Wilcoxon zero_method), |d| ranked with mid-ranks (valid
    under ties because the null distribution is enumerated over all 2^n sign
    patterns of the *observed* mid-ranks). Exact for n <= 20.
    """
    d = np.asarray(d, float)
    nz = d[np.abs(d) > tol]
    n_zero = int(len(d) - len(nz))
    n = len(nz)
    out = dict(n=len(d), n_zero=n_zero, n_eff=n)
    if n == 0:
        out.update(W_plus=0.0, p=1.0, r_rb=0.0, n_tie_groups=0)
        return out
    absd = np.abs(nz)
    ranks = stats.rankdata(absd)
    _, cnt = np.unique(np.round(absd, 12), return_counts=True)
    out["n_tie_groups"] = int((cnt > 1).sum())
    w_plus = float(ranks[nz > 0].sum())
    null = _bits(n) @ ranks
    eps = 1e-9
    if alternative == "two-sided":
        e = ranks.sum() / 2
        p = float(np.mean(np.abs(null - e) >= abs(w_plus - e) - eps))
    elif alternative == "greater":
        p = float(np.mean(null >= w_plus - eps))
    else:
        p = float(np.mean(null <= w_plus + eps))
    out.update(W_plus=w_plus, p=p,
               r_rb=float((w_plus - (ranks.sum() - w_plus)) / ranks.sum()))
    return out


def scipy_auto_p(a, b) -> float:
    """What scipy.stats.wilcoxon(mode/method='auto') returns (as revision_stats.py did)."""
    try:
        return float(stats.wilcoxon(a, b, zero_method="wilcox", correction=False,
                                    alternative="two-sided", method="auto").pvalue)
    except Exception:
        return np.nan


def holm(p) -> np.ndarray:
    p = np.asarray(p, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for k, i in enumerate(order):
        run = max(run, (m - k) * p[i])
        adj[i] = min(run, 1.0)
    return adj


def tci(x, level=0.95):
    x = np.asarray(x, float)
    n = len(x)
    m = x.mean()
    h = stats.t.ppf(0.5 + level / 2, n - 1) * x.std(ddof=1) / np.sqrt(n)
    return m, m - h, m + h


def tost_t(d, margin) -> float:
    d = np.asarray(d, float)
    n = len(d)
    se = max(d.std(ddof=1) / np.sqrt(n), 1e-12)
    p_lo = stats.t.sf((d.mean() + margin) / se, n - 1)
    p_hi = stats.t.cdf((d.mean() - margin) / se, n - 1)
    return float(max(p_lo, p_hi))


def tost_wilcoxon(d, margin) -> float:
    d = np.asarray(d, float)
    p_lo = signflip_wilcoxon(d + margin, "greater")["p"]
    p_hi = signflip_wilcoxon(d - margin, "less")["p"]
    return float(max(p_lo, p_hi))


def hodges_lehmann(d) -> float:
    d = np.asarray(d, float)
    i, j = np.triu_indices(len(d))
    return float(np.median((d[i] + d[j]) / 2))


def boot_ci(d, level=0.90, B=10000, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    d = np.asarray(d, float)
    bs = d[rng.integers(0, len(d), (B, len(d)))].mean(1)
    a = (1 - level) / 2
    return float(np.quantile(bs, a)), float(np.quantile(bs, 1 - a))


def mde(sd, n=18, alpha=0.05, power=0.8) -> float:
    df = n - 1
    return float((stats.t.ppf(1 - alpha / 2, df) + stats.t.ppf(power, df)) * sd / np.sqrt(n))


def ccc(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    sxy = np.mean((x - x.mean()) * (y - y.mean()))
    return float(2 * sxy / (x.var() + y.var() + (x.mean() - y.mean()) ** 2))


def cls_metrics(y, p) -> dict:
    cm = np.zeros((3, 3), np.int64)
    np.add.at(cm, (y, p), 1)
    tp = np.diag(cm).astype(float)
    rec = tp / np.maximum(cm.sum(1), 1)
    prec = tp / np.maximum(cm.sum(0), 1)
    f1 = np.where(2 * tp + (cm.sum(0) - tp) + (cm.sum(1) - tp) > 0,
                  2 * tp / np.maximum(2 * tp + (cm.sum(0) - tp) + (cm.sum(1) - tp), 1), 0)
    n = cm.sum()
    po = tp.sum() / n
    pe = (cm.sum(0) * cm.sum(1)).sum() / n ** 2
    out = dict(accuracy=po, balanced_accuracy=rec.mean(), macro_f1=f1.mean(),
               kappa=(po - pe) / (1 - pe))
    for c, nm in enumerate(CLASSES):
        out[f"recall_{nm}"] = rec[c]
        out[f"precision_{nm}"] = prec[c]
        out[f"f1_{nm}"] = f1[c]
    return out, cm


def fmt_p(p):
    return f"{p:.2g}" if p < 0.001 else f"{p:.3f}"


# ════════════════════════════════════════════════════════════════════════════
# Data loading
# ════════════════════════════════════════════════════════════════════════════
def load_all():
    print("loading feature cache ...")
    lab, feat, ts0 = {}, {}, {}
    for a in ANIMALS:
        df = pd.read_csv(CACHE / f"animal-{a:02d}_1s_features.csv",
                         usecols=FEATS + ["behavior", "timestamp"])
        lab[a] = df["behavior"].to_numpy(np.int64)
        feat[a] = df[FEATS].to_numpy(np.float64)
        ts0[a] = str(df["timestamp"].iloc[0])
    print("loading LOAO predictions ...")
    P = {"accel": {m: {} for m in ACCEL7}, "bp": {m: {} for m in NEURAL6},
         "lag1": {m: {} for m in NEURAL6}}
    T = {}
    for a in ANIMALS:
        h = np.load(R / f"python_pipeline_loao/animal-{a:02d}/loao_last_predictions.npy",
                    allow_pickle=True).item()
        b = np.load(R / f"python_pipeline_loao_ablation/animal-{a:02d}/loao_last_predictions.npy",
                    allow_pickle=True).item()
        src = {}
        for m in ["LSTM", "GRU", "Transformer", "1D-CNN", "XGBoost"]:
            src[("accel", m)] = h[m]
        src[("accel", "STA-LSTM-H")] = b["STA-LSTM-H (accel)"]
        src[("accel", "STA-LSTM")] = b["STA-LSTM (accel)"]
        src[("bp", "STA-LSTM-H")] = h["STA-LSTM-H"]
        src[("bp", "STA-LSTM")] = h["STA-LSTM"]
        for m in ["LSTM", "GRU", "Transformer", "1D-CNN"]:
            src[("bp", m)] = b[f"{m} + bp"]
        for m in NEURAL6:
            src[("lag1", m)] = b[f"{m} + bp_lag1"]
        truth = lab[a][SEQ:]
        for (inp, m), v in src.items():
            t = np.asarray(v["true_cls"], np.int64)
            assert np.array_equal(t, truth), f"label misalignment animal {a} {inp} {m}"
            P[inp][m][a] = np.asarray(v["pred_cls"], np.int64)
        T[a] = truth
    return lab, feat, ts0, P, T


def read_readme():
    txt = (RAW / "Readme.md").read_text()
    rows = []
    for line in txt.splitlines():
        m = re.match(r"\|\s*(\d+)\s*\|\s*(\d)\s*\|\s*(Yes|No)\s*\|.*?\|\s*([\d\- :]+?)\s*\|\s*([\d\- :]+?)\s*\|.*\|\s*(\d+)\s*\|\s*$", line)
        if m:
            rows.append(dict(animal=int(m.group(1)), trial=int(m.group(2)),
                             publisher_test=m.group(3) == "Yes",
                             start=pd.Timestamp(m.group(4)), end=pd.Timestamp(m.group(5)),
                             drinking_samples=int(m.group(6))))
    return pd.DataFrame(rows).set_index("animal")


def runs(x):
    x = np.asarray(x)
    idx = np.flatnonzero(np.diff(x)) + 1
    starts = np.r_[0, idx]
    lens = np.diff(np.r_[starts, len(x)])
    return x[starts], lens, starts


# ════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-floors", action="store_true",
                    help="reuse Results/revision2/loao_trivial_floors_per_animal.csv")
    args = ap.parse_args()
    t_start = time.time()
    lab, feat, ts0, P, T = load_all()
    readme = read_readme()
    assert len(readme) == 18

    # per-animal accuracy matrices --------------------------------------------
    met_rows, cms = [], {}
    for inp in ["accel", "bp", "lag1"]:
        for m, d in P[inp].items():
            for a in ANIMALS:
                mt, cm = cls_metrics(T[a], d[a])
                cms[(inp, m, a)] = cm
                met_rows.append(dict(input=inp, model=m, animal=a, trial=TRIAL[a], **mt))
    MET = pd.DataFrame(met_rows)
    ACC = MET[MET.input == "accel"].pivot(index="animal", columns="model", values="accuracy")[ACCEL7]

    # ════════════════════════════════════════════════════════════════════════
    # 1. 10-s LABEL GRID  (R1.M1, R2.M1, R3.M3)
    # ════════════════════════════════════════════════════════════════════════
    print("[1] label grid ...")
    grid_rows, epoch_seqs = [], {}
    for a in ANIMALS:
        c = pd.read_csv(RAW / f"halter-{a:02d}.csv", usecols=["classification"],
                        nrows=864000)["classification"].to_numpy(np.int64)
        vals, lens, starts = runs(c)
        inner = lens[1:-1]
        chg = starts[1:]
        res = np.bincount(chg % 100, minlength=100)
        res_sorted = np.sort(res)[::-1]
        n_res95 = int(np.searchsorted(np.cumsum(res_sorted) / res.sum(), 0.95) + 1)
        k = np.rint(lens / 100).astype(int)
        eseq = np.repeat(vals, k)
        epoch_seqs[a] = eseq
        s = lab[a]
        v1, l1, _ = runs(s)
        in1 = l1[1:-1]
        y, yp, yp2 = s[SEQ:], s[SEQ - 1:-1], s[SEQ - 2:-2]
        grid_rows.append(dict(
            animal=a, trial=TRIAL[a],
            raw_runs_interior=len(inner),
            raw_frac_runs_multiple_of_100_samples=float(np.mean(inner % 100 == 0)),
            raw_min_interior_run_samples=int(inner.min()),
            raw_change_residues_mod100_for_95pct=n_res95,
            raw_top_residue_share=float(res_sorted[0] / res.sum()),
            sec_runs_interior=len(in1),
            sec_frac_runs_multiple_of_10s=float(np.mean(in1 % 10 == 0)),
            sec_frac_runs_9_or_1_mod_10=float(np.mean(np.isin(in1 % 10, [1, 9]))),
            sec_min_interior_run_s=int(in1.min()),
            sec_frac_runs_lt10s=float(np.mean(in1 < 10)),
            predict_prev_1s=float(np.mean(y == yp)),
            predict_two_back_1s=float(np.mean(y == yp2)),
            lag10_persistence_1s=float(np.mean(s[10:] == s[:-10])),
            n_epochs_10s=len(eseq),
            predict_prev_10s_epoch=float(np.mean(eseq[1:] == eseq[:-1])),
            n_runs_per_day=len(v1), n_transitions_per_day=len(v1) - 1,
            eat_to_rum_switches=int(np.sum((s[:-1] == 2) & (s[1:] == 1))),
            rum_to_eat_switches=int(np.sum((s[:-1] == 1) & (s[1:] == 2))),
            **{f"median_run_s_{nm}": float(np.median(l1[v1 == ci])) for ci, nm in enumerate(CLASSES)},
            **{f"n_runs_{nm}": int(np.sum(v1 == ci)) for ci, nm in enumerate(CLASSES)},
        ))
    G = pd.DataFrame(grid_rows)
    G["er_direct_switches_per_day"] = G.eat_to_rum_switches + G.rum_to_eat_switches
    src_grid = save(G, "label_grid_per_animal.csv")
    pooled_runs = np.concatenate([runs(lab[a])[1][1:-1] for a in ANIMALS])
    pooled_vals = np.concatenate([runs(lab[a])[0][1:-1] for a in ANIMALS])
    hist = pd.DataFrame({"run_length_s": np.arange(1, 601)})
    hist["count_all"] = [int(np.sum(pooled_runs == L)) for L in hist.run_length_s]
    for ci, nm in enumerate(CLASSES):
        hist[f"count_{nm}"] = [int(np.sum((pooled_runs == L) & (pooled_vals == ci))) for L in hist.run_length_s]
    save(hist, "run_length_histogram_1s.csv")
    rl = pd.DataFrame({"run_length_s": pooled_runs, "class": pooled_vals})
    save(rl, "run_lengths_interior_1s.csv")

    # lag-k persistence pooled + class-specific
    lags = np.unique(np.r_[np.arange(1, 601),
                           np.round(np.logspace(np.log10(600), np.log10(6 * 3600), 60)).astype(int)])
    lag_rows = []
    for k in lags:
        num = den = 0
        nc = np.zeros(3); dc = np.zeros(3)
        for a in ANIMALS:
            s = lab[a]
            eq = s[:-k] == s[k:]
            num += eq.sum(); den += len(eq)
            for ci in range(3):
                msk = s[:-k] == ci
                nc[ci] += (eq & msk).sum(); dc[ci] += msk.sum()
        lag_rows.append(dict(lag_s=int(k), p_same_pooled=num / den,
                             **{f"p_same_given_{nm}": nc[ci] / dc[ci] for ci, nm in enumerate(CLASSES)}))
    LAG = pd.DataFrame(lag_rows)
    src_lag = save(LAG, "lag_persistence_curve.csv")
    lagv = LAG.set_index("lag_s")
    pooled_ep = np.mean(np.concatenate([e[1:] == e[:-1] for e in epoch_seqs.values()]))

    # bout criterion (merge same-class runs separated by < g seconds of anything else)
    bout_rows = []
    for a in ANIMALS:
        s = lab[a]
        for ci, nm in [(1, "Ruminating"), (2, "Eating")]:
            on = (s == ci).astype(int)
            v, L, st = runs(on)
            for g in [0, 60, 300, 600]:
                on_runs = [(st[i], L[i]) for i in range(len(v)) if v[i] == 1]
                merged = []
                for s0, l0 in on_runs:
                    if merged and s0 - (merged[-1][0] + merged[-1][1]) < g:
                        merged[-1] = (merged[-1][0], s0 + l0 - merged[-1][0])
                    else:
                        merged.append((s0, l0))
                durs = np.array([m[1] for m in merged]) / 60
                bout_rows.append(dict(animal=a, trial=TRIAL[a], behaviour=nm, gap_criterion_s=g,
                                      n_bouts_per_day=len(merged),
                                      median_bout_min=float(np.median(durs)),
                                      mean_bout_min=float(durs.mean())))
    BOUT = pd.DataFrame(bout_rows)
    src_bout = save(BOUT, "bout_criterion_counts.csv")
    bsum = BOUT.groupby(["behaviour", "gap_criterion_s"])[["n_bouts_per_day", "median_bout_min"]].agg(["median", "min", "max"]).round(1)
    save(bsum.reset_index(), "bout_criterion_summary.csv")

    # effective sample size of the per-second correctness indicator
    def ess(x, kmax=20000):
        x = np.asarray(x, float) - np.mean(x)
        n = len(x)
        if x.var() == 0:
            return float(n)
        f = np.fft.rfft(x, 2 * n)
        ac = np.fft.irfft(f * np.conj(f))[:kmax + 1]
        ac = ac / ac[0]
        neg = np.flatnonzero(ac[1:] <= 0)
        K = neg[0] if len(neg) else kmax
        return float(n / (1 + 2 * ac[1:K + 1].sum()))
    ess_rows = []
    for m in ACCEL7:
        for a in ANIMALS:
            ess_rows.append(dict(model=m, animal=a, n=len(T[a]),
                                 ess_correct=ess(P["accel"][m][a] == T[a])))
    ESS = pd.DataFrame(ess_rows)
    save(ESS, "effective_sample_size_correctness.csv")
    ess_s = ESS.groupby("model").ess_correct.agg(["median", "min", "max"]).round(0)

    # epoch-level re-scoring of predictions (one score per 10-s label epoch).
    # The grid phase is not constant over the day, so epochs are aligned to the
    # label runs themselves: each 1-s label run of length L inside the target
    # range is split into max(1, round(L/10)) consecutive, near-equal chunks.
    ep_rows = []
    for a in ANIMALS:
        y = T[a]
        v, L, st = runs(y)
        eid = np.empty(len(y), np.int64)
        e0 = 0
        for s0, l0 in zip(st, L):
            k_ = max(1, int(round(l0 / 10)))
            eid[s0:s0 + l0] = e0 + (np.arange(l0) * k_) // l0
            e0 += k_
        ne = e0
        sizes = np.bincount(eid, minlength=ne)
        tcount = np.zeros((ne, 3)); np.add.at(tcount, (eid, y), 1)
        t_ep = tcount.argmax(1)
        for inp, models in [("accel", ACCEL7), ("bp", NEURAL6), ("lag1", NEURAL6)]:
            for m in models:
                pc = np.zeros((ne, 3)); np.add.at(pc, (eid, P[inp][m][a]), 1)
                mt, _ = cls_metrics(t_ep, pc.argmax(1))
                ep_rows.append(dict(input=inp, model=m, animal=a, n_epochs=int(ne),
                                    epoch_size_median=float(np.median(sizes)),
                                    frac_epochs_size_9_to_11=float(np.mean((sizes >= 9) & (sizes <= 11))),
                                    epoch_accuracy=mt["accuracy"],
                                    epoch_balanced_accuracy=mt["balanced_accuracy"],
                                    epoch_macro_f1=mt["macro_f1"]))
        ep_rows.append(dict(input="baseline", model="predict_prev_epoch", animal=a, n_epochs=int(ne),
                            epoch_size_median=float(np.median(sizes)),
                            frac_epochs_size_9_to_11=float(np.mean((sizes >= 9) & (sizes <= 11))),
                            epoch_accuracy=float(np.mean(t_ep[1:] == t_ep[:-1])),
                            epoch_balanced_accuracy=np.nan, epoch_macro_f1=np.nan))
    EP = pd.DataFrame(ep_rows)
    src_ep = save(EP, "epoch10s_rescoring_per_animal.csv")
    ep_acc = EP[EP.input == "accel"].groupby("model")[["epoch_accuracy", "epoch_balanced_accuracy", "epoch_macro_f1"]].mean()
    one_s = MET[MET.input == "accel"].groupby("model")[["accuracy", "balanced_accuracy", "macro_f1"]].mean()
    epc = ep_acc.join(one_s).loc[ACCEL7]
    epc["delta_acc_pp"] = (epc.epoch_accuracy - epc.accuracy) * 100
    save(epc.reset_index(), "epoch10s_rescoring_summary.csv")
    ep_ppe = EP[EP.model == "predict_prev_epoch"].epoch_accuracy

    key = "R1.M1 / R2.M1 / R3.M3 — 10-s label grid"
    note(key,
         f"Source: {src_grid}, {src_lag}, `run_length_histogram_1s.csv`, {src_bout}, {src_ep}; raw `data/raw/halter-NN.csv` first 864,000 rows (24 h at 10 Hz) and the 1-s cache labels used by every model.",
         f"- Raw 10 Hz halter: interior label runs that are exact multiples of 100 samples (10 s): per-animal range {G.raw_frac_runs_multiple_of_100_samples.min():.3f}–{G.raw_frac_runs_multiple_of_100_samples.max():.3f} (median {G.raw_frac_runs_multiple_of_100_samples.median():.3f}); minimum interior run {G.raw_min_interior_run_samples.min()}–{G.raw_min_interior_run_samples.max()} samples; change positions need {G.raw_change_residues_mod100_for_95pct.min()}–{G.raw_change_residues_mod100_for_95pct.max()} residues (mod 100 samples) to cover 95% of changes (top residue holds {G.raw_top_residue_share.min():.2f}–{G.raw_top_residue_share.max():.2f}), i.e. the phase of the 10-s grid relative to the file start is not constant across the day (run lengths, not absolute positions, are quantised).",
         f"- 1-s labels (cache): interior runs that are multiples of 10 s: {G.sec_frac_runs_multiple_of_10s.min():.3f}–{G.sec_frac_runs_multiple_of_10s.max():.3f} per animal (pooled {np.mean(pooled_runs % 10 == 0):.3f}); runs ≡ 1 or 9 (mod 10) s (1-s binning artefact): pooled {np.mean(np.isin(pooled_runs % 10, [1, 9])):.3f}; runs < 10 s: pooled {np.mean(pooled_runs < 10):.4f}; minimum interior run {G.sec_min_interior_run_s.min()} s.",
         f"- Pooled lag-k same-label probability (all 18 animals, first 24 h): k=1 {lagv.p_same_pooled[1]:.4f}, k=2 {lagv.p_same_pooled[2]:.4f}, k=5 {lagv.p_same_pooled[5]:.4f}, k=10 {lagv.p_same_pooled[10]:.4f}, k=11 {lagv.p_same_pooled[11]:.4f}, k=20 {lagv.p_same_pooled[20]:.4f}, k=25 {lagv.p_same_pooled[25]:.4f}, k=60 {lagv.p_same_pooled[60]:.4f}, k=300 {lagv.p_same_pooled[300]:.4f}, k=600 {lagv.p_same_pooled[600]:.4f}. Decline from k=1 to k=10 is linear (≈{(lagv.p_same_pooled[1]-lagv.p_same_pooled[10])/9*100:.2f} pp per second), the signature of upsampled 10-s epochs.",
         f"- Class-specific P(same after k | class at t), k = 1 / 10 / 60 / 600 s: " + "; ".join(
             f"{nm} {lagv[f'p_same_given_{nm}'][1]:.4f} / {lagv[f'p_same_given_{nm}'][10]:.4f} / {lagv[f'p_same_given_{nm}'][60]:.4f} / {lagv[f'p_same_given_{nm}'][600]:.4f}" for nm in CLASSES) + ".",
         f"- Predict-previous accuracy on the LOAO targets (t = 25…86,399): 1-s resolution mean {G.predict_prev_1s.mean():.4f} (range {G.predict_prev_1s.min():.4f}–{G.predict_prev_1s.max():.4f}); predict-two-back (lag-2) {G.predict_two_back_1s.mean():.4f} (range {G.predict_two_back_1s.min():.4f}–{G.predict_two_back_1s.max():.4f}).",
         f"- Predict-previous at native 10-s epoch resolution (raw runs quantised to round(len/100) epochs): per-animal mean {G.predict_prev_10s_epoch.mean():.4f} (range {G.predict_prev_10s_epoch.min():.4f}–{G.predict_prev_10s_epoch.max():.4f}; pooled {pooled_ep:.4f}). Same quantity on run-aligned epochs of the LOAO target seconds: mean {ep_ppe.mean():.4f}. Check: 1 − (1 − {G.predict_prev_10s_epoch.mean():.4f})/10 = {1-(1-G.predict_prev_10s_epoch.mean())/10:.4f} vs observed 1-s {G.predict_prev_1s.mean():.4f}.",
         f"- Effective number of independent labels per animal-day: 10-s epochs {int(G.n_epochs_10s.median())} (range {G.n_epochs_10s.min()}–{G.n_epochs_10s.max()}) vs 86,400 one-second windows; label runs per day median {int(G.n_runs_per_day.median())} (range {G.n_runs_per_day.min()}–{G.n_runs_per_day.max()}), i.e. ≈{int(G.n_transitions_per_day.median())} transitions per animal-day.",
         f"- Effective sample size of the per-second correctness indicator (FFT autocorrelation summed to first non-positive lag) per animal (n = 86,375): " + "; ".join(f"{m} median {int(ess_s.loc[m,'median'])} [{int(ess_s.loc[m,'min'])}–{int(ess_s.loc[m,'max'])}]" for m in ACCEL7) + ".",
         f"- Median run (\"bout\") length by class, pooled 1-s runs: " + ", ".join(f"{nm} {np.median(pooled_runs[pooled_vals==ci]):.0f} s" for ci, nm in enumerate(CLASSES)) + f"; per-animal medians: " + ", ".join(f"{nm} {G[f'median_run_s_{nm}'].min():.0f}–{G[f'median_run_s_{nm}'].max():.0f} s" for nm in CLASSES) + ". Pooled all-class median (interior runs) = " + f"{np.median(pooled_runs):.0f} s.",
         f"- Runs per animal-day: Ruminating median {int(G.n_runs_Ruminating.median())} ({G.n_runs_Ruminating.min()}–{G.n_runs_Ruminating.max()}), Eating {int(G.n_runs_Eating.median())} ({G.n_runs_Eating.min()}–{G.n_runs_Eating.max()}), Other {int(G.n_runs_Other.median())} ({G.n_runs_Other.min()}–{G.n_runs_Other.max()}).",
         f"- Direct Eating↔Ruminating switches per animal-day: median {int(G.er_direct_switches_per_day.median())} (range {G.er_direct_switches_per_day.min()}–{G.er_direct_switches_per_day.max()}; mean {G.er_direct_switches_per_day.mean():.0f}); cohort totals Eating→Ruminating {G.eat_to_rum_switches.sum()}, Ruminating→Eating {G.rum_to_eat_switches.sum()}.",
         "- Bout criterion sensitivity (merge same-class runs separated by < g s; median [min–max] across animals of bouts/day): " + "; ".join(
             f"{b} g={g}s: {int(BOUT[(BOUT.behaviour==b)&(BOUT.gap_criterion_s==g)].n_bouts_per_day.median())} [{BOUT[(BOUT.behaviour==b)&(BOUT.gap_criterion_s==g)].n_bouts_per_day.min()}–{BOUT[(BOUT.behaviour==b)&(BOUT.gap_criterion_s==g)].n_bouts_per_day.max()}], median bout {BOUT[(BOUT.behaviour==b)&(BOUT.gap_criterion_s==g)].median_bout_min.median():.1f} min"
             for b in ["Ruminating", "Eating"] for g in [0, 60, 300, 600]) + ". The criterion value must be justified from the literature by the author; values are reported, not chosen.",
         "- Epoch-level re-scoring (one score per 10-s label epoch; epochs aligned to label runs because the grid phase relative to file start is not constant — each run of L s is split into round(L/10) chunks; epoch prediction = majority of its per-second predictions) of accel-only LOAO predictions, mean over 18 animals, epoch accuracy / epoch balanced accuracy [1-s accuracy]: " + "; ".join(
             f"{m} {epc.loc[m,'epoch_accuracy']:.4f} / {epc.loc[m,'epoch_balanced_accuracy']:.4f} [{epc.loc[m,'accuracy']:.4f}]" for m in ACCEL7) + f". Epochs per animal {EP.n_epochs.min()}–{EP.n_epochs.max()}; {EP.frac_epochs_size_9_to_11.min():.3f}–{EP.frac_epochs_size_9_to_11.max():.3f} of epochs are 9–11 s long. Epoch scoring changes accuracy by ≤{epc.delta_acc_pp.abs().max():.2f} pp and leaves the model ordering " + ("unchanged" if list(epc.sort_values('epoch_accuracy',ascending=False).index)==list(epc.sort_values('accuracy',ascending=False).index) else "CHANGED: " + " > ".join(epc.sort_values('epoch_accuracy',ascending=False).index)) + ".",
         )

    # ════════════════════════════════════════════════════════════════════════
    # 2. PERSISTENCE TESTS for all six neural models (R1.M6, R3.M7)
    # ════════════════════════════════════════════════════════════════════════
    print("[2] persistence tests ...")
    pr_rows, per_rows = [], []
    for inp, lag in [("bp", 1), ("lag1", 2)]:
        for m in NEURAL6:
            acc_m, acc_b, agree, tr_acc, nt_acc, n_tr = [], [], [], [], [], []
            for a in ANIMALS:
                s = lab[a]
                y = T[a]
                base = s[SEQ - lag:len(s) - lag]
                p = P[inp][m][a]
                acc_m.append(np.mean(p == y)); acc_b.append(np.mean(base == y))
                agree.append(np.mean(p == base))
                trm = y != base
                tr_acc.append(np.mean(p[trm] == y[trm])); nt_acc.append(np.mean(p[~trm] == y[~trm]))
                n_tr.append(int(trm.sum()))
                per_rows.append(dict(input=inp, model=m, animal=a, acc_model=acc_m[-1],
                                     acc_baseline=acc_b[-1], n_correct_model=int(np.sum(p == y)),
                                     n_correct_baseline=int(np.sum(base == y)),
                                     agreement_with_baseline=agree[-1],
                                     transition_accuracy=tr_acc[-1], non_transition_accuracy=nt_acc[-1],
                                     n_transitions=n_tr[-1]))
            acc_m, acc_b = np.array(acc_m), np.array(acc_b)
            cm_ = np.array([r["n_correct_model"] for r in per_rows[-18:]])
            cb_ = np.array([r["n_correct_baseline"] for r in per_rows[-18:]])
            d = acc_m - acc_b
            w2 = signflip_wilcoxon(d)
            wg = signflip_wilcoxon(d, "greater")
            pr_rows.append(dict(
                input=inp, model=m,
                baseline="predict_prev (t-1)" if lag == 1 else "predict_two_back (t-2)",
                mean_model=acc_m.mean(), mean_baseline=acc_b.mean(),
                mean_diff_pp=d.mean() * 100,
                n_above=int((cm_ > cb_).sum()), n_equal=int((cm_ == cb_).sum()), n_below=int((cm_ < cb_).sum()),
                n_eff=w2["n_eff"], n_tie_groups_abs_diff=w2["n_tie_groups"],
                p_exact_two_sided=w2["p"], p_scipy_auto_two_sided=scipy_auto_p(acc_m, acc_b),
                p_exact_one_sided_model_greater=wg["p"], r_rb=w2["r_rb"],
                agreement_with_baseline_mean=float(np.mean(agree)),
                agreement_with_baseline_min=float(np.min(agree)),
                transition_accuracy_mean=float(np.mean(tr_acc)),
                transition_accuracy_max=float(np.max(tr_acc)),
                non_transition_accuracy_mean=float(np.mean(nt_acc)),
                transitions_per_animal_median=float(np.median(n_tr))))
    PR = pd.DataFrame(pr_rows)
    for inp in ["bp", "lag1"]:
        msk = PR.input == inp
        PR.loc[msk, "p_holm_two_sided"] = holm(PR.loc[msk, "p_exact_two_sided"])
        PR.loc[msk, "p_holm_one_sided"] = holm(PR.loc[msk, "p_exact_one_sided_model_greater"])
    src_pr = save(PR, "persistence_tests_all_models.csv")
    save(pd.DataFrame(per_rows), "persistence_per_animal.csv")
    # accel-only transition accuracy for reference
    acc_tr = []
    for m in ACCEL7:
        v = [np.mean(P["accel"][m][a][T[a] != lab[a][SEQ - 1:-1]] == T[a][T[a] != lab[a][SEQ - 1:-1]]) for a in ANIMALS]
        acc_tr.append((m, np.mean(v)))
    # reconcile the two text p-values
    hb = PR[(PR.input == "bp") & PR.model.isin(["STA-LSTM-H", "STA-LSTM"])].set_index("model")
    old = pd.read_csv(R / "revision" / "persistence_vs_predictprev.csv")
    tied_animals = [r["animal"] for r in per_rows if r["input"] == "bp" and r["model"] == "STA-LSTM-H" and r["n_correct_model"] == r["n_correct_baseline"]]
    # also against the StratifiedKFold-averaged predict_prev file that the old script used
    ppc = pd.read_csv(R / "predict_prev_cohort.csv").set_index("animal").accuracy
    key = "R1.M6 / R3.M7 / R3.m13 — persistence tests, all six neural models"
    note(key, f"Source: {src_pr}, `persistence_per_animal.csv`. Baseline computed on exactly the same 86,375 LOAO target seconds as the model (y_t vs y_(t-1) for +bp; vs y_(t-2) for +bp_lag1). Exact p = conditional sign-flip enumeration over all 2^n_eff sign patterns of the observed mid-ranks (valid with tied |d|); zeros dropped.")
    for _, r in PR.iterrows():
        note(key, f"- {r.model} +{'bp' if r.input=='bp' else 'bp_lag1'} vs {r.baseline}: model {r.mean_model:.4f} vs baseline {r.mean_baseline:.4f} (Δ {r.mean_diff_pp:+.3f} pp); above/equal/below = {r.n_above}/{r.n_equal}/{r.n_below}; n_eff {r.n_eff}, tied-|d| groups {r.n_tie_groups_abs_diff}; exact two-sided p = {fmt_p(r.p_exact_two_sided)} (Holm-6 {fmt_p(r.p_holm_two_sided)}); scipy 'auto' p = {fmt_p(r.p_scipy_auto_two_sided)}; one-sided p(model > baseline) = {fmt_p(r.p_exact_one_sided_model_greater)}; r_rb = {r.r_rb:+.3f}; agreement with copy-baseline {r.agreement_with_baseline_mean*100:.2f}% (min animal {r.agreement_with_baseline_min*100:.2f}%); accuracy at transition seconds {r.transition_accuracy_mean:.4f} (max animal {r.transition_accuracy_max:.4f}), at non-transition seconds {r.non_transition_accuracy_mean:.4f}; median {r.transitions_per_animal_median:.0f} transition seconds per animal.")
    note(key,
         "- Accel-only models, accuracy at 1-s transition seconds (y_t ≠ y_(t-1)), mean over animals: " + ", ".join(f"{m} {v:.3f}" for m, v in acc_tr) + ".",
         f"- **Reconciliation of the text p-values (main.tex l.170).** Text: STA-LSTM-H p = 1.5×10⁻⁵ (one tie, n_eff = 17), STA-LSTM p = 7.6×10⁻⁶ (n = 18). `Results/revision/persistence_vs_predictprev.csv`: {old.wilcoxon_p.iloc[0]} and {old.wilcoxon_p.iloc[1]}. Recomputed here on the matched targets: STA-LSTM-H exact {fmt_p(hb.loc['STA-LSTM-H','p_exact_two_sided'])} (n_eff {int(hb.loc['STA-LSTM-H','n_eff'])}; tied animal(s): {tied_animals}), scipy-auto {fmt_p(hb.loc['STA-LSTM-H','p_scipy_auto_two_sided'])}; STA-LSTM exact {fmt_p(hb.loc['STA-LSTM','p_exact_two_sided'])} (n_eff {int(hb.loc['STA-LSTM','n_eff'])}), scipy-auto {fmt_p(hb.loc['STA-LSTM','p_scipy_auto_two_sided'])}. When every non-zero difference has the same sign, the exact conditional p is 2/2^n_eff irrespective of tied |d|, so **the text values (2/2^17, 2/2^18) are the correct exact values; the CSV values are scipy's normal approximation (triggered by zeros/ties)**. Note the old CSV compared against `predict_prev_cohort.csv`, which averages StratifiedKFold folds (mean |difference| from the matched-target value {np.mean(np.abs(ppc.loc[ANIMALS].to_numpy() - G.set_index('animal').predict_prev_1s.loc[ANIMALS].to_numpy()))*100:.4f} pp).",
         f"- Wording check: STA-LSTM-H(+bp) is below predict_prev in {int(hb.loc['STA-LSTM-H','n_below'])} animals and tied in {int(hb.loc['STA-LSTM-H','n_equal'])} — \"below in every animal\" is not literally true (tie). 1D-CNN(+bp) is above the ceiling in {int(PR[(PR.input=='bp')&(PR.model=='1D-CNN')].n_above.iloc[0])} animal(s); see table above for all six.",
         )

    # ════════════════════════════════════════════════════════════════════════
    # 3. METRIC DEPENDENCE & CLASS WEIGHTING (R1.M4, R3.M4)
    # ════════════════════════════════════════════════════════════════════════
    print("[3] metrics ...")
    acc7 = MET[MET.input == "accel"].copy()
    src_met = save(acc7, "accel7_metrics_per_animal.csv")
    save(MET, "all_models_metrics_per_animal.csv")
    mcols = ["accuracy", "balanced_accuracy", "macro_f1", "kappa"] + [f"{s}_{c}" for c in CLASSES for s in ["recall", "precision", "f1"]]
    msum = acc7.groupby("model")[mcols].agg(["mean", "std"])
    msum.columns = [f"{a}_{b}" for a, b in msum.columns]
    msum = msum.loc[ACCEL7]
    # pooled per-class
    pooled_rows, conf_rows = [], []
    for m in ACCEL7:
        cmp = sum(cms[("accel", m, a)] for a in ANIMALS)
        rown = np.mean([cms[("accel", m, a)] / cms[("accel", m, a)].sum(1, keepdims=True) for a in ANIMALS], axis=0)
        mt, _ = cls_metrics(np.repeat(np.arange(3), cmp.sum(1)), np.concatenate([np.repeat(np.arange(3), cmp[i]) for i in range(3)]))
        pooled_rows.append(dict(model=m, **{f"pooled_{k}": v for k, v in mt.items()}))
        for i in range(3):
            for j in range(3):
                conf_rows.append(dict(model=m, true=CLASSES[i], pred=CLASSES[j], pooled_count=int(cmp[i, j]),
                                      pooled_row_frac=cmp[i, j] / cmp[i].sum(),
                                      mean_per_animal_row_frac=rown[i, j]))
    POOL = pd.DataFrame(pooled_rows).set_index("model")
    msum = msum.join(POOL)
    src_msum = save(msum.reset_index(), "accel7_metrics_summary.csv")
    CONF = pd.DataFrame(conf_rows)
    src_conf = save(CONF, "accel7_confusion_matrices.csv")
    # consistency with loao_report
    rep = []
    for a in ANIMALS:
        h = pd.read_csv(R / f"python_pipeline_loao/animal-{a:02d}/loao_report.csv").set_index("model")
        b = pd.read_csv(R / f"python_pipeline_loao_ablation/animal-{a:02d}/loao_report.csv").set_index("model")
        for m in ACCEL7:
            r_ = b.loc[f"{m} (accel)"] if m.startswith("STA") else h.loc[m]
            mm = acc7[(acc7.model == m) & (acc7.animal == a)].iloc[0]
            rep.append((abs(r_.accuracy_mean - mm.accuracy), abs(r_.f1_mean - mm.macro_f1)))
    rep = np.array(rep)
    # paired tests XGBoost vs each
    pt_rows = []
    for metric in ["accuracy", "balanced_accuracy", "macro_f1", "kappa", "recall_Eating", "recall_Ruminating", "precision_Eating"]:
        piv = acc7.pivot(index="animal", columns="model", values=metric)
        rows_ = []
        for m in ACCEL7[1:]:
            d = piv["XGBoost"] - piv[m]
            w = signflip_wilcoxon(d)
            m_, lo, hi = tci(d)
            rows_.append(dict(metric=metric, comparison=f"XGBoost - {m}", mean_diff_pp=m_ * 100,
                              ci95_lo_pp=lo * 100, ci95_hi_pp=hi * 100, n_xgb_higher=int((d > 0).sum()),
                              p_exact=w["p"], r_rb=w["r_rb"]))
        ph = holm([r["p_exact"] for r in rows_])
        for r, p in zip(rows_, ph):
            r["p_holm_within_metric"] = p
        pt_rows += rows_
    PT = pd.DataFrame(pt_rows)
    src_pt = save(PT, "xgboost_vs_each_by_metric.csv")
    # class weights actually used
    cw_rows = []
    for a in ANIMALS:
        y = np.concatenate([lab[b] for b in ANIMALS if b != a])
        cnt = np.bincount(y, minlength=3).astype(float)
        w = cnt.sum() / (3 * cnt)
        wn = w / w.sum() * 3
        cw_rows.append(dict(held_out=a, **{f"w_{nm}": wn[i] for i, nm in enumerate(CLASSES)}))
    CW = pd.DataFrame(cw_rows)
    save(CW, "loao_class_weights_used.csv")
    key = "R1.M4 / R3.M4 / R3.m4 — metric dependence, per-class, confusion, class weighting"
    note(key, f"Source: {src_met}, {src_msum}, {src_conf}, {src_pt}, `loao_class_weights_used.csv`. All from `loao_last_predictions.npy` (accel-only: STA-LSTM-H/STA-LSTM from `python_pipeline_loao_ablation`, the other five from `python_pipeline_loao` — two separate jobs). Consistency with `loao_report.csv`: max |Δaccuracy| = {rep[:,0].max():.1e}, max |Δmacro-F1| = {rep[:,1].max():.1e}.",
         "- Mean over 18 animals (± SD): " + "; ".join(
             f"{m}: acc {msum.loc[m,'accuracy_mean']:.4f}±{msum.loc[m,'accuracy_std']:.4f}, BA {msum.loc[m,'balanced_accuracy_mean']:.4f}±{msum.loc[m,'balanced_accuracy_std']:.4f}, macro-F1 {msum.loc[m,'macro_f1_mean']:.4f}±{msum.loc[m,'macro_f1_std']:.4f}, κ {msum.loc[m,'kappa_mean']:.3f}" for m in ACCEL7) + ".",
         "- Ranking by metric: accuracy " + " > ".join(msum.sort_values('accuracy_mean', ascending=False).index) + "; balanced accuracy " + " > ".join(msum.sort_values('balanced_accuracy_mean', ascending=False).index) + "; macro-F1 " + " > ".join(msum.sort_values('macro_f1_mean', ascending=False).index) + ".",
         "- Pooled per-class recall (Other / Ruminating / Eating) and Eating precision: " + "; ".join(
             f"{m} {POOL.loc[m,'pooled_recall_Other']:.3f}/{POOL.loc[m,'pooled_recall_Ruminating']:.3f}/{POOL.loc[m,'pooled_recall_Eating']:.3f}, P_Eat {POOL.loc[m,'pooled_precision_Eating']:.3f}" for m in ACCEL7) + ".",
         )
    for metric in ["accuracy", "balanced_accuracy", "macro_f1", "kappa"]:
        sub = PT[PT.metric == metric]
        note(key, f"- XGBoost − model, {metric} (paired exact Wilcoxon, Holm over 6): " + "; ".join(
            f"vs {r.comparison.split(' - ')[1]} {r.mean_diff_pp:+.2f} pp [{r.ci95_lo_pp:+.2f}, {r.ci95_hi_pp:+.2f}], p={fmt_p(r.p_exact)}, p_Holm={fmt_p(r.p_holm_within_metric)}" for r in sub.itertuples()) + ".")
    cf = CONF.set_index(["model", "true", "pred"])
    note(key,
         "- Largest off-diagonal (pooled row fraction): " + "; ".join(
             f"{m}: " + ", ".join(f"{t}→{p} {cf.loc[(m,t,p),'pooled_row_frac']:.3f}" for t in CLASSES for p in CLASSES if t != p) for m in ACCEL7) + ".",
         f"- Class weighting actually used (code): neural models — class-frequency-weighted CE, weights from `data_loader.audit_class_balance` on the 17 training animals, passed to `trainer._train_fold`; normalised weights across the 18 LOAO splits: Other {CW.w_Other.min():.3f}–{CW.w_Other.max():.3f}, Ruminating {CW.w_Ruminating.min():.3f}–{CW.w_Ruminating.max():.3f}, Eating {CW.w_Eating.min():.3f}–{CW.w_Eating.max():.3f}. XGBoost — `trainer._eval_xgboost` calls `clf.fit(X_tr_flat, Yc_tr)` with **no sample weights** on the flattened 25×14 = 350-dim window. Leakage ladder — weighted CE with un-normalised weights total/(3·count) from the training partition (`leakage_experiment.train_eval`). The asymmetric weighting is confirmed.",
         )

    # ════════════════════════════════════════════════════════════════════════
    # 4. DAILY TIME BUDGETS (R2.M3, R1.M4, R3.m4, R2.m16)
    # ════════════════════════════════════════════════════════════════════════
    print("[4] time budgets ...")
    tb_rows = []
    for m in ACCEL7:
        for a in ANIMALS:
            y, p = T[a], P["accel"][m][a]
            n = len(y)
            for ci, nm in enumerate(CLASSES):
                tm = np.sum(y == ci) / n * 1440
                pm = np.sum(p == ci) / n * 1440
                tb_rows.append(dict(model=m, animal=a, trial=TRIAL[a], behaviour=nm,
                                    halter_min_per_day=tm, predicted_min_per_day=pm,
                                    error_min_per_day=pm - tm))
    TB = pd.DataFrame(tb_rows)
    src_tb = save(TB, "time_budget_per_animal.csv")
    tbs = []
    for (m, b), g in TB.groupby(["model", "behaviour"]):
        e = g.error_min_per_day.to_numpy()
        sd = e.std(ddof=1)
        tbs.append(dict(model=m, behaviour=b, mean_signed_error=e.mean(), sd_error=sd,
                        mae=np.abs(e).mean(), ba_bias=e.mean(), loa_lo=e.mean() - 1.96 * sd,
                        loa_hi=e.mean() + 1.96 * sd,
                        ccc=ccc(g.predicted_min_per_day, g.halter_min_per_day),
                        mean_rel_error_pct=np.mean(e / g.halter_min_per_day) * 100,
                        mean_abs_rel_error_pct=np.mean(np.abs(e) / g.halter_min_per_day) * 100,
                        worst_animal=int(g.animal.iloc[np.argmax(np.abs(e))]), worst_error=e[np.argmax(np.abs(e))]))
    TBS = pd.DataFrame(tbs)
    TBS["model"] = pd.Categorical(TBS.model, ACCEL7)
    TBS = TBS.sort_values(["behaviour", "model"])
    src_tbs = save(TBS, "time_budget_agreement_summary.csv")
    # hourly
    hr_rows = []
    for m in ACCEL7:
        for a in ANIMALS:
            y, p = T[a], P["accel"][m][a]
            nh = len(y) // 3600
            for h in range(nh):
                sl = slice(h * 3600, (h + 1) * 3600)
                for ci, nm in enumerate(CLASSES):
                    hr_rows.append(dict(model=m, animal=a, hour=h, behaviour=nm,
                                        halter_min=np.sum(y[sl] == ci) / 60, pred_min=np.sum(p[sl] == ci) / 60))
    HR = pd.DataFrame(hr_rows)
    HR["err"] = HR.pred_min - HR.halter_min
    hrs = HR.groupby(["model", "behaviour"]).apply(lambda g: pd.Series(dict(
        hourly_mae_min=np.abs(g.err).mean(), hourly_bias_min=g.err.mean(),
        hourly_ccc=ccc(g.pred_min, g.halter_min)))).reset_index()
    save(hrs, "time_budget_hourly_summary.csv")
    # halter budgets + drinking
    hb_rows = []
    for a in ANIMALS:
        s = lab[a]
        rd = readme.loc[a]
        days = (rd.end - rd.start).total_seconds() / 86400
        drink_min = rd.drinking_samples / 10 / 60
        hb_rows.append(dict(animal=a, trial=TRIAL[a], publisher_test_animal=bool(rd.publisher_test),
                            **{f"{nm}_min_per_day": np.sum(s == ci) / len(s) * 1440 for ci, nm in enumerate(CLASSES)},
                            recording_days_full=days, drinking_samples_replaced=rd.drinking_samples,
                            drinking_min_total=drink_min, drinking_min_per_day_full_record=drink_min / days))
    HB = pd.DataFrame(hb_rows)
    HB["drinking_share_of_eating_pct_approx"] = HB.drinking_min_per_day_full_record / HB.Eating_min_per_day * 100
    src_hb = save(HB, "halter_time_budgets_per_animal.csv")
    key = "R2.M3 / R1.M4 / R3.m4 / R2.m16 / R2.M6 — daily time budgets"
    note(key, f"Source: {src_tb}, {src_tbs}, `time_budget_hourly_summary.csv`, {src_hb}. Each held-out animal contributes its 86,375 LOAO target seconds (23.99 h); minutes are scaled explicitly to min/day as count/86,375 × 1,440. Bland–Altman bias = mean(pred − halter); LoA = bias ± 1.96·SD (n = 18 animals); CCC = Lin's concordance.")
    for b in ["Ruminating", "Eating", "Other"]:
        sub = TBS[TBS.behaviour == b].set_index("model")
        note(key, f"- {b} (min/day): " + "; ".join(
            f"{m} bias {sub.loc[m,'mean_signed_error']:+.0f} [LoA {sub.loc[m,'loa_lo']:+.0f}, {sub.loc[m,'loa_hi']:+.0f}], MAE {sub.loc[m,'mae']:.0f}, CCC {sub.loc[m,'ccc']:.2f}, rel. err {sub.loc[m,'mean_rel_error_pct']:+.0f}%" for m in ACCEL7) + ".")
    a8 = TB[(TB.animal == 8) & (TB.behaviour == "Ruminating")].set_index("model").error_min_per_day
    hrs_i = hrs.set_index(["model", "behaviour"])
    note(key,
         "- Animal 08, Ruminating error (min/day): " + ", ".join(f"{m} {a8[m]:+.0f}" for m in ACCEL7) + ".",
         "- Hourly (24 points per animal) MAE in min/h, Ruminating / Eating: " + "; ".join(f"{m} {hrs_i.loc[(m,'Ruminating'),'hourly_mae_min']:.1f} / {hrs_i.loc[(m,'Eating'),'hourly_mae_min']:.1f}" for m in ACCEL7) + ".",
         f"- Halter time budgets (first 24 h, min/day; R2.m16): Other {HB.Other_min_per_day.mean():.0f}±{HB.Other_min_per_day.std():.0f} (range {HB.Other_min_per_day.min():.0f}–{HB.Other_min_per_day.max():.0f}), Ruminating {HB.Ruminating_min_per_day.mean():.0f}±{HB.Ruminating_min_per_day.std():.0f} ({HB.Ruminating_min_per_day.min():.0f}–{HB.Ruminating_min_per_day.max():.0f}), Eating {HB.Eating_min_per_day.mean():.0f}±{HB.Eating_min_per_day.std():.0f} ({HB.Eating_min_per_day.min():.0f}–{HB.Eating_min_per_day.max():.0f}). Per animal in the CSV.",
         f"- Drinking merged into Eating (README counts are over each animal's FULL record, not the analysed 24 h): {HB.drinking_min_per_day_full_record.min():.1f}–{HB.drinking_min_per_day_full_record.max():.1f} min/day (median {HB.drinking_min_per_day_full_record.median():.1f}); relative to the first-24-h Eating budget ≈ {HB.drinking_share_of_eating_pct_approx.min():.1f}–{HB.drinking_share_of_eating_pct_approx.max():.1f}% (approximate, mismatched periods).",
         "- Ladder protocols A–D in time-budget terms (R2.M3c): NOT computable — `scripts/leakage_experiment.py` saves only per-animal accuracy/macro-F1 (`leakage_protocol_per_animal.csv`), no per-window predictions.",
         )

    # ════════════════════════════════════════════════════════════════════════
    # 5. RUN-TO-RUN NOISE + SEED AUDIT (R1.M5, R3.M6) + recipe gap (R1.M3, R3.M1)
    # ════════════════════════════════════════════════════════════════════════
    print("[5] run-to-run ...")
    d = (ACC["STA-LSTM-H"] - ACC["STA-LSTM"]).to_numpy()
    m_, lo, hi = tci(d)
    w = signflip_wilcoxon(d)
    rr = dict(mean_pp=m_ * 100, sd_pp=d.std(ddof=1) * 100, mean_abs_pp=np.abs(d).mean() * 100,
              min_pp=d.min() * 100, max_pp=d.max() * 100, se_mean_pp=d.std(ddof=1) / np.sqrt(18) * 100,
              ci95_lo_pp=lo * 100, ci95_hi_pp=hi * 100, p_exact=w["p"],
              **{f"tost_t_p_{k}pp": tost_t(d, k / 100) for k in [1, 2, 3]},
              **{f"tost_wilcoxon_p_{k}pp": tost_wilcoxon(d, k / 100) for k in [1, 2, 3]},
              mde80_pp_alpha05=mde(d.std(ddof=1)) * 100,
              mde80_pp_alpha_bonf21=mde(d.std(ddof=1), alpha=0.05 / 21) * 100,
              per_animal_sd_implied_single_run_pp=d.std(ddof=1) / np.sqrt(2) * 100)
    RR = pd.DataFrame([rr])
    src_rr = save(RR, "replicate_sta_lstm_h_vs_sta_lstm.csv")
    save(pd.DataFrame({"animal": ANIMALS, "sta_lstm_h": ACC["STA-LSTM-H"].values,
                       "sta_lstm": ACC["STA-LSTM"].values, "diff_pp": d * 100}), "replicate_per_animal.csv")
    # seed audit
    sa = []
    for f in sorted((R / "python_pipeline_seedaudit").glob("animal-*/seed_audit_per_seed.csv")):
        a = int(re.search(r"animal-(\d+)", str(f)).group(1))
        df = pd.read_csv(f).dropna(subset=["accuracy"])
        cfgj = json.loads((f.parent / "seed_audit_run.json").read_text())
        for mdl, g in df.groupby("model"):
            sa.append(dict(animal=a, model=mdl,
                           input="accel+behavior_prev" if mdl in ("lstm_h", "sta_lstm") else "accel only",
                           protocol=f"within-animal {cfgj['folds']}-fold StratifiedKFold shuffle (protocol A)",
                           epochs=cfgj["epochs"], seeds=" ".join(map(str, cfgj["seeds"])),
                           n_seeds=len(g), acc_mean=g.accuracy.mean(), acc_sd=g.accuracy.std(ddof=1),
                           acc_range=g.accuracy.max() - g.accuracy.min()))
    SA = pd.DataFrame(sa)
    src_sa = save(SA, "seed_audit_described.csv")
    sas = SA.groupby(["model", "input"]).agg(acc_mean=("acc_mean", "mean"), max_sd=("acc_sd", "max"), mean_sd=("acc_sd", "mean")).reset_index()
    # within-animal protocol A (pipeline) vs LOAO, with real epochs
    wa_rows = []
    for a in ANIMALS:
        rs_h = json.loads((R / f"python_pipeline/animal-{a:02d}/run_summary.json").read_text())
        rs_b = json.loads((R / f"python_pipeline_ablation/animal-{a:02d}/run_summary.json").read_text())
        h = pd.read_csv(R / f"python_pipeline/animal-{a:02d}/comparison_report.csv").set_index("model")
        b = pd.read_csv(R / f"python_pipeline_ablation/animal-{a:02d}/comparison_report.csv").set_index("model")
        for m, src_df, nm, ep in [("LSTM", h, "LSTM", rs_h["epochs"]), ("GRU", h, "GRU", rs_h["epochs"]),
                                  ("Transformer", h, "Transformer", rs_h["epochs"]),
                                  ("STA-LSTM-H", b, "STA-LSTM-H (accel)", rs_b["epochs"]),
                                  ("STA-LSTM", b, "STA-LSTM (accel)", rs_b["epochs"])]:
            if nm in src_df.index:
                wa_rows.append(dict(animal=a, model=m, within_animal_A=src_df.loc[nm, "accuracy_mean"],
                                    within_epochs=ep, loao=ACC.loc[a, m]))
    WA = pd.DataFrame(wa_rows)
    WA["gap_pp"] = (WA.within_animal_A - WA.loao) * 100
    save(WA, "within_animal_A_vs_loao_per_animal.csv")
    was = WA.groupby("model").agg(epochs=("within_epochs", "first"), within=("within_animal_A", "mean"),
                                  loao=("loao", "mean"), gap_pp=("gap_pp", "mean"),
                                  gap_min=("gap_pp", "min"), gap_max=("gap_pp", "max")).reset_index()
    src_wa = save(was, "within_animal_A_vs_loao_summary.csv")
    key = "R1.M5 / R3.M6 — run-to-run noise and seed audit"
    note(key, f"Source: {src_rr}, `replicate_per_animal.csv`, {src_sa}.",
         f"- STA-LSTM-H(accel) − STA-LSTM(accel), identical architecture, per animal (n=18): mean {rr['mean_pp']:+.2f} pp, **SD {rr['sd_pp']:.2f} pp**, mean |d| {rr['mean_abs_pp']:.2f} pp, range {rr['min_pp']:+.2f} to {rr['max_pp']:+.2f} pp; SE of the cohort-mean difference {rr['se_mean_pp']:.2f} pp, 95% CI [{rr['ci95_lo_pp']:+.2f}, {rr['ci95_hi_pp']:+.2f}]; exact Wilcoxon p = {fmt_p(rr['p_exact'])}. Implied single-run per-animal SD ≈ SD/√2 = {rr['per_animal_sd_implied_single_run_pp']:.2f} pp.",
         f"- TOST for this self-replicate (paired t): p = {rr['tost_t_p_1pp']:.3f} (±1 pp), {rr['tost_t_p_2pp']:.3f} (±2 pp), {rr['tost_t_p_3pp']:.3f} (±3 pp); rank-based TOST: {rr['tost_wilcoxon_p_1pp']:.3f} / {rr['tost_wilcoxon_p_2pp']:.3f} / {rr['tost_wilcoxon_p_3pp']:.3f}. **Equivalence of a model with itself is not established at ±2 pp.**",
         f"- Minimum detectable paired difference (80% power, paired t, n=18, SD = {rr['sd_pp']:.2f} pp): {rr['mde80_pp_alpha05']:.2f} pp at α=0.05; {rr['mde80_pp_alpha_bonf21']:.2f} pp at α=0.05/21 (worst-case Holm).",
         "- Seed-audit file contents (`Results/python_pipeline_seedaudit/animal-{01,09,17}/seed_audit_per_seed.csv`; config from `seed_audit_run.json`): animals 01, 09, 17; seeds 7, 42, 101, 1729, 2026; 40 epochs; within-animal 5-fold shuffled StratifiedKFold (protocol A, via `trainer.run_cross_validation`, which fits normalisation and class weights on the whole animal). Per model (input; mean accuracy over the 3 animals; max / mean across-seed SD): " + "; ".join(
             f"{r.model} ({r.input}; {r.acc_mean:.4f}; {r.max_sd:.4f} / {r.mean_sd:.4f})" for r in sas.itertuples()) + ". The `aggregate_seed_audit_summary.csv` omits cnn1d and xgboost. The manuscript's \"≤0.0038\" is the accel-only Transformer's maximum across-seed SD; LSTM/GRU/Transformer/1D-CNN/XGBoost were accel-only, only lstm_h/sta_lstm had the behaviour channel. Across-seed SD here is of a 5-fold-averaged score under leaky protocol A and is not comparable with a single-split LOAO replicate SD.",
         )
    key2 = "R1.M3 / R3.M1 — recipe of within-animal protocol-A results quoted by the reviewers"
    note(key2, f"Source: {src_wa}, `within_animal_A_vs_loao_per_animal.csv`; epochs read from each `run_summary.json`.",
         "- " + "; ".join(f"{r.model}: within-animal A {r.within:.4f} ({r.epochs} epochs) vs accel-only LOAO {r.loao:.4f}, gap {r.gap_pp:+.1f} pp (per-animal {r.gap_min:+.1f} to {r.gap_max:+.1f})" for r in was.itertuples()) + ".",
         "- **Correction to the reviewers' premise:** the LSTM/GRU/Transformer protocol-A values (0.912/0.919/0.858, `aggregate_neural_summary.csv`, `Results/python_pipeline/`) were trained for **10 epochs** (all 18 `run_summary.json` say `\"epochs\": 10`), not 40; the STA-LSTM-H/STA-LSTM accel-only values (0.955/0.956, `python_pipeline_ablation`) are 40 epochs. Both families use `run_cross_validation` (whole-animal normalisation/class weights; multi-task). The ladder used 5 epochs, batch 256, zero xy targets (`submit_leakage.sh`, `leakage_experiment.py`).",
         )

    # ════════════════════════════════════════════════════════════════════════
    # 6. HETEROGENEITY / DEPENDENCE (R1.M8, R1.M7)
    # ════════════════════════════════════════════════════════════════════════
    print("[6] heterogeneity ...")
    import statsmodels.formula.api as smf
    fr = stats.friedmanchisquare(*[ACC[m].values for m in ACCEL7])
    ranks = ACC.rank(axis=1, ascending=False)
    kendall_w = fr.statistic / (18 * (7 - 1))
    long = ACC.reset_index().melt(id_vars="animal", var_name="model", value_name="acc")
    long["trial"] = long.animal.map(TRIAL).astype(str)
    long["acc_pp"] = long.acc * 100
    m1 = smf.mixedlm("acc_pp ~ C(model, Treatment('XGBoost'))", long, groups=long["animal"]).fit(reml=True)
    va, ve = float(m1.cov_re.iloc[0, 0]), float(m1.scale)
    m2 = smf.mixedlm("acc_pp ~ C(model, Treatment('XGBoost')) + C(trial)", long, groups=long["animal"]).fit(reml=True)
    va2, ve2 = float(m2.cov_re.iloc[0, 0]), float(m2.scale)
    m1ml = smf.mixedlm("acc_pp ~ C(model, Treatment('XGBoost'))", long, groups=long["animal"]).fit(reml=False)
    m2ml = smf.mixedlm("acc_pp ~ C(model, Treatment('XGBoost')) + C(trial)", long, groups=long["animal"]).fit(reml=False)
    lr = 2 * (m2ml.llf - m1ml.llf)
    p_lr = stats.chi2.sf(lr, 2)
    fe = pd.DataFrame({"coef_pp": m1.fe_params, "se_pp": m1.bse_fe, "p": m1.pvalues[m1.fe_params.index]})
    fe2 = pd.DataFrame({"coef_pp": m2.fe_params, "se_pp": m2.bse_fe, "p": m2.pvalues[m2.fe_params.index]})
    save(fe.reset_index().rename(columns={"index": "term"}), "mixed_model_fixed_effects.csv")
    save(fe2.reset_index().rename(columns={"index": "term"}), "mixed_model_with_trial_fixed_effects.csv")
    var_models = float(np.var(ACC.mean().values, ddof=1))
    het = pd.DataFrame([dict(friedman_chi2=fr.statistic, friedman_p=fr.pvalue, kendall_w=kendall_w,
                             icc_animal=va / (va + ve), var_animal_pp2=va, var_resid_pp2=ve,
                             icc_animal_given_trial=va2 / (va2 + ve2), var_animal_given_trial_pp2=va2,
                             var_resid_trial_model_pp2=ve2, lr_trial_chi2_df2=lr, lr_trial_p=p_lr,
                             var_of_model_means_pp2=var_models * 1e4,
                             xgb_rank1_animals=int((ranks["XGBoost"] == 1).sum()))])
    src_het = save(het, "heterogeneity_summary.csv")
    mr = ranks.mean().sort_values()
    save(mr.rename("mean_rank").reset_index(), "friedman_mean_ranks.csv")
    trial_means = ACC.groupby(ACC.index.map(TRIAL)).mean()
    trial_means.index = [TRIAL_NAME[i] for i in trial_means.index]
    lkp = pd.read_csv(R / "leakage_protocol_per_animal.csv")
    lkp["P"] = lkp.protocol.str[0]
    LK = lkp.pivot(index="animal", columns="P", values="accuracy")
    LKF = lkp.pivot(index="animal", columns="P", values="f1")
    lt = LK.groupby(LK.index.map(TRIAL)).mean()
    lt.index = [TRIAL_NAME[i] for i in lt.index]
    tm = trial_means.join(lt.add_prefix("ladder_"))
    src_tm = save(tm.reset_index().rename(columns={"index": "trial"}), "per_trial_means.csv")
    # best vs median model (paired)
    order_ = ACC.mean().sort_values(ascending=False).index.tolist()
    med_model = order_[3]
    dbm = ACC[order_[0]] - ACC[med_model]
    bm = tci(dbm)
    neur = [m for m in order_ if m != "XGBoost"]
    dnm = ACC[neur[0]] - ACC[neur[len(neur) // 2]]
    nm_ = tci(dnm)
    key = "R1.M8 / R1.M7 — heterogeneity, dependence, omnibus tests"
    note(key, f"Source: {src_het}, `friedman_mean_ranks.csv`, `mixed_model_fixed_effects.csv`, `mixed_model_with_trial_fixed_effects.csv`, {src_tm}. 7 accel-only LOAO models × 18 animals.",
         f"- Friedman χ²(6) = {fr.statistic:.1f}, p = {fr.pvalue:.2g}; Kendall's W = {kendall_w:.3f}. Mean ranks (1 = best): " + ", ".join(f"{k} {v:.2f}" for k, v in mr.items()) + f". XGBoost ranks first in {het.xgb_rank1_animals.iloc[0]}/18 animals.",
         f"- Linear mixed model acc(pp) ~ model + (1|animal), REML (statsmodels MixedLM): animal variance {va:.1f} pp², residual {ve:.2f} pp², **ICC(animal) = {va/(va+ve):.3f}**. Fixed effects vs XGBoost (pp): " + ", ".join(f"{t.split('[T.')[1].rstrip(']')} {r.coef_pp:+.2f} (SE {r.se_pp:.2f})" for t, r in fe.iterrows() if "T." in t) + ".",
         f"- Adding trial as a fixed factor: trial effects (vs T1) " + ", ".join(f"{t.split('[T.')[1].rstrip(']')}: {r.coef_pp:+.1f} pp (SE {r.se_pp:.1f}, p={fmt_p(r.p)})" for t, r in fe2.iterrows() if 'trial' in t) + f"; animal variance drops to {va2:.1f} pp² (ICC given trial {va2/(va2+ve2):.3f}); LR test for trial (ML fits) χ²(2) = {lr:.2f}, p = {p_lr:.3f}. Trial is estimable only as a 3-level fixed factor; with 3 trials a random trial effect is not identifiable.",
         "- Per-trial mean LOAO accuracy: " + "; ".join(f"{t}: " + ", ".join(f"{m} {tm.loc[t,m]:.3f}" for m in ACCEL7) for t in tm.index) + ".",
         "- Per-trial ladder means (A/B/C/D): " + "; ".join(f"{t}: {tm.loc[t,'ladder_A']:.3f}/{tm.loc[t,'ladder_B']:.3f}/{tm.loc[t,'ladder_C']:.3f}/{tm.loc[t,'ladder_D']:.3f}" for t in tm.index) + ".",
         f"- Architecture effect as a variance component: variance of the 7 model means = {var_models*1e4:.2f} pp² vs between-animal variance {va:.1f} pp². Paired best − median model ({order_[0]} − {med_model}): {bm[0]*100:+.2f} pp [95% CI {bm[1]*100:+.2f}, {bm[2]*100:+.2f}]; among neural models best − median ({neur[0]} − {neur[len(neur)//2]}): {nm_[0]*100:+.2f} pp [{nm_[1]*100:+.2f}, {nm_[2]*100:+.2f}].",
         "- Dependence caveat: any two LOAO training sets share 16 of 17 animals; p-values/CIs are conditional on this training pool (variation over test animals), not over replications of the study. Not computable here: a corrected-resampled-t (Nadeau–Bengio) requires replicate training runs.",
         )

    # ════════════════════════════════════════════════════════════════════════
    # 7. 21-PAIR BATTERY (Table 5 generator) + TOST SENSITIVITY (R1.M9, R1.m9, R3.M10)
    # ════════════════════════════════════════════════════════════════════════
    print("[7] pairwise + TOST ...")
    pw = []
    for a_, b_ in combinations(ACCEL7, 2):
        hi_, lo_ = (a_, b_) if ACC[a_].mean() >= ACC[b_].mean() else (b_, a_)
        dd = (ACC[hi_] - ACC[lo_]).to_numpy()
        w = signflip_wilcoxon(dd)
        row = dict(pair=f"{hi_} vs {lo_}", d_pp=dd.mean() * 100, sd_diff_pp=dd.std(ddof=1) * 100,
                   p=w["p"], p_scipy_auto=scipy_auto_p(ACC[hi_], ACC[lo_]), r_rb=w["r_rb"],
                   n_zero=w["n_zero"], n_tie_groups=w["n_tie_groups"],
                   hl_pp=hodges_lehmann(dd) * 100, n_hi_better=int((dd > 0).sum()),
                   mde80_pp=mde(dd.std(ddof=1)) * 100)
        bl, bh = boot_ci(dd)
        row.update(boot90_lo_pp=bl * 100, boot90_hi_pp=bh * 100)
        _, c90l, c90h = tci(dd, 0.90)
        row.update(t90_lo_pp=c90l * 100, t90_hi_pp=c90h * 100)
        for k in [1, 2, 3, 4, 5]:
            row[f"tost_t_p_{k}pp"] = tost_t(dd, k / 100)
            row[f"tost_w_p_{k}pp"] = tost_wilcoxon(dd, k / 100)
        pw.append(row)
    PW = pd.DataFrame(pw)
    PW["p_holm"] = holm(PW.p)
    for k in [1, 2, 3, 4, 5]:
        PW[f"tost_t_p_{k}pp_holm"] = holm(PW[f"tost_t_p_{k}pp"])
        PW[f"tost_w_p_{k}pp_holm"] = holm(PW[f"tost_w_p_{k}pp"])
    PW = PW.sort_values("p").reset_index(drop=True)
    src_pw = save(PW, "pairwise_21_accel_loao.csv")
    sens = []
    for k in [1, 2, 3, 4, 5]:
        sens.append(dict(margin_pp=k,
                         equiv_t_raw=int((PW[f"tost_t_p_{k}pp"] < 0.05).sum()),
                         equiv_t_holm=int((PW[f"tost_t_p_{k}pp_holm"] < 0.05).sum()),
                         equiv_wilcoxon_raw=int((PW[f"tost_w_p_{k}pp"] < 0.05).sum()),
                         equiv_wilcoxon_holm=int((PW[f"tost_w_p_{k}pp_holm"] < 0.05).sum()),
                         pairs_t_holm="; ".join(PW.pair[PW[f"tost_t_p_{k}pp_holm"] < 0.05]),
                         pairs_t_raw="; ".join(PW.pair[PW[f"tost_t_p_{k}pp"] < 0.05])))
    SENS = pd.DataFrame(sens)
    src_sens = save(SENS, "tost_margin_sensitivity.csv")
    # compare with manuscript Table 5 (tab:pairwise) and old CSV
    ms_pw = {"XGBoost vs STA-LSTM": (3.9, 1.6e-4, 1.00), "XGBoost vs LSTM": (3.4, 1.6e-4, 1.00),
             "XGBoost vs STA-LSTM-H": (3.0, 2.9e-4, 0.99), "XGBoost vs GRU": (3.2, 4.1e-4, 0.98),
             "XGBoost vs Transformer": (2.7, 2.5e-3, 0.92), "XGBoost vs 1D-CNN": (1.7, 1.1e-2, 0.85),
             "1D-CNN vs LSTM": (1.8, 1.1e-2, 0.85), "1D-CNN vs GRU": (1.5, 0.056, 0.74),
             "1D-CNN vs STA-LSTM-H": (1.2, 0.16, 0.66), "1D-CNN vs STA-LSTM": (2.2, 0.32, 0.59),
             "Transformer vs STA-LSTM": (1.2, 0.53, 0.53), "Transformer vs STA-LSTM-H": (0.2, 0.60, 0.50),
             "Transformer vs LSTM": (0.8, 0.60, 0.51), "1D-CNN vs Transformer": (1.0, 1.00, 0.28),
             "Transformer vs GRU": (0.5, 1.00, 0.42), "STA-LSTM-H vs GRU": (0.3, 1.00, -0.04),
             "STA-LSTM-H vs LSTM": (0.5, 1.00, 0.06), "STA-LSTM-H vs STA-LSTM": (0.9, 1.00, 0.26),
             "GRU vs LSTM": (0.2, 1.00, -0.03), "GRU vs STA-LSTM": (0.6, 1.00, 0.18),
             "LSTM vs STA-LSTM": (0.4, 1.00, 0.15)}
    ms_equiv = {"STA-LSTM-H vs GRU", "STA-LSTM-H vs LSTM", "GRU vs LSTM"}
    mm_rows = []
    pwi = PW.set_index("pair")
    for pair, (dpp, ph, rrb) in ms_pw.items():
        r = pwi.loc[pair]
        mm_rows.append(dict(pair=pair, ms_d=dpp, d=round(r.d_pp, 2), ms_p_holm=ph, p_holm=r.p_holm,
                            ms_r_rb=rrb, r_rb=round(r.r_rb, 3), ms_equiv=pair in ms_equiv,
                            equiv_2pp_t_raw=bool(r.tost_t_p_2pp < 0.05),
                            d_mismatch=abs(r.d_pp - dpp) > 0.0501, d_exact=r.d_pp,
                            p_mismatch=(abs(r.p_holm - ph) / ph > 0.06) if ph < 1 else bool(r.p_holm < 0.995),
                            r_mismatch=abs(round(r.r_rb, 2) - rrb) > 0.011))
    MM = pd.DataFrame(mm_rows)
    src_mm = save(MM, "table5_pairwise_vs_manuscript.csv")
    old7 = pd.read_csv(R / "revision" / "convergence_pairwise_7model.csv").set_index("pair")
    old_cmp = [(p, abs(old7.loc[p, "p"] - pwi.loc[p, "p"]), abs(old7.loc[p, "tost_p"] - pwi.loc[p, "tost_t_p_2pp"])) for p in old7.index if p in pwi.index]
    key = "R1.M9 / R1.m9 / R3.M10 — 21-pair battery (Table 5 generator) and TOST sensitivity"
    note(key, f"Source: {src_pw} (generated here — this replaces the un-scripted `Results/revision/convergence_pairwise_7model.csv`), {src_sens}, {src_mm}.",
         f"- Reproduction of `convergence_pairwise_7model.csv`: max |Δp| = {max(x[1] for x in old_cmp):.1e}, max |ΔTOST p| = {max(x[2] for x in old_cmp):.1e} over {len(old_cmp)} pairs. Uncorrected p here is the exact sign-flip p; pairs with zero or tied |differences| (where scipy falls back to the normal approximation, as the old file did): " + (", ".join(f"{p_} (zeros {int(pwi.loc[p_,'n_zero'])}, tie groups {int(pwi.loc[p_,'n_tie_groups'])}; exact {pwi.loc[p_,'p']:.4f} vs approx {pwi.loc[p_,'p_scipy_auto']:.4f})" for p_ in pwi.index if pwi.loc[p_,'n_zero'] or pwi.loc[p_,'n_tie_groups']) or "none") + ". Only exact p changes; all Holm verdicts are identical.",
         f"- Manuscript Table 5 (tab:pairwise) check: Δ values differing from the exact difference by >0.05 pp: {', '.join(f"{r.pair} (text {r.ms_d}, exact {r.d_exact:.3f})" for r in MM[MM.d_mismatch].itertuples()) or 'none'} (rounding-level only); Holm-p mismatches: {', '.join(MM.pair[MM.p_mismatch]) or 'none'}; r_rb mismatches: {', '.join(MM.pair[MM.r_mismatch]) or 'none'}; equivalence-verdict mismatches: {', '.join(MM.pair[MM.ms_equiv != MM.equiv_2pp_t_raw]) or 'none'}.",
         "- TOST equivalence verdicts by margin (number of the 21 pairs equivalent; paired-t raw / paired-t Holm-21 / rank-based raw / rank-based Holm-21): " + "; ".join(
             f"±{r.margin_pp} pp: {r.equiv_t_raw}/{r.equiv_t_holm}/{r.equiv_wilcoxon_raw}/{r.equiv_wilcoxon_holm}" for r in SENS.itertuples()) + ".",
         "- Pairs equivalent after Holm (paired t): " + "; ".join(f"±{r.margin_pp} pp: [{r.pairs_t_holm or 'none'}]" for r in SENS.itertuples()) + ".",
         f"- The three \"equivalent at ±2 pp\" verdicts in the manuscript are raw (uncorrected): " + ", ".join(f"{p} TOST p={pwi.loc[p,'tost_t_p_2pp']:.4f} (Holm {pwi.loc[p,'tost_t_p_2pp_holm']:.3f})" for p in sorted(ms_equiv)) + ".",
         f"- SD of paired differences across the 21 pairs: {PW.sd_diff_pp.min():.2f}–{PW.sd_diff_pp.max():.2f} pp; MDE (80% power, α=0.05, n=18) {PW.mde80_pp.min():.2f}–{PW.mde80_pp.max():.2f} pp. Transformer pairs: paired-difference SD " + ", ".join(f"{p} {pwi.loc[p,'sd_diff_pp']:.2f}" for p in pwi.index if "Transformer" in p) + " pp — TOST depends on these, not on the marginal SD (R1.M9d).",
         "- Machinery: CI/TOST in the manuscript are paired-t; significance is Wilcoxon. Rank-based TOST (two one-sided exact Wilcoxon tests on d±margin) and bootstrap 90% CIs (10,000 resamples, seed fixed) are given in the CSV for consistency.",
         )

    # ════════════════════════════════════════════════════════════════════════
    # 8. PREDICTION INTERVAL for a new animal (R1.m11)
    # ════════════════════════════════════════════════════════════════════════
    pi_rows = []
    for m in ACCEL7:
        x = ACC[m].to_numpy()
        h = stats.t.ppf(0.975, 17) * x.std(ddof=1) * np.sqrt(1 + 1 / 18)
        pi_rows.append(dict(model=m, mean=x.mean(), sd=x.std(ddof=1), pi95_lo=x.mean() - h,
                            pi95_hi=x.mean() + h, min=x.min(), min_animal=int(ACC[m].idxmin()),
                            max=x.max(), max_animal=int(ACC[m].idxmax()),
                            min_excl_08=ACC[m].drop(8).min()))
    PI = pd.DataFrame(pi_rows)
    src_pi = save(PI, "prediction_interval_new_animal.csv")
    note("R1.m11 — prediction interval for a new animal",
         f"Source: {src_pi}. 95% PI = mean ± t(0.975,17)·SD·√(1+1/18) (normality assumed; animal 08 makes the lower tail heavy).",
         "- " + "; ".join(f"{r.model}: {r.mean:.3f}, PI [{r.pi95_lo:.3f}, {r.pi95_hi:.3f}], per-animal range {r.min:.3f} (animal {r.min_animal:02d}) – {r.max:.3f} (animal {r.max_animal:02d}); lowest excluding 08 {r.min_excl_08:.3f}" for r in PI.itertuples()) + ".")

    # ════════════════════════════════════════════════════════════════════════
    # 9. LEAKAGE LADDER EXTRAS (R1.m13, R3.m5, R1.m2, R1.m12)
    # ════════════════════════════════════════════════════════════════════════
    lad = []
    for hi_, lo_ in [("A", "B"), ("A", "C"), ("A", "D"), ("B", "C")]:
        dd = (LK[hi_] - LK[lo_]).to_numpy()
        w = signflip_wilcoxon(dd)
        m_, l_, h_ = tci(dd)
        df1 = (LKF[hi_] - LKF[lo_]).to_numpy()
        lad.append(dict(contrast=f"{hi_}-{lo_}", delta_pp=m_ * 100, ci95_lo=l_ * 100, ci95_hi=h_ * 100,
                        p_exact=w["p"], r_rb=w["r_rb"], min_pp=dd.min() * 100, max_pp=dd.max() * 100,
                        min_animal=int(LK.index[np.argmin(dd)]), max_animal=int(LK.index[np.argmax(dd)]),
                        n_reversals=int((dd < 0).sum()), macro_f1_delta_pp=df1.mean() * 100,
                        macro_f1_delta_sd_pp=df1.std(ddof=1) * 100))
    LAD = pd.DataFrame(lad)
    LAD["p_holm4"] = holm(LAD.p_exact)
    src_lad = save(LAD, "ladder_contrasts_extended.csv")
    lsum = pd.DataFrame({"acc_mean": LK.mean(), "acc_sd": LK.std(), "macro_f1_mean": LKF.mean(), "macro_f1_sd": LKF.std(),
                         "n_test_windows_per_animal": [86375, 86375, 86375 - int(86375 * 0.8), 3455 - int(3455 * 0.8)],
                         "per_animal_estimate": ["mean of 5 folds (all windows tested)"] * 2 + ["single 20% chronological test segment"] * 2})
    save(lsum.reset_index().rename(columns={"P": "protocol"}), "ladder_protocol_summary.csv")
    note("R1.m13 / R3.m5 / R1.m2 / R1.m12 — leakage ladder extras",
         f"Source: {src_lad}, `ladder_protocol_summary.csv` (from `Results/leakage_protocol_per_animal.csv`).",
         "- " + "; ".join(f"{r.contrast}: {r.delta_pp:+.2f} pp [{r.ci95_lo:+.2f}, {r.ci95_hi:+.2f}], exact p={fmt_p(r.p_exact)} (Holm-4 {fmt_p(r.p_holm4)}), r_rb={r.r_rb:+.3f}, per-animal range {r.min_pp:+.1f} (animal {r.min_animal:02d}) to {r.max_pp:+.1f} (animal {r.max_animal:02d}), reversals {r.n_reversals}; macro-F1 Δ {r.macro_f1_delta_pp:+.2f} pp (SD {r.macro_f1_delta_sd_pp:.2f})" for r in LAD.itertuples()) + ".",
         "- Protocol accuracy / macro-F1 (mean ± SD over 18 animals): " + "; ".join(f"{p}: {LK[p].mean():.4f}±{LK[p].std():.4f} / {LKF[p].mean():.4f}±{LKF[p].std():.4f}" for p in "ABCD") + ".",
         "- Comparability of dispersions (R3.m5): A and B per-animal scores average 5 fold models and test every window; C and D test one chronological 20% segment (≈17,275 and ≈691 windows) from one model, so their across-animal SD adds test-segment (day-part) and single-run variance. The SD increase from A/B to C/D is therefore not evidence that A 'understates variability'. Per-class F1 for the ladder (R2.m6) is NOT computable — no per-window ladder predictions were saved.",
         "- Sign-reversal reasoning (R1.m12/R3.M2): with the single-run per-animal replicate SD of ≈{:.1f} pp (§R1.M5), 3/18 reversals of a ≈+7.8 pp mean contrast with SD {:.1f} pp are consistent with noise; reversals are not diagnostic of a non-leakage component.".format(rr['sd_pp'], (LK['A'] - LK['C']).std() * 100),
         )

    # ════════════════════════════════════════════════════════════════════════
    # 10. TRIVIAL FLOORS UNDER LOAO (R1.m6, R3.m6, R1.m4)
    # ════════════════════════════════════════════════════════════════════════
    print("[10] LOAO floors ...")
    fl_path = OUT / "loao_trivial_floors_per_animal.csv"
    if args.skip_floors and fl_path.exists():
        FL = pd.read_csv(fl_path)
    else:
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import HistGradientBoostingClassifier
        fl = []
        FLP = {}
        t0 = time.time()
        for a in ANIMALS:
            tr = [b for b in ANIMALS if b != a]
            ytr_all = np.concatenate([lab[b] for b in tr])
            maj = int(np.bincount(ytr_all, minlength=3).argmax())
            y = T[a]
            preds = {"majority (training animals)": np.full(len(y), maj),
                     "predict_prev (t-1)": lab[a][SEQ - 1:-1]}
            # current-second features (no context; concurrent classification)
            Xc = np.concatenate([feat[b] for b in tr]); yc = ytr_all
            mu, sd = Xc.mean(0), Xc.std(0)
            Xc = (Xc - mu) / sd
            Xte_cur = (feat[a][SEQ:] - mu) / sd
            Xte_prev = (feat[a][SEQ - 1:-1] - mu) / sd
            # previous-second features -> label at t (one-step-ahead, no context)
            Xp = np.concatenate([(feat[b][:-1] - mu) / sd for b in tr]); yp = np.concatenate([lab[b][1:] for b in tr])
            lrm = LogisticRegression(max_iter=500).fit(Xc[::5], yc[::5])
            preds["logreg, current second (no context)"] = lrm.predict(Xte_cur)
            hg = HistGradientBoostingClassifier(max_iter=100, random_state=0).fit(Xc[::2], yc[::2])
            preds["HGB, current second (no context)"] = hg.predict(Xte_cur)
            hgw = HistGradientBoostingClassifier(max_iter=100, random_state=0, class_weight="balanced").fit(Xc[::2], yc[::2])
            preds["HGB, current second, balanced class weights"] = hgw.predict(Xte_cur)
            hgp = HistGradientBoostingClassifier(max_iter=100, random_state=0).fit(Xp[::2], yp[::2])
            preds["HGB, previous second only (t-1 -> t)"] = hgp.predict(Xte_prev)
            FLP[f"{a:02d}|HGB_current"] = np.asarray(preds["HGB, current second (no context)"], np.int8)
            for nm, p in preds.items():
                mt, _ = cls_metrics(y, np.asarray(p, np.int64))
                tbe = {f"err_min_{c}": (np.sum(p == ci) - np.sum(y == ci)) / len(y) * 1440 for ci, c in enumerate(CLASSES)}
                fl.append(dict(floor=nm, animal=a, trial=TRIAL[a], train_majority_class=CLASSES[maj], **mt, **tbe))
            print(f"   floors animal {a:02d} done ({time.time()-t0:.0f}s)")
        FL = pd.DataFrame(fl)
        FL.to_csv(fl_path, index=False)
        np.savez_compressed(OUT / "loao_floor_predictions_hgb_current.npz", **FLP)
    fls = FL.groupby("floor")[["accuracy", "balanced_accuracy", "macro_f1", "err_min_Ruminating", "err_min_Eating"]].agg(["mean", "std"])
    fls.columns = [f"{a}_{b}" for a, b in fls.columns]
    save(fls.reset_index(), "loao_trivial_floors_summary.csv")
    maj8 = FL[(FL.floor == "majority (training animals)")]
    oldtriv = pd.read_csv(R / "python_pipeline_trivial" / "aggregate_trivial_summary.csv", header=[0, 1], index_col=0)
    fla = FL.pivot(index="animal", columns="floor", values="accuracy")
    ctx_gain = ACC["XGBoost"] - fla["HGB, current second (no context)"]
    note("R1.m6 / R3.m6 / R1.m4 — trivial floors under LOAO (newly computed)",
         "Source: `Results/revision2/loao_trivial_floors_per_animal.csv`, `loao_trivial_floors_summary.csv`. **Newly computed floors** (CPU, sklearn; LOAO over 18 animals, evaluated on the same 86,375 target seconds): z-scoring fitted on the 17 training animals; logistic regression fitted on every 5th training second and HistGradientBoosting (100 iterations) on every 2nd training second for speed (training seconds are highly redundant); no context window.",
         "- Mean ± SD over 18 animals (accuracy / balanced accuracy / macro-F1; Ruminating and Eating time-budget bias, min/day): " + "; ".join(
             f"{f}: {fls.loc[f,'accuracy_mean']:.4f}±{fls.loc[f,'accuracy_std']:.4f} / {fls.loc[f,'balanced_accuracy_mean']:.4f} / {fls.loc[f,'macro_f1_mean']:.4f}; {fls.loc[f,'err_min_Ruminating_mean']:+.0f} / {fls.loc[f,'err_min_Eating_mean']:+.0f}" for f in fls.index) + ".",
         "- Training-majority floor per held-out animal: " + ", ".join(f"{r.animal:02d} {r.accuracy:.3f} ({r.train_majority_class})" for r in maj8.itertuples()) + ".",
         f"- What the 25-s context + XGBoost adds over a no-context HGB on the current second: {ctx_gain.mean()*100:+.2f} pp (per-animal {ctx_gain.min()*100:+.1f} to {ctx_gain.max()*100:+.1f}). Note the neural/XGBoost task is one-step-ahead (inputs t−25…t−1, label t); the 'current second' floors see the target second, the 'previous second only' floor matches the forecasting alignment.",
         f"- Old within-animal protocol-A floors (`python_pipeline_trivial/aggregate_trivial_summary.csv`): majority {oldtriv.loc['majority', ('accuracy','mean')]:.3f}, logreg {oldtriv.loc['logreg', ('accuracy','mean')]:.3f}, gbdt {oldtriv.loc['gbdt', ('accuracy','mean')]:.3f}, predict_prev {oldtriv.loc['predict_prev', ('accuracy','mean')]:.3f}±{oldtriv.loc['predict_prev', ('accuracy','std')]:.3f} — the predict_prev row is the z-scoring-bug version superseded by `predict_prev_baseline.py` and should not be cited.",
         )

    # ════════════════════════════════════════════════════════════════════════
    # 10b. COLLAR–HALTER TEMPORAL ALIGNMENT (R2.M6 clock sync; explains R3.m7)
    # ════════════════════════════════════════════════════════════════════════
    print("[10b] prediction-label lag alignment ...")
    flp = np.load(OUT / "loao_floor_predictions_hgb_current.npz")
    LAGS = np.arange(-60, 61)
    al_rows = []
    def lagged_acc(p, y, k):
        if k >= 0:
            return float(np.mean(p[:len(p) - k] == y[k:]))
        return float(np.mean(p[-k:] == y[:len(y) + k]))
    srcs = [(m, lambda a, m=m: P["accel"][m][a]) for m in ACCEL7] + \
           [("HGB current-second (no context)", lambda a: flp[f"{a:02d}|HGB_current"].astype(np.int64))]
    for m, getp in srcs:
        for a in ANIMALS:
            p, y = getp(a), T[a]
            for k in LAGS:
                al_rows.append(dict(model=m, animal=a, trial=TRIAL[a], lag_s=int(k), acc=lagged_acc(p, y, k)))
    AL = pd.DataFrame(al_rows)
    save(AL, "prediction_label_lag_per_animal.csv")
    best = AL.loc[AL.groupby(["model", "animal"]).acc.idxmax()].merge(
        AL[AL.lag_s == 0][["model", "animal", "acc"]].rename(columns={"acc": "acc_lag0"}), on=["model", "animal"])
    best["gain_pp"] = (best.acc - best.acc_lag0) * 100
    src_best = save(best, "prediction_label_best_lag.csv")
    curve = AL.groupby(["model", "lag_s"]).acc.mean().unstack(0)
    save(curve.reset_index(), "prediction_label_lag_curve.csv")
    # wide, coarse scan (±15 min, 5-s steps) for XGBoost and the no-context model
    WL = np.arange(-900, 901, 5)
    wide_rows = []
    for m, getp in [srcs[0], srcs[-1]]:
        for a in ANIMALS:
            p, y = getp(a), T[a]
            accs = np.array([lagged_acc(p, y, k) for k in WL])
            wide_rows.append(dict(model=m, animal=a, trial=TRIAL[a], best_lag_s=int(WL[accs.argmax()]),
                                  acc_best=accs.max(), acc_lag0=accs[WL == 0][0],
                                  gain_pp=(accs.max() - accs[WL == 0][0]) * 100))
    WIDE = pd.DataFrame(wide_rows)
    src_wide = save(WIDE, "prediction_label_best_lag_wide.csv")
    # drift check: best lag within each 6-h quarter of the day (XGBoost)
    q_rows = []
    for a in ANIMALS:
        p, y = P["accel"]["XGBoost"][a], T[a]
        qn = len(y) // 4
        for q in range(4):
            sl = slice(q * qn, (q + 1) * qn)
            accs = np.array([lagged_acc(p[sl], y[sl], k) for k in WL])
            q_rows.append(dict(animal=a, trial=TRIAL[a], quarter=q + 1, best_lag_s=int(WL[accs.argmax()]),
                               gain_pp=(accs.max() - accs[WL == 0][0]) * 100))
    QL = pd.DataFrame(q_rows)
    save(QL, "prediction_label_best_lag_by_quarter_xgboost.csv")
    qlw = QL.pivot(index="animal", columns="quarter", values="best_lag_s")
    bt = best.groupby(["model", "trial"]).lag_s.median().unstack()
    key = "R2.M6 / R3.m7 — collar–halter temporal alignment (NEW finding)"
    note(key, f"Source: {src_best}, `prediction_label_lag_curve.csv`, `prediction_label_lag_per_animal.csv`. acc(k) = mean[pred(t) == halter(t + k)] on the LOAO target seconds, k = −60…+60 s; positive k means the halter label that best matches a prediction occurs k s *later*.",
         "- Lag maximising mean accuracy (cohort curve) and gain over k = 0: " + "; ".join(
             f"{m}: k* = {int(curve[m].idxmax())} s, +{(curve[m].max()-curve[m].loc[0])*100:.2f} pp" for m in curve.columns) + ".",
         "- Per-animal best lag, median [min–max] by model: " + "; ".join(
             f"{m}: {best[best.model==m].lag_s.median():.0f} [{best[best.model==m].lag_s.min()}–{best[best.model==m].lag_s.max()}] s" for m in curve.columns) + ".",
         "- Per-trial median best lag (s): " + "; ".join(f"{m}: " + ", ".join(f"T{t} {bt.loc[m,t]:.0f}" for t in bt.columns) for m in bt.index) + ".",
         "- Wide scan (±900 s, 5-s steps), per-animal best lag (s) and gain (pp): " + "; ".join(
             f"{m}: " + ", ".join(f"{int(r.animal):02d}:{r.best_lag_s:+d}({r.gain_pp:+.1f})" for r in WIDE[WIDE.model==m].itertuples()) for m in WIDE.model.unique()) + f" (source {src_wide}).",
         "- Drift check, XGBoost best lag (s) in each 6-h quarter of the analysed day (Q1..Q4): " + "; ".join(f"{a:02d}: " + "/".join(f"{int(v):+d}" for v in qlw.loc[a]) for a in ANIMALS) + " (`prediction_label_best_lag_by_quarter_xgboost.csv`).",
         "- CAUTION: k* is selected on the test labels themselves, so the gains are optimistic upper bounds, not a deployable improvement; the analysis is a data-quality diagnostic.",
         "- Interpretation (to be stated cautiously): for the no-context model (features of second t only) the best-matching halter label lies k* s later, i.e. the halter label stream appears to lag the collar accelerometry by roughly that amount (clock offset and/or the RumiWatch converter assigning labels with a delay). For the 25-s-context models (inputs t−25…t−1) the label at t + k* best matches. This also explains why the causal trailing majority filter (which delays predictions) helps more than the centred filter in R3.m7. Whether the offset is a clock-synchronisation error cannot be established from these files; it can be tested only with timestamps of known events.",
         )

    # ════════════════════════════════════════════════════════════════════════
    # 11. COLLAR ORIENTATION / ANIMAL 08 (R2.M2, R3.m3, R2.M6)
    # ════════════════════════════════════════════════════════════════════════
    print("[11] orientation ...")
    or_rows = []
    for a in ANIMALS:
        s = lab[a]
        F_ = feat[a]
        for ci, nm in list(enumerate(CLASSES)) + [(-1, "All")]:
            msk = np.ones(len(s), bool) if ci < 0 else s == ci
            ax_, ay_, az_ = F_[msk, 0].mean(), F_[msk, 1].mean(), F_[msk, 2].mean()
            or_rows.append(dict(animal=a, trial=TRIAL[a], behaviour=nm, ax_mean_mg=ax_, ay_mean_mg=ay_,
                                az_mean_mg=az_,
                                pitch_deg=np.degrees(np.arctan2(ax_, np.hypot(ay_, az_))),
                                roll_deg=np.degrees(np.arctan2(az_, ay_)),
                                vm_static_mg=np.sqrt(ax_ ** 2 + ay_ ** 2 + az_ ** 2)))
    OR = pd.DataFrame(or_rows)
    src_or = save(OR, "orientation_by_animal_class.csv")
    wide = OR.pivot(index="animal", columns="behaviour", values="ax_mean_mg")
    wide["rum_minus_eat_x"] = wide["Ruminating"] - wide["Eating"]
    allr = OR[OR.behaviour == "All"].set_index("animal")
    ors = pd.DataFrame({"trial": [TRIAL[a] for a in ANIMALS], "x_Rum": wide.Ruminating, "x_Eat": wide.Eating,
                        "rum_minus_eat_x": wide.rum_minus_eat_x, "ax_all": allr.ax_mean_mg,
                        "ay_all": allr.ay_mean_mg, "az_all": allr.az_mean_mg, "roll_all_deg": allr.roll_deg,
                        "pitch_all_deg": allr.pitch_deg}, index=ANIMALS)
    ors.index.name = "animal"
    save(ors.reset_index(), "orientation_summary_per_animal.csv")
    # per-animal LOAO table with trial
    pa = ACC.copy().add_prefix("acc_")
    pa = pa.join(MET[MET.input == "accel"].pivot(index="animal", columns="model", values="macro_f1")[ACCEL7].add_prefix("f1_"))
    pa.insert(0, "trial", [TRIAL[a] for a in pa.index])
    pa["majority_floor"] = fla["majority (training animals)"]
    pa["publisher_test_animal"] = [bool(readme.loc[a, "publisher_test"]) for a in pa.index]
    src_pa = save(pa.reset_index(), "loao_per_animal_with_trial.csv")
    # exclude 08
    A17 = ACC.drop(8)
    ex = []
    for m in ACCEL7[1:]:
        dd = (A17["XGBoost"] - A17[m]).to_numpy()
        ex.append(dict(comparison=f"XGBoost - {m}", d_pp=dd.mean() * 100, p_exact=signflip_wilcoxon(dd)["p"]))
    EX = pd.DataFrame(ex)
    EX["p_holm"] = holm(EX.p_exact)
    fr17 = stats.friedmanchisquare(*[A17[m].values for m in ACCEL7])
    exsum = pd.DataFrame({"mean_18": ACC.mean(), "sd_18": ACC.std(), "mean_excl_08": A17.mean(), "sd_excl_08": A17.std()})
    exsum["delta_pp"] = (exsum.mean_excl_08 - exsum.mean_18) * 100
    src_ex = save(exsum.reset_index(), "sensitivity_exclude_animal08.csv")
    save(EX, "sensitivity_exclude_animal08_xgb_tests.csv")
    pub = ACC.loc[[4, 10, 11]].mean()
    tsum = ors.groupby("trial").agg(x_Rum_min=("x_Rum", "min"), x_Rum_max=("x_Rum", "max"),
                                    x_Eat_min=("x_Eat", "min"), x_Eat_max=("x_Eat", "max"),
                                    d_min=("rum_minus_eat_x", "min"), d_max=("rum_minus_eat_x", "max"),
                                    az_min=("az_all", "min"), az_max=("az_all", "max"),
                                    roll_min=("roll_all_deg", "min"), roll_max=("roll_all_deg", "max"))
    save(tsum.reset_index(), "orientation_summary_per_trial.csv")
    key = "R2.M2 / R3.m3 / R1.M8 / R2.M6 — collar orientation, animal 08, per-animal LOAO"
    note(key, f"Source: {src_or}, `orientation_summary_per_animal.csv`, `orientation_summary_per_trial.csv`, {src_pa}, {src_ex}, `sensitivity_exclude_animal08_xgb_tests.csv`. Means are of the cached per-second axis means (mg) over the seconds of each halter class (first 24 h). Pitch = atan2(x, √(y²+z²)), roll = atan2(z, y) using the README axis convention (x along body, y vertical, z lateral).",
         "- Ruminating − Eating mean x (mg) by trial: " + "; ".join(f"{TRIAL_NAME[t]}: x_Rum {r.x_Rum_min:+.0f} to {r.x_Rum_max:+.0f}, x_Eat {r.x_Eat_min:+.0f} to {r.x_Eat_max:+.0f}, difference {r.d_min:+.0f} to {r.d_max:+.0f}" for t, r in tsum.iterrows()) + f". Sign of (Rum − Eat) x: positive in {int((ors[ors.trial<3].rum_minus_eat_x>0).sum())}/10 animals of 2015, negative in {int((ors[ors.trial==3].rum_minus_eat_x<0).sum())}/8 animals of 2016.",
         "- Overall mean z (mg) and roll (deg) by trial: " + "; ".join(f"{TRIAL_NAME[t]}: z {r.az_min:+.0f} to {r.az_max:+.0f}, roll {r.roll_min:+.0f}° to {r.roll_max:+.0f}°" for t, r in tsum.iterrows()) + f". Animal 08: x/y/z = {ors.loc[8,'ax_all']:+.0f}/{ors.loc[8,'ay_all']:+.0f}/{ors.loc[8,'az_all']:+.0f} mg, roll {ors.loc[8,'roll_all_deg']:+.0f}°; cohort roll range {ors.roll_all_deg.min():+.0f}° to {ors.roll_all_deg.max():+.0f}°.",
         "- Animal 08 accel-only LOAO accuracy: " + ", ".join(f"{m} {ACC.loc[8,m]:.3f}" for m in ACCEL7) + f"; its training-majority floor {fla.loc[8,'majority (training animals)']:.3f}, own majority rate {max(np.bincount(T[8],minlength=3))/len(T[8]):.3f}; macro-F1 {pa.loc[8,[f'f1_{m}' for m in ACCEL7]].min():.3f}–{pa.loc[8,[f'f1_{m}' for m in ACCEL7]].max():.3f}.",
         "- Excluding animal 08 from evaluation (models were still trained with 08 in their training sets; no retraining): mean accuracy change " + ", ".join(f"{m} {exsum.loc[m,'delta_pp']:+.2f} pp (→{exsum.loc[m,'mean_excl_08']:.4f}, SD {exsum.loc[m,'sd_18']:.3f}→{exsum.loc[m,'sd_excl_08']:.3f})" for m in ACCEL7) + f". Ordering by mean: " + " > ".join(exsum.sort_values('mean_excl_08', ascending=False).index) + f" (18-animal ordering: " + " > ".join(exsum.sort_values('mean_18', ascending=False).index) + f"). Friedman (n=17) χ²={fr17.statistic:.1f}, p={fr17.pvalue:.2g}. XGBoost − model (n=17, Holm-6): " + ", ".join(f"{r.comparison.split(' - ')[1]} {r.d_pp:+.2f} pp p_Holm={fmt_p(r.p_holm)}" for r in EX.itertuples()) + ".",
         "- Publisher-designated test animals (04, 10, 11; README) — mean of their LOAO accuracies (not the publishers' train-15/test-3 split): " + ", ".join(f"{m} {pub[m]:.3f}" for m in ACCEL7) + ".",
         "- Not computable without retraining: LOAO after harmonising axis signs / orientation-invariant features (R2.M2c), leave-one-trial-out (R2.M2d, R1.M8, R3.m2).",
         )

    # ════════════════════════════════════════════════════════════════════════
    # 12. TEMPORAL SMOOTHING (R3.m7, R2.M8c)  — post hoc
    # ════════════════════════════════════════════════════════════════════════
    print("[12] smoothing ...")
    def mode_filter(p, w, causal=False):
        oh = np.eye(3)[p]
        if causal:
            cs = np.cumsum(np.vstack([np.zeros((1, 3)), oh]), axis=0)
            lo_ = np.maximum(np.arange(len(p)) - w + 1, 0)
            cnt = cs[np.arange(len(p)) + 1] - cs[lo_]
        else:
            cnt = uniform_filter1d(oh, size=w, axis=0, mode="nearest")
        return cnt.argmax(1)
    sm_rows = []
    WINS = [5, 11, 31, 61, 121]
    for m in ACCEL7:
        for a in ANIMALS:
            y, p = T[a], P["accel"][m][a]
            b0, _ = cls_metrics(y, p)
            for filt in ["median (centred)", "majority (centred)", "majority (causal, trailing)"]:
                for w in WINS:
                    if filt.startswith("median"):
                        q = median_filter(p, size=w, mode="nearest")
                    else:
                        q = mode_filter(p, w, causal="causal" in filt)
                    b1, _ = cls_metrics(y, q)
                    sm_rows.append(dict(model=m, animal=a, filter=filt, window_s=w,
                                        acc_raw=b0["accuracy"], acc_smoothed=b1["accuracy"],
                                        ba_raw=b0["balanced_accuracy"], ba_smoothed=b1["balanced_accuracy"],
                                        f1_raw=b0["macro_f1"], f1_smoothed=b1["macro_f1"]))
    SM = pd.DataFrame(sm_rows)
    SM["d_acc_pp"] = (SM.acc_smoothed - SM.acc_raw) * 100
    SM["d_ba_pp"] = (SM.ba_smoothed - SM.ba_raw) * 100
    SM["d_f1_pp"] = (SM.f1_smoothed - SM.f1_raw) * 100
    save(SM, "smoothing_per_animal.csv")
    sms = SM.groupby(["filter", "model", "window_s"]).agg(
        d_acc_pp=("d_acc_pp", "mean"), d_ba_pp=("d_ba_pp", "mean"), d_f1_pp=("d_f1_pp", "mean"),
        n_improved_acc=("d_acc_pp", lambda x: int((x > 0).sum())),
        n_improved_ba=("d_ba_pp", lambda x: int((x > 0).sum())),
        p_acc=("d_acc_pp", lambda x: signflip_wilcoxon(x.to_numpy())["p"])).reset_index()
    src_sm = save(sms, "smoothing_summary.csv")
    key = "R3.m7 / R2.M8c — temporal smoothing of predictions (POST HOC)"
    note(key, f"Source: {src_sm}, `smoothing_per_animal.csv`. **Post-hoc** analysis on saved accel-only LOAO predictions; all filters and window sizes are reported (no selection). Median filter = `scipy.ndimage.median_filter` on the class index (as the reviewer did; ordinal artefact possible for 3 classes); majority = mode of the window (ties → lower class index); centred filters use (w−1)/2 s of future predictions; the causal trailing filter is deployable without look-ahead. Δ in pp; n improved out of 18; exact Wilcoxon p on Δacc.")
    for filt in ["median (centred)", "majority (centred)", "majority (causal, trailing)"]:
        sub = sms[sms["filter"] == filt]
        for m in ACCEL7:
            ss = sub[sub.model == m]
            note(key, f"- {filt}, {m}: " + "; ".join(f"w={r.window_s}: Δacc {r.d_acc_pp:+.2f} ({r.n_improved_acc}/18, p={fmt_p(r.p_acc)}), ΔBA {r.d_ba_pp:+.2f} ({r.n_improved_ba}/18)" for r in ss.itertuples()))

    # ════════════════════════════════════════════════════════════════════════
    # 13. CLASS-PREVALENCE FACT CHECK (R1.m1)
    # ════════════════════════════════════════════════════════════════════════
    print("[13] fold prevalence ...")
    from sklearn.model_selection import KFold, StratifiedKFold
    fp_rows = []
    for a in ANIMALS:
        y = lab[a][SEQ:]                     # stride-1 targets as in leakage_experiment.build_windows
        overall = np.bincount(y, minlength=3) / len(y)
        for prot, splitter in [("A", StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y)),
                               ("B", KFold(5, shuffle=False).split(np.zeros(len(y))))]:
            for k, (tr, te) in enumerate(splitter):
                pte = np.bincount(y[te], minlength=3) / len(te)
                ptr = np.bincount(y[tr], minlength=3) / len(tr)
                fp_rows.append(dict(animal=a, protocol=prot, fold=k,
                                    **{f"test_{c}": pte[i] for i, c in enumerate(CLASSES)},
                                    **{f"train_{c}": ptr[i] for i, c in enumerate(CLASSES)},
                                    max_abs_test_dev_pp=np.abs(pte - overall).max() * 100,
                                    max_abs_train_dev_pp=np.abs(ptr - overall).max() * 100))
    FP = pd.DataFrame(fp_rows)
    src_fp = save(FP, "fold_class_proportions_A_vs_B.csv")
    fps = FP.groupby("protocol")[["max_abs_test_dev_pp", "max_abs_train_dev_pp"]].agg(["median", "max"])
    prev = pd.read_csv(R / "cohort_class_prevalence.csv")
    over45 = prev[prev.dominant_pct > 0.45]
    note("R1.m1 — class-prevalence fact check (protocol A vs B folds)",
         f"Source: {src_fp} (folds regenerated exactly as in `leakage_experiment.py`: StratifiedKFold(5, shuffle, random_state=42) vs KFold(5) on the 86,375 stride-1 targets), `Results/cohort_class_prevalence.csv`.",
         f"- Max |fold − animal| class-proportion deviation, test folds: A median {fps.loc['A',('max_abs_test_dev_pp','median')]:.3f} / max {fps.loc['A',('max_abs_test_dev_pp','max')]:.3f} pp; B median {fps.loc['B',('max_abs_test_dev_pp','median')]:.1f} / max {fps.loc['B',('max_abs_test_dev_pp','max')]:.1f} pp. Training folds: A max {fps.loc['A',('max_abs_train_dev_pp','max')]:.3f} pp; B median {fps.loc['B',('max_abs_train_dev_pp','median')]:.1f} / max {fps.loc['B',('max_abs_train_dev_pp','max')]:.1f} pp. **Stratification is therefore not negligible for B's test folds** (contiguous 4.8-h blocks have quite different class mixes), although the training-fold proportions move less.",
         f"- Animals whose dominant class exceeds 45% ({len(over45)}): " + ", ".join(f"{int(r.animal):02d} ({r.dominant_class_name} {r.dominant_pct*100:.1f}%)" for r in over45.itertuples()) + ". The manuscript's \"no class exceeding ~45% prevalence\" (l.416) is false.",
         )

    # ════════════════════════════════════════════════════════════════════════
    # 14. TABLE REPRODUCTION vs MANUSCRIPT (R1.m9, R3.M10) + misc numbers
    # ════════════════════════════════════════════════════════════════════════
    print("[14] tables ...")
    ms_t4 = {"XGBoost": (0.7998, 0.0854, 0.7498), "1D-CNN": (0.7828, 0.0885, 0.7431), "Transformer": (0.7728, 0.1022, 0.7350),
             "STA-LSTM-H": (0.7703, 0.0810, 0.7367), "GRU": (0.7675, 0.0931, 0.7309), "LSTM": (0.7653, 0.0884, 0.7267),
             "STA-LSTM": (0.7611, 0.0873, 0.7205)}
    ms_t3 = {"STA-LSTM-H": (0.7703, 0.9865, 0.9927), "STA-LSTM": (0.7611, 0.9865, 0.9932), "LSTM": (0.7653, 0.9867, 0.9933),
             "GRU": (0.7675, 0.9868, 0.9934), "Transformer": (0.7728, 0.9866, 0.9929), "1D-CNN": (0.7828, 0.9864, 0.9931)}
    ms_t1 = {"A": (0.8393, 0.0339, 0.8091), "B": (0.7935, 0.0444, 0.7197), "C": (0.7616, 0.0847, 0.7287), "D": (0.7465, 0.0807, 0.7178)}
    tr_rows = []
    for m, (am, asd, f1) in ms_t4.items():
        tr_rows.append(dict(table="4 (tab:loao_convergence)", row=m, ms_acc=am, acc=ACC[m].mean(), ms_sd=asd, sd=ACC[m].std(),
                            ms_f1=f1, f1=MET[(MET.input == "accel") & (MET.model == m)].macro_f1.mean()))
    for m, vals in ms_t3.items():
        for inp, v in zip(["accel", "lag1", "bp"], vals):
            tr_rows.append(dict(table="3 (tab:loao_ablation)", row=f"{m} {inp}", ms_acc=v,
                                acc=MET[(MET.input == inp) & (MET.model == m)].accuracy.mean()))
    for p_, (am, asd, f1) in ms_t1.items():
        tr_rows.append(dict(table="1 (tab:leakage)", row=p_, ms_acc=am, acc=LK[p_].mean(), ms_sd=asd, sd=LK[p_].std(), ms_f1=f1, f1=LKF[p_].mean()))
    TR = pd.DataFrame(tr_rows)
    TR["acc_mismatch"] = (TR.acc.round(4) - TR.ms_acc).abs() > 0.00011
    TR["sd_mismatch"] = (TR.sd.round(4) - TR.ms_sd).abs() > 0.00011
    TR["f1_mismatch"] = (TR.f1.round(4) - TR.ms_f1).abs() > 0.00011
    src_tr = save(TR, "tables_vs_manuscript.csv")
    # cohort prevalence table vs cache
    pv = pd.DataFrame({"animal": ANIMALS, **{f"pct_{c}": [np.mean(lab[a] == i) * 100 for a in ANIMALS] for i, c in enumerate(CLASSES)}})
    pv_m = (pv.set_index("animal") - prev.set_index("animal")[["pct_Other", "pct_Ruminating", "pct_Eating"]] * 100).abs().max().max()
    # the 0.64 vs 0.35 STA-LSTM-H vs STA-LSTM p
    cr = pd.read_csv(R / "comparison_report_loao.csv")
    hb2 = cr[cr.model.isin(["STA-LSTM-H", "STA-LSTM"])].pivot(index="held_out_animal", columns="model", values="accuracy_mean")
    p_bp_pair = stats.wilcoxon(hb2["STA-LSTM-H"], hb2["STA-LSTM"]).pvalue
    # input effects (R1.m7)
    ab = MET.groupby(["input", "model"]).accuracy.mean().unstack(0)
    ie_bp = (ab["bp"] - ab["accel"]).loc[NEURAL6]
    ie_l1 = (ab["lag1"] - ab["accel"]).loc[NEURAL6]
    note("R1.m9 / R3.M10 — table reproduction and hard-coded values",
         f"Source: {src_tr}, `table5_pairwise_vs_manuscript.csv`.",
         f"- Tables 1, 3, 4 vs recomputation (4-dp rounding): accuracy mismatches: {', '.join(TR[TR.acc_mismatch].table + ' ' + TR[TR.acc_mismatch].row) or 'none'}; SD mismatches: {', '.join((TR[TR.sd_mismatch.fillna(False)].table + ' ' + TR[TR.sd_mismatch.fillna(False)].row)) or 'none'}; macro-F1 mismatches: {', '.join((TR[TR.f1_mismatch.fillna(False)].table + ' ' + TR[TR.f1_mismatch.fillna(False)].row)) or 'none'}.",
         f"- Table 6 (tab:cohort_prevalence) vs `cohort_class_prevalence.csv` recomputed from the cache: max |Δ| = {pv_m:.2e} pp.",
         f"- `Results/paper_tables.md` \"STA-LSTM-H vs STA-LSTM p = 0.64\" comes from `wilcoxon_loao_18animals_effects.csv`, i.e. the **+behavior_prev** headline pair (recomputed here: p = {p_bp_pair:.4f}), not the accel-only pair (p = {pwi.loc['STA-LSTM-H vs STA-LSTM','p']:.4f}). It is a different comparison, not a stale value; its caption ('no-attention twin') is wrong because both arms have attention.",
         f"- `make_paper_figures.py` previously hard-coded 4.58 pp and (0.7998−0.7611); now computed from `leakage_protocol_per_animal.csv` and the per-animal accuracy matrix.",
         f"- Input effect (R1.m7): +bp − accel, mean over six models {ie_bp.mean()*100:.2f} pp (per model {ie_bp.min()*100:.1f}–{ie_bp.max()*100:.1f}); +bp_lag1 − accel {ie_l1.mean()*100:.2f} pp ({ie_l1.min()*100:.1f}–{ie_l1.max()*100:.1f}). Bounded above by predict_prev {G.predict_prev_1s.mean():.4f} and predict-two-back {G.predict_two_back_1s.mean():.4f}, both inflated by the 10-s grid (native-resolution predict-previous {G.predict_prev_10s_epoch.mean():.4f}).",
         "- Kalman negative result (R1.m10/R3.M8.1): NOT computable — `TrainConfig.use_kalman_for_lstmh` defaults to False and no saved run enabled it; in the existing code path the filter only modifies (x, y), never the class prediction.",
         )

    # ════════════════════════════════════════════════════════════════════════
    # write SUMMARY.md
    # ════════════════════════════════════════════════════════════════════════
    order = ["R1.M1 / R2.M1 / R3.M3 — 10-s label grid",
             "R1.M6 / R3.M7 / R3.m13 — persistence tests, all six neural models",
             "R1.M4 / R3.M4 / R3.m4 — metric dependence, per-class, confusion, class weighting",
             "R2.M3 / R1.M4 / R3.m4 / R2.m16 / R2.M6 — daily time budgets",
             "R1.M5 / R3.M6 — run-to-run noise and seed audit",
             "R1.M3 / R3.M1 — recipe of within-animal protocol-A results quoted by the reviewers",
             "R1.M8 / R1.M7 — heterogeneity, dependence, omnibus tests",
             "R1.M9 / R1.m9 / R3.M10 — 21-pair battery (Table 5 generator) and TOST sensitivity",
             "R1.m11 — prediction interval for a new animal",
             "R1.m13 / R3.m5 / R1.m2 / R1.m12 — leakage ladder extras",
             "R1.m6 / R3.m6 / R1.m4 — trivial floors under LOAO (newly computed)",
             "R2.M6 / R3.m7 — collar–halter temporal alignment (NEW finding)",
             "R2.M2 / R3.m3 / R1.M8 / R2.M6 — collar orientation, animal 08, per-animal LOAO",
             "R3.m7 / R2.M8c — temporal smoothing of predictions (POST HOC)",
             "R1.m1 — class-prevalence fact check (protocol A vs B folds)",
             "R1.m9 / R3.M10 — table reproduction and hard-coded values"]
    md = ["# Revision-2 statistics (Scientific Reports review, 2026-10-07)", "",
          "Generated by `scripts/revision2_stats.py` (deterministic; run `python scripts/revision2_stats.py`). Every number below is computed by that script from files on disk; CSV sources are given per section. Accuracy differences are in percentage points (pp). 'Exact p' = conditional sign-flip Wilcoxon (all 2^n sign patterns of observed mid-ranks, zeros dropped).", "",
          "Inputs: `Results/python_pipeline_loao{,_ablation}/animal-NN/loao_last_predictions.npy` (per-second LOAO predictions, 86,375 target seconds per animal, labels verified identical to the feature cache `behavior[25:]`), `Results/python_pipeline/feature_cache/`, `data/raw/halter-NN.csv`, `data/raw/Readme.md`, `Results/leakage_protocol_per_animal.csv`, `Results/python_pipeline{,_ablation,_seedaudit,_trivial}/`.", ""]
    for k in order:
        md.append(f"## {k}\n")
        md.extend(SUMMARY.get(k, ["(no output)"]))
        md.append("")
    (OUT / "SUMMARY.md").write_text("\n".join(md))
    print(f"done in {time.time()-t_start:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
