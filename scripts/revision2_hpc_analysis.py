#!/usr/bin/env python
"""scripts/revision2_hpc_analysis.py — analysis of the 2026-10-07 revision HPC runs.

Reads only committed outputs:
  Results/revision2_hpc/leakage_ladder/per_fold/*.json   (revision2_leakage_ladder.py)
  Results/revision2_hpc/loao_controls/*.{json,npz}       (revision2_loao_variant.py)
  Results/revision2_hpc/xgb_controls/*.{json,npz}        (revision2_xgb.py)
  Results/leakage_protocol_per_animal.csv                (original 5-epoch ladder, Slurm 165856)
  Results/python_pipeline_loao/animal-NN/loao_last_predictions.npy (headline LOAO run)
  Results/python_pipeline/feature_cache/*.csv            (labels, for persistence baselines)
  Results/predict_prev_cohort.csv

Statistics (as in the paper): per-animal paired differences (n = 18 unless
stated), mean difference in pp with a paired-t 95% CI, two-sided Wilcoxon
signed-rank p (exact when there are no zero/tied |d|, otherwise scipy's
normal approximation — the column `p_method` says which), matched-pairs
rank-biserial r_rb, number of animals in which the second arm scored higher
("reversals"), and the per-animal range of the difference. Holm correction is
applied within each named family (column `family`).

Writes CSVs + SUMMARY.md into Results/revision2_hpc/.
"""
from __future__ import annotations

import glob
import warnings
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "Results"
OUT = R / "revision2_hpc"
METRICS = ("accuracy", "balanced_accuracy", "f1_macro")


# ─────────────────────────────────────────────────────────────────────────────
def metr(y, p):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
    return {"accuracy": accuracy_score(y, p),
            "balanced_accuracy": balanced_accuracy_score(y, p),
            "f1_macro": f1_score(y, p, average="macro", zero_division=0)}


def paired(a: pd.Series, b: pd.Series, label_a: str, label_b: str, metric: str,
           family: str) -> dict:
    """a − b per animal (index = animal)."""
    idx = a.index.intersection(b.index)
    d = (a.loc[idx] - b.loc[idx]).to_numpy(float)
    n = len(d)
    out = {"family": family, "metric": metric, "A": label_a, "B": label_b, "n": n,
           "mean_A": float(a.loc[idx].mean()), "mean_B": float(b.loc[idx].mean())}
    if n < 2:
        return out
    m, se = d.mean(), d.std(ddof=1) / np.sqrt(n)
    t = stats.t.ppf(0.975, n - 1)
    nz = d[d != 0]
    ties = len(nz) < n or len(np.unique(np.abs(nz))) < len(nz)
    try:
        if not ties and len(nz) <= 25:
            p = stats.wilcoxon(nz, method="exact").pvalue; meth = "exact"
        else:
            p = stats.wilcoxon(nz, method="approx").pvalue; meth = "approx(ties)"
    except ValueError:
        p, meth = np.nan, "n/a"
    r = stats.rankdata(np.abs(nz))
    rrb = (r[nz > 0].sum() - r[nz < 0].sum()) / r.sum() if len(nz) else np.nan
    out.update(diff_pp=100 * m, ci_lo_pp=100 * (m - t * se), ci_hi_pp=100 * (m + t * se),
               sd_diff_pp=100 * d.std(ddof=1), p=p, p_method=meth, r_rb=rrb,
               reversals=int((d < 0).sum()) if m > 0 else int((d > 0).sum()),
               zeros=int((d == 0).sum()),
               min_pp=100 * d.min(), max_pp=100 * d.max())
    return out


def holm(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["p_holm"] = np.nan
    for (fam, met), g in df.groupby(["family", "metric"]):
        g = g.dropna(subset=["p"]).sort_values("p")
        m = len(g)
        adj = np.maximum.accumulate([(m - i) * p for i, p in enumerate(g["p"])])
        df.loc[g.index, "p_holm"] = np.minimum(adj, 1.0)
    return df


def fmt_p(p):
    if pd.isna(p):
        return "n/a"
    return f"{p:.1e}" if p < 1e-3 else f"{p:.3f}"


def table(df: pd.DataFrame, cols: list[str]) -> str:
    def cell(v):
        if isinstance(v, float):
            return "n/a" if np.isnan(v) else f"{v:.4f}" if abs(v) < 1.5 else f"{v:.1f}"
        return str(v)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(cell(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def contrast_md(df: pd.DataFrame) -> str:
    rows = []
    for _, r in df.iterrows():
        rows.append(f"| {r['family']} | {r['A']} − {r['B']} | {r['metric']} | {r['diff_pp']:+.1f} "
                    f"[{r['ci_lo_pp']:+.1f}, {r['ci_hi_pp']:+.1f}] | {r['sd_diff_pp']:.1f} | "
                    f"{fmt_p(r['p'])} ({r['p_method']}) | {fmt_p(r.get('p_holm', np.nan))} | "
                    f"{r['r_rb']:+.2f} | {r['reversals']}/{r['n']} | "
                    f"{r['min_pp']:+.1f} … {r['max_pp']:+.1f} |")
    head = ("| family | Contrast | metric | Δ pp [95% CI] | SD(d) pp | Wilcoxon p | p_Holm | r_rb | "
            "reversals | per-animal range pp |\n|---|---|---|---|---|---|---|---|---|---|")
    return head + "\n" + "\n".join(rows)


def load_json(pattern):
    rows = []
    for f in sorted(glob.glob(str(pattern))):
        d = json.load(open(f))
        d["_file"] = f
        rows.append(d)
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Leakage ladder
# ─────────────────────────────────────────────────────────────────────────────
def ladder(md: list[str]):
    d = load_json(OUT / "leakage_ladder/per_fold/*.json")
    if d.empty:
        md.append("## 1. Leakage ladder\n\nNo results.\n"); return
    exp_folds = {"A": 5, "B": 5, "C": 1, "D": 1, "Deq": 1}
    per = (d.groupby(["model", "protocol", "animal"])
             .agg(**{m: (m, "mean") for m in METRICS}, folds=("fold", "nunique"),
                  epochs=("epochs", "first"), n_train=("n_train", "mean"))
             .reset_index())
    # keep only complete (animal, protocol) cells
    need = per["protocol"].map(lambda p: exp_folds.get(p, 5))
    incomplete = per[per["folds"] < need]
    per = per[per["folds"] >= need]
    per.to_csv(OUT / "ladder_per_animal.csv", index=False)
    summ = (per.groupby(["model", "protocol"])
               .agg(n_animals=("animal", "nunique"),
                    acc_mean=("accuracy", "mean"), acc_sd=("accuracy", "std"),
                    balacc_mean=("balanced_accuracy", "mean"),
                    f1_mean=("f1_macro", "mean"), f1_sd=("f1_macro", "std"),
                    epochs=("epochs", "first"), n_train=("n_train", "mean"))
               .reset_index())
    summ.to_csv(OUT / "ladder_summary.csv", index=False)

    old = pd.read_csv(R / "leakage_protocol_per_animal.csv")
    old["protocol"] = old["protocol"].str[0]
    oldp = old.pivot(index="animal", columns="protocol", values="accuracy")

    rows = []
    for model in per["model"].unique():
        pm = {m: per[per.model == model].pivot(index="animal", columns="protocol", values=m)
              for m in METRICS}
        fam = f"ladder:{model}"
        pairs = [("A", "B"), ("A", "C"), ("A", "D"), ("B", "C"), ("Deq", "D"),
                 ("A", "M_L1_g0"), ("M_L1_g0", "M_L25_g0"), ("M_L25_g0", "M_L25_g26"),
                 ("M_L25_g26", "M_L300_g26"), ("M_L300_g26", "M_L3600_g26"),
                 ("M_L3600_g26", "M_L17280_g26"), ("M_L25_g26", "M_L17280_g26"),
                 ("M_L1_g0", "M_L17280_g26"), ("M_L17280_g26", "B")]
        for e in (5, 10, 20):
            pairs.append((f"A_e{e}", f"B_e{e}"))
        for m in METRICS:
            P = pm[m]
            for a, b in pairs:
                if a in P and b in P and P[a].notna().sum() >= 2:
                    rows.append(paired(P[a].dropna(), P[b].dropna(), a, b, m, fam))
        if model.startswith("STA"):
            P = pm["accuracy"]
            for a in ("A", "B", "C", "D"):
                if a in P:
                    rows.append(paired(P[a], oldp[a], f"{a} (40 ep)", f"{a} (orig 5 ep)",
                                       "accuracy", f"recipe:{model}"))
            if "A" in P and "B" in P:
                rows.append(paired(P["A"] - P["B"], oldp["A"] - oldp["B"],
                                   "A−B (40 ep)", "A−B (orig 5 ep)", "accuracy",
                                   f"recipe:{model}"))
    con = holm(pd.DataFrame(rows))
    con.to_csv(OUT / "ladder_contrasts.csv", index=False)

    md.append("## 1. Leakage ladder at the stated recipe (R1.M2, R1.M3, R3.M1, R3.M2)\n")
    md.append("Recipe: accel-only STA-LSTM-H (`build_model('lstm_h', 14)`, same model as the "
              "original ladder), Adam 2e-3, wd 1e-4, clip 1.2, **batch 64, 40 epochs, "
              "multi-task MSE+0.35·CE with the real pseudo-position target**, weighted CE "
              "(audit_class_balance on the training partition), z-score on the training "
              "partition. XGBoost (trainer._eval_xgboost hyper-parameters, unweighted) run on "
              "the same splits. Per-animal value = mean over folds; n = 18 animals.\n")
    if len(incomplete):
        md.append(f"**Incomplete cells excluded:** {len(incomplete)} (animal, protocol) cells "
                  f"lacking folds: {sorted(set(incomplete.protocol))}.\n")
    s = summ.copy()
    s["orig_5ep_acc"] = s["protocol"].map(oldp.mean()) .where(s["model"].str.startswith("STA"))
    md.append(table(s, ["model", "protocol", "n_animals", "epochs", "n_train", "acc_mean",
                        "acc_sd", "balacc_mean", "f1_mean", "orig_5ep_acc"]) + "\n")
    md.append("\nPaired contrasts (accuracy unless stated; Holm within model × metric):\n")
    md.append(contrast_md(con[con.metric == "accuracy"]) + "\n")
    md.append("\nBalanced accuracy / macro-F1 contrasts are in `ladder_contrasts.csv`.\n")
    return per


# ─────────────────────────────────────────────────────────────────────────────
# 2/3. LOAO controls
# ─────────────────────────────────────────────────────────────────────────────
def headline_loao() -> pd.DataFrame:
    rows = []
    for h in range(1, 19):
        f = R / f"python_pipeline_loao/animal-{h:02d}/loao_last_predictions.npy"
        if not f.exists():
            continue
        D = np.load(f, allow_pickle=True).item()
        for disp, key in (("1D-CNN", "cnn1d"), ("LSTM", "lstm"), ("XGBoost", "xgboost"),
                          ("GRU", "gru"), ("Transformer", "transformer")):
            if disp in D:
                rows.append({"animal": h, "arm": f"{key}|headline", **metr(D[disp]["true_cls"],
                                                                          D[disp]["pred_cls"])})
    return pd.DataFrame(rows)


def minutes_bias(true, pred):
    """predicted − true minutes per (≈24 h) record for Ruminating (1) and Eating (2)."""
    t = np.bincount(true, minlength=3) / 60.0
    p = np.bincount(pred, minlength=3) / 60.0
    return p[1] - t[1], p[2] - t[2]


def loao(md: list[str]):
    nn = load_json(OUT / "loao_controls/*.json")
    xg = load_json(OUT / "xgb_controls/*.json")
    arms = []
    if not nn.empty:
        for _, r in nn[nn.split == "loao"].iterrows():
            z = np.load(r["_file"].replace(".json", ".npz"))
            rb, eb = minutes_bias(z["true"].astype(int), z["pred"].astype(int))
            arms.append({"animal": r["split_id"],
                         "arm": f"{r['model']}|{r['objective']}|{r['ce_weighting']}|s{r['seed']}",
                         **{m: r[m] for m in METRICS}, "rum_min_bias": rb, "eat_min_bias": eb})
    if not xg.empty:
        for _, r in xg[xg["mode"] == "loao"].iterrows():
            z = np.load(r["_file"].replace(".json", ".npz"))
            rb, eb = minutes_bias(z["true"].astype(int), z["pred"].astype(int))
            arms.append({"animal": r["split_id"], "arm": f"xgboost|{r['variant']}",
                         **{m: r[m] for m in METRICS}, "rum_min_bias": rb, "eat_min_bias": eb})
    A = pd.concat([pd.DataFrame(arms), headline_loao()], ignore_index=True)
    A.to_csv(OUT / "loao_controls_per_animal.csv", index=False)
    S = (A.groupby("arm").agg(n=("animal", "nunique"),
                              acc=("accuracy", "mean"), acc_sd=("accuracy", "std"),
                              balacc=("balanced_accuracy", "mean"),
                              f1=("f1_macro", "mean"),
                              rum_min_bias=("rum_min_bias", "mean"),
                              eat_min_bias=("eat_min_bias", "mean"),
                              rum_min_mae=("rum_min_bias", lambda x: np.abs(x).mean()),
                              eat_min_mae=("eat_min_bias", lambda x: np.abs(x).mean()))
          .reset_index())
    S.to_csv(OUT / "loao_controls_summary.csv", index=False)

    def vec(arm, m):
        g = A[A.arm == arm].set_index("animal")[m]
        return g if len(g) else None

    # seed-mean arms
    for base in ("cnn1d|multitask|weighted", "lstm|multitask|weighted", "cnn1d|multitask|none"):
        sub = A[A.arm.str.startswith(base + "|s")]
        if sub["arm"].nunique() >= 2:
            g = sub.groupby("animal")[list(METRICS)].mean().reset_index()
            g["arm"] = base + "|seedmean"
            A = pd.concat([A, g], ignore_index=True)

    fam_w = [("xgboost|balanced", "xgboost|none"),
             ("cnn1d|multitask|none|s1", "cnn1d|multitask|weighted|s1"),
             ("lstm|multitask|none|s1", "lstm|multitask|weighted|s1"),
             ("cnn1d|ce_only|weighted|s1", "cnn1d|multitask|weighted|s1"),
             ("cnn1d|ce_only|none|s1", "cnn1d|multitask|none|s1"),
             ("cnn1d|ce_only|none|s1", "cnn1d|multitask|weighted|s1"),
             ("xgboost|none", "xgboost|headline")]
    fam_x = [("xgboost|none", "cnn1d|multitask|weighted|seedmean"),
             ("xgboost|none", "lstm|multitask|weighted|seedmean"),
             ("xgboost|none", "cnn1d|multitask|none|s1"),
             ("xgboost|none", "lstm|multitask|none|s1"),
             ("xgboost|none", "cnn1d|ce_only|none|s1"),
             ("xgboost|balanced", "cnn1d|multitask|weighted|seedmean"),
             ("xgboost|balanced", "lstm|multitask|weighted|seedmean"),
             ("xgboost|balanced", "cnn1d|ce_only|weighted|s1"),
             ("cnn1d|multitask|none|seedmean", "cnn1d|multitask|weighted|seedmean"),
             ("xgboost|none", "cnn1d|headline"), ("xgboost|none", "lstm|headline")]
    rows = []
    for fam, pairs in (("weighting/objective (one factor)", fam_w),
                       ("XGBoost vs neural (matched controls)", fam_x)):
        for a, b in pairs:
            for m in METRICS:
                va, vb = vec(a, m), vec(b, m)
                if va is not None and vb is not None:
                    rows.append(paired(va, vb, a, b, m, fam))
    con = holm(pd.DataFrame(rows))
    con.to_csv(OUT / "loao_controls_contrasts.csv", index=False)

    md.append("## 2. Class-weighting and objective controls, accel-only LOAO (R1.M4, R3.M4)\n")
    md.append("Arms: `model|objective|ce_weighting|seed` (new runs, CPU, explicit seed), "
              "`xgboost|variant` (new), `*|headline` (re-scored from the saved headline LOAO "
              "predictions). Minute biases = predicted − true minutes per ~24 h record "
              "(mean over animals); MAE = mean |bias|.\n")
    md.append(table(S, ["arm", "n", "acc", "acc_sd", "balacc", "f1", "rum_min_bias",
                        "eat_min_bias", "rum_min_mae", "eat_min_mae"]) + "\n")
    md.append("\n### One-factor contrasts\n")
    md.append(contrast_md(con[con.family.str.startswith("weighting")]) + "\n")
    md.append("\n### XGBoost vs neural under matched controls\n")
    md.append(contrast_md(con[con.family.str.startswith("XGBoost")]) + "\n")

    # loss components
    if not nn.empty:
        lt = []
        for _, r in nn.iterrows():
            for t in r["loss_trace"] or []:
                lt.append({"arm": f"{r['split']}|{r['model']}|{r['objective']}|{r['ce_weighting']}",
                           **t})
        L = pd.DataFrame(lt)
        if len(L):
            L["mse_share"] = L["mse"] / (L["mse"] + L["alpha_ce"])
            G = (L.groupby(["arm", "epoch"])[["mse", "ce", "alpha_ce", "mse_share"]]
                   .mean().reset_index())
            G.to_csv(OUT / "loss_components.csv", index=False)
            md.append("\n### Measured loss components (16k-window training subsample, eval mode; "
                      "MSE on the min–max-normalised target)\n")
            md.append(table(G[G.arm.str.contains("multitask")],
                            ["arm", "epoch", "mse", "ce", "alpha_ce", "mse_share"]) + "\n")
    return A


def seeds(md: list[str], A: pd.DataFrame):
    md.append("## 3a. Seed replicates, accel-only LOAO (R1.M5, R3.M6)\n")
    rows = []
    for base in ("cnn1d|multitask|weighted", "lstm|multitask|weighted", "cnn1d|multitask|none"):
        sub = A[A.arm.str.match(base.replace("|", r"\|") + r"\|s\d$")]
        W = sub.pivot(index="animal", columns="arm", values="accuracy")
        if W.shape[1] < 2:
            continue
        head = base.split("|")[0] + "|headline"
        incl_head = base.endswith("weighted") and head in set(A.arm)
        if incl_head:
            W = W.join(A[A.arm == head].set_index("animal")["accuracy"].rename(head))
        W = W.dropna()
        within_sd = W.std(axis=1, ddof=1)
        cols = list(W.columns)
        diffs = np.concatenate([(W[a] - W[b]).to_numpy() for i, a in enumerate(cols)
                                for b in cols[i + 1:]])
        rows.append({"arms": base, "replicates": ", ".join(c.split("|")[-1] for c in cols),
                     "n_animals": len(W),
                     "cohort_means": ", ".join(f"{W[c].mean():.4f}" for c in cols),
                     "range_of_cohort_means_pp": 100 * (W.mean().max() - W.mean().min()),
                     "mean_within_animal_sd_pp": 100 * within_sd.mean(),
                     "max_within_animal_sd_pp": 100 * within_sd.max(),
                     "sd_pairwise_replicate_diff_pp": 100 * diffs.std(ddof=1),
                     "max_abs_replicate_diff_pp": 100 * np.abs(diffs).max(),
                     "implied_se_cohort_mean_diff_pp": 100 * diffs.std(ddof=1) / np.sqrt(len(W))})
    T = pd.DataFrame(rows)
    T.to_csv(OUT / "seed_replicates.csv", index=False)
    if len(T):
        md.append("Replicates are independent trainings of the same (model, held-out animal); "
                  "`headline` = the original GPU LOAO run (different RNG stream & hardware).\n")
        md.append(table(T, list(T.columns)) + "\n")
    else:
        md.append("No seed results.\n")


def xgb_extra(md: list[str], A: pd.DataFrame):
    xg = load_json(OUT / "xgb_controls/*.json")
    md.append("## 3c. XGBoost with the behaviour-history channel (R1.m8)\n")
    # persistence baselines from caches
    pp = pd.read_csv(R / "predict_prev_cohort.csv").set_index("animal")["accuracy"]
    lag2 = {}
    for h in range(1, 19):
        y = pd.read_csv(R / f"python_pipeline/feature_cache/animal-{h:02d}_1s_features.csv",
                        usecols=["behavior"])["behavior"].to_numpy(int)[:86400]
        t = np.arange(25, len(y))           # target rows of the stride-1 windows
        lag2[h] = float(np.mean(y[t] == y[t - 2]))
    lag2 = pd.Series(lag2)
    rows = []
    for arm, ref, lab in (("xgboost|bp", pp, "predict_prev (lag-1 copy)"),
                          ("xgboost|bp_lag1", lag2, "lag-2 copy")):
        v = A[A.arm == arm].set_index("animal")["accuracy"]
        if len(v):
            rows.append(paired(v, ref, arm, lab, "accuracy", "behaviour channel"))
    if rows:
        C = pd.DataFrame(rows)
        C.to_csv(OUT / "xgb_behaviour_channel.csv", index=False)
        md.append(f"Lag-2 persistence (copy label t−2), cohort mean = {lag2.mean():.4f}; "
                  f"predict_prev = {pp.mean():.4f}.\n")
        md.append(contrast_md(C) + "\n")

    def per_animal_from(df, mode):
        out = []
        for _, r in df[df["mode"] == mode].iterrows():
            if isinstance(r.get("per_animal"), dict):
                for a, mm in r["per_animal"].items():
                    out.append({"animal": int(a), "variant": r.get("variant", "none"),
                                "accuracy": mm["accuracy"],
                                "balanced_accuracy": mm["balanced_accuracy"],
                                "f1_macro": mm["f1_macro"]})
            else:
                out.append({"animal": int(r["split_id"]), "variant": r.get("variant", "none"),
                            **{m: r[m] for m in METRICS}})
        return pd.DataFrame(out, columns=["animal", "variant", *METRICS])

    # LOTO
    md.append("\n## 3b. Leave-one-trial-out (R3.m2, R2.M2d, R1.M8)\n")
    nn = load_json(OUT / "loao_controls/*.json")
    lrows = []
    if not xg.empty:
        L = per_animal_from(xg, "loto")
        for v, g in L.groupby("variant"):
            lrows.append(("xgboost|" + v, g.set_index("animal"), f"xgboost|{v}"))
    if not nn.empty:
        sub = nn[nn.split == "loto"].copy()
        sub["arm"] = sub["model"] + "|" + sub["objective"] + "|" + sub["ce_weighting"]
        for arm, grp in sub.groupby("arm"):
            parts = []
            for _, r in grp.iterrows():
                g = pd.DataFrame(r["per_animal"]).T
                g.index = g.index.astype(int)
                parts.append(g)
            seed = grp["seed"].iloc[0]
            lrows.append((arm, pd.concat(parts), f"{arm}|s{seed}"))
    rows, crow = [], []
    trial_of = {a: k for k, v in {1: range(1, 7), 2: range(7, 11), 3: range(11, 19)}.items() for a in v}
    for name, g, loao_arm in lrows:
        g = g[list(METRICS)].astype(float)
        la = A[A.arm == loao_arm].set_index("animal")["accuracy"]
        for k in (1, 2, 3):
            idx = [a for a in g.index if trial_of[a] == k]
            rows.append({"arm": name, "trial": k, "n_animals": len(idx),
                         "loto_acc": g.loc[idx, "accuracy"].mean(),
                         "loto_balacc": g.loc[idx, "balanced_accuracy"].mean(),
                         "loao_acc_same_animals": la.reindex(idx).mean() if len(la) else np.nan})
        if len(la) and len(g) == 18:
            crow.append(paired(la, g["accuracy"], f"LOAO {loao_arm}", f"LOTO {name}",
                               "accuracy", "LOAO vs LOTO"))
    if rows:
        T = pd.DataFrame(rows)
        T.to_csv(OUT / "loto_summary.csv", index=False)
        md.append(table(T, list(T.columns)) + "\n")
    if crow:
        C = holm(pd.DataFrame(crow))
        C.to_csv(OUT / "loto_contrasts.csv", index=False)
        md.append("\n" + contrast_md(C) + "\n")

    # pooled
    md.append("\n## 3e. Pooled shuffled 5-fold CV across animals vs LOAO, XGBoost (R2.M7)\n")
    if not xg.empty and (xg["mode"] == "pooled").any():
        P = per_animal_from(xg, "pooled")
        pa = P.groupby("animal")[list(METRICS)].mean()   # 5 folds' test windows per animal
        nf = xg[xg["mode"] == "pooled"].shape[0]
        la = A[A.arm == "xgboost|none"].set_index("animal")["accuracy"]
        if len(la):
            C = pd.DataFrame([paired(pa["accuracy"], la, f"pooled shuffled CV ({nf}/5 folds)",
                                     "LOAO", "accuracy", "pooled vs LOAO")])
            C.to_csv(OUT / "pooled_vs_loao.csv", index=False)
            md.append(f"Pooled-CV per-animal accuracy = mean over the {nf} completed folds' "
                      "test windows of that animal.\n")
            md.append(contrast_md(C) + "\n")
    else:
        md.append("Not available.\n")

    # day 2
    md.append("\n## 3d. Day-2 sensitivity, XGBoost LOAO (R3.m1)\n")
    if not xg.empty and (xg["mode"] == "day2").any():
        D2 = per_animal_from(xg, "day2")
        rows = []
        for v, g in D2.groupby("variant"):
            g = g.set_index("animal")
            d1 = A[A.arm == f"xgboost|{v}"].set_index("animal")
            for m in METRICS:
                rows.append(paired(g[m], d1[m], f"day 2 xgboost|{v}", f"day 1 xgboost|{v}",
                                   m, "day2 vs day1"))
        C = holm(pd.DataFrame(rows))
        C.to_csv(OUT / "day2_vs_day1.csv", index=False)
        md.append(contrast_md(C) + "\n")
    else:
        md.append("Not available.\n")


def main():
    md = ["# Revision-2 HPC experiments — summary (auto-generated by "
          "`scripts/revision2_hpc_analysis.py`)\n",
          "All numbers below are computed from the per-task JSON/NPZ outputs in this folder "
          "and the committed headline result files; see the script header for the statistics. "
          "CIs are paired-t 95%; p are two-sided Wilcoxon signed-rank; r_rb = matched-pairs "
          "rank-biserial; 'reversals' = animals whose sign opposes the mean.\n"]
    ladder(md)
    A = loao(md)
    seeds(md, A)
    xgb_extra(md, A)
    (OUT / "SUMMARY.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
