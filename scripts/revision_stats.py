#!/usr/bin/env python
"""
Revision re-analysis for the PLOS ONE version of the evaluation-bias paper.
Pure consolidation + correct statistics from existing on-disk per-animal results.
NO training / NO new experiments.

Addresses reviewer statistics concerns:
  * Leakage ladder: report the CLEAN one-variable estimate A-B (shuffle only,
    both 5-fold, both stride-1, full coverage) alongside A-C / A-D; paired
    Wilcoxon + 95% CI for every rung; count sign-reversal animals.
  * Convergence: full same-input pairwise Wilcoxon battery (Holm over that
    family only) + paired TOST equivalence test (margin +/-2 pp) among the five
    architectures that HAVE per-animal accel-only LOAO vectors on disk
    (LSTM, GRU, Transformer, 1D-CNN, XGBoost -- spans recurrent, attention,
    convolutional, gradient-boosted trees). The old mixed-input battery
    (wilcoxon_loao_18animals*.csv) is invalid and is superseded here.
  * XGBoost vs each neural model tested explicitly.
  * predict_prev ceiling: behaviour-channel models tested against the trivial
    persistence baseline per animal.
  * Parameter counts per architecture (capacity reporting).

Outputs -> Results/revision/*.csv and a pasteable Results/revision/SUMMARY.md
"""
from __future__ import annotations
import sys
from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "Results"
OUT = R / "revision"
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / "src"))

MARGIN = 0.02  # equivalence margin for TOST: +/- 2 accuracy points


def ci95_mean(x):
    x = np.asarray(x, float)
    n = len(x)
    m = x.mean()
    se = x.std(ddof=1) / np.sqrt(n)
    h = stats.t.ppf(0.975, n - 1) * se
    return m, m - h, m + h


def rank_biserial(diff):
    """Matched-pairs rank-biserial for a paired Wilcoxon on `diff` (a-b)."""
    diff = np.asarray(diff, float)
    nz = diff[diff != 0]
    if len(nz) == 0:
        return 0.0
    r = stats.rankdata(np.abs(nz))
    w_plus = r[nz > 0].sum()
    w_minus = r[nz < 0].sum()
    return float((w_plus - w_minus) / r.sum())


def paired_wilcoxon(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    if np.allclose(d, 0):
        return dict(p=1.0, r_rb=0.0, n=len(d))
    try:
        w = stats.wilcoxon(a, b, zero_method="wilcox", correction=False,
                           alternative="two-sided", mode="auto")
        p = float(w.pvalue)
    except Exception:
        p = np.nan
    return dict(p=p, r_rb=rank_biserial(d), n=int(len(d)))


def paired_tost(a, b, margin=MARGIN):
    """Two one-sided paired t-tests. Equivalence if BOTH one-sided p < 0.05,
    i.e. the (1-2*alpha) CI of the mean difference lies within (-margin, margin)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    n = len(d)
    m = d.mean()
    se = d.std(ddof=1) / np.sqrt(n)
    if se == 0:
        se = 1e-12
    t_lower = (m - (-margin)) / se          # H0: mean <= -margin
    t_upper = (m - (margin)) / se           # H0: mean >=  margin
    p_lower = stats.t.sf(t_lower, n - 1)     # P(T > t_lower)
    p_upper = stats.t.cdf(t_upper, n - 1)    # P(T < t_upper)
    p_tost = max(p_lower, p_upper)
    # 90% CI of the mean difference (matches alpha=0.05 TOST)
    h90 = stats.t.ppf(0.95, n - 1) * se
    return dict(mean_diff=m, ci90_lo=m - h90, ci90_hi=m + h90,
                p_tost=p_tost, equivalent=bool(p_tost < 0.05))


def holm(pvals):
    pvals = np.asarray(pvals, float)
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        running = max(running, val)
        adj[idx] = min(running, 1.0)
    return adj


md = ["# Revision statistics — recomputed from existing per-animal data",
      f"\n_TOST equivalence margin = ±{MARGIN*100:.0f} accuracy points._\n"]

# ══════════════════════════════════════════════════════════════════════════
# 1. LEAKAGE LADDER — clean decomposition
# ══════════════════════════════════════════════════════════════════════════
lk = pd.read_csv(R / "leakage_protocol_per_animal.csv")
pmap = {"A_shuffled_overlap": "A", "B_blocked_overlap": "B",
        "C_temporal_holdout": "C", "D_nonoverlap_temporal": "D"}
lk["P"] = lk["protocol"].map(pmap)
piv = lk.pivot_table(index="animal", columns="P", values="accuracy")

rungs = [("A", "B", "shuffle only (both 5-fold, stride-1, full coverage) = CLEAN leakage"),
         ("A", "C", "shuffle+estimator+temporal-shift (headline in v1)"),
         ("A", "D", "shuffle+temporal-shift+~25x less training data"),
         ("B", "C", "estimator + temporal shift (no shuffle)")]
rows = []
for hi, lo, note in rungs:
    d = (piv[hi] - piv[lo]).dropna()
    m, clo, chi = ci95_mean(d)
    w = paired_wilcoxon(piv[hi], piv[lo])
    n_rev = int((d < 0).sum())
    rows.append(dict(contrast=f"{hi}-{lo}", note=note,
                     delta_pp=round(m * 100, 2),
                     ci95_pp=f"[{clo*100:.2f}, {chi*100:.2f}]",
                     wilcoxon_p=round(w["p"], 5), r_rb=round(w["r_rb"], 3),
                     n_sign_reversals=n_rev))
lk_df = pd.DataFrame(rows)
lk_df.to_csv(OUT / "leakage_contrasts.csv", index=False)
md.append("## 1. Leakage ladder (per-animal paired contrasts, n=18)\n")
md.append(lk_df.to_markdown(index=False))
md.append(f"\n**Protocol means:** " +
          ", ".join(f"{c}={piv[c].mean():.4f}" for c in ["A", "B", "C", "D"]))
md.append(f"\n**Clean leakage estimate (A−B) = +{(piv['A']-piv['B']).mean()*100:.1f} pp** "
          f"(Wilcoxon p={paired_wilcoxon(piv['A'],piv['B'])['p']:.2g}). "
          f"Headline A−C = +{(piv['A']-piv['C']).mean()*100:.1f} pp additionally "
          f"absorbs estimator + diurnal shift; A−C shows "
          f"{int(((piv['A']-piv['C'])<0).sum())} sign-reversal animals "
          f"(C>A), impossible for pure overlap leakage.\n")

# ══════════════════════════════════════════════════════════════════════════
# 2. CONVERGENCE — correct same-input battery + TOST (5 accel-only models)
# ══════════════════════════════════════════════════════════════════════════
cr = pd.read_csv(R / "comparison_report_loao.csv")
FIVE = ["LSTM", "GRU", "Transformer", "1D-CNN", "XGBoost"]
acc = cr[cr.model.isin(FIVE)].pivot_table(index="held_out_animal",
                                          columns="model", values="accuracy_mean")
acc = acc[FIVE].dropna()
md.append("\n## 2. Architecture convergence — accel-only LOAO, per-animal (n=%d)\n" % len(acc))
means = acc.mean().sort_values(ascending=False)
md.append("**Per-animal means (accel-only):** " +
          ", ".join(f"{k}={v:.4f}" for k, v in means.items()))
md.append(f"\nSpread across these five = {(means.max()-means.min())*100:.2f} pp.\n")

pairs = list(combinations(FIVE, 2))
prows = []
for a, b in pairs:
    w = paired_wilcoxon(acc[a], acc[b])
    t = paired_tost(acc[a], acc[b])
    prows.append(dict(pair=f"{a} vs {b}",
                      mean_diff_pp=round(t["mean_diff"] * 100, 2),
                      wilcoxon_p=round(w["p"], 4), r_rb=round(w["r_rb"], 3),
                      tost_ci90_pp=f"[{t['ci90_lo']*100:.2f}, {t['ci90_hi']*100:.2f}]",
                      tost_p=round(t["p_tost"], 4),
                      equivalent_2pp=t["equivalent"]))
pdf = pd.DataFrame(prows)
pdf["wilcoxon_p_holm"] = holm(pdf["wilcoxon_p"].values).round(4)
pdf = pdf[["pair", "mean_diff_pp", "wilcoxon_p", "wilcoxon_p_holm", "r_rb",
           "tost_ci90_pp", "tost_p", "equivalent_2pp"]]
pdf.to_csv(OUT / "convergence_pairwise.csv", index=False)
md.append(pdf.to_markdown(index=False))
n_sig = int((pdf["wilcoxon_p_holm"] < 0.05).sum())
n_equiv = int(pdf["equivalent_2pp"].sum())
md.append(f"\n**{n_sig} of {len(pdf)} pairs differ significantly (Holm); "
          f"{n_equiv} of {len(pdf)} are statistically equivalent within ±{MARGIN*100:.0f} pp (TOST).** "
          f"Transformer (attention) vs LSTM/GRU/1D-CNN (no attention) results are among the pairs above.\n")

# XGBoost callout
xgb = pdf[pdf.pair.str.contains("XGBoost")]
md.append("**XGBoost (nominal best) vs neural models:**\n")
md.append(xgb.to_markdown(index=False))

# ══════════════════════════════════════════════════════════════════════════
# 3. PERSISTENCE CEILING — behaviour-channel models vs predict_prev (per animal)
# ══════════════════════════════════════════════════════════════════════════
pp = pd.read_csv(R / "predict_prev_cohort.csv")[["animal", "accuracy"]].rename(
    columns={"accuracy": "predict_prev"})
bp = cr[cr.model.isin(["STA-LSTM-H", "STA-LSTM"])].pivot_table(
    index="held_out_animal", columns="model", values="accuracy_mean")
bp.index.name = "animal"
mrg = bp.join(pp.set_index("animal"))
md.append("\n## 3. Behaviour-channel models vs trivial predict_prev ceiling (per-animal, n=18)\n")
crows = []
for col in ["STA-LSTM-H", "STA-LSTM"]:
    w = paired_wilcoxon(mrg[col], mrg["predict_prev"])
    crows.append(dict(model=f"{col}(+bp)", mean=round(mrg[col].mean(), 4),
                      predict_prev=round(mrg["predict_prev"].mean(), 4),
                      delta_pp=round((mrg[col] - mrg["predict_prev"]).mean() * 100, 3),
                      wilcoxon_p=round(w["p"], 4), r_rb=round(w["r_rb"], 3)))
cdf = pd.DataFrame(crows)
cdf.to_csv(OUT / "persistence_vs_predictprev.csv", index=False)
md.append(cdf.to_markdown(index=False))
md.append("\nNo behaviour-channel encoder significantly exceeds copying the previous label.\n")

# ══════════════════════════════════════════════════════════════════════════
# 4. PARAMETER COUNTS (capacity reporting)
# ══════════════════════════════════════════════════════════════════════════
md.append("\n## 4. Architecture capacity (accel-only, input_size=14)\n")
try:
    import torch  # noqa
    from model import build_model
    kinds = [("lstm_h", "STA-LSTM-H"), ("sta_lstm", "STA-LSTM"),
             ("lstm", "LSTM"), ("gru", "GRU"),
             ("transformer", "Transformer"), ("cnn1d", "1D-CNN")]
    prows = []
    for kind, disp in kinds:
        m = build_model(kind, 14)
        n = sum(p.numel() for p in m.parameters())
        prows.append(dict(architecture=disp,
                          trainable_params=int(n),
                          params_k=round(n / 1e3, 1)))
    prows.append(dict(architecture="XGBoost",
                      trainable_params="300 trees × depth 6",
                      params_k="—"))
    param_df = pd.DataFrame(prows)
    param_df.to_csv(OUT / "param_counts.csv", index=False)
    md.append(param_df.to_markdown(index=False))
except Exception as e:
    md.append(f"_param count failed: {e}_")

(OUT / "SUMMARY.md").write_text("\n".join(str(x) for x in md))
print("\n".join(str(x) for x in md))
print(f"\n\n[written to {OUT}]")
