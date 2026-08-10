#!/usr/bin/env python
"""
Unified figure set for the paper (evaluation-bias decomposition).
One coherent visual system across all three figures:
  * no in-figure titles (captions carry the title)
  * one shared rcParams block (identical fonts/sizes on the page)
  * consistent accent semantics: warm ORANGE = the highlighted/focus quantity,
    BLUE = context; colourblind-safe Okabe-Ito, no green-among-blue
  * fig1 and fig2 share the dot + 95% CI idiom (no truncated-bar deception)
  * no colloquial annotations
All numbers from existing per-animal result files; no new experiments.
Outputs PNG+EPS into manuscript/2026_Scientific_Reports/ and Results/figures/.
"""
from __future__ import annotations
from pathlib import Path
import re, glob
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "Results"
OUT = [R / "figures"]
_ms = ROOT / "manuscript" / "2026_Scientific_Reports"
if _ms.exists():
    OUT.append(_ms)   # also refresh the local manuscript copy when present
for d in OUT: d.mkdir(parents=True, exist_ok=True)

# ---- one shared style: identical on-page type across all three figures ----
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 12,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#444444", "axes.linewidth": 0.8,
    "xtick.color": "#222222", "ytick.color": "#222222",
    "axes.labelcolor": "#222222", "text.color": "#222222",
    "figure.dpi": 200,
})
ACCENT = "#E69F00"   # warm orange: the highlighted / focus quantity
BLUE   = "#0072B2"   # context / baseline
LBLUE  = "#56B4E9"   # secondary context
GREY   = "#7f7f7f"
FIGW   = 6.6         # identical width (inches) -> identical on-page type at 0.8\linewidth

def save(fig, name):
    for d in OUT:
        fig.savefig(d / f"{name}.png", bbox_inches="tight", dpi=200)
        fig.savefig(d / f"{name}.eps", bbox_inches="tight")
    plt.close(fig); print("wrote", name)

def ci95(x):
    x = np.asarray(x, float); n = len(x)
    return stats.t.ppf(0.975, n-1) * x.std(ddof=1)/np.sqrt(n)

# ===================================================== FIG 1: leakage ladder
# dot + 95% CI (same idiom as fig2; no truncated bars)
lk = pd.read_csv(R / "leakage_protocol_per_animal.csv")
pmap = {"A_shuffled_overlap":"A","B_blocked_overlap":"B",
        "C_temporal_holdout":"C","D_nonoverlap_temporal":"D"}
lk["P"] = lk["protocol"].map(pmap)
piv = lk.pivot_table(index="animal", columns="P", values="accuracy")
rows = ["A","B","C","D"]
labels = ["A  shuffled overlap (leaky)","B  blocked overlap",
          "C  temporal holdout","D  non-overlap temporal"]
means = [piv[p].mean() for p in rows]
cis   = [ci95(piv[p]) for p in rows]
y = np.arange(len(rows))[::-1]                       # A at top
cols = [ACCENT, BLUE, BLUE, BLUE]

fig, ax = plt.subplots(figsize=(FIGW, 3.4))
ax.errorbar(means, y, xerr=cis, fmt="none", ecolor="#999999", lw=1.4, capsize=4, zorder=2)
ax.scatter(means, y, s=115, color=cols, edgecolor="white", linewidth=1.2, zorder=3)
for yi, m, c in zip(y, means, cis):
    ax.text(m + c + 0.003, yi, f"{m:.3f}", va="center", ha="left", fontsize=11)
ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=11)
ax.set_xlabel("Accuracy (mean, 95% CI, 18 animals)")
ax.set_xlim(0.71, 0.87)
# compact A-B leakage annotation between the two top rows (detail in caption)
yA, yB = y[0], y[1]
ax.annotate("", xy=(means[1], (yA+yB)/2), xytext=(means[0], (yA+yB)/2),
            arrowprops=dict(arrowstyle="<->", color="#333333", lw=1.4))
ax.text((means[0]+means[1])/2, (yA+yB)/2 - 0.16, "+4.6 pp",
        ha="center", va="top", fontsize=10.5, fontweight="bold", color="#333333")
ax.grid(axis="x", color="#e9e9e9", lw=0.8, zorder=0); ax.set_axisbelow(True)
ax.margins(y=0.22)
save(fig, "fig1_leakage")

# ===================================================== FIG 2: architecture
cr = pd.read_csv(R / "comparison_report_loao.csv")
five = cr[cr.model.isin(["LSTM","GRU","Transformer","1D-CNN","XGBoost"])].pivot_table(
    index="held_out_animal", columns="model", values="accuracy_mean")
rows2 = []
for f in glob.glob(str(R/"python_pipeline_loao_ablation/animal-*/loao_report.csv")):
    a = int(re.search(r"animal-(\d+)", f).group(1)); d = pd.read_csv(f)
    for m, disp in [("STA-LSTM-H (accel)","STA-LSTM-H"),("STA-LSTM (accel)","STA-LSTM")]:
        rr = d[d.model == m]
        if len(rr): rows2.append((a, disp, float(rr.accuracy_mean.iloc[0])))
sta = pd.DataFrame(rows2, columns=["animal","model","acc"]).pivot(index="animal", columns="model", values="acc")
acc = five.join(sta).dropna()
st = pd.DataFrame({"mean": acc.mean(), "ci": {m: ci95(acc[m]) for m in acc.columns}}).sort_values("mean")
order = st.index.tolist()
y = np.arange(len(order))
colors = [ACCENT if m == "XGBoost" else BLUE for m in order]

fig, ax = plt.subplots(figsize=(FIGW, 4.1))
ax.errorbar(st["mean"], y, xerr=st["ci"], fmt="none", ecolor="#999999", lw=1.4, capsize=4, zorder=2)
ax.scatter(st["mean"], y, s=115, color=colors, edgecolor="white", linewidth=1.2, zorder=3)
for yi, (m, row) in zip(y, st.iterrows()):
    ax.text(row["mean"] + row["ci"] + 0.002, yi, f"{row['mean']:.3f}", va="center", ha="left", fontsize=11)
ax.set_yticks(y); ax.set_yticklabels(order, fontsize=11)
ax.set_xlabel("Accelerometer-only cross-animal (LOAO) accuracy (mean, 95% CI, 18 animals)")
ax.set_xlim(0.72, 0.83)
lo, hi = st["mean"].min(), st["mean"].max()
ax.annotate("", xy=(lo, len(y)-0.3), xytext=(hi, len(y)-0.3),
            arrowprops=dict(arrowstyle="<->", color="#333333", lw=1.4))
ax.text((lo+hi)/2, len(y)-0.02, "full spread = 3.9 pp", ha="center", va="bottom",
        fontsize=10.5, color="#333333")
ax.grid(axis="x", color="#e9e9e9", lw=0.8, zorder=0); ax.set_axisbelow(True)
ax.margins(y=0.10)
save(fig, "fig2_architecture")

# ===================================================== FIG 3: effect magnitudes
abl = pd.read_csv(R / "aggregate_loao_ablation_summary.csv")
input_pp = (abl[abl.input_set=="bp"]["acc_mean"].mean() - abl[abl.input_set=="accel"]["acc_mean"].mean())*100
leak_pp  = 4.58
arch_pp  = (0.7998 - 0.7611)*100
levers = [("Input construction\n(behaviour-history channel)", input_pp, ACCENT, f"+{input_pp:.1f} pp"),
          ("Evaluation protocol\n(shuffle leakage, A−B)", leak_pp, BLUE, f"+{leak_pp:.1f} pp"),
          ("Architecture\n(full 7-model spread)", arch_pp, LBLUE, f"{arch_pp:.1f} pp spread")]
fig, ax = plt.subplots(figsize=(FIGW, 3.1))
y = np.arange(len(levers))[::-1]
for yi, (lab, val, c, txt) in zip(y, levers):
    ax.barh(yi, val, height=0.6, color=c, edgecolor="white", linewidth=1.0, zorder=3)
    ax.text(val + 0.4, yi, txt, va="center", ha="left", fontsize=11.5, fontweight="bold")
ax.set_yticks(y); ax.set_yticklabels([l for l,_,_,_ in levers], fontsize=10.5)
ax.set_xlabel("Effect on reported accuracy (percentage points)")
ax.set_xlim(0, 27)
ax.grid(axis="x", color="#e9e9e9", lw=0.8, zorder=0); ax.set_axisbelow(True)
save(fig, "fig3_levers")
print("input_pp=%.1f leak_pp=%.1f arch_pp=%.1f" % (input_pp, leak_pp, arch_pp))
