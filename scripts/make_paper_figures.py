#!/usr/bin/env python
"""
All five figures for the paper (evaluation-bias decomposition), one coherent
visual system: no in-figure titles (captions carry the title), one shared
rcParams block, consistent accent semantics (warm ORANGE = highlighted/focus
quantity, BLUE = context), colourblind-safe Okabe-Ito.

  fig_persistence  : behaviour autocorrelation + bout-length distribution --
                     the mechanism behind BOTH the 0.99 predict-previous ceiling
                     and the window-overlap leakage (from the raw halter labels)
  fig_leakage      : protocol ladder A-D, clean A-B=+4.6pp contrast (dot + 95% CI)
  fig_input        : each architecture's jump from accel-only onto the
                     predict-previous ceiling once a behaviour channel is added
  fig_architecture : accel-only LOAO accuracy for all 7 models (dot + 95% CI)
  fig_levers       : the three effect magnitudes on one percentage-point axis

All numbers from existing result files (+ raw halter labels for fig_persistence);
no new experiments. Outputs PNG+EPS into Results/figures/ (and the local
manuscript copy when present).
"""
from __future__ import annotations
from pathlib import Path
import re, glob
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "Results"; RAW = ROOT / "data" / "raw"
OUT = [R / "figures"]
_ms = ROOT / "manuscript" / "2026_Scientific_Reports"
if _ms.exists(): OUT.append(_ms)   # refresh local manuscript copy when present
for d in OUT: d.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 12,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#444444", "axes.linewidth": 0.8,
    "xtick.color": "#222222", "ytick.color": "#222222",
    "axes.labelcolor": "#222222", "text.color": "#222222", "figure.dpi": 200,
})
ACCENT="#E69F00"; BLUE="#0072B2"; LBLUE="#56B4E9"; GREEN="#009E73"; GREY="#7f7f7f"
FIGW=6.6
def save(fig,name):
    for d in OUT:
        fig.savefig(d/f"{name}.png",bbox_inches="tight",dpi=200)
        fig.savefig(d/f"{name}.eps",bbox_inches="tight")
    plt.close(fig); print("wrote",name)
def ci95(x):
    x=np.asarray(x,float); n=len(x); return stats.t.ppf(0.975,n-1)*x.std(ddof=1)/np.sqrt(n)

# ===================================================== FIG: persistence mechanism
SEC=86400; ROWS=SEC*10
def per_second_labels(aid):
    f=RAW/f"halter-{aid:02d}.csv"
    if not f.exists(): return None
    c=pd.read_csv(f,usecols=["classification"],nrows=ROWS)["classification"].to_numpy()
    n=(len(c)//10)*10
    return stats.mode(c[:n].reshape(-1,10),axis=1,keepdims=False).mode.astype(int)

print("reading halter labels for persistence figure (first 24h x 18 animals)...")
seqs=[per_second_labels(a) for a in range(1,19)]; seqs=[s for s in seqs if s is not None]
maxlag=600
sp=np.zeros(maxlag+1)
for lag in range(1,maxlag+1):
    num=den=0
    for s in seqs:
        num+=int((s[:-lag]==s[lag:]).sum()); den+=len(s)-lag
    sp[lag]=num/den
p1=sp[1]
bouts=np.concatenate([np.diff(np.r_[0,np.flatnonzero(np.diff(s))+1,len(s)]) for s in seqs])
med=float(np.median(bouts))

fig,axs=plt.subplots(1,2,figsize=(9.6,3.5))
ax=axs[0]; lags=np.arange(1,maxlag+1)
ax.axvspan(1,25,color=ACCENT,alpha=0.14,zorder=0)
ax.plot(lags,sp[1:],color=BLUE,lw=2,zorder=2)
ax.scatter([1],[p1],color=ACCENT,s=70,zorder=3,edgecolor="white",linewidth=1)
ax.annotate(f"at 1 s: {p1:.3f}\n(= predict-previous ceiling)",xy=(1,p1),xytext=(95,0.90),
            fontsize=9.5,arrowprops=dict(arrowstyle="->",color="#333"))
ax.text(13,0.62,"25-s\nwindow",fontsize=8.5,color="#9a6b00",ha="center",va="center")
ax.set_xlabel("Time lag (s)"); ax.set_ylabel("P(same behaviour after lag)")
ax.set_xlim(0,maxlag); ax.set_ylim(0.5,1.0)
ax.grid(color="#ececec",lw=0.8); ax.set_axisbelow(True)
ax=axs[1]
b=np.sort(bouts); surv=1-np.arange(1,len(b)+1)/len(b)
ax.plot(b,surv,color=GREEN,lw=2)
ax.axvline(med,color=GREY,ls="--",lw=1.2)
ax.text(med*1.25,0.6,f"median\nbout = {med:.0f} s",fontsize=9.5,color="#333")
ax.set_xscale("log"); ax.set_xlabel("Behaviour-bout length (s, log scale)")
ax.set_ylabel("P(bout longer than x)"); ax.set_xlim(1,b.max()); ax.set_ylim(0,1)
ax.grid(color="#ececec",lw=0.8); ax.set_axisbelow(True)
save(fig,"fig_persistence")
print(f"  P(same|1s)={p1:.4f}  median bout={med:.0f}s")

# ===================================================== FIG: leakage ladder
lk=pd.read_csv(R/"leakage_protocol_per_animal.csv")
pm={"A_shuffled_overlap":"A","B_blocked_overlap":"B","C_temporal_holdout":"C","D_nonoverlap_temporal":"D"}
lk["P"]=lk.protocol.map(pm); piv=lk.pivot_table(index="animal",columns="P",values="accuracy")
rows=["A","B","C","D"]; labels=["A  shuffled overlap (leaky)","B  blocked overlap","C  temporal holdout","D  non-overlap temporal"]
means=[piv[p].mean() for p in rows]; cis=[ci95(piv[p]) for p in rows]
y=np.arange(4)[::-1]; cols=[ACCENT,BLUE,BLUE,BLUE]
fig,ax=plt.subplots(figsize=(FIGW,3.4))
ax.errorbar(means,y,xerr=cis,fmt="none",ecolor="#999",lw=1.4,capsize=4,zorder=2)
ax.scatter(means,y,s=115,color=cols,edgecolor="white",linewidth=1.2,zorder=3)
for yi,m,c in zip(y,means,cis): ax.text(m+c+0.003,yi,f"{m:.3f}",va="center",ha="left",fontsize=11)
ax.set_yticks(y); ax.set_yticklabels(labels,fontsize=11)
ax.set_xlabel("Accuracy (mean, 95% CI, 18 animals)"); ax.set_xlim(0.71,0.87)
yA,yB=y[0],y[1]
ax.annotate("",xy=(means[1],(yA+yB)/2),xytext=(means[0],(yA+yB)/2),arrowprops=dict(arrowstyle="<->",color="#333",lw=1.4))
ax.text((means[0]+means[1])/2,(yA+yB)/2-0.16,"+4.6 pp",ha="center",va="top",fontsize=10.5,fontweight="bold",color="#333")
ax.grid(axis="x",color="#e9e9e9",lw=0.8); ax.set_axisbelow(True); ax.margins(y=0.22)
save(fig,"fig_leakage")

# ===================================================== FIG: input slopegraph
abl=pd.read_csv(R/"aggregate_loao_ablation_summary.csv")
order=["accel","bp_lag1","bp"]; xlab=["accel-\nonly","+ lag-1\nchannel","+ same-step\nchannel"]
archs=["STA-LSTM-H","STA-LSTM","LSTM","GRU","Transformer","1D-CNN"]
pp=pd.read_csv(R/"predict_prev_cohort.csv")["accuracy"].mean()
fig,ax=plt.subplots(figsize=(FIGW,4.0)); x=np.arange(3)
for m in archs:
    ys=[abl[(abl.architecture==m)&(abl.input_set==s)]["acc_mean"].iloc[0] for s in order]
    ax.plot(x,ys,marker="o",ms=6,lw=1.8,color=BLUE,alpha=0.7)
ax.axhline(pp,color=ACCENT,ls="--",lw=1.6)
ax.text(2.5,pp-0.006,f"predict-previous\nceiling ({pp:.3f})",va="top",ha="right",fontsize=9.5,color="#9a6b00")
ax.text(1.4,0.82,"each line: one of\n6 neural architectures",fontsize=9.5,color=BLUE,ha="left")
ax.set_xticks(x); ax.set_xticklabels(xlab,fontsize=10.5)
ax.set_ylabel("Cross-animal (LOAO) accuracy"); ax.set_ylim(0.74,1.005); ax.set_xlim(-0.2,2.75)
ax.grid(axis="y",color="#e9e9e9",lw=0.8); ax.set_axisbelow(True)
save(fig,"fig_input")

# ===================================================== FIG: architecture
cr=pd.read_csv(R/"comparison_report_loao.csv")
five=cr[cr.model.isin(["LSTM","GRU","Transformer","1D-CNN","XGBoost"])].pivot_table(index="held_out_animal",columns="model",values="accuracy_mean")
r2=[]
for f in glob.glob(str(R/"python_pipeline_loao_ablation/animal-*/loao_report.csv")):
    a=int(re.search(r"animal-(\d+)",f).group(1)); d=pd.read_csv(f)
    for m,disp in [("STA-LSTM-H (accel)","STA-LSTM-H"),("STA-LSTM (accel)","STA-LSTM")]:
        rr=d[d.model==m]
        if len(rr): r2.append((a,disp,float(rr.accuracy_mean.iloc[0])))
sta=pd.DataFrame(r2,columns=["animal","model","acc"]).pivot(index="animal",columns="model",values="acc")
acc=five.join(sta).dropna()
st=pd.DataFrame({"mean":acc.mean(),"ci":{m:ci95(acc[m]) for m in acc.columns}}).sort_values("mean")
o=st.index.tolist(); y=np.arange(len(o))
colors=[ACCENT if m=="XGBoost" else BLUE for m in o]
fig,ax=plt.subplots(figsize=(FIGW,4.1))
ax.errorbar(st["mean"],y,xerr=st["ci"],fmt="none",ecolor="#999",lw=1.4,capsize=4,zorder=2)
ax.scatter(st["mean"],y,s=115,color=colors,edgecolor="white",linewidth=1.2,zorder=3)
for yi,(m,row) in zip(y,st.iterrows()): ax.text(row["mean"]+row["ci"]+0.002,yi,f"{row['mean']:.3f}",va="center",ha="left",fontsize=11)
ax.set_yticks(y); ax.set_yticklabels(o,fontsize=11)
ax.set_xlabel("Accelerometer-only cross-animal (LOAO) accuracy (mean, 95% CI, 18 animals)"); ax.set_xlim(0.72,0.83)
lo,hi=st["mean"].min(),st["mean"].max()
ax.annotate("",xy=(lo,len(y)-0.3),xytext=(hi,len(y)-0.3),arrowprops=dict(arrowstyle="<->",color="#333",lw=1.4))
ax.text((lo+hi)/2,len(y)-0.02,"full spread = 3.9 pp",ha="center",va="bottom",fontsize=10.5,color="#333")
ax.grid(axis="x",color="#e9e9e9",lw=0.8); ax.set_axisbelow(True); ax.margins(y=0.10)
save(fig,"fig_architecture")

# ===================================================== FIG: levers
input_pp=(abl[abl.input_set=="bp"]["acc_mean"].mean()-abl[abl.input_set=="accel"]["acc_mean"].mean())*100
levers=[("Input construction\n(behaviour-history channel)",input_pp,ACCENT,f"+{input_pp:.1f} pp"),
        ("Evaluation protocol\n(shuffle leakage, A-B)",4.58,BLUE,"+4.6 pp"),
        ("Architecture\n(full 7-model spread)",(0.7998-0.7611)*100,LBLUE,"3.9 pp spread")]
fig,ax=plt.subplots(figsize=(FIGW,3.1)); y=np.arange(len(levers))[::-1]
for yi,(lab,val,c,txt) in zip(y,levers):
    ax.barh(yi,val,height=0.6,color=c,edgecolor="white",linewidth=1.0,zorder=3)
    ax.text(val+0.4,yi,txt,va="center",ha="left",fontsize=11.5,fontweight="bold")
ax.set_yticks(y); ax.set_yticklabels([l for l,_,_,_ in levers],fontsize=10.5)
ax.set_xlabel("Effect on reported accuracy (percentage points)"); ax.set_xlim(0,27)
ax.grid(axis="x",color="#e9e9e9",lw=0.8); ax.set_axisbelow(True)
save(fig,"fig_levers")
print("all 5 figures written.")
