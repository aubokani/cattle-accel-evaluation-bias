#!/usr/bin/env python
"""
All figures for the paper (evaluation-bias decomposition), one coherent
visual system: no in-figure titles (captions carry the title), one shared
rcParams block, consistent accent semantics (warm ORANGE = highlighted/focus
quantity, BLUE = context), colourblind-safe Okabe-Ito; trials are encoded by
colour AND marker shape.

  fig_persistence  : (a) lag-k label agreement, pooled + class-specific, log lag
                     axis to 6 h; (b) run-length histogram showing the 10-s label
                     grid; (c) class-specific run-length survival
  fig_leakage      : protocol ladder A-D (dot + 95% CI), A-B contrast computed
  fig_input        : accel-only -> +lag-1 -> +same-step channel, broken y-axis,
                     predict-previous and predict-two-back baselines
  fig_architecture : (a) accel-only LOAO accuracy, 18 per-animal points coloured by
                     trial + mean/95% CI; (b) paired difference model - XGBoost
  fig_levers       : the three effect magnitudes (all computed, none hard-coded)
  fig_timebudget   : Bland-Altman of daily time budgets (XGBoost vs 1D-CNN)   [suppl.]
  fig_peranimal    : per-animal heat-map, 7 LOAO models + 4 ladder protocols [suppl.]

Inputs: existing result files plus Results/revision2/*.csv, so run
`python scripts/revision2_stats.py` first. No new experiments.
Outputs PNG+EPS into Results/figures/ (and the local manuscript copy when present).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "Results"; R2 = R / "revision2"
OUT = [R / "figures"]
_ms = ROOT / "manuscript" / "2026_Scientific_Reports"
if _ms.exists(): OUT.append(_ms)   # refresh local manuscript copy when present
for d in OUT: d.mkdir(parents=True, exist_ok=True)
if not (R2 / "accel7_metrics_per_animal.csv").exists():
    raise SystemExit("run scripts/revision2_stats.py first (needs Results/revision2/)")

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 12,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#444444", "axes.linewidth": 0.8,
    "xtick.color": "#222222", "ytick.color": "#222222",
    "axes.labelcolor": "#222222", "text.color": "#222222", "figure.dpi": 200,
})
ACCENT="#E69F00"; BLUE="#0072B2"; LBLUE="#56B4E9"; GREEN="#009E73"; GREY="#7f7f7f"
VERM="#D55E00"; PURPLE="#CC79A7"
TRIAL={a:(1 if a<=6 else 2 if a<=10 else 3) for a in range(1,19)}
TCOL={1:GREEN,2:VERM,3:LBLUE}; TMARK={1:"o",2:"s",3:"^"}
TLAB={1:"Trial 1 (Jun 2015)",2:"Trial 2 (Sep 2015)",3:"Trial 3 (Aug-Oct 2016)"}
FIGW=6.6
def save(fig,name):
    for d in OUT:
        fig.savefig(d/f"{name}.png",bbox_inches="tight",dpi=200)
        fig.savefig(d/f"{name}.eps",bbox_inches="tight")
    plt.close(fig); print("wrote",name)
def ci95(x):
    x=np.asarray(x,float); n=len(x); return stats.t.ppf(0.975,n-1)*x.std(ddof=1)/np.sqrt(n)
def grid(ax,axis="both"):
    ax.grid(axis=axis,color="#ececec",lw=0.8); ax.set_axisbelow(True)

# ===================================================== FIG: persistence mechanism
lag=pd.read_csv(R2/"lag_persistence_curve.csv")
hist=pd.read_csv(R2/"run_length_histogram_1s.csv")
rl=pd.read_csv(R2/"run_lengths_interior_1s.csv")
fig,axs=plt.subplots(1,3,figsize=(14.0,3.9),gridspec_kw=dict(wspace=0.32))
ax=axs[0]
ax.axvspan(1,25,color=ACCENT,alpha=0.14,zorder=0)
for col,c,lab,ls in [("p_same_given_Other",GREY,"Other","--"),("p_same_given_Ruminating",GREEN,"Ruminating","--"),
                     ("p_same_given_Eating",PURPLE,"Eating","--"),("p_same_pooled",BLUE,"all classes","-")]:
    ax.plot(lag.lag_s,lag[col],color=c,lw=2.2 if ls=="-" else 1.4,ls=ls,label=lab,zorder=2)
lv=lag.set_index("lag_s").p_same_pooled
for k,txt,xy in [(1,f"1 s: {lv[1]:.3f}\n(predict-previous)",(1.6,0.55)),(10,f"10 s: {lv[10]:.3f}\n(one label epoch)",(40,0.93))]:
    ax.scatter([k],[lv[k]],color=ACCENT,s=55,zorder=4,edgecolor="white",linewidth=1)
    ax.annotate(txt,xy=(k,lv[k]),xytext=xy,fontsize=9,arrowprops=dict(arrowstyle="->",color="#333",lw=0.9))
ax.text(5,0.06,"25-s\nwindow",fontsize=8.5,color="#9a6b00",ha="center")
ax.set_xscale("log"); ax.set_xlim(1,lag.lag_s.max()); ax.set_ylim(0.0,1.02)
ax.set_xticks([1,10,60,600,3600,21600]); ax.set_xticklabels(["1 s","10 s","1 min","10 min","1 h","6 h"])
ax.set_xlabel("Time lag (log scale)"); ax.set_ylabel("P(same label after lag)")
ax.legend(fontsize=8.5,frameon=False,loc="lower left",bbox_to_anchor=(0.0,0.13)); grid(ax)
ax=axs[1]
h=hist[hist.run_length_s<=150]
mult=(h.run_length_s%10==0)
ax.bar(h.run_length_s[~mult],h.count_all[~mult],width=0.9,color=LBLUE,zorder=3)
ax.bar(h.run_length_s[mult],h.count_all[mult],width=0.9,color=BLUE,zorder=3)
share=np.mean(rl.run_length_s%10==0)
ax.text(0.97,0.95,f"{share*100:.1f}% of label runs are\nexact multiples of 10 s",transform=ax.transAxes,
        ha="right",va="top",fontsize=9.5)
ax.set_xlim(0,152); ax.set_xticks(range(0,151,20))
ax.set_xlabel("Label-run length (s)"); ax.set_ylabel("Number of runs (18 animals)"); grid(ax,"y")
ax=axs[2]
for ci,(nm,c) in enumerate([("Other",GREY),("Ruminating",GREEN),("Eating",PURPLE)]):
    b=np.sort(rl.run_length_s[rl["class"]==ci].to_numpy()); surv=1-np.arange(1,len(b)+1)/len(b)
    ax.step(b,surv,where="post",color=c,lw=1.8,label=f"{nm} (median {np.median(b):.0f} s)")
ax.set_xscale("log"); ax.set_xlim(8,rl.run_length_s.max()); ax.set_ylim(0,1)
ax.set_xticks([10,30,100,300,1000,3000]); ax.set_xticklabels(["10","30","100","300","1000","3000"])
ax.set_xlabel("Label-run length (s, log scale)"); ax.set_ylabel("P(run longer than x)")
ax.legend(fontsize=8.5,frameon=False,loc="upper right"); grid(ax)
for a_,l_ in zip(axs,"abc"): a_.text(-0.16,1.04,l_,transform=a_.transAxes,fontsize=13,fontweight="bold")
save(fig,"fig_persistence")

# ===================================================== FIG: leakage ladder
# original 5-epoch run (Slurm 165856) + recipe-matched re-run (40 epochs, batch 64,
# multi-task; Results/revision2_hpc, scripts/revision2_hpc_analysis.py) + XGBoost.
lk=pd.read_csv(R/"leakage_protocol_per_animal.csv")
pm={"A_shuffled_overlap":"A","B_blocked_overlap":"B","C_temporal_holdout":"C","D_nonoverlap_temporal":"D"}
lk["P"]=lk.protocol.map(pm); piv=lk.pivot_table(index="animal",columns="P",values="accuracy")
rows=["A","B","C","D"]; labels=["A  shuffled overlap","B  contiguous blocks","C  temporal holdout","D  non-overlap temporal"]
ab_pp=(piv["A"]-piv["B"]).mean()*100            # original 5-epoch A-B
R3=R/"revision2_hpc"
lp=pd.read_csv(R3/"ladder_per_animal.csv")
def lpiv(model): return lp[lp.model==model].pivot_table(index="animal",columns="protocol",values="accuracy")
p40=lpiv("STA-LSTM-H (accel)"); pxg=lpiv("XGBoost")
ab40=(p40["A"]-p40["B"]).mean()*100; abxg=(pxg["A"]-pxg["B"]).mean()*100
ovl40=(p40["M_L25_g0"]-p40["M_L25_g26"]).mean()*100
series=[("STA-LSTM-H, 40 epochs (stated recipe)",p40,ACCENT,"o",-0.22),
        ("STA-LSTM-H, 5 epochs (original run)",piv,GREY,"s",0.0),
        ("XGBoost",pxg,BLUE,"D",0.22)]
y=np.arange(4)[::-1].astype(float)
fig,ax=plt.subplots(figsize=(FIGW,4.0))
for lab,P,c,mk,dy in series:
    m=[P[r].mean() for r in rows]; ci=[ci95(P[r]) for r in rows]
    ax.errorbar(m,y-dy,xerr=ci,fmt="none",ecolor=c,alpha=0.6,lw=1.2,capsize=3,zorder=2)
    ax.scatter(m,y-dy,s=60,color=c,marker=mk,edgecolor="white",linewidth=0.9,zorder=3,label=lab)
ax.set_yticks(y); ax.set_yticklabels(labels,fontsize=10.5)
ax.set_xlabel("Within-animal accuracy (mean, 95% CI, 18 animals)")
ax.set_xlim(0.66,1.0); grid(ax,"x"); ax.set_ylim(-0.7,4.6)
ax.legend(fontsize=8.5,frameon=False,loc="upper left",bbox_to_anchor=(0.0,1.0),ncol=1)
ax.text(0.99,0.03,f"A$-$B: +{ab40:.1f} pp (40 ep), +{ab_pp:.1f} pp (5 ep), +{abxg:.1f} pp (XGBoost)",
        transform=ax.transAxes,ha="right",va="bottom",fontsize=8.5,color="#333")
save(fig,"fig_leakage")

# ===================================================== FIG: input slopegraph (broken axis)
met=pd.read_csv(R2/"all_models_metrics_per_animal.csv")
g=pd.read_csv(R2/"label_grid_per_animal.csv")
pp1=g.predict_prev_1s.mean(); pp2=g.predict_two_back_1s.mean()
order=["accel","lag1","bp"]; xlab=["accel-\nonly","+ lag-1\nchannel (t$-$2)","+ same-step\nchannel (t$-$1)"]
archs=["STA-LSTM-H","STA-LSTM","LSTM","GRU","Transformer","1D-CNN"]
mm=met.groupby(["input","model"]).accuracy.mean()
fig,(axt,axb)=plt.subplots(2,1,sharex=True,figsize=(FIGW,4.4),gridspec_kw=dict(height_ratios=[1.25,1],hspace=0.08))
x=np.arange(3)
for m in archs:
    ys=[mm[(s,m)] for s in order]
    for ax in (axt,axb):
        ax.plot(x[:2],ys[:2],lw=0.9,ls=":",color=BLUE,alpha=0.5)          # crosses the axis break
        ax.plot(x[1:],ys[1:],lw=1.6,color=BLUE,alpha=0.7)
        ax.plot(x,ys,ls="",marker="o",ms=6,color=BLUE,alpha=0.8)
axt.axhline(pp1,color=ACCENT,ls="--",lw=1.5); axt.axhline(pp2,color=ACCENT,ls=":",lw=1.5)
axt.text(-0.15,pp1+0.0003,f"predict-previous ({pp1:.4f})",va="bottom",ha="left",fontsize=9,color="#9a6b00")
axt.text(-0.15,pp2+0.0003,f"predict-two-back ({pp2:.4f})",va="bottom",ha="left",fontsize=9,color="#9a6b00")
axt.set_ylim(0.9845,0.9958); axb.set_ylim(0.752,0.792)
axt.spines["bottom"].set_visible(False); axt.tick_params(bottom=False)
d=0.012
for ax,yy in [(axt,0),(axb,1)]:
    kw=dict(transform=ax.transAxes,color="#444",clip_on=False,lw=0.9)
    ax.plot((-d,+d),(yy-d*1.5,yy+d*1.5),**kw)
axb.text(1.35,0.787,"each line: one of\n6 neural architectures",fontsize=9,color=BLUE,ha="left",va="top")
axb.set_xticks(x); axb.set_xticklabels(xlab,fontsize=10)
fig.text(0.0,0.5,"Cross-animal (LOAO) accuracy",rotation=90,va="center",ha="right",fontsize=12)
for ax in (axt,axb): grid(ax,"y"); ax.set_xlim(-0.2,2.25)
save(fig,"fig_input")

# ===================================================== FIG: architecture (per-animal + paired diffs)
acc=met[met.input=="accel"].pivot(index="animal",columns="model",values="accuracy")
st=pd.DataFrame({"mean":acc.mean(),"ci":{m:ci95(acc[m]) for m in acc.columns}}).sort_values("mean")
o=st.index.tolist(); yy=np.arange(len(o))
fig,(ax,ax2)=plt.subplots(1,2,figsize=(11.0,4.4),sharey=True,gridspec_kw=dict(width_ratios=[1.45,1],wspace=0.08))
rng=np.random.default_rng(1)
for yi,m in zip(yy,o):
    for a in acc.index:
        ax.scatter(acc.loc[a,m],yi+rng.uniform(-0.22,0.22),s=22,color=TCOL[TRIAL[a]],marker=TMARK[TRIAL[a]],
                   alpha=0.85,edgecolor="white",linewidth=0.4,zorder=2)
ax.errorbar(st["mean"],yy,xerr=st["ci"],fmt="none",ecolor="#222",lw=1.6,capsize=5,zorder=4)
ax.scatter(st["mean"],yy,s=95,marker="D",color=[ACCENT if m=="XGBoost" else "#222" for m in o],edgecolor="white",linewidth=1,zorder=5)
a8=acc.loc[8].min()
ax.annotate("animal 08",xy=(acc.loc[8,o[0]],0),xytext=(0.47,-0.85),fontsize=8.5,color=VERM,
            arrowprops=dict(arrowstyle="-",color=VERM,lw=0.7))
ax.set_yticks(yy); ax.set_yticklabels([f"{m}  {st.loc[m,'mean']:.3f}" for m in o],fontsize=11)
ax.set_xlim(0.40,0.92); ax.set_ylim(-1.1,len(o)-0.4)
ax.set_xlabel("Accelerometer-only LOAO accuracy\n(points: 18 animals; diamond: mean, 95% CI)")
hd=[plt.Line2D([],[],ls="",marker=TMARK[t],color=TCOL[t],label=TLAB[t],ms=6) for t in (1,2,3)]
ax.legend(handles=hd,fontsize=9,frameon=False,loc="lower left",bbox_to_anchor=(0.0,1.0),ncol=3,handletextpad=0.2,columnspacing=1.0)
grid(ax,"x")
for yi,m in zip(yy,o):
    if m=="XGBoost":
        ax2.text(0.15,yi,"reference",ha="left",va="center",fontsize=9.5,color="#9a6b00"); continue
    dd=(acc[m]-acc["XGBoost"])*100; mu=dd.mean(); c=ci95(dd)
    ax2.errorbar([mu],[yi],xerr=[c],fmt="none",ecolor="#999",lw=1.4,capsize=4,zorder=2)
    ax2.scatter([mu],[yi],s=80,color=BLUE,edgecolor="white",linewidth=1,zorder=3)
    ax2.text(mu-c-0.15,yi,f"{mu:+.1f}",ha="right",va="center",fontsize=9.5)
ax2.axvline(0,color=ACCENT,lw=1.3,ls="--")
ax2.set_xlim(-7.0,2.3)
ax2.set_xlabel("Paired difference vs XGBoost (pp)\n(mean of per-animal differences, 95% CI)")
grid(ax2,"x"); ax2.tick_params(left=False)
for a_,l_ in zip((ax,ax2),"ab"): a_.text(-0.02 if l_=="b" else -0.40,1.02,l_,transform=a_.transAxes,fontsize=13,fontweight="bold")
save(fig,"fig_architecture")

# ===================================================== FIG: levers (all values computed)
input_pp=(mm["bp"].loc[archs].mean()-mm["accel"].loc[archs].mean())*100
arch_spread=(st["mean"].max()-st["mean"].min())*100
levers=[("Input construction\n(oracle behaviour-history channel)",input_pp,ACCENT,f"+{input_pp:.1f} pp"),
        ("Protocol, neural (A$-$B,\nshuffled vs blocked, 40 epochs)",ab40,BLUE,f"+{ab40:.1f} pp"),
        ("Protocol, overlap only\n(neural, purge vs keep)",ovl40,BLUE,f"+{ovl40:.1f} pp"),
        ("Protocol, XGBoost (A$-$B)",abxg,BLUE,f"+{abxg:.1f} pp"),
        ("Architecture\n(max$-$min of 7 model means)",arch_spread,LBLUE,f"{arch_spread:.1f} pp range")]
fig,ax=plt.subplots(figsize=(FIGW,4.2)); y=np.arange(len(levers))[::-1]
for yi,(lab,val,c,txt) in zip(y,levers):
    ax.barh(yi,val,height=0.6,color=c,edgecolor="white",linewidth=1.0,zorder=3)
    ax.text(val+0.4,yi,txt,va="center",ha="left",fontsize=11.5,fontweight="bold")
ax.set_yticks(y); ax.set_yticklabels([l for l,_,_,_ in levers],fontsize=10.5)
ax.set_xlabel("Effect on reported accuracy (percentage points)"); ax.set_xlim(0,input_pp*1.25)
grid(ax,"x")
save(fig,"fig_levers")

# ===================================================== FIG: time budget Bland-Altman (suppl.)
tb=pd.read_csv(R2/"time_budget_per_animal.csv")
fig,axs=plt.subplots(1,3,figsize=(13.5,4.0),gridspec_kw=dict(wspace=0.28))
for ax,b in zip(axs,["Ruminating","Eating","Other"]):
    for m,c,mk,dx in [("XGBoost",ACCENT,"o",0),("1D-CNN",BLUE,"s",1)]:
        s=tb[(tb.model==m)&(tb.behaviour==b)]
        xm=(s.predicted_min_per_day+s.halter_min_per_day)/2; e=s.error_min_per_day
        bias=e.mean(); sd=e.std(ddof=1)
        ax.scatter(xm,e,color=c,marker=mk,s=34,edgecolor="white",linewidth=0.6,zorder=3,
                   label=f"{m}: bias {bias:+.0f}, LoA [{bias-1.96*sd:+.0f}, {bias+1.96*sd:+.0f}]")
        ax.axhline(bias,color=c,lw=1.4,zorder=2)
        for l_ in (bias-1.96*sd,bias+1.96*sd): ax.axhline(l_,color=c,lw=1.0,ls="--",zorder=2)
        r8=s[s.animal==8]
        if b!="Other": ax.annotate("08",xy=(float(xm[r8.index[0]]),float(e[r8.index[0]])),xytext=(4,-10 if dx==0 else 6),
                                   textcoords="offset points",fontsize=8,color=c)
    ax.axhline(0,color="#444",lw=0.8)
    ax.set_xlabel("Mean of halter and predicted (min/day)")
    ax.set_ylabel("Predicted $-$ halter (min/day)" if b=="Ruminating" else "")
    lim=max(abs(tb[tb.behaviour==b].error_min_per_day[tb.model.isin(["XGBoost","1D-CNN"])]).max()*1.15,
            abs(ax.get_ylim()[0]),abs(ax.get_ylim()[1]))
    ax.set_ylim(-lim*1.05,lim*1.45)
    ax.legend(fontsize=8,frameon=False,loc="upper left"); grid(ax)
    ax.text(0.98,0.03,b,transform=ax.transAxes,ha="right",va="bottom",fontsize=11,fontweight="bold")
save(fig,"fig_timebudget")

# ===================================================== FIG: per-animal heat-map (suppl.)
models=st.sort_values("mean",ascending=False).index.tolist()
H=acc[models].copy()
for p in rows: H[f"Ladder {p}"]=piv[p]
H=H.loc[sorted(H.index,key=lambda a:(TRIAL[a],a))]
fig,ax=plt.subplots(figsize=(9.0,7.4))
im=ax.imshow(H.values,cmap="Blues",vmin=0.4,vmax=1.0,aspect="auto")
for i in range(H.shape[0]):
    for j in range(H.shape[1]):
        v=H.values[i,j]; ax.text(j,i,f"{v:.2f}",ha="center",va="center",fontsize=8,color="white" if v>0.78 else "#222")
ax.set_xticks(range(H.shape[1])); ax.set_xticklabels(H.columns,rotation=40,ha="right",fontsize=9.5)
ax.set_yticks(range(H.shape[0])); ax.set_yticklabels([f"{a:02d}  (T{TRIAL[a]})" for a in H.index],fontsize=9.5)
for i,a in enumerate(H.index): ax.get_yticklabels()[i].set_color(TCOL[TRIAL[a]] if TRIAL[a]!=3 else BLUE)
for k in (5.5,9.5): ax.axhline(k,color="#222",lw=1.2)
ax.axvline(len(models)-0.5,color="#222",lw=1.2)
ax.text((len(models)-1)/2,-1.0,"accelerometer-only LOAO",ha="center",fontsize=10)
ax.text(len(models)+1.5,-1.0,"within-animal ladder",ha="center",fontsize=10)
for s_ in ax.spines.values(): s_.set_visible(False)
cb=fig.colorbar(im,ax=ax,fraction=0.035,pad=0.02); cb.set_label("Accuracy")
save(fig,"fig_peranimal")
print("all 7 figures written.")
