import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))).replace("\\", "/") + "/"
import json, numpy as np, matplotlib.pyplot as plt
import figstyle as fs; fs.apply()
q=json.load(open(ROOT + "results/E5_qaoa.json"))
P=np.arange(1,6)
ar=np.array([[i["levels"][p-1]["approx_ratio"] for p in P] for i in q]); po=np.array([[i["levels"][p-1]["p_opt"] for p in P] for i in q])
fig,axs=plt.subplots(1,2,figsize=(6.6,2.8))
for a in ar: axs[0].plot(P,a,color=fs.PAL[0],alpha=0.25,lw=1)
axs[0].plot(P,ar.mean(0),color=fs.PAL[0],marker="o",ms=5,label="Mean")
for a in po: axs[1].plot(P,a,color=fs.PAL[1],alpha=0.25,lw=1)
axs[1].plot(P,po.mean(0),color=fs.PAL[1],marker="o",ms=5,label="Mean")
axs[1].axhline(1/2**15,color=fs.INK2,lw=1,ls="--"); axs[1].text(1.05,1/2**15*1.15,"uniform 1/32,768",fontsize=7.5,color=fs.INK2)
axs[1].set_yscale("log")
axs[0].set_ylabel("Approximation ratio"); axs[1].set_ylabel("P(optimal bit string)")
for ax in axs: ax.set_xlabel("QAOA depth p"); ax.set_xticks(P)
axs[0].text(4.9,ar.mean(0)[-1]-0.012,"mean",fontsize=8,ha="right"); axs[1].text(4.9,po.mean(0)[-1]*1.25,"mean",fontsize=8,ha="right")
fig.tight_layout(); fig.savefig(ROOT + "figures/Fig6_qaoa.png"); print("ok")

# Table 9 summary (mean over instances for each depth)
import pandas as pd
rows = []
for p in P:
    Lv = [i["levels"][p - 1] for i in q]
    rows.append(dict(p=int(p), ar=np.mean([l["approx_ratio"] for l in Lv]), ar_min=np.min([l["approx_ratio"] for l in Lv]),
                     ar_max=np.max([l["approx_ratio"] for l in Lv]), popt=np.mean([l["p_opt"] for l in Lv]),
                     popt_max=np.max([l["p_opt"] for l in Lv]), hit=np.mean([l["opt_in_1024"] for l in Lv]),
                     zero=np.mean([l["shot_gap_pct"] <= 1e-6 for l in Lv]), shot_gap=np.mean([l["shot_gap_pct"] for l in Lv]),
                     mp=np.mean([l["most_prob_is_opt"] for l in Lv]), nfev=np.mean([l["n_eval"] for l in Lv])))
pd.DataFrame(rows).to_csv(ROOT + "results/T9_qaoa.csv", index=False); print("T9 written")
