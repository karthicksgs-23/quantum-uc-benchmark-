import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))).replace("\\", "/") + "/"
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from scipy.stats import spearmanr
import figstyle as fs; fs.apply()
df=pd.read_csv(ROOT + "results/E1_long.csv")
dm=df.groupby(["rho","day","method"]).agg(gap=("gap","mean"),D_mean=("D_mean","first")).reset_index()
LBL={"0%":"0%","actual":"2019 actual","50%":"50%"}
fig,axs=plt.subplots(1,2,figsize=(6.6,3.0))
for ax,m in zip(axs,["D-SA","AL-SA"]):
    for k,rho in enumerate(["0%","actual","50%"]):
        x=dm[(dm.rho==rho)&(dm.method==m)]
        ax.scatter(x.D_mean,x.gap,s=10,color=fs.PAL[k],alpha=0.75,lw=0,label=LBL[rho])
    x=dm[dm.method==m]; r,p=spearmanr(x.D_mean,x.gap)
    ax.set_title(f"{m}: Spearman \u03c1 = {r:.2f}",fontsize=9.5)
    ax.set_xlabel("Daily mean net load (MW)"); ax.set_ylabel("Gap (%)")
axs[0].legend(fontsize=7.5,loc="upper right")
fig.tight_layout(); fig.savefig(ROOT + "figures/Fig4_load_vs_gap.png"); print("ok")
