import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))).replace("\\", "/") + "/"
import pandas as pd, matplotlib.pyplot as plt
import figstyle as fs; fs.apply()
g=pd.read_csv(ROOT + "results/T8_sens.csv")
fig,axs=plt.subplots(1,2,figsize=(6.6,3.5))
for k,K in enumerate([1,2,3,4,5,6,7]):
    x=g[g.K==K]
    axs[0].plot(x.alpha,x.gap,marker="o",ms=5,color=fs.PAL[k],label=f"K = {K} bit{'s' if K>1 else ''} ({int(x.vars.iloc[0]):,} variables)")
    axs[1].plot(x.alpha,x.bal,marker="o",ms=5,color=fs.PAL[k],label=f"K = {K}")
for ax in axs: ax.set_xscale("log"); ax.set_xticks([0.1,0.3,1,3]); ax.set_xticklabels(["0.1","0.3","1","3"]); ax.set_xlabel("Balance penalty \u03b1 ($/MW\u00b2)")
axs[0].set_ylabel("Mean gap after repair (%)"); axs[1].set_ylabel("Hours within \u00b15% balance (of 24)")
axs[1].set_ylim(0,25); axs[0].set_ylim(0,28)
h,l=axs[0].get_legend_handles_labels(); fig.legend(h,l,loc="lower center",ncol=4,fontsize=7.5,frameon=False,columnspacing=1.2,handlelength=1.6)
fig.tight_layout(rect=(0,0.15,1,1)); fig.savefig(ROOT + "figures/Fig5_sensitivity.png"); print("ok")
