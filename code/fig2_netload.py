import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))).replace("\\", "/") + "/"
import numpy as np, matplotlib.pyplot as plt
from uc_core import load_series, net_load
import figstyle as fs; fs.apply()
idx,L,R,sh=load_series()
fig,ax=plt.subplots(figsize=(6.3,3.4))
for k,(name,rho) in enumerate([("0% renewables",0.0),(f"2019 actual ({sh*100:.1f}%)",sh),("50% renewables",0.5)]):
    D,_,_=net_load(L,R,rho,sh); M=D.reshape(365,24)
    med=np.median(M,0); lo,hi=np.percentile(M,[10,90],0); h=np.arange(24)
    ax.fill_between(h,lo,hi,color=fs.PAL[k],alpha=0.15,lw=0)
    ax.plot(h,med,color=fs.PAL[k],label=name)
    ax.text(23.3,med[-1],name.split(" (")[0] if k!=1 else "2019 actual",color=fs.INK,va="center",fontsize=8.5)
ax.set_xlim(0,27.5); ax.set_xticks(range(0,24,3)); ax.set_xlabel("Hour of day"); ax.set_ylabel("Net load (MW)")
ax.legend(loc="upper left",fontsize=8.5,ncol=3); ax.set_ylim(0,1700)
fig.tight_layout(); fig.savefig(ROOT + "figures/Fig2_netload.png"); print("ok")
# stats for text
for rho in (0.0,sh,0.5):
    D,cur,Rs=net_load(L,R,rho,sh); r=np.abs(np.diff(D.reshape(365,24),axis=1))
    print(round(rho,3),"mean",D.mean().round(0),"min",D.min().round(0),"floor hrs",int((D<=150).sum()),"curt % of RES",round(100*cur.sum()/max(Rs.sum(),1),2),"mean|ramp|",r.mean().round(1),"p95 ramp",np.percentile(r,95).round(0))
