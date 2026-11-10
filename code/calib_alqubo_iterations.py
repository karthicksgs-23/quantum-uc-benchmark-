import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))).replace("\\", "/") + "/"
import numpy as np, dimod, time
from dwave.samplers import SimulatedAnnealingSampler
from uc_core import *
idx,L,R,sh=load_series(); D,_,_=net_load(L,R,sh,sh)
def sa(Q,off):
    return SimulatedAnnealingSampler().sample(dimod.BQM(Q,"BINARY"),num_reads=32,num_sweeps=1000,seed=1).record.sample
refs={}
for d in calib_days():
    Dd=D[d*24:(d+1)*24]; um,_,_,_=solve_milp(Dd); refs[d]=evaluate(um,Dd)[0]['cost']
for alpha,w,it in [(0.01,0.003,4),(0.01,0.001,4),(0.01,0.01,8),(0.01,0.003,8),(0.003,0.003,8)]:
    out=[]
    for d in calib_days():
        Dd=D[d*24:(d+1)*24]; t0=time.time(); u,h=hybrid_solve(Dd,sa,iters=it,alpha=alpha,w_res=w); r=evaluate(u,Dd)[0]
        out.append((100*(r['cost']/refs[d]-1),time.time()-t0))
    o=np.array(out); print(alpha,w,it,o[:,0].round(2),"mean",o[:,0].mean().round(3),"t",o[:,1].mean().round(1),flush=True)
