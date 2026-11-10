import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))).replace("\\", "/") + "/"
import numpy as np, time
from uc_core import *
idx,L,R,sh=load_series(); D,_,_=net_load(L,R,sh,sh)
refs={}
for d in calib_days():
    Dd=D[d*24:(d+1)*24]; um,_,_,_=solve_milp(Dd); refs[d]=evaluate(um,Dd)[0]['cost']
for Tq,G0 in [(0.01,2),(0.03,2),(0.03,5),(0.1,2),(0.1,5)]:
    out=[]
    for d in calib_days():
        Dd=D[d*24:(d+1)*24]; t0=time.time()
        u,h=hybrid_solve(Dd,lambda Q,o: sqa_solve(Q,P=16,sweeps=250,reads=8,Tq=Tq,G0=G0,seed=1),iters=6,alpha=0.01,w_res=0.01)
        r=evaluate(u,Dd)[0]; out.append((100*(r['cost']/refs[d]-1),time.time()-t0))
    o=np.array(out); print(Tq,G0,o[:,0].round(2),"mean",o[:,0].mean().round(3),"t",o[:,1].mean().round(1),flush=True)
