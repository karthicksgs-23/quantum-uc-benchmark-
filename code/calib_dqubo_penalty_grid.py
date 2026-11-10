import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))).replace("\\", "/") + "/"
import numpy as np, time, dimod, itertools, json
from dwave.samplers import SimulatedAnnealingSampler
from uc_core import *
idx,L,R,sh=load_series(); D,_,_=net_load(L,R,sh,sh)
days=calib_days()[:4]
ref={}
for d in days:
    Dd=D[d*24:(d+1)*24]; um,_,_,_=solve_milp(Dd); ref[d]=evaluate(um,Dd)[0]['cost']
res=[]
for lam1,gstar,link,alpha in itertools.product([0.2,1,5],[100,400],[1500,4000],[0.3,1]):
    gaps=[];bal=[];lv=[]
    for d in days:
        Dd=D[d*24:(d+1)*24]; q=Qubo(Dd,K=3,alpha=alpha,lam1=lam1,gstar=gstar,link=link)
        bqm=dimod.BQM(q.Q,"BINARY"); bqm.offset=q.offset
        ss=SimulatedAnnealingSampler().sample(bqm,num_reads=16,num_sweeps=2000,seed=1)
        X=ss.record.sample;E=q.energy(X);b=X[E.argmin()]; m,u=q.raw_metrics(b); rr,_,_=evaluate(u,Dd)
        gaps.append(100*(rr['cost']/ref[d]-1)); bal.append(m['hours_bal_ok']); lv.append(m['link_viol'])
    print(lam1,gstar,link,alpha,"gap",np.round(gaps,1),"mean",round(np.mean(gaps),2),"bal",bal,"link",lv,flush=True)
