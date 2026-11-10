import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))).replace("\\", "/") + "/"
"""E2/E3: direct-QUBO sensitivity to balance penalty alpha and bit depth K (SA, 52 test days, 2019-actual mix)."""
import numpy as np, json, itertools, os, sys
from multiprocessing import Pool
OUT=ROOT + "results/E23_sens.jsonl"
def task(a):
    K, alpha, d = a
    import dimod
    from dwave.samplers import SimulatedAnnealingSampler
    from uc_core import load_series, net_load, Qubo, evaluate, solve_milp
    idx,L,R,sh=load_series(); D,_,_=net_load(L,R,sh,sh); Dd=D[d*24:(d+1)*24]
    ref=evaluate(solve_milp(Dd)[0],Dd)[0]['cost']
    q=Qubo(Dd,K=K,alpha=alpha,lam1=1.0,gstar=400.0,link=1500.0); bqm=dimod.BQM(q.Q,"BINARY")
    X=SimulatedAnnealingSampler().sample(bqm,num_reads=32,num_sweeps=1000,seed=1).record.sample
    E=q.energy(X); m,u=q.raw_metrics(X[E.argmin()]); r=evaluate(u,Dd)[0]
    return dict(K=K,alpha=alpha,day=d,n=q.n,gap=100*(r['cost']/ref-1),**m,n_starts=r['n_starts'],repair=r['repair_actions'])
if __name__=="__main__":
    sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
    from uc_core import test_days
    jobs=list(itertools.product([1,2,3,4,5,6,7],[0.1,0.3,1.0,3.0],test_days()))
    with Pool(7) as p, open(OUT,"w") as f:
        for r in p.imap_unordered(task,jobs,chunksize=4): f.write(json.dumps(r)+"\n")
