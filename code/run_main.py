import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))).replace("\\", "/") + "/"
"""E1 main benchmark. Writes results/E1_main.jsonl"""
import numpy as np, json, time, sys, os
from multiprocessing import Pool
OUT = ROOT + "results/E1_main.jsonl"
CFG = dict(D_QUBO=dict(K=3, alpha=0.3, lam1=1.0, gstar=400.0, link=1500.0),
           AL=dict(iters=6, alpha=0.01, w_res=0.01),
           SA=dict(num_reads=32, num_sweeps=1000), SQA=dict(P=16, sweeps=250, reads=8, Tq=0.01, G0=2.0),
           TABU=dict(num_reads=8, timeout=200), GA=dict(pop=40, gens=150), SEEDS=[1, 2, 3])

def task(args):
    rho_name, rho, d = args
    import dimod
    from dwave.samplers import SimulatedAnnealingSampler, TabuSampler
    from uc_core import (load_series, net_load, solve_milp, evaluate, priority_list, Qubo, sqa_solve,
                         ga_solve, hybrid_solve, T, NU)
    idx, L, R, sh = load_series()
    r_ = sh if rho == "actual" else rho
    D, cur, Rs = net_load(L, R, r_, sh)
    sl = slice(d * 24, (d + 1) * 24)
    Dd = D[sl]; Ld = L[sl]
    ramps = np.diff(Dd)
    rec = dict(rho=rho_name, day=int(d), date=str(idx[d * 24].date()), D_mean=float(Dd.mean()), D_max=float(Dd.max()),
               D_min=float(Dd.min()), D_std=float(Dd.std()), ramp_mean_abs=float(np.abs(ramps).mean()),
               ramp_max=float(np.abs(ramps).max()), ren_share_day=float(min(Rs[sl].sum(), Ld.sum()) / Ld.sum()),
               curtail_MWh=float(cur[sl].sum()), floor_hours=int((Dd <= 150.0 + 1e-9).sum()))
    res = {}
    t0 = time.perf_counter(); um, dt, st, fun = solve_milp(Dd); r, _, _ = evaluate(um, Dd)
    res["MILP"] = dict(**r, time=dt, status=int(st))
    t0 = time.perf_counter(); r, _, _ = evaluate(priority_list(Dd), Dd); res["PL"] = dict(**r, time=time.perf_counter() - t0)
    t0 = time.perf_counter(); r, _, _ = evaluate(ga_solve(Dd, seed=1, **CFG["GA"]), Dd); res["GA"] = dict(**r, time=time.perf_counter() - t0)
    # direct QUBO
    q = Qubo(Dd, **CFG["D_QUBO"]); bqm = dimod.BQM(q.Q, "BINARY"); bqm.offset = q.offset
    for name, fn in (("D-SA", lambda s: SimulatedAnnealingSampler().sample(bqm, seed=s, **CFG["SA"]).record.sample),
                     ("D-SQA", lambda s: sqa_solve(q.Q, seed=s, **CFG["SQA"]))):
        runs = []
        for s in CFG["SEEDS"]:
            t0 = time.perf_counter(); X = fn(s); dt = time.perf_counter() - t0
            E = q.energy(X); b = X[E.argmin()]; m, u = q.raw_metrics(b); r, _, _ = evaluate(u, Dd)
            runs.append(dict(**r, **m, qubo_E=float(E.min()), time=dt, seed=s))
        res[name] = runs
    res["D_n_vars"] = q.n; res["D_n_quad"] = int(np.count_nonzero(np.triu(q.Q, 1)))
    # hybrid AL-QUBO
    samplers = {"AL-SA": lambda s: (lambda Q, o: SimulatedAnnealingSampler().sample(dimod.BQM(Q, "BINARY"), seed=s, **CFG["SA"]).record.sample),
                "AL-SQA": lambda s: (lambda Q, o: sqa_solve(Q, seed=s, **CFG["SQA"])),
                "AL-Tabu": lambda s: (lambda Q, o: TabuSampler().sample(dimod.BQM(Q, "BINARY"), seed=s, **CFG["TABU"]).record.sample)}
    for name, mk in samplers.items():
        runs = []
        for s in (CFG["SEEDS"] if name != "AL-Tabu" else CFG["SEEDS"][:1]):
            t0 = time.perf_counter(); u, h = hybrid_solve(Dd, mk(s), **CFG["AL"]); dt = time.perf_counter() - t0
            r, _, _ = evaluate(u, Dd); runs.append(dict(**r, time=dt, seed=s, hist=h))
        res[name] = runs
    rec["res"] = res
    return rec

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from uc_core import test_days
    jobs = [(n, r, d) for n, r in (("0%", 0.0), ("actual", "actual"), ("50%", 0.5)) for d in test_days()]
    done = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            j = json.loads(line); done.add((j["rho"], j["day"]))
    jobs = [j for j in jobs if (j[0], j[2]) not in done]
    print("jobs", len(jobs), flush=True)
    json.dump(CFG, open(OUT.replace(".jsonl", "_config.json"), "w"), indent=1)
    with Pool(7) as p, open(OUT, "a") as f:
        for k, rec in enumerate(p.imap_unordered(task, jobs)):
            f.write(json.dumps(rec) + "\n"); f.flush()
            if k % 10 == 0: print(k, rec["rho"], rec["date"], flush=True)
