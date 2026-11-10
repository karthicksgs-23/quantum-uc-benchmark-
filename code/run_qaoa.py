import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))).replace("\\", "/") + "/"
"""E5: QAOA (noiseless statevector, NumPy) on single-hour commitment QUBOs (10 unit qubits + 5 reserve-slack qubits).
Exact optimum by enumeration of all 2^15 states. Parameters: COBYLA, multistart at p=1, INTERP initialisation for p>1."""
import numpy as np, json, time
from scipy.optimize import minimize
from uc_core import *

KS, DS, ALPHA, WRES = 5, 50.0, 0.01, 0.01


def hour_qubo(Dt, lam):
    pstar = np.clip((lam - B) / (2 * C), PMIN, PMAX)
    kappa = A + B * pstar + C * pstar ** 2 - lam * pstar
    n = NU + KS
    Q = np.zeros((n, n)); off = 0.0
    def add_sq(w, cf, const, idx):
        nonlocal off
        for a_ in range(len(idx)):
            Q[idx[a_], idx[a_]] += w * (cf[a_] ** 2 + 2 * const * cf[a_])
            for b_ in range(a_ + 1, len(idx)):
                Q[idx[a_], idx[b_]] += 2 * w * cf[a_] * cf[b_]
        off += w * const * const
    for i in range(NU):
        Q[i, i] += kappa[i]
    add_sq(ALPHA, list(pstar), -Dt, list(range(NU)))
    add_sq(WRES, list(PMAX) + [-DS * 2 ** k for k in range(KS)], -(1 + RES) * Dt, list(range(n)))
    return Q, off


def single_hour_cost(u, Dt):
    """exact single-hour cost (no startup) after shared repair; returns cost"""
    U = np.tile(u, (T, 1)); Dd = np.full(T, Dt)
    Ur, _ = repair(U, Dd)
    p = economic_dispatch(Ur, Dd)
    return float((Ur[0] * (A + B * p[0] + C * p[0] ** 2)).sum()), Ur[0]


def run_instance(Dt, lam, pmax=5, starts=8, seed=0):
    Q, off = hour_qubo(Dt, lam)
    n = Q.shape[0]; N = 2 ** n
    bits = ((np.arange(N)[:, None] >> np.arange(n)) & 1).astype(float)
    E = np.einsum("ri,ij,rj->r", bits, Q, bits) + off
    scale = np.abs(Q).sum()
    En = E / scale
    kopt = int(E.argmin())
    # exact commitment optimum (enumerate 2^10 commitments)
    best_c = min(single_hour_cost(np.array([(k >> i) & 1 for i in range(NU)]), Dt)[0] for k in range(1, 2 ** NU))
    qubo_opt_cost = single_hour_cost(bits[kopt, :NU].astype(int), Dt)[0]

    def state(params, p):
        g, b = params[:p], params[p:]
        psi = np.full(N, 1 / np.sqrt(N), complex)
        for l in range(p):
            psi = psi * np.exp(-1j * g[l] * En)
            c, s = np.cos(b[l]), -1j * np.sin(b[l])
            for j in range(n):
                v = psi.reshape(-1, 2, 2 ** j)
                a0 = v[:, 0, :].copy(); a1 = v[:, 1, :].copy()
                v[:, 0, :] = c * a0 + s * a1; v[:, 1, :] = c * a1 + s * a0
                psi = v.reshape(N)
        return psi

    def f(params, p):
        psi = state(params, p); return float((np.abs(psi) ** 2 * En).sum())

    rng = np.random.default_rng(seed)
    out = []; prev = None
    Emin, Emax = E.min(), E.max()
    for p in range(1, pmax + 1):
        t0 = time.perf_counter()
        if p == 1:
            best = None
            for s in range(starts):
                x0 = np.concatenate([rng.uniform(0, 2 * np.pi, 1) * 0.2, rng.uniform(0, np.pi / 2, 1)])
                r = minimize(f, x0, args=(1,), method="COBYLA", options=dict(maxiter=300))
                if best is None or r.fun < best.fun: best = r
        else:  # INTERP initialisation (Zhou et al.)
            g0, b0 = prev[:p - 1], prev[p - 1:]
            def interp(v):
                vv = np.concatenate([[0], v, [0]]); i = np.arange(1, p + 1)
                return (i - 1) / (p - 1) * vv[i - 1] + (p - i) / (p - 1) * vv[i]
            x0 = np.concatenate([interp(g0), interp(b0)])
            best = minimize(f, x0, args=(p,), method="COBYLA", options=dict(maxiter=400 * p))
        prev = best.x
        psi = state(best.x, p); pr = np.abs(psi) ** 2
        expE = float((pr * E).sum())
        ar = (Emax - expE) / (Emax - Emin)
        p_opt = float(pr[kopt])
        # 1024 shots
        shots = rng.choice(N, size=1024, p=pr / pr.sum())
        kb = shots[np.argmin(E[shots])]
        cost_best_shot, _ = single_hour_cost(bits[kb, :NU].astype(int), Dt)
        kmp = int(pr.argmax())
        out.append(dict(p=p, approx_ratio=ar, p_opt=p_opt, opt_in_1024=bool((shots == kopt).any()),
                        shot_gap_pct=100 * (cost_best_shot / best_c - 1), most_prob_is_opt=kmp == kopt,
                        n_eval=int(best.nfev), time=time.perf_counter() - t0))
    return dict(D=float(Dt), lam=float(lam), n_qubits=n, qubo_opt_gap_pct=100 * (qubo_opt_cost / best_c - 1),
                levels=out)


if __name__ == "__main__":
    idx, L, R, sh = load_series(); D, _, _ = net_load(L, R, sh, sh)
    picks = [test_days()[k] for k in (2, 15, 28, 41)]
    res = []
    for d in picks:
        Dd = D[d * 24:(d + 1) * 24]
        lam = ed_with_lambda(priority_list(Dd), Dd)
        for h in (3, 12, 19):
            r = run_instance(Dd[h], lam[h], seed=d * 100 + h)
            r.update(date=str(idx[d * 24].date()), hour=h)
            res.append(r)
            print(r["date"], h, round(r["D"]), [(l["p"], round(l["approx_ratio"], 3), round(l["p_opt"], 4), round(l["shot_gap_pct"], 2)) for l in r["levels"]], flush=True)
    json.dump(res, open(ROOT + "results/E5_qaoa.json", "w"), indent=1)
