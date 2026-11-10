import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))).replace("\\", "/") + "/"
"""Core model for the QUBO unit-commitment benchmark.

Test system: 10-unit benchmark of Kazarlis, Bakirtzis & Petridis (1996), IEEE TPWRS 11(1):83-92.
Net load: Germany 2019 hourly load / solar / wind (Open Power System Data, v2020-10-06), rescaled.
"""
import numpy as np, pandas as pd, time
from numba import njit

# Pmax, Pmin, a, b, c, MU, MD, hot-start cost
UNITS = np.array([
    [455, 150, 1000, 16.19, 0.00048, 8, 8, 4500],
    [455, 150,  970, 17.26, 0.00031, 8, 8, 5000],
    [130,  20,  700, 16.60, 0.00200, 5, 5,  550],
    [130,  20,  680, 16.50, 0.00211, 5, 5,  560],
    [162,  25,  450, 19.70, 0.00398, 6, 6,  900],
    [ 80,  20,  370, 22.26, 0.00712, 3, 3,  170],
    [ 85,  25,  480, 27.74, 0.00079, 3, 3,  260],
    [ 55,  10,  660, 25.92, 0.00413, 1, 1,   30],
    [ 55,  10,  665, 27.27, 0.00222, 1, 1,   30],
    [ 55,  10,  670, 27.79, 0.00173, 1, 1,   30]], dtype=float)
PMAX, PMIN, A, B, C, MU, MD, SU = (UNITS[:, k] for k in range(8))
NU, T = 10, 24
U_INIT = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
RES = 0.10           # spinning reserve fraction
PEAK = 1500.0        # MW, Kazarlis peak demand
FLOOR = 150.0        # MW, minimum thermal net load (renewables curtailed below this)
DATA = ROOT + "data/DE_2019_hourly.csv"


# ---------------------------------------------------------------- data
def load_series():
    df = pd.read_csv(DATA, index_col=0, parse_dates=True)
    k = PEAK / df.load_MW.max()
    L = df.load_MW.values * k
    R = (df.solar_MW.values + df.wind_MW.values) * k
    actual_share = R.sum() / L.sum()
    return df.index, L, R, actual_share


def net_load(L, R, rho, actual_share):
    scale = 0.0 if rho == 0 else rho / actual_share
    raw = L - scale * R
    D = np.maximum(raw, FLOOR)
    curtailed = (D - raw).clip(min=0)
    return D, curtailed, scale * R


def test_days():
    return [7 * k + (k % 7) for k in range(52)]


def calib_days():
    return [7 * k + ((k + 3) % 7) for k in (3, 10, 16, 22, 29, 35, 42, 48)]


# ---------------------------------------------------------------- QUBO
class Qubo:
    def __init__(self, D, K=3, alpha=1.0, lam1=10.0, gstar=100.0, link=2000.0):
        self.D, self.K = np.asarray(D, float), K
        nb = 1 + K
        self.nb = nb
        self.n = T * NU * nb
        n = self.n
        Q = np.zeros((n, n))
        self.step = (PMAX - PMIN) / (2 ** K - 1)
        lam2 = lam1 / (2 * gstar)
        off = 0.0

        def ix(t, i, j):  # j=0 -> u, j>=1 -> bit j-1
            return (t * NU + i) * nb + j

        def add_lin(i, v):
            Q[i, i] += v

        def add_q(i, j, v):
            if i == j:
                Q[i, i] += v
            elif i < j:
                Q[i, j] += v
            else:
                Q[j, i] += v

        def add_sq(w, idx, cf, const):
            nonlocal off
            for a_, (i, ci) in enumerate(zip(idx, cf)):
                add_lin(i, w * (ci * ci + 2 * const * ci))
                for (j, cj) in list(zip(idx, cf))[a_ + 1:]:
                    add_q(i, j, 2 * w * ci * cj)
            off += w * const * const

        for t in range(T):
            all_idx, all_cf, res_idx, res_cf = [], [], [], []
            for i in range(NU):
                idx = [ix(t, i, 0)] + [ix(t, i, 1 + k) for k in range(K)]
                cf = [PMIN[i]] + [self.step[i] * 2 ** k for k in range(K)]
                # generation cost a*u + b*p + c*p^2
                add_lin(idx[0], A[i])
                for v, c_ in zip(idx, cf):
                    add_lin(v, B[i] * c_)
                add_sq(C[i], idx, cf, 0.0)
                # link: bit may be 1 only if unit on
                for v in idx[1:]:
                    add_lin(v, link)
                    add_q(v, idx[0], -link)
                # startup  S*u_t*(1-u_{t-1})
                if t == 0:
                    if U_INIT[i] == 0:
                        add_lin(idx[0], SU[i])
                else:
                    add_lin(idx[0], SU[i])
                    add_q(idx[0], ix(t - 1, i, 0), -SU[i])
                all_idx += idx; all_cf += cf
                res_idx.append(idx[0]); res_cf.append(PMAX[i])
            # demand balance
            add_sq(alpha, all_idx, all_cf, -self.D[t])
            # spinning reserve by unbalanced penalization: g = sum Pmax u - (1+r)D >= 0 ; pen = -lam1 g + lam2 g^2
            Rt = (1 + RES) * self.D[t]
            for v, c_ in zip(res_idx, res_cf):
                add_lin(v, -lam1 * c_)
            off += lam1 * Rt
            add_sq(lam2, res_idx, res_cf, -Rt)
        self.Q, self.offset = Q, off
        self._ix = ix

    def energy(self, X):
        X = np.atleast_2d(X).astype(float)
        return np.einsum("ri,ij,rj->r", X, self.Q, X) + self.offset

    def decode(self, x):
        x = np.asarray(x).reshape(T, NU, self.nb)
        u = x[:, :, 0].astype(int)
        bits = x[:, :, 1:]
        w = 2 ** np.arange(self.K)
        p = PMIN * u + self.step * (bits * w).sum(-1)
        link_viol = int((bits.sum(-1) * (1 - u)[:, :]).astype(bool).sum())
        return u, p, link_viol

    def raw_metrics(self, x):
        u, p, lv = self.decode(x)
        mism = p.sum(1) - self.D
        res_ok = (u * PMAX).sum(1) >= (1 + RES) * self.D - 1e-6
        bal_ok = np.abs(mism) <= 0.02 * self.D
        feas = bool(res_ok.all() and bal_ok.all() and lv == 0)
        return dict(raw_feasible=feas, hours_res_ok=int(res_ok.sum()), hours_bal_ok=int(bal_ok.sum()),
                    mean_abs_mismatch=float(np.abs(mism).mean()), link_viol=lv), u


# ---------------------------------------------------------------- evaluation
def avg_full_cost():
    return (A + B * PMAX + C * PMAX ** 2) / PMAX


def repair(u, D):
    """Shared greedy repair: close reserve shortfall with the cheapest adequate unit,
    then drop units while minimum-output exceeds demand. Applied identically to every solver."""
    u = u.copy()
    order_off = np.argsort(avg_full_cost())[::-1]
    actions = 0
    for t in range(T):
        while (u[t] * PMAX).sum() < (1 + RES) * D[t] - 1e-9:
            short = (1 + RES) * D[t] - (u[t] * PMAX).sum()
            cand = [i for i in range(NU) if u[t, i] == 0]
            if not cand:
                break
            prev_on = U_INIT if t == 0 else u[t - 1]
            def score(i):
                return A[i] + B[i] * PMIN[i] + C[i] * PMIN[i] ** 2 + SU[i] * (1 - prev_on[i])
            adequate = [i for i in cand if PMAX[i] >= short]
            pick = min(adequate, key=score) if adequate else max(cand, key=lambda i: PMAX[i])
            u[t, pick] = 1; actions += 1
        for i in order_off:
            if (u[t] * PMIN).sum() <= D[t]:
                break
            if u[t, i] == 1 and ((u[t] * PMAX).sum() - PMAX[i]) >= (1 + RES) * D[t]:
                u[t, i] = 0; actions += 1
    return u, actions


def economic_dispatch(u, D):
    """Exact quadratic ED per hour by bisection on lambda. u (T,NU), D (T,)"""
    u = u.astype(float)
    lo = np.full(T, B.min() - 1.0)
    hi = np.full(T, (B + 2 * C * PMAX).max() + 1.0)
    for _ in range(60):
        lam = 0.5 * (lo + hi)
        p = np.clip((lam[:, None] - B) / (2 * C), PMIN, PMAX) * u
        s = p.sum(1)
        lo = np.where(s < D, lam, lo)
        hi = np.where(s >= D, lam, hi)
    p = np.clip((hi[:, None] - B) / (2 * C), PMIN, PMAX) * u
    return p


def evaluate(u, D):
    """Repair, dispatch and cost a commitment. Returns dict."""
    ur, actions = repair(u, D)
    p = economic_dispatch(ur, D)
    gen = (ur * (A + B * p + C * p ** 2)).sum()
    prev = np.vstack([U_INIT, ur[:-1]])
    starts = (ur * (1 - prev))
    stc = (starts * SU).sum()
    bal_err = np.abs(p.sum(1) - D).max()
    feas = bool(((ur * PMAX).sum(1) >= (1 + RES) * D - 1e-6).all() and bal_err < 1e-3)
    return dict(cost=float(gen + stc), gen_cost=float(gen), startup_cost=float(stc), n_starts=int(starts.sum()),
                repair_actions=int(actions), feasible_after_repair=feas, mud_viol=mud_violations(ur)), ur, p


def mud_violations(u):
    v = 0
    for i in range(NU):
        s = u[:, i]
        runs, cur, ln = [], s[0], 1
        for x in s[1:]:
            if x == cur:
                ln += 1
            else:
                runs.append((cur, ln)); cur, ln = x, 1
        runs.append((cur, ln))
        for r_i, (val, ln) in enumerate(runs):
            if r_i == 0 or r_i == len(runs) - 1:
                continue  # boundary runs are truncated by the horizon
            if val == 1 and ln < MU[i]:
                v += 1
            if val == 0 and ln < MD[i]:
                v += 1
    return v


# ---------------------------------------------------------------- MILP reference
def solve_milp(D, nseg=8, time_limit=60):
    from scipy.optimize import milp, LinearConstraint, Bounds
    from scipy.sparse import lil_matrix
    nv_per = 2 + nseg
    n = T * NU * nv_per
    def iu(t, i): return (t * NU + i) * nv_per
    def isu(t, i): return iu(t, i) + 1
    def idl(t, i, s): return iu(t, i) + 2 + s
    c = np.zeros(n); lb = np.zeros(n); ub = np.ones(n); integ = np.zeros(n)
    seglen = (PMAX - PMIN) / nseg
    for t in range(T):
        for i in range(NU):
            F = lambda x: A[i] + B[i] * x + C[i] * x * x
            c[iu(t, i)] = F(PMIN[i]); integ[iu(t, i)] = 1
            c[isu(t, i)] = SU[i]
            for s in range(nseg):
                x0 = PMIN[i] + s * seglen[i]
                c[idl(t, i, s)] = (F(x0 + seglen[i]) - F(x0)) / seglen[i]
                ub[idl(t, i, s)] = seglen[i]
    rows = []; blo = []; bhi = []
    M = lil_matrix((T * 2 + T * NU * (1 + nseg), n))
    r = 0
    for t in range(T):
        for i in range(NU):
            M[r, iu(t, i)] = PMIN[i]
            for s in range(nseg):
                M[r, idl(t, i, s)] = 1
        blo.append(D[t]); bhi.append(D[t]); r += 1
        for i in range(NU):
            M[r, iu(t, i)] = PMAX[i]
        blo.append((1 + RES) * D[t]); bhi.append(np.inf); r += 1
    for t in range(T):
        for i in range(NU):
            M[r, isu(t, i)] = 1; M[r, iu(t, i)] = -1
            if t > 0:
                M[r, iu(t - 1, i)] = 1; blo.append(0)
            else:
                blo.append(-U_INIT[i])
            bhi.append(np.inf); r += 1
            for s in range(nseg):
                M[r, idl(t, i, s)] = 1; M[r, iu(t, i)] = -seglen[i]
                blo.append(-np.inf); bhi.append(0); r += 1
    t0 = time.perf_counter()
    res = milp(c, constraints=LinearConstraint(M.tocsr(), blo, bhi), integrality=integ, bounds=Bounds(lb, ub),
               options=dict(time_limit=time_limit, mip_rel_gap=1e-5))
    dt = time.perf_counter() - t0
    x = res.x
    u = np.array([[round(x[iu(t, i)]) for i in range(NU)] for t in range(T)])
    return u, dt, res.status, float(res.fun)


# ---------------------------------------------------------------- simulated quantum annealing (PIMC)
@njit(cache=True)
def _sqa(h, indptr, nbr, Jv, P, sweeps, Tq, G0, G1, seed):
    np.random.seed(seed)
    n = h.shape[0]
    s = np.empty((P, n))
    for k in range(P):
        for i in range(n):
            s[k, i] = 1.0 if np.random.random() < 0.5 else -1.0
    for sw in range(sweeps):
        G = G0 + (G1 - G0) * sw / max(sweeps - 1, 1)
        Jp = -0.5 * P * Tq * np.log(np.tanh(G / (P * Tq)))
        for k in range(P):
            km = (k - 1) % P
            kp = (k + 1) % P
            for i in range(n):
                f = h[i]
                for q in range(indptr[i], indptr[i + 1]):
                    f += Jv[q] * s[k, nbr[q]]
                si = s[k, i]
                dE = -2.0 * si * f / P + 2.0 * Jp * si * (s[km, i] + s[kp, i])
                if dE <= 0.0 or np.random.random() < np.exp(-dE / Tq):
                    s[k, i] = -si
    return s


def qubo_to_ising(Q):
    """x=(1+s)/2. E = s^T J s (i<j) + h s + const"""
    Qs = np.triu(Q, 1)
    d = np.diag(Q)
    J = Qs / 4.0
    h = d / 2.0 + (Qs.sum(1) + Qs.sum(0)) / 4.0
    const = d.sum() / 2.0 + Qs.sum() / 4.0
    return h, J, const


def sqa_solve(Q, P=16, sweeps=250, reads=8, Tq=0.05, G0=3.0, G1=1e-3, seed=0):
    h, J, _ = qubo_to_ising(Q)
    sc = max(np.abs(h).max(), np.abs(J).max())
    h = h / sc; J = J / sc
    Jf = J + J.T
    indptr = [0]; nbr = []; Jv = []
    for i in range(len(h)):
        nz = np.nonzero(Jf[i])[0]
        nbr.extend(nz); Jv.extend(Jf[i, nz]); indptr.append(len(nbr))
    indptr = np.array(indptr, np.int64); nbr = np.array(nbr, np.int64); Jv = np.array(Jv)
    best = []
    for r in range(reads):
        s = _sqa(h, indptr, nbr, Jv, P, sweeps, Tq, G0, G1, seed * 1000 + r)
        X = (s + 1) / 2
        best.append(X)
    return np.vstack(best)  # all slices of all reads


# ---------------------------------------------------------------- GA on commitments
def ga_solve(D, pop=40, gens=150, seed=0):
    rng = np.random.default_rng(seed)
    def fit(u):
        r, ur, _ = evaluate(u, D)
        return r["cost"], ur
    # seed population: priority-list style random
    P = []
    for _ in range(pop):
        u = (rng.random((T, NU)) < 0.5).astype(int)
        P.append(u)
    F = []
    for k in range(pop):
        f, P[k] = fit(P[k]); F.append(f)
    F = np.array(F)
    for g in range(gens):
        newP = [P[int(F.argmin())].copy()]
        while len(newP) < pop:
            a, b = rng.integers(pop, size=2); p1 = P[a] if F[a] < F[b] else P[b]
            a, b = rng.integers(pop, size=2); p2 = P[a] if F[a] < F[b] else P[b]
            # window crossover over time blocks (keeps commitment runs)
            cut = rng.integers(1, T)
            child = np.vstack([p1[:cut], p2[cut:]]) if rng.random() < 0.5 else np.where(rng.random(NU) < 0.5, p1, p2)
            m = rng.random((T, NU)) < 1.0 / (T * NU) * 2
            child = np.where(m, 1 - child, child)
            # swap-window mutation: smooth one unit over a 3-h window
            if rng.random() < 0.3:
                i = rng.integers(NU); t0 = rng.integers(0, T - 3); child[t0:t0 + 3, i] = child[t0, i]
            newP.append(child)
        P = newP
        F = []
        for k in range(pop):
            f, P[k] = fit(P[k]); F.append(f)
        F = np.array(F)
    k = int(F.argmin())
    return P[k]


# ---------------------------------------------------------------- hybrid augmented-Lagrangian commitment QUBO
def ed_with_lambda(u, D):
    u = u.astype(float)
    lo = np.full(T, B.min() - 1.0)
    hi = np.full(T, (B + 2 * C * PMAX).max() + 1.0)
    for _ in range(60):
        lam = 0.5 * (lo + hi)
        s = (np.clip((lam[:, None] - B) / (2 * C), PMIN, PMAX) * u).sum(1)
        lo = np.where(s < D, lam, lo); hi = np.where(s >= D, lam, hi)
    return hi


def priority_list(D):
    order = np.argsort(avg_full_cost())
    u = np.zeros((T, NU), int)
    for t in range(T):
        cap = 0.0
        for i in order:
            if cap >= (1 + RES) * D[t]:
                break
            u[t, i] = 1; cap += PMAX[i]
    return repair(u, D)[0]


def al_qubo(D, lam_t, alpha=0.01, w_res=0.05, ks=5, ds=50.0):
    """QUBO over u (T*NU vars, index t*NU+i) plus ks reserve-slack bits per hour.
    Objective: reduced cost kappa*u + startup + alpha*(sum p* u - D)^2 + w_res*(sum Pmax u - 1.1D - slack)^2."""
    nu = T * NU
    n = nu + T * ks
    Q = np.zeros((n, n)); off = 0.0
    pstar = np.clip((lam_t[:, None] - B) / (2 * C), PMIN, PMAX)
    kappa = A + B * pstar + C * pstar ** 2 - lam_t[:, None] * pstar
    def add(i, j, v):
        if i == j: Q[i, i] += v
        elif i < j: Q[i, j] += v
        else: Q[j, i] += v
    def add_sq(w, idx, cf, const):
        nonlocal off
        for a_ in range(len(idx)):
            add(idx[a_], idx[a_], w * (cf[a_] ** 2 + 2 * const * cf[a_]))
            for b_ in range(a_ + 1, len(idx)):
                add(idx[a_], idx[b_], 2 * w * cf[a_] * cf[b_])
        off += w * const * const
    for t in range(T):
        ids = [t * NU + i for i in range(NU)]
        for i in range(NU):
            add(ids[i], ids[i], kappa[t, i])
            if t == 0:
                if U_INIT[i] == 0: add(ids[i], ids[i], SU[i])
            else:
                add(ids[i], ids[i], SU[i]); add(ids[i], (t - 1) * NU + i, -SU[i])
        add_sq(alpha, ids, list(pstar[t]), -D[t])
        sl = [nu + t * ks + k for k in range(ks)]
        add_sq(w_res, ids + sl, list(PMAX) + [-ds * 2 ** k for k in range(ks)], -(1 + RES) * D[t])
    return Q, off


def hybrid_solve(D, sampler_fn, iters=4, alpha=0.01, w_res=0.05, damp=0.5):
    """sampler_fn(Q, offset) -> array of binary samples. Returns best u, history."""
    u0 = priority_list(D)
    best_r, _, _ = evaluate(u0, D); best_u = u0
    lam = ed_with_lambda(best_u, D)
    hist = [best_r["cost"]]
    for k in range(iters):
        Q, off = al_qubo(D, lam, alpha, w_res)
        X = sampler_fn(Q, off)
        E = np.einsum("ri,ij,rj->r", X.astype(float), Q, X.astype(float))
        order = np.argsort(E)[:4]
        for j in order:
            u = X[j][:T * NU].reshape(T, NU).astype(int)
            r, ur, _ = evaluate(u, D)
            if r["cost"] < best_r["cost"] - 1e-6:
                best_r, best_u = r, ur
        lam_new = ed_with_lambda(best_u, D)
        lam = damp * lam + (1 - damp) * lam_new
        hist.append(best_r["cost"])
    return best_u, hist
