import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))).replace("\\", "/") + "/"
import json, numpy as np, pandas as pd
from scipy.stats import wilcoxon, spearmanr
import matplotlib.pyplot as plt
import figstyle as fs
fs.apply()
RES = ROOT + "results/"
FIG = ROOT + "figures/"
recs = [json.loads(l) for l in open(RES + "E1_main.jsonl")]
METHODS = ["PL", "GA", "D-SA", "D-SQA", "AL-SA", "AL-SQA", "AL-Tabu"]
RHOS = ["0%", "actual", "50%"]
rows = []
for r in recs:
    ref = r["res"]["MILP"]["cost"]
    base = dict(rho=r["rho"], day=r["day"], date=r["date"], ramp=r["ramp_mean_abs"], D_std=r["D_std"], D_mean=r["D_mean"],
                floor_hours=r["floor_hours"], milp_time=r["res"]["MILP"]["time"], milp_status=r["res"]["MILP"]["status"],
                milp_starts=r["res"]["MILP"]["n_starts"], milp_mud=r["res"]["MILP"]["mud_viol"])
    for m in METHODS:
        v = r["res"][m]
        runs = v if isinstance(v, list) else [v]
        for k, x in enumerate(runs):
            rows.append(dict(**base, method=m, seed=x.get("seed", 0), gap=100 * (x["cost"] / ref - 1), time=x["time"],
                             starts=x["n_starts"], mud=x["mud_viol"], repair=x["repair_actions"],
                             feas=x["feasible_after_repair"], raw_feas=x.get("raw_feasible"),
                             hrs_res=x.get("hours_res_ok"), hrs_bal=x.get("hours_bal_ok"),
                             mism=x.get("mean_abs_mismatch"), link=x.get("link_viol")))
df = pd.DataFrame(rows)
df.to_csv(RES + "E1_long.csv", index=False)
print("instances", len(recs), "MILP status counts", df.drop_duplicates(["rho", "day"]).milp_status.value_counts().to_dict(),
      "MILP time max", df.milp_time.max().round(2), "all feasible after repair", bool(df.feas.all()))

# seed-averaged per day
dm = df.groupby(["rho", "day", "method"]).agg(gap=("gap", "mean"), gap_sd=("gap", "std"), time=("time", "mean"),
        starts=("starts", "mean"), mud=("mud", "mean"), repair=("repair", "mean"), ramp=("ramp", "first"),
        D_std=("D_std", "first"), hrs_bal=("hrs_bal", "mean"), hrs_res=("hrs_res", "mean"), mism=("mism", "mean"),
        link=("link", "mean"), raw_feas=("raw_feas", "mean")).reset_index()

rng = np.random.default_rng(0)
def boot(x, B=10000):
    x = np.asarray(x); i = rng.integers(0, len(x), (B, len(x))); m = x[i].mean(1)
    return np.percentile(m, [2.5, 97.5])

T3 = []
for rho in RHOS:
    for m in METHODS:
        x = dm[(dm.rho == rho) & (dm.method == m)]
        if len(x) == 0: continue
        lo, hi = boot(x.gap)
        T3.append(dict(rho=rho, method=m, n=len(x), mean=x.gap.mean(), median=x.gap.median(), ci_lo=lo, ci_hi=hi,
                       p90=np.percentile(x.gap, 90), max=x.gap.max(), within_0_5=(x.gap <= 0.5).mean() * 100,
                       time=x.time.mean(), starts=x.starts.mean(), mud=x.mud.mean(), seed_sd=x.gap_sd.mean()))
T3 = pd.DataFrame(T3); T3.to_csv(RES + "T4_gaps.csv", index=False)
pd.set_option("display.width", 220)
print(T3.round(3).to_string())
ms = df.drop_duplicates(["rho", "day"]).groupby("rho").agg(milp_starts=("milp_starts", "mean"), milp_mud=("milp_mud", "mean"),
                                                          milp_time=("milp_time", "mean"))
print(ms.round(2))

# raw D-QUBO quality
raw = dm[dm.method.isin(["D-SA", "D-SQA"])].groupby(["rho", "method"]).agg(hrs_bal=("hrs_bal", "mean"), hrs_res=("hrs_res", "mean"),
        mism=("mism", "mean"), link=("link", "mean"), raw_feas=("raw_feas", "mean"), repair=("repair", "mean")).reset_index()
raw.to_csv(RES + "T6_raw_dqubo.csv", index=False); print(raw.round(2).to_string())

# Wilcoxon tests
tests = []
pairs = [("AL-SA", "D-SA"), ("AL-SQA", "D-SQA"), ("AL-SQA", "AL-SA"), ("D-SQA", "D-SA"), ("AL-SA", "PL"), ("AL-SA", "GA"),
         ("AL-Tabu", "AL-SA"), ("GA", "PL")]
for rho in RHOS:
    for a, b in pairs:
        xa = dm[(dm.rho == rho) & (dm.method == a)].set_index("day").gap
        xb = dm[(dm.rho == rho) & (dm.method == b)].set_index("day").gap
        j = xa.index.intersection(xb.index)
        d = (xa[j] - xb[j])
        if (d != 0).sum() < 5:
            p = np.nan
        else:
            p = wilcoxon(xa[j], xb[j], zero_method="wilcox").pvalue
        tests.append(dict(rho=rho, a=a, b=b, n=len(j), mean_diff=d.mean(), a_better=(d < -1e-9).sum(), b_better=(d > 1e-9).sum(), p=p))
tests = pd.DataFrame(tests); tests.to_csv(RES + "T5_wilcoxon.csv", index=False); print(tests.round(4).to_string())

# variability vs gap (Spearman), pooled over rho and per rho
sp = []
for m in METHODS:
    for rho in RHOS + ["pooled"]:
        x = dm[(dm.method == m) & ((dm.rho == rho) if rho != "pooled" else True)]
        if len(x) < 5: continue
        r1, p1 = spearmanr(x.ramp, x.gap)
        sp.append(dict(method=m, rho=rho, n=len(x), rho_s=r1, p=p1))
sp = pd.DataFrame(sp); sp.to_csv(RES + "T7_spearman.csv", index=False); print(sp.round(4).to_string())

# Kruskal across rho for each method
from scipy.stats import kruskal
for m in METHODS:
    g = [dm[(dm.method == m) & (dm.rho == r)].gap.values for r in RHOS]
    if all(len(x) > 3 for x in g): print("Kruskal", m, round(kruskal(*g).pvalue, 5))

# ---------- Fig 2: gap distribution by method and rho (strip + median)
LBL = {"0%": "0%", "actual": "2019 actual", "50%": "50%"}
fig, ax = plt.subplots(figsize=(6.6, 3.6))
w = 0.26
for k, rho in enumerate(RHOS):
    for j, m in enumerate(METHODS):
        x = dm[(dm.rho == rho) & (dm.method == m)].gap.values
        if len(x) == 0: continue
        xpos = j + (k - 1) * w
        jit = (np.random.default_rng(j * 10 + k).random(len(x)) - 0.5) * w * 0.7
        ax.scatter(np.full(len(x), xpos) + jit, np.maximum(x, 1e-3), s=6, color=fs.PAL[k], alpha=0.55, lw=0,
                   label=LBL[rho] if j == 0 else None)
        ax.plot([xpos - w * 0.4, xpos + w * 0.4], [max(np.median(x), 1e-3)] * 2, color=fs.INK, lw=1.4)
ax.set_yscale("log"); ax.set_ylim(1e-3, 60)
ax.set_xticks(range(len(METHODS))); ax.set_xticklabels(METHODS)
ax.set_ylabel("Optimality gap vs MILP (%)")
ax.legend(title="Renewable share", fontsize=8, title_fontsize=8, loc="upper right", ncol=3, markerscale=2)
ax.text(-0.45, 1.25e-3, "zero-gap days plotted at 0.001%", fontsize=7, color=fs.INK2)
fig.tight_layout(); fig.savefig(FIG + "Fig3_gaps.png")

print("figs done")

# ---------- partial Spearman: ramp vs gap controlling for mean net load (rank-residual method)
from scipy.stats import rankdata, pearsonr
def partial_spearman(x, y, z):
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    ex = rx - np.polyval(np.polyfit(rz, rx, 1), rz); ey = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
    return pearsonr(ex, ey)
ps = []
for m in METHODS:
    x = dm[dm.method == m].copy()
    dmean = df.drop_duplicates(["rho", "day"]).set_index(["rho", "day"]).D_mean
    x["D_mean"] = [dmean[(a, b)] for a, b in zip(x.rho, x.day)]
    r_m, p_m = spearmanr(x.D_mean, x.gap)
    r_p, p_p = partial_spearman(x.ramp, x.gap, x.D_mean)
    ps.append(dict(method=m, n=len(x), rho_gap_meanload=r_m, p_meanload=p_m, partial_ramp_given_load=r_p, p_partial=p_p))
ps = pd.DataFrame(ps); ps.to_csv(RES + "T7b_partial.csv", index=False); print(ps.round(4).to_string())
