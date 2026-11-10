import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))).replace("\\", "/") + "/"
import pandas as pd, json
d=pd.DataFrame([json.loads(l) for l in open(ROOT + "results/E23_sens.jsonl")])
print(len(d), d.groupby("K").size().to_dict())
g=d.groupby(["K","alpha"]).agg(n=("gap","size"),vars=("n","first"),gap=("gap","mean"),gap_med=("gap","median"),bal=("hours_bal_ok","mean"),res=("hours_res_ok","mean"),mism=("mean_abs_mismatch","mean"),link=("link_viol","mean"),raw=("raw_feasible","mean"),starts=("n_starts","mean"),rep=("repair","mean")).reset_index()
g.to_csv(ROOT + "results/T8_sens.csv",index=False); print(g.round(2).to_string())
print("raw feasible runs:", int(d.raw_feasible.sum()), "of", len(d), d[d.raw_feasible].groupby(["K","alpha"]).size().to_dict())
