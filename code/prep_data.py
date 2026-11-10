import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))).replace("\\", "/") + "/"
import pandas as pd, numpy as np, json
src=ROOT + "data/opsd_60min.csv"
cols=["utc_timestamp","cet_cest_timestamp","DE_load_actual_entsoe_transparency","DE_solar_generation_actual",
      "DE_wind_onshore_generation_actual","DE_wind_offshore_generation_actual"]
df=pd.read_csv(src,usecols=cols,on_bad_lines="skip")
df["t"]=pd.to_datetime(df["cet_cest_timestamp"].str[:13],format="%Y-%m-%dT%H")
d=df[(df.t>="2019-01-01")&(df.t<"2020-01-01")].copy()
print("rows",len(d)); print(d.isna().sum())
d=d.drop_duplicates("t").set_index("t").sort_index()
full=pd.date_range("2019-01-01","2019-12-31 23:00",freq="h")
d=d.reindex(full)
print("missing after reindex",d.isna().sum().to_dict())
d=d.drop(columns=["utc_timestamp","cet_cest_timestamp"]).interpolate(limit=3).ffill().bfill()
out=pd.DataFrame({"load_MW":d.DE_load_actual_entsoe_transparency,"solar_MW":d.DE_solar_generation_actual,
  "wind_MW":d.DE_wind_onshore_generation_actual+d.DE_wind_offshore_generation_actual},index=full)
out.index.name="local_time"
out.to_csv(ROOT + "data/DE_2019_hourly.csv")
print(out.describe().round(0))
print("annual TWh",(out.sum()/1e6).round(2).to_dict())
