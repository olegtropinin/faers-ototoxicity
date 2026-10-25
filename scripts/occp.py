"""Reporter-type mix of FAERS hearing-loss cases for selected suspect ingredients."""
import sys
import pandas as pd
sys.path.insert(0, "src")
from signals import HL_PTS
F = "data/work/faers/"
rep = pd.read_parquet(F + "reports.parquet", columns=["rid", "occp_cod", "country"])
ing = pd.read_csv(F + "ing_vocab.csv", index_col="code")["name"]
pt = pd.read_csv(F + "pt_vocab.csv", index_col="code", keep_default_na=False)["name"]
hl_codes = set(pt[pt.isin(HL_PTS)].index)
rc = pd.read_parquet(F + "reacs.parquet")
hl = set(rc.loc[rc["pt"].isin(hl_codes), "rid"])
rep["hl"] = rep["rid"].isin(hl)
print("all HL", rep[rep.hl].occp_cod.value_counts(normalize=True).round(2).head(4).to_dict())
dr = pd.read_parquet(F + "drugs.parquet")
drugs = sys.argv[1].split(",")
codes = {n: c for c, n in ing.items() if n in drugs}
dr = dr[dr["ing"].isin(codes.values()) & dr["role"].isin(["PS", "SS", "I"])]
for n, c in codes.items():
    ids = set(dr.loc[dr["ing"] == c, "rid"])
    x = rep[rep["rid"].isin(ids) & rep["hl"]]
    print(f"{n:22s} HL={len(x):5d}", x.occp_cod.value_counts(normalize=True).round(2).head(3).to_dict())
