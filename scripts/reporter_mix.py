"""Per-ingredient provenance of FAERS (non-Canadian) cases: share reported by consumers and share from the US,
for hearing loss and tinnitus cases where the ingredient is suspect (>= 3 cases)."""
import sys

import pandas as pd

sys.path.insert(0, "src")
from signals import HL_PTS, TIN_PTS  # noqa: E402

F = "data/work/faers/"
rep = pd.read_parquet(F + "reports.parquet", columns=["rid", "country", "occp_cod"])
rep = rep[rep["country"] != "CA"].set_index("rid")
ing = pd.read_csv(F + "ing_vocab.csv", index_col="code", keep_default_na=False)["name"]
pt = pd.read_csv(F + "pt_vocab.csv", index_col="code", keep_default_na=False)["name"]
dr = pd.read_parquet(F + "drugs.parquet")
dr = dr[dr["role"].isin(["PS", "SS", "I"])][["rid", "ing"]].drop_duplicates()
rc = pd.read_parquet(F + "reacs.parquet")
out = []
for ev, pts in (("hl", HL_PTS), ("tin", TIN_PTS)):
    cases = set(rc.loc[rc["pt"].isin(set(pt[pt.isin(pts)].index)), "rid"]) & set(rep.index)
    d = dr[dr["rid"].isin(cases)].copy()
    d["consumer"] = (rep.loc[d["rid"].values, "occp_cod"].values == "CN")
    d["us"] = (rep.loc[d["rid"].values, "country"].values == "US")
    g = d.groupby("ing").agg(cases=("rid", "size"), consumer=("consumer", "mean"), us=("us", "mean"))
    g = g[g["cases"] >= 3]
    g.index = ing.reindex(g.index).values
    g["event"] = ev
    out.append(g)
pd.concat(out).rename_axis("ingredient").to_csv("results/reporter_mix.csv")
print(pd.concat(out).describe())
