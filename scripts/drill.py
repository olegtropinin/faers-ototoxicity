"""Case-level profile of FAERS reports for drug X with event set (hl|tin)."""
import sys
import pandas as pd
sys.path.insert(0, "src")
from signals import HL_PTS, TIN_PTS
F = "data/work/faers/"
drug, ev = sys.argv[1], sys.argv[2]
pts = HL_PTS if ev == "hl" else TIN_PTS
rep = pd.read_parquet(F + "reports.parquet")
ing = pd.read_csv(F + "ing_vocab.csv", index_col="code", keep_default_na=False)["name"]
pt = pd.read_csv(F + "pt_vocab.csv", index_col="code", keep_default_na=False)["name"]
code = int(ing[ing == drug].index[0])
dr = pd.read_parquet(F + "drugs.parquet")
ids = set(dr.loc[(dr["ing"] == code) & dr["role"].isin(["PS", "SS", "I"]), "rid"])
rc = pd.read_parquet(F + "reacs.parquet")
evc = set(pt[pt.isin(pts)].index)
case_ids = set(rc.loc[rc["pt"].isin(evc), "rid"]) & ids
x = rep[rep["rid"].isin(case_ids)]
print(drug, ev, "cases", len(x))
print("year", x["year"].value_counts().sort_index().to_dict())
print("reporter", x["occp_cod"].value_counts().head(5).to_dict())
print("country", x["country"].value_counts().head(5).to_dict())
print("sex", x["sex"].value_counts().to_dict(), "age median", x["age_y"].median())
co = dr[dr["rid"].isin(case_ids) & (dr["ing"] != code)]
print("top co-reported", co["ing"].map(ing).value_counts().head(8).to_dict())
ind = pd.read_parquet(F + "indis.parquet")
ii = ind[ind["rid"].isin(case_ids)]
print("top indications", ii["indi_pt"].map(pt).value_counts().head(6).to_dict())
cr = rc[rc["rid"].isin(case_ids) & ~rc["pt"].isin(evc)]
print("top other reactions", cr["pt"].map(pt).value_counts().head(10).to_dict())
