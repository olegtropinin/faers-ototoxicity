"""Characteristics of hearing-loss and tinnitus cases vs all other FAERS reports (READUS-PV 8b)."""
import sys
import pandas as pd
sys.path.insert(0, "src")
from signals import HL_PTS, TIN_PTS
F = "data/work/faers/"
rep = pd.read_parquet(F + "reports.parquet")
pt = pd.read_csv(F + "pt_vocab.csv", index_col="code", keep_default_na=False)["name"]
rc = pd.read_parquet(F + "reacs.parquet")
rep["hl"] = rep["rid"].isin(set(rc.loc[rc["pt"].isin(set(pt[pt.isin(HL_PTS)].index)), "rid"]))
rep["tin"] = rep["rid"].isin(set(rc.loc[rc["pt"].isin(set(pt[pt.isin(TIN_PTS)].index)), "rid"]))
age = rep["age_y"]
rep["age_band"] = pd.cut(age, [-1, 17.999, 44.999, 64.999, 200], labels=["<18", "18-44", "45-64", ">=65"]).astype(str).replace("nan", "unknown")
rep["sex_"] = rep["sex"].map({"M": "male", "F": "female"}).fillna("unknown")
rep["reporter"] = rep["occp_cod"].map({"MD": "physician", "HP": "other health professional", "PH": "pharmacist",
                                        "CN": "consumer", "LW": "lawyer", "OT": "other health professional"}).fillna("unknown")
rep["region"] = rep["country"].map(lambda c: "United States" if c == "US" else ("unknown" if c in ("", None) else "other"))
rep["period"] = pd.cut(rep["year"].fillna(0), [0, 2012, 2018, 2100], labels=["2004-2012", "2013-2018", "2019-2026"]).astype(str)
rows = []
for grp, mask in (("hearing loss", rep["hl"]), ("tinnitus", rep["tin"]), ("all other reports", ~rep["hl"] & ~rep["tin"])):
    sub = rep[mask]
    rows.append({"group": grp, "variable": "N", "level": "", "n": len(sub), "pct": 100.0})
    for v in ("age_band", "sex_", "reporter", "region", "period"):
        vc = sub[v].value_counts()
        for lvl, n in vc.items():
            rows.append({"group": grp, "variable": v, "level": lvl, "n": int(n), "pct": round(100 * n / len(sub), 1)})
    rows.append({"group": grp, "variable": "age_median", "level": "", "n": float(sub["age_y"].median()), "pct": None})
pd.DataFrame(rows).to_csv("results/case_characteristics.csv", index=False)
print(pd.DataFrame(rows).pivot_table(index=["variable", "level"], columns="group", values="pct").to_string())
