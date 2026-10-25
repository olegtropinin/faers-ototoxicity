"""Case-level profile for a list of (drug, event) candidates: concentration and report-size indicators.
FAERS reports of Canadian origin are excluded, matching the replication analysis."""
import sys
import pandas as pd
sys.path.insert(0, "src")
from signals import HL_PTS, TIN_PTS, KNOWN_OTOTOXIC
F = "data/work/faers/"
cands = [l.strip().split("\t") for l in open(sys.argv[1]) if l.strip()]
rep = pd.read_parquet(F + "reports.parquet")
rep = rep[rep["country"] != "CA"].set_index("rid")
ing = pd.read_csv(F + "ing_vocab.csv", index_col="code", keep_default_na=False)["name"]
pt = pd.read_csv(F + "pt_vocab.csv", index_col="code", keep_default_na=False)["name"]
dr = pd.read_parquet(F + "drugs.parquet")
ndrug = dr.groupby("rid").size()
rc = pd.read_parquet(F + "reacs.parquet")
nreac = rc.groupby("rid").size()
ind = pd.read_parquet(F + "indis.parquet")
kcodes = set(ing[ing.isin(KNOWN_OTOTOXIC)].index)
rows = []
for drug, ev in cands:
    pts = HL_PTS if ev == "hl" else TIN_PTS
    code = ing[ing == drug].index
    if len(code) == 0:
        continue
    code = int(code[0])
    ids = set(dr.loc[(dr["ing"] == code) & dr["role"].isin(["PS", "SS", "I"]), "rid"])
    evc = set(pt[pt.isin(pts)].index)
    cases = sorted(set(rc.loc[rc["pt"].isin(evc), "rid"]) & ids & set(rep.index))
    x = rep.loc[cases]
    co = dr[dr["rid"].isin(cases) & (dr["ing"] != code)]
    ii = ind[ind["rid"].isin(cases)]["indi_pt"].map(pt)
    nd = ndrug.reindex(cases)
    rows.append({
        "drug": drug, "event": ev, "cases": len(cases),
        "top_country": f"{x['country'].value_counts().index[0]} {x['country'].value_counts(normalize=True).iloc[0]:.0%}",
        "top_year": f"{int(x['year'].value_counts().index[0])} {x['year'].value_counts(normalize=True).iloc[0]:.0%}",
        "consumer": f"{(x['occp_cod'] == 'CN').mean():.0%}",
        "median_drugs": float(nd.median()), "mega_ge15_drugs": f"{(nd >= 15).mean():.0%}",
        "median_reactions": float(nreac.reindex(cases).median()),
        "with_known_ototoxic": f"{co['ing'].isin(kcodes).groupby(co['rid']).any().reindex(cases, fill_value=False).mean():.0%}",
        "top_coreported": "; ".join(co["ing"].map(ing).value_counts().head(3).index),
        "top_indication": "; ".join(ii[~ii.isin(["PRODUCT USED FOR UNKNOWN INDICATION", ""])].value_counts().head(2).index),
        "female": f"{(x['sex'] == 'F').mean():.0%}", "age_median": x["age_y"].median(),
        "foetal_exposure": f"{ii.str.contains('FOETAL EXPOSURE|FETAL EXPOSURE', na=False).groupby(ind.loc[ii.index, 'rid']).any().reindex(cases, fill_value=False).mean():.0%}",
        "top_indications": "; ".join(f"{k} ({v})" for k, v in ii[~ii.isin(["PRODUCT USED FOR UNKNOWN INDICATION", ""])].value_counts().head(4).items()),
    })
pd.DataFrame(rows).to_csv(sys.argv[2], index=False)
print(pd.DataFrame(rows).to_string())
