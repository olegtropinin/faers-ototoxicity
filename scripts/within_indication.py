"""Active-comparator check (post hoc): ROR of a drug vs all other suspect drugs, restricted to FAERS reports
(non-Canadian) whose indication matches the drug's main indication. If the crude signal were due to the indication,
it should vanish here. Also: co-reported intracranial-hypertension terms among doxycycline tinnitus cases."""
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/homeserver/faers/src")
from signals import HL_PTS, TIN_PTS  # noqa: E402

F = "/home/homeserver/faers/data/work/faers/"
CHECKS = [  # drug, event, indication regex
    ("PREGABALIN", "hl", r"PAIN|NEURALGIA|NEUROPATHY|FIBROMYALGIA"),
    ("AMLODIPINE", "hl", r"HYPERTENSION"),
    ("CHLORTHALIDONE", "tin", r"HYPERTENSION"),
    ("ETANERCEPT", "hl", r"RHEUMATOID ARTHRITIS"),
    ("TOFACITINIB", "hl", r"RHEUMATOID ARTHRITIS"),
    ("GLYCOPYRROLATE", "hl", r"CHRONIC OBSTRUCTIVE|EMPHYSEMA|COPD"),
    ("TRAVOPROST", "hl", r"GLAUCOMA|INTRAOCULAR PRESSURE|OCULAR HYPERTENSION"),
    ("INSULIN ASPART", "hl", r"DIABETES"),
    ("GALCANEZUMAB", "tin", r"MIGRAINE"),
    ("LIOTHYRONINE", "tin", r"HYPOTHYROIDISM"),
    ("DOXYCYCLINE", "tin", r"INFECTION|ACNE|ROSACEA|LYME|PNEUMONIA|BRONCHITIS|Q FEVER"),
]
rep = pd.read_parquet(F + "reports.parquet", columns=["rid", "country"])
keep = set(rep.loc[rep["country"] != "CA", "rid"])
ing = pd.read_csv(F + "ing_vocab.csv", index_col="code", keep_default_na=False)["name"]
pt = pd.read_csv(F + "pt_vocab.csv", index_col="code", keep_default_na=False)["name"]
dr = pd.read_parquet(F + "drugs.parquet")
dr = dr[dr["role"].isin(["PS", "SS", "I"]) & dr["rid"].isin(keep)]
rc = pd.read_parquet(F + "reacs.parquet")
ind = pd.read_parquet(F + "indis.parquet")
ind = ind[ind["rid"].isin(keep)]
ind_names = pt.reindex(ind["indi_pt"].values).values
ev_rids = {"hl": set(rc.loc[rc["pt"].isin(set(pt[pt.isin(HL_PTS)].index)), "rid"]),
           "tin": set(rc.loc[rc["pt"].isin(set(pt[pt.isin(TIN_PTS)].index)), "rid"])}
rows = []
for drug, ev, rx in CHECKS:
    pop = set(ind["rid"].values[pd.Series(ind_names).str.contains(rx, na=False).values])
    code = int(ing[ing == drug].index[0])
    exposed = set(dr.loc[dr["ing"] == code, "rid"]) & pop
    other = pop - exposed
    e = ev_rids[ev]
    a, b = len(exposed & e), len(exposed - e)
    c, d = len(other & e), len(other - e)
    ror = (a * d) / (b * c) if b and c and a else np.nan
    se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d) if a and b and c and d else np.nan
    rows.append({"drug": drug, "event": ev, "indication": rx, "reports_in_indication": len(pop), "exposed": len(exposed),
                 "a": a, "ror": ror, "lo": np.exp(np.log(ror) - 1.96 * se), "hi": np.exp(np.log(ror) + 1.96 * se)})
out = pd.DataFrame(rows)
out.to_csv("/home/homeserver/faers/results/within_indication.csv", index=False)
print(out.round(2).to_string())

# doxycycline tinnitus: share of cases co-reporting intracranial hypertension features
code = int(ing[ing == "DOXYCYCLINE"].index[0])
cases = (set(dr.loc[dr["ing"] == code, "rid"]) & ev_rids["tin"])
iih = re.compile(r"INTRACRANIAL (PRESSURE INCREASED|HYPERTENSION)|PAPILLOEDEMA|IDIOPATHIC INTRACRANIAL|BENIGN INTRACRANIAL|PSEUDOTUMOR")
vis = re.compile(r"VISION BLURRED|VISUAL IMPAIRMENT|DIPLOPIA|VISUAL ACUITY REDUCED")
sub = rc[rc["rid"].isin(cases)]
names = pt.reindex(sub["pt"].values).values
s = pd.Series(names)
print("doxycycline tinnitus cases", len(cases),
      "IIH/papilloedema", sub["rid"].values[s.str.contains(iih).values].size and len(set(sub["rid"].values[s.str.contains(iih).values])),
      "headache", len(set(sub["rid"].values[(s == "HEADACHE").values])),
      "visual", len(set(sub["rid"].values[s.str.contains(vis).values])))
