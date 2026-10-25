"""Check signals.py against statsmodels on synthetic reports."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.contingency_tables import StratifiedTable, Table2x2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from signals import run, strata  # noqa: E402

rng = np.random.default_rng(1)
R = 40000
rep = pd.DataFrame({"rid": np.arange(R), "age_y": rng.choice([10, 30, 50, 70, np.nan], R),
                    "sex": rng.choice(["M", "F", ""], R), "year": rng.choice([2008, 2015, 2022], R)})
drugA = rng.random(R) < 0.05 + 0.10 * (rep["age_y"].fillna(0).values > 60)    # confounded by age
p_ev = 0.01 + 0.03 * (rep["age_y"].fillna(0).values > 60) + 0.02 * drugA        # true effect + age effect
ev = rng.random(R) < p_ev
drugB = rng.random(R) < 0.08
sus = pd.concat([pd.DataFrame({"rid": np.where(drugA)[0], "ing": "A"}),
                 pd.DataFrame({"rid": np.where(drugB)[0], "ing": "B"}),
                 pd.DataFrame({"rid": np.where(~drugA & ~drugB)[0], "ing": "OTHER"})])
reac = pd.DataFrame({"rid": np.arange(R), "pt": np.where(ev, "DEAFNESS", "NAUSEA")})
indi = pd.DataFrame({"rid": [0], "indi_pt": ["HYPERTENSION"]})

out = run(rep, sus, sus, reac, indi, {"DEAFNESS"})

t = Table2x2(np.array([[ev[drugA].sum(), (~ev[drugA]).sum()], [ev[~drugA].sum(), (~ev[~drugA]).sum()]]))
assert abs(out.loc["A", "ror"] - t.oddsratio) < 1e-9, (out.loc["A", "ror"], t.oddsratio)
assert abs(out.loc["A", "ror_lo"] - t.oddsratio_confint()[0]) < 1e-9

st = strata(rep)
tables = []
for s in np.unique(st):
    m = st == s
    a = (drugA & ev & m).sum(); b = (drugA & ~ev & m).sum()
    c = (~drugA & ev & m).sum(); d = (~drugA & ~ev & m).sum()
    tables.append(np.array([[a, b], [c, d]]))
stt = StratifiedTable(np.dstack(tables))
assert abs(out.loc["A", "mh_or"] - stt.oddsratio_pooled) < 1e-9, (out.loc["A", "mh_or"], stt.oddsratio_pooled)
lo, hi = stt.oddsratio_pooled_confint()
assert abs(out.loc["A", "mh_lo"] - lo) < 1e-6 and abs(out.loc["A", "mh_hi"] - hi) < 1e-6, (out.loc["A", ["mh_lo", "mh_hi"]], lo, hi)
print("crude A", round(t.oddsratio, 3), "MH A", round(stt.oddsratio_pooled, 3), "B crude", round(out.loc["B", "ror"], 3))
print("OK")
