"""Build paper/supplement.tex (Tables S2-S5) from result files; S1 (READUS-PV) is maintained by hand."""
from pathlib import Path

import numpy as np
import pandas as pd

RES, P = Path("results"), Path("paper")


def esc(s):
    return str(s).replace("&", "\\&").replace("%", "\\%").replace("_", "\\_").replace("#", "\\#")


def ci(o, lo, hi):
    return "--" if not np.isfinite(o) else f"{o:.2f} ({lo:.2f}--{hi:.2f})"


def s2_known():
    t = pd.read_csv(RES / "table_known.csv", index_col=0)
    rows = [f"{esc(d.title())} & {int(r.faers_a)} & {ci(r.faers_ror, r.faers_ror_lo, r.faers_ror_hi)} & "
            f"{int(r.faers_a_adj)} & {ci(r.faers_mh_or, r.faers_mh_lo, r.faers_mh_hi)} & "
            f"{int(r.canada_a)} & {ci(r.canada_ror, r.canada_ror_lo, r.canada_ror_hi)} \\\\" for d, r in t.iterrows()]
    return "\n".join(rows)


def s3_review():
    rv = pd.read_csv(RES / "case_review.csv")
    a2 = pd.read_csv(RES / "case_review_assessor2.csv").rename(columns={"category": "cat2", "basis": "basis2"})
    rv = rv.merge(a2, on=["drug", "event"], how="left")
    rows = []
    for _, r in rv.iterrows():
        rows.append(f"{esc(r.drug.title())} & {'HL' if r.event == 'hl' else 'Tin'} & {r.category} & {r.cat2} & {int(r.cases)} & "
                    f"{esc(r.top_country)} & {esc(r.top_year)} & {esc(r.consumer)} & {r.median_drugs:g} & "
                    f"{esc(r.mega_ge15_drugs)} & {esc(r.with_known_ototoxic)} & {esc(r.within_indication) if isinstance(r.within_indication, str) else ''} & {esc(r.note)} \\\\")
    return "\n".join(rows)


def s4_characteristics():
    c = pd.read_csv(RES / "case_characteristics.csv")
    names = {"age_band": "Age", "sex_": "Sex", "reporter": "Reporter", "region": "Country", "period": "FDA receipt"}
    order = {"age_band": ["<18", "18-44", "45-64", ">=65", "unknown"], "sex_": ["female", "male", "unknown"],
             "reporter": ["consumer", "physician", "pharmacist", "other health professional", "lawyer", "unknown"],
             "region": ["United States", "other", "unknown"], "period": ["2004-2012", "2013-2018", "2019-2026"]}
    groups = ["hearing loss", "tinnitus", "all other reports"]
    n = {g: int(c[(c.group == g) & (c.variable == "N")]["n"].iloc[0]) for g in groups}
    rows = ["Reports, n & " + " & ".join(f"{n[g]:,}" for g in groups) + " \\\\"]
    med = {g: c[(c.group == g) & (c.variable == "age_median")]["n"].iloc[0] for g in groups}
    rows.append("Age, median (years) & " + " & ".join(f"{med[g]:.0f}" for g in groups) + " \\\\")
    for v, lv in order.items():
        rows.append(f"\\textit{{{names[v]}}} & & & \\\\")
        for l in lv:
            vals = []
            for g in groups:
                x = c[(c.group == g) & (c.variable == v) & (c.level == l)]
                vals.append(f"{x['pct'].iloc[0]:.1f}" if len(x) else "0.0")
            label = esc(l).replace(">=", "$\\geq$")
            rows.append(f"\\quad {label} (\\%) & " + " & ".join(vals) + " \\\\")
    return "\n".join(rows)


def s5_all_robust():
    out = []
    for ev in ("hl", "tin"):
        r = pd.read_csv(RES / f"{ev}_robust_x.csv", index_col=0, low_memory=False)
        for d, x in r.iterrows():
            out.append(f"{esc(d.title())} & {'HL' if ev == 'hl' else 'Tin'} & {int(x.faersnocax_a_adj)} & "
                       f"{ci(x.faersnocax_mh_or, x.faersnocax_mh_lo, x.faersnocax_mh_hi)} & {int(x.canadax_a_adj)} & "
                       f"{ci(x.canadax_mh_or, x.canadax_mh_lo, x.canadax_mh_hi)} & "
                       f"{'yes' if str(x.labelled) == 'True' else 'no'} \\\\")
    return "\n".join(out)


def s6_prior():
    p = pd.read_csv(RES / "prior_screens.csv")
    p = p[~p.study.isin(["li2024", "liu2025"])]
    return "\n".join(f"{esc(r.study)} & {esc(r.scope)} & {r.reported} & {r.crude_here} & {r.adjusted_here} & "
                     f"{r.adjusted_replicated} & {r.posthoc_robust} & {esc(str(r.posthoc_robust_drugs) if isinstance(r.posthoc_robust_drugs, str) else '')} \\\\"
                     for r in p.itertuples())


if __name__ == "__main__":
    tpl = (P / "supplement_template.tex").read_text()
    tex = (tpl.replace("%%S2%%", s2_known()).replace("%%S3%%", s3_review())
              .replace("%%S4%%", s4_characteristics()).replace("%%S5%%", s5_all_robust())
              .replace("%%S6%%", s6_prior()))
    (P / "supplement.tex").write_text(tex)
    print("written paper/supplement.tex")
