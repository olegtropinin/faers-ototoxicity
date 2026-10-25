"""Write LaTeX table bodies and a numbers file for the manuscript straight from the result files."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from evaluate import adj_sig, crude_sig, replicated, testable  # noqa: E402

RES, OUT = Path("results"), Path("paper/generated")
OUT.mkdir(parents=True, exist_ok=True)


def ci(o, lo, hi):
    if not np.isfinite(o):
        return "--"
    return f"{o:.2f} ({lo:.2f}--{hi:.2f})"


def pct(v):
    return f"{v['k']}/{v['n']} ({100 * v['rate']:.0f}\\%)"


def table_known():
    t = pd.read_csv(RES / "table_known.csv", index_col=0)
    lines = []
    for d, r in t.iterrows():
        lines.append(f"{d.title()} & {int(r.faers_a)} & {ci(r.faers_ror, r.faers_ror_lo, r.faers_ror_hi)} & "
                     f"{int(r.faers_a_adj)} & {ci(r.faers_mh_or, r.faers_mh_lo, r.faers_mh_hi)} & "
                     f"{int(r.canada_a) if np.isfinite(r.canada_a) else '--'} & "
                     f"{ci(r.canada_ror, r.canada_ror_lo, r.canada_ror_hi)} \\\\")
    (OUT / "table_known.tex").write_text("\n".join(lines) + "\n")


def table_methods():
    rows = []
    for ev, name in (("hl", "Hearing loss"), ("tin", "Tinnitus")):
        s = json.load(open(RES / f"summary_{ev}.json"))
        r, f, c, p, t = s["reference"], s["flow"], s["replication_canada"], s["posthoc"], s["replication_temporal"]
        rows.append(f"\\multicolumn{{3}}{{l}}{{\\textit{{{name}}}}} \\\\")
        rows.append(f"Reference set (positive/negative ingredients) & \\multicolumn{{2}}{{c}}{{{r['n_pos']}/{r['n_neg']}}} \\\\")
        rows.append(f"AUC & {r['auc']['crude_IC025']:.3f} & {r['auc']['adjusted_logMH025']:.3f} \\\\")
        rows.append(f"\\quad difference (95\\% CI) & \\multicolumn{{2}}{{c}}{{{r['auc_diff_adj_minus_crude']:+.3f} "
                    f"({r['auc_diff_ci'][0]:+.3f} to {r['auc_diff_ci'][1]:+.3f})}} \\\\")
        rows.append(f"Sensitivity / specificity of signal rule & {r['crude_sens']:.2f} / {r['crude_spec']:.2f} & "
                    f"{r['adj_sens']:.2f} / {r['adj_spec']:.2f} \\\\")
        rows.append(f"Signals (FAERS without Canadian reports) & {f['crude_signals']} & {f['adjusted_signals']} \\\\")
        rows.append(f"\\quad lost / gained on adjustment & \\multicolumn{{2}}{{c}}{{{f['crude_not_adjusted']} / "
                    f"{f['adjusted_not_crude']}}} \\\\")
        rows.append(f"Replicated in Canada, crude rule: labelled adjusted signals & \\multicolumn{{2}}{{c}}{{{pct(c['adjusted_labelled'])}}} \\\\")
        rows.append(f"\\quad unlabelled adjusted signals & \\multicolumn{{2}}{{c}}{{{pct(c['adjusted_unlabelled'])}}} \\\\")
        rows.append(f"\\quad crude-only signals & \\multicolumn{{2}}{{c}}{{{pct(c['crude_only_signals'])}}} \\\\")
        rows.append(f"\\quad label-negative non-signals (background) & \\multicolumn{{2}}{{c}}{{{pct(c['label_negative_non_signals'])}}} \\\\")
        rows.append(f"Replicated, adjusted in both databases$^a$: labelled & \\multicolumn{{2}}{{c}}{{{pct(p['adjrepl_adjusted_x_labelled'])}}} \\\\")
        rows.append(f"\\quad unlabelled & \\multicolumn{{2}}{{c}}{{{pct(p['adjrepl_adjusted_x_unlabelled'])}}} \\\\")
        rows.append(f"\\quad crude-only & \\multicolumn{{2}}{{c}}{{{pct(p['adjrepl_crude_only'])}}} \\\\")
        rows.append(f"\\quad background & \\multicolumn{{2}}{{c}}{{{pct(p['adjrepl_label_negative_non_signals'])}}} \\\\")
        rows.append(f"Temporal ($\\leq$2014 $\\rightarrow$ $\\geq$2015): adjusted / crude-only / background & "
                    f"\\multicolumn{{2}}{{c}}{{{pct(t['early_adjusted'])} / {pct(t['early_crude_only'])} / "
                    f"{pct(t['label_negative_non_signals'])}}} \\\\")
        rows.append("\\midrule")
    (OUT / "table_methods.tex").write_text("\n".join(rows[:-1]) + "\n")


def table_prior():
    m = pd.read_csv(RES / "hl_master.csv", index_col="ingredient", low_memory=False)
    m["crude"], m["adj"] = crude_sig(m, "faers"), adj_sig(m, "faers")
    m["t"], m["r"] = testable(m, "canada"), replicated(m, "canada")
    lines, nums = [], {}
    names = {"li2024": "Li et al. 2024 (all ages)", "liu2025": "Liu et al. 2025 (children)"}
    for st, label in names.items():
        p = pd.read_csv(f"data/raw/prior/{st}.csv")
        for lab_flag, lab_name in (("yes", "labelled"), ("no", "unlabelled")):
            d = p[p["labeled_per_authors"] == lab_flag]["drug"].str.upper().unique()
            x = m.reindex(d)
            row = [len(d), int(x["crude"].sum()), int(x["adj"].sum()), int((x["adj"] & x["t"]).sum()),
                   int((x["adj"] & x["r"]).sum())]
            nums[f"{st}_{lab_name}"] = row
            lines.append(f"{label} & {lab_name} & " + " & ".join(map(str, row)) + " \\\\")
    (OUT / "table_prior.tex").write_text("\n".join(lines) + "\n")
    return nums


def table_candidates():
    rv = pd.read_csv(RES / "case_review.csv")
    rv = rv[rv["category"] == "E"]
    lines = []
    for _, c in rv.sort_values(["event", "cases"], ascending=[True, False]).iterrows():
        r = pd.read_csv(RES / f"{c.event}_robust_posthoc.csv", index_col=0, low_memory=False).loc[c.drug]
        lines.append(f"{c.drug.title()} & {'Hearing loss' if c.event == 'hl' else 'Tinnitus'} & {int(c.cases)} & "
                     f"{ci(r.faersnocax_mh_or, r.faersnocax_mh_lo, r.faersnocax_mh_hi)} & "
                     f"{int(r.canadanostudy_a)} & {ci(r.canadanostudy_ror, r.canadanostudy_ror_lo, r.canadanostudy_ror_hi)} & "
                     f"{c.consumer.replace('%', chr(92) + '%')} & {int(c.pubmed_n)} \\\\")
    (OUT / "table_candidates.tex").write_text("\n".join(lines) + "\n")


def category_counts():
    rv = pd.read_csv(RES / "case_review.csv")
    return rv.groupby(["event", "category"]).size().unstack(fill_value=0).to_dict(orient="index")


if __name__ == "__main__":
    table_known()
    table_methods()
    nums = {"prior": table_prior(), "categories": category_counts()}
    table_candidates()
    json.dump(nums, open(OUT / "numbers.json", "w"), indent=1, default=int)
    print(json.dumps(nums, indent=1, default=int))
