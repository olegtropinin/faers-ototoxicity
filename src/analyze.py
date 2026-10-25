"""Run the disproportionality analysis in FAERS (all countries, and excluding Canadian
reports) and in Canada Vigilance, for hearing loss and tinnitus; join label status.

Writes results/<event>_master.csv and results/population.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from drugnorm import canonical, clean, strip_salt  # noqa: E402
from signals import HL_PTS, STRONG_OTOTOXIC, TIN_PTS, run  # noqa: E402

F = Path("data/work/faers")
C = Path("data/work/canada")
RES = Path("results")
RES.mkdir(exist_ok=True)


def load_faers():
    rep = pd.read_parquet(F / "reports.parquet")
    ing = pd.read_csv(F / "ing_vocab.csv", index_col="code")["name"]
    pt = pd.read_csv(F / "pt_vocab.csv", index_col="code", keep_default_na=False)["name"]
    # vocab codes are 0..K-1 in creation order, so they are valid categorical codes
    assert (ing.index == np.arange(len(ing))).all() and (pt.index == np.arange(len(pt))).all()
    ing_cats, pt_cats = ing.fillna("").astype(str).values, pt.astype(str).values
    dr = pd.read_parquet(F / "drugs.parquet")
    dr["ing"] = pd.Categorical.from_codes(dr["ing"].values, categories=pd.Index(ing_cats).unique()) \
        if pd.Index(ing_cats).is_unique else ing.reindex(dr["ing"].values).values
    rc = pd.read_parquet(F / "reacs.parquet")
    rc["pt"] = pd.Categorical.from_codes(rc["pt"].values, categories=pt_cats)
    ind = pd.read_parquet(F / "indis.parquet")
    ind["indi_pt"] = pd.Categorical.from_codes(ind["indi_pt"].values, categories=pt_cats)
    return rep, dr, rc, ind


def reindex(rep, dr, rc, ind):
    """Renumber reports 0..R-1 (required by the bincount-based statistics)."""
    rep = rep.reset_index(drop=True)
    m = pd.Series(np.arange(len(rep)), index=rep["rid"].values)
    out = []
    for df in (dr, rc, ind):
        df = df[df["rid"].isin(m.index)].copy()
        df["rid"] = m.loc[df["rid"].values].values
        out.append(df)
    rep = rep.assign(rid=np.arange(len(rep)))
    return (rep, *out)


def analyse(name, rep, dr, rc, ind, suspect_roles, lawyer_col=None, **kw):
    rep, dr, rc, ind = reindex(rep, dr, rc, ind)
    sus = dr[dr["role"].isin(suspect_roles)][["rid", "ing"]].drop_duplicates()
    lawyer = (rep[lawyer_col] == "LW").values if lawyer_col else None
    res, pops = {}, {}
    for ev_name, pts in (("hl", HL_PTS), ("tin", TIN_PTS)):
        r = run(rep, sus, dr, rc, ind, pts, lawyer, **kw)
        res[ev_name] = r.add_prefix(f"{name}_")
        pops[f"{name}_{ev_name}"] = r.attrs["population"]
        print(name, ev_name, r.attrs["population"], flush=True)
    return res, pops


def label_table():
    lab = pd.read_parquet("data/work/labels.parquet")
    vocab = set(pd.read_csv(F / "ing_vocab.csv")["name"].dropna())
    lab["ing"] = lab["substance"].map(lambda s: canonical(s if s in vocab else strip_salt(s, vocab)))
    lab = lab[lab["ing"] != ""]
    g = lab.groupby("ing").agg(n_labels=("set_id", "size"), rx=("rx", "any"), hl_any=("hl_any", "any"),
                               hl_strong=("hl_strong", "any"), tin_any=("tin_any", "any"))
    rx = lab[lab["rx"]].groupby("ing").agg(n_rx=("set_id", "size"), hl_frac_rx=("hl_any", "mean"),
                                           tin_frac_rx=("tin_any", "mean"))
    g = g.join(rx, how="left")
    # reference classes: consistent Rx labelling = positive; no hearing/tinnitus wording anywhere = negative
    g["ref_hl"] = np.where(g["hl_frac_rx"] >= 0.5, "pos",
                           np.where(~g["hl_any"] & ~g["tin_any"], "neg", "ambiguous"))
    g["ref_tin"] = np.where(g["tin_frac_rx"] >= 0.5, "pos",
                            np.where(~g["tin_any"] & ~g["hl_any"], "neg", "ambiguous"))
    # labels of combination products, used only for the novelty check ("is the event on any US label?")
    cb = pd.read_parquet("data/work/labels_combo.parquet")
    cb["ing"] = cb["substance"].map(lambda s: canonical(s if s in vocab else strip_salt(s, vocab)))
    cb = cb[cb["ing"] != ""].groupby("ing").agg(hl_combo=("hl_any", "any"), tin_combo=("tin_any", "any"))
    g = g.join(cb, how="outer")
    return g


def save(parts, pops, tag):
    for name, res in parts.items():
        for ev, df in res.items():
            df.to_csv(RES / f"part_{name}_{ev}.csv", index_label="ingredient")
    json.dump(pops, open(RES / f"population_{tag}.json", "w"), indent=1)


def join():
    labels = label_table()
    pops = {}
    for f in sorted(RES.glob("population_*.json")):
        pops.update(json.load(open(f)))
    json.dump(pops, open(RES / "population.json", "w"), indent=1)
    for ev in ("hl", "tin"):
        parts = [pd.read_csv(f, index_col="ingredient") for f in sorted(RES.glob(f"part_*_{ev}.csv"))]
        m = parts[0].join(parts[1:], how="outer").join(labels, how="left")
        m.index.name = "ingredient"
        m.to_csv(RES / f"{ev}_master.csv")
        print(ev, m.shape)


# Post-hoc / sensitivity variants (see PROTOCOL.md deviations 3-4). Every variant uses the extended
# indication filter; they differ in how ototoxic co-medication and report clusters are handled.
VARIANTS = {
    "x": dict(extended_indication=True),
    "xstrat": dict(extended_indication=True, comed="stratify"),
    "xstrong": dict(extended_indication=True, ototoxic=STRONG_OTOTOXIC),
    "xnomega": dict(extended_indication=True, mega_max=15),
}


def load_canada():
    return (pd.read_parquet(C / "an_reports.parquet"), pd.read_parquet(C / "an_drugs.parquet"),
            pd.read_parquet(C / "an_reacs.parquet"), pd.read_parquet(C / "an_indis.parquet"))


if __name__ == "__main__":
    mode = sys.argv[1]
    pops, parts = {}, {}
    if mode == "faers":
        rep, dr, rc, ind = load_faers()
        print("faers loaded", len(rep), flush=True)
        roles = ["PS", "SS", "I"]
        subsets = {"faers": rep, "faersnoca": rep[rep["country"] != "CA"],
                   # temporal split: signals from reports received <= 2014, checked in reports >= 2015
                   "early": rep[rep["year"] <= 2014], "late": rep[rep["year"] >= 2015]}
        for name, sub in subsets.items():
            parts[name], p = analyse(name, sub, dr, rc, ind, roles, "occp_cod")
            pops.update(p)
        save(parts, pops, "faers")
    elif mode == "faers_posthoc":
        rep, dr, rc, ind = load_faers()
        noca = rep[rep["country"] != "CA"]
        for v, kw in VARIANTS.items():
            parts[f"faersnoca{v}"], p = analyse(f"faersnoca{v}", noca, dr, rc, ind, ["PS", "SS", "I"], "occp_cod", **kw)
            pops.update(p)
        save(parts, pops, "faers_posthoc")
    elif mode == "canada":
        crep, cdr, crc, cind = load_canada()
        parts["canada"], p = analyse("canada", crep, cdr, crc, cind, ["S"])
        pops.update(p)
        # reports sent straight to Health Canada (not via manufacturers), so they cannot reach FAERS
        parts["canadadirect"], p = analyse("canadadirect", crep[crep["source"] != "MAH"], cdr, crc, cind, ["S"])
        pops.update(p)
        save(parts, pops, "canada")
    elif mode == "canada_posthoc":
        # post-hoc: drop solicited ("Study") reports, which carry patient-support-programme reporting
        crep, cdr, crc, cind = load_canada()
        ns = crep[crep["report_type"] != "Study"]
        parts["canadanostudy"], p = analyse("canadanostudy", ns, cdr, crc, cind, ["S"])
        pops.update(p)
        for v, kw in VARIANTS.items():
            parts[f"canada{v}"], p = analyse(f"canada{v}", ns, cdr, crc, cind, ["S"], **kw)
            pops.update(p)
        parts["canada2004x"], p = analyse("canada2004x", ns[ns["year"] >= 2004], cdr, crc, cind, ["S"],
                                          **VARIANTS["x"])
        pops.update(p)
        save(parts, pops, "canada_posthoc")
    elif mode == "join":
        join()
