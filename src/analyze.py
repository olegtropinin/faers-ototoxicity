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
from drugnorm import clean, strip_salt  # noqa: E402
from signals import HL_PTS, TIN_PTS, run  # noqa: E402

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


def analyse(name, rep, dr, rc, ind, suspect_roles, lawyer_col=None):
    rep, dr, rc, ind = reindex(rep, dr, rc, ind)
    sus = dr[dr["role"].isin(suspect_roles)][["rid", "ing"]].drop_duplicates()
    lawyer = (rep[lawyer_col] == "LW").values if lawyer_col else None
    res, pops = {}, {}
    for ev_name, pts in (("hl", HL_PTS), ("tin", TIN_PTS)):
        r = run(rep, sus, dr, rc, ind, pts, lawyer)
        res[ev_name] = r.add_prefix(f"{name}_")
        pops[f"{name}_{ev_name}"] = r.attrs["population"]
        print(name, ev_name, r.attrs["population"], flush=True)
    return res, pops


def label_table():
    lab = pd.read_parquet("data/work/labels.parquet")
    vocab = set(pd.read_csv(F / "ing_vocab.csv")["name"].dropna())
    lab["ing"] = lab["substance"].map(lambda s: s if s in vocab else strip_salt(s, vocab))
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
    return g


if __name__ == "__main__":
    pops = {}
    rep, dr, rc, ind = load_faers()
    print("faers loaded", len(rep), flush=True)
    f_all, p = analyse("faers", rep, dr, rc, ind, ["PS", "SS", "I"], "occp_cod")
    pops.update(p)
    noca = rep[rep["country"] != "CA"]
    f_noca, p = analyse("faersnoca", noca, dr, rc, ind, ["PS", "SS", "I"], "occp_cod")
    pops.update(p)
    # temporal split: signals from reports received <= 2014, checked in reports received >= 2015
    f_early, p = analyse("early", rep[rep["year"] <= 2014], dr, rc, ind, ["PS", "SS", "I"], "occp_cod")
    pops.update(p)
    f_late, p = analyse("late", rep[rep["year"] >= 2015], dr, rc, ind, ["PS", "SS", "I"], "occp_cod")
    pops.update(p)
    del rep, dr, rc, ind, noca

    crep = pd.read_parquet(C / "an_reports.parquet")
    cdr = pd.read_parquet(C / "an_drugs.parquet")
    crc = pd.read_parquet(C / "an_reacs.parquet")
    cind = pd.read_parquet(C / "an_indis.parquet")
    c_all, p = analyse("canada", crep, cdr, crc, cind, ["S"])
    pops.update(p)
    # reports sent straight to Health Canada (not via manufacturers), so they cannot reach FAERS
    direct = crep[crep["source"] != "MAH"]
    c_sp, p = analyse("canadadirect", direct, cdr, crc, cind, ["S"])
    pops.update(p)

    labels = label_table()
    for ev in ("hl", "tin"):
        m = f_all[ev].join([f_noca[ev], f_early[ev], f_late[ev], c_all[ev], c_sp[ev]], how="outer") \
            .join(labels, how="left")
        m.index.name = "ingredient"
        m.to_csv(RES / f"{ev}_master.csv")
    json.dump(pops, open(RES / "population.json", "w"), indent=1)
