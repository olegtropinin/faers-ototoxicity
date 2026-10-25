"""Disproportionality statistics, crude and adjusted, for one event across all ingredients.

Unit of analysis: a deduplicated report. Exposure: the ingredient is a suspect
(FAERS PS/SS/I; Canada Suspect/Interacting). Comparator: all other reports in the
same population.

Crude:     ROR (Woolf 95 % CI), IC with IC025 (BCPNN approximation, Noren 2006).
Adjusted:  Mantel-Haenszel OR over age x sex x period strata (Robins-Breslow-Greenland
           variance), computed in a population that excludes
             (1) reports where any drug's indication is an ear/hearing condition,
             (2) reports co-reporting an a-priori known ototoxic drug (other than the
                 drug under study) in any role -- the "innocent bystander" filter,
             (3) FAERS reports submitted by lawyers (stimulated reporting).
"""
import re

Z = 1.959963984540054  # two-sided 95 %

import numpy as np
import pandas as pd

HL_PTS = {"DEAFNESS", "DEAFNESS UNILATERAL", "DEAFNESS BILATERAL", "DEAFNESS NEUROSENSORY",
          "DEAFNESS TRANSITORY", "DEAFNESS PERMANENT", "DEAFNESS MIXED", "HYPOACUSIS",
          "NEUROSENSORY HYPOACUSIS", "SUDDEN HEARING LOSS", "OTOTOXICITY", "HEARING IMPAIRED",
          "HEARING DISABILITY", "HIGH FREQUENCY HEARING LOSS"}
TIN_PTS = {"TINNITUS"}

EAR_INDICATION = re.compile(
    r"\b(?:DEAF\w*|HEARING|HYPOACUSIS|TINNITUS|OTOTOX\w*|MENIERE\w*|OTITIS|EAR|LABYRINTH\w*|VERTIGO|"
    r"VESTIBUL\w*|COCHLE\w*|ACOUSTIC|OTOSCLEROSIS|AUDIT\w*|OTOLOGIC\w*)\b")

# A-priori established ototoxic drugs (classes reviewed in Lanvers-Kaminsky et al. 2017).
KNOWN_OTOTOXIC = {
    "GENTAMICIN", "TOBRAMYCIN", "AMIKACIN", "STREPTOMYCIN", "NEOMYCIN", "KANAMYCIN", "NETILMICIN",
    "PLAZOMICIN", "PAROMOMYCIN", "CAPREOMYCIN", "VANCOMYCIN",
    "CISPLATIN", "CARBOPLATIN", "OXALIPLATIN",
    "FUROSEMIDE", "BUMETANIDE", "TORSEMIDE", "ETHACRYNIC ACID",
    "QUININE", "CHLOROQUINE", "HYDROXYCHLOROQUINE", "MEFLOQUINE",
    "ASPIRIN", "ERYTHROMYCIN", "AZITHROMYCIN", "CLARITHROMYCIN",
    "DEFEROXAMINE", "VINCRISTINE", "EFLORNITHINE",
}


def strata(rep: pd.DataFrame) -> np.ndarray:
    age = rep["age_y"]
    ab = np.select([age < 18, age < 45, age < 65, age >= 65], [0, 1, 2, 3], 4)
    sx = rep["sex"].map({"M": 0, "F": 1}).fillna(2).astype(int).values
    yr = rep["year"].fillna(0).values
    per = np.select([yr < 2004, yr < 2013, yr < 2019], [0, 1, 2], 3)
    return (ab * 3 + sx) * 4 + per          # 5 * 3 * 4 = 60 strata


def _crude(a, n, H, N):
    b, c = n - a, H - a
    d = N - n - c
    with np.errstate(divide="ignore", invalid="ignore"):
        ror = (a * d) / (b * c)
        se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
        lo, hi = np.exp(np.log(ror) - Z * se), np.exp(np.log(ror) + Z * se)
    E = n * H / N
    ic = np.log2((a + 0.5) / (E + 0.5))
    ic025 = ic - 3.3 * (a + 0.5) ** -0.5 - 2 * (a + 0.5) ** -1.5
    return dict(a=a, n=n, E=E, ror=ror, ror_lo=lo, ror_hi=hi, ic=ic, ic025=ic025)


def _mh(ing, rid, ev, st, n_ing, n_st, pop):
    """MH OR for every ingredient from per-(ingredient, stratum) counts."""
    sel = pop[rid]
    i, r = ing[sel], rid[sel]
    key = i.astype(np.int64) * n_st + st[r]
    a_s = np.bincount(key, weights=ev[r], minlength=n_ing * n_st).reshape(n_ing, n_st)
    n_s = np.bincount(key, minlength=n_ing * n_st).reshape(n_ing, n_st).astype(float)
    N_s = np.bincount(st[pop], minlength=n_st).astype(float)
    H_s = np.bincount(st[pop], weights=ev[pop], minlength=n_st)
    b_s, c_s = n_s - a_s, H_s - a_s
    d_s = N_s - n_s - c_s
    with np.errstate(divide="ignore", invalid="ignore"):
        Ns = np.where(N_s > 0, N_s, 1)
        R = a_s * d_s / Ns
        S = b_s * c_s / Ns
        P = (a_s + d_s) / Ns
        Qv = (b_s + c_s) / Ns
        sR, sS = R.sum(1), S.sum(1)
        orr = sR / sS
        var = ((P * R).sum(1) / (2 * sR ** 2) + (P * S + Qv * R).sum(1) / (2 * sR * sS)
               + (Qv * S).sum(1) / (2 * sS ** 2))
        se = np.sqrt(var)
        lo, hi = np.exp(np.log(orr) - Z * se), np.exp(np.log(orr) + Z * se)
    return a_s.sum(1), n_s.sum(1), orr, lo, hi


def run(rep, sus, any_drug, reac_pts, indi_pts, event_pts, lawyer=None):
    """rep: reports with rid 0..R-1; sus: (rid, ing_name) suspect pairs; any_drug: (rid, ing_name)
    any role; reac_pts / indi_pts: (rid, name). Returns one row per suspect ingredient."""
    R = len(rep)
    ev = np.zeros(R)
    ev[reac_pts.loc[reac_pts["pt"].isin(event_pts), "rid"].values] = 1
    names = pd.Index(sorted(sus["ing"].unique()))
    ing = names.get_indexer(sus["ing"])
    rid = sus["rid"].values
    n_ing = len(names)

    # crude
    allpop = np.ones(R, bool)
    a = np.bincount(ing, weights=ev[rid], minlength=n_ing)
    n = np.bincount(ing, minlength=n_ing).astype(float)
    out = pd.DataFrame(_crude(a, n, ev.sum(), R), index=names)

    # adjusted population
    ear_ind = np.zeros(R, bool)
    ear_ind[indi_pts.loc[indi_pts["indi_pt"].astype(str).str.contains(EAR_INDICATION).fillna(False).values, "rid"].values] = True
    base = ~ear_ind
    if lawyer is not None:
        base &= ~lawyer
    k = any_drug[any_drug["ing"].isin(KNOWN_OTOTOXIC)][["rid", "ing"]].drop_duplicates()
    k_count = np.bincount(k["rid"].values, minlength=R)
    st = strata(rep)
    n_st = 60

    pop = base & (k_count == 0)
    a_adj, n_adj, orr, lo, hi = _mh(ing, rid, ev, st, n_ing, n_st, pop)
    out["a_adj"], out["n_adj"], out["mh_or"], out["mh_lo"], out["mh_hi"] = a_adj, n_adj, orr, lo, hi

    # known ototoxic drugs: exclude only reports with *another* known ototoxic drug
    for kname in sorted(KNOWN_OTOTOXIC & set(names)):
        has_self = np.zeros(R, bool)
        has_self[k.loc[k["ing"] == kname, "rid"].values] = True
        popk = base & ((k_count - has_self) == 0)
        j = names.get_loc(kname)
        m = ing == j
        aa, nn, o, l, h = _mh(np.zeros(m.sum(), np.int32), rid[m], ev, st, 1, n_st, popk)
        out.loc[kname, ["a_adj", "n_adj", "mh_or", "mh_lo", "mh_hi"]] = [aa[0], nn[0], o[0], l[0], h[0]]

    out["known_ototoxic"] = out.index.isin(KNOWN_OTOTOXIC)
    out.attrs["population"] = dict(reports=R, events=int(ev.sum()), adj_reports=int(pop.sum()),
                                   adj_events=int(ev[pop].sum()), ear_indication=int(ear_ind.sum()),
                                   with_known_ototoxic=int((k_count > 0).sum()),
                                   lawyer=int(lawyer.sum()) if lawyer is not None else 0)
    return out
