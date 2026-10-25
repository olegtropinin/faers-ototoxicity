"""Case-level review of robust unlabelled signals (post hoc).

Robust = adjusted FAERS signal (Canadian reports excluded, extended indication filter) replicated with an
adjusted estimate in Canada Vigilance without solicited reports; unlabelled = the event is on no US label of
any product containing the drug. Profiles: results/case_profiles.csv (FAERS without Canadian reports).

Rubric, applied in this order; the first rule that applies sets the category:
  K  established or previously described: on the a-priori ototoxic list, or a prospective / cohort
     clinical series documents the association (PMID given)
  A  report cluster: >= 50 % of FAERS cases are reports listing >= 15 drugs
  B  ototoxic co-medication: >= 50 % of FAERS cases co-report an established ototoxic drug
  C  stimulated or solicited reporting: mass vaccination, litigation-associated reporting, or
     patient-support-programme reporting documented for the drug (a high consumer share alone does not count)
  D  indication or comorbidity: the drug's main recorded indication is itself associated with the
     event (meta-analysis, systematic review or cohort; PMID given) AND, where tested, the signal does not
     persist against other drugs used for that indication (active-comparator ROR lower bound <= 1;
     results/within_indication.csv)
  E  none of the above
The knowledge-based rules (K, C, D) cite their source; A and B are computed.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from signals import KNOWN_OTOTOXIC  # noqa: E402

DESCRIBED = {  # K: prior clinical series
    "INTERFERON ALFA-2B": "prospective series: tinnitus/hearing loss in 43 % (PMID 8561688)",
    "PEGINTERFERON ALFA-2B": "interferon-associated sensorineural loss (PMID 8561688)",
    "INTERFERON ALFA-2A": "interferon-associated sensorineural loss (PMID 8561688)",
}
STIMULATED = {  # C
    "TOZINAMERAN": "mass vaccination; cohort evidence conflicting (Israel SIR 1.35, PMID 35201275; Denmark null, PMID 37288514)",
    "ELASOMERAN": "mass vaccination; media attention to hearing effects",
    "ROFECOXIB": "litigation-era reporting (2004-2006)",
    "FINASTERIDE": "post-finasteride syndrome advocacy reporting",
    "ONDANSETRON": "pregnancy-exposure litigation pattern (foetal exposure indication, drug-dense reports)",
}
INDICATION = {  # D
    "TOCILIZUMAB": "giant cell arteritis / RA, both associated with hearing loss (PMIDs 12544707, 32488430)",
    "ETANERCEPT": "rheumatoid arthritis associated with SNHL (PMID 32488430); used for autoimmune inner ear disease",
    "ADALIMUMAB": "rheumatoid arthritis associated with SNHL (PMID 32488430)",
    "TRAVOPROST": "glaucoma associated with SNHL (PMID 31935130)",
    "LATANOPROST": "glaucoma associated with SNHL (PMID 31935130)",
    "GLYCOPYRROLATE": "COPD associated with hearing loss (PMID 38828668)",
    "TIOTROPIUM": "COPD associated with hearing loss (PMID 38828668)",
    "FORMOTEROL": "COPD associated with hearing loss (PMID 38828668)",
    "POLYMYXIN B": "otic use for ear infection / otitis externa",
    "CLONAZEPAM": "used to treat tinnitus (PMID 14763236)",
    "TOFACITINIB": "rheumatoid arthritis associated with SNHL (PMID 32488430)",
    "INSULIN ASPART": "diabetes associated with hearing impairment (meta-analysis, PMID 23150692)",
    "INSULIN": "diabetes associated with hearing impairment (meta-analysis, PMID 23150692)",
    "GALCANEZUMAB": "migraine associated with tinnitus (systematic review, PMID 36459425; a meta-analysis found low evidence, PMID 36380120)",
    "AMLODIPINE": "hypertension associated with hearing loss (meta-analysis, PMID 39839866)",
    "CHLORTHALIDONE": "hypertension associated with tinnitus (meta-analysis, PMID 26881064)",
    "LIOTHYRONINE": "hypothyroidism associated with tinnitus (PMID 35233053)",
    "LEVOTHYROXINE": "hypothyroidism associated with tinnitus (PMID 35233053)",
}
NOTES = {  # context for category E, with evidence for and against
    "DOXYCYCLINE": "pulsatile tinnitus can accompany labelled drug-induced intracranial hypertension (PMID 38516546), "
                   "but only 15/220 cases co-report intracranial hypertension or papilloedema",
    "TOFACITINIB": "persists against other RA drugs; 82 % consumer reports",
    "GLYCOPYRROLATE": "persists against other COPD drugs",
    "TRAVOPROST": "persists against other glaucoma drugs; median age 80",
    "LIOTHYRONINE": "persists against other hypothyroidism drugs; 32 % drug-dense reports (UK cluster overlap)",
    "PAROMOMYCIN": "oral aminoglycoside, poorly absorbed; 3 cases",

    "PREGABALIN": "alpha-2-delta ligand; one case report of reversible loss (PMID 32297964); diabetes/neuropathy "
                  "and age (median 71) confound; labels list tinnitus and hyperacusis, not hearing loss",
    "AMLODIPINE": "labels list tinnitus; VA cohort found lower SNHL risk with calcium-channel blockers (PMID 34855682)",
}


WI = pd.read_csv("results/within_indication.csv").set_index(["drug", "event"])


def persists_within_indication(drug, event):
    """True if an active-comparator ROR within the drug's indication was computed and its lower bound exceeds 1."""
    return (drug, event) in WI.index and WI.loc[(drug, event), "lo"] > 1


def category(r):
    mega = float(r["mega_ge15_drugs"].rstrip("%")) / 100
    oto = float(r["with_known_ototoxic"].rstrip("%")) / 100
    d = r["drug"]
    if d in KNOWN_OTOTOXIC or d in DESCRIBED:
        return "K"
    if mega >= 0.5:
        return "A"
    if oto >= 0.5:
        return "B"
    if d in STIMULATED:
        return "C"
    if d in INDICATION and not persists_within_indication(d, r["event"]):
        return "D"
    return "E"


if __name__ == "__main__":
    prof = pd.read_csv("results/case_profiles.csv")
    prof["category"] = prof.apply(category, axis=1)
    # category before the active-comparator check (the step the second assessor also applied)
    prof["category_pre"] = [("D" if (c == "E" and r["drug"] in INDICATION) else c) for c, (_, r) in
                            zip(prof["category"], prof.iterrows())]
    prof["note"] = [((NOTES.get(d, "") + "; " if d in NOTES else "") + ({**DESCRIBED, **STIMULATED, **INDICATION}.get(d, ""))).strip("; ")
                    for d in prof["drug"]]
    for i, r in prof.iterrows():
        if (r["drug"], r["event"]) in WI.index:
            w = WI.loc[(r["drug"], r["event"])]
            prof.loc[i, "within_indication"] = f"{w['ror']:.2f} ({w['lo']:.2f}-{w['hi']:.2f})"
    prof["within_indication"] = prof.get("within_indication", pd.Series("", index=prof.index)).fillna("")
    auto = {"A": lambda r: f"{r['mega_ge15_drugs']} of cases in reports listing >= 15 drugs",
            "K": lambda r: "on the a-priori ototoxic list" + (" (aminoglycoside)" if r["drug"] == "PAROMOMYCIN" else "")}
    for i, r in prof.iterrows():
        if r["category"] in auto:
            prof.loc[i, "note"] = auto[r["category"]](r) + ("; " + r["note"] if r["note"] else "")
    prof.sort_values(["event", "category", "cases"], ascending=[True, True, False]).to_csv(
        "results/case_review.csv", index=False)
    print(prof.groupby(["event", "category"]).size().unstack(fill_value=0))
    print(prof[["drug", "event", "category", "cases", "mega_ge15_drugs", "with_known_ototoxic", "note"]].to_string())
