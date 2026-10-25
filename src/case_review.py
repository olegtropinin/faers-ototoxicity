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
     patient-support-programme reporting documented for the drug
  D  indication or comorbidity: the drug's main recorded indication is itself associated with the
     event (PMID given)
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
    "TOZINAMERAN": "mass vaccination; media attention to hearing effects",
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
    "GALCANEZUMAB": "migraine associated with tinnitus and hearing loss (systematic review, PMID 36459425)",
    "LIOTHYRONINE": "hypothyroidism associated with tinnitus (PMID 35233053)",
    "LEVOTHYROXINE": "hypothyroidism associated with tinnitus (PMID 35233053)",
}
NOTES = {  # context for category E, with evidence for and against
    "DOXYCYCLINE": "pulsatile tinnitus can be a symptom of doxycycline-induced intracranial hypertension, which is "
                   "labelled (PMID 38516546); infections treated (e.g. Lyme disease) can affect the ear",
    "CHLORTHALIDONE": "10 cases; hypertension and age confound",
    "PREGABALIN": "alpha-2-delta ligand; one case report of reversible loss (PMID 32297964); diabetes/neuropathy "
                  "and age (median 71) confound; labels list tinnitus and hyperacusis, not hearing loss",
    "AMLODIPINE": "labels list tinnitus; hypertension and age confound; VA cohort found lower SNHL risk with "
                  "calcium-channel blockers (PMID 34855682)",
}


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
    if d in INDICATION:
        return "D"
    return "E"


if __name__ == "__main__":
    prof = pd.read_csv("results/case_profiles.csv")
    prof["category"] = prof.apply(category, axis=1)
    prof["note"] = prof["drug"].map({**DESCRIBED, **STIMULATED, **INDICATION, **NOTES}).fillna("")
    auto = {"A": lambda r: f"{r['mega_ge15_drugs']} of cases in reports listing >= 15 drugs",
            "K": lambda r: "on the a-priori ototoxic list" + (" (aminoglycoside)" if r["drug"] == "PAROMOMYCIN" else "")}
    for i, r in prof.iterrows():
        if r["category"] in auto:
            prof.loc[i, "note"] = auto[r["category"]](r) + ("; " + r["note"] if r["note"] else "")
    prof.sort_values(["event", "category", "cases"], ascending=[True, True, False]).to_csv(
        "results/case_review.csv", index=False)
    print(prof.groupby(["event", "category"]).size().unstack(fill_value=0))
    print(prof[["drug", "event", "category", "cases", "mega_ge15_drugs", "with_known_ototoxic", "note"]].to_string())
