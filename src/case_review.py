"""Case-level review of robust unlabelled signals (post-hoc, strict definition).

Rule-based flags are computed from the FAERS case profiles (results/case_profiles.csv,
produced by scratch script caseprofile.py on the deduplicated tables); the reviewer note is a
short clinical reading of each profile. Category precedence: first matching rule wins.

  A  report-cluster artefact      >= 50 % of cases are reports listing >= 15 drugs
  B  treatment-context confounding >= 50 % of cases co-report an established ototoxic drug
  C  stimulated / mass-exposure    vaccines and drugs with litigation-era reporting
  D  indication or comorbidity     the indication itself is linked to the event
  E  candidate                     none of the above
"""
import pandas as pd

VACCINE_OR_STIMULATED = {"TOZINAMERAN", "ELASOMERAN", "ROFECOXIB", "FINASTERIDE"}
INDICATION_LINKED = {
    "POLYMYXIN B": "otic use: top indications ear infection / otitis externa",
    "GALCANEZUMAB": "migraine, itself associated with tinnitus; 3 cases",
    "OLMESARTAN": "hypertension, associated with tinnitus",
    "VALSARTAN": "hypertension / heart failure; also US patient-programme reporting",
    "CHLORTHALIDONE": "hypertension; 18 cases",
    "LIOTHYRONINE": "thyroid disease / over-replacement",
    "LEVOTHYROXINE": "thyroid disease; 47 % of cases in drug-dense reports",
    "MINERALS": "non-specific supplement category; elderly multi-morbid patients",
    "CHOLINE": "4 cases, all in one 2013 HIV-treatment series",
}
NOTES = {
    "PREGABALIN": "1,795 cases, 61 % consumer, few co-reported ototoxics; alpha-2-delta ligand; partial label wording (ambiguous)",
    "GLYCOPYRRONIUM": "inhaled LAMA in COPD; 32 % of cases in 2025",
    "TRAVOPROST": "topical prostaglandin analogue; elderly glaucoma patients, 76 % consumer",
    "AMLODIPINE": "tinnitus labelled; hearing loss not; elderly hypertensive patients",
    "INTERFERON ALFA-2B": "hearing loss with interferon reported in the literature",
    "ETANERCEPT": "weak (adjusted OR 1.14) despite 2,001 cases",
    "TOCILIZUMAB": "weak; 45 % Canadian, 37 % drug-dense reports",
    "ONDANSETRON": "supportive-care drug; median 28 reactions per report",
    "CLONAZEPAM": "tinnitus on benzodiazepine withdrawal is described in the literature",
    "PENICILLIN G PROCAINE": "procaine (Hoigne) reactions include auditory symptoms; 5 cases",
    "PAROMOMYCIN": "aminoglycoside (oral); class effect plausible; 4 cases",
}


def category(r):
    mega = float(r["mega_ge15_drugs"].rstrip("%")) / 100
    oto = float(r["with_known_ototoxic"].rstrip("%")) / 100
    if mega >= 0.5:
        return "A"
    if oto >= 0.5:
        return "B"
    if r["drug"] in VACCINE_OR_STIMULATED:
        return "C"
    if r["drug"] in INDICATION_LINKED:
        return "D"
    return "E"


if __name__ == "__main__":
    prof = pd.read_csv("results/case_profiles.csv")
    pub = pd.concat([pd.read_csv("results/hl_pubmed.csv").assign(event="hl"),
                     pd.read_csv("results/tin_pubmed.csv").assign(event="tin")])
    prof = prof.merge(pub[["ingredient", "event", "pubmed_n"]], left_on=["drug", "event"],
                      right_on=["ingredient", "event"], how="left").drop(columns="ingredient")
    prof["category"] = prof.apply(category, axis=1)
    prof["note"] = prof["drug"].map({**INDICATION_LINKED, **NOTES}).fillna("")
    prof.sort_values(["event", "category", "cases"], ascending=[True, True, False]).to_csv(
        "results/case_review.csv", index=False)
    print(prof.groupby(["event", "category"]).size().unstack(fill_value=0))
    print(prof[prof["category"] == "E"][["drug", "event", "cases", "consumer", "pubmed_n", "note"]].to_string())
