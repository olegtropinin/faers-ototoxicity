"""Reference set from current US drug labels (openFDA bulk SPL export).

For every single-ingredient label, record whether the safety sections mention
hearing loss (and, separately, tinnitus). Output per ingredient:
  n_labels, hl_any (any section), hl_strong (boxed warning / warnings / precautions),
  tinnitus_any, rx (any prescription label), first/last effective_time of mentioning labels.
"""
import json
import re
import sys
import zipfile
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from drugnorm import clean  # noqa: E402

LAB = Path("data/raw/labels")
OUT = Path("data/work/labels.parquet")

HL = re.compile(r"\b(hearing (loss|impairment|impaired|disorder|decrease|decreased|disturbance)|"
                r"deaf(ness)?\b|ototoxic\w*|hypoacusis|loss of hearing|decreased hearing|"
                r"sudden (sensorineural )?hearing|sensorineural hearing)", re.I)
REPEAT = re.compile(r"(?=\b((?:[A-Za-z0-9\-]+ ){0,4}[A-Za-z0-9\-]+) (\1)\b)", re.I)


def generic_from_elements(el: str):
    """Phrase written once in lower/title case and repeated in capitals ('teprotumumab TEPROTUMUMAB');
    failing that, a phrase simply repeated ('Losartan Potassium Losartan Potassium')."""
    ms = list(REPEAT.finditer(el))
    for m in ms:
        if m.group(2).isupper() and not m.group(1).isupper():
            w = m.group(1).split()
            half = len(w) // 2
            if len(w) % 2 == 0 and [x.lower() for x in w[:half]] == [x.lower() for x in w[half:]]:
                return " ".join(w[:half])          # "Mirtazapine Mirtazapine MIRTAZAPINE MIRTAZAPINE"
            return m.group(1)
    for m in ms:
        if not m.group(1).isupper() and len(m.group(1)) > 3:
            return m.group(1)
    return None
TIN = re.compile(r"\b(tinnitus|ringing in (the|your) ears)", re.I)
STRONG = ["boxed_warning", "warnings_and_cautions", "warnings", "precautions"]
OTHER = ["adverse_reactions", "adverse_reactions_table", "postmarketing_experience",
         "contraindications", "overdosage", "user_safety_warnings", "do_not_use", "stop_use", "ask_doctor",
         # added after review: withdrawal effects are described here (e.g. tinnitus with benzodiazepines)
         "drug_abuse_and_dependence", "dependence", "abuse"]
COMBO_OUT = Path("data/work/labels_combo.parquet")
OTHER0 = OTHER[:-3]   # the pre-specified section list (before "drug abuse and dependence" was added)


# Phrases that contain a hearing word without describing an adverse effect of the drug.
NOT_ADR = re.compile(r"maternally inherited diabetes and deafness|\(MIDD\)|"
                     r"(deaf|hard of hearing|hearing[- ]impaired)[^.]{0,60}\b(TTY|TDD|TTD|relay)\b|"
                     r"\b(TTY|TDD|TTD)\b[^.]{0,60}(deaf|hard of hearing|hearing)", re.I)


def text(rec, fields):
    return NOT_ADR.sub(" ", " ".join(" ".join(rec.get(f, [])) for f in fields))


def scan(path):
    rows = []
    with zipfile.ZipFile(path) as z:
        data = json.loads(z.read(z.namelist()[0]))
    for rec in data["results"]:
        of = rec.get("openfda", {})
        subs = of.get("substance_name") or []
        if not subs:
            # some BLA labels carry no substance_name; fall back to the generic name,
            # dropping the four-letter biologic suffix (e.g. TEPROTUMUMAB-TRBW)
            gen = of.get("generic_name") or []
            subs = [re.sub(r"-[A-Z]{4}$", "", g.strip().upper()) for g in gen]
            if len(subs) == 1 and re.search(r"\bAND\b|,|/", subs[0]):
                continue
        if not subs:
            # ~40 % of labels have an empty openfda block; the product-data string reads
            # "BRAND generic ACTIVE INGREDIENT ...", so the generic is the phrase repeated in capitals
            el = (rec.get("spl_product_data_elements") or [""])[0]
            g = generic_from_elements(el[:200])
            if g and not re.search(r"\band\b|,", g, re.I):
                subs = [re.sub(r"-[a-z]{4}$", "", g, flags=re.I)]
        strong = text(rec, STRONG)
        other = text(rec, OTHER)
        if len(subs) > 1:
            hl, tin = bool(HL.search(strong) or HL.search(other)), bool(TIN.search(strong) or TIN.search(other))
            for sub in subs:
                rows.append({"substance": clean(sub), "combo": True, "rx": "HUMAN PRESCRIPTION DRUG" in
                             (of.get("product_type") or []), "hl_strong": False, "hl_any": hl, "tin_any": tin,
                             "effective_time": rec.get("effective_time", ""), "set_id": rec.get("set_id", "") + sub})
            continue
        if len(subs) != 1:
            continue
        other0 = text(rec, OTHER0)
        rows.append({
            "hl_any0": bool(HL.search(strong) or HL.search(other0)),
            "tin_any0": bool(TIN.search(strong) or TIN.search(other0)),
            "substance": clean(subs[0]), "combo": False,
            "rx": "HUMAN PRESCRIPTION DRUG" in (of.get("product_type") or []),
            "hl_strong": bool(HL.search(strong)),
            "hl_any": bool(HL.search(strong) or HL.search(other)),
            "tin_any": bool(TIN.search(strong) or TIN.search(other)),
            "effective_time": rec.get("effective_time", ""),
            "set_id": rec.get("set_id", ""),
        })
    return rows


if __name__ == "__main__":
    files = sorted(LAB.glob("*.zip"))
    with Pool(min(7, len(files))) as pool:
        rows = [r for part in pool.imap_unordered(scan, files) for r in part]
    df = pd.DataFrame(rows).drop_duplicates("set_id")
    df[df["combo"]].to_parquet(COMBO_OUT)
    df = df[~df["combo"]]
    df.to_parquet(OUT)
    print(len(df), "single-ingredient labels;", df["substance"].nunique(), "substances")
    print(df[["hl_strong", "hl_any", "tin_any"]].mean())
