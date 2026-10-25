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
TIN = re.compile(r"\b(tinnitus|ringing in (the|your) ears)", re.I)
STRONG = ["boxed_warning", "warnings_and_cautions", "warnings", "precautions"]
OTHER = ["adverse_reactions", "adverse_reactions_table", "postmarketing_experience",
         "contraindications", "overdosage", "user_safety_warnings", "do_not_use", "stop_use", "ask_doctor"]


def text(rec, fields):
    return " ".join(" ".join(rec.get(f, [])) for f in fields)


def scan(path):
    rows = []
    with zipfile.ZipFile(path) as z:
        data = json.loads(z.read(z.namelist()[0]))
    for rec in data["results"]:
        of = rec.get("openfda", {})
        subs = of.get("substance_name") or []
        if len(subs) != 1:
            continue
        strong = text(rec, STRONG)
        other = text(rec, OTHER)
        rows.append({
            "substance": clean(subs[0]),
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
    df.to_parquet(OUT)
    print(len(df), "single-ingredient labels;", df["substance"].nunique(), "substances")
    print(df[["hl_strong", "hl_any", "tin_any"]].mean())
