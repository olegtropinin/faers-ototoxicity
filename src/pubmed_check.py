"""PubMed literature count per candidate ingredient (NCBI E-utilities, no key: <= 3 req/s).

For each ingredient: number of PubMed records whose title/abstract mention the drug
together with hearing loss (or tinnitus), and the PMIDs of up to 5 of them.
Usage: python src/pubmed_check.py candidates.txt out.csv [hl|tin]
"""
import sys
import time

import pandas as pd
import requests

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
HL = ('("hearing loss"[tiab] OR deafness[tiab] OR ototoxic*[tiab] OR hypoacusis[tiab] '
      'OR "hearing impairment"[tiab] OR "Hearing Loss"[mh] OR "Ototoxicity"[mh])')
TIN = '(tinnitus[tiab] OR "Tinnitus"[mh])'


def count(drug: str, event: str):
    term = f'("{drug.lower()}"[tiab] OR "{drug.lower()}"[nm]) AND {HL if event == "hl" else TIN}'
    for attempt in range(4):
        try:
            r = requests.get(EUTILS, params={"db": "pubmed", "term": term, "retmode": "json", "retmax": 5},
                             timeout=30)
            r.raise_for_status()
            j = r.json()["esearchresult"]
            return int(j["count"]), ";".join(j.get("idlist", []))
        except Exception:
            time.sleep(2 + 3 * attempt)
    return -1, ""


if __name__ == "__main__":
    drugs = [l.strip() for l in open(sys.argv[1]) if l.strip()]
    event = sys.argv[3] if len(sys.argv) > 3 else "hl"
    rows = []
    for d in drugs:
        n, ids = count(d, event)
        rows.append({"ingredient": d, "pubmed_n": n, "pmids": ids})
        time.sleep(0.4)
    pd.DataFrame(rows).to_csv(sys.argv[2], index=False)
    print(pd.DataFrame(rows).to_string())
