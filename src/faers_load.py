"""Convert each FAERS / legacy AERS quarterly ASCII zip into normalised parquet tables.

Legacy AERS (2004Q1-2012Q3) keys reports by ISR/CASE; FAERS (2012Q4+) by
primaryid/caseid/caseversion. Both are mapped to one schema:

  demo:  id, caseid, ver, fda_dt, event_dt, sex, age_y, country, rept_cod, occp_cod, src, q
  drug:  id, drug_seq, role, drugname, prod_ai
  reac:  id, pt
  indi:  id, drug_seq, indi_pt
  dele:  caseid   (cases FDA lists as deleted in that quarter)
"""
import csv
import os
import io
import re
import sys
import zipfile
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

RAW = Path("data/raw/faers")
OUT = Path("data/work/faers/q")
OUT.mkdir(parents=True, exist_ok=True)

AGE_FACTOR = {"YR": 1, "Y": 1, "MON": 1 / 12, "WK": 1 / 52, "DY": 1 / 365, "HR": 1 / 8760, "DEC": 10}
LEGACY_COUNTRY = {"UNITED STATES": "US", "CANADA": "CA", "UNITED KINGDOM": "GB", "JAPAN": "JP",
                  "FRANCE": "FR", "GERMANY": "DE", "ITALY": "IT", "SPAIN": "ES"}


def read_member(z, pattern):
    names = [n for n in z.namelist() if re.search(pattern, n, re.I)]
    if not names:
        return None
    frames = []
    for n in names:
        raw = z.read(n).decode("latin-1")
        df = pd.read_csv(io.StringIO(raw), sep="$", dtype=str, quoting=csv.QUOTE_NONE,
                         index_col=False, on_bad_lines="skip", keep_default_na=False)
        df.columns = [re.sub(r"^[^a-z]+", "", c.strip().lower()) for c in df.columns]  # BOM read as latin-1
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def age_years(age, unit):
    a = pd.to_numeric(age, errors="coerce")
    f = unit.str.upper().map(AGE_FACTOR)
    y = a * f
    return y.where((y >= 0) & (y <= 120))


def convert(zpath: Path):
    q = re.search(r"(\d{4}q\d)", zpath.name, re.I).group(1).lower()
    if (OUT / f"{q}.demo.parquet").exists():
        return q, "cached"
    legacy = zpath.name.startswith("aers")
    with zipfile.ZipFile(zpath) as z:
        demo = read_member(z, r"/DEMO\d\dQ\d[^/]*\.txt$")
        drug = read_member(z, r"/DRUG\d\dQ\d[^/]*\.txt$")
        reac = read_member(z, r"/REAC\d\dQ\d[^/]*\.txt$")
        indi = read_member(z, r"/INDI\d\dQ\d[^/]*\.txt$")
        dele_names = [n for n in z.namelist() if re.search(r"DELE", n, re.I)]
        dele = []
        for n in dele_names:
            dele += [l.strip() for l in z.read(n).decode("latin-1").splitlines() if l.strip().isdigit()]

    # Column sets drift between quarters (e.g. early AERS has no reporter_country).
    for c in ("reporter_country", "occr_country", "event_dt", "occp_cod", "rept_cod", "age", "age_cod",
              "gndr_cod", "sex"):
        if c not in demo:
            demo[c] = ""

    if legacy:
        d = pd.DataFrame({
            "id": demo["isr"], "caseid": demo["case"], "ver": demo["isr"].astype("int64"),
            "fda_dt": demo["fda_dt"], "event_dt": demo["event_dt"], "sex": demo["gndr_cod"],
            "age_y": age_years(demo["age"], demo["age_cod"]),
            "country": demo["reporter_country"].str.upper().map(LEGACY_COUNTRY).fillna(demo["reporter_country"].str.upper()),
            "rept_cod": demo["rept_cod"], "occp_cod": demo["occp_cod"]})
        dr = pd.DataFrame({"id": drug["isr"], "drug_seq": drug["drug_seq"], "role": drug["role_cod"],
                           "drugname": drug["drugname"], "prod_ai": ""})
        rc = pd.DataFrame({"id": reac["isr"], "pt": reac["pt"]})
        ind = pd.DataFrame({"id": indi["isr"], "drug_seq": indi["drug_seq"], "indi_pt": indi["indi_pt"]})
    else:
        sex = demo["sex"].where(demo["sex"] != "", demo["gndr_cod"])
        country = demo["reporter_country"].copy()
        if "occr_country" in demo:
            country = demo["occr_country"].where(demo["occr_country"].str.strip() != "", country)
        d = pd.DataFrame({
            "id": demo["primaryid"], "caseid": demo["caseid"],
            "ver": pd.to_numeric(demo["caseversion"], errors="coerce").fillna(0).astype("int64"),
            "fda_dt": demo["fda_dt"], "event_dt": demo["event_dt"], "sex": sex,
            "age_y": age_years(demo["age"], demo["age_cod"]), "country": country.str.upper(),
            "rept_cod": demo["rept_cod"], "occp_cod": demo["occp_cod"]})
        dr = pd.DataFrame({"id": drug["primaryid"], "drug_seq": drug["drug_seq"], "role": drug["role_cod"],
                           "drugname": drug["drugname"],
                           "prod_ai": drug["prod_ai"] if "prod_ai" in drug else ""})
        rc = pd.DataFrame({"id": reac["primaryid"], "pt": reac["pt"]})
        ind = pd.DataFrame({"id": indi["primaryid"], "drug_seq": indi["indi_drug_seq"], "indi_pt": indi["indi_pt"]})

    d["src"] = "aers" if legacy else "faers"
    d["q"] = q
    for df in (dr, rc, ind):
        for c in df.columns:
            if df[c].dtype == object:
                df[c] = df[c].str.strip().str.upper()
    d.to_parquet(OUT / f"{q}.demo.parquet")
    dr.to_parquet(OUT / f"{q}.drug.parquet")
    rc.to_parquet(OUT / f"{q}.reac.parquet")
    ind.to_parquet(OUT / f"{q}.indi.parquet")
    pd.DataFrame({"caseid": dele}).to_parquet(OUT / f"{q}.dele.parquet")
    return q, f"demo={len(d)} drug={len(dr)} reac={len(rc)} dele={len(dele)}"


def safe_convert(zpath):
    try:
        return convert(zpath)
    except Exception as e:  # report and continue; a failed quarter is listed, never silently dropped
        return zpath.name, f"FAILED {type(e).__name__}: {e}"


if __name__ == "__main__":
    zips = sorted(RAW.glob("*.zip")) if len(sys.argv) == 1 else [Path(p) for p in sys.argv[1:]]
    with Pool(int(os.environ.get("NPROC", "8"))) as pool:
        for q, msg in pool.imap_unordered(safe_convert, zips):
            print(q, msg, flush=True)
