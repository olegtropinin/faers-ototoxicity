"""Build Canada Vigilance analysis tables in the same shape as the FAERS ones.

Canada Vigilance holds reports received by Health Canada (1965-2026-06). Reports
flagged 'Duplicate' of another report present in the extract are dropped
(keeping the lower REPORT_ID of a mutually-linked pair). Ingredient names come
from the product-ingredient table and are normalised with the same rules as FAERS.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from drugnorm import clean, strip_salt

C = Path("data/work/canada")
log = {}

rep = pd.read_parquet(C / "reports.parquet")
log["reports_raw"] = len(rep)
links = pd.read_parquet(C / "links.parquet")
dup = links[links["RECORD_TYPE_ENG"] == "Duplicate"].copy()
no_to_id = dict(zip(rep["REPORT_NO"], rep["REPORT_ID"]))
dup["other"] = dup["REPORT_LINK_NO"].map(no_to_id)
dup = dup.dropna(subset=["other"])
dup["rid"] = dup["REPORT_ID"].astype("int64")
dup["other"] = dup["other"].astype("int64")
pairs = set(zip(dup["rid"], dup["other"]))
drop = {a for a, b in pairs if not ((b, a) in pairs and a < b)}
rep = rep[~rep["REPORT_ID"].astype("int64").isin(drop)]
log["duplicates_dropped"] = len(drop)
log["reports_after_dedup"] = len(rep)

rep["rid"] = rep["REPORT_ID"].astype("int64")
rep["year"] = pd.to_numeric(rep["DATINTRECEIVED"].str[-2:], errors="coerce")
rep["year"] = np.where(rep["year"] > 30, 1900 + rep["year"], 2000 + rep["year"])
rep["sex"] = rep["GENDER_ENG"].map({"Male": "M", "Female": "F"}).fillna("")
rep["age_y"] = pd.to_numeric(rep["AGE_Y"], errors="coerce")
rep = rep[["rid", "year", "sex", "age_y", "REPORT_TYPE_ENG", "SOURCE_ENG"]].rename(
    columns={"REPORT_TYPE_ENG": "report_type", "SOURCE_ENG": "source"})
keep = set(rep["rid"])

# ingredient vocabulary shared with FAERS so names line up
faers_vocab = set(pd.read_csv("data/work/faers/ing_vocab.csv")["name"].dropna())
ing = pd.read_parquet(C / "ingredients.parquet")
ing["ing"] = ing["ACTIVE_INGREDIENT_NAME"].fillna("").map(clean)
uniq = {n: (n if n in faers_vocab else strip_salt(n, faers_vocab)) for n in ing["ing"].unique()}
ing["ing"] = ing["ing"].map(uniq)
ing = ing[ing["ing"] != ""][["DRUG_PRODUCT_ID", "ing"]].drop_duplicates()

drug = pd.read_parquet(C / "report_drug.parquet")
drug["rid"] = pd.to_numeric(drug["REPORT_ID"], errors="coerce")
drug = drug[drug["rid"].isin(keep)]
drug["role"] = drug["DRUGINVOLV_ENG"].map({"Suspect": "S", "Interacting": "S", "Concomitant": "C"}).fillna("O")
drugs = drug.merge(ing, on="DRUG_PRODUCT_ID", how="inner")[["rid", "ing", "role"]].drop_duplicates()
log["drug_rows_without_ingredient"] = int((~drug["DRUG_PRODUCT_ID"].isin(ing["DRUG_PRODUCT_ID"])).sum())

reac = pd.read_parquet(C / "reactions.parquet")
reac["rid"] = pd.to_numeric(reac["REPORT_ID"], errors="coerce")
reacs = reac[reac["rid"].isin(keep)].assign(pt=lambda d: d["PT_NAME_ENG"].str.upper())[["rid", "pt"]].drop_duplicates()

ind = pd.read_parquet(C / "indications.parquet")
ind["rid"] = pd.to_numeric(ind["REPORT_ID"], errors="coerce")
indis = ind[ind["rid"].isin(keep)].assign(indi_pt=lambda d: d["INDICATION_NAME_ENG"].str.upper())[["rid", "indi_pt"]]

rep = rep[rep["rid"].isin(set(drugs["rid"])) & rep["rid"].isin(set(reacs["rid"]))]
log["final_reports"] = len(rep)
rep.to_parquet(C / "an_reports.parquet")
drugs.to_parquet(C / "an_drugs.parquet")
reacs.to_parquet(C / "an_reacs.parquet")
indis.drop_duplicates().to_parquet(C / "an_indis.parquet")
json.dump(log, open(C / "build_log.json", "w"), indent=1)
print(json.dumps(log, indent=1))
