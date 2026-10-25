"""Load the Canada Vigilance extract into compact parquet tables.

Files are '$'-delimited, every field quoted, no header (column order from the
Health Canada data-structure page). Only the columns the analysis needs are kept.
"""
import sys
from pathlib import Path

import pandas as pd

RAW = Path(sys.argv[1] if len(sys.argv) > 1 else "data/raw/canada/cvponline_extract_20260630")
OUT = Path("data/work/canada")
OUT.mkdir(parents=True, exist_ok=True)


def read(name, cols, keep):
    df = pd.read_csv(RAW / f"{name}.txt", sep="$", quotechar='"', header=None,
                     names=cols, usecols=keep, dtype=str, encoding="utf-8",
                     on_bad_lines="warn", low_memory=False)
    print(name, len(df), flush=True)
    return df


REPORT_COLS = ("REPORT_ID REPORT_NO VERSION_NO DATRECEIVED DATINTRECEIVED MAH_NO "
               "REPORT_TYPE_CODE REPORT_TYPE_ENG REPORT_TYPE_FR GENDER_CODE GENDER_ENG GENDER_FR "
               "AGE AGE_Y AGE_UNIT_ENG AGE_UNIT_FR OUTCOME_CODE OUTCOME_ENG OUTCOME_FR "
               "WEIGHT WEIGHT_UNIT_ENG WEIGHT_UNIT_FR HEIGHT HEIGHT_UNIT_ENG HEIGHT_UNIT_FR "
               "SERIOUSNESS_CODE SERIOUSNESS_ENG SERIOUSNESS_FR DEATH DISABILITY CONGENITAL_ANOMALY "
               "LIFE_THREATENING HOSP_REQUIRED OTHER_MEDICALLY_IMP_COND REPORTER_TYPE_ENG "
               "REPORTER_TYPE_FR SOURCE_CODE SOURCE_ENG SOURCE_FR E2B_IMP_SAFETYREPORT_ID "
               "AUTHORITY_NUMB COMPANY_NUMB").split()
reports = read("reports", REPORT_COLS,
               ["REPORT_ID", "REPORT_NO", "VERSION_NO", "DATINTRECEIVED", "REPORT_TYPE_ENG",
                "GENDER_ENG", "AGE_Y", "SERIOUSNESS_ENG", "REPORTER_TYPE_ENG", "SOURCE_ENG"])
reports.to_parquet(OUT / "reports.parquet")

reac = read("reactions", "REACTION_ID REPORT_ID DURATION DURATION_UNIT_ENG DURATION_UNIT_FR "
            "PT_NAME_ENG PT_NAME_FR SOC_NAME_ENG SOC_NAME_FR MEDDRA_VERSION".split(),
            ["REPORT_ID", "PT_NAME_ENG"])
reac.to_parquet(OUT / "reactions.parquet")

DRUG_COLS = ("REPORT_DRUG_ID REPORT_ID DRUG_PRODUCT_ID DRUGNAME DRUGINVOLV_ENG DRUGINVOLV_FR "
             "ROUTEADMIN_ENG ROUTEADMIN_FR UNIT_DOSE_QTY DOSE_UNIT_ENG DOSE_UNIT_FR FREQUENCY "
             "FREQ_TIME FREQUENCY_TIME_ENG FREQUENCY_TIME_FR FREQ_TIME_UNIT_ENG FREQ_TIME_UNIT_FR "
             "THERAPY_DURATION THERAPY_DURATION_UNIT_ENG THERAPY_DURATION_UNIT_FR "
             "DOSAGEFORM_ENG DOSAGEFORM_FR").split()
drug = read("report_drug", DRUG_COLS,
            ["REPORT_DRUG_ID", "REPORT_ID", "DRUG_PRODUCT_ID", "DRUGNAME", "DRUGINVOLV_ENG"])
drug.to_parquet(OUT / "report_drug.parquet")

ingr = read("drug_product_ingredients",
            "DRUG_PRODUCT_INGREDIENT_ID DRUG_PRODUCT_ID DRUGNAME ACTIVE_INGREDIENT_ID ACTIVE_INGREDIENT_NAME".split(),
            ["DRUG_PRODUCT_ID", "ACTIVE_INGREDIENT_NAME"])
ingr.to_parquet(OUT / "ingredients.parquet")

indi = read("report_drug_indication",
            "REPORT_DRUG_ID REPORT_ID DRUG_PRODUCT_ID DRUGNAME INDICATION_NAME_ENG INDICATION_NAME_FR".split(),
            ["REPORT_ID", "DRUG_PRODUCT_ID", "INDICATION_NAME_ENG"])
indi.to_parquet(OUT / "indications.parquet")

links = read("report_links", "REPORT_LINK_ID REPORT_ID RECORD_TYPE_ENG RECORD_TYPE_FR REPORT_LINK_NO".split(),
             ["REPORT_ID", "RECORD_TYPE_ENG", "REPORT_LINK_NO"])
links.to_parquet(OUT / "links.parquet")
