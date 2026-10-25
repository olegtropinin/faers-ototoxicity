"""Build the deduplicated, ingredient-level FAERS analysis tables.

Output (data/work/faers/):
  reports.parquet  rid, caseid, src, fda_dt, year, sex, age_y, country, occp_cod
  drugs.parquet    rid, ing (int code), role (PS/SS/C/I)
  reacs.parquet    rid, pt (int code)
  indis.parquet    rid, indi_pt (int code)
  ing_vocab.csv / pt_vocab.csv   code -> name
  build_log.json   counts at every step (participant flow for the paper)
"""
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from drugnorm import build_base_map, canonical, clean, split_prod_ai

Q = Path(os.environ.get("FAERS_Q", "data/work/faers/q"))
OUT = Path(os.environ.get("FAERS_OUT", "data/work/faers"))
log = {}

quarters = sorted({p.name.split(".")[0] for p in Q.glob("*.demo.parquet")})
log["quarters"] = len(quarters)
print("quarters", len(quarters), flush=True)

# ---------- 1. reports: latest version per case, deleted cases removed ----------
demo = pd.concat([pd.read_parquet(Q / f"{q}.demo.parquet") for q in quarters], ignore_index=True)
log["demo_rows"] = len(demo)
demo["rid_src"] = np.where(demo["src"] == "faers", 1, 0)
demo["id"] = pd.to_numeric(demo["id"], errors="coerce")
demo["caseid"] = pd.to_numeric(demo["caseid"], errors="coerce")
demo = demo.dropna(subset=["id", "caseid"])
demo["key"] = demo["id"].astype("int64") + demo["rid_src"].astype("int64") * 10**12
demo["fda_dt_n"] = pd.to_numeric(demo["fda_dt"].str[:8], errors="coerce").fillna(0)
demo = demo.sort_values(["caseid", "fda_dt_n", "rid_src", "ver", "key"])
latest = demo.drop_duplicates("caseid", keep="last")
log["cases_after_latest_version"] = len(latest)

deleted = set()
for q in quarters:
    p = Q / f"{q}.dele.parquet"
    if p.exists():
        deleted |= set(pd.to_numeric(pd.read_parquet(p)["caseid"], errors="coerce").dropna().astype("int64"))
latest = latest[~latest["caseid"].astype("int64").isin(deleted)]
log["deleted_case_ids"] = len(deleted)
log["cases_after_deletions"] = len(latest)
keep_keys = set(latest["key"].tolist())
print("cases", len(latest), flush=True)


keep_index = pd.Index(np.sort(np.fromiter(keep_keys, dtype=np.int64)))  # hash table built once


def keyed(df, src):
    k = pd.to_numeric(df["id"], errors="coerce")
    df = df[k.notna()].copy()
    df["key"] = k[k.notna()].astype("int64") + (10**12 if src == "faers" else 0)
    return df[keep_index.get_indexer(df["key"].values) != -1]


# ---------- 2. ingredient vocabulary ----------
# pass A: ingredient strings from prod_ai, and drugname -> prod_ai co-occurrence
ai_counts = Counter()
name_to_ai = {}
for q in quarters:
    dr = pd.read_parquet(Q / f"{q}.drug.parquet", columns=["drugname", "prod_ai"])
    dr = dr[dr["prod_ai"] != ""]
    if dr.empty:
        continue
    for ai, n in dr["prod_ai"].value_counts().items():
        for part in split_prod_ai(ai):
            ai_counts[part] += n
    pairs = dr.groupby([dr["drugname"].map(clean), dr["prod_ai"]]).size()
    for (nm, ai), n in pairs.items():
        name_to_ai.setdefault(nm, Counter())[ai] += n
base = build_base_map(ai_counts)
vocab_set = set(base.values())


def ai_to_ings(ai: str):
    return tuple(sorted({base.get(p, p) for p in split_prod_ai(ai)}))


# drugname -> ingredient tuple: the dominant prod_ai (>= 70 % of >= 5 rows), or the name itself if it is an ingredient
name_map = {}
for nm, cnt in name_to_ai.items():
    tot = sum(cnt.values())
    ai, n = cnt.most_common(1)[0]
    if tot >= 5 and n / tot >= 0.7:
        name_map[nm] = ai_to_ings(ai)


def name_to_ings(nm: str):
    if nm in name_map:
        return name_map[nm]
    if nm in vocab_set:
        return (nm,)
    b = base.get(nm)
    if b is None:
        from drugnorm import strip_salt
        b = strip_salt(nm, vocab_set)
    return (b,) if b in vocab_set else ()


ing_code = {}


def code_of(name):
    c = ing_code.get(name)
    if c is None:
        c = ing_code[name] = len(ing_code)
    return c


# ---------- 3. drugs / reactions / indications per quarter ----------
drug_parts, reac_parts, indi_parts = [], [], []
pt_code = {}
mapped_rows = unmapped_rows = 0
unmapped_names = Counter()
for q in quarters:
    src = "aers" if q < "2012q4" else "faers"
    dr = keyed(pd.read_parquet(Q / f"{q}.drug.parquet"), src)
    uniq = dr[["drugname", "prod_ai"]].drop_duplicates().reset_index(drop=True)
    pair_rows, ing_rows = [], []
    for i, (nm, ai) in enumerate(uniq.itertuples(index=False)):
        t = ai_to_ings(ai) if ai else ()
        if not t:
            t = name_to_ings(clean(nm))
        for x in t:
            x = canonical(x)
            if not x:
                continue
            pair_rows.append(i)
            ing_rows.append(code_of(x))
    uniq["pair"] = np.arange(len(uniq))
    expl = pd.DataFrame({"pair": np.asarray(pair_rows, dtype=np.int64), "ing": np.asarray(ing_rows, dtype=np.int32)})
    dr = dr.merge(uniq, on=["drugname", "prod_ai"], how="left")
    unm = ~dr["pair"].isin(expl["pair"].unique())
    mapped_rows += int((~unm).sum())
    unmapped_rows += int(unm.sum())
    unmapped_names.update(dr.loc[unm, "drugname"].value_counts().head(500).to_dict())
    m = dr[["key", "role", "pair"]].merge(expl, on="pair", how="inner")
    drug_parts.append(pd.DataFrame({"key": m["key"].values, "ing": m["ing"].values, "role": m["role"].values}))

    rc = keyed(pd.read_parquet(Q / f"{q}.reac.parquet"), src)
    for v in rc["pt"].unique():
        pt_code.setdefault(v, len(pt_code))
    rc["pt"] = rc["pt"].map(pt_code).astype(np.int32)
    reac_parts.append(rc[["key", "pt"]].drop_duplicates())

    ind = keyed(pd.read_parquet(Q / f"{q}.indi.parquet"), src)
    for v in ind["indi_pt"].unique():
        pt_code.setdefault(v, len(pt_code))
    ind["indi_pt"] = ind["indi_pt"].map(pt_code).astype(np.int32)
    indi_parts.append(ind[["key", "indi_pt"]].drop_duplicates())
    print(q, len(dr), flush=True)

log["drug_rows_mapped"] = mapped_rows
log["drug_rows_unmapped"] = unmapped_rows
drugs = pd.concat(drug_parts, ignore_index=True)
reacs = pd.concat(reac_parts, ignore_index=True)
indis = pd.concat(indi_parts, ignore_index=True)

# ---------- 4. compact report index ----------
latest = latest[latest["key"].isin(set(reacs["key"].unique()))]       # a report needs >= 1 reaction
latest = latest[latest["key"].isin(set(drugs["key"].unique()))]       # and >= 1 mapped drug
log["cases_with_reaction_and_mapped_drug"] = len(latest)
# a handful of report ids appear under two case ids across quarters; keep the later one
log["report_ids_under_two_cases"] = int(latest["key"].duplicated().sum())
latest = latest.drop_duplicates("key", keep="last").reset_index(drop=True)
latest["rid"] = np.arange(len(latest), dtype=np.int32)
rid_index = pd.Index(latest["key"].values)


def to_rid(df):
    pos = rid_index.get_indexer(df["key"].values)
    df = df[pos != -1].copy()
    df["rid"] = latest["rid"].values[pos[pos != -1]]
    return df.drop(columns="key")


drugs, reacs, indis = to_rid(drugs), to_rid(reacs), to_rid(indis)

# ---------- 5. secondary deduplication on content ----------
# Same demographics + same suspect-ingredient set + same reaction set = one report.
def mix64(x):
    """splitmix64 finaliser; summing mixed codes per report gives an order-free set hash."""
    with np.errstate(over="ignore"):
        z = x.astype(np.uint64) + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return (z ^ (z >> np.uint64(31))).view(np.int64)


def set_hash(df, col):
    u = df[["rid", col]].drop_duplicates()
    return pd.Series(mix64(u[col].values), index=u["rid"].values).groupby(level=0).sum()


sus = set_hash(drugs[drugs["role"].isin(["PS", "SS"])], "ing")
rx = set_hash(reacs, "pt")
latest["sig"] = (latest["rid"].map(sus).fillna(0).astype("int64").astype(str) + "|" +
                 latest["rid"].map(rx).fillna(0).astype("int64").astype(str) + "|" +
                 latest["sex"].fillna("") + "|" + latest["age_y"].round(0).astype(str) + "|" +
                 latest["event_dt"].fillna("") + "|" + latest["country"].fillna(""))
informative = latest["event_dt"].fillna("").str.len().ge(6) & latest["age_y"].notna()
dup = latest.duplicated("sig", keep="last") & informative
log["content_duplicates_removed"] = int(dup.sum())
latest = latest[~dup]
keep = set(latest["rid"])
drugs = drugs[drugs["rid"].isin(keep)]
reacs = reacs[reacs["rid"].isin(keep)]
indis = indis[indis["rid"].isin(keep)]
log["final_reports"] = len(latest)

latest["year"] = pd.to_numeric(latest["fda_dt"].str[:4], errors="coerce")
latest[["rid", "caseid", "src", "fda_dt", "year", "sex", "age_y", "country", "occp_cod", "rept_cod"]] \
    .to_parquet(OUT / "reports.parquet")
drugs["role"] = drugs["role"].astype("category")
drugs.to_parquet(OUT / "drugs.parquet")
reacs.to_parquet(OUT / "reacs.parquet")
indis.to_parquet(OUT / "indis.parquet")
pd.Series({v: k for k, v in ing_code.items()}).rename("name").to_csv(OUT / "ing_vocab.csv", index_label="code")
pd.Series({v: k for k, v in pt_code.items()}).rename("name").to_csv(OUT / "pt_vocab.csv", index_label="code")
pd.Series(dict(unmapped_names.most_common(300))).to_csv(OUT / "unmapped_top.csv")
json.dump(log, open(OUT / "build_log.json", "w"), indent=1)
print(json.dumps(log, indent=1))
