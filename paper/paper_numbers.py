"""Load every number the manuscript quotes from the result files, with half-up rounding helpers.

Used by paper/fill.py so that no figure in the text is typed by hand.
"""
import json
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"


def r(x, nd=0):
    """Round half up (Python's round() is half-to-even)."""
    q = Decimal(1).scaleb(-nd)
    return Decimal(str(float(x))).quantize(q, rounding=ROUND_HALF_UP)


def pct(x):
    return f"{r(100 * x)}\\%"


def frac(d):
    return f"{d['k']}/{d['n']} ({pct(d['rate'])})"


def thou(n):
    return f"{int(n):,}".replace(",", "{,}")


def ci(o, lo, hi, nd=2):
    return f"{r(o, nd)} ({r(lo, nd)}--{r(hi, nd)})"


def load():
    S = {ev: json.load(open(RES / f"summary_{ev}.json")) for ev in ("hl", "tin")}
    pop = json.load(open(RES / "population.json"))
    fb = json.load(open(ROOT / "data/work/faers/build_log.json"))
    cb = json.load(open(ROOT / "data/work/canada/build_log.json"))
    M = {ev: pd.read_csv(RES / f"{ev}_master.csv", index_col="ingredient", low_memory=False) for ev in ("hl", "tin")}
    return S, pop, fb, cb, M
