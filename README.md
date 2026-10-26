# Drug–hearing-loss signals in FAERS: adjustment and independent replication

Code, pre-registered protocol and result tables for the manuscript
*"Which drug–hearing-loss signals in FAERS survive adjustment and independent replication?
A disproportionality analysis of FAERS and Canada Vigilance"* (Oleg Tropynin, independent researcher).

Disproportionality analysis measures reporting, not risk. The signals listed here are hypotheses,
not evidence that any drug causes hearing loss or tinnitus.

## What is here

| Path | Content |
|---|---|
| `PROTOCOL.md` | Analysis protocol committed before the first result, with every later deviation dated and labelled (including the post-hoc analyses) |
| `src/faers_load.py` | FAERS / legacy AERS quarterly ASCII → per-quarter parquet |
| `src/build_faers.py` | Latest case version, FDA deletions, content deduplication, ingredient mapping |
| `src/build_canada.py`, `src/canada_load.py` | Canada Vigilance extract → analysis tables |
| `src/drugnorm.py` | Ingredient normalisation (salts, combination products) |
| `src/label_ref.py` | Reference set from openFDA structured product labels |
| `src/signals.py` | ROR, IC, Mantel–Haenszel OR in the restricted population |
| `src/analyze.py` | Runs all populations (`faers`, `canada`, `faers_posthoc`, `canada_posthoc`, `join`) |
| `src/evaluate.py` | Reference-set AUC, external and temporal replication, prior-screen comparison |
| `src/case_review.py` | Rule-based case-level review categories |
| `src/tables.py`, `src/figures.py`, `src/supplement.py`, `paper/build.py` | Manuscript tables, figures, supplement |
| `tests/test_signals.py` | Checks ROR and MH estimates against `statsmodels` |
| `results/` | Per-ingredient results (`hl_master.csv`, `tin_master.csv`), summaries, case review |
| `paper/` | Manuscript (`manuscript.pdf`) and supplement |

## Reproducing

Python 3.10, numpy 2.2.6, pandas 2.3.3, scipy 1.15.3, statsmodels 0.15.0, matplotlib 3.10.9, pyarrow 25.0.1;
LaTeX via tectonic 0.17.0.

1. Download the 90 quarterly ASCII files from the FDA FAERS page into `data/raw/faers/`, the Canada
   Vigilance data extract into `data/raw/canada/`, and the openFDA drug-label bulk files into `data/raw/labels/`.
2. `python src/faers_load.py && python src/build_faers.py` (about 25 GB RAM at peak)
3. `python src/canada_load.py && python src/build_canada.py && python src/label_ref.py`
4. `python src/analyze.py faers && python src/analyze.py canada && python src/analyze.py faers_posthoc && python src/analyze.py canada_posthoc && python src/analyze.py join`
5. `python scripts/caseprofile.py <candidates.tsv> results/case_profiles.csv`, `python scripts/within_indication.py`, `python scripts/reporter_mix.py`
6. `python src/evaluate.py && python src/case_review.py && python src/figures.py && python src/flow_figure.py && python src/supplement.py && python paper/fill.py && python paper/build.py`

Case-level and provenance tables come from scripts run against the deduplicated FAERS tables:
`scripts/caseprofile.py` (case profiles), `scripts/within_indication.py` (active-comparator check),
`scripts/reporter_mix.py` (consumer and US share per ingredient), `scripts/chars.py` (Table S4).
Run them from the repository root.

## Status

Submitted manuscript; not peer reviewed. The protocol was committed before the first result (commit `ff77c40`);
every later change is dated and labelled in `PROTOCOL.md`. Parts of the analysis and the first draft were produced
with an AI assistant under the author's direction (see the manuscript's Methods).

## Data sources and licences

- FDA Adverse Event Reporting System (FAERS) quarterly data extracts, U.S. Food and Drug Administration (public domain).
- Canada Vigilance Adverse Reaction Online Database, Health Canada. Contains information licensed under the
  Open Government Licence – Canada.
- openFDA drug-label bulk export (public domain).

Code: MIT licence. Result tables and manuscript text: CC BY 4.0.
