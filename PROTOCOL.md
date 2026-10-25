# Analysis protocol (fixed before any result was inspected)

Written 2026-10-26 ~02:05 (UTC+5), before the deduplicated FAERS tables were built.
Committed to git before the first run of `src/analyze.py`. Any deviation is listed
at the end under "Deviations", with the reason.

## Data
- FAERS / legacy AERS quarterly ASCII files 2004Q1–2026Q2 (90 quarters).
- Canada Vigilance Adverse Reaction Online Database extract (data to 2026-06-30).
- openFDA bulk drug-label export (2026-10-23).

## Pre-processing
- FAERS: latest version per case (case id; FDA date, then FAERS over AERS, then version);
  cases on FDA deletion lists removed; then content duplicates (same suspect-ingredient set,
  reaction set, sex, age, event date, country; only when event date and age are present)
  collapsed.
- Ingredients: `prod_ai` split on separators; salts/hydrates removed when the base is itself
  an observed ingredient; free-text names mapped to the dominant `prod_ai` (≥ 70 % of ≥ 5 rows).
- Canada: reports flagged "Duplicate" of a report present in the extract are dropped.

## Events
- Primary: hearing loss = MedDRA PTs Deafness, Deafness unilateral/bilateral/neurosensory/
  transitory/permanent/mixed, Hypoacusis, Neurosensory hypoacusis, Sudden hearing loss,
  Ototoxicity, Hearing impaired, Hearing disability, High frequency hearing loss.
  Conductive and congenital deafness excluded.
- Secondary: Tinnitus.

## Exposure
- Ingredient reported as suspect (FAERS PS/SS/I; Canada Suspect/Interacting).

## Statistics
- Crude: ROR (Woolf 95 % CI) and IC (BCPNN approximation with IC025).
- Adjusted: Mantel–Haenszel OR over 60 strata (5 age bands × 3 sex × 4 periods) in a
  population excluding (1) reports with an ear/hearing indication for any drug,
  (2) reports co-reporting an a-priori established ototoxic drug in any role (other than
  the drug studied), (3) FAERS reports from lawyers.
- Crude signal: a ≥ 3, ROR lower 95 % bound > 1 and IC025 > 0.
- Adjusted signal: adjusted a ≥ 3 and MH lower 95 % bound > 1.

## Validation
1. Reference set (US labels, single-ingredient SPLs): positive = hearing-loss term in any
   safety section; negative = labelled ingredient with no hearing-loss and no tinnitus term
   anywhere. Evaluated among ingredients with ≥ 1,000 suspect FAERS reports.
   Metrics: AUC of crude IC025 vs adjusted log MH lower bound; sensitivity/specificity
   of the crude and adjusted signal rules.
2. External replication: FAERS without Canadian reports vs Canada Vigilance.
   Testable if the ingredient has ≥ 3 expected-or-observed hearing-loss reports in Canada
   (max(a, E) ≥ 3). Replicated if Canada crude ROR lower bound > 1 with a ≥ 3.
   Compared: replication rate of adjusted signals, crude-only signals, and label-negative
   ingredients (the last is the false-replication baseline).
3. Temporal replication: signals from reports received ≤ 2014, checked in reports
   received ≥ 2015 with the same rule as (2).

## Outputs
- Robust unlabelled signal = adjusted FAERS signal + replicated in Canada, and not
  hearing-labelled in the US. These go to a literature/novelty check (PubMed, prior FAERS
  screens) and case-level review. They are hypotheses, not causal findings.
- Prior screens (Li 2024, Wu 2025, Liu 2025, Narasimhaswamy 2026): share of their reported
  signals that are adjusted signals and that replicate.

## Deviations
1. (02:12, before any FAERS result existed) Reference positives tightened from "any label
   mentions hearing loss" to "hearing-loss term in >= 50 % of the ingredient's prescription
   labels"; ingredients with partial mention are "ambiguous" and left out of the AUC.
   Reason: label spot-checks showed non-ADR matches (metformin: the disease name "maternally
   inherited diabetes and deafness"; TTY/TDD lines for deaf users) and single outlier
   labels among hundreds of generics. Non-ADR phrases are now removed before matching.
2. (02:20, before any FAERS result existed) Added a sensitivity analysis restricting Canada
   Vigilance to reports submitted directly to Health Canada (source not "MAH"). Reason: the
   public documentation does not state explicitly that the extract is domestic-only, and
   manufacturer-submitted reports are the route by which the same case could also reach FAERS.
3. (03:25, AFTER the primary results were inspected — post-hoc, reported as such)
   a. Canada replication repeated without solicited ("Study") reports. Reason: label-negative
      biologics replicated in Canada through patient-support-programme reports (e.g. 93 % of
      ocrelizumab and mepolizumab hearing-loss cases were "Study" reports vs ~1 % for
      gentamicin and cisplatin).
   b. Indication exclusion extended to diseases that themselves cause hearing loss (Fabry,
      mucopolysaccharidoses, cryopyrin-associated syndromes, Alport, meningitis, mitochondrial
      disease, Usher, Wolfram, CMV, NF2, Paget, osteogenesis imperfecta, hypophosphatasia,
      Cogan, Susac, mumps, measles, syphilis). Reason: enzyme-replacement therapies and
      canakinumab appeared among robust unlabelled signals.
   The primary, pre-specified results are reported unchanged; these are sensitivity analyses.
