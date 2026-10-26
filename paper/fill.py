"""Generate the Results, Discussion, abstract, key points and tables of the manuscript from result files.

Writes paper/generated/*.tex; paper/build.py splices them into main.tex. No number is typed by hand.
"""
import json

import numpy as np
import pandas as pd

from paper_numbers import ROOT, RES, ci, frac, load, pct, r, thou

OUT = ROOT / "paper" / "generated"
OUT.mkdir(exist_ok=True)
S, POP, FB, CB, M = load()
HL, TIN = S["hl"], S["tin"]
SH = json.load(open(RES / "canada_study_shares.json"))
PR = pd.read_csv(RES / "prior_screens.csv")
RV = pd.read_csv(RES / "case_review.csv")
WI = pd.read_csv(RES / "within_indication.csv").set_index(["drug", "event"])
A2 = RES / "case_review_assessor2.csv"
m = M["hl"]
WORDS = ["no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"]


def _kappa(c1, c2):
    cats = sorted(set(c1) | set(c2))
    po = (c1 == c2).mean()
    pe = sum((c1 == c).mean() * (c2 == c).mean() for c in cats)
    return (po - pe) / (1 - pe)


def kappa():
    """Agreement before the active-comparator check (rubric as both assessors applied it), and after applying the
    same mechanical rule (D -> E when the signal persists within the indication) to both assessors."""
    b = pd.read_csv(A2)
    j = RV.merge(b, on=["drug", "event"], suffixes=("_1", "_2"))
    persists = [((d, e) in WI.index and WI.loc[(d, e), "lo"] > 1) for d, e in zip(j["drug"], j["event"])]
    c2_post = np.where((j["category_2"] == "D") & np.array(persists), "E", j["category_2"])
    return {"n": len(j), "agree_pre": int((j["category_pre"] == j["category_2"]).sum()),
            "kappa_pre": _kappa(j["category_pre"], j["category_2"]),
            "agree": int((j["category_1"] == c2_post).sum()), "kappa": _kappa(j["category_1"], pd.Series(c2_post)),
            "e2": int((c2_post == "E").sum()),
            "disagreements": j.assign(category_2_post=c2_post)[j["category_1"] != c2_post][
                ["drug", "event", "category_1", "category_2_post"]]}


K = kappa()
cat_counts = RV.groupby("category").size().to_dict()
E_list = RV[RV["category"] == "E"]
nE, n_rob = len(E_list), len(RV)
n_rob_hl, n_rob_tin = int((RV.event == "hl").sum()), int((RV.event == "tin").sum())
E_hl = [d.lower() for d in E_list[E_list.event == "hl"].drug]
E_tin = [d.lower() for d in E_list[E_list.event == "tin"].drug]

ref, ref_t = HL["reference"], TIN["reference"]
fl = HL["flow"]
rc, rc_t = HL["replication_canada"], TIN["replication_canada"]
cal, cal_t = HL["calibrated_canada"], TIN["calibrated_canada"]
px, px_t = HL["posthoc"]["x"], TIN["posthoc"]["x"]
tp, tp_t = HL["replication_temporal"], TIN["replication_temporal"]
removed_deleted = FB["cases_after_latest_version"] - FB["cases_after_deletions"]
unmapped = FB["drug_rows_unmapped"] / (FB["drug_rows_unmapped"] + FB["drug_rows_mapped"])
eto, taf, sac = m.loc["ETOPOSIDE"], m.loc["TAFAMIDIS"], m.loc["SACUBITRIL"]
known = pd.read_csv(RES / "table_known.csv", index_col=0)
li_u = PR[(PR.study == "li2024") & (PR.labelled_per_authors == "no")].iloc[0]
liu_u = PR[(PR.study == "liu2025") & (PR.labelled_per_authors == "no")].iloc[0]
n_ing = int((m["faersnoca_n"] > 0).sum())
nocapop = POP["faersnoca_hl"]
oe_range = lambda key: (r(min(HL["posthoc"][v][key]["obs_over_exp"] for v in HL["posthoc"]), 2),  # noqa: E731
                        r(max(HL["posthoc"][v][key]["obs_over_exp"] for v in HL["posthoc"]), 2))


def oe(d):
    return (f"{d['observed']} observed vs {r(d['expected_if_true'], 1)} expected; O/E {r(d['obs_over_exp'], 2)}, "
            f"95\\% CI {r(d['ci'][0], 2)}--{r(d['ci'][1], 2)}")


def kc(name):
    k = known.loc[name]
    return ci(k.faers_mh_or, k.faers_mh_lo, k.faers_mh_hi, 1)


def wi(drug, ev):
    w = WI.loc[(drug, ev)]
    return ci(w["ror"], w["lo"], w["hi"])


def join(xs):
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]


MIX = pd.read_csv(RES / "reporter_mix.csv").query("event == 'hl'").set_index("ingredient")
pre_E = RV[RV["category_pre"] == "E"]
x_strat = m.loc[["CARBOPLATIN", "VINCRISTINE", "ETOPOSIDE"], ["faersnocax_mh_or", "faersnocaxstrat_mh_or"]]
nomega = HL["posthoc"]["xnomega"]

results = rf"""
\section{{Results}}

\subsection{{Data}}
The 90 FAERS extracts held {thou(FB['demo_rows'])} report versions, from which {thou(FB['cases_after_latest_version'])}
cases remained after keeping the latest version; {thou(removed_deleted)} cases on FDA deletion lists, cases without a
reaction or mappable ingredient, {thou(FB['report_ids_under_two_cases'])} report identifiers recorded under two cases and
{thou(FB['content_duplicates_removed'])} content duplicates were then removed, leaving \textbf{{{thou(FB['final_reports'])}
reports}} ({pct(unmapped)} of drug rows could not be mapped; Supplementary Figure~S1). They contained
{thou(POP['faers_hl']['events'])} hearing-loss and {thou(POP['faers_tin']['events'])} tinnitus reports; hearing-loss reports
came more often from consumers and from patients aged $\geq$65 years than other reports (Supplementary Table~S4). Without
Canadian reports, the population used for signal detection held {thou(nocapop['reports'])} reports and {thou(n_ing)} suspect
ingredients; its adjusted subset excluded {thou(nocapop['ear_indication'])} reports with an ear or hearing indication,
{thou(nocapop['with_known_ototoxic'])} reports naming a listed ototoxic drug and {thou(nocapop['lawyer'])} lawyer reports,
leaving {thou(nocapop['adj_reports'])} reports ({thou(nocapop['adj_events'])} hearing-loss cases). Canada Vigilance
contributed {thou(CB['final_reports'])} reports after removing {thou(CB['duplicates_dropped'])} duplicates
({thou(POP['canada_hl']['events'])} hearing-loss and {thou(POP['canada_tin']['events'])} tinnitus reports).

Drugs of the established ototoxic classes gave strong signals in both databases (Supplementary Table~S2), for example
amikacin (FAERS adjusted OR {kc('AMIKACIN')}), gentamicin ({kc('GENTAMICIN')}) and cisplatin ({kc('CISPLATIN')}); so did
the PDE5 inhibitors sildenafil ({kc('SILDENAFIL')}) and tadalafil ({kc('TADALAFIL')}), whose labels carry hearing-loss
warnings, and teprotumumab ({kc('TEPROTUMUMAB')}). Some label-positive drugs did not signal: oxaliplatin (crude ROR
{r(known.loc['OXALIPLATIN','faers_ror'],2)}) and vardenafil ({int(known.loc['VARDENAFIL','faers_a'])} cases); carboplatin was a
crude signal that fell below 1 after adjustment ({r(known.loc['CARBOPLATIN','faers_ror'],2)} to
{r(known.loc['CARBOPLATIN','faers_mh_or'],2)}), as did vincristine's estimate, an example of the sensitivity that the
co-medication filter costs.

\subsection{{Adjustment changes the signal list but not its agreement with labels}}
In FAERS without Canadian reports the crude rule flagged {fl['crude_signals']} ingredients for hearing loss and the adjusted
rule {fl['adjusted_signals']}; {fl['crude_not_adjusted']} crude signals were lost and {fl['adjusted_not_crude']} appeared
(Table~\ref{{tab:methods}}). Against the reference set ({ref['n_pos']} label-positive and {ref['n_neg']} label-negative
ingredients with $\geq$1{{,}}000 suspect reports), discrimination was moderate and practically identical: crude IC$_{{025}}$
versus adjusted lower bound, as pre-specified, AUC {r(ref['auc']['crude_IC025'],3)} vs {r(ref['auc']['adjusted_logMH025'],3)};
crude versus adjusted lower bound, the like-for-like pair, {r(ref['auc']['crude_logROR025'],3)} vs
{r(ref['auc']['adjusted_logMH025'],3)} (difference {r(ref['auc_diff_observed'],3)}, 95\% CI {r(ref['auc_diff_ci'][0],3)} to
{r(ref['auc_diff_ci'][1],3)}). The same held for point estimates ({r(ref['auc']['crude_logROR'],3)} vs
{r(ref['auc']['adjusted_logMH'],3)}), after excluding the {ref['adjusted_score_at_floor']} ingredients without an estimable
adjusted bound ({r(HL['reference_sensitivity']['without_floor']['crude_logROR025'],3)} vs
{r(HL['reference_sensitivity']['without_floor']['adjusted_logMH025'],3)}), at lower report thresholds, without the 29 listed
drugs, and for tinnitus ({r(ref_t['auc']['crude_logROR025'],3)} vs {r(ref_t['auc']['adjusted_logMH025'],3)}). Adjustment lowered
sensitivity ({r(ref['crude_sens'],2)} to {r(ref['adj_sens'],2)}) with little change in specificity
({r(ref['crude_spec'],3)} to {r(ref['adj_spec'],3)}).

\subsection{{Replication in Canada Vigilance}}
Under the pre-specified rule, adjusted hearing-loss signals already on a label replicated in {frac(rc['adjusted_labelled'])}
of testable cases, unlabelled adjusted signals in {frac(rc['adjusted_unlabelled'])}, label-negative adjusted signals in
{frac(rc['adjusted_label_negative'])}, crude-only signals in {frac(rc['crude_only'])}, and label-negative ingredients that
were neither crude nor adjusted signals in {frac(rc['background'])} (all label-negative ingredients, as literally specified in
the protocol: {frac(rc['protocol_background_all_label_negative'])}; Table~\ref{{tab:methods}}). In this pre-specified
comparison unlabelled signals replicated more often than the background (odds ratio
{r(rc['test_unlabelled_vs_background']['odds_ratio'],2)}, Fisher $p<0.001$) but less often than labelled signals. The
power-calibrated comparison (post hoc) placed this contrast on a common scale (Figure~\ref{{fig:repl}}): labelled signals
(n={cal['adjusted_labelled']['n']} ingredients) replicated about as often as their shrunken FAERS estimates predicted
({oe(cal['adjusted_labelled'])}), whereas unlabelled signals (n={cal['adjusted_unlabelled']['n']}) replicated less than half
as often ({oe(cal['adjusted_unlabelled'])}); the ratio of the two was {r(HL['oe_ratio_unlabelled_vs_labelled']['ratio'],2)}
(95\% CI {r(HL['oe_ratio_unlabelled_vs_labelled']['ci'][0],2)}--{r(HL['oe_ratio_unlabelled_vs_labelled']['ci'][1],2)}). The
label-negative signals, a subset of the unlabelled group, behaved alike ({oe(cal['adjusted_label_negative'])}). Results were the
same with unshrunken estimates, with the pre-specified label definition, and for tinnitus ({oe(cal_t['adjusted_labelled'])} for
labelled and {oe(cal_t['adjusted_unlabelled'])} for unlabelled signals).

Unlabelled signals were more often reported by consumers (median share of FAERS cases
{pct(HL['consumer_share_median']['unlabelled'])} vs {pct(HL['consumer_share_median']['labelled'])} for labelled signals) and
more often from the United States ({pct(HL['us_share_median']['unlabelled'])} vs {pct(HL['us_share_median']['labelled'])}), so
provenance could explain the contrast. It did not: among signals with a consumer share at or below the median, O/E was
{r(HL['by_consumer_share']['labelled_low']['obs_over_exp'],2)} for labelled and
{r(HL['by_consumer_share']['unlabelled_low']['obs_over_exp'],2)} (95\% CI
{r(HL['by_consumer_share']['unlabelled_low']['ci'][0],2)}--{r(HL['by_consumer_share']['unlabelled_low']['ci'][1],2)}) for
unlabelled signals, and above the median {r(HL['by_consumer_share']['labelled_high']['obs_over_exp'],2)} and
{r(HL['by_consumer_share']['unlabelled_high']['obs_over_exp'],2)}. Pooled across ingredients (random effects), the Canadian ROR
was {ci(HL['pooled_canada_crude']['labelled']['or'], HL['pooled_canada_crude']['labelled']['lo'], HL['pooled_canada_crude']['labelled']['hi'])}
for labelled and {ci(HL['pooled_canada_crude']['unlabelled']['or'], HL['pooled_canada_crude']['unlabelled']['lo'], HL['pooled_canada_crude']['unlabelled']['hi'])}
for unlabelled signals: as a group, unlabelled signals were attenuated rather than absent.

The replication test was not conservative: label-negative non-signals replicated {r(cal['background']['observed'] /
cal['background']['expected_if_null'], 1)} times as often as expected without any association
({cal['background']['observed']} vs {r(cal['background']['expected_if_null'],1)}). Solicited reports explained much of this:
\emph{{Study}} reports, which include manufacturers' patient-support programmes, made up {pct(SH['background_replicated'][1])}
of Canadian hearing-loss cases for the background drugs that replicated, against {pct(SH['labelled_replicated'][1])} for
replicated labelled signals and {pct(SH['all_HL_cases'][1])} overall; for ocrelizumab, mepolizumab and omalizumab the share
was {pct(SH['OCRELIZUMAB'][1])}, {pct(SH['MEPOLIZUMAB'][1])} and {pct(SH['OMALIZUMAB'][1])}, against
{pct(SH['GENTAMICIN'][1])} for gentamicin and {pct(SH['CISPLATIN'][1])} for cisplatin. The mirror image appeared in FAERS:
tafamidis (adjusted OR {ci(taf.faersnoca_mh_or, taf.faersnoca_mh_lo, taf.faersnoca_mh_hi, 1)}) had
{thou(taf.faersnoca_a)} hearing-loss reports in FAERS ({pct(MIX.loc['TAFAMIDIS','consumer'])} from consumers,
{pct(MIX.loc['TAFAMIDIS','us'])} from the United States) but none in Canada Vigilance against {r(taf.canada_E,1)} expected, and
sacubitril ({thou(sac.faersnoca_a)} FAERS hearing-loss reports; {pct(MIX.loc['SACUBITRIL','consumer'])} from consumers) none against
{r(sac.canada_E,1)} expected (too few to be testable).

Co-medication bias can also replicate when crude estimates are compared. Etoposide, given in cisplatin-containing regimens,
was a crude signal in both databases (FAERS ROR {ci(eto.faersnoca_ror, eto.faersnoca_ror_lo, eto.faersnoca_ror_hi)};
Canada {ci(eto.canada_ror, eto.canada_ror_lo, eto.canada_ror_hi)}); after adjustment the FAERS estimate reversed
({ci(eto.faersnoca_mh_or, eto.faersnoca_mh_lo, eto.faersnoca_mh_hi)}) and the Canadian one rested on
{int(eto.canada_a_adj)} cases ({ci(eto.canada_mh_or, eto.canada_mh_lo, eto.canada_mh_hi)}). Temporal replication within FAERS
did not favour adjustment for hearing loss: early crude-only signals recurred after 2014 at least as often
({frac(tp['early_crude_only'])}) as early adjusted signals ({frac(tp['early_adjusted'])}; background {frac(tp['background'])}),
consistent with biases that persist within one database; for tinnitus adjusted signals recurred more often
({frac(tp_t['early_adjusted'])} vs {frac(tp_t['early_crude_only'])}).

\subsection{{Post-hoc analyses}}
With the extended indication filter, Canada Vigilance restricted to unsolicited reports and both estimates adjusted
(Table~\ref{{tab:posthoc}}), labelled hearing-loss signals replicated in {frac(px['labelled'])}, unlabelled signals in
{frac(px['unlabelled'])} and the background in {frac(px['background'])} (Fisher $p={r(px['test_unlabelled_vs_background']['p'],2)}$);
calibrated on adjusted Canadian counts, O/E was {r(px['calibrated_labelled']['obs_over_exp'],2)} (95\% CI
{r(px['calibrated_labelled']['ci'][0],2)}--{r(px['calibrated_labelled']['ci'][1],2)}) for labelled and
{r(px['calibrated_unlabelled']['obs_over_exp'],2)} ({r(px['calibrated_unlabelled']['ci'][0],2)}--{r(px['calibrated_unlabelled']['ci'][1],2)})
for unlabelled signals (ratio {r(px['oe_ratio_unlabelled_vs_labelled']['ratio'],2)}, 95\% CI
{r(px['oe_ratio_unlabelled_vs_labelled']['ci'][0],2)}--{r(px['oe_ratio_unlabelled_vs_labelled']['ci'][1],2)}); the pooled
Canadian adjusted OR was {ci(px['pooled_canada_adjusted']['labelled']['or'], px['pooled_canada_adjusted']['labelled']['lo'], px['pooled_canada_adjusted']['labelled']['hi'])}
for labelled, {ci(px['pooled_canada_adjusted']['unlabelled']['or'], px['pooled_canada_adjusted']['unlabelled']['lo'], px['pooled_canada_adjusted']['unlabelled']['hi'])}
for unlabelled and {ci(px['pooled_canada_adjusted']['label_negative']['or'], px['pooled_canada_adjusted']['label_negative']['lo'], px['pooled_canada_adjusted']['label_negative']['hi'])}
for label-negative signals. For tinnitus, labelled signals replicated less fully in this framework (O/E
{r(px_t['calibrated_labelled']['obs_over_exp'],2)}, 95\% CI {r(px_t['calibrated_labelled']['ci'][0],2)}--{r(px_t['calibrated_labelled']['ci'][1],2)}),
with unlabelled signals lower still ({r(px_t['calibrated_unlabelled']['obs_over_exp'],2)}). Across the co-medication variants and the restriction of Canada Vigilance to reports from 2004,
unlabelled O/E ranged {oe_range('calibrated_unlabelled')[0]}--{oe_range('calibrated_unlabelled')[1]} and labelled
{oe_range('calibrated_labelled')[0]}--{oe_range('calibrated_labelled')[1]}. The comparison with the background depended on
report clusters: after excluding reports that list $\geq$15 drugs, the background fell to {frac(nomega['background'])} and
unlabelled signals ({frac(nomega['unlabelled'])}) exceeded it (Fisher $p={r(nomega['test_unlabelled_vs_background']['p'],3)}$),
although still replicating far below expectation (O/E {r(nomega['calibrated_unlabelled']['obs_over_exp'],2)}). Handling
co-medication as a stratum rather than an exclusion removed the below-1 estimates of label-positive drugs (carboplatin
{r(x_strat.loc['CARBOPLATIN','faersnocax_mh_or'],2)} to {r(x_strat.loc['CARBOPLATIN','faersnocaxstrat_mh_or'],2)}; vincristine
{r(x_strat.loc['VINCRISTINE','faersnocax_mh_or'],2)} to {r(x_strat.loc['VINCRISTINE','faersnocaxstrat_mh_or'],2)}), consistent
with selection bias from the exclusion, but did not remove the etoposide association
({r(x_strat.loc['ETOPOSIDE','faersnocaxstrat_mh_or'],2)}), consistent with residual co-medication bias under a binary stratum.

\subsection{{Signals from previous screens}}
Of the {int(li_u.reported)} hearing-loss signals that Li et al.\ classified as unlabelled \cite{{li2024}},
{int(li_u.crude_here)} were crude and {int(li_u.adjusted_here)} adjusted signals in FAERS without Canadian reports,
{int(li_u.adjusted_testable)} were testable in Canada and {int(li_u.adjusted_replicated)} replicated; {int(li_u.posthoc_robust)}
met the post-hoc robust definition ({str(li_u.posthoc_robust_drugs).lower()}). Their strongest unlabelled signals included
sacubitril/valsartan and tafamidis. Of the {int(liu_u.reported)} unlabelled paediatric signals of Liu et al.\ \cite{{liu2025}},
{int(liu_u.adjusted_here)} were adjusted signals in our all-age data and {int(liu_u.posthoc_robust)} met the robust definition
({str(liu_u.posthoc_robust_drugs).lower()}, which is on the a-priori ototoxic list);
{join([x.lower() for x in str(liu_u.lost_on_adjustment).split(';')])} were lost on adjustment (Table~\ref{{tab:prior}}). Only
abstract-level lists were available for the other screens (Supplementary Table~S6).

\subsection{{Case-level review}}
Under the post-hoc definition, {n_rob_hl} hearing-loss and {n_rob_tin} tinnitus signals were robust and absent from every US
label (Table~\ref{{tab:cand}}, Figure~\ref{{fig:cand}}). The review attributed {cat_counts.get('A',0)} to report clusters (for
example candesartan--tinnitus: {RV.set_index(['drug','event']).loc[('CANDESARTAN','tin'),'top_country'].split()[1].replace('%', chr(92) + '%')}
of cases from the United Kingdom, median {RV.set_index(['drug','event']).loc[('CANDESARTAN','tin'),'median_drugs']:g} drugs per
report), {cat_counts.get('C',0)} to stimulated or mass-exposure reporting (mRNA COVID-19 vaccines, rofecoxib, finasteride),
{cat_counts.get('K',0)} to an established ototoxic class (oral paromomycin) and {cat_counts.get('D',0)} to an indication, with an
active-comparator estimate consistent with it (etanercept against other rheumatoid-arthritis drugs,
ROR {wi('ETANERCEPT','hl')}; insulin aspart within diabetes, {wi('INSULIN ASPART','hl')}; galcanezumab within migraine,
{wi('GALCANEZUMAB','tin')}; and chlorthalidone within hypertension, {wi('CHLORTHALIDONE','tin')}, which is indeterminate with
{int(WI.loc[('CHLORTHALIDONE','tin'),'a'])} cases). For {WORDS[nE]} signals the
indication did not account for the association, because it persisted against other drugs for the same indication:
{join(E_hl)} for hearing loss (for example pregabalin within pain and neuropathy, ROR {wi('PREGABALIN','hl')}; amlodipine
within hypertension, {wi('AMLODIPINE','hl')}) and {join(E_tin)} for tinnitus (doxycycline, {wi('DOXYCYCLINE','tin')}).
A second, AI-based assessor applying the same rubric to the same profiles agreed on {K['agree_pre']} of {K['n']} assignments
before the active-comparator step (Cohen's $\kappa={r(K['kappa_pre'],2)}$) and on {K['agree']} after the same mechanical rule
was applied to both ($\kappa={r(K['kappa'],2)}$); with its assignments, {WORDS[K['e2']]} signals would remain unexplained
(Supplementary Table~S3). Before the post-hoc active-comparator rule, only {WORDS[len(pre_E)]} ({join([d.lower() for d in pre_E.drug])})
were unexplained; the residual count therefore ranges from {len(pre_E)} to {K['e2']} depending on post-hoc choices and assessor.
"""

discussion = rf"""
\section{{Discussion}}
Three findings matter for anyone screening spontaneous reports for ototoxicity. First, adjusting for indication, ototoxic
co-medication, lawyer reports and demographics changed a substantial part of the hearing-loss signal list
({fl['crude_not_adjusted']} of {fl['crude_signals']} crude signals lost, {fl['adjusted_not_crude']} gained) but did not
improve agreement with labels. Second, replication in an independent national database behaved as a validation test should
for labelled hearing-loss associations---they replicated about as often as their strength predicted---whereas unlabelled
signals replicated less than half as often, a contrast that held within strata of consumer reporting and whose pooled
Canadian estimate (attenuated, not absent) was close to that of signals for drugs whose labels are silent on hearing. Third,
replication is not a clean validator on its own: solicited patient-support-programme reports and report clusters produced
``replications'' for drugs with no known association, and co-medication bias can replicate when crude estimates are compared.
After adjusted replication, a review of report provenance and an active-comparator check, {nE} of {n_rob} robust unlabelled
signals remained ({len(pre_E)} to {K['e2']} under alternative post-hoc choices and assessors). For tinnitus the same pattern held,
but labelled signals also replicated less fully once both databases were adjusted, so the evidence there is weaker.

These results qualify the previous all-drug screens \cite{{li2024,wu2025,liu2025}} rather than contradict them. Their
labelled signals largely held; their unlabelled signals mostly did not, and several of the strongest are consistent with how
reports were collected rather than with pharmacology. Tafamidis and sacubitril/valsartan are examples: hundreds to thousands
of FAERS hearing-loss reports, mostly from US consumers and consistent with manufacturer programmes, and none in Canada. The same mechanism
produced spurious Canadian ``replications'' for biologics. Solicited reports are known to distort disproportionality
\cite{{psp_solicited}}, and their share differs between databases, so report provenance can manufacture both a signal and its
replication.

In this study three steps did most of the work: comparing \emph{{adjusted}} estimates in both databases;
profiling each case series for concentration---solicited or consumer reports, a single country or year, very drug-dense
reports; and comparing the drug with others used for the same indication. The power-calibrated comparison is simple and avoids
the outcome-dependent testability of rules based on observed counts; it remains sensitive to overdispersion, and empirical
calibration against negative controls \cite{{schuemie2014}} or regression-based adjustment for co-reported drugs
\cite{{caster2010}} would be natural extensions. We suggest that screens for hearing outcomes report these provenance
indicators for every signal, in line with the case-by-case item of READUS-PV \cite{{readuspv,readuspv_ee}}.

The remaining candidates are hypotheses. Pregabalin, an $\alpha_2\delta$ ligand, had {thou(m.loc['PREGABALIN','faersnoca_a'])}
FAERS hearing-loss reports (adjusted OR
{ci(m.loc['PREGABALIN','faersnocax_mh_or'], m.loc['PREGABALIN','faersnocax_mh_lo'], m.loc['PREGABALIN','faersnocax_mh_hi'])})
that replicated in Canada and persisted against other drugs for pain and neuropathy; a case of reversible, dose-related hearing
loss has been described \cite{{pregabalin_case}}, but age and diabetes confound \cite{{diabetes_hearing}}, and its labels list
tinnitus and hyperacusis. For amlodipine, hypertension is associated with hearing loss \cite{{htn_hearing}}, yet the signal
persisted against other antihypertensives, while a large cohort found a \emph{{lower}} risk of sensorineural hearing loss with
calcium-channel blockers \cite{{ccb_cohort}}, which argues against a causal effect. Tofacitinib, travoprost and glycopyrrolate
persisted within rheumatoid arthritis, glaucoma and COPD, diseases themselves linked to hearing loss
\cite{{ra_hearing,glaucoma_hearing,copd_hearing}}, so residual confounding by disease severity remains possible. Doxycycline
can cause intracranial hypertension, of which pulsatile tinnitus is a symptom \cite{{doxy_iih}}, although few cases reported
it. None of these should change practice without studies that have exposure data.

\paragraph{{Limitations.}} Spontaneous reports lack exposure denominators, and disproportionality measures reporting, not risk
or incidence. The label reference set is imperfect: labels are partly informed by spontaneous reports, list rare events
without causal assessment, and route-specific labels (for example topical gentamicin) blur ingredient-level status; we used it
only to compare methods. Excluding reports with an ototoxic co-medication conditions on a variable that reporters may add
after observing the event, which pushed some label-positive drugs below 1; stratifying instead removed that artefact but left
residual co-medication bias, and both gave the same replication contrast. Indications are often missing; unreported
co-treatment (for example aminoglycosides in mycobacterial disease) remains; content deduplication was not applied to Canada
Vigilance and cannot catch near-duplicate series, which the report-cluster indicator only partly addresses; and combination
products contribute to each ingredient (valsartan inherits sacubitril/valsartan reports). Canadian reports were removed from
FAERS by occurrence country, or reporter country when missing. The Canada Vigilance documentation does not state that the
extract is domestic; restricting it to reports sent directly to Health Canada gave similar results
({frac(HL['replication_canada_direct']['adjusted'])} of adjusted signals replicated, background
{frac(HL['replication_canada_direct']['background'])}). About {thou(round(n_ing, -2))} ingredients were screened without
multiplicity correction; the replication and calibration analyses, rather than individual $p$-values, carry the conclusions,
and the calibration's expected counts depend on Poisson assumptions that the background shows to be optimistic. Corrections
made after the first results (ingredient synonyms, comparator handling for listed drugs, label sections) changed the primary
numbers slightly; they are listed in the protocol file. The solicited-report exclusion, the extended indication filter,
adjusted-to-adjusted and calibrated replication, the co-medication variants, the active-comparator check and the case-level
review were added after the primary results were seen, partly in response to internal review, and are labelled post hoc. The
case-level review was applied by the author and by an AI assessor that used the same profiles and search tools, not by
independent clinicians; the two assessments are therefore not fully independent, and the agreement statistic should be read
in that light. Within-indication comparisons can be affected by channelling (for example tofacitinib as a second-line drug).
FAERS report-source codes that would identify solicited reports directly were not used.

\paragraph{{Conclusions.}} In FAERS, most drug--hearing-loss signals that are not already on a label do not survive adjusted
replication in Canada Vigilance and a review of report provenance; disproportionality analysis remains a
hypothesis-generating tool. Replication in a second spontaneous-reporting database is informative only when both estimates are
adjusted and solicited reports and report clusters are handled explicitly. The few remaining candidates, led by pregabalin,
merit study with exposure-based designs.
"""

abstract = rf"""\begin{{abstract}}
\noindent\textbf{{Introduction:}} Recent analyses of the US Food and Drug Administration Adverse Event
Reporting System (FAERS) reported many drug--hearing-loss signals absent from labels. We assessed which survive adjustment for
hearing-specific biases and external replication.

\noindent\textbf{{Methods:}} Disproportionality analysis of {r(FB['final_reports']/1e6,1)} million deduplicated FAERS reports
(2004--2026) and {r(CB['final_reports']/1e6,1)} million Canada Vigilance reports, following a protocol committed before
analysis. For hearing loss and tinnitus, crude reporting odds ratios (signal: lower 95\% bound $>1$, $\geq$3 cases) were
compared with Mantel--Haenszel estimates stratified by age, sex and period after excluding ear-related indications, reports
naming established ototoxic drugs and lawyer reports. Methods were compared on a label-derived reference set (area under the
receiver operating characteristic curve, AUC) and by replication in Canada Vigilance; robust unlabelled signals underwent
case-by-case review.

\noindent\textbf{{Results:}} Adjustment changed the hearing-loss signal list ({fl['crude_not_adjusted']} of
{fl['crude_signals']} crude signals lost) but not agreement with labels (AUC, crude vs adjusted lower bound:
{r(ref['auc']['crude_logROR025'],3)} vs {r(ref['auc']['adjusted_logMH025'],3)}). Unlabelled signals replicated in Canada more than background but less than labelled ones; power-calibrated (post hoc), labelled signals replicated as predicted (observed/expected
{r(cal['adjusted_labelled']['obs_over_exp'],2)}, 95\% CI {r(cal['adjusted_labelled']['ci'][0],2)}--{r(cal['adjusted_labelled']['ci'][1],2)})
and unlabelled ones did not ({r(cal['adjusted_unlabelled']['obs_over_exp'],2)}, {r(cal['adjusted_unlabelled']['ci'][0],2)}--{r(cal['adjusted_unlabelled']['ci'][1],2)}),
also within strata of consumer reporting. Solicited reports, report clusters and shared
co-medication produced spurious replications. Of {n_rob} robust unlabelled signals (post hoc), {n_rob - nE} had artefactual or known
explanations; {nE} remained ({len(pre_E)}--{K['e2']} under
alternative choices), led by pregabalin.

\noindent\textbf{{Conclusions:}} Most unlabelled FAERS hearing-loss signals do not survive adjusted replication and provenance
review. Findings are hypothesis-generating; cross-database replication needs adjusted estimates and explicit
handling of solicited reports.
\end{{abstract}}"""

keypoints = rf"""\begin{{itemize}}
\item Adjusting FAERS hearing-loss screens for indication, ototoxic co-medication and stimulated reporting changes which drugs
are flagged but not their agreement with drug labels.
\item In Canada Vigilance, labelled hearing-loss signals replicated about as often as predicted and unlabelled ones less than half
as often, also among drugs with similar consumer reporting; solicited reports and shared co-medication produced spurious
replications.
\item After adjusted replication and case-by-case review of report provenance, {nE} of {n_rob} robust unlabelled signals
remained, led by pregabalin; they are hypotheses for exposure-based studies.
\end{{itemize}}"""

# ---------------------------------------------------------------- tables
def methods_table():
    lines = []
    for name, s in (("Hearing loss", HL), ("Tinnitus", TIN)):
        rf_, f_, c_, k_ = s["reference"], s["flow"], s["replication_canada"], s["calibrated_canada"]
        lines += [rf"\multicolumn{{3}}{{l}}{{\textit{{{name}}}}} \\",
                  rf"Reference set, positive/negative ingredients & \multicolumn{{2}}{{c}}{{{rf_['n_pos']}/{rf_['n_neg']}}} \\",
                  rf"AUC, lower 95\% bound & {r(rf_['auc']['crude_logROR025'],3)} & {r(rf_['auc']['adjusted_logMH025'],3)} \\",
                  rf"\quad difference (95\% CI) & \multicolumn{{2}}{{c}}{{{r(rf_['auc_diff_observed'],3)} ({r(rf_['auc_diff_ci'][0],3)} to {r(rf_['auc_diff_ci'][1],3)})}} \\",
                  rf"Sensitivity / specificity of the signal rule & {r(rf_['crude_sens'],2)} / {r(rf_['crude_spec'],2)} & {r(rf_['adj_sens'],2)} / {r(rf_['adj_spec'],2)} \\",
                  rf"Signals (FAERS without Canadian reports) & {f_['crude_signals']} & {f_['adjusted_signals']} \\",
                  rf"Replicated in Canada$^a$: labelled adjusted signals & \multicolumn{{2}}{{c}}{{{frac(c_['adjusted_labelled'])}}} \\",
                  rf"\quad unlabelled adjusted signals & \multicolumn{{2}}{{c}}{{{frac(c_['adjusted_unlabelled'])}}} \\",
                  rf"\quad label-negative adjusted signals & \multicolumn{{2}}{{c}}{{{frac(c_['adjusted_label_negative'])}}} \\",
                  rf"\quad crude-only signals & \multicolumn{{2}}{{c}}{{{frac(c_['crude_only'])}}} \\",
                  rf"\quad background (label-negative non-signals) & \multicolumn{{2}}{{c}}{{{frac(c_['background'])}}} \\",
                  rf"\quad all label-negative ingredients (protocol wording) & \multicolumn{{2}}{{c}}{{{frac(c_['protocol_background_all_label_negative'])}}} \\",
                  rf"Observed/expected replications$^b$: labelled & \multicolumn{{2}}{{c}}{{{k_['adjusted_labelled']['observed']}/{r(k_['adjusted_labelled']['expected_if_true'],1)} = {r(k_['adjusted_labelled']['obs_over_exp'],2)} ({r(k_['adjusted_labelled']['ci'][0],2)}--{r(k_['adjusted_labelled']['ci'][1],2)})}} \\",
                  rf"\quad unlabelled & \multicolumn{{2}}{{c}}{{{k_['adjusted_unlabelled']['observed']}/{r(k_['adjusted_unlabelled']['expected_if_true'],1)} = {r(k_['adjusted_unlabelled']['obs_over_exp'],2)} ({r(k_['adjusted_unlabelled']['ci'][0],2)}--{r(k_['adjusted_unlabelled']['ci'][1],2)})}} \\",
                  rf"\quad label-negative & \multicolumn{{2}}{{c}}{{{k_['adjusted_label_negative']['observed']}/{r(k_['adjusted_label_negative']['expected_if_true'],1)} = {r(k_['adjusted_label_negative']['obs_over_exp'],2)} ({r(k_['adjusted_label_negative']['ci'][0],2)}--{r(k_['adjusted_label_negative']['ci'][1],2)})}} \\",
                  r"\midrule"]
    return "\n".join(lines[:-1])


def posthoc_table():
    names = {"x": "Exclusion (as primary)", "xstrat": "Co-medication as stratum", "xstrong": "Strong ototoxic list only",
             "xnomega": "Without reports listing $\\geq$15 drugs", "2004x": "Exclusion; Canada from 2004"}
    lines = []
    for ev, s in (("Hearing loss", HL), ("Tinnitus", TIN)):
        lines.append(rf"\multicolumn{{9}}{{l}}{{\textit{{{ev}}}}} \\")
        for v, p in s["posthoc"].items():
            lines.append(rf"{names[v]} & {p['adjusted_signals']} & {r(p['auc_adjusted_logMH025'],3)} & "
                         rf"{frac(p['labelled'])} & {frac(p['unlabelled'])} & {frac(p['background'])} & "
                         rf"{'$<$0.001' if p['test_unlabelled_vs_background']['p'] < 0.001 else r(p['test_unlabelled_vs_background']['p'], 3)} & "
                         rf"{r(p['calibrated_labelled']['obs_over_exp'],2)} / {r(p['calibrated_unlabelled']['obs_over_exp'],2)} & "
                         rf"{p['robust']} ({p['robust_unlabelled']}) \\")
        lines.append(r"\midrule")
    return "\n".join(lines[:-1])


def prior_table():
    lines = []
    lab = {"li2024": "Li et al. 2024 (all ages)", "liu2025": "Liu et al. 2025 (children)"}
    for st in ("li2024", "liu2025"):
        for flag, fname in (("yes", "labelled"), ("no", "unlabelled")):
            x = PR[(PR.study == st) & (PR.labelled_per_authors == flag)].iloc[0]
            lines.append(rf"{lab[st]} & {fname} & {int(x.reported)} & {int(x.crude_here)} & {int(x.adjusted_here)} & "
                         rf"{int(x.adjusted_testable)} & {int(x.adjusted_replicated)} & {int(x.posthoc_robust)} \\")
    return "\n".join(lines)


def cand_table():
    lines = []
    for ev in ("hl", "tin"):
        rob = pd.read_csv(RES / f"{ev}_robust_x.csv", index_col=0, low_memory=False)
        sub = RV[RV.event == ev].sort_values(["category", "cases"], ascending=[True, False])
        for _, c in sub.iterrows():
            x = rob.loc[c.drug]
            name = c.drug.title()
            name = rf"\textbf{{{name}}}" if c.category == "E" else name
            cc = ci(x.canadax_mh_or, x.canadax_mh_lo, x.canadax_mh_hi) if np.isfinite(x.canadax_mh_hi) else \
                f"{r(x.canadax_mh_or,2)} ({r(x.canadax_mh_lo,2)}--$\\infty$)"
            lines.append(rf"{name} & {'HL' if ev == 'hl' else 'Tin'} & {int(c.cases)} & "
                         rf"{ci(x.faersnocax_mh_or, x.faersnocax_mh_lo, x.faersnocax_mh_hi)} & {cc} & "
                         rf"{c.consumer.replace('%', chr(92) + '%')} & {c.mega_ge15_drugs.replace('%', chr(92) + '%')} & {c.category} \\")
    return "\n".join(lines)


for name, txt in {"results.tex": results, "discussion.tex": discussion, "abstract.tex": abstract,
                  "keypoints.tex": keypoints, "table_methods.tex": methods_table(),
                  "table_posthoc.tex": posthoc_table(), "table_prior.tex": prior_table(),
                  "table_candidates.tex": cand_table()}.items():
    (OUT / name).write_text(txt)
K["disagreements"].to_csv(RES / "case_review_disagreements.csv", index=False)
print("kappa:", K["agree_pre"], K["agree"], K["n"], round(K["kappa"], 3), "E2:", K["e2"])
print("E:", list(E_list.drug))
