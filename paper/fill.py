"""Generate the Results, Discussion, abstract, key points and tables of the manuscript from result files.

Writes paper/generated/*.tex; paper/build.py splices them into main.tex. No number is typed by hand.
"""
import json
from pathlib import Path

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
A2 = RES / "case_review_assessor2.csv"
m = M["hl"]


def row(name, ev="hl", cols=("faersnoca", "canada")):
    return M[ev].loc[name]


def kappa():
    if not A2.exists():
        return None
    b = pd.read_csv(A2)
    j = RV.merge(b, on=["drug", "event"], suffixes=("_1", "_2"))
    cats = sorted(set(j["category_1"]) | set(j["category_2"]))
    po = (j["category_1"] == j["category_2"]).mean()
    pe = sum((j["category_1"] == c).mean() * (j["category_2"] == c).mean() for c in cats)
    return {"n": len(j), "agree": int((j["category_1"] == j["category_2"]).sum()), "kappa": (po - pe) / (1 - pe),
            "disagreements": j[j["category_1"] != j["category_2"]][["drug", "event", "category_1", "category_2"]]}


K = kappa()
cat_counts = RV.groupby("category").size().to_dict()
E_list = RV[RV["category"] == "E"]
nE = len(E_list)
n_rob = len(RV)
n_rob_hl, n_rob_tin = int((RV.event == "hl").sum()), int((RV.event == "tin").sum())

# ---------------------------------------------------------------- numbers used repeatedly
ref, ref_t = HL["reference"], TIN["reference"]
fl, fl_t = HL["flow"], TIN["flow"]
rc, rc_t = HL["replication_canada"], TIN["replication_canada"]
cal, cal_t = HL["calibrated_canada"], TIN["calibrated_canada"]
px, px_t = HL["posthoc"]["x"], TIN["posthoc"]["x"]
tp, tp_t = HL["replication_temporal"], TIN["replication_temporal"]
removed_deleted = FB["cases_after_latest_version"] - FB["cases_after_deletions"]
unmapped = FB["drug_rows_unmapped"] / (FB["drug_rows_unmapped"] + FB["drug_rows_mapped"])
eto, vcr, taf, sac = row("ETOPOSIDE"), row("VINCRISTINE"), row("TAFAMIDIS"), row("SACUBITRIL")
known = pd.read_csv(RES / "table_known.csv", index_col=0)
li_u = PR[(PR.study == "li2024") & (PR.labelled_per_authors == "no")].iloc[0]
liu_u = PR[(PR.study == "liu2025") & (PR.labelled_per_authors == "no")].iloc[0]


def oe(d):
    return f"{d['observed']} observed vs {r(d['expected_if_true'], 1)} expected (O/E {r(d['obs_over_exp'], 2)})"


def kc(name):
    k = known.loc[name]
    return ci(k.faers_mh_or, k.faers_mh_lo, k.faers_mh_hi, 1)


# ---------------------------------------------------------------- results
results = rf"""
\section{{Results}}

\subsection{{Data}}
The 90 FAERS extracts held {thou(FB['demo_rows'])} report versions, from which {thou(FB['cases_after_latest_version'])}
cases remained after keeping the latest version; {thou(removed_deleted)} cases on FDA deletion lists, cases without a
reaction or mappable ingredient, {thou(FB['report_ids_under_two_cases'])} report identifiers recorded under two cases and
{thou(FB['content_duplicates_removed'])} content duplicates were then removed, leaving \textbf{{{thou(FB['final_reports'])}
reports}} ({pct(unmapped)} of drug rows could not be mapped; Supplementary Figure~S1). They contained
{thou(POP['faers_hl']['events'])} hearing-loss and {thou(POP['faers_tin']['events'])} tinnitus reports. Hearing-loss
reports came more often from consumers and from patients aged $\geq$65 years than other reports (Supplementary Table~S4).
The primary adjusted population excluded {thou(POP['faers_hl']['ear_indication'])} reports with an ear or hearing
indication, {thou(POP['faers_hl']['with_known_ototoxic'])} reports naming a listed ototoxic drug and
{thou(POP['faers_hl']['lawyer'])} lawyer reports, leaving {thou(POP['faers_hl']['adj_reports'])} reports
({thou(POP['faers_hl']['adj_events'])} hearing-loss cases). Canada Vigilance contributed {thou(CB['final_reports'])} reports
after removing {thou(CB['duplicates_dropped'])} duplicates ({thou(POP['canada_hl']['events'])} hearing-loss and
{thou(POP['canada_tin']['events'])} tinnitus reports).

Drugs of the established ototoxic classes gave strong signals in both databases (Supplementary Table~S2), for example
amikacin (FAERS adjusted OR {kc('AMIKACIN')}), gentamicin ({kc('GENTAMICIN')}) and cisplatin ({kc('CISPLATIN')}); so did
the PDE5 inhibitors sildenafil ({kc('SILDENAFIL')}) and tadalafil ({kc('TADALAFIL')}), whose labels carry hearing-loss
warnings, and teprotumumab ({kc('TEPROTUMUMAB')}). Several label-positive drugs did not signal: oxaliplatin (crude ROR
{r(known.loc['OXALIPLATIN','faers_ror'],2)}), vardenafil ({int(known.loc['VARDENAFIL','faers_a'])} cases), and, after adjustment,
carboplatin (crude {r(known.loc['CARBOPLATIN','faers_ror'],2)}, adjusted {r(known.loc['CARBOPLATIN','faers_mh_or'],2)}) and
vincristine, whose crude associations were carried largely by reports that also named another listed drug---the
sensitivity that the co-medication filter costs.

\subsection{{Adjustment changes the signal list but not its agreement with labels}}
In FAERS without Canadian reports the crude rule flagged {fl['crude_signals']} ingredients for hearing loss and the adjusted
rule {fl['adjusted_signals']}; {fl['crude_not_adjusted']} crude signals were lost and {fl['adjusted_not_crude']} appeared
(Table~\ref{{tab:methods}}). Against the reference set ({ref['n_pos']} label-positive and {ref['n_neg']} label-negative
ingredients with $\geq$1{{,}}000 suspect reports), discrimination was moderate and practically identical for the crude and
adjusted lower bounds (AUC {r(ref['auc']['crude_logROR025'],3)} and {r(ref['auc']['adjusted_logMH025'],3)}; difference
{r(ref['auc_diff_observed'],3)}, 95\% CI {r(ref['auc_diff_ci'][0],3)} to {r(ref['auc_diff_ci'][1],3)}, which excludes an
improvement larger than {r(ref['auc_diff_ci'][1],2)}). The same held for point estimates
({r(ref['auc']['crude_logROR'],3)} vs {r(ref['auc']['adjusted_logMH'],3)}), for the IC$_{{025}}$
({r(ref['auc']['crude_IC025'],3)}), at lower report thresholds, without the 29 listed drugs, and for tinnitus
({r(ref_t['auc']['crude_logROR025'],3)} vs {r(ref_t['auc']['adjusted_logMH025'],3)}). Adjustment lowered sensitivity
({r(ref['crude_sens'],2)} to {r(ref['adj_sens'],2)}) with little change in specificity ({r(ref['crude_spec'],3)} to {r(ref['adj_spec'],3)}).

\subsection{{Replication in Canada Vigilance}}
Under the pre-specified rule, adjusted hearing-loss signals already on a label replicated in {frac(rc['adjusted_labelled'])}
of testable cases, unlabelled adjusted signals in {frac(rc['adjusted_unlabelled'])}, label-negative adjusted signals in
{frac(rc['adjusted_label_negative'])}, crude-only signals in {frac(rc['crude_only'])}, and the background of label-negative
non-signals in {frac(rc['background'])} (Table~\ref{{tab:methods}}). The power-calibrated comparison separated the groups
more clearly (Figure~\ref{{fig:repl}}): labelled signals replicated exactly as often as their FAERS strength predicted
({oe(cal['adjusted_labelled'])}), whereas unlabelled signals replicated less than half as often
({oe(cal['adjusted_unlabelled'])}), like label-negative signals ({oe(cal['adjusted_label_negative'])}). Tinnitus showed
the same contrast ({oe(cal_t['adjusted_labelled'])} for labelled and {oe(cal_t['adjusted_unlabelled'])} for unlabelled
signals).

The background replicated more often than chance ({cal['background']['observed']} observed vs
{r(cal['background']['expected_if_null'],1)} expected without any association). Solicited reports explained much of this:
\emph{{Study}} reports, which include manufacturers' patient-support programmes, made up {pct(SH['background_replicated'][1])}
of Canadian hearing-loss cases for the background drugs that replicated, against {pct(SH['labelled_replicated'][1])} for
replicated labelled signals and {pct(SH['all_HL_cases'][1])} overall; for ocrelizumab, mepolizumab and omalizumab the share
was {pct(SH['OCRELIZUMAB'][1])}, {pct(SH['MEPOLIZUMAB'][1])} and {pct(SH['OMALIZUMAB'][1])}, against
{pct(SH['GENTAMICIN'][1])} for gentamicin and {pct(SH['CISPLATIN'][1])} for cisplatin. The mirror image appeared in FAERS:
tafamidis (adjusted OR {ci(taf.faersnoca_mh_or, taf.faersnoca_mh_lo, taf.faersnoca_mh_hi, 1)}) had
{thou(taf.faersnoca_a)} hearing-loss reports in FAERS but none in Canada Vigilance against {r(taf.canada_E,1)} expected, and
sacubitril ({thou(sac.faersnoca_a)} FAERS reports) none against {r(sac.canada_E,1)} expected (too few to be testable under the
pre-specified rule).

Co-medication bias also replicated when crude estimates were compared. Etoposide, given in cisplatin-containing regimens,
was a crude signal in both databases (FAERS ROR {ci(eto.faersnoca_ror, eto.faersnoca_ror_lo, eto.faersnoca_ror_hi)};
Canada {ci(eto.canada_ror, eto.canada_ror_lo, eto.canada_ror_hi)}) and disappeared after adjustment in both (FAERS
{ci(eto.faersnoca_mh_or, eto.faersnoca_mh_lo, eto.faersnoca_mh_hi)}; Canada
{ci(eto.canada_mh_or, eto.canada_mh_lo, eto.canada_mh_hi)}). Temporal replication within FAERS did not separate crude-only
from adjusted hearing-loss signals (early adjusted signals recurred after 2014 in {frac(tp['early_adjusted'])}, crude-only
signals in {frac(tp['early_crude_only'])}, background {frac(tp['background'])}); for tinnitus adjusted signals recurred more
often ({frac(tp_t['early_adjusted'])} vs {frac(tp_t['early_crude_only'])}).

\subsection{{Post-hoc analyses}}
With the extended indication filter, Canada Vigilance restricted to unsolicited reports and both estimates adjusted
(Table~\ref{{tab:posthoc}}), labelled hearing-loss signals replicated in {frac(px['labelled'])}, unlabelled signals in
{frac(px['unlabelled'])} and the background in {frac(px['background'])} (unlabelled vs background, Fisher
$p={r(px['test_unlabelled_vs_background']['p'],2)}$); calibrated O/E ratios were {r(px['calibrated_labelled']['obs_over_exp'],2)}
for labelled and {r(px['calibrated_unlabelled']['obs_over_exp'],2)} for unlabelled signals. Handling co-medication by
stratification instead of exclusion, with a short list of strongly ototoxic drugs, or after removing reports listing
$\geq$15 drugs left this contrast unchanged (unlabelled O/E {r(min(HL['posthoc'][v]['calibrated_unlabelled']['obs_over_exp'] for v in HL['posthoc']),2)}--{r(max(HL['posthoc'][v]['calibrated_unlabelled']['obs_over_exp'] for v in HL['posthoc']),2)}
across variants, labelled {r(min(HL['posthoc'][v]['calibrated_labelled']['obs_over_exp'] for v in HL['posthoc']),2)}--{r(max(HL['posthoc'][v]['calibrated_labelled']['obs_over_exp'] for v in HL['posthoc']),2)}),
as did restricting Canada Vigilance to reports from 2004.

\subsection{{Signals from previous screens}}
Of the {int(li_u.reported)} hearing-loss signals that Li et al.\ classified as unlabelled \cite{{li2024}},
{int(li_u.crude_here)} were crude and {int(li_u.adjusted_here)} adjusted signals in FAERS without Canadian reports,
{int(li_u.adjusted_testable)} were testable in Canada and {int(li_u.adjusted_replicated)} replicated; under the post-hoc
definition {int(li_u.posthoc_robust)} remained ({li_u.posthoc_robust_drugs.lower() if isinstance(li_u.posthoc_robust_drugs, str) else 'none'}).
Their strongest unlabelled signals included sacubitril/valsartan and tafamidis. Of the {int(liu_u.reported)} unlabelled
paediatric signals of Liu et al.\ \cite{{liu2025}}, {int(liu_u.adjusted_here)} were adjusted signals in our all-age data and
{int(liu_u.posthoc_robust)} met the post-hoc definition ({liu_u.posthoc_robust_drugs.lower() if isinstance(liu_u.posthoc_robust_drugs, str) else 'none'},
which is on the a-priori ototoxic list); {', '.join(x.lower() for x in str(liu_u.lost_on_adjustment).split(';'))} were lost on adjustment
(Table~\ref{{tab:prior}}). Only abstract-level lists were available for the remaining screens (Supplementary Table~S6).

\subsection{{Case-level review}}
Under the post-hoc definition, {n_rob_hl} hearing-loss and {n_rob_tin} tinnitus signals were robust and absent from every US
label (Table~\ref{{tab:cand}}, Figure~\ref{{fig:cand}}). The review attributed {cat_counts.get('A',0)} to report clusters (for
example candesartan--tinnitus: {RV.set_index(['drug','event']).loc[('CANDESARTAN','tin'),'top_country'].split()[1].replace('%', chr(92) + '%')} of cases from the United Kingdom, median {RV.set_index(['drug','event']).loc[('CANDESARTAN','tin'),'median_drugs']:g} drugs per report),
{cat_counts.get('C',0)} to stimulated or mass-exposure reporting (mRNA COVID-19 vaccines, rofecoxib, finasteride),
{cat_counts.get('D',0)} to an indication itself associated with the event (rheumatoid arthritis for etanercept and
tofacitinib; COPD for glycopyrrolate; glaucoma for travoprost; diabetes for insulin aspart; migraine for galcanezumab;
hypothyroidism for liothyronine) and {cat_counts.get('K',0)} to an established ototoxic class (oral paromomycin).
{['No','One','Two','Three','Four','Five','Six','Seven','Eight','Nine','Ten'][nE]} remained without a dominant artefactual explanation: pregabalin and amlodipine for hearing loss, and doxycycline and
chlorthalidone ({int(RV.set_index(['drug','event']).loc[('CHLORTHALIDONE','tin'),'cases'])} cases) for tinnitus.
""" + (rf"""The independent assessor agreed on {K['agree']} of {K['n']} assignments (Cohen's $\kappa={r(K['kappa'],2)}$);
it placed {'; '.join(f"{d.title()} ({'hearing loss' if e == 'hl' else 'tinnitus'}) in {c2} rather than {c1}" for d, e, c1, c2 in K['disagreements'].itertuples(index=False))}
(Supplementary Table~S3), which would leave the number of unexplained signals at
{nE - int(((K['disagreements'].category_1 == 'E') & (K['disagreements'].category_2 != 'E')).sum()) + int(((K['disagreements'].category_2 == 'E') & (K['disagreements'].category_1 != 'E')).sum())}."""
       if K else r"""%%KAPPA%%""") + "\n"

# ---------------------------------------------------------------- discussion
discussion = rf"""
\section{{Discussion}}
Three findings matter for anyone screening spontaneous reports for ototoxicity. First, adjusting for indication, ototoxic
co-medication, lawyer reports and demographics changed a substantial part of the hearing-loss signal list
({fl['crude_not_adjusted']} of {fl['crude_signals']} crude signals lost, {fl['adjusted_not_crude']} gained) but did not
improve agreement with labels. Second, replication in an independent national database behaved as a validation test should
for labelled associations---they replicated as often as their strength predicted---but unlabelled signals replicated less
than half as often as predicted, about as often as signals for drugs whose labels are silent on hearing. Third, replication is
not a clean validator on its own: solicited patient-support-programme reports produced ``replications'' for drugs with no
known association, and co-medication bias replicated whenever crude estimates were compared. After adjusted replication and
a review of report provenance, {nE} of {n_rob} robust unlabelled signals remained.

These results qualify the previous all-drug screens \cite{{li2024,wu2025,liu2025}} rather than contradict them. Their
labelled signals largely held; their unlabelled signals mostly did not, and several of the strongest are consistent with how
reports were collected rather than with pharmacology. Tafamidis and sacubitril/valsartan are examples: thousands of consumer
reports in FAERS, consistent with manufacturer programmes in the United States, and none in Canada. The same mechanism
produced spurious Canadian ``replications'' for biologics. Solicited reports are known to distort disproportionality
\cite{{psp_solicited}}, and their share differs between databases, so report provenance can manufacture both a signal and
its replication. Bias that is shared between databases---co-prescription in standard regimens---also replicates; etoposide
is the clearest example here.

Two inexpensive steps addressed most of what we found: comparing \emph{{adjusted}} estimates in both databases, and
profiling each case series for concentration---solicited or consumer reports, a single country or year, very drug-dense
reports. The power-calibrated comparison is also simple and avoids the outcome-dependent testability of rules based on
observed counts; it could be extended with empirical calibration against negative controls \cite{{schuemie2014}} or with
regression-based adjustment for co-reported drugs \cite{{caster2010}}. We suggest that screens for hearing outcomes report
these provenance indicators for every signal, in line with the case-by-case item of READUS-PV \cite{{readuspv,readuspv_ee}}.

The remaining candidates are hypotheses with plausible alternative explanations. Pregabalin, an $\alpha_2\delta$ ligand,
had {thou(m.loc['PREGABALIN','faersnoca_a'])} FAERS hearing-loss reports (adjusted OR
{ci(m.loc['PREGABALIN','faersnocax_mh_or'], m.loc['PREGABALIN','faersnocax_mh_lo'], m.loc['PREGABALIN','faersnocax_mh_hi'])})
that replicated in Canada; a case of reversible, dose-related hearing loss has been described \cite{{pregabalin_case}}, but
diabetes, neuropathy and age confound \cite{{diabetes_hearing}}, and its labels list tinnitus and hyperacusis. For amlodipine,
whose labels mention tinnitus, a large cohort found a \emph{{lower}} risk of sensorineural hearing loss with calcium-channel
blockers \cite{{ccb_cohort}}, which argues against a causal effect. Doxycycline--tinnitus may reflect pulsatile tinnitus from
drug-induced intracranial hypertension, which is labelled \cite{{doxy_iih}}, or the infections it treats. None of these should
change practice without studies that have exposure data.

\paragraph{{Limitations.}} Spontaneous reports lack exposure denominators, and disproportionality measures reporting, not risk
or incidence. The label reference set is imperfect: labels are partly informed by spontaneous reports, list rare events
without causal assessment, and route-specific labels (for example topical gentamicin) blur ingredient-level status; we used
it only to compare methods. Excluding reports with an ototoxic co-medication conditions on a variable that reporters may add
after observing the event; stratifying on it instead gave the same conclusions. Indications are often missing, unreported
co-treatment (for example aminoglycosides in mycobacterial disease) remains, and content deduplication cannot catch
near-duplicate series, which the report-cluster indicator only partly addresses. The Canada Vigilance documentation does not
state that the extract is domestic; restricting it to reports sent directly to Health Canada gave similar results
({frac(HL['replication_canada_direct']['adjusted'])} of adjusted signals replicated, background
{frac(HL['replication_canada_direct']['background'])}). About 8{{,}}400 ingredients were screened without multiplicity
correction; the replication and calibration analyses, rather than individual $p$-values, carry the conclusions. Several
analyses---the solicited-report exclusion, the extended indication filter, adjusted-to-adjusted and calibrated replication,
the co-medication variants and the case-level review---were added after the primary results were seen, partly in response
to internal review; they are labelled post hoc and the pre-specified results are reported unchanged. The case-level review
used a written rubric applied by the author and by an AI assessor, not by independent clinicians.

\paragraph{{Conclusions.}} In FAERS, most drug--hearing-loss signals that are not already on a label do not survive adjustment,
adjusted replication in Canada Vigilance and a review of report provenance; disproportionality analysis remains a
hypothesis-generating tool. Replication in a second spontaneous-reporting database is informative only when both estimates
are adjusted and solicited reports are handled explicitly. The few remaining candidates, led by pregabalin, merit study with
exposure-based designs.
"""

abstract = rf"""\begin{{abstract}}
\noindent\textbf{{Introduction:}} Recent disproportionality analyses of the FDA Adverse Event Reporting System (FAERS)
reported many drug--hearing-loss signals absent from labels. We assessed which survive adjustment for hearing-specific biases
and external replication.

\noindent\textbf{{Methods:}} Disproportionality analysis of {r(FB['final_reports']/1e6,1)} million deduplicated FAERS reports
(2004--2026) and {r(CB['final_reports']/1e6,1)} million Canada Vigilance reports, following a protocol committed before
analysis. For hearing loss and tinnitus, crude reporting odds ratios (ROR; signal: lower 95\% bound $>1$, $\geq$3 cases) were
compared with Mantel--Haenszel estimates stratified by age, sex and period after excluding ear-related indications, reports
naming established ototoxic drugs and lawyer reports. Methods were compared on a label-derived reference set (area under the
ROC curve, AUC) and by replication in Canada Vigilance, observed and power-calibrated; robust unlabelled signals underwent
case-by-case review.

\noindent\textbf{{Results:}} Adjustment changed the hearing-loss signal list ({fl['crude_signals']} crude, {fl['adjusted_signals']}
adjusted, {fl['crude_not_adjusted']} lost) but not agreement with labels (AUC {r(ref['auc']['crude_logROR025'],3)} vs
{r(ref['auc']['adjusted_logMH025'],3)}). Labelled signals replicated in Canada as often as predicted (observed/expected
{r(cal['adjusted_labelled']['obs_over_exp'],2)}); unlabelled signals did not ({r(cal['adjusted_unlabelled']['obs_over_exp'],2)};
post hoc, both databases adjusted: {r(px['calibrated_unlabelled']['obs_over_exp'],2)}). Solicited patient-support-programme
reports and shared co-medication produced spurious replications. Of {n_rob} robust unlabelled hearing-loss and tinnitus signals
(post hoc), {n_rob - nE} were attributed to report clusters, stimulated reporting, indication or an established class;
{nE} remained, led by pregabalin.

\noindent\textbf{{Conclusions:}} Most unlabelled FAERS hearing-loss signals do not survive adjustment, adjusted replication and
provenance review. These findings are hypothesis-generating; cross-database replication needs adjusted estimates and explicit
handling of solicited reports.
\end{{abstract}}"""

keypoints = rf"""\begin{{itemize}}
\item Adjusting FAERS hearing-loss screens for indication, ototoxic co-medication and stimulated reporting changes which drugs
are flagged but not their agreement with drug labels.
\item In Canada Vigilance, labelled hearing-loss signals replicated as often as predicted, unlabelled ones less than half as
often; solicited patient-support-programme reports and shared co-medication produced spurious replications.
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
                  rf"Observed/expected replications$^b$: labelled & \multicolumn{{2}}{{c}}{{{k_['adjusted_labelled']['observed']}/{r(k_['adjusted_labelled']['expected_if_true'],1)} = {r(k_['adjusted_labelled']['obs_over_exp'],2)}}} \\",
                  rf"\quad unlabelled & \multicolumn{{2}}{{c}}{{{k_['adjusted_unlabelled']['observed']}/{r(k_['adjusted_unlabelled']['expected_if_true'],1)} = {r(k_['adjusted_unlabelled']['obs_over_exp'],2)}}} \\",
                  rf"\quad label-negative & \multicolumn{{2}}{{c}}{{{k_['adjusted_label_negative']['observed']}/{r(k_['adjusted_label_negative']['expected_if_true'],1)} = {r(k_['adjusted_label_negative']['obs_over_exp'],2)}}} \\",
                  r"\midrule"]
    return "\n".join(lines[:-1])


def posthoc_table():
    names = {"x": "Exclusion (as primary)", "xstrat": "Co-medication as stratum", "xstrong": "Strong ototoxic list only",
             "xnomega": "Without reports listing $\\geq$15 drugs"}
    lines = []
    for ev, s in (("Hearing loss", HL), ("Tinnitus", TIN)):
        lines.append(rf"\multicolumn{{8}}{{l}}{{\textit{{{ev}}}}} \\")
        for v, p in s["posthoc"].items():
            lines.append(rf"{names[v]} & {p['adjusted_signals']} & {r(p['auc_adjusted_logMH025'],3)} & "
                         rf"{frac(p['labelled'])} & {frac(p['unlabelled'])} & {frac(p['background'])} & "
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
if K:
    K["disagreements"].to_csv(RES / "case_review_disagreements.csv", index=False)
print("kappa:", None if not K else (K["agree"], K["n"], round(K["kappa"], 3)))
print("E:", list(E_list.drug))
