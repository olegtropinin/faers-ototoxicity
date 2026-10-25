"""Evaluate crude vs adjusted signal detection: label reference set, replication in Canada Vigilance
(observed and power-calibrated), temporal replication, sensitivity variants, prior screens.

Reads results/<event>_master.csv; writes results/summary_<event>.json, results/<event>_robust*.csv,
results/prior_screens.csv and figures/roc_<event>.png.
"""
import json
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, poisson

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from signals import KNOWN_OTOTOXIC  # noqa: E402

RES, FIG = Path("results"), Path("figures")
FIG.mkdir(exist_ok=True)
rng = np.random.default_rng(20261026)
MIN_REPORTS = 1000
FLOOR = -20.0
VARIANTS = ["x", "xstrat", "xstrong", "xnomega"]


def half_up(x, nd=0):
    return float(Decimal(str(x)).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP))


def crude_sig(m, p):
    return (m[f"{p}_a"] >= 3) & (m[f"{p}_ror_lo"] > 1) & (m[f"{p}_ic025"] > 0)


def ror_sig(m, p):
    """Crude rule with the same single criterion as the adjusted rule (symmetric comparison)."""
    return (m[f"{p}_a"] >= 3) & (m[f"{p}_ror_lo"] > 1)


def adj_sig(m, p):
    return (m[f"{p}_a_adj"] >= 3) & (m[f"{p}_mh_lo"] > 1)


def testable(m, p):
    return np.maximum(m[f"{p}_a"].fillna(0), m[f"{p}_E"].fillna(0)) >= 3


def replicated(m, p):
    return (m[f"{p}_a"] >= 3) & (m[f"{p}_ror_lo"] > 1)


def auc(score, y):
    pos, neg = score[y], score[~y]
    r = pd.Series(np.concatenate([pos, neg])).rank().values
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def wilson(k, n):
    if n == 0:
        return (np.nan, np.nan, np.nan)
    p, z = k / n, 1.959963984540054
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (p, c - h, c + h)


def rate(den, num):
    n, k = int(den.sum()), int((den & num).sum())
    p, lo, hi = wilson(k, n)
    return {"k": k, "n": n, "rate": p, "lo": lo, "hi": hi}


def fisher(r1, r2):
    """Two rate dicts -> odds ratio and two-sided Fisher p for r1 vs r2."""
    o, p = fisher_exact([[r1["k"], r1["n"] - r1["k"]], [r2["k"], r2["n"] - r2["k"]]])
    return {"odds_ratio": o, "p": p}


def score(x):
    return np.log(x).replace([-np.inf, np.inf], FLOOR).fillna(FLOOR).values


def reference(m, ref, min_reports=MIN_REPORTS, exclude=None):
    base = (m["faers_n"] >= min_reports) & ref.isin(["pos", "neg"])
    if exclude is not None:
        base &= ~m.index.isin(exclude)
    r = m[base]
    y = (ref[base] == "pos").values
    scores = {"crude_logROR025": score(r["faers_ror_lo"]), "adjusted_logMH025": score(r["faers_mh_lo"]),
              "crude_IC025": r["faers_ic025"].fillna(FLOOR).values,
              "crude_logROR": score(r["faers_ror"]), "adjusted_logMH": score(r["faers_mh_or"])}
    a = {k: auc(v, y) for k, v in scores.items()}
    idx, boot = np.arange(len(y)), []
    for _ in range(2000):
        b = rng.choice(idx, len(idx))
        if y[b].all() or (~y[b]).all():
            continue
        boot.append(auc(scores["adjusted_logMH025"][b], y[b]) - auc(scores["crude_logROR025"][b], y[b]))
    cs, rs, as_ = crude_sig(r, "faers").values, ror_sig(r, "faers").values, adj_sig(r, "faers").values
    return {"n_pos": int(y.sum()), "n_neg": int((~y).sum()), "min_reports": min_reports, "auc": a,
            "auc_diff_observed": a["adjusted_logMH025"] - a["crude_logROR025"],
            "auc_diff_ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "adjusted_score_at_floor": int((scores["adjusted_logMH025"] == FLOOR).sum()),
            "crude_sens": float(cs[y].mean()), "crude_spec": float(1 - cs[~y].mean()),
            "ror_sens": float(rs[y].mean()), "ror_spec": float(1 - rs[~y].mean()),
            "adj_sens": float(as_[y].mean()), "adj_spec": float(1 - as_[~y].mean())}, scores, y


def roc_figure(ev, scores, y, aucs):
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    for k, c in (("crude_logROR025", "#999"), ("crude_IC025", "#bbb"), ("adjusted_logMH025", "#0a8a5f")):
        v = scores[k]
        o = np.argsort(-v)
        tpr = np.concatenate([[0], np.cumsum(y[o]) / y.sum()])
        fpr = np.concatenate([[0], np.cumsum(~y[o]) / (~y).sum()])
        ax.plot(fpr, tpr, color=c, lw=2 if "adj" in k else 1.2, label=f"{k} (AUC {aucs[k]:.3f})")
    ax.plot([0, 1], [0, 1], ":", color="#ccc")
    ax.set_xlabel("1 - specificity (label-negative)")
    ax.set_ylabel("sensitivity (label-positive)")
    ax.set_title(f"{'Hearing loss' if ev == 'hl' else 'Tinnitus'}: {int(y.sum())} positive, "
                 f"{int((~y).sum())} negative ingredients", fontsize=9)
    ax.legend(fontsize=6.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / f"roc_{ev}.png", dpi=200)
    plt.close(fig)


def calibrated(m, mask, faers_or, ca):
    """Observed vs power-expected replications. Replication = Canadian count >= a_crit, the smallest count >= 3
    with Poisson P(X >= a | E) < 0.025. Expected if true uses lambda = E x FAERS adjusted OR (floored at 1)."""
    E = m[f"{ca}_E"].fillna(0).values
    a = m[f"{ca}_a"].fillna(0).values
    orr = m[faers_or].replace([np.inf], np.nan).fillna(1).clip(lower=1).values
    sel = mask.fillna(False).values & (E > 0)
    obs = exp1 = exp0 = 0.0
    n = 0
    for e, aa, o in zip(E[sel], a[sel], orr[sel]):
        crit = 3
        while poisson.sf(crit - 1, e) >= 0.025:
            crit += 1
        obs += aa >= crit
        exp1 += poisson.sf(crit - 1, e * o)
        exp0 += poisson.sf(crit - 1, e)
        n += 1
    return {"n": n, "observed": int(obs), "expected_if_true": exp1, "expected_if_null": exp0,
            "obs_over_exp": obs / exp1 if exp1 > 0 else np.nan}


def evaluate(ev):
    m = pd.read_csv(RES / f"{ev}_master.csv", index_col="ingredient", low_memory=False)
    ref = m[f"ref_{ev}"]
    lab_col = "hl_any" if ev == "hl" else "tin_any"
    combo_col = "hl_combo" if ev == "hl" else "tin_combo"
    # "labelled" = the event is mentioned in any US label containing the drug (single-ingredient or combination);
    # "unlabelled" therefore includes drugs with no US label in the openFDA export
    labelled = (m[lab_col] == True) | (m.get(combo_col, pd.Series(False, index=m.index)) == True)  # noqa: E712
    neg = ref == "neg"
    out = {}

    # 1. reference set
    out["reference"], scores, y = reference(m, ref)
    roc_figure(ev, scores, y, out["reference"]["auc"])
    out["reference_sensitivity"] = {f"min_reports_{k}": reference(m, ref, k)[0]["auc"] for k in (100, 500)}
    out["reference_sensitivity"]["without_29_listed"] = reference(m, ref, exclude=KNOWN_OTOTOXIC)[0]["auc"]

    # 2. pre-specified replication: FAERS without Canada -> Canada (crude), all reports
    m["crude"], m["adjusted"] = crude_sig(m, "faersnoca"), adj_sig(m, "faersnoca")
    t, r = testable(m, "canada"), replicated(m, "canada")
    bg = neg & ~m["crude"] & ~m["adjusted"]
    out["flow"] = {"crude_signals": int(m["crude"].sum()), "adjusted_signals": int(m["adjusted"].sum()),
                   "crude_not_adjusted": int((m["crude"] & ~m["adjusted"]).sum()),
                   "adjusted_not_crude": int((m["adjusted"] & ~m["crude"]).sum()),
                   "symmetric_ror_signals": int(ror_sig(m, "faersnoca").sum()),
                   "symmetric_ror_not_adjusted": int((ror_sig(m, "faersnoca") & ~m["adjusted"]).sum()),
                   "adjusted_not_symmetric_ror": int((m["adjusted"] & ~ror_sig(m, "faersnoca")).sum())}
    rc = {"adjusted_labelled": rate(t & m["adjusted"] & labelled, r),
          "adjusted_unlabelled": rate(t & m["adjusted"] & ~labelled, r),
          "adjusted_label_negative": rate(t & m["adjusted"] & neg, r),
          "crude_only": rate(t & m["crude"] & ~m["adjusted"], r),
          "background": rate(t & bg, r)}
    rc["test_unlabelled_vs_background"] = fisher(rc["adjusted_unlabelled"], rc["background"])
    rc["test_labelled_vs_background"] = fisher(rc["adjusted_labelled"], rc["background"])
    out["replication_canada"] = rc
    td, rd = testable(m, "canadadirect"), replicated(m, "canadadirect")
    out["replication_canada_direct"] = {"adjusted": rate(td & m["adjusted"], rd), "background": rate(td & bg, rd)}
    out["calibrated_canada"] = {
        "adjusted_labelled": calibrated(m, m["adjusted"] & labelled, "faersnoca_mh_or", "canada"),
        "adjusted_unlabelled": calibrated(m, m["adjusted"] & ~labelled, "faersnoca_mh_or", "canada"),
        "adjusted_label_negative": calibrated(m, m["adjusted"] & neg, "faersnoca_mh_or", "canada"),
        "background": calibrated(m, bg, "faersnoca_mh_or", "canada")}

    # 3. temporal
    m["early_crude"], m["early_adj"] = crude_sig(m, "early"), adj_sig(m, "early")
    lt, lr = testable(m, "late"), replicated(m, "late")
    out["replication_temporal"] = {
        "early_adjusted": rate(lt & m["early_adj"], lr),
        "early_crude_only": rate(lt & m["early_crude"] & ~m["early_adj"], lr),
        "background": rate(lt & neg & ~m["early_crude"] & ~m["early_adj"], lr)}

    # 4. post hoc: extended indication filter, Canada without solicited reports, both adjusted
    sel_ref = (m["faers_n"] >= MIN_REPORTS) & ref.isin(["pos", "neg"])
    ph = {}
    for v in VARIANTS:
        f, c = f"faersnoca{v}", f"canada{v}"
        if f"{f}_mh_lo" not in m or f"{c}_mh_lo" not in m:
            continue
        sig = adj_sig(m, f)
        rep_adj = (m[f"{c}_a_adj"] >= 3) & (m[f"{c}_mh_lo"] > 1)
        tst = np.maximum(m[f"{c}_a"].fillna(0), m[f"{c}_E"].fillna(0)) >= 3
        bgv = neg & ~m["crude"] & ~sig
        res = {"adjusted_signals": int(sig.sum()),
               "lost_vs_primary": int((m["adjusted"] & ~sig).sum()),
               "auc_adjusted_logMH025": auc(score(m.loc[sel_ref, f"{f}_mh_lo"]), (ref[sel_ref] == "pos").values),
               "labelled": rate(tst & sig & labelled, rep_adj),
               "unlabelled": rate(tst & sig & ~labelled, rep_adj),
               "label_negative_signals": rate(tst & sig & neg, rep_adj),
               "background": rate(tst & bgv, rep_adj),
               "calibrated_unlabelled": calibrated(m, sig & ~labelled, f"{f}_mh_or", c),
               "calibrated_labelled": calibrated(m, sig & labelled, f"{f}_mh_or", c)}
        res["test_unlabelled_vs_background"] = fisher(res["unlabelled"], res["background"])
        robust = m[sig & rep_adj].copy()
        robust["labelled"] = labelled[robust.index]
        res["robust"] = len(robust)
        res["robust_unlabelled"] = int((~robust["labelled"]).sum())
        robust.to_csv(RES / f"{ev}_robust_{v}.csv")
        ph[v] = res
    out["posthoc"] = ph
    json.dump(out, open(RES / f"summary_{ev}.json", "w"), indent=1, default=float)
    return out


def prior_screens():
    rows = []
    for f in sorted(Path("data/raw/prior").glob("*.csv")):
        p = pd.read_csv(f)
        for scope, ev in (("hearing_loss", "hl"), ("hearing_any", "hl"), ("tinnitus", "tin")):
            sub = p[p["event_scope"] == scope]
            if sub.empty:
                continue
            m = pd.read_csv(RES / f"{ev}_master.csv", index_col="ingredient", low_memory=False)
            for lab_flag in ("yes", "no", "unknown"):
                names = sub[sub["labeled_per_authors"] == lab_flag]["drug"].str.upper().str.strip().drop_duplicates()
                if names.empty:
                    continue
                x = m.reindex(names)
                crude, adj = crude_sig(x, "faersnoca"), adj_sig(x, "faersnoca")
                t, r = testable(x, "canada"), replicated(x, "canada")
                adjx = adj_sig(x, "faersnocax")
                radj = (x["canadax_a_adj"] >= 3) & (x["canadax_mh_lo"] > 1)
                rows.append({"study": f.stem, "scope": scope, "labelled_per_authors": lab_flag,
                             "reported": len(names), "matched": int(x["faersnoca_n"].notna().sum()),
                             "crude_here": int(crude.sum()), "adjusted_here": int(adj.sum()),
                             "adjusted_testable": int((adj & t).sum()), "adjusted_replicated": int((adj & r).sum()),
                             "posthoc_robust": int((adjx & radj).sum()),
                             "posthoc_robust_drugs": ";".join(sorted(x.index[(adjx & radj).fillna(False).values])),
                             "lost_on_adjustment": ";".join(sorted(x.index[(crude & ~adj).fillna(False).values]))})
    out = pd.DataFrame(rows)
    out.to_csv(RES / "prior_screens.csv", index=False)
    return out


if __name__ == "__main__":
    if sys.argv[1:] == ["prior"]:
        print(prior_screens().to_string())
        sys.exit()
    for ev in sys.argv[1:] or ["hl", "tin"]:
        print(ev, json.dumps(evaluate(ev), indent=1, default=float)[:3000])
