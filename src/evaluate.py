"""Evaluate crude vs adjusted signal detection against the label reference set, the
Canadian replication and the temporal replication; list robust signals; draw figures.

Reads results/<event>_master.csv; writes results/<event>_*.csv, results/summary_<event>.json
and figures/*.png.
"""
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RES, FIG = Path("results"), Path("figures")
FIG.mkdir(exist_ok=True)
rng = np.random.default_rng(20261026)
MIN_REPORTS = 1000


def crude_sig(m, p):
    return (m[f"{p}_a"] >= 3) & (m[f"{p}_ror_lo"] > 1) & (m[f"{p}_ic025"] > 0)


def adj_sig(m, p):
    return (m[f"{p}_a_adj"] >= 3) & (m[f"{p}_mh_lo"] > 1)


def testable(m, p):
    return np.maximum(m[f"{p}_a"].fillna(0), m[f"{p}_E"].fillna(0)) >= 3


def replicated(m, p):
    return (m[f"{p}_a"] >= 3) & (m[f"{p}_ror_lo"] > 1)


def auc(score, y):
    """Mann-Whitney AUC; ties count 1/2."""
    pos, neg = score[y], score[~y]
    r = pd.Series(np.concatenate([pos, neg])).rank().values
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def wilson(k, n):
    if n == 0:
        return (np.nan, np.nan, np.nan)
    p, z = k / n, 1.96
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (p, c - h, c + h)


def rate(mask_den, mask_num):
    n, k = int(mask_den.sum()), int((mask_den & mask_num).sum())
    p, lo, hi = wilson(k, n)
    return {"k": k, "n": n, "rate": p, "lo": lo, "hi": hi}


def forest(robust, ev, top=30):
    """Adjusted FAERS OR next to the Canadian ROR for robust signals (log scale)."""
    lab = "hl_any" if ev == "hl" else "tin_any"
    r = robust.head(top).iloc[::-1]
    if r.empty:
        return
    fig, ax = plt.subplots(figsize=(6.2, 0.26 * len(r) + 1.2))
    y = np.arange(len(r))
    ax.errorbar(r["faers_mh_or"], y + 0.15, xerr=[r["faers_mh_or"] - r["faers_mh_lo"], r["faers_mh_hi"] - r["faers_mh_or"]],
                fmt="o", ms=3, color="#0a6", lw=1, label="FAERS, adjusted MH OR")
    ax.errorbar(r["canada_ror"], y - 0.15, xerr=[r["canada_ror"] - r["canada_ror_lo"], r["canada_ror_hi"] - r["canada_ror"]],
                fmt="s", ms=3, color="#c60", lw=1, label="Canada Vigilance, ROR")
    names = [f"{n.title()}{'' if l is True else ' *'}" for n, l in zip(r.index, r[lab])]
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xscale("log")
    ax.axvline(1, color="#999", lw=0.8)
    ax.set_xlabel("odds ratio (95% CI), log scale")
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title(f"Robust {'hearing-loss' if ev == 'hl' else 'tinnitus'} signals (* = no US label mention)", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / f"forest_{ev}.png", dpi=200)
    plt.close(fig)


def evaluate(ev):
    m = pd.read_csv(RES / f"{ev}_master.csv", index_col="ingredient")
    ref = m[f"ref_{ev}"] if f"ref_{ev}" in m else m["ref_hl"]
    out = {}

    # --- 1. reference set ---
    base = (m["faers_n"] >= MIN_REPORTS) & ref.isin(["pos", "neg"])
    r = m[base]
    y = (ref[base] == "pos").values
    floor = -20.0
    scores = {
        "crude_IC025": r["faers_ic025"].fillna(floor).values,
        "crude_logROR025": np.log(r["faers_ror_lo"]).replace([-np.inf, np.inf], floor).fillna(floor).values,
        "adjusted_logMH025": np.log(r["faers_mh_lo"]).replace([-np.inf, np.inf], floor).fillna(floor).values,
    }
    aucs = {k: auc(v, y) for k, v in scores.items()}
    boot = []
    idx = np.arange(len(y))
    for _ in range(2000):
        b = rng.choice(idx, len(idx))
        if y[b].all() or (~y[b]).all():
            continue
        boot.append(auc(scores["adjusted_logMH025"][b], y[b]) - auc(scores["crude_IC025"][b], y[b]))
    cs, as_ = crude_sig(r, "faers").values, adj_sig(r, "faers").values
    out["reference"] = {
        "n_pos": int(y.sum()), "n_neg": int((~y).sum()), "min_reports": MIN_REPORTS,
        "auc": aucs, "auc_diff_adj_minus_crude": float(np.mean(boot)),
        "auc_diff_ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "crude_sens": float(cs[y].mean()), "crude_spec": float(1 - cs[~y].mean()),
        "adj_sens": float(as_[y].mean()), "adj_spec": float(1 - as_[~y].mean()),
    }
    # ROC figure
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    for (k, v), c in zip(scores.items(), ["#888", "#bbb", "#0a6"]):
        o = np.argsort(-v)
        tpr = np.concatenate([[0], np.cumsum(y[o]) / y.sum()])
        fpr = np.concatenate([[0], np.cumsum(~y[o]) / (~y).sum()])
        ax.plot(fpr, tpr, color=c, lw=2 if "adj" in k else 1.2, label=f"{k} (AUC {aucs[k]:.2f})")
    ax.plot([0, 1], [0, 1], ":", color="#ccc")
    ax.set_xlabel("1 - specificity (label-negative ingredients)")
    ax.set_ylabel("sensitivity (label-positive ingredients)")
    ax.set_title(f"{'Hearing loss' if ev == 'hl' else 'Tinnitus'}: label reference set\n"
                 f"{int(y.sum())} positive, {int((~y).sum())} negative ingredients", fontsize=9)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / f"roc_{ev}.png", dpi=200)
    plt.close(fig)

    # --- 2. signal flow and external replication (FAERS without Canada -> Canada) ---
    m["crude"] = crude_sig(m, "faersnoca")
    m["adjusted"] = adj_sig(m, "faersnoca")
    m["ca_testable"] = testable(m, "canada")
    m["ca_repl"] = replicated(m, "canada")
    m["label_mention"] = m["hl_any"] if ev == "hl" else m["tin_any"]
    unl = m["label_mention"] != True  # noqa: E712  (NaN = no single-ingredient US label found)
    neg = ref == "neg"
    out["flow"] = {"crude_signals": int(m["crude"].sum()), "adjusted_signals": int(m["adjusted"].sum()),
                   "crude_not_adjusted": int((m["crude"] & ~m["adjusted"]).sum()),
                   "adjusted_not_crude": int((m["adjusted"] & ~m["crude"]).sum())}
    t = m["ca_testable"]
    out["replication_canada"] = {
        "adjusted_signals": rate(t & m["adjusted"], m["ca_repl"]),
        "adjusted_unlabelled": rate(t & m["adjusted"] & unl, m["ca_repl"]),
        "adjusted_labelled": rate(t & m["adjusted"] & ~unl, m["ca_repl"]),
        "crude_only_signals": rate(t & m["crude"] & ~m["adjusted"], m["ca_repl"]),
        "label_negative_non_signals": rate(t & neg & ~m["crude"], m["ca_repl"]),
        "label_negative_all": rate(t & neg & (m["faersnoca_n"] >= MIN_REPORTS), m["ca_repl"]),
    }
    td, rd = testable(m, "canadadirect"), replicated(m, "canadadirect")
    out["replication_canada_direct_only"] = {
        "adjusted_signals": rate(td & m["adjusted"], rd),
        "crude_only_signals": rate(td & m["crude"] & ~m["adjusted"], rd),
        "label_negative_non_signals": rate(td & neg & ~m["crude"], rd),
    }
    # --- 2b. post-hoc: extended disease-indication exclusion, Canada without solicited reports ---
    if "faersnocax_mh_lo" in m and "canadanostudy_a" in m:
        m["adjusted_x"] = adj_sig(m, "faersnocax")
        tn, rn = testable(m, "canadanostudy"), replicated(m, "canadanostudy")
        out["posthoc"] = {
            "adjusted_signals_extended": int(m["adjusted_x"].sum()),
            "primary_signals_lost_by_extension": sorted(m.index[m["adjusted"] & ~m["adjusted_x"]].tolist()),
            "repl_nostudy_adjusted_x": rate(tn & m["adjusted_x"], rn),
            "repl_nostudy_adjusted_x_unlabelled": rate(tn & m["adjusted_x"] & unl, rn),
            "repl_nostudy_adjusted_x_labelled": rate(tn & m["adjusted_x"] & ~unl, rn),
            "repl_nostudy_crude_only": rate(tn & m["crude"] & ~m["adjusted"], rn),
            "repl_nostudy_label_negative_non_signals": rate(tn & neg & ~m["crude"], rn),
        }
        m["ns_testable"], m["ns_repl"] = tn, rn
        strict = m[m["adjusted_x"] & rn]
        strict.to_csv(RES / f"{ev}_robust_posthoc.csv")
        out["posthoc"]["robust_strict"] = len(strict)
        out["posthoc"]["robust_strict_unlabelled"] = int((strict["label_mention"] != True).sum())  # noqa: E712

    # --- 3. temporal replication (<= 2014 -> >= 2015) ---
    m["early_crude"], m["early_adj"] = crude_sig(m, "early"), adj_sig(m, "early")
    lt, lr = testable(m, "late"), replicated(m, "late")
    out["replication_temporal"] = {
        "early_adjusted": rate(lt & m["early_adj"], lr),
        "early_crude_only": rate(lt & m["early_crude"] & ~m["early_adj"], lr),
        "label_negative_non_signals": rate(lt & neg & ~m["early_crude"], lr),
    }

    # --- 4. robust signals ---
    cols = ["faers_n", "faers_a", "faers_ror", "faers_ror_lo", "faers_a_adj", "faers_mh_or", "faers_mh_lo",
            "faers_mh_hi", "faersnoca_a_adj", "faersnoca_mh_or", "faersnoca_mh_lo", "canada_n", "canada_a",
            "canada_E", "canada_ror", "canada_ror_lo", "canada_ror_hi", "canada_a_adj", "canada_mh_or",
            "canada_mh_lo", "known_ototoxic" if "known_ototoxic" in m else "faers_known_ototoxic",
            "n_labels", "hl_any", "tin_any", f"ref_{ev}", "crude", "adjusted", "ca_testable", "ca_repl"]
    cols = [c for c in cols if c in m]
    robust = m[m["adjusted"] & m["ca_repl"]][cols].sort_values("faers_mh_lo", ascending=False)
    robust.to_csv(RES / f"{ev}_robust_signals.csv")
    robust[robust["hl_any" if ev == "hl" else "tin_any"] != True].to_csv(  # noqa: E712
        RES / f"{ev}_robust_unlabelled.csv")
    m[m["crude"] | m["adjusted"]][cols].to_csv(RES / f"{ev}_all_signals.csv")
    forest(robust, ev)
    out["robust"] = {"all": len(robust),
                     "unlabelled": int((robust["hl_any" if ev == "hl" else "tin_any"] != True).sum())}  # noqa: E712
    json.dump(out, open(RES / f"summary_{ev}.json", "w"), indent=1, default=float)
    return out


def prior_screens():
    """How the signals reported by earlier FAERS screens fare here (crude -> adjusted -> Canada)."""
    rows = []
    for f in sorted(Path("data/raw/prior").glob("*.csv")):
        p = pd.read_csv(f)
        if p.empty or "drug" not in p:
            continue
        for scope, ev in (("hearing_loss", "hl"), ("hearing_any", "hl"), ("tinnitus", "tin")):
            sub = p[p["event_scope"] == scope]
            if sub.empty:
                continue
            m = pd.read_csv(RES / f"{ev}_master.csv", index_col="ingredient")
            names = sub["drug"].str.upper().str.strip().drop_duplicates()
            found = names[names.isin(m.index)]
            mm = m.loc[found]
            crude, adj = crude_sig(mm, "faers"), adj_sig(mm, "faers")
            t, r = testable(mm, "canada"), replicated(mm, "canada")
            rows.append({"study": f.stem, "scope": scope, "reported": len(names), "matched": len(found),
                         "crude_here": int(crude.sum()), "adjusted_here": int(adj.sum()),
                         "adjusted_and_testable": int((adj & t).sum()), "adjusted_and_replicated": int((adj & r).sum()),
                         "lost_on_adjustment": ";".join(sorted(found[(crude & ~adj).values])),
                         "unmatched": ";".join(sorted(set(names) - set(found)))})
    out = pd.DataFrame(rows)
    out.to_csv(RES / "prior_screens.csv", index=False)
    return out


if __name__ == "__main__":
    if sys.argv[1:] == ["prior"]:
        print(prior_screens().to_string())
        sys.exit()
    for ev in sys.argv[1:] or ["hl", "tin"]:
        print(ev, json.dumps(evaluate(ev), indent=1, default=float))
