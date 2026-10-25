"""Paper figures: power-calibrated replication (Figure 1) and case-reviewed robust unlabelled signals (Figure 2)."""
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RES, FIG = Path("results"), Path("figures")
COL = {"lab": "#2b6cb0", "unl": "#0a8a5f", "neg": "#b7791f"}
SETS = [("Primary\n(crude Canada,\nall reports)", None), ("Post hoc\nadjusted both\n(exclusion)", "x"),
        ("...co-medication\nas stratum", "xstrat"), ("...strong\nototoxic list", "xstrong"),
        ("...without\n>=15-drug reports", "xnomega")]


def replication_figure():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), sharey=True)
    for ax, ev in zip(axes, ["hl", "tin"]):
        s = json.load(open(RES / f"summary_{ev}.json"))
        w = 0.26
        for gi, (label, v) in enumerate(SETS):
            if v is None:
                c = s["calibrated_canada"]
                vals = [c["adjusted_labelled"], c["adjusted_unlabelled"], c["adjusted_label_negative"]]
            else:
                p = s["posthoc"][v]
                vals = [p["calibrated_labelled"], p["calibrated_unlabelled"], None]
            for vi, (d, col) in enumerate(zip(vals, [COL["lab"], COL["unl"], COL["neg"]])):
                if not d:
                    continue
                x = gi + (vi - 1) * w
                ax.bar(x, d["obs_over_exp"], w, color=col,
                       label=["labelled signals", "unlabelled signals", "label-negative signals"][vi] if gi == 0 else None)
                ax.text(x, d["obs_over_exp"] + 0.03, f"{d['observed']}/{d['expected_if_true']:.0f}", ha="center",
                        fontsize=5.8, rotation=90, va="bottom")
        ax.axhline(1, color="#555", lw=0.8, ls="--")
        ax.set_xticks(range(len(SETS)))
        ax.set_xticklabels([s_[0] for s_ in SETS], fontsize=6.3)
        ax.set_title("Hearing loss" if ev == "hl" else "Tinnitus", fontsize=9)
        ax.set_ylim(0, 1.45)
    axes[0].set_ylabel("observed / expected replications\n(expected if the FAERS signal were true)", fontsize=8)
    axes[0].legend(fontsize=6.5, loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "replication.png", dpi=220)
    plt.close(fig)


def candidates_figure():
    rv = pd.read_csv(RES / "case_review.csv")
    rows = []
    for ev in ["hl", "tin"]:
        r = pd.read_csv(RES / f"{ev}_robust_x.csv", index_col=0, low_memory=False)
        for _, c in rv[rv["event"] == ev].iterrows():
            x = r.loc[c.drug]
            rows.append(dict(drug=c.drug, ev=ev, cat=c.category, f=x["faersnocax_mh_or"], flo=x["faersnocax_mh_lo"],
                             fhi=x["faersnocax_mh_hi"], c=x["canadax_mh_or"], clo=x["canadax_mh_lo"],
                             chi=x["canadax_mh_hi"]))
    d = pd.DataFrame(rows).sort_values(["ev", "cat", "flo"], ascending=[True, False, True]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(6.6, 0.24 * len(d) + 1.5))
    y = np.arange(len(d))
    ax.errorbar(d["f"], y + 0.15, xerr=[d["f"] - d["flo"], d["fhi"] - d["f"]], fmt="o", ms=3, color="#0a8a5f",
                lw=0.9, label="FAERS adjusted MH OR")
    chi = d["chi"].replace(np.inf, np.nan).fillna(d["c"] * 50)
    ax.errorbar(d["c"], y - 0.15, xerr=[d["c"] - d["clo"], chi - d["c"]], fmt="s", ms=3, color="#b7791f",
                lw=0.9, label="Canada adjusted MH OR (no solicited reports)")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.drug.title()} - {'hearing loss' if r.ev == 'hl' else 'tinnitus'} [{r.cat}]"
                        for r in d.itertuples()], fontsize=6.6)
    for i, r in enumerate(d.itertuples()):
        if r.cat == "E":
            ax.get_yticklabels()[i].set_fontweight("bold")
    ax.set_xscale("log")
    ax.axvline(1, color="#999", lw=0.7)
    ax.set_xlabel("odds ratio (95% CI), log scale", fontsize=8)
    ax.legend(fontsize=6.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "candidates.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    replication_figure()
    candidates_figure()
