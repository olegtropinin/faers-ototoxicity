"""Paper figures: replication rates by signal class, and the case-reviewed candidates."""
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RES, FIG = Path("results"), Path("figures")
COL = {"lab": "#2b6cb0", "unl": "#0a8a5f", "crude": "#b7791f", "neg": "#8a8a8a"}


def replication_figure():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, ev in zip(axes, ["hl", "tin"]):
        s = json.load(open(RES / f"summary_{ev}.json"))
        c, p, t = s["replication_canada"], s.get("posthoc", {}), s["replication_temporal"]
        groups = [
            ("Canada, all reports", [c["adjusted_labelled"], c["adjusted_unlabelled"], c["crude_only_signals"],
                                     c["label_negative_non_signals"]]),
            ("Canada, no solicited\n(post hoc)", [p.get("repl_nostudy_adjusted_x_labelled"),
                                                  p.get("repl_nostudy_adjusted_x_unlabelled"),
                                                  p.get("repl_nostudy_crude_only"),
                                                  p.get("repl_nostudy_label_negative_non_signals")]),
            ("FAERS ≤2014 → ≥2015", [None, t["early_adjusted"], t["early_crude_only"],
                                     t["label_negative_non_signals"]]),
        ]
        labels = ["adjusted, labelled", "adjusted, unlabelled" if ev else "", "crude only", "label-negative non-signal"]
        colors = [COL["lab"], COL["unl"], COL["crude"], COL["neg"]]
        w = 0.2
        for gi, (gname, vals) in enumerate(groups):
            for vi, v in enumerate(vals):
                if not v:
                    continue
                x = gi + (vi - 1.5) * w
                if gname.startswith("FAERS") and vi == 1:
                    lab = "adjusted (all)"
                ax.bar(x, v["rate"], w, color=colors[vi], label=labels[vi] if gi == 0 else None)
                ax.errorbar(x, v["rate"], yerr=[[v["rate"] - v["lo"]], [v["hi"] - v["rate"]]], color="#333", lw=0.8,
                            capsize=2)
                ax.text(x, v["hi"] + 0.02, f"{v['k']}/{v['n']}", ha="center", fontsize=5.5, rotation=90)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([g[0] for g in groups], fontsize=7)
        ax.set_title("Hearing loss" if ev == "hl" else "Tinnitus", fontsize=9)
        ax.set_ylim(0, 1.15)
    axes[0].set_ylabel("share replicated (95% CI)")
    axes[0].legend(fontsize=6.5, loc="upper right", frameon=False)
    fig.text(0.5, -0.02, "Temporal panel: 'adjusted, unlabelled' bar shows all early adjusted signals.",
             ha="center", fontsize=6.5, color="#555")
    fig.tight_layout()
    fig.savefig(FIG / "replication.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def candidates_figure():
    rv = pd.read_csv(RES / "case_review.csv")
    rows = []
    for ev in ["hl", "tin"]:
        r = pd.read_csv(RES / f"{ev}_robust_posthoc.csv", index_col=0, low_memory=False)
        sub = rv[rv["event"] == ev].set_index("drug")
        for d in sub.index:
            if d in r.index:
                x = r.loc[d]
                rows.append(dict(drug=d, ev=ev, cat=sub.loc[d, "category"], f=x["faersnocax_mh_or"],
                                 flo=x["faersnocax_mh_lo"], fhi=x["faersnocax_mh_hi"], c=x["canadanostudy_ror"],
                                 clo=x["canadanostudy_ror_lo"], chi=x["canadanostudy_ror_hi"]))
    d = pd.DataFrame(rows).sort_values(["ev", "cat", "flo"], ascending=[True, False, True]).reset_index(drop=True)
    catname = {"A": "report cluster", "B": "ototoxic co-medication", "C": "stimulated reporting",
               "D": "indication / comorbidity", "E": "candidate"}
    fig, ax = plt.subplots(figsize=(6.6, 0.2 * len(d) + 1.4))
    y = np.arange(len(d))
    ax.errorbar(d["f"], y + 0.15, xerr=[d["f"] - d["flo"], d["fhi"] - d["f"]], fmt="o", ms=3, color="#0a8a5f",
                lw=0.9, label="FAERS adjusted MH OR")
    ax.errorbar(d["c"], y - 0.15, xerr=[d["c"] - d["clo"], d["chi"] - d["c"]], fmt="s", ms=3, color="#b7791f",
                lw=0.9, label="Canada (no solicited) ROR")
    lbl = [f"{r.drug.title()} — {'HL' if r.ev == 'hl' else 'tinnitus'} [{r.cat}]" for r in d.itertuples()]
    ax.set_yticks(y)
    ax.set_yticklabels(lbl, fontsize=6.3)
    for i, r in enumerate(d.itertuples()):
        if r.cat == "E":
            ax.get_yticklabels()[i].set_fontweight("bold")
    ax.set_xscale("log")
    ax.axvline(1, color="#999", lw=0.7)
    ax.set_xlabel("odds ratio (95% CI), log scale", fontsize=8)
    ax.legend(fontsize=6.5, loc="lower right")
    ax.set_title("Robust signals without US label mention (case-review category in brackets)\n"
                 + "; ".join(f"{k} {v}" for k, v in list(catname.items())[:3]) + ";\n"
                 + "; ".join(f"{k} {v}" for k, v in list(catname.items())[3:]), fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "candidates.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    replication_figure()
    candidates_figure()
