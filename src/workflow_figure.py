"""Figure 1: study workflow, pre-specified analyses in blue and post-hoc analyses in orange.

Numbers come from paper/generated/workflow.json, written by paper/fill.py; run fill.py first.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

w = json.load(open("paper/generated/workflow.json"))
PRE, POST = ("#e8f0fb", "#2f5f9e"), ("#fdf0e2", "#b7652a")


def m(x):
    return f"{x / 1e6:.1f} million"


steps = [
    (PRE, f"FAERS 2004-2026: {m(w['faers_reports'])} deduplicated reports\n"
          f"Canada Vigilance: {m(w['canada_reports'])} reports"),
    (PRE, f"Hearing-loss signals, FAERS without Canadian reports: crude {w['crude_signals']} vs adjusted {w['adjusted_signals']}\n"
          "(adjusted: ear indications, ototoxic co-medication and lawyer reports excluded; stratified)"),
    (PRE, f"Agreement with US labels: AUC {w['auc_crude']:.3f} crude vs {w['auc_adjusted']:.3f} adjusted"),
    (PRE, f"Replication in Canada Vigilance (rule-based): labelled {w['repl_labelled']}, "
          f"unlabelled {w['repl_unlabelled']}"),
    (POST, f"Power-calibrated replication: observed/expected {w['oe_labelled']:.2f} labelled "
           f"vs {w['oe_unlabelled']:.2f} unlabelled"),
    (POST, "Robustness: extended indication filter, Canada without solicited reports,\n"
           "both databases adjusted, four ways of handling co-medication"),
    (POST, f"{w['robust_unlabelled']} robust unlabelled signals: case-level review of report provenance\n"
           "and active-comparator check within the indication"),
    (POST, f"{w['remaining']} remaining signals ({w['remaining_min']}\u2013{w['remaining_max']} under alternative choices): "
           "hypotheses, not causal evidence"),
]

fig, ax = plt.subplots(figsize=(8, 7.2))
ax.axis("off")
n = len(steps)
gap = 1 / n
for i, ((fc, ec), text) in enumerate(steps):
    y = 1 - (i + 0.5) * gap
    ax.text(0.5, y, text, ha="center", va="center", fontsize=8.6, transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.55", fc=fc, ec=ec, lw=1.1))
    if i < n - 1:
        ax.annotate("", xy=(0.5, y - 0.68 * gap), xytext=(0.5, y - 0.32 * gap),
                    xycoords="axes fraction", arrowprops=dict(arrowstyle="->", color="#555"))
for (fc, ec), label in ((PRE, "Pre-specified"), (POST, "Post hoc")):
    ax.plot([], [], "s", ms=10, mfc=fc, mec=ec, label=label)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=2, frameon=False, fontsize=9)
fig.tight_layout()
Path("figures").mkdir(exist_ok=True)
fig.savefig("figures/workflow.png", dpi=220, bbox_inches="tight")
print("figures/workflow.png")
