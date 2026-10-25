"""Supplementary Figure S1: report flow for FAERS and Canada Vigilance (READUS-PV item 8a)."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

fb = json.load(open("data/work/faers/build_log.json"))
cb = json.load(open("data/work/canada/build_log.json"))
pop = json.load(open("results/population.json"))


def n(x):
    return f"{x:,}"


faers = [
    ("FAERS / AERS report versions, 2004Q1-2026Q2 (90 quarters)", fb["demo_rows"]),
    ("Latest version of each case", fb["cases_after_latest_version"]),
    (f"After FDA deletion lists (-{n(fb['cases_after_latest_version'] - fb['cases_after_deletions'])})",
     fb["cases_after_deletions"]),
    ("With >= 1 reaction and >= 1 mapped ingredient", fb["cases_with_reaction_and_mapped_drug"]),
    (f"After id collisions (-{n(fb['report_ids_under_two_cases'])}) and content duplicates "
     f"(-{n(fb['content_duplicates_removed'])})", fb["final_reports"]),
    (f"Without Canadian reports (replication analyses)", pop["faersnoca_hl"]["reports"]),
    ("Primary adjusted population (from all 19.9M): no ear indication, no listed ototoxic drug, no lawyer report",
     pop["faers_hl"]["adj_reports"]),
]
canada = [
    ("Canada Vigilance reports, 1965-2026-06", cb["reports_raw"]),
    (f"After duplicates (-{n(cb['duplicates_dropped'])})", cb["reports_after_dedup"]),
    ("With >= 1 reaction and >= 1 mapped ingredient", cb["final_reports"]),
    ("Without solicited (Study) reports (post hoc)", pop["canadanostudy_hl"]["reports"]),
    ("Adjusted population (post hoc, extended indication filter)", pop["canadax_hl"]["adj_reports"]),
]
fig, axes = plt.subplots(1, 2, figsize=(11, 6.2))
for ax, steps, title in ((axes[0], faers, "FAERS"), (axes[1], canada, "Canada Vigilance")):
    ax.axis("off")
    ax.set_title(title, fontsize=11, fontweight="bold")
    y = 0.95
    for i, (label, val) in enumerate(steps):
        ax.text(0.5, y, f"{label}\n{n(val)} reports", ha="center", va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.4", fc="#f4f6f8", ec="#888"), transform=ax.transAxes, wrap=True)
        if i < len(steps) - 1:
            ax.annotate("", xy=(0.5, y - 0.105), xytext=(0.5, y - 0.075), xycoords="axes fraction",
                        arrowprops=dict(arrowstyle="->", color="#555"))
        y -= 0.135
fig.tight_layout()
Path("figures").mkdir(exist_ok=True)
fig.savefig("figures/flow.png", dpi=200)
print("figures/flow.png")
