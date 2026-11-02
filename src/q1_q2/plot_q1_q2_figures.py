#!/usr/bin/env python3
"""Create compact Q1/Q2 figures for the curiosity project."""

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_Q1 = REPO_ROOT / "outputs" / "llm_judged.csv"
DEFAULT_Q2 = REPO_ROOT / "outputs" / "matched_q2"
DEFAULT_OUT = REPO_ROOT / "outputs" / "figures"

STATUS_ORDER = ["curiosity", "not_curiosity", "unclear", "problematic"]
STATUS_COLORS = {
    "curiosity": "#2A9D8F",
    "not_curiosity": "#6C757D",
    "unclear": "#E9C46A",
    "problematic": "#D62828",
}
TYPE_COLORS = {
    "information_gap_curiosity": "#2A9D8F",
    "causal_explanatory": "#457B9D",
    "exploratory_broadening": "#F4A261",
    "reflective_metacognitive": "#8E6C8A",
    "perspective_taking": "#E76F51",
    "counterfactual_hypothetical": "#5E548E",
}
OUTCOME_LABELS = {
    "next_user_has_question": "User asks any question",
    "next_user_question_count": "Question count",
    "next_user_exploratory_question_count": "Exploratory questions",
    "next_user_epistemic_marker_count": "Epistemic markers",
    "next_user_word_count": "Word count",
    "next_user_epistemic_markers_per_100w": "Epistemic markers / 100w",
    "next_user_exploratory_questions_per_100w": "Exploratory questions / 100w",
}


def style():
    plt.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def save(fig, outdir: Path, name: str):
    fig.tight_layout()
    fig.savefig(outdir / f"{name}.png", bbox_inches="tight")
    fig.savefig(outdir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def add_bar_labels(ax, values, total=None, fmt="count_pct"):
    xmax = max(values) if values else 1
    for patch, val in zip(ax.patches, values):
        if fmt == "count_pct" and total:
            label = f"{int(val):,} ({val / total * 100:.1f}%)"
        elif fmt == "pct":
            label = f"{val:.1f}%"
        else:
            label = f"{val:,.2f}" if isinstance(val, float) else f"{int(val):,}"
        ax.text(patch.get_width() + xmax * 0.012, patch.get_y() + patch.get_height() / 2, label, va="center")


def plot_q1_status(df: pd.DataFrame, outdir: Path):
    counts = df["llm_status"].value_counts().reindex(STATUS_ORDER).fillna(0).astype(int)
    total = int(counts.sum())
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    colors = [STATUS_COLORS[s] for s in counts.index]
    ax.barh(counts.index[::-1], counts.values[::-1], color=colors[::-1])
    ax.set_title("Q1: Top-level labels for assistant question events")
    ax.set_xlabel("Number of assistant-question events")
    add_bar_labels(ax, list(counts.values[::-1]), total=total)
    ax.set_xlim(0, counts.max() * 1.28)
    ax.grid(axis="x", alpha=0.2)
    save(fig, outdir, "q1_top_level_status")
    counts.rename("count").to_csv(outdir / "q1_top_level_status_counts.csv")


def plot_q1_types_by_status(df: pd.DataFrame, outdir: Path, top_n: int):
    type_counts = df["llm_question_type"].value_counts().head(top_n)
    top_types = list(type_counts.index)
    tab = pd.crosstab(df["llm_question_type"], df["llm_status"]).reindex(top_types).fillna(0)
    for status in STATUS_ORDER:
        if status not in tab.columns:
            tab[status] = 0
    tab = tab[STATUS_ORDER]
    tab = tab.iloc[::-1]

    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    left = pd.Series(0, index=tab.index, dtype=float)
    for status in STATUS_ORDER:
        vals = tab[status]
        ax.barh(tab.index, vals, left=left, label=status, color=STATUS_COLORS[status])
        left += vals
    ax.set_title(f"Q1: Question types by top-level label (top {top_n})")
    ax.set_xlabel("Number of assistant-question events")
    ax.legend(ncol=4, loc="lower right", frameon=False)
    ax.grid(axis="x", alpha=0.2)
    save(fig, outdir, "q1_question_types_by_status")
    tab.to_csv(outdir / "q1_question_types_by_status_counts.csv")


def plot_q1_curiosity_forms(df: pd.DataFrame, outdir: Path):
    cur = df[df["llm_status"] == "curiosity"].copy()
    curiosity_order = [
        "information_gap_curiosity",
        "causal_explanatory",
        "exploratory_broadening",
        "reflective_metacognitive",
        "perspective_taking",
        "counterfactual_hypothetical",
    ]
    counts = cur["llm_question_type"].value_counts().reindex(curiosity_order).fillna(0).astype(int)
    counts = counts[counts > 0]
    total = int(counts.sum())

    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    colors = [TYPE_COLORS.get(t, "#888888") for t in counts.index[::-1]]
    ax.barh(counts.index[::-1], counts.values[::-1], color=colors)
    ax.set_title("Q1: Forms of curiosity-oriented questions")
    ax.set_xlabel("Number of curiosity events")
    add_bar_labels(ax, list(counts.values[::-1]), total=total)
    ax.set_xlim(0, counts.max() * 1.32)
    ax.grid(axis="x", alpha=0.2)
    save(fig, outdir, "q1_curiosity_forms")
    counts.rename("count").to_csv(outdir / "q1_curiosity_forms_counts.csv")


def plot_q2_effects(q2dir: Path, outdir: Path):
    matched = pd.read_csv(q2dir / "matched_summary.csv")
    reg = pd.read_csv(q2dir / "regression_summary.csv")
    keep = [
        "next_user_has_question",
        "next_user_question_count",
        "next_user_exploratory_question_count",
        "next_user_epistemic_marker_count",
        "next_user_word_count",
        "next_user_epistemic_markers_per_100w",
        "next_user_exploratory_questions_per_100w",
    ]
    matched = matched.set_index("outcome").loc[keep].reset_index()
    reg = reg.set_index("outcome").loc[keep].reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.0), sharey=True)
    y = list(range(len(keep)))[::-1]
    labels = [OUTCOME_LABELS[o] for o in keep][::-1]

    m = matched.iloc[::-1]
    axes[0].errorbar(
        m["difference"], y,
        xerr=[m["difference"] - m["bootstrap_ci_low"], m["bootstrap_ci_high"] - m["difference"]],
        fmt="o", color="#2A9D8F", ecolor="#8ABDB5", capsize=3,
    )
    axes[0].axvline(0, color="#333333", linewidth=0.8)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    axes[0].set_title("Matched-pair difference")
    axes[0].set_xlabel("Curiosity - matched control")
    axes[0].grid(axis="x", alpha=0.2)

    r = reg.iloc[::-1]
    axes[1].errorbar(
        r["coef_treatment"], y,
        xerr=[r["coef_treatment"] - r["ci_low"], r["ci_high"] - r["coef_treatment"]],
        fmt="o", color="#457B9D", ecolor="#A8C4D5", capsize=3,
    )
    axes[1].axvline(0, color="#333333", linewidth=0.8)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(labels)
    axes[1].set_title("Adjusted regression coefficient")
    axes[1].set_xlabel("Coefficient on curiosity")
    axes[1].grid(axis="x", alpha=0.2)

    fig.suptitle("Q2: Association between model curiosity and next-user epistemic engagement", y=1.02, fontsize=13)
    save(fig, outdir, "q2_matched_and_regression_effects")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q1-csv", type=Path, default=DEFAULT_Q1)
    ap.add_argument("--q2-dir", type=Path, default=DEFAULT_Q2)
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--top-types", type=int, default=12)
    args = ap.parse_args()

    style()
    args.outdir.mkdir(parents=True, exist_ok=True)
    usecols = ["llm_status", "llm_question_type", "llm_keep", "llm_artifact_type"]
    df = pd.read_csv(args.q1_csv, usecols=usecols)

    plot_q1_status(df, args.outdir)
    plot_q1_types_by_status(df, args.outdir, args.top_types)
    plot_q1_curiosity_forms(df, args.outdir)
    plot_q2_effects(args.q2_dir, args.outdir)

    print("Wrote figures to", args.outdir)
    for path in sorted(args.outdir.glob("*.png")):
        print(path)


if __name__ == "__main__":
    main()
