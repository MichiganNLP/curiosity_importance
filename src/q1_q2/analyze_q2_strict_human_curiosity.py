#!/usr/bin/env python3
"""Analyze broad Q2 engagement outcomes vs strict LLM-judged human curiosity.

Inputs are llm_judge_human_curiosity.py outputs, where each matched pair has
one treatment arm and one control arm. The treatment arm follows an assistant
curiosity question; the control arm follows a matched non-curiosity question.
"""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "outputs"
DEFAULT_INPUTS = {
    "WildChat": OUTPUT_DIR / "wildchat_q2_human_curiosity_judged.jsonl",
    "OASST1": OUTPUT_DIR / "oasst1_q2_human_curiosity_judged.jsonl",
    "ShareGPT": OUTPUT_DIR / "sharegpt_q2_human_curiosity_judged.jsonl",
    "LMSYS": OUTPUT_DIR / "lmsys_q2_human_curiosity_judged.jsonl",
}

RULE_METRICS = [
    "rule_next_user_has_question",
    "rule_next_user_question_count",
    "rule_next_user_exploratory_question_count",
    "rule_next_user_epistemic_marker_count",
    "rule_next_user_word_count",
]

STRICT_FIELDS = [
    "strict_human_curiosity",
    "strict_human_curiosity_moderate_strong",
    "strict_not_task_or_answer_only",
]


def load_jsonl(path):
    with Path(path).open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def to_float(x):
    try:
        if x in {"", None}:
            return 0.0
        return float(x)
    except Exception:
        return 0.0


def enrich(row):
    row = dict(row)
    row["strict_human_curiosity"] = int(row.get("status") == "human_curiosity")
    row["strict_human_curiosity_moderate_strong"] = int(
        row.get("status") == "human_curiosity" and int(row.get("curiosity_intensity") or 0) >= 2
    )
    row["strict_not_task_or_answer_only"] = int(
        row.get("status") == "human_curiosity"
        and not bool(row.get("is_task_continuation_only"))
        and not bool(row.get("is_answering_assistant_question_only"))
    )
    for m in RULE_METRICS:
        row[m] = to_float(row.get(m))
    return row


def paired_diffs(rows, outcome):
    by_pair = defaultdict(dict)
    for r in rows:
        by_pair[str(r.get("pair_id"))][r.get("arm")] = r
    diffs = []
    tvals = []
    cvals = []
    for pair_id, arms in by_pair.items():
        if "treatment" not in arms or "control" not in arms:
            continue
        t = to_float(arms["treatment"].get(outcome))
        c = to_float(arms["control"].get(outcome))
        tvals.append(t)
        cvals.append(c)
        diffs.append(t - c)
    return tvals, cvals, diffs


def summarize_outcome(dataset, rows, outcome):
    tvals, cvals, diffs = paired_diffs(rows, outcome)
    if not diffs:
        return None
    return {
        "dataset": dataset,
        "outcome": outcome,
        "n_pairs": len(diffs),
        "treatment_mean": mean(tvals),
        "control_mean": mean(cvals),
        "diff_t_minus_c": mean(diffs),
    }


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", type=Path, default=OUTPUT_DIR / "q2_strict_human_curiosity")
    ap.add_argument("--n-examples", type=int, default=12)
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    summary_rows = []
    type_rows = []
    function_rows = []
    example_rows = []

    for dataset, path in DEFAULT_INPUTS.items():
        p = Path(path)
        if not p.exists():
            print(f"Missing {dataset}: {p}")
            continue
        rows = [enrich(r) for r in load_jsonl(p)]
        all_rows.extend(rows)
        for outcome in RULE_METRICS + STRICT_FIELDS:
            s = summarize_outcome(dataset, rows, outcome)
            if s:
                summary_rows.append(s)

        for arm in ["treatment", "control"]:
            subset = [r for r in rows if r.get("arm") == arm]
            total = len(subset) or 1
            for status, n in Counter(r.get("status") for r in subset).most_common():
                type_rows.append({"dataset": dataset, "arm": arm, "field": "status", "label": status, "n": n, "pct": 100*n/total})
            curious = [r for r in subset if r.get("status") == "human_curiosity"]
            ctotal = len(curious) or 1
            for typ, n in Counter(r.get("curiosity_type") for r in curious).most_common():
                type_rows.append({"dataset": dataset, "arm": arm, "field": "curiosity_type_given_curious", "label": typ, "n": n, "pct": 100*n/ctotal})
            noncur = [r for r in subset if r.get("status") == "not_human_curiosity"]
            ntotal = len(noncur) or 1
            for fn, n in Counter(r.get("non_curiosity_function") for r in noncur).most_common():
                function_rows.append({"dataset": dataset, "arm": arm, "non_curiosity_function": fn, "n": n, "pct": 100*n/ntotal})

        # Examples where rule-based question/epistemic proxies are positive but strict judge says not curiosity.
        candidates = [
            r for r in rows
            if r.get("arm") == "treatment"
            and r.get("status") == "not_human_curiosity"
            and (r.get("rule_next_user_question_count", 0) > 0 or r.get("rule_next_user_epistemic_marker_count", 0) > 0)
        ]
        for r in candidates[: args.n_examples]:
            example_rows.append({
                "dataset": dataset,
                "pair_id": r.get("pair_id"),
                "assistant_question_type": r.get("assistant_question_type"),
                "assistant_question_text": r.get("assistant_question_text"),
                "next_user_text": r.get("next_user_text"),
                "status": r.get("status"),
                "non_curiosity_function": r.get("non_curiosity_function"),
                "evidence": r.get("evidence"),
                "rule_next_user_question_count": r.get("rule_next_user_question_count"),
                "rule_next_user_epistemic_marker_count": r.get("rule_next_user_epistemic_marker_count"),
            })

    write_csv(args.outdir / "q2_broad_vs_strict_summary.csv", summary_rows, [
        "dataset", "outcome", "n_pairs", "treatment_mean", "control_mean", "diff_t_minus_c"
    ])
    write_csv(args.outdir / "q2_strict_status_and_type_distribution.csv", type_rows, [
        "dataset", "arm", "field", "label", "n", "pct"
    ])
    write_csv(args.outdir / "q2_non_curiosity_function_distribution.csv", function_rows, [
        "dataset", "arm", "non_curiosity_function", "n", "pct"
    ])
    write_csv(args.outdir / "q2_rule_positive_but_not_strict_examples.csv", example_rows, [
        "dataset", "pair_id", "assistant_question_type", "assistant_question_text", "next_user_text",
        "status", "non_curiosity_function", "evidence", "rule_next_user_question_count",
        "rule_next_user_epistemic_marker_count",
    ])

    # Human-readable Markdown summary.
    lines = ["# Q2 Broad vs Strict Human Curiosity", ""]
    lines.append("Broad outcomes are rule-based next-user proxies. Strict outcomes use the LLM judge and exclude task continuation, repair, and merely answering the assistant question.")
    lines.append("")
    lines.append("## Main Paired Effects")
    lines.append("")
    lines.append("| Dataset | Outcome | Treatment | Control | Diff |")
    lines.append("|---|---:|---:|---:|---:|")
    for r in summary_rows:
        if r["outcome"] in {"rule_next_user_question_count", "rule_next_user_epistemic_marker_count", "strict_human_curiosity", "strict_human_curiosity_moderate_strong", "strict_not_task_or_answer_only"}:
            lines.append(f"| {r['dataset']} | `{r['outcome']}` | {r['treatment_mean']:.3f} | {r['control_mean']:.3f} | {r['diff_t_minus_c']:.3f} |")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("Use the broad rule-based outcomes as evidence of downstream engagement. Use `strict_human_curiosity` as the cleaner outcome for curiosity contagion because it excludes clarification, repair, task continuation, and simple answers to the assistant's question.")
    (args.outdir / "q2_broad_vs_strict_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("Wrote:", args.outdir)
    print("Rows analyzed:", len(all_rows))
    for p in sorted(args.outdir.iterdir()):
        print(p)

if __name__ == "__main__":
    main()
