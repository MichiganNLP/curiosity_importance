#!/usr/bin/env python3
"""Causal-style matched analysis using LLM-judged model curiosity and LLM-judged human curiosity.

Treatment: next-user response after an assistant question labeled as curiosity.
Control: next-user response after a matched assistant question labeled as non-curiosity.
Outcome: Q2 LLM judge labels the next-user response as human_curiosity.

No new LLM calls are made; this consumes existing llm_judge_human_curiosity.py outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Tuple

try:
    from scipy import stats
except Exception:  # pragma: no cover
    stats = None

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "outputs"
DEFAULT_INPUTS = {
    "wildchat": OUTPUT_DIR / "wildchat_q2_human_curiosity_judged.jsonl",
    "oasst1": OUTPUT_DIR / "oasst1_q2_human_curiosity_judged.jsonl",
    "lmsys": OUTPUT_DIR / "lmsys_q2_human_curiosity_judged.jsonl",
    "sharegpt": OUTPUT_DIR / "sharegpt_q2_human_curiosity_judged.jsonl",
}

OUTCOMES = {
    "strict_human_curiosity": lambda r: int(r.get("status") == "human_curiosity"),
    "strict_human_curiosity_moderate_strong": lambda r: int(r.get("status") == "human_curiosity" and int(r.get("curiosity_intensity") or 0) >= 2),
}

RULE_OUTCOMES = {
    "rule_next_user_has_question": lambda r: float(r.get("rule_next_user_has_question") or 0),
    "rule_next_user_question_count": lambda r: float(r.get("rule_next_user_question_count") or 0),
    "rule_next_user_exploratory_question_count": lambda r: float(r.get("rule_next_user_exploratory_question_count") or 0),
    "rule_next_user_epistemic_marker_count": lambda r: float(r.get("rule_next_user_epistemic_marker_count") or 0),
    "rule_next_user_word_count": lambda r: float(r.get("rule_next_user_word_count") or 0),
}


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj


def bootstrap_ci(diffs: List[float], n_boot: int, seed: int) -> Tuple[float, float]:
    if not diffs:
        return math.nan, math.nan
    rng = random.Random(seed)
    vals = []
    n = len(diffs)
    for _ in range(n_boot):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        vals.append(mean(sample))
    vals.sort()
    lo = vals[int(0.025 * (n_boot - 1))]
    hi = vals[int(0.975 * (n_boot - 1))]
    return lo, hi


def paired_p_value(diffs: List[float]) -> float:
    if not diffs:
        return math.nan
    if stats is not None:
        res = stats.ttest_1samp(diffs, 0.0)
        return float(res.pvalue)
    # Normal approximation fallback.
    if len(diffs) < 2:
        return math.nan
    m = mean(diffs)
    var = sum((d - m) ** 2 for d in diffs) / (len(diffs) - 1)
    se = math.sqrt(var / len(diffs)) if var > 0 else 0.0
    if se == 0:
        return 0.0 if m != 0 else 1.0
    z = abs(m / se)
    return math.erfc(z / math.sqrt(2.0))


def by_pair(rows: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    pairs: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for r in rows:
        pair_id = str(r.get("pair_id"))
        arm = str(r.get("arm"))
        if arm in {"treatment", "control"}:
            pairs[pair_id][arm] = r
    return pairs


def paired_summary(dataset: str, pairs: Dict[str, Dict[str, Dict[str, Any]]], outcome_name: str, fn, n_boot: int, seed: int) -> Dict[str, Any]:
    tvals, cvals, diffs = [], [], []
    discordant_t1c0 = 0
    discordant_t0c1 = 0
    for arms in pairs.values():
        if "treatment" not in arms or "control" not in arms:
            continue
        t = float(fn(arms["treatment"]))
        c = float(fn(arms["control"]))
        tvals.append(t)
        cvals.append(c)
        diffs.append(t - c)
        if t == 1 and c == 0:
            discordant_t1c0 += 1
        if t == 0 and c == 1:
            discordant_t0c1 += 1
    ci_low, ci_high = bootstrap_ci(diffs, n_boot, seed)
    p = paired_p_value(diffs)
    return {
        "dataset": dataset,
        "outcome": outcome_name,
        "n_pairs": len(diffs),
        "treatment_mean": mean(tvals) if tvals else math.nan,
        "control_mean": mean(cvals) if cvals else math.nan,
        "att_t_minus_c": mean(diffs) if diffs else math.nan,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value_paired": p,
        "discordant_treatment_only": discordant_t1c0,
        "discordant_control_only": discordant_t0c1,
    }


def subtype_summary(dataset: str, pairs: Dict[str, Dict[str, Dict[str, Any]]], n_boot: int, seed: int) -> List[Dict[str, Any]]:
    grouped: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = defaultdict(dict)
    for pid, arms in pairs.items():
        if "treatment" not in arms or "control" not in arms:
            continue
        subtype = arms["treatment"].get("assistant_question_type") or "unknown"
        grouped[str(subtype)][pid] = arms
    rows = []
    for subtype, subpairs in sorted(grouped.items()):
        if len(subpairs) < 20:
            continue
        rows.append({"subtype": subtype, **paired_summary(dataset, subpairs, "strict_human_curiosity", OUTCOMES["strict_human_curiosity"], n_boot, seed)})
    return rows


def type_distribution(dataset: str, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for arm in ["treatment", "control"]:
        subset = [r for r in rows if r.get("arm") == arm]
        total = len(subset) or 1
        for status, n in Counter(r.get("status") for r in subset).most_common():
            out.append({"dataset": dataset, "arm": arm, "field": "status", "label": status, "n": n, "pct": 100*n/total})
        curious = [r for r in subset if r.get("status") == "human_curiosity"]
        ctotal = len(curious) or 1
        for typ, n in Counter(r.get("curiosity_type") for r in curious).most_common():
            out.append({"dataset": dataset, "arm": arm, "field": "curiosity_type_given_curious", "label": typ, "n": n, "pct": 100*n/ctotal})
        noncur = [r for r in subset if r.get("status") == "not_human_curiosity"]
        ntotal = len(noncur) or 1
        for fn, n in Counter(r.get("non_curiosity_function") for r in noncur).most_common():
            out.append({"dataset": dataset, "arm": arm, "field": "non_curiosity_function_given_not_curious", "label": fn, "n": n, "pct": 100*n/ntotal})
    return out


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def write_readme(outdir: Path) -> None:
    text = """# Q2 Causal-Style Analysis with LLM-Judged Human Curiosity

This folder contains the Q2 matched causal-style analysis where both the treatment and outcome use LLM-judge annotations.

Treatment: the next-user turn after an assistant question labeled as model curiosity by the Q1 assistant-question judge.
Control: the next-user turn after a matched assistant question labeled as non-curiosity.
Outcome: whether the next-user turn is labeled `human_curiosity` by the Q2 downstream human-curiosity judge.

No new LLM calls were made for this analysis. It reuses the existing Q2 judged matched-pair files.

Files:
- `llm_judged_curiosity_att_summary.csv`: matched ATT estimates for strict human-curiosity outcomes and broad rule outcomes.
- `llm_judged_curiosity_subtype_att_summary.csv`: strict human-curiosity ATT by assistant curiosity subtype.
- `llm_judged_curiosity_status_type_distribution.csv`: treatment/control distributions of judged human-curiosity statuses, types, and non-curiosity functions.

Interpretation: These are observational matched estimates, not randomized causal effects. They are best described as causal-style or matched ATT estimates.
"""
    (outdir / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, default=OUTPUT_DIR / "q2_causal_llm_judged_human_curiosity")
    ap.add_argument("--copy-to", type=Path, default=None, help="Optional second output directory")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    att_rows: List[Dict[str, Any]] = []
    subtype_rows: List[Dict[str, Any]] = []
    dist_rows: List[Dict[str, Any]] = []

    all_pairs: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for dataset, path in DEFAULT_INPUTS.items():
        if not path.exists():
            print("Missing", dataset, path)
            continue
        rows = list(read_jsonl(path))
        pairs = by_pair(rows)
        print(dataset, "rows", len(rows), "pairs", len([p for p in pairs.values() if "treatment" in p and "control" in p]))
        for name, fn in {**OUTCOMES, **RULE_OUTCOMES}.items():
            att_rows.append(paired_summary(dataset, pairs, name, fn, args.n_boot, args.seed))
        subtype_rows.extend(subtype_summary(dataset, pairs, args.n_boot, args.seed))
        dist_rows.extend(type_distribution(dataset, rows))
        for pid, arms in pairs.items():
            all_pairs[f"{dataset}:{pid}"] = arms

    for name, fn in {**OUTCOMES, **RULE_OUTCOMES}.items():
        att_rows.append(paired_summary("all", all_pairs, name, fn, args.n_boot, args.seed))
    subtype_rows.extend(subtype_summary("all", all_pairs, args.n_boot, args.seed))

    write_csv(args.outdir / "llm_judged_curiosity_att_summary.csv", att_rows)
    write_csv(args.outdir / "llm_judged_curiosity_subtype_att_summary.csv", subtype_rows)
    write_csv(args.outdir / "llm_judged_curiosity_status_type_distribution.csv", dist_rows)
    write_readme(args.outdir)

    if args.copy_to:
        args.copy_to.mkdir(parents=True, exist_ok=True)
        for p in args.outdir.iterdir():
            if p.is_file():
                (args.copy_to / p.name).write_bytes(p.read_bytes())

    print("Wrote", args.outdir)
    if args.copy_to:
        print("Copied", args.copy_to)


if __name__ == "__main__":
    main()
