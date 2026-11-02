#!/usr/bin/env python3
"""
Matched observational analysis for WildChat curiosity events.

This script uses the existing LLM-judged assistant-question file. It matches
curiosity questions to non-curiosity question controls with similar pre-response
context, then compares the next user turn.

Main use:
  python3 src/q1_q2/matched_observational_analysis.py \
    --input outputs/wildchat_llm_judged.jsonl \
    --outdir outputs/wildchat_matched_q2 \
    --control-types engagement_maintenance
"""

import argparse
import csv
import json
import bisect
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

QUESTION_RE = re.compile(r"\?")
WORD_RE = re.compile(r"\b\w+\b")
EPISTEMIC_RE = re.compile(
    r"\b(why|how|what if|i wonder|wonder|curious|not sure|unsure|maybe|perhaps|"
    r"could|might|i think|i believe|assume|assumption|evidence|explain|understand|"
    r"confused|unclear|reason|cause|possible|possibility)\b",
    re.I,
)
EXPLORATORY_Q_RE = re.compile(
    r"\b(why|how|what if|how come|what would|what might|could|would|is it possible|"
    r"what are other|what else|in what way)\b[^?]*\?",
    re.I,
)
USER_REQUESTED_QUESTIONS_RE = re.compile(
    r"\b(ask (me )?(more|some|other)?\s*questions?|generate questions?|"
    r"write questions?|make questions?|interview questions?|quiz questions?|"
    r"questionnaire|questions to ask|ask follow[- ]?up)\b",
    re.I,
)

TOPIC_PATTERNS = [
    ("coding", r"\b(code|python|javascript|java|html|css|sql|api|bug|error|function|script|program|terminal|powershell|linux|github)\b"),
    ("writing_creative", r"\b(write|rewrite|story|song|poem|lyrics|scene|dialogue|character|novel|creative|paragraph|essay)\b"),
    ("academic_research", r"\b(research|paper|proposal|hypothesis|thesis|study|literature|citation|abstract|methodology)\b"),
    ("learning_explanation", r"\b(explain|teach|understand|concept|theory|why does|how does|difference between|compare|solve|calculate|test statistic|hypothesis test)\b"),
    ("advice_personal", r"\b(should i|what should i do|relationship|friend|family|girlfriend|boyfriend|parent|boss|coworker|feel|worried)\b"),
    ("argument_reasoning", r"\b(argue|argument|claim|because|therefore|evidence|assumption|agree|disagree|fair|unfair|moral|ethical)\b"),
    ("translation", r"\b(translate|translation|chinese|spanish|french|simplified|language)\b"),
    ("business_marketing", r"\b(marketing|product|customer|seo|keyword|brand|sales|business|title|lazada|amazon)\b"),
]
TOPIC_RES = [(name, re.compile(pat, re.I)) for name, pat in TOPIC_PATTERNS]

OUTCOMES = [
    "next_user_has_question",
    "next_user_question_count",
    "next_user_exploratory_question_count",
    "next_user_epistemic_marker_count",
    "next_user_word_count",
    "next_user_epistemic_markers_per_100w",
    "next_user_exploratory_questions_per_100w",
]

NUMERIC_MATCH_FEATURES = [
    "prev_user_word_count",
    "prev_user_question_count",
    "prev_user_epistemic_marker_count",
    "prev_user_exploratory_question_count",
    "assistant_word_count",
    "conversation_turn_count",
    "assistant_turn_index",
]


def read_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text or ""))


def question_count(text: str) -> int:
    return len(QUESTION_RE.findall(text or ""))


def epistemic_count(text: str) -> int:
    return len(EPISTEMIC_RE.findall(text or ""))


def exploratory_question_count(text: str) -> int:
    return len(EXPLORATORY_Q_RE.findall(text or ""))


def model_family(model: Optional[str]) -> str:
    m = (model or "unknown").lower()
    if "gpt-4" in m:
        return "gpt-4"
    if "gpt-3.5" in m or "gpt-35" in m:
        return "gpt-3.5"
    return m or "unknown"


def topic_category(text: str) -> str:
    t = text or ""
    for name, rx in TOPIC_RES:
        if rx.search(t):
            return name
    return "other"


def judged_status(row: Dict) -> str:
    return (row.get("llm_judge") or {}).get("status") or row.get("status") or ""


def judged_type(row: Dict) -> str:
    status = judged_status(row)
    qtype = (row.get("llm_judge") or {}).get("question_type") or row.get("question_type") or row.get("primary_function") or ""
    # Same post-hoc recode used in the main analyses.
    if qtype == "information_request" and status == "curiosity":
        return "information_gap_curiosity"
    if qtype == "information_request" and status == "not_curiosity":
        return "context_request"
    return qtype


def judged_keep(row: Dict) -> bool:
    judge = row.get("llm_judge") or {}
    if "keep" in judge:
        return bool(judge["keep"])
    return True


def make_event(row: Dict) -> Optional[Dict]:
    prev = row.get("previous_user_text") or ""
    assistant = row.get("assistant_turn_text") or ""
    nxt = row.get("next_user_text") or ""
    nw = int(row.get("next_user_word_count") or word_count(nxt))
    event = dict(row)
    event.update({
        "status": judged_status(row),
        "question_type": judged_type(row),
        "keep": judged_keep(row),
        "model_family": model_family(row.get("model")),
        "topic_category": topic_category(prev),
        "prev_user_word_count": word_count(prev),
        "prev_user_question_count": question_count(prev),
        "prev_user_epistemic_marker_count": epistemic_count(prev),
        "prev_user_exploratory_question_count": exploratory_question_count(prev),
        "assistant_word_count": word_count(assistant),
        "next_user_word_count": nw,
        "next_user_has_question": int(bool(row.get("next_user_has_question"))),
        "next_user_question_count": int(row.get("next_user_question_count") or question_count(nxt)),
        "next_user_exploratory_question_count": int(row.get("next_user_exploratory_question_count") or exploratory_question_count(nxt)),
        "next_user_epistemic_marker_count": int(row.get("next_user_epistemic_marker_count") or epistemic_count(nxt)),
    })
    denom = max(nw, 1) / 100.0
    event["next_user_epistemic_markers_per_100w"] = event["next_user_epistemic_marker_count"] / denom
    event["next_user_exploratory_questions_per_100w"] = event["next_user_exploratory_question_count"] / denom
    return event


def should_drop(event: Dict, args) -> bool:
    if args.keep_only and not event["keep"]:
        return True
    if args.drop_unclear_problematic and event["status"] in {"unclear", "problematic"}:
        return True
    if args.require_next_user and not event.get("next_user_text"):
        return True
    if args.exclude_user_requested_questions and USER_REQUESTED_QUESTIONS_RE.search(event.get("previous_user_text") or ""):
        return True
    if event["prev_user_word_count"] < args.min_prev_words:
        return True
    if event["assistant_word_count"] < args.min_assistant_words:
        return True
    return False


def zscore_params(events: Sequence[Dict]) -> Dict[str, Tuple[float, float]]:
    params = {}
    for feat in NUMERIC_MATCH_FEATURES:
        vals = [float(e.get(feat) or 0.0) for e in events]
        mu = mean(vals) if vals else 0.0
        var = mean([(v - mu) ** 2 for v in vals]) if vals else 0.0
        sd = math.sqrt(var) if var > 0 else 1.0
        params[feat] = (mu, sd)
    return params


def dist(a: Dict, b: Dict, params: Dict[str, Tuple[float, float]], weights: Dict[str, float]) -> float:
    total = 0.0
    for feat in NUMERIC_MATCH_FEATURES:
        mu, sd = params[feat]
        za = (float(a.get(feat) or 0.0) - mu) / sd
        zb = (float(b.get(feat) or 0.0) - mu) / sd
        total += weights.get(feat, 1.0) * abs(za - zb)
    return total


def build_blocks(events: Sequence[Dict], exact_fields: Sequence[str]) -> Dict[Tuple, List[int]]:
    blocks = defaultdict(list)
    for i, e in enumerate(events):
        blocks[tuple(e.get(f) for f in exact_fields)].append(i)
    return blocks


def greedy_match(treat: List[Dict], control: List[Dict], args) -> List[Tuple[Dict, Dict, float]]:
    all_events = treat + control
    params = zscore_params(all_events)
    weights = {
        "prev_user_word_count": 1.5,
        "assistant_word_count": 1.0,
        "assistant_turn_index": 0.75,
        "conversation_turn_count": 0.5,
        "prev_user_question_count": 1.0,
        "prev_user_epistemic_marker_count": 0.75,
        "prev_user_exploratory_question_count": 0.75,
    }

    # Index controls by exact-match block and sort within each block by previous user length.
    block_items = defaultdict(list)
    for ci, c in enumerate(control):
        key = tuple(c.get(f) for f in args.exact_fields)
        block_items[key].append((c["prev_user_word_count"], ci))
    for key in list(block_items):
        block_items[key].sort()

    unused = {i for i in range(len(control))}
    matches = []

    treat_order = sorted(
        range(len(treat)),
        key=lambda i: (treat[i]["topic_category"], treat[i]["model_family"], treat[i]["prev_user_word_count"]),
    )
    half_window = max(args.candidate_window // 2, 1)
    for ti in treat_order:
        t = treat[ti]
        key = tuple(t.get(f) for f in args.exact_fields)
        items = block_items.get(key, [])
        if not items:
            continue
        lengths = [x[0] for x in items]
        pos = bisect.bisect_left(lengths, t["prev_user_word_count"])
        lo = max(pos - half_window, 0)
        hi = min(pos + half_window, len(items))
        candidates = [ci for _, ci in items[lo:hi] if ci in unused]

        # If the local window was exhausted by prior matches, try the full exact block.
        if not candidates:
            candidates = [ci for _, ci in items if ci in unused]
        if not candidates:
            continue

        scored = [(dist(t, control[ci], params, weights), ci) for ci in candidates]
        d, ci = min(scored, key=lambda x: x[0])
        if args.caliper is not None and d > args.caliper:
            continue
        unused.remove(ci)
        matches.append((t, control[ci], d))
    return matches

def ci_bootstrap(diffs: List[float], reps: int, seed: int) -> Tuple[float, float]:
    if not diffs:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    vals = []
    n = len(diffs)
    for _ in range(reps):
        vals.append(mean(diffs[rng.randrange(n)] for _ in range(n)))
    vals.sort()
    lo = vals[int(0.025 * (reps - 1))]
    hi = vals[int(0.975 * (reps - 1))]
    return lo, hi


def summarize(matches: Sequence[Tuple[Dict, Dict, float]], reps: int, seed: int) -> List[Dict]:
    rows = []
    for out in OUTCOMES:
        tv = [float(t.get(out) or 0.0) for t, _, _ in matches]
        cv = [float(c.get(out) or 0.0) for _, c, _ in matches]
        diffs = [a - b for a, b in zip(tv, cv)]
        lo, hi = ci_bootstrap(diffs, reps, seed)
        rows.append({
            "outcome": out,
            "n_pairs": len(matches),
            "treatment_mean": mean(tv) if tv else float("nan"),
            "control_mean": mean(cv) if cv else float("nan"),
            "difference": mean(diffs) if diffs else float("nan"),
            "bootstrap_ci_low": lo,
            "bootstrap_ci_high": hi,
        })
    return rows


def write_pairs(path: Path, matches: Sequence[Tuple[Dict, Dict, float]]) -> None:
    fields = [
        "pair_id", "distance",
        "treatment_conversation_id", "control_conversation_id",
        "treatment_question_type", "control_question_type",
        "treatment_model", "control_model",
        "topic_category",
        "treatment_prev_user_text", "control_prev_user_text",
        "treatment_question_text", "control_question_text",
        "treatment_next_user_text", "control_next_user_text",
    ] + ["treatment_" + x for x in OUTCOMES] + ["control_" + x for x in OUTCOMES]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, (t, c, d) in enumerate(matches):
            row = {
                "pair_id": i,
                "distance": d,
                "treatment_conversation_id": t.get("conversation_id"),
                "control_conversation_id": c.get("conversation_id"),
                "treatment_question_type": t.get("question_type"),
                "control_question_type": c.get("question_type"),
                "treatment_model": t.get("model"),
                "control_model": c.get("model"),
                "topic_category": t.get("topic_category"),
                "treatment_prev_user_text": re.sub(r"\s+", " ", t.get("previous_user_text") or "").strip(),
                "control_prev_user_text": re.sub(r"\s+", " ", c.get("previous_user_text") or "").strip(),
                "treatment_question_text": t.get("question_text"),
                "control_question_text": c.get("question_text"),
                "treatment_next_user_text": re.sub(r"\s+", " ", t.get("next_user_text") or "").strip(),
                "control_next_user_text": re.sub(r"\s+", " ", c.get("next_user_text") or "").strip(),
            }
            for out in OUTCOMES:
                row["treatment_" + out] = t.get(out)
                row["control_" + out] = c.get(out)
            w.writerow(row)


def write_summary(path: Path, rows: Sequence[Dict]) -> None:
    fields = ["outcome", "n_pairs", "treatment_mean", "control_mean", "difference", "bootstrap_ci_low", "bootstrap_ci_high"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def design_matrix(events: Sequence[Dict]) -> Tuple["object", List[str]]:
    import numpy as np

    cols = ["intercept", "treatment"]
    numeric = NUMERIC_MATCH_FEATURES
    cols.extend(numeric)

    model_levels = sorted({e.get("model_family") for e in events})
    topic_levels = sorted({e.get("topic_category") for e in events})
    # Drop first level for each categorical family to avoid perfect collinearity.
    model_dummies = model_levels[1:]
    topic_dummies = topic_levels[1:]
    cols.extend(["model=" + x for x in model_dummies])
    cols.extend(["topic=" + x for x in topic_dummies])

    # Standardize numeric controls for stable coefficients.
    params = zscore_params(events)
    x = []
    for e in events:
        row = [1.0, float(e.get("treatment") or 0.0)]
        for feat in numeric:
            mu, sd = params[feat]
            row.append((float(e.get(feat) or 0.0) - mu) / sd)
        row.extend([1.0 if e.get("model_family") == lev else 0.0 for lev in model_dummies])
        row.extend([1.0 if e.get("topic_category") == lev else 0.0 for lev in topic_dummies])
        x.append(row)
    return np.asarray(x, dtype=float), cols


def ols_hc1(events: Sequence[Dict], outcome: str) -> Dict:
    import numpy as np

    y = np.asarray([float(e.get(outcome) or 0.0) for e in events], dtype=float)
    x, cols = design_matrix(events)
    n, k = x.shape
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ beta
    xtx_inv = np.linalg.pinv(x.T @ x)
    meat = x.T @ ((resid ** 2)[:, None] * x)
    scale = n / max(n - k, 1)
    cov = scale * xtx_inv @ meat @ xtx_inv
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    idx = cols.index("treatment")
    coef = float(beta[idx])
    stderr = float(se[idx])
    tstat = coef / stderr if stderr > 0 else float("nan")
    return {
        "outcome": outcome,
        "n": n,
        "k": k,
        "coef_treatment": coef,
        "robust_se": stderr,
        "t_stat": tstat,
        "ci_low": coef - 1.96 * stderr if stderr == stderr else float("nan"),
        "ci_high": coef + 1.96 * stderr if stderr == stderr else float("nan"),
    }


def regression_summary(treatment: Sequence[Dict], control: Sequence[Dict]) -> List[Dict]:
    events = []
    for e in treatment:
        row = dict(e)
        row["treatment"] = 1
        events.append(row)
    for e in control:
        row = dict(e)
        row["treatment"] = 0
        events.append(row)
    return [ols_hc1(events, out) for out in OUTCOMES]


def write_regression_summary(path: Path, rows: Sequence[Dict]) -> None:
    fields = ["outcome", "n", "k", "coef_treatment", "robust_se", "t_stat", "ci_low", "ci_high"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--control-types", nargs="+", default=["engagement_maintenance"], help="Question types to use as controls.")
    ap.add_argument("--treatment-types", nargs="*", default=[], help="Optional curiosity question types to keep; default all curiosity.")
    ap.add_argument("--exact-fields", nargs="+", default=["model_family", "topic_category"], help="Fields that must match exactly.")
    ap.add_argument("--caliper", type=float, default=None, help="Maximum numeric distance; default no caliper.")
    ap.add_argument("--candidate-window", type=int, default=800, help="Controls to score per exact block near the treatment previous-user length.")
    ap.add_argument("--relax-topic", action="store_true", help="If no same-topic controls exist, allow matching on remaining exact fields.")
    ap.add_argument("--keep-only", action="store_true", default=True)
    ap.add_argument("--drop-unclear-problematic", action="store_true", default=True)
    ap.add_argument("--require-next-user", action="store_true", default=True)
    ap.add_argument("--exclude-user-requested-questions", action="store_true", default=True)
    ap.add_argument("--min-prev-words", type=int, default=2)
    ap.add_argument("--min-assistant-words", type=int, default=2)
    ap.add_argument("--bootstrap-reps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=13)
    return ap.parse_args()


def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    events = []
    raw = 0
    for row in read_jsonl(args.input):
        raw += 1
        event = make_event(row)
        if event is None or should_drop(event, args):
            continue
        events.append(event)

    treatment = [e for e in events if e["status"] == "curiosity"]
    if args.treatment_types:
        treatment = [e for e in treatment if e["question_type"] in set(args.treatment_types)]
    control_type_set = set(args.control_types)
    control = [e for e in events if e["status"] == "not_curiosity" and e["question_type"] in control_type_set]

    matches = greedy_match(treatment, control, args)
    summary = summarize(matches, args.bootstrap_reps, args.seed)
    reg_summary = regression_summary(treatment, control)

    write_pairs(args.outdir / "matched_pairs.csv", matches)
    write_summary(args.outdir / "matched_summary.csv", summary)
    write_regression_summary(args.outdir / "regression_summary.csv", reg_summary)

    metadata = {
        "input": str(args.input),
        "raw_rows": raw,
        "eligible_events": len(events),
        "treatment_events": len(treatment),
        "control_events": len(control),
        "matched_pairs": len(matches),
        "control_types": args.control_types,
        "treatment_types": args.treatment_types or "all_curiosity",
        "exact_fields": args.exact_fields,
        "caliper": args.caliper,
        "exclude_user_requested_questions": args.exclude_user_requested_questions,
        "topic_counts_treatment": Counter(e["topic_category"] for e in treatment),
        "topic_counts_control": Counter(e["topic_category"] for e in control),
        "question_type_counts_treatment": Counter(e["question_type"] for e in treatment),
        "question_type_counts_control": Counter(e["question_type"] for e in control),
    }
    with (args.outdir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("Raw rows:", raw)
    print("Eligible events:", len(events))
    print("Treatment events:", len(treatment))
    print("Control events:", len(control))
    print("Matched pairs:", len(matches))
    print("Wrote:", args.outdir / "matched_summary.csv")
    print("Wrote:", args.outdir / "regression_summary.csv")
    for r in summary:
        print(f"MATCHED {r['outcome']}: T={r['treatment_mean']:.4f} C={r['control_mean']:.4f} diff={r['difference']:.4f} CI=({r['bootstrap_ci_low']:.4f},{r['bootstrap_ci_high']:.4f})")
    for r in reg_summary:
        print(f"REG {r['outcome']}: coef={r['coef_treatment']:.4f} SE={r['robust_se']:.4f} CI=({r['ci_low']:.4f},{r['ci_high']:.4f})")


if __name__ == "__main__":
    main()
