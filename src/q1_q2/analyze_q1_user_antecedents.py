#!/usr/bin/env python3
"""Analyze what kinds of user turns precede curiosity-oriented assistant questions."""

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "outputs"
DEFAULT_INPUTS = {
    "wildchat": OUTPUT_DIR / "wildchat_llm_judged.jsonl",
    "oasst1": OUTPUT_DIR / "oasst1_llm_judged.jsonl",
    "lmsys": OUTPUT_DIR / "lmsys_llm_judged.jsonl",
    "sharegpt": OUTPUT_DIR / "sharegpt_llm_judged.jsonl",
}

WORD_RE = re.compile(r"\b\w+\b")
QUESTION_RE = re.compile(r"\?")
MAX_STORED_TEXT_CHARS = 1200
MAX_STORED_QUESTION_CHARS = 500

TOPIC_PATTERNS = [
    ("coding_technical", r"\b(code|python|javascript|java|html|css|sql|api|bug|error|function|script|program|terminal|powershell|linux|github|server|database|regex|install|package|debug)\b"),
    ("writing_creative", r"\b(write|rewrite|story|song|poem|lyrics|scene|dialogue|character|novel|creative|paragraph|essay|draft|tone|style)\b"),
    ("academic_research", r"\b(research|paper|proposal|hypothesis|thesis|study|literature|citation|abstract|methodology|experiment|survey|analysis)\b"),
    ("learning_studying", r"\b(explain|teach|understand|concept|theory|why does|how does|difference between|compare|solve|calculate|homework|exam|test|quiz|study|learn|math|chemistry|physics|biology|statistics)\b"),
    ("life_advice", r"\b(should i|what should i do|relationship|friend|family|girlfriend|boyfriend|parent|boss|coworker|feel|worried|anxious|advice|career|job|decision)\b"),
    ("argument_reasoning", r"\b(argue|argument|claim|because|therefore|evidence|assumption|agree|disagree|fair|unfair|moral|ethical|opinion|debate)\b"),
    ("translation_language", r"\b(translate|translation|chinese|spanish|french|japanese|korean|german|simplified|language|grammar|rephrase|paraphrase)\b"),
    ("business_marketing", r"\b(marketing|product|customer|seo|keyword|brand|sales|business|startup|proposal|market|company|client|advertising)\b"),
    ("gaming_roleplay", r"\b(game|gaming|minecraft|roblox|steam|character|roleplay|role play|dnd|pokemon|half-life|quest|player|npc|anime)\b"),
    ("health_medical", r"\b(health|medical|doctor|symptom|diagnosis|medicine|therapy|disease|cancer|pain|diet|sleep|mental health)\b"),
]
TOPIC_RES = [(name, re.compile(pat, re.I)) for name, pat in TOPIC_PATTERNS]

QUESTION_TYPE_PATTERNS = [
    ("why_causal", r"\bwhy\b|\bhow come\b|\bwhat causes?\b|\bwhat caused\b"),
    ("how_procedural", r"\bhow (do|does|did|can|could|should|would|to)\b|\bhow\s+to\b"),
    ("what_information", r"\bwhat\b|\bwhich\b|\bwho\b|\bwhere\b|\bwhen\b"),
    ("should_advice", r"\bshould i\b|\bwhat should i do\b|\bdo you think i should\b|\bis it okay\b"),
    ("what_if_hypothetical", r"\bwhat if\b|\bsuppose\b|\bimagine if\b|\bif .* were\b"),
    ("request_generation", r"\b(write|rewrite|draft|generate|create|make|compose|produce|give me|list|summarize)\b"),
    ("clarification_followup", r"\b(can you explain|explain more|elaborate|what do you mean|i don't understand|confused|unclear)\b"),
    ("command_no_question", r"^[^?]{0,220}\b(write|rewrite|draft|generate|create|make|compose|produce|summarize|translate|fix|help|solve|continue)\b"),
]
QUESTION_TYPE_RES = [(name, re.compile(pat, re.I | re.S)) for name, pat in QUESTION_TYPE_PATTERNS]

EPISTEMIC_RE = re.compile(r"\b(why|how|what if|wonder|curious|not sure|unsure|maybe|perhaps|could|might|i think|i believe|assume|assumption|evidence|explain|understand|confused|unclear|reason|cause|possible)\b", re.I)


def read_jsonl(path: Path) -> Iterable[Dict]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def judge_status(row: Dict) -> str:
    return (row.get("llm_judge") or {}).get("status") or row.get("llm_status") or ""


def judge_type(row: Dict) -> str:
    status = judge_status(row)
    qtype = (row.get("llm_judge") or {}).get("question_type") or row.get("llm_question_type") or row.get("primary_function") or ""
    if qtype == "information_request" and status == "curiosity":
        return "information_gap_curiosity"
    if qtype == "information_request" and status == "not_curiosity":
        return "context_request"
    return qtype


def judge_keep(row: Dict) -> bool:
    judge = row.get("llm_judge") or {}
    if "keep" in judge:
        return bool(judge["keep"])
    val = row.get("llm_keep")
    if isinstance(val, bool):
        return val
    return str(val).lower() == "true"


def topic(text: str) -> str:
    for name, rx in TOPIC_RES:
        if rx.search(text or ""):
            return name
    return "other"


def user_question_type(text: str) -> str:
    text = text or ""
    for name, rx in QUESTION_TYPE_RES:
        if rx.search(text):
            return name
    if "?" in text:
        return "other_question"
    return "other_statement"


def truncate_text(text: str, max_chars: int = MAX_STORED_TEXT_CHARS) -> str:
    text = text or ""
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + " ..."


def text_features(text: str) -> Dict:
    words = WORD_RE.findall(text or "")
    return {
        "prev_user_words": len(words),
        "prev_user_has_question": int("?" in (text or "")),
        "prev_user_question_marks": len(QUESTION_RE.findall(text or "")),
        "prev_user_epistemic_markers": len(EPISTEMIC_RE.findall(text or "")),
    }


def load_events(path: Path) -> List[Dict]:
    events = []
    for row in read_jsonl(path):
        prev = row.get("previous_user_text") or ""
        status = judge_status(row)
        qtype = judge_type(row)
        keep = judge_keep(row)
        rec = {
            "conversation_id": row.get("conversation_id"),
            "model": row.get("model"),
            "question_text": truncate_text(row.get("question_text") or "", MAX_STORED_QUESTION_CHARS),
            "previous_user_text": truncate_text(prev),
            "assistant_turn_text": truncate_text(row.get("assistant_turn_text") or ""),
            "llm_keep": keep,
            "llm_status": status,
            "llm_question_type": qtype,
            "user_topic": topic(prev),
            "user_question_type": user_question_type(prev),
        }
        rec.update(text_features(prev))
        events.append(rec)
    return events


def write_csv(path: Path, rows: List[Dict], fields: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def summarize_group(events: List[Dict], group_key: str, min_n: int) -> List[Dict]:
    groups = defaultdict(list)
    for e in events:
        if not e["llm_keep"]:
            continue
        if e["llm_status"] in {"unclear", "problematic"}:
            continue
        groups[e[group_key]].append(e)
    rows = []
    for label, items in groups.items():
        n = len(items)
        if n < min_n:
            continue
        cur = [x for x in items if x["llm_status"] == "curiosity"]
        non = [x for x in items if x["llm_status"] == "not_curiosity"]
        cur_types = Counter(x["llm_question_type"] for x in cur)
        rows.append({
            group_key: label,
            "n_real_questions": n,
            "n_curiosity": len(cur),
            "n_not_curiosity": len(non),
            "curiosity_rate_pct": len(cur) / n * 100 if n else 0,
            "top_curiosity_type": cur_types.most_common(1)[0][0] if cur_types else "",
            "top_curiosity_type_count": cur_types.most_common(1)[0][1] if cur_types else 0,
            "avg_prev_user_words": sum(x["prev_user_words"] for x in items) / n,
            "avg_prev_user_epistemic_markers": sum(x["prev_user_epistemic_markers"] for x in items) / n,
            "prev_user_has_question_pct": sum(x["prev_user_has_question"] for x in items) / n * 100,
        })
    return sorted(rows, key=lambda r: (r["curiosity_rate_pct"], r["n_curiosity"]), reverse=True)


def summarize_cross(events: List[Dict], min_n: int) -> List[Dict]:
    groups = defaultdict(list)
    for e in events:
        if not e["llm_keep"] or e["llm_status"] in {"unclear", "problematic"}:
            continue
        groups[(e["user_topic"], e["user_question_type"])].append(e)
    rows = []
    for (topic_label, qtype_label), items in groups.items():
        n = len(items)
        if n < min_n:
            continue
        cur = [x for x in items if x["llm_status"] == "curiosity"]
        cur_types = Counter(x["llm_question_type"] for x in cur)
        rows.append({
            "user_topic": topic_label,
            "user_question_type": qtype_label,
            "n_real_questions": n,
            "n_curiosity": len(cur),
            "curiosity_rate_pct": len(cur) / n * 100 if n else 0,
            "top_curiosity_type": cur_types.most_common(1)[0][0] if cur_types else "",
            "top_curiosity_type_count": cur_types.most_common(1)[0][1] if cur_types else 0,
        })
    return sorted(rows, key=lambda r: (r["curiosity_rate_pct"], r["n_curiosity"]), reverse=True)


def write_examples(path: Path, events: List[Dict], max_per_group: int = 3):
    rows = []
    seen = Counter()
    for e in events:
        if not e["llm_keep"] or e["llm_status"] != "curiosity":
            continue
        key = (e["user_topic"], e["user_question_type"], e["llm_question_type"])
        if seen[key] >= max_per_group:
            continue
        seen[key] += 1
        rows.append(e)
    rows = sorted(rows, key=lambda r: (r["user_topic"], r["user_question_type"], r["llm_question_type"]))
    fields = ["user_topic", "user_question_type", "llm_question_type", "question_text", "previous_user_text", "assistant_turn_text"]
    write_csv(path, rows, fields)


def copy_to_home(nfs_dir: Path, home_dir: Path):
    home_dir.mkdir(parents=True, exist_ok=True)
    for p in nfs_dir.iterdir():
        if p.is_file():
            (home_dir / p.name).write_bytes(p.read_bytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(DEFAULT_INPUTS))
    ap.add_argument("--input", type=Path, default=None)
    ap.add_argument("--outdir", type=Path, default=None)
    ap.add_argument("--home-copy", type=Path, default=None)
    ap.add_argument("--min-n", type=int, default=50)
    args = ap.parse_args()

    inp = args.input or DEFAULT_INPUTS[args.dataset]
    outdir = args.outdir or (OUTPUT_DIR / f"{args.dataset}_q1_user_antecedents")
    outdir.mkdir(parents=True, exist_ok=True)

    events = load_events(inp)
    topic_rows = summarize_group(events, "user_topic", args.min_n)
    qtype_rows = summarize_group(events, "user_question_type", args.min_n)
    cross_rows = summarize_cross(events, args.min_n)

    write_csv(outdir / "topic_curiosity_rates.csv", topic_rows, ["user_topic", "n_real_questions", "n_curiosity", "n_not_curiosity", "curiosity_rate_pct", "top_curiosity_type", "top_curiosity_type_count", "avg_prev_user_words", "avg_prev_user_epistemic_markers", "prev_user_has_question_pct"])
    write_csv(outdir / "user_question_type_curiosity_rates.csv", qtype_rows, ["user_question_type", "n_real_questions", "n_curiosity", "n_not_curiosity", "curiosity_rate_pct", "top_curiosity_type", "top_curiosity_type_count", "avg_prev_user_words", "avg_prev_user_epistemic_markers", "prev_user_has_question_pct"])
    write_csv(outdir / "topic_by_user_question_type_curiosity_rates.csv", cross_rows, ["user_topic", "user_question_type", "n_real_questions", "n_curiosity", "curiosity_rate_pct", "top_curiosity_type", "top_curiosity_type_count"])
    write_examples(outdir / "curiosity_examples_by_user_topic_type.csv", events)

    with (outdir / "summary.txt").open("w", encoding="utf-8") as f:
        f.write(f"Dataset: {args.dataset}\nInput: {inp}\nTotal question events: {len(events)}\n")
        real = [e for e in events if e["llm_keep"] and e["llm_status"] not in {"unclear", "problematic"}]
        cur = [e for e in real if e["llm_status"] == "curiosity"]
        f.write(f"Real classified assistant questions: {len(real)}\nCuriosity questions: {len(cur)} ({len(cur)/max(len(real),1)*100:.2f}%)\n\n")
        f.write("Top user topics by curiosity rate:\n")
        for r in topic_rows[:12]:
            f.write(f"  {r['user_topic']}: {r['curiosity_rate_pct']:.2f}% ({r['n_curiosity']}/{r['n_real_questions']}), top={r['top_curiosity_type']}\n")
        f.write("\nTop user question/request types by curiosity rate:\n")
        for r in qtype_rows[:12]:
            f.write(f"  {r['user_question_type']}: {r['curiosity_rate_pct']:.2f}% ({r['n_curiosity']}/{r['n_real_questions']}), top={r['top_curiosity_type']}\n")
        f.write("\nTop topic x user-type cells by curiosity rate:\n")
        for r in cross_rows[:20]:
            f.write(f"  {r['user_topic']} / {r['user_question_type']}: {r['curiosity_rate_pct']:.2f}% ({r['n_curiosity']}/{r['n_real_questions']}), top={r['top_curiosity_type']}\n")

    print("Wrote", outdir)
    if args.home_copy:
        copy_to_home(outdir, args.home_copy)
        print("Copied", args.home_copy)
    print((outdir / "summary.txt").read_text())


if __name__ == "__main__":
    main()
