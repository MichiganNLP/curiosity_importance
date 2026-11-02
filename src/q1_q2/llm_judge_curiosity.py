#!/usr/bin/env python3
"""
Use an OpenAI LLM judge to classify WildChat assistant-question events.

Input should be the JSONL produced by curiosity_mapping.py. The judge receives
the previous user turn, assistant turn, extracted question, and next user turn.
It returns a strict JSON label with explanations for the status/type decision.

Environment:
  export OPENAI_API_KEY=...

Example:
  python3 src/q1_q2/llm_judge_curiosity.py \
    --input outputs/question_events.jsonl \
    --output-jsonl outputs/llm_judged.jsonl \
    --output-csv outputs/llm_judged.csv \
    --start 0 \
    --limit 200
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "outputs"
DEFAULT_INPUT = OUTPUT_DIR / "question_events.jsonl"
DEFAULT_OUTPUT_JSONL = OUTPUT_DIR / "llm_judged.jsonl"
DEFAULT_OUTPUT_CSV = OUTPUT_DIR / "llm_judged.csv"
DEFAULT_FAILED_JSONL = OUTPUT_DIR / "llm_judge_failed_batches.jsonl"


STATUS_VALUES = ["curiosity", "not_curiosity", "problematic", "unclear"]
QUESTION_TYPES = [
    "causal_explanatory",
    "counterfactual_hypothetical",
    "exploratory_broadening",
    "perspective_taking",
    "reflective_metacognitive",
    "information_request",
    "clarification_repair",
    "task_management",
    "engagement_maintenance",
    "privacy_probing",
    "leading_steering",
    "unknown_question",
]
ARTIFACT_TYPES = [
    "none",
    "generated_question_content",
    "roleplay_or_dialogue",
    "quiz_or_game_mechanic",
    "example_or_template",
    "generated_writing",
    "embedded_curiosity_content",
    "roleplay_curiosity",
    "malformed_fragment",
]


JUDGE_SCHEMA = {
    "name": "curiosity_question_judgment",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "keep": {
                "type": "boolean",
                "description": "True only if this is a real assistant-to-user question, not generated content.",
            },
            "artifact_type": {
                "type": "string",
                "enum": ARTIFACT_TYPES,
                "description": "If keep is false, the reason the question is excluded; otherwise none.",
            },
            "status": {
                "type": "string",
                "enum": STATUS_VALUES,
                "description": "Top-level category for the retained question.",
            },
            "question_type": {
                "type": "string",
                "enum": QUESTION_TYPES,
                "description": "Specific function/type of the question.",
            },
            "confidence": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
                "description": "Judge confidence in the label.",
            },
            "type_explanation": {
                "type": "string",
                "description": "One concise sentence explaining why this question_type was chosen.",
            },
            "artifact_explanation": {
                "type": "string",
                "description": "One concise sentence explaining keep/artifact_type.",
            },
            "curiosity_explanation": {
                "type": "string",
                "description": "One concise sentence explaining whether the question opens/deepens an epistemic gap.",
            },
        },
        "required": [
            "keep",
            "artifact_type",
            "status",
            "question_type",
            "confidence",
            "type_explanation",
            "artifact_explanation",
            "curiosity_explanation",
        ],
    },
}

BATCH_JUDGE_SCHEMA = {
    "name": "curiosity_question_batch_judgment",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "id": {"type": "integer"},
                        "keep": JUDGE_SCHEMA["schema"]["properties"]["keep"],
                        "artifact_type": JUDGE_SCHEMA["schema"]["properties"]["artifact_type"],
                        "status": JUDGE_SCHEMA["schema"]["properties"]["status"],
                        "question_type": JUDGE_SCHEMA["schema"]["properties"]["question_type"],
                        "confidence": JUDGE_SCHEMA["schema"]["properties"]["confidence"],
                        "type_explanation": JUDGE_SCHEMA["schema"]["properties"]["type_explanation"],
                        "artifact_explanation": JUDGE_SCHEMA["schema"]["properties"]["artifact_explanation"],
                        "curiosity_explanation": JUDGE_SCHEMA["schema"]["properties"]["curiosity_explanation"],
                    },
                    "required": [
                        "id",
                        "keep",
                        "artifact_type",
                        "status",
                        "question_type",
                        "confidence",
                        "type_explanation",
                        "artifact_explanation",
                        "curiosity_explanation",
                    ],
                },
            }
        },
        "required": ["items"],
    },
}


SYSTEM_PROMPT = """You are an expert annotator for human-LLM conversation research.

Task: classify assistant questions from WildChat conversations. You may receive one item or a batch of items. Return one judgment for every item id.

First decide whether the extracted question is truly the assistant asking the user a question in the live conversation.
The main analysis is about interactional model curiosity: assistant-to-user questions that affect the epistemic state of the current human-LM interaction.
Do NOT keep questions that are generated content requested by the user, roleplay/dialogue,
quiz/game mechanics, examples/templates, generated writing, embedded curiosity-like content, or malformed fragments.

Top-level status:
- curiosity: the assistant asks a question that opens, deepens, examines, or resolves an epistemic gap.
- not_curiosity: the assistant asks a useful task/conversation-management question, but it does not deepen epistemic inquiry.
- problematic: the assistant question probes sensitive information or steers the user toward a belief/action/disclosure.
- unclear: insufficient context or ambiguous function.

Question types and definitions:
- causal_explanatory: asks why/how something happens, works, is caused, or can be explained.
- counterfactual_hypothetical: asks what-if, suppose, imagine, or alternative-world questions.
- exploratory_broadening: asks about alternatives, possibilities, implications, extensions, or broader exploration.
- perspective_taking: asks about another person/group/stakeholder/audience/culture/viewpoint.
- reflective_metacognitive: asks about assumptions, evidence, reasoning, confidence, bias, goals, values, criteria, tradeoffs, or the user's own thinking process.
- information_request: asks for missing facts/context/details needed to continue; not primarily epistemic exploration.
- clarification_repair: asks the user to clarify ambiguous wording or an underspecified request.
- task_management: asks about output format, style, length, tone, audience, workflow, or constraints.
- engagement_maintenance: generic continuation/helpfulness question, e.g. would you like me to continue, how can I help, anything else.
- privacy_probing: asks for sensitive personal information, including medical, financial, location, identity, political/religious, or private details.
- leading_steering: question form pushes the user toward a specific answer, belief, action, or disclosure.
- unknown_question: use only if no other type fits.

Artifact types:
- none: keep the question.
- generated_question_content: user asked the assistant to generate questions, FAQs, things to ask, survey items, prompts, etc.
- roleplay_or_dialogue: question belongs to a fictional character, script, scene, dialogue, or roleplay.
- quiz_or_game_mechanic: question is part of a quiz, exam, test, trivia, guessing game, or task mechanic.
- example_or_template: question appears inside an example, sample, template, checklist, form, or placeholder.
- generated_writing: question is rhetorical or textual content inside an article, story, post, tweet, flirty/composed response, or other generated document.
- embedded_curiosity_content: generated text contains a curiosity-like question, but the assistant is not asking the user in the live interaction.
- roleplay_curiosity: a fictional/persona/roleplay character asks a curiosity-like question inside the generated scene or roleplay.
- malformed_fragment: fragment or punctuation artifact, not a real question.

Rules:
- A question mark alone is not enough.
- "Would you like me to continue?" is not_curiosity / engagement_maintenance.
- "What topic should this be about?" is usually not_curiosity / information_request or task_management.
- "What assumptions are you making?" is curiosity / reflective_metacognitive.
- "How might another stakeholder view this?" is curiosity / perspective_taking.
- A character asking "What if I cannot handle it?" inside a story is roleplay_curiosity, not interactional curiosity.
- A witty/flirty/composed response that ends in a question is generated_writing unless the assistant is genuinely asking the user for information.
- If the user explicitly asks the assistant to generate or ask questions, mark the generated questions as generated_question_content unless the question is a live clarification needed to continue the interaction.
- If keep is false, artifact_type MUST NOT be none, status MUST be unclear, and question_type MUST be unknown_question.
- If keep is true, artifact_type MUST be none.
- Keep explanations short and concrete.

Examples:

1. Previous user: "Give me 20 questions about astronomy."
   Assistant question: "What is a black hole?"
   Label: keep=false, artifact_type=generated_question_content, status=unclear,
   question_type=unknown_question.
   Reason: the assistant is generating requested question content, not asking the user.

2. Previous user: "Can you help me think through this argument?"
   Assistant question: "What assumptions are you making?"
   Label: keep=true, artifact_type=none, status=curiosity,
   question_type=reflective_metacognitive.
   Reason: the question asks the user to examine assumptions behind their reasoning.

3. Previous user: "Write this as a table."
   Assistant question: "Do you want this as bullets or a table?"
   Label: keep=true, artifact_type=none, status=not_curiosity,
   question_type=task_management.
   Reason: the question asks about output format, not epistemic inquiry.

4. Previous user: "I think this policy is unfair."
   Assistant question: "How might another stakeholder view this policy?"
   Label: keep=true, artifact_type=none, status=curiosity,
   question_type=perspective_taking.
   Reason: the question asks the user to consider another viewpoint.

5. Previous user: "Explain this topic."
   Assistant question: "Would you like me to continue?"
   Label: keep=true, artifact_type=none, status=not_curiosity,
   question_type=engagement_maintenance.
   Reason: the question is a generic continuation prompt.

6. Previous user: "Write a dramatic scene."
   Assistant question: "Character: What should we do now?"
   Label: keep=false, artifact_type=roleplay_or_dialogue, status=unclear,
   question_type=unknown_question.
   Reason: the question belongs to fictional dialogue, not the assistant-user interaction.

7. Previous user: "Write a scene where a character doubts herself."
   Assistant question: "What if I cannot handle it?"
   Label: keep=false, artifact_type=roleplay_curiosity, status=unclear,
   question_type=unknown_question.
   Reason: the question is curiosity-like, but it is performed by a character inside generated roleplay, not asked by the assistant to the user.

8. Previous user: "Write a funny, flirty response to: let me know when and where we can meet."
   Assistant question: "Should we meet at the library and flirt over intellectual banter?"
   Label: keep=false, artifact_type=generated_writing, status=unclear,
   question_type=unknown_question.
   Reason: the question is part of a composed response requested by the user, not an assistant-to-user inquiry.
"""


CSV_FIELDS = [
    "conversation_id",
    "model",
    "question_text",
    "regex_primary_function",
    "regex_curiosity_status",
    "regex_curiosity_subtype",
    "llm_keep",
    "llm_artifact_type",
    "llm_status",
    "llm_question_type",
    "llm_confidence",
    "llm_type_explanation",
    "llm_artifact_explanation",
    "llm_curiosity_explanation",
    "previous_user_text",
    "assistant_turn_text",
    "next_user_text",
]


def truncate(value, max_chars):
    value = "" if value is None else str(value)
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 3] + "..."


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def output_key(row):
    return (
        str(row.get("conversation_id", "")),
        str(row.get("assistant_turn_index", "")),
        str(row.get("question_index_in_turn", "")),
        str(row.get("question_text", "")),
    )


def load_done_keys(path):
    done = set()
    if not path.exists():
        return done
    for row in load_jsonl(path):
        done.add(output_key(row))
    return done


def build_item_payload(row, item_id, max_context_chars):
    return {
        "id": item_id,
        "previous_user_text": truncate(row.get("previous_user_text", ""), max_context_chars),
        "assistant_turn_text": truncate(row.get("assistant_turn_text", ""), max_context_chars),
        "question_text": row.get("question_text", ""),
        "next_user_text": truncate(row.get("next_user_text", ""), max_context_chars),
        "current_regex_labels": {
            "primary_function": row.get("primary_function", ""),
            "curiosity_status": row.get("curiosity_status", ""),
            "curiosity_subtype": row.get("curiosity_subtype", ""),
        },
    }


def build_batch_prompt(rows, max_context_chars):
    payload = {
        "instructions": "Classify every item independently. Return exactly one judgment per item id.",
        "items": [build_item_payload(row, i, max_context_chars) for i, row in enumerate(rows)],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def normalize_batch_judgments(judgments):
    for judgment in judgments:
        keep = judgment.get("keep")
        artifact_type = judgment.get("artifact_type")
        status = judgment.get("status")
        question_type = judgment.get("question_type")

        if keep is True and artifact_type != "none":
            judgment["keep"] = False
            judgment["status"] = "unclear"
            judgment["question_type"] = "unknown_question"
            judgment["artifact_explanation"] = (
                judgment.get("artifact_explanation", "")
                + " Normalized: non-none artifact_type means the row is excluded."
            ).strip()
        elif keep is False and artifact_type == "none":
            # Most often this means the judge intended to keep a not-curiosity row.
            judgment["keep"] = True
            judgment["artifact_explanation"] = (
                judgment.get("artifact_explanation", "")
                + " Normalized: artifact_type=none means the row is retained."
            ).strip()

        if judgment.get("keep") is False:
            judgment["status"] = "unclear"
            judgment["question_type"] = "unknown_question"

    return judgments


def validate_batch_judgments(judgments):
    for judgment in judgments:
        keep = judgment.get("keep")
        artifact_type = judgment.get("artifact_type")
        status = judgment.get("status")
        question_type = judgment.get("question_type")
        if keep is True and artifact_type != "none":
            raise ValueError("Invalid judgment after normalization: keep=true requires artifact_type=none")
        if keep is False:
            if artifact_type == "none":
                raise ValueError("Invalid judgment after normalization: keep=false requires a non-none artifact_type")
            if status != "unclear" or question_type != "unknown_question":
                raise ValueError("Invalid judgment after normalization: keep=false requires unclear/unknown_question")


def call_openai_batch(client, model, rows, max_context_chars, max_retries, single_fallback=True):
    prompt = build_batch_prompt(rows, max_context_chars)
    expected_ids = set(range(len(rows)))
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                response_format={
                    "type": "json_schema",
                    "json_schema": BATCH_JUDGE_SCHEMA,
                },
            )
            content = response.choices[0].message.content
            parsed = json.loads(content)
            items = parsed.get("items", [])
            by_id = {item.get("id"): item for item in items}
            if set(by_id) != expected_ids:
                raise ValueError("LLM response ids did not match input batch ids")
            judgments = [by_id[i] for i in range(len(rows))]
            judgments = normalize_batch_judgments(judgments)
            validate_batch_judgments(judgments)
            return judgments
        except Exception as exc:
            last_error = exc
            if attempt >= max_retries:
                if single_fallback and len(rows) > 1:
                    print("Batch failed after retries; falling back to single-row calls:", exc, file=sys.stderr)
                    return call_openai_single_fallback(client, model, rows, max_context_chars, max_retries)
                raise
            time.sleep(2 ** attempt)
    raise last_error


def call_openai_single_fallback(client, model, rows, max_context_chars, max_retries):
    judgments = []
    for row in rows:
        single = call_openai_batch(client, model, [row], max_context_chars, max_retries, single_fallback=False)
        judgments.append(single[0])
    return judgments


def write_csv(jsonl_path, csv_path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in load_jsonl(jsonl_path):
            judge = row.get("llm_judge", {})
            writer.writerow(
                {
                    "conversation_id": row.get("conversation_id", ""),
                    "model": row.get("model", ""),
                    "question_text": row.get("question_text", ""),
                    "regex_primary_function": row.get("primary_function", ""),
                    "regex_curiosity_status": row.get("curiosity_status", ""),
                    "regex_curiosity_subtype": row.get("curiosity_subtype", ""),
                    "llm_keep": judge.get("keep", ""),
                    "llm_artifact_type": judge.get("artifact_type", ""),
                    "llm_status": judge.get("status", ""),
                    "llm_question_type": judge.get("question_type", ""),
                    "llm_confidence": judge.get("confidence", ""),
                    "llm_type_explanation": judge.get("type_explanation", ""),
                    "llm_artifact_explanation": judge.get("artifact_explanation", ""),
                    "llm_curiosity_explanation": judge.get("curiosity_explanation", ""),
                    "previous_user_text": truncate(row.get("previous_user_text", ""), 1200),
                    "assistant_turn_text": truncate(row.get("assistant_turn_text", ""), 1200),
                    "next_user_text": truncate(row.get("next_user_text", ""), 1200),
                }
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-jsonl", type=Path, default=DEFAULT_OUTPUT_JSONL)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--start", type=int, default=0, help="Zero-based input row offset for batch runs")
    parser.add_argument("--limit", type=int, default=None, help="Maximum input rows to judge after --start")
    parser.add_argument("--batch-size", type=int, default=10, help="Number of rows per OpenAI API request")
    parser.add_argument("--max-context-chars", type=int, default=2500)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--failed-jsonl", type=Path, default=DEFAULT_FAILED_JSONL, help="Rows from batches that still fail after retries/fallback")
    parser.add_argument("--skip-failed-batches", action="store_true", help="Skip failed batches instead of aborting the run")
    parser.add_argument("--no-single-fallback", action="store_true", help="After a multi-row batch fails retries, skip/log the whole batch instead of falling back to single-row calls. Use with --skip-failed-batches for faster large runs.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--csv-only", action="store_true")
    args = parser.parse_args()

    if args.csv_only:
        write_csv(args.output_jsonl, args.output_csv)
        print("Wrote CSV:", args.output_csv)
        return

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is not set.", file=sys.stderr)
        sys.exit(2)

    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: Python package 'openai' is not installed. Try: pip install openai", file=sys.stderr)
        sys.exit(2)

    client = OpenAI()
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.failed_jsonl.parent.mkdir(parents=True, exist_ok=True)
    done = load_done_keys(args.output_jsonl) if args.resume else set()
    if args.resume and args.skip_failed_batches and args.failed_jsonl.exists():
        failed_done = load_done_keys(args.failed_jsonl)
        done |= failed_done
        print("Resume: also skipping", len(failed_done), "rows already present in failed JSONL")
    mode = "a" if args.resume else "w"

    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")

    n_seen = 0
    n_after_start = 0
    n_written = 0
    pending = []

    n_failed = 0

    def write_failed_batch(rows, exc):
        nonlocal n_failed
        with args.failed_jsonl.open("a", encoding="utf-8") as failed_out:
            for row in rows:
                row = dict(row)
                row["llm_judge_error"] = str(exc)
                failed_out.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_failed += 1

    def flush_batch(out_file):
        nonlocal n_written, pending
        if not pending:
            return
        try:
            judgments = call_openai_batch(
                client=client,
                model=args.model,
                rows=pending,
                max_context_chars=args.max_context_chars,
                max_retries=args.max_retries,
                single_fallback=not args.no_single_fallback,
            )
        except Exception as exc:
            if not args.skip_failed_batches:
                raise
            print("Skipping failed batch of", len(pending), "rows:", exc, file=sys.stderr)
            write_failed_batch(pending, exc)
            pending = []
            return

        for row, judgment in zip(pending, judgments):
            judgment.pop("id", None)
            row["llm_judge"] = judgment
            out_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_written += 1
        out_file.flush()
        pending = []
        if n_written % 50 == 0:
            print("Judged", n_written, "new rows")

    with args.output_jsonl.open(mode, encoding="utf-8") as out:
        for input_index, row in enumerate(load_jsonl(args.input)):
            if input_index < args.start:
                continue
            if args.limit is not None and n_after_start >= args.limit:
                break
            n_seen += 1
            n_after_start += 1
            if output_key(row) in done:
                continue

            pending.append(row)
            if len(pending) >= args.batch_size:
                flush_batch(out)

        flush_batch(out)

    write_csv(args.output_jsonl, args.output_csv)
    print("Input start offset:", args.start)
    print("Input rows considered:", n_seen)
    print("Batch size:", args.batch_size)
    print("New rows judged:", n_written)
    print("Rows skipped after failures:", n_failed)
    if n_failed:
        print("Wrote failed rows:", args.failed_jsonl)
    print("Wrote JSONL:", args.output_jsonl)
    print("Wrote CSV:", args.output_csv)


if __name__ == "__main__":
    main()
