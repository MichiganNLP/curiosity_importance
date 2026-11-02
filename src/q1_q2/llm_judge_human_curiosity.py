#!/usr/bin/env python3
"""
LLM validation judge for downstream human curiosity in Q2 matched pairs.

This script labels the *next user turn* after a model question. It is intended
as a validation layer for the rule-based Q2 outcomes, not as a replacement for
them.

Example:
  python3 src/q1_q2/llm_judge_human_curiosity.py \
    --matched-pairs outputs/wildchat_matched_q2/matched_pairs.csv \
    --dataset WildChat \
    --output-jsonl outputs/wildchat_q2_human_curiosity_judged.jsonl \
    --output-csv outputs/wildchat_q2_human_curiosity_judged.csv \
    --sample-pairs 1000 \
    --batch-size 10 \
    --resume
"""

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path


HUMAN_CURIOSITY_STATUS = [
    "human_curiosity",
    "not_human_curiosity",
    "unclear",
]

HUMAN_CURIOSITY_TYPES = [
    "information_seeking",
    "causal_explanatory",
    "exploratory_broadening",
    "reflective_metacognitive",
    "perspective_taking",
    "counterfactual_hypothetical",
    "none",
]

NON_CURIOSITY_FUNCTIONS = [
    "task_continuation",
    "compliance_or_acknowledgment",
    "format_or_style_request",
    "correction_or_repair",
    "answering_assistant_question",
    "new_unrelated_task",
    "other_non_curiosity",
    "none",
]


BATCH_SCHEMA = {
    "name": "human_curiosity_batch_judgment",
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
                        "status": {
                            "type": "string",
                            "enum": HUMAN_CURIOSITY_STATUS,
                        },
                        "curiosity_type": {
                            "type": "string",
                            "enum": HUMAN_CURIOSITY_TYPES,
                        },
                        "non_curiosity_function": {
                            "type": "string",
                            "enum": NON_CURIOSITY_FUNCTIONS,
                        },
                        "curiosity_intensity": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 3,
                            "description": "0 none, 1 weak, 2 moderate, 3 strong.",
                        },
                        "is_task_continuation_only": {"type": "boolean"},
                        "is_answering_assistant_question_only": {"type": "boolean"},
                        "confidence": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                        },
                        "evidence": {
                            "type": "string",
                            "description": "Short explanation using the user turn.",
                        },
                    },
                    "required": [
                        "id",
                        "status",
                        "curiosity_type",
                        "non_curiosity_function",
                        "curiosity_intensity",
                        "is_task_continuation_only",
                        "is_answering_assistant_question_only",
                        "confidence",
                        "evidence",
                    ],
                },
            }
        },
        "required": ["items"],
    },
}


SYSTEM_PROMPT = """You are an expert annotator for human-LLM conversation research.

Task: judge whether the user's next turn after an assistant question shows human curiosity.

You will receive context:
- previous user turn
- assistant question
- next user turn to label
- whether this came after a curiosity-labeled assistant question or a matched control question

Label ONLY the next user turn. Do not label the assistant question.

Human curiosity means the user shows an epistemic desire to know, understand, explore, explain, reflect, compare perspectives, or consider possibilities.

Status labels:
- human_curiosity: the user asks, explores, reasons, requests explanation, broadens the topic, reflects on assumptions/evidence/goals, asks about perspectives, or raises hypotheticals.
- not_human_curiosity: the user only continues the task, gives requested information, says yes/no/thanks, asks for formatting/style, corrects the assistant, or gives a new command without epistemic exploration.
- unclear: the turn is too short, ambiguous, or impossible to interpret.

Curiosity types:
- information_seeking: asks for facts, definitions, examples, sources, or more information.
- causal_explanatory: asks why/how something happens, works, or can be explained.
- exploratory_broadening: asks about alternatives, implications, extensions, comparisons, or related possibilities.
- reflective_metacognitive: reflects on assumptions, evidence, uncertainty, goals, values, reasoning, criteria, tradeoffs, or own understanding.
- perspective_taking: asks about another person, group, stakeholder, audience, culture, or viewpoint.
- counterfactual_hypothetical: asks what-if, suppose, imagine, or alternative-scenario questions.
- none: no curiosity type applies.

Non-curiosity functions:
- task_continuation: continues the practical task without epistemic exploration.
- compliance_or_acknowledgment: yes/no/okay/thanks or simple acceptance.
- format_or_style_request: asks for formatting, wording, tone, length, style, or presentation.
- correction_or_repair: corrects, repairs, or clarifies prior instructions.
- answering_assistant_question: only provides the information requested by the assistant.
- new_unrelated_task: switches to a different task.
- other_non_curiosity: non-curiosity but none of the above.
- none: use when status is human_curiosity or unclear.

Important rules:
- A question mark alone is not enough.
- If the user merely answers the assistant's question, mark not_human_curiosity unless they also ask/explore/reflect.
- If the user says "yes", "continue", "make it shorter", "write it again", or gives a missing detail, mark not_human_curiosity.
- If the user asks for why/how, alternatives, implications, evidence, assumptions, perspectives, or hypotheticals, mark human_curiosity.
- If the user asks the model to generate many questions, label based on the user's intent. If the user is seeking possible inquiry directions, this can be human_curiosity; if it is only a formatting/task request, it is not_human_curiosity.

Return one JSON item for every input id."""


def load_csv(path):
    with path.open("r", encoding="utf-8", newline="") as f:
        yield from csv.DictReader(f)


def load_done_ids(path):
    done = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                done.add(json.loads(line)["validation_id"])
            except Exception:
                continue
    return done


def build_items(rows, dataset, sample_pairs=None, seed=13):
    rows = list(rows)
    if sample_pairs is not None and sample_pairs < len(rows):
        rng = random.Random(seed)
        rows = rng.sample(rows, sample_pairs)

    items = []
    for row in rows:
        pair_id = row.get("pair_id", "")
        topic = row.get("topic_category", "")
        for arm in ["treatment", "control"]:
            validation_id = f"{dataset}:{pair_id}:{arm}"
            items.append(
                {
                    "validation_id": validation_id,
                    "dataset": dataset,
                    "pair_id": pair_id,
                    "arm": arm,
                    "topic_category": topic,
                    "assistant_question_type": row.get(f"{arm}_question_type", ""),
                    "previous_user_text": row.get(f"{arm}_prev_user_text", ""),
                    "assistant_question_text": row.get(f"{arm}_question_text", ""),
                    "next_user_text": row.get(f"{arm}_next_user_text", ""),
                    "rule_next_user_has_question": row.get(f"{arm}_next_user_has_question", ""),
                    "rule_next_user_question_count": row.get(f"{arm}_next_user_question_count", ""),
                    "rule_next_user_exploratory_question_count": row.get(
                        f"{arm}_next_user_exploratory_question_count", ""
                    ),
                    "rule_next_user_epistemic_marker_count": row.get(
                        f"{arm}_next_user_epistemic_marker_count", ""
                    ),
                    "rule_next_user_word_count": row.get(f"{arm}_next_user_word_count", ""),
                }
            )
    return items


def prompt_for_batch(batch):
    compact = []
    for i, item in enumerate(batch):
        compact.append(
            {
                "id": i,
                "validation_id": item["validation_id"],
                "dataset": item["dataset"],
                "arm": item["arm"],
                "assistant_question_type": item["assistant_question_type"],
                "previous_user_turn": item["previous_user_text"][:1200],
                "assistant_question": item["assistant_question_text"][:700],
                "next_user_turn_to_label": item["next_user_text"][:1200],
            }
        )
    return "Label these next-user turns:\n" + json.dumps(compact, ensure_ascii=False, indent=2)


def call_openai_batch(client, model, batch, temperature=0):
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_for_batch(batch)},
        ],
        response_format={"type": "json_schema", "json_schema": BATCH_SCHEMA},
    )
    data = json.loads(response.choices[0].message.content)
    items = data["items"]
    if len(items) != len(batch):
        raise ValueError(f"Expected {len(batch)} judgments, got {len(items)}")
    by_id = {x["id"]: x for x in items}
    if set(by_id) != set(range(len(batch))):
        raise ValueError("LLM response ids did not match input batch ids")
    return [by_id[i] for i in range(len(batch))]


def write_csv(jsonl_path, csv_path):
    rows = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        return
    fields = [
        "validation_id",
        "dataset",
        "pair_id",
        "arm",
        "topic_category",
        "assistant_question_type",
        "status",
        "curiosity_type",
        "non_curiosity_function",
        "curiosity_intensity",
        "is_task_continuation_only",
        "is_answering_assistant_question_only",
        "confidence",
        "evidence",
        "rule_next_user_has_question",
        "rule_next_user_question_count",
        "rule_next_user_exploratory_question_count",
        "rule_next_user_epistemic_marker_count",
        "rule_next_user_word_count",
        "previous_user_text",
        "assistant_question_text",
        "next_user_text",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matched-pairs", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--failed-jsonl", type=Path)
    parser.add_argument("--sample-pairs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.2)
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)

    from openai import OpenAI

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    if args.failed_jsonl:
        args.failed_jsonl.parent.mkdir(parents=True, exist_ok=True)

    rows = list(load_csv(args.matched_pairs))
    items = build_items(rows, args.dataset, args.sample_pairs, args.seed)
    done = load_done_ids(args.output_jsonl) if args.resume else set()
    items = [x for x in items if x["validation_id"] not in done]

    client = OpenAI()
    new_rows = 0
    with args.output_jsonl.open("a", encoding="utf-8") as out:
        for start in range(0, len(items), args.batch_size):
            batch = items[start : start + args.batch_size]
            try:
                judgments = call_openai_batch(client, args.model, batch)
            except Exception as exc:
                if args.failed_jsonl:
                    with args.failed_jsonl.open("a", encoding="utf-8") as fail:
                        fail.write(
                            json.dumps(
                                {
                                    "error": str(exc),
                                    "validation_ids": [x["validation_id"] for x in batch],
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                print(f"Skipping failed batch of {len(batch)} rows: {exc}", file=sys.stderr)
                continue

            for item, judgment in zip(batch, judgments):
                merged = dict(item)
                merged.update({k: v for k, v in judgment.items() if k != "id"})
                out.write(json.dumps(merged, ensure_ascii=False) + "\n")
                new_rows += 1
            out.flush()
            print(f"Judged {new_rows} new rows")
            time.sleep(args.sleep)

    write_csv(args.output_jsonl, args.output_csv)
    print(f"Wrote JSONL: {args.output_jsonl}")
    print(f"Wrote CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
