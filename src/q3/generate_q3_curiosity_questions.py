#!/usr/bin/env python3
"""Generate non-generic Q3 curiosity follow-up questions for Prolific stimuli.

Input: q3_preference_study_seed_60.csv
Output: same rows with option_curiosity_followup_llm filled.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

from openai import OpenAI

SYSTEM = """You are helping design a human-subjects study about useful curiosity in AI responses. Your task is to write one specific follow-up question that an assistant could add after its response. The question should make the user more curious or help them think more deeply, without being generic, annoying, invasive, or off-task."""

USER_TEMPLATE = """Write ONE curiosity-oriented follow-up question for this interaction.

Target curiosity type: {target_curiosity_type}

User message:
{user_message}

Assistant response:
{assistant_response}

Weak draft question, if any:
{draft_question}

Rules:
- Make the question specific to the user's actual goal/context.
- Do NOT use generic wording like "What part is unclear?", "Would you like more details?", "What else would you like to explore?", or "What aspect would be most useful?".
- Do NOT ask for unnecessary private information.
- Do NOT derail the user from the task.
- The question should match the target curiosity type whenever possible:
  * information_gap_curiosity: ask for a specific missing detail that matters for the task.
  * causal_explanatory: ask why/how something happens or what mechanism explains it.
  * exploratory_broadening: ask about adjacent angles, implications, alternatives, or future directions.
  * reflective_metacognitive: ask the user to reflect on goals, assumptions, uncertainty, or priorities.
  * perspective_taking: ask about a stakeholder, audience, reader, user, or alternative viewpoint.
  * counterfactual_hypothetical: ask a what-if or alternative-scenario question.
- Keep it under 25 words.
- Output JSON only: {{"curiosity_question": "...", "curiosity_type": "information_gap_curiosity|causal_explanatory|exploratory_broadening|reflective_metacognitive|perspective_taking|counterfactual_hypothetical", "rationale": "short reason"}}
"""

FORBIDDEN = [
    "what part is unclear",
    "what aspect of this",
    "would you like more",
    "do you want more",
    "would you like me",
    "is there anything else",
    "what would you like to explore",
    "what part would you like clarified",
]


def read_csv(path: Path):
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows(rows)


def call_model(client, model, row, max_retries=4):
    prompt = USER_TEMPLATE.format(
        user_message=row.get('user_message','')[:2500],
        assistant_response=row.get('assistant_response','')[:3500],
        draft_question=row.get('option_curiosity_followup_draft',''),
        target_curiosity_type=row.get('target_curiosity_type') or row.get('recommended_curiosity_type') or 'unspecified',
    )
    last = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0.4,
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
            )
            text = resp.choices[0].message.content
            obj = json.loads(text)
            q = str(obj.get('curiosity_question','')).strip()
            qlow = q.lower()
            if not q.endswith('?'):
                q += '?'
            if len(q.split()) > 30 or any(bad in qlow for bad in FORBIDDEN):
                raise ValueError(f'generic_or_too_long: {q}')
            return {
                'option_curiosity_followup_llm': q,
                'llm_curiosity_type': obj.get('curiosity_type',''),
                'llm_question_rationale': obj.get('rationale',''),
                'llm_generation_error': '',
            }
        except Exception as exc:
            last = exc
            time.sleep(2 ** attempt)
    return {
        'option_curiosity_followup_llm': '',
        'llm_curiosity_type': '',
        'llm_question_rationale': '',
        'llm_generation_error': str(last),
    }


def main():
    ap = argparse.ArgumentParser()
    repo_root = Path(__file__).resolve().parents[2]
    ap.add_argument('--input', type=Path, default=repo_root / 'outputs' / 'q3_preference_study_seed.csv')
    ap.add_argument('--output', type=Path, default=repo_root / 'outputs' / 'q3_preference_study_llm_generated.csv')
    ap.add_argument('--model', default='gpt-5.5')
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args()

    if not os.environ.get('OPENAI_API_KEY'):
        raise SystemExit('ERROR: OPENAI_API_KEY is not set')

    rows = read_csv(args.input)
    if args.limit:
        rows = rows[:args.limit]

    done = {}
    if args.resume and args.output.exists():
        for r in read_csv(args.output):
            if r.get('option_curiosity_followup_llm'):
                done[r['item_id']] = r

    client = OpenAI()
    out_rows = []
    extra = ['option_curiosity_followup_llm', 'llm_curiosity_type', 'llm_question_rationale', 'llm_generation_error']
    fieldnames = list(rows[0].keys()) + [e for e in extra if e not in rows[0]]

    for i, row in enumerate(rows, start=1):
        if row['item_id'] in done:
            out_rows.append(done[row['item_id']])
            continue
        gen = call_model(client, args.model, row)
        row = dict(row)
        row.update(gen)
        out_rows.append(row)
        write_csv(args.output, out_rows, fieldnames)
        print(f"{i}/{len(rows)} {row['item_id']}: {row.get('option_curiosity_followup_llm') or row.get('llm_generation_error')}", flush=True)

    write_csv(args.output, out_rows, fieldnames)
    print('Wrote', args.output)


if __name__ == '__main__':
    main()
