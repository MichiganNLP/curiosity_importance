#!/usr/bin/env python3
"""LLM refine and filter Q3 launch-quality ask-before-answer stimuli."""
from __future__ import annotations
import argparse, csv, json, os, re, time
from pathlib import Path
from openai import OpenAI

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO_ROOT / 'outputs' / 'q3_launch_quality_study' / 'q3_launch_quality_items_125_seed.csv'
DEFAULT_OUTPUT = REPO_ROOT / 'outputs' / 'q3_launch_quality_study' / 'q3_launch_quality_items_125_llm_refined.csv'
TARGET_TYPES = ['information_gap_curiosity','causal_explanatory','exploratory_broadening','reflective_metacognitive','perspective_taking','counterfactual_hypothetical']

SYSTEM = """You are helping design a Prolific/Qualtrics human-subjects study about when an AI assistant should ask a useful curiosity-oriented question before answering. You are not answering the user. You are writing and quality-checking candidate assistant next moves for a survey."""

USER_TEMPLATE = """Participants will see a USER MESSAGE and compare three possible assistant next moves.

Choice A is an existing direct answer. Do NOT rewrite Choice A.
Choice B should be a related but non-curiosity question before answering.
Choice C should be a curiosity-oriented question before answering.
The draft B/C questions are only rough placeholders. Rewrite them from scratch when needed. Do not reject an otherwise good item merely because the draft B or C is malformed, generic, or awkward.

The survey is testing whether asking before answering helps. Therefore Choice C should only be kept when the question is actually useful in context, not merely interesting.

STUDY ROLE:
{study_role}

USER MESSAGE:
{user_message}

EXISTING DIRECT ANSWER, CHOICE A:
{choice_a}

TARGET CURIOSITY TYPE FOR CHOICE C:
{target_curiosity_type}

DRAFT CHOICE B:
{draft_b}

DRAFT CHOICE C:
{draft_c}

Rules for keeping/rejecting:
- Reject urgent, safety-sensitive, medical, legal, sexual, toxic, roleplay, fiction, malformed, overly long, or hidden-context-dependent items.
- Reject if the user clearly needs an immediate direct answer and asking first would be annoying or unsafe.
- Reject if Choice A is too long or unreadable for Prolific.
- Reject if the USER MESSAGE itself is unsuitable, or if no high-quality Choice C can be written in under 26 words after rewriting from scratch.
- For study_role useful_curiosity_candidate: keep only if asking first could plausibly improve the final answer by clarifying goals, constraints, criteria, assumptions, audience, stakes, causes, alternatives, or intended use.
- For study_role curiosity_not_needed_contrast or manual_review_contrast: keep only if it is a safe case where a relevant curiosity question can be written, but the direct answer is plausibly sufficient.

Choice B requirements:
- Related to the user message.
- Asks about practical task parameters only: format, length, level, environment, scope, constraints.
- Does not invite reflection, explanation, perspective-taking, alternatives, or deeper reasoning.
- Not generic. Avoid: "Would you like more detail?", "What format should I use?" unless it names the topic-specific format.

Choice C requirements:
- Directly names the user's topic using natural wording, not keyword fragments.
- Is useful before answering.
- Is not generic or templated.
- Does not ask the user to do the assistant's job.
- Should sound like a plausible assistant question in a real conversation, not a rubric template.
- Does not merely ask whether they want a brief or detailed answer.
- Should match the target type when possible:
  * information_gap_curiosity: asks for a missing detail/constraint that would change the answer.
  * causal_explanatory: asks about a reason, mechanism, diagnosis, or explanatory contrast.
  * exploratory_broadening: asks about implications, alternatives, applications, or next steps.
  * reflective_metacognitive: asks about goals, assumptions, evidence, priorities, tradeoffs, or uncertainty.
  * perspective_taking: asks about audience, stakeholder, reader, customer, learner, or affected person concerns.
  * counterfactual_hypothetical: asks about a changed condition or what-if scenario.

Return JSON only:
{{
  "keep": true/false,
  "launch_role": "useful_curiosity_candidate|curiosity_not_needed_contrast|reject",
  "reject_reason": "none or short reason",
  "choice_B_related_noncuriosity_question": "... ?",
  "choice_C_curiosity_question": "... ?",
  "choice_C_type": "information_gap_curiosity|causal_explanatory|exploratory_broadening|reflective_metacognitive|perspective_taking|counterfactual_hypothetical",
  "why_C_is_or_is_not_useful": "short explanation"
}}
"""

FORBIDDEN = ['would you like me to explain more','what part feels unclear','what would you like to explore','what level of detail','is there anything else','what format should i use']

def read_csv(path: Path) -> list[dict]:
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))
def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader(); w.writerows(rows)
def flatten(s: str) -> str:
    return re.sub(r'\s+', ' ', str(s or '')).strip()
def is_true(v) -> bool:
    return str(v).lower() in {'true','1','yes'}

def call_model(client: OpenAI, model: str, row: dict, max_retries: int) -> dict:
    prompt = USER_TEMPLATE.format(
        study_role=row.get('study_role',''),
        user_message=row.get('user_message','')[:2200],
        choice_a=row.get('choice_A_direct_answer','')[:2500],
        target_curiosity_type=row.get('target_curiosity_type',''),
        draft_b=row.get('choice_B_related_noncuriosity_question',''),
        draft_c=row.get('choice_C_curiosity_question',''),
    )
    last = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0.25,
                response_format={'type':'json_object'},
                messages=[{'role':'system','content':SYSTEM},{'role':'user','content':prompt}],
            )
            obj = json.loads(resp.choices[0].message.content)
            b = flatten(obj.get('choice_B_related_noncuriosity_question',''))
            c = flatten(obj.get('choice_C_curiosity_question',''))
            if b and not b.endswith('?'): b += '?'
            if c and not c.endswith('?'): c += '?'
            keep = bool(obj.get('keep'))
            if keep and (not b or not c):
                raise ValueError('keep=true but missing B/C')
            low = ' '.join([b.lower(), c.lower()])
            if keep and any(x in low for x in FORBIDDEN):
                raise ValueError('generic forbidden phrase')
            ctype = obj.get('choice_C_type','')
            if ctype not in TARGET_TYPES:
                ctype = row.get('target_curiosity_type','')
            role = obj.get('launch_role','')
            if role not in {'useful_curiosity_candidate','curiosity_not_needed_contrast','reject'}:
                role = 'reject' if not keep else row.get('study_role','useful_curiosity_candidate')
            return {
                'llm_keep': keep,
                'llm_launch_role': role,
                'llm_reject_reason': flatten(obj.get('reject_reason','none')),
                'choice_B_related_noncuriosity_question': b,
                'choice_C_curiosity_question': c,
                'llm_choice_C_type': ctype,
                'llm_choice_C_rationale': flatten(obj.get('why_C_is_or_is_not_useful','')),
                'llm_generation_error': '',
            }
        except Exception as exc:
            last = exc
            time.sleep(2 ** attempt)
    return {'llm_keep': False,'llm_launch_role':'reject','llm_reject_reason':'generation_failed','choice_B_related_noncuriosity_question':row.get('choice_B_related_noncuriosity_question',''),'choice_C_curiosity_question':row.get('choice_C_curiosity_question',''),'llm_choice_C_type':row.get('target_curiosity_type',''),'llm_choice_C_rationale':'','llm_generation_error':str(last)}

def write_loop_files(rows: list[dict], outdir: Path, n_blocks: int) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    for p in outdir.glob('*.tsv'):
        p.unlink()
    sizes = [len(rows)//n_blocks]*n_blocks
    for i in range(len(rows)%n_blocks): sizes[i] += 1
    fields = ['Field 1 item_id','Field 2 user_message','Field 3 choice_A_direct_answer','Field 4 choice_B_related_noncuriosity_question','Field 5 choice_C_curiosity_question','Field 6 target_curiosity_type','dataset','study_role','example_order']
    idx = 0
    for bnum, size in enumerate(sizes, 1):
        block = rows[idx:idx+size]; idx += size
        with (outdir / f'block_{bnum:02d}_loop_merge_paste_ready.tsv').open('w', encoding='utf-8', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields, delimiter='\t', lineterminator='\n')
            w.writeheader()
            for j, r in enumerate(block, 1):
                w.writerow({'Field 1 item_id':r.get('item_id',''),'Field 2 user_message':flatten(r.get('user_message','')),'Field 3 choice_A_direct_answer':flatten(r.get('choice_A_direct_answer','')),'Field 4 choice_B_related_noncuriosity_question':flatten(r.get('choice_B_related_noncuriosity_question','')),'Field 5 choice_C_curiosity_question':flatten(r.get('choice_C_curiosity_question','')),'Field 6 target_curiosity_type':r.get('llm_choice_C_type') or r.get('target_curiosity_type',''),'dataset':r.get('dataset',''),'study_role':r.get('llm_launch_role') or r.get('study_role',''),'example_order':j})

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    ap.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument('--model', default='gpt-5.5')
    ap.add_argument('--limit', type=int)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--max-retries', type=int, default=4)
    ap.add_argument('--n-blocks', type=int, default=25)
    args = ap.parse_args()
    if not os.environ.get('OPENAI_API_KEY'):
        raise SystemExit('ERROR: OPENAI_API_KEY is not set')
    rows = read_csv(args.input)
    if args.limit: rows = rows[:args.limit]
    existing = {}
    if args.resume and args.output.exists():
        for r in read_csv(args.output): existing[r.get('item_id')] = r
    client = OpenAI(); out_rows = []
    for idx, row in enumerate(rows, 1):
        if row.get('item_id') in existing and not existing[row.get('item_id')].get('llm_generation_error'):
            out_rows.append(existing[row.get('item_id')]); continue
        new = dict(row); new.update(call_model(client, args.model, row, args.max_retries)); out_rows.append(new)
        if idx % 10 == 0:
            print(f'Processed {idx} rows')
            write_csv(args.output, out_rows, list(out_rows[0].keys()))
    fields = list(out_rows[0].keys()) if out_rows else []
    write_csv(args.output, out_rows, fields)
    kept = [r for r in out_rows if is_true(r.get('llm_keep')) and r.get('llm_launch_role') != 'reject']
    kept_path = args.output.with_name(args.output.stem + '_kept.csv')
    write_csv(kept_path, kept, fields)
    write_loop_files(kept, args.output.parent / 'loop_merge_25_blocks_tsv_paste_ready_llm_refined', args.n_blocks)
    print('Wrote refined CSV:', args.output)
    print('Wrote kept CSV:', kept_path)
    print('Kept rows:', len(kept), 'of', len(out_rows))
    print('Wrote Loop & Merge TSVs:', args.output.parent / 'loop_merge_25_blocks_tsv_paste_ready_llm_refined')
if __name__ == '__main__': main()
