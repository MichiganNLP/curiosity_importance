#!/usr/bin/env python3
"""Build a stricter Q3 launch-quality ask-before-answer candidate pool."""
from __future__ import annotations
import argparse, csv, hashlib, json, random, re
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = REPO_ROOT / 'outputs' / 'missed_opportunity_discovery'
DEFAULT_OUTDIR = REPO_ROOT / 'outputs' / 'q3_launch_quality_study'
DATASET_FILES = {
    'wildchat_10k': SOURCE_DIR / 'wildchat_10k_missed_opportunities.jsonl',
    'oasst1': SOURCE_DIR / 'oasst1_missed_opportunities.jsonl',
    'lmsys_50k': SOURCE_DIR / 'lmsys_50k_missed_opportunities.jsonl',
    'sharegpt': SOURCE_DIR / 'sharegpt_missed_opportunities.jsonl',
}
TARGET_TYPES = ['information_gap_curiosity','causal_explanatory','exploratory_broadening','reflective_metacognitive','perspective_taking','counterfactual_hypothetical']
HTML_RE = re.compile(r'<[^>]+>')
UNSAFE_RE = re.compile(r'\b(sex|sexy|sexual|submissive|fantas|cock|penis|vagina|nude|porn|erotic|toxic|harmful|hurt|kill|suicide|self[- ]harm|bomb|weapon|slur|racist|jailbreak)\b', re.I)
URGENT_RE = re.compile(r'\b(coal gas|gas leak|break in|emergency|urgent|bleeding|chest pain|overdose|poison|fire|stuck inside|call 911|legal consequences)\b', re.I)
MEDLEGAL_RE = re.compile(r'\b(nurse|patient|placenta|diagnos|medical|doctor|lawyer|legal agreement|contract|signing agreement|wary of signing)\b', re.I)
ROLEPLAY_RE = re.compile(r'\b(roleplay|pretend you are|act as|fictional|story|screenplay|novel|fanfic|character|anime|scooby|dungeon|adventure|in this universe|as a robot from|mermaid|dragon|bird species that breathes fire)\b', re.I)
CONTEXT_DEP_RE = re.compile(r'^\s*(and what|also|then|so|but|ok|okay|yes|no|it|this|that|these|those|they|he|she|we)\b|\b(above|previous|earlier|same thing|as before|last answer|you said|that one|this one)\b', re.I)
CODE_HEAVY_RE = re.compile(r'```|<\w+>|function\s+\w+\(|def\s+\w+\(|SELECT\s+.+\s+FROM|import\s+\w+', re.I | re.S)
ASSISTANT_CHATTER_RE = re.compile(r'\b(nice to meet you|how should i call you|what should i call you|what is your name|who are you|are you chatgpt|hello assistant|hi assistant)\b', re.I)
TOO_MECHANICAL_RE = re.compile(r'\b(write|draft|rephrase|rewrite|translate|summarize|continue|add to the story|give me \d+|list \d+|hashtags|caption|poem|tweet)\b', re.I)
DIRECT_FACTUAL_RE = re.compile(r'\b(what is|define|difference between|explain the difference|probability|calculate|solve|bash script|python|r code|sql|formula|regex|install|setup|50 word|briefly|concisely)\b', re.I)
USEFUL_PATTERNS = {
    'information_gap_curiosity': re.compile(r'\b(recommend|best|choose|which|suggest|idea for|help me decide|what should i|constraints|requirements|preference|context)\b', re.I),
    'causal_explanatory': re.compile(r'\b(why|how can|how do|what causes|reason|mechanism|diagnose|improve|learn|explain how|best way to learn|understand)\b', re.I),
    'exploratory_broadening': re.compile(r'\b(strategy|plan|proposal|design|product design|project|research|study|business|application|use case|roadmap|brainstorm|idea)\b', re.I),
    'reflective_metacognitive': re.compile(r'\b(should i|decision|tradeoff|priority|goal|criteria|assumption|uncertain|wary|concern|risk|value|ethic|bias)\b', re.I),
    'perspective_taking': re.compile(r'\b(audience|target audience|reader|customer|client|stakeholder|teacher|student|manager|employee|user persona|senior executive)\b', re.I),
    'counterfactual_hypothetical': re.compile(r'\b(what if|suppose|if .* would|if .* could|if .* might|alternative scenario|changed condition)\b', re.I),
}
TYPE_FROM_OPPORTUNITY = {'uncertainty_gap':'information_gap_curiosity','reasoning_claim':'causal_explanatory','learning_explanation':'causal_explanatory','learning_or_explanation':'causal_explanatory','advice_dilemma':'reflective_metacognitive','advice_or_dilemma':'reflective_metacognitive','creative_or_research_framing':'exploratory_broadening'}
STOP = {'please','could','would','should','explain','write','create','give','tell','show','make','help','about','with','what','which','when','where','how','why','the','and','for','that','this','from','into','your','you','can','need','want','using','used','more','some','best','good','like'}

def flatten(text: str) -> str:
    return re.sub(r'\s+', ' ', HTML_RE.sub(' ', str(text or ''))).strip()
def word_count(text: str) -> int:
    return len(re.findall(r'\b\w+\b', str(text or '')))
def stable_id(prefix: str, dataset: str, row: dict) -> str:
    raw = '|'.join([dataset, str(row.get('conversation_id','')), str(row.get('assistant_turn_index','')), row.get('previous_user_text','')[:240]])
    return prefix + hashlib.sha1(raw.encode('utf-8')).hexdigest()[:10]
def content_words(text: str, n: int = 8) -> str:
    words = [w for w in re.findall(r"[A-Za-z][A-Za-z0-9'-]*", flatten(text)) if w.lower() not in STOP and len(w) > 2]
    return ' '.join(words[:n]) or 'this request'
def base_valid(row: dict) -> tuple[bool, str]:
    user = flatten(row.get('previous_user_text','')); answer = flatten(row.get('assistant_turn_text','')); wc = word_count(user)
    if row.get('assistant_turn_index') != 1: return False, 'not_first_assistant_turn'
    if not user or not answer: return False, 'missing_user_or_answer'
    if wc < 8: return False, 'too_short'
    if wc > 70: return False, 'too_long'
    if len(answer) > 1200: return False, 'answer_too_long'
    if user.count('?') > 3: return False, 'too_many_questions'
    joined = user + ' ' + answer
    if UNSAFE_RE.search(joined): return False, 'unsafe_or_sensitive'
    if URGENT_RE.search(joined): return False, 'urgent_direct_answer_needed'
    if MEDLEGAL_RE.search(user): return False, 'medical_legal_direct_answer_needed'
    if ROLEPLAY_RE.search(user): return False, 'roleplay_or_fiction'
    if CONTEXT_DEP_RE.search(user): return False, 'context_dependent'
    if CODE_HEAVY_RE.search(user): return False, 'code_heavy'
    if ASSISTANT_CHATTER_RE.search(user): return False, 'assistant_identity_or_chitchat'
    return True, 'base_valid'
def infer_type(row: dict) -> str:
    user = flatten(row.get('previous_user_text','')); suggested = row.get('suggested_curiosity_type','')
    if suggested in TARGET_TYPES and USEFUL_PATTERNS[suggested].search(user): return suggested
    for t in ['perspective_taking','counterfactual_hypothetical','reflective_metacognitive','exploratory_broadening','causal_explanatory','information_gap_curiosity']:
        if USEFUL_PATTERNS[t].search(user): return t
    for op in row.get('opportunity_types', []) or []:
        if op in TYPE_FROM_OPPORTUNITY: return TYPE_FROM_OPPORTUNITY[op]
    return suggested if suggested in TARGET_TYPES else 'information_gap_curiosity'
def classify(row: dict) -> tuple[str, str, str]:
    user = flatten(row.get('previous_user_text','')); t = infer_type(row)
    useful_signal = USEFUL_PATTERNS[t].search(user) is not None
    mechanical = TOO_MECHANICAL_RE.search(user) is not None
    direct = DIRECT_FACTUAL_RE.search(user) is not None
    if useful_signal and not (mechanical and not re.search(r'\b(audience|target audience|senior executive|product design|design|research|study|business|proposal|plan|strategy)\b', user, re.I)):
        return 'useful_curiosity_candidate', t, 'opening turn has ambiguity/stakes/audience/criteria where asking first may improve answer'
    if direct or mechanical:
        return 'curiosity_not_needed_contrast', t, 'safe direct/mechanical request where direct answer likely sufficient'
    return 'manual_review_contrast', t, 'safe opening turn but usefulness of asking first is uncertain'
def draft_b(user: str) -> str:
    low = user.lower()
    if re.search(r'\b(audience|reader|senior executive|student|teacher)\b', low): return 'What length and tone should I use for this audience?'
    if re.search(r'\b(recommend|best|choose|suggest)\b', low): return 'What constraints should the recommendation prioritize?'
    if re.search(r'\b(product|design|proposal|plan|strategy|project)\b', low): return 'What format should the plan or design use?'
    if re.search(r'\b(paper|research|study|bias|dataset)\b', low): return 'What scope or field should the answer focus on?'
    if re.search(r'\b(learn|study|explain|understand)\b', low): return 'Should I keep the explanation beginner-friendly or more technical?'
    return 'What output format should I use?'
def draft_c(user: str, t: str) -> str:
    topic = content_words(user)
    if t == 'information_gap_curiosity': return f'What constraint or missing detail about {topic} would most change the answer?'
    if t == 'causal_explanatory': return f'What mechanism or reason behind {topic} do you most want to understand?'
    if t == 'exploratory_broadening': return f'What broader use case or next step for {topic} should the answer consider?'
    if t == 'reflective_metacognitive': return f'What goal, tradeoff, or assumption about {topic} matters most here?'
    if t == 'perspective_taking': return f'Whose perspective or concerns should shape the answer about {topic}?'
    if t == 'counterfactual_hypothetical': return f'What changed condition or what-if scenario about {topic} should the answer account for?'
    return f'What missing detail about {topic} would most improve the answer?'
def read_jsonl(path: Path):
    with path.open(encoding='utf-8') as f:
        for line in f:
            if line.strip(): yield json.loads(line)
def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ['item_id','study_role','dataset','model','conversation_id','target_curiosity_type','opportunity_types','selection_reason','user_message','choice_A_direct_answer','choice_B_related_noncuriosity_question','choice_C_curiosity_question','original_next_user_turn','source_assistant_question_mode','source_assistant_turn_index']
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
def write_loop_files(rows: list[dict], outdir: Path, n_blocks: int = 25) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    for p in outdir.glob('*.tsv'): p.unlink()
    sizes = [len(rows) // n_blocks] * n_blocks
    for i in range(len(rows) % n_blocks): sizes[i] += 1
    fields = ['Field 1 item_id','Field 2 user_message','Field 3 choice_A_direct_answer','Field 4 choice_B_related_noncuriosity_question','Field 5 choice_C_curiosity_question','Field 6 target_curiosity_type','dataset','study_role','example_order']
    idx = 0
    for bnum, size in enumerate(sizes, 1):
        block = rows[idx:idx+size]; idx += size
        with (outdir / f'block_{bnum:02d}_loop_merge_paste_ready.tsv').open('w', encoding='utf-8', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields, delimiter='\t', lineterminator='\n'); w.writeheader()
            for j, r in enumerate(block, 1):
                w.writerow({'Field 1 item_id': r['item_id'],'Field 2 user_message': r['user_message'],'Field 3 choice_A_direct_answer': r['choice_A_direct_answer'],'Field 4 choice_B_related_noncuriosity_question': r['choice_B_related_noncuriosity_question'],'Field 5 choice_C_curiosity_question': r['choice_C_curiosity_question'],'Field 6 target_curiosity_type': r['target_curiosity_type'],'dataset': r['dataset'],'study_role': r['study_role'],'example_order': j})
def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument('--outdir', type=Path, default=DEFAULT_OUTDIR); ap.add_argument('--n-useful', type=int, default=100); ap.add_argument('--n-contrast', type=int, default=25); ap.add_argument('--seed', type=int, default=71); args = ap.parse_args(); rng = random.Random(args.seed)
    useful_by_type = defaultdict(list); contrast = []; rejects = Counter(); seen_user = set()
    for dataset, path in DATASET_FILES.items():
        for row in read_jsonl(path):
            ok, reason = base_valid(row)
            if not ok: rejects[reason] += 1; continue
            user = flatten(row.get('previous_user_text','')); key = re.sub(r'\W+', ' ', user.lower()).strip()[:180]
            if key in seen_user: rejects['near_duplicate_user'] += 1; continue
            seen_user.add(key)
            role, t, sel_reason = classify(row)
            item = {'item_id': stable_id('q3_launch_', dataset, row),'study_role': role,'dataset': dataset,'model': row.get('model',''),'conversation_id': row.get('conversation_id',''),'target_curiosity_type': t,'opportunity_types': ';'.join(row.get('opportunity_types', []) or []),'selection_reason': sel_reason,'user_message': user,'choice_A_direct_answer': flatten(row.get('assistant_turn_text','')),'choice_B_related_noncuriosity_question': draft_b(user),'choice_C_curiosity_question': draft_c(user, t),'original_next_user_turn': flatten(row.get('next_user_text','')),'source_assistant_question_mode': row.get('assistant_question_mode',''),'source_assistant_turn_index': row.get('assistant_turn_index','')}
            if role == 'useful_curiosity_candidate': useful_by_type[t].append(item)
            else: contrast.append(item)
    for xs in useful_by_type.values(): rng.shuffle(xs)
    rng.shuffle(contrast)
    useful = []; per_type = args.n_useful // len(TARGET_TYPES); remainder = args.n_useful % len(TARGET_TYPES)
    for i, t in enumerate(TARGET_TYPES): useful.extend(useful_by_type[t][:(per_type + (1 if i < remainder else 0))])
    if len(useful) < args.n_useful:
        used = {r['item_id'] for r in useful}; leftovers = []
        for t in TARGET_TYPES: leftovers.extend([r for r in useful_by_type[t] if r['item_id'] not in used])
        rng.shuffle(leftovers); useful.extend(leftovers[:args.n_useful - len(useful)])
    selected_contrast = contrast[:args.n_contrast]; selected = useful + selected_contrast; rng.shuffle(selected)
    args.outdir.mkdir(parents=True, exist_ok=True)
    all_candidates = []
    for t in TARGET_TYPES: all_candidates.extend(useful_by_type[t])
    all_candidates.extend(contrast)
    write_csv(args.outdir / 'q3_launch_quality_all_candidates.csv', all_candidates)
    write_csv(args.outdir / 'q3_launch_quality_items_125_seed.csv', selected)
    write_csv(args.outdir / 'q3_launch_quality_useful_candidates_seed.csv', useful)
    write_csv(args.outdir / 'q3_launch_quality_contrast_candidates_seed.csv', selected_contrast)
    write_loop_files(selected, args.outdir / 'loop_merge_25_blocks_tsv_paste_ready_seed', n_blocks=25)
    (args.outdir / 'README.md').write_text('# Q3 Launch-Quality Ask-Before-Answer Pool\n\nThis folder rebuilds the Prolific stimulus pool after filtering out cases where curiosity-before-answer is clearly inappropriate or weak.\n\n- Source: full missed-opportunity archive.\n- Unit: conversation-opening user turn and first assistant response only.\n- Seed target: 100 useful-curiosity candidates + 25 curiosity-not-needed contrasts.\n- Blocks: 25 TSV files, 5 items per block.\n\nRun the LLM refinement script on `q3_launch_quality_items_125_seed.csv` before launch.\n\nUseful candidates available by type:\n\n' + '\n'.join(f'- `{t}`: {len(useful_by_type[t])}' for t in TARGET_TYPES) + '\n\nSelected useful by type:\n\n' + '\n'.join(f'- `{t}`: {sum(1 for r in useful if r["target_curiosity_type"] == t)}' for t in TARGET_TYPES) + f'\n\nSelected contrasts: {len(selected_contrast)}\n\nRejected by reason:\n\n' + '\n'.join(f'- `{k}`: {v}' for k, v in rejects.most_common()) + '\n', encoding='utf-8')
    print('Wrote', args.outdir / 'q3_launch_quality_items_125_seed.csv')
    print('Wrote', args.outdir / 'loop_merge_25_blocks_tsv_paste_ready_seed')
    print('Selected total', len(selected), 'useful', len(useful), 'contrast', len(selected_contrast))
    print('Useful selected by type:', Counter(r['target_curiosity_type'] for r in useful))
    print('Contrast by type:', Counter(r['target_curiosity_type'] for r in selected_contrast))
    print('Dataset selected:', Counter(r['dataset'] for r in selected))
if __name__ == '__main__': main()
