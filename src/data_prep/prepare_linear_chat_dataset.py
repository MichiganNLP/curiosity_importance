#!/usr/bin/env python3
"""
Convert common linear chat datasets to the conversation JSONL format used by
curiosity_mapping.py.

Supports Hugging Face datasets or local JSON/JSONL. Intended for datasets like
LMSYS Chat-1M and ShareGPT mirrors whose rows contain one of:
  conversation, conversations, messages, turns, dialogue
with turns like {role/content}, {from/value}, {speaker/text}, etc.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

ROLE_MAP = {
    'assistant': 'assistant', 'gpt': 'assistant', 'chatgpt': 'assistant', 'bot': 'assistant', 'model': 'assistant',
    'user': 'user', 'human': 'user', 'prompter': 'user', 'client': 'user',
}
TEXT_KEYS = ('content', 'text', 'value', 'message')
TURN_KEYS = ('conversation', 'conversations', 'messages', 'turns', 'dialogue')


def load_hf(dataset: str, split: str, streaming: bool, data_files: Optional[str]) -> Iterable[Dict[str, Any]]:
    from datasets import load_dataset
    kwargs = {}
    if data_files:
        kwargs['data_files'] = data_files
    ds = load_dataset(dataset, split=split, streaming=streaming, **kwargs)
    for row in ds:
        yield dict(row)


def load_local(path: Path) -> Iterable[Dict[str, Any]]:
    if path.suffix.lower() == '.jsonl':
        with path.open(encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        yield obj
    else:
        obj = json.loads(path.read_text(encoding='utf-8'))
        if isinstance(obj, list):
            for row in obj:
                if isinstance(row, dict):
                    yield row
        elif isinstance(obj, dict):
            # Some ShareGPT dumps are wrapped under a top-level key.
            emitted = False
            for key in ('data', 'train', 'conversations', 'items'):
                if isinstance(obj.get(key), list):
                    for row in obj[key]:
                        if isinstance(row, dict):
                            yield row
                    emitted = True
                    break
            if not emitted:
                yield obj


def normalize_role(raw: Any) -> Optional[str]:
    if raw is None:
        return None
    return ROLE_MAP.get(str(raw).strip().lower())


def turn_text(turn: Dict[str, Any]) -> str:
    for key in TEXT_KEYS:
        val = turn.get(key)
        if isinstance(val, str):
            return val.strip()
    return ''


def row_turns(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw_turns = None
    for key in TURN_KEYS:
        if isinstance(row.get(key), list):
            raw_turns = row[key]
            break
    if raw_turns is None:
        return []
    out = []
    for raw in raw_turns:
        if not isinstance(raw, dict):
            continue
        role = normalize_role(raw.get('role') or raw.get('from') or raw.get('speaker'))
        text = turn_text(raw)
        if not role or not text:
            continue
        if out and out[-1]['role'] == role:
            out[-1]['content'] += '\n\n' + text
        else:
            out.append({'role': role, 'content': text})
    return out


def get_id(row: Dict[str, Any], i: int, prefix: str) -> str:
    for key in ('conversation_id', 'id', 'chat_id', 'session_id', 'uid', 'conversation_hash'):
        if row.get(key) is not None:
            return f'{prefix}_{row[key]}'
    return f'{prefix}_row_{i}'


def get_language(row: Dict[str, Any], default: Optional[str]) -> str:
    for key in ('language', 'lang'):
        if row.get(key):
            val = str(row[key])
            if val.lower() == 'en':
                return 'English'
            return val
    return default or ''


def get_model(row: Dict[str, Any], default: str) -> str:
    for key in ('model', 'model_name', 'assistant_model'):
        if row.get(key):
            return str(row[key])
    return default


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--hf-dataset')
    src.add_argument('--input', type=Path)
    ap.add_argument('--split', default='train')
    ap.add_argument('--streaming', action='store_true')
    ap.add_argument('--data-files', default=None)
    ap.add_argument('--dataset-prefix', default='chat')
    ap.add_argument('--default-language', default='English')
    ap.add_argument('--language-filter', default=None, help='Keep rows with this normalized language label, e.g. English')
    ap.add_argument('--default-model', default='unknown')
    ap.add_argument('--min-turns', type=int, default=2)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()

    rows = load_local(args.input) if args.input else load_hf(args.hf_dataset, args.split, args.streaming, args.data_files)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    seen = kept = wrote = 0
    with args.output.open('w', encoding='utf-8') as out:
        for i, row in enumerate(rows, start=1):
            seen += 1
            turns = row_turns(row)
            if len(turns) < args.min_turns:
                continue
            lang = get_language(row, args.default_language)
            if args.language_filter and lang.lower() != args.language_filter.lower():
                continue
            kept += 1
            conv = {
                'conversation_id': get_id(row, i, args.dataset_prefix),
                'source_dataset': args.hf_dataset or str(args.input),
                'source_split': args.split if args.hf_dataset else None,
                'language': lang,
                'model': get_model(row, args.default_model),
                'turn': len(turns),
                'timestamp': str(row.get('tstamp') or row.get('timestamp') or row.get('created_at') or '') or None,
                'conversation': turns,
            }
            out.write(json.dumps(conv, ensure_ascii=False) + '\n')
            wrote += 1
            if args.limit and wrote >= args.limit:
                break
    print('Saw rows:', seen)
    print('Kept conversations:', kept)
    print('Wrote conversations:', wrote)
    print('Output:', args.output)


if __name__ == '__main__':
    main()
