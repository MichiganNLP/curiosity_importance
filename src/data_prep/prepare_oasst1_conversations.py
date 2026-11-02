#!/usr/bin/env python3
"""
Convert OpenAssistant OASST1 message-tree rows into conversation-path JSONL.

OASST1 is stored as one row per message with parent_id/message_tree_id. The
curiosity pipeline expects one row per linear conversation, so this script
reconstructs root-to-leaf paths and writes WildChat-style JSONL:

  {"conversation_id": ..., "language": "English", "model": "oasst1", "conversation": [...]}

Notes:
- This is path-level data. If a tree branches, shared prefix messages appear in
  multiple paths. This is useful for Q2 because different branches can have
  different next-user followups, but Q1 counts should be described as path-level
  assistant-question events unless deduplicated later.
- By default, keeps reviewed, non-deleted English messages.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

LANG_MAP = {"en": "English"}
ROLE_MAP = {"prompter": "user", "assistant": "assistant"}


def load_hf_rows(dataset_name: str, split: str, streaming: bool) -> Iterable[Dict]:
    from datasets import load_dataset
    ds = load_dataset(dataset_name, split=split, streaming=streaming)
    for row in ds:
        yield dict(row)


def keep_row(row: Dict, lang: str, reviewed_only: bool, skip_deleted: bool) -> bool:
    if lang and str(row.get("lang", "")).lower() != lang.lower():
        return False
    if reviewed_only and row.get("review_result") is not True:
        return False
    if skip_deleted and row.get("deleted") is True:
        return False
    if row.get("role") not in ROLE_MAP:
        return False
    if not str(row.get("text") or "").strip():
        return False
    return True


def role(row: Dict) -> str:
    return ROLE_MAP.get(row.get("role"), str(row.get("role") or ""))


def build_path(node_id: str, nodes: Dict[str, Dict]) -> Optional[List[Dict]]:
    path = []
    seen = set()
    cur = node_id
    while cur:
        if cur in seen:
            return None
        seen.add(cur)
        node = nodes.get(cur)
        if node is None:
            return None
        path.append(node)
        cur = node.get("parent_id")
    path.reverse()
    return path


def alternating_turns(path: List[Dict]) -> List[Dict]:
    turns = []
    for row in path:
        r = role(row)
        text = str(row.get("text") or "").strip()
        if not text:
            continue
        if turns and turns[-1]["role"] == r:
            turns[-1]["content"] = turns[-1]["content"].rstrip() + "\n\n" + text
        else:
            turns.append({"role": r, "content": text})
    return turns


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hf-dataset", default="OpenAssistant/oasst1")
    ap.add_argument("--split", default="train")
    ap.add_argument("--streaming", action="store_true")
    ap.add_argument("--lang", default="en")
    ap.add_argument("--reviewed-only", action="store_true", default=True)
    ap.add_argument("--include-unreviewed", action="store_false", dest="reviewed_only")
    ap.add_argument("--skip-deleted", action="store_true", default=True)
    ap.add_argument("--include-deleted", action="store_false", dest="skip_deleted")
    ap.add_argument("--leaf-role", choices=["any", "assistant", "prompter"], default="any")
    ap.add_argument("--min-turns", type=int, default=2)
    ap.add_argument("--limit-paths", type=int, default=None)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    nodes = {}
    children = defaultdict(list)
    seen = kept = 0
    for row in load_hf_rows(args.hf_dataset, args.split, args.streaming):
        seen += 1
        if not keep_row(row, args.lang, args.reviewed_only, args.skip_deleted):
            continue
        kept += 1
        mid = row.get("message_id")
        if not mid:
            continue
        nodes[mid] = row

    for mid, row in nodes.items():
        parent = row.get("parent_id")
        if parent in nodes:
            children[parent].append(mid)

    leaf_ids = []
    for mid, row in nodes.items():
        if children.get(mid):
            continue
        if args.leaf_role != "any" and row.get("role") != args.leaf_role:
            continue
        leaf_ids.append(mid)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    wrote = 0
    with args.output.open("w", encoding="utf-8") as out:
        for leaf_id in leaf_ids:
            path = build_path(leaf_id, nodes)
            if not path:
                continue
            turns = alternating_turns(path)
            if len(turns) < args.min_turns:
                continue
            tree_id = path[0].get("message_tree_id") or path[-1].get("message_tree_id")
            conv = {
                "conversation_id": f"oasst1_{args.split}_{tree_id}_{leaf_id}",
                "source_dataset": args.hf_dataset,
                "source_split": args.split,
                "source_tree_id": tree_id,
                "source_leaf_id": leaf_id,
                "language": LANG_MAP.get(args.lang, args.lang),
                "model": "oasst1",
                "turn": len(turns),
                "timestamp": path[-1].get("created_date"),
                "conversation": turns,
            }
            out.write(json.dumps(conv, ensure_ascii=False) + "\n")
            wrote += 1
            if args.limit_paths is not None and wrote >= args.limit_paths:
                break

    print("Saw rows:", seen)
    print("Kept message rows:", kept)
    print("Trees:", len({r.get('message_tree_id') for r in nodes.values()}))
    print("Leaf paths available:", len(leaf_ids))
    print("Wrote conversation paths:", wrote)
    print("Output:", args.output)


if __name__ == "__main__":
    main()
