#!/usr/bin/env python3
"""Download/convert RyokoAI/ShareGPT52K raw JSON files to conversation JSONL."""

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

FILES = ["sg_90k_part1.json", "sg_90k_part2.json"]
ROLE_MAP = {"human": "user", "gpt": "assistant", "user": "user", "assistant": "assistant"}
REPO_ROOT = Path(__file__).resolve().parents[2]


def download_files(repo_id: str, cache_dir: Path, filenames: List[str]) -> List[Path]:
    from huggingface_hub import hf_hub_download
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for filename in filenames:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            local_dir=str(cache_dir),
            local_dir_use_symlinks=False,
        )
        paths.append(Path(path))
    return paths


def iter_json_array(path: Path) -> Iterable[Dict]:
    """Incrementally parse a JSON array of objects without loading all at once."""
    decoder = json.JSONDecoder()
    buf = ""
    started = False
    with path.open("r", encoding="utf-8") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk and not buf.strip():
                break
            buf += chunk
            while True:
                buf = buf.lstrip()
                if not started:
                    if not buf:
                        break
                    if buf[0] == "[":
                        started = True
                        buf = buf[1:]
                        continue
                    # Some files might be JSONL; fallback line mode is handled elsewhere.
                    raise ValueError(f"Expected JSON array in {path}, got {buf[:50]!r}")
                if not buf:
                    break
                if buf[0] == "]":
                    return
                if buf[0] == ",":
                    buf = buf[1:]
                    continue
                try:
                    obj, idx = decoder.raw_decode(buf)
                except json.JSONDecodeError:
                    if not chunk:
                        raise
                    break
                if isinstance(obj, dict):
                    yield obj
                buf = buf[idx:]
            if not chunk:
                break


def normalize_turns(row: Dict) -> List[Dict]:
    raw_turns = row.get("conversations") or row.get("conversation") or row.get("messages") or []
    turns = []
    for raw in raw_turns:
        if not isinstance(raw, dict):
            continue
        role = ROLE_MAP.get(str(raw.get("from") or raw.get("role") or "").lower())
        text = raw.get("value") or raw.get("content") or raw.get("text") or ""
        text = str(text).strip()
        if not role or not text:
            continue
        if turns and turns[-1]["role"] == role:
            turns[-1]["content"] += "\n\n" + text
        else:
            turns.append({"role": role, "content": text})
    return turns


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-id", default="RyokoAI/ShareGPT52K")
    ap.add_argument("--cache-dir", type=Path, default=REPO_ROOT / "data" / "raw" / "sharegpt")
    ap.add_argument("--files", nargs="+", default=FILES)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--min-turns", type=int, default=2)
    args = ap.parse_args()

    paths = download_files(args.repo_id, args.cache_dir, args.files)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    seen = wrote = 0
    with args.output.open("w", encoding="utf-8") as out:
        for path in paths:
            print("Reading", path, flush=True)
            for row in iter_json_array(path):
                seen += 1
                turns = normalize_turns(row)
                if len(turns) < args.min_turns:
                    continue
                cid = row.get("id") or row.get("conversation_id") or f"sharegpt_{seen}"
                conv = {
                    "conversation_id": f"sharegpt_{cid}",
                    "source_dataset": args.repo_id,
                    "source_file": path.name,
                    "language": "English",
                    "model": "sharegpt",
                    "turn": len(turns),
                    "conversation": turns,
                }
                out.write(json.dumps(conv, ensure_ascii=False) + "\n")
                wrote += 1
                if args.limit and wrote >= args.limit:
                    print("Saw rows:", seen)
                    print("Wrote conversations:", wrote)
                    print("Output:", args.output)
                    return
    print("Saw rows:", seen)
    print("Wrote conversations:", wrote)
    print("Output:", args.output)


if __name__ == "__main__":
    main()
