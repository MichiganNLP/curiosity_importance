#!/usr/bin/env python3
"""
Map assistant questions in WildChat-style conversations to curiosity types.

Input: JSONL where each line is one conversation. The script supports several
common schemas:
  - {"conversation": [{"role": "user", "content": "..."}, ...]}
  - {"messages": [{"role": "assistant", "content": "..."}, ...]}
  - {"turns": [{"from": "human", "value": "..."}, ...]}

Output: JSONL, one row per assistant question, with a question-function label
and lightweight metrics for the next user turn.
"""

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


DEFAULT_HF_DATASET = "allenai/WildChat"


QUESTION_START_RE = re.compile(
    r"^\s*(why|how|what|when|where|who|which|whose|whom|can|could|would|"
    r"should|do|does|did|is|are|am|was|were|have|has|had|may|might|will|"
    r"shall)\b",
    re.IGNORECASE,
)

QUESTION_LIKE_RE = re.compile(
    r"^\s*(why|how|what|when|where|who|which|whose|whom|can|could|would|"
    r"should|do|does|did|is|are|am|was|were|have|has|had|may|might|will|"
    r"shall)\b|\b(anything else|right|correct|okay|ok)\?$",
    re.IGNORECASE,
)

SENTENCE_SPLIT_RE = re.compile(r"(?<=[?!.])\s+|\n+")

EPISTEMIC_MARKERS = {
    "wonder",
    "curious",
    "why",
    "how",
    "because",
    "reason",
    "evidence",
    "explain",
    "explanation",
    "hypothesis",
    "assume",
    "assumption",
    "unclear",
    "uncertain",
    "maybe",
    "perhaps",
    "possibly",
    "could",
    "might",
    "alternative",
    "compare",
    "contrast",
    "implication",
    "mechanism",
}

EXPLORATORY_QUESTION_RE = re.compile(
    r"\b(why|how|what if|what would|could|might|would|alternative|"
    r"other ways|what are the implications|what explains)\b",
    re.IGNORECASE,
)

USER_ADDRESSED_RE = re.compile(
    r"\b(you|your|would you|do you|did you|can you|could you|should you|"
    r"tell me|let me know|would like|want me|prefer|need me)\b",
    re.IGNORECASE,
)

DIRECT_ASSISTANT_ADDRESS_RE = re.compile(
    r"\b(would you like|do you want me|want me to|shall i|"
    r"should i (continue|explain|provide|show|include|make|write|draft|create|add)|"
    r"can i (help|provide|show|explain)|could i (help|provide|show|explain)|"
    r"i can|let me know|tell me if|prefer|would you prefer)\b",
    re.IGNORECASE,
)

CREATIVE_REQUEST_RE = re.compile(
    r"\b(script|screenplay|dialogue|dialog|scene|story|fiction|novel|poem|"
    r"roleplay|role play|character|fanfic|skit|play|monologue|lyrics)\b",
    re.IGNORECASE,
)

GENERATIVE_WRITING_REQUEST_RE = re.compile(
    r"\b(write|draft|generate|create|compose|produce)\b.{0,80}"
    r"\b(article|blog|guide book|travel guide|essay|post|copy|content|"
    r"newsletter|speech|story|script|scene|dialogue|poem|lyrics|tweet|tweets|"
    r"social media posts|captions)\b",
    re.IGNORECASE | re.DOTALL,
)

GENERATED_QUESTION_LIST_REQUEST_RE = re.compile(
    r"\b(give me|tell me|write|draft|generate|create|make|produce|provide|list)\b.{0,120}"
    r"\b(questions|most asked questions|frequently asked questions|faqs|multiple choice|mcq|"
    r"quiz|exam|test|questionnaire|survey|interview questions|practice problems|flashcards)\b",
    re.IGNORECASE | re.DOTALL,
)

GENERATED_ARTIFACT_RE = re.compile(
    r"\b(title:|int\.|ext\.|fade out|scene \d+|chapter \d+|act \d+|"
    r"character|dialogue|dialog|script|screenplay|q:|a:)\b",
    re.IGNORECASE,
)


@dataclass
class Turn:
    role: str
    text: str


@dataclass
class QuestionLabel:
    primary_function: str
    curiosity_status: str
    curiosity_subtype: str
    epistemic_depth: int
    confidence: float
    rationale: str


def load_hf_dataset(
    dataset_name: str,
    split: str,
    streaming: bool,
) -> Iterable[Dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Hugging Face input requires the datasets package. "
            "Install it with: pip install datasets"
        ) from exc

    dataset = load_dataset(dataset_name, split=split, streaming=streaming)
    for row_no, row in enumerate(dataset, start=1):
        row = dict(row)
        row["_line_no"] = row_no
        yield row


def load_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no}: {exc}") from exc
            obj["_line_no"] = line_no
            yield obj


def normalize_role(raw_role: Any) -> Optional[str]:
    role = str(raw_role or "").lower().strip()
    if role in {"assistant", "bot", "gpt", "chatgpt", "model"}:
        return "assistant"
    if role in {"user", "human", "client", "prompter"}:
        return "user"
    return None


def get_text(turn: Dict[str, Any]) -> str:
    for key in ("content", "text", "value", "message"):
        value = turn.get(key)
        if isinstance(value, str):
            return value.strip()
    return ""


def normalize_turns(conversation: Dict[str, Any]) -> List[Turn]:
    raw_turns = None
    for key in ("conversation", "messages", "turns", "dialogue"):
        if isinstance(conversation.get(key), list):
            raw_turns = conversation[key]
            break

    if raw_turns is None and isinstance(conversation.get("conversations"), list):
        raw_turns = conversation["conversations"]

    if raw_turns is None:
        return []

    turns = []  # type: List[Turn]
    for raw in raw_turns:
        if not isinstance(raw, dict):
            continue
        role = normalize_role(raw.get("role") or raw.get("from") or raw.get("speaker"))
        text = get_text(raw)
        if role and text:
            turns.append(Turn(role=role, text=text))
    return turns


def conversation_id(obj: Dict[str, Any]) -> str:
    for key in ("conversation_id", "id", "chat_id", "session_id", "uid"):
        if key in obj:
            return str(obj[key])
    return f"line_{obj.get('_line_no', 'unknown')}"


def extract_questions(text: str) -> List[str]:
    questions = []  # type: List[str]
    if "?" not in text:
        return questions
    for part in SENTENCE_SPLIT_RE.split(text):
        part = part.strip()
        if not part or "?" not in part:
            continue
        for q in re.findall(r"[^?]*\?", part):
            q = clean_question_text(q)
            if q:
                questions.append(q)
    return questions


def clean_question_text(question: str) -> str:
    question = re.sub(r"^[\s>\-*#`'\"]+", "", question.strip())
    question = re.sub(r"\s+", " ", question)
    return question


def line_context(text: str, question: str) -> Tuple[str, str, str]:
    index = text.find(question)
    if index < 0:
        return "", "", ""
    prefix = text[:index]
    suffix = text[index + len(question):]
    previous_lines = [line.strip() for line in prefix.splitlines() if line.strip()]
    current_start = prefix.rfind("\n") + 1
    current_end = suffix.find("\n")
    if current_end < 0:
        current_line = text[current_start:].strip()
    else:
        current_line = text[current_start:index + len(question) + current_end].strip()
    previous_line = previous_lines[-1] if previous_lines else ""
    next_line = suffix.splitlines()[0].strip() if suffix.splitlines() else ""
    return previous_line, current_line, next_line


def is_dialogue_or_script_line(previous_line: str, current_line: str) -> bool:
    context = previous_line + " " + current_line
    if re.match(r"^[A-Z][A-Z0-9 ._'-]{1,45}$", previous_line):
        return True
    if re.match(r"^[A-Z][A-Z0-9 ._'-]{1,45}\s*[:(]", current_line):
        return True
    if re.search(r"\b[A-Z][A-Z ._'-]{1,45}\s*\([^)]{1,80}\)", current_line):
        return True
    if re.search(r"\b(INT\.|EXT\.|FADE OUT|CUT TO:|SCENE:|SETTING:|CHARACTERS:)\b", context):
        return True
    if current_line.startswith('"') or current_line.startswith("'"):
        return True
    return False


def is_likely_generated_artifact_question(
    question: str,
    previous_user: str,
    assistant_turn: str,
) -> Tuple[bool, str]:
    q = question.strip()
    q_lower = q.lower()
    prev_lower = previous_user.lower()
    turn_lower = assistant_turn.lower()
    previous_line, current_line, _ = line_context(assistant_turn, q)

    q_words = re.findall(r"[A-Za-z0-9']+", q)
    if len(q.replace("?", "").strip()) < 3 or len(q_words) < 2:
        return True, "too_short_or_malformed_question"
    if len(q_words) == 2 and q_words[0].lower() in {"why", "how", "what", "when", "where", "who", "which"}:
        return True, "too_short_or_malformed_question"

    if re.match(r"^[A-Z][A-Za-z0-9 _.'-]{1,35}:", q):
        return True, "speaker_prefixed_question"

    if re.search(r"\b(research question|aims to investigate|interviewer:)\b", q_lower):
        return True, "generated_prompt_or_interview_question"

    if not QUESTION_LIKE_RE.search(q) and not DIRECT_ASSISTANT_ADDRESS_RE.search(q):
        return True, "non_interrogative_fragment"

    in_creative_context = CREATIVE_REQUEST_RE.search(prev_lower) and len(assistant_turn.split()) > 80
    in_writing_generation = GENERATIVE_WRITING_REQUEST_RE.search(prev_lower) and len(assistant_turn.split()) > 80
    in_question_list_generation = GENERATED_QUESTION_LIST_REQUEST_RE.search(prev_lower) and len(assistant_turn.split()) > 40
    in_generated_artifact = GENERATED_ARTIFACT_RE.search(turn_lower) and len(assistant_turn.split()) > 180

    if is_dialogue_or_script_line(previous_line, current_line):
        return True, "script_or_dialogue_line"

    if in_creative_context and not DIRECT_ASSISTANT_ADDRESS_RE.search(q):
        return True, "creative_generation_context"

    if in_writing_generation and not DIRECT_ASSISTANT_ADDRESS_RE.search(q):
        return True, "generated_writing_context"

    if in_question_list_generation and not DIRECT_ASSISTANT_ADDRESS_RE.search(q):
        return True, "generated_question_list_context"

    if in_generated_artifact and not DIRECT_ASSISTANT_ADDRESS_RE.search(q):
        return True, "generated_artifact_context"

    if USER_ADDRESSED_RE.search(q):
        return False, ""

    if re.search(r"\b(example|sample|template|exercise|quiz|interview question|faq)\b", turn_lower):
        if not USER_ADDRESSED_RE.search(q_lower):
            return True, "example_or_template_question"

    return False, ""


def extract_addressed_questions(
    text: str,
    previous_user: str = "",
) -> Tuple[List[str], List[Dict[str, str]]]:
    kept = []  # type: List[str]
    skipped = []  # type: List[Dict[str, str]]
    for question in extract_questions(text):
        skip, reason = is_likely_generated_artifact_question(question, previous_user, text)
        if skip:
            skipped.append({"question_text": question, "skip_reason": reason})
        else:
            kept.append(question)
    return kept, skipped


def classify_question(question: str, previous_user: str = "", assistant_turn: str = "") -> QuestionLabel:
    q = " ".join(question.lower().split())
    prev = previous_user.lower()
    full = assistant_turn.lower()

    if re.search(r"\b(format|bullet|bullets|table|paragraph|tone|length|style|language|file|csv|json|markdown|formal|casual|specific reason)\b", q):
        return QuestionLabel(
            "task_management",
            "non_curiosity",
            "none",
            0,
            0.84,
            "Asks about output preferences or task constraints.",
        )

    if re.search(r"\b(would you like|do you want|want me to|shall i|should i continue|need anything else|anything else)\b", q):
        return QuestionLabel(
            "engagement_maintenance",
            "non_curiosity",
            "none",
            0,
            0.86,
            "Generic continuation or offer to keep the interaction going.",
        )

    if re.search(r"\b(format|bullet|bullets|table|paragraph|tone|length|style|language|file|csv|json|markdown)\b", q):
        return QuestionLabel(
            "task_management",
            "non_curiosity",
            "none",
            0,
            0.82,
            "Asks about output preferences or task constraints.",
        )

    if re.search(r"\b(what do you mean|could you clarify|can you clarify|which .* mean|more context|provide more details)\b", q):
        return QuestionLabel(
            "clarification_repair",
            "adjacent_inquiry",
            "none",
            1,
            0.82,
            "Repairs ambiguity needed to proceed.",
        )

    if re.search(r"\b(don't you think|wouldn't you agree|isn't it true|surely|obviously)\b", q):
        return QuestionLabel(
            "leading_steering",
            "problematic_curiosity_like",
            "none",
            1,
            0.78,
            "Question form steers the user toward an answer.",
        )

    if re.search(r"\b(personal|address|phone|ssn|social security|password|income|medical|diagnosis|religion|political)\b", q):
        return QuestionLabel(
            "privacy_probing",
            "problematic_curiosity_like",
            "none",
            1,
            0.70,
            "Requests potentially sensitive personal information.",
        )

    if re.search(r"\b(what if|suppose|imagine if|if .* were|if .* had|counterfactual)\b", q):
        return QuestionLabel(
            "epistemic_curiosity",
            "core_curiosity",
            "counterfactual",
            3,
            0.84,
            "Explores a hypothetical or alternative scenario.",
        )

    if re.search(r"\b(why|how does|how do|how did|what caused|what causes|mechanism|explain)\b", q):
        return QuestionLabel(
            "epistemic_curiosity",
            "core_curiosity",
            "causal",
            3,
            0.84,
            "Seeks causal or explanatory understanding.",
        )

    if re.search(r"\b(assumption|assumptions|reasoning|evidence|what makes you think|how do you know|uncertain|confidence)\b", q):
        return QuestionLabel(
            "epistemic_curiosity",
            "core_curiosity",
            "reflective_metacognitive",
            3,
            0.82,
            "Invites reflection on reasoning, evidence, or assumptions.",
        )

    if re.search(r"\b(perspective|stakeholder|someone else|another person|others view|they see|from .* point of view)\b", q):
        return QuestionLabel(
            "epistemic_curiosity",
            "core_curiosity",
            "perspective_taking",
            3,
            0.78,
            "Invites another viewpoint or stakeholder perspective.",
        )

    if re.search(r"\b(other|alternative|another way|possibilities|implications|explore|broader|deeper|tell me more)\b", q):
        # "Tell me more" is ambiguous; downgrade if it appears as a generic final hook.
        if q in {"can you tell me more?", "tell me more?", "could you tell me more?"} and len(full.split()) < 80:
            return QuestionLabel(
                "clarification_repair",
                "adjacent_inquiry",
                "none",
                1,
                0.58,
                "Ambiguous request for more information.",
            )
        return QuestionLabel(
            "epistemic_curiosity",
            "core_curiosity",
            "exploratory_broadening",
            2,
            0.72,
            "Broadens the inquiry toward alternatives or implications.",
        )

    if re.search(r"\b(what|which|who|where|when|can you provide|could you provide)\b", q):
        status = "adjacent_inquiry" if len(prev.split()) < 25 else "core_curiosity"
        subtype = "none" if status == "adjacent_inquiry" else "information_gap"
        return QuestionLabel(
            "information_request",
            status,
            subtype,
            1 if status == "adjacent_inquiry" else 2,
            0.62,
            "Requests missing information; context determines whether this is task repair or curiosity.",
        )

    return QuestionLabel(
        "unknown_question",
        "needs_review",
        "none",
        0,
        0.30,
        "No high-confidence taxonomy rule matched.",
    )


def count_question_marks(text: str) -> int:
    return text.count("?")


def marker_count(text: str, markers: Set[str]) -> int:
    words = re.findall(r"[a-zA-Z']+", text.lower())
    return sum(1 for word in words if word in markers)


def next_user_turn(turns: List[Turn], start_index: int) -> Tuple[Optional[int], str]:
    for i in range(start_index + 1, len(turns)):
        if turns[i].role == "user":
            return i, turns[i].text
    return None, ""


def user_followup_metrics(text: str) -> Dict[str, Any]:
    questions = extract_questions(text)
    return {
        "next_user_word_count": len(re.findall(r"[a-zA-Z0-9']+", text)),
        "next_user_has_question": bool(questions),
        "next_user_question_count": len(questions),
        "next_user_exploratory_question_count": sum(1 for q in questions if EXPLORATORY_QUESTION_RE.search(q)),
        "next_user_epistemic_marker_count": marker_count(text, EPISTEMIC_MARKERS),
        "next_user_text": text,
    }


def keep_conversation(
    conversation: Dict[str, Any],
    language: Optional[str],
    model: Optional[str],
    skip_redacted: bool,
    skip_toxic: bool,
) -> bool:
    if language and str(conversation.get("language", "")).lower() != language.lower():
        return False
    if model and str(conversation.get("model", "")).lower() != model.lower():
        return False
    if skip_redacted and bool(conversation.get("redacted", False)):
        return False
    if skip_toxic and bool(conversation.get("toxic", False)):
        return False
    return True


def iter_question_events(conversation: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    turns = normalize_turns(conversation)
    conv_id = conversation_id(conversation)

    for turn_index, turn in enumerate(turns):
        if turn.role != "assistant":
            continue
        previous_user = ""
        for j in range(turn_index - 1, -1, -1):
            if turns[j].role == "user":
                previous_user = turns[j].text
                break

        questions, skipped_questions = extract_addressed_questions(turn.text, previous_user)
        if not questions:
            continue

        next_index, next_text = next_user_turn(turns, turn_index)
        followup = user_followup_metrics(next_text)

        for question_index, question in enumerate(questions):
            label = classify_question(question, previous_user=previous_user, assistant_turn=turn.text)
            yield {
                "conversation_id": conv_id,
                "assistant_turn_index": turn_index,
                "question_index_in_turn": question_index,
                "question_text": question,
                "skipped_artifact_question_count": len(skipped_questions),
                "model": conversation.get("model"),
                "conversation_language": conversation.get("language"),
                "conversation_turn_count": conversation.get("turn"),
                "timestamp": str(conversation.get("timestamp")) if conversation.get("timestamp") is not None else None,
                "assistant_turn_text": turn.text,
                "previous_user_text": previous_user,
                "next_user_turn_index": next_index,
                **asdict(label),
                **followup,
            }


def write_events(
    conversations: Iterable[Dict[str, Any]],
    output_path: Path,
    limit: Optional[int] = None,
    language: Optional[str] = None,
    model: Optional[str] = None,
    skip_redacted: bool = False,
    skip_toxic: bool = False,
) -> None:
    n_seen = 0
    n_kept = 0
    n_events = 0
    with output_path.open("w", encoding="utf-8") as out:
        for conversation in conversations:
            n_seen += 1
            if not keep_conversation(conversation, language, model, skip_redacted, skip_toxic):
                continue
            n_kept += 1
            for event in iter_question_events(conversation):
                n_events += 1
                out.write(json.dumps(event, ensure_ascii=False) + "\n")
            if limit is not None and n_kept >= limit:
                break
    print(f"Saw {n_seen} conversations")
    print(f"Kept {n_kept} conversations after filters")
    print(f"Wrote {n_events} assistant-question events to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input", type=Path, help="Path to WildChat-style JSONL")
    input_group.add_argument("--hf-dataset", default=None, help="Hugging Face dataset name, e.g. allenai/WildChat")
    parser.add_argument("--split", default="train", help="Hugging Face split name")
    parser.add_argument("--streaming", action="store_true", help="Stream Hugging Face rows instead of downloading first")
    parser.add_argument("--limit", type=int, default=None, help="Maximum kept conversations to process")
    parser.add_argument("--language", default=None, help="Keep only this WildChat language label, e.g. English")
    parser.add_argument("--model", default=None, help="Keep only this WildChat model label, e.g. gpt-4")
    parser.add_argument("--skip-redacted", action="store_true", help="Skip conversations marked redacted")
    parser.add_argument("--skip-toxic", action="store_true", help="Skip conversations marked toxic")
    parser.add_argument("--output", required=True, type=Path, help="Path for event-level JSONL output")
    args = parser.parse_args()

    if args.input is not None:
        conversations = load_jsonl(args.input)
    else:
        conversations = load_hf_dataset(args.hf_dataset or DEFAULT_HF_DATASET, args.split, args.streaming)

    write_events(
        conversations,
        args.output,
        limit=args.limit,
        language=args.language,
        model=args.model,
        skip_redacted=args.skip_redacted,
        skip_toxic=args.skip_toxic,
    )


if __name__ == "__main__":
    main()
