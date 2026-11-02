# What Makes a Question Curious?

Paper -- **“What Makes a Question Curious? Measuring Model Curiosity in Human–LLM Interactions.”** (under review, https://arxiv.org/pdf/2610.33950v1) 

This repository studies observable model curiosity: assistant questions that surface knowledge gaps, seek explanations, examine assumptions, or invite new perspectives and alternatives. It contains the taxonomy pipeline, Q1/Q2 observational analyses, Q3 human-study construction code, sanitized row-level samples, aggregate result tables, paper figures)
## Repository layout

```text
.
├── src/
│   ├── data_prep/   # public-dataset normalization and question extraction
│   ├── q1_q2/       # taxonomy judging, matching, engagement, and plotting
│   └── q3/          # human-study pool construction and question generation
├── data/
│   ├── q1_q2/       # sanitized structural samples
│   └── q3/          # sanitized survey/modeling samples
├── results/
│   ├── q1_q2/       # aggregate paper-facing result tables
│   └── q3/          # aggregate human-study and classifier tables
├── figures/         # selected paper figures
├── docs/           
└── outputs/         # local run outputs; ignored by Git except .gitkeep
```

Raw public conversation dumps, full model-judged event files, raw survey
exports, free-text participant responses, and direct identifiers are not
included. The sanitized samples preserve analysis-relevant labels and numeric
features while omitting raw conversation text and participant identifiers.

## Installation

Python 3.10 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

LLM-assisted scripts use the OpenAI API and read the key only from the
environment:

```bash
export OPENAI_API_KEY="..."
```

Never commit `.env` files, API keys, or model-provider logs.

The released defaults follow the manuscript: `gpt-5-mini` for the contextual
taxonomy and downstream-curiosity judges, and `gpt-5.5` for Q3 candidate
question generation. Each script exposes `--model` so a compatible replacement
can be supplied explicitly.

## Minimal pipeline

### 1. Normalize or load a public conversation dataset

WildChat can be streamed directly:

```bash
python src/data_prep/curiosity_mapping.py \
  --hf-dataset allenai/WildChat \
  --split train \
  --streaming \
  --language English \
  --limit 1000 \
  --output outputs/question_events.jsonl
```

OASST1 and ShareGPT conversion utilities are in `src/data_prep/`. The generic
`prepare_linear_chat_dataset.py` supports common local JSON/JSONL and Hugging
Face chat schemas.

### 2. Apply the contextual taxonomy judge

```bash
python src/q1_q2/llm_judge_curiosity.py \
  --input outputs/question_events.jsonl \
  --output-jsonl outputs/llm_judged.jsonl \
  --output-csv outputs/llm_judged.csv \
  --limit 200
```

### 3. Run the matched downstream-engagement analysis

```bash
python src/q1_q2/matched_observational_analysis.py \
  --input outputs/llm_judged.jsonl \
  --outdir outputs/matched_q2 \
  --control-types engagement_maintenance
```

The estimates are observational associations, not randomized causal effects.
Matching and adjusted regression reduce observed imbalance but cannot remove
unobserved confounding.

### 4. Validate next-user curiosity with an LLM judge

```bash
python src/q1_q2/llm_judge_human_curiosity.py \
  --matched-pairs outputs/matched_q2/matched_pairs.csv \
  --dataset WildChat \
  --output-jsonl outputs/wildchat_q2_human_curiosity_judged.jsonl \
  --output-csv outputs/wildchat_q2_human_curiosity_judged.csv
```

### 5. Human-study construction

The scripts under `src/q3/` build a balanced ask-before-answer candidate pool
and optionally refine candidate questions with an LLM. They require the
paper-specific missed-opportunity inputs generated from public conversation
data. Sanitized structural samples and aggregate results are provided, but raw
survey exports are intentionally excluded.

## Released data and results

- `data/q1_q2/` contains sanitized event, validation, and trajectory samples.
- `data/q3/` contains sanitized survey, demographic, and useful-curiosity
  modeling samples.
- `results/` contains aggregate tables used in the manuscript and figures.
- `figures/` contains selected publication-quality PNGs.

Dataset licenses and terms continue to govern any source conversations obtained
from WildChat, OASST1, LMSYS-Chat-1M, or ShareGPT. This repository does not
redistribute their raw conversation text.


