---
title: AutoDataLab Data Cleaning Env
emoji: 🛒
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
tags:
  - openenv
  - data-cleaning
  - ecommerce
  - reinforcement-learning
---

# AutoDataLab — E-Commerce Data Analyst Environment

A **hackathon-ready** [OpenEnv](https://github.com/meta-pytorch/OpenEnv) environment that simulates a real e-commerce data analyst workflow.

Clean order-level data, compute business metrics, and declare insightful charts — all scored deterministically against a ground-truth CSV.

---

## What Is This?

Data teams spend a huge fraction of their time on:

- Deduplication
- Missing value handling
- Outlier detection and removal
- Derived columns and rollups
- Visualization declarations

This environment turns that daily workflow into a **step / reset / state** RL-style API with **deterministic graders** and **dense reward shaping**.

---

## Example Dataset

Columns: `OrderID`, `CustomerID`, `ExpiryDays`, `Product`, `Category`, `Price`, `Quantity`, `OrderDate`

| OrderID | CustomerID | ExpiryDays | Product | Category    | Price     | Quantity | OrderDate  |            |
|---------|------------|-----------|---------|-------------|-----------|----------|------------|------------|
| 101     | C001       | 25        | Shoes   | Fashion     | 2000      | 1        | 2023-01-01 |            |
| 101     | C001       | 25        | Shoes   | Fashion     | 2000      | 1        | 2023-01-01 | ← duplicate |
| 102     | C002       | *(empty)* | Laptop  | Electronics | 60000     | 1        | 2023-01-02 | ← missing expiry |
| 103     | C003       | 200       | T-shirt | Fashion     | 500       | 2        | 2023-01-03 | ← short shelf-life style field |
| 104     | C004       | 30        | Phone   | Electronics | *(empty)* | 1        | 2023-01-04 | ← missing price |

---

## Tasks

| Task | Difficulty | Analyst Goal |
|------|------------|--------------|
| `easy` | 🟢 Easy | **Data Cleaning** — dedupe and impute missing Price (mean). ExpiryDays is warning-only, not a required transform. |
| `medium` | 🟡 Medium | **Business Metrics** — pre-cleaned data given; run `compute_metrics` → category-level revenue table. |
| `medium_plus` | 🟠 Medium+ | **Full KPIs** — run `compute_kpis` → `Metric / Value` table with TotalRevenue + AvgOrderValue. |
| `hard` | 🔴 Hard | **Cleaning + Insight** — full cleaning as in `easy`, then `derive_revenue`, then two declared plots. |
| `expert` | 🟣 Expert | **Full Pipeline** — cleaning + revenue derivation + both plots (higher step cap). |

Grading uses **cell-wise equality** (with float tolerance for imputed values) vs `tasks/<task>/ground_truth.csv`, weighted with plot correctness when `metadata.json` lists `expected_plots`.

---

## Action Space

`DataCleaningAction` fields:

| Field | Meaning |
|-------|---------|
| `action_type` | `remove_duplicates`, `fill_missing`, `drop_column`, `normalize`, `remove_outliers`, `derive_revenue`, `compute_metrics`, `compute_kpis`, `plot`, `export_csv`, `submit`, `noop` |
| `column` | Target column for column-wise ops |
| `method` | `mean` / `median` / `mode` — for `fill_missing` |
| `z_threshold` | Cutoff for robust outlier removal (modified z-score on `log1p` values) |
| `x`, `y`, `plot_type` | `scatter` / `bar` / `histogram` — for `plot` |

- **`derive_revenue`** adds `Revenue = Price × Quantity`
- **`compute_metrics`** produces category-level aggregates
- **`compute_kpis`** computes TotalRevenue + AvgOrderValue
- **`submit`** finalises the episode; `terminal_grader_score` in `[0, 1]` is returned

---

## Observation Space

`DataCleaningObservation` includes:

| Field | Description |
|-------|-------------|
| `preview` | First rows of the working dataframe |
| `issues` | Heuristic tags (`duplicates`, `missing_values`, …) |
| `task_name`, `task_difficulty` | Task metadata |
| `instruction` | Human-readable task description |
| `history` | Serialised actions taken so far |
| `reward`, `reward_breakdown` | Immediate, cumulative, and terminal grader score |
| `terminal_grader_score` | Final score when `done=True` |

---

## Reward Design

**Shaping rewards** — applied at each step:

- Small positive reward for productive actions (rows removed, nulls filled, etc.)
- Penalties for `noop`, destructive `drop_column`, and repeated identical actions

**Terminal reward** — applied on `submit` (or when `max_steps` is hit):

- `terminal_grader_score` in `[0, 1]` is added to the final step's reward
- This ensures the last transition carries the primary learning signal

---

## Gradio Web UI

An interactive demo is included at `app.py`:

```bash
pip install -e "./data_cleaning_env[openai]"
pip install gradio matplotlib

python app.py           # opens http://127.0.0.1:7861
python app.py --share   # public Gradio link
```

**Features:**

- Live **data preview** — table updates after each action
- **Issue detector** panel — duplicates, missing values, outliers
- **Manual action form** — dropdowns + fields → JSON preview
- **Oracle step / Run full oracle** buttons
- **Score badge** on submit
- **Plot gallery** — renders charts inline when plot actions are taken
- **Download Word report** (`.docx`) button
- Task selector with pipeline hints (dark mode compatible)

---

## Setup

```bash
cd data_cleaning_env
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Validate the OpenEnv layout:

```bash
openenv validate --verbose
```

---

## Run the FastAPI Server

```bash
cd data_cleaning_env
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

- Web UI: `/web`
- API docs: `/docs`
- Health check: `/health`

---

## Baseline Inference (Reproducible Scores)

### Oracle (no API key — deterministic 1.0 on all tasks)

```bash
cd data_cleaning_env

python baseline_inference.py --oracle

python baseline_inference.py --oracle --tasks easy,medium,medium_plus,hard,expert
```

### Word Reports

Install `python-docx`:

```bash
pip install -e ".[report]"
# or
pip install -e ".[openai]"
```

By default, reports are written to `./reports/` (`session_report.docx` plus per-task episode reports).

Override with `--report-dir DIR` or `AUTODATALAB_REPORT_DIR`. Disable with `--no-report`.

---

## LLM Baseline

### OpenAI

```bash
export OPENAI_API_KEY=sk-...
# optional: OPENAI_BASE_URL, MODEL_NAME (default: gpt-4o-mini)

python baseline_inference.py --provider openai
```

### Groq

```bash
export GROQ_API_KEY=gsk_...

export OPENAI_BASE_URL=https://api.groq.com/openai/v1
export MODEL_NAME=llama-3.1-8b-instant

python baseline_inference.py --provider groq
```

### Google Gemini

```bash
export GEMINI_API_KEY=...   # from Google AI Studio

# optional: GEMINI_MODEL=gemini-1.5-flash (default)

python baseline_inference.py --provider gemini
```

### Auto Provider Detection

`LLM_PROVIDER=auto` (or unset) checks keys in this order: `OPENAI_API_KEY` → `GROQ_API_KEY` → `GEMINI_API_KEY`.

> **Tip:** if `OPENAI_API_KEY` looks like a Groq key (`gsk_`) and `GEMINI_API_KEY` is also set, Gemini is used automatically. Use `--provider groq` or `--provider gemini` to force a specific backend.

### Tips for Faster LLM Runs

- Use `--tasks easy` (or `easy,medium`) for quick single-task checks
- Pass `--no-report` to skip Word export overhead
- Use `--provider groq` with `llama-3.1-8b-instant` for the fastest completions
- Set `LLM_JSON_MODE=0` if JSON mode causes extra round-trips on your backend
- Lower `--llm-retry-delay` only if you are not hitting 429 rate limits

---

## Baseline Scores (Oracle)

| Task | Terminal Grader |
|------|:---------------:|
| easy | **1.0** |
| medium | **1.0** |
| hard | **1.0** |
| **mean** | **1.0** |

---

## Docker (Local)

From the repository root:

```bash
docker build -t autodatalab-openenv .
docker run --rm -p 7860:7860 autodatalab-openenv
```

---

## Hugging Face Spaces

1. Push this repo to a **Docker** Space on Hugging Face.
2. Use the root **`Dockerfile`** (build context = repo root).
3. The server listens on **`PORT`** (default `7860`; matches `openenv.yaml`).
4. Tag the Space with **`openenv`** (see `data_cleaning_env/README.md` frontmatter).
5. Health check: `GET /health` should return **200**.

Or push directly from `data_cleaning_env/`:

```bash
openenv push
```

*(requires `huggingface-cli` login)*

---

## Submission / Course Alignment

This repo follows the OpenEnv environment pattern from [**Building RL Environments with OpenEnv**](https://github.com/raun/openenv-course).

**Root `inference.py`** — required entry point for hackathons:

| Variable | Role |
|----------|------|
| `API_BASE_URL` | OpenAI-compatible base URL |
| `MODEL_NAME` | Model ID |
| `HF_TOKEN` | API key (mapped to `OPENAI_API_KEY`) |

Copy **`.env.example`** to `.env` and fill values. Run `--oracle` to validate without an API key.

**Pre-flight checks:**

```bash
pip install -e ./data_cleaning_env
pip install -e "./data_cleaning_env[dev]"   # optional: pytest

python validate_submission.py
```

This runs `openenv validate`, `pytest`, grader checks on easy / medium / hard, and `python inference.py --oracle`. Add `--docker` to also run `docker build`.

---

## Project Layout

```
.
├── data_cleaning_env/        # OpenEnv package (models, server, client, tasks, openenv.yaml)
├── app.py                    # Gradio Web UI
├── Dockerfile                # HF Spaces / container entrypoint
├── inference.py              # Root inference script (delegates to baseline_inference.py)
├── validate_submission.py    # Pre-submission checks
└── .env.example              # Template for API_BASE_URL, MODEL_NAME, HF_TOKEN
```

---

---

## Submission Checklist

Use this before pasting your HF Spaces URL.

### Step 1 — Set environment variables

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

| Variable | Required | Description |
|----------|----------|-------------|
| `API_BASE_URL` | Yes | OpenAI-compatible base URL (e.g. Groq, OpenAI) |
| `MODEL_NAME` | Yes | Model ID (e.g. `llama-3.1-8b-instant`) |
| `HF_TOKEN` | Yes | Your API key (mapped to `OPENAI_API_KEY`) |

### Step 2 — Run the validator

```bash
python validate_submission.py
```

This automatically checks all of the following:

| Check | What it validates |
|-------|------------------|
| `openenv validate` | `openenv.yaml` spec, typed models, endpoint layout |
| `pytest` | Unit tests pass |
| Grader identity check | `easy`, `medium`, `hard` graders return scores in `[0.0, 1.0]` |
| Oracle inference | `inference.py --oracle` completes without error and produces scores |
| Docker build *(optional)* | `python validate_submission.py --docker` |

### Step 3 — Deploy to HF Spaces

```bash
# From data_cleaning_env/
openenv push --repo-id your-username/my-env
```

Or push the repo to a **Docker** Space manually (see Hugging Face Spaces section above).

### Step 4 — Verify the live Space

```bash
# Must return HTTP 200
curl https://your-username-my-env.hf.space/health

# Must accept reset()
curl -X POST https://your-username-my-env.hf.space/reset \
     -H "Content-Type: application/json" \
     -d '{"task": "easy"}'
```

### Step 5 — Final pre-submit check

- [ ] `inference.py` is in the **root directory**
- [ ] `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN` are defined in your environment / Space secrets
- [ ] All LLM calls go through the **OpenAI Python client** (`from openai import OpenAI`)
- [ ] Oracle inference finishes in **< 20 minutes** on 2 vCPU / 8 GB RAM
- [ ] `python validate_submission.py` exits with code **0**
- [ ] HF Space URL returns **200** and responds to **`POST /reset`**
- [ ] At least **3 tasks** are registered in `openenv.yaml` with working graders

---

## License

Environment scaffold includes Meta/OpenEnv BSD-style headers; see files inside `data_cleaning_env/`.
