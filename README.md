---
title: AI Code Review Env
emoji: 🧪
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
---

# AI Code Review Environment

Real-world OpenEnv environment for training and evaluating AI agents on software code review tasks.

## 1) Motivation

Code review is a high-value, real engineering task with measurable outcomes. This environment simulates professional review workflows where the agent must identify defects, security vulnerabilities, and performance risks, then produce a decision and improvement guidance.

## 2) OpenEnv Interface

The environment follows `step() / reset() / state()` patterns implemented in `env/environment.py` with typed Pydantic models in `env/models.py`.

### Action Space

- `issues_found: List[str]`
- `severity: Literal["low", "medium", "high"]`
- `suggestion: str`
- `decision: Literal["approve", "reject", "needs_changes", "continue_review"]`
- `confidence: float` in `[0.0, 1.0]`

### Observation Space

- `code: str`
- `language: str`
- `history: List[str]`
- `current_step: int`
- `max_steps: int`

### Reward Space

- `score: float` in `[0.0, 1.0]`
- `details: dict`

Reward combines issue detection, severity match, suggestion quality, decision quality, confidence calibration, and step penalties.

## 3) Tasks and Difficulty

Defined in `env/tasks.py` with deterministic graders in `env/graders.py`.

1. Easy (`task_easy_1`)
- Detect division-by-zero risk in average calculation.

2. Medium (`task_medium_1`)
- Detect hardcoded credential and weak validation/security behavior.

3. Hard (`task_hard_1`)
- Detect multi-issue scenarios: complexity issues, injection risk, resource leaks, and indexing errors.

All graders output bounded scores in `[0.0, 1.0]`.

## 4) Baseline Inference

The required baseline runner is `inference.py` in repo root.

### Environment Variables

| Variable | Description |
|---|---|
| `API_KEY` or `OPENAI_API_KEY` | **Required.** API key for the LLM endpoint. |
| `API_BASE_URL` | The LLM API endpoint (default: `https://api.openai.com/v1`). |
| `MODEL_NAME` | The model identifier to use (default: `gpt-4`). |
| `HF_TOKEN` | Hugging Face token for deployment workflows. |

The inference script uses the **OpenAI client** initialized with the injected `API_BASE_URL` and `API_KEY` environment variables:

```python
client = OpenAI(
    base_url=os.environ.get("API_BASE_URL"),
    api_key=os.environ.get("API_KEY"),
)
```

### Run

```bash
pip install -r requirements.txt
API_KEY=<your-key> python inference.py
```

Outputs per-task scores, structured `[START]`/`[STEP]`/`[END]` logs, and writes `baseline_results.json`.

## 5) OpenEnv Server Endpoints

`server/app.py` exposes the runtime API expected by validators and Spaces.

- `POST /reset`
- `POST /step`
- `GET /state`

Run locally:

```bash
python -m server.app
```

## 6) Validation Checklist

```bash
python -m openenv.cli validate
python inference.py
```

Optional container checks:

```bash
docker build -t ai-code-review-env .
docker run --rm -e API_KEY=$API_KEY ai-code-review-env
```

## 7) Hugging Face Space Deployment

1. Push this repository to a HF Space configured as Docker.
2. Ensure environment variables are set in Space settings:
   - `API_KEY`
   - `API_BASE_URL`
   - `MODEL_NAME`
   - `HF_TOKEN`
3. Start Space and verify `POST /reset` returns HTTP 200.

## 8) Reproducibility Notes

- Deterministic task definitions and graders.
- Bounded score outputs.
- Baseline script logs full task-by-task outputs with structured `[START]`/`[STEP]`/`[END]` format.
