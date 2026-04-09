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

### What is different here (novelty for judges)

- **Pull-request framing:** Every observation includes structured `review_context` (title, path, team policy, risk band). Agents must integrate policy text with the diff, not just pattern-match bare snippets.
- **Four concrete episodes:** Minimum requirement is three graded tasks; this env ships **four** distinct defect classes (pagination logic, webhook crypto, **shell/command misuse**, distributed ledger).
- **Transparent rubric:** Scoring is token overlap + severity + decision + calibrated confidence plus step penalties — reproducible and hard to “game” with constant outputs (see §3).

## 2) OpenEnv Interface

The environment follows `step() / reset() / state()` patterns implemented in `env/environment.py` with typed Pydantic models in `env/models.py`.

### Action Space

- `issues_found: List[str]`
- `severity: Literal["low", "medium", "high"]`
- `suggestion: str`
- `decision: Literal["approve", "reject", "needs_changes", "continue_review"]`
- `confidence: float` in `[0.0, 1.0]`

### Observation Space

- `task_id: str`
- `difficulty: "easy" | "medium" | "hard"`
- `review_context: dict[str, str]` — simulated PR metadata (`pr_title`, `file_path`, `team_policy`, …). **Does not contain hidden labels or answers.**
- `code: str`
- `language: str`
- `history: List[str]`
- `current_step: int` (number of completed steps at this point in the episode)
- `max_steps: int`

### Reward Space

- `score: float` in `[0.0, 1.0]`
- `details: dict`

Reward combines issue detection (fuzzy token overlap vs expected issue phrases), severity match, suggestion keyword coverage, **terminal decision** match, confidence calibration vs a correctness proxy, and step penalties. Intermediate steps using `continue_review` get a **shaped partial reward** (improvement minus stagnation), so trajectories are not all-or-nothing.

## 3) Tasks and Difficulty

Defined in `env/tasks.py` with deterministic graders in `env/graders.py`.

1. Easy (`task_easy_1`) — Pagination / off-by-one offset in SQL `LIMIT/OFFSET`.
2. Medium (`task_medium_1`) — Webhook HMAC: timing leak on compare and missing replay/timestamp defenses.
3. Medium (`task_medium_2`) — **Process invocation:** `subprocess` + `shell=True` around operator-controlled text (`bash -lc`), command-execution class distinct from webhook crypto.
4. Hard (`task_hard_1`) — Ledger transfer: async TOC/TOU on balance plus IDOR / missing ownership on `from_account`.

All graders output bounded scores in `[0.0, 1.0]` and are deterministic for identical actions.

## 4) Baseline Inference

The required baseline runner is `inference.py` in repo root.

### Environment Variables

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` / `API_KEY` / `HF_TOKEN` | **One required.** API key for the LLM (prefer `OPENAI_API_KEY` per hackathon spec; `HF_TOKEN` works as fallback on Spaces). |
| `API_BASE_URL` | The LLM API endpoint (default: `https://api.openai.com/v1`). |
| `MODEL_NAME` | The model identifier to use (default: `gpt-4`). |
| `HF_TOKEN` | Hugging Face token (also used as API key fallback when others are unset). |

The inference script uses the **OpenAI client** initialized with `API_BASE_URL` and the first available key:

```python
client = OpenAI(
    base_url=os.environ.get("API_BASE_URL"),
    api_key=os.environ.get("OPENAI_API_KEY")
    or os.environ.get("API_KEY")
    or os.environ.get("HF_TOKEN"),
)
```

### Run

```bash
pip install -r requirements.txt
API_KEY=<your-key> python inference.py
```

Outputs per-task scores, structured `[START]`/`[STEP]`/`[END]` logs, and writes `baseline_results.json`.

**Episode score (`[END]` and `normalized_score`):** mean per-step reward in `[0, 1]`, so a strong single-step answer is not penalized for not consuming all five steps.

## 5) OpenEnv Server Endpoints

`server/app.py` exposes the runtime API expected by validators and Spaces.

- `GET /` and `GET /health` — liveness (`status: ok`)
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

## 9) Reference baseline scores (illustrative)

Scores depend on the model, temperature, and endpoint. After a local run, read `baseline_results.json` and paste summarized numbers here for reviewers. Example shape (not a guarantee):

| Task | Difficulty | Normalized score (mean per-step reward) |
|------|------------|------------------------------------------|
| `task_easy_1` | easy | *run `inference.py`* |
| `task_medium_1` | medium | *run `inference.py`* |
| `task_medium_2` | medium | *run `inference.py`* |
| `task_hard_1` | hard | *run `inference.py`* |

Run unit checks:

```bash
python -m unittest tests.test_graders
```
