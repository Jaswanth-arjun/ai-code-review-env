#!/usr/bin/env python3
"""Baseline inference script for AI Code Review Environment."""

import os
import json
import sys
from typing import Dict, Any, List
from openai import OpenAI
from env.environment import CodeReviewEnv
from env.models import Action
from env.tasks import task_manager


# ── Configuration ────────────────────────────────────────────────────────────
# Read the proxy endpoint and key injected by the hackathon validator.
# Support both API_KEY (primary, per hackathon docs) and OPENAI_API_KEY (fallback).
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4")
API_KEY = os.environ.get("API_KEY") or os.environ.get("OPENAI_API_KEY", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

MAX_STEPS = 5
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.0"))
MAX_TOKENS = 500
SUCCESS_SCORE_THRESHOLD = 0.6
MAX_TOTAL_REWARD = float(MAX_STEPS)


# ── Structured logging helpers ───────────────────────────────────────────────

def _safe_text(value: str) -> str:
    """Normalize text for single-line structured logs."""
    return (value or "").replace("\n", " ").replace("\r", " ").strip()


def log_start(task: str, env: str, model: str) -> None:
    """Emit a structured start log line."""
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    """Emit a structured per-step log line."""
    error_text = _safe_text(error) if error else ""
    print(
        f"[STEP] step={step} reward={reward:.4f} done={str(done).lower()} "
        f"error={error_text} action={action}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    """Emit a structured end log line."""
    rewards_text = ",".join(f"{r:.4f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.4f} rewards=[{rewards_text}]",
        flush=True,
    )


# ── LLM interaction ─────────────────────────────────────────────────────────

def create_system_prompt() -> str:
    """Create system prompt for the LLM."""
    return """You are an AI code reviewer. Your task is to analyze code, identify issues, and make review decisions.

You must output your response in the following JSON format:
{
    "issues_found": ["list", "of", "issues"],
    "severity": "low|medium|high",
    "suggestion": "Your detailed suggestion for improvement",
    "decision": "approve|reject|needs_changes|continue_review",
    "confidence": 0.0-1.0
}

Consider:
- Security vulnerabilities
- Performance issues
- Logic errors
- Code quality
- Best practices

Be specific and professional in your suggestions."""


def get_model_response(client: OpenAI, code: str, history: List[str], step: int, max_steps: int) -> str:
    """Call the LLM through the proxy and return the raw response text.

    This function ALWAYS makes an API call through the provided client.
    If the call fails, it raises the exception so the caller can handle it.
    """
    user_content = f"""Review the following code:
{code}

Previous actions history: {history if history else 'None'}

Current step: {step} of {max_steps}
Please provide your code review in JSON format as specified."""

    messages = [
        {"role": "system", "content": create_system_prompt()},
        {"role": "user", "content": user_content}
    ]

    print(f"[DEBUG] Calling LLM via proxy at {API_BASE_URL} with model {MODEL_NAME}", flush=True)

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        stream=False,
    )
    return completion.choices[0].message.content or ""


def parse_llm_response(response_text: str) -> Dict[str, Any]:
    """Parse LLM response into Action dictionary."""
    try:
        # Try to extract JSON from response
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1

        if start_idx >= 0 and end_idx > start_idx:
            json_str = response_text[start_idx:end_idx]
            action_dict = json.loads(json_str)

            # Validate required fields
            required_fields = ['issues_found', 'severity', 'suggestion', 'decision', 'confidence']
            for field in required_fields:
                if field not in action_dict:
                    raise ValueError(f"Missing field: {field}")

            # Clamp confidence to [0, 1]
            action_dict['confidence'] = min(max(float(action_dict['confidence']), 0.0), 1.0)

            # Normalize severity
            if action_dict['severity'] not in ('low', 'medium', 'high'):
                action_dict['severity'] = 'medium'

            # Normalize decision
            valid_decisions = ('approve', 'reject', 'needs_changes', 'continue_review')
            if action_dict['decision'] not in valid_decisions:
                action_dict['decision'] = 'needs_changes'

            return action_dict
        else:
            raise ValueError("No JSON found in response")

    except (json.JSONDecodeError, ValueError) as e:
        print(f"[DEBUG] Error parsing response: {e}", flush=True)
        print(f"[DEBUG] Raw response: {response_text[:300]}", flush=True)

        # Return a sensible default when parsing fails
        return {
            "issues_found": ["Unable to parse LLM response"],
            "severity": "medium",
            "suggestion": "Please ensure response is in valid JSON format",
            "decision": "needs_changes",
            "confidence": 0.5,
        }


# ── Episode runner ───────────────────────────────────────────────────────────

def run_episode(env: CodeReviewEnv, client: OpenAI) -> Dict[str, Any]:
    """Run a single episode, always using the LLM proxy for decisions."""
    observation = env.reset()
    print(f"\n{'='*60}", flush=True)
    print(f"Task: {env.current_task.difficulty.upper()} - {env.task_id}", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Code to review:\n{observation.code}\n", flush=True)

    total_reward = 0.0
    rewards: List[float] = []
    steps_taken = 0
    history: List[str] = []

    log_start(task=env.task_id, env="ai-code-review-env", model=MODEL_NAME)

    for step in range(1, MAX_STEPS + 1):
        print(f"\n--- Step {step}/{MAX_STEPS} ---", flush=True)

        error_text = None

        # ── ALWAYS call the LLM through the proxy ──
        try:
            response_text = get_model_response(
                client, observation.code, history, step, MAX_STEPS
            )
            print(f"[DEBUG] LLM response received ({len(response_text)} chars)", flush=True)
        except Exception as exc:
            error_text = _safe_text(str(exc))
            print(f"[DEBUG] Model request failed: {exc}", flush=True)
            # On API error, use a simple default so the episode can continue
            response_text = json.dumps({
                "issues_found": ["LLM call failed, using default review"],
                "severity": "medium",
                "suggestion": "Unable to get LLM response, defaulting to needs_changes",
                "decision": "needs_changes",
                "confidence": 0.3,
            })

        action_dict = parse_llm_response(response_text)
        print(f"Action: {json.dumps(action_dict, indent=2)}", flush=True)

        action = Action(**action_dict)

        observation, reward, done, info = env.step(action)
        reward_score = reward.score
        total_reward += reward_score
        rewards.append(reward_score)
        steps_taken = step

        print(f"Reward: {reward_score:.3f} | Total: {total_reward:.3f} | Done: {done}", flush=True)
        print(f"Details: {reward.details}", flush=True)

        action_json = json.dumps(action_dict, ensure_ascii=True)
        log_step(step=step, action=action_json, reward=reward_score, done=done, error=error_text)

        history.append(f"Step {step}: {action_dict['decision']!r} -> reward {reward_score:+.2f}")

        if done:
            print("\nEpisode complete!", flush=True)
            break

    score = sum(rewards) / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else 0.0
    score = min(max(score, 0.0), 1.0)
    success = score >= SUCCESS_SCORE_THRESHOLD

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {
        'task_id': env.task_id,
        'difficulty': env.current_task.difficulty,
        'total_reward': total_reward,
        'normalized_score': score,
        'steps_taken': steps_taken,
        'history': history,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    """Main execution function."""
    # ── Validate that API credentials exist ──
    if not API_KEY:
        print(
            "ERROR: No API key found. Set API_KEY or OPENAI_API_KEY environment variable.",
            flush=True,
        )
        sys.exit(1)

    # ── Always create the OpenAI client using the injected proxy URL + key ──
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY,
    )
    print(f"[DEBUG] OpenAI client created — base_url={API_BASE_URL}", flush=True)

    all_tasks = task_manager.get_all_tasks()
    results: List[Dict[str, Any]] = []

    print(f"\n{'#'*60}", flush=True)
    print(f"AI Code Review Environment - Baseline Evaluation", flush=True)
    print(f"Model: {MODEL_NAME}", flush=True)
    print(f"API Base: {API_BASE_URL}", flush=True)
    print(f"Total tasks: {len(all_tasks)}", flush=True)
    print(f"{'#'*60}", flush=True)

    for task in all_tasks:
        env = CodeReviewEnv(task_id=task.task_id)
        result = run_episode(env, client)
        results.append(result)

        print(f"\n{'='*60}", flush=True)
        print(f"Task {result['task_id']} ({result['difficulty']})", flush=True)
        print(f"Score: {result['normalized_score']:.3f}", flush=True)
        print(f"Steps: {result['steps_taken']}", flush=True)
        print(f"{'='*60}", flush=True)

    print("\n\n" + "#" * 60, flush=True)
    print("SUMMARY", flush=True)
    print("#" * 60, flush=True)

    for result in results:
        print(
            f"{result['task_id']} ({result['difficulty']}): "
            f"score={result['normalized_score']:.3f}  reward={result['total_reward']:.3f}",
            flush=True,
        )

    avg_score = sum(r['normalized_score'] for r in results) / len(results) if results else 0.0
    print(f"\nAverage Score: {avg_score:.3f}", flush=True)

    with open('baseline_results.json', 'w') as f:
        json.dump({
            'model': MODEL_NAME,
            'results': results,
            'average_score': avg_score,
        }, f, indent=2)

    print("\nResults saved to baseline_results.json", flush=True)


if __name__ == "__main__":
    main()