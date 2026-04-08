#!/usr/bin/env python3
"""Baseline inference script for AI Code Review Environment."""

import os
import json
from typing import Dict, Any
import sys
from openai import OpenAI
from env.environment import CodeReviewEnv
from env.models import Action
from env.tasks import task_manager


# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4")
API_KEY = os.getenv("API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
MAX_STEPS = 5
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))
MAX_TOKENS = 500


def _safe_text(value: str) -> str:
    """Normalize text for single-line structured logs."""
    return (value or "").replace("\n", " ").replace("\r", " ").strip()


def log_start(task: str, env_name: str, model: str) -> None:
    """Emit a structured start log line."""
    print(f"[START] task={task} env={env_name} model={model}", flush=True)


def log_step(step: int, action: Dict[str, Any], reward: float, done: bool, error: str | None) -> None:
    """Emit a structured per-step log line."""
    action_json = json.dumps(action, ensure_ascii=True)
    error_text = _safe_text(error) if error else ""
    print(
        f"[STEP] step={step} reward={reward:.4f} done={str(done).lower()} "
        f"error={error_text} action={action_json}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    """Emit a structured end log line."""
    rewards_text = ",".join(f"{r:.4f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.4f} rewards=[{rewards_text}]",
        flush=True,
    )


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


def parse_llm_response(response_text: str) -> Dict[str, Any]:
    """Parse LLM response into Action dictionary."""
    try:
        # Try to extract JSON from response
        # Find first { and last }
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
            
            return action_dict
        else:
            raise ValueError("No JSON found in response")
    
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error parsing response: {e}")
        print(f"Raw response: {response_text[:200]}")
        
        # Return a default action
        return {
            "issues_found": ["Unable to parse response"],
            "severity": "medium",
            "suggestion": "Please ensure response is in valid JSON format",
            "decision": "continue_review",
            "confidence": 0.5
        }


def generate_fallback_action(code: str) -> Dict[str, Any]:
    """Generate a simple local review action when LLM access is unavailable."""
    issues = []
    code_lower = code.lower()

    if "hardcoded" in code_lower or "password" in code_lower:
        issues.append("Potential hardcoded credentials or password handling issue")
    if "eval(" in code_lower or "exec(" in code_lower:
        issues.append("Unsafe dynamic code execution detected")
    if "for" in code_lower and ".append(" in code_lower and "sum(" in code_lower:
        issues.append("Possible performance issue from iterative accumulation")

    if not issues:
        issues = ["No critical issues detected by offline fallback reviewer"]
        severity = "low"
        decision = "approve"
        confidence = 0.55
    else:
        severity = "medium"
        decision = "needs_changes"
        confidence = 0.65

    return {
        "issues_found": issues,
        "severity": severity,
        "suggestion": "Validate input, remove insecure patterns, and add targeted tests.",
        "decision": decision,
        "confidence": confidence,
    }


def run_episode(env: CodeReviewEnv, client: OpenAI | None) -> Dict[str, Any]:
    """Run a single episode."""
    observation = env.reset()
    print(f"\n{'='*60}")
    print(f"Task: {env.current_task.difficulty.upper()} - {env.task_id}")
    print(f"{'='*60}")
    print(f"Code to review:\n{observation.code}\n")
    
    total_reward = 0.0
    rewards: list[float] = []
    log_start(task=env.task_id, env_name="ai-code-review-env", model=MODEL_NAME)
    
    for step in range(MAX_STEPS):
        print(f"\n--- Step {step + 1}/{MAX_STEPS} ---")
        
        # Prepare messages
        user_content = f"""Review the following code:
{observation.code}

Previous actions history: {observation.history if observation.history else 'None'}

Current step: {step + 1} of {MAX_STEPS}
Please provide your code review in JSON format as specified."""

        messages = [
            {"role": "system", "content": create_system_prompt()},
            {"role": "user", "content": user_content}
        ]
        
        error_text = None

        if client is None:
            response_text = json.dumps(generate_fallback_action(observation.code))
        else:
            try:
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                    stream=False
                )
                response_text = completion.choices[0].message.content or ""

            except Exception as e:
                error_text = _safe_text(str(e))
                print(f"Error calling LLM: {e}")
                response_text = json.dumps({
                    "issues_found": ["Error in LLM call"],
                    "severity": "medium",
                    "suggestion": f"Error: {e}",
                    "decision": "needs_changes",
                    "confidence": 0.3
                })
        
        action_dict = parse_llm_response(response_text)
        print(f"Action: {json.dumps(action_dict, indent=2)}")
        
        action = Action(**action_dict)
        
        observation, reward, done, info = env.step(action)
        total_reward += reward.score
        rewards.append(reward.score)
        
        print(f"Reward: {reward.score:.3f} | Total: {total_reward:.3f} | Done: {done}")
        print(f"Details: {reward.details}")
        log_step(step=step + 1, action=action_dict, reward=reward.score, done=done, error=error_text)
        
        if done:
            print("\nEpisode complete!")
            break

    max_total_reward = float(MAX_STEPS)
    normalized_score = min(max(total_reward / max_total_reward, 0.0), 1.0)
    success = normalized_score >= 0.6
    log_end(success=success, steps=env.current_step, score=normalized_score, rewards=rewards)
    
    return {
        'task_id': env.task_id,
        'difficulty': env.current_task.difficulty,
        'total_reward': total_reward,
        'normalized_score': normalized_score,
        'steps_taken': env.current_step,
        'history': env.history
    }


def main():
    """Main execution function."""
    client = None
    if not API_KEY:
        print("Warning: API_KEY not set")
        print("Running in offline fallback mode")
    else:
        client = OpenAI(
            base_url=API_BASE_URL,
            api_key=API_KEY
        )
    
    all_tasks = task_manager.get_all_tasks()
    results = []
    
    print(f"\n{'#'*60}")
    print(f"AI Code Review Environment - Baseline Evaluation")
    print(f"Model: {MODEL_NAME}")
    print(f"API Base: {API_BASE_URL}")
    print(f"Total tasks: {len(all_tasks)}")
    print(f"{'#'*60}")
    
    for task in all_tasks:
        env = CodeReviewEnv(task_id=task.task_id)
        result = run_episode(env, client)
        results.append(result)
        
        print(f"\n{'='*60}")
        print(f"Task {result['task_id']} ({result['difficulty']})")
        print(f"Score: {result['total_reward']:.3f}")
        print(f"Steps: {result['steps_taken']}")
        print(f"{'='*60}")
    
    print("\n\n" + "#"*60)
    print("SUMMARY")
    print("#"*60)
    
    for result in results:
        print(f"{result['task_id']} ({result['difficulty']}): {result['total_reward']:.3f}")
    
    avg_score = sum(r['total_reward'] for r in results) / len(results)
    print(f"\nAverage Score: {avg_score:.3f}")
    
    with open('baseline_results.json', 'w') as f:
        json.dump({
            'model': MODEL_NAME,
            'results': results,
            'average_score': avg_score
        }, f, indent=2)
    
    print("\nResults saved to baseline_results.json")


if __name__ == "__main__":
    main()