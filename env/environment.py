"""Core environment implementation for AI Code Review."""

from typing import Dict, Any, Tuple, Optional
from env.models import Observation, Action, Reward
from env.tasks import task_manager
from env.graders import TaskGrader


class CodeReviewEnv:
    """
    OpenEnv-compliant environment for AI code review.
    
    Simulates a code review task where an AI agent must identify issues,
    suggest fixes, and make review decisions.
    """
    
    def __init__(self, task_id: Optional[str] = None):
        """
        Initialize the environment.
        
        Args:
            task_id: Specific task to use. If None, will be set on reset.
        """
        self.task_id = task_id
        self.current_task = None
        self.current_step = 0
        self.max_steps = 5
        self.history = []
        self.grader = None
        self.done = False
        self.total_reward = 0.0
        self.best_partial_quality = 0.0
        
    def reset(self, task_id: Optional[str] = None) -> Observation:
        """
        Reset the environment to initial state.
        
        Args:
            task_id: Optional specific task to use. If not provided,
                    uses the default task.
        
        Returns:
            Initial observation
        """
        # Set task
        if task_id:
            self.task_id = task_id
        elif not self.task_id:
            # Use first task by default
            self.task_id = task_manager.get_all_tasks()[0].task_id
        
        # Load task
        self.current_task = task_manager.get_task(self.task_id)
        self.current_step = 0
        self.history = []
        self.grader = TaskGrader(self.task_id, self.current_task.ground_truth)
        self.done = False
        self.total_reward = 0.0
        self.best_partial_quality = 0.0
        
        # Create initial observation
        return Observation(
            task_id=self.current_task.task_id,
            difficulty=self.current_task.difficulty,
            review_context=self.current_task.review_context,
            code=self.current_task.code,
            language=self.current_task.language,
            history=[],
            current_step=0,
            max_steps=self.max_steps,
        )
    
    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action: The agent's action (code review output)
        
        Returns:
            Tuple of (observation, reward, done, info)
        """
        if self.done:
            raise RuntimeError("Episode already finished. Call reset() first.")
        
        # Update state
        self.current_step += 1
        
        # Record action in history
        action_summary = f"Step {self.current_step}: {action.decision} (confidence: {action.confidence})"
        self.history.append(action_summary)
        
        # Calculate reward if this is the final step or we've reached max steps
        reward_value = 0.0
        reward_details = {}
        
        if self.current_step == self.max_steps or action.decision in ['approve', 'reject', 'needs_changes']:
            # Final evaluation
            grader_result = self.grader.evaluate(action, self.current_step, self.max_steps)
            reward_value = grader_result['total']
            reward_details = grader_result['details']
            self.done = True
        else:
            # Intermediate reward reflects incremental review quality improvements.
            grader_result = self.grader.evaluate(action, self.current_step, self.max_steps)
            details = grader_result['details']
            quality = (
                0.45 * details.get('issue_detection', 0.0)
                + 0.25 * details.get('severity_match', 0.0)
                + 0.20 * details.get('suggestion_quality', 0.0)
                + 0.10 * details.get('confidence_alignment', 0.0)
            )

            improvement = max(0.0, quality - self.best_partial_quality)
            stagnation_penalty = 0.02 if self.current_step > 1 and improvement == 0.0 else 0.0
            reward_value = max(0.0, min(0.25, 0.04 + (0.6 * improvement) - stagnation_penalty))
            self.best_partial_quality = max(self.best_partial_quality, quality)
            reward_details = {
                'partial_quality': quality,
                'improvement': improvement,
                'stagnation_penalty': stagnation_penalty,
                'base_grader_breakdown': details,
            }
        
        self.total_reward += reward_value
        
        # Create reward object
        reward = Reward(
            score=reward_value,
            details=reward_details
        )
        
        # Create next observation
        observation = Observation(
            task_id=self.current_task.task_id,
            difficulty=self.current_task.difficulty,
            review_context=self.current_task.review_context,
            code=self.current_task.code,
            language=self.current_task.language,
            history=self.history.copy(),
            current_step=self.current_step,
            max_steps=self.max_steps,
        )
        
        # Prepare info dict
        info = {
            'task_id': self.task_id,
            'difficulty': self.current_task.difficulty,
            'step': self.current_step,
            'total_reward': self.total_reward
        }
        
        return observation, reward, self.done, info
    
    def state(self) -> Dict[str, Any]:
        """
        Get the current state of the environment.
        
        Returns:
            Dictionary containing current state
        """
        return {
            'task_id': self.task_id,
            'current_step': self.current_step,
            'max_steps': self.max_steps,
            'history': self.history.copy(),
            'done': self.done,
            'total_reward': self.total_reward,
            'current_task': {
                'code': self.current_task.code if self.current_task else None,
                'language': self.current_task.language if self.current_task else None,
                'difficulty': self.current_task.difficulty if self.current_task else None
            }
        }
    
    def close(self):
        """Clean up resources."""
        # No resources to clean up in this implementation
        pass