"""Pydantic models for the Code Review Environment."""

from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict


class Observation(BaseModel):
    """Observation from the code review environment."""
    
    task_id: str = Field(description="Active task identifier")
    difficulty: Literal["easy", "medium", "hard"] = Field(description="Task difficulty band")
    review_context: Dict[str, str] = Field(
        default_factory=dict,
        description="Simulated pull-request metadata (title, path, policy). No hidden answers.",
    )
    code: str = Field(description="The code to be reviewed")
    language: str = Field(description="Programming language of the code")
    history: List[str] = Field(default_factory=list, description="History of previous actions")
    current_step: int = Field(description="Number of completed steps in this episode")
    max_steps: int = Field(description="Maximum steps allowed")
    
    class Config:
        json_schema_extra = {
            "example": {
                "task_id": "task_easy_1",
                "difficulty": "easy",
                "review_context": {"pr_title": "[internal] pagination helper"},
                "code": "def add(a,b):\n    return a+b",
                "language": "python",
                "history": [],
                "current_step": 0,
                "max_steps": 5
            }
        }


class Action(BaseModel):
    """Action taken by the agent in code review."""
    
    issues_found: List[str] = Field(default_factory=list, description="List of issues detected")
    severity: Literal["low", "medium", "high"] = Field(description="Overall severity level")
    suggestion: str = Field(description="Suggested fix or improvement")
    decision: Literal["approve", "reject", "needs_changes", "continue_review"] = Field(
        description="Review decision; use continue_review to gather more evidence before finalizing"
    )
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the review (0.0-1.0)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "issues_found": ["Division by zero possible"],
                "severity": "high",
                "suggestion": "Add input validation",
                "decision": "needs_changes",
                "confidence": 0.85
            }
        }


class Reward(BaseModel):
    """Reward structure for code review actions."""
    
    score: float = Field(ge=0.0, le=1.0, description="Overall reward score (0.0-1.0)")
    details: dict = Field(default_factory=dict, description="Breakdown of reward components")
    
    class Config:
        json_schema_extra = {
            "example": {
                "score": 0.75,
                "details": {
                    "issue_detection": 0.4,
                    "severity_match": 0.2,
                    "suggestion_quality": 0.1,
                    "decision_match": 0.05,
                    "confidence_alignment": 0.0
                }
            }
        }