"""Graders for evaluating agent performance."""

from typing import List, Dict, Any
import re


class BaseGrader:
    """Base class for all graders."""
    
    def grade(self, action: Any, ground_truth: Dict[str, Any]) -> float:
        """Grade the action against ground truth. Returns score 0.0-1.0."""
        raise NotImplementedError


class CodeReviewGrader(BaseGrader):
    """Grader for code review tasks."""
    
    def __init__(self, ground_truth: Dict[str, Any]):
        """
        Initialize grader with ground truth.
        
        ground_truth should contain:
        - issues: List of expected issues
        - severity: Expected severity
        - expected_decision: Expected decision
        - expected_suggestion_keywords: List of keywords for good suggestions
        """
        self.ground_truth = ground_truth
        
    def grade(self, action, max_steps_used: int = 1, max_steps: int = 5) -> Dict[str, float]:
        """
        Grade the action and return detailed scores.
        
        Returns dictionary with overall score and component scores.
        """
        scores = {
            'issue_detection': self._grade_issues(action.issues_found),
            'severity_match': self._grade_severity(action.severity),
            'suggestion_quality': self._grade_suggestion(action.suggestion),
            'decision_match': self._grade_decision(action.decision),
            'confidence_alignment': 0.0,
            'step_penalty': self._grade_steps(max_steps_used, max_steps),
        }

        correctness_proxy = (
            0.5 * scores['issue_detection']
            + 0.2 * scores['severity_match']
            + 0.2 * scores['suggestion_quality']
            + 0.1 * scores['decision_match']
        )
        scores['confidence_alignment'] = self._grade_confidence(action.confidence, correctness_proxy)
        
        # Weighted average of components
        weights = {
            'issue_detection': 0.4,
            'severity_match': 0.2,
            'suggestion_quality': 0.2,
            'decision_match': 0.1,
            'confidence_alignment': 0.1,
            'step_penalty': 0.0  # Penalty separate
        }
        
        total_score = sum(scores[k] * weights[k] for k in weights if weights[k] > 0)
        total_score = max(0.0, min(1.0, total_score - scores['step_penalty']))
        
        return {
            'total': total_score,
            'details': scores
        }
    
    def _tokenize(self, text: str) -> set:
        """Convert free-form issue text into normalized tokens for matching."""
        tokens = re.findall(r"[a-zA-Z0-9_]+", text.lower())
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'to', 'and', 'or', 'of', 'for', 'in',
            'on', 'with', 'from', 'by', 'when', 'be', 'no', 'not', 'missing'
        }
        return {t for t in tokens if len(t) > 2 and t not in stop_words}

    def _issue_match_score(self, expected_issue: str, detected_issue: str) -> float:
        """Compute overlap score between expected and detected issue strings."""
        expected_tokens = self._tokenize(expected_issue)
        detected_tokens = self._tokenize(detected_issue)
        if not expected_tokens:
            return 0.0
        overlap = len(expected_tokens & detected_tokens)
        return overlap / len(expected_tokens)

    def _grade_issues(self, detected_issues: List[str]) -> float:
        """Grade issue detection with semantic keyword overlap instead of exact string match."""
        expected_issues = self.ground_truth.get('issues', [])

        if not expected_issues:
            return 1.0 if len(detected_issues) == 0 else 0.0

        if not detected_issues:
            return 0.0

        matched_expected = 0
        for expected in expected_issues:
            best = max(self._issue_match_score(expected, detected) for detected in detected_issues)
            if best >= 0.4:
                matched_expected += 1

        precision_like = matched_expected / max(1, len(detected_issues))
        recall_like = matched_expected / len(expected_issues)
        if precision_like + recall_like == 0:
            return 0.0
        return 2 * precision_like * recall_like / (precision_like + recall_like)
    
    def _grade_severity(self, detected_severity: str) -> float:
        """Grade severity match."""
        expected_severity = self.ground_truth.get('severity', 'low')
        severity_map = {'low': 0, 'medium': 1, 'high': 2}
        
        expected_idx = severity_map.get(expected_severity, 0)
        detected_idx = severity_map.get(detected_severity, 0)
        
        # Return 1.0 if exact match, 0.5 if off by one, 0 if completely wrong
        if expected_idx == detected_idx:
            return 1.0
        elif abs(expected_idx - detected_idx) == 1:
            return 0.5
        else:
            return 0.0
    
    def _grade_suggestion(self, suggestion: str) -> float:
        """Grade suggestion quality based on keyword matching."""
        expected_keywords = self.ground_truth.get('suggestion_keywords', [])
        
        if not expected_keywords:
            return 1.0
        
        suggestion_lower = suggestion.lower()
        matched = sum(1 for kw in expected_keywords if kw.lower() in suggestion_lower)
        return matched / len(expected_keywords)
    
    def _grade_decision(self, decision: str) -> float:
        """Grade final decision."""
        expected_decision = self.ground_truth.get('expected_decision', 'needs_changes')
        return 1.0 if decision == expected_decision else 0.0
    
    def _grade_confidence(self, confidence: float, correctness_proxy: float) -> float:
        """Grade confidence alignment by comparing confidence with graded correctness."""
        gap = abs(confidence - correctness_proxy)
        return max(0.0, 1.0 - gap)
    
    def _grade_steps(self, steps_used: int, max_steps: int) -> float:
        """Penalize for using too many steps."""
        if steps_used <= 1:
            return 0.0
        # Penalty increases with steps used
        return min(0.3, (steps_used - 1) * 0.1)


class TaskGrader:
    """Manages grading for different tasks."""
    
    def __init__(self, task_id: str, ground_truth: Dict[str, Any]):
        self.task_id = task_id
        self.grader = CodeReviewGrader(ground_truth)
    
    def evaluate(self, action, steps_used: int = 1, max_steps: int = 5) -> Dict[str, float]:
        """Evaluate the agent's performance."""
        return self.grader.grade(action, steps_used, max_steps)