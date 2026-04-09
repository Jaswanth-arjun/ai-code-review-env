"""Deterministic grader smoke tests (stdlib only)."""

import unittest

from env.graders import TaskGrader
from env.models import Action
from env.tasks import task_manager


class GraderTests(unittest.TestCase):
    def test_easy_perfect_roundtrip(self) -> None:
        task = task_manager.get_task("task_easy_1")
        g = TaskGrader(task.task_id, task.ground_truth)
        action = Action(
            issues_found=["Incorrect pagination offset skips first page"],
            severity="medium",
            suggestion="Use offset = (page - 1) * page_size for 1-based pages.",
            decision="needs_changes",
            confidence=0.9,
        )
        out = g.evaluate(action, steps_used=1, max_steps=5)
        self.assertGreaterEqual(out["total"], 0.85)
        self.assertLessEqual(out["total"], 1.0)

    def test_medium_reject_expected(self) -> None:
        task = task_manager.get_task("task_medium_1")
        g = TaskGrader(task.task_id, task.ground_truth)
        action = Action(
            issues_found=[
                "Timing attack vulnerability due to standard string comparison (!=)",
                "Replay attack vulnerability due to lack of payload timestamp validation",
            ],
            severity="high",
            suggestion="Use hmac.compare_digest for constant-time compare; add timestamp and reject stale events.",
            decision="reject",
            confidence=0.88,
        )
        out = g.evaluate(action, steps_used=1, max_steps=5)
        self.assertGreaterEqual(out["total"], 0.85)

    def test_grades_are_stable(self) -> None:
        task = task_manager.get_task("task_hard_1")
        g = TaskGrader(task.task_id, task.ground_truth)
        action = Action(
            issues_found=[
                "Race between balance check and debit allows double-spend under concurrency",
                "IDOR: from_account is not tied to authenticated user",
            ],
            severity="high",
            suggestion="Serialize transfers with per-account locks or DB transactions; authorize from_account == current_user.id.",
            decision="reject",
            confidence=0.9,
        )
        a = g.evaluate(action, 1, 5)["total"]
        b = g.evaluate(action, 1, 5)["total"]
        self.assertEqual(a, b)

    def test_medium_shell_injection_reject(self) -> None:
        task = task_manager.get_task("task_medium_2")
        g = TaskGrader(task.task_id, task.ground_truth)
        action = Action(
            issues_found=[
                "Remote code/command execution risk: subprocess with shell=True executes operator-controlled fragments",
                "bash -lc wrapper around user text remains unsafe even with quoting helpers",
            ],
            severity="high",
            suggestion="Remove shell entirely: use execve-style argument lists or a tightly scoped interpreter sandbox; never shell=True on operator text.",
            decision="reject",
            confidence=0.9,
        )
        out = g.evaluate(action, steps_used=1, max_steps=5)
        self.assertGreaterEqual(out["total"], 0.8)

    def test_scores_in_unit_interval(self) -> None:
        task = task_manager.get_task("task_easy_1")
        g = TaskGrader(task.task_id, task.ground_truth)
        action = Action(
            issues_found=[],
            severity="low",
            suggestion="lgtm",
            decision="approve",
            confidence=0.99,
        )
        out = g.evaluate(action, steps_used=5, max_steps=5)
        self.assertGreaterEqual(out["total"], 0.0)
        self.assertLessEqual(out["total"], 1.0)


if __name__ == "__main__":
    unittest.main()
