"""Task definitions for the AI Code Review Environment."""

from typing import List, Dict, Any, Literal


TaskDifficulty = Literal["easy", "medium", "hard"]


class CodeReviewTask:
    """A single code review task."""
    
    def __init__(
        self,
        task_id: str,
        code: str,
        language: str,
        ground_truth: Dict[str, Any],
        difficulty: TaskDifficulty,
        review_context: Dict[str, str] | None = None,
    ):
        self.task_id = task_id
        self.code = code
        self.language = language
        self.ground_truth = ground_truth
        self.difficulty = difficulty
        self.review_context = dict(review_context or {})
    
    def get_initial_state(self) -> Dict[str, Any]:
        """Get initial state for this task."""
        return {
            'code': self.code,
            'language': self.language,
            'ground_truth': self.ground_truth
        }


# Dataset of highly realistic code snippets exposing production-grade vulnerabilities
CODE_DATASET = [
    # Task 1: Easy - Subtle Business Logic Flaw
    {
        'id': 'task_easy_1',
        'review_context': {
            'pr_title': 'fix: customer list pagination',
            'file_path': 'services/api/query_utils.py',
            'team_policy': 'All DB helpers must be reviewed for off-by-one and SQL injection.',
            'risk_band': 'Customer-facing read path; PII-adjacent result sets.',
        },
        'code': '''
def get_paginated_results(db_session, query_base, page: int = 1, page_size: int = 20):
    """
    Fetch paginated results from the database.
    Assumes page is 1-indexed (e.g., page=1 means the first page).
    """
    if page < 1:
        page = 1
        
    # Calculate offset based on page number
    offset = page * page_size
    
    query = f"{query_base} LIMIT {page_size} OFFSET {offset}"
    return db_session.execute(query)
''',
        'language': 'python',
        'difficulty': 'easy',
        'ground_truth': {
            'issues': ['Incorrect pagination offset calculation skips the first page'],
            'severity': 'medium',
            'expected_decision': 'needs_changes',
            'suggestion_keywords': ['offset', 'page - 1', 'calculation', 'subtract', 'index']
        }
    },
    
    # Task 2: Medium - Modern Crypto & Security Flaws
    {
        'id': 'task_medium_1',
        'review_context': {
            'pr_title': 'chore: tighten GitHub webhook verification',
            'file_path': 'app/security/webhooks.py',
            'team_policy': 'Webhook handlers must be constant-time and replay-resistant.',
            'risk_band': 'Production ingress; forged events could trigger internal workflows.',
        },
        'code': '''
import hmac
import hashlib
from fastapi import Request, HTTPException

async def verify_github_webhook(request: Request, secret_token: str):
    """
    Verify incoming GitHub webhook payload signature.
    """
    payload_body = await request.body()
    signature_header = request.headers.get('X-Hub-Signature-256')
    
    if not signature_header:
        raise HTTPException(status_code=403, detail="Signature missing")

    # Calculate expected HMAC
    expected_mac = "sha256=" + hmac.new(
        secret_token.encode('utf-8'),
        payload_body,
        hashlib.sha256
    ).hexdigest()

    # Verify signature matches
    if signature_header != expected_mac:
        raise HTTPException(status_code=403, detail="Invalid signature")
        
    return payload_body
''',
        'language': 'python',
        'difficulty': 'medium',
        'ground_truth': {
            'issues': [
                'Timing attack vulnerability due to standard string comparison (!=)',
                'Replay attack vulnerability due to lack of payload timestamp validation'
            ],
            'severity': 'high',
            'expected_decision': 'reject',
            'suggestion_keywords': [
                'hmac.compare_digest', 'constant time', 'timing', 'compare_digest', 
                'timestamp', 'replay', 'expiration'
            ]
        }
    },

    # Task 3b: Medium — command execution (distinct from crypto/webhook class)
    {
        'id': 'task_medium_2',
        'review_context': {
            'pr_title': 'feat: admin preview for saved automation snippets',
            'file_path': 'tools/admin/snippet_runner.py',
            'team_policy': 'Never invoke a shell with operator-supplied strings; SOC2 releasable blocker.',
            'risk_band': 'Internal admin surface; input originates from partially trusted operators.',
        },
        'code': '''
import subprocess
import shlex

def preview_operator_snippet(snippet: str, timeout_sec: int = 5) -> str:
    """
    Run a one-liner saved by an operator so they can sanity-check output
    before promoting it to a scheduled job. Snippet is stored in our dashboard DB.
    """
    # NOTE: snippet is UTF-8 from the dashboard; admins are mostly trusted but not unix experts.
    cmd = f"bash -lc {shlex.quote(snippet)}"
    completed = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True,
        timeout=timeout_sec,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "preview failed")
    return completed.stdout
''',
        'language': 'python',
        'difficulty': 'medium',
        'ground_truth': {
            'issues': [
                'Remote code/command execution risk: subprocess with shell=True executes operator-controlled fragments',
                'Even with shlex.quote, composing bash -lc around user text is unsafe and unpredictable',
            ],
            'severity': 'high',
            'expected_decision': 'reject',
            'suggestion_keywords': [
                'shell', 'subprocess', 'shell=True', 'injection', 'argv', 'no shell',
                'exec', 'sandbox', 'whitelist', 'validate', 'disable',
            ],
        },
    },

    # Task 4: Hard - Distributed Systems & Concurrency Flaws
    {
        'id': 'task_hard_1',
        'review_context': {
            'pr_title': 'feat: async balance transfers',
            'file_path': 'api/ledger/transfers.py',
            'team_policy': 'Ledger mutations must be atomic per account; enforce ownership on from_account.',
            'risk_band': 'Money movement; concurrent requests expected at scale.',
        },
        'code': '''
from fastapi import FastAPI, Depends, HTTPException
import asyncio

app = FastAPI()

# Mock in-memory database representing Redis/Postgres
DB = {"users": {"usr_123": {"balance": 1000}, "usr_456": {"balance": 500}}}

async def get_current_user_token(authorization: str = None):
    # Simplistic mock token extraction (Assume this securely returns the authenticated user context)
    return {"id": "usr_123", "role": "standard"}

@app.post("/api/v1/transfer")
async def transfer_funds(from_account: str, to_account: str, amount: float, current_user = Depends(get_current_user_token)):
    """Async endpoint to transfer funds between accounts."""
    
    if amount <= 0:
        raise HTTPException(status_code=400, detail="Invalid transfer amount")
        
    if from_account not in DB["users"] or to_account not in DB["users"]:
        raise HTTPException(status_code=404, detail="Account not found")
        
    sender_balance = DB["users"][from_account]["balance"]
    
    if sender_balance < amount:
        raise HTTPException(status_code=400, detail="Insufficient funds")
        
    # Simulate async network call to auditing service before processing transfer
    await asyncio.sleep(0.1) 
    
    # Execute transfer
    DB["users"][from_account]["balance"] -= amount
    DB["users"][to_account]["balance"] += amount
    
    return {"status": "success", "tx_amount": amount}
''',
        'language': 'python',
        'difficulty': 'hard',
        'ground_truth': {
            'issues': [
                'Race condition in balance deduction (Time-of-Check to Time-of-Use) allowing negative balances',
                'Insecure Direct Object Reference (IDOR) missing authorization checking if current_user owns from_account'
            ],
            'severity': 'high',
            'expected_decision': 'reject',
            'suggestion_keywords': [
                'race condition', 'lock', 'atomic', 'transaction', 'mutex',
                'idor', 'authorization', 'authorize', 'ownership', 'current_user.id'
            ]
        }
    }
]


class TaskManager:
    """Manages tasks for the environment."""
    
    def __init__(self):
        self.tasks = []
        self._load_tasks()
    
    def _load_tasks(self):
        """Load all tasks from the dataset."""
        for task_data in CODE_DATASET:
            task = CodeReviewTask(
                task_id=task_data['id'],
                code=task_data['code'].strip(),
                language=task_data['language'],
                ground_truth=task_data['ground_truth'],
                difficulty=task_data['difficulty'],
                review_context=task_data.get('review_context'),
            )
            self.tasks.append(task)
    
    def get_task(self, task_id: str) -> CodeReviewTask:
        """Get a specific task by ID."""
        for task in self.tasks:
            if task.task_id == task_id:
                return task
        raise ValueError(f"Task {task_id} not found")
    
    def get_tasks_by_difficulty(self, difficulty: str) -> List[CodeReviewTask]:
        """Get all tasks of a specific difficulty."""
        return [t for t in self.tasks if t.difficulty == difficulty]
    
    def get_all_tasks(self) -> List[CodeReviewTask]:
        """Get all tasks."""
        return self.tasks
    
    def get_task_count(self) -> int:
        """Get number of tasks."""
        return len(self.tasks)


# Create a global task manager instance
task_manager = TaskManager()