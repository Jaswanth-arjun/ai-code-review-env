"""Task definitions for the AI Code Review Environment."""

from typing import List, Dict, Any


class CodeReviewTask:
    """A single code review task."""
    
    def __init__(self, task_id: str, code: str, language: str, ground_truth: Dict[str, Any], difficulty: str):
        self.task_id = task_id
        self.code = code
        self.language = language
        self.ground_truth = ground_truth
        self.difficulty = difficulty
    
    def get_initial_state(self) -> Dict[str, Any]:
        """Get initial state for this task."""
        return {
            'code': self.code,
            'language': self.language,
            'ground_truth': self.ground_truth
        }


# Dataset of realistic code snippets with bugs
CODE_DATASET = [
    # Task 1: Easy - Simple bug detection
    {
        'id': 'task_easy_1',
        'code': '''
def calculate_average(numbers):
    """Calculate average of a list of numbers."""
    total = 0
    for i in range(len(numbers)):
        total += numbers[i]
    return total / len(numbers)
''',
        'language': 'python',
        'difficulty': 'easy',
        'ground_truth': {
            'issues': ['Division by zero when list is empty'],
            'severity': 'high',
            'expected_decision': 'reject',
            'suggestion_keywords': ['empty', 'check', 'if', 'len', 'handle']
        }
    },
    
    # Task 2: Medium - Logic and security issues
    {
        'id': 'task_medium_1',
        'code': '''
def authenticate_user(username, password):
    """Authenticate user with hardcoded credentials."""
    if username == "admin" and password == "admin123":
        print("Authentication successful")
        return True
    else:
        print("Invalid credentials")
        return False
    
def process_payment(amount, user_id):
    """Process payment without validation."""
    if amount > 0:
        print(f"Processing payment of ${amount} for user {user_id}")
        # Process payment logic here
        return True
    return False
''',
        'language': 'python',
        'difficulty': 'medium',
        'ground_truth': {
            'issues': [
                'Hardcoded credentials',
                'Plain text password exposure',
                'Missing input validation for user_id',
                'No rate limiting or security checks'
            ],
            'severity': 'high',
            'expected_decision': 'reject',
            'suggestion_keywords': [
                'database', 'hash', 'environment', 'variables',
                'validation', 'authentication', 'secure'
            ]
        }
    },
    
    # Task 3: Hard - Multiple issues with performance and logic
    {
        'id': 'task_hard_1',
        'code': '''
def find_duplicates(items):
    """Find duplicates in a list."""
    duplicates = []
    for i in range(len(items)):
        for j in range(len(items)):
            if i != j and items[i] == items[j]:
                if items[i] not in duplicates:
                    duplicates.append(items[i])
    return duplicates

def process_user_data(data):
    """Process user data with potential issues."""
    result = []
    for user in data:
        if user['active'] == True:
            # SQL injection vulnerability
            query = f"SELECT * FROM users WHERE id = {user['id']}"
            # Memory leak - not closing resources
            connection = create_connection()
            cursor = connection.cursor()
            cursor.execute(query)
            
            # Inefficient operation
            processed = []
            for i in range(len(result)):
                processed.append(result[i])
            
            # Off-by-one error
            for i in range(len(user.get('items', [])) + 1):
                result.append(user['items'][i])
    
    return result
''',
        'language': 'python',
        'difficulty': 'hard',
        'ground_truth': {
            'issues': [
                'Quadratic complexity O(n²) in find_duplicates',
                'SQL injection vulnerability',
                'Memory leak from unclosed connections',
                'Off-by-one error in loop boundary',
                'Inefficient list operations',
                'Missing error handling'
            ],
            'severity': 'high',
            'expected_decision': 'reject',
            'suggestion_keywords': [
                'set', 'hash', 'complexity', 'parameterized', 'queries',
                'context manager', 'with', 'error handling', 'optimization'
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
                difficulty=task_data['difficulty']
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