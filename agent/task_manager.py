"""
Task Manager

Parses TASKS.md and manages task state for the autonomous agent.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


class TaskStatus(Enum):
    """Status of a task."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"


class TaskPriority(Enum):
    """Priority level of a task."""
    P1 = 1
    P2 = 2
    P3 = 3


class TaskDifficulty(Enum):
    """Difficulty level of a task."""
    EASY = "Easy"
    MEDIUM = "Medium"
    HARD = "Hard"


@dataclass
class Task:
    """Represents a task from TASKS.md."""
    id: str
    title: str
    priority: TaskPriority
    difficulty: TaskDifficulty
    description: str
    dependencies: list[str] = field(default_factory=list)
    estimated_time: str = ""
    requirements: list[str] = field(default_factory=list)
    acceptance_criteria: list[tuple[bool, str]] = field(default_factory=list)
    files_to_modify: list[str] = field(default_factory=list)
    current_state: str = ""
    status: TaskStatus = TaskStatus.NOT_STARTED
    
    @property
    def is_completed(self) -> bool:
        """Check if all acceptance criteria are met."""
        if not self.acceptance_criteria:
            return False
        return all(checked for checked, _ in self.acceptance_criteria)
    
    @property
    def completion_percentage(self) -> float:
        """Get completion percentage based on acceptance criteria."""
        if not self.acceptance_criteria:
            return 0.0
        completed = sum(1 for checked, _ in self.acceptance_criteria if checked)
        return (completed / len(self.acceptance_criteria)) * 100


class TaskManager:
    """Manages tasks from TASKS.md."""
    
    def __init__(self, tasks_path: Path):
        self.tasks_path = tasks_path
        self.tasks: dict[str, Task] = {}
        self._load_tasks()
    
    def _load_tasks(self):
        """Load and parse tasks from TASKS.md."""
        if not self.tasks_path.exists():
            return
        
        content = self.tasks_path.read_text()
        self._parse_tasks(content)
    
    def _parse_tasks(self, content: str):
        """Parse task definitions from markdown content."""
        # Split by task headers (### Task X.Y:)
        task_pattern = r'### Task (\d+\.\d+): (.+?)(?=### Task \d+\.\d+:|## Task Template|## Contributing|$)'
        matches = re.findall(task_pattern, content, re.DOTALL)
        
        for task_id, task_content in matches:
            task = self._parse_single_task(task_id, task_content)
            if task:
                self.tasks[task.id] = task
    
    def _parse_single_task(self, task_id: str, content: str) -> Optional[Task]:
        """Parse a single task from its markdown content."""
        lines = content.strip().split('\n')
        if not lines:
            return None
        
        title = lines[0].strip()
        
        # Extract priority
        priority_match = re.search(r'\*\*Priority\*\*:\s*(P\d)', content)
        priority = TaskPriority.P2
        if priority_match:
            priority = TaskPriority[priority_match.group(1)]
        
        # Extract difficulty
        difficulty_match = re.search(r'\*\*Difficulty\*\*:\s*(Easy|Medium|Hard)', content)
        difficulty = TaskDifficulty.MEDIUM
        if difficulty_match:
            difficulty = TaskDifficulty(difficulty_match.group(1))
        
        # Extract dependencies
        deps_match = re.search(r'\*\*Dependencies\*\*:\s*(.+)', content)
        dependencies = []
        if deps_match:
            deps_str = deps_match.group(1).strip()
            if deps_str.lower() != "none":
                # Extract task IDs like "Task 1.1" or just "1.1"
                dep_ids = re.findall(r'(?:Task\s+)?(\d+\.\d+)', deps_str)
                dependencies = dep_ids
        
        # Extract estimated time
        time_match = re.search(r'\*\*Estimated Time\*\*:\s*(.+)', content)
        estimated_time = time_match.group(1).strip() if time_match else ""
        
        # Extract description
        desc_match = re.search(r'\*\*Description\*\*:\s*\n(.+?)(?=\*\*Current State\*\*|\*\*Requirements\*\*|$)', content, re.DOTALL)
        description = desc_match.group(1).strip() if desc_match else ""
        
        # Extract current state
        state_match = re.search(r'\*\*Current State\*\*:\s*\n(.+?)(?=\*\*Requirements\*\*|$)', content, re.DOTALL)
        current_state = state_match.group(1).strip() if state_match else ""
        
        # Extract requirements
        req_match = re.search(r'\*\*Requirements\*\*:\s*\n(.+?)(?=\*\*Acceptance Criteria\*\*|$)', content, re.DOTALL)
        requirements = []
        if req_match:
            req_lines = req_match.group(1).strip().split('\n')
            requirements = [
                re.sub(r'^\d+\.\s*', '', line.strip())
                for line in req_lines
                if line.strip() and re.match(r'^\d+\.', line.strip())
            ]
        
        # Extract acceptance criteria
        criteria_match = re.search(r'\*\*Acceptance Criteria\*\*:\s*\n(.+?)(?=\*\*Files to Modify\*\*|$)', content, re.DOTALL)
        acceptance_criteria = []
        if criteria_match:
            criteria_lines = criteria_match.group(1).strip().split('\n')
            for line in criteria_lines:
                line = line.strip()
                if line.startswith('- [x]'):
                    acceptance_criteria.append((True, line[5:].strip()))
                elif line.startswith('- [ ]'):
                    acceptance_criteria.append((False, line[5:].strip()))
        
        # Extract files to modify
        files_match = re.search(r'\*\*Files to Modify\*\*:\s*\n(.+?)(?=---|$)', content, re.DOTALL)
        files_to_modify = []
        if files_match:
            file_lines = files_match.group(1).strip().split('\n')
            files_to_modify = [
                line.strip().lstrip('- ').strip('`')
                for line in file_lines
                if line.strip().startswith('-')
            ]
        
        # Determine status
        status = TaskStatus.NOT_STARTED
        if acceptance_criteria:
            if all(checked for checked, _ in acceptance_criteria):
                status = TaskStatus.COMPLETED
            elif any(checked for checked, _ in acceptance_criteria):
                status = TaskStatus.IN_PROGRESS
        
        return Task(
            id=task_id,
            title=title,
            priority=priority,
            difficulty=difficulty,
            description=description,
            dependencies=dependencies,
            estimated_time=estimated_time,
            requirements=requirements,
            acceptance_criteria=acceptance_criteria,
            files_to_modify=files_to_modify,
            current_state=current_state,
            status=status,
        )
    
    def get_next_task(self) -> Optional[Task]:
        """Get the next task to work on (highest priority, dependencies met)."""
        available_tasks = []
        
        for task in self.tasks.values():
            if task.status == TaskStatus.COMPLETED:
                continue
            
            # Check dependencies
            deps_met = all(
                self.tasks.get(dep_id, Task(dep_id, "", TaskPriority.P1, TaskDifficulty.EASY, "")).status == TaskStatus.COMPLETED
                for dep_id in task.dependencies
            )
            
            if deps_met:
                available_tasks.append(task)
        
        if not available_tasks:
            return None
        
        # Sort by priority (P1 first), then by ID
        available_tasks.sort(key=lambda t: (t.priority.value, t.id))
        return available_tasks[0]
    
    def get_tasks_by_priority(self, priority: TaskPriority) -> list[Task]:
        """Get all tasks with a specific priority."""
        return [t for t in self.tasks.values() if t.priority == priority]
    
    def get_tasks_by_status(self, status: TaskStatus) -> list[Task]:
        """Get all tasks with a specific status."""
        return [t for t in self.tasks.values() if t.status == status]
    
    def update_task_status(self, task_id: str, status: TaskStatus):
        """Update the status of a task."""
        if task_id in self.tasks:
            self.tasks[task_id].status = status
    
    def mark_criterion_complete(self, task_id: str, criterion_index: int):
        """Mark a specific acceptance criterion as complete."""
        if task_id in self.tasks:
            task = self.tasks[task_id]
            if 0 <= criterion_index < len(task.acceptance_criteria):
                _, text = task.acceptance_criteria[criterion_index]
                task.acceptance_criteria[criterion_index] = (True, text)
    
    def get_summary(self) -> dict:
        """Get a summary of task status."""
        total = len(self.tasks)
        completed = len([t for t in self.tasks.values() if t.status == TaskStatus.COMPLETED])
        in_progress = len([t for t in self.tasks.values() if t.status == TaskStatus.IN_PROGRESS])
        not_started = len([t for t in self.tasks.values() if t.status == TaskStatus.NOT_STARTED])
        
        return {
            "total": total,
            "completed": completed,
            "in_progress": in_progress,
            "not_started": not_started,
            "completion_percentage": (completed / total * 100) if total > 0 else 0,
        }
    
    def to_markdown(self) -> str:
        """Convert current task state back to markdown format."""
        # For now, just return the original file content
        # A full implementation would regenerate the markdown
        if self.tasks_path.exists():
            return self.tasks_path.read_text()
        return ""
    
    def reload(self):
        """Reload tasks from file."""
        self.tasks.clear()
        self._load_tasks()
