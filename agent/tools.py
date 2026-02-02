"""
Custom Tools for Copilot

Tools that give the Copilot agent the ability to interact with the repository.
"""

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Callable, Optional

from agent.config import AgentConfig
from agent.task_manager import TaskManager

try:
    from copilot.types import Tool, ToolInvocation, ToolResult
    HAS_COPILOT_SDK = True
except ImportError:
    Tool = None  # type: ignore[assignment]
    ToolInvocation = dict  # type: ignore[assignment]
    ToolResult = dict  # type: ignore[assignment]
    HAS_COPILOT_SDK = False


def _build_tool_handler(tool_handler: "ToolHandler") -> Callable[[ToolInvocation], ToolResult]:
    async def handler(invocation: ToolInvocation) -> ToolResult:
        try:
            arguments = invocation.get("arguments") or {}
            result = tool_handler.handle_tool_call(invocation.get("tool_name", ""), arguments)
            return {
                "textResultForLlm": result,
                "resultType": "success",
                "toolTelemetry": {},
            }
        except Exception as exc:  # pylint: disable=broad-except
            return {
                "textResultForLlm": "Invoking this tool produced an error.",
                "resultType": "failure",
                "error": str(exc),
                "toolTelemetry": {},
            }

    return handler


def create_tool_definitions(
    config: AgentConfig,
    tool_handler: Optional["ToolHandler"] = None,
) -> list[Any]:
    """Create tool definitions for the Copilot SDK."""
    definitions = [
        {
            "name": "read_file",
            "description": "Read the contents of a file in the repository",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file relative to repository root"
                    }
                },
                "required": ["path"]
            }
        },
        {
            "name": "write_file",
            "description": "Write content to a file in the repository. Creates the file if it doesn't exist.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file relative to repository root"
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write to the file"
                    }
                },
                "required": ["path", "content"]
            }
        },
        {
            "name": "list_directory",
            "description": "List files and directories in a path",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the directory relative to repository root. Use '.' for root."
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "Whether to list recursively (default: false)"
                    }
                },
                "required": ["path"]
            }
        },
        {
            "name": "run_command",
            "description": "Run a shell command in the repository",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to run"
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds (default: 300)"
                    }
                },
                "required": ["command"]
            }
        },
        {
            "name": "run_tests",
            "description": "Run pytest tests for the project",
            "parameters": {
                "type": "object",
                "properties": {
                    "test_path": {
                        "type": "string",
                        "description": "Specific test file or directory (default: all tests)"
                    },
                    "verbose": {
                        "type": "boolean",
                        "description": "Enable verbose output"
                    }
                },
                "required": []
            }
        },
        {
            "name": "get_tasks",
            "description": "Get all tasks from TASKS.md with their status",
            "parameters": {
                "type": "object",
                "properties": {
                    "status_filter": {
                        "type": "string",
                        "description": "Filter by status: 'not_started', 'in_progress', 'completed', or 'all'"
                    }
                },
                "required": []
            }
        },
        {
            "name": "get_next_task",
            "description": "Get the next task to work on (highest priority with dependencies met)",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        },
        {
            "name": "get_task_summary",
            "description": "Get a summary of overall task progress",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        },
        {
            "name": "propose_new_task",
            "description": "Propose a new task to add to TASKS.md",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Title of the new task"
                    },
                    "priority": {
                        "type": "string",
                        "description": "Priority: P1, P2, or P3"
                    },
                    "difficulty": {
                        "type": "string",
                        "description": "Difficulty: Easy, Medium, or Hard"
                    },
                    "description": {
                        "type": "string",
                        "description": "Detailed description of what needs to be done"
                    },
                    "dependencies": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of task IDs this depends on"
                    },
                    "estimated_time": {
                        "type": "string",
                        "description": "Estimated time to complete (e.g., '2-3 hours')"
                    },
                    "requirements": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of specific requirements"
                    },
                    "acceptance_criteria": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of acceptance criteria"
                    },
                    "files_to_modify": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of files that will need modification"
                    }
                },
                "required": ["title", "priority", "difficulty", "description"]
            }
        },
        {
            "name": "search_code",
            "description": "Search for patterns in code files using grep",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "The pattern to search for"
                    },
                    "file_pattern": {
                        "type": "string",
                        "description": "File pattern to search in (e.g., '*.py')"
                    }
                },
                "required": ["pattern"]
            }
        },
        {
            "name": "git_status",
            "description": "Get the current git status",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        },
        {
            "name": "git_commit",
            "description": "Commit current changes with a message",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Commit message"
                    }
                },
                "required": ["message"]
            }
        },
    ]

    if not HAS_COPILOT_SDK or tool_handler is None:
        return definitions

    handler = _build_tool_handler(tool_handler)
    return [
        Tool(
            name=definition["name"],
            description=definition["description"],
            parameters=definition.get("parameters"),
            handler=handler,
        )
        for definition in definitions
    ]


class ToolHandler:
    """Handles tool execution for the agent."""
    
    def __init__(self, config: AgentConfig, task_manager: TaskManager):
        self.config = config
        self.task_manager = task_manager
        self._proposed_tasks: list[dict] = []
    
    def handle_tool_call(self, name: str, parameters: dict[str, Any]) -> str:
        """Execute a tool and return the result."""
        handlers = {
            "read_file": self._read_file,
            "write_file": self._write_file,
            "list_directory": self._list_directory,
            "run_command": self._run_command,
            "run_tests": self._run_tests,
            "get_tasks": self._get_tasks,
            "get_next_task": self._get_next_task,
            "get_task_summary": self._get_task_summary,
            "propose_new_task": self._propose_new_task,
            "search_code": self._search_code,
            "git_status": self._git_status,
            "git_commit": self._git_commit,
        }
        
        handler = handlers.get(name)
        if not handler:
            return json.dumps({"error": f"Unknown tool: {name}"})
        
        try:
            result = handler(parameters)
            return json.dumps(result) if isinstance(result, (dict, list)) else str(result)
        except Exception as e:
            return json.dumps({"error": str(e)})
    
    def _read_file(self, params: dict) -> dict:
        """Read a file from the repository."""
        path = self.config.repo_root / params["path"]
        if not path.exists():
            return {"error": f"File not found: {params['path']}"}
        if not path.is_file():
            return {"error": f"Not a file: {params['path']}"}
        
        try:
            content = path.read_text()
            return {"path": params["path"], "content": content, "size": len(content)}
        except Exception as e:
            return {"error": f"Failed to read file: {e}"}
    
    def _write_file(self, params: dict) -> dict:
        """Write content to a file."""
        if self.config.dry_run:
            return {"status": "dry_run", "path": params["path"], "message": "Would write file (dry run)"}
        
        path = self.config.repo_root / params["path"]
        
        # Create parent directories if needed
        path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            path.write_text(params["content"])
            return {"status": "success", "path": params["path"], "size": len(params["content"])}
        except Exception as e:
            return {"error": f"Failed to write file: {e}"}
    
    def _list_directory(self, params: dict) -> dict:
        """List directory contents."""
        dir_path = params.get("path", ".")
        recursive = params.get("recursive", False)
        
        path = self.config.repo_root / dir_path
        if not path.exists():
            return {"error": f"Directory not found: {dir_path}"}
        if not path.is_dir():
            return {"error": f"Not a directory: {dir_path}"}
        
        items = []
        if recursive:
            for item in path.rglob("*"):
                if ".git" not in str(item):
                    rel_path = item.relative_to(self.config.repo_root)
                    items.append({
                        "path": str(rel_path),
                        "type": "directory" if item.is_dir() else "file"
                    })
        else:
            for item in path.iterdir():
                if item.name != ".git":
                    rel_path = item.relative_to(self.config.repo_root)
                    items.append({
                        "path": str(rel_path),
                        "type": "directory" if item.is_dir() else "file"
                    })
        
        return {"directory": dir_path, "items": items}
    
    def _run_command(self, params: dict) -> dict:
        """Run a shell command."""
        command = params["command"]
        timeout = params.get("timeout", 300)
        
        if self.config.dry_run:
            return {"status": "dry_run", "command": command, "message": "Would run command (dry run)"}
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=self.config.repo_root,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            return {
                "command": command,
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
        except subprocess.TimeoutExpired:
            return {"error": f"Command timed out after {timeout} seconds"}
        except Exception as e:
            return {"error": f"Failed to run command: {e}"}
    
    def _run_tests(self, params: dict) -> dict:
        """Run pytest tests."""
        test_path = params.get("test_path", "")
        verbose = params.get("verbose", False)
        
        cmd = ["python", "-m", "pytest"]
        if verbose:
            cmd.append("-v")
        if test_path:
            cmd.append(test_path)
        
        if self.config.dry_run:
            return {"status": "dry_run", "command": " ".join(cmd)}
        
        try:
            result = subprocess.run(
                cmd,
                cwd=self.config.repo_root,
                capture_output=True,
                text=True,
                timeout=600
            )
            return {
                "exit_code": result.returncode,
                "passed": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
        except Exception as e:
            return {"error": f"Failed to run tests: {e}"}
    
    def _get_tasks(self, params: dict) -> dict:
        """Get tasks from TASKS.md."""
        self.task_manager.reload()
        status_filter = params.get("status_filter", "all")
        
        tasks = []
        for task in self.task_manager.tasks.values():
            if status_filter != "all" and task.status.value != status_filter:
                continue
            tasks.append({
                "id": task.id,
                "title": task.title,
                "priority": task.priority.name,
                "difficulty": task.difficulty.value,
                "status": task.status.value,
                "dependencies": task.dependencies,
                "completion_percentage": task.completion_percentage,
            })
        
        return {"tasks": sorted(tasks, key=lambda t: (t["priority"], t["id"]))}
    
    def _get_next_task(self, params: dict) -> dict:
        """Get the next task to work on."""
        self.task_manager.reload()
        task = self.task_manager.get_next_task()
        
        if not task:
            return {"message": "No available tasks. All tasks are either completed or blocked by dependencies."}
        
        return {
            "id": task.id,
            "title": task.title,
            "priority": task.priority.name,
            "difficulty": task.difficulty.value,
            "description": task.description,
            "requirements": task.requirements,
            "acceptance_criteria": [{"completed": c, "text": t} for c, t in task.acceptance_criteria],
            "files_to_modify": task.files_to_modify,
            "current_state": task.current_state,
        }
    
    def _get_task_summary(self, params: dict) -> dict:
        """Get task summary."""
        self.task_manager.reload()
        return self.task_manager.get_summary()
    
    def _propose_new_task(self, params: dict) -> dict:
        """Propose a new task."""
        task_proposal = {
            "title": params["title"],
            "priority": params["priority"],
            "difficulty": params["difficulty"],
            "description": params["description"],
            "dependencies": params.get("dependencies", []),
            "estimated_time": params.get("estimated_time", ""),
            "requirements": params.get("requirements", []),
            "acceptance_criteria": params.get("acceptance_criteria", []),
            "files_to_modify": params.get("files_to_modify", []),
        }
        
        self._proposed_tasks.append(task_proposal)
        
        # Generate the task ID based on existing tasks
        existing_ids = list(self.task_manager.tasks.keys())
        if existing_ids:
            max_phase = max(int(tid.split('.')[0]) for tid in existing_ids)
            phase_tasks = [tid for tid in existing_ids if tid.startswith(f"{max_phase}.")]
            if phase_tasks:
                max_task = max(int(tid.split('.')[1]) for tid in phase_tasks)
                new_id = f"{max_phase}.{max_task + 1}"
            else:
                new_id = f"{max_phase}.1"
        else:
            new_id = "1.1"
        
        task_proposal["id"] = new_id
        
        return {
            "status": "proposed",
            "task_id": new_id,
            "message": f"Task '{params['title']}' proposed with ID {new_id}. Use write_file to update TASKS.md with this task."
        }
    
    def _search_code(self, params: dict) -> dict:
        """Search for patterns in code."""
        pattern = params["pattern"]
        file_pattern = params.get("file_pattern", "*")
        
        try:
            cmd = ["grep", "-rn", "--include", file_pattern, pattern, str(self.config.repo_root)]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            matches = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    matches.append(line)
            
            return {"pattern": pattern, "matches": matches[:50]}  # Limit results
        except Exception as e:
            return {"error": f"Search failed: {e}"}
    
    def _git_status(self, params: dict) -> dict:
        """Get git status."""
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=self.config.repo_root,
                capture_output=True,
                text=True
            )
            
            changes = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    status = line[:2]
                    file_path = line[3:]
                    changes.append({"status": status.strip(), "path": file_path})
            
            return {"changes": changes, "has_changes": len(changes) > 0}
        except Exception as e:
            return {"error": f"Git status failed: {e}"}
    
    def _git_commit(self, params: dict) -> dict:
        """Commit changes."""
        message = params["message"]
        
        if self.config.dry_run:
            return {"status": "dry_run", "message": "Would commit (dry run)"}
        
        try:
            # Stage all changes
            subprocess.run(["git", "add", "."], cwd=self.config.repo_root, check=True)
            
            # Commit
            result = subprocess.run(
                ["git", "commit", "-m", message],
                cwd=self.config.repo_root,
                capture_output=True,
                text=True
            )
            
            return {
                "status": "committed" if result.returncode == 0 else "no_changes",
                "message": message,
                "output": result.stdout
            }
        except Exception as e:
            return {"error": f"Git commit failed: {e}"}
    
    def get_proposed_tasks(self) -> list[dict]:
        """Get all proposed tasks."""
        return self._proposed_tasks
