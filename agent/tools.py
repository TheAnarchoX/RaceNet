"""
Custom Tools for Copilot

Tools that give the Copilot agent the ability to interact with the repository.
"""

import json
import os
import re
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
        # === Planner Mode Tools ===
        {
            "name": "get_task_details",
            "description": "Get detailed information about a specific task including full description, requirements, and acceptance criteria",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "The task ID (e.g., '1.1', '2.3')"
                    }
                },
                "required": ["task_id"]
            }
        },
        {
            "name": "update_task",
            "description": "Update an existing task's details (description, requirements, priority, etc.)",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "The task ID to update"
                    },
                    "description": {
                        "type": "string",
                        "description": "New description for the task"
                    },
                    "priority": {
                        "type": "string",
                        "description": "New priority: P1, P2, or P3"
                    },
                    "requirements": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Updated list of requirements"
                    },
                    "acceptance_criteria": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Updated acceptance criteria"
                    },
                    "current_state": {
                        "type": "string",
                        "description": "Updated current implementation state"
                    }
                },
                "required": ["task_id"]
            }
        },
        {
            "name": "get_file_structure",
            "description": "Get a tree-view structure of a directory with file sizes and types",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to analyze (relative to repo root)"
                    },
                    "max_depth": {
                        "type": "integer",
                        "description": "Maximum depth to traverse (default: 3)"
                    },
                    "include_hidden": {
                        "type": "boolean",
                        "description": "Include hidden files/directories (default: false)"
                    }
                },
                "required": []
            }
        },
        {
            "name": "analyze_code_quality",
            "description": "Analyze code quality of a file or directory (complexity, issues, TODOs, missing docstrings)",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to file or directory to analyze"
                    }
                },
                "required": ["path"]
            }
        },
        {
            "name": "find_todos_and_fixmes",
            "description": "Find all TODO, FIXME, HACK, and XXX comments in the codebase",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to search in (default: entire repo)"
                    }
                },
                "required": []
            }
        },
        {
            "name": "get_test_coverage",
            "description": "Get test coverage information showing which files have tests and which don't",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        },
        {
            "name": "mark_task_complete",
            "description": "Mark a task or specific acceptance criterion as complete",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "The task ID to mark complete"
                    },
                    "criterion_index": {
                        "type": "integer",
                        "description": "Optional: specific criterion index (0-based) to mark complete"
                    }
                },
                "required": ["task_id"]
            }
        },
        {
            "name": "get_dependencies_graph",
            "description": "Get task dependencies as a graph showing which tasks block others",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
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
            # Planner mode tools
            "get_task_details": self._get_task_details,
            "update_task": self._update_task,
            "get_file_structure": self._get_file_structure,
            "analyze_code_quality": self._analyze_code_quality,
            "find_todos_and_fixmes": self._find_todos_and_fixmes,
            "get_test_coverage": self._get_test_coverage,
            "mark_task_complete": self._mark_task_complete,
            "get_dependencies_graph": self._get_dependencies_graph,
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
    
    # =========================================================================
    # Planner Mode Tools
    # =========================================================================
    
    def _get_task_details(self, params: dict) -> dict:
        """Get detailed information about a specific task."""
        self.task_manager.reload()
        task_id = params["task_id"]
        
        task = self.task_manager.tasks.get(task_id)
        if not task:
            return {"error": f"Task not found: {task_id}"}
        
        return {
            "id": task.id,
            "title": task.title,
            "priority": task.priority.name,
            "difficulty": task.difficulty.value,
            "status": task.status.value,
            "description": task.description,
            "dependencies": task.dependencies,
            "estimated_time": task.estimated_time,
            "requirements": task.requirements,
            "acceptance_criteria": [
                {"completed": c, "text": t} 
                for c, t in task.acceptance_criteria
            ],
            "files_to_modify": task.files_to_modify,
            "current_state": task.current_state,
            "completion_percentage": task.completion_percentage,
        }
    
    def _update_task(self, params: dict) -> dict:
        """Update an existing task's details (for use by planner)."""
        task_id = params["task_id"]
        self.task_manager.reload()
        
        task = self.task_manager.tasks.get(task_id)
        if not task:
            return {"error": f"Task not found: {task_id}"}
        
        # Collect updates (we'll store them for later TASKS.md modification)
        updates = {}
        if "description" in params:
            updates["description"] = params["description"]
        if "priority" in params:
            updates["priority"] = params["priority"]
        if "requirements" in params:
            updates["requirements"] = params["requirements"]
        if "acceptance_criteria" in params:
            updates["acceptance_criteria"] = params["acceptance_criteria"]
        if "current_state" in params:
            updates["current_state"] = params["current_state"]
        
        if self.config.dry_run:
            return {
                "status": "dry_run",
                "task_id": task_id,
                "updates": updates,
                "message": "Would update task (dry run). Use write_file to update TASKS.md"
            }
        
        return {
            "status": "pending",
            "task_id": task_id,
            "updates": updates,
            "message": "Task update prepared. Use write_file to update TASKS.md with these changes."
        }
    
    def _get_file_structure(self, params: dict) -> dict:
        """Get a tree-view structure of a directory."""
        path_str = params.get("path", ".")
        max_depth = params.get("max_depth", 3)
        include_hidden = params.get("include_hidden", False)
        
        path = self.config.repo_root / path_str
        if not path.exists():
            return {"error": f"Path not found: {path_str}"}
        if not path.is_dir():
            return {"error": f"Not a directory: {path_str}"}
        
        def build_tree(dir_path: Path, depth: int = 0) -> list[dict]:
            if depth >= max_depth:
                return []
            
            items = []
            try:
                for item in sorted(dir_path.iterdir()):
                    # Skip hidden files unless requested
                    if item.name.startswith('.') and not include_hidden:
                        continue
                    # Always skip .git
                    if item.name == ".git":
                        continue
                    
                    rel_path = str(item.relative_to(self.config.repo_root))
                    entry = {
                        "name": item.name,
                        "path": rel_path,
                        "type": "directory" if item.is_dir() else "file",
                    }
                    
                    if item.is_file():
                        try:
                            entry["size"] = item.stat().st_size
                            entry["extension"] = item.suffix
                        except OSError:
                            pass
                    
                    if item.is_dir() and depth + 1 < max_depth:
                        entry["children"] = build_tree(item, depth + 1)
                    
                    items.append(entry)
            except PermissionError:
                pass
            
            return items
        
        return {
            "root": path_str,
            "structure": build_tree(path),
        }
    
    def _analyze_code_quality(self, params: dict) -> dict:
        """Analyze code quality of a file or directory."""
        import re
        
        path_str = params["path"]
        path = self.config.repo_root / path_str
        
        if not path.exists():
            return {"error": f"Path not found: {path_str}"}
        
        files_to_analyze = []
        if path.is_file():
            files_to_analyze = [path]
        else:
            files_to_analyze = list(path.rglob("*.py"))
        
        results = {
            "total_files": len(files_to_analyze),
            "issues": [],
            "summary": {
                "missing_docstrings": 0,
                "todos": 0,
                "fixmes": 0,
                "long_functions": 0,
                "hardcoded_values": 0,
            }
        }
        
        for file_path in files_to_analyze[:50]:  # Limit for performance
            try:
                content = file_path.read_text()
                rel_path = str(file_path.relative_to(self.config.repo_root))
                
                # Check for missing docstrings on functions/classes
                func_pattern = r'^(\s*)(def|class)\s+(\w+)'
                for match in re.finditer(func_pattern, content, re.MULTILINE):
                    indent, keyword, name = match.groups()
                    # Check if next non-empty line is a docstring
                    pos = match.end()
                    remaining = content[pos:pos+200]
                    if '"""' not in remaining.split('\n')[0:3] and "'''" not in remaining.split('\n')[0:3]:
                        results["issues"].append({
                            "file": rel_path,
                            "type": "missing_docstring",
                            "message": f"{keyword} '{name}' missing docstring"
                        })
                        results["summary"]["missing_docstrings"] += 1
                
                # Count TODOs and FIXMEs
                for pattern, key in [("TODO", "todos"), ("FIXME", "fixmes")]:
                    count = len(re.findall(rf'#\s*{pattern}', content, re.IGNORECASE))
                    results["summary"][key] += count
                
                # Check for magic numbers (hardcoded values)
                magic_numbers = re.findall(r'(?<!["\'\w])(\d{3,})(?!["\'\w])', content)
                results["summary"]["hardcoded_values"] += len(magic_numbers)
                
            except Exception as e:
                results["issues"].append({
                    "file": str(file_path),
                    "type": "error",
                    "message": str(e)
                })
        
        # Limit issues returned
        results["issues"] = results["issues"][:30]
        
        return results
    
    def _find_todos_and_fixmes(self, params: dict) -> dict:
        """Find all TODO, FIXME, HACK, and XXX comments in the codebase."""
        import re
        
        path_str = params.get("path", ".")
        path = self.config.repo_root / path_str
        
        if not path.exists():
            return {"error": f"Path not found: {path_str}"}
        
        patterns = ["TODO", "FIXME", "HACK", "XXX", "BUG", "NOTE"]
        findings = []
        
        files = list(path.rglob("*.py")) if path.is_dir() else [path]
        
        for file_path in files[:100]:  # Limit files
            try:
                content = file_path.read_text()
                lines = content.split('\n')
                rel_path = str(file_path.relative_to(self.config.repo_root))
                
                for i, line in enumerate(lines):
                    for pattern in patterns:
                        if re.search(rf'#\s*{pattern}', line, re.IGNORECASE):
                            findings.append({
                                "file": rel_path,
                                "line": i + 1,
                                "type": pattern,
                                "content": line.strip()[:100]
                            })
            except Exception:
                pass
        
        return {
            "total": len(findings),
            "findings": findings[:50],  # Limit results
            "by_type": {
                pattern: len([f for f in findings if f["type"] == pattern])
                for pattern in patterns
            }
        }
    
    def _get_test_coverage(self, params: dict) -> dict:
        """Get test coverage information."""
        src_path = self.config.repo_root / "src"
        tests_path = self.config.repo_root / "tests"
        
        if not src_path.exists():
            return {"error": "No src/ directory found"}
        
        # Find all source files
        source_files = []
        for py_file in src_path.rglob("*.py"):
            if "__pycache__" not in str(py_file):
                rel_path = str(py_file.relative_to(self.config.repo_root))
                source_files.append(rel_path)
        
        # Find all test files
        test_files = []
        tested_modules = set()
        if tests_path.exists():
            for py_file in tests_path.rglob("test_*.py"):
                rel_path = str(py_file.relative_to(self.config.repo_root))
                test_files.append(rel_path)
                # Infer tested module from test file name
                module_name = py_file.stem.replace("test_", "")
                tested_modules.add(module_name)
        
        # Categorize source files
        covered = []
        not_covered = []
        
        for src_file in source_files:
            file_name = Path(src_file).stem
            if file_name == "__init__":
                continue
            if file_name in tested_modules:
                covered.append(src_file)
            else:
                not_covered.append(src_file)
        
        return {
            "source_files": len(source_files),
            "test_files": len(test_files),
            "covered_modules": covered,
            "not_covered_modules": not_covered,
            "coverage_percentage": (
                len(covered) / (len(covered) + len(not_covered)) * 100
                if (covered or not_covered) else 0
            )
        }
    
    def _mark_task_complete(self, params: dict) -> dict:
        """Mark a task or specific acceptance criterion as complete in TASKS.md."""
        task_id = params["task_id"]
        criterion_index = params.get("criterion_index")
        
        self.task_manager.reload()
        task = self.task_manager.tasks.get(task_id)
        
        if not task:
            return {"error": f"Task not found: {task_id}"}
        
        if self.config.dry_run:
            return {
                "status": "dry_run",
                "task_id": task_id,
                "message": "Would mark task/criterion complete (dry run)"
            }
        
        # Read current TASKS.md content
        tasks_path = self.task_manager.tasks_path
        content = tasks_path.read_text()
        
        # Find the task section in the file
        task_pattern = rf'(### Task {re.escape(task_id)}:.*?)(?=### Task \d+\.\d+:|## Task Template|## Contributing|$)'
        task_match = re.search(task_pattern, content, re.DOTALL)
        
        if not task_match:
            return {"error": f"Could not find Task {task_id} section in TASKS.md"}
        
        task_section = task_match.group(1)
        new_task_section = task_section
        
        if criterion_index is not None:
            # Mark specific criterion complete
            # Find all acceptance criteria checkboxes in the task section
            criteria_pattern = r'- \[ \]'
            matches = list(re.finditer(criteria_pattern, new_task_section))
            
            if criterion_index >= len(matches):
                return {"error": f"Criterion index {criterion_index} out of range (task has {len(matches)} unchecked criteria)"}
            
            # Replace the specific checkbox
            match = matches[criterion_index]
            new_task_section = (
                new_task_section[:match.start()] +
                '- [x]' +
                new_task_section[match.end():]
            )
            
            self.task_manager.mark_criterion_complete(task_id, criterion_index)
            message = f"Marked criterion {criterion_index} complete in TASKS.md"
        else:
            # Mark all criteria as complete
            new_task_section = re.sub(r'- \[ \]', '- [x]', new_task_section)
            
            from agent.task_manager import TaskStatus
            self.task_manager.update_task_status(task_id, TaskStatus.COMPLETED)
            message = f"Marked all acceptance criteria for Task {task_id} as complete in TASKS.md"
        
        # Replace the task section in the file
        new_content = content[:task_match.start()] + new_task_section + content[task_match.end():]
        
        # Write updated content back to file
        tasks_path.write_text(new_content)
        
        return {
            "status": "success",
            "task_id": task_id,
            "message": message
        }
    
    def _get_dependencies_graph(self, params: dict) -> dict:
        """Get task dependencies as a graph."""
        self.task_manager.reload()
        
        nodes = []
        edges = []
        blocked_by = {}
        
        for task_id, task in self.task_manager.tasks.items():
            nodes.append({
                "id": task_id,
                "title": task.title,
                "status": task.status.value,
                "priority": task.priority.name,
            })
            
            for dep_id in task.dependencies:
                edges.append({
                    "from": dep_id,
                    "to": task_id,
                })
                
                if task_id not in blocked_by:
                    blocked_by[task_id] = []
                blocked_by[task_id].append(dep_id)
        
        # Find tasks ready to work on (deps met, not completed)
        ready_tasks = []
        for task_id, task in self.task_manager.tasks.items():
            if task.status.value == "completed":
                continue
            
            deps_met = all(
                self.task_manager.tasks.get(dep_id, None) is None or
                self.task_manager.tasks[dep_id].status.value == "completed"
                for dep_id in task.dependencies
            )
            
            if deps_met:
                ready_tasks.append(task_id)
        
        return {
            "nodes": nodes,
            "edges": edges,
            "blocked_by": blocked_by,
            "ready_tasks": ready_tasks,
        }
    
    def get_proposed_tasks(self) -> list[dict]:
        """Get all proposed tasks."""
        return self._proposed_tasks
