"""
Multi-Agent Orchestrator

Manages multiple autonomous agents working together as a hive mind.
Coordinates task assignment, prevents conflicts, and synchronizes knowledge.
"""

import asyncio
import json
import logging
import random
import signal
import sys
import tracemalloc
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

from agent.config import AgentConfig
from agent.memory import (
    RepositoryMemory,
    KnowledgeEntry,
    KnowledgeType,
    Importance,
)
from agent.perf import PerfLogger
from agent.task_manager import TaskManager, Task, TaskStatus
from agent.tools import ToolHandler, create_tool_definitions
from agent.self_improvement import ActionOutcome, OutcomeType, SelfImprovementEngine

try:
    from copilot.types import Tool, ToolInvocation, ToolResult
    HAS_COPILOT_SDK = True
except ImportError:
    Tool = None  # type: ignore[assignment]
    ToolInvocation = dict  # type: ignore[assignment]
    ToolResult = dict  # type: ignore[assignment]
    HAS_COPILOT_SDK = False


logger = logging.getLogger(__name__)


class AgentRole(Enum):
    """Roles that agents can take."""
    GENERALIST = "generalist"         # Works on any task
    PHYSICS = "physics"               # Specializes in physics tasks
    TRACK = "track"                   # Specializes in track generation
    ML = "ml"                         # Specializes in ML integration
    TESTING = "testing"               # Focuses on tests
    DOCUMENTATION = "documentation"   # Focuses on docs
    REVIEWER = "reviewer"             # Reviews other agents' work


@dataclass
class OrchestratorConfig:
    """Configuration for the multi-agent orchestrator."""
    
    # Agent settings
    num_agents: int = 3
    model: str = "gpt-5.2-codex"
    
    # Repository settings
    repo_root: Path = field(default_factory=lambda: Path.cwd())
    tasks_file: str = "TASKS.md"
    
    # Orchestration settings
    max_total_iterations: int = 500
    task_timeout_minutes: int = 30
    sync_interval_seconds: int = 30
    heartbeat_interval_seconds: int = 10
    turn_timeout_seconds: int | None = None
    fast_mode: bool = False
    
    # Agent roles (if empty, all are generalists)
    agent_roles: list[AgentRole] = field(default_factory=list)
    
    # Behavior
    auto_propose_tasks: bool = True
    enable_peer_review: bool = True
    require_tests: bool = True

    # Advanced capabilities
    enable_self_improvement: bool = True
    
    # Safety
    dry_run: bool = False
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[Path] = None

    # Performance logging
    perf_log_path: Optional[Path] = None
    perf_sample_interval_seconds: int = 30
    enable_tracemalloc: bool = False
    
    # Copilot SDK
    cli_url: Optional[str] = None


class AgentWorker:
    """
    A single agent worker that participates in the hive.
    
    Each worker:
    - Has a unique ID and optional role
    - Shares memory with other workers
    - Coordinates via the orchestrator
    - Can be assigned tasks
    """
    
    def __init__(
        self,
        agent_id: str,
        config: OrchestratorConfig,
        memory: RepositoryMemory,
        task_manager: TaskManager,
        role: AgentRole = AgentRole.GENERALIST,
        self_improvement: SelfImprovementEngine | None = None,
    ):
        self.agent_id = agent_id
        self.config = config
        self.memory = memory
        self.task_manager = task_manager
        self.role = role
        self.self_improvement = self_improvement
        
        # Create agent-specific config
        self.agent_config = AgentConfig(
            model=config.model,
            repo_root=config.repo_root,
            dry_run=config.dry_run,
            cli_url=config.cli_url,
            perf_log_path=config.perf_log_path,
            perf_sample_interval_seconds=config.perf_sample_interval_seconds,
            enable_tracemalloc=config.enable_tracemalloc,
            turn_timeout_seconds=config.turn_timeout_seconds,
            fast_mode=config.fast_mode,
        )

        self.perf_logger = PerfLogger(config.perf_log_path, agent_id=agent_id)
        
        self.tool_handler = ToolHandler(
            self.agent_config,
            task_manager,
            memory=memory,
            self_improvement=self_improvement,
            agent_id=agent_id,
            perf_logger=self.perf_logger,
        )
        self._current_task: Optional[Task] = None
        self._running = False
        self._shutting_down = False
        self._response_content = ""
        self._last_turn_end_time: float | None = None
        self._last_turn_start_time: float | None = None
        self._tool_start_times: dict[str, float] = {}
        self._tool_call_names: dict[str, str] = {}
        self._perf_task: asyncio.Task | None = None
        self._last_event_time = time.monotonic()
        self._inflight_send_task: asyncio.Task | None = None
        self._client = None
        self._session = None
        self._session_config: dict | None = None
    
    async def start(self):
        """Start the agent worker."""
        self._running = True

        if self.config.enable_tracemalloc:
            tracemalloc.start()
            if self.perf_logger.enabled:
                self.perf_logger.log("tracemalloc.start")
        
        # Register with memory
        self.memory.register_agent(self.agent_id)
        self.memory.update_agent_state(self.agent_id, status="starting")
        
        logger.info(f"[{self.agent_id}] Starting (role: {self.role.value})")
        
        try:
            self._start_perf_sampler()
            await self._run()
        except asyncio.CancelledError:
            logger.info(f"[{self.agent_id}] Cancelled")
        except Exception as e:
            logger.exception(f"[{self.agent_id}] Error: {e}")
        finally:
            self.memory.update_agent_state(self.agent_id, status="stopped")
            await self._cleanup()
    
    async def stop(self):
        """Stop the agent worker."""
        self._running = False
    
    async def _run(self):
        """Main agent loop."""
        # Determine whether to use mock client
        use_mock = self.config.dry_run
        
        if not use_mock:
            # Try to import Copilot SDK
            try:
                from copilot import CopilotClient
            except ImportError:
                logger.warning(f"[{self.agent_id}] Copilot SDK not available, using mock")
                use_mock = True
        
        if use_mock:
            from agent.autonomous_agent import MockCopilotClient
            self._client = MockCopilotClient()
            self._client.tool_handler = self.tool_handler
            logger.debug(f"[{self.agent_id}] Using mock client for dry-run mode")
        else:
            # Initialize real client
            client_options = {}
            if self.config.cli_url:
                client_options["cli_url"] = self.config.cli_url
            self._client = CopilotClient(**client_options)
        
        # Start client explicitly (SDK best practice)
        try:
            await self._client.start()
            logger.info(f"[{self.agent_id}] Client started")
        except (FileNotFoundError, ConnectionError) as exc:
            logger.warning(
                f"[{self.agent_id}] Copilot client unavailable ({exc}); "
                "falling back to mock client"
            )
            from agent.autonomous_agent import MockCopilotClient
            self._client = MockCopilotClient()
            self._client.tool_handler = self.tool_handler
            await self._client.start()
        
        # Create session with hive mind context
        tools = create_tool_definitions(self.agent_config, self.tool_handler)
        tools.extend(self._get_hive_tools())
        
        system_message = self._get_system_message()
        
        self._session = await self._client.create_session(
            {
                "system_message": {"content": system_message},
                "model": self.config.model,
                "tools": tools,
            }
        )
        self._session_config = {
            "system_message": {"content": system_message},
            "model": self.config.model,
            "tools": tools,
        }
        
        # Register event handler using on() pattern (SDK best practice)
        self._session.on(self._handle_event)
        
        logger.info(f"[{self.agent_id}] Session created: {self._session.session_id}")
        
        self.memory.update_agent_state(self.agent_id, status="idle")
        
        # Calculate per-worker iteration limit
        max_iterations_per_worker = max(
            1, 
            self.config.max_total_iterations // self.config.num_agents
        )
        iterations = 0
        
        # Main loop
        while self._running and not self._shutting_down and iterations < max_iterations_per_worker:
            iterations += 1
            
            # Check for messages from other agents
            await self._process_messages()
            
            # Get next task if idle
            if not self._current_task:
                self._current_task = await self._get_next_task()
            
            if self._current_task:
                await self._work_on_task()
            else:
                # No tasks available, wait
                await asyncio.sleep(5)
            
            # Heartbeat
            self.memory.update_agent_state(self.agent_id, status="idle")
        
        logger.info(f"[{self.agent_id}] Completed {iterations} iterations")
    
    def _build_hive_tool_handler(self) -> Callable[[ToolInvocation], ToolResult]:
        async def handler(invocation: ToolInvocation) -> ToolResult:
            try:
                arguments = invocation.get("arguments") or {}
                result = await self._handle_tool_call(
                    invocation.get("tool_name", ""),
                    arguments,
                )
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

    def _get_hive_tools(self) -> list[Any]:
        """Get additional tools for hive coordination."""
        definitions = [
            {
                "name": "share_knowledge",
                "description": "Share a piece of knowledge with other agents",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "knowledge_type": {
                            "type": "string",
                            "description": "Type: code_fact, pattern, task_learning, bug_fix, optimization, convention"
                        },
                        "content": {
                            "type": "string",
                            "description": "The knowledge content"
                        },
                        "importance": {
                            "type": "string",
                            "description": "Importance: low, medium, high, critical"
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Tags for categorization"
                        },
                        "related_file": {
                            "type": "string",
                            "description": "Related file path"
                        }
                    },
                    "required": ["knowledge_type", "content"]
                }
            },
            {
                "name": "query_knowledge",
                "description": "Query the shared knowledge base",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query"
                        },
                        "knowledge_type": {
                            "type": "string",
                            "description": "Filter by type"
                        },
                        "file_path": {
                            "type": "string",
                            "description": "Filter by related file"
                        }
                    },
                    "required": []
                }
            },
            {
                "name": "message_agents",
                "description": "Send a message to other agents",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "Message content"
                        },
                        "message_type": {
                            "type": "string",
                            "description": "Type: info, warning, request_help, completed, discovered"
                        }
                    },
                    "required": ["message", "message_type"]
                }
            },
            {
                "name": "get_hive_status",
                "description": "Get status of all agents in the hive",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
            {
                "name": "acquire_file",
                "description": "Acquire a lock on a file before editing",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the file"
                        }
                    },
                    "required": ["file_path"]
                }
            },
            {
                "name": "release_file",
                "description": "Release a file lock after editing",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the file"
                        }
                    },
                    "required": ["file_path"]
                }
            },
            {
                "name": "mark_knowledge_useful",
                "description": "Mark a piece of knowledge as useful (positive feedback)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "knowledge_id": {
                            "type": "string",
                            "description": "ID of the knowledge entry"
                        }
                    },
                    "required": ["knowledge_id"]
                }
            },
        ]

        if not HAS_COPILOT_SDK:
            return definitions

        handler = self._build_hive_tool_handler()
        return [
            Tool(
                name=definition["name"],
                description=definition["description"],
                parameters=definition.get("parameters"),
                handler=handler,
            )
            for definition in definitions
        ]

    def _get_system_message(self) -> str:
        """Get the system message with hive context."""
        base_message = f"""You are Agent {self.agent_id}, part of a hive working on RaceNet.

    Role: {self.role.value}. {"Focus on tasks related to " + self.role.value if self.role != AgentRole.GENERALIST else "You can work on any type of task."}

    Coordination:
    - Query shared knowledge first.
    - Acquire/release file locks before/after edits.
    - Share useful findings and message peers when blocked.

    Autonomy:
    - Do NOT ask for user confirmation. If a decision is needed, pick a reasonable default and proceed.
    - If you create a plan, immediately execute it. Never wait for “start”.

    Current hive status:
    """
        # Add current hive status
        summary = self.memory.get_summary()
        base_message += f"""
    - Active agents: {summary['active_agents']}
    - Total shared knowledge: {summary['total_knowledge']}
    - Agent IDs: {', '.join(summary['agent_ids'])}

    Workflow:
    1) Read -> 2) Edit -> 3) Test -> 4) Share knowledge -> 5) Release locks -> 6) Commit.
    """
        return base_message
    
    async def _get_next_task(self) -> Optional[Task]:
        """Get the next task appropriate for this agent's role."""
        self.task_manager.reload()
        
        # Get available tasks
        available = []
        for task in self.task_manager.tasks.values():
            if task.status == TaskStatus.COMPLETED:
                continue
            
            # Check dependencies
            deps_met = all(
                self.task_manager.tasks.get(dep_id, Task(dep_id, "", task.priority, task.difficulty, "")).status == TaskStatus.COMPLETED
                for dep_id in task.dependencies
            )
            
            if not deps_met:
                continue
            
            # Check if another agent is working on it
            agents = self.memory.get_active_agents()
            being_worked = any(a.current_task == task.id for a in agents if a.agent_id != self.agent_id)
            
            if being_worked:
                continue
            
            # Role-based filtering
            if self.role != AgentRole.GENERALIST:
                task_lower = task.title.lower() + task.description.lower()
                role_keywords = {
                    AgentRole.PHYSICS: ["physics", "tire", "engine", "aero", "suspension", "dynamics"],
                    AgentRole.TRACK: ["track", "racing line", "elevation", "kerb", "segment"],
                    AgentRole.ML: ["ml", "learning", "agent", "environment", "reward", "curriculum"],
                    AgentRole.TESTING: ["test", "coverage", "unit test", "integration"],
                    AgentRole.DOCUMENTATION: ["documentation", "readme", "docs", "example"],
                }
                
                keywords = role_keywords.get(self.role, [])
                if keywords and not any(kw in task_lower for kw in keywords):
                    continue
            
            available.append(task)
        
        if not available:
            return None
        
        # Sort by priority and pick
        available.sort(key=lambda t: (t.priority.value, t.id))
        return available[0]
    
    async def _work_on_task(self):
        """Work on the current task."""
        task = self._current_task
        if not task:
            return

        start_time = asyncio.get_event_loop().time()
        
        self.memory.update_agent_state(
            self.agent_id,
            status="working",
            current_task=task.id
        )
        
        logger.info(f"[{self.agent_id}] Working on Task {task.id}: {task.title}")
        
        # Get context from knowledge base
        context = self.memory.get_context_for_task(task.id, task.files_to_modify)
        context = self._truncate(context) if context else "None"
        description = self._truncate(task.description)
        requirements = self._truncate(", ".join(task.requirements)) if task.requirements else "None"
        acceptance = self._truncate(", ".join(t for _, t in task.acceptance_criteria)) if task.acceptance_criteria else "None"
        
        # Create prompt
        prompt = f"""Task {task.id}: {task.title}

    Shared knowledge: {context}
    Priority: {task.priority.name} | Difficulty: {task.difficulty.value}
    Description: {description}
    Requirements: {requirements}
    Acceptance: {acceptance}
    Files: {', '.join(task.files_to_modify) if task.files_to_modify else 'Not specified'}

    Instructions: query_knowledge, acquire locks, implement, test, share, release locks, commit.
    Autonomy: do NOT ask for confirmation. If you create a plan, execute it immediately.
    """

        try:
            await self._send_message(prompt)
            await self._check_and_start_implementation()
            self._maybe_mark_task_complete(task)
            
            # Task completed (or attempted)
            self.memory.update_agent_state(
                self.agent_id,
                tasks_completed=self.memory.get_agent_state(self.agent_id).tasks_completed + 1
            )
            
            # Broadcast completion
            self.memory.broadcast(
                self.agent_id,
                "completed",
                f"Completed work on Task {task.id}: {task.title}"
            )

            if self.self_improvement and self.config.enable_self_improvement:
                duration = max(0.0, asyncio.get_event_loop().time() - start_time)
                outcome_id = f"task_{task.id}_{int(datetime.now().timestamp() * 1000)}"
                self.self_improvement.record_outcome(
                    ActionOutcome(
                        id=outcome_id,
                        action_type="work_on_task",
                        outcome=OutcomeType.SUCCESS,
                        agent_id=self.agent_id,
                        task_id=task.id,
                        duration_seconds=duration,
                    )
                )
            
        except Exception as e:
            logger.error(f"[{self.agent_id}] Error working on task: {e}")

            if self.self_improvement and self.config.enable_self_improvement:
                duration = max(0.0, asyncio.get_event_loop().time() - start_time)
                outcome_id = f"task_{task.id}_{int(datetime.now().timestamp() * 1000)}"
                self.self_improvement.record_outcome(
                    ActionOutcome(
                        id=outcome_id,
                        action_type="work_on_task",
                        outcome=OutcomeType.ERROR,
                        agent_id=self.agent_id,
                        task_id=task.id,
                        duration_seconds=duration,
                        error_message=str(e),
                    )
                )
            
            # Share the error as knowledge
            self.memory.store_knowledge(KnowledgeEntry(
                id="",
                type=KnowledgeType.TASK_LEARNING,
                content=f"Error on Task {task.id}: {str(e)}",
                importance=Importance.HIGH,
                source_agent=self.agent_id,
                source_task=task.id,
                tags=["error", "task_failure"],
            ))
        
        finally:
            self._current_task = None
            self.memory.update_agent_state(self.agent_id, current_task="")

    def _truncate(self, text: str) -> str:
        max_chars = max(256, int(self.agent_config.prompt_max_chars))
        if len(text) <= max_chars:
            return text
        return text[:max_chars - 3] + "..."
    
    async def _send_message(self, content: str):
        """Send a message to Copilot using SDK best practices."""
        async def _log_waiting():
            elapsed = 0
            interval = 15
            while True:
                await asyncio.sleep(interval)
                elapsed += interval
                logger.info(f"[{self.agent_id}] Waiting for response... {elapsed}s elapsed")

        progress_task = None

        try:
            # Reset response buffer
            self._response_content = ""

            # Start progress logger
            progress_task = asyncio.create_task(_log_waiting())
            
            # Use send_and_wait() which combines send() with waiting for idle
            # Events are still delivered to on() handlers while waiting
            timeout = self.agent_config.request_timeout
            inactivity_timeout = self.agent_config.turn_timeout_seconds
            start = asyncio.get_event_loop().time()

            send_task = asyncio.create_task(
                self._session.send_and_wait({"prompt": content}, timeout=timeout)
            )
            self._inflight_send_task = send_task

            if inactivity_timeout:
                inactivity_timeout = int(inactivity_timeout)
                while True:
                    done, _ = await asyncio.wait({send_task}, timeout=inactivity_timeout)
                    if send_task in done:
                        response = await send_task
                        break
                    if (time.monotonic() - self._last_event_time) >= inactivity_timeout:
                        send_task.cancel()
                        try:
                            await send_task
                        except asyncio.CancelledError:
                            pass
                        raise asyncio.TimeoutError()
            else:
                response = await send_task

            if self.perf_logger.enabled:
                self.perf_logger.log(
                    "sdk.send_and_wait",
                    duration_seconds=max(0.0, asyncio.get_event_loop().time() - start),
                    timeout_seconds=timeout,
                )
            
            print()  # Newline after response
            
            return response
            
        except asyncio.TimeoutError:
            if self.perf_logger.enabled:
                self.perf_logger.log(
                    "sdk.send_timeout",
                    duration_seconds=max(0.0, asyncio.get_event_loop().time() - start),
                    timeout_seconds=timeout,
                )
            logger.error(f"[{self.agent_id}] Request timed out after {timeout // 60} minutes")
            self._running = False
            self._shutting_down = True
            self.memory.update_agent_state(self.agent_id, status="stopped", current_task="")
            await self._recreate_session()
        except Exception as e:
            logger.error(f"[{self.agent_id}] Error: {e}")
        finally:
            if progress_task:
                progress_task.cancel()
            if self._inflight_send_task is send_task:
                self._inflight_send_task = None

    async def _check_and_start_implementation(self) -> None:
        response_lower = self._response_content.lower()
        plan_indicators = [
            'say "start"',
            "say 'start'",
            "plan created",
            "proceed with implementation",
            "confirm to proceed",
            "please confirm",
            "tell me to start",
        ]
        if any(indicator in response_lower for indicator in plan_indicators):
            logger.info(f"[{self.agent_id}] Plan detected, auto-starting implementation...")
            await self._send_message("start")

    def _maybe_mark_task_complete(self, task: Task) -> None:
        response_lower = self._response_content.lower()
        completion_signals = [
            "no code changes needed",
            "already implemented",
            "already present",
            "nothing to change",
            "tests already passing",
            "all tests passed",
            "requirements are present",
        ]
        if not any(signal in response_lower for signal in completion_signals):
            return

        try:
            result = self.tool_handler.handle_tool_call(
                "mark_task_complete",
                {"task_id": task.id},
            )
            logger.info(f"[{self.agent_id}] Marked Task {task.id} complete due to completion signals")
            if self.perf_logger.enabled:
                self.perf_logger.log("task.auto_mark_complete", task_id=task.id, result=result)
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning(f"[{self.agent_id}] Failed to auto-mark Task {task.id} complete: {exc}")

    async def _recreate_session(self) -> None:
        if not self._client or not self._session_config:
            return
        try:
            if self._session:
                await asyncio.wait_for(self._session.destroy(), timeout=5.0)
        except Exception:
            pass
        try:
            self._session = await self._client.create_session(self._session_config)
            self._session.on(self._handle_event)
        except Exception as exc:
            logger.warning(f"[{self.agent_id}] Failed to recreate session: {exc}")
    
    def _handle_event(self, event):
        """Handle events from Copilot session.
        
        SDK events are SessionEvent objects with:
        - event.type: SessionEventType enum
        - event.data: Event-specific data object
        """
        # Get event type - can be enum or string
        event_type = getattr(event, "type", None)
        if hasattr(event_type, "value"):
            event_type_str = event_type.value
        else:
            event_type_str = str(event_type) if event_type else ""
        
        event_data = getattr(event, "data", None)
        self._last_event_time = time.monotonic()

        def _extract_message_text(data):
            if not data:
                return ""
            return (
                getattr(data, "content", "") or
                getattr(data, "delta_content", "") or
                getattr(data, "partial_output", "") or
                getattr(data, "message", "") or
                getattr(data, "summary_content", "") or
                ""
            )
        
        if event_type_str == "assistant.message":
            # Handle assistant message (streaming-safe)
            content = _extract_message_text(event_data)
            if content:
                print(f"[{self.agent_id}] {content}", end="", flush=True)
                self._response_content += content
                max_chars = self.agent_config.max_response_chars
                if len(self._response_content) > max_chars:
                    self._response_content = self._response_content[-max_chars:]
        elif event_type_str == "assistant.turn_start":
            logger.info(f"[{self.agent_id}] Assistant turn started")
            now = asyncio.get_event_loop().time()
            if self._last_turn_end_time is not None and self.perf_logger.enabled:
                self.perf_logger.log(
                    "sdk.turn_gap",
                    gap_seconds=max(0.0, now - self._last_turn_end_time),
                )
            self._last_turn_start_time = now
        elif event_type_str == "assistant.turn_end":
            logger.info(f"[{self.agent_id}] Assistant turn ended")
            now = asyncio.get_event_loop().time()
            if self._last_turn_start_time is not None and self.perf_logger.enabled:
                self.perf_logger.log(
                    "sdk.turn_duration",
                    duration_seconds=max(0.0, now - self._last_turn_start_time),
                )
            self._last_turn_end_time = now
        
        elif event_type_str == "tool.execution_start":
            tool_name = (
                getattr(event_data, "toolName", None) or
                getattr(event_data, "tool_name", None) or
                getattr(event_data, "name", None) or
                ""
            ) if event_data else ""
            tool_call_id = getattr(event_data, "toolCallId", "") if event_data else ""
            if tool_call_id:
                self._tool_start_times[tool_call_id] = asyncio.get_event_loop().time()
                if tool_name:
                    self._tool_call_names[tool_call_id] = tool_name
            logger.info(f"[{self.agent_id}]   → Running: {tool_name}")
        
        elif event_type_str == "tool.execution_complete":
            tool_call_id = getattr(event_data, "toolCallId", "") if event_data else ""
            tool_name = (
                getattr(event_data, "toolName", None) or
                getattr(event_data, "tool_name", None) or
                getattr(event_data, "name", None) or
                ""
            ) if event_data else ""
            if not tool_name and tool_call_id:
                tool_name = self._tool_call_names.pop(tool_call_id, "")
            if tool_call_id in self._tool_start_times and self.perf_logger.enabled:
                start = self._tool_start_times.pop(tool_call_id)
                self.perf_logger.log(
                    "sdk.tool_duration",
                    tool=tool_name or tool_call_id,
                    duration_seconds=max(0.0, asyncio.get_event_loop().time() - start),
                )
            logger.info(f"[{self.agent_id}]   ✓ Completed: {tool_name or tool_call_id}")
        
        elif event_type_str == "session.error":
            message = getattr(event_data, "message", "Unknown error") if event_data else "Unknown error"
            logger.error(f"[{self.agent_id}] Session error: {message}")
        elif event_type_str == "session.idle":
            logger.info(f"[{self.agent_id}] Session idle")
    
    async def _handle_tool_call(self, name: str, params: dict) -> str:
        """Handle tool calls including hive-specific tools."""
        # Hive tools
        if name == "share_knowledge":
            return self._share_knowledge(params)
        elif name == "query_knowledge":
            return self._query_knowledge(params)
        elif name == "message_agents":
            return self._message_agents(params)
        elif name == "get_hive_status":
            return self._get_hive_status(params)
        elif name == "acquire_file":
            return self._acquire_file(params)
        elif name == "release_file":
            return self._release_file(params)
        elif name == "mark_knowledge_useful":
            return self._mark_useful(params)
        
        # Standard tools
        return await asyncio.to_thread(self.tool_handler.handle_tool_call, name, params)
    
    def _share_knowledge(self, params: dict) -> str:
        """Share knowledge with the hive."""
        try:
            k_type = KnowledgeType(params["knowledge_type"])
        except ValueError:
            k_type = KnowledgeType.AGENT_INSIGHT
        
        try:
            importance = Importance[params.get("importance", "medium").upper()]
        except KeyError:
            importance = Importance.MEDIUM
        
        entry = KnowledgeEntry(
            id="",
            type=k_type,
            content=params["content"],
            importance=importance,
            source_agent=self.agent_id,
            source_file=params.get("related_file", ""),
            source_task=self._current_task.id if self._current_task else "",
            tags=params.get("tags", []),
        )
        
        kid = self.memory.store_knowledge(entry)
        
        # Update contribution count
        state = self.memory.get_agent_state(self.agent_id)
        if state:
            self.memory.update_agent_state(
                self.agent_id,
                knowledge_contributed=state.knowledge_contributed + 1
            )
        
        return json.dumps({"status": "shared", "knowledge_id": kid})
    
    def _query_knowledge(self, params: dict) -> str:
        """Query the knowledge base."""
        k_type = None
        if "knowledge_type" in params:
            try:
                k_type = KnowledgeType(params["knowledge_type"])
            except ValueError:
                pass
        
        results = self.memory.search_knowledge(
            query=params.get("query"),
            knowledge_type=k_type,
            source_file=params.get("file_path"),
            limit=20,
        )
        
        return json.dumps({
            "count": len(results),
            "results": [
                {
                    "id": r.id,
                    "type": r.type.value,
                    "content": r.content,
                    "importance": r.importance.name,
                    "source_agent": r.source_agent,
                    "usefulness": r.usefulness_score,
                }
                for r in results
            ]
        })
    
    def _message_agents(self, params: dict) -> str:
        """Send a message to other agents."""
        self.memory.broadcast(
            self.agent_id,
            params["message_type"],
            params["message"]
        )
        return json.dumps({"status": "sent"})
    
    def _get_hive_status(self, params: dict) -> str:
        """Get hive status."""
        agents = [a for a in self.memory.get_active_agents() if a.status != "stopped"]
        summary = self.memory.get_summary()
        
        return json.dumps({
            "active_agents": len(agents),
            "total_knowledge": summary["total_knowledge"],
            "agents": [
                {
                    "id": a.agent_id,
                    "status": a.status,
                    "current_task": a.current_task,
                    "tasks_completed": a.tasks_completed,
                    "knowledge_contributed": a.knowledge_contributed,
                }
                for a in agents
            ],
            "locked_files": self.memory.get_locked_files(),
        })
    
    def _acquire_file(self, params: dict) -> str:
        """Acquire a file lock."""
        success = self.memory.acquire_file_lock(params["file_path"], self.agent_id)
        if success:
            self.memory.update_agent_state(self.agent_id, current_file=params["file_path"])
            return json.dumps({"status": "acquired", "file": params["file_path"]})
        else:
            locks = self.memory.get_locked_files()
            holder = locks.get(params["file_path"], "unknown")
            return json.dumps({"status": "failed", "held_by": holder})
    
    def _release_file(self, params: dict) -> str:
        """Release a file lock."""
        self.memory.release_file_lock(params["file_path"], self.agent_id)
        return json.dumps({"status": "released", "file": params["file_path"]})
    
    def _mark_useful(self, params: dict) -> str:
        """Mark knowledge as useful."""
        self.memory.update_usefulness(params["knowledge_id"], 1.0)
        return json.dumps({"status": "marked_useful"})
    
    async def _process_messages(self):
        """Process messages from other agents."""
        messages = self.memory.get_messages(self.agent_id)
        
        for msg in messages:
            logger.debug(
                f"[{self.agent_id}] Message from {msg['from_agent']}: "
                f"{msg['message_type']} - {msg['content'][:50]}..."
            )
    
    async def _cleanup(self):
        """Clean up resources using SDK best practices."""
        logger.info(f"[{self.agent_id}] Cleaning up worker resources...")
        if self._perf_task:
            self._perf_task.cancel()
        if self._inflight_send_task and not self._inflight_send_task.done():
            self._inflight_send_task.cancel()
            try:
                await asyncio.wait_for(self._inflight_send_task, timeout=2.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass
        # Release any held file locks
        locks = self.memory.get_locked_files()
        for file_path, holder in locks.items():
            if holder == self.agent_id:
                self.memory.release_file_lock(file_path, self.agent_id)
        
        # Destroy session (not close - SDK best practice)
        if self._session:
            try:
                await asyncio.wait_for(self._session.destroy(), timeout=5.0)
                logger.debug(f"[{self.agent_id}] Session destroyed")
            except asyncio.TimeoutError:
                logger.warning(f"[{self.agent_id}] Session destroy timed out")
            except OSError as e:
                if getattr(e, "errno", None) == 32:
                    logger.debug(f"[{self.agent_id}] Session destroy broken pipe during shutdown")
                else:
                    logger.warning(f"[{self.agent_id}] Error destroying session: {e}")
            except Exception as e:
                logger.warning(f"[{self.agent_id}] Error destroying session: {e}")
        
        # Stop client and get cleanup errors (SDK best practice)
        if self._client:
            try:
                errors = await asyncio.wait_for(self._client.stop(), timeout=5.0)
                if errors:
                    for error in errors:
                        if "Broken pipe" in getattr(error, "message", ""):
                            logger.debug(f"[{self.agent_id}] Cleanup warning suppressed: {error.message}")
                        else:
                            logger.warning(f"[{self.agent_id}] Cleanup error: {error.message}")
                logger.debug(f"[{self.agent_id}] Client stopped")
            except asyncio.TimeoutError:
                logger.warning(f"[{self.agent_id}] Client stop timed out")
            except Exception as e:
                logger.warning(f"[{self.agent_id}] Error stopping client: {e}")

    def _start_perf_sampler(self) -> None:
        if not self.perf_logger.enabled or self._perf_task is not None:
            return

        interval = max(1, int(self.config.perf_sample_interval_seconds))

        async def _sample_loop() -> None:
            next_time = asyncio.get_event_loop().time() + interval
            while True:
                await asyncio.sleep(interval)
                now = asyncio.get_event_loop().time()
                lag = max(0.0, now - next_time)
                next_time = now + interval
                if self.config.enable_tracemalloc:
                    current, peak = tracemalloc.get_traced_memory()
                    self.perf_logger.log(
                        "memory.sample",
                        current_bytes=current,
                        peak_bytes=peak,
                        loop_lag_seconds=lag,
                    )
                else:
                    self.perf_logger.log(
                        "loop.sample",
                        loop_lag_seconds=lag,
                    )

        self._perf_task = asyncio.create_task(_sample_loop())


class MultiAgentOrchestrator:
    """
    Orchestrates multiple agents working together.
    
    Responsibilities:
    - Spawn and manage agent workers
    - Coordinate task assignment
    - Monitor progress
    - Handle failures and restarts
    - Maintain the shared knowledge base
    """
    
    def __init__(self, config: OrchestratorConfig):
        self.config = config
        self.memory = RepositoryMemory(config.repo_root)
        self.self_improvement = (
            SelfImprovementEngine(config.repo_root)
            if config.enable_self_improvement
            else None
        )
        self.task_manager = TaskManager(config.repo_root / config.tasks_file)
        self.workers: list[AgentWorker] = []
        self._running = False
        self._tasks: list[asyncio.Task] = []
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Configure logging."""
        log_format = "%(asctime)s [%(levelname)s] %(message)s"
        handlers = [logging.StreamHandler(sys.stdout)]
        
        if self.config.log_file:
            handlers.append(logging.FileHandler(self.config.log_file))
        
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format=log_format,
            handlers=handlers
        )
    
    async def start(self):
        """Start the orchestrator and all agents."""
        self._running = True
        
        # Clear stale agents from previous runs
        stale = self.memory.clear_stale_agents(timeout_seconds=300)
        if stale:
            logger.debug(f"Cleared {len(stale)} stale agents from previous runs")
        
        logger.info("=" * 70)
        logger.info("RaceNet Multi-Agent Orchestrator Starting")
        logger.info(f"Number of agents: {self.config.num_agents}")
        logger.info(f"Model: {self.config.model}")
        logger.info(f"Repository: {self.config.repo_root}")
        logger.info("=" * 70)

        # Clean up stale agents and locks from previous runs
        try:
            cleanup = self.memory.cleanup_stale_state(timeout_seconds=120)
            if cleanup["stale_agents_removed"] or cleanup["orphaned_locks_removed"]:
                logger.info(
                    "Cleanup on start: removed stale agents %s, orphaned locks %s",
                    cleanup["stale_agents_removed"],
                    cleanup["orphaned_locks_removed"],
                )
            else:
                logger.info("Cleanup on start: no stale agents or orphaned locks found")
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning(f"Startup cleanup failed: {exc}")
        
        # Setup signal handlers
        for sig in (signal.SIGINT, signal.SIGTERM):
            asyncio.get_event_loop().add_signal_handler(
                sig, lambda: asyncio.create_task(self.stop())
            )
        
        try:
            # Spawn workers
            await self._spawn_workers()
            
            # Run monitoring loop
            await self._monitor_loop()
            
        except asyncio.CancelledError:
            logger.info("Orchestrator cancelled")
        except Exception as e:
            logger.exception(f"Orchestrator error: {e}")
        finally:
            await self._cleanup()
    
    async def stop(self):
        """Stop all agents and the orchestrator."""
        logger.info("Stopping orchestrator...")
        self._running = False
        
        # Stop all workers
        for worker in self.workers:
            await worker.stop()
        
        # Cancel all tasks
        for task in self._tasks:
            task.cancel()
    
    async def _spawn_workers(self):
        """Spawn all agent workers."""
        roles = self.config.agent_roles.copy()
        
        # Fill in with generalists if not enough roles specified
        while len(roles) < self.config.num_agents:
            roles.append(AgentRole.GENERALIST)
        
        for i in range(self.config.num_agents):
            agent_id = f"agent-{i+1:02d}"
            role = roles[i] if i < len(roles) else AgentRole.GENERALIST
            
            worker = AgentWorker(
                agent_id=agent_id,
                config=self.config,
                memory=self.memory,
                task_manager=self.task_manager,
                role=role,
                self_improvement=self.self_improvement,
            )
            
            self.workers.append(worker)
            
            # Start worker as async task
            task = asyncio.create_task(worker.start())
            self._tasks.append(task)
            
            logger.info(f"Spawned {agent_id} with role {role.value}")
            
            # Stagger starts slightly to avoid race conditions
            await asyncio.sleep(0.5)
    
    async def _monitor_loop(self):
        """Monitor agents and coordinate work."""
        last_summary_time = 0
        
        while self._running:
            await asyncio.sleep(self.config.sync_interval_seconds)
            
            # Print status every minute
            now = asyncio.get_event_loop().time()
            if now - last_summary_time > 60:
                self._print_status()
                last_summary_time = now
            
            # Check for stalled agents
            await self._check_agent_health()

            # Stop if all agents are stopped/idle (nothing running)
            agents_all = self.memory.get_all_agents()
            active = [
                a for a in self.memory.get_active_agents(timeout_seconds=120)
                if a.status != "stopped"
            ]
            if not active and agents_all:
                if all(a.status in ("stopped", "idle") for a in agents_all):
                    logger.info("All agents stopped or idle; stopping orchestrator.")
                    break
            
            # Reload tasks
            self.task_manager.reload()
            
            # Check if all tasks are done
            summary = self.task_manager.get_summary()
            if summary["not_started"] == 0 and summary["in_progress"] == 0:
                if self.config.auto_propose_tasks:
                    logger.info("All tasks complete! Asking agents to propose new tasks...")
                else:
                    logger.info("All tasks complete!")
                    break
    
    def _print_status(self):
        """Print current status."""
        agents = [a for a in self.memory.get_active_agents() if a.status != "stopped"]
        task_summary = self.task_manager.get_summary()
        knowledge_summary = self.memory.get_summary()
        
        logger.info("-" * 50)
        logger.info("HIVE STATUS")
        logger.info(f"  Active agents: {len(agents)}/{self.config.num_agents}")
        logger.info(f"  Tasks: {task_summary['completed']}/{task_summary['total']} completed")
        logger.info(f"  Shared knowledge: {knowledge_summary['total_knowledge']} entries")
        
        for agent in agents:
            status = f"    {agent.agent_id}: {agent.status}"
            if agent.current_task:
                status += f" (Task {agent.current_task})"
            status += f" | {agent.tasks_completed} tasks, {agent.knowledge_contributed} knowledge"
            logger.info(status)
        
        logger.info("-" * 50)
    
    async def _check_agent_health(self):
        """Check if any agents are stalled and restart them."""
        agents = self.memory.get_all_agents()
        active = self.memory.get_active_agents(timeout_seconds=120)
        
        active_ids = {a.agent_id for a in active}
        
        for agent in agents:
            if agent.agent_id not in active_ids and agent.status not in ("stopped", "idle"):
                logger.warning(f"Agent {agent.agent_id} appears stalled, marking as stopped")
                self.memory.update_agent_state(agent.agent_id, status="stopped")
    
    async def _cleanup(self):
        """Clean up resources."""
        logger.info("Cleaning up orchestrator...")
        
        # Wait for all worker tasks to complete
        if self._tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self._tasks, return_exceptions=True),
                    timeout=10.0,
                )
            except asyncio.TimeoutError:
                logger.warning("Timeout waiting for worker tasks to finish; cancelling")
                for task in self._tasks:
                    task.cancel()
                await asyncio.gather(*self._tasks, return_exceptions=True)
        
        # Print final summary
        self._print_status()
        
        logger.info("=" * 70)
        logger.info("Orchestrator stopped")
        logger.info("=" * 70)
