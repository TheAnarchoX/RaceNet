"""
Autonomous Agent

Main orchestration for the autonomous RaceNet development agent.
Uses the GitHub Copilot SDK to work on tasks and evolve the project.
"""

import asyncio
import logging
import re
import sys
import time
from typing import Any, Callable, Optional

from agent.config import AgentConfig
from agent.task_manager import TaskManager
from agent.tools import ToolHandler, create_tool_definitions

# Try to import the Copilot SDK
try:
    from copilot import CopilotClient
    HAS_COPILOT_SDK = True
except ImportError:
    HAS_COPILOT_SDK = False


logger = logging.getLogger(__name__)


SYSTEM_MESSAGE = """You are an autonomous development agent for RaceNet, a GT3-style racing simulation framework for machine learning.

Your mission is to continuously improve and evolve the RaceNet project by:
1. Working on tasks from TASKS.md
2. Writing high-quality, well-tested code
3. Following the project's coding conventions
4. Proposing new tasks when you identify improvements
5. Keeping the simulation realistic and ML-ready

## Project Context
RaceNet simulates GT3 racing cars with:
- Realistic physics (engine, tires, aero, suspension)
- Procedural track generation
- Telemetry system for data export
- ML environment (Gymnasium-compatible)
- Scoring system for lap times and driving style

## Workflow
1. First, get the next available task using get_next_task
2. Read relevant files to understand the current implementation
3. Make incremental changes, testing frequently
4. When done, commit your changes
5. If you discover improvements, propose new tasks
6. Move on to the next task

## Coding Guidelines
- Use Python type hints for all functions
- Follow PEP 8 with 88-char line limit (Black formatting)
- Use dataclasses for config and state objects
- Use SI units (meters, seconds, kilograms, Newtons)
- Store angles in radians internally
- Add docstrings to all public functions
- Run tests after making changes

## GT3 Reference Data
- Mass: ~1300 kg (with driver)
- Power: ~500-550 hp
- Max cornering: ~1.5-1.6g
- Peak tire slip angle: ~8-10 degrees
- Peak slip ratio: ~8-10%
- Optimal tire temp: ~85-100°C

Be thorough, methodical, and write production-quality code.
"""


PLANNER_SYSTEM_MESSAGE = """You are a strategic planner for RaceNet, a GT3-style racing simulation framework for ML experimentation.

Your mission is to DEEPLY ANALYZE the codebase and create, refine, and prioritize tasks in TASKS.md.

## CRITICAL: Deep Code Analysis Required

You MUST read and analyze the ACTUAL SOURCE CODE - not just documentation. For every area you evaluate:
1. Read the implementation files directly (not just docstrings)
2. Look at how components connect and interact
3. Identify actual bugs, missing features, or code smells
4. Find hardcoded values that should be configurable
5. Spot duplicated logic or abstraction opportunities
6. Check test coverage gaps by reading test files

## Task Categories to Consider

**Feature Development**:
- Missing physics models (aero, suspension, drivetrain)
- Track feature improvements
- ML environment enhancements
- Telemetry/visualization

**Tech Debt**:
- Code that doesn't follow project patterns
- Missing type hints or docstrings
- Hardcoded magic numbers
- Poor separation of concerns
- Missing error handling

**DevOps & Infrastructure**:
- CI/CD improvements
- Test infrastructure
- Development tooling
- Build/packaging

**Repository Hygiene**:
- Documentation gaps
- Outdated comments
- Dead code
- Inconsistent naming
- Missing examples

## Workflow

1. **READ TASKS.md** first to understand current state
2. **DEEP DIVE into src/** - read actual implementation files:
   - `src/racenet/car/` - all vehicle components
   - `src/racenet/track/` - track generation
   - `src/racenet/simulation/` - physics and world
   - `src/racenet/ml/` - RL environment
   - `src/racenet/telemetry/` - data recording
   - `src/racenet/scoring/` - reward system
3. **READ tests/** to understand coverage gaps
4. **Analyze** what's actually implemented vs. what's stubbed/incomplete
5. **Create or refine tasks** with specific, actionable requirements

## Task Quality Standards

Each task you create/refine must have:
- Clear, specific title
- Concrete requirements (not vague goals)
- Acceptance criteria that can be verified
- Accurate file references
- Realistic difficulty assessment
- Proper dependency chains

## GT3 Reference (for physics tasks)
- Mass: ~1300 kg | Power: ~500-550 hp
- Cornering: ~1.5-1.6g | Braking: ~1.8-2.0g
- Peak tire slip: 8-10% long, 8° lateral
- Optimal tire temp: 85-100°C

Remember: Your value comes from READING THE CODE and finding real issues, not making generic suggestions.
"""


class AutonomousAgent:
    """Autonomous development agent for RaceNet."""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.task_manager = TaskManager(config.tasks_path)
        self.tool_handler = ToolHandler(config, self.task_manager)
        self.client = None
        self.session = None
        self._iteration = 0
        self._tasks_completed = 0
        self._shutting_down = False
        self._response_content = ""
        self._last_event_time = time.monotonic()
        
        # Setup logging
        self._setup_logging()
        
        self._main_task = None
    
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
        """Start the autonomous agent."""
        logger.info("=" * 60)
        logger.info("RaceNet Autonomous Agent Starting")
        logger.info(f"Model: {self.config.model}")
        logger.info(f"Repository: {self.config.repo_root}")
        logger.info(f"Dry run: {self.config.dry_run}")
        logger.info("=" * 60)
        
        # Determine whether to use mock client
        use_mock = self.config.dry_run or not HAS_COPILOT_SDK
        if use_mock:
            if self.config.dry_run:
                logger.info("Using mock client for dry-run mode")
            else:
                logger.warning("Copilot SDK not installed - using mock client")
        
        try:
            # Create a task so signal handler can cancel it
            self._main_task = asyncio.current_task()
            await self._run()
        except asyncio.CancelledError:
            logger.info("Agent cancelled")
        except KeyboardInterrupt:
            logger.info("Agent interrupted by user")
        except FileNotFoundError:
            logger.error("Copilot CLI not found. Please install it first.")
        except ConnectionError:
            logger.error("Could not connect to Copilot CLI server.")
        except Exception as e:
            logger.exception(f"Agent failed with error: {e}")
        finally:
            await self._cleanup()
    
    async def _run(self):
        """Main agent loop."""
        # Determine whether to use mock client
        use_mock = self.config.dry_run or not HAS_COPILOT_SDK
        
        if use_mock:
            # Use mock client for dry-run or when SDK not available
            self.client = MockCopilotClient()
            self.client.tool_handler = self.tool_handler
        else:
            # Use real Copilot client
            client_options = {}
            if self.config.cli_url:
                client_options["cli_url"] = self.config.cli_url
            self.client = CopilotClient(**client_options)
        
        # Start the client explicitly (best practice from SDK cookbook)
        await self.client.start()
        logger.info("Copilot client started")
        
        # Create session with tools
        tools = create_tool_definitions(self.config, self.tool_handler)
        
        # Use planner or task worker system message
        system_msg = PLANNER_SYSTEM_MESSAGE if self.config.planner_mode else SYSTEM_MESSAGE

        self.session = await self.client.create_session(
            {
                "system_message": {"content": system_msg},
                "model": self.config.model,
                "tools": tools,
            }
        )
        
        # Register event handler using session.on() pattern
        self.session.on(self._handle_event)

        
        logger.info(f"Copilot session created: {self.session.session_id}")
        
        # Get initial task summary
        summary = self.task_manager.get_summary()
        logger.info(f"Task Summary: {summary['completed']}/{summary['total']} completed ({summary['completion_percentage']:.1f}%)")
        
        # Run planner mode or task worker mode
        if self.config.planner_mode:
            await self._run_planner_mode()
        else:
            await self._run_task_worker_mode()
    
    async def _run_planner_mode(self):
        """Run in dedicated planning mode - analyze codebase and manage tasks."""
        logger.info("\n🗺️  Running in PLANNER MODE")
        logger.info("Deep diving into codebase to analyze and manage tasks...\n")
        
        # Single comprehensive planning prompt
        prompt = """Please analyze the RaceNet codebase and manage the task list.

## Your Mission

1. **Read TASKS.md** to understand current task state
2. **Deep dive into the source code** - READ THE ACTUAL IMPLEMENTATION FILES:
   - `src/racenet/car/*.py` - all vehicle component implementations
   - `src/racenet/track/*.py` - track generation code
   - `src/racenet/simulation/*.py` - physics engine
   - `src/racenet/ml/*.py` - RL environment
   - `src/racenet/telemetry/*.py` - data recording
   - `src/racenet/scoring/*.py` - reward system
3. **Read tests/** to find coverage gaps
4. **Identify issues** - bugs, incomplete features, tech debt, missing tests
5. **Update TASKS.md** with new or refined tasks

## What to Look For

- **Incomplete implementations**: Stubs, TODOs, hardcoded values
- **Missing features**: Compare to GT3 specs in copilot-instructions.md
- **Tech debt**: Code that doesn't follow patterns, missing types/docs
- **Test gaps**: Components without adequate test coverage
- **DevOps needs**: CI/CD, tooling, build improvements
- **Documentation**: Outdated or missing docs

## Output

After your analysis, update TASKS.md by:
- Adding new specific, actionable tasks
- Refining existing task descriptions with more detail
- Adjusting priorities based on dependencies and impact
- Marking any completed tasks

DO NOT just read documentation - READ THE CODE and understand what's actually implemented.
Start by reading TASKS.md, then systematically explore the source code."""

        await self._send_message(prompt)
        await self._check_and_start_implementation()
        
        logger.info("\n🗺️  Planning session complete!")
    
    async def _run_task_worker_mode(self):
        """Run normal task worker mode - pick up and complete tasks."""
        # Main loop
        while self._iteration < self.config.max_iterations and not self._shutting_down:
            self._iteration += 1
            logger.info(f"\n--- Iteration {self._iteration} ---")
            
            # Check if we should stop
            if self._tasks_completed >= self.config.max_tasks_per_session:
                logger.info(f"Completed {self._tasks_completed} tasks, stopping for this session")
                break
            
            # Get next task and work on it
            await self._work_on_next_task()
    
    async def _work_on_next_task(self):
        """Work on the next available task."""
        # Reload tasks
        self.task_manager.reload()
        
        # Get next task
        task = self.task_manager.get_next_task()
        
        if not task:
            logger.info("No more tasks available!")
            
            if self.config.auto_propose_tasks:
                logger.info("Asking agent to propose new tasks...")
                await self._request_new_tasks()
            return
        
        logger.info(f"Working on Task {task.id}: {task.title}")
        logger.info(f"Priority: {task.priority.name}, Difficulty: {task.difficulty.value}")
        
        # Create prompt for the task
        prompt = f"""Please work on Task {task.id}: {task.title}

Description: {task.description}

Requirements:
{chr(10).join(f'- {r}' for r in task.requirements)}

Acceptance Criteria:
{chr(10).join(f'- [{" x" if c else " "}] {t}' for c, t in task.acceptance_criteria)}

Files to modify: {', '.join(task.files_to_modify)}

Current state: {task.current_state}

IMPORTANT: This is an autonomous agent session. Do NOT wait for manual confirmation.
After creating any plan, immediately proceed with implementation.
Do NOT ask me to say "start" - just implement directly.

Please:
1. First read the relevant files to understand the current implementation
2. Make the necessary changes to fulfill the requirements
3. Run tests to verify your changes work
4. Commit your changes when done
5. CRITICAL: When you've completed all requirements, use the mark_task_complete tool with task_id="{task.id}" to mark it done in TASKS.md

Start by exploring the codebase and then implement the changes."""

        # Send to Copilot and handle response
        await self._send_message(prompt)
        
        # Check if Copilot created a plan and is waiting for "start"
        # If so, automatically proceed with implementation
        await self._check_and_start_implementation()
    
    async def _check_and_start_implementation(self):
        """Check if Copilot created a plan and auto-start implementation."""
        response_lower = self._response_content.lower()
        
        # Detect plan creation patterns
        plan_indicators = [
            'say "start"',
            "say 'start'",
            "plan created",
            "proceed with implementation",
            "confirm to proceed",
            "tell me to start",
        ]
        
        if any(indicator in response_lower for indicator in plan_indicators):
            logger.info("Plan detected, automatically starting implementation...")
            await self._send_message("start")
    
    async def _request_new_tasks(self):
        """Ask the agent to propose new tasks."""
        prompt = """Based on the current state of the RaceNet project, please propose 2-3 new tasks that would improve the simulation, ML capabilities, or code quality.

Use the get_task_summary tool to see current progress, then propose new tasks using the propose_new_task tool.

Focus on:
1. Improvements you noticed while working on previous tasks
2. Features that would make the simulation more realistic
3. Enhancements for ML training
4. Code quality and testing improvements

For each task, provide clear requirements and acceptance criteria."""

        await self._send_message(prompt)
    
    async def _send_message(self, content: str):
        """Send a message and wait for response using SDK best practices."""
        logger.info("Sending message to Copilot...")

        async def _log_waiting():
            elapsed = 0
            interval = 15
            while True:
                await asyncio.sleep(interval)
                elapsed += interval
                since_last = time.monotonic() - self._last_event_time
                logger.info(
                    "Waiting for response... %ss elapsed (last event %ss ago)",
                    elapsed,
                    int(since_last),
                )

        progress_task = None

        try:
            # Reset response buffer
            self._response_content = ""
            self._last_event_time = time.monotonic()

            # Start progress logger
            progress_task = asyncio.create_task(_log_waiting())

            # Use send_and_wait() which combines send() with waiting for idle
            # Events are still delivered to on() handlers while waiting
            timeout = self.config.request_timeout
            response = await self.session.send_and_wait(
                {"prompt": content},
                timeout=timeout
            )

            print()  # New line after response
            logger.info("Response complete")

            return response

        except asyncio.TimeoutError:
            logger.error(f"Request timed out after {timeout // 60} minutes")
        except Exception as e:
            logger.error(f"Error sending message: {e}")
        finally:
            if progress_task:
                progress_task.cancel()
    
    def _handle_event(self, event: Any):
        """Handle events from Copilot session.
        
        SDK events are SessionEvent objects with:
        - event.type: SessionEventType enum (e.g., SessionEventType.ASSISTANT_MESSAGE)
        - event.data: Event-specific data object
        """
        # Get event type - can be enum or string
        event_type = getattr(event, "type", None)
        if hasattr(event_type, "value"):
            # It's an enum, get string value
            event_type_str = event_type.value
        else:
            event_type_str = str(event_type) if event_type else ""
        
        event_data = getattr(event, "data", None)
        self._last_event_time = time.monotonic()

        def _extract_message_text(data: Any) -> str:
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
                print(content, end="", flush=True)
                self._response_content += content
        elif event_type_str == "assistant.turn_start":
            logger.info("Assistant turn started")
        elif event_type_str == "assistant.turn_end":
            logger.info("Assistant turn ended")
        
        elif event_type_str == "tool.execution_start":
            # Log tool execution start - try multiple attribute names
            tool_name = (
                getattr(event_data, "toolName", None) or
                getattr(event_data, "tool_name", None) or
                getattr(event_data, "name", None) or
                ""
            ) if event_data else ""
            logger.info(f"  → Running: {tool_name}")
        
        elif event_type_str == "tool.execution_complete":
            # Log tool execution complete
            tool_call_id = getattr(event_data, "toolCallId", "") if event_data else ""
            tool_name = (
                getattr(event_data, "toolName", None) or
                getattr(event_data, "tool_name", None) or
                getattr(event_data, "name", None) or
                ""
            ) if event_data else ""
            logger.info(f"  ✓ Completed: {tool_name or tool_call_id}")
        
        elif event_type_str == "session.error":
            # Handle session errors
            message = getattr(event_data, "message", "Unknown error") if event_data else "Unknown error"
            logger.error(f"Session error: {message}")
        
        elif event_type_str == "session.idle":
            # Session is idle (request complete)
            logger.info("Session idle")
    
    async def _cleanup(self):
        """Clean up resources using SDK best practices with timeout protection."""
        logger.info("Starting agent cleanup...")
        # Destroy session (not close - SDK best practice)
        if self.session:
            try:
                await asyncio.wait_for(self.session.destroy(), timeout=5.0)
                logger.debug("Session destroyed")
            except asyncio.TimeoutError:
                logger.warning("Session destroy timed out")
            except Exception as e:
                logger.warning(f"Error destroying session: {e}")
        
        # Stop client with timeout to prevent hanging
        if self.client:
            try:
                errors = await asyncio.wait_for(self.client.stop(), timeout=5.0)
                if errors:
                    for error in errors:
                        logger.warning(f"Cleanup error: {error.message}")
                logger.debug("Client stopped")
            except asyncio.TimeoutError:
                logger.warning("Client stop timed out, forcing exit")
            except Exception as e:
                logger.warning(f"Error stopping client: {e}")
        
        # Log summary
        logger.info("=" * 60)
        logger.info("Agent cleanup finished")
        logger.info("Agent Session Complete")
        logger.info(f"Iterations: {self._iteration}")
        logger.info(f"Tasks completed: {self._tasks_completed}")
        
        # Show proposed tasks
        proposed = self.tool_handler.get_proposed_tasks()
        if proposed:
            logger.info(f"New tasks proposed: {len(proposed)}")
            for task in proposed:
                logger.info(f"  - {task['id']}: {task['title']}")
        
        logger.info("=" * 60)


class MockEventData:
    """Mock event data for testing."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class MockSessionEvent:
    """Mock session event matching SDK SessionEvent interface."""
    def __init__(self, event_type: str, data: dict = None):
        self.type = event_type  # Store as string directly
        self.data = MockEventData(**(data or {}))


class MockCopilotSession:
    """Mock session for testing without the actual SDK."""
    
    def __init__(self, tool_handler: ToolHandler):
        self.tool_handler = tool_handler
        self.session_id = "mock-session-001"
        self.messages = []
        self._event_handlers = []
        self._iteration_count = 0
    
    def on(self, handler: Callable):
        """Register event handler (SDK pattern)."""
        self._event_handlers.append(handler)
        return lambda: self._event_handlers.remove(handler)
    
    async def send_and_wait(self, options: dict, timeout: float = 60):
        """Send a message and wait for idle (SDK pattern)."""
        prompt = options.get("prompt", "")
        self.messages.append({"role": "user", "content": prompt})
        self._iteration_count += 1
        
        # Add a small delay to simulate real API latency
        await asyncio.sleep(0.5)
        
        # Build a more informative mock response
        response_parts = []
        prompt_lower = prompt.lower()
        
        # Check if this is a task work prompt (handles both single agent and hive prompts)
        is_task_prompt = (
            ("Task" in prompt and "work on" in prompt_lower) or
            ("assigned to task" in prompt_lower)
        )
        
        if is_task_prompt:
            task_match = re.search(r'Task (\d+\.\d+)', prompt)
            task_id = task_match.group(1) if task_match else "unknown"
            
            response_parts.append(f"\n[Mock Mode] Working on Task {task_id}")
            response_parts.append("[Mock Mode] Simulating tool calls (SDK not installed):")
            
            # Simulate some tool calls
            if self.tool_handler:
                response_parts.append("  → get_task_summary")
                try:
                    self.tool_handler.handle_tool_call("get_task_summary", {})
                    response_parts.append("    Task progress loaded")
                except Exception:
                    pass
                
                response_parts.append("  → Simulating file reads and edits")
                response_parts.append("  → Task simulation complete")
            
            response_parts.append("[Mock Mode] Install github-copilot-sdk for real agent behavior.\n")
        
        elif "planner" in prompt_lower or "analyze" in prompt_lower:
            response_parts.append("\n[Mock Mode] Running planner analysis...")
            response_parts.append("[Mock Mode] Simulating codebase analysis:")
            
            # Simulate planner tools
            if self.tool_handler:
                response_parts.append("  → get_file_structure")
                response_parts.append("  → find_todos_and_fixmes")
                response_parts.append("  → get_test_coverage")
                response_parts.append("  → get_dependencies_graph")
            
            response_parts.append("[Mock Mode] Analysis complete. Would update TASKS.md.\n")
        
        else:
            response_parts.append(f"\n[Mock Mode] Received prompt ({len(prompt)} chars)")
            response_parts.append("[Mock Mode] SDK not installed - simulating response.\n")
        
        response_content = "\n".join(response_parts)
        
        # Dispatch mock events to handlers (using proper mock event objects)
        for handler in self._event_handlers:
            # Simulate assistant message
            handler(MockSessionEvent(
                "assistant.message",
                {"content": response_content}
            ))
            # Simulate session idle
            handler(MockSessionEvent("session.idle", {}))
        
        return None  # No final message in mock
    
    async def destroy(self):
        """Destroy the session (SDK pattern)."""
        pass


class MockCopilotClient:
    """Mock client for testing without the actual SDK."""
    
    def __init__(self, **kwargs):
        self.tool_handler = None
    
    async def start(self):
        """Start the client (SDK pattern)."""
        pass
    
    async def create_session(self, config: dict = None):
        """Create a mock session."""
        return MockCopilotSession(self.tool_handler)
    
    async def stop(self):
        """Stop the client (SDK pattern)."""
        return []  # No errors


# For environments without the SDK, provide a fallback
if not HAS_COPILOT_SDK:
    CopilotClient = MockCopilotClient
