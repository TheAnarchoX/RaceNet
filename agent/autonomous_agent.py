"""
Autonomous Agent

Main orchestration for the autonomous RaceNet development agent.
Uses the GitHub Copilot SDK to work on tasks and evolve the project.
"""

import asyncio
import json
import logging
import signal
import sys
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
        
        # Setup logging
        self._setup_logging()
        
        # Setup signal handlers for graceful shutdown
        self._setup_signal_handlers()
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(sig, frame):
            logger.info("\nShutdown signal received...")
            self._shutting_down = True
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
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
        
        if not HAS_COPILOT_SDK:
            logger.error("Copilot SDK not installed. Install with: pip install copilot-sdk")
            return
        
        try:
            await self._run()
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
        # Initialize Copilot client
        client_options = {}
        if self.config.cli_url:
            client_options["cli_url"] = self.config.cli_url
        
        self.client = CopilotClient(**client_options)
        
        # Start the client explicitly (best practice from SDK cookbook)
        await self.client.start()
        logger.info("Copilot client started")
        
        # Create session with tools
        tools = create_tool_definitions(self.config, self.tool_handler)

        self.session = await self.client.create_session(
            {
                "system_message": {"content": SYSTEM_MESSAGE},
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

Please:
1. First read the relevant files to understand the current implementation
2. Make the necessary changes to fulfill the requirements
3. Run tests to verify your changes work
4. Commit your changes when done

Start by exploring the codebase and then implement the changes."""

        # Send to Copilot and handle response
        await self._send_message(prompt)
    
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
        
        try:
            # Reset response buffer
            self._response_content = ""
            
            # Use send_and_wait() which combines send() with waiting for idle
            # Events are still delivered to on() handlers while waiting
            response = await self.session.send_and_wait(
                {"prompt": content},
                timeout=300  # 5 minute timeout
            )
            
            print()  # New line after response
            logger.info("Response complete")
            
            return response
            
        except asyncio.TimeoutError:
            logger.error("Request timed out after 5 minutes")
        except Exception as e:
            logger.error(f"Error sending message: {e}")
    
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
        
        if event_type_str == "assistant.message":
            # Handle assistant message
            content = getattr(event_data, "content", "") if event_data else ""
            if content:
                print(content, end="", flush=True)
                self._response_content += content
        
        elif event_type_str == "tool.execution_start":
            # Log tool execution start
            tool_name = getattr(event_data, "toolName", "") if event_data else ""
            logger.info(f"  → Running: {tool_name}")
        
        elif event_type_str == "tool.execution_complete":
            # Log tool execution complete
            tool_call_id = getattr(event_data, "toolCallId", "") if event_data else ""
            logger.debug(f"  ✓ Completed: {tool_call_id}")
        
        elif event_type_str == "session.error":
            # Handle session errors
            message = getattr(event_data, "message", "Unknown error") if event_data else "Unknown error"
            logger.error(f"Session error: {message}")
        
        elif event_type_str == "session.idle":
            # Session is idle (request complete)
            logger.debug("Session idle")
    
    async def _cleanup(self):
        """Clean up resources using SDK best practices."""
        # Destroy session (not close - SDK best practice)
        if self.session:
            try:
                await self.session.destroy()
                logger.debug("Session destroyed")
            except Exception as e:
                logger.warning(f"Error destroying session: {e}")
        
        # Stop client and get any cleanup errors (SDK best practice)
        if self.client:
            try:
                errors = await self.client.stop()
                if errors:
                    for error in errors:
                        logger.warning(f"Cleanup error: {error.message}")
                logger.debug("Client stopped")
            except Exception as e:
                logger.warning(f"Error stopping client: {e}")
        
        # Log summary
        logger.info("=" * 60)
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


class MockCopilotSession:
    """Mock session for testing without the actual SDK."""
    
    def __init__(self, tool_handler: ToolHandler):
        self.tool_handler = tool_handler
        self.session_id = "mock-session-001"
        self.messages = []
        self._event_handlers = []
    
    def on(self, handler: Callable):
        """Register event handler (SDK pattern)."""
        self._event_handlers.append(handler)
        return lambda: self._event_handlers.remove(handler)
    
    async def send_and_wait(self, options: dict, timeout: float = 60):
        """Send a message and wait for idle (SDK pattern)."""
        prompt = options.get("prompt", "")
        self.messages.append({"role": "user", "content": prompt})
        
        # Dispatch mock events to handlers
        for handler in self._event_handlers:
            handler({"type": "assistant.message", "data": {"content": f"\n[Mock] Received: {prompt[:100]}...\n"}})
            handler({"type": "session.idle", "data": {}})
        
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
