"""
Autonomous Agent

Main orchestration for the autonomous RaceNet development agent.
Uses the GitHub Copilot SDK to work on tasks and evolve the project.
"""

import json
import logging
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
        
        # Create session with tools
        tools = create_tool_definitions(self.config)
        
        self.session = await self.client.create_session(
            system_message={"content": SYSTEM_MESSAGE},
            model=self.config.model,
            tools=tools,
        )
        
        logger.info("Copilot session created successfully")
        
        # Get initial task summary
        summary = self.task_manager.get_summary()
        logger.info(f"Task Summary: {summary['completed']}/{summary['total']} completed ({summary['completion_percentage']:.1f}%)")
        
        # Main loop
        while self._iteration < self.config.max_iterations:
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
        """Send a message and handle the streaming response with tool calls."""
        logger.info(f"Sending message to Copilot...")
        
        try:
            # Use streaming to handle tool calls
            async for event in self.session.send_message_stream(content):
                await self._handle_event(event)
        except Exception as e:
            logger.error(f"Error sending message: {e}")
    
    async def _handle_event(self, event: Any):
        """Handle a streaming event from Copilot."""
        event_type = getattr(event, "type", None)
        
        if event_type == "text":
            # Print text content as it streams
            text = getattr(event, "text", "")
            if text:
                print(text, end="", flush=True)
        
        elif event_type == "tool_call":
            # Handle tool calls
            tool_name = event.name
            tool_params = event.parameters
            
            logger.info(f"Tool call: {tool_name}")
            logger.debug(f"Parameters: {json.dumps(tool_params, indent=2)}")
            
            # Execute the tool
            result = self.tool_handler.handle_tool_call(tool_name, tool_params)
            
            # Send the result back
            await self.session.submit_tool_result(event.call_id, result)
        
        elif event_type == "done":
            print()  # New line after streaming
            logger.info("Response complete")
    
    async def _cleanup(self):
        """Clean up resources."""
        if self.session:
            await self.session.close()
        if self.client:
            await self.client.close()
        
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
        self.messages = []
    
    async def send_message_stream(self, content: str):
        """Simulate streaming response."""
        self.messages.append({"role": "user", "content": content})
        
        # Yield a simple response
        class TextEvent:
            type = "text"
            text = f"\n[Mock Response] Received: {content[:100]}...\n"
        
        yield TextEvent()
        
        class DoneEvent:
            type = "done"
        
        yield DoneEvent()
    
    async def submit_tool_result(self, call_id: str, result: str):
        """Handle tool result."""
        pass
    
    async def close(self):
        """Clean up."""
        pass


class MockCopilotClient:
    """Mock client for testing without the actual SDK."""
    
    def __init__(self, **kwargs):
        self.tool_handler = None
    
    async def create_session(self, **kwargs):
        """Create a mock session."""
        # We'll need to set the tool handler after creation
        return MockCopilotSession(self.tool_handler)
    
    async def close(self):
        """Clean up."""
        pass


# For environments without the SDK, provide a fallback
if not HAS_COPILOT_SDK:
    CopilotClient = MockCopilotClient
