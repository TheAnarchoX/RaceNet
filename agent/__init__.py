"""
RaceNet Autonomous Agent

An autonomous development agent system powered by the GitHub Copilot SDK.

Features:
- Single agent or multi-agent hive mind operation
- Shared knowledge base for collective learning
- Self-improvement based on performance metrics
- Automatic task proposal and prioritization
- Coordinated file locking to prevent conflicts

Usage:
    # Single agent
    python run_agent.py

    # Multi-agent hive mind
    python run_agent.py --hive
    python run_agent.py --multi 5

    # Show improvement report
    python run_agent.py --report
"""

from agent.config import AgentConfig
from agent.task_manager import TaskManager, Task, TaskStatus
from agent.autonomous_agent import AutonomousAgent
from agent.memory import RepositoryMemory, KnowledgeEntry, KnowledgeType
from agent.orchestrator import MultiAgentOrchestrator, OrchestratorConfig, AgentRole
from agent.self_improvement import SelfImprovementEngine, PerformanceMetric, ActionOutcome

__all__ = [
    # Config
    "AgentConfig",
    "OrchestratorConfig",
    
    # Task Management
    "TaskManager",
    "Task",
    "TaskStatus",
    
    # Agents
    "AutonomousAgent",
    "MultiAgentOrchestrator",
    "AgentRole",
    
    # Memory & Knowledge
    "RepositoryMemory",
    "KnowledgeEntry",
    "KnowledgeType",
    
    # Self-Improvement
    "SelfImprovementEngine",
    "PerformanceMetric",
    "ActionOutcome",
]
