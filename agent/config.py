"""
Agent Configuration

Configuration settings for the autonomous RaceNet agent.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class AgentConfig:
    """Configuration for the autonomous agent."""
    
    # Model settings
    model: str = "gpt-5.2-codex"
    
    # Repository settings
    repo_root: Path = field(default_factory=lambda: Path.cwd())
    tasks_file: str = "TASKS.md"
    
    # Agent behavior
    max_iterations: int = 100
    max_tasks_per_session: int = 10
    auto_propose_tasks: bool = True
    auto_run_tests: bool = True
    request_timeout: int = 1800  # 30 minute timeout per request
    
    # Safety settings
    require_approval_for_writes: bool = False
    dry_run: bool = False
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[Path] = None
    
    # Copilot SDK settings
    cli_url: Optional[str] = None  # Use external CLI server if provided
    
    @property
    def tasks_path(self) -> Path:
        """Get the full path to TASKS.md."""
        return self.repo_root / self.tasks_file
    
    def __post_init__(self):
        """Validate and normalize configuration."""
        if isinstance(self.repo_root, str):
            self.repo_root = Path(self.repo_root)
        if self.log_file and isinstance(self.log_file, str):
            self.log_file = Path(self.log_file)
