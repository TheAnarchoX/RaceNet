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
    planner_mode: bool = False  # Dedicated planning/task management mode
    turn_timeout_seconds: int | None = None
    fast_mode: bool = False

    # Performance tuning
    read_file_max_chars: int = 20000
    list_directory_max_items: int = 200
    tool_output_max_chars: int = 20000
    search_code_max_matches: int = 50
    max_response_chars: int = 120000
    tool_cache_ttl_seconds: int = 30
    perf_log_path: Optional[Path] = None
    perf_sample_interval_seconds: int = 30
    enable_tracemalloc: bool = False
    prompt_max_chars: int = 4000

    # Advanced capabilities
    enable_self_improvement: bool = True
    enable_memory: bool = True
    
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
        if self.perf_log_path and isinstance(self.perf_log_path, str):
            self.perf_log_path = Path(self.perf_log_path)

        if self.fast_mode:
            self.auto_run_tests = False
            self.auto_propose_tasks = False
            self.prompt_max_chars = min(self.prompt_max_chars, 2000)
            self.read_file_max_chars = min(self.read_file_max_chars, 8000)
            self.list_directory_max_items = min(self.list_directory_max_items, 100)
            self.tool_output_max_chars = min(self.tool_output_max_chars, 8000)
            self.search_code_max_matches = min(self.search_code_max_matches, 20)
            self.tool_cache_ttl_seconds = max(self.tool_cache_ttl_seconds, 60)
