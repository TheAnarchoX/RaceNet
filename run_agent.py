#!/usr/bin/env python3
"""
RaceNet Autonomous Agent Runner

An autonomous development agent system that uses the GitHub Copilot SDK to:
- Work on tasks from TASKS.md (single or multi-agent)
- Coordinate as a hive mind with shared knowledge
- Learn and improve from its own performance
- Propose new tasks and improvements
- Continuously evolve the simulation framework

Usage:
    python run_agent.py                    # Run single agent
    python run_agent.py --plan             # Strategic planner mode
    python run_agent.py --multi 3          # Run 3 coordinated agents
    python run_agent.py --hive             # Run hive mind (default 3 agents)
    python run_agent.py --report           # Show improvement report
    python run_agent.py --dry-run          # Preview without changes
    python run_agent.py --help             # Show all options
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add agent to path
sys.path.insert(0, str(Path(__file__).parent))

from agent.config import AgentConfig
from agent.autonomous_agent import AutonomousAgent
from agent.orchestrator import MultiAgentOrchestrator, OrchestratorConfig, AgentRole
from agent.self_improvement import SelfImprovementEngine


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="RaceNet Autonomous Development Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run a single agent
    python run_agent.py

    # Run strategic planner - deep codebase analysis and task management
    python run_agent.py --plan

    # Run 3 coordinated agents as a hive mind
    python run_agent.py --hive
    python run_agent.py --multi 3

    # Run 5 agents with specific roles
    python run_agent.py --multi 5 --roles physics,track,ml,testing,generalist

    # Dry run - preview what would be done
    python run_agent.py --dry-run

    # Show self-improvement report
    python run_agent.py --report

    # Run improvement cycle
    python run_agent.py --improve

    # Run with verbose logging
    python run_agent.py --log-level DEBUG

    # Connect to external CLI server
    python run_agent.py --cli-url http://localhost:4321
        """
    )
    
    # Mode selection
    mode_group = parser.add_argument_group("Mode Selection")
    mode_group.add_argument(
        "--multi",
        type=int,
        metavar="N",
        help="Run N coordinated agents as a hive mind"
    )
    mode_group.add_argument(
        "--hive",
        action="store_true",
        help="Run in hive mind mode with default 3 agents"
    )
    mode_group.add_argument(
        "--roles",
        type=str,
        help="Comma-separated agent roles: physics,track,ml,testing,documentation,reviewer,generalist"
    )
    mode_group.add_argument(
        "--report",
        action="store_true",
        help="Show self-improvement report and exit"
    )
    mode_group.add_argument(
        "--improve",
        action="store_true",
        help="Run improvement cycle and show results"
    )
    mode_group.add_argument(
        "--plan",
        action="store_true",
        help="Run dedicated planning mode to analyze codebase and manage tasks"
    )
    
    # Model settings
    parser.add_argument(
        "--model",
        default="gpt-5.2-codex",
        help="Model to use (default: gpt-5.2-codex)"
    )
    
    # Behavior settings
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=100,
        help="Maximum iterations before stopping (default: 100)"
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=10,
        dest="max_tasks_per_session",
        help="Maximum tasks to complete per session (default: 10)"
    )
    parser.add_argument(
        "--no-auto-propose",
        action="store_false",
        dest="auto_propose_tasks",
        help="Disable automatic task proposal"
    )
    parser.add_argument(
        "--no-auto-test",
        action="store_false",
        dest="auto_run_tests",
        help="Disable automatic test running"
    )
    parser.add_argument(
        "--no-self-improve",
        action="store_false",
        dest="enable_self_improvement",
        help="Disable self-improvement tracking"
    )
    parser.add_argument(
        "--turn-timeout",
        type=int,
        help="Timeout per assistant turn in seconds (overrides request timeout)"
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Enable fast mode with aggressive prompt/tool limits"
    )
    
    # Safety settings
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without actually making them"
    )
    parser.add_argument(
        "--require-approval",
        action="store_true",
        dest="require_approval_for_writes",
        help="Require approval before writing files"
    )
    
    # Logging
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Log level (default: INFO)"
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        help="Log file path (in addition to stdout)"
    )

    # Performance/benchmarking
    parser.add_argument(
        "--perf-log",
        type=Path,
        help="Path to JSONL performance log file"
    )
    parser.add_argument(
        "--perf-sample-interval",
        type=int,
        default=30,
        help="Seconds between perf samples (default: 30)"
    )
    parser.add_argument(
        "--tracemalloc",
        action="store_true",
        help="Enable tracemalloc memory tracking"
    )
    parser.add_argument(
        "--profile",
        type=Path,
        help="Write cProfile stats to the given file"
    )
    
    # Copilot SDK settings
    parser.add_argument(
        "--cli-url",
        help="URL of external Copilot CLI server (e.g., http://localhost:4321)"
    )
    
    # Repository
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root directory (default: current directory)"
    )
    
    return parser.parse_args()


def print_banner(mode: str = "single"):
    """Print startup banner."""
    if mode == "hive":
        banner = """
╔═══════════════════════════════════════════════════════════════════════╗
║                                                                       ║
║   🏎️  RaceNet Autonomous Agent Hive                                     ║
║                                                                       ║
║   Powered by GitHub Copilot SDK + Hive Mind                           ║
║   Model: GPT-5.2-Codex                                                ║
║                                                                       ║
║   Multiple agents working together:                                   ║
║   • Shared knowledge base for collective learning                     ║
║   • Coordinated task assignment (no conflicts)                        ║
║   • Self-improving based on performance                               ║
║   • Automatic role specialization                                     ║
║                                                                       ║
║   Press Ctrl+C to stop                                                ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════╝
"""
    elif mode == "planner":
        banner = """
╔═══════════════════════════════════════════════════════════════════════╗
║                                                                       ║
║   🗺️  RaceNet Strategic Planner                                         ║
║                                                                       ║
║   Powered by GitHub Copilot SDK                                       ║
║   Model: GPT-5.2-Codex                                                ║
║                                                                       ║
║   Deep codebase analysis mode:                                        ║
║   • Read actual source code (not just docs)                           ║
║   • Identify bugs, tech debt, missing features                        ║
║   • Create new tasks with specific requirements                       ║
║   • Refine and prioritize existing tasks                              ║
║   • Analyze test coverage gaps                                        ║
║                                                                       ║
║   Press Ctrl+C to stop                                                ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════╝
"""
    else:
        banner = """
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║   🏎️  RaceNet Autonomous Agent                                  ║
║                                                               ║
║   Powered by GitHub Copilot SDK                               ║
║   Model: GPT-5.2-Codex                                        ║
║                                                               ║
║   The agent will autonomously:                                ║
║   • Work on tasks from TASKS.md                               ║
║   • Write and test code                                       ║
║   • Learn from successes and failures                         ║
║   • Propose new improvements                                  ║
║   • Evolve the simulation framework                           ║
║                                                               ║
║   Press Ctrl+C to stop                                        ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
"""
    print(banner)


def check_prerequisites(
    repo_root: Path,
    tasks_file: str = "TASKS.md",
    cli_url: str | None = None,
) -> bool:
    """Check that prerequisites are met."""
    issues = []
    
    # Check repo root exists
    if not repo_root.exists():
        issues.append(f"Repository root not found: {repo_root}")
    
    # Check TASKS.md exists
    tasks_path = repo_root / tasks_file
    if not tasks_path.exists():
        issues.append(f"Tasks file not found: {tasks_path}")
    
    # Check for copilot-sdk and Copilot CLI (warn only, allow mock fallback)
    import importlib.util

    has_sdk = importlib.util.find_spec("copilot") is not None
    if not has_sdk:
        print("⚠️  copilot-sdk not installed. Will run in mock mode.")
        print("   Install with: pip install github-copilot-sdk")

    if has_sdk and not cli_url:
        import shutil
        if not shutil.which("copilot"):
            print("⚠️  Copilot CLI not found. Will attempt mock fallback.")
            print("   Install from: https://docs.github.com/en/copilot/how-tos/set-up/install-copilot-cli")
    
    if issues:
        print("❌ Prerequisites check failed:")
        for issue in issues:
            print(f"   • {issue}")
        return False
    
    print("✅ Prerequisites check passed")
    return True


def parse_roles(roles_str: str) -> list[AgentRole]:
    """Parse role string into list of AgentRole."""
    role_map = {
        "physics": AgentRole.PHYSICS,
        "track": AgentRole.TRACK,
        "ml": AgentRole.ML,
        "testing": AgentRole.TESTING,
        "documentation": AgentRole.DOCUMENTATION,
        "reviewer": AgentRole.REVIEWER,
        "generalist": AgentRole.GENERALIST,
    }
    
    roles = []
    for role_name in roles_str.split(","):
        role_name = role_name.strip().lower()
        if role_name in role_map:
            roles.append(role_map[role_name])
        else:
            print(f"⚠️  Unknown role '{role_name}', using generalist")
            roles.append(AgentRole.GENERALIST)
    
    return roles


def show_improvement_report(repo_root: Path):
    """Show the self-improvement report."""
    engine = SelfImprovementEngine(repo_root)
    report = engine.get_improvement_report()
    print(report)


def run_improvement_cycle(repo_root: Path):
    """Run an improvement cycle and show results."""
    engine = SelfImprovementEngine(repo_root)
    results = engine.run_improvement_cycle()
    
    print("\n" + "=" * 60)
    print("IMPROVEMENT CYCLE RESULTS")
    print("=" * 60)
    print(f"\nTimestamp: {results['timestamp']}")
    print(f"Insights generated: {results['insights_generated']}")
    
    if results['suggestions']:
        print("\n📋 Recommendations:")
        for rec in results['suggestions'][:10]:
            print(f"   {rec['recommendation']}")
    
    if results.get('role_adjustments'):
        print("\n👥 Agent Role Adjustments:")
        for adj in results['role_adjustments']:
            print(f"   • {adj['agent_id']}: {adj['suggestion']}")
    
    if results.get('improvement_tasks'):
        print("\n🔧 Suggested Self-Improvement Tasks:")
        for task in results['improvement_tasks']:
            print(f"   • [{task['priority']}] {task['title']}")
    
    print("\n" + "=" * 60)


async def run_planner(args: argparse.Namespace):
    """Run in dedicated planning mode to analyze codebase and manage tasks."""
    config = AgentConfig(
        model=args.model,
        repo_root=args.repo_root,
        max_iterations=1,  # Single planning iteration
        max_tasks_per_session=1,
        auto_propose_tasks=False,
        auto_run_tests=False,
        require_approval_for_writes=args.require_approval_for_writes,
        dry_run=args.dry_run,
        log_level=args.log_level,
        log_file=args.log_file,
        cli_url=args.cli_url,
        planner_mode=True,
        enable_self_improvement=args.enable_self_improvement,
        perf_log_path=args.perf_log,
        perf_sample_interval_seconds=args.perf_sample_interval,
        enable_tracemalloc=args.tracemalloc,
        turn_timeout_seconds=args.turn_timeout,
        fast_mode=args.fast,
    )
    
    print(f"📁 Repository: {config.repo_root}")
    print(f"🤖 Model: {config.model}")
    print(f"🗺️  Mode: PLANNER")
    if config.dry_run:
        print("⚠️  DRY RUN MODE - No changes will be made")
    print()
    
    # Check prerequisites
    if not check_prerequisites(config.repo_root, cli_url=config.cli_url):
        sys.exit(1)
    
    # Create and start the agent in planner mode
    agent = AutonomousAgent(config)
    if args.profile:
        import cProfile
        profiler = cProfile.Profile()
        profiler.enable()
        await agent.start()
        profiler.disable()
        profiler.dump_stats(args.profile)
    else:
        await agent.start()


async def run_single_agent(args: argparse.Namespace):
    """Run a single autonomous agent."""
    config = AgentConfig(
        model=args.model,
        repo_root=args.repo_root,
        max_iterations=args.max_iterations,
        max_tasks_per_session=args.max_tasks_per_session,
        auto_propose_tasks=args.auto_propose_tasks,
        auto_run_tests=args.auto_run_tests,
        require_approval_for_writes=args.require_approval_for_writes,
        dry_run=args.dry_run,
        log_level=args.log_level,
        log_file=args.log_file,
        cli_url=args.cli_url,
        enable_self_improvement=args.enable_self_improvement,
        perf_log_path=args.perf_log,
        perf_sample_interval_seconds=args.perf_sample_interval,
        enable_tracemalloc=args.tracemalloc,
        turn_timeout_seconds=args.turn_timeout,
        fast_mode=args.fast,
    )
    
    print(f"📁 Repository: {config.repo_root}")
    print(f"🤖 Model: {config.model}")
    print(f"🔄 Max iterations: {config.max_iterations}")
    print(f"📋 Max tasks per session: {config.max_tasks_per_session}")
    if config.dry_run:
        print("⚠️  DRY RUN MODE - No changes will be made")
    print()
    
    # Check prerequisites
    if not check_prerequisites(config.repo_root, cli_url=config.cli_url):
        sys.exit(1)
    
    # Create and start the agent
    agent = AutonomousAgent(config)
    if args.profile:
        import cProfile
        profiler = cProfile.Profile()
        profiler.enable()
        await agent.start()
        profiler.disable()
        profiler.dump_stats(args.profile)
    else:
        await agent.start()


async def run_hive_mind(args: argparse.Namespace):
    """Run multiple coordinated agents as a hive mind."""
    num_agents = args.multi if args.multi else 3
    
    # Parse roles
    roles = []
    if args.roles:
        roles = parse_roles(args.roles)
    
    config = OrchestratorConfig(
        num_agents=num_agents,
        model=args.model,
        repo_root=args.repo_root,
        max_total_iterations=args.max_iterations * num_agents,
        agent_roles=roles,
        auto_propose_tasks=args.auto_propose_tasks,
        require_tests=args.auto_run_tests,
        dry_run=args.dry_run,
        log_level=args.log_level,
        log_file=args.log_file,
        cli_url=args.cli_url,
        enable_self_improvement=args.enable_self_improvement,
        perf_log_path=args.perf_log,
        perf_sample_interval_seconds=args.perf_sample_interval,
        enable_tracemalloc=args.tracemalloc,
        turn_timeout_seconds=args.turn_timeout,
        fast_mode=args.fast,
    )
    
    print(f"📁 Repository: {config.repo_root}")
    print(f"🤖 Model: {config.model}")
    print(f"👥 Agents: {config.num_agents}")
    if roles:
        print(f"🎭 Roles: {', '.join(r.value for r in roles)}")
    print(f"🔄 Max total iterations: {config.max_total_iterations}")
    if config.dry_run:
        print("⚠️  DRY RUN MODE - No changes will be made")
    print()
    
    # Check prerequisites
    if not check_prerequisites(config.repo_root, cli_url=config.cli_url):
        sys.exit(1)
    
    # Create and start the orchestrator
    orchestrator = MultiAgentOrchestrator(config)
    if args.profile:
        import cProfile
        profiler = cProfile.Profile()
        profiler.enable()
        await orchestrator.start()
        profiler.disable()
        profiler.dump_stats(args.profile)
    else:
        await orchestrator.start()


async def main():
    """Main entry point."""
    args = parse_args()
    
    # Handle report/improve modes
    if args.report:
        show_improvement_report(args.repo_root)
        return
    
    if args.improve:
        run_improvement_cycle(args.repo_root)
        return
    
    # Handle planner mode
    if args.plan:
        print_banner("planner")
        await run_planner(args)
        return
    
    # Determine mode
    is_hive = args.hive or args.multi
    
    print_banner("hive" if is_hive else "single")
    
    if is_hive:
        await run_hive_mind(args)
    else:
        await run_single_agent(args)


def run_with_timeout():
    """Run the main async function with a hard shutdown timeout."""
    import signal
    import os
    import asyncio
    
    # Track if we're already shutting down
    shutdown_initiated = False
    
    def force_exit(signum, frame):
        nonlocal shutdown_initiated
        if shutdown_initiated:
            try:
                os.write(2, "\n🔴 Forced shutdown!\n".encode("utf-8"))
            except Exception:
                pass
            os._exit(1)
        else:
            shutdown_initiated = True
            try:
                os.write(2, "\n⏳ Shutting down (press Ctrl+C again to force)...\n".encode("utf-8"))
            except Exception:
                pass
    
    # Create event loop manually so we can cancel tasks on shutdown
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    main_task = loop.create_task(main())

    def _cancel_all_tasks():
        for task in asyncio.all_tasks(loop):
            task.cancel()

    # Set up signal handlers before running the loop
    def force_exit(signum, frame):
        nonlocal shutdown_initiated
        if shutdown_initiated:
            try:
                os.write(2, "\n🔴 Forced shutdown!\n".encode("utf-8"))
            except Exception:
                pass
            os._exit(1)
        else:
            shutdown_initiated = True
            try:
                os.write(2, "\n⏳ Shutting down (press Ctrl+C again to force)...\n".encode("utf-8"))
            except Exception:
                pass
            _cancel_all_tasks()

    signal.signal(signal.SIGINT, force_exit)
    signal.signal(signal.SIGTERM, force_exit)

    try:
        loop.run_until_complete(main_task)
    except asyncio.CancelledError:
        try:
            os.write(2, "\n✅ Shutdown complete\n".encode("utf-8"))
        except Exception:
            pass
    except KeyboardInterrupt:
        try:
            os.write(2, "\n✅ Shutdown complete\n".encode("utf-8"))
        except Exception:
            pass
    except Exception as e:
        try:
            os.write(2, f"\n❌ Error: {e}\n".encode("utf-8"))
        except Exception:
            pass
        sys.exit(1)
    finally:
        try:
            loop.run_until_complete(loop.shutdown_asyncgens())
        finally:
            loop.close()


if __name__ == "__main__":
    run_with_timeout()
