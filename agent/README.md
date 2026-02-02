# RaceNet Autonomous Agent System

An autonomous, self-improving development agent system powered by the GitHub Copilot SDK. Multiple agents can work together as a "hive mind" to continuously evolve the RaceNet racing simulation.

## Features

### 🤖 Single Agent Mode
- Autonomous task execution from TASKS.md
- Code writing, testing, and committing
- Automatic task proposal for improvements

### 🐝 Hive Mind Mode
- Multiple coordinated agents working together
- Shared knowledge base for collective learning
- File locking to prevent conflicts
- Inter-agent messaging and coordination
- Role specialization (physics, ML, testing, etc.)

### 📈 Self-Improvement
- Performance metrics tracking
- Learning from successes and failures
- Automatic insight generation
- Dynamic prompt optimization
- Agent role adjustment recommendations

## Quick Start

### Prerequisites

1. **GitHub Copilot CLI** installed and authenticated:
   ```bash
   # Install Copilot CLI
   # See: https://docs.github.com/en/copilot/how-tos/set-up/install-copilot-cli
   
   # Verify installation
   copilot --version
   ```

2. **Copilot SDK** (optional, falls back to mock mode):
   ```bash
   pip install github-copilot-sdk
   ```

### Running the Agent

```bash
# Single agent (default)
python run_agent.py

# Multi-agent hive mind (3 agents)
python run_agent.py --hive

# 5 agents with specific roles
python run_agent.py --multi 5 --roles physics,track,ml,testing,generalist

# Dry run (preview without changes)
python run_agent.py --dry-run

# Show self-improvement report
python run_agent.py --report
```

## Architecture

```
agent/
├── __init__.py           # Package exports
├── config.py             # Agent configuration
├── task_manager.py       # TASKS.md parsing and management
├── tools.py              # Copilot tools (file ops, git, etc.)
├── autonomous_agent.py   # Single agent implementation
├── memory.py             # Shared knowledge base (SQLite)
├── orchestrator.py       # Multi-agent coordination
└── self_improvement.py   # Performance tracking and learning
```

## How It Works

### Task Execution Flow

1. **Task Selection**: Agent reads TASKS.md and picks the highest-priority task with met dependencies
2. **Context Gathering**: Queries shared knowledge base for relevant information
3. **File Locking**: Acquires locks on files to prevent conflicts with other agents
4. **Implementation**: Uses Copilot to write code, following project conventions
5. **Testing**: Runs tests to verify changes
6. **Knowledge Sharing**: Stores learnings in the shared knowledge base
7. **Commit**: Commits changes and moves to next task

### Hive Mind Coordination

```
┌─────────────────────────────────────────────────────────────┐
│                    Orchestrator                              │
│  - Spawns agents         - Monitors health                  │
│  - Assigns tasks         - Prevents conflicts               │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
   ┌─────────┐           ┌─────────┐           ┌─────────┐
   │ Agent 1 │           │ Agent 2 │           │ Agent 3 │
   │ Physics │           │   ML    │           │ Testing │
   └────┬────┘           └────┬────┘           └────┬────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              ▼
        ┌─────────────────────────────────────────────┐
        │           Shared Knowledge Base             │
        │  - Code facts      - Task learnings         │
        │  - Patterns        - File locks             │
        │  - Bug fixes       - Agent messages         │
        └─────────────────────────────────────────────┘
```

### Self-Improvement System

The agent system continuously improves by:

1. **Recording Metrics**: Every action is tracked with outcome, duration, tools used
2. **Analyzing Patterns**: Identifies what leads to success vs failure
3. **Generating Insights**: Creates actionable recommendations
4. **Optimizing Behavior**: Adjusts prompts and strategies based on learnings

```bash
# View improvement report
python run_agent.py --report

# Run improvement cycle
python run_agent.py --improve
```

## Agent Roles

| Role | Description | Best For |
|------|-------------|----------|
| `generalist` | Works on any task | Default, all-purpose |
| `physics` | Vehicle dynamics, tires, aero | Physics-related tasks |
| `track` | Track generation, racing line | Track generation tasks |
| `ml` | RL environments, training | ML integration tasks |
| `testing` | Test coverage, quality | Testing tasks |
| `documentation` | Docs, examples | Documentation |
| `reviewer` | Code review | Review changes |

## Configuration

### Command Line Options

```
Mode Selection:
  --multi N          Run N coordinated agents
  --hive             Run 3 agents (shortcut)
  --roles ROLES      Comma-separated agent roles
  --report           Show improvement report
  --improve          Run improvement cycle

Model:
  --model MODEL      Model to use (default: gpt-5.2-codex)

Behavior:
  --max-iterations N Maximum iterations (default: 100)
  --max-tasks N      Max tasks per session (default: 10)
  --no-auto-propose  Disable task proposal
  --no-auto-test     Disable automatic testing
  --no-self-improve  Disable performance tracking

Safety:
  --dry-run          Preview without changes
  --require-approval Require approval for writes

Logging:
  --log-level LEVEL  DEBUG, INFO, WARNING, ERROR
  --log-file FILE    Log to file

Copilot:
  --cli-url URL      External CLI server URL
  --repo-root PATH   Repository root directory
```

## Knowledge Types

The shared knowledge base stores these types of knowledge:

| Type | Description |
|------|-------------|
| `code_fact` | Facts about code structure/behavior |
| `pattern` | Code patterns and conventions |
| `task_learning` | What worked/didn't on tasks |
| `dependency` | Module/file dependencies |
| `bug_fix` | Bug fixes and solutions |
| `optimization` | Performance optimizations |
| `test_result` | Test results and coverage |
| `agent_insight` | Agent observations |
| `convention` | Coding conventions |
| `todo` | Things to do later |

## Data Storage

The agent system creates two SQLite databases:

- `.racenet_memory.db` - Shared knowledge, agent states, file locks, messages
- `.racenet_performance.db` - Performance metrics, outcomes, insights

Add to `.gitignore`:
```
.racenet_memory.db
.racenet_performance.db
```

## Example Session

```
$ python run_agent.py --hive --roles physics,track,ml

╔═══════════════════════════════════════════════════════════════════════╗
║   🏎️  RaceNet Autonomous Agent Hive                                  ║
║   Multiple agents working together as a hive mind                     ║
╚═══════════════════════════════════════════════════════════════════════╝

📁 Repository: /path/to/RaceNet
🤖 Model: gpt-5.2-codex
👥 Agents: 3
🎭 Roles: physics, track, ml
✅ Prerequisites check passed

2024-01-15 10:00:00 [INFO] Spawned agent-01 with role physics
2024-01-15 10:00:01 [INFO] Spawned agent-02 with role track
2024-01-15 10:00:02 [INFO] Spawned agent-03 with role ml

--------------------------------------------------
HIVE STATUS
  Active agents: 3/3
  Tasks: 2/16 completed
  Shared knowledge: 15 entries
    agent-01: working (Task 1.1) | 1 tasks, 5 knowledge
    agent-02: working (Task 2.1) | 1 tasks, 7 knowledge
    agent-03: idle | 0 tasks, 3 knowledge
--------------------------------------------------
```

## Contributing

The agent system can be extended by:

1. **Adding new tools** in `tools.py`
2. **Adding new roles** in `orchestrator.py`
3. **Customizing insights** in `self_improvement.py`
4. **Extending knowledge types** in `memory.py`

## License

MIT License - See LICENSE file for details.
