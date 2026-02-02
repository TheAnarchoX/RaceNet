"""
Self-Improvement System

Enables the agent system to improve itself based on performance metrics.
Tracks what works, what doesn't, and adjusts behavior accordingly.
"""

import json
import statistics
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Optional
import sqlite3


class MetricType(Enum):
    """Types of performance metrics."""
    TASK_COMPLETION_TIME = "task_completion_time"
    TEST_PASS_RATE = "test_pass_rate"
    CODE_QUALITY_SCORE = "code_quality_score"
    KNOWLEDGE_USEFULNESS = "knowledge_usefulness"
    TOOL_SUCCESS_RATE = "tool_success_rate"
    ERROR_RATE = "error_rate"
    ITERATION_COUNT = "iteration_count"
    FILES_MODIFIED = "files_modified"
    LINES_CHANGED = "lines_changed"


class OutcomeType(Enum):
    """Outcome of an action."""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILURE = "failure"
    ERROR = "error"


@dataclass
class PerformanceMetric:
    """A single performance metric."""
    id: str
    metric_type: MetricType
    value: float
    context: dict = field(default_factory=dict)
    agent_id: str = ""
    task_id: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> dict:
        data = asdict(self)
        data["metric_type"] = self.metric_type.value
        return data


@dataclass
class ActionOutcome:
    """Records the outcome of an agent action."""
    id: str
    action_type: str  # e.g., "edit_file", "run_tests", "implement_task"
    outcome: OutcomeType
    agent_id: str
    task_id: str = ""
    details: dict = field(default_factory=dict)
    duration_seconds: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # What led to this outcome
    approach_used: str = ""
    tools_used: list[str] = field(default_factory=list)
    files_involved: list[str] = field(default_factory=list)
    
    # Learning signals
    error_message: str = ""
    success_factors: list[str] = field(default_factory=list)
    failure_reasons: list[str] = field(default_factory=list)


@dataclass
class LearningInsight:
    """An insight learned from performance analysis."""
    id: str
    insight_type: str  # "do_more", "do_less", "avoid", "prefer", "optimize"
    description: str
    confidence: float  # 0.0 to 1.0
    evidence_count: int
    context: dict = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    applied_count: int = 0
    effectiveness_score: float = 0.0


@dataclass
class PromptOptimization:
    """An optimization to agent prompts."""
    id: str
    target: str  # "system_prompt", "task_prompt", "tool_description"
    original: str
    optimized: str
    reason: str
    performance_improvement: float = 0.0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    active: bool = True


class SelfImprovementEngine:
    """
    Engine that enables the agent system to improve itself.
    
    Capabilities:
    - Track performance metrics over time
    - Identify patterns in successes and failures
    - Generate learning insights
    - Optimize prompts and behavior
    - Adjust agent configurations
    - Propose code improvements to itself
    """
    
    def __init__(self, repo_root: Path, db_name: str = ".racenet_performance.db"):
        self.repo_root = repo_root
        self.db_path = repo_root / db_name
        self._init_database()
        
        # In-memory caches for quick access
        self._insights_cache: list[LearningInsight] = []
        self._prompt_optimizations: dict[str, PromptOptimization] = {}
        self._load_active_optimizations()
    
    def _init_database(self):
        """Initialize the performance database."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                id TEXT PRIMARY KEY,
                metric_type TEXT,
                value REAL,
                context TEXT,
                agent_id TEXT,
                task_id TEXT,
                timestamp TEXT
            )
        """)
        
        # Action outcomes table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS outcomes (
                id TEXT PRIMARY KEY,
                action_type TEXT,
                outcome TEXT,
                agent_id TEXT,
                task_id TEXT,
                details TEXT,
                duration_seconds REAL,
                timestamp TEXT,
                approach_used TEXT,
                tools_used TEXT,
                files_involved TEXT,
                error_message TEXT,
                success_factors TEXT,
                failure_reasons TEXT
            )
        """)
        
        # Learning insights table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS insights (
                id TEXT PRIMARY KEY,
                insight_type TEXT,
                description TEXT,
                confidence REAL,
                evidence_count INTEGER,
                context TEXT,
                created_at TEXT,
                applied_count INTEGER DEFAULT 0,
                effectiveness_score REAL DEFAULT 0.0
            )
        """)
        
        # Prompt optimizations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prompt_optimizations (
                id TEXT PRIMARY KEY,
                target TEXT,
                original TEXT,
                optimized TEXT,
                reason TEXT,
                performance_improvement REAL,
                created_at TEXT,
                active INTEGER DEFAULT 1
            )
        """)
        
        # Agent performance history
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS agent_performance (
                agent_id TEXT,
                period TEXT,
                tasks_completed INTEGER,
                success_rate REAL,
                avg_completion_time REAL,
                knowledge_quality REAL,
                PRIMARY KEY (agent_id, period)
            )
        """)
        
        # Strategy effectiveness
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS strategies (
                id TEXT PRIMARY KEY,
                name TEXT,
                description TEXT,
                success_count INTEGER DEFAULT 0,
                failure_count INTEGER DEFAULT 0,
                avg_improvement REAL DEFAULT 0.0,
                active INTEGER DEFAULT 1
            )
        """)
        
        conn.commit()
        conn.close()
    
    def _get_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))
    
    # =========================================================================
    # Metric Recording
    # =========================================================================
    
    def record_metric(self, metric: PerformanceMetric):
        """Record a performance metric."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO metrics (id, metric_type, value, context, agent_id, task_id, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            metric.id,
            metric.metric_type.value,
            metric.value,
            json.dumps(metric.context),
            metric.agent_id,
            metric.task_id,
            metric.timestamp,
        ))
        
        conn.commit()
        conn.close()
    
    def record_outcome(self, outcome: ActionOutcome):
        """Record an action outcome."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO outcomes 
            (id, action_type, outcome, agent_id, task_id, details, duration_seconds,
             timestamp, approach_used, tools_used, files_involved, error_message,
             success_factors, failure_reasons)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            outcome.id,
            outcome.action_type,
            outcome.outcome.value,
            outcome.agent_id,
            outcome.task_id,
            json.dumps(outcome.details),
            outcome.duration_seconds,
            outcome.timestamp,
            outcome.approach_used,
            json.dumps(outcome.tools_used),
            json.dumps(outcome.files_involved),
            outcome.error_message,
            json.dumps(outcome.success_factors),
            json.dumps(outcome.failure_reasons),
        ))
        
        conn.commit()
        conn.close()
        
        # Trigger analysis after recording
        self._analyze_recent_outcomes()
    
    # =========================================================================
    # Performance Analysis
    # =========================================================================
    
    def get_agent_performance(self, agent_id: str, days: int = 7) -> dict:
        """Get performance summary for an agent."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        
        # Get outcomes
        cursor.execute("""
            SELECT outcome, COUNT(*), AVG(duration_seconds)
            FROM outcomes
            WHERE agent_id = ? AND timestamp > ?
            GROUP BY outcome
        """, (agent_id, cutoff))
        
        outcome_stats = {}
        total_actions = 0
        for row in cursor.fetchall():
            outcome_stats[row[0]] = {"count": row[1], "avg_duration": row[2]}
            total_actions += row[1]
        
        # Calculate success rate
        success_count = outcome_stats.get("success", {}).get("count", 0)
        success_rate = success_count / total_actions if total_actions > 0 else 0
        
        # Get metrics
        cursor.execute("""
            SELECT metric_type, AVG(value), MIN(value), MAX(value)
            FROM metrics
            WHERE agent_id = ? AND timestamp > ?
            GROUP BY metric_type
        """, (agent_id, cutoff))
        
        metrics = {}
        for row in cursor.fetchall():
            metrics[row[0]] = {"avg": row[1], "min": row[2], "max": row[3]}
        
        conn.close()
        
        return {
            "agent_id": agent_id,
            "period_days": days,
            "total_actions": total_actions,
            "success_rate": success_rate,
            "outcome_stats": outcome_stats,
            "metrics": metrics,
        }
    
    def get_task_type_performance(self) -> dict:
        """Analyze performance by task type."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT task_id, outcome, COUNT(*), AVG(duration_seconds)
            FROM outcomes
            WHERE task_id != ''
            GROUP BY task_id, outcome
        """)
        
        task_stats = {}
        for row in cursor.fetchall():
            task_id = row[0]
            if task_id not in task_stats:
                task_stats[task_id] = {"outcomes": {}, "total": 0}
            task_stats[task_id]["outcomes"][row[1]] = {
                "count": row[2],
                "avg_duration": row[3]
            }
            task_stats[task_id]["total"] += row[2]
        
        conn.close()
        
        return task_stats
    
    def get_tool_effectiveness(self) -> dict:
        """Analyze which tools are most effective."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT tools_used, outcome FROM outcomes")
        
        tool_stats = {}
        for row in cursor.fetchall():
            tools = json.loads(row[0]) if row[0] else []
            outcome = row[1]
            
            for tool in tools:
                if tool not in tool_stats:
                    tool_stats[tool] = {"success": 0, "failure": 0, "total": 0}
                tool_stats[tool]["total"] += 1
                if outcome == "success":
                    tool_stats[tool]["success"] += 1
                elif outcome in ("failure", "error"):
                    tool_stats[tool]["failure"] += 1
        
        # Calculate effectiveness scores
        for tool, stats in tool_stats.items():
            if stats["total"] > 0:
                stats["effectiveness"] = stats["success"] / stats["total"]
            else:
                stats["effectiveness"] = 0
        
        conn.close()
        
        return tool_stats
    
    def get_approach_effectiveness(self) -> dict:
        """Analyze which approaches work best."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT approach_used, outcome, COUNT(*), AVG(duration_seconds)
            FROM outcomes
            WHERE approach_used != ''
            GROUP BY approach_used, outcome
        """)
        
        approach_stats = {}
        for row in cursor.fetchall():
            approach = row[0]
            if approach not in approach_stats:
                approach_stats[approach] = {"outcomes": {}, "total": 0}
            approach_stats[approach]["outcomes"][row[1]] = {
                "count": row[2],
                "avg_duration": row[3]
            }
            approach_stats[approach]["total"] += row[2]
        
        # Calculate success rates
        for approach, stats in approach_stats.items():
            success = stats["outcomes"].get("success", {}).get("count", 0)
            stats["success_rate"] = success / stats["total"] if stats["total"] > 0 else 0
        
        conn.close()
        
        return approach_stats
    
    # =========================================================================
    # Learning and Insight Generation
    # =========================================================================
    
    def _analyze_recent_outcomes(self):
        """Analyze recent outcomes to generate insights."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Get recent outcomes
        cutoff = (datetime.now() - timedelta(hours=24)).isoformat()
        cursor.execute("""
            SELECT * FROM outcomes WHERE timestamp > ? ORDER BY timestamp DESC LIMIT 100
        """, (cutoff,))
        
        rows = cursor.fetchall()
        conn.close()
        
        if len(rows) < 5:
            return  # Not enough data
        
        # Analyze patterns
        self._analyze_success_patterns(rows)
        self._analyze_failure_patterns(rows)
        self._analyze_timing_patterns(rows)
    
    def _analyze_success_patterns(self, outcomes: list):
        """Identify what leads to success."""
        successes = [o for o in outcomes if o[2] == "success"]
        
        if len(successes) < 3:
            return
        
        # Analyze common tools in successes
        tool_counts = {}
        for outcome in successes:
            tools = json.loads(outcome[9]) if outcome[9] else []
            for tool in tools:
                tool_counts[tool] = tool_counts.get(tool, 0) + 1
        
        # Find tools that appear in most successes
        for tool, count in tool_counts.items():
            if count >= len(successes) * 0.7:  # 70% threshold
                self._create_insight(
                    insight_type="prefer",
                    description=f"Tool '{tool}' is associated with high success rate",
                    confidence=count / len(successes),
                    evidence_count=count,
                    context={"tool": tool, "success_count": count}
                )
    
    def _analyze_failure_patterns(self, outcomes: list):
        """Identify what leads to failure."""
        failures = [o for o in outcomes if o[2] in ("failure", "error")]
        
        if len(failures) < 3:
            return
        
        # Analyze common error messages
        error_counts = {}
        for outcome in failures:
            error = outcome[11]  # error_message
            if error:
                # Normalize error message
                error_key = error[:100]  # First 100 chars
                error_counts[error_key] = error_counts.get(error_key, 0) + 1
        
        # Find recurring errors
        for error, count in error_counts.items():
            if count >= 2:
                self._create_insight(
                    insight_type="avoid",
                    description=f"Recurring error pattern: {error}",
                    confidence=min(count / len(failures), 0.9),
                    evidence_count=count,
                    context={"error_pattern": error, "occurrence_count": count}
                )
    
    def _analyze_timing_patterns(self, outcomes: list):
        """Analyze timing patterns."""
        # Group by action type
        by_action = {}
        for outcome in outcomes:
            action = outcome[1]  # action_type
            duration = outcome[6]  # duration_seconds
            result = outcome[2]  # outcome
            
            if action not in by_action:
                by_action[action] = {"times": [], "outcomes": []}
            by_action[action]["times"].append(duration)
            by_action[action]["outcomes"].append(result)
        
        # Find actions that are too slow
        for action, data in by_action.items():
            if len(data["times"]) >= 3:
                avg_time = statistics.mean(data["times"])
                if avg_time > 300:  # More than 5 minutes average
                    success_rate = data["outcomes"].count("success") / len(data["outcomes"])
                    
                    if success_rate < 0.5:
                        self._create_insight(
                            insight_type="optimize",
                            description=f"Action '{action}' is slow (avg {avg_time:.0f}s) with low success rate ({success_rate:.0%})",
                            confidence=0.7,
                            evidence_count=len(data["times"]),
                            context={"action": action, "avg_time": avg_time, "success_rate": success_rate}
                        )
    
    def _create_insight(
        self,
        insight_type: str,
        description: str,
        confidence: float,
        evidence_count: int,
        context: dict
    ):
        """Create and store a learning insight."""
        # Check for duplicate insights
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT id FROM insights WHERE description = ?",
            (description,)
        )
        
        if cursor.fetchone():
            # Update existing insight
            cursor.execute("""
                UPDATE insights 
                SET confidence = ?, evidence_count = evidence_count + ?, context = ?
                WHERE description = ?
            """, (confidence, evidence_count, json.dumps(context), description))
        else:
            # Create new insight
            insight_id = f"insight_{int(time.time() * 1000)}"
            cursor.execute("""
                INSERT INTO insights 
                (id, insight_type, description, confidence, evidence_count, context, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                insight_id,
                insight_type,
                description,
                confidence,
                evidence_count,
                json.dumps(context),
                datetime.now().isoformat(),
            ))
        
        conn.commit()
        conn.close()
    
    def get_insights(self, min_confidence: float = 0.5) -> list[LearningInsight]:
        """Get learning insights above a confidence threshold."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM insights 
            WHERE confidence >= ?
            ORDER BY confidence DESC, evidence_count DESC
        """, (min_confidence,))
        
        insights = []
        for row in cursor.fetchall():
            insights.append(LearningInsight(
                id=row[0],
                insight_type=row[1],
                description=row[2],
                confidence=row[3],
                evidence_count=row[4],
                context=json.loads(row[5]) if row[5] else {},
                created_at=row[6],
                applied_count=row[7],
                effectiveness_score=row[8],
            ))
        
        conn.close()
        return insights
    
    def get_actionable_insights(self) -> list[dict]:
        """Get insights formatted as actionable recommendations."""
        insights = self.get_insights(min_confidence=0.6)
        
        recommendations = []
        for insight in insights:
            rec = {
                "type": insight.insight_type,
                "recommendation": "",
                "confidence": insight.confidence,
                "evidence": insight.evidence_count,
            }
            
            if insight.insight_type == "prefer":
                rec["recommendation"] = f"✅ DO: {insight.description}"
            elif insight.insight_type == "avoid":
                rec["recommendation"] = f"❌ DON'T: {insight.description}"
            elif insight.insight_type == "optimize":
                rec["recommendation"] = f"🔧 OPTIMIZE: {insight.description}"
            elif insight.insight_type == "do_more":
                rec["recommendation"] = f"📈 DO MORE: {insight.description}"
            elif insight.insight_type == "do_less":
                rec["recommendation"] = f"📉 DO LESS: {insight.description}"
            else:
                rec["recommendation"] = insight.description
            
            recommendations.append(rec)
        
        return recommendations
    
    # =========================================================================
    # Prompt Optimization
    # =========================================================================
    
    def _load_active_optimizations(self):
        """Load active prompt optimizations."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM prompt_optimizations WHERE active = 1")
        
        for row in cursor.fetchall():
            self._prompt_optimizations[row[1]] = PromptOptimization(
                id=row[0],
                target=row[1],
                original=row[2],
                optimized=row[3],
                reason=row[4],
                performance_improvement=row[5],
                created_at=row[6],
                active=bool(row[7]),
            )
        
        conn.close()
    
    def get_optimized_prompt(self, target: str, default: str) -> str:
        """Get an optimized prompt if available."""
        if target in self._prompt_optimizations:
            opt = self._prompt_optimizations[target]
            if opt.active:
                return opt.optimized
        return default
    
    def generate_prompt_improvements(self) -> list[dict]:
        """Generate suggestions for prompt improvements based on performance."""
        insights = self.get_insights(min_confidence=0.6)
        
        improvements = []
        
        # Generate improvement suggestions based on insights
        for insight in insights:
            if insight.insight_type == "prefer":
                improvements.append({
                    "target": "system_prompt",
                    "suggestion": f"Add emphasis on: {insight.description}",
                    "reason": f"This approach has shown {insight.confidence:.0%} success correlation",
                })
            elif insight.insight_type == "avoid":
                improvements.append({
                    "target": "system_prompt",
                    "suggestion": f"Add warning against: {insight.description}",
                    "reason": f"This pattern is associated with failures ({insight.evidence_count} occurrences)",
                })
        
        return improvements
    
    def get_dynamic_system_prompt_additions(self) -> str:
        """Get dynamic additions to the system prompt based on learnings."""
        insights = self.get_actionable_insights()
        
        if not insights:
            return ""
        
        additions = ["\n## Learned Best Practices\n"]
        additions.append("Based on performance analysis, follow these guidelines:\n")
        
        for rec in insights[:10]:  # Top 10 recommendations
            additions.append(f"- {rec['recommendation']}")
        
        return "\n".join(additions)
    
    # =========================================================================
    # Self-Improvement Actions
    # =========================================================================
    
    def suggest_agent_role_adjustments(self) -> list[dict]:
        """Suggest adjustments to agent roles based on performance."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Get performance by agent
        cursor.execute("""
            SELECT agent_id, 
                   SUM(CASE WHEN outcome = 'success' THEN 1 ELSE 0 END) as successes,
                   COUNT(*) as total,
                   AVG(duration_seconds) as avg_time
            FROM outcomes
            GROUP BY agent_id
        """)
        
        agent_stats = {}
        for row in cursor.fetchall():
            agent_stats[row[0]] = {
                "successes": row[1],
                "total": row[2],
                "success_rate": row[1] / row[2] if row[2] > 0 else 0,
                "avg_time": row[3],
            }
        
        # Get task type performance by agent
        cursor.execute("""
            SELECT agent_id, task_id, 
                   SUM(CASE WHEN outcome = 'success' THEN 1 ELSE 0 END) as successes,
                   COUNT(*) as total
            FROM outcomes
            WHERE task_id != ''
            GROUP BY agent_id, task_id
        """)
        
        agent_task_stats = {}
        for row in cursor.fetchall():
            agent_id = row[0]
            if agent_id not in agent_task_stats:
                agent_task_stats[agent_id] = {}
            agent_task_stats[agent_id][row[1]] = {
                "successes": row[2],
                "total": row[3],
                "success_rate": row[2] / row[3] if row[3] > 0 else 0,
            }
        
        conn.close()
        
        # Generate suggestions
        suggestions = []
        
        for agent_id, stats in agent_stats.items():
            if stats["success_rate"] < 0.5 and stats["total"] >= 5:
                suggestions.append({
                    "agent_id": agent_id,
                    "issue": "Low success rate",
                    "current_rate": stats["success_rate"],
                    "suggestion": "Consider assigning simpler tasks or providing more context",
                })
            
            if stats["avg_time"] > 600 and stats["total"] >= 3:  # > 10 minutes avg
                suggestions.append({
                    "agent_id": agent_id,
                    "issue": "Slow task completion",
                    "avg_time": stats["avg_time"],
                    "suggestion": "Consider breaking tasks into smaller pieces",
                })
        
        return suggestions
    
    def generate_self_improvement_tasks(self) -> list[dict]:
        """Generate tasks for improving the agent system itself."""
        tasks = []
        
        # Analyze tool effectiveness
        tool_stats = self.get_tool_effectiveness()
        for tool, stats in tool_stats.items():
            if stats["effectiveness"] < 0.4 and stats["total"] >= 5:
                tasks.append({
                    "title": f"Improve tool '{tool}' effectiveness",
                    "description": f"Tool '{tool}' has low effectiveness ({stats['effectiveness']:.0%}). Analyze failures and improve the tool implementation.",
                    "priority": "P2",
                    "type": "self_improvement",
                })
        
        # Check for recurring errors
        insights = self.get_insights()
        error_insights = [i for i in insights if i.insight_type == "avoid" and i.confidence > 0.7]
        
        for insight in error_insights[:3]:
            tasks.append({
                "title": f"Fix recurring issue: {insight.description[:50]}...",
                "description": f"Address the recurring problem: {insight.description}",
                "priority": "P1",
                "type": "self_improvement",
            })
        
        return tasks
    
    def get_improvement_report(self) -> str:
        """Generate a comprehensive improvement report."""
        report = ["# Self-Improvement Report", ""]
        report.append(f"Generated: {datetime.now().isoformat()}")
        report.append("")
        
        # Overall stats
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM outcomes")
        total_outcomes = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM outcomes WHERE outcome = 'success'")
        total_successes = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM insights")
        total_insights = cursor.fetchone()[0]
        
        conn.close()
        
        overall_success = total_successes / total_outcomes if total_outcomes > 0 else 0
        
        report.append("## Overall Performance")
        report.append(f"- Total actions recorded: {total_outcomes}")
        report.append(f"- Overall success rate: {overall_success:.1%}")
        report.append(f"- Learning insights generated: {total_insights}")
        report.append("")
        
        # Top recommendations
        report.append("## Key Recommendations")
        recommendations = self.get_actionable_insights()
        for rec in recommendations[:5]:
            report.append(f"- {rec['recommendation']} (confidence: {rec['confidence']:.0%})")
        report.append("")
        
        # Tool effectiveness
        report.append("## Tool Effectiveness")
        tool_stats = self.get_tool_effectiveness()
        sorted_tools = sorted(tool_stats.items(), key=lambda x: x[1]["effectiveness"], reverse=True)
        for tool, stats in sorted_tools[:10]:
            report.append(f"- {tool}: {stats['effectiveness']:.0%} effective ({stats['total']} uses)")
        report.append("")
        
        # Suggested improvements
        report.append("## Suggested Improvements")
        improvements = self.generate_self_improvement_tasks()
        for imp in improvements:
            report.append(f"- [{imp['priority']}] {imp['title']}")
        
        return "\n".join(report)
    
    # =========================================================================
    # Continuous Improvement Loop
    # =========================================================================
    
    def run_improvement_cycle(self) -> dict:
        """Run a full improvement cycle and return results."""
        results = {
            "timestamp": datetime.now().isoformat(),
            "insights_generated": 0,
            "optimizations_created": 0,
            "suggestions": [],
        }
        
        # Analyze all outcomes
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM outcomes ORDER BY timestamp DESC LIMIT 500")
        outcomes = cursor.fetchall()
        conn.close()
        
        if outcomes:
            self._analyze_success_patterns(outcomes)
            self._analyze_failure_patterns(outcomes)
            self._analyze_timing_patterns(outcomes)
        
        # Count new insights
        insights = self.get_insights()
        results["insights_generated"] = len(insights)
        
        # Generate suggestions
        results["suggestions"] = self.get_actionable_insights()
        
        # Role adjustment suggestions
        results["role_adjustments"] = self.suggest_agent_role_adjustments()
        
        # Self-improvement tasks
        results["improvement_tasks"] = self.generate_self_improvement_tasks()
        
        return results
