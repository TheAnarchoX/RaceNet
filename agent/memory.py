"""
Repository Knowledge and Memory System

A shared knowledge base that enables agents to work as a hive mind.
Agents can store, retrieve, and synchronize knowledge about the codebase.
"""

import json
import hashlib
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional
import sqlite3


class KnowledgeType(Enum):
    """Types of knowledge that can be stored."""
    CODE_FACT = "code_fact"           # Facts about code structure/behavior
    PATTERN = "pattern"                # Code patterns and conventions
    TASK_LEARNING = "task_learning"   # What worked/didn't work on tasks
    DEPENDENCY = "dependency"          # Module/file dependencies
    BUG_FIX = "bug_fix"               # Bug fixes and their solutions
    OPTIMIZATION = "optimization"      # Performance optimizations discovered
    TEST_RESULT = "test_result"       # Test results and coverage
    AGENT_INSIGHT = "agent_insight"   # Agent observations and insights
    CONVENTION = "convention"          # Coding conventions discovered
    TODO = "todo"                      # Things to do later


class Importance(Enum):
    """Importance level of knowledge."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class KnowledgeEntry:
    """A single piece of knowledge."""
    id: str
    type: KnowledgeType
    content: str
    context: dict = field(default_factory=dict)
    importance: Importance = Importance.MEDIUM
    source_agent: str = ""
    source_file: str = ""
    source_task: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    references: list[str] = field(default_factory=list)  # Related knowledge IDs
    tags: list[str] = field(default_factory=list)
    access_count: int = 0
    usefulness_score: float = 0.0  # Updated based on agent feedback
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        data = asdict(self)
        data["type"] = self.type.value
        data["importance"] = self.importance.value
        return data
    
    @classmethod
    def from_dict(cls, data: dict) -> "KnowledgeEntry":
        """Create from dictionary."""
        data["type"] = KnowledgeType(data["type"])
        data["importance"] = Importance(data["importance"])
        return cls(**data)


@dataclass
class AgentState:
    """State of an agent for coordination."""
    agent_id: str
    status: str = "idle"  # idle, working, waiting, completed
    current_task: str = ""
    current_file: str = ""
    last_heartbeat: str = field(default_factory=lambda: datetime.now().isoformat())
    knowledge_contributed: int = 0
    tasks_completed: int = 0
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> "AgentState":
        return cls(**data)


class RepositoryMemory:
    """
    Shared memory system for multi-agent coordination.
    
    Provides:
    - Persistent knowledge storage (SQLite)
    - Thread-safe access for concurrent agents
    - Knowledge categorization and retrieval
    - Agent state coordination
    - File locking to prevent conflicts
    """
    
    def __init__(self, repo_root: Path, db_name: str = ".racenet_memory.db"):
        self.repo_root = repo_root
        self.db_path = repo_root / db_name
        self._lock = threading.RLock()
        self._file_locks: dict[str, str] = {}  # file -> agent_id
        self._init_database()
    
    def _init_database(self):
        """Initialize the SQLite database."""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Knowledge table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS knowledge (
                    id TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    context TEXT,
                    importance INTEGER,
                    source_agent TEXT,
                    source_file TEXT,
                    source_task TEXT,
                    created_at TEXT,
                    updated_at TEXT,
                    related_refs TEXT,
                    tags TEXT,
                    access_count INTEGER DEFAULT 0,
                    usefulness_score REAL DEFAULT 0.0
                )
            """)
            
            # Agent state table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS agent_states (
                    agent_id TEXT PRIMARY KEY,
                    status TEXT,
                    current_task TEXT,
                    current_file TEXT,
                    last_heartbeat TEXT,
                    knowledge_contributed INTEGER DEFAULT 0,
                    tasks_completed INTEGER DEFAULT 0
                )
            """)
            
            # File locks table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS file_locks (
                    file_path TEXT PRIMARY KEY,
                    agent_id TEXT,
                    locked_at TEXT
                )
            """)
            
            # Messages table for agent communication
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    from_agent TEXT,
                    to_agent TEXT,
                    message_type TEXT,
                    content TEXT,
                    created_at TEXT,
                    read INTEGER DEFAULT 0
                )
            """)
            
            # Indexes for faster queries
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_type ON knowledge(type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_tags ON knowledge(tags)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_to ON messages(to_agent, read)")
            
            conn.commit()
            conn.close()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection."""
        return sqlite3.connect(str(self.db_path), timeout=30)
    
    # =========================================================================
    # Knowledge Management
    # =========================================================================
    
    def store_knowledge(self, entry: KnowledgeEntry) -> str:
        """Store a piece of knowledge."""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Generate ID if not provided
            if not entry.id:
                content_hash = hashlib.sha256(
                    f"{entry.type.value}:{entry.content}".encode()
                ).hexdigest()[:12]
                entry.id = f"k_{content_hash}"
            
            cursor.execute("""
                INSERT OR REPLACE INTO knowledge 
                (id, type, content, context, importance, source_agent, source_file,
                 source_task, created_at, updated_at, related_refs, tags, 
                 access_count, usefulness_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entry.id,
                entry.type.value,
                entry.content,
                json.dumps(entry.context),
                entry.importance.value,
                entry.source_agent,
                entry.source_file,
                entry.source_task,
                entry.created_at,
                entry.updated_at,
                json.dumps(entry.references),
                json.dumps(entry.tags),
                entry.access_count,
                entry.usefulness_score,
            ))
            
            conn.commit()
            conn.close()
            
            return entry.id
    
    def get_knowledge(self, knowledge_id: str) -> Optional[KnowledgeEntry]:
        """Retrieve a specific piece of knowledge."""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM knowledge WHERE id = ?", (knowledge_id,))
            row = cursor.fetchone()
            
            if row:
                # Update access count
                cursor.execute(
                    "UPDATE knowledge SET access_count = access_count + 1 WHERE id = ?",
                    (knowledge_id,)
                )
                conn.commit()
            
            conn.close()
            
            if row:
                return self._row_to_entry(row)
            return None
    
    def _row_to_entry(self, row) -> KnowledgeEntry:
        """Convert a database row to KnowledgeEntry."""
        return KnowledgeEntry(
            id=row[0],
            type=KnowledgeType(row[1]),
            content=row[2],
            context=json.loads(row[3]) if row[3] else {},
            importance=Importance(row[4]),
            source_agent=row[5] or "",
            source_file=row[6] or "",
            source_task=row[7] or "",
            created_at=row[8] or "",
            updated_at=row[9] or "",
            references=json.loads(row[10]) if row[10] else [],
            tags=json.loads(row[11]) if row[11] else [],
            access_count=row[12] or 0,
            usefulness_score=row[13] or 0.0,
        )
    
    def search_knowledge(
        self,
        query: Optional[str] = None,
        knowledge_type: Optional[KnowledgeType] = None,
        tags: Optional[list[str]] = None,
        min_importance: Optional[Importance] = None,
        source_file: Optional[str] = None,
        limit: int = 50
    ) -> list[KnowledgeEntry]:
        """Search for relevant knowledge."""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            sql = "SELECT * FROM knowledge WHERE 1=1"
            params = []
            
            if knowledge_type:
                sql += " AND type = ?"
                params.append(knowledge_type.value)
            
            if min_importance:
                sql += " AND importance >= ?"
                params.append(min_importance.value)
            
            if source_file:
                sql += " AND source_file LIKE ?"
                params.append(f"%{source_file}%")
            
            if query:
                sql += " AND content LIKE ?"
                params.append(f"%{query}%")
            
            if tags:
                for tag in tags:
                    sql += " AND tags LIKE ?"
                    params.append(f"%{tag}%")
            
            sql += " ORDER BY importance DESC, usefulness_score DESC, access_count DESC"
            sql += f" LIMIT {limit}"
            
            cursor.execute(sql, params)
            rows = cursor.fetchall()
            conn.close()
            
            return [self._row_to_entry(row) for row in rows]
    
    def get_knowledge_for_file(self, file_path: str) -> list[KnowledgeEntry]:
        """Get all knowledge related to a specific file."""
        return self.search_knowledge(source_file=file_path)
    
    def get_knowledge_for_task(self, task_id: str) -> list[KnowledgeEntry]:
        """Get all knowledge related to a specific task."""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT * FROM knowledge WHERE source_task = ? ORDER BY importance DESC",
                (task_id,)
            )
            rows = cursor.fetchall()
            conn.close()
            
            return [self._row_to_entry(row) for row in rows]
    
    def update_usefulness(self, knowledge_id: str, delta: float):
        """Update the usefulness score of knowledge (positive or negative)."""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute(
                "UPDATE knowledge SET usefulness_score = usefulness_score + ? WHERE id = ?",
                (delta, knowledge_id)
            )
            
            conn.commit()
            conn.close()
    
    def get_most_useful_knowledge(self, limit: int = 20) -> list[KnowledgeEntry]:
        """Get the most useful knowledge entries."""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute(
                """SELECT * FROM knowledge 
                   ORDER BY usefulness_score DESC, access_count DESC 
                   LIMIT ?""",
                (limit,)
            )
            rows = cursor.fetchall()
            conn.close()
            
            return [self._row_to_entry(row) for row in rows]
    
    # =========================================================================
    # Agent Coordination
    # =========================================================================
    
    def register_agent(self, agent_id: str) -> AgentState:
        """Register an agent in the system."""
        with self._lock:
            state = AgentState(agent_id=agent_id)
            
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO agent_states 
                (agent_id, status, current_task, current_file, last_heartbeat,
                 knowledge_contributed, tasks_completed)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                state.agent_id,
                state.status,
                state.current_task,
                state.current_file,
                state.last_heartbeat,
                state.knowledge_contributed,
                state.tasks_completed,
            ))
            
            conn.commit()
            conn.close()
            
            return state
    
    def update_agent_state(self, agent_id: str, **updates):
        """Update an agent's state."""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            updates["last_heartbeat"] = datetime.now().isoformat()
            
            set_clause = ", ".join(f"{k} = ?" for k in updates.keys())
            values = list(updates.values()) + [agent_id]
            
            cursor.execute(
                f"UPDATE agent_states SET {set_clause} WHERE agent_id = ?",
                values
            )
            
            conn.commit()
            conn.close()
    
    def get_agent_state(self, agent_id: str) -> Optional[AgentState]:
        """Get an agent's current state."""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM agent_states WHERE agent_id = ?", (agent_id,))
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return AgentState(
                    agent_id=row[0],
                    status=row[1],
                    current_task=row[2],
                    current_file=row[3],
                    last_heartbeat=row[4],
                    knowledge_contributed=row[5],
                    tasks_completed=row[6],
                )
            return None
    
    def get_all_agents(self) -> list[AgentState]:
        """Get all registered agents."""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM agent_states")
            rows = cursor.fetchall()
            conn.close()
            
            return [
                AgentState(
                    agent_id=row[0],
                    status=row[1],
                    current_task=row[2],
                    current_file=row[3],
                    last_heartbeat=row[4],
                    knowledge_contributed=row[5],
                    tasks_completed=row[6],
                )
                for row in rows
            ]
    
    def get_active_agents(self, timeout_seconds: int = 60) -> list[AgentState]:
        """Get agents that have sent a heartbeat recently."""
        cutoff = datetime.now().timestamp() - timeout_seconds
        agents = self.get_all_agents()
        
        active = []
        for agent in agents:
            try:
                heartbeat = datetime.fromisoformat(agent.last_heartbeat).timestamp()
                if heartbeat > cutoff:
                    active.append(agent)
            except (ValueError, TypeError):
                pass
        
        return active
    
    def clear_all_agents(self):
        """Clear all agent states from the database.
        
        Call this on orchestrator startup to remove stale agents from previous runs.
        """
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("DELETE FROM agent_states")
            cursor.execute("DELETE FROM file_locks")
            cursor.execute("DELETE FROM messages")
            
            conn.commit()
            conn.close()
    
    def clear_stale_agents(self, timeout_seconds: int = 120):
        """Clear agents that haven't sent a heartbeat recently."""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cutoff = datetime.now().timestamp() - timeout_seconds
            
            # Get stale agent IDs
            cursor.execute("SELECT agent_id, last_heartbeat FROM agent_states")
            stale_ids = []
            for row in cursor.fetchall():
                try:
                    heartbeat = datetime.fromisoformat(row[1]).timestamp()
                    if heartbeat < cutoff:
                        stale_ids.append(row[0])
                except (ValueError, TypeError):
                    stale_ids.append(row[0])
            
            # Delete stale agents
            for agent_id in stale_ids:
                cursor.execute("DELETE FROM agent_states WHERE agent_id = ?", (agent_id,))
                cursor.execute("DELETE FROM file_locks WHERE agent_id = ?", (agent_id,))
            
            conn.commit()
            conn.close()
            
            return stale_ids
    
    # =========================================================================
    # File Locking for Conflict Prevention
    # =========================================================================
    
    def acquire_file_lock(self, file_path: str, agent_id: str, timeout: int = 30) -> bool:
        """Try to acquire a lock on a file."""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Check if already locked
            cursor.execute("SELECT agent_id, locked_at FROM file_locks WHERE file_path = ?", (file_path,))
            row = cursor.fetchone()
            
            if row:
                # Check if lock is stale
                try:
                    locked_at = datetime.fromisoformat(row[1]).timestamp()
                    if time.time() - locked_at > timeout:
                        # Stale lock, can take over
                        pass
                    elif row[0] == agent_id:
                        # Already own the lock
                        conn.close()
                        return True
                    else:
                        # Someone else has it
                        conn.close()
                        return False
                except (ValueError, TypeError):
                    pass
            
            # Acquire the lock
            cursor.execute("""
                INSERT OR REPLACE INTO file_locks (file_path, agent_id, locked_at)
                VALUES (?, ?, ?)
            """, (file_path, agent_id, datetime.now().isoformat()))
            
            conn.commit()
            conn.close()
            
            return True
    
    def release_file_lock(self, file_path: str, agent_id: str):
        """Release a file lock."""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute(
                "DELETE FROM file_locks WHERE file_path = ? AND agent_id = ?",
                (file_path, agent_id)
            )
            
            conn.commit()
            conn.close()
    
    def get_locked_files(self) -> dict[str, str]:
        """Get all currently locked files."""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("SELECT file_path, agent_id FROM file_locks")
            rows = cursor.fetchall()
            conn.close()
            
            return {row[0]: row[1] for row in rows}
    
    # =========================================================================
    # Agent Messaging
    # =========================================================================
    
    def send_message(
        self,
        from_agent: str,
        to_agent: str,
        message_type: str,
        content: str
    ):
        """Send a message to another agent (or broadcast with to_agent='*')."""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO messages (from_agent, to_agent, message_type, content, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (from_agent, to_agent, message_type, content, datetime.now().isoformat()))
            
            conn.commit()
            conn.close()
    
    def get_messages(self, agent_id: str, unread_only: bool = True) -> list[dict]:
        """Get messages for an agent."""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            sql = """
                SELECT id, from_agent, message_type, content, created_at 
                FROM messages 
                WHERE (to_agent = ? OR to_agent = '*')
            """
            params = [agent_id]
            
            if unread_only:
                sql += " AND read = 0"
            
            sql += " ORDER BY created_at DESC"
            
            cursor.execute(sql, params)
            rows = cursor.fetchall()
            
            # Mark as read
            if unread_only and rows:
                ids = [row[0] for row in rows]
                cursor.execute(
                    f"UPDATE messages SET read = 1 WHERE id IN ({','.join('?' * len(ids))})",
                    ids
                )
                conn.commit()
            
            conn.close()
            
            return [
                {
                    "id": row[0],
                    "from_agent": row[1],
                    "message_type": row[2],
                    "content": row[3],
                    "created_at": row[4],
                }
                for row in rows
            ]
    
    def broadcast(self, from_agent: str, message_type: str, content: str):
        """Broadcast a message to all agents."""
        self.send_message(from_agent, "*", message_type, content)
    
    # =========================================================================
    # Knowledge Synthesis
    # =========================================================================
    
    def get_context_for_task(self, task_id: str, file_paths: list[str]) -> str:
        """Get synthesized context for working on a task."""
        context_parts = []
        
        # Get task-specific knowledge
        task_knowledge = self.get_knowledge_for_task(task_id)
        if task_knowledge:
            context_parts.append("## Previous Knowledge About This Task")
            for k in task_knowledge[:5]:
                context_parts.append(f"- [{k.type.value}] {k.content}")
        
        # Get file-specific knowledge
        for file_path in file_paths[:5]:
            file_knowledge = self.get_knowledge_for_file(file_path)
            if file_knowledge:
                context_parts.append(f"\n## Knowledge About {file_path}")
                for k in file_knowledge[:3]:
                    context_parts.append(f"- [{k.type.value}] {k.content}")
        
        # Get most useful general knowledge
        useful = self.get_most_useful_knowledge(10)
        if useful:
            context_parts.append("\n## Most Useful General Knowledge")
            for k in useful[:5]:
                context_parts.append(f"- [{k.type.value}] {k.content}")
        
        return "\n".join(context_parts)
    
    def get_summary(self) -> dict:
        """Get a summary of the knowledge base."""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Count by type
            cursor.execute("""
                SELECT type, COUNT(*) FROM knowledge GROUP BY type
            """)
            type_counts = dict(cursor.fetchall())
            
            # Total count
            cursor.execute("SELECT COUNT(*) FROM knowledge")
            total = cursor.fetchone()[0]
            
            # Active agents
            agents = self.get_active_agents()
            
            conn.close()
            
            return {
                "total_knowledge": total,
                "by_type": type_counts,
                "active_agents": len(agents),
                "agent_ids": [a.agent_id for a in agents],
            }
