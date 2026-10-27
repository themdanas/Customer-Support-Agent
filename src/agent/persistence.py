from __future__ import annotations
import os
from typing import Optional
from psycopg_pool import ConnectionPool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres import PostgresSaver

_POOL: Optional[ConnectionPool] = None
_CHECKPOINTER: Optional[BaseCheckpointSaver] = None

def get_connection_pool(database_url: Optional[str] = None) -> ConnectionPool:
    """Returns or creates a shared ConnectionPool for PostgreSQL checkpointing."""
    global _POOL
    if _POOL is None or _POOL.closed:
        url = database_url or os.environ.get(
            "DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/support_agent"
        )
        _POOL = ConnectionPool(
            conninfo=url,
            max_size=10,
            open=True,
            kwargs={"autocommit": True},
        )
    return _POOL

def close_connection_pool() -> None:
    """Closes the shared ConnectionPool if open."""
    global _POOL, _CHECKPOINTER
    if _POOL and not _POOL.closed:
        _POOL.close()
    _POOL = None
    _CHECKPOINTER = None

def get_checkpointer(
    use_postgres: bool = True,
    database_url: Optional[str] = None,
) -> BaseCheckpointSaver:
    """
    Returns an initialized LangGraph checkpointer.
    
    If use_postgres is True and database is reachable, returns an initialized
    PostgresSaver. Falls back to MemorySaver if disabled or if connection fails.
    """
    global _CHECKPOINTER
    if _CHECKPOINTER is not None:
        return _CHECKPOINTER

    if use_postgres:
        try:
            pool = get_connection_pool(database_url)
            saver = PostgresSaver(pool)
            saver.setup()
            _CHECKPOINTER = saver
            return _CHECKPOINTER
        except Exception:
            # Fall back to MemorySaver for resilient execution (e.g. offline testing)
            pass

    _CHECKPOINTER = MemorySaver()
    return _CHECKPOINTER

def get_thread_config(thread_id: str) -> dict:
    """Standardizes LangGraph thread configuration dictionary."""
    return {"configurable": {"thread_id": str(thread_id)}}
