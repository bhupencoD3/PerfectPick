import json
from collections import deque
import psycopg2
from psycopg2.extras import Json
from utils.logger import get_logger
import atexit  # auto-close connection

logger = get_logger(__name__)

class SessionMemoryDB:
    def __init__(self, db_url, max_load=10):
        """
        Initialize session memory database with PostgreSQL using a persistent connection.
        """
        self.db_url = db_url
        self.max_load = max_load
        try:
            self.client = psycopg2.connect(self.db_url)  # persistent connection
            self._init_db()
            logger.info("✅ Connected to PostgreSQL and initialized session memory table")
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise

        # Auto-close connection at exit
        atexit.register(self.close)

    def _init_db(self):
        """Initialize PostgreSQL table for session memory."""
        try:
            c = self.client.cursor()
            c.execute("""
                CREATE TABLE IF NOT EXISTS session_memory (
                    session_id TEXT PRIMARY KEY,
                    memory JSONB
                )
            """)
            self.client.commit()
        except Exception as e:
            logger.error(f"Failed to initialize DB: {e}")
            raise

    def load_memory(self, session_id):
        """Load conversation history for a session_id."""
        try:
            c = self.client.cursor()
            c.execute("SELECT memory FROM session_memory WHERE session_id = %s", (session_id,))
            row = c.fetchone()
            if row:
                mem_list = row[0]  # JSONB returns as Python list
                if self.max_load is not None:
                    mem_list = mem_list[-self.max_load:]
                return deque(mem_list, maxlen=self.max_load if self.max_load is not None else None)
            return deque(maxlen=self.max_load if self.max_load is not None else None)
        except Exception as e:
            logger.error(f"Failed to load memory for {session_id}: {e}")
            return deque(maxlen=self.max_load if self.max_load is not None else None)

    def save_memory(self, session_id, memory_deque):
        """Save conversation history for a session_id."""
        mem_list = list(memory_deque)
        try:
            c = self.client.cursor()
            c.execute("""
                INSERT INTO session_memory (session_id, memory)
                VALUES (%s, %s)
                ON CONFLICT (session_id) DO UPDATE
                SET memory = EXCLUDED.memory
            """, (session_id, Json(mem_list)))
            self.client.commit()
            logger.debug(f"Saved memory for session {session_id}")
        except Exception as e:
            logger.error(f"Failed to save memory for {session_id}: {e}")
            raise

    def close(self):
        """Close persistent connection."""
        try:
            if self.client:
                self.client.close()
                logger.info("✅ PostgreSQL connection closed")
        except Exception as e:
            logger.warning(f"Failed to close PostgreSQL connection: {e}")
