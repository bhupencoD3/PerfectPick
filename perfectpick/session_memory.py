import json
from collections import deque
import psycopg2
from psycopg2.extras import Json
from psycopg2.pool import SimpleConnectionPool
from utils.logger import get_logger
import atexit
import time
import socket
from urllib.parse import urlparse
import os
import threading

logger = get_logger(__name__)

class SessionMemoryDB:
    def __init__(self, db_url, max_load=10, max_retries=3, retry_delay=2, pool_size=5):
        self.db_url = db_url
        self.max_load = max_load
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.pool = None
        self._lock = threading.Lock()
        
        self._init_pool()
        self._init_db()
        atexit.register(self.close_pool)

    def _init_pool(self):
        """Initialize connection pool"""
        try:
            parsed = urlparse(self.db_url)
            
            # Connection parameters for pool
            conn_params = {
                'dbname': parsed.path[1:] or 'postgres',
                'user': parsed.username,
                'password': parsed.password,
                'host': parsed.hostname,
                'port': parsed.port or 5432,
                'connect_timeout': 10,
                'sslmode': 'require'
            }
            
            self.pool = SimpleConnectionPool(
                minconn=1,
                maxconn=5,  # Reduced for Supabase limits
                **conn_params
            )
            logger.info("✅ PostgreSQL connection pool initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize connection pool: {e}")
            raise

    def _get_connection(self):
        """Get connection from pool with retry logic"""
        for attempt in range(1, self.max_retries + 1):
            try:
                conn = self.pool.getconn()
                # Test connection
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                cursor.close()
                return conn
            except (psycopg2.InterfaceError, psycopg2.OperationalError) as e:
                logger.warning(f"Connection attempt {attempt} failed: {e}")
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * attempt)
                else:
                    raise e
            except Exception as e:
                logger.error(f"Unexpected error getting connection: {e}")
                raise

    def _return_connection(self, conn):
        """Return connection to pool"""
        try:
            if conn and not conn.closed:
                self.pool.putconn(conn)
        except Exception as e:
            logger.warning(f"Failed to return connection to pool: {e}")

    def _execute_with_connection(self, operation, params=None):
        """Execute operation with managed connection"""
        conn = None
        try:
            conn = self._get_connection()
            with conn.cursor() as cursor:
                if params:
                    cursor.execute(operation, params)
                else:
                    cursor.execute(operation)
                conn.commit()
                
                # For SELECT operations, return results
                if operation.strip().upper().startswith('SELECT'):
                    return cursor.fetchall()
                return True
                
        except (psycopg2.InterfaceError, psycopg2.OperationalError) as e:
            logger.error(f"Database operation failed: {e}")
            if conn:
                try:
                    conn.rollback()
                except:
                    pass
            raise
        except Exception as e:
            logger.error(f"Unexpected database error: {e}")
            if conn:
                try:
                    conn.rollback()
                except:
                    pass
            raise
        finally:
            if conn:
                self._return_connection(conn)

    def _init_db(self):
        """Initialize PostgreSQL table for session memory"""
        try:
            self._execute_with_connection("""
                CREATE TABLE IF NOT EXISTS session_memory (
                    session_id TEXT PRIMARY KEY,
                    memory JSONB,
                    updated_at TIMESTAMP DEFAULT NOW()
                )
            """)
            logger.info("✅ Session memory table initialized")
        except Exception as e:
            logger.error(f"Failed to initialize DB: {e}")
            # Don't raise, continue without table

    def load_memory(self, session_id):
        """Load conversation history for a session_id"""
        try:
            results = self._execute_with_connection(
                "SELECT memory FROM session_memory WHERE session_id = %s", 
                (session_id,)
            )
            
            if results and results[0] and results[0][0]:
                mem_list = results[0][0]
                if self.max_load is not None:
                    mem_list = mem_list[-self.max_load:]
                logger.debug(f"Loaded {len(mem_list)} memory entries for session {session_id}")
                return deque(mem_list, maxlen=self.max_load)
            logger.debug(f"No memory found for session {session_id}")
            return deque(maxlen=self.max_load)
            
        except Exception as e:
            logger.error(f"Failed to load memory for {session_id}: {e}")
            return deque(maxlen=self.max_load)

    def save_memory(self, session_id, memory_deque):
        """Save conversation history for a session_id"""
        if not memory_deque:
            logger.debug(f"No memory to save for session {session_id}")
            return False
            
        mem_list = list(memory_deque)
        try:
            success = self._execute_with_connection("""
                INSERT INTO session_memory (session_id, memory, updated_at)
                VALUES (%s, %s, NOW())
                ON CONFLICT (session_id) DO UPDATE
                SET memory = EXCLUDED.memory, updated_at = NOW()
            """, (session_id, Json(mem_list)))
            
            if success:
                logger.debug(f"Saved {len(mem_list)} memory entries for session {session_id}")
            else:
                logger.warning(f"Memory save returned False for session {session_id}")
            return success
            
        except Exception as e:
            logger.error(f"Failed to save memory for {session_id}: {e}")
            return False

    def close_pool(self):
        """Close connection pool"""
        try:
            if self.pool:
                self.pool.closeall()
                logger.info("✅ PostgreSQL connection pool closed")
        except Exception as e:
            logger.warning(f"Failed to close connection pool: {e}")