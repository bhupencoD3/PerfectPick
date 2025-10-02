import json
from collections import deque
import psycopg2
from psycopg2.extras import Json
from utils.logger import get_logger
import atexit
import time
import socket
from urllib.parse import urlparse
import os

logger = get_logger(__name__)

class SessionMemoryDB:
    def __init__(self, db_url, max_load=10, max_retries=3, retry_delay=2):
        self.db_url = db_url
        self.max_load = max_load
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.client = None

        atexit.register(self.close)
        self._connect()
        self._init_db()

    def _resolve_with_public_dns(self, hostname):
        """Resolve hostname using public DNS servers."""
        dns_servers = ['8.8.8.8', '1.1.1.1', '8.8.4.4']  # Google DNS, Cloudflare DNS
        
        for dns_server in dns_servers:
            try:
                # Use nslookup via subprocess as a fallback
                import subprocess
                result = subprocess.run(
                    ['nslookup', hostname, dns_server], 
                    capture_output=True, 
                    text=True, 
                    timeout=5
                )
                if result.returncode == 0:
                    for line in result.stdout.split('\n'):
                        if 'Address:' in line and not '#' in line:
                            ip = line.split('Address:')[1].strip()
                            if ip and ip != dns_server:
                                logger.info(f"Resolved {hostname} to {ip} via {dns_server}")
                                return ip
            except:
                continue
        
        # Fallback to system DNS
        try:
            return socket.gethostbyname(hostname)
        except:
            raise Exception(f"Could not resolve {hostname}")

    def _connect_with_ip(self, parsed_url):
        """Connect using IP address instead of hostname."""
        hostname = parsed_url.hostname
        port = parsed_url.port or 5432
        
        try:
            # Try to resolve to IP
            ip_address = self._resolve_with_public_dns(hostname)
            logger.info(f"Using IP address: {ip_address}")
            
            # Build connection with IP
            conn_params = {
                'dbname': parsed_url.path[1:] or 'postgres',
                'user': parsed_url.username,
                'password': parsed_url.password,
                'host': ip_address,
                'port': port,
                'connect_timeout': 10,
                'sslmode': 'require'
            }
            
            return psycopg2.connect(**conn_params)
        except Exception as e:
            logger.error(f"IP connection failed: {e}")
            raise

    def _connect(self):
        """Connect to PostgreSQL with DNS fallback."""
        parsed = urlparse(self.db_url)
        last_error = None
        
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(f"Attempt {attempt}/{self.max_retries} connecting to PostgreSQL")
                
                # First try direct connection
                try:
                    self.client = psycopg2.connect(self.db_url, connect_timeout=10)
                except Exception as direct_error:
                    logger.warning(f"Direct connection failed, trying IP connection: {direct_error}")
                    # If direct fails, try IP connection
                    self.client = self._connect_with_ip(parsed)
                
                # Test the connection
                cursor = self.client.cursor()
                cursor.execute("SELECT 1")
                cursor.close()
                
                logger.info("✅ Connected to PostgreSQL successfully")
                return
                
            except psycopg2.OperationalError as e:
                last_error = e
                if attempt < self.max_retries:
                    wait_time = self.retry_delay * (2 ** (attempt - 1))
                    logger.warning(f"Connection failed, retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)
                else:
                    logger.error(f"❌ All connection attempts failed: {e}")
                    raise last_error
            except Exception as e:
                last_error = e
                logger.error(f"Unexpected error: {e}")
                raise

    def _init_db(self):
        """Initialize PostgreSQL table for session memory."""
        try:
            with self.client.cursor() as c:
                c.execute("""
                    CREATE TABLE IF NOT EXISTS session_memory (
                        session_id TEXT PRIMARY KEY,
                        memory JSONB,
                        updated_at TIMESTAMP DEFAULT NOW()
                    )
                """)
            self.client.commit()
            logger.info("✅ Session memory table initialized")
        except Exception as e:
            logger.error(f"Failed to initialize DB: {e}")
            self.client.rollback()
            raise

    def load_memory(self, session_id):
        """Load conversation history for a session_id."""
        try:
            with self.client.cursor() as c:
                c.execute("SELECT memory FROM session_memory WHERE session_id = %s", (session_id,))
                row = c.fetchone()
                if row:
                    mem_list = row[0] or []
                    if self.max_load is not None:
                        mem_list = mem_list[-self.max_load:]
                    return deque(mem_list, maxlen=self.max_load)
                return deque(maxlen=self.max_load)
        except Exception as e:
            logger.error(f"Failed to load memory for {session_id}: {e}")
            return deque(maxlen=self.max_load)

    def save_memory(self, session_id, memory_deque):
        """Save conversation history for a session_id."""
        mem_list = list(memory_deque)
        try:
            with self.client.cursor() as c:
                c.execute("""
                    INSERT INTO session_memory (session_id, memory, updated_at)
                    VALUES (%s, %s, NOW())
                    ON CONFLICT (session_id) DO UPDATE
                    SET memory = EXCLUDED.memory, updated_at = NOW()
                """, (session_id, Json(mem_list)))
            self.client.commit()
            logger.debug(f"Saved memory for session {session_id}")
        except Exception as e:
            logger.error(f"Failed to save memory for {session_id}: {e}")
            self.client.rollback()
            raise

    def close(self):
        """Close persistent connection."""
        try:
            if self.client and not self.client.closed:
                self.client.close()
                logger.info("✅ PostgreSQL connection closed")
        except Exception as e:
            logger.warning(f"Failed to close PostgreSQL connection: {e}")