import os
import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor
import logging
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO').upper(), logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PostgresDB:
    def __init__(self):
        self.pool = None
        self.init_pool()

    def init_pool(self):
        """Initialize connection pool"""
        try:
            self.pool = psycopg2.pool.SimpleConnectionPool(
                minconn=1,
                maxconn=10,
                host=os.getenv('PGHOST'),
                database=os.getenv('POSTGRES_DB'),
                user=os.getenv('PGUSER'),
                password=os.getenv('PGPASSWORD'),
                port=os.getenv('PGPORT', '5432'),
                cursor_factory=RealDictCursor
            )
            self.initialize_schema()
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL pool: {str(e)}")
            raise

    def initialize_schema(self):
        """Initialize database schema"""
        conn = None
        try:
            conn = self.get_conn()
            with conn.cursor() as cur:
                # Create schema if not exists
                cur.execute("CREATE SCHEMA IF NOT EXISTS media_credibility")

                # Create tables
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS media_credibility.news (
                        id SERIAL PRIMARY KEY,
                        title TEXT NOT NULL,
                        source TEXT NOT NULL,
                        content TEXT NOT NULL,
                        integrity REAL,
                        fact_check REAL,
                        sentiment REAL,
                        bias REAL,
                        credibility_level TEXT,
                        index_of_credibility REAL,
                        url TEXT,
                        analysis_date TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                        short_summary TEXT,
                        is_buzz_article BOOLEAN DEFAULT FALSE
                    )
                """)

                cur.execute("""
                    CREATE TABLE IF NOT EXISTS media_credibility.analysis (
                        id SERIAL PRIMARY KEY,
                        article_id INTEGER REFERENCES media_credibility.news(id) ON DELETE CASCADE,
                        analysis_data JSONB NOT NULL,
                        analysis_type TEXT NOT NULL,
                        created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                cur.execute("""
                    CREATE TABLE IF NOT EXISTS media_credibility.source_stats (
                        source TEXT PRIMARY KEY,
                        high INTEGER DEFAULT 0,
                        medium INTEGER DEFAULT 0,
                        low INTEGER DEFAULT 0,
                        total_analyzed INTEGER DEFAULT 0
                    )
                """)

                conn.commit()
        except Exception as e:
            logger.error(f"Error creating schema: {str(e)}")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                self.release_conn(conn)

    def get_conn(self):
        """Get connection from pool"""
        try:
            return self.pool.getconn()
        except Exception as e:
            logger.error(f"Error getting connection: {str(e)}")
            raise

    def release_conn(self, conn):
        """Release connection back to pool"""
        try:
            if conn:
                self.pool.putconn(conn)
        except Exception as e:
            logger.error(f"Error releasing connection: {str(e)}")
            try:
                conn.close()
            except:
                pass

    def save_article(self, title, source, content, url=None):
        """Save article to database"""
        conn = None
        try:
            conn = self.get_conn()
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO media_credibility.news
                    (title, source, content, url, short_summary)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    title,
                    source,
                    content,
                    url,
                    content[:200] + '...' if len(content) > 200 else content
                ))
                article_id = cur.fetchone()['id']
                conn.commit()
                return article_id
        except Exception as e:
            logger.error(f"Error saving article: {str(e)}")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                self.release_conn(conn)

    def save_analysis(self, article_id, analysis_data, analysis_type='standard'):
        """Save analysis to database"""
        conn = None
        try:
            conn = self.get_conn()
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO media_credibility.analysis
                    (article_id, analysis_data, analysis_type)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (article_id, analysis_type)
                    DO UPDATE SET analysis_data = EXCLUDED.analysis_data
                """, (
                    article_id,
                    json.dumps(analysis_data),
                    analysis_type
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Error saving analysis: {str(e)}")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                self.release_conn(conn)
