import os
import psycopg2
import logging
from psycopg2.extras import RealDictCursor
from psycopg2 import pool

# Настройка логов
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PostgresDB:
    def __init__(self):
        self.pool = None
        self.init_pool()

    def init_pool(self):
        try:
            self.pool = psycopg2.pool.SimpleConnectionPool(
                minconn=1,
                maxconn=10,
                host=os.getenv('DB_HOST'),
                port=os.getenv('DB_PORT', '5432'),
                user=os.getenv('DB_USER'),
                password=os.getenv('DB_PASSWORD'),
                database=os.getenv('DB_NAME'),
                cursor_factory=RealDictCursor
            )
            logger.info("PostgreSQL connection pool initialized")
        except Exception as e:
            logger.exception("Failed to initialize PostgreSQL pool")
            raise

    def get_conn(self):
        return self.pool.getconn()

    def release_conn(self, conn):
        if conn:
            self.pool.putconn(conn)

    def initialize_schema(self):
        conn = self.get_conn()
        try:
            with conn.cursor() as cur:
                # Таблицы анализа
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS news (
                        id SERIAL PRIMARY KEY,
                        title TEXT,
                        source TEXT,
                        content TEXT,
                        integrity REAL,
                        fact_check REAL,
                        sentiment REAL,
                        bias REAL,
                        credibility_level TEXT,
                        index_of_credibility REAL,
                        url TEXT UNIQUE,
                        analysis_date TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                        short_summary TEXT
                    );
                """)

                cur.execute("""
                    CREATE TABLE IF NOT EXISTS source_stats (
                        source TEXT PRIMARY KEY,
                        high INTEGER DEFAULT 0,
                        medium INTEGER DEFAULT 0,
                        low INTEGER DEFAULT 0,
                        total_analyzed INTEGER DEFAULT 0
                    );
                """)

                cur.execute("""
                    CREATE TABLE IF NOT EXISTS feedback (
                        id SERIAL PRIMARY KEY,
                        name TEXT NOT NULL,
                        email TEXT NOT NULL,
                        type TEXT NOT NULL,
                        message TEXT NOT NULL,
                        date TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                    );
                """)

                cur.execute("""
                    CREATE TABLE IF NOT EXISTS daily_buzz (
                        id SERIAL PRIMARY KEY,
                        article_id INTEGER REFERENCES news(id) ON DELETE CASCADE,
                        date DATE UNIQUE
                    );
                """)

                cur.execute("""
                    CREATE TABLE IF NOT EXISTS article_votes (
                        id SERIAL PRIMARY KEY,
                        article_id INTEGER REFERENCES news(id) ON DELETE CASCADE,
                        user_id TEXT,
                        vote_type TEXT CHECK (vote_type IN ('upvote', 'downvote')),
                        created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                    );
                """)

                cur.execute("""
                    CREATE TABLE IF NOT EXISTS credibility_votes (
                        id SERIAL PRIMARY KEY,
                        article_id INTEGER REFERENCES news(id) ON DELETE CASCADE,
                        user_id TEXT,
                        credibility_rating INTEGER CHECK (credibility_rating BETWEEN 1 AND 5),
                        created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                    );
                """)

                # Таблицы ленты
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS news_feed (
                        id SERIAL PRIMARY KEY,
                        title TEXT NOT NULL,
                        description TEXT,
                        url TEXT UNIQUE NOT NULL,
                        source TEXT NOT NULL,
                        published_at TIMESTAMPTZ,
                        content TEXT,
                        image_url TEXT,
                        category TEXT,
                        country TEXT,
                        language TEXT,
                        created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                    );
                """)

                cur.execute("""
                    CREATE TABLE IF NOT EXISTS news_categories (
                        id SERIAL PRIMARY KEY,
                        name TEXT UNIQUE NOT NULL,
                        description TEXT
                    );
                """)

                cur.execute("""
                    CREATE TABLE IF NOT EXISTS user_preferences (
                        id SERIAL PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        preferred_categories TEXT,
                        preferred_sources TEXT,
                        last_updated TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                    );
                """)

                conn.commit()
                logger.info("PostgreSQL schema initialized")
        except Exception as e:
            conn.rollback()
            logger.exception("Error creating schema")
            raise
        finally:
            self.release_conn(conn)

