import os
import psycopg2
from psycopg2 import pool, sql
from psycopg2.extras import RealDictCursor
import logging
from typing import Optional, Dict, List, Any
from datetime import datetime

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PostgresDB:
    def __init__(self):
        self.pool = None
        self.init_pool()

    def init_pool(self):
        """Инициализация пула соединений с PostgreSQL"""
        try:
            self.pool = psycopg2.pool.SimpleConnectionPool(
                minconn=1,
                maxconn=10,
                host=os.getenv('DB_HOST'),
                port=os.getenv('DB_PORT', '5432'),
                user=os.getenv('DB_USER'),
                password=os.getenv('DB_PASSWORD'),
                database=os.getenv('DB_NAME'),
                cursor_factory=RealDictCursor,
                connect_timeout=5
            )
            logger.info("PostgreSQL connection pool initialized")
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL pool: {str(e)}")
            raise

    def get_conn(self):
        """Получение соединения из пула"""
        try:
            return self.pool.getconn()
        except Exception as e:
            logger.error(f"Error getting connection: {str(e)}")
            raise

    def release_conn(self, conn):
        """Возврат соединения в пул"""
        try:
            if conn:
                self.pool.putconn(conn)
        except Exception as e:
            logger.error(f"Error releasing connection: {str(e)}")
            try:
                conn.close()
            except:
                pass

    def initialize_schema(self):
        """Инициализация схемы базы данных"""
        conn = None
        try:
            conn = self.get_conn()
            with conn.cursor() as cur:
                # Создаем схему для организации таблиц
                cur.execute("CREATE SCHEMA IF NOT EXISTS media_credibility")

                # Таблица новостей
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
                        url TEXT UNIQUE,
                        analysis_date TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                        short_summary TEXT,
                        is_buzz_article BOOLEAN DEFAULT FALSE,
                        buzz_date DATE
                    )
                """)

                # Таблица статистики источников
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS media_credibility.source_stats (
                        source TEXT PRIMARY KEY,
                        high INTEGER DEFAULT 0,
                        medium INTEGER DEFAULT 0,
                        low INTEGER DEFAULT 0,
                        total_analyzed INTEGER DEFAULT 0
                    )
                """)

                # Таблица обратной связи
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS media_credibility.feedback (
                        id SERIAL PRIMARY KEY,
                        name TEXT NOT NULL,
                        email TEXT NOT NULL,
                        type TEXT NOT NULL,
                        message TEXT NOT NULL,
                        date TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # Таблица ежедневных статей
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS media_credibility.daily_buzz (
                        id SERIAL PRIMARY KEY,
                        article_id INTEGER REFERENCES media_credibility.news(id) ON DELETE CASCADE,
                        date DATE UNIQUE
                    )
                """)

                # Таблица голосов за статьи
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS media_credibility.article_votes (
                        id SERIAL PRIMARY KEY,
                        article_id INTEGER REFERENCES media_credibility.news(id) ON DELETE CASCADE,
                        user_id TEXT NOT NULL,
                        vote_type TEXT CHECK (vote_type IN ('upvote', 'downvote')),
                        created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(article_id, user_id)
                    )
                """)

                # Таблица оценок достоверности
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS media_credibility.credibility_votes (
                        id SERIAL PRIMARY KEY,
                        article_id INTEGER REFERENCES media_credibility.news(id) ON DELETE CASCADE,
                        user_id TEXT NOT NULL,
                        credibility_rating INTEGER CHECK (credibility_rating BETWEEN 1 AND 5),
                        created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(article_id, user_id)
                    )
                """)

                # Таблица ленты новостей
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS media_credibility.news_feed (
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
                    )
                """)

                # Таблица категорий новостей
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS media_credibility.news_categories (
                        id SERIAL PRIMARY KEY,
                        name TEXT UNIQUE NOT NULL,
                        description TEXT
                    )
                """)

                # Таблица предпочтений пользователей
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS media_credibility.user_preferences (
                        id SERIAL PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        preferred_categories TEXT,
                        preferred_sources TEXT,
                        last_updated TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # Индексы для ускорения запросов
                cur.execute("CREATE INDEX IF NOT EXISTS idx_news_url ON media_credibility.news(url)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_news_buzz_date ON media_credibility.news(buzz_date) WHERE is_buzz_article = TRUE")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_article_votes ON media_credibility.article_votes(article_id)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_credibility_votes ON media_credibility.credibility_votes(article_id)")

                conn.commit()
                logger.info("PostgreSQL schema initialized successfully")
        except Exception as e:
            logger.error(f"Error creating schema: {str(e)}")
            if conn:
                conn.rollback()
            raise
        finally:
            self.release_conn(conn)

    def save_article(self, title: str, source: str, content: str, url: Optional[str] = None,
                   is_buzz_article: bool = False, buzz_date: Optional[datetime] = None) -> int:
        """Сохранение статьи в базу данных"""
        conn = None
        try:
            conn = self.get_conn()
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO media_credibility.news
                    (title, source, content, url, is_buzz_article, buzz_date, short_summary)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (url) DO UPDATE SET
                    title = EXCLUDED.title,
                    source = EXCLUDED.source,
                    content = EXCLUDED.content,
                    short_summary = EXCLUDED.short_summary,
                    is_buzz_article = EXCLUDED.is_buzz_article,
                    buzz_date = EXCLUDED.buzz_date
                    RETURNING id
                """, (
                    title,
                    source,
                    content,
                    url,
                    is_buzz_article,
                    buzz_date,
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
            self.release_conn(conn)

    def save_analysis(self, article_id: int, analysis_data: Dict, user_id: Optional[str] = None) -> None:
        """Сохранение анализа статьи"""
        conn = None
        try:
            conn = self.get_conn()
            with conn.cursor() as cur:
                # Обновляем статью с данными анализа
                cur.execute("""
                    UPDATE media_credibility.news
                    SET
                        integrity = %s,
                        fact_check = %s,
                        sentiment = %s,
                        bias = %s,
                        credibility_level = %s,
                        index_of_credibility = %s,
                        short_summary = %s
                    WHERE id = %s
                """, (
                    analysis_data.get('news_integrity', 0.0),
                    analysis_data.get('fact_check_needed_score', 0.0),
                    analysis_data.get('sentiment_score', 0.0),
                    analysis_data.get('bias_score', 0.0),
                    analysis_data.get('credibility_level', 'Medium'),
                    analysis_data.get('index_of_credibility', 0.0),
                    analysis_data.get('short_summary', 'No summary available'),
                    article_id
                ))

                # Обновляем статистику источника
                credibility_level = analysis_data.get('credibility_level', 'Medium')
                cur.execute("""
                    INSERT INTO media_credibility.source_stats (source, high, medium, low, total_analyzed)
                    VALUES (%s, %s, %s, %s, 1)
                    ON CONFLICT (source) DO UPDATE SET
                        high = CASE WHEN %s = 'High' THEN media_credibility.source_stats.high + 1 ELSE media_credibility.source_stats.high END,
                        medium = CASE WHEN %s = 'Medium' THEN media_credibility.source_stats.medium + 1 ELSE media_credibility.source_stats.medium END,
                        low = CASE WHEN %s = 'Low' THEN media_credibility.source_stats.low + 1 ELSE media_credibility.source_stats.low END,
                        total_analyzed = media_credibility.source_stats.total_analyzed + 1
                """, (
                    analysis_data.get('source', 'Unknown'),
                    1 if credibility_level == 'High' else 0,
                    1 if credibility_level == 'Medium' else 0,
                    1 if credibility_level == 'Low' else 0,
                    credibility_level,
                    credibility_level,
                    credibility_level
                ))

                conn.commit()
        except Exception as e:
            logger.error(f"Error saving analysis: {str(e)}")
            if conn:
                conn.rollback()
            raise
        finally:
            self.release_conn(conn)

    def get_buzz_article(self) -> Optional[Dict]:
        """Получение статьи для BuzzFeed"""
        conn = None
        try:
            conn = self.get_conn()
            with conn.cursor() as cur:
                # Получаем сегодняшнюю дату
                today = datetime.now().date()

                # Пытаемся получить сегодняшнюю статью BuzzFeed
                cur.execute("""
                    SELECT n.*, s.high, s.medium, s.low, s.total_analyzed
                    FROM media_credibility.news n
                    LEFT JOIN media_credibility.source_stats s ON n.source = s.source
                    WHERE n.is_buzz_article = TRUE AND n.buzz_date = %s
                    ORDER BY n.analysis_date DESC
                    LIMIT 1
                """, (today,))
                article = cur.fetchone()

                if not article:
                    # Если статья не найдена, получаем последнюю статью о конфликте Израиль-Иран
                    cur.execute("""
                        SELECT n.*, s.high, s.medium, s.low, s.total_analyzed
                        FROM media_credibility.news n
                        LEFT JOIN media_credibility.source_stats s ON n.source = s.source
                        WHERE n.content LIKE '%Israel%' OR n.content LIKE '%Iran%'
                        ORDER BY n.analysis_date DESC
                        LIMIT 1
                    """)
                    article = cur.fetchone()

                    if article:
                        # Обновляем статью как сегодняшнюю статью BuzzFeed
                        cur.execute("""
                            UPDATE media_credibility.news
                            SET is_buzz_article = TRUE, buzz_date = %s
                            WHERE id = %s
                        """, (today, article['id']))
                        conn.commit()

                return article
        except Exception as e:
            logger.error(f"Error getting buzz article: {str(e)}")
            return None
        finally:
            self.release_conn(conn)

    def close_all_connections(self):
        """Закрытие всех соединений в пуле"""
        if self.pool:
            self.pool.closeall()
            logger.info("All database connections closed")

# Создаем экземпляр базы данных
pg_db = PostgresDB()
