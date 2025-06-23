import os
import sqlite3
import json
import logging
from typing import Optional, List, Dict, Any, Tuple, Union
from datetime import datetime, timedelta
from urllib.parse import urlparse
from contextlib import contextmanager
import threading

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Database:
    FEATURED_ARTICLE_TITLE = "Israel-Iran Conflict: Comprehensive Analysis"
    DATABASE_VERSION = 1
    LOCK = threading.Lock()

    def __init__(self, db_path: str = 'instance/credibility_index.db'):
        self.db_path = db_path
        self._ensure_db_exists()
        self._initialize_schema()
        self._initialize_db_version()

    def _ensure_db_exists(self) -> None:
        """Убедиться, что база данных существует"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        if not os.path.exists(self.db_path):
            logger.info(f"Creating new SQLite database at {self.db_path}")
            open(self.db_path, 'w').close()

    def _initialize_db_version(self) -> None:
        """Инициализация версии базы данных"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS db_version (
                        version INTEGER PRIMARY KEY,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                cursor.execute("SELECT version FROM db_version")
                version = cursor.fetchone()
                if not version:
                    cursor.execute("INSERT INTO db_version (version) VALUES (?)", (self.DATABASE_VERSION,))
                    conn.commit()
                elif version[0] != self.DATABASE_VERSION:
                    logger.warning(f"Database version mismatch. Current: {version[0]}, Expected: {self.DATABASE_VERSION}")
                    # Здесь можно добавить логику миграции
        except Exception as e:
            logger.error(f"Error initializing database version: {str(e)}", exc_info=True)

    @contextmanager
    def get_connection(self) -> sqlite3.Connection:
        """Контекстный менеджер для работы с соединением"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA foreign_keys = ON;")
            conn.execute("PRAGMA busy_timeout = 5000;")
            yield conn
            conn.commit()
        except Exception as e:
            logger.error(f"Database connection error: {str(e)}", exc_info=True)
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                conn.close()

    def _initialize_schema(self) -> None:
        """Инициализировать схему базы данных"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                # Создаем таблицы, если их нет
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS articles (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        title TEXT NOT NULL,
                        source TEXT NOT NULL,
                        url TEXT,
                        content TEXT NOT NULL,
                        short_summary TEXT,
                        analysis_data TEXT,
                        credibility_level TEXT CHECK(credibility_level IN ('High', 'Medium', 'Low')),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        is_featured BOOLEAN DEFAULT 0
                    )
                """)

                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS source_stats (
                        source TEXT PRIMARY KEY,
                        high INTEGER DEFAULT 0,
                        medium INTEGER DEFAULT 0,
                        low INTEGER DEFAULT 0,
                        total_analyzed INTEGER DEFAULT 0,
                        is_initial INTEGER DEFAULT 0,
                        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(source)
                    )
                """)

                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS analysis_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        article_id INTEGER,
                        analysis_data TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (article_id) REFERENCES articles(id) ON DELETE CASCADE
                    )
                """)

                # Создаем индексы для ускорения поиска
                self._create_indexes(cursor)

                # Добавляем статью дня, если её нет
                self._add_featured_article_if_not_exists(conn)

                # Инициализируем с начальными данными
                self._initialize_with_initial_data_if_needed(conn)

                logger.info("Database schema initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing database schema: {str(e)}", exc_info=True)
            raise

    def _create_indexes(self, cursor: sqlite3.Cursor) -> None:
        """Создание индексов для ускорения запросов"""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_articles_title ON articles(title)",
            "CREATE INDEX IF NOT EXISTS idx_articles_url ON articles(url)",
            "CREATE INDEX IF NOT EXISTS idx_articles_created_at ON articles(created_at)",
            "CREATE INDEX IF NOT EXISTS idx_articles_source ON articles(source)",
            "CREATE INDEX IF NOT EXISTS idx_articles_credibility ON articles(credibility_level)",
            "CREATE INDEX IF NOT EXISTS idx_articles_is_featured ON articles(is_featured)",
            "CREATE INDEX IF NOT EXISTS idx_analysis_history_article ON analysis_history(article_id)",
            "CREATE INDEX IF NOT EXISTS idx_analysis_history_created ON analysis_history(created_at)"
        ]

        for index in indexes:
            try:
                cursor.execute(index)
            except sqlite3.OperationalError as e:
                logger.warning(f"Index creation warning: {str(e)}")

    def _add_featured_article_if_not_exists(self, conn: sqlite3.Connection) -> None:
        """Добавить статью дня в базу данных, если её нет"""
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM articles WHERE title = ?", (self.FEATURED_ARTICLE_TITLE,))
        if cursor.fetchone()[0] == 0:
            self._add_featured_article(conn)

    def _initialize_with_initial_data_if_needed(self, conn: sqlite3.Connection) -> None:
        """Инициализация с начальными данными, если это необходимо"""
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM source_stats")
        if cursor.fetchone()[0] == 0:
            self._initialize_with_initial_data(conn)

    def _add_featured_article(self, conn: sqlite3.Connection) -> None:
        """Добавить статью дня в базу данных"""
        featured_article = {
            "title": self.FEATURED_ARTICLE_TITLE,
            "source": "Media Credibility Index",
            "url": "",
            "content": """Comprehensive analysis of the current Israel-Iran conflict.
The ongoing tensions between Israel and Iran represent one of the most complex geopolitical
challenges in the Middle East. This analysis examines the historical context, current dynamics,
and potential future scenarios of this conflict.
Key aspects include:
1. Historical background of Israel-Iran relations
2. Current military and political tensions
3. Regional implications and proxy conflicts
4. International responses and diplomatic efforts
5. Media coverage and information warfare""",
            "short_summary": """Analysis of the current situation between Israel and Iran, including historical context,
military dynamics, regional implications, and international responses. The analysis evaluates
media coverage from multiple perspectives and assesses the credibility of various sources.""",
            "analysis_data": json.dumps({
                "news_integrity": 0.85,
                "index_of_credibility": 0.82,
                "sentiment_score": 0.6,
                "bias_score": 0.3,
                "topics": ["Middle East", "Conflict", "International Relations", "Media Analysis"],
                "key_arguments": [
                    "The conflict has deep historical roots dating back to the Iranian Revolution",
                    "Current tensions are exacerbated by regional proxy conflicts",
                    "Media coverage varies significantly between Western and Middle Eastern sources",
                    "International diplomatic efforts have had limited success in de-escalation",
                    "The conflict has significant implications for regional stability and global energy markets"
                ],
                "western_perspective": {
                    "credibility": "High",
                    "summary": "Western media generally portrays Israel as defending itself against Iranian aggression and proxy threats.",
                    "key_points": [
                        "Focus on Israeli security concerns and right to self-defense",
                        "Coverage of Iranian support for groups like Hezbollah and Hamas",
                        "Emphasis on Iranian nuclear program as a regional threat"
                    ]
                },
                "iranian_perspective": {
                    "credibility": "Medium",
                    "summary": "Iranian media frames the conflict as resistance against Israeli occupation and Western imperialism.",
                    "key_points": [
                        "Historical context of Palestinian struggle and Israeli occupation",
                        "Portrayal of Iran as a defender of Muslim and Arab causes",
                        "Criticism of Western support for Israel"
                    ]
                },
                "israeli_perspective": {
                    "credibility": "High",
                    "summary": "Israeli perspective emphasizes security threats from Iran and its proxies.",
                    "key_points": [
                        "Iranian nuclear program as existential threat",
                        "Need to counter Iranian influence in Syria, Lebanon, and Gaza",
                        "Right to defend against attacks from Iranian-backed groups"
                    ]
                },
                "neutral_perspective": {
                    "credibility": "High",
                    "summary": "Neutral analysis acknowledges legitimate security concerns on both sides while warning about escalation risks.",
                    "key_points": [
                        "Complex historical and geopolitical factors driving the conflict",
                        "Danger of miscalculation leading to wider regional war",
                        "Need for diplomatic solutions to address root causes"
                    ]
                }
            }),
            "credibility_level": "High",
            "is_featured": True
        }

        cursor.execute("""
            INSERT INTO articles
            (title, source, url, content, short_summary, analysis_data, credibility_level, created_at, updated_at, is_featured)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            featured_article["title"],
            featured_article["source"],
            featured_article["url"],
            featured_article["content"],
            featured_article["short_summary"],
            featured_article["analysis_data"],
            featured_article["credibility_level"],
            datetime.now().isoformat(),
            datetime.now().isoformat(),
            featured_article["is_featured"]
        ))

    def _initialize_with_initial_data(self, conn: sqlite3.Connection) -> None:
        """Инициализация базы данных с начальными данными"""
        initial_sources = [
            ('BBC News', 5, 1, 0),
            ('Reuters', 5, 1, 0),
            ('The New York Times', 4, 2, 0),
            ('The Guardian', 4, 2, 0),
            ('CNN', 3, 3, 0),
            ('Fox News', 2, 3, 1),
            ('Al Jazeera', 3, 2, 1),
            ('RT', 1, 2, 3),
            ('Breitbart', 1, 1, 4),
            ('Daily Mail', 2, 2, 2)
        ]

        cursor = conn.cursor()
        for source, high, medium, low in initial_sources:
            cursor.execute("""
                INSERT INTO source_stats
                (source, high, medium, low, total_analyzed, is_initial)
                VALUES (?, ?, ?, ?, ?, 1)
            """, (source, high, medium, low, high + medium + low))

        logger.info("Database initialized with initial data successfully")

    def article_exists(self, title: Optional[str] = None, url: Optional[str] = None) -> bool:
        """Проверить, существует ли статья с таким заголовком или URL"""
        if not title and not url:
            return False

        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                if title and url:
                    cursor.execute("SELECT 1 FROM articles WHERE title = ? OR url = ?", (title, url))
                elif title:
                    cursor.execute("SELECT 1 FROM articles WHERE title = ?", (title,))
                else:
                    cursor.execute("SELECT 1 FROM articles WHERE url = ?", (url,))

                return cursor.fetchone() is not None
        except Exception as e:
            logger.error(f"Error checking if article exists: {str(e)}", exc_info=True)
            return False

    def get_article_by_id(self, article_id: int) -> Optional[Dict[str, Any]]:
        """Получить статью по ID с полным анализом"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM articles
                    WHERE id = ?
                    LIMIT 1
                """, (article_id,))

                article = cursor.fetchone()
                if not article:
                    return None

                article_dict = dict(article)
                article_dict['analysis'] = json.loads(article_dict['analysis_data']) if article_dict['analysis_data'] else {}

                # Получаем историю анализа для этой статьи
                cursor.execute("""
                    SELECT analysis_data, created_at
                    FROM analysis_history
                    WHERE article_id = ?
                    ORDER BY created_at DESC
                """, (article_id,))

                article_dict['analysis_history'] = [{
                    'analysis': json.loads(row['analysis_data']),
                    'date': row['created_at']
                } for row in cursor.fetchall()]

                return article_dict
        except Exception as e:
            logger.error(f"Error getting article by ID: {str(e)}", exc_info=True)
            return None

    def get_article_by_url(self, url: str) -> Optional[Dict[str, Any]]:
        """Получить статью по URL с полным анализом"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM articles
                    WHERE url = ?
                    LIMIT 1
                """, (url,))

                article = cursor.fetchone()
                if not article:
                    return None

                article_dict = dict(article)
                article_dict['analysis'] = json.loads(article_dict['analysis_data']) if article_dict['analysis_data'] else {}

                # Получаем историю анализа для этой статьи
                cursor.execute("""
                    SELECT analysis_data, created_at
                    FROM analysis_history
                    WHERE article_id = ?
                    ORDER BY created_at DESC
                """, (article_dict['id'],))

                article_dict['analysis_history'] = [{
                    'analysis': json.loads(row['analysis_data']),
                    'date': row['created_at']
                } for row in cursor.fetchall()]

                return article_dict
        except Exception as e:
            logger.error(f"Error getting article by URL: {str(e)}", exc_info=True)
            return None

    def save_article(self, title: str, source: str, url: Optional[str], content: str,
                   short_summary: str, analysis_data: Dict[str, Any], credibility_level: str,
                   is_featured: bool = False) -> int:
        """Сохранить статью в базу данных с анализом"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                # Проверяем, существует ли статья с таким URL
                existing_article = None
                if url:
                    cursor.execute("SELECT id FROM articles WHERE url = ?", (url,))
                    existing_article = cursor.fetchone()

                current_time = datetime.now().isoformat()

                if existing_article:
                    # Обновляем существующую статью
                    article_id = existing_article['id']

                    cursor.execute("""
                        UPDATE articles
                        SET
                            title = ?,
                            source = ?,
                            content = ?,
                            short_summary = ?,
                            analysis_data = ?,
                            credibility_level = ?,
                            updated_at = ?,
                            is_featured = ?
                        WHERE id = ?
                    """, (
                        title,
                        source,
                        content,
                        short_summary,
                        json.dumps(analysis_data),
                        credibility_level,
                        current_time,
                        is_featured,
                        article_id
                    ))
                else:
                    # Вставляем новую статью
                    cursor.execute("""
                        INSERT INTO articles
                        (title, source, url, content, short_summary, analysis_data,
                         credibility_level, created_at, updated_at, is_featured)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        title,
                        source,
                        url,
                        content,
                        short_summary,
                        json.dumps(analysis_data),
                        credibility_level,
                        current_time,
                        current_time,
                        is_featured
                    ))
                    article_id = cursor.lastrowid

                # Обновляем статистику источника
                self._update_source_stats(cursor, source, credibility_level)

                # Сохраняем историю анализа
                self._save_analysis_history(cursor, article_id, analysis_data)

                conn.commit()
                return article_id
        except Exception as e:
            logger.error(f"Error saving article: {str(e)}", exc_info=True)
            raise

    def _update_source_stats(self, cursor: sqlite3.Cursor, source: str, credibility_level: str) -> None:
        """Обновить статистику источников"""
        current_time = datetime.now().isoformat()

        if credibility_level == "High":
            cursor.execute("""
                INSERT INTO source_stats (source, high, medium, low, total_analyzed, is_initial, last_updated)
                VALUES (?, 1, 0, 0, 1, 0, ?)
                ON CONFLICT(source) DO UPDATE SET
                    high = high + 1,
                    total_analyzed = total_analyzed + 1,
                    is_initial = 0,
                    last_updated = ?
            """, (source, current_time, current_time))
        elif credibility_level == "Medium":
            cursor.execute("""
                INSERT INTO source_stats (source, high, medium, low, total_analyzed, is_initial, last_updated)
                VALUES (?, 0, 1, 0, 1, 0, ?)
                ON CONFLICT(source) DO UPDATE SET
                    medium = medium + 1,
                    total_analyzed = total_analyzed + 1,
                    is_initial = 0,
                    last_updated = ?
            """, (source, current_time, current_time))
        else:
            cursor.execute("""
                INSERT INTO source_stats (source, high, medium, low, total_analyzed, is_initial, last_updated)
                VALUES (?, 0, 0, 1, 1, 0, ?)
                ON CONFLICT(source) DO UPDATE SET
                    low = low + 1,
                    total_analyzed = total_analyzed + 1,
                    is_initial = 0,
                    last_updated = ?
            """, (source, current_time, current_time))

    def _save_analysis_history(self, cursor: sqlite3.Cursor, article_id: int, analysis_data: Dict[str, Any]) -> None:
        """Сохранить историю анализа для статьи"""
        cursor.execute("""
            INSERT INTO analysis_history (article_id, analysis_data)
            VALUES (?, ?)
        """, (article_id, json.dumps(analysis_data)))

    def update_article(self, article_id: int, **kwargs) -> bool:
        """Обновить данные статьи с возможностью частичного обновления"""
        if not kwargs:
            return False

        valid_fields = {
            'title', 'source', 'url', 'content', 'short_summary',
            'analysis_data', 'credibility_level', 'is_featured'
        }

        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                # Получаем текущие данные статьи
                cursor.execute("SELECT * FROM articles WHERE id = ?", (article_id,))
                current_article = cursor.fetchone()
                if not current_article:
                    return False

                # Проверяем, что обновляются только допустимые поля
                update_fields = {
                    k: v for k, v in kwargs.items()
                    if k in valid_fields and v is not None
                }

                if not update_fields:
                    return False

                # Подготавливаем данные для обновления
                set_clauses = []
                params = []

                for field, value in update_fields.items():
                    if field == 'analysis_data' and isinstance(value, dict):
                        value = json.dumps(value)
                    set_clauses.append(f"{field} = ?")
                    params.append(value)

                # Добавляем обновление updated_at
                set_clauses.append("updated_at = ?")
                params.append(datetime.now().isoformat())
                params.append(article_id)

                # Формируем и выполняем запрос
                query = f"UPDATE articles SET {', '.join(set_clauses)} WHERE id = ?"
                cursor.execute(query, params)

                # Если изменился источник или уровень достоверности, обновляем статистику
                if 'source' in update_fields or 'credibility_level' in update_fields:
                    source = update_fields.get('source', current_article['source'])
                    old_source = current_article['source']
                    credibility_level = update_fields.get('credibility_level', current_article['credibility_level'])
                    old_credibility = current_article['credibility_level']

                    if old_source != source or old_credibility != credibility_level:
                        # Удаляем старую запись из статистики
                        if old_source != source:
                            cursor.execute("""
                                UPDATE source_stats
                                SET
                                    high = high - CASE WHEN ? = 'High' THEN 1 ELSE 0 END,
                                    medium = medium - CASE WHEN ? = 'Medium' THEN 1 ELSE 0 END,
                                    low = low - CASE WHEN ? = 'Low' THEN 1 ELSE 0 END,
                                    total_analyzed = total_analyzed - 1
                                WHERE source = ?
                            """, (old_credibility, old_credibility, old_credibility, old_source))

                        # Добавляем новую запись в статистику
                        self._update_source_stats(cursor, source, credibility_level)

                # Если обновлялся анализ, сохраняем его в историю
                if 'analysis_data' in update_fields:
                    self._save_analysis_history(cursor, article_id, update_fields['analysis_data'])

                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Error updating article: {str(e)}", exc_info=True)
            return False

    def get_daily_buzz(self) -> Dict[str, Any]:
        """Получить статью дня с полным анализом"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM articles
                    WHERE is_featured = 1
                    LIMIT 1
                """)

                article = cursor.fetchone()
                if not article:
                    return {
                        'status': 'error',
                        'message': 'Daily buzz article not found'
                    }

                article_dict = dict(article)
                article_dict['analysis'] = json.loads(article_dict['analysis_data']) if article_dict['analysis_data'] else {}

                # Получаем историю анализа для этой статьи
                cursor.execute("""
                    SELECT analysis_data, created_at
                    FROM analysis_history
                    WHERE article_id = ?
                    ORDER BY created_at DESC
                """, (article_dict['id'],))

                article_dict['analysis_history'] = [{
                    'analysis': json.loads(row['analysis_data']),
                    'date': row['created_at']
                } for row in cursor.fetchall()]

                return {
                    'status': 'success',
                    'article': article_dict
                }
        except Exception as e:
            logger.error(f"Error getting daily buzz: {str(e)}", exc_info=True)
            return {
                'status': 'error',
                'message': 'Failed to retrieve daily buzz'
            }

    def get_source_credibility_chart(self) -> Dict[str, Any]:
        """Получить данные для чарта достоверности источников"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                # Получаем данные для чарта
                cursor.execute("""
                    SELECT
                        source,
                        high,
                        medium,
                        low,
                        total_analyzed,
                        ROUND((high * 1.0 + medium * 0.5) / NULLIF(total_analyzed, 0), 2) as credibility_score
                    FROM source_stats
                    WHERE total_analyzed > 0
                    ORDER BY credibility_score DESC
                """)

                rows = cursor.fetchall()
                sources = []
                credibility_scores = []

                for row in rows:
                    sources.append(row['source'])
                    credibility_scores.append(row['credibility_score'])

                # Получаем общее количество источников
                cursor.execute("SELECT COUNT(*) FROM source_stats WHERE total_analyzed > 0")
                total_sources = cursor.fetchone()[0]

                return {
                    'status': 'success',
                    'data': {
                        'sources': sources,
                        'credibility_scores': credibility_scores,
                        'total_sources': total_sources
                    }
                }
        except Exception as e:
            logger.error(f"Error getting source credibility chart: {str(e)}", exc_info=True)
            return {
                'status': 'error',
                'message': 'Failed to retrieve source credibility chart'
            }

    def get_source_statistics(self) -> Dict[str, Any]:
        """Получить полную статистику по источникам"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                # Получаем статистику по источникам
                cursor.execute("""
                    SELECT
                        source,
                        high,
                        medium,
                        low,
                        total_analyzed,
                        ROUND((high * 1.0 + medium * 0.5) / NULLIF(total_analyzed, 0), 2) as credibility_score,
                        last_updated
                    FROM source_stats
                    WHERE total_analyzed > 0
                    ORDER BY credibility_score DESC
                """)

                stats = []
                for row in cursor.fetchall():
                    stats.append({
                        'source': row['source'],
                        'high': row['high'],
                        'medium': row['medium'],
                        'low': row['low'],
                        'total': row['total_analyzed'],
                        'score': row['credibility_score'],
                        'last_updated': row['last_updated']
                    })

                # Получаем общее количество источников
                cursor.execute("SELECT COUNT(*) FROM source_stats WHERE total_analyzed > 0")
                total_sources = cursor.fetchone()[0]

                return {
                    'status': 'success',
                    'data': {
                        'sources': stats,
                        'total_sources': total_sources
                    }
                }
        except Exception as e:
            logger.error(f"Error getting source statistics: {str(e)}", exc_info=True)
            return {
                'status': 'error',
                'message': 'Failed to retrieve source statistics'
            }

    def get_analysis_history(self, limit: int = 10, offset: int = 0) -> Dict[str, Any]:
        """Получить историю анализа с пагинацией и полными данными"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                # Получаем статьи с пагинацией
                cursor.execute("""
                    SELECT
                        a.id, a.title, a.source, a.short_summary, a.credibility_level,
                        a.url, a.created_at, a.updated_at,
                        ah.analysis_data as latest_analysis
                    FROM articles a
                    LEFT JOIN (
                        SELECT article_id, analysis_data
                        FROM analysis_history
                        WHERE (article_id, created_at) IN (
                            SELECT article_id, MAX(created_at)
                            FROM analysis_history
                            GROUP BY article_id
                        )
                    ) ah ON a.id = ah.article_id
                    ORDER BY a.created_at DESC
                    LIMIT ? OFFSET ?
                """, (limit, offset))

                articles = []
                for article in cursor.fetchall():
                    article_dict = dict(article)
                    article_dict['analysis'] = json.loads(article_dict['latest_analysis']) if article_dict['latest_analysis'] else {}

                    # Получаем количество анализов для этой статьи
                    cursor.execute("""
                        SELECT COUNT(*) FROM analysis_history
                        WHERE article_id = ?
                    """, (article_dict['id'],))
                    article_dict['analysis_count'] = cursor.fetchone()[0]

                    articles.append(article_dict)

                # Получаем общее количество записей для пагинации
                cursor.execute("SELECT COUNT(*) FROM articles")
                total = cursor.fetchone()[0]

                return {
                    'status': 'success',
                    'history': articles,
                    'pagination': {
                        'total': total,
                        'limit': limit,
                        'offset': offset,
                        'has_more': offset + limit < total
                    }
                }
        except Exception as e:
            logger.error(f"Error getting analysis history: {str(e)}", exc_info=True)
            return {
                'status': 'error',
                'message': 'Failed to retrieve analysis history'
            }

    def get_similar_articles(self, topics: List[str], limit: int = 5) -> List[Dict[str, Any]]:
        """Получить похожие статьи на основе тем с учетом достоверности и полными данными"""
        try:
            if not topics:
                return []

            with self.get_connection() as conn:
                cursor = conn.cursor()

                # Создаем условия для поиска по темам
                conditions = []
                params = []

                for topic in topics:
                    conditions.append("(short_summary LIKE ? OR content LIKE ?)")
                    params.extend([f"%{topic}%", f"%{topic}%"])

                # Формируем запрос с учетом приоритета достоверности
                query = f"""
                    SELECT
                        a.id, a.title, a.source, a.short_summary, a.credibility_level,
                        a.url, a.created_at, a.updated_at,
                        ah.analysis_data as latest_analysis
                    FROM articles a
                    LEFT JOIN (
                        SELECT article_id, analysis_data
                        FROM analysis_history
                        WHERE (article_id, created_at) IN (
                            SELECT article_id, MAX(created_at)
                            FROM analysis_history
                            GROUP BY article_id
                        )
                    ) ah ON a.id = ah.article_id
                    WHERE ({' OR '.join(conditions)})
                    AND credibility_level IN ('High', 'Medium')
                    ORDER BY
                        CASE credibility_level
                            WHEN 'High' THEN 1
                            WHEN 'Medium' THEN 2
                            ELSE 3
                        END,
                        created_at DESC
                    LIMIT ?
                """

                params.append(limit)
                cursor.execute(query, params)

                articles = []
                for article in cursor.fetchall():
                    article_dict = dict(article)
                    article_dict['analysis'] = json.loads(article_dict['latest_analysis']) if article_dict['latest_analysis'] else {}

                    # Получаем количество анализов для этой статьи
                    cursor.execute("""
                        SELECT COUNT(*) FROM analysis_history
                        WHERE article_id = ?
                    """, (article_dict['id'],))
                    article_dict['analysis_count'] = cursor.fetchone()[0]

                    articles.append(article_dict)

                return articles
        except Exception as e:
            logger.error(f"Error getting similar articles: {str(e)}", exc_info=True)
            return []

    def search_articles(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Поиск статей по запросу с полными данными"""
        try:
            if not query:
                return []

            search_term = f"%{query}%"

            with self.get_connection() as conn:
                cursor = conn.cursor()

                # Формируем запрос с учетом релевантности
                cursor.execute("""
                    SELECT
                        a.id, a.title, a.source, a.short_summary, a.credibility_level,
                        a.url, a.created_at, a.updated_at,
                        ah.analysis_data as latest_analysis
                    FROM articles a
                    LEFT JOIN (
                        SELECT article_id, analysis_data
                        FROM analysis_history
                        WHERE (article_id, created_at) IN (
                            SELECT article_id, MAX(created_at)
                            FROM analysis_history
                            GROUP BY article_id
                        )
                    ) ah ON a.id = ah.article_id
                    WHERE title LIKE ? OR short_summary LIKE ? OR content LIKE ?
                    ORDER BY
                        CASE
                            WHEN title LIKE ? THEN 1
                            WHEN short_summary LIKE ? THEN 2
                            ELSE 3
                        END,
                        CASE credibility_level
                            WHEN 'High' THEN 1
                            WHEN 'Medium' THEN 2
                            ELSE 3
                        END,
                        created_at DESC
                    LIMIT ?
                """, (search_term, search_term, search_term, search_term, search_term, limit))

                articles = []
                for article in cursor.fetchall():
                    article_dict = dict(article)
                    article_dict['analysis'] = json.loads(article_dict['latest_analysis']) if article_dict['latest_analysis'] else {}

                    # Получаем количество анализов для этой статьи
                    cursor.execute("""
                        SELECT COUNT(*) FROM analysis_history
                        WHERE article_id = ?
                    """, (article_dict['id'],))
                    article_dict['analysis_count'] = cursor.fetchone()[0]

                    articles.append(article_dict)

                return articles
        except Exception as e:
            logger.error(f"Error searching articles: {str(e)}", exc_info=True)
            return []

    def get_articles_by_source(self, source: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Получить статьи от конкретного источника с полными данными"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT
                        a.id, a.title, a.short_summary, a.credibility_level,
                        a.url, a.created_at, a.updated_at,
                        ah.analysis_data as latest_analysis
                    FROM articles a
                    LEFT JOIN (
                        SELECT article_id, analysis_data
                        FROM analysis_history
                        WHERE (article_id, created_at) IN (
                            SELECT article_id, MAX(created_at)
                            FROM analysis_history
                            GROUP BY article_id
                        )
                    ) ah ON a.id = ah.article_id
                    WHERE a.source = ?
                    ORDER BY a.created_at DESC
                    LIMIT ?
                """, (source, limit))

                articles = []
                for article in cursor.fetchall():
                    article_dict = dict(article)
                    article_dict['analysis'] = json.loads(article_dict['latest_analysis']) if article_dict['latest_analysis'] else {}

                    # Получаем количество анализов для этой статьи
                    cursor.execute("""
                        SELECT COUNT(*) FROM analysis_history
                        WHERE article_id = ?
                    """, (article_dict['id'],))
                    article_dict['analysis_count'] = cursor.fetchone()[0]

                    articles.append(article_dict)

                return articles
        except Exception as e:
            logger.error(f"Error getting articles by source: {str(e)}", exc_info=True)
            return []

    def get_credibility_distribution(self) -> Dict[str, Any]:
        """Получить распределение статей по уровням достоверности с дополнительной статистикой"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                # Получаем распределение по уровням достоверности
                cursor.execute("""
                    SELECT
                        credibility_level,
                        COUNT(*) as count
                    FROM articles
                    GROUP BY credibility_level
                """)

                distribution = {}
                total = 0

                for row in cursor.fetchall():
                    distribution[row['credibility_level']] = {
                        'count': row['count'],
                        'percentage': 0  # Будет рассчитано позже
                    }
                    total += row['count']

                # Рассчитываем проценты
                for level in distribution:
                    distribution[level]['percentage'] = round(distribution[level]['count'] / total * 100, 1) if total > 0 else 0

                # Получаем дополнительную статистику
                cursor.execute("SELECT COUNT(*) FROM articles")
                total_articles = cursor.fetchone()[0]

                cursor.execute("SELECT COUNT(DISTINCT source) FROM articles")
                total_sources = cursor.fetchone()[0]

                return {
                    'status': 'success',
                    'data': {
                        'distribution': distribution,
                        'total_articles': total_articles,
                        'total_sources': total_sources
                    }
                }
        except Exception as e:
            logger.error(f"Error getting credibility distribution: {str(e)}", exc_info=True)
            return {
                'status': 'error',
                'message': 'Failed to retrieve credibility distribution'
            }

    def get_recent_articles(self, days: int = 7, limit: int = 10) -> List[Dict[str, Any]]:
        """Получить недавние статьи за указанное количество дней с полными данными"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT
                        a.id, a.title, a.source, a.short_summary, a.credibility_level,
                        a.url, a.created_at, a.updated_at,
                        ah.analysis_data as latest_analysis
                    FROM articles a
                    LEFT JOIN (
                        SELECT article_id, analysis_data
                        FROM analysis_history
                        WHERE (article_id, created_at) IN (
                            SELECT article_id, MAX(created_at)
                            FROM analysis_history
                            GROUP BY article_id
                        )
                    ) ah ON a.id = ah.article_id
                    WHERE a.created_at >= datetime('now', ?)
                    ORDER BY a.created_at DESC
                    LIMIT ?
                """, (f"-{days} days", limit))

                articles = []
                for article in cursor.fetchall():
                    article_dict = dict(article)
                    article_dict['analysis'] = json.loads(article_dict['latest_analysis']) if article_dict['latest_analysis'] else {}

                    # Получаем количество анализов для этой статьи
                    cursor.execute("""
                        SELECT COUNT(*) FROM analysis_history
                        WHERE article_id = ?
                    """, (article_dict['id'],))
                    article_dict['analysis_count'] = cursor.fetchone()[0]

                    articles.append(article_dict)

                return articles
        except Exception as e:
            logger.error(f"Error getting recent articles: {str(e)}", exc_info=True)
            return []

    def get_article_analysis_history(self, article_id: int) -> List[Dict[str, Any]]:
        """Получить историю анализа для конкретной статьи с полными данными"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                # Получаем историю анализа
                cursor.execute("""
                    SELECT analysis_data, created_at
                    FROM analysis_history
                    WHERE article_id = ?
                    ORDER BY created_at DESC
                """, (article_id,))

                return [{
                    'analysis': json.loads(row['analysis_data']),
                    'date': row['created_at']
                } for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error getting article analysis history: {str(e)}", exc_info=True)
            return []

    def get_articles_with_filters(
        self,
        source: Optional[str] = None,
        credibility_level: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        limit: int = 10,
        offset: int = 0
    ) -> Dict[str, Any]:
        """Получить статьи с фильтрами и пагинацией с полными данными"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                # Базовый запрос
                query = """
                    SELECT
                        a.id, a.title, a.source, a.short_summary, a.credibility_level,
                        a.url, a.created_at, a.updated_at,
                        ah.analysis_data as latest_analysis
                    FROM articles a
                    LEFT JOIN (
                        SELECT article_id, analysis_data
                        FROM analysis_history
                        WHERE (article_id, created_at) IN (
                            SELECT article_id, MAX(created_at)
                            FROM analysis_history
                            GROUP BY article_id
                        )
                    ) ah ON a.id = ah.article_id
                """

                # Условия фильтрации
                conditions = []
                params = []

                if source:
                    conditions.append("a.source = ?")
                    params.append(source)

                if credibility_level:
                    conditions.append("a.credibility_level = ?")
                    params.append(credibility_level)

                if date_from:
                    conditions.append("a.created_at >= ?")
                    params.append(date_from)

                if date_to:
                    conditions.append("a.created_at <= ?")
                    params.append(date_to)

                if conditions:
                    query += " WHERE " + " AND ".join(conditions)

                # Пагинация и сортировка
                query += " ORDER BY a.created_at DESC LIMIT ? OFFSET ?"
                params.extend([limit, offset])

                # Получаем статьи
                cursor.execute(query, params)

                articles = []
                for article in cursor.fetchall():
                    article_dict = dict(article)
                    article_dict['analysis'] = json.loads(article_dict['latest_analysis']) if article_dict['latest_analysis'] else {}

                    # Получаем количество анализов для этой статьи
                    cursor.execute("""
                        SELECT COUNT(*) FROM analysis_history
                        WHERE article_id = ?
                    """, (article_dict['id'],))
                    article_dict['analysis_count'] = cursor.fetchone()[0]

                    articles.append(article_dict)

                # Получаем общее количество записей для пагинации
                count_query = "SELECT COUNT(*) FROM articles"
                if conditions:
                    count_query += " WHERE " + " AND ".join(conditions)
                cursor.execute(count_query, params[:-2])
                total = cursor.fetchone()[0]

                return {
                    'status': 'success',
                    'articles': articles,
                    'pagination': {
                        'total': total,
                        'limit': limit,
                        'offset': offset,
                        'has_more': offset + limit < total
                    }
                }
        except Exception as e:
            logger.error(f"Error getting filtered articles: {str(e)}", exc_info=True)
            return {
                'status': 'error',
                'message': 'Failed to retrieve filtered articles'
            }

    def get_articles_by_time_range(
        self,
        start_date: str,
        end_date: str,
        limit: int = 10,
        offset: int = 0
    ) -> Dict[str, Any]:
        """Получить статьи за определенный временной диапазон с полными данными"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT
                        a.id, a.title, a.source, a.short_summary, a.credibility_level,
                        a.url, a.created_at, a.updated_at,
                        ah.analysis_data as latest_analysis
                    FROM articles a
                    LEFT JOIN (
                        SELECT article_id, analysis_data
                        FROM analysis_history
                        WHERE (article_id, created_at) IN (
                            SELECT article_id, MAX(created_at)
                            FROM analysis_history
                            GROUP BY article_id
                        )
                    ) ah ON a.id = ah.article_id
                    WHERE a.created_at BETWEEN ? AND ?
                    ORDER BY a.created_at DESC
                    LIMIT ? OFFSET ?
                """, (start_date, end_date, limit, offset))

                articles = []
                for article in cursor.fetchall():
                    article_dict = dict(article)
                    article_dict['analysis'] = json.loads(article_dict['latest_analysis']) if article_dict['latest_analysis'] else {}

                    # Получаем количество анализов для этой статьи
                    cursor.execute("""
                        SELECT COUNT(*) FROM analysis_history
                        WHERE article_id = ?
                    """, (article_dict['id'],))
                    article_dict['analysis_count'] = cursor.fetchone()[0]

                    articles.append(article_dict)

                # Получаем общее количество записей для пагинации
                cursor.execute("""
                    SELECT COUNT(*)
                    FROM articles
                    WHERE created_at BETWEEN ? AND ?
                """, (start_date, end_date))
                total = cursor.fetchone()[0]

                return {
                    'status': 'success',
                    'articles': articles,
                    'pagination': {
                        'total': total,
                        'limit': limit,
                        'offset': offset,
                        'has_more': offset + limit < total
                    }
                }
        except Exception as e:
            logger.error(f"Error getting articles by time range: {str(e)}", exc_info=True)
            return {
                'status': 'error',
                'message': 'Failed to retrieve articles by time range'
            }

    def get_articles_with_credibility_trend(
        self,
        days: int = 30,
        limit: int = 10
    ) -> Dict[str, Any]:
        """Получить статьи с трендом изменения достоверности за период"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                # Получаем статьи с несколькими анализами за период
                cursor.execute("""
                    SELECT
                        a.id, a.title, a.source,
                        a.credibility_level as current_credibility,
                        a.created_at,
                        COUNT(ah.id) as analysis_count,
                        MIN(ah.created_at) as first_analysis_date,
                        MAX(ah.created_at) as last_analysis_date
                    FROM articles a
                    JOIN analysis_history ah ON a.id = ah.article_id
                    WHERE a.created_at >= datetime('now', ?)
                    GROUP BY a.id
                    HAVING analysis_count > 1
                    ORDER BY analysis_count DESC
                    LIMIT ?
                """, (f"-{days} days", limit))

                articles = []
                for article in cursor.fetchall():
                    article_dict = dict(article)

                    # Получаем историю изменений достоверности
                    cursor.execute("""
                        SELECT
                            a.credibility_level,
                            ah.created_at
                        FROM articles a
                        JOIN analysis_history ah ON a.id = ah.article_id
                        WHERE a.id = ?
                        ORDER BY ah.created_at
                    """, (article_dict['id'],))

                    article_dict['credibility_history'] = [{
                        'credibility': row['credibility_level'],
                        'date': row['created_at']
                    } for row in cursor.fetchall()]

                    articles.append(article_dict)

                return {
                    'status': 'success',
                    'articles': articles
                }
        except Exception as e:
            logger.error(f"Error getting articles with credibility trend: {str(e)}", exc_info=True)
            return {
                'status': 'error',
                'message': 'Failed to retrieve articles with credibility trend'
            }

    def get_articles_with_topics(
        self,
        topics: List[str],
        limit: int = 10,
        offset: int = 0
    ) -> Dict[str, Any]:
        """Получить статьи по темам с полными данными"""
        try:
            if not topics:
                return {
                    'status': 'error',
                    'message': 'No topics provided'
                }

            with self.get_connection() as conn:
                cursor = conn.cursor()

                # Создаем условия для поиска по темам
                conditions = []
                params = []

                for topic in topics:
                    conditions.append("(a.short_summary LIKE ? OR a.content LIKE ?)")
                    params.extend([f"%{topic}%", f"%{topic}%"])

                # Формируем запрос
                query = f"""
                    SELECT
                        a.id, a.title, a.source, a.short_summary, a.credibility_level,
                        a.url, a.created_at, a.updated_at,
                        ah.analysis_data as latest_analysis
                    FROM articles a
                    LEFT JOIN (
                        SELECT article_id, analysis_data
                        FROM analysis_history
                        WHERE (article_id, created_at) IN (
                            SELECT article_id, MAX(created_at)
                            FROM analysis_history
                            GROUP BY article_id
                        )
                    ) ah ON a.id = ah.article_id
                    WHERE {' OR '.join(conditions)}
                    ORDER BY a.created_at DESC
                    LIMIT ? OFFSET ?
                """

                params.extend([limit, offset])
                cursor.execute(query, params)

                articles = []
                for article in cursor.fetchall():
                    article_dict = dict(article)
                    article_dict['analysis'] = json.loads(article_dict['latest_analysis']) if article_dict['latest_analysis'] else {}

                    # Получаем количество анализов для этой статьи
                    cursor.execute("""
                        SELECT COUNT(*) FROM analysis_history
                        WHERE article_id = ?
                    """, (article_dict['id'],))
                    article_dict['analysis_count'] = cursor.fetchone()[0]

                    articles.append(article_dict)

                # Получаем общее количество записей для пагинации
                count_query = f"SELECT COUNT(*) FROM articles WHERE {' OR '.join(conditions)}"
                cursor.execute(count_query, params[:-2])
                total = cursor.fetchone()[0]

                return {
                    'status': 'success',
                    'articles': articles,
                    'pagination': {
                        'total': total,
                        'limit': limit,
                        'offset': offset,
                        'has_more': offset + limit < total
                    }
                }
        except Exception as e:
            logger.error(f"Error getting articles by topics: {str(e)}", exc_info=True)
            return {
                'status': 'error',
                'message': 'Failed to retrieve articles by topics'
            }

    def get_articles_with_analysis_complexity(
        self,
        limit: int = 10
    ) -> Dict[str, Any]:
        """Получить статьи с анализом сложности анализа"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                # Получаем статьи с несколькими анализами
                cursor.execute("""
                    SELECT
                        a.id, a.title, a.source,
                        COUNT(ah.id) as analysis_count,
                        MIN(ah.created_at) as first_analysis_date,
                        MAX(ah.created_at) as last_analysis_date
                    FROM articles a
                    JOIN analysis_history ah ON a.id = ah.article_id
                    GROUP BY a.id
                    HAVING analysis_count > 1
                    ORDER BY analysis_count DESC
                    LIMIT ?
                """, (limit,))

                articles = []
                for article in cursor.fetchall():
                    article_dict = dict(article)

                    # Получаем историю анализа
                    cursor.execute("""
                        SELECT analysis_data, created_at
                        FROM analysis_history
                        WHERE article_id = ?
                        ORDER BY created_at
                    """, (article_dict['id'],))

                    article_dict['analysis_history'] = [{
                        'analysis': json.loads(row['analysis_data']),
                        'date': row['created_at']
                    } for row in cursor.fetchall()]

                    articles.append(article_dict)

                return {
                    'status': 'success',
                    'articles': articles
                }
        except Exception as e:
            logger.error(f"Error getting articles with analysis complexity: {str(e)}", exc_info=True)
            return {
                'status': 'error',
                'message': 'Failed to retrieve articles with analysis complexity'
            }

    def get_articles_with_credibility_changes(
        self,
        limit: int = 10
    ) -> Dict[str, Any]:
        """Получить статьи с изменениями уровня достоверности"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                # Получаем статьи с изменениями уровня достоверности
                cursor.execute("""
                    SELECT DISTINCT a.id, a.title, a.source
                    FROM articles a
                    JOIN analysis_history ah1 ON a.id = ah1.article_id
                    JOIN analysis_history ah2 ON a.id = ah2.article_id
                    WHERE ah1.credibility_level != ah2.credibility_level
                    AND ah1.created_at < ah2.created_at
                    LIMIT ?
                """, (limit,))

                articles = []
                for article in cursor.fetchall():
                    article_dict = dict(article)

                    # Получаем историю изменений достоверности
                    cursor.execute("""
                        SELECT
                            a.credibility_level,
                            ah.created_at
                        FROM articles a
                        JOIN analysis_history ah ON a.id = ah.article_id
                        WHERE a.id = ?
                        ORDER BY ah.created_at
                    """, (article_dict['id'],))

                    article_dict['credibility_history'] = [{
                        'credibility': row['credibility_level'],
                        'date': row['created_at']
                    } for row in cursor.fetchall()]

                    articles.append(article_dict)

                return {
                    'status': 'success',
                    'articles': articles
                }
        except Exception as e:
            logger.error(f"Error getting articles with credibility changes: {str(e)}", exc_info=True)
            return {
                'status': 'error',
                'message': 'Failed to retrieve articles with credibility changes'
            }

    def get_articles_with_multiple_analyses(
        self,
        limit: int = 10
    ) -> Dict[str, Any]:
        """Получить статьи с несколькими анализами"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                # Получаем статьи с несколькими анализами
                cursor.execute("""
                    SELECT
                        a.id, a.title, a.source,
                        COUNT(ah.id) as analysis_count,
                        MIN(ah.created_at) as first_analysis_date,
                        MAX(ah.created_at) as last_analysis_date
                    FROM articles a
                    JOIN analysis_history ah ON a.id = ah.article_id
                    GROUP BY a.id
                    HAVING analysis_count > 1
                    ORDER BY analysis_count DESC
                    LIMIT ?
                """, (limit,))

                articles = []
                for article in cursor.fetchall():
                    article_dict = dict(article)

                    # Получаем историю анализа
                    cursor.execute("""
                        SELECT analysis_data, created_at
                        FROM analysis_history
                        WHERE article_id = ?
                        ORDER BY created_at
                    """, (article_dict['id'],))

                    article_dict['analysis_history'] = [{
                        'analysis': json.loads(row['analysis_data']),
                        'date': row['created_at']
                    } for row in cursor.fetchall()]

                    articles.append(article_dict)

                return {
                    'status': 'success',
                    'articles': articles
                }
        except Exception as e:
            logger.error(f"Error getting articles with multiple analyses: {str(e)}", exc_info=True)
            return {
                'status': 'error',
                'message': 'Failed to retrieve articles with multiple analyses'
            }

    def get_articles_with_analysis_trends(
        self,
        days: int = 30,
        limit: int = 10
    ) -> Dict[str, Any]:
        """Получить статьи с трендами анализа за период"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                # Получаем статьи с несколькими анализами за период
                cursor.execute("""
                    SELECT
                        a.id, a.title, a.source,
                        COUNT(ah.id) as analysis_count,
                        MIN(ah.created_at) as first_analysis_date,
                        MAX(ah.created_at) as last_analysis_date
                    FROM articles a
                    JOIN analysis_history ah ON a.id = ah.article_id
                    WHERE a.created_at >= datetime('now', ?)
                    GROUP BY a.id
                    HAVING analysis_count > 1
                    ORDER BY analysis_count DESC
                    LIMIT ?
                """, (f"-{days} days", limit))

                articles = []
                for article in cursor.fetchall():
                    article_dict = dict(article)

                    # Получаем историю анализа
                    cursor.execute("""
                        SELECT analysis_data, created_at
                        FROM analysis_history
                        WHERE article_id = ?
                        ORDER BY created_at
                    """, (article_dict['id'],))

                    article_dict['analysis_history'] = [{
                        'analysis': json.loads(row['analysis_data']),
                        'date': row['created_at']
                    } for row in cursor.fetchall()]

                    articles.append(article_dict)

                return {
                    'status': 'success',
                    'articles': articles
                }
        except Exception as e:
            logger.error(f"Error getting articles with analysis trends: {str(e)}", exc_info=True)
            return {
                'status': 'error',
                'message': 'Failed to retrieve articles with analysis trends'
            }

    def get_articles_with_analysis_comparison(
        self,
        limit: int = 10
    ) -> Dict[str, Any]:
        """Получить статьи с сравнением анализов"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                # Получаем статьи с несколькими анализами
                cursor.execute("""
                    SELECT
                        a.id, a.title, a.source,
                        COUNT(ah.id) as analysis_count,
                        MIN(ah.created_at) as first_analysis_date,
                        MAX(ah.created_at) as last_analysis_date
                    FROM articles a
                    JOIN analysis_history ah ON a.id = ah.article_id
                    GROUP BY a.id
                    HAVING analysis_count > 1
                    ORDER BY analysis_count DESC
                    LIMIT ?
                """, (limit,))

                articles = []
                for article in cursor.fetchall():
                    article_dict = dict(article)

                    # Получаем историю анализа
                    cursor.execute("""
                        SELECT analysis_data, created_at
                        FROM analysis_history
                        WHERE article_id = ?
                        ORDER BY created_at
                    """, (article_dict['id'],))

                    analysis_history = [{
                        'analysis': json.loads(row['analysis_data']),
                        'date': row['created_at']
                    } for row in cursor.fetchall()]

                    # Сравниваем анализы
                    if len(analysis_history) > 1:
                        article_dict['analysis_comparison'] = {
                            'first_analysis': analysis_history[0]['analysis'],
                            'last_analysis': analysis_history[-1]['analysis'],
                            'changes': self._compare_analyses(
                                analysis_history[0]['analysis'],
                                analysis_history[-1]['analysis']
                            )
                        }

                    article_dict['analysis_history'] = analysis_history
                    articles.append(article_dict)

                return {
                    'status': 'success',
                    'articles': articles
                }
        except Exception as e:
            logger.error(f"Error getting articles with analysis comparison: {str(e)}", exc_info=True)
            return {
                'status': 'error',
                'message': 'Failed to retrieve articles with analysis comparison'
            }

    def _compare_analyses(self, analysis1: Dict, analysis2: Dict) -> Dict:
        """Сравнить два анализа и вернуть изменения"""
        changes = {}

        # Сравниваем числовые показатели
        numeric_fields = ['credibility_score', 'sentiment_score', 'bias_score']
        for field in numeric_fields:
            if field in analysis1 and field in analysis2:
                changes[field] = {
                    'old': analysis1[field],
                    'new': analysis2[field],
                    'change': analysis2[field] - analysis1[field]
                }

        # Сравниваем темы
        if 'topics' in analysis1 and 'topics' in analysis2:
            old_topics = set(analysis1['topics'])
            new_topics = set(analysis2['topics'])
            changes['topics'] = {
                'added': list(new_topics - old_topics),
                'removed': list(old_topics - new_topics),
                'common': list(old_topics & new_topics)
            }

        # Сравниваем перспективы
        if 'perspectives' in analysis1 and 'perspectives' in analysis2:
            perspective_changes = {}
            for perspective in set(analysis1['perspectives'].keys()).union(set(analysis2['perspectives'].keys())):
                old_perspective = analysis1['perspectives'].get(perspective, {})
                new_perspective = analysis2['perspectives'].get(perspective, {})

                perspective_changes[perspective] = {
                    'credibility': {
                        'old': old_perspective.get('credibility'),
                        'new': new_perspective.get('credibility')
                    },
                    'summary_changed': old_perspective.get('summary') != new_perspective.get('summary')
                }

            changes['perspectives'] = perspective_changes

        return changes

    def get_articles_with_analysis_quality(
        self,
        limit: int = 10
    ) -> Dict[str, Any]:
        """Получить статьи с анализом качества анализа"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                # Получаем статьи с несколькими анализами
                cursor.execute("""
                    SELECT
                        a.id, a.title, a.source,
                        COUNT(ah.id) as analysis_count,
                        MIN(ah.created_at) as first_analysis_date,
                        MAX(ah.created_at) as last_analysis_date
                    FROM articles a
                    JOIN analysis_history ah ON a.id = ah.article_id
                    GROUP BY a.id
                    HAVING analysis_count > 1
                    ORDER BY analysis_count DESC
                    LIMIT ?
                """, (limit,))

                articles = []
                for article in cursor.fetchall():
                    article_dict = dict(article)

                    # Получаем историю анализа
                    cursor.execute("""
                        SELECT analysis_data, created_at
                        FROM analysis_history
                        WHERE article_id = ?
                        ORDER BY created_at
                    """, (article_dict['id'],))

                    analysis_history = [{
                        'analysis': json.loads(row['analysis_data']),
                        'date': row['created_at']
                    } for row in cursor.fetchall()]

                    # Анализируем качество анализа
                    if len(analysis_history) > 1:
                        article_dict['analysis_quality'] = self._analyze_analysis_quality(analysis_history)

                    article_dict['analysis_history'] = analysis_history
                    articles.append(article_dict)

                return {
                    'status': 'success',
                    'articles': articles
                }
        except Exception as e:
            logger.error(f"Error getting articles with analysis quality: {str(e)}", exc_info=True)
            return {
                'status': 'error',
                'message': 'Failed to retrieve articles with analysis quality'
            }

    def _analyze_analysis_quality(self, analysis_history: List[Dict]) -> Dict:
        """Анализ качества анализа на основе истории"""
        quality_metrics = {
            'consistency': self._calculate_consistency(analysis_history),
            'depth': self._calculate_depth(analysis_history),
            'completeness': self._calculate_completeness(analysis_history),
            'timeliness': self._calculate_timeliness(analysis_history)
        }

        # Рассчитываем общий рейтинг качества
        quality_metrics['overall_rating'] = sum(quality_metrics.values()) / len(quality_metrics)

        return quality_metrics

    def _calculate_consistency(self, analysis_history: List[Dict]) -> float:
        """Рассчитать показатель согласованности анализов"""
        if len(analysis_history) < 2:
            return 0.0

        # Сравниваем числовые показатели
        numeric_fields = ['credibility_score', 'sentiment_score', 'bias_score']
        total_changes = 0
        field_count = 0

        for field in numeric_fields:
            values = []
            for analysis in analysis_history:
                if field in analysis['analysis']:
                    values.append(analysis['analysis'][field])

            if len(values) > 1:
                max_change = max(abs(values[i] - values[i-1]) for i in range(1, len(values)))
                total_changes += max_change
                field_count += 1

        if field_count == 0:
            return 1.0

        # Нормализуем показатель согласованности (1.0 - максимальное изменение)
        consistency = 1.0 - (total_changes / field_count)
        return max(0.0, min(1.0, consistency))

    def _calculate_depth(self, analysis_history: List[Dict]) -> float:
        """Рассчитать показатель глубины анализа"""
        if not analysis_history:
            return 0.0

        # Проверяем наличие различных типов анализа
        depth_factors = 0
        max_depth = 5  # Максимальное количество факторов глубины

        # Проверяем наличие различных перспектив
        if 'perspectives' in analysis_history[0]['analysis']:
            depth_factors += min(1.0, len(analysis_history[0]['analysis']['perspectives']) / 4)

        # Проверяем наличие ключевых аргументов
        if 'key_arguments' in analysis_history[0]['analysis']:
            depth_factors += min(1.0, len(analysis_history[0]['analysis']['key_arguments']) / 5)

        # Проверяем наличие тем
        if 'topics' in analysis_history[0]['analysis']:
            depth_factors += min(1.0, len(analysis_history[0]['analysis']['topics']) / 5)

        # Проверяем наличие оценок
        numeric_fields = ['credibility_score', 'sentiment_score', 'bias_score']
        for field in numeric_fields:
            if field in analysis_history[0]['analysis']:
                depth_factors += 0.3

        return depth_factors / max_depth

    def _calculate_completeness(self, analysis_history: List[Dict]) -> float:
        """Рассчитать показатель полноты анализа"""
        if not analysis_history:
            return 0.0

        # Проверяем наличие различных компонентов анализа
        completeness_factors = 0
        max_factors = 8  # Максимальное количество факторов полноты

        required_fields = [
            'credibility_score', 'sentiment_score', 'bias_score',
            'topics', 'key_arguments', 'perspectives', 'summary'
        ]

        for field in required_fields:
            if field in analysis_history[0]['analysis']:
                completeness_factors += 1

        return completeness_factors / max_factors

    def _calculate_timeliness(self, analysis_history: List[Dict]) -> float:
        """Рассчитать показатель своевременности анализа"""
        if len(analysis_history) < 2:
            return 0.0

        # Рассчитываем средний интервал между анализами
        dates = [datetime.fromisoformat(analysis['date']) for analysis in analysis_history]
        intervals = [(dates[i] - dates[i-1]).total_seconds() / 3600 for i in range(1, len(dates))]

        if not intervals:
            return 0.0

        avg_interval = sum(intervals) / len(intervals)

        # Нормализуем показатель своевременности (чем меньше интервал, тем лучше)
        timeliness = 1.0 / (1.0 + avg_interval / 24)  # Нормализация на основе 24-часового интервала
        return max(0.0, min(1.0, timeliness))

    def get_articles_with_analysis_evolution(
        self,
        limit: int = 10
    ) -> Dict[str, Any]:
        """Получить статьи с эволюцией анализа"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                # Получаем статьи с несколькими анализами
                cursor.execute("""
                    SELECT
                        a.id, a.title, a.source,
                        COUNT(ah.id) as analysis_count,
                        MIN(ah.created_at) as first_analysis_date,
                        MAX(ah.created_at) as last_analysis_date
                    FROM articles a
                    JOIN analysis_history ah ON a.id = ah.article_id
                    GROUP BY a.id
                    HAVING analysis_count > 1
                    ORDER BY analysis_count DESC
                    LIMIT ?
                """, (limit,))

                articles = []
                for article in cursor.fetchall():
                    article_dict = dict(article)

                    # Получаем историю анализа
                    cursor.execute("""
                        SELECT analysis_data, created_at
                        FROM analysis_history
                        WHERE article_id = ?
                        ORDER BY created_at
                    """, (article_dict['id'],))

                    analysis_history = [{
                        'analysis': json.loads(row['analysis_data']),
                        'date': row['created_at']
                    } for row in cursor.fetchall()]

                    # Анализируем эволюцию анализа
                    if len(analysis_history) > 1:
                        article_dict['analysis_evolution'] = self._analyze_analysis_evolution(analysis_history)

                    article_dict['analysis_history'] = analysis_history
                    articles.append(article_dict)

                return {
                    'status': 'success',
                    'articles': articles
                }
        except Exception as e:
            logger.error(f"Error getting articles with analysis evolution: {str(e)}", exc_info=True)
            return {
                'status': 'error',
                'message': 'Failed to retrieve articles with analysis evolution'
            }

    def _analyze_analysis_evolution(self, analysis_history: List[Dict]) -> Dict:
        """Анализ эволюции анализа на основе истории"""
        evolution_metrics = {
            'credibility_trend': self._calculate_credibility_trend(analysis_history),
            'sentiment_trend': self._calculate_sentiment_trend(analysis_history),
            'bias_trend': self._calculate_bias_trend(analysis_history),
            'topic_evolution': self._calculate_topic_evolution(analysis_history),
            'perspective_evolution': self._calculate_perspective_evolution(analysis_history)
        }

        return evolution_metrics

    def _calculate_credibility_trend(self, analysis_history: List[Dict]) -> Dict:
        """Рассчитать тренд изменения достоверности"""
        if len(analysis_history) < 2:
            return {'trend': 'stable', 'change': 0.0}

        values = []
        for analysis in analysis_history:
            if 'credibility_score' in analysis['analysis']:
                values.append(analysis['analysis']['credibility_score'])

        if len(values) < 2:
            return {'trend': 'stable', 'change': 0.0}

        # Рассчитываем изменение
        first_value = values[0]
        last_value = values[-1]
        change = last_value - first_value

        # Определяем тренд
        if change > 0.1:
            trend = 'increasing'
        elif change < -0.1:
            trend = 'decreasing'
        else:
            trend = 'stable'

        return {
            'trend': trend,
            'change': change,
            'first_value': first_value,
            'last_value': last_value
        }

    def _calculate_sentiment_trend(self, analysis_history: List[Dict]) -> Dict:
        """Рассчитать тренд изменения сентимента"""
        if len(analysis_history) < 2:
            return {'trend': 'stable', 'change': 0.0}

        values = []
        for analysis in analysis_history:
            if 'sentiment_score' in analysis['analysis']:
                values.append(analysis['analysis']['sentiment_score'])

        if len(values) < 2:
            return {'trend': 'stable', 'change': 0.0}

        # Рассчитываем изменение
        first_value = values[0]
        last_value = values[-1]
        change = last_value - first_value

        # Определяем тренд
        if change > 0.1:
            trend = 'more_positive'
        elif change < -0.1:
            trend = 'more_negative'
        else:
            trend = 'stable'

        return {
            'trend': trend,
            'change': change,
            'first_value': first_value,
            'last_value': last_value
        }

    def _calculate_bias_trend(self, analysis_history: List[Dict]) -> Dict:
        """Рассчитать тренд изменения предвзятости"""
        if len(analysis_history) < 2:
            return {'trend': 'stable', 'change': 0.0}

        values = []
        for analysis in analysis_history:
            if 'bias_score' in analysis['analysis']:
                values.append(analysis['analysis']['bias_score'])

        if len(values) < 2:
            return {'trend': 'stable', 'change': 0.0}

        # Рассчитываем изменение
        first_value = values[0]
        last_value = values[-1]
        change = last_value - first_value

        # Определяем тренд
        if change > 0.1:
            trend = 'increasing'
        elif change < -0.1:
            trend = 'decreasing'
        else:
            trend = 'stable'

        return {
            'trend': trend,
            'change': change,
            'first_value': first_value,
            'last_value': last_value
        }

    def _calculate_topic_evolution(self, analysis_history: List[Dict]) -> Dict:
        """Рассчитать эволюцию тем"""
        if len(analysis_history) < 2:
            return {'changes': 0, 'new_topics': [], 'removed_topics': []}

        # Получаем все уникальные темы
        all_topics = set()
        for analysis in analysis_history:
            if 'topics' in analysis['analysis']:
                all_topics.update(analysis['analysis']['topics'])

        # Анализируем изменения тем
        topic_changes = []
        for i in range(1, len(analysis_history)):
            prev_topics = set(analysis_history[i-1]['analysis'].get('topics', []))
            current_topics = set(analysis_history[i]['analysis'].get('topics', []))

            added = list(current_topics - prev_topics)
            removed = list(prev_topics - current_topics)

            if added or removed:
                topic_changes.append({
                    'date': analysis_history[i]['date'],
                    'added': added,
                    'removed': removed
                })

        return {
            'changes': len(topic_changes),
            'topic_changes': topic_changes,
            'all_topics': list(all_topics)
        }

    def _calculate_perspective_evolution(self, analysis_history: List[Dict]) -> Dict:
        """Рассчитать эволюцию перспектив"""
        if len(analysis_history) < 2:
            return {'changes': 0, 'perspective_changes': []}

        # Получаем все уникальные перспективы
        all_perspectives = set()
        for analysis in analysis_history:
            if 'perspectives' in analysis['analysis']:
                all_perspectives.update(analysis['analysis']['perspectives'].keys())

        # Анализируем изменения перспектив
        perspective_changes = []
        for i in range(1, len(analysis_history)):
            prev_perspectives = set(analysis_history[i-1]['analysis'].get('perspectives', {}).keys())
            current_perspectives = set(analysis_history[i]['analysis'].get('perspectives', {}).keys())

            added = list(current_perspectives - prev_perspectives)
            removed = list(prev_perspectives - current_perspectives)

            if added or removed:
                perspective_changes.append({
                    'date': analysis_history[i]['date'],
                    'added': added,
                    'removed': removed
                })

        return {
            'changes': len(perspective_changes),
            'perspective_changes': perspective_changes,
            'all_perspectives': list(all_perspectives)
        }

    def get_articles_with_analysis_comparison_summary(
        self,
        limit: int = 10
    ) -> Dict[str, Any]:
        """Получить статьи с сравнительным анализом"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                # Получаем статьи с несколькими анализами
                cursor.execute("""
                    SELECT
                        a.id, a.title, a.source,
                        COUNT(ah.id) as analysis_count,
                        MIN(ah.created_at) as first_analysis_date,
                        MAX(ah.created_at) as last_analysis_date
                    FROM articles a
                    JOIN analysis_history ah ON a.id = ah.article_id
                    GROUP BY a.id
                    HAVING analysis_count > 1
                    ORDER BY analysis_count DESC
                    LIMIT ?
                """, (limit,))

                articles = []
                for article in cursor.fetchall():
                    article_dict = dict(article)

                    # Получаем историю анализа
                    cursor.execute("""
                        SELECT analysis_data, created_at
                        FROM analysis_history
                        WHERE article_id = ?
                        ORDER BY created_at
                    """, (article_dict['id'],))

                    analysis_history = [{
                        'analysis': json.loads(row['analysis_data']),
                        'date': row['created_at']
                    } for row in cursor.fetchall()]

                    # Сравниваем анализы
                    if len(analysis_history) > 1:
                        article_dict['analysis_comparison'] = self._compare_analyses(
                            analysis_history[0]['analysis'],
                            analysis_history[-1]['analysis']
                        )

                        # Добавляем сводку изменений
                        article_dict['analysis_comparison_summary'] = self._generate_comparison_summary(
                            analysis_history[0]['analysis'],
                            analysis_history[-1]['analysis']
                        )

                    article_dict['analysis_history'] = analysis_history
                    articles.append(article_dict)

                return {
                    'status': 'success',
                    'articles': articles
                }
        except Exception as e:
            logger.error(f"Error getting articles with analysis comparison summary: {str(e)}", exc_info=True)
            return {
                'status': 'error',
                'message': 'Failed to retrieve articles with analysis comparison summary'
            }

    def _generate_comparison_summary(self, analysis1: Dict, analysis2: Dict) -> str:
        """Сгенерировать текстовое описание изменений между анализами"""
        summary_parts = []

        # Сравниваем числовые показатели
        numeric_fields = {
            'credibility_score': 'credibility',
            'sentiment_score': 'sentiment',
            'bias_score': 'bias'
        }

        for field, name in numeric_fields.items():
            if field in analysis1 and field in analysis2:
                change = analysis2[field] - analysis1[field]
                if abs(change) > 0.1:
                    direction = "increased" if change > 0 else "decreased"
                    summary_parts.append(
                        f"The {name} score {direction} from {analysis1[field]:.2f} to {analysis2[field]:.2f}"
                    )

        # Сравниваем темы
        if 'topics' in analysis1 and 'topics' in analysis2:
            old_topics = set(analysis1['topics'])
            new_topics = set(analysis2['topics'])
            added = new_topics - old_topics
            removed = old_topics - new_topics

            if added:
                summary_parts.append(
                    f"Added topics: {', '.join(added)}"
                )
            if removed:
                summary_parts.append(
                    f"Removed topics: {', '.join(removed)}"
                )

        # Сравниваем перспективы
        if 'perspectives' in analysis1 and 'perspectives' in analysis2:
            old_perspectives = set(analysis1['perspectives'].keys())
            new_perspectives = set(analysis2['perspectives'].keys())
            added = new_perspectives - old_perspectives
            removed = old_perspectives - new_perspectives

            if added:
                summary_parts.append(
                    f"Added perspectives: {', '.join(added)}"
                )
            if removed:
                summary_parts.append(
                    f"Removed perspectives: {', '.join(removed)}"
                )

        # Сравниваем ключевые аргументы
        if 'key_arguments' in analysis1 and 'key_arguments' in analysis2:
            old_args = set(analysis1['key_arguments'])
            new_args = set(analysis2['key_arguments'])
            added = new_args - old_args
            removed = old_args - new_args

            if added:
                summary_parts.append(
                    f"Added key arguments: {', '.join(added)}"
                )
            if removed:
                summary_parts.append(
                    f"Removed key arguments: {', '.join(removed)}"
                )

        if not summary_parts:
            return "No significant changes detected between analyses."

        return "Analysis comparison summary: " + ". ".join(summary_parts) + "."

    def get_articles_with_analysis_quality_summary(
        self,
        limit: int = 10
    ) -> Dict[str, Any]:
        """Получить статьи с анализом качества анализа и сводкой"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                # Получаем статьи с несколькими анализами
                cursor.execute("""
                    SELECT
                        a.id, a.title, a.source,
                        COUNT(ah.id) as analysis_count,
                        MIN(ah.created_at) as first_analysis_date,
                        MAX(ah.created_at) as last_analysis_date
                    FROM articles a
                    JOIN analysis_history ah ON a.id = ah.article_id
                    GROUP BY a.id
                    HAVING analysis_count > 1
                    ORDER BY analysis_count DESC
                    LIMIT ?
                """, (limit,))

                articles = []
                for article in cursor.fetchall():
                    article_dict = dict(article)

                    # Получаем историю анализа
                    cursor.execute("""
                        SELECT analysis_data, created_at
                        FROM analysis_history
                        WHERE article_id = ?
                        ORDER BY created_at
                    """, (article_dict['id'],))

                    analysis_history = [{
                        'analysis': json.loads(row['analysis_data']),
                        'date': row['created_at']
                    } for row in cursor.fetchall()]

                    # Анализируем качество анализа
                    if len(analysis_history) > 1:
                        quality_metrics = self._analyze_analysis_quality(analysis_history)
                        article_dict['analysis_quality'] = quality_metrics

                        # Добавляем текстовое описание качества
                        article_dict['quality_summary'] = self._generate_quality_summary(quality_metrics)

                    article_dict['analysis_history'] = analysis_history
                    articles.append(article_dict)

                return {
                    'status': 'success',
                    'articles': articles
                }
        except Exception as e:
            logger.error(f"Error getting articles with analysis quality summary: {str(e)}", exc_info=True)
            return {
                'status': 'error',
                'message': 'Failed to retrieve articles with analysis quality summary'
            }

    def _generate_quality_summary(self, quality_metrics: Dict) -> str:
        """Сгенерировать текстовое описание качества анализа"""
        summary_parts = []

        # Оценка согласованности
        consistency = quality_metrics.get('consistency', 0)
        if consistency > 0.8:
            summary_parts.append("The analysis shows high consistency between different evaluations")
        elif consistency > 0.5:
            summary_parts.append("The analysis shows moderate consistency between different evaluations")
        else:
            summary_parts.append("The analysis shows low consistency between different evaluations")

        # Оценка глубины
        depth = quality_metrics.get('depth', 0)
        if depth > 0.8:
            summary_parts.append("The analysis demonstrates comprehensive depth")
        elif depth > 0.5:
            summary_parts.append("The analysis shows moderate depth")
        else:
            summary_parts.append("The analysis appears relatively shallow")

        # Оценка полноты
        completeness = quality_metrics.get('completeness', 0)
        if completeness > 0.8:
            summary_parts.append("The analysis is highly complete with all expected components")
        elif completeness > 0.5:
            summary_parts.append("The analysis is moderately complete")
        else:
            summary_parts.append("The analysis is incomplete with missing components")

        # Оценка своевременности
        timeliness = quality_metrics.get('timeliness', 0)
        if timeliness > 0.8:
            summary_parts.append("The analysis shows excellent timeliness with frequent updates")
        elif timeliness > 0.5:
            summary_parts.append("The analysis shows moderate timeliness")
        else:
            summary_parts.append("The analysis shows poor timeliness with infrequent updates")

        # Общая оценка
        overall = quality_metrics.get('overall_rating', 0)
        if overall > 0.8:
            summary_parts.append("Overall, this represents a high-quality analysis")
        elif overall > 0.5:
            summary_parts.append("Overall, this represents a moderate-quality analysis")
        else:
            summary_parts.append("Overall, this represents a low-quality analysis")

        return "Analysis quality summary: " + ". ".join(summary_parts) + "."

    def get_articles_with_analysis_evolution_summary(
        self,
        limit: int = 10
    ) -> Dict[str, Any]:
        """Получить статьи с анализом эволюции анализа и сводкой"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                # Получаем статьи с несколькими анализами
                cursor.execute("""
                    SELECT
                        a.id, a.title, a.source,
                        COUNT(ah.id) as analysis_count,
                        MIN(ah.created_at) as first_analysis_date,
                        MAX(ah.created_at) as last_analysis_date
                    FROM articles a
                    JOIN analysis_history ah ON a.id = ah.article_id
                    GROUP BY a.id
                    HAVING analysis_count > 1
                    ORDER BY analysis_count DESC
                    LIMIT ?
                """, (limit,))

                articles = []
                for article in cursor.fetchall():
                    article_dict = dict(article)

                    # Получаем историю анализа
                    cursor.execute("""
                        SELECT analysis_data, created_at
                        FROM analysis_history
                        WHERE article_id = ?
                        ORDER BY created_at
                    """, (article_dict['id'],))

                    analysis_history = [{
                        'analysis': json.loads(row['analysis_data']),
                        'date': row['created_at']
                    } for row in cursor.fetchall()]

                    # Анализируем эволюцию анализа
                    if len(analysis_history) > 1:
                        evolution_metrics = self._analyze_analysis_evolution(analysis_history)
                        article_dict['analysis_evolution'] = evolution_metrics

                        # Добавляем текстовое описание эволюции
                        article_dict['evolution_summary'] = self._generate_evolution_summary(evolution_metrics)

                    article_dict['analysis_history'] = analysis_history
                    articles.append(article_dict)

                return {
                    'status': 'success',
                    'articles': articles
                }
        except Exception as e:
            logger.error(f"Error getting articles with analysis evolution summary: {str(e)}", exc_info=True)
            return {
                'status': 'error',
                'message': 'Failed to retrieve articles with analysis evolution summary'
            }

    def _generate_evolution_summary(self, evolution_metrics: Dict) -> str:
        """Сгенерировать текстовое описание эволюции анализа"""
        summary_parts = []

        # Анализ тренда достоверности
        credibility_trend = evolution_metrics.get('credibility_trend', {})
        if credibility_trend:
            if credibility_trend['trend'] == 'increasing':
                summary_parts.append(
                    f"The credibility score has increased from {credibility_trend['first_value']:.2f} "
                    f"to {credibility_trend['last_value']:.2f}, showing improved credibility"
                )
            elif credibility_trend['trend'] == 'decreasing':
                summary_parts.append(
                    f"The credibility score has decreased from {credibility_trend['first_value']:.2f} "
                    f"to {credibility_trend['last_value']:.2f}, showing reduced credibility"
                )
            else:
                summary_parts.append(
                    f"The credibility score has remained stable around {credibility_trend['first_value']:.2f}"
                )

        # Анализ тренда сентимента
        sentiment_trend = evolution_metrics.get('sentiment_trend', {})
        if sentiment_trend:
            if sentiment_trend['trend'] == 'more_positive':
                summary_parts.append(
                    f"The sentiment score has improved from {sentiment_trend['first_value']:.2f} "
                    f"to {sentiment_trend['last_value']:.2f}, showing more positive sentiment"
                )
            elif sentiment_trend['trend'] == 'more_negative
