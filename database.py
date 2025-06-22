import os
import sqlite3
import json
import logging
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
from urllib.parse import urlparse

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Database:
    FEATURED_ARTICLE_TITLE = "Israel-Iran Conflict: Comprehensive Analysis"

    def __init__(self, db_path: str = 'instance/credibility_index.db'):
        self.db_path = db_path
        self._ensure_db_exists()
        self._initialize_schema()

    def _ensure_db_exists(self) -> None:
        """Убедиться, что база данных существует"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        if not os.path.exists(self.db_path):
            logger.info(f"Creating new SQLite database at {self.db_path}")
            open(self.db_path, 'w').close()

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
                        credibility_level TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
                        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS analysis_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        article_id INTEGER,
                        analysis_data TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (article_id) REFERENCES articles(id)
                    )
                """)

                # Создаем индексы для ускорения поиска
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_articles_title ON articles(title)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_articles_url ON articles(url)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_articles_created_at ON articles(created_at)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_articles_source ON articles(source)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_articles_credibility ON articles(credibility_level)")

                # Добавляем статью дня, если её нет
                cursor.execute("SELECT COUNT(*) FROM articles WHERE title = ?", (self.FEATURED_ARTICLE_TITLE,))
                if cursor.fetchone()[0] == 0:
                    self._add_featured_article(conn)

                # Инициализируем с начальными данными
                self._initialize_with_initial_data(conn)

                conn.commit()
                logger.info("Database schema initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing database schema: {str(e)}", exc_info=True)
            raise

    def _initialize_with_initial_data(self, conn: sqlite3.Connection) -> None:
        """Инициализировать базу данных с начальными данными"""
        try:
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
            cursor.execute("SELECT COUNT(*) FROM source_stats")
            count = cursor.fetchone()[0]

            if count == 0:
                for source, high, medium, low in initial_sources:
                    cursor.execute("""
                        INSERT INTO source_stats
                        (source, high, medium, low, total_analyzed, is_initial)
                        VALUES (?, ?, ?, ?, ?, 1)
                    """, (source, high, medium, low, high + medium + low))

                logger.info("Database initialized with initial data successfully")

        except Exception as e:
            logger.error(f"Error initializing database with initial data: {str(e)}", exc_info=True)
            raise

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
            "credibility_level": "High"
        }

        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO articles
            (title, source, url, content, short_summary, analysis_data, credibility_level, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            featured_article["title"],
            featured_article["source"],
            featured_article["url"],
            featured_article["content"],
            featured_article["short_summary"],
            featured_article["analysis_data"],
            featured_article["credibility_level"],
            datetime.now().isoformat(),
            datetime.now().isoformat()
        ))

    def get_connection(self) -> sqlite3.Connection:
        """Получить соединение с базой данных"""
        try:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA foreign_keys = ON;")
            return conn
        except Exception as e:
            logger.error(f"Error getting database connection: {str(e)}", exc_info=True)
            raise

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
        """Получить статью по ID"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM articles
                    WHERE id = ?
                    LIMIT 1
                """, (article_id,))

                article = cursor.fetchone()
                if article:
                    article_dict = dict(article)
                    article_dict['analysis'] = json.loads(article_dict['analysis_data']) if article_dict['analysis_data'] else {}
                    return article_dict
                return None

        except Exception as e:
            logger.error(f"Error getting article by ID: {str(e)}", exc_info=True)
            return None

    def get_article_by_url(self, url: str) -> Optional[Dict[str, Any]]:
        """Получить статью по URL"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM articles
                    WHERE url = ?
                    LIMIT 1
                """, (url,))

                article = cursor.fetchone()
                if article:
                    article_dict = dict(article)
                    article_dict['analysis'] = json.loads(article_dict['analysis_data']) if article_dict['analysis_data'] else {}
                    return article_dict
                return None

        except Exception as e:
            logger.error(f"Error getting article by URL: {str(e)}", exc_info=True)
            return None

    def save_article(self, title: str, source: str, url: Optional[str], content: str,
                    short_summary: str, analysis_data: Dict[str, Any], credibility_level: str) -> int:
        """Сохранить статью в базу данных"""
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
                    cursor.execute("""
                        UPDATE articles
                        SET title = ?,
                            source = ?,
                            content = ?,
                            short_summary = ?,
                            analysis_data = ?,
                            credibility_level = ?,
                            updated_at = ?
                        WHERE url = ?
                    """, (
                        title,
                        source,
                        content,
                        short_summary,
                        json.dumps(analysis_data),
                        credibility_level,
                        current_time,
                        url
                    ))
                    article_id = existing_article['id']
                else:
                    # Вставляем новую статью
                    cursor.execute("""
                        INSERT INTO articles
                        (title, source, url, content, short_summary, analysis_data, credibility_level, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        title,
                        source,
                        url,
                        content,
                        short_summary,
                        json.dumps(analysis_data),
                        credibility_level,
                        current_time,
                        current_time
                    ))
                    article_id = cursor.lastrowid

                # Обновляем статистику источника
                self.update_source_stats(source, credibility_level)

                conn.commit()
                return article_id

        except Exception as e:
            logger.error(f"Error saving article: {str(e)}", exc_info=True)
            raise

    def update_article(self, article_id: int, **kwargs) -> bool:
        """Обновить данные статьи"""
        if not kwargs:
            return False

        valid_fields = {'title', 'source', 'url', 'content', 'short_summary',
                       'analysis_data', 'credibility_level'}

        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                # Получаем текущие данные статьи
                cursor.execute("SELECT * FROM articles WHERE id = ?", (article_id,))
                current_article = cursor.fetchone()

                if not current_article:
                    return False

                # Проверяем, что обновляются только допустимые поля
                update_fields = {k: v for k, v in kwargs.items() if k in valid_fields}
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

                params.append(datetime.now().isoformat())
                params.append(article_id)

                # Добавляем обновление updated_at
                set_clauses.append("updated_at = ?")

                query = f"UPDATE articles SET {', '.join(set_clauses)} WHERE id = ?"
                cursor.execute(query, params)

                # Если изменился источник или уровень достоверности, обновляем статистику
                if 'source' in update_fields or 'credibility_level' in update_fields:
                    source = update_fields.get('source', current_article['source'])
                    credibility_level = update_fields.get('credibility_level', current_article['credibility_level'])

                    # Сначала откатываем старую статистику
                    old_source = current_article['source']
                    old_credibility = current_article['credibility_level']

                    if old_source != source or old_credibility != credibility_level:
                        # Удаляем старую запись из статистики
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

                conn.commit()
                return cursor.rowcount > 0

        except Exception as e:
            logger.error(f"Error updating article: {str(e)}", exc_info=True)
            return False

    def _update_source_stats(self, cursor: sqlite3.Cursor, source: str, credibility_level: str) -> None:
        """Внутренний метод для обновления статистики источников"""
        current_time = datetime.now().isoformat()

        cursor.execute("SELECT * FROM source_stats WHERE source = ?", (source,))
        existing = cursor.fetchone()

        if existing:
            if credibility_level == "High":
                cursor.execute("""
                    UPDATE source_stats
                    SET high = high + 1,
                        total_analyzed = total_analyzed + 1,
                        is_initial = 0,
                        last_updated = ?
                    WHERE source = ?
                """, (current_time, source))
            elif credibility_level == "Medium":
                cursor.execute("""
                    UPDATE source_stats
                    SET medium = medium + 1,
                        total_analyzed = total_analyzed + 1,
                        is_initial = 0,
                        last_updated = ?
                    WHERE source = ?
                """, (current_time, source))
            else:
                cursor.execute("""
                    UPDATE source_stats
                    SET low = low + 1,
                        total_analyzed = total_analyzed + 1,
                        is_initial = 0,
                        last_updated = ?
                    WHERE source = ?
                """, (current_time, source))
        else:
            if credibility_level == "High":
                cursor.execute("""
                    INSERT INTO source_stats
                    (source, high, medium, low, total_analyzed, is_initial, last_updated)
                    VALUES (?, 1, 0, 0, 1, 0, ?)
                """, (source, current_time))
            elif credibility_level == "Medium":
                cursor.execute("""
                    INSERT INTO source_stats
                    (source, high, medium, low, total_analyzed, is_initial, last_updated)
                    VALUES (?, 0, 1, 0, 1, 0, ?)
                """, (source, current_time))
            else:
                cursor.execute("""
                    INSERT INTO source_stats
                    (source, high, medium, low, total_analyzed, is_initial, last_updated)
                    VALUES (?, 0, 0, 1, 1, 0, ?)
                """, (source, current_time))

    def update_source_stats(self, source: str, credibility_level: str) -> None:
        """Обновить статистику источников"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                self._update_source_stats(cursor, source, credibility_level)
                conn.commit()
        except Exception as e:
            logger.error(f"Error updating source stats: {str(e)}", exc_info=True)
            raise

    def get_daily_buzz(self) -> Dict[str, Any]:
        """Получить статью дня"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM articles
                    WHERE title = ?
                    LIMIT 1
                """, (self.FEATURED_ARTICLE_TITLE,))

                article = cursor.fetchone()
                if not article:
                    return {
                        'status': 'error',
                        'message': 'Daily buzz article not found'
                    }

                article_dict = dict(article)
                article_dict['analysis'] = json.loads(article_dict['analysis_data']) if article_dict['analysis_data'] else {}

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

                return {
                    'status': 'success',
                    'data': {
                        'sources': sources,
                        'credibility_scores': credibility_scores
                    }
                }
        except Exception as e:
            logger.error(f"Error getting source credibility chart: {str(e)}", exc_info=True)
            return {
                'status': 'error',
                'message': 'Failed to retrieve source credibility chart'
            }

    def get_updated_source_credibility_chart(self) -> Dict[str, Any]:
        """Получить обновленные данные для чарта достоверности источников"""
        return self.get_source_credibility_chart()

    def get_source_statistics(self) -> Dict[str, Any]:
        """Получить статистику по всем источникам"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

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

                return {
                    'status': 'success',
                    'data': stats
                }
        except Exception as e:
            logger.error(f"Error getting source statistics: {str(e)}", exc_info=True)
            return {
                'status': 'error',
                'message': 'Failed to retrieve source statistics'
            }

    def get_analysis_history(self, limit: int = 10, offset: int = 0) -> Dict[str, Any]:
        """Получить историю анализа с пагинацией"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT
                        id, title, source, short_summary, credibility_level,
                        created_at, url
                    FROM articles
                    ORDER BY created_at DESC
                    LIMIT ? OFFSET ?
                """, (limit, offset))

                articles = []
                for article in cursor.fetchall():
                    articles.append({
                        'id': article['id'],
                        'title': article['title'],
                        'source': article['source'],
                        'summary': article['short_summary'],
                        'credibility': article['credibility_level'],
                        'date': article['created_at'],
                        'url': article['url']
                    })

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
        """Получить похожие статьи на основе тем с учетом достоверности"""
        try:
            if not topics:
                return []

            with self.get_connection() as conn:
                cursor = conn.cursor()

                # Создаем запрос с учетом приоритета достоверности
                conditions = []
                params = []

                for topic in topics:
                    conditions.append("(short_summary LIKE ? OR content LIKE ?)")
                    params.extend([f"%{topic}%", f"%{topic}%"])

                query = f"""
                    SELECT
                        id, title, source, short_summary, credibility_level, url, created_at
                    FROM articles
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

                return [{
                    'id': article['id'],
                    'title': article['title'],
                    'source': article['source'],
                    'summary': article['short_summary'],
                    'credibility': article['credibility_level'],
                    'url': article['url'],
                    'date': article['created_at']
                } for article in cursor.fetchall()]

        except Exception as e:
            logger.error(f"Error getting similar articles: {str(e)}", exc_info=True)
            return []

    def search_articles(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Поиск статей по запросу"""
        try:
            if not query:
                return []

            search_term = f"%{query}%"

            with self.get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT
                        id, title, source, short_summary, credibility_level, url, created_at
                    FROM articles
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

                return [{
                    'id': article['id'],
                    'title': article['title'],
                    'source': article['source'],
                    'summary': article['short_summary'],
                    'credibility': article['credibility_level'],
                    'url': article['url'],
                    'date': article['created_at']
                } for article in cursor.fetchall()]

        except Exception as e:
            logger.error(f"Error searching articles: {str(e)}", exc_info=True)
            return []

    def get_article_analysis_history(self, article_id: int) -> List[Dict[str, Any]]:
        """Получить историю анализа для конкретной статьи"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

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

    def save_analysis_history(self, article_id: int, analysis_data: Dict[str, Any]) -> bool:
        """Сохранить историю анализа для статьи"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    INSERT INTO analysis_history
                    (article_id, analysis_data)
                    VALUES (?, ?)
                """, (article_id, json.dumps(analysis_data)))

                conn.commit()
                return True

        except Exception as e:
            logger.error(f"Error saving analysis history: {str(e)}", exc_info=True)
            return False

    def get_articles_by_source(self, source: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Получить статьи от конкретного источника"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT
                        id, title, short_summary, credibility_level, url, created_at
                    FROM articles
                    WHERE source = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (source, limit))

                return [{
                    'id': article['id'],
                    'title': article['title'],
                    'summary': article['short_summary'],
                    'credibility': article['credibility_level'],
                    'url': article['url'],
                    'date': article['created_at']
                } for article in cursor.fetchall()]

        except Exception as e:
            logger.error(f"Error getting articles by source: {str(e)}", exc_info=True)
            return []

    def get_credibility_distribution(self) -> Dict[str, Any]:
        """Получить распределение статей по уровням достоверности"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

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
                    distribution[row['credibility_level']] = row['count']
                    total += row['count']

                # Добавляем проценты
                for level in distribution:
                    distribution[level] = {
                        'count': distribution[level],
                        'percentage': round(distribution[level] / total * 100, 1) if total > 0 else 0
                    }

                return {
                    'status': 'success',
                    'data': distribution,
                    'total': total
                }

        except Exception as e:
            logger.error(f"Error getting credibility distribution: {str(e)}", exc_info=True)
            return {
                'status': 'error',
                'message': 'Failed to retrieve credibility distribution'
            }

    def get_recent_articles(self, days: int = 7, limit: int = 10) -> List[Dict[str, Any]]:
        """Получить недавние статьи за указанное количество дней"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT
                        id, title, source, short_summary, credibility_level, url, created_at
                    FROM articles
                    WHERE created_at >= datetime('now', ?)
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (f"-{days} days", limit))

                return [{
                    'id': article['id'],
                    'title': article['title'],
                    'source': article['source'],
                    'summary': article['short_summary'],
                    'credibility': article['credibility_level'],
                    'url': article['url'],
                    'date': article['created_at']
                } for article in cursor.fetchall()]

        except Exception as e:
            logger.error(f"Error getting recent articles: {str(e)}", exc_info=True)
            return []
