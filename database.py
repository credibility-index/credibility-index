import os
import sqlite3
import json
import logging
from typing import Optional, List, Dict, Any
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
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS source_stats (
                        source TEXT PRIMARY KEY,
                        high INTEGER DEFAULT 0,
                        medium INTEGER DEFAULT 0,
                        low INTEGER DEFAULT 0,
                        total_analyzed INTEGER DEFAULT 0
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

                # Добавляем статью дня, если её нет
                cursor.execute("SELECT COUNT(*) FROM articles WHERE title = ?", (self.FEATURED_ARTICLE_TITLE,))
                if cursor.fetchone()[0] == 0:
                    self._add_featured_article(conn)

                conn.commit()
                logger.info("Database schema initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing database schema: {str(e)}", exc_info=True)
            raise

    def article_exists(self, title: str, url: Optional[str] = None) -> bool:
        """Проверить, существует ли статья с таким заголовком или URL"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                if url:
                    cursor.execute("""
                        SELECT COUNT(*) FROM articles WHERE url = ? OR title = ?
                    """, (url, title))
                else:
                    cursor.execute("""
                        SELECT COUNT(*) FROM articles WHERE title = ?
                    """, (title,))
                count = cursor.fetchone()[0]
                return count > 0
        except Exception as e:
            logger.error(f"Error checking if article exists: {str(e)}", exc_info=True)
            return False

    def _add_featured_article(self, conn: sqlite3.Connection) -> None:
        """Добавить статью дня в базу данных"""
        featured_article = {
            "title": self.FEATURED_ARTICLE_TITLE,
            "source": "Media Credibility Index",
            "url": "",
            "content": "Comprehensive analysis of the current Israel-Iran conflict...",
            "short_summary": "Analysis of the current situation between Israel and Iran",
            "analysis_data": json.dumps({
                "news_integrity": 0.85,
                "index_of_credibility": 0.82,
                "sentiment_score": 0.6,
                "bias_score": 0.3,
                "topics": ["Middle East", "Conflict", "International Relations"],
                "key_arguments": [
                    "Israel's security concerns",
                    "Iran's regional influence",
                    "International diplomatic efforts"
                ],
                "western_perspective": {
                    "summary": "The Western perspective emphasizes Israel's right to defend itself and the need for regional stability."
                },
                "iranian_perspective": {
                    "summary": "Iran views the conflict as a response to Israeli aggression and support for Iranian sovereignty."
                },
                "israeli_perspective": {
                    "summary": "Israel sees the conflict as necessary for its national security against Iranian threats."
                },
                "neutral_perspective": {
                    "summary": "A neutral analysis suggests both sides have legitimate concerns but escalation risks regional stability."
                }
            }),
            "credibility_level": "High"
        }

        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO articles
            (title, source, url, content, short_summary, analysis_data, credibility_level)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            featured_article["title"],
            featured_article["source"],
            featured_article["url"],
            featured_article["content"],
            featured_article["short_summary"],
            featured_article["analysis_data"],
            featured_article["credibility_level"]
        ))

    def get_connection(self) -> sqlite3.Connection:
        """Получить соединение с базой данных"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            return conn
        except Exception as e:
            logger.error(f"Error getting database connection: {str(e)}", exc_info=True)
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
def _initialize_with_mock_data(self) -> None:
    """Инициализировать базу данных с моковыми данными"""
    try:
        mock_sources = [
            ('BBC News', 'High'),
            ('Reuters', 'High'),
            ('The New York Times', 'High'),
            ('The Guardian', 'High'),
            ('CNN', 'Medium'),
            ('Fox News', 'Medium'),
            ('Al Jazeera', 'Medium'),
            ('RT', 'Low'),
            ('Breitbart', 'Low'),
            ('Daily Mail', 'Low')
        ]

        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Проверяем, есть ли уже данные в source_stats
            cursor.execute("SELECT COUNT(*) FROM source_stats")
            count = cursor.fetchone()[0]

            if count == 0:
                # Добавляем моковые данные, если таблица пуста
                for source, credibility in mock_sources:
                    if credibility == "High":
                        cursor.execute("""
                            INSERT INTO source_stats (source, high, medium, low, total_analyzed)
                            VALUES (?, 5, 1, 0, 6)
                        """, (source,))
                    elif credibility == "Medium":
                        cursor.execute("""
                            INSERT INTO source_stats (source, high, medium, low, total_analyzed)
                            VALUES (?, 2, 3, 1, 6)
                        """, (source,))
                    else:
                        cursor.execute("""
                            INSERT INTO source_stats (source, high, medium, low, total_analyzed)
                            VALUES (?, 1, 1, 4, 6)
                        """, (source,))

                conn.commit()
                logger.info("Database initialized with mock data successfully")
    except Exception as e:
        logger.error(f"Error initializing database with mock data: {str(e)}", exc_info=True)
        raise

    def get_source_credibility_chart(self) -> Dict[str, Any]:
        """Получить данные для чарта достоверности источников"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT source, high, medium, low
                    FROM source_stats
                    ORDER BY (high + medium + low) DESC
                """)

                data = cursor.fetchall()
                sources = []
                credibility_scores = []

                for row in data:
                    source = row['source']
                    high = row['high']
                    medium = row['medium']
                    low = row['low']
                    total = high + medium + low
                    score = (high * 1.0 + medium * 0.5) / total if total > 0 else 0.5
                    sources.append(source)
                    credibility_scores.append(round(score, 2))

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

    def get_analysis_history(self, limit: int = 5) -> Dict[str, Any]:
        """Получить историю анализа"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, title, source, short_summary, credibility_level, created_at
                    FROM articles
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (limit,))

                articles = cursor.fetchall()
                articles_list = []

                for article in articles:
                    articles_list.append({
                        'id': article['id'],
                        'title': article['title'],
                        'source': article['source'],
                        'summary': article['short_summary'],
                        'credibility': article['credibility_level'],
                        'date': article['created_at']
                    })

                return {
                    'status': 'success',
                    'history': articles_list
                }

        except Exception as e:
            logger.error(f"Error getting analysis history: {str(e)}", exc_info=True)
            return {
                'status': 'error',
                'message': 'Failed to retrieve analysis history'
            }

    def save_article(self, title: str, source: str, url: Optional[str], content: str,
                     short_summary: str, analysis_data: Dict[str, Any], credibility_level: str) -> int:
        """Сохранить статью в базу данных"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO articles
                    (title, source, url, content, short_summary, analysis_data, credibility_level)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    title,
                    source,
                    url,
                    content,
                    short_summary,
                    json.dumps(analysis_data),
                    credibility_level
                ))
                article_id = cursor.lastrowid
                conn.commit()
                return article_id
        except Exception as e:
            logger.error(f"Error saving article: {str(e)}", exc_info=True)
            raise

    def update_source_stats(self, source: str, credibility_level: str) -> None:
        """Обновить статистику источников"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                # Проверяем, существует ли источник
                cursor.execute("SELECT * FROM source_stats WHERE source = ?", (source,))
                existing = cursor.fetchone()

                if existing:
                    # Обновляем существующий источник
                    if credibility_level == "High":
                        cursor.execute("""
                            UPDATE source_stats
                            SET high = high + 1, total_analyzed = total_analyzed + 1
                            WHERE source = ?
                        """, (source,))
                    elif credibility_level == "Medium":
                        cursor.execute("""
                            UPDATE source_stats
                            SET medium = medium + 1, total_analyzed = total_analyzed + 1
                            WHERE source = ?
                        """, (source,))
                    else:
                        cursor.execute("""
                            UPDATE source_stats
                            SET low = low + 1, total_analyzed = total_analyzed + 1
                            WHERE source = ?
                        """, (source,))
                else:
                    # Вставляем новый источник
                    if credibility_level == "High":
                        cursor.execute("""
                            INSERT INTO source_stats (source, high, medium, low, total_analyzed)
                            VALUES (?, 1, 0, 0, 1)
                        """, (source,))
                    elif credibility_level == "Medium":
                        cursor.execute("""
                            INSERT INTO source_stats (source, high, medium, low, total_analyzed)
                            VALUES (?, 0, 1, 0, 1)
                        """, (source,))
                    else:
                        cursor.execute("""
                            INSERT INTO source_stats (source, high, medium, low, total_analyzed)
                            VALUES (?, 0, 0, 1, 1)
                        """, (source,))

                conn.commit()

        except Exception as e:
            logger.error(f"Error updating source stats: {str(e)}", exc_info=True)
            raise

    def get_similar_articles(self, topics: List[str]) -> List[Dict[str, Any]]:
        """Получить похожие статьи на основе тем"""
        try:
            if not topics:
                return []

            with self.get_connection() as conn:
                cursor = conn.cursor()

                # Создаем запрос для поиска статей с похожими темами
                conditions = []
                params = []
                for topic in topics:
                    conditions.append("(short_summary LIKE ? OR content LIKE ?)")
                    params.extend([f"%{topic}%", f"%{topic}%"])

                query = f"""
                    SELECT id, title, source, short_summary, credibility_level, url
                    FROM articles
                    WHERE {' OR '.join(conditions)}
                    ORDER BY created_at DESC
                    LIMIT 5
                """

                cursor.execute(query, params)
                articles = cursor.fetchall()

                return [{
                    'id': article['id'],
                    'title': article['title'],
                    'source': article['source'],
                    'summary': article['short_summary'],
                    'credibility': article['credibility_level'],
                    'url': article['url']
                } for article in articles]

        except Exception as e:
            logger.error(f"Error getting similar articles: {str(e)}", exc_info=True)
            return []
