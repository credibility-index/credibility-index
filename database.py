import os
import sqlite3
import json
import logging
from typing import Optional, List, Dict, Any, Tuple
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

                cursor.execute("SELECT COUNT(*) FROM articles WHERE title = ?", (self.FEATURED_ARTICLE_TITLE,))
                if cursor.fetchone()[0] == 0:
                    self._add_featured_article(conn)

                conn.commit()
                logger.info("Database schema initialized successfully")

        except Exception:
            logger.exception("Error initializing database schema")
            raise

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
        except Exception:
            logger.exception("Error getting database connection")
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

        except Exception:
            logger.exception("Error getting daily buzz")
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

        except Exception:
            logger.exception("Error getting source credibility chart")
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

        except Exception:
            logger.exception("Error getting analysis history")
            return {
                'status': 'error',
                'message': 'Failed to retrieve analysis history'
            }

    def analyze_article(self, input_text: str, source_name_manual: str = '') -> Dict[str, Any]:
        """Анализировать статью и сохранить результаты"""
        try:
            if input_text.startswith(('http://', 'https://')):
                content, source, title = self._extract_text_from_url(input_text)
                if not content:
                    return {
                        'status': 'error',
                        'message': 'Could not extract article content'
                    }
            else:
                content = input_text
                source = source_name_manual if source_name_manual else 'Direct Input'
                title = 'User-provided Text'

            analysis = self._analyze_with_claude(content, source)
            credibility_level = self._determine_credibility_level(analysis.get('index_of_credibility', 0.0))

            with self.get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    INSERT INTO articles
                    (title, source, url, content, short_summary, analysis_data, credibility_level)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    title,
                    source,
                    input_text if input_text.startswith(('http://', 'https://')) else None,
                    content,
                    content[:200] + '...' if len(content) > 200 else content,
                    json.dumps(analysis),
                    credibility_level
                ))

                article_id = cursor.lastrowid
                self._update_source_stats(source, credibility_level, conn)

                cursor.execute("""
                    INSERT INTO analysis_history
                    (article_id, analysis_data)
                    VALUES (?, ?)
                """, (
                    article_id,
                    json.dumps(analysis)
                ))

                conn.commit()

            similar_articles = self._get_similar_articles(analysis.get('topics', []))

            return {
                'status': 'success',
                'article': {
                    'id': article_id,
                    'title': title,
                    'source': source,
                    'url': input_text if input_text.startswith(('http://', 'https://')) else None,
                    'short_summary': content[:200] + '...' if len(content) > 200 else content,
                    'analysis': analysis,
                    'credibility_level': credibility_level
                },
                'same_topic_articles': similar_articles
            }

        except Exception:
            logger.exception("Error analyzing article")
            return {
                'status': 'error',
                'message': 'Failed to analyze article'
            }

    def _extract_text_from_url(self, url: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Извлечь текст из URL"""
        try:
            from newspaper import Article, Config

            user_agent = ('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                          'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')
            config = Config()
            config.browser_user_agent = user_agent
            config.request_timeout = 30

            parsed = urlparse(url)
            clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"

            if any(domain in url for domain in ['youtube.com', 'vimeo.com']):
                return None, parsed.netloc.replace('www.', ''), "Video content detected"

            article = Article(clean_url, config=config)
            article.download()
            article.parse()

            if not article.text or len(article.text.strip()) < 100:
                return None, None, None

            domain = parsed.netloc.replace('www.', '')
            title = article.title.strip() if article.title else "No title"

            return article.text.strip(), domain, title

        except Exception:
            logger.exception(f"Error extracting article from {url}")
            return None, None, None

    def _analyze_with_claude(self, content: str, source: str) -> Dict[str, Any]:
        """Анализировать контент с помощью Claude API (заглушка)"""
        try:
            # Здесь должна быть ваша реализация анализа с помощью Claude API
            # Возвращаем моковые данные для примера
            return {
                "news_integrity": 0.85,
                "fact_check_needed_score": 0.15,
                "sentiment_score": 0.6,
                "bias_score": 0.2,
                "topics": ["Middle East", "Conflict", "International Relations"],
                "key_arguments": [
                    "Israel's security concerns",
                    "Iran's regional influence",
                    "International diplomatic efforts"
                ],
                "mentioned_facts": [
                    "Recent missile attacks",
                    "Diplomatic negotiations",
                    "Regional tensions"
                ],
                "author_purpose": "inform",
                "potential_biases_identified": ["Pro-Israel bias"],
                "short_summary": "Analysis of the current conflict between Israel and Iran",
                "index_of_credibility": 0.82,
                "western_perspective": {"summary": "Western viewpoint focusing on Israel's defense rights."},
                "iranian_perspective": {"summary": "Iranian viewpoint highlighting resistance to aggression."},
                "israeli_perspective": {"summary": "Israeli viewpoint emphasizing security threats."},
                "neutral_perspective": {"summary": "Neutral analysis suggesting risk of escalation."}
            }
        except Exception:
            logger.exception("Error analyzing content with Claude API")
            return {}

    def _determine_credibility_level(self, index: float) -> str:
        """Определить уровень достоверности по индексу"""
        if index >= 0.75:
            return "High"
        elif index >= 0.5:
            return "Medium"
        else:
            return "Low"

    def _update_source_stats(self, source: str, credibility_level: str, conn: sqlite3.Connection) -> None:
        """Обновить статистику источника"""
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM source_stats WHERE source = ?", (source,))
        stats = cursor.fetchone()

        if not stats:
            high = 1 if credibility_level == "High" else 0
            medium = 1 if credibility_level == "Medium" else 0
            low = 1 if credibility_level == "Low" else 0

            cursor.execute("""
                INSERT INTO source_stats (source, high, medium, low, total_analyzed)
                VALUES (?, ?, ?, ?, 1)
            """, (source, high, medium, low))
        else:
            high = stats['high'] + (1 if credibility_level == "High" else 0)
            medium = stats['medium'] + (1 if credibility_level == "Medium" else 0)
            low = stats['low'] + (1 if credibility_level == "Low" else 0)
            total = stats['total_analyzed'] + 1

            cursor.execute("""
                UPDATE source_stats
                SET high = ?, medium = ?, low = ?, total_analyzed = ?
                WHERE source = ?
            """, (high, medium, low, total, source))

    def _get_similar_articles(self, topics: List[str]) -> List[Dict[str, Any]]:
        """Получить статьи с похожими темами"""
        if not topics:
            return []

        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                placeholders = ','.join(['?'] * len(topics))
                query = f"""
                    SELECT id, title, source, short_summary, credibility_level, created_at
                    FROM articles
                    WHERE
                """

                # Поиск совпадений по ключевым словам в short_summary или content
                topic_clauses = []
                params = []
                for topic in topics:
                    topic_clauses.append("(short_summary LIKE ? OR content LIKE ?)")
                    params.extend([f"%{topic}%", f"%{topic}%"])

                query += " OR ".join(topic_clauses)
                query += " ORDER BY created_at DESC LIMIT 5"

                cursor.execute(query, params)
                rows = cursor.fetchall()

                similar = []
                for row in rows:
                    similar.append({
                        'id': row['id'],
                        'title': row['title'],
                        'source': row['source'],
                        'summary': row['short_summary'],
                        'credibility': row['credibility_level'],
                        'date': row['created_at']
                    })

                return similar

        except Exception:
            logger.exception("Error getting similar articles")
            return []

    def get_article_by_id(self, article_id: int) -> Optional[Dict[str, Any]]:
        """Получить статью по ID"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM articles WHERE id = ?", (article_id,))
                row = cursor.fetchone()
                if row:
                    article = dict(row)
                    article['analysis'] = json.loads(article['analysis_data']) if article['analysis_data'] else {}
                    return article
                return None
        except Exception:
            logger.exception(f"Error getting article by ID: {article_id}")
            return None
