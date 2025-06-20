import os
import sqlite3
import json
import logging
from typing import Optional, List, Dict, Any

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Database:
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
                cursor.execute("SELECT COUNT(*) FROM articles WHERE title = 'Israel-Iran Conflict: Comprehensive Analysis'")
                if cursor.fetchone()[0] == 0:
                    self._add_featured_article(conn)

                conn.commit()
                logger.info("Database schema initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing database schema: {str(e)}")
            raise

    def _add_featured_article(self, conn: sqlite3.Connection) -> None:
        """Добавить статью дня в базу данных"""
        featured_article = {
            "title": "Israel-Iran Conflict: Comprehensive Analysis",
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
            logger.error(f"Error getting database connection: {str(e)}")
            raise

    def get_daily_buzz(self) -> Dict[str, Any]:
        """Получить статью дня"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM articles
                    WHERE title = 'Israel-Iran Conflict: Comprehensive Analysis'
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

                return {
                    'status': 'success',
                    'article': article_dict
                }

        except Exception as e:
            logger.error(f"Error getting daily buzz: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
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

                for source, high, medium, low in data:
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
            logger.error(f"Error getting source credibility chart: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }

    def get_analysis_history(self, limit: int = 5) -> Dict[str, Any]:
        """Получить историю анализа"""
        try:
            with self.get_connection() as conn:
                conn.row_factory = sqlite3.Row
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
            logger.error(f"Error getting analysis history: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }

    def analyze_article(self, input_text: str, source_name_manual: str = '') -> Dict[str, Any]:
        """Анализировать статью и сохранить результаты"""
        try:
            # Извлечение контента статьи
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

            # Анализ с помощью Claude (здесь должна быть ваша реализация)
            analysis = self._analyze_with_claude(content, source)

            # Определение уровня достоверности
            credibility_level = self._determine_credibility_level(analysis['index_of_credibility'])

            # Сохранение в базу данных
            with self.get_connection() as conn:
                cursor = conn.cursor()

                # Сохранение статьи
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

                # Обновление статистики источников
                self._update_source_stats(source, credibility_level)

                # Сохранение в историю анализа
                cursor.execute("""
                    INSERT INTO analysis_history
                    (article_id, analysis_data)
                    VALUES (?, ?)
                """, (
                    article_id,
                    json.dumps(analysis)
                ))

                conn.commit()

                # Получение похожих статей
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

        except Exception as e:
            logger.error(f"Error analyzing article: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }

    def _extract_text_from_url(self, url: str) -> tuple:
        """Извлечь текст из URL"""
        try:
            from newspaper import Article, Config
            user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
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

        except Exception as e:
            logger.error(f"Error extracting article from {url}: {str(e)}")
            return None, None, None

    def _analyze_with_claude(self, content: str, source: str) -> dict:
        """Анализировать контент с помощью Claude API"""
        try:
            import anthropic
            from urllib.parse import urlparse

            # Здесь должна быть ваша реализация анализа с помощью Claude API
            # Для примера возвращаем моковые данные

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
            }

        except Exception as e:
            logger.error(f"Error analyzing with Claude: {str(e)}")
            return {
                "news_integrity": 0.7,
                "fact_check_needed_score": 0.3,
                "sentiment_score": 0.5,
                "bias_score": 0.4,
                "topics": ["default"],
                "key_arguments": ["default"],
                "mentioned_facts": ["default"],
                "author_purpose": "inform",
                "potential_biases_identified": ["none"],
                "short_summary": "Default summary",
                "index_of_credibility": 0.6,
                "western_perspective": {"summary": "Default western perspective"},
                "iranian_perspective": {"summary": "Default iranian perspective"},
                "israeli_perspective": {"summary": "Default israeli perspective"},
                "neutral_perspective": {"summary": "Default neutral perspective"}
            }

    def _determine_credibility_level(self, score: float) -> str:
        """Определить уровень достоверности на основе оценки"""
        if score >= 0.8:
            return "High"
        elif score >= 0.6:
            return "Medium"
        else:
            return "Low"

    def _update_source_stats(self, source: str, credibility_level: str) -> None:
        """Обновить статистику источников в базе данных"""
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
            logger.error(f"Error updating source stats: {str(e)}")

    def _get_similar_articles(self, topics: List[str]) -> List[Dict[str, Any]]:
        """Получить похожие статьи на основе тем"""
        try:
            if not topics or len(topics) == 0:
                return []

            # Для демонстрации возвращаем моковые данные
            # В реальном приложении вы бы запросили News API или аналогичный сервис
            mock_articles = [
                {
                    "title": "Israel and Iran tensions escalate in Middle East",
                    "source": "CNN",
                    "url": "https://edition.cnn.com/world/live-news/israel-iran-conflict-06-20-25-intl-hnk",
                    "description": "The latest developments in the ongoing conflict between Israel and Iran",
                    "credibility": "High"
                },
                {
                    "title": "International response to Israel-Iran conflict",
                    "source": "Aljazeera",
                    "url": "https://www.aljazeera.com/news/liveblog/2025/6/20/live-iran-israel-continue-missile-fire-irans-fm-to-meet-eu-counterparts",
                    "description": "How world leaders are responding to the escalating tensions",
                    "credibility": "High"
                },
                {
                    "title": "Historical context of Israel-Iran relations",
                    "source": "Washington Post",
                    "url": "https://www.washingtonpost.com/world/2025/06/19/iran-israel-conflict-history/",
                    "description": "A look at the history behind the current conflict",
                    "credibility": "High"
                }
            ]

            return mock_articles

        except Exception as e:
            logger.error(f"Error getting similar articles: {str(e)}")
            return []
