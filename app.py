from flask import Flask, render_template, request, jsonify, abort, send_from_directory
import os
import json
import sqlite3
import requests
import anthropic
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from datetime import datetime, timedelta, UTC
import logging
from logging.handlers import RotatingFileHandler
from werkzeug.middleware.proxy_fix import ProxyFix
import re
import plotly.graph_objects as go
from stop_words import get_stop_words
from newspaper import Article
import html

# Настройка приложения Flask
app = Flask(__name__, static_folder='static', template_folder='templates')
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

# Настройка логирования
def setup_logging():
    """Настройка системы логирования."""
    file_handler = RotatingFileHandler('app.log', maxBytes=1024*1024, backupCount=5)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    app.logger.addHandler(file_handler)
    app.logger.addHandler(console_handler)
    app.logger.setLevel(logging.INFO)

    class RequestFilter(logging.Filter):
        def filter(self, record):
            return not (hasattr(record, 'msg') and '404 Not Found' in record.msg)

    logging.getLogger('werkzeug').addFilter(RequestFilter())
    logging.getLogger('werkzeug').setLevel(logging.WARNING)

# Загрузка переменных окружения
load_dotenv()
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
NEWS_API_KEY = os.getenv('NEWS_API_KEY')
NEWS_API_ENABLED = bool(NEWS_API_KEY)
MODEL_NAME = "claude-3-opus-20240229"

if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY is missing! Please set it in your .env file.")
if not NEWS_API_KEY:
    app.logger.warning("NEWS_API_KEY is missing! Similar news functionality will be disabled.")
    NEWS_API_ENABLED = False

# Настройка базы данных
DB_NAME = 'news_analysis.db'

# Начальные данные для надежности источников
INITIAL_SOURCE_COUNTS = {
    "bbc.com": {"high": 15, "medium": 5, "low": 1},
    "reuters.com": {"high": 20, "medium": 3, "low": 0},
    "foxnews.com": {"high": 3, "medium": 7, "low": 15},
    "cnn.com": {"high": 5, "medium": 10, "low": 5},
    "nytimes.com": {"high": 10, "medium": 5, "low": 2},
    "theguardian.com": {"high": 12, "medium": 4, "low": 1},
    "apnews.com": {"high": 18, "medium": 2, "low": 0}
}

# Соответствие доменов и владельцев СМИ
media_owners = {
    "bbc.com": "BBC",
    "reuters.com": "Thomson Reuters",
    "foxnews.com": "Fox Corporation",
    "cnn.com": "Warner Bros. Discovery",
    "nytimes.com": "The New York Times Company",
    "theguardian.com": "Guardian Media Group",
    "apnews.com": "Associated Press",
    "wsj.com": "News Corp",
    "aljazeera.com": "Al Jazeera Media Network"
}

# ID источников NewsAPI
TRUSTED_NEWS_SOURCES_IDS = [
    "bbc-news", "reuters", "associated-press", "the-new-york-times",
    "the-guardian-uk", "the-wall-street-journal", "cnn", "al-jazeera-english"
]

stop_words_en = get_stop_words('en')

# Настройка логирования
setup_logging()

# Middleware для блокировки WordPress-сканеров
@app.before_request
def block_wordpress_scanners():
    wordpress_paths = [
        'wp-admin', 'wp-includes', 'wp-content', 'xmlrpc.php',
        'wp-login.php', 'wp-config.php', 'readme.html', 'license.txt'
    ]
    if any(path in request.path.lower() for path in wordpress_paths):
        app.logger.warning(f"Blocked WordPress scanner request from {request.remote_addr}")
        return abort(404)

# Middleware для безопасности
@app.after_request
def add_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    response.headers['Content-Security-Policy'] = "default-src 'self'"
    return response

# Обработчик 404 ошибок
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

# Функции базы данных
def ensure_db_schema():
    """Создает схему базы данных."""
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS news (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
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
            analysis_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            short_summary TEXT
        )''')
        c.execute('''CREATE TABLE IF NOT EXISTS source_stats (
            source TEXT PRIMARY KEY,
            high INTEGER DEFAULT 0,
            medium INTEGER DEFAULT 0,
            low INTEGER DEFAULT 0,
            total_analyzed INTEGER DEFAULT 0
        )''')
        conn.commit()
        app.logger.info("Database schema ensured successfully")
    except sqlite3.Error as e:
        app.logger.error(f"Error ensuring database schema: {str(e)}")
        raise
    finally:
        if conn:
            conn.close()

def initialize_sources(initial_counts):
    """Инициализирует данные источников."""
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        for source, counts in initial_counts.items():
            c.execute("SELECT total_analyzed FROM source_stats WHERE source = ?", (source,))
            row = c.fetchone()
            if row is None:
                high = counts.get("high", 0)
                medium = counts.get("medium", 0)
                low = counts.get("low", 0)
                c.execute('''INSERT INTO source_stats (source, high, medium, low, total_analyzed)
                         VALUES (?, ?, ?, ?, ?)''',
                         (source, high, medium, low, high+medium+low))
        conn.commit()
        app.logger.info("Initial source initialization completed successfully")
    except sqlite3.Error as e:
        app.logger.error(f"Database error during source initialization: {str(e)}")
        raise
    finally:
        if conn:
            conn.close()

def check_database_integrity():
    """Проверяет целостность базы данных."""
    try:
        if not os.path.exists(DB_NAME):
            app.logger.error(f"Database file {DB_NAME} not found!")
            return False

        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()

        # Проверка таблиц
        c.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [table[0] for table in c.fetchall()]
        required_tables = ['news', 'source_stats']
        for table in required_tables:
            if table not in tables:
                app.logger.error(f"Critical table '{table}' is missing!")
                return False

        # Проверка структуры таблиц
        for table, columns in [('news', ['id', 'title', 'source', 'content', 'integrity',
                                      'fact_check', 'sentiment', 'bias', 'credibility_level',
                                      'index_of_credibility', 'url', 'analysis_date', 'short_summary']),
                             ('source_stats', ['source', 'high', 'medium', 'low', 'total_analyzed'])]:
            c.execute(f"PRAGMA table_info({table})")
            columns_in_table = [row[1] for row in c.fetchall()]
            for column in columns:
                if column not in columns_in_table:
                    app.logger.error(f"Critical column '{column}' is missing in '{table}' table!")
                    return False

        return True
    except Exception as e:
        app.logger.error(f"Error during database check: {str(e)}")
        return False
    finally:
        if conn:
            conn.close()

# Класс для анализа новостей
class ClaudeNewsAnalyzer:
    def __init__(self, api_key, model_name):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model_name = model_name

    def analyze_article_text(self, article_text_content, source_name_for_context):
        """Анализирует текст статьи с использованием API Claude."""
        max_chars_for_claude = 10000
        if len(article_text_content) > max_chars_for_claude:
            article_text_content = article_text_content[:max_chars_for_claude]

        media_owner_display = media_owners.get(source_name_for_context, "Unknown Owner")

        prompt = f"""Вы - аналитический помощник, специализирующийся на анализе новостных статей.
Проанализируйте предоставленную статью и верните результаты в формате JSON.

Текст статьи:
\"\"\"
{article_text_content}
\"\"\"

Источник: {source_name_for_context}
Владелец СМИ: {media_owner_display}

Верните результаты в формате JSON с обязательными полями:
1. news_integrity (0.0-1.0) - целостность информации
2. fact_check_needed_score (0.0-1.0) - необходимость проверки фактов
3. sentiment_score (0.0-1.0) - эмоциональный тон
4. bias_score (0.0-1.0) - степень предвзятости
5. index_of_credibility (0.0-1.0) - общий индекс достоверности
6. short_summary - краткое содержание статьи
7. topics - основные темы статьи
8. key_arguments - ключевые аргументы автора
9. mentioned_facts - упомянутые факты
10. potential_biases_identified - возможные предвзятости"""

        try:
            message = self.client.messages.create(
                model=self.model_name,
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}]
            )
            return json.loads(message.content[0].text)
        except Exception as e:
            app.logger.error(f"Error during Claude analysis: {str(e)}")
            raise

def extract_text_from_url(url):
    """Извлекает текст из URL."""
    try:
        clean_url = re.sub(r'/amp(/)?$', '', url)
        article = Article(clean_url)
        article.download()
        article.parse()
        text = article.text.strip()
        title = article.title.strip() if article.title else ""
        source = urlparse(clean_url).netloc.replace("www.", "")
        return text, source, title
    except Exception as e:
        app.logger.error(f"Error extracting article from URL {url}: {str(e)}")
        return "", "", ""

def calculate_credibility_level(analysis_result):
    """Вычисляет уровень достоверности на основе анализа."""
    integrity = analysis_result.get('news_integrity', 0.0)
    fact_check = analysis_result.get('fact_check_needed_score', 1.0)
    sentiment = analysis_result.get('sentiment_score', 0.5)
    bias = analysis_result.get('bias_score', 1.0)

    # Расчет индекса достоверности
    index = (integrity * 0.45) + ((1.0 - fact_check) * 0.35) + ((1.0 - abs(sentiment - 0.5) * 2) * 0.10) + ((1.0 - bias) * 0.10)

    if index >= 0.75:
        return 'High', index
    elif index >= 0.5:
        return 'Medium', index
    else:
        return 'Low', index

def save_analysis_to_db(url, title, source, content, analysis_result):
    """Сохраняет результаты анализа в базу данных."""
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()

        # Расчет уровня достоверности
        credibility_level, index_of_credibility = calculate_credibility_level(analysis_result)

        c.execute('''INSERT INTO news (url, title, source, content, integrity, fact_check, sentiment, bias,
                     credibility_level, index_of_credibility, short_summary)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                (url, title, source, content,
                 analysis_result.get('news_integrity', 0.0),
                 analysis_result.get('fact_check_needed_score', 1.0),
                 analysis_result.get('sentiment_score', 0.5),
                 analysis_result.get('bias_score', 1.0),
                 credibility_level, index_of_credibility,
                 analysis_result.get('short_summary', 'No summary')))

        # Обновление статистики источников
        c.execute("SELECT high, medium, low FROM source_stats WHERE source = ?", (source,))
        row = c.fetchone()
        if row:
            high, medium, low = row
            if credibility_level == 'High': high += 1
            elif credibility_level == 'Medium': medium += 1
            else: low += 1
            c.execute('''UPDATE source_stats SET high=?, medium=?, low=? WHERE source=?''',
                    (high, medium, low, source))
        else:
            high = 1 if credibility_level == 'High' else 0
            medium = 1 if credibility_level == 'Medium' else 0
            low = 1 if credibility_level == 'Low' else 0
            c.execute('''INSERT INTO source_stats (source, high, medium, low, total_analyzed)
                        VALUES (?, ?, ?, ?, 1)''', (source, high, medium, low))

        conn.commit()
        return credibility_level, index_of_credibility
    except Exception as e:
        app.logger.error(f"Error saving analysis to database: {str(e)}")
        raise
    finally:
        if conn:
            conn.close()

def process_article_analysis(input_text, source_name_manual):
    """Обрабатывает анализ статьи и возвращает индекс достоверности."""
    if input_text.strip().startswith("http"):
        article_url = input_text.strip()
        content, source, title = extract_text_from_url(article_url)
        if not content:
            return "Ошибка извлечения контента", None, None
    else:
        content = input_text
        source = source_name_manual if source_name_manual else "Direct Input"
        title = "User-provided Text"

    if len(content) < 100:
        return "Слишком короткий текст", None, None

    analyzer = ClaudeNewsAnalyzer(ANTHROPIC_API_KEY, MODEL_NAME)
    try:
        analysis_result = analyzer.analyze_article_text(content, source)
        credibility_level, index_of_credibility = save_analysis_to_db(
            input_text if input_text.startswith("http") else None,
            title, source, content, analysis_result
        )

        # Формирование результатов
        output_md = f"""### Анализ статьи: {title}
Источник: {source}
Уровень достоверности: {credibility_level} ({index_of_credibility:.2f})

Основные показатели:
- Целостность информации: {analysis_result.get('news_integrity', 0.0):.2f}
- Необходимость проверки фактов: {1 - analysis_result.get('fact_check_needed_score', 1.0):.2f}
- Эмоциональный тон: {analysis_result.get('sentiment_score', 0.5):.2f}
- Предвзятость: {1 - analysis_result.get('bias_score', 1.0):.2f}
- Индекс достоверности: {index_of_credibility:.2f}

Краткое содержание:
{analysis_result.get('short_summary', 'Нет краткого содержания')}

Основные темы:
{', '.join(analysis_result.get('topics', []))}"""

        scores_for_chart = {
            "Integrity": analysis_result.get('news_integrity', 0.0) * 100,
            "Factuality": (1 - analysis_result.get('fact_check_needed_score', 1.0)) * 100,
            "Sentiment": analysis_result.get('sentiment_score', 0.5) * 100,
            "Bias": (1 - analysis_result.get('bias_score', 1.0)) * 100,
            "Credibility": index_of_credibility * 100
        }

        return output_md, scores_for_chart, analysis_result
    except Exception as e:
        return f"Ошибка анализа: {str(e)}", None, None

def generate_query(analysis_result):
    """Генерирует запрос для поиска похожих новостей."""
    topics = analysis_result.get('topics', [])
    key_arguments = analysis_result.get('key_arguments', [])
    mentioned_facts = analysis_result.get('mentioned_facts', [])

    all_terms = []
    for phrase_list in [topics, key_arguments]:
        for phrase in phrase_list:
            if not phrase.strip(): continue
            if ' ' in phrase.strip() and len(phrase.strip().split()) > 1:
                all_terms.append(f'"{phrase.strip()}"')
            else:
                all_terms.append(phrase.strip())

    for fact in mentioned_facts:
        if not fact.strip(): continue
        words = [word for word in fact.lower().split() if word not in stop_words_en and len(word) > 2]
        all_terms.extend(words)

    unique_terms = list(set(all_terms))

    if len(unique_terms) >= 3:
        query = " AND ".join(unique_terms)
    elif unique_terms:
        query = " OR ".join(unique_terms)
    else:
        query = "current events OR news"

    return query

def fetch_similar_news(analysis_result, days_range=7, max_articles=3):
    """Получает похожие новости с высоким индексом достоверности."""
    if not NEWS_API_ENABLED:
        return []

    initial_query = generate_query(analysis_result)

    # Определяем диапазон дат
    original_published_date_str = analysis_result.get('published_date', 'N/A')
    end_date = datetime.now(UTC).date()

    if original_published_date_str and original_published_date_str != 'N/A':
        try:
            parsed_date = datetime.strptime(original_published_date_str, '%Y-%m-%d').date()
            start_date = parsed_date - timedelta(days=days_range)
            end_date = parsed_date + timedelta(days=days_range)
        except ValueError:
            start_date = end_date - timedelta(days=days_range)
    else:
        start_date = end_date - timedelta(days=days_range)

    # Попытка 1: Специфический запрос с надежными источниками
    params_specific = {
        "q": initial_query,
        "apiKey": NEWS_API_KEY,
        "language": "en",
        "pageSize": max_articles * 2,
        "sortBy": "relevancy",
        "from": start_date.strftime('%Y-%m-%d'),
        "to": end_date.strftime('%Y-%m-%d'),
    }

    if TRUSTED_NEWS_SOURCES_IDS:
        params_specific["sources"] = ",".join(TRUSTED_NEWS_SOURCES_IDS)

    articles_found = []
    try:
        response = requests.get("https://newsapi.org/v2/everything", params=params_specific, timeout=15)
        response.raise_for_status()
        articles_found = response.json().get("articles", [])
    except Exception as e:
        app.logger.error(f"Error fetching similar news: {str(e)}")

    # Попытка 2: Более широкий запрос, если первая попытка дала мало результатов
    if len(articles_found) < max_articles and initial_query != "current events OR news":
        broader_query_terms = list(set(analysis_result.get('topics', [])[:3]))
        broader_query = " OR ".join([f'"{term}"' if ' ' in term else term for term in broader_query_terms if term and term not in stop_words_en])

        if not broader_query:
            broader_query = "current events OR news"

        params_broad = {
            "q": broader_query,
            "apiKey": NEWS_API_KEY,
            "language": "en",
            "pageSize": max_articles * 2,
            "sortBy": "relevancy",
            "from": start_date.strftime('%Y-%m-%d'),
            "to": end_date.strftime('%Y-%m-%d'),
        }

        if TRUSTED_NEWS_SOURCES_IDS:
            params_broad["sources"] = ",".join(TRUSTED_NEWS_SOURCES_IDS)

        try:
            response = requests.get("https://newsapi.org/v2/everything", params=params_broad, timeout=15)
            response.raise_for_status()
            articles_found.extend(response.json().get("articles", []))
        except Exception as e:
            app.logger.error(f"Error fetching similar news: {str(e)}")

    # Удаление дубликатов
    unique_articles = {}
    for article in articles_found:
        if article.get('url'):
            unique_articles[article['url']] = article
    articles_found = list(unique_articles.values())

    # Ранжирование статей
    ranked_articles = []
    predefined_trust_scores = {
        "bbc.com": 0.9, "bbc.co.uk": 0.9, "reuters.com": 0.95, "apnews.com": 0.93,
        "nytimes.com": 0.88, "theguardian.com": 0.85, "wsj.com": 0.82,
        "cnn.com": 0.70, "foxnews.com": 0.40, "aljazeera.com": 0.80
    }

    all_query_terms = []
    if 'initial_query' in locals():
        all_query_terms.extend([t.lower().replace('"', '') for t in initial_query.split(' AND ')])
    if 'broader_query' in locals():
        all_query_terms.extend([t.lower().replace('"', '') for t in broader_query.split(' OR ')])
    all_query_terms = list(set([t for t in all_query_terms if t and t not in stop_words_en]))

    for article in articles_found:
        source_domain = urlparse(article.get("url", '')).netloc.replace('www.', '')
        trust_score = predefined_trust_scores.get(source_domain, 0.5)
        article_text = (article.get('title', '') + " " + article.get('description', '')).lower()
        relevance_score = sum(1 for term in all_query_terms if term in article_text)
        final_score = (relevance_score * 10) + (trust_score * 5)
        ranked_articles.append((article, final_score))

    ranked_articles.sort(key=lambda item: item[1], reverse=True)
    return [item[0] for item in ranked_articles[:max_articles]]

def render_similar_articles_html(articles):
    """Генерирует HTML для отображения похожих статей."""
    if not articles:
        return "<p>No similar articles found for the selected criteria.</p>"

    predefined_trust_scores = {
        "bbc.com": 0.9, "bbc.co.uk": 0.9, "reuters.com": 0.95, "apnews.com": 0.93,
        "nytimes.com": 0.88, "theguardian.com": 0.85, "wsj.com": 0.82,
        "cnn.com": 0.70, "foxnews.com": 0.40, "aljazeera.com": 0.80
    }

    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()

        html_items = []
        for art in articles:
            title = html.escape(art.get("title", "No Title"))
            article_url = html.escape(art.get("url", "#"))
            source_api_name = html.escape(art.get("source", {}).get("name", "Unknown Source"))

            published_at_raw = art.get('publishedAt', 'N/A')
            published_at_display = html.escape(published_at_raw.split('T')[0] if 'T' in published_at_raw and published_at_raw != 'N/A' else published_at_raw)

            description_raw = art.get('description', 'No description available.')
            if description_raw.startswith(art.get("title", "")):
                description_raw = description_raw[len(art.get("title", "")):].strip()
                if description_raw.startswith("- "):
                    description_raw = description_raw[2:].strip()
            description_display = html.escape(description_raw)

            domain = urlparse(art.get("url", "#")).netloc.replace('www.', '')
            trust_display = ""

            c.execute("SELECT high, medium, low, total_analyzed FROM source_stats WHERE source = ?", (domain,))
            row = c.fetchone()

            if row:
                high, medium, low, total_analyzed = row
                if total_analyzed > 0:
                    score = (high * 1.0 + medium * 0.5 + low * 0.0) / total_analyzed
                    trust_display = f" (Hist. Src. Credibility: {score*100:.0f}%)"

            if not trust_display and domain in predefined_trust_scores:
                predefined_score = predefined_trust_scores.get(domain)
                trust_display = f" (Est. Src. Trust: {predefined_score*100:.0f}%)"
            elif not trust_display:
                trust_display = " (Src. Credibility: N/A)"

            html_items.append(
                f"""
                <div class="similar-article">
                    <h4><a href="{article_url}" target="_blank" rel="noopener noreferrer">
                        {title}
                    </a></h4>
                    <p><strong>Source:</strong> {source_api_name}{trust_display} | <strong>Published:</strong> {published_at_display}</p>
                    <p>{description_display}</p>
                </div>
                <hr>
                """
            )

        return f"""
        <div class="similar-articles-container">
            <h3>🔗 Similar News Articles (Ranked by Relevance & Trust):</h3>
            {"".join(html_items)}
        </div>
        """
    except Exception as e:
        app.logger.error(f"Error rendering similar articles: {str(e)}")
        return "<p>Error retrieving similar articles data.</p>"
    finally:
        if conn:
            conn.close()

def get_source_reliability_data():
    """Получает данные о надежности источников для графиков."""
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()

        c.execute("""
            SELECT source, high, medium, low, total_analyzed
            FROM source_stats
            ORDER BY total_analyzed DESC, source ASC
        """)

        data = c.fetchall()
        sources = []
        credibility_indices = []
        high_counts = []
        medium_counts = []
        low_counts = []
        total_analyzed_counts = []

        for source, high, medium, low, total in data:
            total_current = high + medium + low
            if total_current == 0:
                score = 0.5
            else:
                score = (high * 1.0 + medium * 0.5 + low * 0.0) / total_current

            sources.append(source)
            credibility_indices.append(round(score, 2))
            high_counts.append(high)
            medium_counts.append(medium)
            low_counts.append(low)
            total_analyzed_counts.append(total_current)

        return {
            'sources': sources,
            'credibility_indices_for_plot': credibility_indices,
            'high_counts': high_counts,
            'medium_counts': medium_counts,
            'low_counts': low_counts,
            'total_analyzed_counts': total_analyzed_counts
        }
    except Exception as e:
        app.logger.error(f"Error getting source reliability data: {str(e)}")
        return {
            'sources': [],
            'credibility_indices_for_plot': [],
            'high_counts': [],
            'medium_counts': [],
            'low_counts': [],
            'total_analyzed_counts': []
        }
    finally:
        if conn:
            conn.close()

def get_analysis_history_html():
    """Получает историю анализа из базы данных."""
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()

        c.execute("""
            SELECT url, title, source, credibility_level, short_summary,
                   strftime('%Y-%m-%d %H:%M', analysis_date) as formatted_date
            FROM news
            ORDER BY analysis_date DESC
            LIMIT 15
        """)

        rows = c.fetchall()

        if not rows:
            return "<p>No analysis history yet. Analyze an article to see it appear here!</p>"

        html_items = []
        for url, title, source, credibility, short_summary, date_str in rows:
            display_title = title[:70] + '...' if title and len(title) > 70 else title if title else "N/A"
            source_display = source if source else "N/A"
            link_start = f"<a href='{url}' target='_blank' rel='noopener noreferrer'>" if url and url.startswith(('http://', 'https://')) else ""
            link_end = "</a>" if url and url.startswith(('http://', 'https://')) else ""
            summary_display = short_summary if short_summary else 'No summary available.'

            html_items.append(
                f"""
                <li>
                    <strong>{date_str}</strong>: {link_start}{display_title}{link_end} ({source_display}, {credibility})
                    <br>
                    <em>Summary:</em> {summary_display}
                </li>
                """
            )

        return f"<h3>📜 Recent Analyses:</h3><ul>{''.join(html_items)}</ul>"
    except Exception as e:
        app.logger.error(f"Error getting analysis history: {str(e)}")
        return "<p>Error retrieving analysis history.</p>"
    finally:
        if conn:
            conn.close()

# Маршруты Flask
@app.route('/')
def index():
    """Главная страница приложения."""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_route():
    """Обрабатывает запрос на анализ статьи."""
    try:
        data = request.json
        output_md, scores_for_chart, analysis_result = process_article_analysis(
            data.get('input_text'),
            data.get('source_name_manual')
        )

        if analysis_result is None:
            return jsonify({'error_message': output_md}), 400

        # Получаем похожие новости
        similar_news = fetch_similar_news(analysis_result)

        return jsonify({
            'output_md': output_md,
            'scores_for_chart': scores_for_chart,
            'analysis_result': analysis_result,
            'similar_news': render_similar_articles_html(similar_news)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/similar_articles', methods=['POST'])
def similar_articles_endpoint():
    """Получает похожие статьи."""
    try:
        data = request.json
        analysis_result = data.get('analysis_result')

        if not analysis_result:
            return jsonify({'similar_html': "<p>No analysis result provided to fetch similar articles.</p>"})

        similar_articles_list = fetch_similar_news(analysis_result)
        return jsonify({
            'similar_html': render_similar_articles_html(similar_articles_list)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/source_reliability_data')
def source_reliability_data_endpoint():
    """Получает данные о надежности источников для графиков."""
    try:
        data = get_source_reliability_data()

        labeled_sources = []
        if data['sources'] and data['credibility_indices_for_plot']:
            for source, score, total in zip(
                data['sources'],
                data['credibility_indices_for_plot'],
                data['total_analyzed_counts']
            ):
                labeled_sources.append(
                    f"{source}<br>Credibility: {score*100:.0f}%<br>Articles: {total}"
                )

        return jsonify({
            'sources': labeled_sources,
            'credibility_indices_for_plot': data['credibility_indices_for_plot'],
            'high_counts': data['high_counts'],
            'medium_counts': data['medium_counts'],
            'low_counts': data['low_counts'],
            'total_analyzed_counts': data['total_analyzed_counts']
        })
    except Exception as e:
        return jsonify({
            'sources': [],
            'credibility_indices_for_plot': [],
            'high_counts': [],
            'medium_counts': [],
            'low_counts': [],
            'total_analyzed_counts': []
        })

@app.route('/analysis_history_html')
def analysis_history_html_endpoint():
    """Получает историю анализа."""
    try:
        history_html = get_analysis_history_html()
        return jsonify({'history_html': history_html})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/check_db_integrity')
def check_db_integrity_endpoint():
    """Проверяет целостность базы данных."""
    try:
        result = check_database_integrity()
        if result:
            return jsonify({
                'status': 'success',
                'message': 'Database integrity check passed'
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Database integrity check failed'
            }), 500
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Database check failed: {str(e)}'
        }), 500

# Инициализация базы данных
def initialize_database():
    """Инициализирует базу данных."""
    ensure_db_schema()
    check_database_integrity()

if __name__ == '__main__':
    initialize_database()
    app.run(host='0.0.0.0', port=5000)
