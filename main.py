import os
import logging
import sqlite3
import re
import json
import requests
import html
import smtplib
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse
from logging.handlers import RotatingFileHandler
from flask import Flask, request, jsonify, render_template, abort, make_response, session
from werkzeug.middleware.proxy_fix import ProxyFix
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import anthropic
from newspaper import Article
from stop_words import get_stop_words

# Инициализация Flask приложения
app = Flask(__name__, static_folder='static', template_folder='templates')
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)
app.secret_key = os.getenv('SECRET_KEY', 'your-secret-key-here')

# Определение переменных окружения
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
MODEL_NAME = os.getenv('ANTHROPIC_MODEL', 'claude-3-opus-20240229')
NEWS_API_KEY = os.getenv('NEWS_API_KEY')

# Конфигурация для отправки email
app.config['MAIL_SERVER'] = os.getenv('MAIL_SERVER', 'smtp.gmail.com')
app.config['MAIL_PORT'] = int(os.getenv('MAIL_PORT', 587))
app.config['MAIL_USE_TLS'] = os.getenv('MAIL_USE_TLS', 'true').lower() == 'true'
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')
app.config['MAIL_DEFAULT_SENDER'] = os.getenv('MAIL_DEFAULT_SENDER')

# Используем встроенную временную зону UTC
UTC = timezone.utc

# Настройка логгирования
def setup_logging():
    """Настройка системы логгирования"""
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s - %(pathname)s:%(lineno)d')

    # Логирование в файл с ротацией
    file_handler = RotatingFileHandler('app.log', maxBytes=1024*1024, backupCount=5)
    file_handler.setFormatter(formatter)

    # Логирование в консоль
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    app.logger.addHandler(file_handler)
    app.logger.addHandler(console_handler)
    app.logger.setLevel(logging.INFO)

def send_email(subject, body, recipient):
    """Функция для отправки email"""
    try:
        msg = MIMEMultipart()
        msg['From'] = app.config['MAIL_DEFAULT_SENDER']
        msg['To'] = recipient
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        with smtplib.SMTP(app.config['MAIL_SERVER'], app.config['MAIL_PORT']) as server:
            server.starttls()
            server.login(app.config['MAIL_USERNAME'], app.config['MAIL_PASSWORD'])
            server.send_message(msg)

        app.logger.info(f"Email sent to {recipient}")
        return True
    except Exception as e:
        app.logger.error(f"Error sending email: {str(e)}")
        return False

# Проверка обязательных переменных окружения
def check_env_vars():
    """Проверка наличия обязательных переменных окружения"""
    REQUIRED_ENV_VARS = [
        'ANTHROPIC_API_KEY',
        'SECRET_KEY',
        'ANTHROPIC_MODEL',
        'NEWS_API_KEY'
    ]

    missing_vars = [var for var in REQUIRED_ENV_VARS if not os.getenv(var)]
    if missing_vars:
        error_msg = f"Missing required environment variables: {', '.join(missing_vars)}"
        app.logger.critical(error_msg)
        raise ValueError(error_msg)

# Конфигурация приложения
def configure_app():
    """Настройка конфигурации Flask приложения"""
    app.config.update(
        SESSION_COOKIE_SECURE=True,
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE='Lax',
        PERMANENT_SESSION_LIFETIME=timedelta(days=1)
    )

# Инициализация базы данных
DB_NAME = 'news_analysis.db'

def get_db_connection():
    """Создание подключения к базе данных"""
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def ensure_db_schema():
    """Обеспечение наличия необходимых таблиц в базе данных"""
    with get_db_connection() as conn:
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

        c.execute('''CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL,
            type TEXT NOT NULL,
            message TEXT NOT NULL,
            date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )''')
        conn.commit()
        app.logger.info('Database schema ensured successfully.')

def initialize_sources(initial_counts):
    """Инициализация начальных данных для таблицы source_stats"""
    with get_db_connection() as conn:
        c = conn.cursor()
        for source, counts in initial_counts.items():
            c.execute('SELECT total_analyzed FROM source_stats WHERE source = ?', (source,))
            if not c.fetchone():
                high = counts.get('high', 0)
                medium = counts.get('medium', 0)
                low = counts.get('low', 0)
                c.execute('''INSERT INTO source_stats
                             (source, high, medium, low, total_analyzed)
                             VALUES (?, ?, ?, ?, ?)''',
                             (source, high, medium, low, high + medium + low))
        conn.commit()
        app.logger.info('Initial source initialization completed successfully.')

def check_database_integrity():
    """Проверка целостности базы данных"""
    try:
        if not os.path.exists(DB_NAME):
            app.logger.warning(f'Database file {DB_NAME} not found! Creating a new one.')
            return True

        with get_db_connection() as conn:
            c = conn.cursor()
            c.execute('SELECT name FROM sqlite_master WHERE type="table"')
            tables = [table[0] for table in c.fetchall()]
            required_tables = ['news', 'source_stats', 'feedback']

            if not all(table in tables for table in required_tables):
                app.logger.warning('Critical tables missing in the database! Creating them.')
                ensure_db_schema()
                return True

        app.logger.info("Database integrity check passed.")
        return True
    except Exception as e:
        app.logger.error(f'Error during database integrity check: {e}')
        return False

# Данные для инициализации
INITIAL_SOURCE_COUNTS = {
    'bbc.com': {'high': 15, 'medium': 5, 'low': 1},
    'reuters.com': {'high': 20, 'medium': 3, 'low': 0},
    'foxnews.com': {'high': 3, 'medium': 7, 'low': 15},
    'cnn.com': {'high': 5, 'medium': 10, 'low': 5},
    'nytimes.com': {'high': 10, 'medium': 5, 'low': 2},
    'theguardian.com': {'high': 12, 'medium': 4, 'low': 1},
    'apnews.com': {'high': 18, 'medium': 2, 'low': 0}
}

media_owners = {
    'bbc.com': 'BBC',
    'reuters.com': 'Thomson Reuters',
    'foxnews.com': 'Fox Corporation',
    'cnn.com': 'Warner Bros. Discovery',
    'nytimes.com': 'The New York Times Company',
    'theguardian.com': 'Guardian Media Group',
    'apnews.com': 'Associated Press',
    'aljazeera.com': 'Al Jazeera Media Network',
    'wsj.com': 'News Corp'
}

predefined_trust_scores = {
    'bbc.com': 0.9, 'bbc.co.uk': 0.9, 'reuters.com': 0.95, 'apnews.com': 0.93,
    'nytimes.com': 0.88, 'theguardian.com': 0.85, 'wsj.com': 0.82,
    'cnn.com': 0.70, 'foxnews.com': 0.40, 'aljazeera.com': 0.80
}

TRUSTED_NEWS_SOURCES_IDS = [
    'bbc-news', 'reuters', 'associated-press', 'the-new-york-times',
    'the-guardian-uk', 'the-wall-street-journal', 'cnn', 'al-jazeera-english'
]

stop_words_en = get_stop_words('en')

# Защита от WordPress сканеров
WORDPRESS_PATHS = [
    re.compile(r'wp-admin', re.IGNORECASE),
    re.compile(r'wp-includes', re.IGNORECASE),
    re.compile(r'wp-content', re.IGNORECASE),
    re.compile(r'xmlrpc\.php', re.IGNORECASE),
    re.compile(r'wp-login\.php', re.IGNORECASE),
    re.compile(r'wp-config\.php', re.IGNORECASE),
    re.compile(r'readme\.html', re.IGNORECASE),
    re.compile(r'license\.txt', re.IGNORECASE),
    re.compile(r'wp-json', re.IGNORECASE),
]

@app.before_request
def block_wordpress_scanners():
    """Блокировка запросов к WordPress путям"""
    path = request.path.lower()
    if any(pattern.search(path) for pattern in WORDPRESS_PATHS):
        app.logger.warning(f'Blocked WordPress scanner request from {request.remote_addr}')
        return abort(404)

    if any(param in request.query_string.decode('utf-8', 'ignore') for param in ['=http://', '=https://', '=ftp://']):
        app.logger.warning(f'Blocked suspicious query parameter from {request.remote_addr}')
        return abort(404)

@app.after_request
def add_security_headers(response):
    """Добавление заголовков безопасности и CORS"""
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, X-Requested-With'
    response.headers['Access-Control-Max-Age'] = '86400'

    csp = (
        "default-src 'self'; "
        "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://cdn.plot.ly; "
        "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://cdn.plot.ly; "
        "font-src 'self' https://cdn.jsdelivr.net; "
        "img-src 'self' data:;"
    )
    response.headers['Content-Security-Policy'] = csp
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    response.headers['Permissions-Policy'] = 'geolocation=(), microphone=(), camera=()'
    return response

# Обработчик OPTIONS запросов
@app.route('/analyze', methods=['OPTIONS'])
def handle_options():
    """Обработчик OPTIONS запросов для CORS"""
    response = make_response()
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, X-Requested-With'
    response.headers['Access-Control-Max-Age'] = '86400'
    return response

# Обработчики ошибок
@app.errorhandler(404)
def page_not_found(e):
    """Обработчик ошибки 404"""
    response = make_response(render_template('404.html'), 404)
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response

@app.errorhandler(500)
def internal_server_error(e):
    """Обработчик ошибки 500"""
    response = make_response(render_template('500.html'), 500)
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response

# Класс для работы с API Anthropic
class ClaudeNewsAnalyzer:
    """Класс для взаимодействия с API Anthropic Claude"""
    def __init__(self, api_key, model_name):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model_name = model_name

    def analyze_article_text(self, article_text_content, source_name_for_context):
        """Анализ текста статьи с помощью Claude API"""
        try:
            max_chars_for_claude = 10000
            if len(article_text_content) > max_chars_for_claude:
                article_text_content = article_text_content[:max_chars_for_claude]
                app.logger.warning(f"Article content truncated to {max_chars_for_claude} characters for Claude API.")

            media_owner_display = media_owners.get(source_name_for_context, 'Unknown Owner')

            prompt = (
                'You are a highly analytical and neutral AI assistant specializing in news article reliability and content analysis. '
                'Your task is to dissect the provided news article.\n\n'
                f'Article Text:\n"""\n{article_text_content}\n"""\n\n'
                f'Source (for context, if known): {source_name_for_context}\n'
                f'Media Owner: {media_owner_display}\n\n'
                'Please perform the following analyses and return the results ONLY in a single, valid JSON object format. '
                'Do not include any explanatory text before or after the JSON object.\n\n'
                'JSON Fields:\n'
                '- "news_integrity": (Float, 0.0-1.0) Assess the overall integrity and trustworthiness of the information presented.\n'
                '- "fact_check_needed_score": (Float, 0.0-1.0) Likelihood that the article\'s claims require external fact-checking.\n'
                '- "sentiment_score": (Float, 0.0-1.0) Overall emotional tone (0.0 negative, 0.5 neutral, 1.0 positive).\n'
                '- "bias_score": (Float, 0.0-1.0) Degree of perceived bias (0.0 low bias, 1.0 high bias).\n'
                '- "topics": (List of strings) Identify 3-5 main topics or keywords.\n'
                '- "key_arguments": (List of strings) Extract the main arguments or claims.\n'
                '- "mentioned_facts": (List of strings) List any specific facts or statistics mentioned.\n'
                '- "author_purpose": (String) Briefly determine the author\'s likely primary purpose.\n'
                '- "potential_biases_identified": (List of strings) Enumerate any specific signs of potential bias.\n'
                '- "short_summary": (String) A concise summary of the article\'s main content.\n'
                '- "index_of_credibility": (Float, 0.0-1.0) Overall credibility index.\n'
                '- "published_date": (String, YYYY-MM-DD or N/A) The publication date.'
            )

            message = self.client.messages.create(
                model=self.model_name,
                max_tokens=2000,
                temperature=0.2,
                system='You are a JSON-generating expert. Always provide valid JSON.',
                messages=[{'role': 'user', 'content': prompt}]
            )

            raw_json_text = message.content[0].text.strip()
            match = re.search(r'```json\s*(\{.*\})\s*```', raw_json_text, re.DOTALL)

            if match:
                json_str = match.group(1)
            else:
                json_str = raw_json_text

            return json.loads(json_str)

        except Exception as e:
            app.logger.error(f'Error in Claude analysis: {str(e)}')
            raise ValueError(f'Error communicating with AI: {str(e)}')

def extract_text_from_url(url):
    """Извлечение текста из URL"""
    try:
        clean_url = re.sub(r'/amp(/)?\$', '', url)

        if any(domain in url for domain in ['youtube.com', 'vimeo.com']):
            return "Video content detected", urlparse(clean_url).netloc.replace('www.', ''), "Video: " + url

        article = Article(clean_url)
        article.download()
        article.parse()

        if len(article.text) < 100:
            app.logger.warning(f"Extracted article content is too short from {url}")
            return "", urlparse(clean_url).netloc.replace('www.', ''), ""

        return article.text.strip(), urlparse(clean_url).netloc.replace('www.', ''), article.title.strip()
    except Exception as e:
        app.logger.error(f'Error extracting article from URL {url}: {e}')
        return "", "", ""

def calculate_credibility_level(integrity, fact_check_needed, sentiment, bias):
    """Расчет уровня достоверности"""
    fact_check_score = 1.0 - fact_check_needed
    neutral_sentiment_proximity = 1.0 - abs(sentiment - 0.5) * 2
    bias_score_inverted = 1.0 - bias

    avg = (integrity * 0.45) + (fact_check_score * 0.35) + (neutral_sentiment_proximity * 0.10) + (bias_score_inverted * 0.10)

    if avg >= 0.75:
        return 'High'
    if avg >= 0.5:
        return 'Medium'
    return 'Low'

def save_analysis_to_db(url, title, source, content, analysis_result):
    """Сохранение анализа в базу данных"""
    try:
        with get_db_connection() as conn:
            c = conn.cursor()

            integrity = analysis_result.get('news_integrity', 0.0)
            fact_check_needed = analysis_result.get('fact_check_needed_score', 1.0)
            sentiment = analysis_result.get('sentiment_score', 0.5)
            bias = analysis_result.get('bias_score', 1.0)
            short_summary = analysis_result.get('short_summary', 'Summary not available.')
            index_of_credibility = analysis_result.get('index_of_credibility', 0.0)

            credibility_level = calculate_credibility_level(integrity, fact_check_needed, sentiment, bias)
            db_url = url if url else f'text_input_{datetime.now(UTC).timestamp()}'

            c.execute('''INSERT INTO news
                         (url, title, source, content, integrity, fact_check, sentiment, bias,
                          credibility_level, short_summary, index_of_credibility)
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                         ON CONFLICT(url) DO UPDATE SET
                         title=excluded.title, source=excluded.source, content=excluded.content,
                         integrity=excluded.integrity, fact_check=excluded.fact_check,
                         sentiment=excluded.sentiment, bias=excluded.bias,
                         credibility_level=excluded.credibility_level,
                         short_summary=excluded.short_summary,
                         index_of_credibility=excluded.index_of_credibility,
                         analysis_date=CURRENT_TIMESTAMP''',
                         (db_url, title, source, content, integrity, fact_check_needed,
                          sentiment, bias, credibility_level, short_summary, index_of_credibility))

            c.execute('SELECT high, medium, low, total_analyzed FROM source_stats WHERE source = ?', (source,))
            row = c.fetchone()

            if row:
                high, medium, low, total = row
                if credibility_level == 'High': high += 1
                elif credibility_level == 'Medium': medium += 1
                else: low += 1
                total += 1
                c.execute('''UPDATE source_stats SET high=?, medium=?, low=?, total_analyzed=?
                             WHERE source=?''', (high, medium, low, total, source))
            else:
                high = 1 if credibility_level == 'High' else 0
                medium = 1 if credibility_level == 'Medium' else 0
                low = 1 if credibility_level == 'Low' else 0
                c.execute('''INSERT INTO source_stats
                             (source, high, medium, low, total_analyzed)
                             VALUES (?, ?, ?, ?, ?)''',
                             (source, high, medium, low, high + medium + low))

            conn.commit()
            return credibility_level
    except Exception as e:
        app.logger.error(f'Error saving analysis to database: {str(e)}')
        raise

def generate_query(analysis_result):
    """Генерация запроса для поиска похожих статей"""
    topics = analysis_result.get('topics', [])
    key_arguments = analysis_result.get('key_arguments', [])
    mentioned_facts = analysis_result.get('mentioned_facts', [])

    all_terms = []
    for phrase_list in [topics, key_arguments]:
        for phrase in phrase_list:
            if not phrase.strip(): continue
            if ' ' in phrase.strip() and len(phrase.strip().split()) > 1:
                all_terms.append('"' + phrase.strip() + '"')
            else:
                all_terms.append(phrase.strip())

    for fact in mentioned_facts:
        if not fact.strip(): continue
        words = [word for word in fact.lower().split() if word not in stop_words_en and len(word) > 2]
        all_terms.extend(words)

    unique_terms = list(set(all_terms))

    if len(unique_terms) >= 3:
        query = ' AND '.join(unique_terms)
    elif unique_terms:
        query = ' OR '.join(unique_terms)
    else:
        query = 'current events OR news'

    return query

def make_newsapi_request(params):
    """Выполнение запроса к NewsAPI"""
    url = 'https://newsapi.org/v2/everything'
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        return response.json().get('articles', [])
    except Exception as e:
        app.logger.error(f'NewsAPI Request Error: {str(e)}')
        return []

def fetch_same_topic_articles(analysis_result, page=1, per_page=3):
    """Поиск похожих статей по теме с пагинацией"""
    if not NEWS_API_KEY:
        app.logger.warning('NEWS_API_KEY is not configured. Skipping similar news search.')
        return []

    query = generate_query(analysis_result)
    end_date = datetime.now(UTC).date()
    start_date = end_date - timedelta(days=7)

    params = {
        'q': query,
        'apiKey': NEWS_API_KEY,
        'language': 'en',
        'pageSize': per_page,
        'page': page,
        'sortBy': 'relevancy',
        'from': start_date.strftime('%Y-%m-%d'),
        'to': end_date.strftime('%Y-%m-%d'),
    }

    if TRUSTED_NEWS_SOURCES_IDS:
        params['sources'] = ','.join(TRUSTED_NEWS_SOURCES_IDS)

    articles = make_newsapi_request(params)

    if not articles and query != 'current events OR news':
        broader_query = ' OR '.join([f'"{term}"' if ' ' in term else term
                                  for term in analysis_result.get('topics', [])[:3]
                                  if term and term not in stop_words_en])
        if not broader_query:
            broader_query = 'current events OR news'

        params['q'] = broader_query
        additional_articles = make_newsapi_request(params)
        articles.extend(additional_articles)

    unique_articles = {}
    for article in articles:
        if article.get('url'):
            unique_articles[article['url']] = article

    articles = list(unique_articles.values())

    if not articles:
        return []

    all_query_terms = []
    all_query_terms.extend([t.lower().replace('"', '') for t in query.split(' AND ') if t.strip()])
    if 'broader_query' in locals():
        all_query_terms.extend([t.lower().replace('"', '') for t in broader_query.split(' OR ') if t.strip()])
    all_query_terms = list(set([t for t in all_query_terms if t and t not in stop_words_en]))

    ranked_articles = []
    for article in articles:
        source_domain = urlparse(article.get('url', '')).netloc.replace('www.', '')
        trust_score = predefined_trust_scores.get(source_domain, 0.5)

        article_text = (article.get('title', '') + ' ' + article.get('description', '')).lower()
        relevance_score = sum(1 for term in all_query_terms if term in article_text)
        final_score = (relevance_score * 10) + (trust_score * 5)
        ranked_articles.append((article, final_score))

    ranked_articles.sort(key=lambda item: item[1], reverse=True)
    return [item[0] for item in ranked_articles[:per_page]]

def render_same_topic_articles_html(articles):
    """Формирование HTML для похожих статей"""
    if not articles:
        return '<p>No same topic articles found for the selected criteria.</p>'

    html_items = []
    for art in articles:
        title = html.escape(art.get('title', 'No Title'))
        article_url = html.escape(art.get('url', '#'))
        source_api_name = html.escape(art.get('source', {}).get('name', 'Unknown Source'))
        published_at = html.escape(art.get('publishedAt', 'N/A').split('T')[0])
        description = html.escape(art.get('description', 'No description available.'))

        domain = urlparse(art.get('url', '#')).netloc.replace('www.', '')
        trust_score = predefined_trust_scores.get(domain, 0.5)
        trust_display = f' (Credibility: {int(trust_score*100)}%)'

        html_items.append(
            f'<div class="same-topic-article">'
            f'<h4><a href="{article_url}" target="_blank">{title}</a></h4>'
            f'<p><strong>Source:</strong> {source_api_name}{trust_display} | '
            f'<strong>Published:</strong> {published_at}</p>'
            f'<p>{description}</p>'
            f'</div>'
        )

    return '<div class="same-topic-articles-container">' + ''.join(html_items) + '</div>'

def get_source_reliability_data():
    """Получение данных о надежности источников"""
    try:
        with get_db_connection() as conn:
            c = conn.cursor()
            c.execute('''
                SELECT source, high, medium, low, total_analyzed
                FROM source_stats
                ORDER BY total_analyzed DESC, source ASC
            ''')

            data = c.fetchall()
            sources = []
            credibility_indices = []
            high_counts = []
            medium_counts = []
            low_counts = []
            total_analyzed_counts = []

            for source, high, medium, low, total in data:
                total_current = high + medium + low
                score = (high * 1.0 + medium * 0.5 + low * 0.0) / total_current if total_current > 0 else 0.5

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
        app.logger.error(f'Error getting source reliability data: {str(e)}')
        return {
            'sources': [], 'credibility_indices_for_plot': [],
            'high_counts': [], 'medium_counts': [], 'low_counts': [],
            'total_analyzed_counts': []
        }

def get_analysis_history_html():
    """Получение истории анализов"""
    try:
        with get_db_connection() as conn:
            c = conn.cursor()
            c.execute('''
                SELECT url, title, source, credibility_level, short_summary,
                       strftime("%Y-%m-%d %H:%M", analysis_date) as formatted_date
                FROM news
                ORDER BY analysis_date DESC
                LIMIT 15
            ''')

            rows = c.fetchall()
            if not rows:
                return '<p>No analysis history yet. Analyze an article to see it appear here!</p>'

            html_items = []
            for url, title, source, credibility, short_summary, date_str in rows:
                display_title = title[:70] + '...' if title and len(title) > 70 else title
                source_display = source if source else 'N/A'
                link_start = f'<a href="{html.escape(url)}" target="_blank">' if url else ''
                link_end = '</a>' if url else ''
                summary_display = short_summary if short_summary else 'No summary available.'

                html_items.append(
                    f'<li><strong>{html.escape(date_str)}</strong>: '
                    f'{link_start}{html.escape(display_title)}{link_end} '
                    f'({html.escape(source_display)}, {html.escape(credibility)})<br>'
                    f'<em>Summary:</em> {html.escape(summary_display)}</li>'
                )

            return '<h3>Recent Analyses:</h3><ul>' + ''.join(html_items) + '</ul>'
    except Exception as e:
        app.logger.error(f'Error getting analysis history: {str(e)}')
        return '<p>Error retrieving analysis history due to a database issue.</p>'

def prepare_chart_data(analysis_result):
    """Подготовка данных для графика"""
    integrity = analysis_result.get('news_integrity', 0.0)
    fact_check = analysis_result.get('fact_check_needed_score', 1.0)
    sentiment = analysis_result.get('sentiment_score', 0.5)
    bias = analysis_result.get('bias_score', 1.0)

    return {
        'integrity': integrity,
        'fact_check': fact_check,
        'sentiment': sentiment,
        'bias': bias
    }

def format_analysis_results(article_title, source_name, analysis_result, credibility_saved):
    """Форматирование результатов анализа для отображения"""
    integrity = analysis_result.get('news_integrity', 0.0)
    fact_check = analysis_result.get('fact_check_needed_score', 1.0)
    sentiment = analysis_result.get('sentiment_score', 0.5)
    bias = analysis_result.get('bias_score', 1.0)

    output_md = f"""
# Analysis Results

## Article Information
- **Title:** {article_title}
- **Source:** {source_name}
- **Credibility Level:** {credibility_saved}

## Analysis Scores
- **News Integrity:** {integrity:.2f}
- **Fact Check Needed:** {fact_check:.2f}
- **Sentiment Score:** {sentiment:.2f}
- **Bias Score:** {bias:.2f}

## Additional Information
- **Topics:** {', '.join(analysis_result.get('topics', []))}
- **Key Arguments:** {', '.join(analysis_result.get('key_arguments', []))}
- **Mentioned Facts:** {', '.join(analysis_result.get('mentioned_facts', []))}
- **Author Purpose:** {analysis_result.get('author_purpose', 'N/A')}
- **Potential Biases Identified:** {', '.join(analysis_result.get('potential_biases_identified', []))}
- **Short Summary:** {analysis_result.get('short_summary', 'N/A')}
- **Index of Credibility:** {analysis_result.get('index_of_credibility', 0.0):.2f}
- **Published Date:** {analysis_result.get('published_date', 'N/A')}
    """

    return output_md

def process_article_analysis(input_text, source_name_manual):
    """Организация полного процесса анализа статьи"""
    try:
        article_url = None
        article_content = input_text
        article_title = 'User-provided Text'
        source_name = source_name_manual if source_name_manual else 'Direct Input'

        if input_text.strip().startswith('http'):
            article_url = input_text.strip()
            content_from_url, source_from_url, title_from_url = extract_text_from_url(article_url)

            if content_from_url and len(content_from_url) >= 100:
                article_content, source_name, article_title = content_from_url, source_from_url, title_from_url
            else:
                if not content_from_url:
                    return ('❌ Failed to extract content from the provided URL. Please check the link or provide text directly.', None, None)
                else:
                    return ('❌ Extracted article content is too short for analysis (min 100 chars).', None, None)

        if len(article_content) < 100:
            return ('❌ Article content is too short for analysis (min 100 chars).', None, None)

        analyzer = ClaudeNewsAnalyzer(ANTHROPIC_API_KEY, MODEL_NAME)
        analysis_result = analyzer.analyze_article_text(article_content, source_name)

        credibility_saved = save_analysis_to_db(article_url, article_title, source_name, article_content, analysis_result)

        output_md = format_analysis_results(article_title, source_name, analysis_result, credibility_saved)
        scores_for_chart = prepare_chart_data(analysis_result)

        return output_md, scores_for_chart, analysis_result
    except Exception as e:
        app.logger.error(f'Error in process_article_analysis: {str(e)}')
        return (f'❌ Error during analysis: {str(e)}', None, None)

# Эндпоинты API
@app.route('/')
def index():
    """Отображение главной страницы приложения"""
    return render_template('index.html')

@app.route('/analyze', methods=['POST', 'OPTIONS'])
def analyze():
    """API endpoint для анализа статьи"""
    try:
        if request.method == 'OPTIONS':
            response = make_response()
            response.headers['Access-Control-Allow-Origin'] = '*'
            response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
            response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, X-Requested-With'
            response.headers['Access-Control-Max-Age'] = '86400'
            return response

        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 415

        data = request.get_json()
        input_text = data.get('input_text')
        source_name_manual = data.get('source_name_manual')

        if not input_text:
            return jsonify({'error': 'No input text or URL provided'}), 400

        output_md, scores_for_chart, analysis_result = process_article_analysis(input_text, source_name_manual)

        if analysis_result is None:
            return jsonify({'error': output_md}), 400

        session['last_analysis_result'] = analysis_result
        same_topic_news = fetch_same_topic_articles(analysis_result)
        same_topic_html = render_same_topic_articles_html(same_topic_news)

        return jsonify({
            'output_md': output_md,
            'scores_for_chart': scores_for_chart,
            'same_topic_news': same_topic_html
        })
    except Exception as e:
        app.logger.error(f"Error in analyze endpoint: {str(e)}")
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/same_topic_articles', methods=['GET'])
def get_same_topic_articles():
    """Эндпоинт для получения дополнительных статей по той же теме"""
    try:
        page = int(request.args.get('page', 1))
        per_page = 3

        analysis_result = session.get('last_analysis_result')
        if not analysis_result:
            return jsonify({'error': 'No analysis result found. Please analyze an article first.'}), 400

        same_topic_articles = fetch_same_topic_articles(analysis_result, page=page, per_page=per_page)
        same_topic_html = render_same_topic_articles_html(same_topic_articles)

        return jsonify({
            'same_topic_html': same_topic_html
        })
    except Exception as e:
        app.logger.error(f"Error in get_same_topic_articles endpoint: {str(e)}")
        return jsonify({'error': 'An error occurred while fetching same topic articles'}), 500

@app.route('/source_reliability_data', methods=['GET'])
def source_reliability_data_endpoint():
    """API endpoint для предоставления данных о надежности источников"""
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
                    f"{html.escape(source)}<br>Credibility: {int(score*100)}%<br>Articles: {total}"
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
        app.logger.error(f"Error in source_reliability_data_endpoint: {str(e)}")
        return jsonify({'error': 'An error occurred while fetching source reliability data'}), 500

@app.route('/analysis_history', methods=['GET'])
def analysis_history_endpoint():
    """API endpoint для получения истории анализов"""
    try:
        history_html = get_analysis_history_html()
        return jsonify({'history_html': history_html})
    except Exception as e:
        app.logger.error(f"Error in analysis_history_endpoint: {str(e)}")
        return jsonify({'error': 'An error occurred while fetching analysis history'}), 500

@app.route('/feedback', methods=['GET', 'POST'])
def feedback():
    """Обработчик формы обратной связи"""
    try:
        if request.method == 'POST':
            data = request.get_json()
            name = data.get('name')
            email = data.get('email')
            feedback_type = data.get('type')
            message = data.get('message')

            if not all([name, email, feedback_type, message]):
                return jsonify({'error': 'All fields are required'}), 400

            if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
                return jsonify({'error': 'Invalid email address'}), 400

            try:
                with get_db_connection() as conn:
                    c = conn.cursor()
                    c.execute('''
                        INSERT INTO feedback (name, email, type, message, date)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (name, email, feedback_type, message, datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')))
                    conn.commit()
            except Exception as e:
                app.logger.error(f'Error saving feedback to database: {str(e)}')
                return jsonify({'error': 'Error saving feedback to database'}), 500

            email_subject = f"New Feedback: {feedback_type}"
            email_body = f"Name: {name}\nEmail: {email}\nType: {feedback_type}\nMessage: {message}"
            recipient = app.config['MAIL_DEFAULT_SENDER']

            if send_email(email_subject, email_body, recipient):
                return jsonify({'message': 'Thank you for your feedback! We appreciate it.'})
            else:
                return jsonify({'error': 'Error sending feedback email'}), 500

        return render_template('feedback.html')
    except Exception as e:
        app.logger.error(f"Error in feedback endpoint: {str(e)}")
        return jsonify({'error': 'An error occurred while processing feedback'}), 500

if __name__ == '__main__':
    setup_logging()
    check_env_vars()
    configure_app()
    if not check_database_integrity():
        ensure_db_schema()
        initialize_sources(INITIAL_SOURCE_COUNTS)
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
