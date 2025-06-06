from flask import Flask, render_template, request, jsonify, abort
import os
import json
import sqlite3
import requests
import anthropic
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from datetime import datetime, timedelta, UTC
import logging
from logging.handlers import RotatingFileHandler
from werkzeug.middleware.proxy_fix import ProxyFix
import re
import plotly.graph_objects as go
from stop_words import get_stop_words
from newspaper import Article
import html

# Инициализация Flask
app = Flask(__name__, static_folder='static', template_folder='templates')
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)
@app.after_request
def add_security_headers(response):
    # Определяем новую, более разрешающую политику CSP
    csp = (
        "default-src 'self'; "
        "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
        "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://cdn.plot.ly; "
        "font-src 'self' https://cdn.jsdelivr.net; "
        "img-src 'self' data:;"  # Разрешает изображения с вашего домена и встроенные (data:)
    )
    # Устанавливаем заголовок, который переопределит настройки Railway
    response.headers['Content-Security-Policy'] = csp
    return response

# Настройка логирования
def setup_logging():
    file_handler = RotatingFileHandler('app.log', maxBytes=1024*1024, backupCount=5)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    app.logger.addHandler(file_handler)
    app.logger.addHandler(console_handler)
    app.logger.setLevel(logging.INFO)

# Проверка обязательных переменных окружения
def check_required_env_vars():
    required_vars = [
        'ANTHROPIC_API_KEY',
        'SECRET_KEY',
        'NEWS_API_KEY'
    ]

    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        error_msg = f"Missing required environment variables: {', '.join(missing_vars)}"
        app.logger.error(error_msg)
        raise ValueError(error_msg)

# Получение переменных окружения
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
NEWS_API_KEY = os.getenv('NEWS_API_KEY')
NEWS_API_ENABLED = bool(NEWS_API_KEY)
MODEL_NAME = 'claude-3-opus-20240229'
SECRET_KEY = os.getenv('SECRET_KEY')

# Установка секретного ключа
app.secret_key = SECRET_KEY
if not SECRET_KEY:
    raise ValueError('SECRET_KEY is not set in Railway variables!')

# Настройка безопасности
app.config.update(
    SESSION_COOKIE_SECURE=True,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
    PERMANENT_SESSION_LIFETIME=timedelta(days=1)
)

# Проверка API ключей
if not ANTHROPIC_API_KEY:
    raise ValueError('ANTHROPIC_API_KEY is missing! Please set it in Railway variables.')
if not NEWS_API_KEY:
    app.logger.warning('NEWS_API_KEY is missing! Similar news functionality will be disabled.')

# Конфигурация базы данных
DB_NAME = 'news_analysis.db'

# Начальные данные о надежности источников
INITIAL_SOURCE_COUNTS = {
    'bbc.com': {'high': 15, 'medium': 5, 'low': 1},
    'reuters.com': {'high': 20, 'medium': 3, 'low': 0},
    'foxnews.com': {'high': 3, 'medium': 7, 'low': 15},
    'cnn.com': {'high': 5, 'medium': 10, 'low': 5},
    'nytimes.com': {'high': 10, 'medium': 5, 'low': 2},
    'theguardian.com': {'high': 12, 'medium': 4, 'low': 1},
    'apnews.com': {'high': 18, 'medium': 2, 'low': 0}
}

# Соответствие доменов и владельцев СМИ
media_owners = {
    'bbc.com': 'BBC',
    'reuters.com': 'Thomson Reuters',
    'foxnews.com': 'Fox Corporation',
    'cnn.com': 'Warner Bros. Discovery',
    'nytimes.com': 'The New York Times Company',
    'theguardian.com': 'Guardian Media Group',
    'apnews.com': 'Associated Press'
}

# Надежные источники NewsAPI
TRUSTED_NEWS_SOURCES_IDS = [
    'bbc-news', 'reuters', 'associated-press', 'the-new-york-times',
    'the-guardian-uk', 'the-wall-street-journal', 'cnn'
]

stop_words_en = get_stop_words('en')

# Улучшенная защита от WordPress-сканеров
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
    re.compile(r'wp-comments-post\.php', re.IGNORECASE),
    re.compile(r'wp-trackback\.php', re.IGNORECASE),
    re.compile(r'wp-signup\.php', re.IGNORECASE),
    re.compile(r'wp-activate\.php', re.IGNORECASE),
    re.compile(r'wp-blog-header\.php', re.IGNORECASE),
    re.compile(r'wp-cron\.php', re.IGNORECASE),
    re.compile(r'wp-links-opml\.php', re.IGNORECASE),
    re.compile(r'wp-mail\.php', re.IGNORECASE),
    re.compile(r'wp-settings\.php', re.IGNORECASE),
    re.compile(r'wp-config-sample\.php', re.IGNORECASE),
    re.compile(r'wp-load\.php', re.IGNORECASE),
    re.compile(r'wp-.*\.php', re.IGNORECASE),
    re.compile(r'.*wp-admin.*', re.IGNORECASE),
    re.compile(r'.*wp-content.*', re.IGNORECASE),
    re.compile(r'.*wp-includes.*', re.IGNORECASE),
    re.compile(r'.*wp-json.*', re.IGNORECASE)
]

# Middleware для блокировки WordPress-сканеров
@app.before_request
def block_wordpress_scanners():
    path = request.path.lower()
    for pattern in WORDPRESS_PATHS:
        if pattern.search(path):
            app.logger.warning('Blocked WordPress scanner request from ' + request.remote_addr + ' to: ' + request.path)
            return abort(404)

    if any(param in request.query_string.decode('utf-8', 'ignore') for param in ['=http://', '=https://', '=ftp://']):
        app.logger.warning('Blocked suspicious query parameter in: ' + request.path)
        return abort(404)

# Middleware для добавления заголовков безопасности
@app.after_request
def add_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    response.headers['Content-Security-Policy'] = "default-src 'self'"
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    response.headers['Permissions-Policy'] = 'geolocation=(), microphone=(), camera=()'
    return response

# Обработчики ошибок
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    app.logger.error('Server error: ' + str(e))
    return render_template('500.html'), 500

# Функции для работы с базой данных
def ensure_db_schema():
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
        c.execute('''CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL,
            type TEXT NOT NULL,
            message TEXT NOT NULL,
            date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )''')
        conn.commit()
        app.logger.info('Database schema ensured successfully')
    except sqlite3.Error as e:
        app.logger.error('Error ensuring database schema: ' + str(e))
        raise
    finally:
        if conn:
            conn.close()

def initialize_sources(initial_counts):
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        for source, counts in initial_counts.items():
            c.execute('SELECT total_analyzed FROM source_stats WHERE source = ?', (source,))
            row = c.fetchone()
            if row is None:
                high = counts.get('high', 0)
                medium = counts.get('medium', 0)
                low = counts.get('low', 0)
                c.execute('''INSERT INTO source_stats (source, high, medium, low, total_analyzed)
                         VALUES (?, ?, ?, ?, ?)''',
                         (source, high, medium, low, high + medium + low))
        conn.commit()
        app.logger.info('Initial source initialization completed successfully')
    except sqlite3.Error as e:
        app.logger.error('Database error during source initialization: ' + str(e))
        raise
    finally:
        if conn:
            conn.close()

def check_database_integrity():
    try:
        if not os.path.exists(DB_NAME):
            app.logger.error('Database file ' + DB_NAME + ' not found!')
            return False

        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()

        # Проверяем существование таблиц
        c.execute('SELECT name FROM sqlite_master WHERE type="table"')
        tables = [table[0] for table in c.fetchall()]
        required_tables = ['news', 'source_stats', 'feedback']
        for table in required_tables:
            if table not in tables:
                app.logger.error('Critical table ' + table + ' is missing!')
                return False

        return True
    except Exception as e:
        app.logger.error('Error during database check: ' + str(e))
        return False
    finally:
        if conn:
            conn.close()

# Класс для анализа статей
class ClaudeNewsAnalyzer:
    def __init__(self, api_key, model_name):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model_name = model_name

    def analyze_article_text(self, article_text_content, source_name_for_context):
        max_chars_for_claude = 10000
        if len(article_text_content) > max_chars_for_claude:
            article_text_content = article_text_content[:max_chars_for_claude]

        media_owner_display = media_owners.get(source_name_for_context, 'Unknown Owner')

        prompt = 'You are a highly analytical and neutral AI assistant specializing in news article reliability and content analysis. Your task is to dissect the provided news article.\n\n' + \
                'Article Text:\n\"\"\"\n' + article_text_content + '\n\"\"\"\n\n' + \
                'Source (for context, if known): ' + source_name_for_context + '\n' + \
                'Media Owner: ' + media_owner_display + '\n\n' + \
                'Please perform the following analyses and return the results ONLY in a single, valid JSON object format. Do not include any explanatory text before or after the JSON object.\n\n' + \
                'JSON Fields:\n' + \
                '- "news_integrity": (Float, 0.0-1.0) Assess the overall integrity and trustworthiness of the information presented. Higher means more trustworthy.\n' + \
                '- "fact_check_needed_score": (Float, 0.0-1.0) Likelihood that the article\'s claims require external fact-checking. 1.0 means high likelihood.\n' + \
                '- "sentiment_score": (Float, 0.0-1.0) Overall emotional tone (0.0 negative, 0.5 neutral, 1.0 positive).\n' + \
                '- "bias_score": (Float, 0.0-1.0) Degree of perceived bias (0.0 low bias, 1.0 high bias).\n' + \
                '- "topics": (List of strings) Identify 3-5 main topics or keywords that accurately represent the core subject matter.\n' + \
                '- "key_arguments": (List of strings) Extract the main arguments or claims made by the author.\n' + \
                '- "mentioned_facts": (List of strings) List any specific facts, data, or statistics mentioned.\n' + \
                '- "author_purpose": (String) Briefly determine the author\'s likely primary purpose.\n' + \
                '- "potential_biases_identified": (List of strings) Enumerate any specific signs of potential bias or subjectivity observed.\n' + \
                '- "short_summary": (String) A concise summary of the article\'s main content in 2-4 sentences.\n' + \
                '- "index_of_credibility": (Float, 0.0-1.0) Calculate an overall index of credibility based on the above factors.\n' + \
                '- "published_date": (String, YYYY-MM-DD or N/A) The publication date of the article.'

        try:
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

        except anthropic.APIError as e:
            app.logger.error('Anthropic API Error: ' + str(e))
            raise
        except json.JSONDecodeError as e:
            app.logger.error('JSON Decode Error from Claude\'s response: ' + str(e) + '. Raw response was: ' + raw_json_text)
            raise ValueError('Failed to parse AI response: ' + str(e) + '. Raw: ' + raw_json_text[:500] + '...')
        except Exception as e:
            app.logger.error('Unexpected error during Claude analysis: ' + str(e))
            raise

def extract_text_from_url(url):
    try:
        clean_url = re.sub(r'/amp(/)?$', '', url)
        article = Article(clean_url)
        article.download()
        article.parse()
        text = article.text.strip()
        title = article.title.strip() if article.title else ''
        source = urlparse(clean_url).netloc.replace('www.', '')
        return text, source, title
    except Exception as e:
        app.logger.error('Error extracting article from URL ' + url + ': ' + str(e))
        return '', '', ''

def calculate_credibility_level(integrity, fact_check_needed, sentiment, bias):
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
    conn = None
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()

        integrity = analysis_result.get('news_integrity', 0.0)
        fact_check_needed = analysis_result.get('fact_check_needed_score', 1.0)
        sentiment = analysis_result.get('sentiment_score', 0.5)
        bias = analysis_result.get('bias_score', 1.0)
        short_summary = analysis_result.get('short_summary', 'Summary not available.')
        index_of_credibility = analysis_result.get('index_of_credibility', 0.0)

        credibility_level = calculate_credibility_level(integrity, fact_check_needed, sentiment, bias)
        db_url = url if url else 'no_url_' + str(datetime.now(UTC).timestamp())

        c.execute('''INSERT INTO news (url, title, source, content, integrity, fact_check, sentiment, bias,
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
                (db_url, title, source, content, integrity, fact_check_needed, sentiment, bias,
                 credibility_level, short_summary, index_of_credibility))

        # Обновляем таблицу source_stats
        c.execute('SELECT high, medium, low, total_analyzed FROM source_stats WHERE source = ?', (source,))
        row = c.fetchone()

        if row:
            high, medium, low, total = row
            if credibility_level == 'High': high += 1
            elif credibility_level == 'Medium': medium += 1
            else: low += 1
            total += 1
            c.execute('''UPDATE source_stats SET high=?, medium=?, low=?, total_analyzed=? WHERE source=?''',
                    (high, medium, low, total, source))
        else:
            high = 1 if credibility_level == 'High' else 0
            medium = 1 if credibility_level == 'Medium' else 0
            low = 1 if credibility_level == 'Low' else 0
            c.execute('''INSERT INTO source_stats (source, high, medium, low, total_analyzed)
                        VALUES (?, ?, ?, ?, ?)''', (source, high, medium, low, 1))

        conn.commit()
        return credibility_level
    except sqlite3.Error as e:
        app.logger.error('Database error in save_analysis_to_db: ' + str(e))
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()

def process_article_analysis(input_text, source_name_manual):
    article_url = None
    article_content = input_text
    article_title = 'User-provided Text'
    source_name = source_name_manual if source_name_manual else 'Direct Input'

    if input_text.strip().startswith('http'):
        article_url = input_text.strip()
        app.logger.info('Input is a URL: ' + article_url)
        content_from_url, source_from_url, title_from_url = extract_text_from_url(article_url)

        if content_from_url and len(content_from_url) >= 100:
            article_content, source_name, article_title = content_from_url, source_from_url, title_from_url
            app.logger.info('Extracted from URL. Source: ' + source_name + ', Title: ' + article_title)
        else:
            if not content_from_url:
                return ('❌ Failed to extract content from the provided URL. Please check the link or provide text directly.', None, None)
            else:
                return ('❌ Extracted article content is too short for analysis (min 100 chars).', None, None)

    if not article_content or len(article_content) < 100:
        return ('❌ Article content is too short for analysis (min 100 chars).', None, None)

    if not source_name:
        source_name = 'Unknown Source'

    analyzer = ClaudeNewsAnalyzer(ANTHROPIC_API_KEY, MODEL_NAME)
    try:
        analysis_result = analyzer.analyze_article_text(article_content, source_name)
    except Exception as e:
        app.logger.error('Error during Claude analysis: ' + str(e))
        return (f'❌ Error during analysis: {str(e)}', None, None)

    try:
        credibility_saved = save_analysis_to_db(article_url, article_title, source_name, article_content, analysis_result)
        app.logger.info('Analysis saved to DB. Overall Credibility: ' + credibility_saved)
    except Exception as e:
        app.logger.error('Error saving analysis to database: ' + str(e))
        return (f'❌ Error saving analysis: {str(e)}', None, None)

    ni = analysis_result.get('news_integrity', 0.0)
    fcn = analysis_result.get('fact_check_needed_score', 1.0)
    ss = analysis_result.get('sentiment_score', 0.5)
    bs = analysis_result.get('bias_score', 1.0)
    topics = analysis_result.get('topics', [])
    key_arguments = analysis_result.get('key_arguments', [])
    mentioned_facts = analysis_result.get('mentioned_facts', [])
    author_purpose = analysis_result.get('author_purpose', 'N/A')
    potential_biases_identified = analysis_result.get('potential_biases_identified', [])
    short_summary = analysis_result.get('short_summary', 'N/A')
    index_of_credibility = analysis_result.get('index_of_credibility', 0.0)

    factuality_display_score = 1.0 - fcn

    output_md = '### 📊 Credibility Analysis for: "' + article_title + '"\n' + \
                '**Source:** ' + source_name + '\n' + \
                '**Media Owner:** ' + media_owners.get(source_name, 'Unknown Owner') + '\n' + \
                '**Overall Calculated Credibility:** **' + credibility_saved + '** (' + str(index_of_credibility*100) + '%)' + '\n\n' + \
                '---' + '\n' + \
                '#### 📊 Analysis Scores:' + '\n' + \
                '- **Integrity Score:** ' + str(ni*100) + '% - Measures the overall integrity and trustworthiness.' + '\n' + \
                '- **Factuality Score:** ' + str(factuality_display_score*100) + '% - Indicates likelihood of needing fact-checking.' + '\n' + \
                '- **Sentiment Score:** ' + str(ss) + ' - Overall emotional tone (0.0 negative, 0.5 neutral, 1.0 positive).' + '\n' + \
                '- **Bias Score:** ' + str(bs*100) + '% - Degree of perceived bias (0.0 low, 1.0 high).' + '\n' + \
                '- **Index of Credibility:** ' + str(index_of_credibility*100) + '% - Overall credibility index.' + '\n\n' + \
                '---' + '\n' + \
                '#### 📝 Summary:' + '\n' + \
                short_summary + '\n\n' + \
                '#### 🔑 Key Arguments:' + '\n' + \
                ('- ' + '\n- '.join(key_arguments) if key_arguments else 'N/A') + '\n\n' + \
                '#### 📈 Mentioned Facts/Data:' + '\n' + \
                ('- ' + '\n- '.join(mentioned_facts) if mentioned_facts else 'N/A') + '\n\n' + \
                '#### 🎯 Author\'s Purpose:' + '\n' + \
                author_purpose + '\n\n' + \
                '#### 🚩 Potential Biases Identified:' + '\n' + \
                ('- ' + '\n- '.join(potential_biases_identified) if potential_biases_identified else 'N/A') + '\n\n' + \
                '#### 🏷️ Main Topics Identified:' + '\n' + \
                (', '.join(topics) if topics else 'N/A') + '\n\n' + \
                '#### 📌 Media Owner Influence:' + '\n' + \
                'The media owner, ' + media_owners.get(source_name, 'Unknown Owner') + ', may influence source credibility.'

    scores_for_chart = {
        'Integrity': ni * 100,
        'Factuality': factuality_display_score * 100,
        'Neutral Sentiment': (1.0 - abs(ss - 0.5) * 2) * 100,
        'Low Bias': (1.0 - bs) * 100,
        'Overall Credibility Index': index_of_credibility * 100
    }

    return output_md, scores_for_chart, analysis_result

def generate_query(analysis_result):
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

    app.logger.info('Generated NewsAPI query: ' + query)
    return query

def make_newsapi_request(params):
    url = 'https://newsapi.org/v2/everything'
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        return data.get('articles', [])
    except requests.exceptions.RequestException as e:
        app.logger.error('NewsAPI Error: ' + str(e))
        if hasattr(e, 'response') and e.response is not None:
            app.logger.error('NewsAPI Response content: ' + e.response.text)
        return []
    except Exception as e:
        app.logger.error('Unexpected error in NewsAPI request: ' + str(e))
        return []

def fetch_similar_news(analysis_result, days_range=7, max_articles=3):
    if not NEWS_API_ENABLED:
        app.logger.warning('NEWS_API_KEY is not configured or enabled. Skipping similar news search.')
        return []

    initial_query = generate_query(analysis_result)
    url = 'https://newsapi.org/v2/everything'

    # Определяем диапазон дат
    original_published_date_str = analysis_result.get('published_date', 'N/A')
    end_date = datetime.now(UTC).date()

    if original_published_date_str and original_published_date_str != 'N/A':
        try:
            parsed_date = datetime.strptime(original_published_date_str, '%Y-%m-%d').date()
            start_date = parsed_date - timedelta(days=days_range)
            end_date = parsed_date + timedelta(days=days_range)
            app.logger.info('Using original article date (' + str(parsed_date) + ') for NewsAPI search range: ' + str(start_date) + ' to ' + str(end_date))
        except ValueError:
            app.logger.warning('Could not parse original article date ' + original_published_date_str + '. Using default range.')
            start_date = end_date - timedelta(days=days_range)
    else:
        start_date = end_date - timedelta(days=days_range)
        app.logger.info('No original article date found. Using default NewsAPI search range: ' + str(start_date) + ' to ' + str(end_date))

    # Попытка 1: Специфический запрос с надежными источниками
    params_specific = {
        'q': initial_query,
        'apiKey': NEWS_API_KEY,
        'language': 'en',
        'pageSize': max_articles * 2,
        'sortBy': 'relevancy',
        'from': start_date.strftime('%Y-%m-%d'),
        'to': end_date.strftime('%Y-%m-%d'),
    }

    if TRUSTED_NEWS_SOURCES_IDS:
        params_specific['sources'] = ','.join(TRUSTED_NEWS_SOURCES_IDS)

    articles_found = make_newsapi_request(params_specific)
    app.logger.info('[NewsAPI] Attempt 1 found ' + str(len(articles_found)) + ' articles.')

    # Попытка 2: Более широкий запрос, если первая попытка дала мало результатов
    if len(articles_found) < (max_articles / 2) and initial_query != 'current events OR news':
        app.logger.info('Few results from specific query, attempting broader search.')
        broader_query_terms = list(set(analysis_result.get('topics', [])[:3]))
        broader_query = ' OR '.join(['"' + term + '"' if ' ' in term else term for term in broader_query_terms if term and term not in stop_words_en])

        if not broader_query:
            broader_query = 'current events OR news'

        params_broad = {
            'q': broader_query,
            'apiKey': NEWS_API_KEY,
            'language': 'en',
            'pageSize': max_articles * 2,
            'sortBy': 'relevancy',
            'from': start_date.strftime('%Y-%m-%d'),
            'to': end_date.strftime('%Y-%m-%d'),
        }

        if TRUSTED_NEWS_SOURCES_IDS:
            params_broad['sources'] = ','.join(TRUSTED_NEWS_SOURCES_IDS)

        additional_articles = make_newsapi_request(params_broad)
        articles_found.extend(additional_articles)
        app.logger.info('[NewsAPI] Attempt 2 found ' + str(len(additional_articles)) + ' new articles. Total: ' + str(len(articles_found)))

    # Удаляем дубликаты статей
    unique_articles = {}
    for article in articles_found:
        if article.get('url'):
            unique_articles[article['url']] = article
    articles_found = list(unique_articles.values())

    if not articles_found:
        return []

    # Ранжируем статьи по релевантности и доверию
    ranked_articles = []
    predefined_trust_scores = {
        'bbc.com': 0.9, 'bbc.co.uk': 0.9, 'reuters.com': 0.95, 'apnews.com': 0.93,
        'nytimes.com': 0.88, 'theguardian.com': 0.85, 'wsj.com': 0.82,
        'cnn.com': 0.70, 'foxnews.com': 0.40, 'aljazeera.com': 0.80
    }

    # Объединяем термины из обоих запросов для проверки релевантности
    all_query_terms = []
    if 'initial_query' in locals():
        all_query_terms.extend([t.lower().replace('"', '') for t in initial_query.split(' AND ')])
    if 'broader_query' in locals():
        all_query_terms.extend([t.lower().replace('"', '') for t in broader_query.split(' OR ')])
    all_query_terms = list(set([t for t in all_query_terms if t and t not in stop_words_en]))

    for article in articles_found:
        source_domain = urlparse(article.get('url', '')).netloc.replace('www.', '')
        trust_score = predefined_trust_scores.get(source_domain, 0.5)
        article_text = (article.get('title', '') + ' ' + article.get('description', '')).lower()
        relevance_score = sum(1 for term in all_query_terms if term in article_text)
        final_score = (relevance_score * 10) + (trust_score * 5)
        ranked_articles.append((article, final_score))

    # Сортируем и возвращаем лучшие статьи
    ranked_articles.sort(key=lambda item: item[1], reverse=True)
    top_articles = [item[0] for item in ranked_articles[:max_articles]]
    app.logger.info('Returning ' + str(len(top_articles)) + ' top ranked similar articles.')
    return top_articles

def render_similar_articles_html(articles):
    if not articles:
        return '<p>No similar articles found for the selected criteria.</p>'

    predefined_trust_scores = {
        'bbc.com': 0.9, 'bbc.co.uk': 0.9, 'reuters.com': 0.95, 'apnews.com': 0.93,
        'nytimes.com': 0.88, 'theguardian.com': 0.85, 'wsj.com': 0.82,
        'cnn.com': 0.70, 'foxnews.com': 0.40, 'aljazeera.com': 0.80
    }

    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()

        html_items = []
        for art in articles:
            title = html.escape(art.get('title', 'No Title'))
            article_url = html.escape(art.get('url', '#'))
            source_api_name = html.escape(art.get('source', {}).get('name', 'Unknown Source'))

            published_at_raw = art.get('publishedAt', 'N/A')
            published_at_display = html.escape(published_at_raw.split('T')[0] if 'T' in published_at_raw and published_at_raw != 'N/A' else published_at_raw)

            description_raw = art.get('description', 'No description available.')
            if description_raw.startswith(art.get('title', '')):
                description_raw = description_raw[len(art.get('title', '')):].strip()
                if description_raw.startswith('- '):
                    description_raw = description_raw[2:].strip()
            description_display = html.escape(description_raw)

            domain = urlparse(art.get('url', '#')).netloc.replace('www.', '')
            trust_display = ''

            c.execute('SELECT high, medium, low, total_analyzed FROM source_stats WHERE source = ?', (domain,))
            row = c.fetchone()

            if row:
                high, medium, low, total_analyzed = row
                if total_analyzed > 0:
                    score = (high * 1.0 + medium * 0.5 + low * 0.0) / total_analyzed
                    trust_display = ' (Hist. Src. Credibility: ' + str(int(score*100)) + '%)'

            if not trust_display and domain in predefined_trust_scores:
                predefined_score = predefined_trust_scores.get(domain)
                trust_display = ' (Est. Src. Trust: ' + str(int(predefined_score*100)) + '%)'
            elif not trust_display:
                trust_display = ' (Src. Credibility: N/A)'

            html_items.append(
                '<div class="similar-article">' +
                '<h4><a href="' + article_url + '" target="_blank" rel="noopener noreferrer">' +
                title +
                '</a></h4>' +
                '<p><strong>Source:</strong> ' + source_api_name + trust_display + ' | <strong>Published:</strong> ' + published_at_display + '</p>' +
                '<p>' + description_display + '</p>' +
                '</div>' +
                '<hr>'
            )

        return (
            '<div class="similar-articles-container">' +
            '<h3>🔗 Similar News Articles (Ranked by Relevance & Trust):</h3>' +
            ''.join(html_items) +
            '</div>'
        )
    except sqlite3.Error as e:
        app.logger.error('Database error in render_similar_articles_html: ' + str(e))
        return '<p>Error retrieving similar articles data due to a database issue.</p>'
    finally:
        if conn:
            conn.close()

def get_source_reliability_data():
    try:
        conn = sqlite3.connect(DB_NAME)
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
    except sqlite3.Error as e:
        app.logger.error('Database error in get_source_reliability_data: ' + str(e))
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
    try:
        conn = sqlite3.connect(DB_NAME)
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
            display_title = title[:70] + '...' if title and len(title) > 70 else title if title else 'N/A'
            source_display = source if source else 'N/A'
            link_start = '<a href="' + url + '" target="_blank" rel="noopener noreferrer">' if url and url.startswith(('http://', 'https://')) else ''
            link_end = '</a>' if url and url.startswith(('http://', 'https://')) else ''
            summary_display = short_summary if short_summary else 'No summary available.'

            html_items.append(
                '<li>' +
                '<strong>' + date_str + '</strong>: ' + link_start + display_title + link_end + ' (' + source_display + ', ' + credibility + ')' +
                '<br>' +
                '<em>Summary:</em> ' + summary_display +
                '</li>'
            )

        return '<h3>📜 Recent Analyses:</h3><ul>' + ''.join(html_items) + '</ul>'
    except sqlite3.Error as e:
        app.logger.error('Database error in get_analysis_history_html: ' + str(e))
        return '<p>Error retrieving analysis history due to a database issue.</p>'
    finally:
        if conn:
            conn.close()

# Маршруты Flask
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json
        input_text = data.get('input_text')
        source_name_manual = data.get('source_name_manual')

        app.logger.info('Received input_text (first 50 chars): ' + input_text[:50] + '...')
        app.logger.info('Received source_name_manual: ' + str(source_name_manual))

        output_md, scores_for_chart, analysis_result = process_article_analysis(input_text, source_name_manual)

        if analysis_result is None:
            return jsonify({'error_message': output_md}), 400

        # Получаем похожие новости
        similar_news = fetch_similar_news(analysis_result)

        app.logger.info('Analysis result generated. Sending to client.')
        return jsonify({
            'output_md': output_md,
            'scores_for_chart': scores_for_chart,
            'analysis_result': analysis_result,
            'similar_news': render_similar_articles_html(similar_news)
        })
    except Exception as e:
        app.logger.error('Error in analyze endpoint: ' + str(e))
        return jsonify({'error_message': 'An error occurred during analysis: ' + str(e)}), 500

@app.route('/similar_articles', methods=['POST'])
def similar_articles():
    try:
        data = request.json
        analysis_result = data.get('analysis_result')

        if not analysis_result:
            return jsonify({'similar_html': '<p>No analysis result provided to fetch similar articles.</p>'})

        similar_articles_list = fetch_similar_news(analysis_result)
        return jsonify({
            'similar_html': render_similar_articles_html(similar_articles_list)
        })
    except Exception as e:
        app.logger.error('Error in similar_articles endpoint: ' + str(e))
        return jsonify({'similar_html': '<p>Error fetching similar articles: ' + str(e) + '</p>'}), 500

@app.route('/source_reliability_data')
def source_reliability_data():
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
                    source + '<br>Credibility: ' + str(int(score*100)) + '%<br>Articles: ' + str(total)
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
        app.logger.error('Error in source_reliability_data endpoint: ' + str(e))
        return jsonify({
            'sources': [],
            'credibility_indices_for_plot': [],
            'high_counts': [],
            'medium_counts': [],
            'low_counts': [],
            'total_analyzed_counts': []
        }), 500

@app.route('/analysis_history')
def analysis_history():
    try:
        history_html = get_analysis_history_html()
        return jsonify({'history_html': history_html})
    except Exception as e:
        app.logger.error('Error in analysis_history endpoint: ' + str(e))
        return jsonify({'history_html': '<p>Error retrieving analysis history: ' + str(e) + '</p>'}), 500

@app.route('/feedback', methods=['POST'])
def handle_feedback():
    """
    Обработчик для формы обратной связи.
    Принимает JSON с данными формы и сохраняет их в базу данных.
    Возвращает сообщение об успешной отправке или ошибке.
    """
    try:
        # Получаем данные из запроса
        data = request.get_json()

        # Проверяем, что все обязательные поля присутствуют
        required_fields = ['name', 'email', 'type', 'message']
        if not all(field in data for field in required_fields):
            return jsonify({'message': 'All fields are required'}), 400

        # Дополнительная валидация email
        if not re.match(r"[^@]+@[^@]+\.[^@]+", data['email']):
            return jsonify({'message': 'Invalid email address'}), 400

        # Подключаемся к базе данных и сохраняем фидбек
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()

        # Вставляем данные в таблицу feedback
        c.execute('''
            INSERT INTO feedback (name, email, type, message, date)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            data['name'],
            data['email'],
            data['type'],
            data['message'],
            datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')
        ))

        conn.commit()
        conn.close()

        # Возвращаем успешный ответ
        return jsonify({
            'status': 'success',
            'message': 'Thank you for your feedback! We appreciate it.'
        })

    except sqlite3.Error as e:
        app.logger.error(f"Database error handling feedback: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': 'Error saving your feedback. Please try again.'
        }), 500

    except Exception as e:
        app.logger.error(f"Unexpected error handling feedback: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': 'An unexpected error occurred. Please try again.'
        }), 500

@app.route('/feedback')
def feedback_page():
    """Отображает страницу с формой обратной связи"""
    return render_template('feedback.html')

# Инициализация базы данных
def initialize_database():
    ensure_db_schema()
    initialize_sources(INITIAL_SOURCE_COUNTS)
    check_database_integrity()

if __name__ == '__main__':
    # Проверка обязательных переменных окружения
    check_required_env_vars()

    # Настройка логирования
    setup_logging()

    # Инициализация базы данных
    initialize_database()

    # Запуск приложения
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
