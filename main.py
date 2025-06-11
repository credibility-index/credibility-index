import os
import logging
import sqlite3
import re
import json
import requests
import html
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse, urlunparse
from flask import Flask, request, jsonify, render_template, session, make_response
from werkzeug.middleware.proxy_fix import ProxyFix
import anthropic
from newspaper import Article, Config
from stop_words import get_stop_words
from flask_cors import CORS

# Initialize Flask application
app = Flask(__name__, static_folder='static', template_folder='templates')
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)
app.secret_key = os.getenv('SECRET_KEY', 'your-secret-key-here')

# Configure CORS for Railway
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "Accept"]
    }
})

# Environment variables with default values
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY', 'your-anthropic-api-key')
MODEL_NAME = os.getenv('ANTHROPIC_MODEL', 'claude-3-opus-20240229')
NEWS_API_KEY = os.getenv('NEWS_API_KEY', 'your-news-api-key')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global constants
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

# Database initialization
DB_NAME = 'news_analysis.db'

def get_db_connection():
    """Create database connection"""
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def initialize_database():
    """Initialize database schema and populate with test data if empty"""
    try:
        # Check if database directory exists, create if not
        db_dir = os.path.dirname(DB_NAME)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir)

        with get_db_connection() as conn:
            cursor = conn.cursor()

            # Create tables
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS news (
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
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS source_stats (
                    source TEXT PRIMARY KEY,
                    high INTEGER DEFAULT 0,
                    medium INTEGER DEFAULT 0,
                    low INTEGER DEFAULT 0,
                    total_analyzed INTEGER DEFAULT 0
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    email TEXT NOT NULL,
                    type TEXT NOT NULL,
                    message TEXT NOT NULL,
                    date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            conn.commit()

            # Populate with test data only if tables are empty
            cursor.execute("SELECT COUNT(*) FROM source_stats")
            if cursor.fetchone()[0] == 0:
                populate_test_data()

            logger.info("Database initialized successfully")

    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        raise

def populate_test_data():
    """Populate database with test data for demonstration"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()

            # Test data for various sources
            test_sources = [
                ('bbc.com', 45, 10, 5),
                ('reuters.com', 50, 5, 2),
                ('foxnews.com', 15, 20, 30),
                ('cnn.com', 30, 25, 10),
                ('nytimes.com', 35, 15, 5),
                ('theguardian.com', 40, 10, 3),
                ('apnews.com', 48, 5, 2),
                ('washingtonpost.com', 38, 12, 5),
                ('bloomberg.com', 42, 8, 5),
                ('wsj.com', 37, 15, 8),
                ('aljazeera.com', 28, 18, 10),
                ('dailymail.co.uk', 12, 25, 30),
                ('breitbart.com', 8, 15, 40),
                ('infowars.com', 5, 10, 50),
                ('rt.com', 10, 20, 35)
            ]

            for source, high, medium, low in test_sources:
                total = high + medium + low
                cursor.execute('''
                    INSERT INTO source_stats (source, high, medium, low, total_analyzed)
                    VALUES (?, ?, ?, ?, ?)
                ''', (source, high, medium, low, total))

            # Test data for news table
            test_news = [
                (
                    "BBC News: Global Climate Summit Begins",
                    "bbc.com",
                    "Global leaders gather for climate summit...",
                    0.92, 0.15, 0.65, 0.20,
                    "High", 0.88,
                    "https://bbc.com/climate-summit",
                    datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
                    "Global leaders gather to discuss climate change solutions"
                ),
                (
                    "Reuters: Stock Markets Reach Record Highs",
                    "reuters.com",
                    "Stock markets worldwide reached record highs...",
                    0.95, 0.10, 0.70, 0.15,
                    "High", 0.91,
                    "https://reuters.com/stock-markets",
                    datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
                    "Global stock markets reached new record highs"
                ),
            ]

            for item in test_news:
                cursor.execute('''
                    INSERT INTO news
                    (title, source, content, integrity, fact_check, sentiment, bias,
                     credibility_level, index_of_credibility, url, analysis_date, short_summary)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', item)

            conn.commit()
            logger.info("Test data added to database successfully")

    except Exception as e:
        logger.error(f"Error populating test data: {str(e)}")
        raise

def get_source_credibility_data():
    """Get source credibility data from database"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT source, high, medium, low, total_analyzed
                FROM source_stats
                ORDER BY total_analyzed DESC, source ASC
            ''')

            data = cursor.fetchall()
            sources = []
            credibility_scores = []
            high_counts = []
            medium_counts = []
            low_counts = []
            total_counts = []

            for source, high, medium, low, total in data:
                total_current = high + medium + low
                if total_current > 0:
                    score = (high * 1.0 + medium * 0.5 + low * 0.0) / total_current
                else:
                    score = 0.5

                sources.append(source)
                credibility_scores.append(round(score, 2))
                high_counts.append(high)
                medium_counts.append(medium)
                low_counts.append(low)
                total_counts.append(total_current)

            return {
                'sources': sources,
                'credibility_scores': credibility_scores,
                'high_counts': high_counts,
                'medium_counts': medium_counts,
                'low_counts': low_counts,
                'total_counts': total_counts
            }
    except Exception as e:
        logger.error(f"Error getting source credibility data: {str(e)}")
        return {
            'sources': [],
            'credibility_scores': [],
            'high_counts': [],
            'medium_counts': [],
            'low_counts': [],
            'total_counts': []
        }

def get_analysis_history():
    """Get analysis history from database"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT url, title, source, credibility_level, short_summary,
                       strftime("%Y-%m-%d %H:%M", analysis_date) as formatted_date
                FROM news
                ORDER BY analysis_date DESC
                LIMIT 15
            ''')

            rows = cursor.fetchall()
            history = []

            for row in rows:
                history.append({
                    'url': row['url'],
                    'title': row['title'],
                    'source': row['source'],
                    'credibility': row['credibility_level'],
                    'summary': row['short_summary'],
                    'date': row['formatted_date']
                })

            return history
    except Exception as e:
        logger.error(f"Error getting analysis history: {str(e)}")
        return []

# Configure newspaper library
user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
config = Config()
config.browser_user_agent = user_agent
config.request_timeout = 30

@app.before_request
def before_request():
    """Set up before each request"""
    if request.path.startswith('/static/'):
        return

    if request.method == 'OPTIONS':
        response = make_response()
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, Accept'
        return response

@app.after_request
def add_security_headers(response):
    """Add security headers"""
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, Accept'
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    return response
# Основные маршруты приложения
@app.route('/')
def home():
    """Home page route"""
    return render_template('index.html')

@app.route('/faq')
def faq():
    """FAQ page route"""
    return render_template('faq.html')

@app.route('/feedback', methods=['GET', 'POST'])
def feedback():
    """Feedback page and form handler"""
    if request.method == 'POST':
        try:
            # Получаем данные из формы
            name = request.form.get('name')
            email = request.form.get('email')
            feedback_type = request.form.get('type')
            message = request.form.get('message')

            # Проверяем заполненность всех полей
            if not all([name, email, feedback_type, message]):
                return render_template('feedback.html', error="All fields are required")

            # Проверяем корректность email
            if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
                return render_template('feedback.html', error="Invalid email address")

            # Сохраняем feedback в базу данных
            with get_db_connection() as conn:
                conn.execute('''
                    INSERT INTO feedback (name, email, type, message, date)
                    VALUES (?, ?, ?, ?, ?)
                ''', (name, email, feedback_type, message, datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')))
                conn.commit()

            # Перенаправляем на страницу успешной отправки
            return render_template('feedback_success.html')

        except Exception as e:
            logger.error(f'Error saving feedback: {str(e)}')
            return render_template('feedback.html', error="Error saving feedback")

    # Если GET запрос - просто показываем форму
    return render_template('feedback.html')

# Маршрут для получения данных чарта надежности источников
@app.route('/source-credibility-chart', methods=['GET'])
def source_credibility_chart():
    """Endpoint for getting source credibility chart data"""
    try:
        # Получаем данные из базы
        chart_data = get_source_credibility_data()

        # Если данных нет, добавляем тестовые
        if not chart_data['sources']:
            populate_test_data()
            chart_data = get_source_credibility_data()

        # Возвращаем данные в формате JSON
        return jsonify({
            'status': 'success',
            'data': chart_data
        })

    except Exception as e:
        logger.error(f"Error in source_credibility_chart endpoint: {str(e)}")
        return jsonify({
            'error': 'An error occurred while fetching chart data',
            'status': 500,
            'details': str(e)
        }), 500

# Маршрут для получения истории анализа
@app.route('/analysis-history', methods=['GET'])
def analysis_history():
    """Endpoint for getting analysis history"""
    try:
        # Получаем историю из базы данных
        history = get_analysis_history()
        return jsonify({
            'status': 'success',
            'history': history
        })
    except Exception as e:
        logger.error(f"Error in analysis_history endpoint: {str(e)}")
        return jsonify({
            'error': 'An error occurred while fetching analysis history',
            'status': 500,
            'details': str(e)
        }), 500

# Основной маршрут для анализа статьи
@app.route('/analyze', methods=['POST', 'OPTIONS'])
def analyze():
    """Analyze article endpoint with comprehensive error handling"""
    # Обработка OPTIONS запроса для CORS
    if request.method == 'OPTIONS':
        response = make_response()
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Accept'
        return response

    try:
        # Проверяем, что запрос в формате JSON
        if not request.is_json:
            return jsonify({
                'error': 'Request must be JSON',
                'status': 400,
                'details': 'Content-Type header must be application/json'
            }), 400

        # Получаем данные из запроса
        data = request.get_json()
        if not data:
            return jsonify({
                'error': 'Empty request body',
                'status': 400
            }), 400

        # Проверяем наличие обязательного поля
        if 'input_text' not in data:
            return jsonify({
                'error': 'Missing input text',
                'status': 400,
                'details': 'input_text field is required'
            }), 400

        # Очищаем входные данные
        input_text = data['input_text'].strip()
        source_name = data.get('source_name_manual', 'Direct Input').strip()

        # Проверяем, что текст не пустой
        if not input_text:
            return jsonify({
                'error': 'Empty input text',
                'status': 400,
                'details': 'Input text cannot be empty'
            }), 400

        # Обрабатываем URL или текст
        if input_text.startswith(('http://', 'https://')):
            try:
                # Извлекаем контент из URL
                content, source, title = extract_text_from_url(input_text)
                if not content:
                    return jsonify({
                        'error': 'Could not extract article content',
                        'status': 400,
                        'details': 'Failed to download or parse the article from the provided URL'
                    }), 400
            except Exception as e:
                logger.error(f"Error processing URL: {str(e)}")
                return jsonify({
                    'error': 'Error processing URL',
                    'status': 400,
                    'details': str(e)
                }), 400
        else:
            # Проверяем минимальную длину текста
            if len(input_text) < 100:
                return jsonify({
                    'error': 'Content too short',
                    'status': 400,
                    'details': 'Minimum 100 characters required'
                }), 400
            content = input_text
            title = 'User-provided Text'
            source = source_name

        # Анализируем статью (используем mock данные, так как нет реального API ключа)
        try:
            analysis = {
                'news_integrity': 0.85,
                'fact_check_needed_score': 0.2,
                'sentiment_score': 0.6,
                'bias_score': 0.3,
                'topics': ['politics', 'economy'],
                'key_arguments': ['Argument 1', 'Argument 2'],
                'mentioned_facts': ['Fact 1', 'Fact 2'],
                'author_purpose': 'To inform',
                'potential_biases_identified': ['Bias 1'],
                'short_summary': 'This is a test summary',
                'index_of_credibility': 0.75
            }
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            return jsonify({
                'error': 'Analysis failed',
                'status': 500,
                'details': str(e)
            }), 500

        # Сохраняем анализ в базу данных
        try:
            credibility = save_analysis(
                input_text if input_text.startswith(('http://', 'https://')) else None,
                title,
                source,
                content,
                analysis
            )
        except Exception as e:
            logger.error(f"Failed to save analysis: {str(e)}")
            return jsonify({
                'error': 'Failed to save analysis',
                'status': 500,
                'details': str(e)
            }), 500

        # Сохраняем результат анализа в сессии
        session['last_analysis_result'] = analysis

        # Получаем похожие статьи (используем mock данные)
        same_topic_articles = [
            {
                'title': 'Similar Article 1',
                'url': 'https://example.com/article1',
                'source': {'name': 'Example News'},
                'publishedAt': '2023-01-01T00:00:00Z',
                'description': 'This is a similar article about the same topic.'
            }
        ]
        same_topic_html = render_same_topic_articles_html(same_topic_articles)

        # Получаем данные о надежности источников
        source_credibility_data = get_source_credibility_data()

        # Получаем историю анализа
        analysis_history = get_analysis_history()

        # Подготавливаем ответ
        response_data = {
            'status': 'success',
            'analysis': analysis,
            'credibility': credibility,
            'title': title,
            'source': source,
            'scores_for_chart': {
                'Integrity': analysis.get('news_integrity', 0.0),
                'Factuality': 1 - analysis.get('fact_check_needed_score', 1.0),
                'Sentiment': analysis.get('sentiment_score', 0.5),
                'Bias': 1 - analysis.get('bias_score', 1.0),
                'Overall Credibility Index': analysis.get('index_of_credibility', 0.0)
            },
            'source_credibility_data': source_credibility_data,
            'analysis_history': analysis_history,
            'same_topic_html': same_topic_html,
            'output': format_analysis_results(title, source, analysis, credibility)
        }

        return jsonify(response_data)

    except Exception as e:
        logger.error(f"Unexpected error in analyze endpoint: {str(e)}", exc_info=True)
        return jsonify({
            'error': 'Internal server error',
            'status': 500,
            'details': str(e)
        }), 500

# Вспомогательные функции для анализа

def extract_text_from_url(url):
    """Extract text from URL with improved error handling"""
    try:
        logger.info(f"Processing URL: {url}")

        # Нормализуем URL
        parsed = urlparse(url)
        clean_url = urlunparse(parsed._replace(scheme=parsed.scheme.lower(), netloc=parsed.netloc.lower()))

        # Проверяем на видео контент
        if any(domain in url for domain in ['youtube.com', 'vimeo.com']):
            logger.info("Video content detected")
            return "Video content detected", parsed.netloc.replace('www.', ''), "Video: " + url

        # Настраиваем статью с таймаутом и user agent
        article = Article(clean_url, config=config)

        # Загружаем и парсим статью
        article.download()
        if article.download_state != 2:
            logger.error(f"Failed to download article from {url}")
            return None, None, None

        article.parse()
        if not article.text or len(article.text.strip()) < 100:
            logger.warning(f"Short or empty content from {url}")
            return None, None, None

        # Извлекаем домен и заголовок
        domain = parsed.netloc.replace('www.', '')
        title = article.title.strip() if article.title else "No title"

        logger.info(f"Successfully extracted content from {url}")
        return article.text.strip(), domain, title

    except Exception as e:
        logger.error(f"Error extracting article from {url}: {str(e)}")
        return None, None, None

def calculate_credibility(integrity, fact_check, sentiment, bias):
    """Calculate credibility level"""
    fact_check_score = 1.0 - fact_check
    sentiment_score = 1.0 - abs(sentiment - 0.5) * 2
    bias_score = 1.0 - bias

    score = (integrity * 0.45) + (fact_check_score * 0.35) + (sentiment_score * 0.10) + (bias_score * 0.10)

    if score >= 0.75:
        return 'High'
    if score >= 0.5:
        return 'Medium'
    return 'Low'

def save_analysis(url, title, source, content, analysis):
    """Save analysis to database"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()

            # Извлекаем данные из анализа
            integrity = analysis.get('news_integrity', 0.0)
            fact_check = analysis.get('fact_check_needed_score', 1.0)
            sentiment = analysis.get('sentiment_score', 0.5)
            bias = analysis.get('bias_score', 1.0)
            summary = analysis.get('short_summary', 'No summary')
            credibility = analysis.get('index_of_credibility', 0.0)

            # Рассчитываем уровень достоверности
            level = calculate_credibility(integrity, fact_check, sentiment, bias)
            db_url = url if url else f'text_{datetime.now(timezone.utc).timestamp()}'

            # Сохраняем анализ в базу данных
            cursor.execute('''
                INSERT INTO news
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
                analysis_date=CURRENT_TIMESTAMP
            ''', (db_url, title, source, content, integrity, fact_check,
                  sentiment, bias, level, summary, credibility))

            # Обновляем статистику источников
            cursor.execute('SELECT high, medium, low, total_analyzed FROM source_stats WHERE source = ?', (source,))
            row = cursor.fetchone()

            if row:
                high, medium, low, total = row
                if level == 'High': high += 1
                elif level == 'Medium': medium += 1
                else: low += 1
                total += 1
                cursor.execute('''
                    UPDATE source_stats SET high=?, medium=?, low=?, total_analyzed=?
                    WHERE source=?
                ''', (high, medium, low, total, source))
            else:
                counts = {'High': 1, 'Medium': 0, 'Low': 0}
                counts[level] = 1
                cursor.execute('''
                    INSERT INTO source_stats
                    (source, high, medium, low, total_analyzed)
                    VALUES (?, ?, ?, ?, ?)
                ''', (source, counts['High'], counts['Medium'], counts['Low'], 1))

            conn.commit()
            return level
    except Exception as e:
        logger.error(f"Error saving analysis: {str(e)}")
        raise ValueError("Failed to save analysis")

def generate_query(analysis_result):
    """Generate query for finding similar articles"""
    topics = analysis_result.get('topics', [])
    key_arguments = analysis_result.get('key_arguments', [])
    mentioned_facts = analysis_result.get('mentioned_facts', [])

    all_terms = []
    for phrase_list in [topics, key_arguments]:
        for phrase in phrase_list:
            if not phrase.strip():
                continue
            if ' ' in phrase.strip() and len(phrase.strip().split()) > 1:
                all_terms.append('"' + phrase.strip() + '"')
            else:
                all_terms.append(phrase.strip())

    for fact in mentioned_facts:
        if not fact.strip():
            continue
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
    """Make request to NewsAPI"""
    url = 'https://newsapi.org/v2/everything'
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        return response.json().get('articles', [])
    except Exception as e:
        logger.error(f'NewsAPI Request Error: {str(e)}')
        return []

def fetch_same_topic_articles(analysis_result, page=1, per_page=3):
    """Fetch similar articles by topic with pagination"""
    global predefined_trust_scores

    if not NEWS_API_KEY:
        logger.warning('NEWS_API_KEY is not configured. Skipping similar news search.')
        return []

    try:
        query = generate_query(analysis_result)
        end_date = datetime.now(timezone.utc).date()
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

    except Exception as e:
        logger.error(f"Error in fetch_same_topic_articles: {str(e)}")
        return []

def render_same_topic_articles_html(articles):
    """Render HTML for similar articles"""
    if not articles:
        return '<div class="alert alert-info">No similar articles found</div>'

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
            f'''
            <div class="similar-article">
                <h4><a href="{article_url}" target="_blank" rel="noopener noreferrer">{title}</a></h4>
                <div class="article-meta">
                    <span class="article-source"><i class="bi bi-newspaper"></i> {source_api_name}</span>
                    <span class="article-date"><i class="bi bi-calendar"></i> {published_at}</span>
                    <span class="article-credibility">Credibility: {int(trust_score*100)}%</span>
                </div>
                <p class="article-description">{description}</p>
            </div>
            '''
        )

    return '<div class="similar-articles-container">' + ''.join(html_items) + '</div>'

def format_analysis_results(title, source, analysis, credibility):
    """Format analysis results for display"""
    try:
        output = {
            'title': title,
            'source': source,
            'credibility': credibility,
            'analysis': analysis,
            'scores': {
                'Integrity': analysis.get('news_integrity', 0.0),
                'Factuality': 1 - analysis.get('fact_check_needed_score', 1.0),
                'Sentiment': analysis.get('sentiment_score', 0.5),
                'Bias': 1 - analysis.get('bias_score', 1.0),
                'Overall Credibility Index': analysis.get('index_of_credibility', 0.0)
            },
            'output_md': f"""
            <div class="analysis-section">
                <h2>Article Information</h2>
                <p><strong>Title:</strong> {html.escape(title)}</p>
                <p><strong>Source:</strong> {html.escape(source)}</p>
                <p><strong>Credibility Level:</strong> <span class="credibility-badge {credibility.lower()}">{credibility}</span></p>
            </div>

            <div class="analysis-section">
                <h2>Analysis Scores</h2>
                <div class="row">
                    <div class="col-md-3">
                        <div class="score-item">
                            <div class="score-name">Integrity</div>
                            <div class="score-value">{analysis.get('news_integrity', 0.0):.2f}</div>
                            <div class="score-description">Overall integrity and trustworthiness</div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="score-item">
                            <div class="score-name">Factuality</div>
                            <div class="score-value">{1 - analysis.get('fact_check_needed_score', 1.0):.2f}</div>
                            <div class="score-description">Likelihood that claims are factual</div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="score-item">
                            <div class="score-name">Sentiment</div>
                            <div class="score-value">{analysis.get('sentiment_score', 0.5):.2f}</div>
                            <div class="score-description">Emotional tone (0.0 negative, 0.5 neutral, 1.0 positive)</div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="score-item">
                            <div class="score-name">Bias</div>
                            <div class="score-value">{1 - analysis.get('bias_score', 1.0):.2f}</div>
                            <div class="score-description">Degree of perceived bias (1.0 low bias, 0.0 high bias)</div>
                        </div>
                    </div>
                </div>
            </div>

            <div class="analysis-section">
                <h2>Additional Information</h2>
                <div class="detail-item">
                    <h4>Author Purpose</h4>
                    <p>{html.escape(analysis.get('author_purpose', 'Not specified'))}</p>
                </div>

                <div class="detail-item">
                    <h4>Short Summary</h4>
                    <p>{html.escape(analysis.get('short_summary', 'No summary available'))}</p>
                </div>

                <div class="detail-item">
                    <h4>Topics</h4>
                    <div class="d-flex flex-wrap gap-2">
                        {' '.join(f'<span class="badge bg-primary">{html.escape(topic)}</span>' for topic in analysis.get('topics', []))}
                    </div>
                </div>

                <div class="detail-item">
                    <h4>Key Arguments</h4>
                    <ul class="list-unstyled">
                        {''.join(f'<li>{html.escape(arg)}</li>' for arg in analysis.get('key_arguments', []))}
                    </ul>
                </div>

                <div class="detail-item">
                    <h4>Potential Biases Identified</h4>
                    <ul class="list-unstyled">
                        {''.join(f'<li>{html.escape(bias)}</li>' for bias in analysis.get('potential_biases_identified', []))}
                    </ul>
                </div>
            </div>
            """
        }
        return output
    except Exception as e:
        logger.error(f"Error formatting analysis results: {str(e)}")
        return {"error": "Error formatting analysis results"}
# Основные маршруты приложения
@app.route('/')
def home():
    """Home page route"""
    return render_template('index.html')

@app.route('/faq')
def faq():
    """FAQ page route"""
    return render_template('faq.html')

@app.route('/feedback', methods=['GET', 'POST'])
def feedback():
    """Feedback page and form handler"""
    if request.method == 'POST':
        try:
            name = request.form.get('name')
            email = request.form.get('email')
            feedback_type = request.form.get('type')
            message = request.form.get('message')

            if not all([name, email, feedback_type, message]):
                return render_template('feedback.html', error="All fields are required")

            if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
                return render_template('feedback.html', error="Invalid email address")

            with get_db_connection() as conn:
                conn.execute('''
                    INSERT INTO feedback (name, email, type, message, date)
                    VALUES (?, ?, ?, ?, ?)
                ''', (name, email, feedback_type, message, datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')))
                conn.commit()

            return render_template('feedback_success.html')

        except Exception as e:
            logger.error(f'Error saving feedback: {str(e)}')
            return render_template('feedback.html', error="Error saving feedback")

    return render_template('feedback.html')

@app.route('/source-credibility-chart', methods=['GET'])
def source_credibility_chart():
    """Endpoint for getting source credibility chart data"""
    try:
        chart_data = get_source_credibility_data()

        if not chart_data['sources']:
            populate_test_data()
            chart_data = get_source_credibility_data()

        return jsonify({
            'status': 'success',
            'data': {
                'sources': chart_data['sources'],
                'credibility_scores': chart_data['credibility_scores'],
                'high_counts': chart_data['high_counts'],
                'medium_counts': chart_data['medium_counts'],
                'low_counts': chart_data['low_counts'],
                'total_counts': chart_data['total_counts']
            }
        })

    except Exception as e:
        logger.error(f"Error in source_credibility_chart endpoint: {str(e)}")
        return jsonify({
            'error': 'An error occurred while fetching chart data',
            'status': 500,
            'details': str(e)
        }), 500

@app.route('/analysis-history', methods=['GET'])
def analysis_history():
    """Endpoint for getting analysis history"""
    try:
        history = get_analysis_history()
        return jsonify({
            'status': 'success',
            'history': history
        })
    except Exception as e:
        logger.error(f"Error in analysis_history endpoint: {str(e)}")
        return jsonify({
            'error': 'An error occurred while fetching analysis history',
            'status': 500,
            'details': str(e)
        }), 500

@app.route('/analyze', methods=['POST', 'OPTIONS'])
def analyze():
    """Analyze article endpoint with comprehensive error handling"""
    if request.method == 'OPTIONS':
        response = make_response()
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Accept'
        return response

    try:
        # Validate request content type
        if not request.is_json:
            return jsonify({
                'error': 'Request must be JSON',
                'status': 400,
                'details': 'Content-Type header must be application/json'
            }), 400

        # Get and validate data
        data = request.get_json()
        if not data:
            return jsonify({
                'error': 'Empty request body',
                'status': 400
            }), 400

        if 'input_text' not in data:
            return jsonify({
                'error': 'Missing input text',
                'status': 400,
                'details': 'input_text field is required'
            }), 400

        input_text = data['input_text'].strip()
        source_name = data.get('source_name_manual', 'Direct Input').strip()

        if not input_text:
            return jsonify({
                'error': 'Empty input text',
                'status': 400,
                'details': 'Input text cannot be empty'
            }), 400

        # Process article
        if input_text.startswith(('http://', 'https://')):
            try:
                content, source, title = extract_text_from_url(input_text)
                if not content:
                    return jsonify({
                        'error': 'Could not extract article content',
                        'status': 400,
                        'details': 'Failed to download or parse the article from the provided URL'
                    }), 400
            except Exception as e:
                logger.error(f"Error processing URL: {str(e)}")
                return jsonify({
                    'error': 'Error processing URL',
                    'status': 400,
                    'details': str(e)
                }), 400
        else:
            if len(input_text) < 100:
                return jsonify({
                    'error': 'Content too short',
                    'status': 400,
                    'details': 'Minimum 100 characters required'
                }), 400
            content = input_text
            title = 'User-provided Text'
            source = source_name

        # Analyze with mock data (since we don't have real API key)
        try:
            analysis = {
                'news_integrity': 0.85,
                'fact_check_needed_score': 0.2,
                'sentiment_score': 0.6,
                'bias_score': 0.3,
                'topics': ['politics', 'economy'],
                'key_arguments': ['Argument 1', 'Argument 2'],
                'mentioned_facts': ['Fact 1', 'Fact 2'],
                'author_purpose': 'To inform',
                'potential_biases_identified': ['Bias 1'],
                'short_summary': 'This is a test summary',
                'index_of_credibility': 0.75
            }
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            return jsonify({
                'error': 'Analysis failed',
                'status': 500,
                'details': str(e)
            }), 500

        # Save to database
        try:
            credibility = save_analysis(
                input_text if input_text.startswith(('http://', 'https://')) else None,
                title,
                source,
                content,
                analysis
            )
        except Exception as e:
            logger.error(f"Failed to save analysis: {str(e)}")
            return jsonify({
                'error': 'Failed to save analysis',
                'status': 500,
                'details': str(e)
            }), 500

        # Store analysis result in session
        session['last_analysis_result'] = analysis

        # Get similar articles (using mock data)
        same_topic_articles = [
            {
                'title': 'Similar Article 1',
                'url': 'https://example.com/article1',
                'source': {'name': 'Example News'},
                'publishedAt': '2023-01-01T00:00:00Z',
                'description': 'This is a similar article about the same topic.'
            }
        ]
        same_topic_html = render_same_topic_articles_html(same_topic_articles)

        # Get source credibility data
        source_credibility_data = get_source_credibility_data()

        # Get analysis history
        analysis_history = get_analysis_history()

        # Prepare response with structure matching your index.html expectations
        response_data = {
            'status': 'success',
            'analysis': analysis,
            'credibility': credibility,
            'title': title,
            'source': source,
            'scores_for_chart': {
                'news_integrity': analysis.get('news_integrity', 0.0),
                'fact_check_needed_score': analysis.get('fact_check_needed_score', 1.0),
                'sentiment_score': analysis.get('sentiment_score', 0.5),
                'bias_score': analysis.get('bias_score', 1.0),
                'index_of_credibility': analysis.get('index_of_credibility', 0.0)
            },
            'source_credibility_data': source_credibility_data,
            'analysis_history': analysis_history,
            'same_topic_html': same_topic_html,
            'output': {
                'output_md': f"""
                <div class="analysis-section">
                    <h2>Article Information</h2>
                    <p><strong>Title:</strong> {html.escape(title)}</p>
                    <p><strong>Source:</strong> {html.escape(source)}</p>
                    <p><strong>Credibility Level:</strong> {credibility}</p>
                </div>

                <div class="analysis-section">
                    <h2>Analysis Scores</h2>
                    <div class="row">
                        <div class="col-md-3">
                            <div class="score-item">
                                <div class="score-name">News Integrity</div>
                                <div class="score-value">{analysis.get('news_integrity', 0.0):.2f}</div>
                                <div class="score-description">Overall integrity and trustworthiness</div>
                            </div>
                        </div>
                        <div class="col-md-3">
                            <div class="score-item">
                                <div class="score-name">Fact Check Needed</div>
                                <div class="score-value">{analysis.get('fact_check_needed_score', 1.0):.2f}</div>
                                <div class="score-description">Likelihood that claims need fact-checking</div>
                            </div>
                        </div>
                        <div class="col-md-3">
                            <div class="score-item">
                                <div class="score-name">Sentiment</div>
                                <div class="score-value">{analysis.get('sentiment_score', 0.5):.2f}</div>
                                <div class="score-description">Emotional tone (0.0 negative, 0.5 neutral, 1.0 positive)</div>
                            </div>
                        </div>
                        <div class="col-md-3">
                            <div class="score-item">
                                <div class="score-name">Bias</div>
                                <div class="score-value">{analysis.get('bias_score', 1.0):.2f}</div>
                                <div class="score-description">Degree of perceived bias</div>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="analysis-section">
                    <h2>Additional Information</h2>
                    <div class="detail-item">
                        <h4>Author Purpose</h4>
                        <p>{html.escape(analysis.get('author_purpose', 'Not specified'))}</p>
                    </div>
                    <div class="detail-item">
                        <h4>Short Summary</h4>
                        <p>{html.escape(analysis.get('short_summary', 'No summary available'))}</p>
                    </div>
                </div>
                """
            }
        }

        return jsonify(response_data)

    except Exception as e:
        logger.error(f"Unexpected error in analyze endpoint: {str(e)}", exc_info=True)
        return jsonify({
            'error': 'Internal server error',
            'status': 500,
            'details': str(e)
        }), 500

# Вспомогательные функции

def extract_text_from_url(url):
    """Extract text from URL with improved error handling"""
    try:
        logger.info(f"Processing URL: {url}")

        # Normalize URL
        parsed = urlparse(url)
        clean_url = urlunparse(parsed._replace(scheme=parsed.scheme.lower(), netloc=parsed.netloc.lower()))

        # Check for video content
        if any(domain in url for domain in ['youtube.com', 'vimeo.com']):
            logger.info("Video content detected")
            return "Video content detected", parsed.netloc.replace('www.', ''), "Video: " + url

        # Configure article with timeout and user agent
        article = Article(clean_url, config=config)

        # Download and parse article
        article.download()
        if article.download_state != 2:
            logger.error(f"Failed to download article from {url}")
            return None, None, None

        article.parse()
        if not article.text or len(article.text.strip()) < 100:
            logger.warning(f"Short or empty content from {url}")
            return None, None, None

        # Extract domain and title
        domain = parsed.netloc.replace('www.', '')
        title = article.title.strip() if article.title else "No title"

        logger.info(f"Successfully extracted content from {url}")
        return article.text.strip(), domain, title

    except Exception as e:
        logger.error(f"Error extracting article from {url}: {str(e)}")
        return None, None, None

def calculate_credibility(integrity, fact_check, sentiment, bias):
    """Calculate credibility level"""
    fact_check_score = 1.0 - fact_check
    sentiment_score = 1.0 - abs(sentiment - 0.5) * 2
    bias_score = 1.0 - bias

    score = (integrity * 0.45) + (fact_check_score * 0.35) + (sentiment_score * 0.10) + (bias_score * 0.10)

    if score >= 0.75:
        return 'High'
    if score >= 0.5:
        return 'Medium'
    return 'Low'

def save_analysis(url, title, source, content, analysis):
    """Save analysis to database"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()

            integrity = analysis.get('news_integrity', 0.0)
            fact_check = analysis.get('fact_check_needed_score', 1.0)
            sentiment = analysis.get('sentiment_score', 0.5)
            bias = analysis.get('bias_score', 1.0)
            summary = analysis.get('short_summary', 'No summary')
            credibility = analysis.get('index_of_credibility', 0.0)

            level = calculate_credibility(integrity, fact_check, sentiment, bias)
            db_url = url if url else f'text_{datetime.now(timezone.utc).timestamp()}'

            cursor.execute('''
                INSERT INTO news
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
                analysis_date=CURRENT_TIMESTAMP
            ''', (db_url, title, source, content, integrity, fact_check,
                  sentiment, bias, level, summary, credibility))

            # Update source stats
            cursor.execute('SELECT high, medium, low, total_analyzed FROM source_stats WHERE source = ?', (source,))
            row = cursor.fetchone()

            if row:
                high, medium, low, total = row
                if level == 'High': high += 1
                elif level == 'Medium': medium += 1
                else: low += 1
                total += 1
                cursor.execute('''
                    UPDATE source_stats SET high=?, medium=?, low=?, total_analyzed=?
                    WHERE source=?
                ''', (high, medium, low, total, source))
            else:
                counts = {'High': 1, 'Medium': 0, 'Low': 0}
                counts[level] = 1
                cursor.execute('''
                    INSERT INTO source_stats
                    (source, high, medium, low, total_analyzed)
                    VALUES (?, ?, ?, ?, ?)
                ''', (source, counts['High'], counts['Medium'], counts['Low'], 1))

            conn.commit()
            return level
    except Exception as e:
        logger.error(f"Error saving analysis: {str(e)}")
        raise ValueError("Failed to save analysis")

def generate_query(analysis_result):
    """Generate query for finding similar articles"""
    topics = analysis_result.get('topics', [])
    key_arguments = analysis_result.get('key_arguments', [])
    mentioned_facts = analysis_result.get('mentioned_facts', [])

    all_terms = []
    for phrase_list in [topics, key_arguments]:
        for phrase in phrase_list:
            if not phrase.strip():
                continue
            if ' ' in phrase.strip() and len(phrase.strip().split()) > 1:
                all_terms.append('"' + phrase.strip() + '"')
            else:
                all_terms.append(phrase.strip())

    for fact in mentioned_facts:
        if not fact.strip():
            continue
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
    """Make request to NewsAPI"""
    url = 'https://newsapi.org/v2/everything'
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        return response.json().get('articles', [])
    except Exception as e:
        logger.error(f'NewsAPI Request Error: {str(e)}')
        return []

def fetch_same_topic_articles(analysis_result, page=1, per_page=3):
    """Fetch similar articles by topic with pagination"""
    global predefined_trust_scores

    if not NEWS_API_KEY:
        logger.warning('NEWS_API_KEY is not configured. Skipping similar news search.')
        return []

    try:
        query = generate_query(analysis_result)
        end_date = datetime.now(timezone.utc).date()
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

    except Exception as e:
        logger.error(f"Error in fetch_same_topic_articles: {str(e)}")
        return []

def render_same_topic_articles_html(articles):
    """Render HTML for similar articles"""
    if not articles:
        return '<div class="alert alert-info">No similar articles found</div>'

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
            f'''
            <div class="similar-article">
                <h4><a href="{article_url}" target="_blank" rel="noopener noreferrer">{title}</a></h4>
                <div class="article-meta">
                    <span class="article-source"><i class="bi bi-newspaper"></i> {source_api_name}</span>
                    <span class="article-date"><i class="bi bi-calendar"></i> {published_at}</span>
                    <span class="article-credibility">Credibility: {int(trust_score*100)}%</span>
                </div>
                <p class="article-description">{description}</p>
            </div>
            '''
        )

    return '<div class="similar-articles-container">' + ''.join(html_items) + '</div>'

if __name__ == '__main__':
    # Initialize database
    initialize_database()

    # Run the application
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
