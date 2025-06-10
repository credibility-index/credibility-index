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

# Initialize Flask application
app = Flask(__name__, static_folder='static', template_folder='templates')
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)
app.secret_key = os.getenv('SECRET_KEY', 'your-secret-key-here')

# Environment variables
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
MODEL_NAME = os.getenv('ANTHROPIC_MODEL', 'claude-3-opus-20240229')
NEWS_API_KEY = os.getenv('NEWS_API_KEY')

# Email configuration
app.config.update(
    MAIL_SERVER=os.getenv('MAIL_SERVER', 'smtp.gmail.com'),
    MAIL_PORT=int(os.getenv('MAIL_PORT', 587)),
    MAIL_USE_TLS=os.getenv('MAIL_USE_TLS', 'true').lower() == 'true',
    MAIL_USERNAME=os.getenv('MAIL_USERNAME'),
    MAIL_PASSWORD=os.getenv('MAIL_PASSWORD'),
    MAIL_DEFAULT_SENDER=os.getenv('MAIL_DEFAULT_SENDER')
)

# Timezone
UTC = timezone.utc

# Logging setup
def setup_logging():
    """Configure logging system"""
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # File logging with rotation
    file_handler = RotatingFileHandler('app.log', maxBytes=1024*1024, backupCount=5)
    file_handler.setFormatter(formatter)

    # Console logging
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    app.logger.addHandler(file_handler)
    app.logger.addHandler(console_handler)
    app.logger.setLevel(logging.INFO)

# Database initialization
DB_NAME = 'news_analysis.db'

def get_db_connection():
    """Create database connection"""
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def initialize_database():
    """Ensure database schema exists"""
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
        app.logger.info('Database schema initialized successfully')

# Initial data
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

# WordPress scanner protection
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
    """Block WordPress scanner requests"""
    path = request.path.lower()
    if any(pattern.search(path) for pattern in WORDPRESS_PATHS):
        app.logger.warning(f'Blocked WordPress scanner request from {request.remote_addr}')
        return abort(404)

    if any(param in request.query_string.decode('utf-8', 'ignore') for param in ['=http://', '=https://', '=ftp://']):
        app.logger.warning(f'Blocked suspicious query parameter from {request.remote_addr}')
        return abort(404)

@app.after_request
def add_security_headers(response):
    """Add security and CORS headers"""
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    response.headers['Access-Control-Allow-Credentials'] = 'true'
    return response

# Error handlers
@app.errorhandler(404)
def page_not_found(e):
    """404 error handler"""
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(400)
def bad_request(e):
    """400 error handler"""
    return jsonify({'error': 'Bad request'}), 400

@app.errorhandler(500)
def internal_server_error(e):
    """500 error handler"""
    return jsonify({'error': 'Internal server error'}), 500

# Claude API class
class ClaudeNewsAnalyzer:
    """Class for interacting with Anthropic Claude API"""
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        self.model_name = MODEL_NAME

    def analyze_article_text(self, article_text_content, source_name_for_context):
        """Analyze article text using Claude API"""
        try:
            max_chars = 10000
            if len(article_text_content) > max_chars:
                article_text_content = article_text_content[:max_chars]
                app.logger.warning(f"Article content truncated to {max_chars} characters")

            media_owner = media_owners.get(source_name_for_context, 'Unknown Owner')

            prompt = f"""Analyze this news article:

Article Text:
{article_text_content}

Source: {source_name_for_context}
Media Owner: {media_owner}

Return results in JSON format with these fields:
- news_integrity (0.0-1.0)
- fact_check_needed_score (0.0-1.0)
- sentiment_score (0.0-1.0)
- bias_score (0.0-1.0)
- topics (list of strings)
- key_arguments (list of strings)
- mentioned_facts (list of strings)
- author_purpose (string)
- potential_biases_identified (list of strings)
- short_summary (string)
- index_of_credibility (0.0-1.0)
- published_date (string)"""

            message = self.client.messages.create(
                model=self.model_name,
                max_tokens=2000,
                temperature=0.2,
                messages=[{'role': 'user', 'content': prompt}]
            )

            response_text = message.content[0].text.strip()
            json_match = re.search(r'```json\s*(\{.*\})\s*```', response_text, re.DOTALL)

            if json_match:
                return json.loads(json_match.group(1))
            return json.loads(response_text)

        except Exception as e:
            app.logger.error(f'Claude analysis error: {str(e)}')
            raise ValueError(f'AI analysis error: {str(e)}')

# Article processing functions
def extract_text_from_url(url):
    """Extract text from URL"""
    try:
        clean_url = re.sub(r'/amp(/)?$', '', url)
        if any(domain in url for domain in ['youtube.com', 'vimeo.com']):
            return "Video content detected", urlparse(clean_url).netloc.replace('www.', ''), "Video: " + url

        article = Article(clean_url)
        article.download()
        article.parse()

        if len(article.text) < 100:
            app.logger.warning(f"Short content from {url}")
            return None, None, None

        return article.text.strip(), urlparse(clean_url).netloc.replace('www.', ''), article.title.strip()
    except Exception as e:
        app.logger.error(f'Error extracting article from {url}: {str(e)}')
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
            c = conn.cursor()

            integrity = analysis.get('news_integrity', 0.0)
            fact_check = analysis.get('fact_check_needed_score', 1.0)
            sentiment = analysis.get('sentiment_score', 0.5)
            bias = analysis.get('bias_score', 1.0)
            summary = analysis.get('short_summary', 'No summary')
            credibility = analysis.get('index_of_credibility', 0.0)

            level = calculate_credibility(integrity, fact_check, sentiment, bias)
            db_url = url if url else f'text_{datetime.now(UTC).timestamp()}'

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
                (db_url, title, source, content, integrity, fact_check,
                sentiment, bias, level, summary, credibility))

            c.execute('SELECT high, medium, low, total_analyzed FROM source_stats WHERE source = ?', (source,))
            row = c.fetchone()

            if row:
                high, medium, low, total = row
                if level == 'High': high += 1
                elif level == 'Medium': medium += 1
                else: low += 1
                total += 1
                c.execute('''UPDATE source_stats SET high=?, medium=?, low=?, total_analyzed=?
                    WHERE source=?''', (high, medium, low, total, source))
            else:
                counts = {'High': 1, 'Medium': 0, 'Low': 0}
                counts[level] = 1
                c.execute('''INSERT INTO source_stats
                    (source, high, medium, low, total_analyzed)
                    VALUES (?, ?, ?, ?, ?)''',
                    (source, counts['High'], counts['Medium'], counts['Low'], 1))

            conn.commit()
            return level
    except Exception as e:
        app.logger.error(f'Database error: {str(e)}')
        raise

# API Endpoints
@app.route('/')
def home():
    """Home page"""
    return render_template('index.html')

@app.route('/analyze', methods=['POST', 'OPTIONS'])
def analyze():
    """Analyze article endpoint"""
    if request.method == 'OPTIONS':
        return jsonify({}), 200

    try:
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400

        data = request.get_json()
        if not data or 'input_text' not in data:
            return jsonify({'error': 'Missing input text'}), 400

        input_text = data['input_text']
        source_name = data.get('source_name_manual', 'Direct Input')

        if not input_text:
            return jsonify({'error': 'Empty input text'}), 400

        # Process article
        if input_text.startswith('http'):
            content, source, title = extract_text_from_url(input_text)
            if not content:
                return jsonify({'error': 'Could not extract article content'}), 400
        else:
            content = input_text
            title = 'User-provided Text'

        if len(content) < 100:
            return jsonify({'error': 'Content too short (min 100 chars)'}), 400

        # Analyze with Claude
        analyzer = ClaudeNewsAnalyzer()
        analysis = analyzer.analyze_article_text(content, source)

        # Save to database
        credibility = save_analysis(
            input_text if input_text.startswith('http') else None,
            title,
            source,
            content,
            analysis
        )

        # Prepare response
        response = {
            'analysis': analysis,
            'credibility': credibility,
            'title': title,
            'source': source
        }

        return jsonify(response), 200

    except Exception as e:
        app.logger.error(f'Analysis error: {str(e)}')
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/feedback', methods=['POST'])
def feedback():
    """Feedback endpoint"""
    try:
        data = request.get_json()
        if not data or not all(k in data for k in ['name', 'email', 'type', 'message']):
            return jsonify({'error': 'Missing required fields'}), 400

        if not re.match(r"[^@]+@[^@]+\.[^@]+", data['email']):
            return jsonify({'error': 'Invalid email'}), 400

        with get_db_connection() as conn:
            c = conn.cursor()
            c.execute('''INSERT INTO feedback
                (name, email, type, message, date)
                VALUES (?, ?, ?, ?, ?)''',
                (data['name'], data['email'], data['type'], data['message'],
                 datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')))
            conn.commit()

        return jsonify({'message': 'Feedback received'}), 200

    except Exception as e:
        app.logger.error(f'Feedback error: {str(e)}')
        return jsonify({'error': 'Failed to save feedback'}), 500

# Initialize application
def initialize_app():
    """Initialize the application"""
    setup_logging()
    initialize_database()
    app.logger.info("Application initialized successfully")

if __name__ == '__main__':
    initialize_app()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
