from flask import Flask, request, jsonify, render_template, make_response
from werkzeug.middleware.proxy_fix import ProxyFix
import logging
from logging.handlers import RotatingFileHandler
import os
import sqlite3
import re
import json
import requests
import html
import uuid
import time
from datetime import datetime, timezone
from urllib.parse import urlparse
import anthropic
from newspaper import Article, Config
from stop_words import get_stop_words
from flask_cors import CORS
from functools import wraps
import platform
import socket

# Initialize Flask application
app = Flask(__name__, static_folder='static', template_folder='templates')
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)
app.secret_key = os.getenv('SECRET_KEY', str(uuid.uuid4()))

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler('app.log', maxBytes=1000000, backupCount=3),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Security middleware
def security_middleware(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug(f"Security middleware called for path: {request.path}")
        if any(pattern in request.path for pattern in [
            '/wp-admin/', '/admin/', '/phpmyadmin/', '/.env',
            '/config.php', '/setup-config.php'
        ]):
            logger.warning(f"Blocked suspicious request to {request.path}")
            return abort(404)

        user_agent = request.headers.get('User-Agent', '').lower()
        if any(agent in user_agent for agent in [
            'sqlmap', 'nmap', 'nikto', 'wpscan', 'burpsuite', 'acunetix'
        ]):
            logger.warning(f"Blocked request with suspicious user agent: {user_agent}")
            return abort(403)

        return func(*args, **kwargs)
    return wrapper

@app.before_request
def before_request():
    logger.debug("Security middleware applied to request")
    return security_middleware(lambda: None)()

# Configure CORS for Railway
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "Accept"]
    }
})

# Environment variables
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY', 'mock-key')
MODEL_NAME = os.getenv('ANTHROPIC_MODEL', 'claude-3-opus-20240229')
NEWS_API_KEY = os.getenv('NEWS_API_KEY', 'mock-key')
NEWS_API_ENABLED = bool(NEWS_API_KEY and NEWS_API_KEY != 'mock-key')

# Global constants
predefined_trust_scores = {
    'bbc.com': 0.9,
    'reuters.com': 0.95,
    'apnews.com': 0.93,
    'nytimes.com': 0.88,
    'theguardian.com': 0.85,
    'wsj.com': 0.82,
    'cnn.com': 0.70,
    'foxnews.com': 0.40,
    'aljazeera.com': 0.80
}

TRUSTED_NEWS_SOURCES_IDS = [
    'bbc-news', 'reuters', 'associated-press',
    'the-new-york-times', 'the-guardian-uk'
]

stop_words_en = get_stop_words('en')
user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'

# Database configuration
DB_NAME = 'news_analysis.db'

def get_db_connection():
    """Create database connection"""
    logger.debug("Creating database connection")
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def initialize_database():
    """Initialize database schema and populate with test data if empty"""
    try:
        logger.info("Initializing database")
        db_dir = os.path.dirname(DB_NAME)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir)
            logger.debug(f"Created database directory: {db_dir}")

        with get_db_connection() as conn:
            cursor = conn.cursor()
            logger.debug("Database connection established")

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
                    url TEXT,
                    analysis_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    short_summary TEXT,
                    UNIQUE(url)
                )
            ''')
            logger.debug("Created news table")

            cursor.execute('CREATE INDEX IF NOT EXISTS idx_url ON news(url)')
            logger.debug("Created index on news.url")

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS source_stats (
                    source TEXT PRIMARY KEY,
                    high INTEGER DEFAULT 0,
                    medium INTEGER DEFAULT 0,
                    low INTEGER DEFAULT 0,
                    total_analyzed INTEGER DEFAULT 0
                )
            ''')
            logger.debug("Created source_stats table")

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
            logger.debug("Created feedback table")

            conn.commit()

            # Populate with test data if empty
            cursor.execute("SELECT COUNT(*) FROM source_stats")
            if cursor.fetchone()[0] == 0:
                populate_test_data(conn)
                logger.debug("Populated database with test data")

            logger.info("Database initialized successfully")
            return True
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}", exc_info=True)
        return False

def populate_test_data(conn):
    """Populate database with test data"""
    try:
        logger.debug("Populating database with test data")
        cursor = conn.cursor()
        test_sources = [
            ('bbc.com', 45, 10, 5),
            ('reuters.com', 50, 5, 2),
            ('foxnews.com', 15, 20, 30),
            ('cnn.com', 30, 25, 10),
            ('nytimes.com', 35, 15, 5)
        ]

        for source, high, medium, low in test_sources:
            total = high + medium + low
            cursor.execute('''
                INSERT INTO source_stats (source, high, medium, low, total_analyzed)
                VALUES (?, ?, ?, ?, ?)
            ''', (source, high, medium, low, total))
            logger.debug(f"Added test data for source: {source}")

        conn.commit()
        logger.info("Test data added to database successfully")
    except Exception as e:
        logger.error(f"Error populating test data: {str(e)}", exc_info=True)
        raise

def check_claude_connection():
    """Check connection to Claude API"""
    try:
        if not ANTHROPIC_API_KEY or ANTHROPIC_API_KEY == 'mock-key':
            logger.warning("Claude API key is not configured or is a mock key")
            return False

        logger.info("Testing Claude API connection")
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

        # Simple test request
        test_prompt = "Test connection"
        try:
            response = client.messages.create(
                model=MODEL_NAME,
                max_tokens=10,
                temperature=0.2,
                messages=[{"role": "user", "content": test_prompt}]
            )
            logger.info("Successfully connected to Claude API")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Claude API: {str(e)}", exc_info=True)
            return False

    except Exception as e:
        logger.error(f"Error checking Claude connection: {str(e)}", exc_info=True)
        return False

@app.after_request
def add_security_headers(response):
    """Add security headers with proper CSP configuration"""
    logger.debug("Adding security headers to response")
    response.headers['Content-Security-Policy'] = (
        "default-src 'self'; "
        "script-src 'self' https://cdn.jsdelivr.net https://cdn.plot.ly 'unsafe-inline' 'unsafe-eval'; "
        "style-src 'self' https://cdn.jsdelivr.net https://cdn.plot.ly https://fonts.googleapis.com 'unsafe-inline'; "
        "img-src 'self' data: https://cdn.jsdelivr.net; "
        "font-src 'self' https://cdn.jsdelivr.net https://fonts.gstatic.com; "
        "connect-src 'self' https://cdn.jsdelivr.net https://cdn.plot.ly; "
        "frame-src 'self'; "
        "object-src 'none'; "
        "base-uri 'self'; "
        "form-action 'self'; "
        "frame-ancestors 'none'"
    )
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, Accept'
    logger.debug("Security headers added to response")
    return response

@app.route('/')
def index():
    """Home page route"""
    logger.info("Rendering index page")
    return render_template('index.html')

@app.route('/health')
def health_check():
    """Health check endpoint"""
    logger.info("Processing health check request")
    claude_status = "connected" if check_claude_connection() else "disconnected"
    db_status = "connected" if initialize_database() else "disconnected"

    response = jsonify({
        'status': 'healthy',
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'database': db_status,
        'claude_api': claude_status,
        'api_keys': {
            'anthropic': 'configured' if ANTHROPIC_API_KEY and ANTHROPIC_API_KEY != 'mock-key' else 'not_configured',
            'news_api': 'configured' if NEWS_API_ENABLED else 'not_configured'
        }
    })
    logger.debug("Health check completed successfully")
    return response

@app.route('/faq')
def faq():
    """FAQ page route"""
    logger.info("Rendering FAQ page")
    return render_template('faq.html')

@app.route('/feedback', methods=['GET', 'POST'])
def feedback():
    """Feedback page and form handler"""
    if request.method == 'POST':
        logger.info("Processing feedback submission")
        try:
            name = request.form.get('name')
            email = request.form.get('email')
            feedback_type = request.form.get('type')
            message = request.form.get('message')

            if not all([name, email, feedback_type, message]):
                logger.warning("Feedback submission missing required fields")
                return render_template('feedback.html', error="All fields are required")

            if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
                logger.warning(f"Invalid email address: {email}")
                return render_template('feedback.html', error="Invalid email address")

            with get_db_connection() as conn:
                logger.debug("Saving feedback to database")
                conn.execute('''
                    INSERT INTO feedback (name, email, type, message, date)
                    VALUES (?, ?, ?, ?, ?)
                ''', (name, email, feedback_type, message, datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')))
                conn.commit()
                logger.info(f"Feedback saved successfully from {name} ({email})")

            logger.info("Redirecting to feedback success page")
            return render_template('feedback_success.html')

        except Exception as e:
            logger.error(f"Error saving feedback: {str(e)}", exc_info=True)
            return render_template('feedback.html', error="Error saving feedback")

    logger.info("Rendering feedback form")
    return render_template('feedback.html')

def extract_text_from_url(url):
    """Extract text from URL with improved error handling"""
    logger.info(f"Attempting to extract content from URL: {url}")

    try:
        # Validate URL format
        parsed = urlparse(url)
        if not all([parsed.scheme, parsed.netloc]):
            logger.error(f"Invalid URL format: {url}")
            return None, None, None

        # Normalize URL
        clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"

        logger.debug(f"Processing URL: {clean_url}")

        # Special handling for different domains
        if 'reuters.com' in clean_url:
            logger.debug("Using Reuters-specific extraction method")
            return extract_reuters_article(clean_url)
        elif 'bbc.com' in clean_url:
            logger.debug("Using BBC-specific extraction method")
            return extract_bbc_article(clean_url)
        elif any(domain in clean_url for domain in ['nytimes.com', 'theguardian.com']):
            logger.debug("Using NYTimes/Guardian-specific extraction method")
            return extract_standard_article(clean_url, True)
        else:
            logger.debug("Using standard extraction method")
            return extract_standard_article(clean_url)

    except Exception as e:
        logger.error(f"Unexpected error in extract_text_from_url: {str(e)}", exc_info=True)
        return None, None, None

def extract_reuters_article(url):
    """Special extraction method for Reuters articles"""
    try:
        logger.debug(f"Attempting to extract Reuters article: {url}")

        headers = {
            'User-Agent': user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Referer': 'https://www.google.com/',
            'Connection': 'keep-alive'
        }

        logger.debug("Making request to Reuters")
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        logger.debug("Parsing Reuters page")
        soup = BeautifulSoup(response.text, 'html.parser')

        # Try to find article content
        article_body = soup.find('div', {'class': 'ArticleBodyWrapper'})
        if article_body:
            title = soup.find('h1')
            title = title.get_text(strip=True) if title else "No title"

            content = article_body.get_text(separator=' ', strip=True)
            if len(content) > 100:
                domain = urlparse(url).netloc.replace('www.', '')
                logger.info(f"Successfully extracted Reuters article: {title}")
                return content, domain, title

        logger.debug("Trying alternative extraction methods for Reuters")
        return alternative_reuters_extraction(url)

    except Exception as e:
        logger.error(f"Error extracting Reuters article: {str(e)}", exc_info=True)
        return None, None, None

def alternative_reuters_extraction(url):
    """Alternative extraction methods for Reuters"""
    try:
        logger.debug("Trying newspaper3k as fallback for Reuters")
        config = Config()
        config.browser_user_agent = user_agent
        config.request_timeout = 30
        config.memoize_articles = False
        config.fetch_images = False
        config.follow_meta_refresh = True
        config.keep_article_html = True

        article = Article(url, config=config)
        article.download()
        article.parse()

        if article.text and len(article.text.strip()) > 100:
            domain = urlparse(url).netloc.replace('www.', '')
            title = article.title.strip() if article.title else "No title"
            logger.info(f"Successfully extracted Reuters article via newspaper3k: {title}")
            return article.text.strip(), domain, title

        logger.warning("All extraction methods failed for Reuters article")
        return None, None, None

    except Exception as e:
        logger.error(f"Alternative Reuters extraction failed: {str(e)}", exc_info=True)
        return None, None, None

def extract_bbc_article(url):
    """Special extraction method for BBC articles"""
    try:
        logger.debug(f"Attempting to extract BBC article: {url}")

        headers = {
            'User-Agent': user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Referer': 'https://www.google.com/',
            'Connection': 'keep-alive'
        }

        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        # Try to find article content
        article_body = soup.find('div', {'class': 'ssrcss-1q0x1q5-VideoContainer'})
        if not article_body:
            article_body = soup.find('article')

        if article_body:
            title = soup.find('h1')
            title = title.get_text(strip=True) if title else "No title"

            content = article_body.get_text(separator=' ', strip=True)
            if len(content) > 100:
                domain = urlparse(url).netloc.replace('www.', '')
                logger.info(f"Successfully extracted BBC article: {title}")
                return content, domain, title

        logger.warning("BBC article extraction failed")
        return None, None, None

    except Exception as e:
        logger.error(f"Error extracting BBC article: {str(e)}", exc_info=True)
        return None, None, None

def extract_standard_article(url, is_premium=False):
    """Standard article extraction for non-Reuters sites"""
    try:
        logger.debug(f"Attempting to extract standard article: {url}")

        headers = {
            'User-Agent': user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive'
        }

        if is_premium:
            headers['Referer'] = 'https://www.google.com/'

        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        config = Config()
        config.browser_user_agent = user_agent
        config.request_timeout = 30

        article = Article(url, config=config)
        article.download()
        article.parse()

        if article.text and len(article.text.strip()) > 100:
            domain = urlparse(url).netloc.replace('www.', '')
            title = article.title.strip() if article.title else "No title"
            logger.info(f"Successfully extracted standard article: {title}")
            return article.text.strip(), domain, title

        logger.warning("Standard article extraction failed")
        return None, None, None

    except Exception as e:
        logger.error(f"Error in standard article extraction: {str(e)}", exc_info=True)
        return None, None, None

def analyze_with_claude(content, source):
    """Analyze article text using Claude API with fallback to mock data"""
    logger.info("Starting article analysis")

    try:
        if not ANTHROPIC_API_KEY or ANTHROPIC_API_KEY == 'mock-key':
            logger.warning("Claude API key is not configured or is a mock key, using mock data")
            return generate_mock_analysis(source)

        logger.debug("Creating Claude client")
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

        max_chars = 10000
        if len(content) > max_chars:
            content = content[:max_chars]
            logger.warning(f"Article content truncated to {max_chars} characters")

        logger.debug("Preparing analysis prompt")
        prompt = f"""
        Analyze this news article for credibility, bias, and factual accuracy.
        Return results in JSON format with these fields:
        news_integrity (0.0-1.0), fact_check_needed_score (0.0-1.0),
        sentiment_score (0.0-1.0), bias_score (0.0-1.0),
        topics (list), key_arguments (list), mentioned_facts (list),
        author_purpose (string), potential_biases_identified (list),
        short_summary (string), index_of_credibility (0.0-1.0).

        Article: {content}
        Source: {source}
        """

        logger.debug("Sending request to Claude API")
        try:
            response = client.messages.create(
                model=MODEL_NAME,
                max_tokens=2000,
                temperature=0.2,
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = response.content[0].text
            logger.debug("Received response from Claude API")

            json_match = re.search(r'```json\s*(\{.*\})\s*```', response_text, re.DOTALL)
            if json_match:
                logger.debug("Found JSON in response")
                analysis = json.loads(json_match.group(1))
                logger.info("Successfully parsed Claude API response")
                return analysis
            else:
                logger.debug("Trying to parse response as direct JSON")
                try:
                    analysis = json.loads(response_text)
                    logger.info("Successfully parsed direct JSON response")
                    return analysis
                except json.JSONDecodeError:
                    logger.error("Failed to parse API response as JSON")
                    return generate_mock_analysis(source)

        except Exception as e:
            logger.error(f"Error calling Claude API: {str(e)}", exc_info=True)
            return generate_mock_analysis(source)

    except Exception as e:
        logger.error(f"Unexpected error in analyze_with_claude: {str(e)}", exc_info=True)
        return generate_mock_analysis(source)

def generate_mock_analysis(source):
    """Generate mock analysis data"""
    logger.debug("Generating mock analysis data")

    credibility_scores = {
        'bbc.com': 0.9,
        'reuters.com': 0.95,
        'apnews.com': 0.9,
        'nytimes.com': 0.85,
        'theguardian.com': 0.8
    }

    base_score = credibility_scores.get(source, 0.7)
    mock_analysis = {
        "news_integrity": base_score * 0.9 + 0.1,
        "fact_check_needed_score": 1.0 - base_score * 0.8,
        "sentiment_score": 0.5 + (0.1 if "positive" in source else -0.1),
        "bias_score": 1.0 - base_score * 0.9,
        "topics": ["news", "analysis", "sample", "demo"],
        "key_arguments": [
            "This is a sample analysis",
            "Showing how the system works",
            "Demonstrating functionality"
        ],
        "mentioned_facts": ["Sample fact 1", "Sample fact 2"],
        "author_purpose": "To demonstrate the system functionality",
        "potential_biases_identified": ["Sample bias detection"],
        "short_summary": "This is a sample analysis demonstrating how the system would analyze a real article.",
        "index_of_credibility": base_score * 0.9
    }

    logger.debug("Generated mock analysis data successfully")
    return mock_analysis

def calculate_credibility(integrity, fact_check, sentiment, bias):
    """Calculate credibility level"""
    logger.debug("Calculating credibility level")

    fact_check_score = 1.0 - fact_check
    sentiment_score = 1.0 - abs(sentiment - 0.5) * 2
    bias_score = 1.0 - bias
    score = (integrity * 0.45) + (fact_check_score * 0.35) + (sentiment_score * 0.10) + (bias_score * 0.10)

    if score >= 0.75:
        level = 'High'
    elif score >= 0.5:
        level = 'Medium'
    else:
        level = 'Low'

    logger.debug(f"Calculated credibility score: {score:.2f}, level: {level}")
    return level

def save_analysis(url, title, source, content, analysis):
    """Save analysis to database with improved error handling"""
    logger.debug("Saving analysis to database")

    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()

            integrity = analysis.get('news_integrity', 0.5)
            fact_check = analysis.get('fact_check_needed_score', 0.5)
            sentiment = analysis.get('sentiment_score', 0.5)
            bias = analysis.get('bias_score', 0.5)
            summary = analysis.get('short_summary', 'No summary available')
            credibility = analysis.get('index_of_credibility', 0.5)
            level = calculate_credibility(integrity, fact_check, sentiment, bias)

            db_url = url if url and url.startswith(('http://', 'https://')) else f'text_{datetime.now(timezone.utc).timestamp()}'

            logger.debug(f"Attempting to insert analysis for URL: {db_url}")
            try:
                cursor.execute('''
                    INSERT INTO news
                    (url, title, source, content, integrity, fact_check, sentiment, bias,
                    credibility_level, short_summary, index_of_credibility, analysis_date)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (db_url, title, source, content, integrity, fact_check,
                      sentiment, bias, level, summary, credibility,
                      datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')))

                conn.commit()
                logger.info(f"Successfully saved analysis for {title} from {source}")
                return level
            except sqlite3.IntegrityError as e:
                if "UNIQUE constraint failed: news.url" in str(e):
                    logger.warning(f"URL already exists in database: {db_url}")
                    cursor.execute('''
                        UPDATE news
                        SET title = ?,
                            source = ?,
                            content = ?,
                            integrity = ?,
                            fact_check = ?,
                            sentiment = ?,
                            bias = ?,
                            credibility_level = ?,
                            short_summary = ?,
                            index_of_credibility = ?,
                            analysis_date = ?
                        WHERE url = ?
                    ''', (title, source, content, integrity, fact_check,
                          sentiment, bias, level, summary, credibility,
                          datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'), db_url))
                    conn.commit()
                    logger.info(f"Updated existing analysis for URL: {db_url}")
                    return level
                else:
                    raise e
            except Exception as e:
                logger.error(f"Error saving analysis: {str(e)}", exc_info=True)
                raise

    except Exception as e:
        logger.error(f"Unexpected error in save_analysis: {str(e)}", exc_info=True)
        return 'Medium'

def generate_query_from_analysis(analysis_result):
    """Generate search query from analysis results"""
    logger.debug("Generating search query from analysis results")

    topics = analysis_result.get('topics', [])
    key_arguments = analysis_result.get('key_arguments', [])

    terms = []
    for phrase in topics + key_arguments:
        if not phrase or not isinstance(phrase, str):
            continue
        if ' ' in phrase and len(phrase.split()) > 1:
            terms.append(f'"{phrase}"')
        else:
            terms.append(phrase)

    unique_terms = list(set(terms))
    filtered_terms = [t for t in unique_terms if t.lower() not in stop_words_en]

    if len(filtered_terms) > 3:
        query = ' AND '.join(filtered_terms[:5])
    elif filtered_terms:
        query = ' OR '.join(filtered_terms)
    else:
        query = 'news OR current events'

    logger.debug(f"Generated search query: {query}")
    return query

def fetch_same_topic_articles(analysis_result, page=1, per_page=3):
    """Fetch similar articles using News API with fallback to mock data"""
    logger.debug("Fetching similar articles")

    try:
        if not NEWS_API_ENABLED:
            logger.info("News API not enabled, returning mock data")
            return generate_mock_articles(per_page)

        query = generate_query_from_analysis(analysis_result)
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
            'to': end_date.strftime('%Y-%m-%d')
        }

        if TRUSTED_NEWS_SOURCES_IDS:
            params['sources'] = ','.join(TRUSTED_NEWS_SOURCES_IDS)

        try:
            logger.debug(f"Making request to News API with query: {query}")
            response = requests.get(
                'https://newsapi.org/v2/everything',
                params=params,
                timeout=15
            )
            response.raise_for_status()
            articles = response.json().get('articles', [])

            if not articles and query != 'news OR current events':
                broader_query = ' OR '.join([f'"{term}"' if ' ' in term else term
                                          for term in analysis_result.get('topics', [])[:3]
                                          if term and term not in stop_words_en])
                if not broader_query:
                    broader_query = 'news OR current events'

                params['q'] = broader_query
                response = requests.get(
                    'https://newsapi.org/v2/everything',
                    params=params,
                    timeout=15
                )
                response.raise_for_status()
                articles.extend(response.json().get('articles', []))

            # Filter and rank articles
            unique_articles = {}
            for article in articles:
                if article.get('url'):
                    unique_articles[article['url']] = article

            articles = list(unique_articles.values())
            all_query_terms = query.split()
            ranked_articles = []

            for article in articles:
                source_domain = urlparse(article.get('url', '')).netloc.replace('www.', '')
                trust_score = predefined_trust_scores.get(source_domain, 0.5)

                article_text = (article.get('title', '') + ' ' + article.get('description', '')).lower()
                relevance_score = sum(1 for term in all_query_terms if term.lower() in article_text)
                final_score = (relevance_score * 10) + (trust_score * 5)
                ranked_articles.append((article, final_score))

            ranked_articles.sort(key=lambda x: x[1], reverse=True)
            logger.debug(f"Found {len(ranked_articles)} similar articles")
            return [article for article, score in ranked_articles[:per_page]]

        except requests.RequestException as e:
            logger.error(f"News API request failed: {str(e)}", exc_info=True)
            return generate_mock_articles(per_page)

    except Exception as e:
        logger.error(f"Error fetching similar articles: {str(e)}", exc_info=True)
        return generate_mock_articles(per_page)

def generate_mock_articles(count):
    """Generate mock articles for when API is not available"""
    logger.debug(f"Generating {count} mock articles")

    mock_articles = [
        {
            'title': f"Sample Article {i+1} on Similar Topic",
            'url': f"https://example.com/article{i+1}",
            'source': {'name': f"Example News {i+1}"},
            'publishedAt': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
            'description': f"This is a sample article about a similar topic to demonstrate functionality.",
            'trust_score': 0.7 + (i * 0.05)
        }
        for i in range(count)
    ]

    logger.debug("Generated mock articles successfully")
    return mock_articles

def render_same_topic_articles_html(articles):
    """Render HTML for similar articles with show more button"""
    logger.debug("Rendering similar articles HTML")

    if not articles:
        logger.debug("No articles to render")
        return '<div class="alert alert-info">No similar articles found.</div>'

    html_items = []
    for art in articles:
        title = html.escape(art.get('title', 'No Title'))
        article_url = html.escape(art.get('url', '#'))
        source_name = html.escape(art.get('source', {}).get('name', 'Unknown Source'))
        published_at = html.escape(art.get('publishedAt', 'N/A').split('T')[0])
        description = html.escape(art.get('description', 'No description available.'))
        trust_score = art.get('trust_score', 0.7)
        trust_percent = int(trust_score * 100)

        trust_color = "success" if trust_score > 0.7 else "warning" if trust_score > 0.4 else "danger"

        try:
            date_obj = datetime.strptime(published_at, '%Y-%m-%d')
            formatted_date = date_obj.strftime('%B %d, %Y')
        except:
            formatted_date = published_at

        html_items.append(f'''
        <div class="similar-article mb-4">
            <div class="d-flex flex-column flex-md-row gap-3">
                <div class="flex-shrink-0">
                    <div class="trust-indicator text-center p-2 bg-light rounded">
                        <div class="h5 mb-1">{trust_percent}%</div>
                        <div class="small text-muted">Trust</div>
                        <div class="progress mt-2" style="height: 5px;">
                            <div class="progress-bar bg-{trust_color}"
                                 role="progressbar"
                                 style="width: {trust_percent}%"
                                 aria-valuenow="{trust_percent}"
                                 aria-valuemin="0"
                                 aria-valuemax="100">
                            </div>
                        </div>
                    </div>
                </div>
                <div class="flex-grow-1">
                    <h4 class="article-title mb-2">
                        <a href="{article_url}" target="_blank" class="text-decoration-none">{title}</a>
                    </h4>
                    <div class="article-meta mb-2">
                        <span class="article-source fw-bold me-2">{source_name}</span>
                        <span class="article-date text-muted me-2"><i class="bi bi-calendar me-1"></i>{formatted_date}</span>
                    </div>
                    <p class="article-description mb-0">{description}</p>
                </div>
            </div>
        </div>
        ''')

    logger.debug(f"Rendered HTML for {len(html_items)} articles")
    return ''.join(html_items)

def format_analysis_results(title, source, analysis, credibility):
    """Format analysis results for display with all sections"""
    logger.debug("Formatting analysis results")

    try:
        integrity = analysis.get('news_integrity', 0.5)
        fact_check = analysis.get('fact_check_needed_score', 0.5)
        sentiment = analysis.get('sentiment_score', 0.5)
        bias = analysis.get('bias_score', 0.5)
        summary = html.escape(analysis.get('short_summary', 'No summary available'))
        topics = [html.escape(str(t)) for t in analysis.get('topics', [])]
        key_args = [html.escape(str(a)) for a in analysis.get('key_arguments', [])]
        biases = [html.escape(str(b)) for b in analysis.get('potential_biases_identified', [])]

        # Create topic badges
        topic_badges = ' '.join(
            f'<span class="badge bg-primary me-1 mb-1">{topic}</span>'
            for topic in topics
        )

        # Create key arguments list
        key_args_list = ''.join(
            f'<li class="mb-2"><i class="bi bi-check-circle-fill text-primary me-2"></i>{arg}</li>'
            for arg in key_args
        )

        # Create biases list
        biases_list = ''.join(
            f'<li class="mb-2"><i class="bi bi-exclamation-triangle-fill text-warning me-2"></i>{bias}</li>'
            for bias in biases
        )

        # Create score indicators
        integrity_indicator = create_score_indicator(integrity, "News Integrity")
        fact_check_indicator = create_score_indicator(1 - fact_check, "Fact Check Score")
        sentiment_indicator = create_score_indicator(sentiment, "Sentiment")
        bias_indicator = create_score_indicator(1 - bias, "Bias Score")

        # Create credibility gauge
        credibility_gauge = create_credibility_gauge(analysis.get('index_of_credibility', 0.5))

        logger.debug("Successfully formatted analysis results")
        return {
            'title': title,
            'source': source,
            'credibility': credibility,
            'output_md': f"""
            <div class="analysis-results">
                <div class="article-header mb-4 p-3 bg-white rounded shadow-sm">
                    <h2 class="mb-3">{html.escape(title)}</h2>
                    <div class="d-flex flex-wrap justify-content-between align-items-center">
                        <div>
                            <span class="badge bg-secondary me-2">{html.escape(source)}</span>
                            <span class="credibility-badge {credibility.lower()}">{credibility}</span>
                        </div>
                        <div class="credibility-score-display">
                            {credibility_gauge}
                        </div>
                    </div>
                </div>

                <div class="scores-section mb-4">
                    <h3 class="section-title mb-3">Credibility Scores</h3>
                    <div class="row g-3">
                        <div class="col-md-6 col-lg-3">
                            {integrity_indicator}
                        </div>
                        <div class="col-md-6 col-lg-3">
                            {fact_check_indicator}
                        </div>
                        <div class="col-md-6 col-lg-3">
                            {sentiment_indicator}
                        </div>
                        <div class="col-md-6 col-lg-3">
                            {bias_indicator}
                        </div>
                    </div>
                </div>

                <div class="summary-section mb-4 p-3 bg-white rounded shadow-sm">
                    <h3 class="section-title mb-3">Article Summary</h3>
                    <p class="lead">{summary}</p>
                </div>

                <div class="topics-section mb-4 p-3 bg-white rounded shadow-sm">
                    <h3 class="section-title mb-3">Key Topics</h3>
                    <div class="d-flex flex-wrap gap-2">
                        {topic_badges}
                    </div>
                </div>

                <div class="row mb-4">
                    <div class="col-md-6">
                        <div class="arguments-section p-3 bg-white rounded shadow-sm h-100">
                            <h3 class="section-title mb-3">Key Arguments</h3>
                            <ul class="list-unstyled mb-0">
                                {key_args_list}
                            </ul>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="biases-section p-3 bg-white rounded shadow-sm h-100">
                            <h3 class="section-title mb-3">Potential Biases</h3>
                            <ul class="list-unstyled mb-0">
                                {biases_list if biases_list else '<li class="text-muted">No significant biases detected</li>'}
                            </ul>
                        </div>
                    </div>
                </div>
            </div>
            """
        }
    except Exception as e:
        logger.error(f"Error formatting analysis results: {str(e)}", exc_info=True)
        return {
            "error": "Error formatting analysis results",
            "output_md": '<div class="alert alert-danger">Error displaying analysis results</div>'
        }

def create_score_indicator(score: float, label: str) -> str:
    """Create a visual score indicator"""
    score_percent = int(score * 100)
    color_class = "success" if score > 0.7 else "warning" if score > 0.4 else "danger"

    return f"""
    <div class="score-card h-100 p-3 bg-light rounded">
        <div class="d-flex justify-content-between align-items-center mb-2">
            <span class="score-label text-muted">{html.escape(label)}</span>
            <span class="score-value fw-bold h4 mb-0">{score:.2f}</span>
        </div>
        <div class="progress" style="height: 8px;">
            <div class="progress-bar bg-{color_class}"
                 role="progressbar"
                 style="width: {score_percent}%"
                 aria-valuenow="{score_percent}"
                 aria-valuemin="0"
                 aria-valuemax="100">
            </div>
        </div>
        <div class="score-description mt-2 text-muted small">
            {get_score_description(score, label)}
        </div>
    </div>
    """

def get_score_description(score: float, label: str) -> str:
    """Get description for a score based on its value and label"""
    if label == "News Integrity":
        if score > 0.8:
            return "High integrity with reliable sourcing"
        elif score > 0.6:
            return "Generally reliable with minor issues"
        elif score > 0.4:
            return "Some concerns about sourcing"
        else:
            return "Significant integrity concerns"
    elif label == "Fact Check Score":
        if score > 0.8:
            return "High factual accuracy"
        elif score > 0.6:
            return "Mostly accurate with minor issues"
        elif score > 0.4:
            return "Some factual concerns"
        else:
            return "Significant factual inaccuracies likely"
    elif label == "Sentiment":
        if score > 0.6:
            return "Positive tone"
        elif score > 0.4:
            return "Neutral tone"
        else:
            return "Negative tone"
    elif label == "Bias Score":
        if score > 0.8:
            return "Minimal bias detected"
        elif score > 0.6:
            return "Some bias present"
        elif score > 0.4:
            return "Noticeable bias"
        else:
            return "Strong bias detected"
    return "Score description"

def create_credibility_gauge(score: float) -> str:
    """Create a radial credibility gauge"""
    score_percent = int(score * 100)
    color = "#28a745" if score > 0.7 else "#ffc107" if score > 0.4 else "#dc3545"

    return f"""
    <div class="credibility-gauge" style="width: 100px; height: 100px; position: relative;">
        <svg viewBox="0 0 100 100" style="transform: rotate(-90deg); position: absolute; top: 0; left: 0;">
            <circle cx="50" cy="50" r="45" fill="none" stroke="#eee" stroke-width="10" />
            <circle cx="50" cy="50" r="45" fill="none" stroke="{color}" stroke-width="10"
                    stroke-dasharray="{3.14 * 45 * 2 * score} {3.14 * 45 * 2 * (1 - score)}"
                    stroke-linecap="round" />
        </svg>
        <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); text-align: center;">
            <div style="font-size: 1.2rem; font-weight: bold; color: {color};">{score_percent}%</div>
            <div style="font-size: 0.7rem; color: #6c757d;">Credibility</div>
        </div>
    </div>
    """

def get_source_credibility_data():
    """Get source credibility data for chart"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT source, AVG(index_of_credibility) as avg_credibility FROM news GROUP BY source")
            results = cursor.fetchall()

            sources = []
            scores = []

            # Add predefined sources first
            for source, score in predefined_trust_scores.items():
                sources.append(source)
                scores.append(score)

            # Add analyzed sources
            for row in results:
                source = row['source']
                score = row['avg_credibility']
                if source not in sources:
                    sources.append(source)
                    scores.append(score)

            return {
                'sources': sources,
                'credibility_scores': scores
            }
    except Exception as e:
        logger.error(f"Error getting source credibility data: {str(e)}", exc_info=True)
        # Return mock data if there's an error
        return {
            'sources': list(predefined_trust_scores.keys()),
            'credibility_scores': list(predefined_trust_scores.values())
        }

@app.route('/analyze', methods=['POST', 'OPTIONS'])
def analyze():
    """Analyze article endpoint with comprehensive error handling"""
    logger.info(f"Received analyze request. Method: {request.method}, Path: {request.path}")

    # Handle OPTIONS request for CORS preflight
    if request.method == 'OPTIONS':
        logger.debug("Processing OPTIONS request for CORS preflight")
        response = make_response()
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Accept'
        response.headers['Access-Control-Max-Age'] = '3600'
        logger.debug("Successfully processed OPTIONS request")
        return response

    # Validate request content type
    if not request.is_json:
        logger.warning(f"Invalid content type: {request.content_type}")
        return jsonify({
            'error': 'Invalid content type',
            'status': 400,
            'details': 'Content-Type header must be application/json',
            'request_id': str(uuid.uuid4())
        }), 400

    try:
        logger.debug("Processing JSON request data")

        # Get and validate data
        data = request.get_json()
        if not data:
            logger.warning("Empty request body received")
            return jsonify({
                'error': 'Empty request body',
                'status': 400,
                'request_id': str(uuid.uuid4())
            }), 400

        logger.debug(f"Request data: {json.dumps(data, indent=2)}")

        if 'input_text' not in data:
            logger.warning("Missing input_text in request")
            return jsonify({
                'error': 'Missing input text',
                'status': 400,
                'details': 'input_text field is required',
                'request_id': str(uuid.uuid4())
            }), 400

        input_text = data['input_text'].strip()
        source_name = data.get('source_name_manual', 'Direct Input').strip()

        if not input_text:
            logger.warning("Empty input text received")
            return jsonify({
                'error': 'Empty input text',
                'status': 400,
                'details': 'Input text cannot be empty',
                'request_id': str(uuid.uuid4())
            }), 400

        logger.info(f"Processing input of length: {len(input_text)} characters")

        # Process article - either URL or direct text
        if input_text.startswith(('http://', 'https://')):
            logger.info(f"Processing URL input: {input_text[:100]}...")
            content, source, title = extract_text_from_url(input_text)
            if not content:
                if 'reuters.com' in input_text.lower():
                    logger.warning("Failed to extract Reuters article")
                    return jsonify({
                        'error': 'Could not extract article content from Reuters',
                        'status': 400,
                        'details': 'Reuters actively blocks automated requests. Please try:',
                        'suggestions': [
                            'Open the URL in a browser and copy the text manually',
                            'Use a different news source',
                            'Try accessing the article through Google Cache by prefixing the URL with "cache:"',
                            'Try using the Wayback Machine to access an archived version'
                        ],
                        'request_id': str(uuid.uuid4())
                    }), 400

                logger.warning("Failed to extract article content")
                return jsonify({
                    'error': 'Could not extract article content',
                    'status': 400,
                    'details': 'Failed to download or parse the article from the provided URL',
                    'suggestions': [
                        'Check if the URL is correct and accessible',
                        'Try a different URL',
                        'Make sure the website allows scraping',
                        'Alternatively, paste the article text directly'
                    ],
                    'request_id': str(uuid.uuid4())
                }), 400
        else:
            logger.info("Processing direct text input")
            if len(input_text) < 100:
                logger.warning(f"Input text too short: {len(input_text)} characters")
                return jsonify({
                    'error': 'Content too short',
                    'status': 400,
                    'details': f'Minimum 100 characters required, got {len(input_text)}',
                    'request_id': str(uuid.uuid4())
                }), 400
            content = input_text
            title = 'User-provided Text'
            source = source_name

        logger.info(f"Successfully extracted content. Length: {len(content)} characters")

        # Analyze content
        logger.info("Starting article analysis")
        analysis_start_time = time.time()
        analysis = analyze_with_claude(content, source)
        analysis_duration = time.time() - analysis_start_time
        logger.info(f"Completed article analysis in {analysis_duration:.2f} seconds")

        # Save analysis to database
        logger.info("Saving analysis to database")
        save_start_time = time.time()
        credibility = save_analysis(
            input_text if input_text.startswith(('http://', 'https://')) else None,
            title, source, content, analysis
        )
        save_duration = time.time() - save_start_time
        logger.info(f"Saved analysis to database in {save_duration:.2f} seconds")

        # Get similar articles
        logger.info("Fetching similar articles")
        similar_articles = fetch_same_topic_articles(analysis)
        similar_articles_html = render_same_topic_articles_html(similar_articles)
        logger.info(f"Found {len(similar_articles)} similar articles")

        # Get source credibility data
        source_credibility_data = get_source_credibility_data()

        # Format response
        logger.info("Preparing response data")
        response_data = {
            'status': 'success',
            'analysis': analysis,
            'credibility': credibility,
            'title': title,
            'source': source,
            'output': format_analysis_results(title, source, analysis, credibility),
            'similar_articles': similar_articles_html,
            'source_credibility_data': source_credibility_data,
            'scores_for_chart': {
                'news_integrity': analysis.get('news_integrity', 0.5),
                'fact_check_needed_score': analysis.get('fact_check_needed_score', 0.5),
                'sentiment_score': analysis.get('sentiment_score', 0.5),
                'bias_score': analysis.get('bias_score', 0.5),
                'index_of_credibility': analysis.get('index_of_credibility', 0.5)
            },
            'request_id': str(uuid.uuid4()),
            'processing_time': f"{analysis_duration:.2f} seconds"
        }

        logger.info("Successfully processed analysis request")
        return jsonify(response_data)

    except Exception as e:
        request_id = str(uuid.uuid4())
        logger.error(f"Unexpected error in analyze endpoint: {str(e)}. Request ID: {request_id}", exc_info=True)
        return jsonify({
            'error': 'Internal server error',
            'status': 500,
            'details': str(e),
            'suggestions': [
                'Please try again later',
                'Check your internet connection',
                'If the problem persists, contact support with this request ID'
            ],
            'request_id': request_id
        }), 500

if __name__ == '__main__':
    # Initialize database
    if not initialize_database():
        logger.error("Failed to initialize database. Exiting...")
        exit(1)

    # Check Claude connection
    if not check_claude_connection():
        logger.warning("Claude API connection test failed. Application will use mock data.")

    # Start the application
    logger.info("Starting Flask application")
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
