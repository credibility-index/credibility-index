from flask import Flask, request, jsonify, render_template, session, make_response, abort
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
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse, urlunparse
import anthropic
from newspaper import Article, Config
from stop_words import get_stop_words
from flask_cors import CORS
from functools import wraps
import ssl
from bs4 import BeautifulSoup

# Initialize Flask application
app = Flask(__name__, static_folder='static', template_folder='templates')
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)
app.secret_key = os.getenv('SECRET_KEY', str(uuid.uuid4()))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
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
        # Block common attack patterns
        if any(pattern in request.path for pattern in [
            '/wp-admin/',
            '/admin/',
            '/phpmyadmin/',
            '/.env',
            '/config.php',
            '/setup-config.php'
        ]):
            logger.warning(f"Blocked suspicious request to {request.path}")
            return abort(404)

        # Block suspicious user agents
        suspicious_agents = [
            'sqlmap',
            'nmap',
            'nikto',
            'wpscan',
            'burpsuite',
            'acunetix'
        ]

        user_agent = request.headers.get('User-Agent', '').lower()
        if any(agent in user_agent for agent in suspicious_agents):
            logger.warning(f"Blocked request with suspicious user agent: {user_agent}")
            return abort(403)

        return func(*args, **kwargs)
    return wrapper

# Apply security middleware to all routes
@app.before_request
def before_request():
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
    'bbc.com': 0.9, 'reuters.com': 0.95, 'apnews.com': 0.93,
    'nytimes.com': 0.88, 'theguardian.com': 0.85, 'wsj.com': 0.82,
    'cnn.com': 0.70, 'foxnews.com': 0.40, 'aljazeera.com': 0.80
}

TRUSTED_NEWS_SOURCES_IDS = [
    'bbc-news', 'reuters', 'associated-press',
    'the-new-york-times', 'the-guardian-uk'
]

stop_words_en = get_stop_words('en')
user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'

# Database configuration
DB_NAME = 'news_analysis.db'
@app.after_request
def add_security_headers(response):
    """Add security headers with proper CSP configuration"""
    response.headers['Content-Security-Policy'] = (
        "default-src 'self'; "
        "script-src 'self' https://cdn.jsdelivr.net https://cdn.plot.ly; "
        "style-src 'self' https://cdn.jsdelivr.net https://cdn.plot.ly 'unsafe-inline'; "
        "img-src 'self' data:; "
        "font-src 'self' https://cdn.jsdelivr.net; "
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
    return response

def get_db_connection():
    """Create database connection"""
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def initialize_database():
    """Initialize database schema and populate with test data if empty"""
    try:
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
                    url TEXT,
                    analysis_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    short_summary TEXT,
                    UNIQUE(url)
                )
            ''')

            # Create index on url for faster lookups
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_url ON news(url)')

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

            # Populate with test data if empty
            cursor.execute("SELECT COUNT(*) FROM source_stats")
            if cursor.fetchone()[0] == 0:
                populate_test_data(conn)

            logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        raise

def populate_test_data(conn):
    """Populate database with test data"""
    try:
        cursor = conn.cursor()

        # Test data for various sources
        test_sources = [
            ('bbc.com', 45, 10, 5), ('reuters.com', 50, 5, 2),
            ('foxnews.com', 15, 20, 30), ('cnn.com', 30, 25, 10),
            ('nytimes.com', 35, 15, 5)
        ]

        for source, high, medium, low in test_sources:
            total = high + medium + low
            cursor.execute('''
                INSERT INTO source_stats (source, high, medium, low, total_analyzed)
                VALUES (?, ?, ?, ?, ?)
            ''', (source, high, medium, low, total))

        conn.commit()
        logger.info("Test data added to database successfully")
    except Exception as e:
        logger.error(f"Error populating test data: {str(e)}")
        raise

@app.after_request
def add_security_headers(response):
    """Add security headers"""
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, Accept'
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Content-Security-Policy'] = "default-src 'self'"
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    return response

@app.route('/')
def index():
    """Home page route"""
    return render_template('index.html')

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'database': 'connected' if os.path.exists(DB_NAME) else 'disconnected',
        'api_keys': {
            'anthropic': 'configured' if ANTHROPIC_API_KEY and ANTHROPIC_API_KEY != 'mock-key' else 'not_configured',
            'news_api': 'configured' if NEWS_API_ENABLED else 'not_configured'
        }
    })

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
    """Endpoint for getting source credibility chart data with mock data"""
    try:
        mock_data = {
            'sources': ['BBC', 'Reuters', 'CNN', 'Fox News', 'The Guardian'],
            'credibility_scores': [0.9, 0.95, 0.7, 0.4, 0.85],
            'high_counts': [45, 50, 30, 15, 40],
            'medium_counts': [10, 5, 25, 20, 10],
            'low_counts': [5, 2, 10, 30, 5],
            'total_counts': [60, 57, 65, 65, 55]
        }
        return jsonify({'status': 'success', 'data': mock_data})
    except Exception as e:
        logger.error(f"Error in source_credibility_chart endpoint: {str(e)}")
        return jsonify({
            'error': 'An error occurred while fetching chart data',
            'status': 500,
            'details': str(e)
        }), 500

@app.route('/analysis-history', methods=['GET'])
def analysis_history():
    """Endpoint for getting analysis history with mock data"""
    try:
        mock_history = [
            {
                'url': 'https://example.com/article1',
                'title': 'Sample Article 1',
                'source': 'BBC',
                'credibility': 'High',
                'summary': 'This is a sample article about current events',
                'date': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')
            },
            {
                'url': 'https://example.com/article2',
                'title': 'Sample Article 2',
                'source': 'Reuters',
                'credibility': 'Medium',
                'summary': 'Another sample article demonstrating functionality',
                'date': (datetime.now(timezone.utc) - timedelta(days=1)).strftime('%Y-%m-%d %H:%M')
            }
        ]
        return jsonify({'status': 'success', 'history': mock_history})
    except Exception as e:
        logger.error(f"Error in analysis_history endpoint: {str(e)}")
        return jsonify({
            'error': 'An error occurred while fetching analysis history',
            'status': 500,
            'details': str(e)
        }), 500

def extract_text_from_url(url):
    """Extract text from URL with improved error handling and additional checks"""
    try:
        logger.info(f"Attempting to process URL: {url}")

        # Validate URL format
        try:
            parsed = urlparse(url)
            if not all([parsed.scheme, parsed.netloc]):
                logger.error(f"Invalid URL format: {url}")
                return None, None, None
        except Exception as e:
            logger.error(f"URL parsing error for {url}: {str(e)}")
            return None, None, None

        # Normalize URL
        clean_url = urlunparse(parsed._replace(
            scheme=parsed.scheme.lower(),
            netloc=parsed.netloc.lower()
        ))

        # Check for video content
        if any(domain in clean_url for domain in ['youtube.com', 'vimeo.com', 'twitch.tv']):
            logger.info("Video content detected")
            return "Video content detected", parsed.netloc.replace('www.', ''), "Video: " + clean_url

        # Configure custom headers for different domains
        headers = {
            'User-Agent': user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }

        # Special headers for Reuters
        if 'reuters.com' in clean_url:
            headers.update({
                'Referer': 'https://www.google.com/',
                'Accept-Encoding': 'gzip, deflate, br'
            })

        # Check URL accessibility
        try:
            # First try with HEAD request
            try:
                response = requests.head(
                    clean_url,
                    timeout=10,
                    allow_redirects=True,
                    headers=headers
                )
                response.raise_for_status()
            except requests.RequestException as e:
                logger.warning(f"HEAD request failed, trying GET: {str(e)}")
                # If HEAD fails, try GET
                response = requests.get(
                    clean_url,
                    timeout=15,
                    allow_redirects=True,
                    headers=headers
                )
                response.raise_for_status()

        except requests.RequestException as e:
            logger.error(f"URL accessibility check failed for {clean_url}: {str(e)}")
            return None, None, None

        # Configure article with timeout and user agent
        config = Config()
        config.browser_user_agent = user_agent
        config.request_timeout = 30
        config.memoize_articles = False
        config.fetch_images = False
        config.follow_meta_refresh = True

        # Special configuration for Reuters
        if 'reuters.com' in clean_url:
            config.browser_user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            config.fetch_images = False
            config.follow_meta_refresh = True
            config.keep_article_html = True

        article = Article(clean_url, config=config)

        # Download article with timeout
        try:
            article.download()
            if article.download_state != 2:
                logger.error(f"Failed to download article from {clean_url}")
                return None, None, None
        except Exception as e:
            logger.error(f"Article download failed: {str(e)}")
            return None, None, None

        # Parse article
        try:
            article.parse()
            if not article.text or len(article.text.strip()) < 100:
                logger.warning(f"Short or empty content from {clean_url}")
                return None, None, None
        except Exception as e:
            logger.error(f"Article parsing failed: {str(e)}")
            return None, None, None

        # For Reuters, try to extract content from HTML if parsing fails
        if 'reuters.com' in clean_url and (not article.text or len(article.text.strip()) < 100):
            try:
                response = requests.get(clean_url, headers=headers, timeout=15)
                soup = BeautifulSoup(response.text, 'html.parser')

                # Try to find article content in the HTML
                article_body = soup.find('div', {'class': 'ArticleBodyWrapper'})
                if article_body:
                    article.text = article_body.get_text(separator=' ', strip=True)
                    if len(article.text.strip()) > 100:
                        logger.info(f"Successfully extracted content from Reuters article")
                    else:
                        return None, None, None
                else:
                    return None, None, None
            except Exception as e:
                logger.error(f"Failed to extract content from Reuters HTML: {str(e)}")
                return None, None, None

        # Extract domain and title
        domain = parsed.netloc.replace('www.', '')
        title = article.title.strip() if article.title else "No title"

        logger.info(f"Successfully extracted content from {clean_url}")
        return article.text.strip(), domain, title

    except Exception as e:
        logger.error(f"Unexpected error extracting article from {url}: {str(e)}", exc_info=True)
        return None, None, None

def analyze_with_claude(content, source):
    """Analyze article text using Claude API with fallback to mock data"""
    try:
        if not ANTHROPIC_API_KEY or ANTHROPIC_API_KEY == 'mock-key':
            logger.info("Using mock data for Claude API")
            return generate_mock_analysis(source)

        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

        max_chars = 10000
        if len(content) > max_chars:
            content = content[:max_chars]
            logger.warning(f"Article content truncated to {max_chars} characters")

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

        try:
            response = client.messages.create(
                model=MODEL_NAME,
                max_tokens=2000,
                temperature=0.2,
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = response.content[0].text
            json_match = re.search(r'```json\s*(\{.*\})\s*```', response_text, re.DOTALL)

            if json_match:
                return json.loads(json_match.group(1))
            else:
                try:
                    return json.loads(response_text)
                except json.JSONDecodeError:
                    logger.error("Failed to parse API response as JSON")
                    return generate_mock_analysis(source)

        except Exception as e:
            logger.error(f"Error calling Claude API: {str(e)}")
            return generate_mock_analysis(source)

    except Exception as e:
        logger.error(f"Unexpected error in analyze_with_claude: {str(e)}")
        return generate_mock_analysis(source)

def generate_mock_analysis(source):
    """Generate mock analysis data"""
    credibility_scores = {
        'bbc.com': 0.9,
        'reuters.com': 0.95,
        'apnews.com': 0.9,
        'nytimes.com': 0.85,
        'theguardian.com': 0.8
    }

    base_score = credibility_scores.get(source, 0.7)
    return {
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
    """Save analysis to database with improved error handling"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()

            # Safely get all analysis values with defaults
            integrity = analysis.get('news_integrity', 0.5)
            fact_check = analysis.get('fact_check_needed_score', 0.5)
            sentiment = analysis.get('sentiment_score', 0.5)
            bias = analysis.get('bias_score', 0.5)
            summary = analysis.get('short_summary', 'No summary available')
            credibility = analysis.get('index_of_credibility', 0.5)
            level = calculate_credibility(integrity, fact_check, sentiment, bias)

            # Generate unique URL for text inputs
            if url and url.startswith(('http://', 'https://')):
                db_url = url
            else:
                db_url = f'text_{datetime.now(timezone.utc).timestamp()}'

            # Try to insert the record
            try:
                cursor.execute('''
                    INSERT INTO news
                    (url, title, source, content, integrity, fact_check, sentiment, bias,
                    credibility_level, short_summary, index_of_credibility, analysis_date)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (db_url, title, source, content, integrity, fact_check,
                      sentiment, bias, level, summary, credibility,
                      datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')))

                conn.commit()
                logger.info(f"Successfully saved analysis for {title} from {source}")
                return level
            except sqlite3.IntegrityError as e:
                # Handle UNIQUE constraint violation
                if "UNIQUE constraint failed: news.url" in str(e):
                    logger.warning(f"URL already exists in database: {db_url}")

                    # Update existing record instead
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
                logger.error(f"Error saving analysis: {str(e)}")
                raise

    except Exception as e:
        logger.error(f"Unexpected error in save_analysis: {str(e)}")
        return 'Medium'  # Default credibility level

def generate_query_from_analysis(analysis_result):
    """Generate search query from analysis results"""
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
        return ' AND '.join(filtered_terms[:5])
    elif filtered_terms:
        return ' OR '.join(filtered_terms)
    else:
        return 'news OR current events'

def fetch_same_topic_articles(analysis_result, page=1, per_page=3):
    """Fetch similar articles using News API with fallback to mock data"""
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
            return [article for article, score in ranked_articles[:per_page]]

        except requests.RequestException as e:
            logger.error(f"News API request failed: {str(e)}")
            return generate_mock_articles(per_page)

    except Exception as e:
        logger.error(f"Error fetching similar articles: {str(e)}")
        return generate_mock_articles(per_page)

def generate_mock_articles(count):
    """Generate mock articles for when API is not available"""
    return [
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

def render_same_topic_articles_html(articles):
    """Render HTML for similar articles with show more button"""
    if not articles:
        return ''

    html_items = []
    for art in articles:
        title = html.escape(art.get('title', 'No Title'))
        article_url = html.escape(art.get('url', '#'))
        source_name = html.escape(art.get('source', {}).get('name', 'Unknown Source'))
        published_at = html.escape(art.get('publishedAt', 'N/A').split('T')[0])
        description = html.escape(art.get('description', 'No description available.'))
        trust_score = art.get('trust_score', 0.7)

        html_items.append(f'''
        <div class="similar-article">
            <h4><a href="{article_url}" target="_blank">{title}</a></h4>
            <div class="article-meta">
                <span class="article-source">{source_name}</span>
                <span class="article-date">{published_at}</span>
                <span class="article-credibility">Credibility: {int(trust_score*100)}%</span>
            </div>
            <p class="article-description">{description}</p>
        </div>
        ''')

    return ''.join(html_items)

def format_analysis_results(title, source, analysis, credibility):
    """Format analysis results for display with all sections"""
    try:
        integrity = analysis.get('news_integrity', 0.5)
        fact_check = analysis.get('fact_check_needed_score', 0.5)
        sentiment = analysis.get('sentiment_score', 0.5)
        bias = analysis.get('bias_score', 0.5)
        summary = html.escape(analysis.get('short_summary', 'No summary available'))
        topics = [html.escape(str(t)) for t in analysis.get('topics', [])]
        key_args = [html.escape(str(a)) for a in analysis.get('key_arguments', [])]
        biases = [html.escape(str(b)) for b in analysis.get('potential_biases_identified', [])]

        topics_html = ' '.join(f'<span class="badge bg-primary">{topic}</span>' for topic in topics)
        key_args_html = ''.join(f'<li>{arg}</li>' for arg in key_args)
        biases_html = ''.join(f'<li>{bias}</li>' for bias in biases)

        return {
            'title': title,
            'source': source,
            'credibility': credibility,
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
                            <div class="score-name">News Integrity</div>
                            <div class="score-value">{integrity:.2f}</div>
                            <div class="score-description">Overall integrity and trustworthiness</div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="score-item">
                            <div class="score-name">Fact Check Needed</div>
                            <div class="score-value">{fact_check:.2f}</div>
                            <div class="score-description">Likelihood that claims need fact-checking</div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="score-item">
                            <div class="score-name">Sentiment</div>
                            <div class="score-value">{sentiment:.2f}</div>
                            <div class="score-description">Emotional tone (0.0 negative, 0.5 neutral, 1.0 positive)</div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="score-item">
                            <div class="score-name">Bias</div>
                            <div class="score-value">{bias:.2f}</div>
                            <div class="score-description">Degree of perceived bias</div>
                        </div>
                    </div>
                </div>
            </div>

            <div class="analysis-section">
                <h2>Additional Information</h2>
                <div class="detail-item">
                    <h4>Short Summary</h4>
                    <p>{summary}</p>
                </div>

                <div class="detail-item">
                    <h4>Topics</h4>
                    <div class="d-flex flex-wrap gap-2">
                        {topics_html}
                    </div>
                </div>

                <div class="detail-item">
                    <h4>Key Arguments</h4>
                    <ul class="list-unstyled">
                        {key_args_html}
                    </ul>
                </div>

                <div class="detail-item">
                    <h4>Potential Biases Identified</h4>
                    <ul class="list-unstyled">
                        {biases_html}
                    </ul>
                </div>
            </div>
            """
        }
    except Exception as e:
        logger.error(f"Error formatting analysis results: {str(e)}")
        return {
            "error": "Error formatting analysis results",
            "output_md": "<div class='alert alert-danger'>Error displaying analysis results</div>"
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
            try:
                content, source, title = extract_text_from_url(input_text)
                if not content:
                    # Special handling for Reuters
                    if 'reuters.com' in input_text.lower():
                        return jsonify({
                            'error': 'Could not extract article content from Reuters',
                            'status': 400,
                            'details': 'Reuters blocks automated requests. Try accessing the article in a browser or use a different URL.',
                            'suggestions': [
                                'Try opening the URL in a browser and copying the text',
                                'Use a different news source',
                                'Try a different URL from the same source'
                            ],
                            'request_id': str(uuid.uuid4())
                        }), 400

                    return jsonify({
                        'error': 'Could not extract article content',
                        'status': 400,
                        'details': 'Failed to download or parse the article from the provided URL',
                        'suggestions': [
                            'Check if the URL is correct and accessible',
                            'Try a different URL',
                            'Make sure the website allows scraping',
                            'Alternatively, you can paste the article text directly'
                        ],
                        'request_id': str(uuid.uuid4())
                    }), 400
            except Exception as e:
                logger.error(f"Error processing URL: {input_text}. Error: {str(e)}", exc_info=True)
                return jsonify({
                    'error': 'Error processing URL',
                    'status': 400,
                    'details': str(e),
                    'suggestions': [
                        'The website might be blocking our requests',
                        'Try a different URL',
                        'You can paste the article text directly instead of using URL'
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

        # Analyze the article content using Claude API
        try:
            logger.info("Starting article analysis with Claude API")
            analysis_start_time = time.time()
            analysis = analyze_with_claude(content, source)
            analysis_duration = time.time() - analysis_start_time
            logger.info(f"Completed article analysis in {analysis_duration:.2f} seconds")
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}", exc_info=True)
            return jsonify({
                'error': 'Analysis failed',
                'status': 500,
                'details': str(e),
                'request_id': str(uuid.uuid4())
            }), 500

        # Save analysis to database
        try:
            logger.info("Saving analysis to database")
            save_start_time = time.time()
            credibility = save_analysis(
                input_text if input_text.startswith(('http://', 'https://')) else None,
                title,
                source,
                content,
                analysis
            )
            save_duration = time.time() - save_start_time
            logger.info(f"Saved analysis to database in {save_duration:.2f} seconds")
        except Exception as e:
            logger.error(f"Failed to save analysis: {str(e)}", exc_info=True)
            return jsonify({
                'error': 'Failed to save analysis',
                'status': 500,
                'details': str(e),
                'request_id': str(uuid.uuid4())
            }), 500

        # Store analysis result in session
        session['last_analysis_result'] = analysis

        # Get similar articles
        try:
            logger.info("Fetching similar articles")
            same_topic_articles = fetch_same_topic_articles(analysis)
            same_topic_html = render_same_topic_articles_html(same_topic_articles)
            logger.info(f"Found {len(same_topic_articles)} similar articles")
        except Exception as e:
            logger.error(f"Failed to fetch similar articles: {str(e)}", exc_info=True)
            same_topic_html = '<div class="alert alert-warning">Could not fetch similar articles at this time.</div>'

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
            'scores_for_chart': {
                'news_integrity': analysis.get('news_integrity', 0.0),
                'fact_check_needed_score': analysis.get('fact_check_needed_score', 1.0),
                'sentiment_score': analysis.get('sentiment_score', 0.5),
                'bias_score': analysis.get('bias_score', 1.0),
                'index_of_credibility': analysis.get('index_of_credibility', 0.0)
            },
            'source_credibility_data': source_credibility_data,
            'same_topic_html': same_topic_html,
            'output': format_analysis_results(title, source, analysis, credibility),
            'request_id': str(uuid.uuid4()),
            'processing_time': f"{time.time() - analysis_start_time:.2f} seconds"
        }

        logger.info(f"Successfully processed analysis request. Response size: {len(json.dumps(response_data))} bytes")
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
    initialize_database()
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
