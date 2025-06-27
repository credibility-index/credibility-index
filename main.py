import os
import sys
import logging
import time
from pathlib import Path
import re
import json
import string
from datetime import datetime
from urllib.parse import urlparse
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from flask_talisman import Talisman
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_caching import Cache
import random
import anthropic
from newspaper import Article, Config
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import sentry_sdk
from sentry_sdk.integrations.flask import FlaskIntegration
from celery import Celery
import requests
import threading
from typing import Optional, List, Dict, Any, Tuple
from cache import CacheManager
from claude_api import ClaudeAPI
from news_api import NewsAPI
from redis import Redis
from redis.exceptions import ConnectionError, TimeoutError

# Initialize Flask application with improved configuration
app = Flask(__name__,
            static_folder='static',
            static_url_path='/static',
            template_folder='templates')

# Configure application
app.config.update(
    SECRET_KEY=os.getenv('SECRET_KEY', 'your-secret-key-here'),
    PERMANENT_SESSION_LIFETIME=3600,
    TEMPLATES_AUTO_RELOAD=True,
    SEND_FILE_MAX_AGE_DEFAULT=86400
)

# Configure cache with improved settings
app.config['CACHE_TYPE'] = os.getenv('CACHE_TYPE', 'RedisCache')
app.config['CACHE_REDIS_URL'] = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
app.config['CACHE_DEFAULT_TIMEOUT'] = 3600
app.config['CACHE_KEY_PREFIX'] = 'media_credibility_'
cache = Cache(app)

# Configure Talisman for security headers with improved CSP
Talisman(app,
    content_security_policy={
        'default-src': "'self'",
        'script-src': ["'self'", "'unsafe-inline'", "cdn.jsdelivr.net"],
        'style-src': ["'self'", "'unsafe-inline'", "fonts.googleapis.com"],
        'img-src': ["'self'", "data:", "https://*.googleapis.com"],
        'font-src': ["'self'", "fonts.gstatic.com"],
        'connect-src': ["'self'", "https://api.newsapi.org"],
        'frame-src': ["'self'"],
        'object-src': ["'none'"],
        'base-uri': ["'self'"],
        'form-action': ["'self'"]
    },
    force_https=os.getenv('FLASK_ENV') == 'production',
    strict_transport_security=os.getenv('FLASK_ENV') == 'production',
    session_cookie_secure=os.getenv('FLASK_ENV') == 'production',
    session_cookie_http_only=True,
    session_cookie_samesite='Lax'
)

# Configure rate limiting with improved settings
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri=os.getenv('RATELIMIT_STORAGE_URL', 'redis://redis:6379/1'),
    strategy="fixed-window",
    enabled=os.getenv('FLASK_ENV') == 'production'
)

def create_redis_connection():
    """Create Redis connection with improved retry logic"""
    redis_url = f"redis://{os.getenv('REDISUSER', 'default')}:{os.getenv('REDISPASSWORD', '')}@{os.getenv('REDISHOST', 'redis')}:{os.getenv('REDISPORT', '6379')}"
    max_retries = 5
    retry_delay = 1

    for attempt in range(max_retries):
        try:
            redis_client = Redis.from_url(
                redis_url,
                socket_connect_timeout=5,
                socket_timeout=10,
                socket_keepalive=True,
                retry_on_timeout=True,
                decode_responses=True,
                health_check_interval=30,
                max_connections=50
            )
            if redis_client.ping():
                logging.info("Successfully connected to Redis")
                return redis_client
        except (ConnectionError, TimeoutError) as e:
            if attempt == max_retries - 1:
                logging.error(f"Failed to connect to Redis after {max_retries} attempts: {str(e)}")
                return None
            wait_time = retry_delay * (2 ** attempt)
            logging.warning(f"Redis connection failed, retrying in {wait_time} seconds...")
            time.sleep(wait_time)
    return None

def make_celery(app):
    """Create and configure Celery instance with improved settings"""
    redis_url = f"redis://{os.getenv('REDISUSER', 'default')}:{os.getenv('REDISPASSWORD', '')}@{os.getenv('REDISHOST', 'redis')}:{os.getenv('REDISPORT', '6379')}/2"

    celery = Celery(
        app.import_name,
        broker=redis_url,
        backend=redis_url,
        include=['tasks']
    )

    # Configure Celery with improved settings
    celery.conf.update(
        task_serializer='json',
        result_serializer='json',
        accept_content=['json'],
        timezone='UTC',
        enable_utc=True,
        broker_connection_retry=True,
        broker_connection_max_retries=100,
        broker_transport_options={
            'visibility_timeout': 3600,
            'fanout_prefix': True,
            'fanout_patterns': True,
            'max_retries': 3,
            'interval_start': 0,
            'interval_step': 0.2,
            'interval_max': 0.5
        },
        task_default_queue='default',
        task_queues=('default', 'high_priority'),
        task_default_exchange='tasks',
        task_default_routing_key='task.default',
        task_routes={
            'tasks.analyze_article_async': {'queue': 'high_priority'},
        },
        task_time_limit=600,
        task_soft_time_limit=540,
        worker_prefetch_multiplier=4,
        worker_max_tasks_per_child=100,
        worker_max_memory_per_child=200000,
        result_expires=86400,
        broker_pool_limit=10,
        worker_concurrency=4
    )

    class ContextTask(celery.Task):
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)

    celery.Task = ContextTask
    return celery

# Initialize Celery
celery = make_celery(app)

# Global variables
analysis_history = []
history_lock = threading.Lock()

# Configure logging with improved settings
def configure_logging():
    """Configure logging settings with improved format and handlers"""
    log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    if os.getenv('FLASK_ENV') == 'development':
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s - %(filename)s:%(lineno)d'

    handlers = [logging.StreamHandler(sys.stdout)]

    if os.getenv('LOG_FILE'):
        file_handler = logging.FileHandler(os.getenv('LOG_FILE'))
        file_handler.setFormatter(logging.Formatter(log_format))
        handlers.append(file_handler)

    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format=log_format,
        handlers=handlers
    )

    if os.getenv('SENTRY_DSN'):
        sentry_sdk.init(
            dsn=os.getenv('SENTRY_DSN'),
            integrations=[FlaskIntegration()],
            traces_sample_rate=1.0,
            environment=os.getenv('FLASK_ENV', 'development'),
            send_default_pii=True
        )

# Configure logging
configure_logging()

def check_redis_connection():
    """Check Redis connection at startup with improved diagnostics"""
    try:
        redis_url = f"redis://{os.getenv('REDISUSER', 'default')}:{os.getenv('REDISPASSWORD', '')}@{os.getenv('REDISHOST', 'redis')}:{os.getenv('REDISPORT', '6379')}"
        redis_client = Redis.from_url(
            redis_url,
            socket_connect_timeout=5,
            socket_timeout=10,
            socket_keepalive=True,
            retry_on_timeout=True,
            decode_responses=True
        )

        if redis_client.ping():
            logging.info("Redis connection test successful")
            return True

        logging.warning("Redis connection test failed - no response to PING")
        return False
    except Exception as e:
        logging.error(f"Redis connection check failed: {str(e)}")
        return False

# Initialize components with improved error handling
try:
    cache_manager = CacheManager()
    claude_api = ClaudeAPI()
    news_api = NewsAPI()

    if not check_redis_connection():
        logging.warning("Could not connect to Redis - some features may not work properly")
    else:
        logging.info("Successfully connected to Redis")

    # Test NewsAPI connection
    try:
        test_query = "technology"
        articles = news_api.get_everything(query=test_query, page_size=1)
        if articles:
            logging.info("Successfully connected to NewsAPI")
    except Exception as e:
        logging.warning(f"NewsAPI connection test failed: {str(e)}")

except Exception as e:
    logging.error(f"Failed to initialize components: {str(e)}")
    raise

# Test data with improved structure
daily_buzz = {
    "article": {
        "title": "Today's featured analysis",
        "source": "Media Analysis",
        "short_summary": "Analysis of current events and news trends with credibility assessment",
        "analysis": {
            "credibility_score": {
                "score": 0.85,
                "explanation": "Based on multiple reliable sources"
            },
            "topics": ["News", "Analysis", "Media"],
            "summary": "Detailed analysis of current events with credibility assessment",
            "perspectives": {
                "western": {
                    "summary": "Western perspective on current events",
                    "credibility": "High"
                },
                "neutral": {
                    "summary": "Neutral analysis of current events",
                    "credibility": "High"
                }
            },
            "sentiment": {
                "score": 0.7,
                "explanation": "Generally positive sentiment"
            },
            "bias": {
                "level": 0.2,
                "explanation": "Minimal bias detected"
            }
        }
    }
}

source_credibility_data = {
    "sources": [
        "BBC", "Reuters", "CNN", "The Guardian",
        "New York Times", "Washington Post", "Al Jazeera",
        "Associated Press", "Bloomberg", "Financial Times"
    ],
    "credibility_scores": [0.92, 0.88, 0.75, 0.89, 0.91, 0.85, 0.72, 0.90, 0.87, 0.91]
}

# Error handlers with improved responses
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500

@app.errorhandler(403)
def forbidden(e):
    return render_template('403.html'), 403

@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({
        "status": "error",
        "message": "Too many requests",
        "details": "Please try again later"
    }), 429

# Static files route with improved caching
@app.route('/static/<path:filename>')
@cache.cached(timeout=86400)
def static_files(filename):
    cache_timeout = app.get_send_file_max_age(filename)
    response = send_from_directory(app.static_folder, filename)
    response.headers['Cache-Control'] = f'public, max-age={cache_timeout}'
    return response

# Cache headers middleware
@app.after_request
def add_cache_headers(response):
    if request.path.startswith('/static/'):
        response.headers['Cache-Control'] = 'public, max-age=86400'
    elif request.path.startswith('/api/'):
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    return response

# Context processor for global template variables
@app.context_processor
def inject_global_variables():
    return dict(
        app_version=os.getenv('APP_VERSION', '1.0.0'),
        debug_mode=os.getenv('FLASK_ENV') == 'development',
        current_year=datetime.now().year,
        static_version=os.getenv('STATIC_VERSION', '1')
    )

# Routes with improved implementations
@app.route('/')
@limiter.exempt
def index():
    """Main page of the application with improved context"""
    return render_template('index.html',
                         version=os.getenv('APP_VERSION', '1.0.0'),
                         debug=os.getenv('FLASK_ENV') == 'development')

@app.route('/faq')
@cache.cached(timeout=3600)
def faq():
    return render_template('faq.html')

@app.route('/feedback')
def feedback():
    return render_template('feedback.html')

@app.route('/privacy')
@cache.cached(timeout=3600)
def privacy():
    return render_template('privacy.html')

@app.route('/terms')
@cache.cached(timeout=3600)
def terms():
    return render_template('terms.html')

@app.route('/source-credibility-chart')
@cache.cached(timeout=300)
def get_source_credibility_chart():
    """Returns data for source credibility chart with improved response"""
    try:
        return jsonify({
            "status": "success",
            "data": source_credibility_data,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logging.error(f"Error getting source credibility chart: {str(e)}")
        return jsonify({
            "status": "error",
            "message": "Failed to get source credibility data",
            "data": {
                "sources": ["BBC", "Reuters", "CNN"],
                "credibility_scores": [0.9, 0.85, 0.75]
            }
        }), 500

@app.route('/analysis-history')
@cache.cached(timeout=60)
def get_analysis_history():
    """Returns analysis history with improved response format"""
    try:
        with history_lock:
            return jsonify({
                "status": "success",
                "history": analysis_history,
                "count": len(analysis_history),
                "timestamp": datetime.now().isoformat()
            })
    except Exception as e:
        logging.error(f"Error getting analysis history: {str(e)}")
        return jsonify({
            "status": "error",
            "message": "Failed to get analysis history",
            "history": []
        }), 500

@celery.task(bind=True, max_retries=3)
def analyze_article_async(self, url_or_text: str) -> Dict[str, Any]:
    """Asynchronous task for article analysis with improved implementation"""
    try:
        # Check cache first
        cached_result = cache_manager.get_cached_article_analysis(url_or_text)
        if cached_result:
            return {"status": "success", "result": cached_result, "cached": True}

        self.update_state(state='PROGRESS', meta={'progress': 10, 'message': 'Starting analysis'})

        # Extract content with improved handling
        if url_or_text.startswith(('http://', 'https://')):
            content, source, title, error = extract_text_from_url(url_or_text)
            if error:
                return {"status": "error", "message": error}

            self.update_state(state='PROGRESS', meta={'progress': 30, 'message': 'Article extracted'})
        else:
            content = url_or_text
            source = 'Direct Input'
            title = 'User-provided Text'

        # Analyze with Claude API with progress updates
        self.update_state(state='PROGRESS', meta={'progress': 40, 'message': 'Analyzing content'})

        analysis = claude_api.analyze_article(content, source)
        self.update_state(state='PROGRESS', meta={'progress': 60, 'message': 'Analysis completed'})

        # Build query for similar articles
        query = build_newsapi_query(analysis)

        # Get similar articles with progress updates
        similar_articles = []
        if query:
            self.update_state(state='PROGRESS', meta={'progress': 70, 'message': 'Finding similar articles'})
            similar_articles = news_api.get_everything(query=query, page_size=5)
            self.update_state(state='PROGRESS', meta={'progress': 80, 'message': 'Similar articles found'})

        # Determine credibility level
        credibility_level = claude_api.determine_credibility_level(analysis.get('credibility_score', {}).get('score', 0.6))

        # Build comprehensive result
        result = {
            'title': title,
            'source': source,
            'url': url_or_text if url_or_text.startswith(('http://', 'https://')) else None,
            'short_summary': content[:200] + '...' if len(content) > 200 else content,
            'analysis': analysis,
            'credibility_level': credibility_level,
            'similar_articles': similar_articles,
            'search_query': query,
            'metadata': {
                'analysis_timestamp': datetime.now().isoformat(),
                'content_length': len(content),
                'source_credibility': claude_api.determine_credibility_level_from_source(source)
            }
        }

        # Cache result with improved cache key
        cache_key = f"analysis_{hash(url_or_text)}"
        cache_manager.cache_article_analysis(cache_key, result)

        # Update analysis history with thread safety
        with history_lock:
            global analysis_history
            analysis_history.insert(0, {
                "title": title,
                "source": source,
                "url": url_or_text if url_or_text.startswith(('http://', 'https://')) else None,
                "summary": content[:200] + '...' if len(content) > 200 else content,
                "credibility": credibility_level,
                "timestamp": datetime.now().isoformat(),
                "credibility_score": result['analysis']['credibility_score']['score']
            })
            analysis_history = analysis_history[:10]  # Keep only last 10 entries

        self.update_state(state='PROGRESS', meta={'progress': 100, 'message': 'Completed'})
        return {"status": "success", "result": result}

    except Exception as e:
        logging.error(f"Error in async article analysis: {str(e)}", exc_info=True)
        return {"status": "error", "message": str(e)}

def build_newsapi_query(analysis: dict) -> str:
    """Builds query for NewsAPI based on article analysis with improved logic"""
    try:
        query_parts = []

        # Add topics with improved handling
        topics = analysis.get('content_analysis', {}).get('main_topics', [])
        if isinstance(topics, list):
            query_parts.extend([t['name'] if isinstance(t, dict) else t for t in topics[:3]])

        # Add important entities with improved handling
        key_arguments = analysis.get('content_analysis', {}).get('key_arguments', [])
        if isinstance(key_arguments, list):
            important_arguments = [arg['argument'] for arg in key_arguments if isinstance(arg, dict)]
            query_parts.extend(important_arguments[:3])

        # Add named entities if available
        entities = analysis.get('content_analysis', {}).get('named_entities', [])
        if isinstance(entities, list):
            query_parts.extend([e['name'] for e in entities if isinstance(e, dict) and e.get('type') in ['PERSON', 'ORG', 'GPE']][:3])

        return ' OR '.join(query_parts) if query_parts else "technology"

    except Exception as e:
        logging.error(f"Error building NewsAPI query: {str(e)}")
        return "technology"

def extract_text_from_url(url: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Extracts text from URL with improved implementation"""
    try:
        parsed = urlparse(url)
        if not all([parsed.scheme, parsed.netloc]):
            return None, None, None, "Invalid URL format"

        clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"

        # Skip video sites and social media
        if any(domain in parsed.netloc for domain in [
            'youtube.com', 'vimeo.com', 'twitch.tv', 'tiktok.com',
            'facebook.com', 'twitter.com', 'instagram.com'
        ]):
            return None, parsed.netloc.replace('www.', ''), "Video or social media content detected", None

        # Configure newspaper with improved settings
        config = Config()
        config.browser_user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36'
        config.request_timeout = 30
        config.memoize_articles = False
        config.fetch_images = False

        try:
            article = Article(clean_url, config=config)
            article.download()
            article.parse()

            if article.text and len(article.text.strip()) >= 100:
                return (
                    article.text.strip(),
                    parsed.netloc.replace('www.', ''),
                    article.title.strip() if article.title else "No title available",
                    None
                )

        except Exception as e:
            logging.warning(f"Newspaper failed to process {url}: {str(e)}")

        # Alternative content extraction method with improved handling
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1'
            }

            session = requests.Session()
            retries = Retry(
                total=3,
                backoff_factor=1,
                status_forcelist=[500, 502, 503, 504],
                allowed_methods=["GET", "POST"]
            )
            adapter = HTTPAdapter(max_retries=retries)
            session.mount("http://", adapter)
            session.mount("https://", adapter)

            response = session.get(clean_url, headers=headers, timeout=20)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')

            # Remove unwanted elements with improved handling
            for element in soup(['script', 'style', 'noscript', 'iframe', 'svg', 'nav', 'footer', 'header', 'aside', 'form']):
                element.decompose()

            # Try to find main content with improved selectors
            main_content = soup.find('article') or \
                        soup.find('div', {'role': 'main'}) or \
                        soup.find('div', {'class': re.compile('article|content|main|post|entry|story')}) or \
                        soup.find('main') or \
                        soup.find('div', {'id': re.compile('article|content|main|post|entry|story')})

            if main_content:
                paragraphs = main_content.find_all('p')
                if paragraphs:
                    text = ' '.join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
                    if len(text) >= 100:
                        return (
                            text,
                            parsed.netloc.replace('www.', ''),
                            soup.title.string.strip() if soup.title else "No title available",
                            None
                        )

            # If no main content found, get all paragraphs with improved handling
            paragraphs = soup.find_all('p')
            if paragraphs:
                text = ' '.join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
                if len(text) >= 100:
                    return (
                        text,
                        parsed.netloc.replace('www.', ''),
                        soup.title.string.strip() if soup.title else "No title available",
                        None
                    )

            return None, parsed.netloc.replace('www.', ''), "Failed to extract sufficient content", None

        except Exception as e:
            logging.error(f"Alternative extraction failed for {url}: {str(e)}")
            return None, parsed.netloc.replace('www.', ''), "Error occurred during extraction", str(e)

    except Exception as e:
        logging.error(f"Unexpected error extracting article from {url}: {str(e)}")
        return None, None, "Unexpected error occurred", str(e)

def determine_credibility_level(score: float) -> str:
    """Determines credibility level with improved input handling"""
    try:
        if isinstance(score, dict):
            score = score.get('score', 0.6)

        if not isinstance(score, (float, int)):
            score = 0.6  # Default value

        score = float(score)

        if score >= 0.85:
            return "High"
        elif score >= 0.65:
            return "Medium"
        else:
            return "Low"

    except Exception as e:
        logging.error(f"Error determining credibility level: {str(e)}")
        return "Medium"

def determine_credibility_level_from_source(source_name: str) -> str:
    """Determines credibility level from source with improved handling"""
    try:
        source_name = source_name.lower().strip()

        high_credibility_sources = [
            'bbc.com', 'reuters.com', 'nytimes.com', 'theguardian.com',
            'washingtonpost.com', 'wsj.com', 'ft.com', 'economist.com',
            'apnews.com', 'bloomberg.com', 'npr.org', 'scientificamerican.com'
        ]

        medium_credibility_sources = [
            'cnn.com', 'foxnews.com', 'usatoday.com', 'washingtonpost.com',
            'npr.org', 'aljazeera.com', 'theindependent.co.uk',
            'newsweek.com', 'time.com', 'usnews.com'
        ]

        low_credibility_sources = [
            'dailymail.co.uk', 'breitbart.com', 'infowars.com',
            'thesun.co.uk', 'rt.com', 'sputniknews.com',
            'theonion.com', 'nationalenquirer.com'
        ]

        # Extract domain from URL if needed
        if source_name.startswith(('http://', 'https://')):
            parsed = urlparse(source_name)
            domain = parsed.netloc.lower().replace('www.', '')
        else:
            domain = source_name

        # Check for high credibility sources
        if any(hcs in domain for hcs in high_credibility_sources):
            return "High"

        # Check for low credibility sources
        if any(lcs in domain for lcs in low_credibility_sources):
            return "Low"

        # Check for medium credibility sources
        if any(mcs in domain for mcs in medium_credibility_sources):
            return "Medium"

        # Check for government or educational domains
        if domain.endswith('.gov') or domain.endswith('.edu'):
            return "High"

        # Check for known blog platforms
        if any(platform in domain for platform in ['blogspot.com', 'wordpress.com', 'medium.com', 'substack.com']):
            return "Medium"

        # Default to medium credibility
        return "Medium"

    except Exception as e:
        logging.error(f"Error determining source credibility: {str(e)}")
        return "Medium"

@app.route('/start-analysis', methods=['POST', 'OPTIONS'])
@limiter.limit("10 per minute")
def start_analysis():
    """Starts asynchronous article analysis with improved implementation"""
    if request.method == 'OPTIONS':
        return jsonify({}), 200

    try:
        data = request.get_json()
        if not data or 'input_text' not in data:
            return jsonify({'status': 'error', 'message': 'Input text is required'}), 400

        input_text = data['input_text'].strip()
        if not input_text:
            return jsonify({'status': 'error', 'message': 'Input text cannot be empty'}), 400

        # Additional URL validation with improved checks
        if input_text.startswith(('http://', 'https://')):
            valid, error = validate_url(input_text)
            if not valid:
                return jsonify({'status': 'error', 'message': error}), 400
        elif len(input_text) < 50:
            return jsonify({'status': 'error', 'message': 'Content is too short for analysis (minimum 50 characters)'}), 400

        # Create analysis task with improved handling
        task = analyze_article_async.apply_async(
            args=[input_text],
            queue='high_priority',
            priority=5
        )

        return jsonify({
            'status': 'started',
            'task_id': task.id,
            'message': 'Analysis started successfully',
            'estimated_completion': 30  # seconds
        })

    except Exception as e:
        logging.error(f"Error starting analysis: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': 'Failed to start analysis',
            'details': str(e)
        }), 500

@app.route('/task-status/<task_id>')
@cache.cached(timeout=10)
def get_task_status(task_id):
    """Checks status of asynchronous task with improved response"""
    try:
        task = analyze_article_async.AsyncResult(task_id)

        if task.state == 'PENDING':
            response = {
                'status': task.state,
                'message': 'Task not yet started',
                'progress': 0,
                'estimated_completion': 30
            }
        elif task.state == 'PROGRESS':
            response = {
                'status': task.state,
                'progress': task.info.get('progress', 0),
                'message': task.info.get('message', 'Processing'),
                'estimated_completion': max(5, 30 - task.info.get('progress', 0) // 3)
            }
        elif task.state == 'SUCCESS':
            response = {
                'status': task.state,
                'result': task.result,
                'message': 'Task completed successfully',
                'completed_at': datetime.now().isoformat()
            }
        else:  # FAILURE or other states
            response = {
                'status': task.state,
                'message': str(task.info) if task.info else 'Task failed',
                'error': str(task.info) if task.info else 'Unknown error'
            }

        return jsonify(response)

    except Exception as e:
        logging.error(f"Error getting task status: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': 'Failed to get task status',
            'details': str(e)
        }), 500

@app.route('/clear-cache', methods=['POST'])
@limiter.limit("5 per hour")
def clear_cache():
    """Clears analysis cache with improved implementation"""
    try:
        cache_manager.clear_all_caches()
        return jsonify({
            'status': 'success',
            'message': 'Cache cleared successfully',
            'cleared_at': datetime.now().isoformat()
        })
    except Exception as e:
        logging.error(f"Error clearing cache: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': 'Failed to clear cache',
            'details': str(e)
        }), 500

@app.route('/health')
@cache.cached(timeout=60)
def health_check():
    """Application health check with improved diagnostics"""
    try:
        api_status = {
            'news_api': {'status': 'unavailable', 'details': 'Not checked'},
            'claude_api': {'status': 'unavailable', 'details': 'Not checked'},
            'redis': {'status': 'unavailable', 'details': 'Not checked'},
            'database': {'status': 'unavailable', 'details': 'Not checked'}
        }

        # Check Redis connection with improved diagnostics
        try:
            redis_url = f"redis://{os.getenv('REDISUSER', 'default')}:{os.getenv('REDISPASSWORD', '')}@{os.getenv('REDISHOST', 'redis')}:{os.getenv('REDISPORT', '6379')}"
            redis_client = Redis.from_url(
                redis_url,
                socket_connect_timeout=5,
                socket_timeout=10,
                socket_keepalive=True
            )

            if redis_client.ping():
                api_status['redis'] = {
                    'status': 'operational',
                    'details': 'Connection successful',
                    'response_time': 'fast'
                }
            else:
                api_status['redis'] = {
                    'status': 'degraded',
                    'details': 'Connection established but no response to PING',
                    'response_time': 'slow'
                }

        except Exception as e:
            logging.warning(f"Redis health check failed: {str(e)}")
            api_status['redis'] = {
                'status': 'unavailable',
                'details': str(e),
                'response_time': 'none'
            }

        # Check NewsAPI with improved diagnostics
        try:
            start_time = time.time()
            test_result = news_api.get_everything("test", page_size=1)
            response_time = time.time() - start_time

            if test_result:
                api_status['news_api'] = {
                    'status': 'operational',
                    'details': 'API responded successfully',
                    'response_time': f'{response_time:.2f}s'
                }
            else:
                api_status['news_api'] = {
                    'status': 'degraded',
                    'details': 'API responded but returned empty results',
                    'response_time': f'{response_time:.2f}s'
                }

        except Exception as e:
            logging.warning(f"NewsAPI health check failed: {str(e)}")
            api_status['news_api'] = {
                'status': 'unavailable',
                'details': str(e),
                'response_time': 'none'
            }

        # Check ClaudeAPI with improved diagnostics
        try:
            start_time = time.time()
            if hasattr(claude_api, 'health_check') and callable(claude_api.health_check):
                health_status = claude_api.health_check()
                response_time = time.time() - start_time

                if health_status:
                    api_status['claude_api'] = {
                        'status': 'operational',
                        'details': 'API responded successfully',
                        'response_time': f'{response_time:.2f}s'
                    }
                else:
                    api_status['claude_api'] = {
                        'status': 'degraded',
                        'details': 'API responded but health check failed',
                        'response_time': f'{response_time:.2f}s'
                    }
            else:
                api_status['claude_api'] = {
                    'status': 'unknown',
                    'details': 'Health check method not available',
                    'response_time': 'none'
                }

        except Exception as e:
            logging.warning(f"ClaudeAPI health check failed: {str(e)}")
            api_status['claude_api'] = {
                'status': 'unavailable',
                'details': str(e),
                'response_time': 'none'
            }

        # Check database connection if available
        try:
            if hasattr(app, 'db') and hasattr(app.db, 'engine'):
                start_time = time.time()
                with app.db.engine.connect() as conn:
                    conn.execute("SELECT 1")
                    response_time = time.time() - start_time

                api_status['database'] = {
                    'status': 'operational',
                    'details': 'Connection successful',
                    'response_time': f'{response_time:.2f}s'
                }
        except Exception as e:
            logging.warning(f"Database health check failed: {str(e)}")
            api_status['database'] = {
                'status': 'unavailable',
                'details': str(e),
                'response_time': 'none'
            }

        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'api_status': api_status,
            'application': {
                'version': os.getenv('APP_VERSION', '1.0.0'),
                'environment': os.getenv('FLASK_ENV', 'development'),
                'uptime': time.time() - app.start_time if hasattr(app, 'start_time') else 'unknown'
            }
        })

    except Exception as e:
        logging.error(f"Error during health check: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'unhealthy',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

def validate_url(url: str) -> Tuple[bool, Optional[str]]:
    """Validates URL with improved checks"""
    try:
        parsed = urlparse(url)

        if not all([parsed.scheme, parsed.netloc]):
            return False, "Invalid URL format - missing scheme or netloc"

        if parsed.scheme not in ('http', 'https'):
            return False, "Invalid URL scheme - must be http or https"

        if not parsed.netloc.replace('.', '').isalnum():
            return False, "Invalid domain name in URL"

        # Check for common invalid patterns
        invalid_patterns = [
            r'\.\./',  # Path traversal
            r'%00',    # Null byte
            r'<!--',    # HTML comment
            r'<script', # Script tag
            r'onerror=', # XSS attempt
            r'javascript:', # JavaScript protocol
            r'data:', # Data URI
            r'file://' # File protocol
        ]

        for pattern in invalid_patterns:
            if re.search(pattern, url, re.IGNORECASE):
                return False, f"Invalid URL pattern detected: {pattern}"

        # Check domain length
        if len(parsed.netloc) > 253:
            return False, "Domain name too long"

        # Check path length
        if len(parsed.path) > 2048:
            return False, "URL path too long"

        return True, None

    except Exception as e:
        return False, f"URL validation error: {str(e)}"

def check_newsapi_connection():
    """Check NewsAPI connection at startup with improved diagnostics"""
    try:
        test_query = "technology"
        start_time = time.time()
        articles = news_api.get_everything(query=test_query, page_size=1)
        response_time = time.time() - start_time

        if articles:
            logging.info(f"Successfully connected to NewsAPI during startup check (response time: {response_time:.2f}s)")
            return True
        else:
            logging.warning(f"NewsAPI connection test returned empty results (response time: {response_time:.2f}s)")
            return False

    except Exception as e:
        logging.error(f"NewsAPI connection check failed: {str(e)}")
        return False

if __name__ == '__main__':
    # Record application start time
    app.start_time = time.time()

    # Check NewsAPI connection at startup
    if not check_newsapi_connection():
        logging.warning("Could not connect to NewsAPI - will use fallback data")

    # Run application with improved settings
    app.run(
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 5000)),
        threaded=True,
        debug=os.getenv('FLASK_ENV') == 'development',
        use_reloader=os.getenv('FLASK_ENV') == 'development',
        ssl_context='adhoc' if os.getenv('FLASK_ENV') == 'production' else None
    )
