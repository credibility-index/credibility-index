import os
import sys
import logging
from pathlib import Path
import re
import json
from datetime import datetime
from urllib.parse import urlparse
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from flask_talisman import Talisman
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_caching import Cache
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
from news_api import NewsAPI  # Удалено упоминание EnhancedNewsAPI

# Initialize Flask application
app = Flask(__name__, static_folder='static', template_folder='templates')

# Configure cache
app.config['CACHE_TYPE'] = os.getenv('CACHE_TYPE', 'SimpleCache')
cache = Cache(app)

# Configure Talisman for security headers
Talisman(app,
    content_security_policy={
        'default-src': "'self'",
        'script-src': ["'self'", "'unsafe-inline'"],
        'style-src': ["'self'", "'unsafe-inline'"],
        'img-src': ["'self'", "data:"]
    }
)

# Configure rate limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri=os.getenv('RATELIMIT_STORAGE_URL', 'memory://')
)

# Configure Celery with proper Redis connection handling
def make_celery(app):
    """Create and configure Celery instance"""
    celery = Celery(
        app.import_name,
        broker=os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0'),
        backend=os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')
    )

    # Update Celery configuration from Flask app
    celery.conf.update(app.config)

    # Configure task routes
    celery.conf.task_routes = {
        'processtask': {'queue': 'high_priority'},
    }

    # Configure task time limits
    celery.conf.task_time_limit = 300
    celery.conf.task_soft_time_limit = 240

    # Configure result expiration
    celery.conf.result_expires = 3600

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

# Configure logging
def configure_logging():
    """Configure logging settings"""
    log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    if os.getenv('FLASK_ENV') == 'development':
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s - %(filename)s:%(lineno)d'

    handlers = [logging.StreamHandler(sys.stdout)]

    if os.getenv('LOG_FILE'):
        handlers.append(logging.FileHandler(os.getenv('LOG_FILE')))

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
            environment=os.getenv('FLASK_ENV', 'development')
        )

# Configure logging
configure_logging()

# Initialize components
try:
    cache_manager = CacheManager()
    claude_api = ClaudeAPI()
    news_api = NewsAPI()  # Используем только NewsAPI
except Exception as e:
    logger.error(f"Failed to initialize components: {str(e)}")
    raise

# Test data
daily_buzz = {
    "article": {
        "title": "Today's featured analysis",
        "source": "Media Analysis",
        "short_summary": "Analysis of current events...",
        "analysis": {
            "credibility_score": {"score": 0.85},
            "topics": ["News", "Analysis"],
            "summary": "Detailed analysis...",
            "perspectives": {
                "western": {"summary": "Western perspective...", "credibility": "High"},
                "neutral": {"summary": "Neutral analysis...", "credibility": "High"}
            }
        }
    }
}

source_credibility_data = {
    "sources": ["BBC", "Reuters", "CNN"],
    "credibility_scores": [0.92, 0.88, 0.75]
}

# Error handlers
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500

# Routes
@app.route('/')
def index():
    """Main page of the application"""
    return render_template('index.html')

@app.route('/faq')
@cache.cached(timeout=3600)
def faq():
    return render_template('faq.html')

@app.route('/feedback')
def feedback():
    return render_template('feedback.html')

@app.route('/privacy')
def privacy():
    return render_template('privacy.html')

@app.route('/terms')
def terms():
    return render_template('terms.html')

@app.route('/source-credibility-chart')
def get_source_credibility_chart():
    """Returns data for source credibility chart"""
    try:
        return jsonify(source_credibility_data)
    except Exception as e:
        logger.error(f"Error getting source credibility chart: {str(e)}")
        return jsonify({
            "sources": ["BBC", "Reuters", "CNN"],
            "credibility_scores": [0.9, 0.85, 0.75]
        }), 500

@app.route('/analysis-history')
def get_analysis_history():
    """Returns analysis history"""
    try:
        with history_lock:
            return jsonify({"history": analysis_history})
    except Exception as e:
        logger.error(f"Error getting analysis history: {str(e)}")
        return jsonify({"history": []}), 500

@celery.task(bind=True)
def analyze_article_async(self, url_or_text: str) -> Dict[str, Any]:
    """Asynchronous task for article analysis"""
    try:
        # Check cache
        cached_result = cache_manager.get_cached_article_analysis(url_or_text)
        if cached_result:
            return {"status": "success", "result": cached_result, "cached": True}

        self.update_state(state='PROGRESS', meta={'progress': 10, 'message': 'Starting analysis'})

        # Extract content
        if url_or_text.startswith(('http://', 'https://')):
            content, source, title, error = extract_text_from_url(url_or_text)
            if error:
                return {"status": "error", "message": error}
            self.update_state(state='PROGRESS', meta={'progress': 30, 'message': 'Article extracted'})
        else:
            content = url_or_text
            source = 'Direct Input'
            title = 'User-provided Text'

        # Analyze with Claude API
        self.update_state(state='PROGRESS', meta={'progress': 50, 'message': 'Analyzing with Claude API'})
        analysis = claude_api.analyze_article(content, source)

        # Build query for similar articles
        query = build_newsapi_query(analysis)

        # Get similar articles
        similar_articles = []
        if query:
            self.update_state(state='PROGRESS', meta={'progress': 70, 'message': 'Finding similar articles'})
            similar_articles = news_api.get_everything(query=query, page_size=5)

        # Determine credibility level
        credibility_level = claude_api.determine_credibility_level(analysis.get('credibility_score', {}).get('score', 0.6))

        # Build result
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

        # Cache result
        cache_manager.cache_article_analysis(url_or_text, result)

        # Update analysis history
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
        logger.error(f"Error in async article analysis: {str(e)}", exc_info=True)
        return {"status": "error", "message": str(e)}

def build_newsapi_query(analysis: dict) -> str:
    """Builds query for NewsAPI based on article analysis"""
    try:
        query_parts = []

        # Add topics
        topics = analysis.get('content_analysis', {}).get('main_topics', [])
        if isinstance(topics, list):
            query_parts.extend([t['name'] if isinstance(t, dict) else t for t in topics[:3]])

        # Add important entities
        key_arguments = analysis.get('content_analysis', {}).get('key_arguments', [])
        if isinstance(key_arguments, list):
            important_arguments = [arg['argument'] for arg in key_arguments if isinstance(arg, dict)]
            query_parts.extend(important_arguments[:3])

        return ' OR '.join(query_parts) if query_parts else "technology"

    except Exception as e:
        logger.error(f"Error building NewsAPI query: {str(e)}")
        return "technology"

def extract_text_from_url(url: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Extracts text from URL with improved error handling"""
    try:
        parsed = urlparse(url)
        if not all([parsed.scheme, parsed.netloc]):
            return None, None, None, "Invalid URL format"

        clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"

        # Skip video sites
        if any(domain in parsed.netloc for domain in ['youtube.com', 'vimeo.com', 'twitch.tv', 'tiktok.com']):
            return None, parsed.netloc.replace('www.', ''), "Video content detected", None

        # Use newspaper for content extraction
        config = Config()
        config.browser_user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36'
        config.request_timeout = 30

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
            logger.warning(f"Newspaper failed to process {url}: {str(e)}")

        # Alternative content extraction method
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
            }

            session = requests.Session()
            retries = Retry(total=3, backoff_factor=1)
            adapter = HTTPAdapter(max_retries=retries)
            session.mount("http://", adapter)
            session.mount("https://", adapter)

            response = session.get(clean_url, headers=headers, timeout=15)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')

            # Remove unwanted elements
            for element in soup(['script', 'style', 'noscript', 'iframe', 'svg', 'nav', 'footer', 'header']):
                element.decompose()

            # Try to find main content
            main_content = soup.find('article') or \
                        soup.find('div', {'class': re.compile('article|content|main|post|entry')}) or \
                        soup.find('main')

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

            # If no main content found, get all paragraphs
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
            logger.error(f"Alternative extraction failed for {url}: {str(e)}")
            return None, parsed.netloc.replace('www.', ''), "Error occurred during extraction", str(e)

    except Exception as e:
        logger.error(f"Unexpected error extracting article from {url}: {str(e)}")
        return None, None, "Unexpected error occurred", str(e)

def determine_credibility_level(score: float) -> str:
    """Determines credibility level with improved input handling"""
    try:
        if isinstance(score, dict):
            score = score.get('score', 0.6)

        if isinstance(score, (float, int)):
            score = float(score)
        else:
            score = 0.6  # Default value

        if score >= 0.8:
            return "High"
        elif score >= 0.6:
            return "Medium"
        else:
            return "Low"

    except Exception as e:
        logger.error(f"Error determining credibility level: {str(e)}")
        return "Medium"

def determine_credibility_level_from_source(source_name: str) -> str:
    """Determines credibility level from source with improved handling"""
    try:
        source_name = source_name.lower()

        high_credibility_sources = [
            'bbc.com', 'reuters.com', 'nytimes.com', 'theguardian.com',
            'washingtonpost.com', 'wsj.com', 'ft.com', 'economist.com'
        ]

        medium_credibility_sources = [
            'cnn.com', 'foxnews.com', 'usatoday.com', 'washingtonpost.com',
            'npr.org', 'aljazeera.com', 'theindependent.co.uk'
        ]

        low_credibility_sources = [
            'dailymail.co.uk', 'breitbart.com', 'infowars.com',
            'thesun.co.uk', 'rt.com', 'sputniknews.com'
        ]

        domain = source_name
        if source_name.startswith(('http://', 'https://')):
            domain = urlparse(source_name).netloc

        if any(hcs in domain for hcs in high_credibility_sources):
            return "High"
        elif any(lcs in domain for lcs in low_credibility_sources):
            return "Low"
        elif any(mcs in domain for mcs in medium_credibility_sources):
            return "Medium"
        elif domain.endswith('.gov') or domain.endswith('.edu'):
            return "High"
        else:
            return "Medium"

    except Exception as e:
        logger.error(f"Error determining source credibility: {str(e)}")
        return "Medium"

# Routes for analysis
@app.route('/start-analysis', methods=['POST', 'OPTIONS'])
def start_analysis():
    """Starts asynchronous article analysis"""
    if request.method == 'OPTIONS':
        return jsonify({}), 200

    try:
        data = request.get_json()
        if not data or 'input_text' not in data:
            return jsonify({'status': 'error', 'message': 'Input text is required'}), 400

        input_text = data['input_text'].strip()
        if not input_text:
            return jsonify({'status': 'error', 'message': 'Input text cannot be empty'}), 400

        # Additional URL validation
        if input_text.startswith(('http://', 'https://')):
            valid, error = validate_url(input_text)
            if not valid:
                return jsonify({'status': 'error', 'message': error}), 400
        elif len(input_text) < 50:
            return jsonify({'status': 'error', 'message': 'Content is too short for analysis'}), 400

        task = analyze_article_async.delay(input_text)
        return jsonify({
            'status': 'started',
            'task_id': task.id,
            'message': 'Analysis started successfully'
        })

    except Exception as e:
        logger.error(f"Error starting analysis: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': 'Failed to start analysis',
            'details': str(e)
        }), 500

@app.route('/task-status/<task_id>')
def get_task_status(task_id):
    """Checks status of asynchronous task"""
    try:
        task = analyze_article_async.AsyncResult(task_id)

        if task.state == 'PENDING':
            response = {
                'status': task.state,
                'message': 'Task not yet started',
                'progress': 0
            }
        elif task.state == 'PROGRESS':
            response = {
                'status': task.state,
                'progress': task.info.get('progress', 0),
                'message': task.info.get('message', '')
            }
        elif task.state == 'SUCCESS':
            response = {
                'status': task.state,
                'result': task.result,
                'message': 'Task completed successfully'
            }
        else:  # FAILURE or other states
            response = {
                'status': task.state,
                'message': str(task.info) if task.info else 'Task failed'
            }

        return jsonify(response)

    except Exception as e:
        logger.error(f"Error getting task status: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': 'Failed to get task status',
            'details': str(e)
        }), 500

@app.route('/clear-cache', methods=['POST'])
def clear_cache():
    """Clears analysis cache"""
    try:
        cache_manager.clear_all_caches()
        return jsonify({
            'status': 'success',
            'message': 'Cache cleared successfully'
        })
    except Exception as e:
        logger.error(f"Error clearing cache: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': 'Failed to clear cache',
            'details': str(e)
        }), 500

@app.route('/health')
def health_check():
    """Application health check"""
    try:
        api_status = {
            'news_api': 'unavailable',
            'claude_api': 'unavailable',
            'redis': 'unavailable'
        }

        # Check Redis connection
        try:
            from redis import Redis
            redis_client = Redis.from_url(os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0'))
            if redis_client.ping():
                api_status['redis'] = 'operational'
        except Exception as e:
            logger.warning(f"Redis health check failed: {str(e)}")
            api_status['redis'] = f'unavailable: {str(e)}'

        # Check NewsAPI
        try:
            test_result = news_api.get_everything("test", page_size=1)
            if test_result:
                api_status['news_api'] = 'operational'
        except Exception as e:
            logger.warning(f"NewsAPI health check failed: {str(e)}")
            api_status['news_api'] = f'unavailable: {str(e)}'

        # Check ClaudeAPI
        try:
            if claude_api.client:
                api_status['claude_api'] = 'operational'
        except Exception as e:
            logger.warning(f"ClaudeAPI health check failed: {str(e)}")
            api_status['claude_api'] = f'unavailable: {str(e)}'

        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'api_status': api_status,
            'cache_status': 'operational' if cache else 'unavailable'
        })

    except Exception as e:
        logger.error(f"Error during health check: {str(e)}")
        return jsonify({
            'status': 'unhealthy',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

def validate_url(url: str) -> Tuple[bool, Optional[str]]:
    """Validates URL with stricter checks"""
    try:
        parsed = urlparse(url)
        if not all([parsed.scheme, parsed.netloc]):
            return False, "Invalid URL format - missing scheme or netloc"
        if parsed.scheme not in ('http', 'https'):
            return False, "Invalid URL scheme - must be http or https"
        if not parsed.netloc.replace('.', '').isalnum():
            return False, "Invalid domain name in URL"
        return True, None
    except Exception as e:
        return False, f"URL validation error: {str(e)}"

def check_newsapi_connection():
    """Check NewsAPI connection at startup"""
    try:
        test_query = "technology"
        articles = news_api.get_everything(query=test_query, page_size=1)
        if articles:
            logger.info("Successfully connected to NewsAPI during startup check")
            return True
        else:
            logger.warning("NewsAPI connection test returned empty results")
            return False
    except Exception as e:
        logger.error(f"NewsAPI connection check failed: {str(e)}")
        return False

if __name__ == '__main__':
    # Check NewsAPI connection at startup
    if not check_newsapi_connection():
        logger.warning("Could not connect to NewsAPI - will use fallback data")

    # Run application
    app.run(
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 5000)),
        threaded=True,
        debug=os.getenv('FLASK_ENV') == 'development'
    )
