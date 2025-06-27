import os
import sys
import logging
import time
from datetime import datetime
from urllib.parse import urlparse
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from flask_talisman import Talisman
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_caching import Cache
from redis import Redis
from redis.exceptions import ConnectionError, TimeoutError
import requests
import threading
from typing import Optional, List, Dict, Any, Tuple
from newspaper import Article, Config
from bs4 import BeautifulSoup
from claude_api import ClaudeAPI  # Импорт вашего существующего модуля


# Инициализация Flask приложения
app = Flask(__name__,
            static_folder='static',
            static_url_path='/static',
            template_folder='templates')

# Конфигурация приложения
app.config.update(
    SECRET_KEY=os.getenv('SECRET_KEY', 'your-secret-key-here'),
    PERMANENT_SESSION_LIFETIME=3600,
    TEMPLATES_AUTO_RELOAD=True,
    SEND_FILE_MAX_AGE_DEFAULT=86400
)

# Конфигурация кэша с Redis
app.config['CACHE_TYPE'] = 'RedisCache'
app.config['CACHE_REDIS_URL'] = os.getenv('REDIS_URL', 'redis://default:PRxrZBrAMdzypdQxVauTIWOHhFXksxqY@redis:6379/0')
app.config['CACHE_DEFAULT_TIMEOUT'] = 3600
app.config['CACHE_KEY_PREFIX'] = 'media_credibility_'
cache = Cache(app)

# Конфигурация CORS
CORS(app, resources={
    r"/*": {
        "origins": [
            "https://your-app-name.up.railway.app",
            "http://localhost:*"
        ]
    }
})

# Конфигурация безопасности
Talisman(app,
    content_security_policy={
        'default-src': "'self'",
        'script-src': ["'self'", "'unsafe-inline'", "cdn.jsdelivr.net"],
        'style-src': ["'self'", "'unsafe-inline'", "fonts.googleapis.com"],
        'img-src': ["'self'", "data:", "https://*.googleapis.com"],
        'font-src': ["'self'", "fonts.gstatic.com"],
        'connect-src': ["'self'", "https://api.newsapi.org", "https://api.anthropic.com"]
    },
    force_https=os.getenv('FLASK_ENV') == 'production',
    strict_transport_security=os.getenv('FLASK_ENV') == 'production'
)

# Конфигурация ограничения запросов
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri=os.getenv('REDIS_URL', 'redis://default:PRxrZBrAMdzypdQxVauTIWOHhFXksxqY@redis:6379/1'),
    strategy="fixed-window"
)

# Конфигурация логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

def create_redis_connection():
    """Создание подключения к Redis с логикой повторных попыток"""
    redis_url = os.getenv('REDIS_URL', 'redis://default:PRxrZBrAMdzypdQxVauTIWOHhFXksxqY@redis:6379/0')

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
                health_check_interval=30
            )

            if redis_client.ping():
                logger.info("Successfully connected to Redis")
                return redis_client

        except (ConnectionError, TimeoutError) as e:
            if attempt == max_retries - 1:
                logger.error(f"Failed to connect to Redis after {max_retries} attempts: {str(e)}")
                return None

            wait_time = retry_delay * (2 ** attempt)
            logger.warning(f"Redis connection failed, retrying in {wait_time} seconds...")
            time.sleep(wait_time)

    return None

def check_redis_connection():
    """Проверка подключения к Redis при старте"""
    try:
        redis_client = create_redis_connection()
        if redis_client:
            return True
        return False
    except Exception as e:
        logger.error(f"Redis connection check failed: {str(e)}")
        return False

# Mock реализация NewsAPI для тестирования
class MockNewsAPI:
    def get_everything(self, query: str, page_size: int = 5):
        logger.info(f"Mock NewsAPI: Getting articles for query '{query}'")
        return [
            {
                "source": {"id": None, "name": "Mock Source"},
                "author": "Mock Author",
                "title": f"Mock Article about {query}",
                "description": f"This is a mock article about {query}",
                "url": f"https://example.com/mock-article-{query}",
                "urlToImage": None,
                "publishedAt": datetime.now().isoformat(),
                "content": f"This is a mock article content about {query}..."
            }
            for _ in range(min(page_size, 3))
        ]

    def health_check(self):
        return {
            'status': 'operational',
            'details': 'Mock NewsAPI is working',
            'response_time': '0.1s'
        }

# Инициализация компонентов
try:
    # Инициализация Redis
    if not check_redis_connection():
        logger.warning("Could not connect to Redis - some features may not work properly")
    else:
        logger.info("Successfully connected to Redis")

    # Инициализация Claude API из вашего модуля
    claude_api = ClaudeAPI(api_key=os.getenv('ANTHROPIC_API_KEY'))
    logger.info("Initialized Claude API")

    # Инициализация NewsAPI
    news_api = MockNewsAPI()
    logger.info("Initialized Mock NewsAPI")

except Exception as e:
    logger.error(f"Failed to initialize components: {str(e)}")
    raise

# Тестовые данные
source_credibility_data = {
    "sources": ["BBC", "Reuters", "CNN", "The Guardian", "New York Times"],
    "credibility_scores": [0.92, 0.88, 0.75, 0.89, 0.91]
}

# Обработчики ошибок
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500

@app.errorhandler(Exception)
def handle_exception(e):
    """Глобальный обработчик исключений"""
    logger.error(f"Unhandled exception: {str(e)}", exc_info=True)
    return jsonify({
        'status': 'error',
        'message': 'An unexpected error occurred',
        'details': str(e)
    }), 500

def analyze_article_with_claude(content: str, source: str) -> Dict[str, Any]:
    """Анализ статьи с использованием вашего модуля ClaudeAPI"""
    try:
        logger.info(f"Starting article analysis with Claude API. Content length: {len(content)}")

        # Используем ваш существующий модуль ClaudeAPI
        analysis = claude_api.analyze_article(content, source)

        logger.info("Article analysis completed successfully")
        return analysis

    except Exception as e:
        logger.error(f"Error analyzing article with Claude API: {str(e)}")
        return {
            'credibility_score': {
                'score': 0.5,
                'explanation': f'Default score due to analysis error: {str(e)}'
            },
            'content_analysis': {
                'main_topics': ['General'],
                'key_arguments': [],
                'error': str(e)
            }
        }

def build_newsapi_query(analysis: dict) -> str:
    """Создание запроса для NewsAPI на основе анализа статьи"""
    try:
        query_parts = []

        # Добавляем темы
        topics = analysis.get('content_analysis', {}).get('main_topics', [])
        if isinstance(topics, list):
            query_parts.extend([t['name'] if isinstance(t, dict) else t for t in topics[:3]])

        return ' OR '.join(query_parts) if query_parts else "technology"
    except Exception as e:
        logger.error(f"Error building NewsAPI query: {str(e)}")
        return "technology"

def extract_text_from_url(url: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Извлечение текста из URL"""
    try:
        parsed = urlparse(url)
        if not all([parsed.scheme, parsed.netloc]):
            return None, None, None, "Invalid URL format"

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
        }

        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        article = soup.find('article') or soup.find('div', {'class': 'article'})

        if article:
            return (
                ' '.join([p.get_text() for p in article.find_all('p')]),
                parsed.netloc.replace('www.', ''),
                article.find('h1').get_text() if article.find('h1') else "No title available",
                None
            )

        return None, parsed.netloc.replace('www.', ''), "No title available", "Failed to extract content"

    except Exception as e:
        return None, None, None, str(e)

@app.route('/')
@limiter.exempt
def index():
    """Главная страница приложения"""
    return render_template('index.html')

@app.route('/health')
def health_check():
    """Проверка работоспособности приложения"""
    health_status = {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'services': {}
    }

    # Проверка Redis
    try:
        redis_client = create_redis_connection()
        if redis_client:
            health_status['services']['redis'] = {
                'status': 'operational',
                'details': 'Connection successful'
            }
        else:
            health_status['services']['redis'] = {
                'status': 'unavailable',
                'details': 'Could not connect to Redis'
            }
    except Exception as e:
        health_status['services']['redis'] = {
            'status': 'unavailable',
            'details': str(e)
        }

    # Проверка Claude API
    try:
        if hasattr(claude_api, 'health_check'):
            claude_health = claude_api.health_check()
            health_status['services']['claude_api'] = {
                'status': 'operational' if claude_health.get('status') == 'operational' else 'unavailable',
                'details': claude_health.get('details', 'No details'),
                'response_time': claude_health.get('response_time', 'N/A')
            }
        else:
            health_status['services']['claude_api'] = {
                'status': 'unknown',
                'details': 'Health check method not available'
            }
    except Exception as e:
        health_status['services']['claude_api'] = {
            'status': 'unavailable',
            'details': str(e),
            'response_time': 'N/A'
        }

    # Проверка NewsAPI
    try:
        news_health = news_api.health_check()
        health_status['services']['news_api'] = {
            'status': news_health['status'],
            'details': news_health['details'],
            'response_time': news_health['response_time']
        }
    except Exception as e:
        health_status['services']['news_api'] = {
            'status': 'unavailable',
            'details': str(e),
            'response_time': 'N/A'
        }

    return jsonify(health_status)

@app.route('/start-analysis', methods=['POST'])
@limiter.limit("10 per minute")
def start_analysis():
    """Начало анализа статьи"""
    try:
        data = request.get_json()
        if not data or 'input_text' not in data:
            return jsonify({'status': 'error', 'message': 'Input text is required'}), 400

        input_text = data['input_text'].strip()
        if not input_text:
            return jsonify({'status': 'error', 'message': 'Input text cannot be empty'}), 400

        # Для прямого текстового ввода
        if not input_text.startswith(('http://', 'https://')):
            analysis = analyze_article_with_claude(input_text, 'Direct Input')
            return jsonify({
                'status': 'success',
                'result': {
                    'title': 'Direct Text Analysis',
                    'source': 'Direct Input',
                    'analysis': analysis
                }
            })

        # Для URL ввода
        content, source, title, error = extract_text_from_url(input_text)
        if error:
            return jsonify({
                'status': 'error',
                'message': error
            }), 400

        analysis = analyze_article_with_claude(content, source)
        return jsonify({
            'status': 'success',
            'result': {
                'title': title,
                'source': source,
                'analysis': analysis
            }
        })

    except Exception as e:
        logger.error(f"Error starting analysis: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': 'Failed to start analysis',
            'details': str(e)
        }), 500

if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 5000)),
        debug=os.getenv('FLASK_ENV') == 'development'
    )
