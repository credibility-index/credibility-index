import os
import sys
import logging
from pathlib import Path
import re
import json
from datetime import datetime, timedelta
from urllib.parse import urlparse
from flask import Flask, request, jsonify, render_template, send_from_directory, session
from flask_cors import CORS
import anthropic
from newspaper import Article, Config
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from werkzeug.middleware.proxy_fix import ProxyFix
import sentry_sdk
from sentry_sdk.integrations.flask import FlaskIntegration
from celery import Celery
from pydantic import BaseModel, ValidationError, HttpUrl
from typing import Optional, Dict, Any, List
from functools import wraps
import time
import requests

# Инициализация Sentry
if os.getenv('SENTRY_DSN'):
    sentry_sdk.init(
        dsn=os.getenv('SENTRY_DSN'),
        integrations=[FlaskIntegration()],
        traces_sample_rate=1.0,
        environment=os.getenv('FLASK_ENV', 'development')
    )

# Класс NewsAPI
class NewsAPI:
    def __init__(self):
        self.api_key = os.getenv('NEWS_API_KEY')
        self.endpoint = os.getenv('NEWS_ENDPOINT', 'https://newsapi.org/v2')

        if not self.api_key:
            logging.error("NEWS_API_KEY is not set")
            raise ValueError("NEWS_API_KEY environment variable is not set")

    def get_everything(self, query: str, page_size: int = 5, sort_by: str = 'publishedAt',
                      from_param: Optional[str] = None, to: Optional[str] = None) -> Optional[List[Dict[str, Any]]]:
        """
        Получает статьи по запросу из NewsAPI.

        Args:
            query: Запрос для поиска статей
            page_size: Количество статей на странице
            sort_by: Параметр сортировки
            from_param: Дата начала в формате YYYY-MM-DD
            to: Дата окончания в формате YYYY-MM-DD

        Returns:
            Список статей или None в случае ошибки
        """
        url = f"{self.endpoint}/everything"
        params = {
            'q': query,
            'pageSize': page_size,
            'sortBy': sort_by,
            'apiKey': self.api_key
        }

        if from_param:
            params['from'] = from_param
        if to:
            params['to'] = to

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if data.get('status') != 'ok':
                logging.error(f"NewsAPI returned error: {data.get('message', 'Unknown error')}")
                return None

            return data.get('articles', [])

        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching news from NewsAPI: {str(e)}", exc_info=True)
            return None
        except Exception as e:
            logging.error(f"Unexpected error in NewsAPI: {str(e)}", exc_info=True)
            return None

# Инициализация приложения
app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'your-secret-key-here')
app.config['CELERY_BROKER_URL'] = os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0')
app.config['CELERY_RESULT_BACKEND'] = os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')

# Настройка CORS
CORS(app, resources={
    r"/*": {
        "origins": os.getenv('CORS_ORIGINS', '*').split(','),
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "X-CSRFToken", "X-Requested-With"]
    }
})

# Добавляем текущую директорию в путь Python
sys.path.append(str(Path(__file__).parent))

# Декоратор для rate limiting
def rate_limit(max_per_minute=60):
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            client_ip = request.remote_addr
            if not hasattr(app, 'rate_limit_store'):
                app.rate_limit_store = {}
            key = f"rate_limit:{client_ip}:{request.path}"
            current = app.rate_limit_store.get(key, 0)
            if current >= max_per_minute:
                return jsonify({
                    'status': 'error',
                    'message': 'Rate limit exceeded. Please try again later.'
                }), 429
            app.rate_limit_store[key] = current + 1
            if not hasattr(app, 'rate_limit_cleanup'):
                app.rate_limit_cleanup = True
                def cleanup():
                    time.sleep(60)
                    app.rate_limit_store = {}
                import threading
                threading.Thread(target=cleanup, daemon=True).start()
            return f(*args, **kwargs)
        return wrapper
    return decorator

# Настройка для работы за обратным прокси
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

# Инициализация Celery
celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Конфигурация библиотеки newspaper
config = Config()
config.browser_user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36'
config.request_timeout = 30

# Инициализация компонентов
from cache import CacheManager
from claude_api import ClaudeAPI

cache = CacheManager()
claude_api = ClaudeAPI()
news_api = NewsAPI()

# Инициализация клиента Anthropic
try:
    anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
    if not anthropic_api_key:
        raise ValueError("ANTHROPIC_API_KEY is not set")
    anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)
except Exception as e:
    logger.error(f"Failed to initialize Anthropic client: {str(e)}")
    anthropic_client = None

# Тестовые данные
daily_buzz = {
    "article": {
        "title": "Today's featured analysis: Israel-Iran relations",
        "source": "Media Analysis",
        "short_summary": "Analysis of current Israel-Iran relations and recent developments...",
        "analysis": {
            "credibility_score": {"score": 0.85},
            "topics": ["Israel", "Iran", "Middle East"],
            "summary": "Detailed analysis of current situation...",
            "perspectives": {
                "western": {"summary": "Western perspective...", "credibility": "High"},
                "iranian": {"summary": "Iranian perspective...", "credibility": "Medium"},
                "israeli": {"summary": "Israeli perspective...", "credibility": "High"},
                "neutral": {"summary": "Neutral analysis...", "credibility": "High"}
            }
        }
    }
}

source_credibility_data = {
    "sources": ["BBC", "Reuters", "CNN", "The Guardian", "Fox News"],
    "credibility_scores": [0.92, 0.88, 0.75, 0.85, 0.65]
}

analysis_history = []

# Celery задача для анализа статьи
@celery.task(bind=True)
def analyze_article_async(self, url_or_text: str):
    """Асинхронная задача для анализа статьи с улучшенной логикой"""
    try:
        # Проверяем кэш
        cached_result = cache.get_cached_article_analysis(url_or_text)
        if cached_result:
            return {"status": "success", "result": cached_result, "cached": True}

        # Обновляем прогресс
        self.update_state(state='PROGRESS', meta={'progress': 10, 'message': 'Starting analysis'})

        # Извлекаем контент
        if url_or_text.startswith(('http://', 'https://')):
            content, source, title, error = extract_text_from_url(url_or_text)
            if error:
                return {"status": "error", "message": error}
            self.update_state(state='PROGRESS', meta={'progress': 30, 'message': 'Article extracted'})
        else:
            content = url_or_text
            source = 'Direct Input'
            title = 'User-provided Text'

        # Анализ через Claude API
        self.update_state(state='PROGRESS', meta={'progress': 40, 'message': 'Analyzing with Claude API'})
        analysis = claude_api.analyze_article(content, source)

        # Извлекаем ключевые параметры для поиска похожих статей
        topics = [t['name'] if isinstance(t, dict) else t for t in analysis.get('topics', [])]
        dates = analysis.get('dates', [])
        entities = analysis.get('entities', [])

        # Формируем запрос для поиска похожих статей
        query_parts = []

        # Добавляем темы
        if topics:
            query_parts.extend(topics[:3])  # Берем первые 3 темы

        # Добавляем важные сущности
        if entities:
            important_entities = [e for e in entities if e.get('importance', 0) > 0.7]
            query_parts.extend([e['name'] for e in important_entities[:3]])

        # Добавляем даты, если они есть
        date_ranges = []
        if dates:
            for date_info in dates[:2]:  # Берем первые 2 даты
                if isinstance(date_info, dict):
                    date = date_info.get('date')
                    if date:
                        try:
                            parsed_date = datetime.strptime(date, '%Y-%m-%d')
                            date_ranges.append(parsed_date.strftime('%Y-%m-%d'))
                        except ValueError:
                            continue

        # Формируем окончательный запрос
        query = ' OR '.join(query_parts) if query_parts else None

        # Получаем похожие статьи
        similar_articles = []
        if query:
            self.update_state(state='PROGRESS', meta={'progress': 60, 'message': 'Finding similar articles'})

            # Параметры для запроса
            params = {
                'page_size': 5,
                'sort_by': 'publishedAt'
            }

            # Добавляем даты в параметры, если они есть
            if len(date_ranges) == 1:
                params['from_param'] = date_ranges[0]
            elif len(date_ranges) == 2:
                params['from_param'] = date_ranges[0]
                params['to'] = date_ranges[1]

            similar_articles = news_api.get_everything(query=query, **params) or []

        # Определяем уровень достоверности
        credibility_level = determine_credibility_level(analysis.get('credibility_score', {}).get('score', 0.6))

        # Формируем результат
        result = {
            'title': title,
            'source': source,
            'url': url_or_text if url_or_text.startswith(('http://', 'https://')) else None,
            'short_summary': content[:200] + '...' if len(content) > 200 else content,
            'analysis': analysis,
            'credibility_level': credibility_level,
            'similar_articles': similar_articles,
            'search_query': query  # Добавляем использованный запрос для отладки
        }

        # Кэшируем результат
        cache.cache_article_analysis(url_or_text, result)

        self.update_state(state='PROGRESS', meta={'progress': 100, 'message': 'Completed'})
        return {"status": "success", "result": result}

    except Exception as e:
        logger.error(f"Error in async article analysis: {str(e)}", exc_info=True)
        return {"status": "error", "message": str(e)}

def build_newsapi_query(analysis: dict) -> Optional[str]:
    """
    Строит запрос для NewsAPI на основе анализа статьи.
    Args:
        analysis: Результат анализа статьи от Claude API
    Returns:
        Строка запроса для NewsAPI или None, если не удалось построить запрос
    """
    query_parts = []

    # Добавляем темы
    topics = [t['name'] if isinstance(t, dict) else t for t in analysis.get('topics', [])]
    if topics:
        query_parts.extend(topics[:3])  # Берем первые 3 темы

    # Добавляем важные сущности
    entities = analysis.get('entities', [])
    if entities:
        important_entities = [e for e in entities if e.get('importance', 0) > 0.7]
        query_parts.extend([e['name'] for e in important_entities[:3]])

    return ' OR '.join(query_parts) if query_parts else None

# Остальной код остается без изменений
@app.route('/')
def index():
    """Главная страница приложения"""
    return render_template('index.html')

@app.route('/daily-buzz')
def get_daily_buzz():
    """Возвращает статью дня"""
    try:
        cached_buzz = cache.get_cached_buzz_analysis()
        if cached_buzz:
            return jsonify({"article": cached_buzz})
        buzz_analysis = claude_api.get_buzz_analysis()
        cache.cache_buzz_analysis(buzz_analysis)
        return jsonify({"article": buzz_analysis})
    except Exception as e:
        logger.error(f"Error getting daily buzz: {str(e)}")
        return jsonify(daily_buzz)

@app.route('/source-credibility-chart')
def get_source_credibility_chart():
    """Возвращает данные для графика достоверности источников"""
    return jsonify(source_credibility_data)

@app.route('/analysis-history')
def get_analysis_history():
    """Возвращает историю анализов"""
    return jsonify({"history": analysis_history})

@app.route('/start-analysis', methods=['POST'])
@rate_limit(max_per_minute=5)
def start_analysis():
    """Начинает асинхронный анализ статьи"""
    try:
        data = request.get_json()
        if not data or 'input_text' not in data:
            return jsonify({'status': 'error', 'message': 'Input text is required'}), 400
        input_text = data['input_text'].strip()

        try:
            if input_text.startswith(('http://', 'https://')):
                if not re.match(r'^https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', input_text):
                    raise ValueError('Invalid URL format')
            else:
                if len(input_text) < 50:
                    raise ValueError('Content is too short for analysis')
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)}), 400

        task = analyze_article_async.delay(input_text)
        return jsonify({'status': 'started', 'task_id': task.id})
    except Exception as e:
        logger.error(f"Error starting analysis: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/task-status/<task_id>')
def get_task_status(task_id):
    """Проверяет статус асинхронной задачи"""
    task = analyze_article_async.AsyncResult(task_id)
    if task.state == 'PENDING':
        response = {
            'status': task.state,
            'message': 'Task not yet started'
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
            'result': task.result
        }
    else:
        response = {
            'status': task.state,
            'message': str(task.info) if task.info else 'Task failed'
        }
    return jsonify(response)

@app.route('/analyze', methods=['POST'])
@rate_limit(max_per_minute=5)
def analyze_article():
    """Анализирует статью синхронно"""
    try:
        data = request.get_json()
        if not data or 'input_text' not in data:
            return jsonify({'status': 'error', 'message': 'Input text is required'}), 400
        input_text = data['input_text'].strip()

        try:
            if input_text.startswith(('http://', 'https://')):
                if not re.match(r'^https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', input_text):
                    raise ValueError('Invalid URL format')
            else:
                if len(input_text) < 50:
                    raise ValueError('Content is too short for analysis')
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)}), 400

        cached_result = cache.get_cached_article_analysis(input_text)
        if cached_result:
            return jsonify({
                'status': 'success',
                'article': cached_result['article'],
                'similar_articles': cached_result.get('similar_articles', [])
            })

        if input_text.startswith(('http://', 'https://')):
            content, source, title, error = extract_text_from_url(input_text)
            if error:
                return jsonify({
                    'status': 'error',
                    'message': error,
                    'source': source,
                    'title': title
                }), 400
        else:
            content = input_text
            source = 'Direct Input'
            title = 'User-provided Text'

        analysis = claude_api.analyze_article(content, source)
        credibility_level = determine_credibility_level(analysis.get('credibility_score', {}).get('score', 0.6))

        topics = [t['name'] if isinstance(t, dict) else t for t in analysis.get('topics', [])]
        similar_articles = []
        if topics:
            query = build_newsapi_query(analysis)
            if query:
                similar_articles = news_api.get_everything(query=query, page_size=5) or []

        result = {
            'title': title,
            'source': source,
            'url': input_text if input_text.startswith(('http://', 'https://')) else None,
            'short_summary': content[:200] + '...' if len(content) > 200 else content,
            'analysis': analysis,
            'credibility_level': credibility_level,
            'similar_articles': similar_articles
        }

        cache.cache_article_analysis(input_text, result)

        analysis_history.insert(0, {
            "title": title,
            "source": source,
            "url": input_text if input_text.startswith(('http://', 'https://')) else None,
            "summary": content[:200] + '...' if len(content) > 200 else content,
            "credibility": credibility_level
        })
        analysis_history = analysis_history[:10]

        return jsonify({
            'status': 'success',
            'article': result
        })
    except Exception as e:
        logger.error(f"Error analyzing article: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': 'An unexpected error occurred during analysis',
            'details': str(e)
        }), 500

# Вспомогательные функции
def extract_text_from_url(url: str):
    """Извлекает текст из URL"""
    try:
        parsed = urlparse(url)
        if not all([parsed.scheme, parsed.netloc]):
            return None, None, None, "Invalid URL format"

        clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"

        if any(domain in parsed.netloc for domain in ['youtube.com', 'vimeo.com', 'twitch.tv', 'tiktok.com']):
            return None, parsed.netloc.replace('www.', ''), "Video content detected", None

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

        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
            }
            session = requests.Session()
            retries = Retry(total=3, backoff_factor=1)
            session.mount("https://", HTTPAdapter(max_retries=retries))
            response = session.get(clean_url, headers=headers, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            for element in soup(['script', 'style', 'noscript', 'iframe', 'svg', 'nav', 'footer', 'header']):
                element.decompose()

            main_content = soup.find('article') or soup.find('div', {'class': re.compile('article|content|main|post|entry')}) or soup.find('main')

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

            text = ' '.join([p.get_text().strip() for p in soup.find_all('p') if p.get_text().strip()])
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
    """Определяет уровень достоверности"""
    if isinstance(score, dict):
        score = score.get('score', 0.6)
    if score >= 0.8:
        return "High"
    elif score >= 0.6:
        return "Medium"
    else:
        return "Low"

def determine_credibility_level_from_source(source_name: str) -> str:
    """Определяет уровень достоверности на основе источника"""
    source_name = source_name.lower()
    high_credibility_sources = ['bbc', 'reuters', 'associated press', 'the new york times', 'the guardian']
    medium_credibility_sources = ['cnn', 'fox news', 'usa today', 'the washington post', 'npr']

    if any(source in source_name for source in high_credibility_sources):
        return "High"
    elif any(source in source_name for source in medium_credibility_sources):
        return "Medium"
    else:
        return "Low"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
