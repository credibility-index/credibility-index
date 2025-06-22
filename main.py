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

# Добавляем текущую директорию в путь Python
sys.path.append(str(Path(__file__).parent))

# Импортируем наши модули
from claude_api import ClaudeAPI
from news_api import NewsAPI
from cache import CacheManager

# Инициализация Sentry
if os.getenv('SENTRY_DSN'):
    sentry_sdk.init(
        dsn=os.getenv('SENTRY_DSN'),
        integrations=[FlaskIntegration()],
        traces_sample_rate=1.0,
        environment=os.getenv('FLASK_ENV', 'development')
    )

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
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Декоратор для rate limiting
def rate_limit(max_per_minute=60):
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            # Получаем IP клиента
            client_ip = request.remote_addr

            # Используем локальное хранилище для хранения информации о запросах
            if not hasattr(app, 'rate_limit_store'):
                app.rate_limit_store = {}

            # Ключ для хранения количества запросов
            key = f"rate_limit:{client_ip}:{request.path}"

            # Получаем текущее количество запросов
            current = app.rate_limit_store.get(key, 0)

            # Проверяем, не превышен ли лимит
            if current >= max_per_minute:
                return jsonify({
                    'status': 'error',
                    'message': 'Rate limit exceeded. Please try again later.'
                }), 429

            # Увеличиваем счётчик
            app.rate_limit_store[key] = current + 1

            # Сбрасываем счётчик через 60 секунд
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

# Встроенная CSRF защита
@app.before_request
def csrf_protect():
    if request.method == "POST":
        token = session.pop('_csrf_token', None)
        if not token or token != request.form.get('_csrf_token'):
            return jsonify({'status': 'error', 'message': 'CSRF token missing or invalid'}), 403

@app.after_request
def add_csrf_token(response):
    if request.endpoint in app.view_functions and request.method == "GET":
        response.set_cookie('_csrf_token', app.config['SECRET_KEY'])
    return response

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
    """Асинхронная задача для анализа статьи"""
    try:
        # Проверяем кэш
        cached_result = cache.get_cached_article_analysis(url_or_text)
        if cached_result:
            return {"status": "success", "result": cached_result, "cached": True}

        # Обновляем прогресс
        self.update_state(state='PROGRESS', meta={'progress': 10, 'message': 'Starting analysis'})

        # Логика анализа
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
        analysis = claude_api.analyze_article(content, source)
        self.update_state(state='PROGRESS', meta={'progress': 70, 'message': 'Analysis completed'})

        # Получаем похожие статьи
        topics = [t['name'] if isinstance(t, dict) else t for t in analysis.get('topics', [])]
        similar_articles = []

        if topics:
            query = ' OR '.join(topics[:3])
            similar_articles = news_api.get_everything(query=query, page_size=5) or []

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
            'similar_articles': similar_articles
        }

        # Кэшируем результат
        cache.cache_article_analysis(url_or_text, result)

        self.update_state(state='PROGRESS', meta={'progress': 100, 'message': 'Completed'})
        return {"status": "success", "result": result}

    except Exception as e:
        logger.error(f"Error in async article analysis: {str(e)}", exc_info=True)
        return {"status": "error", "message": str(e)}

@app.route('/')
def index():
    """Главная страница приложения"""
    csrf_token = app.config['SECRET_KEY']
    return render_template('index.html', csrf_token=csrf_token)

@app.route('/daily-buzz')
def get_daily_buzz():
    """Возвращает статью дня"""
    try:
        # Проверяем кэш
        cached_buzz = cache.get_cached_buzz_analysis()
        if cached_buzz:
            return jsonify({"article": cached_buzz})

        # Получаем новый анализ
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

        # Валидация входных данных
        try:
            if input_text.startswith(('http://', 'https://')):
                if not re.match(r'^https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', input_text):
                    raise ValueError('Invalid URL format')
            else:
                if len(input_text) < 50:
                    raise ValueError('Content is too short for analysis')
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)}), 400

        # Запускаем асинхронную задачу
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

        # Валидация входных данных
        try:
            if input_text.startswith(('http://', 'https://')):
                if not re.match(r'^https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', input_text):
                    raise ValueError('Invalid URL format')
            else:
                if len(input_text) < 50:
                    raise ValueError('Content is too short for analysis')
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)}), 400

        # Проверяем кэш
        cached_result = cache.get_cached_article_analysis(input_text)
        if cached_result:
            return jsonify({
                'status': 'success',
                'article': cached_result['article'],
                'similar_articles': cached_result.get('similar_articles', [])
            })

        # Анализ статьи
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

        # Анализ через Claude API
        analysis = claude_api.analyze_article(content, source)
        credibility_level = determine_credibility_level(analysis.get('credibility_score', {}).get('score', 0.6))

        # Получаем похожие статьи
        topics = [t['name'] if isinstance(t, dict) else t for t in analysis.get('topics', [])]
        similar_articles = []

        if topics:
            query = ' OR '.join(topics[:3])
            similar_articles = news_api.get_everything(query=query, page_size=5) or []

        # Формируем результат
        result = {
            'title': title,
            'source': source,
            'url': input_text if input_text.startswith(('http://', 'https://')) else None,
            'short_summary': content[:200] + '...' if len(content) > 200 else content,
            'analysis': analysis,
            'credibility_level': credibility_level,
            'similar_articles': similar_articles
        }

        # Кэшируем результат
        cache.cache_article_analysis(input_text, result)

        # Сохраняем в историю
        analysis_history.insert(0, {
            "title": title,
            "source": source,
            "url": input_text if input_text.startswith(('http://', 'https://')) else None,
            "summary": content[:200] + '...' if len(content) > 200 else content,
            "credibility": credibility_level
        })
        analysis_history = analysis_history[:10]  # Оставляем только последние 10

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

@app.route('/feedback')
def feedback():
    """Страница обратной связи"""
    csrf_token = app.config['SECRET_KEY']
    return render_template('feedback.html', csrf_token=csrf_token)

@app.route('/privacy')
def privacy():
    """Страница политики конфиденциальности"""
    return render_template('privacy.html')

@app.route('/terms')
def terms():
    """Страница условий использования"""
    return render_template('terms.html')

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                             'favicon.ico', mimetype='image/vnd.microsoft.icon')

def extract_text_from_url(url: str) -> tuple:
    """Извлекает текст из URL с улучшенной обработкой ошибок"""
    try:
        parsed = urlparse(url)
        if not all([parsed.scheme, parsed.netloc]):
            return None, None, None, "Invalid URL format"

        clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"

        # Проверяем, не является ли это видео-контентом
        if any(domain in parsed.netloc for domain in ['youtube.com', 'vimeo.com', 'twitch.tv']):
            return None, parsed.netloc.replace('www.', ''), "Video content detected", None

        # Пытаемся извлечь текст с помощью newspaper
        try:
            article = Article(clean_url, config=config)
            article.download()
            article.parse()

            if article.text and len(article.text.strip()) >= 100:
                return (article.text.strip(),
                        parsed.netloc.replace('www.', ''),
                        article.title.strip() if article.title else "No title available",
                        None)
        except Exception as e:
            logger.warning(f"Newspaper failed to process {url}: {str(e)}")

        # Альтернативный метод извлечения
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            }

            response = session.get(clean_url, headers=headers, timeout=15)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')

            # Удаляем ненужные элементы
            for element in soup(['script', 'style', 'noscript', 'iframe', 'svg', 'nav', 'footer', 'header']):
                element.decompose()

            # Ищем основной контент
            main_content = soup.find('article') or soup.find('div', {'class': re.compile('article|content|main')})

            if main_content:
                text = ' '.join([p.get_text() for p in main_content.find_all('p')])
                if len(text.strip()) >= 100:
                    return (text.strip(),
                            parsed.netloc.replace('www.', ''),
                            soup.title.string.strip() if soup.title else "No title available",
                            None)

            return None, parsed.netloc.replace('www.', ''), "Failed to extract content", "Content extraction failed"

        except Exception as e:
            logger.error(f"Alternative extraction failed for {url}: {str(e)}")
            return None, parsed.netloc.replace('www.', ''), "Error occurred", str(e)

    except Exception as e:
        logger.error(f"Unexpected error extracting article from {url}: {str(e)}")
        return None, parsed.netloc.replace('www.', ''), "Error occurred", str(e)

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

def get_similar_articles(topics: list) -> list:
    """Возвращает похожие статьи с использованием NewsAPI"""
    if not topics:
        return []

    try:
        query = ' OR '.join([str(t) for t in topics[:3]])  # Берем первые 3 темы
        articles = news_api.get_everything(
            query=query,
            page_size=5,
            sort_by='publishedAt'
        ) or []

        return [
            {
                "title": article['title'],
                "source": article['source']['name'],
                "url": article['url'],
                "summary": article['description'],
                "credibility": determine_credibility_level_from_source(article['source']['name'])
            }
            for article in articles
        ]
    except Exception as e:
        logger.error(f"Error getting similar articles: {str(e)}")
        return []

def determine_credibility_level_from_source(source_name: str) -> str:
    """Определяет уровень достоверности на основе источника"""
    source_name = source_name.lower()

    high_credibility_sources = [
        'bbc', 'reuters', 'associated press', 'the new york times',
        'the guardian', 'the wall street journal', 'bloomberg'
    ]

    medium_credibility_sources = [
        'cnn', 'fox news', 'usa today', 'the washington post',
        'npr', 'al jazeera', 'the independent'
    ]

    if any(source in source_name for source in high_credibility_sources):
        return "High"
    elif any(source in source_name for source in medium_credibility_sources):
        return "Medium"
    else:
        return "Low"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
