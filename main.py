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
from cache import CacheManager  # Предполагается, что этот модуль существует
from claude_api import ClaudeAPI  # Предполагается, что этот модуль существует

# Инициализация приложения Flask
app = Flask(__name__, static_folder='static', template_folder='templates')

# Настройка приложения
app.config.update(
    SECRET_KEY=os.getenv('FLASK_SECRET_KEY', 'your-secret-key-here'),
    CELERY_BROKER_URL=os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0'),
    CELERY_RESULT_BACKEND=os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')
)

# Инициализация Celery
celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)

# Глобальные переменные
analysis_history = []
history_lock = threading.Lock()

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Инициализация Sentry
if os.getenv('SENTRY_DSN'):
    sentry_sdk.init(
        dsn=os.getenv('SENTRY_DSN'),
        integrations=[FlaskIntegration()],
        traces_sample_rate=1.0,
        environment=os.getenv('FLASK_ENV', 'development')
    )

class NewsAPI:
    def __init__(self):
        self.api_key = os.getenv('NEWS_API_KEY')
        self.endpoint = os.getenv('NEWS_ENDPOINT', 'https://newsapi.org/v2')
        self.session = self._create_session()

        if not self.api_key:
            logger.error("NEWS_API_KEY is not set")
            raise ValueError("NEWS_API_KEY environment variable is not set")

    def _create_session(self):
        """Создает сессию с повторными попытками"""
        session = requests.Session()
        retries = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retries)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def get_everything(self, query: str, page_size: int = 5, sort_by: str = 'publishedAt',
                     from_param: Optional[str] = None, to: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Получает статьи по запросу из NewsAPI.
        """
        try:
            url = f"{self.endpoint}/everything"  # Исправленный URL
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

            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if data.get('status') != 'ok':
                logger.error(f"NewsAPI returned error: {data.get('message', 'Unknown error')}")
                return self._get_fallback_articles(query, page_size)

            return self._enrich_articles(data.get('articles', []))

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching news from NewsAPI: {str(e)}")
            return self._get_fallback_articles(query, page_size)
        except Exception as e:
            logger.error(f"Unexpected error in NewsAPI: {str(e)}")
            return self._get_fallback_articles(query, page_size)

    def _get_fallback_articles(self, query: str, count: int) -> List[Dict[str, Any]]:
        """Возвращает резервные статьи при ошибках API"""
        mock_articles = [
            {
                "title": f"Article about {query} (Fallback)",
                "source": {"name": "Fallback Source", "id": None},
                "url": "https://example.com/fallback1",
                "description": f"Sample article about {query}",
                "publishedAt": datetime.now().isoformat(),
                "content": f"This is a fallback article about {query}. In a real application, this would be replaced with actual news content."
            },
            {
                "title": f"Latest news on {query} (Fallback)",
                "source": {"name": "Mock News", "id": None},
                "url": "https://example.com/fallback2",
                "description": f"Recent developments in {query}",
                "publishedAt": datetime.now().isoformat(),
                "content": f"Fallback content about recent {query} developments. This mock article simulates what real news API would return."
            },
            {
                "title": f"Expert analysis of {query} (Fallback)",
                "source": {"name": "Demo News", "id": None},
                "url": "https://example.com/fallback3",
                "description": f"Expert opinions on {query}",
                "publishedAt": datetime.now().isoformat(),
                "content": f"Mock expert analysis of {query}. This fallback content demonstrates what our system would show if the news API was working."
            }
        ]
        return mock_articles[:count]

    def _enrich_articles(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Улучшает данные статей, добавляя дефолтные значения"""
        enriched = []
        for article in articles:
            enriched.append({
                'title': article.get('title', 'Untitled Article'),
                'description': article.get('description', 'No description available'),
                'url': article.get('url', '#'),
                'source': {
                    'name': article.get('source', {}).get('name', 'Unknown Source'),
                    'url': article.get('source', {}).get('url', None)
                },
                'publishedAt': article.get('publishedAt', datetime.now().isoformat()),
                'content': article.get('content', 'Full content not available')
            })
        return enriched

# Настройка CORS для всех маршрутов
CORS(app, resources={
    r"/*": {
        "origins": os.getenv('CORS_ORIGINS', '*').split(','),
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Инициализация компонентов
try:
    cache = CacheManager()
    claude_api = ClaudeAPI()
    news_api = NewsAPI()
except ImportError as e:
    logger.error(f"Failed to import modules: {str(e)}")
    raise
except Exception as e:
    logger.error(f"Failed to initialize components: {str(e)}")
    raise

# Инициализация клиента Anthropic
try:
    anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
    if not anthropic_api_key:
        logger.warning("ANTHROPIC_API_KEY is not set, some features may not work")
        anthropic_client = None
    else:
        anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)
except Exception as e:
    logger.error(f"Failed to initialize Anthropic client: {str(e)}")
    anthropic_client = None

# Тестовые данные (можно вынести в отдельный конфиг)
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

@celery.task(bind=True)
def analyze_article_async(self, url_or_text: str) -> Dict[str, Any]:
    """Асинхронная задача для анализа статьи"""
    try:
        # Проверяем кэш
        cached_result = cache.get_cached_article_analysis(url_or_text)
        if cached_result:
            return {"status": "success", "result": cached_result, "cached": True}

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
        self.update_state(state='PROGRESS', meta={'progress': 50, 'message': 'Analyzing with Claude API'})
        analysis = claude_api.analyze_article(content, source)

        # Формируем запрос для поиска похожих статей
        query = build_newsapi_query(analysis)

        # Получаем похожие статьи
        similar_articles = []
        if query:
            self.update_state(state='PROGRESS', meta={'progress': 70, 'message': 'Finding similar articles'})
            similar_articles = news_api.get_everything(query=query, page_size=5)

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
            'search_query': query
        }

        # Кэшируем результат
        cache.cache_article_analysis(url_or_text, result)

        # Обновляем историю анализа
        with history_lock:
            global analysis_history
            analysis_history.insert(0, {
                "title": title,
                "source": source,
                "url": url_or_text if url_or_text.startswith(('http://', 'https://')) else None,
                "summary": content[:200] + '...' if len(content) > 200 else content,
                "credibility": credibility_level
            })
            analysis_history = analysis_history[:10]  # Оставляем только последние 10 записей

        self.update_state(state='PROGRESS', meta={'progress': 100, 'message': 'Completed'})
        return {"status": "success", "result": result}

    except Exception as e:
        logger.error(f"Error in async article analysis: {str(e)}", exc_info=True)
        return {"status": "error", "message": str(e)}

def build_newsapi_query(analysis: dict) -> str:
    """Строит запрос для NewsAPI на основе анализа статьи"""
    try:
        query_parts = []
        # Добавляем темы
        topics = analysis.get('topics', [])
        if isinstance(topics, list):
            query_parts.extend([t['name'] if isinstance(t, dict) else t for t in topics[:3]])

        # Добавляем важные сущности (если есть в анализе)
        entities = analysis.get('entities', [])
        if isinstance(entities, list):
            important_entities = [e for e in entities if isinstance(e, dict) and e.get('importance', 0) > 0.7]
            query_parts.extend([e['name'] for e in important_entities[:3]])

        return ' OR '.join(query_parts) if query_parts else "technology"
    except Exception as e:
        logger.error(f"Error building NewsAPI query: {str(e)}")
        return "technology"

def extract_text_from_url(url: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Извлекает текст из URL с улучшенной обработкой ошибок"""
    try:
        parsed = urlparse(url)
        if not all([parsed.scheme, parsed.netloc]):
            return None, None, None, "Invalid URL format"

        clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"

        # Пропускаем видео-сайты
        if any(domain in parsed.netloc for domain in ['youtube.com', 'vimeo.com', 'twitch.tv', 'tiktok.com']):
            return None, parsed.netloc.replace('www.', ''), "Video content detected", None

        # Используем newspaper для извлечения контента
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

        # Альтернативный метод извлечения контента
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

            # Удаляем ненужные элементы
            for element in soup(['script', 'style', 'noscript', 'iframe', 'svg', 'nav', 'footer', 'header']):
                element.decompose()

            # Пытаемся найти основной контент статьи
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

            # Если не удалось найти основной контент, берем все абзацы
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
    """Определяет уровень достоверности с улучшенной обработкой входных данных"""
    try:
        if isinstance(score, dict):
            score = score.get('score', 0.6)

        if isinstance(score, (float, int)):
            score = float(score)
        else:
            score = 0.6  # Значение по умолчанию

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
    """Определяет уровень достоверности на основе источника с улучшенной обработкой"""
    try:
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
    except Exception as e:
        logger.error(f"Error determining source credibility: {str(e)}")
        return "Medium"

# Маршруты
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
        # Возвращаем тестовые данные с информацией об ошибке
        return jsonify({
            "article": {
                "title": "Daily Buzz (Fallback)",
                "source": "Fallback Data",
                "short_summary": f"Could not load daily buzz: {str(e)}. Showing fallback data.",
                "analysis": daily_buzz['article']['analysis']
            }
        })

@app.route('/source-credibility-chart')
def get_source_credibility_chart():
    """Возвращает данные для графика достоверности источников"""
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
    """Возвращает историю анализов"""
    try:
        with history_lock:
            return jsonify({"history": analysis_history})
    except Exception as e:
        logger.error(f"Error getting analysis history: {str(e)}")
        return jsonify({"history": []}), 500

@app.route('/start-analysis', methods=['POST', 'OPTIONS'])
def start_analysis():
    """Начинает асинхронный анализ статьи"""
    if request.method == 'OPTIONS':
        return jsonify({}), 200

    try:
        data = request.get_json()
        if not data or 'input_text' not in data:
            return jsonify({'status': 'error', 'message': 'Input text is required'}), 400

        input_text = data['input_text'].strip()
        if not input_text:
            return jsonify({'status': 'error', 'message': 'Input text cannot be empty'}), 400

        # Дополнительная валидация URL
        if input_text.startswith(('http://', 'https://')):
            try:
                parsed = urlparse(input_text)
                if not all([parsed.scheme, parsed.netloc]):
                    return jsonify({'status': 'error', 'message': 'Invalid URL format'}), 400
            except Exception:
                return jsonify({'status': 'error', 'message': 'Invalid URL format'}), 400
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
    """Проверяет статус асинхронной задачи"""
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
        else:  # FAILURE или другие состояния
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

@app.route('/analyze', methods=['POST', 'OPTIONS'])
def analyze_article():
    """Анализирует статью синхронно"""
    if request.method == 'OPTIONS':
        return jsonify({}), 200

    try:
        data = request.get_json()
        if not data or 'input_text' not in data:
            return jsonify({
                'status': 'error',
                'message': 'Input text is required',
                'details': 'The request must include an input_text field'
            }), 400

        input_text = data['input_text'].strip()
        if not input_text:
            return jsonify({
                'status': 'error',
                'message': 'Input text cannot be empty',
                'details': 'The input text field cannot be empty or just whitespace'
            }), 400

        # Дополнительная валидация URL
        if input_text.startswith(('http://', 'https://')):
            try:
                parsed = urlparse(input_text)
                if not all([parsed.scheme, parsed.netloc]):
                    return jsonify({
                        'status': 'error',
                        'message': 'Invalid URL format',
                        'details': 'URL must start with http:// or https:// and contain a valid domain'
                    }), 400
            except Exception:
                return jsonify({
                    'status': 'error',
                    'message': 'Invalid URL format',
                    'details': 'URL must start with http:// or https:// and contain a valid domain'
                }), 400
        elif len(input_text) < 50:
            return jsonify({
                'status': 'error',
                'message': 'Content is too short for analysis',
                'details': 'The input text must be at least 50 characters long'
            }), 400

        # Проверяем кэш
        cached_result = cache.get_cached_article_analysis(input_text)
        if cached_result:
            return jsonify({
                'status': 'success',
                'article': cached_result,
                'cached': True,
                'message': 'Result retrieved from cache'
            })

        # Извлечение контента из URL или использование прямого ввода
        if input_text.startswith(('http://', 'https://')):
            content, source, title, error = extract_text_from_url(input_text)
            if error:
                return jsonify({
                    'status': 'error',
                    'message': f"Failed to extract content from URL: {error}",
                    'source': source,
                    'title': title,
                    'input_text': input_text,
                    'fallback_data': {
                        'similar_articles': news_api._get_fallback_articles("technology", 3),
                        'credibility_level': 'Medium'
                    }
                }), 400
        else:
            content = input_text
            source = 'Direct Input'
            title = 'User-provided Text'

        # Анализ статьи
        analysis = claude_api.analyze_article(content, source)
        credibility_level = determine_credibility_level(analysis.get('credibility_score', {}).get('score', 0.6))

        # Получаем похожие статьи
        query = build_newsapi_query(analysis)
        similar_articles = []
        if query:
            similar_articles = news_api.get_everything(query=query, page_size=5)

        # Формируем результат
        result = {
            'title': title,
            'source': source,
            'url': input_text if input_text.startswith(('http://', 'https://')) else None,
            'short_summary': content[:200] + '...' if len(content) > 200 else content,
            'analysis': analysis,
            'credibility_level': credibility_level,
            'credibility_score': analysis.get('credibility_score', {'score': 0.6}),
            'similar_articles': similar_articles,
            'search_query': query,
            'metadata': {
                'analysis_timestamp': datetime.now().isoformat(),
                'content_length': len(content),
                'source_credibility': determine_credibility_level_from_source(source)
            }
        }

        # Кэшируем результат
        cache.cache_article_analysis(input_text, result)

        # Обновляем историю анализа
        with history_lock:
            global analysis_history
            analysis_history.insert(0, {
                "title": title,
                "source": source,
                "url": input_text if input_text.startswith(('http://', 'https://')) else None,
                "summary": content[:200] + '...' if len(content) > 200 else content,
                "credibility": credibility_level,
                "timestamp": datetime.now().isoformat(),
                "credibility_score": result['credibility_score']['score'] if isinstance(result['credibility_score'], dict) else result['credibility_score']
            })
            analysis_history = analysis_history[:10]  # Оставляем только последние 10 записей

        return jsonify({
            'status': 'success',
            'article': result,
            'message': 'Analysis completed successfully'
        })

    except Exception as e:
        logger.error(f"Error analyzing article: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': 'An unexpected error occurred during analysis',
            'details': str(e),
            'fallback_data': {
                'similar_articles': news_api._get_fallback_articles("technology", 3),
                'credibility_level': 'Medium',
                'analysis': {
                    'credibility_score': {'score': 0.6},
                    'sentiment': {'score': 0.0},
                    'bias': {'level': 0.3},
                    'topics': [],
                    'perspectives': {
                        'neutral': {
                            'summary': 'Fallback analysis due to error',
                            'credibility': 'Medium'
                        }
                    }
                }
            }
        }), 500

@app.route('/clear-cache', methods=['POST'])
def clear_cache():
    """Очищает кэш анализа"""
    try:
        cache.clear_all_caches()
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
    """Проверка состояния приложения"""
    try:
        # Проверяем соединение с базой (если используется)
        # Проверяем доступность API
        api_status = {
            'news_api': 'unavailable',
            'claude_api': 'unavailable'
        }

        # Проверяем NewsAPI
        try:
            test_result = news_api.get_everything("test", page_size=1)
            if test_result:
                api_status['news_api'] = 'operational'
        except Exception as e:
            logger.warning(f"NewsAPI health check failed: {str(e)}")
            api_status['news_api'] = f'unavailable: {str(e)}'

        # Проверяем ClaudeAPI (если возможно)
        try:
            if hasattr(claude_api, 'health_check'):
                if claude_api.health_check():
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

def check_newsapi_connection():
    """Проверка соединения с NewsAPI при старте"""
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

def configure_logging():
    """Настройка логирования"""
    log_level = os.getenv('LOG_LEVEL', 'INFO').upper()

    # Настройка формата логов
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    if os.getenv('FLASK_ENV') == 'development':
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s - %(filename)s:%(lineno)d'

    # Настройка обработчиков логов
    handlers = [logging.StreamHandler(sys.stdout)]
    if os.getenv('LOG_FILE'):
        handlers.append(logging.FileHandler(os.getenv('LOG_FILE')))

    # Применение конфигурации
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

def validate_url(url: str) -> Tuple[bool, Optional[str]]:
    """
    Валидация URL с более строгими проверками
    Возвращает кортеж: (валиден ли URL, сообщение об ошибке)
    """
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

if __name__ == '__main__':
    # Конфигурация логирования
    configure_logging()

    # Проверка соединения с NewsAPI при старте
    if not check_newsapi_connection():
        logger.warning("Could not connect to NewsAPI - will use fallback data")

    # Запуск приложения
    app.run(
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 5000)),
        threaded=True,
        debug=os.getenv('FLASK_ENV') == 'development'
    )
