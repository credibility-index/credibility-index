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
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if data.get('status') != 'ok':
                logger.error(f"NewsAPI returned error: {data.get('message', 'Unknown error')}")
                return []

            return data.get('articles', [])

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching news from NewsAPI: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error in NewsAPI: {str(e)}")
            return []

# Настройка CORS
CORS(app, resources={
    r"/*": {
        "origins": os.getenv('CORS_ORIGINS', '*').split(','),
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Инициализация компонентов
try:
    from cache import CacheManager
    from claude_api import ClaudeAPI
    cache = CacheManager()
    claude_api = ClaudeAPI()
    news_api = NewsAPI()
except ImportError as e:
    logger.error(f"Failed to import modules: {str(e)}")
    raise

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
            analysis_history = analysis_history[:10]  # Оставляем только последние 10

        self.update_state(state='PROGRESS', meta={'progress': 100, 'message': 'Completed'})
        return {"status": "success", "result": result}

    except Exception as e:
        logger.error(f"Error in async article analysis: {str(e)}", exc_info=True)
        return {"status": "error", "message": str(e)}

def build_newsapi_query(analysis: dict) -> str:
    """Строит запрос для NewsAPI на основе анализа статьи"""
    query_parts = []

    # Добавляем темы
    topics = analysis.get('topics', [])
    if isinstance(topics, list):
        query_parts.extend([t['name'] if isinstance(t, dict) else t for t in topics[:3]])

    # Добавляем важные сущности
    entities = analysis.get('entities', [])
    if isinstance(entities, list):
        important_entities = [e for e in entities if isinstance(e, dict) and e.get('importance', 0) > 0.7]
        query_parts.extend([e['name'] for e in important_entities[:3]])

    return ' OR '.join(query_parts) if query_parts else "technology"

def extract_text_from_url(url: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
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
            adapter = HTTPAdapter(max_retries=retries)
            session.mount("http://", adapter)
            session.mount("https://", adapter)

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
        return jsonify(daily_buzz)

@app.route('/source-credibility-chart')
def get_source_credibility_chart():
    """Возвращает данные для графика достоверности источников"""
    return jsonify(source_credibility_data)

@app.route('/analysis-history')
def get_analysis_history():
    """Возвращает историю анализов"""
    with history_lock:
        return jsonify({"history": analysis_history})

@app.route('/start-analysis', methods=['POST'])
def start_analysis():
    """Начинает асинхронный анализ статьи"""
    try:
        data = request.get_json()
        if not data or 'input_text' not in data:
            return jsonify({'status': 'error', 'message': 'Input text is required'}), 400

        input_text = data['input_text'].strip()
        if not input_text:
            return jsonify({'status': 'error', 'message': 'Input text cannot be empty'}), 400

        if input_text.startswith(('http://', 'https://')):
            if not re.match(r'^https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', input_text):
                return jsonify({'status': 'error', 'message': 'Invalid URL format'}), 400
        elif len(input_text) < 50:
            return jsonify({'status': 'error', 'message': 'Content is too short for analysis'}), 400

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
        response = {'status': task.state, 'message': 'Task not yet started'}
    elif task.state == 'PROGRESS':
        response = {
            'status': task.state,
            'progress': task.info.get('progress', 0),
            'message': task.info.get('message', '')
        }
    elif task.state == 'SUCCESS':
        response = {'status': task.state, 'result': task.result}
    else:
        response = {
            'status': task.state,
            'message': str(task.info) if task.info else 'Task failed'
        }

    return jsonify(response)

@app.route('/analyze', methods=['POST'])
def analyze_article():
    """Анализирует статью синхронно"""
    try:
        data = request.get_json()
        if not data or 'input_text' not in data:
            return jsonify({'status': 'error', 'message': 'Input text is required'}), 400

        input_text = data['input_text'].strip()
        if not input_text:
            return jsonify({'status': 'error', 'message': 'Input text cannot be empty'}), 400

        if input_text.startswith(('http://', 'https://')):
            if not re.match(r'^https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', input_text):
                return jsonify({'status': 'error', 'message': 'Invalid URL format'}), 400
        elif len(input_text) < 50:
            return jsonify({'status': 'error', 'message': 'Content is too short for analysis'}), 400

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

        analysis = claude_api.analyze_article(content, source)
        credibility_level = determine_credibility_level(analysis.get('credibility_score', {}).get('score', 0.6))

        # Получаем похожие статьи
        query = build_newsapi_query(analysis)
        similar_articles = news_api.get_everything(query=query, page_size=5) if query else []

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

        # Обновляем историю анализа
        with history_lock:
            global analysis_history
            analysis_history.insert(0, {
                "title": title,
                "source": source,
                "url": input_text if input_text.startswith(('http://', 'https://')) else None,
                "summary": content[:200] + '...' if len(content) > 200 else content,
                "credibility": credibility_level
            })
            analysis_history = analysis_history[:10]

        return jsonify({'status': 'success', 'article': result})

    except Exception as e:
        logger.error(f"Error analyzing article: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': 'An unexpected error occurred during analysis',
            'details': str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
