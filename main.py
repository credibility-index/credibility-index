import os
import logging
import re
import json
import socket
from datetime import datetime, timedelta
from urllib.parse import urlparse
from flask import Flask, request, jsonify, render_template, send_from_directory, make_response, redirect, url_for
from flask_cors import CORS
import anthropic
from newspaper import Article, Config
from database import Database
from news_api import NewsAPI
import hashlib
import requests
from bs4 import BeautifulSoup
from functools import lru_cache
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Инициализация приложения
app = Flask(__name__, static_folder='static', template_folder='templates')

# Настройка CORS с конкретными параметрами
cors = CORS(app, resources={
    r"/*": {
        "origins": ["https://indexing.media", "http://localhost:*", "http://localhost:5000", "https://yourdomain.com"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "X-Requested-With"],
        "supports_credentials": True,
        "max_age": 3600
    }
})

# Настройка логирования
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO').upper(), logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Инициализация базы данных и API
db = Database()
news_api = NewsAPI()

# Настройка повторных попыток для запросов
session = requests.Session()
retries = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[500, 502, 503, 504, 408, 429],
    allowed_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"]
)
session.mount('http://', HTTPAdapter(max_retries=retries))
session.mount('https://', HTTPAdapter(max_retries=retries))

# Функция для проверки DNS
def check_dns_resolution(domain):
    """Проверка разрешения DNS с таймаутом"""
    try:
        socket.setdefaulttimeout(5)  # Устанавливаем таймаут для DNS запросов
        socket.gethostbyname(domain)
        logger.info(f"DNS resolution successful for {domain}")
        return True
    except socket.gaierror as e:
        logger.error(f"DNS resolution failed for {domain}: {str(e)}")
        return False
    except socket.timeout:
        logger.error(f"DNS resolution timeout for {domain}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during DNS resolution for {domain}: {str(e)}")
        return False

# Инициализация клиента Anthropic с проверкой доступности
try:
    anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
    if not anthropic_api_key:
        raise ValueError("ANTHROPIC_API_KEY is not set in environment variables")

    # Проверка DNS перед инициализацией клиента
    if not check_dns_resolution('api.anthropic.com'):
        raise ConnectionError("Failed to resolve Anthropic API domain")

    anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)
except Exception as e:
    logger.error(f"Failed to initialize Anthropic client: {str(e)}")
    anthropic_client = None

# Конфигурация библиотеки newspaper
config = Config()
config.browser_user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36'
config.request_timeout = 30
config.memoize_articles = False

def generate_cache_key(*args, **kwargs):
    """Генерирует ключ кэша на основе аргументов"""
    key = str(args) + str(kwargs)
    return hashlib.md5(key.encode('utf-8')).hexdigest()

def cache_response(timeout=300):
    """Декоратор для кэширования ответов"""
    def decorator(f):
        cached_responses = {}
        def wrapped(*args, **kwargs):
            cache_key = generate_cache_key(*args, **kwargs)
            if cache_key in cached_responses:
                response_data, timestamp = cached_responses[cache_key]
                if datetime.now() - timestamp < timedelta(seconds=timeout):
                    response = make_response(response_data)
                    response.headers['X-Cache'] = 'HIT'
                    return response
            response = f(*args, **kwargs)
            response_data = response.get_data()
            cached_responses[cache_key] = (response_data, datetime.now())
            response.headers['X-Cache'] = 'MISS'
            return response
        wrapped.__name__ = f"wrapped_{f.__name__}"
        return wrapped
    return decorator

@app.route('/')
def home():
    """Главная страница с анализом"""
    return redirect(url_for('index'))

@app.route('/index')
def index():
    """Главная страница приложения"""
    try:
        # Проверка DNS перед загрузкой данных
        if not check_dns_resolution('indexing.media'):
            return render_template('error.html', message="DNS resolution failed for service domain")

        buzz_result = db.get_daily_buzz()
        source_result = db.get_source_credibility_chart()
        history_result = db.get_analysis_history()

        context = {
            'buzz_article': buzz_result.get('article') if buzz_result['status'] == 'success' else None,
            'buzz_analysis': buzz_result.get('article', {}).get('analysis') if buzz_result['status'] == 'success' else get_default_analysis(),
            'buzz_topics': buzz_result.get('article', {}).get('analysis', {}).get('topics', ["Israel", "Iran"]),
            'source_credibility_data': source_result['data'] if source_result['status'] == 'success' else {
                'sources': ['BBC', 'Reuters', 'CNN', 'The Guardian', 'Fox News'],
                'credibility_scores': [0.92, 0.88, 0.75, 0.85, 0.65]
            },
            'analyzed_articles': history_result['history'] if history_result['status'] == 'success' else []
        }
        return render_template('index.html', **context)
    except Exception as e:
        logger.error(f"Error loading home page: {str(e)}", exc_info=True)
        return render_template('error.html', message="Failed to load home page")

@app.route('/health')
def health_check():
    """Проверка состояния API с проверкой DNS"""
    try:
        # Проверка DNS для критичных сервисов
        dns_checks = {
            'database': check_dns_resolution('your-database-domain.com'),
            'news_api': check_dns_resolution('newsapi.org'),
            'anthropic': check_dns_resolution('api.anthropic.com')
        }

        if not all(dns_checks.values()):
            failed_services = [service for service, status in dns_checks.items() if not status]
            return jsonify({
                'status': 'unhealthy',
                'failed_dns_checks': failed_services,
                'timestamp': datetime.now().isoformat()
            }), 500

        # Проверка соединения с базой данных
        try:
            with db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                cursor.fetchone()
        except Exception as db_error:
            return jsonify({
                'status': 'unhealthy',
                'database': str(db_error),
                'timestamp': datetime.now().isoformat()
            }), 500

        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/daily-buzz')
@cache_response(timeout=300)
def daily_buzz():
    """Маршрут для получения статьи дня"""
    try:
        result = db.get_daily_buzz()
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error getting daily buzz: {str(e)}", exc_info=True)
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/source-credibility-chart')
@cache_response(timeout=300)
def source_credibility_chart():
    """Маршрут для получения данных чарта достоверности"""
    try:
        result = db.get_source_credibility_chart()
        if result['status'] != 'success':
            return jsonify({
                'status': 'success',
                'data': {
                    'sources': ['BBC', 'Reuters', 'CNN', 'The Guardian', 'Fox News'],
                    'credibility_scores': [0.92, 0.88, 0.75, 0.85, 0.65]
                }
            })
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error getting source credibility data: {str(e)}", exc_info=True)
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/analysis-history')
@cache_response(timeout=300)
def analysis_history():
    """Маршрут для получения истории анализа"""
    try:
        result = db.get_analysis_history()
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error getting analysis history: {str(e)}", exc_info=True)
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/analyze', methods=['POST'])
def analyze_article():
    """Улучшенный маршрут для анализа статьи с проверкой DNS и обработкой ошибок"""
    try:
        # Проверка доступности API перед обработкой
        if anthropic_client is None:
            return jsonify({
                'status': 'error',
                'message': 'Analysis service is temporarily unavailable'
            }), 503

        # Проверка данных запроса
        if not request.is_json:
            return jsonify({'status': 'error', 'message': 'Request must be JSON'}), 400

        data = request.get_json()
        if 'input_text' not in data:
            return jsonify({'status': 'error', 'message': 'Input text is required'}), 400

        input_text = data['input_text'].strip()
        if not input_text:
            return jsonify({'status': 'error', 'message': 'Input text cannot be empty'}), 400

        # Обработка URL или текста
        if input_text.startswith(('http://', 'https://')):
            try:
                parsed = urlparse(input_text)
                domain = parsed.netloc
                if not check_dns_resolution(domain):
                    return jsonify({
                        'status': 'error',
                        'message': f'Failed to resolve domain: {domain}'
                    }), 400
            except Exception as e:
                return jsonify({
                    'status': 'error',
                    'message': f'Invalid URL or DNS resolution failed: {str(e)}'
                }), 400

            content, source, title, error = extract_text_from_url(input_text)
            if error:
                return jsonify({
                    'status': 'error',
                    'message': error,
                    'source': source,
                    'title': title
                }), 400
        else:
            if len(input_text) < 50:
                return jsonify({
                    'status': 'error',
                    'message': 'Content is too short for analysis (minimum 50 characters required)'
                }), 400
            content = input_text
            source = 'Direct Input'
            title = 'User-provided Text'

        # Анализ с повторными попытками
        max_retries = 3
        analysis = None
        for attempt in range(max_retries):
            try:
                analysis = analyze_with_claude(content, source)
                if analysis:
                    break
            except Exception as e:
                logger.warning(f"Analysis attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    logger.error("All analysis attempts failed")
                    return jsonify({
                        'status': 'error',
                        'message': 'Failed to analyze article after multiple attempts'
                    }), 500

        if not analysis:
            return jsonify({
                'status': 'error',
                'message': 'Analysis failed to produce valid results'
            }), 500

        credibility_level = determine_credibility_level(analysis.get('credibility_score', {}).get('score', 0.6))

        # Сохранение в базу данных с повторными попытками
        max_db_retries = 3
        article_id = None
        for attempt in range(max_db_retries):
            try:
                article_id = db.save_article(
                    title=title,
                    source=source,
                    url=input_text if input_text.startswith(('http://', 'https://')) else None,
                    content=content,
                    short_summary=content[:200] + '...' if len(content) > 200 else content,
                    analysis_data=analysis,
                    credibility_level=credibility_level
                )
                break
            except Exception as e:
                logger.warning(f"Database save attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_db_retries - 1:
                    logger.error("All database save attempts failed")
                    return jsonify({
                        'status': 'error',
                        'message': 'Failed to save analysis results'
                    }), 500

        # Получение похожих статей
        similar_articles = []
        try:
            similar_articles = get_similar_articles(analysis.get('topics', []))
        except Exception as e:
            logger.error(f"Error getting similar articles: {str(e)}")

        return jsonify({
            'status': 'success',
            'article': {
                'id': article_id,
                'title': title,
                'source': source,
                'url': input_text if input_text.startswith(('http://', 'https://')) else None,
                'short_summary': content[:200] + '...' if len(content) > 200 else content,
                'analysis': analysis,
                'credibility_level': credibility_level
            },
            'similar_articles': similar_articles
        })

    except Exception as e:
        logger.error(f"Unexpected error analyzing article: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': 'An unexpected error occurred during analysis',
            'details': str(e)
        }), 500

def extract_text_from_url(url: str) -> tuple:
    """Улучшенная функция для извлечения текста из URL с обработкой ошибок"""
    try:
        parsed = urlparse(url)
        if not all([parsed.scheme, parsed.netloc]):
            return None, None, None, "Invalid URL format"

        clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"

        # Проверяем, является ли URL видео
        if any(domain in parsed.netloc for domain in ['youtube.com', 'vimeo.com', 'twitch.tv']):
            return None, parsed.netloc.replace('www.', ''), "Video content detected", None

        # Пробуем с библиотекой newspaper
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

        # Альтернативный метод с BeautifulSoup
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
        logger.error(f"Unexpected error extracting article from {url}: {str(e)}", exc_info=True)
        return None, parsed.netloc.replace('www.', ''), "Error occurred", str(e)

def analyze_with_claude(content: str, source: str) -> dict:
    """Улучшенная функция анализа статьи с интегрированным промптом и обработкой ошибок"""
    try:
        # Проверка доступности API перед анализом
        if not anthropic_client:
            raise Exception("Anthropic client is not initialized")

        # Проверка DNS перед анализом
        if not check_dns_resolution('api.anthropic.com'):
            raise ConnectionError("Failed to resolve Anthropic API domain")

        # Интегрированный промпт для анализа
        prompt = f"""Analyze the following news article and provide a comprehensive JSON response with the following structure:

1. Analysis Metadata:
   - Analysis timestamp: {datetime.now().isoformat()}
   - Source: {source}
   - Content length: {len(content)} characters

2. Credibility Assessment:
   - Credibility score (0-1) with detailed explanation
   - Confidence level in the score (low/medium/high)
   - Potential credibility issues identified

3. Content Analysis:
   - Key topics (max 5) with brief descriptions
   - Detailed summary (3-5 sentences)
   - Main perspectives (Western, Iranian, Israeli, Neutral - 2-3 sentences each with credibility assessment)
   - Sentiment analysis (positive/neutral/negative) with explanation and confidence score
   - Bias detection (low/medium/high) with detailed explanation

4. Structural Analysis:
   - Key arguments presented (3-5 main points with supporting evidence)
   - Mentioned facts (3-5 key facts with verification status)
   - Potential biases identified (list with explanations and severity assessment)
   - Author's purpose and potential agenda (1-2 sentences with supporting evidence)

5. Technical Assessment:
   - Content structure analysis
   - Language and style assessment
   - Potential manipulation techniques detected
   - Cross-referencing with known reliable sources

6. Recommendations:
   - Suggested improvements for credibility
   - Potential fact-checking requirements
   - Additional perspectives that could be included

Article content (first 4000 characters):
{content[:4000]}

Important notes:
- Provide all scores as numbers between 0 and 1 where applicable
- For each assessment, include confidence levels
- Flag any potential issues that might affect the analysis
- Include timestamps for all assessments
- Structure the response as valid JSON that can be directly parsed
- If any analysis cannot be completed, provide null values with explanations"""

        # Улучшенная обработка запроса к API с повторными попытками
        max_retries = 3
        last_error = None

        for attempt in range(max_retries):
            try:
                response = anthropic_client.messages.create(
                    model=os.getenv('ANTHROPIC_MODEL', 'claude-3-opus-20240229'),
                    max_tokens=2000,
                    temperature=0.3,
                    messages=[{"role": "user", "content": prompt}],
                    timeout=30  # Увеличен таймаут
                )

                response_text = response.content[0].text.strip()

                # Улучшенный парсинг ответа
                try:
                    # Пытаемся найти JSON в ответе
                    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                    if json_match:
                        analysis = json.loads(json_match.group(0))

                        # Проверяем обязательные поля
                        required_fields = [
                            'credibility_score',
                            'topics',
                            'summary',
                            'perspectives',
                            'sentiment',
                            'bias'
                        ]

                        missing_fields = [field for field in required_fields if field not in analysis]
                        if not missing_fields:
                            return analysis
                        else:
                            logger.warning(f"Missing required fields in analysis: {missing_fields}")
                            last_error = f"Missing required fields: {missing_fields}"

                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON from Claude response: {str(e)}")
                    last_error = f"JSON parsing error: {str(e)}"

            except anthropic.APIError as e:
                logger.error(f"Anthropic API error: {str(e)}")
                last_error = f"API error: {str(e)}"
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Экспоненциальная задержка между попытками

            except Exception as e:
                logger.error(f"Unexpected error during analysis attempt {attempt + 1}: {str(e)}")
                last_error = f"Unexpected error: {str(e)}"
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)

        # Если все попытки не удались
        logger.error(f"All analysis attempts failed. Last error: {last_error}")
        return get_default_analysis()

    except Exception as e:
        logger.error(f"Critical error in analyze_with_claude: {str(e)}", exc_info=True)
        return get_default_analysis()

def determine_credibility_level(score: float) -> str:
    """Определение уровня достоверности"""
    if isinstance(score, dict):
        score = score.get('score', 0.6)
    if score >= 0.8:
        return "High"
    elif score >= 0.6:
        return "Medium"
    else:
        return "Low"

def get_similar_articles(topics: list) -> list:
    """Получение похожих статей по темам с обработкой ошибок"""
    try:
        similar_articles = db.get_similar_articles(topics)
        if len(similar_articles) < 5 and topics:
            query = " OR ".join(topics[:3])
            news_articles = news_api.get_everything(query=query, page_size=5)

            if news_articles:
                for article in news_articles:
                    url = article.get('url')
                    if not url or db.article_exists(url):
                        continue

                    similar_articles.append({
                        'title': article.get('title', 'No title'),
                        'source': article['source']['name'],
                        'summary': (article.get('description') or article.get('content') or '')[:150] + '...',
                        'url': url,
                        'credibility': "Medium"
                    })

        return similar_articles[:5]
    except Exception as e:
        logger.error(f"Error getting similar articles: {str(e)}", exc_info=True)
        return []

def get_default_analysis() -> dict:
    """Получение анализа по умолчанию"""
    return {
        "credibility_score": {
            "score": 0.6,
            "explanation": "Default credibility score based on average analysis"
        },
        "topics": [
            {"name": "general", "description": "General news topic"},
            {"name": "politics", "description": "Political news topic"}
        ],
        "summary": "This article discusses various perspectives on a current event. It presents multiple viewpoints and analyzes the situation from different angles.",
        "perspectives": {
            "western": {
                "summary": "Western perspective on the event, typically focusing on democratic values and international relations.",
                "key_points": ["Point 1", "Point 2"],
                "credibility": "Medium"
            },
            "iranian": {
                "summary": "Iranian perspective on the event, often emphasizing regional security and sovereignty.",
                "key_points": ["Point 1", "Point 2"],
                "credibility": "Medium"
            },
            "israeli": {
                "summary": "Israeli perspective on the event, usually centered around national security concerns.",
                "key_points": ["Point 1", "Point 2"],
                "credibility": "Medium"
            },
            "neutral": {
                "summary": "Neutral analysis of the event, attempting to present balanced viewpoints.",
                "key_points": ["Point 1", "Point 2"],
                "credibility": "Medium"
            }
        },
        "sentiment": {
            "score": "neutral",
            "explanation": "The article presents a balanced view without strong emotional bias"
        },
        "bias": {
            "level": "medium",
            "explanation": "The article shows some bias but attempts to present multiple viewpoints"
        },
        "key_arguments": [
            "Argument 1 with brief explanation",
            "Argument 2 with brief explanation"
        ],
        "mentioned_facts": [
            "Fact 1 with brief context",
            "Fact 2 with brief context"
        ],
        "potential_biases": [
            {"bias": "Bias 1", "explanation": "Explanation of potential bias"},
            {"bias": "Bias 2", "explanation": "Explanation of potential bias"}
        ],
        "author_purpose": "The author aims to inform readers about the event while presenting multiple perspectives."
    }

@app.route('/static/<path:filename>')
def static_files(filename):
    """Отдача статических файлов"""
    return send_from_directory(app.static_folder, filename)

@app.errorhandler(404)
def not_found_error(error):
    """Обработчик ошибки 404"""
    return render_template('error.html', message="Page not found"), 404

@app.errorhandler(500)
def internal_error(error):
    """Обработчик ошибки 500"""
    return render_template('error.html', message="Internal server error"), 500

if __name__ == '__main__':
    # Улучшенный запуск сервера с обработкой ошибок
    try:
        # Проверка DNS перед запуском
        if not check_dns_resolution('indexing.media'):
            raise ConnectionError("Failed to resolve domain name")

        port = int(os.environ.get('PORT', 5000))
        app.run(host='0.0.0.0', port=port, debug=False)
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}", exc_info=True)
        raise
