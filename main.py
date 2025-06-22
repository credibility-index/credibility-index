import os
import logging
import re
import json
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

# Инициализация приложения
app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# Настройка логирования
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO').upper(), logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Инициализация базы данных и API
db = Database()
news_api = NewsAPI()
anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
if not anthropic_api_key:
    logger.warning("ANTHROPIC_API_KEY is not set in environment variables")
anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)

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
        # Получаем данные для главной страницы
        buzz_result = db.get_daily_buzz()
        source_result = db.get_source_credibility_chart()
        history_result = db.get_analysis_history()

        # Подготавливаем данные для шаблона
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

@app.route('/faq')
def faq():
    """Страница с часто задаваемыми вопросами"""
    return render_template('faq.html')

@app.route('/feedback')
def feedback():
    """Страница с формой обратной связи"""
    return render_template('feedback.html')

@app.route('/feedback_success')
def feedback_success():
    """Страница подтверждения отправки обратной связи"""
    return render_template('feedback_success.html')

@app.route('/privacy')
def privacy():
    """Страница политики конфиденциальности"""
    return render_template('privacy.html')

@app.route('/terms')
def terms():
    """Страница условий использования"""
    return render_template('terms.html')

@app.route('/maintenance')
def maintenance():
    """Страница технического обслуживания"""
    return render_template('maintenance.html')

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
    """Маршрут для анализа статьи"""
    try:
        data = request.get_json()
        if not data or not data.get('input_text'):
            return jsonify({'status': 'error', 'message': 'Input text is required'}), 400

        input_text = data['input_text'].strip()

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
            if len(input_text) < 50:
                return jsonify({
                    'status': 'error',
                    'message': 'Content is too short for analysis (minimum 50 characters required)'
                }), 400
            content = input_text
            source = 'Direct Input'
            title = 'User-provided Text'

        analysis = analyze_with_claude(content, source)
        credibility_level = determine_credibility_level(analysis.get('index_of_credibility', 0.6))

        article_id = db.save_article(
            title=title,
            source=source,
            url=input_text if input_text.startswith(('http://', 'https://')) else None,
            content=content,
            short_summary=content[:200] + '...' if len(content) > 200 else content,
            analysis_data=analysis,
            credibility_level=credibility_level
        )

        try:
            db.update_source_stats(source, credibility_level)
        except Exception as e:
            logger.error(f"Error updating source stats: {str(e)}")

        similar_articles = get_similar_articles(analysis.get('topics', []))

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
        logger.error(f"Error analyzing article: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': 'An unexpected error occurred during analysis',
            'details': str(e)
        }), 500

def extract_text_from_url(url: str) -> tuple:
    """Улучшенная функция для извлечения текста из URL"""
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

            response = requests.get(clean_url, headers=headers, timeout=15)
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
    """Анализ статьи с помощью Claude с более компактным выводом"""
    try:
        prompt = f"""Analyze this news article and provide a structured JSON response with:
1. Credibility score (0-1)
2. Key topics (max 5)
3. Brief summary (2-3 sentences)
4. Main perspectives (Western, Iranian, Israeli, Neutral - 1 sentence each)
5. Sentiment analysis (positive/neutral/negative)
6. Bias detection (low/medium/high)

Article content (first 3000 characters):
{content[:3000]}"""

        response = anthropic_client.messages.create(
            model=os.getenv('ANTHROPIC_MODEL', 'claude-3-opus-20240229'),
            max_tokens=1500,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}]
        )

        response_text = response.content[0].text.strip()

        try:
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            return json.loads(response_text)
        except json.JSONDecodeError:
            logger.error("Failed to parse JSON from Claude response")
            return get_default_analysis()
    except Exception as e:
        logger.error(f"Error analyzing with Claude: {str(e)}", exc_info=True)
        return get_default_analysis()

def determine_credibility_level(score: float) -> str:
    """Определение уровня достоверности"""
    if score >= 0.8:
        return "High"
    elif score >= 0.6:
        return "Medium"
    else:
        return "Low"

def get_similar_articles(topics: list) -> list:
    """Получение похожих статей по темам"""
    try:
        similar_articles = db.get_similar_articles(topics)
        if len(similar_articles) < 3 and topics:
            query = " OR ".join(topics[:3])
            news_articles = news_api.get_everything(query=query, page_size=3)

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

        return similar_articles[:3]
    except Exception as e:
        logger.error(f"Error getting similar articles: {str(e)}", exc_info=True)
        return []

def get_default_analysis() -> dict:
    """Получение анализа по умолчанию"""
    return {
        "credibility_score": 0.6,
        "topics": ["general"],
        "summary": "This article discusses various perspectives on a current event.",
        "perspectives": {
            "western": "Western perspective on the event.",
            "iranian": "Iranian perspective on the event.",
            "israeli": "Israeli perspective on the event.",
            "neutral": "Neutral analysis of the event."
        },
        "sentiment": "neutral",
        "bias": "medium"
    }

@app.route('/static/<path:filename>')
def static_files(filename):
    """Отдача статических файлов"""
    return send_from_directory(app.static_folder, filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
