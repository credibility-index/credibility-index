import os
iimport os
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
user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
config = Config()
config.browser_user_agent = user_agent
config.request_timeout = 30

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

            # Проверяем, есть ли ответ в кэше
            if cache_key in cached_responses:
                response_data, timestamp = cached_responses[cache_key]
                if datetime.now() - timestamp < timedelta(seconds=timeout):
                    response = make_response(response_data)
                    response.headers['X-Cache'] = 'HIT'
                    return response

            # Если нет в кэше, выполняем функцию
            response = f(*args, **kwargs)
            response_data = response.get_data()

            # Сохраняем ответ в кэше
            cached_responses[cache_key] = (response_data, datetime.now())
            response.headers['X-Cache'] = 'MISS'

            return response

        wrapped.__name__ = f"wrapped_{f.__name__}"
        return wrapped
    return decorator

@app.route('/')
def home():
    """Главная страница с анализом"""
    try:
        buzz_result = db.get_daily_buzz()
        if buzz_result['status'] != 'success':
            logger.error(f"Failed to load featured analysis: {buzz_result.get('message', 'Unknown error')}")
            buzz_analysis = get_default_analysis()
            buzz_article = None
        else:
            buzz_article = buzz_result['article']
            buzz_analysis = buzz_article['analysis']

        buzz_topics = buzz_analysis.get('topics', [])
        if not buzz_topics:
            buzz_topics = ["Israel", "Iran"]

        source_result = db.get_source_credibility_chart()
        if source_result['status'] != 'success':
            logger.error(f"Failed to load source credibility data: {source_result.get('message', 'Unknown error')}")
            source_credibility_data = {
                'sources': ['BBC', 'Reuters', 'CNN', 'The Guardian', 'Fox News'],
                'credibility_scores': [0.92, 0.88, 0.75, 0.85, 0.65]
            }
        else:
            source_credibility_data = source_result['data']

        history_result = db.get_analysis_history()
        if history_result['status'] != 'success':
            logger.error(f"Failed to load analysis history: {history_result.get('message', 'Unknown error')}")
            analyzed_articles = []
        else:
            analyzed_articles = history_result['history']

        return render_template('index.html',
                             buzz_article=buzz_article,
                             buzz_analysis=buzz_analysis,
                             buzz_topics=buzz_topics,
                             analyzed_articles=analyzed_articles,
                             source_credibility_data=source_credibility_data)
    except Exception as e:
        logger.error(f"Error loading home page: {str(e)}", exc_info=True)
        return render_template('error.html', message="Failed to load home page")

@app.route('/index')
def index():
    """Главная страница приложения"""
    return home()

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
def daily_buzz_route():
    """Маршрут для получения статьи дня"""
    try:
        result = db.get_daily_buzz()
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error getting daily buzz: {str(e)}", exc_info=True)
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/source-credibility-chart')
@cache_response(timeout=300)
def source_credibility_chart_route():
    """Маршрут для получения данных чарта достоверности"""
    try:
        result = db.get_source_credibility_chart()
        if result['status'] != 'success':
            logger.error(f"Error getting source credibility data: {result.get('message', 'Unknown error')}")
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
def analysis_history_route():
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
        input_text = data.get('input_text', '').strip()

        if not input_text:
            return jsonify({'status': 'error', 'message': 'Input text is required'}), 400

        if input_text.startswith(('http://', 'https://')):
            try:
                content, source, title = extract_text_from_url(input_text)
                if not content:
                    return jsonify({
                        'status': 'error',
                        'message': 'Could not extract article content',
                        'details': 'The URL might be invalid or the content is not accessible'
                    }), 400
            except Exception as url_error:
                logger.error(f"Error processing URL: {str(url_error)}")
                return jsonify({
                    'status': 'error',
                    'message': 'Error processing the URL',
                    'details': str(url_error)
                }), 400
        else:
            content = input_text
            source = 'Direct Input'
            title = 'User-provided Text'

        try:
            analysis = analyze_with_claude(content, source)
            if not analysis or 'index_of_credibility' not in analysis:
                logger.warning("Received incomplete analysis from Claude, using default values")
                analysis = get_default_analysis()
        except Exception as claude_error:
            logger.error(f"Error with Claude analysis: {str(claude_error)}")
            return jsonify({
                'status': 'error',
                'message': 'Error during article analysis',
                'details': str(claude_error)
            }), 500

        credibility_level = determine_credibility_level(analysis.get('index_of_credibility', 0.0))

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
        except Exception as db_error:
            logger.error(f"Error saving article to database: {str(db_error)}")
            return jsonify({
                'status': 'error',
                'message': 'Error saving analysis results',
                'details': str(db_error)
            }), 500

        try:
            db.update_source_stats(source, credibility_level)
        except Exception as stats_error:
            logger.error(f"Error updating source stats: {str(stats_error)}")

        try:
            similar_articles = get_similar_articles(analysis.get('topics', []))
        except Exception as similar_error:
            logger.error(f"Error getting similar articles: {str(similar_error)}")
            similar_articles = []

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

@app.route('/health')
def health_check():
    """Проверка состояния API"""
    try:
        # Проверяем соединение с базой данных
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()

        # Проверяем доступность API новостей
        test_articles = news_api.get_everything(query="test", page_size=1)

        return jsonify({
            'status': 'healthy',
            'database': 'connected',
            'news_api': 'available' if test_articles else 'unavailable',
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/system-info')
def system_info():
    """Возвращает информацию о системе"""
    return jsonify({
        'system': 'Media Credibility Index',
        'version': '1.0',
        'status': 'operational',
        'timestamp': datetime.now().isoformat(),
        'environment': os.getenv('FLASK_ENV', 'development')
    })

def extract_text_from_url(url: str) -> tuple:
    try:
        parsed = urlparse(url)
        if not all([parsed.scheme, parsed.netloc]):
            logger.error(f"Invalid URL format: {url}")
            return None, None, None

        clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"

        # Проверяем, является ли URL видео
        video_domains = ['youtube.com', 'vimeo.com', 'twitch.tv']
        if any(domain in parsed.netloc for domain in video_domains):
            domain = parsed.netloc.replace('www.', '')
            return None, domain, "Video content detected"

        article = Article(clean_url, config=config)
        article.download()

        if article.download_state != 200:
            logger.error(f"Failed to download article from {url}, status code: {article.download_state}")
            return None, None, None

        article.parse()

        if not article.text or len(article.text.strip()) < 100:
            logger.error(f"Insufficient content extracted from {url}")
            return None, None, None

        domain = parsed.netloc.replace('www.', '')
        title = article.title.strip() if article.title else "No title available"

        return article.text.strip(), domain, title
    except Exception as e:
        logger.error(f"Error extracting article from {url}: {str(e)}", exc_info=True)
        return None, None, None

def analyze_with_claude(content: str, source: str) -> dict:
    try:
        prompt = f"""Analyze the following news article and provide a detailed JSON response with these fields:

1. news_integrity: float between 0.0-1.0 indicating overall integrity
2. fact_check_needed_score: float between 0.0-1.0 indicating likelihood that fact-checking is needed
3. sentiment_score: float between 0.0-1.0 indicating emotional tone (0.0 negative, 0.5 neutral, 1.0 positive)
4. bias_score: float between 0.0-1.0 indicating degree of bias (1.0 low bias, 0.0 high bias)
5. topics: list of main topics covered in the article
6. key_arguments: list of key arguments presented
7. mentioned_facts: list of key facts mentioned
8. author_purpose: string describing the author's main purpose
9. potential_biases_identified: list of potential biases
10. short_summary: concise summary of the article
11. index_of_credibility: overall credibility index between 0.0-1.0
12. western_perspective: object with summary of Western perspective
13. iranian_perspective: object with summary of Iranian perspective
14. israeli_perspective: object with summary of Israeli perspective
15. neutral_perspective: object with summary of neutral perspective

Article content (first 5000 characters):
{content[:5000]}

Provide only the JSON response without any additional text or explanations.
Ensure all required fields are included in the response."""

        response = anthropic_client.messages.create(
            model=os.getenv('ANTHROPIC_MODEL', 'claude-3-opus-20240229'),
            max_tokens=2000,
            temperature=0.5,
            messages=[{"role": "user", "content": prompt}]
        )

        response_text = response.content[0].text.strip()

        try:
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                analysis = json.loads(json_str)

                required_fields = [
                    'news_integrity', 'fact_check_needed_score', 'sentiment_score',
                    'bias_score', 'topics', 'index_of_credibility'
                ]

                for field in required_fields:
                    if field not in analysis:
                        analysis[field] = get_default_analysis()[field]

                return analysis

            return json.loads(response_text)
        except json.JSONDecodeError:
            logger.error("Failed to parse JSON from Claude response")
            return get_default_analysis()
    except Exception as e:
        logger.error(f"Error analyzing with Claude: {str(e)}", exc_info=True)
        return get_default_analysis()

def determine_credibility_level(score: float) -> str:
    if score >= 0.8:
        return "High"
    elif score >= 0.6:
        return "Medium"
    else:
        return "Low"

def get_similar_articles(topics: list) -> list:
    try:
        similar_articles = db.get_similar_articles(topics)
        if len(similar_articles) < 5 and topics:
            query = " OR ".join(topics[:3])
            news_articles = news_api.get_everything(query=query, page_size=5)
            if news_articles:
                for article in news_articles:
                    url = article.get('url')
                    title = article.get('title')
                    source = article['source']['name']
                    summary = article.get('description') or article.get('content') or ''
                    content = article.get('content') or ''
                    short_summary = summary[:200] + '...' if len(summary) > 200 else summary
                    if db.article_exists(url):
                        continue
                    db.save_article(
                        title=title,
                        source=source,
                        url=url,
                        content=content,
                        short_summary=short_summary,
                        analysis_data={
                            "topics": topics,
                            "short_summary": short_summary,
                            "index_of_credibility": 0.6
                        },
                        credibility_level="Medium"
                    )
                    similar_articles.append({
                        'title': title,
                        'source': source,
                        'summary': short_summary,
                        'url': url,
                        'credibility': "Medium"
                    })
        return similar_articles[:5]
    except Exception as e:
        logger.error(f"Error getting similar articles: {str(e)}", exc_info=True)
        return []

def get_default_analysis() -> dict:
    return {
        "news_integrity": 0.7,
        "fact_check_needed_score": 0.3,
        "sentiment_score": 0.5,
        "bias_score": 0.4,
        "topics": ["default"],
        "key_arguments": ["default"],
        "mentioned_facts": ["default"],
        "author_purpose": "inform",
        "potential_biases_identified": ["none"],
        "short_summary": "Default summary",
        "index_of_credibility": 0.6,
        "western_perspective": {"summary": "Default western perspective"},
        "iranian_perspective": {"summary": "Default iranian perspective"},
        "israeli_perspective": {"summary": "Default israeli perspective"},
        "neutral_perspective": {"summary": "Default neutral perspective"}
    }

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(app.static_folder, filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
