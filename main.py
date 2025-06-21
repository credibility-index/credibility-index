import os
import logging
import re
import json
from datetime import datetime
from urllib.parse import urlparse
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import anthropic
from newspaper import Article, Config
from database import Database
from news_api import NewsAPI
import sqlite3

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
anthropic_client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

# Конфигурация библиотеки newspaper
user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
config = Config()
config.browser_user_agent = user_agent
config.request_timeout = 30


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

        # Зададим тему для buzz, например, «Израиль/Иран»
        buzz_topics = buzz_analysis.get('topics', [])
        if not buzz_topics:
            buzz_topics = ["Israel", "Iran"]

        source_result = db.get_source_credibility_chart()
        if source_result['status'] != 'success':
            logger.error(f"Failed to load source credibility data: {source_result.get('message', 'Unknown error')}")
            source_credibility_data = {
                'sources': ['BBC', 'Reuters', 'CNN'],
                'credibility_scores': [0.9, 0.85, 0.8]
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



@app.route('/daily-buzz')
def daily_buzz():
    """Эндпоинт для получения статьи дня"""
    try:
        result = db.get_daily_buzz()
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error getting daily buzz: {str(e)}", exc_info=True)
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/source-credibility-chart')
def source_credibility_chart():
    """Эндпоинт для получения данных достоверности источников"""
    try:
        result = db.get_source_credibility_chart()
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error getting source credibility data: {str(e)}", exc_info=True)
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/analysis-history')
def analysis_history():
    """Эндпоинт для получения истории анализа"""
    try:
        result = db.get_analysis_history()
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error getting analysis history: {str(e)}", exc_info=True)
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/analyze', methods=['POST'])
def analyze_article():
    """Анализ статьи"""
    try:
        data = request.get_json()
        input_text = data.get('input_text', '').strip()

        if not input_text:
            return jsonify({
                'status': 'error',
                'message': 'Input text is required'
            }), 400

        # Извлечение контента статьи
        if input_text.startswith(('http://', 'https://')):
            content, source, title = extract_text_from_url(input_text)
            if not content:
                return jsonify({
                    'status': 'error',
                    'message': 'Could not extract article content'
                }), 400
        else:
            content = input_text
            source = 'Direct Input'
            title = 'User-provided Text'

        # Анализ с помощью Claude
        analysis = analyze_with_claude(content, source)

        # Определение уровня достоверности
        credibility_level = determine_credibility_level(analysis.get('index_of_credibility', 0.0))

        # Сохранение в базу данных
        article_id = db.save_article(
            title=title,
            source=source,
            url=input_text if input_text.startswith(('http://', 'https://')) else None,
            content=content,
            short_summary=content[:200] + '...' if len(content) > 200 else content,
            analysis_data=analysis,
            credibility_level=credibility_level
        )

        # Обновление статистики источников
        db.update_source_stats(source, credibility_level)

        # Получение похожих статей
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
            'message': str(e)
        }), 500


def extract_text_from_url(url: str) -> tuple:
    """Извлечь текст из URL"""
    try:
        parsed = urlparse(url)
        clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"

        if any(domain in url for domain in ['youtube.com', 'vimeo.com']):
            return None, parsed.netloc.replace('www.', ''), "Video content detected"

        article = Article(clean_url, config=config)
        article.download()
        article.parse()

        if not article.text or len(article.text.strip()) < 100:
            return None, None, None

        domain = parsed.netloc.replace('www.', '')
        title = article.title.strip() if article.title else "No title"

        return article.text.strip(), domain, title

    except Exception as e:
        logger.error(f"Error extracting article from {url}: {str(e)}", exc_info=True)
        return None, None, None


def analyze_with_claude(content: str, source: str) -> dict:
    """Анализировать контент с помощью Claude API"""
    try:
        prompt = f"""Analyze this news article and provide a JSON response with these fields:
- news_integrity (0.0-1.0)
- fact_check_needed_score (0.0-1.0)
- sentiment_score (0.0-1.0)
- bias_score (0.0-1.0)
- topics (list of strings)
- key_arguments (list of strings)
- mentioned_facts (list of strings)
- author_purpose (string)
- potential_biases_identified (list of strings)
- short_summary (string)
- index_of_credibility (0.0-1.0)
- western_perspective (object with summary)
- iranian_perspective (object with summary)
- israeli_perspective (object with summary)
- neutral_perspective (object with summary)

Article content:
{content[:5000]}..."""

        response = anthropic_client.messages.create(
            model=os.getenv('ANTHROPIC_MODEL', 'claude-3-opus-20240229'),
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )

        response_text = response.content[0].text.strip()

        # Попытка разобрать JSON ответ
        try:
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            return json.loads(response_text)
        except json.JSONDecodeError:
            logger.error("Failed to parse JSON response from Claude API")
            return get_default_analysis()

    except Exception as e:
        logger.error(f"Error analyzing with Claude: {str(e)}", exc_info=True)
        return get_default_analysis()


def determine_credibility_level(score: float) -> str:
    """Определить уровень достоверности на основе оценки"""
    if score >= 0.8:
        return "High"
    elif score >= 0.6:
        return "Medium"
    else:
        return "Low"


def get_similar_articles(topics: list) -> list:
    """Получить похожие статьи на основе тем"""
    try:
        # Сначала пытаемся получить похожие статьи из нашей базы данных
        similar_articles = db.get_similar_articles(topics)

        # Если у нас недостаточно похожих статей, получаем некоторые из News API
        if len(similar_articles) < 3 and topics:
            query = " OR ".join(topics[:3])  # Используем первые 3 темы для запроса
            news_articles = news_api.get_everything(query=query, page_size=3)

            if news_articles:
                for article in news_articles:
                    similar_articles.append({
                        'title': article['title'],
                        'source': article['source']['name'],
                        'summary': article['description'],
                        'url': article['url'],
                        'credibility': 'Medium'  # Уровень достоверности по умолчанию для внешних статей
                    })

        return similar_articles[:5]  # Возвращаем максимум 5 статей

    except Exception as e:
        logger.error(f"Error getting similar articles: {str(e)}", exc_info=True)
        return []


def get_default_analysis() -> dict:
    """Получить данные анализа по умолчанию"""
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
