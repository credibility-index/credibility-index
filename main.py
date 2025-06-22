import os
import sys
import logging
from pathlib import Path
import re
import json
import requests  # Добавлен импорт requests
from datetime import datetime, timedelta
from urllib.parse import urlparse
from flask import Flask, request, jsonify, render_template_string, send_from_directory, session
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
app = Flask(__name__, static_folder='static')
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
            # В production среде лучше использовать Redis
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

# Настройка повторных попыток для запросов
session = requests.Session()
retries = Retry(total=3, backoff_factor=2, status_forcelist=[500, 502, 503, 504, 408, 429])
session.mount('http://', HTTPAdapter(max_retries=retries))
session.mount('https://', HTTPAdapter(max_retries=retries))

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
    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Media Analysis Dashboard</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 0; display: flex; height: 100vh; }
            .left-panel { width: 60%; padding: 20px; overflow-y: auto; }
            .right-panel { width: 40%; padding: 20px; border-left: 1px solid #ddd; overflow-y: auto; }
            .article { margin-bottom: 20px; padding: 15px; border: 1px solid #eee; border-radius: 5px; }
            .chart-container { height: 300px; border: 1px solid #eee; margin-bottom: 20px; }
            .analysis-form { margin-bottom: 20px; }
            textarea { width: 100%; height: 100px; margin-bottom: 10px; }
            button {
                padding: 10px 15px;
                background: #4CAF50;
                color: white;
                border: none;
                border-radius: 4px;
                cursor: pointer;
            }
            button:hover { background: #45a049; }
            button:disabled {
                background: #cccccc;
                cursor: not-allowed;
            }
            .credibility-high { color: green; }
            .credibility-medium { color: orange; }
            .credibility-low { color: red; }
            .navbar {
                background-color: #333;
                color: white;
                padding: 10px 20px;
                display: flex;
                justify-content: space-between;
            }
            .navbar a {
                color: white;
                text-decoration: none;
                margin-left: 15px;
            }
            .navbar a:hover { text-decoration: underline; }
            .footer {
                background-color: #333;
                color: white;
                text-align: center;
                padding: 10px;
                position: fixed;
                bottom: 0;
                width: 100%;
            }
            #progress-container {
                display: none;
                margin: 10px 0;
                padding: 10px;
                background: #f0f0f0;
                border-radius: 5px;
            }
            #progress-bar {
                width: 100%;
                background-color: #ddd;
                border-radius: 5px;
            }
            #progress {
                width: 0%;
                height: 20px;
                background-color: #4CAF50;
                border-radius: 5px;
                text-align: center;
                line-height: 20px;
                color: white;
            }
            .status-message {
                margin-top: 5px;
                font-size: 14px;
            }
        </style>
    </head>
    <body>
        <div class="navbar">
            <div>
                <a href="/">Media Analysis</a>
            </div>
            <div>
                <a href="/feedback">Feedback</a>
                <a href="/privacy">Privacy</a>
                <a href="/terms">Terms</a>
            </div>
        </div>

        <div class="left-panel">
            <h1>Daily Buzz</h1>
            <div class="article" id="daily-buzz">
                <h2>Loading...</h2>
                <p>Loading daily buzz article...</p>
            </div>
            <h2>Analysis History</h2>
            <div id="analysis-history">
                <p>Loading analysis history...</p>
            </div>
        </div>

        <div class="right-panel">
            <div class="analysis-form">
                <h2>Analyze Article</h2>
                <form id="analysis-form" method="post">
                    <input type="hidden" name="_csrf_token" value="{{ csrf_token }}">
                    <textarea id="article-input" name="article-input" placeholder="Enter article URL or text..."></textarea>
                    <button type="submit" onclick="startAnalysis(event)" id="analyze-btn">Analyze</button>
                </form>
                <div id="progress-container">
                    <div id="progress-bar">
                        <div id="progress">0%</div>
                    </div>
                    <div class="status-message" id="status-message">Starting analysis...</div>
                </div>
            </div>
            <div id="analysis-results">
                <p>Analysis results will appear here...</p>
            </div>
            <h2>Source Credibility Chart</h2>
            <div class="chart-container" id="credibility-chart">
                <p>Loading chart...</p>
            </div>
            <h2>Similar Articles</h2>
            <div id="similar-articles">
                <p>Similar articles will appear here...</p>
            </div>
        </div>

        <div class="footer">
            <p>&copy; 2023 Media Analysis. All rights reserved.</p>
        </div>

        <script>
            // Загрузка данных при старте
            document.addEventListener('DOMContentLoaded', function() {
                loadDailyBuzz();
                loadAnalysisHistory();
                loadCredibilityChart();
            });

            // Функция для анализа статьи
            function startAnalysis(event) {
                event.preventDefault();
                const input = document.getElementById('article-input').value.trim();
                const analyzeBtn = document.getElementById('analyze-btn');
                const progressContainer = document.getElementById('progress-container');
                const progressBar = document.getElementById('progress-bar');
                const progress = document.getElementById('progress');
                const statusMessage = document.getElementById('status-message');
                const resultsDiv = document.getElementById('analysis-results');

                if (!input) {
                    alert('Please enter article URL or text');
                    return false;
                }

                // Отключаем кнопку и показываем прогресс
                analyzeBtn.disabled = true;
                analyzeBtn.textContent = 'Analyzing...';
                progressContainer.style.display = 'block';
                progress.style.width = '0%';
                progress.textContent = '0%';
                statusMessage.textContent = 'Starting analysis...';

                // Получаем task_id из сервера
                fetch('/start-analysis', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ input_text: input })
                })
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'error') {
                        throw new Error(data.message);
                    }

                    const taskId = data.task_id;
                    checkTaskStatus(taskId);
                })
                .catch(error => {
                    console.error('Error starting analysis:', error);
                    resultsDiv.innerHTML = `<p>Error: ${error.message}</p>`;
                    resetAnalysisUI();
                });
                return false;
            }

            // Проверка статуса задачи
            function checkTaskStatus(taskId) {
                fetch(`/task-status/${taskId}`)
                    .then(response => response.json())
                    .then(data => {
                        if (data.status === 'PROGRESS') {
                            updateProgress(data.progress, data.message);
                            setTimeout(() => checkTaskStatus(taskId), 1000);
                        } else if (data.status === 'SUCCESS') {
                            displayResults(data.result);
                        } else {
                            throw new Error(data.message || 'Analysis failed');
                        }
                    })
                    .catch(error => {
                        console.error('Error checking task status:', error);
                        resultsDiv.innerHTML = `<p>Error: ${error.message}</p>`;
                        resetAnalysisUI();
                    });
            }

            // Обновление прогресса
            function updateProgress(progressValue, message) {
                const progress = document.getElementById('progress');
                const statusMessage = document.getElementById('status-message');

                progress.style.width = `${progressValue}%`;
                progress.textContent = `${progressValue}%`;
                statusMessage.textContent = message;
            }

            // Отображение результатов
            function displayResults(result) {
                const resultsDiv = document.getElementById('analysis-results');
                const article = result.article;

                resultsDiv.innerHTML = `
                    <div class="article">
                        <h2>${article.title}</h2>
                        <p><strong>Source:</strong> ${article.source}</p>
                        <p>${article.short_summary}</p>
                        <p><strong>Credibility:</strong> <span class="credibility-${article.credibility_level.toLowerCase()}">${article.credibility_level}</span></p>
                        <h3>Analysis:</h3>
                        <p><strong>Topics:</strong> ${article.analysis.topics.map(t => t.name || t).join(', ')}</p>
                    </div>
                `;

                // Загрузка похожих статей
                if (result.similar_articles && result.similar_articles.length > 0) {
                    const similarDiv = document.getElementById('similar-articles');
                    similarDiv.innerHTML = result.similar_articles.map(article => `
                        <div class="article">
                            <h3><a href="${article.url}" target="_blank">${article.title}</a></h3>
                            <p><strong>Source:</strong> ${article.source}</p>
                            <p>${article.summary}</p>
                            <p><strong>Credibility:</strong> <span class="credibility-${article.credibility.toLowerCase()}">${article.credibility}</span></p>
                        </div>
                    `).join('');
                }

                resetAnalysisUI();
            }

            // Сброс UI анализа
            function resetAnalysisUI() {
                const analyzeBtn = document.getElementById('analyze-btn');
                const progressContainer = document.getElementById('progress-container');

                analyzeBtn.disabled = false;
                analyzeBtn.textContent = 'Analyze';
                progressContainer.style.display = 'none';
            }

            // Загрузка статьи дня
            function loadDailyBuzz() {
                fetch('/daily-buzz')
                    .then(response => response.json())
                    .then(data => {
                        const buzz = data.article;
                        const articleDiv = document.getElementById('daily-buzz');
                        articleDiv.innerHTML = `
                            <h2>${buzz.title}</h2>
                            <p><strong>Source:</strong> ${buzz.source}</p>
                            <p>${buzz.short_summary}</p>
                            <h3>Analysis:</h3>
                            <p><strong>Credibility:</strong> <span class="credibility-${buzz.analysis.credibility_score.score >= 0.8 ? 'high' : buzz.analysis.credibility_score.score >= 0.6 ? 'medium' : 'low'}">
                                ${buzz.analysis.credibility_score.score >= 0.8 ? 'High' : buzz.analysis.credibility_score.score >= 0.6 ? 'Medium' : 'Low'}
                            </span></p>
                            <p><strong>Topics:</strong> ${buzz.analysis.topics.join(', ')}</p>
                        `;
                    })
                    .catch(error => {
                        console.error('Error loading daily buzz:', error);
                        document.getElementById('daily-buzz').innerHTML = '<p>Failed to load daily buzz article</p>';
                    });
            }

            // Загрузка истории анализа
            function loadAnalysisHistory() {
                fetch('/analysis-history')
                    .then(response => response.json())
                    .then(data => {
                        const historyDiv = document.getElementById('analysis-history');
                        if (data.history && data.history.length > 0) {
                            historyDiv.innerHTML = data.history.map(article => `
                                <div class="article">
                                    <h3><a href="${article.url}" target="_blank">${article.title}</a></h3>
                                    <p><strong>Source:</strong> ${article.source}</p>
                                    <p>${article.summary}</p>
                                    <p><strong>Credibility:</strong> <span class="credibility-${article.credibility.toLowerCase()}">${article.credibility}</span></p>
                                </div>
                            `).join('');
                        } else {
                            historyDiv.innerHTML = '<p>No analysis history available</p>';
                        }
                    })
                    .catch(error => {
                        console.error('Error loading analysis history:', error);
                        document.getElementById('analysis-history').innerHTML = '<p>Failed to load analysis history</p>';
                    });
            }

            // Загрузка графика достоверности
            function loadCredibilityChart() {
                fetch('/source-credibility-chart')
                    .then(response => response.json())
                    .then(data => {
                        const chartDiv = document.getElementById('credibility-chart');
                        chartDiv.innerHTML = `
                            <h3>Source Credibility Scores</h3>
                            <ul>
                                ${data.sources.map((source, index) => `
                                    <li>
                                        ${source}: ${data.credibility_scores[index]}
                                        <span class="credibility-${data.credibility_scores[index] >= 0.8 ? 'high' : data.credibility_scores[index] >= 0.6 ? 'medium' : 'low'}">
                                            ${data.credibility_scores[index] >= 0.8 ? 'High' : data.credibility_scores[index] >= 0.6 ? 'Medium' : 'Low'}
                                        </span>
                                    </li>
                                `).join('')}
                            </ul>
                        `;
                    })
                    .catch(error => {
                        console.error('Error loading credibility chart:', error);
                        document.getElementById('credibility-chart').innerHTML = '<p>Failed to load credibility chart</p>';
                    });
            }
        </script>
    </body>
    </html>
    '''.replace('{{ csrf_token }}', csrf_token))

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
    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Feedback</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 20px; }
            .navbar { background-color: #333; color: white; padding: 10px 20px; margin-bottom: 20px; }
            .navbar a { color: white; text-decoration: none; margin-right: 15px; }
            .navbar a:hover { text-decoration: underline; }
            .container { max-width: 800px; margin: 0 auto; }
            textarea { width: 100%; height: 150px; margin-bottom: 10px; }
            button { padding: 10px 15px; background: #4CAF50; color: white; border: none; border-radius: 4px; cursor: pointer; }
            button:hover { background: #45a049; }
        </style>
    </head>
    <body>
        <div class="navbar">
            <a href="/">Home</a>
            <a href="/privacy">Privacy</a>
            <a href="/terms">Terms</a>
        </div>

        <div class="container">
            <h1>Feedback</h1>
            <p>We appreciate your feedback about our media analysis service.</p>

            <form id="feedback-form" method="post">
                <input type="hidden" name="_csrf_token" value="{{ csrf_token }}">
                <div>
                    <label for="name">Name:</label>
                    <input type="text" id="name" name="name">
                </div>
                <div>
                    <label for="email">Email:</label>
                    <input type="email" id="email" name="email">
                </div>
                <div>
                    <label for="feedback">Your Feedback:</label>
                    <textarea id="feedback" name="feedback" required></textarea>
                </div>
                <button type="submit">Submit Feedback</button>
            </form>
        </div>

        <script>
            document.getElementById('feedback-form').addEventListener('submit', function(e) {
                e.preventDefault();
                alert('Thank you for your feedback!');
                this.reset();
            });
        </script>
    </body>
    </html>
    '''.replace('{{ csrf_token }}', csrf_token))

@app.route('/privacy')
def privacy():
    """Страница политики конфиденциальности"""
    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Privacy Policy</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 20px; line-height: 1.6; }
            .navbar { background-color: #333; color: white; padding: 10px 20px; margin-bottom: 20px; }
            .navbar a { color: white; text-decoration: none; margin-right: 15px; }
            .navbar a:hover { text-decoration: underline; }
            .container { max-width: 800px; margin: 0 auto; }
            h1, h2 { color: #333; }
        </style>
    </head>
    <body>
        <div class="navbar">
            <a href="/">Home</a>
            <a href="/feedback">Feedback</a>
            <a href="/terms">Terms</a>
        </div>

        <div class="container">
            <h1>Privacy Policy</h1>

            <h2>1. Information We Collect</h2>
            <p>We collect information you provide directly to us, such as when you submit content for analysis.</p>

            <h2>2. How We Use Your Information</h2>
            <p>We use the information we collect to provide and improve our services, and to respond to your requests.</p>

            <h2>3. Information Sharing</h2>
            <p>We do not share your personal information with third parties except as described in this policy.</p>

            <h2>4. Security</h2>
            <p>We take reasonable measures to help protect your information from loss, theft, misuse and unauthorized access.</p>

            <h2>5. Changes to This Policy</h2>
            <p>We may update this privacy policy from time to time. We will notify you of any changes by posting the new policy on this page.</p>

            <p>Last updated: June 2023</p>
        </div>
    </body>
    </html>
    ''')

@app.route('/terms')
def terms():
    """Страница условий использования"""
    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Terms of Service</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 20px; line-height: 1.6; }
            .navbar { background-color: #333; color: white; padding: 10px 20px; margin-bottom: 20px; }
            .navbar a { color: white; text-decoration: none; margin-right: 15px; }
            .navbar a:hover { text-decoration: underline; }
            .container { max-width: 800px; margin: 0 auto; }
            h1, h2 { color: #333; }
        </style>
    </head>
    <body>
        <div class="navbar">
            <a href="/">Home</a>
            <a href="/feedback">Feedback</a>
            <a href="/privacy">Privacy</a>
        </div>

        <div class="container">
            <h1>Terms of Service</h1>

            <h2>1. Acceptance of Terms</h2>
            <p>By using our service, you agree to these Terms of Service.</p>

            <h2>2. Use of Service</h2>
            <p>You agree to use our service only for lawful purposes and in accordance with these Terms.</p>

            <h2>3. Intellectual Property</h2>
            <p>The service and its original content are owned by us and are protected by international copyright laws.</p>

            <h2>4. Disclaimer</h2>
            <p>Our service is provided "as is" without any warranties of any kind.</p>

            <h2>5. Limitation of Liability</h2>
            <p>In no event shall we be liable for any indirect, incidental, special, consequential or punitive damages.</p>

            <h2>6. Changes to Terms</h2>
            <p>We reserve the right to modify these terms at any time. We will notify you of any changes by posting the new terms on this page.</p>

            <p>Last updated: June 2023</p>
        </div>
    </body>
    </html>
    ''')

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
