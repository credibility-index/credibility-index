import os
import logging
import re
import json
import socket
import time
from datetime import datetime, timedelta
from urllib.parse import urlparse
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import anthropic
from newspaper import Article, Config
import hashlib
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from app.claude_api import ClaudeAPI

# Инициализация приложения
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Конфигурация библиотеки newspaper
config = Config()
config.browser_user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36'
config.request_timeout = 30

# Настройка повторных попыток для запросов
session = requests.Session()
retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504, 408, 429])
session.mount('http://', HTTPAdapter(max_retries=retries))
session.mount('https://', HTTPAdapter(max_retries=retries))
# В вашем main.py
claude_api = ClaudeAPI()
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

analysis_history = [
    {
        "title": "Analysis of recent Middle East developments",
        "source": "Media Analysis",
        "url": "https://example.com/article1",
        "summary": "Analysis of recent events...",
        "credibility": "High"
    },
    {
        "title": "Economic impact of recent conflicts",
        "source": "Financial Times",
        "url": "https://example.com/article2",
        "summary": "Economic analysis...",
        "credibility": "Medium"
    }
]

@app.route('/')
def index():
    """Главная страница приложения"""
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
            button { padding: 10px 15px; background: #4CAF50; color: white; border: none; border-radius: 4px; cursor: pointer; }
            button:hover { background: #45a049; }
            .credibility-high { color: green; }
            .credibility-medium { color: orange; }
            .credibility-low { color: red; }
        </style>
    </head>
    <body>
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
                <textarea id="article-input" placeholder="Enter article URL or text..."></textarea>
                <button onclick="analyzeArticle()">Analyze</button>
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

        <script>
            // Загрузка данных при старте
            document.addEventListener('DOMContentLoaded', function() {
                loadDailyBuzz();
                loadAnalysisHistory();
                loadCredibilityChart();
            });

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

            // Анализ статьи
            function analyzeArticle() {
                const input = document.getElementById('article-input').value.trim();
                if (!input) {
                    alert('Please enter article URL or text');
                    return;
                }

                const resultsDiv = document.getElementById('analysis-results');
                resultsDiv.innerHTML = '<p>Analyzing...</p>';

                fetch('/analyze', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ input_text: input })
                })
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'success') {
                        const article = data.article;
                        resultsDiv.innerHTML = `
                            <div class="article">
                                <h2>${article.title}</h2>
                                <p><strong>Source:</strong> ${article.source}</p>
                                <p>${article.short_summary}</p>
                                <p><strong>Credibility:</strong> <span class="credibility-${article.credibility_level.toLowerCase()}">${article.credibility_level}</span></p>
                                <h3>Analysis:</h3>
                                <p><strong>Topics:</strong> ${article.analysis.topics.map(t => t.name).join(', ')}</p>
                            </div>
                        `;

                        // Загрузка похожих статей
                        if (data.similar_articles && data.similar_articles.length > 0) {
                            const similarDiv = document.getElementById('similar-articles');
                            similarDiv.innerHTML = data.similar_articles.map(article => `
                                <div class="article">
                                    <h3><a href="${article.url}" target="_blank">${article.title}</a></h3>
                                    <p><strong>Source:</strong> ${article.source}</p>
                                    <p>${article.summary}</p>
                                    <p><strong>Credibility:</strong> <span class="credibility-${article.credibility.toLowerCase()}">${article.credibility}</span></p>
                                </div>
                            `).join('');
                        }
                    } else {
                        resultsDiv.innerHTML = `<p>Error: ${data.message}</p>`;
                    }
                })
                .catch(error => {
                    console.error('Error analyzing article:', error);
                    resultsDiv.innerHTML = '<p>Failed to analyze article</p>';
                });
            }
        </script>
    </body>
    </html>
    ''')

@app.route('/daily-buzz')
def get_daily_buzz():
    """Возвращает статью дня"""
    return jsonify(daily_buzz)

@app.route('/source-credibility-chart')
def get_source_credibility_chart():
    """Возвращает данные для графика достоверности источников"""
    return jsonify(source_credibility_data)

@app.route('/analysis-history')
def get_analysis_history():
    """Возвращает историю анализов"""
    return jsonify({"history": analysis_history})

@app.route('/analyze', methods=['POST'])
def analyze_article():
    """Анализирует статью"""
    try:
        data = request.get_json()
        if not data or 'input_text' not in data:
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
                    'message': 'Content is too short for analysis'
                }), 400
            content = input_text
            source = 'Direct Input'
            title = 'User-provided Text'

        analysis = analyze_with_claude(content, source)
        credibility_level = determine_credibility_level(analysis.get('credibility_score', {}).get('score', 0.6))

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
            'article': {
                'title': title,
                'source': source,
                'url': input_text if input_text.startswith(('http://', 'https://')) else None,
                'short_summary': content[:200] + '...' if len(content) > 200 else content,
                'analysis': analysis,
                'credibility_level': credibility_level
            },
            'similar_articles': get_similar_articles(analysis.get('topics', []))
        })

    except Exception as e:
        logger.error(f"Error analyzing article: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': 'An unexpected error occurred during analysis',
            'details': str(e)
        }), 500

def extract_text_from_url(url: str) -> tuple:
    """Извлекает текст из URL"""
    try:
        parsed = urlparse(url)
        if not all([parsed.scheme, parsed.netloc]):
            return None, None, None, "Invalid URL format"

        clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"

        if any(domain in parsed.netloc for domain in ['youtube.com', 'vimeo.com', 'twitch.tv']):
            return None, parsed.netloc.replace('www.', ''), "Video content detected", None

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

        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            }

            response = session.get(clean_url, headers=headers, timeout=15)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')
            for element in soup(['script', 'style', 'noscript', 'iframe', 'svg', 'nav', 'footer', 'header']):
                element.decompose()

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

def analyze_with_claude(content: str, source: str) -> dict:
    """Анализирует статью с помощью Claude"""
    try:
        if not anthropic_client:
            return get_default_analysis()

        prompt = f"""Analyze this news article and provide a comprehensive JSON response with:
1. Credibility score (0-1) with explanation
2. Key topics (max 5) with brief descriptions
3. Detailed summary (3-5 sentences)
4. Main perspectives (Western, Iranian, Israeli, Neutral - 2-3 sentences each)
5. Sentiment analysis (positive/neutral/negative) with explanation
6. Bias detection (low/medium/high) with explanation
7. Key arguments presented (3-5 main points)
8. Mentioned facts (3-5 key facts)
9. Potential biases identified (list with explanations)
10. Author's purpose (1-2 sentences)

Article content (first 4000 characters):
{content[:4000]}"""

        response = anthropic_client.messages.create(
            model=os.getenv('ANTHROPIC_MODEL', 'claude-3-opus-20240229'),
            max_tokens=2000,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}],
            timeout=30
        )

        response_text = response.content[0].text.strip()

        try:
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                analysis = json.loads(json_match.group(0))
                required_fields = ['credibility_score', 'topics', 'summary', 'perspectives', 'sentiment', 'bias']
                if all(field in analysis for field in required_fields):
                    return analysis
        except Exception:
            pass

        return get_default_analysis()

    except Exception as e:
        logger.error(f"Error analyzing with Claude: {str(e)}")
        return get_default_analysis()

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
    """Возвращает похожие статьи"""
    # В реальном приложении здесь был бы вызов к базе данных или API
    return [
        {
            "title": "Related article about Israel-Iran relations",
            "source": "Media Analysis",
            "summary": "Analysis of recent developments in Israel-Iran relations...",
            "url": "https://example.com/related1",
            "credibility": "High"
        },
        {
            "title": "Middle East economic outlook",
            "source": "Financial Times",
            "summary": "Economic analysis of the Middle East region...",
            "url": "https://example.com/related2",
            "credibility": "Medium"
        }
    ]

def get_default_analysis() -> dict:
    """Возвращает анализ по умолчанию"""
    return {
        "credibility_score": {"score": 0.6, "explanation": "Default credibility score"},
        "topics": [{"name": "general", "description": "General news topic"}],
        "summary": "This article discusses various perspectives on a current event.",
        "perspectives": {
            "western": {"summary": "Western perspective", "key_points": ["Point 1"], "credibility": "Medium"},
            "iranian": {"summary": "Iranian perspective", "key_points": ["Point 1"], "credibility": "Medium"},
            "israeli": {"summary": "Israeli perspective", "key_points": ["Point 1"], "credibility": "Medium"},
            "neutral": {"summary": "Neutral analysis", "key_points": ["Point 1"], "credibility": "Medium"}
        },
        "sentiment": {"score": "neutral", "explanation": "Balanced view"},
        "bias": {"level": "medium", "explanation": "Some bias detected"}
    }

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
