from flask import Flask, request, jsonify, render_template, session, make_response, abort
from werkzeug.middleware.proxy_fix import ProxyFix
import logging
from logging.handlers import RotatingFileHandler
import os
import sqlite3
import re
import json
import requests
import html
import uuid
import time
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse, urlunparse
import anthropic
from newspaper import Article, Config
from newsapi import NewsApiClient
import feedparser
from gnews import GNews
import html2text
from bs4 import BeautifulSoup
from flask_cors import CORS
from functools import wraps
import plotly.graph_objects as go
import plotly.express as px

# Initialize Flask application
app = Flask(__name__, static_folder='static', template_folder='templates')
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)
app.secret_key = os.getenv('SECRET_KEY', str(uuid.uuid4()))

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler('app.log', maxBytes=1000000, backupCount=3),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Environment variables
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY', 'mock-key')
MODEL_NAME = os.getenv('ANTHROPIC_MODEL', 'claude-3-opus-20240229')
NEWS_API_KEY = os.getenv('NEWS_API_KEY', 'mock-key')
NEWS_API_ENABLED = bool(NEWS_API_KEY and NEWS_API_KEY != 'mock-key')

# Content Extractor Class
class ContentExtractor:
    def __init__(self):
        self.config = Config()
        self.config.browser_user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        self.config.request_timeout = 20
        self.h2t = html2text.HTML2Text()
        self.h2t.ignore_links = True

    def extract_content(self, url_or_text):
        """Main content extraction method"""
        if url_or_text.startswith(('http://', 'https://')):
            return self.extract_from_url(url_or_text)
        return self.extract_from_text(url_or_text)

    def extract_from_url(self, url):
        """Extract content from URL using multiple methods"""
        methods = [
            self._try_newspaper,
            self._try_requests_bs4,
            self._try_readability
        ]
        
        errors = []
        for method in methods:
            try:
                content = method(url)
                if content and len(content.strip()) > 100:
                    return self._clean_content(content)
            except Exception as e:
                errors.append(f"{method.__name__}: {str(e)}")
                continue
                
        logger.error(f"All extraction methods failed for {url}: {errors}")
        return None

    def _try_newspaper(self, url):
        """Method 1: Using newspaper3k"""
        article = Article(url, config=self.config)
        article.download()
        article.parse()
        return article.text

    def _try_requests_bs4(self, url):
        """Method 2: Direct parsing with BeautifulSoup"""
        headers = {
            'User-Agent': self.config.browser_user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        }
        response = requests.get(url, headers=headers, timeout=self.config.request_timeout)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove unwanted elements
        for tag in soup.find_all(['script', 'style', 'nav', 'header', 'footer', 'ads']):
            tag.decompose()
            
        # Try to find main content
        content = None
        for selector in [
            'article', 'main', '.article-body', '.story-body',
            '[role="main"]', '.content-body', '.story-content'
        ]:
            content = soup.select_one(selector)
            if content:
                break
                
        return content.get_text(' ', strip=True) if content else None

    def _try_readability(self, url):
        """Method 3: Using readability"""
        response = requests.get(url, headers={'User-Agent': self.config.browser_user_agent})
        from readability import Document
        doc = Document(response.text)
        return self.h2t.handle(doc.summary())

    def _clean_content(self, content):
        """Clean and format extracted content"""
        if not content:
            return None
            
        # Remove extra spaces and newlines
        content = re.sub(r'\s+', ' ', content)
        content = content.strip()
        
        # Remove advertising markers and typical unwanted phrases
        content = re.sub(r'Advertisement|Sponsored Content|Read more:', '', content)
        
        # Remove email addresses
        content = re.sub(r'\S+@\S+', '', content)
        
        # Remove URLs
        content = re.sub(r'http\S+', '', content)
        
        return content.strip()

    def extract_from_text(self, text):
        """Process direct text input"""
        return self._clean_content(text)

# News Extractor Class
class NewsExtractor:
    def __init__(self):
        self.newsapi = NewsApiClient(api_key=NEWS_API_KEY) if NEWS_API_KEY else None
        self.google_news = GNews(language='en', country='US', max_results=3)
        self.h2t = html2text.HTML2Text()
        self.h2t.ignore_links = True
def extract_from_all_sources(self, query):
        """Extract news from all available sources"""
        results = []
        
        # NewsAPI
        if self.newsapi:
            try:
                newsapi_results = self.newsapi.get_everything(
                    q=query,
                    language='en',
                    sort_by='relevancy',
                    page_size=3
                )
                results.extend(self._format_newsapi_results(newsapi_results))
            except Exception as e:
                logger.error(f"NewsAPI error: {e}")

        # Google News
        try:
            gnews_results = self.google_news.get_news(query)
            results.extend(self._format_gnews_results(gnews_results))
        except Exception as e:
            logger.error(f"GNews error: {e}")

        # RSS Feeds
        try:
            rss_results = self._get_rss_news(query)
            results.extend(rss_results)
        except Exception as e:
            logger.error(f"RSS error: {e}")

        return self._deduplicate_results(results)

    def _get_rss_news(self, query):
        """Get news from RSS feeds"""
        rss_feeds = {
            'Reuters': 'http://feeds.reuters.com/reuters/topNews',
            'Associated Press': 'https://feeds.ap.org/rss/topnews',
            'BBC': 'http://feeds.bbci.co.uk/news/rss.xml'
        }
        
        results = []
        for source, url in rss_feeds.items():
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:2]:
                    if query.lower() in entry.title.lower() or \
                       (hasattr(entry, 'description') and query.lower() in entry.description.lower()):
                        results.append({
                            'title': entry.title,
                            'url': entry.link,
                            'source': source,
                            'date': entry.published if hasattr(entry, 'published') else None,
                            'description': entry.description if hasattr(entry, 'description') else None
                        })
            except Exception as e:
                logger.error(f"RSS feed error for {source}: {e}")
                
        return results

    def _format_newsapi_results(self, results):
        """Format results from NewsAPI"""
        formatted = []
        for article in results.get('articles', []):
            formatted.append({
                'title': article.get('title'),
                'url': article.get('url'),
                'source': article.get('source', {}).get('name'),
                'date': article.get('publishedAt'),
                'description': article.get('description')
            })
        return formatted

    def _format_gnews_results(self, results):
        """Format results from Google News"""
        formatted = []
        for article in results:
            formatted.append({
                'title': article.get('title'),
                'url': article.get('url'),
                'source': article.get('publisher', {}).get('name'),
                'date': article.get('published date'),
                'description': article.get('description')
            })
        return formatted

    def _deduplicate_results(self, results):
        """Remove duplicates based on URL"""
        seen_urls = set()
        unique_results = []
        
        for result in results:
            if result['url'] not in seen_urls:
                seen_urls.add(result['url'])
                unique_results.append(result)
                
        return unique_results

class NewsTrustAnalyzer:
    def __init__(self):
        self.content_extractor = ContentExtractor()
        self.news_extractor = NewsExtractor()
        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        
        # Predefined trust scores for known sources
        self.trusted_sources = {
            'reuters.com': 0.95,
            'apnews.com': 0.95,
            'bloomberg.com': 0.93,
            'ft.com': 0.92,
            'wsj.com': 0.91,
            'economist.com': 0.90,
            'bbc.com': 0.90
        }

    def analyze(self, url_or_text, source=None):
        """Main analysis method"""
        try:
            # Extract content
            content = self.content_extractor.extract_content(url_or_text)
            if not content:
                raise ValueError("Could not extract content")

            # Get analysis from Claude
            analysis = self.analyze_with_claude(content, source)

            # Get related news
            related_news = self.news_extractor.extract_from_all_sources(
                analysis.get('topics', ['news'])[0]
            )

            return {
                'content': content,
                'analysis': analysis,
                'related_news': related_news
            }

        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}", exc_info=True)
            return {
                'error': str(e),
                'content': None,
                'analysis': None,
                'related_news': []
            }

    def analyze_with_claude(self, content, source=None):
        """Analyze content using Claude API"""
        try:
            prompt = f"""Analyze this news article for credibility, bias, and factual accuracy.
            Return results in JSON format with these fields:
            news_integrity (0.0-1.0), fact_check_needed_score (0.0-1.0),
            sentiment_score (0.0-1.0), bias_score (0.0-1.0),
            topics (list), key_arguments (list), mentioned_facts (list),
            author_purpose (string), potential_biases_identified (list),
            short_summary (string), index_of_credibility (0.0-1.0).

            Article: {content[:10000]}  # Limiting content length
            Source: {source if source else 'unknown'}
            """

            response = self.client.messages.create(
                model=MODEL_NAME,
                max_tokens=2000,
                temperature=0.2,
                messages=[{"role": "user", "content": prompt}]
            )

            analysis = json.loads(response.content[0].text)
            
            # Adjust score based on trusted sources
            if source in self.trusted_sources:
                analysis['index_of_credibility'] = max(
                    analysis['index_of_credibility'],
                    self.trusted_sources[source] * 0.8
                )

            return analysis

        except Exception as e:
            logger.error(f"Claude analysis failed: {str(e)}", exc_info=True)
            return self._generate_mock_analysis()

    def _generate_mock_analysis(self):
        """Generate mock analysis when API fails"""
        return {
            'news_integrity': 0.7,
            'fact_check_needed_score': 0.3,
            'sentiment_score': 0.5,
            'bias_score': 0.3,
            'topics': ['news', 'general'],
            'key_arguments': ['Mock analysis generated due to API error'],
            'mentioned_facts': ['Unable to analyze actual content'],
            'author_purpose': 'Unknown (analysis failed)',
            'potential_biases_identified': ['Analysis unavailable'],
            'short_summary': 'Analysis failed - using mock data',
            'index_of_credibility': 0.7
        }
    def generate_credibility_gauge(self, score):
    """Generate gauge chart for credibility score"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score * 100,
        title = {'text': "Credibility Score"},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 40], 'color': "lightgray"},
                {'range': [40, 70], 'color': "gray"},
                {'range': [70, 100], 'color': "darkgray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': score * 100
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=40, b=10)
    )
    return fig

def generate_sources_chart(self):
    """Generate bar chart for news sources credibility"""
    sources = list(self.trusted_sources.keys())
    scores = list(self.trusted_sources.values())
    
    fig = go.Figure(data=[
        go.Bar(
            x=sources,
            y=[score * 100 for score in scores],
            marker_color='darkblue'
        )
    ])
    
    fig.update_layout(
        title="News Sources Credibility Scores",
        xaxis_title="Source",
        yaxis_title="Credibility Score (%)",
        yaxis_range=[0, 100],
        height=400,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    return fig
# API Routes
@app.route('/api/analyze', methods=['POST'])
def analyze_content():
    """API endpoint for content analysis"""
    try:
        data = request.get_json()
        if not data or 'content' not in data:
            return jsonify({
                'error': 'No content provided',
                'status': 400
            }), 400

        analyzer = NewsTrustAnalyzer()
        result = analyzer.analyze(
            data['content'],
            data.get('source')
        )

        # Generate charts
        credibility_chart = analyzer.generate_credibility_gauge(
            result['analysis']['index_of_credibility']
        )
        sources_chart = analyzer.generate_sources_chart()

        # Convert charts to JSON
        result['charts'] = {
            'credibility': credibility_chart.to_json(),
            'sources': sources_chart.to_json()
        }

        return jsonify(result)

    except Exception as e:
        logger.error(f"Analysis request failed: {str(e)}", exc_info=True)
        return jsonify({
            'error': str(e),
            'status': 500
        }), 500
@app.route('/api/extract', methods=['POST'])
def extract_from_url():
    """API endpoint for URL content extraction"""
    try:
        data = request.get_json()
        if not data or 'url' not in data:
            return jsonify({
                'error': 'No URL provided',
                'status': 400
            }), 400

        extractor = ContentExtractor()
        content = extractor.extract_content(data['url'])

        if not content:
            return jsonify({
                'error': 'Could not extract content',
                'status': 400
            }), 400

        return jsonify({
            'content': content,
            'status': 200
        })

    except Exception as e:
        logger.error(f"Extraction request failed: {str(e)}", exc_info=True)
        return jsonify({
            'error': str(e),
            'status': 500
        }), 500

@app.route('/api/related-news', methods=['POST'])
def get_related_news():
    """API endpoint for finding related news"""
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({
                'error': 'No query provided',
                'status': 400
            }), 400

        extractor = NewsExtractor()
        results = extractor.extract_from_all_sources(data['query'])

        return jsonify({
            'articles': results,
            'status': 200
        })

    except Exception as e:
        logger.error(f"Related news request failed: {str(e)}", exc_info=True)
        return jsonify({
            'error': str(e),
            'status': 500
        }), 500

# Error Handlers
@app.errorhandler(404)
def not_found_error(error):
    return jsonify({
        'error': 'Not found',
        'status': 404
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'error': 'Internal server error',
        'status': 500
    }), 500

# CORS Configuration
CORS(app, resources={
    r"/api/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Security Headers
@app.after_request
def add_security_headers(response):
    response.headers['Content-Security-Policy'] = "default-src 'self'"
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    return response

# Health Check
@app.route('/faq')
def faq():
    return render_template('faq.html')

@app.route('/feedback', methods=['GET', 'POST'])
def feedback():
    if request.method == 'POST':
        feedback_data = request.form.to_dict()
        # Здесь можно добавить сохранение feedback в базу данных
        return jsonify({'status': 'success'})
    return render_template('feedback.html')
@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'version': '1.0.0'
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
