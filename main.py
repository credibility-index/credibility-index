import os
import logging
import sqlite3
import re
import json
import requests
import html
import uuid
import time
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse, urlunparse
from flask import Flask, request, jsonify, render_template, session, make_response
from werkzeug.middleware.proxy_fix import ProxyFix
import anthropic
from newspaper import Article, Config
from stop_words import get_stop_words
from flask_cors import CORS

# Initialize Flask application
app = Flask(__name__, static_folder='static', template_folder='templates')
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)
app.secret_key = os.getenv('SECRET_KEY', str(uuid.uuid4()))

# Configure CORS for Railway
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "Accept"]
    }
})

# Environment variables
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY', 'mock-key')
MODEL_NAME = os.getenv('ANTHROPIC_MODEL', 'claude-3-opus-20240229')
NEWS_API_KEY = os.getenv('NEWS_API_KEY', 'mock-key')
NEWS_API_ENABLED = bool(NEWS_API_KEY and NEWS_API_KEY != 'mock-key')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global constants
predefined_trust_scores = {
    'bbc.com': 0.9, 'reuters.com': 0.95, 'apnews.com': 0.93,
    'nytimes.com': 0.88, 'theguardian.com': 0.85, 'wsj.com': 0.82,
    'cnn.com': 0.70, 'foxnews.com': 0.40, 'aljazeera.com': 0.80
}

TRUSTED_NEWS_SOURCES_IDS = [
    'bbc-news', 'reuters', 'associated-press',
    'the-new-york-times', 'the-guardian-uk'
]

stop_words_en = get_stop_words('en')
user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'

# Database configuration
DB_NAME = 'news_analysis.db'

def get_db_connection():
    """Create database connection"""
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def initialize_database():
    """Initialize database schema and populate with test data if empty"""
    try:
        db_dir = os.path.dirname(DB_NAME)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir)

        with get_db_connection() as conn:
            cursor = conn.cursor()

            # Create tables
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS news (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT,
                    source TEXT,
                    content TEXT,
                    integrity REAL,
                    fact_check REAL,
                    sentiment REAL,
                    bias REAL,
                    credibility_level TEXT,
                    index_of_credibility REAL,
                    url TEXT UNIQUE,
                    analysis_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    short_summary TEXT
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS source_stats (
                    source TEXT PRIMARY KEY,
                    high INTEGER DEFAULT 0,
                    medium INTEGER DEFAULT 0,
                    low INTEGER DEFAULT 0,
                    total_analyzed INTEGER DEFAULT 0
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    email TEXT NOT NULL,
                    type TEXT NOT NULL,
                    message TEXT NOT NULL,
                    date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            conn.commit()

            # Populate with test data if empty
            cursor.execute("SELECT COUNT(*) FROM source_stats")
            if cursor.fetchone()[0] == 0:
                populate_test_data(conn)

            logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        raise

def populate_test_data(conn):
    """Populate database with test data"""
    try:
        cursor = conn.cursor()

        # Test data for various sources
        test_sources = [
            ('bbc.com', 45, 10, 5), ('reuters.com', 50, 5, 2),
            ('foxnews.com', 15, 20, 30), ('cnn.com', 30, 25, 10),
            ('nytimes.com', 35, 15, 5)
        ]

        for source, high, medium, low in test_sources:
            total = high + medium + low
            cursor.execute('''
                INSERT INTO source_stats (source, high, medium, low, total_analyzed)
                VALUES (?, ?, ?, ?, ?)
            ''', (source, high, medium, low, total))

        conn.commit()
        logger.info("Test data added to database successfully")
    except Exception as e:
        logger.error(f"Error populating test data: {str(e)}")
        raise

@app.before_request
def before_request():
    """Set up before each request"""
    if request.path.startswith('/static/'):
        return

    if request.method == 'OPTIONS':
        response = make_response()
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, Accept'
        return response

@app.after_request
def add_security_headers(response):
    """Add security headers"""
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, Accept'
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    return response

@app.route('/')
def index():
    """Home page route"""
    return render_template('index.html')

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'database': 'connected' if os.path.exists(DB_NAME) else 'disconnected',
        'api_keys': {
            'anthropic': 'configured' if ANTHROPIC_API_KEY and ANTHROPIC_API_KEY != 'mock-key' else 'not_configured',
            'news_api': 'configured' if NEWS_API_ENABLED else 'not_configured'
        }
    })

@app.route('/faq')
def faq():
    """FAQ page route"""
    return render_template('faq.html')

@app.route('/feedback', methods=['GET', 'POST'])
def feedback():
    """Feedback page and form handler"""
    if request.method == 'POST':
        try:
            name = request.form.get('name')
            email = request.form.get('email')
            feedback_type = request.form.get('type')
            message = request.form.get('message')

            if not all([name, email, feedback_type, message]):
                return render_template('feedback.html', error="All fields are required")

            if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
                return render_template('feedback.html', error="Invalid email address")

            with get_db_connection() as conn:
                conn.execute('''
                    INSERT INTO feedback (name, email, type, message, date)
                    VALUES (?, ?, ?, ?, ?)
                ''', (name, email, feedback_type, message, datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')))
                conn.commit()

            return render_template('feedback_success.html')
        except Exception as e:
            logger.error(f'Error saving feedback: {str(e)}')
            return render_template('feedback.html', error="Error saving feedback")

    return render_template('feedback.html')

@app.route('/source-credibility-chart', methods=['GET'])
def source_credibility_chart():
    """Endpoint for getting source credibility chart data with mock data"""
    try:
        mock_data = {
            'sources': ['BBC', 'Reuters', 'CNN', 'Fox News', 'The Guardian'],
            'credibility_scores': [0.9, 0.95, 0.7, 0.4, 0.85],
            'high_counts': [45, 50, 30, 15, 40],
            'medium_counts': [10, 5, 25, 20, 10],
            'low_counts': [5, 2, 10, 30, 5],
            'total_counts': [60, 57, 65, 65, 55]
        }
        return jsonify({'status': 'success', 'data': mock_data})
    except Exception as e:
        logger.error(f"Error in source_credibility_chart endpoint: {str(e)}")
        return jsonify({
            'error': 'An error occurred while fetching chart data',
            'status': 500,
            'details': str(e)
        }), 500

@app.route('/analysis-history', methods=['GET'])
def analysis_history():
    """Endpoint for getting analysis history with mock data"""
    try:
        mock_history = [
            {
                'url': 'https://example.com/article1',
                'title': 'Sample Article 1',
                'source': 'BBC',
                'credibility': 'High',
                'summary': 'This is a sample article about current events',
                'date': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')
            },
            {
                'url': 'https://example.com/article2',
                'title': 'Sample Article 2',
                'source': 'Reuters',
                'credibility': 'Medium',
                'summary': 'Another sample article demonstrating functionality',
                'date': (datetime.now(timezone.utc) - timedelta(days=1)).strftime('%Y-%m-%d %H:%M')
            }
        ]
        return jsonify({'status': 'success', 'history': mock_history})
    except Exception as e:
        logger.error(f"Error in analysis_history endpoint: {str(e)}")
        return jsonify({
            'error': 'An error occurred while fetching analysis history',
            'status': 500,
            'details': str(e)
        }), 500

@app.route('/analyze', methods=['POST', 'OPTIONS'])
def analyze():
    """Analyze article endpoint with full functionality"""
    if request.method == 'OPTIONS':
        response = make_response()
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Accept'
        return response

    try:
        # Validate request
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON', 'status': 400}), 400

        data = request.get_json()
        if not data or 'input_text' not in data:
            return jsonify({'error': 'Missing input text', 'status': 400}), 400

        input_text = data['input_text'].strip()
        source_name = data.get('source_name_manual', 'Direct Input').strip()

        if not input_text:
            return jsonify({'error': 'Empty input text', 'status': 400}), 400

        # Process article
        if input_text.startswith(('http://', 'https://')):
            content, source, title = extract_text_from_url(input_text)
            if not content:
                return jsonify({
                    'error': 'Could not extract article content',
                    'status': 400,
                    'suggestions': [
                        'Check if the URL is correct',
                        'Try a different URL',
                        'Make sure the website allows scraping',
                        'Alternatively, paste the article text directly'
                    ]
                }), 400
        else:
            if len(input_text) < 50:
                return jsonify({
                    'error': 'Content too short',
                    'status': 400,
                    'details': 'Minimum 50 characters required'
                }), 400
            content = input_text
            title = 'User-provided Text'
            source = source_name

        # Analyze content using Claude API
        analysis = analyze_with_claude(content, source)

        # Save to database
        credibility = save_analysis(
            input_text if input_text.startswith(('http://', 'https://')) else None,
            title, source, content, analysis
        )

        # Get similar articles (first page)
        similar_articles = fetch_same_topic_articles(analysis, page=1)
        similar_articles_html = render_same_topic_articles_html(similar_articles)

        # Store current analysis in session for pagination
        session['current_analysis'] = analysis
        session['current_page'] = 1

        # Format response
        response_data = {
            'status': 'success',
            'analysis': analysis,
            'credibility': credibility,
            'title': title,
            'source': source,
            'scores_for_chart': {
                'news_integrity': analysis.get('news_integrity', 0.5),
                'fact_check_needed_score': analysis.get('fact_check_needed_score', 0.5),
                'sentiment_score': analysis.get('sentiment_score', 0.5),
                'bias_score': analysis.get('bias_score', 0.5),
                'index_of_credibility': analysis.get('index_of_credibility', 0.5)
            },
            'source_credibility_data': {
                'sources': ['BBC', 'Reuters', 'CNN'],
                'credibility_scores': [0.9, 0.95, 0.7],
                'high_counts': [45, 50, 30],
                'medium_counts': [10, 5, 25],
                'low_counts': [5, 2, 10],
                'total_counts': [60, 57, 65]
            },
            'same_topic_html': similar_articles_html,
            'show_more_button': len(similar_articles) > 0,
            'output': format_analysis_results(title, source, analysis, credibility)
        }

        return jsonify(response_data)

    except Exception as e:
        logger.error(f"Error in analyze endpoint: {str(e)}", exc_info=True)
        return jsonify({
            'error': str(e),
            'status': 500,
            'details': 'An unexpected error occurred during analysis'
        }), 500

@app.route('/more-articles', methods=['GET'])
def more_articles():
    """Endpoint for loading more similar articles"""
    try:
        current_page = session.get('current_page', 1) + 1
        analysis = session.get('current_analysis', {})

        if not analysis:
            return jsonify({
                'error': 'No current analysis found',
                'status': 400
            }), 400

        # Get next page of similar articles
        similar_articles = fetch_same_topic_articles(analysis, page=current_page)
        similar_articles_html = render_same_topic_articles_html(similar_articles)

        # Update session
        session['current_page'] = current_page

        return jsonify({
            'status': 'success',
            'same_topic_html': similar_articles_html,
            'show_more_button': len(similar_articles) > 0
        })

    except Exception as e:
        logger.error(f"Error in more_articles endpoint: {str(e)}")
        return jsonify({
            'error': str(e),
            'status': 500
        }), 500

def extract_text_from_url(url):
    """Extract text from URL with improved error handling"""
    try:
        parsed = urlparse(url)
        if not all([parsed.scheme, parsed.netloc]):
            return None, None, None

        clean_url = urlunparse(parsed._replace(
            scheme=parsed.scheme.lower(),
            netloc=parsed.netloc.lower()
        ))

        if any(domain in clean_url for domain in ['youtube.com', 'vimeo.com', 'twitch.tv']):
            return "Video content detected", parsed.netloc.replace('www.', ''), "Video: " + clean_url

        try:
            response = requests.head(clean_url, timeout=10, allow_redirects=True, headers={'User-Agent': user_agent})
            if response.status_code != 200:
                return None, None, None
        except requests.RequestException:
            return None, None, None

        article = Article(clean_url, config=Config(browser_user_agent=user_agent))
        try:
            article.download()
            if article.download_state != 2:
                return None, None, None
        except Exception:
            return None, None, None

        try:
            article.parse()
            if not article.text or len(article.text.strip()) < 100:
                return None, None, None
        except Exception:
            return None, None, None

        domain = parsed.netloc.replace('www.', '')
        title = article.title.strip() if article.title else "No title"
        return article.text.strip(), domain, title

    except Exception:
        return None, None, None

def analyze_with_claude(content, source):
    """Analyze article text using Claude API with fallback to mock data"""
    try:
        if not ANTHROPIC_API_KEY or ANTHROPIC_API_KEY == 'mock-key':
            logger.info("Using mock data for Claude API")
            return generate_mock_analysis(source)

        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

        max_chars = 10000
        if len(content) > max_chars:
            content = content[:max_chars]
            logger.warning(f"Article content truncated to {max_chars} characters")

        prompt = f"""
        Analyze this news article for credibility, bias, and factual accuracy.
        Return results in JSON format with these fields:
        news_integrity (0.0-1.0), fact_check_needed_score (0.0-1.0),
        sentiment_score (0.0-1.0), bias_score (0.0-1.0),
        topics (list), key_arguments (list), mentioned_facts (list),
        author_purpose (string), potential_biases_identified (list),
        short_summary (string), index_of_credibility (0.0-1.0).

        Article: {content}
        Source: {source}
        """

        try:
            response = client.messages.create(
                model=MODEL_NAME,
                max_tokens=2000,
                temperature=0.2,
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = response.content[0].text
            json_match = re.search(r'```json\s*(\{.*\})\s*```', response_text, re.DOTALL)

            if json_match:
                return json.loads(json_match.group(1))
            else:
                try:
                    return json.loads(response_text)
                except json.JSONDecodeError:
                    logger.error("Failed to parse API response as JSON")
                    return generate_mock_analysis(source)

        except Exception as e:
            logger.error(f"Error calling Claude API: {str(e)}")
            return generate_mock_analysis(source)

    except Exception as e:
        logger.error(f"Unexpected error in analyze_with_claude: {str(e)}")
        return generate_mock_analysis(source)

def generate_mock_analysis(source):
    """Generate mock analysis data"""
    credibility_scores = {
        'bbc.com': 0.9,
        'reuters.com': 0.95,
        'apnews.com': 0.9,
        'nytimes.com': 0.85,
        'theguardian.com': 0.8
    }

    base_score = credibility_scores.get(source, 0.7)
    return {
        "news_integrity": base_score * 0.9 + 0.1,
        "fact_check_needed_score": 1.0 - base_score * 0.8,
        "sentiment_score": 0.5 + (0.1 if "positive" in source else -0.1),
        "bias_score": 1.0 - base_score * 0.9,
        "topics": ["news", "analysis", "sample", "demo"],
        "key_arguments": [
            "This is a sample analysis",
            "Showing how the system works",
            "Demonstrating functionality"
        ],
        "mentioned_facts": ["Sample fact 1", "Sample fact 2"],
        "author_purpose": "To demonstrate the system functionality",
        "potential_biases_identified": ["Sample bias detection"],
        "short_summary": "This is a sample analysis demonstrating how the system would analyze a real article.",
        "index_of_credibility": base_score * 0.9
    }

def calculate_credibility(integrity, fact_check, sentiment, bias):
    """Calculate credibility level"""
    fact_check_score = 1.0 - fact_check
    sentiment_score = 1.0 - abs(sentiment - 0.5) * 2
    bias_score = 1.0 - bias
    score = (integrity * 0.45) + (fact_check_score * 0.35) + (sentiment_score * 0.10) + (bias_score * 0.10)

    if score >= 0.75:
        return 'High'
    if score >= 0.5:
        return 'Medium'
    return 'Low'

def save_analysis(url, title, source, content, analysis):
    """Save analysis to database"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()

            integrity = analysis.get('news_integrity', 0.5)
            fact_check = analysis.get('fact_check_needed_score', 0.5)
            sentiment = analysis.get('sentiment_score', 0.5)
            bias = analysis.get('bias_score', 0.5)
            summary = analysis.get('short_summary', 'No summary available')
            credibility = analysis.get('index_of_credibility', 0.5)
            level = calculate_credibility(integrity, fact_check, sentiment, bias)
            db_url = url if url else f'text_{datetime.now(timezone.utc).timestamp()}'

            cursor.execute('''
                INSERT INTO news
                (url, title, source, content, integrity, fact_check, sentiment, bias,
                credibility_level, short_summary, index_of_credibility)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (db_url, title, source, content, integrity, fact_check,
                  sentiment, bias, level, summary, credibility))

            conn.commit()
            return level
    except Exception as e:
        logger.error(f"Error saving analysis: {str(e)}")
        return 'Medium'  # Default credibility level

def generate_query_from_analysis(analysis_result):
    """Generate search query from analysis results"""
    topics = analysis_result.get('topics', [])
    key_arguments = analysis_result.get('key_arguments', [])

    terms = []
    for phrase in topics + key_arguments:
        if not phrase or not isinstance(phrase, str):
            continue
        if ' ' in phrase and len(phrase.split()) > 1:
            terms.append(f'"{phrase}"')
        else:
            terms.append(phrase)

    unique_terms = list(set(terms))
    filtered_terms = [t for t in unique_terms if t.lower() not in stop_words_en]

    if len(filtered_terms) > 3:
        return ' AND '.join(filtered_terms[:5])
    elif filtered_terms:
        return ' OR '.join(filtered_terms)
    else:
        return 'news OR current events'

def fetch_same_topic_articles(analysis_result, page=1, per_page=3):
    """Fetch similar articles using News API with fallback to mock data"""
    try:
        if not NEWS_API_ENABLED:
            logger.info("News API not enabled, returning mock data")
            return generate_mock_articles(per_page)

        query = generate_query_from_analysis(analysis_result)
        end_date = datetime.now(timezone.utc).date()
        start_date = end_date - timedelta(days=7)

        params = {
            'q': query,
            'apiKey': NEWS_API_KEY,
            'language': 'en',
            'pageSize': per_page,
            'page': page,
            'sortBy': 'relevancy',
            'from': start_date.strftime('%Y-%m-%d'),
            'to': end_date.strftime('%Y-%m-%d')
        }

        if TRUSTED_NEWS_SOURCES_IDS:
            params['sources'] = ','.join(TRUSTED_NEWS_SOURCES_IDS)

        try:
            response = requests.get(
                'https://newsapi.org/v2/everything',
                params=params,
                timeout=15
            )
            response.raise_for_status()
            articles = response.json().get('articles', [])

            if not articles and query != 'news OR current events':
                broader_query = ' OR '.join([f'"{term}"' if ' ' in term else term
                                          for term in analysis_result.get('topics', [])[:3]
                                          if term and term not in stop_words_en])
                if not broader_query:
                    broader_query = 'news OR current events'

                params['q'] = broader_query
                response = requests.get(
                    'https://newsapi.org/v2/everything',
                    params=params,
                    timeout=15
                )
                response.raise_for_status()
                articles.extend(response.json().get('articles', []))

            # Filter and rank articles
            unique_articles = {}
            for article in articles:
                if article.get('url'):
                    unique_articles[article['url']] = article

            articles = list(unique_articles.values())
            all_query_terms = query.split()
            ranked_articles = []

            for article in articles:
                source_domain = urlparse(article.get('url', '')).netloc.replace('www.', '')
                trust_score = predefined_trust_scores.get(source_domain, 0.5)

                article_text = (article.get('title', '') + ' ' + article.get('description', '')).lower()
                relevance_score = sum(1 for term in all_query_terms if term.lower() in article_text)
                final_score = (relevance_score * 10) + (trust_score * 5)
                ranked_articles.append((article, final_score))

            ranked_articles.sort(key=lambda x: x[1], reverse=True)
            return [article for article, score in ranked_articles[:per_page]]

        except requests.RequestException as e:
            logger.error(f"News API request failed: {str(e)}")
            return generate_mock_articles(per_page)

    except Exception as e:
        logger.error(f"Error fetching similar articles: {str(e)}")
        return generate_mock_articles(per_page)

def generate_mock_articles(count):
    """Generate mock articles for when API is not available"""
    return [
        {
            'title': f"Sample Article {i+1} on Similar Topic",
            'url': f"https://example.com/article{i+1}",
            'source': {'name': f"Example News {i+1}"},
            'publishedAt': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
            'description': f"This is a sample article about a similar topic to demonstrate functionality.",
            'trust_score': 0.7 + (i * 0.05)
        }
        for i in range(count)
    ]

def render_same_topic_articles_html(articles):
    """Render HTML for similar articles with show more button"""
    if not articles:
        return '<div class="alert alert-info">No similar articles found</div>'

    html_items = []
    for art in articles:
        title = html.escape(art.get('title', 'No Title'))
        article_url = html.escape(art.get('url', '#'))
        source_name = html.escape(art.get('source', {}).get('name', 'Unknown Source'))
        published_at = html.escape(art.get('publishedAt', 'N/A').split('T')[0])
        description = html.escape(art.get('description', 'No description available.'))
        trust_score = art.get('trust_score', 0.7)

        html_items.append(f'''
        <div class="similar-article">
            <h4><a href="{article_url}" target="_blank">{title}</a></h4>
            <div class="article-meta">
                <span class="article-source">{source_name}</span>
                <span class="article-date">{published_at}</span>
                <span class="article-credibility">Credibility: {int(trust_score*100)}%</span>
            </div>
            <p class="article-description">{description}</p>
        </div>
        ''')

    return '<div class="similar-articles-container">' + ''.join(html_items) + '</div>'

def format_analysis_results(title, source, analysis, credibility):
    """Format analysis results for display with all sections"""
    try:
        integrity = analysis.get('news_integrity', 0.5)
        fact_check = analysis.get('fact_check_needed_score', 0.5)
        sentiment = analysis.get('sentiment_score', 0.5)
        bias = analysis.get('bias_score', 0.5)
        summary = html.escape(analysis.get('short_summary', 'No summary available'))
        topics = [html.escape(str(t)) for t in analysis.get('topics', [])]
        key_args = [html.escape(str(a)) for a in analysis.get('key_arguments', [])]
        biases = [html.escape(str(b)) for b in analysis.get('potential_biases_identified', [])]

        topics_html = ' '.join(f'<span class="badge bg-primary">{topic}</span>' for topic in topics)
        key_args_html = ''.join(f'<li>{arg}</li>' for arg in key_args)
        biases_html = ''.join(f'<li>{bias}</li>' for bias in biases)

        return {
            'title': title,
            'source': source,
            'credibility': credibility,
            'output_md': f"""
            <div class="analysis-section">
                <h2>Article Information</h2>
                <p><strong>Title:</strong> {html.escape(title)}</p>
                <p><strong>Source:</strong> {html.escape(source)}</p>
                <p><strong>Credibility Level:</strong> <span class="credibility-badge {credibility.lower()}">{credibility}</span></p>
            </div>

            <div class="analysis-section">
                <h2>Analysis Scores</h2>
                <div class="row">
                    <div class="col-md-3">
                        <div class="score-item">
                            <div class="score-name">News Integrity</div>
                            <div class="score-value">{integrity:.2f}</div>
                            <div class="score-description">Overall integrity and trustworthiness</div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="score-item">
                            <div class="score-name">Fact Check Needed</div>
                            <div class="score-value">{fact_check:.2f}</div>
                            <div class="score-description">Likelihood that claims need fact-checking</div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="score-item">
                            <div class="score-name">Sentiment</div>
                            <div class="score-value">{sentiment:.2f}</div>
                            <div class="score-description">Emotional tone (0.0 negative, 0.5 neutral, 1.0 positive)</div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="score-item">
                            <div class="score-name">Bias</div>
                            <div class="score-value">{bias:.2f}</div>
                            <div class="score-description">Degree of perceived bias</div>
                        </div>
                    </div>
                </div>
            </div>

            <div class="analysis-section">
                <h2>Additional Information</h2>
                <div class="detail-item">
                    <h4>Short Summary</h4>
                    <p>{summary}</p>
                </div>

                <div class="detail-item">
                    <h4>Topics</h4>
                    <div class="d-flex flex-wrap gap-2">
                        {topics_html}
                    </div>
                </div>

                <div class="detail-item">
                    <h4>Key Arguments</h4>
                    <ul class="list-unstyled">
                        {key_args_html}
                    </ul>
                </div>

                <div class="detail-item">
                    <h4>Potential Biases Identified</h4>
                    <ul class="list-unstyled">
                        {biases_html}
                    </ul>
                </div>
            </div>
            """
        }
    except Exception as e:
        logger.error(f"Error formatting analysis results: {str(e)}")
        return {
            "error": "Error formatting analysis results",
            "output_md": "<div class='alert alert-danger'>Error displaying analysis results</div>"
        }

if __name__ == '__main__':
    initialize_database()
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
