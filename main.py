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
app.secret_key = os.getenv('SECRET_KEY', 'your-secret-key-here')

# Configure CORS
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "Accept"]
    }
})

# Environment variables
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
MODEL_NAME = os.getenv('ANTHROPIC_MODEL', 'claude-3-opus-20240229')
NEWS_API_KEY = os.getenv('NEWS_API_KEY')
NEWS_API_ENABLED = bool(NEWS_API_KEY)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global constants
predefined_trust_scores = {
    'bbc.com': 0.9, 'bbc.co.uk': 0.9, 'reuters.com': 0.95, 'apnews.com': 0.93,
    'nytimes.com': 0.88, 'theguardian.com': 0.85, 'wsj.com': 0.82,
    'cnn.com': 0.70, 'foxnews.com': 0.40, 'aljazeera.com': 0.80
}

TRUSTED_NEWS_SOURCES_IDS = [
    'bbc-news', 'reuters', 'associated-press', 'the-new-york-times',
    'the-guardian-uk', 'the-wall-street-journal', 'cnn', 'al-jazeera-english'
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
    """Initialize database schema"""
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
            logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        raise

def populate_test_data():
    """Populate database with test data"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()

            # Test data for various sources
            test_sources = [
                ('bbc.com', 45, 10, 5), ('reuters.com', 50, 5, 2),
                ('foxnews.com', 15, 20, 30), ('cnn.com', 30, 25, 10),
                ('nytimes.com', 35, 15, 5), ('theguardian.com', 40, 10, 3),
                ('apnews.com', 48, 5, 2), ('washingtonpost.com', 38, 12, 5),
                ('bloomberg.com', 42, 8, 5), ('wsj.com', 37, 15, 8),
                ('aljazeera.com', 28, 18, 10), ('dailymail.co.uk', 12, 25, 30),
                ('breitbart.com', 8, 15, 40), ('infowars.com', 5, 10, 50),
                ('rt.com', 10, 20, 35)
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

def get_source_credibility_data():
    """Get source credibility data from database"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT source, high, medium, low, total_analyzed
                FROM source_stats
                ORDER BY total_analyzed DESC, source ASC
            ''')

            data = cursor.fetchall()
            sources = []
            credibility_scores = []
            high_counts = []
            medium_counts = []
            low_counts = []
            total_counts = []

            for source, high, medium, low, total in data:
                total_current = high + medium + low
                score = (high * 1.0 + medium * 0.5 + low * 0.0) / total_current if total_current > 0 else 0.5

                sources.append(source)
                credibility_scores.append(round(score, 2))
                high_counts.append(high)
                medium_counts.append(medium)
                low_counts.append(low)
                total_counts.append(total_current)

            return {
                'sources': sources,
                'credibility_scores': credibility_scores,
                'high_counts': high_counts,
                'medium_counts': medium_counts,
                'low_counts': low_counts,
                'total_counts': total_counts
            }
    except Exception as e:
        logger.error(f"Error getting source credibility data: {str(e)}")
        return {
            'sources': [], 'credibility_scores': [],
            'high_counts': [], 'medium_counts': [],
            'low_counts': [], 'total_counts': []
        }

def get_analysis_history():
    """Get analysis history from database"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT url, title, source, credibility_level, short_summary,
                       strftime("%Y-%m-%d %H:%M", analysis_date) as formatted_date
                FROM news
                ORDER BY analysis_date DESC
                LIMIT 15
            ''')

            rows = cursor.fetchall()
            return [{
                'url': row['url'],
                'title': row['title'],
                'source': row['source'],
                'credibility': row['credibility_level'],
                'summary': row['short_summary'],
                'date': row['formatted_date']
            } for row in rows]
    except Exception as e:
        logger.error(f"Error getting analysis history: {str(e)}")
        return []

# Configure newspaper library
config = Config()
config.browser_user_agent = user_agent
config.request_timeout = 30

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
            'anthropic': 'configured' if ANTHROPIC_API_KEY else 'not_configured',
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
    """Endpoint for getting source credibility chart data"""
    try:
        chart_data = get_source_credibility_data()
        if not chart_data['sources']:
            populate_test_data()
            chart_data = get_source_credibility_data()

        return jsonify({
            'status': 'success',
            'data': chart_data
        })
    except Exception as e:
        logger.error(f"Error in source_credibility_chart endpoint: {str(e)}")
        return jsonify({
            'error': 'An error occurred while fetching chart data',
            'status': 500,
            'details': str(e)
        }), 500

@app.route('/analysis-history', methods=['GET'])
def analysis_history():
    """Endpoint for getting analysis history"""
    try:
        history = get_analysis_history()
        return jsonify({
            'status': 'success',
            'history': history
        })
    except Exception as e:
        logger.error(f"Error in analysis_history endpoint: {str(e)}")
        return jsonify({
            'error': 'An error occurred while fetching analysis history',
            'status': 500,
            'details': str(e)
        }), 500

@app.route('/analyze', methods=['POST', 'OPTIONS'])
def analyze():
    """Analyze article endpoint"""
    logger.info(f"Received analyze request. Method: {request.method}")

    if request.method == 'OPTIONS':
        response = make_response()
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Accept'
        return response

    try:
        if not request.is_json:
            return jsonify({
                'error': 'Request must be JSON',
                'status': 400
            }), 400

        data = request.get_json()
        if not data or 'input_text' not in data:
            return jsonify({
                'error': 'Missing input text',
                'status': 400
            }), 400

        input_text = data['input_text'].strip()
        source_name = data.get('source_name_manual', 'Direct Input').strip()

        if not input_text:
            return jsonify({
                'error': 'Empty input text',
                'status': 400
            }), 400

        # Process article
        if input_text.startswith(('http://', 'https://')):
            content, source, title = extract_text_from_url(input_text)
            if not content:
                return jsonify({
                    'error': 'Could not extract article content',
                    'status': 400
                }), 400
        else:
            if len(input_text) < 100:
                return jsonify({
                    'error': 'Content too short',
                    'status': 400
                }), 400
            content = input_text
            title = 'User-provided Text'
            source = source_name

        # Analyze the article content
        analysis = analyze_with_claude(content, source)
        credibility = save_analysis(
            input_text if input_text.startswith(('http://', 'https://')) else None,
            title, source, content, analysis
        )

        # Prepare response
        response_data = {
            'status': 'success',
            'analysis': analysis,
            'credibility': credibility,
            'title': title,
            'source': source,
            'output': format_analysis_results(title, source, analysis, credibility)
        }

        return jsonify(response_data)

    except Exception as e:
        logger.error(f"Error in analyze endpoint: {str(e)}")
        return jsonify({
            'error': str(e),
            'status': 500
        }), 500

def extract_text_from_url(url):
    """Extract text from URL"""
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

        article = Article(clean_url, config=config)
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
    """Analyze article text using Claude API"""
    try:
        if not ANTHROPIC_API_KEY:
            raise ValueError("Anthropic API key is not configured")

        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        max_chars_for_claude = 10000
        if len(content) > max_chars_for_claude:
            content = content[:max_chars_for_claude]

        prompt = (
            'Analyze this news article for credibility, bias, and factual accuracy.\n\n'
            f'Article Text:\n"""\n{content}\n"""\n\n'
            f'Source: {source}\n\n'
            'Return results in JSON format with these fields: '
            'news_integrity, fact_check_needed_score, sentiment_score, '
            'bias_score, topics, key_arguments, mentioned_facts, '
            'author_purpose, potential_biases_identified, short_summary, index_of_credibility'
        )

        message = client.messages.create(
            model=MODEL_NAME,
            max_tokens=2000,
            temperature=0.2,
            messages=[{'role': 'user', 'content': prompt}]
        )

        raw_json_text = message.content[0].text.strip()
        match = re.search(r'```json\s*(\{.*\})\s*```', raw_json_text, re.DOTALL)

        if match:
            json_str = match.group(1)
        else:
            json_str = raw_json_text

        return json.loads(json_str)

    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        raise

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
            integrity = analysis.get('news_integrity', 0.0)
            fact_check = analysis.get('fact_check_needed_score', 1.0)
            sentiment = analysis.get('sentiment_score', 0.5)
            bias = analysis.get('bias_score', 1.0)
            summary = analysis.get('short_summary', 'No summary')
            credibility = analysis.get('index_of_credibility', 0.0)
            level = calculate_credibility(integrity, fact_check, sentiment, bias)
            db_url = url if url else f'text_{datetime.now(timezone.utc).timestamp()}'

            cursor.execute('''
                INSERT INTO news
                (url, title, source, content, integrity, fact_check, sentiment, bias,
                credibility_level, short_summary, index_of_credibility)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(url) DO UPDATE SET
                title=excluded.title, source=excluded.source, content=excluded.content,
                integrity=excluded.integrity, fact_check=excluded.fact_check,
                sentiment=excluded.sentiment, bias=excluded.bias,
                credibility_level=excluded.credibility_level,
                short_summary=excluded.short_summary,
                index_of_credibility=excluded.index_of_credibility,
                analysis_date=CURRENT_TIMESTAMP
            ''', (db_url, title, source, content, integrity, fact_check,
                  sentiment, bias, level, summary, credibility))

            # Update source stats
            cursor.execute('SELECT high, medium, low, total_analyzed FROM source_stats WHERE source = ?', (source,))
            row = cursor.fetchone()

            if row:
                high, medium, low, total = row
                if level == 'High': high += 1
                elif level == 'Medium': medium += 1
                else: low += 1
                total += 1
                cursor.execute('''
                    UPDATE source_stats SET high=?, medium=?, low=?, total_analyzed=?
                    WHERE source=?
                ''', (high, medium, low, total, source))
            else:
                counts = {'High': 1, 'Medium': 0, 'Low': 0}
                counts[level] = 1
                cursor.execute('''
                    INSERT INTO source_stats
                    (source, high, medium, low, total_analyzed)
                    VALUES (?, ?, ?, ?, ?)
                ''', (source, counts['High'], counts['Medium'], counts['Low'], 1))

            conn.commit()
            return level
    except Exception as e:
        logger.error(f"Error saving analysis: {str(e)}")
        raise

def format_analysis_results(title, source, analysis, credibility):
    """Format analysis results for display"""
    try:
        return {
            'title': title,
            'source': source,
            'credibility': credibility,
            'analysis': analysis,
            'output_md': f"""
            <div class="analysis-section">
                <h2>Article Information</h2>
                <p><strong>Title:</strong> {html.escape(title)}</p>
                <p><strong>Source:</strong> {html.escape(source)}</p>
                <p><strong>Credibility:</strong> {credibility}</p>
            </div>
            <div class="analysis-section">
                <h2>Analysis Results</h2>
                <p><strong>Integrity:</strong> {analysis.get('news_integrity', 0.0):.2f}</p>
                <p><strong>Fact Check Needed:</strong> {analysis.get('fact_check_needed_score', 1.0):.2f}</p>
                <p><strong>Sentiment:</strong> {analysis.get('sentiment_score', 0.5):.2f}</p>
                <p><strong>Bias:</strong> {analysis.get('bias_score', 1.0):.2f}</p>
            </div>
            """
        }
    except Exception as e:
        logger.error(f"Error formatting analysis results: {str(e)}")
        return {"error": "Error formatting analysis results"}

if __name__ == '__main__':
    initialize_database()
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
