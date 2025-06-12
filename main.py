from flask import Flask, request, jsonify, render_template, make_response, abort
from werkzeug.middleware.proxy_fix import ProxyFix
import logging
from logging.handlers import RotatingFileHandler
import os
import sqlite3
import re
import json
import html
import uuid
import time
from datetime import datetime, timezone
import anthropic
from flask_cors import CORS
from functools import wraps

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

# Security middleware
def security_middleware(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug(f"Security middleware called for path: {request.path}")
        if any(pattern in request.path for pattern in [
            '/wp-admin/', '/admin/', '/phpmyadmin/', '/.env',
            '/config.php', '/setup-config.php'
        ]):
            logger.warning(f"Blocked suspicious request to {request.path}")
            return abort(404)

        user_agent = request.headers.get('User-Agent', '').lower()
        if any(agent in user_agent for agent in [
            'sqlmap', 'nmap', 'nikto', 'wpscan', 'burpsuite', 'acunetix'
        ]):
            logger.warning(f"Blocked request with suspicious user agent: {user_agent}")
            return abort(403)

        return func(*args, **kwargs)
    return wrapper

@app.before_request
def before_request():
    logger.debug("Security middleware applied to request")
    return security_middleware(lambda: None)()

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

# Database configuration
DB_NAME = 'news_analysis.db'

def get_db_connection():
    """Create database connection"""
    logger.debug("Creating database connection")
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def initialize_database():
    """Initialize database schema"""
    try:
        logger.info("Initializing database")
        db_dir = os.path.dirname(DB_NAME)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir)
            logger.debug(f"Created database directory: {db_dir}")

        with get_db_connection() as conn:
            cursor = conn.cursor()
            logger.debug("Database connection established")

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
                    url TEXT,
                    analysis_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    short_summary TEXT,
                    UNIQUE(url)
                )
            ''')
            logger.debug("Created news table")

            cursor.execute('CREATE INDEX IF NOT EXISTS idx_url ON news(url)')
            logger.debug("Created index on news.url")

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
            logger.debug("Created feedback table")

            conn.commit()
            logger.info("Database initialized successfully")
            return True
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}", exc_info=True)
        return False

def check_claude_connection():
    """Check connection to Claude API"""
    try:
        if not ANTHROPIC_API_KEY or ANTHROPIC_API_KEY == 'mock-key':
            logger.warning("Claude API key is not configured or is a mock key")
            return False

        logger.info("Testing Claude API connection")
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

        # Simple test request
        test_prompt = "Test connection"
        try:
            response = client.messages.create(
                model=MODEL_NAME,
                max_tokens=10,
                temperature=0.2,
                messages=[{"role": "user", "content": test_prompt}]
            )
            logger.info("Successfully connected to Claude API")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Claude API: {str(e)}", exc_info=True)
            return False

    except Exception as e:
        logger.error(f"Error checking Claude connection: {str(e)}", exc_info=True)
        return False

@app.after_request
def add_security_headers(response):
    """Add security headers with proper CSP configuration"""
    logger.debug("Adding security headers to response")
    response.headers['Content-Security-Policy'] = (
        "default-src 'self'; "
        "script-src 'self' https://cdn.jsdelivr.net 'unsafe-inline' 'unsafe-eval'; "
        "style-src 'self' https://cdn.jsdelivr.net https://fonts.googleapis.com 'unsafe-inline'; "
        "img-src 'self' data: https://cdn.jsdelivr.net; "
        "font-src 'self' https://cdn.jsdelivr.net https://fonts.gstatic.com; "
        "connect-src 'self' https://cdn.jsdelivr.net; "
        "frame-src 'self'; "
        "object-src 'none'; "
        "base-uri 'self'; "
        "form-action 'self'; "
        "frame-ancestors 'none'"
    )
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, Accept'
    logger.debug("Security headers added to response")
    return response

@app.route('/')
def index():
    """Home page route"""
    logger.info("Rendering index page")
    return render_template('index.html')

@app.route('/health')
def health_check():
    """Health check endpoint"""
    logger.info("Processing health check request")
    claude_status = "connected" if check_claude_connection() else "disconnected"
    db_status = "connected" if initialize_database() else "disconnected"

    response = jsonify({
        'status': 'healthy',
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'database': db_status,
        'claude_api': claude_status,
        'api_keys': {
            'anthropic': 'configured' if ANTHROPIC_API_KEY and ANTHROPIC_API_KEY != 'mock-key' else 'not_configured'
        }
    })
    logger.debug("Health check completed successfully")
    return response

@app.route('/faq')
def faq():
    """FAQ page route"""
    logger.info("Rendering FAQ page")
    return render_template('faq.html')

@app.route('/feedback', methods=['GET', 'POST'])
def feedback():
    """Feedback page and form handler"""
    if request.method == 'POST':
        logger.info("Processing feedback submission")
        try:
            name = request.form.get('name')
            email = request.form.get('email')
            feedback_type = request.form.get('type')
            message = request.form.get('message')

            if not all([name, email, feedback_type, message]):
                logger.warning("Feedback submission missing required fields")
                return render_template('feedback.html', error="All fields are required")

            if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
                logger.warning(f"Invalid email address: {email}")
                return render_template('feedback.html', error="Invalid email address")

            with get_db_connection() as conn:
                logger.debug("Saving feedback to database")
                conn.execute('''
                    INSERT INTO feedback (name, email, type, message, date)
                    VALUES (?, ?, ?, ?, ?)
                ''', (name, email, feedback_type, message, datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')))
                conn.commit()
                logger.info(f"Feedback saved successfully from {name} ({email})")

            logger.info("Redirecting to feedback success page")
            return render_template('feedback_success.html')

        except Exception as e:
            logger.error(f"Error saving feedback: {str(e)}", exc_info=True)
            return render_template('feedback.html', error="Error saving feedback")

    logger.info("Rendering feedback form")
    return render_template('feedback.html')

def analyze_with_claude(content, source):
    """Analyze article text using Claude API with full analysis and similar articles search"""
    logger.info("Starting comprehensive article analysis with Claude")

    try:
        if not ANTHROPIC_API_KEY or ANTHROPIC_API_KEY == 'mock-key':
            logger.warning("Claude API key is not configured or is a mock key, using mock data")
            return generate_mock_analysis(source)

        logger.debug("Creating Claude client")
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

        # Truncate content if too long
        max_chars = 20000
        if len(content) > max_chars:
            content = content[:max_chars]
            logger.warning(f"Article content truncated to {max_chars} characters")

        logger.debug("Preparing comprehensive analysis prompt")

        # Current date for article search
        current_date = datetime.now(timezone.utc).strftime('%Y-%m-%d')

        prompt = f"""
        Perform a comprehensive analysis of this news article and find similar recent articles. Return results in JSON format with these sections:

        1. Article Analysis:
        - news_integrity (0.0-1.0)
        - fact_check_needed_score (0.0-1.0)
        - sentiment_score (0.0-1.0)
        - bias_score (0.0-1.0)
        - topics (list of key topics)
        - key_arguments (list of main arguments)
        - mentioned_facts (list of key facts)
        - author_purpose (string)
        - potential_biases_identified (list)
        - short_summary (string)
        - index_of_credibility (0.0-1.0)

        2. Similar Articles:
        - Find 3 recent articles (published within the last week) on similar topics
        - For each article provide:
          * title
          * source
          * url
          * publication_date
          * summary
          * relevance_score (0.0-1.0)
          * trust_score (0.0-1.0)

        Current date: {current_date}
        Article source: {source}

        Article content:
        {content}
        """

        logger.debug("Sending request to Claude API for comprehensive analysis")
        try:
            response = client.messages.create(
                model=MODEL_NAME,
                max_tokens=4000,
                temperature=0.2,
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = response.content[0].text
            logger.debug("Received response from Claude API")

            # Try to extract JSON from response
            json_match = re.search(r'```json\s*(\{.*\})\s*```', response_text, re.DOTALL)
            if json_match:
                logger.debug("Found JSON in response")
                analysis = json.loads(json_match.group(1))

                # Validate the response structure
                if not all(key in analysis for key in ['article_analysis', 'similar_articles']):
                    raise ValueError("Invalid response structure from Claude API")

                logger.info("Successfully parsed comprehensive analysis from Claude API")
                return analysis
            else:
                # Try to parse as direct JSON if no code block found
                try:
                    analysis = json.loads(response_text)
                    if not all(key in analysis for key in ['article_analysis', 'similar_articles']):
                        raise ValueError("Invalid response structure")
                    logger.info("Successfully parsed direct JSON response")
                    return analysis
                except json.JSONDecodeError:
                    logger.error("Failed to parse API response as JSON")
                    return generate_mock_analysis(source)

        except Exception as e:
            logger.error(f"Error calling Claude API: {str(e)}", exc_info=True)
            return generate_mock_analysis(source)

    except Exception as e:
        logger.error(f"Unexpected error in analyze_with_claude: {str(e)}", exc_info=True)
        return generate_mock_analysis(source)

def generate_mock_analysis(source):
    """Generate mock analysis data with similar articles"""
    logger.debug("Generating mock analysis data with similar articles")

    credibility_scores = {
        'bbc.com': 0.9,
        'reuters.com': 0.95,
        'apnews.com': 0.9,
        'nytimes.com': 0.85,
        'theguardian.com': 0.8
    }

    base_score = credibility_scores.get(source, 0.7)

    # Generate mock similar articles
    mock_articles = [
        {
            "title": f"Sample Article {i+1} on Similar Topic",
            "source": f"Example News {i+1}",
            "url": f"https://example.com/article{i+1}",
            "publication_date": datetime.now(timezone.utc).strftime('%Y-%m-%d'),
            "summary": f"This is a sample article about a similar topic to demonstrate functionality.",
            "relevance_score": 0.8 + (i * 0.05),
            "trust_score": 0.7 + (i * 0.05)
        }
        for i in range(3)
    ]

    mock_analysis = {
        "article_analysis": {
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
        },
        "similar_articles": mock_articles
    }

    logger.debug("Generated mock analysis data with similar articles successfully")
    return mock_analysis

def calculate_credibility(integrity, fact_check, sentiment, bias):
    """Calculate credibility level"""
    logger.debug("Calculating credibility level")

    fact_check_score = 1.0 - fact_check
    sentiment_score = 1.0 - abs(sentiment - 0.5) * 2
    bias_score = 1.0 - bias
    score = (integrity * 0.45) + (fact_check_score * 0.35) + (sentiment_score * 0.10) + (bias_score * 0.10)

    if score >= 0.75:
        level = 'High'
    elif score >= 0.5:
        level = 'Medium'
    else:
        level = 'Low'

    logger.debug(f"Calculated credibility score: {score:.2f}, level: {level}")
    return level

def save_analysis(url, title, source, content, analysis):
    """Save analysis to database with improved error handling"""
    logger.debug("Saving analysis to database")

    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()

            article_analysis = analysis.get('article_analysis', {})
            integrity = article_analysis.get('news_integrity', 0.5)
            fact_check = article_analysis.get('fact_check_needed_score', 0.5)
            sentiment = article_analysis.get('sentiment_score', 0.5)
            bias = article_analysis.get('bias_score', 0.5)
            summary = article_analysis.get('short_summary', 'No summary available')
            credibility = article_analysis.get('index_of_credibility', 0.5)
            level = calculate_credibility(integrity, fact_check, sentiment, bias)

            db_url = url if url and url.startswith(('http://', 'https://')) else f'text_{datetime.now(timezone.utc).timestamp()}'

            logger.debug(f"Attempting to insert analysis for URL: {db_url}")
            try:
                cursor.execute('''
                    INSERT INTO news
                    (url, title, source, content, integrity, fact_check, sentiment, bias,
                    credibility_level, short_summary, index_of_credibility, analysis_date)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (db_url, title, source, content, integrity, fact_check,
                      sentiment, bias, level, summary, credibility,
                      datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')))

                conn.commit()
                logger.info(f"Successfully saved analysis for {title} from {source}")
                return level
            except sqlite3.IntegrityError as e:
                if "UNIQUE constraint failed: news.url" in str(e):
                    logger.warning(f"URL already exists in database: {db_url}")
                    cursor.execute('''
                        UPDATE news
                        SET title = ?,
                            source = ?,
                            content = ?,
                            integrity = ?,
                            fact_check = ?,
                            sentiment = ?,
                            bias = ?,
                            credibility_level = ?,
                            short_summary = ?,
                            index_of_credibility = ?,
                            analysis_date = ?
                        WHERE url = ?
                    ''', (title, source, content, integrity, fact_check,
                          sentiment, bias, level, summary, credibility,
                          datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'), db_url))
                    conn.commit()
                    logger.info(f"Updated existing analysis for URL: {db_url}")
                    return level
                else:
                    raise e
            except Exception as e:
                logger.error(f"Error saving analysis: {str(e)}", exc_info=True)
                raise

    except Exception as e:
        logger.error(f"Unexpected error in save_analysis: {str(e)}", exc_info=True)
        return 'Medium'

def format_analysis_results(title, source, analysis, credibility):
    """Format analysis results with similar articles section"""
    logger.debug("Formatting analysis results with similar articles")

    try:
        # Extract article analysis data
        article_analysis = analysis.get('article_analysis', {})
        integrity = article_analysis.get('news_integrity', 0.5)
        fact_check = article_analysis.get('fact_check_needed_score', 0.5)
        sentiment = article_analysis.get('sentiment_score', 0.5)
        bias = article_analysis.get('bias_score', 0.5)
        summary = html.escape(article_analysis.get('short_summary', 'No summary available'))
        topics = [html.escape(str(t)) for t in article_analysis.get('topics', [])]
        key_args = [html.escape(str(a)) for a in article_analysis.get('key_arguments', [])]
        biases = [html.escape(str(b)) for b in article_analysis.get('potential_biases_identified', [])]

        # Create topic badges
        topic_badges = ' '.join(
            f'<span class="badge bg-primary me-1 mb-1">{topic}</span>'
            for topic in topics
        )

        # Create key arguments list
        key_args_list = ''.join(
            f'<li class="mb-2"><i class="bi bi-check-circle-fill text-primary me-2"></i>{arg}</li>'
            for arg in key_args
        )

        # Create biases list
        biases_list = ''.join(
            f'<li class="mb-2"><i class="bi bi-exclamation-triangle-fill text-warning me-2"></i>{bias}</li>'
            for bias in biases
        )

        # Create score indicators
        integrity_indicator = create_score_indicator(integrity, "News Integrity")
        fact_check_indicator = create_score_indicator(1 - fact_check, "Fact Check Score")
        sentiment_indicator = create_score_indicator(sentiment, "Sentiment")
        bias_indicator = create_score_indicator(1 - bias, "Bias Score")

        # Create credibility gauge
        credibility_gauge = create_credibility_gauge(article_analysis.get('index_of_credibility', 0.5))

        # Format similar articles
        similar_articles = analysis.get('similar_articles', [])
        similar_articles_html = ''

        if similar_articles:
            similar_articles_html = '<div class="similar-articles-section mt-5">'
            similar_articles_html += '<h3 class="section-title mb-4">Similar Recent Articles</h3>'
            similar_articles_html += '<div class="row g-4">'

            for article in similar_articles:
                title = html.escape(article.get('title', 'No Title'))
                source = html.escape(article.get('source', 'Unknown Source'))
                url = html.escape(article.get('url', '#'))
                date = html.escape(article.get('publication_date', 'N/A'))
                summary = html.escape(article.get('summary', 'No summary available'))
                relevance = int(article.get('relevance_score', 0.7) * 100)
                trust = int(article.get('trust_score', 0.7) * 100)

                # Determine color classes based on scores
                relevance_color = "success" if relevance > 70 else "warning" if relevance > 40 else "danger"
                trust_color = "success" if trust > 70 else "warning" if trust > 40 else "danger"

                similar_articles_html += f'''
                <div class="col-md-4">
                    <div class="card h-100">
                        <div class="card-body">
                            <div class="d-flex justify-content-between mb-2">
                                <span class="badge bg-secondary">{source}</span>
                                <span class="badge bg-light text-dark">{date}</span>
                            </div>
                            <h5 class="card-title">{title}</h5>
                            <p class="card-text">{summary}</p>
                        </div>
                        <div class="card-footer bg-transparent">
                            <div class="d-flex justify-content-between">
                                <div>
                                    <small class="text-muted">Relevance</small>
                                    <div class="progress mt-1" style="height: 5px;">
                                        <div class="progress-bar bg-{relevance_color}"
                                             role="progressbar"
                                             style="width: {relevance}%"
                                             aria-valuenow="{relevance}"
                                             aria-valuemin="0"
                                             aria-valuemax="100">
                                        </div>
                                    </div>
                                </div>
                                <div>
                                    <small class="text-muted">Trust</small>
                                    <div class="progress mt-1" style="height: 5px;">
                                        <div class="progress-bar bg-{trust_color}"
                                             role="progressbar"
                                             style="width: {trust}%"
                                             aria-valuenow="{trust}"
                                             aria-valuemin="0"
                                             aria-valuemax="100">
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <a href="{url}" target="_blank" class="btn btn-primary btn-sm mt-2 w-100">Read Article</a>
                        </div>
                    </div>
                </div>
                '''
            similar_articles_html += '</div></div>'

        logger.debug("Successfully formatted analysis results with similar articles")
        return {
            'title': title,
            'source': source,
            'credibility': credibility,
            'output_md': f"""
            <div class="analysis-results">
                <div class="article-header mb-4 p-3 bg-white rounded shadow-sm">
                    <h2 class="mb-3">{html.escape(title)}</h2>
                    <div class="d-flex flex-wrap justify-content-between align-items-center">
                        <div>
                            <span class="badge bg-secondary me-2">{html.escape(source)}</span>
                            <span class="credibility-badge {credibility.lower()}">{credibility}</span>
                        </div>
                        <div class="credibility-score-display">
                            {credibility_gauge}
                        </div>
                    </div>
                </div>

                <div class="scores-section mb-4">
                    <h3 class="section-title mb-3">Credibility Scores</h3>
                    <div class="row g-3">
                        <div class="col-md-6 col-lg-3">
                            {integrity_indicator}
                        </div>
                        <div class="col-md-6 col-lg-3">
                            {fact_check_indicator}
                        </div>
                        <div class="col-md-6 col-lg-3">
                            {sentiment_indicator}
                        </div>
                        <div class="col-md-6 col-lg-3">
                            {bias_indicator}
                        </div>
                    </div>
                </div>

                <div class="summary-section mb-4 p-3 bg-white rounded shadow-sm">
                    <h3 class="section-title mb-3">Article Summary</h3>
                    <p class="lead">{summary}</p>
                </div>

                <div class="topics-section mb-4 p-3 bg-white rounded shadow-sm">
                    <h3 class="section-title mb-3">Key Topics</h3>
                    <div class="d-flex flex-wrap gap-2">
                        {topic_badges}
                    </div>
                </div>

                <div class="row mb-4">
                    <div class="col-md-6">
                        <div class="arguments-section p-3 bg-white rounded shadow-sm h-100">
                            <h3 class="section-title mb-3">Key Arguments</h3>
                            <ul class="list-unstyled mb-0">
                                {key_args_list}
                            </ul>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="biases-section p-3 bg-white rounded shadow-sm h-100">
                            <h3 class="section-title mb-3">Potential Biases</h3>
                            <ul class="list-unstyled mb-0">
                                {biases_list if biases_list else '<li class="text-muted">No significant biases detected</li>'}
                            </ul>
                        </div>
                    </div>
                </div>

                {similar_articles_html}
            </div>
            """
        }
    except Exception as e:
        logger.error(f"Error formatting analysis results: {str(e)}", exc_info=True)
        return {
            "error": "Error formatting analysis results",
            "output_md": '<div class="alert alert-danger">Error displaying analysis results</div>'
        }

def create_score_indicator(score: float, label: str) -> str:
    """Create a visual score indicator"""
    score_percent = int(score * 100)
    color_class = "success" if score > 0.7 else "warning" if score > 0.4 else "danger"

    return f"""
    <div class="score-card h-100 p-3 bg-light rounded">
        <div class="d-flex justify-content-between align-items-center mb-2">
            <span class="score-label text-muted">{html.escape(label)}</span>
            <span class="score-value fw-bold h4 mb-0">{score:.2f}</span>
        </div>
        <div class="progress" style="height: 8px;">
            <div class="progress-bar bg-{color_class}"
                 role="progressbar"
                 style="width: {score_percent}%"
                 aria-valuenow="{score_percent}"
                 aria-valuemin="0"
                 aria-valuemax="100">
            </div>
        </div>
        <div class="score-description mt-2 text-muted small">
            {get_score_description(score, label)}
        </div>
    </div>
    """

def get_score_description(score: float, label: str) -> str:
    """Get description for a score based on its value and label"""
    if label == "News Integrity":
        if score > 0.8:
            return "High integrity with reliable sourcing"
        elif score > 0.6:
            return "Generally reliable with minor issues"
        elif score > 0.4:
            return "Some concerns about sourcing"
        else:
            return "Significant integrity concerns"
    elif label == "Fact Check Score":
        if score > 0.8:
            return "High factual accuracy"
        elif score > 0.6:
            return "Mostly accurate with minor issues"
        elif score > 0.4:
            return "Some factual concerns"
        else:
            return "Significant factual inaccuracies likely"
    elif label == "Sentiment":
        if score > 0.6:
            return "Positive tone"
        elif score > 0.4:
            return "Neutral tone"
        else:
            return "Negative tone"
    elif label == "Bias Score":
        if score > 0.8:
            return "Minimal bias detected"
        elif score > 0.6:
            return "Some bias present"
        elif score > 0.4:
            return "Noticeable bias"
        else:
            return "Strong bias detected"
    return "Score description"

def create_credibility_gauge(score: float) -> str:
    """Create a radial credibility gauge"""
    score_percent = int(score * 100)
    color = "#28a745" if score > 0.7 else "#ffc107" if score > 0.4 else "#dc3545"

    return f"""
    <div class="credibility-gauge" style="width: 100px; height: 100px; position: relative;">
        <svg viewBox="0 0 100 100" style="transform: rotate(-90deg); position: absolute; top: 0; left: 0;">
            <circle cx="50" cy="50" r="45" fill="none" stroke="#eee" stroke-width="10" />
            <circle cx="50" cy="50" r="45" fill="none" stroke="{color}" stroke-width="10"
                    stroke-dasharray="{3.14 * 45 * 2 * score} {3.14 * 45 * 2 * (1 - score)}"
                    stroke-linecap="round" />
        </svg>
        <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); text-align: center;">
            <div style="font-size: 1.2rem; font-weight: bold; color: {color};">{score_percent}%</div>
            <div style="font-size: 0.7rem; color: #6c757d;">Credibility</div>
        </div>
    </div>
    """

@app.route('/analyze', methods=['POST', 'OPTIONS'])
def analyze():
    """Analyze article endpoint with comprehensive error handling"""
    if request.method == 'OPTIONS':
        response = make_response()
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Accept'
        response.headers['Access-Control-Max-Age'] = '3600'
        return response

    logger.info(f"Received analyze request. Method: {request.method}, Path: {request.path}")

    # Validate request content type
    if not request.is_json:
        logger.warning(f"Invalid content type: {request.content_type}")
        return jsonify({
            'error': 'Invalid content type',
            'status': 400,
            'details': 'Content-Type header must be application/json',
            'request_id': str(uuid.uuid4())
        }), 400

    try:
        logger.debug("Processing JSON request data")

        # Get and validate data
        data = request.get_json()
        if not data:
            logger.warning("Empty request body received")
            return jsonify({
                'error': 'Empty request body',
                'status': 400,
                'request_id': str(uuid.uuid4())
            }), 400

        logger.debug(f"Request data: {json.dumps(data, indent=2)}")

        if 'input_text' not in data:
            logger.warning("Missing input_text in request")
            return jsonify({
                'error': 'Missing input text',
                'status': 400,
                'details': 'input_text field is required',
                'request_id': str(uuid.uuid4())
            }), 400

        input_text = data['input_text'].strip()
        source_name = data.get('source_name_manual', 'Direct Input').strip()

        if not input_text:
            logger.warning("Empty input text received")
            return jsonify({
                'error': 'Empty input text',
                'status': 400,
                'details': 'Input text cannot be empty',
                'request_id': str(uuid.uuid4())
            }), 400

        logger.info(f"Processing input of length: {len(input_text)} characters")

        # Process direct text input
        logger.info("Processing direct text input")
        if len(input_text) < 100:
            logger.warning(f"Input text too short: {len(input_text)} characters")
            return jsonify({
                'error': 'Content too short',
                'status': 400,
                'details': f'Minimum 100 characters required, got {len(input_text)}',
                'request_id': str(uuid.uuid4())
            }), 400

        content = input_text
        title = 'User-provided Text'
        source = source_name

        logger.info(f"Successfully extracted content. Length: {len(content)} characters")

        # Analyze content
        logger.info("Starting article analysis")
        analysis_start_time = time.time()
        analysis = analyze_with_claude(content, source)
        analysis_duration = time.time() - analysis_start_time
        logger.info(f"Completed article analysis in {analysis_duration:.2f} seconds")

        # Save analysis to database
        logger.info("Saving analysis to database")
        save_start_time = time.time()
        credibility = save_analysis(None, title, source, content, analysis)
        save_duration = time.time() - save_start_time
        logger.info(f"Saved analysis to database in {save_duration:.2f} seconds")

        # Format response
        logger.info("Preparing response data")
        response_data = {
            'status': 'success',
            'analysis': analysis,
            'credibility': credibility,
            'title': title,
            'source': source,
            'output': format_analysis_results(title, source, analysis, credibility),
            'request_id': str(uuid.uuid4()),
            'processing_time': f"{analysis_duration:.2f} seconds"
        }

        logger.info("Successfully processed analysis request")
        return jsonify(response_data)

    except Exception as e:
        request_id = str(uuid.uuid4())
        logger.error(f"Unexpected error in analyze endpoint: {str(e)}. Request ID: {request_id}", exc_info=True)
        return jsonify({
            'error': 'Internal server error',
            'status': 500,
            'details': str(e),
            'suggestions': [
                'Please try again later',
                'Check your internet connection',
                'If the problem persists, contact support with this request ID'
            ],
            'request_id': request_id
        }), 500

if __name__ == '__main__':
    # Initialize database
    if not initialize_database():
        logger.error("Failed to initialize database. Exiting...")
        exit(1)

    # Check Claude connection
    if not check_claude_connection():
        logger.warning("Claude API connection test failed. Application will use mock data.")

    # Start the application
    logger.info("Starting Flask application")
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))


