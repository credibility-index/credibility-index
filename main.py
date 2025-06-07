import os
import logging
import sqlite3
import re
import json
import requests
import html # For escaping HTML in render_same_topic_articles_html
from datetime import datetime, timedelta
from urllib.parse import urlparse
from logging.handlers import RotatingFileHandler
from flask import Flask, request, jsonify, render_template, abort
from werkzeug.middleware.proxy_fix import ProxyFix
import anthropic # For the Claude API
from newspaper import Article # For extract_text_from_url
from stop_words import get_stop_words # For generate_query

# =========================================================
# FLASK APP SETUP
# =========================================================
app = Flask(__name__, static_folder='static', template_folder='templates')
# Use ProxyFix if running behind a proxy like Nginx or a cloud platform
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

# =========================================================
# CONFIGURATION & ENVIRONMENT VARIABLES
# =========================================================
# --- Logging Setup ---
def setup_logging():
    # File handler for logging to a file
    file_handler = RotatingFileHandler('app.log', maxBytes=1024*1024, backupCount=5)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    
    # Console handler for logging to standard output (useful for development/Docker)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    app.logger.addHandler(file_handler)
    app.logger.addHandler(console_handler)
    app.logger.setLevel(logging.INFO) # Set default logging level

# Call logging setup early to ensure all subsequent logs are captured
setup_logging()

# --- Environment Variables Check ---
REQUIRED_ENV_VARS = [
    'ANTHROPIC_API_KEY',
    'SECRET_KEY',
]
# NEWS_API_KEY is optional, so it's not in REQUIRED_ENV_VARS
for var in REQUIRED_ENV_VARS:
    if not os.getenv(var):
        error_msg = f"Missing required environment variable: {var}"
        app.logger.critical(error_msg) # Use critical as it's an unrecoverable error
        raise ValueError(error_msg)

ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
NEWS_API_KEY = os.getenv('NEWS_API_KEY')
NEWS_API_ENABLED = bool(NEWS_API_KEY) # True if key exists, False otherwise
MODEL_NAME = 'claude-3-opus-20240229' # Default Claude model
SECRET_KEY = os.getenv('SECRET_KEY')

app.secret_key = SECRET_KEY # Set the Flask secret key for session management

if not NEWS_API_ENABLED:
    app.logger.warning('NEWS_API_KEY is missing! Similar news functionality will be disabled.')

# Naive UTC timezone for datetime.now(UTC) for database consistency
class UTC(datetime.tzinfo):
    def utcoffset(self, dt): return timedelta(0)
    def tzname(self, dt): return "UTC"
    def dst(self, dt): return timedelta(0)

# Flask app specific configurations
app.config.update(
    SESSION_COOKIE_SECURE=True, # Ensure cookies are sent over HTTPS only
    SESSION_COOKIE_HTTPONLY=True, # Prevent client-side JavaScript access to cookies
    SESSION_COOKIE_SAMESITE='Lax', # Protect against CSRF attacks
    PERMANENT_SESSION_LIFETIME=timedelta(days=1) # Session duration
)

# Database configuration
DB_NAME = 'news_analysis.db'

# Initial source reliability data for source_stats table
INITIAL_SOURCE_COUNTS = {
    'bbc.com': {'high': 15, 'medium': 5, 'low': 1},
    'reuters.com': {'high': 20, 'medium': 3, 'low': 0},
    'foxnews.com': {'high': 3, 'medium': 7, 'low': 15},
    'cnn.com': {'high': 5, 'medium': 10, 'low': 5},
    'nytimes.com': {'high': 10, 'medium': 5, 'low': 2},
    'theguardian.com': {'high': 12, 'medium': 4, 'low': 1},
    'apnews.com': {'high': 18, 'medium': 2, 'low': 0}
}

# Mapping of domain to media owner for context
media_owners = {
    'bbc.com': 'BBC',
    'reuters.com': 'Thomson Reuters',
    'foxnews.com': 'Fox Corporation',
    'cnn.com': 'Warner Bros. Discovery',
    'nytimes.com': 'The New York Times Company',
    'theguardian.com': 'Guardian Media Group',
    'apnews.com': 'Associated Press',
    'aljazeera.com': 'Al Jazeera Media Network',
    'wsj.com': 'News Corp' # Added for comprehensive source list
}

# Predefined trust scores for known domains (used in similar articles rendering)
predefined_trust_scores = {
    'bbc.com': 0.9, 'bbc.co.uk': 0.9, 'reuters.com': 0.95, 'apnews.com': 0.93,
    'nytimes.com': 0.88, 'theguardian.com': 0.85, 'wsj.com': 0.82,
    'cnn.com': 0.70, 'foxnews.com': 0.40, 'aljazeera.com': 0.80
}


# Trusted NewsAPI sources IDs for filtering similar news
TRUSTED_NEWS_SOURCES_IDS = [
    'bbc-news', 'reuters', 'associated-press', 'the-new-york-times',
    'the-guardian-uk', 'the-wall-street-journal', 'cnn', 'al-jazeera-english'
]

# Stop words for query generation (English)
stop_words_en = get_stop_words('en')

# WordPress scanner protection paths (regex patterns)
WORDPRESS_PATHS = [
    re.compile(r'wp-admin', re.IGNORECASE),
    re.compile(r'wp-includes', re.IGNORECASE),
    re.compile(r'wp-content', re.IGNORECASE),
    re.compile(r'xmlrpc\.php', re.IGNORECASE),
    re.compile(r'wp-login\.php', re.IGNORECASE),
    re.compile(r'wp-config\.php', re.IGNORECASE),
    re.compile(r'readme\.html', re.IGNORECASE),
    re.compile(r'license\.txt', re.IGNORECASE),
    re.compile(r'wp-json', re.IGNORECASE),
    re.compile(r'wp-comments-post\.php', re.IGNORECASE),
    re.compile(r'wp-trackback\.php', re.IGNORECASE),
    re.compile(r'wp-signup\.php', re.IGNORECASE),
    re.compile(r'wp-activate\.php', re.IGNORECASE),
    re.compile(r'wp-blog-header\.php', re.IGNORECASE),
    re.compile(r'wp-cron\.php', re.IGNORECASE),
    re.compile(r'wp-links-opml\.php', re.IGNORECASE),
    re.compile(r'wp-mail\.php', re.IGNORECASE),
    re.compile(r'wp-settings\.php', re.IGNORECASE),
    re.compile(r'wp-config-sample\.php', re.IGNORECASE),
    re.compile(r'wp-load\.php', re.IGNORECASE),
    re.compile(r'wp-.*\.php', re.IGNORECASE), # Catch common WordPress PHP files
    re.compile(r'.*wp-admin.*', re.IGNORECASE),
    re.compile(r'.*wp-content.*', re.IGNORECASE),
    re.compile(r'.*wp-includes.*', re.IGNORECASE),
    re.compile(r'.*wp-json.*', re.IGNORECASE)
]

# =========================================================
# MIDDLEWARE AND ERROR HANDLERS
# =========================================================
@app.before_request
def block_wordpress_scanners():
    """
    Blocks requests to common WordPress paths to deter automated scanners.
    Also blocks suspicious query parameters often used in attacks.
    """
    path = request.path.lower()
    for pattern in WORDPRESS_PATHS:
        if pattern.search(path):
            app.logger.warning(f'Blocked WordPress scanner request from {request.remote_addr} to: {request.path}')
            return abort(404)

    # Check for suspicious query parameters often used for LFI/RFI
    if any(param in request.query_string.decode('utf-8', 'ignore') for param in ['=http://', '=https://', '=ftp://']):
        app.logger.warning(f'Blocked suspicious query parameter from {request.remote_addr} in: {request.path}?{request.query_string.decode()}')
        return abort(404)

@app.after_request
def add_security_headers(response):
    """
    Adds essential security headers to all responses.
    This version allows necessary CDNs for static assets.
    """
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    
    # Content-Security-Policy (CSP)
    # Allows self, plus specific CDNs for styles, scripts, and fonts.
    # 'unsafe-inline' for styles/scripts is often needed for Plotly/JS libraries,
    # consider refining this if possible (e.g., using nonces or hashes).
    csp = (
        "default-src 'self'; "
        "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
        "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://cdn.plot.ly; "
        "font-src 'self' https://cdn.jsdelivr.net; "
        "img-src 'self' data:;" # Allow images from self and data URIs
    )
    response.headers['Content-Security-Policy'] = csp
    
    # HSTS (Strict-Transport-Security) for HTTPS enforcement
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    # Permissions-Policy to control browser features
    response.headers['Permissions-Policy'] = 'geolocation=(), microphone=(), camera=()'
    return response

# Error handlers for custom error pages
@app.errorhandler(404)
def page_not_found(e):
    app.logger.warning(f"404 Not Found: {request.path} from {request.remote_addr}")
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    app.logger.exception(f'Internal Server Error for request to {request.path} from {request.remote_addr}: {e}') # log the full traceback
    return render_template('500.html'), 500

# =========================================================
# DATABASE FUNCTIONS
# =========================================================
def ensure_db_schema():
    """Ensures the database tables exist with the correct schema."""
    conn = None
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS news (
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
        )''')
        c.execute('''CREATE TABLE IF NOT EXISTS source_stats (
            source TEXT PRIMARY KEY,
            high INTEGER DEFAULT 0,
            medium INTEGER DEFAULT 0,
            low INTEGER DEFAULT 0,
            total_analyzed INTEGER DEFAULT 0
        )''')
        c.execute('''CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL,
            type TEXT NOT NULL,
            message TEXT NOT NULL,
            date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )''')
        conn.commit()
        app.logger.info('Database schema ensured successfully.')
    except sqlite3.Error as e:
        app.logger.error(f'Error ensuring database schema: {e}', exc_info=True)
        raise # Re-raise to stop application if DB is not setup correctly
    finally:
        if conn:
            conn.close()

def initialize_sources(initial_counts):
    """Initializes source_stats table with predefined counts if sources are new."""
    conn = None
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        for source, counts in initial_counts.items():
            c.execute('SELECT total_analyzed FROM source_stats WHERE source = ?', (source,))
            row = c.fetchone()
            if row is None: # Only insert if source does not exist
                high = counts.get('high', 0)
                medium = counts.get('medium', 0)
                low = counts.get('low', 0)
                c.execute('''INSERT INTO source_stats (source, high, medium, low, total_analyzed)
                             VALUES (?, ?, ?, ?, ?)''',
                             (source, high, medium, low, high + medium + low))
        conn.commit()
        app.logger.info('Initial source initialization completed successfully.')
    except sqlite3.Error as e:
        app.logger.error(f'Database error during source initialization: {e}', exc_info=True)
        raise
    finally:
        if conn:
            conn.close()

def check_database_integrity():
    """Performs a basic integrity check on the database."""
    conn = None
    try:
        if not os.path.exists(DB_NAME):
            app.logger.critical(f'Database file {DB_NAME} not found! Please check setup.')
            return False

        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()

        c.execute('SELECT name FROM sqlite_master WHERE type="table"')
        tables = [table[0] for table in c.fetchall()]
        required_tables = ['news', 'source_stats', 'feedback']
        for table in required_tables:
            if table not in tables:
                app.logger.critical(f'Critical table "{table}" is missing in the database!')
                return False

        app.logger.info("Database integrity check passed.")
        return True
    except Exception as e:
        app.logger.critical(f'Error during database integrity check: {e}', exc_info=True)
        return False
    finally:
        if conn:
            conn.close()

# =========================================================
# ARTICLE ANALYSIS AND DATA PROCESSING FUNCTIONS
# =========================================================
class ClaudeNewsAnalyzer:
    """Handles interaction with Anthropic's Claude API for news analysis."""
    def __init__(self, api_key, model_name):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model_name = model_name

    def analyze_article_text(self, article_text_content, source_name_for_context):
        """
        Sends article text to Claude for analysis and returns the parsed JSON response.
        """
        max_chars_for_claude = 10000 # Claude has context window limits
        if len(article_text_content) > max_chars_for_claude:
            article_text_content = article_text_content[:max_chars_for_claude]
            app.logger.warning(f"Article content truncated to {max_chars_for_claude} characters for Claude API.")

        media_owner_display = media_owners.get(source_name_for_context, 'Unknown Owner')

        prompt = (
            'You are a highly analytical and neutral AI assistant specializing in news article reliability and content analysis. '
            'Your task is to dissect the provided news article.\n\n'
            f'Article Text:\n"""\n{article_text_content}\n"""\n\n'
            f'Source (for context, if known): {source_name_for_context}\n'
            f'Media Owner: {media_owner_display}\n\n'
            'Please perform the following analyses and return the results ONLY in a single, valid JSON object format. '
            'Do not include any explanatory text before or after the JSON object.\n\n'
            'JSON Fields:\n'
            '- "news_integrity": (Float, 0.0-1.0) Assess the overall integrity and trustworthiness of the information presented. Higher means more trustworthy.\n'
            '- "fact_check_needed_score": (Float, 0.0-1.0) Likelihood that the article\'s claims require external fact-checking. 1.0 means high likelihood.\n'
            '- "sentiment_score": (Float, 0.0-1.0) Overall emotional tone (0.0 negative, 0.5 neutral, 1.0 positive).\n'
            '- "bias_score": (Float, 0.0-1.0) Degree of perceived bias (0.0 low bias, 1.0 high bias).\n'
            '- "topics": (List of strings) Identify 3-5 main topics or keywords that accurately represent the core subject matter.\n'
            '- "key_arguments": (List of strings) Extract the main arguments or claims made by the author.\n'
            '- "mentioned_facts": (List of strings) List any specific facts, data, or statistics mentioned.\n'
            '- "author_purpose": (String) Briefly determine the author\'s likely primary purpose.\n'
            '- "potential_biases_identified": (List of strings) Enumerate any specific signs of potential bias or subjectivity observed.\n'
            '- "short_summary": (String) A concise summary of the article\'s main content in 2-4 sentences.\n'
            '- "index_of_credibility": (Float, 0.0-1.0) Calculate an overall index of credibility based on the above factors.\n'
            '- "published_date": (String, YYYY-MM-DD or N/A) The publication date of the article. Respond "N/A" if cannot be determined.'
        )

        try:
            message = self.client.messages.create(
                model=self.model_name,
                max_tokens=2000, # Max tokens for the AI's response
                temperature=0.2, # Lower temperature for more factual/less creative responses
                system='You are a JSON-generating expert. Always provide valid JSON.',
                messages=[{'role': 'user', 'content': prompt}]
            )

            raw_json_text = message.content[0].text.strip()
            # Claude often wraps JSON in ```json...```, extract it if present
            match = re.search(r'```json\s*(\{.*\})\s*```', raw_json_text, re.DOTALL)

            if match:
                json_str = match.group(1)
            else:
                json_str = raw_json_text # Assume it's pure JSON if no markdown block

            return json.loads(json_str)

        except anthropic.APIError as e:
            app.logger.error(f'Anthropic API Error: {e}', exc_info=True)
            raise ValueError(f'Error communicating with AI: {e}')
        except json.JSONDecodeError as e:
            app.logger.error(f'JSON Decode Error from Claude\'s response: {e}. Raw response was: {raw_json_text}', exc_info=True)
            raise ValueError(f'Failed to parse AI response: {e}. Raw: {raw_json_text[:500]}...')
        except Exception as e:
            app.logger.error(f'Unexpected error during Claude analysis: {e}', exc_info=True)
            raise ValueError(f'An unexpected error occurred during AI analysis: {e}')

def extract_text_from_url(url):
    """
    Extracts main article text, source, and title from a given URL.
    Uses the 'newspaper' library.
    """
    try:
        clean_url = re.sub(r'/amp(/)?$', '', url) # Remove AMP specific parts
        article = Article(clean_url)
        article.download()
        article.parse()
        text = article.text.strip()
        title = article.title.strip() if article.title else ''
        source = urlparse(clean_url).netloc.replace('www.', '') # Extract domain
        app.logger.info(f"Successfully extracted text from {clean_url}. Title: '{title[:50]}...' Source: {source}")
        return text, source, title
    except Exception as e:
        app.logger.error(f'Error extracting article from URL {url}: {e}', exc_info=True)
        return '', '', '' # Return empty strings on failure

def calculate_credibility_level(integrity, fact_check_needed, sentiment, bias):
    """
    Calculates an overall credibility level (High, Medium, Low) based on AI scores.
    """
    # Invert fact_check_needed: higher (closer to 1.0) is better
    fact_check_score = 1.0 - fact_check_needed
    # Proximity to neutral sentiment: closer to 0.5 is better (0.5 is 1.0, 0.0 or 1.0 is 0.0)
    neutral_sentiment_proximity = 1.0 - abs(sentiment - 0.5) * 2
    # Invert bias: lower (closer to 0.0) is better
    bias_score_inverted = 1.0 - bias

    # Weighted average for overall score (weights can be tuned)
    avg = (integrity * 0.45) + \
          (fact_check_score * 0.35) + \
          (neutral_sentiment_proximity * 0.10) + \
          (bias_score_inverted * 0.10)

    if avg >= 0.75:
        return 'High'
    if avg >= 0.5:
        return 'Medium'
    return 'Low'

def save_analysis_to_db(url, title, source, content, analysis_result):
    """
    Saves the analysis results to the 'news' table and updates 'source_stats'.
    """
    conn = None
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()

        # Extract scores from analysis_result, with defaults
        integrity = analysis_result.get('news_integrity', 0.0)
        fact_check_needed = analysis_result.get('fact_check_needed_score', 1.0)
        sentiment = analysis_result.get('sentiment_score', 0.5)
        bias = analysis_result.get('bias_score', 1.0)
        short_summary = analysis_result.get('short_summary', 'Summary not available.')
        index_of_credibility = analysis_result.get('index_of_credibility', 0.0)

        # Determine credibility level based on calculated score
        credibility_level = calculate_credibility_level(integrity, fact_check_needed, sentiment, bias)
        
        # Ensure a unique URL for text-only inputs in the database
        db_url = url if url else f'text_input_{datetime.now(UTC).timestamp()}'

        # Insert or update news analysis into the 'news' table
        c.execute('''INSERT INTO news (url, title, source, content, integrity, fact_check, sentiment, bias,
                             credibility_level, short_summary, index_of_credibility)
                             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                             ON CONFLICT(url) DO UPDATE SET
                             title=excluded.title, source=excluded.source, content=excluded.content,
                             integrity=excluded.integrity, fact_check=excluded.fact_check,
                             sentiment=excluded.sentiment, bias=excluded.bias,
                             credibility_level=excluded.credibility_level,
                             short_summary=excluded.short_summary,
                             index_of_credibility=excluded.index_of_credibility,
                             analysis_date=CURRENT_TIMESTAMP''',
                         (db_url, title, source, content, integrity, fact_check_needed, sentiment, bias,
                          credibility_level, short_summary, index_of_credibility))

        # Update source_stats table for the source of the analyzed article
        c.execute('SELECT high, medium, low, total_analyzed FROM source_stats WHERE source = ?', (source,))
        row = c.fetchone()

        if row: # If source already exists, update counts
            high, medium, low, total = row
            if credibility_level == 'High': high += 1
            elif credibility_level == 'Medium': medium += 1
            else: low += 1
            total += 1
            c.execute('''UPDATE source_stats SET high=?, medium=?, low=?, total_analyzed=? WHERE source=?''',
                        (high, medium, low, total, source))
        else: # If new source, insert new record
            high = 1 if credibility_level == 'High' else 0
            medium = 1 if credibility_level == 'Medium' else 0
            low = 1 if credibility_level == 'Low' else 0
            c.execute('''INSERT INTO source_stats (source, high, medium, low, total_analyzed)
                         VALUES (?, ?, ?, ?, ?)''', (source, high, medium, low, 1))

        conn.commit()
        app.logger.info(f"Analysis for '{title[:50]}...' saved/updated. Credibility: {credibility_level}")
        return credibility_level
    except sqlite3.Error as e:
        app.logger.error(f'Database error in save_analysis_to_db: {e}', exc_info=True)
        if conn:
            conn.rollback() # Rollback changes on error
        raise # Re-raise for error handling upstream
    finally:
        if conn:
            conn.close()

def process_article_analysis(input_text, source_name_manual):
    """
    Orchestrates the full article analysis process:
    1. Extracts content from URL or uses raw text.
    2. Sends content to Claude for AI analysis.
    3. Saves results to the database.
    4. Formats output for frontend display.
    """
    article_url = None
    article_content = input_text
    article_title = 'User-provided Text'
    source_name = source_name_manual if source_name_manual else 'Direct Input'

    # Determine if input is a URL and try to extract content
    if input_text.strip().startswith('http'):
        article_url = input_text.strip()
        app.logger.info(f'Input is a URL: {article_url}')
        content_from_url, source_from_url, title_from_url = extract_text_from_url(article_url)

        if content_from_url and len(content_from_url) >= 100:
            article_content, source_name, article_title = content_from_url, source_from_url, title_from_url
            app.logger.info(f'Extracted from URL. Source: {source_name}, Title: {article_title[:50]}...')
        else:
            if not content_from_url:
                return ('❌ Failed to extract content from the provided URL. Please check the link or provide text directly.', None, None)
            else:
                return ('❌ Extracted article content is too short for analysis (min 100 chars).', None, None)

    # Validate article content length
    if not article_content or len(article_content) < 100:
        return ('❌ Article content is too short for analysis (min 100 chars).', None, None)

    # Ensure source_name is set, even if still 'Unknown Source'
    if not source_name:
        source_name = 'Unknown Source'

    # Initialize Claude analyzer and perform analysis
    analyzer = ClaudeNewsAnalyzer(ANTHROPIC_API_KEY, MODEL_NAME)
    try:
        analysis_result = analyzer.analyze_article_text(article_content, source_name)
    except ValueError as e: # Catch specific value errors from analysis (e.g., JSON parse fail)
        return (f'❌ Error during AI analysis: {str(e)}', None, None)
    except Exception as e:
        app.logger.error(f'Critical error during Claude analysis: {e}', exc_info=True)
        return (f'❌ An unexpected error occurred during analysis: {str(e)}', None, None)

    # Save analysis results to database
    try:
        credibility_saved = save_analysis_to_db(article_url, article_title, source_name, article_content, analysis_result)
        app.logger.info(f'Analysis saved to DB. Overall Credibility: {credibility_saved}')
    except Exception as e:
        app.logger.error(f'Error saving analysis to database: {e}', exc_info=True)
        return (f'❌ Error saving analysis to database: {str(e)}', None, None)

    # Extract results for display
    ni = analysis_result.get('news_integrity', 0.0)
    fcn = analysis_result.get('fact_check_needed_score', 1.0)
    ss = analysis_result.get('sentiment_score', 0.5)
    bs = analysis_result.get('bias_score', 1.0)
    topics = analysis_result.get('topics', [])
    key_arguments = analysis_result.get('key_arguments', [])
    mentioned_facts = analysis_result.get('mentioned_facts', [])
    author_purpose = analysis_result.get('author_purpose', 'N/A')
    potential_biases_identified = analysis_result.get('potential_biases_identified', [])
    short_summary = analysis_result.get('short_summary', 'N/A')
    index_of_credibility = analysis_result.get('index_of_credibility', 0.0)

    # Convert fact_check_needed_score to a display-friendly factuality score
    factuality_display_score = 1.0 - fcn

    # Format output as Markdown
    output_md = (
        f'### 📊 Credibility Analysis for: "{article_title}"\n'
        f'**Source:** {source_name}\n'
        f'**Media Owner:** {media_owners.get(source_name, "Unknown Owner")}\n'
        f'**Overall Calculated Credibility:** **{credibility_saved}** ({index_of_credibility*100:.1f}%)'
        '\n\n---\n'
        '#### 📊 Analysis Scores:\n'
        f'- **Integrity Score:** {ni*100:.1f}% - Measures the overall integrity and trustworthiness.\n'
        f'- **Factuality Score:** {factuality_display_score*100:.1f}% - Indicates likelihood of needing fact-checking.\n'
        f'- **Sentiment Score:** {ss:.2f} - Overall emotional tone (0.0 negative, 0.5 neutral, 1.0 positive).\n'
        f'- **Bias Score:** {bs*100:.1f}% - Degree of perceived bias (0.0 low, 1.0 high).\n'
        f'- **Index of Credibility:** {index_of_credibility*100:.1f}% - Overall credibility index.'
        '\n\n---\n'
        '#### 📝 Summary:\n'
        f'{short_summary}\n\n'
        '#### 🔑 Key Arguments:\n'
        f'{("- " + "\\n- ".join(key_arguments) if key_arguments else "N/A")}\n\n'
        '#### 📈 Mentioned Facts/Data:\n'
        f'{("- " + "\\n- ".join(mentioned_facts) if mentioned_facts else "N/A")}\n\n'
        '#### 🎯 Author\'s Purpose:\n'
        f'{author_purpose}\n\n'
        '#### 🚩 Potential Biases Identified:\n'
        f'{("- " + "\\n- ".join(potential_biases_identified) if potential_biases_identified else "N/A")}\n\n'
        '#### 🏷️ Main Topics Identified:\n'
        f'{", ".join(topics) if topics else "N/A"}\n\n'
        '#### 📌 Media Owner Influence:\n'
        f'The media owner, {media_owners.get(source_name, "Unknown Owner")}, may influence source credibility.'
    )

    # Prepare scores for charting on the frontend
    scores_for_chart = {
        'Integrity': ni * 100,
        'Factuality': factuality_display_score * 100,
        'Neutral Sentiment': (1.0 - abs(ss - 0.5) * 2) * 100, # Score how close to neutral
        'Low Bias': (1.0 - bs) * 100, # Invert bias for a 'low bias' score
        'Overall Credibility Index': index_of_credibility * 100
    }

    return output_md, scores_for_chart, analysis_result

def generate_query(analysis_result):
    """Generates a search query for NewsAPI based on article topics, arguments, and facts."""
    topics = analysis_result.get('topics', [])
    key_arguments = analysis_result.get('key_arguments', [])
    mentioned_facts = analysis_result.get('mentioned_facts', [])

    all_terms = []

    # Prioritize multi-word phrases by enclosing them in quotes
    for phrase_list in [topics, key_arguments]:
        for phrase in phrase_list:
            if not phrase.strip(): continue
            if ' ' in phrase.strip() and len(phrase.strip().split()) > 1:
                all_terms.append('"' + phrase.strip() + '"')
            else:
                all_terms.append(phrase.strip())

    # Add relevant words from mentioned facts, excluding stop words
    for fact in mentioned_facts:
        if not fact.strip(): continue
        words = [word for word in fact.lower().split() if word not in stop_words_en and len(word) > 2]
        all_terms.extend(words)

    unique_terms = list(set(all_terms)) # Remove duplicates

    # Formulate query: prefer 'AND' for specificity, fall back to 'OR' or general query
    if len(unique_terms) >= 3:
        query = ' AND '.join(unique_terms)
    elif unique_terms:
        query = ' OR '.join(unique_terms)
    else:
        query = 'current events OR news' # Fallback query if no specific terms

    app.logger.info(f'Generated NewsAPI query: {query}')
    return query

def make_newsapi_request(params):
    """Helper function to make requests to the NewsAPI."""
    url = 'https://newsapi.org/v2/everything'
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        data = response.json()
        return data.get('articles', [])
    except requests.exceptions.RequestException as e:
        app.logger.error(f'NewsAPI Request Error: {e}', exc_info=True)
        if hasattr(e, 'response') and e.response is not None:
            app.logger.error(f'NewsAPI detailed error response: {e.response.text}')
        return []
    except Exception as e:
        app.logger.error(f'Unexpected error during NewsAPI request: {e}', exc_info=True)
        return []

def fetch_same_topic_news(analysis_result, days_range=7, max_articles=3):
    """
    Fetches news articles on the same topic using NewsAPI,
    prioritizing trusted sources and relevance.
    """
    if not NEWS_API_ENABLED:
        app.logger.warning('NEWS_API_KEY is not configured or enabled. Skipping similar news search.')
        return []

    initial_query = generate_query(analysis_result)
    
    # Determine date range for NewsAPI search
    original_published_date_str = analysis_result.get('published_date', 'N/A')
    end_date = datetime.now(UTC).date()

    if original_published_date_str and original_published_date_str != 'N/A':
        try:
            # Parse published date and create a range around it
            parsed_date = datetime.strptime(original_published_date_str, '%Y-%m-%d').date()
            start_date = parsed_date - timedelta(days=days_range)
            end_date = parsed_date + timedelta(days=days_range) 
            app.logger.info(f'Using original article date ({parsed_date}) for NewsAPI search range: {start_date} to {end_date}')
        except ValueError:
            app.logger.warning(f'Could not parse original article date "{original_published_date_str}". Using default range (last {days_range} days).')
            start_date = end_date - timedelta(days=days_range)
    else:
        start_date = end_date - timedelta(days=days_range)
        app.logger.info(f'No original article date found. Using default NewsAPI search range: {start_date} to {end_date}.')

    # Attempt 1: Specific query with trusted sources
    params_specific = {
        'q': initial_query,
        'apiKey': NEWS_API_KEY,
        'language': 'en',
        'pageSize': max_articles * 2, # Fetch more to filter later
        'sortBy': 'relevancy',
        'from': start_date.strftime('%Y-%m-%d'),
        'to': end_date.strftime('%Y-%m-%d'),
    }

    if TRUSTED_NEWS_SOURCES_IDS:
        params_specific['sources'] = ','.join(TRUSTED_NEWS_SOURCES_IDS)

    articles_found = make_newsapi_request(params_specific)
    app.logger.info(f'[NewsAPI] Attempt 1 found {len(articles_found)} articles with specific query and trusted sources.')

    # Attempt 2: Broader query if first attempt yielded few results
    if len(articles_found) < (max_articles / 2) and initial_query != 'current events OR news':
        app.logger.info('Few results from specific query, attempting broader search.')
        # Use main topics for a broader search
        broader_query_terms = list(set(analysis_result.get('topics', [])[:3]))
        broader_query = ' OR '.join([f'"{term}"' if ' ' in term else term for term in broader_query_terms if term and term not in stop_words_en])

        if not broader_query:
            broader_query = 'current events OR news' # Fallback for broader if topics are empty

        params_broad = {
            'q': broader_query,
            'apiKey': NEWS_API_KEY,
            'language': 'en',
            'pageSize': max_articles * 2,
            'sortBy': 'relevancy',
            'from': start_date.strftime('%Y-%m-%d'),
            'to': end_date.strftime('%Y-%m-%d'),
        }

        if TRUSTED_NEWS_SOURCES_IDS:
            params_broad['sources'] = ','.join(TRUSTED_NEWS_SOURCES_IDS)

        additional_articles = make_newsapi_request(params_broad)
        articles_found.extend(additional_articles)
        app.logger.info(f'[NewsAPI] Attempt 2 found {len(additional_articles)} new articles. Total unique: {len(articles_found)}')

    # Remove duplicate articles based on URL
    unique_articles = {}
    for article in articles_found:
        if article.get('url'):
            unique_articles[article['url']] = article
    articles_found = list(unique_articles.values())

    if not articles_found:
        app.logger.info('No unique articles found after all NewsAPI attempts.')
        return []

    # Rank articles by relevance and trust
    ranked_articles = []
    
    # Combine all potential query terms for relevance scoring
    all_query_terms = []
    # Add terms from initial query (handle 'AND' and quotes)
    all_query_terms.extend([t.lower().replace('"', '') for t in initial_query.split(' AND ') if t.strip()])
    # Add terms from broader query (handle 'OR' and quotes)
    if 'broader_query' in locals(): # Check if broader_query was defined
        all_query_terms.extend([t.lower().replace('"', '') for t in broader_query.split(' OR ') if t.strip()])
    all_query_terms = list(set([t for t in all_query_terms if t and t not in stop_words_en]))

    for article in articles_found:
        source_domain = urlparse(article.get('url', '')).netloc.replace('www.', '')
        # Get trust score from predefined list, default to 0.5 (medium)
        trust_score = predefined_trust_scores.get(source_domain, 0.5) 
        
        article_text_for_relevance = (article.get('title', '') + ' ' + article.get('description', '')).lower()
        
        # Calculate relevance: count how many query terms appear in article title/description
        relevance_score = sum(1 for term in all_query_terms if term in article_text_for_relevance)
        
        # Combine relevance (integer count) and trust (float 0-1)
        # Weights can be adjusted based on desired impact of each factor
        final_score = (relevance_score * 10) + (trust_score * 5) 
        ranked_articles.append((article, final_score))

    # Sort articles by the combined score in descending order
    ranked_articles.sort(key=lambda item: item[1], reverse=True)
    top_articles = [item[0] for item in ranked_articles[:max_articles]]
    app.logger.info(f'Returning {len(top_articles)} top ranked similar articles.')
    return top_articles

def render_same_topic_articles_html(articles):
    """
    Generates HTML string for displaying a list of similar articles.
    Includes historical source credibility if available in DB.
    """
    if not articles:
        return '<p>No same topic articles found for the selected criteria.</p>'

    conn = None 
    html_items = []
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()

        for art in articles:
            # HTML escape all content to prevent XSS
            title = html.escape(art.get('title', 'No Title'))
            article_url = html.escape(art.get('url', '#'))
            source_api_name = html.escape(art.get('source', {}).get('name', 'Unknown Source'))

            published_at_raw = art.get('publishedAt', 'N/A')
            # Format date for display
            published_at_display = html.escape(published_at_raw.split('T')[0] if 'T' in published_at_raw and published_at_raw != 'N/A' else published_at_raw)

            description_raw = art.get('description', 'No description available.')
            # Attempt to remove title from start of description if present (common in NewsAPI)
            if title != 'No Title' and description_raw.startswith(art.get('title', '')):
                description_raw = description_raw[len(art.get('title', '')):].strip()
                if description_raw.startswith('- '): 
                    description_raw = description_raw[2:].strip()
            description_display = html.escape(description_raw)

            # Get domain for source lookup
            domain = urlparse(art.get('url', '#')).netloc.replace('www.', '')
            trust_display = ''

            # Check historical credibility from your DB (source_stats)
            c.execute('SELECT high, medium, low, total_analyzed FROM source_stats WHERE source = ?', (domain,))
            row = c.fetchone()

            if row:
                high, medium, low, total_analyzed = row
                if total_analyzed > 0:
                    score = (high * 1.0 + medium * 0.5 + low * 0.0) / total_analyzed
                    trust_display = f' (Hist. Src. Credibility: {int(score*100)}%)'
            
            # If no historical data, use predefined trust scores
            if not trust_display and domain in predefined_trust_scores:
                predefined_score = predefined_trust_scores.get(domain)
                trust_display = f' (Predefined Credibility: {int(predefined_score*100)}%)'


            html_items.append(
                f'<div class="similar-article">'
                f'<h4><a href="{article_url}" target="_blank" rel="noopener noreferrer">{title}</a></h4>'
                f'<p><strong>Source:</strong> {source_api_name}{trust_display} | <strong>Published:</strong> {published_at_display}</p>'
                f'<p>{description_display}</p>'
                f'</div>'
                f'<hr>' # Horizontal rule for separation
            )

        return (
            '<div class="similar-articles-container">'
            '<h3>🔗 Same Topic News Articles (Ranked by Relevance & Trust):</h3>'
            ''.join(html_items) +
            '</div>'
        )
    except sqlite3.Error as e:
        app.logger.error(f'Database error in render_similar_articles_html: {e}', exc_info=True)
        return '<p>Error retrieving same topic articles data due to a database issue.</p>'
    except Exception as e: 
        app.logger.error(f'Unexpected error in render_same_topic_articles_html: {e}', exc_info=True)
        return '<p>An unexpected error occurred while rendering similar articles.</p>'
    finally:
        if conn:
            conn.close()

def get_source_reliability_data():
    """Retrieves source reliability data for charting and display."""
    conn = None
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()

        c.execute('''
            SELECT source, high, medium, low, total_analyzed
            FROM source_stats
            ORDER BY total_analyzed DESC, source ASC
        ''')

        data = c.fetchall()
        sources = []
        credibility_indices = []
        high_counts = []
        medium_counts = []
        low_counts = []
        total_analyzed_counts = []

        for source, high, medium, low, total in data:
            total_current = high + medium + low
            if total_current == 0:
                score = 0.5 # Default score for sources with no analyses yet
            else:
                score = (high * 1.0 + medium * 0.5 + low * 0.0) / total_current

            sources.append(source)
            credibility_indices.append(round(score, 2))
            high_counts.append(high)
            medium_counts.append(medium)
            low_counts.append(low)
            total_analyzed_counts.append(total_current)

        return {
            'sources': sources,
            'credibility_indices_for_plot': credibility_indices,
            'high_counts': high_counts,
            'medium_counts': medium_counts,
            'low_counts': low_counts,
            'total_analyzed_counts': total_analyzed_counts
        }
    except sqlite3.Error as e:
        app.logger.error(f'Database error in get_source_reliability_data: {e}', exc_info=True)
        return { # Return empty data on error
            'sources': [], 'credibility_indices_for_plot': [],
            'high_counts': [], 'medium_counts': [], 'low_counts': [],
            'total_analyzed_counts': []
        }
    finally:
        if conn:
            conn.close()

def get_analysis_history_html():
    """Generates HTML string for displaying recent analysis history."""
    conn = None
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()

        c.execute('''
            SELECT url, title, source, credibility_level, short_summary,
                   strftime("%Y-%m-%d %H:%M", analysis_date) as formatted_date
            FROM news
            ORDER BY analysis_date DESC
            LIMIT 15
        ''')

        rows = c.fetchall()

        if not rows:
            return '<p>No analysis history yet. Analyze an article to see it appear here!</p>'

        html_items = []
        for url, title, source, credibility, short_summary, date_str in rows:
            # Shorten title if too long
            display_title = title[:70] + '...' if title and len(title) > 70 else title if title else 'N/A'
            source_display = source if source else 'N/A'
            # Create a clickable link if a valid URL exists
            link_start = f'<a href="{html.escape(url)}" target="_blank" rel="noopener noreferrer">' if url and url.startswith(('http://', 'https://')) else ''
            link_end = '</a>' if url and url.startswith(('http://', 'https://')) else ''
            summary_display = short_summary if short_summary else 'No summary available.'

            html_items.append(
                f'<li>'
                f'<strong>{html.escape(date_str)}</strong>: {link_start}{html.escape(display_title)}{link_end} ({html.escape(source_display)}, {html.escape(credibility)})'
                f'<br>'
                f'<em>Summary:</em> {html.escape(summary_display)}'
                f'</li>'
            )

        return '<h3>📜 Recent Analyses:</h3><ul>' + ''.join(html_items) + '</ul>'
    except sqlite3.Error as e:
        app.logger.error(f'Database error in get_analysis_history_html: {e}', exc_info=True)
        return '<p>Error retrieving analysis history due to a database issue.</p>'
    finally:
        if conn:
            conn.close()

# =========================================================
# FLASK ROUTES
# =========================================================
@app.route('/')
def index():
    """Renders the main application page."""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """
    API endpoint to analyze an article (URL or text) and return analysis results.
    """
    try:
        data = request.json
        input_text = data.get('input_text')
        source_name_manual = data.get('source_name_manual')

        if not input_text:
            return jsonify({'error_message': 'No input text or URL provided.'}), 400

        app.logger.info(f'Received input_text (first 50 chars): {input_text[:50]}...')
        app.logger.info(f'Received source_name_manual: {source_name_manual}')

        # Perform the analysis
        output_md, scores_for_chart, analysis_result = process_article_analysis(input_text, source_name_manual)

        if analysis_result is None: # This means process_article_analysis returned an error message
            return jsonify({'error_message': output_md}), 400

        # Fetch and render similar news if analysis was successful
        same_topic_news = fetch_same_topic_news(analysis_result)
        same_topic_html = render_same_topic_articles_html(same_topic_news)

        app.logger.info('Analysis result generated and similar news fetched. Sending to client.')
        return jsonify({
            'output_md': output_md,
            'scores_for_chart': scores_for_chart,
            'analysis_result': analysis_result, # Send this for potential frontend use in subsequent requests
            'same_topic_news': same_topic_html
        })
    except Exception as e:
        app.logger.error(f'Error in /analyze endpoint: {e}', exc_info=True) # Log full traceback
        return jsonify({'error_message': f'An unexpected error occurred during analysis: {str(e)}'}), 500

@app.route('/same_topic_articles', methods=['POST'])
def same_topic_articles_endpoint():
    """
    API endpoint to fetch and render HTML for same topic articles based on a prior analysis result.
    """
    try:
        data = request.json
        analysis_result = data.get('analysis_result')

        if not analysis_result:
            return jsonify({'same_topic_html': '<p>No analysis result provided to fetch same topic articles.</p>'}), 400

        similar_articles_list = fetch_same_topic_news(analysis_result)
        return jsonify({
            'same_topic_html': render_same_topic_articles_html(similar_articles_list)
        })
    except Exception as e:
        app.logger.error(f'Error in /same_topic_articles endpoint: {e}', exc_info=True)
        return jsonify({'same_topic_html': f'<p>Error fetching same topic articles: {str(e)}</p>'}), 500

@app.route('/source_reliability_data')
def source_reliability_data_endpoint():
    """
    API endpoint to provide data for the source reliability chart.
    """
    try:
        data = get_source_reliability_data()

        labeled_sources = []
        if data['sources'] and data['credibility_indices_for_plot']:
            for source, score, total in zip(
                data['sources'],
                data['credibility_indices_for_plot'],
                data['total_analyzed_counts']
            ):
                # Format source labels for Plotly tooltip/hover text
                labeled_sources.append(
                    f"{html.escape(source)}<br>Credibility: {int(score*100)}%<br>Articles: {total}"
                )

        return jsonify({
            'sources': labeled_sources,
            'credibility_indices_for_plot': data['credibility_indices_for_plot'],
            'high_counts': data['high_counts'],
            'medium_counts': data['medium_counts'],
            'low_counts': data['low_counts'],
            'total_analyzed_counts': data['total_analyzed_counts']
        })
    except Exception as e:
        app.logger.error(f'Error in /source_reliability_data endpoint: {e}', exc_info=True)
        return jsonify({ # Return empty data on error
            'sources': [], 'credibility_indices_for_plot': [],
            'high_counts': [], 'medium_counts': [], 'low_counts': [],
            'total_analyzed_counts': []
        }), 500

@app.route('/analysis_history')
def analysis_history_endpoint():
    """
    API endpoint to fetch and render HTML for the recent analysis history.
    """
    try:
        history_html = get_analysis_history_html()
        return jsonify({'history_html': history_html})
    except Exception as e:
        app.logger.error(f'Error in /analysis_history endpoint: {e}', exc_info=True)
        return jsonify({'history_html': f'<p>Error retrieving analysis history: {str(e)}</p>'}), 500

@app.route('/feedback', methods=['POST'])
def handle_feedback():
    """
    API endpoint for handling feedback form submissions.
    Accepts JSON data, validates it, and saves to the database.
    """
    try:
        data = request.get_json()

        required_fields = ['name', 'email', 'type', 'message']
        if not all(field in data for field in required_fields):
            app.logger.warning(f"Feedback submission missing fields: {', '.join([f for f in required_fields if f not in data])} from {request.remote_addr}")
            return jsonify({'message': 'All fields are required'}), 400

        if not re.match(r"[^@]+@[^@]+\.[^@]+", data['email']):
            app.logger.warning(f"Feedback submission with invalid email: {data.get('email')} from {request.remote_addr}")
            return jsonify({'message': 'Invalid email address'}), 400

        conn = None
        try:
            conn = sqlite3.connect(DB_NAME)
            c = conn.cursor()

            c.execute('''
                INSERT INTO feedback (name, email, type, message, date)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                data['name'],
                data['email'],
                data['type'],
                data['message'],
                datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')
            ))

            conn.commit()
            app.logger.info(f"Feedback received from {data['email']} (Type: {data['type']}) from {request.remote_addr}")
            return jsonify({
                'status': 'success',
                'message': 'Thank you for your feedback! We appreciate it.'
            })
        except sqlite3.Error as e:
            app.logger.error(f"Database error handling feedback from {request.remote_addr}: {e}", exc_info=True)
            return jsonify({
                'status': 'error',
                'message': 'Error saving your feedback. Please try again.'
            }), 500
        finally:
            if conn:
                conn.close()

    except Exception as e:
        app.logger.error(f"Unexpected error handling feedback from {request.remote_addr}: {e}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': 'An unexpected error occurred. Please try again.'
        }), 500

@app.route('/feedback')
def feedback_page():
    """Renders the feedback form HTML page."""
    return render_template('feedback.html')

# =========================================================
# APPLICATION INITIALIZATION AND RUN
# =========================================================
def initialize_application():
    """Performs all necessary setup when the application starts."""
    # Logging is already set up at the top of the file
    
    # Check for required environment variables (already done globally)

    # Initialize database schema and initial source data
    app.logger.info("Initializing database schema and sources...")
    ensure_db_schema()
    initialize_sources(INITIAL_SOURCE_COUNTS)
    check_database_integrity()
    app.logger.info("Database initialization complete.")

    app.logger.info("Flask application initialized and ready to serve.")

if __name__ == '__main__':
    # Perform application initialization tasks
    initialize_application()
    
    # Run the Flask application
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
