from flask import Flask, render_template, request, jsonify, abort
import os
import logging
from logging.handlers import RotatingFileHandler
from werkzeug.middleware.proxy_fix import ProxyFix
import sqlite3
import requests
import anthropic
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from datetime import datetime, timedelta, UTC
import re
import plotly.graph_objects as go
from stop_words import get_stop_words
from newspaper import Article
import html

# Initialize Flask app with enhanced security
app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

# Enhanced logging configuration
def setup_logging():
    """Configure comprehensive logging system."""
    file_handler = RotatingFileHandler(
        'app.log',
        maxBytes=1024 * 1024,
        backupCount=5
    )
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    ))

    app.logger.addHandler(file_handler)
    app.logger.addHandler(console_handler)
    app.logger.setLevel(logging.INFO)

    # Enhanced WordPress scanner filter
    class WordPressFilter(logging.Filter):
        def filter(self, record):
            wordpress_paths = [
                'wp-admin', 'wp-includes', 'wp-content', 'xmlrpc.php',
                'wp-login.php', 'wp-config.php', 'readme.html', 'license.txt',
                'wp-json', 'wp-comments-post.php', 'wp-trackback.php',
                'wp-signup.php', 'wp-activate.php', 'wp-blog-header.php',
                'wp-cron.php', 'wp-links-opml.php', 'wp-mail.php',
                'wp-settings.php', 'wp-config-sample.php', 'wp-load.php'
            ]
            return not any(path in str(record.msg) for path in wordpress_paths)

    logging.getLogger('werkzeug').addFilter(WordPressFilter())
    logging.getLogger('werkzeug').setLevel(logging.WARNING)

# Load environment variables
load_dotenv()
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
NEWS_API_KEY = os.getenv('NEWS_API_KEY')
NEWS_API_ENABLED = bool(NEWS_API_KEY)
MODEL_NAME = "claude-3-opus-20240229"

# Check for API keys
if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY is missing! Please set it in your .env file.")
if not NEWS_API_KEY:
    app.logger.warning("NEWS_API_KEY is missing! Similar news functionality will be disabled.")

# Database configuration
DB_NAME = 'news_analysis.db'

# Enhanced WordPress scanner protection
WORDPRESS_PATHS = [
    'wp-admin', 'wp-includes', 'wp-content', 'xmlrpc.php',
    'wp-login.php', 'wp-config.php', 'readme.html', 'license.txt',
    'wp-json', 'wp-comments-post.php', 'wp-trackback.php',
    'wp-signup.php', 'wp-activate.php', 'wp-blog-header.php',
    'wp-cron.php', 'wp-links-opml.php', 'wp-mail.php',
    'wp-settings.php', 'wp-config-sample.php', 'wp-load.php'
]

# Middleware to block WordPress scanners
@app.before_request
def block_wordpress_scanners():
    """Block requests from WordPress scanners."""
    if any(path in request.path.lower() for path in WORDPRESS_PATHS):
        app.logger.warning(f"Blocked WordPress scanner request from {request.remote_addr} to: {request.path}")
        return abort(404)

    # Additional protection against common scanner patterns
    if request.path.lower().endswith(('.php', '.bak', '.sql', '.log')):
        app.logger.warning(f"Blocked suspicious request to: {request.path}")
        return abort(404)

# Security headers middleware
@app.after_request
def add_security_headers(response):
    """Add security headers to responses."""
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    response.headers['Content-Security-Policy'] = "default-src 'self'"
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    response.headers['Permissions-Policy'] = 'geolocation=(), microphone=(), camera=()'
    return response

# Error handlers
@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors."""
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    """Handle 500 errors."""
    app.logger.error(f"Server error: {str(e)}")
    return render_template('500.html'), 500

# Database functions
def ensure_db_schema():
    """Ensure database schema exists."""
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
        conn.commit()
        app.logger.info("Database schema ensured successfully")
    except sqlite3.Error as e:
        app.logger.error(f"Error ensuring database schema: {str(e)}")
        raise
    finally:
        if conn:
            conn.close()

# Rest of your application code remains the same
# (ClaudeNewsAnalyzer, extract_text_from_url, calculate_credibility_level, etc.)
# (process_article_analysis, generate_query, fetch_similar_news, etc.)
# (render_similar_articles_html, get_source_reliability_data, etc.)

# Initialize database
def initialize_database():
    """Initialize database schema and data."""
    ensure_db_schema()
    check_database_integrity()

if __name__ == '__main__':
    # Setup logging
    setup_logging()

    # Initialize database
    initialize_database()

    # Run application
    app.run(host='0.0.0.0', port=5000)
