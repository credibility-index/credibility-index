import os
import logging
import re
import json
import requests
import html
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse, urlunparse
from flask import Flask, request, jsonify, render_template, session, make_response
from werkzeug.middleware.proxy_fix import ProxyFix
from flask_cors import CORS
import anthropic
from newspaper import Article, Config
from stop_words import get_stop_words
from pg_database import PostgresDB  # Импортируем наш класс для работы с PostgreSQL

# Initialize Flask application
app = Flask(__name__, static_folder='static', template_folder='templates')
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)
app.secret_key = os.getenv('SECRET_KEY', 'your-secret-key-here')

# Configure CORS for Railway
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Environment variables
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
MODEL_NAME = os.getenv('ANTHROPIC_MODEL', 'claude-3-opus-20240229')
NEWS_API_KEY = os.getenv('NEWS_API_KEY')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize PostgreSQL database
pg_db = PostgresDB()

# Initial data
INITIAL_SOURCE_COUNTS = {
    'bbc.com': {'high': 15, 'medium': 5, 'low': 1},
    'reuters.com': {'high': 20, 'medium': 3, 'low': 0},
    'foxnews.com': {'high': 3, 'medium': 7, 'low': 15},
    'cnn.com': {'high': 5, 'medium': 10, 'low': 5},
    'nytimes.com': {'high': 10, 'medium': 5, 'low': 2},
    'theguardian.com': {'high': 12, 'medium': 4, 'low': 1},
    'apnews.com': {'high': 18, 'medium': 2, 'low': 0}
}

media_owners = {
    'bbc.com': 'BBC',
    'reuters.com': 'Thomson Reuters',
    'foxnews.com': 'Fox Corporation',
    'cnn.com': 'Warner Bros. Discovery',
    'nytimes.com': 'The New York Times Company',
    'theguardian.com': 'Guardian Media Group',
    'apnews.com': 'Associated Press',
    'aljazeera.com': 'Al Jazeera Media Network',
    'wsj.com': 'News Corp'
}

predefined_trust_scores = {
    'bbc.com': 0.9, 'bbc.co.uk': 0.9, 'reuters.com': 0.95, 'apnews.com': 0.93,
    'nytimes.com': 0.88, 'theguardian.com': 0.85, 'wsj.com': 0.82,
    'cnn.com': 0.70, 'foxnews.com': 0.40, 'aljazeera.com': 0.80
}

TRUSTED_NEWS_SOURCES_IDS = [
    'bbc-news', 'reuters', 'associated-press', 'the-new-york-times',
    'the-guardian-uk', 'the-wall-street-journal', 'cnn', 'al-jazeera-english',
    'bloomberg', 'the-washington-post', 'fox-news', 'nbc-news', 'cbs-news',
    'abc-news', 'usa-today', 'the-verge', 'techcrunch', 'wired'
]

stop_words_en = get_stop_words('en')

# Configure newspaper library
user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
config = Config()
config.browser_user_agent = user_agent
config.request_timeout = 30

@app.after_request
def after_request(response):
    """Add CORS headers to all responses"""
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors"""
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    """Handle 500 errors"""
    return render_template('500.html'), 500

@app.before_request
def before_request():
    """Set up before each request"""
    if request.path.startswith('/static/'):
        return

    if request.method == 'OPTIONS':
        response = make_response()
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        return response

@app.after_request
def add_security_headers(response):
    """Add security headers"""
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    return response

@app.route('/')
def home():
    """Home page"""
    return render_template('index.html')

@app.route('/faq')
def faq():
    """FAQ page"""
    return render_template('faq.html')

@app.route('/privacy')
def privacy():
    """Privacy Policy page"""
    return render_template('privacy.html')

@app.route('/terms')
def terms():
    """Terms of Service page"""
    return render_template('terms.html')

@app.route('/feedback', methods=['GET', 'POST'])
def feedback():
    """Feedback page and form handler"""
    if request.method == 'POST':
        try:
            # Check if request has JSON content
            if not request.is_json:
                return jsonify({
                    'status': 'error',
                    'message': 'Request must be JSON'
                }), 400

            data = request.get_json()
            name = data.get('name')
            email = data.get('email')
            feedback_type = data.get('type')
            message = data.get('message')

            # Validate required fields
            if not all([name, email, feedback_type, message]):
                return jsonify({
                    'status': 'error',
                    'message': 'All fields are required'
                }), 400

            # Validate email format
            if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
                return jsonify({
                    'status': 'error',
                    'message': 'Invalid email address'
                }), 400

            # Save feedback to database
            try:
                conn = pg_db.get_conn()
                with conn.cursor() as cursor:
                    cursor.execute('''
                        INSERT INTO media_credibility.feedback (name, email, type, message)
                        VALUES (%s, %s, %s, %s)
                    ''', (name, email, feedback_type, message))
                conn.commit()
                pg_db.release_conn(conn)

                return jsonify({
                    'status': 'success',
                    'message': 'Thank you for your feedback! We appreciate your input.'
                })
            except Exception as e:
                logger.error(f'Database error saving feedback: {str(e)}')
                return jsonify({
                    'status': 'error',
                    'message': 'Database error occurred. Please try again later.'
                }), 500

        except Exception as e:
            logger.error(f'Error saving feedback: {str(e)}')
            return jsonify({
                'status': 'error',
                'message': 'An unexpected error occurred. Please try again later.'
            }), 500

    # For GET requests, return the feedback page template
    return render_template('feedback.html')

@app.route('/daily-buzz', methods=['GET'])
def daily_buzz():
    """Get the daily buzz article with full analysis"""
    try:
        # Get today's buzz article
        article = pg_db.get_buzz_article()

        if article:
            votes = get_article_votes(article['id'])
            return jsonify({
                'status': 'success',
                'article': article,
                'votes': votes
            })
        else:
            # Create default article if none exists
            default_article = {
                'id': 0,
                'title': 'Israel-Iran Conflict: Current Situation Analysis',
                'source': 'Media Credibility Index',
                'url': '#',
                'short_summary': 'Ongoing tensions between Israel and Iran continue to escalate.',
                'credibility_level': 'Medium',
                'analysis_date': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
                'content': 'The conflict between Israel and Iran has reached a critical point with recent escalations in military and diplomatic tensions.',
                'analysis': {
                    'news_integrity': 0.75,
                    'fact_check_needed_score': 0.25,
                    'sentiment_score': 0.4,
                    'bias_score': 0.3,
                    'topics': ['Israel-Iran conflict', 'Middle East tensions'],
                    'key_arguments': [
                        'The conflict has reached a critical point',
                        'International mediators are attempting to broker peace'
                    ],
                    'mentioned_facts': [
                        'Recent military exercises conducted by both nations',
                        'Diplomatic efforts led by the United Nations'
                    ],
                    'author_purpose': 'To inform about the current state of Israel-Iran relations',
                    'potential_biases_identified': [
                        'Possible pro-Western perspective'
                    ],
                    'short_summary': 'Ongoing tensions between Israel and Iran continue to escalate.',
                    'index_of_credibility': 0.65
                }
            }

            # Save default article to database
            try:
                conn = pg_db.get_conn()
                with conn.cursor() as cursor:
                    # Save article
                    cursor.execute('''
                        INSERT INTO media_credibility.news
                        (title, source, content, integrity, fact_check, sentiment, bias,
                         credibility_level, index_of_credibility, url, analysis_date, short_summary)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        RETURNING id
                    ''', (
                        default_article['title'],
                        default_article['source'],
                        default_article['content'],
                        default_article['analysis']['news_integrity'],
                        default_article['analysis']['fact_check_needed_score'],
                        default_article['analysis']['sentiment_score'],
                        default_article['analysis']['bias_score'],
                        default_article['credibility_level'],
                        default_article['analysis']['index_of_credibility'],
                        default_article['url'],
                        datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
                        default_article['short_summary']
                    ))
                    article_id = cursor.fetchone()['id']

                    # Set as today's buzz article
                    today = datetime.now(timezone.utc).date()
                    cursor.execute('''
                        INSERT INTO media_credibility.daily_buzz (article_id, date)
                        VALUES (%s, %s)
                    ''', (article_id, today))

                    # Get the article with votes
                    cursor.execute('''
                        SELECT n.*, s.high, s.medium, s.low, s.total_analyzed
                        FROM media_credibility.news n
                        LEFT JOIN media_credibility.source_stats s ON n.source = s.source
                        WHERE n.id = %s
                    ''', (article_id,))
                    article = cursor.fetchone()

                conn.commit()
                pg_db.release_conn(conn)

                votes = get_article_votes(article_id)

                return jsonify({
                    'status': 'success',
                    'article': article,
                    'votes': votes
                })
            except Exception as e:
                logger.error(f"Error creating default buzz article: {str(e)}")
                return jsonify({
                    'status': 'error',
                    'message': 'Error creating default buzz article'
                }), 500

    except Exception as e:
        logger.error(f"Error in daily_buzz endpoint: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': 'An error occurred while fetching daily buzz'
        }), 500

def get_article_votes(article_id):
    """Get votes for an article"""
    try:
        conn = pg_db.get_conn()
        with conn.cursor() as cursor:
            # Get votes
            cursor.execute('''
                SELECT
                    COUNT(*) FILTER (WHERE vote_type = 'upvote') as upvotes,
                    COUNT(*) FILTER (WHERE vote_type = 'downvote') as downvotes
                FROM media_credibility.article_votes
                WHERE article_id = %s
            ''', (article_id,))
            votes = cursor.fetchone()

            # Get credibility ratings
            cursor.execute('''
                SELECT
                    AVG(credibility_rating) as avg_rating,
                    COUNT(*) as rating_count
                FROM media_credibility.credibility_votes
                WHERE article_id = %s
            ''', (article_id,))
            credibility = cursor.fetchone()

            return {
                'upvotes': votes['upvotes'] if votes else 0,
                'downvotes': votes['downvotes'] if votes else 0,
                'avg_rating': credibility['avg_rating'] if credibility else 0,
                'rating_count': credibility['rating_count'] if credibility else 0
            }
    except Exception as e:
        logger.error(f"Error getting article votes: {str(e)}")
        return {
            'upvotes': 0,
            'downvotes': 0,
            'avg_rating': 0,
            'rating_count': 0
        }
    finally:
        pg_db.release_conn(conn)

@app.route('/vote', methods=['POST'])
def vote():
    """Vote for an article"""
    try:
        data = request.get_json()
        article_id = data.get('article_id')
        user_id = data.get('user_id')
        vote_type = data.get('vote_type')

        if not all([article_id, user_id, vote_type]):
            return jsonify({
                'status': 'error',
                'message': 'Missing required parameters'
            }), 400

        if vote_type not in ['upvote', 'downvote']:
            return jsonify({
                'status': 'error',
                'message': 'Invalid vote type'
            }), 400

        try:
            conn = pg_db.get_conn()
            with conn.cursor() as cursor:
                # Check if user already voted
                cursor.execute('''
                    SELECT id FROM media_credibility.article_votes
                    WHERE article_id = %s AND user_id = %s
                ''', (article_id, user_id))
                existing_vote = cursor.fetchone()

                if existing_vote:
                    # Update existing vote
                    cursor.execute('''
                        UPDATE media_credibility.article_votes
                        SET vote_type = %s, created_at = CURRENT_TIMESTAMP
                        WHERE id = %s
                    ''', (vote_type, existing_vote['id']))
                else:
                    # Add new vote
                    cursor.execute('''
                        INSERT INTO media_credibility.article_votes (article_id, user_id, vote_type)
                        VALUES (%s, %s, %s)
                    ''', (article_id, user_id, vote_type))

                conn.commit()

            votes = get_article_votes(article_id)
            return jsonify({
                'status': 'success',
                'message': 'Vote recorded successfully',
                'votes': votes
            })
        except Exception as e:
            logger.error(f"Error voting for article: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': 'Failed to record vote'
            }), 500

    except Exception as e:
        logger.error(f"Error in vote endpoint: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': 'An error occurred while processing vote'
        }), 500

@app.route('/rate-credibility', methods=['POST'])
def rate_credibility_endpoint():
    """Rate article credibility"""
    try:
        data = request.get_json()
        article_id = data.get('article_id')
        user_id = data.get('user_id')
        rating = data.get('rating')

        if not all([article_id, user_id, rating]):
            return jsonify({
                'status': 'error',
                'message': 'Missing required parameters'
            }), 400

        if not isinstance(rating, int) or rating < 1 or rating > 5:
            return jsonify({
                'status': 'error',
                'message': 'Rating must be an integer between 1 and 5'
            }), 400

        try:
            conn = pg_db.get_conn()
            with conn.cursor() as cursor:
                # Check if user already rated
                cursor.execute('''
                    SELECT id FROM media_credibility.credibility_votes
                    WHERE article_id = %s AND user_id = %s
                ''', (article_id, user_id))
                existing_rating = cursor.fetchone()

                if existing_rating:
                    # Update existing rating
                    cursor.execute('''
                        UPDATE media_credibility.credibility_votes
                        SET credibility_rating = %s, created_at = CURRENT_TIMESTAMP
                        WHERE id = %s
                    ''', (rating, existing_rating['id']))
                else:
                    # Add new rating
                    cursor.execute('''
                        INSERT INTO media_credibility.credibility_votes (article_id, user_id, credibility_rating)
                        VALUES (%s, %s, %s)
                    ''', (article_id, user_id, rating))

                conn.commit()

            votes = get_article_votes(article_id)
            return jsonify({
                'status': 'success',
                'message': 'Rating recorded successfully',
                'votes': votes
            })
        except Exception as e:
            logger.error(f"Error rating article credibility: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': 'Failed to record rating'
            }), 500

    except Exception as e:
        logger.error(f"Error in rate_credibility endpoint: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': 'An error occurred while processing rating'
        }), 500

@app.route('/source-credibility-chart', methods=['GET'])
def source_credibility_chart():
    """Endpoint for getting source credibility chart data"""
    try:
        conn = pg_db.get_conn()
        with conn.cursor() as cursor:
            cursor.execute('''
                SELECT source, high, medium, low, total_analyzed
                FROM media_credibility.source_stats
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
                if total_current > 0:
                    score = (high * 1.0 + medium * 0.5 + low * 0.0) / total_current
                else:
                    score = 0.5

                sources.append(source)
                credibility_scores.append(round(score, 2))
                high_counts.append(high)
                medium_counts.append(medium)
                low_counts.append(low)
                total_counts.append(total_current)

            return jsonify({
                'status': 'success',
                'data': {
                    'sources': sources,
                    'credibility_scores': credibility_scores,
                    'high_counts': high_counts,
                    'medium_counts': medium_counts,
                    'low_counts': low_counts,
                    'total_counts': total_counts
                }
            })
    except Exception as e:
        logger.error(f"Error in source_credibility_chart endpoint: {str(e)}")
        return jsonify({
            'error': 'An error occurred while fetching chart data',
            'status': 500,
            'details': str(e)
        }), 500
    finally:
        pg_db.release_conn(conn)

@app.route('/analysis-history', methods=['GET'])
def analysis_history():
    """Endpoint for getting analysis history"""
    try:
        conn = pg_db.get_conn()
        with conn.cursor() as cursor:
            cursor.execute('''
                SELECT id, title, source, credibility_level, short_summary,
                       analysis_date::TEXT as formatted_date, url
                FROM media_credibility.news
                ORDER BY analysis_date DESC
                LIMIT 15
            ''')

            rows = cursor.fetchall()
            history = []

            for row in rows:
                history.append({
                    'id': row['id'],
                    'url': row['url'],
                    'title': row['title'],
                    'source': row['source'],
                    'credibility': row['credibility_level'],
                    'summary': row['short_summary'],
                    'date': row['formatted_date']
                })

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
    finally:
        pg_db.release_conn(conn)

@app.route('/analyze', methods=['POST', 'OPTIONS'])
def analyze():
    """Analyze article endpoint with comprehensive error handling"""
    if request.method == 'OPTIONS':
        response = make_response()
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        return response

    try:
        # Validate request content type
        if not request.is_json:
            return jsonify({
                'error': 'Request must be JSON',
                'status': 400,
                'details': 'Content-Type header must be application/json'
            }), 400

        # Get and validate data
        data = request.get_json()
        if not data:
            return jsonify({
                'error': 'Empty request body',
                'status': 400
            }), 400

        if 'input_text' not in data:
            return jsonify({
                'error': 'Missing input text',
                'status': 400,
                'details': 'input_text field is required'
            }), 400

        input_text = data['input_text'].strip()
        source_name = data.get('source_name_manual', 'Direct Input').strip()

        if not input_text:
            return jsonify({
                'error': 'Empty input text',
                'status': 400,
                'details': 'Input text cannot be empty'
            }), 400

        # Process article
        if input_text.startswith(('http://', 'https://')):
            try:
                content, source, title = extract_text_from_url(input_text)
                if not content:
                    return jsonify({
                        'error': 'Could not extract article content',
                        'status': 400,
                        'details': 'Failed to download or parse the article from the provided URL'
                    }), 400
            except Exception as e:
                logger.error(f"Error processing URL: {str(e)}")
                return jsonify({
                    'error': 'Error processing URL',
                    'status': 400,
                    'details': str(e)
                }), 400
        else:
            if len(input_text) < 100:
                return jsonify({
                    'error': 'Content too short',
                    'status': 400,
                    'details': 'Minimum 100 characters required'
                }), 400
            content = input_text
            title = 'User-provided Text'

        # Analyze with Claude
        try:
            analyzer = ClaudeNewsAnalyzer()
            analysis = analyzer.analyze_article_text(content, source)
        except ValueError as e:
            return jsonify({
                'error': str(e),
                'status': 400
            }), 400
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            return jsonify({
                'error': 'Analysis failed',
                'status': 500,
                'details': str(e)
            }), 500

        # Save to database
        try:
            article_id = pg_db.save_article(
                title=title,
                source=source,
                content=content,
                url=input_text if input_text.startswith(('http://', 'https://')) else None
            )
            pg_db.save_analysis(article_id, analysis)
            credibility = analysis.get('credibility_level', 'Medium')
        except Exception as e:
            logger.error(f"Failed to save analysis: {str(e)}")
            return jsonify({
                'error': 'Failed to save analysis',
                'status': 500,
                'details': str(e)
            }), 500

        # Get similar articles
        try:
            same_topic_articles = fetch_same_topic_articles(analysis)
            same_topic_html = render_same_topic_articles_html(same_topic_articles)
        except Exception as e:
            logger.error(f"Failed to fetch similar articles: {str(e)}")
            same_topic_html = '<p>Could not fetch similar articles at this time.</p>'

        # Get source credibility data
        source_credibility_data = get_source_credibility_data()

        # Get analysis history
        analysis_history = get_analysis_history()

        # Prepare response
        response_data = {
            'status': 'success',
            'analysis': analysis,
            'credibility': credibility,
            'title': title,
            'source': source,
            'source_credibility_data': source_credibility_data,
            'analysis_history': analysis_history,
            'same_topic_articles': same_topic_articles,
            'same_topic_html': same_topic_html,
            'output': format_analysis_results(title, source, analysis, credibility)
        }

        return jsonify(response_data)

    except Exception as e:
        logger.error(f"Unexpected error in analyze endpoint: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'status': 500,
            'details': str(e)
        }), 500

def extract_text_from_url(url):
    """Extract text from URL with improved error handling"""
    try:
        # Normalize URL
        parsed = urlparse(url)
        clean_url = urlunparse(parsed._replace(scheme=parsed.scheme.lower(), netloc=parsed.netloc.lower()))

        # Check for video content
        if any(domain in url for domain in ['youtube.com', 'vimeo.com']):
            return "Video content detected", parsed.netloc.replace('www.', ''), "Video: " + url

        # Configure article with timeout and user agent
        article = Article(clean_url, config=config)

        # Download and parse article
        article.download()
        if article.download_state != 2:
            logger.error(f"Failed to download article from {url}")
            return None, None, None

        article.parse()
        if not article.text or len(article.text.strip()) < 100:
            logger.warning(f"Short or empty content from {url}")
            return None, None, None

        # Extract domain and title
        domain = parsed.netloc.replace('www.', '')
        title = article.title.strip() if article.title else "No title"

        return article.text.strip(), domain, title

    except Exception as e:
        logger.error(f"Error extracting article from {url}: {str(e)}")
        return None, None, None

class ClaudeNewsAnalyzer:
    """Class for interacting with Anthropic Claude API"""
    def __init__(self):
        if not ANTHROPIC_API_KEY:
            raise ValueError("Anthropic API key is not configured")

        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        self.model_name = MODEL_NAME

    def analyze_article_text(self, article_text_content, source_name_for_context):
        """Analyze article text using Claude API with improved error handling"""
        try:
            # Validate input
            if not article_text_content or not isinstance(article_text_content, str):
                raise ValueError("Invalid article content")

            # Limit the article length
            max_chars = 10000
            if len(article_text_content) > max_chars:
                article_text_content = article_text_content[:max_chars]
                logger.warning(f"Article content truncated to {max_chars} characters")

            # Create prompt for analysis
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

Article content:
{article_text_content[:5000]}..."""

            # Make API request with error handling
            try:
                response = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=2000,
                    messages=[{"role": "user", "content": prompt}]
                )

                response_text = response.content[0].text.strip()

                # Try to find and parse JSON in the response
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    try:
                        return json.loads(json_match.group(0))
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse JSON: {str(e)}. Response was: {response_text}")
                        raise ValueError("Invalid JSON in response")

                try:
                    return json.loads(response_text)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse response as JSON: {str(e)}. Response was: {response_text}")
                    raise ValueError("Response is not valid JSON")

            except anthropic.APIError as e:
                logger.error(f"Anthropic API error: {str(e)}")
                raise ValueError(f"API Error: {str(e)}")

        except Exception as e:
            logger.error(f"Analysis error: {str(e)}")
            raise ValueError(f"Analysis failed: {str(e)}")

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

def generate_query(analysis_result):
    """Generate query for finding similar articles"""
    topics = analysis_result.get('topics', [])
    key_arguments = analysis_result.get('key_arguments', [])
    mentioned_facts = analysis_result.get('mentioned_facts', [])

    all_terms = []
    for phrase_list in [topics, key_arguments]:
        for phrase in phrase_list:
            if not phrase.strip():
                continue
            if ' ' in phrase.strip() and len(phrase.strip().split()) > 1:
                all_terms.append('"' + phrase.strip() + '"')
            else:
                all_terms.append(phrase.strip())

    for fact in mentioned_facts:
        if not fact.strip():
            continue
        words = [word for word in fact.lower().split() if word not in stop_words_en and len(word) > 2]
        all_terms.extend(words)

    unique_terms = list(set(all_terms))

    if len(unique_terms) >= 3:
        query = ' AND '.join(unique_terms)
    elif unique_terms:
        query = ' OR '.join(unique_terms)
    else:
        query = 'Israel OR Iran OR conflict'

    return query

def make_newsapi_request(params):
    """Make request to NewsAPI"""
    url = 'https://newsapi.org/v2/everything'
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        return response.json().get('articles', [])
    except Exception as e:
        logger.error(f'NewsAPI Request Error: {str(e)}')
        return []

def fetch_same_topic_articles(analysis_result, page=1, per_page=3):
    """Fetch similar articles by topic with pagination"""
    if not NEWS_API_KEY:
        logger.warning('NEWS_API_KEY is not configured. Skipping similar news search.')
        return []

    query = generate_query(analysis_result)
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
        'to': end_date.strftime('%Y-%m-%d'),
    }

    if TRUSTED_NEWS_SOURCES_IDS:
        params['sources'] = ','.join(TRUSTED_NEWS_SOURCES_IDS)

    articles = make_newsapi_request(params)

    if not articles and query != 'Israel OR Middle East OR conflict':
        broader_query = ' OR '.join([f'"{term}"' if ' ' in term else term
                                  for term in analysis_result.get('topics', [])[:3]
                                  if term and term not in stop_words_en])
        if not broader_query:
            broader_query = 'Israel OR Middle East OR conflict'

        params['q'] = broader_query
        additional_articles = make_newsapi_request(params)
        articles.extend(additional_articles)

    unique_articles = {}
    for article in articles:
        if article.get('url'):
            unique_articles[article['url']] = article

    articles = list(unique_articles.values())

    if not articles:
        return []

    all_query_terms = []
    all_query_terms.extend([t.lower().replace('"', '') for t in query.split(' AND ') if t.strip()])
    if 'broader_query' in locals():
        all_query_terms.extend([t.lower().replace('"', '') for t in broader_query.split(' OR ') if t.strip()])
    all_query_terms = list(set([t for t in all_query_terms if t and t not in stop_words_en]))

    ranked_articles = []
    for article in articles:
        source_domain = urlparse(article.get('url', '')).netloc.replace('www.', '')
        trust_score = predefined_trust_scores.get(source_domain, 0.5)

        article_text = (article.get('title', '') + ' ' + article.get('description', '')).lower()
        relevance_score = sum(1 for term in all_query_terms if term in article_text)
        final_score = (relevance_score * 10) + (trust_score * 5)
        ranked_articles.append((article, final_score))

    ranked_articles.sort(key=lambda item: item[1], reverse=True)
    return [item[0] for item in ranked_articles[:per_page]]

def render_same_topic_articles_html(articles):
    """Render HTML for similar articles"""
    if not articles:
        return '<div class="alert alert-info">No similar articles found</div>'

    html_items = []
    for art in articles:
        title = html.escape(art.get('title', 'No Title'))
        article_url = html.escape(art.get('url', '#'))
        source_api_name = html.escape(art.get('source', {}).get('name', 'Unknown Source'))
        published_at = html.escape(art.get('publishedAt', 'N/A').split('T')[0])
        description = html.escape(art.get('description', 'No description available.'))

        domain = urlparse(art.get('url', '#')).netloc.replace('www.', '')
        trust_score = predefined_trust_scores.get(domain, 0.5)
        trust_display = f' (Credibility: {int(trust_score*100)}%)'

        html_items.append(
            f'''
            <div class="similar-article">
                <h4><a href="{article_url}" target="_blank" rel="noopener noreferrer">{title}</a></h4>
                <div class="article-meta">
                    <span class="article-source"><i class="bi bi-newspaper"></i> {source_api_name}</span>
                    <span class="article-date"><i class="bi bi-calendar"></i> {published_at}</span>
                    <span class="article-credibility">Credibility: {int(trust_score*100)}%</span>
                </div>
                <p class="article-description">{description}</p>
            </div>
            '''
        )

    return '<div class="similar-articles-container">' + ''.join(html_items) + '</div>'

def format_analysis_results(title, source, analysis, credibility):
    """Format analysis results for display"""
    try:
        output = {
            'title': title,
            'source': source,
            'credibility': credibility,
            'analysis': analysis,
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
                            <div class="score-name">Integrity</div>
                            <div class="score-value">{analysis.get('news_integrity', 0.0):.2f}</div>
                            <div class="score-description">Overall integrity and trustworthiness</div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="score-item">
                            <div class="score-name">Factuality</div>
                            <div class="score-value">{1 - analysis.get('fact_check_needed_score', 1.0):.2f}</div>
                            <div class="score-description">Likelihood that claims are factual</div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="score-item">
                            <div class="score-name">Sentiment</div>
                            <div class="score-value">{analysis.get('sentiment_score', 0.5):.2f}</div>
                            <div class="score-description">Emotional tone (0.0 negative, 0.5 neutral, 1.0 positive)</div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="score-item">
                            <div class="score-name">Bias</div>
                            <div class="score-value">{1 - analysis.get('bias_score', 1.0):.2f}</div>
                            <div class="score-description">Degree of perceived bias (1.0 low bias, 0.0 high bias)</div>
                        </div>
                    </div>
                </div>
            </div>

            <div class="analysis-section">
                <h2>Additional Information</h2>
                <div class="detail-item">
                    <h4>Author Purpose</h4>
                    <p>{html.escape(analysis.get('author_purpose', 'Not specified'))}</p>
                </div>

                <div class="detail-item">
                    <h4>Short Summary</h4>
                    <p>{html.escape(analysis.get('short_summary', 'No summary available'))}</p>
                </div>

                <div class="detail-item">
                    <h4>Topics</h4>
                    <div class="d-flex flex-wrap gap-2">
                        {' '.join(f'<span class="badge bg-primary">{html.escape(topic)}</span>' for topic in analysis.get('topics', []))}
                    </div>
                </div>

                <div class="detail-item">
                    <h4>Key Arguments</h4>
                    <ul class="list-unstyled">
                        {''.join(f'<li>{html.escape(arg)}</li>' for arg in analysis.get('key_arguments', []))}
                    </ul>
                </div>

                <div class="detail-item">
                    <h4>Potential Biases Identified</h4>
                    <ul class="list-unstyled">
                        {''.join(f'<li>{html.escape(bias)}</li>' for bias in analysis.get('potential_biases_identified', []))}
                    </ul>
                </div>
            </div>
            """
        }
        return output
    except Exception as e:
        logger.error(f"Error formatting analysis results: {str(e)}")
        return {"error": "Error formatting analysis results"}

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
