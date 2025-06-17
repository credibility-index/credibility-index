import os
import logging
import re
import json
import requests
import html
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse, urlunparse
from flask import Flask, request, jsonify, render_template, make_response
from werkzeug.middleware.proxy_fix import ProxyFix
from flask_cors import CORS
import anthropic
from newspaper import Article, Config
from stop_words import get_stop_words
from pg_database import PostgresDB

# Initialize Flask application
app = Flask(__name__, static_folder='static', template_folder='templates')
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)
app.secret_key = os.getenv('SECRET_KEY')

# Configure CORS for Railway
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Initialize API clients
anthropic_client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO').upper(), logging.INFO),
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

predefined_trust_scores = {
    'bbc.com': 0.9, 'bbc.co.uk': 0.9, 'reuters.com': 0.95, 'apnews.com': 0.93,
    'nytimes.com': 0.88, 'theguardian.com': 0.85, 'wsj.com': 0.82,
    'cnn.com': 0.70, 'foxnews.com': 0.40, 'aljazeera.com': 0.80
}

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

@app.route('/')
def home():
    """Home page with Buzz Feed analysis"""
    try:
        # Get buzz analysis
        buzz_analysis = get_buzz_analysis()

        # Get analyzed articles
        analyzed_articles = get_analyzed_articles()

        # Get source credibility data for chart
        source_credibility_data = get_source_credibility_data()

        return render_template('index.html',
                             buzz_analysis=buzz_analysis,
                             analyzed_articles=analyzed_articles,
                             source_credibility_data=source_credibility_data)

    except Exception as e:
        logger.error(f"Error loading home page: {str(e)}")
        return render_template('error.html', message="Failed to load home page")

def get_buzz_analysis():
    """Get the current buzz analysis from database or create default"""
    try:
        conn = pg_db.get_conn()
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT an.analysis_data
                FROM media_credibility.analysis an
                JOIN media_credibility.news a ON an.article_id = a.id
                WHERE an.analysis_type = 'comprehensive'
                ORDER BY an.created_at DESC
                LIMIT 1
            """)
            analysis = cursor.fetchone()

            if analysis:
                return analysis['analysis_data']

            # Create default analysis
            default_analysis = {
                "western_perspective": {
                    "summary": "Western perspective on the situation",
                    "details": ""
                },
                "iranian_perspective": {
                    "summary": "Iranian perspective on the situation",
                    "details": ""
                },
                "israeli_perspective": {
                    "summary": "Israeli perspective on the situation",
                    "details": ""
                },
                "neutral_perspective": {
                    "summary": "Neutral perspective on the situation",
                    "details": ""
                },
                "historical_context": {
                    "summary": "Historical background of the situation",
                    "details": ""
                },
                "balanced_summary": "Balanced summary of the situation",
                "common_points": [],
                "disagreements": [],
                "potential_solutions": [],
                "credibility_assessment": "Medium credibility"
            }

            cursor.execute("""
                INSERT INTO media_credibility.news
                (title, source, content, short_summary, is_buzz_article)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
            """, (
                "Comprehensive Analysis: Current Situation",
                "Media Credibility Index",
                "Default comprehensive analysis content",
                "Default comprehensive analysis summary",
                True
            ))
            article_id = cursor.fetchone()['id']

            cursor.execute("""
                INSERT INTO media_credibility.analysis
                (article_id, analysis_data, analysis_type)
                VALUES (%s, %s, %s)
            """, (
                article_id,
                json.dumps(default_analysis),
                'comprehensive'
            ))

            conn.commit()
            return default_analysis

    except Exception as e:
        logger.error(f"Error getting buzz analysis: {str(e)}")
        return {
            "western_perspective": {"summary": "Western perspective on the situation"},
            "iranian_perspective": {"summary": "Iranian perspective on the situation"},
            "israeli_perspective": {"summary": "Israeli perspective on the situation"},
            "neutral_perspective": {"summary": "Neutral perspective on the situation"},
            "historical_context": {"summary": "Historical background of the situation"},
            "balanced_summary": "Balanced summary of the situation",
            "common_points": [],
            "disagreements": [],
            "potential_solutions": [],
            "credibility_assessment": "Medium credibility"
        }
    finally:
        pg_db.release_conn(conn)

def get_analyzed_articles(limit=5):
    """Get recently analyzed articles"""
    try:
        conn = pg_db.get_conn()
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT id, title, source, short_summary, credibility_level, url
                FROM media_credibility.news
                WHERE content IS NOT NULL
                ORDER BY analysis_date DESC
                LIMIT %s
            """, (limit,))
            articles = cursor.fetchall()
            return articles
    except Exception as e:
        logger.error(f"Error getting analyzed articles: {str(e)}")
        return []
    finally:
        pg_db.release_conn(conn)

def get_source_credibility_data():
    """Get source credibility data for chart"""
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

            for source, high, medium, low, total in data:
                total_current = high + medium + low
                score = (high * 1.0 + medium * 0.5 + low * 0.0) / total_current if total_current > 0 else 0.5

                sources.append(source)
                credibility_scores.append(round(score, 2))

            return {
                'sources': sources,
                'credibility_scores': credibility_scores
            }
    except Exception as e:
        logger.error(f"Error getting source credibility data: {str(e)}")
        return {
            'sources': [],
            'credibility_scores': []
        }
    finally:
        pg_db.release_conn(conn)

@app.route('/analyze', methods=['POST'])
def analyze_article():
    """Analyze article endpoint"""
    try:
        data = request.get_json()
        input_text = data.get('input_text', '').strip()
        source_name = data.get('source_name', 'Direct Input').strip()

        if not input_text:
            return jsonify({
                'status': 'error',
                'message': 'Input text is required'
            }), 400

        # Process article
        if input_text.startswith(('http://', 'https://')):
            content, source, title = extract_text_from_url(input_text)
            if not content:
                return jsonify({
                    'status': 'error',
                    'message': 'Could not extract article content'
                }), 400
        else:
            if len(input_text) < 100:
                return jsonify({
                    'status': 'error',
                    'message': 'Content too short, minimum 100 characters required'
                }), 400
            content = input_text
            title = 'User-provided Text'
            source = source_name

        # Analyze with Claude
        analysis = analyze_with_claude(content, source)

        # Save to database
        article_id = pg_db.save_article(
            title=title,
            source=source,
            content=content,
            url=input_text if input_text.startswith(('http://', 'https://')) else None
        )
        pg_db.save_analysis(article_id, analysis)

        # Get related articles
        related_articles = fetch_related_articles(analysis)

        return jsonify({
            'status': 'success',
            'analysis': analysis,
            'title': title,
            'source': source,
            'related_articles': related_articles
        })

    except Exception as e:
        logger.error(f"Error analyzing article: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

def extract_text_from_url(url):
    """Extract text from URL"""
    try:
        parsed = urlparse(url)
        clean_url = urlunparse(parsed._replace(scheme=parsed.scheme.lower(), netloc=parsed.netloc.lower()))

        if any(domain in url for domain in ['youtube.com', 'vimeo.com']):
            return "Video content detected", parsed.netloc.replace('www.', ''), "Video: " + url

        article = Article(clean_url, config=config)
        article.download()
        article.parse()

        if not article.text or len(article.text.strip()) < 100:
            return None, None, None

        domain = parsed.netloc.replace('www.', '')
        title = article.title.strip() if article.title else "No title"

        return article.text.strip(), domain, title

    except Exception as e:
        logger.error(f"Error extracting article from {url}: {str(e)}")
        return None, None, None

def analyze_with_claude(content, source):
    """Analyze content with Claude API"""
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

Article content:
{content[:5000]}..."""

        response = anthropic_client.messages.create(
            model=os.getenv('ANTHROPIC_MODEL', 'claude-3-opus-20240229'),
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )

        response_text = response.content[0].text.strip()

        # Try to parse JSON response
        try:
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            return json.loads(response_text)
        except json.JSONDecodeError:
            return {
                "news_integrity": 0.75,
                "fact_check_needed_score": 0.25,
                "sentiment_score": 0.5,
                "bias_score": 0.5,
                "topics": ["default topic"],
                "key_arguments": ["default argument"],
                "mentioned_facts": ["default fact"],
                "author_purpose": "Default purpose",
                "potential_biases_identified": ["default bias"],
                "short_summary": "Default summary",
                "index_of_credibility": 0.65
            }

    except Exception as e:
        logger.error(f"Error analyzing with Claude: {str(e)}")
        return {
            "news_integrity": 0.75,
            "fact_check_needed_score": 0.25,
            "sentiment_score": 0.5,
            "bias_score": 0.5,
            "topics": ["default topic"],
            "key_arguments": ["default argument"],
            "mentioned_facts": ["default fact"],
            "author_purpose": "Default purpose",
            "potential_biases_identified": ["default bias"],
            "short_summary": "Default summary",
            "index_of_credibility": 0.65
        }

def fetch_related_articles(analysis):
    """Fetch related articles based on analysis"""
    try:
        # Generate search query from analysis
        topics = analysis.get('topics', [])
        key_arguments = analysis.get('key_arguments', [])
        mentioned_facts = analysis.get('mentioned_facts', [])

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
            words = [word for word in fact.lower().split() if len(word) > 2]
            all_terms.extend(words)

        unique_terms = list(set(all_terms))

        if len(unique_terms) >= 3:
            query = ' AND '.join(unique_terms)
        elif unique_terms:
            query = ' OR '.join(unique_terms)
        else:
            query = 'Israel OR Iran OR conflict'

        # First try to get articles from our database
        conn = pg_db.get_conn()
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT id, title, source, short_summary, credibility_level, url
                FROM media_credibility.news
                WHERE content @@ to_tsquery(%s)
                ORDER BY analysis_date DESC
                LIMIT 5
            """, (query,))
            db_articles = cursor.fetchall()

        # If we don't have enough from database, use NewsAPI
        if len(db_articles) < 5 and os.getenv('NEWS_API_KEY'):
            params = {
                'q': query,
                'apiKey': os.getenv('NEWS_API_KEY'),
                'language': 'en',
                'pageSize': 5 - len(db_articles),
                'sortBy': 'relevancy'
            }

            try:
                response = requests.get(
                    os.getenv('NEWS_ENDPOINT', 'https://newsapi.org/v2/everything'),
                    params=params,
                    timeout=15
                )
                response.raise_for_status()
                api_articles = response.json().get('articles', [])

                # Combine and deduplicate articles
                all_articles = list(db_articles)
                unique_urls = set(art['url'] for art in db_articles)

                for article in api_articles:
                    if article.get('url') not in unique_urls:
                        unique_urls.add(article['url'])
                        all_articles.append({
                            'title': article.get('title', 'No title'),
                            'source': article.get('source', {}).get('name', 'Unknown'),
                            'short_summary': article.get('description', 'No description'),
                            'credibility_level': 'Medium',
                            'url': article.get('url', '#')
                        })

                return all_articles

            except Exception as e:
                logger.error(f"Error fetching related articles from NewsAPI: {str(e)}")
                return db_articles if db_articles else []

        return db_articles if db_articles else []

    except Exception as e:
        logger.error(f"Error fetching related articles: {str(e)}")
        return []

@app.route('/update-buzz-analysis', methods=['POST'])
def update_buzz_analysis():
    """Endpoint to manually update the Buzz Feed analysis"""
    try:
        # Check authorization
        if request.headers.get('Authorization') != os.getenv('UPDATE_AUTH_TOKEN'):
            return jsonify({
                'status': 'error',
                'message': 'Unauthorized'
            }), 403

        data = request.get_json()
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No data provided'
            }), 400

        # Validate the analysis data structure
        required_fields = [
            'western_perspective', 'iranian_perspective', 'israeli_perspective',
            'neutral_perspective', 'historical_context', 'balanced_summary',
            'common_points', 'disagreements', 'potential_solutions',
            'credibility_assessment'
        ]

        for field in required_fields:
            if field not in data:
                return jsonify({
                    'status': 'error',
                    'message': f'Missing required field: {field}'
                }), 400

        try:
            conn = pg_db.get_conn()
            with conn.cursor() as cursor:
                # Get existing buzz analysis article
                cursor.execute("""
                    SELECT a.id
                    FROM media_credibility.news a
                    JOIN media_credibility.analysis an ON a.id = an.article_id
                    WHERE an.analysis_type = 'comprehensive'
                    ORDER BY an.created_at DESC
                    LIMIT 1
                """)
                existing = cursor.fetchone()

                if existing:
                    article_id = existing['id']
                else:
                    # Create new article if none exists
                    cursor.execute("""
                        INSERT INTO media_credibility.news
                        (title, source, content, short_summary, is_buzz_article)
                        VALUES (%s, %s, %s, %s, %s)
                        RETURNING id
                    """, (
                        "Comprehensive Analysis: Current Situation",
                        "Media Credibility Index",
                        "Comprehensive analysis content",
                        "Comprehensive analysis summary",
                        True
                    ))
                    article_id = cursor.fetchone()['id']

                # Update the analysis
                cursor.execute("""
                    INSERT INTO media_credibility.analysis
                    (article_id, analysis_data, analysis_type)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (article_id, analysis_type)
                    DO UPDATE SET analysis_data = EXCLUDED.analysis_data
                """, (
                    article_id,
                    json.dumps(data),
                    'comprehensive'
                ))

                conn.commit()
                return jsonify({
                    'status': 'success',
                    'message': 'Buzz Feed analysis updated successfully',
                    'article_id': article_id
                })

        except Exception as e:
            logger.error(f"Database error updating buzz analysis: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': 'Database error occurred while updating analysis'
            }), 500

    except Exception as e:
        logger.error(f"Error in update buzz analysis: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
