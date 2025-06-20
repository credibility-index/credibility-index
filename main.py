import os
import logging
import re
import json
import requests
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse, urlunparse
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.middleware.proxy_fix import ProxyFix
from flask_cors import CORS
import anthropic
from newspaper import Article, Config
from stop_words import get_stop_words
from pg_database import PostgresDB

# Initialize Flask application
app = Flask(__name__, static_folder='static', template_folder='templates')
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)
app.secret_key = os.getenv('SECRET_KEY', 'your-secret-key-here')

# Configure CORS
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

# Initialize database
pg_db = PostgresDB()
@app.route('/test-db')
def test_db():
    """Test database connection"""
    try:
        conn = pg_db.get_conn()
        with conn.cursor() as cursor:
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
        pg_db.release_conn(conn)
        return jsonify({
            'status': 'success',
            'message': 'Database connection successful',
            'result': result
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Database connection failed: {str(e)}'
        }), 500

# Configure newspaper library
user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
config = Config()
config.browser_user_agent = user_agent
config.request_timeout = 30

# Static files route
@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(app.static_folder, filename)

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico')

@app.route('/')
def home():
    """Home page with analysis"""
    try:
        # Get analysis data
        buzz_analysis = get_buzz_analysis()
        analyzed_articles = get_analyzed_articles()
        source_credibility_data = get_source_credibility_data()

        return render_template('index.html',
                             buzz_analysis=buzz_analysis,
                             analyzed_articles=analyzed_articles,
                             source_credibility_data=source_credibility_data)
    except Exception as e:
        logger.error(f"Error loading home page: {str(e)}")
        return render_template('error.html', message="Failed to load home page")

def get_buzz_analysis():
    """Get current analysis data"""
    try:
        # Try to get from database
        conn = pg_db.get_conn()
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT analysis_data
                FROM media_credibility.analysis
                WHERE analysis_type = 'comprehensive'
                ORDER BY created_at DESC
                LIMIT 1
            """)
            result = cursor.fetchone()

            if result:
                return result['analysis_data']

            # Return default analysis
            return {
                "western_perspective": {"summary": "Default western perspective"},
                "iranian_perspective": {"summary": "Default iranian perspective"},
                "israeli_perspective": {"summary": "Default israeli perspective"},
                "neutral_perspective": {"summary": "Default neutral perspective"},
                "historical_context": {"summary": "Default historical context"},
                "balanced_summary": "Default balanced summary",
                "common_points": [],
                "disagreements": [],
                "potential_solutions": [],
                "credibility_assessment": "Medium"
            }
    except Exception as e:
        logger.error(f"Error getting buzz analysis: {str(e)}")
        return {
            "western_perspective": {"summary": "Error loading western perspective"},
            "iranian_perspective": {"summary": "Error loading iranian perspective"},
            "israeli_perspective": {"summary": "Error loading israeli perspective"},
            "neutral_perspective": {"summary": "Error loading neutral perspective"},
            "historical_context": {"summary": "Error loading historical context"},
            "balanced_summary": "Error loading balanced summary",
            "common_points": [],
            "disagreements": [],
            "potential_solutions": [],
            "credibility_assessment": "Error"
        }
    finally:
        if 'conn' in locals():
            pg_db.release_conn(conn)

def get_analyzed_articles(limit=5):
    """Get analyzed articles"""
    try:
        conn = pg_db.get_conn()
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT id, title, source, short_summary, credibility_level, url
                FROM media_credibility.news
                ORDER BY analysis_date DESC
                LIMIT %s
            """, (limit,))
            return cursor.fetchall()
    except Exception as e:
        logger.error(f"Error getting analyzed articles: {str(e)}")
        return []
    finally:
        if 'conn' in locals():
            pg_db.release_conn(conn)

def get_source_credibility_data():
    """Get source credibility data"""
    try:
        conn = pg_db.get_conn()
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT source, high, medium, low
                FROM media_credibility.source_stats
                ORDER BY (high + medium + low) DESC
            """)

            data = cursor.fetchall()
            sources = []
            scores = []

            for source, high, medium, low in data:
                total = high + medium + low
                score = (high * 1.0 + medium * 0.5) / total if total > 0 else 0.5
                sources.append(source)
                scores.append(round(score, 2))

            return {
                'sources': sources,
                'scores': scores
            }
    except Exception as e:
        logger.error(f"Error getting source credibility data: {str(e)}")
        return {
            'sources': [],
            'scores': []
        }
    finally:
        if 'conn' in locals():
            pg_db.release_conn(conn)

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze article endpoint"""
    try:
        data = request.get_json()
        input_text = data.get('input_text', '').strip()

        if not input_text:
            return jsonify({
                'status': 'error',
                'message': 'Input text is required'
            }), 400

        # Extract article content
        if input_text.startswith(('http://', 'https://')):
            content, source, title = extract_text_from_url(input_text)
            if not content:
                return jsonify({
                    'status': 'error',
                    'message': 'Could not extract article content'
                }), 400
        else:
            content = input_text
            source = 'Direct Input'
            title = 'User-provided Text'

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

        return jsonify({
            'status': 'success',
            'analysis': analysis,
            'title': title,
            'source': source
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
            return None, parsed.netloc.replace('www.', ''), "Video content detected"

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
                "news_integrity": 0.7,
                "fact_check_needed_score": 0.3,
                "sentiment_score": 0.5,
                "bias_score": 0.4,
                "topics": ["default"],
                "key_arguments": ["default"],
                "mentioned_facts": ["default"],
                "author_purpose": "inform",
                "potential_biases_identified": ["none"],
                "short_summary": "Default summary",
                "index_of_credibility": 0.6
            }

    except Exception as e:
        logger.error(f"Error analyzing with Claude: {str(e)}")
        return {
            "news_integrity": 0.7,
            "fact_check_needed_score": 0.3,
            "sentiment_score": 0.5,
            "bias_score": 0.4,
            "topics": ["default"],
            "key_arguments": ["default"],
            "mentioned_facts": ["default"],
            "author_purpose": "inform",
            "potential_biases_identified": ["none"],
            "short_summary": "Default summary",
            "index_of_credibility": 0.6
        }

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

