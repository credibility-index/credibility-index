import os
import logging
import sqlite3
import re
import json
import requests
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse
from flask import Flask, request, jsonify, render_template, session
from werkzeug.middleware.proxy_fix import ProxyFix
import anthropic
from newspaper import Article
from stop_words import get_stop_words

# Initialize Flask application
app = Flask(__name__, static_folder='static', template_folder='templates')
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)
app.secret_key = os.getenv('SECRET_KEY', 'your-secret-key-here')

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

# Database initialization
DB_NAME = 'news_analysis.db'

def get_db_connection():
    """Create database connection"""
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def initialize_database():
    """Initialize database schema"""
    with get_db_connection() as conn:
        conn.execute('''
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
        conn.execute('''
            CREATE TABLE IF NOT EXISTS source_stats (
                source TEXT PRIMARY KEY,
                high INTEGER DEFAULT 0,
                medium INTEGER DEFAULT 0,
                low INTEGER DEFAULT 0,
                total_analyzed INTEGER DEFAULT 0
            )
        ''')
        conn.commit()

# Initialize database
initialize_database()

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
    'the-guardian-uk', 'the-wall-street-journal', 'cnn', 'al-jazeera-english'
]

stop_words_en = get_stop_words('en')

# WordPress scanner protection
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
]

@app.before_request
def block_wordpress_scanners():
    """Block WordPress scanner requests"""
    path = request.path.lower()
    if any(pattern.search(path) for pattern in WORDPRESS_PATHS):
        logger.warning(f'Blocked WordPress scanner request from {request.remote_addr}')
        return jsonify({'error': 'Not found'}), 404

@app.after_request
def add_security_headers(response):
    """Add security and CORS headers"""
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    response.headers['Access-Control-Allow-Credentials'] = 'true'
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    return response

# Error handlers
@app.errorhandler(404)
def page_not_found(e):
    """404 error handler"""
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(400)
def bad_request(e):
    """400 error handler"""
    return jsonify({'error': 'Bad request'}), 400

@app.errorhandler(500)
def internal_server_error(e):
    """500 error handler"""
    return jsonify({'error': 'Internal server error'}), 500

# Claude API class
class ClaudeNewsAnalyzer:
    """Class for interacting with Anthropic Claude API"""
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        self.model_name = MODEL_NAME

    def analyze_article_text(self, article_text_content, source_name_for_context):
        """Analyze article text using Claude API"""
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

Article: {article_text_content[:5000]}..."""

            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = response.content[0].text.strip()
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

        except Exception as e:
            logger.error(f'Claude analysis error: {str(e)}')
            raise ValueError(f'AI analysis error: {str(e)}')

def extract_text_from_url(url):
    """Extract text from URL"""
    try:
        clean_url = re.sub(r'/amp(/)?$', '', url)
        if any(domain in url for domain in ['youtube.com', 'vimeo.com']):
            return "Video content detected", urlparse(clean_url).netloc.replace('www.', ''), "Video: " + url

        article = Article(clean_url)
        article.download()
        article.parse()

        if len(article.text) < 100:
            logger.warning(f"Short content from {url}")
            return None, None, None

        return (article.text.strip(),
                urlparse(clean_url).netloc.replace('www.', ''),
                article.title.strip())
    except Exception as e:
        logger.error(f'Error extracting article from {url}: {str(e)}')
        return None, None, None

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
            integrity = analysis.get('news_integrity', 0.0)
            fact_check = analysis.get('fact_check_needed_score', 1.0)
            sentiment = analysis.get('sentiment_score', 0.5)
            bias = analysis.get('bias_score', 1.0)
            summary = analysis.get('short_summary', 'No summary')
            credibility = analysis.get('index_of_credibility', 0.0)

            level = calculate_credibility(integrity, fact_check, sentiment, bias)
            db_url = url if url else f'text_{datetime.now(timezone.utc).timestamp()}'

            conn.execute('''
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

            conn.commit()
            return level
    except Exception as e:
        logger.error(f'Database error: {str(e)}')
        raise

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
        query = 'current events OR news'

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

    if not articles and query != 'current events OR news':
        broader_query = ' OR '.join([f'"{term}"' if ' ' in term else term
                                  for term in analysis_result.get('topics', [])[:3]
                                  if term and term not in stop_words_en])
        if not broader_query:
            broader_query = 'current events OR news'

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
        return '<p>No similar articles found.</p>'

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
            f'<div class="same-topic-article">'
            f'<h4><a href="{article_url}" target="_blank">{title}</a></h4>'
            f'<p><strong>Source:</strong> {source_api_name}{trust_display} | '
            f'<strong>Published:</strong> {published_at}</p>'
            f'<p>{description}</p>'
            f'</div>'
        )

    return '<div class="same-topic-articles-container">' + ''.join(html_items) + '</div>'

@app.route('/')
def home():
    """Home page"""
    return render_template('index.html')

@app.route('/analyze', methods=['POST', 'OPTIONS'])
def analyze():
    """Analyze article endpoint"""
    if request.method == 'OPTIONS':
        return jsonify({}), 200

    try:
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400

        data = request.get_json()
        if not data or 'input_text' not in data:
            return jsonify({'error': 'Missing input text'}), 400

        input_text = data['input_text']
        source_name = data.get('source_name_manual', 'Direct Input')

        if not input_text:
            return jsonify({'error': 'Empty input text'}), 400

        # Process article
        if input_text.startswith('http'):
            content, source, title = extract_text_from_url(input_text)
            if not content:
                return jsonify({'error': 'Could not extract article content'}), 400
        else:
            content = input_text
            title = 'User-provided Text'
            source = source_name

        if len(content) < 100:
            return jsonify({'error': 'Content too short (min 100 chars)'}), 400

        # Analyze with Claude
        analyzer = ClaudeNewsAnalyzer()
        analysis = analyzer.analyze_article_text(content, source)

        # Save to database
        credibility = save_analysis(
            input_text if input_text.startswith('http') else None,
            title,
            source,
            content,
            analysis
        )

        # Store analysis result in session
        session['last_analysis_result'] = analysis

        # Get similar articles
        same_topic_articles = fetch_same_topic_articles(analysis)
        same_topic_html = render_same_topic_articles_html(same_topic_articles)

        # Prepare response
        response = {
            'status': 'success',
            'analysis': analysis,
            'credibility': credibility,
            'title': title,
            'source': source,
            'same_topic_articles': same_topic_html
        }

        return jsonify(response), 200

    except ValueError as e:
        logger.error(f'Analysis error: {str(e)}')
        return jsonify({'status': 'error', 'message': str(e)}), 400
    except Exception as e:
        logger.error(f'Unexpected error: {str(e)}')
        return jsonify({'status': 'error', 'message': 'Internal server error'}), 500

@app.route('/same_topic_articles', methods=['GET'])
def get_same_topic_articles():
    """Endpoint for getting more articles on the same topic"""
    try:
        page = int(request.args.get('page', 1))
        per_page = 3

        analysis_result = session.get('last_analysis_result')
        if not analysis_result:
            return jsonify({'error': 'No analysis result found. Please analyze an article first.'}), 400

        same_topic_articles = fetch_same_topic_articles(analysis_result, page=page, per_page=per_page)
        same_topic_html = render_same_topic_articles_html(same_topic_articles)

        return jsonify({
            'same_topic_html': same_topic_html
        })

    except Exception as e:
        logger.error(f"Error in get_same_topic_articles endpoint: {str(e)}")
        return jsonify({'error': 'An error occurred while fetching same topic articles'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))

