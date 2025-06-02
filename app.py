from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import json
import sqlite3
import requests
import anthropic
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from datetime import datetime, timedelta, UTC
import logging
import re
import plotly.graph_objects as go
from stop_words import get_stop_words
from newspaper import Article
import html

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Загрузка переменных окружения
load_dotenv()
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
NEWS_API_KEY = os.getenv('NEWS_API_KEY')
NEWS_API_ENABLED = bool(NEWS_API_KEY)
MODEL_NAME = "claude-3-opus-20240229"

# Проверка наличия API ключей
if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY is missing! Please set it in your .env file.")
if not NEWS_API_KEY:
    logger.warning("NEWS_API_KEY is missing! Similar news functionality will be disabled.")
    NEWS_API_ENABLED = False

# Настройка базы данных
DB_NAME = 'news_analysis.db'

# Начальные данные для надежности источников
INITIAL_SOURCE_COUNTS = {
    "bbc.com": {"high": 15, "medium": 5, "low": 1},
    "reuters.com": {"high": 20, "medium": 3, "low": 0},
    "foxnews.com": {"high": 3, "medium": 7, "low": 15},
    "cnn.com": {"high": 5, "medium": 10, "low": 5},
    "nytimes.com": {"high": 10, "medium": 5, "low": 2},
    "theguardian.com": {"high": 12, "medium": 4, "low": 1},
    "apnews.com": {"high": 18, "medium": 2, "low": 0}
}

# Соответствие доменов и владельцев СМИ
media_owners = {
    "bbc.com": "BBC",
    "bbc.co.uk": "BBC",
    "reuters.com": "Thomson Reuters",
    "foxnews.com": "Fox Corporation",
    "cnn.com": "Warner Bros. Discovery",
    "nytimes.com": "The New York Times Company",
    "theguardian.com": "Guardian Media Group",
    "apnews.com": "Associated Press",
    "wsj.com": "News Corp",
    "aljazeera.com": "Al Jazeera Media Network"
}

# ID источников NewsAPI для фильтрации
TRUSTED_NEWS_SOURCES_IDS = [
    "bbc-news", "reuters", "associated-press", "the-new-york-times",
    "the-guardian-uk", "the-wall-street-journal", "cnn", "al-jazeera-english"
]

stop_words_en = get_stop_words('en')

# Инициализация Flask приложения
app = Flask(__name__, static_folder='static', template_folder='templates')

# Настройка статических файлов
@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(app.static_folder, filename)

def check_database_integrity():
    """Проверяет целостность базы данных."""
    try:
        logger.info("Performing database integrity check...")

        if not os.path.exists(DB_NAME):
            logger.error(f"Database file {DB_NAME} not found!")
            return False

        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()

        # Проверка существования таблиц
        c.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [table[0] for table in c.fetchall()]

        required_tables = ['news', 'source_stats']
        for table in required_tables:
            if table not in tables:
                logger.error(f"Critical table '{table}' is missing!")
                return False

        # Проверка структуры таблицы news
        c.execute("PRAGMA table_info(news)")
        news_columns = [column[1] for column in c.fetchall()]
        required_news_columns = [
            'id', 'title', 'source', 'content', 'integrity',
            'fact_check', 'sentiment', 'bias', 'credibility_level',
            'index_of_credibility', 'url', 'analysis_date', 'short_summary'
        ]

        for column in required_news_columns:
            if column not in news_columns:
                logger.error(f"Critical column '{column}' is missing in 'news' table!")
                return False

        # Проверка структуры таблицы source_stats
        c.execute("PRAGMA table_info(source_stats)")
        stats_columns = [column[1] for column in c.fetchall()]
        required_stats_columns = ['source', 'high', 'medium', 'low', 'total_analyzed']

        for column in required_stats_columns:
            if column not in stats_columns:
                logger.error(f"Critical column '{column}' is missing in 'source_stats' table!")
                return False

        # Проверка наличия данных в таблице source_stats
        c.execute("SELECT COUNT(*) FROM source_stats")
        source_count = c.fetchone()[0]
        if source_count == 0:
            logger.warning("source_stats table is empty! Initializing with default data.")
            initialize_sources(INITIAL_SOURCE_COUNTS)

        logger.info("Database integrity check completed successfully")
        return True

    except sqlite3.Error as e:
        logger.error(f"Database error during integrity check: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during database check: {e}")
        return False
    finally:
        if 'conn' in locals() and conn:
            conn.close()

def ensure_db_schema():
    """Обеспечивает наличие схемы базы данных."""
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()

        # Создание таблицы news
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

        # Создание таблицы source_stats
        c.execute('''CREATE TABLE IF NOT EXISTS source_stats (
            source TEXT PRIMARY KEY,
            high INTEGER DEFAULT 0,
            medium INTEGER DEFAULT 0,
            low INTEGER DEFAULT 0,
            total_analyzed INTEGER DEFAULT 0
        )''')

        conn.commit()
        logger.info("Database schema ensured successfully")

    except sqlite3.Error as e:
        logger.error(f"Error ensuring database schema: {e}")
        raise
    finally:
        if conn:
            conn.close()

def initialize_sources(initial_counts):
    """Инициализирует статистику источников начальными значениями."""
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        logger.info("Initializing sources...")

        for source, counts in initial_counts.items():
            c.execute("SELECT total_analyzed FROM source_stats WHERE source = ?", (source,))
            row = c.fetchone()

            if row is None:
                high = counts.get("high", 0)
                medium = counts.get("medium", 0)
                low = counts.get("low", 0)
                total_analyzed = high + medium + low

                logger.info(f"Adding source: {source} with initial counts (H:{high}, M:{medium}, L:{low}, Total:{total_analyzed})")

                try:
                    c.execute('''INSERT INTO source_stats (source, high, medium, low, total_analyzed)
                             VALUES (?, ?, ?, ?, ?)''',
                             (source, high, medium, low, total_analyzed))
                    conn.commit()
                except sqlite3.Error as e:
                    logger.error(f"Error initializing source {source}: {e}")
                    conn.rollback()

        logger.info("Initial source initialization completed successfully")

    except sqlite3.Error as e:
        logger.error(f"Database error during source initialization: {e}")
        raise
    finally:
        if conn:
            conn.close()

class ClaudeNewsAnalyzer:
    """Класс для анализа новостных статей с использованием API Claude."""
    def __init__(self, api_key, model_name):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model_name = model_name

    def analyze_article_text(self, article_text_content, source_name_for_context):
        """Анализирует текст статьи с использованием API Claude."""
        max_chars_for_claude = 10000
        if len(article_text_content) > max_chars_for_claude:
            logger.warning(f"Article text truncated to {max_chars_for_claude} characters for Claude analysis.")
            article_text_content = article_text_content[:max_chars_for_claude]

        media_owner_display = media_owners.get(source_name_for_context, "Unknown Owner")

        prompt = f"""You are a highly analytical and neutral AI assistant specializing in news article reliability and content analysis. Your task is to dissect the provided news article.

Article Text:
\"\"\"
{article_text_content}
\"\"\"

Source (for context, if known): {source_name_for_context}
Media Owner: {media_owner_display}

Please perform the following analyses and return the results ONLY in a single, valid JSON object format. Do not include any explanatory text before or after the JSON object.

JSON Fields:
- "news_integrity": (Float, 0.0-1.0) Assess the overall integrity and trustworthiness of the information presented. Higher means more trustworthy.
- "fact_check_needed_score": (Float, 0.0-1.0) Likelihood that the article's claims require external fact-checking. 1.0 means high likelihood.
- "sentiment_score": (Float, 0.0-1.0) Overall emotional tone (0.0 negative, 0.5 neutral, 1.0 positive).
- "bias_score": (Float, 0.0-1.0) Degree of perceived bias (0.0 low bias, 1.0 high bias).
- "topics": (List of strings) Identify 3-5 main topics or keywords that accurately represent the core subject matter.
- "key_arguments": (List of strings) Extract the main arguments or claims made by the author.
- "mentioned_facts": (List of strings) List any specific facts, data, or statistics mentioned.
- "author_purpose": (String) Briefly determine the author's likely primary purpose.
- "potential_biases_identified": (List of strings) Enumerate any specific signs of potential bias or subjectivity observed.
- "short_summary": (String) A concise summary of the article's main content in 2-4 sentences.
- "index_of_credibility": (Float, 0.0-1.0) Calculate an overall index of credibility based on the above factors.
- "published_date": (String, YYYY-MM-DD or N/A) The publication date of the article.
"""

        try:
            message = self.client.messages.create(
                model=self.model_name,
                max_tokens=2000,
                temperature=0.2,
                system="You are a JSON-generating expert. Always provide valid JSON.",
                messages=[{"role": "user", "content": prompt}]
            )

            raw_json_text = message.content[0].text.strip()
            match = re.search(r'```json\s*(\{.*\})\s*```', raw_json_text, re.DOTALL)

            if match:
                json_str = match.group(1)
            else:
                json_str = raw_json_text

            return json.loads(json_str)

        except anthropic.APIError as e:
            logger.error(f"Anthropic API Error: {e}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"JSON Decode Error from Claude's response: {e}. Raw response was: {raw_json_text}")
            raise ValueError(f"Failed to parse AI response: {e}. Raw: {raw_json_text[:500]}...")
        except Exception as e:
            logger.error(f"Unexpected error during Claude analysis: {e}")
            raise

def extract_text_from_url(url):
    """Извлекает текст статьи из URL с использованием newspaper3k."""
    try:
        clean_url = re.sub(r'/amp(/)?$', '', url)
        article = Article(clean_url)
        article.download()
        article.parse()

        text = article.text.strip()
        title = article.title.strip() if article.title else ""
        source = urlparse(clean_url).netloc.replace("www.", "")

        if not text:
            logger.warning(f"Newspaper3k extracted empty text from {clean_url}")
            return "", "", ""

        return text, source, title

    except Exception as e:
        logger.error(f"Error extracting article from URL {url}: {e}")
        return "", "", ""

def calculate_credibility_level(integrity, fact_check_needed, sentiment, bias):
    """Вычисляет уровень достоверности на основе оценок ИИ."""
    fact_check_score = 1.0 - fact_check_needed
    neutral_sentiment_proximity = 1.0 - abs(sentiment - 0.5) * 2
    bias_score_inverted = 1.0 - bias

    avg = (integrity * 0.45) + (fact_check_score * 0.35) + (neutral_sentiment_proximity * 0.10) + (bias_score_inverted * 0.10)

    if avg >= 0.75:
        return 'High'
    if avg >= 0.5:
        return 'Medium'
    return 'Low'

def save_analysis_to_db(url, title, source, content, analysis_result):
    """Сохраняет результаты анализа в базу данных SQLite."""
    conn = None
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()

        integrity = analysis_result.get('news_integrity', 0.0)
        fact_check_needed = analysis_result.get('fact_check_needed_score', 1.0)
        sentiment = analysis_result.get('sentiment_score', 0.5)
        bias = analysis_result.get('bias_score', 1.0)
        short_summary = analysis_result.get('short_summary', 'Summary not available.')
        index_of_credibility = analysis_result.get('index_of_credibility', 0.0)

        credibility_level = calculate_credibility_level(integrity, fact_check_needed, sentiment, bias)
        db_url = url if url else f"no_url_{datetime.now(UTC).timestamp()}"

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

        # Обновление таблицы source_stats
        c.execute("SELECT high, medium, low, total_analyzed FROM source_stats WHERE source = ?", (source,))
        row = c.fetchone()

        if row:
            high, medium, low, total = row
            if credibility_level == 'High':
                high += 1
            elif credibility_level == 'Medium':
                medium += 1
            else:
                low += 1
            total += 1
            c.execute('''UPDATE source_stats SET high=?, medium=?, low=?, total_analyzed=? WHERE source=?''',
                    (high, medium, low, total, source))
        else:
            high = 1 if credibility_level == 'High' else 0
            medium = 1 if credibility_level == 'Medium' else 0
            low = 1 if credibility_level == 'Low' else 0
            c.execute('''INSERT INTO source_stats (source, high, medium, low, total_analyzed)
                        VALUES (?, ?, ?, ?, ?)''', (source, high, medium, low, 1))

        conn.commit()
        logger.info(f"Analysis for '{title}' ({source}) saved to database.")
        return credibility_level

    except sqlite3.Error as e:
        logger.error(f"Database error in save_analysis_to_db: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()

def process_article_analysis(input_text, source_name_manual):
    """Обрабатывает анализ статьи."""
    article_url = None
    article_content = input_text
    article_title = "User-provided Text"
    source_name = source_name_manual if source_name_manual else "Direct Input"

    if input_text.strip().startswith("http"):
        article_url = input_text.strip()
        logger.info(f"Input is a URL: {article_url}")
        content_from_url, source_from_url, title_from_url = extract_text_from_url(article_url)

        if content_from_url and len(content_from_url) >= 100:
            article_content, source_name, article_title = content_from_url, source_from_url, title_from_url
            logger.info(f"Extracted from URL. Source: {source_name}, Title: {article_title}")
        else:
            if not content_from_url:
                return ("❌ Failed to extract content from the provided URL. Please check the link or provide text directly.", None, None)
            else:
                return ("❌ Extracted article content is too short for analysis (min 100 chars).", None, None)

    if not article_content or len(article_content) < 100:
        return ("❌ Article content is too short for analysis (min 100 chars).", None, None)

    if not source_name:
        source_name = "Unknown Source"

    analyzer = ClaudeNewsAnalyzer(ANTHROPIC_API_KEY, MODEL_NAME)
    try:
        analysis_result = analyzer.analyze_article_text(article_content, source_name)
    except Exception as e:
        logger.error(f"Error during Claude analysis: {str(e)}")
        return (f"❌ Error during analysis: {str(e)}", None, None)

    try:
        credibility_saved = save_analysis_to_db(article_url, article_title, source_name, article_content, analysis_result)
        logger.info(f"Analysis saved to DB. Overall Credibility: {credibility_saved}")
    except Exception as e:
        logger.error(f"Error saving analysis to database: {str(e)}")
        return (f"❌ Error saving analysis: {str(e)}", None, None)

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

    factuality_display_score = 1.0 - fcn

    output_md = f"""### 📊 Credibility Analysis for: "{article_title}"
**Source:** {source_name}
**Media Owner:** {media_owners.get(source_name, "Unknown Owner")}
**Overall Calculated Credibility:** **{credibility_saved}** ({index_of_credibility*100:.1f}%)

---
#### 📊 Analysis Scores:
- **Integrity Score:** {ni*100:.1f}% - Measures the overall integrity and trustworthiness.
- **Factuality Score:** {factuality_display_score*100:.1f}% - Indicates likelihood of needing fact-checking.
- **Sentiment Score:** {ss:.2f} - Overall emotional tone (0.0 negative, 0.5 neutral, 1.0 positive).
- **Bias Score:** {bs*100:.1f}% - Degree of perceived bias (0.0 low, 1.0 high).
- **Index of Credibility:** {index_of_credibility*100:.1f}% - Overall credibility index.

---
#### 📝 Summary:
{short_summary}

#### 🔑 Key Arguments:
{("- " + "\\n- ".join(key_arguments)) if key_arguments else "N/A"}

#### 📈 Mentioned Facts/Data:
{("- " + "\\n- ".join(mentioned_facts)) if mentioned_facts else "N/A"}

#### 🎯 Author's Purpose:
{author_purpose}

#### 🚩 Potential Biases Identified:
{("- " + "\\n- ".join(potential_biases_identified)) if potential_biases_identified else "N/A"}

#### 🏷️ Main Topics Identified:
{", ".join(topics) if topics else "N/A"}

#### 📌 Media Owner Influence:
The media owner, {media_owners.get(source_name, "Unknown Owner")}, may influence source credibility.
"""

    scores_for_chart = {
        "Integrity": ni * 100,
        "Factuality": factuality_display_score * 100,
        "Neutral Sentiment": (1.0 - abs(ss - 0.5) * 2) * 100,
        "Low Bias": (1.0 - bs) * 100,
        "Overall Credibility Index": index_of_credibility * 100
    }

    return output_md, scores_for_chart, analysis_result

def generate_query(analysis_result):
    """Генерирует оптимизированный запрос для NewsAPI на основе результатов анализа."""
    topics = analysis_result.get('topics', [])
    key_arguments = analysis_result.get('key_arguments', [])
    mentioned_facts = analysis_result.get('mentioned_facts', [])

    all_terms = []

    for phrase_list in [topics, key_arguments]:
        for phrase in phrase_list:
            if not phrase.strip():
                continue
            if ' ' in phrase.strip() and len(phrase.strip().split()) > 1:
                all_terms.append(f'"{phrase.strip()}"')
            else:
                all_terms.append(phrase.strip())

    for fact in mentioned_facts:
        if not fact.strip():
            continue
        words = [word for word in fact.lower().split() if word not in stop_words_en and len(word) > 2]
        all_terms.extend(words)

    unique_terms = list(set(all_terms))

    if len(unique_terms) >= 3:
        query = " AND ".join(unique_terms)
    elif unique_terms:
        query = " OR ".join(unique_terms)
    else:
        query = "current events OR news"

    logger.info(f"Generated NewsAPI query: {query}")
    return query

def make_newsapi_request(params):
    """Вспомогательная функция для выполнения запросов к NewsAPI."""
    url = "https://newsapi.org/v2/everything"
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        return data.get("articles", [])
    except requests.exceptions.RequestException as e:
        logger.error(f"NewsAPI Error: {e}")
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"NewsAPI Response content: {e.response.text}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error in NewsAPI request: {e}")
        return []

def fetch_similar_news(analysis_result, days_range=4, max_articles=10):
    """Получает похожие новостные статьи с использованием NewsAPI."""
    if not NEWS_API_ENABLED:
        logger.warning("NEWS_API_KEY is not configured or enabled. Skipping similar news search.")
        return []

    initial_query = generate_query(analysis_result)
    url = "https://newsapi.org/v2/everything"

    # Определение диапазона дат
    original_published_date_str = analysis_result.get('published_date', 'N/A')
    end_date = datetime.now(UTC).date()

    if original_published_date_str and original_published_date_str != 'N/A':
        try:
            parsed_date = datetime.strptime(original_published_date_str, '%Y-%m-%d').date()
            start_date = parsed_date - timedelta(days=days_range)
            end_date = parsed_date + timedelta(days=days_range)
            logger.info(f"Using original article date ({parsed_date}) for NewsAPI search range: {start_date} to {end_date}")
        except ValueError:
            logger.warning(f"Could not parse original article date '{original_published_date_str}'. Using default range.")
            start_date = end_date - timedelta(days=days_range)
    else:
        start_date = end_date - timedelta(days=days_range)
        logger.info(f"No original article date found. Using default NewsAPI search range: {start_date} to {end_date}")

    # Попытка 1: Специфический запрос с надежными источниками
    params_specific = {
        "q": initial_query,
        "apiKey": NEWS_API_KEY,
        "language": "en",
        "pageSize": max_articles * 2,
        "sortBy": "relevancy",
        "from": start_date.strftime('%Y-%m-%d'),
        "to": end_date.strftime('%Y-%m-%d'),
    }

    if TRUSTED_NEWS_SOURCES_IDS:
        params_specific["sources"] = ",".join(TRUSTED_NEWS_SOURCES_IDS)

    articles_found = make_newsapi_request(params_specific)
    logger.info(f"[NewsAPI] Attempt 1 found {len(articles_found)} articles.")

    # Попытка 2: Более широкий запрос, если первая попытка дала мало результатов
    if len(articles_found) < (max_articles / 2) and initial_query != "current events OR news":
        logger.info("Few results from specific query, attempting broader search.")
        broader_query_terms = list(set(analysis_result.get('topics', [])[:3]))
        broader_query = " OR ".join([f'"{term}"' if ' ' in term else term for term in broader_query_terms if term and term not in stop_words_en])

        if not broader_query:
            broader_query = "current events OR news"

        params_broad = {
            "q": broader_query,
            "apiKey": NEWS_API_KEY,
            "language": "en",
            "pageSize": max_articles * 2,
            "sortBy": "relevancy",
            "from": start_date.strftime('%Y-%m-%d'),
            "to": end_date.strftime('%Y-%m-%d'),
        }

        if TRUSTED_NEWS_SOURCES_IDS:
            params_broad["sources"] = ",".join(TRUSTED_NEWS_SOURCES_IDS)

        additional_articles = make_newsapi_request(params_broad)
        articles_found.extend(additional_articles)
        logger.info(f"[NewsAPI] Attempt 2 found {len(additional_articles)} new articles. Total: {len(articles_found)}")

    # Удаление дубликатов статей
    unique_articles = {}
    for article in articles_found:
        if article.get('url'):
            unique_articles[article['url']] = article
    articles_found = list(unique_articles.values())

    if not articles_found:
        return []

    # Ранжирование статей по релевантности и доверию
    ranked_articles = []
    predefined_trust_scores = {
        "bbc.com": 0.9, "bbc.co.uk": 0.9, "reuters.com": 0.95, "apnews.com": 0.93,
        "nytimes.com": 0.88, "theguardian.com": 0.85, "wsj.com": 0.82,
        "cnn.com": 0.70, "foxnews.com": 0.40, "aljazeera.com": 0.80
    }

    # Объединение терминов из обоих запросов для проверки релевантности
    all_query_terms = []
    if 'initial_query' in locals():
        all_query_terms.extend([t.lower().replace('"', '') for t in initial_query.split(' AND ')])
    if 'broader_query' in locals():
        all_query_terms.extend([t.lower().replace('"', '') for t in broader_query.split(' OR ')])
    all_query_terms = list(set([t for t in all_query_terms if t and t not in stop_words_en]))

    for article in articles_found:
        source_domain = urlparse(article.get("url", '')).netloc.replace('www.', '')
        trust_score = predefined_trust_scores.get(source_domain, 0.5)

        # Вычисление оценки релевантности
        article_text = (article.get('title', '') + " " + article.get('description', '')).lower()
        relevance_score = sum(1 for term in all_query_terms if term in article_text)

        # Объединение оценок
        final_score = (relevance_score * 10) + (trust_score * 5)
        ranked_articles.append((article, final_score))

    # Сортировка и возврат лучших статей
    ranked_articles.sort(key=lambda item: item[1], reverse=True)
    top_articles = [item[0] for item in ranked_articles[:max_articles]]
    logger.info(f"Returning {len(top_articles)} top ranked similar articles.")
    return top_articles

def render_similar_articles_html(articles):
    """Генерирует HTML для отображения похожих статей."""
    if not articles:
        return "<p>No similar articles found for the selected criteria.</p>"

    predefined_trust_scores = {
        "bbc.com": 0.9, "bbc.co.uk": 0.9, "reuters.com": 0.95, "apnews.com": 0.93,
        "nytimes.com": 0.88, "theguardian.com": 0.85, "wsj.com": 0.82,
        "cnn.com": 0.70, "foxnews.com": 0.40, "aljazeera.com": 0.80
    }

    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()

        html_items = []
        for art in articles:
            title = html.escape(art.get("title", "No Title"))
            article_url = html.escape(art.get("url", "#"))
            source_api_name = html.escape(art.get("source", {}).get("name", "Unknown Source"))

            published_at_raw = art.get('publishedAt', 'N/A')
            published_at_display = html.escape(published_at_raw.split('T')[0] if 'T' in published_at_raw and published_at_raw != 'N/A' else published_at_raw)

            description_raw = art.get('description', 'No description available.')
            if description_raw.startswith(art.get("title", "")):
                description_raw = description_raw[len(art.get("title", "")):].strip()
                if description_raw.startswith("- "):
                    description_raw = description_raw[2:].strip()
            description_display = html.escape(description_raw)

            domain = urlparse(art.get("url", "#")).netloc.replace('www.', '')
            trust_display = ""

            # Получение исторической достоверности из source_stats
            c.execute("SELECT high, medium, low, total_analyzed FROM source_stats WHERE source = ?", (domain,))
            row = c.fetchone()

            if row:
                high, medium, low, total_analyzed = row
                if total_analyzed > 0:
                    score = (high * 1.0 + medium * 0.5 + low * 0.0) / total_analyzed
                    trust_display = f" (Hist. Src. Credibility: {score*100:.0f}%)"

            # Использование предопределенных оценок доверия, если нет данных в базе
            if not trust_display and domain in predefined_trust_scores:
                predefined_score = predefined_trust_scores.get(domain)
                trust_display = f" (Est. Src. Trust: {predefined_score*100:.0f}%)"
            elif not trust_display:
                trust_display = " (Src. Credibility: N/A)"

            trust_display = html.escape(trust_display)

            html_items.append(
                f"""
                <div class="similar-article">
                    <h4><a href="{article_url}" target="_blank" rel="noopener noreferrer">
                        {title}
                    </a></h4>
                    <p><strong>Source:</strong> {source_api_name}{trust_display} | <strong>Published:</strong> {published_at_display}</p>
                    <p>{description_display}</p>
                </div>
                <hr>
                """
            )

        return f"""
        <div class="similar-articles-container">
            <h3>🔗 Similar News Articles (Ranked by Relevance & Trust):</h3>
            {"".join(html_items)}
        </div>
        """

    except sqlite3.Error as e:
        logger.error(f"Database error in render_similar_articles_html: {e}")
        return "<p>Error retrieving similar articles data due to a database issue.</p>"
    finally:
        if 'conn' in locals() and conn:
            conn.close()

def get_source_reliability_data():
    """Получает данные о надежности источников для графика Plotly."""
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()

        c.execute("""
            SELECT source, high, medium, low, total_analyzed
            FROM source_stats
            ORDER BY total_analyzed DESC, source ASC
        """)

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
                score = 0.5
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
        logger.error(f"Database error in get_source_reliability_data: {e}")
        return {
            'sources': [],
            'credibility_indices_for_plot': [],
            'high_counts': [],
            'medium_counts': [],
            'low_counts': [],
            'total_analyzed_counts': []
        }
    finally:
        if 'conn' in locals() and conn:
            conn.close()

def get_analysis_history_html():
    """Получает и форматирует историю анализа из базы данных."""
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()

        c.execute("""
            SELECT url, title, source, credibility_level, short_summary,
                   strftime('%Y-%m-%d %H:%M', analysis_date) as formatted_date
            FROM news
            ORDER BY analysis_date DESC
            LIMIT 15
        """)

        rows = c.fetchall()

        if not rows:
            return "<p>No analysis history yet. Analyze an article to see it appear here!</p>"

        html_items = []
        for url, title, source, credibility, short_summary, date_str in rows:
            display_title = title[:70] + '...' if title and len(title) > 70 else title if title else "N/A"
            source_display = source if source else "N/A"
            link_start = f"<a href='{url}' target='_blank' rel='noopener noreferrer'>" if url and url.startswith(('http://', 'https://')) else ""
            link_end = "</a>" if url and url.startswith(('http://', 'https://')) else ""
            summary_display = short_summary if short_summary else 'No summary available.'

            html_items.append(
                f"""
                <li>
                    <strong>{date_str}</strong>: {link_start}{display_title}{link_end} ({source_display}, {credibility})
                    <br>
                    <em>Summary:</em> {summary_display}
                </li>
                """
            )

        return f"<h3>📜 Recent Analyses:</h3><ul>{''.join(html_items)}</ul>"

    except sqlite3.Error as e:
        logger.error(f"Database error in get_analysis_history_html: {e}")
        return "<p>Error retrieving analysis history due to a database issue.</p>"
    finally:
        if 'conn' in locals() and conn:
            conn.close()

# Маршруты Flask
@app.route('/')
def index():
    """Отображает главную страницу."""
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error rendering index template: {e}")
        return "Welcome to the News Credibility Analyzer", 200

@app.route('/analyze', methods=['POST'])
def analyze():
    """Обрабатывает запрос на анализ статьи."""
    try:
        data = request.json
        input_text = data.get('input_text')
        source_name_manual = data.get('source_name_manual')

        logger.info(f"Received input_text (first 50 chars): {input_text[:50]}...")
        logger.info(f"Received source_name_manual: {source_name_manual}")

        output_md, scores_for_chart, analysis_result = process_article_analysis(input_text, source_name_manual)

        if analysis_result is None:
            return jsonify({'error_message': output_md}), 400

        logger.info(f"Analysis result generated. Sending to client.")
        return jsonify({
            'output_md': output_md,
            'scores_for_chart': scores_for_chart,
            'analysis_result': analysis_result
        })

    except Exception as e:
        logger.error(f"Error in analyze endpoint: {e}")
        return jsonify({'error_message': f"An error occurred during analysis: {str(e)}"}), 500

@app.route('/similar_articles', methods=['POST'])
def similar_articles_endpoint():
    """Получает похожие статьи."""
    try:
        data = request.json
        analysis_result = data.get('analysis_result')

        if not analysis_result:
            return jsonify({'similar_html': "<p>No analysis result provided to fetch similar articles.</p>"})

        similar_articles_list = fetch_similar_news(analysis_result)
        similar_html = render_similar_articles_html(similar_articles_list)

        return jsonify({'similar_html': similar_html})

    except Exception as e:
        logger.error(f"Error in similar_articles_endpoint: {e}")
        return jsonify({'similar_html': f"<p>Error fetching similar articles: {str(e)}</p>"}), 500

@app.route('/source_reliability_data')
def source_reliability_data_endpoint():
    """Получает данные о надежности источников."""
    try:
        data = get_source_reliability_data()

        labeled_sources = []
        if data['sources'] and data['credibility_indices_for_plot']:
            for source, score, total in zip(
                data['sources'],
                data['credibility_indices_for_plot'],
                data['total_analyzed_counts']
            ):
                labeled_sources.append(
                    f"{source}<br>Credibility: {score*100:.0f}%<br>Articles: {total}"
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
        logger.error(f"Error in source_reliability_data endpoint: {e}")
        return jsonify({
            'sources': [],
            'credibility_indices_for_plot': [],
            'high_counts': [],
            'medium_counts': [],
            'low_counts': [],
            'total_analyzed_counts': []
        }), 500

@app.route('/analysis_history_html')
def analysis_history_html_endpoint():
    """Получает HTML истории анализа."""
    try:
        history_html = get_analysis_history_html()
        return jsonify({'history_html': history_html})
    except Exception as e:
        logger.error(f"Error in analysis_history_html_endpoint: {e}")
        return jsonify({'history_html': f"<p>Error retrieving analysis history: {str(e)}</p>"}), 500

@app.route('/check_db_integrity')
def check_db_integrity_endpoint():
    """Проверяет целостность базы данных."""
    try:
        result = check_database_integrity()
        if result:
            return jsonify({
                'status': 'success',
                'message': 'Database integrity check passed',
                'details': 'All database tables and structures are valid'
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Database integrity check failed',
                'details': 'Critical database issues detected'
            }), 500
    except Exception as e:
        logger.error(f"Error in check_db_integrity_endpoint: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Database check failed: {str(e)}',
            'details': 'Unexpected error during database check'
        }), 500

@app.before_request
def maintenance_mode():
    """Режим обслуживания."""
    if os.getenv("MAINTENANCE") == "true":
        return render_template("maintenance.html"), 503

def initialize_database():
    """Инициализирует схему базы данных и проверяет целостность."""
    try:
        ensure_db_schema()

        if not check_database_integrity():
            logger.error("Database integrity check failed. Attempting to recover...")
            try:
                logger.info("Attempting to recreate database schema...")
                ensure_db_schema()
                logger.info("Reinitializing sources with default data...")
                initialize_sources(INITIAL_SOURCE_COUNTS)
                logger.info("Database recovery attempt completed")
            except Exception as e:
                logger.error(f"Failed to recover database: {e}")
                raise
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise

if __name__ == '__main__':
    try:
        # Инициализация базы данных
        initialize_database()

        # Запуск Flask приложения
        app.run(debug=True, host='0.0.0.0', port=5000)
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        exit(1)
