from flask import Flask, render_template, request, jsonify
import os
import json
import sqlite3
import requests
import anthropic
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from dotenv import load_dotenv
# import html # Убрали импорт html, так как больше не будем экранировать
from datetime import datetime, timedelta, UTC
import logging
import re
import plotly.graph_objects as go
from stop_words import get_stop_words # pip install stop-words
from newspaper import Article # pip install newspaper3k
import html # Для html.escape, если еще не импортирован глобально

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
NEWS_API_KEY = os.getenv('NEWS_API_KEY')
NEWS_API_ENABLED = bool(NEWS_API_KEY)
MODEL_NAME = "claude-3-opus-20240229"

if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY is missing! Please set it in your .env file.")
if not NEWS_API_KEY:
    logger.warning("NEWS_API_KEY is missing! Similar news functionality will be disabled.")
    NEWS_API_ENABLED = False

# --- Database Setup ---
DB_NAME = 'news_analysis.db'

# Initial data for source reliability
INITIAL_SOURCE_COUNTS = {
    "bbc.com": {"high": 15, "medium": 5, "low": 1},
    "reuters.com": {"high": 20, "medium": 3, "low": 0},
    "foxnews.com": {"high": 3, "medium": 7, "low": 15},
    "cnn.com": {"high": 5, "medium": 10, "low": 5},
    "nytimes.com": {"high": 10, "medium": 5, "low": 2},
    "theguardian.com": {"high": 12, "medium": 4, "low": 1},
    "apnews.com": {"high": 18, "medium": 2, "low": 0}
}

# Mapping of domains to media owners
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
    "aljazeera.com": "Al Jazeera Media Network" # Added Al Jazeera
}

# NewsAPI source IDs for filtering
TRUSTED_NEWS_SOURCES_IDS = [
    "bbc-news", "reuters", "associated-press", "the-new-york-times",
    "the-guardian-uk", "the-wall-street-journal", "cnn", "al-jazeera-english"
]

stop_words_en = get_stop_words('en')

def ensure_db_schema():
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
    conn.close()
    logger.info("Database schema ensured.")

def initialize_sources(initial_counts):
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
            logger.info(f" - Adding source: {source} with initial counts (H:{high}, M:{medium}, L:{low}, Total:{total_analyzed})")
            try:
                c.execute('''INSERT INTO source_stats (source, high, medium, low, total_analyzed)
                                 VALUES (?, ?, ?, ?, ?)''', (source, high, medium, low, total_analyzed))
                conn.commit()
            except sqlite3.Error as e:
                logger.error(f"Error initializing source {source}: {e}")
                conn.rollback()
    conn.close()
    logger.info("Initial source initialization completed.")

# Ensure DB and initialize sources on app startup
ensure_db_schema()
initialize_sources(INITIAL_SOURCE_COUNTS)

class ClaudeNewsAnalyzer:
    def __init__(self, api_key, model_name):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model_name = model_name

    def analyze_article_text(self, article_text_content, source_name_for_context):
        max_chars_for_claude = 10000 # Claude 3 Opus context window is 200k tokens
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
- "topics": (List of strings) Identify 3-5 main topics or keywords that accurately represent the core subject matter. These should be suitable for searching for related articles.
- "key_arguments": (List of strings) Extract the main arguments or claims made by the author.
- "mentioned_facts": (List of strings) List any specific facts, data, or statistics mentioned.
- "author_purpose": (String) Briefly determine the author's likely primary purpose (e.g., "to inform objectively", "to persuade readers of a viewpoint", "to evoke emotion", "to report breaking news").
- "potential_biases_identified": (List of strings) Enumerate any specific signs of potential bias or subjectivity observed (e.g., "loaded language", "one-sided reporting", "appeal to emotion", "unattributed sources").
- "short_summary": (String) A concise summary of the article's main content in 2-4 sentences.
- "index_of_credibility": (Float, 0.0-1.0) Calculate an overall index of credibility based on the above factors (news_integrity, fact_check_needed_score, sentiment_score, bias_score). Higher is better.
- "published_date": (String, YYYY-MM-DD or N/A) The publication date of the article. Extract this from the article text if possible, otherwise state "N/A".
"""
        try:
            message = self.client.messages.create(
                model=self.model_name,
                max_tokens=2000,
                temperature=0.2, # Keep temperature low for factual analysis
                system="You are a JSON-generating expert. Always provide valid JSON.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            raw_json_text = message.content[0].text.strip()
            # Try to extract JSON from code block if present (Claude sometimes wraps it)
            match = re.search(r'```json\s*(\{.*\})\s*```', raw_json_text, re.DOTALL)
            if match:
                json_str = match.group(1)
            else:
                json_str = raw_json_text # Assume direct JSON if no code block

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
    """Extracts article content, source, and title from a URL using newspaper3k."""
    try:
        # Clean URL for newspaper3k to handle AMP links better
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
    """Calculates a qualitative credibility level based on AI scores."""
    fact_check_score = 1.0 - fact_check_needed
    neutral_sentiment_proximity = 1.0 - abs(sentiment - 0.5) * 2 # Higher if closer to 0.5
    bias_score_inverted = 1.0 - bias # Higher if lower bias

    # Weighted average for overall credibility
    # Adjusted weights to make integrity and factuality more dominant
    avg = (integrity * 0.45) + \
          (fact_check_score * 0.35) + \
          (neutral_sentiment_proximity * 0.10) + \
          (bias_score_inverted * 0.10)

    if avg >= 0.75: return 'High'
    if avg >= 0.5: return 'Medium'
    return 'Low'

def save_analysis_to_db(url, title, source, content, analysis_result):
    """Saves analysis results to the SQLite database and updates source statistics."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    integrity = analysis_result.get('news_integrity', 0.0)
    fact_check_needed = analysis_result.get('fact_check_needed_score', 1.0)
    sentiment = analysis_result.get('sentiment_score', 0.5)
    bias = analysis_result.get('bias_score', 1.0)
    short_summary = analysis_result.get('short_summary', 'Summary not available.')
    index_of_credibility = analysis_result.get('index_of_credibility', 0.0)

    credibility_level = calculate_credibility_level(
        integrity, fact_check_needed, sentiment, bias
    )

    db_url = url if url else f"no_url_{datetime.now(UTC).timestamp()}"
    
    try:
        c.execute('''INSERT INTO news (url, title, source, content, integrity, fact_check, sentiment, bias, credibility_level, short_summary, index_of_credibility)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                     ON CONFLICT(url) DO UPDATE SET
                     title=excluded.title, source=excluded.source, content=excluded.content,
                     integrity=excluded.integrity, fact_check=excluded.fact_check,
                     sentiment=excluded.sentiment, bias=excluded.bias,
                     credibility_level=excluded.credibility_level,
                     short_summary=excluded.short_summary,
                     index_of_credibility=excluded.index_of_credibility,
                     analysis_date=CURRENT_TIMESTAMP''',
                  (db_url, title, source, content,
                   integrity, fact_check_needed, sentiment, bias,
                   credibility_level, short_summary, index_of_credibility))

        # Update source_stats table
        c.execute("SELECT high, medium, low, total_analyzed FROM source_stats WHERE source = ?", (source,))
        row = c.fetchone()
        if row:
            high, medium, low, total = row
            if credibility_level == 'High': high += 1
            elif credibility_level == 'Medium': medium += 1
            else: low += 1
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
    except sqlite3.Error as e:
        logger.error(f"Database error in save_analysis_to_db: {e}")
        conn.rollback()
    finally:
        conn.close()
    
    return credibility_level

def process_article_analysis(input_text, source_name_manual):
    """Orchestrates the full analysis pipeline for an article."""
    article_url = None
    article_content = input_text
    article_title = "User-provided Text"
    source_name = source_name_manual if source_name_manual else "Direct Input"

    if input_text.strip().startswith("http"):
        article_url = input_text.strip()
        logger.info(f"Input is a URL: {article_url}")
        content_from_url, source_from_url, title_from_url = extract_text_from_url(article_url)
        if content_from_url and len(content_from_url) >= 100: # Ensure extracted content is substantial
            article_content, source_name, article_title = content_from_url, source_from_url, title_from_url
            logger.info(f"Extracted from URL. Source: {source_name}, Title: {article_title}")
        else:
            if not content_from_url:
                return ("❌ Failed to extract content from the provided URL. Please check the link or provide text directly. Ensure the URL is publicly accessible.", None, None)
            else: # content_from_url is too short
                return ("❌ Extracted article content is too short for analysis (min 100 chars). Please check the link or provide more text directly.", None, None)

    if not article_content or len(article_content) < 100:
        return ("❌ Article content is too short for analysis (min 100 chars).", None, None)

    if not source_name:
        source_name = "Unknown Source"

    analyzer = ClaudeNewsAnalyzer(ANTHROPIC_API_KEY, MODEL_NAME)
    try:
        analysis_result = analyzer.analyze_article_text(article_content, source_name)
    except Exception as e:
        return (f"❌ Error during Claude analysis: {str(e)}", None, None)

    credibility_saved = save_analysis_to_db(article_url, article_title, source_name, article_content, analysis_result)
    logger.info(f"Analysis saved to DB. Overall Credibility: {credibility_saved}")

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


    factuality_display_score = 1.0 - fcn # Convert "fact_check_needed" to "factuality"

    # Changed output_md to use standard Markdown syntax for readability
    output_md = f"""### 📊 Credibility Analysis for: "{article_title}"
**Source:** {source_name}
**Media Owner:** {media_owners.get(source_name, "Unknown Owner")}
**Overall Calculated Credibility:** **{credibility_saved}** ({index_of_credibility*100:.1f}%)

---
#### **📊 Analysis Scores:**
- **Integrity Score:** {ni*100:.1f}% - Measures the overall integrity and trustworthiness of the information presented.
- **Factuality Score:** {factuality_display_score*100:.1f}% - Indicates the likelihood that the article's claims require external fact-checking.
- **Sentiment Score:** {ss:.2f} - Overall emotional tone (0.0 negative, 0.5 neutral, 1.0 positive).
- **Bias Score:** {bs*100:.1f}% - Degree of perceived bias (0.0 low bias, 1.0 high bias).
- **Index of Credibility:** {index_of_credibility*100:.1f}% - Overall index of credibility based on the above factors.

---
#### **📝 Summary:**
{short_summary}

#### **🔑 Key Arguments:**
{("- " + "\\n- ".join(key_arguments)) if key_arguments else "N/A"}

#### **📈 Mentioned Facts/Data:**
{("- " + "\\n- ".join(mentioned_facts)) if mentioned_facts else "N/A"}

#### **🎯 Author's Purpose:**
{author_purpose}

#### **🚩 Potential Biases Identified:**
{("- " + "\\n- ".join(potential_biases_identified)) if potential_biases_identified else "N/A"}

#### **🏷️ Main Topics Identified:**
{", ".join(topics) if topics else "N/A"}

#### **📌 Media Owner Influence:**
The media owner, {media_owners.get(source_name, "Unknown Owner")}, may influence the credibility of the source. Different media owners have varying levels of trustworthiness and potential biases.
"""
    scores_for_chart = {
        "Integrity": ni * 100,
        "Factuality": factuality_display_score * 100,
        "Neutral Sentiment": (1.0 - abs(ss - 0.5) * 2) * 100,
        "Low Bias": (1.0 - bs) * 100,
        "Overall Credibility Index": index_of_credibility * 100 # Изменено название для графика
    }

    return output_md, scores_for_chart, analysis_result

def generate_query(analysis_result):
    """Generates an optimized NewsAPI query from Claude's analysis results."""
    topics = analysis_result.get('topics', [])
    key_arguments = analysis_result.get('key_arguments', [])
    mentioned_facts = analysis_result.get('mentioned_facts', [])

    all_terms = []
    # Prioritize specific phrases from topics and key arguments
    for phrase_list in [topics, key_arguments]:
        for phrase in phrase_list:
            if not phrase.strip(): continue
            # If phrase contains multiple words, wrap it in quotes for exact match
            if ' ' in phrase.strip() and len(phrase.strip().split()) > 1:
                all_terms.append(f'"{phrase.strip()}"')
            else: # Single word or already quoted, just add it
                all_terms.append(phrase.strip())
    
    # Add important single words from mentioned facts, filtering stop words
    for fact in mentioned_facts:
        if not fact.strip(): continue
        words = [word for word in fact.lower().split() if word not in stop_words_en and len(word) > 2]
        all_terms.extend(words)
    
    # Remove duplicates and join with 'AND' for specificity, or 'OR' for broader if too few terms
    unique_terms = list(set(all_terms))
    
    if len(unique_terms) >= 3: # If enough specific terms, use AND
        query = " AND ".join(unique_terms)
    elif unique_terms: # Otherwise use OR
        query = " OR ".join(unique_terms)
    else: # Fallback if no useful terms extracted
        query = "current events OR news"
    
    logger.info(f"Generated NewsAPI query: {query}")
    return query

def fetch_similar_news(analysis_result, days_range=3, max_articles=10): # Увеличил days_range с 3 до 7
    """Fetches similar news articles using NewsAPI based on analysis results."""
    if not NEWS_API_ENABLED:
        logger.warning("NEWS_API_KEY is not configured or enabled. Skipping similar news search.")
        return []

    initial_query = generate_query(analysis_result)
    
    # Determine date range based on original article's published date, if available
    original_published_date_str = analysis_result.get('published_date', 'N/A')
    start_date = (datetime.now(UTC) - timedelta(days=days_range)).date()
    end_date = datetime.now(UTC).date()

    if original_published_date_str and original_published_date_str != 'N/A':
        try:
            parsed_date = datetime.strptime(original_published_date_str, '%Y-%m-%d').date()
            # Search within a window around the original article's date
            # Расширяем окно поиска вокруг даты публикации статьи
            start_date = parsed_date - timedelta(days=days_range) 
            end_date = parsed_date + timedelta(days=days_range)
            logger.info(f"Using original article date ({parsed_date}) for NewsAPI search range: {start_date} to {end_date}")
        except ValueError:
            logger.warning(f"Could not parse original article date '{original_published_date_str}'. Using default range (last {days_range} days).")
    else:
        logger.info(f"No original article date found. Using default NewsAPI search range (last {days_range} days): {start_date} to {end_date}")

    url = "https://newsapi.org/v2/everything"
    
    # --- Attempt 1: Specific Query with Trusted Sources ---
    params_specific = {
        "q": initial_query,
        "apiKey": NEWS_API_KEY,
        "language": "en",
        "pageSize": max_articles * 3, # Fetch more to allow for better filtering/ranking
        "sortBy": "relevancy",
        "from": start_date.strftime('%Y-%m-%d'),
        "to": end_date.strftime('%Y-%m-%d'),
        "sources": ",".join(TRUSTED_NEWS_SOURCES_IDS) if TRUSTED_NEWS_SOURCES_IDS else None
    }
    if not params_specific["sources"]:
        logger.warning("No trusted NewsAPI sources specified. Attempting search without source filter.")
        del params_specific["sources"] # Remove if empty to avoid API error

    articles_found = []
    try:
        response = requests.get(url, params=params_specific, timeout=15)
        response.raise_for_status()
        data = response.json()
        articles_found = data.get("articles", [])
        logger.info(f"[NewsAPI] Attempt 1 (specific query, trusted sources) found {len(articles_found)} articles.")

    except requests.exceptions.RequestException as e:
        logger.error(f"[NewsAPI] Attempt 1 API Error for query '{initial_query}': {e}")
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"[NewsAPI] Attempt 1 Response content: {e.response.text}")
    except Exception as e:
        logger.error(f"[NewsAPI] Unexpected error in Attempt 1: {e}")

    # --- Attempt 2: Broader Query if Attempt 1 yields too few results ---
    # Increased threshold for trying broader query
    if len(articles_found) < (max_articles / 2) and initial_query != "current events OR news": # Попробуем более широкий запрос, если мало результатов
        logger.info("Few results from specific query, attempting broader search.")
        broader_query_terms = list(set(analysis_result.get('topics', [])[:3]))
        broader_query = " OR ".join([f'"{term}"' if ' ' in term else term for term in broader_query_terms if term and term not in stop_words_en])
        
        if not broader_query: # Fallback if even topics are empty
            broader_query = "current events OR news"
        
        params_broad = {
            "q": broader_query,
            "apiKey": NEWS_API_KEY,
            "language": "en",
            "pageSize": max_articles * 3,
            "sortBy": "relevancy",
            "from": start_date.strftime('%Y-%m-%d'),
            "to": end_date.strftime('%Y-%m-%d'),
            "sources": ",".join(TRUSTED_NEWS_SOURCES_IDS) if TRUSTED_NEWS_SOURCES_IDS else None
        }
        if not params_broad["sources"]:
            del params_broad["sources"]

        try:
            response = requests.get(url, params=params_broad, timeout=15)
            response.raise_for_status()
            data = response.json()
            articles_found.extend(data.get("articles", [])) # Add to existing
            logger.info(f"[NewsAPI] Attempt 2 (broader query) found {len(data.get('articles', []))} new articles. Total: {len(articles_found)}")
        except requests.exceptions.RequestException as e:
            logger.error(f"[NewsAPI] Attempt 2 API Error for query '{broader_query}': {e}")
        except Exception as e:
            logger.error(f"[NewsAPI] Unexpected error in Attempt 2: {e}")

    # Deduplicate articles based on URL
    unique_articles = {}
    for article in articles_found:
        if article.get('url'):
            unique_articles[article['url']] = article
    articles_found = list(unique_articles.values())
    
    if not articles_found:
        return []

    # Simple ranking by pre-defined trust scores and keyword relevance
    ranked_articles = []
    predefined_trust_scores = {
        "bbc.com": 0.9, "bbc.co.uk": 0.9, "reuters.com": 0.95, "apnews.com": 0.93,
        "nytimes.com": 0.88, "theguardian.com": 0.85, "wsj.com": 0.82,
        "cnn.com": 0.70, "foxnews.com": 0.40, "aljazeera.com": 0.80
    }

    # Combine terms from both initial and broader queries for comprehensive relevance check
    all_query_terms_for_relevance = []
    if 'initial_query' in locals():
        all_query_terms_for_relevance.extend([t.lower().replace('"', '') for t in initial_query.split(' AND ')])
    if 'broader_query' in locals():
        all_query_terms_for_relevance.extend([t.lower().replace('"', '') for t in broader_query.split(' OR ')])
    all_query_terms_for_relevance = list(set([t for t in all_query_terms_for_relevance if t and t not in stop_words_en]))

    for article in articles_found:
        source_domain = urlparse(article.get('url', '')).netloc.replace("www.", "")
        trust_score = predefined_trust_scores.get(source_domain, 0.5) # Default to 0.5 if not in our list
        
        # Calculate relevance based on presence of query terms in title/description
        article_text_for_relevance = (article.get('title', '') + " " + article.get('description', '')).lower()
        relevance_score = sum(1 for term in all_query_terms_for_relevance if term in article_text_for_relevance)
        
        # Combine scores: Higher weight for relevance, moderate for trust
        # Adjusted weights for better balance. Relevance is now more critical.
        # Увеличил вес для relevance_score, чтобы он был более определяющим
        final_score = (relevance_score * 15) + (trust_score * 5) 
        ranked_articles.append((article, final_score))
    
    ranked_articles.sort(key=lambda item: item[1], reverse=True)
    top_articles = [item[0] for item in ranked_articles[:max_articles]]
    logger.info(f"Returning {len(top_articles)} top ranked similar articles.")
    return top_articles


# Предполагается, что DB_NAME = 'news_analysis.db' определена глобально в вашем скрипте

def render_similar_articles_html(articles):
    """Generates HTML for displaying similar articles with credibility scores."""
    if not articles:
        return "<p>No similar articles found for the selected criteria. Try a different article or adjust the search range.</p>"

    # Ваши predefined_trust_scores, они будут использоваться как fallback
    predefined_trust_scores = {
        "bbc.com": 0.9, "bbc.co.uk": 0.9, "reuters.com": 0.95, "apnews.com": 0.93,
        "nytimes.com": 0.88, "theguardian.com": 0.85, "wsj.com": 0.82,
        "cnn.com": 0.70, "foxnews.com": 0.40, "aljazeera.com": 0.80
    }

    conn = None # Инициализируем conn
    try:
        conn = sqlite3.connect(DB_NAME) # Используем глобальную переменную DB_NAME
        c = conn.cursor()

        html_items = []
        for art in articles:
            title = html.escape(art.get("title", "No Title")) # Экранируем для безопасности
            article_url = html.escape(art.get("url", "#")) # Экранируем URL
            source_api_name = html.escape(art.get("source", {}).get("name", "Unknown Source"))
            
            published_at_raw = art.get('publishedAt', 'N/A')
            published_at_display = html.escape(published_at_raw.split('T')[0] if 'T' in published_at_raw and published_at_raw != 'N/A' else published_at_raw)
            
            description_raw = art.get('description', 'No description available.')
            if description_raw.startswith(art.get("title", "")): # Avoid duplicating title in description
                description_raw = description_raw[len(art.get("title", "")):].strip()
                if description_raw.startswith("- "): description_raw = description_raw[2:].strip()
            description_display = html.escape(description_raw)


            domain = urlparse(art.get("url", "#")).netloc.replace('www.', '') # Не экранируем domain, т.к. он для логики
            
            trust_display = "" # Инициализируем строку для отображения доверия

            # 1. Пытаемся получить историческую достоверность из source_stats
            c.execute("SELECT high, medium, low, total_analyzed FROM source_stats WHERE source = ?", (domain,))
            row = c.fetchone()

            if row:
                high, medium, low, total_analyzed = row
                if total_analyzed > 0:
                    # Рассчитываем исторический средний балл достоверности источника
                    score = (high * 1.0 + medium * 0.5 + low * 0.0) / total_analyzed
                    trust_display = f" (Hist. Src. Credibility: {score*100:.0f}%)"
            
            # 2. Если не нашли в source_stats или там нет анализов, используем predefined_trust_scores
            if not trust_display and domain in predefined_trust_scores:
                predefined_score = predefined_trust_scores.get(domain)
                trust_display = f" (Est. Src. Trust: {predefined_score*100:.0f}%)"
            elif not trust_display: # Если все еще нет информации о доверии
                trust_display = " (Src. Credibility: N/A)"
            
            trust_display = html.escape(trust_display) # Экранируем результат

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
    except sqlite3.Error as e:
        print(f"Database error in render_similar_articles_html: {e}")
        return "<p>Error retrieving similar articles data due to a database issue.</p>"
    finally:
        if conn:
            conn.close()
            
    return f"""
    <div class="similar-articles-container">
        <h3>🔗 Similar News Articles (Ranked by Relevance & Trust):</h3>
        {"".join(html_items) if html_items else "<p>No articles to display or an error occurred.</p>"}
    </div>
    """

def get_source_reliability_data():
    """Fetches source reliability data from DB for Plotly chart."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT source, high, medium, low, total_analyzed FROM source_stats ORDER BY total_analyzed DESC, source ASC")
    data = c.fetchall()
    conn.close()

    sources = []
    high_counts = []
    medium_counts = []
    low_counts = []
    # trust_scores_for_plot теперь будут называться credibility_indices_for_plot
    credibility_indices_for_plot = []
    total_analyzed_counts = []

    if not data:
        # Provide sample data if DB is empty for initial display
        sample_data = [
            ('Sample Source A (High Trust)', 10, 2, 1),
            ('Sample Source B (Mixed Trust)', 5, 5, 3),
            ('Sample Source C (Low Trust)', 1, 2, 10)
        ]
        for s, h, m, l in sample_data:
            total = h + m + l
            score = (h * 1.0 + m * 0.5 + l * 0.0) / total if total > 0 else 0.5
            sources.append(f"<b>{s}</b><br>{score*100:.1f}% ({total})")
            credibility_indices_for_plot.append(score)
            high_counts.append(h)
            medium_counts.append(m)
            low_counts.append(l)
            total_analyzed_counts.append(total)
        return sources, credibility_indices_for_plot, high_counts, medium_counts, low_counts, total_analyzed_counts

    for source, high, medium, low, total in data:
        total_current = high + medium + low # Ensure total_analyzed is correct
        if total_current == 0:
            score = 0.5
            score_display = "N/A"
            total_display = "0"
        else:
            score = (high * 1.0 + medium * 0.5 + low * 0.0) / total_current
            score_display = f"{score*100:.1f}%"
            total_display = str(total_current)

        x_axis_label = f"<b>{source}</b><br>{score_display} ({total_display})"
        sources.append(x_axis_label)
        credibility_indices_for_plot.append(score)
        high_counts.append(high)
        medium_counts.append(medium)
        low_counts.append(low)
        total_analyzed_counts.append(total_current)

    return sources, credibility_indices_for_plot, high_counts, medium_counts, low_counts, total_analyzed_counts


def get_analysis_history_html():
    """Retrieves and formats recent analysis history from the database."""
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
    conn.close()

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

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    input_text = data.get('input_text')
    source_name_manual = data.get('source_name_manual')

    logger.info(f"Received input_text (first 50 chars): {input_text[:50]}...")
    logger.info(f"Received source_name_manual: {source_name_manual}")

    output_md, scores_for_chart, analysis_result = process_article_analysis(input_text, source_name_manual)

    if analysis_result is None: # Error occurred during analysis
        return jsonify({'error_message': output_md}), 400
    
    logger.info(f"Analysis result generated. Sending to client.")
    return jsonify({
        'output_md': output_md,
        'scores_for_chart': scores_for_chart,
        'analysis_result': analysis_result # Pass the full analysis_result for similar news
    })

@app.route('/similar_articles', methods=['POST'])
def similar_articles_endpoint():
    data = request.json
    analysis_result = data.get('analysis_result')

    if not analysis_result:
        return jsonify({'similar_html': "<p>No analysis result provided to fetch similar articles.</p>"})

    similar_articles_list = fetch_similar_news(analysis_result)
    similar_html = render_similar_articles_html(similar_articles_list)

    return jsonify({
        'similar_html': similar_html
    })

@app.route('/source_reliability_data')
def source_reliability_data():
    # Изменено название переменной
    sources, credibility_indices_for_plot, high_counts, medium_counts, low_counts, total_analyzed_counts = get_source_reliability_data()
    return jsonify({
        'sources': sources,
        'credibility_indices_for_plot': credibility_indices_for_plot, # Изменено здесь
        'high_counts': high_counts,
        'medium_counts': medium_counts,
        'low_counts': low_counts,
        'total_analyzed_counts': total_analyzed_counts
    })

@app.route('/analysis_history_html')
def analysis_history_html_endpoint():
    history_html = get_analysis_history_html()
    return jsonify({
        'history_html': history_html
    })

if __name__ == '__main__':
    app.run(debug=True)
