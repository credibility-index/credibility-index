import os
import logging
import json
import time
import re
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from urllib.parse import urlparse
import anthropic
from pydantic import BaseModel
from cache import CacheManager
from news_api import NewsAPI  # Используем только NewsAPI вместо EnhancedNewsAPI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ArticleAnalysis(BaseModel):
    """Model for article analysis results"""
    credibility_score: Dict[str, Union[float, Dict[str, float]]]
    sentiment: Dict[str, Union[float, Dict[str, float]]]
    bias: Dict[str, Union[float, Dict[str, float]]]
    topics: List[Dict[str, Any]]
    perspectives: Dict[str, Dict[str, Any]]
    key_arguments: List[str]

class ClaudeAPI:
    """API client for interacting with Anthropic's Claude model"""

    def __init__(self):
        self.cache = CacheManager()
        self.news_api = NewsAPI()  # Используем NewsAPI вместо EnhancedNewsAPI
        self.client = self._initialize_anthropic_client()
        self.max_retries = 3
        self.retry_delay = 1
        self.fallback_topics = [
            {"name": "General News", "relevance": 0.8, "description": "General news"},
            {"name": "Politics", "relevance": 0.7, "description": "Political news"},
            {"name": "Economy", "relevance": 0.6, "description": "Economic news"}
        ]

    def _initialize_anthropic_client(self) -> Optional[anthropic.Anthropic]:
        """Initializes Anthropic client with error handling"""
        try:
            api_key = os.getenv('ANTHROPIC_API_KEY')
            if not api_key:
                logger.warning("ANTHROPIC_API_KEY is not set, mock mode will be used")
                return None
            return anthropic.Anthropic(api_key=api_key)
        except Exception as e:
            logger.error(f"Error initializing client: {str(e)}")
            return None

    def _make_api_request_with_retry(self, method: str, **kwargs) -> Any:
        """Makes API request with retries"""
        retry_count = 0
        last_error = None
        if not self.client:
            raise Exception("Anthropic client is not initialized")
        while retry_count < self.max_retries:
            try:
                if method == "messages.create":
                    response = self.client.messages.create(**kwargs)
                    return response
                else:
                    raise ValueError(f"Unknown API method: {method}")
            except anthropic.APIStatusError as e:
                status_code = e.status_code
                if status_code in (529, 429, 500, 502, 503, 504):
                    retry_after = int(e.response.headers.get('Retry-After', 5))
                    wait_time = min(retry_after, self.retry_delay * (2 ** retry_count))
                    logger.warning(f"API returned code {status_code}. Retrying in {wait_time} seconds")
                    time.sleep(wait_time)
                    retry_count += 1
                    continue
                raise
            except Exception as e:
                retry_count += 1
                wait_time = self.retry_delay * (2 ** (retry_count - 1))
                logger.warning(f"API error. Retrying in {wait_time} seconds")
                time.sleep(wait_time)
                continue
        logger.error(f"All retries exhausted. Last error: {str(last_error)}")
        raise Exception(f"All retries exhausted. Last error: {str(last_error)}")

    def _safe_get(self, data: Dict[str, Any], key: str, default: Any = None) -> Any:
        """Safely extracts value from dictionary"""
        try:
            if isinstance(data, dict) and key in data:
                value = data[key]
                if value is None or (isinstance(value, (list, dict)) and not value):
                    return default
                return value
            return default
        except Exception:
            return default

    def _normalize_score(self, score: Union[float, Dict[str, float]]) -> Dict[str, float]:
        """Normalizes score to dictionary format"""
        try:
            if isinstance(score, dict):
                score_value = self._safe_get(score, 'score', 0.6)
                confidence = self._safe_get(score, 'confidence', 0.8)
                return {
                    'score': float(score_value),
                    'confidence': float(confidence)
                }
            elif isinstance(score, (float, int)):
                return {
                    'score': float(score),
                    'confidence': 0.8
                }
            return {'score': 0.6, 'confidence': 0.7}
        except Exception:
            return {'score': 0.6, 'confidence': 0.7}

    def _build_analysis_prompt(self, content: str, source: str) -> str:
        """Builds an improved prompt for deep article analysis"""
        try:
            if not content:
                content = "No content provided"
            if not source:
                source = "Unknown source"
            return f"""Perform a comprehensive analysis of the following article content with a focus on critical thinking and credibility assessment.

Article Source: {source}
Article Content: {content[:10000]}  # Using first 10,000 characters

Please provide a detailed analysis in JSON format with the following structure:

{{
    "credibility_assessment": {{
        "score": <float between 0 and 1>,
        "explanation": "<detailed explanation of the credibility assessment>",
        "supporting_evidence": [
            "<evidence supporting the credibility score>",
            "<additional evidence>"
        ],
        "contradictory_evidence": [
            "<any evidence that might contradict the assessment>",
            "<additional contradictory points>"
        ]
    }},
    "sentiment_analysis": {{
        "score": <float between -1 and 1>,
        "explanation": "<detailed explanation of the sentiment analysis>",
        "emotional_tones": [
            {{
                "tone": "<emotional tone identified>",
                "intensity": <float between 0 and 1>,
                "evidence": "<text supporting this tone>"
            }}
        ]
    }},
    "bias_analysis": {{
        "level": <float between 0 and 1>,
        "types": [
            {{
                "type": "<type of bias identified>",
                "intensity": <float between 0 and 1>,
                "evidence": "<text supporting this bias identification>"
            }}
        ]
    }},
    "content_analysis": {{
        "main_topics": [
            {{
                "topic": "<main topic identified>",
                "relevance": <float between 0 and 1>,
                "key_points": [
                    "<key point about this topic>",
                    "<additional key points>"
                ]
            }}
        ],
        "key_arguments": [
            {{
                "argument": "<key argument presented>",
                "supporting_evidence": [
                    "<evidence supporting this argument>",
                    "<additional evidence>"
                ],
                "counter_arguments": [
                    "<potential counter arguments>",
                    "<additional counter arguments>"
                ]
            }}
        ],
        "unanswered_questions": [
            "<important questions left unanswered>",
            "<additional unanswered questions>"
        ],
        "suggested_followup_questions": [
            "<questions readers should ask themselves>",
            "<additional follow-up questions>"
        ]
    }},
    "perspective_analysis": {{
        "presented_perspectives": [
            {{
                "perspective": "<perspective presented in the article>",
                "supporting_evidence": [
                    "<evidence supporting this perspective>",
                    "<additional evidence>"
                ]
            }}
        ],
        "missing_perspectives": [
            {{
                "perspective": "<important perspective missing from the article>",
                "potential_impact": "<how this missing perspective might affect understanding>"
            }}
        ]
    }},
    "similar_articles_query": "<suggested search query for finding similar articles>"
}}
"""
        except Exception as e:
            logger.error(f"Error building analysis prompt: {str(e)}")
            return f"Analyze the following article content from {source}: {content[:1000]}"

    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """Parses the improved response from API"""
        try:
            if not response_text:
                return self._create_fallback_analysis("Empty response from API")

            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                try:
                    parsed = json.loads(json_match.group(0))
                    return self._normalize_analysis(parsed)
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error: {str(e)}")
                    return self._create_fallback_analysis_from_text(response_text)

            return self._create_fallback_analysis_from_text(response_text)
        except Exception as e:
            logger.error(f"Error parsing response: {str(e)}")
            return self._create_fallback_analysis("Response parsing error")

    def _create_fallback_analysis_from_text(self, text: str) -> Dict[str, Any]:
        """Creates structured analysis from text"""
        try:
            topics = self._extract_topics_from_content(text)
            return {
                "credibility_assessment": {
                    "score": 0.6,
                    "explanation": "Automated credibility assessment based on text content",
                    "supporting_evidence": [],
                    "contradictory_evidence": []
                },
                "sentiment_analysis": {
                    "score": 0.0,
                    "explanation": "Automated sentiment analysis",
                    "emotional_tones": []
                },
                "bias_analysis": {
                    "level": 0.3,
                    "types": []
                },
                "content_analysis": {
                    "main_topics": topics,
                    "key_arguments": [],
                    "unanswered_questions": [],
                    "suggested_followup_questions": []
                },
                "perspective_analysis": {
                    "presented_perspectives": [],
                    "missing_perspectives": []
                },
                "similar_articles_query": "technology news",
                "fallback": True
            }
        except Exception:
            return self._create_fallback_analysis("Text analysis error")

    def _extract_topics_from_content(self, content: str) -> List[Dict[str, Any]]:
        """Extracts topics from text"""
        try:
            if not content:
                return self.fallback_topics[:2]

            content_lower = content.lower()
            topic_keywords = {
                'politics': ['politic', 'government', 'election', 'law', 'policy'],
                'economy': ['economy', 'market', 'finance', 'trade', 'tax'],
                'technology': ['tech', 'software', 'ai', 'computer', 'internet'],
                'science': ['science', 'research', 'study', 'experiment'],
                'health': ['health', 'medical', 'disease', 'vaccine']
            }

            detected_topics = []
            for topic, keywords in topic_keywords.items():
                if any(kw in content_lower for kw in keywords):
                    detected_topics.append({
                        "name": topic,
                        "relevance": 0.8,
                        "description": f"Content related to {topic}"
                    })

            if not detected_topics:
                return self.fallback_topics[:2]

            return detected_topics[:3]
        except Exception:
            return self.fallback_topics[:2]

    def _create_fallback_analysis(self, error: str) -> Dict[str, Any]:
        """Creates fallback analysis when errors occur"""
        return {
            "credibility_assessment": {
                "score": 0.6,
                "explanation": f"Fallback analysis due to error: {error}",
                "supporting_evidence": [],
                "contradictory_evidence": []
            },
            "sentiment_analysis": {
                "score": 0.0,
                "explanation": "Fallback sentiment analysis",
                "emotional_tones": []
            },
            "bias_analysis": {
                "level": 0.3,
                "types": []
            },
            "content_analysis": {
                "main_topics": self.fallback_topics[:2],
                "key_arguments": [],
                "unanswered_questions": [],
                "suggested_followup_questions": []
            },
            "perspective_analysis": {
                "presented_perspectives": [],
                "missing_perspectives": []
            },
            "similar_articles_query": "technology news",
            "error": error,
            "fallback": True
        }

    def analyze_article(self, content: str, source: str) -> Dict[str, Any]:
        """Analyzes article with retries"""
        try:
            # Check cache
            cache_key = f"{source[:50]}:{hash(content) % 10000}"
            cached = self.cache.get_cached_article_analysis(cache_key)
            if cached:
                return cached

            # Prepare content
            prompt_content = self._prepare_content_for_analysis(content)
            prompt = self._build_analysis_prompt(prompt_content, source)

            # Make request with retries
            try:
                if self.client:
                    response = self._make_api_request_with_retry(
                        method="messages.create",
                        model="claude-3-opus-20240229",
                        max_tokens=4000,
                        temperature=0.5,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    analysis = self._parse_response(response.content[0].text)
                else:
                    analysis = self._create_fallback_analysis("API not initialized")
            except Exception as e:
                logger.error(f"Error analyzing article: {str(e)}")
                analysis = self._create_fallback_analysis(str(e))

            # Get similar articles
            similar_articles = self._get_similar_articles(analysis, content)

            # Build result
            result = {
                "title": "Article Analysis",
                "source": source,
                "short_summary": content[:200] + '...' if len(content) > 200 else content,
                "analysis": analysis,
                "similar_articles": similar_articles,
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "source_credibility": self.determine_credibility_level_from_source(source)
                }
            }

            # Cache result
            self.cache.cache_article_analysis(cache_key, result)
            return result
        except Exception as e:
            logger.error(f"Error in analyze_article: {str(e)}")
            return self._create_fallback_analysis(str(e))

    def _get_similar_articles(self, analysis: Dict[str, Any], content: str) -> List[Dict[str, Any]]:
        """Gets similar articles"""
        try:
            query = analysis.get('similar_articles_query', '')
            if not query:
                topics = analysis.get('content_analysis', {}).get('main_topics', [])
                topic_names = [t.get('name', '') for t in topics[:3] if isinstance(t, dict)]
                query = ' OR '.join(topic_names) if topic_names else "technology"
            return self.news_api.get_everything(query=query, page_size=3)
        except Exception:
            return []

    def _prepare_content_for_analysis(self, content: str) -> str:
        """Prepares content for analysis"""
        try:
            max_length = 10000
            if not content:
                return "No content available"
            if len(content) <= max_length:
                return content

            preview_length = 200
            middle_part = content[preview_length:-preview_length] if len(content) > 2*preview_length else ""
            truncated = (
                content[:preview_length] +
                "\n\n[CONTENT TRIMMED FOR ANALYSIS]\n\n" +
                middle_part[:max_length - 2*preview_length - 50] +
                "\n\n..." +
                content[-preview_length:]
            )
            return truncated[:max_length]
        except Exception:
            return content[:1000] if content else "No content available"

    def determine_credibility_level_from_source(self, source_name: str) -> str:
        """Determines credibility level from source"""
        try:
            if not source_name:
                return "Medium"

            source_name = source_name.lower()
            high_sources = ['bbc', 'reuters', 'nytimes', 'theguardian']
            medium_sources = ['cnn', 'fox', 'washingtonpost']
            low_sources = ['dailymail', 'breitbart']

            if any(s in source_name for s in high_sources):
                return "High"
            elif any(s in source_name for s in medium_sources):
                return "Medium"
            elif any(s in source_name for s in low_sources):
                return "Low"

            if source_name.startswith(('http://', 'https://')):
                domain = urlparse(source_name).netloc
                if domain.endswith('.gov') or domain.endswith('.edu'):
                    return "High"

            return "Medium"
        except Exception:
            return "Medium"
