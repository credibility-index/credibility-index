import os
import logging
import json
from typing import Dict, Any, Optional, List, Union
import anthropic
from pydantic import BaseModel
from datetime import datetime
import re
from cache import CacheManager
from news_api import NewsAPI  # Убедитесь, что этот модуль существует и правильно импортируется

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ArticleAnalysis(BaseModel):
    credibility_score: Dict[str, Union[float, Dict[str, float]]]
    sentiment: Dict[str, Union[float, Dict[str, float]]]
    bias: Dict[str, Union[float, Dict[str, float]]]
    topics: List[Dict[str, Any]]
    perspectives: Dict[str, Dict[str, Any]]
    key_arguments: List[str]

class ClaudeAPI:
    def __init__(self):
        self.cache = CacheManager()
        self.news_api = NewsAPI()  # Инициализируем NewsAPI
        self.client = self._initialize_anthropic_client()

    def _initialize_anthropic_client(self):
        """Инициализирует клиент Anthropic с обработкой ошибок"""
        try:
            api_key = os.getenv('ANTHROPIC_API_KEY')
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY environment variable not set")
            return anthropic.Anthropic(api_key=api_key)
        except Exception as e:
            logger.error(f"Failed to initialize Anthropic client: {str(e)}")
            return None

    def _safe_get(self, data: Dict[str, Any], key: str, default: Any = None) -> Any:
        """Безопасное извлечение значения из словаря"""
        try:
            if isinstance(data, dict) and key in data:
                value = data[key]
                if isinstance(value, (float, int)) and not isinstance(default, (float, int)):
                    return default
                return value
            return default
        except Exception:
            return default

    def _normalize_score(self, score: Union[float, Dict[str, float]]) -> Dict[str, float]:
        """Нормализует оценку в формат словаря"""
        if isinstance(score, dict):
            return score
        if isinstance(score, (float, int)):
            return {'score': float(score)}
        return {'score': 0.6}  # Значение по умолчанию

    def get_buzz_analysis(self) -> Dict[str, Any]:
        """Получает анализ популярных новостей с обработкой ошибок"""
        try:
            cached = self.cache.get_cached_buzz_analysis()
            if cached:
                return cached

            prompt = """Analyze current global news trends focusing on:
1. Key geopolitical developments
2. Economic indicators
3. Technological advancements
4. Social trends

Provide a comprehensive analysis with:
- Credibility assessment (0-1 scale)
- Multiple perspectives
- Key arguments
- Potential biases
- Future implications

Format the response as JSON with the following structure:
{
    "title": "string",
    "source": "string",
    "short_summary": "string",
    "analysis": {
        "credibility_score": {"score": number},
        "sentiment": {"score": number},
        "bias": {"level": number},
        "topics": [{"name": "string", "relevance": number}],
        "perspectives": {
            "western": {"summary": "string", "credibility": "string"},
            "eastern": {"summary": "string", "credibility": "string"},
            "neutral": {"summary": "string", "credibility": "string"}
        },
        "key_arguments": ["string"]
    }
}"""

            try:
                response = self.client.messages.create(
                    model="claude-3-opus-20240229",
                    max_tokens=2000,
                    temperature=0.7,
                    system="You are a professional news analyst with expertise in media credibility assessment.",
                    messages=[{"role": "user", "content": prompt}]
                )

                analysis_text = response.content[0].text
                analysis = json.loads(analysis_text)

                # Нормализуем структуру данных
                analysis['credibility_score'] = self._normalize_score(analysis.get('credibility_score', 0.6))
                analysis['sentiment'] = self._normalize_score(analysis.get('sentiment', 0.1))
                analysis['bias'] = self._normalize_score(analysis.get('bias', 0.2))

                self.cache.cache_buzz_analysis(analysis)
                return analysis
            except json.JSONDecodeError:
                logger.error("Failed to parse analysis response as JSON")
                return {
                    "article": {
                        "title": "Today's featured analysis: Global News Trends",
                        "source": "Media Analysis",
                        "short_summary": "Analysis of current global news trends and their credibility patterns.",
                        "analysis": {
                            "credibility_score": {"score": 0.85},
                            "sentiment": {"score": 0.1},
                            "bias": {"level": 0.2},
                            "topics": [
                                {"name": "Geopolitics", "relevance": 0.9},
                                {"name": "Economy", "relevance": 0.8}
                            ],
                            "perspectives": {
                                "western": {
                                    "summary": "Western perspective on current events",
                                    "credibility": "High"
                                }
                            },
                            "key_arguments": [
                                "Global economic growth continues",
                                "Technological advancements accelerating"
                            ]
                        }
                    }
                }
            except Exception as e:
                logger.error(f"Error getting buzz analysis: {str(e)}")
                return {
                    "article": {
                        "title": "Today's featured analysis: Global News Trends",
                        "source": "Media Analysis",
                        "short_summary": "Analysis of current global news trends and their credibility patterns.",
                        "analysis": {
                            "credibility_score": {"score": 0.85},
                            "sentiment": {"score": 0.1},
                            "bias": {"level": 0.2},
                            "topics": [
                                {"name": "Geopolitics"},
                                {"name": "Economy"}
                            ],
                            "perspectives": {
                                "neutral": {
                                    "summary": "Basic analysis of current events",
                                    "credibility": "Medium"
                                }
                            },
                            "key_arguments": [
                                "Global economic trends",
                                "Technological developments"
                            ]
                        }
                    }
                }
        except Exception as e:
            logger.error(f"Unexpected error in get_buzz_analysis: {str(e)}")
            return {
                "article": {
                    "title": "Error loading analysis",
                    "source": "System",
                    "short_summary": "Failed to load current news analysis",
                    "analysis": {
                        "credibility_score": {"score": 0.6},
                        "sentiment": {"score": 0.0},
                        "bias": {"level": 0.3},
                        "topics": [],
                        "perspectives": {},
                        "key_arguments": [
                            "System error occurred"
                        ]
                    }
                }
            }

    def analyze_article(self, content: str, source: str) -> Dict[str, Any]:
        """Анализирует статью с использованием Claude API"""
        try:
            # Проверяем кэш
            cached = self.cache.get_cached_article_analysis(content)
            if cached:
                return cached

            prompt = f"""Analyze the following article content with these requirements:
1. Assess credibility (0-1 scale)
2. Determine sentiment (-1 to 1 scale)
3. Identify potential biases (0-1 scale)
4. Extract key topics
5. Provide multiple perspectives
6. Identify key arguments

Article source: {source}
Article content: {content[:2000]}...

Format the response as JSON with this structure:
{{
    "credibility_score": {{"score": number}},
    "sentiment": {{"score": number}},
    "bias": {{"level": number}},
    "topics": [{{"name": "string"}}],
    "perspectives": {{
        "western": {{"summary": "string", "credibility": "string"}},
        "eastern": {{"summary": "string", "credibility": "string"}},
        "neutral": {{"summary": "string", "credibility": "string"}}
    }},
    "key_arguments": ["string"]
}}"""

            try:
                response = self.client.messages.create(
                    model="claude-3-opus-20240229",
                    max_tokens=2000,
                    temperature=0.5,
                    system="You are a professional media analyst. Provide a detailed, structured analysis of the article content.",
                    messages=[{"role": "user", "content": prompt}]
                )

                analysis_text = response.content[0].text
                analysis = json.loads(analysis_text)

                # Нормализуем структуру данных
                analysis['credibility_score'] = self._normalize_score(analysis.get('credibility_score', 0.6))
                analysis['sentiment'] = self._normalize_score(analysis.get('sentiment', 0.1))
                analysis['bias'] = self._normalize_score(analysis.get('bias', 0.2))

                # Получаем похожие статьи
                topics = [t['name'] if isinstance(t, dict) else t for t in analysis.get('topics', [])]
                similar_articles = []
                if topics and self.news_api:
                    try:
                        query = ' OR '.join([str(t) for t in topics[:3]])
                        similar_articles = self.news_api.get_everything(query=query, page_size=3) or []
                    except Exception as e:
                        logger.error(f"Error getting similar articles: {str(e)}")
                        similar_articles = []

                # Формируем результат
                result = {
                    "credibility_score": analysis['credibility_score'],
                    "sentiment": analysis['sentiment'],
                    "bias": analysis['bias'],
                    "topics": analysis.get('topics', []),
                    "perspectives": analysis.get('perspectives', {}),
                    "key_arguments": analysis.get('key_arguments', []),
                    "similar_articles": similar_articles
                }

                self.cache.cache_article_analysis(content, result)
                return result
            except json.JSONDecodeError:
                logger.error("Failed to parse analysis response as JSON")
                return {
                    "credibility_score": {"score": 0.7},
                    "sentiment": {"score": 0.1},
                    "bias": {"level": 0.2},
                    "topics": [{"name": "General"}],
                    "perspectives": {
                        "neutral": {
                            "summary": "Basic analysis of the provided content",
                            "credibility": "Medium"
                        }
                    },
                    "key_arguments": ["Content requires more context for detailed analysis"],
                    "similar_articles": []
                }
            except Exception as e:
                logger.error(f"Error analyzing article: {str(e)}")
                return {
                    "credibility_score": {"score": 0.6},
                    "sentiment": {"score": 0.0},
                    "bias": {"level": 0.3},
                    "topics": [{"name": "General"}],
                    "perspectives": {
                        "neutral": {
                            "summary": "Error occurred during analysis",
                            "credibility": "Low"
                        }
                    },
                    "key_arguments": ["Analysis failed due to technical issues"],
                    "similar_articles": []
                }
        except Exception as e:
            logger.error(f"Unexpected error in analyze_article: {str(e)}")
            return {
                "credibility_score": {"score": 0.5},
                "sentiment": {"score": 0.0},
                "bias": {"level": 0.4},
                "topics": [],
                "perspectives": {},
                "key_arguments": ["System error occurred during analysis"],
                "similar_articles": []
            }

    def determine_credibility_level(self, score: Any) -> str:
        """Определяет уровень достоверности на основе оценки"""
        try:
            if isinstance(score, dict):
                score_value = self._safe_get(score, 'score', 0.6)
            elif isinstance(score, (float, int)):
                score_value = float(score)
            else:
                score_value = 0.6

            if score_value >= 0.8:
                return "High"
            elif score_value >= 0.6:
                return "Medium"
            else:
                return "Low"
        except Exception as e:
            logger.error(f"Error determining credibility level: {str(e)}")
            return "Medium"

    def get_similar_articles(self, topics: List[str]) -> List[Dict[str, Any]]:
        """Получает похожие статьи на основе тем"""
        if not topics or not self.news_api:
            return []

        try:
            query = ' OR '.join([str(t) for t in topics[:3]])
            articles = self.news_api.get_everything(query=query, page_size=3) or []

            return [
                {
                    "title": article.get('title', 'No title'),
                    "source": article.get('source', {}).get('name', 'Unknown'),
                    "url": article.get('url', '#'),
                    "summary": article.get('description', 'No summary available'),
                    "credibility": self.determine_credibility_level_from_source(
                        article.get('source', {}).get('name', 'Unknown')
                    )
                }
                for article in articles
            ]
        except Exception as e:
            logger.error(f"Error getting similar articles: {str(e)}")
            return []
