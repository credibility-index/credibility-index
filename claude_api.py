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
from news_api import NewsAPI

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ArticleAnalysis(BaseModel):
    """Модель для результатов анализа статьи"""
    credibility_score: Dict[str, Union[float, Dict[str, float]]]
    sentiment: Dict[str, Union[float, Dict[str, float]]]
    bias: Dict[str, Union[float, Dict[str, float]]]
    topics: List[Dict[str, Any]]
    perspectives: Dict[str, Dict[str, Any]]
    key_arguments: List[str]

class CredibilityIndex:
    """Класс для расчета индекса достоверности"""
    def __init__(self):
        self.source_weights = {
            'high': 0.3,
            'medium': 0.2,
            'low': 0.1
        }
        self.content_weights = {
            'evidence': 0.25,
            'logical_consistency': 0.2,
            'completeness': 0.15,
            'bias': 0.1
        }

    def calculate_index(self, source_credibility: str, analysis: Dict[str, Any]) -> float:
        """Рассчитывает индекс достоверности на основе анализа и источника"""
        try:
            # Базовый индекс на основе источника
            base_index = self.source_weights.get(source_credibility.lower(), 0.15)

            # Добавляем оценки из анализа контента
            content_score = 0

            # Оценка достоверности
            credibility = analysis.get('credibility', {})
            content_score += credibility.get('score', 0.5) * self.content_weights['evidence']

            # Логическая согласованность
            content_score += 0.7 * self.content_weights['logical_consistency']  # По умолчанию 0.7

            # Полнота
            content_score += 0.6 * self.content_weights['completeness']  # По умолчанию 0.6

            # Смещение
            bias_level = analysis.get('bias', {}).get('level', 0.3)
            content_score += (1 - bias_level) * self.content_weights['bias']

            # Общий индекс достоверности
            credibility_index = base_index + content_score

            # Нормализуем до диапазона 0-1
            return min(1.0, max(0.0, credibility_index))

        except Exception as e:
            logger.error(f"Error calculating credibility index: {str(e)}")
            return 0.5  # Значение по умолчанию при ошибке

class ClaudeAPI:
    """API клиент для работы с моделью Claude от Anthropic"""
    def __init__(self):
        self.cache = CacheManager()
        self.news_api = NewsAPI()
        self.client = self._initialize_anthropic_client()
        self.credibility_index = CredibilityIndex()
        self.max_retries = 3
        self.retry_delay = 1
        self.fallback_topics = [
            {"name": "General News", "relevance": 0.8, "description": "General news"},
            {"name": "Politics", "relevance": 0.7, "description": "Political news"},
            {"name": "Economy", "relevance": 0.6, "description": "Economic news"}
        ]

    def _initialize_anthropic_client(self) -> Optional[anthropic.Anthropic]:
        """Инициализация клиента Anthropic с обработкой ошибок"""
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
        """Выполняет API запрос с повторными попытками"""
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
                last_error = e
                continue

        logger.error(f"All retries exhausted. Last error: {str(last_error)}")
        raise Exception(f"All retries exhausted. Last error: {str(last_error)}")

    def _build_analysis_prompt(self, content: str, source: str) -> str:
        """Создает упрощенный промпт для анализа статьи"""
        try:
            if not content:
                content = "No content provided"
            if not source:
                source = "Unknown source"

            content_preview = self._prepare_content_for_analysis(content)

            return f"""Analyze the following article content with focus on credibility assessment:

Article Source: {source}
Article Content Preview: {content_preview}

Provide analysis in JSON format with this structure:
{{
    "credibility": {{
        "score": <number from 0 to 1>,
        "explanation": "<brief explanation of credibility assessment>"
    }},
    "topics": [
        {{
            "name": "<main topic>",
            "relevance": <number from 0 to 1>
        }}
    ],
    "key_points": ["<main points of the article>"],
    "potential_biases": ["<potential biases identified>"],
    "similar_articles_query": "<suggested search query for similar articles>"
}}

Consider these factors for credibility assessment:
1. Source reputation and reliability
2. Evidence quality and support for claims
3. Logical consistency of arguments
4. Completeness of coverage
5. Potential conflicts of interest"""
        except Exception as e:
            logger.error(f"Error building analysis prompt: {str(e)}")
            return f"Analyze this article from {source}: {content[:1000]}"

    def _prepare_content_for_analysis(self, content: str) -> str:
        """Подготавливает контент для анализа, создавая превью"""
        try:
            max_length = 5000
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

    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """Парсит ответ от API"""
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
        """Создает структурированный анализ из текста"""
        try:
            topics = self._extract_topics_from_content(text)
            return {
                "credibility": {
                    "score": 0.6,
                    "explanation": "Automated credibility assessment based on text content"
                },
                "topics": topics,
                "key_points": ["Main points could not be determined"],
                "potential_biases": ["No biases identified"],
                "similar_articles_query": "general news"
            }
        except Exception:
            return self._create_fallback_analysis("Text analysis error")

    def _extract_topics_from_content(self, content: str) -> List[Dict[str, Any]]:
        """Извлекает темы из текста"""
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
        """Создает резервный анализ при возникновении ошибок"""
        return {
            "credibility": {
                "score": 0.6,
                "explanation": f"Fallback analysis due to error: {error}"
            },
            "topics": self.fallback_topics[:2],
            "key_points": [],
            "potential_biases": [],
            "similar_articles_query": "general news",
            "error": error
        }

    def _normalize_analysis(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Нормализует анализ"""
        try:
            if 'credibility' not in analysis:
                analysis['credibility'] = {
                    'score': 0.6,
                    'explanation': 'Normalized credibility score'
                }

            if 'topics' not in analysis:
                analysis['topics'] = self.fallback_topics[:2]

            return analysis
        except Exception:
            return self._create_fallback_analysis("Analysis normalization error")

    def determine_credibility_level_from_source(self, source_name: str) -> str:
        """Определяет уровень достоверности источника"""
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

    def analyze_article(self, content: str, source: str) -> Dict[str, Any]:
        """Анализирует статью с повторными попытками"""
        try:
            # Проверяем кэш
            cache_key = f"{source[:50]}:{hash(content) % 10000}"
            cached = self.cache.get_cached_article_analysis(cache_key)
            if cached:
                return cached

            # Подготавливаем контент
            prompt = self._build_analysis_prompt(content, source)

            # Делаем запрос с повторными попытками
            try:
                if self.client:
                    response = self._make_api_request_with_retry(
                        method="messages.create",
                        model="claude-3-opus-20240229",
                        max_tokens=2000,
                        temperature=0.5,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    analysis = self._parse_response(response.content[0].text)
                else:
                    analysis = self._create_fallback_analysis("API not initialized")
            except Exception as e:
                logger.error(f"Error analyzing article: {str(e)}")
                analysis = self._create_fallback_analysis(str(e))

            # Рассчитываем индекс достоверности
            source_credibility = self.determine_credibility_level_from_source(source)
            credibility_index = self.credibility_index.calculate_index(source_credibility, analysis)

            # Добавляем индекс достоверности в результат
            analysis['credibility_index'] = {
                'score': credibility_index,
                'source_contribution': self.source_weights.get(source_credibility.lower(), 0.15),
                'content_contribution': credibility_index - self.source_weights.get(source_credibility.lower(), 0.15)
            }

            # Получаем похожие статьи
            similar_articles = self._get_similar_articles(analysis, content)

            # Строим результат
            result = {
                "title": "Article Analysis",
                "source": source,
                "short_summary": content[:200] + '...' if len(content) > 200 else content,
                "analysis": analysis,
                "similar_articles": similar_articles,
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "source_credibility": source_credibility,
                    "credibility_index": credibility_index
                }
            }

            # Кэшируем результат
            self.cache.cache_article_analysis(cache_key, result)
            return result

        except Exception as e:
            logger.error(f"Error in analyze_article: {str(e)}")
            return self._create_fallback_analysis(str(e))

    def _get_similar_articles(self, analysis: Dict[str, Any], content: str) -> List[Dict[str, Any]]:
        """Получает похожие статьи"""
        try:
            query = analysis.get('similar_articles_query', '')
            if not query:
                topics = analysis.get('topics', [])
                topic_names = [t.get('name', '') for t in topics[:3] if isinstance(t, dict)]
                query = ' OR '.join(topic_names) if topic_names else "general news"

            return self.news_api.get_everything(query=query, page_size=3)
        except Exception:
            return []
