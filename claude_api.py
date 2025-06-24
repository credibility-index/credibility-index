import os
import logging
import json
import time
import re
import requests
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta
from urllib.parse import urlparse
import anthropic
from pydantic import BaseModel
from cache import CacheManager

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

class EnhancedNewsAPI:
    """Улучшенная версия NewsAPI с надежной обработкой ошибок"""
    def __init__(self):
        self.api_key = os.getenv('NEWS_API_KEY', 'mock-api-key')
        self.base_url = "https://newsapi.org/v2"
        self.fallback_articles = [
            {
                "title": "Технологические новости (резерв)",
                "source": {"name": "Tech Demo News", "url": None},
                "url": None,
                "description": "Резервные данные о технологических трендах",
                "publishedAt": datetime.now().isoformat(),
                "content": "Это резервная статья, показываемая при временных проблемах с API."
            },
            {
                "title": "Экономический обзор (резерв)",
                "source": {"name": "Econ Demo", "url": None},
                "url": None,
                "description": "Резервные данные об экономических показателях",
                "publishedAt": datetime.now().isoformat(),
                "content": "Это резервная статья, показываемая при временных проблемах с API."
            }
        ]
        self.max_retries = 3
        self.retry_delay = 1

    def get_everything(self, query: str, page_size: int = 5, **kwargs) -> List[Dict[str, Any]]:
        """Получает статьи с обработкой ошибок и повторными попытками"""
        retry_count = 0
        last_error = None
        while retry_count < self.max_retries:
            try:
                # Подготавливаем параметры запроса
                params = {
                    'q': query,
                    'pageSize': page_size,
                    'apiKey': self.api_key,
                    **kwargs
                }
                # Используем правильный endpoint
                url = f"{self.base_url}/everything"
                # Делаем запрос с таймаутом
                response = requests.get(
                    url,
                    params=params,
                    timeout=15
                )
                # Обрабатываем ответ
                if response.status_code == 200:
                    data = response.json()
                    if data.get('status') == 'ok':
                        articles = data.get('articles', [])
                        processed_articles = []
                        for article in articles:
                            try:
                                processed = {
                                    'title': article.get('title', 'Untitled Article'),
                                    'description': article.get('description', 'No description available'),
                                    'url': article.get('url', '#'),
                                    'source': {
                                        'name': article.get('source', {}).get('name', 'Unknown Source'),
                                        'url': article.get('source', {}).get('url', None)
                                    },
                                    'publishedAt': article.get('publishedAt', datetime.now().isoformat()),
                                    'content': article.get('content', 'Full content not available')
                                }
                                processed_articles.append(processed)
                            except Exception as e:
                                logger.error(f"Ошибка обработки статьи: {str(e)}")
                                continue
                        if processed_articles:
                            return processed_articles[:page_size]
                        return self._get_fallback_articles(query, page_size)
                    logger.warning(f"NewsAPI вернул не-ok статус: {data.get('message', 'No message')}")
                    return self._get_fallback_articles(query, page_size)
                elif response.status_code == 404:
                    logger.error("NewsAPI вернул ошибку 404 - неверный URL или ресурс не найден")
                    return self._get_fallback_articles(query, page_size)
                elif response.status_code in (500, 502, 503, 504):
                    retry_count += 1
                    wait_time = self.retry_delay * (2 ** (retry_count - 1))
                    logger.warning(f"Серверная ошибка. Повторная попытка через {wait_time} секунд")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"NewsAPI вернул ошибку {response.status_code}")
                    return self._get_fallback_articles(query, page_size)
            except requests.exceptions.RequestException as e:
                retry_count += 1
                wait_time = self.retry_delay * (2 ** (retry_count - 1))
                logger.warning(f"Ошибка соединения. Повторная попытка через {wait_time} секунд")
                time.sleep(wait_time)
                continue
            except Exception as e:
                logger.error(f"Неожиданная ошибка: {str(e)}")
                return self._get_fallback_articles(query, page_size)
        logger.error(f"Исчерпаны все попытки. Последняя ошибка: {str(last_error)}")
        return self._get_fallback_articles(query, page_size)

    def _get_fallback_articles(self, query: str, count: int) -> List[Dict[str, Any]]:
        """Возвращает резервные статьи при ошибках API"""
        mock_articles = [
            {
                "title": f"Статья о {query} (резерв)",
                "source": {"name": f"{query.capitalize()} News"},
                "url": f"https://example.com/{query.replace(' ', '-')}-1",
                "description": f"Резервная статья о {query}",
                "publishedAt": datetime.now().isoformat(),
                "content": f"Это резервная статья о {query}. В реальной системе здесь были бы настоящие новости."
            },
            {
                "title": f"Анализ {query} (резерв)",
                "source": {"name": f"{query.capitalize()} Analysis"},
                "url": f"https://example.com/{query.replace(' ', '-')}-2",
                "description": f"Резервный анализ {query}",
                "publishedAt": datetime.now().isoformat(),
                "content": f"Это резервная аналитическая статья о {query}. В реальной системе здесь был бы профессиональный анализ."
            }
        ]
        return mock_articles[:count]

class ClaudeAPI:
    def __init__(self):
        self.cache = CacheManager()
        self.news_api = EnhancedNewsAPI()
        self.client = self._initialize_anthropic_client()
        self.max_retries = 3
        self.retry_delay = 1
        self.fallback_topics = [
            {"name": "General News", "relevance": 0.8, "description": "Общие новости"},
            {"name": "Politics", "relevance": 0.7, "description": "Политические новости"},
            {"name": "Economy", "relevance": 0.6, "description": "Экономические новости"}
        ]

    def _initialize_anthropic_client(self) -> Optional[anthropic.Anthropic]:
        """Инициализирует клиент Anthropic с обработкой ошибок"""
        try:
            api_key = os.getenv('ANTHROPIC_API_KEY')
            if not api_key:
                logger.warning("ANTHROPIC_API_KEY не установлен, будет использоваться mock-режим")
                return None
            return anthropic.Anthropic(api_key=api_key)
        except Exception as e:
            logger.error(f"Ошибка инициализации клиента: {str(e)}")
            return None

    def _make_api_request_with_retry(self, method: str, **kwargs) -> Any:
        """Выполняет API запрос с повторными попытками"""
        retry_count = 0
        last_error = None
        while retry_count < self.max_retries:
            try:
                if not self.client:
                    raise Exception("Anthropic client не инициализирован")
                if method == "messages.create":
                    response = self.client.messages.create(**kwargs)
                    return response
                else:
                    raise ValueError(f"Неизвестный метод API: {method}")
            except anthropic.APIStatusError as e:
                status_code = e.status_code
                if status_code in (529, 429, 500, 502, 503, 504):
                    retry_after = int(e.response.headers.get('Retry-After', 5))
                    wait_time = min(retry_after, self.retry_delay * (2 ** retry_count))
                    logger.warning(f"API вернул код {status_code}. Повтор через {wait_time} секунд")
                    time.sleep(wait_time)
                    retry_count += 1
                    continue
                raise
            except Exception as e:
                retry_count += 1
                wait_time = self.retry_delay * (2 ** (retry_count - 1))
                logger.warning(f"Ошибка API. Повтор через {wait_time} секунд")
                time.sleep(wait_time)
                continue
        logger.error(f"Исчерпаны все попытки. Последняя ошибка: {str(last_error)}")
        raise Exception(f"Исчерпаны все попытки. Последняя ошибка: {str(last_error)}")

    def _safe_get(self, data: Dict[str, Any], key: str, default: Any = None) -> Any:
        """Безопасное извлечение значения из словаря"""
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
        """Нормализует оценку в формат словаря"""
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

    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """Парсит ответ от API с улучшенной обработкой ошибок"""
        try:
            if not response_text:
                return self._create_fallback_analysis("Пустой ответ от API")
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                try:
                    parsed = json.loads(json_match.group(0))
                    return self._normalize_analysis(parsed)
                except json.JSONDecodeError:
                    pass
            return self._create_fallback_analysis_from_text(response_text)
        except Exception as e:
            logger.error(f"Ошибка парсинга ответа: {str(e)}")
            return self._create_fallback_analysis("Ошибка парсинга ответа")

    def _create_fallback_analysis_from_text(self, text: str) -> Dict[str, Any]:
        """Создает структурированный анализ на основе текста"""
        try:
            topics = self._extract_topics_from_content(text)
            return {
                "credibility_score": {"score": 0.6, "confidence": 0.7},
                "sentiment": {"score": 0.0, "confidence": 0.7},
                "bias": {"level": 0.3, "confidence": 0.6},
                "topics": topics,
                "perspectives": {
                    "neutral": {
                        "summary": "Анализ на основе текста",
                        "credibility": "Medium",
                        "supporting_evidence": ["Анализ основан на содержимом статьи"]
                    }
                },
                "key_arguments": [],
                "fallback": True
            }
        except Exception:
            return self._create_fallback_analysis("Ошибка анализа текста")

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
                        "description": f"Контент, связанный с {topic}"
                    })
            if not detected_topics:
                return self.fallback_topics[:2]
            return detected_topics[:3]
        except Exception:
            return self.fallback_topics[:2]

    def _create_fallback_analysis(self, error: str) -> Dict[str, Any]:
        """Создает резервный анализ при ошибках"""
        return {
            "credibility_score": {"score": 0.6, "confidence": 0.7},
            "sentiment": {"score": 0.0, "confidence": 0.7},
            "bias": {"level": 0.3, "confidence": 0.6},
            "topics": self.fallback_topics[:2],
            "perspectives": {
                "neutral": {
                    "summary": f"Резервный анализ: {error}",
                    "credibility": "Medium"
                }
            },
            "key_arguments": [],
            "credibility_index": 0.5,  # значение по умолчанию
            "error": error,
            "fallback": True
        }

    def _calculate_credibility_index(self, analysis: Dict[str, Any]) -> float:
        """
        Рассчитывает индекс достоверности на основе различных факторов.

        Args:
            analysis: Результат анализа статьи

        Returns:
            Индекс достоверности (от 0 до 1)
        """
        try:
            # Получаем оценку достоверности
            credibility_score = self._safe_get(analysis, 'credibility_score', {}).get('score', 0.6)

            # Получаем уровень предвзятости (чем ниже, тем лучше)
            bias_level = self._safe_get(analysis, 'bias', {}).get('level', 0.5)

            # Получаем тональность (нейтральная тональность может быть более достоверной)
            sentiment_score = self._safe_get(analysis, 'sentiment', {}).get('score', 0.0)
            sentiment_factor = 1.0 - abs(sentiment_score)  # нейтральная тональность дает больше веса

            # Рассчитываем индекс достоверности
            # Веса для каждого фактора
            weights = {
                'credibility': 0.6,
                'bias': 0.2,
                'sentiment': 0.1,
                'source': 0.1
            }

            # Нормализуем значения
            normalized_credibility = credibility_score
            normalized_bias = 1.0 - bias_level  # инвертируем, так как меньшая предвзятость лучше
            normalized_sentiment = sentiment_factor

            # Рассчитываем индекс
            credibility_index = (
                weights['credibility'] * normalized_credibility +
                weights['bias'] * normalized_bias +
                weights['sentiment'] * normalized_sentiment
            )

            # Учитываем источник (если он известен)
            source = self._safe_get(analysis, 'source', '')
            source_credibility = self.determine_credibility_level_from_source(source)
            if source_credibility == "High":
                credibility_index += weights['source'] * 0.8
            elif source_credibility == "Medium":
                credibility_index += weights['source'] * 0.5
            else:
                credibility_index += weights['source'] * 0.2

            # Ограничиваем значение от 0 до 1
            credibility_index = max(0.0, min(1.0, credibility_index))

            return round(credibility_index, 2)
        except Exception as e:
            logger.error(f"Ошибка расчета индекса достоверности: {str(e)}")
            return 0.5  # среднее значение по умолчанию

    def _generate_credibility_explanation(self, score: float, source: str) -> Dict[str, Any]:
        """Генерирует объяснение для оценки достоверности"""
        try:
            if isinstance(score, dict):
                score = score.get('score', 0.6)
            score = float(score)
            source_level = self.determine_credibility_level_from_source(source)

            # Рассчитываем индекс достоверности
            credibility_index = self._calculate_credibility_index({
                'credibility_score': {'score': score},
                'bias': {'level': 0.3},  # значение по умолчанию
                'sentiment': {'score': 0.0},  # значение по умолчанию
                'source': source
            })

            if score >= 0.8:
                explanation = f"Высокий рейтинг достоверности ({score:.2f})"
            elif score >= 0.6:
                explanation = f"Средний рейтинг достоверности ({score:.2f})"
            else:
                explanation = f"Низкий рейтинг достоверности ({score:.2f})"

            return {
                "level": self.determine_credibility_level(score),
                "explanation": explanation,
                "score": score,
                "source_level": source_level,
                "credibility_index": credibility_index,
                "index_explanation": f"Индекс достоверности {credibility_index:.2f} (0-1), учитывающий несколько факторов достоверности"
            }
        except Exception:
            return {
                "level": "Medium",
                "explanation": "Не удалось сгенерировать объяснение",
                "score": 0.6,
                "source_level": "Medium",
                "credibility_index": 0.5,
                "index_explanation": "Индекс достоверности не рассчитан из-за ошибки"
            }

    def _normalize_analysis(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Нормализует структуру анализа"""
        try:
            normalized = {
                "credibility_score": self._normalize_score(analysis.get('credibility_score', {'score': 0.6})),
                "sentiment": self._normalize_score(analysis.get('sentiment', {'score': 0.0})),
                "bias": self._normalize_score(analysis.get('bias', {'level': 0.3})),
                "topics": analysis.get('topics', []),
                "perspectives": analysis.get('perspectives', {}),
                "key_arguments": analysis.get('key_arguments', [])
            }

            # Добавляем индекс достоверности
            credibility_index = self._calculate_credibility_index(normalized)
            normalized['credibility_index'] = credibility_index

            # Добавляем объяснения
            if isinstance(normalized['credibility_score'], dict):
                score = normalized['credibility_score']['score']
                explanation = (
                    "Высокий уровень достоверности"
                    if score >= 0.8 else
                    "Средний уровень достоверности"
                    if score >= 0.6 else
                    "Низкий уровень достоверности"
                )
                normalized['credibility_score']['explanation'] = explanation
            return normalized
        except Exception:
            return self._create_fallback_analysis("Ошибка нормализации анализа")

    def _prepare_content_for_analysis(self, content: str) -> str:
        """Подготавливает контент для анализа"""
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

    def _build_analysis_prompt(self, content: str, source: str) -> str:
        """Строит промпт для анализа статьи"""
        try:
            if not content:
                content = "No content provided"
            if not source:
                source = "Unknown source"
            return f"""Проанализируйте следующее содержимое статьи:
Источник: {source}
Содержимое: {content}
Предоставьте анализ в формате JSON со следующей структурой:
{{
    "credibility_score": {{"score": number}},
    "sentiment": {{"score": number}},
    "bias": {{"level": number}},
    "topics": [{{"name": "string"}}],
    "perspectives": {{}},
    "key_arguments": ["string"]
}}"""
        except Exception:
            return f"Анализ статьи из источника: {source}"

    def analyze_article(self, content: str, source: str) -> Dict[str, Any]:
        """Анализирует статью с повторными попытками"""
        try:
            # Проверяем кэш
            cache_key = f"{source[:50]}:{hash(content) % 10000}"
            cached = self.cache.get_cached_article_analysis(cache_key)
            if cached:
                return cached

            # Подготавливаем контент
            prompt_content = self._prepare_content_for_analysis(content)
            prompt = self._build_analysis_prompt(prompt_content, source)

            # Выполняем запрос с повторными попытками
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
                    analysis = self._create_fallback_analysis("API не инициализирован")
            except Exception as e:
                logger.error(f"Ошибка анализа статьи: {str(e)}")
                analysis = self._create_fallback_analysis(str(e))

            # Добавляем объяснение достоверности и индекс достоверности
            credibility_score = analysis.get('credibility_score', {'score': 0.7})
            score = credibility_score.get('score', 0.7)
            credibility_info = self._generate_credibility_explanation(score, source)

            # Обновляем анализ с учетом индекса достоверности
            if 'credibility_index' in credibility_info:
                analysis['credibility_index'] = credibility_info['credibility_index']
                analysis['credibility_score']['index'] = credibility_info['credibility_index']

            # Получаем похожие статьи
            similar_articles = self._get_similar_articles(analysis, content)

            # Формируем результат
            result = {
                "title": "Article Analysis",
                "source": source,
                "short_summary": content[:200] + '...' if len(content) > 200 else content,
                "analysis": analysis,
                "similar_articles": similar_articles,
                "credibility_info": credibility_info,  # Добавляем информацию о достоверности
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "source_credibility": self.determine_credibility_level_from_source(source)
                }
            }

            # Кэшируем результат
            self.cache.cache_article_analysis(cache_key, result)
            return result
        except Exception as e:
            logger.error(f"Ошибка в analyze_article: {str(e)}")
            return self._create_fallback_analysis(str(e))

    def _get_similar_articles(self, analysis: Dict[str, Any], content: str) -> List[Dict[str, Any]]:
        """Получает похожие статьи"""
        try:
            topics = analysis.get('topics', [])
            topic_names = [t.get('name', '') if isinstance(t, dict) else str(t) for t in topics[:3]]
            query = ' OR '.join(topic_names) if topic_names else "technology"
            return self.news_api.get_everything(query=query, page_size=3)
        except Exception:
            return []

    def get_buzz_analysis(self) -> Dict[str, Any]:
        """Получает анализ трендов с повторными попытками"""
        try:
            cached = self.cache.get_cached_buzz_analysis()
            if cached:
                return cached
            if not self.client:
                return self._create_mock_buzz_analysis()
            prompt = """Проанализируйте текущие глобальные новостные тренды"""
            try:
                response = self._make_api_request_with_retry(
                    method="messages.create",
                    model="claude-3-opus-20240229",
                    max_tokens=4000,
                    messages=[{"role": "user", "content": prompt}]
                )
                analysis = self._parse_response(response.content[0].text)
                self.cache.cache_buzz_analysis(analysis)
                return analysis
            except Exception as e:
                logger.error(f"Ошибка получения анализа трендов: {str(e)}")
                return self._create_mock_buzz_analysis()
        except Exception as e:
            logger.error(f"Ошибка в get_buzz_analysis: {str(e)}")
            return self._create_mock_buzz_analysis()

    def determine_credibility_level(self, score: Any) -> str:
        """Определяет уровень достоверности"""
        try:
            if isinstance(score, dict):
                # Если это словарь, проверяем наличие индекса достоверности
                if 'credibility_index' in score:
                    score = score['credibility_index']
                else:
                    score = score.get('score', 0.6)

            score = float(score)

            if score >= 0.8:
                return "High"
            elif score >= 0.6:
                return "Medium"
            return "Low"
        except Exception:
            return "Medium"

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

    def _create_mock_buzz_analysis(self) -> Dict[str, Any]:
        """
        Возвращает заглушку анализа "горячих" тем, если API недоступен.
        """
        return {
            "title": "Today's Featured Analysis (Mock)",
            "source": "Mock Analysis",
            "short_summary": "This is a mock analysis of today's featured news topics.",
            "analysis": {
                "credibility_score": {"score": 0.8},
                "topics": ["featured", "news"],
                "summary": "In today's featured analysis, we look at the most important news topics of the day. This is a mock analysis since the Claude API is not available.",
                "perspectives": {
                    "neutral": {
                        "summary": "Neutral perspective on today's featured topics. This is a mock analysis.",
                        "credibility": "High"
                    }
                }
            }
        }

if __name__ == "__main__":
    # Тестовый код
    api = ClaudeAPI()

    # Пример анализа статьи
    test_content = """Изменение климата ускоряется беспрецедентными темпами.
    Ученые предупреждают о необходимости немедленных действий."""
    result = api.analyze_article(test_content, "BBC News")
    print("Анализ статьи:")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    # Выводим индекс достоверности
    if 'analysis' in result and 'credibility_index' in result['analysis']:
        print(f"\nИндекс достоверности: {result['analysis']['credibility_index']}")
    elif 'credibility_info' in result and 'credibility_index' in result['credibility_info']:
        print(f"\nИндекс достоверности: {result['credibility_info']['credibility_index']}")

    # Пример анализа трендов
    buzz = api.get_buzz_analysis()
    print("\nАнализ трендов:")
    print(json.dumps(buzz, indent=2, ensure_ascii=False))
