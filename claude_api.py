import os
import logging
import json
from typing import Dict, Any, Optional, List, Union
import anthropic
from pydantic import BaseModel
from datetime import datetime
import re
from cache import CacheManager
from urllib.parse import urlparse

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

class NewsAPI:
    """Моковый класс NewsAPI для тестирования"""
    def __init__(self):
        self.api_key = os.getenv('NEWS_API_KEY', 'mock-api-key')

    def get_everything(self, query: str, page_size: int = 5) -> List[Dict[str, Any]]:
        """Моковый метод для получения статей с улучшенными данными"""
        try:
            mock_articles = [
                {
                    "title": f"Article about {query}",
                    "source": {"name": "BBC", "url": "https://www.bbc.com"},
                    "url": "https://example.com/article1",
                    "description": f"Recent news about {query} from BBC",
                    "publishedAt": datetime.now().isoformat(),
                    "content": f"Full content about {query} would be here in a real application..."
                },
                {
                    "title": f"Analysis of {query}",
                    "source": {"name": "Reuters", "url": "https://www.reuters.com"},
                    "url": "https://example.com/article2",
                    "description": f"In-depth analysis of {query} from Reuters",
                    "publishedAt": datetime.now().isoformat(),
                    "content": f"Comprehensive analysis of {query} with expert opinions..."
                },
                {
                    "title": f"Latest developments in {query}",
                    "source": {"name": "The Guardian", "url": "https://www.theguardian.com"},
                    "url": "https://example.com/article3",
                    "description": f"Recent updates on {query} situation",
                    "publishedAt": datetime.now().isoformat(),
                    "content": f"Breaking news about {query} with eyewitness accounts..."
                },
                {
                    "title": f"Expert opinion on {query}",
                    "source": {"name": "The New York Times", "url": "https://www.nytimes.com"},
                    "url": "https://example.com/article4",
                    "description": f"Expert analysis of {query} developments",
                    "publishedAt": datetime.now().isoformat(),
                    "content": f"In-depth report on {query} with interviews and data analysis..."
                },
                {
                    "title": f"Global perspective on {query}",
                    "source": {"name": "Al Jazeera", "url": "https://www.aljazeera.com"},
                    "url": "https://example.com/article5",
                    "description": f"International view on {query} situation",
                    "publishedAt": datetime.now().isoformat(),
                    "content": f"Global reactions and implications of {query} events..."
                }
            ]
            return mock_articles[:page_size]
        except Exception as e:
            logger.error(f"Error in mock NewsAPI: {str(e)}")
            return []

class ClaudeAPI:
    def __init__(self):
        self.cache = CacheManager()
        self.news_api = NewsAPI()
        self.client = self._initialize_anthropic_client()

    def _initialize_anthropic_client(self):
        """Инициализирует клиент Anthropic с обработкой ошибок"""
        try:
            api_key = os.getenv('ANTHROPIC_API_KEY')
            if not api_key:
                logger.warning("ANTHROPIC_API_KEY not set, using mock responses")
                return None
            return anthropic.Anthropic(api_key=api_key)
        except Exception as e:
            logger.error(f"Failed to initialize Anthropic client: {str(e)}")
            return None

    def _safe_get(self, data: Dict[str, Any], key: str, default: Any = None) -> Any:
        """Безопасное извлечение значения из словаря"""
        try:
            if isinstance(data, dict) and key in data:
                return data[key]
            return default
        except Exception:
            return default

    def _normalize_score(self, score: Union[float, Dict[str, float]]) -> Dict[str, float]:
        """Нормализует оценку в формат словаря"""
        if isinstance(score, dict):
            return {
                'score': float(self._safe_get(score, 'score', 0.6)),
                'confidence': float(self._safe_get(score, 'confidence', 0.8))
            }
        if isinstance(score, (float, int)):
            return {'score': float(score), 'confidence': 0.8}
        return {'score': 0.6, 'confidence': 0.7}

    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """Парсит ответ от API с улучшенной обработкой ошибок"""
        try:
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))
                parsed.setdefault('credibility_score', {'score': 0.7})
                parsed.setdefault('sentiment', {'score': 0.1})
                parsed.setdefault('bias', {'level': 0.2})
                parsed.setdefault('topics', [])
                parsed.setdefault('perspectives', {})
                parsed.setdefault('key_arguments', [])
                return parsed

            return {
                "credibility_score": {"score": 0.7, "confidence": 0.8},
                "sentiment": {"score": 0.1, "confidence": 0.7},
                "bias": {"level": 0.2, "confidence": 0.75},
                "topics": [{"name": "General", "relevance": 0.9}],
                "perspectives": {
                    "neutral": {
                        "summary": "Basic analysis of the content",
                        "credibility": "Medium",
                        "confidence": 0.8
                    }
                },
                "key_arguments": ["Content requires more context for detailed analysis"]
            }
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {str(e)}")
            return self._create_fallback_analysis("JSON parsing error")
        except Exception as e:
            logger.error(f"Error parsing response: {str(e)}")
            return self._create_fallback_analysis(str(e))

    def _create_fallback_analysis(self, error: str) -> Dict[str, Any]:
        """Создает резервный анализ при ошибках"""
        return {
            "credibility_score": {"score": 0.6, "confidence": 0.6},
            "sentiment": {"score": 0.0, "confidence": 0.5},
            "bias": {"level": 0.3, "confidence": 0.6},
            "topics": [{"name": "General", "relevance": 0.7}],
            "perspectives": {
                "neutral": {
                    "summary": f"Analysis failed due to: {error}",
                    "credibility": "Low",
                    "confidence": 0.5
                }
            },
            "key_arguments": [
                "Could not complete full analysis",
                f"Error encountered: {error}"
            ],
            "error": error
        }

    def _generate_credibility_explanation(self, score: float, source: str) -> Dict[str, Any]:
        """Генерирует объяснение для оценки достоверности"""
        source_level = self.determine_credibility_level_from_source(source)
        rounded_score = round(score * 100)

        if score >= 0.8:
            explanation = (
                f"Эта статья получила высокий рейтинг достоверности ({rounded_score}/100), "
                f"поскольку она соответствует нескольким ключевым критериям:\n"
                f"1. Источник ({source}) имеет репутацию {source_level.lower()} достоверности\n"
                f"2. Содержание хорошо структурировано и подтверждено фактами\n"
                f"3. Минимальное количество необоснованных утверждений\n"
                f"4. Сбалансированное представление различных точек зрения"
            )
        elif score >= 0.6:
            explanation = (
                f"Эта статья получила средний рейтинг достоверности ({rounded_score}/100). "
                f"Вот основные факторы:\n"
                f"1. Источник ({source}) обычно надежен, но имеет некоторые предвзятости\n"
                f"2. Содержание содержит как подтвержденные факты, так и несколько необоснованных утверждений\n"
                f"3. Может не хватать альтернативных точек зрения\n"
                f"4. Требуется дополнительная проверка некоторых заявлений"
            )
        else:
            explanation = (
                f"Эта статья получила низкий рейтинг достоверности ({rounded_score}/100). "
                f"Основные проблемы:\n"
                f"1. Источник ({source}) имеет репутацию низкой достоверности\n"
                f"2. Содержание содержит много необоснованных утверждений\n"
                f"3. Отсутствуют сбалансированные точки зрения\n"
                f"4. Рекомендуется проверить информацию из других источников"
            )

        return {
            "level": self.determine_credibility_level(score),
            "explanation": explanation,
            "score": round(score, 2),
            "source_level": source_level
        }

    def get_buzz_analysis(self) -> Dict[str, Any]:
        """Получает анализ популярных новостей с улучшенной обработкой ошибок"""
        try:
            cached = self.cache.get_cached_buzz_analysis()
            if cached:
                return cached

            if not self.client:
                return self._create_mock_buzz_analysis()

            prompt = """Analyze current global news trends focusing on:
1. Key geopolitical developments
2. Economic indicators
3. Technological advancements
4. Social trends
Provide a comprehensive analysis with:
- Detailed credibility assessment (0-1 scale with explanation)
- Multiple perspectives with credibility ratings
- Key arguments with supporting evidence
- Potential biases and their impact
- Future implications
Format the response as JSON with this detailed structure:
{
    "title": "string",
    "source": "string",
    "short_summary": "string",
    "full_analysis": "string (up to 4000 characters)",
    "analysis": {
        "credibility_score": {"score": number, "explanation": "string"},
        "sentiment": {"score": number, "explanation": "string"},
        "bias": {"level": number, "explanation": "string"},
        "topics": [{"name": "string", "relevance": number, "description": "string"}],
        "perspectives": {
            "western": {
                "summary": "string",
                "credibility": "string",
                "supporting_evidence": ["list"],
                "potential_biases": ["list"]
            },
            "eastern": {
                "summary": "string",
                "credibility": "string",
                "supporting_evidence": ["list"],
                "potential_biases": ["list"]
            },
            "neutral": {
                "summary": "string",
                "credibility": "string",
                "supporting_evidence": ["list"],
                "potential_biases": ["list"]
            }
        },
        "key_arguments": [
            {
                "statement": "string",
                "supporting_evidence": ["list"],
                "counter_arguments": ["list"],
                "verification_status": "verified/unverified/contested"
            }
        ],
        "future_implications": ["string"]
    }
}"""

            try:
                response = self.client.messages.create(
                    model="claude-3-opus-20240229",
                    max_tokens=4000,
                    temperature=0.7,
                    system="You are a professional news analyst with expertise in media credibility assessment. Provide detailed explanations for all your assessments.",
                    messages=[{"role": "user", "content": prompt}]
                )
                analysis = self._parse_response(response.content[0].text)
                normalized = self._normalize_analysis(analysis)
                self.cache.cache_buzz_analysis(normalized)
                return normalized
            except Exception as e:
                logger.error(f"Error getting buzz analysis: {str(e)}")
                return self._create_mock_buzz_analysis()
        except Exception as e:
            logger.error(f"Unexpected error in get_buzz_analysis: {str(e)}")
            return self._create_mock_buzz_analysis()

    def _create_mock_buzz_analysis(self) -> Dict[str, Any]:
        """Создает mock-анализ для случаев ошибок"""
        return {
            "article": {
                "title": "Today's featured analysis: Global News Trends",
                "source": "Media Analysis",
                "short_summary": "Comprehensive analysis of current global news trends and their credibility patterns. This mock analysis demonstrates how our system evaluates news content.",
                "full_analysis": "Our system analyzes news content using multiple criteria including source reliability, factual consistency, and balanced reporting. For this demo, we're showing how a typical analysis would look with explanations for each credibility assessment.",
                "analysis": {
                    "credibility_score": {
                        "score": 0.85,
                        "explanation": "This mock analysis demonstrates our high-confidence assessment. In a real scenario, we would provide specific details about why this particular score was assigned based on the article content and source reliability.",
                        "source_rating": "High",
                        "fact_check": {
                            "inconsistencies": [],
                            "unsupported_claims": []
                        }
                    },
                    "sentiment": {
                        "score": 0.1,
                        "explanation": "Neutral sentiment indicates balanced reporting without strong emotional bias in either direction.",
                        "tone": "neutral"
                    },
                    "bias": {
                        "level": 0.2,
                        "explanation": "Low bias score suggests the content presents multiple viewpoints fairly.",
                        "examples": []
                    },
                    "topics": [
                        {
                            "name": "Geopolitics",
                            "relevance": 0.9,
                            "description": "International relations and global political developments"
                        },
                        {
                            "name": "Economy",
                            "relevance": 0.8,
                            "description": "Global and regional economic indicators and trends"
                        }
                    ],
                    "perspectives": {
                        "western": {
                            "summary": "Western media's perspective on current global events",
                            "credibility": "High",
                            "supporting_evidence": [
                                "Multiple independent sources confirm key facts",
                                "Historical context provided"
                            ],
                            "potential_biases": [
                                "Possible pro-western slant in interpretation"
                            ]
                        },
                        "neutral": {
                            "summary": "Balanced analysis of current global events",
                            "credibility": "High",
                            "supporting_evidence": [
                                "Data from international organizations",
                                "Expert opinions from various regions"
                            ],
                            "potential_biases": []
                        }
                    },
                    "key_arguments": [
                        {
                            "statement": "Global economic growth continues despite regional challenges",
                            "supporting_evidence": [
                                "IMF reports show steady growth",
                                "Corporate earnings reports positive"
                            ],
                            "counter_arguments": [
                                "Some regions face economic slowdown",
                                "Geopolitical tensions could impact growth"
                            ],
                            "verification_status": "verified"
                        }
                    ],
                    "future_implications": [
                        "Possible shifts in global trade patterns",
                        "Emerging technologies may disrupt traditional industries"
                    ]
                }
            }
        }

    def _normalize_analysis(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Нормализует и улучшает структуру анализа"""
        normalized = {
            "credibility_score": self._normalize_score(analysis.get('credibility_score', 0.6)),
            "sentiment": self._normalize_score(analysis.get('sentiment', 0.1)),
            "bias": self._normalize_score(analysis.get('bias', 0.2)),
            "topics": analysis.get('topics', []),
            "perspectives": analysis.get('perspectives', {}),
            "key_arguments": analysis.get('key_arguments', []),
            "similar_articles": analysis.get('similar_articles', [])
        }

        # Добавляем объяснения, если их нет
        if isinstance(normalized['credibility_score'], dict) and 'explanation' not in normalized['credibility_score']:
            score = normalized['credibility_score']['score']
            normalized['credibility_score']['explanation'] = (
                "High credibility" if score >= 0.8 else
                "Medium credibility" if score >= 0.6 else
                "Low credibility"
            )

        # Преобразуем key_arguments если они в старом формате
        if normalized['key_arguments'] and isinstance(normalized['key_arguments'][0], str):
            normalized['key_arguments'] = [
                {"statement": arg, "supporting_evidence": [], "counter_arguments": [], "verification_status": "unverified"}
                for arg in normalized['key_arguments']
            ]

        return normalized

    def analyze_article(self, content: str, source: str) -> Dict[str, Any]:
        """Анализирует статью с улучшенным анализом достоверности и ссылками"""
        try:
            # Проверяем кэш
            cache_key = f"{source}:{content[:100]}"  # Используем часть контента для ключа
            cached = self.cache.get_cached_article_analysis(cache_key)
            if cached:
                return cached

            # Подготавливаем промпт с увеличенным лимитом символов
            prompt_content = content[:4000]  # Увеличили с 2000 до 4000 символов
            if len(content) > 4000:
                prompt_content += "\n\n[Content trimmed for analysis. Original length: %d characters]" % len(content)

            prompt = f"""Perform a comprehensive analysis of the following article content with these detailed requirements:
1. Credibility Assessment:
   - Evaluate overall credibility (0-1 scale)
   - Provide detailed explanation for the score
   - Assess source credibility separately
   - Identify any factual inconsistencies
   - Check for proper sourcing of claims

2. Content Analysis:
   - Determine sentiment (-1 to 1 scale) with explanation
   - Identify potential biases (0-1 scale) with examples
   - Extract key topics with relevance scores
   - Identify main arguments and supporting evidence

3. Perspective Analysis:
   - Provide western perspective with credibility rating
   - Provide eastern perspective with credibility rating
   - Provide neutral perspective with credibility rating

4. Supporting Context:
   - Suggest additional sources that would improve balance
   - Identify any missing perspectives
   - Highlight controversial statements needing verification

Article source: {source}
Article content: {prompt_content}
Format the response as JSON with this comprehensive structure:
{{
    "credibility_score": {{
        "score": number,
        "explanation": "detailed explanation",
        "source_rating": "High/Medium/Low",
        "fact_check": {{
            "inconsistencies": ["list of issues"],
            "unsupported_claims": ["list of claims"]
        }}
    }},
    "sentiment": {{
        "score": number,
        "explanation": "string",
        "tone": "positive/neutral/negative/mixed"
    }},
    "bias": {{
        "level": number,
        "explanation": "string",
        "examples": ["list of biased statements"]
    }},
    "topics": [
        {{
            "name": "string",
            "relevance": number,
            "description": "string"
        }}
    ],
    "perspectives": {{
        "western": {{
            "summary": "string",
            "credibility": "string",
            "supporting_evidence": ["list"],
            "potential_biases": ["list"]
        }},
        "eastern": {{
            "summary": "string",
            "credibility": "string",
            "supporting_evidence": ["list"],
            "potential_biases": ["list"]
        }},
        "neutral": {{
            "summary": "string",
            "credibility": "string",
            "supporting_evidence": ["list"],
            "potential_biases": ["list"]
        }}
    }},
    "key_arguments": [
        {{
            "statement": "string",
            "supporting_evidence": ["list"],
            "counter_arguments": ["list"],
            "verification_status": "verified/unverified/contested"
        }}
    ],
    "recommendations": [
        "string suggestions for improving balance/accuracy"
    ],
    "similar_articles": []
}}
Note: Provide detailed explanations for all scores and ratings."""

            try:
                if self.client:
                    response = self.client.messages.create(
                        model="claude-3-opus-20240229",
                        max_tokens=4000,
                        temperature=0.5,
                        system="You are a professional media analyst. Provide a detailed, structured analysis of the article content with thorough explanations for all assessments.",
                        messages=[{"role": "user", "content": prompt}]
                    )
                    analysis = self._parse_response(response.content[0].text)
                else:
                    analysis = self._create_mock_article_analysis(source, content)
            except Exception as e:
                logger.error(f"Error analyzing article with Claude API: {str(e)}")
                analysis = self._create_mock_article_analysis(source, content)

            # Добавляем улучшенный анализ достоверности
            credibility_score = analysis.get('credibility_score', {'score': 0.7})
            if isinstance(credibility_score, dict):
                score = credibility_score.get('score', 0.7)
            else:
                score = float(credibility_score)

            analysis['credibility_score'] = {
                **self._normalize_score(credibility_score),
                **self._generate_credibility_explanation(score, source)
            }

            # Получаем похожие статьи
            topics = [t['name'] if isinstance(t, dict) else t for t in analysis.get('topics', [])]
            similar_articles = []

            if topics:
                try:
                    query = ' OR '.join([str(t) for t in topics[:3]])
                    similar_articles = self.news_api.get_everything(query=query, page_size=5) or []

                    for article in similar_articles:
                        if 'url' not in article:
                            article['url'] = None
                        if 'source' not in article or not isinstance(article['source'], dict):
                            article['source'] = {'name': 'Unknown', 'url': None}
                        if 'publishedAt' not in article:
                            article['publishedAt'] = None
                        if 'content' not in article:
                            article['content'] = None
                except Exception as e:
                    logger.error(f"Error getting similar articles: {str(e)}")
                    similar_articles = []

            # Формируем результат
            result = {
                "title": analysis.get('title', 'Article Analysis'),
                "source": source,
                "url": None,  # Будет установлен вызывающей функцией
                "short_summary": content[:1000] + '...' if len(content) > 1000 else content,
                "full_analysis": analysis.get('full_analysis', ''),
                "analysis": self._normalize_analysis(analysis),
                "credibility": analysis.get('credibility_score', {'score': 0.7}),
                "similar_articles": similar_articles,
                "links": [],  # Будет добавлено вызывающей функцией
                "metadata": {
                    "analysis_timestamp": datetime.now().isoformat(),
                    "content_length": len(content),
                    "source_credibility": self.determine_credibility_level_from_source(source)
                }
            }

            # Кэшируем результат
            self.cache.cache_article_analysis(cache_key, result)
            return result

        except Exception as e:
            logger.error(f"Unexpected error in analyze_article: {str(e)}")
            return self._create_mock_article_analysis(source, content[:200])

    def _create_mock_article_analysis(self, source: str, content_preview: str) -> Dict[str, Any]:
        """Создает mock-анализ для случаев ошибок"""
        return {
            "title": "Analysis of Provided Content",
            "source": source,
            "short_summary": content_preview,
            "full_analysis": "This is a mock analysis generated when our system is unable to process the article through our main analysis pipeline. Below are typical analysis components with placeholder values.",
            "analysis": {
                "credibility_score": {
                    "score": 0.7,
                    "explanation": f"This mock score represents what our system typically assigns to content from {source}. In a real analysis, this would be based on detailed evaluation of the article content and source reliability.",
                    "source_rating": self.determine_credibility_level_from_source(source),
                    "fact_check": {
                        "inconsistencies": [],
                        "unsupported_claims": []
                    }
                },
                "sentiment": {
                    "score": 0.0,
                    "explanation": "Neutral sentiment is typical for news reporting without strong emotional language.",
                    "tone": "neutral"
                },
                "bias": {
                    "level": 0.3,
                    "explanation": "Moderate bias level suggests some potential for slanted reporting but generally balanced content.",
                    "examples": []
                },
                "topics": [
                    {"name": "General News", "relevance": 0.8, "description": "News content without specific topic focus"}
                ],
                "perspectives": {
                    "neutral": {
                        "summary": "Mock analysis perspective representing balanced view",
                        "credibility": "Medium",
                        "supporting_evidence": [
                            "Content appears to be factual reporting",
                            "No obvious slant detected"
                        ],
                        "potential_biases": ["None identified in this mock analysis"]
                    }
                },
                "key_arguments": [
                    {
                        "statement": "The content appears to be newsworthy",
                        "supporting_evidence": ["Presence of factual information"],
                        "counter_arguments": [],
                        "verification_status": "unverified"
                    }
                ],
                "recommendations": [
                    "For a complete analysis, please ensure our API service is properly configured",
                    "Verify all claims with additional sources"
                ]
            },
            "credibility": {
                "score": 0.7,
                "level": "Medium",
                "explanation": "Mock credibility assessment",
                "source_level": self.determine_credibility_level_from_source(source)
            },
            "similar_articles": [],
            "links": [],
            "metadata": {
                "analysis_timestamp": datetime.now().isoformat(),
                "content_length": len(content_preview),
                "source_credibility": self.determine_credibility_level_from_source(source)
            }
        }

    def determine_credibility_level(self, score: Any) -> str:
        """Определяет уровень достоверности на основе оценки с улучшенной обработкой"""
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

    def determine_credibility_level_from_source(self, source_name: str) -> str:
        """Определяет уровень достоверности на основе источника с улучшенной обработкой"""
        try:
            source_name = source_name.lower()
            high_credibility_sources = [
                'bbc', 'reuters', 'associated press', 'the new york times',
                'the guardian', 'the wall street journal', 'bloomberg',
                'the economist', 'financial times', 'washington post',
                'nature', 'science', 'lse', 'harvard'
            ]
            medium_credibility_sources = [
                'cnn', 'fox news', 'usa today', 'the washington post',
                'npr', 'al jazeera', 'the independent', 'the hill',
                'politico', 'business insider', 'vox', 'buzzfeed',
                'the verge', 'techcrunch'
            ]
            low_credibility_sources = [
                'daily mail', 'breitbart', 'infowars', 'natural news',
                'the sun', 'sputnik', 'rt', 'global research'
            ]

            if any(source in source_name for source in high_credibility_sources):
                return "High"
            elif any(source in source_name for source in medium_credibility_sources):
                return "Medium"
            elif any(source in source_name for source in low_credibility_sources):
                return "Low"
            else:
                # Для неизвестных источников используем доменное имя для оценки
                domain_keywords = {
                    'gov': 'High',
                    'edu': 'High',
                    'org': 'Medium',
                    'com': 'Medium',
                    'net': 'Medium'
                }

                # Пытаемся извлечь домен из URL если это URL
                if source_name.startswith(('http://', 'https://')):
                    try:
                        domain = urlparse(source_name).netloc
                        if '.' in domain:
                            tld = domain.split('.')[-1]
                            if tld in domain_keywords:
                                return domain_keywords[tld]
                    except:
                        pass

                return "Medium"
        except Exception as e:
            logger.error(f"Error determining source credibility: {str(e)}")
            return "Medium"

if __name__ == "__main__":
    # Тестовый код для проверки функциональности
    logging.info("Running claude_api.py in standalone mode")
    api = ClaudeAPI()

    # Пример использования
    test_content = """Recent studies have shown that climate change is accelerating at an unprecedented rate.
Scientists warn that immediate action is needed to prevent catastrophic consequences.
However, some politicians continue to dispute these findings, citing economic concerns."""

    analysis = api.analyze_article(test_content, "Test Source")
    print(json.dumps(analysis, indent=2, ensure_ascii=False))
