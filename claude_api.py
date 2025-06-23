import os
import logging
import json
from typing import Dict, Any, Optional, List
import anthropic
from pydantic import BaseModel
from datetime import datetime
import re
from cache import CacheManager

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ArticleAnalysis(BaseModel):
    credibility_score: Dict[str, float]
    sentiment: Dict[str, float]
    bias: Dict[str, float]
    topics: List[Dict[str, Any]]
    perspectives: Dict[str, Dict[str, Any]]
    key_arguments: List[str]

class ClaudeAPI:
    def __init__(self):
        self.cache = CacheManager()
        self.client = self._initialize_anthropic_client()

    def _initialize_anthropic_client(self):
        """Инициализирует клиент Anthropic"""
        try:
            api_key = os.getenv('ANTHROPIC_API_KEY')
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY environment variable not set")
            return anthropic.Anthropic(api_key=api_key)
        except Exception as e:
            logger.error(f"Failed to initialize Anthropic client: {str(e)}")
            return None

    def get_buzz_analysis(self) -> Dict[str, Any]:
        """Получает анализ популярных новостей"""
        try:
            cached = self.cache.get_cached_buzz_analysis()
            if cached:
                return cached

            # Получаем актуальный анализ
            prompt = """Analyze current global news trends focusing on:
1. Key geopolitical developments
2. Economic indicators
3. Technological advancements
4. Social trends

Provide a comprehensive analysis with:
- Credibility assessment
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

            response = self.client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=2000,
                temperature=0.7,
                system="You are a professional news analyst with expertise in media credibility assessment.",
                messages=[{"role": "user", "content": prompt}]
            )

            analysis = json.loads(response.content[0].text)
            self.cache.cache_buzz_analysis(analysis)
            return analysis
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
                            {"name": "Geopolitics", "relevance": 0.9},
                            {"name": "Economy", "relevance": 0.8},
                            {"name": "Technology", "relevance": 0.7}
                        ],
                        "perspectives": {
                            "western": {
                                "summary": "Western perspective on current events",
                                "credibility": "High"
                            },
                            "eastern": {
                                "summary": "Eastern perspective on current events",
                                "credibility": "Medium"
                            },
                            "neutral": {
                                "summary": "Neutral analysis of current events",
                                "credibility": "High"
                            }
                        },
                        "key_arguments": [
                            "Global economic growth continues",
                            "Technological advancements accelerating",
                            "Geopolitical tensions remain"
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

            response = self.client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=2000,
                temperature=0.5,
                system="You are a professional media analyst. Provide a detailed, structured analysis of the article content.",
                messages=[{"role": "user", "content": prompt}]
            )

            # Безопасное извлечение данных из ответа
            try:
                analysis_text = response.content[0].text
                analysis = json.loads(analysis_text)

                # Проверяем и исправляем структуру данных
                if isinstance(analysis.get('credibility_score'), float):
                    analysis['credibility_score'] = {'score': analysis['credibility_score']}
                if isinstance(analysis.get('sentiment'), float):
                    analysis['sentiment'] = {'score': analysis['sentiment']}
                if isinstance(analysis.get('bias'), float):
                    analysis['bias'] = {'level': analysis['bias']}

                # Кэшируем результат
                self.cache.cache_article_analysis(content, analysis)
                return analysis
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
                    "key_arguments": ["Content requires more context for detailed analysis"]
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
                "key_arguments": ["Analysis failed due to technical issues"]
            }

    def determine_credibility_level(self, score: Any) -> str:
        """Определяет уровень достоверности на основе оценки"""
        try:
            # Обрабатываем разные типы входных данных
            if isinstance(score, dict):
                score_value = score.get('score', 0.6)
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
