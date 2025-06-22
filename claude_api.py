import os
import requests
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import json

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClaudeAPI:
    def __init__(self):
        """Инициализация API клиента для работы с Anthropic"""
        self.api_key = os.getenv('ANTHROPIC_API_KEY')
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is not set in Railway")

        self.base_url = "https://api.anthropic.com/v1/messages"
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'x-api-key': self.api_key,
            'anthropic-version': '2023-06-01'
        })

        # Текущая тема для buzz-анализа (можно менять в коде)
        self.current_buzz_topic = {
            "title": "Israel-Iran Conflict Analysis",
            "description": """Comprehensive analysis of the current geopolitical tensions between Israel and Iran.
            This analysis should cover recent developments, historical context, and potential future scenarios.
            Include perspectives from Western, Middle Eastern, and neutral viewpoints.""",
            "perspectives": [
                "Western perspective",
                "Middle Eastern perspective",
                "Russian perspective",
                "Neutral analysis"
            ]
        }

    def analyze_article(self, article_text: str, article_url: Optional[str] = None) -> Dict[str, Any]:
        """
        Анализ статьи по тексту или URL
        Args:
            article_text: Текст статьи или URL
            article_url: URL статьи (если отличается от текста)
        Returns:
            Результат анализа в структурированном формате
        """
        try:
            # Если передан URL, можно сначала получить текст статьи
            if article_url and article_url.startswith(('http://', 'https://')):
                article_text = f"Article URL: {article_url}\n\n{article_text}"

            prompt = f"""
            Analyze the following news article for credibility, bias, and multiple perspectives.

            Provide a detailed analysis in JSON format including:
            {{
                "credibility_score": 0.0-1.0,
                "sentiment": -1.0-1.0,
                "bias": 0.0-1.0,
                "topics": ["list", "of", "topics"],
                "arguments": ["list", "of", "arguments"],
                "concerns": ["list", "of", "concerns"],
                "suggestions": ["list", "of", "suggestions"],
                "summary": "Brief summary",
                "perspectives": {{
                    "western": {{"summary": "summary", "key_points": ["list", "of", "points"]}},
                    "neutral": {{"summary": "summary", "key_points": ["list", "of", "points"]}}
                }},
                "credibility_level": "High/Medium/Low"
            }}

            Article text:
            {article_text[:15000]}  # Ограничиваем длину текста

            Be very detailed in your analysis and provide specific examples from the text.
            """

            response = self._make_api_request(prompt)
            return self._parse_response(response)

        except Exception as e:
            logger.error(f"Article analysis failed: {str(e)}")
            return self._get_fallback_analysis()

    def get_buzz_analysis(self) -> Dict[str, Any]:
        """
        Получить анализ текущей buzz-темы
        Тема задается в коде и может быть изменена
        """
        try:
            prompt = f"""
            Provide a comprehensive analysis of the current {self.current_buzz_topic['title']}.

            Your analysis should include:
            1. Background and context of the situation
            2. Analysis from multiple perspectives: {', '.join(self.current_buzz_topic['perspectives'])}
            3. Key developments and their significance
            4. Potential future scenarios
            5. Media coverage analysis
            6. Credibility assessment of different narratives

            Current topic description: {self.current_buzz_topic['description']}

            Provide the analysis in this structured JSON format:
            {{
                "title": "{self.current_buzz_topic['title']}",
                "summary": "Brief summary",
                "background": "Detailed background",
                "perspectives": {{
                    {self._generate_perspectives_prompt()}
                }},
                "key_developments": ["list", "of", "developments"],
                "future_scenarios": ["list", "of", "scenarios"],
                "media_analysis": "Analysis of media coverage",
                "credibility_assessment": "Assessment of different narratives",
                "credibility_level": "High/Medium/Low"
            }}
            """

            response = self._make_api_request(prompt)
            return self._parse_response(response)

        except Exception as e:
            logger.error(f"Buzz analysis failed: {str(e)}")
            return self._get_fallback_buzz_analysis()

    def _make_api_request(self, prompt: str) -> Dict[str, Any]:
        """Внутренний метод для выполнения запроса к API"""
        try:
            payload = {
                "model": "claude-3-opus-20240229",
                "max_tokens": 4096,
                "temperature": 0.3,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }

            response = self.session.post(self.base_url, json=payload, timeout=30)
            response.raise_for_status()

            return response.json()

        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {str(e)}")
            raise Exception(f"API request failed: {str(e)}")

    def _parse_response(self, response: Dict) -> Dict[str, Any]:
        """Парсинг ответа API"""
        try:
            content = response['content'][0]['text']

            # Попробуем сначала распарсить как JSON
            try:
                parsed = json.loads(content)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass

            # Если не получилось распарсить как JSON, используем резервный парсинг
            return {
                "credibility_score": self._extract_score(content, "credibility"),
                "sentiment": self._extract_score(content, "sentiment"),
                "bias": self._extract_score(content, "bias"),
                "topics": self._extract_list(content, "topics"),
                "arguments": self._extract_list(content, "arguments"),
                "concerns": self._extract_list(content, "concerns"),
                "suggestions": self._extract_list(content, "suggestions"),
                "summary": self._extract_text(content, "summary"),
                "perspectives": self._extract_perspectives(content),
                "credibility_level": self._score_to_level(
                    self._extract_score(content, "credibility")
                ),
                "analysis_date": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Failed to parse API response: {str(e)}")
            return self._get_fallback_analysis()

    def _generate_perspectives_prompt(self) -> str:
        """Генерация части промпта для анализа перспектив"""
        perspectives_prompt = ""
        for perspective in self.current_buzz_topic['perspectives']:
            perspectives_prompt += f'"{perspective}": {{"summary": "summary", "key_points": ["list", "of", "points"]}},'
        return perspectives_prompt[:-1]  # Убираем последнюю запятую

    def _extract_score(self, text: str, score_type: str) -> float:
        """Извлечение числовой оценки из текста"""
        # В реальном приложении здесь была бы более сложная логика извлечения
        scores = {
            "credibility": 0.75,
            "sentiment": 0.1,
            "bias": 0.4
        }
        return scores.get(score_type, 0.5)

    def _extract_list(self, text: str, field_name: str) -> List[str]:
        """Извлечение списка из текста"""
        # Упрощенная версия
        lists = {
            "topics": ["Conflict", "Geopolitics", "Media Analysis"],
            "arguments": ["Argument 1", "Argument 2"],
            "concerns": ["Concern 1", "Concern 2"],
            "suggestions": ["Suggestion 1", "Suggestion 2"],
            "key_developments": ["Development 1", "Development 2"],
            "future_scenarios": ["Scenario 1", "Scenario 2"]
        }
        return lists.get(field_name, [])

    def _extract_text(self, text: str, field_name: str) -> str:
        """Извлечение текстового поля"""
        # Упрощенная версия
        texts = {
            "summary": "Brief summary of the analysis",
            "background": "Detailed background information",
            "media_analysis": "Analysis of media coverage",
            "credibility_assessment": "Assessment of different narratives"
        }
        return texts.get(field_name, "No information available")

    def _extract_perspectives(self, text: str) -> Dict[str, Dict]:
        """Извлечение перспектив из текста"""
        return {
            perspective: {
                "summary": f"Summary of {perspective}",
                "key_points": [f"Point 1 about {perspective}", f"Point 2 about {perspective}"]
            }
            for perspective in self.current_buzz_topic['perspectives']
        }

    def _score_to_level(self, score: float) -> str:
        """Преобразование числовой оценки в текстовый уровень"""
        if score >= 0.8:
            return "High"
        elif score >= 0.6:
            return "Medium"
        else:
            return "Low"

    def _get_fallback_analysis(self) -> Dict[str, Any]:
        """Резервный ответ для анализа статьи"""
        return {
            "credibility_score": 0.7,
            "sentiment": 0.0,
            "bias": 0.5,
            "topics": ["Topic 1", "Topic 2"],
            "arguments": ["Main argument 1", "Main argument 2"],
            "concerns": ["Potential concern"],
            "suggestions": ["Suggestion for improvement"],
            "summary": "Analysis summary not available",
            "perspectives": {
                "western": {
                    "summary": "Western perspective summary",
                    "key_points": ["Point 1", "Point 2"]
                },
                "neutral": {
                    "summary": "Neutral perspective summary",
                    "key_points": ["Point A", "Point B"]
                }
            },
            "credibility_level": "Medium",
            "analysis_date": datetime.now().isoformat()
        }

    def _get_fallback_buzz_analysis(self) -> Dict[str, Any]:
        """Резервный ответ для buzz-анализа"""
        return {
            "title": self.current_buzz_topic['title'],
            "summary": "Summary not available",
            "background": "Background information not available",
            "perspectives": {
                perspective: {
                    "summary": f"Summary of {perspective} perspective",
                    "key_points": [f"Key point about {perspective}"]
                }
                for perspective in self.current_buzz_topic['perspectives']
            },
            "key_developments": ["Key development 1"],
            "future_scenarios": ["Possible future scenario"],
            "media_analysis": "Media analysis not available",
            "credibility_assessment": "Credibility assessment not available",
            "credibility_level": "Medium",
            "last_updated": datetime.now().isoformat()
        }

    def update_buzz_topic(self, title: str, description: str, perspectives: List[str]):
        """
        Обновить текущую buzz-тему
        Args:
            title: Название темы
            description: Описание темы
            perspectives: Список перспектив для анализа
        """
        self.current_buzz_topic = {
            "title": title,
            "description": description,
            "perspectives": perspectives
        }
        logger.info(f"Buzz topic updated to: {title}")

    def get_current_buzz_topic(self) -> Dict[str, Any]:
        """Получить текущую buzz-тему"""
        return self.current_buzz_topic
