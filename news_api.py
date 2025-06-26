import os
import logging
from typing import Dict, Optional, List, Union
from datetime import datetime
import time
import requests
from urllib.parse import urlparse

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NewsAPI:
    def __init__(self, api_key: str = None):
        """
        Инициализация клиента NewsAPI.org.
        Args:
            api_key: Ключ API для NewsAPI.org. Если не указан, берется из переменной окружения NEWS_API_KEY.
        """
        self.api_key = api_key or os.getenv('NEWS_API_KEY')
        if not self.api_key:
            raise ValueError("NewsAPI key is required. Set NEWS_API_KEY environment variable or pass it to constructor.")

        self.base_url = os.getenv('NEWS_ENDPOINT', 'https://newsapi.org/v2')
        self.user_agent = "MediaCredibilityAnalyzer/1.0"
        self.max_retries = 3
        self.retry_delay = 1

        # Настройка сессии с повторными попытками
        self.session = requests.Session()
        retries = requests.adapters.HTTPAdapter(max_retries=3)
        self.session.mount("http://", retries)
        self.session.mount("https://", retries)

    def _make_request(self, endpoint: str, params: Dict) -> Optional[Dict]:
        """
        Вспомогательный метод для выполнения запросов к API с повторными попытками.
        Args:
            endpoint: Конечная точка API (например, '/everything')
            params: Параметры запроса
        Returns:
            Словарь с данными ответа или None в случае ошибки
        """
        params['apiKey'] = self.api_key

        try:
            response = self.session.get(
                f"{self.base_url}{endpoint}",
                params=params,
                headers={'User-Agent': self.user_agent},
                timeout=15
            )

            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'ok':
                    return data
                else:
                    logger.error(f"API error: {data.get('message', 'Unknown error')}")
                    return None
            elif response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', 5))
                logger.warning(f"Rate limited. Retrying after {retry_after} seconds")
                time.sleep(retry_after)
                return self._make_request(endpoint, params)  # Рекурсивный повтор
            else:
                logger.error(f"HTTP error: {response.status_code}")
                return None

        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {str(e)}")
            return None

    def get_everything(self, query: str, page_size: int = 5, **kwargs) -> List[Dict[str, Any]]:
        """
        Получение статей по запросу.
        Args:
            query: Поисковый запрос
            page_size: Количество статей для возврата
            **kwargs: Дополнительные параметры запроса
        Returns:
            Список статей или пустой список в случае ошибки
        """
        params = {
            'q': query,
            'pageSize': page_size,
            **kwargs
        }

        data = self._make_request('/everything', params)
        if not data or 'articles' not in data:
            return []

        return data['articles']

    def get_article_by_url(self, url: str) -> Optional[Dict]:
        """
        Получение статьи по URL.
        Args:
            url: URL статьи для поиска
        Returns:
            Словарь с данными статьи или None, если не найдено
        """
        if not url:
            logger.error("Empty URL provided")
            return None

        try:
            parsed_url = urlparse(url)
            domain = parsed_url.netloc.replace('www.', '')

            params = {
                'domains': domain,
                'pageSize': 1,
                'sortBy': 'publishedAt'
            }

            data = self._make_request('/everything', params)
            if not data or 'articles' not in data:
                return None

            for article in data['articles']:
                if article.get('url') == url:
                    return article

            return None

        except Exception as e:
            logger.error(f"Error in get_article_by_url: {str(e)}")
            return None

    def get_top_headlines(self, country: str = 'us', category: str = None,
                        sources: str = None, page_size: int = 20) -> Dict:
        """
        Получение топовых новостей.
        Args:
            country: Код страны
            category: Категория новостей
            sources: Источники новостей
            page_size: Количество новостей
        Returns:
            Словарь с результатами
        """
        params = {
            'country': country,
            'pageSize': page_size,
        }

        if category:
            params['category'] = category
        if sources:
            params['sources'] = sources

        data = self._make_request('/top-headlines', params)
        if not data:
            return {'status': 'error', 'message': 'Failed to fetch top headlines'}

        return {
            'status': 'success',
            'articles': data.get('articles', []),
            'total_results': data.get('totalResults', 0)
        }

    def search_news(self, query: str, from_date: str = None, to_date: str = None,
                 language: str = 'en', sort_by: str = 'publishedAt',
                 page_size: int = 20, domains: List[str] = None) -> Dict:
        """
        Поиск новостей по ключевым словам.
        Args:
            query: Поисковый запрос
            from_date: Начальная дата
            to_date: Конечная дата
            language: Язык
            sort_by: Поле для сортировки
            page_size: Количество результатов
            domains: Домены для поиска
        Returns:
            Словарь с результатами поиска
        """
        if not query:
            return {'status': 'error', 'message': 'Query parameter is required'}

        params = {
            'q': query,
            'language': language,
            'sortBy': sort_by,
            'pageSize': page_size,
        }

        if from_date:
            params['from'] = from_date
        if to_date:
            params['to'] = to_date
        if domains:
            params['domains'] = ','.join(domains)

        data = self._make_request('/everything', params)
        if not data:
            return {'status': 'error', 'message': 'Failed to search news'}

        return {
            'status': 'success',
            'articles': data.get('articles', []),
            'total_results': data.get('totalResults', 0)
        }

    def get_sources(self, category: str = None, language: str = 'en',
                 country: str = None) -> Dict:
        """
        Получение списка доступных источников.
        Args:
            category: Категория источника
            language: Язык
            country: Страна
        Returns:
            Словарь с источниками
        """
        params = {}

        if category:
            params['category'] = category
        if language:
            params['language'] = language
        if country:
            params['country'] = country

        data = self._make_request('/sources', params)
        if not data:
            return {'status': 'error', 'message': 'Failed to fetch sources'}

        return {
            'status': 'success',
            'sources': data.get('sources', []),
            'total': len(data.get('sources', []))
        }
