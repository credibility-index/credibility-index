import os
import requests
import logging
from typing import Dict, Optional, List
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

class NewsAPI:
    def __init__(self, api_key: str = None):
        """
        Инициализация клиента NewsAPI.org.

        :param api_key: Ключ API для NewsAPI.org. Если не указан, берется из переменной окружения NEWS_API_KEY.
        """
        self.api_key = api_key or os.getenv('NEWS_API_KEY')
        if not self.api_key:
            raise ValueError("NewsAPI key is required. Set NEWS_API_KEY environment variable or pass it to constructor.")

        self.base_url = "https://newsapi.org/v2"
        self.user_agent = "MediaCredibilityAnalyzer/1.0"

    def get_article_by_url(self, url: str) -> Optional[Dict]:
        """
        Получение статьи по URL. Обратите внимание, что NewsAPI.org не предоставляет
        прямой доступ к статьям по URL в своем API, поэтому этот метод пытается
        найти статью в результатах поиска по домену и части URL.

        :param url: URL статьи для поиска
        :return: Словарь с данными статьи или None, если не найдено
        """
        try:
            domain = urlparse(url).netloc
            # Извлекаем путь из URL для поиска
            path = urlparse(url).path.strip('/')
            if path:
                # Берем последние 2-3 части пути как ключевые слова для поиска
                parts = [p for p in path.split('/') if p]
                keywords = ' '.join(parts[-3:]) if len(parts) > 3 else path.replace('/', ' ')

                # Ищем новости с этим доменом и ключевыми словами
                params = {
                    'q': keywords,
                    'domains': domain.replace('www.', ''),
                    'pageSize': 5,  # Ограничиваем количество результатов
                    'sortBy': 'publishedAt',
                    'apiKey': self.api_key
                }

                response = requests.get(f"{self.base_url}/everything", params=params)
                response.raise_for_status()
                data = response.json()

                if data.get('articles'):
                    # Ищем статью с наиболее похожим URL
                    for article in data['articles']:
                        if article['url'] == url:
                            return self._process_article_data(article)

                    # Если точного совпадения нет, берем первую статью
                    return self._process_article_data(data['articles'][0])

                return None

        except Exception as e:
            logger.error(f"Error fetching article by URL: {str(e)}")
            return None

    def _process_article_data(self, article_data: Dict) -> Dict:
        """
        Обработка данных статьи из NewsAPI для приведения к единому формату.
        """
        return {
            'title': article_data.get('title', 'No title available'),
            'authors': article_data.get('author', 'Unknown author'),
            'published_date': article_data.get('publishedAt'),
            'text': article_data.get('description', '') + '\n\n' + article_data.get('content', ''),
            'source': article_data.get('source', {}).get('name', 'Unknown source'),
            'url': article_data.get('url', ''),
            'top_image': article_data.get('urlToImage')
        }

    def get_top_headlines(self, country: str = 'us', category: str = None,
                         sources: str = None, page_size: int = 20) -> Dict:
        """
        Получение топовых новостей.

        :param country: Код страны (например, 'us', 'gb', 'ru')
        :param category: Категория новостей (бизнес, развлечения и т.д.)
        :param sources: Идентификаторы источников, разделенные запятыми
        :param page_size: Количество новостей для возврата (макс. 100)
        :return: Словарь с результатами
        """
        try:
            params = {
                'country': country,
                'pageSize': min(page_size, 100),  # Максимум 100 по API
                'apiKey': self.api_key
            }

            if category:
                params['category'] = category
            if sources:
                params['sources'] = sources

            response = requests.get(f"{self.base_url}/top-headlines", params=params)
            response.raise_for_status()
            data = response.json()

            return {
                'status': 'success',
                'articles': [self._process_article_data(article) for article in data.get('articles', [])],
                'total_results': data.get('totalResults', 0)
            }

        except Exception as e:
            logger.error(f"Error fetching top headlines: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }

    def search_news(self, query: str, from_date: str = None, to_date: str = None,
                   language: str = 'en', sort_by: str = 'publishedAt',
                   page_size: int = 20) -> Dict:
        """
        Поиск новостей по ключевым словам.

        :param query: Поисковый запрос
        :param from_date: Дата начала в формате YYYY-MM-DD
        :param to_date: Дата окончания в формате YYYY-MM-DD
        :param language: Код языка (например, 'en', 'ru')
        :param sort_by: Поле для сортировки (publishedAt, relevancy, popularity)
        :param page_size: Количество результатов
        :return: Словарь с результатами поиска
        """
        try:
            params = {
                'q': query,
                'language': language,
                'sortBy': sort_by,
                'pageSize': min(page_size, 100),  # Максимум 100 по API
                'apiKey': self.api_key
            }

            if from_date:
                params['from'] = from_date
            if to_date:
                params['to'] = to_date

            response = requests.get(f"{self.base_url}/everything", params=params)
            response.raise_for_status()
            data = response.json()

            return {
                'status': 'success',
                'articles': [self._process_article_data(article) for article in data.get('articles', [])],
                'total_results': data.get('totalResults', 0)
            }

        except Exception as e:
            logger.error(f"Error searching news: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }

    def get_sources(self, category: str = None, language: str = 'en',
                   country: str = None) -> Dict:
        """
        Получение списка доступных источников.

        :param category: Категория источника
        :param language: Код языка
        :param country: Код страны
        :return: Словарь с источниками
        """
        try:
            params = {
                'apiKey': self.api_key
            }

            if category:
                params['category'] = category
            if language:
                params['language'] = language
            if country:
                params['country'] = country

            response = requests.get(f"{self.base_url}/top-headlines/sources", params=params)
            response.raise_for_status()
            data = response.json()

            return {
                'status': 'success',
                'sources': data.get('sources', []),
                'total': len(data.get('sources', []))
            }

        except Exception as e:
            logger.error(f"Error getting sources: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }
