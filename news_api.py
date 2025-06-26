import os
import requests
import logging
from typing import Dict, Optional, List, Union
from urllib.parse import urlparse
from datetime import datetime
import time

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
        self.base_url = "https://newsapi.org/v2"
        self.user_agent = "MediaCredibilityAnalyzer/1.0"
        self.max_retries = 3
        self.retry_delay = 1

    def _make_request(self, endpoint: str, params: Dict) -> Dict:
        """
        Вспомогательный метод для выполнения запросов к API с повторными попытками.

        Args:
            endpoint: Конечная точка API (например, '/everything')
            params: Параметры запроса

        Returns:
            Словарь с данными ответа или None в случае ошибки
        """
        params['apiKey'] = self.api_key
        retry_count = 0

        while retry_count < self.max_retries:
            try:
                response = requests.get(
                    f"{self.base_url}{endpoint}",
                    params=params,
                    headers={'User-Agent': self.user_agent},
                    timeout=15
                )

                # Проверяем статус код
                if response.status_code == 200:
                    data = response.json()
                    if data.get('status') == 'ok':
                        return data
                    elif data.get('status') == 'error':
                        logger.error(f"NewsAPI error: {data.get('message', 'Unknown error')}")
                        return None
                    else:
                        logger.error(f"Unexpected response status: {data.get('status')}")
                        return None
                elif response.status_code == 401:
                    logger.error("Unauthorized - check your API key")
                    return None
                elif response.status_code == 429:
                    # Too many requests - ждем и повторяем
                    retry_after = int(response.headers.get('Retry-After', 5))
                    logger.warning(f"Rate limited. Retrying after {retry_after} seconds")
                    time.sleep(retry_after)
                    retry_count += 1
                    continue
                else:
                    logger.error(f"HTTP error: {response.status_code}")
                    return None

            except requests.exceptions.RequestException as e:
                retry_count += 1
                if retry_count < self.max_retries:
                    logger.warning(f"Request failed, retrying... ({retry_count}/{self.max_retries})")
                    time.sleep(self.retry_delay)
                    continue
                logger.error(f"Request failed after {self.max_retries} retries: {str(e)}")
                return None
            except Exception as e:
                logger.error(f"Unexpected error: {str(e)}")
                return None

        return None

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
            path = parsed_url.path.strip('/')

            if not domain:
                logger.error("Invalid URL format - no domain found")
                return None

            # Извлекаем ключевые слова из пути URL
            keywords = []
            if path:
                parts = [p for p in path.split('/') if p]
                if len(parts) > 3:
                    keywords = parts[-3:]
                else:
                    keywords = parts

            # Формируем запрос
            params = {
                'domains': domain,
                'pageSize': 5,
                'sortBy': 'publishedAt'
            }

            if keywords:
                params['q'] = ' '.join(keywords)

            data = self._make_request('/everything', params)
            if not data or 'articles' not in data:
                return None

            # Ищем статью с точным совпадением URL
            for article in data['articles']:
                if article.get('url') == url:
                    return self._process_article_data(article)

            # Если точного совпадения нет, берем первую статью
            if data['articles']:
                return self._process_article_data(data['articles'][0])

            return None

        except Exception as e:
            logger.error(f"Error in get_article_by_url: {str(e)}")
            return None

    def _process_article_data(self, article_data: Dict) -> Dict:
        """
        Обработка данных статьи из NewsAPI для приведения к единому формату.

        Args:
            article_data: Исходные данные статьи из API

        Returns:
            Словарь с обработанными данными статьи
        """
        if not article_data:
            return {}

        processed = {
            'title': article_data.get('title', 'No title available'),
            'authors': article_data.get('author', 'Unknown author'),
            'published_date': article_data.get('publishedAt'),
            'text': '',
            'source': article_data.get('source', {}).get('name', 'Unknown source'),
            'url': article_data.get('url', ''),
            'top_image': article_data.get('urlToImage'),
            'raw_data': article_data  # сохраняем исходные данные
        }

        # Объединяем description и content
        description = article_data.get('description', '')
        content = article_data.get('content', '')

        if description and content:
            processed['text'] = f"{description}\n\n{content}"
        elif description:
            processed['text'] = description
        else:
            processed['text'] = content

        # Добавляем метаданные
        processed['metadata'] = {
            'language': article_data.get('language', 'en'),
            'country': article_data.get('country'),
            'category': article_data.get('category'),
            'published_at': article_data.get('publishedAt'),
            'retrieved_at': datetime.utcnow().isoformat()
        }

        return processed

    def get_top_headlines(self, country: str = 'us', category: str = None,
                         sources: str = None, page_size: int = 20) -> Dict:
        """
        Получение топовых новостей.

        Args:
            country: Код страны (например, 'us', 'gb', 'ru')
            category: Категория новостей (бизнес, развлечения и т.д.)
            sources: Идентификаторы источников, разделенные запятыми
            page_size: Количество новостей для возврата (макс. 100)

        Returns:
            Словарь с результатами
        """
        params = {
            'country': country,
            'pageSize': min(page_size, 100),  # Максимум 100 по API
        }

        if category:
            params['category'] = category
        if sources:
            params['sources'] = sources

        data = self._make_request('/top-headlines', params)
        if not data:
            return {
                'status': 'error',
                'message': 'Failed to fetch top headlines'
            }

        return {
            'status': 'success',
            'articles': [self._process_article_data(article) for article in data.get('articles', [])],
            'total_results': data.get('totalResults', 0)
        }

    def search_news(self, query: str, from_date: str = None, to_date: str = None,
                   language: str = 'en', sort_by: str = 'publishedAt',
                   page_size: int = 20, domains: List[str] = None) -> Dict:
        """
        Поиск новостей по ключевым словам.

        Args:
            query: Поисковый запрос
            from_date: Дата начала в формате YYYY-MM-DD
            to_date: Дата окончания в формате YYYY-MM-DD
            language: Код языка (например, 'en', 'ru')
            sort_by: Поле для сортировки (publishedAt, relevancy, popularity)
            page_size: Количество результатов
            domains: Список доменов для поиска

        Returns:
            Словарь с результатами поиска
        """
        if not query:
            return {
                'status': 'error',
                'message': 'Query parameter is required'
            }

        params = {
            'q': query,
            'language': language,
            'sortBy': sort_by,
            'pageSize': min(page_size, 100),  # Максимум 100 по API
        }

        if from_date:
            params['from'] = from_date
        if to_date:
            params['to'] = to_date
        if domains:
            params['domains'] = ','.join(domains)

        data = self._make_request('/everything', params)
        if not data:
            return {
                'status': 'error',
                'message': 'Failed to search news'
            }

        return {
            'status': 'success',
            'articles': [self._process_article_data(article) for article in data.get('articles', [])],
            'total_results': data.get('totalResults', 0)
        }

    def get_sources(self, category: str = None, language: str = 'en',
                   country: str = None) -> Dict:
        """
        Получение списка доступных источников.

        Args:
            category: Категория источника
            language: Код языка
            country: Код страны

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

        data = self._make_request('/top-headlines/sources', params)
        if not data:
            return {
                'status': 'error',
                'message': 'Failed to fetch sources'
            }

        return {
            'status': 'success',
            'sources': data.get('sources', []),
            'total': len(data.get('sources', []))
        }

    def get_article_details(self, url: str) -> Dict:
        """
        Получение полных деталей статьи. Пытается получить как можно больше информации.
        Использует get_article_by_url и дополняет информацию из других источников.

        Args:
            url: URL статьи

        Returns:
            Словарь с полными данными статьи
        """
        article = self.get_article_by_url(url)
        if not article:
            return {
                'status': 'error',
                'message': 'Article not found'
            }

        # Здесь можно добавить дополнительную логику для получения
        # более полной информации о статье из других источников

        return {
            'status': 'success',
            'article': article
        }
    import os
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnhancedNewsAPI:
    """Enhanced version of NewsAPI with robust error handling"""

    def __init__(self):
        self.api_key = os.getenv('NEWS_API_KEY')
        self.base_url = "https://newsapi.org/v2"
        self.fallback_articles = [
            {
                "title": "Technology News (Fallback)",
                "source": {"name": "Tech Demo News", "url": None},
                "url": None,
                "description": "Fallback data about technology trends",
                "publishedAt": datetime.now().isoformat(),
                "content": "This is a fallback article shown when there are temporary issues with the API."
            },
            {
                "title": "Economic Overview (Fallback)",
                "source": {"name": "Econ Demo", "url": None},
                "url": None,
                "description": "Fallback data about economic indicators",
                "publishedAt": datetime.now().isoformat(),
                "content": "This is a fallback article shown when there are temporary issues with the API."
            }
        ]
        self.max_retries = 3
        self.retry_delay = 1

    def get_everything(self, query: str, page_size: int = 5, **kwargs) -> List[Dict[str, Any]]:
        """Gets articles with error handling and retries"""
        retry_count = 0
        last_error = None

        if not self.api_key:
            logger.error("NEWS_API_KEY is not set")
            return self._get_fallback_articles(query, page_size)

        while retry_count < self.max_retries:
            try:
                params = {
                    'q': query,
                    'pageSize': page_size,
                    'apiKey': self.api_key,
                    **kwargs
                }
                url = f"{self.base_url}/everything"

                # Configure session with retries
                session = requests.Session()
                retries = Retry(
                    total=3,
                    backoff_factor=1,
                    status_forcelist=[500, 502, 503, 504]
                )
                adapter = HTTPAdapter(max_retries=retries)
                session.mount("http://", adapter)
                session.mount("https://", adapter)

                response = session.get(url, params=params, timeout=15)

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
                                logger.error(f"Error processing article: {str(e)}")
                                continue

                        if processed_articles:
                            return processed_articles[:page_size]

                        return self._get_fallback_articles(query, page_size)

                    logger.warning(f"NewsAPI returned non-ok status: {data.get('message', 'No message')}")
                    return self._get_fallback_articles(query, page_size)

                elif response.status_code == 404:
                    logger.error("NewsAPI returned 404 - invalid URL or resource not found")
                    return self._get_fallback_articles(query, page_size)

                elif response.status_code in (500, 502, 503, 504):
                    retry_count += 1
                    wait_time = self.retry_delay * (2 ** (retry_count - 1))
                    logger.warning(f"Server error. Retrying in {wait_time} seconds")
                    time.sleep(wait_time)
                    continue

                else:
                    logger.error(f"NewsAPI returned error {response.status_code}")
                    return self._get_fallback_articles(query, page_size)

            except requests.exceptions.RequestException as e:
                retry_count += 1
                wait_time = self.retry_delay * (2 ** (retry_count - 1))
                logger.warning(f"Connection error. Retrying in {wait_time} seconds")
                time.sleep(wait_time)
                continue

            except Exception as e:
                logger.error(f"Unexpected error: {str(e)}")
                return self._get_fallback_articles(query, page_size)

        logger.error(f"All retries exhausted. Last error: {str(last_error)}")
        return self._get_fallback_articles(query, page_size)

    def _get_fallback_articles(self, query: str, count: int) -> List[Dict[str, Any]]:
        """Returns fallback articles when API fails"""
        mock_articles = [
            {
                "title": f"Article about {query} (fallback)",
                "source": {"name": f"{query.capitalize()} News"},
                "url": f"https://example.com/{query.replace(' ', '-')}-1",
                "description": f"Fallback article about {query}",
                "publishedAt": datetime.now().isoformat(),
                "content": f"This is a fallback article about {query}. In a real system, this would contain actual news content."
            },
            {
                "title": f"Analysis of {query} (fallback)",
                "source": {"name": f"{query.capitalize()} Analysis"},
                "url": f"https://example.com/{query.replace(' ', '-')}-2",
                "description": f"Fallback analysis of {query}",
                "publishedAt": datetime.now().isoformat(),
                "content": f"This is a fallback analytical article about {query}. In a real system, this would contain professional analysis."
            }
        ]
        return mock_articles[:count]

    def analyze_article(self, url: str) -> Dict:
        """
        Анализ статьи на основе ее URL.

        Args:
            url: URL статьи для анализа

        Returns:
            Словарь с результатами анализа
        """
        article_data = self.get_article_details(url)
        if article_data['status'] != 'success':
            return {
                'status': 'error',
                'message': article_data.get('message', 'Failed to get article data')
            }

        article = article_data['article']

        # Здесь можно добавить логику анализа статьи
        # Например, оценку достоверности, тональности и т.д.

        analysis = {
            'credibility_score': 0.7,  # Пример значения
            'sentiment': 'neutral',
            'bias': 'neutral',
            'topics': [],
            'summary': article.get('text', '')[:200] + '...' if len(article.get('text', '')) > 200 else article.get('text', '')
        }

        return {
            'status': 'success',
            'article': article,
            'analysis': analysis
        }
