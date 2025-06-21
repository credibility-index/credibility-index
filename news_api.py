import os
import requests
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

class NewsAPI:
    def __init__(self):
        self.api_key = os.getenv('NEWS_API_KEY')
        if not self.api_key:
            raise ValueError("NEWS_API_KEY environment variable is not set")
        self.base_url = "https://newsapi.org/v2"
        self.session = requests.Session()  # Используем сессию для повторного использования соединений

    def get_everything(self, query: str, language: str = 'en', page_size: int = 5,
                      sources: Optional[str] = None, domains: Optional[str] = None) -> Optional[List[Dict]]:
        """Получить новостные статьи из News API с дополнительными параметрами"""
        try:
            params = {
                'q': query,
                'language': language,
                'pageSize': page_size,
                'apiKey': self.api_key,
                'sortBy': 'publishedAt'
            }

            # Добавляем дополнительные параметры, если они предоставлены
            if sources:
                params['sources'] = sources
            if domains:
                params['domains'] = domains

            response = self.session.get(f"{self.base_url}/everything", params=params)
            response.raise_for_status()

            data = response.json()
            if data['status'] == 'ok':
                return data['articles']
            logger.error(f"API returned error: {data.get('message', 'Unknown error')}")
            return None

        except requests.exceptions.RequestException as e:
            logger.error(f"Request error fetching news: {str(e)}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching news: {str(e)}", exc_info=True)
            return None

    def get_top_headlines(self, country: str = 'us', category: str = 'general',
                         page_size: int = 5, sources: Optional[str] = None) -> Optional[List[Dict]]:
        """Получить топ заголовки из News API с дополнительными параметрами"""
        try:
            params = {
                'country': country,
                'category': category,
                'pageSize': page_size,
                'apiKey': self.api_key
            }

            if sources:
                params['sources'] = sources

            response = self.session.get(f"{self.base_url}/top-headlines", params=params)
            response.raise_for_status()

            data = response.json()
            if data['status'] == 'ok':
                return data['articles']
            logger.error(f"API returned error: {data.get('message', 'Unknown error')}")
            return None

        except requests.exceptions.RequestException as e:
            logger.error(f"Request error fetching headlines: {str(e)}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching headlines: {str(e)}", exc_info=True)
            return None
