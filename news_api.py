import os
import requests
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

class NewsAPI:
    def __init__(self):
        self.api_key = os.getenv('NEWS_API_KEY')
        self.base_url = "https://newsapi.org/v2"

    def get_everything(self, query: str, language: str = 'en', page_size: int = 5) -> Optional[List[Dict]]:
        """Get news articles from News API"""
        try:
            params = {
                'q': query,
                'language': language,
                'pageSize': page_size,
                'apiKey': self.api_key
            }

            response = requests.get(f"{self.base_url}/everything", params=params)
            response.raise_for_status()

            data = response.json()
            if data['status'] == 'ok':
                return data['articles']
            return None

        except Exception as e:
            logger.error(f"Error fetching news from News API: {str(e)}")
            return None

    def get_top_headlines(self, country: str = 'us', category: str = 'general', page_size: int = 5) -> Optional[List[Dict]]:
        """Get top headlines from News API"""
        try:
            params = {
                'country': country,
                'category': category,
                'pageSize': page_size,
                'apiKey': self.api_key
            }

            response = requests.get(f"{self.base_url}/top-headlines", params=params)
            response.raise_for_status()

            data = response.json()
            if data['status'] == 'ok':
                return data['articles']
            return None

        except Exception as e:
            logger.error(f"Error fetching headlines from News API: {str(e)}")
            return None
