import os
import redis
import json
import logging
from datetime import datetime, timedelta
from typing import Optional, Any, Dict

logger = logging.getLogger(__name__)

class CacheManager:
    def __init__(self):
        """Инициализирует менеджер кэша"""
        self.redis_conn = None
        self.use_redis = os.getenv('USE_REDIS', 'false').lower() == 'true'

        if self.use_redis:
            try:
                self.redis_conn = redis.Redis(
                    host=os.getenv('REDIS_HOST', 'localhost'),
                    port=int(os.getenv('REDIS_PORT', '6379')),
                    password=os.getenv('REDIS_PASSWORD', None),
                    db=int(os.getenv('REDIS_DB', '0')),
                    decode_responses=True
                )
                # Проверяем соединение
                self.redis_conn.ping()
                logger.info("Connected to Redis cache")
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {str(e)}")
                self.use_redis = False

        # Локальный кэш для разработки/тестирования
        self.local_cache = {}

    def get(self, key: str) -> Optional[Any]:
        """Получает значение из кэша"""
        if self.use_redis and self.redis_conn:
            try:
                cached_value = self.redis_conn.get(key)
                if cached_value:
                    return json.loads(cached_value)
            except Exception as e:
                logger.warning(f"Redis get error: {str(e)}")
                return None

        # Локальный кэш
        value = self.local_cache.get(key)
        if value and isinstance(value, dict) and 'expiry' in value:
            if datetime.now() < datetime.fromisoformat(value['expiry']):
                return value['data']
            else:
                self.delete(key)
        return None

    def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Устанавливает значение в кэш с TTL (в секундах)"""
        if not key or not value:
            return False

        if isinstance(value, (dict, list)):
            data_to_cache = json.dumps(value)
        else:
            data_to_cache = str(value)

        if self.use_redis and self.redis_conn:
            try:
                self.redis_conn.setex(key, ttl, data_to_cache)
                return True
            except Exception as e:
                logger.error(f"Redis set error: {str(e)}")
                return False

        # Локальный кэш
        expiry_time = (datetime.now() + timedelta(seconds=ttl)).isoformat()
        self.local_cache[key] = {'data': value, 'expiry': expiry_time}
        return True

    def delete(self, key: str) -> bool:
        """Удаляет значение из кэша"""
        if self.use_redis and self.redis_conn:
            try:
                self.redis_conn.delete(key)
            except Exception as e:
                logger.error(f"Redis delete error: {str(e)}")
                return False

        if key in self.local_cache:
            del self.local_cache[key]
        return True

    def cache_key(self, prefix: str, *args) -> str:
        """Создает стандартный ключ кэша"""
        key_parts = [prefix] + [str(arg) for arg in args]
        return ":".join(key_parts)

    def cache_article_analysis(self, url_or_text: str, analysis_data: Dict) -> bool:
        """Кэширует результаты анализа статьи"""
        key = self.cache_key("analysis", url_or_text[:50])  # Используем первые 50 символов как ключ
        return self.set(key, analysis_data)

    def get_cached_article_analysis(self, url_or_text: str) -> Optional[Dict]:
        """Получает закэшированный анализ статьи"""
        key = self.cache_key("analysis", url_or_text[:50])
        return self.get(key)

    def cache_buzz_analysis(self, analysis_data: Dict) -> bool:
        """Кэширует результаты buzz анализа"""
        key = self.cache_key("buzz", "daily")
        return self.set(key, analysis_data, ttl=86400)  # Кэшируем на 24 часа

    def get_cached_buzz_analysis(self) -> Optional[Dict]:
        """Получает закэшированный buzz анализ"""
        key = self.cache_key("buzz", "daily")
        return self.get(key)
