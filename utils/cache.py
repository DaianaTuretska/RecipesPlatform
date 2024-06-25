from django.utils import timezone


class CacheStorage:
    def __init__(self, lifetime):
        self.lifetime = lifetime
        self._cache_dict = {}

    def get_value(self, key):
        if key in self._cache_dict:
            cache = self._cache_dict[key]
            if cache.is_expired():
                self._cache_dict.pop(key)
            else:
                return cache.data
        return None

    def add_value(self, key, data):
        self._cache_dict[key] = Cache(data, self.lifetime)


class Cache:
    def __init__(self, data, lifetime):
        """value 'lifetime' uses minutes as a unit of measurement"""
        self.data = data
        self._last_update = timezone.now()
        self.lifetime = lifetime

    def is_expired(self):
        return (timezone.now() - self._last_update).total_seconds() / 60 > self.lifetime
