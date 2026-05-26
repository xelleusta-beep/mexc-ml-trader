"""
MEXC ML Trading System — Performance Cache v1.0
TTL cache, feature cache, async batch processor.
"""

import time
import hashlib
import json
import asyncio
import logging
from collections import OrderedDict
from typing import Any, Optional, Callable

logger = logging.getLogger(__name__)


# ── TTL Cache ────────────────────────────────────────────────────────────────

class TTLCache:
    """
    Time-To-Live cache with max size.
    En eski entry otomatik silinir (FIFO + TTL).
    """

    def __init__(self, max_size: int = 500, default_ttl: float = 60.0):
        self._max = max_size
        self._ttl = default_ttl
        self._store: OrderedDict = OrderedDict()

    def _is_expired(self, entry: tuple) -> bool:
        _, ts = entry
        return (time.time() - ts) > self._ttl

    def get(self, key: str) -> Optional[Any]:
        entry = self._store.get(key)
        if entry is None:
            return None
        val, ts = entry
        if self._is_expired(entry):
            del self._store[key]
            return None
        # Move to end (most recently used)
        self._store.move_to_end(key)
        return val

    def set(self, key: str, value: Any):
        if len(self._store) >= self._max:
            # Remove oldest
            self._store.popitem(last=False)
        self._store[key] = (value, time.time())

    def invalidate(self, key: str):
        self._store.pop(key, None)

    def invalidate_all(self):
        self._store.clear()

    @property
    def size(self) -> int:
        return len(self._store)


# ── Feature Cache ─────────────────────────────────────────────────────────────

class FeatureCache:
    """
    Feature vektoru cache'i.
    Ayni klines girdisi icin tekrar hesaplamayi onler.
    """

    def __init__(self, max_size: int = 1000, ttl: float = 120.0):
        self._cache = TTLCache(max_size=max_size, default_ttl=ttl)
        self._hits = 0
        self._misses = 0

    def _make_key(self, klines: dict, use_v2: bool = False) -> str:
        """Klines'dan hash key olustur."""
        close = klines.get("close", [])
        if close is None or len(close) == 0:
            return ""
        # Son 3 fiyat + bar sayisi + v2 flag -> benzersiz key
        raw = f"{close[-1]:.8f}_{close[-2] if len(close)>1 else 0:.8f}_{len(close)}_{use_v2}"
        return hashlib.md5(raw.encode()).hexdigest()

    def get(self, klines: dict, use_v2: bool = False) -> Optional[Any]:
        key = self._make_key(klines, use_v2)
        if not key:
            return None
        val = self._cache.get(key)
        if val is not None:
            self._hits += 1
        else:
            self._misses += 1
        return val

    def set(self, klines: dict, features, use_v2: bool = False):
        key = self._make_key(klines, use_v2)
        if key:
            self._cache.set(key, features)

    def invalidate(self, klines: dict, use_v2: bool = False):
        key = self._make_key(klines, use_v2)
        if key:
            self._cache.invalidate(key)

    @property
    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "size":     self._cache.size,
            "hits":     self._hits,
            "misses":   self._misses,
            "hit_rate": round(self._hits / max(1, total) * 100, 1),
        }


# ── Async Batch Processor ────────────────────────────────────────────────────

class AsyncBatchProcessor:
    """
    Toplu async islemleri yonetir.
    - Batch boyutu
    - Aralik (rate limiting)
    - Hata toleransi
    """

    def __init__(self, batch_size: int = 10, interval: float = 0.3,
                 max_concurrent: int = 5):
        self._batch_size = batch_size
        self._interval = interval
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._total_processed = 0
        self._total_errors = 0
        self._total_time_ms = 0.0

    async def process_batch(self, items: list,
                            worker: Callable,
                            item_key: Callable = lambda x: str(x)) -> list:
        """
        Bir listeyi batch'lere bolup asenkron isler.

        items:   islenecek ogeler
        worker:  async callable(item) -> result
        item_key: log icin key fonksiyonu
        """
        results = []
        t0 = time.perf_counter()

        for i in range(0, len(items), self._batch_size):
            batch = items[i:i + self._batch_size]
            batch_results = await self._process_single_batch(batch, worker, item_key)
            results.extend(batch_results)

            if i + self._batch_size < len(items) and self._interval > 0:
                await asyncio.sleep(self._interval)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        self._total_time_ms += elapsed_ms
        self._total_processed += len(items)

        logger.debug(f"Batch: {len(items)} items in {elapsed_ms:.0f}ms "
                     f"({len(items)/max(elapsed_ms,1)*1000:.0f} items/sec)")
        return results

    async def _process_single_batch(self, batch: list,
                                    worker: Callable,
                                    item_key: Callable) -> list:
        """Tek bir batch'i paralel isle."""
        tasks = []
        for item in batch:
            task = self._process_single(item, worker, item_key)
            tasks.append(task)

        return await asyncio.gather(*tasks)

    async def _process_single(self, item, worker: Callable,
                              item_key: Callable) -> Any:
        """Tek bir ogeyi semaphore ile isle."""
        async with self._semaphore:
            try:
                key = item_key(item)
                t0 = time.perf_counter()
                result = await worker(item)
                t_ms = (time.perf_counter() - t0) * 1000
                logger.debug(f"  {key}: {t_ms:.0f}ms")
                return result
            except Exception as e:
                self._total_errors += 1
                logger.error(f"  {item_key(item)}: {e}")
                return None

    @property
    def stats(self) -> dict:
        return {
            "processed":  self._total_processed,
            "errors":     self._total_errors,
            "total_time_ms": round(self._total_time_ms, 1),
            "avg_time_ms": round(self._total_time_ms / max(1, self._total_processed), 1)
            if self._total_processed > 0 else 0,
            "error_rate": round(self._total_errors / max(1, self._total_processed) * 100, 2),
        }


# ── Klines Fetcher (connection-pooled) ────────────────────────────────────────

_http_client: Optional[Any] = None

_HAVE_HTTPX = False
try:
    import httpx
    _HAVE_HTTPX = True
except ImportError:
    pass


async def get_http_client() -> Optional['httpx.AsyncClient']:
    """Singleton HTTPX client with connection pooling and retry mechanism."""
    if not _HAVE_HTTPX:
        return None
    global _http_client
    if _http_client is None or _http_client.is_closed:
        limits = httpx.Limits(max_keepalive_connections=20,
                              max_connections=50,
                              keepalive_expiry=30)
        _http_client = httpx.AsyncClient(
            limits=limits, 
            timeout=10.0,
            # Retry mechanism for connection resilience
            transport=httpx.AsyncHTTPTransport(retries=3)
        )
    return _http_client


async def close_http_client():
    global _http_client
    if _http_client and not _http_client.is_closed:
        await _http_client.aclose()
        _http_client = None


# ── Performance Monitor ──────────────────────────────────────────────────────

import functools


def timed(operation: str = ""):
    """
    Decorator: fonksiyon suresini olcer ve metrics_tracker'a kaydeder.
    Kullanim: @timed("predict") async def predict(...): ...
    """
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # metrics_tracker'i args/kwargs'tan bul
            tracker = None
            for arg in args:
                if hasattr(arg, 'record_latency'):
                    tracker = arg
                    break
            t0 = time.perf_counter()
            try:
                return await func(*args, **kwargs)
            finally:
                ms = (time.perf_counter() - t0) * 1000
                if tracker:
                    tracker.record_latency(operation or func.__name__, ms)
        return async_wrapper
    return decorator
