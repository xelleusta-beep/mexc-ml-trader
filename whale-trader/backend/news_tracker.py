"""
Kripto haber takibi - CoinGecko API (ücretsiz, API key gerekmez).
"""

import time
import logging
from collections import defaultdict

from config import config

logger = logging.getLogger(__name__)

_news_cache = []
_last_fetch = 0
_symbol_sentiment = {}

# CryptoPanic yok, CoinGecko'nun status_updates endpoint'ini kullan
COINGECKO_NEWS = "https://api.coingecko.com/api/v3/news"
COINGECKO_TRENDING = "https://api.coingecko.com/api/v3/search/trending"


async def fetch_news(client):
    """CoinGecko'dan son haberleri çek (API key gerekmez)."""
    global _news_cache, _last_fetch, _symbol_sentiment

    if time.time() - _last_fetch < 120:
        return _news_cache

    parsed = []

    # CoinGecko News
    try:
        r = await client.get(COINGECKO_NEWS, timeout=10)
        if r.status_code == 200:
            data = r.json()
            for item in (data if isinstance(data, list) else data.get("data", []))[:20]:
                title = item.get("title", "")
                url_link = item.get("url", "")
                parsed.append({
                    "title": title,
                    "url": url_link,
                    "source": "CoinGecko",
                    "ts": time.time(),
                })
    except Exception as e:
        logger.debug(f"CoinGecko news error: {e}")

    # Trending coins (hangi coinler konuşuluyor)
    try:
        r = await client.get(COINGECKO_TRENDING, timeout=10)
        if r.status_code == 200:
            data = r.json()
            coins = data.get("coins", [])
            trending = [c["item"]["symbol"].upper() for c in coins[:10]]
            if trending:
                parsed.append({
                    "title": f"Trending: {', '.join(trending)}",
                    "url": "",
                    "source": "CoinGecko Trending",
                    "ts": time.time(),
                })
    except Exception as e:
        logger.debug(f"CoinGecko trending error: {e}")

    if parsed:
        _news_cache = parsed
        _last_fetch = time.time()
        _symbol_sentiment = _calc_sentiment(parsed)
        logger.info(f"News: {len(parsed)} items")

    return _news_cache


def _calc_sentiment(posts):
    """Haber başlıklarından sentiment çıkar."""
    sentiment = defaultdict(lambda: 0)
    bullish_kw = ["etf", "adoption", "partnership", "listing", "buyback", "upgrade", "approval", "bull", "rally", "surge"]
    bearish_kw = ["hack", "exploit", "ban", "regulation", "crackdown", "fraud", "investigation", "delist", "crash", "dump"]

    for post in posts:
        title = post["title"].lower()
        score = 0
        for kw in bullish_kw:
            if kw in title:
                score += 1
        for kw in bearish_kw:
            if kw in title:
                score -= 1

        for sym in config.TRACKED_SYMBOLS:
            if sym.lower() in title:
                sentiment[sym] += score

    result = {}
    for sym, total in sentiment.items():
        result[sym] = round(max(-1, min(1, total / 3)), 3)
    return result


async def check_symbol_news(client, symbol):
    if not _news_cache or time.time() - _last_fetch > 300:
        await fetch_news(client)
    return _symbol_sentiment.get(symbol, 0)


def get_news_feed(limit=20):
    return _news_cache[:limit]


def get_sentiment_summary():
    return [{"symbol": sym, "sentiment": score} for sym, score in sorted(_symbol_sentiment.items(), key=lambda x: abs(x[1]), reverse=True)][:15]
