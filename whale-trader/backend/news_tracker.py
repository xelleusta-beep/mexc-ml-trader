"""
Kripto haber takibi + sentiment analizi.
CryptoPanic API kullanır (ücretsiz: 100 istek/gün).
"""

import time
import logging
from collections import defaultdict

from config import config

logger = logging.getLogger(__name__)

_news_cache = []
_last_fetch = 0
_symbol_sentiment = {}  # symbol -> average sentiment score (-1 to +1)


async def fetch_news(client):
    """CryptoPanic'ten son haberleri çek."""
    global _news_cache, _last_fetch, _symbol_sentiment

    if not config.CRYPTOPANIC_KEY:
        return []

    # Rate limit: en fazla 5dk'da bir
    if time.time() - _last_fetch < 120:
        return _news_cache

    url = f"https://cryptopanic.com/api/v1/posts/?auth_token={config.CRYPTOPANIC_KEY}&public=true"
    try:
        r = await client.get(url, timeout=10)
        data = r.json()
        results = data.get("results", [])

        parsed = []
        for post in results[:30]:
            title = post.get("title", "")
            url_link = post.get("url", "")
            source = post.get("source", {}).get("title", "unknown")
            kind = post.get("kind", "news")
            domain = post.get("domain", "")
            published = post.get("published_at", "")
            currencies = [c["code"] for c in post.get("currencies", []) if c.get("code")]

            if not currencies:
                # Genel haber, tüm sembolleri etkileyebilir
                currencies = ["GENERAL"]

            parsed.append({
                "title": title,
                "url": url_link,
                "source": source,
                "kind": kind,
                "domain": domain,
                "currencies": currencies,
                "published_at": published,
                "ts": time.time(),
            })

        _news_cache = parsed
        _last_fetch = time.time()

        # Sentiment hesapla
        _symbol_sentiment = _calc_sentiment(parsed)
        logger.info(f"News: {len(parsed)} haber, {len(_symbol_sentiment)} sembol sentiment")

    except Exception as e:
        logger.error(f"News fetch error: {e}")

    return _news_cache


def _calc_sentiment(posts):
    """Haber başlıklarına göre sentiment skoru hesapla."""
    sentiment = defaultdict(lambda: {"score": 0, "count": 0})
    bullish_kw = config.NEWS_KEYWORDS.get("bullish", [])
    bearish_kw = config.NEWS_KEYWORDS.get("bearish", [])

    for post in posts:
        title = post["title"].lower()
        currencies = post["currencies"]

        score = 0
        for kw in bullish_kw:
            if kw in title:
                score += 1
        for kw in bearish_kw:
            if kw in title:
                score -= 1

        if "not" in title:
            score *= -1

        if score != 0:
            for sym in currencies:
                sentiment[sym]["score"] += score
                sentiment[sym]["count"] += 1

    result = {}
    for sym, data in sentiment.items():
        if data["count"] > 0:
            avg = data["score"] / data["count"]
            result[sym] = round(max(-1, min(1, avg)), 3)
        else:
            result[sym] = 0
    return result


async def check_symbol_news(client, symbol):
    """Belirli bir sembol için haber sentiment skoru döndür."""
    if not _news_cache or time.time() - _last_fetch > 300:
        await fetch_news(client)
    return _symbol_sentiment.get(symbol, 0)


def get_news_feed(limit=20):
    """Dashboard için haber akışı."""
    return sorted(_news_cache, key=lambda x: x.get("published_at", ""), reverse=True)[:limit]


def get_sentiment_summary():
    """Dashboard için sentiment özeti."""
    result = []
    for sym, score in sorted(_symbol_sentiment.items(), key=lambda x: abs(x[1]), reverse=True):
        if sym == "GENERAL":
            continue
        result.append({"symbol": sym, "sentiment": score})
    return result[:20]
