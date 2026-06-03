"""
Kripto haber takibi - RSS (ücretsiz, API key gerekmez).
CoinGecko kapandı, RSS feed'ler ile devam.
"""

import re
import time
import logging
from collections import defaultdict

from config import config

logger = logging.getLogger(__name__)

_news_cache = []
_last_fetch = 0
_symbol_sentiment = {}

# RSS kaynakları (API key gerektirmez)
RSS_FEEDS = [
    "https://cointelegraph.com/rss",
    "https://cointelegraph.com/editors.rss",
    "https://cryptonews.com/news/feed/",
    "https://www.newsbtc.com/feed/",
]


async def fetch_news(client):
    """RSS feed'lerden haber çek."""
    global _news_cache, _last_fetch, _symbol_sentiment

    if time.time() - _last_fetch < 120:
        return _news_cache

    parsed = []
    for feed_url in RSS_FEEDS:
        try:
            r = await client.get(feed_url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code != 200:
                logger.debug(f"RSS {feed_url}: {r.status_code}")
                continue

            text = r.text
            is_atom = "<feed" in text

            items = []
            if is_atom:
                # Atom: namespace'yi temizle, basit string parse
                import re
                titles = re.findall(r'<entry>.*?<title[^>]*>(.*?)</title>', text, re.DOTALL)
                for t in titles[:15]:
                    parsed.append({
                        "title": t.strip()[:200],
                        "source": feed_url.split("//")[1].split("/")[0],
                        "ts": time.time(),
                    })
            else:
                # RSS 2.0: regex ile title'ları çek
                import re
                titles = re.findall(r'<item>.*?<title[^>]*>(.*?)</title>', text, re.DOTALL)
                for t in titles[:15]:
                    parsed.append({
                        "title": t.strip()[:200],
                        "source": feed_url.split("//")[1].split("/")[0],
                        "ts": time.time(),
                    })
        except Exception as e:
            logger.debug(f"RSS error {feed_url}: {e}")

    if parsed:
        _news_cache = parsed
        _last_fetch = time.time()
        _symbol_sentiment = _calc_sentiment(parsed)
        logger.info(f"News: {len(parsed)} items from RSS")
    else:
        logger.warning("News fetch returned 0 items")

    return _news_cache


def _calc_sentiment(posts):
    sentiment = defaultdict(lambda: 0)
    bullish_kw = ["etf", "adoption", "partnership", "listing", "buyback", "upgrade", "approval", "bull", "rally", "surge", "ath", "breakthrough", "positive"]
    bearish_kw = ["hack", "exploit", "ban", "regulation", "crackdown", "fraud", "investigation", "delist", "crash", "dump", "bear", "restriction", "fine"]

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
