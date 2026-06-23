import time
import httpx
from .base_agent import BaseAgent


class SentimentAgent(BaseAgent):
    def __init__(self):
        super().__init__("Sentiment")
        self.fear_greed_index = 50
        self.fear_greed_label = "Neutral"
        self.news_sentiment = 0.0
        self.social_sentiment = 0.0
        self.overall_sentiment = 0.0
        self.trending_coins: list[str] = []
        self.last_fng_update = 0
        self._client: httpx.AsyncClient | None = None
        self.news_headlines: list[dict] = []

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=15, limits=httpx.Limits(max_connections=20))
        return self._client

    async def analyze(self, data: dict) -> dict:
        try:
            self.update_status("running")
            client = await self._get_client()

            await self._fetch_fear_greed(client)
            await self._fetch_news_sentiment(client)

            self.social_sentiment = self._estimate_social_sentiment(data)
            self.overall_sentiment = self._calculate_overall()
            self.trending_coins = self._extract_trending(data)

            self.update_status("ready")
            return {
                "fear_greed_index": self.fear_greed_index,
                "fear_greed_label": self.fear_greed_label,
                "fear_greed_history": self._get_fear_greed_context(),
                "news_sentiment": self.news_sentiment,
                "news_headlines": self.news_headlines[:10],
                "news_stats": self._get_news_stats(),
                "social_sentiment": self.social_sentiment,
                "overall_sentiment": self.overall_sentiment,
                "trending_coins": self.trending_coins,
                "market_mood": self._get_market_mood(),
                "mood_analysis": self._get_mood_analysis(),
                "thinking": self._build_thinking(),
            }

        except Exception as e:
            self.update_status("error", str(e))
            return {
                "fear_greed_index": self.fear_greed_index,
                "fear_greed_label": self.fear_greed_label,
                "overall_sentiment": self.overall_sentiment,
                "error": str(e),
                "thinking": [f"Hata olustu: {str(e)}"],
            }

    async def _fetch_fear_greed(self, client: httpx.AsyncClient):
        if time.time() - self.last_fng_update < 3600:
            return
        try:
            resp = await client.get("https://api.alternative.me/fng/?limit=7")
            resp.raise_for_status()
            data = resp.json()
            entries = data.get("data", [])
            if entries:
                entry = entries[0]
                self.fear_greed_index = int(entry.get("value", 50))
                self.fear_greed_label = entry.get("value_classification", "Neutral")
                self.last_fng_update = time.time()
        except Exception:
            pass

    async def _fetch_news_sentiment(self, client: httpx.AsyncClient):
        try:
            resp = await client.get(
                "https://min-api.cryptocompare.com/data/v2/news/?lang=EN&limit=20"
            )
            resp.raise_for_status()
            data = resp.json()
            articles = data.get("Data", [])

            positive_keywords = ["surge", "rally", "bullish", "soar", "gain", "rise", "breakout", "adoption", "institutional", "etf", "approval"]
            negative_keywords = ["crash", "dump", "bearish", "plunge", "drop", "fall", "hack", "scam", "ban", "regulation", "sec", "lawsuit"]
            neutral_keywords = ["update", "announce", "launch", "report", "study"]

            pos_count = 0
            neg_count = 0
            self.news_headlines = []
            for article in articles:
                title = (article.get("title", "") + " " + article.get("body", "")).lower()
                headline = article.get("title", "")
                source = article.get("source", "")
                sentiment = "neutral"
                for kw in positive_keywords:
                    if kw in title:
                        pos_count += 1
                        sentiment = "positive"
                for kw in negative_keywords:
                    if kw in title:
                        neg_count += 1
                        sentiment = "negative"
                self.news_headlines.append({
                    "title": headline,
                    "source": source,
                    "sentiment": sentiment,
                })

            total = pos_count + neg_count
            if total > 0:
                self.news_sentiment = (pos_count - neg_count) / total
            else:
                self.news_sentiment = 0.0
        except Exception:
            pass

    def _get_news_stats(self) -> dict:
        pos = sum(1 for h in self.news_headlines if h.get("sentiment") == "positive")
        neg = sum(1 for h in self.news_headlines if h.get("sentiment") == "negative")
        total = len(self.news_headlines)
        return {
            "total": total,
            "positive": pos,
            "negative": neg,
            "neutral": total - pos - neg,
        }

    def _estimate_social_sentiment(self, data: dict) -> float:
        scanner_data = data.get("scanner", {})
        hot_pairs = scanner_data.get("hot_pairs", [])
        if not hot_pairs:
            return 0.0
        avg_change = sum(p.get("change_24h", 0) for p in hot_pairs[:10]) / min(len(hot_pairs), 10)
        sentiment = max(-1.0, min(1.0, avg_change / 10.0))
        return sentiment

    def _calculate_overall(self) -> float:
        fng_normalized = (self.fear_greed_index - 50) / 50.0
        overall = (
            fng_normalized * 0.35
            + self.news_sentiment * 0.35
            + self.social_sentiment * 0.30
        )
        return max(-1.0, min(1.0, overall))

    def _get_market_mood(self) -> str:
        idx = self.fear_greed_index
        if idx <= 20:
            return "extreme_fear"
        elif idx <= 40:
            return "fear"
        elif idx <= 60:
            return "neutral"
        elif idx <= 80:
            return "greed"
        else:
            return "extreme_greed"

    def _get_fear_greed_context(self) -> str:
        idx = self.fear_greed_index
        if idx <= 20:
            return "Piyasa asiri korkuda - bu genellikle dip sinyalidir, contra investing firsati olabilir."
        elif idx <= 40:
            return "Piyasa korku modunda - dikkatli olun ama firsat arayin."
        elif idx <= 60:
            return "Piyasa notr - net bir yon yok, beklemek en iyisi."
        elif idx <= 80:
            return "Piyasa aclgozlu模nda - kar almak icin iyi bir zaman olabilir."
        else:
            return "Piyasa asiri aclgozlu模nda - dikkatli olun, duzeltme gelebilir."

    def _get_mood_analysis(self) -> dict:
        idx = self.fear_greed_index
        if idx <= 20:
            return {
                "signal": "contrarian_buy",
                "description": "Asiri korku - tarihsel olarak alim firsati",
                "recommendation": "Short pozisyonlarda kar al, long firsatlarini degerlendir",
            }
        elif idx <= 40:
            return {
                "signal": "cautious",
                "description": "Korku hakim - piyasa hassas",
                "recommendation": "Kucuk pozisyonlarla giris yapin, stop loss mutlaka koyun",
            }
        elif idx <= 60:
            return {
                "signal": "neutral",
                "description": "Notr piyasa - net sinyal yok",
                "recommendation": "Beklemede kalun, net sinyaller bekleyin",
            }
        elif idx <= 80:
            return {
                "signal": "take_profit",
                "description": "Aclgozlu模 artiyor - kar alinabilir",
                "recommendation": "Mevcut long pozisyonlarda kar al, short firsatlarini degerlendir",
            }
        else:
            return {
                "signal": "contrarian_short",
                "description": "Asiri aclgozlu模 - duzeltme riski yuksek",
                "recommendation": "Long pozisyonlarda kar al, short pozisyonlara gec",
            }

    def _extract_trending(self, data: dict) -> list[str]:
        scanner_data = data.get("scanner", {})
        hot_pairs = scanner_data.get("hot_pairs", [])
        return [p["baseCoin"] for p in hot_pairs[:10] if "baseCoin" in p]

    def _build_thinking(self) -> list[str]:
        steps = []
        steps.append(f"Fear & Greed Index: {self.fear_greed_index} ({self.fear_greed_label})")
        steps.append(f"Haber sentiment: {'+' if self.news_sentiment > 0 else ''}{self.news_sentiment:.2f}")
        steps.append(f"Sosyal sentiment: {'+' if self.social_sentiment > 0 else ''}{self.social_sentiment:.2f}")
        steps.append(f"Genel sentiment: {'+' if self.overall_sentiment > 0 else ''}{self.overall_sentiment:.2f}")
        steps.append(f"Piyasa modu: {self._get_market_mood()}")
        return steps
