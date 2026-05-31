"""
MEXC ML Trading System — Data Feeds V1.0
==========================================
Yeni Veri Kaynaklari: Sentiment, Order Book, Cross-Exchange

OZELLIKLER:
  1. Sentiment: Fear & Greed, Social Media, News
  2. Order Book: Depth, Imbalance, Large Orders
  3. Cross-Exchange: Binance, OKX, Fiyat Karsilastirma
  4. On-Chain: Whale Activity, Exchange Flow

NOT: Tum API'ler public (API key gerekmez) veya opsiyonel
"""

import asyncio
import numpy as np
import logging
import time
import json
from typing import Dict, List, Optional, Tuple
from collections import deque
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# SENTIMENT FEED
# ══════════════════════════════════════════════════════════════════════════════

class SentimentFeed:
    """
    Duygu analizi verileri:
    - Fear & Greed Index (alternative.me)
    - Social sentiment (opsiyonel API)
    - News sentiment (opsiyonel API)
    """

    # Fear & Greed Index API
    FNG_API = "https://api.alternative.me/fng/"

    def __init__(self, cache_ttl: int = 3600):
        self._cache = {}
        self._cache_ttl = cache_ttl
        self._last_fetch = {}

    async def fetch_fear_greed_index(self, client=None) -> Dict:
        """
        Fear & Greed Index cek.
        Donus: {"value": 0-100, "classification": "Fear"|"Greed", "timestamp": ...}
        """
        cache_key = "fear_greed"
        if cache_key in self._cache:
            age = time.time() - self._last_fetch.get(cache_key, 0)
            if age < self._cache_ttl:
                return self._cache[cache_key]

        try:
            if client is None:
                import httpx
                async with httpx.AsyncClient() as temp_client:
                    r = await temp_client.get(self.FNG_API, timeout=10)
            else:
                r = await client.get(self.FNG_API, timeout=10)

            if r.status_code == 200:
                data = r.json()
                if "data" in data and len(data["data"]) > 0:
                    entry = data["data"][0]
                    result = {
                        "value": int(entry.get("value", 50)),
                        "classification": entry.get("value_classification", "Neutral"),
                        "timestamp": entry.get("timestamp", ""),
                    }
                    self._cache[cache_key] = result
                    self._last_fetch[cache_key] = time.time()
                    return result
        except Exception as e:
            logger.debug(f"Fear & Greed API hatası: {e}")

        return {"value": 50, "classification": "Neutral", "timestamp": ""}

    async def fetch_social_sentiment(self, symbol: str = "BTC",
                                     client=None) -> Dict:
        """
        Sosyal medya sentiment (opsiyonel).
        Simdi: Basit proxy (gercek API ekleyebilirsiniz).
        """
        # Placeholder - gercek API entegrasyonu icin
        return {
            "twitter": 0.0,
            "reddit": 0.0,
            "overall": 0.0,
            "volume": 0,
        }

    async def get_sentiment_data(self, symbol: str = "BTC",
                                 client=None) -> Dict:
        """
        Tum sentiment verilerini topla.

        Donus: {
            "fear_greed": 0-100,
            "social_score": -1 ile 1,
            "news_sentiment": -1 ile 1,
            "overall_sentiment": -1 ile 1
        }
        """
        fng = await self.fetch_fear_greed_index(client)
        social = await self.fetch_social_sentiment(symbol, client)

        # Fear & Greed'i -1 ile 1 arasi normalize et
        fng_normalized = (fng["value"] - 50) / 50.0

        return {
            "fear_greed": fng["value"],
            "fear_greed_class": fng["classification"],
            "social_score": social.get("overall", 0.0),
            "news_sentiment": 0.0,  # Placeholder
            "overall_sentiment": fng_normalized,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


# ══════════════════════════════════════════════════════════════════════════════
# ORDER BOOK FEED
# ══════════════════════════════════════════════════════════════════════════════

class OrderBookFeed:
    """
    Order book verileri:
    - MEXC depth API
    - Imbalance hesaplama
    - Large order tespiti
    - Likidite analizi
    """

    MEXC_DEPTH_URL = "https://contract.mexc.com/api/v1/contract/depth/{symbol}"

    def __init__(self, cache_ttl: int = 10):
        self._cache = {}
        self._cache_ttl = cache_ttl
        self._last_fetch = {}

    async def fetch_depth(self, symbol: str, limit: int = 20,
                          client=None) -> Dict:
        """
        MEXC order book cek.

        Donus: {"bids": [[price, qty], ...], "asks": [[price, qty], ...]}
        """
        cache_key = f"depth_{symbol}"
        if cache_key in self._cache:
            age = time.time() - self._last_fetch.get(cache_key, 0)
            if age < self._cache_ttl:
                return self._cache[cache_key]

        url = self.MEXC_DEPTH_URL.format(symbol=symbol)
        try:
            if client is None:
                import httpx
                async with httpx.AsyncClient() as temp_client:
                    r = await temp_client.get(url, params={"limit": limit}, timeout=5)
            else:
                r = await client.get(url, params={"limit": limit}, timeout=5)

            if r.status_code == 200:
                data = r.json()
                if data.get("success") and data.get("data"):
                    depth = data["data"]
                    result = {
                        "bids": depth.get("bids", []),
                        "asks": depth.get("asks", []),
                        "timestamp": depth.get("timestamp", 0),
                    }
                    self._cache[cache_key] = result
                    self._last_fetch[cache_key] = time.time()
                    return result
        except Exception as e:
            logger.debug(f"Order book API hatası ({symbol}): {e}")

        return {"bids": [], "asks": [], "timestamp": 0}

    def calculate_imbalance(self, bids: List, asks: List,
                            levels: int = 10) -> float:
        """
        Order book dengesizlik orani.
        Positive: alim baskisi, Negative: satis baskisi
        """
        if not bids or not asks:
            return 0.0

        bid_vol = sum(b[1] for b in bids[:levels])
        ask_vol = sum(a[1] for a in asks[:levels])
        total = bid_vol + ask_vol

        if total < 1e-10:
            return 0.0

        return float(np.clip((bid_vol - ask_vol) / total, -1.0, 1.0))

    def detect_large_orders(self, bids: List, asks: List,
                            threshold_multiplier: float = 3.0) -> Dict:
        """
        Buyuk siparis tespiti.

        Returns: {
            "large_bids": [[price, qty], ...],
            "large_asks": [[price, qty], ...],
            "bid_wall": price,
            "ask_wall": price,
        }
        """
        if not bids or not asks:
            return {"large_bids": [], "large_asks": [],
                    "bid_wall": 0, "ask_wall": 0}

        # Ortalama hacim
        bid_volumes = [b[1] for b in bids]
        ask_volumes = [a[1] for a in asks]
        avg_bid = np.mean(bid_volumes) if bid_volumes else 0
        avg_ask = np.mean(ask_volumes) if ask_volumes else 0

        # Esik degeri
        bid_threshold = avg_bid * threshold_multiplier
        ask_threshold = avg_ask * threshold_multiplier

        large_bids = [b for b in bids if b[1] > bid_threshold]
        large_asks = [a for a in asks if a[1] > ask_threshold]

        # Duvar fiyatlari (en buyuk siparis)
        bid_wall = max(bids, key=lambda x: x[1])[0] if bids else 0
        ask_wall = max(asks, key=lambda x: x[1])[0] if asks else 0

        return {
            "large_bids": large_bids[:5],
            "large_asks": large_asks[:5],
            "bid_wall": bid_wall,
            "ask_wall": ask_wall,
        }

    def calculate_liquidity_score(self, bids: List, asks: List) -> float:
        """
        Likidite skoru: 0 (dusuk) - 1 (yuksek).
        Derinlik ve spread analizi.
        """
        if not bids or not asks:
            return 0.0

        # Spread
        best_bid = bids[0][0] if bids else 0
        best_ask = asks[0][0] if asks else 0
        if best_bid <= 0:
            return 0.0
        spread = (best_ask - best_bid) / best_bid

        # Derinlik (toplam hacim)
        total_bid_vol = sum(b[1] for b in bids[:10])
        total_ask_vol = sum(a[1] for a in asks[:10])
        depth_score = min((total_bid_vol + total_ask_vol) / 1000, 1.0)

        # Spread skoru (kucuk spread = iyi)
        spread_score = max(0, 1.0 - spread * 100)

        return float(np.clip((depth_score + spread_score) / 2, 0, 1))

    async def get_order_book_data(self, symbol: str,
                                   client=None) -> Dict:
        """
        Tum order book verilerini topla.

        Donus: {
            "imbalance": -1 ile 1,
            "large_orders": {...},
            "liquidity_score": 0-1,
            "spread_bps": float,
            "depth_total": float,
        }
        """
        depth = await self.fetch_depth(symbol, limit=20, client=client)
        bids = depth.get("bids", [])
        asks = depth.get("asks", [])

        imbalance = self.calculate_imbalance(bids, asks)
        large_orders = self.detect_large_orders(bids, asks)
        liquidity = self.calculate_liquidity_score(bids, asks)

        # Spread hesapla
        best_bid = bids[0][0] if bids else 0
        best_ask = asks[0][0] if asks else 0
        spread_bps = 0.0
        if best_bid > 0:
            spread_bps = (best_ask - best_bid) / best_bid * 10000

        # Toplam derinlik
        total_bid = sum(b[1] for b in bids[:10])
        total_ask = sum(a[1] for a in asks[:10])

        return {
            "imbalance": imbalance,
            "large_orders": large_orders,
            "liquidity_score": liquidity,
            "spread_bps": round(spread_bps, 2),
            "depth_total": total_bid + total_ask,
            "bid_depth": total_bid,
            "ask_depth": total_ask,
        }


# ══════════════════════════════════════════════════════════════════════════════
# CROSS-EXCHANGE FEED
# ══════════════════════════════════════════════════════════════════════════════

class CrossExchangeFeed:
    """
    Coklu borsa verileri:
    - Binance, OKX, Bybit fiyatlari
    - Arbitraj firsatlari
    - Fiyat farki analizi
    """

    EXCHANGE_APIS = {
        "binance": "https://fapi.binance.com/fapi/v1/ticker/price?symbol={symbol}",
        "okx": "https://www.okx.com/api/v5/market/ticker?instId={symbol}-USDT-SWAP",
        "bybit": "https://api.bybit.com/v5/market/tickers?category=linear&symbol={symbol}",
    }

    def __init__(self, cache_ttl: int = 30):
        self._cache = {}
        self._cache_ttl = cache_ttl
        self._last_fetch = {}

    async def fetch_exchange_price(self, exchange: str, symbol: str,
                                   client=None) -> Optional[float]:
        """Tek borsadan fiyat cek."""
        cache_key = f"{exchange}_{symbol}"
        if cache_key in self._cache:
            age = time.time() - self._last_fetch.get(cache_key, 0)
            if age < self._cache_ttl:
                return self._cache[cache_key]

        url_template = self.EXCHANGE_APIS.get(exchange)
        if not url_template:
            return None

        # Symbol donusumu
        exchange_symbol = symbol.replace("_", "")
        if exchange == "binance":
            exchange_symbol = symbol.replace("_", "")  # BTCUSDT
        elif exchange == "okx":
            exchange_symbol = symbol.replace("_", "-")  # BTC-USDT
        elif exchange == "bybit":
            exchange_symbol = symbol.replace("_", "")  # BTCUSDT

        url = url_template.format(symbol=exchange_symbol)

        try:
            if client is None:
                import httpx
                async with httpx.AsyncClient() as temp_client:
                    r = await temp_client.get(url, timeout=5)
            else:
                r = await client.get(url, timeout=5)

            if r.status_code == 200:
                data = r.json()
                price = None

                if exchange == "binance":
                    price = float(data.get("price", 0))
                elif exchange == "okx":
                    if data.get("data"):
                        price = float(data["data"][0].get("last", 0))
                elif exchange == "bybit":
                    if data.get("result") and data["result"].get("list"):
                        price = float(data["result"]["list"][0].get("lastPrice", 0))

                if price and price > 0:
                    self._cache[cache_key] = price
                    self._last_fetch[cache_key] = time.time()
                    return price
        except Exception as e:
            logger.debug(f"{exchange} API hatası ({symbol}): {e}")

        return None

    async def fetch_all_prices(self, symbol: str,
                                client=None) -> Dict[str, float]:
        """Tum borsalardan fiyat cek."""
        prices = {}
        for exchange in self.EXCHANGE_APIS:
            price = await self.fetch_exchange_price(exchange, symbol, client)
            if price:
                prices[exchange] = price
        return prices

    def calculate_arbitrage(self, prices: Dict[str, float]) -> Dict:
        """
        Arbitraj firsatlari hesapla.

        Returns: {
            "spread_pct": float,
            "buy_exchange": str,
            "sell_exchange": str,
            "opportunity": bool,
        }
        """
        if len(prices) < 2:
            return {"spread_pct": 0, "buy_exchange": "", "sell_exchange": "",
                    "opportunity": False}

        # En dusuk ve en yuksek fiyat
        min_price = min(prices.values())
        max_price = max(prices.values())
        min_exchange = min(prices, key=prices.get)
        max_exchange = max(prices, key=prices.get)

        spread_pct = (max_price - min_price) / min_price * 100

        return {
            "spread_pct": round(spread_pct, 4),
            "buy_exchange": min_exchange,
            "sell_exchange": max_exchange,
            "opportunity": spread_pct > 0.1,  # %0.1'den fazla fark
        }

    async def get_cross_exchange_data(self, symbol: str,
                                       client=None) -> Dict:
        """
        Tum coklu borsa verilerini topla.

        Donus: {
            "prices": {"binance": float, "okx": float, "bybit": float},
            "avg_price": float,
            "volatility": float,
            "arbitrage": {...},
            "correlation": float,
        }
        """
        prices = await self.fetch_all_prices(symbol, client)

        if not prices:
            return {"prices": {}, "avg_price": 0, "volatility": 0,
                    "arbitrage": {}, "correlation": 0}

        avg_price = float(np.mean(list(prices.values())))
        volatility = float(np.std(list(prices.values())) / (avg_price + 1e-10))
        arbitrage = self.calculate_arbitrage(prices)

        return {
            "prices": prices,
            "avg_price": round(avg_price, 2),
            "volatility": round(volatility, 6),
            "arbitrage": arbitrage,
            "exchange_count": len(prices),
        }


# ══════════════════════════════════════════════════════════════════════════════
# ON-CHAIN FEED (Opsiyonel)
# ══════════════════════════════════════════════════════════════════════════════

class OnChainFeed:
    """
    On-chain veriler (opsiyonel API'ler):
    - Whale activity
    - Exchange flow
    - Active addresses
    """

    def __init__(self):
        self._cache = {}
        self._cache_ttl = 300  # 5 dakika

    async def fetch_whale_activity(self, client=None) -> Dict:
        """
        Whale aktivitesi (placeholder).
        Gercek API: whale-alert.io, cryptopotato.com
        """
        return {
            "large_transactions": 0,
            "total_volume_usd": 0,
            "net_flow": 0,
        }

    async def fetch_exchange_flow(self, client=None) -> Dict:
        """
        Exchange akisi (placeholder).
        Gercek API: glassnode.com, cryptoquant.com
        """
        return {
            "inflow": 0,
            "outflow": 0,
            "net_flow": 0,
        }

    async def get_onchain_data(self, client=None) -> Dict:
        """Tum on-chain verileri topla."""
        whale = await self.fetch_whale_activity(client)
        flow = await self.fetch_exchange_flow(client)

        return {
            "whale_activity": whale,
            "exchange_flow": flow,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


# ══════════════════════════════════════════════════════════════════════════════
# ANA DATA FEED MANAGER
# ══════════════════════════════════════════════════════════════════════════════

class DataFeedManager:
    """
    Tum veri kaynaklarini yonetir:
    - Sentiment
    - Order Book
    - Cross-Exchange
    - On-Chain
    """

    def __init__(self):
        self.sentiment = SentimentFeed()
        self.order_book = OrderBookFeed()
        self.cross_exchange = CrossExchangeFeed()
        self.onchain = OnChainFeed()

        self._cache = {}
        self._cache_ttl = 60  # 1 dakika

    async def get_all_data(self, symbol: str,
                           client=None) -> Dict:
        """
        Tum veri kaynaklarindan veri topla.

        Donus: {
            "sentiment": {...},
            "order_book": {...},
            "cross_exchange": {...},
            "onchain": {...},
            "combined_features": [...],
        }
        """
        # Paralel veri cekimi
        tasks = [
            self.sentiment.get_sentiment_data(symbol, client),
            self.order_book.get_order_book_data(symbol, client),
            self.cross_exchange.get_cross_exchange_data(symbol, client),
            self.onchain.get_onchain_data(client),
        ]

        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            sentiment = results[0] if not isinstance(results[0], Exception) else {}
            order_book = results[1] if not isinstance(results[1], Exception) else {}
            cross_exchange = results[2] if not isinstance(results[2], Exception) else {}
            onchain = results[3] if not isinstance(results[3], Exception) else {}
        except Exception as e:
            logger.error(f"Veri cekme hatası: {e}")
            return {"sentiment": {}, "order_book": {}, "cross_exchange": {}, "onchain": {}}

        # Birlesik feature'lar olustur
        combined = self._create_combined_features(
            sentiment, order_book, cross_exchange, onchain
        )

        return {
            "sentiment": sentiment,
            "order_book": order_book,
            "cross_exchange": cross_exchange,
            "onchain": onchain,
            "combined_features": combined,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def _create_combined_features(self, sentiment: Dict, order_book: Dict,
                                   cross_exchange: Dict, onchain: Dict) -> List[float]:
        """
        Birlesik feature vektoru olustur.
        """
        features = []

        # Sentiment features (3)
        features.append(sentiment.get("overall_sentiment", 0.0))
        features.append((sentiment.get("fear_greed", 50) - 50) / 50.0)
        features.append(sentiment.get("social_score", 0.0))

        # Order book features (4)
        features.append(order_book.get("imbalance", 0.0))
        features.append(order_book.get("liquidity_score", 0.0))
        features.append(min(order_book.get("spread_bps", 0) / 100, 1.0))
        features.append(min(order_book.get("depth_total", 0) / 10000, 1.0))

        # Cross-exchange features (3)
        features.append(cross_exchange.get("volatility", 0.0) * 10)
        features.append(min(cross_exchange.get("arbitrage", {}).get("spread_pct", 0) / 10, 1.0))
        features.append(cross_exchange.get("exchange_count", 0) / 3.0)

        # On-chain features (2)
        whale = onchain.get("whale_activity", {})
        features.append(min(whale.get("large_transactions", 0) / 10, 1.0))
        flow = onchain.get("exchange_flow", {})
        features.append(float(np.clip(flow.get("net_flow", 0) / 1000, -1, 1)))

        return features

    def get_feature_names(self) -> List[str]:
        """Feature isimlerini dondur."""
        return [
            "sentiment_overall", "sentiment_fng", "sentiment_social",
            "ob_imbalance", "ob_liquidity", "ob_spread", "ob_depth",
            "xchange_volatility", "xchange_arbitrage", "xchange_count",
            "onchain_whale", "onchain_flow",
        ]


# ══════════════════════════════════════════════════════════════════════════════
# EXPORT
# ══════════════════════════════════════════════════════════════════════════════

__all__ = [
    "DataFeedManager",
    "SentimentFeed",
    "OrderBookFeed",
    "CrossExchangeFeed",
    "OnChainFeed",
]
