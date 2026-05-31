"""
MEXC ML Trading System — Market Data Enrichment
=================================================
Funding rate, Open Interest, Fear & Greed, Cross-asset korelasyon

OZELLIKLER:
  1. Funding Rate: MEXC funding rate analizi
  2. Open Interest: Acik pozisyon degisimi
  3. Fear & Greed: Piyasa duyarlılık endeksi
  4. Cross-Asset: BTC/ETH korelasyon analizi
"""

import numpy as np
import logging
import time
import asyncio
from typing import Dict, List, Optional, Tuple
from collections import deque

logger = logging.getLogger(__name__)


class FundingRateAnalyzer:
    """
    Funding rate analizi.
    Yuksek funding rate = cok long pozisyon = SHORT lehine
    Dusuk funding rate = cok short pozisyon = LONG lehine
    """

    MEXC_FUNDING_URL = "https://contract.mexc.com/api/v1/contract/funding/{symbol}"

    def __init__(self, cache_ttl: int = 300):
        self._cache: Dict = {}
        self._cache_ttl = cache_ttl
        self._history: Dict[str, deque] = {}

    async def fetch_funding_rate(self, symbol: str, client=None) -> Optional[float]:
        """
        MEXC funding rate cek.

        Donus: Funding rate (ornegin 0.0001 = %0.01)
        """
        cache_key = f"funding_{symbol}"
        if cache_key in self._cache:
            age = time.time() - self._cache.get(f"{cache_key}_time", 0)
            if age < self._cache_ttl:
                return self._cache[cache_key]

        url = self.MEXC_FUNDING_URL.format(symbol=symbol)
        try:
            if client is None:
                import httpx
                async with httpx.AsyncClient() as temp_client:
                    r = await temp_client.get(url, timeout=5)
            else:
                r = await client.get(url, timeout=5)

            if r.status_code == 200:
                data = r.json()
                if data.get("success") and data.get("data"):
                    funding_rate = float(data["data"].get("currentFundingRate", 0))
                    self._cache[cache_key] = funding_rate
                    self._cache[f"{cache_key}_time"] = time.time()
                    return funding_rate
            elif r.status_code == 403:
                logger.debug(f"Funding rate 403: {symbol} - API erisim engellendi")
            elif r.status_code == 429:
                logger.debug(f"Funding rate 429: {symbol} - Rate limit")
        except Exception as e:
            logger.debug(f"Funding rate API hatası ({symbol}): {e}")

        return None

    def analyze_funding_rate(self, funding_rate: float, symbol: str = "") -> Dict:
        """
        Funding rate analizi.

        Returns: {
            "rate": float,
            "rate_pct": float,
            "signal": "long_bias" | "short_bias" | "neutral",
            "signal_strength": float,  # -1 ile 1 arasi
            "is_extreme": bool,
        }
        """
        if funding_rate is None:
            return {
                "rate": 0, "rate_pct": 0,
                "signal": "neutral", "signal_strength": 0,
                "is_extreme": False,
            }

        rate_pct = funding_rate * 100  # Yuzdeye cevir

        # Sinyal belirleme
        # Yuksek pozitif funding = cok long = SHORT lehine
        if rate_pct > 0.05:  # %0.05+ funding
            signal = "short_bias"
            signal_strength = min(1.0, rate_pct / 0.1)  # %0.1'de max guc
            is_extreme = rate_pct > 0.1
        elif rate_pct < -0.05:  # Negatif funding
            signal = "long_bias"
            signal_strength = max(-1.0, rate_pct / 0.1)
            is_extreme = rate_pct < -0.1
        else:
            signal = "neutral"
            signal_strength = rate_pct / 0.1  # Normalize
            is_extreme = False

        # History
        if symbol:
            if symbol not in self._history:
                self._history[symbol] = deque(maxlen=100)
            self._history[symbol].append(funding_rate)

        return {
            "rate": round(funding_rate, 6),
            "rate_pct": round(rate_pct, 4),
            "signal": signal,
            "signal_strength": round(float(np.clip(signal_strength, -1, 1)), 3),
            "is_extreme": is_extreme,
        }


class OpenInterestAnalyzer:
    """
    Open Interest analizi.
    OI artisi + fiyat dususu = SHORT sinyali
    OI artisi + fiyat yukselisi = LONG sinyali
    """

    def __init__(self):
        self._oi_history: Dict[str, deque] = {}

    def calculate_oi_change(self, current_oi: float, symbol: str = "") -> Dict:
        """
        Open Interest degisimi analizi.

        Returns: {
            "current_oi": float,
            "oi_change_pct": float,
            "oi_trend": "increasing" | "decreasing" | "stable",
            "signal_modifier": float,
        }
        """
        if symbol not in self._oi_history:
            self._oi_history[symbol] = deque(maxlen=100)

        history = list(self._oi_history[symbol])
        prev_oi = history[-1] if history else current_oi

        self._oi_history[symbol].append(current_oi)

        if prev_oi <= 0:
            return {
                "current_oi": current_oi,
                "oi_change_pct": 0,
                "oi_trend": "stable",
                "signal_modifier": 0,
            }

        oi_change_pct = (current_oi - prev_oi) / prev_oi * 100

        # Trend
        if oi_change_pct > 5:
            oi_trend = "increasing"
        elif oi_change_pct < -5:
            oi_trend = "decreasing"
        else:
            oi_trend = "stable"

        # Signal modifier (fiyatla birlikte degerlendirilmeli)
        # Sadece OI degisimi yeterli degil, fiyat yonuyle birlikte bakilmali
        signal_modifier = 0.0  # main.py'de fiyat yonuyle birlestirilecek

        return {
            "current_oi": round(current_oi, 2),
            "oi_change_pct": round(oi_change_pct, 2),
            "oi_trend": oi_trend,
            "signal_modifier": round(signal_modifier, 3),
        }


class FearGreedAnalyzer:
    """
    Fear & Greed Index analizi.
    Extreme Fear = alim firsati (LONG lehine)
    Extreme Greed = satis firsati (SHORT lehine)
    """

    FNG_API = "https://api.alternative.me/fng/"

    def __init__(self, cache_ttl: int = 3600):
        self._cache: Dict = {}
        self._cache_ttl = cache_ttl

    async def fetch_fear_greed_index(self, client=None) -> Optional[int]:
        """
        Fear & Greed Index cek.

        Donus: 0-100 arasi deger
        """
        cache_key = "fear_greed"
        if cache_key in self._cache:
            age = time.time() - self._cache.get(f"{cache_key}_time", 0)
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
                    value = int(data["data"][0].get("value", 50))
                    self._cache[cache_key] = value
                    self._cache[f"{cache_key}_time"] = time.time()
                    return value
        except Exception as e:
            logger.debug(f"Fear & Greed API hatası: {e}")

        return None

    def analyze_fear_greed(self, value: int) -> Dict:
        """
        Fear & Greed analizi.

        Returns: {
            "value": int,
            "classification": str,
            "signal": "buy" | "sell" | "neutral",
            "signal_strength": float,
        }
        """
        if value is None:
            return {
                "value": 50,
                "classification": "Neutral",
                "signal": "neutral",
                "signal_strength": 0,
            }

        # Siniflandirma
        if value < 20:
            classification = "Extreme Fear"
            signal = "buy"  # Asiri korku = alim firsati
            signal_strength = 0.8
        elif value < 40:
            classification = "Fear"
            signal = "buy"
            signal_strength = 0.4
        elif value < 60:
            classification = "Neutral"
            signal = "neutral"
            signal_strength = 0.0
        elif value < 80:
            classification = "Greed"
            signal = "sell"
            signal_strength = -0.4
        else:
            classification = "Extreme Greed"
            signal = "sell"  # Asiri hirs = satis firsati
            signal_strength = -0.8

        return {
            "value": value,
            "classification": classification,
            "signal": signal,
            "signal_strength": round(float(np.clip(signal_strength, -1, 1)), 3),
        }


class CrossAssetAnalyzer:
    """
    Cross-asset korelasyon analizi.
    BTC → ETH → ALT korelasyon zinciri
    """

    def __init__(self):
        self._price_history: Dict[str, deque] = {}

    def update_price(self, symbol: str, price: float):
        """Fiyat guncelle."""
        if symbol not in self._price_history:
            self._price_history[symbol] = deque(maxlen=200)
        self._price_history[symbol].append(price)

    def calculate_correlation(self, symbol_a: str, symbol_b: str,
                               window: int = 50) -> float:
        """
        Iki varlik arasindaki korelasyon hesapla.

        Donus: -1 ile 1 arasi korelasyon katsayisi
        """
        if symbol_a not in self._price_history or \
           symbol_b not in self._price_history:
            return 0.0

        prices_a = list(self._price_history[symbol_a])[-window:]
        prices_b = list(self._price_history[symbol_b])[-window:]

        if len(prices_a) < 10 or len(prices_b) < 10:
            return 0.0

        # Getiri hesapla
        arr_a = np.array(prices_a, dtype=np.float64)
        arr_b = np.array(prices_b, dtype=np.float64)

        if len(arr_a) < 2 or len(arr_b) < 2:
            return 0.0

        returns_a = np.diff(arr_a) / (arr_a[:-1] + 1e-10)
        returns_b = np.diff(arr_b) / (arr_b[:-1] + 1e-10)

        # Korelasyon
        if len(returns_a) < 5 or len(returns_b) < 5:
            return 0.0

        min_len = min(len(returns_a), len(returns_b))
        if min_len < 2:
            return 0.0

        corr = np.corrcoef(returns_a[-min_len:], returns_b[-min_len:])[0, 1]

        if np.isnan(corr):
            return 0.0

        return float(corr)

    def analyze_btc_eth_relationship(self) -> Dict:
        """
        BTC-ETH iliskisi analizi.

        Returns: {
            "correlation": float,
            "eth_btc_ratio": float,
            "btc_trend": str,
            "eth_relative_strength": float,
        }
        """
        btc_prices = list(self._price_history.get("BTC_USDT", []))
        eth_prices = list(self._price_history.get("ETH_USDT", []))

        if len(btc_prices) < 20 or len(eth_prices) < 20:
            return {
                "correlation": 0,
                "eth_btc_ratio": 0,
                "btc_trend": "unknown",
                "eth_relative_strength": 0,
            }

        # Diziye cevir
        btc_arr = np.array(btc_prices[-20:], dtype=np.float64)
        eth_arr = np.array(eth_prices[-20:], dtype=np.float64)

        # Korelasyon
        correlation = self.calculate_correlation("BTC_USDT", "ETH_USDT")

        # ETH/BTC ratio
        if btc_arr[-1] > 0:
            eth_btc_ratio = eth_arr[-1] / btc_arr[-1]
        else:
            eth_btc_ratio = 0.0

        # BTC trendi
        btc_returns = np.diff(btc_arr) / (btc_arr[:-1] + 1e-10)
        if len(btc_returns) > 0:
            btc_trend_strength = float(np.mean(btc_returns))
        else:
            btc_trend_strength = 0.0

        if btc_trend_strength > 0.001:
            btc_trend = "up"
        elif btc_trend_strength < -0.001:
            btc_trend = "down"
        else:
            btc_trend = "sideways"

        # ETH goreceli gucu
        eth_returns = np.diff(eth_arr) / (eth_arr[:-1] + 1e-10)
        if len(eth_returns) > 0:
            eth_strength = float(np.mean(eth_returns))
        else:
            eth_strength = 0.0

        eth_relative_strength = eth_strength - btc_trend_strength

        return {
            "correlation": round(correlation, 3),
            "eth_btc_ratio": round(eth_btc_ratio, 6),
            "btc_trend": btc_trend,
            "btc_trend_strength": round(float(np.clip(btc_trend_strength * 100, -1, 1)), 3),
            "eth_relative_strength": round(float(np.clip(eth_relative_strength * 100, -1, 1)), 3),
        }

    def get_cross_asset_signal(self, symbol: str) -> Dict:
        """
        Cross-asset sinyal uret.

        Returns: {
            "signal_modifier": float,
            "reason": str,
        }
        """
        btc_eth = self.analyze_btc_eth_relationship()

        signal_modifier = 0.0
        reasons = []

        # BTC trend etkisi
        if btc_eth["btc_trend"] == "up":
            signal_modifier += 0.1
            reasons.append("BTC yukseliste")
        elif btc_eth["btc_trend"] == "down":
            signal_modifier -= 0.1
            reasons.append("BTC dususte")

        # ETH goreceli guc
        if btc_eth["eth_relative_strength"] > 0.2:
            signal_modifier += 0.05
            reasons.append("ETH BTC'den guclu")
        elif btc_eth["eth_relative_strength"] < -0.2:
            signal_modifier -= 0.05
            reasons.append("ETH BTC'den zayif")

        # Korelasyon etkisi
        if abs(btc_eth["correlation"]) > 0.7:
            # Yuksek korelasyon → BTC yonu onemli
            if btc_eth["btc_trend"] == "up" and "LONG" in symbol.upper():
                signal_modifier += 0.05
            elif btc_eth["btc_trend"] == "down" and "SHORT" in symbol.upper():
                signal_modifier += 0.05

        return {
            "signal_modifier": round(float(np.clip(signal_modifier, -0.3, 0.3)), 3),
            "reason": "; ".join(reasons) if reasons else "Nötr",
            "btc_eth": btc_eth,
        }


class MarketDataEnrichment:
    """
    Tum veri zenginlestirme modullerini birlestirir.
    """

    def __init__(self):
        self.funding = FundingRateAnalyzer()
        self.oi = OpenInterestAnalyzer()
        self.fear_greed = FearGreedAnalyzer()
        self.cross_asset = CrossAssetAnalyzer()

    async def get_all_market_data(self, symbol: str,
                                   client=None) -> Dict:
        """
        Tum piyasa verilerini topla ve analiz et.

        Returns: {
            "funding": {...},
            "open_interest": {...},
            "fear_greed": {...},
            "cross_asset": {...},
            "total_signal_modifier": float,
        }
        """
        # Funding rate
        funding_rate = await self.funding.fetch_funding_rate(symbol, client)
        funding_analysis = self.funding.analyze_funding_rate(funding_rate, symbol)

        # Fear & Greed
        fng_value = await self.fear_greed.fetch_fear_greed_index(client)
        fng_analysis = self.fear_greed.analyze_fear_greed(fng_value)

        # Cross-asset
        cross_asset_signal = self.cross_asset.get_cross_asset_signal(symbol)

        # Toplam sinyal degistirici
        total_modifier = 0.0
        total_modifier += funding_analysis["signal_strength"] * 0.3
        total_modifier += fng_analysis["signal_strength"] * 0.2
        total_modifier += cross_asset_signal["signal_modifier"]

        return {
            "funding": funding_analysis,
            "open_interest": {"current_oi": 0, "oi_change_pct": 0, "oi_trend": "stable"},
            "fear_greed": fng_analysis,
            "cross_asset": cross_asset_signal,
            "total_signal_modifier": round(float(np.clip(total_modifier, -0.5, 0.5)), 3),
        }


# ══════════════════════════════════════════════════════════════════════════════
# EXPORT
# ══════════════════════════════════════════════════════════════════════════════

__all__ = [
    "MarketDataEnrichment",
    "FundingRateAnalyzer",
    "OpenInterestAnalyzer",
    "FearGreedAnalyzer",
    "CrossAssetAnalyzer",
]
