"""
MEXC ML Trading System — Feature Engineering V3.0
==================================================
Gelismis ozellik cikarma: Multi-Timeframe, Microstructure, Cross-Asset, Regime

OZELLIKLER:
  1. Multi-Timeframe: 15m, 1h, 4h, 1g timeframe analizi
  2. Market Microstructure: Order book, trade flow, likidite
  3. Cross-Asset: BTC dominance, ETH/BTC, stablecoin flow
  4. Regime: Piyasa durumu siniflandirmasi
  5. Sentiment: Fear & Greed, social sentiment (API bagimli)
"""

import numpy as np
import logging
import time
from typing import Optional, Dict, List, Tuple
from collections import deque
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Numpy uyarilarini sustur
np.seterr(all='ignore')


# ══════════════════════════════════════════════════════════════════════════════
# MULTI-TIMEFRAME FEATURES
# ══════════════════════════════════════════════════════════════════════════════

class MultiTimeframeFeatures:
    """
    Farkli zaman dilimlerinden feature cikarir.
    15m ana timeframe, 1h/4h/1g higher timeframe.
    """

    TIMEFRAMES = {
        "15m": 1,      # ana
        "1h":  4,      # 4x 15m
        "4h":  16,     # 16x 15m
        "1d":  96,     # 96x 15m
    }

    @staticmethod
    def resample_klines(klines: dict, factor: int) -> dict:
        """
        Klines'i higher timeframe'e yeniden ornekler.
        factor: kac 15m bar'in birlestirilecegi
        """
        if factor <= 1:
            return klines

        close = np.asarray(klines["close"], dtype=np.float64)
        high = np.asarray(klines["high"], dtype=np.float64)
        low = np.asarray(klines["low"], dtype=np.float64)
        volume = np.asarray(klines["volume"], dtype=np.float64)
        n = len(close)

        if n < factor + 1:
            return klines

        new_n = n // factor
        new_close = np.zeros(new_n)
        new_high = np.zeros(new_n)
        new_low = np.zeros(new_n)
        new_volume = np.zeros(new_n)

        for i in range(new_n):
            start = i * factor
            end = min((i + 1) * factor, n)
            new_close[i] = close[end - 1]  # Son kapanis
            new_high[i] = np.max(high[start:end])
            new_low[i] = np.min(low[start:end])
            new_volume[i] = np.sum(volume[start:end])

        return {
            "close": new_close,
            "high": new_high,
            "low": new_low,
            "volume": new_volume,
        }

    @staticmethod
    def compute_rsi_trend(c: np.ndarray, period: int = 14) -> float:
        """RSI trend yonu: artan/dusan/ yatay."""
        if len(c) < period + 5:
            return 0.0
        rsi_vals = []
        for i in range(max(period, len(c) - 10), len(c) + 1):
            if i < period + 2:
                continue
            d = np.diff(c[i - (period + 5):i])
            g = np.where(d > 0, d, 0.0)
            lo = np.where(d < 0, -d, 0.0)
            ag, al = np.mean(g[-period:]), np.mean(lo[-period:])
            rsi = 100.0 if al < 1e-12 else 100 - 100 / (1 + ag / al)
            rsi_vals.append(rsi)

        if len(rsi_vals) < 3:
            return 0.0
        arr = np.array(rsi_vals)
        slope = float(np.polyfit(np.arange(len(arr)), arr, 1)[0])
        return float(np.clip(slope / 10.0, -1.0, 1.0))

    @staticmethod
    def compute_macd_divergence(c: np.ndarray) -> float:
        """MACD diverjansi: fiyat ve MACD uyumsuzlugu."""
        if len(c) < 50:
            return 0.0
        # MACD hesapla
        e12 = MultiTimeframeFeatures._ema(c, 12)
        e26 = MultiTimeframeFeatures._ema(c, 26)
        macd_line = e12 - e26
        signal = MultiTimeframeFeatures._ema(macd_line, 9)
        hist = macd_line - signal

        # Son 20 bardaki diverjans
        if len(c) < 20 or len(hist) < 20:
            return 0.0

        price_trend = c[-1] - c[-20]
        macd_trend = hist[-1] - hist[-20]

        if price_trend > 0 and macd_trend < 0:
            return -1.0  # Bearish diverjans
        elif price_trend < 0 and macd_trend > 0:
            return 1.0   # Bullish diverjans
        return 0.0

    @staticmethod
    def _ema(c: np.ndarray, p: int) -> np.ndarray:
        """EMA hesaplama."""
        if len(c) == 0:
            return np.zeros(1)
        k = 2 / (p + 1)
        out = np.empty(len(c))
        out[0] = c[0]
        for i in range(1, len(c)):
            out[i] = c[i] * k + out[i - 1] * (1 - k)
        return out

    @classmethod
    def build_multi_tf_features(cls, klines: dict, timestamp=None) -> np.ndarray:
        """
        Multi-timeframe feature vektoru olustur.

        Donus:
          - 4 timeframe x 6 ozellik = 24 ozellik
        """
        features = []

        for tf_name, factor in cls.TIMEFRAMES.items():
            tf_klines = cls.resample_klines(klines, factor)
            c = np.asarray(tf_klines["close"], dtype=np.float64)
            h = np.asarray(tf_klines["high"], dtype=np.float64)
            lo = np.asarray(tf_klines["low"], dtype=np.float64)
            v = np.asarray(tf_klines["volume"], dtype=np.float64)

            if len(c) < 20:
                features.extend([0.0] * 6)
                continue

            # 1. RSI trend
            rsi_trend = cls.compute_rsi_trend(c, 14)
            features.append(rsi_trend)

            # 2. MACD diverjans
            macd_div = cls.compute_macd_divergence(c)
            features.append(macd_div)

            # 3. Fiyat/Moving Average mesafesi
            ema20 = cls._ema(c, 20)[-1]
            price_dist = float(np.clip((c[-1] - ema20) / (ema20 + 1e-10), -0.1, 0.1))
            features.append(price_dist)

            # 4. Volatilite (ATR/Price)
            if len(c) > 14:
                tr = np.maximum(h[1:] - lo[1:],
                               np.maximum(np.abs(h[1:] - c[:-1]),
                                         np.abs(lo[1:] - c[:-1])))
                atr = float(np.mean(tr[-14:]))
                atr_pct = float(np.clip(atr / (c[-1] + 1e-10), 0, 0.1))
            else:
                atr_pct = 0.0
            features.append(atr_pct)

            # 5. Hacim trendi
            if len(v) > 10:
                vol_avg = float(np.mean(v[-20:-10]) + 1e-10)
                vol_recent = float(np.mean(v[-5:]))
                vol_trend = float(np.clip(vol_recent / vol_avg, 0, 3))
            else:
                vol_trend = 1.0
            features.append(vol_trend)

            # 6. Fiyat pozisyonu (high-low arasi)
            if len(c) > 20:
                h20 = float(np.max(h[-20:]))
                l20 = float(np.min(lo[-20:]))
                rng = h20 - l20 + 1e-10
                pos = float((c[-1] - l20) / rng)
            else:
                pos = 0.5
            features.append(pos)

        return np.array(features, dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# MARKET MICROSTRUCTURE FEATURES
# ══════════════════════════════════════════════════════════════════════════════

class MicrostructureFeatures:
    """
    Piyasa mikroyapisi ozellikleri:
    - Order book analizi (API bagimli)
    - Trade flow analizi
    - Likidite metrikleri
    - VWAP analizi
    """

    @staticmethod
    def order_book_imbalance(bids: np.ndarray, asks: np.ndarray,
                             levels: int = 5) -> float:
        """
        Order book dengesizlik orani.
        Positive: alim baskisi, Negative: satis baskisi
        """
        if len(bids) < levels or len(asks) < levels:
            return 0.0

        bid_vol = np.sum(bids[:levels, 1])  # volume
        ask_vol = np.sum(asks[:levels, 1])
        total = bid_vol + ask_vol

        if total < 1e-10:
            return 0.0

        imbalance = (bid_vol - ask_vol) / total
        return float(np.clip(imbalance, -1.0, 1.0))

    @staticmethod
    def trade_flow_toxicity(trades: list, window: int = 50) -> float:
        """
        Trade flow zehirlilik orani (VPIN benzeri).
        Agresif alim/satis orani.
        """
        if len(trades) < window:
            return 0.0

        recent = trades[-window:]
        buy_vol = sum(t["qty"] for t in recent if t["side"] == "buy")
        sell_vol = sum(t["qty"] for t in recent if t["side"] == "sell")
        total = buy_vol + sell_vol

        if total < 1e-10:
            return 0.0

        toxicity = (buy_vol - sell_vol) / total
        return float(np.clip(toxicity, -1.0, 1.0))

    @staticmethod
    def implementation_shortfall(prices: np.ndarray, volumes: np.ndarray,
                                  n_bars: int = 20) -> float:
        """
        Uygulama eksikligi: Beklenen ile gerceklesen fiyat farki.
        High shortfall = likidite problemi
        """
        if len(prices) < n_bars + 1:
            return 0.0

        # VWAP hesapla
        vwap = np.sum(prices[-n_bars:] * volumes[-n_bars:]) / (np.sum(volumes[-n_bars:]) + 1e-10)

        # Mevcut fiyat - VWAP
        shortfall = float((prices[-1] - vwap) / (vwap + 1e-10))
        return float(np.clip(shortfall, -0.05, 0.05))

    @staticmethod
    def amihud_illiquidity(returns: np.ndarray, volumes: np.ndarray,
                           window: int = 20) -> float:
        """
        Amihud illikitlik olcusu: |return| / volume
        Yuksek = likidite dusuk
        """
        if len(returns) < window or len(volumes) < window:
            return 0.0

        abs_returns = np.abs(returns[-window:])
        vol = volumes[-window:] + 1e-10
        illiq = np.mean(abs_returns / vol)
        return float(np.clip(illiq * 1e6, 0, 10))  # Olceklendirme

    @staticmethod
    def kyle_lambda(prices: np.ndarray, volumes: np.ndarray,
                    window: int = 20) -> float:
        """
        Kyle's Lambda: Fiyat etkisi katsayisi.
        Yuksek = buyuk emirler fiyati etkiliyor
        """
        if len(prices) < window + 1 or len(volumes) < window:
            return 0.0

        price_changes = np.diff(prices[-window - 1:])
        vol = volumes[-window:]

        # Basit regresyon
        if np.sum(vol) < 1e-10:
            return 0.0

        covariance = np.cov(price_changes, vol)[0, 1]
        variance = np.var(vol)

        if variance < 1e-10:
            return 0.0

        kyle = covariance / variance
        return float(np.clip(kyle * 1e6, -10, 10))  # Olceklendirme

    @classmethod
    def build_microstructure_features(cls, klines: dict,
                                       order_book: dict = None,
                                       trades: list = None) -> np.ndarray:
        """
        Mikroyapisi feature vektoru olustur.

        Donus: 6 ozellik
        """
        c = np.asarray(klines["close"], dtype=np.float64)
        v = np.asarray(klines["volume"], dtype=np.float64)
        h = np.asarray(klines["high"], dtype=np.float64)
        lo = np.asarray(klines["low"], dtype=np.float64)

        features = []

        # 1. Order book imbalance (varsa)
        if order_book and "bids" in order_book and "asks" in order_book:
            ob_imb = cls.order_book_imbalance(
                np.array(order_book["bids"]),
                np.array(order_book["asks"])
            )
        else:
            # Proxy: high-low analizi
            if len(h) > 5:
                bid_proxy = np.mean(h[-5:])
                ask_proxy = np.mean(lo[-5:])
                ob_imb = float(np.clip((bid_proxy - ask_proxy) / (bid_proxy + 1e-10), -1, 1))
            else:
                ob_imb = 0.0
        features.append(ob_imb)

        # 2. Trade flow (varsa)
        if trades and len(trades) > 10:
            tf_toxic = cls.trade_flow_toxicity(trades)
        else:
            # Proxy:_VOLUME-price trend
            if len(c) > 10:
                price_ret = np.diff(c[-11:]) / (c[-11:-1] + 1e-10)
                vol_norm = v[-10:] / (np.mean(v[-20:-10]) + 1e-10)
                tf_toxic = float(np.clip(np.mean(np.sign(price_ret) * vol_norm), -1, 1))
            else:
                tf_toxic = 0.0
        features.append(tf_toxic)

        # 3. Implementation shortfall
        is_val = cls.implementation_shortfall(c, v)
        features.append(is_val)

        # 4. Amihud illiquidity
        if len(c) > 21:
            returns = np.diff(c[-21:]) / (c[-21:-1] + 1e-10)
        else:
            returns = np.diff(c) / (c[:-1] + 1e-10) if len(c) > 1 else np.zeros(1)
        amihud = cls.amihud_illiquidity(returns, v)
        features.append(amihud)

        # 5. Kyle's Lambda
        kyle = cls.kyle_lambda(c, v)
        features.append(kyle)

        # 6. Spread proxy (bid-ask approximation)
        if len(h) > 5 and len(lo) > 5:
            spread = float(np.mean((h[-5:] - lo[-5:]) / (c[-5:] + 1e-10)))
        else:
            spread = 0.0
        features.append(float(np.clip(spread, 0, 0.05)))

        return np.array(features, dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# CROSS-ASSET FEATURES
# ══════════════════════════════════════════════════════════════════════════════

class CrossAssetFeatures:
    """
    Coklu varlik analizi:
    - BTC dominance etkisi
    - ETH/BTC ratio
    - Stablecoin flow
    - Korelasyon analizi
    """

    @staticmethod
    def btc_dominance_impact(btc_change: float, altcoin_change: float) -> float:
        """
        BTC dominans etkisi.
        Positive: BTC gucleniyor (altcoin icin negatif)
        Negative: BTC zayifliyor (altcoin icin pozitif)
        """
        diff = btc_change - altcoin_change
        return float(np.clip(diff / 10.0, -1.0, 1.0))

    @staticmethod
    def eth_btc_ratio(eth_price: float, btc_price: float) -> float:
        """
        ETH/BTC ratio degisimi.
        Yuksek: ETH performansi iyi
        Dusuk: BTC performansi iyi
        """
        if btc_price < 1e-10:
            return 0.0
        ratio = eth_price / btc_price
        # Normalize: tarihsel ortalama ~0.05-0.08 civari
        normalized = (ratio - 0.06) / 0.02
        return float(np.clip(normalized, -1.0, 1.0))

    @staticmethod
    def correlation_features(returns_a: np.ndarray, returns_b: np.ndarray,
                            window: int = 20) -> dict:
        """
        Iki varlik arasindaki korelasyon ozellikleri.
        """
        if len(returns_a) < window or len(returns_b) < window:
            return {"correlation": 0.0, "beta": 1.0, "spread": 0.0}

        a = returns_a[-window:]
        b = returns_b[-window:]

        # Korelasyon
        corr = float(np.corrcoef(a, b)[0, 1])
        if np.isnan(corr):
            corr = 0.0

        # Beta (b'ye karsi)
        cov_ab = np.cov(a, b)[0, 1]
        var_b = np.var(b)
        beta = cov_ab / (var_b + 1e-10)
        beta = float(np.clip(beta, -3.0, 3.0))

        # Spread (farkli getiri)
        spread = float(np.mean(a - b))

        return {
            "correlation": corr,
            "beta": beta,
            "spread": spread,
        }

    @staticmethod
    def stablecoin_flow(usdt_change: float, usdc_change: float) -> float:
        """
        Stablecoin akisi: Piyasaya giris/cikis.
        Positive: giris (bullish), Negative: cikis (bearish)
        """
        total_change = usdt_change + usdc_change
        return float(np.clip(total_change / 5.0, -1.0, 1.0))

    @classmethod
    def build_cross_asset_features(cls, market_data: dict = None) -> np.ndarray:
        """
        Cross-asset feature vektoru olustur.

        Donus: 8 ozellik
        """
        features = []

        if market_data is None:
            market_data = {}

        # 1. BTC dominance impact
        btc_chg = market_data.get("btc_24h_change", 0.0)
        alt_avg = market_data.get("altcoin_avg_change", 0.0)
        features.append(cls.btc_dominance_impact(btc_chg, alt_avg))

        # 2. ETH/BTC ratio
        eth_price = market_data.get("eth_price", 3000.0)
        btc_price = market_data.get("btc_price", 60000.0)
        features.append(cls.eth_btc_ratio(eth_price, btc_price))

        # 3. BTC volatility regime
        btc_vol = market_data.get("btc_volatility", 0.0)
        features.append(float(np.clip(btc_vol / 0.05, 0, 2)))

        # 4. Market cap change
        mcap_change = market_data.get("market_cap_change", 0.0)
        features.append(float(np.clip(mcap_change / 5.0, -1, 1)))

        # 5. Stablecoin flow
        usdt_chg = market_data.get("usdt_change", 0.0)
        usdc_chg = market_data.get("usdc_change", 0.0)
        features.append(cls.stablecoin_flow(usdt_chg, usdc_chg))

        # 6. Funding rate (varsa)
        funding = market_data.get("funding_rate", 0.0)
        features.append(float(np.clip(funding * 100, -1, 1)))

        # 7. Open interest change
        oi_change = market_data.get("oi_change", 0.0)
        features.append(float(np.clip(oi_change / 10.0, -1, 1)))

        # 8. Long/Short ratio
        ls_ratio = market_data.get("long_short_ratio", 1.0)
        features.append(float(np.clip((ls_ratio - 1.0) * 2, -1, 1)))

        return np.array(features, dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# REGIME FEATURES
# ══════════════════════════════════════════════════════════════════════════════

class RegimeFeatures:
    """
    Piyasa durumu siniflandirmasi:
    - Accumulation (birikim)
    - Markup (yukselis)
    - Distribution (dagilim)
    - Markdown (dusus)

    Ayrica:
    - Volatilite rejimi
    - Trend gucu
    - Momentum rejimi
    """

    @staticmethod
    def detect_market_phase(c: np.ndarray, v: np.ndarray,
                            window: int = 50) -> float:
        """
        Piyasa asamasini tespit et.
        Donus: -1.0 (markdown) ile 1.0 (markup) arasi
        """
        if len(c) < window:
            return 0.0

        # Trend
        ema_fast = float(np.mean(c[-10:]))
        ema_slow = float(np.mean(c[-30:])) if len(c) > 30 else ema_fast
        trend = (ema_fast - ema_slow) / (ema_slow + 1e-10)

        # Hacim trendi
        vol_recent = float(np.mean(v[-10:]))
        vol_old = float(np.mean(v[-30:])) if len(v) > 30 else vol_recent
        vol_trend = vol_recent / (vol_old + 1e-10) - 1.0

        # Fiyat/Hacim uyumu
        price_up = c[-1] > c[-window]
        vol_up = vol_trend > 0

        if price_up and vol_up:
            return 0.8   # Markup (guclu yukselis)
        elif price_up and not vol_up:
            return 0.3   # Distribution (zayif yukselis)
        elif not price_up and vol_up:
            return -0.3  # Markdown baslangici (satis baskisi)
        else:
            return -0.8  # Markdown (guclu dusus)

    @staticmethod
    def volatility_regime(c: np.ndarray, window: int = 20) -> float:
        """
        Volatilite rejimi: 0=dusuk, 1=orta, 2=yuksek
        ATR percentile kullanarak.
        """
        if len(c) < window + 14:
            return 1.0  # Orta

        # ATR hesapla (basit proxy)
        returns = np.abs(np.diff(c[-window - 1:]))
        atr_now = float(np.mean(returns[-14:]))
        atr_hist = [float(np.mean(returns[i:i + 14]))
                   for i in range(max(0, len(returns) - 20), len(returns) - 14)]

        if not atr_hist:
            return 1.0

        pct_below = float(np.mean(np.array(atr_hist) < atr_now))

        if pct_below < 0.3:
            return 0.0  # Dusuk volatilite
        elif pct_below > 0.7:
            return 2.0  # Yuksek volatilite
        return 1.0      # Orta

    @staticmethod
    def trend_strength(c: np.ndarray, window: int = 20) -> float:
        """
        Trend gucu: ADX benzeri ama basitlestirilmis.
        0 = yatay, 1 = guclu trend
        """
        if len(c) < window + 5:
            return 0.0

        # Yon degisim sayisi
        changes = np.diff(c[-window - 1:])
        direction_changes = np.sum(np.abs(np.diff(np.sign(changes))))

        # Normalize: max possible changes = window - 1
        max_changes = window - 1
        strength = 1.0 - (direction_changes / (max_changes + 1e-10))

        return float(np.clip(strength, 0, 1))

    @staticmethod
    def momentum_regime(c: np.ndarray, window: int = 20) -> float:
        """
        Momentum rejimi: asiri alim/asiri satim.
        -1 = asiri satim, 0 = nötr, 1 = asiri alim
        """
        if len(c) < window:
            return 0.0

        returns = np.diff(c[-window - 1:]) / (c[-window:] + 1e-10)

        if len(returns) < 5:
            return 0.0

        # Z-score
        mean_ret = float(np.mean(returns))
        std_ret = float(np.std(returns))
        if std_ret < 1e-10:
            return 0.0

        z_score = mean_ret / std_ret

        # Asiri alim/satim
        if z_score > 2.0:
            return 1.0   # Asiri alim
        elif z_score < -2.0:
            return -1.0  # Asiri satim
        return float(np.clip(z_score / 2.0, -1.0, 1.0))

    @classmethod
    def build_regime_features(cls, klines: dict) -> np.ndarray:
        """
        Regime feature vektoru olustur.

        Donus: 8 ozellik
        """
        c = np.asarray(klines["close"], dtype=np.float64)
        v = np.asarray(klines["volume"], dtype=np.float64)
        h = np.asarray(klines["high"], dtype=np.float64)
        lo = np.asarray(klines["low"], dtype=np.float64)

        features = []

        # 1. Market phase
        phase = cls.detect_market_phase(c, v)
        features.append(phase)

        # 2. Volatility regime
        vol_regime = cls.volatility_regime(c)
        features.append(vol_regime / 2.0)  # Normalize [0,1]

        # 3. Trend strength
        trend_str = cls.trend_strength(c)
        features.append(trend_str)

        # 4. Momentum regime
        mom_regime = cls.momentum_regime(c)
        features.append(mom_regime)

        # 5. Price velocity (hiz)
        if len(c) > 5:
            velocity = float((c[-1] - c[-5]) / (c[-5] + 1e-10))
        else:
            velocity = 0.0
        features.append(float(np.clip(velocity, -0.1, 0.1)))

        # 6. Price acceleration (ivme)
        if len(c) > 10:
            vel1 = (c[-5] - c[-10]) / (c[-10] + 1e-10)
            vel2 = (c[-1] - c[-5]) / (c[-5] + 1e-10)
            accel = vel2 - vel1
        else:
            accel = 0.0
        features.append(float(np.clip(accel * 10, -1, 1)))

        # 7. Range expansion
        if len(h) > 20:
            range_now = float(np.mean(h[-5:] - lo[-5:]))
            range_old = float(np.mean(h[-20:-10] - lo[-20:-10])) if len(h) > 20 else range_now
            range_exp = range_now / (range_old + 1e-10)
        else:
            range_exp = 1.0
        features.append(float(np.clip(range_exp - 1.0, -1, 1)))

        # 8. Volume profile
        if len(v) > 20:
            vol_high = float(np.max(v[-20:]))
            vol_low = float(np.min(v[-20:]))
            vol_profile = (v[-1] - vol_low) / (vol_high - vol_low + 1e-10)
        else:
            vol_profile = 0.5
        features.append(float(np.clip(vol_profile, 0, 1)))

        return np.array(features, dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# SENTIMENT FEATURES (API bagimli)
# ══════════════════════════════════════════════════════════════════════════════

class SentimentFeatures:
    """
    Duygu analizi ozellikleri:
    - Fear & Greed Index
    - Social sentiment (Twitter, Reddit)
    - Whale activity
    """

    @staticmethod
    def fear_greed_index(value: int = 50) -> float:
        """
        Fear & Greed Index normalize.
        0 = Extreme Fear, 100 = Extreme Greed
        Donus: -1 (fear) ile 1 (greed) arasi
        """
        return float(np.clip((value - 50) / 50.0, -1.0, 1.0))

    @staticmethod
    def social_sentiment(score: float = 0.0) -> float:
        """
        Sosyal medya sentiment skoru.
        -1 = negatif, 0 = notr, 1 = pozitif
        """
        return float(np.clip(score, -1.0, 1.0))

    @staticmethod
    def whale_activity(large_txns: int = 0, total_txns: int = 100) -> float:
        """
        Balina aktivitesi orani.
        """
        if total_txns < 1:
            return 0.0
        ratio = large_txns / total_txns
        return float(np.clip(ratio * 10, 0, 1))

    @classmethod
    def build_sentiment_features(cls, sentiment_data: dict = None) -> np.ndarray:
        """
        Sentiment feature vektoru olustur.

        Donus: 6 ozellik
        """
        features = []

        if sentiment_data is None:
            sentiment_data = {}

        # 1. Fear & Greed
        fg_value = sentiment_data.get("fear_greed", 50)
        features.append(cls.fear_greed_index(fg_value))

        # 2. Social sentiment
        social = sentiment_data.get("social_score", 0.0)
        features.append(cls.social_sentiment(social))

        # 3. Whale activity
        whale_large = sentiment_data.get("whale_large_txns", 0)
        whale_total = sentiment_data.get("whale_total_txns", 100)
        features.append(cls.whale_activity(whale_large, whale_total))

        # 4. News sentiment
        news = sentiment_data.get("news_sentiment", 0.0)
        features.append(cls.social_sentiment(news))

        # 5. Developer activity
        dev = sentiment_data.get("developer_activity", 0.0)
        features.append(float(np.clip(dev, -1, 1)))

        # 6. Community engagement
        community = sentiment_data.get("community_score", 0.0)
        features.append(float(np.clip(community, -1, 1)))

        return np.array(features, dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# ANA FEATURE BUILDER V3
# ══════════════════════════════════════════════════════════════════════════════

class FeatureBuilderV3:
    """
    Feature Engineering V3.0 - Ana sinif.
    Tum feature builder'lari birlestirir.

    Toplam ozellik sayisi:
      - Multi-Timeframe: 24
      - Microstructure: 6
      - Cross-Asset: 8
      - Regime: 8
      - Sentiment: 6
      - V2 Base: 64
      --------------------------------
      - TOPLAM: 116 ozellik (sentiment hariç 110)
    """

    N_FEATURES_BASE = 64    # V2 base
    N_FEATURES_MTF = 24     # Multi-timeframe
    N_FEATURES_MICRO = 6    # Microstructure
    N_FEATURES_CROSS = 8    # Cross-asset
    N_FEATURES_REGIME = 8   # Regime
    N_FEATURES_SENT = 6     # Sentiment

    N_FEATURES_TOTAL = 116  # Toplam

    def __init__(self, use_mtf: bool = True, use_micro: bool = True,
                 use_cross: bool = True, use_regime: bool = True,
                 use_sentiment: bool = True):
        self._use_mtf = use_mtf
        self._use_micro = use_micro
        self._use_cross = use_cross
        self._use_regime = use_regime
        self._use_sentiment = use_sentiment

        self._feature_names = self._init_feature_names()
        logger.info(f"FeatureBuilderV3: {len(self._feature_names)} ozellik hazir")

    def _init_feature_names(self) -> list:
        """Feature isimlerini olustur."""
        names = []

        # V2 base isimleri (64)
        base_names = [
            "rsi14", "stoch_rsi14", "mfi14", "williams_r14",
            "macd_line", "macd_signal", "macd_hist",
            "ema9_21_cross", "price_ema50_dist",
            "bb_position", "bb_width", "atr14",
            "volume_ratio", "obv_trend",
            "lag_ret1", "lag_ret2", "lag_ret3",
            "roll_ret5", "roll_ret10", "roll_ret20",
            "roll_vol5", "roll_vol20",
            "price_pos_h20", "price_pos_l20",
            "momentum_acc",
            "candle_body", "candle_hl", "upper_wick",
            "vwap_dist", "momentum_slope",
            "rsi_div_dist", "market_regime",
            "volume_slope", "lower_wick",
            "ema_cross_delta", "atr_percentile",
        ]
        names.extend(base_names)

        # V2 new isimleri (~28)
        v2_new = [
            "rsi_divergence", "macd_histogram_momentum",
            "bb_width_pct", "atr_slope", "volatility_regime",
            "price_acceleration", "efficient_ratio", "kama",
            "adx", "volume_ratio_ema", "volume_price_trend",
            "cumulative_delta", "pivot_distance",
            "hour_of_day_sin", "hour_of_day_cos",
            "day_of_week_sin", "day_of_week_cos", "session",
            "kurtosis", "entropy",
        ]
        names.extend(v2_new)

        # Multi-timeframe isimleri
        if self._use_mtf:
            for tf in ["15m", "1h", "4h", "1d"]:
                names.extend([
                    f"mtf_rsi_trend_{tf}",
                    f"mtf_macd_div_{tf}",
                    f"mtf_price_dist_{tf}",
                    f"mtf_atr_pct_{tf}",
                    f"mtf_vol_trend_{tf}",
                    f"mtf_price_pos_{tf}",
                ])

        # Microstructure isimleri
        if self._use_micro:
            names.extend([
                "ob_imbalance", "tf_toxicity",
                "impl_shortfall", "amihud_illiq",
                "kyle_lambda", "spread_proxy",
            ])

        # Cross-asset isimleri
        if self._use_cross:
            names.extend([
                "btc_dominance", "eth_btc_ratio",
                "btc_vol_regime", "mcap_change",
                "stablecoin_flow", "funding_rate",
                "oi_change", "ls_ratio",
            ])

        # Regime isimleri
        if self._use_regime:
            names.extend([
                "market_phase", "vol_regime",
                "trend_strength", "momentum_regime",
                "price_velocity", "price_accel",
                "range_expansion", "volume_profile",
            ])

        # Sentiment isimleri
        if self._use_sentiment:
            names.extend([
                "fear_greed", "social_sentiment",
                "whale_activity", "news_sentiment",
                "dev_activity", "community_score",
            ])

        return names

    @property
    def n_features(self) -> int:
        return len(self._feature_names)

    @property
    def feature_names(self) -> list:
        return list(self._feature_names)

    def build(self, klines: dict, market_data: dict = None,
              sentiment_data: dict = None, timestamp=None,
              order_book: dict = None, trades: list = None) -> np.ndarray:
        """
        Tam feature vektoru olustur.

        Parametreler:
          klines: OHLCV verisi
          market_data: Cross-asset verileri (varsa)
          sentiment_data: Sentiment verileri (varsa)
          timestamp: Zaman damgasi
          order_book: Order book verisi (varsa)
          trades: Islem verisi (varsa)

        Donus: (n_features,) boyutunda numpy array
        """
        # V2 base features (64 ozellik)
        try:
            from features import FeatureBuilderV2
            fb_v2 = FeatureBuilderV2(use_new_features=True)
            base_features = fb_v2.build(klines, timestamp=timestamp)
        except ImportError:
            # Fallback: ml_engine.FeatureBuilder
            from ml_engine import FeatureBuilder
            base_features = FeatureBuilder.build(klines)

        if base_features is None:
            return None

        all_features = [base_features]

        # Multi-timeframe (24 ozellik)
        if self._use_mtf:
            mtf_features = MultiTimeframeFeatures.build_multi_tf_features(klines, timestamp)
            all_features.append(mtf_features)

        # Microstructure (6 ozellik)
        if self._use_micro:
            micro_features = MicrostructureFeatures.build_microstructure_features(
                klines, order_book, trades
            )
            all_features.append(micro_features)

        # Cross-asset (8 ozellik)
        if self._use_cross:
            if market_data is None:
                market_data = {}
            cross_features = CrossAssetFeatures.build_cross_asset_features(market_data)
            all_features.append(cross_features)

        # Regime (8 ozellik)
        if self._use_regime:
            regime_features = RegimeFeatures.build_regime_features(klines)
            all_features.append(regime_features)

        # Sentiment (6 ozellik)
        if self._use_sentiment:
            if sentiment_data is None:
                sentiment_data = {}
            sent_features = SentimentFeatures.build_sentiment_features(sentiment_data)
            all_features.append(sent_features)

        # Birlestir
        combined = np.concatenate(all_features)
        return np.where(np.isfinite(combined), combined, 0.0)

    def build_dataset(self, klines: dict, lookahead: int = 3,
                      threshold: float = 0.004,
                      market_data: dict = None,
                      sentiment_data: dict = None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Dataset olustur (tum bar'lar icin).

        Donus: (X, y) - feature matrix ve label vector
        """
        c = np.asarray(klines["close"], dtype=np.float64)
        MIN = 40
        if len(c) < MIN + lookahead:
            return None, None

        # Dinamik threshold
        if len(c) >= 50:
            avg_move = float(np.mean(np.abs(np.diff(c[-50:])) / (c[-50:-1] + 1e-10)))
            dyn_thr = float(np.clip(avg_move * 0.6, 0.002, 0.006))
        else:
            dyn_thr = threshold * 0.7

        X_rows, y_rows = [], []
        for i in range(MIN, len(c) - lookahead):
            sub = {
                "close": c[:i],
                "high": np.asarray(klines["high"], dtype=np.float64)[:i],
                "low": np.asarray(klines["low"], dtype=np.float64)[:i],
                "volume": np.asarray(klines["volume"], dtype=np.float64)[:i],
            }
            feat = self.build(sub, market_data, sentiment_data)
            if feat is None:
                continue
            ret = (c[i + lookahead] - c[i]) / (c[i] + 1e-10)
            label = 1 if ret > dyn_thr else (-1 if ret < -dyn_thr else 0)
            X_rows.append(feat)
            y_rows.append(label)

        if len(X_rows) < 10:
            return None, None

        return np.array(X_rows, dtype=np.float32), np.array(y_rows, dtype=np.int32)


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE IMPORTANCE TRACKER
# ══════════════════════════════════════════════════════════════════════════════

class FeatureImportanceTracker:
    """
    Feature onem skorlarini takip eder.
    - Permutation importance
    - SHAP benzeri analiz
    - Feature selection onerileri
    """

    def __init__(self, feature_names: list):
        self._names = feature_names
        self._importances = np.zeros(len(feature_names))
        self._counts = np.zeros(len(feature_names))
        self._history = deque(maxlen=100)

    def update(self, importances: np.ndarray):
        """Feature importance guncelle."""
        if len(importances) != len(self._names):
            return
        self._importances += importances
        self._counts += 1

    def get_importance(self, top_k: int = 20) -> list:
        """En onemli K ozelligi getir."""
        avg = self._importances / (self._counts + 1e-10)
        idx = np.argsort(-avg)[:top_k]
        return [(self._names[i], round(float(avg[i]), 4)) for i in idx]

    def get_selected_features(self, threshold: float = 0.01) -> list:
        """Onem esiginin ustundeki feature'lari sec."""
        avg = self._importances / (self._counts + 1e-10)
        max_imp = np.max(avg)
        if max_imp < 1e-10:
            return self._names
        normalized = avg / max_imp
        selected = [self._names[i] for i in range(len(self._names))
                   if normalized[i] > threshold]
        return selected

    def get_stats(self) -> dict:
        """Istatistikleri getir."""
        avg = self._importances / (self._counts + 1e-10)
        return {
            "total_features": len(self._names),
            "avg_importance": round(float(np.mean(avg)), 4),
            "max_importance": round(float(np.max(avg)), 4),
            "min_importance": round(float(np.min(avg)), 4),
            "updates": int(np.sum(self._counts)),
        }


# ══════════════════════════════════════════════════════════════════════════════
# EXPORT
# ══════════════════════════════════════════════════════════════════════════════

__all__ = [
    "FeatureBuilderV3",
    "MultiTimeframeFeatures",
    "MicrostructureFeatures",
    "CrossAssetFeatures",
    "RegimeFeatures",
    "SentimentFeatures",
    "FeatureImportanceTracker",
]
