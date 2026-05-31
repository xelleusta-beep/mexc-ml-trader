"""
MEXC ML Trading System — Advanced Order Book Analysis
=====================================================
Gelismis order book analizi: Multi-level imbalance, wall detection, spread analysis

OZELLIKLER:
  1. Multi-Level Imbalance: Farkli derinlik seviyelerinde dengesizlik
  2. Wall Detection: Duvar tespiti ve mesafe analizi
  3. Spread Analysis: Real-time spread takibi
  4. Depth Profile: Derinlik profili olusturma
  5. Liquidity Heatmap: Likidite haritasi
"""

import numpy as np
import logging
import time
from typing import Dict, List, Optional, Tuple
from collections import deque

logger = logging.getLogger(__name__)

# Numpy uyarilarini sustur
np.seterr(all='ignore')


class AdvancedOrderBook:
    """
    Gelismis order book analiz sinifi.
    """

    def __init__(self, cache_ttl: float = 5.0):
        self._cache_ttl = cache_ttl
        self._cache: Dict = {}
        self._last_fetch: Dict = {}
        self._spread_history: Dict[str, deque] = {}
        self._imbalance_history: Dict[str, deque] = {}

    def calculate_multi_level_imbalance(self, bids: List, asks: List,
                                         levels: List[int] = [5, 10, 20]) -> Dict:
        """
        Farkli derinlik seviyelerinde dengesizlik hesapla.

        Returns: {
            "imbalance_5": float,   # 5 seviye dengesizlik
            "imbalance_10": float,  # 10 seviye dengesizlik
            "imbalance_20": float,  # 20 seviye dengesizlik
            "imbalance_trend": float,  # Dengesizlik trendi
        }
        """
        result = {}

        for level in levels:
            bid_vol = sum(b[1] for b in bids[:level])
            ask_vol = sum(a[1] for a in asks[:level])
            total = bid_vol + ask_vol

            if total < 1e-10:
                result[f"imbalance_{level}"] = 0.0
            else:
                result[f"imbalance_{level}"] = float(np.clip(
                    (bid_vol - ask_vol) / total, -1.0, 1.0
                ))

        # Trend: 5 seviye vs 20 seviye farkı
        if "imbalance_5" in result and "imbalance_20" in result:
            result["imbalance_trend"] = result["imbalance_5"] - result["imbalance_20"]

        return result

    def detect_walls_with_proximity(self, bids: List, asks: List,
                                     current_price: float,
                                     threshold_multiplier: float = 3.0) -> Dict:
        """
        Duvar tespiti ve mevcut fiyata uzaklik analizi.

        Returns: {
            "bid_walls": [{"price": float, "qty": float, "distance_pct": float}],
            "ask_walls": [{"price": float, "qty": float, "distance_pct": float}],
            "nearest_bid_wall": float,
            "nearest_ask_wall": float,
            "wall_support": float,  # Destek gucu
            "wall_resistance": float,  # Direnç gucu
        }
        """
        if not bids or not asks or current_price <= 0:
            return {
                "bid_walls": [], "ask_walls": [],
                "nearest_bid_wall": 0, "nearest_ask_wall": 0,
                "wall_support": 0, "wall_resistance": 0,
            }

        # Ortalama hacim
        bid_volumes = [b[1] for b in bids]
        ask_volumes = [a[1] for a in asks]
        avg_bid = np.mean(bid_volumes) if bid_volumes else 0
        avg_ask = np.mean(ask_volumes) if ask_volumes else 0

        bid_threshold = avg_bid * threshold_multiplier
        ask_threshold = avg_ask * threshold_multiplier

        # Bid duvarlari (destek)
        bid_walls = []
        for b in bids:
            if b[1] > bid_threshold:
                distance_pct = (current_price - b[0]) / current_price * 100
                bid_walls.append({
                    "price": b[0],
                    "qty": b[1],
                    "distance_pct": round(distance_pct, 2),
                    "strength": round(b[1] / (avg_bid + 1e-10), 2),
                })

        # Ask duvarlari (direnç)
        ask_walls = []
        for a in asks:
            if a[1] > ask_threshold:
                distance_pct = (a[0] - current_price) / current_price * 100
                ask_walls.append({
                    "price": a[0],
                    "qty": a[1],
                    "distance_pct": round(distance_pct, 2),
                    "strength": round(a[1] / (avg_ask + 1e-10), 2),
                })

        # En yakın duvarlar
        nearest_bid = min([w["distance_pct"] for w in bid_walls], default=100)
        nearest_ask = min([w["distance_pct"] for w in ask_walls], default=100)

        # Destek/Direnç gücü (yakınlık ve büyüklüğe göre)
        wall_support = sum(w["strength"] / (1 + w["distance_pct"]) for w in bid_walls)
        wall_resistance = sum(w["strength"] / (1 + w["distance_pct"]) for w in ask_walls)

        return {
            "bid_walls": bid_walls[:5],
            "ask_walls": ask_walls[:5],
            "nearest_bid_wall": round(nearest_bid, 2),
            "nearest_ask_wall": round(nearest_ask, 2),
            "wall_support": round(float(np.clip(wall_support, 0, 10)), 2),
            "wall_resistance": round(float(np.clip(wall_resistance, 0, 10)), 2),
        }

    def analyze_spread(self, bids: List, asks: List,
                       symbol: str = "") -> Dict:
        """
        Spread analizi ve trendi.

        Returns: {
            "spread_bps": float,
            "spread_pct": float,
            "spread_zscore": float,  # Son Z-score
            "spread_trend": float,   # Trend (-1 dusus, +1 yukselis)
            "is_widening": bool,
        }
        """
        if not bids or not asks:
            return {
                "spread_bps": 0, "spread_pct": 0,
                "spread_zscore": 0, "spread_trend": 0,
                "is_widening": False,
            }

        best_bid = bids[0][0]
        best_ask = asks[0][0]

        if best_bid <= 0:
            return {
                "spread_bps": 0, "spread_pct": 0,
                "spread_zscore": 0, "spread_trend": 0,
                "is_widening": False,
            }

        spread_bps = (best_ask - best_bid) / best_bid * 10000
        spread_pct = (best_ask - best_bid) / best_bid * 100

        # Spread history
        if symbol not in self._spread_history:
            self._spread_history[symbol] = deque(maxlen=100)
        self._spread_history[symbol].append(spread_bps)

        # Z-score hesapla
        history = list(self._spread_history[symbol])
        if len(history) >= 10:
            mean_spread = np.mean(history)
            std_spread = np.std(history)
            zscore = (spread_bsp - mean_spread) / (std_spread + 1e-10)
        else:
            zscore = 0.0

        # Trend (son 5 deger)
        if len(history) >= 5:
            recent = history[-5:]
            trend = float(np.polyfit(range(len(recent)), recent, 1)[0])
            is_widening = trend > 0
        else:
            trend = 0.0
            is_widening = False

        return {
            "spread_bps": round(spread_bps, 2),
            "spread_pct": round(spread_pct, 4),
            "spread_zscore": round(float(zscore), 2),
            "spread_trend": round(float(np.clip(trend / 10, -1, 1)), 2),
            "is_widening": is_widening,
        }

    def calculate_depth_profile(self, bids: List, asks: List,
                                 n_levels: int = 20) -> Dict:
        """
        Derinlik profili olustur.

        Returns: {
            "bid_depth_curve": [float],  # Bid derinlik egrisi
            "ask_depth_curve": [float],  # Ask derinlik egrisi
            "depth_imbalance": float,     # Derinlik dengesizligi
            "total_bid_depth": float,
            "total_ask_depth": float,
            "mid_price_depth": float,     # Orta fiyat derinligi
        }
        """
        if not bids or not asks:
            return {
                "bid_depth_curve": [], "ask_depth_curve": [],
                "depth_imbalance": 0, "total_bid_depth": 0,
                "total_ask_depth": 0, "mid_price_depth": 0,
            }

        # Kümülatif derinlik
        bid_depth = []
        ask_depth = []
        cum_bid = 0
        cum_ask = 0

        for i in range(min(n_levels, len(bids))):
            cum_bid += bids[i][1]
            bid_depth.append(cum_bid)

        for i in range(min(n_levels, len(asks))):
            cum_ask += asks[i][1]
            ask_depth.append(cum_ask)

        # Dengesizlik
        total_bid = cum_bid
        total_ask = cum_ask
        total = total_bid + total_ask

        if total < 1e-10:
            depth_imbalance = 0.0
        else:
            depth_imbalance = (total_bid - total_ask) / total

        # Orta fiyat derinligi
        mid_price = (bids[0][0] + asks[0][0]) / 2
        mid_depth = 0
        for b in bids:
            if abs(b[0] - mid_price) / mid_price < 0.001:  # %0.1 yakınlık
                mid_depth += b[1]
                break

        return {
            "bid_depth_curve": [round(d, 2) for d in bid_depth],
            "ask_depth_curve": [round(d, 2) for d in ask_depth],
            "depth_imbalance": round(float(np.clip(depth_imbalance, -1, 1)), 3),
            "total_bid_depth": round(total_bid, 2),
            "total_ask_depth": round(total_ask, 2),
            "mid_price_depth": round(mid_depth, 2),
        }

    def calculate_liquidity_zones(self, bids: List, asks: List,
                                   current_price: float) -> Dict:
        """
        Likidite bolgeleri tespit et.

        Returns: {
            "strong_support": float,   # Guclu destek seviyesi
            "strong_resistance": float, # Guclu direnç seviyesi
            "support_distance_pct": float,
            "resistance_distance_pct": float,
            "liquidity_score": float,  # 0-1 arasi
        }
        """
        if not bids or not asks or current_price <= 0:
            return {
                "strong_support": 0, "strong_resistance": 0,
                "support_distance_pct": 100, "resistance_distance_pct": 100,
                "liquidity_score": 0,
            }

        # En buyuk bid (destek)
        max_bid = max(bids, key=lambda x: x[1])
        support = max_bid[0]
        support_dist = (current_price - support) / current_price * 100

        # En buyuk ask (direnç)
        max_ask = max(asks, key=lambda x: x[1])
        resistance = max_ask[0]
        resistance_dist = (resistance - current_price) / current_price * 100

        # Likidite skoru
        bid_depth = sum(b[1] for b in bids[:10])
        ask_depth = sum(a[1] for a in asks[:10])
        total_depth = bid_depth + ask_depth

        # Yayilma skoru
        spread = (asks[0][0] - bids[0][0]) / bids[0][0]
        spread_score = max(0, 1.0 - spread * 100)

        # Derinlik skoru
        depth_score = min(total_depth / 10000, 1.0)

        liquidity_score = (spread_score * 0.5 + depth_score * 0.5)

        return {
            "strong_support": round(support, 6),
            "strong_resistance": round(resistance, 6),
            "support_distance_pct": round(support_dist, 2),
            "resistance_distance_pct": round(resistance_dist, 2),
            "liquidity_score": round(float(np.clip(liquidity_score, 0, 1)), 3),
        }

    def get_comprehensive_analysis(self, bids: List, asks: List,
                                    current_price: float,
                                    symbol: str = "") -> Dict:
        """
        Kapsamli order book analizi.

        Returns: {
            "imbalance": {...},
            "walls": {...},
            "spread": {...},
            "depth": {...},
            "liquidity": {...},
            "signal_modifier": float,  # -1 ile 1 arasi sinyal degistirici
        }
        """
        imbalance = self.calculate_multi_level_imbalance(bids, asks)
        walls = self.detect_walls_with_proximity(bids, asks, current_price)
        spread = self.analyze_spread(bids, asks, symbol)
        depth = self.calculate_depth_profile(bids, asks)
        liquidity = self.calculate_liquidity_zones(bids, asks, current_price)

        # Sinyal degistirici hesapla
        signal_modifier = 0.0

        # 1. Imbalance etkisi
        imbalance_5 = imbalance.get("imbalance_5", 0)
        if imbalance_5 > 0.3:  # Guclu alim baskisi
            signal_modifier += 0.2
        elif imbalance_5 < -0.3:  # Guclu satis baskisi
            signal_modifier -= 0.2

        # 2. Duvar etkisi
        if walls["wall_support"] > walls["wall_resistance"] * 1.5:
            signal_modifier += 0.15  # Destek guclu
        elif walls["wall_resistance"] > walls["wall_support"] * 1.5:
            signal_modifier -= 0.15  # Direnç guclu

        # 3. Spread etkisi
        if spread["spread_bps"] > 10:  # Genis spread
            signal_modifier -= 0.1  # Dikkatli ol
        elif spread["spread_bps"] < 2:  # Dar spread
            signal_modifier += 0.05  # Likidite iyi

        # 4. Derinlik etkisi
        if depth["depth_imbalance"] > 0.2:
            signal_modifier += 0.1
        elif depth["depth_imbalance"] < -0.2:
            signal_modifier -= 0.1

        signal_modifier = float(np.clip(signal_modifier, -0.5, 0.5))

        return {
            "imbalance": imbalance,
            "walls": walls,
            "spread": spread,
            "depth": depth,
            "liquidity": liquidity,
            "signal_modifier": round(signal_modifier, 3),
        }


# ══════════════════════════════════════════════════════════════════════════════
# MULTI-TIMEFRAME CONFIRMATION
# ══════════════════════════════════════════════════════════════════════════════

class MultiTimeframeConfirmation:
    """
    Coklu zaman dilimi onay sistemi.

    15m sinyal + 1h onay + 4h trend = Daha guvenli giris
    """

    def __init__(self):
        self._trend_cache: Dict = {}

    def analyze_trend(self, klines: dict, period: int = 20) -> Dict:
        """
        Trend analizi.

        Returns: {
            "trend": "UP" | "DOWN" | "SIDEWAYS",
            "strength": float,  # 0-1 arasi
            "ema_fast": float,
            "ema_slow": float,
            "momentum": float,
        }
        """
        c = np.asarray(klines.get("close", []), dtype=np.float64)
        if len(c) < period + 10:
            return {"trend": "SIDEWAYS", "strength": 0, "ema_fast": 0,
                    "ema_slow": 0, "momentum": 0}

        # EMA hesapla
        def ema(data, p):
            k = 2 / (p + 1)
            out = np.empty(len(data))
            out[0] = data[0]
            for i in range(1, len(data)):
                out[i] = data[i] * k + out[i - 1] * (1 - k)
            return out

        ema_fast = ema(c, 9)[-1]
        ema_slow = ema(c, 21)[-1]
        ema_50 = ema(c, min(50, len(c) - 1))[-1]

        # Trend belirleme
        if ema_fast > ema_slow > ema_50:
            trend = "UP"
            strength = min(1.0, (ema_fast - ema_slow) / (ema_slow + 1e-10) * 100)
        elif ema_fast < ema_slow < ema_50:
            trend = "DOWN"
            strength = min(1.0, (ema_slow - ema_fast) / (ema_fast + 1e-10) * 100)
        else:
            trend = "SIDEWAYS"
            strength = 0.0

        # Momentum
        momentum = float(np.polyfit(range(5), c[-5:], 1)[0] / (c[-1] + 1e-10))

        return {
            "trend": trend,
            "strength": round(float(np.clip(strength, 0, 1)), 3),
            "ema_fast": round(float(ema_fast), 6),
            "ema_slow": round(float(ema_slow), 6),
            "momentum": round(float(np.clip(momentum, -0.01, 0.01)), 5),
        }

    def confirm_signal(self, signal: str, klines_15m: dict,
                       klines_1h: dict, klines_4h: dict) -> Dict:
        """
        Coklu zaman dilimi sinyal onayi.

        Returns: {
            "confirmed": bool,
            "confidence_boost": float,  # Guven artisi/azalması
            "reason": str,
            "trend_alignment": float,  # Trend uyumu
        }
        """
        trend_15m = self.analyze_trend(klines_15m)
        trend_1h = self.analyze_trend(klines_1h)
        trend_4h = self.analyze_trend(klines_4h)

        # Trend uyumu
        trends = [trend_15m["trend"], trend_1h["trend"], trend_4h["trend"]]

        if signal == "LONG":
            if trends.count("UP") >= 2:
                confirmed = True
                confidence_boost = 0.15
                reason = "BUY onayli: 2/3 trend yukseliste"
            elif trends.count("DOWN") >= 2:
                confirmed = False
                confidence_boost = -0.25
                reason = "BUY reddedildi: 2/3 trend dususte"
            else:
                confirmed = True
                confidence_boost = 0.05
                reason = "BUY nötr: trend uyumsuz"

        elif signal == "SHORT":
            if trends.count("DOWN") >= 2:
                confirmed = True
                confidence_boost = 0.15
                reason = "SELL onayli: 2/3 trend dususte"
            elif trends.count("UP") >= 2:
                confirmed = False
                confidence_boost = -0.25
                reason = "SELL reddedildi: 2/3 trend yukseliste"
            else:
                confirmed = True
                confidence_boost = 0.05
                reason = "SELL nötr: trend uyumsuz"
        else:
            confirmed = False
            confidence_boost = 0.0
            reason = "WAIT sinyali"

        # Trend uyum skoru
        if signal in ("LONG", "SHORT"):
            expected = "UP" if signal == "LONG" else "DOWN"
            alignment = sum(1 for t in trends if t == expected) / 3
        else:
            alignment = 0.0

        return {
            "confirmed": confirmed,
            "confidence_boost": round(confidence_boost, 3),
            "reason": reason,
            "trend_alignment": round(float(alignment), 3),
            "trends": {
                "15m": trend_15m,
                "1h": trend_1h,
                "4h": trend_4h,
            },
        }


# ══════════════════════════════════════════════════════════════════════════════
# DYNAMIC SL/TP
# ══════════════════════════════════════════════════════════════════════════════

class DynamicSLTP:
    """
    Dinamik Stop-Loss ve Take-Profit hesaplama.
    ATR bazli, volatiliteye gore ayarlanan SL/TP.
    """

    @staticmethod
    def calculate_atr(klines: dict, period: int = 14) -> float:
        """
        Average True Range hesapla.
        """
        h = np.asarray(klines.get("high", []), dtype=np.float64)
        lo = np.asarray(klines.get("low", []), dtype=np.float64)
        c = np.asarray(klines.get("close", []), dtype=np.float64)

        if len(c) < period + 1:
            return 0.0

        tr = np.maximum(
            h[1:] - lo[1:],
            np.maximum(
                np.abs(h[1:] - c[:-1]),
                np.abs(lo[1:] - c[:-1])
            )
        )

        return float(np.mean(tr[-period:]))

    @classmethod
    def calculate_dynamic_sl_tp(cls, klines: dict, price: float,
                                 signal: str, leverage: int = 5,
                                 risk_reward_ratio: float = 2.0) -> Dict:
        """
        Dinamik SL/TP hesapla.

        Returns: {
            "sl_price": float,
            "tp_price": float,
            "sl_pct": float,
            "tp_pct": float,
            "atr": float,
            "method": str,
        }
        """
        atr = cls.calculate_atr(klines)

        if atr <= 0 or price <= 0:
            # Fallback: sabit SL/TP
            if signal == "LONG":
                sl_pct, tp_pct = 0.018, 0.054
            else:
                sl_pct, tp_pct = 0.018, 0.054
            return {
                "sl_price": round(price * (1 - sl_pct), 10) if signal == "LONG" else round(price * (1 + sl_pct), 10),
                "tp_price": round(price * (1 + tp_pct), 10) if signal == "LONG" else round(price * (1 - tp_pct), 10),
                "sl_pct": sl_pct,
                "tp_pct": tp_pct,
                "atr": 0,
                "method": "fixed",
            }

        # ATR bazli SL/TP
        atr_pct = atr / price

        # Kaldıraç ayarı: yüksek kaldıraç → daha dar SL/TP
        leverage_factor = 1.0 / (leverage / 5)

        sl_pct = atr_pct * 2.0 * leverage_factor
        tp_pct = sl_pct * risk_reward_ratio

        # Limitler
        sl_pct = float(np.clip(sl_pct, 0.005, 0.03))
        tp_pct = float(np.clip(tp_pct, 0.01, 0.08))

        if signal == "LONG":
            sl_price = round(price * (1 - sl_pct), 10)
            tp_price = round(price * (1 + tp_pct), 10)
        else:
            sl_price = round(price * (1 + sl_pct), 10)
            tp_price = round(price * (1 - tp_pct), 10)

        return {
            "sl_price": sl_price,
            "tp_price": tp_price,
            "sl_pct": round(sl_pct, 4),
            "tp_pct": round(tp_pct, 4),
            "atr": round(atr, 6),
            "method": "atr_based",
        }

    @staticmethod
    def calculate_trailing_stop(entry_price: float, current_price: float,
                                 highest_since_entry: float,
                                 signal: str, atr: float,
                                 trailing_pct: float = 0.5) -> float:
        """
        Trailing stop hesapla.

        Kar belirli bir eseri gectiyse, stop loss'u yukari tasir.
        """
        if atr <= 0:
            return 0.0

        if signal == "LONG":
            # Long pozisyonda: fiyat yukseldikce stop yukselir
            trailing_distance = atr * trailing_pct
            trailing_stop = highest_since_entry - trailing_distance

            # Entry fiyatindan asagi dusmesin
            trailing_stop = max(trailing_stop, entry_price * 0.99)
        else:
            # Short pozisyonda: fiyat dustukce stop asagi iner
            trailing_distance = atr * trailing_pct
            trailing_stop = highest_since_entry + trailing_distance

            # Entry fiyatindan yukari cikmasin
            trailing_stop = min(trailing_stop, entry_price * 1.01)

        return round(trailing_stop, 10)


# ══════════════════════════════════════════════════════════════════════════════
# EXPORT
# ══════════════════════════════════════════════════════════════════════════════

__all__ = [
    "AdvancedOrderBook",
    "MultiTimeframeConfirmation",
    "DynamicSLTP",
]
