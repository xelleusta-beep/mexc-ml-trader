"""
MEXC ML Trading System — Advanced Feature Engineering v1.0
Extended indicators, feature pipeline, selection, importance tracking.
"""

import numpy as np
from collections import deque
from typing import Optional, Callable


# ── Feature Registry ──────────────────────────────────────────────────────────

class FeatureRegistry:
    """
    Feature tanimlama ve metadata.
    name -> (compute_fn, category, description)
    """
    def __init__(self):
        self._registry: dict = {}

    def register(self, name: str, category: str, description: str = ""):
        def decorator(fn):
            self._registry[name] = {"fn": fn, "cat": category, "desc": description}
            return fn
        return decorator

    def list_features(self, category: str = "") -> list:
        if category:
            return [n for n, v in self._registry.items() if v["cat"] == category]
        return list(self._registry.keys())

    def compute(self, name: str, c, h, lo, v, **kw) -> float:
        entry = self._registry.get(name)
        if not entry:
            return 0.0
        try:
            return float(entry["fn"](c, h, lo, v, **kw))
        except Exception:
            return 0.0


FEATURES = FeatureRegistry()


# ── Core Indicators (v1, optimized) ──────────────────────────────────────────

class Indicators:
    """Mevcut indicator'ler korunur, yenileri eklenir."""

    @staticmethod
    def rsi(c, p=14):
        if len(c) < p + 2: return 50.0
        d = np.diff(c[-(p+5):])
        g = np.where(d > 0, d, 0.0); lo_ = np.where(d < 0, -d, 0.0)
        ag, al = np.mean(g[-p:]), np.mean(lo_[-p:])
        return 100.0 if al < 1e-12 else float(100 - 100 / (1 + ag / al))

    @staticmethod
    def ema(c, p):
        if len(c) == 0: return np.zeros(1)
        k = 2 / (p + 1); out = np.empty(len(c)); out[0] = c[0]
        for i in range(1, len(c)): out[i] = c[i] * k + out[i-1] * (1 - k)
        return out

    @staticmethod
    def macd(c):
        if len(c) < 35: return 0.0, 0.0, 0.0
        e12 = Indicators.ema(c, 12); e26 = Indicators.ema(c, 26); ml = e12 - e26
        sig = Indicators.ema(ml, 9)
        return float(ml[-1]), float(sig[-1]), float(ml[-1] - sig[-1])

    @staticmethod
    def bb(c, p=20):
        if len(c) < p: v = float(c[-1]) if len(c) else 0; return v, v, v
        s, std = float(np.mean(c[-p:])), float(np.std(c[-p:]))
        return s + 2*std, s, s - 2*std

    @staticmethod
    def atr(h, lo, c, p=14):
        if len(c) < p + 1: return 0.0
        tr = np.maximum(h[1:]-lo[1:], np.maximum(np.abs(h[1:]-c[:-1]), np.abs(lo[1:]-c[:-1])))
        return float(np.mean(tr[-p:]))


# ── New Indicator Functions (registered) ──────────────────────────────────────

@FEATURES.register("rsi_divergence", "momentum", "RSI divergence signal: 1=bullish, -1=bearish, 0=none")
def rsi_divergence(c, h, lo, v, **kw):
    """Price-RSI divergence: price makes lower low but RSI makes higher low → bullish."""
    if len(c) < 30: return 0.0
    rsi_vals = np.array([Indicators.rsi(c[:i], 14) for i in range(max(14, len(c)-20), len(c)+1)])
    px = c[-(len(rsi_vals)):]
    if len(rsi_vals) < 5: return 0.0
    # Son 5 bardaki local min/max
    px_min_idx = np.argmin(px[-5:]) if len(px) >= 5 else -1
    rsi_min_idx = np.argmin(rsi_vals[-5:]) if len(rsi_vals) >= 5 else -1
    if px_min_idx > 0 and rsi_min_idx > 0:
        px_lower = px[-5 + px_min_idx] < px[-5 + px_min_idx - 1]
        rsi_higher = rsi_vals[-5 + rsi_min_idx] > rsi_vals[-5 + min(rsi_min_idx + 1, 4)]
        if px_lower and rsi_higher:
            return 1.0  # bullish divergence
    # Bearish divergence
    px_max_idx = np.argmax(px[-5:])
    rsi_max_idx = np.argmax(rsi_vals[-5:])
    if px_max_idx > 0 and rsi_max_idx > 0:
        px_higher = px[-5 + px_max_idx] > px[-5 + px_max_idx - 1]
        rsi_lower = rsi_vals[-5 + rsi_max_idx] < rsi_vals[-5 + min(rsi_max_idx + 1, 4)]
        if px_higher and rsi_lower:
            return -1.0
    return 0.0


@FEATURES.register("macd_histogram_momentum", "momentum", "MACD histogram slope (momentum acceleration)")
def macd_histogram_momentum(c, h, lo, v, **kw):
    """MACD histograminin son 3 bardaki egimi → momentum ivmesi."""
    if len(c) < 40: return 0.0
    hist_vals = []
    for i in range(max(35, len(c)-5), len(c)+1):
        _, _, hist = Indicators.macd(c[:i])
        hist_vals.append(hist)
    if len(hist_vals) < 3: return 0.0
    arr = np.array(hist_vals)
    slope = float(np.polyfit(np.arange(len(arr)), arr, 1)[0])
    return float(np.clip(slope * 1000, -1, 1))


@FEATURES.register("bb_width_pct", "volatility", "Bollinger Band width as % of price")
def bb_width_pct(c, h, lo, v, **kw):
    if len(c) < 20: return 0.0
    s, std = float(np.mean(c[-20:])), float(np.std(c[-20:]))
    return float(np.clip(std / (s + 1e-10), 0, 0.05)) * 100


@FEATURES.register("atr_slope", "volatility", "ATR slope over last 5 bars")
def atr_slope(c, h, lo, v, **kw):
    """ATR egimi: volatilite artiyor/azaliyor."""
    if len(c) < 25: return 0.0
    atrs = [Indicators.atr(h[:i], lo[:i], c[:i], 14) for i in range(max(20, len(c)-5), len(c)+1)]
    if len(atrs) < 3: return 0.0
    slope = float(np.polyfit(np.arange(len(atrs)), atrs, 1)[0])
    ref = float(np.mean(atrs) + 1e-10)
    return float(np.clip(slope / ref, -1, 1))


@FEATURES.register("volatility_regime", "volatility", "Vol regime: 0=low, 1=medium, 2=high")
def volatility_regime(c, h, lo, v, **kw):
    """ATR percentile based regime classification."""
    if len(c) < 30: return 1.0
    atr_now = Indicators.atr(h, lo, c, 14)
    hist = [Indicators.atr(h[:i], lo[:i], c[:i], 14) for i in range(max(20, len(c)-20), len(c))]
    if not hist: return 1.0
    pct = float(np.mean(np.array(hist) < atr_now))
    if pct < 0.3: return 0.0
    if pct > 0.7: return 2.0
    return 1.0


@FEATURES.register("price_acceleration", "momentum", "2nd derivative of price (acceleration)")
def price_acceleration(c, h, lo, v, **kw):
    """Fiyat ivmesi: son 5 bardaki egim degisimi."""
    if len(c) < 7: return 0.0
    y = c[-5:]; x = np.arange(5)
    slope1 = float(np.polyfit(x, y, 1)[0])
    y2 = c[-7:-2]; x2 = np.arange(5)
    slope0 = float(np.polyfit(x2, y2, 1)[0])
    acc = (slope1 - slope0) / (float(np.mean(c[-5:])) + 1e-10)
    return float(np.clip(acc * 100, -1, 1))


@FEATURES.register("efficient_ratio", "momentum", "Efficiency Ratio (EMA of directionality)")
def efficient_ratio(c, h, lo, v, **kw):
    """Kaufman Efficiency Ratio: directionality / noise."""
    if len(c) < 12: return 0.0
    direction = abs(float(c[-1] - c[-10]))
    noise = float(np.sum(np.abs(np.diff(c[-10:]))))
    if noise < 1e-10: return 1.0
    return float(np.clip(direction / noise, 0, 1))


@FEATURES.register("kama", "trend", "Kaufman Adaptive MA delta from price")
def kama(c, h, lo, v, **kw):
    """KAMA - price distance. Fast in trends, slow in chop."""
    if len(c) < 15: return 0.0
    er = efficient_ratio(c, h, lo, v)
    fast_sc, slow_sc = 2/(2+1), 2/(30+1)
    sc = float(er * (fast_sc - slow_sc) + slow_sc)
    kama_val = c[-10]
    for i in range(-9, 0):
        kama_val = kama_val + sc * (c[i] - kama_val)
    return float(np.clip((c[-1] - kama_val) / (kama_val + 1e-10), -0.05, 0.05))


@FEATURES.register("adx", "trend", "Average Directional Index (trend strength)")
def adx(c, h, lo, v, **kw):
    """ADX: trend guclu mu (25+) yoksa zayif mi."""
    if len(c) < 20: return 0.0
    tr = np.maximum(h[1:]-lo[1:], np.maximum(np.abs(h[1:]-c[:-1]), np.abs(lo[1:]-c[:-1])))
    up_move = h[1:] - h[:-1]; down_move = lo[:-1] - lo[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr_smooth = np.mean(tr[-14:]) + 1e-10
    plus_di = 100 * np.mean(plus_dm[-14:]) / atr_smooth
    minus_di = 100 * np.mean(minus_dm[-14:]) / atr_smooth
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    return float(np.clip(dx / 100, 0, 1))


@FEATURES.register("volume_ratio_ema", "volume", "Volume ratio (current / EMA of volume)")
def volume_ratio_ema(c, h, lo, v, **kw):
    if len(v) < 10: return 0.0
    v_ema = float(np.mean(v[-5:]) + 1e-10)
    v_hist = float(np.mean(v[-20:-5]) + 1e-10)
    return float(np.clip(v_ema / v_hist, 0, 5))


@FEATURES.register("volume_price_trend", "volume", "VPT indicator (volume-price confirmation)")
def volume_price_trend(c, h, lo, v, **kw):
    """Volume-Price Trend: hacim fiyati dogruluyor mu."""
    if len(c) < 15: return 0.0
    vpt = 0.0
    for i in range(1, 15):
        ret = (c[-i] - c[-i-1]) / (c[-i-1] + 1e-10)
        vpt += ret * v[-i]
    vpt /= 14
    return float(np.clip(vpt * 1000, -1, 1))


@FEATURES.register("cumulative_delta", "volume", "CVD approximation (cumulative volume delta)")
def cumulative_delta(c, h, lo, v, **kw):
    """Approximate CVD: up-volume minus down-volume over last 10 bars."""
    if len(c) < 12: return 0.0
    cvd = 0.0
    for i in range(1, 11):
        if c[-i] > c[-i-1]: cvd += v[-i]
        elif c[-i] < c[-i-1]: cvd -= v[-i]
    avg_v = float(np.mean(v[-11:]) + 1e-10)
    return float(np.clip(cvd / avg_v, -3, 3))


@FEATURES.register("pivot_distance", "price_action", "Distance from nearest pivot high/low")
def pivot_distance(c, h, lo, v, **kw):
    """Son pivot high/low'a uzaklik."""
    if len(c) < 15: return 0.0
    pivots_high, pivots_low = [], []
    for i in range(2, min(13, len(c)-1)):
        if h[-i] > h[-i-1] and h[-i] > h[-i+1]:
            pivots_high.append((i, h[-i]))
        if lo[-i] < lo[-i-1] and lo[-i] < lo[-i+1]:
            pivots_low.append((i, lo[-i]))
    if pivots_high and pivots_low:
        nearest_high = min((abs(c[-1] - ph[1]) / (c[-1] + 1e-10), ph[0]) for ph in pivots_high)
        nearest_low  = min((abs(c[-1] - pl[1]) / (c[-1] + 1e-10), pl[0]) for pl in pivots_low)
        return float(np.clip(nearest_low[0] - nearest_high[0], -0.05, 0.05))
    return 0.0


@FEATURES.register("hour_of_day", "time", "Hour of day (0-23) sin/cos encoded")
def hour_of_day_sin(c, h, lo, v, **kw):
    """Hour of day as sin component (cyclical encoding)."""
    ts = kw.get("timestamp")
    if ts is None: return 0.0
    try: hour = ts.hour + ts.minute / 60
    except: return 0.0
    return float(np.sin(2 * np.pi * hour / 24))


@FEATURES.register("hour_of_day_cos", "time", "Hour of day cos component")
def hour_of_day_cos(c, h, lo, v, **kw):
    ts = kw.get("timestamp")
    if ts is None: return 0.0
    try: hour = ts.hour + ts.minute / 60
    except: return 0.0
    return float(np.cos(2 * np.pi * hour / 24))


@FEATURES.register("day_of_week", "time", "Day of week sin component")
def day_of_week_sin(c, h, lo, v, **kw):
    ts = kw.get("timestamp")
    if ts is None: return 0.0
    try: dow = ts.weekday() + ts.hour / 24
    except: return 0.0
    return float(np.sin(2 * np.pi * dow / 7))


@FEATURES.register("day_of_week_cos", "time", "Day of week cos component")
def day_of_week_cos(c, h, lo, v, **kw):
    ts = kw.get("timestamp")
    if ts is None: return 0.0
    try: dow = ts.weekday() + ts.hour / 24
    except: return 0.0
    return float(np.cos(2 * np.pi * dow / 7))


@FEATURES.register("session", "time", "Trading session: 0=asia, 1=eu, 2=us, 3=overlap")
def trading_session(c, h, lo, v, **kw):
    ts = kw.get("timestamp")
    if ts is None: return 0.0
    try:
        h24 = ts.hour + ts.minute / 60
        if 0 <= h24 < 8: return 0.0     # Asia
        if 8 <= h24 < 13: return 1.0    # EU
        if 13 <= h24 < 17: return 2.0   # EU-US overlap
        return 3.0                       # US
    except:
        return 0.0


@FEATURES.register("kurtosis", "distribution", "Price distribution kurtosis (tail risk)")
def price_kurtosis(c, h, lo, v, **kw):
    if len(c) < 20: return 0.0
    rets = np.diff(c[-20:]) / (c[-21:-1] + 1e-10)
    if len(rets) < 5: return 0.0
    return float(np.clip(np.mean(rets ** 4) / (np.std(rets) ** 4 + 1e-10) - 3, -1, 5)) / 5


@FEATURES.register("entropy", "distribution", "Approximate entropy of price changes")
def price_entropy(c, h, lo, v, **kw):
    """Approximate entropy: ne kadar kaotik/rastgele."""
    if len(c) < 15: return 0.0
    rets = np.diff(c[-15:]) / (c[-16:-1] + 1e-10)
    bins = np.linspace(-0.02, 0.02, 10)
    hist, _ = np.histogram(rets, bins=bins)
    hist = hist + 1e-10
    prob = hist / np.sum(hist)
    ent = -np.sum(prob * np.log(prob))
    return float(np.clip(ent / np.log(10), 0, 1))


# ── V2 Feature Builder ───────────────────────────────────────────────────────

class FeatureBuilderV2:
    """
    Gelişmiş feature builder:
    - Mevcut 36 ozelligi + ~28 yeni ozelligi birlestirir
    - Feature importance takibi
    - Feature caching
    - Otomatik normalizasyon
    """

    N_FEATURES_BASE = 36     # Mevcut ozellik sayisi
    N_FEATURES_NEW  = 20     # Yeni ozellik sayisi
    N_FEATURES_TOTAL = 64    # Toplam

    def __init__(self, use_new_features: bool = True, cache_size: int = 200):
        self._use_new = use_new_features
        self._feature_names: list = []
        self._importance: dict = {}
        self._cache: dict = {}  # symbol -> (last_klines_hash, features)
        self._cache_size = cache_size

        # Feature isimlerini olustur
        self._init_feature_names()

    def _init_feature_names(self):
        """Base feature names."""
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
        self._feature_names = list(base_names)
        if self._use_new:
            self._feature_names.extend(FEATURES.list_features())

    @property
    def n_features(self) -> int:
        return len(self._feature_names)

    @property
    def feature_names(self) -> list:
        return list(self._feature_names)

    def get_importance(self, top_k: int = 10) -> list:
        """En onemli K ozelligi getir."""
        sorted_f = sorted(self._importance.items(), key=lambda x: x[1], reverse=True)
        return [(n, round(v, 4)) for n, v in sorted_f[:top_k]]

    def update_importance(self, importances: np.ndarray):
        """Model egitiminden gelen feature importance'lari kaydet."""
        if len(importances) != len(self._feature_names):
            return
        self._importance = {
            name: float(imp)
            for name, imp in zip(self._feature_names, importances)
        }

    def _cache_key(self, klines) -> str:
        """Klines hash (hizli)."""
        close = klines.get("close", [])
        if len(close) == 0:
            return ""
        return f"{close[-1]:.8f}_{close[-3] if len(close) > 3 else 0:.8f}_{len(close)}"

    def build(self, klines, timestamp=None) -> np.ndarray:
        """
        Feature vektoru olustur.
        timestamp: opsiyonel datetime (zaman bazli ozellikler icin).
        """
        c  = np.asarray(klines["close"],  dtype=np.float64)
        h  = np.asarray(klines["high"],   dtype=np.float64)
        lo = np.asarray(klines["low"],    dtype=np.float64)
        v  = np.asarray(klines["volume"], dtype=np.float64)
        if len(c) < 30: return None

        # Base features (inline copy of build() from ml_engine)
        base = self._build_base(c, h, lo, v)
        if base is None: return None

        if not self._use_new:
            return base

        # New features
        new_feats = []
        for fname in FEATURES.list_features():
            val = FEATURES.compute(fname, c, h, lo, v, timestamp=timestamp)
            new_feats.append(val)

        arr = np.concatenate([base, np.array(new_feats, dtype=np.float32)])
        return np.where(np.isfinite(arr), arr, 0.0)

    def _build_base(self, c, h, lo, v) -> np.ndarray:
        """Copy of original FeatureBuilder.build()."""
        # pylint: disable=import-outside-toplevel
        from ml_engine import Indicators as Ind
        if len(c) < 30: return None
        f = []; pr = c[-1] + 1e-10

        f.append(Ind.rsi(c, 14) / 100.0)
        f.append(Ind.stoch_rsi(c, 14) / 100.0)
        f.append(Ind.mfi(h, lo, c, v, 14) / 100.0)
        f.append(Ind.williams_r(h, lo, c, 14) / -100.0)

        ml, ms, mh = Ind.macd(c)
        f.extend([np.clip(ml/pr,-0.05,0.05), np.clip(ms/pr,-0.05,0.05), np.clip(mh/pr,-0.05,0.05)])
        e9  = Ind.ema(c, 9)[-1]; e21 = Ind.ema(c, 21)[-1]; e50 = Ind.ema(c, min(50,len(c)-1))[-1]
        f.append(np.clip((e9 - e21) / pr, -0.05, 0.05))
        f.append(np.clip((c[-1] - e50) / pr, -0.1, 0.1))

        bbu, _, bbl = Ind.bb(c, 20); bb_rng = bbu - bbl + 1e-10
        f.append(np.clip((c[-1] - bbl) / bb_rng, 0, 1))
        f.append(np.clip(bb_rng / pr, 0, 0.2))
        f.append(np.clip(Ind.atr(h, lo, c, 14) / pr, 0, 0.05))

        vavg = float(np.mean(v[-21:-1]) if len(v) >= 21 else v.mean()) + 1e-10
        f.append(np.clip(v[-1] / vavg, 0, 5))
        f.append(Ind.obv_trend(c, v))

        for lag in [1, 2, 3]:
            ret = (c[-1]-c[-1-lag])/(c[-1-lag]+1e-10) if len(c) > lag else 0.0
            f.append(np.clip(ret, -0.1, 0.1))
        for w in [5, 10, 20]:
            ret = (c[-1]-c[-1-w])/(c[-1-w]+1e-10) if len(c) > w else 0.0
            f.append(np.clip(ret, -0.3, 0.3))
        for w in [5, 20]:
            s = float(np.std(c[-w:])) / pr if len(c) >= w else 0.0
            f.append(np.clip(s, 0, 0.1))

        h20, l20 = float(np.max(h[-20:])), float(np.min(lo[-20:])); rng = h20 - l20 + 1e-10
        f.append((c[-1] - l20) / rng); f.append((h20 - c[-1]) / rng)

        m5 = c[-1]-c[-6] if len(c) >= 6 else 0.0
        mp = c[-2]-c[-7] if len(c) >= 7 else 0.0
        f.append(np.clip((m5 - mp) / pr, -0.05, 0.05))

        body = abs(c[-1]-c[-2])/pr if len(c) >= 2 else 0.0
        f.append(np.clip(body, 0, 0.05))
        hl = (h[-1]-lo[-1])/pr; f.append(np.clip(hl, 0, 0.1))
        uw = (h[-1]-max(c[-1], c[-2] if len(c)>=2 else c[-1])) / (h[-1]-lo[-1]+1e-10)
        f.append(np.clip(float(uw), 0, 1))

        f.append(Ind.vwap_dist(h, lo, c, v))
        f.append(Ind.momentum_slope(c, 10))

        rsi_now  = Ind.rsi(c, 14)
        rsi_prev = Ind.rsi(c[:-3], 14) if len(c) > 17 else rsi_now
        price_chg = (c[-1]-c[-4])/(c[-4]+1e-10) if len(c) > 4 else 0.0
        f.append(float(np.clip(price_chg * (-(rsi_now-rsi_prev)/100.0), -0.05, 0.05)))
        f.append(Ind.market_regime(c, v))

        v_slope = float(np.polyfit(np.arange(5), v[-5:], 1)[0]) if len(v) >= 5 else 0.0
        f.append(float(np.clip(v_slope / (float(np.mean(v[-5:])) + 1e-10), -2, 2)))

        lower_wick = (min(c[-1], c[-2] if len(c)>=2 else c[-1])-lo[-1]) / (h[-1]-lo[-1]+1e-10)
        f.append(float(np.clip(lower_wick, 0, 1)))

        e9a = Ind.ema(c, 9); e21a = Ind.ema(c, 21)
        cross_now  = (e9a[-1]-e21a[-1])/pr
        cross_prev = (e9a[-2]-e21a[-2])/pr if len(c) > 2 else cross_now
        f.append(float(np.clip(cross_now - cross_prev, -0.01, 0.01)))
        f.append(Ind.atr_percentile(h, lo, c))

        arr = np.array(f, dtype=np.float32)
        return np.where(np.isfinite(arr), arr, 0.0)

    def build_dataset(self, klines, lookahead=3, threshold=0.004, timestamp_fn=None):
        """Dataset olustur (tum bar'lar icin)."""
        c  = np.asarray(klines["close"],  dtype=np.float64)
        h  = np.asarray(klines["high"],   dtype=np.float64)
        lo = np.asarray(klines["low"],    dtype=np.float64)
        v  = np.asarray(klines["volume"], dtype=np.float64)
        MIN = 40
        if len(c) < MIN + lookahead: return None, None, None

        if len(c) >= 50:
            avg_move = float(np.mean(np.abs(np.diff(c[-50:])) / (c[-50:-1] + 1e-10)))
            dyn_thr  = float(np.clip(avg_move * 0.6, 0.002, 0.006))
        else:
            dyn_thr = threshold * 0.7

        X_rows, y_rows = [], []
        timestamps = klines.get("timestamp", [])
        for i in range(MIN, len(c) - lookahead):
            sub  = {"close": c[:i], "high": h[:i], "low": lo[:i], "volume": v[:i]}
            ts = timestamp_fn(i) if timestamp_fn else None
            if self._use_new and timestamp_fn:
                feat = self.build(sub, timestamp=ts)
            else:
                feat = self.build(sub)
            if feat is None: continue
            ret   = (c[i+lookahead] - c[i]) / (c[i] + 1e-10)
            label = 1 if ret > dyn_thr else (-1 if ret < -dyn_thr else 0)
            X_rows.append(feat); y_rows.append(label)

        if not X_rows: return None, None, None
        X = np.array(X_rows, dtype=np.float32)
        y = np.array(y_rows, dtype=np.int8)
        return X, y, self._feature_names


# ── Feature Selector ─────────────────────────────────────────────────────────

class FeatureSelector:
    """
    Feature secimi:
    - Correlation bazli: yuksek korelasyonlu feature'lari grupla
    - Importance bazli: dusuk importance filtresi
    - PCA opsiyonu
    """

    def __init__(self, max_features: int = 48, corr_threshold: float = 0.85):
        self._max_features = max_features
        self._corr_threshold = corr_threshold
        self._selected_indices: Optional[list] = None
        self._feature_names: list = []

    def fit(self, X: np.ndarray, importances: np.ndarray, feature_names: list):
        """
        Feature setini optimize et:
        1. Importance < 0.01 olanlari at
        2. Pairwise correlation > threshold olanlardan birini tut
        3. Max features sinirina kadar kes
        """
        self._feature_names = list(feature_names)
        n = X.shape[1]
        if n == 0:
            self._selected_indices = []
            return

        # Importance filtre
        imp_norm = np.abs(importances) / (np.max(np.abs(importances)) + 1e-10)
        keep = np.where(imp_norm > 0.01)[0]
        if len(keep) < 10:
            keep = np.argsort(-imp_norm)[:20]

        # Correlation filtre
        X_sub = X[:, keep]
        corr = np.corrcoef(X_sub.T)
        final_keep = []
        for i in range(len(keep)):
            keep_i = True
            for j in final_keep:
                if abs(corr[i, j]) > self._corr_threshold:
                    keep_i = False
                    break
            if keep_i:
                final_keep.append(i)

        keep = keep[final_keep]

        # Max features
        if len(keep) > self._max_features:
            keep = keep[np.argsort(-imp_norm[keep])[:self._max_features]]

        self._selected_indices = sorted(keep.tolist())

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self._selected_indices is None:
            return X
        return X[:, self._selected_indices]

    def get_selected_names(self) -> list:
        if self._selected_indices is None:
            return self._feature_names
        return [self._feature_names[i] for i in self._selected_indices]

    @property
    def n_selected(self) -> int:
        return len(self._selected_indices or [])
