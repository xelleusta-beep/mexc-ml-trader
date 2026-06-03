import numpy as np


class FeatureBuilder:
    """14 scalping-optimized features for Min5 data."""

    N_FEATURES = 14

    @classmethod
    def build(cls, klines):
        c = np.asarray(klines["close"], dtype=np.float64)
        h = np.asarray(klines["high"], dtype=np.float64)
        lo = np.asarray(klines["low"], dtype=np.float64)
        v = np.asarray(klines["volume"], dtype=np.float64)

        if len(c) < 25:
            return None

        pr = c[-1] + 1e-10
        f = []

        # 0: RSI(7) - hizli scalping
        f.append(_rsi(c, 7) / 100.0)

        # 1: MACD histogram / fiyat
        _, _, mh = _macd(c)
        f.append(float(np.clip(mh / pr, -0.02, 0.02)))

        # 2: EMA9 / EMA21 cross
        e9 = _ema(c, 9)[-1]
        e21 = _ema(c, 21)[-1]
        f.append(float(np.clip((e9 - e21) / pr, -0.02, 0.02)))

        # 3: ATR(7) / fiyat
        f.append(float(np.clip(_atr(h, lo, c, 7) / pr, 0, 0.03)))

        # 4: Volume ratio (current / avg)
        v_avg = float(np.mean(v[-11:-1])) + 1e-10
        f.append(float(np.clip(v[-1] / v_avg, 0, 5)))

        # 5: Bollinger pos (20, 2)
        bb_mid = float(np.mean(c[-20:]))
        bb_std = float(np.std(c[-20:])) + 1e-10
        f.append(float(np.clip((c[-1] - bb_mid) / (2 * bb_std), -1, 1)))

        # 6: Stochastic RSI(7)
        f.append(_stoch_rsi(c, 7) / 100.0)

        # 7: 2-bar return
        ret2 = (c[-1] - c[-3]) / (c[-3] + 1e-10)
        f.append(float(np.clip(ret2, -0.05, 0.05)))

        # 8: 5-bar return
        ret5 = (c[-1] - c[-6]) / (c[-6] + 1e-10)
        f.append(float(np.clip(ret5, -0.08, 0.08)))

        # 9: Volume momentum (son 2 bar vs onceki 5)
        v_short = float(np.mean(v[-3:-1]))
        v_long = float(np.mean(v[-8:-3])) + 1e-10
        f.append(float(np.clip(v_short / v_long, 0, 4)))

        # 10: Price / VWAP
        tp = (h[-10:] + lo[-10:] + c[-10:]) / 3
        vw = float(np.sum(tp * v[-10:]) / (np.sum(v[-10:]) + 1e-10))
        f.append(float(np.clip((c[-1] - vw) / (vw + 1e-10), -0.03, 0.03)))

        # 11: Candle body % (upper vs lower wick bias)
        body = abs(c[-1] - c[-2]) / pr if len(c) >= 2 else 0
        hl = (h[-1] - lo[-1]) / pr if len(c) >= 1 else 1
        f.append(float(np.clip(body / (hl + 1e-10), 0, 1)))

        # 12: Short-term momentum slope (3 bar)
        if len(c) >= 5:
            slope = float(np.polyfit(np.arange(3), c[-3:], 1)[0])
        else:
            slope = 0.0
        f.append(float(np.clip(slope / pr, -0.01, 0.01)))

        # 13: Price position in 10-bar range
        h10 = float(np.max(h[-10:]))
        l10 = float(np.min(lo[-10:]))
        rng = h10 - l10 + 1e-10
        f.append(float((c[-1] - l10) / rng))

        return np.array(f, dtype=np.float32)

    @classmethod
    def build_dataset(cls, klines, lookahead=2, threshold=0.003):
        c = np.asarray(klines["close"], dtype=np.float64)
        h = np.asarray(klines["high"], dtype=np.float64)
        lo = np.asarray(klines["low"], dtype=np.float64)
        v = np.asarray(klines["volume"], dtype=np.float64)

        MIN = 25
        if len(c) < MIN + lookahead:
            return None, None

        X_rows, y_rows = [], []
        for i in range(MIN, len(c) - lookahead):
            sub = {"close": c[:i], "high": h[:i], "low": lo[:i], "volume": v[:i]}
            feat = cls.build(sub)
            if feat is None:
                continue
            ret = float((c[i + lookahead] - c[i]) / (c[i] + 1e-10))
            if ret > threshold:
                label = 1
            elif ret < -threshold:
                label = -1
            else:
                label = 0
            X_rows.append(feat)
            y_rows.append(label)

        if len(X_rows) < 20:
            return None, None
        return np.array(X_rows, np.float32), np.array(y_rows, np.int32)


def _ema(c, p):
    k = 2 / (p + 1)
    out = np.empty(len(c))
    out[0] = c[0]
    for i in range(1, len(c)):
        out[i] = c[i] * k + out[i - 1] * (1 - k)
    return out


def _rsi(c, p):
    if len(c) < p + 2:
        return 50.0
    d = np.diff(c[-(p + 5):])
    g = np.where(d > 0, d, 0.0)
    lo = np.where(d < 0, -d, 0.0)
    ag = float(np.mean(g[-p:]))
    al = float(np.mean(lo[-p:]))
    return 100.0 if al < 1e-12 else float(100 - 100 / (1 + ag / al))


def _macd(c):
    if len(c) < 35:
        return 0.0, 0.0, 0.0
    e12 = _ema(c, 12)
    e26 = _ema(c, 26)
    ml = e12 - e26
    sig = _ema(ml, 9)
    return float(ml[-1]), float(sig[-1]), float(ml[-1] - sig[-1])


def _atr(h, lo, c, p):
    if len(c) < p + 1:
        return 0.0
    tr = np.maximum(h[1:] - lo[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(lo[1:] - c[:-1])))
    return float(np.mean(tr[-p:]))


def _stoch_rsi(c, p):
    n = len(c)
    if n < p * 2:
        return 50.0
    rvals = [_rsi(c[:i], p) for i in range(n - p + 1, n + 1)]
    arr = np.array(rvals)
    mn, mx = arr.min(), arr.max()
    return float((arr[-1] - mn) / (mx - mn + 1e-10) * 100)
