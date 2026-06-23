import numpy as np
from typing import Tuple


def calculate_rsi(close_prices: list[float], period: int = 21) -> list[float | None]:
    """
    Wilder's RSI hesaplar.
    İlk ortalama için SMA, sonrasında Wilder's smoothing (EMA benzeri) kullanır.
    """
    if len(close_prices) < period + 1:
        return [None] * len(close_prices)

    prices = np.array(close_prices, dtype=float)
    deltas = np.diff(prices)

    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    rsi_values = [None] * period

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    if avg_loss == 0:
        rsi_values.append(100.0)
    else:
        rs = avg_gain / avg_loss
        rsi_values.append(100.0 - (100.0 / (1.0 + rs)))

    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        if avg_loss == 0:
            rsi_values.append(100.0)
        else:
            rs = avg_gain / avg_loss
            rsi_values.append(100.0 - (100.0 / (1.0 + rs)))

    return rsi_values


def calculate_rsi_ma(rsi_values: list[float | None], period: int = 21) -> list[float | None]:
    """
    RSI'nin hareketli ortalamasını hesaplar (Sarı çizgi).
    Wilder's smoothing (EMA benzeri) kullanır - TradingView ile uyumlu.
    None değerleri atlar.
    """
    result = []
    multiplier = 2.0 / (period + 1)  # EMA multiplier
    prev_ema = None

    for i in range(len(rsi_values)):
        if rsi_values[i] is None:
            result.append(None)
            continue

        if prev_ema is None:
            # İlk değer için SMA kullan
            window = [v for v in rsi_values[max(0, i - period + 1): i + 1] if v is not None]
            if len(window) >= period:
                prev_ema = sum(window) / len(window)
                result.append(prev_ema)
            else:
                result.append(None)
        else:
            # Wilder's smoothing (EMA)
            prev_ema = (rsi_values[i] - prev_ema) * multiplier + prev_ema
            result.append(prev_ema)

    return result


def calculate_indicators(
    close_prices: list[float],
    rsi_period: int = 21,
    rsi_ma_period: int = 21,
) -> Tuple[list[float | None], list[float | None]]:
    """RSI ve RSI-based MA'yı birlikte hesaplar."""
    rsi = calculate_rsi(close_prices, rsi_period)
    rsi_ma = calculate_rsi_ma(rsi, rsi_ma_period)
    return rsi, rsi_ma


def calculate_ema(close_prices: list[float], period: int) -> list[float | None]:
    """EMA (Exponential Moving Average) hesaplar."""
    if len(close_prices) < period:
        return [None] * len(close_prices)

    multiplier = 2.0 / (period + 1)
    result = [None] * (period - 1)

    # İlk EMA için SMA kullan
    sma = sum(close_prices[:period]) / period
    result.append(sma)

    for i in range(period, len(close_prices)):
        ema = (close_prices[i] - result[-1]) * multiplier + result[-1]
        result.append(ema)

    return result


def calculate_trend_signal(
    close_prices: list[float],
    fast_period: int = 20,
    slow_period: int = 50,
) -> Tuple[list[float | None], list[float | None], list[str]]:
    """
    EMA crossover trend sinyali hesaplar.
    Returns: (ema_fast, ema_slow, signals)
    signal: 'buy' (fast > slow), 'sell' (fast < slow), 'none'
    """
    ema_fast = calculate_ema(close_prices, fast_period)
    ema_slow = calculate_ema(close_prices, slow_period)

    signals = []
    prev_signal = 'none'

    for i in range(len(close_prices)):
        if ema_fast[i] is None or ema_slow[i] is None:
            signals.append('none')
            continue

        if ema_fast[i] > ema_slow[i]:
            current_signal = 'buy'
        else:
            current_signal = 'sell'

        # Kesişim tespiti
        if prev_signal == 'sell' and current_signal == 'buy':
            signals.append('buy')  # Fast, slow'u yukarı kesti
        elif prev_signal == 'buy' and current_signal == 'sell':
            signals.append('sell')  # Fast, slow'u aşağı kesti
        else:
            signals.append('none')

        prev_signal = current_signal

    return ema_fast, ema_slow, signals


# ==================== PROFESYONEL GÖSTERGELER ====================

def calculate_adx(high_prices: list[float], low_prices: list[float], close_prices: list[float], period: int = 14) -> list[float | None]:
    """ADX (Average Directional Index) - Trend gücü göstergesi. 25+ güçlü trend."""
    n = len(close_prices)
    if n < period * 2:
        return [None] * n

    # True Range hesapla
    tr_list = [high_prices[0] - low_prices[0]]
    plus_dm = [0.0]
    minus_dm = [0.0]

    for i in range(1, n):
        high_diff = high_prices[i] - high_prices[i-1]
        low_diff = low_prices[i-1] - low_prices[i]

        tr = max(high_prices[i] - low_prices[i], abs(high_prices[i] - close_prices[i-1]), abs(low_prices[i] - close_prices[i-1]))
        tr_list.append(tr)

        plus_dm.append(high_diff if high_diff > low_diff and high_diff > 0 else 0)
        minus_dm.append(low_diff if low_diff > high_diff and low_diff > 0 else 0)

    # Wilder's smoothing
    atr = sum(tr_list[:period])
    smooth_plus_dm = sum(plus_dm[:period])
    smooth_minus_dm = sum(minus_dm[:period])

    dx_values = []
    for i in range(period, n):
        atr = atr - atr / period + tr_list[i]
        smooth_plus_dm = smooth_plus_dm - smooth_plus_dm / period + plus_dm[i]
        smooth_minus_dm = smooth_minus_dm - smooth_minus_dm / period + minus_dm[i]

        if atr == 0:
            dx_values.append(0)
            continue

        plus_di = (smooth_plus_dm / atr) * 100
        minus_di = (smooth_minus_dm / atr) * 100

        di_sum = plus_di + minus_di
        if di_sum == 0:
            dx_values.append(0)
        else:
            dx_values.append(abs(plus_di - minus_di) / di_sum * 100)

    # ADX = DX'nin EMA'sı
    result = [None] * (period * 2 - 1)
    if len(dx_values) >= period:
        adx = sum(dx_values[:period]) / period
        result.append(adx)
        for dx in dx_values[period:]:
            adx = (adx * (period - 1) + dx) / period
            result.append(adx)

    return result


def calculate_atr(high_prices: list[float], low_prices: list[float], close_prices: list[float], period: int = 14) -> list[float | None]:
    """ATR (Average True Range) - Volatilite göstergesi."""
    n = len(close_prices)
    if n < period:
        return [None] * n

    tr_list = [high_prices[0] - low_prices[0]]
    for i in range(1, n):
        tr = max(
            high_prices[i] - low_prices[i],
            abs(high_prices[i] - close_prices[i-1]),
            abs(low_prices[i] - close_prices[i-1])
        )
        tr_list.append(tr)

    result = [None] * (period - 1)
    atr = sum(tr_list[:period]) / period
    result.append(atr)

    for i in range(period, n):
        atr = (atr * (period - 1) + tr_list[i]) / period
        result.append(atr)

    return result


def calculate_macd(close_prices: list[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[list[float | None], list[float | None], list[float | None]]:
    """MACD - Momentum göstergesi. (MACD, Signal, Histogram)"""
    ema_fast = calculate_ema(close_prices, fast)
    ema_slow = calculate_ema(close_prices, slow)

    macd_line = []
    for i in range(len(close_prices)):
        if ema_fast[i] is not None and ema_slow[i] is not None:
            macd_line.append(ema_fast[i] - ema_slow[i])
        else:
            macd_line.append(None)

    # Signal line (MACD'nin EMA'sı)
    valid_macd = [v for v in macd_line if v is not None]
    signal_line = [None] * len(close_prices)

    if len(valid_macd) >= signal:
        multiplier = 2.0 / (signal + 1)
        sma = sum(valid_macd[:signal]) / signal
        signal_idx = 0
        for i in range(len(close_prices)):
            if macd_line[i] is not None:
                signal_idx += 1
                if signal_idx == signal:
                    signal_line[i] = sma
                elif signal_idx > signal:
                    signal_line[i] = (macd_line[i] - signal_line[i-1]) * multiplier + signal_line[i-1]

    # Histogram
    histogram = []
    for i in range(len(close_prices)):
        if macd_line[i] is not None and signal_line[i] is not None:
            histogram.append(macd_line[i] - signal_line[i])
        else:
            histogram.append(None)

    return macd_line, signal_line, histogram


def calculate_bollinger_bands(close_prices: list[float], period: int = 20, std_dev: float = 2.0) -> Tuple[list[float | None], list[float | None], list[float | None]]:
    """Bollinger Bands - Volatilite bandı. (Upper, Middle, Lower)"""
    n = len(close_prices)
    middle = calculate_ema(close_prices, period)  # SMA yerine EMA kullanıyoruz

    upper = []
    lower = []

    for i in range(n):
        if middle[i] is None:
            upper.append(None)
            lower.append(None)
            continue

        window = close_prices[max(0, i - period + 1):i + 1]
        if len(window) < 2:
            upper.append(None)
            lower.append(None)
            continue

        std = np.std(window, ddof=1)
        upper.append(middle[i] + std_dev * std)
        lower.append(middle[i] - std_dev * std)

    return upper, middle, lower


def calculate_volume_sma(volumes: list[float], period: int = 20) -> list[float | None]:
    """Hacim ortalaması."""
    result = []
    for i in range(len(volumes)):
        if i < period - 1:
            result.append(None)
        else:
            result.append(sum(volumes[i - period + 1:i + 1]) / period)
    return result


def calculate_stochastic_rsi(close_prices: list[float], rsi_period: int = 14, stoch_period: int = 14, k_period: int = 3, d_period: int = 3) -> Tuple[list[float | None], list[float | None]]:
    """Stochastic RSI - Aşırı alım/satım göstergesi."""
    rsi = calculate_rsi(close_prices, rsi_period)
    n = len(rsi)

    k_values = []
    for i in range(n):
        if rsi[i] is None:
            k_values.append(None)
            continue
        window = [v for v in rsi[max(0, i - stoch_period + 1):i + 1] if v is not None]
        if len(window) < 2:
            k_values.append(None)
            continue
        rsi_min = min(window)
        rsi_max = max(window)
        if rsi_max == rsi_min:
            k_values.append(50.0)
        else:
            k_values.append((rsi[i] - rsi_min) / (rsi_max - rsi_min) * 100)

    # %K SMA
    k_smooth = []
    for i in range(n):
        if k_values[i] is None:
            k_smooth.append(None)
            continue
        window = [v for v in k_values[max(0, i - k_period + 1):i + 1] if v is not None]
        if len(window) >= k_period:
            k_smooth.append(sum(window) / len(window))
        else:
            k_smooth.append(None)

    # %D SMA
    d_values = []
    for i in range(n):
        if k_smooth[i] is None:
            d_values.append(None)
            continue
        window = [v for v in k_smooth[max(0, i - d_period + 1):i + 1] if v is not None]
        if len(window) >= d_period:
            d_values.append(sum(window) / len(window))
        else:
            d_values.append(None)

    return k_smooth, d_values
