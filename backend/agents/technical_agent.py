import time
import json
import httpx
from pathlib import Path
from .base_agent import BaseAgent
from indicators import (
    calculate_indicators, calculate_trend_signal, calculate_adx,
    calculate_atr, calculate_macd, calculate_bollinger_bands,
    calculate_volume_sma, calculate_stochastic_rsi, calculate_ema,
)

TIMEFRAME_MAP = {
    "5m": "Min5", "15m": "Min15", "30m": "Min30",
    "1h": "Min60", "4h": "Hour4", "8h": "Hour8", "1D": "Day1",
}

MEXC_BASE = "https://api.mexc.com"
CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "cache"


class TechnicalAgent(BaseAgent):
    def __init__(self):
        super().__init__("Technical")
        self.timeframes = ["5m", "15m", "1h", "4h", "1D"]
        self.results: dict[str, dict] = {}
        self._client: httpx.AsyncClient | None = None
        self._kline_cache: dict[str, dict] = {}

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=30, limits=httpx.Limits(max_connections=50))
        return self._client

    async def analyze(self, data: dict) -> dict:
        try:
            self.update_status("running")
            symbols = data.get("symbols", [])
            client = await self._get_client()

            all_signals = []
            symbol_scores = {}

            for symbol in symbols:
                tf_signals = []
                for tf in self.timeframes:
                    try:
                        klines = await self._get_klines(client, symbol, tf)
                        if not klines or len(klines) < 50:
                            continue

                        signal = self._analyze_symbol_tf(klines, symbol, tf)
                        if signal:
                            tf_signals.append(signal)
                    except Exception:
                        continue

                if tf_signals:
                    combined = self._combine_timeframe_signals(tf_signals)
                    combined["symbol"] = symbol
                    all_signals.append(combined)
                    symbol_scores[symbol] = combined["confidence"]

            all_signals.sort(key=lambda x: x["confidence"], reverse=True)
            self.signals = all_signals[:50]

            for sig in all_signals:
                self.results[sig["symbol"]] = sig

            self.update_status("ready")
            return {
                "signals": self.signals,
                "symbol_scores": symbol_scores,
                "analyzed_count": len(symbols),
                "signal_count": len(all_signals),
            }

        except Exception as e:
            self.update_status("error", str(e))
            return {"signals": [], "error": str(e)}

    async def _get_klines(self, client: httpx.AsyncClient, symbol: str, tf: str) -> list[dict]:
        cache_key = f"{symbol}_{tf}"
        if cache_key in self._kline_cache:
            cached = self._kline_cache[cache_key]
            if time.time() - cached["ts"] < 300:
                return cached["data"]

        tf_value = TIMEFRAME_MAP.get(tf, "Day1")
        try:
            resp = await client.get(f"{MEXC_BASE}/api/v1/contract/kline/{symbol}", params={"interval": tf_value})
            resp.raise_for_status()
            data = resp.json()
            raw = data.get("data", {})

            klines = []
            if isinstance(raw, dict) and "time" in raw:
                times = raw.get("time", [])
                opens = raw.get("open", [])
                highs = raw.get("high", [])
                lows = raw.get("low", [])
                closes = raw.get("close", [])
                vols = raw.get("vol", [])
                for i in range(len(times)):
                    klines.append({
                        "time": times[i] * 1000 if times[i] < 1e12 else times[i],
                        "open": float(opens[i]) if i < len(opens) else 0,
                        "high": float(highs[i]) if i < len(highs) else 0,
                        "low": float(lows[i]) if i < len(lows) else 0,
                        "close": float(closes[i]) if i < len(closes) else 0,
                        "vol": float(vols[i]) if i < len(vols) else 0,
                    })
            klines.sort(key=lambda x: x["time"])

            if klines:
                self._kline_cache[cache_key] = {"data": klines, "ts": time.time()}
            return klines
        except Exception:
            return self._load_cached_klines(symbol, tf)

    def _load_cached_klines(self, symbol: str, tf: str) -> list[dict]:
        tf_value = TIMEFRAME_MAP.get(tf, "Day1")
        safe_sym = symbol.replace("/", "_")
        cache_file = CACHE_DIR / f"{safe_sym}_{tf_value}.json"
        if cache_file.exists():
            try:
                klines = json.loads(cache_file.read_text(encoding="utf-8"))
                if klines:
                    self._kline_cache[f"{symbol}_{tf}"] = {"data": klines, "ts": time.time()}
                return klines
            except Exception:
                pass
        return []

    def _analyze_symbol_tf(self, klines: list[dict], symbol: str, tf: str) -> dict | None:
        if len(klines) < 50:
            return None

        closes = [float(k["close"]) for k in klines]
        highs = [float(k.get("high", k["close"])) for k in klines]
        lows = [float(k.get("low", k["close"])) for k in klines]
        volumes = [float(k.get("vol", 0)) for k in klines]

        rsi, rsi_ma = calculate_indicators(closes, 14, 14)
        ema_fast, ema_slow, trend_signals = calculate_trend_signal(closes, 10, 30)
        adx = calculate_adx(highs, lows, closes, 14)
        atr = calculate_atr(highs, lows, closes, 14)
        macd_line, macd_signal, macd_hist = calculate_macd(closes)
        bb_upper, bb_mid, bb_lower = calculate_bollinger_bands(closes)
        vol_sma = calculate_volume_sma(volumes)
        stoch_k, stoch_d = calculate_stochastic_rsi(closes)

        idx = len(closes) - 1
        if idx < 0:
            return None

        current_price = closes[idx]
        current_rsi = rsi[idx] if idx < len(rsi) else None
        current_rsi_ma = rsi_ma[idx] if idx < len(rsi_ma) else None
        current_trend = trend_signals[idx] if idx < len(trend_signals) else "none"
        current_adx = adx[idx] if adx and idx < len(adx) else None
        current_atr = atr[idx] if atr and idx < len(atr) else None
        current_macd = macd_line[idx] if macd_line and idx < len(macd_line) else None
        current_macd_sig = macd_signal[idx] if macd_signal and idx < len(macd_signal) else None
        current_bb_lower = bb_lower[idx] if bb_lower and idx < len(bb_lower) else None
        current_bb_upper = bb_upper[idx] if bb_upper and idx < len(bb_upper) else None
        current_vol_sma = vol_sma[idx] if vol_sma and idx < len(vol_sma) else None
        current_stoch_k = stoch_k[idx] if stoch_k and idx < len(stoch_k) else None

        prev_idx = idx - 1
        prev_rsi_ma = rsi_ma[prev_idx] if prev_idx >= 0 and prev_idx < len(rsi_ma) else None
        prev_trend = trend_signals[prev_idx] if prev_idx >= 0 and prev_idx < len(trend_signals) else "none"

        signals = []
        confidence = 0.0
        direction = "hold"
        reasons = []

        if current_rsi is not None and current_rsi_ma is not None and prev_rsi_ma is not None:
            if prev_rsi_ma >= 30 and current_rsi_ma < 30:
                signals.append("rsi_oversold_entry")
                confidence += 0.35
                reasons.append("RSI MA 30 altina dustu")
            elif prev_rsi_ma <= 70 and current_rsi_ma > 70:
                signals.append("rsi_overbought_exit")
                confidence += 0.25
                reasons.append("RSI MA 70 ustune cikti")

        if current_trend == "buy" and prev_trend != "buy":
            signals.append("ema_bullish_cross")
            confidence += 0.30
            reasons.append("EMA bullish kesim")
        elif current_trend == "sell" and prev_trend != "sell":
            signals.append("ema_bearish_cross")
            confidence += 0.30
            reasons.append("EMA bearish kesim")

        if current_macd is not None and current_macd_sig is not None:
            prev_macd = macd_line[prev_idx] if prev_idx >= 0 and prev_idx < len(macd_line) else None
            prev_macd_sig = macd_signal[prev_idx] if prev_idx >= 0 and prev_idx < len(macd_signal) else None
            if prev_macd is not None and prev_macd_sig is not None:
                if prev_macd <= prev_macd_sig and current_macd > current_macd_sig:
                    signals.append("macd_bullish")
                    confidence += 0.20
                    reasons.append("MACD bullish kesim")
                elif prev_macd >= prev_macd_sig and current_macd < current_macd_sig:
                    signals.append("macd_bearish")
                    confidence += 0.20
                    reasons.append("MACD bearish kesim")

        if current_bb_lower is not None and current_price <= current_bb_lower:
            signals.append("bb_oversold")
            confidence += 0.15
            reasons.append("Fiyat Bollinger alt bandinda")
        if current_bb_upper is not None and current_price >= current_bb_upper:
            signals.append("bb_overbought")
            confidence += 0.15
            reasons.append("Fiyat Bollinger ust bandinda")

        if current_adx is not None and current_adx > 25:
            confidence += 0.15
            reasons.append(f"ADX {current_adx:.1f} guclu trend")

        if current_stoch_k is not None:
            if current_stoch_k < 20:
                signals.append("stoch_oversold")
                confidence += 0.15
                reasons.append("Stochastic RSI asiri satis")
            elif current_stoch_k > 80:
                signals.append("stoch_overbought")
                confidence += 0.15
                reasons.append("Stochastic RSI asiri alis")

        bullish_count = sum(1 for s in signals if "bullish" in s or "oversold" in s or "entry" in s)
        bearish_count = sum(1 for s in signals if "bearish" in s or "overbought" in s or "exit" in s)

        if bullish_count > bearish_count:
            direction = "long"
        elif bearish_count > bullish_count:
            direction = "short"
        else:
            direction = "hold"

        confidence = min(confidence, 1.0)

        atr_pct = 0.0
        if current_atr and current_price > 0:
            atr_pct = current_atr / current_price

        return {
            "symbol": symbol,
            "timeframe": tf,
            "price": current_price,
            "direction": direction,
            "confidence": round(confidence, 3),
            "signals": signals,
            "reasons": reasons,
            "indicators": {
                "rsi": round(current_rsi, 2) if current_rsi else None,
                "rsi_ma": round(current_rsi_ma, 2) if current_rsi_ma else None,
                "adx": round(current_adx, 2) if current_adx else None,
                "atr": current_atr,
                "atr_pct": round(atr_pct, 4) if atr_pct else None,
                "macd": current_macd,
                "macd_signal": current_macd_sig,
                "bb_lower": current_bb_lower,
                "bb_upper": current_bb_upper,
                "stoch_k": round(current_stoch_k, 2) if current_stoch_k else None,
                "trend": current_trend,
                "ema_fast": ema_fast[idx] if ema_fast and idx < len(ema_fast) else None,
                "ema_slow": ema_slow[idx] if ema_slow and idx < len(ema_slow) else None,
            },
            "volume": volumes[idx] if idx < len(volumes) else 0,
            "volume_sma": current_vol_sma,
            "high": highs[idx],
            "low": lows[idx],
        }

    def _combine_timeframe_signals(self, tf_signals: list[dict]) -> dict:
        if not tf_signals:
            return {}

        tf_weights = {"5m": 0.1, "15m": 0.15, "1h": 0.25, "4h": 0.3, "1D": 0.2}
        weighted_confidence = 0.0
        direction_votes = {"long": 0.0, "short": 0.0}
        all_signals = []
        all_reasons = []
        latest = tf_signals[-1]

        tf_details = []
        for sig in tf_signals:
            tf = sig["timeframe"]
            weight = tf_weights.get(tf, 0.1)
            weighted_confidence += sig["confidence"] * weight
            sig_dir = sig["direction"]
            if sig_dir in direction_votes:
                direction_votes[sig_dir] += weight
            all_signals.extend(sig["signals"])
            all_reasons.extend(sig["reasons"])

            tf_details.append({
                "timeframe": tf,
                "direction": sig["direction"],
                "confidence": round(sig["confidence"], 3),
                "weight": weight,
                "signals": sig["signals"],
                "reasons": sig["reasons"],
                "indicators": sig.get("indicators", {}),
            })

        long_votes = round(direction_votes.get("long", 0), 2)
        short_votes = round(direction_votes.get("short", 0), 2)

        if long_votes > short_votes and long_votes >= 0.1:
            combined_direction = "long"
        elif short_votes > long_votes and short_votes >= 0.1:
            combined_direction = "short"
        else:
            combined_direction = "hold"

        return {
            "symbol": latest["symbol"],
            "direction": combined_direction,
            "confidence": round(min(weighted_confidence, 1.0), 3),
            "price": latest["price"],
            "signals": list(set(all_signals)),
            "reasons": list(set(all_reasons)),
            "timeframes_analyzed": [s["timeframe"] for s in tf_signals],
            "timeframe_details": tf_details,
            "direction_votes": {"long": long_votes, "short": short_votes},
            "indicators": latest["indicators"],
            "atr": latest.get("atr"),
            "atr_pct": latest["indicators"].get("atr_pct"),
            "high": latest["high"],
            "low": latest["low"],
            "volume": latest["volume"],
        }
