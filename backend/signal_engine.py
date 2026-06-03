"""
Sinyal motoru.
Balina hareketleri + haber sentiment → birleşik sinyal.
Her sembol için bir skor hesaplar, eşik aşınca işlem sinyali üretir.
"""

import time
import logging
from collections import defaultdict

from config import config

logger = logging.getLogger(__name__)


class SignalEngine:
    def __init__(self):
        # symbol -> list of events
        self._events = defaultdict(list)
        self._scores = {}
        self._last_signals = {}

    def add_wallet_signal(self, signals):
        """Cüzdan takibinden gelen sinyalleri işle."""
        for sig in signals:
            direction = sig.get("direction", "")
            value = sig.get("diff_eth", 0) or sig.get("value_eth", 0)

            # Her sembolü etkile (genel balina hareketi)
            for sym in config.TRACKED_SYMBOLS:
                score = 0
                if direction == "accumulate":
                    score = min(1.5, value / 500)
                elif direction == "distribute":
                    score = -min(1.5, value / 500)
                else:
                    score = 0.5 if sig.get("type") == "tx_whale_move" else 0

                if score != 0:
                    self._events[sym].append({
                        "source": f"wallet_{sig.get('label', 'unknown')}",
                        "score": score,
                        "ts": time.time(),
                    })

    def add_news_signal(self, symbol, sentiment_score):
        """Haber sentimentini sinyale ekle."""
        if sentiment_score != 0:
            self._events[symbol].append({
                "source": "news",
                "score": sentiment_score * 1.5,
                "ts": time.time(),
            })

    def calculate_scores(self):
        """Tüm semboller için birleşik skor hesapla (zaman ağırlıklı)."""
        now = time.time()
        scores = {}

        for sym, events in self._events.items():
            total = 0
            total_weight = 0

            for ev in events:
                age = now - ev["ts"]
                if age > 3600 * 8:  # 8 saatten eski sinyalleri at
                    continue

                # Zaman ağırlığı: yeni sinyaller daha önemli
                weight = max(0.1, 1 - age / (3600 * 8))
                total += ev["score"] * weight
                total_weight += weight

            if total_weight > 0:
                avg = total / total_weight
                scores[sym] = round(avg, 3)
            else:
                scores[sym] = 0

        self._scores = scores

        # Eski eventleri temizle
        for sym in list(self._events.keys()):
            self._events[sym] = [e for e in self._events[sym] if now - e["ts"] < 3600 * 8]
            if not self._events[sym]:
                del self._events[sym]

        return scores

    def get_trade_signal(self, symbol):
        """Bir sembol için işlem sinyali üret."""
        score = self._scores.get(symbol, 0)

        if score >= config.MIN_SIGNAL_SCORE:
            return {"symbol": symbol, "signal": "LONG", "score": score, "confidence": min(95, 50 + score * 10)}
        elif score <= -config.MIN_SIGNAL_SCORE:
            return {"symbol": symbol, "signal": "SHORT", "score": score, "confidence": min(95, 50 + abs(score) * 10)}
        else:
            return {"symbol": symbol, "signal": "WAIT", "score": score, "confidence": 0}

    def get_all_signals(self):
        """Tüm semboller için sinyal durumu."""
        self.calculate_scores()
        signals = []
        for sym in config.TRACKED_SYMBOLS:
            sig = self.get_trade_signal(sym)
            if sig["signal"] != "WAIT":
                signals.append(sig)
        return signals

    def get_summary(self):
        """Dashboard için özet."""
        self.calculate_scores()
        result = []
        for sym in config.TRACKED_SYMBOLS:
            score = self._scores.get(sym, 0)
            result.append({
                "symbol": sym,
                "score": score,
                "signal": "LONG" if score >= config.MIN_SIGNAL_SCORE else ("SHORT" if score <= -config.MIN_SIGNAL_SCORE else "WAIT"),
                "events": len(self._events.get(sym, [])),
            })
        return sorted(result, key=lambda x: abs(x["score"]), reverse=True)


signal_engine = SignalEngine()
