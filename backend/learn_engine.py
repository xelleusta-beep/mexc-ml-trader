"""
Kendi kendini geliştiren bot.
Trade sonuçlarını analiz eder, parametreleri optimize eder.
"""

import json
import time
import logging
from collections import defaultdict
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class LearnEngine:
    def __init__(self, persist_dir):
        self.persist_dir = persist_dir
        self.path = f"{persist_dir}/learned_params.json"

        # Öğrenilen parametreler
        self.params = {
            "min_signal_score": 2.0,
            "sl_pct": 0.02,
            "tp_pct": 0.04,
            "whale_threshold_eth": 500,
            "whale_weights": {},      # address -> weight multiplier
            "keyword_weights": {},    # keyword -> weight multiplier
            "symbol_biases": {},      # symbol -> bias (-1..1)
        }

        # İstatistik
        self.stats = {
            "total_analyzed": 0,
            "last_adjustment": 0,
            "adjustments_made": 0,
            "performance_by_source": defaultdict(lambda: {"trades": 0, "wins": 0, "total_pnl": 0}),
            "performance_by_symbol": defaultdict(lambda: {"trades": 0, "wins": 0, "total_pnl": 0}),
        }

        self._load()
        logger.info("LearnEngine baslatildi")

    def _load(self):
        try:
            import os
            if os.path.exists(self.path):
                with open(self.path) as f:
                    data = json.load(f)
                    self.params.update(data.get("params", {}))
                    self.stats.update(data.get("stats", {}))
                    logger.info(f"Ogrenilen parametreler yuklendi ({self.stats['total_analyzed']} trade)")
        except Exception as e:
            logger.debug(f"Learn load error: {e}")

    def _save(self):
        try:
            import os
            os.makedirs(self.persist_dir, exist_ok=True)
            with open(self.path, "w") as f:
                json.dump({
                    "params": self.params,
                    "stats": {k: v for k, v in self.stats.items() if k != "performance_by_source" and k != "performance_by_symbol"},
                    "saved_at": datetime.now(timezone.utc).isoformat(),
                }, f, default=str, indent=2)
        except Exception as e:
            logger.debug(f"Learn save error: {e}")

    def record_trade(self, trade):
        """Her trade sonucu kaydet."""
        symbol = trade.get("symbol", "UNKNOWN")
        pnl = trade.get("pnl", 0)
        won = pnl > 0

        sembol = self.stats["performance_by_symbol"][symbol]
        sembol["trades"] += 1
        sembol["wins"] += 1 if won else 0
        sembol["total_pnl"] += pnl

        self.stats["total_analyzed"] += 1

        # Kaynak bazlı takip (whale address vs news keyword)
        source = trade.get("source", "unknown")
        if source and source != "unknown":
            src = self.stats["performance_by_source"][source]
            src["trades"] += 1
            src["wins"] += 1 if won else 0
            src["total_pnl"] += pnl

        # Her 10 trade'de bir optimize et
        if self.stats["total_analyzed"] % 10 == 0:
            self._optimize()

        self._save()

    def _optimize(self):
        """Parametreleri trade sonuçlarına göre optimize et."""
        logger.info("Parametre optimizasyonu basliyor...")
        changed = False

        # Sembol bazlı win rate kontrolü
        for sym, perf in self.stats["performance_by_symbol"].items():
            if perf["trades"] >= 5:
                wr = perf["wins"] / perf["trades"]
                current_bias = self.params["symbol_biases"].get(sym, 0)

                if wr < 0.3 and current_bias < 1:
                    # Kötü performans: bu sembolden uzak dur
                    self.params["symbol_biases"][sym] = max(-1, current_bias - 0.3)
                    changed = True
                    logger.info(f"  {sym}: WR={wr:.0%} -> bias {current_bias:.1f} -> {self.params['symbol_biases'][sym]:.1f}")
                elif wr > 0.7 and current_bias > -1:
                    # İyi performans: bu sembole daha çok güven
                    self.params["symbol_biases"][sym] = min(1, current_bias + 0.2)
                    changed = True
                    logger.info(f"  {sym}: WR={wr:.0%} -> bias {current_bias:.1f} -> {self.params['symbol_biases'][sym]:.1f}")

        # Genel win rate'e göre MIN_SIGNAL_SCORE ayarla
        total_trades = sum(p["trades"] for p in self.stats["performance_by_symbol"].values())
        total_wins = sum(p["wins"] for p in self.stats["performance_by_symbol"].values())

        if total_trades >= 10:
            overall_wr = total_wins / total_trades
            current_threshold = self.params["min_signal_score"]

            if overall_wr < 0.35:
                # Çok kaybediyoruz: daha seçici ol
                new_threshold = min(5.0, current_threshold + 0.5)
                self.params["min_signal_score"] = new_threshold
                changed = True
                logger.info(f"  WR={overall_wr:.0%} -> MIN_SIGNAL_SCORE {current_threshold} -> {new_threshold}")
            elif overall_wr > 0.65 and current_threshold > 1.0:
                # İyi gidiyoruz: daha agresif olabiliriz
                new_threshold = max(1.0, current_threshold - 0.3)
                self.params["min_signal_score"] = new_threshold
                changed = True
                logger.info(f"  WR={overall_wr:.0%} -> MIN_SIGNAL_SCORE {current_threshold} -> {new_threshold}")
            elif total_trades > 20 and total_wins == 0:
                # Hiç kazanamıyorsak: bekle ve gör
                self.params["min_signal_score"] = min(8.0, current_threshold + 1.0)
                changed = True
                logger.info(f"  Hic kazanamadi -> MIN_SIGNAL_SCORE {current_threshold} -> {self.params['min_signal_score']}")

        # Hiç trade yoksa 24 saat sonra threshold'u düşür
        if self.stats["total_analyzed"] == 0 and self.params["min_signal_score"] > 1.0:
            self.params["min_signal_score"] = 1.5
            changed = True

        if changed:
            self.stats["adjustments_made"] += 1
            logger.info(f"  -> {self.stats['adjustments_made']}. ayarlama yapildi")
            self._save()
        else:
            logger.info("  Degisiklik yok")

    def get_symbol_bias(self, symbol):
        """Sembol için bias skoru (-1..1). Pozitif = tercih et, negatif = uzak dur."""
        return self.params["symbol_biases"].get(symbol, 0)

    def get_score_threshold(self):
        return self.params["min_signal_score"]

    def get_whale_weight(self, address):
        return self.params["whale_weights"].get(address, 1.0)

    def get_summary(self):
        """Dashboard için özet."""
        by_symbol = []
        for sym, perf in sorted(self.stats["performance_by_symbol"].items(), key=lambda x: x[1]["trades"], reverse=True)[:10]:
            wr = round(perf["wins"] / perf["trades"] * 100, 1) if perf["trades"] > 0 else 0
            by_symbol.append({
                "symbol": sym,
                "trades": perf["trades"],
                "wins": perf["wins"],
                "win_rate": wr,
                "pnl": round(perf["total_pnl"], 2),
                "bias": self.params["symbol_biases"].get(sym, 0),
            })

        total_trades = sum(p["trades"] for p in self.stats["performance_by_symbol"].values())
        total_wins = sum(p["wins"] for p in self.stats["performance_by_symbol"].values())
        overall_wr = round(total_wins / total_trades * 100, 1) if total_trades > 0 else 0

        return {
            "params": {
                "min_signal_score": self.params["min_signal_score"],
                "whale_threshold_eth": self.params["whale_threshold_eth"],
            },
            "stats": {
                "total_analyzed": self.stats["total_analyzed"],
                "adjustments_made": self.stats["adjustments_made"],
                "overall_win_rate": overall_wr,
                "total_trades": total_trades,
            },
            "by_symbol": by_symbol,
        }
