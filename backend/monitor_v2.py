"""
MEXC ML Trading System — Monitor V2.0
=======================================
Gelismis Monitoring: Prometheus Metrics, Grafana Dashboards, Alert Rules

OZELLIKLER:
  1. Prometheus Metrics: Metrik export
  2. Alert Rules: Telegram/Email uyari sistemi
  3. Performance Dashboard: Gercek zamanli gorunum
  4. A/B Testing: Model karsilastirma
"""

import time
import json
import logging
from collections import deque
from typing import Optional, Dict, List
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# PROMETHEUS METRICS
# ══════════════════════════════════════════════════════════════════════════════

class PrometheusMetrics:
    """
    Prometheus formatinda metrik export.
    """

    def __init__(self):
        self._counters = {}
        self._gauges = {}
        self._histograms = {}

    def inc_counter(self, name: str, value: float = 1.0, labels: Dict = None):
        """Counter artir."""
        key = f"{name}_{json.dumps(labels or {}, sort_keys=True)}"
        self._counters[key] = self._counters.get(key, 0) + value

    def set_gauge(self, name: str, value: float, labels: Dict = None):
        """Gauge ayarla."""
        key = f"{name}_{json.dumps(labels or {}, sort_keys=True)}"
        self._gauges[key] = value

    def observe_histogram(self, name: str, value: float, labels: Dict = None):
        """Histogram gozlemle."""
        key = f"{name}_{json.dumps(labels or {}, sort_keys=True)}"
        if key not in self._histograms:
            self._histograms[key] = []
        self._histograms[key].append(value)
        if len(self._histograms[key]) > 1000:
            self._histograms[key] = self._histograms[key][-1000:]

    def export_text(self) -> str:
        """Prometheus text formatinda export."""
        lines = []

        # Counters
        for key, value in self._counters.items():
            name = key.split("_{")[0]
            lines.append(f"# TYPE {name} counter")
            lines.append(f"{key} {value}")

        # Gauges
        for key, value in self._gauges.items():
            name = key.split("_{")[0]
            lines.append(f"# TYPE {name} gauge")
            lines.append(f"{key} {value}")

        # Histograms
        for key, values in self._histograms.items():
            name = key.split("_{")[0]
            lines.append(f"# TYPE {name} histogram")
            if values:
                lines.append(f"{key}_count {len(values)}")
                lines.append(f"{key}_sum {sum(values)}")
                lines.append(f"{key}_avg {sum(values) / len(values)}")

        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# ALERT RULES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class AlertRule:
    name: str
    condition: str  # "gt", "lt", "eq"
    threshold: float
    metric_name: str
    message_template: str
    cooldown_seconds: int = 300
    enabled: bool = True


class AlertManager:
    """
    Uyari yonetimi:
    - Kural bazli uyari sistemi
    - Cooldown mekanizmasi
    - Telegram/Email entegrasyonu
    """

    def __init__(self):
        self._rules: List[AlertRule] = []
        self._last_alerts: Dict[str, float] = {}
        self._alert_history: deque = deque(maxlen=100)

    def add_rule(self, rule: AlertRule):
        """Kural ekle."""
        self._rules.append(rule)

    def check_alerts(self, metrics: Dict) -> List[Dict]:
        """
        Metriklere gore uyari kontrolu.

        Donus: Tetiklenen uyari listesi
        """
        triggered = []
        now = time.time()

        for rule in self._rules:
            if not rule.enabled:
                continue

            value = metrics.get(rule.metric_name)
            if value is None:
                continue

            # Cooldown kontrolu
            last_alert = self._last_alerts.get(rule.name, 0)
            if now - last_alert < rule.cooldown_seconds:
                continue

            # Kosul kontrolu
            triggered_alert = False
            if rule.condition == "gt" and value > rule.threshold:
                triggered_alert = True
            elif rule.condition == "lt" and value < rule.threshold:
                triggered_alert = True
            elif rule.condition == "eq" and value == rule.threshold:
                triggered_alert = True

            if triggered_alert:
                alert = {
                    "rule": rule.name,
                    "metric": rule.metric_name,
                    "value": value,
                    "threshold": rule.threshold,
                    "message": rule.message_template.format(value=value, threshold=rule.threshold),
                    "timestamp": now,
                }
                triggered.append(alert)
                self._last_alerts[rule.name] = now
                self._alert_history.append(alert)
                logger.warning(f"ALERT: {rule.name} - {alert['message']}")

        return triggered

    def get_alert_history(self) -> List[Dict]:
        """Alert gecmisini dondur."""
        return list(self._alert_history)

    def get_status(self) -> Dict:
        return {
            "total_rules": len(self._rules),
            "enabled_rules": sum(1 for r in self._rules if r.enabled),
            "recent_alerts": len(self._alert_history),
        }


# ══════════════════════════════════════════════════════════════════════════════
# PERFORMANCE TRACKER V2
# ══════════════════════════════════════════════════════════════════════════════

class PerformanceTrackerV2:
    """
    Gelismis performans takibi:
    - Islem bazli metrikler
    - Zaman bazli metrikler
    - Model karsilastirma (A/B testing)
    """

    def __init__(self, max_trades: int = 1000):
        self._trades: deque = deque(maxlen=max_trades)
        self._hourly_stats: Dict = {}
        self._daily_stats: Dict = {}
        self._model_stats: Dict = {}

    def record_trade(self, trade: Dict):
        """Trade kaydet."""
        self._trades.append(trade)

        # Model istatistikleri
        model = trade.get("model", "unknown")
        if model not in self._model_stats:
            self._model_stats[model] = {"trades": 0, "wins": 0, "pnl": 0.0}
        self._model_stats[model]["trades"] += 1
        if trade.get("pnl", 0) > 0:
            self._model_stats[model]["wins"] += 1
        self._model_stats[model]["pnl"] += trade.get("pnl", 0)

    def get_win_rate(self, window: int = 100) -> float:
        """Win rate hesapla."""
        if not self._trades:
            return 0.0
        recent = list(self._trades)[-window:]
        wins = sum(1 for t in recent if t.get("pnl", 0) > 0)
        return wins / len(recent) if recent else 0.0

    def get_avg_pnl(self, window: int = 100) -> float:
        """Ortalama PnL hesapla."""
        if not self._trades:
            return 0.0
        recent = list(self._trades)[-window:]
        pnls = [t.get("pnl", 0) for t in recent]
        return sum(pnls) / len(pnls) if pnls else 0.0

    def get_sharpe_ratio(self, window: int = 100) -> float:
        """Sharpe ratio hesapla."""
        if not self._trades:
            return 0.0
        recent = list(self._trades)[-window:]
        returns = [t.get("pnl", 0) for t in recent]
        if len(returns) < 3:
            return 0.0
        import numpy as np
        arr = np.array(returns)
        if arr.std() < 1e-10:
            return 0.0
        return float(arr.mean() / arr.std() * np.sqrt(252))

    def get_max_drawdown(self) -> float:
        """Maksimum drawdown hesapla."""
        if not self._trades:
            return 0.0
        equity = [1.0]
        for t in self._trades:
            pnl = t.get("pnl", 0)
            equity.append(equity[-1] * (1 + pnl))

        equity_arr = np.array(equity) if 'np' in dir() else equity
        peaks = equity_arr
        dd = [(peaks[i] - equity_arr[i]) / peaks[i] for i in range(len(equity_arr))]
        return max(dd) if dd else 0.0

    def get_model_comparison(self) -> Dict:
        """Model karsilastirmasi (A/B testing)."""
        comparison = {}
        for model, stats in self._model_stats.items():
            trades = stats["trades"]
            wins = stats["wins"]
            pnl = stats["pnl"]
            win_rate = wins / trades if trades > 0 else 0
            avg_pnl = pnl / trades if trades > 0 else 0

            comparison[model] = {
                "trades": trades,
                "win_rate": round(win_rate * 100, 1),
                "total_pnl": round(pnl, 2),
                "avg_pnl": round(avg_pnl, 4),
            }

        return comparison

    def get_comprehensive_stats(self) -> Dict:
        """Kapsamli istatistikler."""
        return {
            "total_trades": len(self._trades),
            "win_rate": round(self.get_win_rate() * 100, 1),
            "avg_pnl": round(self.get_avg_pnl(), 4),
            "sharpe_ratio": round(self.get_sharpe_ratio(), 3),
            "max_drawdown": round(self.get_max_drawdown() * 100, 2),
            "model_comparison": self.get_model_comparison(),
        }


# ══════════════════════════════════════════════════════════════════════════════
# MONITOR MANAGER V2
# ══════════════════════════════════════════════════════════════════════════════

class MonitorManagerV2:
    """
    Monitoring V2 - Ana sinif:
    - Prometheus export
    - Alert yonetimi
    - Performans takibi
    """

    def __init__(self):
        self.prometheus = PrometheusMetrics()
        self.alert_manager = AlertManager()
        self.performance = PerformanceTrackerV2()

        self._start_time = time.time()
        self._setup_default_alerts()

    def _setup_default_alerts(self):
        """Varsayilan alert kurallari."""
        self.alert_manager.add_rule(AlertRule(
            name="low_win_rate",
            condition="lt",
            threshold=0.4,
            metric_name="win_rate",
            message_template="Win rate dusuk: %{value:.1f} (esik: %{threshold:.1f})",
            cooldown_seconds=3600,
        ))

        self.alert_manager.add_rule(AlertRule(
            name="high_drawdown",
            condition="gt",
            threshold=0.10,
            metric_name="drawdown",
            message_template="Drawdown yuksek: %{value:.1%} (esik: %{threshold:.1%})",
            cooldown_seconds=1800,
        ))

        self.alert_manager.add_rule(AlertRule(
            name="low_sharpe",
            condition="lt",
            threshold=0.5,
            metric_name="sharpe",
            message_template="Sharpe dusuk: %{value:.3f} (esik: %{threshold:.3f})",
            cooldown_seconds=7200,
        ))

    def update_metrics(self, metrics: Dict):
        """Metrikleri guncelle."""
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.prometheus.set_gauge(f"mexc_{key}", value)

        # Alert kontrolu
        self.alert_manager.check_alerts(metrics)

    def record_trade(self, trade: Dict):
        """Trade kaydet."""
        self.performance.record_trade(trade)
        self.prometheus.inc_counter("mexc_trades_total")

        pnl = trade.get("pnl", 0)
        if pnl > 0:
            self.prometheus.inc_counter("mexc_trades_wins")
        else:
            self.prometheus.inc_counter("mexc_trades_losses")

    def get_uptime(self) -> float:
        """Uptime suresi."""
        return time.time() - self._start_time

    def get_comprehensive_status(self) -> Dict:
        """Kapsamli durum raporu."""
        return {
            "uptime_hours": round(self.get_uptime() / 3600, 1),
            "performance": self.performance.get_comprehensive_stats(),
            "alerts": self.alert_manager.get_status(),
            "prometheus": {
                "counters": len(self.prometheus._counters),
                "gauges": len(self.prometheus._gauges),
                "histograms": len(self.prometheus._histograms),
            },
        }


# ══════════════════════════════════════════════════════════════════════════════
# EXPORT
# ══════════════════════════════════════════════════════════════════════════════

__all__ = [
    "MonitorManagerV2",
    "PrometheusMetrics",
    "AlertManager",
    "PerformanceTrackerV2",
    "AlertRule",
]
