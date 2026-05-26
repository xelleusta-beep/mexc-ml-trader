"""
MEXC ML Trading System — Monitor v1.0
Sistem metrikleri, latency takibi, performans izleme.
"""

import time
import json
import logging
from collections import deque
from typing import Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class LatencyRecord:
    operation: str
    duration_ms: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class PredictionRecord:
    symbol: str
    signal: str
    confidence: float
    latency_ms: float
    source: str
    timestamp: float = field(default_factory=time.time)


class MetricsTracker:
    """Sistem metriklerini toplar ve analiz eder."""

    def __init__(self, max_records: int = 1000):
        self._max = max_records

        # Latency kayitlari
        self._latencies: deque = deque(maxlen=max_records)

        # Prediction kayitlari
        self._predictions: deque = deque(maxlen=max_records)

        # Dakikadaki prediction sayisi (throughput)
        self._prediction_timestamps: deque = deque(maxlen=300)

        # Hata sayaci
        self._error_count: int = 0
        self._last_error_time: float = 0.0

        # ML model metrikleri (train_log'dan)
        self._train_metrics: list = []

        # Uptime
        self._start_time: float = time.time()

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self._start_time

    # ── LATENCY ──────────────────────────────────────────────────────────

    def record_latency(self, operation: str, duration_ms: float):
        self._latencies.append(LatencyRecord(operation, duration_ms))

    def get_avg_latency(self, operation: str = "", window: int = 100) -> float:
        recs = list(self._latencies)[-window:]
        if operation:
            recs = [r for r in recs if r.operation == operation]
        if not recs:
            return 0.0
        return float(sum(r.duration_ms for r in recs)) / len(recs)

    def get_latency_stats(self) -> dict:
        """Latency istatistikleri: min, max, avg, p50, p95, p99."""
        if not self._latencies:
            return {"avg": 0, "min": 0, "max": 0, "p50": 0, "p95": 0, "p99": 0}
        vals = sorted(r.duration_ms for r in self._latencies)
        n = len(vals)
        return {
            "avg": round(float(sum(vals)) / n, 2),
            "min": round(vals[0], 2),
            "max": round(vals[-1], 2),
            "p50": round(vals[int(n * 0.50)], 2),
            "p95": round(vals[int(n * 0.95)], 2),
            "p99": round(vals[int(n * 0.99)], 2),
            "count": n,
        }

    # ── PREDICTIONS ──────────────────────────────────────────────────────

    def record_prediction(self, symbol: str, signal: str, confidence: float,
                          latency_ms: float, source: str = ""):
        self._predictions.append(
            PredictionRecord(symbol, signal, confidence, latency_ms, source)
        )
        self._prediction_timestamps.append(time.time())

    def get_prediction_stats(self) -> dict:
        """Prediction istatistikleri."""
        if not self._predictions:
            return {"total": 0}
        window = list(self._predictions)[-200:]

        # Sinyal dagilimi
        sig_dist = {}
        for p in window:
            sig_dist[p.signal] = sig_dist.get(p.signal, 0) + 1

        # Kaynak dagilimi
        src_dist = {}
        for p in window:
            s = p.source or "unknown"
            src_dist[s] = src_dist.get(s, 0) + 1

        # Ortalama confidence
        avg_conf = float(sum(p.confidence for p in window)) / len(window)

        # Throughput (son 1 dk)
        now = time.time()
        recent = [t for t in self._prediction_timestamps if now - t < 60]
        throughput = len(recent)

        total = len(self._predictions)

        return {
            "total":           total,
            "throughput_1m":   throughput,
            "avg_confidence":  round(avg_conf, 1),
            "signal_dist":     {k: round(v/len(window)*100, 1) for k, v in sig_dist.items()},
            "source_dist":     {k: round(v/len(window)*100, 1) for k, v in src_dist.items()},
            "last_signal":     window[-1].signal if window else "",
        }

    # ── ERRORS ───────────────────────────────────────────────────────────

    def record_error(self):
        self._error_count += 1
        self._last_error_time = time.time()

    def get_error_rate(self) -> float:
        uptime = self.uptime_seconds
        if uptime < 1:
            return 0.0
        return self._error_count / uptime * 3600  # saatlik hata

    # ── TRAIN METRICS ────────────────────────────────────────────────────

    def record_train(self, metrics: dict):
        """ML egitim metriklerini kaydet."""
        metrics["_timestamp"] = time.time()
        self._train_metrics.append(metrics)
        if len(self._train_metrics) > 50:
            self._train_metrics = self._train_metrics[-50:]

    def get_train_history(self) -> list:
        """Egitim gecmisi (frontend icin)."""
        return [
            {
                "time":     m.get("trained_at", "")[:16] if m.get("trained_at") else "",
                "symbol":   m.get("symbol", "GLOBAL"),
                "samples":  m.get("n_samples", 0),
                "wf_acc":   m.get("wf_accuracy", 0),
                "sharpe":   m.get("backtest_sharpe", 0),
                "roi":      m.get("backtest_roi", 0),
                "duration": round(m.get("train_time_s", 0), 1),
            }
            for m in self._train_metrics[-20:]
        ][::-1]

    # ── FULL STATUS ──────────────────────────────────────────────────────

    def get_status(self) -> dict:
        return {
            "uptime_sec":     round(self.uptime_seconds),
            "uptime_hours":   round(self.uptime_seconds / 3600, 1),
            "latency":        self.get_latency_stats(),
            "predictions":    self.get_prediction_stats(),
            "errors_per_hour": round(self.get_error_rate(), 2),
            "error_count":    self._error_count,
            "train_count":    len(self._train_metrics),
        }


# Context manager for timing
class Timer:
    """with Timer('operation') as t:  seklinde kullanim."""

    def __init__(self, tracker: MetricsTracker, operation: str = ""):
        self.tracker = tracker
        self.operation = operation
        self.start: float = 0.0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        if self.start:
            ms = (time.perf_counter() - self.start) * 1000
            if self.tracker:
                self.tracker.record_latency(self.operation, ms)
