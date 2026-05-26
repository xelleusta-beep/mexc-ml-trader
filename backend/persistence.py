"""
Dosya bazli veri kaliciligi — trade gecmisi, portfoy, aktif pozisyonlar.
JSON formatinda data/ klasorune kaydedilir, startup'ta geri yuklenir.
"""

import os, json, time, logging
from typing import Any
from collections import deque

logger = logging.getLogger(__name__)

MAX_TRADE_HISTORY = 500


class StateStore:
    """Disk tabanli state saklayici — her degisiklikte yazar."""

    def __init__(self, persist_dir: str):
        self._dir = persist_dir
        os.makedirs(self._dir, exist_ok=True)
        self._files = {
            "portfolio":     os.path.join(self._dir, "portfolio.json"),
            "trade_history": os.path.join(self._dir, "trade_history.json"),
            "active_trades": os.path.join(self._dir, "active_trades.json"),
            "pnl_timeline":  os.path.join(self._dir, "pnl_timeline.json"),
        }

    # ── Load ───────────────────────────────────────────────────────────────

    def load_portfolio(self) -> dict:
        return self._load("portfolio", {
            "capital": 100000.0,
            "initial_capital": 100000.0,
            "total_closed_trades": 0,
            "total_closed_notional": 0.0,
            "peak_capital": 100000.0,
        })

    def load_trade_history(self) -> deque:
        data = self._load("trade_history", [])
        return deque(data[:MAX_TRADE_HISTORY], maxlen=MAX_TRADE_HISTORY)

    def load_active_trades(self) -> dict:
        return self._load("active_trades", {})

    def load_pnl_timeline(self) -> list:
        return self._load("pnl_timeline", [])

    # ── Save ───────────────────────────────────────────────────────────────

    def save_portfolio(self, data: dict):
        self._save("portfolio", data)

    def save_trade_history(self, data: deque):
        self._save("trade_history", list(data)[-MAX_TRADE_HISTORY:])

    def save_active_trades(self, data: dict):
        self._save("active_trades", data)

    def save_pnl_timeline(self, data: list):
        self._save("pnl_timeline", data[-500:])

    # ── Internals ──────────────────────────────────────────────────────────

    def _load(self, key: str, default: Any) -> Any:
        path = self._files.get(key)
        if not path or not os.path.exists(path):
            return default
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("StateLoad: %s bozuk, varsayilan kullaniliyor (%s)", key, e)
            return default

    def _save(self, key: str, data: Any):
        path = self._files.get(key)
        if not path:
            return
        try:
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, default=str, ensure_ascii=False)
            os.replace(tmp, path)
        except OSError as e:
            logger.warning("StateSave: %s yazilamadi (%s)", key, e)

    @property
    def stats(self) -> dict:
        sizes = {}
        for name, path in self._files.items():
            if os.path.exists(path):
                try:
                    sizes[name] = os.path.getsize(path)
                except OSError:
                    sizes[name] = 0
        return {
            "dir": self._dir,
            "file_sizes": sizes,
            "total_bytes": sum(sizes.values()),
        }
