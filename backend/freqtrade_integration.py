"""
Freqtrade Integration
- Freqtrade webhook sinyallerini alır
- Sinyal analizleri üretir
- Stoploss ve takeprofit sinyalleri gönderir
"""

import json
import time
from pathlib import Path
from datetime import datetime
from typing import Optional
from pydantic import BaseModel

DATA_DIR = Path(__file__).parent.parent / "data" / "freqtrade"
DATA_DIR.mkdir(parents=True, exist_ok=True)


class FreqtradeSignal(BaseModel):
    exchange: str = "mexc"
    pair: str
    stake_amount: Optional[float] = None
    leverage: Optional[int] = None
    direction: Optional[str] = None
    enter_tag: Optional[str] = None
    order_type: str = "limit"
    price: Optional[float] = None
    stoploss: Optional[float] = None
    takeprofit: Optional[float] = None
    timestamp: Optional[int] = None


class FreqtradeAnalyzer:
    def __init__(self):
        self.signals: list[dict] = []
        self.active_pairs: dict[str, dict] = {}
        self._load_signals()

    def _signals_path(self) -> Path:
        return DATA_DIR / "signals.json"

    def _load_signals(self):
        p = self._signals_path()
        if p.exists():
            try:
                self.signals = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                self.signals = []

    def _save_signals(self):
        self._signals_path().write_text(
            json.dumps(self.signals[-200:], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def process_webhook(self, data: dict) -> dict:
        signal_type = data.get("type", "unknown")
        pair = data.get("pair", "")

        record = {
            "type": signal_type,
            "pair": pair,
            "data": data,
            "received_at": datetime.now().isoformat(),
            "timestamp": time.time(),
        }
        self.signals.append(record)
        self._save_signals()

        if signal_type == "enter":
            return self._handle_enter(data)
        elif signal_type == "exit":
            return self._handle_exit(data)
        elif signal_type == "stoploss":
            return self._handle_stoploss(data)
        elif signal_type == "custom_stoploss":
            return self._handle_custom_stoploss(data)
        elif signal_type == "status":
            return self._handle_status(data)
        else:
            return {"status": "received", "type": signal_type}

    def _handle_enter(self, data: dict) -> dict:
        pair = data.get("pair", "")
        direction = data.get("direction", "long")
        price = data.get("price", 0)
        stake = data.get("stake_amount", 0)
        leverage = data.get("leverage", 1)

        self.active_pairs[pair] = {
            "direction": direction,
            "entry_price": price,
            "stake_amount": stake,
            "leverage": leverage,
            "entry_time": time.time(),
            "stoploss": data.get("stoploss"),
            "takeprofit": data.get("takeprofit"),
            "enter_tag": data.get("enter_tag"),
        }

        return {
            "status": "accepted",
            "pair": pair,
            "direction": direction,
            "message": f"Freqtrade giris sinyali alindi: {pair} {direction}",
        }

    def _handle_exit(self, data: dict) -> dict:
        pair = data.get("pair", "")
        if pair in self.active_pairs:
            del self.active_pairs[pair]

        return {
            "status": "accepted",
            "pair": pair,
            "message": f"Freqtrade cikis sinyali alindi: {pair}",
        }

    def _handle_stoploss(self, data: dict) -> dict:
        pair = data.get("pair", "")
        return {
            "status": "accepted",
            "pair": pair,
            "message": f"Freqtrade stoploss sinyali: {pair}",
        }

    def _handle_custom_stoploss(self, data: dict) -> dict:
        pair = data.get("pair", "")
        current_price = data.get("current_price", 0)
        entry_price = data.get("entry_price", 0)

        if entry_price > 0 and current_price > 0:
            pnl_pct = (current_price - entry_price) / entry_price
            if pnl_pct > 0.03:
                return {"status": "accepted", "stoploss": entry_price * 1.01}
            elif pnl_pct > 0.02:
                return {"status": "accepted", "stoploss": entry_price * 1.005}

        return {"status": "accepted", "stoploss": None}

    def _handle_status(self, data: dict) -> dict:
        return {
            "status": "ok",
            "active_pairs": len(self.active_pairs),
            "total_signals": len(self.signals),
        }

    def get_analysis(self, pair: str) -> dict:
        recent_signals = [
            s for s in self.signals
            if s.get("pair") == pair and time.time() - s.get("timestamp", 0) < 86400
        ]

        enter_count = sum(1 for s in recent_signals if s["type"] == "enter")
        exit_count = sum(1 for s in recent_signals if s["type"] == "exit")
        stoploss_count = sum(1 for s in recent_signals if s["type"] == "stoploss")

        avg_stake = 0
        stake_values = [
            s["data"].get("stake_amount", 0)
            for s in recent_signals
            if s["type"] == "enter" and s["data"].get("stake_amount")
        ]
        if stake_values:
            avg_stake = sum(stake_values) / len(stake_values)

        return {
            "pair": pair,
            "signals_24h": len(recent_signals),
            "enter_signals": enter_count,
            "exit_signals": exit_count,
            "stoploss_count": stoploss_count,
            "avg_stake_amount": round(avg_stake, 2),
            "active": pair in self.active_pairs,
            "active_info": self.active_pairs.get(pair),
        }

    def get_all_signals(self, limit: int = 50) -> list[dict]:
        return self.signals[-limit:]

    def get_active_pairs(self) -> dict:
        return self.active_pairs
