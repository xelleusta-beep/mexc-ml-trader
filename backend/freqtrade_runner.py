"""
Freqtrade-Style Strategy Runner
- Render üzerinde çalışır
- Stratejileri yükler ve çalıştırır
- Sinyal üretir, otomatik işlem yapar
"""

import json
import time
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from indicators import (
    calculate_rsi,
    calculate_macd,
    calculate_bollinger_bands,
    calculate_ema,
    calculate_atr,
    calculate_adx,
    calculate_stochastic_rsi,
)
from mexc_client import get_client, BASE_URL, get_all_futures_symbols

DATA_DIR = Path(__file__).parent.parent / "data" / "freqtrade"
DATA_DIR.mkdir(parents=True, exist_ok=True)
STRATEGIES_DIR = DATA_DIR / "strategies"
STRATEGIES_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class Signal:
    pair: str
    direction: str
    enter_tag: str
    price: float
    stake_amount: float = 10.0
    leverage: int = 5
    stoploss: float = 0.0
    takeprofit: float = 0.0
    confidence: float = 0.0
    timestamp: float = field(default_factory=time.time)


class BaseStrategy:
    def __init__(self):
        self.name = "BaseStrategy"
        self.timeframe = "1h"
        self.can_short = True
        self.stoploss = -0.05
        self.trailing_stop = False
        self.trailing_stop_positive = 0.01
        self.trailing_stop_positive_offset = 0.02
        self.process_only_new_candles = True
        self.use_exit_signal = True
        self.exit_profit_only = False
        self.ignore_roi_if_entry_signal = False
        self.minimal_roi = {"0": 0.10, "30": 0.05, "60": 0.02, "120": 0}
        self.order_types = {
            "entry": "limit",
            "exit": "limit",
            "stoploss": "market",
            "stoploss_on_exchange": False,
        }

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        raise NotImplementedError

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        raise NotImplementedError

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        raise NotImplementedError


class MomentumStrategy(BaseStrategy):
    def __init__(self):
        super().__init__()
        self.name = "MomentumStrategy"
        self.timeframe = "1h"
        self.stoploss = -0.04
        self.minimal_roi = {"0": 0.08, "30": 0.04, "60": 0.02, "120": 0}

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        close = dataframe["close"].tolist()
        high = dataframe["high"].tolist()
        low = dataframe["low"].tolist()

        rsi = calculate_rsi(close, 14)
        macd_line, signal_line, histogram = calculate_macd(close)
        bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(close, 20, 2.0)
        ema_9 = calculate_ema(close, 9)
        ema_21 = calculate_ema(close, 21)
        adx = calculate_adx(high, low, close, 14)
        stoch_k, stoch_d = calculate_stochastic_rsi(close, 14, 14, 3)

        dataframe["rsi"] = rsi
        dataframe["macd"] = macd_line
        dataframe["macd_signal"] = signal_line
        dataframe["macd_hist"] = histogram
        dataframe["bb_upper"] = bb_upper
        dataframe["bb_middle"] = bb_middle
        dataframe["bb_lower"] = bb_lower
        dataframe["ema_9"] = ema_9
        dataframe["ema_21"] = ema_21
        dataframe["adx"] = adx
        dataframe["stoch_k"] = stoch_k
        dataframe["stoch_d"] = stoch_d

        dataframe["ema_cross"] = (dataframe["ema_9"] > dataframe["ema_21"]).astype(int)
        dataframe["macd_cross"] = (dataframe["macd"] > dataframe["macd_signal"]).astype(int)

        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe.loc[
            (
                (dataframe["rsi"] < 40) &
                (dataframe["macd"] > dataframe["macd_signal"]) &
                (dataframe["close"] > dataframe["bb_lower"]) &
                (dataframe["adx"] > 20)
            ),
            "enter_long"] = 1

        dataframe.loc[
            (
                (dataframe["rsi"] > 60) &
                (dataframe["macd"] < dataframe["macd_signal"]) &
                (dataframe["close"] < dataframe["bb_upper"]) &
                (dataframe["adx"] > 20)
            ),
            "enter_short"] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe.loc[
            (
                (dataframe["rsi"] > 70) |
                (dataframe["macd"] < dataframe["macd_signal"])
            ),
            "exit_long"] = 1

        dataframe.loc[
            (
                (dataframe["rsi"] < 30) |
                (dataframe["macd"] > dataframe["macd_signal"])
            ),
            "exit_short"] = 1

        return dataframe


class BreakoutStrategy(BaseStrategy):
    def __init__(self):
        super().__init__()
        self.name = "BreakoutStrategy"
        self.timeframe = "15m"
        self.stoploss = -0.03
        self.minimal_roi = {"0": 0.06, "15": 0.03, "30": 0.01, "60": 0}

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        close = dataframe["close"].tolist()
        high = dataframe["high"].tolist()
        low = dataframe["low"].tolist()

        bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(close, 20, 2.0)
        atr = calculate_atr(high, low, close, 14)
        adx = calculate_adx(high, low, close, 14)
        ema_20 = calculate_ema(close, 20)
        ema_50 = calculate_ema(close, 50)

        dataframe["bb_upper"] = bb_upper
        dataframe["bb_middle"] = bb_middle
        dataframe["bb_lower"] = bb_lower
        dataframe["atr"] = atr
        dataframe["adx"] = adx
        dataframe["ema_20"] = ema_20
        dataframe["ema_50"] = ema_50

        dataframe["bb_width"] = (dataframe["bb_upper"] - dataframe["bb_lower"]) / dataframe["bb_middle"]
        dataframe["close_above_bb_upper"] = (dataframe["close"] > dataframe["bb_upper"]).astype(int)
        dataframe["close_below_bb_lower"] = (dataframe["close"] < dataframe["bb_lower"]).astype(int)

        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe.loc[
            (
                (dataframe["close_above_bb_upper"] == 1) &
                (dataframe["adx"] > 25) &
                (dataframe["ema_20"] > dataframe["ema_50"])
            ),
            "enter_long"] = 1

        dataframe.loc[
            (
                (dataframe["close_below_bb_lower"] == 1) &
                (dataframe["adx"] > 25) &
                (dataframe["ema_20"] < dataframe["ema_50"])
            ),
            "enter_short"] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe.loc[
            (dataframe["close"] < dataframe["bb_middle"]),
            "exit_long"] = 1

        dataframe.loc[
            (dataframe["close"] > dataframe["bb_middle"]),
            "exit_short"] = 1

        return dataframe


class MeanReversionStrategy(BaseStrategy):
    def __init__(self):
        super().__init__()
        self.name = "MeanReversionStrategy"
        self.timeframe = "1h"
        self.stoploss = -0.05
        self.minimal_roi = {"0": 0.10, "60": 0.05, "120": 0.02, "240": 0}

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        close = dataframe["close"].tolist()
        high = dataframe["high"].tolist()
        low = dataframe["low"].tolist()

        rsi = calculate_rsi(close, 14)
        bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(close, 20, 2.0)
        ema_50 = calculate_ema(close, 50)
        stoch_k, stoch_d = calculate_stochastic_rsi(close, 14, 14, 3)

        dataframe["rsi"] = rsi
        dataframe["bb_upper"] = bb_upper
        dataframe["bb_middle"] = bb_middle
        dataframe["bb_lower"] = bb_lower
        dataframe["ema_50"] = ema_50
        dataframe["stoch_k"] = stoch_k
        dataframe["stoch_d"] = stoch_d

        dataframe["bb_pct"] = (dataframe["close"] - dataframe["bb_lower"]) / (dataframe["bb_upper"] - dataframe["bb_lower"])

        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe.loc[
            (
                (dataframe["rsi"] < 30) &
                (dataframe["bb_pct"] < 0.1) &
                (dataframe["stoch_k"] < 20)
            ),
            "enter_long"] = 1

        dataframe.loc[
            (
                (dataframe["rsi"] > 70) &
                (dataframe["bb_pct"] > 0.9) &
                (dataframe["stoch_k"] > 80)
            ),
            "enter_short"] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe.loc[
            (dataframe["bb_pct"] > 0.7),
            "exit_long"] = 1

        dataframe.loc[
            (dataframe["bb_pct"] < 0.3),
            "exit_short"] = 1

        return dataframe


STRATEGY_MAP = {
    "momentum": MomentumStrategy,
    "breakout": BreakoutStrategy,
    "mean_reversion": MeanReversionStrategy,
}


class FreqtradeRunner:
    def __init__(self):
        self.active_strategy: Optional[BaseStrategy] = None
        self.strategy_name: str = "momentum"
        self.pairs: list[str] = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        self.is_running = False
        self.last_signals: list[dict] = []
        self.trade_count = 0
        self._load_config()

    def _config_path(self) -> Path:
        return DATA_DIR / "runner_config.json"

    def _load_config(self):
        p = self._config_path()
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                self.strategy_name = data.get("strategy", "momentum")
                self.pairs = data.get("pairs", self.pairs)
            except Exception:
                pass

    def _save_config(self):
        config = {
            "strategy": self.strategy_name,
            "pairs": self.pairs,
            "updated_at": datetime.now().isoformat(),
        }
        self._config_path().write_text(json.dumps(config, indent=2), encoding="utf-8")

    def set_strategy(self, name: str) -> bool:
        if name in STRATEGY_MAP:
            self.strategy_name = name
            self.active_strategy = STRATEGY_MAP[name]()
            self._save_config()
            return True
        return False

    def set_pairs(self, pairs: list[str]):
        self.pairs = pairs
        self._save_config()

    async def _fetch_klines(self, symbol: str, interval: str = "1h", limit: int = 200) -> pd.DataFrame:
        client = await get_client()
        url = f"{BASE_URL}/api/v1/contract/klines/{symbol}"
        params = {"interval": interval, "limit": limit}
        try:
            resp = await client.get(url, params=params)
            if resp.status_code != 200:
                return pd.DataFrame()
            data = resp.json().get("data", [])
            if not data:
                return pd.DataFrame()
            df = pd.DataFrame(data, columns=[
                "timestamp", "open", "high", "low", "close", "volume",
                "turnover", "start_time", "end_time",
            ])
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            return df
        except Exception:
            return pd.DataFrame()

    async def _get_current_price(self, symbol: str) -> float:
        client = await get_client()
        url = f"{BASE_URL}/api/v1/contract/ticker"
        try:
            resp = await client.get(url, params={"symbol": symbol})
            if resp.status_code == 200:
                data = resp.json().get("data", {})
                return float(data.get("lastPrice", 0))
        except Exception:
            pass
        return 0.0

    async def analyze_pair(self, pair: str) -> list[Signal]:
        signals = []

        if not self.active_strategy:
            self.active_strategy = STRATEGY_MAP.get(self.strategy_name, MomentumStrategy)()

        interval = self.active_strategy.timeframe
        limit = 200 if interval in ("1h", "4h") else 500

        df = await self._fetch_klines(pair, interval, limit)
        if df.empty or len(df) < 50:
            return signals

        metadata = {"pair": pair, "timeframe": interval}
        df = self.active_strategy.populate_indicators(df, metadata)
        df = self.active_strategy.populate_entry_trend(df, metadata)
        df = self.active_strategy.populate_exit_trend(df, metadata)

        latest = df.iloc[-1]
        current_price = float(latest["close"])

        rsi = float(latest.get("rsi", 50)) if pd.notna(latest.get("rsi")) else 50
        adx = float(latest.get("adx", 0)) if pd.notna(latest.get("adx")) else 0
        macd_hist = float(latest.get("macd_hist", 0)) if pd.notna(latest.get("macd_hist")) else 0

        confidence = 0.0
        if adx > 25:
            confidence += 0.3
        if abs(rsi - 50) > 15:
            confidence += 0.2
        if abs(macd_hist) > 0:
            confidence += 0.2
        confidence = min(confidence + 0.3, 1.0)

        if latest.get("enter_long", 0) == 1:
            atr = float(latest.get("atr", current_price * 0.01)) if pd.notna(latest.get("atr")) else current_price * 0.01
            sl = current_price - atr * 2
            tp = current_price + atr * 3
            signals.append(Signal(
                pair=pair,
                direction="long",
                enter_tag=f"{self.strategy_name}_long",
                price=current_price,
                stoploss=sl,
                takeprofit=tp,
                confidence=confidence,
            ))

        if latest.get("enter_short", 0) == 1:
            atr = float(latest.get("atr", current_price * 0.01)) if pd.notna(latest.get("atr")) else current_price * 0.01
            sl = current_price + atr * 2
            tp = current_price - atr * 3
            signals.append(Signal(
                pair=pair,
                direction="short",
                enter_tag=f"{self.strategy_name}_short",
                price=current_price,
                stoploss=sl,
                takeprofit=tp,
                confidence=confidence,
            ))

        if latest.get("exit_long", 0) == 1 or latest.get("exit_short", 0) == 1:
            exit_dir = "long" if latest.get("exit_long", 0) == 1 else "short"
            signals.append(Signal(
                pair=pair,
                direction="exit",
                enter_tag=f"{self.strategy_name}_exit_{exit_dir}",
                price=current_price,
                confidence=confidence,
            ))

        return signals

    async def run_analysis(self) -> dict:
        all_signals = []
        pair_results = {}

        for pair in self.pairs:
            signals = await self.analyze_pair(pair)
            all_signals.extend(signals)
            pair_results[pair] = {
                "signals": len(signals),
                "direction": signals[0].direction if signals else "none",
                "confidence": signals[0].confidence if signals else 0,
            }

        self.last_signals = [
            {
                "pair": s.pair,
                "direction": s.direction,
                "tag": s.enter_tag,
                "price": s.price,
                "stoploss": s.stoploss,
                "takeprofit": s.takeprofit,
                "confidence": round(s.confidence, 3),
                "timestamp": s.timestamp,
            }
            for s in all_signals
        ]

        return {
            "strategy": self.strategy_name,
            "pairs_analyzed": len(self.pairs),
            "signals_generated": len(all_signals),
            "pair_results": pair_results,
            "signals": self.last_signals,
        }

    def get_status(self) -> dict:
        return {
            "strategy": self.strategy_name,
            "available_strategies": list(STRATEGY_MAP.keys()),
            "pairs": self.pairs,
            "is_running": self.is_running,
            "last_signals": self.last_signals,
            "trade_count": self.trade_count,
        }


strategy_runner = FreqtradeRunner()
