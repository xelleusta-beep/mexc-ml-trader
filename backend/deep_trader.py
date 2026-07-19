"""
BTC/ETH Deep Learning Trader
- Tüm zamanların fiyat verisi ile training
- Random Forest + Gradient Boosting modelleri
- Destek/Direnç tespiti
- Bağımsız TP/SL yönetimi
- Yüksek kaldıraçlı profesyonel işlemler
"""

import json
import time
import asyncio
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from indicators import (
    calculate_rsi,
    calculate_rsi_ma,
    calculate_macd,
    calculate_bollinger_bands,
    calculate_ema,
    calculate_atr,
    calculate_adx,
    calculate_stochastic_rsi,
)
from mexc_client import get_client, BASE_URL, TAKER_FEE

DATA_DIR = Path(__file__).parent.parent / "data" / "deep_trader"
DATA_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class DeepPosition:
    symbol: str
    direction: str
    entry_price: float
    size_usd: float
    leverage: int
    tp_price: float
    sl_price: float
    entry_time: float
    entry_fee: float = 0.0
    unrealized_pnl: float = 0.0
    current_price: float = 0.0
    model_confidence: float = 0.0
    signals_used: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "direction": self.direction,
            "entry_price": self.entry_price,
            "size_usd": self.size_usd,
            "leverage": self.leverage,
            "tp_price": self.tp_price,
            "sl_price": self.sl_price,
            "entry_time": self.entry_time,
            "entry_fee": self.entry_fee,
            "unrealized_pnl": self.unrealized_pnl,
            "current_price": self.current_price,
            "model_confidence": self.model_confidence,
            "signals_used": self.signals_used,
        }


class DeepTrader:
    def __init__(self, symbol: str = "BTCUSDT", capital: float = 1000.0):
        self.symbol = symbol
        self.total_equity = capital
        self.available_capital = capital
        self.max_leverage = 20
        self.max_risk_per_trade = 0.02

        self.positions: list[DeepPosition] = []
        self.trade_history: list[dict] = []
        self._models_trained = False
        self._rf_model = None
        self._gb_model = None
        self._feature_names: list[str] = []

        self._load_state()

    def _state_path(self) -> Path:
        safe = self.symbol.replace("/", "_")
        return DATA_DIR / f"{safe}_state.json"

    def _history_path(self) -> Path:
        safe = self.symbol.replace("/", "_")
        return DATA_DIR / f"{safe}_history.json"

    def _load_state(self):
        p = self._state_path()
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                self.total_equity = data.get("total_equity", self.total_equity)
                self.available_capital = data.get("available_capital", self.available_capital)
                self.positions = [DeepPosition(**pos) for pos in data.get("positions", [])]
            except Exception:
                pass

        hp = self._history_path()
        if hp.exists():
            try:
                self.trade_history = json.loads(hp.read_text(encoding="utf-8"))
            except Exception:
                pass

    def save_state(self):
        state = {
            "total_equity": self.total_equity,
            "available_capital": self.available_capital,
            "positions": [p.to_dict() for p in self.positions],
            "updated_at": datetime.now().isoformat(),
        }
        self._state_path().write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")

        self._history_path().write_text(
            json.dumps(self.trade_history[-500:], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    async def fetch_all_klines(self, interval: str = "1h", limit: int = 1000) -> pd.DataFrame:
        client = await get_client()
        all_klines = []
        end_time = int(time.time() * 1000)

        while len(all_klines) < limit:
            batch = min(1000, limit - len(all_klines))
            url = f"{BASE_URL}/api/v1/contract/klines/{self.symbol}"
            params = {"interval": interval, "limit": batch}
            if end_time:
                params["end"] = end_time

            try:
                resp = await client.get(url, params=params)
                if resp.status_code != 200:
                    break
                data = resp.json().get("data", [])
                if not data:
                    break
                all_klines = data + all_klines
                end_time = int(data[0][0]) - 1
                if len(data) < batch:
                    break
                await asyncio.sleep(0.1)
            except Exception:
                break

        if not all_klines:
            return pd.DataFrame()

        df = pd.DataFrame(all_klines, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "turnover", "start_time", "end_time",
        ])
        for col in ["open", "high", "low", "close", "volume", "turnover"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        return df

    def _compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"].tolist()
        high = df["high"].tolist()
        low = df["low"].tolist()
        volume = df["volume"].tolist()

        rsi = calculate_rsi(close, 14)
        rsi_ma = calculate_rsi_ma(rsi, 14)
        macd_line, signal_line, histogram = calculate_macd(close)
        bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(close, 20, 2.0)
        ema_9 = calculate_ema(close, 9)
        ema_21 = calculate_ema(close, 21)
        ema_50 = calculate_ema(close, 50)
        atr = calculate_atr(high, low, close, 14)
        adx = calculate_adx(high, low, close, 14)
        stoch_k, stoch_d = calculate_stochastic_rsi(close, 14, 14, 3)

        obv = [0.0]
        for i in range(1, len(close)):
            if close[i] > close[i-1]:
                obv.append(obv[-1] + volume[i])
            elif close[i] < close[i-1]:
                obv.append(obv[-1] - volume[i])
            else:
                obv.append(obv[-1])

        df["rsi"] = rsi
        df["rsi_ma"] = rsi_ma
        df["macd"] = macd_line
        df["macd_signal"] = signal_line
        df["macd_hist"] = histogram
        df["bb_upper"] = bb_upper
        df["bb_middle"] = bb_middle
        df["bb_lower"] = bb_lower
        df["ema_9"] = ema_9
        df["ema_21"] = ema_21
        df["ema_50"] = ema_50
        df["atr"] = atr
        df["adx"] = adx
        df["stoch_k"] = stoch_k
        df["stoch_d"] = stoch_d
        df["obv"] = obv

        df["rsi_divergence"] = df["close"].diff(5).apply(lambda x: 1 if x > 0 else -1) * (-df["rsi"].diff(5).apply(lambda x: 1 if x > 0 else -1))
        df["price_change"] = df["close"].pct_change()
        df["volume_change"] = df["volume"].pct_change()
        df["high_low_range"] = (df["high"] - df["low"]) / df["close"]
        df["close_position"] = (df["close"] - df["low"]) / (df["high"] - df["low"] + 1e-10)

        df["support"] = df["low"].rolling(20).min()
        df["resistance"] = df["high"].rolling(20).max()
        df["dist_to_support"] = (df["close"] - df["support"]) / df["close"]
        df["dist_to_resistance"] = (df["resistance"] - df["close"]) / df["close"]

        df["sma_50"] = df["close"].rolling(50).mean()
        df["sma_200"] = df["close"].rolling(200).mean()
        df["trend_strength"] = (df["sma_50"] - df["sma_200"]) / df["sma_200"]

        return df

    def _create_labels(self, df: pd.DataFrame, lookahead: int = 12) -> pd.Series:
        future_return = df["close"].shift(-lookahead) / df["close"] - 1
        labels = pd.Series(0, index=df.index)
        labels[future_return > 0.005] = 1
        labels[future_return < -0.005] = -1
        return labels

    async def train_models(self):
        print(f"[DeepTrader] {self.symbol} icin model egitimi baslatiliyor...")

        df = await self.fetch_all_klines(interval="1h", limit=2000)
        if df.empty or len(df) < 200:
            print(f"[DeepTrader] Yeterli veri yok: {len(df)} satir")
            return False

        df = self._compute_features(df)
        labels = self._create_labels(df)

        feature_cols = [
            "rsi", "rsi_ma", "macd", "macd_signal", "macd_hist",
            "bb_upper", "bb_middle", "bb_lower",
            "ema_9", "ema_21", "ema_50",
            "adx", "stoch_k", "stoch_d",
            "rsi_divergence", "price_change", "volume_change",
            "high_low_range", "close_position",
            "dist_to_support", "dist_to_resistance",
            "trend_strength",
        ]

        valid_mask = df[feature_cols].notna().all(axis=1) & labels.notna()
        X = df.loc[valid_mask, feature_cols].values
        y = labels.loc[valid_mask].values

        if len(X) < 100:
            print(f"[DeepTrader] Yeterli temiz veri: {len(X)}")
            return False

        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        tscv = TimeSeriesSplit(n_splits=3)

        self._rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1,
        )
        self._gb_model = GradientBoostingClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.05,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42,
        )

        rf_scores = []
        gb_scores = []
        for train_idx, test_idx in tscv.split(X_scaled):
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            self._rf_model.fit(X_train, y_train)
            rf_scores.append(self._rf_model.score(X_test, y_test))

            self._gb_model.fit(X_train, y_train)
            gb_scores.append(self._gb_model.score(X_test, y_test))

        self._rf_model.fit(X_scaled, y)
        self._gb_model.fit(X_scaled, y)
        self._feature_names = feature_cols
        self._scaler = scaler
        self._models_trained = True

        avg_rf = np.mean(rf_scores)
        avg_gb = np.mean(gb_scores)
        print(f"[DeepTrader] Model egitimi tamamlandi.")
        print(f"  Random Forest accuracy: {avg_rf:.3f}")
        print(f"  Gradient Boosting accuracy: {avg_gb:.3f}")
        return True

    def _detect_support_resistance(self, df: pd.DataFrame) -> dict:
        close = df["close"].values
        high = df["high"].values
        low = df["low"].values

        recent = df.tail(100)

        pivot_highs = []
        pivot_lows = []
        for i in range(2, len(recent) - 2):
            if high[i] > high[i-1] and high[i] > high[i-2] and high[i] > high[i+1] and high[i] > high[i+2]:
                pivot_highs.append(high[i])
            if low[i] < low[i-1] and low[i] < low[i-2] and low[i] < low[i+1] and low[i] < low[i+2]:
                pivot_lows.append(low[i])

        current_price = close[-1]

        resistance_levels = sorted([p for p in pivot_highs if p > current_price])[:3]
        support_levels = sorted([p for p in pivot_lows if p < current_price], reverse=True)[:3]

        return {
            "current_price": current_price,
            "resistance": resistance_levels,
            "support": support_levels,
        }

    def _calculate_tp_sl(self, direction: str, entry_price: float, atr: float, sr_levels: dict) -> tuple[float, float]:
        atr_mult = 2.0

        if direction == "long":
            tp1 = entry_price + atr * atr_mult * 2
            tp2 = entry_price + atr * atr_mult * 3
            sl1 = entry_price - atr * atr_mult
            sl2 = entry_price - atr * atr_mult * 1.5

            if sr_levels["resistance"]:
                nearest_res = sr_levels["resistance"][0]
                tp1 = min(tp1, nearest_res * 0.998)

            if sr_levels["support"]:
                nearest_sup = sr_levels["support"][0]
                sl1 = max(sl1, nearest_sup * 1.002)

            return tp1, sl1
        else:
            tp1 = entry_price - atr * atr_mult * 2
            sl1 = entry_price + atr * atr_mult

            if sr_levels["support"]:
                nearest_sup = sr_levels["support"][0]
                tp1 = max(tp1, nearest_sup * 1.002)

            if sr_levels["resistance"]:
                nearest_res = sr_levels["resistance"][0]
                sl1 = min(sl1, nearest_res * 0.998)

            return tp1, sl1

    async def analyze_and_trade(self) -> dict:
        result = {
            "symbol": self.symbol,
            "action": "hold",
            "positions": [p.to_dict() for p in self.positions],
            "equity": self.total_equity,
            "available": self.available_capital,
            "model_trained": self._models_trained,
            "analysis": {},
        }

        for pos in list(self.positions):
            ticker_url = f"{BASE_URL}/api/v1/contract/ticker"
            client = await get_client()
            try:
                resp = await client.get(ticker_url, params={"symbol": pos.symbol})
                if resp.status_code == 200:
                    data = resp.json().get("data", {})
                    if data:
                        pos.current_price = float(data.get("lastPrice", pos.current_price))
            except Exception:
                pass

            if pos.current_price <= 0:
                pos.current_price = pos.entry_price

            if pos.direction == "long":
                pnl_pct = (pos.current_price - pos.entry_price) / pos.entry_price
            else:
                pnl_pct = (pos.entry_price - pos.current_price) / pos.entry_price

            leveraged_pnl = pnl_pct * pos.leverage
            pnl_usd = pos.size_usd * leveraged_pnl

            exit_fee = pos.size_usd * TAKER_FEE
            net_pnl = pnl_usd - pos.entry_fee - exit_fee

            hit_tp = (pos.direction == "long" and pos.current_price >= pos.tp_price) or \
                     (pos.direction == "short" and pos.current_price <= pos.tp_price)
            hit_sl = (pos.direction == "long" and pos.current_price <= pos.sl_price) or \
                     (pos.direction == "short" and pos.current_price >= pos.sl_price)

            if hit_tp or hit_sl:
                reason = "TP" if hit_tp else "SL"
                self.total_equity += net_pnl
                self.available_capital += pos.size_usd + net_pnl
                self.trade_history.append({
                    **pos.to_dict(),
                    "exit_price": pos.current_price,
                    "exit_time": time.time(),
                    "pnl": round(net_pnl, 2),
                    "pnl_pct": round(leveraged_pnl * 100, 2),
                    "reason": reason,
                })
                self.positions.remove(pos)
                self.save_state()
                result["action"] = f"closed_{reason}"
                result["closed"] = {
                    "symbol": pos.symbol,
                    "direction": pos.direction,
                    "pnl": round(net_pnl, 2),
                    "reason": reason,
                }
                return result

            pos.unrealized_pnl = round(net_pnl, 2)

        if not self._models_trained:
            await self.train_models()

        if not self._models_trained:
            return result

        df = await self.fetch_all_klines(interval="1h", limit=500)
        if df.empty:
            return result

        df = self._compute_features(df)
        sr_levels = self._detect_support_resistance(df)

        latest = df.iloc[-1]
        features = np.array([[latest[f] if pd.notna(latest[f]) else 0 for f in self._feature_names]])
        features_scaled = self._scaler.transform(features)

        rf_pred = self._rf_model.predict(features_scaled)[0]
        gb_pred = self._gb_model.predict(features_scaled)[0]
        rf_proba = self._rf_model.predict_proba(features_scaled)[0]
        gb_proba = self._gb_model.predict_proba(features_scaled)[0]

        rf_conf = max(rf_proba)
        gb_conf = max(gb_proba)
        avg_conf = (rf_conf + gb_conf) / 2

        ensemble_pred = rf_pred if rf_conf > gb_conf else gb_pred
        ensemble_conf = max(rf_conf, gb_conf)

        result["analysis"] = {
            "rf_prediction": int(rf_pred),
            "rf_confidence": round(rf_conf, 3),
            "gb_prediction": int(gb_pred),
            "gb_confidence": round(gb_conf, 3),
            "ensemble_prediction": int(ensemble_pred),
            "ensemble_confidence": round(ensemble_conf, 3),
            "support_levels": [round(s, 2) for s in sr_levels["support"]],
            "resistance_levels": [round(r, 2) for r in sr_levels["resistance"]],
            "rsi": round(float(latest.get("rsi", 50)) if pd.notna(latest.get("rsi")) else 50, 1),
            "adx": round(float(latest.get("adx", 0)) if pd.notna(latest.get("adx")) else 0, 1),
            "atr": round(float(latest.get("atr", 0)) if pd.notna(latest.get("atr")) else 0, 4),
        }

        existing = next((p for p in self.positions if p.symbol == self.symbol), None)
        if existing:
            result["action"] = "hold"
            return result

        if ensemble_pred == 0 or ensemble_conf < 0.55:
            return result

        direction = "long" if ensemble_pred == 1 else "short"
        atr_val = float(latest.get("atr", 0)) if pd.notna(latest.get("atr")) else 0
        current_price = float(latest["close"])

        if atr_val <= 0:
            atr_val = current_price * 0.01

        tp_price, sl_price = self._calculate_tp_sl(direction, current_price, atr_val, sr_levels)

        risk_pct = self.max_risk_per_trade * ensemble_conf
        position_size = self.available_capital * risk_pct * 5

        leverage = min(self.max_leverage, max(5, int(ensemble_conf * 20)))

        entry_fee = position_size * TAKER_FEE
        if position_size + entry_fee > self.available_capital:
            position_size = self.available_capital * 0.9
            entry_fee = position_size * TAKER_FEE

        new_pos = DeepPosition(
            symbol=self.symbol,
            direction=direction,
            entry_price=current_price,
            size_usd=round(position_size, 2),
            leverage=leverage,
            tp_price=round(tp_price, 4),
            sl_price=round(sl_price, 4),
            entry_time=time.time(),
            entry_fee=round(entry_fee, 6),
            current_price=current_price,
            model_confidence=round(ensemble_conf, 3),
            signals_used=[
                f"RF:{int(rf_pred)}({rf_conf:.2f})",
                f"GB:{int(gb_pred)}({gb_conf:.2f})",
                f"ADX:{result['analysis']['adx']}",
                f"RSI:{result['analysis']['rsi']}",
            ],
        )

        self.positions.append(new_pos)
        self.available_capital -= (position_size + entry_fee)
        self.save_state()

        result["action"] = f"opened_{direction}"
        result["new_position"] = new_pos.to_dict()
        return result

    def get_status(self) -> dict:
        return {
            "symbol": self.symbol,
            "total_equity": round(self.total_equity, 2),
            "available_capital": round(self.available_capital, 2),
            "positions": [p.to_dict() for p in self.positions],
            "trade_count": len(self.trade_history),
            "model_trained": self._models_trained,
            "recent_trades": self.trade_history[-10:],
        }
