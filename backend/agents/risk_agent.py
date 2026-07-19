import time
from .base_agent import BaseAgent


class RiskManagerAgent(BaseAgent):
    def __init__(self):
        super().__init__("RiskManager")
        self.max_risk_per_trade = 0.01
        self.max_positions = 5
        self.max_leverage = 10
        self.max_daily_loss_pct = 0.05
        self.min_confidence = 0.15
        self.daily_pnl = 0.0
        self.daily_start_equity = 0.0
        self.last_daily_reset = 0
        self.open_positions: list[dict] = []
        self.total_equity = 100.0

    async def analyze(self, data: dict) -> dict:
        try:
            self.update_status("running")

            self.total_equity = data.get("total_equity", self.total_equity)
            self.open_positions = data.get("open_positions", [])

            self._reset_daily_if_needed()

            risk_metrics = self._calculate_risk_metrics()

            signal = data.get("signal", {})
            portfolio = data.get("portfolio", {})

            if signal:
                decision = self._evaluate_signal(signal, portfolio, risk_metrics)
            else:
                decision = {"approved": False, "reason": "Sinyal bulunamadi"}

            self.update_status("ready")
            return {
                "risk_metrics": risk_metrics,
                "decision": decision,
                "daily_pnl": self.daily_pnl,
                "daily_loss_limit": self.max_daily_loss_pct * self.total_equity,
                "daily_remaining": max(0, self.max_daily_loss_pct * self.total_equity - abs(self.daily_pnl)),
            }

        except Exception as e:
            self.update_status("error", str(e))
            return {"decision": {"approved": False, "reason": str(e)}}

    def _reset_daily_if_needed(self):
        now = time.time()
        if now - self.last_daily_reset > 86400:
            self.daily_pnl = 0.0
            self.daily_start_equity = self.total_equity
            self.last_daily_reset = now

    def _calculate_risk_metrics(self) -> dict:
        current_positions = len(self.open_positions)
        available_slots = max(0, self.max_positions - current_positions)

        total_exposure = sum(
            abs(p.get("size_usd", 0)) for p in self.open_positions
        )
        exposure_pct = total_exposure / self.total_equity if self.total_equity > 0 else 0

        daily_loss_pct = abs(self.daily_pnl) / self.total_equity if self.total_equity > 0 and self.daily_pnl < 0 else 0

        return {
            "total_equity": self.total_equity,
            "current_positions": current_positions,
            "available_slots": available_slots,
            "total_exposure_usd": total_exposure,
            "exposure_pct": round(exposure_pct, 4),
            "daily_pnl": self.daily_pnl,
            "daily_loss_pct": round(daily_loss_pct, 4),
            "daily_limit_pct": self.max_daily_loss_pct,
            "can_trade": available_slots > 0 and daily_loss_pct < self.max_daily_loss_pct,
        }

    def _evaluate_signal(self, signal: dict, portfolio: dict, risk_metrics: dict) -> dict:
        if not risk_metrics["can_trade"]:
            reason = "Risk limiti asildi"
            if risk_metrics["available_slots"] <= 0:
                reason = "Maksimum pozisyon sayisina ulasildi"
            elif risk_metrics["daily_loss_pct"] >= self.max_daily_loss_pct:
                reason = "Gunluk kayip limiti asildi"
            return {"approved": False, "reason": reason}

        confidence = signal.get("confidence", 0)
        price = signal.get("price", 0)
        direction = signal.get("direction", "hold")
        atr_pct = signal.get("atr_pct", 0.02)

        if direction == "hold" or price <= 0:
            return {"approved": False, "reason": "Gecersiz sinyal yonu veya fiyat"}

        effective_confidence = max(confidence, 0.20)

        kelly_fraction = self._kelly_criterion(effective_confidence, 0.6, 0.4)
        position_size_pct = max(kelly_fraction * 0.5, 0.05)
        position_size_pct = min(position_size_pct, self.max_risk_per_trade * 10)
        position_size_pct = min(position_size_pct, 0.25)
        position_size_usd = self.total_equity * position_size_pct

        max_position_usd = self.total_equity / self.max_positions
        position_size_usd = min(position_size_usd, max_position_usd)

        leverage = self._calculate_leverage(atr_pct, confidence)

        stop_loss_pct = max(atr_pct * 2, 0.02)
        take_profit_pct = stop_loss_pct * 2.0

        if direction == "long":
            stop_loss = price * (1 - stop_loss_pct)
            take_profit = price * (1 + take_profit_pct)
        else:
            stop_loss = price * (1 + stop_loss_pct)
            take_profit = price * (1 - take_profit_pct)

        risk_reward = take_profit_pct / stop_loss_pct if stop_loss_pct > 0 else 0

        max_loss_usd = position_size_usd * stop_loss_pct * leverage

        return {
            "approved": True,
            "position_size_usd": round(position_size_usd, 2),
            "leverage": leverage,
            "stop_loss": round(stop_loss, 8),
            "take_profit": round(take_profit, 8),
            "stop_loss_pct": round(stop_loss_pct * 100, 2),
            "take_profit_pct": round(take_profit_pct * 100, 2),
            "risk_reward_ratio": round(risk_reward, 2),
            "max_loss_usd": round(max_loss_usd, 2),
            "direction": direction,
            "confidence": confidence,
        }

    def _kelly_criterion(self, win_prob: float, avg_win: float, avg_loss: float) -> float:
        if avg_loss == 0:
            return 0.0
        b = avg_win / avg_loss
        f = (win_prob * b - (1 - win_prob)) / b
        return max(0.0, min(f, 0.25))

    def _calculate_leverage(self, atr_pct: float, confidence: float) -> int:
        if atr_pct <= 0:
            return 1

        base_leverage = int(1.0 / atr_pct) if atr_pct > 0 else 1
        confidence_mult = 0.5 + confidence * 0.5
        leverage = int(base_leverage * confidence_mult)
        leverage = max(1, min(leverage, self.max_leverage))

        if atr_pct > 0.05:
            leverage = min(leverage, 5)
        elif atr_pct > 0.03:
            leverage = min(leverage, 10)
        elif atr_pct > 0.01:
            leverage = min(leverage, 15)

        return leverage

    def record_trade_pnl(self, pnl: float):
        self.daily_pnl += pnl
