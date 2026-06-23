import time
from .base_agent import BaseAgent


class PortfolioManagerAgent(BaseAgent):
    def __init__(self):
        super().__init__("PortfolioManager")
        self.open_positions: list[dict] = []
        self.trade_history: list[dict] = []
        self.total_equity = 10000.0
        self.available_capital = 10000.0
        self.max_positions = 5
        self.max_per_position_pct = 0.25

    async def analyze(self, data: dict) -> dict:
        try:
            self.update_status("running")

            self.total_equity = data.get("total_equity", self.total_equity)
            self.open_positions = data.get("open_positions", [])
            self.available_capital = data.get("available_capital", self.total_equity)

            scanner = data.get("scanner", {})
            technical = data.get("technical", {})
            risk_decision = data.get("risk_decision", {})

            portfolio_status = self._get_portfolio_status()
            allocation = self._calculate_allocation(technical, risk_decision)
            correlation_risk = self._check_correlation()
            rebalance = self._check_rebalance_needed()

            actions = self._determine_actions(technical, risk_decision, portfolio_status)

            self.update_status("ready")
            return {
                "portfolio_status": portfolio_status,
                "allocation_plan": allocation,
                "correlation_risk": correlation_risk,
                "rebalance_needed": rebalance,
                "actions": actions,
                "open_positions": self.open_positions,
                "total_equity": self.total_equity,
                "available_capital": self.available_capital,
            }

        except Exception as e:
            self.update_status("error", str(e))
            return {"actions": [], "error": str(e)}

    def _get_portfolio_status(self) -> dict:
        total_exposure = sum(abs(p.get("size_usd", 0)) for p in self.open_positions)
        long_count = sum(1 for p in self.open_positions if p.get("direction") == "long")
        short_count = sum(1 for p in self.open_positions if p.get("direction") == "short")
        unrealized_pnl = sum(p.get("unrealized_pnl", 0) for p in self.open_positions)

        symbols = [p.get("symbol") for p in self.open_positions]

        return {
            "position_count": len(self.open_positions),
            "max_positions": self.max_positions,
            "total_exposure_usd": total_exposure,
            "exposure_pct": total_exposure / self.total_equity if self.total_equity > 0 else 0,
            "long_count": long_count,
            "short_count": short_count,
            "unrealized_pnl": unrealized_pnl,
            "position_symbols": symbols,
        }

    def _calculate_allocation(self, technical: dict, risk_decision: dict) -> list[dict]:
        if not risk_decision.get("approved"):
            return []

        available = self.available_capital
        max_per = self.total_equity * self.max_per_position_pct

        allocation = {
            "symbol": risk_decision.get("symbol", ""),
            "direction": risk_decision.get("direction", ""),
            "size_usd": min(risk_decision.get("position_size_usd", 0), max_per, available),
            "leverage": risk_decision.get("leverage", 1),
            "stop_loss": risk_decision.get("stop_loss", 0),
            "take_profit": risk_decision.get("take_profit", 0),
        }

        return [allocation] if allocation["size_usd"] > 0 else []

    def _check_correlation(self) -> float:
        if len(self.open_positions) < 2:
            return 0.0

        directions = [p.get("direction", "") for p in self.open_positions]
        same_dir_count = max(directions.count("long"), directions.count("short"))
        correlation = same_dir_count / len(self.open_positions) if self.open_positions else 0
        return correlation

    def _check_rebalance_needed(self) -> bool:
        if not self.open_positions:
            return False

        for pos in self.open_positions:
            size = abs(pos.get("size_usd", 0))
            if size > self.total_equity * 0.3:
                return True

        return False

    def _determine_actions(self, technical: dict, risk_decision: dict, portfolio_status: dict) -> list[dict]:
        actions = []

        if risk_decision.get("approved") and portfolio_status["position_count"] < self.max_positions:
            actions.append({
                "type": "open",
                "symbol": risk_decision.get("symbol", ""),
                "direction": risk_decision.get("direction", ""),
                "size_usd": risk_decision.get("position_size_usd", 0),
                "leverage": risk_decision.get("leverage", 1),
                "stop_loss": risk_decision.get("stop_loss", 0),
                "take_profit": risk_decision.get("take_profit", 0),
                "reason": "Patron onayi ile pozisyon acma",
            })

        for pos in self.open_positions:
            symbol = pos.get("symbol", "")
            tech_signal = technical.get("results", {}).get(symbol, {})
            if tech_signal:
                current_dir = pos.get("direction", "")
                signal_dir = tech_signal.get("direction", "hold")

                if current_dir == "long" and signal_dir == "short":
                    actions.append({
                        "type": "close",
                        "symbol": symbol,
                        "reason": "Teknik analiz yon degistirme sinyali",
                    })
                elif current_dir == "short" and signal_dir == "long":
                    actions.append({
                        "type": "close",
                        "symbol": symbol,
                        "reason": "Teknik analiz yon degistirme sinyali",
                    })

        return actions

    def add_position(self, position: dict):
        self.open_positions.append(position)
        self.available_capital -= abs(position.get("size_usd", 0))

    def remove_position(self, symbol: str):
        self.open_positions = [p for p in self.open_positions if p.get("symbol") != symbol]
        self._recalculate_available()

    def update_position(self, symbol: str, updates: dict):
        for pos in self.open_positions:
            if pos.get("symbol") == symbol:
                pos.update(updates)
                break
        self._recalculate_available()

    def _recalculate_available(self):
        used = sum(abs(p.get("size_usd", 0)) for p in self.open_positions)
        self.available_capital = max(0, self.total_equity - used)
