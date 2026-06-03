import time
import logging
from datetime import datetime, timezone
from collections import deque

logger = logging.getLogger(__name__)


class RiskManager:
    def __init__(self, config):
        self.config = config
        self._daily_pnl = 0.0
        self._daily_trades = 0
        self._consecutive_losses = 0
        self._max_capital = 5000.0
        self._last_daily_reset = datetime.now(timezone.utc).day
        self._drawdown_history = deque(maxlen=100)

    def check_entry_allowed(self, symbol, signal, confidence, price, portfolio, agent_states):
        now = datetime.now(timezone.utc)
        if now.day != self._last_daily_reset:
            self._daily_pnl = 0.0
            self._daily_trades = 0
            self._consecutive_losses = 0
            self._last_daily_reset = now.day

        capital = portfolio.get("capital", 5000)
        initial = portfolio.get("initial_capital", 5000)
        dd = max(0, (initial - capital) / initial)

        if dd > self.config.max_drawdown:
            return False, "max_drawdown_exceeded"

        daily_loss = abs(min(0, self._daily_pnl))
        if daily_loss / initial > self.config.daily_loss_limit:
            return False, "daily_loss_limit"

        if self._consecutive_losses >= self.config.consecutive_loss_limit:
            return False, "consecutive_losses"

        if self._daily_trades >= 50:
            return False, "daily_trade_limit"

        return True, ""

    def calculate_position_size(self, symbol, price, confidence, portfolio, agent_states):
        capital = portfolio.get("capital", 5000)
        base = self.config.base_position_size
        conf_factor = confidence / 100.0
        size = base * conf_factor
        return max(1, min(size, capital * 0.2))

    def save_state(self):
        return {
            "daily_pnl": self._daily_pnl,
            "daily_trades": self._daily_trades,
            "consecutive_losses": self._consecutive_losses,
            "max_capital": self._max_capital,
        }

    def load_state(self, state):
        self._daily_pnl = state.get("daily_pnl", 0)
        self._daily_trades = state.get("daily_trades", 0)
        self._consecutive_losses = state.get("consecutive_losses", 0)
        self._max_capital = state.get("max_capital", 5000)

    def record_trade(self, pnl):
        self._daily_trades += 1
        self._daily_pnl += pnl
        if pnl < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0
