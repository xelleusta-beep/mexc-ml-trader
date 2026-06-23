from position_manager import Position
from typing import Optional
import datetime
import math


class BacktestEngine:
    """Profesyonel, risk kontrollü ve kademeli trade backtest motoru."""

    def __init__(
        self,
        initial_capital: float = 10000.0,
        entry_amount: float = 1000.0,
        dca_30_amount: float = 3000.0,
        dca_60_amount: float = 6000.0,
        entry_threshold: float = 30.0,
        exit_threshold_2: float = 50.0,
        exit_threshold_3: float = 70.0,
        exit_pct_1: float = 25.0,
        exit_pct_2: float = 25.0,
        exit_pct_3: float = 50.0,
        dca_30_drop: float = 0.30,
        dca_60_drop: float = 0.60,
        maker_fee: float = 0.0002,
        taker_fee: float = 0.0006,
        strategy_mode: str = "rsi",
        ema_fast_period: int = 20,
        ema_slow_period: int = 50,
        cooldown_bars: int = 1,
        max_dca_trades: int = 2,
        dca_disable_after_exit_pct: float = 50.0,
        require_profit_for_staged_exit: bool = False,
        min_hold_bars: int = 0,
        force_exit_after_bars: int = 0,
        break_even_stop_after_exit_pct: float = 50.0,
        use_adx_filter: bool = False,
        adx_threshold: float = 25.0,
        use_bb_filter: bool = False,
        use_macd_filter: bool = False,
        use_volume_filter: bool = False,
        use_stochrsi_filter: bool = False,
        use_volatility_filter: bool = False,
        max_atr_pct: float = 0.12,
        use_regime_filter: bool = False,
        ema_regime_period: int = 200,
        use_divergence_filter: bool = False,
        divergence_lookback: int = 20,
        use_breakout_confirmation: bool = False,
        use_trailing_stop: bool = False,
        trailing_stop_pct: float = 0.05,
        use_atr_stop: bool = False,
        atr_multiplier: float = 2.0,
    ):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.entry_amount = entry_amount
        self.dca_30_amount = dca_30_amount
        self.dca_60_amount = dca_60_amount
        self.entry_threshold = entry_threshold
        self.exit_threshold_2 = exit_threshold_2
        self.exit_threshold_3 = exit_threshold_3
        self.exit_pct_1 = exit_pct_1
        self.exit_pct_2 = exit_pct_2
        self.exit_pct_3 = exit_pct_3
        self.dca_30_drop = dca_30_drop
        self.dca_60_drop = dca_60_drop
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.strategy_mode = strategy_mode
        self.ema_fast_period = ema_fast_period
        self.ema_slow_period = ema_slow_period
        self.cooldown_bars = max(0, int(cooldown_bars))
        self.max_dca_trades = max(0, int(max_dca_trades))
        self.dca_disable_after_exit_pct = dca_disable_after_exit_pct
        self.require_profit_for_staged_exit = require_profit_for_staged_exit
        self.min_hold_bars = max(0, int(min_hold_bars))
        self.force_exit_after_bars = max(0, int(force_exit_after_bars))
        self.break_even_stop_after_exit_pct = break_even_stop_after_exit_pct

        self.use_adx_filter = use_adx_filter
        self.adx_threshold = adx_threshold
        self.use_bb_filter = use_bb_filter
        self.use_macd_filter = use_macd_filter
        self.use_volume_filter = use_volume_filter
        self.use_stochrsi_filter = use_stochrsi_filter
        self.use_volatility_filter = use_volatility_filter
        self.max_atr_pct = max_atr_pct
        self.use_regime_filter = use_regime_filter
        self.ema_regime_period = ema_regime_period
        self.use_divergence_filter = use_divergence_filter
        self.divergence_lookback = max(5, int(divergence_lookback))
        self.use_breakout_confirmation = use_breakout_confirmation

        self.use_trailing_stop = use_trailing_stop
        self.trailing_stop_pct = trailing_stop_pct
        self.use_atr_stop = use_atr_stop
        self.atr_multiplier = atr_multiplier

        self.position: Optional[Position] = None
        self.entry_bar: Optional[int] = None
        self.stop_loss_price: float = 0.0
        self.trailing_stop_price: float = 0.0
        self.closed_trades: list[dict] = []
        self.equity_curve: list[dict] = []

    def _apply_fee(self, amount: float, is_maker: bool = False) -> float:
        fee_rate = self.maker_fee if is_maker else self.taker_fee
        return max(0.0, amount * fee_rate)

    def _safe_pct(self, value: float, base: float) -> float:
        if not base or base <= 0:
            return 0.0
        return value / base * 100.0

    def _exit_pct_sold(self) -> float:
        if not self.position:
            return 0.0
        total = 0.0
        if self.position.exit_1_done:
            total += self.exit_pct_1
        if self.position.exit_2_done:
            total += self.exit_pct_2
        if self.position.exit_3_done:
            total += self.exit_pct_3
        return total

    def _dca_count(self) -> int:
        if not self.position:
            return 0
        return sum(1 for t in self.position.trades if t.reason in ("dca_30", "dca_60"))

    def _has_bullish_divergence(self, lows: list[float | None], rsi: list[float | None], idx: int) -> bool:
        if not lows or not rsi or idx <= 0:
            return True

        start = max(0, idx - self.divergence_lookback)
        candidates = []
        for i in range(start, idx):
            low = lows[i]
            rsi_value = rsi[i]
            if low is not None and rsi_value is not None:
                candidates.append((i, low, rsi_value))

        if len(candidates) < 2:
            return True

        previous_low_idx, previous_low, previous_rsi = min(candidates, key=lambda x: x[1])
        current_low = lows[idx]
        current_rsi = rsi[idx]

        if current_low is None or current_rsi is None:
            return True

        return current_low <= previous_low and current_rsi > previous_rsi

    def _buy(self, price: float, amount_usd: float, timestamp: int, reason: str):
        if price <= 0 or amount_usd <= 0 or self.capital <= 0:
            return

        fee_rate = self.taker_fee
        max_affordable = self.capital / (1 + fee_rate)
        amount_usd = min(amount_usd, max_affordable)
        fee = self._apply_fee(amount_usd, is_maker=False)
        total_cost = amount_usd + fee

        if total_cost <= 0 or self.capital < total_cost:
            return

        self.capital -= total_cost

        if self.position is None:
            self.position = Position()
            self.position.enter(price, amount_usd, timestamp, fee)
        else:
            self.position.add_dca(price, amount_usd, timestamp, reason, fee)

    def _sell(self, price: float, pct: float, timestamp: int, reason: str):
        if self.position is None or not self.position.is_open or price <= 0:
            return

        pct = max(0.0, min(100.0, pct))
        gross_received = self.position.coin_amount * (pct / 100.0) * price
        fee = self._apply_fee(gross_received, is_maker=True)
        usd_received = self.position.sell(price, pct, timestamp, reason, fee)
        self.capital += usd_received - fee

        if not self.position.is_open:
            self.stop_loss_price = 0.0
            self.trailing_stop_price = 0.0
            self._close_position(timestamp, reason)

    def _sell_all(self, price: float, timestamp: int, reason: str = "sell_all"):
        if self.position is None or not self.position.is_open or price <= 0:
            return

        gross_received = self.position.coin_amount * price
        fee = self._apply_fee(gross_received, is_maker=True)
        usd_received = self.position.sell_all(price, timestamp, reason, fee)
        self.capital += usd_received - fee

        self.stop_loss_price = 0.0
        self.trailing_stop_price = 0.0
        self._close_position(timestamp, reason)

    def _close_position(self, timestamp: int, reason: str = "closed"):
        if not self.position:
            return

        position = self.position
        entry_date = position.entry_timestamp if position.entry_timestamp else timestamp
        duration_days = (timestamp - entry_date) / (1000 * 86400) if timestamp > entry_date else 0
        first_entry_price = position.first_entry_price if position.first_entry_price > 0 else 0
        avg_price = position.avg_price if position.avg_price > 0 else first_entry_price
        total_entry_invested = position.total_entry_invested if position.total_entry_invested > 0 else position.total_buy_usd

        max_price = position.max_price if position.max_price > 0 else first_entry_price
        min_price = position.min_price if position.min_price != float('inf') and position.min_price > 0 else first_entry_price

        dca_count = self._dca_count()
        exit_pct_sold = self._exit_pct_sold()

        self.closed_trades.append({
            "first_entry_price": first_entry_price,
            "avg_price": avg_price,
            "total_invested": total_entry_invested,
            "entry_date": entry_date,
            "close_date": timestamp,
            "entry_date_str": datetime.datetime.fromtimestamp(entry_date / 1000).strftime('%Y-%m-%d'),
            "close_date_str": datetime.datetime.fromtimestamp(timestamp / 1000).strftime('%Y-%m-%d'),
            "duration_days": round(duration_days, 2),
            "max_price": max_price,
            "min_price": min_price,
            "max_gain_pct": self._safe_pct(max_price - first_entry_price, first_entry_price),
            "max_loss_pct": self._safe_pct(min_price - first_entry_price, first_entry_price),
            "trades": [
                {
                    "type": t.type,
                    "price": t.price,
                    "amount_usd": t.amount_usd,
                    "coin_amount": t.coin_amount,
                    "reason": t.reason,
                    "fee": t.fee,
                    "realized_pnl": t.realized_pnl,
                    "date_str": datetime.datetime.fromtimestamp(t.timestamp / 1000).strftime('%Y-%m-%d'),
                }
                for t in position.trades
            ],
            "close_time": timestamp,
            "pnl": position.realized_pnl,
            "pnl_pct": self._safe_pct(position.realized_pnl, total_entry_invested),
            "close_reason": reason,
            "total_fees": position.total_fees,
            "total_buy_usd": position.total_buy_usd,
            "total_sell_usd": position.total_sell_usd,
            "dca_count": dca_count,
            "exit_pct_sold": exit_pct_sold,
            "max_unrealized_pnl": position.max_unrealized_pnl,
            "min_unrealized_pnl": position.min_unrealized_pnl,
            "max_unrealized_pct": self._safe_pct(position.max_unrealized_pnl, total_entry_invested),
            "min_unrealized_pct": self._safe_pct(position.min_unrealized_pnl, total_entry_invested),
            "quality_score": self._trade_quality_score(position.realized_pnl, total_entry_invested, dca_count, exit_pct_sold),
        })

        self.position = None
        self.entry_bar = None

    def _trade_quality_score(self, pnl: float, invested: float, dca_count: int, exit_pct_sold: float) -> float:
        pnl_pct = self._safe_pct(pnl, invested)
        return pnl_pct - (dca_count * 1.5) + min(exit_pct_sold, 100) * 0.08

    def _entry_allowed(
        self,
        i: int,
        price: float,
        volume: float,
        current_adx: Optional[float],
        current_macd: Optional[float],
        current_macd_sig: Optional[float],
        current_bb_lower: Optional[float],
        current_vol_sma: Optional[float],
        current_stoch_k: Optional[float],
        current_atr: Optional[float],
        current_regime_ema: Optional[float],
        lows: list[float | None],
        rsi: list[float | None],
        prev_close: Optional[float],
    ) -> bool:
        if price <= 0:
            return False

        if self.use_adx_filter and current_adx is not None and current_adx < self.adx_threshold:
            return False

        if self.use_bb_filter and current_bb_lower is not None and price > current_bb_lower:
            return False

        if self.use_macd_filter and current_macd is not None and current_macd_sig is not None:
            if current_macd <= current_macd_sig:
                return False

        if self.use_volume_filter and current_vol_sma is not None and volume < current_vol_sma:
            return False

        if self.use_stochrsi_filter and current_stoch_k is not None and current_stoch_k > 20:
            return False

        if self.use_volatility_filter and current_atr is not None and price > 0:
            if current_atr / price > self.max_atr_pct:
                return False

        if self.use_regime_filter and current_regime_ema is not None and price < current_regime_ema:
            return False

        if self.use_divergence_filter and not self._has_bullish_divergence(lows, rsi, i):
            return False

        if self.use_breakout_confirmation and prev_close is not None and price <= prev_close:
            return False

        return True

    def _should_enter(
        self,
        current_rsi_ma: Optional[float],
        prev_rsi_ma: Optional[float],
        current_trend: str,
        prev_trend: str,
    ) -> tuple[bool, str]:
        if self.strategy_mode == "rsi":
            if current_rsi_ma is not None and prev_rsi_ma is not None and prev_rsi_ma >= self.entry_threshold and current_rsi_ma < self.entry_threshold:
                return True, "rsi_entry"

        elif self.strategy_mode == "trend":
            if current_trend == 'buy' and prev_trend != 'buy':
                return True, "trend_entry"

        elif self.strategy_mode == "combined":
            rsi_signal = current_rsi_ma is not None and prev_rsi_ma is not None and prev_rsi_ma >= self.entry_threshold and current_rsi_ma < self.entry_threshold
            trend_signal = current_trend == 'buy' and prev_trend != 'buy'
            if rsi_signal and trend_signal:
                return True, "combined_entry"

        return False, ""

    def _append_equity(self, timestamp: int, price: float, rsi_value: Optional[float], rsi_ma_value: Optional[float]):
        position_value = self.position.get_value(price) if self.position and self.position.is_open else 0.0
        self.equity_curve.append({
            "time": timestamp,
            "capital": self.capital,
            "position_value": position_value,
            "total_equity": self.capital + position_value,
            "price": price,
            "rsi": rsi_value,
            "rsi_ma": rsi_ma_value,
            "in_position": self.position is not None and self.position.is_open,
        })

    def run(
        self,
        klines: list[dict],
        rsi: list[float | None],
        rsi_ma: list[float | None],
        trend_signals: list[str] = None,
        adx: list[float | None] = None,
        macd: list[float | None] = None,
        macd_signal: list[float | None] = None,
        bb_upper: list[float | None] = None,
        bb_lower: list[float | None] = None,
        volume_sma: list[float | None] = None,
        stoch_k: list[float | None] = None,
        stoch_d: list[float | None] = None,
        atr: list[float | None] = None,
        lows: list[float | None] = None,
        ema_regime: list[float | None] = None,
    ) -> dict:
        self.capital = self.initial_capital
        self.position = None
        self.stop_loss_price = 0.0
        self.trailing_stop_price = 0.0
        self.closed_trades = []
        self.equity_curve = []
        self.entry_bar = None

        prev_rsi_ma = None
        prev_trend = 'none'
        prev_macd = None
        prev_macd_sig = None
        highest_since_entry = 0.0
        last_exit_bar = -10**9

        for i in range(len(klines)):
            candle = klines[i]
            price = candle["close"]
            low_price = candle.get("low", price)
            high_price = candle.get("high", price)
            volume = candle.get("vol", 0)
            timestamp = candle["time"]

            current_rsi = rsi[i] if i < len(rsi) else None
            current_rsi_ma = rsi_ma[i] if i < len(rsi_ma) else None
            current_trend = trend_signals[i] if trend_signals and i < len(trend_signals) else 'none'
            current_adx = adx[i] if adx and i < len(adx) else None
            current_macd = macd[i] if macd and i < len(macd) else None
            current_macd_sig = macd_signal[i] if macd_signal and i < len(macd_signal) else None
            current_bb_upper = bb_upper[i] if bb_upper and i < len(bb_upper) else None
            current_bb_lower = bb_lower[i] if bb_lower and i < len(bb_lower) else None
            current_vol_sma = volume_sma[i] if volume_sma and i < len(volume_sma) else None
            current_stoch_k = stoch_k[i] if stoch_k and i < len(stoch_k) else None
            current_stoch_d = stoch_d[i] if stoch_d and i < len(stoch_d) else None
            current_atr = atr[i] if atr and i < len(atr) else None
            current_regime_ema = ema_regime[i] if ema_regime and i < len(ema_regime) else None
            prev_close = klines[i - 1]["close"] if i > 0 else None

            if self.position and self.position.is_open:
                self.position.record_price_update(price)

                if self.use_trailing_stop:
                    highest_since_entry = max(highest_since_entry, high_price)
                    new_trailing = highest_since_entry * (1 - self.trailing_stop_pct)
                    if new_trailing > self.trailing_stop_price:
                        self.trailing_stop_price = new_trailing

                effective_stop = max(self.stop_loss_price, self.trailing_stop_price)
                if effective_stop > 0 and low_price <= effective_stop:
                    reason = "trailing_stop" if self.trailing_stop_price >= self.stop_loss_price else "stop_loss"
                    self._sell_all(effective_stop, timestamp, reason)
                    self._append_equity(timestamp, price, current_rsi, current_rsi_ma)
                    last_exit_bar = i
                    highest_since_entry = 0.0
                    prev_rsi_ma = current_rsi_ma
                    prev_trend = current_trend
                    prev_macd = current_macd
                    prev_macd_sig = current_macd_sig
                    continue

            if self.position is None or not self.position.is_open:
                should_enter, entry_reason = self._should_enter(current_rsi_ma, prev_rsi_ma, current_trend, prev_trend)
                entry_allowed = should_enter and (i - last_exit_bar > self.cooldown_bars)

                if entry_allowed:
                    entry_allowed = self._entry_allowed(
                        i,
                        price,
                        volume,
                        current_adx,
                        current_macd,
                        current_macd_sig,
                        current_bb_lower,
                        current_vol_sma,
                        current_stoch_k,
                        current_atr,
                        current_regime_ema,
                        lows or [],
                        rsi,
                        prev_close,
                    )

                if entry_allowed:
                    self._buy(price, self.entry_amount, timestamp, entry_reason)
                    self.entry_bar = i
                    highest_since_entry = high_price
                    self.trailing_stop_price = price * (1 - self.trailing_stop_pct) if self.use_trailing_stop else 0.0
                    self.stop_loss_price = price - (current_atr * self.atr_multiplier) if self.use_atr_stop and current_atr is not None else 0.0

            if self.position and self.position.is_open:
                bars_since_entry = i - (self.entry_bar if self.entry_bar is not None else i)
                first_entry = self.position.first_entry_price
                exit_pct_done = self._exit_pct_sold()
                dca_allowed = self.max_dca_trades > 0 and self._dca_count() < self.max_dca_trades and exit_pct_done < self.dca_disable_after_exit_pct

                if dca_allowed and first_entry > 0:
                    drop_pct = (first_entry - price) / first_entry
                    if drop_pct >= self.dca_60_drop and not self.position.dca_60_done:
                        self._buy(price, self.dca_60_amount, timestamp, "dca_60")
                    if drop_pct >= self.dca_30_drop and not self.position.dca_30_done:
                        self._buy(price, self.dca_30_amount, timestamp, "dca_30")

            if self.position and self.position.is_open:
                bars_since_entry = i - (self.entry_bar if self.entry_bar is not None else i)

                if self.force_exit_after_bars > 0 and bars_since_entry >= self.force_exit_after_bars:
                    self._sell_all(price, timestamp, "force_exit")

                elif bars_since_entry >= self.min_hold_bars:
                    in_profit = price >= self.position.avg_price
                    can_stage_exit = not self.require_profit_for_staged_exit or in_profit

                    if can_stage_exit and self.strategy_mode in ("rsi", "combined") and current_rsi is not None:
                        if current_rsi > self.exit_threshold_3 and not self.position.exit_3_done:
                            self._sell(price, self.exit_pct_3, timestamp, "sell_70")
                        elif current_rsi > self.exit_threshold_2 and not self.position.exit_2_done:
                            self._sell(price, self.exit_pct_2, timestamp, "sell_50")
                        elif current_rsi > self.entry_threshold and not self.position.exit_1_done:
                            self._sell(price, self.exit_pct_1, timestamp, "sell_25")

                    elif can_stage_exit and self.strategy_mode == "trend":
                        if current_trend == 'sell' and prev_trend != 'sell':
                            self._sell_all(price, timestamp, "trend_exit")

                    if self.position and self.position.is_open and self.break_even_stop_after_exit_pct > 0:
                        if self._exit_pct_sold() >= self.break_even_stop_after_exit_pct:
                            if self.position.avg_price > 0 and self.stop_loss_price < self.position.avg_price:
                                self.stop_loss_price = self.position.avg_price

            if self.position and self.position.is_open:
                self.position.record_price_update(price)

            self._append_equity(timestamp, price, current_rsi, current_rsi_ma)

            prev_rsi_ma = current_rsi_ma
            prev_trend = current_trend
            prev_macd = current_macd
            prev_macd_sig = current_macd_sig

        if self.position and self.position.is_open:
            last_price = klines[-1]["close"] if klines else 0
            last_time = klines[-1]["time"] if klines else 0
            self._sell_all(last_price, last_time, "backtest_end")

        return self.get_results()

    def _annualized_sharpe(self, equities: list[float]) -> float:
        if len(equities) < 3:
            return 0.0
        returns = []
        for i in range(1, len(equities)):
            if equities[i - 1] > 0:
                returns.append(equities[i] / equities[i - 1] - 1)

        if len(returns) < 2:
            return 0.0

        avg_return = sum(returns) / len(returns)
        variance = sum((r - avg_return) ** 2 for r in returns) / len(returns)
        std = math.sqrt(variance) if variance > 0 else 0.0
        return avg_return / std * math.sqrt(365) if std > 0 else 0.0

    def _annualized_sortino(self, equities: list[float]) -> float:
        if len(equities) < 3:
            return 0.0
        returns = []
        for i in range(1, len(equities)):
            if equities[i - 1] > 0:
                returns.append(equities[i] / equities[i - 1] - 1)

        downside = [r for r in returns if r < 0]
        if len(downside) < 2:
            return 0.0

        avg_return = sum(returns) / len(returns)
        downside_variance = sum(r ** 2 for r in downside) / len(downside)
        downside_std = math.sqrt(downside_variance) if downside_variance > 0 else 0.0
        return avg_return / downside_std * math.sqrt(365) if downside_std > 0 else 0.0

    def _max_consecutive_losses(self) -> int:
        max_streak = 0
        current = 0
        for trade in self.closed_trades:
            if trade.get("pnl", 0) <= 0:
                current += 1
                max_streak = max(max_streak, current)
            else:
                current = 0
        return max_streak

    def get_results(self) -> dict:
        total_pnl = self.capital - self.initial_capital
        total_pnl_pct = self._safe_pct(total_pnl, self.initial_capital)

        winning_trades = [t for t in self.closed_trades if t.get("pnl", 0) > 0]
        losing_trades = [t for t in self.closed_trades if t.get("pnl", 0) <= 0]

        gross_profit = sum(max(t.get("pnl", 0), 0) for t in self.closed_trades)
        gross_loss = abs(sum(min(t.get("pnl", 0), 0) for t in self.closed_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)

        max_drawdown = 0.0
        peak = self.initial_capital
        for point in self.equity_curve:
            equity = point.get("total_equity", 0)
            if equity > peak:
                peak = equity
            drawdown = (peak - equity) / peak * 100 if peak > 0 else 0.0
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        equities = [p.get("total_equity", 0) for p in self.equity_curve if p.get("total_equity", 0) > 0]
        sharpe = self._annualized_sharpe(equities)
        sortino = self._annualized_sortino(equities)

        exposure_days = sum(1 for p in self.equity_curve if p.get("in_position"))
        durations = [t.get("duration_days", 0) for t in self.closed_trades]
        avg_duration = sum(durations) / len(durations) if durations else 0.0
        avg_trade_pnl = total_pnl / len(self.closed_trades) if self.closed_trades else 0.0
        expectancy = avg_trade_pnl / max(self.entry_amount, 1.0)
        max_consecutive_losses = self._max_consecutive_losses()
        total_dca_count = sum(t.get("dca_count", 0) for t in self.closed_trades)
        avg_dca_count = total_dca_count / len(self.closed_trades) if self.closed_trades else 0.0
        total_fees = sum(t.get("total_fees", 0) for t in self.closed_trades)
        win_rate = len(winning_trades) / len(self.closed_trades) * 100 if self.closed_trades else 0.0

        risk_adjusted_score = 0.0
        if max_drawdown >= 0:
            risk_adjusted_score = ((total_pnl_pct + 100) / (max_drawdown + 1)) * (win_rate / 100 + 0.2) * (min(profit_factor, 5) / 5)

        quality_score = total_pnl_pct - max_drawdown * 0.5 + win_rate * 0.15 + min(profit_factor, 5) * 2 - max_consecutive_losses * 3

        return {
            "initial_capital": self.initial_capital,
            "final_capital": self.capital,
            "total_pnl": total_pnl,
            "total_pnl_pct": total_pnl_pct,
            "total_trades": len(self.closed_trades),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": win_rate,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "profit_factor": profit_factor,
            "avg_trade_pnl": avg_trade_pnl,
            "expectancy": expectancy,
            "avg_duration_days": avg_duration,
            "exposure_days": exposure_days,
            "max_consecutive_losses": max_consecutive_losses,
            "total_dca_count": total_dca_count,
            "avg_dca_count": avg_dca_count,
            "total_fees": total_fees,
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
            "risk_adjusted_score": risk_adjusted_score,
            "quality_score": quality_score,
            "best_trade": max(self.closed_trades, key=lambda x: x.get("pnl", 0)) if self.closed_trades else None,
            "worst_trade": min(self.closed_trades, key=lambda x: x.get("pnl", 0)) if self.closed_trades else None,
            "closed_trades": self.closed_trades,
            "equity_curve": self.equity_curve,
        }
