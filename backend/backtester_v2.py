"""
MEXC ML Trading System — Backtester V2.0
==========================================
Gelismis Backtesting: Event-Driven, Monte Carlo, Walk-Forward Optimization

OZELLIKLER:
  1. Event-Driven Engine: Daha gercekci simülasyon
  2. Monte Carlo Simulation: Riske dayali analiz
  3. Walk-Forward Optimization: Parametre optimizasyonu
  4. Multi-Asset Portfolio: Portfoy seviyesi backtest
"""

import numpy as np
import logging
import time
from typing import Optional, Callable, Dict, List, Tuple
from dataclasses import dataclass, field
from collections import deque

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# EVENT-DRIVEN BACKTEST ENGINE
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Event:
    timestamp: int
    event_type: str  # "signal", "fill", "cancel", "timeout"
    data: Dict = field(default_factory=dict)


class EventDrivenEngine:
    """
    Event-driven backtest motoru.
    Daha gercekci emir simülasyonu.
    """

    def __init__(self, initial_capital: float = 10000.0,
                 fee_rate: float = 0.0006,
                 slippage_bps: float = 0.5):
        self.initial_capital = initial_capital
        self.fee_rate = fee_rate
        self.slippage_bps = slippage_bps

        self._capital = initial_capital
        self._positions = {}
        self._pending_orders = []
        self._trades = []
        self._equity_curve = [initial_capital]
        self._current_bar = 0

    def process_bar(self, bar_idx: int, prices: np.ndarray,
                    signals: List[Dict]) -> Dict:
        """
        Tek bar isle.

        signals: [{"action": "LONG"/"SHORT"/"CLOSE", "size": float, ...}]
        """
        self._current_bar = bar_idx

        # Pending order'lari isle
        self._process_pending_orders(prices, bar_idx)

        # Sinyalleri uygula
        for signal in signals:
            self._execute_signal(signal, prices, bar_idx)

        # Equity guncelle
        equity = self._calculate_equity(prices, bar_idx)
        self._equity_curve.append(equity)

        return {
            "bar": bar_idx,
            "equity": equity,
            "positions": len(self._positions),
            "capital": self._capital,
        }

    def _execute_signal(self, signal: Dict, prices: np.ndarray,
                        bar_idx: int):
        """Sinyali uygula."""
        action = signal.get("action", "")
        symbol = signal.get("symbol", "default")
        size = signal.get("size", 0)

        if action == "LONG" and symbol not in self._positions:
            # Alis emri
            price = float(prices[bar_idx])
            slippage = price * self.slippage_bps / 10000
            entry_price = price + slippage

            fee = size * self.fee_rate
            self._capital -= fee

            self._positions[symbol] = {
                "side": "LONG",
                "entry_price": entry_price,
                "size": size,
                "entry_bar": bar_idx,
                "sl": signal.get("sl", entry_price * 0.98),
                "tp": signal.get("tp", entry_price * 1.05),
            }

        elif action == "SHORT" and symbol not in self._positions:
            # Satis emri
            price = float(prices[bar_idx])
            slippage = price * self.slippage_bps / 10000
            entry_price = price - slippage

            fee = size * self.fee_rate
            self._capital -= fee

            self._positions[symbol] = {
                "side": "SHORT",
                "entry_price": entry_price,
                "size": size,
                "entry_bar": bar_idx,
                "sl": signal.get("sl", entry_price * 1.02),
                "tp": signal.get("tp", entry_price * 0.95),
            }

        elif action == "CLOSE" and symbol in self._positions:
            # Kapatma emri
            self._close_position(symbol, prices, bar_idx)

    def _close_position(self, symbol: str, prices: np.ndarray,
                        bar_idx: int):
        """Pozisyonu kapat."""
        pos = self._positions[symbol]
        price = float(prices[bar_idx])

        slippage = price * self.slippage_bps / 10000
        if pos["side"] == "LONG":
            exit_price = price - slippage
            pnl = (exit_price - pos["entry_price"]) / pos["entry_price"] * pos["size"]
        else:
            exit_price = price + slippage
            pnl = (pos["entry_price"] - exit_price) / pos["entry_price"] * pos["size"]

        fee = pos["size"] * self.fee_rate
        net_pnl = pnl - fee

        self._capital += net_pnl

        self._trades.append({
            "symbol": symbol,
            "side": pos["side"],
            "entry_price": pos["entry_price"],
            "exit_price": exit_price,
            "pnl": net_pnl,
            "entry_bar": pos["entry_bar"],
            "exit_bar": bar_idx,
            "duration": bar_idx - pos["entry_bar"],
        })

        del self._positions[symbol]

    def _process_pending_orders(self, prices: np.ndarray, bar_idx: int):
        """Pending order'lari isle."""
        remaining = []
        for order in self._pending_orders:
            symbol = order["symbol"]
            order_type = order["type"]
            target_price = order["target_price"]
            current_price = float(prices[bar_idx])

            filled = False
            if order_type == "stop_loss":
                if order["side"] == "LONG" and current_price <= target_price:
                    self._close_position(symbol, prices, bar_idx)
                    filled = True
                elif order["side"] == "SHORT" and current_price >= target_price:
                    self._close_position(symbol, prices, bar_idx)
                    filled = True
            elif order_type == "take_profit":
                if order["side"] == "LONG" and current_price >= target_price:
                    self._close_position(symbol, prices, bar_idx)
                    filled = True
                elif order["side"] == "SHORT" and current_price <= target_price:
                    self._close_position(symbol, prices, bar_idx)
                    filled = True

            if not filled:
                remaining.append(order)

        self._pending_orders = remaining

    def _calculate_equity(self, prices: np.ndarray, bar_idx: int) -> float:
        """Anlik equity hesapla."""
        equity = self._capital

        for symbol, pos in self._positions.items():
            price = float(prices[bar_idx])
            if pos["side"] == "LONG":
                unrealized = (price - pos["entry_price"]) / pos["entry_price"] * pos["size"]
            else:
                unrealized = (pos["entry_price"] - price) / pos["entry_price"] * pos["size"]
            equity += unrealized

        return equity

    def get_results(self) -> Dict:
        """Sonuclari dondur."""
        if not self._equity_curve:
            return {}

        equity_arr = np.array(self._equity_curve)
        returns = np.diff(equity_arr) / (equity_arr[:-1] + 1e-10)

        # Metrikler
        total_return = (equity_arr[-1] / equity_arr[0] - 1) * 100

        wins = [t for t in self._trades if t["pnl"] > 0]
        losses = [t for t in self._trades if t["pnl"] <= 0]
        win_rate = len(wins) / len(self._trades) * 100 if self._trades else 0

        avg_win = np.mean([t["pnl"] for t in wins]) if wins else 0
        avg_loss = np.mean([abs(t["pnl"]) for t in losses]) if losses else 0

        # Max drawdown
        peaks = np.maximum.accumulate(equity_arr)
        dd = (peaks - equity_arr) / (peaks + 1e-10)
        max_dd = float(np.max(dd)) * 100

        # Sharpe
        if len(returns) > 3 and np.std(returns) > 1e-10:
            sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(252))
        else:
            sharpe = 0.0

        # Sortino
        downside = returns[returns < 0]
        if len(downside) > 2 and np.std(downside) > 1e-10:
            sortino = float(np.mean(returns) / np.std(downside) * np.sqrt(252))
        else:
            sortino = 0.0

        # Profit factor
        gross_win = sum(t["pnl"] for t in wins)
        gross_loss = sum(abs(t["pnl"]) for t in losses)
        profit_factor = gross_win / gross_loss if gross_loss > 0 else 0

        return {
            "total_return_pct": round(total_return, 2),
            "total_trades": len(self._trades),
            "win_rate_pct": round(win_rate, 1),
            "avg_win": round(float(avg_win), 4),
            "avg_loss": round(float(avg_loss), 4),
            "profit_factor": round(profit_factor, 3),
            "max_drawdown_pct": round(max_dd, 2),
            "sharpe": round(sharpe, 3),
            "sortino": round(sortino, 3),
            "final_equity": round(float(equity_arr[-1]), 2),
            "trades": self._trades,
            "equity_curve": [round(float(v), 4) for v in self._equity_curve],
        }


# ══════════════════════════════════════════════════════════════════════════════
# MONTE CARLO SIMULATION
# ══════════════════════════════════════════════════════════════════════════════

class MonteCarloSimulator:
    """
    Monte Carlo simülasyonu:
    - Rassal yeniden ornekleme
    - Riske dayali analiz
    - Olasilik dagilimi
    """

    def __init__(self, n_simulations: int = 1000,
                 confidence_level: float = 0.95):
        self.n_simulations = n_simulations
        self.confidence_level = confidence_level

    def simulate(self, trades: List[Dict], n_periods: int = 100) -> Dict:
        """
        Monte Carlo simülasyonu calistir.

        trades: Gecmis trade'lerin listesi
        n_periods: Simüle edilecek donem sayisi

        Donus: {
            "mean_return": float,
            "var_95": float,
            "worst_case": float,
            "best_case": float,
            "probability_of_profit": float,
            "confidence_interval": (lower, upper),
        }
        """
        if not trades:
            return {}

        # Trade getirilerini al
        returns = np.array([t.get("pnl", 0) for t in trades])
        if len(returns) < 10:
            return {}

        # Simülasyonlar
        simulated_equity = np.zeros(self.n_simulations)
        simulated_min = np.full(self.n_simulations, np.inf)
        simulated_max = np.full(self.n_simulations, -np.inf)

        for i in range(self.n_simulations):
            # Rassal yeniden ornekleme
            sampled_returns = np.random.choice(returns, size=n_periods, replace=True)
            equity = np.cumprod(1 + sampled_returns)
            simulated_equity[i] = equity[-1]
            simulated_min[i] = np.min(equity)
            simulated_max[i] = np.max(equity)

        # Istatistikler
        mean_return = float(np.mean(simulated_equity) - 1) * 100
        var_95 = float(np.percentile(simulated_equity, 5) - 1) * 100
        worst_case = float(np.min(simulated_equity) - 1) * 100
        best_case = float(np.max(simulated_equity) - 1) * 100
        prob_profit = float(np.mean(simulated_equity > 1)) * 100

        # Confidence interval
        lower = float(np.percentile(simulated_equity, (1 - self.confidence_level) / 2 * 100))
        upper = float(np.percentile(simulated_equity, (1 + self.confidence_level) / 2 * 100))

        return {
            "mean_return_pct": round(mean_return, 2),
            "var_95_pct": round(var_95, 2),
            "worst_case_pct": round(worst_case, 2),
            "best_case_pct": round(best_case, 2),
            "probability_of_profit_pct": round(prob_profit, 1),
            "confidence_interval": (round(lower, 4), round(upper, 4)),
            "n_simulations": self.n_simulations,
        }


# ══════════════════════════════════════════════════════════════════════════════
# WALK-FORWARD OPTIMIZER
# ══════════════════════════════════════════════════════════════════════════════

class WalkForwardOptimizer:
    """
    Walk-forward optimizasyon:
    - Parametre arama
    - Overfitting onleme
    - Out-of-sample test
    """

    def __init__(self, n_splits: int = 5,
                 gap_bars: int = 10):
        self.n_splits = n_splits
        self.gap_bars = gap_bars

    def optimize(self, data: np.ndarray, param_grid: Dict,
                 objective_fn: Callable) -> Dict:
        """
        Walk-forward optimizasyon calistir.

        data: Fiyat verisi
        param_grid: Parametre araliklari {"sl": [0.01, 0.02, 0.03], ...}
        objective_fn: Optimizasyon amacli fonksiyon

        Donus: {
            "best_params": {...},
            "oos_performance": float,
            "stability": float,
        }
        """
        n = len(data)
        fold_size = n // (self.n_splits + 1)

        all_params = []
        all_performances = []

        for i in range(self.n_splits):
            # Train/Test bolme
            train_end = (i + 1) * fold_size
            test_start = train_end + self.gap_bars
            test_end = min(test_start + fold_size, n)

            if test_end - test_start < 10:
                continue

            train_data = data[:train_end]
            test_data = data[test_start:test_end]

            # Parametre arama (train uzerinde)
            best_params = None
            best_score = -np.inf

            for params in self._generate_param_combinations(param_grid):
                score = objective_fn(train_data, params)
                if score > best_score:
                    best_score = score
                    best_params = params

            # Test uzerinde degerlendirme
            if best_params:
                oos_score = objective_fn(test_data, best_params)
                all_params.append(best_params)
                all_performances.append(oos_score)

        if not all_params:
            return {"best_params": {}, "oos_performance": 0, "stability": 0}

        # En iyi parametreleri sec (ortalama performansa gore)
        avg_performances = {}
        for params, perf in zip(all_params, all_performances):
            key = str(params)
            if key not in avg_performances:
                avg_performances[key] = []
            avg_performances[key].append(perf)

        best_key = max(avg_performances, key=lambda k: np.mean(avg_performances[k]))
        best_params = eval(best_key)

        oos_performance = float(np.mean(all_performances))
        stability = float(1 - np.std(all_performances) / (abs(oos_performance) + 1e-10))

        return {
            "best_params": best_params,
            "oos_performance": round(oos_performance, 4),
            "stability": round(stability, 4),
            "n_folds": len(all_performances),
        }

    def _generate_param_combinations(self, param_grid: Dict) -> List[Dict]:
        """Parametre kombinasyonlari uret."""
        import itertools
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combinations = list(itertools.product(*values))
        return [dict(zip(keys, combo)) for combo in combinations]


# ══════════════════════════════════════════════════════════════════════════════
# ANA BACKTESTER V2
# ══════════════════════════════════════════════════════════════════════════════

class BacktesterV2:
    """
    Backtester V2 - Ana sinif:
    - Event-Driven engine
    - Monte Carlo simulation
    - Walk-Forward optimization
    """

    def __init__(self, initial_capital: float = 10000.0,
                 fee_rate: float = 0.0006,
                 slippage_bps: float = 0.5):
        self.engine = EventDrivenEngine(initial_capital, fee_rate, slippage_bps)
        self.monte_carlo = MonteCarloSimulator()
        self.wf_optimizer = WalkForwardOptimizer()

    def run_backtest(self, prices: np.ndarray,
                     signal_fn: Callable,
                     sl_pct: float = 0.02,
                     tp_pct: float = 0.05) -> Dict:
        """
        Backtest calistir.

        prices: Fiyat serisi
        signal_fn: Sinyal uretici fonksiyon
        sl_pct: Stop loss yuzdesi
        tp_pct: Take profit yuzdesi
        """
        self.engine = EventDrivenEngine(
            initial_capital=self.engine.initial_capital,
            fee_rate=self.engine.fee_rate,
            slippage_bps=self.engine.slippage_bps
        )

        for i in range(1, len(prices)):
            # Sinyal uret
            signals = signal_fn(prices[:i], i)

            # Bar isle
            self.engine.process_bar(i, prices, signals)

        return self.engine.get_results()

    def run_monte_carlo(self, trades: List[Dict],
                        n_simulations: int = 1000) -> Dict:
        """Monte Carlo simülasyonu calistir."""
        return self.monte_carlo.simulate(trades, n_periods=100)

    def run_optimization(self, prices: np.ndarray,
                         param_grid: Dict,
                         objective_fn: Callable) -> Dict:
        """Walk-forward optimizasyon calistir."""
        return self.wf_optimizer.optimize(prices, param_grid, objective_fn)

    def get_comprehensive_report(self, backtest_result: Dict,
                                  monte_carlo_result: Dict = None,
                                  optimization_result: Dict = None) -> Dict:
        """Kapsamli rapor olustur."""
        report = {
            "backtest": backtest_result,
        }

        if monte_carlo_result:
            report["monte_carlo"] = monte_carlo_result

        if optimization_result:
            report["optimization"] = optimization_result

        # Ozet
        report["summary"] = {
            "total_return": backtest_result.get("total_return_pct", 0),
            "win_rate": backtest_result.get("win_rate_pct", 0),
            "max_drawdown": backtest_result.get("max_drawdown_pct", 0),
            "sharpe": backtest_result.get("sharpe", 0),
            "profit_factor": backtest_result.get("profit_factor", 0),
        }

        if monte_carlo_result:
            report["summary"]["probability_of_profit"] = monte_carlo_result.get("probability_of_profit_pct", 0)

        return report


# ══════════════════════════════════════════════════════════════════════════════
# EXPORT
# ══════════════════════════════════════════════════════════════════════════════

__all__ = [
    "BacktesterV2",
    "EventDrivenEngine",
    "MonteCarloSimulator",
    "WalkForwardOptimizer",
]
