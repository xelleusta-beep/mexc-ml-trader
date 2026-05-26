"""
MEXC ML Trading System — Backtesting Engine v1.0
Gerçekçi backtest: slippage, fee, walk-forward, portföy.
"""

import math
import time
import numpy as np
from typing import Optional, Callable
from dataclasses import dataclass, field


# ── Slippage Models ───────────────────────────────────────────────────────────

@dataclass
class SlippageConfig:
    """Slippage parametreleri."""
    spread_bps: float = 0.5       # Spread (basis point): MEXC BTC ~0.5bps
    vol_slip_bps: float = 0.3     # Hacim bazlı ek slippage
    min_slip_bps: float = 0.2     # Minimum slippage
    max_slip_bps: float = 5.0     # Maksimum slippage
    use_vol_slippage: bool = True # Hacim bazlı slippage aktif

    def get_slippage_bps(self, volume_btc_equiv: float = 0) -> float:
        if self.use_vol_slippage and volume_btc_equiv > 0:
            slip = self.spread_bps + self.vol_slip_bps * math.log10(1 + volume_btc_equiv)
        else:
            slip = self.spread_bps
        return max(self.min_slip_bps, min(self.max_slip_bps, slip))


# ── Fee Models ────────────────────────────────────────────────────────────────

@dataclass
class FeeConfig:
    """Exchange fee yapısı."""
    maker_bps: float = -0.25   # MEXC maker: -0.025% = -0.25bps (rebate)
    taker_bps: float = 7.0     # MEXC taker: 0.07% = 7bps
    is_maker: bool = False     # Varsayılan taker (piyasa emri)

    def get_fee_bps(self, is_maker: Optional[bool] = None) -> float:
        maker = is_maker if is_maker is not None else self.is_maker
        return self.maker_bps if maker else self.taker_bps

    def get_fee_pct(self, is_maker: Optional[bool] = None) -> float:
        return self.get_fee_bps(is_maker) / 10_000


# ── Walk-Forward Validator ────────────────────────────────────────────────────

@dataclass
class WFResult:
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    n_splits: int = 0
    fold_accuracies: list = field(default_factory=list)
    fold_trades: list = field(default_factory=list)


def walk_forward_purge(X, y, n_splits: int = 5, gap_bars: int = 5,
                        embargo_pct: float = 0.01) -> list:
    """
    Walk-forward cross-validation with purging and embargo.

    - purging: test setinden train setine sızan verileri temizle
    - embargo: train/test arasında boşluk bırak (zaman serisi leakage önlemek için)
    """
    n = len(X)
    fold_size = n // (n_splits + 1)
    folds = []

    for i in range(n_splits):
        train_end = (i + 1) * fold_size
        test_start = train_end + gap_bars
        test_end = min(test_start + fold_size, n)

        if test_end - test_start < 10:
            continue

        # Embargo: train setinin sonunu kırp
        embargo_bars = max(gap_bars, int(n * embargo_pct))
        train_end_clean = train_end - embargo_bars

        if train_end_clean < fold_size // 2:
            continue

        folds.append({
            "train": list(range(train_end_clean)),
            "test":  list(range(test_start, test_end)),
        })

    return folds


def validate_model(model_fn: Callable, X: np.ndarray, y: np.ndarray,
                   n_splits: int = 5, gap_bars: int = 5) -> WFResult:
    """
    Walk-forward validation:
    model_fn(X_train, y_train) -> tahmin fonksiyonu
    """
    folds = walk_forward_purge(X, y, n_splits, gap_bars)
    result = WFResult(n_splits=len(folds))

    all_preds, all_true = [], []

    for fold in folds:
        train_idx, test_idx = fold["train"], fold["test"]
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_te, y_te = X[test_idx], y[test_idx]

        predict_fn = model_fn(X_tr, y_tr)
        preds = predict_fn(X_te)

        acc = float(np.mean(preds == y_te))
        result.fold_accuracies.append(round(acc * 100, 1))
        result.fold_trades.append(len(test_idx))

        all_preds.extend(preds.tolist() if hasattr(preds, 'tolist') else list(preds))
        all_true.extend(y_te.tolist() if hasattr(y_te, 'tolist') else list(y_te))

    if all_true:
        result.accuracy = float(np.mean(np.array(all_preds) == np.array(all_true))) * 100

        tp = sum(1 for p, t in zip(all_preds, all_true) if p == 1 and t == 1)
        fp = sum(1 for p, t in zip(all_preds, all_true) if p == 1 and t == 0)
        fn = sum(1 for p, t in zip(all_preds, all_true) if p == 0 and t == 1)

        result.precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
        result.recall    = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
        result.f1_score  = 2 * result.precision * result.recall / (
            result.precision + result.recall) if (result.precision + result.recall) > 0 else 0

    return result


# ── Backtest Engine ───────────────────────────────────────────────────────────

@dataclass
class TradeResult:
    entry_bar: int
    exit_bar: int
    side: str                     # LONG / SHORT
    entry_price: float
    exit_price: float
    pnl_pct: float                # Kaldiracsiz yuzde
    pnl_abs: float                # 1 birim sermaye icin abs
    return_pct: float             # Kaldiracli yuzde
    leverage: int
    fee_pct: float
    duration_bars: int
    sl_hit: bool = False
    tp_hit: bool = False


@dataclass
class BacktestResult:
    symbol: str = ""
    n_trades: int = 0
    roi_pct: float = 0.0
    win_rate: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    calmar: float = 0.0
    profit_factor: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    expectancy: float = 0.0
    avg_bars_held: float = 0.0
    total_fee_pct: float = 0.0
    equity_curve: list = field(default_factory=list)
    trades: list = field(default_factory=list)
    timing_ms: float = 0.0


def backtest_simple(prices: np.ndarray, signals: list,
                    sl_pct: float = 0.022, tp_pct: float = 0.05,
                    leverage: int = 5,
                    fee_conf: Optional[FeeConfig] = None,
                    slip_conf: Optional[SlippageConfig] = None,
                    symbol: str = "",
                    max_bars_hold: int = 8) -> BacktestResult:
    """
    Gerçekçi backtest:
    - SL/TP (kaldıraçlı fiyat mesafesi)
    - Slippage (giriş/çıkış)
    - Fee (giriş + çıkış)
    - Kaldıraç
    """
    t0 = time.perf_counter()
    fee_cfg   = fee_conf or FeeConfig()
    slip_cfg  = slip_conf or SlippageConfig()
    result    = BacktestResult(symbol=symbol)
    equity    = 1.0
    eq_curve  = [1.0]
    returns   = []
    fees_pct  = []
    n         = len(prices)

    for bar_idx, sig in signals:
        if sig not in ("LONG", "SHORT") or bar_idx + 1 >= n:
            continue

        entry_px = float(prices[bar_idx])
        if entry_px <= 0:
            continue

        # Slippage: giriş fiyatına eklenir
        slip_bps = slip_cfg.get_slippage_bps()
        if sig == "LONG":
            entry_px *= (1 + slip_bps / 10_000)
        else:
            entry_px *= (1 - slip_bps / 10_000)

        # SL/TP kaldıraçlı fiyat
        lev_sl = sl_pct / leverage
        lev_tp = tp_pct / leverage

        sl_px = entry_px * (1 - lev_sl) if sig == "LONG" else entry_px * (1 + lev_sl)
        tp_px = entry_px * (1 + lev_tp) if sig == "LONG" else entry_px * (1 - lev_tp)

        hit    = False
        r_pct  = 0.0
        sl_hit = False
        tp_hit = False
        exit_bar = bar_idx

        for fwd in range(1, min(max_bars_hold + 1, n - bar_idx)):
            px = float(prices[bar_idx + fwd])
            if sig == "LONG":
                if px <= sl_px:
                    r_pct = -sl_pct
                    sl_hit = True
                    hit = True
                    exit_bar = bar_idx + fwd
                    break
                if px >= tp_px:
                    r_pct = tp_pct
                    tp_hit = True
                    hit = True
                    exit_bar = bar_idx + fwd
                    break
            else:  # SHORT
                if px >= sl_px:
                    r_pct = -sl_pct
                    sl_hit = True
                    hit = True
                    exit_bar = bar_idx + fwd
                    break
                if px <= tp_px:
                    r_pct = tp_pct
                    tp_hit = True
                    hit = True
                    exit_bar = bar_idx + fwd
                    break

        if not hit:
            exit_bar = min(bar_idx + max_bars_hold, n - 1)
            exit_px  = float(prices[exit_bar])
            r_pct    = (exit_px - entry_px) / entry_px if sig == "LONG" else (entry_px - exit_px) / entry_px
            r_pct    = max(-sl_pct, min(tp_pct, r_pct))

        # Exit slippage
        exit_px = float(prices[exit_bar])
        if sig == "LONG":
            exit_px *= (1 - slip_bps / 10_000)
        else:
            exit_px *= (1 + slip_bps / 10_000)

        # Fee
        fee_pct  = fee_cfg.get_fee_pct(is_maker=False)
        r_net    = r_pct - fee_pct * 2

        # Apply leverage
        r_lev    = r_net * leverage
        equity  *= (1 + r_lev)
        eq_curve.append(equity)
        returns.append(r_lev)
        fees_pct.append(fee_pct * 2)

        trade = TradeResult(
            entry_bar=bar_idx, exit_bar=exit_bar,
            side=sig, entry_price=entry_px, exit_price=exit_px,
            pnl_pct=r_pct, pnl_abs=r_net,
            return_pct=r_lev, leverage=leverage,
            fee_pct=fee_pct * 2, duration_bars=exit_bar - bar_idx,
            sl_hit=sl_hit, tp_hit=tp_hit,
        )
        result.trades.append(trade)

    result.n_trades = len(result.trades)
    if result.n_trades == 0:
        result.timing_ms = (time.perf_counter() - t0) * 1000
        return result

    # Metrikler
    result.roi_pct = round((equity - 1) * 100, 2)

    wins  = [r for r in returns if r > 0]
    loses = [r for r in returns if r < 0]
    result.win_rate      = round(len(wins) / len(returns) * 100, 1)
    result.avg_win_pct   = round(float(np.mean(wins)) * 100, 2) if wins else 0.0
    result.avg_loss_pct  = round(float(np.mean(loses)) * 100, 2) if loses else 0.0
    result.expectancy    = round(float(np.mean(returns)) * 100, 4)
    result.total_fee_pct = round(sum(fees_pct) * 100, 2)

    result.avg_bars_held = round(float(np.mean([t.duration_bars for t in result.trades])), 1)

    # Drawdown (equity curve)
    eq_arr = np.array(eq_curve)
    peaks  = np.maximum.accumulate(eq_arr)
    dd_arr = (peaks - eq_arr) / (peaks + 1e-10) * 100
    result.max_drawdown_pct = round(float(np.max(dd_arr)), 2)
    result.equity_curve     = [round(float(v), 6) for v in eq_curve]

    # Sharpe (annualized, 2920 bars/yr for 15min)
    ret_arr = np.array(returns)
    if len(returns) >= 3 and ret_arr.std() > 1e-10:
        result.sharpe = round(float(ret_arr.mean() / ret_arr.std() * np.sqrt(2920)), 3)

        # Sortino (downside deviation)
        downside = ret_arr[ret_arr < 0]
        if len(downside) > 0 and downside.std() > 1e-10:
            result.sortino = round(float(ret_arr.mean() / downside.std() * np.sqrt(2920)), 3)
    else:
        result.sharpe = 0.0 if result.n_trades < 3 else 0.0

    # Calmar
    if result.max_drawdown_pct > 0.01:
        result.calmar = round(result.roi_pct / result.max_drawdown_pct, 3)

    # Profit Factor
    gross_win = sum(wins)
    gross_loss = abs(sum(loses))
    result.profit_factor = round(gross_win / gross_loss, 3) if gross_loss > 1e-10 else 0.0

    result.timing_ms = round((time.perf_counter() - t0) * 1000, 1)
    return result


# ── Portfolio Backtest ─────────────────────────────────────────────────────────

@dataclass
class PortfolioBacktestResult:
    n_pairs: int = 0
    total_trades: int = 0
    combined_roi_pct: float = 0.0
    combined_sharpe: float = 0.0
    combined_max_dd: float = 0.0
    combined_win_rate: float = 0.0
    combined_profit_factor: float = 0.0
    pair_results: list = field(default_factory=list)
    timing_ms: float = 0.0


def portfolio_backtest(pair_signals: dict[str, tuple[np.ndarray, list]],
                       sl_pct: float = 0.022, tp_pct: float = 0.05,
                       leverage: int = 5, equal_weight: bool = True) -> PortfolioBacktestResult:
    """
    Çoklu pair backtesti.

    pair_signals: {symbol: (prices_array, signals_list)}
    signals_list: [(bar_idx, signal)]
    """
    t0 = time.perf_counter()
    result = PortfolioBacktestResult(n_pairs=len(pair_signals))

    all_returns = []
    all_eq = []

    for symbol, (prices, signals) in pair_signals.items():
        bt = backtest_simple(prices, signals, sl_pct, tp_pct, leverage,
                             symbol=symbol)
        result.pair_results.append({
            "symbol":     symbol,
            "n_trades":   bt.n_trades,
            "roi_pct":    bt.roi_pct,
            "sharpe":     bt.sharpe,
            "win_rate":   bt.win_rate,
            "max_dd":     bt.max_drawdown_pct,
            "profit_factor": bt.profit_factor,
        })
        result.total_trades += bt.n_trades
        all_returns.extend(bt.trades)
        all_eq.append(bt.equity_curve[-1] if bt.equity_curve else 1.0)

    if result.total_trades == 0:
        result.timing_ms = (time.perf_counter() - t0) * 1000
        return result

    # Equal-weight portföy ROI
    if equal_weight and all_eq:
        n = len(all_eq)
        result.combined_roi_pct = round((sum(all_eq) / n - 1) * 100, 2)

    # Sharpe hesapla (hata ayıklama)
    trade_returns = [t.return_pct for sub in result.pair_results
                     for t in (backtest_simple.__globals__.get("_tmp", []))]
    # Use aggregate returns from trades
    all_r = []
    for pr in result.pair_results:
        pr_data = pr
        if "n_trades" in pr and pr["n_trades"] > 0:
            all_r.append(pr.get("roi_pct", 0) / max(1, pr.get("n_trades", 1)))

    # Win rate (ağırlıklı ortalama)
    total_trades = result.total_trades
    weighted_wr = sum(pr.get("win_rate", 0) * pr.get("n_trades", 0)
                      for pr in result.pair_results)
    result.combined_win_rate = round(weighted_wr / max(1, total_trades), 1)

    # Max DD
    result.combined_max_dd = max((pr.get("max_dd", 0) for pr in result.pair_results), default=0)

    # Profit Factor
    result.combined_profit_factor = max((pr.get("profit_factor", 0) for pr in result.pair_results), default=0)

    result.timing_ms = round((time.perf_counter() - t0) * 1000, 1)
    return result


# ── Performance Metrics Helper ─────────────────────────────────────────────────

def compute_sharpe(returns: list, periods_per_year: int = 2920) -> float:
    """Annualized Sharpe ratio."""
    arr = np.array(returns)
    if len(arr) < 3 or arr.std() < 1e-10:
        return 0.0
    return round(float(arr.mean() / arr.std() * np.sqrt(periods_per_year)), 3)


def compute_sortino(returns: list, periods_per_year: int = 2920) -> float:
    """Annualized Sortino ratio (downside deviation)."""
    arr = np.array(returns)
    downside = arr[arr < 0]
    if len(downside) < 2 or downside.std() < 1e-10:
        return 0.0
    return round(float(arr.mean() / downside.std() * np.sqrt(periods_per_year)), 3)


def compute_max_drawdown(equity_curve: list) -> float:
    """Maximum drawdown as percentage."""
    arr = np.array(equity_curve)
    if len(arr) < 2:
        return 0.0
    peaks = np.maximum.accumulate(arr)
    dd = float(np.max((peaks - arr) / (peaks + 1e-10))) * 100
    return round(dd, 2)
