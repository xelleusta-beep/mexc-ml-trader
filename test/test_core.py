"""
Birim testleri — risk_manager, backtester, persistence, cache
"""

import sys, os, json, tempfile, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from risk_manager import RiskManager
from backtester import backtest_simple, FeeConfig, SlippageConfig
from persistence import StateStore


class MockConfig:
    class Risk:
        max_drawdown = 0.15
        daily_loss_limit = 0.05
        kelly_fraction = 0.25
        max_concentration_per_pair = 0.2
        max_concentration_per_group = 0.35
        circuit_breaker_losses = 3
        circuit_breaker_cooldown = 300
    class Trading:
        base_position_size = 10.0
        min_price = 1e-8
        max_open_positions = 5
    risk = Risk()
    trading = Trading()


def test_risk_manager_initial():
    rm = RiskManager(MockConfig())
    dd, cur, lim = rm.is_drawdown_exceeded()
    assert dd is False
    sz = rm.calculate_position_size("BTC_USDT", 50000, 0.7, {"capital": 10000}, {})
    assert sz > 0
    assert isinstance(sz, float)


def test_risk_drawdown():
    rm = RiskManager(MockConfig())
    rm.update_capital(10000)
    rm.update_capital(9000)
    dd, cur, lim = rm.is_drawdown_exceeded()
    assert cur > 0
    rm.update_capital(11000)
    dd2, cur2, _ = rm.is_drawdown_exceeded()
    assert cur2 < cur  # peak reset


def test_backtest_simple_long():
    prices = np.array([100.0, 102.0, 101.0, 103.0, 105.0, 104.0, 106.0, 108.0])
    signals = [(0, "LONG"), (4, "LONG")]
    bt = backtest_simple(prices, signals, sl_pct=0.03, tp_pct=0.05, leverage=3)
    assert bt.n_trades == 2
    assert bt.roi_pct != 0
    assert bt.equity_curve is not None
    assert len(bt.equity_curve) > 0


def test_backtest_simple_short():
    prices = np.array([100.0, 98.0, 99.0, 97.0, 96.0, 95.0, 97.0, 99.0])
    signals = [(0, "SHORT")]
    bt = backtest_simple(prices, signals, sl_pct=0.02, tp_pct=0.04, leverage=2)
    assert bt.n_trades == 1
    assert bt.sharpe is not None


def test_backtest_empty():
    bt = backtest_simple(np.array([100.0]), [])
    assert bt.n_trades == 0
    assert bt.roi_pct == 0


def test_state_store():
    with tempfile.TemporaryDirectory() as tmp:
        store = StateStore(tmp)
        store.save_portfolio({"capital": 50000})
        loaded = store.load_portfolio()
        assert loaded["capital"] == 50000

        store.save_trade_history(["trade1", "trade2"])
        hist = store.load_trade_history()
        assert len(hist) == 2

        store.save_active_trades({"BTC_USDT": {"side": "LONG"}})
        active = store.load_active_trades()
        assert "BTC_USDT" in active

        store.save_pnl_timeline([{"t": "2024-01-01", "v": 100}])
        tl = store.load_pnl_timeline()
        assert len(tl) == 1

        assert "file_sizes" in store.stats
        assert store.stats["total_bytes"] > 0


def test_state_store_defaults():
    with tempfile.TemporaryDirectory() as tmp:
        store = StateStore(tmp)
        assert store.load_portfolio()["capital"] == 100000.0
        assert len(store.load_trade_history()) == 0
        assert len(store.load_active_trades()) == 0


def test_backtest_with_slippage():
    prices = np.array([100.0, 101.0, 102.0])
    signals = [(0, "LONG")]
    fee = FeeConfig(maker_bps=5, taker_bps=10)
    slip = SlippageConfig(spread_bps=2, vol_slip_bps=1)
    bt = backtest_simple(prices, signals, sl_pct=0.03, tp_pct=0.04,
                         leverage=5, fee_conf=fee, slip_conf=slip)
    assert bt.n_trades == 1
    assert bt.total_fee_pct > 0
    # Entry should have been adjusted by slippage
    t = bt.trades[0]
    assert t.entry_price != 100.0  # slippage applied
