from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
import asyncio
import json
import datetime
import threading
import os
from pathlib import Path

from mexc_client import get_all_futures_symbols, get_klines
from indicators import calculate_indicators, calculate_trend_signal, calculate_adx, calculate_atr, calculate_macd, calculate_bollinger_bands, calculate_volume_sma, calculate_stochastic_rsi, calculate_ema
from backtest_engine import BacktestEngine
from orchestrator import Orchestrator

app = FastAPI(title="MEXC Multi-Agent Trading System")

orchestrator = Orchestrator()
orchestrator_task = None

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class StrategyParams(BaseModel):
    rsi_period: int = 21
    rsi_ma_period: int = 21
    entry_threshold: float = 30.0
    exit_threshold_2: float = 50.0
    exit_threshold_3: float = 70.0
    exit_pct_1: float = 25.0
    exit_pct_2: float = 25.0
    exit_pct_3: float = 50.0
    entry_amount: float = 50.0
    dca_30_amount: float = 150.0
    dca_60_amount: float = 300.0
    dca_30_drop: float = 0.30
    dca_60_drop: float = 0.60
    initial_capital: float = 100.0
    maker_fee: float = 0.0002
    taker_fee: float = 0.0006
    strategy_mode: str = "rsi"
    ema_fast_period: int = 20
    ema_slow_period: int = 50
    cooldown_bars: int = 1
    max_dca_trades: int = 2
    dca_disable_after_exit_pct: float = 50.0
    require_profit_for_staged_exit: bool = False
    min_hold_bars: int = 0
    force_exit_after_bars: int = 0
    break_even_stop_after_exit_pct: float = 50.0
    use_adx_filter: bool = False
    adx_threshold: float = 25.0
    use_bb_filter: bool = False
    use_macd_filter: bool = False
    use_volume_filter: bool = False
    use_stochrsi_filter: bool = False
    use_volatility_filter: bool = False
    max_atr_pct: float = 0.12
    use_regime_filter: bool = False
    ema_regime_period: int = 200
    use_divergence_filter: bool = False
    divergence_lookback: int = 20
    use_breakout_confirmation: bool = False
    use_trailing_stop: bool = False
    trailing_stop_pct: float = 0.05
    use_atr_stop: bool = False
    atr_multiplier: float = 2.0


class BacktestRequest(StrategyParams):
    symbol: str
    timeframe: str = "1D"


class MultiBacktestRequest(StrategyParams):
    symbols: list[str]
    timeframe: str = "1D"


TIMEFRAME_MAP = {
    '5m': 'Min5', '15m': 'Min15', '30m': 'Min30',
    '1h': 'Min60', '4h': 'Hour4', '8h': 'Hour8', '1D': 'Day1',
}


class OptimizeRequest(BaseModel):
    symbols: list[str]
    timeframes: list[str] = ["1D"]
    initial_capital: float = 100.0
    entry_amount: float = 50.0
    dca_30_amount: float = 150.0
    dca_60_amount: float = 300.0
    maker_fee: float = 0.0002
    taker_fee: float = 0.0006


class PortfolioRequest(StrategyParams):
    symbols: list[str]
    initial_capital: float = 20000.0
    max_positions: int = 2
    timeframe: str = "1D"


def _timeframe_value(timeframe: str) -> str:
    return TIMEFRAME_MAP.get(timeframe, "Day1")


def _prepare_arrays(klines: list[dict], req: StrategyParams) -> dict:
    close_prices = [float(k["close"]) for k in klines]
    high_prices = [float(k.get("high", k["close"])) for k in klines]
    low_prices = [float(k.get("low", k["close"])) for k in klines]
    volumes = [float(k.get("vol", 0)) for k in klines]

    rsi, rsi_ma = calculate_indicators(close_prices, req.rsi_period, req.rsi_ma_period)

    adx = calculate_adx(high_prices, low_prices, close_prices, 14) if req.use_adx_filter else None
    atr = calculate_atr(high_prices, low_prices, close_prices, 14) if req.use_atr_stop or req.use_volatility_filter else None
    macd_line, macd_signal_line, _ = calculate_macd(close_prices) if req.use_macd_filter else (None, None, None)
    bb_upper, _, bb_lower = calculate_bollinger_bands(close_prices) if req.use_bb_filter else (None, None, None)
    volume_sma = calculate_volume_sma(volumes) if req.use_volume_filter else None
    stoch_k, stoch_d = calculate_stochastic_rsi(close_prices) if req.use_stochrsi_filter else (None, None)

    trend_signals = None
    if req.strategy_mode in ("trend", "combined"):
        _, _, trend_signals = calculate_trend_signal(close_prices, req.ema_fast_period, req.ema_slow_period)

    ema_regime = calculate_ema(close_prices, req.ema_regime_period) if req.use_regime_filter else None

    return {
        "close_prices": close_prices,
        "high_prices": high_prices,
        "low_prices": low_prices,
        "volumes": volumes,
        "rsi": rsi,
        "rsi_ma": rsi_ma,
        "adx": adx,
        "atr": atr,
        "macd_line": macd_line,
        "macd_signal_line": macd_signal_line,
        "bb_upper": bb_upper,
        "bb_lower": bb_lower,
        "volume_sma": volume_sma,
        "stoch_k": stoch_k,
        "stoch_d": stoch_d,
        "trend_signals": trend_signals,
        "ema_regime": ema_regime,
    }


def _engine_from_request(req: StrategyParams, initial_capital: Optional[float] = None) -> BacktestEngine:
    return BacktestEngine(
        initial_capital=req.initial_capital if initial_capital is None else initial_capital,
        entry_amount=req.entry_amount,
        dca_30_amount=req.dca_30_amount,
        dca_60_amount=req.dca_60_amount,
        entry_threshold=req.entry_threshold,
        exit_threshold_2=req.exit_threshold_2,
        exit_threshold_3=req.exit_threshold_3,
        exit_pct_1=req.exit_pct_1,
        exit_pct_2=req.exit_pct_2,
        exit_pct_3=req.exit_pct_3,
        dca_30_drop=req.dca_30_drop,
        dca_60_drop=req.dca_60_drop,
        maker_fee=req.maker_fee,
        taker_fee=req.taker_fee,
        strategy_mode=req.strategy_mode,
        ema_fast_period=req.ema_fast_period,
        ema_slow_period=req.ema_slow_period,
        cooldown_bars=req.cooldown_bars,
        max_dca_trades=req.max_dca_trades,
        dca_disable_after_exit_pct=req.dca_disable_after_exit_pct,
        require_profit_for_staged_exit=req.require_profit_for_staged_exit,
        min_hold_bars=req.min_hold_bars,
        force_exit_after_bars=req.force_exit_after_bars,
        break_even_stop_after_exit_pct=req.break_even_stop_after_exit_pct,
        use_adx_filter=req.use_adx_filter,
        adx_threshold=req.adx_threshold,
        use_bb_filter=req.use_bb_filter,
        use_macd_filter=req.use_macd_filter,
        use_volume_filter=req.use_volume_filter,
        use_stochrsi_filter=req.use_stochrsi_filter,
        use_volatility_filter=req.use_volatility_filter,
        max_atr_pct=req.max_atr_pct,
        use_regime_filter=req.use_regime_filter,
        ema_regime_period=req.ema_regime_period,
        use_divergence_filter=req.use_divergence_filter,
        divergence_lookback=req.divergence_lookback,
        use_breakout_confirmation=req.use_breakout_confirmation,
        use_trailing_stop=req.use_trailing_stop,
        trailing_stop_pct=req.trailing_stop_pct,
        use_atr_stop=req.use_atr_stop,
        atr_multiplier=req.atr_multiplier,
    )


def _summarize_results(results: list[dict], errors: list[dict], total: int, initial_capital: float) -> dict:
    total_pnl = sum(r.get("total_pnl", 0) for r in results)
    total_pnl_pct = (total_pnl / initial_capital) * 100 if initial_capital > 0 else 0
    avg_win_rate = sum(r.get("win_rate", 0) for r in results) / len(results) if results else 0
    avg_drawdown = sum(r.get("max_drawdown", 0) for r in results) / len(results) if results else 0
    avg_profit_factor = sum(r.get("profit_factor", 0) for r in results) / len(results) if results else 0
    avg_risk_score = sum(r.get("risk_adjusted_score", 0) for r in results) / len(results) if results else 0
    total_fees = sum(r.get("total_fees", 0) for r in results)
    best_symbol = max(results, key=lambda x: x.get("total_pnl", 0)) if results else None
    worst_symbol = min(results, key=lambda x: x.get("total_pnl", 0)) if results else None

    return {
        "total_symbols": total,
        "successful": len(results),
        "failed": len(errors),
        "total_pnl": total_pnl,
        "total_pnl_pct": total_pnl_pct,
        "avg_pnl_per_symbol": total_pnl / len(results) if results else 0,
        "avg_win_rate": avg_win_rate,
        "avg_drawdown": avg_drawdown,
        "avg_profit_factor": avg_profit_factor,
        "avg_risk_adjusted_score": avg_risk_score,
        "total_fees": total_fees,
        "best_symbol": best_symbol.get("symbol") if best_symbol else None,
        "best_symbol_pnl": best_symbol.get("total_pnl", 0) if best_symbol else 0,
        "worst_symbol": worst_symbol.get("symbol") if worst_symbol else None,
        "worst_symbol_pnl": worst_symbol.get("total_pnl", 0) if worst_symbol else 0,
    }


@app.get("/api/symbols")
async def list_symbols():
    """Tüm vadeli sembolleri listeler."""
    try:
        symbols = await get_all_futures_symbols()
        return {"symbols": symbols, "count": len(symbols)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/backtest")
async def run_backtest(req: BacktestRequest):
    """Tek bir sembol için backtest çalıştırır."""
    try:
        klines = await get_klines(req.symbol, _timeframe_value(req.timeframe))
        if not klines:
            raise HTTPException(status_code=404, detail=f"{req.symbol} için veri bulunamadı")

        arrays = _prepare_arrays(klines, req)
        engine = _engine_from_request(req)
        results = engine.run(
            klines,
            arrays["rsi"],
            arrays["rsi_ma"],
            arrays["trend_signals"],
            arrays["adx"],
            arrays["macd_line"],
            arrays["macd_signal_line"],
            arrays["bb_upper"],
            arrays["bb_lower"],
            arrays["volume_sma"],
            arrays["stoch_k"],
            arrays["stoch_d"],
            arrays["atr"],
            lows=arrays["low_prices"],
            ema_regime=arrays["ema_regime"],
        )
        results["symbol"] = req.symbol
        results["timeframe"] = req.timeframe
        results["total_candles"] = len(klines)

        return results
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/backtest/multi")
async def run_multi_backtest(req: MultiBacktestRequest):
    """Birden fazla sembol için backtest çalıştırır."""
    results = []
    errors = []

    for symbol in req.symbols:
        try:
            klines = await get_klines(symbol, _timeframe_value(req.timeframe))
            if not klines:
                errors.append({"symbol": symbol, "error": "Veri bulunamadı"})
                continue

            arrays = _prepare_arrays(klines, req)
            engine = _engine_from_request(req)
            result = engine.run(
                klines,
                arrays["rsi"],
                arrays["rsi_ma"],
                arrays["trend_signals"],
                arrays["adx"],
                arrays["macd_line"],
                arrays["macd_signal_line"],
                arrays["bb_upper"],
                arrays["bb_lower"],
                arrays["volume_sma"],
                arrays["stoch_k"],
                arrays["stoch_d"],
                arrays["atr"],
                lows=arrays["low_prices"],
                ema_regime=arrays["ema_regime"],
            )
            result["symbol"] = symbol
            result["timeframe"] = req.timeframe
            result["total_candles"] = len(klines)
            results.append(result)

            await asyncio.sleep(0.1)

        except Exception as e:
            errors.append({"symbol": symbol, "error": str(e)})

    return {
        "results": results,
        "errors": errors,
        "summary": _summarize_results(results, errors, len(req.symbols), req.initial_capital),
    }


@app.post("/api/portfolio")
async def run_portfolio_backtest(req: PortfolioRequest):
    """Tüm coinlerde portföy bazlı, pozisyon limitli simülasyon."""
    all_klines = {}
    tf_value = _timeframe_value(req.timeframe)
    for symbol in req.symbols:
        try:
            klines = await get_klines(symbol, tf_value)
            if klines and len(klines) > 50:
                all_klines[symbol] = klines
        except:
            pass

    if not all_klines:
        raise HTTPException(status_code=404, detail="Veri bulunamadı")

    coin_signals = []
    for symbol, klines in all_klines.items():
        close_prices = [k["close"] for k in klines]
        rsi, rsi_ma = calculate_indicators(close_prices, req.rsi_period, req.rsi_period)

        for i in range(1, len(klines)):
            if rsi_ma[i] is None or rsi_ma[i-1] is None:
                continue
            if rsi_ma[i-1] >= req.entry_threshold and rsi_ma[i] < req.entry_threshold:
                coin_signals.append({
                    "symbol": symbol,
                    "entry_idx": i,
                    "entry_price": klines[i]["close"],
                    "entry_date": klines[i]["time"],
                    "klines": klines,
                    "rsi": rsi,
                    "rsi_ma": rsi_ma,
                })

    coin_signals.sort(key=lambda x: x["entry_date"])

    capital = req.initial_capital
    open_positions = []
    closed_trades = []
    traded_symbols = set()
    equity_curve = [{"time": coin_signals[0]["entry_date"] if coin_signals else 0, "equity": capital}]
    trade_id = 0

    for signal in coin_signals:
        symbol = signal["symbol"]
        klines = signal["klines"]
        rsi = signal["rsi"]
        rsi_ma = signal["rsi_ma"]
        entry_idx = signal["entry_idx"]

        if len(open_positions) >= req.max_positions:
            continue

        if symbol in traded_symbols:
            continue

        remaining_klines = klines[entry_idx:]
        remaining_rsi = rsi[entry_idx:]
        remaining_rsi_ma = rsi_ma[entry_idx:]

        if len(remaining_klines) < 5:
            continue

        engine = _engine_from_request(req, initial_capital=capital)
        result = engine.run(remaining_klines, remaining_rsi, remaining_rsi_ma)

        if result["total_trades"] > 0:
            trade_id += 1
            pnl = result["total_pnl"]
            capital += pnl
            traded_symbols.add(symbol)

            entry_date = signal["entry_date"]
            close_date = result["closed_trades"][-1]["close_date"] if result["closed_trades"] else 0
            duration_days = round((close_date - entry_date) / (1000 * 86400)) if close_date > entry_date else 0

            trade_history = []
            for ct in result["closed_trades"]:
                for t in ct.get("trades", []):
                    trade_history.append(t)

            first_ct = result["closed_trades"][0] if result["closed_trades"] else {}
            last_ct = result["closed_trades"][-1] if result["closed_trades"] else {}

            closed_trades.append({
                "id": trade_id,
                "symbol": symbol,
                "pnl": pnl,
                "pnl_pct": result["total_pnl_pct"],
                "win_rate": result["win_rate"],
                "total_buys": sum(1 for t in trade_history if t["type"] == "buy"),
                "total_sells": sum(1 for t in trade_history if t["type"] == "sell"),
                "entry_date": entry_date,
                "entry_date_str": first_ct.get("entry_date_str", ""),
                "entry_price": first_ct.get("first_entry_price", 0),
                "close_date": close_date,
                "close_date_str": last_ct.get("close_date_str", ""),
                "close_price": last_ct.get("max_price", 0) if last_ct.get("close_reason") == "sell_70" else signal["klines"][-1]["close"] if signal["klines"] else 0,
                "close_reason": last_ct.get("close_reason", ""),
                "duration_days": duration_days,
                "first_entry_price": first_ct.get("first_entry_price", 0),
                "avg_price": first_ct.get("avg_price", 0),
                "max_price": first_ct.get("max_price", 0),
                "min_price": first_ct.get("min_price", 0),
                "max_gain_pct": first_ct.get("max_gain_pct", 0),
                "max_loss_pct": first_ct.get("max_loss_pct", 0),
                "trade_history": trade_history,
            })

            equity_curve.append({
                "time": result["closed_trades"][-1]["close_date"] if result["closed_trades"] else signal["entry_date"],
                "equity": capital,
            })

    winning = [t for t in closed_trades if t["pnl"] > 0]
    losing = [t for t in closed_trades if t["pnl"] <= 0]
    total_pnl = capital - req.initial_capital
    first_date = closed_trades[0]["entry_date"] if closed_trades else 0
    last_date = closed_trades[-1]["close_date"] if closed_trades else 0
    total_days = round((last_date - first_date) / (1000 * 86400)) if last_date > first_date else 0
    first_date_str = datetime.datetime.fromtimestamp(first_date / 1000).strftime('%Y-%m-%d') if first_date else ""
    last_date_str = datetime.datetime.fromtimestamp(last_date / 1000).strftime('%Y-%m-%d') if last_date else ""

    return {
        "initial_capital": req.initial_capital,
        "final_capital": capital,
        "total_pnl": total_pnl,
        "total_pnl_pct": (total_pnl / req.initial_capital) * 100 if req.initial_capital > 0 else 0,
        "total_trades": len(closed_trades),
        "winning_trades": len(winning),
        "losing_trades": len(losing),
        "win_rate": len(winning) / len(closed_trades) * 100 if closed_trades else 0,
        "avg_pnl_per_trade": total_pnl / len(closed_trades) if closed_trades else 0,
        "best_trade": max(closed_trades, key=lambda x: x["pnl"]) if closed_trades else None,
        "worst_trade": min(closed_trades, key=lambda x: x["pnl"]) if closed_trades else None,
        "trades": closed_trades,
        "equity_curve": equity_curve,
        "coin_count": len(all_klines),
        "first_date_str": first_date_str,
        "last_date_str": last_date_str,
        "total_days": total_days,
    }


@app.post("/api/optimize")
async def run_optimization(req: OptimizeRequest):
    """Çoklu zaman dilimi ve parametre optimizasyonu."""
    param_combos = [
        {"mode": "rsi", "rsi": 14, "ma": 14, "entry": 30, "e2": 50, "e3": 70, "d1": 0.25, "d2": 0.50, "ep1": 25, "ep2": 25, "ep3": 50, "cool": 1, "regime": True, "vol": True},
        {"mode": "rsi", "rsi": 21, "ma": 21, "entry": 30, "e2": 50, "e3": 70, "d1": 0.30, "d2": 0.60, "ep1": 25, "ep2": 25, "ep3": 50, "cool": 1, "regime": True, "vol": True},
        {"mode": "rsi", "rsi": 14, "ma": 14, "entry": 35, "e2": 50, "e3": 70, "d1": 0.25, "d2": 0.50, "ep1": 25, "ep2": 25, "ep3": 50, "cool": 2, "regime": True, "vol": True},
        {"mode": "rsi", "rsi": 21, "ma": 21, "entry": 40, "e2": 50, "e3": 70, "d1": 0.30, "d2": 0.60, "ep1": 25, "ep2": 25, "ep3": 50, "cool": 2, "regime": True, "vol": True},
        {"mode": "rsi", "rsi": 14, "ma": 14, "entry": 45, "e2": 55, "e3": 75, "d1": 0.25, "d2": 0.50, "ep1": 25, "ep2": 25, "ep3": 50, "cool": 2, "regime": True, "vol": True},
        {"mode": "trend", "fast": 10, "slow": 30, "entry": 30, "e2": 50, "e3": 70, "d1": 0.25, "d2": 0.50, "ep1": 25, "ep2": 25, "ep3": 50, "cool": 1, "regime": False, "vol": False},
        {"mode": "trend", "fast": 20, "slow": 50, "entry": 30, "e2": 50, "e3": 70, "d1": 0.30, "d2": 0.60, "ep1": 25, "ep2": 25, "ep3": 50, "cool": 1, "regime": False, "vol": False},
        {"mode": "trend", "fast": 9, "slow": 21, "entry": 30, "e2": 50, "e3": 70, "d1": 0.20, "d2": 0.40, "ep1": 25, "ep2": 25, "ep3": 50, "cool": 2, "regime": False, "vol": False},
        {"mode": "rsi", "rsi": 14, "ma": 14, "entry": 30, "e2": 50, "e3": 70, "d1": 0.25, "d2": 0.50, "ep1": 25, "ep2": 25, "ep3": 50, "cool": 1, "trail": 0.05, "regime": True, "vol": True},
        {"mode": "rsi", "rsi": 14, "ma": 14, "entry": 35, "e2": 50, "e3": 70, "d1": 0.25, "d2": 0.50, "ep1": 25, "ep2": 25, "ep3": 50, "cool": 2, "trail": 0.08, "regime": True, "vol": True},
        {"mode": "rsi", "rsi": 21, "ma": 21, "entry": 40, "e2": 55, "e3": 75, "d1": 0.30, "d2": 0.60, "ep1": 25, "ep2": 25, "ep3": 50, "cool": 2, "trail": 0.10, "regime": True, "vol": True},
        {"mode": "trend", "fast": 10, "slow": 30, "entry": 30, "e2": 50, "e3": 70, "d1": 0.25, "d2": 0.50, "ep1": 25, "ep2": 25, "ep3": 50, "cool": 1, "trail": 0.05, "regime": False, "vol": False},
        {"mode": "trend", "fast": 20, "slow": 50, "entry": 30, "e2": 50, "e3": 70, "d1": 0.30, "d2": 0.60, "ep1": 25, "ep2": 25, "ep3": 50, "cool": 1, "trail": 0.08, "regime": False, "vol": False},
    ]

    all_results = []

    for tf_key in req.timeframes:
        tf_value = TIMEFRAME_MAP.get(tf_key, "Day1")
        all_klines = {}
        for symbol in req.symbols:
            try:
                klines = await get_klines(symbol, tf_value)
                if klines and len(klines) > 100:
                    all_klines[symbol] = klines
            except:
                pass
            await asyncio.sleep(0.05)

        for combo in param_combos:
            combo_results = []
            for symbol, klines in all_klines.items():
                try:
                    arrays = _prepare_arrays(
                        klines,
                        BacktestRequest(
                            symbol=symbol,
                            rsi_period=combo["rsi"],
                            rsi_ma_period=combo["ma"],
                            entry_threshold=combo["entry"],
                            exit_threshold_2=combo["e2"],
                            exit_threshold_3=combo["e3"],
                            exit_pct_1=combo["ep1"],
                            exit_pct_2=combo["ep2"],
                            exit_pct_3=combo["ep3"],
                            dca_30_drop=combo["d1"],
                            dca_60_drop=combo["d2"],
                            strategy_mode=combo["mode"],
                            ema_fast_period=combo.get("fast", 20),
                            ema_slow_period=combo.get("slow", 50),
                            cooldown_bars=combo.get("cool", 1),
                            use_regime_filter=combo.get("regime", False),
                            use_volatility_filter=combo.get("vol", False),
                            use_trailing_stop="trail" in combo,
                            trailing_stop_pct=combo.get("trail", 0.05),
                        ),
                    )

                    engine = BacktestEngine(
                        initial_capital=req.initial_capital,
                        entry_amount=req.entry_amount,
                        dca_30_amount=req.dca_30_amount,
                        dca_60_amount=req.dca_60_amount,
                        entry_threshold=combo["entry"],
                        exit_threshold_2=combo["e2"],
                        exit_threshold_3=combo["e3"],
                        exit_pct_1=combo["ep1"],
                        exit_pct_2=combo["ep2"],
                        exit_pct_3=combo["ep3"],
                        dca_30_drop=combo["d1"],
                        dca_60_drop=combo["d2"],
                        maker_fee=req.maker_fee,
                        taker_fee=req.taker_fee,
                        strategy_mode=combo["mode"],
                        ema_fast_period=combo.get("fast", 20),
                        ema_slow_period=combo.get("slow", 50),
                        cooldown_bars=combo.get("cool", 1),
                        use_regime_filter=combo.get("regime", False),
                        use_volatility_filter=combo.get("vol", False),
                        use_trailing_stop="trail" in combo,
                        trailing_stop_pct=combo.get("trail", 0.05),
                    )

                    result = engine.run(
                        klines,
                        arrays["rsi"],
                        arrays["rsi_ma"],
                        arrays["trend_signals"],
                        arrays["adx"],
                        arrays["macd_line"],
                        arrays["macd_signal_line"],
                        arrays["bb_upper"],
                        arrays["bb_lower"],
                        arrays["volume_sma"],
                        arrays["stoch_k"],
                        arrays["stoch_d"],
                        arrays["atr"],
                        lows=arrays["low_prices"],
                        ema_regime=arrays["ema_regime"],
                    )
                    if result["total_trades"] > 0:
                        combo_results.append({
                            "symbol": symbol,
                            "pnl": result["total_pnl"],
                            "pnl_pct": result["total_pnl_pct"],
                            "trades": result["total_trades"],
                            "win_rate": result["win_rate"],
                            "max_drawdown": result["max_drawdown"],
                            "risk_adjusted_score": result["risk_adjusted_score"],
                        })
                except:
                    pass

            if combo_results:
                avg_pnl = sum(r["pnl_pct"] for r in combo_results) / len(combo_results)
                avg_winrate = sum(r["win_rate"] for r in combo_results) / len(combo_results)
                avg_drawdown = sum(r["max_drawdown"] for r in combo_results) / len(combo_results)
                avg_risk = sum(r["risk_adjusted_score"] for r in combo_results) / len(combo_results)
                total_trades = sum(r["trades"] for r in combo_results)

                trail_str = f" + T{combo.get('trail', 0)*100:.0f}%" if "trail" in combo else ""
                regime_str = " + Regime" if combo.get("regime") else ""
                vol_str = " + VolGuard" if combo.get("vol") else ""
                ema_str = f" EMA{combo.get('fast',20)}/{combo.get('slow',50)}" if combo["mode"] == "trend" else ""
                label = f"{combo['mode']}{ema_str} RSI{combo['rsi']}/{combo['ma']} E={combo['entry']} DCA={combo['d1']:.0%}/{combo['d2']:.0%}{trail_str}{regime_str}{vol_str}"

                all_results.append({
                    "timeframe": tf_key,
                    "label": label,
                    "avg_pnl_pct": round(avg_pnl, 2),
                    "avg_win_rate": round(avg_winrate, 1),
                    "avg_drawdown": round(avg_drawdown, 2),
                    "avg_risk_adjusted_score": round(avg_risk, 2),
                    "total_trades": total_trades,
                    "coin_count": len(combo_results),
                    "coins": [r["symbol"] for r in combo_results],
                })

    all_results.sort(key=lambda x: x["avg_pnl_pct"], reverse=True)
    risk_adjusted = sorted(all_results, key=lambda x: x["avg_pnl_pct"] / max(x["avg_drawdown"], 0.1), reverse=True)

    best_per_tf = {}
    for r in all_results:
        if r["timeframe"] not in best_per_tf:
            best_per_tf[r["timeframe"]] = r

    return {
        "best_by_pnl": all_results[0] if all_results else None,
        "best_by_risk": risk_adjusted[0] if risk_adjusted else None,
        "best_per_timeframe": best_per_tf,
        "all_results": all_results[:50],
        "total_tests": len(all_results),
        "risk_adjusted_top10": risk_adjusted[:10],
    }


@app.get("/api/health")
async def health():
    return {"status": "ok"}


ACCESS_KEY = "MEXC2024"
_system_settings = {
    "cycle_interval": 300,
    "min_confidence": 0.15,
    "max_positions": 5,
    "risk_per_trade": 2.0,
    "daily_risk": 5.0,
    "leverage_max": 20,
}


@app.post("/api/auth/verify")
async def verify_key(body: dict):
    key = body.get("key", "")
    if key == ACCESS_KEY:
        return {"valid": True}
    raise HTTPException(status_code=401, detail="Invalid key")


@app.get("/api/config")
async def get_config():
    return _system_settings


@app.post("/api/config")
async def update_config(body: dict):
    for k, v in body.items():
        if k in _system_settings:
            _system_settings[k] = v
    return {"status": "ok", "settings": _system_settings}


@app.get("/api/test/telegram")
async def test_telegram():
    from notifier import _post_telegram
    await _post_telegram("🧪 MEXC Trading Bot test mesaji - Telegram baglantisi basarili!")
    return {"status": "sent"}


@app.get("/api/test/telegram-chart")
async def test_telegram_chart():
    """Telegram'a ornek grafik gonderir (test)."""
    from notifier import generate_chart_svg, send_telegram_photo
    from mexc_client import get_klines

    test_symbols = ["BTC_USDT", "ETH_USDT", "SOL_USDT"]
    for sym in test_symbols:
        try:
            klines = await get_klines(sym, "Min5")
            if klines and len(klines) > 30:
                svg = generate_chart_svg(
                    klines[-60:], "long",
                    entry_price=float(klines[-40]["close"]),
                    exit_price=float(klines[-1]["close"]),
                    tp=float(klines[-40]["close"]) * 1.03,
                    sl=float(klines[-40]["close"]) * 0.98,
                )
                if svg:
                    caption = f"🧪 TEST GRAFİK - {sym} LONG\nGiriş: ${float(klines[-40]['close']):.2f} → Çıkış: ${float(klines[-1]['close']):.2f}\n🏆 Kar/Zarar testi"
                    await send_telegram_photo(svg, caption)
                    return {"status": "sent", "symbol": sym}
        except Exception as e:
            continue

    return {"status": "error", "message": "Grafik olusturulamadi"}


@app.post("/api/backtest/stream")
async def run_backtest_stream(req: MultiBacktestRequest):
    """Birden fazla sembol için SSE ile canlı ilerleme akışı."""
    async def event_generator():
        total = len(req.symbols)
        results = []
        errors = []
        BATCH_SIZE = 50

        yield f"data: {json.dumps({'type': 'progress', 'current': 0, 'total': total, 'percent': 0, 'symbol': '', 'status': 'fetching_data', 'message': f'{total} coin verisi çekiliyor...'})}\n\n"

        all_klines = {}
        tf_value = _timeframe_value(req.timeframe)
        for batch_start in range(0, total, BATCH_SIZE):
            batch = req.symbols[batch_start:batch_start + BATCH_SIZE]
            tasks = [get_klines(sym, tf_value) for sym in batch]
            klines_results = await asyncio.gather(*tasks, return_exceptions=True)

            for sym, klines in zip(batch, klines_results):
                if isinstance(klines, Exception):
                    errors.append({"symbol": sym, "error": str(klines)})
                elif klines:
                    all_klines[sym] = klines
                else:
                    errors.append({"symbol": sym, "error": "Veri bulunamadı"})

            progress_pct = round(((batch_start + len(batch)) / total) * 50, 1)
            yield f"data: {json.dumps({'type': 'progress', 'current': batch_start + len(batch), 'total': total, 'percent': progress_pct, 'symbol': '', 'status': 'fetching', 'message': f'Veri çekildi: {len(all_klines)}/{total} coin'})}\n\n"

        processed = 0
        for symbol in req.symbols:
            if symbol not in all_klines:
                continue

            klines = all_klines[symbol]
            try:
                arrays = _prepare_arrays(klines, req)
                engine = _engine_from_request(req)
                result = engine.run(
                    klines,
                    arrays["rsi"],
                    arrays["rsi_ma"],
                    arrays["trend_signals"],
                    arrays["adx"],
                    arrays["macd_line"],
                    arrays["macd_signal_line"],
                    arrays["bb_upper"],
                    arrays["bb_lower"],
                    arrays["volume_sma"],
                    arrays["stoch_k"],
                    arrays["stoch_d"],
                    arrays["atr"],
                    lows=arrays["low_prices"],
                    ema_regime=arrays["ema_regime"],
                )
                result["symbol"] = symbol
                result["timeframe"] = req.timeframe
                result["total_candles"] = len(klines)
                results.append(result)

                processed += 1
                progress_pct = 50 + round((processed / len(all_klines)) * 50, 1)
                yield f"data: {json.dumps({'type': 'progress', 'current': processed, 'total': len(all_klines), 'percent': progress_pct, 'symbol': symbol, 'status': 'done', 'pnl': result['total_pnl'], 'pnl_pct': result['total_pnl_pct'], 'risk_score': result['risk_adjusted_score']})}\n\n"

            except Exception as e:
                errors.append({"symbol": symbol, "error": str(e)})

        final_data = {
            "type": "complete",
            "results": results,
            "errors": errors,
            "summary": _summarize_results(results, errors, total, req.initial_capital),
        }
        yield f"data: {json.dumps(final_data)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ==================== MULTI-AGENT TRADING SYSTEM ENDPOINTS ====================

@app.get("/api/system/status")
async def system_status():
    """Sistem durumunu döndürür."""
    return orchestrator.get_status()


@app.get("/api/agents/status")
async def agents_status():
    """Tüm ajanların durumunu döndürür."""
    status = orchestrator.get_status()
    return {"agents": status.get("agents", {}), "running": status.get("running", False)}


@app.get("/api/scanner/hot")
async def scanner_hot():
    """En aktif çiftleri döndürür."""
    try:
        result = await orchestrator.scanner.analyze({})
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/technical/signals")
async def technical_signals():
    """Teknik sinyalleri döndürür."""
    try:
        scanner_result = await orchestrator.scanner.analyze({})
        top_pairs = scanner_result.get("hot_pairs", [])[:20]
        symbols = [p["symbol"] for p in top_pairs]
        result = await orchestrator.technical.analyze({"symbols": symbols})
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sentiment/current")
async def sentiment_current():
    """Güncel duygu analizini döndürür."""
    try:
        scanner_result = await orchestrator.scanner.analyze({})
        result = await orchestrator.sentiment.analyze({"scanner": scanner_result})
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/risk/metrics")
async def risk_metrics():
    """Risk metriklerini döndürür."""
    try:
        result = await orchestrator.risk.analyze({
            "total_equity": orchestrator.total_equity,
            "open_positions": orchestrator.open_positions,
        })
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/portfolio/positions")
async def portfolio_positions():
    """Açık pozisyonları döndürür."""
    return {
        "positions": orchestrator.get_open_positions(),
        "total_equity": orchestrator.total_equity,
        "available_capital": orchestrator.available_capital,
    }


@app.get("/api/patron/decisions")
async def patron_decisions():
    """Son patron kararlarını döndürür."""
    results = orchestrator.get_latest_results()
    patron_data = results.get("patron", {})
    return patron_data


@app.get("/api/trading/history")
async def trading_history():
    """İşlem geçmişini döndürür."""
    return {"history": orchestrator.get_trade_history()}


@app.get("/api/trade/{trade_index}/klines")
async def get_trade_klines(trade_index: int):
    """Belirli bir işlem için 5m mum verisi getirir."""
    history = orchestrator.get_trade_history()
    if trade_index < 0 or trade_index >= len(history):
        raise HTTPException(status_code=404, detail="İşlem bulunamadı")

    trade = history[trade_index]
    symbol = trade.get("symbol", "")
    entry_time = trade.get("entry_time", 0)
    close_time = trade.get("close_time", time.time())

    margin = 300
    start_ms = int((entry_time - margin) * 1000)
    end_ms = int((close_time + margin) * 1000)

    try:
        klines = await get_klines(symbol, "Min5")
        if not klines:
            return {"klines": [], "entry_price": trade["entry_price"], "exit_price": trade.get("exit_price", 0),
                    "entry_time": entry_time, "close_time": close_time,
                    "direction": trade.get("direction", ""), "close_reason": trade.get("close_reason", "")}

        filtered = [k for k in klines if start_ms <= k.get("time", 0) <= end_ms]

        return {
            "klines": filtered,
            "entry_price": trade["entry_price"],
            "exit_price": trade.get("exit_price", 0),
            "entry_time": entry_time,
            "close_time": close_time,
            "direction": trade.get("direction", ""),
            "close_reason": trade.get("close_reason", ""),
            "symbol": symbol,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/trading/latest")
async def trading_latest():
    """Son döngü sonuçlarını döndürür."""
    return orchestrator.get_latest_results()


@app.get("/api/position/{symbol}/klines")
async def get_position_klines(symbol: str):
    """Acik pozisyon icin canli mum verisi getirir (5m)."""
    try:
        klines = await get_klines(symbol, "Min5")
        if not klines:
            return {"klines": [], "symbol": symbol}

        position = None
        for pos in orchestrator.open_positions:
            if pos["symbol"] == symbol:
                position = pos
                break

        return {
            "klines": klines[-120:],
            "symbol": symbol,
            "entry_price": position["entry_price"] if position else 0,
            "take_profit": position.get("take_profit", 0) if position else 0,
            "stop_loss": position.get("stop_loss", 0) if position else 0,
            "direction": position.get("direction", "") if position else "",
            "current_price": position.get("current_price", 0) if position else 0,
            "entry_time": position.get("entry_time", 0) if position else 0,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/trading/start")
async def trading_start():
    """Trading'i başlatır."""
    global orchestrator_task
    if orchestrator.running:
        return {"status": "already_running", "message": "Sistem zaten çalışıyor"}

    orchestrator_task = asyncio.create_task(orchestrator.start())
    return {"status": "started", "message": "Trading sistemi başlatıldı"}


@app.post("/api/trading/stop")
async def trading_stop():
    """Trading'i durdurur."""
    await orchestrator.stop()
    return {"status": "stopped", "message": "Trading sistemi durduruldu"}


@app.post("/api/trading/cycle")
async def trading_cycle():
    """Tek bir döngü çalıştırır."""
    try:
        result = await orchestrator.run_cycle()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/trading/equity")
async def update_equity(equity: float):
    """Toplam sermayeyi günceller."""
    orchestrator.total_equity = equity
    orchestrator.available_capital = equity
    orchestrator.risk.total_equity = equity
    orchestrator.portfolio.total_equity = equity
    return {"status": "updated", "equity": equity}


@app.post("/api/trading/close/{symbol}")
async def close_position(symbol: str):
    """Belirli bir pozisyonu kapatır."""
    for pos in list(orchestrator.open_positions):
        if pos["symbol"] == symbol.upper():
            orchestrator._close_position(pos, "Manuel kapatma")
            return {"status": "closed", "symbol": symbol.upper()}
    raise HTTPException(status_code=404, detail=f"{symbol} pozisyonu bulunamadı")


@app.websocket("/ws/live")
async def websocket_live(websocket: WebSocket):
    """Gerçek zamanlı veri akışı için WebSocket."""
    await websocket.accept()
    orchestrator.websocket_clients.append(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
            elif data == "status":
                status = orchestrator.get_status()
                await websocket.send_text(json.dumps(status, default=str))
    except WebSocketDisconnect:
        if websocket in orchestrator.websocket_clients:
            orchestrator.websocket_clients.remove(websocket)
    except Exception:
        if websocket in orchestrator.websocket_clients:
            orchestrator.websocket_clients.remove(websocket)


FRONTEND_DIR = Path(__file__).parent.parent / "frontend" / "dist"

if FRONTEND_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIR / "assets")), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(request: Request, full_path: str):
        file_path = FRONTEND_DIR / full_path
        if file_path.is_file():
            return FileResponse(str(file_path))
        return FileResponse(str(FRONTEND_DIR / "index.html"))
