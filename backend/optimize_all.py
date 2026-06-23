import sys
sys.path.insert(0, r'D:\Ahmet Proje Dosyaları\rsiproject\backend')
import asyncio
import json
import time
from indicators import calculate_indicators, calculate_trend_signal
from backtest_engine import BacktestEngine
from mexc_client import get_klines

TIMEFRAMES = {
    '5m': 'Min5',
    '15m': 'Min15',
    '30m': 'Min30',
    '1h': 'Min60',
    '4h': 'Hour4',
    '8h': 'Hour8',
    '1D': 'Day1',
}

SYMBOLS = ['BTC_USDT', 'ETH_USDT', 'DOGE_USDT', 'SOL_USDT', 'XRP_USDT', 'PEPE_USDT', 'WIF_USDT', 'ADA_USDT']

PARAM_COMBOS = [
    # RSI only
    {"strategy_mode": "rsi", "rsi_period": 14, "rsi_ma_period": 14, "entry": 30, "exit2": 50, "exit3": 70, "dca1": 0.25, "dca2": 0.50},
    {"strategy_mode": "rsi", "rsi_period": 21, "rsi_ma_period": 21, "entry": 30, "exit2": 50, "exit3": 70, "dca1": 0.25, "dca2": 0.50},
    {"strategy_mode": "rsi", "rsi_period": 14, "rsi_ma_period": 14, "entry": 35, "exit2": 50, "exit3": 70, "dca1": 0.25, "dca2": 0.50},
    {"strategy_mode": "rsi", "rsi_period": 21, "rsi_ma_period": 21, "entry": 40, "exit2": 50, "exit3": 70, "dca1": 0.25, "dca2": 0.50},
    {"strategy_mode": "rsi", "rsi_period": 14, "rsi_ma_period": 14, "entry": 45, "exit2": 55, "exit3": 75, "dca1": 0.25, "dca2": 0.50},
    # Trend only
    {"strategy_mode": "trend", "ema_fast": 10, "ema_slow": 30, "entry": 30, "exit2": 50, "exit3": 70, "dca1": 0.25, "dca2": 0.50},
    {"strategy_mode": "trend", "ema_fast": 20, "ema_slow": 50, "entry": 30, "exit2": 50, "exit3": 70, "dca1": 0.25, "dca2": 0.50},
    {"strategy_mode": "trend", "ema_fast": 9, "ema_slow": 21, "entry": 30, "exit2": 50, "exit3": 70, "dca1": 0.20, "dca2": 0.40},
    # With trailing stop
    {"strategy_mode": "rsi", "rsi_period": 14, "rsi_ma_period": 14, "entry": 30, "exit2": 50, "exit3": 70, "dca1": 0.25, "dca2": 0.50, "trailing": 0.05},
    {"strategy_mode": "rsi", "rsi_period": 14, "rsi_ma_period": 14, "entry": 35, "exit2": 50, "exit3": 70, "dca1": 0.25, "dca2": 0.50, "trailing": 0.08},
    {"strategy_mode": "rsi", "rsi_period": 21, "rsi_ma_period": 21, "entry": 40, "exit2": 55, "exit3": 75, "dca1": 0.25, "dca2": 0.50, "trailing": 0.10},
    {"strategy_mode": "trend", "ema_fast": 10, "ema_slow": 30, "entry": 30, "exit2": 50, "exit3": 70, "dca1": 0.25, "dca2": 0.50, "trailing": 0.05},
    {"strategy_mode": "trend", "ema_fast": 20, "ema_slow": 50, "entry": 30, "exit2": 50, "exit3": 70, "dca1": 0.25, "dca2": 0.50, "trailing": 0.08},
]

async def run_test(symbol, tf_key, tf_value, combo, all_klines):
    if symbol not in all_klines:
        return None
    klines = all_klines[symbol]
    if not klines or len(klines) < 100:
        return None

    close_prices = [k["close"] for k in klines]

    rsi_period = combo.get("rsi_period", 21)
    rsi_ma_period = combo.get("rsi_ma_period", 21)
    rsi, rsi_ma = calculate_indicators(close_prices, rsi_period, rsi_ma_period)

    trend_signals = None
    ema_fast = combo.get("ema_fast", 20)
    ema_slow = combo.get("ema_slow", 50)
    if combo["strategy_mode"] in ("trend", "combined"):
        _, _, trend_signals = calculate_trend_signal(close_prices, ema_fast, ema_slow)

    engine = BacktestEngine(
        initial_capital=10000,
        entry_amount=1000,
        dca_30_amount=3000,
        dca_60_amount=6000,
        entry_threshold=combo["entry"],
        exit_threshold_2=combo["exit2"],
        exit_threshold_3=combo["exit3"],
        dca_30_drop=combo["dca1"],
        dca_60_drop=combo["dca2"],
        maker_fee=0.0002,
        taker_fee=0.0006,
        strategy_mode=combo["strategy_mode"],
        ema_fast_period=ema_fast,
        ema_slow_period=ema_slow,
        use_trailing_stop="trailing" in combo,
        trailing_stop_pct=combo.get("trailing", 0.05),
    )

    result = engine.run(klines, rsi, rsi_ma, trend_signals)
    return result

async def main():
    all_results = []
    total = len(TIMEFRAMES) * len(PARAM_COMBOS)
    done = 0

    print(f"=== {len(TIMEFRAMES)} zaman dilimi x {len(PARAM_COMBOS)} parametre = {total} test ===\n")

    for tf_key, tf_value in TIMEFRAMES.items():
        print(f"--- {tf_key} veri çekiliyor ---")
        all_klines = {}
        for symbol in SYMBOLS:
            try:
                klines = await get_klines(symbol, tf_value)
                if klines and len(klines) > 100:
                    all_klines[symbol] = klines
            except:
                pass
            await asyncio.sleep(0.1)

        print(f"  {len(all_klines)}/{len(SYMBOLS)} coin çekildi\n")

        for i, combo in enumerate(PARAM_COMBOS):
            combo_results = []
            for symbol in all_klines:
                result = await run_test(symbol, tf_key, tf_value, combo, all_klines)
                if result and result["total_trades"] > 0:
                    combo_results.append({
                        "symbol": symbol,
                        "pnl": result["total_pnl"],
                        "pnl_pct": result["total_pnl_pct"],
                        "trades": result["total_trades"],
                        "win_rate": result["win_rate"],
                        "max_drawdown": result["max_drawdown"],
                    })

            if combo_results:
                avg_pnl = sum(r["pnl_pct"] for r in combo_results) / len(combo_results)
                avg_winrate = sum(r["win_rate"] for r in combo_results) / len(combo_results)
                avg_drawdown = sum(r["max_drawdown"] for r in combo_results) / len(combo_results)
                total_trades = sum(r["trades"] for r in combo_results)
                coin_count = len(combo_results)

                mode = combo["strategy_mode"]
                trailing_str = f" + trailing {combo.get('trailing', 0)*100:.0f}%" if "trailing" in combo else ""
                ema_str = f" EMA{combo.get('ema_fast', 20)}/{combo.get('ema_slow', 50)}" if mode == "trend" else ""
                label = f"{mode}{ema_str} RSI{combo.get('rsi_period', '-')}/{combo.get('rsi_ma_period', '-')} entry={combo['entry']}{trailing_str}"

                all_results.append({
                    "timeframe": tf_key,
                    "label": label,
                    "avg_pnl_pct": round(avg_pnl, 2),
                    "avg_win_rate": round(avg_winrate, 1),
                    "avg_drawdown": round(avg_drawdown, 2),
                    "total_trades": total_trades,
                    "coin_count": coin_count,
                    "coins": [r["symbol"] for r in combo_results],
                })

            done += 1
            if done % 10 == 0:
                print(f"  {done}/{total} tamamlandı...")

    # Sonuçları sırala
    all_results.sort(key=lambda x: x["avg_pnl_pct"], reverse=True)

    print("\n" + "="*80)
    print("EN IYI 20 SONUC (ortalama pnl%):\n")
    for i, r in enumerate(all_results[:20], 1):
        print(f"{i:2}. {r['timeframe']:3} | {r['avg_pnl_pct']:8.2f}% | WR:{r['avg_win_rate']:5.1f}% | DD:{r['avg_drawdown']:6.2f}% | {r['total_trades']:3} trades | {r['coin_count']} coins | {r['label']}")

    print("\n" + "="*80)
    print("EN IYI 10 RISK/DENGELI (yüksek pnl, düşük drawdown):\n")
    risk_adjusted = sorted(all_results, key=lambda x: x["avg_pnl_pct"] / max(x["avg_drawdown"], 0.1), reverse=True)
    for i, r in enumerate(risk_adjusted[:10], 1):
        score = r["avg_pnl_pct"] / max(r["avg_drawdown"], 0.1)
        print(f"{i:2}. {r['timeframe']:3} | {r['avg_pnl_pct']:8.2f}% | DD:{r['avg_drawdown']:6.2f}% | Score:{score:6.2f} | {r['label']}")

    # Zaman dilimine göre en iyi
    print("\n" + "="*80)
    print("HER ZAMAN DILIMI ICIN EN IYI:\n")
    seen_tfs = set()
    for r in all_results:
        if r["timeframe"] not in seen_tfs:
            seen_tfs.add(r["timeframe"])
            print(f"  {r['timeframe']:3}: {r['avg_pnl_pct']:8.2f}% | WR:{r['avg_win_rate']:5.1f}% | {r['label']}")

    # Dosyaya kaydet
    with open(r"D:\Ahmet Proje Dosyaları\rsiproject\optimization_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSonuçlar kaydedildi: optimization_results.json")

asyncio.run(main())
