"""
İşlem yürütücü.
MEXC API ile pozisyon açar/kapatır.
"""

import time
import logging
from datetime import datetime, timezone
from collections import deque

from config import config

logger = logging.getLogger(__name__)

active_positions = {}
trade_history = deque(maxlen=200)
portfolio = {
    "capital": 5000.0,
    "initial_capital": 5000.0,
    "total_fees": 0.0,
    "total_closed_trades": 0,
}


async def fetch_price(client, symbol):
    """MEXC'den güncel fiyat al."""
    try:
        r = await client.get(f"{config.MEXC_BASE_URL}/ticker", params={"symbol": f"{symbol}_USDT"}, timeout=5)
        if r.status_code == 200:
            d = r.json()
            if d.get("success") and d.get("data"):
                return float(d["data"].get("lastPrice", 0))
    except Exception as e:
        logger.error(f"Price fetch error {symbol}: {e}")
    return 0


async def execute_signal(client, signal):
    """
    Sinyale göre pozisyon aç.
    signal: {"symbol": "BTC", "signal": "LONG", "score": 2.5, "confidence": 75}
    """
    sym = signal["symbol"]
    direction = signal["signal"]
    confidence = signal["confidence"]
    score = signal["score"]

    if sym in active_positions:
        logger.debug(f"Zaten {sym} pozisyonu var")
        return None

    if len(active_positions) >= config.MAX_POSITIONS:
        logger.debug(f"Maks pozisyon sayısı: {config.MAX_POSITIONS}")
        return None

    price = await fetch_price(client, f"{sym}_USDT")
    if price <= 0:
        return None

    lev = min(config.MAX_LEVERAGE, max(1, int(confidence / 15)))
    size_usdt = config.BASE_SIZE_USDT * (confidence / 50)
    notional = size_usdt * lev
    fee = notional * 0.0006

    portfolio["total_fees"] += fee
    portfolio["capital"] -= fee

    base_sl = config.SL_PCT / lev
    base_tp = config.TP_PCT / lev

    if direction == "SHORT":
        sl_price = round(price * (1 + base_sl), 10)
        tp_price = round(price * (1 - base_tp), 10)
    else:
        sl_price = round(price * (1 - base_sl), 10)
        tp_price = round(price * (1 + base_tp), 10)

    pos = {
        "symbol": sym,
        "side": direction,
        "entry_price": price,
        "size": notional,
        "leverage": lev,
        "sl": sl_price,
        "tp": tp_price,
        "entry_time": datetime.now(timezone.utc),
        "unrealized_pnl": 0,
        "score": score,
        "confidence": confidence,
        "source": signal.get("source", "whale_tracker"),
    }
    active_positions[sym] = pos

    logger.info(f"POSITION OPEN: {sym} {direction} @ {price} | {lev}x | SL={sl_price} TP={tp_price}")
    return pos


async def check_positions(client):
    """Açık pozisyonları kontrol et, SL/TP'yi yönet."""
    if not active_positions:
        return []

    closed = []
    for sym, pos in list(active_positions.items()):
        price = await fetch_price(client, f"{sym}_USDT")
        if price <= 0:
            continue

        entry = pos["entry_price"]
        side = pos["side"]
        diff = (price - entry) / entry if side == "LONG" else (entry - price) / entry
        pos["unrealized_pnl"] = round(pos["size"] * diff, 2)

        exit_reason = None
        if side == "LONG":
            if price >= pos["tp"]:
                exit_reason = "TAKE_PROFIT"
            elif price <= pos["sl"]:
                exit_reason = "STOP_LOSS"
        else:
            if price <= pos["tp"]:
                exit_reason = "TAKE_PROFIT"
            elif price >= pos["sl"]:
                exit_reason = "STOP_LOSS"

        if exit_reason:
            pnl = pos["unrealized_pnl"]
            exit_fee = pos["size"] * 0.0006
            net_pnl = pnl - exit_fee
            portfolio["capital"] += (pos["size"] + pnl - exit_fee)
            portfolio["total_fees"] += exit_fee
            portfolio["total_closed_trades"] += 1

            source = pos.get("source", "unknown")
            trade_history.append({
                "symbol": sym,
                "side": side,
                "entry": entry,
                "exit": price,
                "pnl": round(net_pnl, 2),
                "leverage": pos["leverage"],
                "reason": exit_reason,
                "source": source,
                "time": datetime.now(timezone.utc).isoformat(),
            })
            logger.info(f"POSITION CLOSE: {sym} @ {price} | PnL={net_pnl:.2f} | {exit_reason}")
            del active_positions[sym]
            closed.append({"symbol": sym, "pnl": net_pnl, "reason": exit_reason, "source": source})

    return closed


def get_summary():
    """Dashboard için özet."""
    active = []
    for sym, pos in active_positions.items():
        active.append({
            "symbol": sym,
            "side": pos["side"],
            "entry": pos["entry_price"],
            "pnl": pos["unrealized_pnl"],
            "leverage": pos["leverage"],
            "sl": pos["sl"],
            "tp": pos["tp"],
        })

    cap = portfolio["capital"]
    init = portfolio["initial_capital"]
    total_trades = portfolio["total_closed_trades"]
    wins = sum(1 for t in trade_history if t["pnl"] > 0)
    wr = round(wins / total_trades * 100, 1) if total_trades > 0 else 0

    return {
        "capital": round(cap, 2),
        "pnl": round(cap - init, 2),
        "active_positions": active,
        "total_trades": total_trades,
        "win_rate": wr,
    }
