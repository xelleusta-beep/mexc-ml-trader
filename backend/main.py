"""
MEXC ML Trading System - FastAPI Backend
Gerçek MEXC Futures verisi + ML modelleri
"""

import asyncio
import json
import time
import logging
from collections import deque
from datetime import datetime, timedelta
from typing import Optional, List
from contextlib import asynccontextmanager

import os
import numpy as np
import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn

from ml_engine import MLEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Telegram Config ──────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

async def send_telegram_message(text: str):
    logger.info(f"Telegram Message Triggered: {text[:100]}...")
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.info("Telegram credentials missing, skipping actual send")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "HTML"
    }
    try:
        async with httpx.AsyncClient() as client:
            await client.post(url, json=payload, timeout=10)
    except Exception as e:
        logger.error(f"Telegram error: {e}")

# ── Globals ──────────────────────────────────────────────────────────────────
ml_engine = MLEngine()
active_connections: List[WebSocket] = []
scanner_cache: dict = {}
agent_states: dict = {}
trade_history = deque(maxlen=100)
portfolio = {
    "capital": 100000.0,
    "total_fees": 0.0,
    "initial_capital": 100000.0
}

PERSISTENCE_FILE = "persistence.json"
MAX_OPEN_POSITIONS = 30
MIN_VOLUME_24H = 10000000.0  # $10M

def json_serial(obj):
    """JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError ("Type %s not serializable" % type(obj))

def save_data():
    try:
        data = {
            "agent_states": agent_states,
            "trade_history": list(trade_history),
            "portfolio": portfolio,
            "timestamp": datetime.utcnow().isoformat()
        }
        with open(PERSISTENCE_FILE, "w") as f:
            json.dump(data, f, default=json_serial)
    except Exception as e:
        logger.error(f"Save error: {e}")

def load_data():
    global agent_states, trade_history, portfolio
    if os.path.exists(PERSISTENCE_FILE):
        try:
            with open(PERSISTENCE_FILE, "r") as f:
                data = json.load(f)

                # Restore datetime objects in agent_states
                loaded_states = data.get("agent_states", {})
                for sym, st in loaded_states.items():
                    if st.get("last_exit_time"):
                        st["last_exit_time"] = datetime.fromisoformat(st["last_exit_time"])
                    if st.get("active_pos") and st["active_pos"].get("entry_time_raw"):
                        st["active_pos"]["entry_time_raw"] = datetime.fromisoformat(st["active_pos"]["entry_time_raw"])

                agent_states = loaded_states
                trade_history = deque(data.get("trade_history", []), maxlen=100)
                portfolio = data.get("portfolio", portfolio)
                logger.info("Veriler başarıyla yüklendi")
        except Exception as e:
            logger.error(f"Load error: {e}")

MEXC_BASE = "https://contract.mexc.com/api/v1/contract"

# Dynamic list of MEXC Futures pairs
FUTURES_PAIRS = []

# ── MEXC API helpers ──────────────────────────────────────────────────────────
async def fetch_all_futures_pairs(client: httpx.AsyncClient) -> List[str]:
    """Fetch all active USDT-settled futures pairs from MEXC"""
    try:
        r = await client.get(f"{MEXC_BASE}/detail", timeout=10)
        if r.status_code == 200:
            d = r.json()
            if d.get("success") and d.get("data"):
                # Filter for active (state=0) symbols settled in USDT
                pairs = [
                    item["symbol"] for item in d["data"]
                    if item.get("state") == 0 and item.get("settleCoin") == "USDT"
                ]
                logger.info(f"MEXC'den {len(pairs)} aktif pair çekildi.")
                return pairs
    except Exception as e:
        logger.error(f"Pair fetch error: {e}")
    return []

async def fetch_ticker(client: httpx.AsyncClient, symbol: str) -> Optional[dict]:
    """Fetch real-time ticker from MEXC"""
    try:
        r = await client.get(f"{MEXC_BASE}/ticker", params={"symbol": symbol}, timeout=5)
        if r.status_code == 200:
            d = r.json()
            if d.get("success") and d.get("data"):
                return d["data"]
    except Exception as e:
        logger.debug(f"Ticker error {symbol}: {e}")
    return None

async def fetch_klines(client: httpx.AsyncClient, symbol: str, interval: str = "Min15", limit: int = 100) -> Optional[dict]:
    """Fetch OHLCV klines for ML features"""
    try:
        r = await client.get(
            f"{MEXC_BASE}/kline/{symbol}",
            params={"interval": interval, "limit": limit},
            timeout=8
        )
        if r.status_code == 200:
            d = r.json()
            if d.get("success") and d.get("data"):
                rows = d["data"]
                if rows and isinstance(rows, dict):
                    # MEXC returns dict with lists
                    time_list = rows.get("time", [])
                    open_list = rows.get("open", [])
                    close_list = rows.get("close", [])
                    high_list = rows.get("high", [])
                    low_list = rows.get("low", [])
                    vol_list = rows.get("vol", [])
                    if len(close_list) > 20:
                        try:
                            kline_data = {
                                "timestamp": time_list,
                                "open": np.array([float(x) for x in open_list]),
                                "high": np.array([float(x) for x in high_list]),
                                "low": np.array([float(x) for x in low_list]),
                                "close": np.array([float(x) for x in close_list]),
                                "volume": np.array([float(x) for x in vol_list]),
                            }
                            return kline_data
                        except (ValueError, TypeError) as e:
                            logger.warning(f"Kline data conversion error for {symbol}: {e}")
    except Exception as e:
        logger.debug(f"Kline error {symbol}: {e}")
    return None

# ── Lifespan ──────────────────────────────────────────────────────────────────

async def auto_train_on_startup():
    """Sistem açılışında ilk scan tamamlandıktan sonra otomatik model eğit"""
    await asyncio.sleep(30)  # Wait for pairs to be fetched
    logger.info("🤖 Otomatik model eğitimi başlıyor...")
    all_X, all_y = [], []
    async with httpx.AsyncClient() as client:
        targets = FUTURES_PAIRS[:12] if FUTURES_PAIRS else ["BTC_USDT", "ETH_USDT", "SOL_USDT"]
        for sym in targets:
            try:
                klines = await fetch_klines(client, sym, interval="Min15", limit=200)
                if klines is None: continue
                from ml_engine import FeatureBuilder
                X, y = FeatureBuilder.build_dataset(klines, lookahead=5, threshold=0.008)
                if X is not None and len(X) >= 20:
                    all_X.append(X); all_y.append(y)
                    logger.info(f"  {sym}: {len(X)} sample eklendi")
                await asyncio.sleep(1)
            except Exception as e:
                logger.warning(f"  {sym} eğitim verisi alınamadı: {e}")
    if all_X:
        import numpy as np
        Xall = np.vstack(all_X); yall = np.concatenate(all_y)
        logger.info(f"Model eğitiliyor — toplam {len(Xall)} sample, {len(all_X)} pair")
        ml_engine.gbm.fit(Xall, yall)
        ml_engine.rf.fit(Xall, yall)
        ml_engine._trained = True
        # Walk-forward
        from ml_engine import walk_forward_validate
        ml_engine._wf = walk_forward_validate(Xall, yall, n_splits=4)
        logger.info(f"✅ Model hazır — WF Accuracy: {ml_engine._wf.get('accuracy',0)}%, F1: {ml_engine._wf.get('f1',0)}")
        # Backtest BTC ile
        async with httpx.AsyncClient() as client:
            klines_bt = await fetch_klines(client, "BTC_USDT", interval="Min15", limit=300)
            if klines_bt:
                ml_engine.run_backtest(klines_bt)
    else:
        logger.warning("Otomatik eğitim başarısız — MEXC API erişilemiyor olabilir")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 MEXC ML Trading System başlatılıyor...")
    load_data()
    asyncio.create_task(scanner_loop())
    asyncio.create_task(broadcast_loop())
    asyncio.create_task(auto_train_on_startup())
    yield
    save_data()
    logger.info("Sistem kapatılıyor...")

app = FastAPI(title="MEXC ML Trader", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Background tasks ───────────────────────────────────────────────────────────
async def scanner_loop():
    """Continuously scan MEXC futures pairs and run ML predictions"""
    global FUTURES_PAIRS
    logger.info("Scanner loop başladı")
    while True:
        try:
            async with httpx.AsyncClient() as client:
                # Refresh pairs list occasionally
                new_pairs = await fetch_all_futures_pairs(client)
                if new_pairs:
                    FUTURES_PAIRS = new_pairs

                if not FUTURES_PAIRS:
                    await asyncio.sleep(10)
                    continue

                # Scan in batches of 10
                for i in range(0, len(FUTURES_PAIRS), 10):
                    batch = FUTURES_PAIRS[i:i+10]
                    tasks = [process_pair(client, sym) for sym in batch]
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    for sym, result in zip(batch, results):
                        if isinstance(result, dict):
                            scanner_cache[sym] = result
                    await asyncio.sleep(0.5)
            logger.info(f"Tarama tamamlandı — {len(scanner_cache)} pair işlendi")
        except Exception as e:
            logger.error(f"Scanner loop error: {e}")
        await asyncio.sleep(30)  # Re-scan every 30s

async def process_pair(client: httpx.AsyncClient, symbol: str) -> dict:
    """Fetch data + compute indicators + run ML for one pair"""
    ticker = await fetch_ticker(client, symbol)
    klines = await fetch_klines(client, symbol)

    # Base data
    price = float(ticker.get("lastPrice", 0)) if ticker else 0
    change24h = float(ticker.get("riseFallRate", 0)) * 100 if ticker else 0
    volume24h = float(ticker.get("volume24", 0)) if ticker else 0

    # ML prediction
    prediction = ml_engine.predict(symbol, klines, price)

    # Logic for SL/TP based on side - Increased precision to 10 decimals for low-priced assets
    if prediction["signal"] == "SHORT":
        sl = round(price * (1 + 0.025), 10)
        tp = round(price * (1 - 0.045), 10)
    else:
        sl = round(price * (1 - 0.025), 10)
        tp = round(price * (1 + 0.045), 10)

    st = agent_states.get(symbol, {"pnl": 0.0, "trades": 0, "wins": 0, "active_pos": None})

    result = {
        "symbol": symbol,
        "price": price,
        "change24h": round(change24h, 2),
        "volume24h": round(volume24h, 2),
        "signal": prediction["signal"],
        "confidence": prediction["confidence"],
        "indicators": prediction["indicators"],
        "model_used": prediction["model"],
        "entry_price": price,
        "stop_loss": sl if price > 0 else 0,
        "take_profit": tp if price > 0 else 0,
        "leverage": prediction["leverage"],
        "gbm_signal": prediction.get("gbm_signal", "WAIT"),
        "rf_signal": prediction.get("rf_signal", "WAIT"),
        "model_trained": prediction.get("model_trained", False),
        "wf_accuracy": prediction.get("wf_accuracy", 0),
        "backtest_roi": prediction.get("backtest_roi", 0),
        "backtest_sharpe": prediction.get("backtest_sharpe", 0),
        "timestamp": datetime.utcnow().isoformat(),
        "data_source": "real" if klines is not None else "simulated",
        "active_pos": st.get("active_pos")
    }

    # Update agent state (track positions and PnL)
    if symbol not in agent_states:
        agent_states[symbol] = {
            "pnl": 0.0,
            "trades": 0,
            "wins": 0,
            "active_pos": None, # Stores entry details
            "last_exit_time": None
        }

    st = agent_states[symbol]

    # Check for new entry - Strict real data only + Cooldown + Distance check + Limits
    can_enter = st["active_pos"] is None and prediction["signal"] in ["LONG", "SHORT"] and price > 0 and prediction["data_quality"] == "real"

    # Limit checks
    if can_enter:
        active_count = sum(1 for sym in agent_states if agent_states[sym].get("active_pos") is not None)
        if active_count >= MAX_OPEN_POSITIONS:
            can_enter = False
            logger.info(f"Entry skipped for {symbol}: MAX_OPEN_POSITIONS reached ({active_count})")
        elif volume24h < MIN_VOLUME_24H:
            can_enter = False
            logger.info(f"Entry skipped for {symbol}: Volume too low (${volume24h:,.0f} < ${MIN_VOLUME_24H:,.0f})")

    # Cooldown check (5 mins)
    if can_enter and st["last_exit_time"]:
        cooldown_diff = (datetime.utcnow() - st["last_exit_time"]).total_seconds()
        if cooldown_diff < 300: # 5 minutes
            can_enter = False
            logger.debug(f"Cooldown active for {symbol}: {300 - cooldown_diff:.0f}s left")

    # Safety distance check (TP/SL must be at least 0.1% away)
    if can_enter:
        tp_dist = abs(result["take_profit"] - price) / price
        sl_dist = abs(result["stop_loss"] - price) / price
        if tp_dist < 0.001 or sl_dist < 0.001:
            can_enter = False
            logger.warning(f"Entry skipped for {symbol}: TP/SL too close to price ({tp_dist:.4%}/{sl_dist:.4%})")

    if can_enter:
        leverage = prediction["leverage"]
        tp = result["take_profit"]
        sl = result["stop_loss"]
        # TR Time (UTC+3)
        entry_time_raw = datetime.utcnow()
        tr_time = (entry_time_raw + timedelta(hours=3)).strftime("%d.%m.%Y %H:%M:%S")

        st["active_pos"] = {
            "entry_price": price,
            "side": prediction["signal"],
            "leverage": leverage,
            "size": 100 * leverage, # Assuming 100 USDT base size
            "timestamp": tr_time,
            "entry_time_raw": entry_time_raw,
            "tp": tp,
            "sl": sl,
            "indicators": ", ".join(prediction["indicators"])
        }

        # Format prices with high precision
        f_price = f"{price:.10g}"
        f_tp = f"{tp:.10g}"
        f_sl = f"{sl:.10g}"

        entry_fee = (100 * leverage) * 0.0006
        portfolio["total_fees"] += entry_fee
        portfolio["capital"] -= entry_fee

        msg = (
            f"<b>🚀 İŞLEME GİRİLDİ: {symbol}</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"📍 Giriş Fiyatı: <code>{f_price}</code>\n"
            f"↕️ Yön: <b>{prediction['signal']}</b>\n"
            f"⚙️ Kaldıraç: <b>{leverage}x</b>\n"
            f"💰 İşlem Büyüklüğü: <b>${100 * leverage}</b>\n"
            f"💸 Giriş Komisyonu: <b>-${entry_fee:.4f}</b>\n"
            f"🎯 TP: <code>{f_tp}</code>\n"
            f"🛑 SL: <code>{f_sl}</code>\n"
            f"🕒 Zaman: <code>{tr_time}</code> (TR)\n"
            f"🏦 Kasa: <b>${portfolio['capital']:.2f}</b>\n"
            f"🔍 Koşullar: <i>{st['active_pos']['indicators']}</i>"
        )
        asyncio.create_task(send_telegram_message(msg))
        save_data()

    # Monitoring active position for exit using real price
    if st["active_pos"] is not None:
        pos = st["active_pos"]

        # Calculate Unrealized PnL for live updates
        entry_p = pos["entry_price"]
        side = pos["side"]
        lev = pos["leverage"]
        size = pos["size"]

        if side == "LONG":
            diff_pct = (price - entry_p) / entry_p
        else:
            diff_pct = (entry_p - price) / entry_p

        unrealized_pnl = size * diff_pct
        pos["unrealized_pnl"] = round(unrealized_pnl, 2)
        pos["current_price"] = price

        close_pos = False
        exit_reason = ""

        # Check TP/SL conditions
        if pos["side"] == "LONG":
            if price >= pos["tp"]:
                close_pos = True
                exit_reason = "🎯 TAKE PROFIT"
            elif price <= pos["sl"]:
                close_pos = True
                exit_reason = "🛑 STOP LOSS"
        elif pos["side"] == "SHORT":
            if price <= pos["tp"]:
                close_pos = True
                exit_reason = "🎯 TAKE PROFIT"
            elif price >= pos["sl"]:
                close_pos = True
                exit_reason = "🛑 STOP LOSS"

        # Removed signal reversal exit logic as per user request
        # Trades now only exit on TP or SL levels

        if close_pos and price > 0:
            entry_price = pos["entry_price"]
            leverage = pos["leverage"]
            size = pos["size"] # Total leveraged size

            # PnL Calculation
            if pos["side"] == "LONG":
                price_diff_pct = (price - entry_price) / entry_price
            else:
                price_diff_pct = (entry_price - price) / entry_price

            trade_pnl = size * price_diff_pct

            # Fee calculation (0.06% for entry + 0.06% for exit)
            entry_val = size
            exit_val = size * (1 + price_diff_pct)
            total_fee = (entry_val + exit_val) * 0.0006
            net_pnl = trade_pnl - total_fee

            st["pnl"] += net_pnl
            st["trades"] += 1
            if net_pnl > 0:
                st["wins"] += 1

            # Update global portfolio
            exit_fee = exit_val * 0.0006
            portfolio["total_fees"] += exit_fee
            # net_pnl already subtracted both fees from trade_pnl,
            # but we already subtracted entry_fee from capital on entry.
            # net_pnl = trade_pnl - (entry_fee + exit_fee)
            # capital_change = trade_pnl - exit_fee
            capital_change = trade_pnl - exit_fee
            portfolio["capital"] += capital_change

            pnl_emoji = "🟢" if net_pnl > 0 else "🔴"
            exit_time_raw = datetime.utcnow()
            tr_time_exit = (exit_time_raw + timedelta(hours=3)).strftime("%d.%m.%Y %H:%M:%S")

            # Duration calculation
            diff = exit_time_raw - pos["entry_time_raw"]
            seconds = int(diff.total_seconds())
            hours, remainder = divmod(seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            duration_str = f"{hours}sa {minutes}dk {seconds}sn" if hours > 0 else f"{minutes}dk {seconds}sn"

            curr_indicators = ", ".join(prediction["indicators"])

            # Format prices with higher precision for display
            f_entry = f"{entry_price:.10g}"
            f_exit = f"{price:.10g}"

            msg = (
                f"<b>✅ İŞLEM KAPATILDI: {symbol}</b> ({exit_reason})\n"
                f"━━━━━━━━━━━━━━━━━━━━━\n"
                f"📥 Giriş Fiyatı: <code>{f_entry}</code>\n"
                f"📤 Çıkış Fiyatı: <code>{f_exit}</code>\n"
                f"💸 Net Komisyon: <b>-${total_fee:.4f}</b>\n"
                f"💵 Net PnL: <b>{pnl_emoji} ${net_pnl:.2f}</b>\n"
                f"📈 Toplam Sembol PnL: <b>${st['pnl']:.2f}</b>\n"
                f"🕒 Zaman: <code>{tr_time_exit}</code> (TR)\n"
                f"⏳ İşlem Süresi: <b>{duration_str}</b>\n"
                f"🏦 Güncel Kasa: <b>${portfolio['capital']:.2f}</b>\n"
                f"🔍 Çıkış Koşulları: <i>{curr_indicators}</i>"
            )
            asyncio.create_task(send_telegram_message(msg))
            save_data()

            # Add to history
            trade_history.appendleft({
                "symbol": symbol,
                "entry_price": entry_price,
                "exit_price": price,
                "side": pos["side"],
                "pnl": net_pnl,
                "reason": exit_reason,
                "duration": duration_str,
                "time": tr_time_exit
            })

            st["active_pos"] = None
            st["last_exit_time"] = exit_time_raw # For cooldown

    result["pnl"] = round(st["pnl"], 2)
    result["trades"] = st["trades"]
    result["win_rate"] = round(st["wins"] / st["trades"] * 100, 1) if st["trades"] > 0 else 0

    return result

async def broadcast_loop():
    """Send updates to all connected WebSocket clients"""
    await asyncio.sleep(5)  # Wait for first scan
    while True:
        if active_connections and scanner_cache:
            msg = json.dumps({
                "type": "update",
                "data": list(scanner_cache.values()),
                "portfolio": portfolio,
                "timestamp": datetime.utcnow().isoformat(),
                "total_pairs": len(scanner_cache),
            })
            dead = []
            # Use a copy to avoid RuntimeError: list changed size during iteration
            for ws in active_connections[:]:
                try:
                    await ws.send_text(msg)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                if ws in active_connections:
                    active_connections.remove(ws)
        await asyncio.sleep(5)

# ── REST Endpoints ─────────────────────────────────────────────────────────────
@app.get("/api/scan")
async def get_scan():
    """Get all scanned pairs with ML signals"""
    return {
        "success": True,
        "data": list(scanner_cache.values()),
        "total": len(scanner_cache),
        "timestamp": datetime.utcnow().isoformat(),
    }

@app.get("/api/pair/{symbol}")
async def get_pair(symbol: str):
    """Get detailed info for one pair"""
    sym = symbol.upper().replace("-", "_")
    if sym in scanner_cache:
        return {"success": True, "data": scanner_cache[sym]}
    return {"success": False, "error": "Pair not found"}

@app.get("/api/stats")
async def get_stats():
    """Aggregate stats across all agents"""
    if not scanner_cache:
        return {"success": True, "data": {}}

    vals = list(scanner_cache.values())
    total_pnl = sum(agent_states.get(v["symbol"], {}).get("pnl", 0) for v in vals)
    total_trades = sum(agent_states.get(v["symbol"], {}).get("trades", 0) for v in vals)
    total_wins = sum(agent_states.get(v["symbol"], {}).get("wins", 0) for v in vals)

    # Active positions based on real open trades, not just signals
    active_longs = sum(1 for v in vals if v.get("active_pos") and v["active_pos"]["side"] == "LONG")
    active_shorts = sum(1 for v in vals if v.get("active_pos") and v["active_pos"]["side"] == "SHORT")

    # Counts of closed trades per side
    closed_longs = sum(1 for h in trade_history if h["side"] == "LONG")
    closed_shorts = sum(1 for h in trade_history if h["side"] == "SHORT")

    return {
        "success": True,
        "data": {
            "total_pairs": len(vals),
            "total_pnl": round(total_pnl, 2),
            "total_trades": total_trades,
            "win_rate": round(total_wins / total_trades * 100, 1) if total_trades > 0 else 0,
            "active_longs": active_longs,
            "active_shorts": active_shorts,
            "closed_longs": closed_longs,
            "closed_shorts": closed_shorts,
            "portfolio": portfolio,
            "model_accuracy": round(ml_engine.get_accuracy(), 1),
            "timestamp": datetime.utcnow().isoformat(),
        }
    }

@app.get("/api/model")
async def get_model_info():
    """ML model metadata"""
    return {
        "success": True,
        "data": ml_engine.get_info()
    }

@app.get("/api/history")
async def get_trade_history():
    """Get closed trade history"""
    return {
        "success": True,
        "data": list(trade_history)
    }

@app.post("/api/train/{symbol}")
async def trigger_training(symbol: str):
    """Elle model eğitimi tetikle — gerçek MEXC verisiyle"""
    sym = symbol.upper().replace("-","_")
    if sym not in scanner_cache:
        return {"success":False,"error":"Pair bulunamadı — önce scan çalıştır"}
    async with httpx.AsyncClient() as client:
        klines = await fetch_klines(client, sym, interval="Min15", limit=200)
    if klines is None:
        return {"success":False,"error":"MEXC kline verisi alınamadı"}
    result = ml_engine.train(klines, sym)
    return {"success":True,"data":result}

@app.post("/api/train_all")
async def train_all():
    """İlk 5 pairi kullanarak global modeli eğit"""
    trained = []
    all_X, all_y = [], []
    async with httpx.AsyncClient() as client:
        for sym in FUTURES_PAIRS[:5]:
            klines = await fetch_klines(client, sym, interval="Min15", limit=200)
            if klines is None: continue
            from ml_engine import FeatureBuilder
            X, y = FeatureBuilder.build_dataset(klines, lookahead=5, threshold=0.008)
            if X is not None:
                all_X.append(X); all_y.append(y)
                trained.append(sym)
    if all_X:
        import numpy as np
        Xall = np.vstack(all_X); yall = np.concatenate(all_y)
        ml_engine.gbm.fit(Xall, yall)
        ml_engine.rf.fit(Xall, yall)
        ml_engine._trained = True
        logger.info(f"Global model eğitildi — {len(Xall)} sample, {len(trained)} pair")
        return {"success":True,"pairs":trained,"n_samples":len(Xall)}
    return {"success":False,"error":"Veri alınamadı"}

@app.get("/api/backtest/{symbol}")
async def run_backtest(symbol: str):
    """Seçili pair üzerinde backtest çalıştır"""
    sym = symbol.upper().replace("-","_")
    async with httpx.AsyncClient() as client:
        klines = await fetch_klines(client, sym, interval="Min15", limit=300)
    if klines is None:
        return {"success":False,"error":"Veri alınamadı"}
    result = ml_engine.run_backtest(klines)
    return {"success":True,"symbol":sym,"data":result}

@app.get("/api/backtest_all")
async def run_backtest_all():
    """Global backtest durumu"""
    return {"success":True,"data":ml_engine._bt}

@app.get("/health")
async def health_check():
    """Render health check to keep service alive"""
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "total_pairs": len(scanner_cache)
    }

# ── WebSocket ──────────────────────────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    logger.info(f"WebSocket bağlandı — toplam: {len(active_connections)}")
    try:
        # Send current data immediately
        if scanner_cache:
            await websocket.send_text(json.dumps({
                "type": "init",
                "data": list(scanner_cache.values()),
                "portfolio": portfolio,
                "timestamp": datetime.utcnow().isoformat(),
            }))
        while True:
            await websocket.receive_text()  # Keep alive
    except (WebSocketDisconnect, Exception):
        if websocket in active_connections:
            active_connections.remove(websocket)
        logger.info(f"WebSocket ayrıldı — kalan: {len(active_connections)}")

# ── Static frontend ────────────────────────────────────────────────────────────
import os
frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(frontend_path):
    app.mount("/static", StaticFiles(directory=frontend_path), name="static")

@app.get("/ml-details")
async def serve_ml_details():
    idx = os.path.join(frontend_path, "ml_details.html")
    if os.path.exists(idx):
        return FileResponse(idx)
    return {"message": "ml_details.html bulunamadı"}

@app.get("/")
async def serve_frontend():
    idx = os.path.join(frontend_path, "index.html")
    if os.path.exists(idx):
        return FileResponse(idx)
    return {"message": "MEXC ML Trader API çalışıyor", "docs": "/docs"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
