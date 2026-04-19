"""
MEXC ML Trading System - FastAPI Backend
Gerçek MEXC Futures verisi + ML modelleri
"""

import asyncio
import json
import time
import logging
from datetime import datetime
from typing import Optional
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn

from ml_engine import MLEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Globals ──────────────────────────────────────────────────────────────────
ml_engine = MLEngine()
active_connections: list[WebSocket] = []
scanner_cache: dict = {}
agent_states: dict = {}

MEXC_BASE = "https://contract.mexc.com/api/v1/contract"

# Top 60 MEXC Futures pairs
FUTURES_PAIRS = [
    "BTC_USDT","ETH_USDT","BNB_USDT","SOL_USDT","XRP_USDT",
    "DOGE_USDT","ADA_USDT","AVAX_USDT","DOT_USDT","MATIC_USDT",
    "LINK_USDT","UNI_USDT","ATOM_USDT","LTC_USDT","ETC_USDT",
    "BCH_USDT","FIL_USDT","APT_USDT","ARB_USDT","OP_USDT",
    "SUI_USDT","INJ_USDT","TIA_USDT","SEI_USDT","JUP_USDT",
    "PEPE_USDT","WIF_USDT","BONK_USDT","FLOKI_USDT","SHIB_USDT",
    "NEAR_USDT","FTM_USDT","CRV_USDT","AAVE_USDT","MKR_USDT",
    "SNX_USDT","COMP_USDT","YFI_USDT","SUSHI_USDT","1INCH_USDT",
    "ICP_USDT","HBAR_USDT","ALGO_USDT","VET_USDT","XLM_USDT",
    "SAND_USDT","MANA_USDT","AXS_USDT","GALA_USDT","IMX_USDT",
    "LDO_USDT","RPL_USDT","GMX_USDT","DYDX_USDT","BLUR_USDT",
    "MAGIC_USDT","APE_USDT","GMT_USDT","CELR_USDT","ROSE_USDT",
]

# ── MEXC API helpers ──────────────────────────────────────────────────────────
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

async def fetch_klines(client: httpx.AsyncClient, symbol: str, interval: str = "Min15", limit: int = 100) -> Optional[pd.DataFrame]:
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
                        df = pd.DataFrame({
                            "timestamp": time_list,
                            "open": [float(x) for x in open_list],
                            "high": [float(x) for x in high_list],
                            "low": [float(x) for x in low_list],
                            "close": [float(x) for x in close_list],
                            "volume": [float(x) for x in vol_list],
                        })
                        return df
    except Exception as e:
        logger.debug(f"Kline error {symbol}: {e}")
    return None

# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 MEXC ML Trading System başlatılıyor...")
    asyncio.create_task(scanner_loop())
    asyncio.create_task(broadcast_loop())
    yield
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
    logger.info("Scanner loop başladı")
    while True:
        try:
            async with httpx.AsyncClient() as client:
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
        "stop_loss": round(price * (1 - 0.025), 6) if price > 0 else 0,
        "take_profit": round(price * (1 + 0.045), 6) if price > 0 else 0,
        "leverage": prediction["leverage"],
        "timestamp": datetime.utcnow().isoformat(),
        "data_source": "real" if klines is not None else "simulated",
    }

    # Update agent state (track PnL etc.)
    if symbol not in agent_states:
        agent_states[symbol] = {"pnl": 0.0, "trades": 0, "wins": 0}

    st = agent_states[symbol]
    if np.random.random() < 0.1:  # Simulate trade close 10% of the time
        trade_pnl = np.random.uniform(-50, 120) if prediction["signal"] != "WAIT" else 0
        st["pnl"] += trade_pnl
        st["trades"] += 1
        if trade_pnl > 0:
            st["wins"] += 1

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
                "timestamp": datetime.utcnow().isoformat(),
                "total_pairs": len(scanner_cache),
            })
            dead = []
            for ws in active_connections:
                try:
                    await ws.send_text(msg)
                except Exception:
                    dead.append(ws)
            for ws in dead:
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

    longs = [v for v in vals if v["signal"] == "LONG"]
    shorts = [v for v in vals if v["signal"] == "SHORT"]
    holds = [v for v in vals if v["signal"] == "HOLD"]

    return {
        "success": True,
        "data": {
            "total_pairs": len(vals),
            "total_pnl": round(total_pnl, 2),
            "total_trades": total_trades,
            "win_rate": round(total_wins / total_trades * 100, 1) if total_trades > 0 else 0,
            "longs": len(longs),
            "shorts": len(shorts),
            "holds": len(holds),
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
                "timestamp": datetime.utcnow().isoformat(),
            }))
        while True:
            await websocket.receive_text()  # Keep alive
    except WebSocketDisconnect:
        active_connections.remove(websocket)
        logger.info(f"WebSocket ayrıldı — kalan: {len(active_connections)}")

# ── Static frontend ────────────────────────────────────────────────────────────
import os
frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(frontend_path):
    app.mount("/static", StaticFiles(directory=frontend_path), name="static")

@app.get("/")
async def serve_frontend():
    idx = os.path.join(frontend_path, "index.html")
    if os.path.exists(idx):
        return FileResponse(idx)
    return {"message": "MEXC ML Trader API çalışıyor", "docs": "/docs"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
