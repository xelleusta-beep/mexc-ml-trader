import asyncio
import gc
import json
import time
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import os
import hashlib
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from config import config
from wallet_tracker import scan_wallets, get_wallet_summary, get_recent_txs
from news_tracker import fetch_news, get_news_feed, get_sentiment_summary, check_symbol_news
from signal_engine import signal_engine
from trade_executor import execute_signal, check_positions, get_summary, active_positions, trade_history, portfolio

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_AUTH_TOKEN = os.getenv("AUTH_TOKEN", "")
if not _AUTH_TOKEN:
    _AUTH_TOKEN = hashlib.sha256(os.urandom(16)).hexdigest()[:12]
    logger.info("AUTH_TOKEN: %s", _AUTH_TOKEN)


# ── Telegram ───────────────────────────────────────────────────────────────
async def send_tg(text):
    if not config.TG_TOKEN or not config.TG_CHAT:
        return
    try:
        import httpx
        async with httpx.AsyncClient(timeout=10) as c:
            await c.post(
                f"https://api.telegram.org/bot{config.TG_TOKEN}/sendMessage",
                json={"chat_id": config.TG_CHAT, "text": text, "parse_mode": "HTML"},
            )
    except Exception as e:
        logger.error(f"TG error: {e}")


# ── Scanner Loop ───────────────────────────────────────────────────────────
async def main_loop():
    await asyncio.sleep(10)
    logger.info("Main loop started")

    wallet_counter = 0
    news_counter = 0

    while True:
        try:
            import httpx
            async with httpx.AsyncClient(timeout=15) as client:
                # Cüzdan tara (her SCAN_INTERVAL)
                wallet_counter += 1
                if wallet_counter >= config.SCAN_INTERVAL / min(config.TRADING_INTERVAL, 60):
                    wallet_counter = 0
                    if config.TRACKED_WALLETS.get("ethereum") or config.TRACKED_WALLETS.get("bsc"):
                        wallet_sigs = await scan_wallets(client)
                        if wallet_sigs:
                            signal_engine.add_wallet_signal(wallet_sigs)
                            for sig in wallet_sigs[:3]:
                                msg = f"Balina: {sig['label']} | {sig.get('direction','')} {sig.get('diff_eth',sig.get('value_eth',''))}"
                                logger.info(msg)
                                asyncio.create_task(send_tg(f"🐋 {msg}"))

                # Haber tara (her NEWS_INTERVAL)
                news_counter += 1
                if news_counter >= config.NEWS_INTERVAL / min(config.TRADING_INTERVAL, 60):
                    news_counter = 0
                    await fetch_news(client)
                    for sym in config.TRACKED_SYMBOLS:
                        sentiment = await check_symbol_news(client, sym)
                        if abs(sentiment) > 0.3:
                            signal_engine.add_news_signal(sym, sentiment)

                # Sinyalleri hesapla ve işlem yap
                signals = signal_engine.get_all_signals()
                for sig in signals:
                    pos = await execute_signal(client, sig)
                    if pos:
                        asyncio.create_task(send_tg(
                            f"<b>ISLEM: {sig['symbol']}</b>\n"
                            f"Yon: {sig['signal']} | Skor: {sig['score']}\n"
                            f"Fiyat: ${pos['entry_price']:.2f} | {pos['leverage']}x"
                        ))

                # Pozisyon kontrol
                closed = await check_positions(client)
                for c in closed:
                    asyncio.create_task(send_tg(
                        f"<b>KAPANDI: {c['symbol']}</b>\n"
                        f"PnL: ${c['pnl']:.2f} | {c['reason']}"
                    ))

        except Exception as e:
            logger.error(f"Loop error: {e}")

        await asyncio.sleep(config.TRADING_INTERVAL)


# ── GC Loop ────────────────────────────────────────────────────────────────
async def gc_loop():
    while True:
        await asyncio.sleep(180)
        gc.collect()


# ── Lifespan ───────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Whale Trader basliyor...")
    task = asyncio.create_task(main_loop())
    asyncio.create_task(gc_loop())
    yield
    task.cancel()
    _save_state()
    logger.info("Shutdown")


def _save_state():
    try:
        os.makedirs(config.PERSIST_DIR, exist_ok=True)
        path = os.path.join(config.PERSIST_DIR, "state.json")
        state = {
            "portfolio": portfolio,
            "trade_history": list(trade_history),
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(path, "w") as f:
            json.dump(state, f, default=str)
    except Exception as e:
        logger.error(f"Save error: {e}")


# ── FastAPI ────────────────────────────────────────────────────────────────
app = FastAPI(title="Whale Trader Bot", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.middleware("http")
async def auth(request: Request, call_next):
    if request.method == "GET" or request.url.path.startswith("/static"):
        return await call_next(request)
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    if token != _AUTH_TOKEN:
        return JSONResponse(status_code=401, content={"error": "Unauthorized"})
    return await call_next(request)


# ── Frontend ────────────────────────────────────────────────────────────────
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")


@app.get("/")
async def root():
    idx = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(idx):
        return FileResponse(idx)
    return {"status": "ok"}


# ── API ─────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "positions": len(active_positions),
        "trades": portfolio["total_closed_trades"],
    }


@app.get("/api/status")
async def status():
    s = get_summary()
    sigs = signal_engine.get_summary()
    return {**s, "signals": sigs}


@app.get("/api/positions")
async def positions():
    return {"positions": list(active_positions.values())}


@app.get("/api/history")
async def history():
    return {"trades": list(trade_history)}


@app.get("/api/wallets")
async def wallets():
    return {
        "wallets": get_wallet_summary(),
        "recent_txs": get_recent_txs(20),
    }


@app.get("/api/news")
async def news():
    return {
        "news": get_news_feed(20),
        "sentiment": get_sentiment_summary(),
    }


@app.get("/api/signals")
async def signals():
    return {"signals": signal_engine.get_summary()}
