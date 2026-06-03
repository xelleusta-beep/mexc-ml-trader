import asyncio
import gc
import json
import time
import logging
from collections import deque
from datetime import datetime, timedelta, timezone
from contextlib import asynccontextmanager

import os
import numpy as np
import hashlib
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from config import config
from ml_engine import MLEngine
from rl_engine import SimplePPO
from risk_manager import RiskManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Globals
ml_engine = MLEngine()
rl_agent = SimplePPO(
    state_dim=config.rl_state_dim,
    hidden_dim=config.rl_hidden_dim,
    lr=config.rl_learning_rate,
)
risk_mgr = RiskManager(config)

active_connections = []
scanner_cache = {}
_klines_cache = {}
agent_states = {}
trade_history = deque(maxlen=200)
_pnl_timeline = []
FUTURES_PAIRS = []

portfolio = {
    "capital": 5000.0,
    "initial_capital": 5000.0,
    "total_fees": 0.0,
    "total_closed_trades": 0,
    "total_closed_notional": 0.0,
}

MEXC_BASE = config.mexc_base_url
PERSISTENCE_FILE = config.persistence_file
ML_MODEL_FILE = config.ml_model_file
RL_MODEL_FILE = config.rl_model_file
state_lock = asyncio.Lock()


# ── Telegram ───────────────────────────────────────────────────────────────
async def send_tg(text):
    if not config.telegram_bot_token or not config.telegram_chat_id:
        return
    try:
        import httpx
        async with httpx.AsyncClient(timeout=10) as client:
            await client.post(
                f"https://api.telegram.org/bot{config.telegram_bot_token}/sendMessage",
                json={"chat_id": config.telegram_chat_id, "text": text, "parse_mode": "HTML"},
            )
    except Exception as e:
        logger.error(f"TG error: {e}")


# ── MEXC API ───────────────────────────────────────────────────────────────
async def fetch_pairs(client):
    for retry in range(3):
        try:
            r = await client.get(f"{MEXC_BASE}/detail", timeout=10)
            if r.status_code == 200:
                d = r.json()
                if d.get("success") and d.get("data"):
                    pairs = [x["symbol"] for x in d["data"] if x.get("state") == 0 and x.get("settleCoin") == "USDT"]
                    return pairs
            elif r.status_code == 429:
                await asyncio.sleep(min(60 * (retry + 1), 180))
        except Exception as e:
            logger.error(f"fetch_pairs error: {e}")
            await asyncio.sleep(2**retry)
    return []


async def fetch_ticker(client, symbol):
    for retry in range(2):
        try:
            r = await client.get(f"{MEXC_BASE}/ticker", params={"symbol": symbol}, timeout=5)
            if r.status_code == 200:
                d = r.json()
                if d.get("success") and d.get("data"):
                    return d["data"]
        except Exception:
            if retry < 1:
                await asyncio.sleep(0.5)
    return None


async def fetch_klines(client, symbol):
    for retry in range(2):
        try:
            r = await client.get(f"{MEXC_BASE}/kline/{symbol}", params={"interval": config.kline_interval, "limit": config.kline_limit}, timeout=8)
            if r.status_code == 200:
                d = r.json()
                if d.get("success") and d.get("data"):
                    rows = d["data"]
                    if rows and isinstance(rows, dict):
                        cl = rows.get("close", [])
                        if len(cl) > 20:
                            return {
                                "timestamp": rows.get("time", []),
                                "open": np.array([float(x) for x in rows.get("open", [])]),
                                "high": np.array([float(x) for x in rows.get("high", [])]),
                                "low": np.array([float(x) for x in rows.get("low", [])]),
                                "close": np.array([float(x) for x in cl]),
                                "volume": np.array([float(x) for x in rows.get("vol", [])]),
                                "_symbol": symbol,
                            }
        except Exception:
            if retry < 1:
                await asyncio.sleep(0.5)
    return None


# ── Prediction ─────────────────────────────────────────────────────────────
def predict(symbol, klines, price, st):
    ml_pred = ml_engine.predict(klines, price)
    feat = ml_pred.get("_feat")

    pos_type = 0
    if st.get("active_pos"):
        pos_type = 1 if st["active_pos"]["side"] == "LONG" else -1

    unrealized = st["active_pos"].get("unrealized_pnl", 0.0) if st.get("active_pos") else 0.0
    pos_age = 0
    if st.get("active_pos") and st["active_pos"].get("entry_time_raw"):
        pos_age = int((datetime.now(timezone.utc) - st["active_pos"]["entry_time_raw"]).total_seconds() / 300)

    rl_pred = rl_agent.predict(feat, pos_type, unrealized, pos_age, 0.0)

    sig = ml_pred["signal"]
    conf = ml_pred["confidence"]
    lev = rl_pred["leverage"]

    if sig in ("LONG", "SHORT") and conf >= config.min_confidence_entry:
        return {"signal": sig, "confidence": conf, "leverage": lev, "source": "LightGBM", "_feat": feat}
    return {"signal": "WAIT", "confidence": 0, "leverage": 1, "source": "none", "_feat": feat}


# ── Save / Load ───────────────────────────────────────────────────────────
async def save_state():
    async with state_lock:
        try:
            now = datetime.now(timezone.utc)
            pnl = portfolio["capital"] - portfolio["initial_capital"]
            _pnl_timeline.append({"t": now.isoformat()[:16], "v": round(pnl, 2)})
            if len(_pnl_timeline) > 500:
                _pnl_timeline[:] = _pnl_timeline[-500:]

            state = {
                "agent_states": _serialize_states(agent_states),
                "trade_history": list(trade_history),
                "portfolio": portfolio,
                "risk_state": risk_mgr.save_state(),
                "pnl_timeline": _pnl_timeline,
            }

            def _write():
                with open(PERSISTENCE_FILE, "w") as f:
                    json.dump(state, f, default=str)
            await asyncio.to_thread(_write)
        except Exception as e:
            logger.error(f"Save error: {e}")


def _serialize_states(states):
    out = {}
    for sym, st in states.items():
        s = dict(st)
        if s.get("active_pos") and s["active_pos"].get("entry_time_raw"):
            s["active_pos"] = dict(s["active_pos"])
            s["active_pos"]["entry_time_raw"] = s["active_pos"]["entry_time_raw"].isoformat()
        if s.get("last_exit_time"):
            s["last_exit_time"] = s["last_exit_time"].isoformat()
        out[sym] = s
    return out


async def load_state():
    global agent_states, trade_history, portfolio, _pnl_timeline
    if not os.path.exists(PERSISTENCE_FILE):
        return
    async with state_lock:
        try:
            def _read():
                with open(PERSISTENCE_FILE) as f:
                    return json.load(f)
            data = await asyncio.to_thread(_read)
            risk_mgr.load_state(data.get("risk_state", {}))
            loaded = data.get("agent_states", {})
            for sym, st in loaded.items():
                if st.get("last_exit_time") and isinstance(st["last_exit_time"], str):
                    try:
                        st["last_exit_time"] = datetime.fromisoformat(st["last_exit_time"])
                    except:
                        st["last_exit_time"] = None
                if st.get("active_pos"):
                    pos = st["active_pos"]
                    if pos.get("entry_time_raw") and isinstance(pos["entry_time_raw"], str):
                        try:
                            pos["entry_time_raw"] = datetime.fromisoformat(pos["entry_time_raw"])
                        except:
                            pos["entry_time_raw"] = datetime.now(timezone.utc)
                    if not pos.get("tp") or not pos.get("sl") or pos.get("entry_price", 0) <= 0:
                        st["active_pos"] = None
            agent_states = loaded
            trade_history = deque(data.get("trade_history", []), maxlen=200)
            portfolio.update(data.get("portfolio", {}))
            _pnl_timeline = data.get("pnl_timeline", [])
            portfolio.setdefault("initial_capital", 5000)
            portfolio.setdefault("capital", 5000)
            logger.info(f"State loaded: {len(agent_states)} symbols")
        except Exception as e:
            logger.error(f"Load error: {e}")


# ── Scanner ────────────────────────────────────────────────────────────────
async def scanner_loop():
    global FUTURES_PAIRS
    while True:
        try:
            import httpx
            async with httpx.AsyncClient(timeout=15) as client:
                new_pairs = await fetch_pairs(client)
                if new_pairs:
                    FUTURES_PAIRS = new_pairs
                    logger.info(f"{len(new_pairs)} pairs")

                if not FUTURES_PAIRS:
                    await asyncio.sleep(10)
                    continue

                chunk_sz = min(config.scanner_batch_size * 2, 15)
                sem = asyncio.Semaphore(chunk_sz)

                async def bounded(sym):
                    async with sem:
                        return await process_pair(client, sym)

                scanned = 0
                for i in range(0, len(FUTURES_PAIRS), chunk_sz):
                    chunk = FUTURES_PAIRS[i:i + chunk_sz]
                    tasks = [bounded(sym) for sym in chunk]
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    for sym, res in zip(chunk, results):
                        if isinstance(res, dict):
                            scanner_cache[sym] = res
                            scanned += 1
                    if len(FUTURES_PAIRS) > chunk_sz:
                        await asyncio.sleep(0.3)

                # Cache limit
                if len(scanner_cache) > config.max_scanner_cache:
                    keys = list(scanner_cache.keys())
                    for k in keys[:len(keys) - config.max_scanner_cache]:
                        scanner_cache.pop(k, None)

                logger.info(f"Scan: {scanned}/{len(FUTURES_PAIRS)}")
        except Exception as e:
            logger.error(f"Scanner error: {e}")
        await asyncio.sleep(config.scanner_interval_sec)


async def process_pair(client, symbol):
    ticker = await fetch_ticker(client, symbol)
    klines = await fetch_klines(client, symbol)
    if klines is not None:
        _klines_cache[symbol] = klines
        if len(_klines_cache) > config.max_klines_cache:
            keys = list(_klines_cache.keys())
            for k in keys[:len(keys) - config.max_klines_cache]:
                _klines_cache.pop(k, None)

    price = float(ticker.get("lastPrice", 0)) if ticker else 0
    volume24h = float(ticker.get("volume24", 0)) if ticker else 0
    change24h = float(ticker.get("riseFallRate", 0)) * 100 if ticker else 0

    if price < config.min_price and klines is not None:
        try:
            price = float(klines["close"][-1])
        except:
            pass

    if symbol not in agent_states:
        agent_states[symbol] = {"pnl": 0, "trades": 0, "wins": 0, "active_pos": None, "last_exit_time": None, "entry_feat": None, "last_trade_pnl": 0}
    st = agent_states[symbol]

    prediction = predict(symbol, klines, price, st)
    sig = prediction["signal"]
    conf = prediction["confidence"]
    lev = prediction["leverage"]

    # Entry logic
    can_enter = (
        st["active_pos"] is None
        and sig in ("LONG", "SHORT")
        and price >= config.min_price
        and volume24h >= config.min_volume_24h
    )
    if can_enter:
        active_count = sum(1 for s in agent_states.values() if s.get("active_pos"))
        if active_count >= config.max_open_positions:
            can_enter = False
        elif st.get("last_exit_time"):
            cooldown = config.cooldown_loss_sec if st.get("last_trade_pnl", 0) < 0 else config.cooldown_win_sec
            elapsed = (datetime.now(timezone.utc) - st["last_exit_time"]).total_seconds()
            if elapsed < cooldown:
                can_enter = False
        else:
            risk_ok, _ = risk_mgr.check_entry_allowed(symbol, sig, conf, price, portfolio, agent_states)
            if not risk_ok:
                can_enter = False

    if can_enter:
        sl_pct = config.sl_pct / lev
        tp_pct = config.tp_pct / lev
        entry_time = datetime.now(timezone.utc)
        if sig == "SHORT":
            sl_price = round(price * (1 + sl_pct), 10)
            tp_price = round(price * (1 - tp_pct), 10)
        else:
            sl_price = round(price * (1 - sl_pct), 10)
            tp_price = round(price * (1 + tp_pct), 10)

        base_size = risk_mgr.calculate_position_size(symbol, price, conf, portfolio, agent_states)
        notional = base_size * lev
        fee = notional * config.fee_rate
        portfolio["total_fees"] += fee
        portfolio["capital"] -= fee

        st["entry_feat"] = prediction.get("_feat")
        st["active_pos"] = {
            "entry_price": price, "side": sig, "leverage": lev,
            "size": notional, "base_size": base_size,
            "entry_time_raw": entry_time, "tp": tp_price, "sl": sl_price,
        }
        tr_time = (entry_time + timedelta(hours=3)).strftime("%d.%m.%Y %H:%M:%S")
        asyncio.create_task(send_tg(
            f"<b>GIRIS: {symbol}</b>\n"
            f"Fiyat: {price:.6g}\n"
            f"Yon: {sig} | {lev}x\n"
            f"SL: {sl_price:.6g} | TP: {tp_price:.6g}\n"
            f"Kasa: ${portfolio['capital']:.2f}"
        ))
        await save_state()

    # Position tracking
    exit_pos = False
    exit_reason = ""
    if st["active_pos"] is not None:
        pos = st["active_pos"]
        ep = pos["entry_price"]
        side = pos["side"]
        sz = pos["size"]

        diff = (price - ep) / ep if side == "LONG" else (ep - price) / ep
        pos["unrealized_pnl"] = round(sz * diff, 2)
        pos["current_price"] = price

        if side == "LONG":
            if price >= pos["tp"]:
                exit_pos = True
                exit_reason = "TAKE_PROFIT"
            elif price <= pos["sl"]:
                exit_pos = True
                exit_reason = "STOP_LOSS"
        else:
            if price <= pos["tp"]:
                exit_pos = True
                exit_reason = "TAKE_PROFIT"
            elif price >= pos["sl"]:
                exit_pos = True
                exit_reason = "STOP_LOSS"

        if exit_pos:
            pnl = pos["unrealized_pnl"]
            exit_fee = sz * config.fee_rate
            net_pnl = pnl - exit_fee
            portfolio["capital"] += (sz + pnl - exit_fee)
            portfolio["total_fees"] += exit_fee
            portfolio["total_closed_trades"] += 1
            portfolio["total_closed_notional"] += sz

            st["pnl"] = st.get("pnl", 0) + net_pnl
            st["trades"] = st.get("trades", 0) + 1
            st["last_trade_pnl"] = net_pnl
            if net_pnl > 0:
                st["wins"] = st.get("wins", 0) + 1
            st["last_exit_time"] = datetime.now(timezone.utc)
            st["active_pos"] = None
            st["entry_feat"] = None

            risk_mgr.record_trade(net_pnl)
            trade_history.append({
                "symbol": symbol, "side": side, "entry": ep, "exit": price,
                "pnl": round(net_pnl, 2), "leverage": lev,
                "reason": exit_reason, "time": datetime.now(timezone.utc).isoformat(),
            })
            asyncio.create_task(send_tg(
                f"<b>CIKIS: {symbol}</b>\n"
                f"Kar/Zarar: ${net_pnl:.2f}\n"
                f"Sebep: {exit_reason}\n"
                f"Kasa: ${portfolio['capital']:.2f}"
            ))
            await save_state()

    return {"price": price, "signal": sig, "confidence": conf, "volume24h": volume24h}


# ── Training ───────────────────────────────────────────────────────────────
async def training_loop():
    await asyncio.sleep(30)
    while True:
        try:
            should, reason = ml_engine.should_retrain()
            if should and _klines_cache:
                klines_list = []
                for sym in list(_klines_cache.keys())[:10]:
                    kl = _klines_cache.get(sym)
                    if kl is not None and len(kl.get("close", [])) > 25:
                        klines_list.append(kl)
                if klines_list:
                    result = await asyncio.to_thread(ml_engine.train, klines_list[0])
                    if result.get("success"):
                        await asyncio.to_thread(ml_engine.save, ML_MODEL_FILE)
                        logger.info(f"Train OK: wf={result.get('wf_accuracy')}%")
                    else:
                        logger.warning(f"Train failed: {result}")
        except Exception as e:
            logger.error(f"Training error: {e}")
        await asyncio.sleep(config.retrain_interval_sec)


# ── HTTP Client (shared) ──────────────────────────────────────────────────
_http_client = None


async def get_http_client():
    import httpx
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(timeout=15)
    return _http_client


async def close_http():
    global _http_client
    if _http_client and not _http_client.is_closed:
        await _http_client.aclose()
    _http_client = None


# ── Lifespan ───────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("MEXC Scalper basliyor...")
    await load_state()
    if ml_engine.load(ML_MODEL_FILE):
        logger.info("ML model loaded")
    if rl_agent.load(RL_MODEL_FILE):
        logger.info("RL model loaded")

    loop = asyncio.get_event_loop()
    loop.create_task(scanner_loop())
    loop.create_task(training_loop())
    loop.create_task(_gc_loop())
    yield
    await save_state()
    await asyncio.to_thread(ml_engine.save, ML_MODEL_FILE)
    await asyncio.to_thread(rl_agent.save, RL_MODEL_FILE)
    await close_http()
    logger.info("Shutdown")


async def _gc_loop():
    while True:
        await asyncio.sleep(180)
        gc.collect()
        logger.debug("GC done")


# ── FastAPI ────────────────────────────────────────────────────────────────
app = FastAPI(title="MEXC Scalper", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_AUTH_TOKEN = os.getenv("AUTH_TOKEN", "")
if not _AUTH_TOKEN:
    _AUTH_TOKEN = hashlib.sha256(os.urandom(16)).hexdigest()[:12]
    logger.info("AUTH_TOKEN: %s", _AUTH_TOKEN)


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    if request.method == "GET" or request.url.path in ("/", "/sw.js", "/manifest.json") or request.url.path.startswith("/static"):
        return await call_next(request)
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    if token != _AUTH_TOKEN:
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=401, content={"error": "Unauthorized"})
    return await call_next(request)


# ── Frontend ────────────────────────────────────────────────────────────────
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")


@app.get("/")
async def serve_root():
    idx = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(idx):
        return FileResponse(idx)
    return {"status": "ok"}


@app.get("/sw.js")
async def serve_sw():
    sw = os.path.join(FRONTEND_DIR, "sw.js")
    if os.path.exists(sw):
        return FileResponse(sw)
    return {"status": "ok"}


# ── API Routes ─────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "pairs": len(FUTURES_PAIRS), "positions": sum(1 for s in agent_states.values() if s.get("active_pos"))}


@app.get("/api/status")
async def api_status():
    capital = portfolio.get("capital", 0)
    initial = portfolio.get("initial_capital", 5000)
    pnl = capital - initial
    active = sum(1 for s in agent_states.values() if s.get("active_pos"))
    total_trades = portfolio.get("total_closed_trades", 0)
    wins = sum(s.get("wins", 0) for s in agent_states.values())
    wr = round(wins / total_trades * 100, 1) if total_trades > 0 else 0
    return {
        "capital": round(capital, 2),
        "pnl": round(pnl, 2),
        "active_positions": active,
        "total_trades": total_trades,
        "win_rate": wr,
        "pairs": len(FUTURES_PAIRS),
        "scanned": len(scanner_cache),
        "model_trained": ml_engine._trained,
        "rl_trained": rl_agent._is_trained,
    }


@app.get("/api/positions")
async def api_positions():
    positions = []
    for sym, st in agent_states.items():
        if st.get("active_pos"):
            p = dict(st["active_pos"])
            p["symbol"] = sym
            if p.get("entry_time_raw") and isinstance(p["entry_time_raw"], datetime):
                p["entry_time"] = p["entry_time_raw"].isoformat()
            positions.append(p)
    return {"positions": positions, "count": len(positions)}


@app.get("/api/history")
async def api_history():
    return {"trades": list(trade_history), "total": len(trade_history)}


@app.get("/api/pnl")
async def api_pnl():
    return {"timeline": _pnl_timeline}


@app.get("/api/pairs")
async def api_pairs():
    pairs = []
    for sym in (FUTURES_PAIRS or [])[:50]:
        st = agent_states.get(sym, {})
        pairs.append({
            "symbol": sym,
            "price": scanner_cache.get(sym, {}).get("price", 0),
            "signal": scanner_cache.get(sym, {}).get("signal", "WAIT"),
            "pnl": round(st.get("pnl", 0), 2),
            "trades": st.get("trades", 0),
            "wins": st.get("wins", 0),
            "active": st.get("active_pos") is not None,
        })
    return {"pairs": pairs, "total": len(FUTURES_PAIRS)}


@app.get("/api/train")
async def api_train():
    klines_list = [v for v in _klines_cache.values() if len(v.get("close", [])) > 25]
    if not klines_list:
        return {"error": "no_data"}
    result = await asyncio.to_thread(ml_engine.train, klines_list[0])
    if result.get("success"):
        await asyncio.to_thread(ml_engine.save, ML_MODEL_FILE)
    return result
