"""
MEXC ML Trading System - FastAPI Backend v2
Non-blocking sürekli eğitim + gerçek feedback loop
"""

import asyncio
import json
import time
import logging
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Optional, List
from contextlib import asynccontextmanager

import os
import numpy as np
import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn

from ml_engine import MLEngine, FeatureBuilder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Telegram ──────────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID")

async def send_telegram_message(text: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.info(f"[TG-SKIP] {text[:80]}")
        return
    try:
        async with httpx.AsyncClient() as client:
            await client.post(
                f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"},
                timeout=10
            )
    except Exception as e:
        logger.error(f"Telegram error: {e}")

# ── Persistence ───────────────────────────────────────────────────────────────
_PERSIST_DIR    = os.getenv("PERSIST_DIR", os.path.join(os.path.dirname(__file__), ".."))
PERSISTENCE_FILE = os.path.join(_PERSIST_DIR, "persistence.json")
MODEL_FILE       = os.path.join(_PERSIST_DIR, "ml_model.joblib")
MODEL_VERSIONS_DIR = os.path.join(_PERSIST_DIR, "model_versions")

# ── Globals ───────────────────────────────────────────────────────────────────
ml_engine = MLEngine(model_dir=_PERSIST_DIR)
active_connections: List[WebSocket] = []
scanner_cache: dict = {}
agent_states: dict = {}
trade_history = deque(maxlen=200)
portfolio = {"capital": 100000.0, "total_fees": 0.0, "initial_capital": 100000.0}
FUTURES_PAIRS: List[str] = []
MAX_OPEN_POSITIONS = 15
MIN_VOLUME_24H     = 50_000_000.0

# ── Kline önbelleği — retrain için son veriyi sakla ───────────────────────────
_klines_cache: dict = {}   # symbol → klines dict

def json_serial(obj):
    if isinstance(obj, datetime): return obj.isoformat()
    raise TypeError(f"Not serializable: {type(obj)}")

def save_data():
    try:
        data = {"agent_states": agent_states, "trade_history": list(trade_history),
                "portfolio": portfolio, "timestamp": datetime.utcnow().isoformat()}
        with open(PERSISTENCE_FILE, "w") as f:
            json.dump(data, f, default=json_serial)
    except Exception as e:
        logger.error(f"Save error: {e}")

def load_data():
    global agent_states, trade_history, portfolio
    if not os.path.exists(PERSISTENCE_FILE): return
    try:
        with open(PERSISTENCE_FILE) as f:
            data = json.load(f)
        loaded = data.get("agent_states", {})
        for sym, st in loaded.items():
            try:
                if st.get("last_exit_time") and isinstance(st["last_exit_time"], str):
                    st["last_exit_time"] = datetime.fromisoformat(st["last_exit_time"])
            except Exception: st["last_exit_time"] = None
            if st.get("active_pos"):
                pos = st["active_pos"]
                try:
                    if pos.get("entry_time_raw") and isinstance(pos["entry_time_raw"], str):
                        pos["entry_time_raw"] = datetime.fromisoformat(pos["entry_time_raw"])
                    elif not pos.get("entry_time_raw"):
                        pos["entry_time_raw"] = datetime.utcnow()
                except Exception:
                    pos["entry_time_raw"] = datetime.utcnow()
                # Güvenlik: TP/SL mutlaka tanımlı olmalı
                if not pos.get("tp") or not pos.get("sl"):
                    logger.warning(f"[{sym}] Yüklenen pozisyonda TP/SL eksik — pozisyon temizlendi")
                    st["active_pos"] = None
                # entry_price sıfırsa pozisyonu temizle
                if pos and pos.get("entry_price", 0) <= 0:
                    logger.warning(f"[{sym}] Yüklenen pozisyonda geçersiz giriş fiyatı — temizlendi")
                    st["active_pos"] = None
        agent_states   = loaded
        trade_history  = deque(data.get("trade_history", []), maxlen=200)
        portfolio      = data.get("portfolio", portfolio)
        logger.info(f"Veriler yüklendi — {len(agent_states)} sembol, {len(trade_history)} trade")
    except Exception as e:
        logger.error(f"Load error: {e}")

# ── MEXC API ──────────────────────────────────────────────────────────────────
MEXC_BASE = "https://contract.mexc.com/api/v1/contract"

async def fetch_all_futures_pairs(client: httpx.AsyncClient) -> List[str]:
    try:
        r = await client.get(f"{MEXC_BASE}/detail", timeout=10)
        if r.status_code == 200:
            d = r.json()
            if d.get("success") and d.get("data"):
                pairs = [item["symbol"] for item in d["data"]
                         if item.get("state") == 0 and item.get("settleCoin") == "USDT"]
                logger.info(f"MEXC: {len(pairs)} aktif pair")
                return pairs
    except Exception as e:
        logger.error(f"Pair fetch: {e}")
    return []

async def fetch_ticker(client: httpx.AsyncClient, symbol: str) -> Optional[dict]:
    try:
        r = await client.get(f"{MEXC_BASE}/ticker", params={"symbol": symbol}, timeout=5)
        if r.status_code == 200:
            d = r.json()
            if d.get("success") and d.get("data"): return d["data"]
    except Exception as e:
        logger.debug(f"Ticker {symbol}: {e}")
    return None

async def fetch_klines(client: httpx.AsyncClient, symbol: str,
                       interval: str = "Min15", limit: int = 300) -> Optional[dict]:
    try:
        r = await client.get(f"{MEXC_BASE}/kline/{symbol}",
                             params={"interval": interval, "limit": limit}, timeout=8)
        if r.status_code == 200:
            d = r.json()
            if d.get("success") and d.get("data"):
                rows = d["data"]
                if rows and isinstance(rows, dict):
                    cl = rows.get("close", [])
                    if len(cl) > 20:
                        kd = {
                            "timestamp": rows.get("time", []),
                            "open":   np.array([float(x) for x in rows.get("open", [])]),
                            "high":   np.array([float(x) for x in rows.get("high", [])]),
                            "low":    np.array([float(x) for x in rows.get("low",  [])]),
                            "close":  np.array([float(x) for x in cl]),
                            "volume": np.array([float(x) for x in rows.get("vol", [])]),
                            "_symbol": symbol,
                        }
                        return kd
    except Exception as e:
        logger.debug(f"Kline {symbol}: {e}")
    return None

# ── Otomatik eğitim & sürekli retrain döngüsü ────────────────────────────────

async def auto_train_on_startup():
    """Başlangıçta 12 pair üzerinde multi-pair eğitim yap"""
    await asyncio.sleep(35)
    logger.info("🤖 Başlangıç eğitimi başlıyor...")
    klines_list = []
    async with httpx.AsyncClient() as client:
        targets = FUTURES_PAIRS[:12] if FUTURES_PAIRS else ["BTC_USDT","ETH_USDT","SOL_USDT"]
        for sym in targets:
            kl = await fetch_klines(client, sym, limit=300)
            if kl:
                klines_list.append(kl)
                _klines_cache[sym] = kl
                logger.info(f"  {sym}: {len(kl['close'])} mum")
            await asyncio.sleep(0.8)
    if klines_list:
        result = await asyncio.to_thread(ml_engine.train_on_multi_pair, klines_list, "GLOBAL")
        logger.info(f"✅ Başlangıç eğitimi: {result}")
        await asyncio.to_thread(ml_engine.save, MODEL_FILE)
    else:
        logger.warning("Başlangıç eğitimi: MEXC API erişilemiyor")

async def continuous_retrain_loop():
    """
    Sürekli retrain döngüsü — her 5 dakikada tetikleyicileri kontrol eder.
    Non-blocking: asyncio.to_thread ile scanner durmaz.
    Tetikleyiciler:
      1. Zamansal (6 saatte bir)
      2. Feedback (200 trade kapandıktan sonra)
      3. Drift (accuracy düşünce)
    """
    await asyncio.sleep(120)  # İlk eğitimin bitmesini bekle
    logger.info("🔁 Sürekli retrain döngüsü başladı")
    while True:
        try:
            should, reason = ml_engine.should_retrain()
            if should:
                logger.info(f"⚡ Retrain tetiklendi: {reason}")

                # BTC_USDT veya cache'deki en iyi pair ile
                async with httpx.AsyncClient() as client:
                    klines_list = []
                    targets = FUTURES_PAIRS[:8] if FUTURES_PAIRS else ["BTC_USDT","ETH_USDT"]
                    for sym in targets:
                        # Önce cache'e bak, yoksa çek
                        kl = _klines_cache.get(sym)
                        if kl is None:
                            kl = await fetch_klines(client, sym, limit=300)
                        if kl:
                            klines_list.append(kl)
                        await asyncio.sleep(0.3)

                if klines_list:
                    # Non-blocking thread'de eğit
                    result = await asyncio.to_thread(
                        ml_engine.train_on_multi_pair, klines_list, f"RETRAIN_{reason}"
                    )
                    if result.get("success"):
                        logger.info(
                            f"✅ Retrain tamamlandı [{reason}] | "
                            f"wf_acc={result.get('wf_accuracy')}% | "
                            f"n={result.get('n_samples')} sample"
                        )
                        await asyncio.to_thread(ml_engine.save, MODEL_FILE)

                        # Drift retrain için sayacı güncelle
                        if reason == "drift_detected":
                            ml_engine._retrain_triggers["drift"] += 1
                            ml_engine._last_drift_retrain = time.time()
                        elif reason.startswith("temporal"):
                            ml_engine._retrain_triggers["temporal"] += 1
                        elif reason.startswith("feedback"):
                            ml_engine._retrain_triggers["feedback"] += 1
                    else:
                        logger.warning(f"Retrain başarısız: {result.get('reason')}")

        except Exception as e:
            logger.error(f"Retrain döngüsü hatası: {e}")

        await asyncio.sleep(300)  # Her 5 dakikada kontrol et

async def keep_alive_loop():
    """Render free tier uyku önleme — 14 dakikada bir self-ping"""
    await asyncio.sleep(60)
    app_url = os.getenv("RENDER_EXTERNAL_URL", "")
    if not app_url:
        logger.info("RENDER_EXTERNAL_URL yok — keep-alive devre dışı")
        return
    logger.info(f"Keep-alive: {app_url}/health")
    while True:
        try:
            async with httpx.AsyncClient() as client:
                r = await client.get(f"{app_url}/health", timeout=10)
                logger.debug(f"Keep-alive: {r.status_code}")
        except Exception as e:
            logger.debug(f"Keep-alive hata: {e}")
        await asyncio.sleep(840)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 MEXC ML Trading System v2 başlatılıyor...")
    load_data()
    if ml_engine.load(MODEL_FILE):
        logger.info("✅ Önceki model yüklendi")
    else:
        logger.info("Model bulunamadı — başlangıç eğitimi yapılacak")
    asyncio.create_task(scanner_loop())
    asyncio.create_task(broadcast_loop())
    asyncio.create_task(auto_train_on_startup())
    asyncio.create_task(continuous_retrain_loop())
    asyncio.create_task(keep_alive_loop())
    yield
    save_data()
    await asyncio.to_thread(ml_engine.save, MODEL_FILE)
    logger.info("Sistem kapatıldı — veri ve model kaydedildi")

# ── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI(title="MEXC ML Trader v2", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Scanner & Position Manager ───────────────────────────────────────────────

# Deploy sonrası pozisyon koruma: ilk 60 saniye kapatma yok
_startup_time: float = 0.0
_STARTUP_GRACE_SECONDS = 120  # İlk 120 saniye TP/SL tetiklenmesin (deploy sonrası)

async def scanner_loop():
    global FUTURES_PAIRS, _startup_time
    _startup_time = time.time()
    logger.info("Scanner loop başladı")
    while True:
        try:
            async with httpx.AsyncClient() as client:
                new_pairs = await fetch_all_futures_pairs(client)
                if new_pairs: FUTURES_PAIRS = new_pairs
                if not FUTURES_PAIRS:
                    await asyncio.sleep(10); continue
                for i in range(0, len(FUTURES_PAIRS), 10):
                    batch = FUTURES_PAIRS[i:i+10]
                    results = await asyncio.gather(
                        *[process_pair(client, sym) for sym in batch],
                        return_exceptions=True
                    )
                    for sym, result in zip(batch, results):
                        if isinstance(result, dict):
                            scanner_cache[sym] = result
                    await asyncio.sleep(0.5)
            logger.info(f"Tarama tamamlandı — {len(scanner_cache)} pair")
        except Exception as e:
            logger.error(f"Scanner hata: {e}")
        await asyncio.sleep(30)

async def process_pair(client: httpx.AsyncClient, symbol: str) -> dict:
    ticker = await fetch_ticker(client, symbol)
    klines = await fetch_klines(client, symbol, limit=300)

    price      = float(ticker.get("lastPrice", 0))    if ticker else 0
    change24h  = float(ticker.get("riseFallRate", 0)) * 100 if ticker else 0
    volume24h  = float(ticker.get("volume24", 0))     if ticker else 0

    # Klines önbelleğini güncelle
    if klines is not None:
        _klines_cache[symbol] = klines

    prediction = ml_engine.predict(symbol, klines, price)

    # TP/SL hesapla — kaldıraca göre dinamik
    lev = prediction.get("leverage", 10)
    sl_pct = max(0.015, 0.025 - lev * 0.0005)  # Yüksek kaldıraç = daha sıkı SL
    tp_pct = sl_pct * 2.0  # R:R = 1:2 minimum
    if prediction["signal"] == "SHORT":
        sl = round(price * (1 + sl_pct), 10); tp = round(price * (1 - tp_pct), 10)
    else:
        sl = round(price * (1 - sl_pct), 10); tp = round(price * (1 + tp_pct), 10)

    if symbol not in agent_states:
        agent_states[symbol] = {"pnl": 0.0, "trades": 0, "wins": 0,
                                 "active_pos": None, "last_exit_time": None,
                                 "entry_feat": None}
    st = agent_states[symbol]

    # ── GİRİŞ KONTROLÜ ────────────────────────────────────────────────────────
    # Ensemble tutarlılık filtresi: her iki model de WAIT/HOLD DEMEMELİ
    # Önceki: GBM = RF = sinyal (çok katı → az trade)
    # Yeni: En az biri ana sinyalle hemfikir VEYA confidence çok yüksekse geç
    gbm_sig = prediction.get("gbm_signal", "WAIT")
    rf_sig  = prediction.get("rf_signal", "WAIT")
    main_sig = prediction["signal"]
    conf_val = prediction.get("confidence", 0)
    # Her ikisi de aynıysa kesin → Sadece biri aynıysa %70+ conf gerekli
    signals_agree = (
        (gbm_sig == rf_sig == main_sig) or                          # Her ikisi hemfikir
        (gbm_sig == main_sig and conf_val >= 70) or                 # GBM hemfikir + yüksek conf
        (rf_sig  == main_sig and conf_val >= 72)                    # RF hemfikir + yüksek conf
    )

    # Fiyat minimum eşik: 0.000001 USDT altı = delistleniyor veya sıfır hacim
    MIN_PRICE = 0.000001

    can_enter = (
        st["active_pos"] is None and
        prediction["signal"] in ("LONG","SHORT") and
        price >= MIN_PRICE and                  # Sıfır ve çok küçük fiyatları engelle
        prediction["data_quality"] == "real" and
        signals_agree                           # GBM + RF hemfikir olmalı
    )
    if can_enter:
        active_count = sum(1 for s in agent_states.values() if s.get("active_pos"))
        if active_count >= MAX_OPEN_POSITIONS:
            can_enter = False
            logger.debug(f"[{symbol}] Max pozisyon sınırı ({MAX_OPEN_POSITIONS}) doldu")
        elif volume24h < MIN_VOLUME_24H:
            can_enter = False
            logger.debug(f"[{symbol}] Yetersiz hacim: ${volume24h:,.0f} < ${MIN_VOLUME_24H:,.0f}")
    if can_enter and st["last_exit_time"]:
        if (datetime.utcnow() - st["last_exit_time"]).total_seconds() < 300:
            can_enter = False
    if can_enter:
        tp_dist = abs(tp - price) / price
        sl_dist = abs(sl - price) / price
        if tp_dist < 0.001 or sl_dist < 0.001: can_enter = False

    if can_enter:
        leverage = prediction["leverage"]
        entry_time_raw = datetime.utcnow()
        tr_time = (entry_time_raw + timedelta(hours=3)).strftime("%d.%m.%Y %H:%M:%S")
        entry_fee = 100 * leverage * 0.0006
        portfolio["total_fees"] += entry_fee
        portfolio["capital"]    -= entry_fee

        # Entry features'ı sakla (feedback için)
        st["entry_feat"] = prediction.get("_feat")

        st["active_pos"] = {
            "entry_price": price, "side": prediction["signal"],
            "leverage": leverage, "size": 100 * leverage,
            "timestamp": tr_time, "entry_time_raw": entry_time_raw,
            "tp": tp, "sl": sl,
            "indicators": ", ".join(prediction["indicators"])
        }
        asyncio.create_task(send_telegram_message(
            f"<b>🚀 GİRİŞ: {symbol}</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"📍 Fiyat: <code>{price:.10g}</code>\n"
            f"↕️ Yön: <b>{prediction['signal']}</b> | ⚙️ {leverage}x\n"
            f"💰 Büyüklük: <b>${100*leverage}</b>\n"
            f"🎯 TP: <code>{tp:.10g}</code> | 🛑 SL: <code>{sl:.10g}</code>\n"
            f"🕒 {tr_time} (TR) | 🏦 Kasa: ${portfolio['capital']:.2f}\n"
            f"🔍 <i>{st['active_pos']['indicators']}</i>"
        ))
        save_data()

    # ── POZİSYON TAKİBİ & ÇIKIŞ ───────────────────────────────────────────────
    if st["active_pos"] is not None:
        pos = st["active_pos"]
        entry_p = pos.get("entry_price", 0)
        side = pos.get("side", "LONG")
        lev  = pos.get("leverage", 10)
        size = pos.get("size", 1000)

        # FIX: Entry price sıfırsa pozisyon geçersiz — temizle
        if entry_p <= 0:
            logger.warning(f"[{symbol}] Aktif pozisyon geçersiz entry_price={entry_p} — temizlendi")
            st["active_pos"] = None
            st["entry_feat"]  = None
            save_data()

        # FIX: price=0 veya çok küçükse pozisyonu HİÇ işleme
        if st["active_pos"] is not None:
            MIN_PRICE = 0.000001
            if price < MIN_PRICE:
                logger.warning(f"[{symbol}] Geçersiz/sıfır fiyat ({price}) — pozisyon korunuyor, PnL güncellenmedi")
                result["price"] = pos.get("entry_price", 0)
                result["active_pos"] = pos
            else:
                diff_pct = (price - entry_p) / entry_p if side == "LONG" else (entry_p - price) / entry_p
                pos["unrealized_pnl"] = round(size * diff_pct, 2)
                pos["current_price"]  = price

            close_pos = False; exit_reason = ""
            # FIX: price>0 kontrolü TP/SL'den ÖNCE
            if price >= MIN_PRICE:
                if side == "LONG":
                    if price >= pos["tp"]: close_pos = True; exit_reason = "🎯 TAKE PROFIT"
                    elif price <= pos["sl"]: close_pos = True; exit_reason = "🛑 STOP LOSS"
                elif side == "SHORT":
                    if price <= pos["tp"]: close_pos = True; exit_reason = "🎯 TAKE PROFIT"
                    elif price >= pos["sl"]: close_pos = True; exit_reason = "🛑 STOP LOSS"

            # FIX: Deploy grace period — ilk 120 saniye pozisyon kapatma
            in_grace = (time.time() - _startup_time) < _STARTUP_GRACE_SECONDS
            if in_grace and close_pos:
                logger.info(f"[{symbol}] Grace period aktif — {exit_reason} atlandı (fiyat: {price})")
                close_pos = False

        if close_pos:
            price_diff_pct = (price - entry_p)/entry_p if side=="LONG" else (entry_p-price)/entry_p
            trade_pnl = size * price_diff_pct
            exit_val  = size * (1 + price_diff_pct)
            exit_fee  = exit_val * 0.0006
            total_fee = size * 0.0006 + exit_fee
            net_pnl   = trade_pnl - total_fee

            st["pnl"]    += net_pnl
            st["trades"] += 1
            if net_pnl > 0: st["wins"] += 1

            portfolio["total_fees"] += exit_fee
            portfolio["capital"]    += trade_pnl - exit_fee

            # ✅ GERÇEK FEEDBACK — trade kapandıktan sonra
            entry_feat = st.get("entry_feat")
            if entry_feat is not None:
                ml_engine.record_trade_result(
                    features=entry_feat,
                    side=side,
                    net_pnl=net_pnl,
                    symbol=symbol
                )
                st["entry_feat"] = None  # Temizle

            exit_time_raw = datetime.utcnow()
            tr_exit = (exit_time_raw + timedelta(hours=3)).strftime("%d.%m.%Y %H:%M:%S")
            diff = exit_time_raw - pos["entry_time_raw"]
            secs = int(diff.total_seconds())
            h,rem = divmod(secs, 3600); m,s = divmod(rem, 60)
            dur = f"{h}sa {m}dk {s}sn" if h>0 else f"{m}dk {s}sn"

            asyncio.create_task(send_telegram_message(
                f"<b>✅ KAPANDI: {symbol}</b> ({exit_reason})\n"
                f"━━━━━━━━━━━━━━━━━━━━━\n"
                f"📥 Giriş: <code>{entry_p:.10g}</code>\n"
                f"📤 Çıkış: <code>{price:.10g}</code>\n"
                f"💵 Net PnL: <b>{'🟢' if net_pnl>0 else '🔴'} ${net_pnl:.2f}</b>\n"
                f"📈 Sembol PnL: <b>${st['pnl']:.2f}</b>\n"
                f"⏳ Süre: <b>{dur}</b> | 🕒 {tr_exit} (TR)\n"
                f"🏦 Kasa: <b>${portfolio['capital']:.2f}</b>\n"
                f"🔄 Feedback: {ml_engine.feedback.size()} trade birikti"
            ))

            trade_history.appendleft({
                "symbol": symbol, "entry_price": entry_p, "exit_price": price,
                "side": side, "pnl": round(net_pnl, 2), "reason": exit_reason,
                "duration": dur, "time": tr_exit
            })
            st["active_pos"]    = None
            st["last_exit_time"] = exit_time_raw
            save_data()

    result = {
        "symbol": symbol, "price": price,
        "change24h": round(change24h, 2), "volume24h": round(volume24h, 2),
        "signal": prediction["signal"], "confidence": prediction["confidence"],
        "indicators": prediction["indicators"], "model_used": prediction["model"],
        "entry_price": price,
        "stop_loss":   sl if price > 0 else 0,
        "take_profit": tp if price > 0 else 0,
        "leverage": prediction["leverage"],
        "gbm_signal": prediction.get("gbm_signal","WAIT"),
        "rf_signal":  prediction.get("rf_signal","WAIT"),
        "model_trained": prediction.get("model_trained", False),
        "wf_accuracy":   prediction.get("wf_accuracy", 0),
        "backtest_roi":  prediction.get("backtest_roi", 0),
        "backtest_sharpe": prediction.get("backtest_sharpe", 0),
        "drift_status":  prediction.get("drift_status", {}),
        "timestamp":   datetime.utcnow().isoformat(),
        "data_source": "real" if klines else "simulated",
        "active_pos":  st.get("active_pos"),
        "pnl":      round(st["pnl"], 2),
        "trades":   st["trades"],
        "win_rate": round(st["wins"]/st["trades"]*100,1) if st["trades"]>0 else 0,
    }
    return result

async def broadcast_loop():
    await asyncio.sleep(5)
    while True:
        if active_connections and scanner_cache:
            msg = json.dumps({
                "type": "update",
                "data": list(scanner_cache.values()),
                "portfolio": portfolio,
                "timestamp": datetime.utcnow().isoformat(),
                "total_pairs": len(scanner_cache),
                "ml_status": {
                    "trained": ml_engine._trained,
                    "wf_accuracy": ml_engine._wf.get("accuracy", 0),
                    "feedback_size": ml_engine.feedback.size(),
                    "drift": ml_engine.drift.get_status(),
                    "training_in_progress": ml_engine._training_in_progress,
                }
            })
            dead = []
            for ws in active_connections[:]:
                try: await ws.send_text(msg)
                except Exception: dead.append(ws)
            for ws in dead:
                if ws in active_connections: active_connections.remove(ws)
        await asyncio.sleep(5)

# ── REST Endpoints ─────────────────────────────────────────────────────────────
@app.get("/api/scan")
async def get_scan():
    return {"success":True,"data":list(scanner_cache.values()),
            "total":len(scanner_cache),"timestamp":datetime.utcnow().isoformat()}

@app.get("/api/pair/{symbol}")
async def get_pair(symbol: str):
    sym = symbol.upper().replace("-","_")
    if sym in scanner_cache: return {"success":True,"data":scanner_cache[sym]}
    return {"success":False,"error":"Pair not found"}

@app.get("/api/stats")
async def get_stats():
    if not scanner_cache: return {"success":True,"data":{}}
    vals = list(scanner_cache.values())
    total_pnl    = sum(agent_states.get(v["symbol"],{}).get("pnl",0) for v in vals)
    total_trades = sum(agent_states.get(v["symbol"],{}).get("trades",0) for v in vals)
    total_wins   = sum(agent_states.get(v["symbol"],{}).get("wins",0) for v in vals)
    active_longs  = sum(1 for v in vals if v.get("active_pos") and v["active_pos"]["side"]=="LONG")
    active_shorts = sum(1 for v in vals if v.get("active_pos") and v["active_pos"]["side"]=="SHORT")
    closed_longs  = sum(1 for h in trade_history if h["side"]=="LONG")
    closed_shorts = sum(1 for h in trade_history if h["side"]=="SHORT")
    return {"success":True,"data":{
        "total_pairs":len(vals),"total_pnl":round(total_pnl,2),
        "total_trades":total_trades,
        "win_rate":round(total_wins/total_trades*100,1) if total_trades>0 else 0,
        "active_longs":active_longs,"active_shorts":active_shorts,
        "closed_longs":closed_longs,"closed_shorts":closed_shorts,
        "portfolio":portfolio,"model_accuracy":round(ml_engine.get_accuracy(),1),
        "timestamp":datetime.utcnow().isoformat()
    }}

@app.get("/api/model")
async def get_model_info():
    return {"success":True,"data":ml_engine.get_info()}

@app.get("/api/history")
async def get_trade_history():
    return {"success":True,"data":list(trade_history)}

@app.get("/api/backtest/{symbol}")
async def run_backtest(symbol: str):
    sym = symbol.upper().replace("-","_")
    async with httpx.AsyncClient() as client:
        klines = await fetch_klines(client, sym, limit=300)
    if not klines: return {"success":False,"error":"Veri alınamadı"}
    result = await asyncio.to_thread(ml_engine.run_backtest, klines)
    return {"success":True,"symbol":sym,"data":result}

@app.get("/api/backtest_all")
async def run_backtest_all():
    return {"success":True,"data":ml_engine._bt}

@app.post("/api/train/{symbol}")
async def trigger_training(symbol: str):
    sym = symbol.upper().replace("-","_")
    async with httpx.AsyncClient() as client:
        klines = await fetch_klines(client, sym, limit=300)
    if not klines: return {"success":False,"error":"MEXC kline verisi alınamadı"}
    result = await asyncio.to_thread(ml_engine.train, klines, sym, True)
    if result.get("success"):
        await asyncio.to_thread(ml_engine.save, MODEL_FILE)
        ml_engine._retrain_triggers["manual"] += 1
    return {"success":True,"data":result}

@app.post("/api/train_all")
async def train_all():
    klines_list = []
    async with httpx.AsyncClient() as client:
        for sym in FUTURES_PAIRS[:12]:
            kl = await fetch_klines(client, sym, limit=300)
            if kl: klines_list.append(kl)
    if not klines_list: return {"success":False,"error":"Veri alınamadı"}
    result = await asyncio.to_thread(ml_engine.train_on_multi_pair, klines_list, "MANUAL_ALL")
    if result.get("success"):
        await asyncio.to_thread(ml_engine.save, MODEL_FILE)
        ml_engine._retrain_triggers["manual"] += 1
    return {"success":True,"data":result}

@app.get("/api/drift")
async def get_drift_status():
    return {"success":True,"data":ml_engine.drift.get_status()}

@app.get("/api/versions")
async def get_model_versions():
    return {"success":True,"data":ml_engine.version_store.get_versions()}

@app.post("/api/rollback/{version_index}")
async def rollback_model(version_index: int):
    versions = ml_engine.version_store.get_versions()
    if version_index >= len(versions):
        return {"success":False,"error":"Versiyon bulunamadı"}
    v = versions[version_index]
    ok = ml_engine.load(v["path"])
    return {"success":ok,"data":v}

@app.get("/health")
async def health_check():
    return {"status":"ok","timestamp":datetime.utcnow().isoformat(),
            "total_pairs":len(scanner_cache),
            "model_trained":ml_engine._trained,
            "wf_accuracy":ml_engine._wf.get("accuracy",0),
            "feedback_buffer":ml_engine.feedback.size(),
            "drift":ml_engine.drift.is_drifting()}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    logger.info(f"WS bağlandı — toplam: {len(active_connections)}")
    try:
        if scanner_cache:
            await websocket.send_text(json.dumps({
                "type":"init","data":list(scanner_cache.values()),
                "portfolio":portfolio,"timestamp":datetime.utcnow().isoformat()
            }))
        while True: await websocket.receive_text()
    except (WebSocketDisconnect, Exception):
        if websocket in active_connections: active_connections.remove(websocket)
        logger.info(f"WS ayrıldı — kalan: {len(active_connections)}")

# ── Frontend ───────────────────────────────────────────────────────────────────
frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(frontend_path):
    app.mount("/static", StaticFiles(directory=frontend_path), name="static")

@app.get("/ml-details")
async def serve_ml_details():
    idx = os.path.join(frontend_path, "ml_details.html")
    return FileResponse(idx) if os.path.exists(idx) else {"error":"ml_details.html bulunamadı"}

@app.get("/")
async def serve_frontend():
    idx = os.path.join(frontend_path, "index.html")
    return FileResponse(idx) if os.path.exists(idx) else {"message":"MEXC ML Trader v2 API"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
