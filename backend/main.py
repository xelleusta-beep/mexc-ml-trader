"""
MEXC ML Trading System - FastAPI Backend v3.0
RL (PPO) + ML (LightGBM+RF) Hibrit Sistem

v3.0 YENİLİKLER:
  - PPO Agent (rl_engine.py) entegre edildi
  - Hibrit karar: RL + ML ensemble
  - RL ön-eğitim (geçmiş veri backtesting)
  - Online fine-tuning (her 50 trade sonrası)
  - EWC (felaket unutma önleme)
  - WF < %65 → 5x kaldıraç kilidi korundu
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
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn

from ml_engine import MLEngine, FeatureBuilder
from rl_engine  import PPOAgent, TradingEnvironment, OnlineExperienceBuffer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Telegram ──────────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID")

async def send_telegram_message(text: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.info(f"[TG-SKIP] {text[:80]}"); return
    try:
        async with httpx.AsyncClient() as client:
            await client.post(
                f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"},
                timeout=10)
    except Exception as e:
        logger.error(f"Telegram error: {e}")

# ── Persistence ───────────────────────────────────────────────────────────────
_PERSIST_DIR     = os.getenv("PERSIST_DIR", os.path.join(os.path.dirname(__file__), ".."))
PERSISTENCE_FILE = os.path.join(_PERSIST_DIR, "persistence.json")
ML_MODEL_FILE    = os.path.join(_PERSIST_DIR, "ml_model.joblib")
RL_MODEL_FILE    = os.path.join(_PERSIST_DIR, "rl_model.joblib")

# ── Globals ───────────────────────────────────────────────────────────────────
ml_engine          = MLEngine(model_dir=_PERSIST_DIR)
rl_agent           = PPOAgent(state_dim=40, n_actions=5, hidden_dim=64, lr=3e-4)
rl_experience      = OnlineExperienceBuffer()

active_connections: List[WebSocket] = []
scanner_cache:  dict = {}
agent_states:   dict = {}
trade_history        = deque(maxlen=200)
portfolio = {
    "capital":               100000.0,
    "total_fees":            0.0,
    "initial_capital":       100000.0,
    "total_closed_notional": 0.0,
    "total_closed_trades":   0,
}
FUTURES_PAIRS: List[str] = []
MAX_OPEN_POSITIONS = 15
MIN_VOLUME_24H     = 50_000_000.0
MIN_PRICE          = 0.000001
MAX_POS_HOURS      = 8

_klines_cache: dict = {}

def json_serial(obj):
    if isinstance(obj, datetime): return obj.isoformat()
    raise TypeError(f"Not serializable: {type(obj)}")

def save_data():
    try:
        with open(PERSISTENCE_FILE, "w") as f:
            json.dump({
                "agent_states":  agent_states,
                "trade_history": list(trade_history),
                "portfolio":     portfolio,
                "timestamp":     datetime.utcnow().isoformat()
            }, f, default=json_serial)
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
            except Exception:
                st["last_exit_time"] = None
            if st.get("active_pos"):
                pos = st["active_pos"]
                try:
                    if pos.get("entry_time_raw") and isinstance(pos["entry_time_raw"], str):
                        pos["entry_time_raw"] = datetime.fromisoformat(pos["entry_time_raw"])
                    elif not pos.get("entry_time_raw"):
                        pos["entry_time_raw"] = datetime.utcnow()
                except Exception:
                    pos["entry_time_raw"] = datetime.utcnow()
                if not pos.get("tp") or not pos.get("sl"):
                    st["active_pos"] = None
                if pos and pos.get("entry_price", 0) <= 0:
                    st["active_pos"] = None
        agent_states  = loaded
        trade_history = deque(data.get("trade_history", []), maxlen=200)
        portfolio.update(data.get("portfolio", {}))
        portfolio.setdefault("total_closed_notional", 0.0)
        portfolio.setdefault("total_closed_trades", 0)
        logger.info(f"Veriler yüklendi — {len(agent_states)} sembol")
    except Exception as e:
        logger.error(f"Load error: {e}")

# ── MEXC API ──────────────────────────────────────────────────────────────────
MEXC_BASE = "https://contract.mexc.com/api/v1/contract"

async def fetch_all_futures_pairs(client):
    try:
        r = await client.get(f"{MEXC_BASE}/detail", timeout=10)
        if r.status_code == 200:
            d = r.json()
            if d.get("success") and d.get("data"):
                pairs = [x["symbol"] for x in d["data"]
                         if x.get("state") == 0 and x.get("settleCoin") == "USDT"]
                logger.info(f"MEXC: {len(pairs)} aktif pair")
                return pairs
    except Exception as e:
        logger.error(f"Pair fetch: {e}")
    return []

async def fetch_ticker(client, symbol):
    try:
        r = await client.get(f"{MEXC_BASE}/ticker", params={"symbol": symbol}, timeout=5)
        if r.status_code == 200:
            d = r.json()
            if d.get("success") and d.get("data"): return d["data"]
    except Exception as e:
        logger.debug(f"Ticker {symbol}: {e}")
    return None

async def fetch_klines(client, symbol, interval="Min15", limit=300):
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
                        return {
                            "timestamp": rows.get("time", []),
                            "open":   np.array([float(x) for x in rows.get("open", [])]),
                            "high":   np.array([float(x) for x in rows.get("high", [])]),
                            "low":    np.array([float(x) for x in rows.get("low", [])]),
                            "close":  np.array([float(x) for x in cl]),
                            "volume": np.array([float(x) for x in rows.get("vol", [])]),
                            "_symbol": symbol,
                        }
    except Exception as e:
        logger.debug(f"Kline {symbol}: {e}")
    return None

# ── HİBRİT KARAR MOTORU ──────────────────────────────────────────────────────
def hybrid_predict(symbol: str, klines: Optional[dict],
                   price: float, st: dict) -> dict:
    """
    ML + RL hibrit tahmin.
    - ML (LightGBM+RF): teknik sinyal
    - RL (PPO): pozisyon yönetimi ve boyutlandırma
    - Karar: her iki model hemfikirysa giriş, sadece biri varsa ML'e güven
    """
    # ML tahmini
    ml_pred = ml_engine.predict(symbol, klines, price)

    # RL tahmininin bağlamı
    pos_type = 0
    if st.get("active_pos"):
        pos_type = 1 if st["active_pos"]["side"] == "LONG" else -1

    unrealized = st["active_pos"].get("unrealized_pnl", 0.0) \
                 if st.get("active_pos") else 0.0
    pos_age    = 0
    if st.get("active_pos") and st["active_pos"].get("entry_time_raw"):
        pos_age = int((datetime.utcnow() -
                       st["active_pos"]["entry_time_raw"]).total_seconds() / 900)

    feat = ml_pred.get("_feat")
    if feat is None:
        # RL tahmini yapılamaz, sadece ML kullan
        return ml_pred

    rl_pred = rl_agent.predict(
        features       = feat,
        position       = pos_type,
        unrealized_pnl = float(unrealized),
        position_age   = pos_age,
        max_drawdown   = ml_engine.drift._max_dd if hasattr(ml_engine.drift, "_max_dd") else 0.0,
    )

    # ── HİBRİT KARAR MANTIĞI ─────────────────────────────────────────────────
    ml_sig  = ml_pred["signal"]    # LONG / SHORT / HOLD / WAIT
    rl_sig  = rl_pred["signal"]    # LONG / SHORT / WAIT
    ml_conf = ml_pred["confidence"]
    rl_conf = rl_pred["confidence"]

    # 1. İkisi de LONG veya SHORT → yüksek güven, RL leverage'ını kullan
    if ml_sig == rl_sig and ml_sig in ("LONG", "SHORT"):
        final_sig  = ml_sig
        final_conf = (ml_conf * 0.5 + rl_conf * 0.5)
        final_lev  = rl_pred["leverage"]   # RL pozisyon boyutlandırması
        source     = "RL+ML"

    # 2. ML LONG/SHORT ama RL WAIT → ML'e güven, daha küçük pozisyon
    elif ml_sig in ("LONG", "SHORT") and rl_sig == "WAIT":
        final_sig  = ml_sig
        final_conf = ml_conf * 0.7  # Güveni düşür
        final_lev  = 5              # Küçük pozisyon
        source     = "ML_only"

    # 3. RL LONG/SHORT ama ML WAIT/HOLD → RL'i hafifçe dinle
    elif rl_sig in ("LONG", "SHORT") and ml_sig in ("WAIT", "HOLD"):
        if rl_conf >= 70:
            final_sig  = rl_sig
            final_conf = rl_conf * 0.8
            final_lev  = 5
            source     = "RL_only"
        else:
            final_sig  = "WAIT"
            final_conf = 50.0
            final_lev  = 5
            source     = "disagree"

    # 4. Her ikisi de WAIT/HOLD → bekle
    else:
        final_sig  = "WAIT"
        final_conf = 50.0
        final_lev  = 5
        source     = "both_wait"

    # Conf eşiği
    if final_conf < 55:
        final_sig = "WAIT"

    # Sonuç dict
    result = ml_pred.copy()
    result.update({
        "signal":     final_sig,
        "confidence": round(final_conf, 1),
        "leverage":   final_lev,
        "model":      f"PPO+{ml_pred['model']}",
        "rl_signal":  rl_sig,
        "rl_conf":    round(rl_conf, 1),
        "rl_action":  rl_pred["action_desc"],
        "rl_value":   rl_pred["value"],
        "rl_probs":   rl_pred["probs"],
        "hybrid_source": source,
        "_rl_state":  rl_pred.get("_state"),
    })
    return result

# ── EĞİTİM DÖNGÜSÜ ───────────────────────────────────────────────────────────
async def auto_train_on_startup():
    await asyncio.sleep(35)
    logger.info("🤖 Başlangıç eğitimi başlıyor (ML + RL)...")
    klines_list = []
    async with httpx.AsyncClient() as client:
        targets = FUTURES_PAIRS[:20] if FUTURES_PAIRS else \
                  ["BTC_USDT","ETH_USDT","SOL_USDT","BNB_USDT","XRP_USDT"]
        for sym in targets:
            kl = await fetch_klines(client, sym, limit=300)
            if kl:
                klines_list.append(kl); _klines_cache[sym] = kl
                logger.info(f"  {sym}: {len(kl['close'])} mum")
            await asyncio.sleep(0.8)

    if not klines_list:
        logger.warning("Başlangıç eğitimi: MEXC API erişilemiyor"); return

    # ML eğitimi
    ml_result = await asyncio.to_thread(
        ml_engine.train_on_multi_pair, klines_list, "GLOBAL")
    logger.info(f"✅ ML eğitimi: wf={ml_result.get('wf_accuracy')}%")
    await asyncio.to_thread(ml_engine.save, ML_MODEL_FILE)

    # RL ön-eğitimi (backtesting üzerinde)
    await asyncio.to_thread(_rl_pretrain, klines_list)

async def _rl_pretrain_async(klines_list):
    await asyncio.to_thread(_rl_pretrain, klines_list)

def _rl_pretrain(klines_list: list):
    """RL ajanını geçmiş veri üzerinde ön-eğit."""
    logger.info(f"🤖 RL ön-eğitimi başlıyor ({len(klines_list)} pair)...")
    features_list = []
    prices_list   = []

    for kl in klines_list:
        feat_matrix = _build_feature_matrix(kl)
        if feat_matrix is not None and len(feat_matrix) >= 50:
            features_list.append(feat_matrix)
            prices_list.append(kl["close"].copy())

    if not features_list:
        logger.warning("RL ön-eğitimi: Feature oluşturulamadı"); return

    # Render free tier'da 30 iterasyon (~10-15 saniye)
    result = rl_agent.train_on_history(
        features_list, prices_list, n_iterations=100)  # FIX: 30→100
    logger.info(f"✅ RL ön-eğitimi: {result}")

    try:
        rl_agent.save(RL_MODEL_FILE)
        logger.info(f"✅ RL modeli kaydedildi")
    except Exception as e:
        logger.error(f"RL kayıt hatası: {e}")

def _build_feature_matrix(klines: dict) -> Optional[np.ndarray]:
    """Klines'dan (T, 36) feature matrisi oluştur."""
    try:
        c  = np.asarray(klines["close"],  dtype=np.float64)
        h  = np.asarray(klines["high"],   dtype=np.float64)
        lo = np.asarray(klines["low"],    dtype=np.float64)
        v  = np.asarray(klines["volume"], dtype=np.float64)
        MIN = 40
        if len(c) < MIN + 5: return None

        rows = []
        for i in range(MIN, len(c)):
            sub  = {"close": c[:i], "high": h[:i], "low": lo[:i], "volume": v[:i]}
            feat = FeatureBuilder.build(sub)
            if feat is not None:
                rows.append(feat)

        return np.array(rows, dtype=np.float32) if rows else None
    except Exception as e:
        logger.debug(f"Feature matrix hatası: {e}")
        return None

async def continuous_retrain_loop():
    await asyncio.sleep(120)
    logger.info("🔁 Sürekli retrain döngüsü başladı")
    while True:
        try:
            # ML retrain
            should, reason = ml_engine.should_retrain()
            if should:
                logger.info(f"⚡ ML Retrain: {reason}")
                klines_list = []
                async with httpx.AsyncClient() as client:
                    for sym in (FUTURES_PAIRS[:15] or ["BTC_USDT","ETH_USDT"]):
                        kl = _klines_cache.get(sym) or \
                             await fetch_klines(client, sym, limit=300)
                        if kl: klines_list.append(kl)
                        await asyncio.sleep(0.3)

                if klines_list:
                    result = await asyncio.to_thread(
                        ml_engine.train_on_multi_pair, klines_list, f"RETRAIN_{reason}")
                    if result.get("success"):
                        logger.info(f"✅ ML Retrain [{reason}] | wf={result.get('wf_accuracy')}%")
                        await asyncio.to_thread(ml_engine.save, ML_MODEL_FILE)
                        if reason.startswith("temporal"):
                            ml_engine._retrain_triggers["temporal"] += 1
                        elif reason.startswith("feedback"):
                            ml_engine._retrain_triggers["feedback"] += 1
                        elif reason == "drift_detected":
                            ml_engine._last_drift_retrain = time.time()
                            ml_engine._retrain_triggers["drift"] += 1

                        # ML retrain sonrası RL'i de güncelle
                        if klines_list:
                            await asyncio.to_thread(_rl_pretrain, klines_list[:5])

            # RL online update
            if rl_experience.ready():
                exps = rl_experience.get_and_clear()
                losses = await asyncio.to_thread(rl_agent.online_update, exps)
                await asyncio.to_thread(rl_agent.save, RL_MODEL_FILE)
                logger.info(f"✅ RL Online Update: {len(exps)} trade | "
                            f"loss={losses.get('policy_loss', 'N/A')}")

        except Exception as e:
            logger.error(f"Retrain döngüsü hatası: {e}")
        await asyncio.sleep(300)

async def keep_alive_loop():
    await asyncio.sleep(60)
    app_url = os.getenv("RENDER_EXTERNAL_URL", "")
    if not app_url:
        return
    while True:
        try:
            async with httpx.AsyncClient() as c:
                await c.get(f"{app_url}/health", timeout=10)
        except Exception:
            pass
        await asyncio.sleep(840)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 MEXC RL+ML Trading System v3.0 başlatılıyor...")
    load_data()

    # ML model yükle
    if ml_engine.load(ML_MODEL_FILE):
        logger.info("✅ ML modeli yüklendi")
    else:
        logger.info("ML modeli bulunamadı — başlangıç eğitimi yapılacak")

    # RL model yükle
    if rl_agent.load(RL_MODEL_FILE):
        logger.info(f"✅ RL modeli yüklendi | steps={rl_agent._total_steps}")
    else:
        logger.info("RL modeli bulunamadı — ön-eğitim yapılacak")

    asyncio.create_task(scanner_loop())
    asyncio.create_task(broadcast_loop())
    asyncio.create_task(auto_train_on_startup())
    asyncio.create_task(continuous_retrain_loop())
    asyncio.create_task(keep_alive_loop())
    yield
    save_data()
    await asyncio.to_thread(ml_engine.save, ML_MODEL_FILE)
    await asyncio.to_thread(rl_agent.save, RL_MODEL_FILE)
    logger.info("Sistem kapatıldı")

# ── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI(title="MEXC RL+ML Trader v3.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# ── Scanner ───────────────────────────────────────────────────────────────────
_startup_time: float = 0.0
_STARTUP_GRACE_SECONDS = 120

async def scanner_loop():
    global FUTURES_PAIRS, _startup_time
    _startup_time = time.time()
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
                        return_exceptions=True)
                    for sym, res in zip(batch, results):
                        if isinstance(res, dict):
                            scanner_cache[sym] = res
                    await asyncio.sleep(0.5)
            logger.info(f"Tarama tamamlandı — {len(scanner_cache)} pair")
        except Exception as e:
            logger.error(f"Scanner hata: {e}")
        await asyncio.sleep(30)

async def process_pair(client, symbol: str) -> dict:
    ticker = await fetch_ticker(client, symbol)
    klines = await fetch_klines(client, symbol, limit=300)

    price     = float(ticker.get("lastPrice", 0))     if ticker else 0
    change24h = float(ticker.get("riseFallRate", 0)) * 100 if ticker else 0
    volume24h = float(ticker.get("volume24", 0))      if ticker else 0

    # Ticker sıfır fiyat fallback
    if price < MIN_PRICE and klines is not None:
        try:
            fb = float(klines["close"][-1])
            if fb > MIN_PRICE: price = fb
        except (IndexError, KeyError, ValueError):
            pass

    if klines is not None:
        _klines_cache[symbol] = klines

    if symbol not in agent_states:
        agent_states[symbol] = {
            "pnl": 0.0, "trades": 0, "wins": 0,
            "active_pos": None, "last_exit_time": None,
            "entry_feat": None, "last_trade_pnl": 0.0,
            "rl_state": None,  # RL state kaydı
        }
    st = agent_states[symbol]

    # ── HİBRİT TAHMİN ────────────────────────────────────────────────────────
    prediction = hybrid_predict(symbol, klines, price, st)

    # Dinamik SL/TP
    lev = max(1, prediction.get("leverage", 5))
    wf_acc_now = ml_engine._wf.get("accuracy", 0)
    if wf_acc_now < 65 and lev > 5:
        lev = 5

    if lev >= 10:
        sl_pct, tp_pct = 0.012, 0.030
    elif lev >= 7:
        sl_pct, tp_pct = 0.018, 0.040
    else:
        sl_pct, tp_pct = 0.022, 0.050

    if prediction["signal"] == "SHORT":
        sl = round(price * (1 + sl_pct), 10); tp = round(price * (1 - tp_pct), 10)
    else:
        sl = round(price * (1 - sl_pct), 10); tp = round(price * (1 + tp_pct), 10)

    # ── GİRİŞ KONTROLÜ ───────────────────────────────────────────────────────
    gbm_sig  = prediction.get("gbm_signal", "WAIT")
    rf_sig   = prediction.get("rf_signal",  "WAIT")
    rl_sig   = prediction.get("rl_signal",  "WAIT")
    main_sig = prediction["signal"]
    conf_val = prediction.get("confidence", 0)
    source   = prediction.get("hybrid_source", "ML_only")

    # Sinyal uyumu: RL+ML hemfikir veya ML tek başına yeterli güvenli
    signals_agree = (
        source == "RL+ML" or                                       # Her ikisi hemfikir
        (source == "ML_only" and conf_val >= 65) or               # Sadece ML, yüksek conf
        (source == "RL_only" and conf_val >= 70) or               # Sadece RL, çok yüksek
        (gbm_sig == rf_sig == main_sig)                            # GBM+RF hemfikir
    )

    can_enter = (
        st["active_pos"] is None and
        main_sig in ("LONG", "SHORT") and
        price >= MIN_PRICE and
        prediction.get("data_quality") == "real" and
        signals_agree
    )
    if can_enter:
        active_count = sum(1 for s in agent_states.values() if s.get("active_pos"))
        if active_count >= MAX_OPEN_POSITIONS:
            can_enter = False
        elif volume24h < MIN_VOLUME_24H:
            can_enter = False

    if can_enter and st["last_exit_time"]:
        last_pnl     = st.get("last_trade_pnl", 0)
        cooldown_sec = 7200 if last_pnl < 0 else 300
        if (datetime.utcnow() - st["last_exit_time"]).total_seconds() < cooldown_sec:
            can_enter = False

    if can_enter:
        tp_dist = abs(tp - price) / price if price > 0 else 0
        sl_dist = abs(sl - price) / price if price > 0 else 0
        if tp_dist < 0.001 or sl_dist < 0.001:
            can_enter = False

    # ── GİRİŞ ────────────────────────────────────────────────────────────────
    if can_enter:
        leverage       = lev
        entry_time_raw = datetime.utcnow()
        tr_time        = (entry_time_raw + timedelta(hours=3)).strftime("%d.%m.%Y %H:%M:%S")
        base_size      = 100.0
        notional       = base_size * leverage
        entry_fee      = notional * 0.0006
        portfolio["total_fees"] += entry_fee
        portfolio["capital"]    -= entry_fee

        st["entry_feat"] = prediction.get("_feat")
        st["rl_state"]   = prediction.get("_rl_state")  # RL state kaydet
        st["active_pos"] = {
            "entry_price":    price, "side": main_sig,
            "leverage":       leverage, "size": notional,
            "base_size":      base_size, "timestamp": tr_time,
            "entry_time_raw": entry_time_raw, "tp": tp, "sl": sl,
            "indicators":     ", ".join(prediction.get("indicators", [])),
            "hybrid_source":  source,
        }
        asyncio.create_task(send_telegram_message(
            f"<b>🚀 GİRİŞ: {symbol}</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"📍 Fiyat: <code>{price:.10g}</code>\n"
            f"↕️ Yön: <b>{main_sig}</b> | ⚙️ {leverage}x\n"
            f"🤖 Kaynak: <b>{source}</b> | Güven: {conf_val:.0f}%\n"
            f"💰 Nominal: <b>${notional:.0f}</b>\n"
            f"🎯 TP: <code>{tp:.10g}</code> | 🛑 SL: <code>{sl:.10g}</code>\n"
            f"🕒 {tr_time} (TR) | 🏦 Kasa: ${portfolio['capital']:.2f}"
        ))
        save_data()

    # ── POZİSYON TAKİBİ ──────────────────────────────────────────────────────
    close_pos   = False
    exit_reason = ""

    if st["active_pos"] is not None:
        pos     = st["active_pos"]
        entry_p = pos.get("entry_price", 0)
        side    = pos.get("side", "LONG")
        size    = pos.get("size", 500)

        if entry_p <= 0:
            st["active_pos"] = None; st["entry_feat"] = None
            st["rl_state"]   = None; save_data()
        elif price < MIN_PRICE:
            pass
        else:
            diff_pct = (price - entry_p)/entry_p if side == "LONG" \
                       else (entry_p - price)/entry_p
            pos["unrealized_pnl"] = round(size * diff_pct, 2)
            pos["current_price"]  = price

            if side == "LONG":
                if price >= pos["tp"]:   close_pos = True; exit_reason = "🎯 TAKE PROFIT"
                elif price <= pos["sl"]: close_pos = True; exit_reason = "🛑 STOP LOSS"
            else:
                if price <= pos["tp"]:   close_pos = True; exit_reason = "🎯 TAKE PROFIT"
                elif price >= pos["sl"]: close_pos = True; exit_reason = "🛑 STOP LOSS"

            pos_age_h = (datetime.utcnow() - pos["entry_time_raw"]).total_seconds() / 3600
            if not close_pos and pos_age_h >= MAX_POS_HOURS:
                close_pos   = True
                exit_reason = f"⏰ SÜRE DOLDU ({pos_age_h:.1f}sa)"

            in_grace = (time.time() - _startup_time) < _STARTUP_GRACE_SECONDS
            if in_grace and close_pos:
                close_pos = False

    # ── KAPATMA ──────────────────────────────────────────────────────────────
    if close_pos and st["active_pos"] is not None:
        pos     = st["active_pos"]
        entry_p = pos["entry_price"]
        side    = pos["side"]
        size    = pos["size"]
        base_sz = pos.get("base_size", size / max(1, pos.get("leverage", 10)))

        price_diff_pct = (price - entry_p)/entry_p if side == "LONG" \
                         else (entry_p - price)/entry_p
        trade_pnl = size * price_diff_pct
        exit_fee  = size * (1 + price_diff_pct) * 0.0006
        net_pnl   = trade_pnl - exit_fee

        st["pnl"]    += net_pnl; st["trades"] += 1
        if net_pnl > 0: st["wins"] += 1
        portfolio["total_fees"]            += exit_fee
        portfolio["capital"]               += trade_pnl - exit_fee
        portfolio["total_closed_notional"] += base_sz
        portfolio["total_closed_trades"]   += 1

        # ML Feedback
        if st.get("entry_feat") is not None:
            ml_engine.record_trade_result(
                features=st["entry_feat"], side=side,
                net_pnl=net_pnl, symbol=symbol)
            st["entry_feat"] = None

        # RL Online Experience — trade sonucu deneyime ekle
        if st.get("rl_state") is not None:
            # Ödülü hesapla
            rl_reward = float(np.tanh(net_pnl / 10.0))
            # Kapatma aksiyonu: FLAT (0)
            rl_experience.add(
                state      = st["rl_state"],
                action     = 0,  # FLAT — pozisyon kapatma
                reward     = rl_reward,
                next_state = st["rl_state"],  # Basitleştirilmiş
                done       = True,
            )
            st["rl_state"] = None

        exit_time_raw = datetime.utcnow()
        tr_exit = (exit_time_raw + timedelta(hours=3)).strftime("%d.%m.%Y %H:%M:%S")
        secs    = int((exit_time_raw - pos["entry_time_raw"]).total_seconds())
        h, rem  = divmod(secs, 3600); m, s = divmod(rem, 60)
        dur = f"{h}sa {m}dk {s}sn" if h > 0 else f"{m}dk {s}sn"

        asyncio.create_task(send_telegram_message(
            f"<b>✅ KAPANDI: {symbol}</b> ({exit_reason})\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"📥 Giriş: <code>{entry_p:.10g}</code>\n"
            f"📤 Çıkış: <code>{price:.10g}</code>\n"
            f"💵 Net PnL: <b>{'🟢' if net_pnl>0 else '🔴'} ${net_pnl:.2f}</b>\n"
            f"⏳ {dur} | 🕒 {tr_exit} (TR)\n"
            f"🏦 Kasa: <b>${portfolio['capital']:.2f}</b>\n"
            f"🤖 RL Buffer: {rl_experience.size()} trade"
        ))

        trade_history.appendleft({
            "symbol": symbol, "entry_price": entry_p, "exit_price": price,
            "side": side, "pnl": round(net_pnl, 2), "reason": exit_reason,
            "duration": dur, "time": tr_exit,
            "leverage": pos.get("leverage", 1),
            "notional_size": round(base_sz, 2),
            "hybrid_source": pos.get("hybrid_source", "ML"),
        })
        st["active_pos"]     = None
        st["last_exit_time"] = exit_time_raw
        st["last_trade_pnl"] = round(net_pnl, 2)
        save_data()

    # ── SONUÇ ────────────────────────────────────────────────────────────────
    return {
        "symbol":    symbol, "price": price,
        "change24h": round(change24h, 2), "volume24h": round(volume24h, 2),
        "signal":    prediction["signal"], "confidence": prediction["confidence"],
        "indicators":prediction.get("indicators", []),
        "model_used":prediction.get("model", "PPO+ML"),
        "entry_price":price,
        "stop_loss":  sl if price > 0 else 0,
        "take_profit":tp if price > 0 else 0,
        "leverage":   lev,
        "gbm_signal": prediction.get("gbm_signal", "WAIT"),
        "rf_signal":  prediction.get("rf_signal",  "WAIT"),
        "rl_signal":  prediction.get("rl_signal",  "WAIT"),
        "rl_conf":    prediction.get("rl_conf",    0),
        "hybrid_source": prediction.get("hybrid_source", "ML_only"),
        "model_trained":   prediction.get("model_trained", False),
        "wf_accuracy":     prediction.get("wf_accuracy", 0),
        "backtest_roi":    prediction.get("backtest_roi", 0),
        "backtest_sharpe": prediction.get("backtest_sharpe", 0),
        "drift_status":    prediction.get("drift_status", {}),
        "timestamp":  datetime.utcnow().isoformat(),
        "data_source":"real" if klines else "simulated",
        "active_pos": st.get("active_pos"),
        "pnl":        round(st["pnl"], 2),
        "trades":     st["trades"],
        "win_rate":   round(st["wins"]/st["trades"]*100, 1) if st["trades"] > 0 else 0,
    }

async def broadcast_loop():
    await asyncio.sleep(5)
    while True:
        if active_connections and scanner_cache:
            msg = json.dumps({
                "type":       "update",
                "data":       list(scanner_cache.values()),
                "portfolio":  portfolio,
                "timestamp":  datetime.utcnow().isoformat(),
                "total_pairs":len(scanner_cache),
                "ml_status":  {
                    "trained":              ml_engine._trained,
                    "wf_accuracy":          ml_engine._wf.get("accuracy", 0),
                    "feedback_size":        ml_engine.feedback.size(),
                    "drift":                ml_engine.drift.get_status(),
                    "training_in_progress": ml_engine._training_in_progress,
                },
                "rl_status": {
                    "trained":    rl_agent._is_trained,
                    "total_steps":rl_agent._total_steps,
                    "avg_reward": round(float(np.mean(rl_agent._episode_rewards[-20:]))
                                       if rl_agent._episode_rewards else 0, 4),
                    "exp_buffer": rl_experience.size(),
                    "exp_total":  rl_experience.total(),
                    "ewc_active": rl_agent._ewc_fisher is not None,
                },
            })
            dead = []
            for ws in active_connections[:]:
                try:   await ws.send_text(msg)
                except:dead.append(ws)
            for ws in dead:
                if ws in active_connections: active_connections.remove(ws)
        await asyncio.sleep(5)

# ── REST Endpoints ────────────────────────────────────────────────────────────
@app.get("/api/scan")
async def get_scan():
    return {"success":True,"data":list(scanner_cache.values()),
            "total":len(scanner_cache),"timestamp":datetime.utcnow().isoformat()}

@app.get("/api/stats")
async def get_stats():
    if not scanner_cache: return {"success":True,"data":{}}
    vals = list(scanner_cache.values())
    total_pnl    = sum(agent_states.get(v["symbol"],{}).get("pnl",0) for v in vals)
    total_trades = sum(agent_states.get(v["symbol"],{}).get("trades",0) for v in vals)
    total_wins   = sum(agent_states.get(v["symbol"],{}).get("wins",0) for v in vals)
    al = sum(1 for v in vals if v.get("active_pos") and v["active_pos"]["side"]=="LONG")
    as_= sum(1 for v in vals if v.get("active_pos") and v["active_pos"]["side"]=="SHORT")
    cl = sum(1 for h in trade_history if h["side"]=="LONG")
    cs = sum(1 for h in trade_history if h["side"]=="SHORT")
    return {"success":True,"data":{
        "total_pairs":len(vals), "total_pnl":round(total_pnl,2),
        "total_trades":total_trades,
        "win_rate":round(total_wins/total_trades*100,1) if total_trades else 0,
        "active_longs":al, "active_shorts":as_,
        "closed_longs":cl, "closed_shorts":cs,
        "portfolio":portfolio,
        "model_accuracy":round(ml_engine.get_accuracy(),1),
        "total_closed_notional":round(portfolio.get("total_closed_notional",0),2),
        "total_closed_trades":portfolio.get("total_closed_trades",0),
        "timestamp":datetime.utcnow().isoformat()
    }}

@app.get("/api/model")
async def get_model_info():
    info = ml_engine.get_info()
    rl_info = rl_agent.get_info()
    rl_info["exp_buffer"] = rl_experience.size()
    rl_info["exp_total"]  = rl_experience.total()
    rl_info["avg_reward"] = rl_info.get("avg_reward_20", 0)
    info["rl"] = rl_info
    # version_store train_log'dan versiyonları da ekle
    info["versions"] = ml_engine.version_store.get_versions()
    return {"success":True,"data":info}

@app.get("/api/rl")
async def get_rl_info():
    info = rl_agent.get_info()
    # OnlineExperienceBuffer bilgilerini ekle
    info["exp_buffer"] = rl_experience.size()
    info["exp_total"]  = rl_experience.total()
    info["avg_reward"] = info.get("avg_reward_20", 0)
    return {"success":True,"data":info}

@app.get("/api/history")
async def get_trade_history():
    return {"success":True,"data":list(trade_history)}

@app.post("/api/train_rl")
async def train_rl():
    """Manuel RL eğitimi tetikle."""
    klines_list = list(_klines_cache.values())[:10]
    if not klines_list: return {"success":False,"error":"Cache boş"}
    result = await asyncio.to_thread(_rl_pretrain, klines_list)
    return {"success":True,"data":rl_agent.get_info()}

@app.post("/api/train/{symbol}")
async def trigger_training(symbol: str):
    sym = symbol.upper().replace("-","_")
    async with httpx.AsyncClient() as client:
        klines = await fetch_klines(client, sym, limit=300)
    if not klines: return {"success":False,"error":"MEXC kline verisi alınamadı"}
    result = await asyncio.to_thread(ml_engine.train, klines, sym, True)
    if result.get("success"):
        await asyncio.to_thread(ml_engine.save, ML_MODEL_FILE)
        ml_engine._retrain_triggers["manual"] += 1
    return {"success":True,"data":result}

@app.post("/api/train_all")
async def train_all():
    klines_list = []
    async with httpx.AsyncClient() as client:
        for sym in FUTURES_PAIRS[:20]:
            kl = await fetch_klines(client, sym, limit=300)
            if kl: klines_list.append(kl)
    if not klines_list: return {"success":False,"error":"Veri alınamadı"}
    ml_result = await asyncio.to_thread(
        ml_engine.train_on_multi_pair, klines_list, "MANUAL_ALL")
    if ml_result.get("success"):
        await asyncio.to_thread(ml_engine.save, ML_MODEL_FILE)
        ml_engine._retrain_triggers["manual"] += 1
        # RL'i de güncelle
        await asyncio.to_thread(_rl_pretrain, klines_list[:5])
    return {"success":True,"data":ml_result}

@app.get("/api/backtest_all")
async def run_backtest_all():
    return {"success":True,"data":ml_engine._bt}

@app.get("/api/pair/{symbol}")
async def get_pair(symbol: str):
    sym = symbol.upper().replace("-","_")
    if sym in scanner_cache: return {"success":True,"data":scanner_cache[sym]}
    return {"success":False,"error":"Pair not found"}

@app.get("/api/drift")
async def get_drift():
    return {"success":True,"data":ml_engine.drift.get_status()}

@app.get("/api/versions")
async def get_versions():
    return {"success":True,"data":ml_engine.version_store.get_versions()}

@app.get("/health")
async def health():
    return {
        "status":       "ok",
        "timestamp":    datetime.utcnow().isoformat(),
        "total_pairs":  len(scanner_cache),
        "ml_trained":   ml_engine._trained,
        "rl_trained":   rl_agent._is_trained,
        "rl_steps":     rl_agent._total_steps,
        "wf_accuracy":  ml_engine._wf.get("accuracy",0),
        "rl_exp_buffer":rl_experience.size(),
    }

@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    try:
        if scanner_cache:
            await websocket.send_text(json.dumps({
                "type":"init","data":list(scanner_cache.values()),
                "portfolio":portfolio,"timestamp":datetime.utcnow().isoformat()}))
        while True: await websocket.receive_text()
    except (WebSocketDisconnect, Exception):
        if websocket in active_connections: active_connections.remove(websocket)

# ── Frontend ──────────────────────────────────────────────────────────────────
frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(frontend_path):
    app.mount("/static", StaticFiles(directory=frontend_path), name="static")

@app.get("/ml-details")
async def serve_ml_details():
    idx = os.path.join(frontend_path, "ml_details.html")
    return FileResponse(idx) if os.path.exists(idx) else {"error":"bulunamadı"}

@app.get("/")
async def serve_frontend():
    idx = os.path.join(frontend_path, "index.html")
    return FileResponse(idx) if os.path.exists(idx) else \
           {"message":"MEXC RL+ML Trader v3.0"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
