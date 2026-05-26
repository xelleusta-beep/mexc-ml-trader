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
from config     import config
from risk_manager import RiskManager
from monitor   import MetricsTracker, Timer
from backtester import backtest_simple, portfolio_backtest, SlippageConfig, FeeConfig
from cache     import FeatureCache, AsyncBatchProcessor, get_http_client, close_http_client, timed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Telegram ──────────────────────────────────────────────────────────────────
async def send_telegram_message(text: str):
    if not config.telegram_bot_token or not config.telegram_chat_id:
        logger.info(f"[TG-SKIP] {text[:80]}"); return
    try:
        async with httpx.AsyncClient() as client:
            await client.post(
                f"https://api.telegram.org/bot{config.telegram_bot_token}/sendMessage",
                json={"chat_id": config.telegram_chat_id, "text": text, "parse_mode": "HTML"},
                timeout=10)
    except Exception as e:
        logger.error(f"Telegram error: {e}")

# ── Persistence ───────────────────────────────────────────────────────────────
PERSISTENCE_FILE = config.persistence_file
ML_MODEL_FILE    = config.ml_model_file
RL_MODEL_FILE    = config.rl_model_file


# ── DEPLOY VERSİYON KAYDI ────────────────────────────────────────────────────
DEPLOY_CHANGELOG = [
    {
        "version": "v3.6",
        "date": "2026-05-20",
        "title": "Kritik Düzeltmeler: Zaman Aşımı İptali + SL/TP + Win Rate",
        "changes": [
            "8 saatlik zaman aşımı kapatma TAMAMEN İPTAL edildi",
            "SL/TP asimetrisi: 3:1 R/R oranı (SL%1.8, TP%5.4)",
            "signals_agree katılaştırıldı: GBM+RF MUTLAKA hemfikir olmalı",
            "Min hacim filtresi 50M→100M (daha likit pairler)",
            "RL online update trigger 40→20 trade",
            "RL ödül sinyali güçlendirildi (/10→/5)",
            "Label threshold düşürüldü (daha dengeli LONG/SHORT)",
            "Feedback threshold 200→100 (daha hızlı öğrenme)",
            "Multi-pair backtest (daha gerçekçi metrik)",
            "Lookahead 5→3 bar (5x kaldıraç için uygun)",
        ],
        "metrics": {"win_rate_target": "50%+", "rr_ratio": "3:1"}
    },
    {
        "version": "v3.5",
        "date": "2026-05-17",
        "title": "WF Koruma + Temporal Retrain Kapatıldı",
        "changes": [
            "Temporal retrain KAPATILDI (999 saat) — WF %37'ye düşüyordu",
            "WF rollback: yeni model eskisinden %2+ kötüyse eski model korunuyor",
            "LONG/SHORT simetri eşiği 1.5→1.2 (SHORT bias azaltıldı)",
            "Online update trigger 50→40 trade",
            "Retrain pair sayısı 15→20",
            "Max açık pozisyon 15→20",
        ],
        "metrics": {"wf_target": "40%+", "win_rate_target": "55%+"}
    },
    {
        "version": "v3.4",
        "date": "2026-05-14",
        "title": "RL Canlıya Alındı + Backtest Düzeltme",
        "changes": [
            "RL conf normalizasyonu: uniform baseline'a göre 50-99 arası",
            "ML_only conf çarpanı 0.7→0.85",
            "signals_agree eşiği %65→%57",
            "Backtest: 8-bar SL/TP simülasyonu eklendi",
            "profit_factor backtest'e eklendi",
            "Version WF/Sharpe kayıt sorunu düzeltildi",
        ],
        "metrics": {"rl_buffer": "45 trade", "win_rate": "62.4%"}
    },
    {
        "version": "v3.3",
        "date": "2026-05-12",
        "title": "RL Ödül ve Conf Düzeltmeleri",
        "changes": [
            "FLAT cezası 0.02→0.005 (-4.0 ödül sorunu çözüldü)",
            "Episode length 200→100",
            "Reward clipping asimetrik (-1, +3)",
            "RL iterasyon 30→100",
            "Conf threshold %55→%52",
        ],
        "metrics": {"rl_reward_before": "-4.0", "rl_reward_after": "-1.0"}
    },
    {
        "version": "v3.2",
        "date": "2026-05-11",
        "title": "Bağlantı Düzeltmeleri + RL Konfig",
        "changes": [
            "renderTicker JS syntax hatası düzeltildi (tüm butonlar çalışmıyordu)",
            "WebSocket yeniden bağlanma sayacı eklendi",
            "Health check + AbortSignal.timeout",
            "Render uyku modu uyarısı eklendi",
            "RL entropy 0.01→0.05 (daha fazla keşif)",
        ],
        "metrics": {}
    },
    {
        "version": "v3.1",
        "date": "2026-05-10",
        "title": "RL+ML Hibrit Sistem İlk Deploy",
        "changes": [
            "PPO Agent (NumPy tabanlı) entegre edildi",
            "Hibrit karar: RL+ML / ML_only / RL_only",
            "EWC (Elastic Weight Consolidation) — felaket unutma önleme",
            "Online fine-tuning her 50 trade sonrası",
            "index.html: RL kolonları eklendi (RL Sig, RL Güv, Kaynak)",
            "ml_details.html: 7 sekme ile tam detay sayfası",
        ],
        "metrics": {}
    },
]

def get_deploy_info() -> dict:
    """En son deploy bilgisini döndür."""
    current = DEPLOY_CHANGELOG[0]
    return {
        "current_version": current["version"],
        "deploy_date":     current["date"],
        "title":           current["title"],
        "total_versions":  len(DEPLOY_CHANGELOG),
        "changelog":       DEPLOY_CHANGELOG,
    }

# ── Globals ───────────────────────────────────────────────────────────────────
feature_cache      = FeatureCache(max_size=1000, ttl=120.0)
batch_processor    = AsyncBatchProcessor(batch_size=config.scanner_batch_size, interval=0.3)
ml_engine          = MLEngine(model_dir=config.persist_dir, use_v2_features=config.ml.use_v2_features, feature_cache=feature_cache)
rl_agent           = PPOAgent(
    state_dim=config.rl.state_dim,
    n_actions=config.rl.n_actions,
    hidden_dim=config.rl.hidden_dim,
    lr=config.rl.learning_rate,
)
rl_experience      = OnlineExperienceBuffer()
risk_mgr           = RiskManager(config)
metrics_tracker    = MetricsTracker()

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
MAX_OPEN_POSITIONS = config.trading.max_open_positions
MIN_VOLUME_24H     = config.trading.min_volume_24h
MIN_PRICE          = config.trading.min_price

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
                "risk_state":    risk_mgr.get_save_state(),
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
        risk_mgr.load_state(data.get("risk_state", {}))
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
MEXC_BASE = config.api.mexc_base_url

async def fetch_all_futures_pairs(client):
    try:
        r = await client.get(f"{MEXC_BASE}/detail", timeout=config.api.http_timeout_detail)
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
        r = await client.get(f"{MEXC_BASE}/ticker", params={"symbol": symbol}, timeout=config.api.http_timeout_ticker)
        if r.status_code == 200:
            d = r.json()
            if d.get("success") and d.get("data"): return d["data"]
    except Exception as e:
        logger.debug(f"Ticker {symbol}: {e}")
    return None

async def fetch_klines(client, symbol, interval=None, limit=None):
    if interval is None: interval = config.api.kline_interval
    if limit is None:    limit    = config.api.kline_limit
    try:
        r = await client.get(f"{MEXC_BASE}/kline/{symbol}",
                             params={"interval": interval, "limit": limit}, timeout=config.api.http_timeout_kline)
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
    # ML tahmini (latency takibi ile)
    with Timer(metrics_tracker, f"ml_predict_{symbol}"):
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

    # 2. ML LONG/SHORT ama RL WAIT → ML'e güven (daha az indirim)
    elif ml_sig in ("LONG", "SHORT") and rl_sig == "WAIT":
        final_sig  = ml_sig
        final_conf = ml_conf * 0.85  # FIX: 0.7→0.85, daha fazla işlem açılsın
        final_lev  = 5               # Küçük pozisyon
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

    # Conf eşiği — config'den
    if final_conf < config.trading.min_confidence_entry:
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
    metrics_tracker.record_prediction(symbol, final_sig, final_conf, 0, source)
    return result

# ── EĞİTİM DÖNGÜSÜ ───────────────────────────────────────────────────────────
async def auto_train_on_startup():
    try:
        await asyncio.sleep(35)
        logger.info("🤖 Başlangıç eğitimi başlıyor (ML + RL)...")
        
        # Scanner'ın FUTURES_PAIRS doldurmasını bekle (en fazla 3 dk)
        for _ in range(18):
            if FUTURES_PAIRS:
                break
            await asyncio.sleep(10)
        
        klines_list = []
        client = await get_http_client()
        if client is None:
            logger.warning("HTTPX hazır değil, baslangıç eğitimi atlandı")
            return
        targets = FUTURES_PAIRS[:20] if FUTURES_PAIRS else \
                  ["BTC_USDT","ETH_USDT","SOL_USDT","BNB_USDT","XRP_USDT"]
        for sym in targets:
            try:
                kl = await fetch_klines(client, sym, limit=300)
                if kl:
                    klines_list.append(kl); _klines_cache[sym] = kl
                    logger.info(f"  {sym}: {len(kl['close'])} mum")
            except Exception as e:
                logger.warning(f"  {sym} kline hatası: {e}")
            await asyncio.sleep(0.5)

        if not klines_list:
            logger.warning("Başlangıç eğitimi: MEXC API'den veri alınamadı, 60sn sonra tekrar deneniyor")
            await asyncio.sleep(60)
            # ikinci deneme — fallback pair'lerle
            for sym in ["BTC_USDT","ETH_USDT","SOL_USDT","BNB_USDT","XRP_USDT"]:
                try:
                    kl = await fetch_klines(client, sym, limit=300)
                    if kl:
                        klines_list.append(kl); _klines_cache[sym] = kl
                except Exception:
                    pass
                await asyncio.sleep(0.3)
            if not klines_list:
                logger.warning("Başlangıç eğitimi: MEXC API erişilemiyor, sonraki retrain döngüsüne bırakıldı")
                return

        # ML eğitimi
        ml_result = await asyncio.to_thread(
            ml_engine.train_on_multi_pair, klines_list, "GLOBAL")
        metrics_tracker.record_train(ml_result)
        if ml_result.get("success"):
            logger.info(f"✅ ML eğitimi: wf={ml_result.get('wf_accuracy')}% | n={ml_result.get('n_samples')}")
            await asyncio.to_thread(ml_engine.save, ML_MODEL_FILE)
        else:
            logger.warning(f"⚠️ ML eğitimi başarısız: {ml_result}")

        # RL ön-eğitimi (backtesting üzerinde)
        await asyncio.to_thread(_rl_pretrain, klines_list)
        # Multi-pair backtest çalıştır
        await asyncio.to_thread(ml_engine.run_backtest_multi, klines_list)
    except Exception as e:
        logger.error(f"Başlangıç eğitimi hatası: {e}")
        metrics_tracker.record_error()

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
        features_list, prices_list, n_iterations=config.rl.rl_pretrain_iterations)
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
                    for sym in (FUTURES_PAIRS[:20] or ["BTC_USDT","ETH_USDT","SOL_USDT"]):  # FIX: 15→20
                        kl = _klines_cache.get(sym) or \
                             await fetch_klines(client, sym, limit=300)
                        if kl: klines_list.append(kl)
                        await asyncio.sleep(0.3)

                if klines_list:
                    # Mevcut WF'yi kaydet
                    prev_wf = ml_engine._wf.get("accuracy", 0)
                    result = await asyncio.to_thread(
                        ml_engine.train_on_multi_pair, klines_list, f"RETRAIN_{reason}")
                    if result.get("success"):
                        metrics_tracker.record_train(result)
                        new_wf = result.get("wf_accuracy", 0)
                        # FIX: Yeni model eskisinden %3'den fazla kötüyse logla
                        if new_wf < prev_wf - 3:
                            logger.warning(f"⚠️ WF düştü: {prev_wf}% → {new_wf}% [{reason}]")
                        logger.info(f"✅ ML Retrain [{reason}] | wf={new_wf}% (önceki:{prev_wf}%)")
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
                            # Multi-pair backtest güncelle
                            await asyncio.to_thread(ml_engine.run_backtest_multi, klines_list)

            # RL online update
            if rl_experience.ready():
                exps = rl_experience.get_and_clear()
                losses = await asyncio.to_thread(rl_agent.online_update, exps)
                await asyncio.to_thread(rl_agent.save, RL_MODEL_FILE)
                avg_rew = float(np.mean(rl_agent._episode_rewards[-20:])) if rl_agent._episode_rewards else 0
                logger.info(f"✅ RL Online Update: {len(exps)} trade | "
                            f"loss={losses.get('policy_loss', 'N/A')} | "
                            f"avg_reward={avg_rew:.4f}")

        except Exception as e:
            logger.error(f"Retrain döngüsü hatası: {e}")
        await asyncio.sleep(config.retrain_loop_interval_sec)

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
    await close_http_client()
    await asyncio.to_thread(ml_engine.save, ML_MODEL_FILE)
    await asyncio.to_thread(rl_agent.save, RL_MODEL_FILE)
    logger.info("Sistem kapatıldı")

# ── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI(title="MEXC RL+ML Trader v3.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# ── Scanner ───────────────────────────────────────────────────────────────────
_startup_time: float = 0.0
_STARTUP_GRACE_SECONDS = config.startup_grace_sec

async def scanner_loop():
    global FUTURES_PAIRS, _startup_time
    _startup_time = time.time()
    while True:
        try:
            client = await get_http_client()
            if client is None:
                logger.warning("HTTPX yuklu degil, scanner pasif")
                await asyncio.sleep(60); continue
            new_pairs = await fetch_all_futures_pairs(client)
            if new_pairs: FUTURES_PAIRS = new_pairs
            if not FUTURES_PAIRS:
                await asyncio.sleep(10); continue
            batch_sz = config.scanner_batch_size
            for i in range(0, len(FUTURES_PAIRS), batch_sz):
                batch = FUTURES_PAIRS[i:i+batch_sz]
                results = await asyncio.gather(
                    *[process_pair(client, sym) for sym in batch],
                    return_exceptions=True)
                for sym, res in zip(batch, results):
                    if isinstance(res, dict):
                        scanner_cache[sym] = res
                await asyncio.sleep(0.3)  # FIX: 0.5->0.3sn rate limit
            logger.info(f"Tarama tamamlandı — {len(scanner_cache)} pair")
        except Exception as e:
            logger.error(f"Scanner hata: {e}")
            metrics_tracker.record_error()
        await asyncio.sleep(config.scanner_interval_sec)

async def process_pair(client, symbol: str) -> dict:
    t0 = time.perf_counter()
    ticker = await fetch_ticker(client, symbol)
    klines = await fetch_klines(client, symbol, limit=None)

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

    # Asimetrik SL/TP — config'den leverage bazinda
    sl_tp = config.trading.sl_tp_levels.get(lev) or \
            config.trading.sl_tp_levels.get(5) or (0.018, 0.054)
    sl_pct, tp_pct = sl_tp

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
    # FIX: Daha katı giriş — GBM ve RF MUTLAKA hemfikir olmalı
    gbm_rf_agree = (gbm_sig == rf_sig == main_sig)
    signals_agree = (
        (source == "RL+ML" and gbm_rf_agree) or
        (source == "ML_only" and conf_val >= config.trading.min_confidence_ml_only and gbm_rf_agree) or
        (source == "RL_only" and conf_val >= config.trading.min_confidence_rl_only) or
        (gbm_rf_agree and conf_val >= config.trading.min_confidence_gbm_rf)
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
        cooldown_sec = config.trading.cooldown_loss_sec if last_pnl < 0 else config.trading.cooldown_win_sec
        if (datetime.utcnow() - st["last_exit_time"]).total_seconds() < cooldown_sec:
            can_enter = False

    if can_enter:
        tp_dist = abs(tp - price) / price if price > 0 else 0
        sl_dist = abs(sl - price) / price if price > 0 else 0
        if tp_dist < 0.001 or sl_dist < 0.001:
            can_enter = False

    # ── RİSK KONTROLÜ ─────────────────────────────────────────────────────────
    if can_enter:
        risk_ok, risk_reason = risk_mgr.check_entry_allowed(
            symbol, main_sig, conf_val, price, portfolio, agent_states, ml_engine
        )
        if not risk_ok:
            can_enter = False

    # ── GİRİŞ ────────────────────────────────────────────────────────────────
    if can_enter:
        leverage       = lev
        entry_time_raw = datetime.utcnow()
        tr_time        = (entry_time_raw + timedelta(hours=3)).strftime("%d.%m.%Y %H:%M:%S")
        base_size      = risk_mgr.calculate_position_size(
            symbol, price, conf_val, portfolio, agent_states
        )
        notional       = base_size * leverage
        entry_fee      = notional * config.trading.fee_rate
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
            # DEVRE DIŞI: Zaman aşımı kapatma iptal edildi
            # if not close_pos and pos_age_h >= MAX_POS_HOURS:
            #     close_pos   = True
            #     exit_reason = f"⏰ SÜRE DOLDU ({pos_age_h:.1f}sa)"

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
            rl_reward = float(np.tanh(net_pnl / 5.0))  # FIX: /10→/5
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
        risk_mgr.record_trade_result(net_pnl, symbol, side)
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
                "risk_status": risk_mgr.get_status(),
                "monitor":     metrics_tracker.get_status(),
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
    """Mevcut ML engine backtest + detayli metrikler."""
    bt = ml_engine._bt
    result = dict(bt) if isinstance(bt, dict) else {}
    result["engine"] = "ml_engine_v2"

    # Yeni backtester ile de degerlendir
    if scanner_cache and ml_engine._trained:
        cached_signals = {}
        for sym, data in list(scanner_cache.items())[:20]:
            sig = data.get("signal", "").upper()
            if sig in ("LONG", "SHORT"):
                px = data.get("price", 0)
                conf = data.get("confidence", 0)
                if px <= config.trading.min_price:
                    continue
                src = data.get("hybrid_source", "")
                cached_signals[sym] = (px, sig, conf, src)

        if cached_signals:
            # Yeni backtester'dan guncel metrikler
            fee_cfg = FeeConfig(maker_bps=config.backtest.maker_bps,
                                taker_bps=config.backtest.taker_bps)
            slip_cfg = SlippageConfig(spread_bps=config.backtest.spread_bps,
                                      vol_slip_bps=config.backtest.vol_slip_bps)
            bt2 = backtest_simple(
                np.array([v[0] for v in cached_signals.values()]),
                [(i, v[1]) for i, v in enumerate(cached_signals.values())],
                sl_pct=config.backtest.sl_pct,
                tp_pct=config.backtest.tp_pct,
                leverage=config.backtest.leverage,
                fee_conf=fee_cfg,
                slip_conf=slip_cfg,
                max_bars_hold=config.backtest.max_bars_hold,
            )
            result["advanced"] = {
                "roi_pct":       bt2.roi_pct,
                "win_rate":      bt2.win_rate,
                "max_drawdown":  bt2.max_drawdown_pct,
                "sharpe":        bt2.sharpe,
                "sortino":       bt2.sortino,
                "calmar":        bt2.calmar,
                "profit_factor": bt2.profit_factor,
                "avg_win_pct":   bt2.avg_win_pct,
                "avg_loss_pct":  bt2.avg_loss_pct,
                "expectancy":    bt2.expectancy,
                "n_trades":      bt2.n_trades,
                "timing_ms":     bt2.timing_ms,
            }
            result["slip_cfg"] = {"spread_bps": slip_cfg.spread_bps, "vol_slip_bps": slip_cfg.vol_slip_bps}
            result["fee_cfg"]  = {"maker_bps": fee_cfg.maker_bps, "taker_bps": fee_cfg.taker_bps}

    result["config"] = {
        "sl_pct": config.backtest.sl_pct,
        "tp_pct": config.backtest.tp_pct,
        "leverage": config.backtest.leverage,
        "max_bars_hold": config.backtest.max_bars_hold,
    }
    return {"success":True,"data":result}

@app.get("/api/backtest/run")
async def run_custom_backtest(symbol: str = "", sl_pct: float = 0.022,
                               tp_pct: float = 0.05, leverage: int = 5,
                               max_bars_hold: int = 8):
    """Detayli backtest (tek pair).
    Query params: ?symbol=BTC_USDT&sl_pct=0.02&tp_pct=0.05&leverage=5
    """
    bt_result = {"success": False, "error": "Henuz calismadi"}

    if symbol and symbol in scanner_cache:
        data = scanner_cache[symbol]
        sig = data.get("signal", "").upper()
        px  = data.get("price", 0)
        conf = data.get("confidence", 0)

        if sig not in ("LONG", "SHORT") or px <= 0:
            return {"success": False, "error": f"{symbol}: uygun sinyal yok ({sig})"}

        fee_cfg = FeeConfig(maker_bps=config.backtest.maker_bps,
                            taker_bps=config.backtest.taker_bps)
        slip_cfg = SlippageConfig(spread_bps=config.backtest.spread_bps,
                                  vol_slip_bps=config.backtest.vol_slip_bps)

        prices = np.array([px])
        signals = [(0, sig)]

        bt = backtest_simple(prices, signals, sl_pct, tp_pct, leverage,
                            fee_cfg, slip_cfg, symbol=symbol,
                            max_bars_hold=max_bars_hold)

        return {"success": True, "data": {
            "symbol":       bt.symbol,
            "n_trades":     bt.n_trades,
            "roi_pct":      bt.roi_pct,
            "win_rate":     bt.win_rate,
            "max_drawdown": bt.max_drawdown_pct,
            "sharpe":       bt.sharpe,
            "sortino":      bt.sortino,
            "calmar":       bt.calmar,
            "profit_factor": bt.profit_factor,
            "avg_win_pct":  bt.avg_win_pct,
            "avg_loss_pct": bt.avg_loss_pct,
            "expectancy":   bt.expectancy,
            "total_fee_pct": bt.total_fee_pct,
            "timing_ms":    bt.timing_ms,
            "params": {
                "sl_pct":       sl_pct,
                "tp_pct":       tp_pct,
                "leverage":     leverage,
                "max_bars_hold": max_bars_hold,
                "spread_bps":   slip_cfg.spread_bps,
                "taker_bps":    fee_cfg.taker_bps,
                "maker_bps":    fee_cfg.maker_bps,
            }
        }}
    return {"success": False, "error": f"Symbol '{symbol}' scanner_cache'de bulunamadi veya gecersiz"}

@app.get("/api/pair/{symbol}")
async def get_pair(symbol: str):
    sym = symbol.upper().replace("-","_")
    if sym in scanner_cache: return {"success":True,"data":scanner_cache[sym]}
    return {"success":False,"error":"Pair not found"}

@app.get("/api/drift")
async def get_drift():
    return {"success":True,"data":ml_engine.drift.get_status()}

@app.get("/api/risk")
async def get_risk():
    """Risk yonetim durumu."""
    risk_mgr.update_capital(portfolio.get("capital", 0))
    return {"success":True,"data":risk_mgr.get_status()}

@app.get("/api/monitor")
async def get_monitor():
    """Sistem metrikleri."""
    status = metrics_tracker.get_status()
    status["cache"]    = feature_cache.stats
    status["batch"]    = batch_processor.stats
    return {"success":True,"data":status}

@app.get("/api/feature_importance")
async def get_feature_importance(top_k: int = 20):
    """Feature importance listesi."""
    try:
        data = ml_engine.get_feature_importance(top_k=top_k)
        return {"success":True,"data":data}
    except Exception as e:
        return {"success":False,"error":f"feature_importance error: {e}","data":[]}

@app.get("/api/train_log")
async def get_train_log():
    """Egitim gecmisi."""
    return {"success":True,"data":metrics_tracker.get_train_history()}


@app.get("/api/versions/deploy")
async def get_deploy_versions():
    """Deploy versiyonlarını ve changelog'u döndür."""
    return {"success": True, "data": get_deploy_info()}

@app.get("/api/versions")
async def get_versions():
    return {"success":True,"data":ml_engine.version_store.get_versions()}

@app.post("/api/rollback/{idx}")
async def rollback_version(idx: int):
    """Belirtilen versiyona geri dön."""
    versions = ml_engine.version_store.get_versions()
    if idx < 0 or idx >= len(versions):
        return {"success":False,"error":"Geçersiz indeks"}
    target = versions[idx]
    path   = target.get("path")
    if not path or not os.path.exists(path):
        return {"success":False,"error":"Model dosyası bulunamadı"}
    if ml_engine.load(path):
        logger.info(f"✅ Rollback: v{idx} yüklendi | wf={target.get('wf_accuracy',0)}%")
        await asyncio.to_thread(ml_engine.save, ML_MODEL_FILE)
        return {"success":True,"data":{"restored_version":idx,"wf_accuracy":target.get('wf_accuracy',0)}}
    return {"success":False,"error":"Model yüklenemedi"}

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
    uvicorn.run("main:app", host="0.0.0.0", port=config.port, reload=False)
