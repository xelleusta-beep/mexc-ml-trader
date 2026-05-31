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
import hashlib
import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn

from ml_engine import MLEngine, FeatureBuilder
from features import FeatureBuilderV2
from rl_engine  import PPOAgent, TradingEnvironment, OnlineExperienceBuffer
from config     import config
from risk_manager import RiskManager
from persistence import StateStore
from monitor   import MetricsTracker, Timer
from backtester import backtest_simple, portfolio_backtest, SlippageConfig, FeeConfig
from cache     import FeatureCache, AsyncBatchProcessor, get_http_client, close_http_client, timed
from orderbook import AdvancedOrderBook, MultiTimeframeConfirmation, DynamicSLTP
from market_data import MarketDataEnrichment
from enhanced_ensemble import EnhancedEnsembleEngine, DynamicModelWeighter
from enhanced_rl import EnhancedPPOAgent, ImprovedRewardFunction

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Telegram ──────────────────────────────────────────────────────────────────
async def send_telegram_message(text: str):
    if not config.telegram_bot_token or not config.telegram_chat_id:
        logger.info(f"[TG-SKIP] {text[:80]}"); return
    try:
        client = await get_http_client()
        if client is None:
            async with httpx.AsyncClient() as temp_client:
                await temp_client.post(
                    f"https://api.telegram.org/bot{config.telegram_bot_token}/sendMessage",
                    json={"chat_id": config.telegram_chat_id, "text": text, "parse_mode": "HTML"},
                    timeout=10)
            return
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
        "version": "v4.0",
        "date": "2026-05-31",
        "title": "V3 Features + V2 RL Agent + WebGL Parallax",
        "changes": [
            "Feature Engineering V3 aktif: 116 ozellik (Multi-Timeframe + Microstructure + Cross-Asset + Regime + Sentiment)",
            "RL Agent V2 aktif: LSTM + Multi-Head Attention + Multi-Task Actor (Entry/Exit/Size)",
            "WebGL parallax arka plan eklendi (partikul sistemi + derinlik katmanlari)",
            "Backtest bolumu duzeltildi: Symbol secici eklendi, calisiyor",
            "Features bolumu guncellendi: V3 feature detaylari tam gorunur",
            "Glassmorphism efektleri eklendi (.glass, .glow-line, .hover-lift)",
            "Risk Manager V2: Dynamic sizing + Portfolio risk + Adaptive drawdown",
            "Data Feeds: Sentiment + Order Book + Cross-Exchange verileri",
            "Monitor V2: Prometheus metrics + Alert rules",
            "Backtester V2: Event-driven + Monte Carlo + Walk-Forward optimization",
        ],
        "metrics": {"features": "116", "rl_architecture": "LSTM+Attention", "webgl": "active"}
    },
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
ml_engine          = MLEngine(model_dir=config.persist_dir, use_v2_features=config.ml.use_v2_features,
                             use_v3_features=config.ml.use_v3_features, feature_cache=feature_cache)
rl_agent           = PPOAgent(
    state_dim=config.rl.state_dim,
    n_actions=config.rl.n_actions,
    hidden_dim=config.rl.hidden_dim,
    lr=config.rl.learning_rate,
)
rl_experience      = OnlineExperienceBuffer()
risk_mgr           = RiskManager(config)
metrics_tracker    = MetricsTracker()

# ── New Modules (Week 1-2) ─────────────────────────────────────────────────
advanced_orderbook = AdvancedOrderBook()
mtf_confirmation   = MultiTimeframeConfirmation()
dynamic_sltp       = DynamicSLTP()

# ── New Modules (Week 3-4) ─────────────────────────────────────────────────
market_data_enrichment = MarketDataEnrichment()

active_connections: List[WebSocket] = []
scanner_cache:  dict = {}
agent_states:   dict = {}
trade_history        = deque(maxlen=200)
_pnl_timeline: list  = []
portfolio = {
    "capital":               5000.0,
    "total_fees":            0.0,
    "initial_capital":       5000.0,
    "total_closed_notional": 0.0,
    "total_closed_trades":   0,
}
FUTURES_PAIRS: List[str] = []
_klines_cache: dict = {}
MAX_OPEN_POSITIONS = config.trading.max_open_positions
MIN_VOLUME_24H     = config.trading.min_volume_24h
MIN_PRICE          = config.trading.min_price

state_lock = asyncio.Lock()

def json_serial(obj):
    if isinstance(obj, datetime): return obj.isoformat()
    raise TypeError(f"Not serializable: {type(obj)}")

async def save_data():
    async with state_lock:
        try:
            # PnL timeline: her trade'den sonra portfolio capital'ini kaydet
            now = datetime.now(timezone.utc)
            pnl_ts = now.isoformat()
            _pnl_timeline.append({"t": pnl_ts[:16], "v": round(portfolio.get("capital", 0) - portfolio.get("initial_capital", 100000), 2)})
            if len(_pnl_timeline) > 500:
                _pnl_timeline[:] = _pnl_timeline[-500:]

            state_to_save = {
                "agent_states":  agent_states,
                "trade_history": list(trade_history),
                "portfolio":     portfolio,
                "risk_state":    risk_mgr.get_save_state(),
                "pnl_timeline":  _pnl_timeline,
                "timestamp":     pnl_ts,
            }
            def write_file():
                with open(PERSISTENCE_FILE, "w") as f:
                    json.dump(state_to_save, f, default=json_serial)
            await asyncio.to_thread(write_file)
        except Exception as e:
            logger.error(f"Save error: {e}")

async def load_data():
    global agent_states, trade_history, portfolio, _pnl_timeline
    if not os.path.exists(PERSISTENCE_FILE): return
    async with state_lock:
        try:
            def read_file():
                with open(PERSISTENCE_FILE) as f:
                    return json.load(f)
            data = await asyncio.to_thread(read_file)
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
                            pos["entry_time_raw"] = datetime.now(timezone.utc)
                    except Exception:
                        pos["entry_time_raw"] = datetime.now(timezone.utc)
                    if not pos.get("tp") or not pos.get("sl"):
                        st["active_pos"] = None
                    if pos and pos.get("entry_price", 0) <= 0:
                        st["active_pos"] = None
            agent_states  = loaded
            trade_history = deque(data.get("trade_history", []), maxlen=200)
            portfolio.update(data.get("portfolio", {}))
            _pnl_timeline = data.get("pnl_timeline", [])
            portfolio.setdefault("total_closed_notional", 0.0)
            portfolio.setdefault("total_closed_trades", 0)
            portfolio["initial_capital"] = 5000.0   # her deployda sabit başlangıç
            portfolio["capital"] = 5000.0            # kasa sıfırlansın
            logger.info(f"Veriler yüklendi — {len(agent_states)} sembol")
        except Exception as e:
            logger.error(f"Load error: {e}")

# ── MEXC API ──────────────────────────────────────────────────────────────────
MEXC_BASE = config.api.mexc_base_url

async def fetch_all_futures_pairs(client):
    """Fetch futures pairs with retry logic."""
    for retry in range(3):
        try:
            r = await client.get(f"{MEXC_BASE}/detail", timeout=config.api.http_timeout_detail)
            if r.status_code == 200:
                d = r.json()
                if d.get("success") and d.get("data"):
                    pairs = [x["symbol"] for x in d["data"]
                             if x.get("state") == 0 and x.get("settleCoin") == "USDT"]
                    logger.info(f"MEXC: {len(pairs)} aktif pair")
                    return pairs
                else:
                    logger.warning(f"MEXC API returned unsuccessful response: {d}")
            elif r.status_code == 429:
                wait = min(60 * (retry + 1), 180)
                logger.warning(f"MEXC rate-limited (429)! {wait}s bekleniyor...")
                await asyncio.sleep(wait)
            else:
                logger.warning(f"MEXC API returned status {r.status_code}")
        except Exception as e:
            logger.error(f"Pair fetch denemesi {retry+1}/3 başarısız: {e}")
            if retry < 2:
                await asyncio.sleep(2 ** retry)
    return []

async def fetch_ticker(client, symbol):
    """Fetch ticker data with retry logic."""
    for retry in range(2):  # Fewer retries for ticker as it's called frequently
        try:
            r = await client.get(f"{MEXC_BASE}/ticker", params={"symbol": symbol}, timeout=config.api.http_timeout_ticker)
            if r.status_code == 200:
                d = r.json()
                if d.get("success") and d.get("data"): 
                    return d["data"]
            else:
                logger.debug(f"Ticker {symbol} returned status {r.status_code}")
        except Exception as e:
            logger.debug(f"Ticker {symbol} denemesi {retry+1}/2 başarısız: {e}")
            if retry < 1:  # Not the last attempt
                await asyncio.sleep(0.5 * (2 ** retry))  # Shorter backoff for frequent calls
    return None

async def fetch_klines(client, symbol, interval=None, limit=None):
    """Fetch klines data with retry logic."""
    if interval is None: interval = config.api.kline_interval
    if limit is None:    limit    = config.api.kline_limit
    for retry in range(2):
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
            else:
                logger.debug(f"Kline {symbol} returned status {r.status_code}")
        except Exception as e:
            logger.debug(f"Kline {symbol} denemesi {retry+1}/2 başarısız: {e}")
            if retry < 1:  # Not the last attempt
                await asyncio.sleep(0.5 * (2 ** retry))  # Shorter backoff for frequent calls
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
        pos_age = int((datetime.now(timezone.utc) -
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
        if rl_conf >= config.trading.min_confidence_rl_only:
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

    # ── PER-COIN MEMORY ADJUSTMENT ────────────────────────────────────────────
    if final_sig in ("LONG", "SHORT") and feat is not None:
        multiplier, hist_sig, n_sim = ml_engine.get_per_coin_adjustment(symbol, feat, k=5)
        if n_sim >= 2:
            # Geçmişte benzer durumlarda bu coin ne yapmış?
            if hist_sig is not None and hist_sig != final_sig:
                # Model ters yönde sinyal veriyor, geçmiş burada kaybettirmiş
                multiplier *= 0.7  # Ciddi indirim
                logger.debug(f"PerCoin [{symbol}]: model={final_sig}, history={hist_sig} -> penalty")
            elif hist_sig is not None and hist_sig == final_sig:
                # Modelle geçmiş aynı yönde, bu yön kazandırmış
                multiplier *= 1.15
                logger.debug(f"PerCoin [{symbol}]: model={final_sig}, history={hist_sig} -> boost")
            final_conf = final_conf * multiplier
            final_conf = min(final_conf, 99.0)
            source += f"_mem({n_sim})"

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
def _run_initial_training():
    """Senkron başlangıç eğitimi — scanner döngüsünden veya auto_train'dan çağrılır."""
    klines_list = []
    targets = (FUTURES_PAIRS or [])[:20]
    if not targets:
        targets = ["BTC_USDT","ETH_USDT","SOL_USDT","BNB_USDT","XRP_USDT"]

    # _klines_cache'den doldur, yoksa fetch etme (async gerektirmez)
    for sym in targets:
        kl = _klines_cache.get(sym)
        if kl is not None and len(kl.get("close", [])) > 50:
            klines_list.append(kl)

    if not klines_list:
        logger.warning("Başlangıç eğitimi: _klines_cache boş, scanner verisi bekleniyor")
        return False

    # ML eğitimi
    logger.info(f"ML eğitimi başlıyor ({len(klines_list)} pair/{sum(len(k['close']) for k in klines_list)} mum)...")
    ml_result = ml_engine.train_on_multi_pair(klines_list, "GLOBAL")
    metrics_tracker.record_train(ml_result)
    if ml_result.get("success"):
        logger.info(f"✅ ML eğitimi: wf={ml_result.get('wf_accuracy')}% | n={ml_result.get('n_samples')}")
        try:
            ml_engine.save(ML_MODEL_FILE)
        except Exception as e:
            logger.error(f"ML model kayıt hatası: {e}")

        # RL ön-eğitimi
        _rl_pretrain(klines_list)
        # Multi-pair backtest
        ml_engine.run_backtest_multi(klines_list)
        return True
    else:
        logger.warning(f"⚠️ ML eğitimi başarısız: {ml_result}")
        return False

async def _initial_train_waiter():
    """Scanner'ın ilk taramasını bekle, sonra _run_initial_training ile eğit."""
    try:
        # Scanner'ın ilk turu bitsin (max 3 dk, daha hızlı)
        for i in range(36):
            if ml_engine._trained:
                return
            # Daha erken başla: 100+ pair yeterli klines varsa başlat
            cache_ready = sum(1 for k in _klines_cache.values()
                             if len(k.get("close", [])) > 50)
            if cache_ready >= 100:
                logger.info(f"🤖 {cache_ready} pair için klines hazır, erken eğitim başlıyor...")
                break
            await asyncio.sleep(5)  # Daha sık kontrol (5sn)
        if ml_engine._trained:
            return
        logger.info(f"🤖 İlk eğitim: {len(scanner_cache)} pair tarandı, eğitim başlıyor...")
        await asyncio.to_thread(_run_initial_training)
    except Exception as e:
        logger.error(f"_initial_train_waiter hatası: {e}")

async def auto_train_on_startup():
    """Başlangıç eğitimi: ML ve RL."""
    try:
        await asyncio.sleep(90)
        
        # Scanner'dan klines topla
        klines_list = []
        targets = list(_klines_cache.keys())[:20] or \
                  ["BTC_USDT","ETH_USDT","SOL_USDT","BNB_USDT","XRP_USDT"]
        for sym in targets:
            kl = _klines_cache.get(sym)
            if kl is not None and len(kl.get("close", [])) > 50:
                klines_list.append(kl)

        # ML eğitimi (sadece eğitilmemişse)
        if not ml_engine._trained and klines_list:
            logger.info("🤖 auto_train_on_startup: ML eğitimi başlıyor...")
            ml_result = await asyncio.to_thread(
                ml_engine.train_on_multi_pair, klines_list, "GLOBAL")
            metrics_tracker.record_train(ml_result)
            if ml_result.get("success"):
                logger.info(f"✅ ML eğitimi: wf={ml_result.get('wf_accuracy')}%")
                await asyncio.to_thread(ml_engine.save, ML_MODEL_FILE)
                await asyncio.to_thread(ml_engine.run_backtest_multi, klines_list)
            else:
                logger.warning(f"⚠️ ML eğitimi başarısız: {ml_result}")

        # RL eğitimi (her zaman dene, trained=false ise)
        if not rl_agent._is_trained and klines_list:
            logger.info("🤖 auto_train_on_startup: RL eğitimi başlıyor...")
            await asyncio.to_thread(_rl_pretrain, klines_list)
        elif not klines_list:
            logger.warning("auto_train_on_startup: cache boş, eğitim atlandı")
    except Exception as e:
        logger.error(f"auto_train_on_startup hatası: {e}")
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
    """Klines'dan (T, 64) feature matrisi oluştur (V2)."""
    try:
        c  = np.asarray(klines["close"],  dtype=np.float64)
        h  = np.asarray(klines["high"],   dtype=np.float64)
        lo = np.asarray(klines["low"],    dtype=np.float64)
        v  = np.asarray(klines["volume"], dtype=np.float64)
        MIN = 40
        if len(c) < MIN + 5: return None

        rows = []
        fb_v2 = getattr(ml_engine, "_fb_v2", None)
        builder = fb_v2.build if fb_v2 else FeatureBuilder.build
        for i in range(MIN, len(c)):
            sub  = {"close": c[:i], "high": h[:i], "low": lo[:i], "volume": v[:i]}
            feat = builder(sub)
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
                client = await get_http_client()
                if client is None:
                    logger.warning("HTTPX yuklu degil, retrain pasif")
                    continue
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

            # RL bağımsız ön-eğitim (henüz eğitilmemişse)
            if not rl_agent._is_trained:
                rl_klines = list(_klines_cache.values())[:5]
                if rl_klines or FUTURES_PAIRS:
                    if not rl_klines:
                        client = await get_http_client()
                        if client:
                            for sym in FUTURES_PAIRS[:3]:
                                kl = await fetch_klines(client, sym, limit=200)
                                if kl: rl_klines.append(kl)
                    if rl_klines:
                        logger.info("🤖 RL ön-eğitimi (retrain loop)")
                        await asyncio.to_thread(_rl_pretrain, rl_klines)

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
            client = await get_http_client()
            if client is not None:
                await client.get(f"{app_url}/health", timeout=10)
        except Exception:
            pass
        await asyncio.sleep(840)

def _log_config_status():
    """Başlangıçta konfigürasyon durumunu logla."""
    logger.info("═══════════════════════════════════════")
    logger.info("  MEXC RL+ML Trader v3.6 — Konfigürasyon")
    logger.info("═══════════════════════════════════════")
    # API
    logger.info(f"  MEXC API   : {config.api.mexc_base_url}")
    logger.info(f"  Interval   : {config.api.kline_interval}")
    logger.info(f"  Limit      : {config.api.kline_limit}")
    # Trading
    logger.info(f"  Max Pozisyon: {config.trading.max_open_positions}")
    logger.info(f"  Min Hacim  : ${config.trading.min_volume_24h:,.0f}")
    logger.info(f"  Fee        : {config.trading.fee_rate*100:.3f}%")
    # ML
    feat_ver = "V3" if config.ml.use_v3_features else ("V2" if config.ml.use_v2_features else "V1")
    logger.info(f"  ML Features: {config.ml.n_features} ({feat_ver})")
    logger.info(f"  Lookahead  : {config.ml.lookahead} bar")
    logger.info(f"  Conf eşik  : >{config.ml.conf_threshold_predict}%")
    # RL
    logger.info(f"  RL States  : {config.rl.state_dim}")
    logger.info(f"  RL Actions : {config.rl.n_actions}")
    logger.info(f"  RL Pretrain: {config.rl.rl_pretrain_iterations} iterasyon")
    # Risk
    logger.info(f"  Max Drawdown: {config.risk.max_drawdown*100:.0f}%")
    logger.info(f"  Kaldıraç   : max {config.risk.max_leverage}x (WF<65% → 5x kilit)")
    # Telegram
    has_tg = bool(config.telegram_bot_token and config.telegram_chat_id)
    logger.info(f"  Telegram   : {'✅ Aktif' if has_tg else '⭕ Pasif'}")
    # Persistence
    logger.info(f"  Persist    : {config.persist_dir}")
    logger.info(f"  ML Model   : {os.path.basename(config.ml_model_file)}")
    logger.info(f"  RL Model   : {os.path.basename(config.rl_model_file)}")
    # Uyarılar
    if not config.api.mexc_base_url.startswith("https://"):
        logger.warning("⚠️ MEXC API HTTPS değil — güvenlik riski!")
    if config.scanner_interval_sec < 15:
        logger.warning("⚠️ Scanner interval çok kısa (<15sn) — MEXC rate-limit!")
    logger.info("═══════════════════════════════════════")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _startup_time
    _startup_time = time.time()
    logger.info("🚀 MEXC RL+ML Trading System v3.0 başlatılıyor...")
    _log_config_status()
    await load_data()

    # ML model yükle — ana dosya yoksa versiyon store'dan dene
    if ml_engine.load(ML_MODEL_FILE):
        logger.info("✅ ML modeli yüklendi")
    else:
        # Versiyon store'daki en iyi modeli dene
        best = ml_engine.version_store.get_best_version()
        if best and best.get("path") and ml_engine.load(best["path"]):
            logger.info(f"✅ ML modeli versiyon store'dan yüklendi (WF: {best.get('wf_acc',0)}%)")
            # Ana dosyaya kopyala
            import shutil
            try:
                shutil.copy2(best["path"], ML_MODEL_FILE)
            except Exception:
                pass
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
    asyncio.create_task(_initial_train_waiter())
    yield
    await save_data()
    await close_http_client()
    await asyncio.to_thread(ml_engine.save, ML_MODEL_FILE)
    await asyncio.to_thread(rl_agent.save, RL_MODEL_FILE)
    logger.info("Sistem kapatıldı")

# ── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI(title="MEXC RL+ML Trader v3.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# ── Auth Middleware ─────────────────────────────────────────────────────
_AUTH_TOKEN = os.getenv("AUTH_TOKEN", "")
if not _AUTH_TOKEN:
    _AUTH_TOKEN = hashlib.sha256(os.urandom(16)).hexdigest()[:12]
    logger.info("No AUTH_TOKEN set, generated random: %s", _AUTH_TOKEN)

@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    # GET requests ve public yollar herkese acik
    if request.method == "GET" or request.url.path in (
        "/", "/ml-details", "/sw.js", "/manifest.json",
    ) or request.url.path.startswith("/static"):
        return await call_next(request)
    # POST/PUT/DELETE -> token gerekli
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    if token != _AUTH_TOKEN:
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=401, content={"error": "Unauthorized"})
    return await call_next(request)

# ── Scanner ───────────────────────────────────────────────────────────────────
_startup_time: float = 0.0
_startup_scan_completed: bool = False
_scanner_fail_count: int = 0
_scanner_last_error: str = ""
_STARTUP_GRACE_SECONDS = config.startup_grace_sec

async def scanner_loop():
    global FUTURES_PAIRS, _startup_scan_completed, _scanner_fail_count, _scanner_last_error
    consecutive_failures = 0
    max_consecutive_failures = 5
    
    while True:
        try:
            client = await get_http_client()
            if client is None:
                logger.warning("HTTPX yuklu degil, scanner pasif")
                await asyncio.sleep(60); continue
                
            # Fetch pairs with retry logic
            new_pairs = []
            for retry in range(3):
                try:
                    new_pairs = await fetch_all_futures_pairs(client)
                    if new_pairs:
                        break
                    logger.warning(f"Boş pair listesi alındı, deneme {retry+1}/3")
                except Exception as e:
                    logger.error(f"Pair fetch denemesi {retry+1}/3 başarısız: {e}")
                    if retry < 2:  # Not the last attempt
                        await asyncio.sleep(2 ** retry)  # Exponential backoff
            
            if new_pairs:
                FUTURES_PAIRS = new_pairs
                consecutive_failures = 0
                _startup_scan_completed = True
                logger.info(f"MEXC: {len(new_pairs)} aktif pair")
            elif not FUTURES_PAIRS:  # Only sleep if we have no pairs at all
                await asyncio.sleep(10); continue
            
            if not FUTURES_PAIRS:
                await asyncio.sleep(10); continue
                
            # Adaptive batch size: 8->15 based on pair count
            batch_sz = min(config.scanner_batch_size * 2, max(8, len(FUTURES_PAIRS) // 5))
            sem = asyncio.Semaphore(batch_sz)
            async def bounded_process(sym):
                async with sem:
                    return await process_pair(client, sym)
            tasks = [bounded_process(sym) for sym in FUTURES_PAIRS]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for sym, res in zip(FUTURES_PAIRS, results):
                if isinstance(res, dict):
                    scanner_cache[sym] = res
                elif isinstance(res, Exception):
                    logger.debug(f"Scanner: {sym} hata: {res}")
            logger.info(f"Tarama tamamlandı — {len(scanner_cache)}/{len(FUTURES_PAIRS)} pair")
            consecutive_failures = 0  # Reset on successful scan
            _scanner_fail_count = 0
            _scanner_last_error = ""
        except Exception as e:
            consecutive_failures += 1
            _scanner_fail_count = consecutive_failures
            _scanner_last_error = str(e)
            logger.error(f"Scanner hata: {e}")
            metrics_tracker.record_error()
            
            # Exponential backoff for consecutive failures
            if consecutive_failures >= max_consecutive_failures:
                wait_time = min(300, 2 ** min(consecutive_failures, 8))  # Cap at 5 minutes
                logger.warning(f"Çok fazla ardışık hata, {wait_time}s bekleme")
                # HTTP client'ı sıfırla — belki connection pool sorunu vardır
                try:
                    await close_http_client()
                    logger.info("HTTP client sıfırlandı")
                except Exception:
                    pass
                await asyncio.sleep(wait_time)
            else:
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

    # Dinamik SL/TP — ATR bazlı
    lev = max(1, prediction.get("leverage", 5))
    wf_acc_now = ml_engine._wf.get("accuracy", 0)
    if wf_acc_now < 65 and lev > 5:
        lev = 5

    # ATR bazlı dinamik SL/TP
    if klines is not None and len(klines.get("close", [])) > 20:
        sl_tp_result = dynamic_sltp.calculate_dynamic_sl_tp(
            klines, price, prediction["signal"], leverage=lev
        )
        sl = sl_tp_result["sl_price"]
        tp = sl_tp_result["tp_price"]
        sl_pct = sl_tp_result["sl_pct"]
        tp_pct = sl_tp_result["tp_pct"]
    else:
        # Fallback: sabit SL/TP
        sl_tp = config.trading.sl_tp_levels.get(lev) or \
                config.trading.sl_tp_levels.get(5) or (0.018, 0.054)
        sl_pct, tp_pct = sl_tp
        if prediction["signal"] == "SHORT":
            sl = round(price * (1 + sl_pct), 10)
            tp = round(price * (1 - tp_pct), 10)
        else:
            sl = round(price * (1 - sl_pct), 10)
            tp = round(price * (1 + tp_pct), 10)

    # ── GİRİŞ KONTROLÜ ───────────────────────────────────────────────────────
    gbm_sig  = prediction.get("gbm_signal", "WAIT")
    rf_sig   = prediction.get("rf_signal",  "WAIT")
    rl_sig   = prediction.get("rl_signal",  "WAIT")
    main_sig = prediction["signal"]
    conf_val = prediction.get("confidence", 0)
    source   = prediction.get("hybrid_source", "ML_only")

    # Multi-Timeframe Onay
    mtf_boost = 0.0
    if main_sig in ("LONG", "SHORT") and klines is not None:
        try:
            # 1h ve 4h klines icin cache'den al veya basitce 15m'yi kullan
            klines_1h = _klines_cache.get(symbol, klines)  # Fallback: 15m
            klines_4h = _klines_cache.get(symbol, klines)  # Fallback: 15m
            mtf_result = mtf_confirmation.confirm_signal(
                main_sig, klines, klines_1h, klines_4h
            )
            mtf_boost = mtf_result.get("confidence_boost", 0)
            if not mtf_result["confirmed"]:
                conf_val *= 0.7  # Onaysiz sinyal → guven dususu
        except Exception as e:
            logger.debug(f"MTF onay hatası: {e}")

    # Market Data Enrichment — Funding, Fear&Greed, Cross-Asset
    market_modifier = 0.0
    try:
        market_data = await market_data_enrichment.get_all_market_data(symbol)
        market_modifier = market_data.get("total_signal_modifier", 0)
        # Funding rateSHORT lehineyse ve sinyal LONG'sa dikkatli ol
        funding_signal = market_data.get("funding", {}).get("signal", "neutral")
        if funding_signal == "short_bias" and main_sig == "LONG":
            conf_val *= 0.85  # Funding SHORT lehine → LONG guveni dus
        elif funding_signal == "long_bias" and main_sig == "SHORT":
            conf_val *= 0.85  # Funding LONG lehine → SHORT guveni dus
    except Exception as e:
        logger.debug(f"Market data enrichment hatası: {e}")

    # Sinyal uyumu: RL+ML hemfikir veya ML tek başına yeterli güvenli
    # FIX: Daha katı giriş — GBM ve RF MUTLAKA hemfikir olmalı
    gbm_rf_agree = (gbm_sig == rf_sig == main_sig)
    signals_agree = (
        (source == "RL+ML" and gbm_rf_agree) or
        (source == "ML_only" and conf_val >= config.trading.min_confidence_ml_only and gbm_rf_agree) or
        (source == "RL_only" and conf_val >= config.trading.min_confidence_rl_only) or
        (gbm_rf_agree and conf_val >= config.trading.min_confidence_gbm_rf)
    )

    # Guven guncellemesi (MTF boost + Market data modifier)
    conf_val = min(99.0, conf_val + mtf_boost * 100 + market_modifier * 50)

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
        if (datetime.now(timezone.utc) - st["last_exit_time"]).total_seconds() < cooldown_sec:
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
        entry_time_raw = datetime.now(timezone.utc)
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
        await save_data()

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
            st["rl_state"]   = None; await save_data()
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

            # Trailing Stop — Kar kilitleme
            if not close_pos and klines is not None:
                try:
                    atr_val = dynamic_sltp.calculate_atr(klines)
                    if atr_val > 0:
                        # En yuksek/dusuk fiyat guncelle
                        if "highest_since_entry" not in pos:
                            pos["highest_since_entry"] = price
                        if "lowest_since_entry" not in pos:
                            pos["lowest_since_entry"] = price

                        if side == "LONG":
                            pos["highest_since_entry"] = max(pos["highest_since_entry"], price)
                            profit_pct = (price - entry_p) / entry_p
                            if profit_pct > 0.025:  # %2.5 kardan sonra trailing baslar
                                trailing_stop = dynamic_sltp.calculate_trailing_stop(
                                    entry_p, price, pos["highest_since_entry"],
                                    side, atr_val, trailing_pct=0.5
                                )
                                if trailing_stop > pos["sl"]:
                                    pos["sl"] = trailing_stop
                                    logger.info(f"[TRAILING] {symbol} SL guncellendi: {trailing_stop:.6f}")
                        else:
                            pos["lowest_since_entry"] = min(pos["lowest_since_entry"], price)
                            profit_pct = (entry_p - price) / entry_p
                            if profit_pct > 0.025:
                                trailing_stop = dynamic_sltp.calculate_trailing_stop(
                                    entry_p, price, pos["lowest_since_entry"],
                                    side, atr_val, trailing_pct=0.5
                                )
                                if trailing_stop < pos["sl"]:
                                    pos["sl"] = trailing_stop
                                    logger.info(f"[TRAILING] {symbol} SL guncellendi: {trailing_stop:.6f}")
                except Exception as e:
                    logger.debug(f"Trailing stop hatası: {e}")

            pos_age_h = (datetime.now(timezone.utc) - pos["entry_time_raw"]).total_seconds() / 3600
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

        exit_time_raw = datetime.now(timezone.utc)
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
        await save_data()

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
        "timestamp":  datetime.now(timezone.utc).isoformat(),
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
                "timestamp":  datetime.now(timezone.utc).isoformat(),
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
            "total":len(scanner_cache),"timestamp":datetime.now(timezone.utc).isoformat()}

@app.get("/api/stats")
async def get_stats():
    if not scanner_cache: return {"success":True,"data":{}}
    vals = list(scanner_cache.values())
    total_pnl    = sum(agent_states.get(v["symbol"],{}).get("pnl",0) for v in vals)
    total_trades = sum(agent_states.get(v["symbol"],{}).get("trades",0) for v in vals)
    total_wins   = sum(agent_states.get(v["symbol"],{}).get("wins",0) for v in vals)
    al = sum(1 for v in vals if v.get("active_pos") and v["active_pos"]["side"]=="LONG")
    active_shorts = sum(1 for v in vals if v.get("active_pos") and v["active_pos"]["side"]=="SHORT")
    cl = sum(1 for h in trade_history if h["side"]=="LONG")
    cs = sum(1 for h in trade_history if h["side"]=="SHORT")
    return {"success":True,"data":{
        "total_pairs":len(vals), "total_pnl":round(total_pnl,2),
        "total_trades":total_trades,
        "win_rate":round(total_wins/total_trades*100,1) if total_trades else 0,
        "active_longs":al, "active_shorts":active_shorts,
        "closed_longs":cl, "closed_shorts":cs,
        "portfolio":portfolio,
        "model_accuracy":round(ml_engine.get_accuracy(),1),
        "total_closed_notional":round(portfolio.get("total_closed_notional",0),2),
        "total_closed_trades":portfolio.get("total_closed_trades",0),
        "timestamp":datetime.now(timezone.utc).isoformat()
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
    try:
        klines_list = list(_klines_cache.values())[:10]
        if not klines_list: return {"success":False,"error":"Cache boş"}
        result = await asyncio.to_thread(_rl_pretrain, klines_list)
        return {"success":True,"data":rl_agent.get_info()}
    except Exception as e:
        logger.error(f"RL train hatası: {e}")
        return {"success":False,"error":str(e)}

@app.post("/api/train/{symbol}")
async def trigger_training(symbol: str):
    try:
        sym = symbol.upper().replace("-","_")
        client = await get_http_client()
        if client is None:
            return {"success":False,"error":"HTTPX istemcisi kullanılamıyor"}
        klines = await fetch_klines(client, sym, limit=300)
        if not klines: return {"success":False,"error":"MEXC kline verisi alınamadı"}
        result = await asyncio.to_thread(ml_engine.train, klines, sym, True)
        if result.get("success"):
            await asyncio.to_thread(ml_engine.save, ML_MODEL_FILE)
            ml_engine._retrain_triggers["manual"] += 1
        return {"success":True,"data":result}
    except Exception as e:
        logger.error(f"Train {symbol} hatası: {e}")
        return {"success":False,"error":str(e)}

@app.post("/api/train_all")
async def train_all():
    try:
        klines_list = []
        client = await get_http_client()
        if client is None:
            return {"success":False,"error":"HTTPX istemcisi kullanılamıyor"}
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
    except Exception as e:
        logger.error(f"Train all hatası: {e}")
        return {"success":False,"error":str(e)}

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
            trade_list = []
            for t in bt2.trades:
                trade_list.append({
                    "side": t.side, "entry": round(float(t.entry_price), 6),
                    "exit": round(float(t.exit_price), 6),
                    "pnl": round(float(t.return_pct) * 100, 4),
                    "bars": t.duration_bars,
                })
            result["advanced"] = {
                "roi_pct":       bt2.roi_pct,
                "win_rate":      bt2.win_rate / 100.0,
                "max_drawdown":  bt2.max_drawdown_pct / 100.0,
                "sharpe":        bt2.sharpe,
                "sortino":       bt2.sortino,
                "calmar":        bt2.calmar,
                "profit_factor": bt2.profit_factor,
                "avg_win_pct":   bt2.avg_win_pct,
                "avg_loss_pct":  bt2.avg_loss_pct,
                "expectancy":    bt2.expectancy,
                "n_trades":      bt2.n_trades,
                "timing_ms":     bt2.timing_ms,
                "trades_array":  trade_list,
                "equity_curve":  bt2.equity_curve,
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
    """Detayli backtest (tek pair) — gerçek kline verisiyle.
    Query params: ?symbol=BTC_USDT&sl_pct=0.02&tp_pct=0.05&leverage=5
    """
    if not symbol:
        return {"success": False, "error": "symbol parametresi gerekli"}

    # Fetch klines from cache or API
    klines = _klines_cache.get(symbol)
    if klines is None or len(klines.get("close", [])) < 10:
        try:
            client = await get_http_client()
            klines = await fetch_klines(client, symbol, limit=200)
            if klines is not None and len(klines.get("close", [])) >= 10:
                _klines_cache[symbol] = klines
        except Exception as e:
            logger.warning(f"Backtest: {symbol} veri alinamadi: {e}")
            return {"success": False, "error": f"Veri alinamadi: {e}"}

    if klines is None or len(klines.get("close", [])) < 10:
        return {"success": False, "error": f"{symbol}: yetersiz veri"}

    try:
        prices = np.asarray(klines["close"], dtype=np.float64)
    except (IndexError, TypeError, ValueError) as e:
        return {"success": False, "error": f"Fiyat verisi islenemedi: {e}"}

    if len(prices) < 10:
        return {"success": False, "error": f"{symbol}: gecerli fiyat yok"}

    # Generate signals: use scanner signal, distribute across series
    signals = []
    scan_sig = scanner_cache.get(symbol, {}).get("signal", "").upper() if symbol in scanner_cache else "WAIT"
    if scan_sig in ("LONG", "SHORT"):
        n = len(prices)
        for i in range(0, n, max(1, n // 10)):
            signals.append((i, scan_sig))

    if len(signals) < 2:
        return {"success": False, "error": f"{symbol}: sinyal yok (mevcut: {scan_sig})"}

    fee_cfg = FeeConfig(maker_bps=config.backtest.maker_bps,
                        taker_bps=config.backtest.taker_bps)
    slip_cfg = SlippageConfig(spread_bps=config.backtest.spread_bps,
                              vol_slip_bps=config.backtest.vol_slip_bps)

    bt = backtest_simple(prices, signals, sl_pct, tp_pct, leverage,
                        fee_cfg, slip_cfg, symbol=symbol,
                        max_bars_hold=max_bars_hold)

    trade_list = []
    for t in bt.trades:
        trade_list.append({
            "side": t.side,
            "entry": round(float(t.entry_price), 6),
            "exit": round(float(t.exit_price), 6),
            "pnl": round(float(t.return_pct) * 100, 4),
            "bars": t.duration_bars,
            "sl_hit": t.sl_hit,
            "tp_hit": t.tp_hit,
        })

    return {"success": True, "data": {
        "symbol":        bt.symbol,
        "n_trades":      bt.n_trades,
        "roi_pct":       bt.roi_pct,
        "win_rate":      bt.win_rate,
        "max_drawdown":  bt.max_drawdown_pct,
        "sharpe":        bt.sharpe,
        "sortino":       bt.sortino,
        "calmar":        bt.calmar,
        "profit_factor": bt.profit_factor,
        "avg_win_pct":   bt.avg_win_pct,
        "avg_loss_pct":  bt.avg_loss_pct,
        "expectancy":    bt.expectancy,
        "total_fee_pct": bt.total_fee_pct,
        "timing_ms":     bt.timing_ms,
        "equity_curve":  bt.equity_curve,
        "trades":        trade_list,
        "n_prices":      len(prices),
        "params": {
            "sl_pct": sl_pct, "tp_pct": tp_pct,
            "leverage": leverage, "max_bars_hold": max_bars_hold,
            "spread_bps": slip_cfg.spread_bps,
            "taker_bps": fee_cfg.taker_bps,
            "maker_bps": fee_cfg.maker_bps,
        },
    }}

@app.get("/api/backtest/monte_carlo")
async def run_monte_carlo(n_simulations: int = 1000):
    """Monte Carlo simülasyonu calistir."""
    bt = ml_engine._bt
    trades = bt.get("trades", []) if isinstance(bt, dict) else []

    if not trades:
        # Scanner'dan trade listesi olustur
        for sym, data in list(scanner_cache.items())[:20]:
            sig = data.get("signal", "").upper()
            if sig in ("LONG", "SHORT"):
                pnl = data.get("pnl", 0)
                trades.append({"pnl": pnl})

    if not trades:
        return {"success": False, "error": "Trade verisi yok"}

    mc = MonteCarloSimulator(n_simulations=n_simulations)
    result = mc.simulate(trades)

    return {"success": True, "data": result}

@app.get("/api/backtest/walk_forward")
async def run_walk_forward():
    """Walk-forward optimizasyon sonuclari."""
    wf_result = ml_engine._wf
    if not wf_result:
        return {"success": False, "error": "WF sonucu yok"}

    return {"success": True, "data": {
        "accuracy": wf_result.get("accuracy", 0),
        "f1": wf_result.get("f1", 0),
        "n_samples": wf_result.get("n_samples", 0),
        "n_folds": wf_result.get("n_folds", 0),
        "fold_accuracies": wf_result.get("fold_accuracies", []),
    }}

@app.get("/api/paper_trading")
async def get_paper_trading_status():
    """Paper trading durumu."""
    active_positions = sum(1 for st in agent_states.values() if st.get("active_pos"))
    total_trades = sum(st.get("trades", 0) for st in agent_states.values())
    total_wins = sum(st.get("wins", 0) for st in agent_states.values())
    total_pnl = sum(st.get("pnl", 0) for st in agent_states.values())

    win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0

    return {"success": True, "data": {
        "status": "ACTIVE",
        "mode": "PAPER_TRADING",
        "active_positions": active_positions,
        "total_trades": total_trades,
        "total_wins": total_wins,
        "win_rate": round(win_rate, 1),
        "total_pnl": round(total_pnl, 2),
        "portfolio_capital": portfolio.get("capital", 0),
        "initial_capital": portfolio.get("initial_capital", 0),
        "note": "Gerçek emir acılmıyor — sadece sinyal uretimi",
    }}

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

@app.get("/api/status")
async def get_system_status():
    """Kapsamlı sistem durumu — tüm servislerin anlık durumu."""
    uptime_sec = time.time() - _startup_time if _startup_time > 0 else 0
    ws_connected = len(active_connections)
    mexc_connected = len(scanner_cache) > 0
    last_scan_str = ""
    if scanner_cache:
        last_timestamps = [d.get("timestamp","") for d in scanner_cache.values() if d.get("timestamp")]
        last_scan_str = max(last_timestamps) if last_timestamps else ""
    scanner_degraded = _scanner_fail_count >= 5
    return {"success": True, "data": {
        "server": {
            "uptime_sec": round(uptime_sec, 1),
            "uptime_hours": round(uptime_sec / 3600, 1),
            "started_at": datetime.fromtimestamp(_startup_time, tz=timezone.utc).isoformat() if _startup_time > 0 else None,
            "healthy": True,
            "degraded_mode": scanner_degraded,
            "scanner_fail_count": _scanner_fail_count,
            "scanner_last_error": _scanner_last_error,
        },
        "mexc_api": {
            "connected": mexc_connected,
            "degraded": scanner_degraded,
            "total_pairs": len(scanner_cache),
            "real_data_count": sum(1 for d in scanner_cache.values() if d.get("data_source") == "real"),
            "last_scan": last_scan_str,
            "scanner_active": True,
        },
        "ml_engine": {
            "trained": ml_engine._trained,
            "wf_accuracy": ml_engine._wf.get("accuracy", 0),
            "training_in_progress": ml_engine._training_in_progress,
            "prediction_count": ml_engine._pred_count,
            "feedback_buffer": ml_engine.feedback.size(),
            "per_coin_memory": ml_engine.per_coin_mem.total_records(),
            "per_coin_stats": ml_engine.per_coin_mem.get_stats(),
        },
        "rl_agent": {
            "trained": rl_agent._is_trained,
            "total_steps": rl_agent._total_steps,
            "avg_reward_20": round(float(np.mean(rl_agent._episode_rewards[-20:]))
                                   if rl_agent._episode_rewards else 0, 4),
            "experience_buffer": rl_experience.size(),
            "ewc_active": rl_agent._ewc_fisher is not None,
        },
        "portfolio": {
            "capital": round(portfolio.get("capital", 0), 2),
            "initial_capital": round(portfolio.get("initial_capital", 0), 2),
            "total_pnl": round(portfolio.get("capital", 0) - portfolio.get("initial_capital", 0), 2),
            "total_trades": portfolio.get("total_closed_trades", 0),
            "total_notional": round(portfolio.get("total_closed_notional", 0), 2),
        },
        "websocket": {
            "active_connections": ws_connected,
            "broadcast_active": True,
        },
        "risk": risk_mgr.get_status(),
        "drift": ml_engine.drift.get_status(),
        "cache": feature_cache.stats,
        "monitor": metrics_tracker.get_status(),
    }}

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

@app.get("/api/per_coin_memory")
async def get_per_coin_memory(symbol: str = "", top: int = 50):
    """Coine ozel feature hafizasi istatistikleri."""
    try:
        if symbol:
            data = ml_engine.per_coin_mem.get_stats(symbol)
        else:
            all_stats = ml_engine.per_coin_mem.get_stats()
            sorted_coins = sorted(all_stats.items(), key=lambda x: x[1]["trades"], reverse=True)
            data = {s: st for s, st in sorted_coins[:top]}
        return {"success": True, "data": data, "total": ml_engine.per_coin_mem.total_records()}
    except Exception as e:
        return {"success": False, "error": str(e)}


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
        "timestamp":    datetime.now(timezone.utc).isoformat(),
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
                "portfolio":portfolio,"timestamp":datetime.now(timezone.utc).isoformat()}))
        while True: await websocket.receive_text()
    except (WebSocketDisconnect, Exception):
        if websocket in active_connections: active_connections.remove(websocket)

# ── Telegram Webhook ─────────────────────────────────────────────────────────
@app.get("/api/health")
async def health_check():
    """Render health check — server yanit veriyorsa saglikli."""
    return {
        "status": "healthy",
        "uptime_sec": round(time.time() - _startup_time, 1) if _startup_time > 0 else 0,
        "pairs_cached": len(scanner_cache),
        "trained": ml_engine._trained,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

@app.get("/api/pnl_timeline")
async def get_pnl_timeline():
    return {"success": True, "data": _pnl_timeline[-200:]}

@app.post("/api/telegram_webhook")
async def telegram_webhook(payload: dict):
    try:
        msg = payload.get("message", {}).get("text", "").strip().lower()
        chat_id = payload.get("message", {}).get("chat", {}).get("id")
        if not msg or not chat_id:
            return {"success": False}
        reply = "Bilinmeyen komut. Kullanilabilir: /status, /capital, /train, /rl"
        if msg == "/status":
            uptime = time.time() - _startup_time
            pairs = len(scanner_cache)
            active_pos = sum(1 for s in agent_states.values() if s.get("active_pos"))
            reply = (f"MEXC ML Trader v3.0\nCalisma Suresi: {uptime/3600:.1f}s\n"
                     f"Pair: {pairs}\nAktif Pozisyon: {active_pos}\n"
                     f"ML Egitildi: {'Evet' if ml_engine._trained else 'Hayir'}\n"
                     f"RL Egitildi: {'Evet' if rl_agent._is_trained else 'Hayir'}")
        elif msg == "/capital":
            reply = f"Kasa: ${portfolio.get('capital',0):.2f}\nToplam PnL: ${portfolio.get('capital',0)-portfolio.get('initial_capital',0):.2f}"
        elif msg == "/train":
            asyncio.create_task(_run_training())
            reply = "ML egitimi baslatildi."
        elif msg == "/rl":
            asyncio.create_task(_rl_pretrain(list(_klines_cache.values())[:5]))
            reply = "RL egitimi baslatildi."
        if config.telegram_bot_token and config.telegram_chat_id:
            await send_telegram_message(reply)
    except Exception as e:
        logger.warning(f"Telegram webhook error: {e}")
    return {"success": True}

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
