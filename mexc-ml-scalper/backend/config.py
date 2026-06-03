import os
from dataclasses import dataclass, field
from typing import Dict, Tuple


def _env(key: str, default):
    val = os.getenv(key)
    if val is None: return default
    if isinstance(default, bool): return val.lower() in ("1", "true", "yes")
    if isinstance(default, int): return int(val)
    if isinstance(default, float): return float(val)
    return val


@dataclass
class Config:
    port: int = _env("PORT", 8000)
    persist_dir: str = _env("PERSIST_DIR", os.path.join(os.path.dirname(os.path.dirname(__file__)), "data"))
    telegram_bot_token: str = _env("TELEGRAM_BOT_TOKEN", "")
    telegram_chat_id: str = _env("TELEGRAM_CHAT_ID", "")

    # MEXC API
    mexc_base_url: str = _env("MEXC_BASE_URL", "https://contract.mexc.com/api/v1/contract")
    kline_interval: str = _env("KLINE_INTERVAL", "Min5")
    kline_limit: int = _env("KLINE_LIMIT", 100)

    # Scanner
    scanner_interval_sec: int = _env("SCANNER_INTERVAL_SEC", 60)
    scanner_batch_size: int = _env("SCANNER_BATCH_SIZE", 5)
    startup_grace_sec: int = _env("STARTUP_GRACE_SEC", 60)

    # Trading
    max_open_positions: int = _env("MAX_OPEN_POSITIONS", 5)
    min_volume_24h: float = _env("MIN_VOLUME_24H", 50_000_000.0)
    min_price: float = _env("MIN_PRICE", 0.000001)
    base_position_size: float = _env("BASE_POSITION_SIZE", 10.0)
    fee_rate: float = _env("FEE_RATE", 0.0006)
    sl_pct: float = _env("SL_PCT", 0.005)
    tp_pct: float = _env("TP_PCT", 0.010)
    cooldown_loss_sec: int = _env("COOLDOWN_LOSS_SEC", 1800)
    cooldown_win_sec: int = _env("COOLDOWN_WIN_SEC", 120)
    max_leverage: int = _env("MAX_LEVERAGE", 10)
    min_confidence_entry: float = _env("MIN_CONFIDENCE_ENTRY", 55.0)

    # ML
    n_features: int = _env("N_FEATURES", 14)
    lookahead: int = _env("LOOKAHEAD", 2)
    label_threshold: float = _env("LABEL_THRESHOLD", 0.003)
    lgbm_n_estimators: int = _env("LGBM_N_ESTIMATORS", 200)
    lgbm_learning_rate: float = _env("LGBM_LEARNING_RATE", 0.03)
    lgbm_max_depth: int = _env("LGBM_MAX_DEPTH", 5)
    retrain_interval_sec: int = _env("RETRAIN_INTERVAL_SEC", 600)

    # RL (sadece pozisyon boyutlandırma)
    rl_state_dim: int = _env("RL_STATE_DIM", 18)
    rl_hidden_dim: int = _env("RL_HIDDEN_DIM", 32)
    rl_learning_rate: float = _env("RL_LEARNING_RATE", 3e-4)

    # Risk
    max_drawdown: float = _env("MAX_DRAWDOWN", 0.10)
    daily_loss_limit: float = _env("DAILY_LOSS_LIMIT", 0.04)
    consecutive_loss_limit: int = _env("CONSECUTIVE_LOSS_LIMIT", 3)
    kelly_fraction: float = _env("KELLY_FRACTION", 0.25)

    # Memory limits (Render free tier)
    max_klines_cache: int = _env("MAX_KLINES_CACHE", 80)
    max_scanner_cache: int = _env("MAX_SCANNER_CACHE", 80)

    @property
    def ml_model_file(self):
        return os.path.join(self.persist_dir, "ml_model.joblib")

    @property
    def rl_model_file(self):
        return os.path.join(self.persist_dir, "rl_model.joblib")

    @property
    def persistence_file(self):
        return os.path.join(self.persist_dir, "persistence.json")


config = Config()
