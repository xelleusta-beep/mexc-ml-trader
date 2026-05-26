"""
MEXC ML Trading System — Central Configuration v1.0
Tum sabitler ve parametreler buradan yonetilir.
Environment variable ile override edilebilir.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Tuple


def _env(key: str, default):
    """Environment variable'dan deger al, tipini koru."""
    val = os.getenv(key)
    if val is None:
        return default
    if isinstance(default, bool):
        return val.lower() in ("1", "true", "yes")
    if isinstance(default, int):
        return int(val)
    if isinstance(default, float):
        return float(val)
    return val


@dataclass
class APIConfig:
    mexc_base_url: str = _env("MEXC_BASE_URL", "https://contract.mexc.com/api/v1/contract")
    kline_interval: str = _env("KLINE_INTERVAL", "Min15")
    kline_limit: int = _env("KLINE_LIMIT", 300)
    http_timeout_ticker: int = _env("HTTP_TIMEOUT_TICKER", 5)
    http_timeout_kline: int = _env("HTTP_TIMEOUT_KLINE", 8)
    http_timeout_detail: int = _env("HTTP_TIMEOUT_DETAIL", 10)
    max_retries: int = _env("API_MAX_RETRIES", 3)
    retry_backoff_factor: float = _env("API_RETRY_BACKOFF_FACTOR", 2.0)


@dataclass
class TradingConfig:
    max_open_positions: int = _env("MAX_OPEN_POSITIONS", 20)
    min_volume_24h: float = _env("MIN_VOLUME_24H", 100_000_000.0)
    min_price: float = _env("MIN_PRICE", 0.000001)
    base_position_size: float = _env("BASE_POSITION_SIZE", 100.0)
    fee_rate: float = _env("FEE_RATE", 0.0006)

    # Cooldown: zararli trade sonrasi bekleme (saniye)
    cooldown_loss_sec: int = _env("COOLDOWN_LOSS_SEC", 3600)
    cooldown_win_sec: int = _env("COOLDOWN_WIN_SEC", 180)

    # SL/TP asimetrik oranlari — leverage bazinda (sl_pct, tp_pct)
    sl_tp_levels: Dict[int, Tuple[float, float]] = field(default_factory=lambda: {
        10: (0.010, 0.035),
        7:  (0.015, 0.045),
        5:  (0.018, 0.054),
    })

    # Giris thresholdlari
    min_confidence_entry: float = _env("MIN_CONFIDENCE_ENTRY", 58.0)
    min_confidence_ml_only: float = _env("MIN_CONFIDENCE_ML_ONLY", 62.0)
    min_confidence_rl_only: float = _env("MIN_CONFIDENCE_RL_ONLY", 68.0)
    min_confidence_gbm_rf: float = _env("MIN_CONFIDENCE_GBM_RF", 60.0)


@dataclass
class RiskConfig:
    # Drawdown limitleri
    max_drawdown: float = _env("MAX_DRAWDOWN", 0.15)
    daily_loss_limit: float = _env("DAILY_LOSS_LIMIT", 0.05)

    # Pozisyon buyuklugu
    kelly_fraction: float = _env("KELLY_FRACTION", 0.25)
    max_concentration_per_pair: float = _env("MAX_CONCENTRATION", 0.20)

    # Circuit breaker
    consecutive_loss_limit: int = _env("CONSECUTIVE_LOSS_LIMIT", 5)

    # Leverage
    max_leverage: int = _env("MAX_LEVERAGE", 15)
    wf_leverage_lock_threshold: float = _env("WF_LEVERAGE_LOCK", 65.0)

    # Pozisyon yasi (mum)
    max_position_age_bars: int = _env("MAX_POSITION_AGE_BARS", 32)


@dataclass
class MLConfig:
    n_features: int = _env("N_FEATURES", 36)
    lookahead: int = _env("LOOKAHEAD", 3)
    label_threshold: float = _env("LABEL_THRESHOLD", 0.004)

    # LightGBM
    lgbm_n_estimators: int = _env("LGBM_N_ESTIMATORS", 400)
    lgbm_learning_rate: float = _env("LGBM_LEARNING_RATE", 0.025)
    lgbm_max_depth: int = _env("LGBM_MAX_DEPTH", 6)

    # RF
    rf_n_estimators: int = _env("RF_N_ESTIMATORS", 150)
    rf_max_depth: int = _env("RF_MAX_DEPTH", 8)

    # Ensemble
    gbm_weight: float = _env("GBM_WEIGHT", 0.6)
    rf_weight: float = _env("RF_WEIGHT", 0.4)

    # Egitim
    min_samples: int = _env("MIN_SAMPLES", 80)
    wf_n_splits: int = _env("WF_N_SPLITS", 5)
    balance_target_ratio: float = _env("BALANCE_TARGET_RATIO", 0.30)
    balance_target_ratio_multi: float = _env("BALANCE_TARGET_RATIO_MULTI", 0.38)

    # Retrain
    feedback_threshold: int = _env("FEEDBACK_THRESHOLD", 100)
    drift_cooldown_min: int = _env("DRIFT_COOLDOWN_MIN", 360)
    temporal_retrain_hours: int = _env("TEMPORAL_RETRAIN_HOURS", 999)

    # Feature Engineering v2
    use_v2_features: bool = _env("USE_V2_FEATURES", True)
    feature_auto_select: bool = _env("FEATURE_AUTO_SELECT", True)
    feature_max_v2: int = _env("FEATURE_MAX_V2", 64)
    feature_corr_threshold: float = _env("FEATURE_CORR_THRESHOLD", 0.85)

    # Confidence
    conf_threshold_predict: float = _env("CONF_THRESHOLD_PREDICT", 52.0)
    conf_high_leverage_1: float = _env("CONF_HIGH_LEV_1", 82.0)
    conf_high_leverage_2: float = _env("CONF_HIGH_LEV_2", 75.0)
    lev_high: int = _env("LEV_HIGH", 15)
    lev_medium: int = _env("LEV_MEDIUM", 10)
    lev_low: int = _env("LEV_LOW", 5)

    # Drift
    drift_window: int = _env("DRIFT_WINDOW", 50)
    drift_threshold_sigma: float = _env("DRIFT_THRESHOLD_SIGMA", 2.5)
    drift_absolute_drop: float = _env("DRIFT_ABSOLUTE_DROP", 0.55)


@dataclass
class RLConfig:
    state_dim: int = _env("RL_STATE_DIM", 60)    # 56 V2 features + 4 position features
    n_actions: int = _env("RL_N_ACTIONS", 5)
    hidden_dim: int = _env("RL_HIDDEN_DIM", 64)
    learning_rate: float = _env("RL_LEARNING_RATE", 3e-4)

    # PPO hyperparams
    clip_epsilon: float = _env("RL_CLIP_EPS", 0.2)
    gamma: float = _env("RL_GAMMA", 0.99)
    lam: float = _env("RL_LAM", 0.95)
    n_epochs: int = _env("RL_N_EPOCHS", 4)
    batch_size: int = _env("RL_BATCH_SIZE", 256)
    entropy_coef: float = _env("RL_ENTROPY_COEF", 0.05)
    vf_coef: float = _env("RL_VF_COEF", 0.5)

    # Environment
    episode_length: int = _env("RL_EPISODE_LENGTH", 100)
    margin: float = _env("RL_MARGIN", 100.0)

    # Online
    online_trigger: int = _env("RL_ONLINE_TRIGGER", 20)
    online_epochs: int = _env("RL_ONLINE_EPOCHS", 2)
    ewc_lambda: float = _env("RL_EWC_LAMBDA", 0.1)

    # Training
    rl_pretrain_iterations: int = _env("RL_PRETRAIN_ITERATIONS", 100)


@dataclass
class BacktestConfig:
    sl_pct: float = _env("BT_SL_PCT", 0.022)
    tp_pct: float = _env("BT_TP_PCT", 0.05)
    leverage: int = _env("BT_LEVERAGE", 5)
    max_bars_hold: int = _env("BT_MAX_BARS_HOLD", 8)
    spread_bps: float = _env("BT_SPREAD_BPS", 0.5)
    vol_slip_bps: float = _env("BT_VOL_SLIP_BPS", 0.3)
    maker_bps: float = _env("BT_MAKER_BPS", -0.25)
    taker_bps: float = _env("BT_TAKER_BPS", 7.0)
    wf_n_splits: int = _env("BT_WF_N_SPLITS", 5)
    wf_gap_bars: int = _env("BT_WF_GAP_BARS", 5)


@dataclass
class AppConfig:
    # Port
    port: int = _env("PORT", 8000)

    # Persistence
    persist_dir: str = _env("PERSIST_DIR",
                             os.path.join(os.path.dirname(os.path.dirname(__file__)), "data"))
    ml_model_file: str = ""
    rl_model_file: str = ""

    # Telegram
    telegram_bot_token: str = _env("TELEGRAM_BOT_TOKEN", "")
    telegram_chat_id: str = _env("TELEGRAM_CHAT_ID", "")

    # Scanner
    scanner_interval_sec: int = _env("SCANNER_INTERVAL_SEC", 30)
    scanner_batch_size: int = _env("SCANNER_BATCH_SIZE", 10)
    retrain_loop_interval_sec: int = _env("RETRAIN_LOOP_INTERVAL", 300)
    startup_grace_sec: int = _env("STARTUP_GRACE_SEC", 120)

    # Sub-configs
    api: APIConfig = field(default_factory=APIConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)

    def __post_init__(self):
        _persist = self.persist_dir
        self.ml_model_file = _env("ML_MODEL_FILE", os.path.join(_persist, "ml_model.joblib"))
        self.rl_model_file = _env("RL_MODEL_FILE", os.path.join(_persist, "rl_model.joblib"))
        self.persistence_file = _env("PERSISTENCE_FILE", os.path.join(_persist, "persistence.json"))


# Global singleton
config = AppConfig()
