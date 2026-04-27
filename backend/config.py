"""
Configuration Management for MEXC Trading System
"""
import os
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # MEXC API Configuration
    mexc_api_key: str = ""
    mexc_api_secret: str = ""
    mexc_base_url: str = "https://contract.mexc.com"
    ws_url: str = "wss://contract.mexc.com/realtime"

    # Trading Settings
    min_volume_24h_usd: float = 20_000_000  # $20M minimum volume
    confidence_threshold: float = 0.75  # 75% confidence minimum
    max_leverage: int = 20
    default_leverage: int = 10
    max_position_size_usd: float = 1000
    stop_loss_pct: float = 0.02  # 2%
    take_profit_pct: float = 0.04  # 4%

    # ML Settings
    model_cache_dir: str = "./data/models_cache"
    feature_count: int = 48
    training_timeframe: str = "4h"
    prediction_timeframe: str = "15m"

    # Server Settings
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False

    # Telegram Notifications
    telegram_bot_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
