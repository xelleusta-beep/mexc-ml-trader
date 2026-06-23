import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")


class Config:
    MEXC_API_KEY = os.getenv("MEXC_API_KEY", "")
    MEXC_SECRET_KEY = os.getenv("MEXC_SECRET_KEY", "")
    MEXC_BASE_URL = "https://api.mexc.com"
    MEXC_WS_FUTURES = "wss://contract.mexc.com/edge"

    MAX_POSITIONS = int(os.getenv("MAX_POSITIONS", "5"))
    MAX_RISK_PER_TRADE = float(os.getenv("MAX_RISK_PER_TRADE", "0.02"))
    MAX_LEVERAGE = int(os.getenv("MAX_LEVERAGE", "20"))
    MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", "0.70"))
    TRADE_INTERVAL = int(os.getenv("TRADE_INTERVAL_SECONDS", "300"))

    SCANNER_INTERVAL = int(os.getenv("SCANNER_INTERVAL", "300"))
    TECHNICAL_TIMEFRAMES = os.getenv("TECHNICAL_TIMEFRAMES", "5m,15m,1h,4h,1D").split(",")
    SENTIMENT_UPDATE_INTERVAL = int(os.getenv("SENTIMENT_UPDATE_INTERVAL", "3600"))

    MAX_DAILY_LOSS_PCT = float(os.getenv("MAX_DAILY_LOSS_PCT", "0.05"))
    MAX_DRAWDOWN_PCT = float(os.getenv("MAX_DRAWDOWN_PCT", "0.15"))
    CORRELATION_LIMIT = float(os.getenv("CORRELATION_LIMIT", "0.60"))

    MAKER_FEE = 0.0001
    TAKER_FEE = 0.0005

    CACHE_DIR = Path(__file__).parent.parent / "data" / "cache"
    LOG_DIR = Path(__file__).parent.parent / "data" / "logs"

    TIMEFRAME_MAP = {
        "5m": "Min5", "15m": "Min15", "30m": "Min30",
        "1h": "Min60", "4h": "Hour4", "8h": "Hour8", "1D": "Day1",
    }

    @classmethod
    def get_timeframe_value(cls, tf: str) -> str:
        return cls.TIMEFRAME_MAP.get(tf, "Day1")


config = Config()
