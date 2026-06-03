import os

class Config:
    PORT = int(os.getenv("PORT", 8000))
    PERSIST_DIR = os.getenv("PERSIST_DIR", os.path.join(os.path.dirname(os.path.dirname(__file__)), "data"))

    MEXC_BASE_URL = os.getenv("MEXC_BASE_URL", "https://contract.mexc.com/api/v1/contract")

    # Etherscan API key (ücretsiz)
    ETHERSCAN_KEY = os.getenv("ETHERSCAN_KEY", "J75REUKRVKVZ5IQFR6TVDENE5MDN8P86PU")

    # Telegram
    TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TG_CHAT = os.getenv("TELEGRAM_CHAT_ID", "")

    # Tarama aralıkları (saniye)
    SCAN_INTERVAL = int(os.getenv("SCAN_INTERVAL", 120))
    NEWS_INTERVAL = int(os.getenv("NEWS_INTERVAL", 300))
    TRADING_INTERVAL = int(os.getenv("TRADING_INTERVAL", 60))

    # Ticaret
    MAX_POSITIONS = int(os.getenv("MAX_POSITIONS", 3))
    BASE_SIZE_USDT = float(os.getenv("BASE_SIZE_USDT", 10))
    MAX_LEVERAGE = int(os.getenv("MAX_LEVERAGE", 5))
    SL_PCT = float(os.getenv("SL_PCT", 0.02))
    TP_PCT = float(os.getenv("TP_PCT", 0.04))
    MIN_SIGNAL_SCORE = float(os.getenv("MIN_SIGNAL_SCORE", 2.0))

    # ========================================================
    # OTOMATİK BALİNA TESPİT AYARLARI
    # ========================================================
    # Kaç ETH üzeri transferler "balina" sayılsın?
    WHALE_THRESHOLD_ETH = float(os.getenv("WHALE_THRESHOLD_ETH", 500))
    # Kaç blok geriye taranacak?
    BLOCK_RANGE = int(os.getenv("BLOCK_RANGE", 5))
    # Aynı anda maksimum kaç balina cüzdan takip edilsin?
    MAX_WATCHED_WALLETS = int(os.getenv("MAX_WATCHED_WALLETS", 50))

    # TAKİP EDİLECEK SEMBOLLER
    TRACKED_SYMBOLS = ["BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE", "DOT", "AVAX", "LINK"]


config = Config()
