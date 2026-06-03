import os

class Config:
    PORT = int(os.getenv("PORT", 8000))
    PERSIST_DIR = os.getenv("PERSIST_DIR", os.path.join(os.path.dirname(os.path.dirname(__file__)), "data"))

    # MEXC
    MEXC_BASE_URL = os.getenv("MEXC_BASE_URL", "https://contract.mexc.com/api/v1/contract")

    # Etherscan / BSCScan (en az biri gerekli)
    ETHERSCAN_KEY = os.getenv("ETHERSCAN_KEY", "")
    BSCSCAN_KEY = os.getenv("BSCSCAN_KEY", "")

    # News API
    CRYPTOPANIC_KEY = os.getenv("CRYPTOPANIC_KEY", "")  # cryptopanic.com

    # Telegram bildirim
    TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TG_CHAT = os.getenv("TELEGRAM_CHAT_ID", "")

    # Tarama aralıkları (saniye)
    SCAN_INTERVAL = int(os.getenv("SCAN_INTERVAL", 300))       # 5 dk cüzdan tara
    NEWS_INTERVAL = int(os.getenv("NEWS_INTERVAL", 600))        # 10 dk haber tara
    TRADING_INTERVAL = int(os.getenv("TRADING_INTERVAL", 60))   # 1 dk pozisyon kontrol

    # Ticaret
    MAX_POSITIONS = int(os.getenv("MAX_POSITIONS", 3))
    BASE_SIZE_USDT = float(os.getenv("BASE_SIZE_USDT", 10))
    MAX_LEVERAGE = int(os.getenv("MAX_LEVERAGE", 5))
    SL_PCT = float(os.getenv("SL_PCT", 0.02))
    TP_PCT = float(os.getenv("TP_PCT", 0.04))

    # Sinyal eşikleri
    MIN_SIGNAL_SCORE = float(os.getenv("MIN_SIGNAL_SCORE", 2.0))
    MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", 60.0))

    # ========================================================
    # TAKİP EDİLECEK CÜZDANLAR
    # ========================================================
    # Bunları kendin ekleyeceksin. Örnek olarak birkaç tane koydum.
    # Gerçek balina adreslerini bulmak için: etherscan.io → accounts → label "Fake_Phishing" vs
    # Veya https://etherscan.io/accounts/label/smart-contract
    TRACKED_WALLETS = {
        "ethereum": [
            # Örnek: Binance cold wallet
            # {"label": "Binance Cold", "address": "0x...", "chain": "ethereum"},
            # {"label": "Justin Sun", "address": "0x...", "chain": "ethereum"},
        ],
        "bsc": [
            # BSC cüzdanları
        ],
    }

    # TAKİP EDİLECEK TOKEN'LAR / SEMBOLLER
    TRACKED_SYMBOLS = ["BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE", "DOT", "AVAX", "LINK"]

    # HABER ANAHTAR KELİMELERİ
    NEWS_KEYWORDS = {
        "bullish": ["etf", "adoption", "partnership", "listing", "buyback", "upgrade", "approval", "institutional"],
        "bearish": ["hack", "exploit", "ban", "regulation", "crackdown", "fraud", "investigation", "delist"],
    }


config = Config()
