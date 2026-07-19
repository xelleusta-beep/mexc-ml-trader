import time
import json
import random
import httpx
from pathlib import Path
from .base_agent import BaseAgent

MEXC_BASE = "https://api.mexc.com"
CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "cache"


MOCK_PRICES = {
    "BTC": 103500, "ETH": 2450, "SOL": 145, "XRP": 0.62, "DOGE": 0.165,
    "ADA": 0.58, "AVAX": 18.5, "LINK": 12.8, "DOT": 4.2, "MATIC": 0.22,
    "UNI": 6.5, "SHIB": 0.000012, "LTC": 65, "BCH": 320, "ATOM": 4.8,
    "FIL": 3.5, "APT": 4.2, "ARB": 0.32, "OP": 0.35, "SUI": 1.85,
    "PEPE": 0.0000085, "WIF": 0.42, "FET": 0.72, "INJ": 12.5, "TIA": 2.8,
    "SEI": 0.18, "NEAR": 3.2, "RENDER": 5.8, "STX": 0.45, "IMX": 0.52,
}


class ScannerAgent(BaseAgent):
    BLACKLIST_BASE_COINS = {
        "USDC", "BUSD", "DAI", "TUSD", "USDP", "FDUSD", "PYUSD",
        "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD", "CNY",
        "OIL", "GOLD", "SILVER", "NATGAS", "COPPER", "PLATINUM",
        "DOW", "SP500", "NASDAQ", "SPX", "DAX", "FTSE", "NIKKEI",
        "XAU", "XAG", "XPT", "XPD",
        "STOCK", "ETF", "BOND", "FUND", "INDEX", "FUTURES",
        "NVDA", "TSLA", "AAPL", "AMZN", "MSFT", "META", "GOOGL", "GOOG",
        "COIN", "MSTR", "NFLX", "AMD", "INTC", "CRM", "ORCL", "IBM",
        "NIO", "XPEV", "LI", "BABA", "JD", "PDD", "BIDU",
        "V", "MA", "JPM", "GS", "MS", "WFC", "C", "BAC",
        "DIS", "SNAP", "PINS", "UBER", "LYFT", "ABNB", "DASH",
        "PYPL", "SQ", "SHOP", "SE", "MELI",
        "SPY", "QQQ", "IWM", "DIA", "VTI", "VOO", "ARKK",
        "SLV", "GLD", "USO", "UNG", "DBC",
    }

    def __init__(self):
        super().__init__("Scanner")
        self.all_symbols: list[dict] = []
        self.hot_pairs: list[dict] = []
        self.volatile_pairs: list[dict] = []
        self.volume_leaders: list[dict] = []
        self.tickers: dict[str, dict] = {}
        self._client: httpx.AsyncClient | None = None

    def _is_stock_symbol(self, symbol: str, base_coin: str) -> bool:
        base_upper = base_coin.upper()
        if base_upper in self.BLACKLIST_BASE_COINS:
            return True
        if symbol.endswith("USD") and not symbol.endswith("USDT"):
            return True
        return False

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=10, limits=httpx.Limits(max_connections=50))
        return self._client

    def _load_cached_symbols(self) -> list[dict]:
        cache_file = CACHE_DIR / "all_symbols.json"
        if cache_file.exists():
            try:
                cached = json.loads(cache_file.read_text(encoding="utf-8"))
                return [
                    s for s in cached
                    if not self._is_stock_symbol(s.get("symbol", ""), s.get("baseCoin", ""))
                ]
            except Exception:
                pass
        return []

    def _generate_tickers_from_cache(self) -> dict[str, dict]:
        symbols = self.all_symbols
        tickers = {}
        for sym_info in symbols:
            symbol = sym_info["symbol"]
            base = sym_info.get("baseCoin", "")
            base_price = MOCK_PRICES.get(base, 1.0)
            price = base_price * (1 + random.uniform(-0.05, 0.05))
            vol = random.uniform(500000, 50000000)
            change = random.uniform(-8, 8)
            high = price * (1 + abs(change) / 100 + random.uniform(0, 0.02))
            low = price * (1 - abs(change) / 100 - random.uniform(0, 0.02))
            spread = price * random.uniform(0.0001, 0.001)

            tickers[symbol] = {
                "symbol": symbol,
                "lastPrice": str(price),
                "volume24h": str(vol),
                "change24h": str(change),
                "highPrice": str(high),
                "lowPrice": str(low),
                "bid1Price": str(price - spread / 2),
                "ask1Price": str(price + spread / 2),
            }
        return tickers

    async def analyze(self, data: dict) -> dict:
        try:
            self.update_status("running")

            api_ok = False
            try:
                client = await self._get_client()
                resp = await client.get(f"{MEXC_BASE}/api/v1/contract/detail")
                resp.raise_for_status()
                raw = resp.json().get("data", [])
                api_ok = True

                self.all_symbols = []
                for item in raw:
                    if item.get("futureType") == 1 and item.get("quoteCoin") == "USDT":
                        sym = item["symbol"]
                        base = item.get("baseCoin", "")
                        if self._is_stock_symbol(sym, base):
                            continue
                        self.all_symbols.append({
                            "symbol": sym,
                            "baseCoin": base,
                            "displayNameEn": item.get("displayNameEn", ""),
                            "priceScale": item.get("priceScale", 4),
                            "volScale": item.get("volScale", 0),
                            "amountScale": item.get("amountScale", 4),
                        })

                ticker_resp = await client.get(f"{MEXC_BASE}/api/v1/contract/ticker")
                if ticker_resp.status_code == 200:
                    self.tickers = {}
                    for item in ticker_resp.json().get("data", []):
                        sym = item.get("symbol", "")
                        if sym:
                            self.tickers[sym] = item
            except Exception:
                api_ok = False

            if not api_ok or not self.all_symbols or not self.tickers:
                cached = self._load_cached_symbols()
                if cached:
                    self.all_symbols = cached
                if not self.tickers:
                    self.tickers = self._generate_tickers_from_cache()

            scored = []
            major_coins = {"BTC", "ETH", "SOL", "XRP", "DOGE", "ADA", "AVAX", "LINK", "DOT", "PEPE", "WIF", "SUI", "SEI", "NEAR", "ARB", "OP", "APT", "FIL", "INJ", "TIA"}

            cached_symbols = set()
            if CACHE_DIR.exists():
                for f in CACHE_DIR.glob("*_Day1.json"):
                    cached_symbols.add(f.name.replace("_Day1.json", ""))

            for sym_info in self.all_symbols:
                symbol = sym_info["symbol"]
                base = sym_info.get("baseCoin", "")
                if base not in major_coins and len(scored) > 100:
                    continue

                ticker = self.tickers.get(symbol, {})
                if not ticker:
                    continue

                last_price = float(ticker.get("lastPrice", 0) or 0)
                volume_24h = float(ticker.get("volume24", 0) or ticker.get("volume24h", 0) or 0)
                change_24h = float(ticker.get("riseFallRate", 0) or ticker.get("change24h", 0) or 0)
                high_24h = float(ticker.get("high24Price", 0) or ticker.get("highPrice", 0) or ticker.get("h24High", 0) or 0)
                low_24h = float(ticker.get("lower24Price", 0) or ticker.get("lowPrice", 0) or ticker.get("h24Low", 0) or 0)
                bid = float(ticker.get("bid1", 0) or ticker.get("bid1Price", 0) or 0)
                ask = float(ticker.get("ask1", 0) or ticker.get("ask1Price", 0) or 0)

                if last_price <= 0 or volume_24h <= 0:
                    continue

                if high_24h > 0 and low_24h > 0:
                    volatility = (high_24h - low_24h) / last_price
                else:
                    volatility = abs(change_24h) / 100 if change_24h != 0 else 0.02

                spread_pct = (ask - bid) / bid if bid > 0 and ask > 0 else 0

                vol_score = min(volume_24h / 1_000_000, 10.0)
                volat_score = min(volatility * 100, 10.0)
                hot_score = vol_score * 0.5 + volat_score * 0.3 + min(abs(change_24h) / 5, 10) * 0.2

                if symbol in cached_symbols:
                    hot_score *= 1.5

                scored.append({
                    "symbol": symbol,
                    "baseCoin": base,
                    "displayName": sym_info.get("displayNameEn", ""),
                    "last_price": last_price,
                    "volume_24h": volume_24h,
                    "change_24h": change_24h,
                    "high_24h": high_24h,
                    "low_24h": low_24h,
                    "volatility": volatility,
                    "spread_pct": spread_pct,
                    "bid": bid,
                    "ask": ask,
                    "volume_score": vol_score,
                    "volatility_score": volat_score,
                    "hot_score": hot_score,
                })

            scored.sort(key=lambda x: x["volume_24h"], reverse=True)
            self.volume_leaders = scored[:20]

            scored_vol = sorted(scored, key=lambda x: x["volatility"], reverse=True)
            self.volatile_pairs = scored_vol[:20]

            scored_hot = sorted(scored, key=lambda x: x["hot_score"], reverse=True)
            self.hot_pairs = scored_hot[:20]

            self.update_status("ready")
            return {
                "hot_pairs": self.hot_pairs,
                "volatile_pairs": self.volatile_pairs,
                "volume_leaders": self.volume_leaders,
                "all_symbols": self.all_symbols,
                "ticker_count": len(self.tickers),
                "symbol_count": len(self.all_symbols),
                "data_source": "live" if api_ok else "cache",
            }

        except Exception as e:
            self.update_status("error", str(e))
            return {
                "hot_pairs": self.hot_pairs,
                "volatile_pairs": self.volatile_pairs,
                "volume_leaders": self.volume_leaders,
                "all_symbols": self.all_symbols,
                "error": str(e),
            }

    def get_symbol_ticker(self, symbol: str) -> dict | None:
        return self.tickers.get(symbol)

    def get_top_pairs(self, count: int = 20) -> list[dict]:
        return self.hot_pairs[:count]
