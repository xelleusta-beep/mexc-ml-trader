import httpx
import json
import os
import time
import hmac
import hashlib
import asyncio
from pathlib import Path
from typing import Optional

FUTURES_API_URL = "https://api.mexc.com"
BASE_URL = "https://api.mexc.com"
CACHE_DIR = Path(__file__).parent.parent / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

MEXC_API_KEY = os.environ.get("MEXC_API_KEY", "")
MEXC_SECRET_KEY = os.environ.get("MEXC_SECRET_KEY", "")

BLACKLIST_BASE_COINS = {
    "USDC", "BUSD", "DAI", "TUSD", "USDP", "FDUSD", "PYUSD",
    "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD", "CNY",
    "OIL", "GOLD", "SILVER", "NATGAS", "COPPER", "PLATINUM",
    "DOW", "SP500", "NASDAQ", "SPX", "DAX", "FTSE", "NIKKEI",
    "XAU", "XAG", "XPT", "XPD",
    "STOCK", "ETF", "BOND", "FUND", "INDEX", "FUTURES",
    "CRCLSTOCK", "METASTOCK", "MSTRSTOCK", "NVIDIA",
}


def _is_stock_symbol(symbol: str, base_coin: str) -> bool:
    base_upper = base_coin.upper()
    if base_upper in BLACKLIST_BASE_COINS:
        return True
    return False

# Connection pooling için global client
_client: Optional[httpx.AsyncClient] = None


async def get_client() -> httpx.AsyncClient:
    """Global HTTP client döndürür (connection pooling)."""
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            timeout=30,
            limits=httpx.Limits(
                max_connections=100,
                max_keepalive_connections=50,
                keepalive_expiry=30,
            ),
        )
    return _client


async def get_all_futures_symbols() -> list[dict]:
    """MEXC'deki tüm vadeli işlem gören sembolleri çeker."""
    cache_file = CACHE_DIR / "all_symbols.json"

    if cache_file.exists():
        age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
        if age_hours < 24:
            return json.loads(cache_file.read_text(encoding="utf-8"))

    client = await get_client()
    resp = await client.get(f"{BASE_URL}/api/v1/contract/detail")
    resp.raise_for_status()
    data = resp.json()

    symbols = []
    for item in data.get("data", []):
        if item.get("futureType") == 1 and item.get("quoteCoin") == "USDT":
            sym = item["symbol"]
            base = item.get("baseCoin", "")
            if _is_stock_symbol(sym, base):
                continue
            symbols.append({
                "symbol": sym,
                "baseCoin": base,
                "displayNameEn": item.get("displayNameEn", ""),
                "priceScale": item.get("priceScale", 4),
                "volScale": item.get("volScale", 0),
                "amountScale": item.get("amountScale", 4),
            })

    cache_file.write_text(json.dumps(symbols, ensure_ascii=False, indent=2), encoding="utf-8")
    return symbols


async def get_klines(
    symbol: str,
    interval: str = "Day1",
    start: Optional[int] = None,
    end: Optional[int] = None,
) -> list[dict]:
    """Belirli bir sembol için mum verilerini çeker."""
    safe_symbol = symbol.replace("/", "_")
    cache_file = CACHE_DIR / f"{safe_symbol}_{interval}.json"

    if cache_file.exists():
        age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
        if age_hours < 12:
            return json.loads(cache_file.read_text(encoding="utf-8"))

    params = {"interval": interval}
    if start:
        params["start"] = start
    if end:
        params["end"] = end

    client = await get_client()
    resp = await client.get(
        f"{BASE_URL}/api/v1/contract/kline/{safe_symbol}",
        params=params,
    )
    resp.raise_for_status()
    data = resp.json()

    klines = []
    raw_data = data.get("data", {})

    if isinstance(raw_data, dict) and "time" in raw_data:
        times = raw_data.get("time", [])
        opens = raw_data.get("open", [])
        highs = raw_data.get("high", [])
        lows = raw_data.get("low", [])
        closes = raw_data.get("close", [])
        vols = raw_data.get("vol", [])

        for i in range(len(times)):
            klines.append({
                "time": times[i] * 1000 if times[i] < 1e12 else times[i],
                "open": float(opens[i]) if i < len(opens) else 0,
                "high": float(highs[i]) if i < len(highs) else 0,
                "low": float(lows[i]) if i < len(lows) else 0,
                "close": float(closes[i]) if i < len(closes) else 0,
                "vol": float(vols[i]) if i < len(vols) else 0,
            })
    elif isinstance(raw_data, list):
        for item in raw_data:
            klines.append({
                "time": item.get("time", 0),
                "open": float(item.get("open", 0)),
                "high": float(item.get("high", 0)),
                "low": float(item.get("low", 0)),
                "close": float(item.get("close", 0)),
                "vol": float(item.get("vol", 0)),
            })

    klines.sort(key=lambda x: x["time"])

    cache_file.write_text(json.dumps(klines, ensure_ascii=False, indent=2), encoding="utf-8")
    return klines


async def get_klines_batch(symbols: list[str], interval: str = "Day1") -> dict:
    """Birden fazla sembol için eşzamanlı veri çeker."""
    tasks = [get_klines(sym, interval) for sym in symbols]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    klines_dict = {}
    for sym, result in zip(symbols, results):
        if isinstance(result, Exception):
            raise result
        klines_dict[sym] = result

    return klines_dict


def _sign_request(req_time: str, param_str: str) -> str:
    """MEXC Futures API HMAC-SHA256 imzası."""
    to_sign = MEXC_API_KEY + req_time + param_str
    return hmac.new(MEXC_SECRET_KEY.encode(), to_sign.encode(), hashlib.sha256).hexdigest()


def _get_auth_headers(param_str: str = "") -> dict:
    """Imzali request header'lari olusturur."""
    req_time = str(int(time.time() * 1000))
    signature = _sign_request(req_time, param_str)
    return {
        "ApiKey": MEXC_API_KEY,
        "Request-Time": req_time,
        "Signature": signature,
        "Content-Type": "application/json",
    }


async def futures_set_leverage(symbol: str, leverage: int, open_type: int = 1, position_type: int = 1) -> dict:
    """Futures kaldirac degistir. open_type: 1=izole, 2=cross. position_type: 1=long, 2=short."""
    body = json.dumps({
        "symbol": symbol,
        "leverage": leverage,
        "openType": open_type,
        "positionType": position_type,
    })
    headers = _get_auth_headers(body)
    client = await get_client()
    resp = await client.post(
        f"{FUTURES_API_URL}/api/v1/private/position/change_leverage",
        content=body,
        headers=headers,
    )
    return resp.json()


async def futures_submit_order(
    symbol: str,
    side: int,
    vol: float,
    leverage: int = 1,
    open_type: int = 1,
    order_type: int = 2,
    price: float = 0,
) -> dict:
    """
    Futures emir gonder.
    side: 1=long ac, 2=short ac, 3=short kapat, 4=long kapat
    open_type: 1=izole, 2=cross
    order_type: 2=market
    vol: kontrat miktari
    """
    order_data = {
        "symbol": symbol,
        "side": side,
        "type": order_type,
        "openType": open_type,
        "vol": vol,
        "leverage": leverage,
    }
    if order_type == 1 and price > 0:
        order_data["price"] = price

    body = json.dumps(order_data)
    headers = _get_auth_headers(body)
    client = await get_client()
    resp = await client.post(
        f"{FUTURES_API_URL}/api/v1/private/order/submit",
        content=body,
        headers=headers,
    )
    return resp.json()


async def futures_get_positions() -> list:
    """Acik pozisyonlari cek."""
    req_time = str(int(time.time() * 1000))
    param_str = ""
    signature = _sign_request(req_time, param_str)
    headers = {
        "ApiKey": MEXC_API_KEY,
        "Request-Time": req_time,
        "Signature": signature,
    }
    client = await get_client()
    resp = await client.get(
        f"{FUTURES_API_URL}/api/v1/private/position/open",
        headers=headers,
    )
    data = resp.json()
    return data.get("data", []) if data.get("success") else []


async def futures_get_assets() -> dict:
    """Hesap varliklarini cek."""
    req_time = str(int(time.time() * 1000))
    param_str = ""
    signature = _sign_request(req_time, param_str)
    headers = {
        "ApiKey": MEXC_API_KEY,
        "Request-Time": req_time,
        "Signature": signature,
    }
    client = await get_client()
    resp = await client.get(
        f"{FUTURES_API_URL}/api/v1/private/account/assets",
        headers=headers,
    )
    data = resp.json()
    if data.get("success"):
        for asset in data.get("data", []):
            if asset.get("currency") == "USDT":
                return {
                    "total": float(asset.get("availableBalance", 0)) + float(asset.get("positionMargin", 0)),
                    "available": float(asset.get("availableBalance", 0)),
                    "position_margin": float(asset.get("positionMargin", 0)),
                    "unrealized_pnl": float(asset.get("positionProfit", 0)),
                }
    return {}


async def futures_get_contract_info(symbol: str) -> dict:
    """Kontrat bilgisini cek (min vol, price scale vs)."""
    client = await get_client()
    resp = await client.get(f"{FUTURES_API_URL}/api/v1/contract/detail")
    data = resp.json()
    for item in data.get("data", []):
        if item.get("symbol") == symbol:
            return item
    return {}


def calc_contract_vol(size_usd: float, price: float, leverage: int, contract_size: float = 1.0) -> float:
    """USD miktarindan kontrat sayisini hesapla."""
    notional = size_usd * leverage
    vol = notional / (price * contract_size) if price > 0 and contract_size > 0 else 0
    return max(1, round(vol))
