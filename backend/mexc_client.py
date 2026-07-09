import httpx
import json
import os
import time
import asyncio
from pathlib import Path
from typing import Optional

BASE_URL = "https://api.mexc.co"
CACHE_DIR = Path(__file__).parent.parent / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

STOCK_KEYWORDS = {"STOCK", "ETF", "BOND", "FUND", "INDEX", "FUTURES"}


def _is_stock_symbol(symbol: str, base_coin: str) -> bool:
    sym_upper = symbol.upper()
    base_upper = base_coin.upper()
    for kw in STOCK_KEYWORDS:
        if kw in sym_upper or kw in base_upper:
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
