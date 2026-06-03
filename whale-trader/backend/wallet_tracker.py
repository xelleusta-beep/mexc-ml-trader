"""
Balina cüzdan takibi.
Etherscan/BSCScan API ile cüzdan bakiyelerini ve son işlemleri kontrol eder.
Büyük hareketlerde sinyal üretir.
"""

import time
import logging
from datetime import datetime, timezone
from collections import defaultdict

from config import config

logger = logging.getLogger(__name__)

# Önbellek
_wallet_cache = {}       # address -> {"balance": ..., "last_tx": ..., "ts": ...}
_recent_tx_cache = []    # son tespit edilen büyük işlemler

CHAIN_API = {
    "ethereum": "https://api.etherscan.io/api",
    "bsc": "https://api.bscscan.com/api",
}

CHAIN_API_KEY = {
    "ethereum": config.ETHERSCAN_KEY,
    "bsc": config.BSCSCAN_KEY,
}

USDT_CONTRACTS = {
    "ethereum": "0xdAC17F958D2ee523a2206206994597C13D831ec7",
    "bsc": "0x55d398326f99059fF775485246999027B3197955",
}

USDC_CONTRACTS = {
    "ethereum": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
    "bsc": "0x8AC76a51cc950d9822D68b83fE1Ad97B32Cd580d",
}


async def fetch_balance(client, address, chain="ethereum"):
    """ETH/BSC ana token bakiyesini sorgula (USD cinsinden değil, native token)."""
    base = CHAIN_API.get(chain)
    key = CHAIN_API_KEY.get(chain)
    if not base or not key:
        return 0

    url = f"{base}?module=account&action=balance&address={address}&tag=latest&apikey={key}"
    try:
        r = await client.get(url, timeout=10)
        data = r.json()
        if data.get("status") == "1":
            bal = int(data["result"]) / 1e18
            return round(bal, 4)
    except Exception as e:
        logger.debug(f"Balance fetch error {address}: {e}")
    return 0


async def fetch_token_balance(client, address, contract, chain="ethereum"):
    """ERC-20/BEP-20 token bakiyesi sorgula."""
    base = CHAIN_API.get(chain)
    key = CHAIN_API_KEY.get(chain)
    if not base or not key:
        return 0

    url = f"{base}?module=account&action=tokenbalance&contractaddress={contract}&address={address}&tag=latest&apikey={key}"
    try:
        r = await client.get(url, timeout=10)
        data = r.json()
        if data.get("status") == "1":
            bal = int(data["result"]) / 1e18
            return round(bal, 4)
    except Exception as e:
        logger.debug(f"Token balance error {address}: {e}")
    return 0


async def fetch_recent_txs(client, address, chain="ethereum"):
    """Son 50 işlemi getir, büyük transferleri tespit et."""
    base = CHAIN_API.get(chain)
    key = CHAIN_API_KEY.get(chain)
    if not base or not key:
        return []

    url = f"{base}?module=account&action=txlist&address={address}&sort=desc&offset=50&page=1&apikey={key}"
    try:
        r = await client.get(url, timeout=10)
        data = r.json()
        if data.get("status") == "1":
            return data["result"]
    except Exception as e:
        logger.debug(f"TX fetch error {address}: {e}")
    return []


async def fetch_token_txs(client, address, contract, chain="ethereum"):
    """ERC-20 token transferlerini getir."""
    base = CHAIN_API.get(chain)
    key = CHAIN_API_KEY.get(chain)
    if not base or not key:
        return []

    url = f"{base}?module=account&action=tokentx&contractaddress={contract}&address={address}&sort=desc&offset=50&page=1&apikey={key}"
    try:
        r = await client.get(url, timeout=10)
        data = r.json()
        if data.get("status") == "1":
            return data["result"]
    except Exception as e:
        logger.debug(f"Token TX error {address}: {e}")
    return []


def _classify_tx(tx, chain="ethereum"):
    """İşlemi sınıflandır: büyük transfer mi, exchange mi vs."""
    value_eth = int(tx.get("value", 0)) / 1e18
    if value_eth < 10:  # 10 ETH altı küçük işlem
        return None

    tx_hash = tx.get("hash", "")[:10]
    to_addr = tx.get("to", "").lower()
    from_addr = tx.get("from", "").lower()

    # Basit kural: bilinen exchange adreslerine giden = muhtemel satış
    # (gerçek exchange adresleri eklenecek)
    known_exchanges = [
        "0x28c6c06298d514db089934071355e5743bf21d60",  # Binance 14
        "0x21a31ee1afc51d94c2efccaa2092ad1028285549",  # Binance 15
        "0xBE0eB53F46cd790Cd13851d5EFf43D12404d33E8",  # Binance 18
        "0xf977814e90da44bfa03b6295a0616a897441acec",  # Binance 19
        "0x3f5CE5FBFe3E9af3971dD833D26bA5b1C7F9E8d",  # Binance 20
    ]

    event_type = "unknown"
    if to_addr in known_exchanges or from_addr in known_exchanges:
        event_type = "exchange"
    elif value_eth >= 1000:
        event_type = "whale_move"
    elif value_eth >= 100:
        event_type = "large_move"

    return {
        "hash": tx_hash,
        "from": from_addr[:10],
        "to": to_addr[:10],
        "value_eth": round(value_eth, 2),
        "type": event_type,
        "chain": chain,
        "timestamp": tx.get("timeStamp", ""),
        "ts": time.time(),
    }


async def scan_wallets(client):
    """Tüm takip edilen cüzdanları tara, büyük hareketleri raporla."""
    global _recent_tx_cache

    signals = []
    now = time.time()

    for chain, wallets in config.TRACKED_WALLETS.items():
        for wallet in wallets:
            address = wallet["address"]
            label = wallet["label"]

            # ETH bakiyesi
            balance = await fetch_balance(client, address, chain)
            prev = _wallet_cache.get(address, {})

            # Büyük bakiye değişimi
            if prev and "balance" in prev:
                diff = balance - prev["balance"]
                if abs(diff) > 10:  # 10 ETH üzeri değişim
                    signals.append({
                        "type": "balance_change",
                        "label": label,
                        "address": address[:10],
                        "chain": chain,
                        "diff_eth": round(diff, 2),
                        "direction": "accumulate" if diff > 0 else "distribute",
                        "ts": now,
                    })
                    logger.info(f"Balance change [{label}]: {diff:+.2f} ETH")

            _wallet_cache[address] = {"balance": balance, "ts": now}

            # Son işlemler
            txs = await fetch_recent_txs(client, address, chain)
            for tx in txs[:10]:
                event = _classify_tx(tx, chain)
                if event and event["value_eth"] >= 100:
                    already_seen = any(t["hash"] == event["hash"] for t in _recent_tx_cache[-50:])
                    if not already_seen:
                        _recent_tx_cache.append(event)
                        signals.append({
                            "type": f"tx_{event['type']}",
                            "label": label,
                            "address": address[:10],
                            "chain": chain,
                            "value_eth": event["value_eth"],
                            "tx_hash": event["hash"],
                            "ts": now,
                        })
                        logger.info(f"Big TX [{label}]: {event['value_eth']} ETH ({event['type']})")

            # Token bakiyeleri (USDT/USDC)
            for stable in [USDT_CONTRACTS[chain], USDC_CONTRACTS[chain]]:
                stable_bal = await fetch_token_balance(client, address, stable, chain)
                if stable_bal > 100_000:
                    logger.debug(f"[{label}] {stable_bal:.0f} stablecoin")

    # Cache temizlik
    if len(_recent_tx_cache) > 200:
        _recent_tx_cache[:] = _recent_tx_cache[-100:]

    return signals


def get_wallet_summary():
    """Dashboard için özet."""
    result = []
    for chain, wallets in config.TRACKED_WALLETS.items():
        for w in wallets:
            info = _wallet_cache.get(w["address"], {})
            result.append({
                "label": w["label"],
                "address": w["address"][:12] + "...",
                "chain": chain,
                "balance_eth": info.get("balance", 0),
                "last_seen": info.get("ts", 0),
            })
    return result


def get_recent_txs(limit=20):
    """Son büyük işlemler."""
    return sorted(_recent_tx_cache, key=lambda x: x["ts"], reverse=True)[:limit]
