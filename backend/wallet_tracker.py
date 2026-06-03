"""
Balina cüzdan takibi - OTOMATİK TESPİT.
Kullanıcının hiçbir şey girmesine gerek yok.
Bot, zincirdeki büyük transferleri otomatik bulup takibe alır.
"""

import asyncio
import time
import logging
from collections import defaultdict

from config import config

logger = logging.getLogger(__name__)

# Dinamik takip listesi (otomatik tespit edilen balina cüzdanlar)
_watched_wallets = {}         # address -> {"label": "...", "balance": 0, "first_seen": ts, "last_seen": ts, "total_sent": 0, "total_received": 0, "tx_count": 0}
_recent_txs = []              # son büyük işlemler
_all_time_whales = set()      # bugüne kadar tespit edilen tüm balina adresleri

ETHERSCAN_BASE = "https://api.etherscan.io/v2/api"


async def _get(client, params):
    """Etherscan V2 API'ye GET isteği."""
    params["chainid"] = 1
    params["apikey"] = config.ETHERSCAN_KEY
    try:
        r = await client.get(ETHERSCAN_BASE, params=params, timeout=15)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        logger.debug(f"Etherscan error: {e}")
    return None


async def get_latest_block(client):
    """En son blok numarasını al."""
    data = await _get(client, {"module": "proxy", "action": "eth_blockNumber"})
    if data and data.get("result"):
        return int(data["result"], 16)
    return 0


async def get_block_transactions(client, block_number):
    """Belirli bir bloktaki işlemleri getir."""
    data = await _get(client, {"module": "proxy", "action": "eth_getBlockByNumber", "tag": hex(block_number), "boolean": "true"})
    if data and data.get("result"):
        return data["result"].get("transactions", [])
    return []


async def scan_for_whales(client):
    """
    Son blokları tara, büyük ETH transferlerini tespit et.
    Hiçbir ön tanımlı cüzdan olmadan çalışır.
    """
    global _watched_wallets, _recent_txs, _all_time_whales

    signals = []
    latest = await get_latest_block(client)
    if latest == 0:
        return signals

    # Sadece en son 5 bloğu tara (rate limit 5 call/sn, free tier)
    start_block = max(0, latest - 5)
    logger.debug(f"Scanning blocks {start_block}-{latest}")

    for bn in range(latest, start_block, -1):
        txs = await get_block_transactions(client, bn)
        for tx in txs[:50]:
            value_hex = tx.get("value", "0x0")
            try:
                value_wei = int(value_hex, 16)
            except:
                continue

            value_eth = value_wei / 1e18
            if value_eth < config.WHALE_THRESHOLD_ETH:
                continue

            from_addr = tx.get("from", "").lower()
            to_addr = tx.get("to", "").lower()
            tx_hash = tx.get("hash", "")[:12]

            if not from_addr or not to_addr:
                continue
            if from_addr == "0x0000000000000000000000000000000000000000":
                continue

            # Her iki adresi de takip listesine ekle
            for addr, role in [(from_addr, "sender"), (to_addr, "receiver")]:
                if addr not in _watched_wallets and len(_watched_wallets) < config.MAX_WATCHED_WALLETS:
                    label = f"Whale-{addr[:6]}"
                    _watched_wallets[addr] = {
                        "label": label,
                        "balance": 0,
                        "first_seen": time.time(),
                        "last_seen": time.time(),
                        "total_sent": 0,
                        "total_received": 0,
                        "tx_count": 0,
                    }
                    _all_time_whales.add(addr)
                    logger.info(f"New whale detected: {label} ({addr[:12]}...), {value_eth:.0f} ETH")

                if addr in _watched_wallets:
                    w = _watched_wallets[addr]
                    w["last_seen"] = time.time()
                    w["tx_count"] += 1
                    if role == "sender":
                        w["total_sent"] += value_eth
                    else:
                        w["total_received"] += value_eth

            # İşlemi kaydet
            _recent_txs.append({
                "hash": tx_hash,
                "from": from_addr[:10],
                "to": to_addr[:10],
                "value_eth": round(value_eth, 2),
                "block": bn,
                "ts": time.time(),
            })

            # Sinyal üret
            direction = "accumulate" if from_addr in _watched_wallets else "distribute"
            signals.append({
                "type": "whale_move",
                "label": _watched_wallets.get(to_addr, {}).get("label", "Unknown"),
                "from": from_addr[:10],
                "to": to_addr[:10],
                "value_eth": round(value_eth, 2),
                "direction": direction,
                "ts": time.time(),
            })

        await asyncio.sleep(0.3)  # rate limit koruması

    # Cache temizlik
    if len(_recent_txs) > 200:
        _recent_txs[:] = _recent_txs[-100:]

    logger.info(f"Scan done: {len(_watched_wallets)} watched, {len(signals)} new signals")
    return signals


async def check_wallet_balances(client):
    """Takip edilen cüzdanların güncel ETH bakiyelerini sorgula."""
    if not _watched_wallets:
        return []

    signals = []
    for addr, info in list(_watched_wallets.items()):
        data = await _get(client, {"module": "account", "action": "balance", "address": addr, "tag": "latest"})
        if data and data.get("status") == "1":
            balance = int(data["result"]) / 1e18
            prev = info["balance"]
            info["balance"] = round(balance, 4)

            if prev > 0 and abs(balance - prev) > 50:
                signals.append({
                    "type": "balance_change",
                    "label": info["label"],
                    "address": addr[:10],
                    "diff_eth": round(balance - prev, 2),
                    "direction": "accumulate" if balance > prev else "distribute",
                    "ts": time.time(),
                })

    return signals


def get_summary():
    """Dashboard için özet."""
    whales = []
    for addr, info in sorted(_watched_wallets.items(), key=lambda x: x[1]["total_sent"] + x[1]["total_received"], reverse=True)[:20]:
        whales.append({
            "label": info["label"],
            "address": addr[:12] + "...",
            "balance_eth": info["balance"],
            "total_sent": round(info["total_sent"], 1),
            "total_received": round(info["total_received"], 1),
            "tx_count": info["tx_count"],
            "last_seen": info["last_seen"],
        })
    return whales


def get_recent_txs(limit=20):
    return sorted(_recent_txs, key=lambda x: x["ts"], reverse=True)[:limit]


def get_stats():
    return {
        "total_whales_detected": len(_all_time_whales),
        "currently_watched": len(_watched_wallets),
        "total_txs_tracked": len(_recent_txs),
    }
