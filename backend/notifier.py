import os
import time
import httpx

THREADS_ACCESS_TOKEN = os.environ.get("THREADS_ACCESS_TOKEN", "")
THREADS_USER_ID = os.environ.get("THREADS_USER_ID", "")


def _format_time(ts):
    if not ts:
        return ""
    return time.strftime("%d.%m.%Y %H:%M:%S", time.localtime(ts))


async def notify_position_opened(pos: dict):
    symbol = pos.get("symbol", "?")
    direction = pos.get("direction", "?").upper()
    entry = pos.get("entry_price", 0)
    size = pos.get("size_usd", 0)
    leverage = pos.get("leverage", 1)
    sl = pos.get("stop_loss", 0)
    tp = pos.get("take_profit", 0)
    score = pos.get("patron_score", 0)
    entry_time = _format_time(pos.get("entry_time"))

    emoji = "🟢" if direction == "LONG" else "🔴"
    dir_text = "YUKARI" if direction == "LONG" else "ASAGI"

    text = f"""
{emoji} YENI POZISYON ACILDI

 {symbol} / {direction}
 Giris: ${entry:,.4f}
 Boyut: ${size:,.0f} x{leverage}
 SL: ${sl:,.4f} | TP: ${tp:,.4f}
 Skor: %{score*100:.0f}
 {entry_time}
""".strip()

    await _post_threads(text)


async def notify_position_closed(pos: dict, reason: str):
    symbol = pos.get("symbol", "?")
    direction = pos.get("direction", "?").upper()
    entry = pos.get("entry_price", 0)
    exit_price = pos.get("close_price", pos.get("current_price", 0))
    pnl = pos.get("pnl", 0)
    pnl_pct = pos.get("pnl_pct", 0)
    size = pos.get("size_usd", 0)
    leverage = pos.get("leverage", 1)
    close_time = _format_time(pos.get("close_time"))

    is_profit = pnl >= 0
    emoji = "💰" if is_profit else "💔"
    result = "KAZANC" if is_profit else "ZARAR"

    text = f"""
{emoji} POZISYON KAPATILDI - {result}

 {symbol} / {direction}
 Giris: ${entry:,.4f}
 Cikis: ${exit_price:,.4f}
 PnL: {'+' if is_profit else ''}{pnl:,.2f}$ ({'+' if is_profit else ''}{pnl_pct:.1f}%)
 Boyut: ${size:,.0f} x{leverage}
 Sebep: {reason}
 {close_time}
""".strip()

    await _post_threads(text)


async def _post_threads(text: str):
    if not THREADS_ACCESS_TOKEN or not THREADS_USER_ID:
        print("[NOTIFIER] Threads token/ID tanimli degin, atlanıyor")
        return

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            # 1. Medya olustur (text container)
            create_resp = await client.post(
                "https://graph.threads.net/v1.0/me/threads",
                data={
                    "media_type": "TEXT",
                    "text": text,
                    "access_token": THREADS_ACCESS_TOKEN,
                },
            )
            create_data = create_resp.json()

            if "id" not in create_data:
                print(f"[NOTIFIER] Threads container olusturma hatasi: {create_data}")
                return

            container_id = create_data["id"]

            # 2. Yayinla
            publish_resp = await client.post(
                "https://graph.threads.net/v1.0/me/threads_publish",
                data={
                    "creation_id": container_id,
                    "access_token": THREADS_ACCESS_TOKEN,
                },
            )
            publish_data = publish_resp.json()

            if "id" in publish_data:
                print(f"[NOTIFIER] Threads paylasimi basarili: {publish_data['id']}")
            else:
                print(f"[NOTIFIER] Threads yayinlama hatasi: {publish_data}")

    except Exception as e:
        print(f"[NOTIFIER] Threads gonderim hatasi: {e}")
