import os
import time
import httpx

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")


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
    dir_emoji = "📈" if direction == "LONG" else "📉"

    msg = f"""
{emoji} <b>YENİ POZİSYON AÇILDI</b> {dir_emoji}

 пар: <b>{symbol}</b>
 yön: <b>{direction}</b>
 giriş: <b>${entry:,.4f}</b>
 boyut: <b>${size:,.0f}</b> x{leverage}
 sl: <b>${sl:,.4f}</b>
 tp: <b>${tp:,.4f}</b>
 skor: <b>%{score*100:.0f}</b>
 tarih: {entry_time}
""".strip()

    await _send(msg)


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
    result = "KAZANÇ" if is_profit else "ZARAR"

    msg = f"""
{emoji} <b>POZİSYON KAPATILDI</b> - {result}

 пар: <b>{symbol}</b>
 yön: <b>{direction}</b>
 giriş: <b>${entry:,.4f}</b>
 çıkış: <b>${exit_price:,.4f}</b>
 pnl: <b>{'+' if is_profit else ''}{pnl:,.2f}$</b> ({'+' if is_profit else ''}{pnl_pct:.1f}%)
 boyut: <b>${size:,.0f}</b> x{leverage}
 sebep: <b>{reason}</b>
 tarih: {close_time}
""".strip()

    await _send(msg)


async def _send(text: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            await client.post(
                f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                json={
                    "chat_id": TELEGRAM_CHAT_ID,
                    "text": text,
                    "parse_mode": "HTML",
                    "disable_web_page_preview": True,
                },
            )
    except Exception as e:
        print(f"[NOTIFIER] Telegram gonderim hatasi: {e}")
