import os
import time
import io
import httpx

THREADS_ACCESS_TOKEN = os.environ.get("THREADS_ACCESS_TOKEN", "")
THREADS_USER_ID = os.environ.get("THREADS_USER_ID", "")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "5869116212:AAEAUTxVifmqDYecqfAn0m9DEv3l5osufDY")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "-1001685438330")


def _format_time(ts):
    if not ts:
        return ""
    return time.strftime("%d.%m.%Y %H:%M:%S", time.localtime(ts))


def generate_chart_svg(klines: list, direction: str, entry_price: float, exit_price: float = 0, tp: float = 0, sl: float = 0) -> str:
    """Klines verisinden basit SVG mum grafik olusturur."""
    if not klines or len(klines) < 2:
        return ""

    W, H = 800, 400
    PAD = 60
    chart_w = W - PAD * 2
    chart_h = H - PAD * 2

    prices = []
    for k in klines:
        prices.extend([float(k["open"]), float(k["high"]), float(k["low"]), float(k["close"])])
    all_prices = prices[:]
    if entry_price > 0:
        all_prices.append(entry_price)
    if exit_price > 0:
        all_prices.append(exit_price)
    if tp > 0:
        all_prices.append(tp)
    if sl > 0:
        all_prices.append(sl)

    min_p = min(all_prices) * 0.998
    max_p = max(all_prices) * 1.002
    price_range = max_p - min_p if max_p > min_p else 1

    def y(price):
        return PAD + chart_h - ((price - min_p) / price_range * chart_h)

    candle_w = max(1, chart_w / len(klines) - 1)

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">',
        f'<rect width="{W}" height="{H}" fill="#0a0e17"/>',
        f'<rect x="{PAD}" y="{PAD}" width="{chart_w}" height="{chart_h}" fill="none" stroke="#1a1e2e" stroke-width="0.5"/>',
    ]

    for i in range(5):
        py = PAD + chart_h * i / 4
        p = max_p - price_range * i / 4
        svg_parts.append(f'<line x1="{PAD}" y1="{py}" x2="{W-PAD}" y2="{py}" stroke="#1a1e2e" stroke-width="0.5"/>')
        svg_parts.append(f'<text x="{PAD-5}" y="{py+4}" text-anchor="end" fill="#555" font-size="9" font-family="monospace">${p:.4f}</text>')

    for i, k in enumerate(klines):
        o, h, l, c = float(k["open"]), float(k["high"]), float(k["low"]), float(k["close"])
        cx = PAD + i * (chart_w / len(klines)) + candle_w / 2
        is_up = c >= o
        color = "#00ff88" if is_up else "#ff3366"
        svg_parts.append(f'<line x1="{cx}" y1="{y(h)}" x2="{cx}" y2="{y(l)}" stroke="{color}" stroke-width="1"/>')
        body_top = y(max(o, c))
        body_bot = y(min(o, c))
        body_h = max(1, body_bot - body_top)
        svg_parts.append(f'<rect x="{cx - candle_w/2}" y="{body_top}" width="{candle_w}" height="{body_h}" fill="{color}"/>')

    if entry_price > 0:
        ey = y(entry_price)
        svg_parts.append(f'<line x1="{PAD}" y1="{ey}" x2="{W-PAD}" y2="{ey}" stroke="#ff9800" stroke-width="2" stroke-dasharray="0"/>')
        marker = "▲" if direction == "long" else "▼"
        svg_parts.append(f'<text x="{W-PAD+5}" y="{ey+4}" fill="#ff9800" font-size="11" font-family="monospace" font-weight="bold">{marker} GİRİŞ ${entry_price:.4f}</text>')

    if tp > 0:
        ty = y(tp)
        svg_parts.append(f'<line x1="{PAD}" y1="{ty}" x2="{W-PAD}" y2="{ty}" stroke="#76d672" stroke-width="2" stroke-dasharray="0"/>')
        svg_parts.append(f'<text x="{W-PAD+5}" y="{ty+4}" fill="#76d672" font-size="9" font-family="monospace">TP ${tp:.4f}</text>')

    if sl > 0:
        sy = y(sl)
        svg_parts.append(f'<line x1="{PAD}" y1="{sy}" x2="{W-PAD}" y2="{sy}" stroke="#ff3366" stroke-width="2" stroke-dasharray="0"/>')
        svg_parts.append(f'<text x="{W-PAD+5}" y="{sy+4}" fill="#ff3366" font-size="9" font-family="monospace">SL ${sl:.4f}</text>')

    if exit_price > 0:
        xy = y(exit_price)
        svg_parts.append(f'<circle cx="{W-PAD-10}" cy="{xy}" r="8" fill="#ffd700" opacity="0.9"/>')
        svg_parts.append(f'<text x="{W-PAD-10}" y="{xy+4}" text-anchor="middle" fill="#000" font-size="10">🏆</text>')
        svg_parts.append(f'<line x1="{PAD}" y1="{xy}" x2="{W-PAD-20}" y2="{xy}" stroke="#ffd700" stroke-width="1" stroke-dasharray="3,2"/>')
        svg_parts.append(f'<text x="{W-PAD+5}" y="{xy+4}" fill="#ffd700" font-size="9" font-family="monospace">🏆 ${exit_price:.4f}</text>')

    svg_parts.append('</svg>')
    return '\n'.join(svg_parts)


async def send_telegram_photo(svg_content: str, caption: str):
    """SVG icerigi Telegram'a photo olarak gonderir."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[NOTIFIER] Telegram token/ID tanimli degil")
        return

    try:
        svg_bytes = svg_content.encode('utf-8')
        files = {"photo": ("chart.svg", io.BytesIO(svg_bytes), "image/svg+xml")}

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto",
                data={"chat_id": TELEGRAM_CHAT_ID, "caption": caption, "parse_mode": "HTML"},
                files=files,
            )
            data = resp.json()
            if data.get("ok"):
                print(f"[NOTIFIER] Telegram grafik gonderimi basarili: {data['result']['message_id']}")
            else:
                print(f"[NOTIFIER] Telegram grafik gonderim hatasi: {data}")
    except Exception as e:
        print(f"[NOTIFIER] Telegram grafik gonderim hatasi: {e}")


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
    await _post_telegram(text)


async def notify_position_closed(pos: dict, reason: str):
    from mexc_client import get_klines

    symbol = pos.get("symbol", "?")
    direction = pos.get("direction", "?").upper()
    entry = pos.get("entry_price", 0)
    exit_price = pos.get("close_price", pos.get("current_price", 0))
    pnl = pos.get("net_pnl", pos.get("pnl", 0))
    pnl_pct = pos.get("pnl_pct", 0)
    size = pos.get("size_usd", 0)
    leverage = pos.get("leverage", 1)
    fees = pos.get("total_fees", 0)
    close_time = _format_time(pos.get("close_time"))
    entry_time = pos.get("entry_time", 0)
    close_time_ts = pos.get("close_time", time.time())
    margin_type = pos.get("margin_type", "isolated")

    is_profit = pnl >= 0
    emoji = "💰" if is_profit else "💔"
    result = "KAZANC" if is_profit else "ZARAR"

    text = f"""
{emoji} POZISYON KAPATILDI - {result}

 {symbol} / {direction} [{margin_type.upper()}]
 Giris: ${entry:,.4f}
 Cikis: ${exit_price:,.4f}
 PnL: {'+' if is_profit else ''}{pnl:,.4f}$ ({'+' if is_profit else ''}{pnl_pct:.1f}%)
 Fee: -${fees:,.4f}
 Boyut: ${size:,.0f} x{leverage}
 Sebep: {reason}
 {close_time}
""".strip()

    await _post_threads(text)
    await _post_telegram(text)

    try:
        klines = await get_klines(symbol, "Min5")
        if klines and len(klines) > 5:
            margin = 300
            filtered = [k for k in klines if (entry_time - margin) * 1000 <= k.get("time", 0) <= (close_time_ts + margin) * 1000]
            if not filtered:
                filtered = klines[-60:]

            svg = generate_chart_svg(
                filtered, direction.lower(), entry, exit_price,
                tp=pos.get("take_profit", 0), sl=pos.get("stop_loss", 0)
            )
            if svg:
                chart_caption = f"📊 {symbol} {direction} | Giriş: ${entry:.4f} → Çıkış: ${exit_price:.4f} | PnL: {'+' if is_profit else ''}{pnl:.4f}$ 🏆"
                await send_telegram_photo(svg, chart_caption)
    except Exception as e:
        print(f"[NOTIFIER] Grafik olusturma hatasi: {e}")


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


async def _post_telegram(text: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[NOTIFIER] Telegram token/ID tanimli degil, atlanıyor")
        return

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                json={
                    "chat_id": TELEGRAM_CHAT_ID,
                    "text": text,
                    "parse_mode": "HTML",
                },
            )
            data = resp.json()

            if data.get("ok"):
                print(f"[NOTIFIER] Telegram gonderimi basarili: {data['result']['message_id']}")
            else:
                print(f"[NOTIFIER] Telegram gonderim hatasi: {data}")

    except Exception as e:
        print(f"[NOTIFIER] Telegram gonderim hatasi: {e}")
