import asyncio
import time
import json
from pathlib import Path
from agents import (
    ScannerAgent, TechnicalAgent, SentimentAgent,
    RiskManagerAgent, PortfolioManagerAgent, PatronAgent,
)
from mexc_client import get_klines, get_client, futures_submit_order, futures_set_leverage, futures_get_positions, futures_get_assets, futures_get_contract_info, calc_contract_vol
from data_store import save_state, load_state
from notifier import notify_position_opened, notify_position_closed

LOG_DIR = Path(__file__).parent.parent / "data" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


class Orchestrator:
    def __init__(self):
        self.scanner = ScannerAgent()
        self.technical = TechnicalAgent()
        self.sentiment = SentimentAgent()
        self.risk = RiskManagerAgent()
        self.portfolio = PortfolioManagerAgent()
        self.patron = PatronAgent()

        self.running = False
        self.cycle_count = 0
        self.last_cycle_time = 0.0
        self.total_cycles = 0
        self.trade_count = 0

        self.open_positions: list[dict] = []
        self.trade_history: list[dict] = []
        self.total_equity = 100.0
        self.available_capital = 100.0

        self.websocket_clients: list = []
        self.latest_results: dict = {}
        self._position_updater_task = None

        self._load_persisted_state()

    async def start(self, interval: int = 300):
        self.running = True
        self.total_equity = self.risk.total_equity
        self.available_capital = self.total_equity
        self._log("Orchestrator baslatildi")
        self._position_updater_task = asyncio.create_task(self._position_updater_loop())

        while self.running:
            try:
                await self.run_cycle()
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._log(f"Dongu hatasi: {e}")
                await asyncio.sleep(30)

    async def stop(self):
        self.running = False
        if self._position_updater_task:
            self._position_updater_task.cancel()
        self._persist_state()
        self._log("Orchestrator durduruldu")

    def _load_persisted_state(self):
        state = load_state()
        if state:
            self.trade_history = state.get("trade_history", [])
            self.open_positions = state.get("open_positions", [])
            self.total_equity = state.get("total_equity", 100.0)
            self.available_capital = state.get("available_capital", 100.0)
            self.cycle_count = state.get("cycle_count", 0)
            self.trade_count = state.get("trade_count", 0)
            if self.total_equity > 200:
                self.total_equity = 100.0
                self.available_capital = 100.0
                self.open_positions = []
                self.trade_history = []
                self._log("Eski state temizlendi - $100 kasa ile baslatildi")
            else:
                self._log(f"Kalici veri yuklendi: {len(self.trade_history)} islem, {len(self.open_positions)} pozisyon, ${self.total_equity:.2f}")

    def _persist_state(self):
        save_state(
            trade_history=self.trade_history,
            open_positions=self.open_positions,
            total_equity=self.total_equity,
            available_capital=self.available_capital,
            cycle_count=self.cycle_count,
            trade_count=self.trade_count,
        )

    async def run_cycle(self) -> dict:
        cycle_start = time.time()
        self.cycle_count += 1
        self._log(f"Dongu #{self.cycle_count} baslatildi")

        try:
            scanner_result = await self.scanner.analyze({})
            self._log(f"Scanner: {scanner_result.get('symbol_count', 0)} sembol tarandi")

            self._update_positions_with_live_prices(scanner_result)
            closed_by_sl_tp = self._check_sl_tp()
            self._update_unrealized_pnl()

            top_pairs = scanner_result.get("hot_pairs", [])[:20]
            symbols_to_analyze = [p["symbol"] for p in top_pairs]

            sentiment_result = await self.sentiment.analyze({"scanner": scanner_result})
            self._log(f"Sentiment: Fear/Greed={sentiment_result.get('fear_greed_index', '?')}")

            technical_result = await self.technical.analyze({"symbols": symbols_to_analyze})
            self._log(f"Technical: {technical_result.get('signal_count', 0)} sinyal uretildi")

            portfolio_result = await self.portfolio.analyze({
                "total_equity": self.total_equity,
                "available_capital": self.available_capital,
                "open_positions": self.open_positions,
                "scanner": scanner_result,
                "technical": technical_result,
            })

            best_signal = None
            signals = technical_result.get("signals", [])
            if signals:
                best_signal = max(signals, key=lambda x: x.get("confidence", 0))
                best_signal["symbol"] = best_signal.get("symbol", signals[0].get("symbol", ""))

            risk_result = await self.risk.analyze({
                "total_equity": self.total_equity,
                "open_positions": self.open_positions,
                "signal": best_signal,
                "portfolio": portfolio_result,
            })

            patron_result = await self.patron.analyze({
                "scanner": scanner_result,
                "technical": technical_result,
                "sentiment": sentiment_result,
                "risk": risk_result,
                "portfolio": portfolio_result,
            })

            executed = await self._execute_decisions(patron_result)

            risk_result_final = await self.risk.analyze({
                "total_equity": self.total_equity,
                "open_positions": self.open_positions,
            })

            cycle_time = time.time() - cycle_start
            self.last_cycle_time = cycle_time

            self.latest_results = {
                "cycle": self.cycle_count,
                "timestamp": time.time(),
                "cycle_time": round(cycle_time, 2),
                "scanner": {
                    "symbol_count": scanner_result.get("symbol_count", 0),
                    "hot_pairs": scanner_result.get("hot_pairs", [])[:20],
                },
                "sentiment": {
                    "fear_greed_index": sentiment_result.get("fear_greed_index", 50),
                    "fear_greed_label": sentiment_result.get("fear_greed_label", "Neutral"),
                    "fear_greed_context": sentiment_result.get("fear_greed_context", ""),
                    "overall_sentiment": sentiment_result.get("overall_sentiment", 0),
                    "market_mood": sentiment_result.get("market_mood", "neutral"),
                    "mood_analysis": sentiment_result.get("mood_analysis", {}),
                    "news_stats": sentiment_result.get("news_stats", {}),
                    "news_headlines": sentiment_result.get("news_headlines", []),
                },
                "technical": {
                    "signal_count": technical_result.get("signal_count", 0),
                    "analyzed_count": technical_result.get("analyzed_count", 0),
                    "signals": technical_result.get("signals", []),
                },
                "risk": risk_result_final.get("risk_metrics", {}),
                "portfolio": {
                    "total_equity": self.total_equity,
                    "available_capital": self.available_capital,
                    "position_count": len(self.open_positions),
                    "total_exposure_usd": sum(p.get("size_usd", 0) for p in self.open_positions),
                    "total_unrealized_pnl": sum(p.get("unrealized_pnl", 0) for p in self.open_positions),
                },
                "patron": {
                    "market_regime": patron_result.get("market_regime", "neutral"),
                    "overall_confidence": patron_result.get("overall_confidence", 0),
                    "approved_count": patron_result.get("approved_count", 0),
                    "rejected_count": patron_result.get("rejected_count", 0),
                    "top_picks": patron_result.get("top_picks", []),
                    "decisions": patron_result.get("decisions", []),
                },
                "positions": self.open_positions,
                "trade_history": self.trade_history[-20:],
                "executed": executed,
                "closed_by_sl_tp": closed_by_sl_tp,
            }

            await self._broadcast_update(self.latest_results)
            self._log(f"Dongu #{self.cycle_count} tamamlandi ({cycle_time:.1f}s) - Acik: {len(self.open_positions)} - Kapali: {len(closed_by_sl_tp)} - PnL: ${sum(p.get('unrealized_pnl', 0) for p in self.open_positions):.2f}")

            return self.latest_results

        except Exception as e:
            self._log(f"Dongu #{self.cycle_count} hatasi: {e}")
            return {"error": str(e), "cycle": self.cycle_count}

    def _update_positions_with_live_prices(self, scanner_result: dict):
        hot_pairs = scanner_result.get("hot_pairs", [])
        ticker_map = {p["symbol"]: p for p in hot_pairs}

        for pos in self.open_positions:
            symbol = pos["symbol"]
            ticker = ticker_map.get(symbol)
            if ticker:
                pos["current_price"] = ticker.get("last_price", pos.get("current_price", pos["entry_price"]))
                pos["price_source"] = "live"
            else:
                if "current_price" not in pos:
                    pos["current_price"] = pos["entry_price"]
                    pos["price_source"] = "entry_fallback"

    def _check_sl_tp(self) -> list[dict]:
        closed = []
        for pos in list(self.open_positions):
            current = pos.get("current_price", pos["entry_price"])
            if current <= 0:
                continue

            sl = pos.get("stop_loss", 0)
            tp = pos.get("take_profit", 0)
            direction = pos.get("direction", "")

            close_reason = None
            if direction == "long":
                if sl > 0 and current <= sl:
                    close_reason = "SL tetiklendi"
                elif tp > 0 and current >= tp:
                    close_reason = "TP tetiklendi"
            elif direction == "short":
                if sl > 0 and current >= sl:
                    close_reason = "SL tetiklendi"
                elif tp > 0 and current <= tp:
                    close_reason = "TP tetiklendi"

            if close_reason:
                entry = pos["entry_price"]
                if direction == "long":
                    pnl_pct = (current - entry) / entry
                else:
                    pnl_pct = (entry - current) / entry

                leveraged_pnl_pct = pnl_pct * pos.get("leverage", 1)
                pnl_usd = pos.get("size_usd", 0) * leveraged_pnl_pct

                pos["unrealized_pnl"] = round(pnl_usd, 2)
                pos["unrealized_pnl_pct"] = round(leveraged_pnl_pct * 100, 2)
                pos["current_price"] = current
                pos["close_price"] = current
                pos["close_time"] = time.time()
                pos["close_reason"] = close_reason
                pos["pnl"] = round(pnl_usd, 2)
                pos["pnl_pct"] = round(leveraged_pnl_pct * 100, 2)

                self._close_position(pos, close_reason)
                closed.append(pos)

        return closed

    def _update_unrealized_pnl(self):
        for pos in self.open_positions:
            current = pos.get("current_price", pos["entry_price"])
            entry = pos["entry_price"]
            direction = pos.get("direction", "")
            leverage = pos.get("leverage", 1)
            size = pos.get("size_usd", 0)

            if direction == "long":
                pnl_pct = (current - entry) / entry if entry > 0 else 0
            else:
                pnl_pct = (entry - current) / entry if entry > 0 else 0

            leveraged_pnl_pct = pnl_pct * leverage
            pnl_usd = size * leveraged_pnl_pct

            pos["unrealized_pnl"] = round(pnl_usd, 2)
            pos["unrealized_pnl_pct"] = round(leveraged_pnl_pct * 100, 2)

    async def _execute_decisions(self, patron_result: dict) -> list[dict]:
        executed = []
        decisions = patron_result.get("decisions", [])

        for decision in decisions:
            if not decision.get("approved"):
                continue

            symbol = decision.get("symbol", "")
            direction = decision.get("direction", "")
            price = decision.get("price", 0)

            if not symbol or not direction or price <= 0:
                continue

            existing = next((p for p in self.open_positions if p.get("symbol") == symbol), None)
            if existing:
                continue

            if len(self.open_positions) >= self.risk.max_positions:
                break

            actual_price = await self._get_current_price(symbol)
            if actual_price <= 0:
                actual_price = price

            price_diff_pct = abs(actual_price - price) / price if price > 0 else 0
            if price_diff_pct > 0.03:
                self._log(f"Fiyat uyumsuz: {symbol} karar={price} piyasa={actual_price} (%{price_diff_pct*100:.1f} fark) - atlandi")
                continue

            individual_risk = await self.risk.analyze({
                "total_equity": self.total_equity - sum(p.get("size_usd", 0) for p in self.open_positions),
                "open_positions": self.open_positions,
                "signal": {
                    "symbol": symbol,
                    "direction": direction,
                    "confidence": decision.get("composite_score", 0.5),
                    "price": actual_price,
                    "atr_pct": 0.02,
                },
                "portfolio": {},
            })

            risk_decision = individual_risk.get("decision", {})
            if not risk_decision.get("approved"):
                continue

            position = {
                "symbol": symbol,
                "direction": direction,
                "size_usd": risk_decision.get("position_size_usd", 500),
                "leverage": risk_decision.get("leverage", 1),
                "stop_loss": risk_decision.get("stop_loss", 0),
                "take_profit": risk_decision.get("take_profit", 0),
                "entry_price": actual_price,
                "current_price": actual_price,
                "entry_time": time.time(),
                "unrealized_pnl": 0,
                "unrealized_pnl_pct": 0,
                "status": "open",
                "patron_score": decision.get("composite_score", 0),
            }

            self.open_positions.append(position)
            self.available_capital -= position["size_usd"]
            self.trade_count += 1

            executed.append({
                "action": "open",
                "position": position,
                "decision": decision,
            })

            self.patron.log_decision(decision)
            self._log(f"ISLEM ACILDI: {symbol} {direction.upper()} ${position['size_usd']:.0f} x{position['leverage']} @${actual_price} SL:{position['stop_loss']} TP:{position['take_profit']}")
            asyncio.create_task(notify_position_opened(position))

        self._persist_state()

        for pos in list(self.open_positions):
            symbol = pos["symbol"]
            patron_decision = next(
                (d for d in decisions if d.get("symbol") == symbol and d.get("approved")),
                None
            )

            if patron_decision and patron_decision.get("direction") != pos["direction"]:
                self._close_position(pos, "Patron yon degistirme")
                executed.append({
                    "action": "close",
                    "symbol": symbol,
                    "reason": "Yon degistirme",
                })

        return executed

    async def _get_current_price(self, symbol: str) -> float:
        try:
            client = await get_client()
            resp = await client.get(f"https://api.mexc.com/api/v1/contract/ticker")
            if resp.status_code == 200:
                for item in resp.json().get("data", []):
                    if item.get("symbol") == symbol:
                        return float(item.get("lastPrice", 0))
        except Exception:
            pass

        try:
            klines = await get_klines(symbol, "Min1")
            if klines and len(klines) > 0:
                return float(klines[-1].get("close", 0))
        except Exception:
            pass

        return 0

    def _close_position(self, position: dict, reason: str):
        symbol = position["symbol"]
        current = position.get("current_price", position["entry_price"])
        entry = position["entry_price"]
        direction = position.get("direction", "")
        leverage = position.get("leverage", 1)
        size = position.get("size_usd", 0)

        if direction == "long":
            pnl_pct = (current - entry) / entry if entry > 0 else 0
        else:
            pnl_pct = (entry - current) / entry if entry > 0 else 0

        leveraged_pnl_pct = pnl_pct * leverage
        pnl_usd = size * leveraged_pnl_pct

        self.total_equity += pnl_usd
        self.available_capital += size + pnl_usd

        self.open_positions = [p for p in self.open_positions if p["symbol"] != symbol]
        self.trade_count += 1

        trade_record = {
            "symbol": symbol,
            "direction": direction,
            "entry_price": entry,
            "exit_price": current,
            "size_usd": size,
            "leverage": leverage,
            "entry_time": position.get("entry_time", time.time()),
            "close_time": time.time(),
            "close_reason": reason,
            "pnl": round(pnl_usd, 2),
            "pnl_pct": round(leveraged_pnl_pct * 100, 2),
            "patron_score": position.get("patron_score", 0),
            "status": "closed",
        }
        self.trade_history.append(trade_record)

        emoji = "+" if pnl_usd >= 0 else ""
        self._log(f"POZISYON KAPATILDI: {symbol} {direction.upper()} @${current} | PnL: {emoji}${pnl_usd:.2f} ({emoji}{leveraged_pnl_pct*100:.1f}%) - {reason}")
        asyncio.create_task(notify_position_closed(position, reason))

    async def _broadcast_update(self, data: dict):
        message = json.dumps(data, default=str)
        disconnected = []
        for ws in self.websocket_clients:
            try:
                await ws.send_text(message)
            except Exception:
                disconnected.append(ws)
        for ws in disconnected:
            self.websocket_clients.remove(ws)

    def _log(self, message: str):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"[{timestamp}] {message}"
        print(log_line)

        try:
            log_file = LOG_DIR / f"orchestrator_{time.strftime('%Y%m%d')}.log"
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(log_line + "\n")
        except Exception:
            pass

    async def _position_updater_loop(self):
        """Her 10 sn'de bir pozisyon fiyatlarini guncelle ve 1dk mum ile TP/SL kontrol et."""
        while self.running:
            try:
                await asyncio.sleep(10)
                if not self.open_positions:
                    continue

                await self._fast_update_positions()
                await self._broadcast_positions_only()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._log(f"Hizli guncelleme hatasi: {e}")
                await asyncio.sleep(5)

    async def _fast_update_positions(self):
        """Acik pozisyonlarin fiyatlarini ticker API ile 10sn'de bir guncelle."""
        symbols = list(set(p["symbol"] for p in self.open_positions))
        if not symbols:
            return

        ticker_prices = {}
        try:
            client = await get_client()
            resp = await client.get("https://api.mexc.com/api/v1/contract/ticker")
            if resp.status_code == 200:
                for item in resp.json().get("data", []):
                    sym = item.get("symbol", "")
                    lp = float(item.get("lastPrice", 0) or 0)
                    if sym and lp > 0:
                        ticker_prices[sym] = lp
        except Exception:
            pass

        if not ticker_prices:
            try:
                klines = await get_klines(symbols[0], "Min1") if symbols else None
                if klines and len(klines) > 0:
                    ticker_prices[symbols[0]] = float(klines[-1].get("close", 0))
            except Exception:
                pass

        now = time.time()
        for pos in list(self.open_positions):
            symbol = pos["symbol"]
            price = ticker_prices.get(symbol)
            if not price:
                continue

            pos["current_price"] = price
            pos["price_source"] = "ticker"
            pos["last_update_time"] = now

            entry = pos["entry_price"]
            direction = pos.get("direction", "")
            leverage = pos.get("leverage", 1)
            size = pos.get("size_usd", 0)

            if direction == "long":
                pnl_pct = (price - entry) / entry if entry > 0 else 0
            else:
                pnl_pct = (entry - price) / entry if entry > 0 else 0

            leveraged_pnl_pct = pnl_pct * leverage
            pnl_usd = size * leveraged_pnl_pct
            pos["unrealized_pnl"] = round(pnl_usd, 2)
            pos["unrealized_pnl_pct"] = round(leveraged_pnl_pct * 100, 2)

            sl = pos.get("stop_loss", 0)
            tp = pos.get("take_profit", 0)
            close_reason = None

            if direction == "long":
                if sl > 0 and price <= sl:
                    close_reason = "SL tetiklendi"
                elif tp > 0 and price >= tp:
                    close_reason = "TP tetiklendi"
            elif direction == "short":
                if sl > 0 and price >= sl:
                    close_reason = "SL tetiklendi"
                elif tp > 0 and price <= tp:
                    close_reason = "TP tetiklendi"

            if close_reason:
                close_price = price
                if direction == "long":
                    final_pnl_pct = (close_price - entry) / entry if entry > 0 else 0
                else:
                    final_pnl_pct = (entry - close_price) / entry if entry > 0 else 0

                final_leveraged = final_pnl_pct * leverage
                final_usd = size * final_leveraged

                pos["unrealized_pnl"] = round(final_usd, 2)
                pos["unrealized_pnl_pct"] = round(final_leveraged * 100, 2)
                pos["current_price"] = close_price
                pos["close_price"] = close_price
                pos["close_time"] = time.time()
                pos["close_reason"] = close_reason
                pos["pnl"] = round(final_usd, 2)
                pos["pnl_pct"] = round(final_leveraged * 100, 2)

                self._close_position(pos, close_reason)
                self._log(f"SL/TP TETIK: {symbol} {close_reason} @${close_price}")

        self._persist_state()

    async def _broadcast_positions_only(self):
        """Sadece pozisyon verisini broadcast et (hizli guncelleme icin)."""
        message = json.dumps({
            "type": "position_update",
            "positions": self.open_positions,
            "trade_history": self.trade_history[-20:],
            "portfolio": {
                "total_equity": self.total_equity,
                "available_capital": self.available_capital,
                "position_count": len(self.open_positions),
                "total_exposure_usd": sum(p.get("size_usd", 0) for p in self.open_positions),
                "total_unrealized_pnl": sum(p.get("unrealized_pnl", 0) for p in self.open_positions),
            },
            "timestamp": time.time(),
        }, default=str)
        disconnected = []
        for ws in self.websocket_clients:
            try:
                await ws.send_text(message)
            except Exception:
                disconnected.append(ws)
        for ws in disconnected:
            self.websocket_clients.remove(ws)

    def get_status(self) -> dict:
        total_pnl = sum(p.get("unrealized_pnl", 0) for p in self.open_positions)
        return {
            "running": self.running,
            "cycle_count": self.cycle_count,
            "last_cycle_time": self.last_cycle_time,
            "total_equity": self.total_equity,
            "available_capital": self.available_capital,
            "open_positions": len(self.open_positions),
            "trade_count": self.trade_count,
            "total_unrealized_pnl": round(total_pnl, 2),
            "closed_trades": len(self.trade_history),
            "winning_trades": len([t for t in self.trade_history if t.get("pnl", 0) > 0]),
            "losing_trades": len([t for t in self.trade_history if t.get("pnl", 0) <= 0]),
            "agents": {
                "scanner": self.scanner.get_status(),
                "technical": self.technical.get_status(),
                "sentiment": self.sentiment.get_status(),
                "risk": self.risk.get_status(),
                "portfolio": self.portfolio.get_status(),
                "patron": self.patron.get_status(),
            },
        }

    def get_latest_results(self) -> dict:
        return self.latest_results

    def get_open_positions(self) -> list[dict]:
        return self.open_positions

    def get_trade_history(self) -> list[dict]:
        return self.trade_history[-50:]
