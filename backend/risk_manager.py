"""
MEXC ML Trading System — Risk Manager v1.0
Portfoy bazinda risk yonetimi, Kelly Criterion pozisyon buyuklugu,
drawdown limiti, circuit breaker, konsantrasyon kontrolu.
"""

import time
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Basit korrelasyon grubu tanimi (pair -> grup)
CORRELATION_GROUPS: Dict[str, str] = {
    "BTC_USDT":  "btc_eth",
    "ETH_USDT":  "btc_eth",
    "SOL_USDT":  "sol",
    "BNB_USDT":  "bnb",
    "XRP_USDT":  "xrp",
    "ADA_USDT":  "ada",
    "DOGE_USDT": "doge",
    "DOT_USDT":  "dot",
    "LINK_USDT": "link",
}


@dataclass
class TradeRecord:
    symbol: str
    side: str
    pnl: float
    timestamp: float = field(default_factory=time.time)


class RiskManager:
    """
    Profesyonel risk yonetim sistemi.

    Ozellikler:
      - Drawdown takibi ve limiti
      - Gunluk kayip limiti
      - Kelly Criterion pozisyon buyuklugu
      - Konsantrasyon limiti (pair ve grup bazinda)
      - Circuit breaker (ardisik zarar)
      - Trade kayitlari
    """

    def __init__(self, config):
        self.cfg = config.risk
        self.trading_cfg = config.trading

        # Portfoy takibi
        self._peak_capital: float = 0.0
        self._current_drawdown: float = 0.0

        # Gunluk PnL
        self._daily_pnl: float = 0.0
        self._daily_trades: int = 0
        self._daily_reset_time: float = time.time()

        # Ardısık zarar
        self._consecutive_losses: int = 0
        self._circuit_breaker_until: float = 0.0

        # Trade kayitlari (son 200)
        self._trades: deque = deque(maxlen=200)

        # Kazanma kaybi istatistikleri (Kelly icin)
        self._wins: List[float] = []
        self._losses: List[float] = []
        self._max_losses: deque = deque(maxlen=20)  # son 20 trade

    # ── GUNLUK RESET ──────────────────────────────────────────────────────

    def _check_daily_reset(self):
        """Her gun basi gunluk PnL'i sifirla."""
        now = time.time()
        # 24 saat gecmis mi kontrol et
        if now - self._daily_reset_time > 86400:
            self._daily_pnl = 0.0
            self._daily_trades = 0
            self._daily_reset_time = now
            logger.info("Risk: Gunluk PnL sifirlandi")

    # ── DRAWDOWN ──────────────────────────────────────────────────────────

    def update_capital(self, current_capital: float):
        """Anlik sermayeyi guncelle, drawdown hesapla."""
        if current_capital > self._peak_capital:
            self._peak_capital = current_capital
            self._current_drawdown = 0.0
        elif self._peak_capital > 0:
            self._current_drawdown = (self._peak_capital - current_capital) / self._peak_capital

    def get_drawdown(self) -> float:
        return self._current_drawdown

    def is_drawdown_exceeded(self) -> Tuple[bool, float, float]:
        """Drawdown limit asildi mi? (asildi, current, limit)"""
        return (
            self._current_drawdown >= self.cfg.max_drawdown,
            self._current_drawdown,
            self.cfg.max_drawdown,
        )

    # ── GUNLUK KAYIP ──────────────────────────────────────────────────────

    def is_daily_loss_exceeded(self) -> Tuple[bool, float, float]:
        """Gunluk kayip limiti asildi mi? (asildi, current, limit)"""
        self._check_daily_reset()
        daily_loss_pct = abs(self._daily_pnl) / max(self._peak_capital, 1)
        return (
            daily_loss_pct >= self.cfg.daily_loss_limit,
            daily_loss_pct,
            self.cfg.daily_loss_limit,
        )

    # ── CIRCUIT BREAKER ──────────────────────────────────────────────────

    def is_circuit_breaker_active(self) -> bool:
        """Devre kesici aktif mi? (ardisik zarar limiti asildiysa)"""
        if time.time() < self._circuit_breaker_until:
            remaining = int(self._circuit_breaker_until - time.time())
            logger.warning(f"Risk: Circuit breaker aktif — {remaining}s kaldi")
            return True
        return False

    # ── POZISYON BUYUKLUGU (KELLY CRITERION) ──────────────────────────────

    def calculate_position_size(
        self,
        symbol: str,
        price: float,
        confidence: float,
        portfolio: dict,
        agent_states: dict,
    ) -> float:
        """
        Kelly Criterion ile pozisyon buyuklugu hesapla.

        Kelly % = (p * b - q) / b
          p = kazanma olasiligi (win_rate)
          q = kaybetme olasiligi (1 - p)
          b = kazanc/kayip orani (avg_win / avg_loss)

        Nihai buyukluk = capital * kelly_fraction * Kelly%
        """
        capital = portfolio.get("capital", 10000.0)
        if capital <= 0:
            return self.trading_cfg.base_position_size

        # Win rate ve avg win/loss
        win_rate, avg_win, avg_loss = self._get_kelly_params()

        # Yeterli veri yoksa varsayilan kullan
        if win_rate <= 0 or avg_loss <= 0:
            return self.trading_cfg.base_position_size

        # Kelly hesapla
        b = avg_win / avg_loss if avg_loss > 0 else 1.0
        p = win_rate
        q = 1.0 - p
        kelly_pct = max(0.0, (p * b - q) / b)

        # Fraksiyonel Kelly (guvenli)
        kelly_pct *= self.cfg.kelly_fraction

        # Konfidansa gore ayarla (0.5x - 1.5x)
        conf_multiplier = 0.5 + (confidence / 100.0)
        kelly_pct *= min(conf_multiplier, 1.5)

        # Konsantrasyon limiti uygula
        pair_exposure = self._get_pair_exposure(symbol, agent_states)
        max_exposure = capital * self.cfg.max_concentration_per_pair
        available = max(0.0, max_exposure - pair_exposure)

        position_value = capital * kelly_pct
        position_value = min(position_value, available)

        # Base size'dan az olmasin
        position_value = max(position_value, self.trading_cfg.base_position_size)

        return round(position_value, 2)

    def _get_kelly_params(self) -> Tuple[float, float, float]:
        """Win rate, avg win, avg loss hesapla (son 50 trade)."""
        if len(self._trades) < 5:
            return 0.0, 0.0, 0.0

        recent = list(self._trades)[-50:]
        wins = [t.pnl for t in recent if t.pnl > 0]
        losses = [t.pnl for t in recent if t.pnl <= 0]

        if not wins or not losses:
            return 0.0, 0.0, 0.0

        win_rate = len(wins) / len(recent)
        avg_win = float(np.mean(wins))
        avg_loss = float(np.mean([abs(l) for l in losses]))
        return win_rate, avg_win, max(avg_loss, 0.01)

    # ── KONSANTRASYON KONTROLU ────────────────────────────────────────────

    def _get_pair_exposure(self, symbol: str, agent_states: dict) -> float:
        """Belirli bir pair'in toplam pozisyon buyuklugunu hesapla."""
        st = agent_states.get(symbol, {})
        pos = st.get("active_pos")
        if pos:
            return pos.get("size", 0) * max(1, pos.get("leverage", 1))
        return 0.0

    def check_concentration(
        self, symbol: str, desired_size: float, portfolio: dict, agent_states: dict
    ) -> Tuple[bool, str]:
        """
        Konsantrasyon limiti kontrolu.
        - Pair bazinda: max %20
        - Grup bazinda (BTC+ETH): max %35
        """
        capital = portfolio.get("capital", 10000.0)
        if capital <= 0:
            return False, "Sermaye yetersiz"

        # Pair bazli
        pair_exp = self._get_pair_exposure(symbol, agent_states) + desired_size
        max_pair = capital * self.cfg.max_concentration_per_pair
        if pair_exp > max_pair:
            return False, f"{symbol} konsantrasyon limiti asildi ({pair_exp:.0f} > {max_pair:.0f})"

        # Grup bazli (korrelasyon)
        group = CORRELATION_GROUPS.get(symbol)
        if group:
            group_exp = desired_size
            for sym, st in agent_states.items():
                if sym != symbol and CORRELATION_GROUPS.get(sym) == group:
                    pos = st.get("active_pos")
                    if pos:
                        group_exp += pos.get("size", 0) * max(1, pos.get("leverage", 1))
            max_group = capital * self.cfg.max_concentration_per_pair * 1.5  # grup limiti %30
            if group_exp > max_group:
                return False, f"Korrelasyon grubu {group} limiti asildi"

        return True, ""

    # ── GIRIS KONTROLU ────────────────────────────────────────────────────

    def check_entry_allowed(
        self,
        symbol: str,
        signal: str,
        confidence: float,
        price: float,
        portfolio: dict,
        agent_states: dict,
        ml_engine=None,
    ) -> Tuple[bool, str]:
        """
        Giris izni kontrolu — tum risk kontrollerini tek noktada toplar.
        Returns: (allowed: bool, reason: str)
        """
        # 1. Circuit breaker
        if self.is_circuit_breaker_active():
            return False, "Circuit breaker aktif"

        # 2. Drawdown
        dd_exceeded, dd_curr, dd_limit = self.is_drawdown_exceeded()
        if dd_exceeded:
            return False, f"Drawdown limiti asildi (%{dd_curr*100:.1f} > %{dd_limit*100:.1f})"

        # 3. Gunluk kayip
        dl_exceeded, dl_curr, dl_limit = self.is_daily_loss_exceeded()
        if dl_exceeded:
            return False, f"Gunluk kayip limiti asildi (%{dl_curr*100:.1f} > %{dl_limit*100:.1f})"

        # 4. WF koruma
        if ml_engine is not None:
            wf_acc = ml_engine._wf.get("accuracy", 0)
            if wf_acc < self.cfg.wf_leverage_lock_threshold and confidence < 70:
                return False, f"WF dusuk (%{wf_acc:.0f}), yetersiz guven (%{confidence:.0f})"

        # 5. Konsantrasyon
        base_size = self.calculate_position_size(
            symbol, price, confidence, portfolio, agent_states
        )
        conc_ok, conc_reason = self.check_concentration(
            symbol, base_size, portfolio, agent_states
        )
        if not conc_ok:
            return False, conc_reason

        return True, ""

    # ── TRADE KAYDI ────────────────────────────────────────────────────────

    def record_trade_result(self, pnl: float, symbol: str = "", side: str = ""):
        """Trade sonucunu kaydet ve risk istatistiklerini guncelle."""
        self._check_daily_reset()

        trade = TradeRecord(symbol=symbol, side=side, pnl=pnl)
        self._trades.append(trade)
        self._daily_pnl += pnl
        self._daily_trades += 1

        if pnl > 0:
            self._wins.append(pnl)
            self._consecutive_losses = 0
        else:
            self._losses.append(pnl)
            self._consecutive_losses += 1
            self._max_losses.append(pnl)

            # Circuit breaker kontrolu
            if self._consecutive_losses >= self.cfg.consecutive_loss_limit:
                cool_min = int(self._consecutive_losses * 15)  # her kayip icin +15dk
                self._circuit_breaker_until = time.time() + cool_min * 60
                logger.warning(
                    f"Risk: Circuit breaker tetiklendi! "
                    f"{self._consecutive_losses} ardısık zarar, "
                    f"{cool_min}dk bekleme"
                )

    # ── DURUM RAPORU ──────────────────────────────────────────────────────

    def get_status(self) -> dict:
        """Risk yonetim durumunu doner (API/frontend icin)."""
        self._check_daily_reset()
        dd_exceeded, dd_curr, dd_limit = self.is_drawdown_exceeded()
        dl_exceeded, dl_curr, dl_limit = self.is_daily_loss_exceeded()

        win_rate, avg_win, avg_loss = self._get_kelly_params()

        return {
            "drawdown_pct":          round(self._current_drawdown * 100, 2),
            "drawdown_limit_pct":    round(self.cfg.max_drawdown * 100, 1),
            "drawdown_exceeded":     dd_exceeded,
            "daily_pnl":             round(self._daily_pnl, 2),
            "daily_loss_pct":        round(dl_curr * 100, 2),
            "daily_loss_limit_pct":  round(self.cfg.daily_loss_limit * 100, 1),
            "daily_loss_exceeded":   dl_exceeded,
            "consecutive_losses":    self._consecutive_losses,
            "circuit_breaker_active": self.is_circuit_breaker_active(),
            "circuit_breaker_until": self._circuit_breaker_until,
            "kelly_win_rate":        round(win_rate * 100, 1),
            "kelly_avg_win":         round(avg_win, 2),
            "kelly_avg_loss":        round(avg_loss, 2),
            "total_trades_tracked":  len(self._trades),
        }

    def load_state(self, data: dict):
        """Kaydedilmis risk durumunu yukle."""
        if not data:
            return
        self._peak_capital = data.get("peak_capital", 0.0)
        self._current_drawdown = data.get("current_drawdown", 0.0)
        self._consecutive_losses = data.get("consecutive_losses", 0)
        self._daily_pnl = data.get("daily_pnl", 0.0)
        self._daily_reset_time = data.get("daily_reset_time", time.time())

    def get_save_state(self) -> dict:
        """Kaydedilecek risk durumu."""
        return {
            "peak_capital":       self._peak_capital,
            "current_drawdown":   self._current_drawdown,
            "consecutive_losses": self._consecutive_losses,
            "daily_pnl":          self._daily_pnl,
            "daily_reset_time":   self._daily_reset_time,
        }
