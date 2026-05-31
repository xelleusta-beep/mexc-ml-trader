"""
MEXC ML Trading System — Risk Manager V2.0
=============================================
Gelismis Risk Yonetimi: Dynamic Sizing, Portfolio Risk, Adaptive Drawdown

OZELLIKLER:
  1. Dynamic Position Sizing: Volatilite ve confidence bazli
  2. Portfolio-Level Risk: Coklu pair risk optimizasyonu
  3. Adaptive Drawdown: Piyasa durumuna gore limit ayarlama
  4. Risk Metrics: VaR, CVaR, Expected Shortfall
  5. Smart Circuit Breaker: Akilli devre kesici
"""

import time
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# RISK METRICS
# ══════════════════════════════════════════════════════════════════════════════

class RiskMetrics:
    """
    Risk metrikleri hesaplama:
    - VaR (Value at Risk)
    - CVaR (Conditional VaR)
    - Expected Shortfall
    - Maximum Drawdown
    - Sharpe Ratio
    """

    @staticmethod
    def var_95(returns: np.ndarray) -> float:
        """VaR %95: Gerceklesme olasiliginin %5'i altindaki kayip."""
        if len(returns) < 20:
            return 0.0
        return float(np.percentile(returns, 5))

    @staticmethod
    def cvar_95(returns: np.ndarray) -> float:
        """CVaR %95: VaR'in altindaki beklenen kayip."""
        if len(returns) < 20:
            return 0.0
        var = np.percentile(returns, 5)
        tail_returns = returns[returns <= var]
        if len(tail_returns) == 0:
            return var
        return float(np.mean(tail_returns))

    @staticmethod
    def expected_shortfall(returns: np.ndarray, alpha: float = 0.05) -> float:
        """Expected Shortfall: Belirli bir alpha icin beklenen kayip."""
        if len(returns) < 20:
            return 0.0
        sorted_returns = np.sort(returns)
        n_tail = max(1, int(len(sorted_returns) * alpha))
        return float(np.mean(sorted_returns[:n_tail]))

    @staticmethod
    def max_drawdown(equity_curve: np.ndarray) -> float:
        """Maksimum drawdown."""
        if len(equity_curve) < 2:
            return 0.0
        peaks = np.maximum.accumulate(equity_curve)
        dd = (peaks - equity_curve) / (peaks + 1e-10)
        return float(np.max(dd))

    @staticmethod
    def sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Yillik Sharpe Ratio."""
        if len(returns) < 3:
            return 0.0
        excess_returns = returns - risk_free_rate / 252
        if np.std(excess_returns) < 1e-10:
            return 0.0
        return float(np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252))

    @staticmethod
    def sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Yillik Sortino Ratio (sadece downside risk)."""
        if len(returns) < 3:
            return 0.0
        excess_returns = returns - risk_free_rate / 252
        downside = excess_returns[excess_returns < 0]
        if len(downside) < 2 or np.std(downside) < 1e-10:
            return 0.0
        return float(np.mean(excess_returns) / np.std(downside) * np.sqrt(252))

    @staticmethod
    def calmar_ratio(total_return: float, max_drawdown: float) -> float:
        """Calmar Ratio: Return / Max Drawdown."""
        if max_drawdown < 0.001:
            return 0.0
        return float(total_return / max_drawdown)


# ══════════════════════════════════════════════════════════════════════════════
# DYNAMIC POSITION SIZER
# ══════════════════════════════════════════════════════════════════════════════

class DynamicPositionSizer:
    """
    Dinamik pozisyon boyutlandirma:
    - Volatilite-adjusted sizing
    - Confidence-based sizing
    - Kelly dynamic fraction
    - Portfolio heat map
    """

    def __init__(self, base_fraction: float = 0.02,
                 max_fraction: float = 0.05,
                 min_fraction: float = 0.005):
        self.base_fraction = base_fraction
        self.max_fraction = max_fraction
        self.min_fraction = min_fraction
        self._trade_history = deque(maxlen=100)
        self._win_rate = 0.5
        self._avg_win = 0.0
        self._avg_loss = 0.0

    def record_trade(self, pnl: float, size: float):
        """Trade sonucunu kaydet."""
        self._trade_history.append({"pnl": pnl, "size": size})
        self._update_stats()

    def _update_stats(self):
        """Istatistikleri guncelle."""
        if len(self._trade_history) < 5:
            return

        wins = [t["pnl"] for t in self._trade_history if t["pnl"] > 0]
        losses = [t["pnl"] for t in self._trade_history if t["pnl"] <= 0]

        if wins:
            self._avg_win = float(np.mean(wins))
        if losses:
            self._avg_loss = float(np.mean([abs(l) for l in losses]))

        total = len(self._trade_history)
        if total > 0:
            self._win_rate = len(wins) / total

    def calculate_kelly_fraction(self) -> float:
        """Kelly Criterion ile optimal bahis fraksiyonu."""
        if self._avg_loss <= 0 or self._win_rate <= 0:
            return self.base_fraction

        b = self._avg_win / self._avg_loss  # Win/Loss ratio
        p = self._win_rate
        q = 1.0 - p

        kelly = (p * b - q) / b
        kelly = max(0.0, kelly)  # Negatif kelly olmaz

        # Fraksiyonel Kelly (risk icin yari indirim)
        kelly *= 0.5

        return float(np.clip(kelly, self.min_fraction, self.max_fraction))

    def size_position(self, capital: float, confidence: float,
                      volatility: float = 0.5,
                      market_regime: str = "neutral") -> float:
        """
        Pozisyon boyutu hesapla.

        Parametreler:
          capital: Toplam sermaye
          confidence: Model guveni (0-100)
          volatility: Piyasa volatilitesi (0-1)
          market_regime: "bull", "bear", "neutral"

        Donus: Pozisyon degeri (USD)
        """
        # 1. Kelly-based fraction
        kelly_frac = self.calculate_kelly_fraction()

        # 2. Confidence adjustment (0.5x - 1.5x)
        conf_mult = 0.5 + (confidence / 100.0)
        conf_mult = min(conf_mult, 1.5)

        # 3. Volatility adjustment (yuksek vol = kucuk pozisyon)
        vol_mult = 1.5 - volatility
        vol_mult = max(0.5, min(vol_mult, 1.5))

        # 4. Market regime adjustment
        regime_mult = {"bull": 1.2, "neutral": 1.0, "bear": 0.8}.get(market_regime, 1.0)

        # 5. Final fraction
        final_fraction = kelly_frac * conf_mult * vol_mult * regime_mult
        final_fraction = float(np.clip(final_fraction, self.min_fraction, self.max_fraction))

        # 6. Pozisyon degeri
        position_value = capital * final_fraction

        return round(position_value, 2)

    def get_stats(self) -> dict:
        return {
            "win_rate": round(self._win_rate * 100, 1),
            "avg_win": round(self._avg_win, 2),
            "avg_loss": round(self._avg_loss, 2),
            "kelly_fraction": round(self.calculate_kelly_fraction(), 4),
            "trade_count": len(self._trade_history),
        }


# ══════════════════════════════════════════════════════════════════════════════
# PORTFOLIO RISK MANAGER
# ══════════════════════════════════════════════════════════════════════════════

class PortfolioRiskManager:
    """
    Portfolio-level risk yonetimi:
    - Correlation-based risk
    - Sector exposure
    - Tail risk hedging
    - Max portfolio VaR
    """

    # Kripto sektor gruplari
    SECTOR_MAP = {
        "BTC_USDT": "store_of_value",
        "ETH_USDT": "smart_contract",
        "SOL_USDT": "smart_contract",
        "BNB_USDT": "exchange",
        "XRP_USDT": "payment",
        "ADA_USDT": "smart_contract",
        "DOGE_USDT": "meme",
        "DOT_USDT": "interop",
        "LINK_USDT": "oracle",
        "AVAX_USDT": "smart_contract",
        "MATIC_USDT": "layer2",
        "UNI_USDT": "defi",
        "AAVE_USDT": "defi",
        "ATOM_USDT": "interop",
        "LTC_USDT": "payment",
    }

    def __init__(self, max_sector_exposure: float = 0.30,
                 max_correlation_risk: float = 0.50,
                 max_portfolio_var: float = 0.05):
        self.max_sector_exposure = max_sector_exposure
        self.max_correlation_risk = max_correlation_risk
        self.max_portfolio_var = max_portfolio_var
        self._returns_history = {}

    def update_returns(self, symbol: str, returns: np.ndarray):
        """Varlik getiri gecmisini guncelle."""
        self._returns_history[symbol] = returns

    def calculate_correlation_matrix(self) -> Optional[np.ndarray]:
        """Korelasyon matrisi hesapla."""
        symbols = list(self._returns_history.keys())
        if len(symbols) < 2:
            return None

        returns_list = []
        for sym in symbols:
            ret = self._returns_history.get(sym, np.array([]))
            if len(ret) > 20:
                returns_list.append(ret[-20:])
            else:
                returns_list.append(np.zeros(20))

        returns_arr = np.array(returns_list)
        if returns_arr.shape[1] < 5:
            return None

        corr_matrix = np.corrcoef(returns_arr)
        return corr_matrix

    def check_sector_exposure(self, positions: Dict[str, float],
                              portfolio_value: float) -> Dict[str, float]:
        """
        Sektorel maruz kontrolu.

        Returns: dict with sector_name -> exposure_ratio
        """
        sector_exposure = {}

        for symbol, pos_value in positions.items():
            sector = self.SECTOR_MAP.get(symbol, "other")
            sector_exposure[sector] = sector_exposure.get(sector, 0) + pos_value

        # Normalize
        if portfolio_value > 0:
            for sector in sector_exposure:
                sector_exposure[sector] /= portfolio_value

        return sector_exposure

    def check_concentration_risk(self, positions: Dict[str, float],
                                  portfolio_value: float) -> Tuple[bool, str]:
        """
        Konsantrasyon risk kontrolu.
        Returns: (allowed, reason)
        """
        if portfolio_value <= 0:
            return False, "Sermaye yetersiz"

        # Tek varlik max %25
        for symbol, pos_value in positions.items():
            ratio = pos_value / portfolio_value
            if ratio > 0.25:
                return False, f"{symbol} konsantrasyon limiti asildi ({ratio:.1%})"

        # Sektorel max %40
        sector_exp = self.check_sector_exposure(positions, portfolio_value)
        for sector, exp in sector_exp.items():
            if exp > 0.40:
                return False, f"{sector} sektor limiti asildi ({exp:.1%})"

        return True, ""

    def calculate_portfolio_var(self, positions: Dict[str, float],
                                returns_history: Dict[str, np.ndarray],
                                confidence: float = 0.95) -> float:
        """
        Portfolio VaR hesapla (Parametric method).
        """
        if not positions or not returns_history:
            return 0.0

        symbols = list(positions.keys())
        weights = np.array([positions[s] for s in symbols])
        total_weight = weights.sum()
        if total_weight <= 0:
            return 0.0
        weights = weights / total_weight

        # Return matrix
        returns_list = []
        for sym in symbols:
            ret = returns_history.get(sym, np.array([0.0]))
            if len(ret) > 0:
                returns_list.append(float(np.mean(ret[-20:])))
            else:
                returns_list.append(0.0)

        returns_arr = np.array(returns_list)

        # Portfolio return ve vol
        portfolio_return = np.dot(weights, returns_arr)

        # Covariance matrix
        cov_data = []
        for sym in symbols:
            ret = returns_history.get(sym, np.zeros(20))
            if len(ret) >= 20:
                cov_data.append(ret[-20:])
            else:
                cov_data.append(np.zeros(20))

        cov_matrix = np.cov(cov_data)
        portfolio_var = np.dot(weights.T, np.dot(cov_matrix, weights))
        portfolio_vol = np.sqrt(portfolio_var)

        # VaR
        from scipy import stats
        z_score = stats.norm.ppf(1 - confidence)
        var = portfolio_return + z_score * portfolio_vol

        return float(abs(var))


# ══════════════════════════════════════════════════════════════════════════════
# ADAPTIVE DRAWDOWN MANAGER
# ══════════════════════════════════════════════════════════════════════════════

class AdaptiveDrawdownManager:
    """
    Adaptif drawdown yonetimi:
    - Dynamic limits (piyasa durumuna gore)
    - Smart circuit breaker
    - Recovery mode
    """

    def __init__(self, base_drawdown_limit: float = 0.15,
                 min_drawdown_limit: float = 0.08,
                 max_drawdown_limit: float = 0.25):
        self.base_drawdown_limit = base_drawdown_limit
        self.min_drawdown_limit = min_drawdown_limit
        self.max_drawdown_limit = max_drawdown_limit

        self._peak_capital = 0.0
        self._current_drawdown = 0.0
        self._drawdown_history = deque(maxlen=100)
        self._circuit_breaker_until = 0.0
        self._recovery_mode = False
        self._consecutive_losses = 0

    def update_capital(self, capital: float):
        """Sermaye guncelle."""
        if capital > self._peak_capital:
            self._peak_capital = capital
            self._current_drawdown = 0.0
            self._recovery_mode = False
        elif self._peak_capital > 0:
            self._current_drawdown = (self._peak_capital - capital) / self._peak_capital

    def get_dynamic_limit(self, market_volatility: float = 0.5,
                          recent_performance: float = 0.0) -> float:
        """
        Dinamik drawdown limiti.

        Parametreler:
          market_volatility: Piyasa volatilitesi (0-1)
          recent_performance: Son performans (-1 ile 1 arasi)

        Donus: Drawdown limiti
        """
        limit = self.base_drawdown_limit

        # Yuksek volatilite → daha yuksek limit
        if market_volatility > 0.7:
            limit *= 1.2
        elif market_volatility < 0.3:
            limit *= 0.8

        # Kötü performans → daha dusuk limit
        if recent_performance < -0.1:
            limit *= 0.9
        elif recent_performance > 0.1:
            limit *= 1.1

        # Recovery mode → daha dusuk limit
        if self._recovery_mode:
            limit *= 0.7

        return float(np.clip(limit, self.min_drawdown_limit, self.max_drawdown_limit))

    def is_drawdown_exceeded(self, market_volatility: float = 0.5,
                             recent_performance: float = 0.0) -> Tuple[bool, float, float]:
        """
        Drawdown limit asildi mi?

        Returns: (exceeded, current, limit)
        """
        limit = self.get_dynamic_limit(market_volatility, recent_performance)
        return (
            self._current_drawdown >= limit,
            self._current_drawdown,
            limit,
        )

    def check_circuit_breaker(self, n_consecutive_losses: int,
                               loss_amount: float) -> bool:
        """
        Circuit breaker kontrolu.

        Parametreler:
          n_consecutive_losses: Ardışık zarar sayisi
          loss_amount: Toplam zarar miktari

        Returns: Circuit breaker aktif mi?
        """
        if time.time() < self._circuit_breaker_until:
            return True

        # Cok fazla ardışık zarar
        if n_consecutive_losses >= 5:
            cooldown_minutes = min(n_consecutive_losses * 15, 120)
            self._circuit_breaker_until = time.time() + cooldown_minutes * 60
            self._consecutive_losses = n_consecutive_losses
            logger.warning(f"Circuit breaker: {n_consecutive_losses} ardışık zarar, "
                          f"{cooldown_minutes}dk bekleme")
            return True

        # Buyuk zarar
        if loss_amount < -0.05:  # %5'ten fazla zarar
            self._circuit_breaker_until = time.time() + 30 * 60  # 30dk bekleme
            logger.warning(f"Circuit breaker: Buyuk zarar ({loss_amount:.1%}), 30dk bekleme")
            return True

        return False

    def enter_recovery_mode(self):
        """Recovery moduna gir."""
        self._recovery_mode = True
        logger.info("Recovery modu baslatildi - kucuk pozisyonlar oncelikli")

    def exit_recovery_mode(self):
        """Recovery modundan cik."""
        self._recovery_mode = False
        logger.info("Recovery modu sona erdi")

    def get_status(self) -> dict:
        return {
            "current_drawdown": round(self._current_drawdown * 100, 2),
            "peak_capital": round(self._peak_capital, 2),
            "dynamic_limit": round(self.get_dynamic_limit() * 100, 2),
            "recovery_mode": self._recovery_mode,
            "circuit_breaker_active": time.time() < self._circuit_breaker_until,
            "consecutive_losses": self._consecutive_losses,
        }


# ══════════════════════════════════════════════════════════════════════════════
# ANA RISK MANAGER V2
# ══════════════════════════════════════════════════════════════════════════════

class RiskManagerV2:
    """
    Risk Yonetimi V2 - Ana sinif:
    - Dynamic Position Sizing
    - Portfolio Risk
    - Adaptive Drawdown
    - Risk Metrics
    """

    def __init__(self, config=None):
        self.cfg = config.risk if config else None

        # Alt moduller
        self.position_sizer = DynamicPositionSizer()
        self.portfolio_risk = PortfolioRiskManager()
        self.drawdown_mgr = AdaptiveDrawdownManager()
        self.risk_metrics = RiskMetrics()

        # Durum
        self._trades = deque(maxlen=200)
        self._daily_pnl = 0.0
        self._daily_reset_time = time.time()
        self._peak_capital = 0.0

    def check_entry_allowed(self, symbol: str, signal: str,
                            confidence: float, price: float,
                            portfolio: dict, agent_states: dict,
                            ml_engine=None) -> Tuple[bool, str]:
        """
        Giris izni kontrolu V2.

        Returns: (allowed, reason)
        """
        # 1. Circuit breaker
        if self.drawdown_mgr._circuit_breaker_until > time.time():
            return False, "Circuit breaker aktif"

        # 2. Drawdown limit
        dd_exceeded, dd_curr, dd_limit = self.drawdown_mgr.is_drawdown_exceeded()
        if dd_exceeded:
            return False, f"Drawdown limiti asildi ({dd_curr:.1%} > {dd_limit:.1%})"

        # 3. Konsantrasyon riski
        positions = {}
        for sym, st in agent_states.items():
            pos = st.get("active_pos")
            if pos:
                positions[sym] = pos.get("size", 0) * max(1, pos.get("leverage", 1))

        conc_ok, conc_reason = self.portfolio_risk.check_concentration_risk(
            positions, portfolio.get("capital", 0)
        )
        if not conc_ok:
            return False, conc_reason

        # 4. WF koruma
        if ml_engine is not None:
            wf_acc = ml_engine._wf.get("accuracy", 0)
            if self.cfg and wf_acc < self.cfg.wf_leverage_lock_threshold:
                if confidence < 70:
                    return False, f"WF dusuk ({wf_acc:.0f}), yetersiz guven ({confidence:.0f})"

        return True, ""

    def calculate_position_size(self, symbol: str, price: float,
                                confidence: float, portfolio: dict,
                                agent_states: dict,
                                market_data: dict = None) -> float:
        """
        Dinamik pozisyon boyutu hesapla V2.
        """
        capital = portfolio.get("capital", 0)
        if capital <= 0:
            return 5.0  # Minimum

        # Volatilite tahmini
        volatility = 0.5
        if market_data:
            volatility = market_data.get("volatility", 0.5)

        # Piyasa rejimi
        regime = "neutral"
        if market_data:
            regime = market_data.get("regime", "neutral")

        # Dynamic sizing
        position_value = self.position_sizer.size_position(
            capital=capital,
            confidence=confidence,
            volatility=volatility,
            market_regime=regime
        )

        return position_value

    def record_trade_result(self, pnl: float, symbol: str = "", side: str = ""):
        """Trade sonucunu kaydet."""
        self._trades.append({"pnl": pnl, "symbol": symbol, "side": side})
        self.position_sizer.record_trade(pnl, 0)

        # Drawdown guncelle
        self.drawdown_mgr.update_capital(pnl)

        # Circuit breaker kontrolu
        recent_trades = list(self._trades)[-10:]
        consecutive = 0
        total_loss = 0.0
        for t in reversed(recent_trades):
            if t["pnl"] <= 0:
                consecutive += 1
                total_loss += t["pnl"]
            else:
                break

        if consecutive >= 3:
            self.drawdown_mgr.check_circuit_breaker(consecutive, total_loss)

    def get_status(self) -> dict:
        """Tam risk durumu."""
        return {
            "drawdown": self.drawdown_mgr.get_status(),
            "position_sizer": self.position_sizer.get_stats(),
            "total_trades": len(self._trades),
            "recent_pnl": sum(t["pnl"] for t in list(self._trades)[-10:]),
        }

    def load_state(self, data: dict):
        """Durum yukle."""
        if not data:
            return
        self._peak_capital = data.get("peak_capital", 0.0)
        self.drawdown_mgr._peak_capital = self._peak_capital

    def get_save_state(self) -> dict:
        """Durum kaydet."""
        return {
            "peak_capital": self._peak_capital,
            "drawdown_status": self.drawdown_mgr.get_status(),
        }


# ══════════════════════════════════════════════════════════════════════════════
# EXPORT
# ══════════════════════════════════════════════════════════════════════════════

__all__ = [
    "RiskManagerV2",
    "DynamicPositionSizer",
    "PortfolioRiskManager",
    "AdaptiveDrawdownManager",
    "RiskMetrics",
]
