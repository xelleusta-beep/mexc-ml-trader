from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Trade:
    """Tek bir alım-satım işlemini temsil eder."""
    type: str
    price: float
    amount_usd: float
    coin_amount: float
    timestamp: int
    reason: str
    fee: float = 0.0
    realized_pnl: float = 0.0


@dataclass
class Position:
    """Açık pozisyonu temsil eder (birleşik, ortalama fiyatlı)."""
    coin_amount: float = 0.0
    total_invested: float = 0.0
    first_entry_price: float = 0.0
    avg_price: float = 0.0
    max_price: float = 0.0
    min_price: float = float('inf')

    dca_30_done: bool = False
    dca_60_done: bool = False
    exit_1_done: bool = False
    exit_2_done: bool = False
    exit_3_done: bool = False

    realized_pnl: float = 0.0
    total_fees: float = 0.0
    total_buy_usd: float = 0.0
    total_sell_usd: float = 0.0
    total_entry_invested: float = 0.0
    entry_timestamp: Optional[int] = None
    last_trade_timestamp: Optional[int] = None

    max_unrealized_pnl: float = 0.0
    min_unrealized_pnl: float = 0.0
    highest_value_since_entry: float = 0.0

    trades: list = field(default_factory=list)

    @property
    def is_open(self) -> bool:
        return self.coin_amount > 0

    def _update_price_bounds(self, price: float):
        if price <= 0:
            return
        if self.max_price <= 0 or price > self.max_price:
            self.max_price = price
        if self.min_price == float('inf') or price < self.min_price:
            self.min_price = price

    def _record_trade(
        self,
        trade_type: str,
        price: float,
        amount_usd: float,
        coin_amount: float,
        timestamp: int,
        reason: str,
        fee: float = 0.0,
        realized_pnl: float = 0.0,
    ):
        self.trades.append(Trade(
            type=trade_type,
            price=price,
            amount_usd=amount_usd,
            coin_amount=coin_amount,
            timestamp=timestamp,
            reason=reason,
            fee=fee,
            realized_pnl=realized_pnl,
        ))
        self.last_trade_timestamp = timestamp
        self.total_fees += fee

    def enter(self, price: float, amount_usd: float, timestamp: int, fee: float = 0.0):
        """İlk giriş işlemi."""
        if price <= 0 or amount_usd <= 0:
            return

        coins = amount_usd / price
        self.coin_amount = coins
        self.total_invested = amount_usd
        self.first_entry_price = price
        self.avg_price = price
        self.max_price = price
        self.min_price = price
        self.dca_30_done = False
        self.dca_60_done = False
        self.exit_1_done = False
        self.exit_2_done = False
        self.exit_3_done = False
        self.realized_pnl = 0.0
        self.total_fees = fee
        self.total_buy_usd = amount_usd
        self.total_sell_usd = 0.0
        self.total_entry_invested = amount_usd
        self.entry_timestamp = timestamp
        self.last_trade_timestamp = timestamp
        self.max_unrealized_pnl = 0.0
        self.min_unrealized_pnl = 0.0
        self.highest_value_since_entry = amount_usd

        self._record_trade("buy", price, amount_usd, coins, timestamp, "entry", fee)

    def add_dca(self, price: float, amount_usd: float, timestamp: int, dca_type: str, fee: float = 0.0):
        """DCA ile pozisyona ekleme yapar."""
        if price <= 0 or amount_usd <= 0:
            return

        coins = amount_usd / price
        self.coin_amount += coins
        self.total_invested += amount_usd
        self.total_buy_usd += amount_usd
        self.total_entry_invested += amount_usd
        self.avg_price = self.total_invested / self.coin_amount if self.coin_amount > 0 else 0.0
        self._update_price_bounds(price)

        if dca_type == "dca_30":
            self.dca_30_done = True
        elif dca_type == "dca_60":
            self.dca_60_done = True

        self._record_trade("buy", price, amount_usd, coins, timestamp, dca_type, fee)

    def sell(self, price: float, pct: float, timestamp: int, reason: str, fee: float = 0.0) -> float:
        """
        Pozisyondan yüzde olarak satış yapar.
        Satılan brüt dolar miktarını döndürür.
        """
        if self.coin_amount <= 0 or price <= 0:
            return 0.0

        pct = max(0.0, min(100.0, pct))
        coins_to_sell = self.coin_amount * (pct / 100.0)
        if coins_to_sell <= 0:
            return 0.0

        cost_basis_per_coin = self.total_invested / self.coin_amount if self.coin_amount > 0 else 0.0
        sell_cost_basis = cost_basis_per_coin * coins_to_sell
        usd_received = coins_to_sell * price
        trade_realized_pnl = usd_received - sell_cost_basis - fee

        self.coin_amount -= coins_to_sell
        self.total_invested -= sell_cost_basis
        self.total_sell_usd += usd_received
        self.realized_pnl += trade_realized_pnl
        self.avg_price = self.total_invested / self.coin_amount if self.coin_amount > 0 else 0.0

        if reason == "sell_25":
            self.exit_1_done = True
        elif reason == "sell_50":
            self.exit_2_done = True
        elif reason in ("sell_70", "trend_exit", "force_exit", "backtest_end", "stop_loss", "trailing_stop"):
            self.exit_3_done = True

        self._record_trade("sell", price, usd_received, coins_to_sell, timestamp, reason, fee, trade_realized_pnl)
        self._update_price_bounds(price)
        return usd_received

    def sell_all(self, price: float, timestamp: int, reason: str = "sell_all", fee: float = 0.0) -> float:
        """Tüm pozisyonu satar."""
        return self.sell(price, 100.0, timestamp, reason, fee)

    def record_price_update(self, current_price: float):
        """Güncel fiyata göre unrealized PnL ve tepe/çukur değerlerini günceller."""
        if not self.is_open or current_price <= 0:
            return

        value = self.get_value(current_price)
        unrealized = value - self.total_invested
        self.max_unrealized_pnl = max(self.max_unrealized_pnl, unrealized)
        self.min_unrealized_pnl = min(self.min_unrealized_pnl, unrealized)
        self.highest_value_since_entry = max(self.highest_value_since_entry, value)
        self._update_price_bounds(current_price)

    def get_value(self, current_price: float) -> float:
        """Pozisyonun güncel piyasa değeri."""
        return self.coin_amount * current_price if current_price > 0 else 0.0

    @property
    def unrealized_pnl(self) -> float:
        """Güncel fiyata göre gerçekleşmemiş kâr/zarar."""
        return self.get_value(self.avg_price) - self.total_invested if self.avg_price > 0 else 0.0

    @property
    def exit_pct_sold(self) -> float:
        total = 0.0
        if self.exit_1_done:
            total += 25.0
        if self.exit_2_done:
            total += 25.0
        if self.exit_3_done:
            total += 50.0
        return total
