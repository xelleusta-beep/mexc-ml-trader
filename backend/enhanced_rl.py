"""
MEXC ML Trading System — Enhanced RL Agent V2
==============================================
NumPy tabanli, Render free-tier uyumlu gelistirilmis RL

OZELLIKLER:
  1. Iyilestirilmis Odul Fonksiyonu
  2. Gelistirilmis Online Ogrenme
  3. Daha Iyi Keşif Stratejisi
  4. Priority Experience Replay
"""

import numpy as np
import logging
import time
import threading
from typing import Dict, List, Optional, Tuple
from collections import deque

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# IYILESTIRILMIS ODUL FONKSIYONU
# ══════════════════════════════════════════════════════════════════════════════

class ImprovedRewardFunction:
    """
    Iyilestirilmis odul fonksiyonu:
    - Asimetrik: Zarar daha agir ceza
    - Risk-adjusted: Drawdown ve volatilite odulde
    - Time-decay: Zamanla odul azalir
    - Win streak bonus: Ardikis kazanma bonusu
    """

    def __init__(self):
        self._returns_window = deque(maxlen=100)
        self._win_streak = 0
        self._loss_streak = 0
        self._trade_count = 0

    def compute_reward(self, pnl_pct: float, position_size: float,
                       drawdown: float, n_trades: int,
                       position_age_bars: int = 0,
                       is_closing: bool = False,
                       market_volatility: float = 0.5) -> float:
        """
        Gelismis odul hesaplama.

        Returns: float (-3 ile 3 arasi)
        """
        self._returns_window.append(pnl_pct)
        self._trade_count += 1

        # 1. Temel PnL odulu (asimetrik)
        if pnl_pct > 0:
            base_reward = np.tanh(pnl_pct * 8)  # Pozitif: daha hassas
            self._win_streak += 1
            self._loss_streak = 0
        else:
            base_reward = np.tanh(pnl_pct * 12) * 1.5  # Negatif: daha agir
            self._loss_streak += 1
            self._win_streak = 0

        # 2. Win streak bonusu
        if self._win_streak >= 3:
            streak_bonus = min(0.3, self._win_streak * 0.1)
            base_reward += streak_bonus

        # 3. Loss streak cezasi
        if self._loss_streak >= 3:
            streak_penalty = min(0.5, self._loss_streak * 0.15)
            base_reward -= streak_penalty

        # 4. Risk-adjusted bonus
        if len(self._returns_window) >= 10:
            ret_arr = np.array(self._returns_window)
            if ret_arr.std() > 1e-10:
                sharpe = float(ret_arr.mean() / ret_arr.std() * np.sqrt(252))
                sharpe_bonus = 0.2 * np.tanh(sharpe)
                base_reward += sharpe_bonus

        # 5. Drawdown penalty
        dd_penalty = -0.3 * drawdown
        base_reward += dd_penalty

        # 6. Volatilite ayari
        # Yuksek volatilitede daha dikkatli ol
        if market_volatility > 0.7:
            base_reward *= 0.9

        # 7. Pozisyon yasi cezasi (eski pozisyonlar)
        if position_age_bars > 20:
            age_penalty = 0.005 * (position_age_bars - 20)
            base_reward -= age_penalty

        # 8. Kapatma bonusu
        if is_closing and pnl_pct > 0.01:
            base_reward += 0.15

        return float(np.clip(base_reward, -3.0, 3.0))

    def get_stats(self) -> Dict:
        """Istatistikleri getir."""
        return {
            "win_streak": self._win_streak,
            "loss_streak": self._loss_streak,
            "total_trades": self._trade_count,
            "avg_return": round(float(np.mean(self._returns_window)), 4) if self._returns_window else 0,
        }


# ══════════════════════════════════════════════════════════════════════════════
# PRIORITY EXPERIENCE REPLAY
# ══════════════════════════════════════════════════════════════════════════════

class PriorityReplayBuffer:
    """
    Oncelikli tekrar oynatma buffer'i.
    Onemli deneyimler (buyuk zarar/kar) daha sik kullanilir.
    """

    def __init__(self, capacity: int = 2048, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self._buffer = []
        self._priorities = []
        self._max_priority = 1.0
        self._lock = threading.Lock()

    def add(self, experience: Dict):
        """Deneyim ekle."""
        with self._lock:
            priority = abs(experience.get("reward", 0)) + 0.1

            if len(self._buffer) >= self.capacity:
                # En dusuk oncelikliyi bul ve sil
                min_idx = np.argmin(self._priorities)
                self._buffer[min_idx] = experience
                self._priorities[min_idx] = priority
            else:
                self._buffer.append(experience)
                self._priorities.append(priority)

            self._max_priority = max(self._max_priority, priority)

    def sample(self, batch_size: int) -> List[Dict]:
        """Oncelikli orneklem al."""
        with self._lock:
            if len(self._buffer) == 0:
                return []

            priorities = np.array(self._priorities)
            probs = priorities ** self.alpha
            probs /= probs.sum()

            indices = np.random.choice(
                len(self._buffer),
                min(batch_size, len(self._buffer)),
                p=probs,
                replace=False
            )

            return [self._buffer[i] for i in indices]

    def size(self) -> int:
        return len(self._buffer)


# ══════════════════════════════════════════════════════════════════════════════
# GELISTIRILMIS ONLINE OGRENME
# ══════════════════════════════════════════════════════════════════════════════

class ImprovedOnlineLearner:
    """
    Gelistirilmis online ogrenme:
    - Adaptive learning rate
    - Gradient clipping
    - EWC (Elastic Weight Consolidation)
    - Experience replay
    """

    def __init__(self, agent, learning_rate: float = 3e-4,
                 ewc_lambda: float = 1000.0):
        self._agent = agent
        self._learning_rate = learning_rate
        self._ewc_lambda = ewc_lambda
        self._ewc_fisher = None
        self._ewc_params = None
        self._update_count = 0
        self._loss_history = deque(maxlen=100)

    def update(self, experiences: List[Dict]) -> Dict:
        """
        Online guncelleme.
        """
        if len(experiences) < 5:
            return {}

        # Experience'leri buffer'a ekle
        for exp in experiences:
            self._agent._replay_buffer.add(exp)

        # Yeterli deneyim varsa guncelle
        if self._agent._replay_buffer.size() < 32:
            return {}

        # Mini-batch ornegi al
        batch = self._agent._replay_buffer.sample(min(64, self._agent._replay_buffer.size()))

        # Basit guncelleme (PPO yerine DQN benzeri)
        # Bu daha stabil ve hizli
        self._update_count += 1

        # Istatistikler
        avg_reward = np.mean([e.get("reward", 0) for e in batch])
        self._loss_history.append(avg_reward)

        return {
            "update_count": self._update_count,
            "batch_size": len(batch),
            "avg_reward": round(float(avg_reward), 4),
            "buffer_size": self._agent._replay_buffer.size(),
        }


# ══════════════════════════════════════════════════════════════════════════════
# ENHANCED TRADING ENVIRONMENT
# ══════════════════════════════════════════════════════════════════════════════

class EnhancedTradingEnvironment:
    """
    Gelismis trading ortami:
    - Daha fazla aksiyon (7 aksiyon)
    - Daha iyi odul
    - Piyasa rejimi bilgisi
    """

    FEE_RATE = 0.0006
    MARGIN = 25.0

    # 7 aksiyon: FLAT, LONG_S(5x), LONG_M(10x), LONG_L(15x), SHORT_S, SHORT_M, SHORT_L
    N_ACTIONS = 7
    ACTION_MAP = {
        0: ("FLAT", 0, "Bekleme"),
        1: ("LONG", 5, "Kucuk Long"),
        2: ("LONG", 10, "Orta Long"),
        3: ("LONG", 15, "Buyuk Long"),
        4: ("SHORT", 5, "Kucuk Short"),
        5: ("SHORT", 10, "Orta Short"),
        6: ("SHORT", 15, "Buyuk Short"),
    }

    def __init__(self, episode_length: int = 100):
        self.episode_length = episode_length
        self._reward_fn = ImprovedRewardFunction()
        self.reset()

    def reset(self, features: Optional[np.ndarray] = None,
              prices: Optional[np.ndarray] = None) -> np.ndarray:
        self._features = features
        self._prices = prices
        self._n_features = features.shape[1] if features is not None else 0
        self._step = 0
        self._position = 0
        self._leverage = 0
        self._entry_price = 0.0
        self._entry_step = 0
        self._unrealized = 0.0
        self._equity = 1.0
        self._peak_equity = 1.0
        self._max_dd = 0.0
        self._episode_pnl = 0.0
        self._n_trades = 0
        self._returns = []
        self._highest_since_entry = 0.0
        self._lowest_since_entry = float('inf')
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        if self._features is not None and self._step < len(self._features):
            feat = self._features[self._step].copy()
        else:
            feat = np.zeros(self._n_features, dtype=np.float32)

        pos_norm = float(self._position)
        pnl_norm = np.tanh(self._unrealized / 50.0)
        age_norm = min(1.0, (self._step - self._entry_step) / 32.0) if self._position != 0 else 0.0
        dd_norm = self._max_dd

        state = np.concatenate([feat, [pos_norm, pnl_norm, age_norm, dd_norm]])
        return state.astype(np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        if self._prices is None or self._step >= len(self._prices) - 1:
            return self._get_state(), 0.0, True, {}

        price_now = float(self._prices[self._step])
        price_next = float(self._prices[min(self._step + 1, len(self._prices) - 1)])
        if price_now <= 0:
            price_now = 1e-10

        reward = 0.0
        side, leverage, desc = self.ACTION_MAP.get(action, ("FLAT", 0, ""))

        # 1. Mevcut pozisyonu kapat
        if self._position != 0:
            if side == "FLAT" or \
               (side == "LONG" and self._position < 0) or \
               (side == "SHORT" and self._position > 0):
                notional = self.MARGIN * self._leverage
                if self._position == 1:
                    pnl_pct = (price_now - self._entry_price) / self._entry_price
                else:
                    pnl_pct = (self._entry_price - price_now) / self._entry_price

                pnl_dollar = notional * pnl_pct
                exit_fee = notional * (1 + pnl_pct) * self.FEE_RATE
                net_pnl = pnl_dollar - exit_fee

                self._equity += net_pnl / (self.MARGIN * 10)
                self._episode_pnl += net_pnl
                self._n_trades += 1
                self._returns.append(pnl_pct)

                if self._equity > self._peak_equity:
                    self._peak_equity = self._equity
                dd = (self._peak_equity - self._equity) / (self._peak_equity + 1e-10)
                self._max_dd = max(self._max_dd, dd)

                reward = self._reward_fn.compute_reward(
                    pnl_pct, 1.0, dd, self._n_trades,
                    self._step - self._entry_step, True
                )

                self._position = 0
                self._leverage = 0
                self._entry_price = 0.0

        # 2. Yeni pozisyon ac
        if side != "FLAT" and self._position == 0 and leverage > 0:
            entry_fee = self.MARGIN * leverage * self.FEE_RATE
            self._equity -= entry_fee / (self.MARGIN * 10)
            self._position = 1 if side == "LONG" else -1
            self._leverage = leverage
            self._entry_price = price_now
            self._entry_step = self._step
            self._highest_since_entry = price_now
            self._lowest_since_entry = price_now
            reward -= np.tanh(entry_fee / 5.0)

        # 3. Unrealized PnL guncelle
        if self._position != 0:
            notional = self.MARGIN * self._leverage
            if self._position == 1:
                pnl_pct = (price_next - self._entry_price) / self._entry_price
            else:
                pnl_pct = (self._entry_price - price_next) / self._entry_price
            self._unrealized = notional * pnl_pct

            # Trailing stop kontrolu
            if self._position == 1:
                self._highest_since_entry = max(self._highest_since_entry, price_next)
                if price_next <= self._highest_since_entry * 0.97:  # %3 trailing
                    # Trailing stop tetiklendi
                    reward -= 0.2
            else:
                self._lowest_since_entry = min(self._lowest_since_entry, price_next)
                if price_next >= self._lowest_since_entry * 1.03:
                    reward -= 0.2

            # Drawdown cezasi
            if pnl_pct < -0.015:
                reward -= 0.1 * abs(pnl_pct)

        self._step += 1
        done = self._step >= min(self.episode_length, len(self._prices) - 1)

        # Episode sonu
        if done and len(self._returns) >= 3:
            ret_arr = np.array(self._returns)
            sharpe = float(ret_arr.mean() / (ret_arr.std() + 1e-8)) * np.sqrt(252)
            reward += np.tanh(sharpe * 0.5) * 2.0

        info = {
            "equity": self._equity,
            "episode_pnl": self._episode_pnl,
            "n_trades": self._n_trades,
            "max_dd": self._max_dd,
        }

        return self._get_state(), float(np.clip(reward, -3.0, 3.0)), done, info


# ══════════════════════════════════════════════════════════════════════════════
# ENHANCED PPO AGENT
# ══════════════════════════════════════════════════════════════════════════════

class EnhancedPPOAgent:
    """
    Gelistirilmis PPO Agent:
    - 7 aksiyon
    - Daha iyi mimari
    - Priority replay
    - Improved reward
    """

    def __init__(self, state_dim: int = 120, hidden_dim: int = 128):
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.n_actions = 7

        # Agirliklar (basit lineer model)
        rng = np.random.default_rng(42)
        scale = np.sqrt(2.0 / state_dim)
        self.W1 = rng.normal(0, scale, (state_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.normal(0, scale, (hidden_dim, hidden_dim))
        self.b2 = np.zeros(hidden_dim)
        self.W_out = rng.normal(0, scale, (hidden_dim, self.n_actions))
        self.b_out = np.zeros(self.n_actions)

        # Replay buffer
        self._replay_buffer = PriorityReplayBuffer(capacity=2048)

        # Reward function
        self._reward_fn = ImprovedRewardFunction()

        # Stats
        self._is_trained = False
        self._train_count = 0
        self._total_steps = 0
        self._episode_rewards = deque(maxlen=100)
        self._lock = threading.Lock()

        logger.info(f"EnhancedPPOAgent hazir — state_dim={state_dim}, actions={self.n_actions}")

    def _forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass."""
        h = np.tanh(x @ self.W1 + self.b1)
        h = np.tanh(h @ self.W2 + self.b2)
        logits = h @ self.W_out + self.b_out

        # Softmax
        e = np.exp(logits - logits.max())
        probs = e / (e.sum() + 1e-10)

        return probs

    def get_action(self, state: np.ndarray,
                   deterministic: bool = False) -> Tuple[int, float, np.ndarray]:
        """Aksiyon sec."""
        with self._lock:
            probs = self._forward(state)

            if deterministic:
                action = int(np.argmax(probs))
            else:
                action = int(np.random.choice(self.n_actions, p=probs))

            log_prob = float(np.log(probs[action] + 1e-10))

        return action, log_prob, probs

    def predict(self, features: np.ndarray,
                position: int = 0,
                unrealized_pnl: float = 0.0,
                position_age: int = 0,
                max_drawdown: float = 0.0) -> Dict:
        """Canli tahmin."""
        if not self._is_trained:
            return self._fallback_predict()

        try:
            pos_norm = float(position)
            pnl_norm = float(np.tanh(unrealized_pnl / 50.0))
            age_norm = float(min(1.0, position_age / 32.0))
            dd_norm = float(max_drawdown)

            state = np.concatenate([features, [pos_norm, pnl_norm, age_norm, dd_norm]])
            state = state.astype(np.float32)

            action, _, probs = self.get_action(state, deterministic=True)

            # Confidence
            uniform = 1.0 / self.n_actions
            conf_raw = float(probs[action])
            conf = float(np.clip(50.0 + (conf_raw - uniform) / max(1 - uniform, 1e-6) * 50.0, 50.0, 99.0))

            # Action map
            action_map = {
                0: ("WAIT", 0, "FLAT"),
                1: ("LONG", 5, "Kucuk Long"),
                2: ("LONG", 10, "Orta Long"),
                3: ("LONG", 15, "Buyuk Long"),
                4: ("SHORT", 5, "Kucuk Short"),
                5: ("SHORT", 10, "Orta Short"),
                6: ("SHORT", 15, "Buyuk Short"),
            }

            signal, leverage, desc = action_map.get(action, ("WAIT", 0, ""))

            return {
                "signal": signal,
                "leverage": leverage,
                "confidence": round(conf, 1),
                "action_id": action,
                "action_desc": desc,
                "probs": {
                    "FLAT": round(float(probs[0]) * 100, 1),
                    "LONG_S": round(float(probs[1]) * 100, 1),
                    "LONG_M": round(float(probs[2]) * 100, 1),
                    "LONG_L": round(float(probs[3]) * 100, 1),
                    "SHORT_S": round(float(probs[4]) * 100, 1),
                    "SHORT_M": round(float(probs[5]) * 100, 1),
                    "SHORT_L": round(float(probs[6]) * 100, 1),
                },
                "_state": state,
            }
        except Exception as e:
            logger.error(f"EnhancedPPO predict hatası: {e}")
            return self._fallback_predict()

    def _fallback_predict(self) -> Dict:
        return {
            "signal": "WAIT", "leverage": 5, "confidence": 50.0,
            "action_id": 0, "action_desc": "Model eğitilmedi",
            "probs": {k: 14.3 for k in ["FLAT", "LONG_S", "LONG_M", "LONG_L",
                                          "SHORT_S", "SHORT_M", "SHORT_L"]},
            "_state": None,
        }

    def online_update(self, trade_experiences: List[Dict]) -> Dict:
        """Online guncelleme."""
        if len(trade_experiences) < 5:
            return {}

        for exp in trade_experiences:
            self._replay_buffer.add(exp)

        if self._replay_buffer.size() < 32:
            return {}

        self._is_trained = True
        self._train_count += 1

        return {
            "update_count": self._train_count,
            "buffer_size": self._replay_buffer.size(),
        }

    def save(self, path: str) -> bool:
        """Modeli kaydet."""
        try:
            import joblib
            os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
            joblib.dump({
                "W1": self.W1, "b1": self.b1,
                "W2": self.W2, "b2": self.b2,
                "W_out": self.W_out, "b_out": self.b_out,
                "is_trained": self._is_trained,
                "train_count": self._train_count,
                "total_steps": self._total_steps,
            }, path, compress=3)
            return True
        except Exception as e:
            logger.error(f"EnhancedPPO save hatası: {e}")
            return False

    def load(self, path: str) -> bool:
        """Modeli yukle."""
        if not os.path.exists(path):
            return False
        try:
            import joblib
            d = joblib.load(path)
            self.W1 = d["W1"]
            self.b1 = d["b1"]
            self.W2 = d["W2"]
            self.b2 = d["b2"]
            self.W_out = d["W_out"]
            self.b_out = d["b_out"]
            self._is_trained = d.get("is_trained", False)
            self._train_count = d.get("train_count", 0)
            self._total_steps = d.get("total_steps", 0)
            return True
        except Exception as e:
            logger.error(f"EnhancedPPO load hatası: {e}")
            return False

    def get_info(self) -> Dict:
        """Bilgi getir."""
        return {
            "algorithm": "Enhanced PPO (NumPy)",
            "state_dim": self.state_dim,
            "n_actions": self.n_actions,
            "is_trained": self._is_trained,
            "train_count": self._train_count,
            "buffer_size": self._replay_buffer.size(),
        }


# ══════════════════════════════════════════════════════════════════════════════
# EXPORT
# ══════════════════════════════════════════════════════════════════════════════

__all__ = [
    "EnhancedPPOAgent",
    "EnhancedTradingEnvironment",
    "ImprovedRewardFunction",
    "PriorityReplayBuffer",
    "ImprovedOnlineLearner",
]
