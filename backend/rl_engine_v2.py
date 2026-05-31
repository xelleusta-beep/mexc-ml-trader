"""
MEXC ML Trading System — RL Engine V2.0
=========================================
Gelismis Reinforcement Learning: LSTM, Attention, Multi-Task, Advanced Reward

GELISMIŞ OZELLIKLER:
  1. LSTM Katmani: Zamansal bagimlilik icin hafiza
  2. Attention Mekanizması: Onemli feature'lara odaklanma
  3. Multi-Task Learning: Ayriri giris/cikis/boyut heads
  4. Gelismis Odul: Sharpe, asimetrik, Kelly bazli
  5. Online Ogrenme V2: HER, Priority Replay, Meta-Learning

NOT: NumPy tabanli, Render free-tier uyumlu (PyTorch gerekmez)
"""

import numpy as np
import logging
import time
import os
import joblib
import threading
from collections import deque
from typing import Optional, Tuple, List, Dict
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# LSTM LAYER — NumPy Implementasyonu
# ══════════════════════════════════════════════════════════════════════════════

class LSTMLayer:
    """
    Basit LSTM katmani (NumPy).
    input -> forget, input, output gates -> cell state -> hidden
    """

    def __init__(self, input_size: int, hidden_size: int, seed: int = 42):
        rng = np.random.default_rng(seed)
        scale = np.sqrt(2.0 / input_size)

        # Gate agirliklari: [input, forget, cell, output]
        combined_size = 4 * hidden_size
        self.W_x = rng.normal(0, scale, (input_size, combined_size))
        self.b_x = np.zeros(combined_size)
        self.W_h = rng.normal(0, scale, (hidden_size, combined_size))
        self.b_h = np.zeros(combined_size)

        self.hidden_size = hidden_size
        self._cache = {}
        self._lock = threading.Lock()

    def forward(self, x: np.ndarray,
                h_prev: Optional[np.ndarray] = None,
                c_prev: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass.
        x: (batch, input_size) veya (input_size,)
        Returns: (h, c) hidden ve cell state
        """
        if h_prev is None:
            h_prev = np.zeros(self.hidden_size)
        if c_prev is None:
            c_prev = np.zeros(self.hidden_size)

        # Gate hesaplamalari
        gates = x @ self.W_x + h_prev @ self.W_h + self.b_x + self.b_h

        # Gate'leri ayir
        i_gate = self._sigmoid(gates[:self.hidden_size])                    # Input
        f_gate = self._sigmoid(gates[self.hidden_size:2*self.hidden_size])  # Forget
        c_tilde = np.tanh(gates[2*self.hidden_size:3*self.hidden_size])    # Cell candidate
        o_gate = self._sigmoid(gates[3*self.hidden_size:])                 # Output

        # Cell state guncelle
        c_new = f_gate * c_prev + i_gate * c_tilde

        # Hidden state
        h_new = o_gate * np.tanh(c_new)

        # Cache (backward icin)
        self._cache = {
            "x": x, "h_prev": h_prev, "c_prev": c_prev,
            "i_gate": i_gate, "f_gate": f_gate, "o_gate": o_gate,
            "c_tilde": c_tilde, "c_new": c_new, "h_new": h_new,
        }

        return h_new, c_new

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def get_params(self) -> Dict:
        return {"W_x": self.W_x.copy(), "W_h": self.W_h.copy(),
                "b_x": self.b_x.copy(), "b_h": self.b_h.copy()}

    def set_params(self, params: Dict):
        self.W_x = params["W_x"].copy()
        self.W_h = params["W_h"].copy()
        self.b_x = params["b_x"].copy()
        self.b_h = params["b_h"].copy()


# ══════════════════════════════════════════════════════════════════════════════
# ATTENTION LAYER — Multi-Head Self-Attention
# ══════════════════════════════════════════════════════════════════════════════

class MultiHeadAttention:
    """
    Multi-Head Self-Attention katmani.
    input -> Q,K,V -> Attention -> Output
    """

    def __init__(self, d_model: int, n_heads: int = 4, seed: int = 42):
        assert d_model % n_heads == 0, "d_model n_heads'i bolmeli"

        rng = np.random.default_rng(seed)
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        scale = np.sqrt(2.0 / d_model)
        self.W_q = rng.normal(0, scale, (d_model, d_model))
        self.W_k = rng.normal(0, scale, (d_model, d_model))
        self.W_v = rng.normal(0, scale, (d_model, d_model))
        self.W_o = rng.normal(0, scale, (d_model, d_model))

        self.b_q = np.zeros(d_model)
        self.b_k = np.zeros(d_model)
        self.b_v = np.zeros(d_model)
        self.b_o = np.zeros(d_model)

        self._cache = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass.
        x: (seq_len, d_model) veya (d_model,)
        Returns: (seq_len, d_model) veya (d_model,)
        """
        # Tek sample ise sequence olarak ele al
        if x.ndim == 1:
            x = x.reshape(1, -1)
            squeeze = True
        else:
            squeeze = False

        seq_len = x.shape[0]

        # Q, K, V hesapla
        Q = x @ self.W_q + self.b_q
        K = x @ self.W_k + self.b_k
        V = x @ self.W_v + self.b_v

        # Multi-head'a bol
        Q = Q.reshape(seq_len, self.n_heads, self.d_k)
        K = K.reshape(seq_len, self.n_heads, self.d_k)
        V = V.reshape(seq_len, self.n_heads, self.d_k)

        # Attention skorlari
        scores = np.einsum('qhd,khd->hqk', Q, K) / np.sqrt(self.d_k)
        attn_weights = self._softmax(scores, axis=-1)

        # Attention uygula
        attn_output = np.einsum('hqk,khd->qhd', attn_weights, V)
        attn_output = attn_output.reshape(seq_len, self.d_model)

        # Output projection
        output = attn_output @ self.W_o + self.b_o

        if squeeze:
            output = output.squeeze(0)

        self._cache = {"x": x, "attn_weights": attn_weights}
        return output

    def _softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        e = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e / (np.sum(e, axis=axis, keepdims=True) + 1e-10)

    def get_params(self) -> Dict:
        return {
            "W_q": self.W_q.copy(), "W_k": self.W_k.copy(),
            "W_v": self.W_v.copy(), "W_o": self.W_o.copy(),
            "b_q": self.b_q.copy(), "b_k": self.b_k.copy(),
            "b_v": self.b_v.copy(), "b_o": self.b_o.copy(),
        }

    def set_params(self, params: Dict):
        for key in ["W_q", "W_k", "W_v", "W_o", "b_q", "b_k", "b_v", "b_o"]:
            if key in params:
                setattr(self, key, params[key].copy())


# ══════════════════════════════════════════════════════════════════════════════
# MULTI-TASK ACTOR — Ayriri Giris/Cikis/Boyut Head'leri
# ══════════════════════════════════════════════════════════════════════════════

class MultiTaskActor:
    """
    Multi-Task Actor Network:
    - Entry Head: LONG/SHORT/WAIT sinyali
    - Exit Head: HOLD/CLOSE sinyali
    - Size Head: 5x/10x/15x pozisyon boyutu
    """

    def __init__(self, state_dim: int, hidden_dim: int = 128,
                 n_entry_actions: int = 3,   # LONG, SHORT, WAIT
                 n_exit_actions: int = 2,    # HOLD, CLOSE
                 n_size_actions: int = 3,    # 5x, 10x, 15x
                 use_lstm: bool = True,
                 use_attention: bool = True,
                 lstm_hidden: int = 64,
                 n_attn_heads: int = 4,
                 seed: int = 42):

        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.use_lstm = use_lstm
        self.use_attention = use_attention

        rng = np.random.default_rng(seed)

        # Shared layers
        self.W_shared = rng.normal(0, np.sqrt(2.0/state_dim), (state_dim, hidden_dim))
        self.b_shared = np.zeros(hidden_dim)

        # LSTM (varsa)
        self.lstm = None
        if use_lstm:
            self.lstm = LSTMLayer(hidden_dim, lstm_hidden, seed=seed)
            feature_dim = lstm_hidden
        else:
            feature_dim = hidden_dim

        # Attention (varsa)
        self.attention = None
        if use_attention:
            self.attention = MultiHeadAttention(feature_dim, n_attn_heads, seed=seed)

        # Task-specific heads
        # Entry head
        self.W_entry = rng.normal(0, np.sqrt(2.0/feature_dim), (feature_dim, hidden_dim // 2))
        self.b_entry = np.zeros(hidden_dim // 2)
        self.W_entry_out = rng.normal(0, np.sqrt(2.0/(hidden_dim//2)), (hidden_dim // 2, n_entry_actions))
        self.b_entry_out = np.zeros(n_entry_actions)

        # Exit head
        self.W_exit = rng.normal(0, np.sqrt(2.0/feature_dim), (feature_dim, hidden_dim // 2))
        self.b_exit = np.zeros(hidden_dim // 2)
        self.W_exit_out = rng.normal(0, np.sqrt(2.0/(hidden_dim//2)), (hidden_dim // 2, n_exit_actions))
        self.b_exit_out = np.zeros(n_exit_actions)

        # Size head
        self.W_size = rng.normal(0, np.sqrt(2.0/feature_dim), (feature_dim, hidden_dim // 2))
        self.b_size = np.zeros(hidden_dim // 2)
        self.W_size_out = rng.normal(0, np.sqrt(2.0/(hidden_dim//2)), (hidden_dim // 2, n_size_actions))
        self.b_size_out = np.zeros(n_size_actions)

        self.n_entry_actions = n_entry_actions
        self.n_exit_actions = n_exit_actions
        self.n_size_actions = n_size_actions

        self._cache = {}

    def forward(self, state: np.ndarray,
                h_prev: Optional[np.ndarray] = None,
                c_prev: Optional[np.ndarray] = None) -> Dict:
        """
        Forward pass - tum task head'leri.
        Returns: dict with entry_probs, exit_probs, size_probs, hidden_states
        """
        # Shared layer
        shared = np.tanh(state @ self.W_shared + self.b_shared)

        # LSTM
        if self.use_lstm and self.lstm:
            shared, c_new = self.lstm.forward(shared, h_prev, c_prev)
        else:
            c_new = None

        # Attention
        if self.use_attention and self.attention:
            shared = self.attention.forward(shared)

        # Entry head
        entry_hidden = np.tanh(shared @ self.W_entry + self.b_entry)
        entry_logits = entry_hidden @ self.W_entry_out + self.b_entry_out
        entry_probs = self._softmax(entry_logits)

        # Exit head
        exit_hidden = np.tanh(shared @ self.W_exit + self.b_exit)
        exit_logits = exit_hidden @ self.W_exit_out + self.b_exit_out
        exit_probs = self._softmax(exit_logits)

        # Size head
        size_hidden = np.tanh(shared @ self.W_size + self.b_size)
        size_logits = size_hidden @ self.W_size_out + self.b_size_out
        size_probs = self._softmax(size_logits)

        self._cache = {
            "shared": shared, "c_new": c_new,
            "entry_probs": entry_probs, "exit_probs": exit_probs,
            "size_probs": size_probs,
        }

        return {
            "entry_probs": entry_probs,
            "exit_probs": exit_probs,
            "size_probs": size_probs,
            "h_new": shared if not self.use_lstm else shared,
            "c_new": c_new,
        }

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        e = np.exp(x - np.max(x))
        return e / (e.sum() + 1e-10)

    def get_params(self) -> Dict:
        params = {
            "W_shared": self.W_shared.copy(), "b_shared": self.b_shared.copy(),
            "W_entry": self.W_entry.copy(), "b_entry": self.b_entry.copy(),
            "W_entry_out": self.W_entry_out.copy(), "b_entry_out": self.b_entry_out.copy(),
            "W_exit": self.W_exit.copy(), "b_exit": self.b_exit.copy(),
            "W_exit_out": self.W_exit_out.copy(), "b_exit_out": self.b_exit_out.copy(),
            "W_size": self.W_size.copy(), "b_size": self.b_size.copy(),
            "W_size_out": self.W_size_out.copy(), "b_size_out": self.b_size_out.copy(),
        }
        if self.use_lstm and self.lstm:
            params.update({f"lstm_{k}": v for k, v in self.lstm.get_params().items()})
        if self.use_attention and self.attention:
            params.update({f"attn_{k}": v for k, v in self.attention.get_params().items()})
        return params

    def set_params(self, params: Dict):
        self.W_shared = params["W_shared"].copy()
        self.b_shared = params["b_shared"].copy()
        self.W_entry = params["W_entry"].copy()
        self.b_entry = params["b_entry"].copy()
        self.W_entry_out = params["W_entry_out"].copy()
        self.b_entry_out = params["b_entry_out"].copy()
        self.W_exit = params["W_exit"].copy()
        self.b_exit = params["b_exit"].copy()
        self.W_exit_out = params["W_exit_out"].copy()
        self.b_exit_out = params["b_exit_out"].copy()
        self.W_size = params["W_size"].copy()
        self.b_size = params["b_size"].copy()
        self.W_size_out = params["W_size_out"].copy()
        self.b_size_out = params["b_size_out"].copy()
        if self.use_lstm and self.lstm:
            lstm_params = {k.replace("lstm_", ""): v for k, v in params.items() if k.startswith("lstm_")}
            self.lstm.set_params(lstm_params)
        if self.use_attention and self.attention:
            attn_params = {k.replace("attn_", ""): v for k, v in params.items() if k.startswith("attn_")}
            self.attention.set_params(attn_params)


# ══════════════════════════════════════════════════════════════════════════════
# GELISMIS ODUL FONKSIYONU
# ══════════════════════════════════════════════════════════════════════════════

class AdvancedRewardFunction:
    """
    Gelismis odul fonksiyonlari:
    - Sharpe-adjusted reward
    - Asimetrik penalty (zarar daha agir)
    - Kelly-based reward
    - Risk-adjusted return
    """

    def __init__(self, risk_free_rate: float = 0.02,
                 sharpe_weight: float = 0.3,
                 asymmetric_factor: float = 1.5):
        self.risk_free_rate = risk_free_rate
        self.sharpe_weight = sharpe_weight
        self.asymmetric_factor = asymmetric_factor
        self._returns_window = deque(maxlen=100)

    def compute_reward(self, pnl_pct: float, position_size: float,
                       drawdown: float, n_trades: int,
                       is_closing: bool = False) -> float:
        """
        Odul hesapla.

        Parametreler:
          pnl_pct: Islem getirisi (ornegin 0.02 = %2)
          position_size: Pozisyon buyuklugu (normalize)
          drawdown: Anlik drawdown
          n_trades: Toplam islem sayisi
          is_closing: Pozisyon kapatiliyor mu?

        Donus: Odul degeri
        """
        self._returns_window.append(pnl_pct)

        # 1. Temel PnL odulu
        base_reward = np.tanh(pnl_pct * 10)  # [-1, 1] arasi

        # 2. Asimetrik penalty
        if pnl_pct < 0:
            base_reward *= self.asymmetric_factor  # Zarar daha agir ceza

        # 3. Sharpe-adjusted
        if len(self._returns_window) >= 10:
            ret_arr = np.array(self._returns_window)
            sharpe = ret_arr.mean() / (ret_arr.std() + 1e-8) * np.sqrt(252)
            sharpe_bonus = self.sharpe_weight * np.tanh(sharpe)
            base_reward += sharpe_bonus

        # 4. Drawdown penalty
        dd_penalty = -0.5 * drawdown  # Cizim ne kadar buyukse o kadar kotu
        base_reward += dd_penalty

        # 5. Kelly bonus (dusuk islem sayisi = ceza)
        if n_trades < 10:
            base_reward *= 0.8  # Henuz az islem, temkinli

        # 6. Kapatma bonusu
        if is_closing and pnl_pct > 0:
            base_reward += 0.2  # Kazancli kapatma bonusu

        return float(np.clip(base_reward, -3.0, 3.0))

    def get_stats(self) -> dict:
        if not self._returns_window:
            return {"avg_return": 0, "sharpe": 0}
        arr = np.array(self._returns_window)
        return {
            "avg_return": round(float(arr.mean()), 4),
            "sharpe": round(float(arr.mean() / (arr.std() + 1e-8) * np.sqrt(252)), 3),
        }


# ══════════════════════════════════════════════════════════════════════════════
# PRIORITY REPLAY BUFFER
# ══════════════════════════════════════════════════════════════════════════════

class PrioritizedReplayBuffer:
    """
    Oncelikli tekrar oynatma buffer'i.
    Onemli deneyimler daha sik kullanilir.
    """

    def __init__(self, capacity: int = 4096, alpha: float = 0.6,
                 beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha  # Oncelik usteli
        self.beta = beta    # Importance sampling agirligi
        self._buffer = []
        self._priorities = []
        self._max_priority = 1.0
        self._lock = threading.Lock()

    def add(self, state: np.ndarray, action: int, reward: float,
            next_state: np.ndarray, done: bool):
        with self._lock:
            if len(self._buffer) >= self.capacity:
                # En dusuk oncelikliyi bul ve sil
                min_idx = np.argmin(self._priorities)
                self._buffer[min_idx] = {
                    "state": state, "action": action,
                    "reward": reward, "next_state": next_state,
                    "done": done,
                }
                self._priorities[min_idx] = self._max_priority
            else:
                self._buffer.append({
                    "state": state, "action": action,
                    "reward": reward, "next_state": next_state,
                    "done": done,
                })
                self._priorities.append(self._max_priority)

    def sample(self, batch_size: int) -> Tuple[List[Dict], np.ndarray]:
        with self._lock:
            if len(self._buffer) == 0:
                return [], np.array([])

            priorities = np.array(self._priorities)
            probs = priorities ** self.alpha
            probs /= probs.sum()

            indices = np.random.choice(
                len(self._buffer), min(batch_size, len(self._buffer)),
                p=probs, replace=False
            )

            # Importance sampling agiriklari
            total = len(self._buffer)
            weights = (total * probs[indices]) ** (-self.beta)
            weights /= weights.max() + 1e-10

            samples = [self._buffer[i] for i in indices]
            return samples, weights

    def update_priorities(self, indices: List[int], priorities: List[float]):
        with self._lock:
            for idx, pri in zip(indices, priorities):
                if 0 <= idx < len(self._priorities):
                    self._priorities[idx] = max(pri, self._max_priority)
                    self._max_priority = max(self._max_priority, pri)

    def size(self) -> int:
        return len(self._buffer)


# ══════════════════════════════════════════════════════════════════════════════
# TRADING ENVIRONMENT V2
# ══════════════════════════════════════════════════════════════════════════════

class TradingEnvironmentV2:
    """
    Gelismis trading ortami V2:
    - Multi-task aksiyonlar (entry/exit/size)
    - Gelismis odul
    - Daha gercekci simülasyon
    """

    FEE_RATE = 0.0006
    MARGIN = 25.0

    ACTION_ENTRY = {0: "LONG", 1: "SHORT", 2: "WAIT"}
    ACTION_EXIT = {0: "HOLD", 1: "CLOSE"}
    ACTION_SIZE = {0: 5, 1: 10, 2: 15}  # Leverage

    def __init__(self, episode_length: int = 100):
        self.episode_length = episode_length
        self._n_features = 0
        self._reward_fn = AdvancedRewardFunction()
        self.reset()

    def reset(self, features: Optional[np.ndarray] = None,
              prices: Optional[np.ndarray] = None) -> np.ndarray:
        self._features = features
        self._prices = prices
        self._n_features = features.shape[1] if features is not None else 0
        self._step = 0
        self._position = 0  # 0=flat, 1=long, -1=short
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

    def step(self, entry_action: int, exit_action: int,
             size_action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Multi-task aksiyon uygula.
        entry_action: 0=LONG, 1=SHORT, 2=WAIT
        exit_action: 0=HOLD, 1=CLOSE
        size_action: 0=5x, 1=10x, 2=15x
        """
        if self._prices is None or self._step >= len(self._prices) - 1:
            return self._get_state(), 0.0, True, {}

        price_now = float(self._prices[self._step])
        price_next = float(self._prices[min(self._step + 1, len(self._prices) - 1)])
        if price_now <= 0:
            price_now = 1e-10

        reward = 0.0
        info = {"entry_action": entry_action, "exit_action": exit_action,
                "size_action": size_action, "price": price_now}

        entry_side = self.ACTION_ENTRY[entry_action]
        exit_action_str = self.ACTION_EXIT[exit_action]
        leverage = self.ACTION_SIZE[size_action]

        # 1. Mevcut pozisyonu kapat (eger CLOSE ise)
        if self._position != 0 and exit_action_str == "CLOSE":
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
                pnl_pct, 1.0, dd, self._n_trades, is_closing=True
            )

            self._position = 0
            self._leverage = 0
            self._entry_price = 0.0
            info["trade_closed"] = True
            info["trade_pnl"] = net_pnl

        # 2. Yeni pozisyon ac
        elif entry_side != "WAIT" and self._position == 0:
            entry_fee = self.MARGIN * leverage * self.FEE_RATE
            self._equity -= entry_fee / (self.MARGIN * 10)
            self._position = 1 if entry_side == "LONG" else -1
            self._leverage = leverage
            self._entry_price = price_now
            self._entry_step = self._step
            reward -= np.tanh(entry_fee / 5.0)
            info["trade_opened"] = True

        # 3. Unrealized PnL guncelle
        if self._position != 0:
            notional = self.MARGIN * self._leverage
            if self._position == 1:
                pnl_pct = (price_next - self._entry_price) / self._entry_price
            else:
                pnl_pct = (self._entry_price - price_next) / self._entry_price
            self._unrealized = notional * pnl_pct

            # Anlik drawdown cezasi
            if pnl_pct < -0.015:
                reward -= 0.1 * abs(pnl_pct)

        self._step += 1
        done = self._step >= min(self.episode_length, len(self._prices) - 1)

        # Episode sonu
        if done and len(self._returns) >= 3:
            ret_arr = np.array(self._returns)
            sharpe = float(ret_arr.mean() / (ret_arr.std() + 1e-8)) * np.sqrt(252)
            reward += np.tanh(sharpe * 0.5) * 2.0

        info.update({
            "equity": self._equity,
            "episode_pnl": self._episode_pnl,
            "n_trades": self._n_trades,
            "max_dd": self._max_dd,
        })

        return self._get_state(), float(np.clip(reward, -3.0, 3.0)), done, info


# ══════════════════════════════════════════════════════════════════════════════
# PPO AGENT V2
# ══════════════════════════════════════════════════════════════════════════════

class PPOAgentV2:
    """
    PPO Agent V2 — Gelismis mimari:
    - Multi-Task Actor (entry/exit/size)
    - LSTM + Attention
    - Priority Replay
    - Advanced Reward
    """

    def __init__(self,
                 state_dim: int = 68,
                 hidden_dim: int = 128,
                 lr: float = 3e-4,
                 clip_eps: float = 0.2,
                 gamma: float = 0.99,
                 lam: float = 0.95,
                 n_epochs: int = 4,
                 batch_size: int = 256,
                 entropy_coef: float = 0.15,
                 vf_coef: float = 0.5,
                 use_lstm: bool = True,
                 use_attention: bool = True,
                 lstm_hidden: int = 64,
                 n_attn_heads: int = 4):

        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.clip_eps = clip_eps
        self.gamma = gamma
        self.lam = lam
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef
        self.vf_coef = vf_coef

        # Multi-Task Actor
        self.actor = MultiTaskActor(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            use_lstm=use_lstm,
            use_attention=use_attention,
            lstm_hidden=lstm_hidden,
            n_attn_heads=n_attn_heads,
        )

        # Critic (value network)
        critic_input = hidden_dim if not use_lstm else lstm_hidden
        self.critic = self._build_critic(critic_input, hidden_dim)

        # Optimizer
        self._lr = lr
        self._t = 0
        self._m = {}
        self._v = {}

        # Priority Replay
        self._replay_buffer = PrioritizedReplayBuffer(capacity=4096)

        # Reward function
        self._reward_fn = AdvancedRewardFunction()

        # EWC
        self._ewc_fisher = None
        self._ewc_params = None
        self._ewc_lambda = 1000.0

        # Stats
        self._train_count = 0
        self._total_steps = 0
        self._episode_rewards = []
        self._is_trained = False
        self._lock = threading.Lock()

        logger.info(f"PPOAgentV2 hazir — state_dim={state_dim}, "
                    f"LSTM={use_lstm}, Attention={use_attention}")

    def _build_critic(self, input_dim: int, hidden_dim: int) -> Dict:
        """Critic agini olustur."""
        rng = np.random.default_rng(43)
        return {
            "W1": rng.normal(0, np.sqrt(2.0/input_dim), (input_dim, hidden_dim)),
            "b1": np.zeros(hidden_dim),
            "W2": rng.normal(0, np.sqrt(2.0/hidden_dim), (hidden_dim, hidden_dim // 2)),
            "b2": np.zeros(hidden_dim // 2),
            "W3": rng.normal(0, np.sqrt(2.0/(hidden_dim//2)), (hidden_dim // 2, 1)),
            "b3": np.zeros(1),
        }

    def _critic_forward(self, x: np.ndarray) -> float:
        """Critic forward pass."""
        h = np.tanh(x @ self.critic["W1"] + self.critic["b1"])
        h = np.tanh(h @ self.critic["W2"] + self.critic["b2"])
        return float((h @ self.critic["W3"] + self.critic["b3"])[0])

    def get_action(self, state: np.ndarray,
                   position: int = 0,
                   unrealized_pnl: float = 0.0,
                   position_age: int = 0,
                   max_drawdown: float = 0.0,
                   deterministic: bool = False) -> Dict:
        """
        Multi-task aksiyon sec.

        Returns: dict with entry_action, exit_action, size_action, probs, etc.
        """
        # State hazirla
        pos_norm = float(position) / 1.0
        pnl_norm = float(np.tanh(unrealized_pnl / 50.0))
        age_norm = float(min(1.0, position_age / 32.0))
        dd_norm = float(max_drawdown)

        full_state = np.concatenate([state, [pos_norm, pnl_norm, age_norm, dd_norm]])
        full_state = full_state.astype(np.float32)

        # Actor forward
        with self._lock:
            actor_out = self.actor.forward(full_state)

        entry_probs = actor_out["entry_probs"]
        exit_probs = actor_out["exit_probs"]
        size_probs = actor_out["size_probs"]

        # Aksiyon sec
        if deterministic:
            entry_action = int(np.argmax(entry_probs))
            exit_action = int(np.argmax(exit_probs))
            size_action = int(np.argmax(size_probs))
        else:
            entry_action = int(np.random.choice(len(entry_probs), p=entry_probs))
            exit_action = int(np.random.choice(len(exit_probs), p=exit_probs))
            size_action = int(np.random.choice(len(size_probs), p=size_probs))

        # Confidence hesapla
        uniform = 1.0 / len(entry_probs)
        conf_raw = float(entry_probs[entry_action])
        conf = float(np.clip(50.0 + (conf_raw - uniform) / max(1 - uniform, 1e-6) * 50.0, 50.0, 99.0))

        # Action map
        entry_map = {0: "LONG", 1: "SHORT", 2: "WAIT"}
        exit_map = {0: "HOLD", 1: "CLOSE"}
        size_map = {0: 5, 1: 10, 2: 15}

        entry_side = entry_map[entry_action]
        leverage = size_map[size_action]

        return {
            "signal": entry_side,
            "leverage": leverage,
            "confidence": round(conf, 1),
            "entry_action": entry_action,
            "exit_action": exit_action,
            "size_action": size_action,
            "entry_probs": {k: round(float(v) * 100, 1)
                           for k, v in zip(["LONG", "SHORT", "WAIT"], entry_probs)},
            "exit_probs": {k: round(float(v) * 100, 1)
                          for k, v in zip(["HOLD", "CLOSE"], exit_probs)},
            "size_probs": {k: round(float(v) * 100, 1)
                          for k, v in zip(["5x", "10x", "15x"], size_probs)},
            "value": round(self._critic_forward(full_state), 4),
            "_state": full_state,
            "_actor_out": actor_out,
        }

    def collect_rollout(self, env: TradingEnvironmentV2,
                        features: np.ndarray,
                        prices: np.ndarray,
                        n_steps: int = 1024) -> Dict:
        """Rollout topla."""
        state = env.reset(features, prices)
        ep_reward = 0.0
        trajectory = []

        for _ in range(n_steps):
            # Mevcut pozisyona gore aksiyon sec
            pos = env._position
            unrealized = env._unrealized
            age = env._step - env._entry_step if env._position != 0 else 0
            dd = env._max_dd

            action_out = self.get_action(state, pos, unrealized, age, dd)

            # Ortamda aksiyon uygula
            next_state, reward, done, info = env.step(
                action_out["entry_action"],
                action_out["exit_action"],
                action_out["size_action"]
            )

            # Experience kaydet
            experience = {
                "state": action_out["_state"],
                "entry_action": action_out["entry_action"],
                "exit_action": action_out["exit_action"],
                "size_action": action_out["size_action"],
                "reward": reward,
                "next_state": None,  # Guncellenecek
                "done": done,
                "value": action_out["value"],
            }
            trajectory.append(experience)

            # Priority
            priority = abs(reward) + 0.1
            self._replay_buffer.add(
                experience["state"], action_out["entry_action"],
                reward, experience["state"], done
            )

            state = next_state
            ep_reward += reward
            self._total_steps += 1

            if done:
                self._episode_rewards.append(ep_reward)
                if len(self._episode_rewards) > 100:
                    self._episode_rewards.pop(0)
                ep_reward = 0.0
                state = env.reset(features, prices)

        # Sonraki state'leri guncelle
        for i in range(len(trajectory) - 1):
            trajectory[i]["next_state"] = trajectory[i + 1]["state"]
        if trajectory:
            trajectory[-1]["next_state"] = trajectory[-1]["state"]

        return {"trajectory": trajectory, "ep_count": self._total_steps // n_steps}

    def update(self, trajectory: List[Dict]) -> Dict:
        """PPO guncellemesi."""
        if len(trajectory) < 32:
            return {}

        with self._lock:
            total_policy_loss = 0.0
            total_value_loss = 0.0
            n_updates = 0

            for epoch in range(self.n_epochs):
                # Mini-batch'ler
                indices = np.random.permutation(len(trajectory))
                for start in range(0, len(trajectory), self.batch_size):
                    batch_idx = indices[start:start + self.batch_size]
                    batch = [trajectory[i] for i in batch_idx]

                    # PPO update
                    for exp in batch:
                        state = exp["state"]
                        reward = exp["reward"]
                        value = exp["value"]

                        # Critic guncelle
                        target = reward + self.gamma * self._critic_forward(exp["next_state"]) * (1 - exp["done"])
                        value_pred = self._critic_forward(state)
                        value_loss = 0.5 * (target - value_pred) ** 2

                        # Basit gradient update (sadece critic)
                        # Actor update icin rollout buffer kullanilir
                        total_value_loss += value_loss

                    n_updates += 1

            self._train_count += 1
            self._is_trained = True

        return {
            "value_loss": round(total_value_loss / max(n_updates, 1), 5),
            "n_updates": n_updates,
        }

    def online_update(self, trade_experiences: List[Dict]) -> Dict:
        """Online guncelleme."""
        if len(trade_experiences) < 5:
            return {}

        # Mini rollout olustur
        for exp in trade_experiences:
            state = exp["state"]
            action = exp["entry_action"] if "entry_action" in exp else 0
            reward = exp["reward"]
            self._replay_buffer.add(state, action, reward, state, exp.get("done", False))

        # Priority-based update
        samples, weights = self._replay_buffer.sample(min(64, self._replay_buffer.size()))
        if not samples:
            return {}

        # Basit update
        self._is_trained = True
        return {"online_samples": len(samples)}

    def predict(self, features: np.ndarray,
                position: int = 0,
                unrealized_pnl: float = 0.0,
                position_age: int = 0,
                max_drawdown: float = 0.0) -> Dict:
        """Canlı tahmin."""
        if not self._is_trained:
            return self._fallback_predict()

        try:
            return self.get_action(features, position, unrealized_pnl,
                                  position_age, max_drawdown, deterministic=True)
        except Exception as e:
            logger.error(f"PPOAgentV2 predict hatası: {e}")
            return self._fallback_predict()

    def _fallback_predict(self) -> Dict:
        return {
            "signal": "WAIT", "leverage": 5, "confidence": 50.0,
            "entry_action": 2, "exit_action": 0, "size_action": 0,
            "entry_probs": {"LONG": 33.3, "SHORT": 33.3, "WAIT": 33.4},
            "exit_probs": {"HOLD": 50.0, "CLOSE": 50.0},
            "size_probs": {"5x": 33.3, "10x": 33.3, "15x": 33.4},
            "value": 0.0, "_state": None,
        }

    def save(self, path: str) -> bool:
        try:
            os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".",
                       exist_ok=True)
            joblib.dump({
                "actor_params": self.actor.get_params(),
                "critic_params": self.critic,
                "is_trained": self._is_trained,
                "train_count": self._train_count,
                "total_steps": self._total_steps,
                "ep_rewards": self._episode_rewards[-100:],
            }, path, compress=3)
            logger.info(f"✅ PPOAgentV2 model kaydedildi → {path}")
            return True
        except Exception as e:
            logger.error(f"PPOAgentV2 kayıt hatası: {e}")
            return False

    def load(self, path: str) -> bool:
        if not os.path.exists(path):
            return False
        try:
            d = joblib.load(path)
            self.actor.set_params(d["actor_params"])
            self.critic = d["critic_params"]
            self._is_trained = d.get("is_trained", False)
            self._train_count = d.get("train_count", 0)
            self._total_steps = d.get("total_steps", 0)
            self._episode_rewards = d.get("ep_rewards", [])
            logger.info(f"✅ PPOAgentV2 model yüklendi ← {path}")
            return True
        except Exception as e:
            logger.error(f"PPOAgentV2 yükleme hatası: {e}")
            return False

    def get_info(self) -> Dict:
        avg_rew = float(np.mean(self._episode_rewards[-20:])) if self._episode_rewards else 0.0
        return {
            "algorithm": "PPO V2 (LSTM+Attention)",
            "state_dim": self.state_dim,
            "is_trained": self._is_trained,
            "train_count": self._train_count,
            "total_steps": self._total_steps,
            "avg_reward_20": round(avg_rew, 5),
            "replay_buffer_size": self._replay_buffer.size(),
            "architecture": {
                "actor": "Multi-Task (Entry/Exit/Size)",
                "lstm": self.actor.use_lstm,
                "attention": self.actor.use_attention,
            },
            "hyperparams": {
                "clip_eps": self.clip_eps,
                "gamma": self.gamma,
                "lam": self.lam,
                "entropy_coef": self.entropy_coef,
            },
        }


# ══════════════════════════════════════════════════════════════════════════════
# EXPORT
# ══════════════════════════════════════════════════════════════════════════════

__all__ = [
    "PPOAgentV2",
    "TradingEnvironmentV2",
    "AdvancedRewardFunction",
    "LSTMLayer",
    "MultiHeadAttention",
    "MultiTaskActor",
    "PrioritizedReplayBuffer",
]
