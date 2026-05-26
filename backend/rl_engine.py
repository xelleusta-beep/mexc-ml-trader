"""
MEXC ML Trading System — RL Engine v1.0
=========================================
Gerçek Reinforcement Learning — PPO (Proximal Policy Optimization)
NumPy tabanlı, Render free tier uyumlu (PyTorch gerekmez)

MİMARİ:
  TradingEnvironment (Gym-uyumlu)
    ↓
  PPOAgent (Actor-Critic, NumPy)
    - Actor:  [state_dim → 64 → 64 → n_actions] (policy)
    - Critic: [state_dim → 64 → 64 → 1]         (value)
    - Adam optimizer, GAE, clipped surrogate loss
    ↓
  OnlineLearner (canlıda fine-tuning)
    - Her 50 kapatılan trade sonrası mini-update
    - EWC (Elastic Weight Consolidation) — felaket unutma önleme
    ↓
  Backtest ile ön-eğitim → Canlıda online öğrenme

AKSIYON UZAYI (5 ayrık aksiyon):
  0: FLAT      — pozisyon yok, bekleme
  1: LONG_S    — küçük long  ($100 marjin, 5x)
  2: LONG_L    — büyük long  ($100 marjin, 10x)
  3: SHORT_S   — küçük short ($100 marjin, 5x)
  4: SHORT_L   — büyük short ($100 marjin, 10x)

DURUM UZAYI (68 boyut):
  64 teknik feature (V2) + 4 portföy durumu
  [features..., pnl_normalized, position_type, position_age, drawdown]

ÖDÜL FONKSİYONU:
  r = pnl_pct - 0.5*drawdown_penalty - fee - 0.001*overtime_penalty
  Normalize: tanh(r * 100)
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

# ─────────────────────────────────────────────────────────────────────────────
# NEURAL NETWORK — NumPy tabanlı Actor-Critic
# ─────────────────────────────────────────────────────────────────────────────
class NeuralNetwork:
    """
    Basit 3 katmanlı tam bağlantılı ağ.
    Aktivasyon: tanh (hidden), softmax/linear (output)
    """
    def __init__(self, layer_sizes: List[int], output_activation="linear", seed=42):
        rng = np.random.default_rng(seed)
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            fan_in  = layer_sizes[i]
            fan_out = layer_sizes[i+1]
            # He initialization
            scale = np.sqrt(2.0 / fan_in)
            W = rng.normal(0.0, scale, size=(fan_in, fan_out))
            b = np.zeros(fan_out)
            self.layers.append({"W": W, "b": b})
        self.output_activation = output_activation
        self._cache = {}
        self._lock = threading.Lock()

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._cache["inputs"] = [x]
        h = x
        for i, layer in enumerate(self.layers[:-1]):
            z = h @ layer["W"] + layer["b"]
            h = np.tanh(z)
            self._cache["inputs"].append(h)
        # Output layer
        z = h @ self.layers[-1]["W"] + self.layers[-1]["b"]
        if self.output_activation == "softmax":
            z = z - z.max()
            e = np.exp(z); out = e / (e.sum() + 1e-10)
        elif self.output_activation == "tanh":
            out = np.tanh(z)
        else:
            out = z
        return out

    def backward(self, grad_out: np.ndarray, cache: Optional[List[np.ndarray]] = None) -> List[Dict]:
        """
        Basit backprop — zincirleme kural.
        Tüm layer gradyanlarını döndürür.
        cache: forward'dan dönen activasyon listesi (None ise self._cache["inputs"] kullanılır).
        """
        if cache is None:
            cache = list(self._cache.get("inputs", []))
        grads = []
        n_layers = len(self.layers)
        if len(cache) < n_layers:
            logger.error(f"backward: cache ({len(cache)}) < layers ({n_layers})")
            return grads
        delta = grad_out

        for i in reversed(range(n_layers)):
            inp = cache[i]
            W   = self.layers[i]["W"]
            dW  = np.outer(inp, delta)
            db  = delta.copy()
            grads.insert(0, {"W": dW, "b": db})
            if i > 0:
                delta = (delta @ W.T) * (1 - cache[i]**2)

        return grads

    def get_params(self) -> Dict:
        return {f"W{i}": l["W"].copy() for i, l in enumerate(self.layers)} | \
               {f"b{i}": l["b"].copy() for i, l in enumerate(self.layers)}

    def set_params(self, params: Dict):
        n = len(self.layers)
        for i in range(n):
            self.layers[i]["W"] = params[f"W{i}"].copy()
            self.layers[i]["b"] = params[f"b{i}"].copy()

    def copy(self) -> "NeuralNetwork":
        import copy
        return copy.deepcopy(self)


# ─────────────────────────────────────────────────────────────────────────────
# ADAM OPTIMIZER
# ─────────────────────────────────────────────────────────────────────────────
class AdamOptimizer:
    def __init__(self, lr=3e-4, beta1=0.9, beta2=0.999, eps=1e-8, max_grad_norm=0.5):
        self.lr = lr; self.beta1 = beta1; self.beta2 = beta2
        self.eps = eps; self.max_grad_norm = max_grad_norm
        self.m = {}; self.v = {}; self.t = 0

    def step(self, params: Dict, grads: Dict) -> Dict:
        self.t += 1
        # Grad norm clipping
        total_norm = np.sqrt(sum(np.sum(g**2) for g in grads.values()))
        if total_norm > self.max_grad_norm:
            scale = self.max_grad_norm / (total_norm + 1e-6)
            grads = {k: v * scale for k, v in grads.items()}

        updated = {}
        for key in params:
            if key not in self.m:
                self.m[key] = np.zeros_like(params[key])
                self.v[key] = np.zeros_like(params[key])
            g = grads.get(key, np.zeros_like(params[key]))
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * g
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * g**2
            m_hat = self.m[key] / (1 - self.beta1**self.t)
            v_hat = self.v[key] / (1 - self.beta2**self.t)
            updated[key] = params[key] - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        return updated


# ─────────────────────────────────────────────────────────────────────────────
# TRADING ENVIRONMENT — Gym-uyumlu
# ─────────────────────────────────────────────────────────────────────────────
class TradingEnvironment:
    """
    Tek pair için episode-bazlı trading ortamı.

    State: [N feature + position_type(normalized) + unrealized_pnl + age + max_drawdown]
    Action: 0=FLAT, 1=LONG_S(5x), 2=LONG_L(10x), 3=SHORT_S(5x), 4=SHORT_L(10x)
    Reward: tanh(sharpe_step * 100)
    """
    N_ACTIONS   = 5
    FEE_RATE    = 0.0006
    MARGIN      = 100.0   # USD
    ACTION_LEV  = {0: 0, 1: 5, 2: 10, 3: 5, 4: 10}
    ACTION_SIDE = {0: "FLAT", 1: "LONG", 2: "LONG", 3: "SHORT", 4: "SHORT"}

    def __init__(self, episode_length: int = 100):  # FIX: 200→100, hızlı öğrenme
        self.episode_length = episode_length
        self._n_features = 0
        self.reset()

    def reset(self, features: Optional[np.ndarray] = None,
              prices: Optional[np.ndarray] = None) -> np.ndarray:
        """Yeni episode başlat."""
        self._features  = features
        self._prices    = prices
        self._n_features = features.shape[1] if features is not None else 0
        self._step      = 0
        self._position  = 0         # 0=flat, 1=long, -1=short
        self._leverage  = 0
        self._entry_price = 0.0
        self._entry_step  = 0
        self._unrealized  = 0.0
        self._equity      = 1.0     # Normalize başlangıç sermayesi
        self._peak_equity = 1.0
        self._max_dd      = 0.0
        self._episode_pnl = 0.0
        self._n_trades    = 0
        self._total_fees  = 0.0
        self._returns     = []
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """Mevcut state vektörünü oluştur."""
        if self._features is not None and self._step < len(self._features):
            feat = self._features[self._step].copy()
        else:
            feat = np.zeros(self._n_features, dtype=np.float32)

        # Portföy durumu (normalize edilmiş)
        pos_norm  = float(self._position) / 1.0        # -1, 0, 1
        pnl_norm  = np.tanh(self._unrealized / 50.0)   # PnL normalize
        age_norm  = min(1.0, (self._step - self._entry_step) / 32.0) if self._position != 0 else 0.0
        dd_norm   = self._max_dd

        state = np.concatenate([feat, [pos_norm, pnl_norm, age_norm, dd_norm]])
        return state.astype(np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Aksiyon uygula ve (next_state, reward, done, info) döndür.
        action: 0-4
        """
        if self._prices is None or self._step >= len(self._prices) - 1:
            return self._get_state(), 0.0, True, {}

        price_now  = float(self._prices[self._step])
        price_next = float(self._prices[min(self._step + 1, len(self._prices)-1)])
        if price_now <= 0: price_now = 1e-10

        reward     = 0.0
        fee_paid   = 0.0
        info       = {"action": action, "price": price_now}

        side = self.ACTION_SIDE[action]
        lev  = self.ACTION_LEV[action]

        # ── POZİSYON YÖNETİMİ ────────────────────────────────────────────────
        # 1. Mevcut pozisyonu kapat (yön değişimi veya FLAT komutu)
        if self._position != 0:
            if side == "FLAT" or \
               (side == "LONG" and self._position < 0) or \
               (side == "SHORT" and self._position > 0):
                # Çıkış PnL hesapla
                notional = self.MARGIN * self._leverage
                if self._position == 1:  # LONG
                    pnl_pct = (price_now - self._entry_price) / self._entry_price
                else:                     # SHORT
                    pnl_pct = (self._entry_price - price_now) / self._entry_price
                pnl_dollar = notional * pnl_pct
                exit_fee   = notional * (1 + pnl_pct) * self.FEE_RATE
                net_pnl    = pnl_dollar - exit_fee

                self._equity      += net_pnl / (self.MARGIN * 10)  # Normalize
                self._episode_pnl += net_pnl
                self._total_fees  += exit_fee
                self._n_trades    += 1
                self._returns.append(pnl_pct)
                fee_paid += exit_fee

                # Drawdown güncelle
                if self._equity > self._peak_equity:
                    self._peak_equity = self._equity
                dd = (self._peak_equity - self._equity) / (self._peak_equity + 1e-10)
                self._max_dd = max(self._max_dd, dd)

                # Ödül: çıkış PnL + maliyet
                reward += np.tanh(net_pnl / 5.0)  # FIX: /10→/5, daha güçlü ödül sinyali

                self._position   = 0
                self._leverage   = 0
                self._entry_price = 0.0
                info["trade_closed"] = True
                info["trade_pnl"]    = net_pnl

        # 2. Yeni pozisyon aç
        if side != "FLAT" and self._position == 0:
            entry_fee = self.MARGIN * lev * self.FEE_RATE
            self._equity      -= entry_fee / (self.MARGIN * 10)
            self._total_fees  += entry_fee
            fee_paid          += entry_fee
            self._position     = 1 if side == "LONG" else -1
            self._leverage     = lev
            self._entry_price  = price_now
            self._entry_step   = self._step
            reward             -= np.tanh(entry_fee / 5.0)  # Fee cezası

        # 3. Açık pozisyon unrealized PnL güncelle
        if self._position != 0:
            notional = self.MARGIN * self._leverage
            if self._position == 1:
                pnl_pct = (price_next - self._entry_price) / self._entry_price
            else:
                pnl_pct = (self._entry_price - price_next) / self._entry_price
            self._unrealized = notional * pnl_pct

            # Zaman cezası (8 saat = 32 bar sonrası)
            age = self._step - self._entry_step
            if age > 32:
                reward -= 0.005 * (age - 32) / 32.0

            # Drawdown cezası (anlık)
            if pnl_pct < -0.015:  # %1.5'den fazla zarar
                reward -= 0.1 * abs(pnl_pct)

        # ── BEKLEME ÖDÜLÜ ─────────────────────────────────────────────────────
        # FLAT kalmak için dengeli ceza
        if self._position == 0 and action == 0:
            reward -= 0.005  # Denge: 0.005×200adım=-1, ama kazanç >1 olabilir

        # ── ADIM İLERLE ──────────────────────────────────────────────────────
        self._step += 1
        done = (self._step >= min(self.episode_length, len(self._prices) - 1))

        # Episode sonu bonusu: Sharpe ratio
        if done and len(self._returns) >= 3:
            ret_arr = np.array(self._returns)
            sharpe  = float(ret_arr.mean() / (ret_arr.std() + 1e-8)) * np.sqrt(252)
            reward += np.tanh(sharpe * 0.1)

        next_state = self._get_state()
        info.update({
            "equity":     self._equity,
            "episode_pnl":self._episode_pnl,
            "n_trades":   self._n_trades,
            "max_dd":     self._max_dd,
        })
        return next_state, float(np.clip(reward, -1.0, 3.0)), done, info  # FIX: asimetrik


# ─────────────────────────────────────────────────────────────────────────────
# ROLLOUT BUFFER — PPO için veri toplama
# ─────────────────────────────────────────────────────────────────────────────
class RolloutBuffer:
    def __init__(self, capacity: int = 2048):
        self.capacity = capacity
        self.clear()

    def clear(self):
        self.states   = []
        self.actions  = []
        self.rewards  = []
        self.values   = []
        self.logprobs = []
        self.dones    = []

    def add(self, state, action, reward, value, logprob, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.logprobs.append(logprob)
        self.dones.append(done)

    def size(self): return len(self.states)

    def compute_returns_and_advantages(self, next_value: float,
                                        gamma: float = 0.99,
                                        lam: float = 0.95) -> Tuple:
        rewards  = np.array(self.rewards,  dtype=np.float32)
        values   = np.array(self.values,   dtype=np.float32)
        dones    = np.array(self.dones,    dtype=np.float32)
        n        = len(rewards)
        advs     = np.zeros(n, dtype=np.float32)
        last_adv = 0.0

        for t in reversed(range(n)):
            nv      = next_value if t == n - 1 else values[t + 1]
            mask    = 1.0 - dones[t]
            delta   = rewards[t] + gamma * nv * mask - values[t]
            last_adv = delta + gamma * lam * mask * last_adv
            advs[t] = last_adv

        returns = advs + values
        # Normalize advantages
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        return advs, returns

    def get_batches(self, batch_size: int = 256):
        n = self.size()
        states   = np.array(self.states,   dtype=np.float32)
        actions  = np.array(self.actions,  dtype=np.int32)
        logprobs = np.array(self.logprobs, dtype=np.float32)
        indices  = np.random.permutation(n)
        for start in range(0, n, batch_size):
            idx = indices[start:start + batch_size]
            yield (states[idx], actions[idx], logprobs[idx], idx)


# ─────────────────────────────────────────────────────────────────────────────
# PPO AGENT — Ana RL ajanı
# ─────────────────────────────────────────────────────────────────────────────
class PPOAgent:
    """
    Proximal Policy Optimization — NumPy implementasyonu.

    Hiperparametreler:
      clip_eps    : PPO clipping (varsayılan 0.2)
      gamma       : İndirim faktörü (0.99)
      lam         : GAE lambda (0.95)
      n_epochs    : Her update'te kaç epoch (4)
      batch_size  : Mini-batch büyüklüğü (256)
      entropy_coef: Entropi bonusu — keşfetmeyi teşvik (0.01)
      vf_coef     : Value loss katsayısı (0.5)
    """
    def __init__(self,
                 state_dim:    int   = 40,
                 n_actions:    int   = 5,
                 hidden_dim:   int   = 64,
                 lr:           float = 3e-4,
                 clip_eps:     float = 0.2,
                 gamma:        float = 0.99,
                 lam:          float = 0.95,
                 n_epochs:     int   = 4,
                 batch_size:   int   = 256,
                 entropy_coef: float = 0.05,  # FIX: 0.01 → 0.05 (daha fazla keşif)
                 vf_coef:      float = 0.5):

        self.state_dim    = state_dim
        self.n_actions    = n_actions
        self.clip_eps     = clip_eps
        self.gamma        = gamma
        self.lam          = lam
        self.n_epochs     = n_epochs
        self.batch_size   = batch_size
        self.entropy_coef = entropy_coef
        self.vf_coef      = vf_coef

        # Actor (policy network)
        self.actor  = NeuralNetwork([state_dim, hidden_dim, hidden_dim, n_actions],
                                     output_activation="softmax", seed=42)
        # Critic (value network)
        self.critic = NeuralNetwork([state_dim, hidden_dim, hidden_dim, 1],
                                     output_activation="linear", seed=43)

        self.actor_opt  = AdamOptimizer(lr=lr)
        self.critic_opt = AdamOptimizer(lr=lr)

        self.buffer       = RolloutBuffer(capacity=2048)
        self._train_count = 0
        self._total_steps = 0
        self._episode_rewards: List[float] = []
        self._is_trained  = False
        self._lock        = threading.Lock()   # predict/update thread safety

        # EWC (Elastic Weight Consolidation) — felaket unutma önleme
        self._ewc_fisher: Optional[Dict] = None
        self._ewc_params: Optional[Dict] = None
        self._ewc_lambda  = 1000.0

        logger.info(f"PPOAgent hazır — state_dim={state_dim}, n_actions={n_actions}")

    # ── AKSIYON SEÇİMİ ───────────────────────────────────────────────────────
    def get_action(self, state: np.ndarray,
                   deterministic: bool = False) -> Tuple[int, float, float]:
        """
        Verilen state'e göre aksiyon seç.
        Returns: (action, log_prob, value)
        """
        probs = self.actor.forward(state)
        value = float(self.critic.forward(state)[0])

        if deterministic:
            action = int(np.argmax(probs))
        else:
            # Temperature scaling — keşif için
            action = int(np.random.choice(self.n_actions, p=probs))

        log_prob = float(np.log(probs[action] + 1e-10))
        return action, log_prob, value

    # ── ROLLOUT TOPLAMA ──────────────────────────────────────────────────────
    def collect_rollout(self, env: TradingEnvironment,
                        features: np.ndarray,
                        prices:   np.ndarray,
                        n_steps:  int = 2048) -> Dict:
        """
        Ortamda n_steps adım at ve buffer'ı doldur.
        """
        self.buffer.clear()
        state    = env.reset(features, prices)
        ep_reward = 0.0
        ep_count  = 0

        for _ in range(n_steps):
            action, log_prob, value = self.get_action(state)
            next_state, reward, done, info = env.step(action)

            self.buffer.add(state, action, reward, value, log_prob, done)
            state      = next_state
            ep_reward += reward
            self._total_steps += 1

            if done:
                self._episode_rewards.append(ep_reward)
                if len(self._episode_rewards) > 100:
                    self._episode_rewards.pop(0)
                ep_reward = 0.0
                ep_count += 1
                state = env.reset(features, prices)

        # Son state'in değeri (bootstrap)
        _, _, next_value = self.get_action(state)
        return {"next_value": next_value, "ep_count": ep_count}

    # ── PPO UPDATE ───────────────────────────────────────────────────────────
    def update(self, next_value: float) -> Dict:
        """
        PPO güncelleme adımı.
        Toplanan rollout üzerinde n_epochs epoch eğit.
        """
        if self.buffer.size() < 64:
            return {}

        with self._lock:
            advantages, returns = self.buffer.compute_returns_and_advantages(
                next_value, self.gamma, self.lam
            )
            returns_arr = returns

            total_policy_loss = 0.0
            total_value_loss  = 0.0
            total_entropy     = 0.0
            n_updates         = 0

            for epoch in range(self.n_epochs):
                for states_b, actions_b, old_logprobs_b, idx_b in \
                        self.buffer.get_batches(self.batch_size):

                    adv_b = advantages[idx_b]
                    ret_b = returns_arr[idx_b]

                    # ── Actor (Policy) Update ─────────────────────────────────
                    actor_grads = {f"W{i}": np.zeros_like(l["W"])
                                   for i, l in enumerate(self.actor.layers)}
                    actor_grads |= {f"b{i}": np.zeros_like(l["b"])
                                    for i, l in enumerate(self.actor.layers)}

                    policy_loss = 0.0
                    entropy_sum = 0.0

                    for j, (s, a, old_lp, adv) in enumerate(
                            zip(states_b, actions_b, old_logprobs_b, adv_b)):
                        probs    = self.actor.forward(s)
                        new_lp   = float(np.log(probs[a] + 1e-10))
                        ratio    = np.exp(new_lp - float(old_lp))
                        clipped  = np.clip(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
                        ppo_obj  = min(ratio * adv, clipped * adv)
                        policy_loss -= ppo_obj

                        # Entropy (keşif teşvik)
                        entropy   = -float(np.sum(probs * np.log(probs + 1e-10)))
                        entropy_sum += entropy
                        policy_loss -= self.entropy_coef * entropy

                        # Backprop
                        grad_out = np.zeros(self.n_actions)
                        # dL/d_logit ≈ -(ratio * adv) * probs + probs * (sum)
                        # Basitleştirilmiş policy gradient
                        pg_coef = -(ratio * adv) if ratio * adv <= clipped * adv \
                                  else -(clipped * adv)
                        grad_out[a] = pg_coef * (1 - probs[a])  # softmax'ın türevi
                        for k in range(self.n_actions):
                            if k != a:
                                grad_out[k] = pg_coef * (-probs[a]) * probs[k] / (probs[k] + 1e-10)
                        # Entropy grad
                        grad_out -= self.entropy_coef * (np.log(probs + 1e-10) + 1)

                        layer_grads = self.actor.backward(grad_out / len(states_b))
                        for li, lg in enumerate(layer_grads):
                            actor_grads[f"W{li}"] += lg["W"]
                            actor_grads[f"b{li}"] += lg["b"]

                    # EWC penalty — önceki öğrenmeyi koru
                    if self._ewc_fisher is not None:
                        actor_params = self.actor.get_params()
                        for key in actor_params:
                            if key in self._ewc_fisher:
                                diff = actor_params[key] - self._ewc_params[key]
                                actor_grads[key] += (self._ewc_lambda *
                                                      self._ewc_fisher[key] * diff /
                                                      len(states_b))

                    actor_params = self.actor.get_params()
                    new_actor_params = self.actor_opt.step(actor_params, actor_grads)
                    self.actor.set_params(new_actor_params)

                    # ── Critic (Value) Update ─────────────────────────────────
                    critic_grads = {f"W{i}": np.zeros_like(l["W"])
                                    for i, l in enumerate(self.critic.layers)}
                    critic_grads |= {f"b{i}": np.zeros_like(l["b"])
                                     for i, l in enumerate(self.critic.layers)}

                    value_loss = 0.0
                    for s, ret in zip(states_b, ret_b):
                        v        = float(self.critic.forward(s)[0])
                        vl       = 0.5 * (v - float(ret))**2
                        value_loss += vl

                        dv       = np.array([v - float(ret)])
                        lg       = self.critic.backward(dv / len(states_b))
                        for li, g in enumerate(lg):
                            critic_grads[f"W{li}"] += g["W"]
                            critic_grads[f"b{li}"] += g["b"]

                    critic_params = self.critic.get_params()
                    new_critic_params = self.critic_opt.step(critic_params, critic_grads)
                    self.critic.set_params(new_critic_params)

                    total_policy_loss += policy_loss / len(states_b)
                    total_value_loss  += value_loss / len(states_b)
                    total_entropy     += entropy_sum / len(states_b)
                    n_updates         += 1

            self._train_count += 1
            self._is_trained   = True

        return {
            "policy_loss": round(total_policy_loss / max(n_updates, 1), 5),
            "value_loss":  round(total_value_loss  / max(n_updates, 1), 5),
            "entropy":     round(total_entropy     / max(n_updates, 1), 5),
            "n_updates":   n_updates,
        }

    # ── FULL TRAINING (Backtesting üzerinde) ─────────────────────────────────
    def train_on_history(self,
                         features_list: List[np.ndarray],
                         prices_list:   List[np.ndarray],
                         n_iterations:  int = 50) -> Dict:
        """
        Geçmiş veri üzerinde PPO eğitimi.
        features_list: Her pair için (T, 64) feature matrisi listesi
        prices_list:   Her pair için (T,) fiyat serisi listesi
        """
        if not features_list:
            return {"success": False, "reason": "no_data"}

        env = TradingEnvironment(episode_length=200)
        t0  = time.time()
        all_losses = []

        logger.info(f"PPO eğitimi başlıyor — {n_iterations} iterasyon, "
                    f"{len(features_list)} pair")

        for it in range(n_iterations):
            # Pair'leri döngüsel olarak kullan
            idx      = it % len(features_list)
            features = features_list[idx]
            prices   = prices_list[idx]

            if len(prices) < 50:
                continue

            info = self.collect_rollout(env, features, prices,
                                        n_steps=min(2048, len(prices) - 1))
            losses = self.update(info["next_value"])
            if losses:
                all_losses.append(losses)

            if (it + 1) % 10 == 0:
                avg_rew = np.mean(self._episode_rewards[-10:]) \
                          if self._episode_rewards else 0
                logger.info(f"  iter={it+1}/{n_iterations} | "
                            f"avg_reward={avg_rew:.4f} | "
                            f"steps={self._total_steps}")

        elapsed = time.time() - t0

        # EWC Fisher bilgisini güncelle
        self._update_ewc_fisher(features_list[0], prices_list[0], env)

        avg_rew = np.mean(self._episode_rewards[-20:]) if self._episode_rewards else 0
        result = {
            "success":       True,
            "n_iterations":  n_iterations,
            "avg_reward":    round(float(avg_rew), 5),
            "total_steps":   self._total_steps,
            "train_time_s":  round(elapsed, 2),
            "trained_at":    datetime.now(timezone.utc).isoformat(),
        }
        if all_losses:
            result["final_policy_loss"] = all_losses[-1].get("policy_loss", 0)
            result["final_value_loss"]  = all_losses[-1].get("value_loss", 0)

        logger.info(f"✅ PPO eğitimi tamamlandı | {elapsed:.1f}s | "
                    f"avg_reward={avg_rew:.4f}")
        return result

    # ── ONLINE FINE-TUNING ────────────────────────────────────────────────────
    def online_update(self, trade_experiences: List[Dict]) -> Dict:
        """
        Canlıda kapatılan trade'lerden online güncelleme.
        trade_experiences: [{"state", "action", "reward", "next_state", "done"}]
        EWC sayesinde önceki bilgi korunur.
        """
        if len(trade_experiences) < 5:
            return {}

        # Mini rollout oluştur
        self.buffer.clear()
        for exp in trade_experiences:
            state    = exp["state"]
            action   = exp["action"]
            reward   = exp["reward"]
            value    = float(self.critic.forward(state)[0])
            probs    = self.actor.forward(state)
            log_prob = float(np.log(probs[action] + 1e-10))
            done     = exp.get("done", False)
            self.buffer.add(state, action, reward, value, log_prob, done)

        next_state = trade_experiences[-1].get("next_state",
                                                trade_experiences[-1]["state"])
        next_value = float(self.critic.forward(next_state)[0])

        # Sadece 2 epoch — online update için hafif
        old_epochs     = self.n_epochs
        self.n_epochs  = 2
        losses         = self.update(next_value)
        self.n_epochs  = old_epochs

        logger.debug(f"Online update: {len(trade_experiences)} trade | "
                     f"policy_loss={losses.get('policy_loss', 0):.5f}")
        return losses

    # ── TAHMİN (Canlı kullanım) ───────────────────────────────────────────────
    def predict(self, features: np.ndarray,
                position: int = 0,
                unrealized_pnl: float = 0.0,
                position_age: int = 0,
                max_drawdown: float = 0.0) -> Dict:
        """
        Canlı veri için aksiyon tahmin et.
        Deterministic mode (en yüksek olasılıklı aksiyon).
        """
        if not self._is_trained:
            return self._fallback_predict()

        try:
            pos_norm = float(position) / 1.0
            pnl_norm = float(np.tanh(unrealized_pnl / 50.0))
            age_norm = float(min(1.0, position_age / 32.0))
            dd_norm  = float(max_drawdown)
            state    = np.concatenate([features,
                                        [pos_norm, pnl_norm, age_norm, dd_norm]
                                       ]).astype(np.float32)

            with self._lock:
                probs  = self.actor.forward(state)
                value  = float(self.critic.forward(state)[0])
            action = int(np.argmax(probs))
            # FIX: Uniform baseline'a göre normalize et
            # Uniform = %20, max %100 → scale to 50-100 range
            uniform = 1.0 / self.n_actions  # 0.20
            conf_raw = float(probs[action])
            # (prob - uniform) / (1 - uniform) → 0-1, sonra 50-100'e scale
            conf_norm = (conf_raw - uniform) / max(1 - uniform, 1e-6)
            conf = float(np.clip(50.0 + conf_norm * 50.0, 50.0, 99.0))

            action_map = {
                0: ("WAIT",  0,  "FLAT — Bekleme"),
                1: ("LONG",  5,  "LONG  5x — Küçük"),
                2: ("LONG",  10, "LONG 10x — Büyük"),
                3: ("SHORT", 5,  "SHORT  5x — Küçük"),
                4: ("SHORT", 10, "SHORT 10x — Büyük"),
            }
            signal, leverage, desc = action_map[action]

            return {
                "signal":     signal,
                "leverage":   leverage,
                "confidence": round(conf, 1),
                "action_id":  action,
                "action_desc":desc,
                "value":      round(value, 4),
                "probs": {
                    "FLAT":    round(float(probs[0]) * 100, 1),
                    "LONG_S":  round(float(probs[1]) * 100, 1),
                    "LONG_L":  round(float(probs[2]) * 100, 1),
                    "SHORT_S": round(float(probs[3]) * 100, 1),
                    "SHORT_L": round(float(probs[4]) * 100, 1),
                },
                "model":       "PPO-RL v1.0",
                "data_quality":"real",
                "_state":      state,
            }
        except Exception as e:
            logger.error(f"PPO predict hatası: {e}")
            return self._fallback_predict()

    def _fallback_predict(self) -> Dict:
        return {
            "signal": "WAIT", "leverage": 5, "confidence": 50.0,
            "action_id": 0, "action_desc": "Model eğitilmedi",
            "value": 0.0, "probs": {k: 20.0 for k in
                                     ["FLAT","LONG_S","LONG_L","SHORT_S","SHORT_L"]},
            "model": "PPO-RL v1.0", "data_quality": "insufficient", "_state": None,
        }

    # ── EWC (Felaket Unutma Önleme) ───────────────────────────────────────────
    def _update_ewc_fisher(self, features: np.ndarray, prices: np.ndarray,
                            env: TradingEnvironment, n_samples: int = 200):
        """
        Fisher bilgi matrisini hesapla (EWC için).
        Online update sırasında eski bilginin silinmesini önler.
        """
        try:
            fisher = {k: np.zeros_like(v)
                      for k, v in self.actor.get_params().items()}
            state  = env.reset(features, prices)

            for _ in range(min(n_samples, len(prices) - 1)):
                probs   = self.actor.forward(state)
                action  = int(np.random.choice(self.n_actions, p=probs))
                log_p   = np.log(probs[action] + 1e-10)

                grad_out   = np.zeros(self.n_actions)
                grad_out[action] = 1.0 / (probs[action] + 1e-10)
                layer_grads = self.actor.backward(grad_out)

                for li, lg in enumerate(layer_grads):
                    fisher[f"W{li}"] += lg["W"]**2 / n_samples
                    fisher[f"b{li}"] += lg["b"]**2 / n_samples

                next_state, _, done, _ = env.step(action)
                state = env.reset(features, prices) if done else next_state

            self._ewc_fisher = fisher
            self._ewc_params = self.actor.get_params()
            logger.debug("EWC Fisher matrisi güncellendi")
        except Exception as e:
            logger.warning(f"EWC Fisher hesaplama hatası: {e}")

    # ── KAYIT / YÜKLEME ──────────────────────────────────────────────────────
    def save(self, path: str) -> bool:
        try:
            os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".",
                        exist_ok=True)
            joblib.dump({
                "actor_params":  self.actor.get_params(),
                "critic_params": self.critic.get_params(),
                "actor_m":       self.actor_opt.m,
                "actor_v":       self.actor_opt.v,
                "actor_t":       self.actor_opt.t,
                "critic_m":      self.critic_opt.m,
                "critic_v":      self.critic_opt.v,
                "critic_t":      self.critic_opt.t,
                "train_count":   self._train_count,
                "total_steps":   self._total_steps,
                "is_trained":    self._is_trained,
                "ewc_fisher":    self._ewc_fisher,
                "ewc_params":    self._ewc_params,
                "ep_rewards":    self._episode_rewards[-100:],
            }, path, compress=3)
            logger.info(f"✅ PPO model kaydedildi → {path} "
                        f"({os.path.getsize(path)//1024}KB)")
            return True
        except Exception as e:
            logger.error(f"PPO kayıt hatası: {e}"); return False

    def load(self, path: str) -> bool:
        if not os.path.exists(path): return False
        try:
            d = joblib.load(path)
            # Boyut uyuşmazlığı kontrolü — eski model yeni state_dim ile uyumsuzsa
            saved_W0 = d.get("actor_params", {}).get("W0")
            if saved_W0 is not None and saved_W0.shape[0] != self.state_dim:
                logger.warning(f"PPO model boyut uyuşmazlığı: saved={saved_W0.shape[0]}, "
                              f"cfg={self.state_dim}. Taze ağırlıklarla başla.")
                return False
            self.actor.set_params(d["actor_params"])
            self.critic.set_params(d["critic_params"])
            self.actor_opt.m  = d["actor_m"];  self.actor_opt.v  = d["actor_v"]
            self.actor_opt.t  = d["actor_t"]
            self.critic_opt.m = d["critic_m"]; self.critic_opt.v = d["critic_v"]
            self.critic_opt.t = d["critic_t"]
            self._train_count     = d.get("train_count", 0)
            self._total_steps     = d.get("total_steps", 0)
            self._is_trained      = d.get("is_trained", False)
            self._ewc_fisher      = d.get("ewc_fisher")
            self._ewc_params      = d.get("ewc_params")
            self._episode_rewards = d.get("ep_rewards", [])
            logger.info(f"✅ PPO model yüklendi ← {path} | "
                        f"steps={self._total_steps}")
            return True
        except Exception as e:
            logger.error(f"PPO yükleme hatası: {e}"); return False

    def get_info(self) -> Dict:
        avg_rew = float(np.mean(self._episode_rewards[-20:])) \
                  if self._episode_rewards else 0.0
        return {
            "algorithm":     "PPO (NumPy)",
            "state_dim":     self.state_dim,
            "n_actions":     self.n_actions,
            "is_trained":    self._is_trained,
            "train_count":   self._train_count,
            "total_steps":   self._total_steps,
            "avg_reward_20": round(avg_rew, 5),
            "avg_reward":    round(avg_rew, 5),  # FIX: frontend alias
            "ewc_active":    self._ewc_fisher is not None,
            "exp_buffer":    0,   # OnlineExperienceBuffer ayrı nesne, main.py'den gelir
            "exp_total":     0,   # main.py'de rl_experience.total() ile doldurulur
            "architecture": {
                "actor":  f"[{self.state_dim}→64→64→{self.n_actions}] softmax",
                "critic": f"[{self.state_dim}→64→64→1] linear",
            },
            "hyperparams": {
                "clip_eps":     self.clip_eps,
                "gamma":        self.gamma,
                "lam":          self.lam,
                "n_epochs":     self.n_epochs,
                "entropy_coef": self.entropy_coef,
            },
        }


# ─────────────────────────────────────────────────────────────────────────────
# ONLINE EXPERIENCE BUFFER — Canlı trade deneyimlerini biriktir
# ─────────────────────────────────────────────────────────────────────────────
class OnlineExperienceBuffer:
    """
    Canlıda kapatılan trade'lerin deneyimlerini saklar.
    50 trade birikince PPOAgent'ı online olarak günceller.
    """
    TRIGGER_SIZE = 20

    def __init__(self):
        self._buffer: List[Dict] = []
        self._lock = threading.Lock()
        self._total = 0

    def add(self, state: np.ndarray, action: int, reward: float,
            next_state: np.ndarray, done: bool):
        with self._lock:
            self._buffer.append({
                "state":      state,
                "action":     action,
                "reward":     reward,
                "next_state": next_state,
                "done":       done,
            })
            self._total += 1

    def ready(self) -> bool:
        return len(self._buffer) >= self.TRIGGER_SIZE

    def get_and_clear(self) -> List[Dict]:
        with self._lock:
            data = self._buffer.copy()
            self._buffer.clear()
        return data

    def size(self)  -> int: return len(self._buffer)
    def total(self) -> int: return self._total
