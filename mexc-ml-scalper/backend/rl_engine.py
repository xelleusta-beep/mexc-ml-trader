"""
Basit RL agent — sadece pozisyon boyutlandırma için.
LightGBM sinyal üretir, RL kaç lot/kaldıraç kullanılacağına karar verir.
"""

import numpy as np
import logging
from collections import deque

logger = logging.getLogger(__name__)


class SimplePPO:
    """
    Minimal PPO — sadece pozisyon boyutlandırma (3 aksiyon: 1x, 3x, 5x).
    State: [signal_strength, confidence, current_pos, unrealized_pnl, atr, volatility]
    """

    def __init__(self, state_dim=18, hidden_dim=32, action_dim=3, lr=3e-4):
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.lr = lr

        # Basit lineer policy
        self._policy = np.random.randn(state_dim, action_dim) * 0.01
        self._value = np.random.randn(state_dim) * 0.01
        self._optimizer_steps = 0
        self._is_trained = False
        self._episode_rewards = deque(maxlen=50)
        self._total_steps = 0

    def _features(self, ml_feat, position_type, unrealized_pnl, position_age, max_dd):
        """ML feature'larini RL state'e donustur."""
        if ml_feat is None:
            return np.zeros(self.state_dim, dtype=np.float32)
        if len(ml_feat) > self.state_dim:
            ml_feat = ml_feat[:self.state_dim]
        f = np.zeros(self.state_dim, dtype=np.float32)
        n = min(len(ml_feat), self.state_dim - 4)
        f[:n] = ml_feat[:n]
        f[self.state_dim - 4] = position_type / 1.0
        f[self.state_dim - 3] = float(np.clip(unrealized_pnl / 100, -1, 1))
        f[self.state_dim - 2] = float(np.clip(position_age / 20, 0, 1))
        f[self.state_dim - 1] = float(max_dd)
        return f

    def get_action(self, state):
        """Leverage karari: 0->1x, 1->3x, 2->5x"""
        logits = state @ self._policy
        exp_l = np.exp(logits - np.max(logits))
        probs = exp_l / (exp_l.sum() + 1e-10)
        action = np.random.choice(self.action_dim, p=probs)
        return action, probs

    def get_value(self, state):
        return float(state @ self._value)

    def predict(self, ml_feat, position_type, unrealized_pnl, position_age, max_dd):
        state = self._features(ml_feat, position_type, unrealized_pnl, position_age, max_dd)
        action, probs = self.get_action(state)
        value = self.get_value(state)

        lev_map = {0: 1, 1: 3, 2: 5}
        leverage = lev_map.get(action, 5)

        return {
            "leverage": leverage,
            "action": int(action),
            "action_desc": f"{leverage}x",
            "value": round(value, 4),
            "probs": [round(float(p), 4) for p in probs],
            "_state": state,
        }

    def online_update(self, experiences):
        """
        experiences: list of (state, action, reward, next_state, done)
        Minimal update: policy gradient with baseline.
        """
        if len(experiences) < 3:
            return {"policy_loss": 0, "value_loss": 0}

        states = np.array([e[0] for e in experiences])
        actions = np.array([e[1] for e in experiences])
        rewards = np.array([e[2] for e in experiences], dtype=np.float32)
        dones = np.array([e[4] for e in experiences])

        # Normalize rewards
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-10)

        # Policy gradient
        advantages = rewards.copy()
        logits = states @ self._policy
        exp_l = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_l / (exp_l.sum(axis=1, keepdims=True) + 1e-10)

        action_mask = np.zeros((len(actions), self.action_dim))
        action_mask[np.arange(len(actions)), actions] = 1

        log_prob = np.log(probs + 1e-10) * action_mask
        pg_loss = -float(np.mean(log_prob.sum(axis=1) * advantages))

        # Value loss
        values = states @ self._value
        v_loss = float(np.mean((values - rewards) ** 2))

        # Simple SGD update
        grad_p = -(states.T @ (action_mask - probs)) / len(states) * advantages.mean()
        grad_v = (states.T @ (values - rewards)) / len(states) * 2

        self._policy -= self.lr * grad_p
        self._value -= self.lr * grad_v
        self._optimizer_steps += 1
        self._total_steps += len(experiences)
        self._is_trained = True

        if rewards.mean() > 0:
            self._episode_rewards.append(float(rewards.mean()))

        return {"policy_loss": round(pg_loss, 4), "value_loss": round(v_loss, 4)}

    def save(self, path):
        try:
            import joblib
            joblib.dump({
                "policy": self._policy,
                "value": self._value,
                "steps": self._total_steps,
                "trained": self._is_trained,
            }, path)
        except Exception as e:
            logger.error(f"RL save error: {e}")

    def load(self, path):
        try:
            import os
            if not os.path.exists(path):
                return False
            import joblib
            data = joblib.load(path)
            self._policy = data.get("policy", self._policy)
            self._value = data.get("value", self._value)
            self._total_steps = data.get("steps", 0)
            self._is_trained = data.get("trained", False)
            return True
        except Exception as e:
            logger.error(f"RL load error: {e}")
            return False
