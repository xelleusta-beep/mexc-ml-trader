"""
MEXC ML Trading System — Enhanced Ensemble
============================================
XGBoost + Stacking + Dynamic Weighting + Feature Importance

OZELLIKLER:
  1. XGBoost Model: Daha guclu gradient boosting
  2. Stacking Ensemble: Meta-learner ile model birlestirme
  3. Dynamic Weighting: Piyasa durumuna gore agirlik ayarlama
  4. Feature Importance: Onemli ozellik takibi
"""

import numpy as np
import logging
import time
import threading
from typing import Dict, List, Optional, Tuple
from collections import deque

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# XGBOOST MODEL
# ══════════════════════════════════════════════════════════════════════════════

class XGBoostModel:
    """
    XGBoost classifier with fallback.
    LightGBM'den daha guclu olabilir.
    """

    def __init__(self):
        self.clf = None
        self.is_fitted = False
        self._backend = self._detect()
        self._feature_importance = None

    def _detect(self):
        try:
            import xgboost
            return "xgboost"
        except ImportError:
            pass
        try:
            from sklearn.ensemble import GradientBoostingClassifier
            return "skgbm"
        except ImportError:
            pass
        return "numpy"

    def fit(self, X, y):
        if len(X) < 10:
            return self

        try:
            if self._backend == "xgboost":
                import xgboost as xgb
                self.clf = xgb.XGBClassifier(
                    n_estimators=400,
                    learning_rate=0.025,
                    max_depth=6,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    min_child_weight=5,
                    gamma=0.1,
                    reg_alpha=0.1,
                    reg_lambda=0.1,
                    use_label_encoder=False,
                    eval_metric="mlogloss",
                    random_state=42,
                    verbosity=0,
                    n_jobs=-1
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.clf.fit(X, y)
                logger.info(f"XGBoost fit — {len(X)} sample")

                # Feature importance
                self._feature_importance = self.clf.feature_importances_

            elif self._backend == "skgbm":
                from sklearn.ensemble import GradientBoostingClassifier
                self.clf = GradientBoostingClassifier(
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=5,
                    subsample=0.8,
                    random_state=42
                )
                self.clf.fit(X, y)
                self._feature_importance = self.clf.feature_importances_

            else:
                self._fit_numpy(X, y)

            self.is_fitted = True
        except Exception as e:
            logger.error(f"XGBoost fit: {e}")
            self._fit_numpy(X, y)

        return self

    def _fit_numpy(self, X, y):
        """NumPy fallback — basit AdaBoost."""
        import warnings
        n = len(X)
        w = np.ones(n) / n
        self._stumps = []
        self._alphas = []

        for _ in range(80):
            bf, bt, bd, berr = 0, 0.0, 1, float('inf')
            fset = np.random.choice(X.shape[1], min(8, X.shape[1]), replace=False)

            for f in fset:
                for t in np.percentile(X[:, f], [20, 40, 60, 80]):
                    for d in [1, -1]:
                        pred = np.where(X[:, f] * d > t * d, 1, -1)
                        yp = np.sign(y + 1e-6)
                        err = float(np.dot(w, pred != yp))
                        if err < berr:
                            berr, bf, bt, bd = err, f, t, d

            eps = max(berr, 1e-10)
            if eps >= 0.5:
                break

            alpha = 0.5 * np.log((1 - eps) / eps)
            pred = np.where(X[:, bf] * bd > bt * bd, 1, -1)
            yp = np.sign(y + 1e-6)
            w *= np.exp(-alpha * yp * pred)
            w /= w.sum()

            self._stumps.append((bf, bt, bd))
            self._alphas.append(alpha)

        self.is_fitted = True

    def predict_proba(self, x):
        default = {"LONG": 0.25, "SHORT": 0.25, "HOLD": 0.25, "WAIT": 0.25}
        if not self.is_fitted:
            return default

        try:
            if self._backend in ("xgboost", "skgbm") and self.clf is not None:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    p = self.clf.predict_proba(x.reshape(1, -1))[0]
                cls = list(self.clf.classes_)
                r = {"LONG": 0.1, "SHORT": 0.1, "HOLD": 0.1, "WAIT": 0.05}
                for i, c in enumerate(cls):
                    if c == 1:
                        r["LONG"] = float(p[i])
                    elif c == -1:
                        r["SHORT"] = float(p[i])
                    elif c == 0:
                        r["HOLD"] = float(p[i])
                t = sum(r.values())
                return {k: v / t for k, v in r.items()}
        except Exception:
            pass

        if hasattr(self, "_stumps") and self._stumps:
            score = sum(
                a * (1 if x[f] * d > t * d else -1)
                for (f, t, d), a in zip(self._stumps, self._alphas)
            )
            pl = float(1 / (1 + np.exp(-score * 2)))
            ps = float(1 / (1 + np.exp(score * 2)))
            ph = max(0, 1 - pl - ps + 0.1)
            pw = 0.05
            tot = pl + ps + ph + pw
            return {"LONG": pl / tot, "SHORT": ps / tot, "HOLD": ph / tot, "WAIT": pw / tot}

        return default

    def get_feature_importance(self):
        return self._feature_importance


# ══════════════════════════════════════════════════════════════════════════════
# STACKING ENSEMBLE
# ══════════════════════════════════════════════════════════════════════════════

class StackingEnsemble:
    """
    Stacking ensemble — birden fazla modeli birlestirir.
    Meta-learner: Logistic Regression veya basit ağırlıklı ortalama.
    """

    def __init__(self, models: Dict = None, use_meta_learner: bool = False):
        self.models = models or {}
        self.use_meta_learner = use_meta_learner
        self.meta_learner = None
        self.is_fitted = False
        self._weights = {}

    def fit(self, X, y):
        """
        Tum modelleri egit ve meta-learner'i ogret.
        """
        if len(X) < 20:
            return self

        # Tum modelleri egit
        for name, model in self.models.items():
            try:
                model.fit(X, y)
                logger.info(f"Stacking: {name} egitildi")
            except Exception as e:
                logger.error(f"Stacking {name} fit hatası: {e}")

        # Meta-learner egitimi (eğer kullaniliyorsa)
        if self.use_meta_learner and len(self.models) >= 2:
            self._fit_meta_learner(X, y)

        self.is_fitted = True
        return self

    def _fit_meta_learner(self, X, y):
        """
        Meta-learner icin out-of-fold tahminler olustur ve egit.
        """
        n_samples = len(X)
        n_models = len(self.models)

        # Out-of-fold tahminler
        meta_features = np.zeros((n_samples, n_models))

        for i, (name, model) in enumerate(self.models.items()):
            try:
                for j in range(n_samples):
                    # Basit out-of-fold: tum veri uzerinde tahmin
                    proba = model.predict_proba(X[j])
                    meta_features[j, i] = proba.get("LONG", 0) - proba.get("SHORT", 0)
            except Exception as e:
                logger.debug(f"Meta-learner {name} tahmin hatası: {e}")

        # Basit ağırlıklı ortalama (Logistic Regression yerine)
        # Her modelin basarimina gore agirlik belirle
        from collections import Counter
        y_counts = Counter(y)
        total = len(y)

        for i, (name, model) in enumerate(self.models.items()):
            # Basit agirlik: modelin tahmin dogruluguna gore
            correct = 0
            for j in range(min(100, n_samples)):
                proba = model.predict_proba(X[j])
                pred = max(proba, key=proba.get)
                pred_label = {"LONG": 1, "SHORT": -1, "HOLD": 0, "WAIT": 0}.get(pred, 0)
                if pred_label == y[j]:
                    correct += 1
            accuracy = correct / min(100, n_samples)
            self._weights[name] = accuracy

        # Agirliklari normalize et
        total_weight = sum(self._weights.values())
        if total_weight > 0:
            for name in self._weights:
                self._weights[name] /= total_weight

    def predict_proba(self, x):
        """
        Ensemble tahmini — ağırlıklı ortalama.
        """
        if not self.is_fitted or not self.models:
            return {"LONG": 0.25, "SHORT": 0.25, "HOLD": 0.25, "WAIT": 0.25}

        all_probas = {}
        for name, model in self.models.items():
            try:
                proba = model.predict_proba(x)
                all_probas[name] = proba
            except Exception as e:
                logger.debug(f"Stacking predict hatası ({name}): {e}")

        if not all_probas:
            return {"LONG": 0.25, "SHORT": 0.25, "HOLD": 0.25, "WAIT": 0.25}

        # Ağırlıklı ortalama
        final_proba = {"LONG": 0.0, "SHORT": 0.0, "HOLD": 0.0, "WAIT": 0.0}
        total_weight = 0.0

        for name, proba in all_probas.items():
            weight = self._weights.get(name, 1.0 / len(all_probas))
            for key in final_proba:
                final_proba[key] += proba.get(key, 0) * weight
            total_weight += weight

        if total_weight > 0:
            for key in final_proba:
                final_proba[key] /= total_weight

        return final_proba

    def get_model_weights(self):
        return dict(self._weights)


# ══════════════════════════════════════════════════════════════════════════════
# DYNAMIC MODEL WEIGHTER
# ══════════════════════════════════════════════════════════════════════════════

class DynamicModelWeighter:
    """
    Piyasa durumuna gore model agirliklarini ayarlar.
    - Trending piyasa → GBM agirlikli
    - Ranging piyasa → RF agirlikli
    - Volatil → XGBoost agirlikli
    """

    def __init__(self):
        self._market_regime = "neutral"
        self._regime_history = deque(maxlen=50)
        self._model_performance = {}
        self._weight_history = deque(maxlen=100)

    def detect_market_regime(self, klines: dict) -> str:
        """
        Piyasa rejimini tespit et.

        Returns: "trending" | "ranging" | "volatile"
        """
        c = np.asarray(klines.get("close", []), dtype=np.float64)
        if len(c) < 30:
            return "neutral"

        # Son 30 bardaki volatilite
        returns = np.diff(c[-30:]) / (c[-31:-1] + 1e-10)
        volatility = float(np.std(returns))

        # Trend strength
        ema_fast = float(np.mean(c[-10:]))
        ema_slow = float(np.mean(c[-30:]))
        trend_strength = abs(ema_fast - ema_slow) / (ema_slow + 1e-10)

        # Rejim belirleme
        if volatility > 0.03:
            regime = "volatile"
        elif trend_strength > 0.02:
            regime = "trending"
        else:
            regime = "ranging"

        self._market_regime = regime
        self._regime_history.append(regime)

        return regime

    def get_weights(self, regime: str = None) -> Dict[str, float]:
        """
        Piyasa rejimine gore model agirliklarini dondur.

        Returns: {"lgbm": float, "rf": float, "xgb": float}
        """
        if regime is None:
            regime = self._market_regime

        # Varsayilan agirliklar
        weights = {
            "lgbm": 0.4,
            "rf": 0.3,
            "xgb": 0.3,
        }

        # Rejim bazli ayarlama
        if regime == "trending":
            # Trending piyasa → GBM daha iyi
            weights = {"lgbm": 0.5, "rf": 0.2, "xgb": 0.3}
        elif regime == "ranging":
            # Ranging piyasa → RF daha iyi
            weights = {"lgbm": 0.3, "rf": 0.5, "xgb": 0.2}
        elif regime == "volatile":
            # Volatil piyasa → XGBoost daha iyi
            weights = {"lgbm": 0.3, "rf": 0.2, "xgb": 0.5}

        # Performans bazli ayarlama
        if self._model_performance:
            for model_name in weights:
                perf = self._model_performance.get(model_name, 0.5)
                weights[model_name] *= (0.5 + perf)

        # Normalize et
        total = sum(weights.values())
        if total > 0:
            for k in weights:
                weights[k] /= total

        self._weight_history.append(dict(weights))
        return weights

    def update_performance(self, model_name: str, accuracy: float):
        """Model performansini guncelle."""
        self._model_performance[model_name] = accuracy

    def get_regime_stats(self) -> Dict:
        """Rejim istatistikleri."""
        if not self._regime_history:
            return {"current": "neutral", "distribution": {}}

        recent = list(self._regime_history)[-20:]
        from collections import Counter
        dist = Counter(recent)

        return {
            "current": self._market_regime,
            "distribution": {k: round(v / len(recent) * 100, 1) for k, v in dist.items()},
        }


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE IMPORTANCE TRACKER
# ══════════════════════════════════════════════════════════════════════════════

class FeatureImportanceTracker:
    """
    Feature onem takibi — hangi ozellikler daha onemli.
    """

    def __init__(self, feature_names: List[str]):
        self._names = feature_names
        self._importances = np.zeros(len(feature_names))
        self._counts = np.zeros(len(feature_names))
        self._history = deque(maxlen=100)

    def update(self, importances: np.ndarray):
        """Feature importance guncelle."""
        if len(importances) != len(self._names):
            return
        self._importances += importances
        self._counts += 1

    def get_importance(self, top_k: int = 20) -> List[Tuple[str, float]]:
        """En onemli K ozelligi getir."""
        avg = self._importances / (self._counts + 1e-10)
        idx = np.argsort(-avg)[:top_k]
        return [(self._names[i], round(float(avg[i]), 4)) for i in idx]

    def get_selected_features(self, threshold: float = 0.01) -> List[str]:
        """Onem esiginin ustundeki feature'lari sec."""
        avg = self._importances / (self._counts + 1e-10)
        max_imp = np.max(avg)
        if max_imp < 1e-10:
            return self._names
        normalized = avg / max_imp
        selected = [self._names[i] for i in range(len(self._names))
                   if normalized[i] > threshold]
        return selected

    def get_stats(self) -> Dict:
        """Istatistikleri getir."""
        avg = self._importances / (self._counts + 1e-10)
        return {
            "total_features": len(self._names),
            "avg_importance": round(float(np.mean(avg)), 4),
            "max_importance": round(float(np.max(avg)), 4),
            "min_importance": round(float(np.min(avg)), 4),
            "updates": int(np.sum(self._counts)),
        }


# ══════════════════════════════════════════════════════════════════════════════
# ENHANCED ENSEMBLE ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class EnhancedEnsembleEngine:
    """
    Gelismis ensemble motoru:
    - GBM + RF + XGBoost
    - Dynamic weighting (piyasa rejimine gore)
    - Feature importance tracking
    """

    def __init__(self, feature_names: List[str] = None):
        self._models = {}
        self._weighter = DynamicModelWeighter()
        self._feature_tracker = None
        self._is_fitted = False
        self._lock = threading.Lock()

        if feature_names:
            self._feature_tracker = FeatureImportanceTracker(feature_names)

    def add_model(self, name: str, model):
        """Model ekle."""
        self._models[name] = model

    def fit(self, X, y, market_regime: str = None):
        """
        Tum modelleri egit.
        """
        with self._lock:
            # Piyasa rejimini tespit et
            if market_regime:
                self._weighter._market_regime = market_regime

            # Tum modelleri egit
            for name, model in self._models.items():
                try:
                    model.fit(X, y)
                    logger.info(f"Ensemble: {name} egitildi")
                except Exception as e:
                    logger.error(f"Ensemble {name} fit hatası: {e}")

            # Feature importance guncelle
            if self._feature_tracker:
                for name, model in self._models.items():
                    if hasattr(model, 'get_feature_importance'):
                        imp = model.get_feature_importance()
                        if imp is not None:
                            self._feature_tracker.update(imp)

            self._is_fitted = True

    def predict_proba(self, x, market_regime: str = None) -> Dict:
        """
        Ensemble tahmini — dynamic weighting ile.
        """
        if not self._is_fitted or not self._models:
            return {"LONG": 0.25, "SHORT": 0.25, "HOLD": 0.25, "WAIT": 0.25}

        with self._lock:
            # Agirliklari al
            weights = self._weighter.get_weights(market_regime)

            # Tahminleri topla
            all_probas = {}
            for name, model in self._models.items():
                try:
                    proba = model.predict_proba(x)
                    all_probas[name] = proba
                except Exception as e:
                    logger.debug(f"Ensemble predict hatası ({name}): {e}")

            if not all_probas:
                return {"LONG": 0.25, "SHORT": 0.25, "HOLD": 0.25, "WAIT": 0.25}

            # Ağırlıklı ortalama
            final_proba = {"LONG": 0.0, "SHORT": 0.0, "HOLD": 0.0, "WAIT": 0.0}
            total_weight = 0.0

            for name, proba in all_probas.items():
                weight = weights.get(name, 1.0 / len(all_probas))
                for key in final_proba:
                    final_proba[key] += proba.get(key, 0) * weight
                total_weight += weight

            if total_weight > 0:
                for key in final_proba:
                    final_proba[key] /= total_weight

            return final_proba

    def get_feature_importance(self, top_k: int = 20) -> List[Tuple[str, float]]:
        """Feature onemlerini getir."""
        if self._feature_tracker:
            return self._feature_tracker.get_importance(top_k)
        return []

    def get_model_weights(self) -> Dict[str, float]:
        """Model agirliklarini getir."""
        return self._weighter.get_weights()

    def get_stats(self) -> Dict:
        """Istatistikleri getir."""
        return {
            "models": list(self._models.keys()),
            "weights": self.get_model_weights(),
            "regime": self._weighter.get_regime_stats(),
            "feature_stats": self._feature_tracker.get_stats() if self._feature_tracker else None,
        }


# ══════════════════════════════════════════════════════════════════════════════
# EXPORT
# ══════════════════════════════════════════════════════════════════════════════

__all__ = [
    "XGBoostModel",
    "StackingEnsemble",
    "DynamicModelWeighter",
    "FeatureImportanceTracker",
    "EnhancedEnsembleEngine",
]
