import numpy as np
import logging
import warnings
import time
from datetime import datetime, timezone

from config import config
from features import FeatureBuilder

logger = logging.getLogger(__name__)


class LightGBMModel:
    def __init__(self):
        self.clf = None
        self.is_fitted = False

    def fit(self, X, y):
        if len(X) < 20:
            return self
        try:
            import lightgbm as lgb
            self.clf = lgb.LGBMClassifier(
                n_estimators=config.lgbm_n_estimators,
                learning_rate=config.lgbm_learning_rate,
                max_depth=config.lgbm_max_depth,
                num_leaves=15,
                min_child_samples=5,
                subsample=0.8,
                colsample_bytree=0.8,
                class_weight="balanced",
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42,
                verbose=-1,
                n_jobs=1,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.clf.fit(X, y)
            self.is_fitted = True
            logger.info(f"LightGBM fit: {len(X)} samples")
        except ImportError:
            logger.warning("LightGBM not installed, using numpy fallback")
            self._fit_numpy(X, y)
        return self

    def _fit_numpy(self, X, y):
        n = len(X)
        w = np.ones(n) / n
        self._stumps = []
        self._alphas = []
        for _ in range(40):
            bf, bt, bd, berr = 0, 0.0, 1, float("inf")
            fset = np.random.choice(X.shape[1], min(6, X.shape[1]), replace=False)
            for f in fset:
                for t in np.percentile(X[:, f], [25, 50, 75]):
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
        default = {"LONG": 0.25, "SHORT": 0.25, "HOLD": 0.3, "WAIT": 0.2}
        if not self.is_fitted:
            return default
        try:
            if self.clf is not None:
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
            score = sum(a * (1 if x[f] * d > t * d else -1) for (f, t, d), a in zip(self._stumps, self._alphas))
            pl = float(1 / (1 + np.exp(-score * 2)))
            ps = float(1 / (1 + np.exp(score * 2)))
            ph = max(0, 1 - pl - ps + 0.1)
            tot = pl + ps + ph + 0.05
            return {"LONG": pl / tot, "SHORT": ps / tot, "HOLD": ph / tot, "WAIT": 0.05 / tot}
        return default


class MLEngine:
    def __init__(self):
        self.model = LightGBMModel()
        self._trained = False
        self._wf = {}
        self._train_log = []
        self._last_train_ts = 0.0
        self._training = False

    def n_features(self):
        return FeatureBuilder.N_FEATURES

    def train(self, klines):
        if self._training:
            return {"success": False, "reason": "training_in_progress"}
        self._training = True
        t0 = time.time()
        try:
            X, y = FeatureBuilder.build_dataset(klines, config.lookahead, config.label_threshold)
            if X is None:
                return {"success": False, "reason": "insufficient_data"}
            if len(X) < 40:
                return {"success": False, "reason": "too_few_samples", "n": len(X)}

            Xb, yb = _balance(X, y)
            split = int(len(Xb) * 0.8)
            Xtr, Xte = Xb[:split], Xb[split:]
            ytr, yte = yb[:split], yb[split:]

            self.model.fit(Xtr, ytr)
            self._trained = True

            lmap = {"LONG": 1, "SHORT": -1, "HOLD": 0}
            preds = np.array([lmap[max(p := self.model.predict_proba(x), key=p.get)] for x in Xte])
            acc = float(np.mean(preds == yte)) * 100

            self._wf = _wf_validate(X, y)
            wf_acc = self._wf.get("accuracy", 0)

            elapsed = time.time() - t0
            rec = {
                "success": True,
                "n_samples": len(X),
                "n_train": len(Xtr),
                "n_test": len(Xte),
                "test_accuracy": round(acc, 1),
                "wf_accuracy": wf_acc,
                "train_time_s": round(elapsed, 2),
                "trained_at": datetime.now(timezone.utc).isoformat(),
            }
            self._train_log.append(rec)
            if len(self._train_log) > 10:
                self._train_log.pop(0)
            self._last_train_ts = time.time()
            logger.info(f"LightGBM train: {len(X)} samples, acc={acc:.1f}%, wf={wf_acc}%")
            return rec
        finally:
            self._training = False

    def predict(self, klines, price):
        feat = FeatureBuilder.build(klines)
        if feat is None:
            return {"signal": "WAIT", "confidence": 0, "_feat": None}

        proba = self.model.predict_proba(feat)
        long_conf = proba["LONG"] * 100
        short_conf = proba["SHORT"] * 100
        hold_conf = proba["HOLD"] * 100

        if long_conf > short_conf and long_conf > hold_conf:
            sig = "LONG"
            conf = long_conf
        elif short_conf > long_conf and short_conf > hold_conf:
            sig = "SHORT"
            conf = short_conf
        else:
            sig = "WAIT"
            conf = hold_conf

        if conf < config.min_confidence_entry:
            sig = "WAIT"

        return {
            "signal": sig,
            "confidence": round(conf, 1),
            "long_conf": round(long_conf, 1),
            "short_conf": round(short_conf, 1),
            "_feat": feat,
            "model": "LightGBM",
        }

    def should_retrain(self):
        if not self._trained:
            return True, "not_trained"
        elapsed = time.time() - self._last_train_ts
        if elapsed > config.retrain_interval_sec:
            return True, "temporal"
        return False, ""

    def save(self, path):
        try:
            import joblib
            joblib.dump({"model": self.model, "wf": self._wf, "train_log": self._train_log}, path)
            logger.info(f"ML model saved: {path}")
        except Exception as e:
            logger.error(f"ML save error: {e}")

    def load(self, path):
        try:
            import os
            if not os.path.exists(path):
                return False
            import joblib
            data = joblib.load(path)
            self.model = data.get("model", self.model)
            self._wf = data.get("wf", {})
            self._train_log = data.get("train_log", [])
            self._trained = self.model.is_fitted
            logger.info(f"ML model loaded: {path}")
            return True
        except Exception as e:
            logger.error(f"ML load error: {e}")
            return False


def _balance(X, y, target_ratio=0.35):
    unique, counts = np.unique(y, return_counts=True)
    if len(unique) < 2:
        return X, y
    total = len(y)
    max_count = int(total * (1 - target_ratio * (len(unique) - 1)))
    X_parts, y_parts = [], []

    long_mask = y == 1
    short_mask = y == -1
    long_cnt = long_mask.sum()
    short_cnt = short_mask.sum()
    if long_cnt > 0 and short_cnt > 0:
        sym_target = min(long_cnt, short_cnt)
        if max(long_cnt, short_cnt) > sym_target * 1.2:
            if long_cnt > sym_target:
                idx = np.random.choice(long_cnt, sym_target, replace=False)
                X = np.vstack([X[~long_mask], X[long_mask][idx]])
                y = np.concatenate([y[~long_mask], y[long_mask][idx]])
            elif short_cnt > sym_target:
                idx = np.random.choice(short_cnt, sym_target, replace=False)
                X = np.vstack([X[~short_mask], X[short_mask][idx]])
                y = np.concatenate([y[~short_mask], y[short_mask][idx]])
        unique, counts = np.unique(y, return_counts=True)
        total = len(y)
        max_count = int(total * (1 - target_ratio * (len(unique) - 1)))

    for cls, cnt in zip(unique, counts):
        mask = y == cls
        Xc, yc = X[mask], y[mask]
        if cls == 0 and cnt > max_count:
            idx = np.random.choice(cnt, max_count, replace=False)
            Xc, yc = Xc[idx], yc[idx]
        X_parts.append(Xc)
        y_parts.append(yc)

    Xb = np.vstack(X_parts)
    yb = np.concatenate(y_parts)
    idx = np.random.permutation(len(Xb))
    return Xb[idx], yb[idx]


def _wf_validate(X, y, n_splits=3):
    if X is None or len(X) < 40:
        return {"accuracy": 0, "n_samples": 0}
    n = len(X)
    sz = n // (n_splits + 1)
    accs = []
    for i in range(1, n_splits + 1):
        te = sz * (i + 1)
        Xtr, ytr = X[:sz * i], y[:sz * i]
        Xte, yte = X[sz * i:te], y[sz * i:te]
        if len(Xtr) < 20 or len(Xte) < 5:
            continue
        if len(np.unique(yte)) < 2:
            continue
        m = LightGBMModel()
        m.fit(Xtr, ytr)
        lmap = {"LONG": 1, "SHORT": -1, "HOLD": 0}
        preds = np.array([lmap[max(p := m.predict_proba(x), key=p.get)] for x in Xte])
        accs.append(float(np.mean(preds == yte)))
    return {"accuracy": round(float(np.mean(accs)) * 100, 1) if accs else 0, "n_samples": n, "n_folds": len(accs)}
