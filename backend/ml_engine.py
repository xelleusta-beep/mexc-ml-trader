"""
MEXC ML Trading System — GERÇEK ML Engine
==========================================
✅ model.fit() — gerçekten öğrenen modeller
✅ Walk-forward validation (time-series safe)
✅ Backtest (ROI, Sharpe, Drawdown)
✅ Feedback loop (canlı sonuç → retrain)
✅ 28 gerçek feature (lag, rolling, momentum)
✅ LightGBM/sklearn GBM + RandomForest ensemble
✅ Hiç rastgele sinyal yok — sadece veriden öğrenir
"""

import numpy as np
import logging
import time
import os
from typing import Optional
from collections import deque
from datetime import datetime

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. TEKNİK İNDİKATÖRLER
# ═══════════════════════════════════════════════════════════════════════════════

class Indicators:
    @staticmethod
    def rsi(c: np.ndarray, p=14) -> float:
        if len(c) < p + 2: return 50.0
        d = np.diff(c[-(p+5):])
        g = np.where(d > 0, d, 0.0)
        lo = np.where(d < 0, -d, 0.0)
        ag, al = np.mean(g[-p:]), np.mean(lo[-p:])
        return 100.0 if al < 1e-12 else float(100 - 100 / (1 + ag / al))

    @staticmethod
    def ema(c: np.ndarray, p: int) -> np.ndarray:
        if len(c) == 0: return np.zeros(1)
        k = 2 / (p + 1)
        out = np.empty(len(c))
        out[0] = c[0]
        for i in range(1, len(c)):
            out[i] = c[i] * k + out[i-1] * (1 - k)
        return out

    @staticmethod
    def macd(c: np.ndarray):
        if len(c) < 35: return 0.0, 0.0, 0.0
        e12 = Indicators.ema(c, 12)
        e26 = Indicators.ema(c, 26)
        ml = e12 - e26
        sig = Indicators.ema(ml, 9)
        return float(ml[-1]), float(sig[-1]), float(ml[-1] - sig[-1])

    @staticmethod
    def bb(c: np.ndarray, p=20):
        if len(c) < p:
            v = float(c[-1]) if len(c) else 0
            return v, v, v
        s, std = float(np.mean(c[-p:])), float(np.std(c[-p:]))
        return s + 2*std, s, s - 2*std

    @staticmethod
    def atr(h, lo, c, p=14) -> float:
        if len(c) < p+1: return 0.0
        tr = np.maximum(h[1:]-lo[1:], np.maximum(np.abs(h[1:]-c[:-1]), np.abs(lo[1:]-c[:-1])))
        return float(np.mean(tr[-p:]))

    @staticmethod
    def obv_trend(c, v) -> float:
        if len(c) < 20: return 0.0
        obv = np.zeros(len(c))
        obv[0] = v[0]
        for i in range(1, len(c)):
            obv[i] = obv[i-1] + (v[i] if c[i] > c[i-1] else -v[i] if c[i] < c[i-1] else 0)
        ref = float(np.abs(np.mean(obv[-20:-10])) + 1e-10)
        return float(np.clip((np.mean(obv[-10:]) - np.mean(obv[-20:-10])) / ref, -3, 3))

    @staticmethod
    def mfi(h, lo, c, v, p=14) -> float:
        if len(c) < p+1: return 50.0
        tp = (h + lo + c) / 3
        mf = tp * v
        pos = np.where(np.diff(tp) > 0, mf[1:], 0)
        neg = np.where(np.diff(tp) < 0, mf[1:], 0)
        ps, ns = float(np.sum(pos[-p:])), float(np.sum(neg[-p:]))
        return 100.0 if ns < 1e-12 else float(100 - 100 / (1 + ps / ns))

    @staticmethod
    def stoch_rsi(c, p=14) -> float:
        rvals = [Indicators.rsi(c[:i], p) for i in range(p+1, len(c)+1)]
        if len(rvals) < p: return 50.0
        arr = np.array(rvals[-p:])
        mn, mx = arr.min(), arr.max()
        return float((arr[-1] - mn) / (mx - mn + 1e-10) * 100)

    @staticmethod
    def williams_r(h, lo, c, p=14) -> float:
        if len(c) < p: return -50.0
        hh, ll = float(np.max(h[-p:])), float(np.min(lo[-p:]))
        return float(-100 * (hh - c[-1]) / (hh - ll + 1e-10))


# ═══════════════════════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING — 28 gerçek özellik
# ═══════════════════════════════════════════════════════════════════════════════

class FeatureBuilder:
    N_FEATURES = 28

    @classmethod
    def build(cls, klines: dict) -> Optional[np.ndarray]:
        """Tek bar için 28 özellik vektörü üret"""
        c = np.asarray(klines["close"], dtype=np.float64)
        h = np.asarray(klines["high"],  dtype=np.float64)
        lo= np.asarray(klines["low"],   dtype=np.float64)
        v = np.asarray(klines["volume"],dtype=np.float64)
        if len(c) < 30: return None

        ti = Indicators()
        f = []
        pr = c[-1] + 1e-10

        # — Momentum oscillators —
        f.append(ti.rsi(c, 14) / 100.0)               # 0
        f.append(ti.stoch_rsi(c, 14) / 100.0)          # 1
        f.append(ti.mfi(h, lo, c, v, 14) / 100.0)      # 2
        f.append(ti.williams_r(h, lo, c, 14) / -100.0) # 3  → 0-1

        # — Trend —
        ml, ms, mh = ti.macd(c)
        f.extend([np.clip(ml/pr, -0.05, 0.05),          # 4
                  np.clip(ms/pr, -0.05, 0.05),          # 5
                  np.clip(mh/pr, -0.05, 0.05)])         # 6

        e9  = ti.ema(c, 9)[-1]
        e21 = ti.ema(c, 21)[-1]
        e50 = ti.ema(c, min(50, len(c)-1))[-1]
        f.append(np.clip((e9 - e21) / pr, -0.05, 0.05)) # 7
        f.append(np.clip((c[-1] - e50) / pr, -0.1, 0.1))# 8

        # — Volatility —
        bbu, _, bbl = ti.bb(c, 20)
        bb_rng = bbu - bbl + 1e-10
        f.append(np.clip((c[-1] - bbl) / bb_rng, 0, 1)) # 9   BB %B
        f.append(np.clip(bb_rng / pr, 0, 0.2))           # 10  BB width
        f.append(np.clip(ti.atr(h, lo, c, 14) / pr, 0, 0.05)) # 11

        # — Volume —
        vavg = float(np.mean(v[-21:-1]) if len(v)>=21 else v.mean()) + 1e-10
        f.append(np.clip(v[-1] / vavg, 0, 5))            # 12 volume ratio
        f.append(ti.obv_trend(c, v))                      # 13 OBV trend

        # — Lag returns t-1, t-2, t-3 —
        for lag in [1, 2, 3]:
            ret = (c[-1] - c[-1-lag]) / (c[-1-lag]+1e-10) if len(c) > lag else 0.0
            f.append(np.clip(ret, -0.1, 0.1))             # 14,15,16

        # — Rolling returns 5, 10, 20 —
        for w in [5, 10, 20]:
            ret = (c[-1]-c[-1-w])/(c[-1-w]+1e-10) if len(c)>w else 0.0
            f.append(np.clip(ret, -0.3, 0.3))             # 17,18,19

        # — Rolling volatility 5, 20 —
        for w in [5, 20]:
            s = float(np.std(c[-w:])) / pr if len(c)>=w else 0.0
            f.append(np.clip(s, 0, 0.1))                  # 20,21

        # — Price position in HL range —
        h20, l20 = float(np.max(h[-20:])), float(np.min(lo[-20:]))
        rng = h20 - l20 + 1e-10
        f.append((c[-1] - l20) / rng)                     # 22
        f.append((h20 - c[-1]) / rng)                     # 23

        # — Momentum acceleration (5-bar mom diff) —
        m5 = c[-1] - c[-6] if len(c)>=6 else 0.0
        mp = c[-2] - c[-7] if len(c)>=7 else 0.0
        f.append(np.clip((m5 - mp) / pr, -0.05, 0.05))    # 24

        # — Candle structure —
        body = abs(c[-1]-c[-2]) / pr if len(c)>=2 else 0.0
        f.append(np.clip(body, 0, 0.05))                   # 25
        hl = (h[-1]-lo[-1]) / pr
        f.append(np.clip(hl, 0, 0.1))                     # 26
        uw = (h[-1]-max(c[-1],c[-2] if len(c)>=2 else c[-1])) / (h[-1]-lo[-1]+1e-10)
        f.append(np.clip(float(uw), 0, 1))                 # 27

        arr = np.array(f, dtype=np.float32)
        return np.where(np.isfinite(arr), arr, 0.0)

    @classmethod
    def build_dataset(cls, klines: dict, lookahead=5, threshold=0.008):
        """
        Geçmiş kline verisinden (X, y) dataset üret.
        Label: lookahead bar sonra threshold% yukarı → 1 (LONG)
               threshold% aşağı → -1 (SHORT), ortada → 0 (HOLD)
        """
        c  = np.asarray(klines["close"],  dtype=np.float64)
        h  = np.asarray(klines["high"],   dtype=np.float64)
        lo = np.asarray(klines["low"],    dtype=np.float64)
        v  = np.asarray(klines["volume"], dtype=np.float64)
        MIN = 40
        if len(c) < MIN + lookahead: return None, None

        X_rows, y_rows = [], []
        for i in range(MIN, len(c) - lookahead):
            sub = {"close": c[:i], "high": h[:i], "low": lo[:i], "volume": v[:i]}
            feat = cls.build(sub)
            if feat is None: continue
            ret = (c[i+lookahead] - c[i]) / (c[i]+1e-10)
            label = 1 if ret > threshold else (-1 if ret < -threshold else 0)
            X_rows.append(feat)
            y_rows.append(label)

        if len(X_rows) < 10: return None, None
        return np.array(X_rows, np.float32), np.array(y_rows, np.int32)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. GERÇEK ML MODELLERİ — model.fit() ile öğrenir
# ═══════════════════════════════════════════════════════════════════════════════

class GBMModel:
    """LightGBM → sklearn GBM → numpy AdaBoost (öncelik sırasıyla)"""

    def __init__(self):
        self.clf = None
        self.is_fitted = False
        self._backend = self._detect()

    def _detect(self):
        try:
            import lightgbm; return "lgbm"
        except ImportError: pass
        try:
            from sklearn.ensemble import GradientBoostingClassifier; return "skgbm"
        except ImportError: pass
        return "numpy"

    def fit(self, X, y):
        if len(X) < 10: return self
        try:
            if self._backend == "lgbm":
                import lightgbm as lgb
                import warnings
                feat_names = [f"f{i}" for i in range(X.shape[1])]
                self.clf = lgb.LGBMClassifier(
                    n_estimators=300, learning_rate=0.03, max_depth=6,
                    num_leaves=31, min_child_samples=5, subsample=0.8,
                    colsample_bytree=0.8, class_weight="balanced",
                    random_state=42, verbose=-1, n_jobs=-1)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.clf.fit(X, y)
                self._feat_names = feat_names
                logger.info(f"LightGBM fit — {len(X)} sample")

            elif self._backend == "skgbm":
                from sklearn.ensemble import GradientBoostingClassifier
                self.clf = GradientBoostingClassifier(
                    n_estimators=150, learning_rate=0.08, max_depth=4,
                    random_state=42)
                self.clf.fit(X, y)
                logger.info(f"sklearn GBM fit — {len(X)} sample")

            else:
                self._fit_numpy_ada(X, y)

            self.is_fitted = True
        except Exception as e:
            logger.error(f"GBM fit: {e}")
            self._fit_numpy_ada(X, y)
        return self

    def _fit_numpy_ada(self, X, y):
        """Gerçek AdaBoost (numpy, sıfırdan yazılmış)"""
        n = len(X); w = np.ones(n)/n
        self._stumps, self._alphas = [], []
        for _ in range(60):
            # En iyi decision stump bul
            bf, bt, bd, berr = 0, 0.0, 1, float('inf')
            fset = np.random.choice(X.shape[1], min(8,X.shape[1]), replace=False)
            for f in fset:
                for t in np.percentile(X[:,f], [20,40,60,80]):
                    for d in [1,-1]:
                        pred = np.where(X[:,f]*d > t*d, 1, -1)
                        yp = np.sign(y + 1e-6)  # binary proxy
                        err = float(np.dot(w, pred != yp))
                        if err < berr: berr,bf,bt,bd = err,f,t,d
            eps = max(berr, 1e-10)
            if eps >= 0.5: break
            alpha = 0.5 * np.log((1-eps)/eps)
            pred = np.where(X[:,bf]*bd > bt*bd, 1, -1)
            yp = np.sign(y + 1e-6)
            w *= np.exp(-alpha * yp * pred)
            w /= w.sum()
            self._stumps.append((bf, bt, bd))
            self._alphas.append(alpha)
        self.is_fitted = True
        logger.info(f"Numpy AdaBoost fit — {len(self._stumps)} stumps, {n} sample")

    def predict_proba(self, x: np.ndarray) -> dict:
        default = {"LONG":0.25,"SHORT":0.25,"HOLD":0.25,"WAIT":0.25}
        if not self.is_fitted: return default
        try:
            if self._backend in ("lgbm","skgbm") and self.clf is not None:
                p = self.clf.predict_proba(x.reshape(1,-1))[0]
                cls = list(self.clf.classes_)
                r = {"LONG":0.1,"SHORT":0.1,"HOLD":0.1,"WAIT":0.05}
                for i,c in enumerate(cls):
                    if c==1:  r["LONG"]=float(p[i])
                    elif c==-1: r["SHORT"]=float(p[i])
                    elif c==0: r["HOLD"]=float(p[i])
                t = sum(r.values()); return {k:v/t for k,v in r.items()}
        except Exception: pass
        if hasattr(self,"_stumps") and self._stumps:
            score = sum(a*(1 if x[f]*d>t*d else -1) for (f,t,d),a in zip(self._stumps,self._alphas))
            pl = float(1/(1+np.exp(-score*2))); ps = float(1/(1+np.exp(score*2)))
            ph = max(0, 1-pl-ps+0.1); pw = 0.05
            tot = pl+ps+ph+pw
            return {"LONG":pl/tot,"SHORT":ps/tot,"HOLD":ph/tot,"WAIT":pw/tot}
        return default


class RFModel:
    """Random Forest — sklearn varsa kullanır, yoksa numpy ile"""

    def __init__(self):
        self.clf = None
        self.is_fitted = False
        try:
            from sklearn.ensemble import RandomForestClassifier
            self._backend = "sklearn"
        except ImportError:
            self._backend = "numpy"

    def fit(self, X, y):
        if len(X) < 10: return self
        try:
            if self._backend == "sklearn":
                from sklearn.ensemble import RandomForestClassifier
                self.clf = RandomForestClassifier(
                    n_estimators=100, max_depth=8, min_samples_leaf=3,
                    class_weight="balanced", random_state=42, n_jobs=-1)
                self.clf.fit(X, y)
                logger.info(f"sklearn RF fit — {len(X)} sample")
            else:
                self._fit_numpy_rf(X, y)
            self.is_fitted = True
        except Exception as e:
            logger.error(f"RF fit: {e}")
            self._fit_numpy_rf(X, y)
        return self

    def _fit_numpy_rf(self, X, y):
        self._trees = []
        nf = max(1, int(np.sqrt(X.shape[1])))
        for _ in range(40):
            idx = np.random.choice(len(X), len(X), replace=True)
            Xb, yb = X[idx], y[idx]
            feats = np.random.choice(X.shape[1], nf, replace=False)
            self._trees.append((self._build_tree(Xb[:,feats], yb, depth=4), feats))
        self.is_fitted = True
        logger.info(f"Numpy RF fit — 40 trees, {len(X)} sample")

    def _build_tree(self, X, y, depth):
        if depth==0 or len(np.unique(y))==1 or len(X)<4:
            vals,cnt=np.unique(y,return_counts=True); return ("L",int(vals[cnt.argmax()]))
        bf,bt,bg=0,0.0,-1
        for f in range(X.shape[1]):
            for t in np.percentile(X[:,f],[25,50,75]):
                lm=X[:,f]<=t
                if lm.sum()<2 or (~lm).sum()<2: continue
                g=self._gini(y)-len(y[lm])/len(y)*self._gini(y[lm])-len(y[~lm])/len(y)*self._gini(y[~lm])
                if g>bg: bg,bf,bt=g,f,t
        lm=X[:,bf]<=bt
        return ("S",bf,bt,self._build_tree(X[lm],y[lm],depth-1),self._build_tree(X[~lm],y[~lm],depth-1))

    def _gini(self,y):
        _,c=np.unique(y,return_counts=True); p=c/len(y); return 1-float(np.sum(p**2))

    def _pred_tree(self, node, x):
        if node[0]=="L": return node[1]
        _,f,t,l,r = node
        return self._pred_tree(l if x[f]<=t else r, x)

    def predict_proba(self, x: np.ndarray) -> dict:
        default = {"LONG":0.25,"SHORT":0.25,"HOLD":0.25,"WAIT":0.25}
        if not self.is_fitted: return default
        try:
            if self._backend=="sklearn" and self.clf is not None:
                p=self.clf.predict_proba(x.reshape(1,-1))[0]
                cls=list(self.clf.classes_)
                r={"LONG":0.1,"SHORT":0.1,"HOLD":0.1,"WAIT":0.05}
                for i,c in enumerate(cls):
                    if c==1: r["LONG"]=float(p[i])
                    elif c==-1: r["SHORT"]=float(p[i])
                    elif c==0: r["HOLD"]=float(p[i])
                t=sum(r.values()); return {k:v/t for k,v in r.items()}
        except Exception: pass
        if hasattr(self,"_trees"):
            votes={-1:0,0:0,1:0}
            for (tree,feats) in self._trees:
                votes[self._pred_tree(tree,x[feats])] += 1
            tot=sum(votes.values())
            return {"LONG":votes[1]/tot,"SHORT":votes[-1]/tot,"HOLD":votes[0]/tot,"WAIT":0.05}
        return default


# ═══════════════════════════════════════════════════════════════════════════════
# 4. WALK-FORWARD VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def walk_forward_validate(X, y, n_splits=4) -> dict:
    """Zaman sıralı k-fold — gelecek sızmaz"""
    if X is None or len(X) < 30: return {"accuracy":0,"f1":0,"n_samples":0}
    n = len(X); sz = n//(n_splits+1)
    accs, f1s = [], []
    for i in range(1, n_splits+1):
        te = sz*(i+1)
        Xtr,ytr = X[:sz*i], y[:sz*i]
        Xte,yte = X[sz*i:te], y[sz*i:te]
        if len(Xtr)<5 or len(Xte)<2: continue
        m = GBMModel(); m.fit(Xtr, ytr)
        lmap={"LONG":1,"SHORT":-1,"HOLD":0,"WAIT":0}
        preds=np.array([lmap[max(m.predict_proba(x),key=m.predict_proba(x).get)] for x in Xte])
        accs.append(float(np.mean(preds==yte)))
        # macro F1
        f1_vals=[]
        for cls in [-1,0,1]:
            tp=np.sum((preds==cls)&(yte==cls)); fp=np.sum((preds==cls)&(yte!=cls))
            fn=np.sum((preds!=cls)&(yte==cls))
            pr=tp/(tp+fp+1e-10); re=tp/(tp+fn+1e-10)
            f1_vals.append(2*pr*re/(pr+re+1e-10))
        f1s.append(float(np.mean(f1_vals)))
    return {
        "accuracy": round(float(np.mean(accs))*100,1) if accs else 0,
        "f1": round(float(np.mean(f1s)),3) if f1s else 0,
        "n_samples": n, "n_folds": len(accs)
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 5. BACKTEST
# ═══════════════════════════════════════════════════════════════════════════════

def backtest(c: np.ndarray, signals: list, sl=0.025, tp=0.045, fee=0.0004) -> dict:
    """
    signals: [(bar_idx, "LONG"/"SHORT"), ...]
    Returns: roi, win_rate, max_drawdown, sharpe, n_trades
    """
    equity=1.0; eq_curve=[1.0]; returns=[]; wins=0; ntrades=0
    for bar_idx, sig in signals:
        if sig not in ("LONG","SHORT") or bar_idx+1>=len(c): continue
        e,ex=c[bar_idx],c[bar_idx+1]
        r=(ex-e)/e; r=(-r if sig=="SHORT" else r)
        r=max(-sl, min(tp, r)) - fee*2
        equity*=(1+r); eq_curve.append(equity); returns.append(r)
        ntrades+=1
        if r>0: wins+=1
    if ntrades==0: return {"roi":0,"win_rate":0,"max_drawdown":0,"sharpe":0,"n_trades":0}
    eq=np.array(eq_curve); pk=np.maximum.accumulate(eq)
    dd=float(np.max((pk-eq)/(pk+1e-10)))*100
    ret=np.array(returns)
    sh=float(ret.mean()/ret.std()*np.sqrt(2920)) if ret.std()>0 else 0.0
    return {"roi":round((equity-1)*100,2),"win_rate":round(wins/ntrades*100,1),
            "max_drawdown":round(dd,2),"sharpe":round(sh,3),"n_trades":ntrades}


# ═══════════════════════════════════════════════════════════════════════════════
# 6. ANA ML ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class MLEngine:
    """
    Ana orchestrator:
    ✅ train()      — gerçek model.fit()
    ✅ predict()    — öğrenilmiş modelden tahmin
    ✅ backtest()   — tarihsel performans
    ✅ add_feedback() — feedback loop
    ✅ get_info()   — şeffaf istatistikler
    """

    def __init__(self):
        self.gbm = GBMModel()
        self.rf  = RFModel()
        self._trained = False
        self._wf: dict = {}
        self._bt: dict = {}
        self._train_log: list = []
        self._feedback: deque = deque(maxlen=500)
        self._pred_count = 0
        self._last_retrain = 0.0
        logger.info("MLEngine — Gerçek ML sistemi hazır (GBM + RF)")

    # ── EĞİTİM ────────────────────────────────────────────────────────────────
    def train(self, klines: dict, symbol="GLOBAL") -> dict:
        t0 = time.time()
        X, y = FeatureBuilder.build_dataset(klines, lookahead=5, threshold=0.008)
        if X is None:
            return {"success":False,"reason":"insufficient_data","symbol":symbol}

        unique,cnt = np.unique(y, return_counts=True)
        split = int(len(X)*0.8)
        Xtr,Xte = X[:split], X[split:]
        ytr,yte = y[:split], y[split:]

        self.gbm.fit(Xtr, ytr)
        self.rf.fit(Xtr, ytr)
        self._trained = True

        # Test accuracy
        lmap = {"LONG":1,"SHORT":-1,"HOLD":0,"WAIT":0}
        preds = np.array([lmap[max(self.gbm.predict_proba(x),key=self.gbm.predict_proba(x).get)] for x in Xte])
        acc = float(np.mean(preds==yte))*100 if len(yte)>0 else 0

        # Walk-forward
        self._wf = walk_forward_validate(X, y, n_splits=4)

        elapsed = time.time()-t0
        rec = {
            "success":True,"symbol":symbol,
            "n_samples":len(X),"n_train":len(Xtr),"n_test":len(Xte),
            "class_dist":{int(k):int(v) for k,v in zip(unique,cnt)},
            "test_accuracy":round(acc,1),
            "wf_accuracy":self._wf["accuracy"],"wf_f1":self._wf["f1"],
            "train_time_s":round(elapsed,2),
            "trained_at":datetime.utcnow().isoformat(),
        }
        self._train_log.append(rec)
        if len(self._train_log)>10: self._train_log.pop(0)
        self._last_retrain = time.time()
        logger.info(
            f"✅ Eğitim tamamlandı [{symbol}] | {len(X)} sample | "
            f"test_acc={acc:.1f}% | wf_acc={self._wf['accuracy']}% | "
            f"F1={self._wf['f1']} | {elapsed:.1f}s"
        )
        return rec

    def run_backtest(self, klines: dict) -> dict:
        """Modelin sinyalleriyle backtest çalıştır"""
        X, y = FeatureBuilder.build_dataset(klines, lookahead=5, threshold=0.008)
        if X is None: return {}
        c = np.asarray(klines["close"], dtype=np.float64)
        MIN=40; sigs=[]
        for i,(x,_) in enumerate(zip(X,y)):
            p=self._ensemble(x)
            sigs.append((MIN+i, max(p,key=p.get)))
        self._bt = backtest(c, sigs)
        logger.info(f"Backtest → ROI:{self._bt['roi']}% WR:{self._bt['win_rate']}% Sharpe:{self._bt['sharpe']}")
        return self._bt

    # ── FEEDBACK LOOP ─────────────────────────────────────────────────────────
    def add_feedback(self, features: np.ndarray, outcome: str):
        """Canlı trade sonucu → feedback buffer"""
        lmap={"LONG":1,"SHORT":-1,"HOLD":0,"WAIT":0}
        self._feedback.append((features, lmap.get(outcome,0)))
        # Her 100 sample'da retrain (5dk cooldown)
        if len(self._feedback)>=100 and time.time()-self._last_retrain>300:
            Xf=np.array([f for f,_ in self._feedback],dtype=np.float32)
            yf=np.array([l for _,l in self._feedback],dtype=np.int32)
            self.gbm.fit(Xf,yf); self.rf.fit(Xf,yf)
            self._last_retrain=time.time()
            logger.info(f"🔁 Feedback retrain — {len(Xf)} sample")

    # ── TAHMİN ────────────────────────────────────────────────────────────────
    def predict(self, symbol: str, klines: Optional[dict], current_price: float) -> dict:
        try:
            if klines is None or len(klines.get("close",[])) < 30:
                return self._fallback()

            feat = FeatureBuilder.build(klines)
            if feat is None: return self._fallback()

            # İlk çalışmada bu kline'ı kullanarak eğit
            if not self._trained:
                self.train(klines, symbol)

            proba = self._ensemble(feat)
            sig   = max(proba, key=proba.get)
            conf  = round(proba[sig]*100, 1)
            if conf < 54: sig = "WAIT"

            lev = 20 if conf>85 else 15 if conf>75 else 10 if conf>65 else 5

            self._pred_count += 1
            self._feedback.append((feat, {"LONG":1,"SHORT":-1,"HOLD":0,"WAIT":0}.get(sig,0)))

            return {
                "signal": sig, "confidence": conf,
                "indicators": self._active_inds(klines)[:4],
                "model": "GBM+RF (real fit)",
                "leverage": lev,
                "gbm_signal": max(self.gbm.predict_proba(feat),key=self.gbm.predict_proba(feat).get),
                "rf_signal":  max(self.rf.predict_proba(feat), key=self.rf.predict_proba(feat).get),
                "ensemble_proba": {k:round(v*100,1) for k,v in proba.items()},
                "data_quality": "real",
                "model_trained": self._trained,
                "wf_accuracy": self._wf.get("accuracy",0),
                "backtest_roi": self._bt.get("roi",0),
                "backtest_sharpe": self._bt.get("sharpe",0),
            }
        except Exception as e:
            logger.error(f"predict [{symbol}]: {e}")
            return self._fallback()

    def _ensemble(self, feat: np.ndarray) -> dict:
        gp = self.gbm.predict_proba(feat)
        rp = self.rf.predict_proba(feat)
        out = {k: gp[k]*0.6 + rp[k]*0.4 for k in ["LONG","SHORT","HOLD","WAIT"]}
        t=sum(out.values()); return {k:v/t for k,v in out.items()}

    def _active_inds(self, klines: dict) -> list:
        c=np.asarray(klines["close"],dtype=np.float64)
        h=np.asarray(klines["high"],dtype=np.float64)
        lo=np.asarray(klines["low"],dtype=np.float64)
        ti=Indicators(); inds=[]
        r=ti.rsi(c)
        if r<35: inds.append(f"RSI:{r:.0f}↓OS")
        elif r>65: inds.append(f"RSI:{r:.0f}↑OB")
        else: inds.append(f"RSI:{r:.0f}")
        _,_,mh=ti.macd(c)
        inds.append("MACD+" if mh>0 else "MACD-")
        bbu,_,bbl=ti.bb(c)
        bp=(c[-1]-bbl)/(bbu-bbl+1e-10)
        if bp<0.2: inds.append("BB-low")
        elif bp>0.8: inds.append("BB-high")
        else: inds.append(f"BB:{bp:.2f}")
        e9,e21=ti.ema(c,9)[-1],ti.ema(c,21)[-1]
        inds.append("EMA↑" if e9>e21 else "EMA↓")
        m=ti.mfi(h,lo,c,np.asarray(klines["volume"],dtype=np.float64))
        if m<30: inds.append(f"MFI:{m:.0f}↓")
        elif m>70: inds.append(f"MFI:{m:.0f}↑")
        return inds

    def _fallback(self) -> dict:
        return {"signal":"WAIT","confidence":50.0,
                "indicators":["Veri bekleniyor"],"model":"GBM+RF","leverage":5,
                "gbm_signal":"WAIT","rf_signal":"WAIT",
                "ensemble_proba":{"LONG":25,"SHORT":25,"HOLD":25,"WAIT":25},
                "data_quality":"insufficient","model_trained":self._trained,
                "wf_accuracy":self._wf.get("accuracy",0),
                "backtest_roi":self._bt.get("roi",0),"backtest_sharpe":self._bt.get("sharpe",0)}

    # ── META ──────────────────────────────────────────────────────────────────
    def get_accuracy(self) -> float:
        return self._wf.get("accuracy", 0.0)

    def get_info(self) -> dict:
        gbm_type = self.gbm._backend
        rf_type = self.rf._backend
        return {
            "models": [
                f"GradientBoosting ({gbm_type})",
                f"RandomForest ({rf_type})"
            ],
            "ensemble": {"gbm":0.6,"rf":0.4},
            "n_features": FeatureBuilder.N_FEATURES,
            "features": [
                "RSI","StochRSI","MFI","Williams_R",
                "MACD_line","MACD_sig","MACD_hist","EMA9_21","EMA50_dist",
                "BB_pct","BB_width","ATR_pct","Vol_ratio","OBV_trend",
                "Lag1","Lag2","Lag3","Ret5","Ret10","Ret20",
                "Std5","Std20","Pos_HL","Gap_HL",
                "Mom_acc","Body","HL_range","Upper_wick"
            ],
            "label_method": "lookahead=5bars, threshold=0.8%",
            "validation": "Walk-forward (4-fold, no lookahead)",
            "is_trained": self._trained,
            "wf_result": self._wf,
            "backtest": self._bt,
            "train_log": self._train_log[-10:],
            "feedback_buffer": len(self._feedback),
            "predictions_made": self._pred_count,
            "last_retrain": datetime.fromtimestamp(self._last_retrain, tz=__import__("datetime").timezone.utc).isoformat() if self._last_retrain>0 else None,
            "confidence_threshold": 54,
            "signals": ["LONG","SHORT","HOLD","WAIT"],
        }
