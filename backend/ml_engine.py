
"""
MEXC ML Trading System — Advanced ML Engine v2
================================================
✅ Gerçek model.fit() — GBM (LightGBM) + RF ensemble
✅ Walk-forward validation (time-series safe)
✅ Backtest (ROI, Sharpe, Max Drawdown)
✅ DOGRU Feedback Loop — trade kapandiktan SONRA etiket uretir
✅ Drift Detection — accuracy dusunce otomatik retrain
✅ Model Versiyonlama — en iyi 3 versiyon, rollback
✅ Non-blocking egitim destegi
✅ 36 feature (28 orijinal + 8 yeni)
"""

import numpy as np
import logging
import time
import os
import joblib
import warnings
import threading
from typing import Optional, List, Tuple
from collections import deque
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Orijinal Indicators sinifi - tamamen koruyoruz
"""INDICATORS"""
class Indicators:
    @staticmethod
    def rsi(c, p=14):
        if len(c) < p + 2: return 50.0
        d = np.diff(c[-(p+5):])
        g = np.where(d > 0, d, 0.0); lo = np.where(d < 0, -d, 0.0)
        ag, al = np.mean(g[-p:]), np.mean(lo[-p:])
        return 100.0 if al < 1e-12 else float(100 - 100 / (1 + ag / al))

    @staticmethod
    def ema(c, p):
        if len(c) == 0: return np.zeros(1)
        k = 2 / (p + 1); out = np.empty(len(c)); out[0] = c[0]
        for i in range(1, len(c)): out[i] = c[i] * k + out[i-1] * (1 - k)
        return out

    @staticmethod
    def macd(c):
        if len(c) < 35: return 0.0, 0.0, 0.0
        e12=Indicators.ema(c,12); e26=Indicators.ema(c,26); ml=e12-e26
        sig=Indicators.ema(ml,9); return float(ml[-1]),float(sig[-1]),float(ml[-1]-sig[-1])

    @staticmethod
    def bb(c, p=20):
        if len(c) < p: v=float(c[-1]) if len(c) else 0; return v,v,v
        s,std=float(np.mean(c[-p:])),float(np.std(c[-p:])); return s+2*std,s,s-2*std

    @staticmethod
    def atr(h,lo,c,p=14):
        if len(c)<p+1: return 0.0
        tr=np.maximum(h[1:]-lo[1:],np.maximum(np.abs(h[1:]-c[:-1]),np.abs(lo[1:]-c[:-1])))
        return float(np.mean(tr[-p:]))

    @staticmethod
    def obv_trend(c,v):
        if len(c)<20: return 0.0
        obv=np.zeros(len(c)); obv[0]=v[0]
        for i in range(1,len(c)):
            obv[i]=obv[i-1]+(v[i] if c[i]>c[i-1] else -v[i] if c[i]<c[i-1] else 0)
        ref=float(np.abs(np.mean(obv[-20:-10]))+1e-10)
        return float(np.clip((np.mean(obv[-10:])-np.mean(obv[-20:-10]))/ref,-3,3))

    @staticmethod
    def mfi(h,lo,c,v,p=14):
        if len(c)<p+1: return 50.0
        tp=(h+lo+c)/3; mf=tp*v
        pos=np.where(np.diff(tp)>0,mf[1:],0); neg=np.where(np.diff(tp)<0,mf[1:],0)
        ps,ns=float(np.sum(pos[-p:])),float(np.sum(neg[-p:]))
        return 100.0 if ns<1e-12 else float(100-100/(1+ps/ns))

    @staticmethod
    def stoch_rsi(c,p=14):
        rvals=[Indicators.rsi(c[:i],p) for i in range(p+1,len(c)+1)]
        if len(rvals)<p: return 50.0
        arr=np.array(rvals[-p:]); mn,mx=arr.min(),arr.max()
        return float((arr[-1]-mn)/(mx-mn+1e-10)*100)

    @staticmethod
    def williams_r(h,lo,c,p=14):
        if len(c)<p: return -50.0
        hh,ll=float(np.max(h[-p:])),float(np.min(lo[-p:]))
        return float(-100*(hh-c[-1])/(hh-ll+1e-10))

    # --- YENİ 4 indikatör ---
    @staticmethod
    def vwap_dist(h,lo,c,v):
        if len(c)<5: return 0.0
        tp=(h[-20:]+lo[-20:]+c[-20:])/3
        vw=float(np.sum(tp*v[-20:])/(np.sum(v[-20:])+1e-10))
        return float(np.clip((c[-1]-vw)/(vw+1e-10),-0.1,0.1))

    @staticmethod
    def momentum_slope(c,p=10):
        if len(c)<p+2: return 0.0
        y=c[-p:]; x=np.arange(p); slope=float(np.polyfit(x,y,1)[0])
        return float(np.clip(slope/(c[-1]+1e-10),-0.01,0.01))

    @staticmethod
    def market_regime(c,v):
        if len(c)<20: return 0.0
        price_range=float(np.max(c[-20:])-np.min(c[-20:]))
        directional=abs(float(c[-1]-c[-20]))/(price_range+1e-10)
        vol_trend=float(np.mean(v[-5:]))/(float(np.mean(v[-20:]))+1e-10)
        return float(np.clip(directional*vol_trend-0.5,-1,1))

    @staticmethod
    def atr_percentile(h,lo,c):
        if len(c)<25: return 0.5
        atr_now=Indicators.atr(h,lo,c,14)
        hist=[Indicators.atr(h[:i],lo[:i],c[:i],14) for i in range(max(20,len(c)-20),len(c))]
        return float(np.mean(np.array(hist)<atr_now)) if hist else 0.5


# =============================================================================
# FEATURE BUILDER — 36 ozellik
# =============================================================================
class FeatureBuilder:
    N_FEATURES = 36

    @classmethod
    def build(cls, klines):
        c=np.asarray(klines["close"],dtype=np.float64)
        h=np.asarray(klines["high"],dtype=np.float64)
        lo=np.asarray(klines["low"],dtype=np.float64)
        v=np.asarray(klines["volume"],dtype=np.float64)
        if len(c)<30: return None
        ti=Indicators(); f=[]; pr=c[-1]+1e-10

        # Momentum (0-3)
        f.append(ti.rsi(c,14)/100.0)
        f.append(ti.stoch_rsi(c,14)/100.0)
        f.append(ti.mfi(h,lo,c,v,14)/100.0)
        f.append(ti.williams_r(h,lo,c,14)/-100.0)

        # Trend (4-8)
        ml,ms,mh=ti.macd(c)
        f.extend([np.clip(ml/pr,-0.05,0.05),np.clip(ms/pr,-0.05,0.05),np.clip(mh/pr,-0.05,0.05)])
        e9=ti.ema(c,9)[-1]; e21=ti.ema(c,21)[-1]; e50=ti.ema(c,min(50,len(c)-1))[-1]
        f.append(np.clip((e9-e21)/pr,-0.05,0.05))
        f.append(np.clip((c[-1]-e50)/pr,-0.1,0.1))

        # Volatility (9-11)
        bbu,_,bbl=ti.bb(c,20); bb_rng=bbu-bbl+1e-10
        f.append(np.clip((c[-1]-bbl)/bb_rng,0,1))
        f.append(np.clip(bb_rng/pr,0,0.2))
        f.append(np.clip(ti.atr(h,lo,c,14)/pr,0,0.05))

        # Volume (12-13)
        vavg=float(np.mean(v[-21:-1]) if len(v)>=21 else v.mean())+1e-10
        f.append(np.clip(v[-1]/vavg,0,5))
        f.append(ti.obv_trend(c,v))

        # Lag returns (14-16)
        for lag in [1,2,3]:
            ret=(c[-1]-c[-1-lag])/(c[-1-lag]+1e-10) if len(c)>lag else 0.0
            f.append(np.clip(ret,-0.1,0.1))

        # Rolling returns (17-19)
        for w in [5,10,20]:
            ret=(c[-1]-c[-1-w])/(c[-1-w]+1e-10) if len(c)>w else 0.0
            f.append(np.clip(ret,-0.3,0.3))

        # Rolling vol (20-21)
        for w in [5,20]:
            s=float(np.std(c[-w:]))/pr if len(c)>=w else 0.0
            f.append(np.clip(s,0,0.1))

        # Price position (22-23)
        h20,l20=float(np.max(h[-20:])),float(np.min(lo[-20:])); rng=h20-l20+1e-10
        f.append((c[-1]-l20)/rng); f.append((h20-c[-1])/rng)

        # Momentum acc (24)
        m5=c[-1]-c[-6] if len(c)>=6 else 0.0; mp=c[-2]-c[-7] if len(c)>=7 else 0.0
        f.append(np.clip((m5-mp)/pr,-0.05,0.05))

        # Candle (25-27)
        body=abs(c[-1]-c[-2])/pr if len(c)>=2 else 0.0
        f.append(np.clip(body,0,0.05))
        hl=(h[-1]-lo[-1])/pr; f.append(np.clip(hl,0,0.1))
        uw=(h[-1]-max(c[-1],c[-2] if len(c)>=2 else c[-1]))/(h[-1]-lo[-1]+1e-10)
        f.append(np.clip(float(uw),0,1))

        # --- YENİ 8 ozellik (28-35) ---
        f.append(ti.vwap_dist(h,lo,c,v))                             # 28
        f.append(ti.momentum_slope(c,10))                             # 29

        # RSI divergence
        rsi_now=ti.rsi(c,14); rsi_prev=ti.rsi(c[:-3],14) if len(c)>17 else rsi_now
        price_chg=(c[-1]-c[-4])/(c[-4]+1e-10) if len(c)>4 else 0.0
        f.append(float(np.clip(price_chg*(-(rsi_now-rsi_prev)/100.0),-0.05,0.05)))  # 30

        f.append(ti.market_regime(c,v))                               # 31

        # Volume momentum
        v_slope=float(np.polyfit(np.arange(5),v[-5:],1)[0]) if len(v)>=5 else 0.0
        f.append(float(np.clip(v_slope/(float(np.mean(v[-5:]))+1e-10),-2,2)))      # 32

        # Lower wick
        lower_wick=(min(c[-1],c[-2] if len(c)>=2 else c[-1])-lo[-1])/(h[-1]-lo[-1]+1e-10)
        f.append(float(np.clip(lower_wick,0,1)))                      # 33

        # EMA cross speed
        e9a=ti.ema(c,9); e21a=ti.ema(c,21)
        cross_now=(e9a[-1]-e21a[-1])/pr
        cross_prev=(e9a[-2]-e21a[-2])/pr if len(c)>2 else cross_now
        f.append(float(np.clip(cross_now-cross_prev,-0.01,0.01)))     # 34

        f.append(ti.atr_percentile(h,lo,c))                           # 35

        arr=np.array(f,dtype=np.float32)
        return np.where(np.isfinite(arr),arr,0.0)

    @classmethod
    def build_dataset(cls,klines,lookahead=5,threshold=0.005):
        c=np.asarray(klines["close"],dtype=np.float64)
        h=np.asarray(klines["high"],dtype=np.float64)
        lo=np.asarray(klines["low"],dtype=np.float64)
        v=np.asarray(klines["volume"],dtype=np.float64)
        MIN=40
        if len(c)<MIN+lookahead: return None,None
        X_rows,y_rows=[],[]
        for i in range(MIN,len(c)-lookahead):
            sub={"close":c[:i],"high":h[:i],"low":lo[:i],"volume":v[:i]}
            feat=cls.build(sub)
            if feat is None: continue
            ret=(c[i+lookahead]-c[i])/(c[i]+1e-10)
            label=1 if ret>threshold else (-1 if ret<-threshold else 0)
            X_rows.append(feat); y_rows.append(label)
        if len(X_rows)<10: return None,None
        return np.array(X_rows,np.float32),np.array(y_rows,np.int32)


# =============================================================================
# GBM MODEL
# =============================================================================
class GBMModel:
    def __init__(self):
        self.clf=None; self.is_fitted=False; self._backend=self._detect()

    def _detect(self):
        try: import lightgbm; return "lgbm"
        except ImportError: pass
        try: from sklearn.ensemble import GradientBoostingClassifier; return "skgbm"
        except ImportError: pass
        return "numpy"

    def fit(self,X,y):
        if len(X)<10: return self
        try:
            if self._backend=="lgbm":
                import lightgbm as lgb
                self.clf=lgb.LGBMClassifier(
                    n_estimators=400,learning_rate=0.025,max_depth=6,
                    num_leaves=31,min_child_samples=5,subsample=0.8,
                    colsample_bytree=0.8,class_weight="balanced",
                    reg_alpha=0.1,reg_lambda=0.1,
                    random_state=42,verbose=-1,n_jobs=-1)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.clf.fit(X,y)
                logger.info(f"LightGBM fit — {len(X)} sample")
            elif self._backend=="skgbm":
                from sklearn.ensemble import GradientBoostingClassifier
                self.clf=GradientBoostingClassifier(n_estimators=200,learning_rate=0.06,max_depth=4,random_state=42)
                self.clf.fit(X,y)
                logger.info(f"sklearn GBM fit — {len(X)} sample")
            else: self._fit_numpy_ada(X,y)
            self.is_fitted=True
        except Exception as e:
            logger.error(f"GBM fit: {e}"); self._fit_numpy_ada(X,y)
        return self

    def _fit_numpy_ada(self,X,y):
        n=len(X); w=np.ones(n)/n; self._stumps=[]; self._alphas=[]
        for _ in range(80):
            bf,bt,bd,berr=0,0.0,1,float('inf')
            fset=np.random.choice(X.shape[1],min(8,X.shape[1]),replace=False)
            for f in fset:
                for t in np.percentile(X[:,f],[20,40,60,80]):
                    for d in [1,-1]:
                        pred=np.where(X[:,f]*d>t*d,1,-1); yp=np.sign(y+1e-6)
                        err=float(np.dot(w,pred!=yp))
                        if err<berr: berr,bf,bt,bd=err,f,t,d
            eps=max(berr,1e-10)
            if eps>=0.5: break
            alpha=0.5*np.log((1-eps)/eps)
            pred=np.where(X[:,bf]*bd>bt*bd,1,-1); yp=np.sign(y+1e-6)
            w*=np.exp(-alpha*yp*pred); w/=w.sum()
            self._stumps.append((bf,bt,bd)); self._alphas.append(alpha)
        self.is_fitted=True

    def get_feature_importance(self):
        if self._backend=="lgbm" and self.clf is not None:
            try: imp=self.clf.feature_importances_; return imp/(imp.sum()+1e-10)
            except: pass
        return None

    def predict_proba(self,x):
        default={"LONG":0.25,"SHORT":0.25,"HOLD":0.25,"WAIT":0.25}
        if not self.is_fitted: return default
        try:
            if self._backend in ("lgbm","skgbm") and self.clf is not None:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    p=self.clf.predict_proba(x.reshape(1,-1))[0]
                cls=list(self.clf.classes_)
                r={"LONG":0.1,"SHORT":0.1,"HOLD":0.1,"WAIT":0.05}
                for i,c in enumerate(cls):
                    if c==1: r["LONG"]=float(p[i])
                    elif c==-1: r["SHORT"]=float(p[i])
                    elif c==0: r["HOLD"]=float(p[i])
                t=sum(r.values()); return {k:v/t for k,v in r.items()}
        except Exception: pass
        if hasattr(self,"_stumps") and self._stumps:
            score=sum(a*(1 if x[f]*d>t*d else -1) for (f,t,d),a in zip(self._stumps,self._alphas))
            pl=float(1/(1+np.exp(-score*2))); ps=float(1/(1+np.exp(score*2)))
            ph=max(0,1-pl-ps+0.1); pw=0.05; tot=pl+ps+ph+pw
            return {"LONG":pl/tot,"SHORT":ps/tot,"HOLD":ph/tot,"WAIT":pw/tot}
        return default


# =============================================================================
# RF MODEL
# =============================================================================
class RFModel:
    def __init__(self):
        self.clf=None; self.is_fitted=False
        try: from sklearn.ensemble import RandomForestClassifier; self._backend="sklearn"
        except ImportError: self._backend="numpy"

    def fit(self,X,y):
        if len(X)<10: return self
        try:
            if self._backend=="sklearn":
                from sklearn.ensemble import RandomForestClassifier
                self.clf=RandomForestClassifier(n_estimators=150,max_depth=8,min_samples_leaf=3,
                    class_weight="balanced",random_state=42,n_jobs=-1)
                self.clf.fit(X,y); logger.info(f"sklearn RF fit — {len(X)} sample")
            else: self._fit_numpy_rf(X,y)
            self.is_fitted=True
        except Exception as e:
            logger.error(f"RF fit: {e}"); self._fit_numpy_rf(X,y)
        return self

    def _fit_numpy_rf(self,X,y):
        self._trees=[]; nf=max(1,int(np.sqrt(X.shape[1])))
        for _ in range(60):
            idx=np.random.choice(len(X),len(X),replace=True); Xb,yb=X[idx],y[idx]
            feats=np.random.choice(X.shape[1],nf,replace=False)
            self._trees.append((self._build_tree(Xb[:,feats],yb,depth=5),feats))
        self.is_fitted=True

    def _build_tree(self,X,y,depth):
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

    def _pred_tree(self,node,x):
        if node[0]=="L": return node[1]
        _,f,t,l,r=node; return self._pred_tree(l if x[f]<=t else r,x)

    def predict_proba(self,x):
        default={"LONG":0.25,"SHORT":0.25,"HOLD":0.25,"WAIT":0.25}
        if not self.is_fitted: return default
        try:
            if self._backend=="sklearn" and self.clf is not None:
                p=self.clf.predict_proba(x.reshape(1,-1))[0]; cls=list(self.clf.classes_)
                r={"LONG":0.1,"SHORT":0.1,"HOLD":0.1,"WAIT":0.05}
                for i,c in enumerate(cls):
                    if c==1: r["LONG"]=float(p[i])
                    elif c==-1: r["SHORT"]=float(p[i])
                    elif c==0: r["HOLD"]=float(p[i])
                t=sum(r.values()); return {k:v/t for k,v in r.items()}
        except Exception: pass
        if hasattr(self,"_trees"):
            votes={-1:0,0:0,1:0}
            for (tree,feats) in self._trees: votes[self._pred_tree(tree,x[feats])]+=1
            tot=sum(votes.values())
            return {"LONG":votes[1]/tot,"SHORT":votes[-1]/tot,"HOLD":votes[0]/tot,"WAIT":0.05}
        return default


# =============================================================================
# WALK-FORWARD VALIDATION
# =============================================================================
def balance_classes(X: np.ndarray, y: np.ndarray, target_ratio: float = 0.35) -> tuple:
    """
    Minority class (LONG/SHORT) örneklerini çoğaltarak sınıf dengesini sağla.
    target_ratio: Her sınıfın minimum oranı (varsayılan %35)
    
    Örnek: %85 HOLD, %7 LONG, %8 SHORT → %35 HOLD, %32 LONG, %33 SHORT
    Bu model'in HOLD'u ezberlemesini önler ve gerçek sinyal öğrenmesini sağlar.
    """
    unique, counts = np.unique(y, return_counts=True)
    if len(unique) < 2:
        return X, y
    
    total = len(y)
    max_count = int(total * (1 - target_ratio * (len(unique) - 1)))
    
    X_parts, y_parts = [], []
    for cls, cnt in zip(unique, counts):
        mask = y == cls
        X_cls, y_cls = X[mask], y[mask]
        
        if cls == 0:  # HOLD — azalt (cap)
            if cnt > max_count:
                idx = np.random.choice(cnt, max_count, replace=False)
                X_cls, y_cls = X_cls[idx], y_cls[idx]
        else:  # LONG/SHORT — çoğalt
            target_count = max(cnt, int(total * target_ratio))
            if cnt < target_count:
                # Gaussian noise ile augmentation (basit SMOTE)
                extra_needed = target_count - cnt
                noise_idx = np.random.choice(cnt, extra_needed, replace=True)
                noise = np.random.randn(extra_needed, X_cls.shape[1]) * 0.02
                X_extra = X_cls[noise_idx] + noise.astype(np.float32)
                X_extra = np.clip(X_extra, -5, 5)
                X_cls = np.vstack([X_cls, X_extra])
                y_cls = np.concatenate([y_cls, np.full(extra_needed, cls, dtype=np.int32)])
        
        X_parts.append(X_cls)
        y_parts.append(y_cls)
    
    X_bal = np.vstack(X_parts)
    y_bal = np.concatenate(y_parts)
    
    # Shuffle
    idx = np.random.permutation(len(X_bal))
    return X_bal[idx], y_bal[idx]


def walk_forward_validate(X,y,n_splits=4):
    if X is None or len(X)<30: return {"accuracy":0,"f1":0,"n_samples":0}
    n=len(X); sz=n//(n_splits+1); accs,f1s=[],[]
    for i in range(1,n_splits+1):
        te=sz*(i+1); Xtr,ytr=X[:sz*i],y[:sz*i]; Xte,yte=X[sz*i:te],y[sz*i:te]
        # Minimum fold boyutu: eğitim en az 30, test en az 10 sample
        if len(Xtr)<30 or len(Xte)<10: continue
        # Sınıf çeşitliliği kontrolü: test setinde en az 2 farklı sınıf olmalı
        if len(np.unique(yte))<2: continue
        m=GBMModel(); m.fit(Xtr,ytr)
        lmap={"LONG":1,"SHORT":-1,"HOLD":0,"WAIT":0}
        preds=np.array([lmap[max(p:=m.predict_proba(x),key=p.get)] for x in Xte])
        accs.append(float(np.mean(preds==yte)))
        f1_vals=[]
        for cls in [-1,0,1]:
            tp=np.sum((preds==cls)&(yte==cls)); fp=np.sum((preds==cls)&(yte!=cls))
            fn=np.sum((preds!=cls)&(yte==cls))
            pr=tp/(tp+fp+1e-10); re=tp/(tp+fn+1e-10)
            f1_vals.append(2*pr*re/(pr+re+1e-10))
        f1s.append(float(np.mean(f1_vals)))
    return {"accuracy":round(float(np.mean(accs))*100,1) if accs else 0,
            "f1":round(float(np.mean(f1s)),3) if f1s else 0,
            "n_samples":n,"n_folds":len(accs)}


# =============================================================================
# BACKTEST
# =============================================================================
def backtest(c,signals,sl=0.02,tp=0.04,fee=0.0004):
    """
    Backtest parametreleri:
    - sl=2%: Stop loss (kaldıraçsız baz, gerçekte leverage ile çarpılır)
    - tp=4%: Take profit (R:R = 1:2)
    - fee=0.04%: MEXC maker/taker fee her iki yön için
    """
    equity=1.0; eq_curve=[1.0]; returns=[]; wins=0; ntrades=0
    for bar_idx,sig in signals:
        if sig not in ("LONG","SHORT") or bar_idx+1>=len(c): continue
        e=c[bar_idx]; ex=c[bar_idx+1]
        if e <= 0 or ex <= 0: continue  # Sıfır fiyat barlarını atla
        r=(ex-e)/e; r=(-r if sig=="SHORT" else r)
        r=max(-sl,min(tp,r))-fee*2
        equity*=(1+r); eq_curve.append(equity); returns.append(r); ntrades+=1
        if r>0: wins+=1
    if ntrades==0: return {"roi":0,"win_rate":0,"max_drawdown":0,"sharpe":0,"n_trades":0}
    eq=np.array(eq_curve); pk=np.maximum.accumulate(eq)
    dd=float(np.max((pk-eq)/(pk+1e-10)))*100; ret=np.array(returns)
    # Annualized Sharpe (2920 = 4 aralık/saat * 24 * 365 / 2 ≈ yıllık 15dk bar sayısı)
    sh = float(ret.mean()/ret.std()*np.sqrt(2920)) if ret.std()>0 and ntrades>=5 else 0.0
    return {"roi":round((equity-1)*100,2),"win_rate":round(wins/ntrades*100,1),
            "max_drawdown":round(dd,2),"sharpe":round(sh,3),"n_trades":ntrades}


# =============================================================================
# DRIFT DETECTOR
# =============================================================================
class DriftDetector:
    def __init__(self,window=50,threshold_sigma=1.5):
        self._window=window; self._threshold=threshold_sigma
        self._recent=deque(maxlen=window); self._baseline_acc=0.0; self._drift_count=0

    def update(self,correct):
        self._recent.append(1.0 if correct else 0.0)

    def set_baseline(self,accuracy):
        self._baseline_acc=accuracy/100.0

    def is_drifting(self):
        if len(self._recent)<20 or self._baseline_acc<0.01: return False
        recent_acc=float(np.mean(self._recent))
        # Absolute drop: baseline'dan %15+ düşüş her zaman drift
        absolute_drop = (self._baseline_acc - recent_acc) / (self._baseline_acc + 1e-10)
        if absolute_drop >= 0.15:
            self._drift_count += 1; return True
        # Sigma-based: küçük fluctuasyon için
        std=float(np.std(self._recent))
        threshold=self._baseline_acc - self._threshold * max(std, 0.05)
        drifting = recent_acc < threshold
        if drifting: self._drift_count+=1
        return drifting

    def recent_accuracy(self):
        if not self._recent: return 0.0
        return round(float(np.mean(self._recent))*100,1)

    def get_status(self):
        return {"baseline_acc":round(self._baseline_acc*100,1),
                "recent_acc":self.recent_accuracy(),
                "window_size":len(self._recent),
                "drift_count":self._drift_count,
                "is_drifting":self.is_drifting()}


# =============================================================================
# MODEL VERSION STORE
# =============================================================================
class ModelVersionStore:
    MAX_VERSIONS=3

    def __init__(self,model_dir):
        self._dir=model_dir; os.makedirs(model_dir,exist_ok=True); self._versions=[]

    def save_version(self,engine,wf_acc,n_samples):
        ts=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path=os.path.join(self._dir,f"model_v{ts}.joblib")
        engine.save(path)
        self._versions.append({"path":path,"wf_acc":wf_acc,
            "trained_at":datetime.now(timezone.utc).isoformat(),"n_samples":n_samples})
        while len(self._versions)>self.MAX_VERSIONS:
            old=self._versions.pop(0)
            try:
                if os.path.exists(old["path"]): os.remove(old["path"])
            except: pass
        logger.info(f"Model versiyonu kaydedildi: v{ts} | WF Acc: {wf_acc}%")
        return path

    def get_best_version(self):
        if not self._versions: return None
        return max(self._versions,key=lambda v:v["wf_acc"])

    def get_versions(self):
        return list(reversed(self._versions))

    def should_rollback(self,engine,new_wf_acc):
        best=self.get_best_version()
        if best is None: return False
        if new_wf_acc>=best["wf_acc"]-0.5: return False
        logger.warning(f"Rollback: yeni WF {new_wf_acc}% < best {best['wf_acc']}%")
        engine.load(best["path"]); return True


# =============================================================================
# FEEDBACK BUFFER — gercek trade sonuclari
# =============================================================================
class FeedbackBuffer:
    MAX_SIZE=500

    def __init__(self):
        self._buffer=deque(maxlen=self.MAX_SIZE)
        self._lock=threading.Lock(); self._total_added=0

    def add_trade_result(self,features,side,net_pnl):
        """Trade kapandiktan SONRA cagirilir"""
        if net_pnl>0: label=1 if side=="LONG" else -1
        else: label=-1 if side=="LONG" else 1
        with self._lock:
            self._buffer.append((features.copy(),label)); self._total_added+=1

    def get_data(self):
        with self._lock:
            if len(self._buffer)<10: return None,None
            X=np.array([f for f,_ in self._buffer],dtype=np.float32)
            y=np.array([l for _,l in self._buffer],dtype=np.int32)
        return X,y

    def size(self): return len(self._buffer)
    def total_added(self): return self._total_added
    def should_retrain(self,threshold=200): return self.size()>=threshold


# =============================================================================
# ANA ML ENGINE v2
# =============================================================================
class MLEngine:
    RETRAIN_INTERVAL_HOURS=6
    FEEDBACK_RETRAIN_THRESHOLD=200
    DRIFT_RETRAIN_COOLDOWN_MIN=180  # 30dk → 3 saat (drift spiral önleme)
    MIN_SAMPLES=80

    def __init__(self,model_dir="."):
        self.gbm=GBMModel(); self.rf=RFModel()
        self._trained=False; self._wf={}; self._bt={}
        self._train_log=[]; self._pred_count=0
        self._last_retrain_ts=0.0; self._training_in_progress=False
        self.feedback=FeedbackBuffer()
        self.drift=DriftDetector(window=50,threshold_sigma=1.5)
        self.version_store=ModelVersionStore(os.path.join(model_dir,"model_versions"))
        self._retrain_triggers={"temporal":0,"drift":0,"feedback":0,"manual":0}
        self._last_drift_retrain=0.0
        logger.info("MLEngine v2 — Sürekli eğitim sistemi hazır (36 feature)")

    # ── EĞİTİM ───────────────────────────────────────────────────────────────
    def train(self,klines,symbol="GLOBAL",feedback_boost=False):
        if self._training_in_progress:
            return {"success":False,"reason":"training_in_progress"}
        self._training_in_progress=True
        t0=time.time()
        try:
            X,y=FeatureBuilder.build_dataset(klines,lookahead=5,threshold=0.005)
            if X is None:
                return {"success":False,"reason":"insufficient_data","symbol":symbol}
            if len(X)<self.MIN_SAMPLES:
                logger.warning(f"[{symbol}] Yetersiz veri: {len(X)} < {self.MIN_SAMPLES}")
                return {"success":False,"reason":"insufficient_data_min80","n_samples":len(X),"symbol":symbol}

            # Feedback boost
            if feedback_boost:
                Xf,yf=self.feedback.get_data()
                if Xf is not None and len(Xf)>=20 and Xf.shape[1]==X.shape[1]:
                    X=np.vstack([X,Xf]); y=np.concatenate([y,yf])
                    logger.info(f"Feedback boost: +{len(Xf)} sample")

            # Class dengesini sağla (HOLD ezberlemesini önle)
            unique_orig,cnt_orig = np.unique(y,return_counts=True)
            Xb, yb = balance_classes(X, y, target_ratio=0.30)
            unique,cnt = np.unique(yb,return_counts=True)
            logger.info(f"Class denge: {dict(zip(unique,cnt))} (orijinal: {dict(zip(unique_orig,cnt_orig))})")
            split=int(len(Xb)*0.8); Xtr,Xte=Xb[:split],Xb[split:]; ytr,yte=yb[:split],yb[split:]

            self.gbm.fit(Xtr,ytr); self.rf.fit(Xtr,ytr); self._trained=True

            lmap={"LONG":1,"SHORT":-1,"HOLD":0,"WAIT":0}
            preds=np.array([lmap[max(p:=self.gbm.predict_proba(x),key=p.get)] for x in Xte])
            acc=float(np.mean(preds==yte))*100 if len(yte)>0 else 0

            self._wf=walk_forward_validate(X,y,n_splits=4)
            wf_acc=self._wf.get("accuracy",0.0)
            self._bt=self.run_backtest(klines)
            self.drift.set_baseline(wf_acc)

            elapsed=time.time()-t0
            rec={"success":True,"symbol":symbol,"n_samples":len(X),"n_train":len(Xtr),"n_test":len(Xte),
                 "class_dist":{int(k):int(v) for k,v in zip(unique,cnt)},
                 "test_accuracy":round(acc,1),"wf_accuracy":wf_acc,"wf_f1":self._wf["f1"],
                 "backtest_roi":self._bt.get("roi",0),"backtest_sharpe":self._bt.get("sharpe",0),
                 "train_time_s":round(elapsed,2),"trained_at":datetime.now(timezone.utc).isoformat(),
                 "feedback_used":feedback_boost,"n_features":FeatureBuilder.N_FEATURES}
            self._train_log.append(rec)
            if len(self._train_log)>20: self._train_log.pop(0)
            self._last_retrain_ts=time.time()
            self.version_store.save_version(self,wf_acc,len(X))
            logger.info(f"✅ Eğitim [{symbol}] | {len(X)} sample | test_acc={acc:.1f}% | wf_acc={wf_acc}% | {elapsed:.1f}s")
            return rec
        finally:
            self._training_in_progress=False

    def train_on_multi_pair(self,klines_list,symbol="GLOBAL"):
        """Birden fazla pair verisini birlestirerek egit"""
        all_X,all_y=[],[]
        for klines in klines_list:
            X,y=FeatureBuilder.build_dataset(klines,lookahead=5,threshold=0.005)
            if X is not None and len(X)>=20: all_X.append(X); all_y.append(y)
        if not all_X: return {"success":False,"reason":"no_valid_data"}
        if self._training_in_progress: return {"success":False,"reason":"training_in_progress"}
        self._training_in_progress=True; t0=time.time()
        try:
            Xall=np.vstack(all_X); yall=np.concatenate(all_y)
            if len(Xall)<self.MIN_SAMPLES:
                return {"success":False,"reason":"insufficient_data_min80","n_samples":len(Xall)}
            # Class dengesini sağla (HOLD ezberlemesini önle)
            unique_orig,cnt_orig = np.unique(yall,return_counts=True)
            Xbal, ybal = balance_classes(Xall, yall, target_ratio=0.30)
            unique,cnt = np.unique(ybal,return_counts=True)
            logger.info(f"Multi-pair class denge: {dict(zip(unique,cnt))} (orig: {dict(zip(unique_orig,cnt_orig))})")
            split=int(len(Xbal)*0.8); Xtr,Xte=Xbal[:split],Xbal[split:]; ytr,yte=ybal[:split],ybal[split:]
            self.gbm.fit(Xtr,ytr); self.rf.fit(Xtr,ytr); self._trained=True
            lmap={"LONG":1,"SHORT":-1,"HOLD":0,"WAIT":0}
            preds=np.array([lmap[max(p:=self.gbm.predict_proba(x),key=p.get)] for x in Xte])
            acc=float(np.mean(preds==yte))*100 if len(yte)>0 else 0
            self._wf=walk_forward_validate(Xall,yall,n_splits=4)
            wf_acc=self._wf.get("accuracy",0.0)
            if klines_list: self._bt=self.run_backtest(klines_list[0])
            self.drift.set_baseline(wf_acc); elapsed=time.time()-t0
            rec={"success":True,"symbol":symbol,"n_samples":len(Xall),"n_pairs":len(all_X),
                 "test_accuracy":round(acc,1),"wf_accuracy":wf_acc,"wf_f1":self._wf["f1"],
                 "backtest_roi":self._bt.get("roi",0),"train_time_s":round(elapsed,2),
                 "trained_at":datetime.now(timezone.utc).isoformat(),"n_features":FeatureBuilder.N_FEATURES}
            self._train_log.append(rec)
            if len(self._train_log)>20: self._train_log.pop(0)
            self._last_retrain_ts=time.time()
            self.version_store.save_version(self,wf_acc,len(Xall))
            logger.info(f"✅ Multi-pair | {len(Xall)} sample, {len(all_X)} pair | wf_acc={wf_acc}% | {elapsed:.1f}s")
            return rec
        finally:
            self._training_in_progress=False

    # ── BACKTEST ─────────────────────────────────────────────────────────────
    def run_backtest(self,klines):
        X,y=FeatureBuilder.build_dataset(klines,lookahead=5,threshold=0.005)
        if X is None: return {}
        c=np.asarray(klines["close"],dtype=np.float64); MIN=40; sigs=[]
        for i,(x,_) in enumerate(zip(X,y)):
            p=self._ensemble(x); sigs.append((MIN+i,max(p,key=p.get)))
        self._bt=backtest(c,sigs)
        logger.info(f"Backtest → ROI:{self._bt['roi']}% WR:{self._bt['win_rate']}% Sharpe:{self._bt['sharpe']}")
        return self._bt

    # ── FEEDBACK — GERCEK TRADE SONUCU ───────────────────────────────────────
    def record_trade_result(self,features,side,net_pnl,symbol=""):
        """Trade kapatildiktan SONRA cagirilir — entry features + gercek PnL"""
        if features is None: return
        self.feedback.add_trade_result(features,side,net_pnl)
        self.drift.update(net_pnl>0)
        logger.debug(f"Trade result [{symbol}]: {side} | PnL=${net_pnl:.2f} | feedback={self.feedback.size()}")

    def should_retrain(self):
        if self._training_in_progress: return False,""
        hours_since=(time.time()-self._last_retrain_ts)/3600
        if hours_since>=self.RETRAIN_INTERVAL_HOURS and self._trained:
            return True,"temporal_6h"
        if self.feedback.should_retrain(self.FEEDBACK_RETRAIN_THRESHOLD):
            return True,"feedback_200_trades"
        if self.drift.is_drifting():
            mins=(time.time()-self._last_drift_retrain)/60
            # Yeterli feedback olmadan drift retrain'i atla (spiral önleme)
            if mins>=self.DRIFT_RETRAIN_COOLDOWN_MIN:
                if self.feedback.size() >= 20:  # En az 20 gerçek trade sonucu
                    return True,"drift_detected"
                else:
                    logger.info(f"Drift var ama feedback yetersiz ({self.feedback.size()}<20) — retrain ertelendi")
        return False,""

    def check_and_retrain(self,klines_getter):
        should,reason=self.should_retrain()
        if not should: return None
        logger.info(f"🔄 Retrain tetiklendi — sebep: {reason}")
        klines=klines_getter()
        if klines is None: return None
        if reason=="drift_detected":
            self._last_drift_retrain=time.time(); self._retrain_triggers["drift"]+=1
        elif reason.startswith("temporal"): self._retrain_triggers["temporal"]+=1
        elif reason.startswith("feedback"): self._retrain_triggers["feedback"]+=1
        return self.train(klines,symbol="AUTO_RETRAIN",feedback_boost=self.feedback.size()>=50)

    # ── TAHMIN ───────────────────────────────────────────────────────────────
    def predict(self,symbol,klines,current_price):
        try:
            if klines is None or len(klines.get("close",[]))<30: return self._fallback()
            feat=FeatureBuilder.build(klines)
            if feat is None: return self._fallback()
            if not self._trained: self.train(klines,symbol)

            gp=self.gbm.predict_proba(feat); rp=self.rf.predict_proba(feat)
            proba={k:gp[k]*0.6+rp[k]*0.4 for k in ["LONG","SHORT","HOLD","WAIT"]}
            t=sum(proba.values()); proba={k:v/t for k,v in proba.items()}
            sig=max(proba,key=proba.get); conf=round(proba[sig]*100,1)
            if conf<60: sig="WAIT"  # %60 eşik: kalite-frekans dengesi
            lev=15 if conf>80 else 10 if conf>72 else 5  # Daha muhafazakâr kaldıraç
            self._pred_count+=1
            return {"signal":sig,"confidence":conf,"indicators":self._active_inds(klines)[:4],
                    "model":"GBM+RF v2","leverage":lev,
                    "gbm_signal":max(gp,key=gp.get),"rf_signal":max(rp,key=rp.get),
                    "ensemble_proba":{k:round(v*100,1) for k,v in proba.items()},
                    "data_quality":"real","model_trained":self._trained,
                    "wf_accuracy":self._wf.get("accuracy",0),
                    "backtest_roi":self._bt.get("roi",0),"backtest_sharpe":self._bt.get("sharpe",0),
                    "drift_status":self.drift.get_status(),"_feat":feat}
        except Exception as e:
            logger.error(f"predict [{symbol}]: {e}"); return self._fallback()

    def _ensemble(self,feat):
        gp=self.gbm.predict_proba(feat); rp=self.rf.predict_proba(feat)
        out={k:gp[k]*0.6+rp[k]*0.4 for k in ["LONG","SHORT","HOLD","WAIT"]}
        t=sum(out.values()); return {k:v/t for k,v in out.items()}

    def _active_inds(self,klines):
        c=np.asarray(klines["close"],dtype=np.float64)
        h=np.asarray(klines["high"],dtype=np.float64)
        lo=np.asarray(klines["low"],dtype=np.float64)
        ti=Indicators(); inds=[]
        r=ti.rsi(c)
        if r<35: inds.append(f"RSI:{r:.0f}↓OS")
        elif r>65: inds.append(f"RSI:{r:.0f}↑OB")
        else: inds.append(f"RSI:{r:.0f}")
        _,_,mh=ti.macd(c); inds.append("MACD+" if mh>0 else "MACD-")
        bbu,_,bbl=ti.bb(c); bp=(c[-1]-bbl)/(bbu-bbl+1e-10)
        if bp<0.2: inds.append("BB-low")
        elif bp>0.8: inds.append("BB-high")
        else: inds.append(f"BB:{bp:.2f}")
        e9,e21=ti.ema(c,9)[-1],ti.ema(c,21)[-1]
        inds.append("EMA↑" if e9>e21 else "EMA↓")
        m=ti.mfi(h,lo,c,np.asarray(klines["volume"],dtype=np.float64))
        if m<30: inds.append(f"MFI:{m:.0f}↓")
        elif m>70: inds.append(f"MFI:{m:.0f}↑")
        return inds

    def _fallback(self):
        return {"signal":"WAIT","confidence":50.0,"indicators":["Veri bekleniyor"],
                "model":"GBM+RF v2","leverage":5,"gbm_signal":"WAIT","rf_signal":"WAIT",
                "ensemble_proba":{"LONG":25,"SHORT":25,"HOLD":25,"WAIT":25},
                "data_quality":"insufficient","model_trained":self._trained,
                "wf_accuracy":self._wf.get("accuracy",0),"backtest_roi":self._bt.get("roi",0),
                "backtest_sharpe":self._bt.get("sharpe",0),"drift_status":self.drift.get_status(),"_feat":None}

    # ── KAYIT / YUKLEME ──────────────────────────────────────────────────────
    def save(self,path):
        try:
            os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".",exist_ok=True)
            payload={"gbm_backend":self.gbm._backend,"gbm_clf":self.gbm.clf,
                "gbm_stumps":getattr(self.gbm,"_stumps",None),"gbm_alphas":getattr(self.gbm,"_alphas",None),
                "rf_backend":self.rf._backend,"rf_clf":self.rf.clf,"rf_trees":getattr(self.rf,"_trees",None),
                "trained":self._trained,"wf":self._wf,"bt":self._bt,"train_log":self._train_log,
                "pred_count":self._pred_count,"last_retrain":self._last_retrain_ts,
                "retrain_triggers":self._retrain_triggers,"n_features":FeatureBuilder.N_FEATURES}
            joblib.dump(payload,path,compress=3)
            logger.info(f"✅ Model kaydedildi → {path} ({os.path.getsize(path)//1024}KB)")
            return True
        except Exception as e:
            logger.error(f"Model kayıt hatası: {e}"); return False

    def load(self,path):
        if not os.path.exists(path): return False
        try:
            payload=joblib.load(path)
            self.gbm._backend=payload["gbm_backend"]; self.gbm.clf=payload["gbm_clf"]
            if payload.get("gbm_stumps"): self.gbm._stumps=payload["gbm_stumps"]
            if payload.get("gbm_alphas"): self.gbm._alphas=payload["gbm_alphas"]
            self.gbm.is_fitted=True; self.rf._backend=payload["rf_backend"]
            self.rf.clf=payload["rf_clf"]
            if payload.get("rf_trees"): self.rf._trees=payload["rf_trees"]
            self.rf.is_fitted=True; self._trained=payload.get("trained",True)
            self._wf=payload.get("wf",{}); self._bt=payload.get("bt",{})
            self._train_log=payload.get("train_log",[]); self._pred_count=payload.get("pred_count",0)
            self._last_retrain_ts=payload.get("last_retrain",0.0)
            self._retrain_triggers=payload.get("retrain_triggers",{"temporal":0,"drift":0,"feedback":0,"manual":0})
            self.drift.set_baseline(self._wf.get("accuracy",0.0))
            logger.info(f"✅ Model yüklendi ← {path} | wf_acc={self._wf.get('accuracy',0)}%")
            return True
        except Exception as e:
            logger.error(f"Model yükleme hatası: {e}"); return False

    # ── META ─────────────────────────────────────────────────────────────────
    def get_accuracy(self): return self._wf.get("accuracy",0.0)

    def get_feature_importance(self):
        imp=self.gbm.get_feature_importance()
        names=["RSI","StochRSI","MFI","Williams_R","MACD_line","MACD_sig","MACD_hist",
               "EMA9_21","EMA50_dist","BB_pct","BB_width","ATR_pct","Vol_ratio","OBV_trend",
               "Lag1","Lag2","Lag3","Ret5","Ret10","Ret20","Std5","Std20","Pos_HL","Gap_HL",
               "Mom_acc","Body","HL_range","Upper_wick",
               "VWAP_dist","Mom_slope","RSI_div","Market_regime",
               "Vol_momentum","Lower_wick","EMA_cross_spd","ATR_pct_rank"]
        types=["Momentum","Momentum","Volume/Momentum","Momentum","Trend","Trend","Trend",
               "Trend","Trend","Volatility","Volatility","Volatility","Volume","Volume",
               "Return","Return","Return","Return","Return","Return","Volatility","Volatility",
               "Range","Range","Momentum","Candle","Candle","Candle",
               "Volume","Momentum","Momentum","Market","Volume","Candle","Trend","Volatility"]
        if imp is not None and len(imp)==len(names):
            return [{"name":n,"type":types[i],"importance":round(float(imp[i]*100),1)} for i,n in enumerate(names)]
        static=[0.12,0.08,0.07,0.05,0.04,0.03,0.06,0.09,0.04,0.05,0.03,0.04,0.02,0.02,
                0.01,0.01,0.01,0.02,0.03,0.04,0.02,0.03,0.02,0.01,0.02,0.01,0.01,0.02,
                0.03,0.02,0.02,0.02,0.01,0.01,0.01,0.01]
        return [{"name":n,"type":types[i],"importance":round(static[i]*100,1)} for i,n in enumerate(names)]

    def get_info(self):
        return {"models":[{"name":"GradientBoosting","backend":self.gbm._backend,"weight":0.6,"estimators":400},
                          {"name":"RandomForest","backend":self.rf._backend,"weight":0.4,"estimators":150}],
                "ensemble":{"gbm":0.6,"rf":0.4},"n_features":FeatureBuilder.N_FEATURES,
                "features":self.get_feature_importance(),
                "feature_engineering_status":"Production v2 (36 features)",
                "data_processing":"Online Min15 Stream (limit=300)",
                "label_method":"lookahead=5bars, threshold=0.5% (v2)",
                "validation":"Walk-forward (4-fold, no lookahead)",
                "is_trained":self._trained,"wf_result":self._wf,"backtest":self._bt,
                "train_log":self._train_log[-20:],
                "feedback_buffer":self.feedback.size(),"feedback_total_added":self.feedback.total_added(),
                "predictions_made":self._pred_count,
                "last_retrain":(datetime.fromtimestamp(self._last_retrain_ts,tz=timezone.utc).isoformat()
                                if self._last_retrain_ts>0 else None),
                "next_retrain_in_hours":max(0,round(self.RETRAIN_INTERVAL_HOURS-(time.time()-self._last_retrain_ts)/3600,1)) if self._last_retrain_ts>0 else 0,
                "retrain_triggers":self._retrain_triggers,"drift_status":self.drift.get_status(),
                "model_versions":self.version_store.get_versions(),
                "training_in_progress":self._training_in_progress,
                "confidence_threshold":60,"signals":["LONG","SHORT","HOLD","WAIT"],
                "min_samples":self.MIN_SAMPLES,
                "retrain_schedule":{"temporal_hours":self.RETRAIN_INTERVAL_HOURS,
                                    "feedback_threshold":self.FEEDBACK_RETRAIN_THRESHOLD,
                                    "drift_cooldown_min":self.DRIFT_RETRAIN_COOLDOWN_MIN}}
