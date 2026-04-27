"""
Advanced ML Engine for MEXC Trading System
Features: 48+ features, Ensemble models, Strict validation
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os
from config import settings


class AdvancedMLEngine:
    def __init__(self):
        self.gbm_model = None
        self.rf_model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.is_trained = False
        
    def _ultra_fast_ema(self, prices: np.ndarray, span: int) -> np.ndarray:
        """Ultra-fast EMA calculation"""
        alpha = 2.0 / (span + 1)
        ema = np.zeros_like(prices, dtype=np.float64)
        ema[0] = prices[0]
        for i in range(1, len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
        return ema
    
    def _calculate_sma(self, prices: np.ndarray, window: int) -> np.ndarray:
        """Fast SMA using cumsum"""
        result = np.full_like(prices, np.nan, dtype=np.float64)
        cumsum = np.cumsum(prices)
        result[window-1:] = (cumsum[window-1:] - np.concatenate([[0], cumsum[:-window]])) / window
        return result
    
    def _calculate_obv_trend(self, closes: np.ndarray, volumes: np.ndarray) -> np.ndarray:
        """OBV with trend detection"""
        obv = np.zeros_like(closes, dtype=np.float64)
        for i in range(1, len(closes)):
            if closes[i] > closes[i-1]:
                obv[i] = obv[i-1] + volumes[i]
            elif closes[i] < closes[i-1]:
                obv[i] = obv[i-1] - volumes[i]
            else:
                obv[i] = obv[i-1]
        
        # OBV trend (5-period EMA of OBV)
        obv_ema = self._ultra_fast_ema(obv, 5)
        obv_trend = np.zeros_like(obv)
        obv_trend[1:] = np.sign(obv_ema[1:] - obv_ema[:-1])
        return obv_trend
    
    def _calculate_vwap(self, highs: np.ndarray, lows: np.ndarray, 
                       closes: np.ndarray, volumes: np.ndarray) -> np.ndarray:
        """VWAP calculation"""
        typical_price = (highs + lows + closes) / 3
        cum_vol = np.cumsum(volumes)
        cum_tp_vol = np.cumsum(typical_price * volumes)
        vwap = np.divide(cum_tp_vol, cum_vol, out=np.zeros_like(cum_tp_vol), where=cum_vol!=0)
        return vwap
    
    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build 48 advanced features"""
        closes = df['close'].values.astype(np.float64)
        highs = df['high'].values.astype(np.float64)
        lows = df['low'].values.astype(np.float64)
        opens = df['open'].values.astype(np.float64)
        volumes = df['volume'].values.astype(np.float64)
        
        features = {}
        
        # === MOMENTUM (7 features) ===
        # RSI
        delta = np.diff(closes, prepend=closes[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = self._calculate_sma(gain, 14)
        avg_loss = self._calculate_sma(loss, 14)
        rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
        features['rsi'] = 100 - (100 / (1 + rs))
        
        # Stochastic RSI
        lowest_rsi = np.minimum.accumulate(features['rsi'][::-1])[::-1]
        highest_rsi = np.maximum.accumulate(features['rsi'][::-1])[::-1]
        stoch_rsi = np.divide(
            features['rsi'] - lowest_rsi,
            highest_rsi - lowest_rsi,
            out=np.zeros_like(features['rsi']),
            where=(highest_rsi - lowest_rsi) != 0
        )
        features['stoch_rsi'] = stoch_rsi * 100
        
        # MFI approximation
        typical_price = (highs + lows + closes) / 3
        money_flow = typical_price * volumes
        positive_flow = np.where(typical_price > np.roll(typical_price, 1), money_flow, 0)
        negative_flow = np.where(typical_price < np.roll(typical_price, 1), money_flow, 0)
        avg_positive = self._calculate_sma(positive_flow, 14)
        avg_negative = self._calculate_sma(negative_flow, 14)
        mfi_ratio = np.divide(avg_positive, avg_negative, out=np.zeros_like(avg_positive), where=avg_negative!=0)
        features['mfi'] = 100 - (100 / (1 + mfi_ratio))
        
        # Williams %R
        highest_high = np.maximum.reduce([highs[i:i+14] if i+14 <= len(highs) else highs[i:] 
                                         for i in range(len(highs))], axis=1) if len(highs) >= 14 else highs
        lowest_low = np.minimum.reduce([lows[i:i+14] if i+14 <= len(lows) else lows[i:] 
                                       for i in range(len(lows))], axis=1) if len(lows) >= 14 else lows
        features['williams_r'] = -100 * (highest_high - closes) / (highest_high - lowest_low + 1e-10)
        
        # RSI derivatives
        features['rsi_change'] = np.diff(features['rsi'], prepend=features['rsi'][0])
        features['rsi_ma'] = self._calculate_sma(features['rsi'], 5)
        
        # === TREND (8 features) ===
        # MACD
        ema12 = self._ultra_fast_ema(closes, 12)
        ema26 = self._ultra_fast_ema(closes, 26)
        features['macd'] = ema12 - ema26
        features['macd_signal'] = self._ultra_fast_ema(features['macd'], 9)
        features['macd_hist'] = features['macd'] - features['macd_signal']
        
        # EMAs
        features['ema9'] = self._ultra_fast_ema(closes, 9)
        features['ema21'] = self._ultra_fast_ema(closes, 21)
        features['ema50'] = self._ultra_fast_ema(closes, 50)
        features['ema200'] = self._ultra_fast_ema(closes, 200)
        
        # EMA alignment
        features['ema_alignment'] = (
            (features['ema9'] > features['ema21']) & 
            (features['ema21'] > features['ema50'])
        ).astype(int)
        
        # === VOLATILITY (7 features) ===
        # Bollinger Bands
        features['bb_middle'] = self._calculate_sma(closes, 20)
        bb_std = np.zeros_like(closes)
        for i in range(20, len(closes)):
            bb_std[i] = np.std(closes[i-20:i])
        features['bb_upper'] = features['bb_middle'] + 2 * bb_std
        features['bb_lower'] = features['bb_middle'] - 2 * bb_std
        features['bb_pct'] = (closes - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'] + 1e-10)
        features['bb_width'] = (features['bb_upper'] - features['bb_lower']) / features['bb_middle']
        
        # ATR
        tr1 = highs - lows
        tr2 = np.abs(highs - np.roll(closes, 1))
        tr3 = np.abs(lows - np.roll(closes, 1))
        tr = np.maximum.reduce([tr1, tr2, tr3])
        features['atr'] = self._ultra_fast_ema(tr, 14)
        features['volatility_regime'] = (features['atr'] / closes).astype(int)
        
        # === VOLUME & ORDER FLOW (8 features) ===
        # Volume ratio
        vol_ma = self._calculate_sma(volumes, 20)
        features['volume_ratio'] = volumes / (vol_ma + 1e-10)
        
        # OBV trend
        features['obv_trend'] = self._calculate_obv_trend(closes, volumes)
        
        # VWAP
        features['vwap'] = self._calculate_vwap(highs, lows, closes, volumes)
        features['vwap_deviation'] = (closes - features['vwap']) / features['vwap']
        
        # Money flow
        features['money_flow'] = money_flow
        features['money_flow_ma'] = self._calculate_sma(money_flow, 14)
        
        # === LAG/AUTOCORR (6 features) ===
        for lag in [1, 2, 3, 5]:
            features[f'lag_{lag}'] = np.roll(closes, lag)
            features[f'return_lag_{lag}'] = np.roll((closes - np.roll(closes, 1)) / np.roll(closes, 1), lag)
        
        # Autocorrelation
        returns = np.diff(closes, prepend=closes[0]) / closes
        features['autocorr_5'] = pd.Series(returns).rolling(5).apply(lambda x: np.corrcoef(x[:-1], x[1:])[0,1] if len(x) > 1 else 0)
        
        # === ROLLING STATS (8 features) ===
        for window in [5, 10, 20]:
            ret_series = pd.Series(returns)
            features[f'return_mean_{window}'] = ret_series.rolling(window).mean()
            features[f'return_std_{window}'] = ret_series.rolling(window).std()
            if window >= 3:
                features[f'skewness_{window}'] = ret_series.rolling(window).apply(lambda x: x.skew(), raw=False)
                features[f'kurtosis_{window}'] = ret_series.rolling(window).apply(lambda x: x.kurt(), raw=False)
        
        # === CANDLE STRUCTURE (7 features) ===
        candle_range = highs - lows
        body = closes - opens
        upper_wick = highs - np.maximum(closes, opens)
        lower_wick = np.minimum(closes, opens) - lows
        
        features['range'] = candle_range
        features['body_size'] = np.abs(body)
        features['body_position'] = body / (candle_range + 1e-10)
        features['upper_wick_ratio'] = upper_wick / (candle_range + 1e-10)
        features['lower_wick_ratio'] = lower_wick / (candle_range + 1e-10)
        features['is_bullish'] = (body > 0).astype(int)
        features['doji'] = (np.abs(body) < candle_range * 0.1).astype(int)
        
        # Convert to DataFrame
        feature_df = pd.DataFrame(features)
        self.feature_columns = feature_df.columns.tolist()
        
        # Handle NaN values
        feature_df = feature_df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        return feature_df
    
    def train(self, klines: List[List], symbol: str = "BTC_USDT") -> Dict:
        """Train ensemble models"""
        if len(klines) < 100:
            return {"success": False, "error": "Insufficient data"}
        
        # Prepare data
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        features = self.build_features(df)
        
        # Create target (next period return direction)
        returns = df['close'].pct_change().shift(-1)
        y = (returns > 0).astype(int)
        
        # Remove last row (no target) and first rows (NaN features)
        X = features.iloc[:-1].values
        y = y.iloc[:-1].values
        
        if len(X) < 50 or np.sum(y) == 0 or np.sum(y) == len(y):
            return {"success": False, "error": "Invalid target distribution"}
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train GBM
        self.gbm_model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            min_samples_split=10,
            random_state=42
        )
        self.gbm_model.fit(X_scaled, y)
        
        # Train RF
        self.rf_model = RandomForestClassifier(
            n_estimators=150,
            max_depth=8,
            min_samples_split=10,
            random_state=42,
            n_jobs=-1
        )
        self.rf_model.fit(X_scaled, y)
        
        self.is_trained = True
        
        # Calculate training metrics
        gbm_score = self.gbm_model.score(X_scaled, y)
        rf_score = self.rf_model.score(X_scaled, y)
        
        return {
            "success": True,
            "symbol": symbol,
            "gbm_accuracy": round(gbm_score, 4),
            "rf_accuracy": round(rf_score, 4),
            "feature_count": len(self.feature_columns),
            "samples": len(X)
        }
    
    def predict(self, symbol: str, klines: List[List], price: float, 
                volume_24h: float) -> Dict:
        """Generate prediction with strict validation"""
        if not self.is_trained:
            return {
                "signal": "WAIT",
                "confidence": 0.0,
                "reason": "Model not trained"
            }
        
        # Validate volume
        volume_ok = volume_24h >= settings.min_volume_24h_usd
        
        # Prepare features
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        features = self.build_features(df)
        X = features.iloc[-1:].values
        X_scaled = self.scaler.transform(X)
        
        # Get predictions from both models
        gbm_prob = self.gbm_model.predict_proba(X_scaled)[0][1]
        rf_prob = self.rf_model.predict_proba(X_scaled)[0][1]
        
        # Ensemble confidence
        avg_confidence = (gbm_prob + rf_prob) / 2
        
        # Check model agreement
        gbm_pred = 1 if gbm_prob > 0.5 else 0
        rf_pred = 1 if rf_prob > 0.5 else 0
        models_agree = gbm_pred == rf_pred
        
        # STRICT ENTRY RULES
        confidence_ok = avg_confidence >= settings.confidence_threshold
        
        if not confidence_ok:
            signal = "WAIT"
            reason = f"Confidence {avg_confidence:.2%} < {settings.confidence_threshold:.0%}"
        elif not volume_ok:
            signal = "WAIT"
            reason = f"Volume ${volume_24h/1e6:.1f}M < $20M minimum"
        elif not models_agree:
            signal = "WAIT"
            reason = "Models disagree on direction"
        else:
            signal = "LONG" if gbm_pred == 1 else "SHORT"
            reason = "All conditions met"
        
        # Determine leverage based on confidence and volume
        if avg_confidence > 0.85 and volume_24h > 50_000_000:
            leverage = 20
        elif avg_confidence > 0.80 and volume_24h > 30_000_000:
            leverage = 15
        elif avg_confidence > 0.75 and volume_24h > 20_000_000:
            leverage = 10
        else:
            leverage = 5
        
        return {
            "signal": signal,
            "confidence": round(avg_confidence, 4),
            "gbm_confidence": round(gbm_prob, 4),
            "rf_confidence": round(rf_prob, 4),
            "models_agree": models_agree,
            "volume_ok": volume_ok,
            "volume_24h_usd": volume_24h,
            "leverage": leverage,
            "reason": reason,
            "direction": "BULLISH" if gbm_pred == 1 else "BEARISH"
        }
    
    def save_model(self, path: str = None):
        """Save trained models"""
        if not self.is_trained:
            return False
        
        path = path or settings.model_cache_dir
        os.makedirs(path, exist_ok=True)
        
        joblib.dump(self.gbm_model, os.path.join(path, "gbm_model.pkl"))
        joblib.dump(self.rf_model, os.path.join(path, "rf_model.pkl"))
        joblib.dump(self.scaler, os.path.join(path, "scaler.pkl"))
        joblib.dump(self.feature_columns, os.path.join(path, "features.pkl"))
        
        return True
    
    def load_model(self, path: str = None) -> bool:
        """Load trained models"""
        path = path or settings.model_cache_dir
        
        try:
            self.gbm_model = joblib.load(os.path.join(path, "gbm_model.pkl"))
            self.rf_model = joblib.load(os.path.join(path, "rf_model.pkl"))
            self.scaler = joblib.load(os.path.join(path, "scaler.pkl"))
            self.feature_columns = joblib.load(os.path.join(path, "features.pkl"))
            self.is_trained = True
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
