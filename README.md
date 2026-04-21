# MEXC ML Trader — Gerçek Makine Öğrenimi Trading Sistemi

## Mimari

```
MEXC Futures API  →  Feature Engineering  →  ML Model  →  Sinyal  →  Dashboard
  (Gerçek OHLCV)      (28 özellik)          (GBM+RF)    (LONG/SHORT)  (WebSocket)
```

## Gerçek ML Sistemi — Ne Yapıyor?

### ✅ 1. Gerçek Eğitim (model.fit)
- MEXC API'den 15 dakikalık OHLCV verisi çeker (200 bar)
- Her bar için **28 teknik özellik** hesaplar
- **Label üretir:** 5 bar sonra fiyat +0.8% → LONG, -0.8% → SHORT, ortada → HOLD
- `model.fit(X_train, y_train)` ile gerçekten öğrenir

### ✅ 2. Ensemble Model
- **LightGBM** (öncelikli) → sklearn GBM → NumPy AdaBoost (fallback zinciri)
- **Random Forest** (sklearn → NumPy fallback)
- Ensemble: GBM %60 + RF %40 ağırlıklı ortalama

### ✅ 3. Walk-Forward Validation
- Zaman sıralı 4-fold validation (gelecek sızmaz)
- Her fold: train → test, örtüşmüyor
- Accuracy ve F1 skoru rapor eder

### ✅ 4. Backtest
- Modelin sinyalleriyle geçmiş performans ölçülür
- Metrikler: ROI, Win Rate, Max Drawdown, Sharpe Ratio
- %2.5 Stop-Loss, %4.5 Take-Profit, %0.04 fee dahil

### ✅ 5. Feedback Loop
- Canlı tahminler buffer'a eklenir
- 100 sample biriktikçe model yeniden eğitilir (5dk cooldown)
- Piyasa değişince model adapte olur

### ✅ 6. 28 Gerçek Feature
RSI-14, StochRSI, MFI, Williams%R, MACD(12,26,9), EMA(9,21,50),
Bollinger Bands, ATR, OBV Trend, Volume Ratio,
Lag Returns (t-1,t-2,t-3), Rolling Returns (5,10,20 bar),
Rolling Std, HL Position, Momentum Acceleration, Candle Body/Wick

## Proje Yapısı

```
mexc-real/
├── backend/
│   ├── main.py          # FastAPI + WebSocket + MEXC API + Auto-train
│   ├── ml_engine.py     # Gerçek ML (LightGBM, RF, Backtest, WalkForward)
│   └── requirements.txt
├── frontend/
│   └── index.html       # Canlı Dashboard (WebSocket bağlantılı)
├── Procfile             # Render process tanımı
├── render.yaml          # Render konfigürasyonu
└── runtime.txt          # Python 3.11
```

## API Endpoints

| Endpoint | Method | Açıklama |
|---|---|---|
| `/` | GET | Dashboard UI |
| `/api/scan` | GET | Tüm pair sinyalleri |
| `/api/pair/{symbol}` | GET | Tek pair detayı |
| `/api/stats` | GET | Özet istatistikler |
| `/api/model` | GET | ML model bilgisi + backtest |
| `/api/train/{symbol}` | POST | Elle model eğitimi |
| `/api/train_all` | POST | Global model eğitimi |
| `/api/backtest/{symbol}` | GET | Backtest çalıştır |
| `/ws` | WebSocket | Canlı veri akışı |
| `/health` | GET | Servis durumu |

## Render.com Deploy (Ücretsiz)

### 1. GitHub'a yükle
```bash
git init
git add .
git commit -m "MEXC Real ML Trader"
git remote add origin https://github.com/KULLANICI/mexc-ml-trader.git
git push -u origin main
```

### 2. Render'da deploy
1. render.com → Sign Up (ücretsiz, kart yok)
2. New → Web Service → GitHub repoyu seç
3. Ayarlar:
   - **Runtime:** Python 3
   - **Build Command:** `pip install -r backend/requirements.txt`
   - **Start Command:** `cd backend && uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Plan:** Free
4. Create Web Service → ~3 dakika deploy

### 3. Otomatik çalışma akışı
- Servis açılır
- MEXC API'den veri çeker (60sn bekler)
- İlk 8 pair ile modeli eğitir (LightGBM)
- Walk-forward validation çalışır
- BTC_USDT backtest yapılır
- Tüm pairler 30sn'de bir taranır
- Sinyaller WebSocket ile dashboard'a akar

## MEXC Public API (API Key gerekmez)
- `GET /api/v1/contract/ticker?symbol=BTC_USDT` → Anlık fiyat
- `GET /api/v1/contract/kline/BTC_USDT?interval=Min15&limit=200` → OHLCV

## Önemli Notlar
- Render ücretsiz planda 15dk kullanılmazsa **uyku moduna** girer
- İlk açılışta 30-60 saniye bekleme olabilir
- Model eğitimi ~20-60 saniye sürer (pair sayısına göre)
- Gerçek emir açılmaz — sadece sinyal üretir
- Accuracy %30-45 arasındadır (3 sınıflı problem için normal)
