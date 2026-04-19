# MEXC ML Trader 🤖

Gerçek zamanlı makine öğrenimi ile MEXC vadeli işlem piyasası analiz ve sinyal sistemi.

## Mimari

```
MEXC Futures API → FastAPI Backend → ML Ensemble → WebSocket → Frontend Dashboard
     (Gerçek veri)    (Python)       (RF+XGB+LSTM)   (Canlı)    (Tarayıcı)
```

## ML Modeli

**Ensemble = Random Forest + XGBoost + LSTM**

- **Özellikler:** RSI-14, MACD(12,26,9), Bollinger Bands(20), EMA(9,21), ATR-14, OBV, MFI-14, Volume Ratio
- **Sinyaller:** LONG, SHORT, HOLD, WAIT
- **Güven eşiği:** >52% → işlem sinyali üretilir
- **Kaldıraç:** Güvene göre dinamik (5x–20x)

## Kurulum (Local)

```bash
cd backend
pip install -r requirements.txt
python main.py
```

Sonra tarayıcıda: http://localhost:8000

## Render.com Ücretsiz Deployment (Önerilen)

### 1. GitHub'a Yükle

```bash
git init
git add .
git commit -m "MEXC ML Trader"
git remote add origin https://github.com/KULLANICI/mexc-ml-trader.git
git push -u origin main
```

### 2. Render'da Deploy Et

1. https://render.com → Sign Up (ücretsiz, kredi kartı yok)
2. Dashboard → **New +** → **Web Service**
3. **Connect GitHub** → repoyu seç
4. Ayarlar:
   - **Name:** mexc-ml-trader
   - **Region:** Frankfurt (EU)
   - **Branch:** main
   - **Runtime:** Python 3
   - **Build Command:** `pip install -r backend/requirements.txt`
   - **Start Command:** `cd backend && uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Plan:** Free
5. **Create Web Service** tıkla

### 3. Deploy tamamlandığında

Site adresin: `https://mexc-ml-trader.onrender.com`

> **Not:** Render ücretsiz planda 15 dakika kullanılmazsa uyur, ilk açılışta 30-60 saniye bekleyebilirsin.

## Proje Yapısı

```
mexc-ml-trader/
├── backend/
│   ├── main.py          # FastAPI + WebSocket + MEXC API
│   ├── ml_engine.py     # ML modelleri (RF, XGB, LSTM)
│   └── requirements.txt
├── frontend/
│   └── index.html       # Dashboard UI
├── render.yaml          # Render konfigürasyonu
├── Procfile             # Process tanımı
└── README.md
```

## API Endpoints

| Endpoint | Açıklama |
|----------|----------|
| `GET /` | Dashboard UI |
| `GET /api/scan` | Tüm pair sinyalleri |
| `GET /api/pair/{symbol}` | Tek pair detayı |
| `GET /api/stats` | Özet istatistikler |
| `GET /api/model` | ML model bilgisi |
| `WS /ws` | Gerçek zamanlı veri akışı |

## MEXC API Kullanımı

Sistem şu endpoint'leri kullanır (API key gerekmez, public):
- `GET /api/v1/contract/ticker` → Anlık fiyat
- `GET /api/v1/contract/kline/{symbol}` → OHLCV verisi (ML için)

## Gelecek Geliştirmeler

- [ ] Gerçek emir gönderme (MEXC API key ile)
- [ ] Telegram bot bildirimleri
- [ ] PostgreSQL ile trade geçmişi
- [ ] Backtesting modülü
- [ ] Daha derin LSTM modeli (TensorFlow/Keras)
