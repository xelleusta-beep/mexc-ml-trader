# MEXC ML Trading System - Sistem Özeti

## 🎯 Proje Hedefi
Kurumsal düzeyde, makine öğrenimi destekli otomatik kripto para futures trading sistemi.

## ✅ Tamamlanan Özellikler

### 1. Backend (FastAPI + Python)
- **MEXC Futures API Entegrasyonu**: Tüm pariteler için gerçek zamanlı veri
- **ML Engine**: 48 gelişmiş özellik, ensemble modeller (GBM + RF)
- **Signal Manager**: Otomatik sinyal üretimi ve işlem yönetimi
- **WebSocket Server**: Gerçek zamanlı bildirimler
- **REST API**: 10+ endpoint ile tam kontrol

### 2. Frontend (React + Vite + Tailwind)
- **Market Scanner**: Tüm futures paritelerini tarayan tablo
- **Active Trades**: Açık pozisyonların canlı takibi
- **ML Analysis**: Detaylı model analizi ve grafikler
- **Responsive Design**: Mobil uyumlu borsa arayüzü
- **Real-time Updates**: WebSocket ile anlık güncellemeler

### 3. Makine Öğrenimi
- **48 Özellik Mühendisliği**:
  - Momentum (7): RSI, StochRSI, MFI, Williams %R
  - Trend (8): MACD, EMA'lar, alignment
  - Volatility (7): Bollinger Bands, ATR
  - Volume (8): OBV, VWAP, money flow
  - Lag/Autocorr (6): 1/2/3/5 bar lags
  - Rolling Stats (8): Mean, std, skewness, kurtosis
  - Candle Structure (7): Range, body, wicks

- **Ensemble Model**:
  - Gradient Boosting (200 estimators)
  - Random Forest (150 trees)
  - Model agreement kontrolü

### 4. Strict Entry Kuralları
| Kriter | Değer | Sonuç |
|--------|-------|-------|
| Confidence | ≥%75 | Altında WAIT |
| Volume 24h | ≥$20M | Altında WAIT |
| Model Agreement | GBM == RF | Farklıysa WAIT |

### 5. Dinamik Kaldıraç Sistemi
```
Confidence >85% + Volume >$50M → 20x
Confidence >80% + Volume >$30M → 15x
Confidence >75% + Volume >$20M → 10x
Diğer → 5x
```

### 6. Risk Yönetimi
- Stop Loss: %2 (otomatik)
- Take Profit: %4 (otomatik)
- Maksimum pozisyon: $1000
- Pozisyon başına risk kontrolü

## 📁 Dosya Yapısı

```
/workspace/
├── backend/
│   ├── main.py              # FastAPI uygulama (342 satır)
│   ├── config.py            # Yapılandırma
│   ├── requirements.txt     # Python bağımlılıkları
│   ├── .env                 # Çevre değişkenleri
│   ├── services/
│   │   └── mexc_client.py   # MEXC API (185 satır)
│   ├── models/
│   │   └── ml_engine.py     # ML motoru (357 satır)
│   ├── strategies/
│   │   └── signal_manager.py # Sinyal yönetimi (130 satır)
│   └── utils/
│       └── logger.py        # Loglama (40 satır)
├── frontend/
│   ├── src/
│   │   ├── App.jsx          # Ana uygulama (92 satır)
│   │   ├── main.jsx         # Giriş noktası
│   │   ├── index.css        # Özel stiller (247 satır)
│   │   ├── components/
│   │   │   ├── MarketTable.jsx    # Piyasa tablosu (249 satır)
│   │   │   ├── ActiveTrades.jsx   # Aktif işlemler (211 satır)
│   │   │   └── MLAnalysis.jsx     # ML analizi (303 satır)
│   │   └── utils/
│   │       └── api.js       # API client (107 satır)
│   ├── package.json
│   ├── vite.config.js
│   └── tailwind.config.js
├── data/
│   ├── logs/                # Log dosyaları
│   ├── models_cache/        # ML modelleri
│   └── backtest_results/    # Backtest
├── start.sh                 # Başlatma scripti
└── README.md                # Dokümantasyon
```

## 🚀 Kurulum ve Çalıştırma

### Hızlı Başlangıç
```bash
# 1. Kurulum
bash start.sh setup

# 2. API anahtarlarını düzenle
nano backend/.env

# 3. Sistemi başlat
bash start.sh start
```

### Manuel Kurulum
```bash
# Backend
cd backend
pip install -r requirements.txt
python main.py

# Frontend
cd frontend
npm install
npm run dev
```

## 📊 API Endpoints

### REST
- `GET /` - Sistem durumu
- `GET /api/market/scan` - Piyasa tarama
- `GET /api/market/tickers` - Ticker listesi
- `POST /api/ml/train/{symbol}` - Model eğitme
- `GET /api/ml/predict/{symbol}` - Tahmin alma
- `GET /api/trading/positions` - Pozisyonlar
- `POST /api/trading/execute/{symbol}` - İşlem aç
- `DELETE /api/trading/close/{symbol}` - Pozisyon kapat
- `GET /api/account/info` - Hesap bilgisi

### WebSocket
- `WS /ws` - Real-time updates
  - `market_update`: Yeni sinyaller
  - `position_update`: Pozisyon güncellemeleri
  - `trade_executed`: İşlem gerçekleşti
  - `position_closed`: Pozisyon kapandı

## 🎨 Arayüz Özellikleri

### Market Scanner Tab
- 4 özet kartı (Signals, Long, Short, Avg Confidence)
- Filtrelenmiş tablo (sadece kaliteli sinyaller)
- Volume badge ($20M+ kontrolü)
- Confidence meter (görsel bar)
- Signal badges (LONG/SHORT/WAIT)
- Leverage göstergesi
- Anında analiz butonu

### Active Trades Tab
- 3 özet kartı (Positions, P&L, Risk Level)
- Detaylı pozisyon tablosu
- Canlı P&L güncellemeleri
- Tek tıkla pozisyon kapatma
- Risk uyarıları

### ML Analysis Tab
- 4 özet kartı (Signal, Confidence, Leverage, Direction)
- Model confidence karşılaştırma grafiği
- Validation details (3 kontrol)
- Decision reason açıklaması
- Train model butonu
- Execute trade butonu

## 🔒 Güvenlik Önlemleri

1. **API Key Management**: .env dosyasında güvenli saklama
2. **Volume Filter**: $20M altı coinlere işlem yok
3. **Confidence Threshold**: %75 altı sinyallere işlem yok
4. **Model Agreement**: Modeller farklı yön gösterirse işlem yok
5. **Auto SL/TP**: Her işlemde otomatik stop loss/take profit
6. **Position Sizing**: Maksimum $1000 pozisyon boyutu
7. **Dynamic Leverage**: Güven ve hacme göre otomatik kaldıraç

## 📈 Performans Optimizasyonları

1. **Vectorized NumPy Operations**: EMA, SMA, OBV hesaplamaları
2. **Async HTTP Client**: httpx ile non-blocking API çağrıları
3. **WebSocket Broadcasting**: Efficient real-time updates
4. **Model Caching**: Eğitilmiş modelleri diskte saklama
5. **Background Tasks**: Periodic market scanning

## 🧪 Test ve Validasyon

```bash
# Backend syntax check
python -m py_compile backend/*.py backend/*/*.py

# API test
curl http://localhost:8000/
curl http://localhost:8000/api/market/tickers

# Model test (Python)
from models.ml_engine import AdvancedMLEngine
engine = AdvancedMLEngine()
# Test kodları...
```

## 📝 Önemli Notlar

1. **İlk Kullanım**: Model eğitimi için birkaç dakika gerekir
2. **Test Modu**: API anahtarı olmadan sadece okuma yapılabilir
3. **Canlı İşlem**: Gerçek API anahtarları gerekli
4. **Loglar**: `/data/logs/trading.log` dosyasında tutulur
5. **Model Cache**: `/data/models_cache/` klasöründe saklanır

## ⚠️ Risk Uyarısı

- Kripto para futures işlemleri **yüksek risk** içerir
- Sadece **kaybetmeyi göze alabileceğiniz** tutarlarla işlem yapın
- Bu sistem bir **yatırım tavsiyesi değildir**
- Geçmiş performans gelecek sonuçların garantisi değildir
- **Her zaman kendi araştırmanızı yapın**

## 🛠️ Gelecek Geliştirmeler

- [ ] Backtest modülü
- [ ] Telegram bildirimleri
- [ ] Strateji optimizasyonu
- [ ] Multi-exchange support
- [ ] Paper trading modu
- [ ] Performance analytics dashboard
- [ ] Auto-retrain scheduling

## 📞 Destek

Sorunlar için:
1. Log dosyalarını kontrol edin: `data/logs/`
2. API yanıtlarını inceleyin
3. Browser console'u kontrol edin
4. README.md'deki troubleshooting bölümüne bakın

---

**Versiyon**: 4.0.0  
**Son Güncelleme**: 2024  
**Lisans**: Proprietary  
