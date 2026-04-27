# MEXC ML Trading System - Kurulum Rehberi

## 📁 Proje Yapısı

```
/workspace
├── backend/                 # Python FastAPI Backend
│   ├── main.py             # Ana uygulama
│   ├── config.py           # Yapılandırma
│   ├── requirements.txt    # Python bağımlılıkları
│   ├── .env.example        # Örnek çevre değişkenleri
│   ├── services/
│   │   └── mexc_client.py  # MEXC API istemcisi
│   ├── models/
│   │   └── ml_engine.py    # Makine öğrenimi motoru
│   ├── strategies/
│   │   └── signal_manager.py  # Sinyal yöneticisi
│   └── utils/
│       └── logger.py       # Loglama utility
├── frontend/               # React Frontend
│   ├── src/
│   │   ├── App.jsx        # Ana uygulama
│   │   ├── main.jsx       # Giriş noktası
│   │   ├── index.css      # Stiller
│   │   ├── components/
│   │   │   ├── MarketTable.jsx    # Piyasa tablosu
│   │   │   ├── ActiveTrades.jsx   # Aktif işlemler
│   │   │   └── MLAnalysis.jsx     # ML analizi
│   │   └── utils/
│   │       └── api.js     # API istemcisi
│   ├── package.json
│   └── vite.config.js
└── data/                   # Veri klasörü
    ├── logs/              # Log dosyaları
    ├── models_cache/      # ML modelleri
    └── backtest_results/  # Backtest sonuçları
```

## 🔧 Backend Kurulumu

### 1. Python Bağımlılıklarını Yükleyin

```bash
cd /workspace/backend
pip install -r requirements.txt
```

### 2. Çevre Değişkenlerini Ayarlayın

```bash
cp .env.example .env
# .env dosyasını düzenleyin:
# - MEXC_API_KEY: MEXC API anahtarınız
# - MEXC_API_SECRET: MEXC API sırrınız
```

### 3. Backend Sunucusunu Başlatın

```bash
python main.py
# veya
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Backend http://localhost:8000 adresinde çalışacaktır.

## 🎨 Frontend Kurulumu

### 1. Node.js Bağımlılıklarını Yükleyin

```bash
cd /workspace/frontend
npm install
```

### 2. Frontend Geliştirme Sunucusunu Başlatın

```bash
npm run dev
```

Frontend http://localhost:3000 adresinde çalışacaktır.

## 🚀 Sistem Özellikleri

### Makine Öğrenimi
- **48 Gelişmiş Özellik**: Momentum, trend, volatilite, hacim, otokorelasyon
- **Ensemble Model**: Gradient Boosting + Random Forest
- **Strict Validation**: %75+ güven, $20M+ hacim, model mutabakatı

### Trading Kuralları
- **Minimum Güven**: %75
- **Minimum Hacim**: $20M (24 saat)
- **Dinamik Kaldıraç**: 
  - %85+ güven + $50M+ hacim → 20x
  - %80+ güven + $30M+ hacim → 15x
  - %75+ güven + $20M+ hacim → 10x
  - Diğer → 5x
- **Stop Loss**: %2
- **Take Profit**: %4

### Arayüz
- **Market Scanner**: Tüm futures paritelerini tarar
- **Active Trades**: Açık pozisyonları gösterir
- **ML Analysis**: Detaylı model analizi ve grafikler

## 📡 API Endpoints

### REST API
- `GET /api/market/scan` - Piyasa tarama
- `GET /api/market/tickers` - Ticker bilgileri
- `POST /api/ml/train/{symbol}` - Model eğitme
- `GET /api/ml/predict/{symbol}` - Tahmin alma
- `GET /api/trading/positions` - Pozisyonlar
- `POST /api/trading/execute/{symbol}` - İşlem açma
- `DELETE /api/trading/close/{symbol}` - Pozisyon kapatma

### WebSocket
- `WS /ws` - Gerçek zamanlı güncellemeler

## ⚠️ Risk Uyarısı

Kripto para vadeli işlemleri yüksek risk içerir. Sadece kaybetmeyi göze alabileceğiniz tutarlarla işlem yapın. Bu sistem bir yatırım tavsiyesi değildir.

## 📝 Notlar

- İlk kullanımda model eğitimi için birkaç dakika bekleyin
- Test modunda çalıştırmak için API anahtarı girmeyebilirsiniz (sadece okuma işlemleri)
- Canlı işlem için gerçek API anahtarları gereklidir
- Loglar `/data/logs/trading.log` dosyasında tutulur

## 🛠️ Sorun Giderme

### Backend başlatılamıyor
```bash
# Bağımlılıkları kontrol edin
pip install -r requirements.txt

# Port kullanımını kontrol edin
lsof -i :8000
```

### Frontend bağlanamıyor
```bash
# Backend'in çalıştığından emin olun
curl http://localhost:8000/

# CORS ayarlarını kontrol edin
```

### ML model hataları
```bash
# Yeterli veri olduğundan emin olun
# En az 100 mum veresi gerekli
```

## 📞 Destek

Sorularınız için log dosyalarını kontrol edin ve hata mesajlarını inceleyin.
