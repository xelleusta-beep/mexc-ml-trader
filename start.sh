#!/bin/bash

# MEXC ML Trading System - Başlatma Scripti

echo "🚀 MEXC ML Trading System Başlatılıyor..."
echo ""

# Renk kodları
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Backend kontrolü
check_backend() {
    if ! command -v python &> /dev/null; then
        echo -e "${YELLOW}⚠️ Python bulunamadı. Lütfen Python 3.8+ yükleyin.${NC}"
        exit 1
    fi
    
    if ! command -v pip &> /dev/null; then
        echo -e "${YELLOW}⚠️ pip bulunamadı.${NC}"
        exit 1
    fi
}

# Frontend kontrolü
check_frontend() {
    if ! command -v node &> /dev/null; then
        echo -e "${YELLOW}⚠️ Node.js bulunamadı. Frontend çalıştırılamayacak.${NC}"
        return 1
    fi
    
    if ! command -v npm &> /dev/null; then
        echo -e "${YELLOW}⚠️ npm bulunamadı. Frontend çalıştırılamayacak.${NC}"
        return 1
    fi
    
    return 0
}

# Backend kurulumu
setup_backend() {
    echo -e "${BLUE}📦 Backend bağımlılıkları yükleniyor...${NC}"
    cd backend
    
    if [ ! -d "venv" ]; then
        python -m venv venv
        echo -e "${GREEN}✅ Virtual environment oluşturuldu${NC}"
    fi
    
    source venv/bin/activate
    pip install -r requirements.txt
    echo -e "${GREEN}✅ Backend bağımlılıkları yüklendi${NC}"
    
    cd ..
}

# Frontend kurulumu
setup_frontend() {
    echo -e "${BLUE}📦 Frontend bağımlılıkları yükleniyor...${NC}"
    cd frontend
    
    if [ ! -d "node_modules" ]; then
        npm install
        echo -e "${GREEN}✅ Frontend bağımlılıkları yüklendi${NC}"
    else
        echo -e "${GREEN}✅ Frontend bağımlılıkları zaten mevcut${NC}"
    fi
    
    cd ..
}

# .env dosyası oluşturma
setup_env() {
    if [ ! -f "backend/.env" ]; then
        echo -e "${YELLOW}⚠️ .env dosyası bulunamadı. Örnek dosya kopyalanıyor...${NC}"
        cp backend/.env.example backend/.env
        echo -e "${YELLOW}ℹ️ Lütfen backend/.env dosyasını düzenleyip API anahtarlarınızı ekleyin.${NC}"
    else
        echo -e "${GREEN}✅ .env dosyası mevcut${NC}"
    fi
}

# Backend başlatma
start_backend() {
    echo -e "${BLUE}🔧 Backend başlatılıyor...${NC}"
    cd backend
    source venv/bin/activate
    
    # Arka planda başlat
    nohup python main.py > ../data/logs/backend.log 2>&1 &
    BACKEND_PID=$!
    echo $BACKEND_PID > ../data/logs/backend.pid
    
    sleep 3
    
    if ps -p $BACKEND_PID > /dev/null; then
        echo -e "${GREEN}✅ Backend http://localhost:8000 adresinde çalışıyor${NC}"
    else
        echo -e "${YELLOW}⚠️ Backend başlatılamadı. Logları kontrol edin: data/logs/backend.log${NC}"
    fi
    
    cd ..
}

# Frontend başlatma
start_frontend() {
    echo -e "${BLUE}🎨 Frontend başlatılıyor...${NC}"
    cd frontend
    
    # Arka planda başlat
    nohup npm run dev > ../data/logs/frontend.log 2>&1 &
    FRONTEND_PID=$!
    echo $FRONTEND_PID > ../data/logs/frontend.pid
    
    sleep 5
    
    if ps -p $FRONTEND_PID > /dev/null; then
        echo -e "${GREEN}✅ Frontend http://localhost:3000 adresinde çalışıyor${NC}"
    else
        echo -e "${YELLOW}⚠️ Frontend başlatılamadı. Logları kontrol edin: data/logs/frontend.log${NC}"
    fi
    
    cd ..
}

# Servisleri durdurma
stop_services() {
    echo -e "${BLUE}🛑 Servisler durduruluyor...${NC}"
    
    if [ -f "data/logs/backend.pid" ]; then
        kill $(cat data/logs/backend.pid) 2>/dev/null
        rm data/logs/backend.pid
        echo -e "${GREEN}✅ Backend durduruldu${NC}"
    fi
    
    if [ -f "data/logs/frontend.pid" ]; then
        kill $(cat data/logs/frontend.pid) 2>/dev/null
        rm data/logs/frontend.pid
        echo -e "${GREEN}✅ Frontend durduruldu${NC}"
    fi
    
    # Process kontrolü
    pkill -f "python.*main.py" 2>/dev/null
    pkill -f "npm.*run.*dev" 2>/dev/null
    
    echo -e "${GREEN}✅ Tüm servisler durduruldu${NC}"
}

# Kullanım bilgisi
show_usage() {
    echo "Kullanım: $0 [komut]"
    echo ""
    echo "Komutlar:"
    echo "  setup     - Tüm bağımlılıkları yükle"
    echo "  start     - Backend ve Frontend'i başlat"
    echo "  stop      - Tüm servisleri durdur"
    echo "  restart   - Yeniden başlat"
    echo "  status    - Çalışan servislerin durumunu göster"
    echo "  help      - Bu yardım mesajını göster"
    echo ""
}

# Servis durumu
show_status() {
    echo -e "${BLUE}📊 Servis Durumu${NC}"
    echo ""
    
    # Backend kontrolü
    if pgrep -f "python.*main.py" > /dev/null; then
        echo -e "${GREEN}✅ Backend: Çalışıyor (http://localhost:8000)${NC}"
    else
        echo -e "${YELLOW}❌ Backend: Çalışmıyor${NC}"
    fi
    
    # Frontend kontrolü
    if pgrep -f "npm.*run.*dev" > /dev/null; then
        echo -e "${GREEN}✅ Frontend: Çalışıyor (http://localhost:3000)${NC}"
    else
        echo -e "${YELLOW}❌ Frontend: Çalışmıyor${NC}"
    fi
    
    echo ""
}

# Ana program
case "${1:-help}" in
    setup)
        check_backend
        setup_env
        setup_backend
        if check_frontend; then
            setup_frontend
        fi
        echo ""
        echo -e "${GREEN}✅ Kurulum tamamlandı!${NC}"
        echo -e "${YELLOW}ℹ️ Şimdi 'bash start.sh start' komutuyla sistemi başlatabilirsiniz.${NC}"
        ;;
    
    start)
        start_backend
        if check_frontend; then
            start_frontend
        fi
        echo ""
        echo -e "${GREEN}✅ Sistem başlatıldı!${NC}"
        echo -e "Backend: http://localhost:8000"
        echo -e "Frontend: http://localhost:3000"
        ;;
    
    stop)
        stop_services
        ;;
    
    restart)
        stop_services
        sleep 2
        start_backend
        if check_frontend; then
            start_frontend
        fi
        ;;
    
    status)
        show_status
        ;;
    
    help|*)
        show_usage
        ;;
esac
