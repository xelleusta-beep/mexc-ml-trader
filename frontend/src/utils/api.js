import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000/api';

export const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
});

// Market endpoints
export const marketAPI = {
  scan: () => api.get('/market/scan'),
  getTickers: () => api.get('/market/tickers'),
};

// ML endpoints
export const mlAPI = {
  train: (symbol) => api.post(`/ml/train/${symbol}`),
  predict: (symbol) => api.get(`/ml/predict/${symbol}`),
};

// Trading endpoints
export const tradingAPI = {
  getPositions: () => api.get('/trading/positions'),
  execute: (symbol) => api.post(`/trading/execute/${symbol}`),
  close: (symbol) => api.delete(`/trading/close/${symbol}`),
};

// Account endpoints
export const accountAPI = {
  getInfo: () => api.get('/account/info'),
};

// WebSocket helper
export class WebSocketClient {
  constructor(url = 'ws://localhost:8000/ws') {
    this.url = url;
    this.ws = null;
    this.listeners = new Map();
  }

  connect() {
    this.ws = new WebSocket(this.url);

    this.ws.onopen = () => {
      console.log('WebSocket connected');
      this.emit('connected');
    };

    this.ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        this.emit(data.type, data);
      } catch (e) {
        console.error('WebSocket message parse error:', e);
      }
    };

    this.ws.onclose = () => {
      console.log('WebSocket disconnected');
      this.emit('disconnected');
      // Auto-reconnect after 5 seconds
      setTimeout(() => this.connect(), 5000);
    };

    this.ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      this.emit('error', error);
    };
  }

  subscribe() {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: 'subscribe' }));
    }
  }

  on(event, callback) {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, []);
    }
    this.listeners.get(event).push(callback);
  }

  off(event, callback) {
    if (this.listeners.has(event)) {
      const index = this.listeners.get(event).indexOf(callback);
      if (index > -1) {
        this.listeners.get(event).splice(index, 1);
      }
    }
  }

  emit(event, data) {
    if (this.listeners.has(event)) {
      this.listeners.get(event).forEach((callback) => callback(data));
    }
  }

  disconnect() {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }
}

export const wsClient = new WebSocketClient();
