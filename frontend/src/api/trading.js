const API_BASE = '/api';

export async function fetchSymbols() {
  const resp = await fetch(`${API_BASE}/symbols`);
  if (!resp.ok) throw new Error('Semboller yüklenemedi');
  const data = await resp.json();
  return data.symbols;
}

export async function runBacktest(params) {
  const resp = await fetch(`${API_BASE}/backtest`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  if (!resp.ok) {
    const err = await resp.json();
    throw new Error(err.detail || 'Backtest çalıştırılamadı');
  }
  return resp.json();
}

export async function runMultiBacktest(params) {
  const resp = await fetch(`${API_BASE}/backtest/multi`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  if (!resp.ok) {
    const err = await resp.json();
    throw new Error(err.detail || 'Multi-backtest çalıştırılamadı');
  }
  return resp.json();
}

export function runBacktestStream(params, onProgress, onComplete, onError) {
  const controller = new AbortController();
  fetch(`${API_BASE}/backtest/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
    signal: controller.signal,
  }).then(response => {
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    function processStream() {
      reader.read().then(({ done, value }) => {
        if (done) return;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop();
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));
              if (data.type === 'progress') onProgress(data);
              else if (data.type === 'complete') onComplete(data);
            } catch (e) {}
          }
        }
        processStream();
      }).catch(err => { if (err.name !== 'AbortError') onError(err.message); });
    }
    processStream();
  }).catch(err => { if (err.name !== 'AbortError') onError(err.message); });
  return { abort: () => controller.abort() };
}

export async function runOptimization(params) {
  const resp = await fetch(`${API_BASE}/optimize`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  if (!resp.ok) {
    const err = await resp.json();
    throw new Error(err.detail || 'Optimizasyon çalıştırılamadı');
  }
  return resp.json();
}

export async function runPortfolio(params) {
  const resp = await fetch(`${API_BASE}/portfolio`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  if (!resp.ok) {
    const err = await resp.json();
    throw new Error(err.detail || 'Portföy simülasyonu çalıştırılamadı');
  }
  return resp.json();
}

export async function getSystemStatus() {
  const resp = await fetch(`${API_BASE}/system/status`);
  if (!resp.ok) throw new Error('Sistem durumu alınamadı');
  return resp.json();
}

export async function getAgentsStatus() {
  const resp = await fetch(`${API_BASE}/agents/status`);
  if (!resp.ok) throw new Error('Ajan durumları alınamadı');
  return resp.json();
}

export async function getScannerHot() {
  const resp = await fetch(`${API_BASE}/scanner/hot`);
  if (!resp.ok) throw new Error('Scanner verisi alınamadı');
  return resp.json();
}

export async function getTechnicalSignals() {
  const resp = await fetch(`${API_BASE}/technical/signals`);
  if (!resp.ok) throw new Error('Teknik sinyaller alınamadı');
  return resp.json();
}

export async function getSentimentCurrent() {
  const resp = await fetch(`${API_BASE}/sentiment/current`);
  if (!resp.ok) throw new Error('Sentiment verisi alınamadı');
  return resp.json();
}

export async function getRiskMetrics() {
  const resp = await fetch(`${API_BASE}/risk/metrics`);
  if (!resp.ok) throw new Error('Risk metrikleri alınamadı');
  return resp.json();
}

export async function getPortfolioPositions() {
  const resp = await fetch(`${API_BASE}/portfolio/positions`);
  if (!resp.ok) throw new Error('Pozisyon bilgileri alınamadı');
  return resp.json();
}

export async function getPatronDecisions() {
  const resp = await fetch(`${API_BASE}/patron/decisions`);
  if (!resp.ok) throw new Error('Patron kararları alınamadı');
  return resp.json();
}

export async function getTradingHistory() {
  const resp = await fetch(`${API_BASE}/trading/history`);
  if (!resp.ok) throw new Error('İşlem geçmişi alınamadı');
  return resp.json();
}

export async function getTradingLatest() {
  const resp = await fetch(`${API_BASE}/trading/latest`);
  if (!resp.ok) throw new Error('Son sonuçlar alınamadı');
  return resp.json();
}

export async function startTrading() {
  const resp = await fetch(`${API_BASE}/trading/start`, { method: 'POST' });
  if (!resp.ok) throw new Error('Trading başlatılamadı');
  return resp.json();
}

export async function stopTrading() {
  const resp = await fetch(`${API_BASE}/trading/stop`, { method: 'POST' });
  if (!resp.ok) throw new Error('Trading durdurulamadı');
  return resp.json();
}

export async function runSingleCycle() {
  const resp = await fetch(`${API_BASE}/trading/cycle`, { method: 'POST' });
  if (!resp.ok) throw new Error('Döngü çalıştırılamadı');
  return resp.json();
}

export async function updateEquity(equity) {
  const resp = await fetch(`${API_BASE}/trading/equity`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(equity),
  });
  if (!resp.ok) throw new Error('Sermaye güncellenemedi');
  return resp.json();
}

export async function checkBackend() {
  try {
    const resp = await fetch('/api/health', { signal: AbortSignal.timeout(3000) });
    return resp.ok;
  } catch {
    return false;
  }
}

export function connectWebSocket(onMessage) {
  const WS_PORT = 8000;
  const wsUrl = `ws://localhost:${WS_PORT}/ws/live`;
  let ws = null;
  let reconnectTimer = null;
  let reconnectDelay = 2000;
  let reconnectAttempts = 0;
  const MAX_DELAY = 30000;
  const MAX_ATTEMPTS = 30;
  let stopped = false;

  async function connect() {
    if (stopped) return;

    const alive = await checkBackend();
    if (!alive) {
      onMessage({ type: 'disconnected' });
      scheduleReconnect();
      return;
    }

    try {
      ws = new WebSocket(wsUrl);
    } catch (e) {
      scheduleReconnect();
      return;
    }

    ws.onopen = () => {
      reconnectDelay = 2000;
      reconnectAttempts = 0;
      onMessage({ type: 'connected' });
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        onMessage(data);
      } catch (e) {}
    };

    ws.onclose = () => {
      onMessage({ type: 'disconnected' });
      scheduleReconnect();
    };

    ws.onerror = () => {};
  }

  function scheduleReconnect() {
    if (stopped) return;
    if (reconnectAttempts >= MAX_ATTEMPTS) return;

    clearTimeout(reconnectTimer);
    reconnectAttempts++;
    const jitter = Math.random() * 1000;
    const delay = Math.min(reconnectDelay + jitter, MAX_DELAY);
    reconnectDelay = Math.min(reconnectDelay * 1.3, MAX_DELAY);

    reconnectTimer = setTimeout(() => {
      connect();
    }, delay);
  }

  connect();

  return {
    close: () => {
      stopped = true;
      clearTimeout(reconnectTimer);
      if (ws) {
        ws.onclose = null;
        ws.onerror = null;
        ws.close();
      }
    }
  };
}
