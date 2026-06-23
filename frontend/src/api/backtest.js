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
              if (data.type === 'progress') {
                onProgress(data);
              } else if (data.type === 'complete') {
                onComplete(data);
              }
            } catch (e) {}
          }
        }

        processStream();
      }).catch(err => {
        if (err.name !== 'AbortError') {
          onError(err.message);
        }
      });
    }

    processStream();
  }).catch(err => {
    if (err.name !== 'AbortError') {
      onError(err.message);
    }
  });

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
