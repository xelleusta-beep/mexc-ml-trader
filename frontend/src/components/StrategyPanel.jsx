import { useState, useEffect, useCallback } from 'react'

const API = '/api/strategy'

export default function StrategyPanel() {
  const [status, setStatus] = useState(null)
  const [loading, setLoading] = useState(false)
  const [selectedPairs, setSelectedPairs] = useState('')
  const [result, setResult] = useState(null)

  const fetchStatus = useCallback(async () => {
    try {
      const resp = await fetch(API)
      const data = await resp.json()
      setStatus(data)
      setSelectedPairs(data.pairs?.join(', ') || '')
    } catch (e) {
      console.error('Strategy status error:', e)
    }
  }, [])

  useEffect(() => {
    fetchStatus()
  }, [fetchStatus])

  const handleSetStrategy = async (name) => {
    try {
      await fetch(`${API}/set/${name}`, { method: 'POST' })
      await fetchStatus()
    } catch (e) {
      console.error('Set strategy error:', e)
    }
  }

  const handleSetPairs = async () => {
    const pairs = selectedPairs.split(',').map(p => p.trim().toUpperCase()).filter(p => p)
    try {
      await fetch(`${API}/pairs`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ pairs }),
      })
      await fetchStatus()
    } catch (e) {
      console.error('Set pairs error:', e)
    }
  }

  const handleRun = async () => {
    setLoading(true)
    try {
      const resp = await fetch(`${API}/run`, { method: 'POST' })
      const data = await resp.json()
      setResult(data)
    } catch (e) {
      console.error('Run strategy error:', e)
    }
    setLoading(false)
  }

  if (!status) return <div className="strategy-loading">Yükleniyor...</div>

  const strategyInfo = {
    momentum: { name: 'Momentum', desc: 'RSI + MACD + EMA crossover ile trend takibi', icon: '🚀', color: '#00f0ff' },
    breakout: { name: 'Breakout', desc: 'Bollinger Band breakout + ADX filtresi', icon: '💥', color: '#ff3366' },
    mean_reversion: { name: 'Mean Reversion', desc: 'RSI + BB + Stochastic ile tersine dönüş', icon: '🔄', color: '#39ff14' },
  }

  return (
    <div className="strategy-panel">
      <div className="sp-header">
        <h2 className="gradient-text">
          <span className="sp-icon">⚡</span>
          Freqtrade Strateji Motoru
        </h2>
        <p className="sp-subtitle">Strateji seçin, çiftleri belirleyin, analiz çalıştırın</p>
      </div>

      <div className="sp-strategies">
        {Object.entries(strategyInfo).map(([key, info]) => (
          <button
            key={key}
            className={`sp-strategy-card depth-card ${status.strategy === key ? 'active' : ''}`}
            onClick={() => handleSetStrategy(key)}
            style={{ borderColor: status.strategy === key ? info.color : undefined }}
          >
            <span className="sp-strat-icon">{info.icon}</span>
            <span className="sp-strat-name">{info.name}</span>
            <span className="sp-strat-desc">{info.desc}</span>
            {status.strategy === key && <span className="sp-active-badge">AKTİF</span>}
          </button>
        ))}
      </div>

      <div className="sp-config depth-card">
        <h3>Çiftler</h3>
        <div className="sp-pairs-row">
          <input
            type="text"
            value={selectedPairs}
            onChange={(e) => setSelectedPairs(e.target.value)}
            placeholder="BTCUSDT, ETHUSDT, SOLUSDT..."
            className="sp-pairs-input"
          />
          <button className="sp-btn" onClick={handleSetPairs}>Kaydet</button>
        </div>
      </div>

      <div className="sp-actions">
        <button
          className="sp-btn sp-btn-run"
          onClick={handleRun}
          disabled={loading}
        >
          {loading ? '⏳ Analiz Çalışıyor...' : '▶ Analiz Çalıştır'}
        </button>
      </div>

      {result && (
        <div className="sp-result depth-card">
          <h3>📊 Analiz Sonucu</h3>
          <div className="sp-result-grid">
            <div className="sp-result-item">
              <span className="sp-label">Strateji</span>
              <span className="sp-value">{result.strategy}</span>
            </div>
            <div className="sp-result-item">
              <span className="sp-label">Çiftler</span>
              <span className="sp-value">{result.pairs_analyzed}</span>
            </div>
            <div className="sp-result-item">
              <span className="sp-label">Sinyaller</span>
              <span className="sp-value gradient-text">{result.signals_generated}</span>
            </div>
          </div>

          {result.pair_results && Object.entries(result.pair_results).map(([pair, info]) => (
            <div key={pair} className="sp-pair-result">
              <span className="sp-pair-name">{pair}</span>
              <span className={`sp-pair-dir ${info.direction}`}>
                {info.direction === 'long' ? '📈' : info.direction === 'short' ? '📉' : '⏸️'}
              </span>
              <span className="sp-pair-conf">{(info.confidence * 100).toFixed(0)}%</span>
            </div>
          ))}

          {result.signals?.length > 0 && (
            <div className="sp-signals-list">
              <h4>Sinyaller</h4>
              {result.signals.map((sig, i) => (
                <div key={i} className={`sp-signal ${sig.direction}`}>
                  <span className="sp-sig-pair">{sig.pair}</span>
                  <span className={`sp-sig-dir ${sig.direction}`}>
                    {sig.direction === 'long' ? '📈 LONG' : sig.direction === 'short' ? '📉 SHORT' : '🚪 EXIT'}
                  </span>
                  <span className="sp-sig-price">${sig.price.toLocaleString()}</span>
                  {sig.stoploss > 0 && <span className="sp-sig-sl">SL: ${sig.stoploss.toFixed(4)}</span>}
                  {sig.takeprofit > 0 && <span className="sp-sig-tp">TP: ${sig.takeprofit.toFixed(4)}</span>}
                  <span className="sp-sig-conf">{(sig.confidence * 100).toFixed(0)}%</span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {status.last_signals?.length > 0 && !result && (
        <div className="sp-last-signals depth-card">
          <h3>Son Sinyaller</h3>
          {status.last_signals.slice(-5).reverse().map((sig, i) => (
            <div key={i} className={`sp-signal ${sig.direction}`}>
              <span className="sp-sig-pair">{sig.pair}</span>
              <span className={`sp-sig-dir ${sig.direction}`}>
                {sig.direction === 'long' ? '📈' : sig.direction === 'short' ? '📉' : '🚪'}
              </span>
              <span className="sp-sig-price">${sig.price?.toLocaleString()}</span>
              <span className="sp-sig-conf">{(sig.confidence * 100).toFixed(0)}%</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
