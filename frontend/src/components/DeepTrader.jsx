import { useState, useEffect, useCallback } from 'react'
import { createSeriesMarkers } from 'lightweight-charts'

const API = '/api/deep-trader'

export default function DeepTrader({ symbol = 'BTCUSDT' }) {
  const [status, setStatus] = useState(null)
  const [analysis, setAnalysis] = useState(null)
  const [loading, setLoading] = useState(false)
  const [training, setTraining] = useState(false)
  const [activeTab, setActiveTab] = useState('overview')
  const [selectedSymbol, setSelectedSymbol] = useState(symbol)

  const symbols = ['BTCUSDT', 'ETHUSDT']

  const fetchStatus = useCallback(async () => {
    try {
      const resp = await fetch(`${API}/${selectedSymbol}`)
      const data = await resp.json()
      setStatus(data)
    } catch (e) {
      console.error('Deep trader status error:', e)
    }
  }, [selectedSymbol])

  useEffect(() => {
    fetchStatus()
    const interval = setInterval(fetchStatus, 15000)
    return () => clearInterval(interval)
  }, [fetchStatus])

  const handleTrain = async () => {
    setTraining(true)
    try {
      const resp = await fetch(`${API}/${selectedSymbol}/train`, { method: 'POST' })
      const data = await resp.json()
      setStatus(data.status)
    } catch (e) {
      console.error('Train error:', e)
    }
    setTraining(false)
  }

  const handleAnalyze = async () => {
    setLoading(true)
    try {
      const resp = await fetch(`${API}/${selectedSymbol}/analyze`, { method: 'POST' })
      const data = await resp.json()
      setAnalysis(data)
      await fetchStatus()
    } catch (e) {
      console.error('Analyze error:', e)
    }
    setLoading(false)
  }

  const formatTime = (ts) => {
    if (!ts) return '-'
    const d = new Date(ts * 1000)
    return d.toLocaleString('tr-TR', { hour: '2-digit', minute: '2-digit', day: '2-digit', month: '2-digit' })
  }

  const formatDuration = (start) => {
    if (!start) return '-'
    const diff = Date.now() / 1000 - start
    const hours = Math.floor(diff / 3600)
    const mins = Math.floor((diff % 3600) / 60)
    if (hours > 0) return `${hours}sa ${mins}dk`
    return `${mins}dk`
  }

  if (!status) return (
    <div className="deep-trader-loading">
      <div className="loading-spinner" />
      <span>Yükleniyor...</span>
    </div>
  )

  return (
    <div className="deep-trader">
      <div className="dt-header">
        <div className="dt-title-row">
          <h2 className="gradient-text">
            <span className="dt-icon">🧠</span>
            Deep Learning Trader
          </h2>
          <div className="dt-symbol-selector">
            {symbols.map(s => (
              <button
                key={s}
                className={`dt-symbol-btn ${selectedSymbol === s ? 'active' : ''}`}
                onClick={() => setSelectedSymbol(s)}
              >
                {s.replace('USDT', '')}
              </button>
            ))}
          </div>
        </div>
        <div className="dt-equity-row">
          <div className="dt-equity-card depth-card">
            <span className="dt-label">Toplam Equity</span>
            <span className="dt-value gradient-text">${status.total_equity.toLocaleString()}</span>
          </div>
          <div className="dt-equity-card depth-card">
            <span className="dt-label">Kullanılabilir</span>
            <span className="dt-value">${status.available_capital.toLocaleString()}</span>
          </div>
          <div className="dt-equity-card depth-card">
            <span className="dt-label">İşlem Sayısı</span>
            <span className="dt-value">{status.trade_count}</span>
          </div>
          <div className="dt-equity-card depth-card">
            <span className="dt-label">Model Durumu</span>
            <span className={`dt-value ${status.model_trained ? 'status-online' : 'status-offline'}`}>
              {status.model_trained ? 'Eğitildi' : 'Eğitilmedi'}
            </span>
          </div>
        </div>
        <div className="dt-actions">
          <button
            className="dt-btn dt-btn-train"
            onClick={handleTrain}
            disabled={training}
          >
            {training ? '⏳ Eğitiliyor...' : '🎓 Model Eğit'}
          </button>
          <button
            className="dt-btn dt-btn-analyze"
            onClick={handleAnalyze}
            disabled={loading || !status.model_trained}
          >
            {loading ? '⏳ Analiz...' : '🔍 Analiz & İşlem'}
          </button>
        </div>
      </div>

      <div className="dt-tabs">
        <button className={`dt-tab ${activeTab === 'overview' ? 'active' : ''}`} onClick={() => setActiveTab('overview')}>Genel Bakış</button>
        <button className={`dt-tab ${activeTab === 'positions' ? 'active' : ''}`} onClick={() => setActiveTab('positions')}>Açık Pozisyonlar</button>
        <button className={`dt-tab ${activeTab === 'history' ? 'active' : ''}`} onClick={() => setActiveTab('history')}>İşlem Geçmişi</button>
      </div>

      {activeTab === 'overview' && (
        <div className="dt-overview">
          {analysis && analysis.analysis && (
            <div className="dt-analysis depth-card">
              <h3>📊 Model Analizi</h3>
              <div className="dt-analysis-grid">
                <div className="dt-analysis-item">
                  <span className="dt-label">Tahmin</span>
                  <span className={`dt-prediction ${analysis.analysis.ensemble_prediction === 1 ? 'long' : analysis.analysis.ensemble_prediction === -1 ? 'short' : 'hold'}`}>
                    {analysis.analysis.ensemble_prediction === 1 ? '📈 LONG' : analysis.analysis.ensemble_prediction === -1 ? '📉 SHORT' : '⏸️ HOLD'}
                  </span>
                </div>
                <div className="dt-analysis-item">
                  <span className="dt-label">Güven</span>
                  <span className="dt-value">{(analysis.analysis.ensemble_confidence * 100).toFixed(1)}%</span>
                </div>
                <div className="dt-analysis-item">
                  <span className="dt-label">RF Tahmin</span>
                  <span className="dt-value">
                    {analysis.analysis.rf_prediction === 1 ? 'LONG' : analysis.analysis.rf_prediction === -1 ? 'SHORT' : 'HOLD'}
                    <small> ({(analysis.analysis.rf_confidence * 100).toFixed(1)}%)</small>
                  </span>
                </div>
                <div className="dt-analysis-item">
                  <span className="dt-label">GB Tahmin</span>
                  <span className="dt-value">
                    {analysis.analysis.gb_prediction === 1 ? 'LONG' : analysis.analysis.gb_prediction === -1 ? 'SHORT' : 'HOLD'}
                    <small> ({(analysis.analysis.gb_confidence * 100).toFixed(1)}%)</small>
                  </span>
                </div>
                <div className="dt-analysis-item">
                  <span className="dt-label">RSI</span>
                  <span className="dt-value">{analysis.analysis.rsi}</span>
                </div>
                <div className="dt-analysis-item">
                  <span className="dt-label">ADX</span>
                  <span className="dt-value">{analysis.analysis.adx}</span>
                </div>
              </div>
              {analysis.analysis.support_levels?.length > 0 && (
                <div className="dt-levels">
                  <div className="dt-level-group">
                    <span className="dt-label">Destek</span>
                    <div className="dt-level-values">
                      {analysis.analysis.support_levels.map((s, i) => (
                        <span key={i} className="dt-level support">${s.toLocaleString()}</span>
                      ))}
                    </div>
                  </div>
                  <div className="dt-level-group">
                    <span className="dt-label">Direnç</span>
                    <div className="dt-level-values">
                      {analysis.analysis.resistance_levels.map((r, i) => (
                        <span key={i} className="dt-level resistance">${r.toLocaleString()}</span>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}

          {analysis?.action && analysis.action !== 'hold' && (
            <div className="dt-action-result depth-card">
              <h3>⚡ İşlem Sonucu</h3>
              <div className={`dt-action-badge ${analysis.action.startsWith('opened') ? 'opened' : 'closed'}`}>
                {analysis.action}
              </div>
              {analysis.new_position && (
                <div className="dt-new-pos">
                  <span>Giriş: ${analysis.new_position.entry_price.toLocaleString()}</span>
                  <span>TP: ${analysis.new_position.tp_price.toLocaleString()}</span>
                  <span>SL: ${analysis.new_position.sl_price.toLocaleString()}</span>
                  <span>Kaldıraç: {analysis.new_position.leverage}x</span>
                  <span>Büyüklük: ${analysis.new_position.size_usd.toLocaleString()}</span>
                  <span>Güven: {(analysis.new_position.model_confidence * 100).toFixed(1)}%</span>
                </div>
              )}
              {analysis.closed && (
                <div className="dt-closed-info">
                  <span>Kapanış: ${analysis.closed.exit_price?.toLocaleString()}</span>
                  <span>PnL: ${analysis.closed.pnl}</span>
                  <span>Sebep: {analysis.closed.reason}</span>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {activeTab === 'positions' && (
        <div className="dt-positions">
          {status.positions.length === 0 ? (
            <div className="dt-empty depth-card">Açık pozisyon yok</div>
          ) : (
            status.positions.map((pos, i) => (
              <div key={i} className={`dt-position depth-card ${pos.direction}`}>
                <div className="dt-pos-header">
                  <span className={`dt-direction ${pos.direction}`}>
                    {pos.direction === 'long' ? '📈 LONG' : '📉 SHORT'}
                  </span>
                  <span className="dt-pos-symbol">{pos.symbol}</span>
                  <span className="dt-pos-duration">{formatDuration(pos.entry_time)}</span>
                </div>
                <div className="dt-pos-details">
                  <div className="dt-pos-row">
                    <span>Giriş: ${pos.entry_price.toLocaleString()}</span>
                    <span>Güncel: ${pos.current_price.toLocaleString()}</span>
                  </div>
                  <div className="dt-pos-row">
                    <span>TP: ${pos.tp_price.toLocaleString()}</span>
                    <span>SL: ${pos.sl_price.toLocaleString()}</span>
                  </div>
                  <div className="dt-pos-row">
                    <span>Kaldıraç: {pos.leverage}x</span>
                    <span>Büyüklük: ${pos.size_usd.toLocaleString()}</span>
                  </div>
                  <div className="dt-pos-row">
                    <span>Güven: {(pos.model_confidence * 100).toFixed(1)}%</span>
                    <span className={pos.unrealized_pnl >= 0 ? 'pnl-positive' : 'pnl-negative'}>
                      PnL: ${pos.unrealized_pnl}
                    </span>
                  </div>
                </div>
                <div className="dt-pos-signals">
                  {pos.signals_used?.map((sig, j) => (
                    <span key={j} className="dt-signal-tag">{sig}</span>
                  ))}
                </div>
              </div>
            ))
          )}
        </div>
      )}

      {activeTab === 'history' && (
        <div className="dt-history">
          {status.recent_trades.length === 0 ? (
            <div className="dt-empty depth-card">İşlem geçmişi yok</div>
          ) : (
            <div className="dt-history-table">
              <div className="dt-history-header">
                <span>Zaman</span>
                <span>Yön</span>
                <span>Giriş</span>
                <span>Çıkış</span>
                <span>Kaldıraç</span>
                <span>PnL</span>
                <span>Sebep</span>
              </div>
              {status.recent_trades.slice().reverse().map((trade, i) => (
                <div key={i} className={`dt-history-row ${trade.direction}`}>
                  <span>{formatTime(trade.entry_time)}</span>
                  <span className={trade.direction}>
                    {trade.direction === 'long' ? '📈' : '📉'}
                  </span>
                  <span>${trade.entry_price?.toLocaleString()}</span>
                  <span>${trade.exit_price?.toLocaleString()}</span>
                  <span>{trade.leverage}x</span>
                  <span className={trade.pnl >= 0 ? 'pnl-positive' : 'pnl-negative'}>
                    ${trade.pnl}
                  </span>
                  <span className={`dt-reason ${trade.reason?.toLowerCase()}`}>
                    {trade.reason}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
