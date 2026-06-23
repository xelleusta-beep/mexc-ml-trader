import { useState, useEffect, useRef, useCallback } from 'react'
import AgentStatus from './components/AgentStatus'
import MarketScanner from './components/MarketScanner'
import SignalPanel from './components/SignalPanel'
import PatronDecision from './components/PatronDecision'
import LivePositions from './components/LivePositions'
import RiskMetrics from './components/RiskMetrics'
import TradeHistory from './components/TradeHistory'
import SettingsPanel from './components/SettingsPanel'
import { getSystemStatus, getTradingLatest, startTrading, stopTrading, runSingleCycle, connectWebSocket } from './api/trading'

function App() {
  const [systemStatus, setSystemStatus] = useState(null)
  const [latestData, setLatestData] = useState(null)
  const [isRunning, setIsRunning] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [activeTab, setActiveTab] = useState('dashboard')
  const [wsConnected, setWsConnected] = useState(false)
  const [time, setTime] = useState(new Date())
  const wsRef = useRef(null)

  const fetchStatus = useCallback(async () => {
    try {
      const status = await getSystemStatus()
      setSystemStatus(status)
      setIsRunning(status.running)
    } catch (e) {}
  }, [])

  const fetchLatest = useCallback(async () => {
    try {
      const data = await getTradingLatest()
      if (data && Object.keys(data).length > 0) setLatestData(data)
    } catch (e) {}
  }, [])

  useEffect(() => {
    fetchStatus(); fetchLatest()
    const interval = setInterval(() => { fetchStatus(); fetchLatest() }, 5000)
    return () => clearInterval(interval)
  }, [fetchStatus, fetchLatest])

  useEffect(() => {
    const t = setInterval(() => setTime(new Date()), 1000)
    return () => clearInterval(t)
  }, [])

  useEffect(() => {
    wsRef.current = connectWebSocket((data) => {
      if (data.type === 'connected') setWsConnected(true)
      else if (data.type === 'disconnected') setWsConnected(false)
      else if (data.cycle) setLatestData(data)
    })
    return () => { if (wsRef.current && wsRef.current.close) wsRef.current.close() }
  }, [])

  const handleStart = async () => {
    setLoading(true); setError(null)
    try { await startTrading(); setIsRunning(true); fetchStatus() }
    catch (e) { setError(e.message) } finally { setLoading(false) }
  }

  const handleStop = async () => {
    setLoading(true); setError(null)
    try { await stopTrading(); setIsRunning(false); fetchStatus() }
    catch (e) { setError(e.message) } finally { setLoading(false) }
  }

  const handleRunCycle = async () => {
    setLoading(true); setError(null)
    try { const r = await runSingleCycle(); setLatestData(r); fetchStatus() }
    catch (e) { setError(e.message) } finally { setLoading(false) }
  }

  const formatTime = (d) => d.toLocaleTimeString('tr-TR', { hour: '2-digit', minute: '2-digit', second: '2-digit' })

  const tabs = [
    { id: 'dashboard', label: 'KUMANDA MERKEZİ', icon: '◈' },
    { id: 'scanner', label: 'TARAMA', icon: '◎' },
    { id: 'signals', label: 'SİNYALLER', icon: '◇' },
    { id: 'positions', label: 'POZİSYONLAR', icon: '⬡' },
    { id: 'history', label: 'GEÇMİŞ', icon: '▣' },
    { id: 'settings', label: 'AYARLAR', icon: '⚙' },
  ]

  const totalEquity = latestData?.portfolio?.total_equity || systemStatus?.total_equity || 10000
  const usedCapital = latestData?.portfolio?.total_exposure_usd || (totalEquity - (latestData?.portfolio?.available_capital || systemStatus?.available_capital || totalEquity))

  return (
    <div className="min-h-screen bg-[#030712] grid-bg hex-pattern">
      {/* HEADER */}
      <header className="border-b border-cyan-500/10 bg-[#030712]/90 backdrop-blur-xl sticky top-0 z-50">
        <div className="max-w-[1920px] mx-auto px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-6">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-cyan-500/20 to-purple-500/20 border border-cyan-500/30 flex items-center justify-center">
                <span className="font-orbitron text-lg neon-cyan">M</span>
              </div>
              <div>
                <h1 className="font-orbitron text-lg font-bold tracking-widest">
                  <span className="neon-cyan">MEXC</span>
                  <span className="text-gray-400 ml-2">TRADING</span>
                  <span className="neon-purple ml-2">NEXUS</span>
                </h1>
                <p className="text-sm text-cyan-500/50 tracking-wider">MULTI-AGENT NEURAL NETWORK</p>
              </div>
            </div>

            <div className="flex items-center gap-4 ml-4 pl-4 border-l border-cyan-500/10">
              <div className="flex items-center gap-2">
                <div className={`status-dot ${wsConnected ? 'bg-green-400 text-green-400' : 'bg-red-400 text-red-400'}`} />
                <span className="text-sm tracking-wider font-semibold" style={{ color: wsConnected ? '#39ff14' : '#ff0040' }}>
                  {wsConnected ? 'ONLINE' : 'OFFLINE'}
                </span>
              </div>
              <div className="text-sm text-cyan-500/60 font-semibold">
                {formatTime(time)}
              </div>
            </div>
          </div>

          <div className="flex items-center gap-6">
            {/* EQUITY DISPLAY */}
            <div className="flex items-center gap-6 px-5 py-2 glass-panel rounded-lg">
              <div>
                <p className="text-label text-cyan-500/60">PORTFÖY</p>
                <p className="text-value neon-cyan">${totalEquity.toLocaleString()}</p>
              </div>
              <div className="w-px h-8 bg-cyan-500/10" />
              <div>
                <p className="text-label text-green-500/60">KULLANILAN</p>
                <p className="text-value text-green-400">${usedCapital.toLocaleString()}</p>
              </div>
              <div className="w-px h-8 bg-cyan-500/10" />
              <div>
                <p className="text-label text-purple-500/60">POZİSYON</p>
                <p className="text-value neon-purple">{latestData?.portfolio?.position_count || systemStatus?.open_positions || 0}/5</p>
              </div>
            </div>

            {/* CONTROLS */}
            <div className="flex items-center gap-2">
              {!isRunning ? (
                <button onClick={handleStart} disabled={loading} className="btn-neon btn-neon-green">
                  ▶ BAŞLAT
                </button>
              ) : (
                <button onClick={handleStop} disabled={loading} className="btn-neon btn-neon-red">
                  ■ DURDUR
                </button>
              )}
              <button onClick={handleRunCycle} disabled={loading || isRunning} className="btn-neon btn-neon-purple">
                ▷ TEK DÖNGÜ
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* TABS */}
      <div className="border-b border-cyan-500/10 bg-[#030712]/80 backdrop-blur-sm">
        <div className="max-w-[1920px] mx-auto px-4 flex gap-1">
          {tabs.map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`px-5 py-3 text-sm font-bold tracking-wider transition-all border-b-2 flex items-center gap-2 ${
                activeTab === tab.id
                  ? 'border-cyan-400 text-cyan-400 bg-cyan-500/5'
                  : 'border-transparent text-gray-500 hover:text-gray-300'
              }`}
            >
              <span className="text-base">{tab.icon}</span>
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      {/* ERROR */}
      {error && (
        <div className="max-w-[1920px] mx-auto px-4 mt-4">
          <div className="glass-panel glass-panel-red p-4 rounded-lg flex items-center gap-3">
            <span className="neon-red text-lg">⚠</span>
            <span className="text-red-300 text-base">{error}</span>
            <button onClick={() => setError(null)} className="ml-auto text-red-400 hover:text-red-300 text-lg">✕</button>
          </div>
        </div>
      )}

      {/* MAIN CONTENT */}
      <main className="max-w-[1920px] mx-auto p-4">
        {activeTab === 'dashboard' && (
          <div className="grid grid-cols-12 gap-3">
            <div className="col-span-3">
              <AgentStatus status={systemStatus} />
            </div>
            <div className="col-span-5">
              <PatronDecision data={latestData?.patron} />
            </div>
            <div className="col-span-4">
              <RiskMetrics data={latestData?.risk} portfolio={latestData?.portfolio} sentiment={latestData?.sentiment} />
            </div>
            <div className="col-span-7">
              <LivePositions positions={latestData?.positions || latestData?.executed} portfolio={latestData?.portfolio} />
            </div>
            <div className="col-span-5">
              <SignalPanel data={latestData?.patron} />
            </div>
          </div>
        )}
        {activeTab === 'scanner' && <MarketScanner data={latestData?.scanner} />}
        {activeTab === 'signals' && <SignalPanel data={latestData?.patron} fullPage />}
        {activeTab === 'positions' && <LivePositions positions={latestData?.positions || latestData?.executed} portfolio={latestData?.portfolio} fullPage />}
        {activeTab === 'history' && <TradeHistory data={latestData} />}
        {activeTab === 'settings' && <SettingsPanel />}
      </main>
    </div>
  )
}

export default App
