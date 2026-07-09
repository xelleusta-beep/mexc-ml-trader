import { useState, useEffect, useRef, useCallback } from 'react'
import AgentStatus from './components/AgentStatus'
import MarketScanner from './components/MarketScanner'
import SignalPanel from './components/SignalPanel'
import PatronDecision from './components/PatronDecision'
import LivePositions from './components/LivePositions'
import RiskMetrics from './components/RiskMetrics'
import TradeHistory from './components/TradeHistory'
import SettingsPanel from './components/SettingsPanel'
import KeyAuth from './components/KeyAuth'
import { getSystemStatus, getTradingLatest, getTradingHistory, getSentimentCurrent, startTrading, stopTrading, runSingleCycle, connectWebSocket } from './api/trading'

function App() {
  const [authenticated, setAuthenticated] = useState(() => localStorage.getItem('mexc_auth') === 'true')
  const [showLogin, setShowLogin] = useState(false)
  const [systemStatus, setSystemStatus] = useState(null)
  const [latestData, setLatestData] = useState(null)
  const [tradeHistory, setTradeHistory] = useState([])
  const [sentimentData, setSentimentData] = useState(null)
  const [isRunning, setIsRunning] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [activeTab, setActiveTab] = useState('dashboard')
  const [wsConnected, setWsConnected] = useState(false)
  const [time, setTime] = useState(new Date())
  const [prevEquity, setPrevEquity] = useState(null)
  const [equityFlash, setEquityFlash] = useState(null)
  const wsRef = useRef(null)

  const fetchStatus = useCallback(async () => {
    try {
      const status = await getSystemStatus()
      setSystemStatus(status)
      setIsRunning(status.running)
    } catch (e) {
      setError('Sistem durumu alınamadı')
    }
  }, [])

  const fetchLatest = useCallback(async () => {
    try {
      const data = await getTradingLatest()
      if (data && Object.keys(data).length > 0) {
        const newEquity = data.portfolio?.total_equity || data.total_equity
        if (prevEquity !== null && newEquity && newEquity !== prevEquity) {
          setEquityFlash(newEquity > prevEquity ? 'up' : 'down')
          setTimeout(() => setEquityFlash(null), 1500)
        }
        if (newEquity) setPrevEquity(newEquity)
        setLatestData(data)
      }
    } catch (e) {}
  }, [prevEquity])

  const fetchHistory = useCallback(async () => {
    try {
      const hist = await getTradingHistory()
      if (hist?.history) setTradeHistory(hist.history)
    } catch (e) {}
  }, [])

  const fetchSentiment = useCallback(async () => {
    try {
      const sent = await getSentimentCurrent()
      setSentimentData(sent)
    } catch (e) {}
  }, [])

  useEffect(() => {
    fetchStatus(); fetchLatest(); fetchHistory(); fetchSentiment()
    const interval = setInterval(() => {
      fetchStatus(); fetchLatest(); fetchHistory(); fetchSentiment()
    }, 5000)
    const positionInterval = setInterval(() => {
      fetchLatest()
    }, 10000)
    return () => { clearInterval(interval); clearInterval(positionInterval) }
  }, [fetchStatus, fetchLatest, fetchHistory, fetchSentiment])

  useEffect(() => {
    const t = setInterval(() => setTime(new Date()), 1000)
    return () => clearInterval(t)
  }, [])

  useEffect(() => {
    wsRef.current = connectWebSocket((data) => {
      if (data.type === 'connected') setWsConnected(true)
      else if (data.type === 'disconnected') setWsConnected(false)
      else if (data.type === 'position_update') {
        setLatestData(prev => ({
          ...prev,
          positions: data.positions,
          trade_history: data.trade_history,
          portfolio: data.portfolio,
        }))
      } else if (data.cycle) setLatestData(data)
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
    try { const r = await runSingleCycle(); setLatestData(r); fetchStatus(); fetchHistory() }
    catch (e) { setError(e.message) } finally { setLoading(false) }
  }

  const formatTime = (d) => d.toLocaleTimeString('tr-TR', { hour: '2-digit', minute: '2-digit', second: '2-digit' })

  const tabs = [
    { id: 'dashboard', label: 'KUMANDA MERKEZİ', icon: '◈', auth: false },
    { id: 'scanner', label: 'TARAMA', icon: '◎', auth: false },
    { id: 'signals', label: 'SİNYALLER', icon: '◇', auth: false },
    { id: 'positions', label: 'POZİSYONLAR', icon: '⬡', auth: false },
    { id: 'history', label: 'GEÇMİŞ', icon: '▣', auth: false },
    { id: 'settings', label: 'AYARLAR', icon: '⚙', auth: true },
  ].filter(tab => !tab.auth || authenticated)

  const totalEquity = latestData?.portfolio?.total_equity || systemStatus?.total_equity || 10000
  const usedCapital = latestData?.portfolio?.total_exposure_usd || (totalEquity - (latestData?.portfolio?.available_capital || systemStatus?.available_capital || totalEquity))
  const positionCount = latestData?.positions?.length || latestData?.executed?.length || systemStatus?.open_positions || 0

  return (
    <div className="min-h-screen bg-[#030712] grid-bg hex-pattern">
      {/* HEADER */}
      <header className="border-b border-cyan-500/10 bg-[#030712]/90 backdrop-blur-xl sticky top-0 z-50">
        <div className="max-w-[1920px] mx-auto px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-4 md:gap-6">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-cyan-500/20 to-purple-500/20 border border-cyan-500/30 flex items-center justify-center">
                <span className="font-orbitron text-lg neon-cyan">M</span>
              </div>
              <div>
                <h1 className="font-orbitron text-base md:text-lg font-bold tracking-widest">
                  <span className="neon-cyan">MEXC</span>
                  <span className="text-gray-400 ml-2 hidden sm:inline">TRADING</span>
                  <span className="neon-purple ml-2">NEXUS</span>
                </h1>
                <p className="text-xs text-cyan-500/50 tracking-wider hidden md:block">MULTI-AGENT NEURAL NETWORK</p>
              </div>
            </div>

            <div className="flex items-center gap-3 md:gap-4 ml-2 md:ml-4 pl-4 border-l border-cyan-500/10">
              <div className="flex items-center gap-2">
                <div className={`status-dot ${wsConnected ? 'bg-green-400 text-green-400' : 'bg-red-400 text-red-400'}`} />
                <span className="text-xs md:text-sm tracking-wider font-semibold" style={{ color: wsConnected ? '#39ff14' : '#ff0040' }}>
                  {wsConnected ? 'ONLINE' : 'OFFLINE'}
                </span>
              </div>
              <div className="text-xs md:text-sm text-cyan-500/60 font-semibold hidden sm:block">
                {formatTime(time)}
              </div>
            </div>
          </div>

          <div className="flex items-center gap-3 md:gap-6">
            {/* EQUITY DISPLAY */}
            <div className="flex items-center gap-3 md:gap-6 px-3 md:px-5 py-2 glass-panel rounded-lg">
              <div>
                <p className="text-label text-cyan-500/60">PORTFÖY</p>
                <p className={`text-value neon-cyan transition-all ${equityFlash === 'up' ? 'flash-green' : equityFlash === 'down' ? 'flash-red' : ''}`}>
                  ${totalEquity.toLocaleString()}
                </p>
              </div>
              <div className="w-px h-8 bg-cyan-500/10 hidden sm:block" />
              <div className="hidden sm:block">
                <p className="text-label text-green-500/60">KULLANILAN</p>
                <p className="text-value text-green-400">${usedCapital.toLocaleString()}</p>
              </div>
              <div className="w-px h-8 bg-cyan-500/10 hidden sm:block" />
              <div className="hidden sm:block">
                <p className="text-label text-purple-500/60">POZİSYON</p>
                <p className="text-value neon-purple">{positionCount}/5</p>
              </div>
            </div>

            {/* CONTROLS */}
            <div className="flex items-center gap-2">
              {authenticated ? (
                <>
                  {!isRunning ? (
                    <button onClick={handleStart} disabled={loading} className="btn-neon btn-neon-green text-xs md:text-sm">
                      ▶ BAŞLAT
                    </button>
                  ) : (
                    <button onClick={handleStop} disabled={loading} className="btn-neon btn-neon-red text-xs md:text-sm">
                      ■ DURDUR
                    </button>
                  )}
                  <button onClick={handleRunCycle} disabled={loading || isRunning} className="btn-neon btn-neon-purple text-xs md:text-sm hidden sm:block">
                    ▷ TEK DÖNGÜ
                  </button>
                  <button
                    onClick={() => { localStorage.removeItem('mexc_auth'); setAuthenticated(false) }}
                    className="btn-neon btn-neon-red text-xs md:text-sm ml-2"
                    title="Çıkış"
                  >
                    ◈
                  </button>
                </>
              ) : (
                <button
                  onClick={() => setShowLogin(true)}
                  className="btn-neon btn-neon-cyan text-xs md:text-sm"
                >
                  🔑 GİRİŞ YAP
                </button>
              )}
            </div>
          </div>
        </div>
      </header>

      {/* TABS */}
      <div className="border-b border-cyan-500/10 bg-[#030712]/80 backdrop-blur-sm overflow-x-auto">
        <div className="max-w-[1920px] mx-auto px-4 flex gap-1 min-w-max">
          {tabs.map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`px-3 md:px-5 py-3 text-xs md:text-sm font-bold tracking-wider transition-all border-b-2 flex items-center gap-2 whitespace-nowrap ${
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
            <span className="text-red-300 text-sm flex-1">{error}</span>
            <button onClick={() => setError(null)} className="text-red-400 hover:text-red-300 text-lg px-2">✕</button>
          </div>
        </div>
      )}

      {/* MAIN CONTENT */}
      <main className="max-w-[1920px] mx-auto p-3 md:p-4">
        {activeTab === 'dashboard' && (
          <div className="grid grid-cols-1 md:grid-cols-12 gap-3">
            <div className="md:col-span-3">
              <AgentStatus status={systemStatus} />
            </div>
            <div className="md:col-span-5">
              <PatronDecision data={latestData?.patron} />
            </div>
            <div className="md:col-span-4">
              <RiskMetrics data={latestData?.risk} portfolio={latestData?.portfolio} sentiment={sentimentData || latestData?.sentiment} />
            </div>
            <div className="md:col-span-7">
              <LivePositions positions={latestData?.positions || latestData?.executed} portfolio={latestData?.portfolio} />
            </div>
            <div className="md:col-span-5">
              <SignalPanel data={latestData?.patron} />
            </div>
          </div>
        )}
        {activeTab === 'scanner' && <MarketScanner data={latestData?.scanner} />}
        {activeTab === 'signals' && <SignalPanel data={latestData?.patron} fullPage />}
        {activeTab === 'positions' && <LivePositions positions={latestData?.positions || latestData?.executed} portfolio={latestData?.portfolio} fullPage />}
        {activeTab === 'history' && <TradeHistory trades={tradeHistory} />}
        {activeTab === 'settings' && authenticated && <SettingsPanel isRunning={isRunning} onStart={handleStart} onStop={handleStop} />}
      </main>

      {/* LOGIN MODAL */}
      {showLogin && (
        <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/80 backdrop-blur-sm p-4">
          <div className="relative">
            <button
              onClick={() => setShowLogin(false)}
              className="absolute -top-2 -right-2 w-8 h-8 rounded-full bg-gray-800 border border-gray-600 text-gray-400 hover:text-white hover:border-red-500 transition-all flex items-center justify-center z-10"
            >
              ✕
            </button>
            <KeyAuth
              onAuthenticated={() => {
                setAuthenticated(true)
                setShowLogin(false)
              }}
            />
          </div>
        </div>
      )}
    </div>
  )
}

export default App
