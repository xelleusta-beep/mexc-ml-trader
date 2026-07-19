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
import PositionChart from './components/PositionChart'
import WelcomeScreen from './components/WelcomeScreen'
import DeepTrader from './components/DeepTrader'
import { getSystemStatus, getTradingLatest, getTradingHistory, getSentimentCurrent, startTrading, stopTrading, runSingleCycle, connectWebSocket } from './api/trading'

function App() {
  const [showWelcome, setShowWelcome] = useState(true)
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
  const [selectedPosition, setSelectedPosition] = useState(null)
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
    { id: 'deep-trader', label: 'DEEP LEARNING', icon: '🧠', auth: false },
    { id: 'history', label: 'GEÇMİŞ', icon: '▣', auth: false },
    { id: 'settings', label: 'AYARLAR', icon: '⚙', auth: true },
  ].filter(tab => !tab.auth || authenticated)

  const totalEquity = latestData?.portfolio?.total_equity || systemStatus?.total_equity || 10000
  const usedCapital = latestData?.portfolio?.total_exposure_usd || (totalEquity - (latestData?.portfolio?.available_capital || systemStatus?.available_capital || totalEquity))
  const positionCount = latestData?.positions?.length || latestData?.executed?.length || systemStatus?.open_positions || 0

  return (
    <>
      {showWelcome && <WelcomeScreen onEnter={() => setShowWelcome(false)} />}
      <div className="min-h-screen bg-[#030712] grid-bg hex-pattern relative">
      {/* PARTICLE EFFECTS */}
      <div className="particles">
        {[...Array(12)].map((_, i) => (
          <div key={i} className="particle" style={{
            left: `${Math.random() * 100}%`,
            animationDuration: `${15 + Math.random() * 20}s`,
            animationDelay: `${Math.random() * 10}s`,
          }} />
        ))}
      </div>

      {/* HEADER */}
      <header className="border-b border-white/[0.04] sticky top-0 z-50" style={{ background: 'rgba(3,7,18,0.92)', backdropFilter: 'blur(20px)' }}>
        <div className="max-w-[1920px] mx-auto px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-4 md:gap-6">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl flex items-center justify-center" style={{ background: 'linear-gradient(135deg, #00f0ff12, #b026ff08)', border: '1px solid #00f0ff20', boxShadow: '0 0 20px #00f0ff08' }}>
                <span className="font-orbitron text-base" style={{ color: '#00f0ff' }}>M</span>
              </div>
              <div>
                <h1 className="font-orbitron text-sm md:text-base font-bold tracking-[0.15em]">
                  <span style={{ color: '#00f0ff' }}>MEXC</span>
                  <span className="text-gray-700 mx-1 hidden sm:inline">|</span>
                  <span style={{ color: '#b026ff' }}>NEXUS</span>
                </h1>
                <p className="text-[8px] text-gray-600 tracking-[0.2em] font-mono hidden md:block">MULTI-AGENT NEURAL NETWORK</p>
              </div>
            </div>

            <div className="flex items-center gap-3 ml-2 md:ml-4 pl-4 border-l border-white/[0.05]">
              <div className="flex items-center gap-2">
                <div className="relative">
                  <div className={`w-2 h-2 rounded-full ${wsConnected ? 'bg-green-400' : 'bg-red-400'}`} />
                  {wsConnected && <div className="absolute inset-0 w-2 h-2 rounded-full bg-green-400 animate-ping opacity-40" />}
                </div>
                <span className="text-[10px] md:text-[11px] tracking-wider font-bold font-mono" style={{ color: wsConnected ? '#39ff14' : '#ff3366' }}>
                  {wsConnected ? 'ONLINE' : 'OFFLINE'}
                </span>
              </div>
              <div className="text-[10px] md:text-[11px] text-gray-600 font-mono hidden sm:block">{formatTime(time)}</div>
            </div>
          </div>

          <div className="flex items-center gap-3 md:gap-4">
            <div className="flex items-center gap-4 md:gap-5 px-4 py-2 rounded-xl border border-white/[0.04]" style={{ background: 'rgba(0,240,255,0.02)' }}>
              <div>
                <p className="text-[9px] text-gray-600 tracking-wider font-mono mb-0.5">PORTFÖY</p>
                <p className={`text-[16px] md:text-[18px] font-bold font-mono transition-all ${equityFlash === 'up' ? 'flash-green' : equityFlash === 'down' ? 'flash-red' : ''}`} style={{ color: '#00f0ff' }}>
                  ${totalEquity.toLocaleString()}
                </p>
              </div>
              <div className="w-px h-7 bg-white/[0.05] hidden sm:block" />
              <div className="hidden sm:block">
                <p className="text-[9px] text-gray-600 tracking-wider font-mono mb-0.5">KULLANILAN</p>
                <p className="text-[16px] md:text-[18px] font-bold font-mono" style={{ color: '#39ff14' }}>${usedCapital.toLocaleString()}</p>
              </div>
              <div className="w-px h-7 bg-white/[0.05] hidden sm:block" />
              <div className="hidden sm:block">
                <p className="text-[9px] text-gray-600 tracking-wider font-mono mb-0.5">POZİSYON</p>
                <p className="text-[16px] md:text-[18px] font-bold font-mono" style={{ color: '#b026ff' }}>{positionCount}/5</p>
              </div>
            </div>

            <div className="flex items-center gap-2">
              {authenticated ? (
                <>
                  <button onClick={isRunning ? handleStop : handleStart} disabled={loading}
                    className="px-4 py-2 rounded-xl text-[10px] font-bold tracking-wider font-mono transition-all duration-300"
                    style={{ background: isRunning ? '#ff336608' : '#39ff1408', border: `1px solid ${isRunning ? '#ff336620' : '#39ff1420'}`, color: isRunning ? '#ff3366' : '#39ff14' }}>
                    {isRunning ? '■ DURDUR' : '▶ BAŞLAT'}
                  </button>
                  <button onClick={handleRunCycle} disabled={loading || isRunning}
                    className="px-4 py-2 rounded-xl text-[10px] font-bold tracking-wider font-mono transition-all duration-300 hidden sm:block"
                    style={{ background: '#b026ff08', border: '1px solid #b026ff20', color: '#b026ff' }}>
                    ▷ TEK DÖNGÜ
                  </button>
                  <button onClick={() => { localStorage.removeItem('mexc_auth'); setAuthenticated(false) }}
                    className="w-8 h-8 rounded-lg flex items-center justify-center transition-all duration-300"
                    style={{ background: '#ff336606', border: '1px solid #ff336615', color: '#ff3366' }}>
                    <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" /></svg>
                  </button>
                </>
              ) : (
                <button onClick={() => setShowLogin(true)} className="px-4 py-2 rounded-xl text-[10px] font-bold tracking-wider font-mono transition-all duration-300"
                  style={{ background: '#00f0ff08', border: '1px solid #00f0ff20', color: '#00f0ff' }}>
                  🔑 GİRİŞ
                </button>
              )}
            </div>
          </div>
        </div>
      </header>

      {/* TABS */}
      <div className="border-b border-white/[0.04] overflow-x-auto" style={{ background: 'rgba(3,7,18,0.85)', backdropFilter: 'blur(10px)' }}>
        <div className="max-w-[1920px] mx-auto px-4 flex gap-1 min-w-max">
          {tabs.map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className="px-4 md:px-5 py-3 text-[10px] md:text-[11px] font-bold tracking-[0.12em] font-mono transition-all duration-300 border-b-2 flex items-center gap-2 whitespace-nowrap"
              style={{
                borderColor: activeTab === tab.id ? '#00f0ff40' : 'transparent',
                color: activeTab === tab.id ? '#00f0ff' : '#8b95a560',
                background: activeTab === tab.id ? '#00f0ff04' : 'transparent',
              }}
            >
              <span className="text-[12px]">{tab.icon}</span>
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
              <LivePositions positions={latestData?.positions || latestData?.executed} portfolio={latestData?.portfolio} onPositionClick={setSelectedPosition} />
            </div>
            <div className="md:col-span-5">
              <SignalPanel data={latestData?.patron} />
            </div>
          </div>
        )}
        {activeTab === 'scanner' && <MarketScanner data={latestData?.scanner} />}
        {activeTab === 'signals' && <SignalPanel data={latestData?.patron} fullPage />}
        {activeTab === 'positions' && <LivePositions positions={latestData?.positions || latestData?.executed} portfolio={latestData?.portfolio} fullPage onPositionClick={setSelectedPosition} />}
        {activeTab === 'deep-trader' && <DeepTrader />}
        {activeTab === 'history' && <TradeHistory trades={tradeHistory} />}
        {activeTab === 'settings' && authenticated && <SettingsPanel isRunning={isRunning} onStart={handleStart} onStop={handleStop} />}
      </main>

      {/* POSITION CHART MODAL */}
      {selectedPosition && (
        <PositionChart symbol={selectedPosition} onClose={() => setSelectedPosition(null)} />
      )}

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
    </>
  )
}

export default App
