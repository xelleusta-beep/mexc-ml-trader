import { useState } from 'react'
import TradeDetail from './TradeDetail'

const closeReasonColors = {
  'TP tetiklendi': 'text-green-400 bg-green-500/10 border-green-500/20',
  'SL tetiklendi': 'text-red-400 bg-red-500/10 border-red-500/20',
  'Patron yon degistirme': 'text-yellow-400 bg-yellow-500/10 border-yellow-500/20',
  'Sure doldu': 'text-gray-400 bg-gray-500/10 border-gray-500/20',
}

function formatTimestamp(ts) {
  if (!ts) return ''
  const d = new Date(ts * 1000)
  return d.toLocaleString('tr-TR', {
    day: '2-digit', month: '2-digit', year: 'numeric',
    hour: '2-digit', minute: '2-digit', second: '2-digit'
  })
}

function formatDuration(entryTs, closeTs) {
  if (!entryTs || !closeTs) return ''
  const diff = (closeTs - entryTs) * 1000
  const mins = Math.floor(diff / 60000)
  const hours = Math.floor(mins / 60)
  const days = Math.floor(hours / 24)
  if (days > 0) return `${days}g ${hours % 24}sa`
  if (hours > 0) return `${hours}sa ${mins % 60}dk`
  return `${mins}dk`
}

export default function TradeHistory({ trades: tradesProp }) {
  const trades = tradesProp || []
  const [sortBy, setSortBy] = useState('date')
  const [sortDir, setSortDir] = useState('desc')
  const [selectedTrade, setSelectedTrade] = useState(null)

  const totalPnl = trades.reduce((sum, t) => sum + (t.pnl || 0), 0)
  const winning = trades.filter(t => (t.pnl || 0) > 0)
  const losing = trades.filter(t => (t.pnl || 0) < 0)
  const breakeven = trades.filter(t => (t.pnl || 0) === 0)
  const winRate = trades.length > 0 ? (winning.length / trades.length * 100) : 0
  const avgWin = winning.length > 0 ? winning.reduce((s, t) => s + (t.pnl_pct || 0), 0) / winning.length : 0
  const avgLoss = losing.length > 0 ? losing.reduce((s, t) => s + (t.pnl_pct || 0), 0) / losing.length : 0

  const sorted = [...trades].sort((a, b) => {
    let cmp = 0
    if (sortBy === 'date') cmp = (b.close_time || 0) - (a.close_time || 0)
    else if (sortBy === 'pnl') cmp = (b.pnl || 0) - (a.pnl || 0)
    else if (sortBy === 'symbol') cmp = (a.symbol || '').localeCompare(b.symbol || '')
    return sortDir === 'desc' ? cmp : -cmp
  })

  const handleSort = (field) => {
    if (sortBy === field) setSortDir(d => d === 'desc' ? 'asc' : 'desc')
    else { setSortBy(field); setSortDir('desc') }
  }

  return (
    <div className="glass-panel glass-panel-purple p-4 corner-deco scanline">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <span className="text-purple-400 text-base">◈</span>
          <span className="text-label text-purple-400/70">İŞLEM GEÇMİŞİ</span>
        </div>
        <div className="text-sm text-gray-500">
          {trades.length} işlem
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 md:grid-cols-6 gap-2 mb-4">
        <div className="bg-black/30 rounded px-2 py-2 text-center">
          <p className="text-label text-gray-500" style={{ fontSize: '10px' }}>BAŞARI</p>
          <p className={`text-lg font-bold ${winRate >= 50 ? 'neon-green' : 'neon-red'}`}>
            %{winRate.toFixed(0)}
          </p>
        </div>
        <div className="bg-black/30 rounded px-2 py-2 text-center">
          <p className="text-label text-gray-500" style={{ fontSize: '10px' }}>TOPLAM PNL</p>
          <p className={`text-lg font-bold ${totalPnl >= 0 ? 'neon-green' : 'neon-red'}`}>
            {totalPnl >= 0 ? '+' : ''}{totalPnl.toFixed(0)}
          </p>
        </div>
        <div className="bg-black/30 rounded px-2 py-2 text-center">
          <p className="text-label text-gray-500" style={{ fontSize: '10px' }}>KAZANAN</p>
          <p className="text-lg font-bold neon-green">{winning.length}</p>
        </div>
        <div className="bg-black/30 rounded px-2 py-2 text-center">
          <p className="text-label text-gray-500" style={{ fontSize: '10px' }}>KAYBEDEN</p>
          <p className="text-lg font-bold neon-red">{losing.length}</p>
        </div>
        <div className="bg-black/30 rounded px-2 py-2 text-center">
          <p className="text-label text-gray-500" style={{ fontSize: '10px' }}>ORT. KAZANÇ</p>
          <p className="text-sm font-bold neon-green">+{avgWin.toFixed(1)}%</p>
        </div>
        <div className="bg-black/30 rounded px-2 py-2 text-center">
          <p className="text-label text-gray-500" style={{ fontSize: '10px' }}>ORT. KAYIP</p>
          <p className="text-sm font-bold neon-red">{avgLoss.toFixed(1)}%</p>
        </div>
      </div>

      {/* Sort Controls */}
      <div className="flex items-center gap-2 mb-3">
        <span className="text-xs text-gray-500">SIRALA:</span>
        {[
          { key: 'date', label: 'TARİH' },
          { key: 'pnl', label: 'PNL' },
          { key: 'symbol', label: 'SEMBOL' },
        ].map(s => (
          <button
            key={s.key}
            onClick={() => handleSort(s.key)}
            className={`text-xs px-2 py-1 rounded border transition-all ${
              sortBy === s.key
                ? 'border-cyan-500/30 bg-cyan-500/10 text-cyan-400'
                : 'border-gray-700 text-gray-500 hover:text-gray-300'
            }`}
          >
            {s.label} {sortBy === s.key ? (sortDir === 'desc' ? '▼' : '▲') : ''}
          </button>
        ))}
      </div>

      {/* Trade List */}
      {sorted.length > 0 ? (
        <div className="space-y-1.5">
          {sorted.map((trade, i) => {
            const pnl = trade.pnl || 0
            const isProfit = pnl >= 0
            const reasonClass = closeReasonColors[trade.close_reason] || 'text-gray-400 bg-gray-500/10 border-gray-500/20'
            const duration = formatDuration(trade.entry_time, trade.close_time)
            const originalIndex = trades.findIndex(t => t === trade)

            return (
              <div key={i} onClick={() => setSelectedTrade({ ...trade, _index: originalIndex })}
                className="flex items-center gap-3 p-2.5 rounded-lg bg-black/20 hover:bg-purple-500/5 transition-all border border-white/[0.02] cursor-pointer group">
                <div className={`w-1 h-12 rounded-full ${isProfit ? 'bg-green-500' : pnl < 0 ? 'bg-red-500' : 'bg-gray-500'}`} />
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 flex-wrap">
                    <span className="text-sm font-bold text-white">{trade.symbol}</span>
                    <span className={`tag text-[10px] ${
                      trade.direction === 'long' ? 'border-green-500/30 neon-green' : 'border-red-500/30 neon-red'
                    }`}>
                      {trade.direction?.toUpperCase()}
                    </span>
                    <span className="text-[10px] text-purple-400 bg-purple-500/10 px-1.5 py-0.5 rounded border border-purple-500/20">
                      x{trade.leverage}
                    </span>
                    {trade.close_reason && (
                      <span className={`text-[10px] px-1.5 py-0.5 rounded border ${reasonClass}`}>
                        {trade.close_reason}
                      </span>
                    )}
                    {duration && (
                      <span className="text-[10px] text-gray-600">{duration}</span>
                    )}
                  </div>
                  <div className="flex items-center gap-3 text-xs text-gray-400 mt-1">
                    <span>Giriş: ${trade.entry_price?.toFixed(4)}</span>
                    <span className="text-gray-600">→</span>
                    <span>Çıkış: ${trade.exit_price?.toFixed(4)}</span>
                    <span className="text-gray-600">${trade.size_usd?.toFixed(0)}</span>
                  </div>
                  <div className="flex items-center gap-4 text-[10px] text-gray-500 mt-1">
                    <span className="text-cyan-400/60">Açılış: {formatTimestamp(trade.entry_time)}</span>
                    <span className="text-purple-400/60">Kapanış: {formatTimestamp(trade.close_time)}</span>
                    {duration && <span className="text-gray-600">{duration}</span>}
                  </div>
                </div>
                <div className="text-right">
                  <p className={`text-lg font-bold ${isProfit ? 'neon-green' : pnl < 0 ? 'neon-red' : 'text-gray-400'}`}>
                    {isProfit ? '+' : ''}{pnl.toFixed(2)}
                  </p>
                  <p className={`text-sm ${isProfit ? 'text-green-400' : pnl < 0 ? 'text-red-400' : 'text-gray-500'}`}>
                    {isProfit ? '+' : ''}{trade.pnl_pct?.toFixed(1)}%
                  </p>
                </div>
                <span className="text-gray-600 text-xs group-hover:text-cyan-400 transition-colors">◇</span>
              </div>
            )
          })}
        </div>
      ) : (
        <div className="text-center py-16">
          <div className="text-6xl mb-4 opacity-10">◈</div>
          <p className="text-base text-gray-500 font-semibold">Henüz kapanan işlem yok</p>
          <p className="text-sm text-gray-600 mt-2">SL/TP tetiklendiğinde işlemler burada görünecek</p>
        </div>
      )}

      {selectedTrade && (
        <TradeDetail trade={selectedTrade} onClose={() => setSelectedTrade(null)} />
      )}
    </div>
  )
}
