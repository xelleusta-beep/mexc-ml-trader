import { useState } from 'react'
import TradeDetail from './TradeDetail'

const closeReasonColors = {
  'TP tetiklendi': { color: '#39ff14', bg: '#39ff1408' },
  'SL tetiklendi': { color: '#ff3366', bg: '#ff336608' },
  'Patron yon degistirme': { color: '#ffd700', bg: '#ffd70008' },
  'Sure doldu': { color: '#8b95a5', bg: '#8b95a508' },
}

function formatTimestamp(ts) {
  if (!ts) return ''
  const d = new Date(ts * 1000)
  return d.toLocaleString('tr-TR', { day: '2-digit', month: '2-digit', year: 'numeric', hour: '2-digit', minute: '2-digit' })
}

function formatDuration(entryTs, closeTs) {
  if (!entryTs || !closeTs) return ''
  const diff = (closeTs - entryTs) * 1000
  const mins = Math.floor(diff / 60000)
  const hours = Math.floor(mins / 60)
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
  const winRate = trades.length > 0 ? (winning.length / trades.length * 100) : 0

  const sorted = [...trades].sort((a, b) => {
    let cmp = 0
    if (sortBy === 'date') cmp = (b.close_time || 0) - (a.close_time || 0)
    else if (sortBy === 'pnl') cmp = (b.pnl || 0) - (a.pnl || 0)
    else if (sortBy === 'symbol') cmp = (a.symbol || '').localeCompare(b.symbol || '')
    return sortDir === 'desc' ? cmp : -cmp
  })

  return (
    <div className="cinematic-border rounded-2xl p-5" style={{ background: 'linear-gradient(145deg, rgba(6,10,22,0.95) 0%, rgba(14,10,22,0.9) 100%)' }}>
      <div className="flex items-center justify-between mb-5">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl flex items-center justify-center" style={{ background: '#b026ff08', border: '1px solid #b026ff15' }}>
            <span style={{ color: '#b026ff', fontSize: '13px' }}>◈</span>
          </div>
          <span className="text-[11px] font-bold tracking-[0.15em] text-gray-400 font-mono">İŞLEM GEÇMİŞİ</span>
        </div>
        <div className="text-[11px] text-gray-600 font-mono">{trades.length} işlem</div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-6 gap-2 mb-5">
        {[
          { label: 'BAŞARI', value: `%${winRate.toFixed(0)}`, color: winRate >= 50 ? '#39ff14' : '#ff3366' },
          { label: 'TOPLAM PNL', value: `${totalPnl >= 0 ? '+' : ''}${totalPnl.toFixed(0)}`, color: totalPnl >= 0 ? '#39ff14' : '#ff3366' },
          { label: 'KAZANAN', value: winning.length, color: '#39ff14' },
          { label: 'KAYBEDEN', value: losing.length, color: '#ff3366' },
          { label: 'ORT. KAZANÇ', value: `+${(winning.length > 0 ? winning.reduce((s, t) => s + (t.pnl_pct || 0), 0) / winning.length : 0).toFixed(1)}%`, color: '#39ff14' },
          { label: 'ORT. KAYIP', value: `${(losing.length > 0 ? losing.reduce((s, t) => s + (t.pnl_pct || 0), 0) / losing.length : 0).toFixed(1)}%`, color: '#ff3366' },
        ].map((item, i) => (
          <div key={i} className="rounded-lg px-2 py-2.5 text-center border border-white/[0.03]" style={{ background: `${item.color}03` }}>
            <p className="text-[9px] text-gray-600 tracking-wider font-mono mb-0.5">{item.label}</p>
            <p className="text-[14px] font-bold font-mono" style={{ color: item.color }}>{item.value}</p>
          </div>
        ))}
      </div>

      <div className="flex items-center gap-2 mb-3">
        <span className="text-[10px] text-gray-600 font-mono">SIRALA:</span>
        {[{ key: 'date', label: 'TARİH' }, { key: 'pnl', label: 'PNL' }, { key: 'symbol', label: 'SEMBOL' }].map(s => (
          <button key={s.key} onClick={() => { if (sortBy === s.key) setSortDir(d => d === 'desc' ? 'asc' : 'desc'); else { setSortBy(s.key); setSortDir('desc') } }}
            className="text-[10px] px-2 py-1 rounded-lg border transition-all font-mono" style={{
              borderColor: sortBy === s.key ? '#00f0ff20' : '#ffffff06',
              background: sortBy === s.key ? '#00f0ff08' : 'transparent',
              color: sortBy === s.key ? '#00f0ff' : '#8b95a5',
            }}>
            {s.label} {sortBy === s.key ? (sortDir === 'desc' ? '▼' : '▲') : ''}
          </button>
        ))}
      </div>

      {sorted.length > 0 ? (
        <div className="space-y-1.5">
          {sorted.map((trade, i) => {
            const pnl = trade.pnl || 0
            const isProfit = pnl >= 0
            const reasonStyle = closeReasonColors[trade.close_reason] || { color: '#8b95a5', bg: '#8b95a508' }
            const duration = formatDuration(trade.entry_time, trade.close_time)
            const originalIndex = trades.findIndex(t => t === trade)
            return (
              <div key={i} onClick={() => setSelectedTrade({ ...trade, _index: originalIndex })}
                className="flex items-center gap-3 p-3 rounded-xl border border-white/[0.03] cursor-pointer transition-all hover:border-white/[0.06] group" style={{ background: `${isProfit ? '#39ff14' : '#ff3366'}03` }}>
                <div className="w-1 h-10 rounded-full" style={{ background: isProfit ? '#39ff14' : '#ff3366' }} />
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 flex-wrap">
                    <span className="text-[13px] font-bold text-white font-mono">{trade.symbol}</span>
                    <span className="text-[9px] px-2 py-0.5 rounded-md font-bold tracking-wider font-mono border" style={{
                      color: trade.direction === 'long' ? '#39ff14' : '#ff3366',
                      borderColor: trade.direction === 'long' ? '#39ff1430' : '#ff336630',
                      background: trade.direction === 'long' ? '#39ff1410' : '#ff336610',
                    }}>{trade.direction?.toUpperCase()}</span>
                    <span className="text-[9px] px-1.5 py-0.5 rounded-md font-mono" style={{ color: '#b026ff', border: '1px solid #b026ff20', background: '#b026ff08' }}>x{trade.leverage}</span>
                    {trade.close_reason && <span className="text-[9px] px-1.5 py-0.5 rounded-md font-mono" style={{ color: reasonStyle.color, background: reasonStyle.bg }}>{trade.close_reason}</span>}
                    {duration && <span className="text-[9px] text-gray-700 font-mono">{duration}</span>}
                  </div>
                  <div className="flex items-center gap-3 text-[10px] text-gray-600 mt-1 font-mono">
                    <span>${trade.entry_price?.toFixed(4)}</span>
                    <span className="text-gray-700">→</span>
                    <span>${trade.exit_price?.toFixed(4)}</span>
                    <span className="text-gray-700">${trade.size_usd?.toFixed(0)}</span>
                  </div>
                </div>
                <div className="text-right">
                  <p className="text-[16px] font-bold font-mono" style={{ color: isProfit ? '#39ff14' : '#ff3366' }}>{isProfit ? '+' : ''}{pnl.toFixed(2)}</p>
                  <p className="text-[10px] font-mono" style={{ color: isProfit ? '#39ff1480' : '#ff336680' }}>{isProfit ? '+' : ''}{trade.pnl_pct?.toFixed(1)}%</p>
                </div>
              </div>
            )
          })}
        </div>
      ) : (
        <div className="text-center py-16">
          <div className="text-5xl mb-4 opacity-10">◈</div>
          <p className="text-[14px] text-gray-600">Henüz kapanan işlem yok</p>
          <p className="text-[11px] text-gray-700 mt-1 font-mono">SL/TP tetiklendiğinde işlemler burada görünecek</p>
        </div>
      )}

      {selectedTrade && <TradeDetail trade={selectedTrade} onClose={() => setSelectedTrade(null)} />}
    </div>
  )
}
