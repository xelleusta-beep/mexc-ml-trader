function formatDateTime(ts) {
  if (!ts) return ''
  const d = new Date(ts * 1000)
  return d.toLocaleString('tr-TR', {
    day: '2-digit', month: '2-digit', year: 'numeric',
    hour: '2-digit', minute: '2-digit', second: '2-digit'
  })
}

function formatDuration(entryTs) {
  if (!entryTs) return ''
  const diff = (Date.now() / 1000 - entryTs) * 1000
  const mins = Math.floor(diff / 60000)
  const hours = Math.floor(mins / 60)
  const days = Math.floor(hours / 24)
  if (days > 0) return `${days}g ${hours % 24}sa`
  if (hours > 0) return `${hours}sa ${mins % 60}dk`
  return `${mins}dk`
}

export default function LivePositions({ data, positions: positionsProp, portfolio, fullPage, onPositionClick }) {
  let positions = positionsProp || data?.positions || []

  if (positions.length > 0 && positions[0]?.position) {
    positions = positions.map(p => ({
      ...p.position,
      decision: p.decision,
    }))
  }

  const getDirStyle = (dir) => {
    if (dir === 'long') return { label: 'LONG', color: 'neon-green', border: 'border-green-500/30', bg: 'bg-green-500/5' }
    return { label: 'SHORT', color: 'neon-red', border: 'border-red-500/30', bg: 'bg-red-500/5' }
  }

  return (
    <div className={`glass-panel glass-panel-green p-4 ${fullPage ? '' : 'h-full'} corner-deco scanline`}>
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <span className="text-green-400 text-base">▣</span>
          <span className="text-label text-green-400/70">AÇIK POZİSYONLAR</span>
        </div>
        <div className={`px-3 py-1 rounded-sm border ${
          positions.length > 0 ? 'border-green-500/30 bg-green-500/5' : 'border-gray-700 bg-gray-800/50'
        }`}>
          <span className={`text-sm font-semibold tracking-wider ${
            positions.length > 0 ? 'text-green-400' : 'text-gray-500'
          }`}>
            {positions.length} AÇIK
          </span>
        </div>
      </div>

      {positions.length > 0 ? (
        <div className="space-y-2">
          {positions.map((pos, i) => {
            const dir = getDirStyle(pos.direction)
            const pnl = pos.net_pnl || pos.unrealized_pnl || 0
            const pnlPct = pos.unrealized_pnl_pct || 0
            const isProfit = pnl >= 0
            const entryTime = formatDateTime(pos.entry_time)
            const duration = formatDuration(pos.entry_time)
            const priceSource = pos.price_source === '1m_kline' ? '1M' : pos.price_source === 'live' ? 'LIVE' : ''

            return (
              <div key={i} className={`glass-panel ${dir.bg} ${dir.border} border p-3 rounded-lg transition-all hover:shadow-[0_0_15px_rgba(57,255,20,0.1)] cursor-pointer`} onClick={() => onPositionClick && onPositionClick(pos.symbol)}>
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2 flex-wrap">
                    <span className="text-base font-bold text-white">{pos.symbol}</span>
                    <span className={`tag ${dir.border} ${dir.color}`}>{dir.label}</span>
                    <span className="text-xs px-1.5 py-0.5 rounded bg-purple-500/10 text-purple-400 border border-purple-500/20">
                      x{pos.leverage || 1}
                    </span>
                    <span className="text-[10px] px-1.5 py-0.5 rounded bg-yellow-500/10 text-yellow-400 border border-yellow-500/20">
                      {pos.margin_type === 'isolated' ? 'İzole' : 'Çapraz'}
                    </span>
                    {pos.patron_score && (
                      <span className="text-xs text-gray-500">
                        Skor: %{(pos.patron_score * 100).toFixed(0)}
                      </span>
                    )}
                    {priceSource && (
                      <span className="text-[10px] px-1 py-0.5 rounded bg-cyan-500/10 text-cyan-400 border border-cyan-500/20">
                        {priceSource}
                      </span>
                    )}
                  </div>
                  <div className="text-right">
                    <p className={`text-2xl font-bold ${isProfit ? 'neon-green' : 'neon-red'}`}>
                      {isProfit ? '+' : ''}{pnl.toFixed(2)}
                    </p>
                    <p className={`text-sm ${isProfit ? 'text-green-400' : 'text-red-400'}`}>
                      {isProfit ? '+' : ''}{pnlPct.toFixed(2)}%
                    </p>
                  </div>
                </div>

                {/* Entry Time */}
                <div className="flex items-center gap-3 text-xs text-gray-400 mb-2 bg-black/20 rounded px-2 py-1">
                  <span className="text-cyan-400">GİRİŞ:</span>
                  <span className="text-white font-semibold">{entryTime}</span>
                  <span className="text-gray-600">|</span>
                  <span className="text-purple-400">{duration}</span>
                </div>

                <div className="grid grid-cols-5 gap-1 text-sm">
                  <div className="bg-black/20 rounded p-1.5 text-center">
                    <p className="text-gray-500 text-xs">GİRİŞ</p>
                    <p className="text-gray-200">${pos.entry_price?.toFixed(4)}</p>
                  </div>
                  <div className="bg-black/20 rounded p-1.5 text-center">
                    <p className="text-gray-500 text-xs">ŞU ANKİ</p>
                    <p className="text-white font-bold">${pos.current_price?.toFixed(4) || pos.entry_price?.toFixed(4)}</p>
                  </div>
                  <div className="bg-black/20 rounded p-1.5 text-center">
                    <p className="text-gray-500 text-xs">BOYUT</p>
                    <p className="text-gray-200">${pos.size_usd?.toFixed(2)}</p>
                  </div>
                  <div className="bg-black/20 rounded p-1.5 text-center">
                    <p className="text-gray-500 text-xs">TP</p>
                    <p className="text-green-400">${pos.take_profit?.toFixed(4)}</p>
                  </div>
                  <div className="bg-black/20 rounded p-1.5 text-center">
                    <p className="text-gray-500 text-xs">SL</p>
                    <p className="text-red-400">${pos.stop_loss?.toFixed(4)}</p>
                  </div>
                </div>

                <div className="flex items-center justify-between mt-2 text-[10px] bg-black/20 rounded px-2 py-1">
                  <span className="text-gray-500">
                    Fee: <span className="text-orange-400">${(pos.entry_fee || 0).toFixed(4)}</span>
                  </span>
                  <span className="text-gray-500">
                    Net PnL: <span className={`${(pos.net_pnl || 0) >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                      {(pos.net_pnl || 0) >= 0 ? '+' : ''}{(pos.net_pnl || 0).toFixed(4)}
                    </span>
                  </span>
                </div>
              </div>
            )
          })}
        </div>
      ) : (
        <div className="text-center py-10">
          <div className="text-5xl mb-3 opacity-20">▣</div>
          <p className="text-sm text-gray-500">Açık pozisyon yok</p>
        </div>
      )}
    </div>
  )
}
