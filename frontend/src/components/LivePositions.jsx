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
    if (dir === 'long') return { label: 'LONG', color: 'neon-green', border: 'border-green-500/30', bg: 'bg-green-500/5', gradient: 'from-green-500/10 to-emerald-500/5', glow: 'rgba(57,255,20,0.06)' }
    return { label: 'SHORT', color: 'neon-red', border: 'border-red-500/30', bg: 'bg-red-500/5', gradient: 'from-red-500/10 to-orange-500/5', glow: 'rgba(255,0,64,0.06)' }
  }

  return (
    <div className={`glass-panel glass-panel-green p-4 ${fullPage ? '' : 'h-full'} corner-deco scanline animate-fadeIn`}>
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <div className="w-2.5 h-2.5 rounded-full bg-green-400 animate-pulse-soft" />
          <span className="text-label text-green-400/70">AÇIK POZİSYONLAR</span>
        </div>
        <div className={`px-3 py-1.5 rounded-lg border ${
          positions.length > 0 ? 'border-green-500/30 bg-green-500/5 animate-glowPulse' : 'border-gray-700 bg-gray-800/50'
        }`}>
          <span className={`text-xs font-bold tracking-wider ${
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
              <div
                key={i}
                className={`glass-panel bg-gradient-to-r ${dir.gradient} ${dir.border} border p-3.5 rounded-xl cursor-pointer card-hover`}
                style={{ boxShadow: `0 0 15px ${dir.glow}` }}
                onClick={() => onPositionClick && onPositionClick(pos.symbol)}
              >
                <div className="flex items-center justify-between mb-2.5">
                  <div className="flex items-center gap-2 flex-wrap">
                    <span className="text-base font-bold text-white tracking-wide">{pos.symbol}</span>
                    <span className={`tag ${dir.border} ${dir.color}`}>{dir.label}</span>
                    <span className="text-[10px] px-1.5 py-0.5 rounded-md bg-purple-500/10 text-purple-400 border border-purple-500/20 font-mono">
                      x{pos.leverage || 1}
                    </span>
                    <span className="text-[10px] px-1.5 py-0.5 rounded-md bg-yellow-500/10 text-yellow-400 border border-yellow-500/20">
                      {pos.margin_type === 'isolated' ? 'İzole' : 'Çapraz'}
                    </span>
                    {pos.patron_score && (
                      <span className="text-[10px] text-gray-500 font-mono">
                        %{(pos.patron_score * 100).toFixed(0)}
                      </span>
                    )}
                    {priceSource && (
                      <span className="text-[10px] px-1.5 py-0.5 rounded-md bg-cyan-500/10 text-cyan-400 border border-cyan-500/20 font-mono">
                        {priceSource}
                      </span>
                    )}
                  </div>
                  <div className="text-right">
                    <p className={`text-2xl font-bold ${isProfit ? 'neon-green' : 'neon-red'} font-mono`}>
                      {isProfit ? '+' : ''}{pnl.toFixed(2)}
                    </p>
                    <p className={`text-xs ${isProfit ? 'text-green-400' : 'text-red-400'} font-mono`}>
                      {isProfit ? '+' : ''}{pnlPct.toFixed(2)}%
                    </p>
                  </div>
                </div>

                <div className="flex items-center gap-3 text-xs text-gray-400 mb-2.5 bg-black/20 rounded-lg px-2.5 py-1.5">
                  <span className="text-cyan-400 font-semibold">GİRİŞ:</span>
                  <span className="text-white font-semibold">{entryTime}</span>
                  <span className="text-gray-600">|</span>
                  <span className="text-purple-400 font-mono">{duration}</span>
                </div>

                <div className="grid grid-cols-5 gap-1.5 text-sm">
                  {[
                    { label: 'GİRİŞ', value: `$${pos.entry_price?.toFixed(4)}`, color: 'text-gray-200' },
                    { label: 'ŞU ANKİ', value: `$${pos.current_price?.toFixed(4) || pos.entry_price?.toFixed(4)}`, color: 'text-white font-bold' },
                    { label: 'BOYUT', value: `$${pos.size_usd?.toFixed(2)}`, color: 'text-gray-200' },
                    { label: 'TP', value: `$${pos.take_profit?.toFixed(4)}`, color: 'text-green-400' },
                    { label: 'SL', value: `$${pos.stop_loss?.toFixed(4)}`, color: 'text-red-400' },
                  ].map((item, j) => (
                    <div key={j} className="bg-black/20 rounded-lg p-2 text-center border border-white/[0.02]">
                      <p className="text-gray-500 text-[10px] tracking-wider">{item.label}</p>
                      <p className={`text-xs ${item.color} font-mono`}>{item.value}</p>
                    </div>
                  ))}
                </div>

                <div className="flex items-center justify-between mt-2 text-[10px] bg-black/15 rounded-lg px-2.5 py-1.5">
                  <span className="text-gray-500">
                    Fee: <span className="text-orange-400 font-mono">${(pos.entry_fee || 0).toFixed(4)}</span>
                  </span>
                  <span className="text-gray-500">
                    Net PnL: <span className={`${(pos.net_pnl || 0) >= 0 ? 'text-green-400' : 'text-red-400'} font-mono`}>
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
          <div className="text-5xl mb-3 opacity-15 animate-float">▣</div>
          <p className="text-sm text-gray-500">Açık pozisyon yok</p>
          <p className="text-[10px] text-gray-600 mt-1">Sistem fırsatları tarıyor</p>
        </div>
      )}
    </div>
  )
}
