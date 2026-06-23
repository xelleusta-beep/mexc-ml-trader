export default function LivePositions({ data, positions: positionsProp, portfolio, fullPage }) {
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
            const pnl = pos.unrealized_pnl || 0
            const pnlPct = pos.unrealized_pnl_pct || 0
            const isProfit = pnl >= 0

            return (
              <div key={i} className={`glass-panel ${dir.bg} ${dir.border} border p-3 rounded-lg transition-all hover:shadow-[0_0_15px_rgba(57,255,20,0.1)]`}>
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <span className="text-base font-bold text-white">{pos.symbol}</span>
                    <span className={`tag ${dir.border} ${dir.color}`}>{dir.label}</span>
                    <span className="text-xs px-1.5 py-0.5 rounded bg-purple-500/10 text-purple-400 border border-purple-500/20">
                      x{pos.leverage || 1}
                    </span>
                    {pos.patron_score && (
                      <span className="text-xs text-gray-500">
                        Skor: %{(pos.patron_score * 100).toFixed(0)}
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

                <div className="grid grid-cols-5 gap-1 text-sm">
                  <div className="bg-black/20 rounded p-1.5 text-center">
                    <p className="text-gray-500 text-xs">GİRİŞ</p>
                    <p className="text-gray-200">${pos.entry_price?.toFixed(4)}</p>
                  </div>
                  <div className="bg-black/20 rounded p-1.5 text-center">
                    <p className="text-gray-500 text-xs">ŞU ANKİ</p>
                    <p className="text-white">${pos.current_price?.toFixed(4) || pos.entry_price?.toFixed(4)}</p>
                  </div>
                  <div className="bg-black/20 rounded p-1.5 text-center">
                    <p className="text-gray-500 text-xs">BOYUT</p>
                    <p className="text-gray-200">${pos.size_usd?.toFixed(0)}</p>
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
