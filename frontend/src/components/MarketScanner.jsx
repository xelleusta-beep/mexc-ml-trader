export default function MarketScanner({ data }) {
  const topSymbols = data?.hot_pairs || []

  const getHeatColor = (score) => {
    if (score >= 80) return 'neon-green'
    if (score >= 60) return 'text-yellow-400'
    if (score >= 40) return 'text-orange-400'
    return 'neon-red'
  }

  const getBarWidth = (score) => Math.min(Math.max(score, 5), 100)

  return (
    <div className="glass-panel glass-panel-cyan p-4 h-full corner-deco scanline">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <span className="text-cyan-400 text-base">◎</span>
          <span className="text-label text-cyan-400/70">PİYASA TARAYICI</span>
        </div>
        <div className="text-sm text-gray-500">
          {data?.symbol_count || 0} sembol tarandı
        </div>
      </div>

      {topSymbols.length > 0 ? (
        <div className="space-y-1.5">
          {topSymbols.slice(0, 15).map((sym, i) => {
            const score = Math.round(Math.min(Math.max(sym.hot_score || 0, 0), 100))

            return (
              <div key={i} className="flex items-center gap-3 p-2 rounded bg-black/20 hover:bg-cyan-500/5 transition-colors">
                <span className="text-sm text-gray-500 w-5 text-right">#{i+1}</span>
                <div className="flex-1">
                  <div className="flex items-center justify-between mb-1">
                    <div className="flex items-center gap-3">
                      <span className="text-base font-bold text-white">{sym.symbol}</span>
                      {sym.baseCoin && (
                        <span className="text-xs text-gray-500">{sym.baseCoin}</span>
                      )}
                    </div>
                    <div className="flex items-center gap-3 text-sm">
                      {sym.last_price > 0 && (
                        <span className="text-gray-400">${sym.last_price?.toLocaleString(undefined, {maximumFractionDigits: sym.last_price < 1 ? 6 : 2})}</span>
                      )}
                      {sym.change_24h !== undefined && (
                        <span className={sym.change_24h >= 0 ? 'neon-green' : 'neon-red'}>
                          {sym.change_24h >= 0 ? '+' : ''}{sym.change_24h?.toFixed(2)}%
                        </span>
                      )}
                      <span className={`font-bold ${getHeatColor(score)}`}>
                        {score}
                      </span>
                    </div>
                  </div>
                  <div className="w-full h-1.5 bg-black/40 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-gradient-to-r from-cyan-600 via-purple-500 to-pink-500 rounded-full transition-all duration-700"
                      style={{ width: `${getBarWidth(score)}%` }}
                    />
                  </div>
                </div>
              </div>
            )
          })}
        </div>
      ) : (
        <div className="text-center py-10">
          <div className="text-5xl mb-3 opacity-20">◎</div>
          <p className="text-sm text-gray-500">Henüz tarama yapılmadı</p>
        </div>
      )}

      <div className="mt-4 pt-3 border-t border-cyan-500/10 grid grid-cols-3 gap-2">
        <div className="text-center">
          <p className="text-label text-gray-500" style={{ fontSize: '10px' }}>SEMBOL</p>
          <p className="text-base font-bold neon-cyan">{data?.symbol_count || 0}</p>
        </div>
        <div className="text-center">
          <p className="text-label text-gray-500" style={{ fontSize: '10px' }}>HOT</p>
          <p className="text-base font-bold neon-green">{topSymbols.length}</p>
        </div>
        <div className="text-center">
          <p className="text-label text-gray-500" style={{ fontSize: '10px' }}>DÖNGÜ</p>
          <p className="text-base font-bold neon-purple">#</p>
        </div>
      </div>
    </div>
  )
}
