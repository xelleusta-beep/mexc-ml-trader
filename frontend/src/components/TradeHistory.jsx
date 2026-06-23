export default function TradeHistory({ data }) {
  const trades = data?.trade_history || data?.history || []
  const stats = data?.stats || {}

  const totalPnl = trades.reduce((sum, t) => sum + (t.pnl || 0), 0)
  const winning = trades.filter(t => (t.pnl || 0) > 0)
  const losing = trades.filter(t => (t.pnl || 0) <= 0)
  const winRate = trades.length > 0 ? (winning.length / trades.length * 100) : 0

  return (
    <div className="glass-panel glass-panel-purple p-4 h-full corner-deco scanline">
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
      <div className="grid grid-cols-4 gap-2 mb-4">
        <div className="bg-black/30 rounded px-2 py-2 text-center">
          <p className="text-label text-gray-500" style={{ fontSize: '10px' }}>BAŞARI</p>
          <p className={`text-base font-bold ${winRate >= 50 ? 'neon-green' : 'neon-red'}`}>
            %{winRate.toFixed(0)}
          </p>
        </div>
        <div className="bg-black/30 rounded px-2 py-2 text-center">
          <p className="text-label text-gray-500" style={{ fontSize: '10px' }}>TOPLAM PNL</p>
          <p className={`text-base font-bold ${totalPnl >= 0 ? 'neon-green' : 'neon-red'}`}>
            {totalPnl >= 0 ? '+' : ''}{totalPnl.toFixed(2)}
          </p>
        </div>
        <div className="bg-black/30 rounded px-2 py-2 text-center">
          <p className="text-label text-gray-500" style={{ fontSize: '10px' }}>KAZANAN</p>
          <p className="text-base font-bold neon-green">{winning.length}</p>
        </div>
        <div className="bg-black/30 rounded px-2 py-2 text-center">
          <p className="text-label text-gray-500" style={{ fontSize: '10px' }}>KAYBEDEN</p>
          <p className="text-base font-bold neon-red">{losing.length}</p>
        </div>
      </div>

      {/* Trade List */}
      {trades.length > 0 ? (
        <div className="space-y-1.5">
          {trades.slice(0, 20).map((trade, i) => {
            const pnl = trade.pnl || 0
            const isProfit = pnl >= 0

            return (
              <div key={i} className="flex items-center gap-3 p-2 rounded bg-black/20 hover:bg-purple-500/5 transition-colors">
                <div className={`w-1 h-10 rounded-full ${isProfit ? 'bg-green-500' : 'bg-red-500'}`} />
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="text-base font-bold text-white">{trade.symbol}</span>
                    <span className={`tag ${
                      trade.direction === 'long' ? 'border-green-500/30 neon-green' : 'border-red-500/30 neon-red'
                    }`}>
                      {trade.direction?.toUpperCase()}
                    </span>
                    <span className="text-xs text-gray-500">x{trade.leverage}</span>
                    {trade.close_reason && (
                      <span className="text-[10px] px-1.5 py-0.5 rounded bg-black/30 text-gray-400 border border-white/5">
                        {trade.close_reason}
                      </span>
                    )}
                  </div>
                  <div className="flex items-center gap-3 text-sm text-gray-400 mt-0.5">
                    <span>Giriş: ${trade.entry_price?.toFixed(4)}</span>
                    <span>→</span>
                    <span>Çıkış: ${trade.exit_price?.toFixed(4)}</span>
                    <span>{trade.size_usd?.toFixed(0)}</span>
                  </div>
                </div>
                <div className="text-right">
                  <p className={`text-lg font-bold ${isProfit ? 'neon-green' : 'neon-red'}`}>
                    {isProfit ? '+' : ''}{pnl.toFixed(2)}
                  </p>
                  <p className={`text-sm ${isProfit ? 'text-green-400' : 'text-red-400'}`}>
                    {isProfit ? '+' : ''}{trade.pnl_pct?.toFixed(1)}%
                  </p>
                </div>
              </div>
            )
          })}
        </div>
      ) : (
        <div className="text-center py-10">
          <div className="text-5xl mb-3 opacity-20">◈</div>
          <p className="text-sm text-gray-500">Henüz kapanan işlem yok</p>
          <p className="text-xs text-gray-600 mt-1">SL/TP tetiklendiğinde işlemler burada görünecek</p>
        </div>
      )}
    </div>
  )
}
