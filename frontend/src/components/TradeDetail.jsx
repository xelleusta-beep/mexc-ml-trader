export default function TradeDetail({ data, onClose }) {
  if (!data) return null

  const formatPrice = (p) => {
    if (p === undefined || p === null) return '-'
    return p < 0.01 ? p.toFixed(6) : p < 1 ? p.toFixed(4) : p.toFixed(2)
  }

  const formatDate = (ts) => {
    if (!ts) return '-'
    return new Date(ts).toLocaleDateString('tr-TR', { year: 'numeric', month: 'short', day: 'numeric' })
  }

  const closedTrades = data.closed_trades || []

  return (
    <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50 p-4" onClick={onClose}>
      <div className="bg-gray-800 rounded-xl max-w-4xl w-full max-h-[90vh] overflow-y-auto p-6" onClick={e => e.stopPropagation()}>
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-2xl font-bold text-purple-400">{data.symbol} Detay</h2>
          <button onClick={onClose} className="text-gray-400 hover:text-white text-2xl">X</button>
        </div>

        {/* Genel Bilgiler */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
          <div className="bg-gray-700/50 rounded-lg p-4">
            <p className="text-xs text-gray-400">Başlangıç Sermayesi</p>
            <p className="text-lg font-bold text-white">${data.initial_capital?.toFixed(0)}</p>
          </div>
          <div className="bg-gray-700/50 rounded-lg p-4">
            <p className="text-xs text-gray-400">Son Sermaye</p>
            <p className="text-lg font-bold text-white">${data.final_capital?.toFixed(2)}</p>
          </div>
          <div className="bg-gray-700/50 rounded-lg p-4">
            <p className="text-xs text-gray-400">Toplam Kâr/Zarar</p>
            <p className={`text-lg font-bold ${data.total_pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
              {data.total_pnl >= 0 ? '+' : ''}${data.total_pnl?.toFixed(2)} ({data.total_pnl_pct?.toFixed(2)}%)
            </p>
          </div>
          <div className="bg-gray-700/50 rounded-lg p-4">
            <p className="text-xs text-gray-400">Max Drawdown</p>
            <p className="text-lg font-bold text-red-400">%{data.max_drawdown?.toFixed(2)}</p>
          </div>
        </div>

        <div className="grid grid-cols-3 gap-4 mb-6">
          <div className="bg-gray-700/50 rounded-lg p-4">
            <p className="text-xs text-gray-400">Toplam İşlem</p>
            <p className="text-xl font-bold text-blue-400">{data.total_trades}</p>
          </div>
          <div className="bg-gray-700/50 rounded-lg p-4">
            <p className="text-xs text-gray-400">Kazanan / Kaybeden</p>
            <p className="text-xl font-bold">
              <span className="text-green-400">{data.winning_trades}</span>
              <span className="text-gray-500"> / </span>
              <span className="text-red-400">{data.losing_trades}</span>
            </p>
          </div>
          <div className="bg-gray-700/50 rounded-lg p-4">
            <p className="text-xs text-gray-400">Win Rate</p>
            <p className="text-xl font-bold text-purple-400">%{data.win_rate?.toFixed(1)}</p>
          </div>
        </div>

        {/* İşlem Detayları */}
        <h3 className="text-lg font-semibold text-gray-300 mb-4">İşlem Detayları</h3>
        {closedTrades.length === 0 ? (
          <p className="text-gray-500 text-center py-4">İşlem bulunamadı</p>
        ) : (
          <div className="space-y-4">
            {closedTrades.map((trade, idx) => {
              const entryPrice = trade.first_entry_price
              const avgPrice = trade.avg_price
              const maxGain = trade.max_gain_pct || 0
              const maxLoss = trade.max_loss_pct || 0

              return (
                <div key={idx} className="bg-gray-700/30 border border-gray-600 rounded-lg p-4">
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-3">
                      <span className="text-sm font-bold text-purple-400">İşlem #{idx + 1}</span>
                      <span className={`text-xs px-2 py-1 rounded ${trade.pnl >= 0 ? 'bg-green-900/50 text-green-400' : 'bg-red-900/50 text-red-400'}`}>
                        {trade.pnl >= 0 ? 'Kâr' : 'Zarar'}
                      </span>
                    </div>
                    <span className={`font-bold ${trade.pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                      {trade.pnl >= 0 ? '+' : ''}${trade.pnl?.toFixed(2)}
                    </span>
                  </div>

                  <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm mb-3">
                    <div>
                      <p className="text-xs text-gray-400">Giriş Tarihi</p>
                      <p className="text-white">{trade.entry_date_str}</p>
                    </div>
                    <div>
                      <p className="text-xs text-gray-400">Çıkış Tarihi</p>
                      <p className="text-white">{trade.close_date_str}</p>
                    </div>
                    <div>
                      <p className="text-xs text-gray-400">Süre</p>
                      <p className="text-white">{trade.duration_days} gün</p>
                    </div>
                    <div>
                      <p className="text-xs text-gray-400">Kapanış Sebebi</p>
                      <p className="text-yellow-400">{trade.close_reason}</p>
                    </div>
                  </div>

                  <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm mb-3">
                    <div>
                      <p className="text-xs text-gray-400">İlk Giriş Fiyatı</p>
                      <p className="text-white font-mono">${formatPrice(entryPrice)}</p>
                    </div>
                    <div>
                      <p className="text-xs text-gray-400">Ortalama Giriş</p>
                      <p className="text-white font-mono">${formatPrice(avgPrice)}</p>
                    </div>
                    <div>
                      <p className="text-xs text-gray-400">Maks Yükseliş</p>
                      <p className="text-green-400 font-mono">+{maxGain.toFixed(2)}%</p>
                    </div>
                    <div>
                      <p className="text-xs text-gray-400">Maks Düşüş</p>
                      <p className="text-red-400 font-mono">{maxLoss.toFixed(2)}%</p>
                    </div>
                  </div>

                  {/* İşlem Listesi */}
                  {trade.trades && trade.trades.length > 0 && (
                    <div className="mt-3">
                      <p className="text-xs text-gray-400 mb-2">İşlem Geçmişi:</p>
                      <div className="space-y-1">
                        {trade.trades.map((t, ti) => (
                          <div key={ti} className="flex items-center justify-between text-xs bg-gray-800/50 rounded px-3 py-1.5">
                            <div className="flex items-center gap-2">
                              <span className={t.type === 'buy' ? 'text-green-400' : 'text-red-400'}>
                                {t.type === 'buy' ? 'ALIŞ' : 'SATIŞ'}
                              </span>
                              <span className="text-gray-400">{t.reason}</span>
                            </div>
                            <div className="flex items-center gap-4">
                              <span className="text-gray-400">{t.date_str}</span>
                              <span className="text-white">${formatPrice(t.price)}</span>
                              <span className="text-gray-400">{t.amount_usd?.toFixed(2)}$</span>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        )}
      </div>
    </div>
  )
}
