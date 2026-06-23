export default function ProgressBar({ percent, current, total, symbol, status, pnl, pnl_pct, onCancel, message }) {
  const statusText = {
    starting: 'Başlatılıyor...',
    fetching_data: 'Veri çekiliyor...',
    fetching: 'Veri çekiliyor...',
    processing: 'İşleniyor...',
    no_data: 'Veri yok',
    done: 'Tamamlandı',
    error: 'Hata',
  }

  const isSpinning = status === 'processing' || status === 'fetching' || status === 'fetching_data'

  return (
    <div className="bg-gray-800 border border-purple-500/30 rounded-lg p-5 mb-6">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-3">
          <div className="relative">
            <div className="w-10 h-10 rounded-full bg-purple-600/20 flex items-center justify-center">
              <span className="text-purple-400 font-bold text-sm">%{Math.round(percent)}</span>
            </div>
            {isSpinning && (
              <div className="absolute inset-0 rounded-full border-2 border-purple-500 border-t-transparent animate-spin" />
            )}
          </div>
          <div>
            <p className="text-white font-medium">
              {current > 0 ? `${current} / ${total}` : 'Hazırlanıyor'}
            </p>
            <p className="text-gray-400 text-sm">
              {message || (symbol ? `${symbol} - ${statusText[status] || status}` : 'Semboller yükleniyor...')}
            </p>
          </div>
        </div>

        <button
          onClick={onCancel}
          className="px-4 py-2 bg-red-600 hover:bg-red-700 rounded text-sm font-medium transition"
        >
          İptal
        </button>
      </div>

      <div className="w-full bg-gray-700 rounded-full h-3 overflow-hidden">
        <div
          className="h-full bg-gradient-to-r from-purple-600 to-purple-400 rounded-full transition-all duration-300 ease-out"
          style={{ width: `${percent}%` }}
        />
      </div>

      <div className="flex justify-between mt-2 text-xs text-gray-400">
        <span>%0 Veri Çekme</span>
        <span>%50 Backtest</span>
        <span>%100</span>
      </div>

      {pnl !== undefined && (
        <div className={`mt-3 text-sm font-medium ${pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
          Sonuç: {pnl >= 0 ? '+' : ''}${pnl?.toFixed(2)} ({pnl_pct?.toFixed(2)}%)
        </div>
      )}
    </div>
  )
}
