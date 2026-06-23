import { useState } from 'react'

export default function PortfolioResults({ results }) {
  const [expandedTrade, setExpandedTrade] = useState(null)

  if (!results) return null;

  return (
    <div className="space-y-6">
      <div className="bg-gray-800 border border-purple-500/30 rounded-lg p-4">
        <h2 className="text-2xl font-bold text-white mb-2">Portföy Simülasyonu ($20K / 2 Pozisyon)</h2>
        <div className="flex items-center gap-4 text-sm">
          <span className="text-gray-400">İlk Giriş: <span className="text-white font-medium">{results.first_date_str}</span></span>
          <span className="text-gray-400">→</span>
          <span className="text-gray-400">Son Çıkış: <span className="text-white font-medium">{results.last_date_str}</span></span>
          <span className="text-yellow-400 font-bold">({results.total_days} gün)</span>
        </div>
      </div>

      {/* Özet Kartları */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-gray-800 border border-gray-700 rounded-lg p-4">
          <p className="text-gray-400 text-sm">Başlangıç</p>
          <p className="text-xl font-bold text-white">${results.initial_capital.toLocaleString()}</p>
        </div>
        <div className="bg-gray-800 border border-gray-700 rounded-lg p-4">
          <p className="text-gray-400 text-sm">Son Sermaye</p>
          <p className={`text-xl font-bold ${results.total_pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
            ${results.final_capital.toLocaleString(undefined, {maximumFractionDigits: 2})}
          </p>
        </div>
        <div className="bg-gray-800 border border-gray-700 rounded-lg p-4">
          <p className="text-gray-400 text-sm">Toplam K/Z</p>
          <p className={`text-xl font-bold ${results.total_pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
            {results.total_pnl >= 0 ? '+' : ''}${results.total_pnl.toFixed(2)} ({results.total_pnl_pct.toFixed(2)}%)
          </p>
        </div>
        <div className="bg-gray-800 border border-gray-700 rounded-lg p-4">
          <p className="text-gray-400 text-sm">Win Rate</p>
          <p className="text-xl font-bold text-blue-400">%{results.win_rate.toFixed(1)}</p>
        </div>
      </div>

      {/* Detay İstatistikler */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-gray-800 border border-gray-700 rounded-lg p-4">
          <p className="text-gray-400 text-sm">Toplam İşlem</p>
          <p className="text-xl font-bold text-white">{results.total_trades}</p>
        </div>
        <div className="bg-gray-800 border border-gray-700 rounded-lg p-4">
          <p className="text-gray-400 text-sm">Kazanan</p>
          <p className="text-xl font-bold text-green-400">{results.winning_trades}</p>
        </div>
        <div className="bg-gray-800 border border-gray-700 rounded-lg p-4">
          <p className="text-gray-400 text-sm">Kaybeden</p>
          <p className="text-xl font-bold text-red-400">{results.losing_trades}</p>
        </div>
        <div className="bg-gray-800 border border-gray-700 rounded-lg p-4">
          <p className="text-gray-400 text-sm">Ort. K/Z</p>
          <p className={`text-xl font-bold ${results.avg_pnl_per_trade >= 0 ? 'text-green-400' : 'text-red-400'}`}>
            ${results.avg_pnl_per_trade.toFixed(2)}
          </p>
        </div>
      </div>

      {/* Tüm İşlemler */}
      <div className="bg-gray-800 border border-gray-700 rounded-lg p-4">
        <h3 className="text-lg font-bold text-white mb-4">İşlem Detayları</h3>
        <div className="space-y-3">
          {results.trades.map((trade) => (
            <div key={trade.id} className="bg-gray-700/50 border border-gray-600 rounded-lg overflow-hidden">
              {/* İşlem Başlığı */}
              <div
                className="flex items-center justify-between p-4 cursor-pointer hover:bg-gray-700/80 transition"
                onClick={() => setExpandedTrade(expandedTrade === trade.id ? null : trade.id)}
              >
                <div className="flex items-center gap-4">
                  <span className="text-gray-400 font-mono text-sm">#{trade.id}</span>
                  <span className="text-white font-bold">{trade.symbol}</span>
                  <span className={`px-2 py-0.5 rounded text-xs font-medium ${
                    trade.pnl >= 0 ? 'bg-green-900/50 text-green-400' : 'bg-red-900/50 text-red-400'
                  }`}>
                    {trade.pnl >= 0 ? 'KAR' : 'ZARAR'}
                  </span>
                </div>
                <div className="flex items-center gap-6 text-sm">
                  <span className="text-gray-400">{trade.entry_date_str} → {trade.close_date_str}</span>
                  <span className="text-yellow-400 font-medium">{trade.duration_days} gün</span>
                  <span className={`font-bold ${trade.pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                    {trade.pnl >= 0 ? '+' : ''}${trade.pnl.toFixed(2)}
                  </span>
                </div>
              </div>

              {/* Genişletilmiş Detay */}
              {expandedTrade === trade.id && (
                <div className="border-t border-gray-600 p-4 bg-gray-800/50">
                  {/* Tarih ve Süre */}
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
                    <div className="bg-gray-700/50 rounded p-3">
                      <p className="text-gray-400 text-xs mb-1">Giriş Tarihi</p>
                      <p className="text-white text-sm font-medium">{trade.entry_date_str}</p>
                    </div>
                    <div className="bg-gray-700/50 rounded p-3">
                      <p className="text-gray-400 text-xs mb-1">Çıkış Tarihi</p>
                      <p className="text-white text-sm font-medium">{trade.close_date_str}</p>
                    </div>
                    <div className="bg-gray-700/50 rounded p-3">
                      <p className="text-gray-400 text-xs mb-1">Süre</p>
                      <p className="text-yellow-400 text-sm font-bold">{trade.duration_days} gün</p>
                    </div>
                    <div className="bg-gray-700/50 rounded p-3">
                      <p className="text-gray-400 text-xs mb-1">Kapanış Sebebi</p>
                      <p className="text-purple-400 text-sm font-medium">{trade.close_reason}</p>
                    </div>
                  </div>

                  {/* Fiyat Detayları */}
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
                    <div className="bg-gray-700/50 rounded p-3">
                      <p className="text-gray-400 text-xs mb-1">İlk Giriş Fiyatı</p>
                      <p className="text-white text-sm font-mono">${trade.first_entry_price?.toFixed(6) || '0'}</p>
                    </div>
                    <div className="bg-gray-700/50 rounded p-3">
                      <p className="text-gray-400 text-xs mb-1">Ortalama Giriş</p>
                      <p className="text-white text-sm font-mono">${trade.avg_price?.toFixed(6) || '0'}</p>
                    </div>
                    <div className="bg-gray-700/50 rounded p-3">
                      <p className="text-gray-400 text-xs mb-1">Maksimum Fiyat</p>
                      <p className="text-green-400 text-sm font-mono">${trade.max_price?.toFixed(6) || '0'}</p>
                    </div>
                    <div className="bg-gray-700/50 rounded p-3">
                      <p className="text-gray-400 text-xs mb-1">Minimum Fiyat</p>
                      <p className="text-red-400 text-sm font-mono">${trade.min_price?.toFixed(6) || '0'}</p>
                    </div>
                  </div>

                  {/* Yükseliş/Düşüş */}
                  <div className="grid grid-cols-2 gap-4 mb-4">
                    <div className="bg-green-900/30 border border-green-500/30 rounded p-3">
                      <p className="text-green-400 text-xs mb-1">Maksimum Yükseliş</p>
                      <p className="text-green-400 text-lg font-bold">+{trade.max_gain_pct?.toFixed(2) || '0'}%</p>
                    </div>
                    <div className="bg-red-900/30 border border-red-500/30 rounded p-3">
                      <p className="text-red-400 text-xs mb-1">Maksimum Düşüş</p>
                      <p className="text-red-400 text-lg font-bold">{trade.max_loss_pct?.toFixed(2) || '0'}%</p>
                    </div>
                  </div>

                  {/* İşlem Geçmişi */}
                  <div className="mt-4">
                    <p className="text-gray-400 text-xs mb-2">İşlem Geçmişi ({trade.total_buys} ALIŞ / {trade.total_sells} SATIŞ)</p>
                    <div className="space-y-1">
                      {trade.trade_history.map((t, idx) => (
                        <div key={idx} className="flex items-center justify-between text-sm py-2 px-3 bg-gray-700/30 rounded">
                          <div className="flex items-center gap-3">
                            <span className={`px-2 py-0.5 rounded text-xs font-bold ${
                              t.type === 'buy' ? 'bg-blue-900/50 text-blue-400' : 'bg-red-900/50 text-red-400'
                            }`}>
                              {t.type === 'buy' ? 'ALIŞ' : 'SATIŞ'}
                            </span>
                            <span className="text-gray-400 text-xs">{t.reason}</span>
                          </div>
                          <div className="flex items-center gap-4">
                            <span className="text-gray-400 text-xs">{t.date_str}</span>
                            <span className="text-white font-mono">${t.price?.toFixed(6) || '0'}</span>
                            <span className="text-yellow-400 font-medium">${t.amount_usd?.toFixed(2) || '0'}</span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
