import { useState } from 'react'

const SORT_OPTIONS = [
  { value: 'pnl_desc', label: 'Kâr/Zarar (Yüksek→Düşük)' },
  { value: 'pnl_asc', label: 'Kâr/Zarar (Düşük→Yüksek)' },
  { value: 'pnl_pct_desc', label: 'Getiri % (Yüksek→Düşük)' },
  { value: 'win_rate_desc', label: 'Win Rate (Yüksek→Düşük)' },
  { value: 'trades_desc', label: 'İşlem Sayısı (Çok→Az)' },
  { value: 'trades_asc', label: 'İşlem Sayısı (Az→Çok)' },
  { value: 'dd_asc', label: 'Max DD (Düşük→Yüksek)' },
  { value: 'dd_desc', label: 'Max DD (Yüksek→Düşük)' },
]

export default function ResultsTable({ results, onSelect }) {
  const [sortBy, setSortBy] = useState('pnl_desc')

  if (!results || !results.results) return null

  const sortedResults = [...results.results].sort((a, b) => {
    switch (sortBy) {
      case 'pnl_desc': return b.total_pnl - a.total_pnl
      case 'pnl_asc': return a.total_pnl - b.total_pnl
      case 'pnl_pct_desc': return b.total_pnl_pct - a.total_pnl_pct
      case 'win_rate_desc': return b.win_rate - a.win_rate
      case 'trades_desc': return b.total_trades - a.total_trades
      case 'trades_asc': return a.total_trades - b.total_trades
      case 'dd_asc': return a.max_drawdown - b.max_drawdown
      case 'dd_desc': return b.max_drawdown - a.max_drawdown
      default: return b.total_pnl - a.total_pnl
    }
  })

  return (
    <div className="bg-gray-800 rounded-lg p-6">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-semibold text-purple-300">Backtest Sonuçları</h2>
        <div className="flex items-center gap-2">
          <label className="text-xs text-gray-400">Sırala:</label>
          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value)}
            className="bg-gray-700 rounded px-3 py-1.5 text-sm text-white border border-gray-600 focus:border-purple-500 focus:outline-none"
          >
            {SORT_OPTIONS.map(opt => (
              <option key={opt.value} value={opt.value}>{opt.label}</option>
            ))}
          </select>
        </div>
      </div>

      {results.summary && (
        <div className="grid grid-cols-3 gap-4 mb-6">
          <div className="bg-gray-700 rounded p-4">
            <p className="text-xs text-gray-400">Toplam Kâr/Zarar</p>
            <p className={`text-2xl font-bold ${results.summary.total_pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
              ${results.summary.total_pnl?.toFixed(2)}
            </p>
          </div>
          <div className="bg-gray-700 rounded p-4">
            <p className="text-xs text-gray-400">Toplam Getiri</p>
            <p className={`text-2xl font-bold ${results.summary.total_pnl_pct >= 0 ? 'text-green-400' : 'text-red-400'}`}>
              %{results.summary.total_pnl_pct?.toFixed(2)}
            </p>
          </div>
          <div className="bg-gray-700 rounded p-4">
            <p className="text-xs text-gray-400">Başarılı/Başarısız</p>
            <p className="text-2xl font-bold text-blue-400">
              {results.summary.successful}/{results.summary.total_symbols}
            </p>
          </div>
        </div>
      )}

      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-gray-700">
              <th className="text-left py-3 px-2 text-gray-400">#</th>
              <th className="text-left py-3 px-2 text-gray-400">Coin</th>
              <th className="text-right py-3 px-2 text-gray-400">Son Bakiye</th>
              <th className="text-right py-3 px-2 text-gray-400">Kâr/Zarar</th>
              <th className="text-right py-3 px-2 text-gray-400">Getiri</th>
              <th className="text-right py-3 px-2 text-gray-400">Win Rate</th>
              <th className="text-right py-3 px-2 text-gray-400">Max DD</th>
              <th className="text-center py-3 px-2 text-gray-400">İşlem</th>
            </tr>
          </thead>
          <tbody>
            {sortedResults.map((r, idx) => (
              <tr
                key={r.symbol}
                className="border-b border-gray-700/50 hover:bg-gray-700/50 cursor-pointer"
                onClick={() => onSelect(r)}
              >
                <td className="py-3 px-2 text-gray-500">{idx + 1}</td>
                <td className="py-3 px-2 font-medium">{r.symbol}</td>
                <td className="py-3 px-2 text-right text-gray-300">${r.final_capital?.toFixed(2)}</td>
                <td className={`py-3 px-2 text-right font-medium ${r.total_pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                  {r.total_pnl >= 0 ? '+' : ''}${r.total_pnl?.toFixed(2)}
                </td>
                <td className={`py-3 px-2 text-right font-medium ${r.total_pnl_pct >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                  %{r.total_pnl_pct?.toFixed(2)}
                </td>
                <td className="py-3 px-2 text-right text-gray-300">
                  %{r.win_rate?.toFixed(1)}
                </td>
                <td className="py-3 px-2 text-right text-red-400">
                  %{r.max_drawdown?.toFixed(2)}
                </td>
                <td className="py-3 px-2 text-center">
                  <span className="text-xs bg-gray-700 rounded px-2 py-1">
                    {r.total_trades}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {results.errors && results.errors.length > 0 && (
        <div className="mt-4 p-3 bg-yellow-900/30 border border-yellow-700 rounded">
          <p className="text-sm text-yellow-400">
            {results.errors.length} coin işlenirken hata oluştu
          </p>
        </div>
      )}
    </div>
  )
}
