export default function OptimizationResults({ results }) {
  if (!results) return null;

  const ResultCard = ({ title, data, color, icon }) => {
    if (!data) return null;
    return (
      <div className={`bg-gradient-to-r from-${color}-900/50 to-${color}-800/50 border border-${color}-500/30 rounded-lg p-5`}>
        <h3 className={`text-lg font-bold text-${color}-400 mb-2 flex items-center gap-2`}>
          <span>{icon}</span> {title}
        </h3>
        <p className={`text-3xl font-bold text-${color}-400 mb-1`}>
          {data.avg_pnl_pct > 0 ? '+' : ''}{data.avg_pnl_pct.toFixed(2)}%
        </p>
        <p className="text-sm text-gray-300 mb-2">{data.label}</p>
        <div className="flex flex-wrap gap-3 text-xs text-gray-400">
          <span className="bg-gray-700/50 px-2 py-1 rounded">
            <span className="text-yellow-400 font-bold">{data.timeframe}</span> zaman dilimi
          </span>
          <span className="bg-gray-700/50 px-2 py-1 rounded">
            Win Rate: <span className="text-blue-400 font-bold">%{data.avg_win_rate}</span>
          </span>
          <span className="bg-gray-700/50 px-2 py-1 rounded">
            Max DD: <span className="text-purple-400 font-bold">%{data.avg_drawdown}</span>
          </span>
          <span className="bg-gray-700/50 px-2 py-1 rounded">
            <span className="text-green-400 font-bold">{data.total_trades}</span> işlem
          </span>
          <span className="bg-gray-700/50 px-2 py-1 rounded">
            <span className="text-cyan-400 font-bold">{data.coin_count}</span> coin
          </span>
        </div>
      </div>
    );
  };

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold text-white mb-4">
        Optimizasyon Sonuçları
        <span className="text-sm font-normal text-gray-400 ml-3">
          {results.total_tests} test tamamlandı
        </span>
      </h2>

      {/* Ana Ödüller */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <ResultCard
          title="En Yüksek Getiri"
          data={results.best_by_pnl}
          color="green"
          icon="1"
        />
        <ResultCard
          title="En İyi Risk/Getiri"
          data={results.best_by_risk}
          color="blue"
          icon="2"
        />
      </div>

      {/* Zaman Dilimlerine Göre En İyiler */}
      {results.best_per_timeframe && Object.keys(results.best_per_timeframe).length > 0 && (
        <div className="bg-gray-800 border border-gray-700 rounded-lg p-5">
          <h3 className="text-lg font-bold text-white mb-4">Her Zaman Dilimi İçin En İyi Sonuç</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {Object.entries(results.best_per_timeframe).map(([tf, data]) => (
              <div key={tf} className="bg-gray-700/50 rounded-lg p-4 border border-gray-600">
                <p className="text-2xl font-bold text-purple-400 mb-1">{tf}</p>
                <p className={`text-xl font-bold ${data.avg_pnl_pct >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                  {data.avg_pnl_pct > 0 ? '+' : ''}{data.avg_pnl_pct.toFixed(2)}%
                </p>
                <p className="text-xs text-gray-400 mt-1 truncate">{data.label}</p>
                <div className="flex justify-between text-xs mt-2">
                  <span className="text-blue-400">WR: %{data.avg_win_rate}</span>
                  <span className="text-purple-400">DD: %{data.avg_drawdown}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Risk/Getiri En İyileri */}
      {results.risk_adjusted_top10 && results.risk_adjusted_top10.length > 0 && (
        <div className="bg-gray-800 border border-gray-700 rounded-lg p-5">
          <h3 className="text-lg font-bold text-white mb-4">Risk/Getiri Skoru En Yüksek 10</h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-gray-400 border-b border-gray-700">
                  <th className="text-left p-2">#</th>
                  <th className="text-left p-2">Dilim</th>
                  <th className="text-left p-2">Strateji</th>
                  <th className="text-right p-2">PnL%</th>
                  <th className="text-right p-2">Win Rate</th>
                  <th className="text-right p-2">Drawdown</th>
                  <th className="text-right p-2">İşlem</th>
                </tr>
              </thead>
              <tbody>
                {results.risk_adjusted_top10.map((item, idx) => (
                  <tr key={idx} className="border-b border-gray-700/50 hover:bg-gray-700/30">
                    <td className="p-2 text-yellow-400 font-bold">{idx + 1}</td>
                    <td className="p-2">
                      <span className="bg-purple-900/50 text-purple-300 px-2 py-1 rounded text-xs font-bold">
                        {item.timeframe}
                      </span>
                    </td>
                    <td className="p-2 text-gray-300 text-xs">{item.label}</td>
                    <td className={`p-2 text-right font-bold ${item.avg_pnl_pct >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                      {item.avg_pnl_pct > 0 ? '+' : ''}{item.avg_pnl_pct.toFixed(2)}%
                    </td>
                    <td className="p-2 text-right text-blue-400">%{item.avg_win_rate}</td>
                    <td className="p-2 text-right text-purple-400">%{item.avg_drawdown}</td>
                    <td className="p-2 text-right text-cyan-400">{item.total_trades}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Tüm Sonuçlar */}
      {results.all_results && results.all_results.length > 0 && (
        <div className="bg-gray-800 border border-gray-700 rounded-lg p-5">
          <h3 className="text-lg font-bold text-white mb-4">
            Tüm Sonuçlar ({results.all_results.length})
          </h3>
          <div className="overflow-x-auto max-h-96 overflow-y-auto">
            <table className="w-full text-sm">
              <thead className="sticky top-0 bg-gray-800">
                <tr className="text-gray-400 border-b border-gray-700">
                  <th className="text-left p-2">#</th>
                  <th className="text-left p-2">Dilim</th>
                  <th className="text-left p-2">Strateji</th>
                  <th className="text-right p-2">PnL%</th>
                  <th className="text-right p-2">Win Rate</th>
                  <th className="text-right p-2">Drawdown</th>
                  <th className="text-right p-2">İşlem</th>
                  <th className="text-right p-2">Coin</th>
                </tr>
              </thead>
              <tbody>
                {results.all_results.map((item, idx) => (
                  <tr key={idx} className="border-b border-gray-700/50 hover:bg-gray-700/30">
                    <td className="p-2 text-gray-500">{idx + 1}</td>
                    <td className="p-2">
                      <span className={`px-2 py-1 rounded text-xs font-bold ${
                        idx === 0 ? 'bg-yellow-900/50 text-yellow-300' :
                        idx < 3 ? 'bg-green-900/50 text-green-300' :
                        'bg-gray-700/50 text-gray-400'
                      }`}>
                        {item.timeframe}
                      </span>
                    </td>
                    <td className="p-2 text-gray-300 text-xs max-w-48 truncate">{item.label}</td>
                    <td className={`p-2 text-right font-bold ${item.avg_pnl_pct >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                      {item.avg_pnl_pct > 0 ? '+' : ''}{item.avg_pnl_pct.toFixed(2)}%
                    </td>
                    <td className="p-2 text-right text-blue-400">%{item.avg_win_rate}</td>
                    <td className="p-2 text-right text-purple-400">%{item.avg_drawdown}</td>
                    <td className="p-2 text-right text-cyan-400">{item.total_trades}</td>
                    <td className="p-2 text-right text-gray-400">{item.coin_count}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
