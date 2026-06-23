export default function RiskMetrics({ data, portfolio, sentiment }) {
  const risk = data?.risk_metrics || data || {}
  const dailyUsedPct = risk.daily_limit_pct ? Math.min((Math.abs(risk.daily_pnl || 0) / (risk.daily_limit_pct * (risk.total_equity || 10000))) * 100, 100) : 0
  const exposurePct = Math.min((risk.exposure_pct || 0) * 100, 100)

  const sent = sentiment || {}
  const moodAnalysis = sent.mood_analysis || {}
  const newsStats = sent.news_stats || {}

  return (
    <div className="glass-panel glass-panel-red p-4 h-full corner-deco">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <span className="text-red-400 text-base">⬡</span>
          <span className="text-label text-red-400/70">RİSK & PİYASA</span>
        </div>
        <div className={`flex items-center gap-2 px-3 py-1 rounded-sm border ${
          risk.can_trade ? 'border-green-500/30 bg-green-500/5' : 'border-red-500/30 bg-red-500/5'
        }`}>
          <div className={`w-2 h-2 rounded-full ${risk.can_trade ? 'bg-green-400' : 'bg-red-400 animate-pulse-neon'}`} />
          <span className={`text-sm font-semibold ${risk.can_trade ? 'text-green-400' : 'text-red-400'}`}>
            {risk.can_trade ? 'İŞLEM SERBEST' : 'İŞLEM YOK'}
          </span>
        </div>
      </div>

      {/* Sentiment Section */}
      <div className="mb-3 bg-black/20 rounded-lg p-2.5 border border-purple-500/10">
        <div className="flex items-center justify-between mb-1.5">
          <span className="text-[10px] text-purple-400/60 tracking-wider">PİYASA DUYGUSU</span>
          <span className="text-xs font-bold text-purple-400">
            F&G: {sent.fear_greed_index || '?'} ({sent.fear_greed_label || '?'})
          </span>
        </div>
        {moodAnalysis.description && (
          <p className="text-xs text-gray-400 mb-1">{moodAnalysis.description}</p>
        )}
        {moodAnalysis.recommendation && (
          <p className="text-[10px] text-cyan-400/60">→ {moodAnalysis.recommendation}</p>
        )}
        {newsStats.total > 0 && (
          <div className="flex items-center gap-3 mt-1.5 text-[10px]">
            <span className="text-green-400">+{newsStats.positive} olumlu</span>
            <span className="text-red-400">-{newsStats.negative} olumsuz</span>
            <span className="text-gray-500">{newsStats.neutral} notr</span>
          </div>
        )}
      </div>

      {/* Daily Loss Bar */}
      <div className="mb-3">
        <div className="flex justify-between text-sm mb-1">
          <span className="text-gray-400 font-semibold">GÜNLÜK KAYIP</span>
          <span className="text-gray-300">${risk.daily_pnl?.toFixed(0) || 0}</span>
        </div>
        <div className="progress-bar">
          <div className={`progress-bar-fill ${dailyUsedPct > 80 ? 'bg-gradient-to-r from-red-600 to-red-400' : dailyUsedPct > 50 ? 'bg-gradient-to-r from-yellow-600 to-yellow-400' : 'bg-gradient-to-r from-green-600 to-green-400'}`} style={{ width: `${dailyUsedPct}%` }} />
        </div>
      </div>

      {/* Exposure Bar */}
      <div className="mb-3">
        <div className="flex justify-between text-sm mb-1">
          <span className="text-gray-400 font-semibold">MARJİN</span>
          <span className="text-gray-300">%{exposurePct.toFixed(1)}</span>
        </div>
        <div className="progress-bar">
          <div className={`progress-bar-fill ${exposurePct > 80 ? 'bg-gradient-to-r from-red-600 to-red-400' : exposurePct > 50 ? 'bg-gradient-to-r from-yellow-600 to-yellow-400' : 'bg-gradient-to-r from-cyan-600 to-cyan-400'}`} style={{ width: `${exposurePct}%` }} />
        </div>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-2 gap-2 mb-3">
        {[
          { label: 'AÇIK POZİSYON', value: `${risk.current_positions || 0}/${risk.max_positions || 5}`, color: 'text-cyan-400' },
          { label: 'BOŞ SLOT', value: risk.available_slots || 0, color: 'text-green-400' },
          { label: 'TOPLAM MARJİN', value: `$${(risk.total_exposure_usd || 0).toFixed(0)}`, color: 'text-yellow-400' },
          { label: 'GÜNLÜK LİMİT', value: `$${((risk.daily_limit_pct || 0.05) * (risk.total_equity || 10000)).toFixed(0)}`, color: 'text-orange-400' },
        ].map((item, i) => (
          <div key={i} className="bg-black/30 rounded px-2 py-2">
            <p className="text-label text-gray-500" style={{ fontSize: '10px' }}>{item.label}</p>
            <p className={`text-base font-bold ${item.color}`}>{item.value}</p>
          </div>
        ))}
      </div>

      {/* Rules */}
      <div className="bg-black/20 rounded-lg p-2.5 border border-cyan-500/5">
        <div className="flex items-center gap-2 mb-1.5">
          <span className="text-yellow-400 text-xs">⚠</span>
          <span className="text-label text-yellow-400/70" style={{ fontSize: '10px' }}>RİSK KURALLARI</span>
        </div>
        <div className="space-y-0.5">
          {[
            'İşlem başına max risk: %2',
            'Maksimum kaldıraç: 20x',
            'Günlük kayıp limiti: %5',
            'Maksimum pozisyon: 5',
          ].map((rule, i) => (
            <div key={i} className="text-xs text-gray-500 flex items-center gap-2">
              <span className="text-cyan-500/40">▸</span> {rule}
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
