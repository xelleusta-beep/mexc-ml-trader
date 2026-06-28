function FearGreedGauge({ value, label }) {
  const v = value || 0
  const color = v <= 25 ? '#ff0040' : v <= 45 ? '#ff6b35' : v <= 55 ? '#ffd700' : v <= 75 ? '#00ff88' : '#39ff14'
  const rotation = -90 + (v / 100) * 180

  return (
    <div className="relative w-full flex flex-col items-center">
      <svg viewBox="0 0 200 110" className="w-full max-w-[180px]">
        <defs>
          <linearGradient id="gaugeGrad" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#ff0040" />
            <stop offset="25%" stopColor="#ff6b35" />
            <stop offset="50%" stopColor="#ffd700" />
            <stop offset="75%" stopColor="#00ff88" />
            <stop offset="100%" stopColor="#39ff14" />
          </linearGradient>
        </defs>
        <path d="M 20 100 A 80 80 0 0 1 180 100" fill="none" stroke="rgba(255,255,255,0.05)" strokeWidth="12" strokeLinecap="round" />
        <path d="M 20 100 A 80 80 0 0 1 180 100" fill="none" stroke="url(#gaugeGrad)" strokeWidth="12" strokeLinecap="round" strokeDasharray={`${v * 2.51} 251`} />
        <circle cx={100 + 80 * Math.cos((rotation * Math.PI) / 180)} cy={100 + 80 * Math.sin((rotation * Math.PI) / 180)} r="5" fill={color} className="drop-shadow-lg" />
        <text x="100" y="85" textAnchor="middle" fill={color} fontSize="28" fontFamily="Orbitron" fontWeight="bold">{v}</text>
        <text x="100" y="105" textAnchor="middle" fill="rgba(255,255,255,0.4)" fontSize="10" fontFamily="Rajdhani">{label || 'FEAR & GREED'}</text>
      </svg>
    </div>
  )
}

export default function RiskMetrics({ data, portfolio, sentiment }) {
  const risk = data?.risk_metrics || data || {}
  const dailyUsedPct = risk.daily_limit_pct ? Math.min((Math.abs(risk.daily_pnl || 0) / (risk.daily_limit_pct * (risk.total_equity || 10000))) * 100, 100) : 0
  const exposurePct = Math.min((risk.exposure_pct || 0) * 100, 100)

  const sent = sentiment || {}
  const moodAnalysis = sent.mood_analysis || {}
  const newsStats = sent.news_stats || {}

  const canTrade = risk.can_trade !== undefined ? risk.can_trade : true

  return (
    <div className="glass-panel glass-panel-red p-4 h-full corner-deco">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <span className="text-red-400 text-base">⬡</span>
          <span className="text-label text-red-400/70">RİSK & PİYASA</span>
        </div>
        <div className={`flex items-center gap-2 px-3 py-1 rounded-sm border ${
          canTrade ? 'border-green-500/30 bg-green-500/5' : 'border-red-500/30 bg-red-500/5'
        }`}>
          <div className={`w-2 h-2 rounded-full ${canTrade ? 'bg-green-400' : 'bg-red-400 animate-pulse-neon'}`} />
          <span className={`text-sm font-semibold ${canTrade ? 'text-green-400' : 'text-red-400'}`}>
            {canTrade ? 'İŞLEM SERBEST' : 'İŞLEM DURDURULDU'}
          </span>
        </div>
      </div>

      {/* Fear & Greed Gauge */}
      <div className="mb-3 bg-black/20 rounded-lg p-3 border border-purple-500/10">
        <div className="flex items-center justify-between mb-1">
          <span className="text-[10px] text-purple-400/60 tracking-wider">PİYASA DUYGUSU</span>
          <span className="text-xs font-bold text-purple-400">
            {sent.fear_greed_label || 'N/A'}
          </span>
        </div>
        <FearGreedGauge value={sent.fear_greed_index} label={sent.market_mood?.toUpperCase() || 'DUYGU'} />
        {moodAnalysis.description && (
          <p className="text-xs text-gray-400 mt-2 text-center">{moodAnalysis.description}</p>
        )}
        {moodAnalysis.recommendation && (
          <p className="text-[10px] text-cyan-400/60 text-center mt-1">→ {moodAnalysis.recommendation}</p>
        )}
        {newsStats.total > 0 && (
          <div className="flex items-center justify-center gap-3 mt-2 text-[10px]">
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
          <span className={`font-bold ${dailyUsedPct > 80 ? 'text-red-400' : dailyUsedPct > 50 ? 'text-yellow-400' : 'text-gray-300'}`}>
            ${Math.abs(risk.daily_pnl || 0).toFixed(0)} / ${((risk.daily_limit_pct || 0.05) * (risk.total_equity || 10000)).toFixed(0)}
          </span>
        </div>
        <div className="progress-bar">
          <div className={`progress-bar-fill transition-all duration-500 ${dailyUsedPct > 80 ? 'bg-gradient-to-r from-red-600 to-red-400' : dailyUsedPct > 50 ? 'bg-gradient-to-r from-yellow-600 to-yellow-400' : 'bg-gradient-to-r from-green-600 to-green-400'}`} style={{ width: `${dailyUsedPct}%` }} />
        </div>
      </div>

      {/* Exposure Bar */}
      <div className="mb-3">
        <div className="flex justify-between text-sm mb-1">
          <span className="text-gray-400 font-semibold">MARJİN KULLANIMI</span>
          <span className={`font-bold ${exposurePct > 80 ? 'text-red-400' : exposurePct > 50 ? 'text-yellow-400' : 'text-cyan-400'}`}>
            %{exposurePct.toFixed(1)}
          </span>
        </div>
        <div className="progress-bar">
          <div className={`progress-bar-fill transition-all duration-500 ${exposurePct > 80 ? 'bg-gradient-to-r from-red-600 to-red-400' : exposurePct > 50 ? 'bg-gradient-to-r from-yellow-600 to-yellow-400' : 'bg-gradient-to-r from-cyan-600 to-cyan-400'}`} style={{ width: `${exposurePct}%` }} />
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
            { text: 'İşlem başına max risk: %2', active: dailyUsedPct < 100 },
            { text: 'Maksimum kaldıraç: 20x', active: true },
            { text: 'Günlük kayıp limiti: %5', active: dailyUsedPct < 80 },
            { text: 'Maksimum pozisyon: 5', active: (risk.current_positions || 0) < 5 },
          ].map((rule, i) => (
            <div key={i} className="text-xs flex items-center gap-2">
              <span className={rule.active ? 'text-green-500/60' : 'text-red-500/60'}>{rule.active ? '▸' : '▸'}</span>
              <span className={rule.active ? 'text-gray-400' : 'text-red-400/60'}>{rule.text}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
