function FearGreedGauge({ value, label }) {
  const v = value || 0
  const color = v <= 25 ? '#ff3366' : v <= 45 ? '#ff6b35' : v <= 55 ? '#ffd700' : v <= 75 ? '#39ff14' : '#00ff88'
  const rotation = -90 + (v / 100) * 180

  return (
    <div className="relative w-full flex flex-col items-center">
      <svg viewBox="0 0 200 110" className="w-full max-w-[170px]">
        <defs>
          <linearGradient id="gaugeGradH" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#ff3366" />
            <stop offset="25%" stopColor="#ff6b35" />
            <stop offset="50%" stopColor="#ffd700" />
            <stop offset="75%" stopColor="#39ff14" />
            <stop offset="100%" stopColor="#00ff88" />
          </linearGradient>
          <filter id="gaugeGlow"><feGaussianBlur stdDeviation="2" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
        </defs>
        <path d="M 20 100 A 80 80 0 0 1 180 100" fill="none" stroke="rgba(255,255,255,0.04)" strokeWidth="10" strokeLinecap="round" />
        <path d="M 20 100 A 80 80 0 0 1 180 100" fill="none" stroke="url(#gaugeGradH)" strokeWidth="10" strokeLinecap="round" strokeDasharray={`${v * 2.51} 251`} />
        <circle cx={100 + 80 * Math.cos((rotation * Math.PI) / 180)} cy={100 + 80 * Math.sin((rotation * Math.PI) / 180)} r="4" fill={color} filter="url(#gaugeGlow)" />
        <text x="100" y="82" textAnchor="middle" fill={color} fontSize="28" fontFamily="monospace" fontWeight="bold">{v}</text>
        <text x="100" y="105" textAnchor="middle" fill="rgba(255,255,255,0.25)" fontSize="9" fontFamily="monospace" letterSpacing="2">{label || 'FEAR & GREED'}</text>
      </svg>
    </div>
  )
}

export default function RiskMetrics({ data, portfolio, sentiment }) {
  const risk = data?.risk_metrics || data || {}
  const dailyUsedPct = risk.daily_limit_pct ? Math.min((Math.abs(risk.daily_pnl || 0) / (risk.daily_limit_pct * (risk.total_equity || 100))) * 100, 100) : 0
  const exposurePct = Math.min((risk.exposure_pct || 0) * 100, 100)
  const sent = sentiment || {}
  const moodAnalysis = sent.mood_analysis || {}
  const newsStats = sent.news_stats || {}
  const canTrade = risk.can_trade !== undefined ? risk.can_trade : true

  return (
    <div className="cinematic-border rounded-2xl p-5 h-full" style={{ background: 'linear-gradient(145deg, rgba(6,10,22,0.95) 0%, rgba(22,10,14,0.9) 100%)' }}>
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <div className="relative">
            <div className={`w-2.5 h-2.5 rounded-full ${canTrade ? 'bg-green-400' : 'bg-red-400'}`} />
            {!canTrade && <div className="absolute inset-0 w-2.5 h-2.5 rounded-full bg-red-400 animate-ping opacity-40" />}
          </div>
          <span className="text-[11px] font-bold tracking-[0.15em] text-gray-400 font-mono">RİSK & PİYASA</span>
        </div>
        <div className="px-3 py-1.5 rounded-lg border" style={{
          borderColor: canTrade ? '#39ff1420' : '#ff336620',
          background: canTrade ? '#39ff1406' : '#ff336606',
        }}>
          <span className={`text-[10px] font-bold tracking-wider font-mono ${canTrade ? 'text-green-400' : 'text-red-400'}`}>
            {canTrade ? '● SERBEST' : '● DURDURULDU'}
          </span>
        </div>
      </div>

      <div className="mb-4 rounded-xl p-3 border border-white/[0.04]" style={{ background: '#b026ff04' }}>
        <div className="flex items-center justify-between mb-1">
          <span className="text-[9px] text-gray-600 tracking-[0.12em] font-mono">PİYASA DUYGUSU</span>
          <span className="text-[9px] font-bold font-mono" style={{ color: '#b026ff' }}>{sent.fear_greed_label || 'N/A'}</span>
        </div>
        <FearGreedGauge value={sent.fear_greed_index} label={sent.market_mood?.toUpperCase() || 'DUYGU'} />
        {moodAnalysis.description && <p className="text-[11px] text-gray-500 mt-2 text-center leading-relaxed">{moodAnalysis.description}</p>}
        {moodAnalysis.recommendation && <p className="text-[10px] text-gray-600 text-center mt-1 font-mono">→ {moodAnalysis.recommendation}</p>}
        {newsStats.total > 0 && (
          <div className="flex items-center justify-center gap-3 mt-2 text-[10px] font-mono">
            <span style={{ color: '#39ff14' }}>+{newsStats.positive}</span>
            <span style={{ color: '#ff3366' }}>-{newsStats.negative}</span>
            <span className="text-gray-600">{newsStats.neutral}</span>
          </div>
        )}
      </div>

      <div className="mb-4">
        <div className="flex justify-between mb-1.5">
          <span className="text-[10px] text-gray-500 font-mono">GÜNLÜK KAYIP</span>
          <span className="text-[10px] font-bold font-mono" style={{ color: dailyUsedPct > 80 ? '#ff3366' : dailyUsedPct > 50 ? '#ffd700' : '#8b95a5' }}>
            ${Math.abs(risk.daily_pnl || 0).toFixed(0)} / ${((risk.daily_limit_pct || 0.05) * (risk.total_equity || 100)).toFixed(0)}
          </span>
        </div>
        <div className="h-1.5 bg-white/[0.03] rounded-full overflow-hidden">
          <div className="h-full rounded-full transition-all duration-700" style={{
            width: `${dailyUsedPct}%`,
            background: dailyUsedPct > 80 ? 'linear-gradient(90deg, #ff3366, #ff0040)' : dailyUsedPct > 50 ? 'linear-gradient(90deg, #ffd700, #ff6b35)' : 'linear-gradient(90deg, #39ff14, #00ff88)',
          }} />
        </div>
      </div>

      <div className="mb-4">
        <div className="flex justify-between mb-1.5">
          <span className="text-[10px] text-gray-500 font-mono">MARJİN KULLANIMI</span>
          <span className="text-[10px] font-bold font-mono" style={{ color: exposurePct > 80 ? '#ff3366' : exposurePct > 50 ? '#ffd700' : '#00f0ff' }}>
            %{exposurePct.toFixed(1)}
          </span>
        </div>
        <div className="h-1.5 bg-white/[0.03] rounded-full overflow-hidden">
          <div className="h-full rounded-full transition-all duration-700" style={{
            width: `${exposurePct}%`,
            background: exposurePct > 80 ? 'linear-gradient(90deg, #ff3366, #ff0040)' : exposurePct > 50 ? 'linear-gradient(90deg, #ffd700, #ff6b35)' : 'linear-gradient(90deg, #00f0ff, #b026ff)',
          }} />
        </div>
      </div>

      <div className="grid grid-cols-2 gap-2 mb-4">
        {[
          { label: 'POZİSYON', value: `${risk.current_positions || 0}/${risk.max_positions || 5}`, color: '#00f0ff' },
          { label: 'BOŞ SLOT', value: risk.available_slots || 0, color: '#39ff14' },
          { label: 'MARJİN', value: `$${(risk.total_exposure_usd || 0).toFixed(0)}`, color: '#ffd700' },
          { label: 'LİMİT', value: `$${((risk.daily_limit_pct || 0.05) * (risk.total_equity || 100)).toFixed(0)}`, color: '#ff6b35' },
        ].map((item, i) => (
          <div key={i} className="rounded-lg px-2.5 py-2.5 border border-white/[0.03] hover:border-white/[0.06] transition-colors" style={{ background: `${item.color}03` }}>
            <p className="text-[9px] text-gray-600 tracking-wider font-mono mb-0.5">{item.label}</p>
            <p className="text-[13px] font-bold font-mono" style={{ color: item.color }}>{item.value}</p>
          </div>
        ))}
      </div>

      <div className="rounded-xl p-3 border border-white/[0.04]" style={{ background: 'rgba(255,215,0,0.02)' }}>
        <div className="flex items-center gap-2 mb-2">
          <span className="text-[10px]" style={{ color: '#ffd700' }}>⚠</span>
          <span className="text-[9px] text-gray-600 tracking-[0.12em] font-mono">RİSK KURALLARI</span>
        </div>
        <div className="space-y-1">
          {[
            { text: 'İşlembaşı risk: %1', active: dailyUsedPct < 100 },
            { text: 'Maks kaldıraç: 10x', active: true },
            { text: 'Günlük limit: %5', active: dailyUsedPct < 80 },
            { text: 'Maks pozisyon: 5', active: (risk.current_positions || 0) < 5 },
          ].map((rule, i) => (
            <div key={i} className="text-[10px] flex items-center gap-2 font-mono">
              <span style={{ color: rule.active ? '#39ff1440' : '#ff336640' }}>›</span>
              <span style={{ color: rule.active ? '#8b95a5' : '#ff336680' }}>{rule.text}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
