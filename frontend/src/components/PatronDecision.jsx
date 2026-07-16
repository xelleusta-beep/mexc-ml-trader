import { useState } from 'react'

const regimeConfig = {
  extreme_fear: { label: 'AŞIRI KORKU', color: '#ff3366', icon: '▼▼' },
  fear: { label: 'KORKU', color: '#ff6b35', icon: '▼' },
  neutral: { label: 'NÖTR', color: '#8b95a5', icon: '━' },
  greed: { label: 'AÇGÖZLÜLÜK', color: '#00f0ff', icon: '▲' },
  extreme_greed: { label: 'AŞIRI AÇGÖZLÜLÜK', color: '#39ff14', icon: '▲▲' },
}

const dirConfig = {
  long: { label: 'LONG', color: '#39ff14', bg: '#39ff1408' },
  short: { label: 'SHORT', color: '#ff3366', bg: '#ff336608' },
}

const voteColor = { APPROVE: '#39ff14', REJECT: '#ff3366', NEUTRAL: '#8b95a5' }

export default function PatronDecision({ data }) {
  const regime = regimeConfig[data?.market_regime] || regimeConfig.neutral
  const picks = data?.top_picks || []
  const [expandedDecision, setExpandedDecision] = useState(null)

  return (
    <div className="cinematic-border rounded-2xl p-5 h-full" style={{ background: 'linear-gradient(145deg, rgba(6,10,22,0.95) 0%, rgba(12,10,28,0.9) 100%)' }}>
      <div className="flex items-center justify-between mb-5">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl flex items-center justify-center" style={{ background: 'linear-gradient(135deg, #ffd70012, #b026ff08)', border: '1px solid #ffd70020' }}>
            <span style={{ color: '#ffd700', fontSize: '16px' }}>★</span>
          </div>
          <div>
            <h3 className="text-[13px] font-bold tracking-[0.15em] text-white font-mono">PATRON AGENT</h3>
            <p className="text-[9px] text-gray-600 tracking-wider font-mono">NİHAİ KARAR YETKİSİ</p>
          </div>
        </div>
        <div className="px-3 py-1.5 rounded-lg border flex items-center gap-2" style={{ borderColor: `${regime.color}25`, background: `${regime.color}08` }}>
          <span style={{ color: regime.color, fontSize: '10px' }}>{regime.icon}</span>
          <span className="text-[10px] font-bold tracking-wider font-mono" style={{ color: regime.color }}>{regime.label}</span>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-2 mb-5">
        {[
          { label: 'ONAY', value: data?.approved_count || 0, color: '#39ff14' },
          { label: 'RED', value: data?.rejected_count || 0, color: '#ff3366' },
          { label: 'GÜVEN', value: `%${((data?.overall_confidence || 0) * 100)?.toFixed(0)}`, color: '#b026ff' },
        ].map((item, i) => (
          <div key={i} className="rounded-xl p-3 text-center border border-white/[0.04] transition-all hover:border-white/[0.08]" style={{ background: `${item.color}04` }}>
            <p className="text-[9px] text-gray-600 tracking-wider font-mono mb-1">{item.label}</p>
            <p className="text-[22px] font-bold font-mono" style={{ color: item.color }}>{item.value}</p>
          </div>
        ))}
      </div>

      {data?.thinking && data.thinking.length > 0 && (
        <div className="mb-5 rounded-xl p-3 border border-white/[0.04]" style={{ background: 'rgba(0,240,255,0.02)' }}>
          <p className="text-[9px] text-gray-600 tracking-[0.12em] font-mono mb-2">PATRON DÜŞÜNCE SÜRECİ</p>
          <div className="space-y-1">
            {data.thinking.map((step, i) => (
              <div key={i} className="flex items-center gap-2 text-[11px]">
                <span className="text-gray-700 font-mono">›</span>
                <span className="text-gray-400">{step}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {picks.length > 0 ? (
        <div className="space-y-2">
          <p className="text-[10px] text-gray-600 tracking-[0.12em] font-mono mb-2">EN İYİ SEÇİMLER</p>
          {picks.slice(0, 5).map((pick, i) => {
            const dir = dirConfig[pick.direction] || dirConfig.long
            const isExpanded = expandedDecision === `${pick.symbol}-${i}`
            const votes = pick.agent_votes || {}

            return (
              <div key={i} className="rounded-xl border border-white/[0.04] overflow-hidden transition-all hover:border-white/[0.08]" style={{ background: dir.bg }}>
                <div
                  className="p-3 cursor-pointer flex items-center justify-between"
                  onClick={() => setExpandedDecision(isExpanded ? null : `${pick.symbol}-${i}`)}
                >
                  <div className="flex items-center gap-2.5">
                    <span className="text-[11px] text-gray-700 font-mono">#{i+1}</span>
                    <span className="text-[13px] font-bold text-white font-mono">{pick.symbol}</span>
                    <span className="text-[9px] px-2 py-0.5 rounded-md font-bold tracking-wider font-mono border" style={{ color: dir.color, borderColor: `${dir.color}30`, background: `${dir.color}10` }}>{dir.label}</span>
                  </div>
                  <div className="flex items-center gap-3">
                    <span className="text-[18px] font-bold text-white font-mono">%{(pick.composite_score * 100)?.toFixed(0)}</span>
                    <svg className={`w-3 h-3 text-gray-600 transition-transform duration-200 ${isExpanded ? 'rotate-180' : ''}`} fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" /></svg>
                  </div>
                </div>

                {isExpanded && (
                  <div className="px-3 pb-3 border-t border-white/[0.03] animate-fadeIn">
                    {votes && Object.keys(votes).length > 0 && (
                      <div className="mt-2 space-y-1.5">
                        <p className="text-[9px] text-gray-600 tracking-[0.12em] font-mono">AJAN OYLARI</p>
                        <div className="bg-black/20 rounded-lg p-2.5 space-y-1">
                          {Object.entries(votes).map(([agent, vote]) => (
                            <div key={agent} className="flex items-center gap-2 text-[11px]">
                              <span className="font-bold font-mono w-20" style={{ color: voteColor[vote.vote] || '#8b95a5' }}>
                                {vote.vote === 'APPROVE' ? '✓' : vote.vote === 'REJECT' ? '✗' : '○'} {agent.toUpperCase()}
                              </span>
                              <span className="text-gray-400 flex-1">{vote.detail}</span>
                              {vote.score !== undefined && <span className="text-gray-600 font-mono">%{(vote.score * 100).toFixed(0)}</span>}
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {pick.risk_params && Object.keys(pick.risk_params).length > 0 && (
                      <div className="flex items-center gap-3 text-[11px] text-gray-400 mt-2 bg-black/15 rounded-lg px-2.5 py-1.5 font-mono">
                        <span>${pick.risk_params.position_size_usd?.toFixed(0)}</span>
                        <span style={{ color: '#b026ff' }}>x{pick.risk_params.leverage}</span>
                        <span style={{ color: '#39ff14' }}>TP:{pick.risk_params.take_profit ? `$${pick.risk_params.take_profit?.toFixed(2)}` : '-'}</span>
                        <span style={{ color: '#ff3366' }}>SL:{pick.risk_params.stop_loss ? `$${pick.risk_params.stop_loss?.toFixed(2)}` : '-'}</span>
                        <span style={{ color: '#00f0ff' }}>RR:{pick.risk_params.risk_reward_ratio?.toFixed(1)}</span>
                      </div>
                    )}

                    {pick.reasons && pick.reasons.length > 0 && (
                      <div className="mt-2 flex flex-wrap gap-1">
                        {pick.reasons.slice(0, 4).map((r, ri) => (
                          <span key={ri} className="text-[9px] px-2 py-0.5 rounded-full bg-black/20 text-gray-500 border border-white/[0.03] font-mono">{r}</span>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </div>
            )
          })}
        </div>
      ) : (
        <div className="text-center py-12">
          <div className="text-4xl mb-3 opacity-10">★</div>
          <p className="text-[13px] text-gray-600">Henüz onaylanan seçim yok</p>
          <p className="text-[10px] text-gray-700 mt-1 font-mono">Sistem ilk döngüsünü bekliyor</p>
        </div>
      )}
    </div>
  )
}
