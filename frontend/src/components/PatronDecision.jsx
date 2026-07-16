import { useState } from 'react'

const regimeConfig = {
  extreme_fear: { label: 'AŞIRI KORKU', color: 'neon-red', border: 'border-red-500/30', bg: 'bg-red-500/5', icon: '▼▼', gradient: 'from-red-500/20 to-orange-500/10' },
  fear: { label: 'KORKU', color: 'text-orange-400', border: 'border-orange-500/30', bg: 'bg-orange-500/5', icon: '▼', gradient: 'from-orange-500/20 to-yellow-500/10' },
  neutral: { label: 'NÖTR', color: 'text-gray-400', border: 'border-gray-600/30', bg: 'bg-gray-500/5', icon: '━', gradient: 'from-gray-500/20 to-slate-500/10' },
  greed: { label: 'AÇGÖZLÜLÜK', color: 'text-blue-400', border: 'border-blue-500/30', bg: 'bg-blue-500/5', icon: '▲', gradient: 'from-blue-500/20 to-cyan-500/10' },
  extreme_greed: { label: 'AŞIRI AÇGÖZLÜLÜK', color: 'neon-green', border: 'border-green-500/30', bg: 'bg-green-500/5', icon: '▲▲', gradient: 'from-green-500/20 to-emerald-500/10' },
}

const dirConfig = {
  long: { label: 'LONG', color: 'neon-green', border: 'border-green-500/30', bg: 'bg-green-500/5', gradient: 'from-green-500/15 to-emerald-500/5' },
  short: { label: 'SHORT', color: 'neon-red', border: 'border-red-500/30', bg: 'bg-red-500/5', gradient: 'from-red-500/15 to-orange-500/5' },
}

const voteColor = { APPROVE: 'text-green-400', REJECT: 'text-red-400', NEUTRAL: 'text-gray-500' }

export default function PatronDecision({ data }) {
  const regime = regimeConfig[data?.market_regime] || regimeConfig.neutral
  const picks = data?.top_picks || []
  const [expandedDecision, setExpandedDecision] = useState(null)

  return (
    <div className="glass-panel glass-panel-purple p-4 h-full corner-deco animate-fadeIn">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <div className={`w-11 h-11 rounded-xl bg-gradient-to-br ${regime.gradient} border ${regime.border} flex items-center justify-center animate-glowPulse`}>
            <span className="neon-purple text-xl">★</span>
          </div>
          <div>
            <h3 className="font-orbitron text-sm font-bold tracking-[0.2em] text-white">PATRON AGENT</h3>
            <p className="text-[10px] text-purple-400/50 tracking-wider">NİHAİ KARAR YETKİSİ</p>
          </div>
        </div>
        <div className={`px-3 py-1.5 rounded-lg border ${regime.border} ${regime.bg} backdrop-blur-sm`}>
          <span className={`text-xs font-bold tracking-wider ${regime.color}`}>
            {regime.icon} {regime.label}
          </span>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-2 mb-4">
        {[
          { label: 'ONAY', value: data?.approved_count || 0, color: 'neon-green', border: 'border-green-500/10', glow: 'rgba(57,255,20,0.06)' },
          { label: 'RED', value: data?.rejected_count || 0, color: 'neon-red', border: 'border-red-500/10', glow: 'rgba(255,0,64,0.06)' },
          { label: 'GÜVEN', value: `%${((data?.overall_confidence || 0) * 100)?.toFixed(0)}`, color: 'neon-purple', border: 'border-purple-500/10', glow: 'rgba(176,38,255,0.06)' },
        ].map((item, i) => (
          <div key={i} className={`bg-black/40 rounded-xl p-3 border ${item.border} text-center hover:shadow-lg transition-all`} style={{ boxShadow: `0 0 20px ${item.glow}` }}>
            <p className="text-label text-gray-500" style={{ fontSize: '10px' }}>{item.label}</p>
            <p className={`text-2xl font-bold ${item.color} font-mono`}>{item.value}</p>
          </div>
        ))}
      </div>

      {data?.thinking && data.thinking.length > 0 && (
        <div className="mb-4 bg-black/20 rounded-xl p-3 border border-cyan-500/10 animate-fadeIn">
          <p className="text-[10px] text-cyan-400/50 tracking-[0.15em] font-semibold mb-2">PATRON DÜŞÜNCE SÜRECİ</p>
          <div className="space-y-1">
            {data.thinking.map((step, i) => (
              <div key={i} className="flex items-center gap-2 text-xs text-gray-400">
                <span className="text-cyan-500/40 font-mono">▸</span>
                <span className="leading-relaxed">{step}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {picks.length > 0 ? (
        <div className="space-y-2">
          <p className="text-label text-cyan-400/50 mb-2 tracking-[0.12em]" style={{ fontSize: '11px' }}>EN İYİ SEÇİMLER</p>
          {picks.slice(0, 5).map((pick, i) => {
            const dir = dirConfig[pick.direction] || dirConfig.long
            const isExpanded = expandedDecision === `${pick.symbol}-${i}`
            const votes = pick.agent_votes || {}

            return (
              <div key={i} className={`glass-panel bg-gradient-to-r ${dir.gradient} ${dir.border} border p-3 rounded-xl card-hover`}>
                <div
                  className="flex items-center justify-between mb-2 cursor-pointer"
                  onClick={() => setExpandedDecision(isExpanded ? null : `${pick.symbol}-${i}`)}
                >
                  <div className="flex items-center gap-2.5">
                    <span className="text-xs text-gray-500 font-mono w-5">#{i+1}</span>
                    <span className="text-base font-bold text-white tracking-wide">{pick.symbol}</span>
                    <span className={`tag ${dir.border} ${dir.color}`}>{dir.label}</span>
                  </div>
                  <div className="flex items-center gap-3">
                    <span className="text-2xl font-bold text-white font-mono">%{(pick.composite_score * 100)?.toFixed(0)}</span>
                    <span className="text-gray-600 text-xs transition-transform duration-200" style={{ transform: isExpanded ? 'rotate(180deg)' : 'rotate(0)' }}>▼</span>
                  </div>
                </div>

                {isExpanded && votes && Object.keys(votes).length > 0 && (
                  <div className="mt-2 space-y-1.5 border-t border-white/5 pt-2 animate-fadeIn">
                    <p className="text-[10px] text-cyan-400/50 tracking-[0.12em] font-semibold">AJAN OYLARI</p>
                    <div className="bg-black/20 rounded-lg p-2.5 space-y-1">
                      {Object.entries(votes).map(([agent, vote]) => (
                        <div key={agent} className="flex items-center gap-2 text-xs">
                          <span className={`font-bold w-20 font-mono ${voteColor[vote.vote] || 'text-gray-500'}`}>
                            {vote.vote === 'APPROVE' ? '✓' : vote.vote === 'REJECT' ? '✗' : '○'} {agent.toUpperCase()}
                          </span>
                          <span className="text-gray-400 flex-1">{vote.detail}</span>
                          {vote.score !== undefined && (
                            <span className="text-gray-500 font-mono">%{(vote.score * 100).toFixed(0)}</span>
                          )}
                        </div>
                      ))}
                      {votes.patron && (
                        <div className="mt-1 pt-1.5 border-t border-white/5">
                          <div className="flex items-center gap-2 text-xs">
                            <span className={`font-bold w-20 font-mono ${voteColor[votes.patron.vote]}`}>
                              {votes.patron.vote === 'APPROVE' ? '✓' : '✗'} PATRON
                            </span>
                            <span className="text-gray-300">{votes.patron.detail}</span>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {pick.risk_params && Object.keys(pick.risk_params).length > 0 && (
                  <div className="flex items-center gap-3 text-xs text-gray-400 mt-2 bg-black/10 rounded-lg px-2.5 py-1.5">
                    <span className="font-mono">${pick.risk_params.position_size_usd?.toFixed(0)}</span>
                    <span className="text-purple-400 font-mono">x{pick.risk_params.leverage}</span>
                    <span className="text-green-400 font-mono">TP:{pick.risk_params.take_profit ? `$${pick.risk_params.take_profit?.toFixed(2)}` : '-'}</span>
                    <span className="text-red-400 font-mono">SL:{pick.risk_params.stop_loss ? `$${pick.risk_params.stop_loss?.toFixed(2)}` : '-'}</span>
                    <span className="text-cyan-400 font-mono">RR:{pick.risk_params.risk_reward_ratio?.toFixed(1)}</span>
                  </div>
                )}

                {pick.reasons && pick.reasons.length > 0 && (
                  <div className="mt-2 flex flex-wrap gap-1">
                    {pick.reasons.slice(0, 4).map((r, ri) => (
                      <span key={ri} className="text-[10px] px-2 py-0.5 rounded-full bg-black/30 text-gray-400 border border-white/5">{r}</span>
                    ))}
                  </div>
                )}
              </div>
            )
          })}
        </div>
      ) : (
        <div className="text-center py-10">
          <div className="text-5xl mb-3 opacity-15 animate-float">★</div>
          <p className="text-sm text-gray-500">Henüz onaylanan seçim yok</p>
          <p className="text-[10px] text-gray-600 mt-1">Sistem ilk döngüsünü bekliyor</p>
        </div>
      )}
    </div>
  )
}
