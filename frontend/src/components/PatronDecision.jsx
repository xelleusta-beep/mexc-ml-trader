import { useState } from 'react'

const regimeConfig = {
  extreme_fear: { label: 'AŞIRI KORKU', color: 'neon-red', border: 'border-red-500/30', bg: 'bg-red-500/5', icon: '▼▼' },
  fear: { label: 'KORKU', color: 'text-orange-400', border: 'border-orange-500/30', bg: 'bg-orange-500/5', icon: '▼' },
  neutral: { label: 'NÖTR', color: 'text-gray-400', border: 'border-gray-600/30', bg: 'bg-gray-500/5', icon: '━' },
  greed: { label: 'AÇGÖZLÜLÜK', color: 'text-blue-400', border: 'border-blue-500/30', bg: 'bg-blue-500/5', icon: '▲' },
  extreme_greed: { label: 'AŞIRI AÇGÖZLÜLÜK', color: 'neon-green', border: 'border-green-500/30', bg: 'bg-green-500/5', icon: '▲▲' },
}

const dirConfig = {
  long: { label: 'LONG', color: 'neon-green', border: 'border-green-500/30', bg: 'bg-green-500/5' },
  short: { label: 'SHORT', color: 'neon-red', border: 'border-red-500/30', bg: 'bg-red-500/5' },
}

const voteColor = { APPROVE: 'text-green-400', REJECT: 'text-red-400', NEUTRAL: 'text-gray-500' }

export default function PatronDecision({ data }) {
  const regime = regimeConfig[data?.market_regime] || regimeConfig.neutral
  const picks = data?.top_picks || []
  const [expandedDecision, setExpandedDecision] = useState(null)

  return (
    <div className="glass-panel glass-panel-purple p-4 h-full corner-deco">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-purple-500/20 to-cyan-500/20 border border-purple-500/30 flex items-center justify-center">
            <span className="neon-purple text-xl">★</span>
          </div>
          <div>
            <h3 className="font-orbitron text-base font-bold tracking-widest text-white">PATRON AGENT</h3>
            <p className="text-xs text-purple-400/60">NİHAİ KARAR YETKİSİ</p>
          </div>
        </div>
        <div className={`px-3 py-1.5 rounded-sm border ${regime.border} ${regime.bg}`}>
          <span className={`text-sm font-bold tracking-wider ${regime.color}`}>
            {regime.icon} {regime.label}
          </span>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-2 mb-4">
        <div className="bg-black/40 rounded-lg p-3 border border-green-500/10 text-center">
          <p className="text-label text-green-500/60" style={{ fontSize: '10px' }}>ONAY</p>
          <p className="text-3xl font-bold neon-green">{data?.approved_count || 0}</p>
        </div>
        <div className="bg-black/40 rounded-lg p-3 border border-red-500/10 text-center">
          <p className="text-label text-red-500/60" style={{ fontSize: '10px' }}>RED</p>
          <p className="text-3xl font-bold neon-red">{data?.rejected_count || 0}</p>
        </div>
        <div className="bg-black/40 rounded-lg p-3 border border-purple-500/10 text-center">
          <p className="text-label text-purple-500/60" style={{ fontSize: '10px' }}>GÜVEN</p>
          <p className="text-3xl font-bold neon-purple">%{((data?.overall_confidence || 0) * 100)?.toFixed(0)}</p>
        </div>
      </div>

      {data?.thinking && data.thinking.length > 0 && (
        <div className="mb-4 bg-black/20 rounded-lg p-3 border border-cyan-500/10">
          <p className="text-[10px] text-cyan-400/50 tracking-wider mb-1">PATRON DÜŞÜNCE SÜRECİ</p>
          {data.thinking.map((step, i) => (
            <div key={i} className="flex items-center gap-2 text-xs text-gray-400">
              <span className="text-cyan-500/40">▸</span> {step}
            </div>
          ))}
        </div>
      )}

      {picks.length > 0 ? (
        <div className="space-y-2">
          <p className="text-label text-cyan-400/50 mb-2" style={{ fontSize: '11px' }}>EN İYİ SEÇİMLER</p>
          {picks.slice(0, 5).map((pick, i) => {
            const dir = dirConfig[pick.direction] || dirConfig.long
            const isExpanded = expandedDecision === `${pick.symbol}-${i}`
            const votes = pick.agent_votes || {}

            return (
              <div key={i} className={`glass-panel ${dir.bg} ${dir.border} border p-3 rounded-lg`}>
                <div
                  className="flex items-center justify-between mb-2 cursor-pointer"
                  onClick={() => setExpandedDecision(isExpanded ? null : `${pick.symbol}-${i}`)}
                >
                  <div className="flex items-center gap-2">
                    <span className="text-sm text-gray-500 w-5">#{i+1}</span>
                    <span className="text-base font-bold text-white">{pick.symbol}</span>
                    <span className={`tag ${dir.border} ${dir.color}`}>{dir.label}</span>
                  </div>
                  <div className="flex items-center gap-3">
                    <span className="text-2xl font-bold text-white">%{(pick.composite_score * 100)?.toFixed(0)}</span>
                    <span className="text-gray-600 text-xs">{isExpanded ? '▲' : '▼'}</span>
                  </div>
                </div>

                {isExpanded && votes && Object.keys(votes).length > 0 && (
                  <div className="mt-2 space-y-1.5 border-t border-white/5 pt-2">
                    <p className="text-[10px] text-cyan-400/50 tracking-wider">AJAN OYLARI:</p>
                    {Object.entries(votes).map(([agent, vote]) => (
                      <div key={agent} className="flex items-center gap-2 text-xs">
                        <span className={`font-bold w-20 ${voteColor[vote.vote] || 'text-gray-500'}`}>
                          {vote.vote === 'APPROVE' ? '✓' : vote.vote === 'REJECT' ? '✗' : '○'} {agent.toUpperCase()}
                        </span>
                        <span className="text-gray-400 flex-1">{vote.detail}</span>
                        {vote.score !== undefined && (
                          <span className="text-gray-500">%{(vote.score * 100).toFixed(0)}</span>
                        )}
                      </div>
                    ))}
                    {votes.patron && (
                      <div className="mt-1 pt-1 border-t border-white/5">
                        <div className="flex items-center gap-2 text-xs">
                          <span className={`font-bold w-20 ${voteColor[votes.patron.vote]}`}>
                            {votes.patron.vote === 'APPROVE' ? '✓' : '✗'} PATRON
                          </span>
                          <span className="text-gray-300">{votes.patron.detail}</span>
                        </div>
                      </div>
                    )}
                  </div>
                )}

                {pick.risk_params && Object.keys(pick.risk_params).length > 0 && (
                  <div className="flex items-center gap-4 text-sm text-gray-400 mt-1">
                    <span>${pick.risk_params.position_size_usd?.toFixed(0)}</span>
                    <span className="text-purple-400">x{pick.risk_params.leverage}</span>
                    <span className="text-green-400">TP:{pick.risk_params.take_profit ? `$${pick.risk_params.take_profit?.toFixed(2)}` : '-'}</span>
                    <span className="text-red-400">SL:{pick.risk_params.stop_loss ? `$${pick.risk_params.stop_loss?.toFixed(2)}` : '-'}</span>
                    <span className="text-cyan-400">RR:{pick.risk_params.risk_reward_ratio?.toFixed(1)}</span>
                  </div>
                )}

                {pick.reasons && pick.reasons.length > 0 && (
                  <div className="mt-2 flex flex-wrap gap-1">
                    {pick.reasons.slice(0, 4).map((r, ri) => (
                      <span key={ri} className="text-[10px] px-1.5 py-0.5 rounded bg-black/30 text-gray-400 border border-white/5">{r}</span>
                    ))}
                  </div>
                )}
              </div>
            )
          })}
        </div>
      ) : (
        <div className="text-center py-10">
          <div className="text-5xl mb-3 opacity-20">★</div>
          <p className="text-sm text-gray-500">Henüz onaylanan seçim yok</p>
        </div>
      )}
    </div>
  )
}
