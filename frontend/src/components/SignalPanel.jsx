import { useState } from 'react'

const dirConfig = {
  long: { label: 'LONG', color: 'neon-green', border: 'border-green-500/30', bg: 'bg-green-500/5' },
  short: { label: 'SHORT', color: 'neon-red', border: 'border-red-500/30', bg: 'bg-red-500/5' },
  hold: { label: 'HOLD', color: 'text-gray-500', border: 'border-gray-700', bg: 'bg-gray-800/30' },
}

const confConfig = {
  very_high: { label: 'MAX', color: 'neon-green', bg: 'bg-green-500/10' },
  high: { label: 'YÜKSEK', color: 'text-blue-400', bg: 'bg-blue-500/10' },
  medium: { label: 'ORTA', color: 'text-yellow-400', bg: 'bg-yellow-500/10' },
  low: { label: 'DÜŞÜK', color: 'text-orange-400', bg: 'bg-orange-500/10' },
  very_low: { label: 'ÇOK DÜŞÜK', color: 'text-red-400', bg: 'bg-red-500/10' },
}

export default function SignalPanel({ data, fullPage }) {
  const decisions = data?.decisions || data?.top_picks || []
  const approved = decisions.filter(d => d.approved)
  const rejected = decisions.filter(d => !d.approved)
  const [expandedSignal, setExpandedSignal] = useState(null)

  return (
    <div className={`glass-panel glass-panel-cyan p-4 ${fullPage ? '' : 'h-full'} corner-deco`}>
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <span className="text-cyan-400 text-base">◇</span>
          <span className="text-label text-cyan-400/70">SİNYAL PANELİ</span>
        </div>
        <div className="text-sm text-gray-500">
          {approved.length} onay / {rejected.length} red
        </div>
      </div>

      {approved.length > 0 ? (
        <div className="space-y-2">
          {approved.slice(0, fullPage ? 20 : 5).map((d, i) => {
            const dir = dirConfig[d.direction] || dirConfig.hold
            const conf = confConfig[d.confidence_level] || confConfig.low
            const breakdown = d.breakdown || {}
            const votes = d.agent_votes || {}
            const isExpanded = expandedSignal === `${d.symbol}-${i}`

            return (
              <div key={i} className={`glass-panel ${dir.bg} ${dir.border} border p-3 rounded-lg transition-all`}>
                <div
                  className="flex items-center justify-between mb-2 cursor-pointer"
                  onClick={() => setExpandedSignal(isExpanded ? null : `${d.symbol}-${i}`)}
                >
                  <div className="flex items-center gap-2">
                    <span className="text-base font-bold text-white">{d.symbol}</span>
                    <span className={`tag ${dir.border} ${dir.color}`}>{dir.label}</span>
                    <span className={`px-2 py-0.5 rounded-sm text-xs font-semibold tracking-wider ${conf.bg} ${conf.color}`}>
                      {conf.label}
                    </span>
                  </div>
                  <div className="flex items-center gap-3">
                    <span className="text-2xl font-bold text-white">%{(d.composite_score * 100)?.toFixed(0)}</span>
                    <span className="text-gray-600 text-xs">{isExpanded ? '▲' : '▼'}</span>
                  </div>
                </div>

                {isExpanded && (
                  <div className="mt-2 space-y-3 border-t border-white/5 pt-2">
                    {/* Agent Votes */}
                    {votes && Object.keys(votes).length > 0 && (
                      <div>
                        <p className="text-[10px] text-cyan-400/50 tracking-wider mb-1">AJAN OYLARI:</p>
                        <div className="grid grid-cols-2 gap-1">
                          {Object.entries(votes).map(([agent, vote]) => (
                            <div key={agent} className="flex items-center gap-1.5 text-xs bg-black/20 rounded px-2 py-1">
                              <span className={`font-bold ${vote.vote === 'APPROVE' ? 'text-green-400' : vote.vote === 'REJECT' ? 'text-red-400' : 'text-gray-500'}`}>
                                {vote.vote === 'APPROVE' ? '✓' : vote.vote === 'REJECT' ? '✗' : '○'}
                              </span>
                              <span className="text-gray-300 capitalize">{agent}</span>
                              <span className="text-gray-500 ml-auto text-[10px]">{vote.detail?.substring(0, 40)}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                )}

                {/* Risk Params */}
                {d.risk_params && Object.keys(d.risk_params).length > 0 && (
                  <div className="flex items-center gap-4 text-sm text-gray-400">
                    <span>${d.risk_params.position_size_usd?.toFixed(0)}</span>
                    <span className="text-purple-400">x{d.risk_params.leverage}</span>
                    <span className="text-green-400">TP:{d.risk_params.take_profit ? `$${d.risk_params.take_profit?.toFixed(2)}` : '-'}</span>
                    <span className="text-red-400">SL:{d.risk_params.stop_loss ? `$${d.risk_params.stop_loss?.toFixed(2)}` : '-'}</span>
                    <span className="text-cyan-400">RR:{d.risk_params.risk_reward_ratio?.toFixed(1)}</span>
                  </div>
                )}

                {/* Breakdown bars */}
                {Object.keys(breakdown).length > 0 && (
                  <div className="flex gap-2 mt-1">
                    {Object.entries(breakdown).map(([key, val]) => (
                      <div key={key} className="flex items-center gap-1">
                        <div className="w-14 h-1.5 bg-black/30 rounded-full overflow-hidden">
                          <div className="h-full bg-gradient-to-r from-cyan-500 to-purple-500 rounded-full" style={{ width: `${Math.min(val * 100, 100)}%` }} />
                        </div>
                        <span className="text-[10px] text-gray-500 w-8 text-right">{(val * 100)?.toFixed(0)}%</span>
                      </div>
                    ))}
                  </div>
                )}
                </div>
              )}

              {!isExpanded && d.risk_params && Object.keys(d.risk_params).length > 0 && (
                <div className="flex items-center gap-4 text-sm text-gray-400 mt-1">
                  <span>${d.risk_params.position_size_usd?.toFixed(0)}</span>
                  <span className="text-purple-400">x{d.risk_params.leverage}</span>
                  <span className="text-green-400">TP:{d.risk_params.take_profit ? `$${d.risk_params.take_profit?.toFixed(2)}` : '-'}</span>
                  <span className="text-red-400">SL:{d.risk_params.stop_loss ? `$${d.risk_params.stop_loss?.toFixed(2)}` : '-'}</span>
                  <span className="text-cyan-400">RR:{d.risk_params.risk_reward_ratio?.toFixed(1)}</span>
                </div>
              )}
              </div>
            )
          })}
        </div>
      ) : (
        <div className="text-center py-10">
          <div className="text-5xl mb-3 opacity-20">◇</div>
          <p className="text-sm text-gray-500">Aktif sinyal yok</p>
        </div>
      )}

      {!fullPage && rejected.length > 0 && (
        <div className="mt-3 pt-3 border-t border-cyan-500/10">
          <p className="text-label text-red-400/50 mb-2" style={{ fontSize: '11px' }}>REDDEDİLEN</p>
          <div className="space-y-1">
            {rejected.slice(0, 3).map((d, i) => (
              <div key={i} className="flex items-center justify-between p-2 bg-black/20 rounded text-sm">
                <span className="text-gray-400 font-semibold">{d.symbol}</span>
                <span className="text-red-400/60 truncate ml-2 text-xs">{d.veto_reason?.substring(0, 60)}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
