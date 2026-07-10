import { useState } from 'react'

const STOCK_KEYWORDS = [
  'STOCK', 'ETF', 'BOND', 'FUND', 'INDEX', 'FUTURES',
  'OIL', 'GOLD', 'SILVER', 'NATGAS', 'COPPER', 'PLATINUM',
  'DOW', 'SP500', 'NASDAQ', 'SPX', 'DAX', 'FTSE', 'NIKKEI',
  'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD',
  'XAU', 'XAG', 'XPT', 'XPD',
]

function isStockSymbol(symbol) {
  const sym = (symbol || '').toUpperCase()
  return STOCK_KEYWORDS.some(kw => sym.includes(kw))
}

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
  const decisions = (data?.decisions || data?.top_picks || []).filter(d => !isStockSymbol(d.symbol))
  const approved = decisions.filter(d => d.approved)
  const rejected = decisions.filter(d => !d.approved)
  const [expandedSignal, setExpandedSignal] = useState(null)
  const [showRejected, setShowRejected] = useState(false)
  const [expandedRejected, setExpandedRejected] = useState(null)

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

      {/* Reddedilen Sinyaller */}
      {rejected.length > 0 && (
        <div className="mt-4 pt-3 border-t border-red-500/10">
          <button
            onClick={() => setShowRejected(!showRejected)}
            className="flex items-center justify-between w-full mb-2"
          >
            <p className="text-label text-red-400/70" style={{ fontSize: '11px' }}>
              REDDEDİLEN SİNYALLER ({rejected.length})
            </p>
            <span className="text-red-400/50 text-xs">{showRejected ? '▲ Gizle' : '▼ Göster'}</span>
          </button>

          {showRejected && (
            <div className="space-y-2">
              {rejected.slice(0, fullPage ? 30 : 5).map((d, i) => {
                const dir = dirConfig[d.direction] || dirConfig.hold
                const votes = d.agent_votes || {}
                const isExpanded = expandedRejected === `${d.symbol}-${i}`
                const confidence = d.composite_score || 0
                const confPercent = (confidence * 100).toFixed(1)

                const rejectReasons = []
                if (d.veto_reason) {
                  rejectReasons.push(d.veto_reason)
                }
                Object.entries(votes).forEach(([agent, vote]) => {
                  if (vote.vote === 'REJECT') {
                    rejectReasons.push(`${agent}: ${vote.detail}`)
                  }
                })

                return (
                  <div key={i} className="bg-black/20 rounded-lg border border-red-500/10 overflow-hidden">
                    <div
                      className="flex items-center justify-between p-2.5 cursor-pointer hover:bg-red-500/5 transition-colors"
                      onClick={() => setExpandedRejected(isExpanded ? null : `${d.symbol}-${i}`)}
                    >
                      <div className="flex items-center gap-2">
                        <span className="w-2 h-2 rounded-full bg-red-500" />
                        <span className="text-sm font-bold text-white">{d.symbol}</span>
                        <span className={`tag text-[10px] ${dir.border} ${dir.color}`}>{dir.label}</span>
                        <span className="text-[10px] text-gray-500">
                          Güven: <span className="text-red-400">%{confPercent}</span>
                        </span>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="text-[10px] text-red-400/60 max-w-[200px] truncate">
                          {d.veto_reason?.substring(0, 50)}
                        </span>
                        <span className="text-gray-600 text-xs">{isExpanded ? '▲' : '▼'}</span>
                      </div>
                    </div>

                    {isExpanded && (
                      <div className="px-2.5 pb-2.5 space-y-2 border-t border-red-500/5 pt-2">
                        {/* Red Nedenleri */}
                        {rejectReasons.length > 0 && (
                          <div>
                            <p className="text-[10px] text-red-400/50 tracking-wider mb-1">RED NEDENLERİ:</p>
                            <div className="space-y-1">
                              {rejectReasons.map((reason, ri) => (
                                <div key={ri} className="flex items-start gap-1.5 text-xs bg-red-500/5 rounded px-2 py-1">
                                  <span className="text-red-400 mt-0.5">✗</span>
                                  <span className="text-gray-300">{reason}</span>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}

                        {/* Ajan Oyları */}
                        {votes && Object.keys(votes).length > 0 && (
                          <div>
                            <p className="text-[10px] text-gray-400/50 tracking-wider mb-1">AJAN OYLARI:</p>
                            <div className="grid grid-cols-2 gap-1">
                              {Object.entries(votes).map(([agent, vote]) => (
                                <div key={agent} className="flex items-center gap-1.5 text-xs bg-black/20 rounded px-2 py-1">
                                  <span className={`font-bold ${vote.vote === 'APPROVE' ? 'text-green-400' : vote.vote === 'REJECT' ? 'text-red-400' : 'text-gray-500'}`}>
                                    {vote.vote === 'APPROVE' ? '✓' : vote.vote === 'REJECT' ? '✗' : '○'}
                                  </span>
                                  <span className="text-gray-300 capitalize">{agent}</span>
                                  <span className="text-gray-500 ml-auto text-[10px] truncate max-w-[120px]">{vote.detail}</span>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}

                        {/* Skor Dağılımı */}
                        {d.breakdown && Object.keys(d.breakdown).length > 0 && (
                          <div>
                            <p className="text-[10px] text-gray-400/50 tracking-wider mb-1">SKOR DAĞILIMI:</p>
                            <div className="flex gap-2 flex-wrap">
                              {Object.entries(d.breakdown).map(([key, val]) => (
                                <div key={key} className="flex items-center gap-1">
                                  <div className="w-14 h-1.5 bg-black/30 rounded-full overflow-hidden">
                                    <div className={`h-full rounded-full ${val >= 0.5 ? 'bg-green-500' : val >= 0.3 ? 'bg-yellow-500' : 'bg-red-500'}`} style={{ width: `${Math.min(val * 100, 100)}%` }} />
                                  </div>
                                  <span className="text-[10px] text-gray-500 w-8 text-right">{(val * 100)?.toFixed(0)}%</span>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}

                        {/* Sebepler */}
                        {d.reasons && d.reasons.length > 0 && (
                          <div>
                            <p className="text-[10px] text-gray-400/50 tracking-wider mb-1">TEKNİK SEBEPLER:</p>
                            <div className="flex flex-wrap gap-1">
                              {d.reasons.slice(0, 5).map((reason, ri) => (
                                <span key={ri} className="text-[10px] text-gray-400 bg-black/20 px-1.5 py-0.5 rounded">
                                  {reason}
                                </span>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                )
              })}
              {rejected.length > (fullPage ? 30 : 5) && (
                <p className="text-center text-xs text-gray-600 py-2">
                  +{rejected.length - (fullPage ? 30 : 5)} daha fazla red edilen sinyal
                </p>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
