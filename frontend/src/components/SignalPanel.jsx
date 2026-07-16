import { useState } from 'react'

const BLACKLIST_BASE = ['USDC','BUSD','DAI','TUSD','USDP','FDUSD','PYUSD','EUR','GBP','JPY','AUD','CAD','CHF','NZD','CNY','OIL','GOLD','SILVER','NATGAS','COPPER','PLATINUM','DOW','SP500','NASDAQ','SPX','DAX','FTSE','NIKKEI','XAU','XAG','XPT','XPD','STOCK','ETF','BOND','FUND','INDEX','FUTURES']
function isStockSymbol(symbol) { const sym = (symbol || '').toUpperCase(); return BLACKLIST_BASE.some(kw => sym.includes(kw)) }

const dirConfig = {
  long: { label: 'LONG', color: '#39ff14' },
  short: { label: 'SHORT', color: '#ff3366' },
  hold: { label: 'HOLD', color: '#8b95a5' },
}

const confConfig = {
  very_high: { label: 'MAX', color: '#39ff14' },
  high: { label: 'YÜKSEK', color: '#00f0ff' },
  medium: { label: 'ORTA', color: '#ffd700' },
  low: { label: 'DÜŞÜK', color: '#ff6b35' },
  very_low: { label: 'ÇOK DÜŞÜK', color: '#ff3366' },
}

export default function SignalPanel({ data, fullPage }) {
  const decisions = (data?.decisions || data?.top_picks || []).filter(d => !isStockSymbol(d.symbol))
  const approved = decisions.filter(d => d.approved)
  const rejected = decisions.filter(d => !d.approved)
  const [expandedSignal, setExpandedSignal] = useState(null)
  const [showRejected, setShowRejected] = useState(false)
  const [expandedRejected, setExpandedRejected] = useState(null)

  return (
    <div className={`cinematic-border rounded-2xl p-5 ${fullPage ? '' : 'h-full'}`} style={{ background: 'linear-gradient(145deg, rgba(6,10,22,0.95) 0%, rgba(10,14,28,0.9) 100%)' }}>
      <div className="flex items-center justify-between mb-5">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl flex items-center justify-center" style={{ background: '#00f0ff08', border: '1px solid #00f0ff15' }}>
            <span style={{ color: '#00f0ff', fontSize: '13px' }}>◇</span>
          </div>
          <span className="text-[11px] font-bold tracking-[0.15em] text-gray-400 font-mono">SİNYAL PANELİ</span>
        </div>
        <div className="text-[11px] text-gray-600 font-mono">{approved.length} onay / {rejected.length} red</div>
      </div>

      {approved.length > 0 ? (
        <div className="space-y-2">
          {approved.slice(0, fullPage ? 20 : 5).map((d, i) => {
            const dir = dirConfig[d.direction] || dirConfig.hold
            const conf = confConfig[d.confidence_level] || confConfig.low
            const votes = d.agent_votes || {}
            const isExpanded = expandedSignal === `${d.symbol}-${i}`
            return (
              <div key={i} className="rounded-xl border border-white/[0.04] overflow-hidden transition-all hover:border-white/[0.08]" style={{ background: `${dir.color}04` }}>
                <div className="p-3 cursor-pointer flex items-center justify-between" onClick={() => setExpandedSignal(isExpanded ? null : `${d.symbol}-${i}`)}>
                  <div className="flex items-center gap-2">
                    <span className="text-[13px] font-bold text-white font-mono">{d.symbol}</span>
                    <span className="text-[9px] px-2 py-0.5 rounded-md font-bold tracking-wider font-mono border" style={{ color: dir.color, borderColor: `${dir.color}30`, background: `${dir.color}10` }}>{dir.label}</span>
                    <span className="text-[9px] px-2 py-0.5 rounded-md font-bold tracking-wider font-mono" style={{ color: conf.color, background: `${conf.color}10` }}>{conf.label}</span>
                  </div>
                  <div className="flex items-center gap-3">
                    <span className="text-[18px] font-bold text-white font-mono">%{(d.composite_score * 100)?.toFixed(0)}</span>
                    <svg className={`w-3 h-3 text-gray-600 transition-transform duration-200 ${isExpanded ? 'rotate-180' : ''}`} fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" /></svg>
                  </div>
                </div>
                {isExpanded && (
                  <div className="px-3 pb-3 border-t border-white/[0.03] animate-fadeIn">
                    {votes && Object.keys(votes).length > 0 && (
                      <div className="mt-2 space-y-1.5">
                        <p className="text-[9px] text-gray-600 tracking-[0.12em] font-mono">AJAN OYLARI</p>
                        <div className="grid grid-cols-2 gap-1">
                          {Object.entries(votes).map(([agent, vote]) => (
                            <div key={agent} className="flex items-center gap-1.5 text-[10px] bg-black/20 rounded-lg px-2 py-1 font-mono">
                              <span style={{ color: vote.vote === 'APPROVE' ? '#39ff14' : vote.vote === 'REJECT' ? '#ff3366' : '#8b95a5' }}>
                                {vote.vote === 'APPROVE' ? '✓' : vote.vote === 'REJECT' ? '✗' : '○'}
                              </span>
                              <span className="text-gray-400 capitalize">{agent}</span>
                              <span className="text-gray-600 ml-auto truncate max-w-[100px]">{vote.detail?.substring(0, 35)}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                    {d.risk_params && Object.keys(d.risk_params).length > 0 && (
                      <div className="flex items-center gap-3 text-[10px] text-gray-400 mt-2 bg-black/15 rounded-lg px-2.5 py-1.5 font-mono">
                        <span>${d.risk_params.position_size_usd?.toFixed(0)}</span>
                        <span style={{ color: '#b026ff' }}>x{d.risk_params.leverage}</span>
                        <span style={{ color: '#39ff14' }}>TP:{d.risk_params.take_profit ? `$${d.risk_params.take_profit?.toFixed(2)}` : '-'}</span>
                        <span style={{ color: '#ff3366' }}>SL:{d.risk_params.stop_loss ? `$${d.risk_params.stop_loss?.toFixed(2)}` : '-'}</span>
                        <span style={{ color: '#00f0ff' }}>RR:{d.risk_params.risk_reward_ratio?.toFixed(1)}</span>
                      </div>
                    )}
                  </div>
                )}
                {!isExpanded && d.risk_params && Object.keys(d.risk_params).length > 0 && (
                  <div className="flex items-center gap-3 text-[10px] text-gray-500 px-3 pb-2 font-mono">
                    <span>${d.risk_params.position_size_usd?.toFixed(0)}</span>
                    <span style={{ color: '#b026ff80' }}>x{d.risk_params.leverage}</span>
                    <span style={{ color: '#39ff1480' }}>TP:{d.risk_params.take_profit ? `$${d.risk_params.take_profit?.toFixed(2)}` : '-'}</span>
                    <span style={{ color: '#ff336680' }}>SL:{d.risk_params.stop_loss ? `$${d.risk_params.stop_loss?.toFixed(2)}` : '-'}</span>
                  </div>
                )}
              </div>
            )
          })}
        </div>
      ) : (
        <div className="text-center py-12">
          <div className="text-4xl mb-3 opacity-10">◇</div>
          <p className="text-[13px] text-gray-600">Aktif sinyal yok</p>
        </div>
      )}

      {rejected.length > 0 && (
        <div className="mt-4 pt-3 border-t border-white/[0.04]">
          <button onClick={() => setShowRejected(!showRejected)} className="flex items-center justify-between w-full mb-2">
            <p className="text-[10px] text-gray-600 tracking-[0.12em] font-mono">REDDEDİLEN ({rejected.length})</p>
            <span className="text-gray-700 text-[10px] font-mono">{showRejected ? '▲' : '▼'}</span>
          </button>
          {showRejected && (
            <div className="space-y-1.5">
              {rejected.slice(0, fullPage ? 30 : 5).map((d, i) => {
                const dir = dirConfig[d.direction] || dirConfig.hold
                const isExpanded = expandedRejected === `${d.symbol}-${i}`
                return (
                  <div key={i} className="rounded-lg border border-white/[0.03] overflow-hidden" style={{ background: '#ff336604' }}>
                    <div className="flex items-center justify-between p-2.5 cursor-pointer hover:bg-white/[0.01] transition-colors" onClick={() => setExpandedRejected(isExpanded ? null : `${d.symbol}-${i}`)}>
                      <div className="flex items-center gap-2">
                        <div className="w-1.5 h-1.5 rounded-full" style={{ background: '#ff3366' }} />
                        <span className="text-[12px] font-bold text-white font-mono">{d.symbol}</span>
                        <span className="text-[9px] font-mono" style={{ color: dir.color }}>{dir.label}</span>
                        <span className="text-[9px] text-gray-700 font-mono">%{((d.composite_score || 0) * 100).toFixed(1)}</span>
                      </div>
                      <svg className={`w-3 h-3 text-gray-700 transition-transform duration-200 ${isExpanded ? 'rotate-180' : ''}`} fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" /></svg>
                    </div>
                    {isExpanded && d.veto_reason && (
                      <div className="px-2.5 pb-2.5 border-t border-white/[0.03] pt-2 animate-fadeIn">
                        <p className="text-[9px] text-gray-700 tracking-wider font-mono mb-1">RED NEDENİ</p>
                        <p className="text-[10px] text-gray-500">{d.veto_reason}</p>
                      </div>
                    )}
                  </div>
                )
              })}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
