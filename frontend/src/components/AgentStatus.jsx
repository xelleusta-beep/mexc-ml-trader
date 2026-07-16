import { useState } from 'react'

const agentConfig = {
  Scanner: { icon: '◎', color: '#00f0ff', role: 'PİYASA TARAYICI', desc: '809 sembolü tarar, hot score hesaplar' },
  Technical: { icon: '◇', color: '#b026ff', role: 'TEKNİK ANALİZ', desc: '5 timeframede teknik göstergeleri analiz eder' },
  Sentiment: { icon: '◈', color: '#ff0080', role: 'DUYGU ANALİZİ', desc: 'Fear & Greed + haber sentiment analizi' },
  RiskManager: { icon: '⬡', color: '#ff3366', role: 'RİSK YÖNETİMİ', desc: 'Kelly criterion, SL/TP, pozisyon büyüklüğü' },
  PortfolioManager: { icon: '▣', color: '#39ff14', role: 'PORTFÖY YÖNETİMİ', desc: 'Pozisyon korelasyonu, sermaye yönetimi' },
  Patron: { icon: '★', color: '#ffd700', role: 'PATRON AJAN', desc: 'Nihai karar, tüm ajanların oylarını değerlendirir' },
}

const statusMap = {
  idle: { label: 'BEKLEMEDE', color: 'text-gray-500', dot: 'bg-gray-500' },
  running: { label: 'DÜŞÜNÜYOR', color: 'text-yellow-400', dot: 'bg-yellow-400' },
  ready: { label: 'AKTİF', color: 'text-green-400', dot: 'bg-green-400' },
  error: { label: 'HATA', color: 'text-red-400', dot: 'bg-red-400' },
}

export default function AgentStatus({ status }) {
  const agents = status?.agents || {}
  const isRunning = status?.running
  const [expanded, setExpanded] = useState(null)

  return (
    <div className="cinematic-border rounded-2xl p-5 h-full" style={{ background: 'linear-gradient(145deg, rgba(6,10,22,0.95) 0%, rgba(10,14,28,0.9) 100%)' }}>
      <div className="flex items-center justify-between mb-5">
        <div className="flex items-center gap-3">
          <div className="relative">
            <div className={`w-2.5 h-2.5 rounded-full ${isRunning ? 'bg-green-400' : 'bg-gray-600'}`} />
            {isRunning && <div className="absolute inset-0 w-2.5 h-2.5 rounded-full bg-green-400 animate-ping opacity-50" />}
          </div>
          <span className="text-[11px] font-bold tracking-[0.15em] text-gray-400 font-mono">SİSTEM DURUMU</span>
        </div>
        <div className={`flex items-center gap-2 px-3 py-1.5 rounded-lg border transition-all duration-500 ${
          isRunning ? 'border-green-500/20 bg-green-500/5' : 'border-gray-700/50 bg-gray-800/30'
        }`}>
          <span className={`text-[10px] font-bold tracking-wider font-mono ${isRunning ? 'text-green-400' : 'text-gray-500'}`}>
            {isRunning ? '● LIVE' : '○ IDLE'}
          </span>
        </div>
      </div>

      <div className="space-y-2">
        {Object.entries(agents).map(([name, agent], idx) => {
          const cfg = agentConfig[name] || { icon: '●', color: '#00f0ff', role: name, desc: '' }
          const st = statusMap[agent.status] || statusMap.idle
          const isExpanded = expanded === name

          return (
            <div
              key={name}
              className="rounded-xl border border-white/[0.04] transition-all duration-300 hover:border-white/[0.08] overflow-hidden"
              style={{
                background: isExpanded ? `linear-gradient(135deg, ${cfg.color}06, transparent)` : 'rgba(255,255,255,0.01)',
                boxShadow: isExpanded ? `0 0 20px ${cfg.color}08` : 'none',
              }}
            >
              <div
                className="p-3 cursor-pointer flex items-center gap-3"
                onClick={() => setExpanded(isExpanded ? null : name)}
              >
                <div className="w-9 h-9 rounded-lg flex items-center justify-center transition-transform duration-300" style={{
                  background: `linear-gradient(135deg, ${cfg.color}12, ${cfg.color}05)`,
                  border: `1px solid ${cfg.color}20`,
                  transform: isExpanded ? 'scale(1.1)' : 'scale(1)',
                }}>
                  <span style={{ color: cfg.color, fontSize: '14px' }}>{cfg.icon}</span>
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="text-[13px] font-bold text-white tracking-wide">{name}</span>
                    <div className={`w-1.5 h-1.5 rounded-full ${st.dot}`} />
                    <span className={`text-[9px] font-bold tracking-wider font-mono ${st.color}`}>{st.label}</span>
                  </div>
                  <span className="text-[10px] text-gray-600 tracking-wider">{cfg.role}</span>
                </div>
                <div className="text-right flex-shrink-0">
                  <div className="text-[11px] text-gray-500 font-mono">{agent.signal_count || 0}<span className="text-gray-700 ml-0.5">sig</span></div>
                  <div className="text-[9px] text-gray-700 font-mono">{agent.elapsed_seconds || 0}s</div>
                </div>
                <svg className={`w-3 h-3 text-gray-600 transition-transform duration-300 ${isExpanded ? 'rotate-180' : ''}`} fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" /></svg>
              </div>

              {isExpanded && (
                <div className="px-3 pb-3 border-t border-white/[0.03] animate-fadeIn">
                  <p className="text-[10px] text-gray-600 mt-2 mb-2 italic">{cfg.desc}</p>
                  {agent.thinking && agent.thinking.length > 0 && (
                    <div className="bg-black/30 rounded-lg p-3 border border-white/[0.03]">
                      <p className="text-[9px] text-gray-600 tracking-[0.15em] font-mono mb-2">DÜŞÜNCE SÜRECİ</p>
                      <div className="space-y-1">
                        {agent.thinking.map((step, i) => (
                          <div key={i} className="flex items-start gap-2 text-[11px] py-0.5">
                            <span className="text-gray-700 font-mono mt-0.5">›</span>
                            <span className="text-gray-400 leading-relaxed">{step}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                  {agent.last_error && (
                    <div className="mt-2 text-[11px] text-red-400/70 bg-red-500/5 rounded-lg p-2.5 border border-red-500/10 font-mono">
                      {agent.last_error}
                    </div>
                  )}
                </div>
              )}
            </div>
          )
        })}
      </div>

      <div className="mt-5 pt-4 border-t border-white/[0.04]">
        <div className="grid grid-cols-2 gap-2">
          {[
            { label: 'DÖNGÜ', value: status?.cycle_count || 0, color: '#00f0ff' },
            { label: 'SÜRE', value: `${(status?.last_cycle_time || 0).toFixed(1)}s`, color: '#b026ff' },
            { label: 'İŞLEM', value: status?.trade_count || 0, color: '#39ff14' },
            { label: 'SERMAYE', value: `$${(status?.total_equity || 0).toFixed(0)}`, color: '#00f0ff' },
          ].map((item, i) => (
            <div key={i} className="rounded-lg px-3 py-2.5 border border-white/[0.03] hover:border-white/[0.06] transition-colors" style={{ background: `${item.color}03` }}>
              <p className="text-[9px] text-gray-600 tracking-wider font-mono mb-0.5">{item.label}</p>
              <p className="text-[13px] font-bold font-mono" style={{ color: item.color }}>{item.value}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
