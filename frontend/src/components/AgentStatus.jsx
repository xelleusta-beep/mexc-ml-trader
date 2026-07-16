import { useState } from 'react'

const agentConfig = {
  Scanner: { icon: '◎', color: 'cyan', role: 'PİYASA TARAYICI', desc: '809 sembolü tarar, hot score hesaplar', gradient: 'from-cyan-500/20 to-blue-500/10' },
  Technical: { icon: '◇', color: 'purple', role: 'TEKNİK ANALİZ', desc: '5 timeframede teknik göstergeleri analiz eder', gradient: 'from-purple-500/20 to-pink-500/10' },
  Sentiment: { icon: '◈', color: 'pink', role: 'DUYGU ANALİZİ', desc: 'Fear & Greed + haber sentiment analizi', gradient: 'from-pink-500/20 to-rose-500/10' },
  RiskManager: { icon: '⬡', color: 'red', role: 'RİSK YÖNETİMİ', desc: 'Kelly criterion, SL/TP, pozisyon büyüklüğü', gradient: 'from-red-500/20 to-orange-500/10' },
  PortfolioManager: { icon: '▣', color: 'green', role: 'PORTFÖY YÖNETİMİ', desc: 'Pozisyon korelasyonu, sermaye yönetimi', gradient: 'from-green-500/20 to-emerald-500/10' },
  Patron: { icon: '★', color: 'cyan', role: 'PATRON AJAN', desc: 'Nihai karar, tüm ajanların oylarını değerlendirir', gradient: 'from-cyan-500/20 to-violet-500/10' },
}

const statusMap = {
  idle: { label: 'BEKLEMEDE', color: 'text-gray-500', dot: 'bg-gray-500', anim: '' },
  running: { label: 'DÜŞÜNÜYOR', color: 'text-yellow-400', dot: 'bg-yellow-400', anim: 'animate-pulse-neon' },
  ready: { label: 'AKTİF', color: 'text-green-400', dot: 'bg-green-400', anim: '' },
  error: { label: 'HATA', color: 'text-red-400', dot: 'bg-red-400', anim: '' },
}

const colorMap = {
  cyan: { text: 'neon-cyan', border: 'border-cyan-500/20', bg: 'bg-cyan-500/5', glow: 'rgba(0,240,255,0.08)' },
  purple: { text: 'neon-purple', border: 'border-purple-500/20', bg: 'bg-purple-500/5', glow: 'rgba(176,38,255,0.08)' },
  pink: { text: 'neon-pink', border: 'border-pink-500/20', bg: 'bg-pink-500/5', glow: 'rgba(255,0,128,0.08)' },
  red: { text: 'neon-red', border: 'border-red-500/20', bg: 'bg-red-500/5', glow: 'rgba(255,0,64,0.08)' },
  green: { text: 'neon-green', border: 'border-green-500/20', bg: 'bg-green-500/5', glow: 'rgba(57,255,20,0.08)' },
}

export default function AgentStatus({ status }) {
  const agents = status?.agents || {}
  const isRunning = status?.running
  const [expanded, setExpanded] = useState(null)

  return (
    <div className="glass-panel glass-panel-cyan p-4 h-full corner-deco scanline animate-fadeIn">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <div className={`w-2.5 h-2.5 rounded-full ${isRunning ? 'bg-green-400 animate-pulse-neon' : 'bg-gray-600'}`} />
          <span className="text-label text-cyan-400/70">SİSTEM DURUMU</span>
        </div>
        <div className={`flex items-center gap-2 px-3 py-1.5 rounded-md border transition-all ${
          isRunning ? 'border-green-500/30 bg-green-500/5 animate-glowPulse' : 'border-gray-700 bg-gray-800/50'
        }`}>
          <span className={`text-xs font-bold tracking-wider ${isRunning ? 'text-green-400' : 'text-gray-500'}`}>
            {isRunning ? '● AKTİF' : '○ DURDURULDU'}
          </span>
        </div>
      </div>

      <div className="space-y-1.5">
        {Object.entries(agents).map(([name, agent], idx) => {
          const cfg = agentConfig[name] || { icon: '●', color: 'cyan', role: name, desc: '', gradient: 'from-cyan-500/20 to-blue-500/10' }
          const st = statusMap[agent.status] || statusMap.idle
          const cc = colorMap[cfg.color] || colorMap.cyan
          const isExpanded = expanded === name

          return (
            <div
              key={name}
              className={`agent-card glass-panel ${cc.bg} ${cc.border} border rounded-xl`}
              style={{ animationDelay: `${idx * 60}ms`, boxShadow: isExpanded ? `0 0 20px ${cc.glow}` : 'none' }}
            >
              <div
                className="p-3 cursor-pointer flex items-center gap-3"
                onClick={() => setExpanded(isExpanded ? null : name)}
              >
                <div className={`w-10 h-10 rounded-lg bg-gradient-to-br ${cfg.gradient} border ${cc.border} flex items-center justify-center text-lg transition-transform ${isExpanded ? 'scale-110' : ''}`}>
                  <span className={cc.text}>{cfg.icon}</span>
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-bold text-white tracking-wide">{name}</span>
                    <span className={`status-dot ${st.dot} ${st.anim}`} style={{ width: 6, height: 6 }} />
                    <span className={`text-[10px] font-semibold ${st.color}`}>{st.label}</span>
                  </div>
                  <span className="text-[11px] text-gray-500 tracking-wide">{cfg.role}</span>
                </div>
                <div className="text-right flex-shrink-0">
                  <div className="text-xs text-gray-500 font-mono">{agent.signal_count || 0} <span className="text-gray-600">sinyal</span></div>
                  <div className="text-[10px] text-gray-600 font-mono">{agent.elapsed_seconds || 0}s</div>
                </div>
                <span className="text-gray-600 text-xs transition-transform duration-200" style={{ transform: isExpanded ? 'rotate(180deg)' : 'rotate(0)' }}>▼</span>
              </div>

              {isExpanded && (
                <div className="px-3 pb-3 border-t border-white/5 animate-fadeIn">
                  <p className="text-[11px] text-gray-500 mt-2 mb-1 italic">{cfg.desc}</p>
                  {agent.thinking && agent.thinking.length > 0 && (
                    <div className="mt-2 space-y-1">
                      <p className="text-[10px] text-cyan-400/50 tracking-[0.15em] font-semibold">DÜŞÜNCE SÜRECİ</p>
                      <div className="bg-black/20 rounded-lg p-2.5 border border-cyan-500/5">
                        {agent.thinking.map((step, i) => (
                          <div key={i} className="flex items-start gap-2 text-xs py-0.5">
                            <span className="text-cyan-500/40 mt-0.5 font-mono">▸</span>
                            <span className="text-gray-400 leading-relaxed">{step}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                  {agent.last_error && (
                    <div className="mt-2 text-xs text-red-400/60 bg-red-500/5 rounded-lg p-2.5 border border-red-500/10">
                      ⚠ {agent.last_error}
                    </div>
                  )}
                </div>
              )}
            </div>
          )
        })}
      </div>

      <div className="mt-4 pt-3 border-t border-cyan-500/10">
        <div className="grid grid-cols-2 gap-2">
          {[
            { label: 'DÖNGÜ', value: status?.cycle_count || 0, color: 'text-cyan-400' },
            { label: 'SÜRE', value: `${(status?.last_cycle_time || 0).toFixed(1)}s`, color: 'text-purple-400' },
            { label: 'İŞLEM', value: status?.trade_count || 0, color: 'text-green-400' },
            { label: 'SERMAYE', value: `$${(status?.total_equity || 0).toLocaleString()}`, color: 'text-cyan-400' },
          ].map((item, i) => (
            <div key={i} className="bg-black/30 rounded-lg px-2.5 py-2.5 border border-white/[0.03] hover:border-cyan-500/10 transition-colors">
              <p className="text-label text-gray-500" style={{ fontSize: '10px' }}>{item.label}</p>
              <p className={`text-base font-bold ${item.color} font-mono`}>{item.value}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
