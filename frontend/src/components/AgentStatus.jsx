import { useState } from 'react'

const agentConfig = {
  Scanner: { icon: '◎', color: 'cyan', role: 'PİYASA TARAYICI', desc: '809 sembolü tarar, hot score hesaplar' },
  Technical: { icon: '◇', color: 'purple', role: 'TEKNİK ANALİZ', desc: '5 timeframede teknik göstergeleri analiz eder' },
  Sentiment: { icon: '◈', color: 'pink', role: 'DUYGU ANALİZİ', desc: 'Fear & Greed + haber sentiment analizi' },
  RiskManager: { icon: '⬡', color: 'red', role: 'RİSK YÖNETİMİ', desc: 'Kelly criterion, SL/TP, pozisyon büyüklüğü' },
  PortfolioManager: { icon: '▣', color: 'green', role: 'PORTFÖY YÖNETİMİ', desc: 'Pozisyon korelasyonu, sermaye yönetimi' },
  Patron: { icon: '★', color: 'cyan', role: 'PATRON AJAN', desc: 'Nihai karar, tüm ajanların oylarını değerlendirir' },
}

const statusMap = {
  idle: { label: 'BEKLEMEDE', color: 'text-gray-500', dot: 'bg-gray-500', anim: '' },
  running: { label: 'DÜŞÜNÜYOR', color: 'text-yellow-400', dot: 'bg-yellow-400', anim: 'animate-pulse-neon' },
  ready: { label: 'AKTİF', color: 'text-green-400', dot: 'bg-green-400', anim: '' },
  error: { label: 'HATA', color: 'text-red-400', dot: 'bg-red-400', anim: '' },
}

const colorMap = {
  cyan: { text: 'neon-cyan', border: 'border-cyan-500/20', bg: 'bg-cyan-500/5' },
  purple: { text: 'neon-purple', border: 'border-purple-500/20', bg: 'bg-purple-500/5' },
  pink: { text: 'neon-pink', border: 'border-pink-500/20', bg: 'bg-pink-500/5' },
  red: { text: 'neon-red', border: 'border-red-500/20', bg: 'bg-red-500/5' },
  green: { text: 'neon-green', border: 'border-green-500/20', bg: 'bg-green-500/5' },
}

export default function AgentStatus({ status }) {
  const agents = status?.agents || {}
  const isRunning = status?.running
  const [expanded, setExpanded] = useState(null)

  return (
    <div className="glass-panel glass-panel-cyan p-4 h-full corner-deco scanline">
      <div className="flex items-center justify-between mb-4">
        <span className="text-label text-cyan-400/70">SİSTEM DURUMU</span>
        <div className={`flex items-center gap-2 px-3 py-1 rounded-sm border ${
          isRunning ? 'border-green-500/30 bg-green-500/5' : 'border-gray-700 bg-gray-800/50'
        }`}>
          <div className={`w-2 h-2 rounded-full ${isRunning ? 'bg-green-400 animate-pulse-neon' : 'bg-gray-600'}`} />
          <span className={`text-sm font-semibold ${isRunning ? 'text-green-400' : 'text-gray-500'}`}>
            {isRunning ? 'AKTİF' : 'DURDURULDU'}
          </span>
        </div>
      </div>

      <div className="space-y-2">
        {Object.entries(agents).map(([name, agent]) => {
          const cfg = agentConfig[name] || { icon: '●', color: 'cyan', role: name, desc: '' }
          const st = statusMap[agent.status] || statusMap.idle
          const cc = colorMap[cfg.color] || colorMap.cyan
          const isExpanded = expanded === name

          return (
            <div key={name} className={`glass-panel ${cc.bg} ${cc.border} border rounded-lg transition-all`}>
              <div
                className="p-3 cursor-pointer flex items-center gap-3"
                onClick={() => setExpanded(isExpanded ? null : name)}
              >
                <div className={`w-9 h-9 rounded-md ${cc.bg} border ${cc.border} flex items-center justify-center text-base`}>
                  <span className={cc.text}>{cfg.icon}</span>
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-bold text-white">{name}</span>
                    <span className={`status-dot ${st.dot} ${st.anim}`} style={{ width: 6, height: 6 }} />
                    <span className={`text-xs ${st.color}`}>{st.label}</span>
                  </div>
                  <span className="text-xs text-gray-500">{cfg.role}</span>
                </div>
                <div className="text-right">
                  <div className="text-xs text-gray-600">{agent.signal_count || 0} sinyal</div>
                  <div className="text-[10px] text-gray-600">{agent.elapsed_seconds || 0}s</div>
                </div>
                <span className="text-gray-600 text-xs">{isExpanded ? '▲' : '▼'}</span>
              </div>

              {isExpanded && (
                <div className="px-3 pb-3 border-t border-white/5">
                  <p className="text-xs text-gray-500 mt-2 mb-1">{cfg.desc}</p>
                  {agent.thinking && agent.thinking.length > 0 && (
                    <div className="mt-2 space-y-1">
                      <p className="text-[10px] text-cyan-400/50 tracking-wider">DÜŞÜNCE SÜRECİ:</p>
                      {agent.thinking.map((step, i) => (
                        <div key={i} className="flex items-start gap-2 text-xs">
                          <span className="text-cyan-500/40 mt-0.5">▸</span>
                          <span className="text-gray-400">{step}</span>
                        </div>
                      ))}
                    </div>
                  )}
                  {agent.last_error && (
                    <div className="mt-2 text-xs text-red-400/60 bg-red-500/5 rounded p-2">
                      Hata: {agent.last_error}
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
            <div key={i} className="bg-black/30 rounded px-2 py-2">
              <p className="text-label text-gray-500" style={{ fontSize: '10px' }}>{item.label}</p>
              <p className={`text-base font-bold ${item.color}`}>{item.value}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
