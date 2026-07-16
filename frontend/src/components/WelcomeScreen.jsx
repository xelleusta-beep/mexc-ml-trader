import { useState, useEffect } from 'react'

export default function WelcomeScreen({ onEnter }) {
  const [phase, setPhase] = useState(0)
  const [exiting, setExiting] = useState(false)

  useEffect(() => {
    const t1 = setTimeout(() => setPhase(1), 300)
    const t2 = setTimeout(() => setPhase(2), 800)
    const t3 = setTimeout(() => setPhase(3), 1400)
    return () => { clearTimeout(t1); clearTimeout(t2); clearTimeout(t3) }
  }, [])

  const handleEnter = () => {
    setExiting(true)
    setTimeout(onEnter, 600)
  }

  return (
    <div className={`fixed inset-0 z-[200] flex items-center justify-center transition-opacity duration-500 ${exiting ? 'opacity-0' : 'opacity-100'}`} style={{ background: '#030712' }}>
      {/* ANIMATED BACKGROUND */}
      <div className="absolute inset-0 overflow-hidden">
        {/* Gradient Orbs */}
        <div className="absolute top-1/4 left-1/4 w-[500px] h-[500px] rounded-full opacity-[0.07] animate-float" style={{ background: 'radial-gradient(circle, #00f0ff, transparent 70%)', filter: 'blur(80px)' }} />
        <div className="absolute bottom-1/4 right-1/4 w-[400px] h-[400px] rounded-full opacity-[0.05] animate-float" style={{ background: 'radial-gradient(circle, #b026ff, transparent 70%)', filter: 'blur(80px)', animationDelay: '2s' }} />
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] rounded-full opacity-[0.04]" style={{ background: 'radial-gradient(circle, #39ff14, transparent 70%)', filter: 'blur(100px)' }} />

        {/* Grid Lines */}
        <div className="absolute inset-0" style={{
          backgroundImage: 'linear-gradient(rgba(0,240,255,0.03) 1px, transparent 1px), linear-gradient(90deg, rgba(0,240,255,0.03) 1px, transparent 1px)',
          backgroundSize: '60px 60px',
        }} />

        {/* Floating Particles */}
        {[...Array(20)].map((_, i) => (
          <div key={i} className="absolute rounded-full" style={{
            width: `${1 + Math.random() * 2}px`,
            height: `${1 + Math.random() * 2}px`,
            left: `${Math.random() * 100}%`,
            top: `${Math.random() * 100}%`,
            background: ['#00f0ff', '#b026ff', '#39ff14', '#ff0080'][i % 4],
            opacity: 0.3 + Math.random() * 0.4,
            animation: `particleFloat ${10 + Math.random() * 15}s linear infinite`,
            animationDelay: `${Math.random() * 8}s`,
          }} />
        ))}

        {/* Horizontal Scan Line */}
        <div className="absolute left-0 right-0 h-px opacity-20" style={{
          background: 'linear-gradient(90deg, transparent, #00f0ff, transparent)',
          animation: 'scan 6s linear infinite',
        }} />
      </div>

      {/* CONTENT */}
      <div className="relative z-10 text-center px-6">
        {/* Logo */}
        <div className={`transition-all duration-700 ${phase >= 0 ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'}`}>
          <div className="w-20 h-20 mx-auto mb-8 rounded-2xl bg-gradient-to-br from-cyan-500/15 to-purple-500/15 border border-cyan-500/20 flex items-center justify-center animate-glowPulse" style={{ boxShadow: '0 0 40px rgba(0,240,255,0.15)' }}>
            <span className="font-orbitron text-3xl neon-cyan">M</span>
          </div>
        </div>

        {/* Title */}
        <div className={`transition-all duration-700 delay-200 ${phase >= 1 ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-6'}`}>
          <h1 className="font-orbitron text-4xl md:text-5xl font-bold tracking-[0.2em] mb-3">
            <span className="neon-cyan">MEXC</span>
            <span className="text-gray-500 mx-2">|</span>
            <span className="neon-purple">NEXUS</span>
          </h1>
          <div className="flex items-center justify-center gap-3 mb-2">
            <div className="h-px flex-1 max-w-[80px] bg-gradient-to-r from-transparent to-cyan-500/30" />
            <span className="text-[10px] text-cyan-400/50 tracking-[0.3em] font-mono">MULTI-AGENT NEURAL NETWORK</span>
            <div className="h-px flex-1 max-w-[80px] bg-gradient-to-l from-transparent to-cyan-500/30" />
          </div>
        </div>

        {/* Subtitle */}
        <div className={`transition-all duration-700 delay-400 ${phase >= 2 ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-6'}`}>
          <p className="text-gray-400 text-sm md:text-base max-w-md mx-auto mb-8 leading-relaxed">
            6 yapay zeka ajanı ile kripto piyasalarını analiz eden,<br />
            <span className="text-cyan-400">gerçek zamanlı</span> işlem yapan otonom sistem.
          </p>

          {/* Agent Icons */}
          <div className="flex items-center justify-center gap-4 mb-10">
            {[
              { icon: '◎', color: '#00f0ff', label: 'Scanner' },
              { icon: '◇', color: '#b026ff', label: 'Technical' },
              { icon: '◈', color: '#ff0080', label: 'Sentiment' },
              { icon: '⬡', color: '#ff0040', label: 'Risk' },
              { icon: '▣', color: '#39ff14', label: 'Portfolio' },
              { icon: '★', color: '#ffd700', label: 'Patron' },
            ].map((a, i) => (
              <div key={i} className="flex flex-col items-center gap-1.5 group" style={{ animationDelay: `${600 + i * 100}ms` }}>
                <div className="w-10 h-10 rounded-xl border flex items-center justify-center transition-all duration-300 group-hover:scale-110" style={{
                  borderColor: `${a.color}25`,
                  background: `${a.color}08`,
                  boxShadow: `0 0 15px ${a.color}10`,
                }}>
                  <span style={{ color: a.color, fontSize: '16px' }}>{a.icon}</span>
                </div>
                <span className="text-[9px] text-gray-600 font-mono tracking-wider">{a.label}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Enter Button */}
        <div className={`transition-all duration-700 delay-500 ${phase >= 3 ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-6'}`}>
          <button
            onClick={handleEnter}
            className="group relative px-10 py-4 rounded-xl border border-cyan-500/20 bg-cyan-500/5 text-cyan-400 font-orbitron text-sm font-bold tracking-[0.2em] transition-all duration-300 hover:bg-cyan-500/10 hover:border-cyan-500/40 hover:shadow-[0_0_30px_rgba(0,240,255,0.15)] hover:-translate-y-0.5"
          >
            <span className="relative z-10">SİSTEME GİR</span>
            <div className="absolute inset-0 rounded-xl bg-gradient-to-r from-cyan-500/0 via-cyan-500/5 to-cyan-500/0 opacity-0 group-hover:opacity-100 transition-opacity" />
          </button>

          <div className="mt-8 flex items-center justify-center gap-2 text-[10px] text-gray-600 font-mono">
            <div className="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse-neon" />
            <span>Sistem aktif • Canlı veri</span>
          </div>
        </div>
      </div>
    </div>
  )
}
