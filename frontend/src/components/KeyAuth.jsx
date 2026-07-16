import { useState } from 'react'

export default function KeyAuth({ onAuthenticated }) {
  const [key, setKey] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true); setError('')
    try {
      const resp = await fetch('/api/auth/verify', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ key }) })
      if (resp.ok) { localStorage.setItem('mexc_auth', 'true'); onAuthenticated() }
      else setError('Geçersiz anahtar!')
    } catch { setError('Bağlantı hatası!') }
    setLoading(false)
  }

  return (
    <div className="min-h-screen flex items-center justify-center p-4" style={{ background: '#030712' }}>
      <div className="cinematic-border rounded-2xl p-8 w-full max-w-md" style={{ background: 'linear-gradient(145deg, rgba(6,10,22,0.98) 0%, rgba(10,14,28,0.95) 100%)' }}>
        <div className="text-center mb-8">
          <div className="w-16 h-16 mx-auto mb-5 rounded-2xl flex items-center justify-center" style={{ background: 'linear-gradient(135deg, #00f0ff12, #b026ff08)', border: '1px solid #00f0ff20', boxShadow: '0 0 30px #00f0ff10' }}>
            <span className="font-orbitron text-2xl" style={{ color: '#00f0ff' }}>M</span>
          </div>
          <h1 className="font-orbitron text-xl font-bold tracking-[0.2em] mb-2">
            <span style={{ color: '#00f0ff' }}>MEXC</span>
            <span className="mx-2 text-gray-700">|</span>
            <span style={{ color: '#b026ff' }}>NEXUS</span>
          </h1>
          <p className="text-[10px] text-gray-600 tracking-[0.2em] font-mono">MULTI-AGENT TRADING SYSTEM</p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="text-[10px] text-gray-600 tracking-[0.12em] font-mono block mb-2">ERİŞİM ANAHTARI</label>
            <input type="password" value={key} onChange={(e) => setKey(e.target.value)} placeholder="••••••••" autoFocus
              className="w-full rounded-xl px-4 py-3 text-center text-lg tracking-[0.3em] font-mono focus:outline-none transition-all duration-300"
              style={{ background: 'rgba(0,0,0,0.3)', border: '1px solid #00f0ff15', color: '#00f0ff' }} />
          </div>
          {error && <div className="rounded-xl p-3 text-center" style={{ background: '#ff336608', border: '1px solid #ff336620' }}><span className="text-[12px] font-mono" style={{ color: '#ff3366' }}>{error}</span></div>}
          <button type="submit" disabled={loading || !key} className="w-full py-3 rounded-xl text-[11px] font-bold tracking-[0.15em] font-mono transition-all duration-300"
            style={{ background: loading || !key ? 'rgba(255,255,255,0.02)' : '#00f0ff08', border: `1px solid ${loading || !key ? '#ffffff06' : '#00f0ff20'}`, color: loading || !key ? '#8b95a550' : '#00f0ff' }}>
            {loading ? '● DOĞRULANIIYOR...' : '◆ GİRİŞ YAP'}
          </button>
        </form>
        <p className="mt-6 text-center text-[9px] text-gray-700 tracking-[0.15em] font-mono">GÜVENLİ ERİŞİM GEREKLİDİR</p>
      </div>
    </div>
  )
}
