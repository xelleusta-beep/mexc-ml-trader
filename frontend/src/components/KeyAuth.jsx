import { useState } from 'react'

export default function KeyAuth({ onAuthenticated }) {
  const [key, setKey] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError('')
    try {
      const resp = await fetch('/api/auth/verify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ key }),
      })
      if (resp.ok) {
        localStorage.setItem('mexc_auth', 'true')
        onAuthenticated()
      } else {
        setError('Geçersiz anahtar!')
      }
    } catch (err) {
      setError('Bağlantı hatası!')
    }
    setLoading(false)
  }

  return (
    <div className="min-h-screen bg-[#030712] grid-bg hex-pattern flex items-center justify-center p-4">
      <div className="glass-panel glass-panel-cyan p-8 w-full max-w-md corner-deco">
        <div className="text-center mb-8">
          <div className="w-16 h-16 mx-auto mb-4 rounded-xl bg-gradient-to-br from-cyan-500/20 to-purple-500/20 border border-cyan-500/30 flex items-center justify-center">
            <span className="font-orbitron text-2xl neon-cyan">M</span>
          </div>
          <h1 className="font-orbitron text-xl font-bold tracking-widest mb-2">
            <span className="neon-cyan">MEXC</span>
            <span className="neon-purple ml-2">NEXUS</span>
          </h1>
          <p className="text-sm text-cyan-500/50 tracking-wider">MULTI-AGENT TRADING SYSTEM</p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="text-label text-cyan-400/70 block mb-2">ERİŞİM ANAHTARI</label>
            <input
              type="password"
              value={key}
              onChange={(e) => setKey(e.target.value)}
              placeholder="••••••••"
              className="w-full bg-black/50 border border-cyan-500/20 rounded-lg px-4 py-3 text-cyan-400 font-mono text-center text-lg tracking-[0.3em] focus:outline-none focus:border-cyan-400 transition-colors"
              autoFocus
            />
          </div>

          {error && (
            <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-3 text-center">
              <span className="text-red-400 text-sm">{error}</span>
            </div>
          )}

          <button
            type="submit"
            disabled={loading || !key}
            className={`w-full py-3 rounded-lg text-sm font-bold tracking-wider transition-all ${
              loading || !key
                ? 'bg-gray-800 text-gray-600 cursor-not-allowed border border-gray-700'
                : 'neon-btn bg-cyan-500/10 text-cyan-400 border border-cyan-500/30 hover:bg-cyan-500/20'
            }`}
          >
            {loading ? '● DOĞRULANIIYOR...' : '◆ GİRİŞ YAP'}
          </button>
        </form>

        <div className="mt-6 text-center">
          <p className="text-xs text-gray-600 tracking-wider">GÜVENLİ ERİŞİM GEREKLİDİR</p>
        </div>
      </div>
    </div>
  )
}
