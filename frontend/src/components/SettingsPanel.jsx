import { useState, useEffect } from 'react'

export default function SettingsPanel({ isRunning = false, onStart = () => {}, onStop = () => {} }) {
  const [settings, setSettings] = useState({ cycle_interval: 300, min_confidence: 0.15, max_positions: 5, risk_per_trade: 1.0, daily_risk: 5.0, leverage_max: 10 })
  const [saving, setSaving] = useState(false)
  const [saved, setSaved] = useState(false)

  useEffect(() => {
    fetch('/api/config').then(r => r.json()).then(data => { if (data && Object.keys(data).length > 0) setSettings(s => ({ ...s, ...data })) }).catch(() => {})
  }, [])

  const handleSave = async () => {
    setSaving(true); setSaved(false)
    try {
      const resp = await fetch('/api/config', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(settings) })
      if (resp.ok) { setSaved(true); setTimeout(() => setSaved(false), 2000) }
    } catch {}
    setSaving(false)
  }

  const Slider = ({ item }) => (
    <div>
      <div className="flex justify-between mb-1.5">
        <label className="text-[11px] text-gray-400 font-mono">{item.label}</label>
        <span className="text-[13px] font-bold font-mono" style={{ color: item.color }}>{settings[item.key]}{item.suffix}</span>
      </div>
      <input type="range" min={item.min} max={item.max} step={item.step} value={settings[item.key]}
        onChange={e => setSettings({ ...settings, [item.key]: +e.target.value })}
        className="w-full h-1 rounded-full appearance-none cursor-pointer" style={{ background: `linear-gradient(90deg, ${item.color}40, ${item.color}15)`, accentColor: item.color }} />
      <div className="flex justify-between text-[9px] text-gray-700 mt-1 font-mono"><span>{item.min}{item.suffix}</span><span>{item.max}{item.suffix}</span></div>
    </div>
  )

  return (
    <div className="cinematic-border rounded-2xl p-5 h-full" style={{ background: 'linear-gradient(145deg, rgba(6,10,22,0.95) 0%, rgba(22,10,18,0.9) 100%)' }}>
      <div className="flex items-center justify-between mb-5">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl flex items-center justify-center" style={{ background: '#ff008008', border: '1px solid #ff008015' }}>
            <span style={{ color: '#ff0080', fontSize: '13px' }}>⚙</span>
          </div>
          <span className="text-[11px] font-bold tracking-[0.15em] text-gray-400 font-mono">SİSTEM AYARLARI</span>
        </div>
        {saved && <span className="text-[10px] font-bold tracking-wider font-mono" style={{ color: '#39ff14' }}>● KAYDEDİLDİ</span>}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        <div className="space-y-4">
          <div className="rounded-xl p-4 border border-white/[0.04]" style={{ background: '#00f0ff03' }}>
            <p className="text-[9px] text-gray-600 tracking-[0.12em] font-mono mb-3">SİSTEM KONTROLÜ</p>
            <div className="flex gap-3">
              <button onClick={onStart} disabled={isRunning} className="flex-1 py-3 rounded-xl text-[11px] font-bold tracking-wider font-mono transition-all duration-300"
                style={{ background: isRunning ? 'rgba(255,255,255,0.02)' : '#39ff1408', border: `1px solid ${isRunning ? '#ffffff06' : '#39ff1420'}`, color: isRunning ? '#8b95a550' : '#39ff14' }}>
                {isRunning ? '● AKTİF' : '▶ BAŞLAT'}
              </button>
              <button onClick={onStop} disabled={!isRunning} className="flex-1 py-3 rounded-xl text-[11px] font-bold tracking-wider font-mono transition-all duration-300"
                style={{ background: !isRunning ? 'rgba(255,255,255,0.02)' : '#ff336608', border: `1px solid ${!isRunning ? '#ffffff06' : '#ff336620'}`, color: !isRunning ? '#8b95a550' : '#ff3366' }}>
                {!isRunning ? '■ DURDURULDU' : '■ DURDUR'}
              </button>
            </div>
          </div>

          <div className="rounded-xl p-4 border border-white/[0.04]" style={{ background: '#b026ff03' }}>
            <p className="text-[9px] text-gray-600 tracking-[0.12em] font-mono mb-3">TARAMA AYARLARI</p>
            <div className="space-y-4">
              <Slider item={{ key: 'cycle_interval', label: 'DÖNGÜ SÜRESİ', min: 60, max: 600, step: 30, suffix: 's', color: '#b026ff' }} />
              <Slider item={{ key: 'min_confidence', label: 'MİN GÜVEN', min: 0.05, max: 0.95, step: 0.05, suffix: '', color: '#b026ff' }} />
            </div>
          </div>

          <div className="rounded-xl p-4 border border-white/[0.04]" style={{ background: '#00f0ff03' }}>
            <p className="text-[9px] text-gray-600 tracking-[0.12em] font-mono mb-3">TELEGRAM TEST</p>
            <button onClick={async () => { try { await fetch('/api/test/telegram'); alert('Test mesajı gönderildi!') } catch { alert('Hata!') } }}
              className="w-full py-2.5 rounded-xl text-[11px] font-bold tracking-wider font-mono transition-all duration-300"
              style={{ background: '#00f0ff06', border: '1px solid #00f0ff15', color: '#00f0ff' }}>
              📱 TELEGRAM TEST
            </button>
          </div>
        </div>

        <div className="space-y-4">
          <div className="rounded-xl p-4 border border-white/[0.04]" style={{ background: '#ff336603' }}>
            <p className="text-[9px] text-gray-600 tracking-[0.12em] font-mono mb-3">RİSK YÖNETİMİ</p>
            <div className="space-y-4">
              <Slider item={{ key: 'risk_per_trade', label: 'İŞLEM BAŞINA RİSK (KASA)', min: 0.5, max: 5, step: 0.5, suffix: '%', color: '#ff3366' }} />
              <Slider item={{ key: 'daily_risk', label: 'GÜNLÜK RİSK', min: 1, max: 10, step: 0.5, suffix: '%', color: '#ff6b35' }} />
              <Slider item={{ key: 'leverage_max', label: 'MAKS KALDIRAÇ', min: 1, max: 50, step: 1, suffix: 'x', color: '#ffd700' }} />
              <Slider item={{ key: 'max_positions', label: 'MAKS POZİSYON', min: 1, max: 10, step: 1, suffix: '', color: '#39ff14' }} />
            </div>
          </div>

          <button onClick={handleSave} disabled={saving} className="w-full py-3 rounded-xl text-[11px] font-bold tracking-[0.15em] font-mono transition-all duration-300"
            style={{ background: saving ? 'rgba(255,255,255,0.02)' : saved ? '#39ff1408' : '#00f0ff08', border: `1px solid ${saving ? '#ffffff06' : saved ? '#39ff1420' : '#00f0ff20'}`, color: saving ? '#8b95a550' : saved ? '#39ff14' : '#00f0ff' }}>
            {saving ? '● KAYDEDİLİYOR...' : saved ? '● BAŞARILI' : '◆ AYARLARI KAYDET'}
          </button>
        </div>
      </div>
    </div>
  )
}
