import { useState, useEffect } from 'react'

export default function SettingsPanel({ isRunning = false, onStart = () => {}, onStop = () => {} }) {
  const [settings, setSettings] = useState({
    cycle_interval: 300,
    min_confidence: 0.15,
    max_positions: 5,
    risk_per_trade: 1.0,
    daily_risk: 5.0,
    leverage_max: 10,
  })
  const [saving, setSaving] = useState(false)
  const [saved, setSaved] = useState(false)

  useEffect(() => {
    fetch('/api/config')
      .then(r => r.json())
      .then(data => {
        if (data && Object.keys(data).length > 0) {
          setSettings(s => ({ ...s, ...data }))
        }
      })
      .catch(() => {})
  }, [])

  const handleSave = async () => {
    setSaving(true)
    setSaved(false)
    try {
      const resp = await fetch('/api/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(settings),
      })
      if (resp.ok) {
        setSaved(true)
        setTimeout(() => setSaved(false), 2000)
      }
    } catch (e) {}
    setSaving(false)
  }

  return (
    <div className="glass-panel glass-panel-pink p-6 h-full corner-deco">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-2">
          <span className="text-pink-400 text-base">⚙</span>
          <span className="text-label text-pink-400/70">SİSTEM AYARLARI</span>
        </div>
        <div className={`text-xs font-bold tracking-wider ${saved ? 'text-green-400' : 'text-gray-600'}`}>
          {saved ? '● KAYDEDİLDİ' : ''}
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Left: Controls */}
        <div className="space-y-4">
          {/* Start/Stop */}
          <div className="bg-black/30 rounded-lg p-4 border border-cyan-500/10">
            <p className="text-label text-cyan-400/60 mb-3" style={{ fontSize: '11px' }}>SİSTEM KONTROLÜ</p>
            <div className="flex gap-3">
              <button
                onClick={onStart}
                disabled={isRunning}
                className={`flex-1 py-3 rounded-lg text-sm font-bold tracking-wider transition-all ${
                  isRunning
                    ? 'bg-gray-800 text-gray-600 cursor-not-allowed border border-gray-700'
                    : 'neon-btn bg-green-500/10 text-green-400 border border-green-500/30 hover:bg-green-500/20'
                }`}
              >
                {isRunning ? '● ZATEN AKTİF' : '▶ BAŞLAT'}
              </button>
              <button
                onClick={onStop}
                disabled={!isRunning}
                className={`flex-1 py-3 rounded-lg text-sm font-bold tracking-wider transition-all ${
                  !isRunning
                    ? 'bg-gray-800 text-gray-600 cursor-not-allowed border border-gray-700'
                    : 'neon-btn bg-red-500/10 text-red-400 border border-red-500/30 hover:bg-red-500/20'
                }`}
              >
                {!isRunning ? '■ ZATEN DURDURULDU' : '■ DURDUR'}
              </button>
            </div>
          </div>

          {/* Cycle & Confidence */}
          <div className="bg-black/30 rounded-lg p-4 border border-purple-500/10">
            <p className="text-label text-purple-400/60 mb-3" style={{ fontSize: '11px' }}>TARAMA AYARLARI</p>
            <div className="space-y-3">
              <div>
                <div className="flex justify-between mb-1">
                  <label className="text-sm text-gray-300 font-semibold">DÖNGÜ SÜRESİ</label>
                  <span className="text-base text-purple-400 font-bold">{settings.cycle_interval}s</span>
                </div>
                <input
                  type="range" min="60" max="600" step="30"
                  value={settings.cycle_interval}
                  onChange={e => setSettings({ ...settings, cycle_interval: +e.target.value })}
                  className="w-full accent-purple-500"
                />
                <div className="flex justify-between text-xs text-gray-600 mt-1">
                  <span>60s</span>
                  <span>600s</span>
                </div>
              </div>
              <div>
                <div className="flex justify-between mb-1">
                  <label className="text-sm text-gray-300 font-semibold">MİN GÜVEN</label>
                  <span className="text-base text-purple-400 font-bold">%{(settings.min_confidence * 100).toFixed(0)}</span>
                </div>
                <input
                  type="range" min="0.05" max="0.95" step="0.05"
                  value={settings.min_confidence}
                  onChange={e => setSettings({ ...settings, min_confidence: +e.target.value })}
                  className="w-full accent-purple-500"
                />
                <div className="flex justify-between text-xs text-gray-600 mt-1">
                  <span>%5</span>
                  <span>%95</span>
                </div>
              </div>
            </div>
          </div>

          {/* Test Telegram */}
          <div className="bg-black/30 rounded-lg p-4 border border-blue-500/10">
            <p className="text-label text-blue-400/60 mb-3" style={{ fontSize: '11px' }}>TELEGRAM TEST</p>
            <button
              onClick={async () => {
                try {
                  await fetch('/api/test/telegram')
                  alert('Test mesajı gönderildi!')
                } catch (e) {
                  alert('Gönderim hatası!')
                }
              }}
              className="w-full py-2 rounded-lg text-sm font-bold tracking-wider neon-btn bg-blue-500/10 text-blue-400 border border-blue-500/30 hover:bg-blue-500/20 transition-all"
            >
              📱 TELEGRAM TEST GÖNDER
            </button>
          </div>
        </div>

        {/* Right: Risk */}
        <div className="space-y-4">
          <div className="bg-black/30 rounded-lg p-4 border border-red-500/10">
            <p className="text-label text-red-400/60 mb-3" style={{ fontSize: '11px' }}>RİSK YÖNETİMİ</p>
            <div className="space-y-3">
              {[
                { key: 'risk_per_trade', label: 'İŞLEM BAŞINA RİSK (KASA KULLANIMI)', min: 0.5, max: 5, step: 0.5, suffix: '%', color: 'text-red-400' },
                { key: 'daily_risk', label: 'GÜNLÜK RİSK', min: 1, max: 10, step: 0.5, suffix: '%', color: 'text-orange-400' },
                { key: 'leverage_max', label: 'MAKS KALDIRAÇ', min: 1, max: 50, step: 1, suffix: 'x', color: 'text-yellow-400' },
                { key: 'max_positions', label: 'MAKS POZİSYON', min: 1, max: 10, step: 1, suffix: '', color: 'text-green-400' },
              ].map(item => (
                <div key={item.key}>
                  <div className="flex justify-between mb-1">
                    <label className="text-sm text-gray-300 font-semibold">{item.label}</label>
                    <span className={`text-base font-bold ${item.color}`}>
                      {settings[item.key]}{item.suffix}
                    </span>
                  </div>
                  <input
                    type="range"
                    min={item.min} max={item.max} step={item.step}
                    value={settings[item.key]}
                    onChange={e => setSettings({ ...settings, [item.key]: +e.target.value })}
                    className="w-full accent-red-500"
                  />
                  <div className="flex justify-between text-xs text-gray-600 mt-1">
                    <span>{item.min}{item.suffix}</span>
                    <span>{item.max}{item.suffix}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Save */}
          <button
            onClick={handleSave}
            disabled={saving}
            className={`w-full py-3 rounded-lg text-sm font-bold tracking-wider transition-all ${
              saving
                ? 'bg-gray-800 text-gray-600 cursor-not-allowed border border-gray-700'
                : 'neon-btn bg-cyan-500/10 text-cyan-400 border border-cyan-500/30 hover:bg-cyan-500/20'
            }`}
          >
            {saving ? '● KAYDEDİLİYOR...' : saved ? '● BAŞARIYLA KAYDEDİLDİ' : '◆ AYARLARI KAYDET'}
          </button>
        </div>
      </div>
    </div>
  )
}
