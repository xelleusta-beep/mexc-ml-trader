function formatDateTime(ts) {
  if (!ts) return ''
  const d = new Date(ts * 1000)
  return d.toLocaleString('tr-TR', { day: '2-digit', month: '2-digit', year: 'numeric', hour: '2-digit', minute: '2-digit', second: '2-digit' })
}

function formatDuration(entryTs) {
  if (!entryTs) return ''
  const diff = (Date.now() / 1000 - entryTs) * 1000
  const mins = Math.floor(diff / 60000)
  const hours = Math.floor(mins / 60)
  const days = Math.floor(hours / 24)
  if (days > 0) return `${days}g ${hours % 24}sa`
  if (hours > 0) return `${hours}sa ${mins % 60}dk`
  return `${mins}dk`
}

export default function LivePositions({ data, positions: positionsProp, portfolio, fullPage, onPositionClick }) {
  let positions = positionsProp || data?.positions || []
  if (positions.length > 0 && positions[0]?.position) {
    positions = positions.map(p => ({ ...p.position, decision: p.decision }))
  }

  const getDirStyle = (dir) => {
    if (dir === 'long') return { label: 'LONG', color: '#39ff14', bg: '#39ff1406' }
    return { label: 'SHORT', color: '#ff3366', bg: '#ff336606' }
  }

  return (
    <div className={`cinematic-border rounded-2xl p-5 ${fullPage ? '' : 'h-full'}`} style={{ background: 'linear-gradient(145deg, rgba(6,10,22,0.95) 0%, rgba(10,22,18,0.9) 100%)' }}>
      <div className="flex items-center justify-between mb-5">
        <div className="flex items-center gap-3">
          <div className="relative">
            <div className={`w-2.5 h-2.5 rounded-full ${positions.length > 0 ? 'bg-green-400' : 'bg-gray-600'}`} />
            {positions.length > 0 && <div className="absolute inset-0 w-2.5 h-2.5 rounded-full bg-green-400 animate-ping opacity-40" />}
          </div>
          <span className="text-[11px] font-bold tracking-[0.15em] text-gray-400 font-mono">AÇIK POZİSYONLAR</span>
        </div>
        <div className="px-3 py-1.5 rounded-lg border" style={{
          borderColor: positions.length > 0 ? '#39ff1420' : '#ffffff08',
          background: positions.length > 0 ? '#39ff1406' : 'transparent',
        }}>
          <span className={`text-[10px] font-bold tracking-wider font-mono ${positions.length > 0 ? 'text-green-400' : 'text-gray-600'}`}>
            {positions.length} AÇIK
          </span>
        </div>
      </div>

      {positions.length > 0 ? (
        <div className="space-y-2">
          {positions.map((pos, i) => {
            const dir = getDirStyle(pos.direction)
            const pnl = pos.net_pnl || pos.unrealized_pnl || 0
            const pnlPct = pos.unrealized_pnl_pct || 0
            const isProfit = pnl >= 0
            const entryTime = formatDateTime(pos.entry_time)
            const duration = formatDuration(pos.entry_time)
            const priceSource = pos.price_source === '1m_kline' ? '1M' : pos.price_source === 'live' ? 'LIVE' : ''

            return (
              <div
                key={i}
                className="rounded-xl border border-white/[0.04] p-3.5 cursor-pointer transition-all duration-300 hover:border-white/[0.1]"
                style={{ background: dir.bg, boxShadow: `0 0 20px ${dir.color}05` }}
                onClick={() => onPositionClick && onPositionClick(pos.symbol)}
              >
                <div className="flex items-center justify-between mb-2.5">
                  <div className="flex items-center gap-2 flex-wrap">
                    <span className="text-[14px] font-bold text-white font-mono">{pos.symbol}</span>
                    <span className="text-[9px] px-2 py-0.5 rounded-md font-bold tracking-wider font-mono border" style={{ color: dir.color, borderColor: `${dir.color}30`, background: `${dir.color}10` }}>{dir.label}</span>
                    <span className="text-[9px] px-1.5 py-0.5 rounded-md font-mono" style={{ color: '#b026ff', border: '1px solid #b026ff20', background: '#b026ff08' }}>x{pos.leverage || 1}</span>
                    <span className="text-[9px] px-1.5 py-0.5 rounded-md font-mono" style={{ color: '#ffd700', border: '1px solid #ffd70020', background: '#ffd70008' }}>
                      {pos.margin_type === 'isolated' ? 'İzole' : 'Çapraz'}
                    </span>
                    {priceSource && <span className="text-[9px] px-1.5 py-0.5 rounded-md font-mono" style={{ color: '#00f0ff', border: '1px solid #00f0ff20', background: '#00f0ff08' }}>{priceSource}</span>}
                  </div>
                  <div className="text-right">
                    <p className="text-[20px] font-bold font-mono" style={{ color: isProfit ? '#39ff14' : '#ff3366' }}>
                      {isProfit ? '+' : ''}{pnl.toFixed(2)}
                    </p>
                    <p className="text-[11px] font-mono" style={{ color: isProfit ? '#39ff1480' : '#ff336680' }}>
                      {isProfit ? '+' : ''}{pnlPct.toFixed(2)}%
                    </p>
                  </div>
                </div>

                <div className="flex items-center gap-3 text-[11px] mb-2.5 bg-black/20 rounded-lg px-2.5 py-1.5">
                  <span className="text-orange-400 font-semibold">GİRİŞ:</span>
                  <span className="text-white font-semibold">{entryTime}</span>
                  <span className="text-gray-700">|</span>
                  <span className="font-mono" style={{ color: '#b026ff' }}>{duration}</span>
                </div>

                <div className="grid grid-cols-5 gap-1.5">
                  {[
                    { label: 'GİRİŞ', value: `$${pos.entry_price?.toFixed(4)}`, color: '#8b95a5' },
                    { label: 'ŞU ANKİ', value: `$${pos.current_price?.toFixed(4) || pos.entry_price?.toFixed(4)}`, color: '#ffffff' },
                    { label: 'BOYUT', value: `$${pos.size_usd?.toFixed(2)}`, color: '#8b95a5' },
                    { label: 'TP', value: `$${pos.take_profit?.toFixed(4)}`, color: '#39ff14' },
                    { label: 'SL', value: `$${pos.stop_loss?.toFixed(4)}`, color: '#ff3366' },
                  ].map((item, j) => (
                    <div key={j} className="bg-black/20 rounded-lg p-2 text-center border border-white/[0.02]">
                      <p className="text-[9px] text-gray-600 tracking-wider font-mono">{item.label}</p>
                      <p className="text-[11px] font-mono font-bold" style={{ color: item.color }}>{item.value}</p>
                    </div>
                  ))}
                </div>

                <div className="flex items-center justify-between mt-2 text-[10px] bg-black/15 rounded-lg px-2.5 py-1.5">
                  <span className="text-gray-600 font-mono">Fee: <span style={{ color: '#ff9800' }}>${(pos.entry_fee || 0).toFixed(4)}</span></span>
                  <span className="text-gray-600 font-mono">Net: <span style={{ color: (pos.net_pnl || 0) >= 0 ? '#39ff14' : '#ff3366' }}>{(pos.net_pnl || 0) >= 0 ? '+' : ''}{(pos.net_pnl || 0).toFixed(4)}</span></span>
                </div>
              </div>
            )
          })}
        </div>
      ) : (
        <div className="text-center py-12">
          <div className="text-4xl mb-3 opacity-10">▣</div>
          <p className="text-[13px] text-gray-600">Açık pozisyon yok</p>
          <p className="text-[10px] text-gray-700 mt-1 font-mono">Sistem fırsatları tarıyor</p>
        </div>
      )}
    </div>
  )
}
