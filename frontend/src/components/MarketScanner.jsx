const BLACKLIST_BASE = ['USDC','BUSD','DAI','TUSD','USDP','FDUSD','PYUSD','EUR','GBP','JPY','AUD','CAD','CHF','NZD','CNY','OIL','GOLD','SILVER','NATGAS','COPPER','PLATINUM','DOW','SP500','NASDAQ','SPX','DAX','FTSE','NIKKEI','XAU','XAG','XPT','XPD','STOCK','ETF','BOND','FUND','INDEX','FUTURES']
function isStockSymbol(symbol, baseCoin) {
  const sym = (symbol || '').toUpperCase(); const base = (baseCoin || '').toUpperCase()
  if (BLACKLIST_BASE.includes(base)) return true; return BLACKLIST_BASE.some(kw => sym.includes(kw))
}

export default function MarketScanner({ data }) {
  const topSymbols = (data?.hot_pairs || []).filter(s => !isStockSymbol(s.symbol, s.baseCoin))
  const getHeatColor = (score) => score >= 80 ? '#39ff14' : score >= 60 ? '#ffd700' : score >= 40 ? '#ff6b35' : '#ff3366'

  return (
    <div className="cinematic-border rounded-2xl p-5 h-full" style={{ background: 'linear-gradient(145deg, rgba(6,10,22,0.95) 0%, rgba(10,14,28,0.9) 100%)' }}>
      <div className="flex items-center justify-between mb-5">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl flex items-center justify-center" style={{ background: '#00f0ff08', border: '1px solid #00f0ff15' }}>
            <span style={{ color: '#00f0ff', fontSize: '13px' }}>◎</span>
          </div>
          <span className="text-[11px] font-bold tracking-[0.15em] text-gray-400 font-mono">PİYASA TARAYICI</span>
        </div>
        <div className="text-[11px] text-gray-600 font-mono">{data?.symbol_count || 0} sembol</div>
      </div>

      {topSymbols.length > 0 ? (
        <div className="space-y-1.5">
          {topSymbols.slice(0, 15).map((sym, i) => {
            const score = Math.round(Math.min(Math.max(sym.hot_score || 0, 0), 100))
            const heatColor = getHeatColor(score)
            return (
              <div key={i} className="flex items-center gap-3 p-2.5 rounded-xl border border-white/[0.03] transition-all hover:border-white/[0.06]" style={{ background: `${heatColor}03` }}>
                <span className="text-[11px] text-gray-700 w-5 text-right font-mono">#{i+1}</span>
                <div className="flex-1">
                  <div className="flex items-center justify-between mb-1.5">
                    <div className="flex items-center gap-3">
                      <span className="text-[13px] font-bold text-white font-mono">{sym.symbol}</span>
                      {sym.baseCoin && <span className="text-[10px] text-gray-600 font-mono">{sym.baseCoin}</span>}
                    </div>
                    <div className="flex items-center gap-3">
                      {sym.last_price > 0 && <span className="text-[11px] text-gray-500 font-mono">${sym.last_price?.toLocaleString(undefined, {maximumFractionDigits: sym.last_price < 1 ? 6 : 2})}</span>}
                      {sym.change_24h !== undefined && <span className="text-[11px] font-mono font-bold" style={{ color: sym.change_24h >= 0 ? '#39ff14' : '#ff3366' }}>{sym.change_24h >= 0 ? '+' : ''}{sym.change_24h?.toFixed(2)}%</span>}
                      <span className="text-[13px] font-bold font-mono" style={{ color: heatColor }}>{score}</span>
                    </div>
                  </div>
                  <div className="w-full h-1 bg-white/[0.03] rounded-full overflow-hidden">
                    <div className="h-full rounded-full transition-all duration-700" style={{ width: `${Math.min(Math.max(score, 5), 100)}%`, background: `linear-gradient(90deg, #00f0ff, #b026ff, #ff0080)` }} />
                  </div>
                </div>
              </div>
            )
          })}
        </div>
      ) : (
        <div className="text-center py-12">
          <div className="text-4xl mb-3 opacity-10">◎</div>
          <p className="text-[13px] text-gray-600">Henüz tarama yapılmadı</p>
        </div>
      )}

      <div className="mt-4 pt-4 border-t border-white/[0.04] grid grid-cols-3 gap-2">
        {[
          { label: 'SEMBOL', value: data?.symbol_count || 0, color: '#00f0ff' },
          { label: 'HOT', value: topSymbols.length, color: '#39ff14' },
          { label: 'DÖNGÜ', value: '#', color: '#b026ff' },
        ].map((item, i) => (
          <div key={i} className="text-center rounded-lg py-2 border border-white/[0.03]" style={{ background: `${item.color}03` }}>
            <p className="text-[9px] text-gray-600 tracking-wider font-mono mb-0.5">{item.label}</p>
            <p className="text-[14px] font-bold font-mono" style={{ color: item.color }}>{item.value}</p>
          </div>
        ))}
      </div>
    </div>
  )
}
