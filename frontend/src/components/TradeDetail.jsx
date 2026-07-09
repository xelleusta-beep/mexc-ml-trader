import { useEffect, useRef, useState } from 'react'
import { createChart } from 'lightweight-charts'

function formatDateTime(ts) {
  if (!ts) return ''
  const d = new Date(ts * 1000)
  return d.toLocaleString('tr-TR', {
    day: '2-digit', month: '2-digit', year: 'numeric',
    hour: '2-digit', minute: '2-digit', second: '2-digit'
  })
}

export default function TradeDetail({ trade, onClose }) {
  const chartRef = useRef(null)
  const chartInstance = useRef(null)
  const [klines, setKlines] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    if (!trade) return
    setLoading(true)

    const idx = trade._index ?? 0
    fetch(`/api/trade/${idx}/klines`)
      .then(r => r.json())
      .then(data => {
        setKlines(data.klines || [])
        setLoading(false)
      })
      .catch(() => setLoading(false))
  }, [trade])

  useEffect(() => {
    if (!chartRef.current || klines.length === 0 || !trade) return

    if (chartInstance.current) {
      chartInstance.current.remove()
      chartInstance.current = null
    }

    const chart = createChart(chartRef.current, {
      width: chartRef.current.clientWidth,
      height: 400,
      layout: {
        background: { color: '#030712' },
        textColor: '#9ca3af',
        fontFamily: 'Rajdhani',
      },
      grid: {
        vertLines: { color: 'rgba(0, 240, 255, 0.04)' },
        horzLines: { color: 'rgba(0, 240, 255, 0.04)' },
      },
      crosshair: {
        mode: 0,
        vertLine: { color: 'rgba(0, 240, 255, 0.3)', width: 1, style: 2 },
        horzLine: { color: 'rgba(0, 240, 255, 0.3)', width: 1, style: 2 },
      },
      timeScale: {
        timeVisible: true,
        secondsVisible: false,
        borderColor: 'rgba(0, 240, 255, 0.1)',
      },
      rightPriceScale: {
        borderColor: 'rgba(0, 240, 255, 0.1)',
      },
    })

    const candleSeries = chart.addCandlestickSeries({
      upColor: '#39ff14',
      downColor: '#ff0040',
      borderDownColor: '#ff0040',
      borderUpColor: '#39ff14',
      wickDownColor: '#ff0040',
      wickUpColor: '#39ff14',
    })

    const candleData = klines.map(k => ({
      time: Math.floor(k.time / 1000),
      open: parseFloat(k.open),
      high: parseFloat(k.high),
      low: parseFloat(k.low),
      close: parseFloat(k.close),
    }))

    candleSeries.setData(candleData)

    const isProfit = trade.close_reason === 'TP tetiklendi'
    const entryColor = '#00f0ff'
    const exitColor = isProfit ? '#39ff14' : '#ff0040'

    candleSeries.createPriceLine({
      price: trade.entry_price,
      color: entryColor,
      lineWidth: 1,
      lineStyle: 2,
      axisLabelVisible: true,
      title: 'GİRİŞ',
    })

    if (trade.exit_price) {
      candleSeries.createPriceLine({
        price: trade.exit_price,
        color: exitColor,
        lineWidth: 1,
        lineStyle: 2,
        axisLabelVisible: true,
        title: isProfit ? 'TP' : 'SL',
      })
    }

    const entryTime = Math.floor(trade.entry_time)
    const closeTime = Math.floor(trade.close_time)

    candleSeries.createPriceLine({
      price: trade.entry_price,
      color: 'transparent',
      lineWidth: 0,
      lineStyle: 0,
      axisLabelVisible: false,
      title: '',
    })

    chart.timeScale().fitContent()

    const timeRange = closeTime - entryTime
    const padding = timeRange * 0.15
    chart.timeScale().setVisibleRange({
      from: entryTime - padding,
      to: closeTime + padding,
    })

    chartInstance.current = chart

    const handleResize = () => {
      if (chartRef.current && chartInstance.current) {
        chartInstance.current.applyOptions({ width: chartRef.current.clientWidth })
      }
    }
    window.addEventListener('resize', handleResize)

    return () => {
      window.removeEventListener('resize', handleResize)
      if (chartInstance.current) {
        chartInstance.current.remove()
        chartInstance.current = null
      }
    }
  }, [klines, trade])

  if (!trade) return null

  const isProfit = (trade.pnl || 0) >= 0
  const isTp = trade.close_reason === 'TP tetiklendi'

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center p-4" onClick={onClose}>
      <div className="absolute inset-0 bg-black/80 backdrop-blur-sm" />
      <div
        className="relative glass-panel glass-panel-purple p-6 rounded-xl w-full max-w-4xl max-h-[90vh] overflow-y-auto border border-purple-500/20"
        onClick={e => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <span className={`text-2xl font-bold ${isProfit ? 'neon-green' : 'neon-red'}`}>
              {trade.symbol}
            </span>
            <span className={`tag text-sm ${
              trade.direction === 'long' ? 'border-green-500/30 neon-green' : 'border-red-500/30 neon-red'
            }`}>
              {trade.direction?.toUpperCase()}
            </span>
            <span className={`tag text-sm border ${
              isTp ? 'border-green-500/30 text-green-400 bg-green-500/10' : 'border-red-500/30 text-red-400 bg-red-500/10'
            }`}>
              {trade.close_reason}
            </span>
          </div>
          <button onClick={onClose} className="text-gray-500 hover:text-white text-2xl px-2">✕</button>
        </div>

        {/* Trade Info */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4">
          <div className="bg-black/30 rounded-lg p-3 border border-cyan-500/10">
            <p className="text-[10px] text-cyan-400/60 tracking-wider">GİRİŞ FİYATI</p>
            <p className="text-lg font-bold text-white">${trade.entry_price?.toFixed(4)}</p>
            <p className="text-[10px] text-gray-500 mt-1">{formatDateTime(trade.entry_time)}</p>
          </div>
          <div className="bg-black/30 rounded-lg p-3 border border-cyan-500/10">
            <p className="text-[10px] text-cyan-400/60 tracking-wider">KAPANIŞ FİYATI</p>
            <p className="text-lg font-bold text-white">${trade.exit_price?.toFixed(4)}</p>
            <p className="text-[10px] text-gray-500 mt-1">{formatDateTime(trade.close_time)}</p>
          </div>
          <div className="bg-black/30 rounded-lg p-3 border border-cyan-500/10">
            <p className="text-[10px] text-cyan-400/60 tracking-wider">POZİSYON</p>
            <p className="text-lg font-bold text-white">${trade.size_usd?.toFixed(0)} x{trade.leverage}</p>
          </div>
          <div className={`bg-black/30 rounded-lg p-3 border ${isProfit ? 'border-green-500/10' : 'border-red-500/10'}`}>
            <p className="text-[10px] text-cyan-400/60 tracking-wider">SONUÇ</p>
            <p className={`text-2xl font-bold ${isProfit ? 'neon-green' : 'neon-red'}`}>
              {isProfit ? '+' : ''}{trade.pnl?.toFixed(2)}
            </p>
            <p className={`text-sm ${isProfit ? 'text-green-400' : 'text-red-400'}`}>
              {isProfit ? '+' : ''}{trade.pnl_pct?.toFixed(1)}%
            </p>
          </div>
        </div>

        {/* Chart */}
        <div className="bg-black/30 rounded-lg p-3 border border-purple-500/10">
          <div className="flex items-center gap-2 mb-2">
            <span className="text-purple-400 text-sm">◇</span>
            <span className="text-[10px] text-purple-400/60 tracking-wider">5 DAKİKALIK MUM GRAFİĞİ</span>
          </div>
          {loading ? (
            <div className="h-[400px] flex items-center justify-center">
              <div className="text-center">
                <div className="text-4xl mb-3 opacity-20 animate-pulse-neon">◇</div>
                <p className="text-sm text-gray-500">Grafik yükleniyor...</p>
              </div>
            </div>
          ) : klines.length > 0 ? (
            <div ref={chartRef} className="w-full rounded overflow-hidden" />
          ) : (
            <div className="h-[400px] flex items-center justify-center">
              <p className="text-sm text-gray-500">Mum verisi bulunamadı</p>
            </div>
          )}
        </div>

        {/* Legend */}
        <div className="flex items-center gap-4 mt-3 text-[10px] text-gray-500">
          <div className="flex items-center gap-1">
            <div className="w-4 h-0.5 bg-cyan-400" style={{ borderTop: '1px dashed #00f0ff' }} />
            <span>Giriş Fiyatı</span>
          </div>
          <div className="flex items-center gap-1">
            <div className={`w-4 h-0.5 ${isProfit ? 'bg-green-400' : 'bg-red-400'}`} style={{ borderTop: '1px dashed' }} />
            <span>{isTp ? 'TP Fiyatı' : 'SL Fiyatı'}</span>
          </div>
          <span className="text-gray-600">|</span>
          <span>{klines.length} mum</span>
          <span>{formatDateTime(trade.entry_time)} → {formatDateTime(trade.close_time)}</span>
        </div>
      </div>
    </div>
  )
}
