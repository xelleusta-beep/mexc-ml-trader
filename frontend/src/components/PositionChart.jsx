import { useEffect, useRef, useState } from 'react'
import { createChart } from 'lightweight-charts'
import { getPositionKlines } from '../api/trading'

export default function PositionChart({ symbol, onClose }) {
  const containerRef = useRef(null)
  const chartRef = useRef(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [info, setInfo] = useState(null)

  useEffect(() => {
    if (!symbol || !containerRef.current) return

    let chart = null
    let candleSeries = null
    let volumeSeries = null
    let resizeObs = null
    let disposed = false

    try {
      chart = createChart(containerRef.current, {
        width: containerRef.current.clientWidth || 800,
        height: 400,
        layout: {
          background: { color: '#0a0e17' },
          textColor: '#8b95a5',
          fontSize: 11,
        },
        grid: {
          vertLines: { color: 'rgba(42, 46, 57, 0.4)' },
          horzLines: { color: 'rgba(42, 46, 57, 0.4)' },
        },
        crosshair: {
          mode: 0,
        },
        rightPriceScale: {
          borderColor: 'rgba(42, 46, 57, 0.6)',
        },
        timeScale: {
          borderColor: 'rgba(42, 46, 57, 0.6)',
          timeVisible: true,
          secondsVisible: false,
        },
      })

      if (disposed) { chart.remove(); return }

      candleSeries = chart.addCandlestickSeries({
        upColor: '#00ff88',
        downColor: '#ff3366',
        borderDownColor: '#ff3366',
        borderUpColor: '#00ff88',
        wickDownColor: '#ff3366',
        wickUpColor: '#00ff88',
      })

      volumeSeries = chart.addHistogramSeries({
        priceFormat: { type: 'volume' },
        priceScaleId: 'vol',
      })
      chart.priceScale('vol').applyOptions({
        scaleMargins: { top: 0.85, bottom: 0 },
      })

      chartRef.current = chart

      resizeObs = new ResizeObserver(entries => {
        if (entries[0] && chart) {
          chart.applyOptions({ width: entries[0].contentRect.width })
        }
      })
      resizeObs.observe(containerRef.current)

      loadData(symbol, chart, candleSeries, volumeSeries)
    } catch (e) {
      console.error('Chart init hatası:', e)
      setError('Grafik oluşturulamadı: ' + e.message)
      setLoading(false)
    }

    return () => {
      disposed = true
      if (resizeObs) resizeObs.disconnect()
      if (chart) {
        try { chart.remove() } catch {}
      }
      chartRef.current = null
    }
  }, [symbol])

  async function loadData(sym, chart, candleSeries, volumeSeries) {
    setLoading(true)
    setError(null)
    try {
      const data = await getPositionKlines(sym)
      setInfo(data)

      if (!data.klines || data.klines.length === 0) {
        setError('Mum verisi bulunamadı')
        setLoading(false)
        return
      }

      const candles = data.klines.map(k => ({
        time: Math.floor(k.time / 1000),
        open: parseFloat(k.open),
        high: parseFloat(k.high),
        low: parseFloat(k.low),
        close: parseFloat(k.close),
      }))

      const volumes = data.klines.map(k => ({
        time: Math.floor(k.time / 1000),
        value: parseFloat(k.vol || 0),
        color: parseFloat(k.close) >= parseFloat(k.open)
          ? 'rgba(0, 255, 136, 0.25)'
          : 'rgba(255, 51, 102, 0.25)',
      }))

      candleSeries.setData(candles)
      volumeSeries.setData(volumes)

      if (data.entry_price > 0) {
        candleSeries.createPriceLine({
          price: data.entry_price,
          color: '#00d4ff',
          lineWidth: 2,
          lineStyle: 2,
          axisLabelVisible: true,
          title: 'GİRİŞ',
        })
      }
      if (data.take_profit > 0) {
        candleSeries.createPriceLine({
          price: data.take_profit,
          color: '#00ff88',
          lineWidth: 2,
          lineStyle: 2,
          axisLabelVisible: true,
          title: 'TP',
        })
      }
      if (data.stop_loss > 0) {
        candleSeries.createPriceLine({
          price: data.stop_loss,
          color: '#ff3366',
          lineWidth: 2,
          lineStyle: 2,
          axisLabelVisible: true,
          title: 'SL',
        })
      }

      chart.timeScale().fitContent()
    } catch (e) {
      console.error('Klines yüklenemedi:', e)
      setError('Veri yüklenemedi')
    }
    setLoading(false)
  }

  useEffect(() => {
    if (!chartRef.current || !symbol) return
    const iv = setInterval(() => {
      if (!chartRef.current) return
      const cs = chartRef.current.series || []
      loadData(symbol, chartRef.current, cs[0], cs[1])
    }, 10000)
    return () => clearInterval(iv)
  }, [symbol])

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center"
      style={{ backgroundColor: 'rgba(0,0,0,0.85)' }}
      onMouseDown={(e) => { if (e.target === e.currentTarget) onClose() }}
    >
      <div
        className="bg-[#0a0e17] border border-green-500/20 p-4 rounded-xl w-[95vw] max-w-5xl relative shadow-[0_0_40px_rgba(0,255,136,0.1)]"
        onMouseDown={e => e.stopPropagation()}
      >
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-3 flex-wrap">
            <span className="text-xl font-bold text-white">{symbol}</span>
            <span className={`text-xs px-2 py-0.5 rounded border ${
              info?.direction === 'long'
                ? 'border-green-500/30 text-green-400 bg-green-500/10'
                : 'border-red-500/30 text-red-400 bg-red-500/10'
            }`}>
              {info?.direction?.toUpperCase() || '?'}
            </span>
            {info?.entry_price > 0 && (
              <span className="text-xs text-cyan-400">Giriş: ${info.entry_price?.toFixed(6)}</span>
            )}
            {info?.current_price > 0 && (
              <span className="text-xs text-white font-bold">Şuan: ${info.current_price?.toFixed(6)}</span>
            )}
          </div>
          <div className="flex items-center gap-3">
            {info?.take_profit > 0 && (
              <span className="text-xs text-green-400">TP: ${info.take_profit?.toFixed(6)}</span>
            )}
            {info?.stop_loss > 0 && (
              <span className="text-xs text-red-400">SL: ${info.stop_loss?.toFixed(6)}</span>
            )}
            <button
              onClick={onClose}
              className="w-8 h-8 flex items-center justify-center rounded bg-white/10 hover:bg-red-500/30 text-gray-400 hover:text-white transition-colors text-sm font-bold"
            >
              ✕
            </button>
          </div>
        </div>

        <div className="relative rounded-lg overflow-hidden border border-green-500/10">
          {loading && (
            <div className="absolute inset-0 flex items-center justify-center z-10 bg-black/50">
              <div className="flex items-center gap-2 text-green-400">
                <div className="w-4 h-4 border-2 border-green-400 border-t-transparent rounded-full animate-spin" />
                <span className="text-sm">Yükleniyor...</span>
              </div>
            </div>
          )}
          {error && (
            <div className="absolute inset-0 flex items-center justify-center z-10 bg-black/50">
              <span className="text-red-400 text-sm">{error}</span>
            </div>
          )}
          <div ref={containerRef} style={{ width: '100%', height: 400 }} />
        </div>

        <div className="flex items-center gap-4 mt-2 text-[10px] text-gray-500">
          <span className="flex items-center gap-1">
            <span className="w-3 h-0.5 bg-cyan-400 inline-block rounded" /> GİRİŞ
          </span>
          <span className="flex items-center gap-1">
            <span className="w-3 h-0.5 bg-green-400 inline-block rounded" /> TP
          </span>
          <span className="flex items-center gap-1">
            <span className="w-3 h-0.5 bg-red-400 inline-block rounded" /> SL
          </span>
          <span className="ml-auto">5m mum • 10sn canlı yenileme</span>
        </div>
      </div>
    </div>
  )
}
