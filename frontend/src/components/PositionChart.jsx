import { useEffect, useRef, useState } from 'react'
import { createChart } from 'lightweight-charts'
import { getPositionKlines } from '../api/trading'

export default function PositionChart({ symbol, onClose }) {
  const containerRef = useRef(null)
  const chartRef = useRef(null)
  const candleSeriesRef = useRef(null)
  const volumeSeriesRef = useRef(null)
  const markersRef = useRef([])
  const linesRef = useRef([])
  const [loading, setLoading] = useState(true)
  const [info, setInfo] = useState(null)

  useEffect(() => {
    if (!symbol || !containerRef.current) return

    const chart = createChart(containerRef.current, {
      width: containerRef.current.clientWidth,
      height: 420,
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
        vertLine: { color: 'rgba(0, 255, 136, 0.3)', width: 1, style: 2, labelBackgroundColor: '#00ff88' },
        horzLine: { color: 'rgba(0, 255, 136, 0.3)', width: 1, style: 2, labelBackgroundColor: '#00ff88' },
      },
      rightPriceScale: {
        borderColor: 'rgba(42, 46, 57, 0.6)',
        scaleMargins: { top: 0.05, bottom: 0.2 },
      },
      timeScale: {
        borderColor: 'rgba(42, 46, 57, 0.6)',
        timeVisible: true,
        secondsVisible: false,
      },
    })

    const candleSeries = chart.addCandlestickSeries({
      upColor: '#00ff88',
      downColor: '#ff3366',
      borderDownColor: '#ff3366',
      borderUpColor: '#00ff88',
      wickDownColor: '#ff3366',
      wickUpColor: '#00ff88',
    })

    const volumeSeries = chart.addHistogramSeries({
      priceFormat: { type: 'volume' },
      priceScaleId: '',
    })
    volumeSeries.priceScale().applyOptions({
      scaleMargins: { top: 0.85, bottom: 0 },
    })

    chartRef.current = chart
    candleSeriesRef.current = candleSeries
    volumeSeriesRef.current = volumeSeries

    const resizeObserver = new ResizeObserver(entries => {
      if (entries[0]) {
        chart.applyOptions({ width: entries[0].contentRect.width })
      }
    })
    resizeObserver.observe(containerRef.current)

    loadKlines(symbol, chart, candleSeries, volumeSeries)

    return () => {
      resizeObserver.disconnect()
      chart.remove()
      chartRef.current = null
    }
  }, [symbol])

  async function loadKlines(sym, chart, candleSeries, volumeSeries) {
    setLoading(true)
    try {
      const data = await getPositionKlines(sym)
      setInfo(data)

      const candles = (data.klines || []).map(k => ({
        time: Math.floor(k.time / 1000),
        open: parseFloat(k.open),
        high: parseFloat(k.high),
        low: parseFloat(k.low),
        close: parseFloat(k.close),
      }))

      const volumes = (data.klines || []).map(k => ({
        time: Math.floor(k.time / 1000),
        value: parseFloat(k.vol || 0),
        color: parseFloat(k.close) >= parseFloat(k.open)
          ? 'rgba(0, 255, 136, 0.2)'
          : 'rgba(255, 51, 102, 0.2)',
      }))

      candleSeries.setData(candles)
      volumeSeries.setData(volumes)

      removeLines(chart)
      addHorizontalLine(chart, data.entry_price, '#00d4ff', 'GİRİŞ', data.direction === 'long' ? 'abovePrice' : 'belowPrice')
      if (data.take_profit) addHorizontalLine(chart, data.take_profit, '#00ff88', 'TP')
      if (data.stop_loss) addHorizontalLine(chart, data.stop_loss, '#ff3366', 'SL')

      chart.timeScale().fitContent()
    } catch (e) {
      console.error('Klines yüklenemedi:', e)
    }
    setLoading(false)
  }

  function removeLines(chart) {
    linesRef.current.forEach(line => {
      try { chart.removePriceLine(line) } catch {}
    })
    linesRef.current = []
  }

  function addHorizontalLine(chart, price, color, title, position = 'inScale') {
    if (!price || price <= 0) return
    const line = chart.addPriceLine({
      price: price,
      color: color,
      lineWidth: 2,
      lineStyle: 2,
      axisLabelVisible: true,
      title: title,
    })
    linesRef.current.push(line)
  }

  useEffect(() => {
    if (!chartRef.current || !candleSeriesRef.current) return
    const interval = setInterval(() => {
      loadKlines(
        symbol,
        chartRef.current,
        candleSeriesRef.current,
        volumeSeriesRef.current
      )
    }, 10000)
    return () => clearInterval(interval)
  }, [symbol])

  const dirColor = info?.direction === 'long' ? 'text-green-400' : 'text-red-400'

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm" onClick={onClose}>
      <div className="glass-panel glass-panel-green p-5 rounded-xl w-[95vw] max-w-5xl relative corner-deco scanline" onClick={e => e.stopPropagation()}>
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-3">
            <span className="text-green-400 text-lg">◈</span>
            <span className="text-xl font-bold text-white">{symbol}</span>
            <span className={`tag border ${info?.direction === 'long' ? 'border-green-500/30 text-green-400' : 'border-red-500/30 text-red-400'}`}>
              {info?.direction?.toUpperCase()}
            </span>
            {info?.entry_price > 0 && (
              <span className="text-sm text-cyan-400">
                Giriş: ${info.entry_price?.toFixed(6)}
              </span>
            )}
            {info?.current_price > 0 && (
              <span className="text-sm text-white font-semibold">
                Şuan: ${info.current_price?.toFixed(6)}
              </span>
            )}
          </div>
          <div className="flex items-center gap-3">
            {info?.take_profit > 0 && (
              <span className="text-sm text-green-400">TP: ${info.take_profit?.toFixed(6)}</span>
            )}
            {info?.stop_loss > 0 && (
              <span className="text-sm text-red-400">SL: ${info.stop_loss?.toFixed(6)}</span>
            )}
            <button onClick={onClose} className="text-gray-400 hover:text-white text-xl px-2 py-1 rounded bg-white/5 hover:bg-white/10 transition-colors">✕</button>
          </div>
        </div>

        <div className="relative">
          {loading && (
            <div className="absolute inset-0 flex items-center justify-center z-10 bg-black/40 rounded-lg">
              <div className="flex items-center gap-2 text-green-400">
                <div className="w-4 h-4 border-2 border-green-400 border-t-transparent rounded-full animate-spin" />
                <span className="text-sm">Yükleniyor...</span>
              </div>
            </div>
          )}
          <div ref={containerRef} className="rounded-lg overflow-hidden border border-green-500/10" />
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
          <span className="ml-auto">5m mum • 10sn otomatik yenileme</span>
        </div>
      </div>
    </div>
  )
}
