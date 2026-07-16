import { useEffect, useRef, useState, useCallback } from 'react'
import { createChart, CandlestickSeries, HistogramSeries, createSeriesMarkers } from 'lightweight-charts'
import { getPositionKlines } from '../api/trading'

export default function PositionChart({ symbol, onClose }) {
  const containerRef = useRef(null)
  const chartRef = useRef(null)
  const markersRef = useRef(null)
  const priceLinesRef = useRef([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [info, setInfo] = useState(null)

  const removePriceLines = useCallback((series) => {
    for (const line of priceLinesRef.current) {
      try { series.removePriceLine(line) } catch {}
    }
    priceLinesRef.current = []
  }, [])

  const addPriceLines = useCallback((series, data) => {
    removePriceLines(series)
    const lines = []
    if (data.entry_price > 0) {
      lines.push(series.createPriceLine({
        price: data.entry_price, color: '#ff9800', lineWidth: 2,
        lineStyle: 0, axisLabelVisible: true, title: 'GİRİŞ',
      }))
    }
    if (data.take_profit > 0) {
      lines.push(series.createPriceLine({
        price: data.take_profit, color: '#76d672', lineWidth: 2,
        lineStyle: 0, axisLabelVisible: true, title: 'TP',
      }))
    }
    if (data.stop_loss > 0) {
      lines.push(series.createPriceLine({
        price: data.stop_loss, color: '#ff3366', lineWidth: 2,
        lineStyle: 0, axisLabelVisible: true, title: 'SL',
      }))
    }
    priceLinesRef.current = lines
  }, [removePriceLines])

  const buildMarkers = useCallback((data) => {
    if (!data.klines || data.klines.length === 0) return []
    const dir = data.direction || 'long'
    const entryTimeSec = data.entry_time > 0 ? Math.floor(data.entry_time) : 0
    let closestIdx = 0
    let minDiff = Infinity
    data.klines.forEach((k, i) => {
      const diff = Math.abs(Math.floor(k.time / 1000) - entryTimeSec)
      if (diff < minDiff) { minDiff = diff; closestIdx = i }
    })
    const markers = [{
      time: Math.floor(data.klines[closestIdx].time / 1000),
      position: dir === 'long' ? 'belowBar' : 'aboveBar',
      color: dir === 'long' ? '#76d672' : '#ff3366',
      shape: dir === 'long' ? 'arrowUp' : 'arrowDown',
      text: dir.toUpperCase(),
    }]
    return markers
  }, [])

  useEffect(() => {
    if (!symbol || !containerRef.current) return
    let chart = null
    let candleSeries = null
    let volumeSeries = null
    let resizeObs = null
    let disposed = false
    let iv = null

    try {
      chart = createChart(containerRef.current, {
        width: containerRef.current.clientWidth || 800,
        height: 420,
        layout: {
          background: { color: '#060a14' },
          textColor: '#8b95a5',
          fontSize: 11,
        },
        grid: {
          vertLines: { color: 'rgba(42, 46, 57, 0.3)' },
          horzLines: { color: 'rgba(42, 46, 57, 0.3)' },
        },
        crosshair: { mode: 0 },
        rightPriceScale: { borderColor: 'rgba(42, 46, 57, 0.5)' },
        timeScale: { borderColor: 'rgba(42, 46, 57, 0.5)', timeVisible: true, secondsVisible: false },
      })
      if (disposed) { chart.remove(); return }

      candleSeries = chart.addSeries(CandlestickSeries, {
        upColor: '#00ff88', downColor: '#ff3366',
        borderDownColor: '#ff3366', borderUpColor: '#00ff88',
        wickDownColor: '#ff3366', wickUpColor: '#00ff88',
      })
      volumeSeries = chart.addSeries(HistogramSeries, {
        priceFormat: { type: 'volume' }, priceScaleId: 'vol',
      })
      chart.priceScale('vol').applyOptions({ scaleMargins: { top: 0.85, bottom: 0 } })

      markersRef.current = createSeriesMarkers(candleSeries, [])
      chartRef.current = { chart, candleSeries, volumeSeries, initialized: false }

      resizeObs = new ResizeObserver(entries => {
        if (entries[0] && chart) chart.applyOptions({ width: entries[0].contentRect.width })
      })
      resizeObs.observe(containerRef.current)

      loadInitial(symbol, chart, candleSeries, volumeSeries)

      iv = setInterval(() => {
        loadUpdate(symbol, candleSeries, volumeSeries)
      }, 300000)
    } catch (e) {
      setError('Grafik oluşturulamadı: ' + e.message)
      setLoading(false)
    }

    return () => {
      disposed = true
      if (iv) clearInterval(iv)
      if (resizeObs) resizeObs.disconnect()
      if (chart) { try { chart.remove() } catch {} }
      chartRef.current = null
      markersRef.current = null
    }
  }, [symbol])

  async function loadInitial(sym, chart, candleSeries, volumeSeries) {
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
        open: parseFloat(k.open), high: parseFloat(k.high),
        low: parseFloat(k.low), close: parseFloat(k.close),
      }))
      const volumes = data.klines.map(k => ({
        time: Math.floor(k.time / 1000),
        value: parseFloat(k.vol || 0),
        color: parseFloat(k.close) >= parseFloat(k.open)
          ? 'rgba(0, 255, 136, 0.25)' : 'rgba(255, 51, 102, 0.25)',
      }))
      candleSeries.setData(candles)
      volumeSeries.setData(volumes)
      addPriceLines(candleSeries, data)
      if (markersRef.current) markersRef.current.setMarkers(buildMarkers(data))
      chart.timeScale().fitContent()
      if (chartRef.current) chartRef.current.initialized = true
    } catch (e) {
      setError('Veri yüklenemedi')
    }
    setLoading(false)
  }

  async function loadUpdate(sym, candleSeries, volumeSeries) {
    try {
      const data = await getPositionKlines(sym)
      setInfo(data)
      if (!data.klines || data.klines.length === 0) return
      const last = data.klines[data.klines.length - 1]
      const candle = {
        time: Math.floor(last.time / 1000),
        open: parseFloat(last.open), high: parseFloat(last.high),
        low: parseFloat(last.low), close: parseFloat(last.close),
      }
      candleSeries.update(candle)
      volumeSeries.update({
        time: candle.time,
        value: parseFloat(last.vol || 0),
        color: candle.close >= candle.open
          ? 'rgba(0, 255, 136, 0.25)' : 'rgba(255, 51, 102, 0.25)',
      })
      addPriceLines(candleSeries, data)
      if (markersRef.current) markersRef.current.setMarkers(buildMarkers(data))
    } catch {}
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center"
      style={{ backgroundColor: 'rgba(0,0,0,0.88)' }}
      onMouseDown={(e) => { if (e.target === e.currentTarget) onClose() }}
    >
      <div
        className="bg-[#060a14] border border-green-500/15 p-4 rounded-2xl w-[95vw] max-w-5xl relative shadow-[0_0_60px_rgba(0,255,136,0.08)]"
        onMouseDown={e => e.stopPropagation()}
      >
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-3 flex-wrap">
            <span className="text-xl font-bold text-white font-mono">{symbol}</span>
            <span className={`text-[10px] px-2 py-0.5 rounded-md border font-bold tracking-wider ${
              info?.direction === 'long'
                ? 'border-green-500/30 text-green-400 bg-green-500/10'
                : 'border-red-500/30 text-red-400 bg-red-500/10'
            }`}>
              {info?.direction?.toUpperCase() || '?'}
            </span>
            {info?.entry_price > 0 && (
              <span className="text-xs text-orange-400 font-mono">Giriş: ${info.entry_price?.toFixed(6)}</span>
            )}
            {info?.current_price > 0 && (
              <span className="text-xs text-white font-bold font-mono">Şuan: ${info.current_price?.toFixed(6)}</span>
            )}
          </div>
          <div className="flex items-center gap-3">
            {info?.take_profit > 0 && (
              <span className="text-xs text-green-400 font-mono">TP: ${info.take_profit?.toFixed(6)}</span>
            )}
            {info?.stop_loss > 0 && (
              <span className="text-xs text-red-400 font-mono">SL: ${info.stop_loss?.toFixed(6)}</span>
            )}
            <button onClick={onClose} className="w-8 h-8 flex items-center justify-center rounded-lg bg-white/5 hover:bg-red-500/20 text-gray-400 hover:text-white transition-all text-sm font-bold border border-white/5 hover:border-red-500/30">✕</button>
          </div>
        </div>

        <div className="relative rounded-xl overflow-hidden border border-white/5">
          {loading && (
            <div className="absolute inset-0 flex items-center justify-center z-10 bg-[#060a14]/80 backdrop-blur-sm">
              <div className="flex items-center gap-2 text-green-400">
                <div className="w-4 h-4 border-2 border-green-400 border-t-transparent rounded-full animate-spin" />
                <span className="text-sm font-mono">Yükleniyor...</span>
              </div>
            </div>
          )}
          {error && (
            <div className="absolute inset-0 flex items-center justify-center z-10 bg-[#060a14]/80">
              <span className="text-red-400 text-sm font-mono">{error}</span>
            </div>
          )}
          <div ref={containerRef} style={{ width: '100%', height: 420 }} />
        </div>

        <div className="flex items-center gap-4 mt-2.5 text-[10px] text-gray-500 font-mono">
          <span className="flex items-center gap-1.5"><span className="w-3 h-0.5 bg-[#ff9800] inline-block rounded" /> GİRİŞ</span>
          <span className="flex items-center gap-1.5"><span className="w-3 h-0.5 bg-[#76d672] inline-block rounded" /> TP</span>
          <span className="flex items-center gap-1.5"><span className="w-3 h-0.5 bg-[#ff3366] inline-block rounded" /> SL</span>
          <span className="flex items-center gap-1.5"><span className="text-xs text-[#76d672]">▲</span> LONG</span>
          <span className="flex items-center gap-1.5"><span className="text-xs text-[#ff3366]">▼</span> SHORT</span>
          <span className="flex items-center gap-1.5"><span className="text-xs">🏆</span> ÇIKIŞ</span>
          <span className="ml-auto text-gray-600">5m mum • 5dk canlı</span>
        </div>
      </div>
    </div>
  )
}
