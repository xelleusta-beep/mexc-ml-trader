import { useMemo } from 'react'

export default function EquityChart({ data }) {
  const chartData = useMemo(() => {
    if (!data?.equity_curve?.length) return null

    const curve = data.equity_curve
    const minEquity = Math.min(...curve.map(d => d.total_equity))
    const maxEquity = Math.max(...curve.map(d => d.total_equity))
    const range = maxEquity - minEquity || 1

    const width = 600
    const height = 200
    const padding = 40

    const points = curve.map((d, i) => ({
      x: padding + (i / (curve.length - 1)) * (width - padding * 2),
      y: padding + (1 - (d.total_equity - minEquity) / range) * (height - padding * 2),
      equity: d.total_equity,
      time: d.time,
      price: d.price,
      rsi: d.rsi,
      in_position: d.in_position,
    }))

    const pathD = points.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x} ${p.y}`).join(' ')

    const entryPoints = points.filter(p => p.in_position && (!points[points.indexOf(p) - 1]?.in_position))

    return { points, pathD, entryPoints, width, height, padding, minEquity, maxEquity }
  }, [data])

  if (!chartData) return null

  const formatTime = (ts) => {
    return new Date(ts).toLocaleDateString('tr-TR', { year: 'numeric', month: 'short' })
  }

  return (
    <div className="bg-gray-800 rounded-lg p-6">
      <h2 className="text-xl font-semibold text-purple-300 mb-2">
        {data.symbol} - Equity Curve
      </h2>
      <p className="text-sm text-gray-400 mb-4">
        Başlangıç: ${data.initial_capital} → Son: ${data.final_capital?.toFixed(2)}
        {' | '}
        <span className={data.total_pnl >= 0 ? 'text-green-400' : 'text-red-400'}>
          {data.total_pnl >= 0 ? '+' : ''}${data.total_pnl?.toFixed(2)} ({data.total_pnl_pct?.toFixed(2)}%)
        </span>
      </p>

      <svg viewBox={`0 0 ${chartData.width} ${chartData.height}`} className="w-full">
        <defs>
          <linearGradient id="equityGradient" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor="rgb(147, 51, 234)" stopOpacity="0.3" />
            <stop offset="100%" stopColor="rgb(147, 51, 234)" stopOpacity="0" />
          </linearGradient>
        </defs>

        <line
          x1={chartData.padding}
          y1={chartData.padding}
          x2={chartData.width - chartData.padding}
          y2={chartData.padding}
          stroke="rgb(75, 85, 99)"
          strokeWidth="0.5"
        />
        <line
          x1={chartData.padding}
          y1={chartData.height - chartData.padding}
          x2={chartData.width - chartData.padding}
          y2={chartData.height - chartData.padding}
          stroke="rgb(75, 85, 99)"
          strokeWidth="0.5"
        />

        <text x={5} y={chartData.padding + 4} fill="rgb(156, 163, 175)" fontSize="8">
          ${chartData.maxEquity.toFixed(0)}
        </text>
        <text x={5} y={chartData.height - chartData.padding + 4} fill="rgb(156, 163, 175)" fontSize="8">
          ${chartData.minEquity.toFixed(0)}
        </text>

        <path
          d={`${chartData.pathD} L ${chartData.points[chartData.points.length - 1].x} ${chartData.height - chartData.padding} L ${chartData.points[0].x} ${chartData.height - chartData.padding} Z`}
          fill="url(#equityGradient)"
        />

        <path
          d={chartData.pathD}
          fill="none"
          stroke="rgb(147, 51, 234)"
          strokeWidth="2"
        />

        {chartData.points.filter((_, i) => i % Math.max(1, Math.floor(chartData.points.length / 6)) === 0).map((p, i) => (
          <text key={i} x={p.x} y={chartData.height - 10} fill="rgb(156, 163, 175)" fontSize="7" textAnchor="middle">
            {formatTime(p.time)}
          </text>
        ))}
      </svg>

      <div className="mt-4 grid grid-cols-4 gap-4 text-sm">
        <div className="bg-gray-700/50 rounded p-3">
          <p className="text-xs text-gray-400">Toplam İşlem</p>
          <p className="text-white font-medium">{data.total_trades}</p>
        </div>
        <div className="bg-gray-700/50 rounded p-3">
          <p className="text-xs text-gray-400">Kazanan</p>
          <p className="text-green-400 font-medium">{data.winning_trades}</p>
        </div>
        <div className="bg-gray-700/50 rounded p-3">
          <p className="text-xs text-gray-400">Kaybeden</p>
          <p className="text-red-400 font-medium">{data.losing_trades}</p>
        </div>
        <div className="bg-gray-700/50 rounded p-3">
          <p className="text-xs text-gray-400">Win Rate</p>
          <p className="text-blue-400 font-medium">%{data.win_rate?.toFixed(1)}</p>
        </div>
      </div>
    </div>
  )
}
