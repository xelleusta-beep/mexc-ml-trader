import { useState } from 'react'

const TIMEFRAMES = [
  { value: '5m', label: '5 Dakika' },
  { value: '15m', label: '15 Dakika' },
  {value: '30m', label: '30 Dakika' },
  { value: '1h', label: '1 Saat' },
  { value: '4h', label: '4 Saat' },
  { value: '8h', label: '8 Saat' },
  { value: '1D', label: '1 Gün' },
]

const DEFAULT_PARAMS = {
  rsi_period: 21,
  rsi_ma_period: 21,
  entry_threshold: 30,
  exit_threshold_2: 50,
  exit_threshold_3: 70,
  exit_pct_1: 25,
  exit_pct_2: 25,
  exit_pct_3: 50,
  entry_amount: 1000,
  dca_30_amount: 3000,
  dca_60_amount: 6000,
  dca_30_drop: 30,
  dca_60_drop: 60,
  initial_capital: 10000,
  maker_fee: 0.02,
  taker_fee: 0.06,
  strategy_mode: 'rsi',
  ema_fast_period: 20,
  ema_slow_period: 50,
  timeframe: '1D',
  use_adx_filter: false,
  adx_threshold: 25,
  use_bb_filter: false,
  use_macd_filter: false,
  use_volume_filter: false,
  use_stochrsi_filter: false,
  use_trailing_stop: false,
  trailing_stop_pct: 5,
  use_atr_stop: false,
  atr_multiplier: 2,
}

export default function BacktestForm({ symbols, onSubmit, loading }) {
  const [selectedSymbols, setSelectedSymbols] = useState([])
  const [searchTerm, setSearchTerm] = useState('')
  const [params, setParams] = useState(DEFAULT_PARAMS)
  const [selectAll, setSelectAll] = useState(false)

  const filteredSymbols = symbols.filter(s =>
    s.symbol.toLowerCase().includes(searchTerm.toLowerCase()) ||
    s.baseCoin.toLowerCase().includes(searchTerm.toLowerCase())
  )

  const handleSymbolToggle = (symbol) => {
    setSelectedSymbols(prev =>
      prev.includes(symbol)
        ? prev.filter(s => s !== symbol)
        : [...prev, symbol]
    )
  }

  const handleSelectAll = () => {
    if (selectAll) {
      setSelectedSymbols([])
    } else {
      setSelectedSymbols(filteredSymbols.map(s => s.symbol))
    }
    setSelectAll(!selectAll)
  }

  const handleSubmit = (e) => {
    e.preventDefault()
    if (selectedSymbols.length === 0) return

    onSubmit({
      symbols: selectedSymbols,
      ...params,
      dca_30_drop: params.dca_30_drop / 100,
      dca_60_drop: params.dca_60_drop / 100,
      maker_fee: params.maker_fee / 100,
      taker_fee: params.taker_fee / 100,
      trailing_stop_pct: params.trailing_stop_pct / 100,
    })
  }

  return (
    <form onSubmit={handleSubmit} className="bg-gray-800 rounded-lg p-6 space-y-6">
      <h2 className="text-xl font-semibold text-purple-300">Backtest Ayarları</h2>

      <div>
        <label className="block text-sm text-gray-400 mb-2">Coin Seçimi</label>
        <input
          type="text"
          placeholder="Coin ara..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          className="w-full bg-gray-700 rounded px-3 py-2 text-sm mb-2"
        />
        <div className="flex items-center gap-2 mb-2">
          <input
            type="checkbox"
            checked={selectAll}
            onChange={handleSelectAll}
            className="rounded"
          />
          <span className="text-sm text-gray-400">
            Tümünü seç ({filteredSymbols.length} coin)
          </span>
        </div>
        <div className="max-h-48 overflow-y-auto bg-gray-700 rounded p-2 space-y-1">
          {filteredSymbols.map(s => (
            <label key={s.symbol} className="flex items-center gap-2 text-sm hover:bg-gray-600 rounded px-2 py-1 cursor-pointer">
              <input
                type="checkbox"
                checked={selectedSymbols.includes(s.symbol)}
                onChange={() => handleSymbolToggle(s.symbol)}
                className="rounded"
              />
              <span className="text-white">{s.symbol}</span>
              <span className="text-gray-400 text-xs">{s.baseCoin}</span>
            </label>
          ))}
        </div>
        <p className="text-xs text-gray-500 mt-1">{selectedSymbols.length} coin seçildi</p>
      </div>

      {/* Zaman Dilimi */}
      <div className="border-t border-gray-700 pt-4">
        <h3 className="text-sm font-medium text-gray-300 mb-3">Zaman Dilimi</h3>
        <div className="grid grid-cols-4 gap-2">
          {TIMEFRAMES.map(tf => (
            <button
              key={tf.value}
              type="button"
              onClick={() => setParams({ ...params, timeframe: tf.value })}
              className={`p-2 rounded-lg border text-center transition ${
                params.timeframe === tf.value
                  ? 'bg-purple-900/50 border-purple-500 text-purple-300'
                  : 'bg-gray-700 border-gray-600 text-gray-300 hover:border-gray-500'
              }`}
            >
              <p className="font-medium text-sm">{tf.value}</p>
              <p className="text-xs text-gray-400">{tf.label}</p>
            </button>
          ))}
        </div>
      </div>

      {/* Strateji Modu */}
      <div className="border-t border-gray-700 pt-4">
        <h3 className="text-sm font-medium text-gray-300 mb-3">Strateji Modu</h3>
        <div className="grid grid-cols-3 gap-2">
          {[
            { value: 'rsi', label: 'RSI', desc: 'Sadece RSI indikatörü' },
            { value: 'trend', label: 'Trend', desc: 'Sadece EMA crossover' },
            { value: 'combined', label: 'Hibrit', desc: 'RSI + Trend birlikte' },
          ].map(mode => (
            <button
              key={mode.value}
              type="button"
              onClick={() => setParams({ ...params, strategy_mode: mode.value })}
              className={`p-3 rounded-lg border text-left transition ${
                params.strategy_mode === mode.value
                  ? 'bg-purple-900/50 border-purple-500 text-purple-300'
                  : 'bg-gray-700 border-gray-600 text-gray-300 hover:border-gray-500'
              }`}
            >
              <p className="font-medium text-sm">{mode.label}</p>
              <p className="text-xs text-gray-400 mt-1">{mode.desc}</p>
            </button>
          ))}
        </div>
      </div>

      {/* Trend Ayarları (sadece trend/combined modunda) */}
      {params.strategy_mode !== 'rsi' && (
        <div className="border-t border-gray-700 pt-4">
          <h3 className="text-sm font-medium text-gray-300 mb-3">Trend (EMA) Ayarları</h3>
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-xs text-gray-400">Hızlı EMA</label>
              <input
                type="number"
                value={params.ema_fast_period}
                onChange={(e) => setParams({ ...params, ema_fast_period: +e.target.value })}
                className="w-full bg-gray-700 rounded px-3 py-2 text-sm"
              />
            </div>
            <div>
              <label className="block text-xs text-gray-400">Yavaş EMA</label>
              <input
                type="number"
                value={params.ema_slow_period}
                onChange={(e) => setParams({ ...params, ema_slow_period: +e.target.value })}
                className="w-full bg-gray-700 rounded px-3 py-2 text-sm"
              />
            </div>
          </div>
        </div>
      )}

      {/* RSI Ayarları (sadece rsi/combined modunda) */}
      {params.strategy_mode !== 'trend' && (
        <div className="border-t border-gray-700 pt-4">
          <h3 className="text-sm font-medium text-gray-300 mb-3">RSI Ayarları</h3>
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-xs text-gray-400">RSI Periyodu</label>
              <input
                type="number"
                value={params.rsi_period}
                onChange={(e) => setParams({ ...params, rsi_period: +e.target.value })}
                className="w-full bg-gray-700 rounded px-3 py-2 text-sm"
              />
            </div>
            <div>
              <label className="block text-xs text-gray-400">RSI MA Periyodu</label>
              <input
                type="number"
                value={params.rsi_ma_period}
                onChange={(e) => setParams({ ...params, rsi_ma_period: +e.target.value })}
                className="w-full bg-gray-700 rounded px-3 py-2 text-sm"
              />
            </div>
          </div>
        </div>
      )}

      <div className="border-t border-gray-700 pt-4">
        <h3 className="text-sm font-medium text-gray-300 mb-3">Giriş/Çıkış Eşikleri</h3>
        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className="block text-xs text-gray-400">Giriş Eşiği (RSI MA)</label>
            <input
              type="number"
              value={params.entry_threshold}
              onChange={(e) => setParams({ ...params, entry_threshold: +e.target.value })}
              className="w-full bg-gray-700 rounded px-3 py-2 text-sm"
            />
          </div>
          <div>
            <label className="block text-xs text-gray-400">Çıkış 1 (%50)</label>
            <input
              type="number"
              value={params.exit_threshold_2}
              onChange={(e) => setParams({ ...params, exit_threshold_2: +e.target.value })}
              className="w-full bg-gray-700 rounded px-3 py-2 text-sm"
            />
          </div>
          <div>
            <label className="block text-xs text-gray-400">Çıkış 2 (Tümü)</label>
            <input
              type="number"
              value={params.exit_threshold_3}
              onChange={(e) => setParams({ ...params, exit_threshold_3: +e.target.value })}
              className="w-full bg-gray-700 rounded px-3 py-2 text-sm"
            />
          </div>
        </div>
      </div>

      <div className="border-t border-gray-700 pt-4">
        <h3 className="text-sm font-medium text-gray-300 mb-3">İşlem Miktarları</h3>
        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className="block text-xs text-gray-400">Giriş ($)</label>
            <input
              type="number"
              value={params.entry_amount}
              onChange={(e) => setParams({ ...params, entry_amount: +e.target.value })}
              className="w-full bg-gray-700 rounded px-3 py-2 text-sm"
            />
          </div>
          <div>
            <label className="block text-xs text-gray-400">DCA %30 ($)</label>
            <input
              type="number"
              value={params.dca_30_amount}
              onChange={(e) => setParams({ ...params, dca_30_amount: +e.target.value })}
              className="w-full bg-gray-700 rounded px-3 py-2 text-sm"
            />
          </div>
          <div>
            <label className="block text-xs text-gray-400">DCA %60 ($)</label>
            <input
              type="number"
              value={params.dca_60_amount}
              onChange={(e) => setParams({ ...params, dca_60_amount: +e.target.value })}
              className="w-full bg-gray-700 rounded px-3 py-2 text-sm"
            />
          </div>
          <div>
            <label className="block text-xs text-gray-400">Başlangıç Sermayesi ($)</label>
            <input
              type="number"
              value={params.initial_capital}
              onChange={(e) => setParams({ ...params, initial_capital: +e.target.value })}
              className="w-full bg-gray-700 rounded px-3 py-2 text-sm"
            />
          </div>
        </div>
      </div>

      <div className="border-t border-gray-700 pt-4">
        <h3 className="text-sm font-medium text-gray-300 mb-3">DCA Düşüş Eşikleri (%)</h3>
        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className="block text-xs text-gray-400">DCA %30 Düşüş</label>
            <input
              type="number"
              value={params.dca_30_drop}
              onChange={(e) => setParams({ ...params, dca_30_drop: +e.target.value })}
              className="w-full bg-gray-700 rounded px-3 py-2 text-sm"
            />
          </div>
          <div>
            <label className="block text-xs text-gray-400">DCA %60 Düşüş</label>
            <input
              type="number"
              value={params.dca_60_drop}
              onChange={(e) => setParams({ ...params, dca_60_drop: +e.target.value })}
              className="w-full bg-gray-700 rounded px-3 py-2 text-sm"
            />
          </div>
        </div>
      </div>

      <div className="border-t border-gray-700 pt-4">
        <h3 className="text-sm font-medium text-gray-300 mb-3">Komisyon Oranları (%)</h3>
        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className="block text-xs text-gray-400">Maker</label>
            <input
              type="number"
              step="0.01"
              value={params.maker_fee}
              onChange={(e) => setParams({ ...params, maker_fee: +e.target.value })}
              className="w-full bg-gray-700 rounded px-3 py-2 text-sm"
            />
          </div>
          <div>
            <label className="block text-xs text-gray-400">Taker</label>
            <input
              type="number"
              step="0.01"
              value={params.taker_fee}
              onChange={(e) => setParams({ ...params, taker_fee: +e.target.value })}
              className="w-full bg-gray-700 rounded px-3 py-2 text-sm"
            />
          </div>
        </div>
      </div>

      {/* Profesyonel Filtreler */}
      <div className="border-t border-gray-700 pt-4">
        <h3 className="text-sm font-medium text-gray-300 mb-3">Profesyonel Filtreler</h3>
        <div className="space-y-3">
          <label className="flex items-center gap-3 p-2 rounded hover:bg-gray-700 cursor-pointer">
            <input
              type="checkbox"
              checked={params.use_adx_filter}
              onChange={(e) => setParams({ ...params, use_adx_filter: e.target.checked })}
              className="rounded"
            />
            <div>
              <span className="text-sm text-white">ADX Filtresi</span>
              <span className="text-xs text-gray-400 ml-2">(Trend gücü &gt; {params.adx_threshold})</span>
            </div>
            {params.use_adx_filter && (
              <input
                type="number"
                value={params.adx_threshold}
                onChange={(e) => setParams({ ...params, adx_threshold: +e.target.value })}
                className="w-16 bg-gray-700 rounded px-2 py-1 text-xs ml-auto"
              />
            )}
          </label>

          <label className="flex items-center gap-3 p-2 rounded hover:bg-gray-700 cursor-pointer">
            <input
              type="checkbox"
              checked={params.use_bb_filter}
              onChange={(e) => setParams({ ...params, use_bb_filter: e.target.checked })}
              className="rounded"
            />
            <span className="text-sm text-white">Bollinger Bands Filtresi</span>
            <span className="text-xs text-gray-400 ml-2">(Fiyat lower bandın altında)</span>
          </label>

          <label className="flex items-center gap-3 p-2 rounded hover:bg-gray-700 cursor-pointer">
            <input
              type="checkbox"
              checked={params.use_macd_filter}
              onChange={(e) => setParams({ ...params, use_macd_filter: e.target.checked })}
              className="rounded"
            />
            <span className="text-sm text-white">MACD Filtresi</span>
              <span className="text-xs text-gray-400 ml-2">(MACD &gt; Signal)</span>
          </label>

          <label className="flex items-center gap-3 p-2 rounded hover:bg-gray-700 cursor-pointer">
            <input
              type="checkbox"
              checked={params.use_volume_filter}
              onChange={(e) => setParams({ ...params, use_volume_filter: e.target.checked })}
              className="rounded"
            />
            <span className="text-sm text-white">Volume Filtresi</span>
            <span className="text-xs text-gray-400 ml-2">(Hacim ortalamadan yüksek)</span>
          </label>

          <label className="flex items-center gap-3 p-2 rounded hover:bg-gray-700 cursor-pointer">
            <input
              type="checkbox"
              checked={params.use_stochrsi_filter}
              onChange={(e) => setParams({ ...params, use_stochrsi_filter: e.target.checked })}
              className="rounded"
            />
            <span className="text-sm text-white">Stochastic RSI Filtresi</span>
            <span className="text-xs text-gray-400 ml-2">(Aşırı satım bölgesi)</span>
          </label>
        </div>
      </div>

      {/* Dinamik Stop Loss */}
      <div className="border-t border-gray-700 pt-4">
        <h3 className="text-sm font-medium text-gray-300 mb-3">Dinamik Stop Loss</h3>
        <div className="space-y-3">
          <label className="flex items-center gap-3 p-2 rounded hover:bg-gray-700 cursor-pointer">
            <input
              type="checkbox"
              checked={params.use_trailing_stop}
              onChange={(e) => setParams({ ...params, use_trailing_stop: e.target.checked })}
              className="rounded"
            />
            <div>
              <span className="text-sm text-white">Trailing Stop</span>
              <span className="text-xs text-gray-400 ml-2">(Zirveden %{params.trailing_stop_pct} düşüşte çık)</span>
            </div>
            {params.use_trailing_stop && (
              <input
                type="number"
                step="0.5"
                value={params.trailing_stop_pct}
                onChange={(e) => setParams({ ...params, trailing_stop_pct: +e.target.value })}
                className="w-16 bg-gray-700 rounded px-2 py-1 text-xs ml-auto"
              />
            )}
          </label>

          <label className="flex items-center gap-3 p-2 rounded hover:bg-gray-700 cursor-pointer">
            <input
              type="checkbox"
              checked={params.use_atr_stop}
              onChange={(e) => setParams({ ...params, use_atr_stop: e.target.checked })}
              className="rounded"
            />
            <div>
              <span className="text-sm text-white">ATR Tabanlı Stop</span>
              <span className="text-xs text-gray-400 ml-2">(Giriş - {params.atr_multiplier}x ATR)</span>
            </div>
            {params.use_atr_stop && (
              <input
                type="number"
                step="0.5"
                value={params.atr_multiplier}
                onChange={(e) => setParams({ ...params, atr_multiplier: +e.target.value })}
                className="w-16 bg-gray-700 rounded px-2 py-1 text-xs ml-auto"
              />
            )}
          </label>
        </div>
      </div>

      <button
        type="submit"
        disabled={loading || selectedSymbols.length === 0}
        className="w-full bg-purple-600 hover:bg-purple-700 disabled:bg-gray-600 disabled:cursor-not-allowed rounded py-3 font-medium transition"
      >
        {loading ? 'Backtest Çalıştırılıyor...' : `Backtest Başlat (${selectedSymbols.length} coin)`}
      </button>
    </form>
  )
}
