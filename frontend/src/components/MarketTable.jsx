import React, { useState, useEffect } from 'react';
import { marketAPI, wsClient } from '../utils/api';
import { RefreshCw, TrendingUp, DollarSign, Activity } from 'lucide-react';

const MarketTable = ({ onSelectSymbol }) => {
  const [signals, setSignals] = useState([]);
  const [loading, setLoading] = useState(false);
  const [lastUpdate, setLastUpdate] = useState(null);

  const loadSignals = async () => {
    setLoading(true);
    try {
      const response = await marketAPI.scan();
      if (response.data.success) {
        setSignals(response.data.signals);
        setLastUpdate(new Date());
      }
    } catch (error) {
      console.error('Error loading signals:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadSignals();

    // WebSocket for real-time updates
    wsClient.connect();
    wsClient.subscribe();

    const handleMarketUpdate = (data) => {
      if (data.type === 'market_update' && data.data) {
        setSignals(data.data.top_signals || []);
        setLastUpdate(new Date());
      }
    };

    wsClient.on('market_update', handleMarketUpdate);

    // Auto-refresh every 30 seconds
    const interval = setInterval(loadSignals, 30000);

    return () => {
      clearInterval(interval);
      wsClient.off('market_update', handleMarketUpdate);
    };
  }, []);

  const getSignalClass = (signal) => {
    if (signal === 'LONG') return 'signal-long';
    if (signal === 'SHORT') return 'signal-short';
    return 'signal-wait';
  };

  const getVolumeClass = (volumeOk) => {
    return volumeOk ? 'volume-ok' : 'volume-low';
  };

  const getConfidenceClass = (confidence) => {
    if (confidence >= 0.85) return 'confidence-high';
    if (confidence >= 0.75) return 'confidence-medium';
    return 'confidence-low';
  };

  const formatVolume = (volume) => {
    if (volume >= 1e9) return `$${(volume / 1e9).toFixed(2)}B`;
    if (volume >= 1e6) return `$${(volume / 1e6).toFixed(1)}M`;
    return `$${volume.toFixed(0)}`;
  };

  const filteredSignals = signals.filter(s => s.signal !== 'WAIT' || s.confidence > 0.5);

  return (
    <div className="space-y-4">
      {/* Header Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="card">
          <div className="flex items-center gap-3">
            <div className="p-3 bg-blue-500/10 rounded-lg">
              <Activity className="w-6 h-6 text-blue-500" />
            </div>
            <div>
              <p className="text-sm text-gray-400">Top Signals</p>
              <p className="text-2xl font-bold">{filteredSignals.length}</p>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="flex items-center gap-3">
            <div className="p-3 bg-green-500/10 rounded-lg">
              <TrendingUp className="w-6 h-6 text-green-500" />
            </div>
            <div>
              <p className="text-sm text-gray-400">Long Signals</p>
              <p className="text-2xl font-bold">
                {filteredSignals.filter(s => s.signal === 'LONG').length}
              </p>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="flex items-center gap-3">
            <div className="p-3 bg-red-500/10 rounded-lg">
              <TrendingUp className="w-6 h-6 text-red-500 rotate-180" />
            </div>
            <div>
              <p className="text-sm text-gray-400">Short Signals</p>
              <p className="text-2xl font-bold">
                {filteredSignals.filter(s => s.signal === 'SHORT').length}
              </p>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="flex items-center gap-3">
            <div className="p-3 bg-yellow-500/10 rounded-lg">
              <DollarSign className="w-6 h-6 text-yellow-500" />
            </div>
            <div>
              <p className="text-sm text-gray-400">Avg Confidence</p>
              <p className="text-2xl font-bold">
                {filteredSignals.length > 0 
                  ? `${((filteredSignals.reduce((sum, s) => sum + s.confidence, 0) / filteredSignals.length) * 100).toFixed(1)}%`
                  : '0%'
                }
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Table Controls */}
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-xl font-bold">Market Scanner</h2>
          <p className="text-sm text-gray-400">
            {lastUpdate ? `Updated: ${lastUpdate.toLocaleTimeString()}` : 'Loading...'}
          </p>
        </div>
        <button 
          onClick={loadSignals} 
          className="btn btn-outline"
          disabled={loading}
        >
          <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
          Refresh
        </button>
      </div>

      {/* Market Table */}
      <div className="card overflow-hidden">
        <div className="overflow-x-auto">
          <table className="trading-table">
            <thead>
              <tr>
                <th>Symbol</th>
                <th>Price</th>
                <th>24h Change</th>
                <th>Volume (24h)</th>
                <th>ML Signal</th>
                <th>Confidence</th>
                <th>Leverage</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {filteredSignals.length === 0 ? (
                <tr>
                  <td colSpan="8" className="text-center py-8 text-gray-400">
                    {loading ? 'Scanning market...' : 'No signals found meeting criteria'}
                  </td>
                </tr>
              ) : (
                filteredSignals.map((signal, index) => (
                  <tr key={`${signal.symbol}-${index}`} className="hover:bg-white/5">
                    <td className="font-semibold">{signal.symbol}</td>
                    <td className="font-mono">${signal.price?.toFixed(signal.price < 1 ? 6 : 2)}</td>
                    <td className={signal.change_24h >= 0 ? 'price-up' : 'price-down'}>
                      {signal.change_24h >= 0 ? '+' : ''}{signal.change_24h?.toFixed(2)}%
                    </td>
                    <td>
                      <span className={`volume-badge ${getVolumeClass(signal.volume_ok)}`}>
                        {formatVolume(signal.volume_24h)}
                        {signal.volume_ok ? ' ✓' : ' ✗'}
                      </span>
                    </td>
                    <td>
                      <span className={`signal-badge ${getSignalClass(signal.signal)}`}>
                        {signal.signal === 'LONG' && '📈 '}
                        {signal.signal === 'SHORT' && '📉 '}
                        {signal.signal}
                      </span>
                    </td>
                    <td>
                      <div className="confidence-meter">
                        <div className="confidence-bar">
                          <div 
                            className={`confidence-fill ${getConfidenceClass(signal.confidence)}`}
                            style={{ width: `${signal.confidence * 100}%` }}
                          />
                        </div>
                        <span className="text-sm font-mono w-12 text-right">
                          {(signal.confidence * 100).toFixed(0)}%
                        </span>
                      </div>
                    </td>
                    <td className="font-mono">{signal.leverage}x</td>
                    <td>
                      <button 
                        onClick={() => onSelectSymbol(signal)}
                        className="btn btn-primary px-3 py-1 text-sm"
                      >
                        Analyze
                      </button>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Info Banner */}
      <div className="card bg-blue-500/10 border-blue-500/20">
        <div className="flex items-start gap-3">
          <div className="p-2 bg-blue-500/20 rounded-lg">
            <Activity className="w-5 h-5 text-blue-400" />
          </div>
          <div>
            <h3 className="font-semibold text-blue-400">Trading Criteria</h3>
            <p className="text-sm text-gray-300 mt-1">
              Only coins with <strong>$20M+ 24h volume</strong> and <strong>75%+ ML confidence</strong> 
              with model agreement are recommended for trading. Leverage is automatically adjusted 
              based on confidence and volume.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default MarketTable;
