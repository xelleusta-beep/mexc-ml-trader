import React, { useState, useEffect } from 'react';
import { tradingAPI, wsClient } from '../utils/api';
import { X, TrendingUp, DollarSign, Clock, AlertTriangle } from 'lucide-react';

const ActiveTrades = () => {
  const [positions, setPositions] = useState([]);
  const [loading, setLoading] = useState(false);

  const loadPositions = async () => {
    setLoading(true);
    try {
      const response = await tradingAPI.getPositions();
      if (response.data.success) {
        setPositions(response.data.positions);
      }
    } catch (error) {
      console.error('Error loading positions:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadPositions();

    // WebSocket for real-time updates
    const handlePositionUpdate = (data) => {
      if (data.type === 'position_update' && data.data) {
        setPositions(data.data);
      }
    };

    wsClient.on('position_update', handlePositionUpdate);

    const interval = setInterval(loadPositions, 10000);

    return () => {
      clearInterval(interval);
      wsClient.off('position_update', handlePositionUpdate);
    };
  }, []);

  const handleClosePosition = async (symbol) => {
    if (!confirm(`Are you sure you want to close ${symbol} position?`)) {
      return;
    }

    try {
      const response = await tradingAPI.close(symbol);
      if (response.data.success) {
        loadPositions();
      } else {
        alert('Failed to close position: ' + response.data.reason);
      }
    } catch (error) {
      console.error('Error closing position:', error);
      alert('Error closing position');
    }
  };

  const totalPnl = positions.reduce((sum, pos) => sum + (pos.pnl || 0), 0);

  return (
    <div className="space-y-4">
      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="card">
          <div className="flex items-center gap-3">
            <div className="p-3 bg-blue-500/10 rounded-lg">
              <Clock className="w-6 h-6 text-blue-500" />
            </div>
            <div>
              <p className="text-sm text-gray-400">Active Positions</p>
              <p className="text-2xl font-bold">{positions.length}</p>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="flex items-center gap-3">
            <div className={`p-3 rounded-lg ${totalPnl >= 0 ? 'bg-green-500/10' : 'bg-red-500/10'}`}>
              <DollarSign className={`w-6 h-6 ${totalPnl >= 0 ? 'text-green-500' : 'text-red-500'}`} />
            </div>
            <div>
              <p className="text-sm text-gray-400">Total P&L</p>
              <p className={`text-2xl font-bold ${totalPnl >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                ${totalPnl.toFixed(2)}
              </p>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="flex items-center gap-3">
            <div className="p-3 bg-yellow-500/10 rounded-lg">
              <AlertTriangle className="w-6 h-6 text-yellow-500" />
            </div>
            <div>
              <p className="text-sm text-gray-400">Risk Level</p>
              <p className="text-2xl font-bold">
                {positions.length === 0 ? 'Low' : positions.length <= 3 ? 'Medium' : 'High'}
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Positions Table */}
      <div className="card">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-xl font-bold">Active Positions</h2>
          <button onClick={loadPositions} className="btn btn-outline">
            Refresh
          </button>
        </div>

        {loading ? (
          <div className="text-center py-8 text-gray-400">Loading positions...</div>
        ) : positions.length === 0 ? (
          <div className="text-center py-12 text-gray-400">
            <TrendingUp className="w-16 h-16 mx-auto mb-4 opacity-50" />
            <p className="text-lg">No active positions</p>
            <p className="text-sm mt-2">Open trades from the Market Scanner</p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="trading-table">
              <thead>
                <tr>
                  <th>Symbol</th>
                  <th>Side</th>
                  <th>Entry Price</th>
                  <th>Current Price</th>
                  <th>Size</th>
                  <th>Leverage</th>
                  <th>P&L</th>
                  <th>Confidence</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {positions.map((pos, index) => (
                  <tr key={`${pos.symbol}-${index}`}>
                    <td className="font-semibold">{pos.symbol}</td>
                    <td>
                      <span className={`signal-badge ${pos.side === 'BUY' ? 'signal-long' : 'signal-short'}`}>
                        {pos.side === 'BUY' ? '📈 LONG' : '📉 SHORT'}
                      </span>
                    </td>
                    <td className="font-mono">${pos.entry_price?.toFixed(pos.entry_price < 1 ? 6 : 2)}</td>
                    <td className="font-mono">${pos.current_price?.toFixed(pos.current_price < 1 ? 6 : 2)}</td>
                    <td className="font-mono">{pos.size?.toFixed(4)}</td>
                    <td className="font-mono">{pos.leverage}x</td>
                    <td className={`font-mono font-bold ${pos.pnl >= 0 ? 'price-up' : 'price-down'}`}>
                      ${pos.pnl?.toFixed(2)}
                    </td>
                    <td>
                      <div className="confidence-meter">
                        <div className="confidence-bar">
                          <div 
                            className="confidence-fill confidence-high"
                            style={{ width: `${(pos.confidence || 0) * 100}%` }}
                          />
                        </div>
                        <span className="text-sm font-mono w-12 text-right">
                          {((pos.confidence || 0) * 100).toFixed(0)}%
                        </span>
                      </div>
                    </td>
                    <td>
                      <button 
                        onClick={() => handleClosePosition(pos.symbol)}
                        className="btn btn-danger px-3 py-1 text-sm"
                      >
                        <X className="w-4 h-4" />
                        Close
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Risk Warning */}
      {positions.length > 0 && (
        <div className="card bg-yellow-500/10 border-yellow-500/20">
          <div className="flex items-start gap-3">
            <div className="p-2 bg-yellow-500/20 rounded-lg">
              <AlertTriangle className="w-5 h-5 text-yellow-400" />
            </div>
            <div>
              <h3 className="font-semibold text-yellow-400">Risk Management</h3>
              <ul className="text-sm text-gray-300 mt-2 space-y-1">
                <li>• Stop-loss and take-profit are automatically set at 2% and 4%</li>
                <li>• Monitor your positions regularly</li>
                <li>• Consider closing positions if market conditions change</li>
                <li>• Never risk more than you can afford to lose</li>
              </ul>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ActiveTrades;
