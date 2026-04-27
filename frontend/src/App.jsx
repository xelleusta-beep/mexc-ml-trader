import React, { useState } from 'react';
import MarketTable from './components/MarketTable';
import ActiveTrades from './components/ActiveTrades';
import MLAnalysis from './components/MLAnalysis';
import { wsClient } from './utils/api';
import { Activity, TrendingUp, Brain } from 'lucide-react';

function App() {
  const [activeTab, setActiveTab] = useState('market');
  const [selectedSymbol, setSelectedSymbol] = useState(null);

  const handleSelectSymbol = (symbol) => {
    setSelectedSymbol(symbol);
    setActiveTab('analysis');
  };

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-card-border bg-card-bg/50 backdrop-blur-sm sticky top-0 z-50">
        <div className="container mx-auto px-4 py-4">
          <div className="flex justify-between items-center">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-blue-500/10 rounded-lg">
                <Activity className="w-8 h-8 text-blue-500" />
              </div>
              <div>
                <h1 className="text-2xl font-bold">MEXC ML Trading</h1>
                <p className="text-sm text-gray-400">AI-Powered Futures Trading System v4.0</p>
              </div>
            </div>
            
            <div className="flex items-center gap-4">
              <div className="hidden md:flex items-center gap-2 text-sm">
                <span className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></span>
                <span className="text-gray-400">System Online</span>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Navigation Tabs */}
      <nav className="container mx-auto px-4 py-4">
        <div className="tab-nav w-fit">
          <button
            onClick={() => setActiveTab('market')}
            className={`tab-btn ${activeTab === 'market' ? 'active' : ''}`}
          >
            <Activity className="w-4 h-4 inline mr-2" />
            Market Scanner
          </button>
          <button
            onClick={() => setActiveTab('trades')}
            className={`tab-btn ${activeTab === 'trades' ? 'active' : ''}`}
          >
            <TrendingUp className="w-4 h-4 inline mr-2" />
            Active Trades
          </button>
          <button
            onClick={() => setActiveTab('analysis')}
            className={`tab-btn ${activeTab === 'analysis' ? 'active' : ''}`}
          >
            <Brain className="w-4 h-4 inline mr-2" />
            ML Analysis
          </button>
        </div>
      </nav>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-6">
        {activeTab === 'market' && (
          <MarketTable onSelectSymbol={handleSelectSymbol} />
        )}
        
        {activeTab === 'trades' && (
          <ActiveTrades />
        )}
        
        {activeTab === 'analysis' && (
          <MLAnalysis selectedSymbol={selectedSymbol} />
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-card-border mt-12 py-6">
        <div className="container mx-auto px-4 text-center text-sm text-gray-400">
          <p>MEXC ML Trading System v4.0 - Professional AI-Powered Trading Platform</p>
          <p className="mt-2">
            ⚠️ Trading cryptocurrencies involves significant risk. Only trade with funds you can afford to lose.
          </p>
        </div>
      </footer>
    </div>
  );
}

export default App;
