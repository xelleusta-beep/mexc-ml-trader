import React, { useState } from 'react';
import { mlAPI, tradingAPI } from '../utils/api';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { TrendingUp, Brain, Zap, Shield } from 'lucide-react';

const MLAnalysis = ({ selectedSymbol }) => {
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [training, setTraining] = useState(false);

  const loadPrediction = async () => {
    if (!selectedSymbol) return;
    
    setLoading(true);
    try {
      const response = await mlAPI.predict(selectedSymbol.symbol || selectedSymbol);
      setPrediction(response.data);
    } catch (error) {
      console.error('Error loading prediction:', error);
      alert('Error loading prediction');
    } finally {
      setLoading(false);
    }
  };

  const handleTrain = async () => {
    if (!selectedSymbol) return;
    
    setTraining(true);
    try {
      const symbol = selectedSymbol.symbol || selectedSymbol;
      const response = await mlAPI.train(symbol);
      
      if (response.data.success) {
        alert(`Model trained successfully!\nGBM Accuracy: ${(response.data.gbm_accuracy * 100).toFixed(2)}%\nRF Accuracy: ${(response.data.rf_accuracy * 100).toFixed(2)}%`);
        loadPrediction();
      } else {
        alert('Training failed: ' + response.data.error);
      }
    } catch (error) {
      console.error('Error training model:', error);
      alert('Error training model');
    } finally {
      setTraining(false);
    }
  };

  const handleExecute = async () => {
    if (!selectedSymbol) return;
    
    try {
      const symbol = selectedSymbol.symbol || selectedSymbol;
      const response = await tradingAPI.execute(symbol);
      
      if (response.data.success) {
        alert('Order executed successfully!');
      } else {
        alert('Order not executed: ' + response.data.reason);
      }
    } catch (error) {
      console.error('Error executing trade:', error);
      alert('Error executing trade');
    }
  };

  if (!selectedSymbol) {
    return (
      <div className="card text-center py-12">
        <Brain className="w-16 h-16 mx-auto mb-4 opacity-50" />
        <p className="text-lg text-gray-400">Select a symbol from Market Scanner to analyze</p>
      </div>
    );
  }

  const symbol = selectedSymbol.symbol || selectedSymbol;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-2xl font-bold">{symbol}</h2>
          <p className="text-gray-400">ML Analysis & Prediction</p>
        </div>
        <div className="flex gap-2">
          <button 
            onClick={handleTrain} 
            className="btn btn-outline"
            disabled={training}
          >
            <Brain className={`w-4 h-4 ${training ? 'animate-pulse' : ''}`} />
            {training ? 'Training...' : 'Train Model'}
          </button>
          <button 
            onClick={loadPrediction} 
            className="btn btn-outline"
            disabled={loading}
          >
            <Zap className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
            {loading ? 'Loading...' : 'Get Prediction'}
          </button>
        </div>
      </div>

      {prediction && (
        <>
          {/* Signal Overview */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="card">
              <div className="flex items-center gap-3">
                <div className={`p-3 rounded-lg ${
                  prediction.signal === 'LONG' ? 'bg-green-500/10' :
                  prediction.signal === 'SHORT' ? 'bg-red-500/10' : 'bg-gray-500/10'
                }`}>
                  <TrendingUp className={`w-6 h-6 ${
                    prediction.signal === 'LONG' ? 'text-green-500' :
                    prediction.signal === 'SHORT' ? 'text-red-500 rotate-180' : 'text-gray-500'
                  }`} />
                </div>
                <div>
                  <p className="text-sm text-gray-400">Signal</p>
                  <p className={`text-xl font-bold ${
                    prediction.signal === 'LONG' ? 'text-green-500' :
                    prediction.signal === 'SHORT' ? 'text-red-500' : 'text-gray-500'
                  }`}>
                    {prediction.signal}
                  </p>
                </div>
              </div>
            </div>

            <div className="card">
              <div className="flex items-center gap-3">
                <div className="p-3 bg-blue-500/10 rounded-lg">
                  <Brain className="w-6 h-6 text-blue-500" />
                </div>
                <div>
                  <p className="text-sm text-gray-400">Confidence</p>
                  <p className="text-xl font-bold">{(prediction.confidence * 100).toFixed(1)}%</p>
                </div>
              </div>
            </div>

            <div className="card">
              <div className="flex items-center gap-3">
                <div className="p-3 bg-yellow-500/10 rounded-lg">
                  <Shield className="w-6 h-6 text-yellow-500" />
                </div>
                <div>
                  <p className="text-sm text-gray-400">Recommended Leverage</p>
                  <p className="text-xl font-bold">{prediction.leverage}x</p>
                </div>
              </div>
            </div>

            <div className="card">
              <div className="flex items-center gap-3">
                <div className="p-3 bg-purple-500/10 rounded-lg">
                  <Zap className="w-6 h-6 text-purple-500" />
                </div>
                <div>
                  <p className="text-sm text-gray-400">Direction</p>
                  <p className="text-xl font-bold">{prediction.direction}</p>
                </div>
              </div>
            </div>
          </div>

          {/* Model Confidence Comparison */}
          <div className="card">
            <h3 className="text-lg font-bold mb-4">Model Confidence Comparison</h3>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={[
                { name: 'GBM', confidence: prediction.gbm_confidence },
                { name: 'Random Forest', confidence: prediction.rf_confidence },
                { name: 'Ensemble', confidence: prediction.confidence }
              ]}>
                <XAxis dataKey="name" stroke="#64748b" />
                <YAxis domain={[0, 1]} stroke="#64748b" />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: '#151a2d', 
                    border: '1px solid #2a3149',
                    borderRadius: '8px'
                  }}
                  formatter={(value) => `${(value * 100).toFixed(1)}%`}
                />
                <Bar dataKey="confidence" radius={[4, 4, 0, 0]}>
                  {[
                    { name: 'GBM' },
                    { name: 'Random Forest' },
                    { name: 'Ensemble' }
                  ].map((entry, index) => (
                    <Cell 
                      key={`cell-${index}`}
                      fill={
                        entry.name === 'Ensemble' ? '#3b82f6' :
                        prediction[entry.name === 'GBM' ? 'gbm_confidence' : 'rf_confidence'] >= 0.75 
                          ? '#10b981' 
                          : '#f59e0b'
                      }
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Validation Details */}
          <div className="card">
            <h3 className="text-lg font-bold mb-4">Validation Details</h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="p-4 bg-gray-800/50 rounded-lg">
                <p className="text-sm text-gray-400 mb-2">Volume Check</p>
                <div className="flex items-center gap-2">
                  <span className={`text-2xl ${prediction.volume_ok ? 'text-green-500' : 'text-red-500'}`}>
                    {prediction.volume_ok ? '✓' : '✗'}
                  </span>
                  <span className="font-mono">
                    ${(prediction.volume_24h_usd / 1e6).toFixed(1)}M / $20M
                  </span>
                </div>
              </div>

              <div className="p-4 bg-gray-800/50 rounded-lg">
                <p className="text-sm text-gray-400 mb-2">Model Agreement</p>
                <div className="flex items-center gap-2">
                  <span className={`text-2xl ${prediction.models_agree ? 'text-green-500' : 'text-red-500'}`}>
                    {prediction.models_agree ? '✓' : '✗'}
                  </span>
                  <span>
                    {prediction.models_agree ? 'Both models agree' : 'Models disagree'}
                  </span>
                </div>
              </div>

              <div className="p-4 bg-gray-800/50 rounded-lg">
                <p className="text-sm text-gray-400 mb-2">Confidence Threshold</p>
                <div className="flex items-center gap-2">
                  <span className={`text-2xl ${prediction.confidence >= 0.75 ? 'text-green-500' : 'text-red-500'}`}>
                    {prediction.confidence >= 0.75 ? '✓' : '✗'}
                  </span>
                  <span className="font-mono">
                    {(prediction.confidence * 100).toFixed(1)}% / 75%
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Reason */}
          <div className={`card ${
            prediction.signal !== 'WAIT' ? 'bg-green-500/10 border-green-500/20' : 
            'bg-yellow-500/10 border-yellow-500/20'
          }`}>
            <div className="flex items-start gap-3">
              <div className={`p-2 rounded-lg ${
                prediction.signal !== 'WAIT' ? 'bg-green-500/20' : 'bg-yellow-500/20'
              }`}>
                <Shield className={`w-5 h-5 ${
                  prediction.signal !== 'WAIT' ? 'text-green-400' : 'text-yellow-400'
                }`} />
              </div>
              <div>
                <h3 className={`font-semibold ${
                  prediction.signal !== 'WAIT' ? 'text-green-400' : 'text-yellow-400'
                }`}>
                  Decision Reason
                </h3>
                <p className="text-gray-300 mt-1">{prediction.reason}</p>
              </div>
            </div>
          </div>

          {/* Action Button */}
          {prediction.signal !== 'WAIT' && (
            <div className="flex justify-center">
              <button 
                onClick={handleExecute}
                className={`btn px-8 py-3 text-lg ${
                  prediction.signal === 'LONG' ? 'btn-success' : 'btn-danger'
                }`}
              >
                {prediction.signal === 'LONG' ? '📈 Open LONG' : '📉 Open SHORT'} 
                {' '}({prediction.leverage}x)
              </button>
            </div>
          )}
        </>
      )}

      {!prediction && !loading && (
        <div className="card text-center py-12">
          <Zap className="w-16 h-16 mx-auto mb-4 opacity-50" />
          <p className="text-gray-400">Click "Get Prediction" to analyze {symbol}</p>
        </div>
      )}
    </div>
  );
};

export default MLAnalysis;
