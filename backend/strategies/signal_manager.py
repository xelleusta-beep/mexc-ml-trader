"""
Signal Manager - Manages trading signals and execution logic
"""
from typing import Dict, List, Optional
from datetime import datetime
from models.ml_engine import AdvancedMLEngine
from services.mexc_client import MEXCClient
from config import settings


class SignalManager:
    def __init__(self, ml_engine: AdvancedMLEngine, mexc_client: MEXCClient):
        self.ml_engine = ml_engine
        self.mexc_client = mexc_client
        self.active_signals: Dict[str, Dict] = {}
        self.position_history: List[Dict] = []
        
    async def scan_market(self) -> List[Dict]:
        """Scan all futures pairs and generate signals"""
        tickers = await self.mexc_client.get_futures_tickers()
        results = []
        
        for ticker in tickers:
            symbol = ticker.get('symbol', '')
            if not symbol:
                continue
            
            # Get 24h volume in USD
            volume_24h = float(ticker.get('volumeUSDT', 0))
            
            # Skip low volume coins
            if volume_24h < settings.min_volume_24h_usd:
                continue
            
            # Get recent klines for analysis
            klines = await self.mexc_client.get_klines(symbol, '15m', limit=100)
            if len(klines) < 50:
                continue
            
            # Generate ML prediction
            current_price = float(ticker.get('last', 0))
            prediction = self.ml_engine.predict(
                symbol=symbol,
                klines=klines,
                price=current_price,
                volume_24h=volume_24h
            )
            
            # Add ticker info
            prediction['symbol'] = symbol
            prediction['price'] = current_price
            prediction['volume_24h'] = volume_24h
            prediction['change_24h'] = float(ticker.get('changeRate', 0)) * 100
            prediction['timestamp'] = datetime.utcnow().isoformat()
            
            results.append(prediction)
        
        # Sort by confidence
        results.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        
        return results
    
    async def execute_signal(self, signal: Dict) -> Optional[Dict]:
        """Execute a trading signal"""
        symbol = signal.get('symbol')
        sig_type = signal.get('signal')  # LONG or SHORT
        confidence = signal.get('confidence', 0)
        leverage = signal.get('leverage', 10)
        
        if sig_type == "WAIT":
            return None
        
        # Calculate position size
        position_size = settings.max_position_size_usd / signal.get('price', 1)
        
        # Place order
        side = "BUY" if sig_type == "LONG" else "SELL"
        order_result = await self.mexc_client.place_order(
            symbol=symbol,
            side=side,
            size=position_size,
            leverage=leverage
        )
        
        if order_result:
            # Store active signal
            self.active_signals[symbol] = {
                "symbol": symbol,
                "side": side,
                "entry_price": signal.get('price'),
                "size": position_size,
                "leverage": leverage,
                "confidence": confidence,
                "timestamp": datetime.utcnow().isoformat(),
                "stop_loss": signal.get('price') * (1 - settings.stop_loss_pct) if side == "BUY" else signal.get('price') * (1 + settings.stop_loss_pct),
                "take_profit": signal.get('price') * (1 + settings.take_profit_pct) if side == "BUY" else signal.get('price') * (1 - settings.take_profit_pct)
            }
            
            return self.active_signals[symbol]
        
        return None
    
    async def check_positions(self) -> List[Dict]:
        """Check and manage active positions"""
        positions = await self.mexc_client.get_positions()
        updated_positions = []
        
        for pos in positions:
            symbol = pos.get('symbol')
            if symbol in self.active_signals:
                signal_data = self.active_signals[symbol].copy()
                signal_data['current_price'] = float(pos.get('entryPrice', 0))
                signal_data['pnl'] = float(pos.get('unRealisedProfit', 0))
                updated_positions.append(signal_data)
        
        return updated_positions
    
    async def close_position(self, symbol: str) -> bool:
        """Close a position"""
        if symbol not in self.active_signals:
            return False
        
        position = self.active_signals[symbol]
        side = "SELL" if position['side'] == "BUY" else "BUY"
        
        result = await self.mexc_client.place_order(
            symbol=symbol,
            side=side,
            size=position['size'],
            leverage=position['leverage']
        )
        
        if result:
            # Move to history
            self.position_history.append(self.active_signals.pop(symbol))
            return True
        
        return False
