"""
FastAPI Main Application - MEXC ML Trading System
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from typing import Dict, List
import asyncio
import json

from config import settings
from services.mexc_client import MEXCClient
from models.ml_engine import AdvancedMLEngine
from strategies.signal_manager import SignalManager
from utils.logger import logger

# Initialize FastAPI app
app = FastAPI(
    title="MEXC ML Trading System",
    description="Advanced Machine Learning powered futures trading system",
    version="4.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
mexc_client = MEXCClient()
ml_engine = AdvancedMLEngine()
signal_manager = SignalManager(ml_engine, mexc_client)

# WebSocket connections
active_connections: List[WebSocket] = []


async def broadcast_message(message: Dict):
    """Broadcast message to all connected clients"""
    for connection in active_connections[:]:
        try:
            await connection.send_json(message)
        except:
            active_connections.remove(connection)


@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    logger.info("Starting MEXC ML Trading System...")
    
    # Try to load existing models
    if ml_engine.load_model():
        logger.info("Models loaded successfully")
    else:
        logger.info("No pre-trained models found, will train on first request")
    
    logger.info(f"Server running on http://{settings.host}:{settings.port}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    await mexc_client.close()
    logger.info("Server shutting down")


# ============ REST API ENDPOINTS ============

@app.get("/")
async def root():
    return {
        "status": "online",
        "system": "MEXC ML Trading System",
        "version": "4.0.0",
        "features": [
            "48 advanced ML features",
            "Ensemble models (GBM + RF)",
            "Strict entry validation (75%+ confidence, $20M+ volume)",
            "Real-time market scanning",
            "Automatic position management"
        ]
    }


@app.get("/api/market/scan")
async def scan_market():
    """Scan all futures pairs and return signals"""
    try:
        signals = await signal_manager.scan_market()
        return {
            "success": True,
            "count": len(signals),
            "signals": signals
        }
    except Exception as e:
        logger.error(f"Market scan error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/market/tickers")
async def get_tickers():
    """Get all futures tickers"""
    try:
        tickers = await mexc_client.get_futures_tickers()
        return {
            "success": True,
            "count": len(tickers),
            "tickers": tickers
        }
    except Exception as e:
        logger.error(f"Get tickers error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ml/train/{symbol}")
async def train_model(symbol: str):
    """Train ML model for a specific symbol"""
    try:
        klines = await mexc_client.get_klines(symbol, '4h', limit=500)
        if not klines:
            return {"success": False, "error": "No data available"}
        
        result = ml_engine.train(klines, symbol)
        
        if result.get('success'):
            ml_engine.save_model()
            logger.info(f"Model trained for {symbol}: {result}")
        
        return result
    except Exception as e:
        logger.error(f"Train model error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ml/predict/{symbol}")
async def get_prediction(symbol: str):
    """Get ML prediction for a specific symbol"""
    try:
        # Get ticker info
        tickers = await mexc_client.get_futures_tickers()
        ticker = next((t for t in tickers if t.get('symbol') == symbol.upper()), None)
        
        if not ticker:
            raise HTTPException(status_code=404, detail="Symbol not found")
        
        # Get klines
        klines = await mexc_client.get_klines(symbol, '15m', limit=100)
        if len(klines) < 50:
            raise HTTPException(status_code=400, detail="Insufficient data")
        
        # Get prediction
        price = float(ticker.get('last', 0))
        volume_24h = float(ticker.get('volumeUSDT', 0))
        
        prediction = ml_engine.predict(symbol, klines, price, volume_24h)
        prediction['symbol'] = symbol
        
        return prediction
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/trading/positions")
async def get_positions():
    """Get current positions"""
    try:
        positions = await signal_manager.check_positions()
        return {
            "success": True,
            "count": len(positions),
            "positions": positions
        }
    except Exception as e:
        logger.error(f"Get positions error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/trading/execute/{symbol}")
async def execute_trade(symbol: str):
    """Execute a trade based on ML signal"""
    try:
        # Get current signal
        tickers = await mexc_client.get_futures_tickers()
        ticker = next((t for t in tickers if t.get('symbol') == symbol.upper()), None)
        
        if not ticker:
            raise HTTPException(status_code=404, detail="Symbol not found")
        
        klines = await mexc_client.get_klines(symbol, '15m', limit=100)
        price = float(ticker.get('last', 0))
        volume_24h = float(ticker.get('volumeUSDT', 0))
        
        signal = ml_engine.predict(symbol, klines, price, volume_24h)
        signal['symbol'] = symbol
        
        # Execute if signal is valid
        if signal.get('signal') != "WAIT":
            result = await signal_manager.execute_signal(signal)
            if result:
                await broadcast_message({
                    "type": "trade_executed",
                    "data": result
                })
                return {"success": True, "order": result}
        
        return {"success": False, "reason": signal.get('reason', 'Invalid signal')}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Execute trade error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/trading/close/{symbol}")
async def close_position(symbol: str):
    """Close a position"""
    try:
        result = await signal_manager.close_position(symbol.upper())
        if result:
            await broadcast_message({
                "type": "position_closed",
                "symbol": symbol.upper()
            })
            return {"success": True}
        return {"success": False, "reason": "Position not found"}
    except Exception as e:
        logger.error(f"Close position error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/account/info")
async def get_account_info():
    """Get account information"""
    try:
        info = await mexc_client.get_account_info()
        return {
            "success": True,
            "data": info
        }
    except Exception as e:
        logger.error(f"Get account info error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============ WEBSOCKET ENDPOINT ============

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time updates"""
    await websocket.accept()
    active_connections.append(websocket)
    
    logger.info(f"Client connected. Total connections: {len(active_connections)}")
    
    try:
        while True:
            # Keep connection alive
            data = await websocket.receive_text()
            
            # Handle client messages
            if data:
                msg = json.loads(data)
                msg_type = msg.get('type')
                
                if msg_type == 'subscribe':
                    await websocket.send_json({
                        "type": "subscribed",
                        "status": "success"
                    })
    
    except WebSocketDisconnect:
        active_connections.remove(websocket)
        logger.info(f"Client disconnected. Total connections: {len(active_connections)}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if websocket in active_connections:
            active_connections.remove(websocket)


# ============ BACKGROUND TASKS ============

async def periodic_scan():
    """Periodically scan market and broadcast updates"""
    while True:
        try:
            signals = await signal_manager.scan_market()
            
            # Broadcast top signals
            top_signals = [s for s in signals[:20] if s.get('signal') != 'WAIT']
            
            if top_signals:
                await broadcast_message({
                    "type": "market_update",
                    "data": {
                        "top_signals": top_signals,
                        "total_scanned": len(signals)
                    }
                })
            
            # Check positions
            positions = await signal_manager.check_positions()
            if positions:
                await broadcast_message({
                    "type": "position_update",
                    "data": positions
                })
            
        except Exception as e:
            logger.error(f"Periodic scan error: {e}")
        
        await asyncio.sleep(60)  # Scan every minute


@app.on_event("startup")
async def start_background_tasks():
    """Start background tasks"""
    asyncio.create_task(periodic_scan())


# Serve frontend (for production)
# Uncomment when frontend is built
# app.mount("/", StaticFiles(directory="../frontend/dist", html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level="info" if settings.debug else "warning"
    )
