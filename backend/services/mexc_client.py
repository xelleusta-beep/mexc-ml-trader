"""
MEXC Futures Client - Handles all API interactions
"""
import hashlib
import hmac
import time
from typing import Dict, List, Optional, Any
import httpx
from config import settings


class MEXCClient:
    def __init__(self, api_key: str = None, api_secret: str = None):
        self.api_key = api_key or settings.mexc_api_key
        self.api_secret = api_secret or settings.mexc_api_secret
        self.base_url = settings.mexc_base_url
        self.client = httpx.AsyncClient(timeout=30.0)
        
    def _generate_signature(self, params: Dict) -> str:
        """Generate HMAC SHA256 signature"""
        query_string = '&'.join([f"{k}={v}" for k, v in sorted(params.items())])
        return hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    async def get_futures_tickers(self) -> List[Dict]:
        """Get all futures tickers with 24h stats"""
        try:
            response = await self.client.get(
                f"{self.base_url}/api/v1/contract/ticker"
            )
            response.raise_for_status()
            data = response.json()
            if data.get('success'):
                return data.get('data', [])
            return []
        except Exception as e:
            print(f"Error fetching tickers: {e}")
            return []
    
    async def get_klines(self, symbol: str, interval: str, limit: int = 1000) -> List[List]:
        """Get candlestick data for a symbol"""
        try:
            params = {
                "symbol": symbol.upper(),
                "interval": interval,
                "limit": limit
            }
            response = await self.client.get(
                f"{self.base_url}/api/v1/contract/kline/{symbol.upper()}",
                params=params
            )
            response.raise_for_status()
            data = response.json()
            if data.get('success'):
                return data.get('data', [])
            return []
        except Exception as e:
            print(f"Error fetching klines for {symbol}: {e}")
            return []
    
    async def get_account_info(self) -> Optional[Dict]:
        """Get account information"""
        if not self.api_key or not self.api_secret:
            return None
        
        try:
            params = {"recvWindow": 5000, "timestamp": int(time.time() * 1000)}
            signature = self._generate_signature(params)
            params['signature'] = signature
            
            headers = {
                "ApiKey": self.api_key,
                "Request-Time": str(params['timestamp'])
            }
            
            response = await self.client.get(
                f"{self.base_url}/api/v1/account/assets",
                params=params,
                headers=headers
            )
            response.raise_for_status()
            data = response.json()
            return data.get('data') if data.get('success') else None
        except Exception as e:
            print(f"Error fetching account info: {e}")
            return None
    
    async def get_positions(self) -> List[Dict]:
        """Get current positions"""
        if not self.api_key or not self.api_secret:
            return []
        
        try:
            params = {"recvWindow": 5000, "timestamp": int(time.time() * 1000)}
            signature = self._generate_signature(params)
            params['signature'] = signature
            
            headers = {
                "ApiKey": self.api_key,
                "Request-Time": str(params['timestamp'])
            }
            
            response = await self.client.get(
                f"{self.base_url}/api/v1/contract/position/open",
                params=params,
                headers=headers
            )
            response.raise_for_status()
            data = response.json()
            return data.get('data', []) if data.get('success') else []
        except Exception as e:
            print(f"Error fetching positions: {e}")
            return []
    
    async def place_order(self, symbol: str, side: str, size: float, 
                         leverage: int = 10, price: Optional[float] = None) -> Optional[Dict]:
        """Place a futures order"""
        if not self.api_key or not self.api_secret:
            return None
        
        try:
            params = {
                "symbol": symbol.upper(),
                "side": side.upper(),
                "leverage": leverage,
                "vol": size,
                "type": "LIMIT" if price else "MARKET",
                "recvWindow": 5000,
                "timestamp": int(time.time() * 1000)
            }
            
            if price:
                params["price"] = str(price)
            
            signature = self._generate_signature(params)
            params['signature'] = signature
            
            headers = {
                "ApiKey": self.api_key,
                "Request-Time": str(params['timestamp']),
                "Content-Type": "application/x-www-form-urlencoded"
            }
            
            response = await self.client.post(
                f"{self.base_url}/api/v1/contract/order/submit",
                params=params,
                headers=headers
            )
            response.raise_for_status()
            data = response.json()
            return data.get('data') if data.get('success') else None
        except Exception as e:
            print(f"Error placing order: {e}")
            return None
    
    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel an order"""
        if not self.api_key or not self.api_secret:
            return False
        
        try:
            params = {
                "symbol": symbol.upper(),
                "ids": order_id,
                "recvWindow": 5000,
                "timestamp": int(time.time() * 1000)
            }
            signature = self._generate_signature(params)
            params['signature'] = signature
            
            headers = {
                "ApiKey": self.api_key,
                "Request-Time": str(params['timestamp'])
            }
            
            response = await self.client.request(
                "DELETE",
                f"{self.base_url}/api/v1/contract/order/cancel",
                params=params,
                headers=headers
            )
            response.raise_for_status()
            data = response.json()
            return data.get('success', False)
        except Exception as e:
            print(f"Error cancelling order: {e}")
            return False
    
    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()
