import ccxt
import pandas as pd
import asyncio
from typing import Dict, List, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class MexcDataFetcher:
    def __init__(self, config: dict):
        self.exchange = ccxt.mexc(config)
        self.timeframe = '5m'
        
    async def fetch_ohlcv(self, symbol: str, limit: int = 500) -> pd.DataFrame:
        """دریافت داده‌های OHLCV از صرافی"""
        try:
            ohlcv = await asyncio.to_thread(
                self.exchange.fetch_ohlcv,
                symbol,
                timeframe=self.timeframe,
                limit=limit
            )
            
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    async def fetch_tickers(self, symbols: List[str]) -> Dict:
        """دریافت اطلاعات لحظه‌ای ارزها"""
        tickers = {}
        for symbol in symbols:
            try:
                ticker = await asyncio.to_thread(
                    self.exchange.fetch_ticker,
                    symbol
                )
                tickers[symbol] = {
                    'last': ticker['last'],
                    'volume': ticker['quoteVolume'],
                    'change': ticker['percentage'],
                    'high': ticker['high'],
                    'low': ticker['low']
                }
            except Exception as e:
                logger.error(f"Error fetching ticker for {symbol}: {e}")
        
        return tickers
    
    async def get_top_volume_pairs(self, n: int = 100) -> List[str]:
        """دریافت جفت‌ارزهای با بالاترین حجم معاملات"""
        try:
            markets = self.exchange.load_markets()
            tickers = await asyncio.to_thread(self.exchange.fetch_tickers)
            
            # فیلتر کردن فقط جفت‌های USDT
            usdt_pairs = []
            for symbol, ticker in tickers.items():
                if symbol.endswith('/USDT') and ticker['quoteVolume'] is not None:
                    usdt_pairs.append((symbol, ticker['quoteVolume']))
            
            # مرتب‌سازی بر اساس حجم
            usdt_pairs.sort(key=lambda x: x[1], reverse=True)
            return [pair[0] for pair in usdt_pairs[:n]]
            
        except Exception as e:
            logger.error(f"Error getting top volume pairs: {e}")
            return []
