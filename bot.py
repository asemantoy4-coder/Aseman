import ccxt
import pandas as pd
import numpy as np
import asyncio
import logging
from datetime import datetime, timedelta
import schedule
import time
from telegram import Bot
from telegram.error import TelegramError
import os

from indicators import CombinedIndicators

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FastScalpCompleteBot:
    def __init__(self, config: dict):
        # تنظیمات
        self.telegram_token = config.get('telegram_token')
        self.chat_id = config.get('chat_id')
        self.bot = Bot(token=self.telegram_token) if self.telegram_token else None
        
        # اتصال به صرافی
        self.exchange = ccxt.mexc({
            'apiKey': config.get('mexc_api_key', ''),
            'secret': config.get('mexc_secret_key', ''),
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })
        
        # اندیکاتورها
        self.indicators = CombinedIndicators()
        
        # تنظیمات استراتژی
        self.timeframe = '5m'
        self.top_n = 3
        self.update_interval = 3600
        
        # لیست ارزها
        self.symbols = self._load_symbols()
        
        # کش سیگنال
        self.signal_cache = {}
        
        logger.info("✅ Fast Scalp Complete Bot Initialized")
    
    def _load_symbols(self) -> list:
        """لود کردن لیست ارزها"""
        # لیست 100 ارز برتر (مثال)
        top_100 = [
            'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT',
            'ADA/USDT', 'AVAX/USDT', 'DOGE/USDT', 'TRX/USDT', 'LINK/USDT',
            'DOT/USDT', 'MATIC/USDT', 'SHIB/USDT', 'LTC/USDT', 'BCH/USDT',
            'UNI/USDT', 'ATOM/USDT', 'XLM/USDT', 'ETC/USDT', 'FIL/USDT'
        ]
        return top_100
    
    async def fetch_data(self, symbol: str, limit: int = 300) -> pd.DataFrame:
        """دریافت داده از صرافی"""
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
            logger.error(f"Error fetching {symbol}: {e}")
            return pd.DataFrame()
    
    def analyze_symbol(self, symbol: str, df: pd.DataFrame) -> dict:
        """آنالیز کامل یک ارز"""
        if len(df) < 100:
            return None
        
        try:
            # تولید سیگنال ترکیبی
            signal = self.indicators.generate_combined_signal(df)
            
            if signal['signal_type'] != 'NEUTRAL' and signal['confidence'] >= 65:
                # جلوگیری از سیگنال تکراری
                signal_key = f"{symbol}_{signal['signal_type']}_{signal['timestamp'].strftime('%Y%m%d%H')}"
                
                if signal_key not in self.signal_cache:
                    self.signal_cache[signal_key] = True
                    
                    # اضافه کردن اطلاعات اضافی
                    signal['symbol'] = symbol
                    signal['volume'] = df['volume'].iloc[-1]
                    signal['volume_avg'] = df['volume'].rolling(20).mean().iloc[-1]
                    
                    return signal
            
            return None
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return None
    
    async def send_telegram_signal(self, signal: dict):
        """ارسال سیگنال به تلگرام"""
        try:
            if not self.bot:
                return
            
            emoji = "🟢" if signal['signal_type'] == "BUY" else "🔴"
            
            # پیام کامل با تمام جزئیات
            message = f"""
{emoji} *FAST SCALP SIGNAL* {emoji}

*Symbol:* `{signal['symbol']}`
*Type:* {signal['signal_type']}
*Confidence:* {signal['confidence']}%

💰 *Price:* {signal['price']:.4f}
🛑 *Stop Loss:* {signal['stop_loss']:.4f}
🎯 *Take Profit 1:* {signal['take_profit_1']:.4f}
🎯 *Take Profit 2:* {signal['take_profit_2']:.4f}

📊 *Indicators Summary:*
• Buy Conditions: {len(signal['buy_conditions'])}
• Sell Conditions: {len(signal['sell_conditions'])}
• ATR: {signal['atr']:.4f}

📈 *Key Signals:*
"""
            
            # اضافه کردن سیگنال‌های مهم
            conditions = signal['buy_conditions'] if signal['signal_type'] == "BUY" else signal['sell_conditions']
            for i, cond in enumerate(conditions[:5], 1):
                message += f"  {i}. {cond}\n"
            
            message += f"""
⏰ *Time:* {signal['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} UTC
📊 *Volume:* {signal['volume']:.0f} (Avg: {signal['volume_avg']:.0f})
"""
            
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode='Markdown'
            )
            
            logger.info(f"📤 Signal sent: {signal['symbol']} {signal['signal_type']}")
            
        except Exception as e:
            logger.error(f"Error sending telegram: {e}")
    
    async def scan_market(self):
        """اسکن کامل بازار"""
        logger.info("🔄 Starting complete market scan...")
        
        all_signals = []
        
        # اسکن 20 ارز اول برای سرعت
        for symbol in self.symbols[:20]:
            try:
                df = await self.fetch_data(symbol)
                if df.empty:
                    continue
                
                signal = self.analyze_symbol(symbol, df)
                if signal:
                    all_signals.append(signal)
                    logger.info(f"🎯 Signal found: {symbol} {signal['signal_type']} ({signal['confidence']}%)")
                
                await asyncio.sleep(0.3)  # جلوگیری از rate limit
                
            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}")
                continue
        
        # انتخاب 3 سیگنال برتر
        if all_signals:
            # مرتب‌سازی بر اساس اعتماد و حجم
            all_signals.sort(
                key=lambda x: (x['confidence'], x['volume'] / max(x['volume_avg'], 1)),
                reverse=True
            )
            
            top_signals = all_signals[:self.top_n]
            
            # ارسال سیگنال‌های برتر
            for signal in top_signals:
                await self.send_telegram_signal(signal)
                await asyncio.sleep(1)
            
            logger.info(f"✅ Scan completed: {len(all_signals)} signals found, {len(top_signals)} sent")
            
            # خلاصه اسکن
            summary = f"""
📊 *Market Scan Summary*
Total Symbols: 20
Signals Found: {len(all_signals)}
Top Signals Sent: {len(top_signals)}
Time: {datetime.utcnow().strftime('%H:%M:%S')} UTC
"""
            
            try:
                if self.bot and top_signals:
                    await self.bot.send_message(
                        chat_id=self.chat_id,
                        text=summary,
                        parse_mode='Markdown'
                    )
            except:
                pass
        else:
            logger.info("ℹ️ No signals found in this scan")
    
    async def run(self):
        """اجرای اصلی ربات"""
        logger.info("🚀 Fast Scalp Complete Bot Started")
        
        # ارسال پیام شروع
        if self.bot:
            try:
                await self.bot.send_message(
                    chat_id=self.chat_id,
                    text="🤖 *Fast Scalp Complete Bot Started*\n\nUsing combined indicators:\n• ZLMA Trend + Smart Money Pro\n• RSI + Ichimoku Cloud\n\nScanning every hour...",
                    parse_mode='Markdown'
                )
            except:
                pass
        
        # زمان‌بندی اسکن هر ساعت
        schedule.every().hour.at(":00").do(lambda: asyncio.create_task(self.scan_market()))
        
        # اجرای اولیه
        await self.scan_market()
        
        # حلقه اصلی
        while True:
            schedule.run_pending()
            await asyncio.sleep(1)
