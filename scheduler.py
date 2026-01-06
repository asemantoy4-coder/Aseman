import asyncio
import schedule
import time
import logging
from datetime import datetime, timedelta
from typing import List
import pandas as pd

from data_fetcher import MexcDataFetcher
from indicator_processor import IndicatorProcessor
from signal_generator import SignalGenerator, Signal
from telegram_bot import TelegramBot

logger = logging.getLogger(__name__)

class TradingScheduler:
    def __init__(self, config: dict):
        self.config = config
        self.data_fetcher = MexcDataFetcher(config['MEXC_CONFIG'])
        self.indicator_processor = IndicatorProcessor(config['STRATEGY_CONFIG'])
        self.signal_generator = SignalGenerator(config['STRATEGY_CONFIG'])
        self.telegram_bot = TelegramBot(config['TELEGRAM_CONFIG'])
        
        self.top_pairs = []
        self.daily_signals = []
        
    async def initialize(self):
        """راه‌اندازی اولیه سیستم"""
        logger.info("Initializing Fast Scalp Trading System...")
        
        # تست اتصال به تلگرام
        try:
            await self.telegram_bot.send_test_message()
            logger.info("Telegram connection test successful")
        except Exception as e:
            logger.error(f"Failed to connect to Telegram: {e}")
            raise
        
        # دریافت لیست ارزهای برتر
        self.top_pairs = await self.data_fetcher.get_top_volume_pairs(100)
        logger.info(f"Loaded {len(self.top_pairs)} trading pairs")
        
    async def run_scan(self):
        """اسکن بازار و تولید سیگنال"""
        logger.info("Starting market scan...")
        
        all_signals = []
        processed_pairs = 0
        
        try:
            for symbol in self.top_pairs[:50]:  # اسکن ۵۰ ارز اول برای سرعت
                try:
                    # دریافت داده‌ها
                    df = await self.data_fetcher.fetch_ohlcv(symbol, limit=500)
                    if df.empty:
                        continue
                    
                    # محاسبه اندیکاتورها
                    indicators = {}
                    
                    # ZLMA و EMA
                    indicators['zlma'] = self.indicator_processor.calculate_zlma(df)
                    indicators['ema'] = self.indicator_processor.calculate_ema(df)
                    
                    # Smart Money Signals
                    indicators['smart_money'] = self.indicator_processor.calculate_smart_money_signals(df)
                    
                    # RSI Divergence
                    indicators['rsi_signals'] = self.indicator_processor.calculate_rsi_divergence(df)
                    
                    # Ichimoku Cloud
                    indicators['ichimoku'] = self.indicator_processor.calculate_ichimoku(df)
                    
                    # MACD
                    indicators['macd'] = self.indicator_processor.calculate_macd(df)
                    
                    # ATR برای مدیریت ریسک
                    indicators['atr'] = self.indicator_processor.calculate_atr(df)
                    
                    # تولید سیگنال
                    signals = self.signal_generator.generate_signals(symbol, df, indicators)
                    all_signals.extend(signals)
                    
                    processed_pairs += 1
                    
                    # تاخیر کوچک برای جلوگیری از Rate Limit
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}")
                    continue
            
            logger.info(f"Processed {processed_pairs} pairs, found {len(all_signals)} signals")
            
            # رتبه‌بندی و انتخاب سیگنال‌های برتر
            if all_signals:
                top_signals = self.signal_generator.rank_signals(all_signals)
                
                # ارسال سیگنال‌های برتر به تلگرام
                for signal in top_signals:
                    await self.telegram_bot.send_signal(signal)
                    self.daily_signals.append(signal)
                    
                    # تاخیر بین ارسال سیگنال‌ها
                    await asyncio.sleep(1)
                
                logger.info(f"Sent {len(top_signals)} top signals to Telegram")
            
            else:
                logger.info("No valid signals found in this scan")
                await self.telegram_bot.send_message(
                    "🔄 اسکن بازار انجام شد. هیچ سیگنال معتبری یافت نشد."
                )
                
        except Exception as e:
            logger.error(f"Error in market scan: {e}")
            await self.telegram_bot.send_error_alert(str(e))
    
    async def send_daily_summary(self):
        """ارسال خلاصه روزانه"""
        try:
            if not self.daily_signals:
                logger.info("No signals to summarize today")
                return
            
            buy_signals = [s for s in self.daily_signals if s.signal_type.value == "BUY"]
            sell_signals = [s for s in self.daily_signals if s.signal_type.value == "SELL"]
            
            market_summary = {
                'total_signals': len(self.daily_signals),
                'buy_signals': len(buy_signals),
                'sell_signals': len(sell_signals),
                'avg_confidence': sum(s.confidence for s in self.daily_signals) / len(self.daily_signals)
            }
            
            top_signals = self.signal_generator.rank_signals(self.daily_signals)
            await self.telegram_bot.send_daily_report(top_signals, market_summary)
            
            logger.info("Daily summary sent")
            
            # پاکسازی لیست برای روز بعد
            self.daily_signals = []
            
        except Exception as e:
            logger.error(f"Error sending daily summary: {e}")
    
    def schedule_tasks(self):
        """زمان‌بندی وظایف"""
        # اسکن هر ساعت
        schedule.every().hour.at(":00").do(
            lambda: asyncio.create_task(self.run_scan())
        )
        
        # خلاصه روزانه در پایان روز
        schedule.every().day.at("23:30").do(
            lambda: asyncio.create_task(self.send_daily_summary())
        )
        
        # بروزرسانی لیست ارزها هر ۶ ساعت
        schedule.every(6).hours.do(
            lambda: asyncio.create_task(self.update_top_pairs())
        )
        
        logger.info("Tasks scheduled successfully")
    
    async def update_top_pairs(self):
        """بروزرسانی لیست ارزهای برتر"""
        try:
            new_pairs = await self.data_fetcher.get_top_volume_pairs(100)
            if new_pairs:
                self.top_pairs = new_pairs
                logger.info(f"Updated top pairs list: {len(new_pairs)} pairs")
        except Exception as e:
            logger.error(f"Error updating top pairs: {e}")
    
    async def run(self):
        """اجرای اصلی برنامه"""
        await self.initialize()
        self.schedule_tasks()
        
        logger.info("Fast Scalp Trading System is running...")
        
        try:
            while True:
                schedule.run_pending()
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down...")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            await self.telegram_bot.send_error_alert(f"System crash: {str(e)}")
