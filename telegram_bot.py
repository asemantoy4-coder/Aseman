import asyncio
import logging
from datetime import datetime
from typing import List, Dict
import pandas as pd
from telegram import Bot, ParseMode
from telegram.error import TelegramError

from signal_generator import Signal

logger = logging.getLogger(__name__)

class TelegramBot:
    def __init__(self, config: dict):
        self.token = config['token']
        self.chat_id = config['chat_id']
        self.bot = Bot(token=self.token)
        
    async def send_signal(self, signal: Signal):
        """ارسال یک سیگنال به تلگرام"""
        try:
            emoji = "🟢" if signal.signal_type.value == "BUY" else "🔴"
            
            message = f"""
{emoji} *سیگنال فست اسکلپ* {emoji}

*جفت ارز:* `{signal.symbol}`
*نوع سیگنال:* {signal.signal_type.value}
*اعتماد:* {signal.confidence:.1f}%

💰 *قیمت ورود:* {signal.entry_price:.4f}
🛑 *حد ضرر:* {signal.stop_loss:.4f}
🎯 *هدف اول:* {signal.take_profit_1:.4f}
🎯 *هدف دوم:* {signal.take_profit_2:.4f}

📊 *نسبت ریسک به سود:* 1:2
⏰ *تاریخ:* {signal.timestamp.strftime('%Y-%m-%d %H:%M:%S')}

📝 *دلیل:* {signal.reason}

🔔 *نکته:* این سیگنال برای تایم‌فرم ۵ دقیقه‌ای طراحی شده است.
"""
            
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode=ParseMode.MARKDOWN
            )
            
            logger.info(f"Signal sent to Telegram for {signal.symbol}")
            
        except TelegramError as e:
            logger.error(f"Telegram error: {e}")
        except Exception as e:
            logger.error(f"Error sending signal: {e}")
    
    async def send_daily_report(self, top_signals: List[Signal], 
                               market_summary: Dict):
        """ارسال گزارش روزانه"""
        try:
            report_date = datetime.now().strftime('%Y-%m-%d')
            
            # خلاصه بازار
            summary_text = f"""
📊 *گزارش بازار کریپتو - {report_date}*

📈 *تعداد کل سیگنال‌های امروز:* {market_summary.get('total_signals', 0)}
🟢 *سیگنال‌های خرید:* {market_summary.get('buy_signals', 0)}
🔴 *سیگنال‌های فروش:* {market_summary.get('sell_signals', 0)}
📊 *میانگین اعتماد:* {market_summary.get('avg_confidence', 0):.1f}%
"""
            
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=summary_text,
                parse_mode=ParseMode.MARKDOWN
            )
            
            # سیگنال‌های برتر
            if top_signals:
                top_text = "🏆 *سیگنال‌های برتر امروز:*\n\n"
                for i, signal in enumerate(top_signals[:5], 1):
                    emoji = "🟢" if signal.signal_type.value == "BUY" else "🔴"
                    top_text += f"{i}. {emoji} `{signal.symbol}` - اعتماد: {signal.confidence:.1f}%\n"
                
                await self.bot.send_message(
                    chat_id=self.chat_id,
                    text=top_text,
                    parse_mode=ParseMode.MARKDOWN
                )
                
        except Exception as e:
            logger.error(f"Error sending daily report: {e}")
    
    async def send_error_alert(self, error_message: str):
        """ارسال هشدار خطا"""
        try:
            message = f"""
⚠️ *هشدار خطا در سیستم*

{error_message}

⏰ زمان: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
            
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode=ParseMode.MARKDOWN
            )
            
        except Exception as e:
            logger.error(f"Error sending error alert: {e}")
    
    async def send_test_message(self):
        """ارسال پیام تست"""
        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text="🤖 *ربات فست اسکلپ فعال شد!*\n\nربات آماده دریافت و ارسال سیگنال‌هاست.",
                parse_mode=ParseMode.MARKDOWN
            )
            logger.info("Test message sent successfully")
            
        except Exception as e:
            logger.error(f"Error sending test message: {e}")
            raise
