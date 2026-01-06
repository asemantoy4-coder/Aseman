import asyncio
import logging
import sys
from pathlib import Path

# اضافه کردن مسیر پروژه به sys.path
sys.path.append(str(Path(__file__).parent))

from config import MEXC_CONFIG, TELEGRAM_CONFIG, STRATEGY_CONFIG, TOP_100_CRYPTO
from scheduler import TradingScheduler

# تنظیمات لاگ
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fast_scalp.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

async def main():
    """تابع اصلی اجرای برنامه"""
    try:
        # پیکربندی
        config = {
            'MEXC_CONFIG': MEXC_CONFIG,
            'TELEGRAM_CONFIG': TELEGRAM_CONFIG,
            'STRATEGY_CONFIG': STRATEGY_CONFIG,
            'TOP_100_CRYPTO': TOP_100_CRYPTO
        }
        
        # ایجاد شی زمان‌بند
        scheduler = TradingScheduler(config)
        
        # اجرای سیستم
        await scheduler.run()
        
    except Exception as e:
        logger.error(f"Failed to start system: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # راه‌اندازی برنامه
    print("""
    ╔══════════════════════════════════════╗
    ║   🚀 Fast Scalp Crypto Trading Bot   ║
    ║      با ترکیب دو اندیکاتور پیشرفته     ║
    ╚══════════════════════════════════════╝
    
    🎯 ویژگی‌ها:
    • اسکن ۱۰۰ ارز برتر کریپتو
    • تایم‌فرم ۵ دقیقه‌ای
    • ترکیب ZLMA + Smart Money + RSI Divergence
    • ارسال سیگنال به تلگرام
    • نمایش ۳ سیگنال برتر هر ساعت
    
    ⚠️ توجه: قبل از استفاده، فایل .env را تنظیم کنید
    """)
    
    # بررسی وجود فایل .env
    env_file = Path('.env')
    if not env_file.exists():
        print("\n❌ فایل .env یافت نشد!")
        print("لطفاً فایل .env با محتوای زیر ایجاد کنید:")
        print("""
MEXC_API_KEY=your_api_key_here
MEXC_SECRET_KEY=your_secret_key_here
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
        """)
        sys.exit(1)
    
    # اجرای برنامه
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 برنامه با موفقیت متوقف شد.")
