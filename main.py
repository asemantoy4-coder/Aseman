import os
import sys
import asyncio
import logging
from bot import FastScalpCompleteBot

# تنظیم لاگ
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def load_config():
    """لود کردن تنظیمات"""
    
    print("\n" + "="*60)
    print("🤖 FAST SCALP COMPLETE - RENDER DEPLOYMENT")
    print("="*60 + "\n")
    
    # متغیرهای ضروری
    required = ['TELEGRAM_BOT_TOKEN', 'TELEGRAM_CHAT_ID']
    
    config = {}
    
    for var in required:
        value = os.getenv(var)
        if not value:
            logger.error(f"❌ Missing: {var}")
            logger.error("Set in Render dashboard → Environment")
            sys.exit(1)
        config[var.lower()] = value
    
    # متغیرهای اختیاری
    config['mexc_api_key'] = os.getenv('MEXC_API_KEY', '')
    config['mexc_secret_key'] = os.getenv('MEXC_SECRET_KEY', '')
    
    logger.info("✅ Config loaded")
    return config

async def main():
    """تابع اصلی"""
    try:
        config = load_config()
        bot = FastScalpCompleteBot(config)
        await bot.run()
        
    except KeyboardInterrupt:
        logger.info("\n👋 Bot stopped")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
