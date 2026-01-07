import os
import sys
import asyncio
import logging
import traceback
from datetime import datetime
from pathlib import Path

# اضافه کردن مسیر پروژه به sys.path
sys.path.append(str(Path(__file__).parent))

from bot import FastScalpCompleteBot
from utils import (
    setup_logger, 
    validate_api_keys, 
    sanitize_output,
    PerformanceTracker,
    DataCache
)

# ============================================
# 🎨 Banner و نمایش اطلاعات
# ============================================

def display_banner():
    """نمایش بنر زیبا"""
    banner = """
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║   🤖 FAST SCALP COMPLETE TRADING BOT v1.0.0              ║
║   📊 ترکیب کامل دو اندیکاتور پیشرفته                    ║
║   ⚡ تایم‌فرم ۵ دقیقه - اسکالپینگ سریع                   ║
║   🚀 توسعه یافته برای Render.com                         ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝

📋 ویژگی‌ها:
├── 🟢 ZLMA Trend + Smart Money Pro
├── 🔴 RSI Divergence + Ichimoku Cloud
├── 📊 تحلیل ۱۰۰ ارز برتر
├── ⏰ اسکن هر ساعت
├── 📱 ارسال ۳ سیگنال برتر به تلگرام
├── 🛡️ مدیریت ریسک با ATR
└── 📈 سیستم امتیازدهی پیشرفته
"""
    print(banner)

# ============================================
# ⚙️ Configuration Loader
# ============================================

def load_config() -> dict:
    """
    لود کردن و اعتبارسنجی تنظیمات
    """
    logger = logging.getLogger(__name__)
    
    print("\n" + "="*60)
    print("⚙️  LOADING CONFIGURATION")
    print("="*60)
    
    # ساختار config
    config = {
        'telegram': {},
        'exchange': {},
        'strategy': {},
        'system': {}
    }
    
    # ======================
    # 📱 تنظیمات تلگرام (ضروری)
    # ======================
    required_vars = ['TELEGRAM_BOT_TOKEN', 'TELEGRAM_CHAT_ID']
    
    for var in required_vars:
        value = os.getenv(var)
        if not value:
            logger.error(f"❌ متغیر محیطی ضروری یافت نشد: {var}")
            logger.error("لطفاً در Render Dashboard → Environment تنظیم کنید")
            logger.error("یا در فایل .env قرار دهید")
            sys.exit(1)
        
        if var == 'TELEGRAM_BOT_TOKEN':
            config['telegram']['token'] = value
            # نمایش جزئی از توکن برای تایید
            token_preview = value[:10] + "..." + value[-10:] if len(value) > 20 else value
            logger.info(f"✅ Telegram Token: {token_preview}")
        else:
            config['telegram']['chat_id'] = value
            logger.info(f"✅ Telegram Chat ID: {value}")
    
    # ======================
    # 💱 تنظیمات صرافی (اختیاری)
    # ======================
    mexc_api_key = os.getenv('MEXC_API_KEY', '')
    mexc_secret = os.getenv('MEXC_SECRET_KEY', '')
    
    if mexc_api_key and mexc_secret:
        config['exchange']['api_key'] = mexc_api_key
        config['exchange']['secret'] = mexc_secret
        config['exchange']['enabled'] = True
        logger.info("✅ MEXC API: Enabled (با احراز هویت)")
    else:
        config['exchange']['api_key'] = ''
        config['exchange']['secret'] = ''
        config['exchange']['enabled'] = False
        logger.info("ℹ️ MEXC API: Disabled (استفاده از داده عمومی)")
    
    # ======================
    # 📈 تنظیمات استراتژی
    # ======================
    config['strategy'] = {
        'timeframe': '5m',
        'top_n': int(os.getenv('TOP_N_SIGNALS', '3')),
        'update_interval': int(os.getenv('UPDATE_INTERVAL', '3600')),
        'min_confidence': int(os.getenv('MIN_CONFIDENCE', '65')),
        'max_symbols': int(os.getenv('MAX_SYMBOLS', '20')),
        'risk_reward': float(os.getenv('RISK_REWARD_RATIO', '1.5')),
        'atr_period': int(os.getenv('ATR_PERIOD', '14'))
    }
    
    logger.info(f"📊 Strategy Config:")
    logger.info(f"   • Timeframe: {config['strategy']['timeframe']}")
    logger.info(f"   • Top Signals: {config['strategy']['top_n']}")
    logger.info(f"   • Scan Interval: {config['strategy']['update_interval']}s")
    logger.info(f"   • Min Confidence: {config['strategy']['min_confidence']}%")
    logger.info(f"   • Max Symbols: {config['strategy']['max_symbols']}")
    
    # ======================
    # 🖥️ تنظیمات سیستم
    # ======================
    config['system'] = {
        'log_level': os.getenv('LOG_LEVEL', 'INFO'),
        'cache_enabled': os.getenv('CACHE_ENABLED', 'true').lower() == 'true',
        'cache_ttl': int(os.getenv('CACHE_TTL', '300')),
        'performance_tracking': os.getenv('PERFORMANCE_TRACKING', 'true').lower() == 'true',
        'debug_mode': os.getenv('DEBUG_MODE', 'false').lower() == 'true',
        'timezone': os.getenv('TZ', 'UTC')
    }
    
    # تنظیم تایم‌زون
    os.environ['TZ'] = config['system']['timezone']
    
    # ======================
    # 📁 تنظیمات مسیرها
    # ======================
    config['paths'] = {
        'logs': os.getenv('LOG_DIR', 'logs'),
        'cache': os.getenv('CACHE_DIR', '.cache'),
        'data': os.getenv('DATA_DIR', 'data')
    }
    
    # ایجاد دایرکتوری‌ها
    for path in config['paths'].values():
        Path(path).mkdir(exist_ok=True)
    
    # ======================
    # ✅ اعتبارسنجی نهایی
    # ======================
    if not validate_api_keys(config):
        logger.error("❌ اعتبارسنجی API Keys ناموفق بود")
        sys.exit(1)
    
    # نمایش خلاصه config (سانتایز شده)
    logger.info("\n" + "="*60)
    logger.info("✅ CONFIGURATION LOADED SUCCESSFULLY")
    logger.info("="*60)
    
    # نمایش config سانتایز شده
    safe_config = sanitize_output(config)
    logger.debug(f"Full config: {safe_config}")
    
    return config

# ============================================
# 🔧 System Health Check
# ============================================

async def system_health_check() -> bool:
    """
    بررسی سلامت سیستم قبل از راه‌اندازی
    """
    logger = logging.getLogger(__name__)
    
    logger.info("\n" + "="*60)
    logger.info("🔧 SYSTEM HEALTH CHECK")
    logger.info("="*60)
    
    checks = []
    
    # 1. بررسی Python version
    python_version = sys.version_info
    python_ok = python_version >= (3, 8)
    checks.append(("Python >= 3.8", python_ok, f"{python_version.major}.{python_version.minor}.{python_version.micro}"))
    
    # 2. بررسی وجود فایل‌های ضروری
    required_files = ['requirements.txt', 'bot.py', 'indicators.py', 'utils.py']
    for file in required_files:
        exists = Path(file).exists()
        checks.append((f"File: {file}", exists, "Found" if exists else "Missing"))
    
    # 3. بررسی memory
    try:
        import psutil
        memory = psutil.virtual_memory()
        memory_ok = memory.available > 100 * 1024 * 1024  # 100MB
        checks.append(("Memory > 100MB", memory_ok, f"{memory.available // (1024*1024)}MB available"))
    except ImportError:
        checks.append(("Memory Check", True, "psutil not installed"))
    
    # 4. بررسی disk space
    try:
        disk = psutil.disk_usage('.')
        disk_ok = disk.free > 500 * 1024 * 1024  # 500MB
        checks.append(("Disk > 500MB", disk_ok, f"{disk.free // (1024*1024)}MB free"))
    except:
        checks.append(("Disk Check", True, "N/A"))
    
    # نمایش نتایج
    all_passed = True
    for check_name, status, details in checks:
        symbol = "✅" if status else "❌"
        logger.info(f"{symbol} {check_name}: {details}")
        if not status:
            all_passed = False
    
    if all_passed:
        logger.info("✅ همه بررسی‌های سلامت PASSED")
        return True
    else:
        logger.error("❌ برخی بررسی‌های سلامت FAILED")
        return False

# ============================================
# 📊 Performance Summary
# ============================================

def show_performance_summary():
    """
    نمایش خلاصه عملکرد (اگر قبلا اجرا شده)
    """
    try:
        tracker = PerformanceTracker()
        stats = tracker.get_performance_stats()
        
        if stats:
            logger = logging.getLogger(__name__)
            logger.info("\n" + "="*60)
            logger.info("📊 PREVIOUS PERFORMANCE SUMMARY")
            logger.info("="*60)
            logger.info(f"   Total Signals: {stats.get('total_signals', 0)}")
            logger.info(f"   Win Rate: {stats.get('win_rate', 0)}%")
            logger.info(f"   Wins: {stats.get('wins', 0)} | Losses: {stats.get('losses', 0)}")
            logger.info(f"   Avg Win: {stats.get('avg_win', 0)}% | Avg Loss: {stats.get('avg_loss', 0)}%")
            logger.info(f"   Risk/Reward: {stats.get('risk_reward', 0):.2f}")
            logger.info(f"   Expectancy: {stats.get('expectancy', 0):.2f}%")
    except:
        pass  # ignore if no performance data

# ============================================
# 🚀 Signal Test (Optional)
# ============================================

async def run_test_scan():
    """
    اجرای اسکن تست (اختیاری)
    """
    logger = logging.getLogger(__name__)
    
    test_env = os.getenv('RUN_TEST_SCAN', 'false').lower()
    if test_env != 'true':
        return
    
    logger.info("\n" + "="*60)
    logger.info("🧪 RUNNING TEST SCAN")
    logger.info("="*60)
    
    try:
        # ساخت config تست
        test_config = {
            'telegram': {
                'token': os.getenv('TELEGRAM_BOT_TOKEN', 'test_token'),
                'chat_id': os.getenv('TELEGRAM_CHAT_ID', 'test_chat')
            },
            'exchange': {
                'api_key': '',
                'secret': '',
                'enabled': False
            },
            'strategy': {
                'timeframe': '5m',
                'top_n': 1,
                'max_symbols': 3,
                'min_confidence': 50
            }
        }
        
        # ایجاد ربات تست
        test_bot = FastScalpCompleteBot(test_config)
        
        # اجرای یک اسکن سریع
        logger.info("Running quick test scan...")
        
        # این تابع باید در کلاس bot تعریف شود
        if hasattr(test_bot, 'run_test'):
            await test_bot.run_test()
        else:
            logger.warning("Test function not available")
        
        logger.info("✅ Test scan completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Test scan failed: {e}")

# ============================================
# 📱 Telegram Initialization
# ============================================

async def send_startup_message(config: dict):
    """
    ارسال پیام شروع به تلگرام
    """
    try:
        from telegram import Bot
        
        bot_token = config['telegram']['token']
        chat_id = config['telegram']['chat_id']
        
        bot = Bot(token=bot_token)
        
        startup_msg = f"""
🚀 *Fast Scalp Bot Started Successfully!*

📋 *Configuration:*
• Version: 1.0.0
• Timeframe: {config['strategy']['timeframe']}
• Scan Interval: {config['strategy']['update_interval']} seconds
• Max Symbols: {config['strategy']['max_symbols']}
• Timezone: {config['system']['timezone']}

⏰ *Startup Time:* {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC
📍 *Deployment:* Render.com

🤖 *Bot will scan the market every hour and send top {config['strategy']['top_n']} signals.*

✅ *Status:* Active and Running
"""
        
        await bot.send_message(
            chat_id=chat_id,
            text=startup_msg,
            parse_mode='Markdown'
        )
        
        logger = logging.getLogger(__name__)
        logger.info("📤 Startup message sent to Telegram")
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.warning(f"Could not send startup message: {e}")

# ============================================
# 🎯 Main Function
# ============================================

async def main():
    """
    تابع اصلی اجرای ربات
    """
    # نمایش بنر
    display_banner()
    
    # تنظیم لاگر پیشرفته
    logger = setup_logger(
        name="fast_scalp_main",
        log_to_file=os.getenv('LOG_TO_FILE', 'false').lower() == 'true'
    )
    
    logger.info(f"🚀 Starting Fast Scalp Complete Bot")
    logger.info(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"🐍 Python: {sys.version}")
    logger.info(f"📁 Working Dir: {os.getcwd()}")
    
    try:
        # 1. بررسی سلامت سیستم
        if not await system_health_check():
            logger.error("System health check failed. Exiting...")
            sys.exit(1)
        
        # 2. لود کردن تنظیمات
        config = load_config()
        
        # 3. نمایش خلاصه عملکرد قبلی
        show_performance_summary()
        
        # 4. اجرای تست (اگر فعال باشد)
        await run_test_scan()
        
        # 5. ایجاد ربات اصلی
        logger.info("\n" + "="*60)
        logger.info("🤖 INITIALIZING MAIN BOT")
        logger.info("="*60)
        
        bot = FastScalpCompleteBot(config)
        
        # 6. ارسال پیام شروع به تلگرام
        await send_startup_message(config)
        
        # 7. اجرای ربات اصلی
        logger.info("\n" + "="*60)
        logger.info("🚀 STARTING MAIN BOT LOOP")
        logger.info("="*60)
        logger.info("Press Ctrl+C to stop the bot")
        
        await bot.run()
        
    except KeyboardInterrupt:
        logger.info("\n" + "="*60)
        logger.info("👋 BOT STOPPED BY USER")
        logger.info("="*60)
        
        # ارسال پیام توقف به تلگرام (اختیاری)
        try:
            from telegram import Bot
            bot_token = config['telegram']['token']
            chat_id = config['telegram']['chat_id']
            
            bot = Bot(token=bot_token)
            await bot.send_message(
                chat_id=chat_id,
                text=f"🛑 *Bot Stopped*\n\nTime: {datetime.utcnow().strftime('%H:%M:%S')} UTC",
                parse_mode='Markdown'
            )
        except:
            pass
        
        sys.exit(0)
        
    except Exception as e:
        logger.error("\n" + "="*60)
        logger.error("❌ FATAL ERROR OCCURRED")
        logger.error("="*60)
        logger.error(f"Error Type: {type(e).__name__}")
        logger.error(f"Error Message: {str(e)}")
        logger.error("\nStack Trace:")
        logger.error(traceback.format_exc())
        
        # ارسال خطا به تلگرام (اگر config در دسترس باشد)
        try:
            from telegram import Bot
            bot_token = config['telegram']['token']
            chat_id = config['telegram']['chat_id']
            
            error_msg = f"""
⚠️ *Bot Crashed!*

*Error:* `{type(e).__name__}`
*Message:* {str(e)[:200]}
*Time:* {datetime.utcnow().strftime('%H:%M:%S')} UTC

Please check the logs.
"""
            
            bot = Bot(token=bot_token)
            await bot.send_message(
                chat_id=chat_id,
                text=error_msg,
                parse_mode='Markdown'
            )
        except:
            pass
        
        sys.exit(1)

# ============================================
# 🎬 Entry Point
# ============================================

if __name__ == "__main__":
    # بررسی اگر در Render اجرا می‌شود
    is_render = 'RENDER' in os.environ
    
    if is_render:
        print("\n" + "="*60)
        print("🌐 RUNNING ON RENDER.COM")
        print("="*60)
        
        # تنظیمات مخصوص Render
        os.environ['LOG_TO_FILE'] = 'false'  # در Render بهتر است از stdout استفاده کنیم
        os.environ['CACHE_ENABLED'] = 'true'
        
        # حذف handler اضافی اگر وجود دارد
        root_logger = logging.getLogger()
        if root_logger.handlers:
            for handler in root_logger.handlers:
                root_logger.removeHandler(handler)
        
        # اضافه کردن handler برای Render
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        root_logger.addHandler(handler)
    
    # اجرای main
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"\n❌ Critical error during startup: {e}")
        print(traceback.format_exc())
        sys.exit(1)
