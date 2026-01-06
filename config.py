import os
from dotenv import load_dotenv

load_dotenv()

# تنظیمات صرافی MEXC
MEXC_CONFIG = {
    'apiKey': os.getenv('MEXC_API_KEY'),
    'secret': os.getenv('MEXC_SECRET_KEY'),
    'enableRateLimit': True,
    'options': {
        'defaultType': 'spot'
    }
}

# لیست ۱۰۰ ارز برتر (می‌توانید بروزرسانی کنید)
TOP_100_CRYPTO = [
    'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT',
    'ADA/USDT', 'AVAX/USDT', 'DOGE/USDT', 'TRX/USDT', 'LINK/USDT',
    'DOT/USDT', 'MATIC/USDT', 'SHIB/USDT', 'LTC/USDT', 'BCH/USDT',
    'UNI/USDT', 'ATOM/USDT', 'XLM/USDT', 'ETC/USDT', 'FIL/USDT',
    'APT/USDT', 'NEAR/USDT', 'OP/USDT', 'ARB/USDT', 'VET/USDT',
    'AAVE/USDT', 'ALGO/USDT', 'GRT/USDT', 'QNT/USDT', 'EOS/USDT',
    # ... ادامه لیست تا ۱۰۰ ارز
]

# تنظیمات تلگرام
TELEGRAM_CONFIG = {
    'token': os.getenv('TELEGRAM_BOT_TOKEN'),
    'chat_id': os.getenv('TELEGRAM_CHAT_ID')
}

# تنظیمات استراتژی
STRATEGY_CONFIG = {
    'timeframe': '5m',
    'top_n': 3,  # نمایش ۳ ارز برتر
    'update_interval': 3600,  # هر ۱ ساعت
    'zlma_length': 15,
    'rsi_length': 14,
    'atr_period': 14,
    'risk_reward_ratio': 1.5,
    'min_volume_usdt': 1000000  # حداقل حجم معاملات
}
