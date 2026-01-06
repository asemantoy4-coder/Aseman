# fast-scalp-crypto-bot/utils/helpers.py
import pandas as pd
import numpy as np
from typing import Dict, Any
from datetime import datetime, timedelta

def format_price(price: float, precision: int = 4) -> str:
    """فرمت‌دهی قیمت"""
    return f"{price:.{precision}f}"

def calculate_change_percent(current: float, previous: float) -> float:
    """محاسبه درصد تغییر"""
    if previous == 0:
        return 0
    return ((current - previous) / previous) * 100

def filter_low_volume_pairs(tickers: Dict, min_volume_usdt: float = 1000000) -> list:
    """فیلتر کردن جفت‌های با حجم کم"""
    filtered = []
    for symbol, data in tickers.items():
        volume = data.get('volume', 0)
        if volume and volume >= min_volume_usdt:
            filtered.append((symbol, volume))
    
    # مرتب‌سازی بر اساس حجم
    filtered.sort(key=lambda x: x[1], reverse=True)
    return [pair[0] for pair in filtered]

def validate_dataframe(df: pd.DataFrame, required_columns: list = None) -> bool:
    """اعتبارسنجی دیتافریم"""
    if df.empty:
        return False
    
    if required_columns is None:
        required_columns = ['open', 'high', 'low', 'close', 'volume']
    
    for col in required_columns:
        if col not in df.columns:
            return False
        
        # بررسی مقادیر NaN
        if df[col].isna().any():
            return False
    
    return True

def calculate_volatility(df: pd.DataFrame, period: int = 20) -> float:
    """محاسبه نوسان"""
    returns = df['close'].pct_change().dropna()
    if len(returns) < period:
        return 0
    return returns.tail(period).std() * 100  # نوسان درصدی

def get_market_hours() -> Dict[str, bool]:
    """دریافت ساعات بازار (برای فیلتر زمان)"""
    now = datetime.utcnow()
    hour = now.hour
    
    # ساعات پرتراکنش بازار کریپتو (۰-۲۴ ساعته ولی بعضی ساعات بهترند)
    high_volume_hours = list(range(8, 20))  # 8 AM تا 8 PM UTC
    
    return {
        'is_market_hours': hour in high_volume_hours,
        'current_hour': hour,
        'is_weekend': now.weekday() >= 5
    }
