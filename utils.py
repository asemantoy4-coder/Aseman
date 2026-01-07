"""
📦 Utilities Module - ابزارهای کمکی برای Fast Scalp Bot
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
import json
from pathlib import Path

# ============================================
# 📊 Logger Configuration
# ============================================

def setup_logger(name: str = "fast_scalp", log_to_file: bool = False) -> logging.Logger:
    """
    تنظیم لاگر پیشرفته
    """
    logger = logging.getLogger(name)
    
    if logger.hasHandlers():
        return logger
    
    logger.setLevel(logging.INFO)
    
    # فرمت لاگ
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # هندلر کنسول
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # هندلر فایل (برای Render بهتر است فقط از stdout استفاده کنیم)
    if log_to_file:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        file_handler = logging.FileHandler(log_dir / f"{name}.log")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# ============================================
# 🔧 Data Utilities
# ============================================

def validate_dataframe(df: pd.DataFrame) -> bool:
    """
    اعتبارسنجی دیتافریم OHLCV
    """
    if df.empty:
        return False
    
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    
    # چک کردن وجود ستون‌ها
    for col in required_columns:
        if col not in df.columns:
            return False
    
    # چک کردن مقادیر NaN
    if df[required_columns].isna().any().any():
        return False
    
    # چک کردن حجم داده
    if len(df) < 50:
        return False
    
    return True

def clean_ohlcv_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    تمیز کردن و پیش‌پردازش داده‌های OHLCV
    """
    if df.empty:
        return df
    
    # کپی برای جلوگیری از SettingWithCopyWarning
    df_clean = df.copy()
    
    # حذف ردیف‌های با حجم صفر
    df_clean = df_clean[df_clean['volume'] > 0]
    
    # حذف outliers قیمت (تغییرات بیشتر از 50% در یک کندل)
    price_change = df_clean['close'].pct_change().abs()
    df_clean = df_clean[price_change < 0.5]
    
    # پر کردن مقادیر NaN با forward fill
    df_clean = df_clean.ffill()
    
    return df_clean

def calculate_volume_profile(df: pd.DataFrame, bins: int = 20) -> Dict:
    """
    محاسبه Volume Profile
    """
    if df.empty:
        return {}
    
    # محاسبه سطوح قیمت
    min_price = df['low'].min()
    max_price = df['high'].max()
    
    # ایجاد bins
    price_bins = np.linspace(min_price, max_price, bins + 1)
    
    # محاسبه حجم در هر bin
    volume_profile = {}
    for i in range(bins):
        bin_low = price_bins[i]
        bin_high = price_bins[i + 1]
        
        # انتخاب کندل‌هایی که در این بازه هستند
        mask = (df['low'] >= bin_low) & (df['high'] <= bin_high)
        bin_volume = df.loc[mask, 'volume'].sum()
        
        if bin_volume > 0:
            key = f"{bin_low:.4f}-{bin_high:.4f}"
            volume_profile[key] = {
                'low': float(bin_low),
                'high': float(bin_high),
                'volume': float(bin_volume),
                'price_level': float((bin_low + bin_high) / 2)
            }
    
    # یافتن نقطه کنترل (POC)
    if volume_profile:
        poc_key = max(volume_profile, key=lambda k: volume_profile[k]['volume'])
        volume_profile['poc'] = volume_profile[poc_key]
    
    return volume_profile

# ============================================
# 📈 Technical Utilities
# ============================================

def calculate_support_resistance(df: pd.DataFrame, 
                                 window: int = 20, 
                                 threshold: float = 0.02) -> Dict:
    """
    تشخیص سطوح حمایت و مقاومت
    """
    if len(df) < window * 2:
        return {'supports': [], 'resistances': []}
    
    highs = df['high'].values
    lows = df['low'].values
    
    supports = []
    resistances = []
    
    # تشخیص pivot points
    for i in range(window, len(df) - window):
        # مقاومت (سقف محلی)
        if highs[i] == highs[i-window:i+window].max():
            resistances.append({
                'price': float(highs[i]),
                'strength': int(window),
                'timestamp': df.index[i]
            })
        
        # حمایت (کف محلی)
        if lows[i] == lows[i-window:i+window].min():
            supports.append({
                'price': float(lows[i]),
                'strength': int(window),
                'timestamp': df.index[i]
            })
    
    # حذف سطوح نزدیک به هم
    def merge_levels(levels, threshold_percent=threshold):
        if not levels:
            return []
        
        levels.sort(key=lambda x: x['price'])
        merged = [levels[0]]
        
        for level in levels[1:]:
            last = merged[-1]
            price_diff = abs(level['price'] - last['price']) / last['price']
            
            if price_diff > threshold_percent:
                merged.append(level)
            else:
                # تقویت سطح قبلی
                last['strength'] += level['strength']
        
        return merged
    
    supports = merge_levels(supports)
    resistances = merge_levels(resistances)
    
    # فقط سطوح قوی (با strength بالا)
    min_strength = 2
    supports = [s for s in supports if s['strength'] >= min_strength]
    resistances = [r for r in resistances if r['strength'] >= min_strength]
    
    # سطوح فعلی (آخرین 5 سطح)
    current_price = df['close'].iloc[-1]
    
    # نزدیک‌ترین سطوح
    nearest_support = None
    nearest_resistance = None
    
    if supports:
        supports.sort(key=lambda x: abs(x['price'] - current_price))
        nearest_support = supports[0]
    
    if resistances:
        resistances.sort(key=lambda x: abs(x['price'] - current_price))
        nearest_resistance = resistances[0]
    
    return {
        'supports': supports[-5:],  # آخرین 5 سطح
        'resistances': resistances[-5:],
        'nearest_support': nearest_support,
        'nearest_resistance': nearest_resistance,
        'current_price': float(current_price)
    }

def calculate_market_structure(df: pd.DataFrame) -> Dict:
    """
    تشخیص ساختار بازار (Higher Highs/Lower Lows)
    """
    if len(df) < 50:
        return {'trend': 'neutral', 'structure': []}
    
    highs = df['high'].values
    lows = df['low'].values
    
    # تشخیص swing points
    lookback = 5
    swing_highs = []
    swing_lows = []
    
    for i in range(lookback, len(df) - lookback):
        # Swing High
        if highs[i] == highs[i-lookback:i+lookback].max():
            swing_highs.append({
                'index': i,
                'price': float(highs[i]),
                'time': df.index[i]
            })
        
        # Swing Low
        if lows[i] == lows[i-lookback:i+lookback].min():
            swing_lows.append({
                'index': i,
                'price': float(lows[i]),
                'time': df.index[i]
            })
    
    # تشخیص روند
    trend = 'neutral'
    
    if len(swing_highs) >= 2 and len(swing_lows) >= 2:
        # بررسی Higher Highs
        higher_highs = swing_highs[-1]['price'] > swing_highs[-2]['price']
        higher_lows = swing_lows[-1]['price'] > swing_lows[-2]['price']
        
        # بررسی Lower Lows
        lower_highs = swing_highs[-1]['price'] < swing_highs[-2]['price']
        lower_lows = swing_lows[-1]['price'] < swing_lows[-2]['price']
        
        if higher_highs and higher_lows:
            trend = 'uptrend'
        elif lower_highs and lower_lows:
            trend = 'downtrend'
        else:
            trend = 'ranging'
    
    return {
        'trend': trend,
        'swing_highs': swing_highs[-3:] if swing_highs else [],
        'swing_lows': swing_lows[-3:] if swing_lows else [],
        'current_price': float(df['close'].iloc[-1])
    }

# ============================================
# ⚡ Performance Utilities
# ============================================

class PerformanceTracker:
    """
    ردیابی عملکرد سیگنال‌ها
    """
    
    def __init__(self, cache_file: str = "performance_cache.json"):
        self.cache_file = cache_file
        self.signals = self._load_cache()
    
    def _load_cache(self) -> List[Dict]:
        """لود کش از فایل"""
        try:
            if Path(self.cache_file).exists():
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
        except:
            pass
        return []
    
    def _save_cache(self):
        """ذخیره کش در فایل"""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.signals[-100:], f, indent=2)  # فقط ۱۰۰ مورد آخر
        except:
            pass
    
    def add_signal(self, signal: Dict):
        """افزودن سیگنال جدید"""
        signal['added_at'] = datetime.now().isoformat()
        self.signals.append(signal)
        self._save_cache()
    
    def update_signal_result(self, symbol: str, entry_price: float, 
                            result: str, exit_price: Optional[float] = None):
        """
        آپدیت نتیجه سیگنال
        
        result: 'win', 'loss', 'breakeven'
        """
        for signal in reversed(self.signals):
            if (signal.get('symbol') == symbol and 
                signal.get('price') == entry_price and 
                'result' not in signal):
                
                signal['result'] = result
                signal['exit_price'] = exit_price
                signal['exit_time'] = datetime.now().isoformat()
                
                if exit_price and 'price' in signal:
                    pnl_percent = ((exit_price - signal['price']) / signal['price']) * 100
                    signal['pnl_percent'] = round(pnl_percent, 2)
                
                self._save_cache()
                break
    
    def get_performance_stats(self) -> Dict:
        """آمار عملکرد"""
        if not self.signals:
            return {}
        
        completed_signals = [s for s in self.signals if 'result' in s]
        
        if not completed_signals:
            return {}
        
        wins = [s for s in completed_signals if s['result'] == 'win']
        losses = [s for s in completed_signals if s['result'] == 'loss']
        breakevens = [s for s in completed_signals if s['result'] == 'breakeven']
        
        total = len(completed_signals)
        win_rate = (len(wins) / total * 100) if total > 0 else 0
        
        # محاسبه متوسط سود/ضرر
        avg_win = 0
        avg_loss = 0
        
        if wins:
            avg_win = np.mean([s.get('pnl_percent', 0) for s in wins])
        
        if losses:
            avg_loss = np.mean([s.get('pnl_percent', 0) for s in losses])
        
        # Risk/Reward Ratio
        risk_reward = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        
        # Expectancy
        expectancy = (win_rate/100 * avg_win) - ((100-win_rate)/100 * abs(avg_loss))
        
        return {
            'total_signals': total,
            'win_rate': round(win_rate, 2),
            'wins': len(wins),
            'losses': len(losses),
            'breakevens': len(breakevens),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'risk_reward': round(risk_reward, 2),
            'expectancy': round(expectancy, 2),
            'last_updated': datetime.now().isoformat()
        }
    
    def get_recent_signals(self, limit: int = 10) -> List[Dict]:
        """دریافت آخرین سیگنال‌ها"""
        return self.signals[-limit:]

# ============================================
# 🔐 Security & Validation
# ============================================

def validate_api_keys(config: Dict) -> bool:
    """
    اعتبارسنجی کلیدهای API
    """
    required = ['telegram_token', 'chat_id']
    
    for key in required:
        if not config.get(key):
            return False
    
    # اعتبارسنجی فرمت توکن تلگرام
    telegram_token = config.get('telegram_token', '')
    if not telegram_token.startswith('') or len(telegram_token) < 30:
        return False
    
    return True

def sanitize_output(data: Any) -> Any:
    """
    پاکسازی خروجی برای جلوگیری از اطلاعات حساس
    """
    if isinstance(data, dict):
        sanitized = {}
        for key, value in data.items():
            if any(sensitive in key.lower() for sensitive in ['key', 'token', 'secret', 'password']):
                sanitized[key] = '***' + str(value)[-4:] if value else '***'
            else:
                sanitized[key] = sanitize_output(value)
        return sanitized
    elif isinstance(data, list):
        return [sanitize_output(item) for item in data]
    else:
        return data

# ============================================
# 📊 Formatting Utilities
# ============================================

def format_price(price: float, symbol: str = 'USDT') -> str:
    """
    فرمت‌دهی قیمت بر اساس جفت ارز
    """
    if price >= 1000:
        return f"${price:,.2f}"
    elif price >= 1:
        return f"${price:.4f}"
    elif price >= 0.01:
        return f"${price:.6f}"
    else:
        return f"${price:.8f}"

def format_percentage(value: float) -> str:
    """فرمت‌دهی درصد"""
    return f"{value:+.2f}%"

def format_timestamp(timestamp, format_str: str = '%Y-%m-%d %H:%M:%S') -> str:
    """فرمت‌دهی timestamp"""
    if isinstance(timestamp, (int, float)):
        timestamp = datetime.fromtimestamp(timestamp)
    elif isinstance(timestamp, str):
        try:
            timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        except:
            timestamp = datetime.now()
    
    return timestamp.strftime(format_str)

def format_signal_for_display(signal: Dict) -> str:
    """
    فرمت‌دهی سیگنال برای نمایش
    """
    emoji = "🟢" if signal.get('type') == 'BUY' else "🔴"
    
    lines = [
        f"{emoji} {signal.get('symbol', 'N/A')}",
        f"Type: {signal.get('type', 'N/A')}",
        f"Confidence: {signal.get('confidence', 0)}%",
        f"Price: {format_price(signal.get('price', 0))}",
    ]
    
    if 'stop_loss' in signal:
        lines.append(f"SL: {format_price(signal.get('stop_loss', 0))}")
    
    if 'take_profit_1' in signal:
        lines.append(f"TP1: {format_price(signal.get('take_profit_1', 0))}")
    
    return "\n".join(lines)

# ============================================
# ⏰ Time Utilities
# ============================================

def is_market_hours() -> bool:
    """
    بررسی ساعات فعال بازار کریپتو
    """
    now_utc = datetime.utcnow()
    hour_utc = now_utc.hour
    
    # ساعات پرترافیک بازار (۸ صبح تا ۸ شب UTC)
    return 8 <= hour_utc < 20

def next_scan_time() -> str:
    """زمان اسکن بعدی"""
    now = datetime.utcnow()
    next_hour = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    
    remaining = next_hour - now
    minutes = int(remaining.total_seconds() // 60)
    seconds = int(remaining.total_seconds() % 60)
    
    return f"{minutes:02d}:{seconds:02d}"

def calculate_time_until(target_time: str) -> str:
    """
    محاسبه زمان باقی‌مانده تا زمان هدف
    target_time: 'HH:MM' format
    """
    now = datetime.utcnow()
    target_hour, target_minute = map(int, target_time.split(':'))
    
    target = now.replace(hour=target_hour, minute=target_minute, second=0, microsecond=0)
    
    if target < now:
        target += timedelta(days=1)
    
    remaining = target - now
    hours = int(remaining.total_seconds() // 3600)
    minutes = int((remaining.total_seconds() % 3600) // 60)
    
    return f"{hours}h {minutes}m"

# ============================================
# 📈 Signal Scoring
# ============================================

class SignalScorer:
    """
    سیستم امتیازدهی به سیگنال‌ها
    """
    
    def __init__(self):
        self.weights = {
            'zlma_signal': 25,
            'smart_money': 20,
            'rsi_divergence': 15,
            'ichimoku': 15,
            'volume_confirmation': 10,
            'market_structure': 10,
            'support_resistance': 5
        }
    
    def calculate_score(self, signal_data: Dict, df: pd.DataFrame) -> Dict:
        """
        محاسبه امتیاز سیگنال
        """
        score = 0
        max_score = sum(self.weights.values())
        breakdown = {}
        
        # 1. ZLMA Signal
        if signal_data.get('zlma_signal_up') or signal_data.get('zlma_signal_dn'):
            score += self.weights['zlma_signal']
            breakdown['zlma'] = self.weights['zlma_signal']
        
        # 2. Smart Money
        if signal_data.get('smart_money_buy') or signal_data.get('smart_money_sell'):
            score += self.weights['smart_money']
            breakdown['smart_money'] = self.weights['smart_money']
        
        # 3. RSI Divergence
        if signal_data.get('rsi_bull_div') or signal_data.get('rsi_bear_div'):
            score += self.weights['rsi_divergence']
            breakdown['rsi_divergence'] = self.weights['rsi_divergence']
        
        # 4. Ichimoku
        if signal_data.get('ichimoku_buy') or signal_data.get('ichimoku_sell'):
            score += self.weights['ichimoku']
            breakdown['ichimoku'] = self.weights['ichimoku']
        
        # 5. Volume Confirmation
        current_volume = df['volume'].iloc[-1]
        avg_volume = df['volume'].rolling(20).mean().iloc[-1]
        
        if current_volume > avg_volume * 1.5:
            score += self.weights['volume_confirmation']
            breakdown['volume'] = self.weights['volume_confirmation']
        
        # 6. Market Structure
        market_structure = calculate_market_structure(df)
        trend_aligned = False
        
        if signal_data.get('type') == 'BUY' and market_structure['trend'] == 'uptrend':
            trend_aligned = True
        elif signal_data.get('type') == 'SELL' and market_structure['trend'] == 'downtrend':
            trend_aligned = True
        
        if trend_aligned:
            score += self.weights['market_structure']
            breakdown['market_structure'] = self.weights['market_structure']
        
        # 7. Support/Resistance
        sr_levels = calculate_support_resistance(df)
        near_level = False
        
        if signal_data.get('type') == 'BUY' and sr_levels.get('nearest_support'):
            near_level = True
        elif signal_data.get('type') == 'SELL' and sr_levels.get('nearest_resistance'):
            near_level = True
        
        if near_level:
            score += self.weights['support_resistance']
            breakdown['support_resistance'] = self.weights['support_resistance']
        
        # Normalize score to percentage
        score_percent = (score / max_score) * 100
        
        return {
            'score': round(score_percent, 1),
            'raw_score': score,
            'max_score': max_score,
            'breakdown': breakdown,
            'grade': self._get_grade(score_percent)
        }
    
    def _get_grade(self, score: float) -> str:
        """درجه‌بندی سیگنال"""
        if score >= 80:
            return 'A+ (Excellent)'
        elif score >= 70:
            return 'A (Very Good)'
        elif score >= 60:
            return 'B (Good)'
        elif score >= 50:
            return 'C (Fair)'
        else:
            return 'D (Weak)'

# ============================================
# 🔄 Cache Management
# ============================================

class DataCache:
    """
    مدیریت کش داده
    """
    
    def __init__(self, cache_dir: str = ".cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def get(self, key: str, max_age_minutes: int = 5) -> Optional[Any]:
        """
        دریافت از کش با بررسی عمر
        """
        cache_file = self.cache_dir / f"{key}.json"
        
        if not cache_file.exists():
            return None
        
        # چک کردن عمر کش
        file_age = datetime.now().timestamp() - cache_file.stat().st_mtime
        if file_age > (max_age_minutes * 60):
            cache_file.unlink(missing_ok=True)
            return None
        
        try:
            with open(cache_file, 'r') as f:
                return json.load(f)
        except:
            return None
    
    def set(self, key: str, data: Any):
        """
        ذخیره در کش
        """
        cache_file = self.cache_dir / f"{key}.json"
        
        try:
            with open(cache_file, 'w') as f:
                json.dump(data, f)
        except:
            pass
    
    def clear_old(self, max_age_hours: int = 24):
        """
        پاکسازی کش قدیمی
        """
        now = datetime.now().timestamp()
        
        for cache_file in self.cache_dir.glob("*.json"):
            file_age = now - cache_file.stat().st_mtime
            if file_age > (max_age_hours * 3600):
                cache_file.unlink(missing_ok=True)
    
    def clear_all(self):
        """پاکسازی تمام کش"""
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink(missing_ok=True)

# ============================================
# 🎯 Main Utility Functions
# ============================================

def initialize_utils(config: Dict) -> Dict:
    """
    راه‌اندازی اولیه utilities
    """
    logger = setup_logger("fast_scalp_bot")
    performance_tracker = PerformanceTracker()
    signal_scorer = SignalScorer()
    data_cache = DataCache()
    
    return {
        'logger': logger,
        'performance_tracker': performance_tracker,
        'signal_scorer': signal_scorer,
        'data_cache': data_cache
    }
