# api/collectors.py - فوق‌ساده (۵ خط)
import random
from datetime import datetime

def collect_signals_from_example_site():
    """ساده‌ترین نسخه بدون مشکل"""
    return [{
        "symbol": "BTCUSDT",
        "signal": random.choice(["BUY", "SELL", "HOLD"]),
        "confidence": 0.7,
        "timestamp": datetime.now().isoformat()
    }]

print("✅ collectors.py loaded")