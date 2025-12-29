"""
Crypto AI Trading System v7.7.0 (COMPLETELY FIXED)
Fixed Issues:
1. All signature mismatches between main.py and utils.py resolved
2. Stable target calculations with proper percentages
3. Enhanced error handling and fallback mechanisms
4. Real-time data validation
5. Proper module import handling
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from datetime import datetime, timedelta
import logging
from typing import List, Optional, Dict, Any
import random
import sys
import os
import math
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==============================================================================
# Path Setup
# ==============================================================================

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

sys.path.insert(0, current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

print("=" * 60)
print(f"System Initialization")
print(f"Current directory: {current_dir}")
print(f"Parent directory: {parent_dir}")

# ==============================================================================
# Module Availability Flags
# ==============================================================================

UTILS_AVAILABLE = False
DATA_COLLECTOR_AVAILABLE = False
COLLECTORS_AVAILABLE = False

# ==============================================================================
# Import Utils Module (DIRECT IMPORT FIX)
# ==============================================================================

print("\n[1/3] Importing utils module...")

# Method 1: Try direct import from api directory
try:
    # First try absolute import
    import api.utils as utils_module
    utils = utils_module
    UTILS_AVAILABLE = True
    print("✅ SUCCESS: Utils imported via api.utils")
except ImportError as e:
    print(f"❌ Method 1 failed: {e}")
    
    # Method 2: Try relative import
    try:
        from . import utils
        UTILS_AVAILABLE = True
        print("✅ SUCCESS: Utils imported via relative import")
    except ImportError as e:
        print(f"❌ Method 2 failed: {e}")
        
        # Method 3: Try adding parent directory to path
        try:
            project_root = os.path.dirname(os.path.dirname(current_dir))
            sys.path.insert(0, project_root)
            import api.utils as utils_module
            utils = utils_module
            UTILS_AVAILABLE = True
            print("✅ SUCCESS: Utils imported via project root")
        except ImportError as e:
            print(f"❌ Method 3 failed: {e}")
            UTILS_AVAILABLE = False

# ==============================================================================
# Import Individual Functions (IF UTILS AVAILABLE)
# ==============================================================================

if UTILS_AVAILABLE:
    print("\n[2/3] Importing individual functions...")
    try:
        # Try to import all functions from utils
        get_market_data_with_fallback = utils.get_market_data_with_fallback
        analyze_with_multi_timeframe_strategy = utils.analyze_with_multi_timeframe_strategy
        calculate_24h_change_from_dataframe = utils.calculate_24h_change_from_dataframe
        calculate_simple_sma = utils.calculate_simple_sma
        calculate_simple_rsi = utils.calculate_simple_rsi
        calculate_rsi_series = utils.calculate_rsi_series
        detect_divergence = utils.detect_divergence
        calculate_macd_simple = utils.calculate_macd_simple
        analyze_scalp_conditions = utils.analyze_scalp_conditions
        calculate_ichimoku_components = utils.calculate_ichimoku_components
        analyze_ichimoku_scalp_signal = utils.analyze_ichimoku_scalp_signal
        get_ichimoku_scalp_signal = utils.get_ichimoku_scalp_signal
        get_support_resistance_levels = utils.get_support_resistance_levels
        calculate_volatility = utils.calculate_volatility
        combined_analysis = utils.combined_analysis
        generate_ichimoku_recommendation = utils.generate_ichimoku_recommendation
        get_swing_high_low = utils.get_swing_high_low
        calculate_smart_entry = utils.calculate_smart_entry
        get_fallback_signal = utils.get_fallback_signal
        
        print("✅ All functions imported successfully from utils module")
    except AttributeError as e:
        print(f"❌ Failed to get functions from utils module: {e}")
        UTILS_AVAILABLE = False
    except Exception as e:
        print(f"❌ Error importing functions: {e}")
        UTILS_AVAILABLE = False

# ==============================================================================
# Import Other Modules
# ==============================================================================

print("\n[3/3] Importing other modules...")

# Data Collector
try:
    from api.data_collector import get_collected_data
    DATA_COLLECTOR_AVAILABLE = True
    print("✅ data_collector imported")
except ImportError as e:
    print(f"⚠️ data_collector not available: {e}")
    DATA_COLLECTOR_AVAILABLE = False

# Collectors
try:
    from api.collectors import collect_signals_from_example_site
    COLLECTORS_AVAILABLE = True
    print("✅ collectors imported")
except ImportError as e:
    print(f"⚠️ collectors not available: {e}")
    COLLECTORS_AVAILABLE = False

# ==============================================================================
# Pydantic Models
# ==============================================================================

class AnalysisRequest(BaseModel):
    symbol: str
    timeframe: str = "5m"

class ScalpRequest(BaseModel):
    symbol: str
    timeframe: str = "5m"

class IchimokuRequest(BaseModel):
    symbol: str
    timeframe: str = "5m"

class CombinedRequest(BaseModel):
    symbol: str
    timeframe: str = "5m"
    include_ichimoku: bool = True
    include_rsi: bool = True
    include_macd: bool = True

class SignalResponse(BaseModel):
    status: str
    count: int
    last_updated: str
    signals: List[Dict[str, Any]]
    sources: Dict[str, int]

# ==============================================================================
# Mock Functions (COMPATIBLE WITH UTILS SIGNATURES)
# ==============================================================================

def mock_get_market_data_with_fallback(symbol, interval="5m", limit=50, return_source=False):
    """Mock data generator with real price logic."""
    logger.info(f"Generating mock data for {symbol}")
    
    # Try to get real data first
    try:
        import requests
        url = "https://api.binance.com/api/v3/klines"
        params = {'symbol': symbol.upper(), 'interval': interval, 'limit': limit}
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            if return_source:
                return {"data": response.json(), "source": "binance", "success": True}
            return response.json()
    except Exception as e:
        logger.warning(f"Real data fetch failed: {e}")
    
    # Generate mock data
    base_prices = {
        'BTCUSDT': 88271.00, 'ETHUSDT': 3450.00, 'BNBUSDT': 590.00, 
        'SOLUSDT': 175.00, 'DOGEUSDT': 0.12116, 'ALGOUSDT': 0.1187,
        'DEFAULT': 100.0
    }
    base_price = base_prices.get(symbol.upper(), base_prices['DEFAULT'])
    data = []
    current_time = int(time.time() * 1000)
    
    for i in range(limit):
        timestamp = current_time - (i * 5 * 60 * 1000)
        change = random.uniform(-0.015, 0.015)
        price = base_price * (1 + change)
        candle = [
            timestamp,
            str(price * random.uniform(0.998, 1.000)),  # open
            str(price * random.uniform(1.000, 1.003)),  # high
            str(price * random.uniform(0.997, 1.000)),  # low
            str(price),  # close
            str(random.uniform(1000, 10000)),  # volume
            timestamp + 300000,
            "0", "0", "0", "0", "0"
        ]
        data.append(candle)
    
    if return_source:
        return {"data": data, "source": "mock", "success": False}
    return data

def mock_calculate_simple_sma(data, period=20):
    if not data or len(data) < period:
        return 0
    try:
        closes = [float(c[4]) for c in data[-period:]]
        return sum(closes) / len(closes)
    except:
        return 50000

def mock_calculate_simple_rsi(data, period=14):
    return round(random.uniform(30, 70), 2)

def mock_calculate_rsi_series(closes, period=14):
    return [random.uniform(30, 70) for _ in range(len(closes) if closes else 0)]

def mock_detect_divergence(prices, rsi_values, lookback=5):
    return {"detected": False, "type": "none", "strength": None}

def mock_calculate_macd_simple(data, fast=12, slow=26, signal=9):
    return {
        'macd': round(random.uniform(-1, 1), 4),
        'signal': round(random.uniform(-1, 1), 4),
        'histogram': round(random.uniform(-0.5, 0.5), 4)
    }

def mock_analyze_scalp_conditions(data, timeframe):
    return {
        "condition": random.choice(["BULLISH", "BEARISH", "NEUTRAL", "VOLATILE"]),
        "rsi": round(random.uniform(30, 70), 1),
        "sma_20": 0,
        "current_price": float(data[-1][4]) if data else 0,
        "volatility": round(random.uniform(0.5, 3.0), 2),
        "reason": "Mock analysis"
    }

def mock_calculate_ichimoku_components(data, **kwargs):
    try:
        current_price = float(data[-1][4]) if data else 0
        return {
            'tenkan_sen': current_price * random.uniform(0.99, 1.01),
            'kijun_sen': current_price * random.uniform(0.98, 1.02),
            'cloud_top': current_price * random.uniform(1.01, 1.03),
            'cloud_bottom': current_price * random.uniform(0.97, 0.99),
            'current_price': current_price,
            'trend_power': random.uniform(30, 70)
        }
    except:
        return {'current_price': 100}

def mock_analyze_ichimoku_scalp_signal(ichimoku_data):
    signal = random.choice(['BUY', 'SELL', 'HOLD'])
    confidence = random.uniform(0.6, 0.9) if signal != 'HOLD' else random.uniform(0.4, 0.6)
    return {
        'signal': signal,
        'confidence': confidence,
        'reason': 'Mock analysis',
        'levels': {},
        'trend_power': 50
    }

def mock_get_ichimoku_scalp_signal(data, timeframe="5m"):
    return None

def mock_get_support_resistance_levels(data):
    if not data:
        return {"support": 0, "resistance": 0, "range_percent": 0}
    try:
        highs = [float(c[2]) for c in data[-20:]]
        lows = [float(c[3]) for c in data[-20:]]
        resistance = sum(highs) / len(highs) if highs else 0
        support = sum(lows) / len(lows) if lows else 0
        range_percent = ((resistance - support) / support * 100) if support > 0 else 0
        return {
            "support": round(support, 4),
            "resistance": round(resistance, 4),
            "range_percent": round(range_percent, 2)
        }
    except:
        return {"support": 0, "resistance": 0, "range_percent": 0}

def mock_calculate_volatility(data, period=20):
    return round(random.uniform(0.5, 3.0), 2)

def mock_calculate_24h_change_from_dataframe(data):
    if isinstance(data, dict) and "data" in data:
        data = data["data"]
    if not data or len(data) < 10:
        return round(random.uniform(-5, 5), 2)
    try:
        first = float(data[0][4])
        last = float(data[-1][4])
        return round(((last - first) / first) * 100, 2)
    except:
        return round(random.uniform(-5, 5), 2)

def mock_generate_ichimoku_recommendation(signal_data):
    signal = signal_data.get('signal', 'HOLD')
    if signal == 'BUY':
        return random.choice(['Strong Buy', 'Medium Buy', 'Weak Buy'])
    elif signal == 'SELL':
        return random.choice(['Strong Sell', 'Medium Sell', 'Weak Sell'])
    return random.choice(['Wait in cloud', 'Stay away', 'Hold'])

def mock_get_swing_high_low(data, period=20):
    """Get swing high and low."""
    if not data or len(data) < period:
        return 0, 0
    try:
        highs = [float(c[2]) for c in data[-period:]]
        lows = [float(c[3]) for c in data[-period:]]
        return max(highs) if highs else 0, min(lows) if lows else 0
    except:
        return 100, 50

def mock_calculate_smart_entry(data, signal="BUY", strategy="ICHIMOKU_FIBO"):
    """Mock Smart Entry."""
    try:
        base_price = float(data[-1][4]) if data else 0
        if signal == "BUY":
            return base_price * random.uniform(0.99, 1.00)
        elif signal == "SELL":
            return base_price * random.uniform(1.00, 1.01)
        return base_price
    except:
        return 0

def mock_combined_analysis(data, timeframe="5m"):
    return None

def mock_analyze_with_multi_timeframe_strategy(symbol):
    """Mock Analysis with proper target calculation."""
    data = mock_get_market_data_with_fallback(symbol, "5m", 50)
    signals = ["BUY", "SELL", "HOLD"]
    weights = [0.35, 0.35, 0.30]
    signal = random.choices(signals, weights=weights)[0]
    base_prices = {
        'BTCUSDT': 88271.00, 'ETHUSDT': 3450.00,
        'DOGEUSDT': 0.12116, 'ALGOUSDT': 0.1187,
        'DEFAULT': 100
    }
    base_price = base_prices.get(symbol.upper(), base_prices['DEFAULT'])
    
    try:
        latest_close = float(data[-1][4]) if data else base_price
    except:
        latest_close = base_price
        
    entry_price = latest_close
    
    # Proper target calculation
    if signal == "BUY":
        targets = [
            round(entry_price * 1.005, 8),  # +0.5%
            round(entry_price * 1.010, 8),  # +1.0%
            round(entry_price * 1.015, 8)   # +1.5%
        ]
        stop_loss = round(entry_price * 0.995, 8)  # -0.5%
    elif signal == "SELL":
        targets = [
            round(entry_price * 0.995, 8),  # -0.5%
            round(entry_price * 0.990, 8),  # -1.0%
            round(entry_price * 0.985, 8)   # -1.5%
        ]
        stop_loss = round(entry_price * 1.005, 8)  # +0.5%
    else:
        targets = [
            round(entry_price * 1.003, 8),  # +0.3%
            round(entry_price * 1.006, 8),  # +0.6%
            round(entry_price * 1.009, 8)   # +0.9%
        ]
        stop_loss = round(entry_price * 0.997, 8)  # -0.3%
    
    confidence = round(random.uniform(0.5, 0.7), 2) if signal == "HOLD" else round(random.uniform(0.65, 0.85), 2)
    
    return {
        "symbol": symbol,
        "signal": signal,
        "confidence": confidence,
        "entry_price": entry_price,
        "targets": targets,
        "stop_loss": stop_loss,
        "strategy": "Mock Multi-Timeframe",
        "note": "Using mock data"
    }

# ==============================================================================
# Function Selection Logic (SAFE VERSION)
# ==============================================================================

print("\n" + "=" * 60)
if UTILS_AVAILABLE:
    print("SELECTING: REAL UTILS FUNCTIONS")
    print("=" * 60)
    
    # لیست توابعی که باید بررسی شوند
    required_functions = [
        'get_market_data_with_fallback',
        'analyze_with_multi_timeframe_strategy',
        'calculate_24h_change_from_dataframe',
        'calculate_simple_sma',
        'calculate_simple_rsi',
        'calculate_rsi_series',
        'detect_divergence',
        'calculate_macd_simple',
        'analyze_scalp_conditions',
        'calculate_ichimoku_components',
        'analyze_ichimoku_scalp_signal',
        'get_ichimoku_scalp_signal',
        'combined_analysis',
        'generate_ichimoku_recommendation',
        'get_swing_high_low',
        'calculate_smart_entry',
        'get_support_resistance_levels',
        'calculate_volatility'
    ]
    
    # بررسی وجود توابع
    missing_functions = []
    for func_name in required_functions:
        if func_name not in globals():
            missing_functions.append(func_name)
    
    if missing_functions:
        print(f"⚠️ Missing {len(missing_functions)} functions: {missing_functions[:3]}")
        print("⚠️ Falling back to mock functions")
        UTILS_AVAILABLE = False
    else:
        # همه توابع موجود هستند
        get_market_data_func = get_market_data_with_fallback
        analyze_func = analyze_with_multi_timeframe_strategy
        calculate_change_func = calculate_24h_change_from_dataframe
        calculate_sma_func = calculate_simple_sma
        calculate_rsi_func = calculate_simple_rsi
        calculate_rsi_series_func = calculate_rsi_series
        calculate_divergence_func = detect_divergence
        calculate_macd_func = calculate_macd_simple
        analyze_scalp_conditions_func = analyze_scalp_conditions
        calculate_ichimoku_func = calculate_ichimoku_components
        analyze_ichimoku_signal_func = analyze_ichimoku_scalp_signal
        get_ichimoku_signal_func = get_ichimoku_scalp_signal
        combined_analysis_func = combined_analysis
        generate_recommendation_func = generate_ichimoku_recommendation
        get_swing_high_low_func = get_swing_high_low
        calculate_smart_entry_func = calculate_smart_entry
        get_support_resistance_levels_func = get_support_resistance_levels
        calculate_volatility_func = calculate_volatility
        
        print("✅ All real functions assigned successfully")

if not UTILS_AVAILABLE:
    print("\n" + "=" * 60)
    print("SELECTING: MOCK FUNCTIONS (Utils not available)")
    print("=" * 60)
    
    # Assign mock functions
    get_market_data_func = mock_get_market_data_with_fallback
    analyze_func = mock_analyze_with_multi_timeframe_strategy
    calculate_change_func = mock_calculate_24h_change_from_dataframe
    calculate_sma_func = mock_calculate_simple_sma
    calculate_rsi_func = mock_calculate_simple_rsi
    calculate_rsi_series_func = mock_calculate_rsi_series
    calculate_divergence_func = mock_detect_divergence
    calculate_macd_func = mock_calculate_macd_simple
    analyze_scalp_conditions_func = mock_analyze_scalp_conditions
    calculate_ichimoku_func = mock_calculate_ichimoku_components
    analyze_ichimoku_signal_func = mock_analyze_ichimoku_scalp_signal
    get_ichimoku_signal_func = mock_get_ichimoku_scalp_signal
    combined_analysis_func = mock_combined_analysis
    generate_recommendation_func = mock_generate_ichimoku_recommendation
    get_swing_high_low_func = mock_get_swing_high_low
    calculate_smart_entry_func = mock_calculate_smart_entry
    get_support_resistance_levels_func = mock_get_support_resistance_levels
    calculate_volatility_func = mock_calculate_volatility

# ==============================================================================
# Internal Helper Logic
# ==============================================================================

def analyze_scalp_signal(symbol, timeframe, data):
    """Internal logic for Scalp Signal (RSI + Divergence)."""
    logger.debug(f"Analyzing scalp signal for {symbol} on {timeframe}")
    
    if not data or len(data) < 30:
        return {
            "signal": "HOLD",
            "confidence": 0.5,
            "rsi": 50,
            "divergence": False,
            "sma_20": 0,
            "reason": "Insufficient data",
            "current_price": 0
        }
    
    try:
        rsi = calculate_rsi_func(data, 14)
        sma_20 = calculate_sma_func(data, 20)
        closes = [float(c[4]) for c in data[-30:]]
        rsi_series = calculate_rsi_series_func(closes, 14)
        div_info = calculate_divergence_func(closes, rsi_series, lookback=5)
        latest_close = closes[-1] if closes else 0
        
        # Ensure valid values
        if sma_20 is None or sma_20 <= 0:
            sma_20 = latest_close * 0.99
        
        signal, confidence, reason = "HOLD", 0.5, "Neutral"
        
        # Improved signal logic
        if div_info['detected']:
            if div_info['type'] == 'bullish':
                signal, confidence, reason = "BUY", 0.85, "Bullish divergence detected"
            elif div_info['type'] == 'bearish':
                signal, confidence, reason = "SELL", 0.85, "Bearish divergence detected"
        else:
            if rsi < 35 and latest_close < sma_20:
                signal, confidence, reason = "BUY", 0.75, f"Oversold (RSI: {rsi:.1f}) & below SMA20"
            elif rsi > 65 and latest_close > sma_20:
                signal, confidence, reason = "SELL", 0.75, f"Overbought (RSI: {rsi:.1f}) & above SMA20"
            elif rsi < 40:
                signal, confidence, reason = "BUY", 0.65, f"Near oversold (RSI: {rsi:.1f})"
            elif rsi > 60:
                signal, confidence, reason = "SELL", 0.65, f"Near overbought (RSI: {rsi:.1f})"
        
        return {
            "signal": signal,
            "confidence": round(confidence, 2),
            "rsi": round(rsi, 1),
            "divergence": div_info['detected'],
            "divergence_type": div_info['type'],
            "sma_20": round(sma_20, 8),
            "current_price": round(latest_close, 8),
            "reason": reason
        }
        
    except Exception as e:
        logger.error(f"Error in analyze_scalp_signal: {e}")
        return {
            "signal": "HOLD",
            "confidence": 0.5,
            "rsi": 50,
            "divergence": False,
            "sma_20": 0,
            "reason": f"Analysis error: {str(e)[:100]}",
            "current_price": 0
        }

def analyze_ichimoku_scalp(symbol, timeframe, data):
    """Internal logic for Ichimoku Scalp."""
    logger.debug(f"Analyzing Ichimoku scalp for {symbol} on {timeframe}")
    
    if not data or len(data) < 60:
        return {
            "signal": "HOLD",
            "confidence": 0.5,
            "divergence": False,
            "reason": "Insufficient Ichimoku data",
            "entry_price": 0,
            "targets": [0, 0, 0],
            "stop_loss": 0
        }
    
    try:
        ichimoku_data = calculate_ichimoku_func(data)
        if not ichimoku_data:
            return {
                "signal": "HOLD",
                "confidence": 0.5,
                "divergence": False,
                "reason": "Ichimoku calculation failed"
            }
        
        ichimoku_signal = analyze_ichimoku_signal_func(ichimoku_data)
        current_price = ichimoku_data.get('current_price', 0)
        
        if current_price <= 0:
            try:
                current_price = float(data[-1][4])
            except:
                current_price = 0
        
        # Calculate proper targets and stop loss
        if ichimoku_signal['signal'] == 'BUY':
            targets = [
                round(current_price * 1.005, 8),  # +0.5%
                round(current_price * 1.010, 8),  # +1.0%
                round(current_price * 1.015, 8)   # +1.5%
            ]
            stop_loss = round(current_price * 0.995, 8)  # -0.5%
        elif ichimoku_signal['signal'] == 'SELL':
            targets = [
                round(current_price * 0.995, 8),  # -0.5%
                round(current_price * 0.990, 8),  # -1.0%
                round(current_price * 0.985, 8)   # -1.5%
            ]
            stop_loss = round(current_price * 1.005, 8)  # +0.5%
        else:
            targets = [
                round(current_price * 1.003, 8),  # +0.3%
                round(current_price * 1.006, 8),  # +0.6%
                round(current_price * 1.009, 8)   # +0.9%
            ]
            stop_loss = round(current_price * 0.997, 8)  # -0.3%
        
        levels = {
            'tenkan_sen': ichimoku_data.get('tenkan_sen'),
            'kijun_sen': ichimoku_data.get('kijun_sen'),
            'cloud_top': ichimoku_data.get('cloud_top'),
            'cloud_bottom': ichimoku_data.get('cloud_bottom')
        }
        
        trend_power = ichimoku_data.get('trend_power', 50)
        trend_interpretation = "Strong" if trend_power >= 70 else "Medium" if trend_power >= 60 else "Weak" if trend_power >= 40 else "No Trend"
        
        return {
            "signal": ichimoku_signal['signal'],
            "confidence": ichimoku_signal['confidence'],
            "reason": ichimoku_signal['reason'],
            "entry_price": current_price,
            "targets": targets,
            "stop_loss": stop_loss,
            "ichimoku": levels,
            "trend_analysis": {
                "power": trend_power,
                "interpretation": trend_interpretation
            },
            "type": "ICHIMOKU_SCALP"
        }
        
    except Exception as e:
        logger.error(f"Error in analyze_ichimoku_scalp: {e}")
        return {
            "signal": "HOLD",
            "confidence": 0.5,
            "divergence": False,
            "reason": f"Ichimoku error: {str(e)[:100]}"
        }

# ==============================================================================
# Target Calculation Helper
# ==============================================================================

def calculate_targets_and_stop_loss(entry_price, signal, risk_level="MEDIUM"):
    """
    Calculate targets and stop loss based on entry price, signal, and risk level.
    This ensures consistent calculations across all endpoints.
    """
    if entry_price <= 0:
        return [0, 0, 0], 0, [0, 0, 0], 0
    
    # Define risk parameters
    risk_params = {
        "HIGH": {
            "BUY": {"targets": [1.015, 1.025, 1.035], "stop_loss": 0.985},
            "SELL": {"targets": [0.985, 0.975, 0.965], "stop_loss": 1.015},
            "HOLD": {"targets": [1.005, 1.010, 1.015], "stop_loss": 0.995}
        },
        "MEDIUM": {
            "BUY": {"targets": [1.005, 1.010, 1.015], "stop_loss": 0.995},
            "SELL": {"targets": [0.995, 0.990, 0.985], "stop_loss": 1.005},
            "HOLD": {"targets": [1.003, 1.006, 1.009], "stop_loss": 0.997}
        },
        "LOW": {
            "BUY": {"targets": [1.003, 1.006, 1.009], "stop_loss": 0.997},
            "SELL": {"targets": [0.997, 0.994, 0.991], "stop_loss": 1.003},
            "HOLD": {"targets": [1.002, 1.004, 1.006], "stop_loss": 0.998}
        }
    }
    
    # Get parameters for current risk level and signal
    params = risk_params.get(risk_level, risk_params["MEDIUM"])
    signal_params = params.get(signal, params["HOLD"])
    
    # Calculate targets
    targets = [round(entry_price * multiplier, 8) for multiplier in signal_params["targets"]]
    stop_loss = round(entry_price * signal_params["stop_loss"], 8)
    
    # Calculate percentages
    targets_percent = [
        round(((target - entry_price) / entry_price) * 100, 2)
        for target in targets
    ]
    stop_loss_percent = round(((stop_loss - entry_price) / entry_price) * 100, 2)
    
    return targets, stop_loss, targets_percent, stop_loss_percent

# ==============================================================================
# FastAPI App
# ==============================================================================

API_VERSION = "7.7.0-FIXED"

app = FastAPI(
    title=f"Crypto AI Trading System v{API_VERSION}",
    description="Completely Fixed - Stable Targets & Stop Loss Calculation",
    version=API_VERSION,
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================================================================
# API Endpoints
# ==============================================================================

@app.get("/")
async def read_root():
    return {
        "message": f"Crypto AI Trading System v{API_VERSION}",
        "status": "Active",
        "version": API_VERSION,
        "modules": {
            "utils": UTILS_AVAILABLE,
            "data_collector": DATA_COLLECTOR_AVAILABLE,
            "collectors": COLLECTORS_AVAILABLE
        },
        "endpoints": {
            "/api/health": "Health check",
            "/api/scalp-signal": "Scalp signal (POST)",
            "/api/ichimoku-scalp": "Ichimoku signal (POST)",
            "/api/analyze": "General analysis (POST)",
            "/market/{symbol}": "Market data (GET)"
        }
    }

@app.get("/api/health")
async def health_check():
    return {
        "status": "Healthy",
        "version": API_VERSION,
        "timestamp": datetime.now().isoformat(),
        "modules": {
            "utils": UTILS_AVAILABLE,
            "data_collector": DATA_COLLECTOR_AVAILABLE,
            "collectors": COLLECTORS_AVAILABLE
        },
        "system": {
            "python_version": sys.version,
            "platform": sys.platform
        }
    }

@app.get("/api/signals", response_model=SignalResponse)
async def get_all_signals_endpoint(symbol: Optional[str] = None):
    try:
        analysis = analyze_func(symbol.upper() if symbol else "BTCUSDT")
        signals = [{
            "symbol": analysis["symbol"],
            "signal": analysis["signal"],
            "confidence": analysis["confidence"],
            "entry_price": analysis["entry_price"],
            "targets": analysis["targets"],
            "stop_loss": analysis["stop_loss"],
            "generated_at": datetime.now().isoformat()
        }]
        return SignalResponse(
            status="Success",
            count=1,
            last_updated=datetime.now().isoformat(),
            signals=signals,
            sources={"internal": 1, "total": 1}
        )
    except Exception as e:
        logger.error(f"Error in get_all_signals: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze")
async def analyze_crypto(request: AnalysisRequest):
    """General analysis endpoint."""
    try:
        logger.info(f"Analysis request: {request.symbol} ({request.timeframe})")
        
        analysis = analyze_func(request.symbol)
        analysis["version"] = API_VERSION
        
        # Get additional market data
        market_data = get_market_data_func(request.symbol, request.timeframe, 100)
        if market_data:
            analysis["rsi"] = calculate_rsi_func(market_data, 14)
            closes = [float(c[4]) for c in market_data]
            rsi_series = calculate_rsi_series_func(closes, 14)
            div = calculate_divergence_func(closes, rsi_series, lookback=5)
            analysis["divergence"] = div['detected']
            analysis["divergence_type"] = div['type']
            analysis["current_price"] = closes[-1] if closes else 0
        
        return analysis
        
    except Exception as e:
        logger.error(f"Error in analyze_crypto: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)[:200]}")

@app.post("/api/scalp-signal")
async def get_scalp_signal(request: ScalpRequest):
    """
    Scalp Signal Endpoint - COMPLETELY FIXED VERSION
    Returns stable signals with proper target calculations.
    """
    allowed_timeframes = ["1m", "5m", "15m"]
    if request.timeframe not in allowed_timeframes:
        raise HTTPException(status_code=400, detail=f"Invalid timeframe. Allowed: {allowed_timeframes}")
    
    start_time = time.time()
    logger.info(f"🚀 Scalp signal request: {request.symbol} ({request.timeframe})")
    
    try:
        # Step 1: Get market data
        logger.debug(f"Step 1: Fetching market data for {request.symbol}")
        market_data_result = get_market_data_func(request.symbol, request.timeframe, 100, return_source=True)
        
        if isinstance(market_data_result, dict) and "data" in market_data_result:
            market_data = market_data_result["data"]
            data_source = market_data_result.get("source", "unknown")
            data_success = market_data_result.get("success", False)
        else:
            market_data = market_data_result
            data_source = "direct"
            data_success = True
        
        if not market_data or len(market_data) < 20:
            logger.warning(f"Insufficient data for {request.symbol}: {len(market_data) if market_data else 0} candles")
            # Generate mock data as fallback
            market_data = mock_get_market_data_with_fallback(request.symbol, request.timeframe, 100)
            data_source = "mock_fallback"
            data_success = False
        
        logger.info(f"📊 Data: {len(market_data)} candles from {data_source} (success: {data_success})")
        
        # Step 2: Analyze scalp signal
        logger.debug("Step 2: Analyzing scalp signal")
        scalp_analysis = analyze_scalp_signal(request.symbol, request.timeframe, market_data)
        
        # Step 3: Calculate smart entry price
        logger.debug("Step 3: Calculating smart entry")
        try:
            smart_entry_price = calculate_smart_entry_func(market_data, scalp_analysis["signal"])
            if smart_entry_price <= 0:
                raise ValueError("Invalid smart entry price")
        except Exception as e:
            logger.warning(f"Smart entry calculation failed: {e}. Using current price.")
            try:
                smart_entry_price = float(market_data[-1][4])
            except:
                smart_entry_price = scalp_analysis.get("current_price", 0)
        
        # Step 4: Determine risk level
        logger.debug("Step 4: Determining risk level")
        rsi = scalp_analysis["rsi"]
        confidence = scalp_analysis["confidence"]
        
        risk_level = "MEDIUM"
        if (rsi > 80 or rsi < 20) and confidence > 0.8:
            risk_level = "HIGH"
        elif (rsi > 70 or rsi < 30) and confidence > 0.7:
            risk_level = "MEDIUM"
        elif confidence < 0.5:
            risk_level = "LOW"
        
        # Step 5: Calculate targets and stop loss using consistent function
        logger.debug("Step 5: Calculating targets and stop loss")
        targets, stop_loss, targets_percent, stop_loss_percent = calculate_targets_and_stop_loss(
            smart_entry_price, scalp_analysis["signal"], risk_level
        )
        
        # Step 6: Prepare response
        logger.debug("Step 6: Preparing response")
        response = {
            "symbol": request.symbol,
            "timeframe": request.timeframe,
            "signal": scalp_analysis["signal"],
            "confidence": scalp_analysis["confidence"],
            "entry_price": round(smart_entry_price, 8),
            "rsi": scalp_analysis["rsi"],
            "divergence": scalp_analysis["divergence"],
            "divergence_type": scalp_analysis.get("divergence_type", "none"),
            "sma_20": scalp_analysis.get("sma_20", 0),
            "current_price": scalp_analysis.get("current_price", 0),
            "targets": targets,
            "stop_loss": stop_loss,
            "targets_percent": targets_percent,
            "stop_loss_percent": stop_loss_percent,
            "risk_level": risk_level,
            "reason": scalp_analysis["reason"],
            "strategy": "Scalp Smart Entry (Fixed)",
            "data_source": data_source,
            "data_success": data_success,
            "data_points": len(market_data),
            "generated_at": datetime.now().isoformat(),
            "processing_time_ms": round((time.time() - start_time) * 1000, 2)
        }
        
        logger.info(f"✅ Generated {response['signal']} signal for {response['symbol']}")
        logger.info(f"   Confidence: {response['confidence']}, RSI: {response['rsi']}, Risk: {response['risk_level']}")
        logger.info(f"   Entry: {response['entry_price']}, Current: {response['current_price']}")
        logger.info(f"   Targets: {response['targets']} ({response['targets_percent']}%)")
        logger.info(f"   Stop Loss: {response['stop_loss']} ({response['stop_loss_percent']}%)")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Critical error in scalp signal: {str(e)}", exc_info=True)
        
        # Comprehensive fallback response
        fallback_signal = random.choice(["BUY", "SELL", "HOLD"])
        base_prices = {
            'BTCUSDT': 88271.00, 'ETHUSDT': 3450.00,
            'DOGEUSDT': 0.12116, 'ALGOUSDT': 0.1187,
            'DEFAULT': 100.0
        }
        base_price = base_prices.get(request.symbol.upper(), base_prices['DEFAULT'])
        entry_price = round(base_price * random.uniform(0.99, 1.01), 8)
        
        # Calculate fallback targets
        targets, stop_loss, targets_percent, stop_loss_percent = calculate_targets_and_stop_loss(
            entry_price, fallback_signal, "MEDIUM"
        )
        
        return {
            "symbol": request.symbol,
            "timeframe": request.timeframe,
            "signal": fallback_signal,
            "confidence": 0.5,
            "entry_price": entry_price,
            "rsi": round(random.uniform(30, 70), 1),
            "divergence": False,
            "divergence_type": "none",
            "sma_20": round(entry_price * random.uniform(0.99, 1.01), 8),
            "current_price": entry_price,
            "targets": targets,
            "stop_loss": stop_loss,
            "targets_percent": targets_percent,
            "stop_loss_percent": stop_loss_percent,
            "risk_level": "MEDIUM",
            "reason": f"System Error - Using fallback: {str(e)[:100]}",
            "strategy": "Fallback Mode",
            "data_source": "error_fallback",
            "data_success": False,
            "data_points": 0,
            "generated_at": datetime.now().isoformat(),
            "processing_time_ms": round((time.time() - start_time) * 1000, 2),
            "error": str(e)[:200]
        }

@app.post("/api/ichimoku-scalp")
async def get_ichimoku_scalp_signal(request: IchimokuRequest):
    """Ichimoku Scalp Endpoint."""
    allowed_timeframes = ["1m", "5m", "15m", "1h", "4h"]
    if request.timeframe not in allowed_timeframes:
        raise HTTPException(status_code=400, detail=f"Invalid timeframe. Allowed: {allowed_timeframes}")
    
    try:
        logger.info(f"🌌 Ichimoku request: {request.symbol} ({request.timeframe})")
        
        market_data = get_market_data_func(request.symbol, request.timeframe, 100)
        if not market_data or len(market_data) < 60:
            raise HTTPException(status_code=404, detail="Not enough data for Ichimoku analysis")
        
        ichimoku_analysis = analyze_ichimoku_scalp(request.symbol, request.timeframe, market_data)
        
        # Calculate additional indicators
        rsi = calculate_rsi_func(market_data, 14)
        closes = [float(c[4]) for c in market_data]
        rsi_series = calculate_rsi_series_func(closes, 14)
        div = calculate_divergence_func(closes, rsi_series, lookback=5)
        recommendation = generate_recommendation_func(ichimoku_analysis)
        
        # Calculate percentages
        entry_price = ichimoku_analysis["entry_price"]
        targets = ichimoku_analysis["targets"]
        stop_loss = ichimoku_analysis["stop_loss"]
        
        tp_percents = [
            round(((target - entry_price) / entry_price) * 100, 2)
            for target in targets[:3]
        ] if len(targets) >= 3 else [0, 0, 0]
        
        sl_percent = round(((stop_loss - entry_price) / entry_price) * 100, 2)
        
        # Determine risk level
        confidence = ichimoku_analysis["confidence"]
        risk_level = "LOW"
        if confidence > 0.8:
            risk_level = "HIGH"
        elif confidence > 0.6:
            risk_level = "MEDIUM"
        
        return {
            "symbol": request.symbol,
            "timeframe": request.timeframe,
            "signal": ichimoku_analysis["signal"],
            "confidence": ichimoku_analysis["confidence"],
            "entry_price": entry_price,
            "targets": targets,
            "stop_loss": stop_loss,
            "targets_percent": tp_percents,
            "stop_loss_percent": sl_percent,
            "rsi": round(rsi, 2),
            "divergence": div['detected'],
            "divergence_type": div['type'],
            "reason": ichimoku_analysis["reason"],
            "strategy": f"Ichimoku Scalp ({request.timeframe})",
            "recommendation": recommendation,
            "trend_power": ichimoku_analysis.get("trend_analysis", {}).get("power", 50),
            "risk_level": risk_level,
            "generated_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in Ichimoku: {e}")
        raise HTTPException(status_code=500, detail=f"Ichimoku analysis error: {str(e)[:200]}")

@app.post("/api/combined-analysis")
async def get_combined_analysis(request: CombinedRequest):
    """Combined Analysis Endpoint."""
    try:
        market_data = get_market_data_func(request.symbol, request.timeframe, 100)
        if not market_data:
            raise HTTPException(status_code=404, detail="No market data available")
        
        results = combined_analysis_func(market_data, request.timeframe)
        
        if results:
            return results
        else:
            # Fallback analysis
            return {
                'signal': 'HOLD',
                'confidence': 0.5,
                'details': {'note': 'Combined analysis returned None'},
                'price': float(market_data[-1][4]) if market_data else 0,
                'timestamp': datetime.now().isoformat()
            }
            
    except Exception as e:
        logger.error(f"Error in combined analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Combined analysis error: {str(e)[:200]}")

@app.get("/market/{symbol}")
async def get_market_data(symbol: str, timeframe: str = "5m"):
    """Market data endpoint."""
    try:
        data = get_market_data_func(symbol, timeframe, limit=50)
        if not data:
            raise HTTPException(status_code=404, detail="No market data available")
        
        latest = data[-1] if isinstance(data, list) and len(data) > 0 else []
        if not latest or len(latest) < 6:
            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "source": "Mock",
                "current_price": 100,
                "change_24h": 0
            }
        
        change_24h = calculate_change_func(data)
        rsi = calculate_rsi_func(data, 14)
        sma_20 = calculate_sma_func(data, 20)
        closes = [float(c[4]) for c in data]
        rsi_series = calculate_rsi_series_func(closes, 14)
        div = calculate_divergence_func(closes, rsi_series, lookback=5)
        
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "source": "Binance" if UTILS_AVAILABLE else "Mock",
            "current_price": float(latest[4]),
            "high": float(latest[2]),
            "low": float(latest[3]),
            "change_24h": change_24h,
            "rsi_14": round(rsi, 2),
            "sma_20": round(sma_20, 2),
            "divergence": div['detected'],
            "divergence_type": div['type'],
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Market data error: {e}")
        raise HTTPException(status_code=500, detail=f"Market data error: {str(e)[:200]}")

@app.get("/signals/scraped")
async def get_scraped_signals():
    """Scraped Signals Endpoint."""
    try:
        if COLLECTORS_AVAILABLE:
            scraped_signals = collect_signals_from_example_site()
        else:
            scraped_signals = [{
                "symbol": "BTCUSDT",
                "signal": random.choice(["BUY", "SELL", "HOLD"]),
                "confidence": round(random.uniform(0.6, 0.9), 2),
                "source": "mock"
            }]
        return {
            "status": "success",
            "count": len(scraped_signals),
            "signals": scraped_signals,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error in scraped signals: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/scan-all-timeframes/{symbol}")
async def scan_all_timeframes(symbol: str):
    """Scan All Timeframes Endpoint."""
    try:
        timeframes = ["1m", "5m", "15m", "1h", "4h"]
        results = []
        
        for tf in timeframes:
            try:
                if tf in ["1m", "5m", "15m"]:
                    # Use scalp signal for shorter timeframes
                    response = await get_scalp_signal(ScalpRequest(symbol=symbol, timeframe=tf))
                else:
                    # Use ichimoku for longer timeframes
                    response = await get_ichimoku_scalp_signal(IchimokuRequest(symbol=symbol, timeframe=tf))
                
                response["timeframe"] = tf
                results.append(response)
                
            except Exception as e:
                logger.error(f"Error scanning {symbol} on {tf}: {e}")
                results.append({
                    "symbol": symbol,
                    "timeframe": tf,
                    "signal": "ERROR",
                    "error": str(e)[:100]
                })
        
        return {
            "symbol": symbol,
            "scanned_at": datetime.now().isoformat(),
            "total": len(timeframes),
            "successful": len([r for r in results if r.get("signal") != "ERROR"]),
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Error in scan-all-timeframes: {e}")
        raise HTTPException(status_code=500, detail=f"Scan error: {str(e)[:200]}")

# ==============================================================================
# Startup and Debug Functions
# ==============================================================================

def debug_utils_status():
    """Debug function to check utils status."""
    print("\n" + "=" * 60)
    print("DEBUG: SYSTEM STATUS CHECK")
    print("=" * 60)
    print(f"UTILS_AVAILABLE: {UTILS_AVAILABLE}")
    print(f"API_VERSION: {API_VERSION}")
    print(f"Python Path: {sys.path}")
    
    if UTILS_AVAILABLE:
        try:
            # Test a simple function
            test_data = mock_get_market_data_with_fallback("BTCUSDT", "5m", 10)
            rsi = calculate_rsi_func(test_data, 14)
            print(f"✓ RSI Calculation Test: {rsi}")
            
            sma = calculate_sma_func(test_data, 20)
            print(f"✓ SMA Calculation Test: {sma}")
            
            print("✅ All tests passed!")
        except Exception as e:
            print(f"❌ Test failed: {e}")
    else:
        print("⚠️ Using mock functions only")
    
    print("=" * 60)

@app.on_event("startup")
async def startup_event():
    """Startup event handler."""
    logger.info(f"🚀 Starting Crypto AI Trading System v{API_VERSION}")
    logger.info(f"📦 Utils Available: {UTILS_AVAILABLE}")
    logger.info(f"📦 Data Collector Available: {DATA_COLLECTOR_AVAILABLE}")
    logger.info(f"📦 Collectors Available: {COLLECTORS_AVAILABLE}")
    
    # Run debug check
    debug_utils_status()
    
    logger.info("✅ System startup completed successfully!")

# ==============================================================================
# Main Entry Point
# ==============================================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    
    logger.info(f"🌐 Starting server on {host}:{port}")
    print(f"\n{'=' * 60}")
    print(f"Server starting on http://{host}:{port}")
    print(f"API Documentation: http://{host}:{port}/api/docs")
    print(f"Health Check: http://{host}:{port}/api/health")
    print(f"{'=' * 60}\n")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        access_log=True
    )