import pandas as pd
import numpy as np
from typing import Tuple, Dict
import logging
from ta.trend import MACD
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange

logger = logging.getLogger(__name__)

class IndicatorProcessor:
    def __init__(self, config: dict):
        self.zlma_length = config['zlma_length']
        self.rsi_length = config['rsi_length']
        self.atr_period = config['atr_period']
    
    def calculate_zlma(self, df: pd.DataFrame) -> pd.Series:
        """محاسبه ZLMA (Zero-Lag Moving Average)"""
        ema = df['close'].ewm(span=self.zlma_length).mean()
        correction = df['close'] + (df['close'] - ema)
        zlma = correction.ewm(span=self.zlma_length).mean()
        return zlma
    
    def calculate_ema(self, df: pd.DataFrame) -> pd.Series:
        """محاسبه EMA"""
        return df['close'].ewm(span=self.zlma_length).mean()
    
    def calculate_smart_money_signals(self, df: pd.DataFrame) -> Dict:
        """محاسبه سیگنال‌های Smart Money"""
        signals = {
            'bh_long': False,
            'bh_short': False,
            'in_range': False,
            'near_support': False,
            'near_resistance': False
        }
        
        try:
            # Range Box Detection
            box_lookback = 50
            box_top = df['high'].rolling(window=box_lookback).max()
            box_bottom = df['low'].rolling(window=box_lookback).min()
            box_range = box_top - box_bottom
            
            current_close = df['close'].iloc[-1]
            signals['in_range'] = (current_close > box_bottom.iloc[-1]) and (current_close < box_top.iloc[-1])
            signals['near_support'] = current_close <= (box_bottom.iloc[-1] + box_range.iloc[-1] * 0.2)
            signals['near_resistance'] = current_close >= (box_top.iloc[-1] - box_range.iloc[-1] * 0.2)
            
            # Wave Trend Calculation
            n1, n2, n3 = 9, 6, 3
            src = df[['high', 'low', 'close']].mean(axis=1)
            
            # محاسبه TCI
            ema_n1 = src.ewm(span=n1).mean()
            abs_diff = (src - ema_n1).abs()
            ema_abs = abs_diff.ewm(span=n1).mean()
            tci = ((src - ema_n1) / (0.025 * ema_abs)).ewm(span=n2).mean() + 50
            
            # محاسبه MF
            positive_flow = np.where(src.diff() > 0, df['volume'] * src, 0)
            negative_flow = np.where(src.diff() < 0, df['volume'] * src, 0)
            
            sum_pos = pd.Series(positive_flow).rolling(window=n3).sum()
            sum_neg = pd.Series(negative_flow).rolling(window=n3).sum()
            mf = 100 - 100 / (1 + sum_pos / sum_neg.replace(0, np.nan))
            
            # RSI
            rsi = RSIIndicator(close=src, window=n3).rsi()
            
            # Wave Trend ترکیبی
            wt1 = pd.concat([tci, mf, rsi], axis=1).mean(axis=1)
            wt2 = wt1.rolling(window=6).mean()
            
            # سیگنال‌های Boom Hunter
            q1 = wt1  # ساده‌سازی برای نمونه
            trigger = wt2
            
            signals['bh_long'] = (q1.iloc[-1] > trigger.iloc[-1]) and (q1.iloc[-2] <= trigger.iloc[-2])
            signals['bh_short'] = (q1.iloc[-1] < trigger.iloc[-1]) and (q1.iloc[-2] >= trigger.iloc[-2])
            
        except Exception as e:
            logger.error(f"Error calculating smart money signals: {e}")
        
        return signals
    
    def calculate_rsi_divergence(self, df: pd.DataFrame) -> Dict:
        """محاسبه RSI Divergence"""
        signals = {
            'bull_div': False,
            'bear_div': False,
            'rsi_value': 50
        }
        
        try:
            rsi = RSIIndicator(close=df['close'], window=14).rsi()
            signals['rsi_value'] = rsi.iloc[-1]
            
            # تشخیص سقف و کف قیمتی
            lookback = 15
            pivot_highs = []
            pivot_lows = []
            
            for i in range(lookback, len(df)-lookback):
                if df['high'].iloc[i] == df['high'].iloc[i-lookback:i+lookback].max():
                    pivot_highs.append((i, df['high'].iloc[i], rsi.iloc[i]))
                if df['low'].iloc[i] == df['low'].iloc[i-lookback:i+lookback].min():
                    pivot_lows.append((i, df['low'].iloc[i], rsi.iloc[i]))
            
            # بررسی Bullish Divergence
            if len(pivot_lows) >= 2:
                last_low = pivot_lows[-1]
                prev_low = pivot_lows[-2]
                if (last_low[1] < prev_low[1]) and (last_low[2] > prev_low[2]):
                    signals['bull_div'] = True
            
            # بررسی Bearish Divergence
            if len(pivot_highs) >= 2:
                last_high = pivot_highs[-1]
                prev_high = pivot_highs[-2]
                if (last_high[1] > prev_high[1]) and (last_high[2] < prev_high[2]):
                    signals['bear_div'] = True
                    
        except Exception as e:
            logger.error(f"Error calculating RSI divergence: {e}")
        
        return signals
    
    def calculate_ichimoku(self, df: pd.DataFrame) -> Dict:
        """محاسبه ابر ایچیموکو"""
        signals = {
            'tenkan_sen': 0,
            'kijun_sen': 0,
            'senkou_span_a': 0,
            'senkou_span_b': 0,
            'above_cloud': False,
            'future_cloud_bullish': False
        }
        
        try:
            high = df['high']
            low = df['low']
            
            # Tenkan-sen (Conversion Line)
            period9_high = high.rolling(window=9).max()
            period9_low = low.rolling(window=9).min()
            signals['tenkan_sen'] = ((period9_high + period9_low) / 2).iloc[-1]
            
            # Kijun-sen (Base Line)
            period26_high = high.rolling(window=26).max()
            period26_low = low.rolling(window=26).min()
            signals['kijun_sen'] = ((period26_high + period26_low) / 2).iloc[-1]
            
            # Senkou Span A
            signals['senkou_span_a'] = ((signals['tenkan_sen'] + signals['kijun_sen']) / 2)
            
            # Senkou Span B
            period52_high = high.rolling(window=52).max()
            period52_low = low.rolling(window=52).min()
            signals['senkou_span_b'] = ((period52_high.iloc[-1] + period52_low.iloc[-1]) / 2)
            
            current_close = df['close'].iloc[-1]
            signals['above_cloud'] = current_close > max(signals['senkou_span_a'], signals['senkou_span_b'])
            
            # آینده ابر
            signals['future_cloud_bullish'] = signals['senkou_span_a'] > signals['senkou_span_b']
            
        except Exception as e:
            logger.error(f"Error calculating Ichimoku: {e}")
        
        return signals
    
    def calculate_macd(self, df: pd.DataFrame) -> Dict:
        """محاسبه MACD"""
        signals = {
            'macd': 0,
            'signal': 0,
            'histogram': 0,
            'bullish': False,
            'bearish': False
        }
        
        try:
            macd_indicator = MACD(
                close=df['close'],
                window_fast=12,
                window_slow=26,
                window_sign=9
            )
            
            signals['macd'] = macd_indicator.macd().iloc[-1]
            signals['signal'] = macd_indicator.macd_signal().iloc[-1]
            signals['histogram'] = macd_indicator.macd_diff().iloc[-1]
            signals['bullish'] = signals['histogram'] > 0
            signals['bearish'] = signals['histogram'] < 0
            
        except Exception as e:
            logger.error(f"Error calculating MACD: {e}")
        
        return signals
    
    def calculate_atr(self, df: pd.DataFrame) -> float:
        """محاسبه ATR برای مدیریت ریسک"""
        try:
            atr_indicator = AverageTrueRange(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                window=self.atr_period
            )
            return atr_indicator.average_true_range().iloc[-1]
        except Exception as e:
            logger.error(f"Error calculating ATR: {e}")
            return 0
