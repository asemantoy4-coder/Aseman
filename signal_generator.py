import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class SignalType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    NEUTRAL = "NEUTRAL"

@dataclass
class Signal:
    symbol: str
    signal_type: SignalType
    confidence: float  # 0-100
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    reason: str
    timestamp: pd.Timestamp

class SignalGenerator:
    def __init__(self, config: dict):
        self.config = config
        self.min_confidence = 70  # حداقل اطمینان برای سیگنال
    
    def generate_signals(self, symbol: str, df: pd.DataFrame, 
                        indicators: Dict) -> List[Signal]:
        """تولید سیگنال‌های معاملاتی"""
        signals = []
        
        if len(df) < 100:  # حداقل داده مورد نیاز
            return signals
        
        try:
            # 1. بررسی شرایط فست اسکلپ با ZLMA
            zlma = indicators.get('zlma')
            ema = indicators.get('ema')
            
            if zlma is not None and ema is not None:
                # سیگنال‌های اصلی ZLMA
                current_zlma = zlma.iloc[-1]
                prev_zlma = zlma.iloc[-2]
                current_ema = ema.iloc[-1]
                prev_ema = ema.iloc[-2]
                
                # کراس‌اور صعودی
                if (prev_zlma <= prev_ema) and (current_zlma > current_ema):
                    confidence = self._calculate_confidence(df, indicators, 'buy')
                    if confidence >= self.min_confidence:
                        signal = self._create_buy_signal(symbol, df, indicators, confidence)
                        signals.append(signal)
                
                # کراس‌اور نزولی
                elif (prev_zlma >= prev_ema) and (current_zlma < current_ema):
                    confidence = self._calculate_confidence(df, indicators, 'sell')
                    if confidence >= self.min_confidence:
                        signal = self._create_sell_signal(symbol, df, indicators, confidence)
                        signals.append(signal)
            
            # 2. بررسی RSI Divergence
            rsi_signals = indicators.get('rsi_signals', {})
            if rsi_signals.get('bull_div'):
                confidence = self._calculate_confidence(df, indicators, 'buy')
                confidence *= 1.2  # افزایش اطمینان برای دیورژانس
                if confidence >= self.min_confidence:
                    signal = self._create_buy_signal(symbol, df, indicators, confidence)
                    signal.reason += " + RSI Bullish Divergence"
                    signals.append(signal)
            
            if rsi_signals.get('bear_div'):
                confidence = self._calculate_confidence(df, indicators, 'sell')
                confidence *= 1.2
                if confidence >= self.min_confidence:
                    signal = self._create_sell_signal(symbol, df, indicators, confidence)
                    signal.reason += " + RSI Bearish Divergence"
                    signals.append(signal)
            
            # 3. بررسی هماهنگی ایچیموکو
            ichimoku = indicators.get('ichimoku', {})
            if ichimoku.get('above_cloud') and ichimoku.get('future_cloud_bullish'):
                # تایید روند صعودی
                for signal in signals:
                    if signal.signal_type == SignalType.BUY:
                        signal.confidence *= 1.1
                        signal.reason += " + Ichimoku Cloud Support"
            
            elif not ichimoku.get('above_cloud') and not ichimoku.get('future_cloud_bullish'):
                # تایید روند نزولی
                for signal in signals:
                    if signal.signal_type == SignalType.SELL:
                        signal.confidence *= 1.1
                        signal.reason += " + Ichimoku Cloud Resistance"
            
            # فیلتر کردن سیگنال‌های با اطمینان کافی
            signals = [s for s in signals if s.confidence >= self.min_confidence]
            
        except Exception as e:
            logger.error(f"Error generating signals for {symbol}: {e}")
        
        return signals
    
    def _calculate_confidence(self, df: pd.DataFrame, 
                             indicators: Dict, signal_type: str) -> float:
        """محاسبه میزان اطمینان سیگنال"""
        confidence = 50  # حد پایه
        
        try:
            # 1. حجم معاملات (20 امتیاز)
            current_volume = df['volume'].iloc[-1]
            avg_volume = df['volume'].rolling(20).mean().iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            if volume_ratio > 1.5:
                confidence += 20
            elif volume_ratio > 1.2:
                confidence += 10
            elif volume_ratio < 0.8:
                confidence -= 10
            
            # 2. قدرت روند (15 امتیاز)
            sm_signals = indicators.get('smart_money', {})
            if signal_type == 'buy':
                if sm_signals.get('near_support'):
                    confidence += 15
                if sm_signals.get('bh_long'):
                    confidence += 10
            else:  # sell
                if sm_signals.get('near_resistance'):
                    confidence += 15
                if sm_signals.get('bh_short'):
                    confidence += 10
            
            # 3. تاییدیه MACD (15 امتیاز)
            macd = indicators.get('macd', {})
            if signal_type == 'buy' and macd.get('bullish'):
                confidence += 15
            elif signal_type == 'sell' and macd.get('bearish'):
                confidence += 15
            
            # 4. وضعیت RSI (10 امتیاز)
            rsi_signals = indicators.get('rsi_signals', {})
            rsi_value = rsi_signals.get('rsi_value', 50)
            
            if signal_type == 'buy' and rsi_value < 40:
                confidence += 10
            elif signal_type == 'sell' and rsi_value > 60:
                confidence += 10
            
            # محدود کردن بین 0 تا 100
            confidence = max(0, min(100, confidence))
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
        
        return confidence
    
    def _create_buy_signal(self, symbol: str, df: pd.DataFrame, 
                          indicators: Dict, confidence: float) -> Signal:
        """ایجاد سیگنال خرید"""
        current_price = df['close'].iloc[-1]
        atr = indicators.get('atr', current_price * 0.02)
        
        # محاسبه نقاط خروج بر اساس ATR
        stop_loss = current_price - (atr * 1.5)
        take_profit_1 = current_price + (atr * 1.0)
        take_profit_2 = current_price + (atr * 2.0)
        
        return Signal(
            symbol=symbol,
            signal_type=SignalType.BUY,
            confidence=confidence,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit_1=take_profit_1,
            take_profit_2=take_profit_2,
            reason="Fast Scalp Buy - ZLMA Crossover + Smart Money Confirmation",
            timestamp=pd.Timestamp.now()
        )
    
    def _create_sell_signal(self, symbol: str, df: pd.DataFrame,
                           indicators: Dict, confidence: float) -> Signal:
        """ایجاد سیگنال فروش"""
        current_price = df['close'].iloc[-1]
        atr = indicators.get('atr', current_price * 0.02)
        
        # محاسبه نقاط خروج بر اساس ATR
        stop_loss = current_price + (atr * 1.5)
        take_profit_1 = current_price - (atr * 1.0)
        take_profit_2 = current_price - (atr * 2.0)
        
        return Signal(
            symbol=symbol,
            signal_type=SignalType.SELL,
            confidence=confidence,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit_1=take_profit_1,
            take_profit_2=take_profit_2,
            reason="Fast Scalp Sell - ZLMA Crossunder + Smart Money Confirmation",
            timestamp=pd.Timestamp.now()
        )
    
    def rank_signals(self, signals: List[Signal]) -> List[Signal]:
        """رتبه‌بندی سیگنال‌ها بر اساس کیفیت"""
        # اولویت‌بندی بر اساس اطمینان و حجم
        ranked_signals = sorted(
            signals,
            key=lambda x: x.confidence,
            reverse=True
        )
        return ranked_signals[:self.config['top_n']]  # فقط N ارز برتر
