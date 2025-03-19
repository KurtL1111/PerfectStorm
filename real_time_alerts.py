"""
Real-time Alerts Module for Perfect Storm Dashboard

This module adds functionality to:
1. Send notifications when significant signals occur
2. Alert users to developing "perfect storm" conditions
3. Provide early warnings of potential trend reversals
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
import os
import pickle
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
import time
import threading
import logging
from typing import Dict, List, Tuple, Union, Optional, Callable

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("alerts.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("real_time_alerts")

class AlertCondition:
    """Base class for alert conditions"""
    
    def __init__(self, name: str, description: str, severity: str = "medium"):
        """
        Initialize the AlertCondition
        
        Parameters:
        - name: Name of the alert condition
        - description: Description of the alert condition
        - severity: Severity level (low, medium, high)
        """
        self.name = name
        self.description = description
        self.severity = severity
        self.triggered = False
        self.last_triggered = None
        self.trigger_count = 0
        
    def check(self, data: Dict) -> bool:
        """
        Check if the alert condition is met
        
        Parameters:
        - data: Dictionary with data to check
        
        Returns:
        - triggered: Whether the condition is triggered
        """
        # Base implementation always returns False
        # Override in subclasses
        return False
    
    def reset(self):
        """Reset the alert condition"""
        self.triggered = False
        
    def trigger(self):
        """Trigger the alert condition"""
        self.triggered = True
        self.last_triggered = datetime.now()
        self.trigger_count += 1
        
    def get_message(self) -> str:
        """
        Get the alert message
        
        Returns:
        - message: Alert message
        """
        return f"Alert: {self.name} - {self.description}"
    
    def __str__(self) -> str:
        """String representation of the alert condition"""
        status = "Triggered" if self.triggered else "Not triggered"
        return f"{self.name} ({self.severity}): {status}"

class PriceAlertCondition(AlertCondition):
    """Alert condition for price movements"""
    
    def __init__(self, name: str, description: str, symbol: str, 
                 threshold: float, direction: str = "above", 
                 severity: str = "medium"):
        """
        Initialize the PriceAlertCondition
        
        Parameters:
        - name: Name of the alert condition
        - description: Description of the alert condition
        - symbol: Stock symbol
        - threshold: Price threshold
        - direction: Direction of price movement ("above" or "below")
        - severity: Severity level (low, medium, high)
        """
        super().__init__(name, description, severity)
        self.symbol = symbol
        self.threshold = threshold
        self.direction = direction
        
    def check(self, data: Dict) -> bool:
        """
        Check if the price alert condition is met
        
        Parameters:
        - data: Dictionary with price data
        
        Returns:
        - triggered: Whether the condition is triggered
        """
        if not data or 'price' not in data:
            return False
        
        price = data['price']
        
        if self.direction == "above" and price > self.threshold:
            self.trigger()
            return True
        elif self.direction == "below" and price < self.threshold:
            self.trigger()
            return True
        
        return False
    
    def get_message(self) -> str:
        """
        Get the price alert message
        
        Returns:
        - message: Price alert message
        """
        direction_text = "above" if self.direction == "above" else "below"
        return f"Price Alert: {self.symbol} is {direction_text} {self.threshold}"

class IndicatorAlertCondition(AlertCondition):
    """Alert condition for technical indicators"""
    
    def __init__(self, name: str, description: str, symbol: str, 
                 indicator: str, threshold: float, direction: str = "above", 
                 severity: str = "medium"):
        """
        Initialize the IndicatorAlertCondition
        
        Parameters:
        - name: Name of the alert condition
        - description: Description of the alert condition
        - symbol: Stock symbol
        - indicator: Technical indicator name
        - threshold: Indicator threshold
        - direction: Direction of indicator movement ("above" or "below")
        - severity: Severity level (low, medium, high)
        """
        super().__init__(name, description, severity)
        self.symbol = symbol
        self.indicator = indicator
        self.threshold = threshold
        self.direction = direction
        
    def check(self, data: Dict) -> bool:
        """
        Check if the indicator alert condition is met
        
        Parameters:
        - data: Dictionary with indicator data
        
        Returns:
        - triggered: Whether the condition is triggered
        """
        if not data or self.indicator not in data:
            return False
        
        indicator_value = data[self.indicator]
        
        if self.direction == "above" and indicator_value > self.threshold:
            self.trigger()
            return True
        elif self.direction == "below" and indicator_value < self.threshold:
            self.trigger()
            return True
        elif self.direction == "cross_above" and indicator_value > self.threshold and not self.triggered:
            self.trigger()
            return True
        elif self.direction == "cross_below" and indicator_value < self.threshold and not self.triggered:
            self.trigger()
            return True
        
        return False
    
    def get_message(self) -> str:
        """
        Get the indicator alert message
        
        Returns:
        - message: Indicator alert message
        """
        direction_text = "above" if self.direction == "above" else "below"
        if "cross" in self.direction:
            direction_text = "crossed " + direction_text
            
        return f"Indicator Alert: {self.indicator} for {self.symbol} is {direction_text} {self.threshold}"

class PatternAlertCondition(AlertCondition):
    """Alert condition for chart patterns"""
    
    def __init__(self, name: str, description: str, symbol: str, 
                 pattern: str, confidence: float = 0.7, 
                 severity: str = "medium"):
        """
        Initialize the PatternAlertCondition
        
        Parameters:
        - name: Name of the alert condition
        - description: Description of the alert condition
        - symbol: Stock symbol
        - pattern: Chart pattern name
        - confidence: Confidence threshold (0.0 to 1.0)
        - severity: Severity level (low, medium, high)
        """
        super().__init__(name, description, severity)
        self.symbol = symbol
        self.pattern = pattern
        self.confidence = confidence
        
    def check(self, data: Dict) -> bool:
        """
        Check if the pattern alert condition is met
        
        Parameters:
        - data: Dictionary with pattern data
        
        Returns:
        - triggered: Whether the condition is triggered
        """
        if not data or 'patterns' not in data:
            return False
        
        patterns = data['patterns']
        
        if self.pattern in patterns and patterns[self.pattern]['confidence'] >= self.confidence:
            self.trigger()
            return True
        
        return False
    
    def get_message(self) -> str:
        """
        Get the pattern alert message
        
        Returns:
        - message: Pattern alert message
        """
        return f"Pattern Alert: {self.pattern} detected for {self.symbol} with confidence >= {self.confidence}"

class PerfectStormAlertCondition(AlertCondition):
    """Alert condition for perfect storm configurations"""
    
    def __init__(self, name: str, description: str, symbol: str, 
                 indicators: List[str], thresholds: List[float], 
                 directions: List[str], min_triggers: int = None,
                 severity: str = "high"):
        """
        Initialize the PerfectStormAlertCondition
        
        Parameters:
        - name: Name of the alert condition
        - description: Description of the alert condition
        - symbol: Stock symbol
        - indicators: List of technical indicator names
        - thresholds: List of indicator thresholds
        - directions: List of indicator directions ("above" or "below")
        - min_triggers: Minimum number of indicators to trigger (default: all)
        - severity: Severity level (low, medium, high)
        """
        super().__init__(name, description, severity)
        self.symbol = symbol
        self.indicators = indicators
        self.thresholds = thresholds
        self.directions = directions
        self.min_triggers = min_triggers if min_triggers is not None else len(indicators)
        self.triggered_indicators = []
        
    def check(self, data: Dict) -> bool:
        """
        Check if the perfect storm alert condition is met
        
        Parameters:
        - data: Dictionary with indicator data
        
        Returns:
        - triggered: Whether the condition is triggered
        """
        if not data:
            return False
        
        # Reset triggered indicators
        self.triggered_indicators = []
        
        # Check each indicator
        for i, indicator in enumerate(self.indicators):
            if indicator not in data:
                continue
                
            indicator_value = data[indicator]
            threshold = self.thresholds[i]
            direction = self.directions[i]
            
            if direction == "above" and indicator_value > threshold:
                self.triggered_indicators.append(indicator)
            elif direction == "below" and indicator_value < threshold:
                self.triggered_indicators.append(indicator)
        
        # Check if enough indicators are triggered
        if len(self.triggered_indicators) >= self.min_triggers:
            self.trigger()
            return True
        
        return False
    
    def get_message(self) -> str:
        """
        Get the perfect storm alert message
        
        Returns:
        - message: Perfect storm alert message
        """
        triggered_text = ", ".join(self.triggered_indicators)
        return f"Perfect Storm Alert: {len(self.triggered_indicators)}/{len(self.indicators)} indicators triggered for {self.symbol}: {triggered_text}"

class TrendReversalAlertCondition(AlertCondition):
    """Alert condition for trend reversals"""
    
    def __init__(self, name: str, description: str, symbol: str, 
                 lookback_period: int = 14, threshold: float = 0.1, 
                 severity: str = "high"):
        """
        Initialize the TrendReversalAlertCondition
        
        Parameters:
        - name: Name of the alert condition
        - description: Description of the alert condition
        - symbol: Stock symbol
        - lookback_period: Number of periods to look back
        - threshold: Reversal threshold
        - severity: Severity level (low, medium, high)
        """
        super().__init__(name, description, severity)
        self.symbol = symbol
        self.lookback_period = lookback_period
        self.threshold = threshold
        self.prev_trend = None
        
    def check(self, data: Dict) -> bool:
        """
        Check if the trend reversal alert condition is met
        
        Parameters:
        - data: Dictionary with price and trend data
        
        Returns:
        - triggered: Whether the condition is triggered
        """
        if not data or 'trend' not in data or 'prices' not in data:
            return False
        
        current_trend = data['trend']
        prices = data['prices']
        
        # Check if we have enough price data
        if len(prices) < self.lookback_period:
            return False
        
        # Check if trend has reversed
        if self.prev_trend is not None and current_trend != self.prev_trend:
            # Calculate price change percentage
            price_change = abs(prices[-1] - prices[-self.lookback_period]) / prices[-self.lookback_period]
            
            if price_change >= self.threshold:
                self.trigger()
                return True
        
        # Update previous trend
        self.prev_trend = current_trend
        
        return False
    
    def get_message(self) -> str:
        """
        Get the trend reversal alert message
        
        Returns:
        - message: Trend reversal alert message
        """
        return f"Trend Reversal Alert: {self.symbol} has reversed trend from {self.prev_trend}"

class AnomalyAlertCondition(AlertCondition):
    """Alert condition for market anomalies"""
    
    def __init__(self, name: str, description: str, symbol: str, 
                 anomaly_threshold: float = 3.0, 
                 severity: str = "high"):
        """
        Initialize the AnomalyAlertCondition
        
        Parameters:
        - name: Name of the alert condition
        - description: Description of the alert condition
        - symbol: Stock symbol
        - anomaly_threshold: Z-score threshold for anomaly detection
        - severity: Severity level (low, medium, high)
        """
        super().__init__(name, description, severity)
        self.symbol = symbol
        self.anomaly_threshold = anomaly_threshold
        
    def check(self, data: Dict) -> bool:
        """
        Check if the anomaly alert condition is met
        
        Parameters:
        - data: Dictionary with anomaly data
        
        Returns:
        - triggered: Whether the condition is triggered
        """
        if not data or 'anomaly_score' not in data:
            return False
        
        anomaly_score = data['anomaly_score']
        
        if abs(anomaly_score) >= self.anomaly_threshold:
            self.trigger()
            return True
        
        return False
    
    def get_message(self) -> str:
        """
        Get the anomaly alert message
        
        Returns:
        - message: Anomaly alert message
        """
        return f"Anomaly Alert: Unusual market behavior detected for {self.symbol} (score >= {self.anomaly_threshold})"

class AlertNotifier:
    """Base class for alert notifiers"""
    
    def __init__(self, name: str):
        """
        Initialize the AlertNotifier
        
        Parameters:
        - name: Name of the notifier
        """
        self.name = name
        
    def send(self, message: str, subject: str = None, **kwargs) -> bool:
        """
        Send an alert notification
        
        Parameters:
        - message: Alert message
        - subject: Alert subject (optional)
        - **kwargs: Additional parameters
        
        Returns:
        - success: Whether the notification was sent successfully
        """
        # Base implementation does nothing
        # Override in subclasses
        return False
    
    def __str__(self) -> str:
        """String representation of the alert notifier"""
        return f"Aler<response clipped><NOTE>To save on context only part of this file has been shown to you. You should retry this tool after you have searched inside the file with `grep -n` in order to find the line numbers of what you are looking for.</NOTE>