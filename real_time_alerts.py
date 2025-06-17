import plotly.graph_objects as go
import plotly.tools as tls
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
        
        # If min_triggers not specified, require all indicators
        if min_triggers is None:
            self.min_triggers = len(indicators)
        else:
            self.min_triggers = min_triggers
            
        # Create individual indicator conditions
        self.indicator_conditions = []
        for i in range(len(indicators)):
            condition = IndicatorAlertCondition(
                f"{indicators[i]} Condition",
                f"{indicators[i]} {directions[i]} {thresholds[i]}",
                symbol,
                indicators[i],
                thresholds[i],
                directions[i]
            )
            self.indicator_conditions.append(condition)
        
        # Track triggered indicators
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
        
        # Check each indicator condition
        for condition in self.indicator_conditions:
            if condition.check(data):
                self.triggered_indicators.append(condition.indicator)
        
        # Check if minimum number of indicators triggered
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
        return f"Perfect Storm Alert: {self.symbol} has triggered {len(self.triggered_indicators)}/{len(self.indicators)} indicators: {triggered_text}"

class TrendReversalAlertCondition(AlertCondition):
    """Alert condition for trend reversals"""
    
    def __init__(self, name: str, description: str, symbol: str, 
                 lookback_period: int = 20, threshold: float = 0.1, 
                 severity: str = "high"):
        """
        Initialize the TrendReversalAlertCondition
        
        Parameters:
        - name: Name of the alert condition
        - description: Description of the alert condition
        - symbol: Stock symbol
        - lookback_period: Number of periods to look back (default: 20)
        - threshold: Reversal threshold (default: 0.1 or 10%)
        - severity: Severity level (low, medium, high)
        """
        super().__init__(name, description, severity)
        self.symbol = symbol
        self.lookback_period = lookback_period
        self.threshold = threshold
        
        # Store price history
        self.price_history = []
        
    def check(self, data: Dict) -> bool:
        """
        Check if the trend reversal alert condition is met
        
        Parameters:
        - data: Dictionary with price data
        
        Returns:
        - triggered: Whether the condition is triggered
        """
        if not data or 'price' not in data:
            return False
        
        # Add current price to history
        self.price_history.append(data['price'])
        
        # Keep only the lookback period
        if len(self.price_history) > self.lookback_period:
            self.price_history = self.price_history[-self.lookback_period:]
        
        # Need at least lookback_period prices
        if len(self.price_history) < self.lookback_period:
            return False
        
        # Calculate trend
        first_half = self.price_history[:self.lookback_period//2]
        second_half = self.price_history[self.lookback_period//2:]
        
        first_half_avg = sum(first_half) / len(first_half)
        second_half_avg = sum(second_half) / len(second_half)
        
        # Calculate percentage change
        pct_change = (second_half_avg - first_half_avg) / first_half_avg
        
        # Check for reversal
        if abs(pct_change) >= self.threshold:
            self.trigger()
            return True
        
        return False
    
    def get_message(self) -> str:
        """
        Get the trend reversal alert message
        
        Returns:
        - message: Trend reversal alert message
        """
        return f"Trend Reversal Alert: {self.symbol} has shown a potential trend reversal over the last {self.lookback_period} periods"

class VolatilityAlertCondition(AlertCondition):
    """Alert condition for volatility spikes"""
    
    def __init__(self, name: str, description: str, symbol: str, 
                 lookback_period: int = 20, threshold: float = 2.0, 
                 severity: str = "medium"):
        """
        Initialize the VolatilityAlertCondition
        
        Parameters:
        - name: Name of the alert condition
        - description: Description of the alert condition
        - symbol: Stock symbol
        - lookback_period: Number of periods to look back (default: 20)
        - threshold: Volatility threshold as multiple of average (default: 2.0)
        - severity: Severity level (low, medium, high)
        """
        super().__init__(name, description, severity)
        self.symbol = symbol
        self.lookback_period = lookback_period
        self.threshold = threshold
        
        # Store price history
        self.price_history = []
        
    def check(self, data: Dict) -> bool:
        """
        Check if the volatility alert condition is met
        
        Parameters:
        - data: Dictionary with price data
        
        Returns:
        - triggered: Whether the condition is triggered
        """
        if not data or 'price' not in data:
            return False
        
        # Add current price to history
        self.price_history.append(data['price'])
        
        # Keep only the lookback period
        if len(self.price_history) > self.lookback_period:
            self.price_history = self.price_history[-self.lookback_period:]
        
        # Need at least lookback_period prices
        if len(self.price_history) < self.lookback_period:
            return False
        
        # Calculate returns
        returns = [self.price_history[i] / self.price_history[i-1] - 1 for i in range(1, len(self.price_history))]
        
        # Calculate historical volatility
        historical_volatility = np.std(returns) * np.sqrt(252)  # Annualized
        
        # Calculate current volatility (last 5 periods)
        current_returns = returns[-5:]
        current_volatility = np.std(current_returns) * np.sqrt(252)  # Annualized
        
        # Check for volatility spike
        if current_volatility >= self.threshold * historical_volatility:
            self.trigger()
            return True
        
        return False
    
    def get_message(self) -> str:
        """
        Get the volatility alert message
        
        Returns:
        - message: Volatility alert message
        """
        return f"Volatility Alert: {self.symbol} has experienced a significant increase in volatility"

class VolumeAlertCondition(AlertCondition):
    """Alert condition for volume spikes"""
    
    def __init__(self, name: str, description: str, symbol: str, 
                 lookback_period: int = 20, threshold: float = 2.0, 
                 severity: str = "medium"):
        """
        Initialize the VolumeAlertCondition
        
        Parameters:
        - name: Name of the alert condition
        - description: Description of the alert condition
        - symbol: Stock symbol
        - lookback_period: Number of periods to look back (default: 20)
        - threshold: Volume threshold as multiple of average (default: 2.0)
        - severity: Severity level (low, medium, high)
        """
        super().__init__(name, description, severity)
        self.symbol = symbol
        self.lookback_period = lookback_period
        self.threshold = threshold
        
        # Store volume history
        self.volume_history = []
        
    def check(self, data: Dict) -> bool:
        """
        Check if the volume alert condition is met
        
        Parameters:
        - data: Dictionary with volume data
        
        Returns:
        - triggered: Whether the condition is triggered
        """
        if not data or 'volume' not in data:
            return False
        
        # Add current volume to history
        self.volume_history.append(data['volume'])
        
        # Keep only the lookback period
        if len(self.volume_history) > self.lookback_period:
            self.volume_history = self.volume_history[-self.lookback_period:]
        
        # Need at least lookback_period volumes
        if len(self.volume_history) < self.lookback_period:
            return False
        
        # Calculate average volume (excluding current)
        avg_volume = sum(self.volume_history[:-1]) / (len(self.volume_history) - 1)
        
        # Check for volume spike
        if self.volume_history[-1] >= self.threshold * avg_volume:
            self.trigger()
            return True
        
        return False
    
    def get_message(self) -> str:
        """
        Get the volume alert message
        
        Returns:
        - message: Volume alert message
        """
        return f"Volume Alert: {self.symbol} has experienced a significant increase in trading volume"

class NotificationChannel:
    """Base class for notification channels"""
    
    def __init__(self, name: str):
        """
        Initialize the NotificationChannel
        
        Parameters:
        - name: Name of the notification channel
        """
        self.name = name
        
    def send(self, message: str, subject: str = None, **kwargs) -> bool:
        """
        Send a notification
        
        Parameters:
        - message: Notification message
        - subject: Notification subject (default: None)
        - **kwargs: Additional parameters
        
        Returns:
        - success: Whether the notification was sent successfully
        """
        # Base implementation always returns False
        # Override in subclasses
        return False
    
    def __str__(self) -> str:
        """String representation of the notification channel"""
        return f"NotificationChannel: {self.name}"

class EmailNotificationChannel(NotificationChannel):
    """Email notification channel"""
    
    def __init__(self, name: str, smtp_server: str, smtp_port: int, 
                 username: str, password: str, sender: str, recipients: List[str]):
        """
        Initialize the EmailNotificationChannel
        
        Parameters:
        - name: Name of the notification channel
        - smtp_server: SMTP server address
        - smtp_port: SMTP server port
        - username: SMTP username
        - password: SMTP password
        - sender: Sender email address
        - recipients: List of recipient email addresses
        """
        super().__init__(name)
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.sender = sender
        self.recipients = recipients
        
    def send(self, message: str, subject: str = None, **kwargs) -> bool:
        """
        Send an email notification
        
        Parameters:
        - message: Notification message
        - subject: Notification subject (default: None)
        - **kwargs: Additional parameters
        
        Returns:
        - success: Whether the notification was sent successfully
        """
        if subject is None:
            subject = f"Alert Notification from Perfect Storm Dashboard"
        
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.sender
            msg['To'] = ", ".join(self.recipients)
            msg['Subject'] = subject
            
            # Add message body
            msg.attach(MIMEText(message, 'plain'))
            
            # Connect to SMTP server
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.username, self.password)
            
            # Send email
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Email notification sent to {', '.join(self.recipients)}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email notification: {str(e)}")
            return False

class WebhookNotificationChannel(NotificationChannel):
    """Webhook notification channel"""
    
    def __init__(self, name: str, webhook_url: str, headers: Dict = None):
        """
        Initialize the WebhookNotificationChannel
        
        Parameters:
        - name: Name of the notification channel
        - webhook_url: Webhook URL
        - headers: HTTP headers (default: None)
        """
        super().__init__(name)
        self.webhook_url = webhook_url
        self.headers = headers or {}
        
    def send(self, message: str, subject: str = None, **kwargs) -> bool:
        """
        Send a webhook notification
        
        Parameters:
        - message: Notification message
        - subject: Notification subject (default: None)
        - **kwargs: Additional parameters
        
        Returns:
        - success: Whether the notification was sent successfully
        """
        if subject is None:
            subject = f"Alert Notification from Perfect Storm Dashboard"
        
        try:
            # Create payload
            payload = {
                'subject': subject,
                'message': message,
                'timestamp': datetime.now().isoformat(),
                **kwargs
            }
            
            # Send webhook request
            response = requests.post(
                self.webhook_url,
                json=payload,
                headers=self.headers
            )
            
            # Check response
            if response.status_code >= 200 and response.status_code < 300:
                logger.info(f"Webhook notification sent to {self.webhook_url}")
                return True
            else:
                logger.error(f"Failed to send webhook notification: {response.status_code} {response.text}")
                return False
            
        except Exception as e:
            logger.error(f"Failed to send webhook notification: {str(e)}")
            return False

class SMSNotificationChannel(NotificationChannel):
    """SMS notification channel"""
    
    def __init__(self, name: str, api_key: str, api_secret: str, 
                 sender: str, recipients: List[str], provider: str = "twilio"):
        """
        Initialize the SMSNotificationChannel
        
        Parameters:
        - name: Name of the notification channel
        - api_key: API key for SMS provider
        - api_secret: API secret for SMS provider
        - sender: Sender phone number
        - recipients: List of recipient phone numbers
        - provider: SMS provider (default: "twilio")
        """
        super().__init__(name)
        self.api_key = api_key
        self.api_secret = api_secret
        self.sender = sender
        self.recipients = recipients
        self.provider = provider
        
    def send(self, message: str, subject: str = None, **kwargs) -> bool:
        """
        Send an SMS notification
        
        Parameters:
        - message: Notification message
        - subject: Notification subject (default: None)
        - **kwargs: Additional parameters
        
        Returns:
        - success: Whether the notification was sent successfully
        """
        try:
            if self.provider == "twilio":
                return self._send_twilio(message, **kwargs)
            else:
                logger.error(f"Unsupported SMS provider: {self.provider}")
                return False
            
        except Exception as e:
            logger.error(f"Failed to send SMS notification: {str(e)}")
            return False
    
    def _send_twilio(self, message: str, **kwargs) -> bool:
        """
        Send an SMS notification using Twilio
        
        Parameters:
        - message: Notification message
        - **kwargs: Additional parameters
        
        Returns:
        - success: Whether the notification was sent successfully
        """
        try:
            # Import Twilio client
            from twilio.rest import Client
            
            # Create client
            client = Client(self.api_key, self.api_secret)
            
            # Send SMS to each recipient
            success = True
            for recipient in self.recipients:
                try:
                    # Send message
                    sms = client.messages.create(
                        body=message,
                        from_=self.sender,
                        to=recipient
                    )
                    logger.info(f"SMS notification sent to {recipient}")
                except Exception as e:
                    logger.error(f"Failed to send SMS to {recipient}: {str(e)}")
                    success = False
            
            return success
            
        except ImportError:
            logger.error("Twilio package not installed. Install with: pip install twilio")
            return False
        except Exception as e:
            logger.error(f"Failed to send Twilio SMS: {str(e)}")
            return False

class PushNotificationChannel(NotificationChannel):
    """Push notification channel"""
    
    def __init__(self, name: str, api_key: str, app_id: str, 
                 provider: str = "firebase"):
        """
        Initialize the PushNotificationChannel
        
        Parameters:
        - name: Name of the notification channel
        - api_key: API key for push notification provider
        - app_id: Application ID
        - provider: Push notification provider (default: "firebase")
        """
        super().__init__(name)
        self.api_key = api_key
        self.app_id = app_id
        self.provider = provider
        
    def send(self, message: str, subject: str = None, topic: str = None, 
             tokens: List[str] = None, **kwargs) -> bool:
        """
        Send a push notification
        
        Parameters:
        - message: Notification message
        - subject: Notification subject (default: None)
        - topic: Notification topic (default: None)
        - tokens: List of device tokens (default: None)
        - **kwargs: Additional parameters
        
        Returns:
        - success: Whether the notification was sent successfully
        """
        if subject is None:
            subject = f"Alert Notification"
        
        try:
            if self.provider == "firebase":
                return self._send_firebase(message, subject, topic, tokens, **kwargs)
            else:
                logger.error(f"Unsupported push notification provider: {self.provider}")
                return False
            
        except Exception as e:
            logger.error(f"Failed to send push notification: {str(e)}")
            return False
    
    def _send_firebase(self, message: str, subject: str, topic: str = None, 
                      tokens: List[str] = None, **kwargs) -> bool:
        """
        Send a push notification using Firebase Cloud Messaging
        
        Parameters:
        - message: Notification message
        - subject: Notification subject
        - topic: Notification topic (default: None)
        - tokens: List of device tokens (default: None)
        - **kwargs: Additional parameters
        
        Returns:
        - success: Whether the notification was sent successfully
        """
        try:
            # Import Firebase Admin SDK
            import firebase_admin
            from firebase_admin import credentials, messaging
            
            # Initialize Firebase app if not already initialized
            if not firebase_admin._apps:
                cred = credentials.Certificate(self.api_key)
                firebase_admin.initialize_app(cred)
            
            # Create message
            notification = messaging.Notification(
                title=subject,
                body=message
            )
            
            # Send to topic
            if topic:
                message = messaging.Message(
                    notification=notification,
                    topic=topic,
                    data=kwargs
                )
                response = messaging.send(message)
                logger.info(f"Push notification sent to topic {topic}")
                return True
            
            # Send to tokens
            elif tokens:
                message = messaging.MulticastMessage(
                    notification=notification,
                    tokens=tokens,
                    data=kwargs
                )
                response = messaging.send_multicast(message)
                logger.info(f"Push notification sent to {len(tokens)} devices")
                return True
            
            else:
                logger.error("No topic or tokens provided for push notification")
                return False
            
        except ImportError:
            logger.error("Firebase Admin SDK not installed. Install with: pip install firebase-admin")
            return False
        except Exception as e:
            logger.error(f"Failed to send Firebase push notification: {str(e)}")
            return False

class AlertManager:
    def convert_mpl_to_plotly(self, fig_mpl):
        """
        Convert a Matplotlib figure to a Plotly figure for Dash display.
        """
        try:
            fig_plotly = tls.mpl_to_plotly(fig_mpl)
            plt.close(fig_mpl)
            return fig_plotly
        except Exception as e:
            logger.error(f"Failed to convert Matplotlib figure to Plotly: {str(e)}")
            plt.close(fig_mpl)
            return go.Figure()
    """Class for managing alerts and notifications"""
    
    def __init__(self, config_file: str = None):
        """
        Initialize the AlertManager
        
        Parameters:
        - config_file: Path to configuration file (default: None)
        """
        # Initialize alert conditions
        self.alert_conditions = []
        
        # Initialize notification channels
        self.notification_channels = []
        
        # Initialize alert history
        self.alert_history = []
        
        # Initialize alert handlers
        self.alert_handlers = {}
        
        # Load configuration if provided
        if config_file:
            self.load_config(config_file)
        
        # Initialize monitoring thread
        self.monitoring_thread = None
        self.monitoring_interval = 60  # Default: 60 seconds
        self.monitoring_active = False
        
    def add_alert_condition(self, condition: AlertCondition):
        """
        Add an alert condition
        
        Parameters:
        - condition: Alert condition to add
        """
        self.alert_conditions.append(condition)
        logger.info(f"Added alert condition: {condition.name}")
        
    def add_notification_channel(self, channel: NotificationChannel):
        """
        Add a notification channel
        
        Parameters:
        - channel: Notification channel to add
        """
        self.notification_channels.append(channel)
        logger.info(f"Added notification channel: {channel.name}")
        
    def register_alert_handler(self, alert_type: str, handler: Callable):
        """
        Register a custom alert handler
        
        Parameters:
        - alert_type: Type of alert to handle
        - handler: Handler function
        """
        self.alert_handlers[alert_type] = handler
        logger.info(f"Registered alert handler for {alert_type}")
        
    def check_conditions(self, data: Dict) -> List[AlertCondition]:
        """
        Check all alert conditions
        
        Parameters:
        - data: Dictionary with data to check
        
        Returns:
        - triggered_conditions: List of triggered alert conditions
        """
        triggered_conditions = []
        
        for condition in self.alert_conditions:
            try:
                if condition.check(data):
                    triggered_conditions.append(condition)
                    logger.info(f"Alert condition triggered: {condition.name}")
                    
                    # Add to alert history
                    self.alert_history.append({
                        'timestamp': datetime.now(),
                        'condition': condition.name,
                        'message': condition.get_message(),
                        'severity': condition.severity,
                        'data': data
                    })
                    
                    # Call custom handler if registered
                    alert_type = condition.__class__.__name__
                    if alert_type in self.alert_handlers:
                        try:
                            self.alert_handlers[alert_type](condition, data)
                        except Exception as e:
                            logger.error(f"Error in alert handler for {alert_type}: {str(e)}")
            
            except Exception as e:
                logger.error(f"Error checking alert condition {condition.name}: {str(e)}")
        
        return triggered_conditions
    
    def send_notifications(self, conditions: List[AlertCondition]) -> bool:
        """
        Send notifications for triggered alert conditions
        
        Parameters:
        - conditions: List of triggered alert conditions
        
        Returns:
        - success: Whether all notifications were sent successfully
        """
        if not conditions:
            return True
        
        # Group conditions by severity
        severity_groups = {}
        for condition in conditions:
            if condition.severity not in severity_groups:
                severity_groups[condition.severity] = []
            severity_groups[condition.severity].append(condition)
        
        # Send notifications for each severity group
        success = True
        for severity, conditions_group in severity_groups.items():
            # Create message
            subject = f"{severity.capitalize()} Alert from Perfect Storm Dashboard"
            message = f"The following alerts have been triggered:\n\n"
            
            for condition in conditions_group:
                message += f"- {condition.get_message()}\n"
            
            message += f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            
            # Send to all notification channels
            for channel in self.notification_channels:
                try:
                    channel_success = channel.send(message, subject, 
                                                 severity=severity, 
                                                 conditions=[c.name for c in conditions_group])
                    if not channel_success:
                        success = False
                except Exception as e:
                    logger.error(f"Error sending notification via {channel.name}: {str(e)}")
                    success = False
        
        return success
    
    def process_data(self, data: Dict) -> bool:
        """
        Process data, check conditions, and send notifications
        
        Parameters:
        - data: Dictionary with data to process
        
        Returns:
        - success: Whether processing was successful
        """
        try:
            # Check conditions
            triggered_conditions = self.check_conditions(data)
            
            # Send notifications if any conditions triggered
            if triggered_conditions:
                return self.send_notifications(triggered_conditions)
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing data: {str(e)}")
            return False
    
    def start_monitoring(self, data_provider: Callable, interval: int = None):
        """
        Start monitoring thread
        
        Parameters:
        - data_provider: Function that returns data to process
        - interval: Monitoring interval in seconds (default: None, use self.monitoring_interval)
        """
        if interval:
            self.monitoring_interval = interval
        
        # Stop existing thread if running
        self.stop_monitoring()
        
        # Set monitoring active flag
        self.monitoring_active = True
        
        # Create and start monitoring thread
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(data_provider,),
            daemon=True
        )
        self.monitoring_thread.start()
        
        logger.info(f"Started monitoring thread with interval {self.monitoring_interval} seconds")
    
    def _monitoring_loop(self, data_provider: Callable):
        """
        Monitoring thread loop
        
        Parameters:
        - data_provider: Function that returns data to process
        """
        while self.monitoring_active:
            try:
                # Get data from provider
                data = data_provider()
                
                # Process data
                self.process_data(data)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
            
            # Sleep for interval
            time.sleep(self.monitoring_interval)
    
    def stop_monitoring(self):
        """Stop monitoring thread"""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_active = False
            self.monitoring_thread.join(timeout=5)
            logger.info("Stopped monitoring thread")
    
    def reset_conditions(self):
        """Reset all alert conditions"""
        for condition in self.alert_conditions:
            condition.reset()
        logger.info("Reset all alert conditions")
    
    def get_alert_history(self, start_time: datetime = None, 
                         end_time: datetime = None, 
                         severity: str = None) -> List[Dict]:
        """
        Get alert history
        
        Parameters:
        - start_time: Start time filter (default: None)
        - end_time: End time filter (default: None)
        - severity: Severity filter (default: None)
        
        Returns:
        - filtered_history: Filtered alert history
        """
        filtered_history = self.alert_history
        
        # Apply start time filter
        if start_time:
            filtered_history = [alert for alert in filtered_history 
                               if alert['timestamp'] >= start_time]
        
        # Apply end time filter
        if end_time:
            filtered_history = [alert for alert in filtered_history 
                               if alert['timestamp'] <= end_time]
        
        # Apply severity filter
        if severity:
            filtered_history = [alert for alert in filtered_history 
                               if alert['severity'] == severity]
        
        return filtered_history
    
    def save_config(self, config_file: str):
        """
        Save configuration to file
        
        Parameters:
        - config_file: Path to configuration file
        """
        # Create configuration dictionary
        config = {
            'monitoring_interval': self.monitoring_interval,
            'alert_conditions': [],
            'notification_channels': []
        }
        
        # Add alert conditions
        for condition in self.alert_conditions:
            condition_config = {
                'type': condition.__class__.__name__,
                'name': condition.name,
                'description': condition.description,
                'severity': condition.severity
            }
            
            # Add condition-specific parameters
            if isinstance(condition, PriceAlertCondition):
                condition_config.update({
                    'symbol': condition.symbol,
                    'threshold': condition.threshold,
                    'direction': condition.direction
                })
            elif isinstance(condition, IndicatorAlertCondition):
                condition_config.update({
                    'symbol': condition.symbol,
                    'indicator': condition.indicator,
                    'threshold': condition.threshold,
                    'direction': condition.direction
                })
            elif isinstance(condition, PatternAlertCondition):
                condition_config.update({
                    'symbol': condition.symbol,
                    'pattern': condition.pattern,
                    'confidence': condition.confidence
                })
            elif isinstance(condition, PerfectStormAlertCondition):
                condition_config.update({
                    'symbol': condition.symbol,
                    'indicators': condition.indicators,
                    'thresholds': condition.thresholds,
                    'directions': condition.directions,
                    'min_triggers': condition.min_triggers
                })
            elif isinstance(condition, TrendReversalAlertCondition):
                condition_config.update({
                    'symbol': condition.symbol,
                    'lookback_period': condition.lookback_period,
                    'threshold': condition.threshold
                })
            elif isinstance(condition, VolatilityAlertCondition):
                condition_config.update({
                    'symbol': condition.symbol,
                    'lookback_period': condition.lookback_period,
                    'threshold': condition.threshold
                })
            elif isinstance(condition, VolumeAlertCondition):
                condition_config.update({
                    'symbol': condition.symbol,
                    'lookback_period': condition.lookback_period,
                    'threshold': condition.threshold
                })
            
            config['alert_conditions'].append(condition_config)
        
        # Add notification channels
        for channel in self.notification_channels:
            channel_config = {
                'type': channel.__class__.__name__,
                'name': channel.name
            }
            
            # Add channel-specific parameters
            if isinstance(channel, EmailNotificationChannel):
                channel_config.update({
                    'smtp_server': channel.smtp_server,
                    'smtp_port': channel.smtp_port,
                    'username': channel.username,
                    'password': '********',  # Don't save actual password
                    'sender': channel.sender,
                    'recipients': channel.recipients
                })
            elif isinstance(channel, WebhookNotificationChannel):
                channel_config.update({
                    'webhook_url': channel.webhook_url,
                    'headers': channel.headers
                })
            elif isinstance(channel, SMSNotificationChannel):
                channel_config.update({
                    'api_key': '********',  # Don't save actual API key
                    'api_secret': '********',  # Don't save actual API secret
                    'sender': channel.sender,
                    'recipients': channel.recipients,
                    'provider': channel.provider
                })
            elif isinstance(channel, PushNotificationChannel):
                channel_config.update({
                    'api_key': '********',  # Don't save actual API key
                    'app_id': channel.app_id,
                    'provider': channel.provider
                })
            
            config['notification_channels'].append(channel_config)
        
        # Save to file
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=4)
        
        logger.info(f"Saved configuration to {config_file}")
    
    def load_config(self, config_file: str):
        """
        Load configuration from file
        
        Parameters:
        - config_file: Path to configuration file
        """
        try:
            # Load from file
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            # Set monitoring interval
            if 'monitoring_interval' in config:
                self.monitoring_interval = config['monitoring_interval']
            
            # Clear existing alert conditions and notification channels
            self.alert_conditions = []
            self.notification_channels = []
            
            # Add alert conditions
            for condition_config in config.get('alert_conditions', []):
                condition_type = condition_config.get('type')
                
                if condition_type == 'PriceAlertCondition':
                    condition = PriceAlertCondition(
                        name=condition_config.get('name'),
                        description=condition_config.get('description'),
                        symbol=condition_config.get('symbol'),
                        threshold=condition_config.get('threshold'),
                        direction=condition_config.get('direction'),
                        severity=condition_config.get('severity')
                    )
                    self.add_alert_condition(condition)
                
                elif condition_type == 'IndicatorAlertCondition':
                    condition = IndicatorAlertCondition(
                        name=condition_config.get('name'),
                        description=condition_config.get('description'),
                        symbol=condition_config.get('symbol'),
                        indicator=condition_config.get('indicator'),
                        threshold=condition_config.get('threshold'),
                        direction=condition_config.get('direction'),
                        severity=condition_config.get('severity')
                    )
                    self.add_alert_condition(condition)
                
                elif condition_type == 'PatternAlertCondition':
                    condition = PatternAlertCondition(
                        name=condition_config.get('name'),
                        description=condition_config.get('description'),
                        symbol=condition_config.get('symbol'),
                        pattern=condition_config.get('pattern'),
                        confidence=condition_config.get('confidence'),
                        severity=condition_config.get('severity')
                    )
                    self.add_alert_condition(condition)
                
                elif condition_type == 'PerfectStormAlertCondition':
                    condition = PerfectStormAlertCondition(
                        name=condition_config.get('name'),
                        description=condition_config.get('description'),
                        symbol=condition_config.get('symbol'),
                        indicators=condition_config.get('indicators'),
                        thresholds=condition_config.get('thresholds'),
                        directions=condition_config.get('directions'),
                        min_triggers=condition_config.get('min_triggers'),
                        severity=condition_config.get('severity')
                    )
                    self.add_alert_condition(condition)
                
                elif condition_type == 'TrendReversalAlertCondition':
                    condition = TrendReversalAlertCondition(
                        name=condition_config.get('name'),
                        description=condition_config.get('description'),
                        symbol=condition_config.get('symbol'),
                        lookback_period=condition_config.get('lookback_period'),
                        threshold=condition_config.get('threshold'),
                        severity=condition_config.get('severity')
                    )
                    self.add_alert_condition(condition)
                
                elif condition_type == 'VolatilityAlertCondition':
                    condition = VolatilityAlertCondition(
                        name=condition_config.get('name'),
                        description=condition_config.get('description'),
                        symbol=condition_config.get('symbol'),
                        lookback_period=condition_config.get('lookback_period'),
                        threshold=condition_config.get('threshold'),
                        severity=condition_config.get('severity')
                    )
                    self.add_alert_condition(condition)
                
                elif condition_type == 'VolumeAlertCondition':
                    condition = VolumeAlertCondition(
                        name=condition_config.get('name'),
                        description=condition_config.get('description'),
                        symbol=condition_config.get('symbol'),
                        lookback_period=condition_config.get('lookback_period'),
                        threshold=condition_config.get('threshold'),
                        severity=condition_config.get('severity')
                    )
                    self.add_alert_condition(condition)
            
            # Note: Notification channels typically require credentials that aren't saved in the config
            # You would need to provide these separately or prompt the user for them
            
            logger.info(f"Loaded configuration from {config_file}")
            
        except Exception as e:
            logger.error(f"Error loading configuration from {config_file}: {str(e)}")
    
    def create_perfect_storm_alert(self, symbol: str, indicators: List[str], 
                                  thresholds: List[float], directions: List[str], 
                                  min_triggers: int = None, severity: str = "high"):
        """
        Create a perfect storm alert condition
        
        Parameters:
        - symbol: Stock symbol
        - indicators: List of technical indicator names
        - thresholds: List of indicator thresholds
        - directions: List of indicator directions ("above" or "below")
        - min_triggers: Minimum number of indicators to trigger (default: None, all)
        - severity: Severity level (default: "high")
        
        Returns:
        - condition: Created alert condition
        """
        # Create name and description
        name = f"Perfect Storm Alert for {symbol}"
        description = f"Perfect Storm configuration with {len(indicators)} indicators"
        
        # Create condition
        condition = PerfectStormAlertCondition(
            name=name,
            description=description,
            symbol=symbol,
            indicators=indicators,
            thresholds=thresholds,
            directions=directions,
            min_triggers=min_triggers,
            severity=severity
        )
        
        # Add to alert conditions
        self.add_alert_condition(condition)
        
        return condition
    
    def create_trend_reversal_alert(self, symbol: str, lookback_period: int = 20, 
                                   threshold: float = 0.1, severity: str = "high"):
        """
        Create a trend reversal alert condition
        
        Parameters:
        - symbol: Stock symbol
        - lookback_period: Number of periods to look back (default: 20)
        - threshold: Reversal threshold (default: 0.1 or 10%)
        - severity: Severity level (default: "high")
        
        Returns:
        - condition: Created alert condition
        """
        # Create name and description
        name = f"Trend Reversal Alert for {symbol}"
        description = f"Trend reversal of at least {threshold*100}% over {lookback_period} periods"
        
        # Create condition
        condition = TrendReversalAlertCondition(
            name=name,
            description=description,
            symbol=symbol,
            lookback_period=lookback_period,
            threshold=threshold,
            severity=severity
        )
        
        # Add to alert conditions
        self.add_alert_condition(condition)
        
        return condition
    
    def create_volatility_alert(self, symbol: str, lookback_period: int = 20, 
                               threshold: float = 2.0, severity: str = "medium"):
        """
        Create a volatility alert condition
        
        Parameters:
        - symbol: Stock symbol
        - lookback_period: Number of periods to look back (default: 20)
        - threshold: Volatility threshold as multiple of average (default: 2.0)
        - severity: Severity level (default: "medium")
        
        Returns:
        - condition: Created alert condition
        """
        # Create name and description
        name = f"Volatility Alert for {symbol}"
        description = f"Volatility spike of at least {threshold}x average over {lookback_period} periods"
        
        # Create condition
        condition = VolatilityAlertCondition(
            name=name,
            description=description,
            symbol=symbol,
            lookback_period=lookback_period,
            threshold=threshold,
            severity=severity
        )
        
        # Add to alert conditions
        self.add_alert_condition(condition)
        
        return condition
    
    def create_volume_alert(self, symbol: str, lookback_period: int = 20, 
                           threshold: float = 2.0, severity: str = "medium"):
        """
        Create a volume alert condition
        
        Parameters:
        - symbol: Stock symbol
        - lookback_period: Number of periods to look back (default: 20)
        - threshold: Volume threshold as multiple of average (default: 2.0)
        - severity: Severity level (default: "medium")
        
        Returns:
        - condition: Created alert condition
        """
        # Create name and description
        name = f"Volume Alert for {symbol}"
        description = f"Volume spike of at least {threshold}x average over {lookback_period} periods"
        
        # Create condition
        condition = VolumeAlertCondition(
            name=name,
            description=description,
            symbol=symbol,
            lookback_period=lookback_period,
            threshold=threshold,
            severity=severity
        )
        
        # Add to alert conditions
        self.add_alert_condition(condition)
        
        return condition
    
    def create_email_channel(self, name: str, smtp_server: str, smtp_port: int, 
                            username: str, password: str, sender: str, 
                            recipients: List[str]):
        """
        Create an email notification channel
        
        Parameters:
        - name: Name of the notification channel
        - smtp_server: SMTP server address
        - smtp_port: SMTP server port
        - username: SMTP username
        - password: SMTP password
        - sender: Sender email address
        - recipients: List of recipient email addresses
        
        Returns:
        - channel: Created notification channel
        """
        # Create channel
        channel = EmailNotificationChannel(
            name=name,
            smtp_server=smtp_server,
            smtp_port=smtp_port,
            username=username,
            password=password,
            sender=sender,
            recipients=recipients
        )
        
        # Add to notification channels
        self.add_notification_channel(channel)
        
        return channel
    
    def create_webhook_channel(self, name: str, webhook_url: str, headers: Dict = None):
        """
        Create a webhook notification channel
        
        Parameters:
        - name: Name of the notification channel
        - webhook_url: Webhook URL
        - headers: HTTP headers (default: None)
        
        Returns:
        - channel: Created notification channel
        """
        # Create channel
        channel = WebhookNotificationChannel(
            name=name,
            webhook_url=webhook_url,
            headers=headers
        )
        
        # Add to notification channels
        self.add_notification_channel(channel)
        
        return channel
    
    def create_sms_channel(self, name: str, api_key: str, api_secret: str, 
                          sender: str, recipients: List[str], provider: str = "twilio"):
        """
        Create an SMS notification channel
        
        Parameters:
        - name: Name of the notification channel
        - api_key: API key for SMS provider
        - api_secret: API secret for SMS provider
        - sender: Sender phone number
        - recipients: List of recipient phone numbers
        - provider: SMS provider (default: "twilio")
        
        Returns:
        - channel: Created notification channel
        """
        # Create channel
        channel = SMSNotificationChannel(
            name=name,
            api_key=api_key,
            api_secret=api_secret,
            sender=sender,
            recipients=recipients,
            provider=provider
        )
        
        # Add to notification channels
        self.add_notification_channel(channel)
        
        return channel
    
    def create_push_channel(self, name: str, api_key: str, app_id: str, 
                           provider: str = "firebase"):
        """
        Create a push notification channel
        
        Parameters:
        - name: Name of the notification channel
        - api_key: API key for push notification provider
        - app_id: Application ID
        - provider: Push notification provider (default: "firebase")
        
        Returns:
        - channel: Created notification channel
        """
        # Create channel
        channel = PushNotificationChannel(
            name=name,
            api_key=api_key,
            app_id=app_id,
            provider=provider
        )
        
        # Add to notification channels
        self.add_notification_channel(channel)
        
        return channel
    
    def plot_alert_history(self, start_time: datetime = None, 
                          end_time: datetime = None, 
                          severity: str = None,
                          figsize: Tuple[int, int] = (12, 8)):
        """
        Plot alert history
        
        Parameters:
        - start_time: Start time filter (default: None)
        - end_time: End time filter (default: None)
        - severity: Severity filter (default: None)
        - figsize: Figure size (default: (12, 8))
        
        Returns:
        - fig: Matplotlib figure
        """
        # Get filtered alert history
        history = self.get_alert_history(start_time, end_time, severity)
        
        if not history:
            logger.warning("No alert history to plot")
            return None
        
        # Extract timestamps and severities
        timestamps = [alert['timestamp'] for alert in history]
        severities = [alert['severity'] for alert in history]
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Define severity colors
        severity_colors = {
            'low': 'green',
            'medium': 'orange',
            'high': 'red'
        }
        
        # Plot alerts
        for i, (timestamp, severity) in enumerate(zip(timestamps, severities)):
            color = severity_colors.get(severity, 'blue')
            ax.scatter(timestamp, i, color=color, s=100, marker='o')
        
        # Set labels and title
        ax.set_xlabel('Time')
        ax.set_ylabel('Alert Index')
        ax.set_title('Alert History')
        
        # Add legend
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                     markerfacecolor=color, markersize=10, label=severity)
                          for severity, color in severity_colors.items()]
        ax.legend(handles=legend_elements, title='Severity')
        
        # Format x-axis
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Convert to Plotly figure for Dash, then close Matplotlib fig
        fig_plotly = self.convert_mpl_to_plotly(fig)
        return fig_plotly
    
    def export_alert_history(self, filename: str, format: str = 'csv'):
        """
        Export alert history to file
        
        Parameters:
        - filename: Output filename
        - format: Output format ('csv' or 'json', default: 'csv')
        """
        if not self.alert_history:
            logger.warning("No alert history to export")
            return
        
        try:
            if format == 'csv':
                # Convert to DataFrame
                df = pd.DataFrame(self.alert_history)
                
                # Save to CSV
                df.to_csv(filename, index=False)
                
            elif format == 'json':
                # Save to JSON
                with open(filename, 'w') as f:
                    json.dump(self.alert_history, f, indent=4, default=str)
                    
            else:
                logger.error(f"Unsupported export format: {format}")
                return
            
            logger.info(f"Exported alert history to {filename}")
            
        except Exception as e:
            logger.error(f"Error exporting alert history: {str(e)}")
