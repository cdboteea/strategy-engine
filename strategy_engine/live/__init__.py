"""Live signal detection."""
from .detector import detect_signals_for_strategy, detect_all_promoted, SignalFired
from .notification import send_telegram_signal

__all__ = [
    "detect_signals_for_strategy",
    "detect_all_promoted",
    "SignalFired",
    "send_telegram_signal",
]
