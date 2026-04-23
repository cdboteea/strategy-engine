"""Paper-trading layer — auto-open positions on signal fire, MTM daily, auto-close on exit rules."""
from .book import (
    open_position_from_signal,
    mark_to_market_all,
    close_position,
    snapshot_nav,
    current_nav,
)
from .reporting import (
    list_positions,
    realized_pnl_by_strategy,
    overall_summary,
)

__all__ = [
    "open_position_from_signal",
    "mark_to_market_all",
    "close_position",
    "snapshot_nav",
    "current_nav",
    "list_positions",
    "realized_pnl_by_strategy",
    "overall_summary",
]
