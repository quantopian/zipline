# flake8: noqa
# reexport trading_calendars for backwards compat
from trading_calendars import (
    clear_calendars,
    deregister_calendar,
    get_calendar,
    register_calendar,
    register_calendar_alias,
    register_calendar_type,
    TradingCalendar,
)
