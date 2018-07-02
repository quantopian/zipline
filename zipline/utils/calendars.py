# flake8: noqa
# reexport trading_calendars for backwards compat
from trading_calendars import (
    clear_calendars,
    deregister_calendar,
    exchange_calendar_nyse,
    get_calendar,
    register_calendar,
    register_calendar_alias,
    register_calendar_type,
    trading_calendar,
    TradingCalendar,
    us_futures_calendar
)
