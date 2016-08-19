from zipline.errors import (
    InvalidCalendarName,
    CalendarNameCollision,
)
from zipline.utils.calendars.exchange_calendar_cfe import CFEExchangeCalendar
from zipline.utils.calendars.exchange_calendar_ice import ICEExchangeCalendar
from zipline.utils.calendars.exchange_calendar_nyse import NYSEExchangeCalendar
from zipline.utils.calendars.exchange_calendar_cme import CMEExchangeCalendar
from zipline.utils.calendars.exchange_calendar_bmf import BMFExchangeCalendar
from zipline.utils.calendars.exchange_calendar_lse import LSEExchangeCalendar
from zipline.utils.calendars.exchange_calendar_tsx import TSXExchangeCalendar
from zipline.utils.calendars.us_futures_calendar import (
    QuantopianUSFuturesCalendar,
)


NYSE_CALENDAR_EXCHANGE_NAMES = frozenset([
    "NYSE",
    "NASDAQ",
    "BATS",
])
CME_CALENDAR_EXCHANGE_NAMES = frozenset([
    "CBOT",
    "CME",
    "COMEX",
    "NYMEX",
])
ICE_CALENDAR_EXCHANGE_NAMES = frozenset([
    "ICEUS",
    "NYFE",
])
CFE_CALENDAR_EXCHANGE_NAMES = frozenset(["CFE"])
BMF_CALENDAR_EXCHANGE_NAMES = frozenset(["BMF"])
LSE_CALENDAR_EXCHANGE_NAMES = frozenset(["LSE"])
TSX_CALENDAR_EXCHANGE_NAMES = frozenset(["TSX"])

US_FUTURES_CALENDAR_NAMES = frozenset(["us_futures"])

_default_calendar_factories = {
    NYSE_CALENDAR_EXCHANGE_NAMES: NYSEExchangeCalendar,
    CME_CALENDAR_EXCHANGE_NAMES: CMEExchangeCalendar,
    ICE_CALENDAR_EXCHANGE_NAMES: ICEExchangeCalendar,
    CFE_CALENDAR_EXCHANGE_NAMES: CFEExchangeCalendar,
    BMF_CALENDAR_EXCHANGE_NAMES: BMFExchangeCalendar,
    LSE_CALENDAR_EXCHANGE_NAMES: LSEExchangeCalendar,
    TSX_CALENDAR_EXCHANGE_NAMES: TSXExchangeCalendar,
    US_FUTURES_CALENDAR_NAMES: QuantopianUSFuturesCalendar,
}


class TradingCalendarDispatcher(object):
    """
    A class for dispatching and caching trading calendars.

    Methods of a global instance of this class are provided by
    zipline.utils.calendar_utils.
    """
    def __init__(self, calendar_factories):
        self._calendars = {}
        self._calendar_factories = calendar_factories

    def get_calendar(self, name):
        """
        Retrieves an instance of an TradingCalendar whose name is given.

        Parameters
        ----------
        name : str
            The name of the TradingCalendar to be retrieved.

        Returns
        -------
        TradingCalendar
            The desired calendar.
        """
        try:
            return self._calendars[name]
        except KeyError:
            pass

        for names, factory in self._calendar_factories.items():
            if name in names:
                # Use the same calendar for all exchanges that share the same
                # factory.
                calendar = factory()
                self._calendars.update({n: calendar for n in names})
                return calendar

        raise InvalidCalendarName(calendar_name=name)

    def register_calendar(self, name, calendar, force=False):
        """
        Registers a calendar for retrieval by the get_calendar method.

        Parameters
        ----------
        name: str
            The key with which to register this calendar.
        calendar: TradingCalendar
            The calendar to be registered for retrieval.
        force : bool, optional
            If True, old calendars will be overwritten on a name collision.
            If False, name collisions will raise an exception. Default: False.

        Raises
        ------
        CalendarNameCollision
            If a calendar is already registered with the given calendar's name.
        """
        if force:
            self.deregister_calendar(name)

        if name in self._calendars or name in self._calendar_factories:
            raise CalendarNameCollision(calendar_name=name)

        self._calendars[name] = calendar

    def register_calendar_type(self, name, calendar_type, force=False):
        """
        Registers a calendar by type.

        Parameters
        ----------
        name: str
            The key with which to register this calendar.
        calendar_type: type
            The type of the calendar to register.
        force : bool, optional
            If True, old calendars will be overwritten on a name collision.
            If False, name collisions will raise an exception. Default: False.

        Raises
        ------
        CalendarNameCollision
            If a calendar is already registered with the given calendar's name.
        """
        if force:
            self._calendar_factories.pop(name, None)

        if name in self._calendars or name in self._calendar_factories:
            raise CalendarNameCollision(calendar_name=name)

        self._calendar_factories[name] = calendar_type

    def deregister_calendar(self, name):
        """
        If a calendar is registered with the given name, it is de-registered.

        Parameters
        ----------
        cal_name : str
            The name of the calendar to be deregistered.
        """
        self._calendars.pop(name, None)
        self._calendar_factories.pop(name, None)

    def clear_calendars(self):
        """
        Deregisters all current registered calendars
        """
        self._calendars.clear()
        self._calendar_factories.clear()


# We maintain a global calendar dispatcher so that users can just do
# `register_calendar('my_calendar', calendar) and then use `get_calendar`
# without having to thread around a dispatcher.
global_calendar_dispatcher = TradingCalendarDispatcher(
    _default_calendar_factories
)

get_calendar = global_calendar_dispatcher.get_calendar
clear_calendars = global_calendar_dispatcher.clear_calendars
deregister_calendar = global_calendar_dispatcher.deregister_calendar
register_calendar = global_calendar_dispatcher.register_calendar
register_calendar_type = global_calendar_dispatcher.register_calendar_type
