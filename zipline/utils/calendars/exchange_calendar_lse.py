from datetime import time
from pandas import Timedelta
from pandas.tseries.holiday import(
    AbstractHolidayCalendar,
    Holiday,
    DateOffset,
    MO,
    weekend_to_monday,
    GoodFriday,
    EasterMonday,
)
from pytz import timezone

from zipline.utils.calendars.exchange_calendar import ExchangeCalendar
from zipline.utils.calendars.calendar_helpers import normalize_date

MONDAY, TUESDAY, WEDNESDAY, THURSDAY, FRIDAY, SATURDAY, SUNDAY = range(7)

# New Year's Day
LSENewYearsDay = Holiday(
    "New Year's Day",
    month=1,
    day=1,
    observance=weekend_to_monday,
)
# Early May bank holiday
MayBank = Holiday(
    "Early May Bank Holiday",
    month=5,
    offset=DateOffset(weekday=MO(1)),
)
# Spring bank holiday
SpringBank = Holiday(
    "Spring Bank Holiday",
    month=5,
    day=31,
    offset=DateOffset(weekday=MO(-1)),
)
# Summer bank holiday
SummerBank = Holiday(
    "Summer Bank Holiday",
    month=8,
    day=31,
    offset=DateOffset(weekday=MO(-1)),
)
# Christmas
Christmas = Holiday(
    "Christmas",
    month=12,
    day=25,
)
# If christmas day is Saturday Monday 27th is a holiday
# If christmas day is sunday the Tuesday 27th is a holiday
WeekendChristmas = Holiday(
    "Weekend Christmas",
    month=12,
    day=27,
    days_of_week=(MONDAY, TUESDAY),
)
# Boxing day
BoxingDay = Holiday(
    "Boxing Day",
    month=12,
    day=26,
)
# If boxing day is saturday then Monday 28th is a holiday
# If boxing day is sunday then Tuesday 28th is a holiday
WeekendBoxingDay = Holiday(
    "Weekend Boxing Day",
    month=12,
    day=28,
    days_of_week=(MONDAY, TUESDAY),
)


class LSEHolidayCalendar(AbstractHolidayCalendar):
    """
    Non-trading days for the LSE.

    See NYSEExchangeCalendar for full description.
    """
    rules = [
        LSENewYearsDay,
        GoodFriday,
        EasterMonday,
        MayBank,
        SpringBank,
        SummerBank,
        Christmas,
        WeekendChristmas,
        BoxingDay,
        WeekendBoxingDay,
    ]


class LSEExchangeCalendar(ExchangeCalendar):
    """
    Exchange calendar for the London Stock Exchange

    Open Time: 8:00 AM, GMT
    Close Time: 4:30 PM, GMT

    Regularly-Observed Holidays:
    - New Years Day (observed on first business day on/after)
    - Good Friday
    - Easter Monday
    - Early May Bank Holiday (first Monday in May)
    - Spring Bank Holiday (last Monday in May)
    - Summer Bank Holiday (last Monday in May)
    - Christmas Day
    - Dec. 27th (if Christmas is on a weekend)
    - Boxing Day
    - Dec. 28th (if Boxing Day is on a weekend)
    """

    exchange_name = 'LSE'
    native_timezone = timezone('Europe/London')
    open_time = time(8, 01)
    close_time = time(16, 30)
    open_offset = 0
    close_offset = 0

    holidays_calendar = LSEHolidayCalendar()
    special_opens_calendars = ()
    special_closes_calendars = ()

    holidays_adhoc = ()

    special_opens_adhoc = ()
    special_closes_adhoc = ()

    @property
    def name(self):
        """
        The name of this exchange calendar.
        E.g.: 'NYSE', 'LSE', 'CME Energy'
        """
        return self.exchange_name

    @property
    def tz(self):
        """
        The native timezone of the exchange.
        """
        return self.native_timezone

    def is_open_on_minute(self, dt):
        """
        Is the exchange open (accepting orders) at @dt.

        Parameters
        ----------
        dt : Timestamp

        Returns
        -------
        bool
            True if  exchange is open at the given dt, otherwise False.
        """
        # Retrieve the exchange session relevant for this datetime
        session = self.session_date(dt)
        # Retrieve the open and close for this exchange session
        open, close = self.open_and_close(session)
        # Is @dt within the trading hours for this exchange session
        return open <= dt and dt <= close

    def is_open_on_day(self, dt):
        """
        Is the exchange open (accepting orders) anytime during the calendar day
        containing @dt.

        Parameters
        ----------
        dt : Timestamp

        Returns
        -------
        bool
            True if  exchange is open at any time during the day containing @dt
        """
        dt_normalized = normalize_date(dt)
        return dt_normalized in self.schedule.index

    def trading_days(self, start, end):
        """
        Calculates all of the exchange sessions between the given
        start and end, inclusive.

        SD: Should @start and @end are UTC-canonicalized, as our exchange
        sessions are. If not, then it's not clear how this method should behave
        if @start and @end are both in the middle of the day. Here, I assume we
        need to map @start and @end to session.

        Parameters
        ----------
        start : Timestamp
        end : Timestamp

        Returns
        -------
        DatetimeIndex
            A DatetimeIndex populated with all of the trading days between
            the given start and end.
        """
        start_session = self.session_date(start)
        end_session = self.session_date(end)
        # Increment end_session by one day, beucase .loc[s:e] return all values
        # in the DataFrame up to but not including `e`.
        # end_session += Timedelta(days=1)
        return self.schedule.loc[start_session:end_session]

    def open_and_close(self, dt):
        """
        Given a datetime, returns a tuple of timestamps of the
        open and close of the exchange session containing the datetime.

        SD: Should we accept an arbitrary datetime, or should we first map it
        to and exchange session using session_date. Need to check what the
        consumers expect. Here, I assume we need to map it to a session.

        Parameters
        ----------
        dt : Timestamp
            A dt in a session whose open and close are needed.

        Returns
        -------
        (Timestamp, Timestamp)
            The open and close for the given dt.
        """
        session = self.session_date(dt)
        return self._get_open_and_close(session)

    def _get_open_and_close(self, session_date):
        """
        Retrieves the open and close for a given session.

        Parameters
        ----------
        session_date : Timestamp
            The canonicalized session_date whose open and close are needed.

        Returns
        -------
        (Timestamp, Timestamp) or (None, None)
            The open and close for the given dt, or Nones if the given date is
            not a session.
        """
        # Return a tuple of nones if the given date is not a session.
        if session_date not in self.schedule.index:
            return (None, None)

        o_and_c = self.schedule.loc[session_date]
        # `market_open` and `market_close` should be timezone aware, but pandas
        # 0.16.1 does not appear to support this:
        # http://pandas.pydata.org/pandas-docs/stable/whatsnew.html#datetime-with-tz  # noqa
        return (o_and_c['market_open'].tz_localize('UTC'),
                o_and_c['market_close'].tz_localize('UTC'))

    def session_date(self, dt):
        """
        Given a datetime, returns the UTC-canonicalized date of the exchange
        session in which the time belongs. If the time is not in an exchange
        session (while the market is closed), returns the date of the next
        exchange session after the time.

        Parameters
        ----------
        dt : Timestamp
            A timezone-aware Timestamp.

        Returns
        -------
        Timestamp
            The date of the exchange session in which dt belongs.
        """
        # Check if the dt is after the market close
        # If so, advance to the next day
        if self.is_open_on_day(dt):
            _, close = self._get_open_and_close(normalize_date(dt))
            if dt > close:
                dt += Timedelta(days=1)

        while not self.is_open_on_day(dt):
            dt += Timedelta(days=1)

        return normalize_date(dt)
