from datetime import time
from pandas import Timedelta
from pandas.tseries.holiday import(
    AbstractHolidayCalendar,
    Holiday,
    DateOffset,
    MO,
    weekend_to_monday,
    GoodFriday,
)
from pytz import timezone

from zipline.utils.calendars.exchange_calendar import ExchangeCalendar
from zipline.utils.calendars.calendar_helpers import normalize_date
from zipline.utils.calendars.exchange_calendar_lse import (
    Christmas,
    WeekendChristmas,
    BoxingDay,
    WeekendBoxingDay,
)

MONDAY, TUESDAY, WEDNESDAY, THURSDAY, FRIDAY, SATURDAY, SUNDAY = range(7)

# New Year's Day
TSXNewYearsDay = Holiday(
    "New Year's Day",
    month=1,
    day=1,
    observance=weekend_to_monday,
)
# Ontario Family Day
FamilyDay = Holiday(
    "Family Day",
    month=2,
    day=1,
    offset=DateOffset(weekday=MO(3)),
    start_date='2008-01-01',
)
# Victoria Day
VictoriaDay = Holiday(
    'Victoria Day',
    month=5,
    day=25,
    offset=DateOffset(weekday=MO(-1)),
)
# Canada Day
CanadaDay = Holiday(
    'Canada Day',
    month=7,
    day=1,
    observance=weekend_to_monday,
)
# Civic Holiday
CivicHoliday = Holiday(
    'Civic Holiday',
    month=8,
    day=1,
    offset=DateOffset(weekday=MO(1)),
)
# Labor Day
LaborDay = Holiday(
    'Labor Day',
    month=9,
    day=1,
    offset=DateOffset(weekday=MO(1)),
)
# Thanksgiving
Thanksgiving = Holiday(
    'Thanksgiving',
    month=10,
    day=1,
    offset=DateOffset(weekday=MO(2)),
)


class TSXHolidayCalendar(AbstractHolidayCalendar):
    """
    Non-trading days for the TSX.

    See NYSEExchangeCalendar for full description.
    """
    rules = [
        TSXNewYearsDay,
        FamilyDay,
        GoodFriday,
        VictoriaDay,
        CanadaDay,
        CivicHoliday,
        LaborDay,
        Thanksgiving,
        Christmas,
        WeekendChristmas,
        BoxingDay,
        WeekendBoxingDay,
    ]


class TSXExchangeCalendar(ExchangeCalendar):
    """
    Exchange calendar for the Toronto Stock Exchange

    Open Time: 9:30 AM, EST
    Close Time: 4:00 PM, EST

    Regularly-Observed Holidays:
    - New Years Day (observed on first business day on/after)
    - Family Day (Third Monday in February after 2008)
    - Good Friday
    - Victoria Day (Monday before May 25th)
    - Canada Day (July 1st, observed first business day after)
    - Civic Holiday (First Monday in August)
    - Labor Day (First Monday in September)
    - Thanksgiving (Second Monday in October)
    - Christmas Day
    - Dec. 27th (if Christmas is on a weekend)
    - Boxing Day
    - Dec. 28th (if Boxing Day is on a weekend)
    """

    exchange_name = 'TSX'
    native_timezone = timezone('Canada/Atlantic')
    open_time = time(9, 31)
    close_time = time(16)
    open_offset = 0
    close_offset = 0

    holidays_calendar = TSXHolidayCalendar()
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
