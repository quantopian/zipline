from datetime import time
from pandas import Timedelta
from pandas.tseries.holiday import(
    AbstractHolidayCalendar,
    Holiday,
    Easter,
    Day,
    GoodFriday,
)
from pytz import timezone

from zipline.utils.calendars.exchange_calendar import ExchangeCalendar
from zipline.utils.calendars.calendar_helpers import normalize_date

MONDAY, TUESDAY, WEDNESDAY, THURSDAY, FRIDAY, SATURDAY, SUNDAY = range(7)

# Universal Confraternization (new years day)
ConfUniversal = Holiday(
    'Dia da Confraternizacao Universal',
    month=1,
    day=1,
)
# Sao Paulo city birthday
AniversarioSaoPaulo = Holiday(
    'Aniversario de Sao Paulo',
    month=1,
    day=25,
)
# Carnival Monday
CarnavalSegunda = Holiday(
    'Carnaval Segunda',
    month=1,
    day=1,
    offset=[Easter(), Day(-48)]
)
# Carnival Tuesday
CarnavalTerca = Holiday(
    'Carnaval Terca',
    month=1,
    day=1,
    offset=[Easter(), Day(-47)]
)
# Ash Wednesday (short day)
QuartaCinzas = Holiday(
    'Quarta Cinzas',
    month=1,
    day=1,
    offset=[Easter(), Day(-46)]
)
# Good Friday
SextaPaixao = GoodFriday
# Feast of the Most Holy Body of Christ
CorpusChristi = Holiday(
    'Corpus Christi',
    month=1,
    day=1,
    offset=[Easter(), Day(60)]
)
# Tiradentes Memorial
Tiradentes = Holiday(
    'Tiradentes',
    month=4,
    day=21,
)
# Labor Day
DiaTrabalho = Holiday(
    'Dia Trabalho',
    month=5,
    day=1,
)
# Constitutionalist Revolution
Constitucionalista = Holiday(
    'Constitucionalista',
    month=7,
    day=9,
    start_date='1997-01-01'
)
# Independence Day
Independencia = Holiday(
    'Independencia',
    month=9,
    day=7,
)
# Our Lady of Aparecida
Aparecida = Holiday(
    'Nossa Senhora de Aparecida',
    month=10,
    day=12,
)
# All Souls' Day
Finados = Holiday(
    'Dia dos Finados',
    month=11,
    day=2,
)
# Proclamation of the Republic
ProclamacaoRepublica = Holiday(
    'Proclamacao da Republica',
    month=11,
    day=15,
)
# Day of Black Awareness
ConscienciaNegra = Holiday(
    'Dia da Consciencia Negra',
    month=11,
    day=20,
    start_date='2004-01-01'
)
# Christmas Eve
VesperaNatal = Holiday(
    'Vespera Natal',
    month=12,
    day=24,
)
# Christmas
Natal = Holiday(
    'Natal',
    month=12,
    day=25,
)
# New Year's Eve
AnoNovo = Holiday(
    'Ano Novo',
    month=12,
    day=31,
)
# New Year's Eve falls on Saturday
AnoNovoSabado = Holiday(
    'Ano Novo Sabado',
    month=12,
    day=30,
    days_of_week=(FRIDAY),
)


class BMFHolidayCalendar(AbstractHolidayCalendar):
    """
    Non-trading days for the BM&F.

    See NYSEExchangeCalendar for full description.
    """
    rules = [
        ConfUniversal,
        AniversarioSaoPaulo,
        CarnavalSegunda,
        CarnavalTerca,
        SextaPaixao,
        CorpusChristi,
        Tiradentes,
        DiaTrabalho,
        Constitucionalista,
        Independencia,
        Aparecida,
        Finados,
        ProclamacaoRepublica,
        ConscienciaNegra,
        VesperaNatal,
        Natal,
        AnoNovo,
        AnoNovoSabado,
    ]


class BMFLateOpenCalendar(AbstractHolidayCalendar):
    """
    Regular early close calendar for NYSE
    """
    rules = [
        QuartaCinzas,
    ]


class BMFExchangeCalendar(ExchangeCalendar):
    """
    Exchange calendar for BM&F BOVESPA

    Open Time: 10:00 AM, Brazil/Sao Paulo
    Close Time: 4:00 PM, Brazil/Sao Paulo

    Regularly-Observed Holidays:
    - Universal Confraternization (New year's day, Jan 1)
    - Sao Paulo City Anniversary (Jan 25)
    - Carnaval Monday (48 days before Easter)
    - Carnaval Tuesday (47 days before Easter)
    - Passion of the Christ (Good Friday, 2 days before Easter)
    - Corpus Christi (60 days after Easter)
    - Tiradentes (April 21)
    - Labor day (May 1)
    - Constitutionalist Revolution (July 9 after 1997)
    - Independence Day (September 7)
    - Our Lady of Aparecida Feast (October 12)
    - All Souls' Day (November 2)
    - Proclamation of the Republic (November 15)
    - Day of Black Awareness (November 20 after 2004)
    - Christmas (December 24 and 25)
    - Day before New Year's Eve (December 30 if NYE falls on a Saturday)
    - New Year's Eve (December 31)
    """

    exchange_name = 'BMF'
    native_timezone = timezone('America/Sao_Paulo')
    open_time = time(10, 01)
    close_time = time(17)

    # Does the market open or close on a different calendar day, compared to
    # the calendar day assigned by the exchange to this session?
    open_offset = 0
    close_offset = 0

    holidays_calendar = BMFHolidayCalendar()
    special_opens_calendars = [
        (time(13, 01), BMFLateOpenCalendar()),
    ]
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
