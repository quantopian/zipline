from datetime import time
from pandas.tseries.holiday import(
    AbstractHolidayCalendar,
    Holiday,
    Easter,
    Day,
    GoodFriday,
)
from pytz import timezone

from .trading_calendar import (
    TradingCalendar,
    FRIDAY
)

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
    days_of_week=(FRIDAY,),
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


class BMFExchangeCalendar(TradingCalendar):
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

    name = "BMF"
    tz = timezone('America/Sao_Paulo')
    open_time = time(10, 1)
    close_time = time(17)

    # Does the market open or close on a different calendar day, compared to
    # the calendar day assigned by the exchange to this session?
    open_offset = 0
    close_offset = 0

    holidays_calendar = BMFHolidayCalendar()
    special_opens_calendars = [
        (time(13, 1), BMFLateOpenCalendar()),
    ]
    special_closes_calendars = ()

    holidays_adhoc = ()

    special_opens_adhoc = ()
    special_closes_adhoc = ()
