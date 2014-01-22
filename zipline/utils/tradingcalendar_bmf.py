import pandas as pd
import pytz

from datetime import datetime, timedelta
from dateutil import rrule
from zipline.utils.tradingcalendar import TradingCalendar


class BvmfTradingCalendar(TradingCalendar):
    """docstring for BvmfTradingCaledar"""

    DEFAULT_OPEN_TIME = timedelta(hours=9, minutes=31)
    DEFAULT_CLOSE_TIME = timedelta(hours=16)
    LATE_OPEN_TIME = timedelta(hours=13)

    DEFAULT_OPEN_CLOSE_TIMES = (DEFAULT_OPEN_TIME, DEFAULT_CLOSE_TIME)
    LATE_OPEN_CLOSE_TIMES = (LATE_OPEN_TIME, DEFAULT_CLOSE_TIME)

    def __init__(self, start=pd.Timestamp('1994-01-01', tz='UTC'), end=None):
        super(BvmfTradingCalendar, self).__init__(start, end,
                                                  'America/Sao_Paulo')

    def get_non_trading_ruleset(self, start, end):
        non_trading_rules = []

        weekends = rrule.rrule(
            rrule.YEARLY,
            byweekday=(rrule.SA, rrule.SU),
            cache=True,
            dtstart=start,
            until=end
        )
        non_trading_rules.append(weekends)

        # Universal confraternization
        conf_universal = rrule.rrule(
            rrule.MONTHLY,
            byyearday=1,
            cache=True,
            dtstart=start,
            until=end
        )
        non_trading_rules.append(conf_universal)

        # Sao Paulo city birthday
        aniversario_sao_paulo = rrule.rrule(
            rrule.MONTHLY,
            bymonth=1,
            bymonthday=25,
            cache=True,
            dtstart=start,
            until=end
        )
        non_trading_rules.append(aniversario_sao_paulo)

        # Carnival Monday
        carnaval_segunda = rrule.rrule(
            rrule.MONTHLY,
            byeaster=-48,
            cache=True,
            dtstart=start,
            until=end
        )
        non_trading_rules.append(carnaval_segunda)

        # Carnival Tuesday
        carnaval_terca = rrule.rrule(
            rrule.MONTHLY,
            byeaster=-47,
            cache=True,
            dtstart=start,
            until=end
        )
        non_trading_rules.append(carnaval_terca)

        # Passion of the Christ
        sexta_paixao = rrule.rrule(
            rrule.MONTHLY,
            byeaster=-2,
            cache=True,
            dtstart=start,
            until=end
        )
        non_trading_rules.append(sexta_paixao)

        # Corpus Christi
        corpus_christi = rrule.rrule(
            rrule.MONTHLY,
            byeaster=60,
            cache=True,
            dtstart=start,
            until=end
        )
        non_trading_rules.append(corpus_christi)

        tiradentes = rrule.rrule(
            rrule.MONTHLY,
            bymonth=4,
            bymonthday=21,
            cache=True,
            dtstart=start,
            until=end
        )
        non_trading_rules.append(tiradentes)

        # Labor day
        dia_trabalho = rrule.rrule(
            rrule.MONTHLY,
            bymonth=5,
            bymonthday=1,
            cache=True,
            dtstart=start,
            until=end
        )
        non_trading_rules.append(dia_trabalho)

        # Constitutionalist Revolution
        constitucionalista = rrule.rrule(
            rrule.MONTHLY,
            bymonth=7,
            bymonthday=9,
            cache=True,
            dtstart=datetime(1997, 1, 1, tzinfo=pytz.utc),
            until=end
        )
        non_trading_rules.append(constitucionalista)

        # Independency day
        independencia = rrule.rrule(
            rrule.MONTHLY,
            bymonth=9,
            bymonthday=7,
            cache=True,
            dtstart=start,
            until=end
        )
        non_trading_rules.append(independencia)

        # Our Lady of Aparecida
        aparecida = rrule.rrule(
            rrule.MONTHLY,
            bymonth=10,
            bymonthday=12,
            cache=True,
            dtstart=start,
            until=end
        )
        non_trading_rules.append(aparecida)

        # All Souls' day
        finados = rrule.rrule(
            rrule.MONTHLY,
            bymonth=11,
            bymonthday=2,
            cache=True,
            dtstart=start,
            until=end
        )
        non_trading_rules.append(finados)

        # Proclamation of the Republic
        proclamacao_republica = rrule.rrule(
            rrule.MONTHLY,
            bymonth=11,
            bymonthday=15,
            cache=True,
            dtstart=start,
            until=end
        )
        non_trading_rules.append(proclamacao_republica)

        # Day of Black Awareness
        consciencia_negra = rrule.rrule(
            rrule.MONTHLY,
            bymonth=11,
            bymonthday=20,
            cache=True,
            dtstart=datetime(2004, 1, 1, tzinfo=pytz.utc),
            until=end
        )
        non_trading_rules.append(consciencia_negra)

        # Christmas Eve
        vespera_natal = rrule.rrule(
            rrule.MONTHLY,
            bymonth=12,
            bymonthday=24,
            cache=True,
            dtstart=start,
            until=end
        )
        non_trading_rules.append(vespera_natal)

        # Christmas
        natal = rrule.rrule(
            rrule.MONTHLY,
            bymonth=12,
            bymonthday=25,
            cache=True,
            dtstart=start,
            until=end
        )
        non_trading_rules.append(natal)

        # New Year Eve
        ano_novo = rrule.rrule(
            rrule.MONTHLY,
            bymonth=12,
            bymonthday=31,
            cache=True,
            dtstart=start,
            until=end
        )
        non_trading_rules.append(ano_novo)

        # New Year Eve on saturday
        ano_novo_sab = rrule.rrule(
            rrule.MONTHLY,
            bymonth=12,
            bymonthday=30,
            byweekday=rrule.FR,
            cache=True,
            dtstart=start,
            until=end
        )
        non_trading_rules.append(ano_novo_sab)

        non_trading_ruleset = rrule.rruleset()

        for rule in non_trading_rules:
            non_trading_ruleset.rrule(rule)

        return non_trading_ruleset

    def get_early_closes(self, start, end):
        start = self.canonicalize_datetime(start)
        end = self.canonicalize_datetime(end)

        early_close_rules = []

        # Ash Wednesday
        quarta_cinzas = rrule.rrule(
            rrule.MONTHLY,
            byeaster=-46,
            cache=True,
            dtstart=start,
            until=end
        )

        early_close_rules.append(quarta_cinzas)

        early_close_ruleset = rrule.rruleset()

        for rule in early_close_rules:
            early_close_ruleset.rrule(rule)
        early_closes = early_close_ruleset.between(start, end, inc=True)

        early_closes.sort()
        return pd.DatetimeIndex(early_closes)

    def get_exceptional_open_and_close_times(self, day):
        return self.LATE_OPEN_CLOSE_TIMES

    def get_default_open_close_times(self):
        return self.DEFAULT_OPEN_CLOSE_TIMES
