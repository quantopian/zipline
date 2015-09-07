#
# Copyright 2014 Quantopian, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pandas as pd
import pytz

from datetime import datetime
from dateutil import rrule
from zipline.utils.tradingcalendar import end, canonicalize_datetime, \
    get_open_and_closes

start = pd.Timestamp('1994-01-01', tz='UTC')


def get_non_trading_days(start, end):
    non_trading_rules = []

    start = canonicalize_datetime(start)
    end = canonicalize_datetime(end)

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

    non_trading_days = non_trading_ruleset.between(start, end, inc=True)

    # World Cup 2014 Opening
    non_trading_days.append(datetime(2014, 6, 12, tzinfo=pytz.utc))

    non_trading_days.sort()
    return pd.DatetimeIndex(non_trading_days)

non_trading_days = get_non_trading_days(start, end)
trading_day = pd.tseries.offsets.CDay(holidays=non_trading_days)


def get_trading_days(start, end, trading_day=trading_day):
    return pd.date_range(start=start.date(),
                         end=end.date(),
                         freq=trading_day).tz_localize('UTC')

trading_days = get_trading_days(start, end)


# Ash Wednesday
quarta_cinzas = rrule.rrule(
    rrule.MONTHLY,
    byeaster=-46,
    cache=True,
    dtstart=start,
    until=end
)


def get_early_closes(start, end):
    # TSX closed at 1:00 PM on december 24th.

    start = canonicalize_datetime(start)
    end = canonicalize_datetime(end)

    early_close_rules = []

    early_close_rules.append(quarta_cinzas)

    early_close_ruleset = rrule.rruleset()

    for rule in early_close_rules:
        early_close_ruleset.rrule(rule)
    early_closes = early_close_ruleset.between(start, end, inc=True)

    early_closes.sort()
    return pd.DatetimeIndex(early_closes)

early_closes = get_early_closes(start, end)


def get_open_and_close(day, early_closes):
    # only "early close" event in Bovespa actually is a late start
    # as the market only opens at 1pm
    open_hour = 13 if day in quarta_cinzas else 10
    market_open = pd.Timestamp(
        datetime(
            year=day.year,
            month=day.month,
            day=day.day,
            hour=open_hour,
            minute=00),
        tz='America/Sao_Paulo').tz_convert('UTC')
    market_close = pd.Timestamp(
        datetime(
            year=day.year,
            month=day.month,
            day=day.day,
            hour=16),
        tz='America/Sao_Paulo').tz_convert('UTC')

    return market_open, market_close

open_and_closes = get_open_and_closes(trading_days, early_closes,
                                      get_open_and_close)
