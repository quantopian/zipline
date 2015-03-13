__author__ = 'warren'


from zipline.utils import tradingcalendar_eex as tradingcalendar_eex
from zipline.finance.trading import TradingEnvironment
from unittest import TestCase


class TestTradingCalendarPower(TestCase):

    def test_calendar_vs_environment_eex(self):
        env = TradingEnvironment(
            bm_symbol='^EEX',
            exchange_tz='Europe/Berlin',
            env_trading_calendar=tradingcalendar_eex)
        env_start_index = \
            env.trading_days.searchsorted(tradingcalendar_eex.start)
        env_days = env.trading_days[env_start_index:]
        cal_days = tradingcalendar_eex.trading_days
        self.check_days(env_days, cal_days)

    def check_days(self, env_days, cal_days):
        diff = env_days - cal_days
        self.assertEqual(
            len(diff),
            0,
            "{diff} should be empty".format(diff=diff)
        )

        diff2 = cal_days - env_days
        self.assertEqual(
            len(diff2),
            0,
            "{diff} should be empty".format(diff=diff2)
        )
