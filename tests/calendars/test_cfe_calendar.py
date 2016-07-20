from unittest import TestCase
import pandas as pd

from .test_trading_calendar import ExchangeCalendarTestBase
from zipline.utils.calendars.exchange_calendar_cfe import CFEExchangeCalendar


class CFECalendarTestCase(ExchangeCalendarTestBase, TestCase):
    answer_key_filename = "cfe"
    calendar_class = CFEExchangeCalendar

    MAX_SESSION_HOURS = 8

    def test_2016_holidays(self):
        # new years: jan 1
        # mlk: jan 18
        # presidents: feb 15
        # good friday: mar 25
        # mem day: may 30
        # independence day: july 4
        # labor day: sep 5
        # thanksgiving day: nov 24
        # christmas (observed): dec 26
        # new years (observed): jan 2 2017
        for day in ["2016-01-01", "2016-01-18", "2016-02-15", "2016-03-25",
                    "2016-05-30", "2016-07-04", "2016-09-05", "2016-11-24",
                    "2016-12-26", "2017-01-02"]:
            self.assertFalse(
                self.calendar.is_session(pd.Timestamp(day, tz='UTC'))
            )

    def test_2016_early_closes(self):
        # only early close is day after thanksgiving: nov 25
        dt = pd.Timestamp("2016-11-25", tz='UTC')
        self.assertTrue(dt in self.calendar.early_closes)

        market_close = self.calendar.schedule.loc[dt].market_close
        market_close = market_close.tz_localize("UTC").tz_convert(
            self.calendar.tz
        )
        self.assertEqual(12, market_close.hour)
        self.assertEqual(15, market_close.minute)
