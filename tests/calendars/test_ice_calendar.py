from unittest import TestCase
import pandas as pd

from .test_trading_calendar import ExchangeCalendarTestBase
from zipline.utils.calendars.exchange_calendar_ice import ICEExchangeCalendar


class ICECalendarTestCase(ExchangeCalendarTestBase, TestCase):

    answer_key_filename = 'ice'
    calendar_class = ICEExchangeCalendar
    MAX_SESSION_HOURS = 22

    def test_hurricane_sandy_one_day(self):
        self.assertFalse(
            self.calendar.is_session(pd.Timestamp("2012-10-29", tz='UTC'))
        )

        # ICE wasn't closed on day 2 of hurricane sandy
        self.assertTrue(
            self.calendar.is_session(pd.Timestamp("2012-10-30", tz='UTC'))
        )

    def test_2016_holidays(self):
        # 2016 holidays:
        # new years: 2016-01-01
        # good friday: 2016-03-25
        # christmas (observed): 2016-12-26

        for date in ["2016-01-01", "2016-03-25", "2016-12-26"]:
            self.assertFalse(
                self.calendar.is_session(pd.Timestamp(date, tz='UTC'))
            )

    def test_2016_early_closes(self):
        # 2016 early closes
        # mlk: 2016-01-18
        # presidents: 2016-02-15
        # mem day: 2016-05-30
        # independence day: 2016-07-04
        # labor: 2016-09-05
        # thanksgiving: 2016-11-24
        for date in ["2016-01-18", "2016-02-15", "2016-05-30", "2016-07-04",
                     "2016-09-05", "2016-11-24"]:
            dt = pd.Timestamp(date, tz='UTC')
            self.assertTrue(dt in self.calendar.early_closes)

            market_close = self.calendar.schedule.loc[dt].market_close
            self.assertEqual(
                13,     # all ICE early closes are 1 pm local
                market_close.tz_localize("UTC").tz_convert(
                    self.calendar.tz
                ).hour
            )
