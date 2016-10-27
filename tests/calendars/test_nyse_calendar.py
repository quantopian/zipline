from unittest import TestCase
import pandas as pd

from .test_trading_calendar import ExchangeCalendarTestBase
from zipline.utils.calendars.exchange_calendar_nyse import NYSEExchangeCalendar


class NYSECalendarTestCase(ExchangeCalendarTestBase, TestCase):

    answer_key_filename = 'nyse'
    calendar_class = NYSEExchangeCalendar

    MAX_SESSION_HOURS = 6.5

    def test_2012(self):
        # holidays we expect:
        holidays_2012 = [
            pd.Timestamp("2012-01-02", tz='UTC'),
            pd.Timestamp("2012-01-16", tz='UTC'),
            pd.Timestamp("2012-02-20", tz='UTC'),
            pd.Timestamp("2012-04-06", tz='UTC'),
            pd.Timestamp("2012-05-28", tz='UTC'),
            pd.Timestamp("2012-07-04", tz='UTC'),
            pd.Timestamp("2012-09-03", tz='UTC'),
            pd.Timestamp("2012-11-22", tz='UTC'),
            pd.Timestamp("2012-12-25", tz='UTC')
        ]

        for session_label in holidays_2012:
            self.assertNotIn(session_label, self.calendar.all_sessions)

        # early closes we expect:
        early_closes_2012 = [
            pd.Timestamp("2012-07-03", tz='UTC'),
            pd.Timestamp("2012-11-23", tz='UTC'),
            pd.Timestamp("2012-12-24", tz='UTC')
        ]

        for early_close_session_label in early_closes_2012:
            self.assertIn(early_close_session_label,
                          self.calendar.early_closes)

    def test_special_holidays(self):
        # 9/11
        # Sept 11, 12, 13, 14 2001
        self.assertNotIn(pd.Period("9/11/2001"), self.calendar.all_sessions)
        self.assertNotIn(pd.Period("9/12/2001"), self.calendar.all_sessions)
        self.assertNotIn(pd.Period("9/13/2001"), self.calendar.all_sessions)
        self.assertNotIn(pd.Period("9/14/2001"), self.calendar.all_sessions)

        # Hurricane Sandy
        # Oct 29, 30 2012
        self.assertNotIn(pd.Period("10/29/2012"), self.calendar.all_sessions)
        self.assertNotIn(pd.Period("10/30/2012"), self.calendar.all_sessions)

        # various national days of mourning
        # Gerald Ford - 1/2/2007
        self.assertNotIn(pd.Period("1/2/2007"), self.calendar.all_sessions)

        # Ronald Reagan - 6/11/2004
        self.assertNotIn(pd.Period("6/11/2004"), self.calendar.all_sessions)

        # Richard Nixon - 4/27/1994
        self.assertNotIn(pd.Period("4/27/1994"), self.calendar.all_sessions)

    def test_new_years(self):
        """
        Check whether the TradingCalendar contains certain dates.
        """
        #     January 2012
        # Su Mo Tu We Th Fr Sa
        #  1  2  3  4  5  6  7
        #  8  9 10 11 12 13 14
        # 15 16 17 18 19 20 21
        # 22 23 24 25 26 27 28
        # 29 30 31

        start_session = pd.Timestamp("2012-01-02", tz='UTC')
        end_session = pd.Timestamp("2013-12-31", tz='UTC')
        sessions = self.calendar.sessions_in_range(start_session, end_session)

        day_after_new_years_sunday = pd.Timestamp("2012-01-02",
                                                  tz='UTC')
        self.assertNotIn(day_after_new_years_sunday, sessions,
                         """
 If NYE falls on a weekend, {0} the Monday after is a holiday.
 """.strip().format(day_after_new_years_sunday)
        )

        first_trading_day_after_new_years_sunday = pd.Timestamp("2012-01-03",
                                                                tz='UTC')
        self.assertIn(first_trading_day_after_new_years_sunday, sessions,
                      """
 If NYE falls on a weekend, {0} the Tuesday after is the first trading day.
 """.strip().format(first_trading_day_after_new_years_sunday)
        )

        #     January 2013
        # Su Mo Tu We Th Fr Sa
        #        1  2  3  4  5
        #  6  7  8  9 10 11 12
        # 13 14 15 16 17 18 19
        # 20 21 22 23 24 25 26
        # 27 28 29 30 31

        new_years_day = pd.Timestamp("2013-01-01", tz='UTC')
        self.assertNotIn(new_years_day, sessions,
                         """
 If NYE falls during the week, e.g. {0}, it is a holiday.
 """.strip().format(new_years_day)
        )

        first_trading_day_after_new_years = pd.Timestamp("2013-01-02",
                                                         tz='UTC')
        self.assertIn(first_trading_day_after_new_years, sessions,
                      """
 If the day after NYE falls during the week, {0} \
 is the first trading day.
 """.strip().format(first_trading_day_after_new_years)
        )

    def test_thanksgiving(self):
        """
        Check TradingCalendar Thanksgiving dates.
        """
        #     November 2005
        # Su Mo Tu We Th Fr Sa
        #        1  2  3  4  5
        #  6  7  8  9 10 11 12
        # 13 14 15 16 17 18 19
        # 20 21 22 23 24 25 26
        # 27 28 29 30

        start_session_label = pd.Timestamp('2005-01-01', tz='UTC')
        end_session_label = pd.Timestamp('2012-12-31', tz='UTC')
        sessions = self.calendar.sessions_in_range(start_session_label,
                                                   end_session_label)

        thanksgiving_with_four_weeks = pd.Timestamp("2005-11-24", tz='UTC')

        self.assertNotIn(thanksgiving_with_four_weeks, sessions,
                         """
 If Nov has 4 Thursdays, {0} Thanksgiving is the last Thursday.
 """.strip().format(thanksgiving_with_four_weeks)
        )

        #     November 2006
        # Su Mo Tu We Th Fr Sa
        #           1  2  3  4
        #  5  6  7  8  9 10 11
        # 12 13 14 15 16 17 18
        # 19 20 21 22 23 24 25
        # 26 27 28 29 30
        thanksgiving_with_five_weeks = pd.Timestamp("2006-11-23", tz='UTC')

        self.assertNotIn(thanksgiving_with_five_weeks, sessions,
                         """
 If Nov has 5 Thursdays, {0} Thanksgiving is not the last week.
 """.strip().format(thanksgiving_with_five_weeks)
        )

        first_trading_day_after_new_years_sunday = pd.Timestamp("2012-01-03",
                                                                tz='UTC')

        self.assertIn(first_trading_day_after_new_years_sunday, sessions,
                      """
 If NYE falls on a weekend, {0} the Tuesday after is the first trading day.
 """.strip().format(first_trading_day_after_new_years_sunday)
        )

    def test_day_after_thanksgiving(self):
        #    November 2012
        # Su Mo Tu We Th Fr Sa
        #              1  2  3
        #  4  5  6  7  8  9 10
        # 11 12 13 14 15 16 17
        # 18 19 20 21 22 23 24
        # 25 26 27 28 29 30
        fourth_friday_open = pd.Timestamp('11/23/2012 11:00AM', tz='EST')
        fourth_friday = pd.Timestamp('11/23/2012 3:00PM', tz='EST')
        self.assertTrue(self.calendar.is_open_on_minute(fourth_friday_open))
        self.assertFalse(self.calendar.is_open_on_minute(fourth_friday))

        #    November 2013
        # Su Mo Tu We Th Fr Sa
        #                 1  2
        #  3  4  5  6  7  8  9
        # 10 11 12 13 14 15 16
        # 17 18 19 20 21 22 23
        # 24 25 26 27 28 29 30
        fifth_friday_open = pd.Timestamp('11/29/2013 11:00AM', tz='EST')
        fifth_friday = pd.Timestamp('11/29/2013 3:00PM', tz='EST')
        self.assertTrue(self.calendar.is_open_on_minute(fifth_friday_open))
        self.assertFalse(self.calendar.is_open_on_minute(fifth_friday))

    def test_early_close_independence_day_thursday(self):
        """
        Until 2013, the market closed early the Friday after an
        Independence Day on Thursday.  Since then, the early close is on
        Wednesday.
        """
        #      July 2002
        # Su Mo Tu We Th Fr Sa
        #     1  2  3  4  5  6
        #  7  8  9 10 11 12 13
        # 14 15 16 17 18 19 20
        # 21 22 23 24 25 26 27
        # 28 29 30 31
        wednesday_before = pd.Timestamp('7/3/2002 3:00PM', tz='EST')
        friday_after_open = pd.Timestamp('7/5/2002 11:00AM', tz='EST')
        friday_after = pd.Timestamp('7/5/2002 3:00PM', tz='EST')
        self.assertTrue(self.calendar.is_open_on_minute(wednesday_before))
        self.assertTrue(self.calendar.is_open_on_minute(friday_after_open))
        self.assertFalse(self.calendar.is_open_on_minute(friday_after))

        #      July 2013
        # Su Mo Tu We Th Fr Sa
        #     1  2  3  4  5  6
        #  7  8  9 10 11 12 13
        # 14 15 16 17 18 19 20
        # 21 22 23 24 25 26 27
        # 28 29 30 31
        wednesday_before = pd.Timestamp('7/3/2013 3:00PM', tz='EST')
        friday_after_open = pd.Timestamp('7/5/2013 11:00AM', tz='EST')
        friday_after = pd.Timestamp('7/5/2013 3:00PM', tz='EST')
        self.assertFalse(self.calendar.is_open_on_minute(wednesday_before))
        self.assertTrue(self.calendar.is_open_on_minute(friday_after_open))
        self.assertTrue(self.calendar.is_open_on_minute(friday_after))
