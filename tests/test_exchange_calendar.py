#
# Copyright 2016 Quantopian, Inc.
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

from os.path import (
    abspath,
    dirname,
    join,
)
from unittest import TestCase
from collections import namedtuple

import pandas as pd
import pytz
from pandas import (
    read_csv,
    datetime,
    Timestamp,
    Timedelta,
    date_range,
)
from pandas.util.testing import assert_frame_equal

from zipline.errors import (
    CalendarNameCollision,
    InvalidCalendarName,
)
from zipline.utils.calendars.exchange_calendar_nyse import NYSEExchangeCalendar
from zipline.utils.calendars.exchange_calendar import(
    register_calendar,
    deregister_calendar,
    get_calendar,
    clear_calendars,
)


class CalendarRegistrationTestCase(TestCase):

    def setUp(self):
        self.dummy_cal_type = namedtuple('DummyCal', ('name'))

    def tearDown(self):
        clear_calendars()

    def test_register_calendar(self):
        # Build a fake calendar
        dummy_cal = self.dummy_cal_type('DMY')

        # Try to register and retrieve the calendar
        register_calendar(dummy_cal)
        retr_cal = get_calendar('DMY')
        self.assertEqual(dummy_cal, retr_cal)

        # Try to register again, expecting a name collision
        with self.assertRaises(CalendarNameCollision):
            register_calendar(dummy_cal)

        # Deregister the calendar and ensure that it is removed
        deregister_calendar('DMY')
        with self.assertRaises(InvalidCalendarName):
            get_calendar('DMY')

    def test_force_registration(self):
        dummy_nyse = self.dummy_cal_type('NYSE')

        # Get the actual NYSE calendar
        real_nyse = get_calendar('NYSE')

        # Force a registration of the dummy NYSE
        register_calendar(dummy_nyse, force=True)

        # Ensure that the dummy overwrote the real calendar
        retr_cal = get_calendar('NYSE')
        self.assertNotEqual(real_nyse, retr_cal)


class ExchangeCalendarTestBase(object):

    # Override in subclasses.
    answer_key_filename = None
    calendar_class = None

    @staticmethod
    def load_answer_key(filename):
        """
        Load a CSV from tests/resources/calendars/{filename}.csv
        """
        fullpath = join(
            dirname(abspath(__file__)),
            'resources',
            'calendars',
            filename + '.csv',
        )
        return read_csv(
            fullpath,
            index_col=0,
            # NOTE: Merely passing parse_dates=True doesn't cause pandas to set
            # the dtype correctly, and passing all reasonable inputs to the
            # dtype kwarg cause read_csv to barf.
            parse_dates=[0, 1, 2],
        ).tz_localize('UTC')

    @classmethod
    def setupClass(cls):
        cls.answers = cls.load_answer_key(cls.answer_key_filename)
        cls.start_date = cls.answers.index[0]
        cls.end_date = cls.answers.index[-1]
        cls.calendar = cls.calendar_class(cls.start_date, cls.end_date)

    def test_calculated_against_csv(self):
        assert_frame_equal(self.calendar.schedule, self.answers)

    def test_is_open_on_minute(self):
        for market_minute in self.answers.market_open:
            market_minute_utc = market_minute.tz_localize('UTC')
            # The exchange should be classified as open on its first minute
            self.assertTrue(
                self.calendar.is_open_on_minute(market_minute_utc)
            )
            # Decrement minute by one, to minute where the market was not open
            pre_market = market_minute_utc - pd.Timedelta(minutes=1)
            self.assertFalse(
                self.calendar.is_open_on_minute(pre_market)
            )

    def test_open_and_close(self):
        for index, row in self.answers.iterrows():
            o_and_c = self.calendar.open_and_close(index)
            self.assertEqual(o_and_c[0],
                             row['market_open'].tz_localize('UTC'))
            self.assertEqual(o_and_c[1],
                             row['market_close'].tz_localize('UTC'))

    def test_no_nones_from_open_and_close(self):
        """
        Ensures that, for all minutes in a week, the open_and_close method
        never returns a tuple of Nones.
        """
        start_week = Timestamp('11/18/2012 12:00AM', tz='EST')
        end_week = start_week + Timedelta(days=7)
        minutes_in_week = date_range(start_week, end_week, freq='Min')

        for dt in minutes_in_week:
            open, close = self.calendar.open_and_close(dt)
            self.assertIsNotNone(open, "Open value is None")
            self.assertIsNotNone(close, "Close value is None")

    # def test_minutes_for_date(self):
    #     for date in self.answers.index:
    #         mins_for_date = self.calendar.minutes_for_date(date)

    def test_minute_window(self):
        for open in self.answers.market_open:
            open_tz = open.tz_localize('UTC')
            window = self.calendar.trading_minute_window(open_tz, 390, step=1)
            self.assertEqual(len(window), 390)


class NYSECalendarTestCase(ExchangeCalendarTestBase, TestCase):

    answer_key_filename = 'nyse'
    calendar_class = NYSEExchangeCalendar

    def test_newyears(self):
        """
        Check whether the ExchangeCalendar contains certain dates.
        """
        #     January 2012
        # Su Mo Tu We Th Fr Sa
        #  1  2  3  4  5  6  7
        #  8  9 10 11 12 13 14
        # 15 16 17 18 19 20 21
        # 22 23 24 25 26 27 28
        # 29 30 31

        start_dt = Timestamp('1/1/12', tz='UTC')
        end_dt = Timestamp('12/31/13', tz='UTC')
        trading_days = self.calendar.trading_days(start=start_dt, end=end_dt)

        day_after_new_years_sunday = datetime(
            2012, 1, 2, tzinfo=pytz.utc)

        self.assertNotIn(day_after_new_years_sunday,
                         trading_days.index,
                         """
 If NYE falls on a weekend, {0} the Monday after is a holiday.
 """.strip().format(day_after_new_years_sunday)
        )

        first_trading_day_after_new_years_sunday = datetime(
            2012, 1, 3, tzinfo=pytz.utc)

        self.assertIn(first_trading_day_after_new_years_sunday,
                      trading_days.index,
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

        new_years_day = datetime(
            2013, 1, 1, tzinfo=pytz.utc)

        self.assertNotIn(new_years_day,
                         trading_days.index,
                         """
 If NYE falls during the week, e.g. {0}, it is a holiday.
 """.strip().format(new_years_day)
        )

        first_trading_day_after_new_years = datetime(
            2013, 1, 2, tzinfo=pytz.utc)

        self.assertIn(first_trading_day_after_new_years,
                      trading_days.index,
                      """
 If the day after NYE falls during the week, {0} \
 is the first trading day.
 """.strip().format(first_trading_day_after_new_years)
        )

    def test_thanksgiving(self):
        """
        Check ExchangeCalendar Thanksgiving dates.
        """
        #     November 2005
        # Su Mo Tu We Th Fr Sa
        #        1  2  3  4  5
        #  6  7  8  9 10 11 12
        # 13 14 15 16 17 18 19
        # 20 21 22 23 24 25 26
        # 27 28 29 30

        start_dt = Timestamp('1/1/05', tz='UTC')
        end_dt = Timestamp('12/31/12', tz='UTC')
        trading_days = self.calendar.trading_days(start=start_dt,
                                                  end=end_dt)

        thanksgiving_with_four_weeks = datetime(
            2005, 11, 24, tzinfo=pytz.utc)

        self.assertNotIn(thanksgiving_with_four_weeks,
                         trading_days.index,
                         """
 If Nov has 4 Thursdays, {0} Thanksgiving is the last Thursady.
 """.strip().format(thanksgiving_with_four_weeks)
        )

        #     November 2006
        # Su Mo Tu We Th Fr Sa
        #           1  2  3  4
        #  5  6  7  8  9 10 11
        # 12 13 14 15 16 17 18
        # 19 20 21 22 23 24 25
        # 26 27 28 29 30
        thanksgiving_with_five_weeks = datetime(
            2006, 11, 23, tzinfo=pytz.utc)

        self.assertNotIn(thanksgiving_with_five_weeks,
                         trading_days.index,
                         """
 If Nov has 5 Thursdays, {0} Thanksgiving is not the last week.
 """.strip().format(thanksgiving_with_five_weeks)
        )

        first_trading_day_after_new_years_sunday = datetime(
            2012, 1, 3, tzinfo=pytz.utc)

        self.assertIn(first_trading_day_after_new_years_sunday,
                      trading_days.index,
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
        fourth_friday_open = Timestamp('11/23/2012 11:00AM', tz='EST')
        fourth_friday = Timestamp('11/23/2012 3:00PM', tz='EST')
        self.assertTrue(self.calendar.is_open_on_minute(fourth_friday_open))
        self.assertFalse(self.calendar.is_open_on_minute(fourth_friday))

        #    November 2013
        # Su Mo Tu We Th Fr Sa
        #                 1  2
        #  3  4  5  6  7  8  9
        # 10 11 12 13 14 15 16
        # 17 18 19 20 21 22 23
        # 24 25 26 27 28 29 30
        fifth_friday_open = Timestamp('11/29/2013 11:00AM', tz='EST')
        fifth_friday = Timestamp('11/29/2013 3:00PM', tz='EST')
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
        wednesday_before = Timestamp('7/3/2002 3:00PM', tz='EST')
        friday_after_open = Timestamp('7/5/2002 11:00AM', tz='EST')
        friday_after = Timestamp('7/5/2002 3:00PM', tz='EST')
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
        wednesday_before = Timestamp('7/3/2013 3:00PM', tz='EST')
        friday_after_open = Timestamp('7/5/2013 11:00AM', tz='EST')
        friday_after = Timestamp('7/5/2013 3:00PM', tz='EST')
        self.assertFalse(self.calendar.is_open_on_minute(wednesday_before))
        self.assertTrue(self.calendar.is_open_on_minute(friday_after_open))
        self.assertTrue(self.calendar.is_open_on_minute(friday_after))
