#
# Copyright 2013 Quantopian, Inc.
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

from pandas import read_csv
from pandas.util.testing import assert_frame_equal

from unittest import TestCase
# from zipline.utils import tradingcalendar
from zipline.utils.tradingcalendar_nyse import NYSEExchangeCalendar
# from zipline.utils import tradingcalendar_lse
# from zipline.utils import tradingcalendar_tse
# from zipline.utils import tradingcalendar_bmf
# import pytz
# import datetime
# from zipline.finance.trading import TradingEnvironment
# from nose.tools import nottest


class TradingCalendarTestCase(TestCase):

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


class NYSECalendarTestCase(TradingCalendarTestCase):
    answer_key_filename = 'nyse'
    calendar_class = NYSEExchangeCalendar


del TradingCalendarTestCase


# class TestTradingCalendar(TestCase):

#     def test_newyears(self):
#         """
#         Check whether tradingcalendar contains certain dates.
#         """
#         #     January 2012
#         # Su Mo Tu We Th Fr Sa
#         #  1  2  3  4  5  6  7
#         #  8  9 10 11 12 13 14
#         # 15 16 17 18 19 20 21
#         # 22 23 24 25 26 27 28
#         # 29 30 31

#         day_after_new_years_sunday = datetime.datetime(
#             2012, 1, 2, tzinfo=pytz.utc)

#         self.assertNotIn(day_after_new_years_sunday,
#                          tradingcalendar.trading_days,
#                          """
# If NYE falls on a weekend, {0} the Monday after is a holiday.
# """.strip().format(day_after_new_years_sunday)
#         )

#         first_trading_day_after_new_years_sunday = datetime.datetime(
#             2012, 1, 3, tzinfo=pytz.utc)

#         self.assertIn(first_trading_day_after_new_years_sunday,
#                       tradingcalendar.trading_days,
#                       """
# If NYE falls on a weekend, {0} the Tuesday after is the first trading day.
# """.strip().format(first_trading_day_after_new_years_sunday)
#         )

#         #     January 2013
#         # Su Mo Tu We Th Fr Sa
#         #        1  2  3  4  5
#         #  6  7  8  9 10 11 12
#         # 13 14 15 16 17 18 19
#         # 20 21 22 23 24 25 26
#         # 27 28 29 30 31

#         new_years_day = datetime.datetime(
#             2013, 1, 1, tzinfo=pytz.utc)

#         self.assertNotIn(new_years_day,
#                          tradingcalendar.trading_days,
#                          """
# If NYE falls during the week, e.g. {0}, it is a holiday.
# """.strip().format(new_years_day)
#         )

#         first_trading_day_after_new_years = datetime.datetime(
#             2013, 1, 2, tzinfo=pytz.utc)

#         self.assertIn(first_trading_day_after_new_years,
#                       tradingcalendar.trading_days,
#                       """
# If the day after NYE falls during the week, {0} \
# is the first trading day.
# """.strip().format(first_trading_day_after_new_years)
#         )

#     def test_thanksgiving(self):
#         """
#         Check tradingcalendar Thanksgiving dates.
#         """
#         #     November 2005
#         # Su Mo Tu We Th Fr Sa
#         #        1  2  3  4  5
#         #  6  7  8  9 10 11 12
#         # 13 14 15 16 17 18 19
#         # 20 21 22 23 24 25 26
#         # 27 28 29 30
#         thanksgiving_with_four_weeks = datetime.datetime(
#             2005, 11, 24, tzinfo=pytz.utc)

#         self.assertNotIn(thanksgiving_with_four_weeks,
#                          tradingcalendar.trading_days,
#                          """
# If Nov has 4 Thursdays, {0} Thanksgiving is the last Thursady.
# """.strip().format(thanksgiving_with_four_weeks)
#         )

#         #     November 2006
#         # Su Mo Tu We Th Fr Sa
#         #           1  2  3  4
#         #  5  6  7  8  9 10 11
#         # 12 13 14 15 16 17 18
#         # 19 20 21 22 23 24 25
#         # 26 27 28 29 30
#         thanksgiving_with_five_weeks = datetime.datetime(
#             2006, 11, 23, tzinfo=pytz.utc)

#         self.assertNotIn(thanksgiving_with_five_weeks,
#                          tradingcalendar.trading_days,
#                          """
# If Nov has 5 Thursdays, {0} Thanksgiving is not the last week.
# """.strip().format(thanksgiving_with_five_weeks)
#         )

#         first_trading_day_after_new_years_sunday = datetime.datetime(
#             2012, 1, 3, tzinfo=pytz.utc)

#         self.assertIn(first_trading_day_after_new_years_sunday,
#                       tradingcalendar.trading_days,
#                       """
# If NYE falls on a weekend, {0} the Tuesday after is the first trading day.
# """.strip().format(first_trading_day_after_new_years_sunday)
#         )

#     def test_day_after_thanksgiving(self):
#         early_closes = tradingcalendar.get_early_closes(
#             tradingcalendar.start,
#             tradingcalendar.end.replace(year=tradingcalendar.end.year + 1)
#         )

#         #    November 2012
#         # Su Mo Tu We Th Fr Sa
#         #              1  2  3
#         #  4  5  6  7  8  9 10
#         # 11 12 13 14 15 16 17
#         # 18 19 20 21 22 23 24
#         # 25 26 27 28 29 30
#         fourth_friday = datetime.datetime(2012, 11, 23, tzinfo=pytz.utc)
#         self.assertIn(fourth_friday, early_closes)

#         #    November 2013
#         # Su Mo Tu We Th Fr Sa
#         #                 1  2
#         #  3  4  5  6  7  8  9
#         # 10 11 12 13 14 15 16
#         # 17 18 19 20 21 22 23
#         # 24 25 26 27 28 29 30
#         fifth_friday = datetime.datetime(2013, 11, 29, tzinfo=pytz.utc)
#         self.assertIn(fifth_friday, early_closes)

#     def test_early_close_independence_day_thursday(self):
#         """
#         Until 2013, the market closed early the Friday after an
#         Independence Day on Thursday.  Since then, the early close is on
#         Wednesday.
#         """
#         early_closes = tradingcalendar.get_early_closes(
#             tradingcalendar.start,
#             tradingcalendar.end.replace(year=tradingcalendar.end.year + 1)
#         )
#         #      July 2002
#         # Su Mo Tu We Th Fr Sa
#         #     1  2  3  4  5  6
#         #  7  8  9 10 11 12 13
#         # 14 15 16 17 18 19 20
#         # 21 22 23 24 25 26 27
#         # 28 29 30 31
#         wednesday_before = datetime.datetime(2002, 7, 3, tzinfo=pytz.utc)
#         friday_after = datetime.datetime(2002, 7, 5, tzinfo=pytz.utc)
#         self.assertNotIn(wednesday_before, early_closes)
#         self.assertIn(friday_after, early_closes)

#         #      July 2013
#         # Su Mo Tu We Th Fr Sa
#         #     1  2  3  4  5  6
#         #  7  8  9 10 11 12 13
#         # 14 15 16 17 18 19 20
#         # 21 22 23 24 25 26 27
#         # 28 29 30 31
#         wednesday_before = datetime.datetime(2013, 7, 3, tzinfo=pytz.utc)
#         friday_after = datetime.datetime(2013, 7, 5, tzinfo=pytz.utc)
#         self.assertIn(wednesday_before, early_closes)
#         self.assertNotIn(friday_after, early_closes)
