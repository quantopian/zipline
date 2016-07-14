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

import numpy as np
import pandas as pd
from pandas import (
    read_csv,
    Timestamp,
)
from pandas.util.testing import assert_index_equal
from zipline.errors import (
    CalendarNameCollision,
    InvalidCalendarName,
)
from zipline.utils.calendars.exchange_calendar_nyse import NYSEExchangeCalendar
from zipline.utils.calendars import(
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
            date_parser=lambda x: pd.Timestamp(x, tz='UTC')
        )

    @classmethod
    def setupClass(cls):
        cls.answers = cls.load_answer_key(cls.answer_key_filename)

        cls.start_date = cls.answers.index[0]
        cls.end_date = cls.answers.index[-1]
        cls.calendar = cls.calendar_class(cls.start_date, cls.end_date)

        cls.one_minute = pd.Timedelta(minutes=1)
        cls.one_hour = pd.Timedelta(hours=1)

    def test_calculated_against_csv(self):
        assert_index_equal(self.calendar.schedule.index, self.answers.index)

    def test_is_open_on_minute(self):
        one_minute = pd.Timedelta(minutes=1)

        for market_minute in self.answers.market_open:
            market_minute_utc = market_minute
            # The exchange should be classified as open on its first minute
            self.assertTrue(self.calendar.is_open_on_minute(market_minute_utc))

            # Decrement minute by one, to minute where the market was not open
            pre_market = market_minute_utc - one_minute
            self.assertFalse(self.calendar.is_open_on_minute(pre_market))

        for market_minute in self.answers.market_close:
            close_minute_utc = market_minute
            # should be open on its last minute
            self.assertTrue(self.calendar.is_open_on_minute(close_minute_utc))

            # increment minute by one minute, should be closed
            post_market = close_minute_utc + one_minute
            self.assertFalse(self.calendar.is_open_on_minute(post_market))

    def _verify_minute(self, calendar, minute,
                       next_open_answer, prev_open_answer,
                       next_close_answer, prev_close_answer):
        self.assertEqual(
            calendar.next_open(minute),
            next_open_answer
        )

        self.assertEqual(
            self.calendar.previous_open(minute),
            prev_open_answer
        )

        self.assertEqual(
            self.calendar.next_close(minute),
            next_close_answer
        )

        self.assertEqual(
            self.calendar.previous_close(minute),
            prev_close_answer
        )

    def test_next_prev_open_close(self):
        # for each session, check:
        # - the minute before the open
        # - the first minute of the session
        # - the second minute of the session
        # - the minute before the close
        # - the last minute of the session
        # - the first minute after the close
        answers_to_use = self.answers[1:-2]

        for idx, info in enumerate(answers_to_use.iterrows()):
            open_minute = info[1].iloc[0]
            close_minute = info[1].iloc[1]

            minute_before_open = open_minute - self.one_minute

            # answers_to_use starts at the second element of self.answers,
            # so self.answers.iloc[idx] is one element before, and
            # self.answers.iloc[idx + 2] is one element after the current
            # element
            previous_open = self.answers.iloc[idx].market_open
            next_open = self.answers.iloc[idx + 2].market_open
            previous_close = self.answers.iloc[idx].market_close
            next_close = self.answers.iloc[idx + 2].market_close

            # minute before open
            self._verify_minute(
                self.calendar, minute_before_open, open_minute, previous_open,
                close_minute, previous_close
            )

            # open minute
            self._verify_minute(
                self.calendar, open_minute, next_open, previous_open,
                close_minute, previous_close
            )

            # second minute of session
            self._verify_minute(
                self.calendar, open_minute + self.one_minute, next_open,
                open_minute, close_minute, previous_close
            )

            # minute before the close
            self._verify_minute(
                self.calendar, close_minute - self.one_minute, next_open,
                open_minute, close_minute, previous_close
            )

            # the close
            self._verify_minute(
                self.calendar, close_minute, next_open, open_minute,
                next_close, previous_close
            )

            # minute after the close
            self._verify_minute(
                self.calendar, close_minute + self.one_minute, next_open,
                open_minute, next_close, close_minute
            )

    def test_next_prev_minute(self):
        all_minutes = self.calendar.all_minutes

        # test 20,000 minutes because it takes too long to do the rest.
        for idx, minute in enumerate(all_minutes[1:20000]):
            self.assertEqual(
                all_minutes[idx + 2],
                self.calendar.next_minute(minute)
            )

            self.assertEqual(
                all_minutes[idx],
                self.calendar.previous_minute(minute)
            )

        # test a couple of non-market minutes
        for open_minute in self.answers.market_open[1:]:
            hour_before_open = open_minute - self.one_hour
            self.assertEqual(
                open_minute,
                self.calendar.next_minute(hour_before_open)
            )

        for close_minute in self.answers.market_close[1:]:
            hour_after_close = close_minute + self.one_hour
            self.assertEqual(
                close_minute,
                self.calendar.previous_minute(hour_after_close)
            )

    def test_minute_to_session_label(self):
        for idx, info in enumerate(self.answers[1:-2].iterrows()):
            session_label = info[1].name
            open_minute = info[1].iloc[0]
            close_minute = info[1].iloc[1]
            hour_into_session = open_minute + self.one_hour

            minute_before_session = open_minute - self.one_minute
            minute_after_session = close_minute + self.one_minute

            next_session_label = self.answers.iloc[idx + 2].name
            previous_session_label = self.answers.iloc[idx].name

            # verify that minutes inside a session resolve correctly
            minutes_that_resolve_to_this_session = [
                self.calendar.minute_to_session_label(open_minute),
                self.calendar.minute_to_session_label(open_minute,
                                                      direction="next"),
                self.calendar.minute_to_session_label(open_minute,
                                                      direction="previous"),
                self.calendar.minute_to_session_label(open_minute,
                                                      direction="none"),
                self.calendar.minute_to_session_label(hour_into_session),
                self.calendar.minute_to_session_label(hour_into_session,
                                                      direction="next"),
                self.calendar.minute_to_session_label(hour_into_session,
                                                      direction="previous"),
                self.calendar.minute_to_session_label(hour_into_session,
                                                      direction="none"),
                self.calendar.minute_to_session_label(close_minute),
                self.calendar.minute_to_session_label(close_minute,
                                                      direction="next"),
                self.calendar.minute_to_session_label(close_minute,
                                                      direction="previous"),
                self.calendar.minute_to_session_label(close_minute,
                                                      direction="none"),
                self.calendar.minute_to_session_label(minute_before_session),
                self.calendar.minute_to_session_label(
                    minute_before_session,
                    direction="next"
                ),
                self.calendar.minute_to_session_label(
                    minute_after_session,
                    direction="previous"
                ),
                session_label
            ]

            self.assertTrue(all(x == minutes_that_resolve_to_this_session[0]
                                for x in minutes_that_resolve_to_this_session))

            minutes_that_resolve_to_next_session = [
                self.calendar.minute_to_session_label(minute_after_session),
                self.calendar.minute_to_session_label(minute_after_session,
                                                      direction="next"),
                next_session_label
            ]

            self.assertTrue(all(x == minutes_that_resolve_to_next_session[0]
                                for x in minutes_that_resolve_to_next_session))

            self.assertEqual(
                self.calendar.minute_to_session_label(minute_before_session,
                                                      direction="previous"),
                previous_session_label
            )

            # make sure that exceptions are raised at the right time
            with self.assertRaises(ValueError):
                self.calendar.minute_to_session_label(open_minute, "asdf")

            with self.assertRaises(ValueError):
                self.calendar.minute_to_session_label(minute_before_session,
                                                      direction="none")

    def test_next_prev_session(self):
        session_labels = self.answers.index[1:-2]
        max_idx = len(session_labels) - 1

        # the very first session
        first_session_label = self.answers.index[0]
        with self.assertRaises(ValueError):
            self.calendar.previous_session_label(first_session_label)

        # all the sessions in the middle
        for idx, session_label in enumerate(session_labels):
            if idx < max_idx:
                self.assertEqual(
                    self.calendar.next_session_label(session_label),
                    session_labels[idx + 1]
                )

            if idx > 0:
                self.assertEqual(
                    self.calendar.previous_session_label(session_label),
                    session_labels[idx - 1]
                )

        # the very last session
        last_session_label = self.answers.index[-1]
        with self.assertRaises(ValueError):
            self.calendar.next_session_label(last_session_label)

    @staticmethod
    def _find_full_session(calendar):
        for session_label in calendar.schedule.index:
            if session_label not in calendar.early_closes:
                return session_label

        return None

    def test_minutes_for_period(self):
        # full session
        # find a session that isn't an early close.  start from the first
        # session, should be quick.
        full_session_label = self._find_full_session(self.calendar)
        if full_session_label is None:
            raise ValueError("Cannot find a full session to test!")

        minutes = self.calendar.minutes_for_session(full_session_label)
        _open, _close = self.calendar.open_and_close_for_session(
            full_session_label
        )

        np.testing.assert_array_equal(
            minutes,
            pd.date_range(start=_open, end=_close, freq="min")
        )

        # early close period
        early_close_session_label = self.calendar.early_closes[0]
        minutes_for_early_close = \
            self.calendar.minutes_for_session(early_close_session_label)
        _open, _close = self.calendar.open_and_close_for_session(
            early_close_session_label
        )

        np.testing.assert_array_equal(
            minutes_for_early_close,
            pd.date_range(start=_open, end=_close, freq="min")
        )

    def test_sessions_in_range(self):
        # pick two sessions
        session_count = len(self.calendar.schedule.index)

        first_idx = session_count / 3
        second_idx = 2 * first_idx

        first_session_label = self.calendar.schedule.index[first_idx]
        second_session_label = self.calendar.schedule.index[second_idx]

        answer_key = \
            self.calendar.schedule.index[first_idx:second_idx + 1]

        np.testing.assert_array_equal(
            answer_key,
            self.calendar.sessions_in_range(first_session_label,
                                            second_session_label)
        )

    def _get_session_block(self):
        # find and return a (full session, early close session, full session)
        # block

        shortened_session = self.calendar.early_closes[0]
        shortened_session_idx = \
            self.calendar.schedule.index.get_loc(shortened_session)

        session_before = self.calendar.schedule.index[
            shortened_session_idx - 1
        ]
        session_after = self.calendar.schedule.index[shortened_session_idx + 1]

        return [session_before, shortened_session, session_after]

    def test_minutes_in_range(self):
        sessions = self._get_session_block()

        first_open, first_close = self.calendar.open_and_close_for_session(
            sessions[0]
        )
        minute_before_first_open = first_open - self.one_minute

        middle_open, middle_close = \
            self.calendar.open_and_close_for_session(sessions[1])

        last_open, last_close = self.calendar.open_and_close_for_session(
            sessions[-1]
        )
        minute_after_last_close = last_close + self.one_minute

        # get all the minutes between first_open and last_close
        minutes1 = self.calendar.minutes_in_range(
            first_open,
            last_close
        )
        minutes2 = self.calendar.minutes_in_range(
            minute_before_first_open,
            minute_after_last_close
        )

        np.testing.assert_array_equal(minutes1, minutes2)

        # manually construct the minutes
        all_minutes = np.concatenate([
            pd.date_range(
                start=first_open,
                end=first_close,
                freq="min"
            ),
            pd.date_range(
                start=middle_open,
                end=middle_close,
                freq="min"
            ),
            pd.date_range(
                start=last_open,
                end=last_close,
                freq="min"
            )
        ])

        np.testing.assert_array_equal(all_minutes, minutes1)

    def test_minutes_for_sessions_in_range(self):
        sessions = self._get_session_block()

        minutes = self.calendar.minutes_for_sessions_in_range(
            sessions[0],
            sessions[-1]
        )

        # do it manually
        session0_minutes = self.calendar.minutes_for_session(sessions[0])
        session1_minutes = self.calendar.minutes_for_session(sessions[1])
        session2_minutes = self.calendar.minutes_for_session(sessions[2])

        concatenated_minutes = np.concatenate([
            session0_minutes.values,
            session1_minutes.values,
            session2_minutes.values
        ])

        np.testing.assert_array_equal(
            concatenated_minutes,
            minutes.values
        )

    def test_sessions_window(self):
        sessions = self._get_session_block()

        np.testing.assert_array_equal(
            self.calendar.sessions_window(sessions[0], len(sessions) - 1),
            self.calendar.sessions_in_range(sessions[0], sessions[-1])
        )

        np.testing.assert_array_equal(
            self.calendar.sessions_window(
                sessions[-1],
                -1 * (len(sessions) - 1)),
            self.calendar.sessions_in_range(sessions[0], sessions[-1])
        )

    def test_session_distance(self):
        sessions = self._get_session_block()

        self.assertEqual(2, self.calendar.session_distance(sessions[0],
                                                           sessions[-1]))

    def test_open_and_close_for_session(self):
        for index, row in self.answers.iterrows():
            session_label = row.name
            open_answer = row.iloc[0]
            close_answer = row.iloc[1]

            found_open, found_close = \
                self.calendar.open_and_close_for_session(session_label)

            self.assertEqual(open_answer, found_open)
            self.assertEqual(close_answer, found_close)


class NYSECalendarTestCase(ExchangeCalendarTestBase, TestCase):

    answer_key_filename = 'nyse'
    calendar_class = NYSEExchangeCalendar

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
