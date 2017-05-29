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
from datetime import time
from datetime import datetime
from os.path import (
    abspath,
    dirname,
    join,
)
from unittest import TestCase

import numpy as np
import pandas as pd
from nose_parameterized import parameterized
from pandas import read_csv
from pandas.tslib import Timedelta
from pandas.util.testing import assert_index_equal
from pytz import timezone
from toolz import concat

from zipline.errors import (
    CalendarNameCollision,
    InvalidCalendarName,
)

from zipline.testing.predicates import assert_equal
from zipline.utils.calendars import (
    deregister_calendar,
    deregister_calendar_type,
    get_calendar,
    register_calendar,
)
from zipline.utils.calendars.calendar_utils import (
    _default_calendar_aliases,
    _default_calendar_factories,
    register_calendar_type,

)
from zipline.utils.calendars.trading_calendar import days_at_time, \
    TradingCalendar


class FakeCalendar(TradingCalendar):
    @property
    def name(self):
        return "DMY"

    @property
    def tz(self):
        return "Asia/Ulaanbaatar"

    @property
    def open_time(self):
        return time(11, 13)

    @property
    def close_time(self):
        return time(11, 49)


class CalendarStartEndTestCase(TestCase):
    @parameterized.expand([
        (pd.Timestamp('2010-1-4'), pd.Timestamp('2010-1-8')),
        (datetime(2010, 1, 4), datetime(2010, 1, 8)),
        ('2010-1-4', '2010-1-8'),
    ])
    def test_start_end(self, start, end):
        """
        Check TradingCalendar with defined start/end dates.
        """
        calendar = FakeCalendar(start=start, end=end)
        expected_first = pd.Timestamp(start, tz='UTC')
        expected_last = pd.Timestamp(end, tz='UTC')
        self.assertTrue(calendar.first_trading_session == expected_first)
        self.assertTrue(calendar.last_trading_session == expected_last)


class CalendarRegistrationTestCase(TestCase):
    def setUp(self):
        self.dummy_cal_type = FakeCalendar

    def tearDown(self):
        deregister_calendar('DMY')

    def test_register_calendar(self):
        # Build a fake calendar
        dummy_cal = self.dummy_cal_type()

        # Try to register and retrieve the calendar
        register_calendar('DMY', dummy_cal)
        retr_cal = get_calendar('DMY', start=dummy_cal.start,
                                end=dummy_cal.end)
        self.assertEqual(dummy_cal, retr_cal)

        # Try to register again, expecting a name collision
        with self.assertRaises(CalendarNameCollision):
            register_calendar('DMY', dummy_cal)

        # Deregister the calendar and ensure that it is removed
        deregister_calendar('DMY', start=dummy_cal.start, end=dummy_cal.end)
        with self.assertRaises(InvalidCalendarName):
            get_calendar('DMY', start=dummy_cal.start, end=dummy_cal.end)

    def test_register_calendar_type(self):
        register_calendar_type("DMY", self.dummy_cal_type)
        retr_cal = get_calendar("DMY")
        self.assertEqual(self.dummy_cal_type, type(retr_cal))

        # Try to register again, expecting a name collision
        with self.assertRaises(CalendarNameCollision):
            register_calendar_type('DMY', self.dummy_cal_type)

        # Deregister the calendar type and ensure that it is removed
        deregister_calendar_type('DMY')
        with self.assertRaises(InvalidCalendarName):
            get_calendar('DMY')

    def test_force_registration(self):
        # Build a fake calendar
        dummy_cal = self.dummy_cal_type()

        # Try to register and retrieve the calendar
        register_calendar("DMY", dummy_cal)
        first_dummy = get_calendar("DMY", start=dummy_cal.start,
                                   end=dummy_cal.end)

        # force-register a new instance
        register_calendar("DMY", self.dummy_cal_type(), force=True)

        second_dummy = get_calendar("DMY", start=dummy_cal.start,
                                    end=dummy_cal.end)

        self.assertNotEqual(first_dummy, second_dummy)


class DefaultsTestCase(TestCase):
    def test_default_calendars(self):
        for name in concat([_default_calendar_factories,
                            _default_calendar_aliases]):
            self.assertIsNotNone(get_calendar(name),
                                 "get_calendar(%r) returned None" % name)


class DaysAtTimeTestCase(TestCase):
    @parameterized.expand([
        # NYSE standard day
        (
            '2016-07-19', 0, time(9, 31), timezone('US/Eastern'),
            '2016-07-19 9:31',
        ),
        # CME standard day
        (
            '2016-07-19', -1, time(17, 1), timezone('America/Chicago'),
            '2016-07-18 17:01',
        ),
        # CME day after DST start
        (
            '2004-04-05', -1, time(17, 1), timezone('America/Chicago'),
            '2004-04-04 17:01'
        ),
        # ICE day after DST start
        (
            '1990-04-02', -1, time(19, 1), timezone('America/Chicago'),
            '1990-04-01 19:01',
        ),
    ])
    def test_days_at_time(self, day, day_offset, time_offset, tz, expected):
        days = pd.DatetimeIndex([pd.Timestamp(day, tz=tz)])
        result = days_at_time(days, time_offset, tz, day_offset)[0]
        expected = pd.Timestamp(expected, tz=tz).tz_convert('UTC')
        self.assertEqual(result, expected)


class ExchangeCalendarTestBase(object):

    # Override in subclasses.
    answer_key_filename = None
    calendar_class = None

    GAPS_BETWEEN_SESSIONS = True

    MAX_SESSION_HOURS = 0

    @staticmethod
    def load_answer_key(filename):
        """
        Load a CSV from tests/resources/calendars/{filename}.csv
        """
        fullpath = join(
            dirname(abspath(__file__)),
            '../resources',
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

    def test_sanity_check_session_lengths(self):
        # make sure that no session is longer than self.MAX_SESSION_HOURS hours
        for session in self.calendar.all_sessions:
            o, c = self.calendar.open_and_close_for_session(session)
            delta = c - o
            self.assertTrue((delta.seconds / 3600) <= self.MAX_SESSION_HOURS)

    def test_calculated_against_csv(self):
        assert_index_equal(self.calendar.schedule.index, self.answers.index)

    def test_is_open_on_minute(self):
        one_minute = pd.Timedelta(minutes=1)

        for market_minute in self.answers.market_open:
            market_minute_utc = market_minute
            # The exchange should be classified as open on its first minute
            self.assertTrue(self.calendar.is_open_on_minute(market_minute_utc))

            if self.GAPS_BETWEEN_SESSIONS:
                # Decrement minute by one, to minute where the market was not
                # open
                pre_market = market_minute_utc - one_minute
                self.assertFalse(self.calendar.is_open_on_minute(pre_market))

        for market_minute in self.answers.market_close:
            close_minute_utc = market_minute
            # should be open on its last minute
            self.assertTrue(self.calendar.is_open_on_minute(close_minute_utc))

            if self.GAPS_BETWEEN_SESSIONS:
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
        # - the minute before the open (if gaps exist between sessions)
        # - the first minute of the session
        # - the second minute of the session
        # - the minute before the close
        # - the last minute of the session
        # - the first minute after the close (if gaps exist between sessions)
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
            if self.GAPS_BETWEEN_SESSIONS:
                self._verify_minute(
                    self.calendar, minute_before_open, open_minute,
                    previous_open, close_minute, previous_close
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
            if self.GAPS_BETWEEN_SESSIONS:
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
        if self.GAPS_BETWEEN_SESSIONS:
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
                session_label
            ]

            if self.GAPS_BETWEEN_SESSIONS:
                minutes_that_resolve_to_this_session.append(
                    self.calendar.minute_to_session_label(
                        minute_before_session
                    )
                )
                minutes_that_resolve_to_this_session.append(
                    self.calendar.minute_to_session_label(
                        minute_before_session,
                        direction="next"
                    )
                )

                minutes_that_resolve_to_this_session.append(
                    self.calendar.minute_to_session_label(
                        minute_after_session,
                        direction="previous"
                    )
                )

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

            if self.GAPS_BETWEEN_SESSIONS:
                with self.assertRaises(ValueError):
                    self.calendar.minute_to_session_label(
                        minute_before_session,
                        direction="none"
                    )

    @parameterized.expand([
        (1, 0),
        (2, 0),
        (2, 1),
    ])
    def test_minute_index_to_session_labels(self, interval, offset):
        minutes = self.calendar.minutes_for_sessions_in_range(
            pd.Timestamp('2011-01-04', tz='UTC'),
            pd.Timestamp('2011-04-04', tz='UTC'),
        )
        minutes = minutes[range(offset, len(minutes), interval)]

        np.testing.assert_array_equal(
            np.array(minutes.map(self.calendar.minute_to_session_label),
                     dtype='datetime64[ns]'),
            self.calendar.minute_index_to_session_labels(minutes)
        )

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

        if self.GAPS_BETWEEN_SESSIONS:
            np.testing.assert_array_equal(minutes1, minutes2)
        else:
            # if no gaps, then minutes2 should have 2 extra minutes
            np.testing.assert_array_equal(minutes1, minutes2[1:-1])

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

            # Test that the methods for just session open and close produce the
            # same values as the method for getting both.
            alt_open = self.calendar.session_open(session_label)
            self.assertEqual(alt_open, found_open)

            alt_close = self.calendar.session_close(session_label)
            self.assertEqual(alt_close, found_close)

            self.assertEqual(open_answer, found_open)
            self.assertEqual(close_answer, found_close)

    def test_session_opens_in_range(self):
        found_opens = self.calendar.session_opens_in_range(
            self.answers.index[0],
            self.answers.index[-1],
        )

        assert_equal(found_opens, self.answers['market_open'])

    def test_session_closes_in_range(self):
        found_closes = self.calendar.session_closes_in_range(
            self.answers.index[0],
            self.answers.index[-1],
        )

        assert_equal(found_closes, self.answers['market_close'])

    def test_daylight_savings(self):
        # 2004 daylight savings switches:
        # Sunday 2004-04-04 and Sunday 2004-10-31

        # make sure there's no weirdness around calculating the next day's
        # session's open time.

        for date in ["2004-04-05", "2004-11-01"]:
            next_day = pd.Timestamp(date, tz='UTC')
            open_date = next_day + Timedelta(days=self.calendar.open_offset)

            the_open = self.calendar.schedule.loc[next_day].market_open

            localized_open = the_open.tz_localize("UTC").tz_convert(
                self.calendar.tz
            )

            self.assertEqual(
                (open_date.year, open_date.month, open_date.day),
                (localized_open.year, localized_open.month, localized_open.day)
            )

            self.assertEqual(
                self.calendar.open_time.hour,
                localized_open.hour
            )

            self.assertEqual(
                self.calendar.open_time.minute,
                localized_open.minute
            )
