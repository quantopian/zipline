#
# Copyright 2019 Quantopian, Inc.
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
from functools import partial
from unittest import TestCase
from datetime import timedelta
import pandas as pd
from parameterized import parameterized

from zipline.utils.events import NDaysBeforeLastTradingDayOfWeek, AfterOpen, BeforeClose
from zipline.utils.events import NthTradingDayOfWeek

from .test_events import StatelessRulesTests, StatefulRulesTests, minutes_for_days


class TestStatelessRulesNYSE(StatelessRulesTests, TestCase):
    CALENDAR_STRING = "NYSE"

    HALF_SESSION = pd.Timestamp("2014-07-03")
    FULL_SESSION = pd.Timestamp("2014-09-24")

    def test_edge_cases_for_TradingDayOfWeek(self):
        """
        Test that we account for midweek holidays. Monday 01/20 is a holiday.
        Ensure that the trigger date for that week is adjusted
        appropriately, or thrown out if not enough trading days. Also, test
        that if we start the simulation on a day where we miss the trigger
        for that week, that the trigger is recalculated for next week.
        """

        #    December 2013
        # Su Mo Tu We Th Fr Sa
        #  1  2  3  4  5  6  7
        #  8  9 10 11 12 13 14
        # 15 16 17 18 19 20 21
        # 22 23 24 25 26 27 28
        # 29 30 31

        #    January 2014
        # Su Mo Tu We Th Fr Sa
        #           1  2  3  4
        #  5  6  7  8  9 10 11
        # 12 13 14 15 16 17 18
        # 19 20 21 22 23 24 25
        # 26 27 28 29 30 31

        # Include last day of 2013 to exercise case where the first day of a
        # week is in the previous year.

        # `week_start`
        rule = NthTradingDayOfWeek(0)
        rule.cal = self.cal

        expected = {
            # A Monday before the New Year.
            "2013-12-30": True,
            # Should not trigger on day after.
            "2013-12-31": False,
            # Should not trigger at market open of 1-2, a Thursday,
            # day after a holiday.
            "2014-01-02": False,
            # Test that the next Monday, which is at a start of a
            # 'normal' week successfully triggers.
            "2014-01-06": True,
            # Test around a Monday holiday, MLK day, to exercise week
            # start on a Tuesday.
            # MLK is 2014-01-20 in 2014.
            "2014-01-21": True,
            # Should not trigger at market open of 01-22, a Wednesday.
            "2014-01-22": False,
        }

        results = {
            x: rule.should_trigger(self.cal.session_first_minute(x))
            for x in expected.keys()
        }

        assert expected == results

        # Ensure that offset from start of week also works around edge cases.
        rule = NthTradingDayOfWeek(1)
        rule.cal = self.cal

        expected = {
            # Should trigger at market open of 12-31, day after week start.
            "2013-12-31": True,
            # Should not trigger at market open of 1-2, a Thursday,
            # day after a holiday.
            "2014-01-02": False,
            # Test around a Monday holiday, MLK day, to exercise
            # week start on a Tuesday.
            # MLK is 2014-01-20 in 2014.
            # Should trigger at market open, two days after Monday hoilday.
            "2014-01-22": True,
            # Should not trigger at market open of 01-23, a Thursday.
            "2014-01-23": False,
        }

        results = {
            x: rule.should_trigger(self.cal.session_first_minute(x))
            for x in expected.keys()
        }

        assert expected == results

        # `week_end`
        rule = NDaysBeforeLastTradingDayOfWeek(0)
        rule.cal = self.cal

        expected = {
            # Should trigger at market open of the Friday of the first week.
            "2014-01-03": True,
            # Should not trigger day before the end of the week.
            "2014-01-02": False,
            # Test around a Monday holiday, MLK day, to exercise week
            # start on a Tuesday.
            # MLK is 2014-01-20 in 2014.
            # Should trigger at market open, on Friday after the holiday.
            "2014-01-24": True,
            # Should not trigger at market open of 01-23, a Thursday.
            "2014-01-23": False,
        }

        results = {
            x: rule.should_trigger(self.cal.session_first_minute(x))
            for x in expected.keys()
        }

        assert expected == results

    @parameterized.expand([("week_start",), ("week_end",)])
    def test_week_and_time_composed_rule(self, rule_type):
        week_rule = (
            NthTradingDayOfWeek(0)
            if rule_type == "week_start"
            else NDaysBeforeLastTradingDayOfWeek(4)
        )
        time_rule = AfterOpen(minutes=60)

        week_rule.cal = self.cal
        time_rule.cal = self.cal

        composed_rule = week_rule & time_rule

        should_trigger = composed_rule.should_trigger

        week_minutes = self.cal.sessions_minutes(
            pd.Timestamp("2014-01-06"),
            pd.Timestamp("2014-01-10"),
        )

        dt = pd.Timestamp("2014-01-06 14:30:00", tz="UTC")
        trigger_day_offset = 0
        trigger_minute_offset = 60
        n_triggered = 0

        for m in week_minutes:
            if should_trigger(m):
                assert m == dt + timedelta(days=trigger_day_offset) + timedelta(
                    minutes=trigger_minute_offset
                )
                n_triggered += 1

        assert n_triggered == 1

    def test_offset_too_far(self):
        minute_groups = minutes_for_days(self.cal, ordered_days=True)

        # Neither rule should ever fire, since they are configured to fire
        # 11+ hours after the open or before the close.  a NYSE session is
        # never longer than 6.5 hours.
        after_open_rule = AfterOpen(hours=11, minutes=11)
        after_open_rule.cal = self.cal

        before_close_rule = BeforeClose(hours=11, minutes=5)
        before_close_rule.cal = self.cal

        for session_minutes in minute_groups:
            for minute in session_minutes:
                assert not after_open_rule.should_trigger(minute)
                assert not before_close_rule.should_trigger(minute)


class TestStatefulRulesNYSE(StatefulRulesTests, TestCase):
    CALENDAR_STRING = "NYSE"
