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
from unittest import TestCase
from datetime import timedelta
import pandas as pd
from nose_parameterized import parameterized

from zipline.testing import parameter_space
from zipline.utils.events import NDaysBeforeLastTradingDayOfWeek, AfterOpen
from zipline.utils.events import NthTradingDayOfWeek

from test_events import StatelessRulesTests, StatefulRulesTests


class TestStatelessRulesNYSE(StatelessRulesTests, TestCase):
    CALENDAR_STRING = "NYSE"

    HALF_SESSION = pd.Timestamp("2014-07-03", tz='UTC')
    FULL_SESSION = pd.Timestamp("2014-09-24", tz='UTC')

    @parameter_space(
        rule_offset=(0, 1, 2, 3, 4),
        start_offset=(0, 1, 2, 3, 4),
        type=('week_start', 'week_end')
    )
    def test_edge_cases_for_TradingDayOfWeek(self,
                                             rule_offset,
                                             start_offset,
                                             type):
        """
        Test that we account for midweek holidays. Monday 01/20 is a holiday.
        Ensure that the trigger date for that week is adjusted
        appropriately, or thrown out if not enough trading days. Also, test
        that if we start the simulation on a day where we miss the trigger
        for that week, that the trigger is recalculated for next week.
        """

        sim_start = pd.Timestamp('2014-01-06', tz='UTC') + \
            timedelta(days=start_offset)

        delta = timedelta(days=start_offset)

        jan_minutes = self.cal.minutes_for_sessions_in_range(
            pd.Timestamp("2014-01-06", tz='UTC') + delta,
            pd.Timestamp("2014-01-31", tz='UTC')
        )

        if type == 'week_start':
            rule = NthTradingDayOfWeek
            # Expect to trigger on the first trading day of the week, plus the
            # offset
            trigger_periods = [
                pd.Timestamp('2014-01-06', tz='UTC'),
                pd.Timestamp('2014-01-13', tz='UTC'),
                pd.Timestamp('2014-01-21', tz='UTC'),
                pd.Timestamp('2014-01-27', tz='UTC'),
            ]
            trigger_periods = \
                [x + timedelta(days=rule_offset) for x in trigger_periods]
        else:
            rule = NDaysBeforeLastTradingDayOfWeek
            # Expect to trigger on the last trading day of the week, minus the
            # offset
            trigger_periods = [
                pd.Timestamp('2014-01-10', tz='UTC'),
                pd.Timestamp('2014-01-17', tz='UTC'),
                pd.Timestamp('2014-01-24', tz='UTC'),
                pd.Timestamp('2014-01-31', tz='UTC'),
            ]
            trigger_periods = \
                [x - timedelta(days=rule_offset) for x in trigger_periods]

        rule.cal = self.cal
        should_trigger = rule(rule_offset).should_trigger

        # If offset is 4, there is not enough trading days in the short week,
        # and so it should not trigger
        if rule_offset == 4:
            del trigger_periods[2]

        # Filter out trigger dates that happen before the simulation starts
        trigger_periods = [x for x in trigger_periods if x >= sim_start]

        # Get all the minutes on the trigger dates
        trigger_minutes = self.cal.minutes_for_session(trigger_periods[0])
        for period in trigger_periods[1:]:
            trigger_minutes += self.cal.minutes_for_session(period)

        expected_n_triggered = len(trigger_minutes)
        trigger_minutes_iter = iter(trigger_minutes)

        n_triggered = 0
        for m in jan_minutes:
            if should_trigger(m):
                self.assertEqual(m, next(trigger_minutes_iter))
                n_triggered += 1

        self.assertEqual(n_triggered, expected_n_triggered)

    @parameterized.expand([('week_start',), ('week_end',)])
    def test_week_and_time_composed_rule(self, type):
        week_rule = NthTradingDayOfWeek(0) if type == 'week_start' else \
            NDaysBeforeLastTradingDayOfWeek(4)
        time_rule = AfterOpen(minutes=60)

        week_rule.cal = self.cal
        time_rule.cal = self.cal

        composed_rule = week_rule & time_rule

        should_trigger = composed_rule.should_trigger

        week_minutes = self.cal.minutes_for_sessions_in_range(
            pd.Timestamp("2014-01-06", tz='UTC'),
            pd.Timestamp("2014-01-10", tz='UTC')
        )

        dt = pd.Timestamp('2014-01-06 14:30:00', tz='UTC')
        trigger_day_offset = 0
        trigger_minute_offset = 60
        n_triggered = 0

        for m in week_minutes:
            if should_trigger(m):
                self.assertEqual(m, dt + timedelta(days=trigger_day_offset) +
                                 timedelta(minutes=trigger_minute_offset))
                n_triggered += 1

        self.assertEqual(n_triggered, 1)


class TestStatefulRulesNYSE(StatefulRulesTests, TestCase):
    CALENDAR_STRING = "NYSE"
