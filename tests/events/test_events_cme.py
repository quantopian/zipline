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
import pandas as pd

from test_events import StatefulRulesTests, StatelessRulesTests, \
    minutes_for_days
from zipline.utils.events import AfterOpen


class TestStatelessRulesCME(StatelessRulesTests, TestCase):
    CALENDAR_STRING = "CME"

    HALF_SESSION = pd.Timestamp("2014-07-04", tz='UTC')
    FULL_SESSION = pd.Timestamp("2014-09-24", tz='UTC')

    def test_far_after_open(self):
        minute_groups = minutes_for_days(self.cal, ordered_days=True)
        after_open = AfterOpen(hours=9, minutes=25)
        after_open.cal = self.cal

        for session_minutes in minute_groups:
            for i, minute in enumerate(session_minutes):
                if i != 564:
                    self.assertFalse(after_open.should_trigger(minute))
                else:
                    self.assertTrue(after_open.should_trigger(minute))


class TestStatefulRulesCME(StatefulRulesTests, TestCase):
    CALENDAR_STRING = "CME"
