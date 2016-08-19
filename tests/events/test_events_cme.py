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

from test_events import StatefulRulesTests, StatelessRulesTests


class TestStatelessRulesCME(StatelessRulesTests, TestCase):
    CALENDAR_STRING = "CME"

    HALF_SESSION = pd.Timestamp("2014-07-04", tz='UTC')
    FULL_SESSION = pd.Timestamp("2014-09-24", tz='UTC')


class TestStatefulRulesCME(StatefulRulesTests, TestCase):
    CALENDAR_STRING = "CME"
