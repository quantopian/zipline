#
# Copyright 2012 Quantopian, Inc.
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

import datetime
import pytz

from zipline.utils.date_utils import (
    get_quarter,
    dates_of_quarter
)


class DateUtilsQuarterTests(TestCase):

    def test_dates_quarter_inverses(self):
        last_quarter = None

        for y in xrange(1900, 2050):
            for m in xrange(1, 13):

                dt = datetime.datetime(y, m, 1, tzinfo=pytz.utc)

                q = get_quarter(dt)
                if last_quarter:
                    self.assertGreater(q, last_quarter)

                boundaries = dates_of_quarter(q)
                self.assertTrue(
                    boundaries[0] <= dt <= boundaries[1],
                    "dates_of_quarter not inverse of get_quarter {0}"
                    .format(dt,))

                self.assertEqual(q, get_quarter(boundaries[0]))
                self.assertEqual(q, get_quarter(boundaries[1]))
