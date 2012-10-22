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

from unittest2 import TestCase

import zipline.utils.factory as factory
from zipline.sources import DataFrameSource


class TestDataFrameSource(TestCase):
    def test_streaming_of_df(self):
        source, df = factory.create_test_df_source()

        for expected_dt, expected_price in df.iterrows():
            sid0 = source.next()
            sid1 = source.next()

            assert expected_dt == sid0.dt == sid1.dt
            assert expected_price[0] == sid0.price
            assert expected_price[1] == sid1.price

    def test_sid_filtering(self):
        _, df = factory.create_test_df_source()
        source = DataFrameSource(df, sids=[0])
        assert 1 not in [event.sid for event in source], \
            "DataFrameSource should only stream selected sid 0, not sid 1."
