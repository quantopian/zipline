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
from datetime import timedelta
import os

from unittest import TestCase

from numpy import (
    arange,
    array,
    int64,
    float64,
    full,
    nan,
    zeros,
)
from numpy.testing import assert_almost_equal, assert_array_equal
from pandas import (
    DataFrame,
    DatetimeIndex,
    Timestamp,
    Timedelta,
    NaT,
    date_range,
)
from testfixtures import TempDirectory

from zipline.data.minute_bars import (
    BcolzMinuteBarWriter,
    BcolzMinuteBarReader,
    BcolzMinuteOverlappingData,
    US_EQUITIES_MINUTES_PER_DAY,
)
from zipline.finance.trading import TradingEnvironment


TEST_CALENDAR_START = Timestamp('2015-06-01', tz='UTC')
TEST_CALENDAR_STOP = Timestamp('2015-06-30', tz='UTC')


class BcolzMinuteBarTestCase(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.env = TradingEnvironment()
        all_market_opens = cls.env.open_and_closes.market_open
        all_market_closes = cls.env.open_and_closes.market_close
        indexer = all_market_opens.index.slice_indexer(
            start=TEST_CALENDAR_START,
            end=TEST_CALENDAR_STOP
        )
        cls.market_opens = all_market_opens[indexer]
        cls.market_closes = all_market_closes[indexer]
        cls.test_calendar_start = cls.market_opens.index[0]
        cls.test_calendar_stop = cls.market_opens.index[-1]

    def setUp(self):

        self.dir_ = TempDirectory()
        self.dir_.create()
        self.dest = self.dir_.getpath('minute_bars')
        os.makedirs(self.dest)
        self.writer = BcolzMinuteBarWriter(
            TEST_CALENDAR_START,
            self.dest,
            self.market_opens,
            self.market_closes,
            US_EQUITIES_MINUTES_PER_DAY,
        )
        self.reader = BcolzMinuteBarReader(self.dest)

    def tearDown(self):
        self.dir_.cleanup()

    def test_write_one_ohlcv(self):
        minute = self.market_opens[self.test_calendar_start]
        sid = 1
        data = DataFrame(
            data={
                'open': [10.0],
                'high': [20.0],
                'low': [30.0],
                'close': [40.0],
                'volume': [50.0]
            },
            index=[minute])
        self.writer.write(sid, data)

        open_price = self.reader.get_value(sid, minute, 'open')

        self.assertEquals(10.0, open_price)

        high_price = self.reader.get_value(sid, minute, 'high')

        self.assertEquals(20.0, high_price)

        low_price = self.reader.get_value(sid, minute, 'low')

        self.assertEquals(30.0, low_price)

        close_price = self.reader.get_value(sid, minute, 'close')

        self.assertEquals(40.0, close_price)

        volume_price = self.reader.get_value(sid, minute, 'volume')

        self.assertEquals(50.0, volume_price)

    def test_write_two_bars(self):
        minute_0 = self.market_opens[self.test_calendar_start]
        minute_1 = minute_0 + timedelta(minutes=1)
        sid = 1
        data = DataFrame(
            data={
                'open': [10.0, 11.0],
                'high': [20.0, 21.0],
                'low': [30.0, 31.0],
                'close': [40.0, 41.0],
                'volume': [50.0, 51.0]
            },
            index=[minute_0, minute_1])
        self.writer.write(sid, data)

        open_price = self.reader.get_value(sid, minute_0, 'open')

        self.assertEquals(10.0, open_price)

        high_price = self.reader.get_value(sid, minute_0, 'high')

        self.assertEquals(20.0, high_price)

        low_price = self.reader.get_value(sid, minute_0, 'low')

        self.assertEquals(30.0, low_price)

        close_price = self.reader.get_value(sid, minute_0, 'close')

        self.assertEquals(40.0, close_price)

        volume_price = self.reader.get_value(sid, minute_0, 'volume')

        self.assertEquals(50.0, volume_price)

        open_price = self.reader.get_value(sid, minute_1, 'open')

        self.assertEquals(11.0, open_price)

        high_price = self.reader.get_value(sid, minute_1, 'high')

        self.assertEquals(21.0, high_price)

        low_price = self.reader.get_value(sid, minute_1, 'low')

        self.assertEquals(31.0, low_price)

        close_price = self.reader.get_value(sid, minute_1, 'close')

        self.assertEquals(41.0, close_price)

        volume_price = self.reader.get_value(sid, minute_1, 'volume')

        self.assertEquals(51.0, volume_price)

    def test_write_on_second_day(self):
        second_day = self.test_calendar_start + 1
        minute = self.market_opens[second_day]
        sid = 1
        data = DataFrame(
            data={
                'open': [10.0],
                'high': [20.0],
                'low': [30.0],
                'close': [40.0],
                'volume': [50.0]
            },
            index=[minute])
        self.writer.write(sid, data)

        open_price = self.reader.get_value(sid, minute, 'open')

        self.assertEquals(10.0, open_price)

        high_price = self.reader.get_value(sid, minute, 'high')

        self.assertEquals(20.0, high_price)

        low_price = self.reader.get_value(sid, minute, 'low')

        self.assertEquals(30.0, low_price)

        close_price = self.reader.get_value(sid, minute, 'close')

        self.assertEquals(40.0, close_price)

        volume_price = self.reader.get_value(sid, minute, 'volume')

        self.assertEquals(50.0, volume_price)

    def test_write_empty(self):
        minute = self.market_opens[self.test_calendar_start]
        sid = 1
        data = DataFrame(
            data={
                'open': [0],
                'high': [0],
                'low': [0],
                'close': [0],
                'volume': [0]
            },
            index=[minute])
        self.writer.write(sid, data)

        open_price = self.reader.get_value(sid, minute, 'open')

        assert_almost_equal(nan, open_price)

        high_price = self.reader.get_value(sid, minute, 'high')

        assert_almost_equal(nan, high_price)

        low_price = self.reader.get_value(sid, minute, 'low')

        assert_almost_equal(nan, low_price)

        close_price = self.reader.get_value(sid, minute, 'close')

        assert_almost_equal(nan, close_price)

        volume_price = self.reader.get_value(sid, minute, 'volume')

        assert_almost_equal(0, volume_price)

    def test_write_on_multiple_days(self):

        tds = self.market_opens.index
        days = tds[tds.slice_indexer(
            start=self.test_calendar_start + 1,
            end=self.test_calendar_start + 3
        )]
        minutes = DatetimeIndex([
            self.market_opens[days[0]] + timedelta(minutes=60),
            self.market_opens[days[1]] + timedelta(minutes=120),
        ])
        sid = 1
        data = DataFrame(
            data={
                'open': [10.0, 11.0],
                'high': [20.0, 21.0],
                'low': [30.0, 31.0],
                'close': [40.0, 41.0],
                'volume': [50.0, 51.0]
            },
            index=minutes)
        self.writer.write(sid, data)

        minute = minutes[0]

        open_price = self.reader.get_value(sid, minute, 'open')

        self.assertEquals(10.0, open_price)

        high_price = self.reader.get_value(sid, minute, 'high')

        self.assertEquals(20.0, high_price)

        low_price = self.reader.get_value(sid, minute, 'low')

        self.assertEquals(30.0, low_price)

        close_price = self.reader.get_value(sid, minute, 'close')

        self.assertEquals(40.0, close_price)

        volume_price = self.reader.get_value(sid, minute, 'volume')

        self.assertEquals(50.0, volume_price)

        minute = minutes[1]

        open_price = self.reader.get_value(sid, minute, 'open')

        self.assertEquals(11.0, open_price)

        high_price = self.reader.get_value(sid, minute, 'high')

        self.assertEquals(21.0, high_price)

        low_price = self.reader.get_value(sid, minute, 'low')

        self.assertEquals(31.0, low_price)

        close_price = self.reader.get_value(sid, minute, 'close')

        self.assertEquals(41.0, close_price)

        volume_price = self.reader.get_value(sid, minute, 'volume')

        self.assertEquals(51.0, volume_price)

    def test_no_overwrite(self):
        minute = self.market_opens[TEST_CALENDAR_START]
        sid = 1
        data = DataFrame(
            data={
                'open': [10.0],
                'high': [20.0],
                'low': [30.0],
                'close': [40.0],
                'volume': [50.0]
            },
            index=[minute])
        self.writer.write(sid, data)

        with self.assertRaises(BcolzMinuteOverlappingData):
            self.writer.write(sid, data)

    def test_write_multiple_sids(self):
        """
        Test writing multiple sids.

        Tests both that the data is written to the correct sid, as well as
        ensuring that the logic for creating the subdirectory path to each sid
        does not cause issues from attempts to recreate existing paths.
        (Calling out this coverage, because an assertion of that logic does not
        show up in the test itself, but is exercised by the act of attempting
        to write two consecutive sids, which would be written to the same
        containing directory, `00/00/000001.bcolz` and `00/00/000002.bcolz)

        Before applying a check to make sure the path writing did not
        re-attempt directory creation an OSError like the following would
        occur:

        ```
        OSError: [Errno 17] File exists: '/tmp/tmpR7yzzT/minute_bars/00/00'
        ```
        """
        minute = self.market_opens[TEST_CALENDAR_START]
        sids = [1, 2]
        data = DataFrame(
            data={
                'open': [15.0],
                'high': [17.0],
                'low': [11.0],
                'close': [15.0],
                'volume': [100.0]
            },
            index=[minute])
        self.writer.write(sids[0], data)

        data = DataFrame(
            data={
                'open': [25.0],
                'high': [27.0],
                'low': [21.0],
                'close': [25.0],
                'volume': [200.0]
            },
            index=[minute])
        self.writer.write(sids[1], data)

        sid = sids[0]

        open_price = self.reader.get_value(sid, minute, 'open')

        self.assertEquals(15.0, open_price)

        high_price = self.reader.get_value(sid, minute, 'high')

        self.assertEquals(17.0, high_price)

        low_price = self.reader.get_value(sid, minute, 'low')

        self.assertEquals(11.0, low_price)

        close_price = self.reader.get_value(sid, minute, 'close')

        self.assertEquals(15.0, close_price)

        volume_price = self.reader.get_value(sid, minute, 'volume')

        self.assertEquals(100.0, volume_price)

        sid = sids[1]

        open_price = self.reader.get_value(sid, minute, 'open')

        self.assertEquals(25.0, open_price)

        high_price = self.reader.get_value(sid, minute, 'high')

        self.assertEquals(27.0, high_price)

        low_price = self.reader.get_value(sid, minute, 'low')

        self.assertEquals(21.0, low_price)

        close_price = self.reader.get_value(sid, minute, 'close')

        self.assertEquals(25.0, close_price)

        volume_price = self.reader.get_value(sid, minute, 'volume')

        self.assertEquals(200.0, volume_price)

    def test_pad_data(self):
        """
        Test writing empty data.
        """
        sid = 1
        last_date = self.writer.last_date_in_output_for_sid(sid)
        self.assertIs(last_date, NaT)

        self.writer.pad(sid, TEST_CALENDAR_START)

        last_date = self.writer.last_date_in_output_for_sid(sid)
        self.assertEqual(last_date, TEST_CALENDAR_START)

        freq = self.market_opens.index.freq
        minute = self.market_opens[TEST_CALENDAR_START + freq]
        data = DataFrame(
            data={
                'open': [15.0],
                'high': [17.0],
                'low': [11.0],
                'close': [15.0],
                'volume': [100.0]
            },
            index=[minute])
        self.writer.write(sid, data)

        open_price = self.reader.get_value(sid, minute, 'open')

        self.assertEquals(15.0, open_price)

        high_price = self.reader.get_value(sid, minute, 'high')

        self.assertEquals(17.0, high_price)

        low_price = self.reader.get_value(sid, minute, 'low')

        self.assertEquals(11.0, low_price)

        close_price = self.reader.get_value(sid, minute, 'close')

        self.assertEquals(15.0, close_price)

        volume_price = self.reader.get_value(sid, minute, 'volume')

        self.assertEquals(100.0, volume_price)

    def test_nans(self):
        """
        Test writing empty data.
        """
        sid = 1
        last_date = self.writer.last_date_in_output_for_sid(sid)
        self.assertIs(last_date, NaT)

        self.writer.pad(sid, TEST_CALENDAR_START)

        last_date = self.writer.last_date_in_output_for_sid(sid)
        self.assertEqual(last_date, TEST_CALENDAR_START)

        freq = self.market_opens.index.freq
        minute = self.market_opens[TEST_CALENDAR_START + freq]
        minutes = date_range(minute, periods=9, freq='min')
        data = DataFrame(
            data={
                'open': full(9, nan),
                'high': full(9, nan),
                'low': full(9, nan),
                'close': full(9, nan),
                'volume': full(9, 0),
            },
            index=[minutes])
        self.writer.write(sid, data)

        fields = ['open', 'high', 'low', 'close', 'volume']

        ohlcv_window = self.reader.unadjusted_window(
            fields, minutes[0], minutes[-1], [sid])

        for i, field in enumerate(fields):
            if field != 'volume':
                assert_array_equal(full(9, nan), ohlcv_window[i][0])
            else:
                assert_array_equal(zeros(9), ohlcv_window[i][0])

    def test_differing_nans(self):
        """
        Also test nans of differing values/construction.
        """
        sid = 1
        last_date = self.writer.last_date_in_output_for_sid(sid)
        self.assertIs(last_date, NaT)

        self.writer.pad(sid, TEST_CALENDAR_START)

        last_date = self.writer.last_date_in_output_for_sid(sid)
        self.assertEqual(last_date, TEST_CALENDAR_START)

        freq = self.market_opens.index.freq
        minute = self.market_opens[TEST_CALENDAR_START + freq]
        minutes = date_range(minute, periods=9, freq='min')
        data = DataFrame(
            data={
                'open': ((0b11111111111 << 52) + arange(1, 10, dtype=int64)).
                view(float64),
                'high': ((0b11111111111 << 52) + arange(11, 20, dtype=int64)).
                view(float64),
                'low': ((0b11111111111 << 52) + arange(21, 30, dtype=int64)).
                view(float64),
                'close': ((0b11111111111 << 52) + arange(31, 40, dtype=int64)).
                view(float64),
                'volume': full(9, 0),
            },
            index=[minutes])
        self.writer.write(sid, data)

        fields = ['open', 'high', 'low', 'close', 'volume']

        ohlcv_window = self.reader.unadjusted_window(
            fields, minutes[0], minutes[-1], [sid])

        for i, field in enumerate(fields):
            if field != 'volume':
                assert_array_equal(full(9, nan), ohlcv_window[i][0])
            else:
                assert_array_equal(zeros(9), ohlcv_window[i][0])

    def test_write_cols(self):
        minute_0 = self.market_opens[self.test_calendar_start]
        minute_1 = minute_0 + timedelta(minutes=1)
        sid = 1
        cols = {
            'open': array([10.0, 11.0]),
            'high': array([20.0, 21.0]),
            'low': array([30.0, 31.0]),
            'close': array([40.0, 41.0]),
            'volume': array([50.0, 51.0])
        }
        dts = array([minute_0, minute_1], dtype='datetime64[s]')
        self.writer.write_cols(sid, dts, cols)

        open_price = self.reader.get_value(sid, minute_0, 'open')

        self.assertEquals(10.0, open_price)

        high_price = self.reader.get_value(sid, minute_0, 'high')

        self.assertEquals(20.0, high_price)

        low_price = self.reader.get_value(sid, minute_0, 'low')

        self.assertEquals(30.0, low_price)

        close_price = self.reader.get_value(sid, minute_0, 'close')

        self.assertEquals(40.0, close_price)

        volume_price = self.reader.get_value(sid, minute_0, 'volume')

        self.assertEquals(50.0, volume_price)

        open_price = self.reader.get_value(sid, minute_1, 'open')

        self.assertEquals(11.0, open_price)

        high_price = self.reader.get_value(sid, minute_1, 'high')

        self.assertEquals(21.0, high_price)

        low_price = self.reader.get_value(sid, minute_1, 'low')

        self.assertEquals(31.0, low_price)

        close_price = self.reader.get_value(sid, minute_1, 'close')

        self.assertEquals(41.0, close_price)

        volume_price = self.reader.get_value(sid, minute_1, 'volume')

        self.assertEquals(51.0, volume_price)

    def test_unadjusted_minutes(self):
        """
        Test unadjusted minutes.
        """
        start_minute = self.market_opens[TEST_CALENDAR_START]
        minutes = [start_minute,
                   start_minute + Timedelta('1 min'),
                   start_minute + Timedelta('2 min')]
        sids = [1, 2]
        data_1 = DataFrame(
            data={
                'open': [15.0, nan, 15.1],
                'high': [17.0, nan, 17.1],
                'low': [11.0, nan, 11.1],
                'close': [14.0, nan, 14.1],
                'volume': [1000, 0, 1001]
            },
            index=minutes)
        self.writer.write(sids[0], data_1)

        data_2 = DataFrame(
            data={
                'open': [25.0, nan, 25.1],
                'high': [27.0, nan, 27.1],
                'low': [21.0, nan, 21.1],
                'close': [24.0, nan, 24.1],
                'volume': [2000, 0, 2001]
            },
            index=minutes)
        self.writer.write(sids[1], data_2)

        reader = BcolzMinuteBarReader(self.dest)

        columns = ['open', 'high', 'low', 'close', 'volume']
        sids = [sids[0], sids[1]]
        arrays = reader.unadjusted_window(
            columns, minutes[0], minutes[-1], sids)

        data = {sids[0]: data_1, sids[1]: data_2}

        for i, col in enumerate(columns):
            for j, sid in enumerate(sids):
                assert_almost_equal(data[sid][col], arrays[i][j])
