from textwrap import dedent
from unittest import TestCase

from numbers import Real

import pandas as pd
import numpy as np
from numpy import nan
from numpy.testing import assert_almost_equal

from nose_parameterized import parameterized
from testfixtures import TempDirectory

from zipline import TradingAlgorithm
from zipline._protocol import handle_non_market_minutes
from zipline.assets import Asset
from zipline.data.data_portal import DataPortal, DailyHistoryAggregator
from zipline.data.minute_bars import (
    BcolzMinuteBarReader,
    BcolzMinuteBarWriter,
    US_EQUITIES_MINUTES_PER_DAY
)
from zipline.data.us_equity_pricing import (
    SQLiteAdjustmentWriter,
    SQLiteAdjustmentReader,
    BcolzDailyBarReader)
from zipline.errors import (
    HistoryInInitialize,
    HistoryWindowStartsBeforeData,
)
from zipline.finance.trading import (
    TradingEnvironment,
    SimulationParameters
)
from zipline.protocol import BarData
from zipline.testing import str_to_seconds
from zipline.testing.core import (
    write_minute_data_for_asset,
    DailyBarWriterFromDataFrames,
    MockDailyBarReader
)
from zipline.testing.fixtures import (
    WithBcolzMinutes,
    ZiplineTestCase
)


OHLC = ["open", "high", "low", "close"]
OHLCV = OHLC + ["volume"]
OHLCP = OHLC + ["price"]
ALL_FIELDS = OHLCP + ["volume"]


class HistoryTestCaseBase(TestCase):
    # asset1:
    # - 2014-03-01 (rounds up to TRADING_START_DT) to 2016-01-29.
    # - every minute/day.

    # asset2:
    # - 2015-01-05 to 2015-12-31
    # - every minute/day.

    # asset3:
    # - 2015-01-05 to 2015-12-31
    # - trades every 10 minutes

    # SPLIT_ASSET:
    # - 2015-01-04 to 2015-12-31
    # - trades every minute
    # - splits on 2015-01-05 and 2015-01-06

    # DIVIDEND_ASSET:
    # - 2015-01-04 to 2015-12-31
    # - trades every minute
    # - dividends on 2015-01-05 and 2015-01-06

    # MERGER_ASSET
    # - 2015-01-04 to 2015-12-31
    # - trades every minute
    # - merger on 2015-01-05 and 2015-01-06
    @classmethod
    def setUpClass(cls):
        cls.tempdir = TempDirectory()

        # trading_start (when bcolz files start) = 2014-02-01
        cls.TRADING_START_DT = pd.Timestamp("2014-02-03", tz='UTC')
        cls.TRADING_END_DT = pd.Timestamp("2016-01-29", tz='UTC')

        cls.env = TradingEnvironment(min_date=cls.TRADING_START_DT)

        cls.trading_days = cls.env.days_in_range(
            start=cls.TRADING_START_DT,
            end=cls.TRADING_END_DT
        )

        cls.create_assets()

        cls.ASSET1 = cls.env.asset_finder.retrieve_asset(1)
        cls.ASSET2 = cls.env.asset_finder.retrieve_asset(2)
        cls.ASSET3 = cls.env.asset_finder.retrieve_asset(3)
        cls.SPLIT_ASSET = cls.env.asset_finder.retrieve_asset(4)
        cls.DIVIDEND_ASSET = cls.env.asset_finder.retrieve_asset(5)
        cls.MERGER_ASSET = cls.env.asset_finder.retrieve_asset(6)
        cls.HALF_DAY_TEST_ASSET = cls.env.asset_finder.retrieve_asset(7)
        cls.SHORT_ASSET = cls.env.asset_finder.retrieve_asset(8)

        cls.adj_reader = cls.create_adjustments_reader()

        cls.create_data()

    def setUp(self):
        self.create_data_portal()

    @classmethod
    def create_data_portal(cls):
        raise NotImplementedError()

    @classmethod
    def create_data(cls):
        raise NotImplementedError()

    @classmethod
    def create_assets(cls):
        jan_5_2015 = pd.Timestamp("2015-01-05", tz='UTC')
        day_after_12312015 = cls.env.next_trading_day(
            pd.Timestamp("2015-12-31", tz='UTC')
        )

        cls.env.write_data(equities_data={
            1: {
                "start_date": pd.Timestamp("2014-01-03", tz='UTC'),
                "end_date": cls.TRADING_END_DT,
                "symbol": "ASSET1"
            },
            2: {
                "start_date": jan_5_2015,
                "end_date": day_after_12312015,
                "symbol": "ASSET2"
            },
            3: {
                "start_date": jan_5_2015,
                "end_date": day_after_12312015,
                "symbol": "ASSET3"
            },
            4: {
                "start_date": jan_5_2015,
                "end_date": day_after_12312015,
                "symbol": "SPLIT_ASSET"
            },
            5: {
                "start_date": jan_5_2015,
                "end_date": day_after_12312015,
                "symbol": "DIVIDEND_ASSET"
            },
            6: {
                "start_date": jan_5_2015,
                "end_date": day_after_12312015,
                "symbol": "MERGER_ASSET"
            },
            7: {
                "start_date": pd.Timestamp("2014-07-02", tz='UTC'),
                "end_date": day_after_12312015,
                "symbol": "HALF_DAY_TEST_ASSET"
            },
            8: {
                "start_date": pd.Timestamp("2015-01-05", tz='UTC'),
                "end_date": pd.Timestamp("2015-01-06", tz='UTC'),
                "symbol": "SHORT_ASSET"
            }
        })

    @classmethod
    def tearDownClass(cls):
        cls.tempdir.cleanup()

    @classmethod
    def create_adjustments_reader(cls):
        path = cls.tempdir.getpath("test_adjustments.db")

        adj_writer = SQLiteAdjustmentWriter(
            path,
            cls.env.trading_days,
            MockDailyBarReader()
        )

        splits = pd.DataFrame([
            {
                'effective_date': str_to_seconds("2015-01-06"),
                'ratio': 0.5,
                'sid': cls.SPLIT_ASSET.sid
            },
            {
                'effective_date': str_to_seconds("2015-01-07"),
                'ratio': 0.5,
                'sid': cls.SPLIT_ASSET.sid
            },
        ])

        mergers = pd.DataFrame([
            {
                'effective_date': str_to_seconds("2015-01-06"),
                'ratio': 0.5,
                'sid': cls.MERGER_ASSET.sid
            },
            {
                'effective_date': str_to_seconds("2015-01-07"),
                'ratio': 0.5,
                'sid': cls.MERGER_ASSET.sid
            }
        ])

        # we're using a fake daily reader in the adjustments writer which
        # returns every daily price as 100, so dividend amounts of 2.0 and 4.0
        # correspond to 2% and 4% dividends, respectively.
        dividends = pd.DataFrame([
            {
                # only care about ex date, the other dates don't matter here
                'ex_date':
                    pd.Timestamp("2015-01-06", tz='UTC').to_datetime64(),
                'record_date':
                    pd.Timestamp("2015-01-06", tz='UTC').to_datetime64(),
                'declared_date':
                    pd.Timestamp("2015-01-06", tz='UTC').to_datetime64(),
                'pay_date':
                    pd.Timestamp("2015-01-06", tz='UTC').to_datetime64(),
                'amount': 2.0,
                'sid': cls.DIVIDEND_ASSET.sid
            },
            {
                'ex_date':
                    pd.Timestamp("2015-01-07", tz='UTC').to_datetime64(),
                'record_date':
                    pd.Timestamp("2015-01-07", tz='UTC').to_datetime64(),
                'declared_date':
                    pd.Timestamp("2015-01-07", tz='UTC').to_datetime64(),
                'pay_date':
                    pd.Timestamp("2015-01-07", tz='UTC').to_datetime64(),
                'amount': 4.0,
                'sid': cls.DIVIDEND_ASSET.sid
            }],
            columns=['ex_date',
                     'record_date',
                     'declared_date',
                     'pay_date',
                     'amount',
                     'sid']
        )

        adj_writer.write(splits, mergers, dividends)

        return SQLiteAdjustmentReader(path)

    def verify_regular_dt(self, idx, dt, mode, fields=None, assets=None):
        if mode == "daily":
            freq = "1d"
        else:
            freq = "1m"

        fields = fields if fields is not None else ALL_FIELDS
        assets = assets if assets is not None else [self.ASSET2, self.ASSET3]

        bar_data = BarData(self.data_portal, lambda: dt, mode)
        check_internal_consistency(
            bar_data, assets, fields, 10, freq
        )

        for field in fields:
            for asset in assets:
                asset_series = bar_data.history(asset, field, 10, freq)

                base = MINUTE_FIELD_INFO[field] + 2

                if idx < 9:
                    missing_count = 9 - idx
                    present_count = 9 - missing_count

                    if field in OHLCP:
                        if asset == self.ASSET2:
                            # asset2 should have some leading nans
                            np.testing.assert_array_equal(
                                np.full(missing_count, np.nan),
                                asset_series[0:missing_count]
                            )

                            # asset2 should also have some real values
                            np.testing.assert_array_equal(
                                np.array(range(base,
                                               base + present_count + 1)),
                                asset_series[(9 - present_count):]
                            )

                        if asset == self.ASSET3:
                            # asset3 should be NaN the entire time
                            np.testing.assert_array_equal(
                                np.full(10, np.nan),
                                asset_series
                            )
                    elif field == "volume":
                        if asset == self.ASSET2:
                            # asset2 should have some zeros (instead of nans)
                            np.testing.assert_array_equal(
                                np.zeros(missing_count),
                                asset_series[0:missing_count]
                            )

                            # and some real values
                            np.testing.assert_array_equal(
                                np.array(
                                    range(base, base + present_count + 1)
                                ) * 100,
                                asset_series[(9 - present_count):]
                            )

                        if asset == self.ASSET3:
                            # asset3 is all zeros, no volume yet
                            np.testing.assert_array_equal(
                                np.zeros(10),
                                asset_series
                            )
                else:
                    # asset3 should have data every 10 minutes
                    # construct an array full of nans, put something in the
                    # right slot, and test for comparison

                    position_from_end = ((idx + 1) % 10) + 1

                    # asset3's baseline data is 9 NaNs, then 11, then 9 NaNs,
                    # then 21, etc.  for idx 9 to 19, value_for_asset3 should
                    # be a baseline of 11 (then adjusted for the individual
                    # field), thus the rounding down to the nearest 10.
                    value_for_asset3 = (((idx + 1) // 10) * 10) + \
                        MINUTE_FIELD_INFO[field] + 1

                    if field in OHLC:
                        asset3_answer_key = np.full(10, np.nan)
                        asset3_answer_key[-position_from_end] = \
                            value_for_asset3

                        if asset == self.ASSET2:
                            np.testing.assert_array_equal(
                                np.array(
                                    range(base + idx - 9, base + idx + 1)),
                                asset_series
                            )

                        if asset == self.ASSET3:
                            np.testing.assert_array_equal(
                                asset3_answer_key,
                                asset_series
                            )
                    elif field == "volume":
                        asset3_answer_key = np.zeros(10)
                        asset3_answer_key[-position_from_end] = \
                            value_for_asset3 * 100

                        if asset == self.ASSET2:
                            np.testing.assert_array_equal(
                                np.array(
                                    range(base + idx - 9, base + idx + 1)
                                ) * 100,
                                asset_series
                            )

                        if asset == self.ASSET3:
                            np.testing.assert_array_equal(
                                asset3_answer_key,
                                asset_series
                            )
                    elif field == "price":
                        # price is always forward filled

                        # asset2 has prices every minute, so it's easy

                        if asset == self.ASSET2:
                            # at idx 9, the data is 2 to 11
                            np.testing.assert_array_equal(
                                range(idx - 7, idx + 3),
                                asset_series
                            )

                        if asset == self.ASSET3:
                            first_part = asset_series[0:-position_from_end]
                            second_part = asset_series[-position_from_end:]

                            decile_count = ((idx + 1) // 10)

                            # in our test data, asset3 prices will be nine
                            # NaNs, then ten 11s, ten 21s, ten 31s...

                            if decile_count == 1:
                                np.testing.assert_array_equal(
                                    np.full(len(first_part), np.nan),
                                    first_part
                                )

                                np.testing.assert_array_equal(
                                    np.array([11] * len(second_part)),
                                    second_part
                                )
                            else:
                                np.testing.assert_array_equal(
                                    np.array([decile_count * 10 - 9] *
                                             len(first_part)),
                                    first_part
                                )

                                np.testing.assert_array_equal(
                                    np.array([decile_count * 10 + 1] *
                                             len(second_part)),
                                    second_part
                                )


def check_internal_consistency(bar_data, assets, fields, bar_count, freq):
    if isinstance(assets, Asset):
        asset_list = [assets]
    else:
        asset_list = assets

    if isinstance(fields, str):
        field_list = [fields]
    else:
        field_list = fields

    multi_field_dict = {
        asset: bar_data.history(asset, field_list, bar_count, freq)
        for asset in asset_list
    }

    multi_asset_dict = {
        field: bar_data.history(asset_list, field, bar_count, freq)
        for field in fields
    }

    panel = bar_data.history(asset_list, field_list, bar_count, freq)

    for field in field_list:
        # make sure all the different query forms are internally
        # consistent
        for asset in asset_list:
            series = bar_data.history(asset, field, bar_count, freq)

            np.testing.assert_array_equal(
                series,
                multi_asset_dict[field][asset]
            )

            np.testing.assert_array_equal(
                series,
                multi_field_dict[asset][field]
            )

            np.testing.assert_array_equal(
                series,
                panel[field][asset]
            )


# each minute's OHLCV data has a consistent offset for each field.
# for example, the open is always 1 higher than the close, the high
# is always 2 higher than the close, etc.
MINUTE_FIELD_INFO = {
    "open": 1,
    "high": 2,
    "low": -1,
    "close": 0,
    "price": 0,
    "volume": 0,      # unused, later we'll multiply by 100
}


class MinuteEquityHistoryTestCase(HistoryTestCaseBase):
    @classmethod
    def create_data_portal(cls):
        cls.data_portal = DataPortal(
            cls.env,
            equity_minute_reader=BcolzMinuteBarReader(cls.tempdir.path),
            adjustment_reader=cls.adj_reader
        )

    @classmethod
    def create_data(cls):
        market_opens = cls.env.open_and_closes.market_open.loc[
            cls.trading_days]
        market_closes = cls.env.open_and_closes.market_close.loc[
            cls.trading_days]

        writer = BcolzMinuteBarWriter(
            cls.trading_days[0],
            cls.tempdir.path,
            market_opens,
            market_closes,
            US_EQUITIES_MINUTES_PER_DAY
        )

        write_minute_data_for_asset(
            cls.env,
            writer,
            pd.Timestamp("2014-01-03", tz='UTC'),
            pd.Timestamp("2016-01-30", tz='UTC'),
            1,
            start_val=2
        )

        for sid in [2, 4, 5, 6, cls.SHORT_ASSET.sid]:
            asset = cls.env.asset_finder.retrieve_asset(sid)
            write_minute_data_for_asset(
                cls.env,
                writer,
                asset.start_date,
                asset.end_date,
                asset.sid,
                start_val=2
            )

        write_minute_data_for_asset(
            cls.env,
            writer,
            cls.HALF_DAY_TEST_ASSET.start_date,
            cls.HALF_DAY_TEST_ASSET.end_date,
            cls.HALF_DAY_TEST_ASSET.sid,
            start_val=2
        )

        asset3 = cls.env.asset_finder.retrieve_asset(3)
        write_minute_data_for_asset(
            cls.env,
            writer,
            asset3.start_date,
            asset3.end_date,
            asset3.sid,
            interval=10,
            start_val=2
        )

    def test_history_in_initialize(self):
        algo_text = dedent(
            """\
            from zipline.api import history

            def initialize(context):
                history([1], 10, '1d', 'price')

            def handle_data(context, data):
                pass
            """
        )

        start = pd.Timestamp('2014-04-05', tz='UTC')
        end = pd.Timestamp('2014-04-10', tz='UTC')

        sim_params = SimulationParameters(
            period_start=start,
            period_end=end,
            capital_base=float("1.0e5"),
            data_frequency='minute',
            emission_rate='daily',
            env=self.env,
        )

        test_algo = TradingAlgorithm(
            script=algo_text,
            data_frequency='minute',
            sim_params=sim_params,
            env=self.env,
        )

        with self.assertRaises(HistoryInInitialize):
            test_algo.initialize()

    def test_minute_before_assets_trading(self):
        # since asset2 and asset3 both started trading on 1/5/2015, let's do
        # some history windows that are completely before that
        minutes = self.env.market_minutes_for_day(
            self.env.previous_trading_day(pd.Timestamp("2015-01-05", tz='UTC'))
        )[0:60]

        for idx, minute in enumerate(minutes):
            bar_data = BarData(self.data_portal, lambda: minute, "minute")
            check_internal_consistency(
                bar_data, [self.ASSET2, self.ASSET3], ALL_FIELDS, 10, "1m"
            )

            for field in ALL_FIELDS:
                # OHLCP should be NaN
                # Volume should be 0
                asset2_series = bar_data.history(self.ASSET2, field, 10, "1m")
                asset3_series = bar_data.history(self.ASSET3, field, 10, "1m")

                if field == "volume":
                    np.testing.assert_array_equal(np.zeros(10), asset2_series)
                    np.testing.assert_array_equal(np.zeros(10), asset3_series)
                else:
                    np.testing.assert_array_equal(
                        np.full(10, np.nan),
                        asset2_series
                    )

                    np.testing.assert_array_equal(
                        np.full(10, np.nan),
                        asset3_series
                    )

    @parameterized.expand([
        ('open_sid_2', 'open', 2),
        ('high_sid_2', 'high', 2),
        ('low_sid_2', 'low', 2),
        ('close_sid_2', 'close', 2),
        ('volume_sid_2', 'volume', 2),
        ('open_sid_3', 'open', 3),
        ('high_sid_3', 'high', 3),
        ('low_sid_3', 'low', 3),
        ('close_sid_3', 'close', 3),
        ('volume_sid_3', 'volume', 3),

    ])
    def test_minute_regular(self, name, field, sid):
        # asset2 and asset3 both started on 1/5/2015, but asset3 trades every
        # 10 minutes
        asset = self.env.asset_finder.retrieve_asset(sid)

        minutes = self.env.market_minutes_for_day(
            pd.Timestamp("2015-01-05", tz='UTC')
        )[0:60]

        for idx, minute in enumerate(minutes):
            self.verify_regular_dt(idx, minute, "minute",
                                   assets=[asset],
                                   fields=[field])

    def test_minute_midnight(self):
        midnight = pd.Timestamp("2015-01-06", tz='UTC')
        last_minute = self.env.previous_open_and_close(midnight)[1]

        midnight_bar_data = \
            BarData(self.data_portal, lambda: midnight, "minute")

        yesterday_bar_data = \
            BarData(self.data_portal, lambda: last_minute, "minute")

        with handle_non_market_minutes(midnight_bar_data):
            for field in ALL_FIELDS:
                np.testing.assert_array_equal(
                    midnight_bar_data.history(self.ASSET2, field, 30, "1m"),
                    yesterday_bar_data.history(self.ASSET2, field, 30, "1m")
                )

    def test_minute_after_asset_stopped(self):
        # SHORT_ASSET's last day was 2015-01-06
        # get some history windows that straddle the end
        minutes = self.env.market_minutes_for_day(
            pd.Timestamp("2015-01-07", tz='UTC')
        )[0:60]

        for idx, minute in enumerate(minutes):
            bar_data = BarData(self.data_portal, lambda: minute, "minute")
            check_internal_consistency(
                bar_data, self.SHORT_ASSET, ALL_FIELDS, 30, "1m"
            )

        # Reset data portal because it has advanced past next test date.
        self.create_data_portal()

        # choose a window that contains the last minute of the asset
        bar_data = BarData(self.data_portal, lambda: minutes[15], "minute")

        #                             close  high  low  open  price  volume
        # 2015-01-06 20:47:00+00:00    768   770  767   769    768   76800
        # 2015-01-06 20:48:00+00:00    769   771  768   770    769   76900
        # 2015-01-06 20:49:00+00:00    770   772  769   771    770   77000
        # 2015-01-06 20:50:00+00:00    771   773  770   772    771   77100
        # 2015-01-06 20:51:00+00:00    772   774  771   773    772   77200
        # 2015-01-06 20:52:00+00:00    773   775  772   774    773   77300
        # 2015-01-06 20:53:00+00:00    774   776  773   775    774   77400
        # 2015-01-06 20:54:00+00:00    775   777  774   776    775   77500
        # 2015-01-06 20:55:00+00:00    776   778  775   777    776   77600
        # 2015-01-06 20:56:00+00:00    777   779  776   778    777   77700
        # 2015-01-06 20:57:00+00:00    778   780  777   779    778   77800
        # 2015-01-06 20:58:00+00:00    779   781  778   780    779   77900
        # 2015-01-06 20:59:00+00:00    780   782  779   781    780   78000
        # 2015-01-06 21:00:00+00:00    781   783  780   782    781   78100
        # 2015-01-07 14:31:00+00:00    NaN   NaN  NaN   NaN    NaN       0
        # 2015-01-07 14:32:00+00:00    NaN   NaN  NaN   NaN    NaN       0
        # 2015-01-07 14:33:00+00:00    NaN   NaN  NaN   NaN    NaN       0
        # 2015-01-07 14:34:00+00:00    NaN   NaN  NaN   NaN    NaN       0
        # 2015-01-07 14:35:00+00:00    NaN   NaN  NaN   NaN    NaN       0
        # 2015-01-07 14:36:00+00:00    NaN   NaN  NaN   NaN    NaN       0
        # 2015-01-07 14:37:00+00:00    NaN   NaN  NaN   NaN    NaN       0
        # 2015-01-07 14:38:00+00:00    NaN   NaN  NaN   NaN    NaN       0
        # 2015-01-07 14:39:00+00:00    NaN   NaN  NaN   NaN    NaN       0
        # 2015-01-07 14:40:00+00:00    NaN   NaN  NaN   NaN    NaN       0
        # 2015-01-07 14:41:00+00:00    NaN   NaN  NaN   NaN    NaN       0
        # 2015-01-07 14:42:00+00:00    NaN   NaN  NaN   NaN    NaN       0
        # 2015-01-07 14:43:00+00:00    NaN   NaN  NaN   NaN    NaN       0
        # 2015-01-07 14:44:00+00:00    NaN   NaN  NaN   NaN    NaN       0
        # 2015-01-07 14:45:00+00:00    NaN   NaN  NaN   NaN    NaN       0
        # 2015-01-07 14:46:00+00:00    NaN   NaN  NaN   NaN    NaN       0

        window = bar_data.history(self.SHORT_ASSET, ALL_FIELDS, 30, "1m")

        # there should be 14 values and 16 NaNs/0s
        for field in ALL_FIELDS:
            if field == "volume":
                np.testing.assert_array_equal(
                    range(76800, 78101, 100),
                    window["volume"][0:14]
                )
                np.testing.assert_array_equal(
                    np.zeros(16),
                    window["volume"][-16:]
                )
            else:
                np.testing.assert_array_equal(
                    np.array(range(768, 782)) + MINUTE_FIELD_INFO[field],
                    window[field][0:14]
                )
                np.testing.assert_array_equal(
                    np.full(16, np.nan),
                    window[field][-16:]
                )

        # now do a smaller window that is entirely contained after the asset
        # ends
        window = bar_data.history(self.SHORT_ASSET, ALL_FIELDS, 5, "1m")

        for field in ALL_FIELDS:
            if field == "volume":
                np.testing.assert_array_equal(np.zeros(5), window["volume"])
            else:
                np.testing.assert_array_equal(np.full(5, np.nan),
                                              window[field])

    def test_minute_splits_and_mergers(self):
        # self.SPLIT_ASSET and self.MERGER_ASSET had splits/mergers
        # on 1/6 and 1/7

        jan5 = pd.Timestamp("2015-01-05", tz='UTC')

        # the assets' close column starts at 2 on the first minute of
        # 1/5, then goes up one per minute forever

        for asset in [self.SPLIT_ASSET, self.MERGER_ASSET]:
            # before any of the adjustments, last 10 minutes of jan 5
            window1 = self.data_portal.get_history_window(
                [asset],
                self.env.get_open_and_close(jan5)[1],
                10,
                "1m",
                "close"
            )[asset]

            np.testing.assert_array_equal(np.array(range(382, 392)), window1)

            # straddling the first event
            window2 = self.data_portal.get_history_window(
                [asset],
                pd.Timestamp("2015-01-06 14:35", tz='UTC'),
                10,
                "1m",
                "close"
            )[asset]

            # five minutes from 1/5 should be halved
            np.testing.assert_array_equal(
                [193.5, 194, 194.5, 195, 195.5, 392, 393, 394, 395, 396],
                window2
            )

            # straddling both events!
            window3 = self.data_portal.get_history_window(
                [asset],
                pd.Timestamp("2015-01-07 14:35", tz='UTC'),
                400,    # 5 minutes of 1/7, 390 of 1/6, and 5 minutes of 1/5
                "1m",
                "close"
            )[asset]

            # first five minutes should be 387-391, but quartered
            np.testing.assert_array_equal(
                [96.75, 97, 97.25, 97.5, 97.75],
                window3[0:5]
            )

            # next 390 minutes should be 392-781, but halved
            np.testing.assert_array_equal(
                np.array(range(392, 782), dtype="float64") / 2,
                window3[5:395]
            )

            # final 5 minutes should be 782-787
            np.testing.assert_array_equal(range(782, 787), window3[395:])

            # after last event
            window4 = self.data_portal.get_history_window(
                [asset],
                pd.Timestamp("2015-01-07 14:40", tz='UTC'),
                5,
                "1m",
                "close"
            )[asset]

            # should not be adjusted, should be 787 to 791
            np.testing.assert_array_equal(range(787, 792), window4)

    def test_minute_dividends(self):
        # self.DIVIDEND_ASSET had dividends on 1/6 and 1/7

        # before any of the dividends
        window1 = self.data_portal.get_history_window(
            [self.DIVIDEND_ASSET],
            pd.Timestamp("2015-01-05 21:00", tz='UTC'),
            10,
            "1m",
            "close"
        )[self.DIVIDEND_ASSET]

        np.testing.assert_array_equal(np.array(range(382, 392)), window1)

        # straddling the first dividend
        window2 = self.data_portal.get_history_window(
            [self.DIVIDEND_ASSET],
            pd.Timestamp("2015-01-06 14:35", tz='UTC'),
            10,
            "1m",
            "close"
        )[self.DIVIDEND_ASSET]

        # first dividend is 2%, so the first five values should be 2% lower
        # than before
        np.testing.assert_array_almost_equal(
            np.array(range(387, 392), dtype="float64") * 0.98,
            window2[0:5]
        )

        # second half of window is unadjusted
        np.testing.assert_array_equal(range(392, 397), window2[5:])

        # straddling both dividends
        window3 = self.data_portal.get_history_window(
            [self.DIVIDEND_ASSET],
            pd.Timestamp("2015-01-07 14:35", tz='UTC'),
            400,    # 5 minutes of 1/7, 390 of 1/6, and 5 minutes of 1/5
            "1m",
            "close"
        )[self.DIVIDEND_ASSET]

        # first five minute from 1/7 should be hit by 0.9408 (= 0.98 * 0.96)
        np.testing.assert_array_almost_equal(
            np.around(np.array(range(387, 392), dtype="float64") * 0.9408, 3),
            window3[0:5]
        )

        # next 390 minutes should be hit by 0.96 (second dividend)
        np.testing.assert_array_almost_equal(
            np.array(range(392, 782), dtype="float64") * 0.96,
            window3[5:395]
        )

        # last 5 minutes should not be adjusted
        np.testing.assert_array_equal(np.array(range(782, 787)), window3[395:])

    def test_passing_iterable_to_history_regular_hours(self):
        # regular hours
        current_dt = pd.Timestamp("2015-01-06 9:45", tz='US/Eastern')
        bar_data = BarData(self.data_portal, lambda: current_dt, "minute")

        bar_data.history(pd.Index([self.ASSET1, self.ASSET2]),
                         "high", 5, "1m")

    def test_passing_iterable_to_history_bts(self):
        # before market hours
        current_dt = pd.Timestamp("2015-01-07 8:45", tz='US/Eastern')
        bar_data = BarData(self.data_portal, lambda: current_dt, "minute")

        with handle_non_market_minutes(bar_data):
            bar_data.history(pd.Index([self.ASSET1, self.ASSET2]),
                             "high", 5, "1m")

    def test_overnight_adjustments(self):
        # Should incorporate adjustments on midnight 01/06
        current_dt = pd.Timestamp("2015-01-06 8:45", tz='US/Eastern')
        bar_data = BarData(self.data_portal, lambda: current_dt, "minute")

        expected = {
            'open': np.arange(383, 393) / 2.0,
            'high': np.arange(384, 394) / 2.0,
            'low': np.arange(381, 391) / 2.0,
            'close': np.arange(382, 392) / 2.0,
            'volume': np.arange(382, 392) * 100 * 2.0,
            'price': np.arange(382, 392) / 2.0,
        }

        with handle_non_market_minutes(bar_data):
            # Single field, single asset
            for field in ALL_FIELDS:
                values = bar_data.history(self.SPLIT_ASSET, field, 10, '1m')
                np.testing.assert_array_equal(values.values, expected[field])

            # Multi field, single asset
            values = bar_data.history(
                self.SPLIT_ASSET, ['open', 'volume'], 10, '1m'
            )
            np.testing.assert_array_equal(values.open.values,
                                          expected['open'])
            np.testing.assert_array_equal(values.volume.values,
                                          expected['volume'])

            # Single field, multi asset
            values = bar_data.history(
                [self.SPLIT_ASSET, self.ASSET2], 'open', 10, '1m'
            )
            np.testing.assert_array_equal(values[self.SPLIT_ASSET].values,
                                          expected['open'])
            np.testing.assert_array_equal(values[self.ASSET2].values,
                                          expected['open'] * 2)

            # Multi field, multi asset
            values = bar_data.history(
                [self.SPLIT_ASSET, self.ASSET2], ['open', 'volume'], 10, '1m'
            )
            np.testing.assert_array_equal(
                values.open[self.SPLIT_ASSET].values,
                expected['open']
            )
            np.testing.assert_array_equal(
                values.volume[self.SPLIT_ASSET].values,
                expected['volume']
            )
            np.testing.assert_array_equal(
                values.open[self.ASSET2].values,
                expected['open'] * 2
            )
            np.testing.assert_array_equal(
                values.volume[self.ASSET2].values,
                expected['volume'] / 2
            )

    def test_minute_early_close(self):
        # 2014-07-03 is an early close
        # HALF_DAY_TEST_ASSET started trading on 2014-07-02, how convenient
        #
        # five minutes into the day after the early close, get 20 1m bars

        dt = pd.Timestamp("2014-07-07 13:35:00", tz='UTC')

        window = self.data_portal.get_history_window(
            [self.HALF_DAY_TEST_ASSET],
            dt,
            20,
            "1m",
            "close"
        )[self.HALF_DAY_TEST_ASSET]

        # 390 minutes for 7/2, 210 minutes for 7/3, 7/4-7/6 closed
        # first minute of 7/7 is the 600th trading minute for this asset
        # this asset's first minute had a close value of 2, so every value is
        # 2 + (minute index)
        np.testing.assert_array_equal(range(587, 607), window)

        self.assertEqual(
            window.index[-6],
            pd.Timestamp("2014-07-03 17:00", tz='UTC')
        )

        self.assertEqual(
            window.index[-5],
            pd.Timestamp("2014-07-07 13:31", tz='UTC')
        )

    def test_minute_different_lifetimes(self):
        # at trading start, only asset1 existed
        day = self.env.next_trading_day(self.TRADING_START_DT)

        asset1_minutes = self.env.minutes_for_days_in_range(
            start=self.ASSET1.start_date,
            end=self.ASSET1.end_date
        )

        asset1_idx = asset1_minutes.searchsorted(
            self.env.get_open_and_close(day)[0]
        )

        window = self.data_portal.get_history_window(
            [self.ASSET1, self.ASSET2],
            self.env.get_open_and_close(day)[0],
            100,
            "1m",
            "close"
        )

        np.testing.assert_array_equal(
            range(asset1_idx - 97, asset1_idx + 3),
            window[self.ASSET1]
        )

        np.testing.assert_array_equal(
            np.full(100, np.nan), window[self.ASSET2]
        )

    def test_history_window_before_first_trading_day(self):
        # trading_start is 2/3/2014
        # get a history window that starts before that, and ends after that
        first_day_minutes = self.env.market_minutes_for_day(
            self.TRADING_START_DT
        )
        exp_msg = (
            "History window extends before 2014-02-03. To use this history "
            "window, start the backtest on or after 2014-02-04."
        )
        for field in OHLCP:
            with self.assertRaisesRegexp(
                    HistoryWindowStartsBeforeData, exp_msg):
                self.data_portal.get_history_window(
                    [self.ASSET1], first_day_minutes[5], 15, "1m", "price"
                )[self.ASSET1]


class DailyEquityHistoryTestCase(HistoryTestCaseBase):
    @classmethod
    def create_data_portal(cls):
        daily_path = cls.tempdir.getpath("testdaily.bcolz")

        cls.data_portal = DataPortal(
            cls.env,
            equity_daily_reader=BcolzDailyBarReader(daily_path),
            equity_minute_reader=BcolzMinuteBarReader(cls.tempdir.path),
            adjustment_reader=cls.adj_reader
        )

    @classmethod
    def create_data(cls):
        path = cls.tempdir.getpath("testdaily.bcolz")

        dfs = {
            1: cls.create_df_for_asset(
                cls.TRADING_START_DT,
                pd.Timestamp("2016-01-30", tz='UTC')
            ),
            3: cls.create_df_for_asset(
                pd.Timestamp("2015-01-05", tz='UTC'),
                pd.Timestamp("2015-12-31", tz='UTC'),
                interval=10,
                force_zeroes=True
            ),
            cls.SHORT_ASSET.sid: cls.create_df_for_asset(
                pd.Timestamp("2015-01-05", tz='UTC'),
                pd.Timestamp("2015-01-06", tz='UTC'),
            )
        }

        for sid in [2, 4, 5, 6]:
            asset = cls.env.asset_finder.retrieve_asset(sid)
            dfs[sid] = cls.create_df_for_asset(
                asset.start_date,
                asset.end_date
            )

        days = cls.env.days_in_range(
            cls.TRADING_START_DT,
            cls.TRADING_END_DT
        )

        daily_writer = DailyBarWriterFromDataFrames(dfs)
        daily_writer.write(path, days, dfs)

        market_opens = cls.env.open_and_closes.market_open.loc[
            cls.trading_days]
        market_closes = cls.env.open_and_closes.market_close.loc[
            cls.trading_days]

        minute_writer = BcolzMinuteBarWriter(
            cls.trading_days[0],
            cls.tempdir.path,
            market_opens,
            market_closes,
            US_EQUITIES_MINUTES_PER_DAY
        )

        write_minute_data_for_asset(
            cls.env,
            minute_writer,
            cls.ASSET1.start_date,
            cls.ASSET1.end_date,
            cls.ASSET1.sid,
            start_val=2
        )

        write_minute_data_for_asset(
            cls.env,
            minute_writer,
            cls.ASSET2.start_date,
            cls.ASSET2.end_date,
            cls.ASSET2.sid,
            start_val=2,
            minute_blacklist=[
                pd.Timestamp('2015-01-08 14:31', tz='UTC'),
                pd.Timestamp('2015-01-08 21:00', tz='UTC'),
            ]
        )

    @classmethod
    def create_df_for_asset(cls, start_day, end_day, interval=1,
                            force_zeroes=False):
        days = cls.env.days_in_range(start_day, end_day)
        days_count = len(days)

        # default to 2 because the low array subtracts 1, and we don't
        # want to start with a 0
        days_arr = np.array(range(2, days_count + 2))

        df = pd.DataFrame({
            "open": days_arr + 1,
            "high": days_arr + 2,
            "low": days_arr - 1,
            "close": days_arr,
            "volume": 100 * days_arr,
        })

        if interval > 1:
            counter = 0
            while counter < days_count:
                df[counter:(counter + interval - 1)] = 0
                counter += interval

        df["day"] = [day.value for day in days]

        return df

    def test_daily_before_assets_trading(self):
        # asset2 and asset3 both started trading in 2015

        days = self.env.days_in_range(
            start=pd.Timestamp("2014-12-15", tz='UTC'),
            end=pd.Timestamp("2014-12-18", tz='UTC'),
        )

        for idx, day in enumerate(days):
            bar_data = BarData(self.data_portal, lambda: day, "daily")
            check_internal_consistency(
                bar_data, [self.ASSET2, self.ASSET3], ALL_FIELDS, 10, "1d"
            )

            for field in ALL_FIELDS:
                # OHLCP should be NaN
                # Volume should be 0
                asset2_series = bar_data.history(self.ASSET2, field, 10, "1d")
                asset3_series = bar_data.history(self.ASSET3, field, 10, "1d")

                if field == "volume":
                    np.testing.assert_array_equal(np.zeros(10), asset2_series)
                    np.testing.assert_array_equal(np.zeros(10), asset3_series)
                else:
                    np.testing.assert_array_equal(
                        np.full(10, np.nan),
                        asset2_series
                    )

                    np.testing.assert_array_equal(
                        np.full(10, np.nan),
                        asset3_series
                    )

    def test_daily_regular(self):
        # asset2 and asset3 both started on 1/5/2015, but asset3 trades every
        # 10 days

        # get the first 30 days of 2015
        jan5 = pd.Timestamp("2015-01-04")

        days = self.env.days_in_range(
            start=jan5,
            end=self.env.add_trading_days(30, jan5)
        )

        for idx, day in enumerate(days):
            self.verify_regular_dt(idx, day, "daily")

    def test_daily_some_assets_stopped(self):
        # asset1 ends on 2016-01-30
        # asset2 ends on 2015-12-13

        bar_data = BarData(self.data_portal,
                           lambda: pd.Timestamp("2016-01-06", tz='UTC'),
                           "daily")

        for field in OHLCP:
            window = bar_data.history(
                [self.ASSET1, self.ASSET2], field, 15, "1d"
            )

            # last 2 values for asset2 should be NaN (# of days since asset2
            # delisted)
            np.testing.assert_array_equal(
                np.full(2, np.nan),
                window[self.ASSET2][-2:]
            )

            # third from last value should not be NaN
            self.assertFalse(np.isnan(window[self.ASSET2][-3]))

        volume_window = bar_data.history(
            [self.ASSET1, self.ASSET2], "volume", 15, "1d"
        )

        np.testing.assert_array_equal(
            np.zeros(2),
            volume_window[self.ASSET2][-2:]
        )

        self.assertNotEqual(0, volume_window[self.ASSET2][-3])

    def test_daily_after_asset_stopped(self):
        # SHORT_ASSET trades on 1/5, 1/6, that's it.

        days = self.env.days_in_range(
            start=pd.Timestamp("2015-01-07", tz='UTC'),
            end=pd.Timestamp("2015-01-08", tz='UTC')
        )

        # days has 1/7, 1/8
        for idx, day in enumerate(days):
            bar_data = BarData(self.data_portal, lambda: day, "daily")
            check_internal_consistency(
                bar_data, self.SHORT_ASSET, ALL_FIELDS, 2, "1d"
            )

            for field in ALL_FIELDS:
                asset_series = bar_data.history(
                    self.SHORT_ASSET, field, 2, "1d"
                )

                if idx == 0:
                    # one value, then one NaN.  base value for 1/6 is 3.
                    if field in OHLCP:
                        self.assertEqual(
                            3 + MINUTE_FIELD_INFO[field],
                            asset_series.iloc[0]
                        )

                        self.assertTrue(np.isnan(asset_series.iloc[1]))
                    elif field == "volume":
                        self.assertEqual(300, asset_series.iloc[0])
                        self.assertEqual(0, asset_series.iloc[1])
                else:
                    # both NaNs
                    if field in OHLCP:
                        self.assertTrue(np.isnan(asset_series.iloc[0]))
                        self.assertTrue(np.isnan(asset_series.iloc[1]))
                    elif field == "volume":
                        self.assertEqual(0, asset_series.iloc[0])
                        self.assertEqual(0, asset_series.iloc[1])

    def test_daily_splits_and_mergers(self):
        # self.SPLIT_ASSET and self.MERGER_ASSET had splits/mergers
        # on 1/6 and 1/7.  they both started trading on 1/5

        for asset in [self.SPLIT_ASSET, self.MERGER_ASSET]:
            # before any of the adjustments
            window1 = self.data_portal.get_history_window(
                [asset],
                pd.Timestamp("2015-01-05", tz='UTC'),
                1,
                "1d",
                "close"
            )[asset]

            np.testing.assert_array_equal(window1, [2])

            window1_volume = self.data_portal.get_history_window(
                [asset],
                pd.Timestamp("2015-01-05", tz='UTC'),
                1,
                "1d",
                "volume"
            )[asset]

            np.testing.assert_array_equal(window1_volume, [200])

            # straddling the first event
            window2 = self.data_portal.get_history_window(
                [asset],
                pd.Timestamp("2015-01-06", tz='UTC'),
                2,
                "1d",
                "close"
            )[asset]

            # first value should be halved, second value unadjusted
            np.testing.assert_array_equal([1, 3], window2)

            window2_volume = self.data_portal.get_history_window(
                [asset],
                pd.Timestamp("2015-01-06", tz='UTC'),
                2,
                "1d",
                "volume"
            )[asset]

            if asset == self.SPLIT_ASSET:
                # first value should be doubled, second value unadjusted
                np.testing.assert_array_equal(window2_volume, [400, 300])
            elif asset == self.MERGER_ASSET:
                np.testing.assert_array_equal(window2_volume, [200, 300])

            # straddling both events
            window3 = self.data_portal.get_history_window(
                [asset],
                pd.Timestamp("2015-01-07", tz='UTC'),
                3,
                "1d",
                "close"
            )[asset]

            np.testing.assert_array_equal([0.5, 1.5, 4], window3)

            window3_volume = self.data_portal.get_history_window(
                [asset],
                pd.Timestamp("2015-01-07", tz='UTC'),
                3,
                "1d",
                "volume"
            )[asset]

            if asset == self.SPLIT_ASSET:
                np.testing.assert_array_equal(window3_volume, [800, 600, 400])
            elif asset == self.MERGER_ASSET:
                np.testing.assert_array_equal(window3_volume, [200, 300, 400])

    def test_daily_dividends(self):
        # self.DIVIDEND_ASSET had dividends on 1/6 and 1/7

        # before any dividend
        window1 = self.data_portal.get_history_window(
            [self.DIVIDEND_ASSET],
            pd.Timestamp("2015-01-05", tz='UTC'),
            1,
            "1d",
            "close"
        )[self.DIVIDEND_ASSET]

        np.testing.assert_array_equal(window1, [2])

        # straddling the first dividend
        window2 = self.data_portal.get_history_window(
            [self.DIVIDEND_ASSET],
            pd.Timestamp("2015-01-06", tz='UTC'),
            2,
            "1d",
            "close"
        )[self.DIVIDEND_ASSET]

        # first dividend is 2%, so the first value should be 2% lower than
        # before
        np.testing.assert_array_equal([1.96, 3], window2)

        # straddling both dividends
        window3 = self.data_portal.get_history_window(
            [self.DIVIDEND_ASSET],
            pd.Timestamp("2015-01-07", tz='UTC'),
            3,
            "1d",
            "close"
        )[self.DIVIDEND_ASSET]

        # second dividend is 0.96
        # first value should be 0.9408 of its original value, rounded to 3
        # digits. second value should be 0.96 of its original value
        np.testing.assert_array_equal([1.882, 2.88, 4], window3)

    def test_daily_blended_some_assets_stopped(self):
        # asset1 ends on 2016-01-30
        # asset2 ends on 2016-01-04

        bar_data = BarData(self.data_portal,
                           lambda: pd.Timestamp("2016-01-06 16:00", tz='UTC'),
                           "daily")

        for field in OHLCP:
            window = bar_data.history(
                [self.ASSET1, self.ASSET2], field, 15, "1d"
            )

            # last 2 values for asset2 should be NaN
            np.testing.assert_array_equal(
                np.full(2, np.nan),
                window[self.ASSET2][-2:]
            )

            # third from last value should not be NaN
            self.assertFalse(np.isnan(window[self.ASSET2][-3]))

        volume_window = bar_data.history(
            [self.ASSET1, self.ASSET2], "volume", 15, "1d"
        )

        np.testing.assert_array_equal(
            np.zeros(2),
            volume_window[self.ASSET2][-2:]
        )

        self.assertNotEqual(0, volume_window[self.ASSET2][-3])

    def test_daily_history_blended(self):
        # daily history windows that end mid-day use minute values for the
        # last day

        # January 2015 has both daily and minute data for ASSET2
        day = pd.Timestamp("2015-01-07", tz='UTC')
        minutes = self.env.market_minutes_for_day(day)

        # minute data, baseline:
        # Jan 5: 2 to 391
        # Jan 6: 392 to 781
        # Jan 7: 782 to 1172
        for idx, minute in enumerate(minutes):
            for field in ALL_FIELDS:
                adj = MINUTE_FIELD_INFO[field]

                window = self.data_portal.get_history_window(
                    [self.ASSET2],
                    minute,
                    3,
                    "1d",
                    field
                )[self.ASSET2]

                self.assertEqual(len(window), 3)

                if field == "volume":
                    self.assertEqual(window[0], 200)
                    self.assertEqual(window[1], 300)
                else:
                    self.assertEqual(window[0], 2 + adj)
                    self.assertEqual(window[1], 3 + adj)

                last_val = -1

                if field == "open":
                    last_val = 783
                elif field == "high":
                    # since we increase monotonically, it's just the last
                    # value
                    last_val = 784 + idx
                elif field == "low":
                    # since we increase monotonically, the low is the first
                    # value of the day
                    last_val = 781
                elif field == "close" or field == "price":
                    last_val = 782 + idx
                elif field == "volume":
                    # for volume, we sum up all the minutely volumes so far
                    # today

                    last_val = sum(np.array(range(782, 782 + idx + 1)) * 100)

                self.assertEqual(window[-1], last_val)

    @parameterized.expand(ALL_FIELDS)
    def test_daily_history_blended_gaps(self, field):
        # daily history windows that end mid-day use minute values for the
        # last day

        # January 2015 has both daily and minute data for ASSET2
        day = pd.Timestamp("2015-01-08", tz='UTC')
        minutes = self.env.market_minutes_for_day(day)

        # minute data, baseline:
        # Jan 5: 2 to 391
        # Jan 6: 392 to 781
        # Jan 7: 782 to 1172
        for idx, minute in enumerate(minutes):
            adj = MINUTE_FIELD_INFO[field]

            window = self.data_portal.get_history_window(
                [self.ASSET2],
                minute,
                3,
                "1d",
                field
            )[self.ASSET2]

            self.assertEqual(len(window), 3)

            if field == "volume":
                self.assertEqual(window[0], 300)
                self.assertEqual(window[1], 400)
            else:
                self.assertEqual(window[0], 3 + adj)
                self.assertEqual(window[1], 4 + adj)

            last_val = -1

            if field == "open":
                if idx == 0:
                    last_val = np.nan
                else:
                    last_val = 1174.0
            elif field == "high":
                # since we increase monotonically, it's just the last
                # value
                if idx == 0:
                    last_val = np.nan
                elif idx == 389:
                    last_val = 1562.0
                else:
                    last_val = 1174.0 + idx
            elif field == "low":
                # since we increase monotonically, the low is the first
                # value of the day
                if idx == 0:
                    last_val = np.nan
                else:
                    last_val = 1172.0
            elif field == "close":
                if idx == 0:
                    last_val = np.nan
                elif idx == 389:
                    last_val = 1172.0 + 388
                else:
                    last_val = 1172.0 + idx
            elif field == "price":
                if idx == 0:
                    last_val = 4
                elif idx == 389:
                    last_val = 1172.0 + 388
                else:
                    last_val = 1172.0 + idx
            elif field == "volume":
                # for volume, we sum up all the minutely volumes so far
                # today
                if idx == 0:
                    last_val = 0
                elif idx == 389:
                    last_val = sum(
                        np.array(range(1173, 1172 + 388 + 1)) * 100)
                else:
                    last_val = sum(
                        np.array(range(1173, 1172 + idx + 1)) * 100)

            np.testing.assert_almost_equal(window[-1], last_val,
                                           err_msg="field={0} minute={1}".
                                           format(field, minute))

    def test_history_window_before_first_trading_day(self):
        # trading_start is 2/3/2014
        # get a history window that starts before that, and ends after that

        second_day = self.env.next_trading_day(self.TRADING_START_DT)

        exp_msg = (
            "History window extends before 2014-02-03. To use this history "
            "window, start the backtest on or after 2014-02-07."
        )

        with self.assertRaisesRegexp(HistoryWindowStartsBeforeData, exp_msg):
            self.data_portal.get_history_window(
                [self.ASSET1],
                second_day,
                4,
                "1d",
                "price"
            )[self.ASSET1]

        with self.assertRaisesRegexp(HistoryWindowStartsBeforeData, exp_msg):
            self.data_portal.get_history_window(
                [self.ASSET1],
                second_day,
                4,
                "1d",
                "volume"
            )[self.ASSET1]

        # Use a minute to force minute mode.
        first_minute = self.env.open_and_closes.market_open[
            self.TRADING_START_DT]

        with self.assertRaisesRegexp(HistoryWindowStartsBeforeData, exp_msg):
            self.data_portal.get_history_window(
                [self.ASSET2],
                first_minute,
                4,
                "1d",
                "close"
            )[self.ASSET2]


class MinuteToDailyAggregationTestCase(WithBcolzMinutes,
                                       ZiplineTestCase):

    #    March 2016
    # Su Mo Tu We Th Fr Sa
    #        1  2  3  4  5
    #  6  7  8  9 10 11 12
    # 13 14 15 16 17 18 19
    # 20 21 22 23 24 25 26
    # 27 28 29 30 31

    TRADING_ENV_MIN_DATE = pd.Timestamp("2016-03-01", tz="UTC")
    TRADING_ENV_MAX_DATE = pd.Timestamp("2016-03-31", tz="UTC")

    minutes = pd.date_range('2016-03-15 9:31',
                            '2016-03-15 9:36',
                            freq='min',
                            tz='US/Eastern').tz_convert('UTC')

    @classmethod
    def make_equities_info(cls):
        return pd.DataFrame.from_dict({
            1: {
                "start_date": pd.Timestamp("2016-03-01", tz="UTC"),
                "end_date": pd.Timestamp("2016-03-31", tz="UTC"),
                "symbol": "EQUITY1",
            },
            2: {
                "start_date": pd.Timestamp("2016-03-01", tz='UTC'),
                "end_date": pd.Timestamp("2016-03-31", tz='UTC'),
                "symbol": "EQUITY2"
            },
        },
            orient='index')

    @classmethod
    def make_bcolz_minute_bar_data(cls):
        return {
            # sid data is created so that at least one high is lower than a
            # previous high, and the inverse for low
            1: pd.DataFrame(
                {
                    'open': [nan, 103.50, 102.50, 104.50, 101.50, nan],
                    'high': [nan, 103.90, 102.90, 104.90, 101.90, nan],
                    'low': [nan, 103.10, 102.10, 104.10, 101.10, nan],
                    'close': [nan, 103.30, 102.30, 104.30, 101.30, nan],
                    'volume': [0, 1003, 1002, 1004, 1001, 0]
                },
                index=cls.minutes,
            ),
            # sid 2 is included to provide data on different bars than sid 1,
            # as will as illiquidty mid-day
            2: pd.DataFrame({
                'open': [201.50, nan, 204.50, nan, 200.50, 202.50],
                'high': [201.90, nan, 204.90, nan, 200.90, 202.90],
                'low': [201.10, nan, 204.10, nan, 200.10, 202.10],
                'close': [201.30, nan, 203.50, nan, 200.30, 202.30],
                'volume': [2001, 0, 2004, 0, 2000, 2002],
            },
                index=cls.minutes,
            )
        }

    expected_values = {
        1: pd.DataFrame(
            {
                'open': [nan, 103.50, 103.50, 103.50, 103.50, 103.50],
                'high': [nan, 103.90, 103.90, 104.90, 104.90, 104.90],
                'low': [nan, 103.10, 102.10, 102.10, 101.10, 101.10],
                'close': [nan, 103.30, 102.30, 104.30, 101.30, 101.30],
                'volume': [0, 1003, 2005, 3009, 4010, 4010]
            },
            index=minutes,
        ),
        2: pd.DataFrame(
            {
                'open': [201.50, 201.50, 201.50, 201.50, 201.50, 201.50],
                'high': [201.90, 201.90, 204.90, 204.90, 204.90, 204.90],
                'low': [201.10, 201.10, 201.10, 201.10, 200.10, 200.10],
                'close': [201.30, 201.30, 203.50, 203.50, 200.30, 202.30],
                'volume': [2001, 2001, 4005, 4005, 6005, 8007],
            },
            index=minutes,
        )
    }

    @classmethod
    def init_class_fixtures(cls):
        super(MinuteToDailyAggregationTestCase, cls).init_class_fixtures()

        cls.EQUITIES = {
            1: cls.env.asset_finder.retrieve_asset(1),
            2: cls.env.asset_finder.retrieve_asset(2)
        }

    def init_instance_fixtures(self):
        super(MinuteToDailyAggregationTestCase, self).init_instance_fixtures()
        # Set up a fresh data portal for each test, since order of calling
        # needs to be tested.
        self.equity_daily_aggregator = DailyHistoryAggregator(
            self.env.open_and_closes.market_open,
            self.bcolz_minute_bar_reader,
        )

    @parameterized.expand([
        ('open_sid_1', 'open', 1),
        ('high_1', 'high', 1),
        ('low_1', 'low', 1),
        ('close_1', 'close', 1),
        ('volume_1', 'volume', 1),
        ('open_2', 'open', 2),
        ('high_2', 'high', 2),
        ('low_2', 'low', 2),
        ('close_2', 'close', 2),
        ('volume_2', 'volume', 2),

    ])
    def test_contiguous_minutes_individual(self, name, field, sid):
        # First test each minute in order.
        method_name = field + 's'
        results = []
        repeat_results = []
        asset = self.EQUITIES[sid]
        for minute in self.minutes:
            value = getattr(self.equity_daily_aggregator, method_name)(
                [asset], minute)[0]
            # Prevent regression on building an array when scalar is intended.
            self.assertIsInstance(value, Real)
            results.append(value)

            # Call a second time with the same dt, to prevent regression
            # against case where crossed start and end dts caused a crash
            # instead of the last value.
            value = getattr(self.equity_daily_aggregator, method_name)(
                [asset], minute)[0]
            # Prevent regression on building an array when scalar is intended.
            self.assertIsInstance(value, Real)
            repeat_results.append(value)

        assert_almost_equal(results, self.expected_values[asset][field],
                            err_msg="sid={0} field={1}".format(asset, field))
        assert_almost_equal(repeat_results, self.expected_values[asset][field],
                            err_msg="sid={0} field={1}".format(asset, field))

    @parameterized.expand([
        ('open_sid_1', 'open', 1),
        ('high_1', 'high', 1),
        ('low_1', 'low', 1),
        ('close_1', 'close', 1),
        ('volume_1', 'volume', 1),
        ('open_2', 'open', 2),
        ('high_2', 'high', 2),
        ('low_2', 'low', 2),
        ('close_2', 'close', 2),
        ('volume_2', 'volume', 2),

    ])
    def test_skip_minutes_individual(self, name, field, sid):
        # Test skipping minutes, to exercise backfills.
        # Tests initial backfill and mid day backfill.
        method_name = field + 's'
        for i in [1, 5]:
            minute = self.minutes[i]
            asset = self.EQUITIES[sid]
            value = getattr(self.equity_daily_aggregator, method_name)(
                [asset], minute)[0]
            # Prevent regression on building an array when scalar is intended.
            self.assertIsInstance(value, Real)
            assert_almost_equal(value,
                                self.expected_values[sid][field][i],
                                err_msg="sid={0} field={1} dt={2}".format(
                                    sid, field, minute))

            # Call a second time with the same dt, to prevent regression
            # against case where crossed start and end dts caused a crash
            # instead of the last value.
            value = getattr(self.equity_daily_aggregator, method_name)(
                [asset], minute)[0]
            # Prevent regression on building an array when scalar is intended.
            self.assertIsInstance(value, Real)
            assert_almost_equal(value,
                                self.expected_values[sid][field][i],
                                err_msg="sid={0} field={1} dt={2}".format(
                                    sid, field, minute))

    @parameterized.expand(OHLCV)
    def test_contiguous_minutes_multiple(self, field):
        # First test each minute in order.
        method_name = field + 's'
        assets = sorted(self.EQUITIES.values())
        results = {asset: [] for asset in assets}
        repeat_results = {asset: [] for asset in assets}
        for minute in self.minutes:
            values = getattr(self.equity_daily_aggregator, method_name)(
                assets, minute)
            for j, asset in enumerate(assets):
                value = values[j]
                # Prevent regression on building an array when scalar is
                # intended.
                self.assertIsInstance(value, Real)
                results[asset].append(value)

            # Call a second time with the same dt, to prevent regression
            # against case where crossed start and end dts caused a crash
            # instead of the last value.
            values = getattr(self.equity_daily_aggregator, method_name)(
                assets, minute)
            for j, asset in enumerate(assets):
                value = values[j]
                # Prevent regression on building an array when scalar is
                # intended.
                self.assertIsInstance(value, Real)
                repeat_results[asset].append(value)
        for asset in assets:
            assert_almost_equal(results[asset],
                                self.expected_values[asset][field],
                                err_msg="sid={0} field={1}".format(
                                    asset, field))
            assert_almost_equal(repeat_results[asset],
                                self.expected_values[asset][field],
                                err_msg="sid={0} field={1}".format(
                                    asset, field))

    @parameterized.expand(OHLCV)
    def test_skip_minutes_multiple(self, field):
        # Test skipping minutes, to exercise backfills.
        # Tests initial backfill and mid day backfill.
        method_name = field + 's'
        assets = sorted(self.EQUITIES.values())
        for i in [1, 5]:
            minute = self.minutes[i]
            values = getattr(self.equity_daily_aggregator, method_name)(
                assets, minute)
            for j, asset in enumerate(assets):
                value = values[j]
                # Prevent regression on building an array when scalar is
                # intended.
                self.assertIsInstance(value, Real)
                assert_almost_equal(
                    value,
                    self.expected_values[asset][field][i],
                    err_msg="sid={0} field={1} dt={2}".format(
                        asset, field, minute))

            # Call a second time with the same dt, to prevent regression
            # against case where crossed start and end dts caused a crash
            # instead of the last value.
            values = getattr(self.equity_daily_aggregator, method_name)(
                assets, minute)
            for j, asset in enumerate(assets):
                value = values[j]
                # Prevent regression on building an array when scalar is
                # intended.
                self.assertIsInstance(value, Real)
                assert_almost_equal(
                    value,
                    self.expected_values[asset][field][i],
                    err_msg="sid={0} field={1} dt={2}".format(
                        asset, field, minute))
