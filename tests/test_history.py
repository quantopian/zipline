from textwrap import dedent
from unittest import TestCase

import pandas as pd
import numpy as np

from testfixtures import TempDirectory

from zipline import TradingAlgorithm
from zipline.assets import Asset
from zipline.data.data_portal import DataPortal
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
from zipline.utils.test_utils import str_to_seconds, MockDailyBarReader, \
    DailyBarWriterFromDataFrames, write_minute_data_for_asset


OHLC = ["open", "high", "low", "close"]
OHLCP = OHLC + ["price"]
ALL_FIELDS = OHLCP + ["volume"]


class HistoryTestCaseBase(TestCase):
    # asset1:
    # - 2014-03-01 (rounds up to TRADING_START_DT) to 2016-01-30.
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
        cls.TRADING_END_DT = pd.Timestamp("2016-01-30", tz='UTC')

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
        cls.create_data_portal()

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
                "end_date": cls.env.next_trading_day(
                    pd.Timestamp("2016-01-30", tz='UTC')
                ),
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
                "end_date": pd.Timestamp("2015-01-07", tz='UTC'),
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

    def verify_regular_dt(self, idx, dt, mode):
        if mode == "daily":
            freq = "1d"
        else:
            freq = "1m"

        bar_data = BarData(self.data_portal, lambda: dt, mode)
        check_internal_consistency(
            bar_data, [self.ASSET2, self.ASSET3], ALL_FIELDS, 10, freq
        )

        for field in ALL_FIELDS:
            asset2_series = bar_data.history(self.ASSET2, field, 10, freq)
            asset3_series = bar_data.history(self.ASSET3, field, 10, freq)

            base = MINUTE_FIELD_INFO[field] + 2

            if idx < 9:
                missing_count = 9 - idx
                present_count = 9 - missing_count

                if field in OHLCP:
                    # asset2 should have some leading nans
                    np.testing.assert_array_equal(
                        np.full(missing_count, np.nan),
                        asset2_series[0:missing_count]
                    )

                    # asset2 should also have some real values
                    np.testing.assert_array_equal(
                        np.array(range(base, base + present_count + 1)),
                        asset2_series[(9 - present_count):]
                    )

                    # asset3 should be NaN the entire time
                    np.testing.assert_array_equal(
                        np.full(10, np.nan),
                        asset3_series
                    )
                elif field == "volume":
                    # asset2 should have some zeros (instead of nans)
                    np.testing.assert_array_equal(
                        np.zeros(missing_count),
                        asset2_series[0:missing_count]
                    )

                    # and some real values
                    np.testing.assert_array_equal(
                        np.array(
                            range(base, base + present_count + 1)
                        ) * 100,
                        asset2_series[(9 - present_count):]
                    )

                    # asset3 is all zeros, no volume yet
                    np.testing.assert_array_equal(
                        np.zeros(10),
                        asset3_series
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

                    np.testing.assert_array_equal(
                        np.array(range(base + idx - 9, base + idx + 1)),
                        asset2_series
                    )

                    np.testing.assert_array_equal(
                        asset3_answer_key,
                        asset3_series
                    )
                elif field == "volume":
                    asset3_answer_key = np.zeros(10)
                    asset3_answer_key[-position_from_end] = \
                        value_for_asset3 * 100

                    np.testing.assert_array_equal(
                        np.array(
                            range(base + idx - 9, base + idx + 1)
                        ) * 100,
                        asset2_series
                    )

                    np.testing.assert_array_equal(
                        asset3_answer_key,
                        asset3_series
                    )
                elif field == "price":
                    # price is always forward filled

                    # asset2 has prices every minute, so it's easy

                    # at idx 9, the data is 2 to 11
                    np.testing.assert_array_equal(
                        range(idx - 7, idx + 3),
                        asset2_series
                    )

                    first_part = asset3_series[0:-position_from_end]
                    second_part = asset3_series[-position_from_end:]

                    decile_count = ((idx + 1) // 10)

                    # in our test data, asset3 prices will be nine NaNs,
                    # then ten 11s, ten 21s, ten 31s...

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

        writer = BcolzMinuteBarWriter(
            cls.trading_days[0],
            cls.tempdir.path,
            market_opens,
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

        for sid in [2, 4, 5, 6]:
            write_minute_data_for_asset(
                cls.env,
                writer,
                pd.Timestamp("2015-01-05", tz='UTC'),
                pd.Timestamp("2015-12-31", tz='UTC'),
                sid,
                start_val=2
            )

        write_minute_data_for_asset(
            cls.env,
            writer,
            pd.Timestamp("2014-07-02", tz='UTC'),
            pd.Timestamp("2015-12-31", tz='UTC'),
            cls.HALF_DAY_TEST_ASSET.sid,
            start_val=2
        )

        write_minute_data_for_asset(
            cls.env,
            writer,
            pd.Timestamp("2015-01-05", tz='UTC'),
            pd.Timestamp("2015-12-31", tz='UTC'),
            3,
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

    def test_minute_regular(self):
        # asset2 and asset3 both started on 1/5/2015, but asset3 trades every
        # 10 minutes

        minutes = self.env.market_minutes_for_day(
            pd.Timestamp("2015-01-05", tz='UTC')
        )[0:60]

        for idx, minute in enumerate(minutes):
            self.verify_regular_dt(idx, minute, "minute")

    def test_minute_after_asset_stopped(self):
        # asset2 stopped at 1/4/16

        #  get some history windows that straddle the end
        minutes = self.env.market_minutes_for_day(
            pd.Timestamp("2016-01-04", tz='UTC')
        )[0:60]

        all_asset2_minutes = self.env.minutes_for_days_in_range(
            start=self.ASSET2.start_date,
            end=self.ASSET2.end_date
        )

        for idx, minute in enumerate(minutes):
            bar_data = BarData(self.data_portal, lambda: minute, "minute")
            check_internal_consistency(
                bar_data, self.ASSET2, ALL_FIELDS, 30, "1m"
            )

            # asset2's base minute value started at 2 and just goes up
            # one per minute
            asset2_minute_idx = all_asset2_minutes.searchsorted(minute) + 2

            for field in ALL_FIELDS:
                asset2_series = bar_data.history(self.ASSET2, field, 30, "1m")

                if idx < 30:
                    present_count = 29 - idx
                    missing_count = 30 - present_count

                    offset = MINUTE_FIELD_INFO[field]

                    if field in OHLCP:
                        answer_key = np.array(range(
                            asset2_minute_idx + offset - present_count - idx,
                            asset2_minute_idx + offset - missing_count + 1
                        ))

                        np.testing.assert_array_equal(
                            answer_key,
                            asset2_series[0:present_count]
                        )

                        if missing_count > 0:
                            np.testing.assert_array_equal(
                                np.full(missing_count, np.nan),
                                asset2_series[(30 - missing_count):]
                            )
                    elif field == "volume":
                        answer_key = np.array(range(
                            asset2_minute_idx - present_count - idx,
                            asset2_minute_idx - missing_count + 1
                        )) * 100

                        np.testing.assert_array_equal(
                            answer_key,
                            asset2_series[0:present_count]
                        )

                        if missing_count > 0:
                            np.testing.assert_array_equal(
                                np.zeros(missing_count),
                                asset2_series[(30 - missing_count):]
                            )
                else:
                    # completely after the asset's end date
                    if field in OHLCP:
                        np.testing.assert_array_equal(
                            np.full(30, np.nan),
                            asset2_series
                        )
                    elif field == "volume":
                        np.testing.assert_array_equal(
                            np.zeros(30), asset2_series
                        )

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
            "History window extends beyond environment first date, "
            "2014-02-03. To use history with bar count, 15, start simulation "
            "on or after, 2014-02-04."
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
            dfs[sid] = cls.create_df_for_asset(
                pd.Timestamp("2015-01-05", tz='UTC'),
                pd.Timestamp("2015-12-31", tz='UTC')
            )

        days = cls.env.days_in_range(
            cls.TRADING_START_DT,
            cls.TRADING_END_DT
        )

        daily_writer = DailyBarWriterFromDataFrames(dfs)
        daily_writer.write(path, days, dfs)

        market_opens = cls.env.open_and_closes.market_open.loc[
            cls.trading_days]

        minute_writer = BcolzMinuteBarWriter(
            cls.trading_days[0],
            cls.tempdir.path,
            market_opens,
            US_EQUITIES_MINUTES_PER_DAY
        )

        write_minute_data_for_asset(
            cls.env,
            minute_writer,
            cls.ASSET2.start_date,
            cls.env.previous_trading_day(cls.ASSET2.end_date),
            2,
            start_val=2
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

    def test_daily_after_asset_stopped(self):
        # SHORT_ASSET trades on 1/5, 1/6, that's it.

        days = self.env.days_in_range(
            start=self.SHORT_ASSET.end_date,
            end=self.env.next_trading_day(self.SHORT_ASSET.end_date)
        )

        # days has 1/7, 1/8, 1/9
        for idx, day in enumerate(days):
            bar_data = BarData(self.data_portal, lambda: day, "daily")
            check_internal_consistency(
                bar_data, self.SHORT_ASSET, ALL_FIELDS, 2, "1d"
            )

            for field in ["close"]:
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

            np.testing.assert_array_equal(window2_volume, [100, 300])

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

            np.testing.assert_array_equal(window3_volume, [50, 150, 400])

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

    def test_history_window_before_first_trading_day(self):
        # trading_start is 2/3/2014
        # get a history window that starts before that, and ends after that

        second_day = self.env.next_trading_day(self.TRADING_START_DT)

        exp_msg = (
            "History window extends beyond environment first date, "
            "2014-02-03. To use history with bar count, 4, start simulation "
            "on or after, 2014-02-07."
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
