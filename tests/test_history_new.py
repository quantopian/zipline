from textwrap import dedent
from unittest import TestCase
import pandas as pd
import numpy as np

from testfixtures import TempDirectory

from zipline import TradingAlgorithm
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
from zipline.errors import HistoryInInitialize
from zipline.finance.trading import (
    TradingEnvironment,
    SimulationParameters
)
from zipline.utils.test_utils import str_to_seconds, MockDailyBarReader, \
    DailyBarWriterFromDataFrames


class HistoryTestCaseBase(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = TradingEnvironment()
        cls.tempdir = TempDirectory()

        # trading_start (when bcolz files start) = 2014-02-01
        cls.TRADING_START_DT = pd.Timestamp("2014-02-03", tz='UTC')
        cls.TRADING_END_DT = pd.Timestamp("2016-01-30", tz='UTC')

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
                'effective_date': str_to_seconds("2015-04-21"),
                'ratio': 0.5,
                'sid': cls.SPLIT_ASSET.sid
            },
            {
                'effective_date': str_to_seconds("2015-04-23"),
                'ratio': 0.5,
                'sid': cls.SPLIT_ASSET.sid
            },
        ])

        mergers = pd.DataFrame([
            {
                'effective_date': str_to_seconds("2015-04-21"),
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
                    pd.Timestamp("2015-04-21", tz='UTC').to_datetime64(),
                'record_date':
                    pd.Timestamp("2015-04-21", tz='UTC').to_datetime64(),
                'declared_date':
                    pd.Timestamp("2015-04-21", tz='UTC').to_datetime64(),
                'pay_date':
                    pd.Timestamp("2015-04-21", tz='UTC').to_datetime64(),
                'amount': 2.0,
                'sid': cls.DIVIDEND_ASSET.sid
            },
            {
                'ex_date':
                    pd.Timestamp("2015-04-23", tz='UTC').to_datetime64(),
                'record_date':
                    pd.Timestamp("2015-04-23", tz='UTC').to_datetime64(),
                'declared_date':
                    pd.Timestamp("2015-04-23", tz='UTC').to_datetime64(),
                'pay_date':
                    pd.Timestamp("2015-04-23", tz='UTC').to_datetime64(),
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

    @classmethod
    def write_minute_data_for_asset(cls, writer, start_dt, end_dt, sid,
                                    interval=1):
        asset_minutes = cls.env.minutes_for_days_in_range(start_dt, end_dt)

        minutes_count = len(asset_minutes)

        if interval == 1:
            minutes_arr = np.array(range(1, minutes_count + 1))
        else:
            # write every `interval` slots
            # ie, instead of [1, 2, 3, 4, 5], would be [0, 0, 1, 0, 0, 2, ..]
            # with interval 3
            minutes_arr = np.zeros(minutes_count)

            insert_position = interval
            counter = 1
            while insert_position < minutes_count:
                minutes_arr[insert_position] = counter
                counter += 1
                insert_position += interval

        df = pd.DataFrame({
            "open": minutes_arr + 1,
            "high": minutes_arr + 2,
            "low": minutes_arr - 1,
            "close": minutes_arr,
            "volume": 100 * minutes_arr,
            "dt": asset_minutes
        }).set_index("dt")

        writer.write(sid, df)


class MinuteEquityHistoryTestCase(HistoryTestCaseBase):

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
    # - 2015-01-05 to 2015-12-31
    # - trades every minute
    # - splits on 2015-04-21 and 2014-04-23

    # DIVIDEND_ASSET:
    # - 2015-01-05 to 2015-12-31
    # - trades every minute
    # - dividends on 2015-04-21 and 2014-04-23

    # MERGER_ASSET
    # - 2015-01-05 to 2015-12-31
    # - trades every minute
    # - merger on 2015-04-21

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

        cls.write_minute_data_for_asset(
            writer,
            pd.Timestamp("2014-01-03", tz='UTC'),
            pd.Timestamp("2016-01-30", tz='UTC'),
            1
        )

        for sid in [2, 4, 5, 6]:
            cls.write_minute_data_for_asset(
                writer,
                pd.Timestamp("2015-01-05", tz='UTC'),
                pd.Timestamp("2015-12-31", tz='UTC'),
                sid
            )

        cls.write_minute_data_for_asset(
            writer,
            pd.Timestamp("2015-01-05", tz='UTC'),
            pd.Timestamp("2015-12-31", tz='UTC'),
            3,
            interval=10
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

        start = pd.Timestamp('2007-04-05', tz='UTC')
        end = pd.Timestamp('2007-04-10', tz='UTC')

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

    def test_minute_basic_functionality(self):
        minutes = self.env.market_minutes_for_day(self.TRADING_START_DT)

        # asset1's first minute bar is:
        # 7802 (open), 7803 (high), 7800 (close), 7801 (close), 780100
        # (this is because it started trading before self.TRADING_START_DT)

        asset1_minutes = self.env.minutes_for_days_in_range(
            self.env.get_open_and_close(
                pd.Timestamp("2014-01-03", tz='UTC')
            )[0],
            pd.Timestamp("2016-01-30", tz='UTC')
        )

        starting_base_value = asset1_minutes.searchsorted(minutes[0]) + 1

        field_info = [
            ("open", starting_base_value + 1),
            ("high", starting_base_value + 2),
            ("low", starting_base_value - 1),
            ("close", starting_base_value)
        ]

        for field, adj_base_value in field_info:
            for idx, minute in enumerate(minutes):
                # get rolling 10 minute windows for asset1 starting on first
                # trading day.  for all the minutes before first trading day,
                # we should get NaNs back.
                window = self.data_portal.get_history_window(
                    [self.ASSET1],
                    minute,
                    10,
                    "1m",
                    field
                )[self.ASSET1]

                if idx < 9:
                    # there should be (9 - idx) leading NaNs
                    # and then idx non-NaNs
                    nan_count = 9 - idx
                    np.testing.assert_array_equal(
                        np.full(nan_count, np.nan),
                        window[0:nan_count]
                    )

                    non_nan_count = 9 - nan_count
                    np.testing.assert_array_equal(
                        np.array(range(
                            adj_base_value,
                            adj_base_value + non_nan_count + 1
                        )),
                        window[nan_count:]
                    )
                else:
                    # no more nans
                    np.testing.assert_array_equal(
                        np.array(range(
                            adj_base_value + idx - 9,
                            adj_base_value + idx + 1
                        )),
                        window
                    )

    def test_minute_forward_fill(self):
        dt = self.env.market_minutes_for_day(self.ASSET3.start_date)[60]

        # don't ffill on price
        window = self.data_portal.get_history_window(
            [self.ASSET3],
            dt,
            60,
            "1m",
            "price",
            ffill=False
        )[self.ASSET3]

        for i in range(0, 60):
            if (i + 1) % 10 == 0:
                self.assertEqual(window[i], (i + 1) / 10)
            else:
                self.assertTrue(np.isnan(window[i]))

        # ffill on price
        window2 = self.data_portal.get_history_window(
            [self.ASSET3],
            dt,
            60,
            "1m",
            "price",
            ffill=True
        )[self.ASSET3]

        for i in range(0, 60):
            if i < 9:
                # there is no data to forward-fill from, since we are at the
                # very beginning of asset3's data
                self.assertTrue(np.isnan(window2[i]))
            else:
                self.assertEqual(window2[i], (i + 1) / 10)

        # ffill on price, but craft window in a way that the beginning of
        # the window has no data, but there is data before that (outside the
        # window). verify that the outside-the-window data gets brought into
        # the window due to ffill.
        window3 = self.data_portal.get_history_window(
            [self.ASSET3],
            dt,
            15,
            "1m",
            "price",
            ffill=True
        )[self.ASSET3]

        np.testing.assert_array_equal(
            np.array(([4] * 4) + ([5] * 10) + [6]),
            window3
        )

        # try ffilling on another field, unsuccessfully
        window4 = self.data_portal.get_history_window(
            [self.ASSET3],
            dt,
            60,
            "1m",
            "volume",
            ffill=True
        )[self.ASSET3]

        for i in range(0, 60):
            if (i + 1) % 10 == 0:
                self.assertEqual(window4[i], 100 * (i + 1) / 10)
            else:
                self.assertEqual(0, window4[i])

    def test_minute_history_after_asset_stopped_trading(self):
        # asset2 ends on 12/31

        # get a window that is entirely after an asset's end date.
        # should be all NaNs
        window = self.data_portal.get_history_window(
            [self.ASSET2],
            pd.Timestamp("2016-01-04 15:31", tz='UTC'),
            60,
            "1m",
            "close"
        )[self.ASSET2]

        np.testing.assert_array_equal(
            np.full(60, np.nan),
            window
        )

        # get a window that straddles an asset's end date
        window2 = self.data_portal.get_history_window(
            [self.ASSET2],
            pd.Timestamp("2016-01-04 14:32", tz='UTC'),
            5,
            "1m",
            "close"
        )[self.ASSET2]

        # should be 97528, 97529, 97530, NaN, NaN
        self.assertEqual(97528, window2[0])
        self.assertEqual(97529, window2[1])
        self.assertEqual(97530, window2[2])
        self.assertTrue(np.isnan(window2[3]))
        self.assertTrue(np.isnan(window2[4]))

    def test_minute_multiple_assets(self):
        # sanity check that we can get a history window of multiple assets
        # at the same time.

        # while we're at it, make the window straddle one of the asset's
        # start dates.
        window = self.data_portal.get_history_window(
            [self.ASSET1, self.ASSET2],
            pd.Timestamp("2015-01-05 14:32", tz='UTC'),
            10,
            "1m",
            "close"
        )

        # asset1 should be 97733-97742
        np.testing.assert_array_equal(
            np.array(range(97733, 97743)),
            window[self.ASSET1]
        )

        # asset2 should be 8 NaNs then 1, 2
        np.testing.assert_array_equal(
            np.full(8, np.nan),
            window[self.ASSET2][0:8]
        )

        np.testing.assert_array_equal(
            np.array(range(1, 3)),
            window[self.ASSET2][8:]
        )

    def test_minute_splits(self):
        # self.SPLIT_ASSET had splits on 4/21 and 4/23

        # before any of the splits
        window1 = self.data_portal.get_history_window(
            [self.SPLIT_ASSET],
            pd.Timestamp("2015-04-20 20:00", tz='UTC'),
            10,
            "1m",
            "close"
        )[self.SPLIT_ASSET]

        np.testing.assert_array_equal(
            np.array(range(28461, 28471)),
            window1
        )

        # straddling the first split
        window2 = self.data_portal.get_history_window(
            [self.SPLIT_ASSET],
            pd.Timestamp("2015-04-21 13:35", tz='UTC'),
            10,
            "1m",
            "close"
        )[self.SPLIT_ASSET]

        np.testing.assert_array_equal(
            [14233.0, 14233.5, 14234.0, 14234.5, 14235.0,
             28471, 28472, 28473, 28474, 28475],
            window2
        )

        # straddling the second split
        window3 = self.data_portal.get_history_window(
            [self.SPLIT_ASSET],
            pd.Timestamp("2015-04-23 13:35", tz='UTC'),
            10,
            "1m",
            "close"
        )[self.SPLIT_ASSET]

        np.testing.assert_array_equal(
            [14623.0, 14623.5, 14624.0, 14624.5, 14625.0,
             29251, 29252, 29253, 29254, 29255],
            window3
        )

    def test_minute_mergers(self):
        # self.MERGER_ASSET has a merger on 4/21, ratio 0.5

        # before the merger
        window1 = self.data_portal.get_history_window(
            [self.MERGER_ASSET],
            pd.Timestamp("2015-04-20 20:00", tz='UTC'),
            10,
            "1m",
            "close"
        )[self.MERGER_ASSET]

        np.testing.assert_array_equal(
            np.array(range(28461, 28471)),
            window1
        )

        # straddling the merger
        window2 = self.data_portal.get_history_window(
            [self.MERGER_ASSET],
            pd.Timestamp("2015-04-21 13:35", tz='UTC'),
            10,
            "1m",
            "close"
        )[self.MERGER_ASSET]

        np.testing.assert_array_equal(
            [14233.0, 14233.5, 14234.0, 14234.5, 14235.0,
             28471, 28472, 28473, 28474, 28475],
            window2
        )

    def test_minute_dividends(self):
        # self.DIVIDEND_ASSET had splits on 4/21 and 4/23

        # before any of the dividends
        window1 = self.data_portal.get_history_window(
            [self.DIVIDEND_ASSET],
            pd.Timestamp("2015-04-20 20:00", tz='UTC'),
            10,
            "1m",
            "close"
        )[self.DIVIDEND_ASSET]

        np.testing.assert_array_equal(
            np.array(range(28461, 28471)),
            window1
        )

        # straddling the first split
        window2 = self.data_portal.get_history_window(
            [self.DIVIDEND_ASSET],
            pd.Timestamp("2015-04-21 13:35", tz='UTC'),
            10,
            "1m",
            "close"
        )[self.DIVIDEND_ASSET]

        # -2% hit in the first five
        np.testing.assert_array_equal(
            [27896.68, 27897.66, 27898.64, 27899.62, 27900.6, 28471,
             28472, 28473, 28474, 28475],
            window2
        )

        # straddling the second split
        window3 = self.data_portal.get_history_window(
            [self.DIVIDEND_ASSET],
            pd.Timestamp("2015-04-23 13:35", tz='UTC'),
            10,
            "1m",
            "close"
        )[self.DIVIDEND_ASSET]

        # -4% hit in the first five
        np.testing.assert_array_equal(
            [28076.16, 28077.12, 28078.08, 28079.04, 28080, 29251,
             29252, 29253, 29254, 29255],
            window3
        )

    def test_minute_window_ends_before_trading_start(self):
        # asset2 started on 2015-01-05, so let's go before that
        window = self.data_portal.get_history_window(
            [self.ASSET2],
            pd.Timestamp("2014-12-10 15:00", tz='UTC'),
            10,
            "1m",
            "close"
        )[self.ASSET2]

        np.testing.assert_array_equal(
            np.full(10, np.nan),
            window
        )

    def test_minute_early_close(self):
        # 2014-07-03 is an early close
        #
        # give minutes into the day after the early close, get 20 1m bars
        window = self.data_portal.get_history_window(
            [self.ASSET1],
            pd.Timestamp("2014-07-07 13:35:00", tz='UTC'),
            20,
            "1m",
            "close"
        )[self.ASSET1]

        np.testing.assert_array_equal(
            np.array(range(48946, 48966)),
            window
        )

        self.assertEqual(
            window.index[-6],
            pd.Timestamp("2014-07-03 17:00", tz='UTC')
        )

        self.assertEqual(
            window.index[-5],
            pd.Timestamp("2014-07-07 13:31", tz='UTC')
        )


class DailyEquityHistoryTestCase(HistoryTestCaseBase):

    # asset1:
    # - 2014-03-01 (rounds up to TRADING_START_DT) to 2016-01-30.
    # - every day.

    # asset2:
    # - 2015-01-05 to 2015-12-31
    # - every day, has minute data from 2015-12-01 to 2012-12-31.

    # asset3:
    # - 2015-01-05 to 2015-12-31
    # - trades every 10 days

    # SPLIT_ASSET:
    # - 2015-01-05 to 2015-12-31
    # - trades every day
    # - splits on 2015-04-21 and 2014-04-23

    # DIVIDEND_ASSET:
    # - 2015-01-05 to 2015-12-31
    # - trades every day
    # - dividends on 2015-04-21 and 2014-04-23

    # MERGER_ASSET
    # - 2015-01-05 to 2015-12-31
    # - trades every day
    # - merger on 2015-04-21

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

        cls.write_minute_data_for_asset(
            minute_writer,
            cls.TRADING_START_DT,
            pd.Timestamp("2015-12-31", tz='UTC'),
            1
        )

    @classmethod
    def create_df_for_asset(cls, start_day, end_day, interval=1,
                            force_zeroes=False):
        days = cls.env.days_in_range(start_day, end_day)

        days_count = len(days)

        if interval == 1:
            # default to 2 because the low array subtracts 1, and we don't
            # want to start with a 0
            days_arr = np.array(range(2, days_count + 2))
        else:
            days_arr = np.zeros(days_count)

            insert_position = interval
            counter = 1
            while insert_position < days_count:
                days_arr[insert_position] = counter
                counter += 1
                insert_position += interval

        df = pd.DataFrame({
            "open": days_arr + 1,
            "high": days_arr + 2,
            "low": days_arr - 1,
            "close": days_arr,
            "volume": (days_arr * 100) if force_zeroes else (days_arr + 100),
            "day": [day.value for day in days]
        })

        return df

    def test_daily_basic_functionality(self):
        # take sliding history windows for asset1 from TRADING_START_DT
        # up to TRADING_END_DT

        field_info = [
            ("open", 1),
            ("high", 2),
            ("low", -1),
            ("close", 0),
            ("volume", 100)
        ]

        days = self.env.days_in_range(
            self.TRADING_START_DT,
            self.TRADING_END_DT
        )

        for field, adjustment in field_info:
            for idx, day in enumerate(days):
                window = self.data_portal.get_history_window(
                    [self.ASSET1],
                    day,
                    5,
                    "1d",
                    field
                )[self.ASSET1]

                if idx < 4:
                    # should be (4 - idx) leading NaNs and then idx non-NaNs
                    nan_count = 4 - idx

                    np.testing.assert_array_equal(
                        np.full(nan_count, np.nan),
                        window[0:nan_count]
                    )

                    non_nan_count = 5 - nan_count
                    np.testing.assert_array_equal(
                        np.array(range(adjustment + 2,
                                       adjustment + non_nan_count + 2)),
                        window[nan_count:]
                    )
                else:
                    # no more nans, starts at idx = 4
                    np.testing.assert_array_equal(
                        np.array(range(adjustment + idx - 2,
                                       adjustment + idx + 3)),
                        window
                    )

    def test_daily_history_blended(self):
        # daily history windows that end mid-day use minute values for the
        # last day

        # December 2015 has both daily and minute data for ASSET1.
        # do a bunch of history windows on Dec 16 that extend back to Dec 14

        dec_16 = pd.Timestamp("2015-12-16", tz='UTC')
        minutes = self.env.market_minutes_for_day(dec_16)

        field_info = [
            ("open", 1),
            ("high", 2),
            ("low", -1),
            ("close", 0),
            ("volume", 100)
        ]

        # find the idx of dec 16th for asset1
        day_idx = self.trading_days.searchsorted(dec_16) + 2

        all_minutes = self.env.minutes_for_days_in_range(
            self.trading_days[0],
            self.trading_days[-1]
        )

        # first minute idx of dec 16
        first_minute_idx = all_minutes.searchsorted(
            self.env.get_open_and_close(dec_16)[0]
        ) + 1

        for field, adj in field_info:
            for idx, minute in enumerate(minutes):
                window = self.data_portal.get_history_window(
                    [self.ASSET1],
                    minute,
                    3,
                    "1d",
                    field
                )[self.ASSET1]

                self.assertEqual(len(window), 3)

                self.assertEqual(window[0], day_idx + adj - 2)
                self.assertEqual(window[1], day_idx + adj - 1)

                last_val = -1

                vol_range = range(
                    first_minute_idx * 100,
                    (first_minute_idx + 390) * 100,
                    100
                )

                if field == "open":
                    last_val = first_minute_idx + adj
                elif field == "high":
                    # since we increase monotonically, it's just the last
                    # value
                    last_val = first_minute_idx + idx + adj
                elif field == "low":
                    # since we increase monotonically, the low is the first
                    # value of the day
                    last_val = first_minute_idx + adj
                elif field == "close":
                    last_val = first_minute_idx + idx + adj
                elif field == "volume":
                    # for volume, we sum up all the minutely volumes so far
                    # today

                    # minutely volumes are:
                    # first_minute_idx * 100
                    # (first_minute_idx + 1) * 100
                    # (first_minute_idx + 2) * 100
                    # ...

                    last_val = sum(vol_range[0:idx+1])

                self.assertEqual(window[-1], last_val)

    def test_daily_forward_fill(self):
        # ASSET3 trades every 10 days

        asset3_start_idx = self.trading_days.searchsorted(
            self.ASSET3.start_date
        )

        day_to_use = self.trading_days[asset3_start_idx + 60]

        # don't ffill on price
        window = self.data_portal.get_history_window(
            [self.ASSET3],
            day_to_use,
            60,
            "1d",
            "price",
            ffill=False
        )[self.ASSET3]

        for i in range(0, 60):
            if (i + 1) % 10 == 0:
                self.assertEqual(window[i], (i + 1) / 10)
            else:
                self.assertTrue(np.isnan(window[i]))

        # ffill on price
        window2 = self.data_portal.get_history_window(
            [self.ASSET3],
            day_to_use,
            90,
            "1d",
            "price",
            ffill=True
        )[self.ASSET3]

        for i in range(0, 90):
            if i < 39:
                # 30 leading NaNs because that's before the asset lifetime
                # 10 more leading NaNs because we had 0 as the first 9 prices
                self.assertTrue(np.isnan(window2[i]))
            else:
                self.assertEqual(window2[i], (i - 29) / 10)

        # ffill on price, but craft window in a way that the beginning of
        # the window has no data, but there is data before that (outside the
        # window). verify that the outside-the-window data gets brought into
        # the window due to ffill.
        window3 = self.data_portal.get_history_window(
            [self.ASSET3],
            day_to_use,
            15,
            "1d",
            "price",
            ffill=True
        )[self.ASSET3]

        np.testing.assert_array_equal(
            np.array(([4] * 4) + ([5] * 10) + [6]),
            window3
        )

        # try ffilling on another field, unsuccessfully
        window4 = self.data_portal.get_history_window(
            [self.ASSET3],
            day_to_use,
            60,
            "1d",
            "volume",
            ffill=True
        )[self.ASSET3]

        for i in range(0, 60):
            if (i + 1) % 10 == 0:
                self.assertEqual(window4[i], 100 * (i + 1) / 10)
            else:
                self.assertTrue(np.isnan(window[i]))

    def test_daily_history_after_asset_stopped_trading(self):
        # asset2 ends on 12/31, so this window is entirely afterwards
        window = self.data_portal.get_history_window(
            [self.ASSET2],
            pd.Timestamp("2016-01-08", tz='UTC'),
            3,
            "1d",
            "close"
        )[self.ASSET2]

        np.testing.assert_array_equal(
            np.full(3, np.nan),
            window
        )

        # straddle asset end date
        window2 = self.data_portal.get_history_window(
            [self.ASSET2],
            pd.Timestamp("2016-01-08", tz='UTC'),
            10,
            "1d",
            "close"
        )[self.ASSET2]

        # first five are real values
        np.testing.assert_array_equal(
            range(248, 253),
            window2[0:5]
        )

        # second five are nans
        np.testing.assert_array_equal(
            np.full(5, np.nan),
            window2[5:]
        )

    def test_daily_split(self):
        # self.SPLIT_ASSET had splits on 4/21 and 4/23

        # before any of the splits
        window1 = self.data_portal.get_history_window(
            [self.SPLIT_ASSET],
            pd.Timestamp("2015-04-20", tz='UTC'),
            10,
            "1d",
            "close"
        )[self.SPLIT_ASSET]

        np.testing.assert_array_equal(
            np.array(range(65, 75)),
            window1
        )

        # straddling both splits
        # FIXME finish tests



#     # FIXME daily history tests
#     # - splits, adjustments, mergers
#     # - window end after asset end
#     # - multiple assets
#
#
#     # FIXME futures tests

