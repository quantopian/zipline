import warnings

from mock import patch
import numpy as np
import pandas as pd
from pandas.core.common import PerformanceWarning

from zipline import TradingAlgorithm
from zipline.finance.trading import SimulationParameters
from zipline.testing import (
    MockDailyBarReader,
    create_daily_df_for_asset,
    create_minute_df_for_asset,
    str_to_seconds,
)
from zipline.testing.fixtures import (
    WithCreateBarData,
    WithDataPortal,
    WithSimParams,
    ZiplineTestCase,
)
from zipline.zipline_warnings import ZiplineDeprecationWarning

simple_algo = """
from zipline.api import sid, order
def initialize(context):
    pass

def handle_data(context, data):
    assert sid(1) in data
    assert sid(2) in data
    assert len(data) == 3
    for asset in data:
        pass
"""

history_algo = """
from zipline.api import sid, history

def initialize(context):
    context.sid1 = sid(1)

def handle_data(context, data):
    context.history_window = history(5, "1m", "volume")
"""

history_bts_algo = """
from zipline.api import sid, history, record

def initialize(context):
    context.sid3 = sid(3)
    context.num_bts = 0

def before_trading_start(context, data):
    context.num_bts += 1

    # Get history at the second BTS (beginning of second day)
    if context.num_bts == 2:
        record(history=history(5, "1m", "volume"))

def handle_data(context, data):
    pass
"""

simple_transforms_algo = """
from zipline.api import sid
def initialize(context):
    context.count = 0

def handle_data(context, data):
    if context.count == 2:
        context.mavg = data[sid(1)].mavg(5)
        context.vwap = data[sid(1)].vwap(5)
        context.stddev = data[sid(1)].stddev(5)
        context.returns = data[sid(1)].returns()

    context.count += 1
"""

manipulation_algo = """
def initialize(context):
    context.asset1 = sid(1)
    context.asset2 = sid(2)

def handle_data(context, data):
    assert len(data) == 2
    assert len(data.keys()) == 2
    assert context.asset1 in data.keys()
    assert context.asset2 in data.keys()
"""

sid_accessor_algo = """
from zipline.api import sid

def initialize(context):
    context.asset1 = sid(1)

def handle_data(context,data):
    assert data[sid(1)].sid == context.asset1
    assert data[sid(1)]["sid"] == context.asset1
"""

data_items_algo = """
from zipline.api import sid

def initialize(context):
    context.asset1 = sid(1)
    context.asset2 = sid(2)

def handle_data(context, data):
    iter_list = list(data.iteritems())
    items_list = data.items()
    assert iter_list == items_list
"""


class TestAPIShim(WithCreateBarData,
                  WithDataPortal,
                  WithSimParams,
                  ZiplineTestCase,
                  ):
    START_DATE = pd.Timestamp("2016-01-05", tz='UTC')
    END_DATE = pd.Timestamp("2016-01-28", tz='UTC')
    SIM_PARAMS_DATA_FREQUENCY = 'minute'

    sids = ASSET_FINDER_EQUITY_SIDS = 1, 2, 3

    @classmethod
    def make_equity_minute_bar_data(cls):
        for sid in cls.sids:
            yield sid, create_minute_df_for_asset(
                cls.trading_calendar,
                cls.SIM_PARAMS_START,
                cls.SIM_PARAMS_END,
            )

    @classmethod
    def make_equity_daily_bar_data(cls):
        for sid in cls.sids:
            yield sid, create_daily_df_for_asset(
                cls.trading_calendar,
                cls.SIM_PARAMS_START,
                cls.SIM_PARAMS_END,
            )

    @classmethod
    def make_splits_data(cls):
        return pd.DataFrame([
            {
                'effective_date': str_to_seconds('2016-01-06'),
                'ratio': 0.5,
                'sid': 3,
            }
        ])

    @classmethod
    def make_adjustment_writer_equity_daily_bar_reader(cls):
        return MockDailyBarReader()

    @classmethod
    def init_class_fixtures(cls):
        super(TestAPIShim, cls).init_class_fixtures()

        cls.asset1 = cls.env.asset_finder.retrieve_asset(1)
        cls.asset2 = cls.env.asset_finder.retrieve_asset(2)
        cls.asset3 = cls.env.asset_finder.retrieve_asset(3)

    def create_algo(self, code, filename=None, sim_params=None):
        if sim_params is None:
            sim_params = self.sim_params

        return TradingAlgorithm(
            script=code,
            sim_params=sim_params,
            env=self.env,
            algo_filename=filename
        )

    def test_old_new_data_api_paths(self):
        """
        Test that the new and old data APIs hit the same code paths.

        We want to ensure that the old data API(data[sid(N)].field and
        similar)  and the new data API(data.current(sid(N), field) and
        similar) hit the same code paths on the DataPortal.
        """
        test_start_minute = self.trading_calendar.minutes_for_session(
            self.sim_params.sessions[0]
        )[1]
        test_end_minute = self.trading_calendar.minutes_for_session(
            self.sim_params.sessions[0]
        )[-1]
        bar_data = self.create_bardata(
            lambda: test_end_minute,
        )
        ohlcvp_fields = [
            "open",
            "high",
            "low"
            "close",
            "volume",
            "price",
        ]
        spot_value_meth = 'zipline.data.data_portal.DataPortal.get_spot_value'

        def assert_get_spot_value_called(fun, field):
            """
            Assert that get_spot_value was called during the execution of fun.

            Takes in a function fun and a string field.
            """
            with patch(spot_value_meth) as gsv:
                fun()
                gsv.assert_called_with(
                    self.asset1,
                    field,
                    test_end_minute,
                    'minute'
                )
        # Ensure that data.current(sid(n), field) has the same behaviour as
        # data[sid(n)].field.
        for field in ohlcvp_fields:
            assert_get_spot_value_called(
                lambda: getattr(bar_data[self.asset1], field),
                field,
            )
            assert_get_spot_value_called(
                lambda: bar_data.current(self.asset1, field),
                field,
            )

        history_meth = 'zipline.data.data_portal.DataPortal.get_history_window'

        def assert_get_history_window_called(fun, is_legacy):
            """
            Assert that get_history_window was called during fun().

            Takes in a function fun and a boolean is_legacy.
            """
            with patch(history_meth) as ghw:
                fun()
                # Slightly hacky, but done to get around the fact that
                # history( explicitly passes an ffill param as the last arg,
                # while data.history doesn't.
                if is_legacy:
                    ghw.assert_called_with(
                        [self.asset1, self.asset2, self.asset3],
                        test_end_minute,
                        5,
                        "1m",
                        "volume",
                        True
                    )
                else:
                    ghw.assert_called_with(
                        [self.asset1, self.asset2, self.asset3],
                        test_end_minute,
                        5,
                        "1m",
                        "volume",
                    )

        test_sim_params = SimulationParameters(
            start_session=test_start_minute,
            end_session=test_end_minute,
            data_frequency="minute",
            trading_calendar=self.trading_calendar,
        )

        history_algorithm = self.create_algo(
            history_algo,
            sim_params=test_sim_params
        )
        assert_get_history_window_called(
            lambda: history_algorithm.run(self.data_portal),
            is_legacy=True
        )
        assert_get_history_window_called(
            lambda: bar_data.history(
                [self.asset1, self.asset2, self.asset3],
                "volume",
                5,
                "1m"
            ),
            is_legacy=False
        )

    def test_sid_accessor(self):
        """
        Test that we maintain backwards compat for sid access on a data object.

        We want to support both data[sid(24)].sid, as well as
        data[sid(24)]["sid"]. Since these are deprecated and will eventually
        cease to be supported, we also want to assert that we're seeing a
        deprecation warning.
        """
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("ignore", PerformanceWarning)
            warnings.simplefilter("default", ZiplineDeprecationWarning)
            algo = self.create_algo(sid_accessor_algo)
            algo.run(self.data_portal)

            # Since we're already raising a warning on doing data[sid(x)],
            # we don't want to raise an extra warning on data[sid(x)].sid.
            self.assertEqual(2, len(w))

            # Check that both the warnings raised were in fact
            # ZiplineDeprecationWarnings
            for warning in w:
                self.assertEqual(
                    ZiplineDeprecationWarning,
                    warning.category
                )
                self.assertEqual(
                    "`data[sid(N)]` is deprecated. Use `data.current`.",
                    str(warning.message)
                )

    def test_data_items(self):
        """
        Test that we maintain backwards compat for data.[items | iteritems].

        We also want to assert that we warn that iterating over the assets
        in `data` is deprecated.
        """
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("ignore", PerformanceWarning)
            warnings.simplefilter("default", ZiplineDeprecationWarning)
            algo = self.create_algo(data_items_algo)
            algo.run(self.data_portal)

            self.assertEqual(4, len(w))

            for idx, warning in enumerate(w):
                self.assertEqual(
                    ZiplineDeprecationWarning,
                    warning.category
                )
                if idx % 2 == 0:
                    self.assertEqual(
                        "Iterating over the assets in `data` is deprecated.",
                        str(warning.message)
                    )
                else:
                    self.assertEqual(
                        "`data[sid(N)]` is deprecated. Use `data.current`.",
                        str(warning.message)
                    )

    def test_iterate_data(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("ignore", PerformanceWarning)
            warnings.simplefilter("default", ZiplineDeprecationWarning)

            algo = self.create_algo(simple_algo)
            algo.run(self.data_portal)

            self.assertEqual(4, len(w))

            line_nos = [warning.lineno for warning in w]
            self.assertEqual(4, len(set(line_nos)))

            for idx, warning in enumerate(w):
                self.assertEqual(ZiplineDeprecationWarning,
                                 warning.category)

                self.assertEqual("<string>", warning.filename)
                self.assertEqual(line_nos[idx], warning.lineno)

                if idx < 2:
                    self.assertEqual(
                        "Checking whether an asset is in data is deprecated.",
                        str(warning.message)
                    )
                else:
                    self.assertEqual(
                        "Iterating over the assets in `data` is deprecated.",
                        str(warning.message)
                    )

    def test_history(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("ignore", PerformanceWarning)
            warnings.simplefilter("default", ZiplineDeprecationWarning)

            sim_params = self.sim_params.create_new(
                self.sim_params.sessions[1],
                self.sim_params.end_session
            )

            algo = self.create_algo(history_algo,
                                    sim_params=sim_params)
            algo.run(self.data_portal)

            self.assertEqual(1, len(w))
            self.assertEqual(ZiplineDeprecationWarning, w[0].category)
            self.assertEqual("<string>", w[0].filename)
            self.assertEqual(8, w[0].lineno)
            self.assertEqual("The `history` method is deprecated.  Use "
                             "`data.history` instead.", str(w[0].message))

    def test_old_new_history_bts_paths(self):
        """
        Tests that calling history in before_trading_start gets us the correct
        values, which involves 1) calling data_portal.get_history_window as of
        the previous market minute, 2) getting adjustments between the previous
        market minute and the current time, and 3) applying those adjustments
        """
        algo = self.create_algo(history_bts_algo)
        algo.run(self.data_portal)

        expected_vol_without_split = np.arange(386, 391) * 100
        expected_vol_with_split = np.arange(386, 391) * 200

        window = algo.recorded_vars['history']
        np.testing.assert_array_equal(window[self.asset1].values,
                                      expected_vol_without_split)
        np.testing.assert_array_equal(window[self.asset2].values,
                                      expected_vol_without_split)
        np.testing.assert_array_equal(window[self.asset3].values,
                                      expected_vol_with_split)

    def test_simple_transforms(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("ignore", PerformanceWarning)
            warnings.simplefilter("default", ZiplineDeprecationWarning)

            sim_params = SimulationParameters(
                start_session=self.sim_params.sessions[8],
                end_session=self.sim_params.sessions[-1],
                data_frequency="minute",
                trading_calendar=self.trading_calendar,
            )

            algo = self.create_algo(simple_transforms_algo,
                                    sim_params=sim_params)
            algo.run(self.data_portal)

            self.assertEqual(8, len(w))
            transforms = ["mavg", "vwap", "stddev", "returns"]

            for idx, line_no in enumerate(range(8, 12)):
                warning1 = w[idx * 2]
                warning2 = w[(idx * 2) + 1]

                self.assertEqual("<string>", warning1.filename)
                self.assertEqual("<string>", warning2.filename)

                self.assertEqual(line_no, warning1.lineno)
                self.assertEqual(line_no, warning2.lineno)

                self.assertEqual("`data[sid(N)]` is deprecated. Use "
                                 "`data.current`.",
                                 str(warning1.message))
                self.assertEqual("The `{0}` method is "
                                 "deprecated.".format(transforms[idx]),
                                 str(warning2.message))

            # now verify the transform values
            # minute price
            # 2016-01-11 14:31:00+00:00    1561
            # ...
            # 2016-01-14 20:59:00+00:00    3119
            # 2016-01-14 21:00:00+00:00    3120
            # 2016-01-15 14:31:00+00:00    3121
            # 2016-01-15 14:32:00+00:00    3122
            # 2016-01-15 14:33:00+00:00    3123

            # volume
            # 2016-01-11 14:31:00+00:00    156100
            # ...
            # 2016-01-14 20:59:00+00:00    311900
            # 2016-01-14 21:00:00+00:00    312000
            # 2016-01-15 14:31:00+00:00    312100
            # 2016-01-15 14:32:00+00:00    312200
            # 2016-01-15 14:33:00+00:00    312300

            # daily price (last day built with minute data)
            # 2016-01-14 00:00:00+00:00       9
            # 2016-01-15 00:00:00+00:00    3123

            # mavg = average of all the prices = (1561 + 3123) / 2 = 2342
            # vwap = sum(price * volume) / sum(volumes)
            #      = 889119531400.0 / 366054600.0
            #      = 2428.9259891830343
            # stddev = stddev(price, ddof=1) = 451.3435498597493
            # returns = (todayprice - yesterdayprice) / yesterdayprice
            #         = (3123 - 9) / 9 = 346
            self.assertEqual(2342, algo.mavg)
            self.assertAlmostEqual(2428.92599, algo.vwap, places=5)
            self.assertAlmostEqual(451.34355, algo.stddev, places=5)
            self.assertAlmostEqual(346, algo.returns)

    def test_manipulation(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("ignore", PerformanceWarning)
            warnings.simplefilter("default", ZiplineDeprecationWarning)

            algo = self.create_algo(simple_algo)
            algo.run(self.data_portal)

            self.assertEqual(4, len(w))

            for idx, warning in enumerate(w):
                self.assertEqual("<string>", warning.filename)
                self.assertEqual(7 + idx, warning.lineno)

                if idx < 2:
                    self.assertEqual("Checking whether an asset is in data is "
                                     "deprecated.",
                                     str(warning.message))
                else:
                    self.assertEqual("Iterating over the assets in `data` is "
                                     "deprecated.",
                                     str(warning.message))
