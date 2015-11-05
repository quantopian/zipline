"""
Tests for SimplePipelineEngine
"""
from __future__ import division
from collections import OrderedDict
from unittest import TestCase
from itertools import product

from numpy import (
    array,
    full,
    nan,
    tile,
    zeros,
    float32,
    concatenate,
)
from pandas import (
    DataFrame,
    date_range,
    Int64Index,
    MultiIndex,
    rolling_mean,
    Series,
    Timestamp,
)
from pandas.compat.chainmap import ChainMap
from pandas.util.testing import assert_frame_equal
from six import iteritems, itervalues
from testfixtures import TempDirectory

from zipline.pipeline.loaders.synthetic import (
    ConstantLoader,
    NullAdjustmentReader,
    SyntheticDailyBarWriter,
)
from zipline.data.us_equity_pricing import BcolzDailyBarReader
from zipline.finance.trading import TradingEnvironment
from zipline.pipeline import Pipeline
from zipline.pipeline.data import USEquityPricing, DataSet, Column
from zipline.pipeline.loaders.frame import DataFrameLoader, MULTIPLY
from zipline.pipeline.loaders.equity_pricing_loader import (
    USEquityPricingLoader,
)
from zipline.pipeline.engine import SimplePipelineEngine
from zipline.pipeline import CustomFactor
from zipline.pipeline.factors import (
    MaxDrawdown,
    SimpleMovingAverage,
)
from zipline.utils.memoize import lazyval
from zipline.utils.test_utils import (
    make_rotating_equity_info,
    make_simple_equity_info,
    product_upper_triangle,
    check_arrays,
)


class RollingSumDifference(CustomFactor):
    window_length = 3
    inputs = [USEquityPricing.open, USEquityPricing.close]

    def compute(self, today, assets, out, open, close):
        out[:] = (open - close).sum(axis=0)


class AssetID(CustomFactor):
    """
    CustomFactor that returns the AssetID of each asset.

    Useful for providing a Factor that produces a different value for each
    asset.
    """
    window_length = 1
    # HACK: We currently decide whether to load or compute a Term based on the
    # length of its inputs.  This means we have to provide a dummy input.
    inputs = [USEquityPricing.close]

    def compute(self, today, assets, out, close):
        out[:] = assets


def assert_multi_index_is_product(testcase, index, *levels):
    """Assert that a MultiIndex contains the product of `*levels`."""
    testcase.assertIsInstance(
        index, MultiIndex, "%s is not a MultiIndex" % index
    )
    testcase.assertEqual(set(index), set(product(*levels)))


class ColumnArgs(tuple):
    """A tuple of Columns that defines equivalence based on the order of the
    columns' DataSets, instead of the columns themselves. This is used when
    comparing the columns passed to a loader's load_adjusted_array method,
    since we want to assert that they are ordered by DataSet.
    """
    def __new__(cls, *cols):
        return super(ColumnArgs, cls).__new__(cls, cols)

    @classmethod
    def sorted_by_ds(cls, *cols):
        return cls(*sorted(cols, key=lambda col: col.dataset))

    def by_ds(self):
        return tuple(col.dataset for col in self)

    def __eq__(self, other):
        return set(self) == set(other) and self.by_ds() == other.by_ds()

    def __hash__(self):
        return hash(frozenset(self))


class RecordingConstantLoader(ConstantLoader):
    def __init__(self, *args, **kwargs):
        super(RecordingConstantLoader, self).__init__(*args, **kwargs)

        self.load_calls = []

    def load_adjusted_array(self, columns, dates, assets, mask):
        self.load_calls.append(ColumnArgs(*columns))

        return super(RecordingConstantLoader, self).load_adjusted_array(
            columns, dates, assets, mask,
        )


class RollingSumSum(CustomFactor):
    def compute(self, today, assets, out, *inputs):
        assert len(self.inputs) == len(inputs)
        out[:] = sum(inputs).sum(axis=0)


class ConstantInputTestCase(TestCase):

    def setUp(self):
        self.constants = {
            # Every day, assume every stock starts at 2, goes down to 1,
            # goes up to 4, and finishes at 3.
            USEquityPricing.low: 1,
            USEquityPricing.open: 2,
            USEquityPricing.close: 3,
            USEquityPricing.high: 4,
        }
        self.assets = [1, 2, 3]
        self.dates = date_range('2014-01', '2014-03', freq='D', tz='UTC')
        self.loader = ConstantLoader(
            constants=self.constants,
            dates=self.dates,
            assets=self.assets,
        )

        self.asset_info = make_simple_equity_info(
            self.assets,
            start_date=self.dates[0],
            end_date=self.dates[-1],
        )
        environment = TradingEnvironment()
        environment.write_data(equities_df=self.asset_info)
        self.asset_finder = environment.asset_finder

    def test_bad_dates(self):
        loader = self.loader
        engine = SimplePipelineEngine(
            lambda column: loader, self.dates, self.asset_finder,
        )

        p = Pipeline()

        msg = "start_date must be before or equal to end_date .*"
        with self.assertRaisesRegexp(ValueError, msg):
            engine.run_pipeline(p, self.dates[2], self.dates[1])

    def test_same_day_pipeline(self):
        loader = self.loader
        engine = SimplePipelineEngine(
            lambda column: loader, self.dates, self.asset_finder,
        )
        factor = AssetID()
        asset = self.assets[0]
        p = Pipeline(columns={'f': factor}, screen=factor <= asset)

        # The crux of this is that when we run the pipeline for a single day
        #  (i.e. start and end dates are the same) we should accurately get
        # data for the day prior.
        result = engine.run_pipeline(p, self.dates[1], self.dates[1])
        self.assertEqual(result['f'][0], 1.0)

    def test_screen(self):
        loader = self.loader
        finder = self.asset_finder
        assets = array(self.assets)
        engine = SimplePipelineEngine(
            lambda column: loader, self.dates, self.asset_finder,
        )
        num_dates = 5
        dates = self.dates[10:10 + num_dates]

        factor = AssetID()
        for asset in assets:
            p = Pipeline(columns={'f': factor}, screen=factor <= asset)
            result = engine.run_pipeline(p, dates[0], dates[-1])

            expected_sids = assets[assets <= asset]
            expected_assets = finder.retrieve_all(expected_sids)
            expected_result = DataFrame(
                index=MultiIndex.from_product([dates, expected_assets]),
                data=tile(expected_sids.astype(float), [len(dates)]),
                columns=['f'],
            )

            assert_frame_equal(result, expected_result)

    def test_single_factor(self):
        loader = self.loader
        finder = self.asset_finder
        assets = self.assets
        engine = SimplePipelineEngine(
            lambda column: loader, self.dates, self.asset_finder,
        )
        result_shape = (num_dates, num_assets) = (5, len(assets))
        dates = self.dates[10:10 + num_dates]

        factor = RollingSumDifference()
        expected_result = -factor.window_length

        # Since every asset will pass the screen, these should be equivalent.
        pipelines = [
            Pipeline(columns={'f': factor}),
            Pipeline(
                columns={'f': factor},
                screen=factor.eq(expected_result),
            ),
        ]

        for p in pipelines:
            result = engine.run_pipeline(p, dates[0], dates[-1])
            self.assertEqual(set(result.columns), {'f'})
            assert_multi_index_is_product(
                self, result.index, dates, finder.retrieve_all(assets)
            )

            check_arrays(
                result['f'].unstack().values,
                full(result_shape, expected_result),
            )

    def test_multiple_rolling_factors(self):

        loader = self.loader
        finder = self.asset_finder
        assets = self.assets
        engine = SimplePipelineEngine(
            lambda column: loader, self.dates, self.asset_finder,
        )
        shape = num_dates, num_assets = (5, len(assets))
        dates = self.dates[10:10 + num_dates]

        short_factor = RollingSumDifference(window_length=3)
        long_factor = RollingSumDifference(window_length=5)
        high_factor = RollingSumDifference(
            window_length=3,
            inputs=[USEquityPricing.open, USEquityPricing.high],
        )

        pipeline = Pipeline(
            columns={
                'short': short_factor,
                'long': long_factor,
                'high': high_factor,
            }
        )
        results = engine.run_pipeline(pipeline, dates[0], dates[-1])

        self.assertEqual(set(results.columns), {'short', 'high', 'long'})
        assert_multi_index_is_product(
            self, results.index, dates, finder.retrieve_all(assets)
        )

        # row-wise sum over an array whose values are all (1 - 2)
        check_arrays(
            results['short'].unstack().values,
            full(shape, -short_factor.window_length),
        )
        check_arrays(
            results['long'].unstack().values,
            full(shape, -long_factor.window_length),
        )
        # row-wise sum over an array whose values are all (1 - 3)
        check_arrays(
            results['high'].unstack().values,
            full(shape, -2 * high_factor.window_length),
        )

    def test_numeric_factor(self):
        constants = self.constants
        loader = self.loader
        engine = SimplePipelineEngine(
            lambda column: loader, self.dates, self.asset_finder,
        )
        num_dates = 5
        dates = self.dates[10:10 + num_dates]
        high, low = USEquityPricing.high, USEquityPricing.low
        open, close = USEquityPricing.open, USEquityPricing.close

        high_minus_low = RollingSumDifference(inputs=[high, low])
        open_minus_close = RollingSumDifference(inputs=[open, close])
        avg = (high_minus_low + open_minus_close) / 2

        results = engine.run_pipeline(
            Pipeline(
                columns={
                    'high_low': high_minus_low,
                    'open_close': open_minus_close,
                    'avg': avg,
                },
            ),
            dates[0],
            dates[-1],
        )

        high_low_result = results['high_low'].unstack()
        expected_high_low = 3.0 * (constants[high] - constants[low])
        assert_frame_equal(
            high_low_result,
            DataFrame(expected_high_low, index=dates, columns=self.assets),
        )

        open_close_result = results['open_close'].unstack()
        expected_open_close = 3.0 * (constants[open] - constants[close])
        assert_frame_equal(
            open_close_result,
            DataFrame(expected_open_close, index=dates, columns=self.assets),
        )

        avg_result = results['avg'].unstack()
        expected_avg = (expected_high_low + expected_open_close) / 2.0
        assert_frame_equal(
            avg_result,
            DataFrame(expected_avg, index=dates, columns=self.assets),
        )

    def test_rolling_and_nonrolling(self):
        open_ = USEquityPricing.open
        close = USEquityPricing.close
        volume = USEquityPricing.volume

        # Test for thirty days up to the last day that we think all
        # the assets existed.
        dates_to_test = self.dates[-30:]

        constants = {open_: 1, close: 2, volume: 3}
        loader = ConstantLoader(
            constants=constants,
            dates=self.dates,
            assets=self.assets,
        )
        engine = SimplePipelineEngine(
            lambda column: loader, self.dates, self.asset_finder,
        )

        sumdiff = RollingSumDifference()

        result = engine.run_pipeline(
            Pipeline(
                columns={
                    'sumdiff': sumdiff,
                    'open': open_.latest,
                    'close': close.latest,
                    'volume': volume.latest,
                },
            ),
            dates_to_test[0],
            dates_to_test[-1]
        )
        self.assertIsNotNone(result)
        self.assertEqual(
            {'sumdiff', 'open', 'close', 'volume'},
            set(result.columns)
        )

        result_index = self.assets * len(dates_to_test)
        result_shape = (len(result_index),)
        check_arrays(
            result['sumdiff'],
            Series(index=result_index, data=full(result_shape, -3)),
        )

        for name, const in [('open', 1), ('close', 2), ('volume', 3)]:
            check_arrays(
                result[name],
                Series(index=result_index, data=full(result_shape, const)),
            )

    def test_loader_given_multiple_columns(self):

        class Loader1DataSet1(DataSet):
            col1 = Column(float32)
            col2 = Column(float32)

        class Loader1DataSet2(DataSet):
            col1 = Column(float32)
            col2 = Column(float32)

        class Loader2DataSet(DataSet):
            col1 = Column(float32)
            col2 = Column(float32)

        constants1 = {Loader1DataSet1.col1: 1,
                      Loader1DataSet1.col2: 2,
                      Loader1DataSet2.col1: 3,
                      Loader1DataSet2.col2: 4}
        loader1 = RecordingConstantLoader(constants=constants1,
                                          dates=self.dates,
                                          assets=self.assets)
        constants2 = {Loader2DataSet.col1: 5,
                      Loader2DataSet.col2: 6}
        loader2 = RecordingConstantLoader(constants=constants2,
                                          dates=self.dates,
                                          assets=self.assets)

        engine = SimplePipelineEngine(
            lambda column:
            loader2 if column.dataset == Loader2DataSet else loader1,
            self.dates, self.asset_finder,
        )

        pipe_col1 = RollingSumSum(inputs=[Loader1DataSet1.col1,
                                          Loader1DataSet2.col1,
                                          Loader2DataSet.col1],
                                  window_length=2)

        pipe_col2 = RollingSumSum(inputs=[Loader1DataSet1.col2,
                                          Loader1DataSet2.col2,
                                          Loader2DataSet.col2],
                                  window_length=3)

        pipe_col3 = RollingSumSum(inputs=[Loader2DataSet.col1],
                                  window_length=3)

        columns = OrderedDict([
            ('pipe_col1', pipe_col1),
            ('pipe_col2', pipe_col2),
            ('pipe_col3', pipe_col3),
        ])
        result = engine.run_pipeline(
            Pipeline(columns=columns),
            self.dates[2],  # index is >= the largest window length - 1
            self.dates[-1]
        )
        min_window = min(pip_col.window_length
                         for pip_col in itervalues(columns))
        col_to_val = ChainMap(constants1, constants2)
        vals = {name: (sum(col_to_val[col] for col in pipe_col.inputs)
                       * pipe_col.window_length)
                for name, pipe_col in iteritems(columns)}

        index = MultiIndex.from_product([self.dates[2:], self.assets])
        expected = DataFrame(
            data={col:
                  concatenate((
                      full((columns[col].window_length - min_window)
                           * index.levshape[1],
                           nan),
                      full((index.levshape[0]
                            - (columns[col].window_length - min_window))
                           * index.levshape[1],
                           val)))
                  for col, val in iteritems(vals)},
            index=index,
            columns=columns)

        assert_frame_equal(result, expected)

        self.assertEqual(set(loader1.load_calls),
                         {ColumnArgs.sorted_by_ds(Loader1DataSet1.col1,
                                                  Loader1DataSet2.col1),
                          ColumnArgs.sorted_by_ds(Loader1DataSet1.col2,
                                                  Loader1DataSet2.col2)})
        self.assertEqual(set(loader2.load_calls),
                         {ColumnArgs.sorted_by_ds(Loader2DataSet.col1,
                                                  Loader2DataSet.col2)})


class FrameInputTestCase(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.env = TradingEnvironment()
        day = cls.env.trading_day

        cls.assets = Int64Index([1, 2, 3])
        cls.dates = date_range(
            '2015-01-01',
            '2015-01-31',
            freq=day,
            tz='UTC',
        )

        asset_info = make_simple_equity_info(
            cls.assets,
            start_date=cls.dates[0],
            end_date=cls.dates[-1],
        )
        cls.env.write_data(equities_df=asset_info)
        cls.asset_finder = cls.env.asset_finder

    @classmethod
    def tearDownClass(cls):
        del cls.env
        del cls.asset_finder

    @lazyval
    def base_mask(self):
        return self.make_frame(True)

    def make_frame(self, data):
        return DataFrame(data, columns=self.assets, index=self.dates)

    def test_compute_with_adjustments(self):
        dates, assets = self.dates, self.assets
        low, high = USEquityPricing.low, USEquityPricing.high
        apply_idxs = [3, 10, 16]

        def apply_date(idx, offset=0):
            return dates[apply_idxs[idx] + offset]

        adjustments = DataFrame.from_records(
            [
                dict(
                    kind=MULTIPLY,
                    sid=assets[1],
                    value=2.0,
                    start_date=None,
                    end_date=apply_date(0, offset=-1),
                    apply_date=apply_date(0),
                ),
                dict(
                    kind=MULTIPLY,
                    sid=assets[1],
                    value=3.0,
                    start_date=None,
                    end_date=apply_date(1, offset=-1),
                    apply_date=apply_date(1),
                ),
                dict(
                    kind=MULTIPLY,
                    sid=assets[1],
                    value=5.0,
                    start_date=None,
                    end_date=apply_date(2, offset=-1),
                    apply_date=apply_date(2),
                ),
            ]
        )
        low_base = DataFrame(self.make_frame(30.0))
        low_loader = DataFrameLoader(low, low_base.copy(), adjustments=None)

        # Pre-apply inverse of adjustments to the baseline.
        high_base = DataFrame(self.make_frame(30.0))
        high_base.iloc[:apply_idxs[0], 1] /= 2.0
        high_base.iloc[:apply_idxs[1], 1] /= 3.0
        high_base.iloc[:apply_idxs[2], 1] /= 5.0

        high_loader = DataFrameLoader(high, high_base, adjustments)

        engine = SimplePipelineEngine(
            {low: low_loader, high: high_loader}.__getitem__,
            self.dates,
            self.asset_finder,
        )

        for window_length in range(1, 4):
            low_mavg = SimpleMovingAverage(
                inputs=[USEquityPricing.low],
                window_length=window_length,
            )
            high_mavg = SimpleMovingAverage(
                inputs=[USEquityPricing.high],
                window_length=window_length,
            )
            bounds = product_upper_triangle(range(window_length, len(dates)))
            for start, stop in bounds:
                results = engine.run_pipeline(
                    Pipeline(
                        columns={'low': low_mavg, 'high': high_mavg}
                    ),
                    dates[start],
                    dates[stop],
                )
                self.assertEqual(set(results.columns), {'low', 'high'})
                iloc_bounds = slice(start, stop + 1)  # +1 to include end date

                low_results = results.unstack()['low']
                assert_frame_equal(low_results, low_base.iloc[iloc_bounds])

                high_results = results.unstack()['high']
                assert_frame_equal(high_results, high_base.iloc[iloc_bounds])


class SyntheticBcolzTestCase(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.first_asset_start = Timestamp('2015-04-01', tz='UTC')
        cls.env = TradingEnvironment()
        cls.trading_day = day = cls.env.trading_day
        cls.calendar = date_range('2015', '2015-08', tz='UTC', freq=day)

        cls.asset_info = make_rotating_equity_info(
            num_assets=6,
            first_start=cls.first_asset_start,
            frequency=day,
            periods_between_starts=4,
            asset_lifetime=8,
        )
        cls.last_asset_end = cls.asset_info['end_date'].max()
        cls.all_assets = cls.asset_info.index

        cls.env.write_data(equities_df=cls.asset_info)
        cls.finder = cls.env.asset_finder

        cls.temp_dir = TempDirectory()
        cls.temp_dir.create()

        try:
            cls.writer = SyntheticDailyBarWriter(
                asset_info=cls.asset_info[['start_date', 'end_date']],
                calendar=cls.calendar,
            )
            table = cls.writer.write(
                cls.temp_dir.getpath('testdata.bcolz'),
                cls.calendar,
                cls.all_assets,
            )

            cls.pipeline_loader = USEquityPricingLoader(
                BcolzDailyBarReader(table),
                NullAdjustmentReader(),
            )
        except:
            cls.temp_dir.cleanup()
            raise

    @classmethod
    def tearDownClass(cls):
        del cls.env
        cls.temp_dir.cleanup()

    def write_nans(self, df):
        """
        Write nans to the locations in data corresponding to the (date, asset)
        pairs for which we wouldn't have data for `asset` on `date` in a
        backtest.

        Parameters
        ----------
        df : pd.DataFrame
            A DataFrame with a DatetimeIndex as index and an object index of
            Assets as columns.

        This means that we write nans for dates after an asset's end_date and
        **on or before** an asset's start_date.  The assymetry here is because
        of the fact that, on the morning of an asset's first date, we haven't
        yet seen any trades for that asset, so we wouldn't be able to show any
        useful data to the user.
        """
        # Mask out with nans all the dates on which each asset didn't exist
        index = df.index
        min_, max_ = index[[0, -1]]
        for asset in df.columns:
            if asset.start_date >= min_:
                start = index.get_loc(asset.start_date, method='bfill')
                df.loc[:start + 1, asset] = nan  # +1 to overwrite start_date
            if asset.end_date <= max_:
                end = index.get_loc(asset.end_date)
                df.ix[end + 1:, asset] = nan  # +1 to *not* overwrite end_date

    def test_SMA(self):
        engine = SimplePipelineEngine(
            lambda column: self.pipeline_loader,
            self.env.trading_days,
            self.finder,
        )
        window_length = 5
        assets = self.all_assets
        dates = date_range(
            self.first_asset_start + self.trading_day,
            self.last_asset_end,
            freq=self.trading_day,
        )
        dates_to_test = dates[window_length:]

        SMA = SimpleMovingAverage(
            inputs=(USEquityPricing.close,),
            window_length=window_length,
        )

        results = engine.run_pipeline(
            Pipeline(columns={'sma': SMA}),
            dates_to_test[0],
            dates_to_test[-1],
        )

        # Shift back the raw inputs by a trading day because we expect our
        # computed results to be computed using values anchored on the
        # **previous** day's data.
        expected_raw = rolling_mean(
            self.writer.expected_values_2d(
                dates - self.trading_day, assets, 'close',
            ),
            window_length,
            min_periods=1,
        )

        expected = DataFrame(
            # Truncate off the extra rows needed to compute the SMAs.
            expected_raw[window_length:],
            index=dates_to_test,  # dates_to_test is dates[window_length:]
            columns=self.finder.retrieve_all(assets),
        )
        self.write_nans(expected)
        result = results['sma'].unstack()
        assert_frame_equal(result, expected)

    def test_drawdown(self):
        # The monotonically-increasing data produced by SyntheticDailyBarWriter
        # exercises two pathological cases for MaxDrawdown.  The actual
        # computed results are pretty much useless (everything is either NaN)
        # or zero, but verifying we correctly handle those corner cases is
        # valuable.
        engine = SimplePipelineEngine(
            lambda column: self.pipeline_loader,
            self.env.trading_days,
            self.finder,
        )
        window_length = 5
        assets = self.all_assets
        dates = date_range(
            self.first_asset_start + self.trading_day,
            self.last_asset_end,
            freq=self.trading_day,
        )
        dates_to_test = dates[window_length:]

        drawdown = MaxDrawdown(
            inputs=(USEquityPricing.close,),
            window_length=window_length,
        )

        results = engine.run_pipeline(
            Pipeline(columns={'drawdown': drawdown}),
            dates_to_test[0],
            dates_to_test[-1],
        )

        # We expect NaNs when the asset was undefined, otherwise 0 everywhere,
        # since the input is always increasing.
        expected = DataFrame(
            data=zeros((len(dates_to_test), len(assets)), dtype=float),
            index=dates_to_test,
            columns=self.finder.retrieve_all(assets),
        )
        self.write_nans(expected)
        result = results['drawdown'].unstack()

        assert_frame_equal(expected, result)
