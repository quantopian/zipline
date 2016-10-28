"""
Tests for SimplePipelineEngine
"""
from __future__ import division
from collections import OrderedDict
from itertools import product
from operator import add, sub

from nose_parameterized import parameterized
from numpy import (
    arange,
    array,
    concatenate,
    float32,
    float64,
    full,
    full_like,
    log,
    nan,
    tile,
    where,
    zeros,
)
from numpy.testing import assert_almost_equal
from pandas import (
    Categorical,
    DataFrame,
    date_range,
    Int64Index,
    MultiIndex,
    Series,
    Timestamp,
)
from pandas.compat.chainmap import ChainMap
from pandas.util.testing import assert_frame_equal
from six import iteritems, itervalues
from toolz import merge

from zipline.assets.synthetic import make_rotating_equity_info
from zipline.errors import NoFurtherDataError
from zipline.lib.adjustment import MULTIPLY
from zipline.lib.labelarray import LabelArray
from zipline.pipeline import CustomFactor, Pipeline
from zipline.pipeline.data import Column, DataSet, USEquityPricing
from zipline.pipeline.data.testing import TestingDataSet
from zipline.pipeline.engine import SimplePipelineEngine
from zipline.pipeline.factors import (
    AverageDollarVolume,
    EWMA,
    EWMSTD,
    ExponentialWeightedMovingAverage,
    ExponentialWeightedMovingStdDev,
    MaxDrawdown,
    SimpleMovingAverage,
)
from zipline.pipeline.loaders.equity_pricing_loader import (
    USEquityPricingLoader,
)
from zipline.pipeline.loaders.frame import DataFrameLoader
from zipline.pipeline.loaders.synthetic import (
    PrecomputedLoader,
    make_bar_data,
    expected_bar_values_2d,
)
from zipline.pipeline.sentinels import NotSpecified
from zipline.pipeline.term import InputDates
from zipline.testing import (
    AssetID,
    AssetIDPlusDay,
    ExplodingObject,
    check_arrays,
    make_alternating_boolean_array,
    make_cascading_boolean_array,
    OpenPrice,
    parameter_space,
    product_upper_triangle,
)
from zipline.testing.fixtures import (
    WithAdjustmentReader,
    WithSeededRandomPipelineEngine,
    WithTradingEnvironment,
    ZiplineTestCase,
)
from zipline.testing.predicates import assert_equal
from zipline.utils.memoize import lazyval
from zipline.utils.numpy_utils import bool_dtype, datetime64ns_dtype


class RollingSumDifference(CustomFactor):
    window_length = 3
    inputs = [USEquityPricing.open, USEquityPricing.close]

    def compute(self, today, assets, out, open, close):
        out[:] = (open - close).sum(axis=0)


class MultipleOutputs(CustomFactor):
    window_length = 1
    inputs = [USEquityPricing.open, USEquityPricing.close]
    outputs = ['open', 'close']

    def compute(self, today, assets, out, open, close):
        out.open[:] = open
        out.close[:] = close


class OpenCloseSumAndDiff(CustomFactor):
    """
    Used for testing a CustomFactor with multiple outputs operating over a non-
    trivial window length.
    """
    inputs = [USEquityPricing.open, USEquityPricing.close]

    def compute(self, today, assets, out, open, close):
        out.sum_[:] = open.sum(axis=0) + close.sum(axis=0)
        out.diff[:] = open.sum(axis=0) - close.sum(axis=0)


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


class RecordingPrecomputedLoader(PrecomputedLoader):
    def __init__(self, *args, **kwargs):
        super(RecordingPrecomputedLoader, self).__init__(*args, **kwargs)

        self.load_calls = []

    def load_adjusted_array(self, columns, dates, assets, mask):
        self.load_calls.append(ColumnArgs(*columns))

        return super(RecordingPrecomputedLoader, self).load_adjusted_array(
            columns, dates, assets, mask,
        )


class RollingSumSum(CustomFactor):
    def compute(self, today, assets, out, *inputs):
        assert len(self.inputs) == len(inputs)
        out[:] = sum(inputs).sum(axis=0)


class WithConstantInputs(WithTradingEnvironment):
    asset_ids = ASSET_FINDER_EQUITY_SIDS = 1, 2, 3, 4
    START_DATE = Timestamp('2014-01-01', tz='utc')
    END_DATE = Timestamp('2014-03-01', tz='utc')

    @classmethod
    def init_class_fixtures(cls):
        super(WithConstantInputs, cls).init_class_fixtures()
        cls.constants = {
            # Every day, assume every stock starts at 2, goes down to 1,
            # goes up to 4, and finishes at 3.
            USEquityPricing.low: 1,
            USEquityPricing.open: 2,
            USEquityPricing.close: 3,
            USEquityPricing.high: 4,
        }
        cls.dates = date_range(
            cls.START_DATE,
            cls.END_DATE,
            freq='D',
            tz='UTC',
        )
        cls.loader = PrecomputedLoader(
            constants=cls.constants,
            dates=cls.dates,
            sids=cls.asset_ids,
        )
        cls.assets = cls.asset_finder.retrieve_all(cls.asset_ids)


class ConstantInputTestCase(WithConstantInputs, ZiplineTestCase):
    def test_bad_dates(self):
        loader = self.loader
        engine = SimplePipelineEngine(
            lambda column: loader, self.dates, self.asset_finder,
        )

        p = Pipeline()

        msg = "start_date must be before or equal to end_date .*"
        with self.assertRaisesRegexp(ValueError, msg):
            engine.run_pipeline(p, self.dates[2], self.dates[1])

    def test_fail_usefully_on_insufficient_data(self):
        loader = self.loader
        engine = SimplePipelineEngine(
            lambda column: loader, self.dates, self.asset_finder,
        )

        class SomeFactor(CustomFactor):
            inputs = [USEquityPricing.close]
            window_length = 10

            def compute(self, today, assets, out, closes):
                pass

        p = Pipeline(columns={'t': SomeFactor()})

        # self.dates[9] is the earliest date we should be able to compute.
        engine.run_pipeline(p, self.dates[9], self.dates[9])

        # We shouldn't be able to compute dates[8], since we only know about 8
        # prior dates, and we need a window length of 10.
        with self.assertRaises(NoFurtherDataError):
            engine.run_pipeline(p, self.dates[8], self.dates[8])

    def test_input_dates_provided_by_default(self):
        loader = self.loader
        engine = SimplePipelineEngine(
            lambda column: loader, self.dates, self.asset_finder,
        )

        class TestFactor(CustomFactor):
            inputs = [InputDates(), USEquityPricing.close]
            window_length = 10
            dtype = datetime64ns_dtype

            def compute(self, today, assets, out, dates, closes):
                first, last = dates[[0, -1], 0]
                assert last == today.asm8
                assert len(dates) == len(closes) == self.window_length
                out[:] = first

        p = Pipeline(columns={'t': TestFactor()})
        results = engine.run_pipeline(p, self.dates[9], self.dates[10])

        # All results are the same, so just grab one column.
        column = results.unstack().iloc[:, 0].values
        check_arrays(column, self.dates[:2].values)

    def test_same_day_pipeline(self):
        loader = self.loader
        engine = SimplePipelineEngine(
            lambda column: loader, self.dates, self.asset_finder,
        )
        factor = AssetID()
        asset = self.asset_ids[0]
        p = Pipeline(columns={'f': factor}, screen=factor <= asset)

        # The crux of this is that when we run the pipeline for a single day
        #  (i.e. start and end dates are the same) we should accurately get
        # data for the day prior.
        result = engine.run_pipeline(p, self.dates[1], self.dates[1])
        self.assertEqual(result['f'][0], 1.0)

    def test_screen(self):
        loader = self.loader
        finder = self.asset_finder
        asset_ids = array(self.asset_ids)
        engine = SimplePipelineEngine(
            lambda column: loader, self.dates, self.asset_finder,
        )
        num_dates = 5
        dates = self.dates[10:10 + num_dates]

        factor = AssetID()
        for asset_id in asset_ids:
            p = Pipeline(columns={'f': factor}, screen=factor <= asset_id)
            result = engine.run_pipeline(p, dates[0], dates[-1])

            expected_sids = asset_ids[asset_ids <= asset_id]
            expected_assets = finder.retrieve_all(expected_sids)
            expected_result = DataFrame(
                index=MultiIndex.from_product([dates, expected_assets]),
                data=tile(expected_sids.astype(float), [len(dates)]),
                columns=['f'],
            )

            assert_frame_equal(result, expected_result)

    def test_single_factor(self):
        loader = self.loader
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
                self, result.index, dates, assets
            )

            check_arrays(
                result['f'].unstack().values,
                full(result_shape, expected_result, dtype=float),
            )

    def test_multiple_rolling_factors(self):

        loader = self.loader
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
            self, results.index, dates, assets
        )

        # row-wise sum over an array whose values are all (1 - 2)
        check_arrays(
            results['short'].unstack().values,
            full(shape, -short_factor.window_length, dtype=float),
        )
        check_arrays(
            results['long'].unstack().values,
            full(shape, -long_factor.window_length, dtype=float),
        )
        # row-wise sum over an array whose values are all (1 - 3)
        check_arrays(
            results['high'].unstack().values,
            full(shape, -2 * high_factor.window_length, dtype=float),
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

    def test_masked_factor(self):
        """
        Test that a Custom Factor computes the correct values when passed a
        mask. The mask/filter should be applied prior to computing any values,
        as opposed to computing the factor across the entire universe of
        assets. Any assets that are filtered out should be filled with missing
        values.
        """
        loader = self.loader
        dates = self.dates[5:8]
        assets = self.assets
        asset_ids = self.asset_ids
        constants = self.constants
        num_dates = len(dates)
        num_assets = len(assets)
        open = USEquityPricing.open
        close = USEquityPricing.close
        engine = SimplePipelineEngine(
            lambda column: loader, self.dates, self.asset_finder,
        )

        factor1_value = constants[open]
        factor2_value = 3.0 * (constants[open] - constants[close])

        def create_expected_results(expected_value, mask):
            expected_values = where(mask, expected_value, nan)
            return DataFrame(expected_values, index=dates, columns=assets)

        cascading_mask = AssetIDPlusDay() < (asset_ids[-1] + dates[0].day)
        expected_cascading_mask_result = make_cascading_boolean_array(
            shape=(num_dates, num_assets),
        )

        alternating_mask = (AssetIDPlusDay() % 2).eq(0)
        expected_alternating_mask_result = make_alternating_boolean_array(
            shape=(num_dates, num_assets), first_value=False,
        )

        masks = cascading_mask, alternating_mask
        expected_mask_results = (
            expected_cascading_mask_result,
            expected_alternating_mask_result,
        )
        for mask, expected_mask in zip(masks, expected_mask_results):
            # Test running a pipeline with a single masked factor.
            columns = {'factor1': OpenPrice(mask=mask), 'mask': mask}
            pipeline = Pipeline(columns=columns)
            results = engine.run_pipeline(pipeline, dates[0], dates[-1])

            mask_results = results['mask'].unstack()
            check_arrays(mask_results.values, expected_mask)

            factor1_results = results['factor1'].unstack()
            factor1_expected = create_expected_results(factor1_value,
                                                       mask_results)
            assert_frame_equal(factor1_results, factor1_expected)

            # Test running a pipeline with a second factor. This ensures that
            # adding another factor to the pipeline with a different window
            # length does not cause any unexpected behavior, especially when
            # both factors share the same mask.
            columns['factor2'] = RollingSumDifference(mask=mask)
            pipeline = Pipeline(columns=columns)
            results = engine.run_pipeline(pipeline, dates[0], dates[-1])

            mask_results = results['mask'].unstack()
            check_arrays(mask_results.values, expected_mask)

            factor1_results = results['factor1'].unstack()
            factor2_results = results['factor2'].unstack()
            factor1_expected = create_expected_results(factor1_value,
                                                       mask_results)
            factor2_expected = create_expected_results(factor2_value,
                                                       mask_results)
            assert_frame_equal(factor1_results, factor1_expected)
            assert_frame_equal(factor2_results, factor2_expected)

    def test_rolling_and_nonrolling(self):
        open_ = USEquityPricing.open
        close = USEquityPricing.close
        volume = USEquityPricing.volume

        # Test for thirty days up to the last day that we think all
        # the assets existed.
        dates_to_test = self.dates[-30:]

        constants = {open_: 1, close: 2, volume: 3}
        loader = PrecomputedLoader(
            constants=constants,
            dates=self.dates,
            sids=self.asset_ids,
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

        result_index = self.asset_ids * len(dates_to_test)
        result_shape = (len(result_index),)
        check_arrays(
            result['sumdiff'],
            Series(
                index=result_index,
                data=full(result_shape, -3, dtype=float),
            ),
        )

        for name, const in [('open', 1), ('close', 2), ('volume', 3)]:
            check_arrays(
                result[name],
                Series(
                    index=result_index,
                    data=full(result_shape, const, dtype=float),
                ),
            )

    def test_factor_with_single_output(self):
        """
        Test passing an `outputs` parameter of length 1 to a CustomFactor.
        """
        dates = self.dates[5:10]
        assets = self.assets
        num_dates = len(dates)
        open = USEquityPricing.open
        open_values = [self.constants[open]] * num_dates
        open_values_as_tuple = [(self.constants[open],)] * num_dates
        engine = SimplePipelineEngine(
            lambda column: self.loader, self.dates, self.asset_finder,
        )

        single_output = OpenPrice(outputs=['open'])
        pipeline = Pipeline(
            columns={
                'open_instance': single_output,
                'open_attribute': single_output.open,
            },
        )
        results = engine.run_pipeline(pipeline, dates[0], dates[-1])

        # The instance `single_output` itself will compute a numpy.recarray
        # when added as a column to our pipeline, so we expect its output
        # values to be 1-tuples.
        open_instance_expected = {
            asset: open_values_as_tuple for asset in assets
        }
        open_attribute_expected = {asset: open_values for asset in assets}

        for colname, expected_values in (
                ('open_instance', open_instance_expected),
                ('open_attribute', open_attribute_expected)):
            column_results = results[colname].unstack()
            expected_results = DataFrame(
                expected_values, index=dates, columns=assets, dtype=float64,
            )
            assert_frame_equal(column_results, expected_results)

    def test_factor_with_multiple_outputs(self):
        dates = self.dates[5:10]
        assets = self.assets
        asset_ids = self.asset_ids
        constants = self.constants
        num_dates = len(dates)
        num_assets = len(assets)
        open = USEquityPricing.open
        close = USEquityPricing.close
        engine = SimplePipelineEngine(
            lambda column: self.loader, self.dates, self.asset_finder,
        )

        def create_expected_results(expected_value, mask):
            expected_values = where(mask, expected_value, nan)
            return DataFrame(expected_values, index=dates, columns=assets)

        cascading_mask = AssetIDPlusDay() < (asset_ids[-1] + dates[0].day)
        expected_cascading_mask_result = make_cascading_boolean_array(
            shape=(num_dates, num_assets),
        )

        alternating_mask = (AssetIDPlusDay() % 2).eq(0)
        expected_alternating_mask_result = make_alternating_boolean_array(
            shape=(num_dates, num_assets), first_value=False,
        )

        expected_no_mask_result = full(
            shape=(num_dates, num_assets), fill_value=True, dtype=bool_dtype,
        )

        masks = cascading_mask, alternating_mask, NotSpecified
        expected_mask_results = (
            expected_cascading_mask_result,
            expected_alternating_mask_result,
            expected_no_mask_result,
        )
        for mask, expected_mask in zip(masks, expected_mask_results):
            open_price, close_price = MultipleOutputs(mask=mask)
            pipeline = Pipeline(
                columns={'open_price': open_price, 'close_price': close_price},
            )
            if mask is not NotSpecified:
                pipeline.add(mask, 'mask')

            results = engine.run_pipeline(pipeline, dates[0], dates[-1])
            for colname, case_column in (('open_price', open),
                                         ('close_price', close)):
                if mask is not NotSpecified:
                    mask_results = results['mask'].unstack()
                    check_arrays(mask_results.values, expected_mask)
                output_results = results[colname].unstack()
                output_expected = create_expected_results(
                    constants[case_column], expected_mask,
                )
                assert_frame_equal(output_results, output_expected)

    def test_instance_of_factor_with_multiple_outputs(self):
        """
        Test adding a CustomFactor instance, which has multiple outputs, as a
        pipeline column directly. Its computed values should be tuples
        containing the computed values of each of its outputs.
        """
        dates = self.dates[5:10]
        assets = self.assets
        num_dates = len(dates)
        num_assets = len(assets)
        constants = self.constants
        engine = SimplePipelineEngine(
            lambda column: self.loader, self.dates, self.asset_finder,
        )

        open_values = [constants[USEquityPricing.open]] * num_assets
        close_values = [constants[USEquityPricing.close]] * num_assets
        expected_values = [list(zip(open_values, close_values))] * num_dates
        expected_results = DataFrame(
            expected_values, index=dates, columns=assets, dtype=float64,
        )

        multiple_outputs = MultipleOutputs()
        pipeline = Pipeline(columns={'instance': multiple_outputs})
        results = engine.run_pipeline(pipeline, dates[0], dates[-1])
        instance_results = results['instance'].unstack()
        assert_frame_equal(instance_results, expected_results)

    def test_custom_factor_outputs_parameter(self):
        dates = self.dates[5:10]
        assets = self.assets
        num_dates = len(dates)
        num_assets = len(assets)
        constants = self.constants
        engine = SimplePipelineEngine(
            lambda column: self.loader, self.dates, self.asset_finder,
        )

        def create_expected_results(expected_value):
            expected_values = full(
                (num_dates, num_assets), expected_value, float64,
            )
            return DataFrame(expected_values, index=dates, columns=assets)

        for window_length in range(1, 3):
            sum_, diff = OpenCloseSumAndDiff(
                outputs=['sum_', 'diff'], window_length=window_length,
            )
            pipeline = Pipeline(columns={'sum_': sum_, 'diff': diff})
            results = engine.run_pipeline(pipeline, dates[0], dates[-1])
            for colname, op in ('sum_', add), ('diff', sub):
                output_results = results[colname].unstack()
                output_expected = create_expected_results(
                    op(
                        constants[USEquityPricing.open] * window_length,
                        constants[USEquityPricing.close] * window_length,
                    )
                )
                assert_frame_equal(output_results, output_expected)

    def test_loader_given_multiple_columns(self):

        class Loader1DataSet1(DataSet):
            col1 = Column(float)
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

        loader1 = RecordingPrecomputedLoader(constants=constants1,
                                             dates=self.dates,
                                             sids=self.assets)
        constants2 = {Loader2DataSet.col1: 5,
                      Loader2DataSet.col2: 6}
        loader2 = RecordingPrecomputedLoader(constants=constants2,
                                             dates=self.dates,
                                             sids=self.assets)

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

        def expected_for_col(col):
            val = vals[col]
            offset = columns[col].window_length - min_window
            return concatenate(
                [
                    full(offset * index.levshape[1], nan),
                    full(
                        (index.levshape[0] - offset) * index.levshape[1],
                        val,
                        float,
                    )
                ],
            )

        expected = DataFrame(
            data={col: expected_for_col(col) for col in vals},
            index=index,
            columns=columns,
        )

        assert_frame_equal(result, expected)

        self.assertEqual(set(loader1.load_calls),
                         {ColumnArgs.sorted_by_ds(Loader1DataSet1.col1,
                                                  Loader1DataSet2.col1),
                          ColumnArgs.sorted_by_ds(Loader1DataSet1.col2,
                                                  Loader1DataSet2.col2)})
        self.assertEqual(set(loader2.load_calls),
                         {ColumnArgs.sorted_by_ds(Loader2DataSet.col1,
                                                  Loader2DataSet.col2)})


class FrameInputTestCase(WithTradingEnvironment, ZiplineTestCase):
    asset_ids = ASSET_FINDER_EQUITY_SIDS = 1, 2, 3
    start = START_DATE = Timestamp('2015-01-01', tz='utc')
    end = END_DATE = Timestamp('2015-01-31', tz='utc')

    @classmethod
    def init_class_fixtures(cls):
        super(FrameInputTestCase, cls).init_class_fixtures()
        cls.dates = date_range(
            cls.start,
            cls.end,
            freq=cls.trading_calendar.day,
            tz='UTC',
        )
        cls.assets = cls.asset_finder.retrieve_all(cls.asset_ids)

    @lazyval
    def base_mask(self):
        return self.make_frame(True)

    def make_frame(self, data):
        return DataFrame(data, columns=self.assets, index=self.dates)

    def test_compute_with_adjustments(self):
        dates, asset_ids = self.dates, self.asset_ids
        low, high = USEquityPricing.low, USEquityPricing.high
        apply_idxs = [3, 10, 16]

        def apply_date(idx, offset=0):
            return dates[apply_idxs[idx] + offset]

        adjustments = DataFrame.from_records(
            [
                dict(
                    kind=MULTIPLY,
                    sid=asset_ids[1],
                    value=2.0,
                    start_date=None,
                    end_date=apply_date(0, offset=-1),
                    apply_date=apply_date(0),
                ),
                dict(
                    kind=MULTIPLY,
                    sid=asset_ids[1],
                    value=3.0,
                    start_date=None,
                    end_date=apply_date(1, offset=-1),
                    apply_date=apply_date(1),
                ),
                dict(
                    kind=MULTIPLY,
                    sid=asset_ids[1],
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


class SyntheticBcolzTestCase(WithAdjustmentReader,
                             ZiplineTestCase):
    first_asset_start = Timestamp('2015-04-01', tz='UTC')
    START_DATE = Timestamp('2015-01-01', tz='utc')
    END_DATE = Timestamp('2015-08-01', tz='utc')

    @classmethod
    def make_equity_info(cls):
        cls.equity_info = ret = make_rotating_equity_info(
            num_assets=6,
            first_start=cls.first_asset_start,
            frequency=cls.trading_calendar.day,
            periods_between_starts=4,
            asset_lifetime=8,
        )
        return ret

    @classmethod
    def make_equity_daily_bar_data(cls):
        return make_bar_data(
            cls.equity_info,
            cls.equity_daily_bar_days,
        )

    @classmethod
    def init_class_fixtures(cls):
        super(SyntheticBcolzTestCase, cls).init_class_fixtures()
        cls.all_asset_ids = cls.asset_finder.sids
        cls.last_asset_end = cls.equity_info['end_date'].max()
        cls.pipeline_loader = USEquityPricingLoader(
            cls.bcolz_equity_daily_bar_reader,
            cls.adjustment_reader,
        )

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
            self.trading_calendar.all_sessions,
            self.asset_finder,
        )
        window_length = 5
        asset_ids = self.all_asset_ids
        dates = date_range(
            self.first_asset_start + self.trading_calendar.day,
            self.last_asset_end,
            freq=self.trading_calendar.day,
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
        expected_raw = DataFrame(
            expected_bar_values_2d(
                dates - self.trading_calendar.day,
                self.equity_info,
                'close',
            ),
        ).rolling(
            window_length,
            min_periods=1,
        ).mean(
        ).values

        expected = DataFrame(
            # Truncate off the extra rows needed to compute the SMAs.
            expected_raw[window_length:],
            index=dates_to_test,  # dates_to_test is dates[window_length:]
            columns=self.asset_finder.retrieve_all(asset_ids),
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
            self.trading_calendar.all_sessions,
            self.asset_finder,
        )
        window_length = 5
        asset_ids = self.all_asset_ids
        dates = date_range(
            self.first_asset_start + self.trading_calendar.day,
            self.last_asset_end,
            freq=self.trading_calendar.day,
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
            data=zeros((len(dates_to_test), len(asset_ids)), dtype=float),
            index=dates_to_test,
            columns=self.asset_finder.retrieve_all(asset_ids),
        )
        self.write_nans(expected)
        result = results['drawdown'].unstack()

        assert_frame_equal(expected, result)


class ParameterizedFactorTestCase(WithTradingEnvironment, ZiplineTestCase):
    sids = ASSET_FINDER_EQUITY_SIDS = Int64Index([1, 2, 3])
    START_DATE = Timestamp('2015-01-31', tz='UTC')
    END_DATE = Timestamp('2015-03-01', tz='UTC')

    @classmethod
    def init_class_fixtures(cls):
        super(ParameterizedFactorTestCase, cls).init_class_fixtures()
        day = cls.trading_calendar.day

        cls.dates = dates = date_range(
            '2015-02-01',
            '2015-02-28',
            freq=day,
            tz='UTC',
        )
        sids = cls.sids

        cls.raw_data = DataFrame(
            data=arange(len(dates) * len(sids), dtype=float).reshape(
                len(dates), len(sids),
            ),
            index=dates,
            columns=cls.asset_finder.retrieve_all(sids),
        )
        cls.raw_data_with_nans = cls.raw_data.where((cls.raw_data % 2) != 0)

        open_loader = DataFrameLoader(
            USEquityPricing.open,
            cls.raw_data_with_nans,
        )
        close_loader = DataFrameLoader(USEquityPricing.close, cls.raw_data)
        volume_loader = DataFrameLoader(
            USEquityPricing.volume,
            cls.raw_data * 2,
        )

        cls.engine = SimplePipelineEngine(
            {
                USEquityPricing.open: open_loader,
                USEquityPricing.close: close_loader,
                USEquityPricing.volume: volume_loader,
            }.__getitem__,
            cls.dates,
            cls.asset_finder,
        )

    def expected_ewma(self, window_length, decay_rate):
        alpha = 1 - decay_rate
        span = (2 / alpha) - 1

        # XXX: This is a comically inefficient way to compute a windowed EWMA.
        # Don't use it outside of testing.  We're using rolling-apply of an
        # ewma (which is itself a rolling-window function) because we only want
        # to look at ``window_length`` rows at a time.
        return self.raw_data.rolling(window_length).apply(
            lambda subarray: (DataFrame(subarray)
                              .ewm(span=span)
                              .mean()
                              .values[-1])
        )[window_length:]

    def expected_ewmstd(self, window_length, decay_rate):
        alpha = 1 - decay_rate
        span = (2 / alpha) - 1

        # XXX: This is a comically inefficient way to compute a windowed
        # EWMSTD.  Don't use it outside of testing.  We're using rolling-apply
        # of an ewma (which is itself a rolling-window function) because we
        # only want to look at ``window_length`` rows at a time.
        return self.raw_data.rolling(window_length).apply(
            lambda subarray: (DataFrame(subarray)
                              .ewm(span=span)
                              .std()
                              .values[-1])
        )[window_length:]

    @parameterized.expand([
        (3,),
        (5,),
    ])
    def test_ewm_stats(self, window_length):

        def ewma_name(decay_rate):
            return 'ewma_%s' % decay_rate

        def ewmstd_name(decay_rate):
            return 'ewmstd_%s' % decay_rate

        decay_rates = [0.25, 0.5, 0.75]
        ewmas = {
            ewma_name(decay_rate): EWMA(
                inputs=(USEquityPricing.close,),
                window_length=window_length,
                decay_rate=decay_rate,
            )
            for decay_rate in decay_rates
        }

        ewmstds = {
            ewmstd_name(decay_rate): EWMSTD(
                inputs=(USEquityPricing.close,),
                window_length=window_length,
                decay_rate=decay_rate,
            )
            for decay_rate in decay_rates
        }

        all_results = self.engine.run_pipeline(
            Pipeline(columns=merge(ewmas, ewmstds)),
            self.dates[window_length],
            self.dates[-1],
        )

        for decay_rate in decay_rates:
            ewma_result = all_results[ewma_name(decay_rate)].unstack()
            ewma_expected = self.expected_ewma(window_length, decay_rate)
            assert_frame_equal(ewma_result, ewma_expected)

            ewmstd_result = all_results[ewmstd_name(decay_rate)].unstack()
            ewmstd_expected = self.expected_ewmstd(window_length, decay_rate)
            assert_frame_equal(ewmstd_result, ewmstd_expected)

    @staticmethod
    def decay_rate_to_span(decay_rate):
        alpha = 1 - decay_rate
        return (2 / alpha) - 1

    @staticmethod
    def decay_rate_to_com(decay_rate):
        alpha = 1 - decay_rate
        return (1 / alpha) - 1

    @staticmethod
    def decay_rate_to_halflife(decay_rate):
        return log(.5) / log(decay_rate)

    def ewm_cases():
        return product([EWMSTD, EWMA], [3, 5, 10])

    @parameterized.expand(ewm_cases())
    def test_from_span(self, type_, span):
        from_span = type_.from_span(
            inputs=[USEquityPricing.close],
            window_length=20,
            span=span,
        )
        implied_span = self.decay_rate_to_span(from_span.params['decay_rate'])
        assert_almost_equal(span, implied_span)

    @parameterized.expand(ewm_cases())
    def test_from_halflife(self, type_, halflife):
        from_hl = EWMA.from_halflife(
            inputs=[USEquityPricing.close],
            window_length=20,
            halflife=halflife,
        )
        implied_hl = self.decay_rate_to_halflife(from_hl.params['decay_rate'])
        assert_almost_equal(halflife, implied_hl)

    @parameterized.expand(ewm_cases())
    def test_from_com(self, type_, com):
        from_com = EWMA.from_center_of_mass(
            inputs=[USEquityPricing.close],
            window_length=20,
            center_of_mass=com,
        )
        implied_com = self.decay_rate_to_com(from_com.params['decay_rate'])
        assert_almost_equal(com, implied_com)

    del ewm_cases

    def test_ewm_aliasing(self):
        self.assertIs(ExponentialWeightedMovingAverage, EWMA)
        self.assertIs(ExponentialWeightedMovingStdDev, EWMSTD)

    def test_dollar_volume(self):
        results = self.engine.run_pipeline(
            Pipeline(
                columns={
                    'dv1': AverageDollarVolume(window_length=1),
                    'dv5': AverageDollarVolume(window_length=5),
                    'dv1_nan': AverageDollarVolume(
                        window_length=1,
                        inputs=[USEquityPricing.open, USEquityPricing.volume],
                    ),
                    'dv5_nan': AverageDollarVolume(
                        window_length=5,
                        inputs=[USEquityPricing.open, USEquityPricing.volume],
                    ),
                }
            ),
            self.dates[5],
            self.dates[-1],
        )

        expected_1 = (self.raw_data[5:] ** 2) * 2
        assert_frame_equal(results['dv1'].unstack(), expected_1)

        expected_5 = ((self.raw_data ** 2) * 2).rolling(5).mean()[5:]
        assert_frame_equal(results['dv5'].unstack(), expected_5)

        # The following two use USEquityPricing.open and .volume as inputs.
        # The former uses self.raw_data_with_nans, and the latter uses
        # .raw_data * 2.  Thus we multiply instead of squaring as above.
        expected_1_nan = (self.raw_data_with_nans[5:]
                          * self.raw_data[5:] * 2).fillna(0)
        assert_frame_equal(results['dv1_nan'].unstack(), expected_1_nan)

        expected_5_nan = ((self.raw_data_with_nans * self.raw_data * 2)
                          .fillna(0)
                          .rolling(5).mean()
                          [5:])

        assert_frame_equal(results['dv5_nan'].unstack(), expected_5_nan)


class StringColumnTestCase(WithSeededRandomPipelineEngine,
                           ZiplineTestCase):

    def test_string_classifiers_produce_categoricals(self):
        """
        Test that string-based classifiers produce pandas categoricals as their
        outputs.
        """
        col = TestingDataSet.categorical_col
        pipe = Pipeline(columns={'c': col.latest})

        run_dates = self.trading_days[-10:]
        start_date, end_date = run_dates[[0, -1]]

        result = self.run_pipeline(pipe, start_date, end_date)
        assert isinstance(result.c.values, Categorical)

        expected_raw_data = self.raw_expected_values(
            col,
            start_date,
            end_date,
        )
        expected_labels = LabelArray(expected_raw_data, col.missing_value)
        expected_final_result = expected_labels.as_categorical_frame(
            index=run_dates,
            columns=self.asset_finder.retrieve_all(self.asset_finder.sids),
        )
        assert_frame_equal(result.c.unstack(), expected_final_result)


class WindowSafetyPropagationTestCase(WithSeededRandomPipelineEngine,
                                      ZiplineTestCase):

    SEEDED_RANDOM_PIPELINE_SEED = 5

    def test_window_safety_propagation(self):
        dates = self.trading_days[-30:]
        start_date, end_date = dates[[-10, -1]]

        col = TestingDataSet.float_col
        pipe = Pipeline(
            columns={
                'average_of_rank_plus_one': SimpleMovingAverage(
                    inputs=[col.latest.rank() + 1],
                    window_length=10,
                ),
                'average_of_aliased_rank_plus_one': SimpleMovingAverage(
                    inputs=[col.latest.rank().alias('some_alias') + 1],
                    window_length=10,
                ),
                'average_of_rank_plus_one_aliased': SimpleMovingAverage(
                    inputs=[(col.latest.rank() + 1).alias('some_alias')],
                    window_length=10,
                ),
            }
        )
        results = self.run_pipeline(pipe, start_date, end_date).unstack()

        expected_ranks = DataFrame(
            self.raw_expected_values(
                col,
                dates[-19],
                dates[-1],
            ),
            index=dates[-19:],
            columns=self.asset_finder.retrieve_all(
                self.ASSET_FINDER_EQUITY_SIDS,
            )
        ).rank(axis='columns')

        # All three expressions should be equivalent and evaluate to this.
        expected_result = (
            (expected_ranks + 1)
            .rolling(10)
            .mean()
            .dropna(how='any')
        )

        for colname in results.columns.levels[0]:
            assert_equal(expected_result, results[colname])


class PopulateInitialWorkspaceTestCase(WithConstantInputs, ZiplineTestCase):

    @parameter_space(window_length=[3, 5], pipeline_length=[5, 10])
    def test_populate_initial_workspace(self, window_length, pipeline_length):
        column = USEquityPricing.low
        base_term = column.latest

        # Take a Z-Score here so that the precomputed term is window-safe.  The
        # z-score will never actually get computed because we swap it out.
        precomputed_term = (base_term.zscore()).alias('precomputed_term')

        # A term that has `precomputed_term` as an input.
        depends_on_precomputed_term = precomputed_term + 1
        # A term that requires a window of `precomputed_term`.
        depends_on_window_of_precomputed_term = SimpleMovingAverage(
            inputs=[precomputed_term],
            window_length=window_length,
        )

        precomputed_term_with_window = SimpleMovingAverage(
            inputs=(column,),
            window_length=window_length,
        ).alias('precomputed_term_with_window')
        depends_on_precomputed_term_with_window = (
            precomputed_term_with_window + 1
        )

        column_value = self.constants[column]
        precomputed_term_value = -column_value
        precomputed_term_with_window_value = -(column_value + 1)

        def populate_initial_workspace(initial_workspace,
                                       root_mask_term,
                                       execution_plan,
                                       dates,
                                       assets):
            def shape_for_term(term):
                ndates = len(execution_plan.mask_and_dates_for_term(
                    term,
                    root_mask_term,
                    initial_workspace,
                    dates,
                )[1])
                nassets = len(assets)
                return (ndates, nassets)

            ws = initial_workspace.copy()
            ws[precomputed_term] = full(
                shape_for_term(precomputed_term),
                precomputed_term_value,
                dtype=float64,
            )
            ws[precomputed_term_with_window] = full(
                shape_for_term(precomputed_term_with_window),
                precomputed_term_with_window_value,
                dtype=float64,
            )
            return ws

        def dispatcher(c):
            if c is column:
                # the base_term should never be loaded, its initial refcount
                # should be zero
                return ExplodingObject()
            return self.loader

        engine = SimplePipelineEngine(
            dispatcher,
            self.dates,
            self.asset_finder,
            populate_initial_workspace=populate_initial_workspace,
        )

        results = engine.run_pipeline(
            Pipeline({
                'precomputed_term': precomputed_term,
                'precomputed_term_with_window': precomputed_term_with_window,
                'depends_on_precomputed_term': depends_on_precomputed_term,
                'depends_on_precomputed_term_with_window':
                    depends_on_precomputed_term_with_window,
                'depends_on_window_of_precomputed_term':
                    depends_on_window_of_precomputed_term,
            }),
            self.dates[-pipeline_length],
            self.dates[-1],
        )

        assert_equal(
            results['precomputed_term'].values,
            full_like(
                results['precomputed_term'],
                precomputed_term_value,
            ),
        ),
        assert_equal(
            results['precomputed_term_with_window'].values,
            full_like(
                results['precomputed_term_with_window'],
                precomputed_term_with_window_value,
            ),
        ),
        assert_equal(
            results['depends_on_precomputed_term'].values,
            full_like(
                results['depends_on_precomputed_term'],
                precomputed_term_value + 1,
            ),
        )
        assert_equal(
            results['depends_on_precomputed_term_with_window'].values,
            full_like(
                results['depends_on_precomputed_term_with_window'],
                precomputed_term_with_window_value + 1,
            ),
        )
        assert_equal(
            results['depends_on_window_of_precomputed_term'].values,
            full_like(
                results['depends_on_window_of_precomputed_term'],
                precomputed_term_value,
            ),
        )
