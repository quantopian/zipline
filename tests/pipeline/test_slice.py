"""
Tests for slicing pipeline terms.
"""
from numpy import arange
from pandas import (
    DataFrame,
    date_range,
    Int64Index,
    Timestamp,
)
from pandas.util.testing import assert_frame_equal

from zipline.assets import Asset
from zipline.errors import (
    NonExistentAssetInTimeFrame,
    NonSliceableTerm,
    NonWindowSafeInput,
    UnsupportedPipelineOutput,
)
from zipline.pipeline import CustomFactor, Pipeline
from zipline.pipeline.data import USEquityPricing
from zipline.pipeline.engine import SimplePipelineEngine
from zipline.pipeline.factors import (
    Returns,
    RollingLinearRegressionOfReturns,
    RollingPearsonOfReturns,
    RollingSpearmanOfReturns,
    SimpleMovingAverage,
)
from zipline.pipeline.loaders.frame import DataFrameLoader
from zipline.testing import (
    AssetID,
    check_arrays,
    OpenPrice,
    parameter_space,
)
from zipline.testing.fixtures import WithTradingEnvironment, ZiplineTestCase


class SliceTestCase(WithTradingEnvironment, ZiplineTestCase):
    sids = ASSET_FINDER_EQUITY_SIDS = Int64Index([1, 2, 3])
    START_DATE = Timestamp('2015-01-31', tz='UTC')
    END_DATE = Timestamp('2015-03-01', tz='UTC')

    @classmethod
    def init_class_fixtures(cls):
        super(SliceTestCase, cls).init_class_fixtures()

        day = cls.trading_schedule.day
        sids = cls.sids

        cls.dates = dates = date_range(
            '2015-02-01', '2015-02-28', freq=day, tz='UTC',
        )
        cls.start_date = dates[14]
        cls.end_date = dates[18]

        cls.raw_data = DataFrame(
            data=arange(len(dates) * len(sids), dtype=float).reshape(
                len(dates), len(sids),
            ),
            index=dates,
            columns=cls.asset_finder.retrieve_all(sids),
        )

        close_loader = DataFrameLoader(USEquityPricing.close, cls.raw_data)

        cls.engine = SimplePipelineEngine(
            {USEquityPricing.close: close_loader}.__getitem__,
            cls.dates,
            cls.asset_finder,
        )

    @parameter_space(my_asset_column=[0, 1, 2], window_length_=[1, 2, 3])
    def test_slice(self, my_asset_column, window_length_):
        """
        Test that slices can be created by indexing into a term, and that they
        have the correct shape when used as inputs.
        """
        sids = self.sids
        my_asset = self.asset_finder.retrieve_asset(self.sids[my_asset_column])

        returns = Returns(window_length=2)
        returns_slice = returns[my_asset]

        class UsesSlicedInput(CustomFactor):
            window_length = window_length_
            inputs = [returns, returns_slice]

            def compute(self, today, assets, out, returns, returns_slice):
                # Make sure that our slice is the correct shape (i.e. has only
                # one column) and that it has the same values as the original
                # returns factor from which it is derived.
                assert returns_slice.shape == (self.window_length, 1)
                assert returns.shape == (self.window_length, len(sids))
                check_arrays(returns_slice[:, 0], returns[:, my_asset_column])

        # Assertions about the expected slice data are made in the `compute`
        # function of our custom factor above.
        self.engine.run_pipeline(
            Pipeline(columns={'uses_sliced_input': UsesSlicedInput()}),
            self.start_date,
            self.end_date,
        )

    @parameter_space(unmasked_column=[0, 1, 2], slice_column=[0, 1, 2])
    def test_slice_with_masking(self, unmasked_column, slice_column):
        """
        Test that masking a factor that uses slices as inputs does not mask the
        slice data.
        """
        sids = self.sids
        asset_finder = self.asset_finder
        engine = self.engine
        start_date = self.start_date
        end_date = self.end_date

        # Create a filter that masks out all but a single asset.
        unmasked_asset = asset_finder.retrieve_asset(sids[unmasked_column])
        unmasked_asset_only = (AssetID().eq(unmasked_asset.sid))

        # Asset used to create our slice. In the cases where this is different
        # than `unmasked_asset`, our slice should still have non-missing data
        # when used as an input to our custom factor. That is, it should not be
        # masked out.
        slice_asset = asset_finder.retrieve_asset(sids[slice_column])

        returns = Returns(window_length=2)
        returns_slice = returns[slice_asset]

        returns_results = engine.run_pipeline(
            Pipeline(columns={'returns': returns}), start_date, end_date,
        )
        returns_results = returns_results['returns'].unstack()

        class UsesSlicedInput(CustomFactor):
            window_length = 1
            inputs = [returns, returns_slice]

            def compute(self, today, assets, out, returns, returns_slice):
                # Ensure that our mask correctly affects the `returns` input
                # and does not affect the `returns_slice` input.
                assert returns.shape == (1, 1)
                assert returns_slice.shape == (1, 1)
                assert returns[0, 0] == \
                    returns_results.loc[today, unmasked_asset]
                assert returns_slice[0, 0] == \
                    returns_results.loc[today, slice_asset]

        columns = {'masked': UsesSlicedInput(mask=unmasked_asset_only)}

        # Assertions about the expected data are made in the `compute` function
        # of our custom factor above.
        engine.run_pipeline(Pipeline(columns=columns), start_date, end_date)

    def test_adding_slice_column(self):
        """
        Test that slices cannot be added as a pipeline column.
        """
        my_asset = self.asset_finder.retrieve_asset(self.sids[0])
        open_slice = OpenPrice()[my_asset]

        with self.assertRaises(UnsupportedPipelineOutput):
            Pipeline(columns={'open_slice': open_slice})

        pipe = Pipeline(columns={})
        with self.assertRaises(UnsupportedPipelineOutput):
            pipe.add(open_slice, 'open_slice')

    def test_loadable_term_slices(self):
        """
        Test that slicing loadable terms raises the proper error.
        """
        my_asset = self.asset_finder.retrieve_asset(self.sids[0])

        with self.assertRaises(NonSliceableTerm):
            USEquityPricing.close[my_asset]

    def test_non_existent_asset(self):
        """
        Test that indexing into a term with a non-existent asset raises the
        proper exception.
        """
        my_asset = Asset(0)
        returns = Returns(window_length=2)
        returns_slice = returns[my_asset]

        class UsesSlicedInput(CustomFactor):
            window_length = 1
            inputs = [returns_slice]

            def compute(self, today, assets, out, returns_slice):
                pass

        with self.assertRaises(NonExistentAssetInTimeFrame):
            self.engine.run_pipeline(
                Pipeline(columns={'uses_sliced_input': UsesSlicedInput()}),
                self.start_date,
                self.end_date,
            )

    def test_window_safety_of_slices(self):
        """
        Test that slices correctly inherit the `window_safe` property of the
        term from which they are derived.
        """
        my_asset = self.asset_finder.retrieve_asset(self.sids[0])

        # SimpleMovingAverage is not window safe.
        sma = SimpleMovingAverage(
            inputs=[USEquityPricing.close], window_length=10,
        )
        sma_slice = sma[my_asset]

        class UsesSlicedInput(CustomFactor):
            window_length = 1
            inputs = [sma_slice]

            def compute(self, today, assets, out, sma_slice):
                pass

        with self.assertRaises(NonWindowSafeInput):
            self.engine.run_pipeline(
                Pipeline(columns={'uses_sliced_input': UsesSlicedInput()}),
                self.start_date,
                self.end_date,
            )

        # Make sure that slices of custom factors are not window safe.
        class MyUnsafeFactor(CustomFactor):
            window_length = 1
            inputs = [USEquityPricing.close]

            def compute(self, today, assets, out, close):
                out[:] = close

        my_unsafe_factor = MyUnsafeFactor()
        my_unsafe_factor_slice = my_unsafe_factor[my_asset]

        class UsesSlicedInput(CustomFactor):
            window_length = 1
            inputs = [my_unsafe_factor_slice]

            def compute(self, today, assets, out, my_unsafe_factor_slice):
                pass

        with self.assertRaises(NonWindowSafeInput):
            self.engine.run_pipeline(
                Pipeline(columns={'uses_sliced_input': UsesSlicedInput()}),
                self.start_date,
                self.end_date,
            )

        # Create a window safe factor.
        class MySafeFactor(CustomFactor):
            window_length = 1
            inputs = [USEquityPricing.close]
            window_safe = True

            def compute(self, today, assets, out, close):
                pass

        my_safe_factor = MySafeFactor()
        my_safe_factor_slice = my_safe_factor[my_asset]

        # Make sure that correlations are not safe if either the factor *or*
        # the target slice are not window safe.
        with self.assertRaises(NonWindowSafeInput):
            my_unsafe_factor.rolling_pearsonr(
                target_slice=my_safe_factor_slice,
                correlation_length=10,
            )

        with self.assertRaises(NonWindowSafeInput):
            my_safe_factor.rolling_pearsonr(
                target_slice=my_unsafe_factor_slice,
                correlation_length=10,
            )

    @parameter_space(returns_length=[2, 3], correlation_length=[3, 4])
    def test_factor_correlation_methods(self,
                                        returns_length,
                                        correlation_length):
        """
        Ensure that `Factor.rolling_pearsonr` and `Factor.rolling_spearmanr`
        are consistent with the built-in factors `RollingPearsonOfReturns` and
        `RollingSpearmanOfReturns`.
        """
        my_asset = self.asset_finder.retrieve_asset(self.sids[0])

        returns = Returns(window_length=returns_length)
        returns_slice = returns[my_asset]

        pearson = returns.rolling_pearsonr(
            target_slice=returns_slice, correlation_length=correlation_length,
        )
        spearman = returns.rolling_spearmanr(
            target_slice=returns_slice, correlation_length=correlation_length,
        )
        expected_pearson = RollingPearsonOfReturns(
            target=my_asset,
            returns_length=returns_length,
            correlation_length=correlation_length,
        )
        expected_spearman = RollingSpearmanOfReturns(
            target=my_asset,
            returns_length=returns_length,
            correlation_length=correlation_length,
        )

        columns = {
            'pearson': pearson,
            'spearman': spearman,
            'expected_pearson': expected_pearson,
            'expected_spearman': expected_spearman,
        }

        results = self.engine.run_pipeline(
            Pipeline(columns=columns), self.start_date, self.end_date,
        )
        pearson_results = results['pearson'].unstack()
        spearman_results = results['spearman'].unstack()
        expected_pearson_results = results['expected_pearson'].unstack()
        expected_spearman_results = results['expected_spearman'].unstack()

        assert_frame_equal(pearson_results, expected_pearson_results)
        assert_frame_equal(spearman_results, expected_spearman_results)

    @parameter_space(returns_length=[2, 3], regression_length=[3, 4])
    def test_factor_regression_method(self, returns_length, regression_length):
        """
        Ensure that `Factor.rolling_linear_regression` is consistent with the
        built-in factor `RollingLinearRegressionOfReturns`.
        """
        my_asset = self.asset_finder.retrieve_asset(self.sids[0])

        returns = Returns(window_length=returns_length)
        returns_slice = returns[my_asset]

        regression = returns.rolling_linear_regression(
            target_slice=returns_slice, regression_length=regression_length,
        )
        expected_regression = RollingLinearRegressionOfReturns(
            target=my_asset,
            returns_length=returns_length,
            regression_length=regression_length,
        )

        columns = {
            'regression': regression,
            'expected_regression': expected_regression,
        }

        results = self.engine.run_pipeline(
            Pipeline(columns=columns), self.start_date, self.end_date,
        )
        regression_results = results['regression'].unstack()
        expected_regression_results = results['expected_regression'].unstack()

        assert_frame_equal(regression_results, expected_regression_results)
