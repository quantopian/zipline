"""
Tests for slicing pipeline terms.
"""
from numpy import full_like, nan
from pandas import (
    DataFrame,
    date_range,
    Int64Index,
    Timestamp,
)
from pandas.util.testing import assert_frame_equal
from scipy.stats import (
    linregress,
    pearsonr,
    spearmanr,
)

from zipline.assets import Asset
from zipline.errors import (
    IncompatibleTerms,
    NonExistentAssetInTimeFrame,
    NonSliceableTerm,
    NonWindowSafeInput,
    UnsupportedPipelineOutput,
)
from zipline.pipeline import CustomFactor, Pipeline
from zipline.pipeline.data import USEquityPricing
from zipline.pipeline.data.testing import TestingDataSet
from zipline.pipeline.factors import (
    Returns,
    RollingLinearRegressionOfReturns,
    RollingPearsonOfReturns,
    RollingSpearmanOfReturns,
    SimpleMovingAverage,
)
from zipline.testing import (
    AssetID,
    check_arrays,
    OpenPrice,
    parameter_space,
)
from zipline.testing.fixtures import (
    WithSeededRandomPipelineEngine,
    ZiplineTestCase,
)
from zipline.utils.numpy_utils import datetime64ns_dtype


class SliceTestCase(WithSeededRandomPipelineEngine, ZiplineTestCase):
    sids = ASSET_FINDER_EQUITY_SIDS = Int64Index([1, 2, 3])
    START_DATE = Timestamp('2015-01-31', tz='UTC')
    END_DATE = Timestamp('2015-03-01', tz='UTC')

    @classmethod
    def init_class_fixtures(cls):
        super(SliceTestCase, cls).init_class_fixtures()

        day = cls.trading_schedule.day
        dates = date_range('2015-02-01', '2015-02-28', freq=day, tz='UTC')
        cls.start_date = dates[14]
        cls.end_date = dates[18]

        # Random input for factors.
        cls.col = TestingDataSet.float_col

    @parameter_space(my_asset_column=[0, 1, 2], window_length_=[1, 2, 3])
    def test_slice(self, my_asset_column, window_length_):
        """
        Test that slices can be created by indexing into a term, and that they
        have the correct shape when used as inputs.
        """
        sids = self.sids
        my_asset = self.asset_finder.retrieve_asset(self.sids[my_asset_column])

        returns = Returns(window_length=2, inputs=[self.col])
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
        self.run_pipeline(
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

        returns = Returns(window_length=2, inputs=[self.col])
        returns_slice = returns[slice_asset]

        returns_results = self.run_pipeline(
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
        self.run_pipeline(Pipeline(columns=columns), start_date, end_date)

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
        returns = Returns(window_length=2, inputs=[self.col])
        returns_slice = returns[my_asset]

        class UsesSlicedInput(CustomFactor):
            window_length = 1
            inputs = [returns_slice]

            def compute(self, today, assets, out, returns_slice):
                pass

        with self.assertRaises(NonExistentAssetInTimeFrame):
            self.run_pipeline(
                Pipeline(columns={'uses_sliced_input': UsesSlicedInput()}),
                self.start_date,
                self.end_date,
            )

    def test_window_safety_of_slices(self):
        """
        Test that slices correctly inherit the `window_safe` property of the
        term from which they are derived.
        """
        col = self.col
        my_asset = self.asset_finder.retrieve_asset(self.sids[0])

        # SimpleMovingAverage is not window safe.
        sma = SimpleMovingAverage(inputs=[self.col], window_length=10)
        sma_slice = sma[my_asset]

        class UsesSlicedInput(CustomFactor):
            window_length = 1
            inputs = [sma_slice]

            def compute(self, today, assets, out, sma_slice):
                pass

        with self.assertRaises(NonWindowSafeInput):
            self.run_pipeline(
                Pipeline(columns={'uses_sliced_input': UsesSlicedInput()}),
                self.start_date,
                self.end_date,
            )

        # Make sure that slices of custom factors are not window safe.
        class MyUnsafeFactor(CustomFactor):
            window_length = 1
            inputs = [col]

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
            self.run_pipeline(
                Pipeline(columns={'uses_sliced_input': UsesSlicedInput()}),
                self.start_date,
                self.end_date,
            )

        # Create a window safe factor.
        class MySafeFactor(CustomFactor):
            window_length = 1
            inputs = [col]
            window_safe = True

            def compute(self, today, assets, out, close):
                pass

        my_safe_factor = MySafeFactor()
        my_safe_factor_slice = my_safe_factor[my_asset]

        # Make sure that correlations are not safe if either the factor *or*
        # the target slice are not window safe.
        with self.assertRaises(NonWindowSafeInput):
            my_unsafe_factor.pearsonr(
                target=my_safe_factor_slice, correlation_length=10,
            )

        with self.assertRaises(NonWindowSafeInput):
            my_safe_factor.pearsonr(
                target=my_unsafe_factor_slice, correlation_length=10,
            )


class StatisticalMethodsTestCase(WithSeededRandomPipelineEngine,
                                 ZiplineTestCase):
    sids = ASSET_FINDER_EQUITY_SIDS = Int64Index([1, 2, 3])
    START_DATE = Timestamp('2015-01-31', tz='UTC')
    END_DATE = Timestamp('2015-03-01', tz='UTC')

    @classmethod
    def init_class_fixtures(cls):
        super(StatisticalMethodsTestCase, cls).init_class_fixtures()

        day = cls.trading_schedule.day
        cls.dates = dates = date_range(
            '2015-02-01', '2015-02-28', freq=day, tz='UTC',
        )
        cls.start_date_index = start_date_index = 14
        cls.end_date_index = end_date_index = 18
        cls.start_date = dates[start_date_index]
        cls.end_date = dates[end_date_index]

        # Random input for factors.
        cls.col = TestingDataSet.float_col

    @parameter_space(returns_length=[2, 3], correlation_length=[3, 4])
    def test_factor_correlation_methods(self,
                                        returns_length,
                                        correlation_length):
        """
        Ensure that `Factor.pearsonr` and `Factor.spearmanr` are consistent
        with the built-in factors `RollingPearsonOfReturns` and
        `RollingSpearmanOfReturns`.
        """
        my_asset = self.asset_finder.retrieve_asset(self.sids[0])

        returns = Returns(window_length=returns_length, inputs=[self.col])
        returns_slice = returns[my_asset]

        pearson = returns.pearsonr(
            target=returns_slice, correlation_length=correlation_length,
        )
        spearman = returns.spearmanr(
            target=returns_slice, correlation_length=correlation_length,
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

        # These built-ins construct their own Returns factor to use as inputs,
        # so the only way to set our own inputs is to do so after the fact.
        # This should not be done in practice. It is necessary here because we
        # want Returns to use our random data as an input, but by default it is
        # using USEquityPricing.close.
        expected_pearson.inputs = [returns, returns_slice]
        expected_spearman.inputs = [returns, returns_slice]

        columns = {
            'pearson': pearson,
            'spearman': spearman,
            'expected_pearson': expected_pearson,
            'expected_spearman': expected_spearman,
        }

        results = self.run_pipeline(
            Pipeline(columns=columns), self.start_date, self.end_date,
        )
        pearson_results = results['pearson'].unstack()
        spearman_results = results['spearman'].unstack()
        expected_pearson_results = results['expected_pearson'].unstack()
        expected_spearman_results = results['expected_spearman'].unstack()

        assert_frame_equal(pearson_results, expected_pearson_results)
        assert_frame_equal(spearman_results, expected_spearman_results)

        # Make sure we cannot call the correlation methods on factors or slices
        # of dtype `datetime64[ns]`.
        class DateFactor(CustomFactor):
            window_length = 1
            inputs = []
            dtype = datetime64ns_dtype
            window_safe = True

            def compute(self, today, assets, out):
                pass

        date_factor = DateFactor()
        date_factor_slice = date_factor[my_asset]

        with self.assertRaises(TypeError):
            date_factor.pearsonr(
                target=returns_slice, correlation_length=correlation_length,
            )
        with self.assertRaises(TypeError):
            date_factor.spearmanr(
                target=returns_slice, correlation_length=correlation_length,
            )
        with self.assertRaises(TypeError):
            returns.pearsonr(
                target=date_factor_slice,
                correlation_length=correlation_length,
            )
        with self.assertRaises(TypeError):
            returns.pearsonr(
                target=date_factor_slice,
                correlation_length=correlation_length,
            )

    @parameter_space(returns_length=[2, 3], regression_length=[3, 4])
    def test_factor_regression_method(self, returns_length, regression_length):
        """
        Ensure that `Factor.linear_regression` is consistent with the built-in
        factor `RollingLinearRegressionOfReturns`.
        """
        my_asset = self.asset_finder.retrieve_asset(self.sids[0])

        returns = Returns(window_length=returns_length, inputs=[self.col])
        returns_slice = returns[my_asset]

        regression = returns.linear_regression(
            target=returns_slice, regression_length=regression_length,
        )
        expected_regression = RollingLinearRegressionOfReturns(
            target=my_asset,
            returns_length=returns_length,
            regression_length=regression_length,
        )

        # These built-ins construct their own Returns factor to use as inputs,
        # so the only way to set our own inputs is to do so after the fact.
        # This should not be done in practice. It is necessary here because we
        # want Returns to use our random data as an input, but by default it is
        # using USEquityPricing.close.
        expected_regression.inputs = [returns, returns_slice]

        columns = {
            'regression': regression,
            'expected_regression': expected_regression,
        }

        results = self.run_pipeline(
            Pipeline(columns=columns), self.start_date, self.end_date,
        )
        regression_results = results['regression'].unstack()
        expected_regression_results = results['expected_regression'].unstack()

        assert_frame_equal(regression_results, expected_regression_results)

        # Make sure we cannot call the linear regression method on factors or
        # slices of dtype `datetime64[ns]`.
        class DateFactor(CustomFactor):
            window_length = 1
            inputs = []
            dtype = datetime64ns_dtype
            window_safe = True

            def compute(self, today, assets, out):
                pass

        date_factor = DateFactor()
        date_factor_slice = date_factor[my_asset]

        with self.assertRaises(TypeError):
            date_factor.linear_regression(
                target=returns_slice, regression_length=regression_length,
            )
        with self.assertRaises(TypeError):
            returns.linear_regression(
                target=date_factor_slice, regression_length=regression_length,
            )

    @parameter_space(correlation_length=[1, 2, 3, 4])
    def test_factor_correlation_methods_two_factors(self, correlation_length):
        """
        Tests for `Factor.pearsonr` and `Factor.spearmanr` when passed another
        2D factor instead of a Slice.
        """
        dates = self.dates
        start_date_index = self.start_date_index
        end_date_index = self.end_date_index
        num_days = end_date_index - start_date_index + 1
        assets = self.asset_finder.retrieve_all(self.sids)

        returns_5 = Returns(window_length=5, inputs=[self.col])
        returns_10 = Returns(window_length=10, inputs=[self.col])

        # Ensure that the correlation methods cannot be called with two 2D
        # factors which have different masks.
        returns_5_masked = Returns(
            window_length=5, inputs=[self.col], mask=AssetID().eq(1),
        )
        returns_10_masked = Returns(
            window_length=5, inputs=[self.col], mask=AssetID().eq(2),
        )
        with self.assertRaises(IncompatibleTerms):
            returns_5_masked.pearsonr(
                target=returns_10_masked,
                correlation_length=correlation_length,
            )
        with self.assertRaises(IncompatibleTerms):
            returns_5_masked.spearmanr(
                target=returns_10_masked,
                correlation_length=correlation_length,
            )

        pearson_factor = returns_5.pearsonr(
            target=returns_10, correlation_length=correlation_length,
        )
        spearman_factor = returns_5.spearmanr(
            target=returns_10, correlation_length=correlation_length,
        )

        pipeline = Pipeline(
            columns={'pearson': pearson_factor, 'spearman': spearman_factor},
        )
        results = self.run_pipeline(pipeline, self.start_date, self.end_date)
        pearson_results = results['pearson'].unstack()
        spearman_results = results['spearman'].unstack()

        # Run a separate pipeline that calculates returns starting
        # (correlation_length - 1) days prior to our start date. This is
        # because we need (correlation_length - 1) extra days of returns to
        # compute our expected correlations.
        columns = {'returns_5': returns_5, 'returns_10': returns_10}
        results = self.run_pipeline(
            Pipeline(columns=columns),
            dates[start_date_index - (correlation_length - 1)],
            dates[end_date_index],
        )
        returns_5_results = results['returns_5'].unstack()
        returns_10_results = results['returns_10'].unstack()

        expected_pearson_results = full_like(pearson_results, nan)
        expected_spearman_results = full_like(spearman_results, nan)
        for day in range(num_days):
            todays_returns_5 = returns_5_results.iloc[
                day:day + correlation_length
            ]
            todays_returns_10 = returns_10_results.iloc[
                day:day + correlation_length
            ]
            for asset, asset_returns_5 in todays_returns_5.iteritems():
                asset_column = int(asset) - 1
                asset_returns_10 = todays_returns_10[asset]
                expected_pearson_results[day, asset_column] = pearsonr(
                    asset_returns_5, asset_returns_10,
                )[0]
                expected_spearman_results[day, asset_column] = spearmanr(
                    asset_returns_5, asset_returns_10,
                )[0]

        assert_frame_equal(
            pearson_results,
            DataFrame(
                data=expected_pearson_results,
                index=dates[start_date_index:end_date_index + 1],
                columns=assets,
            ),
        )
        assert_frame_equal(
            spearman_results,
            DataFrame(
                data=expected_spearman_results,
                index=dates[start_date_index:end_date_index + 1],
                columns=assets,
            ),
        )

    def test_factor_regression_method_two_factors(self):
        """
        Tests for `Factor.linear_regression` when passed another 2D factor
        instead of a Slice.
        """
        regression_length = 4
        dates = self.dates
        start_date_index = self.start_date_index
        end_date_index = self.end_date_index
        num_days = end_date_index - start_date_index + 1
        assets = self.asset_finder.retrieve_all(self.sids)

        # The order of these is meant to align with the output of `linregress`.
        outputs = ['beta', 'alpha', 'r_value', 'p_value', 'stderr']

        returns_5 = Returns(window_length=5, inputs=[self.col])
        returns_10 = Returns(window_length=10, inputs=[self.col])

        # Ensure that the `linear_regression` method cannot be called with two
        # 2D factors which have different masks.
        returns_5_masked = Returns(
            window_length=5, inputs=[self.col], mask=AssetID().eq(1),
        )
        returns_10_masked = Returns(
            window_length=5, inputs=[self.col], mask=AssetID().eq(2),
        )
        with self.assertRaises(IncompatibleTerms):
            returns_5_masked.linear_regression(
                target=returns_10_masked,
                regression_length=regression_length,
            )

        regression_factor = returns_5.linear_regression(
            target=returns_10, regression_length=regression_length,
        )
        pipeline = Pipeline(
            columns={
                output: getattr(regression_factor, output)
                for output in outputs
            },
        )
        results = self.run_pipeline(pipeline, self.start_date, self.end_date)

        output_results = {}
        expected_output_results = {}
        for output in outputs:
            output_results[output] = results[output].unstack()
            expected_output_results[output] = full_like(
                output_results[output], nan,
            )

        # Run a separate pipeline that calculates returns starting
        # (regression_length - 1) days prior to our start date. This is because
        # we need (regression_length - 1) extra days of returns to compute our
        # expected regressions.
        columns = {'returns_5': returns_5, 'returns_10': returns_10}
        results = self.run_pipeline(
            Pipeline(columns=columns),
            dates[start_date_index - (regression_length - 1)],
            dates[end_date_index],
        )
        returns_5_results = results['returns_5'].unstack()
        returns_10_results = results['returns_10'].unstack()

        # On each day, calculate the expected regression results for Y ~ X
        # where Y is the asset we are interested in and X is each other asset
        # Each regression is calculated over `regression_length` days of data.
        for day in range(num_days):
            todays_returns_5 = returns_5_results.iloc[
                day:day + regression_length
            ]
            todays_returns_10 = returns_10_results.iloc[
                day:day + regression_length
            ]
            for asset, asset_returns_5 in todays_returns_5.iteritems():
                asset_column = int(asset) - 1
                asset_returns_10 = todays_returns_10[asset]
                expected_regression_results = linregress(
                    y=asset_returns_5, x=asset_returns_10,
                )
                for i, output in enumerate(outputs):
                    expected_output_results[output][day, asset_column] = \
                        expected_regression_results[i]

        for output in outputs:
            output_result = output_results[output]
            expected_output_result = DataFrame(
                data=expected_output_results[output],
                index=dates[start_date_index:end_date_index + 1],
                columns=assets,
            )
            assert_frame_equal(output_result, expected_output_result)
