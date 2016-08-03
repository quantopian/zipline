"""
Tests for slicing pipeline terms.
"""
from numpy import where
from pandas import Int64Index, Timestamp
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
    AssetIDPlusDay,
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

        # Using the date at index 14 as the start date because when running
        # pipelines, especially those involving correlations or regressions, we
        # want to make sure there are enough days to look back on. The end date
        # at index 18 is chosen for convenience, as it makes for a contiguous
        # five day span.
        cls.pipeline_start_date = cls.trading_days[14]
        cls.pipeline_end_date = cls.trading_days[18]

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
            self.pipeline_start_date,
            self.pipeline_end_date,
        )

    @parameter_space(unmasked_column=[0, 1, 2], slice_column=[0, 1, 2])
    def test_slice_with_masking(self, unmasked_column, slice_column):
        """
        Test that masking a factor that uses slices as inputs does not mask the
        slice data.
        """
        sids = self.sids
        asset_finder = self.asset_finder
        start_date = self.pipeline_start_date
        end_date = self.pipeline_end_date

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
        my_asset = Asset(0, exchange="TEST")
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
                self.pipeline_start_date,
                self.pipeline_end_date,
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
                self.pipeline_start_date,
                self.pipeline_end_date,
            )

        # Make sure that slices of custom factors are not window safe.
        class MyUnsafeFactor(CustomFactor):
            window_length = 1
            inputs = [col]

            def compute(self, today, assets, out, col):
                pass

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
                self.pipeline_start_date,
                self.pipeline_end_date,
            )

        # Create a window safe factor.
        class MySafeFactor(CustomFactor):
            window_length = 1
            inputs = [col]
            window_safe = True

            def compute(self, today, assets, out, col):
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

    def test_single_column_output(self):
        """
        Tests for custom factors that compute a 1D out.
        """
        start_date = self.pipeline_start_date
        end_date = self.pipeline_end_date

        alternating_mask = (AssetIDPlusDay() % 2).eq(0)
        cascading_mask = AssetIDPlusDay() < (self.sids[-1] + start_date.day)

        class SingleColumnOutput(CustomFactor):
            window_length = 1
            inputs = [self.col]
            window_safe = True
            ndim = 1

            def compute(self, today, assets, out, col):
                # Because we specified ndim as 1, `out` should be a singleton
                # array but `close` should be a regular sized input.
                assert out.shape == (1,)
                assert col.shape == (1, 3)
                out[:] = col.sum()

        # Since we cannot add single column output factors as pipeline
        # columns, we have to test its output through another factor.
        class UsesSingleColumnOutput(CustomFactor):
            window_length = 1
            inputs = [SingleColumnOutput()]

            def compute(self, today, assets, out, single_column_output):
                # Make sure that `single_column` has the correct shape. That
                # is, it should always have one column regardless of any mask
                # passed to `UsesSingleColumnInput`.
                assert single_column_output.shape == (1, 1)

        for mask in (alternating_mask, cascading_mask):
            columns = {
                'uses_single_column_output': UsesSingleColumnOutput(),
                'uses_single_column_output_masked': UsesSingleColumnOutput(
                    mask=mask,
                ),
            }

            # Assertions about the expected shapes of our data are made in the
            # `compute` function of our custom factors above.
            self.run_pipeline(Pipeline(columns=columns), start_date, end_date)

    def test_masked_single_column_output(self):
        """
        Tests for masking custom factors that compute a 1D out.
        """
        start_date = self.pipeline_start_date
        end_date = self.pipeline_end_date

        alternating_mask = (AssetIDPlusDay() % 2).eq(0)
        cascading_mask = AssetIDPlusDay() < (self.sids[-1] + start_date.day)
        alternating_mask.window_safe = True
        cascading_mask.window_safe = True

        for mask in (alternating_mask, cascading_mask):
            class SingleColumnOutput(CustomFactor):
                window_length = 1
                inputs = [self.col, mask]
                window_safe = True
                ndim = 1

                def compute(self, today, assets, out, col, mask):
                    # Because we specified ndim as 1, `out` should always be a
                    # singleton array but `close` should be a sized based on
                    # the mask we passed.
                    assert out.shape == (1,)
                    assert col.shape == (1, mask.sum())
                    out[:] = col.sum()

            # Since we cannot add single column output factors as pipeline
            # columns, we have to test its output through another factor.
            class UsesSingleColumnInput(CustomFactor):
                window_length = 1
                inputs = [self.col, mask, SingleColumnOutput(mask=mask)]

                def compute(self,
                            today,
                            assets,
                            out,
                            col,
                            mask,
                            single_column_output):
                    # Make sure that `single_column` has the correct value
                    # based on the masked it used.
                    assert single_column_output.shape == (1, 1)
                    single_column_output_value = single_column_output[0][0]
                    expected_value = where(mask, col, 0).sum()
                    assert single_column_output_value == expected_value

            columns = {'uses_single_column_input': UsesSingleColumnInput()}

            # Assertions about the expected shapes of our data are made in the
            # `compute` function of our custom factors above.
            self.run_pipeline(Pipeline(columns=columns), start_date, end_date)

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
            Pipeline(columns=columns),
            self.pipeline_start_date,
            self.pipeline_end_date,
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
            Pipeline(columns=columns),
            self.pipeline_start_date,
            self.pipeline_end_date,
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
