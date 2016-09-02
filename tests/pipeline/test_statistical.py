"""
Tests for statistical pipeline terms.
"""
from numpy import (
    arange,
    full,
    full_like,
    nan,
    where,
)
from pandas import (
    DataFrame,
    date_range,
    Int64Index,
    Timestamp,
)
from pandas.util.testing import assert_frame_equal
from scipy.stats import linregress, pearsonr, spearmanr

from zipline.assets import Equity
from zipline.errors import IncompatibleTerms, NonExistentAssetInTimeFrame
from zipline.pipeline import CustomFactor, Pipeline
from zipline.pipeline.data import USEquityPricing
from zipline.pipeline.data.testing import TestingDataSet
from zipline.pipeline.engine import SimplePipelineEngine
from zipline.pipeline.factors import (
    Returns,
    RollingLinearRegressionOfReturns,
    RollingPearsonOfReturns,
    RollingSpearmanOfReturns,
)
from zipline.pipeline.loaders.frame import DataFrameLoader
from zipline.pipeline.sentinels import NotSpecified
from zipline.testing import (
    AssetID,
    AssetIDPlusDay,
    check_arrays,
    make_alternating_boolean_array,
    make_cascading_boolean_array,
    parameter_space,
)
from zipline.testing.fixtures import (
    WithSeededRandomPipelineEngine,
    WithTradingEnvironment,
    ZiplineTestCase,
)
from zipline.utils.numpy_utils import (
    bool_dtype,
    datetime64ns_dtype,
    float64_dtype,
)


class StatisticalBuiltInsTestCase(WithTradingEnvironment, ZiplineTestCase):
    sids = ASSET_FINDER_EQUITY_SIDS = Int64Index([1, 2, 3])
    START_DATE = Timestamp('2015-01-31', tz='UTC')
    END_DATE = Timestamp('2015-03-01', tz='UTC')

    @classmethod
    def init_class_fixtures(cls):
        super(StatisticalBuiltInsTestCase, cls).init_class_fixtures()

        day = cls.trading_calendar.day
        cls.dates = dates = date_range(
            '2015-02-01', '2015-02-28', freq=day, tz='UTC',
        )

        # Using these start and end dates because they are a contigous span of
        # 5 days (Monday - Friday) and they allow for plenty of days to look
        # back on when computing correlations and regressions.
        cls.start_date_index = start_date_index = 14
        cls.end_date_index = end_date_index = 18
        cls.pipeline_start_date = dates[start_date_index]
        cls.pipeline_end_date = dates[end_date_index]
        cls.num_days = num_days = end_date_index - start_date_index + 1

        sids = cls.sids
        cls.assets = assets = cls.asset_finder.retrieve_all(sids)
        cls.my_asset_column = my_asset_column = 0
        cls.my_asset = assets[my_asset_column]
        cls.num_assets = num_assets = len(assets)

        cls.raw_data = raw_data = DataFrame(
            data=arange(len(dates) * len(sids), dtype=float64_dtype).reshape(
                len(dates), len(sids),
            ),
            index=dates,
            columns=assets,
        )

        # Using mock 'close' data here because the correlation and regression
        # built-ins use USEquityPricing.close as the input to their `Returns`
        # factors. Since there is no way to change that when constructing an
        # instance of these built-ins, we need to test with mock 'close' data
        # to most accurately reflect their true behavior and results.
        close_loader = DataFrameLoader(USEquityPricing.close, raw_data)

        cls.run_pipeline = SimplePipelineEngine(
            {USEquityPricing.close: close_loader}.__getitem__,
            dates,
            cls.asset_finder,
        ).run_pipeline

        cls.cascading_mask = \
            AssetIDPlusDay() < (sids[-1] + dates[start_date_index].day)
        cls.expected_cascading_mask_result = make_cascading_boolean_array(
            shape=(num_days, num_assets),
        )
        cls.alternating_mask = (AssetIDPlusDay() % 2).eq(0)
        cls.expected_alternating_mask_result = make_alternating_boolean_array(
            shape=(num_days, num_assets),
        )
        cls.expected_no_mask_result = full(
            shape=(num_days, num_assets), fill_value=True, dtype=bool_dtype,
        )

    @parameter_space(returns_length=[2, 3], correlation_length=[3, 4])
    def test_correlation_factors(self, returns_length, correlation_length):
        """
        Tests for the built-in factors `RollingPearsonOfReturns` and
        `RollingSpearmanOfReturns`.
        """
        assets = self.assets
        my_asset = self.my_asset
        my_asset_column = self.my_asset_column
        dates = self.dates
        start_date = self.pipeline_start_date
        end_date = self.pipeline_end_date
        start_date_index = self.start_date_index
        end_date_index = self.end_date_index
        num_days = self.num_days
        run_pipeline = self.run_pipeline

        returns = Returns(window_length=returns_length)
        masks = (self.cascading_mask, self.alternating_mask, NotSpecified)
        expected_mask_results = (
            self.expected_cascading_mask_result,
            self.expected_alternating_mask_result,
            self.expected_no_mask_result,
        )

        for mask, expected_mask in zip(masks, expected_mask_results):
            pearson_factor = RollingPearsonOfReturns(
                target=my_asset,
                returns_length=returns_length,
                correlation_length=correlation_length,
                mask=mask,
            )
            spearman_factor = RollingSpearmanOfReturns(
                target=my_asset,
                returns_length=returns_length,
                correlation_length=correlation_length,
                mask=mask,
            )

            columns = {
                'pearson_factor': pearson_factor,
                'spearman_factor': spearman_factor,
            }
            pipeline = Pipeline(columns=columns)
            if mask is not NotSpecified:
                pipeline.add(mask, 'mask')

            results = run_pipeline(pipeline, start_date, end_date)
            pearson_results = results['pearson_factor'].unstack()
            spearman_results = results['spearman_factor'].unstack()
            if mask is not NotSpecified:
                mask_results = results['mask'].unstack()
                check_arrays(mask_results.values, expected_mask)

            # Run a separate pipeline that calculates returns starting
            # (correlation_length - 1) days prior to our start date. This is
            # because we need (correlation_length - 1) extra days of returns to
            # compute our expected correlations.
            results = run_pipeline(
                Pipeline(columns={'returns': returns}),
                dates[start_date_index - (correlation_length - 1)],
                dates[end_date_index],
            )
            returns_results = results['returns'].unstack()

            # On each day, calculate the expected correlation coefficients
            # between the asset we are interested in and each other asset. Each
            # correlation is calculated over `correlation_length` days.
            expected_pearson_results = full_like(pearson_results, nan)
            expected_spearman_results = full_like(spearman_results, nan)
            for day in range(num_days):
                todays_returns = returns_results.iloc[
                    day:day + correlation_length
                ]
                my_asset_returns = todays_returns.iloc[:, my_asset_column]
                for asset, other_asset_returns in todays_returns.iteritems():
                    asset_column = int(asset) - 1
                    expected_pearson_results[day, asset_column] = pearsonr(
                        my_asset_returns, other_asset_returns,
                    )[0]
                    expected_spearman_results[day, asset_column] = spearmanr(
                        my_asset_returns, other_asset_returns,
                    )[0]

            expected_pearson_results = DataFrame(
                data=where(expected_mask, expected_pearson_results, nan),
                index=dates[start_date_index:end_date_index + 1],
                columns=assets,
            )
            assert_frame_equal(pearson_results, expected_pearson_results)

            expected_spearman_results = DataFrame(
                data=where(expected_mask, expected_spearman_results, nan),
                index=dates[start_date_index:end_date_index + 1],
                columns=assets,
            )
            assert_frame_equal(spearman_results, expected_spearman_results)

    @parameter_space(returns_length=[2, 3], regression_length=[3, 4])
    def test_regression_of_returns_factor(self,
                                          returns_length,
                                          regression_length):
        """
        Tests for the built-in factor `RollingLinearRegressionOfReturns`.
        """
        assets = self.assets
        my_asset = self.my_asset
        my_asset_column = self.my_asset_column
        dates = self.dates
        start_date = self.pipeline_start_date
        end_date = self.pipeline_end_date
        start_date_index = self.start_date_index
        end_date_index = self.end_date_index
        num_days = self.num_days
        run_pipeline = self.run_pipeline

        # The order of these is meant to align with the output of `linregress`.
        outputs = ['beta', 'alpha', 'r_value', 'p_value', 'stderr']

        returns = Returns(window_length=returns_length)
        masks = self.cascading_mask, self.alternating_mask, NotSpecified
        expected_mask_results = (
            self.expected_cascading_mask_result,
            self.expected_alternating_mask_result,
            self.expected_no_mask_result,
        )

        for mask, expected_mask in zip(masks, expected_mask_results):
            regression_factor = RollingLinearRegressionOfReturns(
                target=my_asset,
                returns_length=returns_length,
                regression_length=regression_length,
                mask=mask,
            )

            columns = {
                output: getattr(regression_factor, output)
                for output in outputs
            }
            pipeline = Pipeline(columns=columns)
            if mask is not NotSpecified:
                pipeline.add(mask, 'mask')

            results = run_pipeline(pipeline, start_date, end_date)
            if mask is not NotSpecified:
                mask_results = results['mask'].unstack()
                check_arrays(mask_results.values, expected_mask)

            output_results = {}
            expected_output_results = {}
            for output in outputs:
                output_results[output] = results[output].unstack()
                expected_output_results[output] = full_like(
                    output_results[output], nan,
                )

            # Run a separate pipeline that calculates returns starting
            # (regression_length - 1) days prior to our start date. This is
            # because we need (regression_length - 1) extra days of returns to
            # compute our expected regressions.
            results = run_pipeline(
                Pipeline(columns={'returns': returns}),
                dates[start_date_index - (regression_length - 1)],
                dates[end_date_index],
            )
            returns_results = results['returns'].unstack()

            # On each day, calculate the expected regression results for Y ~ X
            # where Y is the asset we are interested in and X is each other
            # asset. Each regression is calculated over `regression_length`
            # days of data.
            for day in range(num_days):
                todays_returns = returns_results.iloc[
                    day:day + regression_length
                ]
                my_asset_returns = todays_returns.iloc[:, my_asset_column]
                for asset, other_asset_returns in todays_returns.iteritems():
                    asset_column = int(asset) - 1
                    expected_regression_results = linregress(
                        y=other_asset_returns, x=my_asset_returns,
                    )
                    for i, output in enumerate(outputs):
                        expected_output_results[output][day, asset_column] = \
                            expected_regression_results[i]

            for output in outputs:
                output_result = output_results[output]
                expected_output_result = DataFrame(
                    where(expected_mask, expected_output_results[output], nan),
                    index=dates[start_date_index:end_date_index + 1],
                    columns=assets,
                )
                assert_frame_equal(output_result, expected_output_result)

    def test_correlation_and_regression_with_bad_asset(self):
        """
        Test that `RollingPearsonOfReturns`, `RollingSpearmanOfReturns` and
        `RollingLinearRegressionOfReturns` raise the proper exception when
        given a nonexistent target asset.
        """
        my_asset = Equity(0, exchange="TEST")
        start_date = self.pipeline_start_date
        end_date = self.pipeline_end_date
        run_pipeline = self.run_pipeline

        # This filter is arbitrary; the important thing is that we test each
        # factor both with and without a specified mask.
        my_asset_filter = AssetID().eq(1)

        for mask in (NotSpecified, my_asset_filter):
            pearson_factor = RollingPearsonOfReturns(
                target=my_asset,
                returns_length=3,
                correlation_length=3,
                mask=mask,
            )
            spearman_factor = RollingSpearmanOfReturns(
                target=my_asset,
                returns_length=3,
                correlation_length=3,
                mask=mask,
            )
            regression_factor = RollingLinearRegressionOfReturns(
                target=my_asset,
                returns_length=3,
                regression_length=3,
                mask=mask,
            )

            with self.assertRaises(NonExistentAssetInTimeFrame):
                run_pipeline(
                    Pipeline(columns={'pearson_factor': pearson_factor}),
                    start_date,
                    end_date,
                )
            with self.assertRaises(NonExistentAssetInTimeFrame):
                run_pipeline(
                    Pipeline(columns={'spearman_factor': spearman_factor}),
                    start_date,
                    end_date,
                )
            with self.assertRaises(NonExistentAssetInTimeFrame):
                run_pipeline(
                    Pipeline(columns={'regression_factor': regression_factor}),
                    start_date,
                    end_date,
                )

    def test_require_length_greater_than_one(self):
        my_asset = Equity(0, exchange="TEST")

        with self.assertRaises(ValueError):
            RollingPearsonOfReturns(
                target=my_asset,
                returns_length=3,
                correlation_length=1,
            )

        with self.assertRaises(ValueError):
            RollingSpearmanOfReturns(
                target=my_asset,
                returns_length=3,
                correlation_length=1,
            )

        with self.assertRaises(ValueError):
            RollingLinearRegressionOfReturns(
                target=my_asset,
                returns_length=3,
                regression_length=1,
            )


class StatisticalMethodsTestCase(WithSeededRandomPipelineEngine,
                                 ZiplineTestCase):
    sids = ASSET_FINDER_EQUITY_SIDS = Int64Index([1, 2, 3])
    START_DATE = Timestamp('2015-01-31', tz='UTC')
    END_DATE = Timestamp('2015-03-01', tz='UTC')

    @classmethod
    def init_class_fixtures(cls):
        super(StatisticalMethodsTestCase, cls).init_class_fixtures()

        # Using these start and end dates because they are a contigous span of
        # 5 days (Monday - Friday) and they allow for plenty of days to look
        # back on when computing correlations and regressions.
        cls.dates = dates = cls.trading_days
        cls.start_date_index = start_date_index = 14
        cls.end_date_index = end_date_index = 18
        cls.pipeline_start_date = cls.trading_days[start_date_index]
        cls.pipeline_end_date = cls.trading_days[end_date_index]

        sids = cls.sids
        cls.assets = assets = cls.asset_finder.retrieve_all(sids)
        cls.my_asset_column = my_asset_column = 0
        cls.my_asset = assets[my_asset_column]
        cls.num_days = num_days = end_date_index - start_date_index + 1
        cls.num_assets = num_assets = len(assets)

        cls.cascading_mask = \
            AssetIDPlusDay() < (sids[-1] + dates[start_date_index].day)
        cls.expected_cascading_mask_result = make_cascading_boolean_array(
            shape=(num_days, num_assets),
        )
        cls.alternating_mask = (AssetIDPlusDay() % 2).eq(0)
        cls.expected_alternating_mask_result = make_alternating_boolean_array(
            shape=(num_days, num_assets),
        )
        cls.expected_no_mask_result = full(
            shape=(num_days, num_assets), fill_value=True, dtype=bool_dtype,
        )

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
        my_asset = self.my_asset
        start_date = self.pipeline_start_date
        end_date = self.pipeline_end_date
        run_pipeline = self.run_pipeline

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

        results = run_pipeline(Pipeline(columns=columns), start_date, end_date)
        pearson_results = results['pearson'].unstack()
        spearman_results = results['spearman'].unstack()
        expected_pearson_results = results['expected_pearson'].unstack()
        expected_spearman_results = results['expected_spearman'].unstack()

        assert_frame_equal(pearson_results, expected_pearson_results)
        assert_frame_equal(spearman_results, expected_spearman_results)

    def test_correlation_methods_bad_type(self):
        """
        Make sure we cannot call the Factor correlation methods on factors or
        slices that are not of float or int dtype.
        """
        # These are arbitrary for the purpose of this test.
        returns_length = 2
        correlation_length = 10

        returns = Returns(window_length=returns_length, inputs=[self.col])
        returns_slice = returns[self.my_asset]

        class BadTypeFactor(CustomFactor):
            inputs = []
            window_length = 1
            dtype = datetime64ns_dtype
            window_safe = True

            def compute(self, today, assets, out):
                pass

        bad_type_factor = BadTypeFactor()
        bad_type_factor_slice = bad_type_factor[self.my_asset]

        with self.assertRaises(TypeError):
            bad_type_factor.pearsonr(
                target=returns_slice, correlation_length=correlation_length,
            )
        with self.assertRaises(TypeError):
            bad_type_factor.spearmanr(
                target=returns_slice, correlation_length=correlation_length,
            )
        with self.assertRaises(TypeError):
            returns.pearsonr(
                target=bad_type_factor_slice,
                correlation_length=correlation_length,
            )
        with self.assertRaises(TypeError):
            returns.spearmanr(
                target=bad_type_factor_slice,
                correlation_length=correlation_length,
            )

    @parameter_space(returns_length=[2, 3], regression_length=[3, 4])
    def test_factor_regression_method(self, returns_length, regression_length):
        """
        Ensure that `Factor.linear_regression` is consistent with the built-in
        factor `RollingLinearRegressionOfReturns`.
        """
        my_asset = self.my_asset
        start_date = self.pipeline_start_date
        end_date = self.pipeline_end_date
        run_pipeline = self.run_pipeline

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

        # This built-in constructs its own Returns factor to use as an input,
        # so the only way to set our own input is to do so after the fact. This
        # should not be done in practice. It is necessary here because we want
        # Returns to use our random data as an input, but by default it is
        # using USEquityPricing.close.
        expected_regression.inputs = [returns, returns_slice]

        columns = {
            'regression': regression,
            'expected_regression': expected_regression,
        }

        results = run_pipeline(Pipeline(columns=columns), start_date, end_date)
        regression_results = results['regression'].unstack()
        expected_regression_results = results['expected_regression'].unstack()

        assert_frame_equal(regression_results, expected_regression_results)

    def test_regression_method_bad_type(self):
        """
        Make sure we cannot call the Factor linear regression method on factors
        or slices that are not of float or int dtype.
        """
        # These are arbitrary for the purpose of this test.
        returns_length = 2
        regression_length = 10

        returns = Returns(window_length=returns_length, inputs=[self.col])
        returns_slice = returns[self.my_asset]

        class BadTypeFactor(CustomFactor):
            window_length = 1
            inputs = []
            dtype = datetime64ns_dtype
            window_safe = True

            def compute(self, today, assets, out):
                pass

        bad_type_factor = BadTypeFactor()
        bad_type_factor_slice = bad_type_factor[self.my_asset]

        with self.assertRaises(TypeError):
            bad_type_factor.linear_regression(
                target=returns_slice, regression_length=regression_length,
            )
        with self.assertRaises(TypeError):
            returns.linear_regression(
                target=bad_type_factor_slice,
                regression_length=regression_length,
            )

    @parameter_space(correlation_length=[2, 3, 4])
    def test_factor_correlation_methods_two_factors(self, correlation_length):
        """
        Tests for `Factor.pearsonr` and `Factor.spearmanr` when passed another
        2D factor instead of a Slice.
        """
        assets = self.assets
        dates = self.dates
        start_date = self.pipeline_start_date
        end_date = self.pipeline_end_date
        start_date_index = self.start_date_index
        end_date_index = self.end_date_index
        num_days = self.num_days
        run_pipeline = self.run_pipeline

        # Ensure that the correlation methods cannot be called with two 2D
        # factors which have different masks.
        returns_masked_1 = Returns(
            window_length=5, inputs=[self.col], mask=AssetID().eq(1),
        )
        returns_masked_2 = Returns(
            window_length=5, inputs=[self.col], mask=AssetID().eq(2),
        )
        with self.assertRaises(IncompatibleTerms):
            returns_masked_1.pearsonr(
                target=returns_masked_2, correlation_length=correlation_length,
            )
        with self.assertRaises(IncompatibleTerms):
            returns_masked_1.spearmanr(
                target=returns_masked_2, correlation_length=correlation_length,
            )

        returns_5 = Returns(window_length=5, inputs=[self.col])
        returns_10 = Returns(window_length=10, inputs=[self.col])

        pearson_factor = returns_5.pearsonr(
            target=returns_10, correlation_length=correlation_length,
        )
        spearman_factor = returns_5.spearmanr(
            target=returns_10, correlation_length=correlation_length,
        )

        columns = {
            'pearson_factor': pearson_factor,
            'spearman_factor': spearman_factor,
        }
        pipeline = Pipeline(columns=columns)

        results = run_pipeline(pipeline, start_date, end_date)
        pearson_results = results['pearson_factor'].unstack()
        spearman_results = results['spearman_factor'].unstack()

        # Run a separate pipeline that calculates returns starting
        # (correlation_length - 1) days prior to our start date. This is
        # because we need (correlation_length - 1) extra days of returns to
        # compute our expected correlations.
        columns = {'returns_5': returns_5, 'returns_10': returns_10}
        results = run_pipeline(
            Pipeline(columns=columns),
            dates[start_date_index - (correlation_length - 1)],
            dates[end_date_index],
        )
        returns_5_results = results['returns_5'].unstack()
        returns_10_results = results['returns_10'].unstack()

        # On each day, calculate the expected correlation coefficients
        # between each asset's 5 and 10 day rolling returns. Each correlation
        # is calculated over `correlation_length` days.
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

        expected_pearson_results = DataFrame(
            data=expected_pearson_results,
            index=dates[start_date_index:end_date_index + 1],
            columns=assets,
        )
        assert_frame_equal(pearson_results, expected_pearson_results)

        expected_spearman_results = DataFrame(
            data=expected_spearman_results,
            index=dates[start_date_index:end_date_index + 1],
            columns=assets,
        )
        assert_frame_equal(spearman_results, expected_spearman_results)

    @parameter_space(regression_length=[2, 3, 4])
    def test_factor_regression_method_two_factors(self, regression_length):
        """
        Tests for `Factor.linear_regression` when passed another 2D factor
        instead of a Slice.
        """
        assets = self.assets
        dates = self.dates
        start_date = self.pipeline_start_date
        end_date = self.pipeline_end_date
        start_date_index = self.start_date_index
        end_date_index = self.end_date_index
        num_days = self.num_days
        run_pipeline = self.run_pipeline

        # The order of these is meant to align with the output of `linregress`.
        outputs = ['beta', 'alpha', 'r_value', 'p_value', 'stderr']

        # Ensure that the `linear_regression` method cannot be called with two
        # 2D factors which have different masks.
        returns_masked_1 = Returns(
            window_length=5, inputs=[self.col], mask=AssetID().eq(1),
        )
        returns_masked_2 = Returns(
            window_length=5, inputs=[self.col], mask=AssetID().eq(2),
        )
        with self.assertRaises(IncompatibleTerms):
            returns_masked_1.linear_regression(
                target=returns_masked_2, regression_length=regression_length,
            )

        returns_5 = Returns(window_length=5, inputs=[self.col])
        returns_10 = Returns(window_length=10, inputs=[self.col])

        regression_factor = returns_5.linear_regression(
            target=returns_10, regression_length=regression_length,
        )

        columns = {
            output: getattr(regression_factor, output)
            for output in outputs
        }
        pipeline = Pipeline(columns=columns)

        results = run_pipeline(pipeline, start_date, end_date)

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
        results = run_pipeline(
            Pipeline(columns=columns),
            dates[start_date_index - (regression_length - 1)],
            dates[end_date_index],
        )
        returns_5_results = results['returns_5'].unstack()
        returns_10_results = results['returns_10'].unstack()

        # On each day, for each asset, calculate the expected regression
        # results of Y ~ X where Y is the asset's rolling 5 day returns and X
        # is the asset's rolling 10 day returns. Each regression is calculated
        # over `regression_length` days of data.
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
                expected_output_results[output],
                index=dates[start_date_index:end_date_index + 1],
                columns=assets,
            )
            assert_frame_equal(output_result, expected_output_result)
