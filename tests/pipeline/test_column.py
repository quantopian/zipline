"""
Tests BoundColumn attributes and methods.
"""
from unittest import skipIf

from pandas import Timestamp, DataFrame
from pandas.util.testing import assert_frame_equal

from zipline.lib.labelarray import LabelArray
from zipline.pipeline import Pipeline
from zipline.pipeline.data.testing import TestingDataSet as TDS
from zipline.testing.fixtures import (
    WithSeededRandomPipelineEngine,
    WithTradingSessions,
    ZiplineTestCase
)
from zipline.utils.pandas_utils import ignore_pandas_nan_categorical_warning, \
    new_pandas, skip_pipeline_new_pandas


class LatestTestCase(WithSeededRandomPipelineEngine,
                     WithTradingSessions,
                     ZiplineTestCase):

    START_DATE = Timestamp('2014-01-01')
    END_DATE = Timestamp('2015-12-31')
    SEEDED_RANDOM_PIPELINE_SEED = 100
    ASSET_FINDER_EQUITY_SIDS = list(range(5))

    @classmethod
    def init_class_fixtures(cls):
        super(LatestTestCase, cls).init_class_fixtures()
        cls.engine = cls.seeded_random_engine
        cls.sids = cls.ASSET_FINDER_EQUITY_SIDS
        cls.assets = cls.engine._finder.retrieve_all(
            cls.ASSET_FINDER_EQUITY_SIDS)

    def expected_latest(self, column, slice_):
        loader = self.engine.get_loader(column)

        index = self.trading_days[slice_]
        columns = self.assets
        values = loader.values(column.dtype, self.trading_days, self.sids)[
            slice_]

        if column.dtype.kind in ('O', 'S', 'U'):
            # For string columns, we expect a categorical in the output.
            return LabelArray(
                values,
                missing_value=column.missing_value,
            ).as_categorical_frame(
                index=index,
                columns=columns,
            )

        return DataFrame(
            loader.values(column.dtype, self.trading_days, self.sids)[slice_],
            index=self.trading_days[slice_],
            columns=self.assets,
        )

    @skipIf(new_pandas, skip_pipeline_new_pandas)
    def test_latest(self):
        columns = TDS.columns
        pipe = Pipeline(
            columns={c.name: c.latest for c in columns},
        )

        cal_slice = slice(20, 40)
        dates_to_test = self.trading_days[cal_slice]
        result = self.engine.run_pipeline(
            pipe,
            dates_to_test[0],
            dates_to_test[-1],
        )
        for column in columns:
            with ignore_pandas_nan_categorical_warning():
                col_result = result[column.name].unstack()

            expected_col_result = self.expected_latest(column, cal_slice)
            assert_frame_equal(col_result, expected_col_result)
