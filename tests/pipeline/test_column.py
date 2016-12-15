"""
Tests BoundColumn attributes and methods.
"""
from contextlib2 import ExitStack
from unittest import TestCase

from pandas import date_range, DataFrame
from pandas.util.testing import assert_frame_equal

from zipline.lib.labelarray import LabelArray
from zipline.pipeline import Pipeline
from zipline.pipeline.data.testing import TestingDataSet as TDS
from zipline.testing import chrange, temp_pipeline_engine
from zipline.utils.pandas_utils import ignore_pandas_nan_categorical_warning


class LatestTestCase(TestCase):

    @classmethod
    def setUpClass(cls):
        cls._stack = stack = ExitStack()
        cls.calendar = cal = date_range('2014', '2015', freq='D', tz='UTC')
        cls.sids = list(range(5))
        cls.engine = stack.enter_context(
            temp_pipeline_engine(
                cal,
                cls.sids,
                random_seed=100,
                symbols=chrange('A', 'E'),
            ),
        )
        cls.assets = cls.engine._finder.retrieve_all(cls.sids)

    @classmethod
    def tearDownClass(cls):
        cls._stack.close()

    def expected_latest(self, column, slice_):
        loader = self.engine.get_loader(column)

        index = self.calendar[slice_]
        columns = self.assets
        values = loader.values(column.dtype, self.calendar, self.sids)[slice_]

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
            loader.values(column.dtype, self.calendar, self.sids)[slice_],
            index=self.calendar[slice_],
            columns=self.assets,
        )

    def test_latest(self):
        columns = TDS.columns
        pipe = Pipeline(
            columns={c.name: c.latest for c in columns},
        )

        cal_slice = slice(20, 40)
        dates_to_test = self.calendar[cal_slice]
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
