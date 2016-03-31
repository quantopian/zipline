"""
Tests for the blaze interface to the pipeline api.
"""
from __future__ import division

from collections import OrderedDict
from datetime import timedelta, time
from itertools import product, chain
from unittest import TestCase
import warnings

import blaze as bz
from datashape import dshape, var, Record
from nose_parameterized import parameterized
import numpy as np
from numpy.testing.utils import assert_array_almost_equal
from odo import odo
import pandas as pd
from pandas.util.testing import assert_frame_equal
from toolz import keymap, valmap, concatv
from toolz.curried import operator as op

from zipline.assets.synthetic import make_simple_equity_info
from zipline.pipeline import Pipeline, CustomFactor
from zipline.pipeline.data import DataSet, BoundColumn
from zipline.pipeline.engine import SimplePipelineEngine
from zipline.pipeline.loaders.blaze import (
    from_blaze,
    BlazeLoader,
    NoDeltasWarning,
)
from zipline.pipeline.loaders.blaze.core import (
    NonNumpyField,
    NonPipelineField,
    no_deltas_rules,
)
from zipline.utils.numpy_utils import (
    float64_dtype,
    int64_dtype,
    repeat_last_axis,
)
from zipline.testing import tmp_asset_finder

nameof = op.attrgetter('name')
dtypeof = op.attrgetter('dtype')
asset_infos = (
    (make_simple_equity_info(
        tuple(map(ord, 'ABC')),
        pd.Timestamp(0),
        pd.Timestamp('2015'),
    ),),
    (make_simple_equity_info(
        tuple(map(ord, 'ABCD')),
        pd.Timestamp(0),
        pd.Timestamp('2015'),
    ),),
)
with_extra_sid = parameterized.expand(asset_infos)
with_ignore_sid = parameterized.expand(
    product(chain.from_iterable(asset_infos), [True, False])
)


def _utc_localize_index_level_0(df):
    """``tz_localize`` the first level of a multiindexed dataframe to utc.

    Mutates df in place.
    """
    idx = df.index
    df.index = pd.MultiIndex.from_product(
        (idx.levels[0].tz_localize('utc'), idx.levels[1]),
        names=idx.names,
    )
    return df


class BlazeToPipelineTestCase(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dates = dates = pd.date_range('2014-01-01', '2014-01-03')
        dates = cls.dates.repeat(3)
        cls.sids = sids = ord('A'), ord('B'), ord('C')
        cls.df = df = pd.DataFrame({
            'sid': sids * 3,
            'value': (0., 1., 2., 1., 2., 3., 2., 3., 4.),
            'int_value': (0, 1, 2, 1, 2, 3, 2, 3, 4),
            'asof_date': dates,
            'timestamp': dates,
        })
        cls.dshape = dshape("""
        var * {
            sid: ?int64,
            value: ?float64,
            int_value: ?int64,
            asof_date: datetime,
            timestamp: datetime
        }
        """)
        cls.macro_df = df[df.sid == 65].drop('sid', axis=1)
        dshape_ = OrderedDict(cls.dshape.measure.fields)
        del dshape_['sid']
        cls.macro_dshape = var * Record(dshape_)

        cls.garbage_loader = BlazeLoader()
        cls.missing_values = {'int_value': 0}

    def test_tabular(self):
        name = 'expr'
        expr = bz.data(self.df, name=name, dshape=self.dshape)
        ds = from_blaze(
            expr,
            loader=self.garbage_loader,
            no_deltas_rule=no_deltas_rules.ignore,
            missing_values=self.missing_values,
        )
        self.assertEqual(ds.__name__, name)
        self.assertTrue(issubclass(ds, DataSet))

        self.assertIs(ds.value.dtype, float64_dtype)
        self.assertIs(ds.int_value.dtype, int64_dtype)

        self.assertTrue(np.isnan(ds.value.missing_value))
        self.assertEqual(ds.int_value.missing_value, 0)

        invalid_type_fields = ('asof_date',)

        for field in invalid_type_fields:
            with self.assertRaises(AttributeError) as e:
                getattr(ds, field)
            self.assertIn("'%s'" % field, str(e.exception))
            self.assertIn("'datetime'", str(e.exception))

        # test memoization
        self.assertIs(
            from_blaze(
                expr,
                loader=self.garbage_loader,
                no_deltas_rule=no_deltas_rules.ignore,
                missing_values=self.missing_values,
            ),
            ds,
        )

    def test_column(self):
        exprname = 'expr'
        expr = bz.data(self.df, name=exprname, dshape=self.dshape)
        value = from_blaze(
            expr.value,
            loader=self.garbage_loader,
            no_deltas_rule=no_deltas_rules.ignore,
            missing_values=self.missing_values,
        )
        self.assertEqual(value.name, 'value')
        self.assertIsInstance(value, BoundColumn)
        self.assertIs(value.dtype, float64_dtype)

        # test memoization
        self.assertIs(
            from_blaze(
                expr.value,
                loader=self.garbage_loader,
                no_deltas_rule=no_deltas_rules.ignore,
                missing_values=self.missing_values,
            ),
            value,
        )
        self.assertIs(
            from_blaze(
                expr,
                loader=self.garbage_loader,
                no_deltas_rule=no_deltas_rules.ignore,
                missing_values=self.missing_values,
            ).value,
            value,
        )

        # test the walk back up the tree
        self.assertIs(
            from_blaze(
                expr,
                loader=self.garbage_loader,
                no_deltas_rule=no_deltas_rules.ignore,
                missing_values=self.missing_values,
            ),
            value.dataset,
        )
        self.assertEqual(value.dataset.__name__, exprname)

    def test_missing_asof(self):
        expr = bz.data(
            self.df.loc[:, ['sid', 'value', 'timestamp']],
            name='expr',
            dshape="""
            var * {
                sid: ?int64,
                value: float64,
                timestamp: datetime,
            }""",
        )

        with self.assertRaises(TypeError) as e:
            from_blaze(
                expr,
                loader=self.garbage_loader,
                no_deltas_rule=no_deltas_rules.ignore,
            )
        self.assertIn("'asof_date'", str(e.exception))
        self.assertIn(repr(str(expr.dshape.measure)), str(e.exception))

    def test_auto_deltas(self):
        expr = bz.data(
            {'ds': self.df,
             'ds_deltas': pd.DataFrame(columns=self.df.columns)},
            dshape=var * Record((
                ('ds', self.dshape.measure),
                ('ds_deltas', self.dshape.measure),
            )),
        )
        loader = BlazeLoader()
        ds = from_blaze(
            expr.ds,
            loader=loader,
            missing_values=self.missing_values,
        )
        self.assertEqual(len(loader), 1)
        exprdata = loader[ds]
        self.assertTrue(exprdata.expr.isidentical(expr.ds))
        self.assertTrue(exprdata.deltas.isidentical(expr.ds_deltas))

    def test_auto_deltas_fail_warn(self):
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter('always')
            loader = BlazeLoader()
            expr = bz.data(self.df, dshape=self.dshape)
            from_blaze(
                expr,
                loader=loader,
                no_deltas_rule=no_deltas_rules.warn,
                missing_values=self.missing_values,
            )
        self.assertEqual(len(ws), 1)
        w = ws[0].message
        self.assertIsInstance(w, NoDeltasWarning)
        self.assertIn(str(expr), str(w))

    def test_auto_deltas_fail_raise(self):
        loader = BlazeLoader()
        expr = bz.data(self.df, dshape=self.dshape)
        with self.assertRaises(ValueError) as e:
            from_blaze(
                expr,
                loader=loader,
                no_deltas_rule=no_deltas_rules.raise_,
            )
        self.assertIn(str(expr), str(e.exception))

    def test_non_numpy_field(self):
        expr = bz.data(
            [],
            dshape="""
            var * {
                 a: datetime,
                 asof_date: datetime,
                 timestamp: datetime,
            }""",
        )
        ds = from_blaze(
            expr,
            loader=self.garbage_loader,
            no_deltas_rule=no_deltas_rules.ignore,
        )
        with self.assertRaises(AttributeError):
            ds.a
        self.assertIsInstance(object.__getattribute__(ds, 'a'), NonNumpyField)

    def test_non_pipeline_field(self):
        # NOTE: This test will fail if we ever allow string types in
        # the Pipeline API. If this happens, change the dtype of the `a` field
        # of expr to another type we don't allow.
        expr = bz.data(
            [],
            dshape="""
            var * {
                 a: string,
                 asof_date: datetime,
                 timestamp: datetime,
            }""",
        )
        ds = from_blaze(
            expr,
            loader=self.garbage_loader,
            no_deltas_rule=no_deltas_rules.ignore,
        )
        with self.assertRaises(AttributeError):
            ds.a
        self.assertIsInstance(
            object.__getattribute__(ds, 'a'),
            NonPipelineField,
        )

    def test_complex_expr(self):
        expr = bz.data(self.df, dshape=self.dshape)
        # put an Add in the table
        expr_with_add = bz.transform(expr, value=expr.value + 1)

        # Test that we can have complex expressions with no deltas
        from_blaze(
            expr_with_add,
            deltas=None,
            loader=self.garbage_loader,
            missing_values=self.missing_values,
        )

        with self.assertRaises(TypeError):
            from_blaze(
                expr.value + 1,  # put an Add in the column
                deltas=None,
                loader=self.garbage_loader,
                missing_values=self.missing_values,
            )

        deltas = bz.data(
            pd.DataFrame(columns=self.df.columns),
            dshape=self.dshape,
        )
        with self.assertRaises(TypeError):
            from_blaze(
                expr_with_add,
                deltas=deltas,
                loader=self.garbage_loader,
                missing_values=self.missing_values,
            )

        with self.assertRaises(TypeError):
            from_blaze(
                expr.value + 1,
                deltas=deltas,
                loader=self.garbage_loader,
                missing_values=self.missing_values,
            )

    def _test_id(self, df, dshape, expected, finder, add):
        expr = bz.data(df, name='expr', dshape=dshape)
        loader = BlazeLoader()
        ds = from_blaze(
            expr,
            loader=loader,
            no_deltas_rule=no_deltas_rules.ignore,
            missing_values=self.missing_values,
        )
        p = Pipeline()
        for a in add:
            p.add(getattr(ds, a).latest, a)
        dates = self.dates

        with tmp_asset_finder() as finder:
            result = SimplePipelineEngine(
                loader,
                dates,
                finder,
            ).run_pipeline(p, dates[0], dates[-1])

        assert_frame_equal(
            result,
            _utc_localize_index_level_0(expected),
            check_dtype=False,
        )

    def test_custom_query_time_tz(self):
        df = self.df.copy()
        df['timestamp'] = (
            pd.DatetimeIndex(df['timestamp'], tz='EST') +
            timedelta(hours=8, minutes=44)
        ).tz_convert('utc').tz_localize(None)
        df.ix[3:5, 'timestamp'] = pd.Timestamp('2014-01-01 13:45')
        expr = bz.data(df, name='expr', dshape=self.dshape)
        loader = BlazeLoader(data_query_time=time(8, 45), data_query_tz='EST')
        ds = from_blaze(
            expr,
            loader=loader,
            no_deltas_rule=no_deltas_rules.ignore,
            missing_values=self.missing_values,
        )
        p = Pipeline()
        p.add(ds.value.latest, 'value')
        p.add(ds.int_value.latest, 'int_value')
        dates = self.dates

        with tmp_asset_finder() as finder:
            result = SimplePipelineEngine(
                loader,
                dates,
                finder,
            ).run_pipeline(p, dates[0], dates[-1])

        expected = df.drop('asof_date', axis=1)
        expected['timestamp'] = expected['timestamp'].dt.normalize().astype(
            'datetime64[ns]',
        ).dt.tz_localize('utc')
        expected.ix[3:5, 'timestamp'] += timedelta(days=1)
        expected.set_index(['timestamp', 'sid'], inplace=True)
        expected.index = pd.MultiIndex.from_product((
            expected.index.levels[0],
            finder.retrieve_all(expected.index.levels[1]),
        ))
        assert_frame_equal(result, expected, check_dtype=False)

    def test_id(self):
        """
        input (self.df):
           asof_date  sid  timestamp  value
        0 2014-01-01   65 2014-01-01      0
        1 2014-01-01   66 2014-01-01      1
        2 2014-01-01   67 2014-01-01      2
        3 2014-01-02   65 2014-01-02      1
        4 2014-01-02   66 2014-01-02      2
        5 2014-01-02   67 2014-01-02      3
        6 2014-01-03   65 2014-01-03      2
        7 2014-01-03   66 2014-01-03      3
        8 2014-01-03   67 2014-01-03      4

        output (expected)
                                   value
        2014-01-01 Equity(65 [A])      0
                   Equity(66 [B])      1
                   Equity(67 [C])      2
        2014-01-02 Equity(65 [A])      1
                   Equity(66 [B])      2
                   Equity(67 [C])      3
        2014-01-03 Equity(65 [A])      2
                   Equity(66 [B])      3
                   Equity(67 [C])      4
        """
        with tmp_asset_finder() as finder:
            expected = self.df.drop('asof_date', axis=1).set_index(
                ['timestamp', 'sid'],
            )
            expected.index = pd.MultiIndex.from_product((
                expected.index.levels[0],
                finder.retrieve_all(expected.index.levels[1]),
            ))
            self._test_id(
                self.df, self.dshape, expected, finder, ('int_value', 'value',)
            )

    def test_id_ffill_out_of_window(self):
        """
        input (df):

           asof_date  timestamp  sid  other  value
        0 2013-12-22 2013-12-22   65      0      0
        1 2013-12-22 2013-12-22   66    NaN      1
        2 2013-12-22 2013-12-22   67      2    NaN
        3 2013-12-23 2013-12-23   65    NaN      1
        4 2013-12-23 2013-12-23   66      2    NaN
        5 2013-12-23 2013-12-23   67      3      3
        6 2013-12-24 2013-12-24   65      2    NaN
        7 2013-12-24 2013-12-24   66      3      3
        8 2013-12-24 2013-12-24   67    NaN      4

        output (expected):
                                   other  value
        2014-01-01 Equity(65 [A])      2      1
                   Equity(66 [B])      3      3
                   Equity(67 [C])      3      4
        2014-01-02 Equity(65 [A])      2      1
                   Equity(66 [B])      3      3
                   Equity(67 [C])      3      4
        2014-01-03 Equity(65 [A])      2      1
                   Equity(66 [B])      3      3
                   Equity(67 [C])      3      4
        """
        dates = self.dates.repeat(3) - timedelta(days=10)
        df = pd.DataFrame({
            'sid': self.sids * 3,
            'value': (0, 1, np.nan, 1, np.nan, 3, np.nan, 3, 4),
            'other': (0, np.nan, 2, np.nan, 2, 3, 2, 3, np.nan),
            'asof_date': dates,
            'timestamp': dates,
        })
        fields = OrderedDict(self.dshape.measure.fields)
        fields['other'] = fields['value']

        with tmp_asset_finder() as finder:
            expected = pd.DataFrame(
                np.array([[2, 1],
                          [3, 3],
                          [3, 4],
                          [2, 1],
                          [3, 3],
                          [3, 4],
                          [2, 1],
                          [3, 3],
                          [3, 4]]),
                columns=['other', 'value'],
                index=pd.MultiIndex.from_product(
                    (self.dates, finder.retrieve_all(self.sids)),
                ),
            )
            self._test_id(
                df,
                var * Record(fields),
                expected,
                finder,
                ('value', 'other'),
            )

    def test_id_multiple_columns(self):
        """
        input (df):
           asof_date  sid  timestamp  value  other
        0 2014-01-01   65 2014-01-01      0      1
        1 2014-01-01   66 2014-01-01      1      2
        2 2014-01-01   67 2014-01-01      2      3
        3 2014-01-02   65 2014-01-02      1      2
        4 2014-01-02   66 2014-01-02      2      3
        5 2014-01-02   67 2014-01-02      3      4
        6 2014-01-03   65 2014-01-03      2      3
        7 2014-01-03   66 2014-01-03      3      4
        8 2014-01-03   67 2014-01-03      4      5

        output (expected):
                                   value  other
        2014-01-01 Equity(65 [A])      0      1
                   Equity(66 [B])      1      2
                   Equity(67 [C])      2      3
        2014-01-02 Equity(65 [A])      1      2
                   Equity(66 [B])      2      3
                   Equity(67 [C])      3      4
        2014-01-03 Equity(65 [A])      2      3
                   Equity(66 [B])      3      4
                   Equity(67 [C])      4      5
        """
        df = self.df.copy()
        df['other'] = df.value + 1
        fields = OrderedDict(self.dshape.measure.fields)
        fields['other'] = fields['value']
        with tmp_asset_finder() as finder:
            expected = df.drop('asof_date', axis=1).set_index(
                ['timestamp', 'sid'],
            ).sort_index(axis=1)
            expected.index = pd.MultiIndex.from_product((
                expected.index.levels[0],
                finder.retrieve_all(expected.index.levels[1]),
            ))
            self._test_id(
                df,
                var * Record(fields),
                expected,
                finder,
                ('value', 'int_value', 'other'),
            )

    def test_id_macro_dataset(self):
        """
        input (self.macro_df)
           asof_date  timestamp  value
        0 2014-01-01 2014-01-01      0
        3 2014-01-02 2014-01-02      1
        6 2014-01-03 2014-01-03      2

        output (expected):
                                   value
        2014-01-01 Equity(65 [A])      0
                   Equity(66 [B])      0
                   Equity(67 [C])      0
        2014-01-02 Equity(65 [A])      1
                   Equity(66 [B])      1
                   Equity(67 [C])      1
        2014-01-03 Equity(65 [A])      2
                   Equity(66 [B])      2
                   Equity(67 [C])      2
        """
        asset_info = asset_infos[0][0]
        nassets = len(asset_info)
        with tmp_asset_finder() as finder:
            expected = pd.DataFrame(
                list(concatv([0] * nassets, [1] * nassets, [2] * nassets)),
                index=pd.MultiIndex.from_product((
                    self.macro_df.timestamp,
                    finder.retrieve_all(asset_info.index),
                )),
                columns=('value',),
            )
            self._test_id(
                self.macro_df,
                self.macro_dshape,
                expected,
                finder,
                ('value',),
            )

    def test_id_ffill_out_of_window_macro_dataset(self):
        """
        input (df):
           asof_date  timestamp  other  value
        0 2013-12-22 2013-12-22    NaN      0
        1 2013-12-23 2013-12-23      1    NaN
        2 2013-12-24 2013-12-24    NaN    NaN

        output (expected):
                                   other  value
        2014-01-01 Equity(65 [A])      1      0
                   Equity(66 [B])      1      0
                   Equity(67 [C])      1      0
        2014-01-02 Equity(65 [A])      1      0
                   Equity(66 [B])      1      0
                   Equity(67 [C])      1      0
        2014-01-03 Equity(65 [A])      1      0
                   Equity(66 [B])      1      0
                   Equity(67 [C])      1      0
        """
        dates = self.dates - timedelta(days=10)
        df = pd.DataFrame({
            'value': (0, np.nan, np.nan),
            'other': (np.nan, 1, np.nan),
            'asof_date': dates,
            'timestamp': dates,
        })
        fields = OrderedDict(self.macro_dshape.measure.fields)
        fields['other'] = fields['value']

        with tmp_asset_finder() as finder:
            expected = pd.DataFrame(
                np.array([[0, 1],
                          [0, 1],
                          [0, 1],
                          [0, 1],
                          [0, 1],
                          [0, 1],
                          [0, 1],
                          [0, 1],
                          [0, 1]]),
                columns=['value', 'other'],
                index=pd.MultiIndex.from_product(
                    (self.dates, finder.retrieve_all(self.sids)),
                ),
            ).sort_index(axis=1)
            self._test_id(
                df,
                var * Record(fields),
                expected,
                finder,
                ('value', 'other'),
            )

    def test_id_macro_dataset_multiple_columns(self):
        """
        input (df):
           asof_date  timestamp  other  value
        0 2014-01-01 2014-01-01      1      0
        3 2014-01-02 2014-01-02      2      1
        6 2014-01-03 2014-01-03      3      2

        output (expected):
                                   other  value
        2014-01-01 Equity(65 [A])      1      0
                   Equity(66 [B])      1      0
                   Equity(67 [C])      1      0
        2014-01-02 Equity(65 [A])      2      1
                   Equity(66 [B])      2      1
                   Equity(67 [C])      2      1
        2014-01-03 Equity(65 [A])      3      2
                   Equity(66 [B])      3      2
                   Equity(67 [C])      3      2
        """
        df = self.macro_df.copy()
        df['other'] = df.value + 1
        fields = OrderedDict(self.macro_dshape.measure.fields)
        fields['other'] = fields['value']

        asset_info = asset_infos[0][0]
        with tmp_asset_finder(equities=asset_info) as finder:
            expected = pd.DataFrame(
                np.array([[0, 1],
                          [1, 2],
                          [2, 3]]).repeat(3, axis=0),
                index=pd.MultiIndex.from_product((
                    df.timestamp,
                    finder.retrieve_all(asset_info.index),
                )),
                columns=('value', 'other'),
            ).sort_index(axis=1)
            self._test_id(
                df,
                var * Record(fields),
                expected,
                finder,
                ('value', 'other'),
            )

    def test_id_take_last_in_group(self):
        T = pd.Timestamp
        df = pd.DataFrame(
            columns=['asof_date',        'timestamp', 'sid', 'other', 'value'],
            data=[
                [T('2014-01-01'), T('2014-01-01 00'),    65,        0,      0],
                [T('2014-01-01'), T('2014-01-01 01'),    65,        1, np.nan],
                [T('2014-01-01'), T('2014-01-01 00'),    66,   np.nan, np.nan],
                [T('2014-01-01'), T('2014-01-01 01'),    66,   np.nan,      1],
                [T('2014-01-01'), T('2014-01-01 00'),    67,        2, np.nan],
                [T('2014-01-01'), T('2014-01-01 01'),    67,   np.nan, np.nan],
                [T('2014-01-02'), T('2014-01-02 00'),    65,   np.nan, np.nan],
                [T('2014-01-02'), T('2014-01-02 01'),    65,   np.nan,      1],
                [T('2014-01-02'), T('2014-01-02 00'),    66,   np.nan, np.nan],
                [T('2014-01-02'), T('2014-01-02 01'),    66,        2, np.nan],
                [T('2014-01-02'), T('2014-01-02 00'),    67,        3,      3],
                [T('2014-01-02'), T('2014-01-02 01'),    67,        3,      3],
                [T('2014-01-03'), T('2014-01-03 00'),    65,        2, np.nan],
                [T('2014-01-03'), T('2014-01-03 01'),    65,        2, np.nan],
                [T('2014-01-03'), T('2014-01-03 00'),    66,        3,      3],
                [T('2014-01-03'), T('2014-01-03 01'),    66,   np.nan, np.nan],
                [T('2014-01-03'), T('2014-01-03 00'),    67,   np.nan, np.nan],
                [T('2014-01-03'), T('2014-01-03 01'),    67,   np.nan,      4],
            ],
        )
        fields = OrderedDict(self.dshape.measure.fields)
        fields['other'] = fields['value']

        with tmp_asset_finder() as finder:
            expected = pd.DataFrame(
                columns=['other', 'value'],
                data=[
                    [1,           0],  # 2014-01-01 Equity(65 [A])
                    [np.nan,      1],             # Equity(66 [B])
                    [2,      np.nan],             # Equity(67 [C])
                    [1,           1],  # 2014-01-02 Equity(65 [A])
                    [2,           1],             # Equity(66 [B])
                    [3,           3],             # Equity(67 [C])
                    [2,           1],  # 2014-01-03 Equity(65 [A])
                    [3,           3],             # Equity(66 [B])
                    [3,           3],             # Equity(67 [C])
                ],
                index=pd.MultiIndex.from_product(
                    (self.dates, finder.retrieve_all(self.sids)),
                ),
            )
            self._test_id(
                df,
                var * Record(fields),
                expected,
                finder,
                ('value', 'other'),
            )

    def test_id_take_last_in_group_macro(self):
        """
        output (expected):

                                   other  value
        2014-01-01 Equity(65 [A])    NaN      1
                   Equity(66 [B])    NaN      1
                   Equity(67 [C])    NaN      1
        2014-01-02 Equity(65 [A])      1      2
                   Equity(66 [B])      1      2
                   Equity(67 [C])      1      2
        2014-01-03 Equity(65 [A])      2      2
                   Equity(66 [B])      2      2
                   Equity(67 [C])      2      2
         """
        T = pd.Timestamp
        df = pd.DataFrame(
            columns=['asof_date',        'timestamp', 'other', 'value'],
            data=[
                [T('2014-01-01'), T('2014-01-01 00'),   np.nan,      1],
                [T('2014-01-01'), T('2014-01-01 01'),   np.nan, np.nan],
                [T('2014-01-02'), T('2014-01-02 00'),        1, np.nan],
                [T('2014-01-02'), T('2014-01-02 01'),   np.nan,      2],
                [T('2014-01-03'), T('2014-01-03 00'),        2, np.nan],
                [T('2014-01-03'), T('2014-01-03 01'),        3,      3],
            ],
        )
        fields = OrderedDict(self.macro_dshape.measure.fields)
        fields['other'] = fields['value']

        with tmp_asset_finder() as finder:
            expected = pd.DataFrame(
                columns=[
                    'other', 'value',
                ],
                data=[
                    [np.nan,      1],  # 2014-01-01 Equity(65 [A])
                    [np.nan,      1],             # Equity(66 [B])
                    [np.nan,      1],             # Equity(67 [C])
                    [1,           2],  # 2014-01-02 Equity(65 [A])
                    [1,           2],             # Equity(66 [B])
                    [1,           2],             # Equity(67 [C])
                    [2,           2],  # 2014-01-03 Equity(65 [A])
                    [2,           2],             # Equity(66 [B])
                    [2,           2],             # Equity(67 [C])
                ],
                index=pd.MultiIndex.from_product(
                    (self.dates, finder.retrieve_all(self.sids)),
                ),
            )
            self._test_id(
                df,
                var * Record(fields),
                expected,
                finder,
                ('value', 'other'),
            )

    def _run_pipeline(self,
                      expr,
                      deltas,
                      expected_views,
                      expected_output,
                      finder,
                      calendar,
                      start,
                      end,
                      window_length,
                      compute_fn):
        loader = BlazeLoader()
        ds = from_blaze(
            expr,
            deltas,
            loader=loader,
            no_deltas_rule=no_deltas_rules.raise_,
            missing_values=self.missing_values,
        )
        p = Pipeline()

        # prevent unbound locals issue in the inner class
        window_length_ = window_length

        class TestFactor(CustomFactor):
            inputs = ds.value,
            window_length = window_length_

            def compute(self, today, assets, out, data):
                assert_array_almost_equal(data, expected_views[today])
                out[:] = compute_fn(data)

        p.add(TestFactor(), 'value')

        result = SimplePipelineEngine(
            loader,
            calendar,
            finder,
        ).run_pipeline(p, start, end)

        assert_frame_equal(
            result,
            _utc_localize_index_level_0(expected_output),
            check_dtype=False,
        )

    @with_ignore_sid
    def test_deltas(self, asset_info, add_extra_sid):
        df = self.df.copy()
        if add_extra_sid:
            extra_sid_df = pd.DataFrame({
                'asof_date': self.dates,
                'timestamp': self.dates,
                'sid': (ord('E'),) * 3,
                'value': (3., 4., 5.,),
                'int_value': (3, 4, 5),
            })
            df = df.append(extra_sid_df, ignore_index=True)
        expr = bz.data(df, name='expr', dshape=self.dshape)
        deltas = bz.data(df, dshape=self.dshape)
        deltas = bz.data(
            odo(
                bz.transform(
                    deltas,
                    value=deltas.value + 10,
                    timestamp=deltas.timestamp + timedelta(days=1),
                ),
                pd.DataFrame,
            ),
            name='delta',
            dshape=self.dshape,
        )
        expected_views = keymap(pd.Timestamp, {
            '2014-01-02': np.array([[10.0, 11.0, 12.0],
                                    [1.0, 2.0, 3.0]]),
            '2014-01-03': np.array([[11.0, 12.0, 13.0],
                                    [2.0, 3.0, 4.0]]),
            '2014-01-04': np.array([[12.0, 13.0, 14.0],
                                    [12.0, 13.0, 14.0]]),
        })

        nassets = len(asset_info)
        if nassets == 4:
            expected_views = valmap(
                lambda view: np.c_[view, [np.nan, np.nan]],
                expected_views,
            )
        with tmp_asset_finder(equities=asset_info) as finder:
            expected_output = pd.DataFrame(
                list(concatv([12] * nassets, [13] * nassets, [14] * nassets)),
                index=pd.MultiIndex.from_product((
                    sorted(expected_views.keys()),
                    finder.retrieve_all(asset_info.index),
                )),
                columns=('value',),
            )
            dates = self.dates
            dates = dates.insert(len(dates), dates[-1] + timedelta(days=1))
            self._run_pipeline(
                expr,
                deltas,
                expected_views,
                expected_output,
                finder,
                calendar=dates,
                start=dates[1],
                end=dates[-1],
                window_length=2,
                compute_fn=np.nanmax,
            )

    @with_extra_sid
    def test_deltas_only_one_delta_in_universe(self, asset_info):
        expr = bz.data(self.df, name='expr', dshape=self.dshape)
        deltas = pd.DataFrame({
            'sid': [65, 66],
            'asof_date': [self.dates[1], self.dates[0]],
            'timestamp': [self.dates[2], self.dates[1]],
            'value': [10, 11],
        })
        deltas = bz.data(deltas, name='deltas', dshape=self.dshape)
        expected_views = keymap(pd.Timestamp, {
            '2014-01-02': np.array([[0.0, 11.0, 2.0],
                                    [1.0, 2.0, 3.0]]),
            '2014-01-03': np.array([[10.0, 2.0, 3.0],
                                    [2.0, 3.0, 4.0]]),
            '2014-01-04': np.array([[2.0, 3.0, 4.0],
                                    [2.0, 3.0, 4.0]]),
        })

        nassets = len(asset_info)
        if nassets == 4:
            expected_views = valmap(
                lambda view: np.c_[view, [np.nan, np.nan]],
                expected_views,
            )

        with tmp_asset_finder(equities=asset_info) as finder:
            expected_output = pd.DataFrame(
                columns=[
                    'value',
                ],
                data=np.array([11, 10, 4]).repeat(len(asset_info.index)),
                index=pd.MultiIndex.from_product((
                    sorted(expected_views.keys()),
                    finder.retrieve_all(asset_info.index),
                )),
            )
            dates = self.dates
            dates = dates.insert(len(dates), dates[-1] + timedelta(days=1))
            self._run_pipeline(
                expr,
                deltas,
                expected_views,
                expected_output,
                finder,
                calendar=dates,
                start=dates[1],
                end=dates[-1],
                window_length=2,
                compute_fn=np.nanmax,
            )

    def test_deltas_macro(self):
        asset_info = asset_infos[0][0]
        expr = bz.data(self.macro_df, name='expr', dshape=self.macro_dshape)
        deltas = bz.data(
            self.macro_df.iloc[:-1],
            name='deltas',
            dshape=self.macro_dshape,
        )
        deltas = bz.transform(
            deltas,
            value=deltas.value + 10,
            timestamp=deltas.timestamp + timedelta(days=1),
        )

        nassets = len(asset_info)
        expected_views = keymap(pd.Timestamp, {
            '2014-01-02': repeat_last_axis(np.array([10.0, 1.0]), nassets),
            '2014-01-03': repeat_last_axis(np.array([11.0, 2.0]), nassets),
        })

        with tmp_asset_finder(equities=asset_info) as finder:
            expected_output = pd.DataFrame(
                list(concatv([10] * nassets, [11] * nassets)),
                index=pd.MultiIndex.from_product((
                    sorted(expected_views.keys()),
                    finder.retrieve_all(asset_info.index),
                )),
                columns=('value',),
            )
            dates = self.dates
            self._run_pipeline(
                expr,
                deltas,
                expected_views,
                expected_output,
                finder,
                calendar=dates,
                start=dates[1],
                end=dates[-1],
                window_length=2,
                compute_fn=np.nanmax,
            )

    @with_extra_sid
    def test_novel_deltas(self, asset_info):
        base_dates = pd.DatetimeIndex([
            pd.Timestamp('2014-01-01'),
            pd.Timestamp('2014-01-04')
        ])
        repeated_dates = base_dates.repeat(3)
        baseline = pd.DataFrame({
            'sid': self.sids * 2,
            'value': (0., 1., 2., 1., 2., 3.),
            'int_value': (0, 1, 2, 1, 2, 3),
            'asof_date': repeated_dates,
            'timestamp': repeated_dates,
        })
        expr = bz.data(baseline, name='expr', dshape=self.dshape)
        deltas = bz.data(
            odo(
                bz.transform(
                    expr,
                    value=expr.value + 10,
                    timestamp=expr.timestamp + timedelta(days=1),
                ),
                pd.DataFrame,
            ),
            name='delta',
            dshape=self.dshape,
        )
        expected_views = keymap(pd.Timestamp, {
            '2014-01-03': np.array([[10.0, 11.0, 12.0],
                                    [10.0, 11.0, 12.0],
                                    [10.0, 11.0, 12.0]]),
            '2014-01-06': np.array([[10.0, 11.0, 12.0],
                                    [10.0, 11.0, 12.0],
                                    [11.0, 12.0, 13.0]]),
        })
        if len(asset_info) == 4:
            expected_views = valmap(
                lambda view: np.c_[view, [np.nan, np.nan, np.nan]],
                expected_views,
            )
            expected_output_buffer = [10, 11, 12, np.nan, 11, 12, 13, np.nan]
        else:
            expected_output_buffer = [10, 11, 12, 11, 12, 13]

        cal = pd.DatetimeIndex([
            pd.Timestamp('2014-01-01'),
            pd.Timestamp('2014-01-02'),
            pd.Timestamp('2014-01-03'),
            # omitting the 4th and 5th to simulate a weekend
            pd.Timestamp('2014-01-06'),
        ])

        with tmp_asset_finder(equities=asset_info) as finder:
            expected_output = pd.DataFrame(
                expected_output_buffer,
                index=pd.MultiIndex.from_product((
                    sorted(expected_views.keys()),
                    finder.retrieve_all(asset_info.index),
                )),
                columns=('value',),
            )
            self._run_pipeline(
                expr,
                deltas,
                expected_views,
                expected_output,
                finder,
                calendar=cal,
                start=cal[2],
                end=cal[-1],
                window_length=3,
                compute_fn=op.itemgetter(-1),
            )

    def test_novel_deltas_macro(self):
        asset_info = asset_infos[0][0]
        base_dates = pd.DatetimeIndex([
            pd.Timestamp('2014-01-01'),
            pd.Timestamp('2014-01-04')
        ])
        baseline = pd.DataFrame({
            'value': (0, 1),
            'asof_date': base_dates,
            'timestamp': base_dates,
        })
        expr = bz.data(baseline, name='expr', dshape=self.macro_dshape)
        deltas = bz.data(baseline, name='deltas', dshape=self.macro_dshape)
        deltas = bz.transform(
            deltas,
            value=deltas.value + 10,
            timestamp=deltas.timestamp + timedelta(days=1),
        )

        nassets = len(asset_info)
        expected_views = keymap(pd.Timestamp, {
            '2014-01-03': repeat_last_axis(
                np.array([10.0, 10.0, 10.0]),
                nassets,
            ),
            '2014-01-06': repeat_last_axis(
                np.array([10.0, 10.0, 11.0]),
                nassets,
            ),
        })

        cal = pd.DatetimeIndex([
            pd.Timestamp('2014-01-01'),
            pd.Timestamp('2014-01-02'),
            pd.Timestamp('2014-01-03'),
            # omitting the 4th and 5th to simulate a weekend
            pd.Timestamp('2014-01-06'),
        ])
        with tmp_asset_finder(equities=asset_info) as finder:
            expected_output = pd.DataFrame(
                list(concatv([10] * nassets, [11] * nassets)),
                index=pd.MultiIndex.from_product((
                    sorted(expected_views.keys()),
                    finder.retrieve_all(asset_info.index),
                )),
                columns=('value',),
            )
            self._run_pipeline(
                expr,
                deltas,
                expected_views,
                expected_output,
                finder,
                calendar=cal,
                start=cal[2],
                end=cal[-1],
                window_length=3,
                compute_fn=op.itemgetter(-1),
            )
