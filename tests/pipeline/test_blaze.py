"""
Tests for the blaze interface to the pipeline api.
"""
from __future__ import division

from collections import OrderedDict
from datetime import timedelta
from itertools import chain
from unittest import TestCase
import warnings

import blaze as bz
from datashape import dshape, var, Record
import numpy as np
import pandas as pd
from pandas.util.testing import assert_frame_equal
from toolz import keymap
from toolz.curried import operator as op

from zipline.pipeline import Pipeline, CustomFactor
from zipline.pipeline.data import DataSet, BoundColumn
from zipline.pipeline.engine import SimplePipelineEngine
from zipline.pipeline.loaders.blaze import (
    from_blaze,
    BlazeLoader,
    NoDeltasWarning,
    NonNumpyField,
    NonPipelineField,
)
from zipline.utils.test_utils import tmp_asset_finder


nameof = op.attrgetter('name')
dtypeof = op.attrgetter('dtype')


class BlazeToPipelineTestCase(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dates = pd.date_range('2014-01-01', '2014-01-03')
        dates = (
            [pd.Timestamp('2014-01-01')] * 3 +
            [pd.Timestamp('2014-01-02')] * 3 +
            [pd.Timestamp('2014-01-03')] * 3
        )
        cls.sids = sids = ord('A'), ord('B'), ord('C')
        cls.df = df = pd.DataFrame({
            'sid': sids * 3,
            'value': tuple(
                chain.from_iterable((a, a + 1, a + 2) for a in range(3)),
            ),
            'asof_date': dates,
            'timestamp': dates,
        })
        cls.dshape = dshape_ = dshape("""
        var * {
            sid: ?int64,
            value: ?float64,
            asof_date: datetime,
            timestamp: datetime
        }
        """)
        cls.macro_df = df[df.sid == 65].drop('sid', axis=1)
        dshape_ = OrderedDict(dshape_.measure.fields)
        del dshape_['sid']
        cls.macro_dshape = var * Record(dshape_)

        cls.garbage_loader = BlazeLoader()

    def test_tabular(self):
        name = 'expr'
        expr = bz.Data(self.df, name=name, dshape=self.dshape)
        ds = from_blaze(
            expr,
            loader=self.garbage_loader,
            no_deltas_rule='ignore',
        )
        self.assertEqual(ds.__name__, name)
        self.assertTrue(issubclass(ds, DataSet))
        self.assertEqual(
            {c.name: c.dtype for c in ds._columns},
            {'sid': np.int64, 'value': np.float64},
        )

        for field in ('timestamp', 'asof_date'):
            with self.assertRaises(AttributeError) as e:
                getattr(ds, field)
            self.assertIn("'%s'" % field, str(e.exception))
            self.assertIn("'datetime'", str(e.exception))

        self.assertIs(
            from_blaze(
                expr,
                loader=self.garbage_loader,
                no_deltas_rule='ignore',
            ),
            ds,
        )

    def test_column(self):
        exprname = 'expr'
        expr = bz.Data(self.df, name=exprname, dshape=self.dshape)
        value = from_blaze(
            expr.value,
            loader=self.garbage_loader,
            no_deltas_rule='ignore',
        )
        self.assertEqual(value.name, 'value')
        self.assertIsInstance(value, BoundColumn)
        self.assertEqual(value.dtype, np.float64)

        # test memoization
        self.assertIs(
            from_blaze(
                expr.value,
                loader=self.garbage_loader,
                no_deltas_rule='ignore',
            ),
            value,
        )
        self.assertIs(
            from_blaze(
                expr,
                loader=self.garbage_loader,
                no_deltas_rule='ignore',
            ).value,
            value,
        )

        # test the walk back up the tree
        self.assertIs(
            from_blaze(
                expr,
                loader=self.garbage_loader,
                no_deltas_rule='ignore',
            ),
            value.dataset,
        )
        self.assertEqual(value.dataset.__name__, exprname)

    def test_missing_asof(self):
        expr = bz.Data(
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
                no_deltas_rule='ignore',
            )
        self.assertIn("'asof_date'", str(e.exception))
        self.assertIn(repr(str(expr.dshape.measure)), str(e.exception))

    def test_auto_deltas(self):
        expr = bz.Data(
            {'ds': self.df,
             'ds_deltas': pd.DataFrame(columns=self.df.columns)},
            dshape=var * Record((
                ('ds', self.dshape.measure),
                ('ds_deltas', self.dshape.measure),
            )),
        )
        loader = BlazeLoader()
        ds = from_blaze(expr.ds, loader=loader)
        self.assertEqual(len(loader), 1)
        exprdata = loader[ds]
        self.assertTrue(exprdata.expr.isidentical(expr.ds))
        self.assertTrue(exprdata.deltas.isidentical(expr.ds_deltas))

    def test_auto_deltas_fail_warn(self):
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter('always')
            loader = BlazeLoader()
            expr = bz.Data(self.df, dshape=self.dshape)
            from_blaze(
                expr,
                loader=loader,
                no_deltas_rule='warn',
            )
        self.assertEqual(len(ws), 1)
        w = ws[0].message
        self.assertIsInstance(w, NoDeltasWarning)
        self.assertIn(str(expr), str(w))

    def test_auto_deltas_fail_raise(self):
        loader = BlazeLoader()
        expr = bz.Data(self.df, dshape=self.dshape)
        with self.assertRaises(ValueError) as e:
            from_blaze(
                expr,
                loader=loader,
                no_deltas_rule='raise',
            )
        self.assertIn(str(expr), str(e.exception))

    def test_non_numpy_field(self):
        expr = bz.Data(
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
            no_deltas_rule='ignore',
        )
        with self.assertRaises(AttributeError):
            ds.a
        self.assertIsInstance(object.__getattribute__(ds, 'a'), NonNumpyField)

    def test_non_pipeline_field(self):
        # NOTE: This test will fail if we ever allow string types in
        # the Pipeline API. If this happens, change the dtype of the `a` field
        # of expr to another type we don't allow.
        expr = bz.Data(
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
            no_deltas_rule='ignore',
        )
        with self.assertRaises(AttributeError):
            ds.a
        self.assertIsInstance(
            object.__getattribute__(ds, 'a'),
            NonPipelineField,
        )

    def test_complex_expr(self):
        expr = bz.Data(self.df, dshape=self.dshape)
        # put an Add in the table
        expr_with_add = bz.transform(expr, value=expr.value + 1)

        # Test that we can have complex expressions with no deltas
        from_blaze(
            expr_with_add,
            deltas=None,
            loader=self.garbage_loader,
        )

        from_blaze(
            expr.value + 1,  # put an Add in the column
            deltas=None,
            loader=self.garbage_loader,
        )

        deltas = bz.Data(
            pd.DataFrame(columns=self.df.columns),
            dshape=self.dshape,
        )
        with self.assertRaises(TypeError):
            from_blaze(
                expr_with_add,
                deltas=deltas,
                loader=self.garbage_loader,
            )

        with self.assertRaises(TypeError):
            from_blaze(
                expr.value + 1,
                deltas=deltas,
                loader=self.garbage_loader,
            )

    def test_id(self):
        expr = bz.Data(self.df, name='expr', dshape=self.dshape)
        loader = BlazeLoader()
        ds = from_blaze(
            expr,
            loader=loader,
            no_deltas_rule='ignore',
        )
        p = Pipeline()
        p.add(ds.value.latest, 'value')
        dates = self.dates

        with tmp_asset_finder() as finder:
            result = SimplePipelineEngine(
                loader,
                dates,
                finder,
            ).run_pipeline(p, dates[0], dates[-1])

        expected = self.df.drop('asof_date', axis=1).set_index(
            ['timestamp', 'sid'],
        )
        expected.index = pd.MultiIndex.from_product((
            expected.index.levels[0],
            expected.index.levels[1].map(finder.retrieve_asset),
        ))
        assert_frame_equal(result, expected, check_dtype=False)

    def test_id_macro_dataset(self):
        expr = bz.Data(self.macro_df, name='expr', dshape=self.macro_dshape)
        loader = BlazeLoader()
        ds = from_blaze(
            expr,
            loader=loader,
            no_deltas_rule='ignore',
        )
        p = Pipeline()
        p.add(ds.value.latest, 'value')
        dates = self.dates

        with tmp_asset_finder() as finder:
            result = SimplePipelineEngine(
                loader,
                dates,
                finder,
            ).run_pipeline(p, dates[0], dates[-1])

        expected = pd.DataFrame(
            [0, 0, 0, 1, 1, 1, 2, 2, 2],
            index=pd.MultiIndex.from_product((
                self.macro_df.timestamp,
                tuple(map(finder.retrieve_asset, self.sids)),
            )),
            columns=('value',),
        )
        assert_frame_equal(result, expected, check_dtype=False)

    def test_deltas(self):
        expr = bz.Data(self.df, name='expr', dshape=self.dshape)
        deltas = bz.Data(self.df.iloc[:-3], name='deltas', dshape=self.dshape)
        deltas = bz.transform(
            deltas,
            value=deltas.value + 10,
            timestamp=deltas.timestamp + timedelta(days=1),
        )
        loader = BlazeLoader()
        ds = from_blaze(
            expr,
            deltas,
            loader=loader,
            no_deltas_rule='raise',
        )
        p = Pipeline()

        expected_views = keymap(pd.Timestamp, {
            '2014-01-02': np.array([[10.0, 11.0, 12.0],
                                    [1.0, 2.0, 3.0]]),
            '2014-01-03': np.array([[11.0, 12.0, 13.0],
                                    [2.0, 3.0, 4.0]]),
        })
        assertTrue = self.assertTrue

        class TestFactor(CustomFactor):
            inputs = ds.value,
            window_length = 2

            def compute(self, today, assets, out, data):
                assertTrue((data == expected_views[today]).all())
                out[:] = np.max(data)

        p.add(TestFactor(), 'value')
        dates = self.dates

        with tmp_asset_finder() as finder:
            result = SimplePipelineEngine(
                loader,
                dates,
                finder,
            ).run_pipeline(p, dates[1], dates[-1])

        assert_frame_equal(
            result,
            pd.DataFrame(
                [12, 12, 12, 13, 13, 13],
                index=pd.MultiIndex.from_product((
                    sorted(expected_views.keys()),
                    tuple(map(finder.retrieve_asset, self.sids)),
                )),
                columns=('value',),
            ),
            check_dtype=False,
        )

    def test_deltas_macro_dataset(self):
        expr = bz.Data(self.macro_df, name='expr', dshape=self.macro_dshape)
        deltas = bz.Data(
            self.macro_df.iloc[:-1],
            name='deltas',
            dshape=self.macro_dshape,
        )
        deltas = bz.transform(
            deltas,
            value=deltas.value + 10,
            timestamp=deltas.timestamp + timedelta(days=1),
        )
        loader = BlazeLoader()
        ds = from_blaze(
            expr,
            deltas,
            loader=loader,
            no_deltas_rule='raise',
        )
        p = Pipeline()

        expected_views = keymap(pd.Timestamp, {
            '2014-01-02': np.array([[10.0, 10.0, 10.0],
                                    [1.0, 1.0, 1.0]]),
            '2014-01-03': np.array([[11.0, 11.0, 11.0],
                                    [2.0, 2.0, 2.0]]),
        })
        assertTrue = self.assertTrue

        class TestFactor(CustomFactor):
            inputs = ds.value,
            window_length = 2

            def compute(self, today, assets, out, data):
                assertTrue((data == expected_views[today]).all())
                out[:] = np.max(data)

        p.add(TestFactor(), 'value')
        dates = self.dates

        with tmp_asset_finder() as finder:
            result = SimplePipelineEngine(
                loader,
                dates,
                finder,
            ).run_pipeline(p, dates[1], dates[-1])

        assert_frame_equal(
            result,
            pd.DataFrame(
                [10, 10, 10, 11, 11, 11],
                index=pd.MultiIndex.from_product((
                    sorted(expected_views.keys()),
                    tuple(map(finder.retrieve_asset, self.sids)),
                )),
                columns=('value',),
            ),
            check_dtype=False,
        )
