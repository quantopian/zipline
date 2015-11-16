"""
Tests for the blaze interface to the pipeline api.
"""
from __future__ import division

from collections import OrderedDict
from datetime import timedelta
from unittest import TestCase
import warnings

import blaze as bz
from datashape import dshape, var, Record
from nose_parameterized import parameterized
import numpy as np
from numpy.testing.utils import assert_array_almost_equal
import pandas as pd
from pandas.util.testing import assert_frame_equal
from toolz import keymap, valmap, concatv
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
from zipline.utils.numpy_utils import repeat_last_axis
from zipline.utils.test_utils import tmp_asset_finder, make_simple_equity_info


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


class BlazeToPipelineTestCase(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dates = dates = pd.date_range('2014-01-01', '2014-01-03')
        dates = cls.dates.repeat(3)
        cls.sids = sids = ord('A'), ord('B'), ord('C')
        cls.df = df = pd.DataFrame({
            'sid': sids * 3,
            'value': (0, 1, 2, 1, 2, 3, 2, 3, 4),
            'asof_date': dates,
            'timestamp': dates,
        })
        cls.dshape = dshape("""
        var * {
            sid: ?int64,
            value: ?float64,
            asof_date: datetime,
            timestamp: datetime
        }
        """)
        cls.macro_df = df[df.sid == 65].drop('sid', axis=1)
        dshape_ = OrderedDict(cls.dshape.measure.fields)
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

        # test memoization
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

        with self.assertRaises(TypeError):
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
            finder.retrieve_all(expected.index.levels[1]),
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

        asset_info = asset_infos[0][0]
        with tmp_asset_finder(equities=asset_info) as finder:
            result = SimplePipelineEngine(
                loader,
                dates,
                finder,
            ).run_pipeline(p, dates[0], dates[-1])

        nassets = len(asset_info)
        expected = pd.DataFrame(
            list(concatv([0] * nassets, [1] * nassets, [2] * nassets)),
            index=pd.MultiIndex.from_product((
                self.macro_df.timestamp,
                finder.retrieve_all(asset_info.index),
            )),
            columns=('value',),
        )
        assert_frame_equal(result, expected, check_dtype=False)

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
            no_deltas_rule='raise',
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
            expected_output,
            check_dtype=False,
        )

    @with_extra_sid
    def test_deltas(self, asset_info):
        expr = bz.Data(self.df, name='expr', dshape=self.dshape)
        deltas = bz.Data(self.df, name='deltas', dshape=self.dshape)
        deltas = bz.transform(
            deltas,
            value=deltas.value + 10,
            timestamp=deltas.timestamp + timedelta(days=1),
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

    def test_deltas_macro(self):
        asset_info = asset_infos[0][0]
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
            'value': (0, 1, 2, 1, 2, 3),
            'asof_date': repeated_dates,
            'timestamp': repeated_dates,
        })
        expr = bz.Data(baseline, name='expr', dshape=self.dshape)
        deltas = bz.Data(baseline, name='deltas', dshape=self.dshape)
        deltas = bz.transform(
            deltas,
            value=deltas.value + 10,
            timestamp=deltas.timestamp + timedelta(days=1),
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
        expr = bz.Data(baseline, name='expr', dshape=self.macro_dshape)
        deltas = bz.Data(baseline, name='deltas', dshape=self.macro_dshape)
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
