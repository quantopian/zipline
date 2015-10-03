"""
Tests for the blaze interface to the pipeline api.
"""
from __future__ import division

from itertools import chain
from unittest import TestCase

import blaze as bz
from datashape import dshape
import numpy as np
import pandas as pd
from pandas.util.testing import assert_frame_equal
from toolz import flip
from toolz.curried import operator as op

from zipline.pipeline import Pipeline
from zipline.pipeline.data import DataSet, BoundColumn
from zipline.pipeline.engine import SimplePipelineEngine
from zipline.pipeline.loaders.blaze import (
    pipeline_api_from_blaze,
    BlazeLoader,
    NonNumpyField,
    NonPipelineField,
    NotPipelineCompatible,
)
from zipline.utils.test_utils import (
    make_simple_asset_info,
    tmp_asset_finder,
)


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
        cls.df = pd.DataFrame({
            'sid': [ord('A'), ord('B'), ord('C')] * 3,
            'value': tuple(
                chain.from_iterable((a, a + 1, a + 2) for a in range(3)),
            ),
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
        cls.garbage_loader = BlazeLoader()

    def test_tabular(self):
        name = 'expr'
        expr = bz.Data(self.df, name=name, dshape=self.dshape)
        ds = pipeline_api_from_blaze(expr, loader=self.garbage_loader)
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
            pipeline_api_from_blaze(expr, loader=self.garbage_loader),
            ds,
        )

    def test_column(self):
        exprname = 'expr'
        expr = bz.Data(self.df, name=exprname, dshape=self.dshape)
        value = pipeline_api_from_blaze(expr.value, loader=self.garbage_loader)
        self.assertEqual(value.name, 'value')
        self.assertIsInstance(value, BoundColumn)
        self.assertEqual(value.dtype, np.float64)

        # test memoization
        self.assertIs(
            pipeline_api_from_blaze(expr.value, loader=self.garbage_loader),
            value,
        )
        self.assertIs(
            pipeline_api_from_blaze(expr, loader=self.garbage_loader).value,
            value,
        )

        # test the walk back up the tree
        self.assertIs(
            pipeline_api_from_blaze(expr, loader=self.garbage_loader),
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
            pipeline_api_from_blaze(expr, loader=self.garbage_loader)
        self.assertIn("'asof_date'", str(e.exception))
        self.assertIn(repr(str(expr.dshape.measure)), str(e.exception))

    def test_id(self):
        name = 'expr'
        expr = bz.Data(self.df, name=name, dshape=self.dshape)
        loader = BlazeLoader()
        ds = pipeline_api_from_blaze(expr, loader=loader)
        p = Pipeline('p')
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
