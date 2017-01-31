"""
Tests for the blaze interface to the pipeline api.
"""
from __future__ import division

from collections import OrderedDict
from datetime import timedelta, time
from itertools import product, chain
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
from zipline.errors import UnsupportedPipelineOutput
from zipline.pipeline import Pipeline, CustomFactor
from zipline.pipeline.data import DataSet, BoundColumn, Column
from zipline.pipeline.engine import SimplePipelineEngine
from zipline.pipeline.loaders.blaze import (
    from_blaze,
    BlazeLoader,
    NoMetaDataWarning,
)
from zipline.pipeline.loaders.blaze.core import (
    ExprData,
    NonPipelineField,
)
from zipline.testing import (
    ZiplineTestCase,
    parameter_space,
    tmp_asset_finder,
)
from zipline.testing.fixtures import WithAssetFinder
from zipline.testing.predicates import assert_equal, assert_isidentical
from zipline.utils.numpy_utils import float64_dtype, int64_dtype


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
simple_asset_info = asset_infos[0][0]
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


class BlazeToPipelineTestCase(WithAssetFinder, ZiplineTestCase):
    START_DATE = pd.Timestamp(0)
    END_DATE = pd.Timestamp('2015')

    @classmethod
    def init_class_fixtures(cls):
        super(BlazeToPipelineTestCase, cls).init_class_fixtures()
        cls.dates = dates = pd.date_range('2014-01-01', '2014-01-03')
        dates = cls.dates.repeat(3)
        cls.df = df = pd.DataFrame({
            'sid': cls.ASSET_FINDER_EQUITY_SIDS * 3,
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

        cls.value_dshape = dshape("""var * {
            sid: ?int64,
            value: float64,
            asof_date: datetime,
            timestamp: datetime,
        }""")

    def test_tabular(self):
        name = 'expr'
        expr = bz.data(self.df, name=name, dshape=self.dshape)
        ds = from_blaze(
            expr,
            loader=self.garbage_loader,
            no_deltas_rule='ignore',
            no_checkpoints_rule='ignore',
            missing_values=self.missing_values,
        )
        self.assertEqual(ds.__name__, name)
        self.assertTrue(issubclass(ds, DataSet))

        self.assertIs(ds.value.dtype, float64_dtype)
        self.assertIs(ds.int_value.dtype, int64_dtype)

        self.assertTrue(np.isnan(ds.value.missing_value))
        self.assertEqual(ds.int_value.missing_value, 0)

        # test memoization
        self.assertIs(
            from_blaze(
                expr,
                loader=self.garbage_loader,
                no_deltas_rule='ignore',
                no_checkpoints_rule='ignore',
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
            no_deltas_rule='ignore',
            no_checkpoints_rule='ignore',
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
                no_deltas_rule='ignore',
                no_checkpoints_rule='ignore',
                missing_values=self.missing_values,
            ),
            value,
        )
        self.assertIs(
            from_blaze(
                expr,
                loader=self.garbage_loader,
                no_deltas_rule='ignore',
                no_checkpoints_rule='ignore',
                missing_values=self.missing_values,
            ).value,
            value,
        )

        # test the walk back up the tree
        self.assertIs(
            from_blaze(
                expr,
                loader=self.garbage_loader,
                no_deltas_rule='ignore',
                no_checkpoints_rule='ignore',
                missing_values=self.missing_values,
            ),
            value.dataset,
        )
        self.assertEqual(value.dataset.__name__, exprname)

    def test_missing_asof(self):
        expr = bz.data(
            self.df.loc[:, ['sid', 'value', 'timestamp']],
            name='expr',
            dshape="""var * {
                sid: int64,
                value: float64,
                timestamp: datetime,
            }""",
        )

        with self.assertRaises(TypeError) as e:
            from_blaze(
                expr,
                loader=self.garbage_loader,
                no_deltas_rule='ignore',
                no_checkpoints_rule='ignore',
            )
        self.assertIn("'asof_date'", str(e.exception))
        self.assertIn(repr(str(expr.dshape.measure)), str(e.exception))

    def test_missing_timestamp(self):
        expr = bz.data(
            self.df.loc[:, ['sid', 'value', 'asof_date']],
            name='expr',
            dshape="""var * {
                sid: int64,
                value: float64,
                asof_date: datetime,
            }""",
        )

        loader = BlazeLoader()

        from_blaze(
            expr,
            loader=loader,
            no_deltas_rule='ignore',
            no_checkpoints_rule='ignore',
        )

        self.assertEqual(len(loader), 1)
        exprdata, = loader.values()

        assert_isidentical(
            exprdata.expr,
            bz.transform(expr, timestamp=expr.asof_date),
        )

    def test_from_blaze_no_resources_dataset_expr(self):
        expr = bz.symbol('expr', self.dshape)

        with self.assertRaises(ValueError) as e:
            from_blaze(
                expr,
                loader=self.garbage_loader,
                no_deltas_rule='ignore',
                no_checkpoints_rule='ignore',
                missing_values=self.missing_values,
            )
        assert_equal(
            str(e.exception),
            'no resources provided to compute expr',
        )

    @parameter_space(metadata={'deltas', 'checkpoints'})
    def test_from_blaze_no_resources_metadata_expr(self, metadata):
        expr = bz.data(self.df, name='expr', dshape=self.dshape)
        metadata_expr = bz.symbol('metadata', self.dshape)

        with self.assertRaises(ValueError) as e:
            from_blaze(
                expr,
                loader=self.garbage_loader,
                no_deltas_rule='ignore',
                no_checkpoints_rule='ignore',
                missing_values=self.missing_values,
                **{metadata: metadata_expr}
            )
        assert_equal(
            str(e.exception),
            'no resources provided to compute %s' % metadata,
        )

    def test_from_blaze_mixed_resources_dataset_expr(self):
        expr = bz.data(self.df, name='expr', dshape=self.dshape)

        with self.assertRaises(ValueError) as e:
            from_blaze(
                expr,
                resources={expr: self.df},
                loader=self.garbage_loader,
                no_deltas_rule='ignore',
                no_checkpoints_rule='ignore',
                missing_values=self.missing_values,
            )
        assert_equal(
            str(e.exception),
            'explicit and implicit resources provided to compute expr',
        )

    @parameter_space(metadata={'deltas', 'checkpoints'})
    def test_from_blaze_mixed_resources_metadata_expr(self, metadata):
        expr = bz.symbol('expr', self.dshape)
        metadata_expr = bz.data(self.df, name=metadata, dshape=self.dshape)

        with self.assertRaises(ValueError) as e:
            from_blaze(
                expr,
                resources={metadata_expr: self.df},
                loader=self.garbage_loader,
                no_deltas_rule='ignore',
                no_checkpoints_rule='ignore',
                missing_values=self.missing_values,
                **{metadata: metadata_expr}
            )
        assert_equal(
            str(e.exception),
            'explicit and implicit resources provided to compute %s' %
            metadata,
        )

    @parameter_space(deltas={True, False}, checkpoints={True, False})
    def test_auto_metadata(self, deltas, checkpoints):
        select_level = op.getitem(('ignore', 'raise'))
        m = {'ds': self.df}
        if deltas:
            m['ds_deltas'] = pd.DataFrame(columns=self.df.columns),
        if checkpoints:
            m['ds_checkpoints'] = pd.DataFrame(columns=self.df.columns),
        expr = bz.data(
            m,
            dshape=var * Record((k, self.dshape.measure) for k in m),
        )
        loader = BlazeLoader()
        ds = from_blaze(
            expr.ds,
            loader=loader,
            missing_values=self.missing_values,
            no_deltas_rule=select_level(deltas),
            no_checkpoints_rule=select_level(checkpoints),
        )
        self.assertEqual(len(loader), 1)
        exprdata = loader[ds]
        self.assertTrue(exprdata.expr.isidentical(expr.ds))
        if deltas:
            self.assertTrue(exprdata.deltas.isidentical(expr.ds_deltas))
        else:
            self.assertIsNone(exprdata.deltas)
        if checkpoints:
            self.assertTrue(
                exprdata.checkpoints.isidentical(expr.ds_checkpoints),
            )
        else:
            self.assertIsNone(exprdata.checkpoints)

    @parameter_space(deltas={True, False}, checkpoints={True, False})
    def test_auto_metadata_fail_warn(self, deltas, checkpoints):
        select_level = op.getitem(('ignore', 'warn'))
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter('always')
            loader = BlazeLoader()
            expr = bz.data(self.df, dshape=self.dshape)
            from_blaze(
                expr,
                loader=loader,
                no_deltas_rule=select_level(deltas),
                no_checkpoints_rule=select_level(checkpoints),
                missing_values=self.missing_values,
            )
            self.assertEqual(len(ws), deltas + checkpoints)

        for w in ws:
            w = w.message
            self.assertIsInstance(w, NoMetaDataWarning)
            self.assertIn(str(expr), str(w))

    @parameter_space(deltas={True, False}, checkpoints={True, False})
    def test_auto_metadata_fail_raise(self, deltas, checkpoints):
        if not (deltas or checkpoints):
            # not a real case
            return
        select_level = op.getitem(('ignore', 'raise'))
        loader = BlazeLoader()
        expr = bz.data(self.df, dshape=self.dshape)
        with self.assertRaises(ValueError) as e:
            from_blaze(
                expr,
                loader=loader,
                no_deltas_rule=select_level(deltas),
                no_checkpoints_rule=select_level(checkpoints),
            )
        self.assertIn(str(expr), str(e.exception))

    def test_non_pipeline_field(self):
        expr = bz.data(
            [],
            dshape="""
            var * {
                 a: complex,
                 asof_date: datetime,
                 timestamp: datetime,
            }""",
        )
        ds = from_blaze(
            expr,
            loader=self.garbage_loader,
            no_deltas_rule='ignore',
            no_checkpoints_rule='ignore',
        )
        with self.assertRaises(AttributeError):
            ds.a
        self.assertIsInstance(
            object.__getattribute__(ds, 'a'),
            NonPipelineField,
        )

    def test_cols_with_all_missing_vals(self):
        """
        Tests that when there is no known data, we get output where the
        columns have the right dtypes and the right missing values filled in.

        input (self.df):
        Empty DataFrame
        Columns: [sid, float_value, str_value, int_value, bool_value, dt_value,
            asof_date, timestamp]
        Index: []

        output (expected)
                                          str_value  float_value  int_value
        2014-01-01 Equity(65 [A])      None          NaN          0
                   Equity(66 [B])      None          NaN          0
                   Equity(67 [C])      None          NaN          0
        2014-01-02 Equity(65 [A])      None          NaN          0
                   Equity(66 [B])      None          NaN          0
                   Equity(67 [C])      None          NaN          0
        2014-01-03 Equity(65 [A])      None          NaN          0
                   Equity(66 [B])      None          NaN          0
                   Equity(67 [C])      None          NaN          0

                                  dt_value  bool_value
        2014-01-01 Equity(65 [A])      NaT  False
                   Equity(66 [B])      NaT  False
                   Equity(67 [C])      NaT  False
        2014-01-02 Equity(65 [A])      NaT  False
                   Equity(66 [B])      NaT  False
                   Equity(67 [C])      NaT  False
        2014-01-03 Equity(65 [A])      NaT  False
                   Equity(66 [B])      NaT  False
                   Equity(67 [C])      NaT  False
        """
        df = pd.DataFrame(columns=['sid', 'float_value', 'str_value',
                                   'int_value', 'bool_value', 'dt_value',
                                   'asof_date', 'timestamp'])

        expr = bz.data(
            df,
            dshape="""
            var * {
                 sid: int64,
                 float_value: float64,
                 str_value: string,
                 int_value: int64,
                 bool_value: bool,
                 dt_value: datetime,
                 asof_date: datetime,
                 timestamp: datetime,
            }""",
        )
        fields = OrderedDict(expr.dshape.measure.fields)

        expected = pd.DataFrame({
            "str_value": np.array([None,
                                   None,
                                   None,
                                   None,
                                   None,
                                   None,
                                   None,
                                   None,
                                   None],
                                  dtype='object'),
            "float_value": np.array([np.NaN,
                                     np.NaN,
                                     np.NaN,
                                     np.NaN,
                                     np.NaN,
                                     np.NaN,
                                     np.NaN,
                                     np.NaN,
                                     np.NaN],
                                    dtype='float64'),
            "int_value": np.array([0,
                                   0,
                                   0,
                                   0,
                                   0,
                                   0,
                                   0,
                                   0,
                                   0],
                                  dtype='int64'),
            "bool_value": np.array([False,
                                    False,
                                    False,
                                    False,
                                    False,
                                    False,
                                    False,
                                    False,
                                    False],
                                   dtype='bool'),
            "dt_value": [pd.NaT,
                         pd.NaT,
                         pd.NaT,
                         pd.NaT,
                         pd.NaT,
                         pd.NaT,
                         pd.NaT,
                         pd.NaT,
                         pd.NaT],
        },
            columns=['str_value', 'float_value', 'int_value', 'bool_value',
                     'dt_value'],
            index=pd.MultiIndex.from_product(
                (self.dates, self.asset_finder.retrieve_all(
                    self.ASSET_FINDER_EQUITY_SIDS
                ))
            )
        )

        self._test_id(
            df,
            var * Record(fields),
            expected,
            self.asset_finder,
            ('float_value', 'str_value', 'int_value', 'bool_value',
             'dt_value'),
        )

    def test_cols_with_some_missing_vals(self):
        """
        Tests the following:
            1) Forward filling replaces missing values correctly for the data
            types supported in pipeline.
            2) We don't forward fill when the missing value is the actual value
             we got for a date in the case of int/bool columns.
            3) We get the correct type of missing value in the output.

        input (self.df):
           asof_date bool_value   dt_value  float_value  int_value  sid
        0 2014-01-01       True 2011-01-01            0          1   65
        1 2014-01-03       True 2011-01-02            1          2   66
        2 2014-01-01       True 2011-01-03            2          3   67
        3 2014-01-02      False        NaT          NaN          0   67

          str_value  timestamp
        0         a  2014-01-01
        1         b  2014-01-03
        2         c  2014-01-01
        3      None  2014-01-02

        output (expected)
                                  str_value  float_value  int_value bool_value
        2014-01-01 Equity(65 [A])         a            0          1       True
                   Equity(66 [B])      None          NaN          0      False
                   Equity(67 [C])         c            2          3       True
        2014-01-02 Equity(65 [A])         a            0          1       True
                   Equity(66 [B])      None          NaN          0      False
                   Equity(67 [C])         c            2          0      False
        2014-01-03 Equity(65 [A])         a            0          1       True
                   Equity(66 [B])         b            1          2       True
                   Equity(67 [C])         c            2          0      False

                                    dt_value
        2014-01-01 Equity(65 [A]) 2011-01-01
                   Equity(66 [B])        NaT
                   Equity(67 [C]) 2011-01-03
        2014-01-02 Equity(65 [A]) 2011-01-01
                   Equity(66 [B])        NaT
                   Equity(67 [C]) 2011-01-03
        2014-01-03 Equity(65 [A]) 2011-01-01
                   Equity(66 [B]) 2011-01-02
                   Equity(67 [C]) 2011-01-03
        """
        dates = (self.dates[0], self.dates[-1], self.dates[0], self.dates[1])
        df = pd.DataFrame({
            'sid': self.ASSET_FINDER_EQUITY_SIDS[:-1] +
            (self.ASSET_FINDER_EQUITY_SIDS[-1],)*2,
            'float_value': (0., 1., 2., np.NaN),
            'str_value': ("a", "b", "c", None),
            'int_value': (1, 2, 3, 0),
            'bool_value': (True, True, True, False),
            'dt_value': (pd.Timestamp('2011-01-01'),
                         pd.Timestamp('2011-01-02'),
                         pd.Timestamp('2011-01-03'),
                         pd.NaT),
            'asof_date': dates,
            'timestamp': dates,
        })

        expr = bz.data(
            df,
            dshape="""
            var * {
                 sid: int64,
                 float_value: float64,
                 str_value: string,
                 int_value: int64,
                 bool_value: bool,
                 dt_value: datetime,
                 asof_date: datetime,
                 timestamp: datetime,
            }""",
        )
        fields = OrderedDict(expr.dshape.measure.fields)

        expected = pd.DataFrame({
            "str_value": np.array(["a",
                                   None,
                                   "c",
                                   "a",
                                   None,
                                   "c",
                                   "a",
                                   "b",
                                   "c"],
                                  dtype='object'),
            "float_value": np.array([0,
                                     np.NaN,
                                     2,
                                     0,
                                     np.NaN,
                                     2,
                                     0,
                                     1,
                                     2],
                                    dtype='float64'),
            "int_value": np.array([1,
                                   0,
                                   3,
                                   1,
                                   0,
                                   0,
                                   1,
                                   2,
                                   0],
                                  dtype='int64'),
            "bool_value": np.array([True,
                                    False,
                                    True,
                                    True,
                                    False,
                                    False,
                                    True,
                                    True,
                                    False],
                                   dtype='bool'),
            "dt_value": [pd.Timestamp('2011-01-01'),
                         pd.NaT,
                         pd.Timestamp('2011-01-03'),
                         pd.Timestamp('2011-01-01'),
                         pd.NaT,
                         pd.Timestamp('2011-01-03'),
                         pd.Timestamp('2011-01-01'),
                         pd.Timestamp('2011-01-02'),
                         pd.Timestamp('2011-01-03')],
        },
            columns=['str_value', 'float_value', 'int_value', 'bool_value',
                     'dt_value'],
            index=pd.MultiIndex.from_product(
                (self.dates, self.asset_finder.retrieve_all(
                    self.ASSET_FINDER_EQUITY_SIDS
                ))
            )
        )

        self._test_id(
            df,
            var * Record(fields),
            expected,
            self.asset_finder,
            ('float_value', 'str_value', 'int_value', 'bool_value',
             'dt_value'),
        )

    def test_complex_expr(self):
        expr = bz.data(self.df, dshape=self.dshape, name='expr')
        # put an Add in the table
        expr_with_add = bz.transform(expr, value=expr.value + 1)

        # test that we can have complex expressions with no metadata
        from_blaze(
            expr_with_add,
            deltas=None,
            checkpoints=None,
            loader=self.garbage_loader,
            missing_values=self.missing_values,
            no_checkpoints_rule='ignore',
        )

        with self.assertRaises(TypeError) as e:
            # test that we cannot create a single column from a non field
            from_blaze(
                expr.value + 1,  # put an Add in the column
                deltas=None,
                checkpoints=None,
                loader=self.garbage_loader,
                missing_values=self.missing_values,
                no_checkpoints_rule='ignore',
            )
        assert_equal(
            str(e.exception),
            "expression 'expr.value + 1' was array-like but not a simple field"
            " of some larger table",
        )

        deltas = bz.data(
            pd.DataFrame(columns=self.df.columns),
            dshape=self.dshape,
            name='deltas',
        )
        checkpoints = bz.data(
            pd.DataFrame(columns=self.df.columns),
            dshape=self.dshape,
            name='checkpoints',
        )

        # test that we can have complex expressions with explicit metadata
        from_blaze(
            expr_with_add,
            deltas=deltas,
            checkpoints=checkpoints,
            loader=self.garbage_loader,
            missing_values=self.missing_values,
        )

        with self.assertRaises(TypeError) as e:
            # test that we cannot create a single column from a non field
            # even with explicit metadata
            from_blaze(
                expr.value + 1,
                deltas=deltas,
                checkpoints=checkpoints,
                loader=self.garbage_loader,
                missing_values=self.missing_values,
            )
        assert_equal(
            str(e.exception),
            "expression 'expr.value + 1' was array-like but not a simple field"
            " of some larger table",
        )

    def _test_id(self, df, dshape, expected, finder, add):
        expr = bz.data(df, name='expr', dshape=dshape)
        loader = BlazeLoader()
        ds = from_blaze(
            expr,
            loader=loader,
            no_deltas_rule='ignore',
            no_checkpoints_rule='ignore',
            missing_values=self.missing_values,
        )
        p = Pipeline()
        for a in add:
            p.add(getattr(ds, a).latest, a)
        dates = self.dates

        result = SimplePipelineEngine(
            loader,
            dates,
            finder,
        ).run_pipeline(p, dates[0], dates[-1])
        assert_frame_equal(
            result.sort_index(axis=1),
            _utc_localize_index_level_0(expected.sort_index(axis=1)),
            check_dtype=False,
        )

    def _test_id_macro(self, df, dshape, expected, finder, add):
        dates = self.dates
        expr = bz.data(df, name='expr', dshape=dshape)
        loader = BlazeLoader()
        ds = from_blaze(
            expr,
            loader=loader,
            no_deltas_rule='ignore',
            no_checkpoints_rule='ignore',
            missing_values=self.missing_values,
        )

        p = Pipeline()
        macro_inputs = []
        for column_name in add:
            column = getattr(ds, column_name)
            macro_inputs.append(column)
            with self.assertRaises(UnsupportedPipelineOutput):
                # Single column output terms cannot be added to a pipeline.
                p.add(column.latest, column_name)

        class UsesMacroInputs(CustomFactor):
            inputs = macro_inputs
            window_length = 1

            def compute(self, today, assets, out, *inputs):
                e = expected.loc[today]
                for i, input_ in enumerate(inputs):
                    # Each macro input should only have one column.
                    assert input_.shape == (self.window_length, 1)
                    assert_equal(input_[0, 0], e[i])

        # Run the pipeline with our custom factor. Assertions about the
        # expected macro data are made in the `compute` function of our custom
        # factor above.
        p.add(UsesMacroInputs(), 'uses_macro_inputs')
        engine = SimplePipelineEngine(loader, dates, finder)
        engine.run_pipeline(p, dates[0], dates[-1])

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
            no_deltas_rule='ignore',
            no_checkpoints_rule='ignore',
            missing_values=self.missing_values,
        )
        p = Pipeline()
        p.add(ds.value.latest, 'value')
        p.add(ds.int_value.latest, 'int_value')
        dates = self.dates

        result = SimplePipelineEngine(
            loader,
            dates,
            self.asset_finder,
        ).run_pipeline(p, dates[0], dates[-1])

        expected = df.drop('asof_date', axis=1)
        expected['timestamp'] = expected['timestamp'].dt.normalize().astype(
            'datetime64[ns]',
        ).dt.tz_localize('utc')
        expected.ix[3:5, 'timestamp'] += timedelta(days=1)
        expected.set_index(['timestamp', 'sid'], inplace=True)
        expected.index = pd.MultiIndex.from_product((
            expected.index.levels[0],
            self.asset_finder.retrieve_all(expected.index.levels[1]),
        ))
        assert_frame_equal(result, expected, check_dtype=False)

    def test_id(self):
        """
        input (self.df):
           asof_date  sid  timestamp int_value value
        0 2014-01-01   65 2014-01-01         0     0
        1 2014-01-01   66 2014-01-01         1     1
        2 2014-01-01   67 2014-01-01         2     2
        3 2014-01-02   65 2014-01-02         1     1
        4 2014-01-02   66 2014-01-02         2     2
        5 2014-01-02   67 2014-01-02         3     3
        6 2014-01-03   65 2014-01-03         2     2
        7 2014-01-03   66 2014-01-03         3     3
        8 2014-01-03   67 2014-01-03         4     4

        output (expected)
                                  int_value value
        2014-01-01 Equity(65 [A])         0     0
                   Equity(66 [B])         1     1
                   Equity(67 [C])         2     2
        2014-01-02 Equity(65 [A])         1     1
                   Equity(66 [B])         2     2
                   Equity(67 [C])         3     3
        2014-01-03 Equity(65 [A])         2     2
                   Equity(66 [B])         3     3
                   Equity(67 [C])         4     4
        """
        expected = self.df.drop('asof_date', axis=1).set_index(
            ['timestamp', 'sid'],
        )
        expected.index = pd.MultiIndex.from_product((
            expected.index.levels[0],
            self.asset_finder.retrieve_all(expected.index.levels[1]),
        ))
        self._test_id(
            self.df, self.dshape, expected, self.asset_finder,
            ('int_value', 'value',)
        )

    def test_id_with_asof_date(self):
        """
        input (self.df):
           asof_date  sid  timestamp int_value value
        0 2014-01-01   65 2014-01-01         0     0
        1 2014-01-01   66 2014-01-01         1     1
        2 2014-01-01   67 2014-01-01         2     2
        3 2014-01-02   65 2014-01-02         1     1
        4 2014-01-02   66 2014-01-02         2     2
        5 2014-01-02   67 2014-01-02         3     3
        6 2014-01-03   65 2014-01-03         2     2
        7 2014-01-03   66 2014-01-03         3     3
        8 2014-01-03   67 2014-01-03         4     4

        output (expected)
                                    asof_date
        2014-01-01 Equity(65 [A])  2014-01-01
                   Equity(66 [B])  2014-01-01
                   Equity(67 [C])  2014-01-01
        2014-01-02 Equity(65 [A])  2014-01-02
                   Equity(66 [B])  2014-01-02
                   Equity(67 [C])  2014-01-02
        2014-01-03 Equity(65 [A])  2014-01-03
                   Equity(66 [B])  2014-01-03
                   Equity(67 [C])  2014-01-03
        """
        expected = self.df.drop(['value', 'int_value'], axis=1).set_index(
            ['timestamp', 'sid'],
        )
        expected.index = pd.MultiIndex.from_product((
            expected.index.levels[0],
            self.asset_finder.retrieve_all(expected.index.levels[1]),
        ))
        self._test_id(
            self.df, self.dshape, expected, self.asset_finder,
            ('asof_date',)
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
            'sid': self.ASSET_FINDER_EQUITY_SIDS * 3,
            'value': (0, 1, np.nan, 1, np.nan, 3, np.nan, 3, 4),
            'other': (0, np.nan, 2, np.nan, 2, 3, 2, 3, np.nan),
            'asof_date': dates,
            'timestamp': dates,
        })
        fields = OrderedDict(self.dshape.measure.fields)
        fields['other'] = fields['value']

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
                (self.dates, self.asset_finder.retrieve_all(
                    self.ASSET_FINDER_EQUITY_SIDS
                )),
            ),
        )
        self._test_id(
            df,
            var * Record(fields),
            expected,
            self.asset_finder,
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
        expected = df.drop('asof_date', axis=1).set_index(
            ['timestamp', 'sid'],
        ).sort_index(axis=1)
        expected.index = pd.MultiIndex.from_product((
            expected.index.levels[0],
            self.asset_finder.retrieve_all(expected.index.levels[1]),
        ))
        self._test_id(
            df,
            var * Record(fields),
            expected,
            self.asset_finder,
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
        2014-01-01      0
        2014-01-02      1
        2014-01-03      2
        """
        expected = pd.DataFrame(
            data=[[0],
                  [1],
                  [2]],
            columns=['value'],
            index=self.dates,
        )
        self._test_id_macro(
            self.macro_df,
            self.macro_dshape,
            expected,
            self.asset_finder,
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
        2014-01-01      1      0
        2014-01-02      1      0
        2014-01-03      1      0
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

        expected = pd.DataFrame(
            data=[[0, 1],
                  [0, 1],
                  [0, 1]],
            columns=['other', 'value'],
            index=self.dates,
        )
        self._test_id_macro(
            df,
            var * Record(fields),
            expected,
            self.asset_finder,
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
        2014-01-01      1      0
        2014-01-02      2      1
        2014-01-03      3      2
        """
        df = self.macro_df.copy()
        df['other'] = df.value + 1
        fields = OrderedDict(self.macro_dshape.measure.fields)
        fields['other'] = fields['value']

        with tmp_asset_finder(equities=simple_asset_info) as finder:
            expected = pd.DataFrame(
                data=[[0, 1],
                      [1, 2],
                      [2, 3]],
                columns=['value', 'other'],
                index=self.dates,
                dtype=np.float64,
            )
            self._test_id_macro(
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
                (self.dates, self.asset_finder.retrieve_all(
                    self.ASSET_FINDER_EQUITY_SIDS
                )),
            ),
        )
        self._test_id(
            df,
            var * Record(fields),
            expected,
            self.asset_finder,
            ('value', 'other'),
        )

    def test_id_take_last_in_group_macro(self):
        """
        output (expected):

                    other  value
        2014-01-01    NaN      1
        2014-01-02      1      2
        2014-01-03      2      2
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

        expected = pd.DataFrame(
            data=[[np.nan, 1],   # 2014-01-01
                  [1,      2],   # 2014-01-02
                  [2,      2]],  # 2014-01-03
            columns=['other', 'value'],
            index=self.dates,
        )
        self._test_id_macro(
            df,
            var * Record(fields),
            expected,
            self.asset_finder,
            ('other', 'value'),
        )

    def _run_pipeline(self,
                      expr,
                      deltas,
                      checkpoints,
                      expected_views,
                      expected_output,
                      finder,
                      calendar,
                      start,
                      end,
                      window_length,
                      compute_fn,
                      apply_deltas_adjustments=True):
        loader = BlazeLoader()
        ds = from_blaze(
            expr,
            deltas,
            checkpoints,
            apply_deltas_adjustments=apply_deltas_adjustments,
            loader=loader,
            no_deltas_rule='raise',
            no_checkpoints_rule='ignore',
            missing_values=self.missing_values,
        )
        p = Pipeline()

        # prevent unbound locals issue in the inner class
        window_length_ = window_length

        class TestFactor(CustomFactor):
            inputs = ds.value,
            window_length = window_length_

            def compute(self, today, assets, out, data):
                assert_array_almost_equal(
                    data,
                    expected_views[today],
                    err_msg=str(today),
                )
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
                None,
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
                None,
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

        nassets = len(simple_asset_info)
        expected_views = keymap(pd.Timestamp, {
            '2014-01-02': np.array([[10.0],
                                    [1.0]]),
            '2014-01-03': np.array([[11.0],
                                    [2.0]]),
        })

        with tmp_asset_finder(equities=simple_asset_info) as finder:
            expected_output = pd.DataFrame(
                list(concatv([10] * nassets, [11] * nassets)),
                index=pd.MultiIndex.from_product((
                    sorted(expected_views.keys()),
                    finder.retrieve_all(simple_asset_info.index),
                )),
                columns=('value',),
            )
            dates = self.dates
            self._run_pipeline(
                expr,
                deltas,
                None,
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
            'sid': self.ASSET_FINDER_EQUITY_SIDS * 2,
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
        expected_views_all_deltas = keymap(pd.Timestamp, {
            '2014-01-03': np.array([[10.0, 11.0, 12.0],
                                    [10.0, 11.0, 12.0],
                                    [10.0, 11.0, 12.0]]),
            '2014-01-06': np.array([[10.0, 11.0, 12.0],
                                    [10.0, 11.0, 12.0],
                                    [11.0, 12.0, 13.0]]),
        })
        # The only novel delta is on 2014-01-05, because it modifies a
        # baseline data point that occurred on 2014-01-04, which is on a
        # Saturday. The other delta, occurring on 2014-01-02, is seen after
        # we already see the baseline data it modifies, and so it is a
        # non-novel delta. Thus, the only delta seen in the expected view for
        # novel deltas is on 2014-01-06 at (2, 0), (2, 1), and (2, 2).
        expected_views_novel_deltas = keymap(pd.Timestamp, {
            '2014-01-03': np.array([[0.0, 1.0, 2.0],
                                    [0.0, 1.0, 2.0],
                                    [0.0, 1.0, 2.0]]),
            '2014-01-06': np.array([[0.0, 1.0, 2.0],
                                    [0.0, 1.0, 2.0],
                                    [11.0, 12.0, 13.0]]),
        })

        def get_fourth_asset_view(expected_views, window_length):
            return valmap(
                lambda view: np.c_[view, [np.nan] * window_length],
                expected_views,
            )

        if len(asset_info) == 4:
            expected_views_all_deltas = get_fourth_asset_view(
                expected_views_all_deltas, window_length=3
            )
            expected_views_novel_deltas = get_fourth_asset_view(
                expected_views_novel_deltas, window_length=3
            )
            expected_output_buffer_all_deltas = [
                10, 11, 12, np.nan, 11, 12, 13, np.nan
            ]
            expected_output_buffer_novel_deltas = [
                0, 1, 2, np.nan, 11, 12, 13, np.nan
            ]
        else:
            expected_output_buffer_all_deltas = [
                10, 11, 12, 11, 12, 13
            ]
            expected_output_buffer_novel_deltas = [
                0, 1, 2, 11, 12, 13
            ]

        cal = pd.DatetimeIndex([
            pd.Timestamp('2014-01-01'),
            pd.Timestamp('2014-01-02'),
            pd.Timestamp('2014-01-03'),
            # omitting the 4th and 5th to simulate a weekend
            pd.Timestamp('2014-01-06'),
        ])

        with tmp_asset_finder(equities=asset_info) as finder:
            expected_output_all_deltas = pd.DataFrame(
                expected_output_buffer_all_deltas,
                index=pd.MultiIndex.from_product((
                    sorted(expected_views_all_deltas.keys()),
                    finder.retrieve_all(asset_info.index),
                )),
                columns=('value',),
            )
            expected_output_novel_deltas = pd.DataFrame(
                expected_output_buffer_novel_deltas,
                index=pd.MultiIndex.from_product((
                    sorted(expected_views_novel_deltas.keys()),
                    finder.retrieve_all(asset_info.index),
                )),
                columns=('value',),
            )

            it = (
                (
                    True,
                    expected_views_all_deltas,
                    expected_output_all_deltas
                ),
                (
                    False,
                    expected_views_novel_deltas,
                    expected_output_novel_deltas
                )
            )
            for apply_deltas_adjs, expected_views, expected_output in it:
                self._run_pipeline(
                    expr,
                    deltas,
                    None,
                    expected_views,
                    expected_output,
                    finder,
                    calendar=cal,
                    start=cal[2],
                    end=cal[-1],
                    window_length=3,
                    compute_fn=op.itemgetter(-1),
                    apply_deltas_adjustments=apply_deltas_adjs,
                )

    def test_novel_deltas_macro(self):
        base_dates = pd.DatetimeIndex([
            pd.Timestamp('2014-01-01'),
            pd.Timestamp('2014-01-04')
        ])
        baseline = pd.DataFrame({
            'value': (0., 1.),
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
        nassets = len(simple_asset_info)
        expected_views_all_deltas = keymap(pd.Timestamp, {
            '2014-01-03': np.array([[10.0],
                                    [10.0],
                                    [10.0]]),
            '2014-01-06': np.array([[10.0],
                                    [10.0],
                                    [11.0]]),
        })
        # The only novel delta is on 2014-01-05, because it modifies a
        # baseline data point that occurred on 2014-01-04, which is on a
        # Saturday. The other delta, occurring on 2014-01-02, is seen after
        # we already see the baseline data it modifies, and so it is a
        # non-novel delta. Thus, the only delta seen in the expected view for
        # novel deltas is on 2014-01-06 at (2, 0).
        expected_views_novel_deltas = keymap(pd.Timestamp, {
            '2014-01-03': np.array([[0.0],
                                    [0.0],
                                    [0.0]]),
            '2014-01-06': np.array([[0.0],
                                    [0.0],
                                    [11.0]]),
        })

        cal = pd.DatetimeIndex([
            pd.Timestamp('2014-01-01'),
            pd.Timestamp('2014-01-02'),
            pd.Timestamp('2014-01-03'),
            # omitting the 4th and 5th to simulate a weekend
            pd.Timestamp('2014-01-06'),
        ])

        def get_expected_output(expected_views, values, asset_info):
            return pd.DataFrame(
                list(concatv(*([value] * nassets for value in values))),
                index=pd.MultiIndex.from_product(
                    (sorted(expected_views.keys()),
                     finder.retrieve_all(asset_info.index),)
                ), columns=('value',),
            )
        with tmp_asset_finder(equities=simple_asset_info) as finder:
            expected_output_all_deltas = get_expected_output(
                expected_views_all_deltas,
                [10, 11],
                simple_asset_info,
            )
            expected_output_novel_deltas = get_expected_output(
                expected_views_novel_deltas,
                [0, 11],
                simple_asset_info,
            )
            it = (
                (
                    True,
                    expected_views_all_deltas,
                    expected_output_all_deltas
                ),
                (
                    False,
                    expected_views_novel_deltas,
                    expected_output_novel_deltas
                )
            )
            for apply_deltas_adjs, expected_views, expected_output in it:
                self._run_pipeline(
                    expr,
                    deltas,
                    None,
                    expected_views,
                    expected_output,
                    finder,
                    calendar=cal,
                    start=cal[2],
                    end=cal[-1],
                    window_length=3,
                    compute_fn=op.itemgetter(-1),
                    apply_deltas_adjustments=apply_deltas_adjs,
                )

    def _test_checkpoints_macro(self, checkpoints, ffilled_value=-1.0):
        """Simple checkpoints test that accepts a checkpoints dataframe and
        the expected value for 2014-01-03 for macro datasets.

        The underlying data has value -1.0 on 2014-01-01 and 1.0 on 2014-01-04.

        Parameters
        ----------
        checkpoints : pd.DataFrame
            The checkpoints data.
        ffilled_value : float, optional
            The value to be read on the third, if not provided, it will be the
            value in the base data that will be naturally ffilled there.
        """
        dates = pd.Timestamp('2014-01-01'), pd.Timestamp('2014-01-04')
        baseline = pd.DataFrame({
            'value': [-1.0, 1.0],
            'asof_date': dates,
            'timestamp': dates,
        })

        nassets = len(simple_asset_info)
        expected_views = keymap(pd.Timestamp, {
            '2014-01-03': np.array([[ffilled_value]]),
            '2014-01-04': np.array([[1.0]]),
        })

        with tmp_asset_finder(equities=simple_asset_info) as finder:
            expected_output = pd.DataFrame(
                list(concatv([ffilled_value] * nassets, [1.0] * nassets)),
                index=pd.MultiIndex.from_product((
                    sorted(expected_views.keys()),
                    finder.retrieve_all(simple_asset_info.index),
                )),
                columns=('value',),
            )

            self._run_pipeline(
                bz.data(baseline, name='expr', dshape=self.macro_dshape),
                None,
                bz.data(
                    checkpoints,
                    name='expr_checkpoints',
                    dshape=self.macro_dshape,
                ),
                expected_views,
                expected_output,
                finder,
                calendar=pd.date_range('2014-01-01', '2014-01-04'),
                start=pd.Timestamp('2014-01-03'),
                end=dates[-1],
                window_length=1,
                compute_fn=op.itemgetter(-1),
            )

    def test_checkpoints_macro(self):
        ffilled_value = 0.0

        checkpoints_ts = pd.Timestamp('2014-01-02')
        checkpoints = pd.DataFrame({
            'value': [ffilled_value],
            'asof_date': checkpoints_ts,
            'timestamp': checkpoints_ts,
        })

        self._test_checkpoints_macro(checkpoints, ffilled_value)

    def test_empty_checkpoints_macro(self):
        empty_checkpoints = pd.DataFrame({
            'value': [],
            'asof_date': [],
            'timestamp': [],
        })

        self._test_checkpoints_macro(empty_checkpoints)

    def test_checkpoints_out_of_bounds_macro(self):
        # provide two checkpoints, one before the data in the base table
        # and one after, these should not affect the value on the third
        dates = pd.to_datetime(['2013-12-31', '2014-01-05'])
        checkpoints = pd.DataFrame({
            'value': [-2, 2],
            'asof_date': dates,
            'timestamp': dates,
        })

        self._test_checkpoints_macro(checkpoints)

    def _test_checkpoints(self, checkpoints, ffilled_values=None):
        """Simple checkpoints test that accepts a checkpoints dataframe and
        the expected value for 2014-01-03.

        The underlying data has value -1.0 on 2014-01-01 and 1.0 on 2014-01-04.

        Parameters
        ----------
        checkpoints : pd.DataFrame
            The checkpoints data.
        ffilled_value : float, optional
            The value to be read on the third, if not provided, it will be the
            value in the base data that will be naturally ffilled there.
        """
        nassets = len(simple_asset_info)

        dates = pd.to_datetime(['2014-01-01', '2014-01-04'])
        dates_repeated = np.tile(dates, nassets)
        values = np.arange(nassets) + 1
        values = np.hstack((values[::-1], values))
        baseline = pd.DataFrame({
            'sid': np.tile(simple_asset_info.index, 2),
            'value': values,
            'asof_date': dates_repeated,
            'timestamp': dates_repeated,
        })

        if ffilled_values is None:
            ffilled_values = baseline.value.iloc[:nassets]

        updated_values = baseline.value.iloc[nassets:]

        expected_views = keymap(pd.Timestamp, {
            '2014-01-03': [ffilled_values],
            '2014-01-04': [updated_values],
        })

        with tmp_asset_finder(equities=simple_asset_info) as finder:
            expected_output = pd.DataFrame(
                list(concatv(ffilled_values, updated_values)),
                index=pd.MultiIndex.from_product((
                    sorted(expected_views.keys()),
                    finder.retrieve_all(simple_asset_info.index),
                )),
                columns=('value',),
            )

            self._run_pipeline(
                bz.data(baseline, name='expr', dshape=self.value_dshape),
                None,
                bz.data(
                    checkpoints,
                    name='expr_checkpoints',
                    dshape=self.value_dshape,
                ),
                expected_views,
                expected_output,
                finder,
                calendar=pd.date_range('2014-01-01', '2014-01-04'),
                start=pd.Timestamp('2014-01-03'),
                end=dates[-1],
                window_length=1,
                compute_fn=op.itemgetter(-1),
            )

    def test_checkpoints(self):
        nassets = len(simple_asset_info)
        ffilled_values = (np.arange(nassets, dtype=np.float64) + 1) * 10
        dates = [pd.Timestamp('2014-01-02')] * nassets
        checkpoints = pd.DataFrame({
            'sid': simple_asset_info.index,
            'value': ffilled_values,
            'asof_date': dates,
            'timestamp': dates,
        })

        self._test_checkpoints(checkpoints, ffilled_values)

    def test_empty_checkpoints(self):
        checkpoints = pd.DataFrame({
            'sid': [],
            'value': [],
            'asof_date': [],
            'timestamp': [],
        })

        self._test_checkpoints(checkpoints)

    def test_checkpoints_out_of_bounds(self):
        nassets = len(simple_asset_info)
        # provide two sets of checkpoints, one before the data in the base
        # table and one after, these should not affect the value on the third
        dates = pd.to_datetime(['2013-12-31', '2014-01-05'])
        dates_repeated = np.tile(dates, nassets)
        ffilled_values = (np.arange(nassets) + 2) * 10
        ffilled_values = np.hstack((ffilled_values[::-1], ffilled_values))
        checkpoints = pd.DataFrame({
            'sid': np.tile(simple_asset_info.index, 2),
            'value': ffilled_values,
            'asof_date': dates_repeated,
            'timestamp': dates_repeated,
        })

        self._test_checkpoints(checkpoints)


class MiscTestCase(ZiplineTestCase):
    def test_exprdata_repr(self):
        strd = set()

        class BadRepr(object):
            """A class which cannot be repr'd.
            """
            def __init__(self, name):
                self._name = name

            def __repr__(self):  # pragma: no cover
                raise AssertionError('ayy')

            def __str__(self):
                strd.add(self)
                return self._name

        assert_equal(
            repr(ExprData(
                expr=BadRepr('expr'),
                deltas=BadRepr('deltas'),
                checkpoints=BadRepr('checkpoints'),
                odo_kwargs={'a': 'b'},
            )),
            "ExprData(expr='expr', deltas='deltas',"
            " checkpoints='checkpoints', odo_kwargs={'a': 'b'}, "
            "apply_deltas_adjustments=True)",
        )

    def test_blaze_loader_repr(self):
        assert_equal(repr(BlazeLoader()), '<BlazeLoader: {}>')

    def test_blaze_loader_lookup_failure(self):
        class D(DataSet):
            c = Column(dtype='float64')

        with self.assertRaises(KeyError) as e:
            BlazeLoader()(D.c)
        assert_equal(str(e.exception), 'D.c::float64')
