import os

from nose_parameterized import parameterized
import pandas as pd
from toolz import valmap
import toolz.curried.operator as op

from zipline.assets.synthetic import make_simple_equity_info
from zipline.data.bundles import UnknownBundle, from_bundle_ingest_dirname
from zipline.data.bundles.core import _make_bundle_core
from zipline.lib.adjustment import Float64Multiply
from zipline.pipeline.loaders.synthetic import (
    make_bar_data,
    expected_bar_values_2d,
)
from zipline.testing import (
    subtest,
    str_to_seconds,
)
from zipline.testing.fixtures import WithInstanceTmpDir, ZiplineTestCase
from zipline.testing.predicates import (
    assert_equal,
    assert_false,
    assert_in,
    assert_is,
    assert_is_instance,
    assert_is_none,
    assert_raises,
    assert_true,
)
from zipline.utils.cache import dataframe_cache
from zipline.utils.functional import apply
from zipline.utils.calendars import get_calendar
import zipline.utils.paths as pth


_1_ns = pd.Timedelta(1, unit='ns')


class BundleCoreTestCase(WithInstanceTmpDir, ZiplineTestCase):
    def init_instance_fixtures(self):
        super(BundleCoreTestCase, self).init_instance_fixtures()
        (self.bundles,
         self.register,
         self.unregister,
         self.ingest,
         self.load,
         self.clean) = _make_bundle_core()
        self.environ = {'ZIPLINE_ROOT': self.instance_tmpdir.path}

    def test_register_decorator(self):
        @apply
        @subtest(((c,) for c in 'abcde'), 'name')
        def _(name):
            @self.register(name)
            def ingest(*args):
                pass

            assert_in(name, self.bundles)
            assert_is(self.bundles[name].ingest, ingest)

        self._check_bundles(set('abcde'))

    def test_register_call(self):
        def ingest(*args):
            pass

        @apply
        @subtest(((c,) for c in 'abcde'), 'name')
        def _(name):
            self.register(name, ingest)
            assert_in(name, self.bundles)
            assert_is(self.bundles[name].ingest, ingest)

        assert_equal(
            valmap(op.attrgetter('ingest'), self.bundles),
            {k: ingest for k in 'abcde'},
        )
        self._check_bundles(set('abcde'))

    def _check_bundles(self, names):
        assert_equal(set(self.bundles.keys()), names)

        for name in names:
            self.unregister(name)

        assert_false(self.bundles)

    def test_register_no_create(self):
        called = [False]

        @self.register('bundle', create_writers=False)
        def bundle_ingest(environ,
                          asset_db_writer,
                          minute_bar_writer,
                          daily_bar_writer,
                          adjustment_writer,
                          calendar,
                          cache,
                          show_progress,
                          output_dir):
            assert_is_none(asset_db_writer)
            assert_is_none(minute_bar_writer)
            assert_is_none(daily_bar_writer)
            assert_is_none(adjustment_writer)
            called[0] = True

        self.ingest('bundle', self.environ)
        assert_true(called[0])

    def test_ingest(self):
        start = pd.Timestamp('2014-01-06', tz='utc')
        end = pd.Timestamp('2014-01-10', tz='utc')
        trading_days = get_calendar('NYSE').all_sessions
        calendar = trading_days[trading_days.slice_indexer(start, end)]
        minutes = get_calendar('NYSE').minutes_for_sessions_in_range(
            calendar[0], calendar[-1]
        )

        sids = tuple(range(3))
        equities = make_simple_equity_info(
            sids,
            calendar[0],
            calendar[-1],
        )

        daily_bar_data = make_bar_data(equities, calendar)
        minute_bar_data = make_bar_data(equities, minutes)
        first_split_ratio = 0.5
        second_split_ratio = 0.1
        splits = pd.DataFrame.from_records([
            {
                'effective_date': str_to_seconds('2014-01-08'),
                'ratio': first_split_ratio,
                'sid': 0,
            },
            {
                'effective_date': str_to_seconds('2014-01-09'),
                'ratio': second_split_ratio,
                'sid': 1,
            },
        ])

        schedule = get_calendar('NYSE').schedule

        @self.register(
            'bundle',
            calendar=calendar,
            opens=schedule.market_open[calendar[0]:calendar[-1]],
            closes=schedule.market_close[calendar[0]: calendar[-1]],
        )
        def bundle_ingest(environ,
                          asset_db_writer,
                          minute_bar_writer,
                          daily_bar_writer,
                          adjustment_writer,
                          calendar,
                          cache,
                          show_progress,
                          output_dir):
            assert_is(environ, self.environ)

            asset_db_writer.write(equities=equities)
            minute_bar_writer.write(minute_bar_data)
            daily_bar_writer.write(daily_bar_data)
            adjustment_writer.write(splits=splits)

            assert_is_instance(calendar, pd.DatetimeIndex)
            assert_is_instance(cache, dataframe_cache)
            assert_is_instance(show_progress, bool)

        self.ingest('bundle', environ=self.environ)
        bundle = self.load('bundle', environ=self.environ)

        assert_equal(set(bundle.asset_finder.sids), set(sids))

        columns = 'open', 'high', 'low', 'close', 'volume'

        actual = bundle.equity_minute_bar_reader.load_raw_arrays(
            columns,
            minutes[0],
            minutes[-1],
            sids,
        )

        for actual_column, colname in zip(actual, columns):
            assert_equal(
                actual_column,
                expected_bar_values_2d(minutes, equities, colname),
                msg=colname,
            )

        actual = bundle.equity_daily_bar_reader.load_raw_arrays(
            columns,
            calendar[0],
            calendar[-1],
            sids,
        )
        for actual_column, colname in zip(actual, columns):
            assert_equal(
                actual_column,
                expected_bar_values_2d(calendar, equities, colname),
                msg=colname,
            )
        adjustments_for_cols = bundle.adjustment_reader.load_adjustments(
            columns,
            calendar,
            pd.Index(sids),
        )
        for column, adjustments in zip(columns, adjustments_for_cols[:-1]):
            # iterate over all the adjustments but `volume`
            assert_equal(
                adjustments,
                {
                    2: [Float64Multiply(
                        first_row=0,
                        last_row=2,
                        first_col=0,
                        last_col=0,
                        value=first_split_ratio,
                    )],
                    3: [Float64Multiply(
                        first_row=0,
                        last_row=3,
                        first_col=1,
                        last_col=1,
                        value=second_split_ratio,
                    )],
                },
                msg=column,
            )

        # check the volume, the value should be 1/ratio
        assert_equal(
            adjustments_for_cols[-1],
            {
                2: [Float64Multiply(
                    first_row=0,
                    last_row=2,
                    first_col=0,
                    last_col=0,
                    value=1 / first_split_ratio,
                )],
                3: [Float64Multiply(
                    first_row=0,
                    last_row=3,
                    first_col=1,
                    last_col=1,
                    value=1 / second_split_ratio,
                )],
            },
            msg='volume',
        )

    @parameterized.expand([('clean',), ('load',)])
    def test_bundle_doesnt_exist(self, fnname):
        with assert_raises(UnknownBundle) as e:
            getattr(self, fnname)('ayy', environ=self.environ)

        assert_equal(e.exception.name, 'ayy')

    def test_load_no_data(self):
        # register but do not ingest data
        self.register('bundle', lambda *args: None)

        ts = pd.Timestamp('2014')

        with assert_raises(ValueError) as e:
            self.load('bundle', timestamp=ts, environ=self.environ)

        assert_in(
            "no data for bundle 'bundle' on or before %s" % ts,
            str(e.exception),
        )

    def _list_bundle(self):
        return {
            os.path.join(pth.data_path(['bundle', d], environ=self.environ))
            for d in os.listdir(
                pth.data_path(['bundle'], environ=self.environ),
            )
        }

    def _empty_ingest(self, _wrote_to=[]):
        """Run the nth empty ingest.

        Returns
        -------
        wrote_to : str
            The timestr of the bundle written.
        """
        if not self.bundles:
            @self.register('bundle',
                           calendar=pd.DatetimeIndex([pd.Timestamp('2014')]))
            def _(environ,
                  asset_db_writer,
                  minute_bar_writer,
                  daily_bar_writer,
                  adjustment_writer,
                  calendar,
                  cache,
                  show_progress,
                  output_dir):
                _wrote_to.append(output_dir)

        _wrote_to.clear()
        self.ingest('bundle', environ=self.environ)
        assert_equal(len(_wrote_to), 1, msg='ingest was called more than once')
        ingestions = self._list_bundle()
        assert_in(
            _wrote_to[0],
            ingestions,
            msg='output_dir was not in the bundle directory',
        )
        return _wrote_to[0]

    def test_clean_keep_last(self):
        first = self._empty_ingest()

        assert_equal(
            self.clean('bundle', keep_last=1, environ=self.environ),
            set(),
        )
        assert_equal(
            self._list_bundle(),
            {first},
            msg='directory should not have changed',
        )

        second = self._empty_ingest()
        assert_equal(
            self._list_bundle(),
            {first, second},
            msg='two ingestions are not present',
        )
        assert_equal(
            self.clean('bundle', keep_last=1, environ=self.environ),
            {first},
        )
        assert_equal(
            self._list_bundle(),
            {second},
            msg='first ingestion was not removed with keep_last=2',
        )

        third = self._empty_ingest()
        fourth = self._empty_ingest()
        fifth = self._empty_ingest()

        assert_equal(
            self._list_bundle(),
            {second, third, fourth, fifth},
            msg='larger set of ingestions did not happen correctly',
        )

        assert_equal(
            self.clean('bundle', keep_last=2, environ=self.environ),
            {second, third},
        )

        assert_equal(
            self._list_bundle(),
            {fourth, fifth},
            msg='keep_last=2 did not remove the correct number of ingestions',
        )

    @staticmethod
    def _ts_of_run(run):
        return from_bundle_ingest_dirname(run.rsplit(os.path.sep, 1)[-1])

    def test_clean_before_after(self):
        first = self._empty_ingest()
        assert_equal(
            self.clean(
                'bundle',
                before=self._ts_of_run(first),
                environ=self.environ,
            ),
            set(),
        )
        assert_equal(
            self._list_bundle(),
            {first},
            msg='directory should not have changed (before)',
        )

        assert_equal(
            self.clean(
                'bundle',
                after=self._ts_of_run(first),
                environ=self.environ,
            ),
            set(),
        )
        assert_equal(
            self._list_bundle(),
            {first},
            msg='directory should not have changed (after)',
        )

        assert_equal(
            self.clean(
                'bundle',
                before=self._ts_of_run(first) + _1_ns,
                environ=self.environ,
            ),
            {first},
        )
        assert_equal(
            self._list_bundle(),
            set(),
            msg='directory now be empty (before)',
        )

        second = self._empty_ingest()
        assert_equal(
            self.clean(
                'bundle',
                after=self._ts_of_run(second) - _1_ns,
                environ=self.environ,
            ),
            {second},
        )
        assert_equal(
            self._list_bundle(),
            set(),
            msg='directory now be empty (after)',
        )

        third = self._empty_ingest()
        fourth = self._empty_ingest()
        fifth = self._empty_ingest()
        sixth = self._empty_ingest()

        assert_equal(
            self._list_bundle(),
            {third, fourth, fifth, sixth},
            msg='larger set of ingestions did no happen correctly',
        )

        assert_equal(
            self.clean(
                'bundle',
                before=self._ts_of_run(fourth),
                after=self._ts_of_run(fifth),
                environ=self.environ,
            ),
            {third, sixth},
        )

        assert_equal(
            self._list_bundle(),
            {fourth, fifth},
            msg='did not strip first and last directories',
        )
