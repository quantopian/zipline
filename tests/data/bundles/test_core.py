import pandas as pd
from toolz import valmap
import toolz.curried.operator as op

from zipline.assets.synthetic import make_simple_equity_info
from zipline.data.bundles import load
from zipline.data.bundles.core import _make_bundle_core
from zipline.lib.adjustment import Float64Multiply
from zipline.pipeline.loaders.synthetic import (
    make_bar_data,
    expected_bar_values_2d,
)
from zipline.testing import (
    subtest,
    tmp_dir,
    str_to_seconds,
    tmp_trading_env,
)
from zipline.testing.fixtures import ZiplineTestCase
from zipline.testing.predicates import (
    assert_equal,
    assert_false,
    assert_in,
    assert_is,
    assert_is_instance,
)
from zipline.utils.cache import dataframe_cache
from zipline.utils.functional import apply
from zipline.utils.tradingcalendar import trading_days


class BundleCoreTestCase(ZiplineTestCase):
    def init_instance_fixtures(self):
        super(BundleCoreTestCase, self).init_instance_fixtures()
        (self.bundles,
         self.register,
         self.unregister,
         self.ingest) = _make_bundle_core()

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

    def test_ingest(self):
        zipline_root = self.enter_instance_context(tmp_dir()).path
        env = self.enter_instance_context(tmp_trading_env())

        start = pd.Timestamp('2014-01-06', tz='utc')
        end = pd.Timestamp('2014-01-10', tz='utc')
        calendar = trading_days[trading_days.slice_indexer(start, end)]
        minutes = env.minutes_for_days_in_range(calendar[0], calendar[-1])
        outer_environ = {
            'ZIPLINE_ROOT': zipline_root,
        }

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

        @self.register('bundle',
                       calendar=calendar,
                       opens=env.opens_in_range(calendar[0], calendar[-1]),
                       closes=env.closes_in_range(calendar[0], calendar[-1]))
        def bundle_ingest(environ,
                          asset_db_writer,
                          minute_bar_writer,
                          daily_bar_writer,
                          adjustment_writer,
                          calendar,
                          cache,
                          show_progress):
            assert_is(environ, outer_environ)

            asset_db_writer.write(equities=equities)
            minute_bar_writer.write(minute_bar_data)
            daily_bar_writer.write(daily_bar_data)
            adjustment_writer.write(splits=splits)

            assert_is_instance(calendar, pd.DatetimeIndex)
            assert_is_instance(cache, dataframe_cache)
            assert_is_instance(show_progress, bool)

        self.ingest('bundle', environ=outer_environ)
        bundle = load('bundle', environ=outer_environ)

        assert_equal(set(bundle.asset_finder.sids), set(sids))

        columns = 'open', 'high', 'low', 'close', 'volume'

        actual = bundle.minute_bar_reader.load_raw_arrays(
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

        actual = bundle.daily_bar_reader.load_raw_arrays(
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
