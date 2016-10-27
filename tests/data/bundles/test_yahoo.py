from __future__ import division

import numpy as np
import pandas as pd
from six.moves.urllib.parse import urlparse, parse_qs
from toolz import flip, identity
from toolz.curried import merge_with, operator as op

from zipline.data.bundles.core import _make_bundle_core
from zipline.data.bundles import yahoo_equities
from zipline.lib.adjustment import Float64Multiply
from zipline.testing import test_resource_path, tmp_dir, read_compressed
from zipline.testing.fixtures import WithResponses, ZiplineTestCase
from zipline.testing.predicates import assert_equal
from zipline.utils.calendars import get_calendar


class YahooBundleTestCase(WithResponses, ZiplineTestCase):
    symbols = 'AAPL', 'IBM', 'MSFT'
    columns = 'open', 'high', 'low', 'close', 'volume'
    asset_start = pd.Timestamp('2014-01-02', tz='utc')
    asset_end = pd.Timestamp('2014-12-31', tz='utc')
    calendar = get_calendar('NYSE')
    sessions = calendar.sessions_in_range(asset_start, asset_end)

    @classmethod
    def init_class_fixtures(cls):
        super(YahooBundleTestCase, cls).init_class_fixtures()
        (cls.bundles,
         cls.register,
         cls.unregister,
         cls.ingest,
         cls.load,
         cls.clean) = map(staticmethod, _make_bundle_core())

    def _expected_data(self):
        sids = 0, 1, 2
        modifier = {
            'low': 0,
            'open': 1,
            'close': 2,
            'high': 3,
            'volume': 0,
        }
        pricing = [
            np.hstack((
                np.arange(252, dtype='float64')[:, np.newaxis] +
                1 +
                sid * 10000 +
                modifier[column] * 1000
                for sid in sorted(sids)
            ))
            for column in self.columns
        ]

        # There are two dividends and 1 split for each company.

        def dividend_adjustment(sid, which):
            """The dividends occur at indices 252 // 4 and 3 * 252 / 4
            with a cash amount of sid + 1 / 10 and sid + 2 / 10
            """
            if which == 'first':
                idx = 252 // 4
            else:
                idx = 3 * 252 // 4

            return {
                idx: [Float64Multiply(
                    first_row=0,
                    last_row=idx,
                    first_col=sid,
                    last_col=sid,
                    value=float(
                        1 -
                        ((sid + 1 + (which == 'second')) / 10) /
                        (idx - 1 + sid * 10000 + 2000)
                    ),
                )],
            }

        def split_adjustment(sid, volume):
            """The splits occur at index 252 // 2 with a ratio of (sid + 1):1
            """
            idx = 252 // 2
            return {
                idx: [Float64Multiply(
                    first_row=0,
                    last_row=idx,
                    first_col=sid,
                    last_col=sid,
                    value=(identity if volume else op.truediv(1))(sid + 2),
                )],
            }

        merge_adjustments = merge_with(flip(sum, []))

        adjustments = [
            # ohlc
            merge_adjustments(
                *tuple(dividend_adjustment(sid, 'first') for sid in sids) +
                tuple(dividend_adjustment(sid, 'second') for sid in sids) +
                tuple(split_adjustment(sid, volume=False) for sid in sids)
            )
        ] * (len(self.columns) - 1) + [
            # volume
            merge_adjustments(
                split_adjustment(sid, volume=True) for sid in sids
            ),
        ]

        return pricing, adjustments

    def test_bundle(self):

        def get_symbol_from_url(url):
            params = parse_qs(urlparse(url).query)
            symbol, = params['s']
            return symbol

        def pricing_callback(request):
            headers = {
                'content-encoding': 'gzip',
                'content-type': 'text/csv',
            }
            path = test_resource_path(
                'yahoo_samples',
                get_symbol_from_url(request.url) + '.csv.gz',
            )
            with open(path, 'rb') as f:
                return (
                    200,
                    headers,
                    f.read(),
                )

        for _ in range(3):
            self.responses.add_callback(
                self.responses.GET,
                'http://ichart.finance.yahoo.com/table.csv',
                pricing_callback,
            )

        def adjustments_callback(request):
            path = test_resource_path(
                'yahoo_samples',
                get_symbol_from_url(request.url) + '.adjustments.gz',
            )
            return 200, {}, read_compressed(path)

        for _ in range(3):
            self.responses.add_callback(
                self.responses.GET,
                'http://ichart.finance.yahoo.com/x',
                adjustments_callback,
            )

        self.register(
            'bundle',
            yahoo_equities(self.symbols),
            calendar_name='NYSE',
            start_session=self.asset_start,
            end_session=self.asset_end,
        )

        zipline_root = self.enter_instance_context(tmp_dir()).path
        environ = {
            'ZIPLINE_ROOT': zipline_root,
        }

        self.ingest('bundle', environ=environ, show_progress=False)
        bundle = self.load('bundle', environ=environ)

        sids = 0, 1, 2
        equities = bundle.asset_finder.retrieve_all(sids)
        for equity, expected_symbol in zip(equities, self.symbols):
            assert_equal(equity.symbol, expected_symbol)

        for equity in bundle.asset_finder.retrieve_all(sids):
            assert_equal(equity.start_date, self.asset_start, msg=equity)
            assert_equal(equity.end_date, self.asset_end, msg=equity)

        sessions = self.sessions
        actual = bundle.equity_daily_bar_reader.load_raw_arrays(
            self.columns,
            sessions[sessions.get_loc(self.asset_start, 'bfill')],
            sessions[sessions.get_loc(self.asset_end, 'ffill')],
            sids,
        )
        expected_pricing, expected_adjustments = self._expected_data()
        assert_equal(actual, expected_pricing, array_decimal=2)

        adjustments_for_cols = bundle.adjustment_reader.load_adjustments(
            self.columns,
            self.sessions,
            pd.Index(sids),
        )

        for column, adjustments, expected in zip(self.columns,
                                                 adjustments_for_cols,
                                                 expected_adjustments):
            assert_equal(
                adjustments,
                expected,
                msg=column,
                decimal=4,
            )
