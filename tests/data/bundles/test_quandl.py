from __future__ import division

import numpy as np
import pandas as pd
from toolz import merge
import toolz.curried.operator as op

from zipline import get_calendar
from zipline.data.bundles import ingest, load, bundles
from zipline.data.bundles.quandl import (
    format_wiki_url,
    format_metadata_url,
)
from zipline.lib.adjustment import Float64Multiply
from zipline.testing import (
    test_resource_path,
    tmp_dir,
    patch_read_csv,
)
from zipline.testing.fixtures import ZiplineTestCase
from zipline.testing.predicates import (
    assert_equal,
)
from zipline.utils.functional import apply


class QuandlBundleTestCase(ZiplineTestCase):
    symbols = 'AAPL', 'BRK_A', 'MSFT', 'ZEN'
    asset_start = pd.Timestamp('2014-01', tz='utc')
    asset_end = pd.Timestamp('2015-01', tz='utc')
    bundle = bundles['quandl']
    calendar = get_calendar(bundle.calendar_name)
    start_date = calendar.first_session
    end_date = calendar.last_session
    api_key = 'ayylmao'
    columns = 'open', 'high', 'low', 'close', 'volume'

    def _expected_data(self, asset_finder):
        sids = {
            symbol: asset_finder.lookup_symbol(
                symbol,
                self.asset_start,
            ).sid
            for symbol in self.symbols
        }

        def per_symbol(symbol):
            df = pd.read_csv(
                test_resource_path('quandl_samples', symbol + '.csv.gz'),
                parse_dates=['Date'],
                index_col='Date',
                usecols=[
                    'Open',
                    'High',
                    'Low',
                    'Close',
                    'Volume',
                    'Date',
                    'Ex-Dividend',
                    'Split Ratio',
                ],
                na_values=['NA'],
            ).rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume',
                'Date': 'date',
                'Ex-Dividend': 'ex_dividend',
                'Split Ratio': 'split_ratio',
            })
            df['sid'] = sids[symbol]
            return df

        all_ = pd.concat(map(per_symbol, self.symbols)).set_index(
            'sid',
            append=True,
        ).unstack()

        # fancy list comprehension with statements
        @list
        @apply
        def pricing():
            for column in self.columns:
                vs = all_[column].values
                if column == 'volume':
                    vs = np.nan_to_num(vs)
                yield vs

        # the first index our written data will appear in the files on disk
        start_idx = (
            self.calendar.all_sessions.get_loc(self.asset_start, 'ffill') + 1
        )

        # convert an index into the raw dataframe into an index into the
        # final data
        i = op.add(start_idx)

        def expected_dividend_adjustment(idx, symbol):
            sid = sids[symbol]
            return (
                1 -
                all_.ix[idx, ('ex_dividend', sid)] /
                all_.ix[idx - 1, ('close', sid)]
            )

        adjustments = [
            # ohlc
            {
                # dividends
                i(24): [Float64Multiply(
                    first_row=0,
                    last_row=i(24),
                    first_col=sids['AAPL'],
                    last_col=sids['AAPL'],
                    value=expected_dividend_adjustment(24, 'AAPL'),
                )],
                i(87): [Float64Multiply(
                    first_row=0,
                    last_row=i(87),
                    first_col=sids['AAPL'],
                    last_col=sids['AAPL'],
                    value=expected_dividend_adjustment(87, 'AAPL'),
                )],
                i(150): [Float64Multiply(
                    first_row=0,
                    last_row=i(150),
                    first_col=sids['AAPL'],
                    last_col=sids['AAPL'],
                    value=expected_dividend_adjustment(150, 'AAPL'),
                )],
                i(214): [Float64Multiply(
                    first_row=0,
                    last_row=i(214),
                    first_col=sids['AAPL'],
                    last_col=sids['AAPL'],
                    value=expected_dividend_adjustment(214, 'AAPL'),
                )],

                i(31): [Float64Multiply(
                    first_row=0,
                    last_row=i(31),
                    first_col=sids['MSFT'],
                    last_col=sids['MSFT'],
                    value=expected_dividend_adjustment(31, 'MSFT'),
                )],
                i(90): [Float64Multiply(
                    first_row=0,
                    last_row=i(90),
                    first_col=sids['MSFT'],
                    last_col=sids['MSFT'],
                    value=expected_dividend_adjustment(90, 'MSFT'),
                )],
                i(222): [Float64Multiply(
                    first_row=0,
                    last_row=i(222),
                    first_col=sids['MSFT'],
                    last_col=sids['MSFT'],
                    value=expected_dividend_adjustment(222, 'MSFT'),
                )],

                # splits
                i(108): [Float64Multiply(
                    first_row=0,
                    last_row=i(108),
                    first_col=sids['AAPL'],
                    last_col=sids['AAPL'],
                    value=1.0 / 7.0,
                )],
            },
        ] * (len(self.columns) - 1) + [
            # volume
            {
                i(108): [Float64Multiply(
                    first_row=0,
                    last_row=i(108),
                    first_col=sids['AAPL'],
                    last_col=sids['AAPL'],
                    value=7.0,
                )],
            }
        ]
        return pricing, adjustments

    def test_bundle(self):
        url_map = merge(
            {
                format_wiki_url(
                    self.api_key,
                    symbol,
                    self.start_date,
                    self.end_date,
                ): test_resource_path('quandl_samples', symbol + '.csv.gz')
                for symbol in self.symbols
            },
            {
                format_metadata_url(self.api_key, n): test_resource_path(
                    'quandl_samples',
                    'metadata-%d.csv.gz' % n,
                )
                for n in (1, 2)
            },
        )
        zipline_root = self.enter_instance_context(tmp_dir()).path
        environ = {
            'ZIPLINE_ROOT': zipline_root,
            'QUANDL_API_KEY': self.api_key,
        }

        with patch_read_csv(url_map, strict=True):
            ingest('quandl', environ=environ)

        bundle = load('quandl', environ=environ)
        sids = 0, 1, 2, 3
        assert_equal(set(bundle.asset_finder.sids), set(sids))

        for equity in bundle.asset_finder.retrieve_all(sids):
            assert_equal(equity.start_date, self.asset_start, msg=equity)
            assert_equal(equity.end_date, self.asset_end, msg=equity)

        sessions = self.calendar.all_sessions
        actual = bundle.equity_daily_bar_reader.load_raw_arrays(
            self.columns,
            sessions[sessions.get_loc(self.asset_start, 'bfill')],
            sessions[sessions.get_loc(self.asset_end, 'ffill')],
            sids,
        )
        expected_pricing, expected_adjustments = self._expected_data(
            bundle.asset_finder,
        )
        assert_equal(actual, expected_pricing, array_decimal=2)

        adjustments_for_cols = bundle.adjustment_reader.load_adjustments(
            self.columns,
            sessions,
            pd.Index(sids),
        )

        for column, adjustments, expected in zip(self.columns,
                                                 adjustments_for_cols,
                                                 expected_adjustments):
            assert_equal(
                adjustments,
                expected,
                msg=column,
            )
