from __future__ import division

import numpy as np
import pandas as pd
from toolz import merge
import toolz.curried.operator as op

from zipline import get_calendar
from zipline.data.bundles import ingest, load, bundles, register
from zipline.data.bundles.quandl import (
    format_wiki_url,
    format_metadata_url,
)
from zipline.data.bundles.zacks_quandl import from_zacks_dump
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


class ZacksBundleTestCase(ZiplineTestCase):
    """
    Class for testing the Zacks daily data bundle.
    An test file is stored in tests/resources/zacks_samples/fictious.csv

    """
    symbols = 'MFF', 'JMH', 'PBH'
    asset_start = pd.Timestamp('2016-04-18', tz='utc')
    asset_end = pd.Timestamp('2016-07-06', tz='utc')
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

        # load data from CSV
        df = pd.read_csv(test_resource_path('zacks_samples', 'fictitious.csv'),
                         index_col='date',
                         parse_dates=['date'],
                         usecols=[
                             'date',
                             'open',
                             'high',
                             'low',
                             'close',
                             'volume',
                             'ticker'
                         ],
                         na_values=['NA']
                         )
        df = df.dropna()  # drop NA rows (non trading days) or loader will wipe out entire column

        df = df.replace({"ticker": sids})            # convert ticker to sids
        df = df.rename(columns={"ticker": "sid"})
        df["volume"] = np.floor(df["volume"])        # zacks data contains fractional shares, these get dropped

        # split one large DataFrame into one per sid (this also drops unwanted tickers)
        subs = [df[df['sid'] == sid] for sid in sorted(sids.values())]

        # package up data from CSV so that it is in the same format as data coming out of the bundle
        # the format is a list of 5 2D arrays one for each OHLCV
        pricing = []
        for column in self.columns:
            vs = np.zeros((subs[0].shape[0], len(subs)))
            for i, sub in enumerate(subs):
                vs[:, i] = sub[column].values
            if column == 'volume':
                vs = np.nan_to_num(vs)
            pricing.append(vs)

        return pricing, []

    def test_bundle(self):
        zipline_root = self.enter_instance_context(tmp_dir()).path
        environ = {
            'ZIPLINE_ROOT': zipline_root,
            'QUANDL_API_KEY': self.api_key,
        }

        # custom bundles need to be registered before use or they will not be recognized
        register('ZacksQuandl', from_zacks_dump(test_resource_path('zacks_samples', 'fictitious.csv')))
        ingest('ZacksQuandl', environ=environ)

        # load bundle now that it has been ingested
        bundle = load('ZacksQuandl', environ=environ)
        sids = 0, 1, 2

        # check sids match
        assert_equal(set(bundle.asset_finder.sids), set(sids))

        # check asset_{start, end} is the same as {start, end}_date
        for equity in bundle.asset_finder.retrieve_all(sids):
            assert_equal(equity.start_date, self.asset_start, msg=equity)
            assert_equal(equity.end_date, self.asset_end, msg=equity)

        # get daily OHLCV data from bundle
        sessions = self.calendar.all_sessions
        actual = bundle.equity_daily_bar_reader.load_raw_arrays(
            self.columns,
            sessions[sessions.get_loc(self.asset_start, 'bfill')],
            sessions[sessions.get_loc(self.asset_end, 'ffill')],
            sids,
        )

        # get expected data from csv
        expected_pricing, expected_adjustments = self._expected_data(
            bundle.asset_finder,
        )

        # check OHLCV data matches
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
