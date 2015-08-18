#
# Copyright 2015 Quantopian, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Tests for the zipline.assets package
"""

import sys
from unittest import TestCase

from datetime import datetime, timedelta
import pickle
import uuid
import warnings
import sqlite3

import pandas as pd
from pandas.tseries.tools import normalize_date
from pandas.util.testing import assert_frame_equal

from nose_parameterized import parameterized
from numpy import full

from zipline.assets import Asset, Equity, Future, AssetFinder
from zipline.assets.futures import FutureChain
from zipline.assets.asset_writer import AssetDBWriterFromDataFrame
from zipline.errors import (
    SymbolNotFound,
    MultipleSymbolsFound,
    SidAssignmentError,
    RootSymbolNotFound,
)
from zipline.finance.trading import with_environment
from zipline.utils.test_utils import (
    all_subindices,
    make_rotating_asset_info,
)


def build_lookup_generic_cases():
    """
    Generate test cases for AssetFinder test_lookup_generic.
    """

    unique_start = pd.Timestamp('2013-01-01', tz='UTC')
    unique_end = pd.Timestamp('2014-01-01', tz='UTC')

    dupe_0_start = pd.Timestamp('2013-01-01', tz='UTC')
    dupe_0_end = dupe_0_start + timedelta(days=1)

    dupe_1_start = pd.Timestamp('2013-01-03', tz='UTC')
    dupe_1_end = dupe_1_start + timedelta(days=1)

    frame = pd.DataFrame.from_records(
        [
            {
                'sid': 0,
                'asset_name': 'duplicated_0',
                'start_date': dupe_0_start.value,
                'end_date': dupe_0_end.value,
                'exchange': '',
            },
            {
                'sid': 1,
                'asset_name': 'duplicated_1',
                'start_date': dupe_1_start.value,
                'end_date': dupe_1_end.value,
                'exchange': '',
            },
            {
                'sid': 2,
                'asset_name': 'unique',
                'start_date': unique_start.value,
                'end_date': unique_end.value,
                'exchange': '',
            },
        ],
        index='sid')
    db_path = '~/temp.db'
    conn = sqlite3.connect(db_path)
    asset_writer = AssetDBWriterFromDataFrame(equities=frame)
    asset_writer.write_all(conn)
    finder = AssetFinder(conn)
    dupe_0, dupe_1, unique = assets = [
        finder.retrieve_asset(i)
        for i in range(3)
    ]

    dupe_0_start = dupe_0.start_date
    dupe_1_start = dupe_1.start_date
    cases = [
        ##
        # Scalars

        # Asset object
        (finder, assets[0], None, assets[0]),
        (finder, assets[1], None, assets[1]),
        (finder, assets[2], None, assets[2]),
        # int
        (finder, 0, None, assets[0]),
        (finder, 1, None, assets[1]),
        (finder, 2, None, assets[2]),
        # Duplicated symbol with resolution date
        (finder, 'duplicated', dupe_0_start, dupe_0),
        (finder, 'duplicated', dupe_1_start, dupe_1),
        # Unique symbol, with or without resolution date.
        (finder, 'unique', unique_start, unique),
        (finder, 'unique', None, unique),

        ##
        # Iterables

        # Iterables of Asset objects.
        (finder, assets, None, assets),
        (finder, iter(assets), None, assets),
        # Iterables of ints
        (finder, (0, 1), None, assets[:-1]),
        (finder, iter((0, 1)), None, assets[:-1]),
        # Iterables of symbols.
        (finder, ('duplicated', 'unique'), dupe_0_start, [dupe_0, unique]),
        (finder, ('duplicated', 'unique'), dupe_1_start, [dupe_1, unique]),
        # Mixed types
        (finder,
         ('duplicated', 2, 'unique', 1, dupe_1),
         dupe_0_start,
         [dupe_0, assets[2], unique, assets[1], dupe_1]),
    ]
    return cases


class AssetTestCase(TestCase):

    def test_asset_object(self):
        self.assertEquals({5061: 'foo'}[Asset(5061)], 'foo')
        self.assertEquals(Asset(5061), 5061)
        self.assertEquals(5061, Asset(5061))

        self.assertEquals(Asset(5061), Asset(5061))
        self.assertEquals(int(Asset(5061)), 5061)

        self.assertEquals(str(Asset(5061)), 'Asset(5061)')

    def test_asset_is_pickleable(self):

        # Very wow
        s = Asset(
            1337,
            symbol="DOGE",
            asset_name="DOGECOIN",
            start_date=pd.Timestamp('2013-12-08 9:31AM', tz='UTC'),
            end_date=pd.Timestamp('2014-06-25 11:21AM', tz='UTC'),
            first_traded=pd.Timestamp('2013-12-08 9:31AM', tz='UTC'),
            exchange='THE MOON',
        )
        s_unpickled = pickle.loads(pickle.dumps(s))

        attrs_to_check = ['end_date',
                          'exchange',
                          'first_traded',
                          'end_date',
                          'asset_name',
                          'start_date',
                          'sid',
                          'start_date',
                          'symbol']

        for attr in attrs_to_check:
            self.assertEqual(getattr(s, attr), getattr(s_unpickled, attr))

    def test_asset_comparisons(self):

        s_23 = Asset(23)
        s_24 = Asset(24)

        self.assertEqual(s_23, s_23)
        self.assertEqual(s_23, 23)
        self.assertEqual(23, s_23)

        self.assertNotEqual(s_23, s_24)
        self.assertNotEqual(s_23, 24)
        self.assertNotEqual(s_23, "23")
        self.assertNotEqual(s_23, 23.5)
        self.assertNotEqual(s_23, [])
        self.assertNotEqual(s_23, None)

        self.assertLess(s_23, s_24)
        self.assertLess(s_23, 24)
        self.assertGreater(24, s_23)
        self.assertGreater(s_24, s_23)

    def test_lt(self):
        self.assertTrue(Asset(3) < Asset(4))
        self.assertFalse(Asset(4) < Asset(4))
        self.assertFalse(Asset(5) < Asset(4))

    def test_le(self):
        self.assertTrue(Asset(3) <= Asset(4))
        self.assertTrue(Asset(4) <= Asset(4))
        self.assertFalse(Asset(5) <= Asset(4))

    def test_eq(self):
        self.assertFalse(Asset(3) == Asset(4))
        self.assertTrue(Asset(4) == Asset(4))
        self.assertFalse(Asset(5) == Asset(4))

    def test_ge(self):
        self.assertFalse(Asset(3) >= Asset(4))
        self.assertTrue(Asset(4) >= Asset(4))
        self.assertTrue(Asset(5) >= Asset(4))

    def test_gt(self):
        self.assertFalse(Asset(3) > Asset(4))
        self.assertFalse(Asset(4) > Asset(4))
        self.assertTrue(Asset(5) > Asset(4))

    def test_type_mismatch(self):
        if sys.version_info.major < 3:
            self.assertIsNotNone(Asset(3) < 'a')
            self.assertIsNotNone('a' < Asset(3))
        else:
            with self.assertRaises(TypeError):
                Asset(3) < 'a'
            with self.assertRaises(TypeError):
                'a' < Asset(3)


class TestFuture(TestCase):
    future = Future(
        2468,
        symbol='OMH15',
        root_symbol='OM',
        notice_date=pd.Timestamp('2014-01-20', tz='UTC'),
        expiration_date=pd.Timestamp('2014-02-20', tz='UTC'),
        contract_multiplier=500
    )

    def test_str(self):
        strd = self.future.__str__()
        self.assertEqual("Future(2468 [OMH15])", strd)

    def test_repr(self):
        reprd = self.future.__repr__()
        self.assertTrue("Future" in reprd)
        self.assertTrue("2468" in reprd)
        self.assertTrue("OMH15" in reprd)
        self.assertTrue("root_symbol='OM'" in reprd)
        self.assertTrue(("notice_date=Timestamp('2014-01-20 00:00:00+0000', "
                        "tz='UTC')") in reprd)
        self.assertTrue("expiration_date=Timestamp('2014-02-20 00:00:00+0000'"
                        in reprd)
        self.assertTrue("contract_multiplier=500" in reprd)

    def test_reduce(self):
        reduced = self.future.__reduce__()
        self.assertEqual(Future, reduced[0])

    def test_to_and_from_dict(self):
        dictd = self.future.to_dict()
        self.assertTrue('root_symbol' in dictd)
        self.assertTrue('notice_date' in dictd)
        self.assertTrue('expiration_date' in dictd)
        self.assertTrue('contract_multiplier' in dictd)

        from_dict = Future.from_dict(dictd)
        self.assertTrue(isinstance(from_dict, Future))
        self.assertEqual(self.future, from_dict)

    def test_root_symbol(self):
        self.assertEqual('OM', self.future.root_symbol)


class AssetFinderTestCase(TestCase):

    def test_lookup_symbol_fuzzy(self):
        as_of = pd.Timestamp('2013-01-01', tz='UTC')
        frame = pd.DataFrame.from_records(
            [
                {
                    'sid': i,
                    'file_name':  'TEST@%d' % i,
                    'company_name': "company%d" % i,
                    'start_date_nano': as_of.value,
                    'end_date_nano': as_of.value,
                    'exchange': uuid.uuid4().hex,
                }
                for i in range(3)
            ]
        )
        finder = AssetFinder(frame, fuzzy_char='@')
        asset_0, asset_1, asset_2 = (
            finder.retrieve_asset(i) for i in range(3)
        )

        for i in range(2):  # we do it twice to test for caching bugs
            self.assertIsNone(finder.lookup_symbol('test', as_of))
            self.assertEqual(
                asset_1,
                finder.lookup_symbol('test@1', as_of)
            )

            # Adding an unnecessary fuzzy shouldn't matter.
            self.assertEqual(
                asset_1,
                finder.lookup_symbol('test@1', as_of, fuzzy=True)
            )

            # Shouldn't find this with no fuzzy_str passed.
            self.assertIsNone(finder.lookup_symbol('test1', as_of))
            # Should find exact match.
            self.assertEqual(
                asset_1,
                finder.lookup_symbol('test1', as_of, fuzzy=True),
            )

    def test_lookup_symbol_resolve_multiple(self):

        # Incrementing by two so that start and end dates for each
        # generated Asset don't overlap (each Asset's end_date is the
        # day after its start date.)
        dates = pd.date_range('2013-01-01', freq='2D', periods=5, tz='UTC')
        df = pd.DataFrame.from_records(
            [
                {
                    'sid': i,
                    'file_name':  'existing',
                    'company_name': 'existing',
                    'start_date_nano': date.value,
                    'end_date_nano': (date + timedelta(days=1)).value,
                    'exchange': 'NYSE',
                }
                for i, date in enumerate(dates)
            ]
        )

        finder = AssetFinder(df)
        for _ in range(2):  # Run checks twice to test for caching bugs.
            with self.assertRaises(SymbolNotFound):
                finder.lookup_symbol_resolve_multiple('non_existing', dates[0])

            with self.assertRaises(MultipleSymbolsFound):
                finder.lookup_symbol_resolve_multiple('existing', None)

            for i, date in enumerate(dates):
                # Verify that we correctly resolve multiple symbols using
                # the supplied date
                result = finder.lookup_symbol_resolve_multiple(
                    'existing',
                    date,
                )
                self.assertEqual(result.symbol, 'existing')
                self.assertEqual(result.sid, i)

    @parameterized.expand(
        build_lookup_generic_cases()
    )
    def test_lookup_generic(self, finder, symbols, reference_date, expected):
        """
        Ensure that lookup_generic works with various permutations of inputs.
        """
        results, missing = finder.lookup_generic(symbols, reference_date)
        self.assertEqual(results, expected)
        self.assertEqual(missing, [])

    def test_lookup_generic_handle_missing(self):
        data = pd.DataFrame.from_records(
            [
                {
                    'sid': 0,
                    'file_name': 'real',
                    'company_name': 'real',
                    'start_date_nano': pd.Timestamp('2013-1-1', tz='UTC'),
                    'end_date_nano': pd.Timestamp('2014-1-1', tz='UTC'),
                    'exchange': '',
                },
                {
                    'sid': 1,
                    'file_name': 'also_real',
                    'company_name': 'also_real',
                    'start_date_nano': pd.Timestamp('2013-1-1', tz='UTC'),
                    'end_date_nano': pd.Timestamp('2014-1-1', tz='UTC'),
                    'exchange': '',
                },
                # Sid whose end date is before our query date.  We should
                # still correctly find it.
                {
                    'sid': 2,
                    'file_name': 'real_but_old',
                    'company_name': 'real_but_old',
                    'start_date_nano': pd.Timestamp('2002-1-1', tz='UTC'),
                    'end_date_nano': pd.Timestamp('2003-1-1', tz='UTC'),
                    'exchange': '',
                },
                # Sid whose start_date is **after** our query date.  We should
                # **not** find it.
                {
                    'sid': 3,
                    'file_name': 'real_but_in_the_future',
                    'company_name': 'real_but_in_the_future',
                    'start_date_nano': pd.Timestamp('2014-1-1', tz='UTC'),
                    'end_date_nano': pd.Timestamp('2020-1-1', tz='UTC'),
                    'exchange': 'THE FUTURE',
                },
            ]
        )
        finder = AssetFinder(data)
        results, missing = finder.lookup_generic(
            ['real', 1, 'fake', 'real_but_old', 'real_but_in_the_future'],
            pd.Timestamp('2013-02-01', tz='UTC'),
        )

        self.assertEqual(len(results), 3)
        self.assertEqual(results[0].symbol, 'real')
        self.assertEqual(results[0].sid, 0)
        self.assertEqual(results[1].symbol, 'also_real')
        self.assertEqual(results[1].sid, 1)
        self.assertEqual(results[2].symbol, 'real_but_old')
        self.assertEqual(results[2].sid, 2)

        self.assertEqual(len(missing), 2)
        self.assertEqual(missing[0], 'fake')
        self.assertEqual(missing[1], 'real_but_in_the_future')

    def test_insert_metadata(self):
        finder = AssetFinder()
        finder.insert_metadata(0,
                               asset_type='equity',
                               start_date='2014-01-01',
                               end_date='2015-01-01',
                               symbol="PLAY",
                               foo_data="FOO",)

        # Test proper insertion
        equity = finder.retrieve_asset(0)
        self.assertIsInstance(equity, Equity)
        self.assertEqual('PLAY', equity.symbol)
        self.assertEqual(pd.Timestamp('2015-01-01', tz='UTC'),
                         equity.end_date)

        # Test invalid field
        with self.assertRaises(AttributeError):
            equity.foo_data

    def test_consume_metadata(self):

        # Test dict consumption
        finder = AssetFinder()
        dict_to_consume = {0: {'symbol': 'PLAY'},
                           1: {'symbol': 'MSFT'}}
        finder.consume_metadata(dict_to_consume)

        equity = finder.retrieve_asset(0)
        self.assertIsInstance(equity, Equity)
        self.assertEqual('PLAY', equity.symbol)

        finder = AssetFinder()

        # Test dataframe consumption
        df = pd.DataFrame(columns=['asset_name', 'exchange'], index=[0, 1])
        df['asset_name'][0] = "Dave'N'Busters"
        df['exchange'][0] = "NASDAQ"
        df['asset_name'][1] = "Microsoft"
        df['exchange'][1] = "NYSE"
        finder.consume_metadata(df)
        self.assertEqual('NASDAQ', finder.retrieve_asset(0).exchange)
        self.assertEqual('Microsoft', finder.retrieve_asset(1).asset_name)

    def test_consume_asset_as_identifier(self):
        # Build some end dates
        eq_end = pd.Timestamp('2012-01-01', tz='UTC')
        fut_end = pd.Timestamp('2008-01-01', tz='UTC')

        # Build some simple Assets
        equity_asset = Equity(1, symbol="TESTEQ", end_date=eq_end)
        future_asset = Future(200, symbol="TESTFUT", end_date=fut_end)

        # Consume the Assets
        finder = AssetFinder()
        finder.consume_identifiers([equity_asset, future_asset])

        # Test equality with newly built Assets
        self.assertEqual(equity_asset, finder.retrieve_asset(1))
        self.assertEqual(future_asset, finder.retrieve_asset(200))
        self.assertEqual(eq_end, finder.retrieve_asset(1).end_date)
        self.assertEqual(fut_end, finder.retrieve_asset(200).end_date)

    def test_sid_assignment(self):

        # This metadata does not contain SIDs
        metadata = {'PLAY': {'symbol': 'PLAY'},
                    'MSFT': {'symbol': 'MSFT'}}

        today = normalize_date(pd.Timestamp('2015-07-09', tz='UTC'))

        # Build a finder that is allowed to assign sids
        finder = AssetFinder(metadata=metadata,
                             allow_sid_assignment=True)

        # Verify that Assets were built and different sids were assigned
        play = finder.lookup_symbol('PLAY', today)
        msft = finder.lookup_symbol('MSFT', today)
        self.assertEqual('PLAY', play.symbol)
        self.assertIsNotNone(play.sid)
        self.assertNotEqual(play.sid, msft.sid)

    def test_sid_assignment_failure(self):

        # This metadata does not contain SIDs
        metadata = {'PLAY': {'symbol': 'PLAY'},
                    'MSFT': {'symbol': 'MSFT'}}

        # Build a finder that is not allowed to assign sids, asserting failure
        with self.assertRaises(SidAssignmentError):
            AssetFinder(metadata=metadata, allow_sid_assignment=False)

    def test_security_dates_warning(self):

        # Build an asset with an end_date
        eq_end = pd.Timestamp('2012-01-01', tz='UTC')
        equity_asset = Equity(1, symbol="TESTEQ", end_date=eq_end)

        # Catch all warnings
        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered
            warnings.simplefilter("always")
            equity_asset.security_start_date
            equity_asset.security_end_date
            equity_asset.security_name
            # Verify the warning
            self.assertEqual(3, len(w))
            for warning in w:
                self.assertTrue(issubclass(warning.category,
                                           DeprecationWarning))

    def test_lookup_future_chain(self):
        metadata = {
            # Notice day is today, so not valid
            2: {
                'symbol': 'ADN15',
                'root_symbol': 'AD',
                'asset_type': 'future',
                'notice_date': pd.Timestamp('2015-05-14', tz='UTC'),
                'start_date': pd.Timestamp('2015-01-01', tz='UTC')
            },
            1: {
                'symbol': 'ADV15',
                'root_symbol': 'AD',
                'asset_type': 'future',
                'notice_date': pd.Timestamp('2015-08-14', tz='UTC'),
                'start_date': pd.Timestamp('2015-01-01', tz='UTC')
            },
            # Starts trading today, so should be valid.
            0: {
                'symbol': 'ADF16',
                'root_symbol': 'AD',
                'asset_type': 'future',
                'notice_date': pd.Timestamp('2015-11-16', tz='UTC'),
                'start_date': pd.Timestamp('2015-05-14', tz='UTC')
            },
            # Copy of the above future, but starts trading in August,
            # so it isn't valid.
            3: {
                'symbol': 'ADF16',
                'root_symbol': 'AD',
                'asset_type': 'future',
                'notice_date': pd.Timestamp('2015-11-16', tz='UTC'),
                'start_date': pd.Timestamp('2015-08-01', tz='UTC')
            },

        }

        finder = AssetFinder(metadata=metadata)
        dt = pd.Timestamp('2015-05-14', tz='UTC')
        last_year = pd.Timestamp('2014-01-01', tz='UTC')
        first_day = pd.Timestamp('2015-01-01', tz='UTC')

        # Check that we get the expected number of contracts, in the
        # right order
        ad_contracts = finder.lookup_future_chain('AD', dt, dt)
        self.assertEqual(len(ad_contracts), 2)
        self.assertEqual(ad_contracts[0].sid, 1)
        self.assertEqual(ad_contracts[1].sid, 0)

        # Check that pd.NaT for knowledge_date uses the value of as_of_date
        ad_contracts = finder.lookup_future_chain('AD', dt, pd.NaT)
        self.assertEqual(len(ad_contracts), 2)

        # Check that we get nothing if our knowledge date is last year
        ad_contracts = finder.lookup_future_chain('AD', dt, last_year)
        self.assertEqual(len(ad_contracts), 0)

        # Check that we get things that start on the knowledge date
        ad_contracts = finder.lookup_future_chain('AD', dt, first_day)
        self.assertEqual(len(ad_contracts), 1)

        # Check that pd.NaT for as_of_date gives the whole chain
        ad_contracts = finder.lookup_future_chain('AD', pd.NaT, first_day)
        self.assertEqual(len(ad_contracts), 4)

    def test_map_identifier_index_to_sids(self):
        # Build an empty finder and some Assets
        dt = pd.Timestamp('2014-01-01', tz='UTC')
        finder = AssetFinder()
        asset1 = Equity(1, symbol="AAPL")
        asset2 = Equity(2, symbol="GOOG")
        asset200 = Future(200, symbol="CLK15")
        asset201 = Future(201, symbol="CLM15")

        # Check for correct mapping and types
        pre_map = [asset1, asset2, asset200, asset201]
        post_map = finder.map_identifier_index_to_sids(pre_map, dt)
        self.assertListEqual([1, 2, 200, 201], post_map)
        for sid in post_map:
            self.assertIsInstance(sid, int)

        # Change order and check mapping again
        pre_map = [asset201, asset2, asset200, asset1]
        post_map = finder.map_identifier_index_to_sids(pre_map, dt)
        self.assertListEqual([201, 2, 200, 1], post_map)

    @with_environment()
    def test_compute_lifetimes(self, env=None):
        num_assets = 4
        trading_day = env.trading_day
        first_start = pd.Timestamp('2015-04-01', tz='UTC')

        frame = make_rotating_asset_info(
            num_assets=num_assets,
            first_start=first_start,
            frequency=env.trading_day,
            periods_between_starts=3,
            asset_lifetime=5
        )
        finder = AssetFinder(frame)

        all_dates = pd.date_range(
            start=first_start,
            end=frame.end_date.max(),
            freq=trading_day,
        )

        for dates in all_subindices(all_dates):
            expected_mask = full(
                shape=(len(dates), num_assets),
                fill_value=False,
                dtype=bool,
            )

            for i, date in enumerate(dates):
                it = frame[['start_date', 'end_date']].itertuples()
                for j, start, end in it:
                    if start <= date <= end:
                        expected_mask[i, j] = True

            # Filter out columns with all-empty columns.
            expected_result = pd.DataFrame(
                data=expected_mask,
                index=dates,
                columns=frame.sid.values,
            )
            actual_result = finder.lifetimes(dates)
            assert_frame_equal(actual_result, expected_result)


class TestFutureChain(TestCase):
    metadata = {
        0: {
            'symbol': 'CLG06',
            'root_symbol': 'CL',
            'asset_type': 'future',
            'start_date': pd.Timestamp('2005-12-01', tz='UTC'),
            'notice_date': pd.Timestamp('2005-12-20', tz='UTC'),
            'expiration_date': pd.Timestamp('2006-01-20', tz='UTC')},
        1: {
            'root_symbol': 'CL',
            'symbol': 'CLK06',
            'asset_type': 'future',
            'start_date': pd.Timestamp('2005-12-01', tz='UTC'),
            'notice_date': pd.Timestamp('2006-03-20', tz='UTC'),
            'expiration_date': pd.Timestamp('2006-04-20', tz='UTC')},
        2: {
            'symbol': 'CLQ06',
            'root_symbol': 'CL',
            'asset_type': 'future',
            'start_date': pd.Timestamp('2005-12-01', tz='UTC'),
            'notice_date': pd.Timestamp('2006-06-20', tz='UTC'),
            'expiration_date': pd.Timestamp('2006-07-20', tz='UTC')},
        3: {
            'symbol': 'CLX06',
            'root_symbol': 'CL',
            'asset_type': 'future',
            'start_date': pd.Timestamp('2006-02-01', tz='UTC'),
            'notice_date': pd.Timestamp('2006-09-20', tz='UTC'),
            'expiration_date': pd.Timestamp('2006-10-20', tz='UTC')}
    }

    asset_finder = AssetFinder(metadata=metadata)

    def test_len(self):
        """ Test the __len__ method of FutureChain.
        """
        # None of the contracts have started yet.
        cl = FutureChain(self.asset_finder, lambda: '2005-11-30', 'CL')
        self.assertEqual(len(cl), 0)

        # Sids 0, 1, & 2 have started, 3 has not yet started.
        cl = FutureChain(self.asset_finder, lambda: '2005-12-01', 'CL')
        self.assertEqual(len(cl), 3)

        # Sid 0 is still valid the day before its notice date.
        cl = FutureChain(self.asset_finder, lambda: '2005-12-19', 'CL')
        self.assertEqual(len(cl), 3)

        # Sid 0 is now invalid, leaving only Sids 1 & 2 valid.
        cl = FutureChain(self.asset_finder, lambda: '2005-12-20', 'CL')
        self.assertEqual(len(cl), 2)

        # Sid 3 has started, so 1, 2, & 3 are now valid.
        cl = FutureChain(self.asset_finder, lambda: '2006-02-01', 'CL')
        self.assertEqual(len(cl), 3)

        # All contracts are no longer valid.
        cl = FutureChain(self.asset_finder, lambda: '2006-09-20', 'CL')
        self.assertEqual(len(cl), 0)

    def test_getitem(self):
        """ Test the __getitem__ method of FutureChain.
        """
        cl = FutureChain(self.asset_finder, lambda: '2005-12-01', 'CL')
        self.assertEqual(cl[0], 0)
        self.assertEqual(cl[1], 1)
        self.assertEqual(cl[2], 2)
        with self.assertRaises(IndexError):
            cl[3]

        cl = FutureChain(self.asset_finder, lambda: '2005-12-19', 'CL')
        self.assertEqual(cl[0], 0)

        cl = FutureChain(self.asset_finder, lambda: '2005-12-20', 'CL')
        self.assertEqual(cl[0], 1)

        cl = FutureChain(self.asset_finder, lambda: '2006-02-01', 'CL')
        self.assertEqual(cl[-1], 3)

    def test_root_symbols(self):
        """ Test that different variations on root symbols are handled
        as expected.
        """
        # Make sure this successfully gets the chain for CL.
        cl = FutureChain(self.asset_finder, lambda: '2005-12-01', 'CL')
        self.assertEqual(cl.root_symbol, 'CL')

        # These root symbols don't exist, so RootSymbolNotFound should
        # be raised immediately.
        with self.assertRaises(RootSymbolNotFound):
            FutureChain(self.asset_finder, lambda: '2005-12-01', 'CLZ')

        with self.assertRaises(RootSymbolNotFound):
            FutureChain(self.asset_finder, lambda: '2005-12-01', '')

    def test_repr(self):
        """ Test the __repr__ method of FutureChain.
        """
        cl = FutureChain(self.asset_finder, lambda: '2005-12-01', 'CL')
        cl_feb = FutureChain(self.asset_finder, lambda: '2005-12-01', 'CL',
                             as_of_date='2006-02-01')

        # The default chain should not include the as of date.
        self.assertEqual(repr(cl), "FutureChain(root_symbol='CL')")

        # An explicit as of date should show up in the repr.
        self.assertEqual(
            repr(cl_feb),
            ("FutureChain(root_symbol='CL', "
             "as_of_date='2006-02-01 00:00:00+00:00')")
        )

    def test_as_of(self):
        """ Test the as_of method of FutureChain.
        """
        cl = FutureChain(self.asset_finder, lambda: '2005-12-01', 'CL')

        # Test that the as_of_date is set correctly to the future
        feb = '2006-02-01'
        cl_feb = cl.as_of(feb)
        self.assertEqual(
            cl_feb.as_of_date,
            pd.Timestamp(feb, tz='UTC')
        )

        # Test that the as_of_date is set correctly to the past, with
        # args of str, datetime.datetime, and pd.Timestamp.
        feb_prev = '2005-02-01'
        cl_feb_prev = cl.as_of(feb_prev)
        self.assertEqual(
            cl_feb_prev.as_of_date,
            pd.Timestamp(feb_prev, tz='UTC')
        )

        feb_prev = datetime(year=2005, month=2, day=1)
        cl_feb_prev = cl.as_of(feb_prev)
        self.assertEqual(
            cl_feb_prev.as_of_date,
            pd.Timestamp(feb_prev, tz='UTC')
        )

        feb_prev = pd.Timestamp('2005-02-01')
        cl_feb_prev = cl.as_of(feb_prev)
        self.assertEqual(
            cl_feb_prev.as_of_date,
            pd.Timestamp(feb_prev, tz='UTC')
        )

        # The chain as of the current dt should always be the same as
        # the defualt chain. Tests date as str, pd.Timestamp, and
        # datetime.datetime.
        self.assertEqual(cl[0], cl.as_of('2005-12-01')[0])
        self.assertEqual(cl[0], cl.as_of(pd.Timestamp('2005-12-01'))[0])
        self.assertEqual(
            cl[0],
            cl.as_of(datetime(year=2005, month=12, day=1))[0]
        )

    def test_offset(self):
        """ Test the offset method of FutureChain.
        """
        cl = FutureChain(self.asset_finder, lambda: '2005-12-01', 'CL')

        # Test that an offset forward sets as_of_date as expected
        self.assertEqual(
            cl.offset('3 days').as_of_date,
            cl.as_of_date + pd.Timedelta(days=3)
        )

        # Test that an offset backward sets as_of_date as expected, with
        # time delta given as str, datetime.timedelta, and pd.Timedelta.
        self.assertEqual(
            cl.offset('-1000 days').as_of_date,
            cl.as_of_date + pd.Timedelta(days=-1000)
        )
        self.assertEqual(
            cl.offset(timedelta(days=-1000)).as_of_date,
            cl.as_of_date + pd.Timedelta(days=-1000)
        )
        self.assertEqual(
            cl.offset(pd.Timedelta('-1000 days')).as_of_date,
            cl.as_of_date + pd.Timedelta(days=-1000)
        )

        # An offset of zero should give the original chain.
        self.assertEqual(cl[0], cl.offset(0)[0])
        self.assertEqual(cl[0], cl.offset("0 days")[0])

        # A string that doesn't represent a time delta should raise a
        # ValueError.
        with self.assertRaises(ValueError):
            cl.offset("blah")
