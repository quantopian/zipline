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

from datetime import (
    timedelta,
)
import pickle
import pprint
import uuid
import warnings
import pandas as pd

from nose_parameterized import parameterized

from zipline.finance.trading import with_environment
from zipline.assets import Asset, Equity, Future, AssetFinder
from zipline.errors import (
    SymbolNotFound,
    MultipleSymbolsFound,
    SidAssignmentError,
)


class FakeTable(object):
    def __init__(self, name, count, dt, fuzzy_str):
        self.name = name
        self.count = count
        self.dt = dt
        self.fuzzy_str = fuzzy_str
        self.df = pd.DataFrame.from_records(
            [
                {
                    'sid': i,
                    'file_name':  'TEST%s%s' % (self.fuzzy_str, i),
                    'company_name': self.name + str(i),
                    'start_date_nano': pd.Timestamp(dt, tz='UTC').value,
                    'end_date_nano': pd.Timestamp(dt, tz='UTC').value,
                    'exchange': self.name,
                }
                for i in range(1, self.count + 1)
            ]
        )

    def read(self, *args, **kwargs):
        return self.df.to_records()


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
                'file_name':  'duplicated',
                'company_name': 'duplicated_0',
                'start_date_nano': dupe_0_start.value,
                'end_date_nano': dupe_0_end.value,
                'exchange': '',
            },
            {
                'sid': 1,
                'file_name':  'duplicated',
                'company_name': 'duplicated_1',
                'start_date_nano': dupe_1_start.value,
                'end_date_nano': dupe_1_end.value,
                'exchange': '',
            },
            {
                'sid': 2,
                'file_name':  'unique',
                'company_name': 'unique',
                'start_date_nano': unique_start.value,
                'end_date_nano': unique_end.value,
                'exchange': '',
            },
        ],
    )
    finder = AssetFinder(metadata=frame)
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

    future = Future(2468,
                    symbol='OMK15',
                    notice_date='2014-01-20',
                    expiration_date='2014-02-20',
                    contract_multiplier=500)

    def test_str(self):
        strd = self.future.__str__()
        self.assertEqual("Future(2468 [OMK15])", strd)

    def test_repr(self):
        reprd = self.future.__repr__()
        self.assertTrue("Future" in reprd)
        self.assertTrue("2468" in reprd)
        self.assertTrue("OMK15" in reprd)
        self.assertTrue("notice_date='2014-01-20'" in reprd)
        self.assertTrue("expiration_date='2014-02-20'" in reprd)
        self.assertTrue("contract_multiplier=500" in reprd)

    def test_reduce(self):
        reduced = self.future.__reduce__()
        self.assertEqual(Future, reduced[0])

    def test_to_and_from_dict(self):
        dictd = self.future.to_dict()
        self.assertTrue('notice_date' in dictd)
        self.assertTrue('expiration_date' in dictd)
        self.assertTrue('contract_multiplier' in dictd)

        from_dict = Future.from_dict(dictd)
        self.assertTrue(isinstance(from_dict, Future))
        self.assertEqual(self.future, from_dict)


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
        finder = AssetFinder(frame)
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
                finder.lookup_symbol('test@1', as_of, fuzzy='@')
            )

            # Shouldn't find this with no fuzzy_str passed.
            self.assertIsNone(finder.lookup_symbol('test1', as_of))
            # Shouldn't find this with an incorrect fuzzy_str.
            self.assertIsNone(finder.lookup_symbol('test1', as_of, fuzzy='*'))
            # Should find it with the correct fuzzy_str.
            self.assertEqual(
                asset_1,
                finder.lookup_symbol('test1', as_of, fuzzy='@'),
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

    @with_environment()
    def test_lookup_symbol_nasdaq_underscore_collisions(self, env=None):
        """
        Ensure that each NASDAQ symbol without underscores maps back to the
        original symbol when using fuzzy matching.
        """
        sf = env.asset_finder
        fuzzy_str = '_'
        collisions = []

        try:
            for sid in sf.sids:
                sec = sf.retrieve_asset(sid)
                if sec.exchange.startswith('NASDAQ'):
                    found = sf.lookup_symbol(sec.symbol.replace(fuzzy_str, ''),
                                             sec.end_date, fuzzy=fuzzy_str)
                    if found != sec:
                        collisions.append((found, sec))

            # KNOWN BUG: Filter out assets that have intersections in their
            # start and end dates.  We can't correctly resolve these.
            unexpected_errors = []
            for first, second in collisions:
                overlapping_dates = (
                    first.end_date >= second.start_date or
                    second.end_date >= first.end_date
                )
                if not overlapping_dates:
                    unexpected_errors.append((first, second))

            self.assertFalse(
                unexpected_errors,
                pprint.pformat(unexpected_errors),
            )
        finally:
            env.update_asset_finder(clear_metadata=True)

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
                # Sids that will be found when we do lookups.
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
                # Sid whose end date is before our query date.  We should
                # still correctly find it.
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
        self.assertEqual('equity', finder.metadata_cache[0]['asset_type'])
        self.assertEqual('PLAY', finder.metadata_cache[0]['symbol'])
        self.assertEqual('2015-01-01', finder.metadata_cache[0]['end_date'])

        # Test invalid field
        self.assertFalse('foo_data' in finder.metadata_cache[0])

        # Test updating fields
        finder.insert_metadata(0,
                               asset_type='equity',
                               start_date='2014-01-01',
                               end_date='2015-02-01',
                               symbol="PLAY",
                               exchange="NYSE",)
        self.assertEqual('2015-02-01', finder.metadata_cache[0]['end_date'])
        self.assertEqual('NYSE', finder.metadata_cache[0]['exchange'])

        # Check that old data survived
        self.assertEqual('PLAY', finder.metadata_cache[0]['symbol'])

    def test_consume_metadata(self):

        # Test dict consumption
        finder = AssetFinder({0: {'asset_type': 'equity'}})
        dict_to_consume = {0: {'symbol': 'PLAY'},
                           1: {'symbol': 'MSFT'}}
        finder.consume_metadata(dict_to_consume)
        self.assertEqual('equity', finder.metadata_cache[0]['asset_type'])
        self.assertEqual('PLAY', finder.metadata_cache[0]['symbol'])

        # Test dataframe consumption
        df = pd.DataFrame(columns=['asset_name', 'exchange'], index=[0, 1])
        df['asset_name'][0] = "Dave'N'Busters"
        df['exchange'][0] = "NASDAQ"
        df['asset_name'][1] = "Microsoft"
        df['exchange'][1] = "NYSE"
        finder.consume_metadata(df)
        self.assertEqual('NASDAQ', finder.metadata_cache[0]['exchange'])
        self.assertEqual('Microsoft', finder.metadata_cache[1]['asset_name'])
        # Check that old data survived
        self.assertEqual('equity', finder.metadata_cache[0]['asset_type'])

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
        finder.populate_cache()

        # Test equality with newly built Assets
        self.assertEqual(equity_asset, finder.retrieve_asset(1))
        self.assertEqual(future_asset, finder.retrieve_asset(200))
        self.assertEqual(eq_end, finder.retrieve_asset(1).end_date)
        self.assertEqual(fut_end, finder.retrieve_asset(200).end_date)

    def test_sid_assignment(self):

        # This metadata does not contain SIDs
        metadata = {'PLAY': {'symbol': 'PLAY'},
                    'MSFT': {'symbol': 'MSFT'}}

        # Build a finder that is allowed to assign sids
        finder = AssetFinder(metadata=metadata, allow_sid_assignment=True)

        # Verify that Assets were built
        play = finder.retrieve_asset_by_identifier('PLAY')
        self.assertEqual('PLAY', play.symbol)

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
