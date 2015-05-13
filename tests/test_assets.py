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
    datetime,
    timedelta,
)
import pickle
import pprint
import pytz
import uuid

import pandas as pd

from nose_parameterized import parameterized

from zipline.finance.trading import with_environment
from zipline.assets import Asset, Future, AssetMetaData
from zipline.errors import (
    SymbolNotFound,
    MultipleSymbolsFound,
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


class FakeTableIdenticalSymbols(object):
    def __init__(self, name, as_of_dates):
        self.name = name
        self.as_of_dates = as_of_dates
        self.df = pd.DataFrame.from_records(
            [
                {
                    'sid': i,
                    'file_name':  self.name,
                    'company_name': self.name,
                    'start_date_nano': date.value,
                    'end_date_nano': (date + timedelta(days=1)).value,
                    'exchange': self.name,
                }
                for i, date in enumerate(self.as_of_dates)
            ]
        )

    def read(self, *args, **kwargs):
        return self.df.to_records()


class FakeTableFromRecords(object):

    def __init__(self, records):
        self.records = records
        self.df = pd.DataFrame.from_records(self.records)

    def read(self, *args, **kwargs):
        return self.df.to_records()


@with_environment()
def build_lookup_generic_cases(env=None):
    """
    Generate test cases for AssetFinder test_lookup_generic.
    """

    unique_start = pd.Timestamp('2013-01-01', tz='UTC')
    unique_end = pd.Timestamp('2014-01-01', tz='UTC')

    dupe_0_start = pd.Timestamp('2013-01-01', tz='UTC')
    dupe_0_end = dupe_0_start + timedelta(days=1)

    dupe_1_start = pd.Timestamp('2013-01-03', tz='UTC')
    dupe_1_end = dupe_1_start + timedelta(days=1)

    table = FakeTableFromRecords(
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
    env.update_asset_finder(asset_metadata=table.df)
    dupe_0, dupe_1, unique = assets = [
        env.asset_finder.retrieve_asset(i)
        for i in range(3)
    ]

    # This expansion code is run at module import time, which means we have to
    # clear the AssetFinder here or else it will interfere with the cache
    # for other tests.
    env.update_asset_finder(erase_existing=True)

    dupe_0_start = dupe_0.start_date
    dupe_1_start = dupe_1.start_date
    cases = [
        ##
        # Scalars

        # Asset object
        (table, assets[0], None, assets[0]),
        (table, assets[1], None, assets[1]),
        (table, assets[2], None, assets[2]),
        # int
        (table, 0, None, assets[0]),
        (table, 1, None, assets[1]),
        (table, 2, None, assets[2]),
        # Duplicated symbol with resolution date
        (table, 'duplicated', dupe_0_start, dupe_0),
        (table, 'duplicated', dupe_1_start, dupe_1),
        # Unique symbol, with or without resolution date.
        (table, 'unique', unique_start, unique),
        (table, 'unique', None, unique),

        ##
        # Iterables

        # Iterables of Asset objects.
        (table, assets, None, assets),
        (table, iter(assets), None, assets),
        # Iterables of ints
        (table, (0, 1), None, assets[:-1]),
        (table, iter((0, 1)), None, assets[:-1]),
        # Iterables of symbols.
        (table, ('duplicated', 'unique'), dupe_0_start, [dupe_0, unique]),
        (table, ('duplicated', 'unique'), dupe_1_start, [dupe_1, unique]),
        # Mixed types
        (table,
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
                          'asset_end_date',
                          'asset_name',
                          'asset_start_date',
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

    @with_environment()
    def test_lookup_symbol_fuzzy(self, env=None):
        fuzzy_str = '@'
        as_of_date = datetime(2013, 1, 1, tzinfo=pytz.utc)
        table = FakeTable(uuid.uuid4().hex, 2, as_of_date,
                          fuzzy_str)
        env.update_asset_finder(asset_metadata=table.df)
        sf = env.asset_finder

        try:
            for i in range(2):  # we do it twice to test for caching bugs
                self.assertIsNone(sf.lookup_symbol('test', as_of_date))
                self.assertIsNotNone(sf.lookup_symbol(
                    'test%s%s' % (fuzzy_str, 1), as_of_date))
                self.assertIsNone(sf.lookup_symbol('test%s' % 1, as_of_date))

                self.assertIsNone(sf.lookup_symbol(table.name, as_of_date,
                                                   fuzzy=fuzzy_str))
                self.assertIsNotNone(sf.lookup_symbol(
                    'test%s%s' % (fuzzy_str, 1), as_of_date, fuzzy=fuzzy_str))
                self.assertIsNotNone(sf.lookup_symbol(
                    'test%s' % 1, as_of_date, fuzzy=fuzzy_str))
        finally:
            env.update_asset_finder(erase_existing=True)

    @with_environment()
    def test_lookup_symbol_resolve_multiple(self, env=None):

        as_of_dates = [
            pd.Timestamp('2013-01-01', tz='UTC') + timedelta(days=i)
            # Incrementing by two so that start and end dates for each
            # generated Asset don't overlap (each Asset's end_date is the
            # day after its start date.)
            for i in range(0, 10, 2)
        ]

        table = FakeTableIdenticalSymbols(
            name='existing',
            as_of_dates=as_of_dates,
        )
        env.update_asset_finder(asset_metadata=table.df)
        sf = env.asset_finder

        try:
            for _ in range(2):  # we do it twice to test for caching bugs
                with self.assertRaises(SymbolNotFound):
                    sf.lookup_symbol_resolve_multiple('non_existing',
                                                      as_of_dates[0])
                with self.assertRaises(MultipleSymbolsFound):
                    sf.lookup_symbol_resolve_multiple('existing',
                                                      None)

                for i, date in enumerate(as_of_dates):
                    # Verify that we correctly resolve multiple symbols using
                    # the supplied date
                    result = sf.lookup_symbol_resolve_multiple(
                        'existing',
                        date,
                    )
                    self.assertEqual(result.symbol, 'existing')
                    self.assertEqual(result.sid, i)

        finally:
            env.update_asset_finder(erase_existing=True)

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
            env.update_asset_finder(erase_existing=True)

    @parameterized.expand(
        build_lookup_generic_cases()
    )
    @with_environment()
    def test_lookup_generic(self, table, symbols, reference_date, expected,
                            env=None):
        """
        Ensure that lookup_generic works with various permutations of inputs.
        """
        try:
            env.update_asset_finder(asset_metadata=table.df)
            finder = env.asset_finder
            results, missing = finder.lookup_generic(symbols, reference_date)
            self.assertEqual(results, expected)
            self.assertEqual(missing, [])
        finally:
            env.update_asset_finder(erase_existing=True)

    @with_environment()
    def test_lookup_generic_handle_missing(self, env=None):
        try:
            table = FakeTableFromRecords(
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
            env.update_asset_finder(asset_metadata=table.df)
            symbols = [
                'real', 1, 'fake', 'real_but_old', 'real_but_in_the_future',
            ]

            results, missing = env.asset_finder.lookup_generic(
                symbols,
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

        finally:
            env.update_asset_finder(erase_existing=True)


class TestAssetMetaData(TestCase):

    def test_insert_metadata(self):
        amd = AssetMetaData()
        amd.insert_metadata(0,
                            asset_type='equity',
                            start_date='2014-01-01',
                            end_date='2015-01-01',
                            symbol="PLAY",
                            foo_data="FOO"
                            )

        # Test proper insertion
        self.assertEqual('equity', amd.retrieve_metadata(0)['asset_type'])
        self.assertEqual('PLAY', amd.retrieve_metadata(0)['symbol'])
        self.assertEqual('2015-01-01', amd.retrieve_metadata(0)['end_date'])

        # Test invalid field
        self.assertFalse('foo_data' in amd.retrieve_metadata(0))

        # Test updating fields
        amd.insert_metadata(0,
                            asset_type='equity',
                            start_date='2014-01-01',
                            end_date='2015-02-01',
                            symbol="PLAY",
                            exchange="NYSE"
                            )
        self.assertEqual('2015-02-01', amd.retrieve_metadata(0)['end_date'])
        self.assertEqual('NYSE', amd.retrieve_metadata(0)['exchange'])

        # Check that old data survived
        self.assertEqual('PLAY', amd.retrieve_metadata(0)['symbol'])

    def test_consume_metadata(self):

        # Test dict consumption
        amd = AssetMetaData({0: {'asset_type': 'equity'}})
        dict_to_consume = {0: {'symbol': 'PLAY'},
                           1: {'symbol': 'MSFT'}}
        amd.consume_metadata(dict_to_consume)
        self.assertEqual('equity', amd.retrieve_metadata(0)['asset_type'])
        self.assertEqual('PLAY', amd.retrieve_metadata(0)['symbol'])

        # Test dataframe consumption
        df = pd.DataFrame(columns=['asset_name', 'exchange'], index=[0, 1])
        df['asset_name'][0] = "Dave'N'Busters"
        df['exchange'][0] = "NASDAQ"
        df['asset_name'][1] = "Microsoft"
        df['exchange'][1] = "NYSE"
        amd.consume_metadata(df)
        self.assertEqual('NASDAQ', amd.retrieve_metadata(0)['exchange'])
        self.assertEqual('Microsoft', amd.retrieve_metadata(1)['asset_name'])
        # Check that old data survived
        self.assertEqual('equity', amd.retrieve_metadata(0)['asset_type'])

        # Test AssetMetaData consumption
        amd2 = AssetMetaData({2: {'symbol': 'AAPL'}})
        amd.consume_metadata(amd2)
        self.assertEqual('AAPL', amd.retrieve_metadata(2)['symbol'])
        # Check that old data survived
        self.assertEqual('equity', amd.retrieve_metadata(0)['asset_type'])
