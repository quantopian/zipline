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
from collections import namedtuple
from datetime import timedelta
from functools import partial
import os
import pickle
import string
import sys
from types import GetSetDescriptorType
from unittest import TestCase
import uuid
import warnings

from nose_parameterized import parameterized
import numpy as np
from numpy import full, int32, int64
import pandas as pd
from six import viewkeys
import sqlalchemy as sa

from zipline.assets import (
    Asset,
    ExchangeInfo,
    Equity,
    Future,
    AssetDBWriter,
    AssetFinder,
)
from zipline.assets.assets import OwnershipPeriod
from zipline.assets.synthetic import (
    make_commodity_future_info,
    make_rotating_equity_info,
    make_simple_equity_info,
)
from six import itervalues, integer_types
from toolz import valmap, concat

from zipline.assets.asset_writer import (
    check_version_info,
    write_version_info,
    _futures_defaults,
    SQLITE_MAX_VARIABLE_NUMBER,
)
from zipline.assets.asset_db_schema import ASSET_DB_VERSION
from zipline.assets.asset_db_migrations import (
    downgrade
)
from zipline.errors import (
    EquitiesNotFound,
    FutureContractsNotFound,
    MultipleSymbolsFound,
    MultipleValuesFoundForField,
    MultipleValuesFoundForSid,
    NoValueForSid,
    AssetDBVersionError,
    SidsNotFound,
    SymbolNotFound,
    AssetDBImpossibleDowngrade,
    ValueNotFoundForField,
)
from zipline.testing import (
    all_subindices,
    empty_assets_db,
    parameter_space,
    powerset,
    tmp_assets_db,
    tmp_asset_finder,
)
from zipline.testing.predicates import assert_equal, assert_not_equal
from zipline.testing.fixtures import (
    WithAssetFinder,
    ZiplineTestCase,
    WithTradingCalendars,
    WithTmpDir,
    WithInstanceTmpDir,
)
from zipline.utils.range import range


Case = namedtuple('Case', 'finder inputs as_of country_code expected')

minute = pd.Timedelta(minutes=1)


def build_lookup_generic_cases():
    """
    Generate test cases for the type of asset finder specific by
    asset_finder_type for test_lookup_generic.
    """
    unique_start = pd.Timestamp('2013-01-01', tz='UTC')
    unique_end = pd.Timestamp('2014-01-01', tz='UTC')

    dupe_old_start = pd.Timestamp('2013-01-01', tz='UTC')
    dupe_old_end = pd.Timestamp('2013-01-02', tz='UTC')
    dupe_new_start = pd.Timestamp('2013-01-03', tz='UTC')
    dupe_new_end = pd.Timestamp('2013-01-03', tz='UTC')

    equities = pd.DataFrame.from_records(
        [
            # These symbols are duplicated within the US, but have different
            # lifetimes.
            {
                'sid': 0,
                'symbol': 'duplicated_in_us',
                'start_date': dupe_old_start.value,
                'end_date': dupe_old_end.value,
                'exchange': 'US_EXCHANGE',
            },
            {
                'sid': 1,
                'symbol': 'duplicated_in_us',
                'start_date': dupe_new_start.value,
                'end_date': dupe_new_end.value,
                'exchange': 'US_EXCHANGE',
            },
            # This asset is unique.
            {
                'sid': 2,
                'symbol': 'unique',
                'start_date': unique_start.value,
                'end_date': unique_end.value,
                'exchange': 'US_EXCHANGE',
            },
            # These assets appear with the same ticker at the same time in
            # different countries.
            {
                'sid': 3,
                'symbol': 'duplicated_globally',
                'start_date': unique_start.value,
                'end_date': unique_start.value,
                'exchange': 'US_EXCHANGE',
            },
            {
                'sid': 4,
                'symbol': 'duplicated_globally',
                'start_date': unique_start.value,
                'end_date': unique_start.value,
                'exchange': 'CA_EXCHANGE',
            },
        ],
        index='sid'
    )

    fof14_sid = 10000

    futures = pd.DataFrame.from_records(
        [
            {
                'sid': fof14_sid,
                'symbol': 'FOF14',
                'root_symbol': 'FO',
                'start_date': unique_start.value,
                'end_date': unique_end.value,
                'auto_close_date': unique_end.value,
                'exchange': 'US_FUT',
            },
        ],
        index='sid'
    )

    root_symbols = pd.DataFrame({
        'root_symbol': ['FO'],
        'root_symbol_id': [1],
        'exchange': ['US_FUT'],
    })

    exchanges = pd.DataFrame.from_records([
        {'exchange': 'US_EXCHANGE', 'country_code': 'US'},
        {'exchange': 'CA_EXCHANGE', 'country_code': 'CA'},
        {'exchange': 'US_FUT', 'country_code': 'US'},
    ])

    temp_db = tmp_assets_db(
        equities=equities,
        futures=futures,
        root_symbols=root_symbols,
        exchanges=exchanges,
    )

    with temp_db as assets_db:
        finder = AssetFinder(assets_db)

        case = partial(Case, finder)

        equities = finder.retrieve_all(range(5))
        dupe_old, dupe_new, unique, dupe_us, dupe_ca = equities

        fof14 = finder.retrieve_asset(fof14_sid)
        cf = finder.create_continuous_future(
            root_symbol=fof14.root_symbol,
            offset=0,
            roll_style='volume',
            adjustment=None,
        )

        all_assets = list(equities) + [fof14, cf]

        for asset in list(equities) + [fof14, cf]:
            # Looking up an asset object directly should yield itself.
            yield case(asset, None, None, asset)
            # Looking up an asset by sid should yield the asset.
            yield case(asset.sid, None, None, asset)

        # Duplicated US equity symbol with resolution date.
        for country in ('US', None):
            # On or before dupe_new_start, we should get dupe_old.
            yield case('DUPLICATED_IN_US', dupe_old_start, country, dupe_old)
            yield case(
                'DUPLICATED_IN_US', dupe_new_start - minute, country, dupe_old,
            )
            # After that, we should get dupe_new.
            yield case('DUPLICATED_IN_US', dupe_new_start, country, dupe_new)
            yield case(
                'DUPLICATED_IN_US', dupe_new_start + minute, country, dupe_new,
            )

        # Unique symbol, disambiguated by country, with or without resolution
        # date.
        for asset, country in ((dupe_us, 'US'),
                               (dupe_ca, 'CA')):
            yield case('DUPLICATED_GLOBALLY', unique_start, country, asset)
            yield case('DUPLICATED_GLOBALLY', None, country, asset)

        # Future symbols should be unique, but including as_of date
        # make sure that code path is exercised.
        yield case('FOF14', None, None, fof14)
        yield case('FOF14', unique_start, None, fof14)

        ##
        # Iterables
        # Iterables of Asset objects.
        yield case(all_assets, None, None, all_assets)
        yield case(iter(all_assets), None, None, all_assets)

        # Iterables of ints
        yield case((0, 1), None, None, equities[:2])
        yield case(iter((0, 1)), None, None, equities[:2])

        # Iterables of symbols.
        yield case(
            inputs=('DUPLICATED_IN_US', 'UNIQUE', 'DUPLICATED_GLOBALLY'),
            as_of=dupe_old_start,
            country_code='US',
            expected=[dupe_old, unique, dupe_us],
        )
        yield case(
            inputs=['DUPLICATED_GLOBALLY'],
            as_of=dupe_new_start,
            country_code='CA',
            expected=[dupe_ca],
        )

        # Mixed types
        yield case(
            inputs=(
                'DUPLICATED_IN_US',     # dupe_old b/c of as_of
                dupe_new,               # dupe_new
                2,                      # unique
                'UNIQUE',               # unique
                'DUPLICATED_GLOBALLY',  # dupe_us b/c of country_code
                dupe_ca,                # dupe_ca
            ),
            as_of=dupe_old_start,
            country_code='US',
            expected=[dupe_old, dupe_new, unique, unique, dupe_us, dupe_ca],
        )

        # Futures and Equities
        yield case(['FOF14', 0], None, None, [fof14, equities[0]])
        yield case(
            inputs=['FOF14', 'DUPLICATED_IN_US', 'DUPLICATED_GLOBALLY'],
            as_of=dupe_new_start,
            country_code='US',
            expected=[fof14, dupe_new, dupe_us],
        )

        # ContinuousFuture and Equity
        yield case([cf, 0], None, None, [cf, equities[0]])
        yield case(
            [cf, 'DUPLICATED_IN_US', 'DUPLICATED_GLOBALLY'],
            as_of=dupe_new_start,
            country_code='US',
            expected=[cf, dupe_new, dupe_us],
        )


class AssetTestCase(TestCase):

    # Dynamically list the Asset properties we want to test.
    asset_attrs = [name for name, value in vars(Asset).items()
                   if isinstance(value, GetSetDescriptorType)]

    # Very wow
    asset = Asset(
        1337,
        symbol="DOGE",
        asset_name="DOGECOIN",
        start_date=pd.Timestamp('2013-12-08 9:31', tz='UTC'),
        end_date=pd.Timestamp('2014-06-25 11:21', tz='UTC'),
        first_traded=pd.Timestamp('2013-12-08 9:31', tz='UTC'),
        auto_close_date=pd.Timestamp('2014-06-26 11:21', tz='UTC'),
        exchange_info=ExchangeInfo('THE MOON', 'MOON', '??'),
    )

    test_exchange = ExchangeInfo('test full', 'test', '??')
    asset3 = Asset(3, exchange_info=test_exchange)
    asset4 = Asset(4, exchange_info=test_exchange)
    asset5 = Asset(
        5,
        exchange_info=ExchangeInfo('still testing', 'still testing', '??'),
    )

    def test_asset_object(self):
        the_asset = Asset(
            5061,
            exchange_info=ExchangeInfo('bar', 'bar', '??'),
        )

        self.assertEquals({5061: 'foo'}[the_asset], 'foo')
        self.assertEquals(the_asset, 5061)
        self.assertEquals(5061, the_asset)

        self.assertEquals(the_asset, the_asset)
        self.assertEquals(int(the_asset), 5061)

        self.assertEquals(str(the_asset), 'Asset(5061)')

    def test_to_and_from_dict(self):
        asset_from_dict = Asset.from_dict(self.asset.to_dict())
        for attr in self.asset_attrs:
            self.assertEqual(
                getattr(self.asset, attr), getattr(asset_from_dict, attr),
            )

    def test_asset_is_pickleable(self):
        asset_unpickled = pickle.loads(pickle.dumps(self.asset))
        for attr in self.asset_attrs:
            self.assertEqual(
                getattr(self.asset, attr), getattr(asset_unpickled, attr),
            )

    def test_asset_comparisons(self):

        s_23 = Asset(23, exchange_info=self.test_exchange)
        s_24 = Asset(24, exchange_info=self.test_exchange)

        self.assertEqual(s_23, s_23)
        self.assertEqual(s_23, 23)
        self.assertEqual(23, s_23)
        self.assertEqual(int32(23), s_23)
        self.assertEqual(int64(23), s_23)
        self.assertEqual(s_23, int32(23))
        self.assertEqual(s_23, int64(23))
        # Check all int types (includes long on py2):
        for int_type in integer_types:
            self.assertEqual(int_type(23), s_23)
            self.assertEqual(s_23, int_type(23))

        self.assertNotEqual(s_23, s_24)
        self.assertNotEqual(s_23, 24)
        self.assertNotEqual(s_23, "23")
        self.assertNotEqual(s_23, 23.5)
        self.assertNotEqual(s_23, [])
        self.assertNotEqual(s_23, None)
        # Compare to a value that doesn't fit into a platform int:
        self.assertNotEqual(s_23, sys.maxsize + 1)

        self.assertLess(s_23, s_24)
        self.assertLess(s_23, 24)
        self.assertGreater(24, s_23)
        self.assertGreater(s_24, s_23)

    def test_lt(self):
        self.assertTrue(self.asset3 < self.asset4)
        self.assertFalse(self.asset4 < self.asset4)
        self.assertFalse(self.asset5 < self.asset4)

    def test_le(self):
        self.assertTrue(self.asset3 <= self.asset4)
        self.assertTrue(self.asset4 <= self.asset4)
        self.assertFalse(self.asset5 <= self.asset4)

    def test_eq(self):
        self.assertFalse(self.asset3 == self.asset4)
        self.assertTrue(self.asset4 == self.asset4)
        self.assertFalse(self.asset5 == self.asset4)

    def test_ge(self):
        self.assertFalse(self.asset3 >= self.asset4)
        self.assertTrue(self.asset4 >= self.asset4)
        self.assertTrue(self.asset5 >= self.asset4)

    def test_gt(self):
        self.assertFalse(self.asset3 > self.asset4)
        self.assertFalse(self.asset4 > self.asset4)
        self.assertTrue(self.asset5 > self.asset4)

    def test_type_mismatch(self):
        if sys.version_info.major < 3:
            self.assertIsNotNone(self.asset3 < 'a')
            self.assertIsNotNone('a' < self.asset3)
        else:
            with self.assertRaises(TypeError):
                self.asset3 < 'a'
            with self.assertRaises(TypeError):
                'a' < self.asset3


class TestFuture(WithAssetFinder, ZiplineTestCase):
    @classmethod
    def make_futures_info(cls):
        return pd.DataFrame.from_dict(
            {
                2468: {
                    'symbol': 'OMH15',
                    'root_symbol': 'OM',
                    'notice_date': pd.Timestamp('2014-01-20', tz='UTC'),
                    'expiration_date': pd.Timestamp('2014-02-20', tz='UTC'),
                    'auto_close_date': pd.Timestamp('2014-01-18', tz='UTC'),
                    'tick_size': .01,
                    'multiplier': 500.0,
                    'exchange': "TEST",
                },
                0: {
                    'symbol': 'CLG06',
                    'root_symbol': 'CL',
                    'start_date': pd.Timestamp('2005-12-01', tz='UTC'),
                    'notice_date': pd.Timestamp('2005-12-20', tz='UTC'),
                    'expiration_date': pd.Timestamp('2006-01-20', tz='UTC'),
                    'multiplier': 1.0,
                    'exchange': 'TEST',
                },
            },
            orient='index',
        )

    @classmethod
    def init_class_fixtures(cls):
        super(TestFuture, cls).init_class_fixtures()
        cls.future = cls.asset_finder.lookup_future_symbol('OMH15')
        cls.future2 = cls.asset_finder.lookup_future_symbol('CLG06')

    def test_repr(self):
        reprd = repr(self.future)
        self.assertEqual("Future(2468 [OMH15])", reprd)

    def test_reduce(self):
        assert_equal(
            pickle.loads(pickle.dumps(self.future)).to_dict(),
            self.future.to_dict(),
        )

    def test_to_and_from_dict(self):
        dictd = self.future.to_dict()
        for field in _futures_defaults.keys():
            self.assertTrue(field in dictd)

        from_dict = Future.from_dict(dictd)
        self.assertTrue(isinstance(from_dict, Future))
        self.assertEqual(self.future, from_dict)

    def test_root_symbol(self):
        self.assertEqual('OM', self.future.root_symbol)

    def test_lookup_future_symbol(self):
        """
        Test the lookup_future_symbol method.
        """
        om = TestFuture.asset_finder.lookup_future_symbol('OMH15')
        self.assertEqual(om.sid, 2468)
        self.assertEqual(om.symbol, 'OMH15')
        self.assertEqual(om.root_symbol, 'OM')
        self.assertEqual(om.notice_date, pd.Timestamp('2014-01-20', tz='UTC'))
        self.assertEqual(om.expiration_date,
                         pd.Timestamp('2014-02-20', tz='UTC'))
        self.assertEqual(om.auto_close_date,
                         pd.Timestamp('2014-01-18', tz='UTC'))

        cl = TestFuture.asset_finder.lookup_future_symbol('CLG06')
        self.assertEqual(cl.sid, 0)
        self.assertEqual(cl.symbol, 'CLG06')
        self.assertEqual(cl.root_symbol, 'CL')
        self.assertEqual(cl.start_date, pd.Timestamp('2005-12-01', tz='UTC'))
        self.assertEqual(cl.notice_date, pd.Timestamp('2005-12-20', tz='UTC'))
        self.assertEqual(cl.expiration_date,
                         pd.Timestamp('2006-01-20', tz='UTC'))

        with self.assertRaises(SymbolNotFound):
            TestFuture.asset_finder.lookup_future_symbol('')

        with self.assertRaises(SymbolNotFound):
            TestFuture.asset_finder.lookup_future_symbol('#&?!')

        with self.assertRaises(SymbolNotFound):
            TestFuture.asset_finder.lookup_future_symbol('FOOBAR')

        with self.assertRaises(SymbolNotFound):
            TestFuture.asset_finder.lookup_future_symbol('XXX99')


class AssetFinderTestCase(WithTradingCalendars, ZiplineTestCase):
    asset_finder_type = AssetFinder

    def write_assets(self, **kwargs):
        self._asset_writer.write(**kwargs)

    def init_instance_fixtures(self):
        super(AssetFinderTestCase, self).init_instance_fixtures()

        conn = self.enter_instance_context(empty_assets_db())
        self._asset_writer = AssetDBWriter(conn)
        self.asset_finder = self.asset_finder_type(conn)

    def test_blocked_lookup_symbol_query(self):
        # we will try to query for more variables than sqlite supports
        # to make sure we are properly chunking on the client side
        as_of = pd.Timestamp('2013-01-01', tz='UTC')
        # we need more sids than we can query from sqlite
        nsids = SQLITE_MAX_VARIABLE_NUMBER + 10
        sids = range(nsids)
        frame = pd.DataFrame.from_records(
            [
                {
                    'sid': sid,
                    'symbol':  'TEST.%d' % sid,
                    'start_date': as_of.value,
                    'end_date': as_of.value,
                    'exchange': uuid.uuid4().hex
                }
                for sid in sids
            ]
        )
        self.write_assets(equities=frame)
        assets = self.asset_finder.retrieve_equities(sids)
        assert_equal(viewkeys(assets), set(sids))

    def test_lookup_symbol_delimited(self):
        as_of = pd.Timestamp('2013-01-01', tz='UTC')
        frame = pd.DataFrame.from_records(
            [
                {
                    'sid': i,
                    'symbol':  'TEST.%d' % i,
                    'company_name': "company%d" % i,
                    'start_date': as_of.value,
                    'end_date': as_of.value,
                    'exchange': uuid.uuid4().hex
                }
                for i in range(3)
            ]
        )
        self.write_assets(equities=frame)
        finder = self.asset_finder
        asset_0, asset_1, asset_2 = (
            finder.retrieve_asset(i) for i in range(3)
        )

        # we do it twice to catch caching bugs
        for i in range(2):
            with self.assertRaises(SymbolNotFound):
                finder.lookup_symbol('TEST', as_of)
            with self.assertRaises(SymbolNotFound):
                finder.lookup_symbol('TEST1', as_of)
            # '@' is not a supported delimiter
            with self.assertRaises(SymbolNotFound):
                finder.lookup_symbol('TEST@1', as_of)

            # Adding an unnecessary fuzzy shouldn't matter.
            for fuzzy_char in ['-', '/', '_', '.']:
                self.assertEqual(
                    asset_1,
                    finder.lookup_symbol('TEST%s1' % fuzzy_char, as_of)
                )

    def test_lookup_symbol_fuzzy(self):
        metadata = pd.DataFrame.from_records([
            {'symbol': 'PRTY_HRD', 'exchange': "TEST"},
            {'symbol': 'BRKA', 'exchange': "TEST"},
            {'symbol': 'BRK_A', 'exchange': "TEST"},
        ])
        self.write_assets(equities=metadata)
        finder = self.asset_finder
        dt = pd.Timestamp('2013-01-01', tz='UTC')

        # Try combos of looking up PRTYHRD with and without a time or fuzzy
        # Both non-fuzzys get no result
        with self.assertRaises(SymbolNotFound):
            finder.lookup_symbol('PRTYHRD', None)
        with self.assertRaises(SymbolNotFound):
            finder.lookup_symbol('PRTYHRD', dt)
        # Both fuzzys work
        self.assertEqual(0, finder.lookup_symbol('PRTYHRD', None, fuzzy=True))
        self.assertEqual(0, finder.lookup_symbol('PRTYHRD', dt, fuzzy=True))

        # Try combos of looking up PRTY_HRD, all returning sid 0
        self.assertEqual(0, finder.lookup_symbol('PRTY_HRD', None))
        self.assertEqual(0, finder.lookup_symbol('PRTY_HRD', dt))
        self.assertEqual(0, finder.lookup_symbol('PRTY_HRD', None, fuzzy=True))
        self.assertEqual(0, finder.lookup_symbol('PRTY_HRD', dt, fuzzy=True))

        # Try combos of looking up BRKA, all returning sid 1
        self.assertEqual(1, finder.lookup_symbol('BRKA', None))
        self.assertEqual(1, finder.lookup_symbol('BRKA', dt))
        self.assertEqual(1, finder.lookup_symbol('BRKA', None, fuzzy=True))
        self.assertEqual(1, finder.lookup_symbol('BRKA', dt, fuzzy=True))

        # Try combos of looking up BRK_A, all returning sid 2
        self.assertEqual(2, finder.lookup_symbol('BRK_A', None))
        self.assertEqual(2, finder.lookup_symbol('BRK_A', dt))
        self.assertEqual(2, finder.lookup_symbol('BRK_A', None, fuzzy=True))
        self.assertEqual(2, finder.lookup_symbol('BRK_A', dt, fuzzy=True))

    def test_lookup_symbol_change_ticker(self):
        T = partial(pd.Timestamp, tz='utc')
        metadata = pd.DataFrame.from_records(
            [
                # sid 0
                {
                    'symbol': 'A',
                    'asset_name': 'Asset A',
                    'start_date': T('2014-01-01'),
                    'end_date': T('2014-01-05'),
                    'exchange': "TEST",
                },
                {
                    'symbol': 'B',
                    'asset_name': 'Asset B',
                    'start_date': T('2014-01-06'),
                    'end_date': T('2014-01-10'),
                    'exchange': "TEST",
                },

                # sid 1
                {
                    'symbol': 'C',
                    'asset_name': 'Asset C',
                    'start_date': T('2014-01-01'),
                    'end_date': T('2014-01-05'),
                    'exchange': "TEST",
                },
                {
                    'symbol': 'A',  # claiming the unused symbol 'A'
                    'asset_name': 'Asset A',
                    'start_date': T('2014-01-06'),
                    'end_date': T('2014-01-10'),
                    'exchange': "TEST",
                },
            ],
            index=[0, 0, 1, 1],
        )
        self.write_assets(equities=metadata)
        finder = self.asset_finder

        # note: these assertions walk forward in time, starting at assertions
        # about ownership before the start_date and ending with assertions
        # after the end_date; new assertions should be inserted in the correct
        # locations

        # no one held 'A' before 01
        with self.assertRaises(SymbolNotFound):
            finder.lookup_symbol('A', T('2013-12-31'))

        # no one held 'C' before 01
        with self.assertRaises(SymbolNotFound):
            finder.lookup_symbol('C', T('2013-12-31'))

        for asof in pd.date_range('2014-01-01', '2014-01-05', tz='utc'):
            # from 01 through 05 sid 0 held 'A'
            A_result = finder.lookup_symbol('A', asof)
            assert_equal(
                A_result,
                finder.retrieve_asset(0),
                msg=str(asof),
            )
            # The symbol and asset_name should always be the last held values
            assert_equal(A_result.symbol, 'B')
            assert_equal(A_result.asset_name, 'Asset B')

            # from 01 through 05 sid 1 held 'C'
            C_result = finder.lookup_symbol('C', asof)
            assert_equal(
                C_result,
                finder.retrieve_asset(1),
                msg=str(asof),
            )
            # The symbol and asset_name should always be the last held values
            assert_equal(C_result.symbol, 'A')
            assert_equal(C_result.asset_name, 'Asset A')

        # no one held 'B' before 06
        with self.assertRaises(SymbolNotFound):
            finder.lookup_symbol('B', T('2014-01-05'))

        # no one held 'C' after 06, however, no one has claimed it yet
        # so it still maps to sid 1
        assert_equal(
            finder.lookup_symbol('C', T('2014-01-07')),
            finder.retrieve_asset(1),
        )

        for asof in pd.date_range('2014-01-06', '2014-01-11', tz='utc'):
            # from 06 through 10 sid 0 held 'B'
            # we test through the 11th because sid 1 is the last to hold 'B'
            # so it should ffill
            B_result = finder.lookup_symbol('B', asof)
            assert_equal(
                B_result,
                finder.retrieve_asset(0),
                msg=str(asof),
            )
            assert_equal(B_result.symbol, 'B')
            assert_equal(B_result.asset_name, 'Asset B')

            # from 06 through 10 sid 1 held 'A'
            # we test through the 11th because sid 1 is the last to hold 'A'
            # so it should ffill
            A_result = finder.lookup_symbol('A', asof)
            assert_equal(
                A_result,
                finder.retrieve_asset(1),
                msg=str(asof),
            )
            assert_equal(A_result.symbol, 'A')
            assert_equal(A_result.asset_name, 'Asset A')

    def test_lookup_symbol(self):

        # Incrementing by two so that start and end dates for each
        # generated Asset don't overlap (each Asset's end_date is the
        # day after its start date.)
        dates = pd.date_range('2013-01-01', freq='2D', periods=5, tz='UTC')
        df = pd.DataFrame.from_records(
            [
                {
                    'sid': i,
                    'symbol':  'existing',
                    'start_date': date.value,
                    'end_date': (date + timedelta(days=1)).value,
                    'exchange': 'NYSE',
                }
                for i, date in enumerate(dates)
            ]
        )
        self.write_assets(equities=df)
        finder = self.asset_finder
        for _ in range(2):  # Run checks twice to test for caching bugs.
            with self.assertRaises(SymbolNotFound):
                finder.lookup_symbol('NON_EXISTING', dates[0])

            with self.assertRaises(MultipleSymbolsFound):
                finder.lookup_symbol('EXISTING', None)

            for i, date in enumerate(dates):
                # Verify that we correctly resolve multiple symbols using
                # the supplied date
                result = finder.lookup_symbol('EXISTING', date)
                self.assertEqual(result.symbol, 'EXISTING')
                self.assertEqual(result.sid, i)

    def test_fail_to_write_overlapping_data(self):
        df = pd.DataFrame.from_records(
            [
                {
                    'sid': 1,
                    'symbol': 'multiple',
                    'start_date': pd.Timestamp('2010-01-01'),
                    'end_date': pd.Timestamp('2012-01-01'),
                    'exchange': 'NYSE'
                },
                # Same as asset 1, but with a later end date.
                {
                    'sid': 2,
                    'symbol': 'multiple',
                    'start_date': pd.Timestamp('2010-01-01'),
                    'end_date': pd.Timestamp('2013-01-01'),
                    'exchange': 'NYSE'
                },
                # Same as asset 1, but with a later start_date
                {
                    'sid': 3,
                    'symbol': 'multiple',
                    'start_date': pd.Timestamp('2011-01-01'),
                    'end_date': pd.Timestamp('2012-01-01'),
                    'exchange': 'NYSE'
                },
            ]
        )

        with self.assertRaises(ValueError) as e:
            self.write_assets(equities=df)

        expected_error_msg = (
            "Ambiguous ownership for 1 symbol, multiple assets held the"
            " following symbols:\n"
            "MULTIPLE (??):\n"
            "  intersections: (('2010-01-01 00:00:00', '2012-01-01 00:00:00'),"
            " ('2011-01-01 00:00:00', '2012-01-01 00:00:00'))\n"
            "      start_date   end_date\n"
            "  sid                      \n"
            "  1   2010-01-01 2012-01-01\n"
            "  2   2010-01-01 2013-01-01\n"
            "  3   2011-01-01 2012-01-01"
        )
        self.assertEqual(str(e.exception), expected_error_msg)

    def test_lookup_generic(self):
        """
        Ensure that lookup_generic works with various permutations of inputs.
        """
        cases = build_lookup_generic_cases()
        # Make sure we clean up temp resources in the generator if we don't
        # consume the whole thing because of a failure.
        self.add_instance_callback(cases.close)
        for finder, inputs, reference_date, country, expected in cases:
            results, missing = finder.lookup_generic(
                inputs, reference_date, country,
            )
            self.assertEqual(results, expected)
            self.assertEqual(missing, [])

    def test_lookup_none_raises(self):
        """
        If lookup_symbol is vectorized across multiple symbols, and one of them
        is None, want to raise a TypeError.
        """

        with self.assertRaises(TypeError):
            self.asset_finder.lookup_symbol(None, pd.Timestamp('2013-01-01'))

    def test_lookup_mult_are_one(self):
        """
        Ensure that multiple symbols that return the same sid are collapsed to
        a single returned asset.
        """

        date = pd.Timestamp('2013-01-01', tz='UTC')

        df = pd.DataFrame.from_records(
            [
                {
                    'sid': 1,
                    'symbol': symbol,
                    'start_date': date.value,
                    'end_date': (date + timedelta(days=30)).value,
                    'exchange': 'NYSE',
                }
                for symbol in ('FOOB', 'FOO_B')
            ]
        )
        self.write_assets(equities=df)
        finder = self.asset_finder

        # If we are able to resolve this with any result, means that we did not
        # raise a MultipleSymbolError.
        result = finder.lookup_symbol('FOO/B', date + timedelta(1), fuzzy=True)
        self.assertEqual(result.sid, 1)

    def test_endless_multiple_resolves(self):
        """
        Situation:
        1. Asset 1 w/ symbol FOOB changes to FOO_B, and then is delisted.
        2. Asset 2 is listed with symbol FOO_B.

        If someone asks for FOO_B with fuzzy matching after 2 has been listed,
        they should be able to correctly get 2.
        """

        date = pd.Timestamp('2013-01-01', tz='UTC')

        df = pd.DataFrame.from_records(
            [
                {
                    'sid': 1,
                    'symbol': 'FOOB',
                    'start_date': date.value,
                    'end_date': date.max.value,
                    'exchange': 'NYSE',
                },
                {
                    'sid': 1,
                    'symbol': 'FOO_B',
                    'start_date': (date + timedelta(days=31)).value,
                    'end_date': (date + timedelta(days=60)).value,
                    'exchange': 'NYSE',
                },
                {
                    'sid': 2,
                    'symbol': 'FOO_B',
                    'start_date': (date + timedelta(days=61)).value,
                    'end_date': date.max.value,
                    'exchange': 'NYSE',
                },

            ]
        )
        self.write_assets(equities=df)
        finder = self.asset_finder

        # If we are able to resolve this with any result, means that we did not
        # raise a MultipleSymbolError.
        result = finder.lookup_symbol(
            'FOO/B',
            date + timedelta(days=90),
            fuzzy=True
        )
        self.assertEqual(result.sid, 2)

    def test_lookup_generic_handle_missing(self):
        data = pd.DataFrame.from_records(
            [
                {
                    'sid': 0,
                    'symbol': 'real',
                    'start_date': pd.Timestamp('2013-1-1', tz='UTC'),
                    'end_date': pd.Timestamp('2014-1-1', tz='UTC'),
                    'exchange': 'TEST',
                },
                {
                    'sid': 1,
                    'symbol': 'also_real',
                    'start_date': pd.Timestamp('2013-1-1', tz='UTC'),
                    'end_date': pd.Timestamp('2014-1-1', tz='UTC'),
                    'exchange': 'TEST',
                },
                # Sid whose end date is before our query date.  We should
                # still correctly find it.
                {
                    'sid': 2,
                    'symbol': 'real_but_old',
                    'start_date': pd.Timestamp('2002-1-1', tz='UTC'),
                    'end_date': pd.Timestamp('2003-1-1', tz='UTC'),
                    'exchange': 'TEST',
                },
                # Sid whose start_date is **after** our query date.  We should
                # **not** find it.
                {
                    'sid': 3,
                    'symbol': 'real_but_in_the_future',
                    'start_date': pd.Timestamp('2014-1-1', tz='UTC'),
                    'end_date': pd.Timestamp('2020-1-1', tz='UTC'),
                    'exchange': 'THE FUTURE',
                },
            ]
        )
        self.write_assets(equities=data)
        finder = self.asset_finder
        results, missing = finder.lookup_generic(
            ['REAL', 1, 'FAKE', 'REAL_BUT_OLD', 'REAL_BUT_IN_THE_FUTURE'],
            pd.Timestamp('2013-02-01', tz='UTC'),
            country_code=None,
        )

        self.assertEqual(len(results), 3)
        self.assertEqual(results[0].symbol, 'REAL')
        self.assertEqual(results[0].sid, 0)
        self.assertEqual(results[1].symbol, 'ALSO_REAL')
        self.assertEqual(results[1].sid, 1)
        self.assertEqual(results[2].symbol, 'REAL_BUT_OLD')
        self.assertEqual(results[2].sid, 2)

        self.assertEqual(len(missing), 2)
        self.assertEqual(missing[0], 'FAKE')
        self.assertEqual(missing[1], 'REAL_BUT_IN_THE_FUTURE')

    def test_lookup_generic_multiple_symbols_across_countries(self):
        data = pd.DataFrame.from_records(
            [
                {
                    'sid': 0,
                    'symbol': 'real',
                    'start_date': pd.Timestamp('2013-1-1', tz='UTC'),
                    'end_date': pd.Timestamp('2014-1-1', tz='UTC'),
                    'exchange': 'US_EXCHANGE',
                },
                {
                    'sid': 1,
                    'symbol': 'real',
                    'start_date': pd.Timestamp('2013-1-1', tz='UTC'),
                    'end_date': pd.Timestamp('2014-1-1', tz='UTC'),
                    'exchange': 'CA_EXCHANGE',
                },
            ]
        )
        exchanges = pd.DataFrame.from_records([
            {'exchange': 'US_EXCHANGE', 'country_code': 'US'},
            {'exchange': 'CA_EXCHANGE', 'country_code': 'CA'},
        ])

        self.write_assets(equities=data, exchanges=exchanges)

        with self.assertRaises(MultipleSymbolsFound):
            self.asset_finder.lookup_generic(
                'real',
                as_of_date=pd.Timestamp('2014-1-1', tz='UTC'),
                country_code=None,
            )

        matches, missing = self.asset_finder.lookup_generic(
            'real',
            as_of_date=pd.Timestamp('2014-1-1', tz='UTC'),
            country_code='US',
        )
        self.assertEqual([matches], [self.asset_finder.retrieve_asset(0)])
        self.assertEqual(missing, [])

        matches, missing = self.asset_finder.lookup_generic(
            'real',
            as_of_date=pd.Timestamp('2014-1-1', tz='UTC'),
            country_code='CA',
        )
        self.assertEqual([matches], [self.asset_finder.retrieve_asset(1)])
        self.assertEqual(missing, [])

    def test_security_dates_warning(self):

        # Build an asset with an end_date
        eq_end = pd.Timestamp('2012-01-01', tz='UTC')
        equity_asset = Equity(1, symbol="TESTEQ", end_date=eq_end,
                              exchange_info=ExchangeInfo("TEST", "TEST", "??"))

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

    def test_compute_lifetimes(self):
        assets_per_exchange = 4
        trading_day = self.trading_calendar.day
        first_start = pd.Timestamp('2015-04-01', tz='UTC')

        equities = pd.concat(
            [
                make_rotating_equity_info(
                    num_assets=assets_per_exchange,
                    first_start=first_start,
                    frequency=trading_day,
                    periods_between_starts=3,
                    asset_lifetime=5,
                    exchange=exchange,
                )
                for exchange in (
                    'US_EXCHANGE_1',
                    'US_EXCHANGE_2',
                    'CA_EXCHANGE',
                    'JP_EXCHANGE',
                )
            ],
            ignore_index=True,
        )
        # make every symbol unique
        equities['symbol'] = list(string.ascii_uppercase[:len(equities)])

        # shuffle up the sids so they are not contiguous per exchange
        sids = np.arange(len(equities))
        np.random.RandomState(1337).shuffle(sids)
        equities.index = sids
        permute_sid = dict(zip(sids, range(len(sids)))).__getitem__

        exchanges = pd.DataFrame.from_records([
            {'exchange': 'US_EXCHANGE_1', 'country_code': 'US'},
            {'exchange': 'US_EXCHANGE_2', 'country_code': 'US'},
            {'exchange': 'CA_EXCHANGE', 'country_code': 'CA'},
            {'exchange': 'JP_EXCHANGE', 'country_code': 'JP'},
        ])
        sids_by_country = {
            'US': equities.index[:2 * assets_per_exchange],
            'CA': equities.index[
                2 * assets_per_exchange:3 * assets_per_exchange
            ],
            'JP': equities.index[3 * assets_per_exchange:],
        }
        self.write_assets(equities=equities, exchanges=exchanges)
        finder = self.asset_finder

        all_dates = pd.date_range(
            start=first_start,
            end=equities.end_date.max(),
            freq=trading_day,
        )

        for dates in all_subindices(all_dates):
            expected_with_start_raw = full(
                shape=(len(dates), assets_per_exchange),
                fill_value=False,
                dtype=bool,
            )
            expected_no_start_raw = full(
                shape=(len(dates), assets_per_exchange),
                fill_value=False,
                dtype=bool,
            )

            for i, date in enumerate(dates):
                it = equities.iloc[:4][['start_date', 'end_date']].itertuples(
                    index=False,
                )
                for j, (start, end) in enumerate(it):
                    # This way of doing the checks is redundant, but very
                    # clear.
                    if start <= date <= end:
                        expected_with_start_raw[i, j] = True
                        if start < date:
                            expected_no_start_raw[i, j] = True

            for country_codes in powerset(exchanges.country_code.unique()):
                expected_sids = pd.Int64Index(sorted(concat(
                    sids_by_country[country_code]
                    for country_code in country_codes
                )))
                permuted_sids = [
                    sid for sid in sorted(expected_sids, key=permute_sid)
                ]
                tile_count = len(country_codes) + ('US' in country_codes)
                expected_with_start = pd.DataFrame(
                    data=np.tile(
                        expected_with_start_raw,
                        tile_count,
                    ),
                    index=dates,
                    columns=pd.Int64Index(permuted_sids),
                )
                result = finder.lifetimes(
                    dates,
                    include_start_date=True,
                    country_codes=country_codes,
                )
                assert_equal(result.columns, expected_sids)
                result = result[permuted_sids]
                assert_equal(result, expected_with_start)

                expected_no_start = pd.DataFrame(
                    data=np.tile(
                        expected_no_start_raw,
                        tile_count,
                    ),
                    index=dates,
                    columns=pd.Int64Index(permuted_sids),
                )
                result = finder.lifetimes(
                    dates,
                    include_start_date=False,
                    country_codes=country_codes,
                )
                assert_equal(result.columns, expected_sids)
                result = result[permuted_sids]
                assert_equal(result, expected_no_start)

    def test_sids(self):
        # Ensure that the sids property of the AssetFinder is functioning
        self.write_assets(equities=make_simple_equity_info(
            [0, 1, 2],
            pd.Timestamp('2014-01-01'),
            pd.Timestamp('2014-01-02'),
        ))
        self.assertEqual({0, 1, 2}, set(self.asset_finder.sids))

    def test_lookup_by_supplementary_field(self):
        equities = pd.DataFrame.from_records(
            [
                {
                    'sid': 0,
                    'symbol': 'A',
                    'start_date': pd.Timestamp('2013-1-1', tz='UTC'),
                    'end_date': pd.Timestamp('2014-1-1', tz='UTC'),
                    'exchange': 'TEST',
                },
                {
                    'sid': 1,
                    'symbol': 'B',
                    'start_date': pd.Timestamp('2013-1-1', tz='UTC'),
                    'end_date': pd.Timestamp('2014-1-1', tz='UTC'),
                    'exchange': 'TEST',
                },
                {
                    'sid': 2,
                    'symbol': 'C',
                    'start_date': pd.Timestamp('2013-7-1', tz='UTC'),
                    'end_date': pd.Timestamp('2014-1-1', tz='UTC'),
                    'exchange': 'TEST',
                },
            ]
        )

        equity_supplementary_mappings = pd.DataFrame.from_records(
            [
                {
                    'sid': 0,
                    'field': 'ALT_ID',
                    'value': '100000000',
                    'start_date': pd.Timestamp('2013-1-1', tz='UTC'),
                    'end_date': pd.Timestamp('2013-6-28', tz='UTC'),
                },
                {
                    'sid': 1,
                    'field': 'ALT_ID',
                    'value': '100000001',
                    'start_date': pd.Timestamp('2013-1-1', tz='UTC'),
                    'end_date': pd.Timestamp('2014-1-1', tz='UTC'),
                },
                {
                    'sid': 0,
                    'field': 'ALT_ID',
                    'value': '100000002',
                    'start_date': pd.Timestamp('2013-7-1', tz='UTC'),
                    'end_date': pd.Timestamp('2014-1-1', tz='UTC'),
                },
                {
                    'sid': 2,
                    'field': 'ALT_ID',
                    'value': '100000000',
                    'start_date': pd.Timestamp('2013-7-1', tz='UTC'),
                    'end_date': pd.Timestamp('2014-1-1', tz='UTC'),
                },
            ]
        )

        self.write_assets(
            equities=equities,
            equity_supplementary_mappings=equity_supplementary_mappings,
        )

        af = self.asset_finder

        # Before sid 0 has changed ALT_ID.
        dt = pd.Timestamp('2013-6-28', tz='UTC')

        asset_0 = af.lookup_by_supplementary_field('ALT_ID', '100000000', dt)
        self.assertEqual(asset_0.sid, 0)

        asset_1 = af.lookup_by_supplementary_field('ALT_ID', '100000001', dt)
        self.assertEqual(asset_1.sid, 1)

        # We don't know about this ALT_ID yet.
        with self.assertRaisesRegexp(
            ValueNotFoundForField,
            "Value '{}' was not found for field '{}'.".format(
                '100000002',
                'ALT_ID',
            )
        ):
            af.lookup_by_supplementary_field('ALT_ID', '100000002', dt)

        # After all assets have ended.
        dt = pd.Timestamp('2014-01-02', tz='UTC')

        asset_2 = af.lookup_by_supplementary_field('ALT_ID', '100000000', dt)
        self.assertEqual(asset_2.sid, 2)

        asset_1 = af.lookup_by_supplementary_field('ALT_ID', '100000001', dt)
        self.assertEqual(asset_1.sid, 1)

        asset_0 = af.lookup_by_supplementary_field('ALT_ID', '100000002', dt)
        self.assertEqual(asset_0.sid, 0)

        # At this point both sids 0 and 2 have held this value, so an
        # as_of_date is required.
        expected_in_repr = (
            "Multiple occurrences of the value '{}' found for field '{}'."
        ).format('100000000', 'ALT_ID')

        with self.assertRaisesRegexp(
            MultipleValuesFoundForField,
            expected_in_repr,
        ):
            af.lookup_by_supplementary_field('ALT_ID', '100000000', None)

    def test_get_supplementary_field(self):
        equities = pd.DataFrame.from_records(
            [
                {
                    'sid': 0,
                    'symbol': 'A',
                    'start_date': pd.Timestamp('2013-1-1', tz='UTC'),
                    'end_date': pd.Timestamp('2014-1-1', tz='UTC'),
                    'exchange': 'TEST',
                },
                {
                    'sid': 1,
                    'symbol': 'B',
                    'start_date': pd.Timestamp('2013-1-1', tz='UTC'),
                    'end_date': pd.Timestamp('2014-1-1', tz='UTC'),
                    'exchange': 'TEST',
                },
                {
                    'sid': 2,
                    'symbol': 'C',
                    'start_date': pd.Timestamp('2013-7-1', tz='UTC'),
                    'end_date': pd.Timestamp('2014-1-1', tz='UTC'),
                    'exchange': 'TEST',
                },
            ]
        )

        equity_supplementary_mappings = pd.DataFrame.from_records(
            [
                {
                    'sid': 0,
                    'field': 'ALT_ID',
                    'value': '100000000',
                    'start_date': pd.Timestamp('2013-1-1', tz='UTC'),
                    'end_date': pd.Timestamp('2013-6-28', tz='UTC'),
                },
                {
                    'sid': 1,
                    'field': 'ALT_ID',
                    'value': '100000001',
                    'start_date': pd.Timestamp('2013-1-1', tz='UTC'),
                    'end_date': pd.Timestamp('2014-1-1', tz='UTC'),
                },
                {
                    'sid': 0,
                    'field': 'ALT_ID',
                    'value': '100000002',
                    'start_date': pd.Timestamp('2013-7-1', tz='UTC'),
                    'end_date': pd.Timestamp('2014-1-1', tz='UTC'),
                },
                {
                    'sid': 2,
                    'field': 'ALT_ID',
                    'value': '100000000',
                    'start_date': pd.Timestamp('2013-7-1', tz='UTC'),
                    'end_date': pd.Timestamp('2014-1-1', tz='UTC'),
                },
            ]
        )

        self.write_assets(
            equities=equities,
            equity_supplementary_mappings=equity_supplementary_mappings,
        )
        finder = self.asset_finder

        # Before sid 0 has changed ALT_ID and sid 2 has started.
        dt = pd.Timestamp('2013-6-28', tz='UTC')

        for sid, expected in [(0, '100000000'), (1, '100000001')]:
            self.assertEqual(
                finder.get_supplementary_field(sid, 'ALT_ID', dt),
                expected,
            )

        # Since sid 2 has not yet started, we don't know about its
        # ALT_ID.
        with self.assertRaisesRegexp(
            NoValueForSid,
            "No '{}' value found for sid '{}'.".format('ALT_ID', 2),
        ):
            finder.get_supplementary_field(2, 'ALT_ID', dt),

        # After all assets have ended.
        dt = pd.Timestamp('2014-01-02', tz='UTC')

        for sid, expected in [
            (0, '100000002'), (1, '100000001'), (2, '100000000'),
        ]:
            self.assertEqual(
                finder.get_supplementary_field(sid, 'ALT_ID', dt),
                expected,
            )

        # Sid 0 has historically held two values for ALT_ID by this dt.
        with self.assertRaisesRegexp(
            MultipleValuesFoundForSid,
            "Multiple '{}' values found for sid '{}'.".format('ALT_ID', 0),
        ):
            finder.get_supplementary_field(0, 'ALT_ID', None),

    def test_group_by_type(self):
        equities = make_simple_equity_info(
            range(5),
            start_date=pd.Timestamp('2014-01-01'),
            end_date=pd.Timestamp('2015-01-01'),
        )
        futures = make_commodity_future_info(
            first_sid=6,
            root_symbols=['CL'],
            years=[2014],
        )
        # Intersecting sid queries, to exercise loading of partially-cached
        # results.
        queries = [
            ([0, 1, 3], [6, 7]),
            ([0, 2, 3], [7, 10]),
            (list(equities.index), list(futures.index)),
        ]
        self.write_assets(
            equities=equities,
            futures=futures,
        )
        finder = self.asset_finder
        for equity_sids, future_sids in queries:
            results = finder.group_by_type(equity_sids + future_sids)
            self.assertEqual(
                results,
                {'equity': set(equity_sids), 'future': set(future_sids)},
            )

    @parameterized.expand([
        (Equity, 'retrieve_equities', EquitiesNotFound),
        (Future, 'retrieve_futures_contracts', FutureContractsNotFound),
    ])
    def test_retrieve_specific_type(self, type_, lookup_name, failure_type):
        equities = make_simple_equity_info(
            range(5),
            start_date=pd.Timestamp('2014-01-01'),
            end_date=pd.Timestamp('2015-01-01'),
        )
        max_equity = equities.index.max()
        futures = make_commodity_future_info(
            first_sid=max_equity + 1,
            root_symbols=['CL'],
            years=[2014],
        )
        equity_sids = [0, 1]
        future_sids = [max_equity + 1, max_equity + 2, max_equity + 3]
        if type_ == Equity:
            success_sids = equity_sids
            fail_sids = future_sids
        else:
            fail_sids = equity_sids
            success_sids = future_sids

        self.write_assets(
            equities=equities,
            futures=futures,
        )
        finder = self.asset_finder
        # Run twice to exercise caching.
        lookup = getattr(finder, lookup_name)
        for _ in range(2):
            results = lookup(success_sids)
            self.assertIsInstance(results, dict)
            self.assertEqual(set(results.keys()), set(success_sids))
            self.assertEqual(
                valmap(int, results),
                dict(zip(success_sids, success_sids)),
            )
            self.assertEqual(
                {type_},
                {type(asset) for asset in itervalues(results)},
            )
            with self.assertRaises(failure_type):
                lookup(fail_sids)
            with self.assertRaises(failure_type):
                # Should fail if **any** of the assets are bad.
                lookup([success_sids[0], fail_sids[0]])

    def test_retrieve_all(self):
        equities = make_simple_equity_info(
            range(5),
            start_date=pd.Timestamp('2014-01-01'),
            end_date=pd.Timestamp('2015-01-01'),
        )
        max_equity = equities.index.max()
        futures = make_commodity_future_info(
            first_sid=max_equity + 1,
            root_symbols=['CL'],
            years=[2014],
        )
        self.write_assets(
            equities=equities,
            futures=futures,
        )
        finder = self.asset_finder
        all_sids = finder.sids
        self.assertEqual(len(all_sids), len(equities) + len(futures))
        queries = [
            # Empty Query.
            (),
            # Only Equities.
            tuple(equities.index[:2]),
            # Only Futures.
            tuple(futures.index[:3]),
            # Mixed, all cache misses.
            tuple(equities.index[2:]) + tuple(futures.index[3:]),
            # Mixed, all cache hits.
            tuple(equities.index[2:]) + tuple(futures.index[3:]),
            # Everything.
            all_sids,
            all_sids,
        ]
        for sids in queries:
            equity_sids = [i for i in sids if i <= max_equity]
            future_sids = [i for i in sids if i > max_equity]
            results = finder.retrieve_all(sids)
            self.assertEqual(sids, tuple(map(int, results)))

            self.assertEqual(
                [Equity for _ in equity_sids] +
                [Future for _ in future_sids],
                list(map(type, results)),
            )
            self.assertEqual(
                (
                    list(equities.symbol.loc[equity_sids]) +
                    list(futures.symbol.loc[future_sids])
                ),
                list(asset.symbol for asset in results),
            )

    @parameterized.expand([
        (EquitiesNotFound, 'equity', 'equities'),
        (FutureContractsNotFound, 'future contract', 'future contracts'),
        (SidsNotFound, 'asset', 'assets'),
    ])
    def test_error_message_plurality(self,
                                     error_type,
                                     singular,
                                     plural):
        try:
            raise error_type(sids=[1])
        except error_type as e:
            self.assertEqual(
                str(e),
                "No {singular} found for sid: 1.".format(singular=singular)
            )
        try:
            raise error_type(sids=[1, 2])
        except error_type as e:
            self.assertEqual(
                str(e),
                "No {plural} found for sids: [1, 2].".format(plural=plural)
            )


class AssetFinderMultipleCountries(WithTradingCalendars, ZiplineTestCase):
    def write_assets(self, **kwargs):
        self._asset_writer.write(**kwargs)

    def init_instance_fixtures(self):
        super(AssetFinderMultipleCountries, self).init_instance_fixtures()

        conn = self.enter_instance_context(empty_assets_db())
        self._asset_writer = AssetDBWriter(conn)
        self.asset_finder = AssetFinder(conn)

    @staticmethod
    def country_code(n):
        return 'A' + chr(ord('A') + n)

    def test_lookup_symbol_delimited(self):
        as_of = pd.Timestamp('2013-01-01', tz='UTC')
        num_assets = 3
        sids = list(range(num_assets))
        frame = pd.DataFrame.from_records(
            [
                {
                    'sid': sid,
                    'symbol':  'TEST.A',
                    'company_name': "company %d" % sid,
                    'start_date': as_of.value,
                    'end_date': as_of.value,
                    'exchange': 'EXCHANGE %d' % sid,
                }
                for sid in sids
            ]
        )

        exchanges = pd.DataFrame({
            'exchange': frame['exchange'],
            'country_code': [self.country_code(n) for n in range(num_assets)],
        })
        self.write_assets(equities=frame, exchanges=exchanges)
        finder = self.asset_finder
        assets = finder.retrieve_all(sids)

        def shouldnt_resolve(ticker):
            with self.assertRaises(SymbolNotFound):
                finder.lookup_symbol(ticker, as_of)
            for n in range(num_assets):
                with self.assertRaises(SymbolNotFound):
                    finder.lookup_symbol(
                        ticker,
                        as_of,
                        country_code=self.country_code(n),
                    )

        # we do it twice to catch caching bugs
        for _ in range(2):
            shouldnt_resolve('TEST')
            shouldnt_resolve('TESTA')
            # '@' is not a supported delimiter
            shouldnt_resolve('TEST@A')

            # Adding an unnecessary delimiter shouldn't matter.
            for delimiter in '-', '/', '_', '.':
                ticker = 'TEST%sA' % delimiter
                with self.assertRaises(MultipleSymbolsFound):
                    finder.lookup_symbol(ticker, as_of)

                for n in range(num_assets):
                    actual_asset = finder.lookup_symbol(
                        ticker,
                        as_of,
                        country_code=self.country_code(n),
                    )
                    assert_equal(actual_asset, assets[n])
                    assert_equal(
                        actual_asset.exchange_info.country_code,
                        self.country_code(n),
                    )

    def test_lookup_symbol_fuzzy(self):
        num_countries = 3
        metadata = pd.DataFrame.from_records([
            {'symbol': symbol, 'exchange': 'EXCHANGE %d' % n}
            for n in range(num_countries)
            for symbol in ('PRTY_HRD', 'BRKA', 'BRK_A')
        ])
        exchanges = pd.DataFrame({
            'exchange': metadata['exchange'].unique(),
            'country_code': list(map(self.country_code, range(num_countries))),
        })
        self.write_assets(equities=metadata, exchanges=exchanges)
        finder = self.asset_finder
        dt = pd.Timestamp('2013-01-01', tz='UTC')

        # Try combos of looking up PRTYHRD with and without a time or fuzzy
        # Both non-fuzzys get no result
        with self.assertRaises(SymbolNotFound):
            finder.lookup_symbol('PRTYHRD', None)
        with self.assertRaises(SymbolNotFound):
            finder.lookup_symbol('PRTYHRD', dt)

        for n in range(num_countries):
            # Given that this ticker isn't defined in any country, explicitly
            # passing a country code should still fail.
            with self.assertRaises(SymbolNotFound):
                finder.lookup_symbol(
                    'PRTYHRD',
                    None,
                    country_code=self.country_code(n),
                )
            with self.assertRaises(SymbolNotFound):
                finder.lookup_symbol(
                    'PRTYHRD',
                    dt,
                    country_code=self.country_code(n),
                )

        with self.assertRaises(MultipleSymbolsFound):
            finder.lookup_symbol('PRTYHRD', None, fuzzy=True)

        with self.assertRaises(MultipleSymbolsFound):
            finder.lookup_symbol('PRTYHRD', dt, fuzzy=True)

        def check_sid(expected_sid, ticker, country_code):
            params = (
                {'as_of_date': None},
                {'as_of_date': dt},
                {'as_of_date': None, 'fuzzy': True},
                {'as_of_date': dt, 'fuzzy': True},
            )
            for extra_params in params:
                with self.assertRaises(MultipleSymbolsFound):
                    finder.lookup_symbol(ticker, **extra_params)

                self.assertEqual(
                    expected_sid,
                    finder.lookup_symbol(
                        ticker,
                        country_code=country_code,
                        **extra_params
                    ),
                )

        for n in range(num_countries):
            check_sid(n * 3, 'PRTY_HRD', self.country_code(n))
            check_sid(n * 3 + 1, 'BRKA', self.country_code(n))
            check_sid(n * 3 + 2, 'BRK_A', self.country_code(n))

    def test_lookup_symbol_change_ticker(self):
        T = partial(pd.Timestamp, tz='utc')
        num_countries = 3
        metadata = pd.DataFrame.from_records(
            [
                # first sid per country
                {
                    'symbol': 'A',
                    'asset_name': 'Asset A',
                    'start_date': T('2014-01-01'),
                    'end_date': T('2014-01-05'),
                },
                {
                    'symbol': 'B',
                    'asset_name': 'Asset B',
                    'start_date': T('2014-01-06'),
                    'end_date': T('2014-01-10'),
                },

                # second sid per country
                {
                    'symbol': 'C',
                    'asset_name': 'Asset C',
                    'start_date': T('2014-01-01'),
                    'end_date': T('2014-01-05'),
                },
                {
                    'symbol': 'A',  # claiming the unused symbol 'A'
                    'asset_name': 'Asset A',
                    'start_date': T('2014-01-06'),
                    'end_date': T('2014-01-10'),
                },
            ] * num_countries,
            index=np.repeat(np.arange(num_countries * 2), 2),
        )
        metadata['exchange'] = np.repeat(
            ['EXCHANGE %d' % n for n in range(num_countries)],
            4,
        )
        exchanges = pd.DataFrame({
            'exchange': ['EXCHANGE %d' % n for n in range(num_countries)],
            'country_code': [
                self.country_code(n) for n in range(num_countries)
            ]
        })
        self.write_assets(equities=metadata, exchanges=exchanges)
        finder = self.asset_finder

        def assert_doesnt_resolve(symbol, as_of_date):
            # check across all countries
            with self.assertRaises(SymbolNotFound):
                finder.lookup_symbol(symbol, as_of_date)

            # check in each country individually
            for n in range(num_countries):
                with self.assertRaises(SymbolNotFound):
                    finder.lookup_symbol(
                        symbol,
                        as_of_date,
                        country_code=self.country_code(n),
                    )

        def assert_resolves_in_each_country(symbol,
                                            as_of_date,
                                            sid_from_country_ix,
                                            expected_symbol,
                                            expected_name):
            # ensure this is ambiguous across all countries
            with self.assertRaises(MultipleSymbolsFound):
                finder.lookup_symbol(symbol, as_of_date)

            for n in range(num_countries):
                result = finder.lookup_symbol(
                    symbol,
                    as_of_date,
                    country_code=self.country_code(n),
                )
                assert_equal(
                    result,
                    finder.retrieve_asset(sid_from_country_ix(n)),
                    msg=str(asof),
                )
                # The symbol and asset_name should always be the last held
                # values
                assert_equal(result.symbol, expected_symbol)
                assert_equal(result.asset_name, expected_name)

        # note: these assertions walk forward in time, starting at assertions
        # about ownership before the start_date and ending with assertions
        # after the end_date; new assertions should be inserted in the correct
        # locations

        # no one held 'A' before 01
        assert_doesnt_resolve('A', T('2013-12-31'))

        # no one held 'C' before 01
        assert_doesnt_resolve('C', T('2013-12-31'))

        for asof in pd.date_range('2014-01-01', '2014-01-05', tz='utc'):
            # from 01 through 05 the first sid on the exchange held 'A'
            assert_resolves_in_each_country(
                'A',
                asof,
                sid_from_country_ix=lambda n: n * 2,
                expected_symbol='B',
                expected_name='Asset B',
            )

            # from 01 through 05 the second sid on the exchange held 'C'
            assert_resolves_in_each_country(
                'C',
                asof,
                sid_from_country_ix=lambda n: n * 2 + 1,
                expected_symbol='A',
                expected_name='Asset A',
            )

        # no one held 'B' before 06
        assert_doesnt_resolve('B', T('2014-01-05'))

        # no one held 'C' after 06, however, no one has claimed it yet
        # so it still maps to sid 1
        assert_resolves_in_each_country(
            'C',
            T('2014-01-07'),
            sid_from_country_ix=lambda n: n * 2 + 1,
            expected_symbol='A',
            expected_name='Asset A',
        )

        for asof in pd.date_range('2014-01-06', '2014-01-11', tz='utc'):
            # from 06 through 10 sid 0 held 'B'
            # we test through the 11th because sid 1 is the last to hold 'B'
            # so it should ffill
            assert_resolves_in_each_country(
                'B',
                asof,
                sid_from_country_ix=lambda n: n * 2,
                expected_symbol='B',
                expected_name='Asset B',
            )

            # from 06 through 10 sid 1 held 'A'
            # we test through the 11th because sid 1 is the last to hold 'A'
            # so it should ffill
            assert_resolves_in_each_country(
                'A',
                asof,
                sid_from_country_ix=lambda n: n * 2 + 1,
                expected_symbol='A',
                expected_name='Asset A',
            )

    def test_lookup_symbol(self):
        num_countries = 3
        # Incrementing by two so that start and end dates for each
        # generated Asset don't overlap (each Asset's end_date is the
        # day after its start date.)
        dates = pd.date_range('2013-01-01', freq='2D', periods=5, tz='UTC')
        df = pd.DataFrame.from_records(
            [
                {
                    'sid': n * len(dates) + i,
                    'symbol':  'existing',
                    'start_date': date.value,
                    'end_date': (date + timedelta(days=1)).value,
                    'exchange': 'EXCHANGE %d' % n,
                }
                for n in range(num_countries)
                for i, date in enumerate(dates)
            ]
        )
        exchanges = pd.DataFrame({
            'exchange': ['EXCHANGE %d' % n for n in range(num_countries)],
            'country_code': [
                self.country_code(n) for n in range(num_countries)
            ],
        })
        self.write_assets(equities=df, exchanges=exchanges)
        finder = self.asset_finder
        for _ in range(2):  # Run checks twice to test for caching bugs.
            with self.assertRaises(SymbolNotFound):
                finder.lookup_symbol('NON_EXISTING', dates[0])
            for n in range(num_countries):
                with self.assertRaises(SymbolNotFound):
                    finder.lookup_symbol(
                        'NON_EXISTING',
                        dates[0],
                        country_code=self.country_code(n),
                    )

            with self.assertRaises(MultipleSymbolsFound):
                finder.lookup_symbol('EXISTING', None)

            for n in range(num_countries):
                with self.assertRaises(MultipleSymbolsFound):
                    finder.lookup_symbol(
                        'EXISTING',
                        None,
                        country_code=self.country_code(n),
                    )

            for i, date in enumerate(dates):
                # Verify that we correctly resolve multiple symbols using
                # the supplied date
                with self.assertRaises(MultipleSymbolsFound):
                    finder.lookup_symbol('EXISTING', date)

                for n in range(num_countries):
                    result = finder.lookup_symbol(
                        'EXISTING',
                        date,
                        country_code=self.country_code(n),
                    )
                    self.assertEqual(result.symbol, 'EXISTING')
                    expected_sid = n * len(dates) + i
                    self.assertEqual(result.sid, expected_sid)

    def test_fail_to_write_overlapping_data(self):
        num_countries = 3
        df = pd.DataFrame.from_records(concat(
            [
                {
                    'sid': n * 3,
                    'symbol': 'multiple',
                    'start_date': pd.Timestamp('2010-01-01'),
                    'end_date': pd.Timestamp('2012-01-01'),
                    'exchange': 'EXCHANGE %d' % n,
                },
                # Same as asset 1, but with a later end date.
                {
                    'sid': n * 3 + 1,
                    'symbol': 'multiple',
                    'start_date': pd.Timestamp('2010-01-01'),
                    'end_date': pd.Timestamp('2013-01-01'),
                    'exchange': 'EXCHANGE %d' % n,
                },
                # Same as asset 1, but with a later start_date
                {
                    'sid': n * 3 + 2,
                    'symbol': 'multiple',
                    'start_date': pd.Timestamp('2011-01-01'),
                    'end_date': pd.Timestamp('2012-01-01'),
                    'exchange': 'EXCHANGE %d' % n,
                },
            ]
            for n in range(num_countries)
        ))
        exchanges = pd.DataFrame({
            'exchange': ['EXCHANGE %d' % n for n in range(num_countries)],
            'country_code': [
                self.country_code(n) for n in range(num_countries)
            ],
        })

        with self.assertRaises(ValueError) as e:
            self.write_assets(equities=df, exchanges=exchanges)

        expected_error_msg = (
            "Ambiguous ownership for 3 symbols, multiple assets held the"
            " following symbols:\n"
            "MULTIPLE (%s):\n"
            "  intersections: (('2010-01-01 00:00:00', '2012-01-01 00:00:00'),"
            " ('2011-01-01 00:00:00', '2012-01-01 00:00:00'))\n"
            "      start_date   end_date\n"
            "  sid                      \n"
            "  0   2010-01-01 2012-01-01\n"
            "  1   2010-01-01 2013-01-01\n"
            "  2   2011-01-01 2012-01-01\n"
            "MULTIPLE (%s):\n"
            "  intersections: (('2010-01-01 00:00:00', '2012-01-01 00:00:00'),"
            " ('2011-01-01 00:00:00', '2012-01-01 00:00:00'))\n"
            "      start_date   end_date\n"
            "  sid                      \n"
            "  3   2010-01-01 2012-01-01\n"
            "  4   2010-01-01 2013-01-01\n"
            "  5   2011-01-01 2012-01-01\n"
            "MULTIPLE (%s):\n"
            "  intersections: (('2010-01-01 00:00:00', '2012-01-01 00:00:00'),"
            " ('2011-01-01 00:00:00', '2012-01-01 00:00:00'))\n"
            "      start_date   end_date\n"
            "  sid                      \n"
            "  6   2010-01-01 2012-01-01\n"
            "  7   2010-01-01 2013-01-01\n"
            "  8   2011-01-01 2012-01-01" % (
                self.country_code(0),
                self.country_code(1),
                self.country_code(2),
            )
        )
        self.assertEqual(str(e.exception), expected_error_msg)

    def test_endless_multiple_resolves(self):
        """
        Situation:
        1. Asset 1 w/ symbol FOOB changes to FOO_B, and then is delisted.
        2. Asset 2 is listed with symbol FOO_B.

        If someone asks for FOO_B with fuzzy matching after 2 has been listed,
        they should be able to correctly get 2.
        """

        date = pd.Timestamp('2013-01-01', tz='UTC')
        num_countries = 3
        df = pd.DataFrame.from_records(concat(
            [
                {
                    'sid': n * 2,
                    'symbol': 'FOOB',
                    'start_date': date.value,
                    'end_date': date.max.value,
                    'exchange': 'EXCHANGE %d' % n,
                },
                {
                    'sid': n * 2,
                    'symbol': 'FOO_B',
                    'start_date': (date + timedelta(days=31)).value,
                    'end_date': (date + timedelta(days=60)).value,
                    'exchange': 'EXCHANGE %d' % n,
                },
                {
                    'sid': n * 2 + 1,
                    'symbol': 'FOO_B',
                    'start_date': (date + timedelta(days=61)).value,
                    'end_date': date.max.value,
                    'exchange': 'EXCHANGE %d' % n,
                },

            ]
            for n in range(num_countries)
        ))
        exchanges = pd.DataFrame({
            'exchange': ['EXCHANGE %d' % n for n in range(num_countries)],
            'country_code': [
                self.country_code(n) for n in range(num_countries)
            ],
        })
        self.write_assets(equities=df, exchanges=exchanges)
        finder = self.asset_finder

        with self.assertRaises(MultipleSymbolsFound):
            finder.lookup_symbol(
                'FOO/B',
                date + timedelta(days=90),
                fuzzy=True,
            )

        for n in range(num_countries):
            result = finder.lookup_symbol(
                'FOO/B',
                date + timedelta(days=90),
                fuzzy=True,
                country_code=self.country_code(n)
            )
            self.assertEqual(result.sid, n * 2 + 1)


class TestAssetDBVersioning(ZiplineTestCase):

    def init_instance_fixtures(self):
        super(TestAssetDBVersioning, self).init_instance_fixtures()
        self.engine = eng = self.enter_instance_context(empty_assets_db())
        self.metadata = sa.MetaData(eng, reflect=True)

    def test_check_version(self):
        version_table = self.metadata.tables['version_info']

        # This should not raise an error
        check_version_info(self.engine, version_table, ASSET_DB_VERSION)

        # This should fail because the version is too low
        with self.assertRaises(AssetDBVersionError):
            check_version_info(
                self.engine,
                version_table,
                ASSET_DB_VERSION - 1,
            )

        # This should fail because the version is too high
        with self.assertRaises(AssetDBVersionError):
            check_version_info(
                self.engine,
                version_table,
                ASSET_DB_VERSION + 1,
            )

    def test_write_version(self):
        version_table = self.metadata.tables['version_info']
        version_table.delete().execute()

        # Assert that the version is not present in the table
        self.assertIsNone(sa.select((version_table.c.version,)).scalar())

        # This should fail because the table has no version info and is,
        # therefore, consdered v0
        with self.assertRaises(AssetDBVersionError):
            check_version_info(self.engine, version_table, -2)

        # This should not raise an error because the version has been written
        write_version_info(self.engine, version_table, -2)
        check_version_info(self.engine, version_table, -2)

        # Assert that the version is in the table and correct
        self.assertEqual(sa.select((version_table.c.version,)).scalar(), -2)

        # Assert that trying to overwrite the version fails
        with self.assertRaises(sa.exc.IntegrityError):
            write_version_info(self.engine, version_table, -3)

    def test_finder_checks_version(self):
        version_table = self.metadata.tables['version_info']
        version_table.delete().execute()
        write_version_info(self.engine, version_table, -2)
        check_version_info(self.engine, version_table, -2)

        # Assert that trying to build a finder with a bad db raises an error
        with self.assertRaises(AssetDBVersionError):
            AssetFinder(engine=self.engine)

        # Change the version number of the db to the correct version
        version_table.delete().execute()
        write_version_info(self.engine, version_table, ASSET_DB_VERSION)
        check_version_info(self.engine, version_table, ASSET_DB_VERSION)

        # Now that the versions match, this Finder should succeed
        AssetFinder(engine=self.engine)

    def test_downgrade(self):
        # Attempt to downgrade a current assets db all the way down to v0
        conn = self.engine.connect()

        # first downgrade to v3
        downgrade(self.engine, 3)
        metadata = sa.MetaData(conn)
        metadata.reflect()
        check_version_info(conn, metadata.tables['version_info'], 3)
        self.assertFalse('exchange_full' in metadata.tables)

        # now go all the way to v0
        downgrade(self.engine, 0)

        # Verify that the db version is now 0
        metadata = sa.MetaData(conn)
        metadata.reflect()
        version_table = metadata.tables['version_info']
        check_version_info(conn, version_table, 0)

        # Check some of the v1-to-v0 downgrades
        self.assertTrue('futures_contracts' in metadata.tables)
        self.assertTrue('version_info' in metadata.tables)
        self.assertFalse('tick_size' in
                         metadata.tables['futures_contracts'].columns)
        self.assertTrue('contract_multiplier' in
                        metadata.tables['futures_contracts'].columns)

    def test_impossible_downgrade(self):
        # Attempt to downgrade a current assets db to a
        # higher-than-current version
        with self.assertRaises(AssetDBImpossibleDowngrade):
            downgrade(self.engine, ASSET_DB_VERSION + 5)

    def test_v5_to_v4_selects_most_recent_ticker(self):
        T = pd.Timestamp
        equities = pd.DataFrame(
            [['A', 'A', T('2014-01-01'), T('2014-01-02')],
             ['B', 'B', T('2014-01-01'), T('2014-01-02')],
             # these two are both ticker sid 2
             ['B', 'C', T('2014-01-03'), T('2014-01-04')],
             ['C', 'C', T('2014-01-01'), T('2014-01-02')]],
            index=[0, 1, 2, 2],
            columns=['symbol', 'asset_name', 'start_date', 'end_date'],
        )
        equities['exchange'] = 'NYSE'

        AssetDBWriter(self.engine).write(equities=equities)

        downgrade(self.engine, 4)
        metadata = sa.MetaData(self.engine)
        metadata.reflect()

        def select_fields(r):
            return r.sid, r.symbol, r.asset_name, r.start_date, r.end_date

        expected_data = {
            (0, 'A', 'A', T('2014-01-01').value, T('2014-01-02').value),
            (1, 'B', 'B', T('2014-01-01').value, T('2014-01-02').value),
            (2, 'B', 'C', T('2014-01-01').value, T('2014-01-04').value),
        }
        actual_data = set(map(
            select_fields,
            sa.select(metadata.tables['equities'].c).execute(),
        ))

        assert_equal(expected_data, actual_data)

    def test_v7_to_v6_only_keeps_US(self):
        T = pd.Timestamp
        equities = pd.DataFrame(
            [['A', T('2014-01-01'), T('2014-01-02'), 'NYSE'],
             ['B', T('2014-01-01'), T('2014-01-02'), 'JPX'],
             ['C', T('2014-01-03'), T('2014-01-04'), 'NYSE'],
             ['D', T('2014-01-01'), T('2014-01-02'), 'JPX']],
            index=[0, 1, 2, 3],
            columns=['symbol', 'start_date', 'end_date', 'exchange'],
        )
        exchanges = pd.DataFrame.from_records([
            {'exchange': 'NYSE', 'country_code': 'US'},
            {'exchange': 'JPX', 'country_code': 'JP'},
        ])
        AssetDBWriter(self.engine).write(
            equities=equities,
            exchanges=exchanges,
        )

        downgrade(self.engine, 6)
        metadata = sa.MetaData(self.engine)
        metadata.reflect()

        expected_sids = {0, 2}
        actual_sids = set(map(
            lambda r: r.sid,
            sa.select(metadata.tables['equities'].c).execute(),
        ))

        assert_equal(expected_sids, actual_sids)


class TestVectorizedSymbolLookup(WithAssetFinder, ZiplineTestCase):

    @classmethod
    def make_equity_info(cls):
        T = partial(pd.Timestamp, tz='UTC')

        def asset(sid, symbol, start_date, end_date):
            return dict(
                sid=sid,
                symbol=symbol,
                start_date=T(start_date),
                end_date=T(end_date),
                exchange='NYSE',
            )

        records = [
            asset(1, 'A', '2014-01-02', '2014-01-31'),
            asset(2, 'A', '2014-02-03', '2015-01-02'),
            asset(3, 'B', '2014-01-02', '2014-01-15'),
            asset(4, 'B', '2014-01-17', '2015-01-02'),
            asset(5, 'C', '2001-01-02', '2015-01-02'),
            asset(6, 'D', '2001-01-02', '2015-01-02'),
            asset(7, 'FUZZY', '2001-01-02', '2015-01-02'),
        ]
        return pd.DataFrame.from_records(records)

    @parameter_space(
        as_of=pd.to_datetime([
            '2014-01-02',
            '2014-01-15',
            '2014-01-17',
            '2015-01-02',
        ], utc=True),
        symbols=[
            [],
            ['A'], ['B'], ['C'], ['D'],
            list('ABCD'),
            list('ABCDDCBA'),
            list('AABBAABBACABD'),
        ],
    )
    def test_lookup_symbols(self, as_of, symbols):
        af = self.asset_finder
        expected = [
            af.lookup_symbol(symbol, as_of) for symbol in symbols
        ]
        result = af.lookup_symbols(symbols, as_of)
        assert_equal(result, expected)

    def test_fuzzy(self):
        af = self.asset_finder

        # FUZZ.Y shouldn't resolve unless fuzzy=True.
        syms = ['A', 'B', 'FUZZ.Y']
        dt = pd.Timestamp('2014-01-15', tz='UTC')

        with self.assertRaises(SymbolNotFound):
            af.lookup_symbols(syms, pd.Timestamp('2014-01-15', tz='UTC'))

        with self.assertRaises(SymbolNotFound):
            af.lookup_symbols(
                syms,
                pd.Timestamp('2014-01-15', tz='UTC'),
                fuzzy=False,
            )

        results = af.lookup_symbols(syms, dt, fuzzy=True)
        assert_equal(results, af.retrieve_all([1, 3, 7]))
        assert_equal(
            results,
            [af.lookup_symbol(sym, dt, fuzzy=True) for sym in syms],
        )


class TestAssetFinderPreprocessors(WithTmpDir, ZiplineTestCase):

    def test_asset_finder_doesnt_silently_create_useless_empty_files(self):
        nonexistent_path = self.tmpdir.getpath(self.id() + '__nothing_here')

        with self.assertRaises(ValueError) as e:
            AssetFinder(nonexistent_path)
        expected = "SQLite file {!r} doesn't exist.".format(nonexistent_path)
        self.assertEqual(str(e.exception), expected)

        # sqlite3.connect will create an empty file if you connect somewhere
        # nonexistent. Test that we don't do that.
        self.assertFalse(os.path.exists(nonexistent_path))


class TestExchangeInfo(ZiplineTestCase):
    def test_equality(self):
        a = ExchangeInfo('FULL NAME', 'E', 'US')
        b = ExchangeInfo('FULL NAME', 'E', 'US')

        assert_equal(a, b)

        # same full name but different canonical name
        c = ExchangeInfo('FULL NAME', 'NOT E', 'US')
        assert_not_equal(c, a)

        # same canonical name but different full name
        d = ExchangeInfo('DIFFERENT FULL NAME', 'E', 'US')
        assert_not_equal(d, a)

        # same names but different country

        e = ExchangeInfo('FULL NAME', 'E', 'JP')
        assert_not_equal(e, a)

    def test_repr(self):
        e = ExchangeInfo('FULL NAME', 'E', 'US')
        assert_equal(repr(e), "ExchangeInfo('FULL NAME', 'E', 'US')")

    def test_read_from_asset_finder(self):
        sids = list(range(8))
        exchange_names = [
            'NEW YORK STOCK EXCHANGE',
            'NEW YORK STOCK EXCHANGE',
            'NASDAQ STOCK MARKET',
            'NASDAQ STOCK MARKET',
            'TOKYO STOCK EXCHANGE',
            'TOKYO STOCK EXCHANGE',
            'OSAKA STOCK EXCHANGE',
            'OSAKA STOCK EXCHANGE',
        ]
        equities = pd.DataFrame({
            'sid': sids,
            'exchange': exchange_names,
            'symbol': [chr(65 + sid) for sid in sids],
        })
        exchange_infos = [
            ExchangeInfo('NEW YORK STOCK EXCHANGE', 'NYSE', 'US'),
            ExchangeInfo('NASDAQ STOCK MARKET', 'NYSE', 'US'),
            ExchangeInfo('TOKYO STOCK EXCHANGE', 'JPX', 'JP'),
            ExchangeInfo('OSAKA STOCK EXCHANGE', 'JPX', 'JP'),
        ]
        exchange_info_table = pd.DataFrame(
            [
                (info.name, info.canonical_name, info.country_code)
                for info in exchange_infos
            ],
            columns=['exchange', 'canonical_name', 'country_code'],
        )
        expected_exchange_info_map = {
            info.name: info for info in exchange_infos
        }

        ctx = tmp_asset_finder(
            equities=equities,
            exchanges=exchange_info_table,
        )
        with ctx as af:
            actual_exchange_info_map = af.exchange_info
            assets = af.retrieve_all(sids)

        assert_equal(actual_exchange_info_map, expected_exchange_info_map)

        for asset in assets:
            expected_exchange_info = expected_exchange_info_map[
                exchange_names[asset.sid]
            ]
            assert_equal(asset.exchange_info, expected_exchange_info)


class TestWrite(WithInstanceTmpDir, ZiplineTestCase):
    def init_instance_fixtures(self):
        super(TestWrite, self).init_instance_fixtures()
        self.assets_db_path = path = os.path.join(
            self.instance_tmpdir.path,
            'assets.db',
        )
        self.writer = AssetDBWriter(path)

    def new_asset_finder(self):
        return AssetFinder(self.assets_db_path)

    def test_write_multiple_exchanges(self):
        # Incrementing by two so that start and end dates for each
        # generated Asset don't overlap (each Asset's end_date is the
        # day after its start date).
        dates = pd.date_range('2013-01-01', freq='2D', periods=5, tz='UTC')
        sids = list(range(5))
        df = pd.DataFrame.from_records(
            [
                {
                    'sid': sid,
                    'symbol':  str(sid),
                    'start_date': date.value,
                    'end_date': (date + timedelta(days=1)).value,

                    # Change the exchange with each mapping period. We don't
                    # currently support point in time exchange information,
                    # so we just take the most recent by end date.
                    'exchange': 'EXCHANGE-%d-%d' % (sid, n),
                }
                for n, date in enumerate(dates)
                for sid in sids
            ]
        )
        self.writer.write(equities=df)

        reader = self.new_asset_finder()
        equities = reader.retrieve_all(reader.sids)

        for eq in equities:
            expected_exchange = 'EXCHANGE-%d-%d' % (eq.sid, len(dates) - 1)
            assert_equal(eq.exchange, expected_exchange)

    def test_write_direct(self):
        # don't include anything with a default to test that those work.
        equities = pd.DataFrame({
            'sid': [0, 1],
            'asset_name': ['Ayy Inc.', 'Lmao LP'],
            # the full exchange name
            'exchange': ['NYSE', 'TSE'],
        })
        equity_symbol_mappings = pd.DataFrame({
            'sid': [0, 1],
            'symbol': ['AYY', 'LMAO'],
            'company_symbol':  ['AYY', 'LMAO'],
            'share_class_symbol': ['', ''],
        })
        equity_supplementary_mappings = pd.DataFrame({
            'sid': [0, 1],
            'field': ['QSIP', 'QSIP'],
            'value': [str(hash(s)) for s in ['AYY', 'LMAO']],
        })
        exchanges = pd.DataFrame({
            'exchange': ['NYSE', 'TSE'],
            'country_code': ['US', 'JP'],
        })

        self.writer.write_direct(
            equities=equities,
            equity_symbol_mappings=equity_symbol_mappings,
            equity_supplementary_mappings=equity_supplementary_mappings,
            exchanges=exchanges,
        )

        reader = self.new_asset_finder()

        equities = reader.retrieve_all(reader.sids)
        expected_equities = [
            Equity(
                0,
                ExchangeInfo('NYSE', 'NYSE', 'US'),
                symbol='AYY',
                asset_name='Ayy Inc.',
                start_date=pd.Timestamp(0, tz='UTC'),
                end_date=pd.Timestamp.max.tz_localize('UTC'),
                first_traded=None,
                auto_close_date=None,
                tick_size=0.01,
                multiplier=1.0,
            ),
            Equity(
                1,
                ExchangeInfo('TSE', 'TSE', 'JP'),
                symbol='LMAO',
                asset_name='Lmao LP',
                start_date=pd.Timestamp(0, tz='UTC'),
                end_date=pd.Timestamp.max.tz_localize('UTC'),
                first_traded=None,
                auto_close_date=None,
                tick_size=0.01,
                multiplier=1.0,
            )
        ]
        assert_equal(equities, expected_equities)

        exchange_info = reader.exchange_info
        expected_exchange_info = {
            'NYSE': ExchangeInfo('NYSE', 'NYSE', 'US'),
            'TSE': ExchangeInfo('TSE', 'TSE', 'JP'),
        }
        assert_equal(exchange_info, expected_exchange_info)

        supplementary_map = reader.equity_supplementary_map
        expected_supplementary_map = {
            ('QSIP', str(hash('AYY'))): (
                OwnershipPeriod(
                    start=pd.Timestamp(0, tz='UTC'),
                    end=pd.Timestamp.max.tz_localize('UTC'),
                    sid=0,
                    value=str(hash('AYY')),
                ),
            ),
            ('QSIP', str(hash('LMAO'))): (
                OwnershipPeriod(
                    start=pd.Timestamp(0, tz='UTC'),
                    end=pd.Timestamp.max.tz_localize('UTC'),
                    sid=1,
                    value=str(hash('LMAO')),
                ),
            ),
        }
        assert_equal(supplementary_map, expected_supplementary_map)
