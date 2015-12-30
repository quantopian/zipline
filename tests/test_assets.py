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
from contextlib import contextmanager
from datetime import datetime, timedelta
import pickle
import sys
from unittest import TestCase
import uuid
import warnings

import pandas as pd
from pandas.tseries.tools import normalize_date
from pandas.util.testing import assert_frame_equal

from nose_parameterized import parameterized
from numpy import full
import sqlalchemy as sa

from zipline.assets import (
    Asset,
    Equity,
    Future,
    AssetFinder,
    AssetFinderCachedEquities,
)
from six import itervalues
from toolz import valmap

from zipline.assets.futures import (
    cme_code_to_month,
    FutureChain,
    month_to_cme_code
)
from zipline.assets.asset_writer import (
    check_version_info,
    write_version_info,
)
from zipline.assets.asset_db_schema import (
    ASSET_DB_VERSION,
    _version_table_schema,
)
from zipline.errors import (
    EquitiesNotFound,
    FutureContractsNotFound,
    MultipleSymbolsFound,
    RootSymbolNotFound,
    AssetDBVersionError,
    SidAssignmentError,
    SidsNotFound,
    SymbolNotFound,
)
from zipline.finance.trading import TradingEnvironment, noop_load
from zipline.utils.test_utils import (
    all_subindices,
    make_commodity_future_info,
    make_rotating_equity_info,
    make_simple_equity_info,
    tmp_assets_db,
    tmp_asset_finder,
)


@contextmanager
def build_lookup_generic_cases(asset_finder_type):
    """
    Generate test cases for the type of asset finder specific by
    asset_finder_type for test_lookup_generic.
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
                'symbol': 'duplicated',
                'start_date': dupe_0_start.value,
                'end_date': dupe_0_end.value,
                'exchange': '',
            },
            {
                'sid': 1,
                'symbol': 'duplicated',
                'start_date': dupe_1_start.value,
                'end_date': dupe_1_end.value,
                'exchange': '',
            },
            {
                'sid': 2,
                'symbol': 'unique',
                'start_date': unique_start.value,
                'end_date': unique_end.value,
                'exchange': '',
            },
        ],
        index='sid')
    with tmp_assets_db(equities=frame) as assets_db:
        finder = asset_finder_type(assets_db)
        dupe_0, dupe_1, unique = assets = [
            finder.retrieve_asset(i)
            for i in range(3)
        ]

        dupe_0_start = dupe_0.start_date
        dupe_1_start = dupe_1.start_date
        yield (
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
            (finder, 'DUPLICATED', dupe_0_start, dupe_0),
            (finder, 'DUPLICATED', dupe_1_start, dupe_1),
            # Unique symbol, with or without resolution date.
            (finder, 'UNIQUE', unique_start, unique),
            (finder, 'UNIQUE', None, unique),

            ##
            # Iterables

            # Iterables of Asset objects.
            (finder, assets, None, assets),
            (finder, iter(assets), None, assets),
            # Iterables of ints
            (finder, (0, 1), None, assets[:-1]),
            (finder, iter((0, 1)), None, assets[:-1]),
            # Iterables of symbols.
            (finder, ('DUPLICATED', 'UNIQUE'), dupe_0_start, [dupe_0, unique]),
            (finder, ('DUPLICATED', 'UNIQUE'), dupe_1_start, [dupe_1, unique]),
            # Mixed types
            (finder,
             ('DUPLICATED', 2, 'UNIQUE', 1, dupe_1),
             dupe_0_start,
             [dupe_0, assets[2], unique, assets[1], dupe_1]),
        )


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

    @classmethod
    def setUpClass(cls):
        cls.future = Future(
            2468,
            symbol='OMH15',
            root_symbol='OM',
            notice_date=pd.Timestamp('2014-01-20', tz='UTC'),
            expiration_date=pd.Timestamp('2014-02-20', tz='UTC'),
            auto_close_date=pd.Timestamp('2014-01-18', tz='UTC'),
            contract_multiplier=500
        )
        cls.future2 = Future(
            0,
            symbol='CLG06',
            root_symbol='CL',
            start_date=pd.Timestamp('2005-12-01', tz='UTC'),
            notice_date=pd.Timestamp('2005-12-20', tz='UTC'),
            expiration_date=pd.Timestamp('2006-01-20', tz='UTC')
        )
        env = TradingEnvironment(load=noop_load)
        env.write_data(futures_identifiers=[TestFuture.future,
                                            TestFuture.future2])
        cls.asset_finder = env.asset_finder

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
        self.assertTrue("auto_close_date=Timestamp('2014-01-18 00:00:00+0000'"
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
        self.assertTrue('auto_close_date' in dictd)
        self.assertTrue('contract_multiplier' in dictd)

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


class AssetFinderTestCase(TestCase):

    def setUp(self):
        self.env = TradingEnvironment(load=noop_load)
        self.asset_finder_type = AssetFinder

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
        self.env.write_data(equities_df=frame)
        finder = self.asset_finder_type(self.env.engine)
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
        metadata = {
            0: {'symbol': 'PRTY_HRD'},
            1: {'symbol': 'BRKA'},
            2: {'symbol': 'BRK_A'},
        }
        self.env.write_data(equities_data=metadata)
        finder = self.env.asset_finder
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
        self.env.write_data(equities_df=df)
        finder = self.asset_finder_type(self.env.engine)
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

    def test_lookup_symbol_from_multiple_valid(self):
        # This test asserts that we resolve conflicts in accordance with the
        # following rules when we have multiple assets holding the same symbol
        # at the same time:

        # If multiple SIDs exist for symbol S at time T, return the candidate
        # SID whose start_date is highest. (200 cases)

        # If multiple SIDs exist for symbol S at time T, the best candidate
        # SIDs share the highest start_date, return the SID with the highest
        # end_date. (34 cases)

        # It is the opinion of the author (ssanderson) that we should consider
        # this malformed input and fail here.  But this is the current indended
        # behavior of the code, and I accidentally broke it while refactoring.
        # These will serve as regression tests until the time comes that we
        # decide to enforce this as an error.

        # See https://github.com/quantopian/zipline/issues/837 for more
        # details.

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

        def check(expected_sid, date):
            result = finder.lookup_symbol(
                'MULTIPLE', date,
            )
            self.assertEqual(result.symbol, 'MULTIPLE')
            self.assertEqual(result.sid, expected_sid)

        with tmp_asset_finder(finder_cls=self.asset_finder_type,
                              equities=df) as finder:
            self.assertIsInstance(finder, self.asset_finder_type)

            # Sids 1 and 2 are eligible here.  We should get asset 2 because it
            # has the later end_date.
            check(2, pd.Timestamp('2010-12-31'))

            # Sids 1, 2, and 3 are eligible here.  We should get sid 3 because
            # it has a later start_date
            check(3, pd.Timestamp('2011-01-01'))

    def test_lookup_generic(self):
        """
        Ensure that lookup_generic works with various permutations of inputs.
        """
        with build_lookup_generic_cases(self.asset_finder_type) as cases:
            for finder, symbols, reference_date, expected in cases:
                results, missing = finder.lookup_generic(symbols,
                                                         reference_date)
                self.assertEqual(results, expected)
                self.assertEqual(missing, [])

    def test_lookup_generic_handle_missing(self):
        data = pd.DataFrame.from_records(
            [
                {
                    'sid': 0,
                    'symbol': 'real',
                    'start_date': pd.Timestamp('2013-1-1', tz='UTC'),
                    'end_date': pd.Timestamp('2014-1-1', tz='UTC'),
                    'exchange': '',
                },
                {
                    'sid': 1,
                    'symbol': 'also_real',
                    'start_date': pd.Timestamp('2013-1-1', tz='UTC'),
                    'end_date': pd.Timestamp('2014-1-1', tz='UTC'),
                    'exchange': '',
                },
                # Sid whose end date is before our query date.  We should
                # still correctly find it.
                {
                    'sid': 2,
                    'symbol': 'real_but_old',
                    'start_date': pd.Timestamp('2002-1-1', tz='UTC'),
                    'end_date': pd.Timestamp('2003-1-1', tz='UTC'),
                    'exchange': '',
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
        self.env.write_data(equities_df=data)
        finder = self.asset_finder_type(self.env.engine)
        results, missing = finder.lookup_generic(
            ['REAL', 1, 'FAKE', 'REAL_BUT_OLD', 'REAL_BUT_IN_THE_FUTURE'],
            pd.Timestamp('2013-02-01', tz='UTC'),
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

    def test_insert_metadata(self):
        data = {0: {'start_date': '2014-01-01',
                    'end_date': '2015-01-01',
                    'symbol': "PLAY",
                    'foo_data': "FOO"}}
        self.env.write_data(equities_data=data)
        finder = self.asset_finder_type(self.env.engine)
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
        dict_to_consume = {0: {'symbol': 'PLAY'},
                           1: {'symbol': 'MSFT'}}
        self.env.write_data(equities_data=dict_to_consume)
        finder = self.asset_finder_type(self.env.engine)

        equity = finder.retrieve_asset(0)
        self.assertIsInstance(equity, Equity)
        self.assertEqual('PLAY', equity.symbol)

        # Test dataframe consumption
        df = pd.DataFrame(columns=['asset_name', 'exchange'], index=[0, 1])
        df['asset_name'][0] = "Dave'N'Busters"
        df['exchange'][0] = "NASDAQ"
        df['asset_name'][1] = "Microsoft"
        df['exchange'][1] = "NYSE"
        self.env = TradingEnvironment(load=noop_load)
        self.env.write_data(equities_df=df)
        finder = self.asset_finder_type(self.env.engine)
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
        self.env.write_data(equities_identifiers=[equity_asset],
                            futures_identifiers=[future_asset])
        finder = self.asset_finder_type(self.env.engine)

        # Test equality with newly built Assets
        self.assertEqual(equity_asset, finder.retrieve_asset(1))
        self.assertEqual(future_asset, finder.retrieve_asset(200))
        self.assertEqual(eq_end, finder.retrieve_asset(1).end_date)
        self.assertEqual(fut_end, finder.retrieve_asset(200).end_date)

    def test_sid_assignment(self):

        # This metadata does not contain SIDs
        metadata = ['PLAY', 'MSFT']

        today = normalize_date(pd.Timestamp('2015-07-09', tz='UTC'))

        # Write data with sid assignment
        self.env.write_data(equities_identifiers=metadata,
                            allow_sid_assignment=True)

        # Verify that Assets were built and different sids were assigned
        finder = self.asset_finder_type(self.env.engine)
        play = finder.lookup_symbol('PLAY', today)
        msft = finder.lookup_symbol('MSFT', today)
        self.assertEqual('PLAY', play.symbol)
        self.assertIsNotNone(play.sid)
        self.assertNotEqual(play.sid, msft.sid)

    def test_sid_assignment_failure(self):

        # This metadata does not contain SIDs
        metadata = ['PLAY', 'MSFT']

        # Write data without sid assignment, asserting failure
        with self.assertRaises(SidAssignmentError):
            self.env.write_data(equities_identifiers=metadata,
                                allow_sid_assignment=False)

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
            # Notice day is today, so should be valid.
            0: {
                'symbol': 'ADN15',
                'root_symbol': 'AD',
                'notice_date': pd.Timestamp('2015-05-14', tz='UTC'),
                'expiration_date': pd.Timestamp('2015-06-14', tz='UTC'),
                'start_date': pd.Timestamp('2015-01-01', tz='UTC')
            },
            1: {
                'symbol': 'ADV15',
                'root_symbol': 'AD',
                'notice_date': pd.Timestamp('2015-08-14', tz='UTC'),
                'expiration_date': pd.Timestamp('2015-09-14', tz='UTC'),
                'start_date': pd.Timestamp('2015-01-01', tz='UTC')
            },
            # Starts trading today, so should be valid.
            2: {
                'symbol': 'ADF16',
                'root_symbol': 'AD',
                'notice_date': pd.Timestamp('2015-11-16', tz='UTC'),
                'expiration_date': pd.Timestamp('2015-12-16', tz='UTC'),
                'start_date': pd.Timestamp('2015-05-14', tz='UTC')
            },
            # Starts trading in August, so not valid.
            3: {
                'symbol': 'ADX16',
                'root_symbol': 'AD',
                'notice_date': pd.Timestamp('2015-11-16', tz='UTC'),
                'expiration_date': pd.Timestamp('2015-12-16', tz='UTC'),
                'start_date': pd.Timestamp('2015-08-01', tz='UTC')
            },
            # Notice date comes after expiration
            4: {
                'symbol': 'ADZ16',
                'root_symbol': 'AD',
                'notice_date': pd.Timestamp('2016-11-25', tz='UTC'),
                'expiration_date': pd.Timestamp('2016-11-16', tz='UTC'),
                'start_date': pd.Timestamp('2015-08-01', tz='UTC')
            },
            # This contract has no start date and also this contract should be
            # last in all chains
            5: {
                'symbol': 'ADZ20',
                'root_symbol': 'AD',
                'notice_date': pd.Timestamp('2020-11-25', tz='UTC'),
                'expiration_date': pd.Timestamp('2020-11-16', tz='UTC')
            },
        }
        self.env.write_data(futures_data=metadata)
        finder = self.asset_finder_type(self.env.engine)
        dt = pd.Timestamp('2015-05-14', tz='UTC')
        dt_2 = pd.Timestamp('2015-10-14', tz='UTC')
        dt_3 = pd.Timestamp('2016-11-17', tz='UTC')

        # Check that we get the expected number of contracts, in the
        # right order
        ad_contracts = finder.lookup_future_chain('AD', dt)
        self.assertEqual(len(ad_contracts), 6)
        self.assertEqual(ad_contracts[0].sid, 0)
        self.assertEqual(ad_contracts[1].sid, 1)
        self.assertEqual(ad_contracts[5].sid, 5)

        # Check that, when some contracts have expired, the chain has advanced
        # properly to the next contracts
        ad_contracts = finder.lookup_future_chain('AD', dt_2)
        self.assertEqual(len(ad_contracts), 4)
        self.assertEqual(ad_contracts[0].sid, 2)
        self.assertEqual(ad_contracts[3].sid, 5)

        # Check that when the expiration_date has passed but the
        # notice_date hasn't, contract is still considered invalid.
        ad_contracts = finder.lookup_future_chain('AD', dt_3)
        self.assertEqual(len(ad_contracts), 1)
        self.assertEqual(ad_contracts[0].sid, 5)

        # Check that pd.NaT for as_of_date gives the whole chain
        ad_contracts = finder.lookup_future_chain('AD', pd.NaT)
        self.assertEqual(len(ad_contracts), 6)
        self.assertEqual(ad_contracts[5].sid, 5)

    def test_map_identifier_index_to_sids(self):
        # Build an empty finder and some Assets
        dt = pd.Timestamp('2014-01-01', tz='UTC')
        finder = self.asset_finder_type(self.env.engine)
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

    def test_compute_lifetimes(self):
        num_assets = 4
        trading_day = self.env.trading_day
        first_start = pd.Timestamp('2015-04-01', tz='UTC')

        frame = make_rotating_equity_info(
            num_assets=num_assets,
            first_start=first_start,
            frequency=self.env.trading_day,
            periods_between_starts=3,
            asset_lifetime=5
        )

        self.env.write_data(equities_df=frame)
        finder = self.env.asset_finder

        all_dates = pd.date_range(
            start=first_start,
            end=frame.end_date.max(),
            freq=trading_day,
        )

        for dates in all_subindices(all_dates):
            expected_with_start_raw = full(
                shape=(len(dates), num_assets),
                fill_value=False,
                dtype=bool,
            )
            expected_no_start_raw = full(
                shape=(len(dates), num_assets),
                fill_value=False,
                dtype=bool,
            )

            for i, date in enumerate(dates):
                it = frame[['start_date', 'end_date']].itertuples()
                for j, start, end in it:
                    # This way of doing the checks is redundant, but very
                    # clear.
                    if start <= date <= end:
                        expected_with_start_raw[i, j] = True
                        if start < date:
                            expected_no_start_raw[i, j] = True

            expected_with_start = pd.DataFrame(
                data=expected_with_start_raw,
                index=dates,
                columns=frame.index.values,
            )
            result = finder.lifetimes(dates, include_start_date=True)
            assert_frame_equal(result, expected_with_start)

            expected_no_start = pd.DataFrame(
                data=expected_no_start_raw,
                index=dates,
                columns=frame.index.values,
            )
            result = finder.lifetimes(dates, include_start_date=False)
            assert_frame_equal(result, expected_no_start)

    def test_sids(self):
        # Ensure that the sids property of the AssetFinder is functioning
        self.env.write_data(equities_identifiers=[1, 2, 3])
        sids = self.env.asset_finder.sids
        self.assertEqual(3, len(sids))
        self.assertTrue(1 in sids)
        self.assertTrue(2 in sids)
        self.assertTrue(3 in sids)

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
        with tmp_asset_finder(equities=equities, futures=futures) as finder:
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

        with tmp_asset_finder(equities=equities, futures=futures) as finder:
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

        with tmp_asset_finder(equities=equities, futures=futures) as finder:
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


class AssetFinderCachedEquitiesTestCase(AssetFinderTestCase):

    def setUp(self):
        self.env = TradingEnvironment(load=noop_load)
        self.asset_finder_type = AssetFinderCachedEquities


class TestFutureChain(TestCase):

    @classmethod
    def setUpClass(cls):
        metadata = {
            0: {
                'symbol': 'CLG06',
                'root_symbol': 'CL',
                'start_date': pd.Timestamp('2005-12-01', tz='UTC'),
                'notice_date': pd.Timestamp('2005-12-20', tz='UTC'),
                'expiration_date': pd.Timestamp('2006-01-20', tz='UTC')},
            1: {
                'root_symbol': 'CL',
                'symbol': 'CLK06',
                'start_date': pd.Timestamp('2005-12-01', tz='UTC'),
                'notice_date': pd.Timestamp('2006-03-20', tz='UTC'),
                'expiration_date': pd.Timestamp('2006-04-20', tz='UTC')},
            2: {
                'symbol': 'CLQ06',
                'root_symbol': 'CL',
                'start_date': pd.Timestamp('2005-12-01', tz='UTC'),
                'notice_date': pd.Timestamp('2006-06-20', tz='UTC'),
                'expiration_date': pd.Timestamp('2006-07-20', tz='UTC')},
            3: {
                'symbol': 'CLX06',
                'root_symbol': 'CL',
                'start_date': pd.Timestamp('2006-02-01', tz='UTC'),
                'notice_date': pd.Timestamp('2006-09-20', tz='UTC'),
                'expiration_date': pd.Timestamp('2006-10-20', tz='UTC')}
        }

        env = TradingEnvironment(load=noop_load)
        env.write_data(futures_data=metadata)
        cls.asset_finder = env.asset_finder

    @classmethod
    def tearDownClass(cls):
        del cls.asset_finder

    def test_len(self):
        """ Test the __len__ method of FutureChain.
        """
        # Sids 0, 1, & 2 have started, 3 has not yet started, but all are in
        # the chain
        cl = FutureChain(self.asset_finder, lambda: '2005-12-01', 'CL')
        self.assertEqual(len(cl), 4)

        # Sid 0 is still valid on its notice date.
        cl = FutureChain(self.asset_finder, lambda: '2005-12-20', 'CL')
        self.assertEqual(len(cl), 4)

        # Sid 0 is now invalid, leaving Sids 1 & 2 valid (and 3 not started).
        cl = FutureChain(self.asset_finder, lambda: '2005-12-21', 'CL')
        self.assertEqual(len(cl), 3)

        # Sid 3 has started, so 1, 2, & 3 are now valid.
        cl = FutureChain(self.asset_finder, lambda: '2006-02-01', 'CL')
        self.assertEqual(len(cl), 3)

        # All contracts are no longer valid.
        cl = FutureChain(self.asset_finder, lambda: '2006-09-21', 'CL')
        self.assertEqual(len(cl), 0)

    def test_getitem(self):
        """ Test the __getitem__ method of FutureChain.
        """
        cl = FutureChain(self.asset_finder, lambda: '2005-12-01', 'CL')
        self.assertEqual(cl[0], 0)
        self.assertEqual(cl[1], 1)
        self.assertEqual(cl[2], 2)

        cl = FutureChain(self.asset_finder, lambda: '2005-12-20', 'CL')
        self.assertEqual(cl[0], 0)

        cl = FutureChain(self.asset_finder, lambda: '2005-12-21', 'CL')
        self.assertEqual(cl[0], 1)

        cl = FutureChain(self.asset_finder, lambda: '2006-02-01', 'CL')
        self.assertEqual(cl[-1], 3)

    def test_iter(self):
        """ Test the __iter__ method of FutureChain.
        """
        cl = FutureChain(self.asset_finder, lambda: '2005-12-01', 'CL')
        for i, contract in enumerate(cl):
            self.assertEqual(contract, i)

        # First contract is now invalid, so sids will be offset by one
        cl = FutureChain(self.asset_finder, lambda: '2005-12-21', 'CL')
        for i, contract in enumerate(cl):
            self.assertEqual(contract, i + 1)

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
                             as_of_date=pd.Timestamp('2006-02-01', tz='UTC'))

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
        feb = pd.Timestamp('2006-02-01', tz='UTC')
        cl_feb = cl.as_of(feb)
        self.assertEqual(
            cl_feb.as_of_date,
            pd.Timestamp(feb, tz='UTC')
        )

        # Test that the as_of_date is set correctly to the past, with
        # args of str, datetime.datetime, and pd.Timestamp.
        feb_prev = pd.Timestamp('2005-02-01', tz='UTC')
        cl_feb_prev = cl.as_of(feb_prev)
        self.assertEqual(
            cl_feb_prev.as_of_date,
            pd.Timestamp(feb_prev, tz='UTC')
        )

        feb_prev = pd.Timestamp(datetime(year=2005, month=2, day=1), tz='UTC')
        cl_feb_prev = cl.as_of(feb_prev)
        self.assertEqual(
            cl_feb_prev.as_of_date,
            pd.Timestamp(feb_prev, tz='UTC')
        )

        feb_prev = pd.Timestamp('2005-02-01', tz='UTC')
        cl_feb_prev = cl.as_of(feb_prev)
        self.assertEqual(
            cl_feb_prev.as_of_date,
            pd.Timestamp(feb_prev, tz='UTC')
        )

        # Test that the as_of() method works with str args
        feb_str = '2006-02-01'
        cl_feb = cl.as_of(feb_str)
        self.assertEqual(
            cl_feb.as_of_date,
            pd.Timestamp(feb, tz='UTC')
        )

        # The chain as of the current dt should always be the same as
        # the defualt chain.
        self.assertEqual(cl[0], cl.as_of(pd.Timestamp('2005-12-01'))[0])

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

    def test_cme_code_to_month(self):
        codes = {
            'F': 1,   # January
            'G': 2,   # February
            'H': 3,   # March
            'J': 4,   # April
            'K': 5,   # May
            'M': 6,   # June
            'N': 7,   # July
            'Q': 8,   # August
            'U': 9,   # September
            'V': 10,  # October
            'X': 11,  # November
            'Z': 12   # December
        }
        for key in codes:
            self.assertEqual(codes[key], cme_code_to_month(key))

    def test_month_to_cme_code(self):
        codes = {
            1: 'F',   # January
            2: 'G',   # February
            3: 'H',   # March
            4: 'J',   # April
            5: 'K',   # May
            6: 'M',   # June
            7: 'N',   # July
            8: 'Q',   # August
            9: 'U',   # September
            10: 'V',  # October
            11: 'X',  # November
            12: 'Z',  # December
        }
        for key in codes:
            self.assertEqual(codes[key], month_to_cme_code(key))


class TestAssetDBVersioning(TestCase):

    def test_check_version(self):
        env = TradingEnvironment(load=noop_load)
        version_table = env.asset_finder.version_info

        # This should not raise an error
        check_version_info(version_table, ASSET_DB_VERSION)

        # This should fail because the version is too low
        with self.assertRaises(AssetDBVersionError):
            check_version_info(version_table, ASSET_DB_VERSION - 1)

        # This should fail because the version is too high
        with self.assertRaises(AssetDBVersionError):
            check_version_info(version_table, ASSET_DB_VERSION + 1)

    def test_write_version(self):
        env = TradingEnvironment(load=noop_load)
        metadata = sa.MetaData(bind=env.engine)
        version_table = _version_table_schema(metadata)
        version_table.delete().execute()

        # Assert that the version is not present in the table
        self.assertIsNone(sa.select((version_table.c.version,)).scalar())

        # This should fail because the table has no version info and is,
        # therefore, consdered v0
        with self.assertRaises(AssetDBVersionError):
            check_version_info(version_table, -2)

        # This should not raise an error because the version has been written
        write_version_info(version_table, -2)
        check_version_info(version_table, -2)

        # Assert that the version is in the table and correct
        self.assertEqual(sa.select((version_table.c.version,)).scalar(), -2)

        # Assert that trying to overwrite the version fails
        with self.assertRaises(sa.exc.IntegrityError):
            write_version_info(version_table, -3)

    def test_finder_checks_version(self):
        # Create an env and give it a bogus version number
        env = TradingEnvironment(load=noop_load)
        metadata = sa.MetaData(bind=env.engine)
        version_table = _version_table_schema(metadata)
        version_table.delete().execute()
        write_version_info(version_table, -2)
        check_version_info(version_table, -2)

        # Assert that trying to build a finder with a bad db raises an error
        with self.assertRaises(AssetDBVersionError):
            AssetFinder(engine=env.engine)

        # Change the version number of the db to the correct version
        version_table.delete().execute()
        write_version_info(version_table, ASSET_DB_VERSION)
        check_version_info(version_table, ASSET_DB_VERSION)

        # Now that the versions match, this Finder should succeed
        AssetFinder(engine=env.engine)
