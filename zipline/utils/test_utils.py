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

from contextlib import contextmanager
from itertools import (
    product,
)
import operator
import os
import shutil
from string import ascii_uppercase
import tempfile

from logbook import FileHandler
from mock import patch
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from zipline.data.data_portal import DataPortal
from zipline.data.minute_writer import MinuteBarWriterFromDataFrames
from zipline.finance.order import ORDER_STATUS
from zipline.utils import security_list
import pandas as pd
from six import (
    itervalues,
    iteritems)
from six.moves import filter
from sqlalchemy import create_engine

from zipline.assets import AssetFinder
from zipline.assets.asset_writer import AssetDBWriterFromDataFrame


from numpy import (
    float64,
    uint32
)
from bcolz import ctable

from zipline.data.us_equity_pricing import (
    BcolzDailyBarWriter,
    OHLC,
    UINT32_MAX,
    SQLiteAdjustmentWriter)


EPOCH = pd.Timestamp(0, tz='UTC')


def seconds_to_timestamp(seconds):
    return pd.Timestamp(seconds, unit='s', tz='UTC')


def to_utc(time_str):
    """Convert a string in US/Eastern time to UTC"""
    return pd.Timestamp(time_str, tz='US/Eastern').tz_convert('UTC')


def str_to_seconds(s):
    """
    Convert a pandas-intelligible string to (integer) seconds since UTC.

    >>> from pandas import Timestamp
    >>> (Timestamp('2014-01-01') - Timestamp(0)).total_seconds()
    1388534400.0
    >>> str_to_seconds('2014-01-01')
    1388534400
    """
    return int((pd.Timestamp(s, tz='UTC') - EPOCH).total_seconds())


def setup_logger(test, path='test.log'):
    test.log_handler = FileHandler(path)
    test.log_handler.push_application()


def teardown_logger(test):
    test.log_handler.pop_application()
    test.log_handler.close()


def drain_zipline(test, zipline):
    output = []
    transaction_count = 0
    msg_counter = 0
    # start the simulation
    for update in zipline:
        msg_counter += 1
        output.append(update)
        if 'daily_perf' in update:
            transaction_count += \
                len(update['daily_perf']['transactions'])

    return output, transaction_count


def check_algo_results(test,
                       results,
                       expected_transactions_count=None,
                       expected_order_count=None,
                       expected_positions_count=None,
                       sid=None):

    if expected_transactions_count is not None:
        txns = flatten_list(results["transactions"])
        test.assertEqual(expected_transactions_count, len(txns))

    if expected_positions_count is not None:
        raise NotImplementedError

    if expected_order_count is not None:
        # de-dup orders on id, because orders are put back into perf packets
        # whenever they a txn is filled
        orders = set([order['id'] for order in
                      flatten_list(results["orders"])])

        test.assertEqual(expected_order_count, len(orders))


def flatten_list(list):
    return [item for sublist in list for item in sublist]


def assert_single_position(test, zipline):

    output, transaction_count = drain_zipline(test, zipline)

    if 'expected_transactions' in test.zipline_test_config:
        test.assertEqual(
            test.zipline_test_config['expected_transactions'],
            transaction_count
        )
    else:
        test.assertEqual(
            test.zipline_test_config['order_count'],
            transaction_count
        )

    # the final message is the risk report, the second to
    # last is the final day's results. Positions is a list of
    # dicts.
    closing_positions = output[-2]['daily_perf']['positions']

    # confirm that all orders were filled.
    # iterate over the output updates, overwriting
    # orders when they are updated. Then check the status on all.
    orders_by_id = {}
    for update in output:
        if 'daily_perf' in update:
            if 'orders' in update['daily_perf']:
                for order in update['daily_perf']['orders']:
                    orders_by_id[order['id']] = order

    for order in itervalues(orders_by_id):
        test.assertEqual(
            order['status'],
            ORDER_STATUS.FILLED,
            order)

    test.assertEqual(
        len(closing_positions),
        1,
        "Portfolio should have one position."
    )

    sid = test.zipline_test_config['sid']
    test.assertEqual(
        closing_positions[0]['sid'],
        sid,
        "Portfolio should have one position in " + str(sid)
    )

    return output, transaction_count


class ExceptionSource(object):

    def __init__(self):
        pass

    def get_hash(self):
        return "ExceptionSource"

    def __iter__(self):
        return self

    def next(self):
        5 / 0

    def __next__(self):
        5 / 0


@contextmanager
def security_list_copy():
    old_dir = security_list.SECURITY_LISTS_DIR
    new_dir = tempfile.mkdtemp()
    try:
        for subdir in os.listdir(old_dir):
            shutil.copytree(os.path.join(old_dir, subdir),
                            os.path.join(new_dir, subdir))
            with patch.object(security_list, 'SECURITY_LISTS_DIR', new_dir), \
                    patch.object(security_list, 'using_copy', True,
                                 create=True):
                yield
    finally:
        shutil.rmtree(new_dir, True)


def add_security_data(adds, deletes):
    if not hasattr(security_list, 'using_copy'):
        raise Exception('add_security_data must be used within '
                        'security_list_copy context')
    directory = os.path.join(
        security_list.SECURITY_LISTS_DIR,
        "leveraged_etf_list/20150127/20150125"
    )
    if not os.path.exists(directory):
        os.makedirs(directory)
    del_path = os.path.join(directory, "delete")
    with open(del_path, 'w') as f:
        for sym in deletes:
            f.write(sym)
            f.write('\n')
    add_path = os.path.join(directory, "add")
    with open(add_path, 'w') as f:
        for sym in adds:
            f.write(sym)
            f.write('\n')


def all_pairs_matching_predicate(values, pred):
    """
    Return an iterator of all pairs, (v0, v1) from values such that

    `pred(v0, v1) == True`

    Parameters
    ----------
    values : iterable
    pred : function

    Returns
    -------
    pairs_iterator : generator
       Generator yielding pairs matching `pred`.

    Examples
    --------
    >>> from zipline.utils.test_utils import all_pairs_matching_predicate
    >>> from operator import eq, lt
    >>> list(all_pairs_matching_predicate(range(5), eq))
    [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
    >>> list(all_pairs_matching_predicate("abcd", lt))
    [('a', 'b'), ('a', 'c'), ('a', 'd'), ('b', 'c'), ('b', 'd'), ('c', 'd')]
    """
    return filter(lambda pair: pred(*pair), product(values, repeat=2))


def product_upper_triangle(values, include_diagonal=False):
    """
    Return an iterator over pairs, (v0, v1), drawn from values.

    If `include_diagonal` is True, returns all pairs such that v0 <= v1.
    If `include_diagonal` is False, returns all pairs such that v0 < v1.
    """
    return all_pairs_matching_predicate(
        values,
        operator.le if include_diagonal else operator.lt,
    )


def all_subindices(index):
    """
    Return all valid sub-indices of a pandas Index.
    """
    return (
        index[start:stop]
        for start, stop in product_upper_triangle(range(len(index) + 1))
    )


def make_rotating_asset_info(num_assets,
                             first_start,
                             frequency,
                             periods_between_starts,
                             asset_lifetime):
    """
    Create a DataFrame representing lifetimes of assets that are constantly
    rotating in and out of existence.

    Parameters
    ----------
    num_assets : int
        How many assets to create.
    first_start : pd.Timestamp
        The start date for the first asset.
    frequency : str or pd.tseries.offsets.Offset (e.g. trading_day)
        Frequency used to interpret next two arguments.
    periods_between_starts : int
        Create a new asset every `frequency` * `periods_between_new`
    asset_lifetime : int
        Each asset exists for `frequency` * `asset_lifetime` days.

    Returns
    -------
    info : pd.DataFrame
        DataFrame representing newly-created assets.
    """
    return pd.DataFrame(
        {
            'sid': range(num_assets),
            'symbol': [chr(ord('A') + i) for i in range(num_assets)],
            'asset_type': ['equity'] * num_assets,
            # Start a new asset every `periods_between_starts` days.
            'start_date': pd.date_range(
                first_start,
                freq=(periods_between_starts * frequency),
                periods=num_assets,
            ),
            # Each asset lasts for `asset_lifetime` days.
            'end_date': pd.date_range(
                first_start + (asset_lifetime * frequency),
                freq=(periods_between_starts * frequency),
                periods=num_assets,
            ),
            'exchange': 'TEST',
        }
    )


def make_simple_asset_info(assets, start_date, end_date, symbols=None):
    """
    Create a DataFrame representing assets that exist for the full duration
    between `start_date` and `end_date`.

    Parameters
    ----------
    assets : array-like
    start_date : pd.Timestamp
    end_date : pd.Timestamp
    symbols : list, optional
        Symbols to use for the assets.
        If not provided, symbols are generated from the sequence 'A', 'B', ...

    Returns
    -------
    info : pd.DataFrame
        DataFrame representing newly-created assets.
    """
    num_assets = len(assets)
    if symbols is None:
        symbols = list(ascii_uppercase[:num_assets])
    return pd.DataFrame(
        {
            'sid': assets,
            'symbol': symbols,
            'asset_type': ['equity'] * num_assets,
            'start_date': [start_date] * num_assets,
            'end_date': [end_date] * num_assets,
            'exchange': 'TEST',
        }
    )


def check_allclose(actual,
                   desired,
                   rtol=1e-07,
                   atol=0,
                   err_msg='',
                   verbose=True):
    """
    Wrapper around np.testing.assert_allclose that also verifies that inputs
    are ndarrays.

    See Also
    --------
    np.assert_allclose
    """
    if type(actual) != type(desired):
        raise AssertionError("%s != %s" % (type(actual), type(desired)))
    return assert_allclose(actual, desired, err_msg=err_msg, verbose=True)


def check_arrays(x, y, err_msg='', verbose=True):
    """
    Wrapper around np.testing.assert_array_equal that also verifies that inputs
    are ndarrays.

    See Also
    --------
    np.assert_array_equal
    """
    if type(x) != type(y):
        raise AssertionError("%s != %s" % (type(x), type(y)))
    return assert_array_equal(x, y, err_msg=err_msg, verbose=True)


class UnexpectedAttributeAccess(Exception):
    pass


class ExplodingObject(object):
    """
    Object that will raise an exception on any attribute access.

    Useful for verifying that an object is never touched during a
    function/method call.
    """
    def __getattribute__(self, name):
        raise UnexpectedAttributeAccess(name)


class DailyBarWriterFromDataFrames(BcolzDailyBarWriter):
    _csv_dtypes = {
        'open': float64,
        'high': float64,
        'low': float64,
        'close': float64,
        'volume': float64,
    }

    def __init__(self, asset_map):
        self._asset_map = asset_map

    def gen_tables(self, assets):
        for asset in assets:
            yield asset, ctable.fromdataframe(assets[asset])

    def to_uint32(self, array, colname):
        arrmax = array.max()
        if colname in OHLC:
            self.check_uint_safe(arrmax * 1000, colname)
            return (array * 1000).astype(uint32)
        elif colname == 'volume':
            self.check_uint_safe(arrmax, colname)
            return array.astype(uint32)
        elif colname == 'day':
            nanos_per_second = (1000 * 1000 * 1000)
            self.check_uint_safe(arrmax.view(int) / nanos_per_second, colname)
            return (array.view(int) / nanos_per_second).astype(uint32)

    @staticmethod
    def check_uint_safe(value, colname):
        if value >= UINT32_MAX:
            raise ValueError(
                "Value %s from column '%s' is too large" % (value, colname)
            )


def write_minute_data(tempdir, minutes, sids, sid_path_func=None):
    assets = {}

    length = len(minutes)

    for sid_idx, sid in enumerate(sids):
        assets[sid] = pd.DataFrame({
            "open": (np.array(range(10, 10 + length)) + sid_idx) * 1000,
            "high": (np.array(range(15, 15 + length)) + sid_idx) * 1000,
            "low": (np.array(range(8, 8 + length)) + sid_idx) * 1000,
            "close": (np.array(range(10, 10 + length)) + sid_idx) * 1000,
            "volume": np.array(range(100, 100 + length)) + sid_idx,
            "minute": minutes
        }, index=minutes)

    MinuteBarWriterFromDataFrames().write(tempdir.path, assets,
                                          sid_path_func=sid_path_func)

    return tempdir.path


def write_daily_data(tempdir, sim_params, sids):
    path = os.path.join(tempdir.path, "testdaily.bcolz")
    assets = {}
    length = sim_params.days_in_period
    for sid_idx, sid in enumerate(sids):
        assets[sid] = pd.DataFrame({
            "open": (np.array(range(10, 10 + length)) + sid_idx),
            "high": (np.array(range(15, 15 + length)) + sid_idx),
            "low": (np.array(range(8, 8 + length)) + sid_idx),
            "close": (np.array(range(10, 10 + length)) + sid_idx),
            "volume": np.array(range(100, 100 + length)) + sid_idx,
            "day": [day.value for day in sim_params.trading_days]
        }, index=sim_params.trading_days)

    DailyBarWriterFromDataFrames(assets).write(
        path,
        sim_params.trading_days,
        assets
    )

    return path


def create_data_portal(env, tempdir, sim_params, sids, sid_path_func=None,
                       adjustment_reader=None):
    if sim_params.data_frequency == "daily":
        daily_path = write_daily_data(tempdir, sim_params, sids)

        return DataPortal(
            env,
            daily_equities_path=daily_path,
            sim_params=sim_params,
            asset_finder=env.asset_finder,
            adjustment_reader=adjustment_reader
        )
    else:
        minutes = env.minutes_for_days_in_range(
            sim_params.first_open,
            sim_params.last_close
        )

        minute_path = write_minute_data(tempdir, minutes, sids,
                                        sid_path_func)

        return DataPortal(
            env,
            minutes_equities_path=minute_path,
            sim_params=sim_params,
            asset_finder=env.asset_finder,
            adjustment_reader=adjustment_reader
        )


def create_data_portal_from_trade_history(env, tempdir, sim_params,
                                          trades_by_sid):
    if sim_params.data_frequency == "daily":
        path = os.path.join(tempdir.path, "testdaily.bcolz")
        assets = {}
        for sidint, trades in iteritems(trades_by_sid):
            opens = []
            highs = []
            lows = []
            closes = []
            volumes = []
            for trade in trades:
                opens.append(trade["open_price"])
                highs.append(trade["high"])
                lows.append(trade["low"])
                closes.append(trade["close_price"])
                volumes.append(trade["volume"])

            assets[sidint] = pd.DataFrame({
                "open": np.array(opens),
                "high": np.array(highs),
                "low": np.array(lows),
                "close": np.array(closes),
                "volume": np.array(volumes),
                "day": [day.value for day in sim_params.trading_days]
            }, index=sim_params.trading_days)

        DailyBarWriterFromDataFrames(assets).write(
            path,
            sim_params.trading_days,
            assets
        )

        return DataPortal(
            env,
            daily_equities_path=path,
            sim_params=sim_params,
            asset_finder=env.asset_finder
        )
    else:
        minutes = env.minutes_for_days_in_range(
            sim_params.first_open,
            sim_params.last_close
        )

        length = len(minutes)
        assets = {}

        for sidint, trades in trades_by_sid.iteritems():
            opens = np.zeros(length)
            highs = np.zeros(length)
            lows = np.zeros(length)
            closes = np.zeros(length)
            volumes = np.zeros(length)

            for trade in trades:
                # put them in the right place
                idx = minutes.searchsorted(trade.dt)

                opens[idx] = trade.open_price * 1000
                highs[idx] = trade.high * 1000
                lows[idx] = trade.low * 1000
                closes[idx] = trade.close_price * 1000
                volumes[idx] = trade.volume

            assets[sidint] = pd.DataFrame({
                "open": opens,
                "high": highs,
                "low": lows,
                "close": closes,
                "volume": volumes,
                "minute": minutes
            }, index=minutes)

        MinuteBarWriterFromDataFrames().write(tempdir.path, assets)

        return DataPortal(
            env,
            minutes_equities_path=tempdir.path,
            sim_params=sim_params,
            asset_finder=env.asset_finder
        )


class FakeDataPortal(object):

    def __init__(self):
        self._adjustment_reader = None


class tmp_assets_db(object):
    """Create a temporary assets sqlite database.
    This is meant to be used as a context manager.

    Paramaters
    ----------
    data : pd.DataFrame, optional
        The data to feed to the writer. By default this maps:
        ('A', 'B', 'C') -> map(ord, 'ABC')
    """
    def __init__(self, data=None, start_date=None, end_date=None):
        self._eng = None
        if start_date is None:
            start_date = pd.Timestamp(0)

        if end_date is None:
            end_date = pd.Timestamp(2015)

        if data is None:
            data = make_simple_asset_info(
                list(map(ord, 'ABC')),
                start_date,
                end_date,
            )

        self._data = AssetDBWriterFromDataFrame(data)

    def __enter__(self):
        self._eng = eng = create_engine('sqlite://')
        self._data.write_all(eng)
        return eng

    def __exit__(self, *excinfo):
        assert self._eng is not None, '_eng was not set in __enter__'
        self._eng.dispose()


class tmp_asset_finder(tmp_assets_db):
    """Create a temporary asset finder using an in memory sqlite db.

    Parameters
    ----------
    data : dict, optional
        The data to feed to the writer
    """
    def __enter__(self):
        return AssetFinder(super(tmp_asset_finder, self).__enter__())


class MockDailyBarReader(object):
    def spot_price(self, col, sid, dt):
        return 100


def create_mock_adjustments(tempdir, days, splits=None, dividends=None,
                       mergers=None):
    path = tempdir.getpath("test_adjustments.db")

    # create a split for the last day
    writer = SQLiteAdjustmentWriter(path, days, MockDailyBarReader())
    if splits is None:
        splits = pd.DataFrame({
             # Hackery to make the dtypes correct on an empty frame.
            'effective_date': np.array([], dtype=int),
            'ratio': np.array([], dtype=float),
            'sid': np.array([], dtype=int),
        }, index=pd.DatetimeIndex([], tz='UTC'))
    else:
        splits = pd.DataFrame(splits)

    if mergers is None:
        mergers = pd.DataFrame({
             # Hackery to make the dtypes correct on an empty frame.
            'effective_date': np.array([], dtype=int),
            'ratio': np.array([], dtype=float),
            'sid': np.array([], dtype=int),
        }, index=pd.DatetimeIndex([], tz='UTC'))
    else:
        mergers = pd.DataFrame(mergers)

    if dividends is None:
        dividends = pd.DataFrame({
                # Hackery to make the dtypes correct on an empty frame.
                'ex_date': np.array([], dtype='datetime64[ns]'),
                'pay_date': np.array([], dtype='datetime64[ns]'),
                'record_date': np.array([], dtype='datetime64[ns]'),
                'declared_date': np.array([], dtype='datetime64[ns]'),
                'amount': np.array([], dtype=float),
                'sid': np.array([], dtype=int),
            },
            index=pd.DatetimeIndex([], tz='UTC'),
            columns=['ex_date',
                     'pay_date',
                     'record_date',
                     'declared_date',
                     'amount',
                     'sid']
        )
    else:
        if not isinstance(dividends, pd.DataFrame):
            dividends = pd.DataFrame(dividends)

    writer.write(splits, mergers, dividends)

    return path
