from abc import ABCMeta, abstractmethod, abstractproperty
from contextlib import contextmanager
from functools import wraps
import gzip
from inspect import getargspec
from itertools import (
    combinations,
    count,
    product,
)
import operator
import os
from os.path import abspath, dirname, join, realpath
import shutil
from sys import _getframe
import tempfile

from logbook import TestHandler
from mock import patch
from nose.tools import nottest
from numpy.testing import assert_allclose, assert_array_equal
import pandas as pd
from six import itervalues, iteritems, with_metaclass
from six.moves import filter, map
from sqlalchemy import create_engine
from testfixtures import TempDirectory
from toolz import concat, curry

from zipline.assets import AssetFinder, AssetDBWriter
from zipline.assets.synthetic import make_simple_equity_info
from zipline.data.data_portal import DataPortal
from zipline.data.minute_bars import (
    BcolzMinuteBarReader,
    BcolzMinuteBarWriter,
    US_EQUITIES_MINUTES_PER_DAY
)
from zipline.data.us_equity_pricing import (
    BcolzDailyBarReader,
    BcolzDailyBarWriter,
    SQLiteAdjustmentWriter,
)
from zipline.finance.blotter import Blotter
from zipline.finance.trading import TradingEnvironment
from zipline.finance.order import ORDER_STATUS
from zipline.lib.labelarray import LabelArray
from zipline.pipeline.data import USEquityPricing
from zipline.pipeline.engine import SimplePipelineEngine
from zipline.pipeline.factors import CustomFactor
from zipline.pipeline.loaders.testing import make_seeded_random_loader
from zipline.utils import security_list
from zipline.utils.calendars import get_calendar
from zipline.utils.input_validation import expect_dimensions
from zipline.utils.numpy_utils import as_column, isnat
from zipline.utils.pandas_utils import timedelta_to_integral_seconds
from zipline.utils.sentinel import sentinel

import numpy as np
from numpy import float64


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
    return timedelta_to_integral_seconds(pd.Timestamp(s, tz='UTC') - EPOCH)


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
            "")

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
    >>> from zipline.testing import all_pairs_matching_predicate
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


def chrange(start, stop):
    """
    Construct an iterable of length-1 strings beginning with `start` and ending
    with `stop`.

    Parameters
    ----------
    start : str
        The first character.
    stop : str
        The last character.

    Returns
    -------
    chars: iterable[str]
        Iterable of strings beginning with start and ending with stop.

    Example
    -------
    >>> chrange('A', 'C')
    ['A', 'B', 'C']
    """
    return list(map(chr, range(ord(start), ord(stop) + 1)))


def make_trade_data_for_asset_info(dates,
                                   asset_info,
                                   price_start,
                                   price_step_by_date,
                                   price_step_by_sid,
                                   volume_start,
                                   volume_step_by_date,
                                   volume_step_by_sid,
                                   frequency,
                                   writer=None):
    """
    Convert the asset info dataframe into a dataframe of trade data for each
    sid, and write to the writer if provided. Write NaNs for locations where
    assets did not exist. Return a dict of the dataframes, keyed by sid.
    """
    trade_data = {}
    sids = asset_info.index

    price_sid_deltas = np.arange(len(sids), dtype=float64) * price_step_by_sid
    price_date_deltas = (np.arange(len(dates), dtype=float64) *
                         price_step_by_date)
    prices = (price_sid_deltas + as_column(price_date_deltas)) + price_start

    volume_sid_deltas = np.arange(len(sids)) * volume_step_by_sid
    volume_date_deltas = np.arange(len(dates)) * volume_step_by_date
    volumes = volume_sid_deltas + as_column(volume_date_deltas) + volume_start

    for j, sid in enumerate(sids):
        start_date, end_date = asset_info.loc[sid, ['start_date', 'end_date']]
        # Normalize here so the we still generate non-NaN values on the minutes
        # for an asset's last trading day.
        for i, date in enumerate(dates.normalize()):
            if not (start_date <= date <= end_date):
                prices[i, j] = 0
                volumes[i, j] = 0

        df = pd.DataFrame(
            {
                "open": prices[:, j],
                "high": prices[:, j],
                "low": prices[:, j],
                "close": prices[:, j],
                "volume": volumes[:, j],
            },
            index=dates,
        )

        if writer:
            writer.write_sid(sid, df)

        trade_data[sid] = df

    return trade_data


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
    return assert_allclose(
        actual,
        desired,
        atol=atol,
        rtol=rtol,
        err_msg=err_msg,
        verbose=verbose,
    )


def check_arrays(x, y, err_msg='', verbose=True, check_dtypes=True):
    """
    Wrapper around np.testing.assert_array_equal that also verifies that inputs
    are ndarrays.

    See Also
    --------
    np.assert_array_equal
    """
    assert type(x) == type(y), "{x} != {y}".format(x=type(x), y=type(y))
    assert x.dtype == y.dtype, "{x.dtype} != {y.dtype}".format(x=x, y=y)

    if isinstance(x, LabelArray):
        # Check that both arrays have missing values in the same locations...
        assert_array_equal(
            x.is_missing(),
            y.is_missing(),
            err_msg=err_msg,
            verbose=verbose,
        )
        # ...then check the actual values as well.
        x = x.as_string_array()
        y = y.as_string_array()
    elif x.dtype.kind in 'mM':
        x_isnat = isnat(x)
        y_isnat = isnat(y)
        assert_array_equal(
            x_isnat,
            y_isnat,
            err_msg="NaTs not equal",
            verbose=verbose,
        )
        # Fill NaTs with zero for comparison.
        x = np.where(x_isnat, np.zeros_like(x), x)
        y = np.where(y_isnat, np.zeros_like(y), y)

    return assert_array_equal(x, y, err_msg=err_msg, verbose=verbose)


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


def write_minute_data(trading_calendar, tempdir, minutes, sids):
    first_session = trading_calendar.minute_to_session_label(
        minutes[0], direction="none"
    )
    last_session = trading_calendar.minute_to_session_label(
        minutes[-1], direction="none"
    )

    sessions = trading_calendar.sessions_in_range(first_session, last_session)

    write_bcolz_minute_data(
        trading_calendar,
        sessions,
        tempdir.path,
        create_minute_bar_data(minutes, sids),
    )
    return tempdir.path


def create_minute_bar_data(minutes, sids):
    length = len(minutes)
    for sid_idx, sid in enumerate(sids):
        yield sid, pd.DataFrame(
            {
                'open': np.arange(length) + 10 + sid_idx,
                'high': np.arange(length) + 15 + sid_idx,
                'low': np.arange(length) + 8 + sid_idx,
                'close': np.arange(length) + 10 + sid_idx,
                'volume': 100 + sid_idx,
            },
            index=minutes,
        )


def create_daily_bar_data(sessions, sids):
    length = len(sessions)
    for sid_idx, sid in enumerate(sids):
        yield sid, pd.DataFrame(
            {
                "open": (np.array(range(10, 10 + length)) + sid_idx),
                "high": (np.array(range(15, 15 + length)) + sid_idx),
                "low": (np.array(range(8, 8 + length)) + sid_idx),
                "close": (np.array(range(10, 10 + length)) + sid_idx),
                "volume": np.array(range(100, 100 + length)) + sid_idx,
                "day": [session.value for session in sessions]
            },
            index=sessions,
        )


def write_daily_data(tempdir, sim_params, sids, trading_calendar):
    path = os.path.join(tempdir.path, "testdaily.bcolz")
    BcolzDailyBarWriter(path, trading_calendar,
                        sim_params.start_session,
                        sim_params.end_session).write(
        create_daily_bar_data(sim_params.sessions, sids),
    )

    return path


def create_data_portal(asset_finder, tempdir, sim_params, sids,
                       trading_calendar, adjustment_reader=None):
    if sim_params.data_frequency == "daily":
        daily_path = write_daily_data(tempdir, sim_params, sids,
                                      trading_calendar)

        equity_daily_reader = BcolzDailyBarReader(daily_path)

        return DataPortal(
            asset_finder, trading_calendar,
            first_trading_day=equity_daily_reader.first_trading_day,
            equity_daily_reader=equity_daily_reader,
            adjustment_reader=adjustment_reader
        )
    else:
        minutes = trading_calendar.minutes_in_range(
            sim_params.first_open,
            sim_params.last_close
        )

        minute_path = write_minute_data(trading_calendar, tempdir, minutes,
                                        sids)

        equity_minute_reader = BcolzMinuteBarReader(minute_path)

        return DataPortal(
            asset_finder, trading_calendar,
            first_trading_day=equity_minute_reader.first_trading_day,
            equity_minute_reader=equity_minute_reader,
            adjustment_reader=adjustment_reader
        )


def write_bcolz_minute_data(trading_calendar, days, path, data):
    BcolzMinuteBarWriter(
        path,
        trading_calendar,
        days[0],
        days[-1],
        US_EQUITIES_MINUTES_PER_DAY
    ).write(data)


def create_minute_df_for_asset(trading_calendar,
                               start_dt,
                               end_dt,
                               interval=1,
                               start_val=1,
                               minute_blacklist=None):

    asset_minutes = trading_calendar.minutes_for_sessions_in_range(
        start_dt, end_dt
    )
    minutes_count = len(asset_minutes)
    minutes_arr = np.array(range(start_val, start_val + minutes_count))

    df = pd.DataFrame(
        {
            "open": minutes_arr + 1,
            "high": minutes_arr + 2,
            "low": minutes_arr - 1,
            "close": minutes_arr,
            "volume": 100 * minutes_arr,
        },
        index=asset_minutes,
    )

    if interval > 1:
        counter = 0
        while counter < len(minutes_arr):
            df[counter:(counter + interval - 1)] = 0
            counter += interval

    if minute_blacklist is not None:
        for minute in minute_blacklist:
            df.loc[minute] = 0

    return df


def create_daily_df_for_asset(trading_calendar, start_day, end_day,
                              interval=1):
    days = trading_calendar.sessions_in_range(start_day, end_day)
    days_count = len(days)
    days_arr = np.arange(days_count) + 2

    df = pd.DataFrame(
        {
            "open": days_arr + 1,
            "high": days_arr + 2,
            "low": days_arr - 1,
            "close": days_arr,
            "volume": days_arr * 100,
        },
        index=days,
    )

    if interval > 1:
        # only keep every 'interval' rows
        for idx, _ in enumerate(days_arr):
            if (idx + 1) % interval != 0:
                df["open"].iloc[idx] = 0
                df["high"].iloc[idx] = 0
                df["low"].iloc[idx] = 0
                df["close"].iloc[idx] = 0
                df["volume"].iloc[idx] = 0

    return df


def trades_by_sid_to_dfs(trades_by_sid, index):
    for sidint, trades in iteritems(trades_by_sid):
        opens = []
        highs = []
        lows = []
        closes = []
        volumes = []
        for trade in trades:
            opens.append(trade.open_price)
            highs.append(trade.high)
            lows.append(trade.low)
            closes.append(trade.close_price)
            volumes.append(trade.volume)

        yield sidint, pd.DataFrame(
            {
                "open": opens,
                "high": highs,
                "low": lows,
                "close": closes,
                "volume": volumes,
            },
            index=index,
        )


def create_data_portal_from_trade_history(asset_finder, trading_calendar,
                                          tempdir, sim_params, trades_by_sid):
    if sim_params.data_frequency == "daily":
        path = os.path.join(tempdir.path, "testdaily.bcolz")
        writer = BcolzDailyBarWriter(
            path, trading_calendar,
            sim_params.start_session,
            sim_params.end_session
        )
        writer.write(
            trades_by_sid_to_dfs(trades_by_sid, sim_params.sessions),
        )

        equity_daily_reader = BcolzDailyBarReader(path)

        return DataPortal(
            asset_finder, trading_calendar,
            first_trading_day=equity_daily_reader.first_trading_day,
            equity_daily_reader=equity_daily_reader,
        )
    else:
        minutes = trading_calendar.minutes_in_range(
            sim_params.first_open,
            sim_params.last_close
        )

        length = len(minutes)
        assets = {}

        for sidint, trades in iteritems(trades_by_sid):
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
                "dt": minutes
            }).set_index("dt")

        write_bcolz_minute_data(
            trading_calendar,
            sim_params.sessions,
            tempdir.path,
            assets
        )

        equity_minute_reader = BcolzMinuteBarReader(tempdir.path)

        return DataPortal(
            asset_finder, trading_calendar,
            first_trading_day=equity_minute_reader.first_trading_day,
            equity_minute_reader=equity_minute_reader,
        )


class FakeDataPortal(DataPortal):
    def __init__(self, env, trading_calendar=None,
                 first_trading_day=None):
        if trading_calendar is None:
            trading_calendar = get_calendar("NYSE")

        super(FakeDataPortal, self).__init__(env.asset_finder,
                                             trading_calendar,
                                             first_trading_day)

    def get_spot_value(self, asset, field, dt, data_frequency):
        if field == "volume":
            return 100
        else:
            return 1.0

    def get_history_window(self, assets, end_dt, bar_count, frequency, field,
                           data_frequency, ffill=True):
        if frequency == "1d":
            end_idx = \
                self.trading_calendar.all_sessions.searchsorted(end_dt)
            days = self.trading_calendar.all_sessions[
                (end_idx - bar_count + 1):(end_idx + 1)
            ]

            df = pd.DataFrame(
                np.full((bar_count, len(assets)), 100.0),
                index=days,
                columns=assets
            )

            return df


class FetcherDataPortal(DataPortal):
    """
    Mock dataportal that returns fake data for history and non-fetcher
    spot value.
    """
    def __init__(self, asset_finder, trading_calendar, first_trading_day=None):
        super(FetcherDataPortal, self).__init__(asset_finder, trading_calendar,
                                                first_trading_day)

    def get_spot_value(self, asset, field, dt, data_frequency):
        # if this is a fetcher field, exercise the regular code path
        if self._is_extra_source(asset, field, self._augmented_sources_map):
            return super(FetcherDataPortal, self).get_spot_value(
                asset, field, dt, data_frequency)

        # otherwise just return a fixed value
        return int(asset)

    # XXX: These aren't actually the methods that are used by the superclasses,
    # so these don't do anything, and this class will likely produce unexpected
    # results for history().
    def _get_daily_window_for_sid(self, asset, field, days_in_window,
                                  extra_slot=True):
        return np.arange(days_in_window, dtype=np.float64)

    def _get_minute_window_for_asset(self, asset, field, minutes_for_window):
        return np.arange(minutes_for_window, dtype=np.float64)


class tmp_assets_db(object):
    """Create a temporary assets sqlite database.
    This is meant to be used as a context manager.

    Parameters
    ----------
    url : string
        The URL for the database connection.
    **frames
        The frames to pass to the AssetDBWriter.
        By default this maps equities:
        ('A', 'B', 'C') -> map(ord, 'ABC')

    See Also
    --------
    empty_assets_db
    tmp_asset_finder
    """
    _default_equities = sentinel('_default_equities')

    def __init__(self,
                 url='sqlite:///:memory:',
                 equities=_default_equities,
                 **frames):
        self._url = url
        self._eng = None
        if equities is self._default_equities:
            equities = make_simple_equity_info(
                list(map(ord, 'ABC')),
                pd.Timestamp(0),
                pd.Timestamp('2015'),
            )

        frames['equities'] = equities
        self._frames = frames
        self._eng = None  # set in enter and exit

    def __enter__(self):
        self._eng = eng = create_engine(self._url)
        AssetDBWriter(eng).write(**self._frames)
        return eng

    def __exit__(self, *excinfo):
        assert self._eng is not None, '_eng was not set in __enter__'
        self._eng.dispose()
        self._eng = None


def empty_assets_db():
    """Context manager for creating an empty assets db.

    See Also
    --------
    tmp_assets_db
    """
    return tmp_assets_db(equities=None)


class tmp_asset_finder(tmp_assets_db):
    """Create a temporary asset finder using an in memory sqlite db.

    Parameters
    ----------
    url : string
        The URL for the database connection.
    finder_cls : type, optional
        The type of asset finder to create from the assets db.
    **frames
        Forwarded to ``tmp_assets_db``.

    See Also
    --------
    tmp_assets_db
    """
    def __init__(self,
                 url='sqlite:///:memory:',
                 finder_cls=AssetFinder,
                 **frames):
        self._finder_cls = finder_cls
        super(tmp_asset_finder, self).__init__(url=url, **frames)

    def __enter__(self):
        return self._finder_cls(super(tmp_asset_finder, self).__enter__())


def empty_asset_finder():
    """Context manager for creating an empty asset finder.

    See Also
    --------
    empty_assets_db
    tmp_assets_db
    tmp_asset_finder
    """
    return tmp_asset_finder(equities=None)


class tmp_trading_env(tmp_asset_finder):
    """Create a temporary trading environment.

    Parameters
    ----------
    finder_cls : type, optional
        The type of asset finder to create from the assets db.
    **frames
        Forwarded to ``tmp_assets_db``.

    See Also
    --------
    empty_trading_env
    tmp_asset_finder
    """
    def __enter__(self):
        return TradingEnvironment(
            asset_db_path=super(tmp_trading_env, self).__enter__().engine,
        )


def empty_trading_env():
    return tmp_trading_env(equities=None)


class SubTestFailures(AssertionError):
    def __init__(self, *failures):
        self.failures = failures

    def __str__(self):
        return 'failures:\n  %s' % '\n  '.join(
            '\n    '.join((
                ', '.join('%s=%r' % item for item in scope.items()),
                '%s: %s' % (type(exc).__name__, exc),
            )) for scope, exc in self.failures,
        )


@nottest
def subtest(iterator, *_names):
    """
    Construct a subtest in a unittest.

    Consider using ``zipline.testing.parameter_space`` when subtests
    are constructed over a single input or over the cross-product of multiple
    inputs.

    ``subtest`` works by decorating a function as a subtest. The decorated
    function will be run by iterating over the ``iterator`` and *unpacking the
    values into the function. If any of the runs fail, the result will be put
    into a set and the rest of the tests will be run. Finally, if any failed,
    all of the results will be dumped as one failure.

    Parameters
    ----------
    iterator : iterable[iterable]
        The iterator of arguments to pass to the function.
    *name : iterator[str]
        The names to use for each element of ``iterator``. These will be used
        to print the scope when a test fails. If not provided, it will use the
        integer index of the value as the name.

    Examples
    --------

    ::

       class MyTest(TestCase):
           def test_thing(self):
               # Example usage inside another test.
               @subtest(([n] for n in range(100000)), 'n')
               def subtest(n):
                   self.assertEqual(n % 2, 0, 'n was not even')
               subtest()

           @subtest(([n] for n in range(100000)), 'n')
           def test_decorated_function(self, n):
               # Example usage to parameterize an entire function.
               self.assertEqual(n % 2, 1, 'n was not odd')

    Notes
    -----
    We use this when we:

    * Will never want to run each parameter individually.
    * Have a large parameter space we are testing
      (see tests/utils/test_events.py).

    ``nose_parameterized.expand`` will create a test for each parameter
    combination which bloats the test output and makes the travis pages slow.

    We cannot use ``unittest2.TestCase.subTest`` because nose, pytest, and
    nose2 do not support ``addSubTest``.

    See Also
    --------
    zipline.testing.parameter_space
    """
    def dec(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            names = _names
            failures = []
            for scope in iterator:
                scope = tuple(scope)
                try:
                    f(*args + scope, **kwargs)
                except Exception as e:
                    if not names:
                        names = count()
                    failures.append((dict(zip(names, scope)), e))
            if failures:
                raise SubTestFailures(*failures)

        return wrapped
    return dec


class MockDailyBarReader(object):
    def get_value(self, col, sid, dt):
        return 100


def create_mock_adjustment_data(splits=None, dividends=None, mergers=None):
    if splits is None:
        splits = create_empty_splits_mergers_frame()
    elif not isinstance(splits, pd.DataFrame):
        splits = pd.DataFrame(splits)

    if mergers is None:
        mergers = create_empty_splits_mergers_frame()
    elif not isinstance(mergers, pd.DataFrame):
        mergers = pd.DataFrame(mergers)

    if dividends is None:
        dividends = create_empty_dividends_frame()
    elif not isinstance(dividends, pd.DataFrame):
        dividends = pd.DataFrame(dividends)

    return splits, mergers, dividends


def create_mock_adjustments(tempdir, days, splits=None, dividends=None,
                            mergers=None):
    path = tempdir.getpath("test_adjustments.db")
    SQLiteAdjustmentWriter(path, MockDailyBarReader(), days).write(
        *create_mock_adjustment_data(splits, dividends, mergers)
    )
    return path


def assert_timestamp_equal(left, right, compare_nat_equal=True, msg=""):
    """
    Assert that two pandas Timestamp objects are the same.

    Parameters
    ----------
    left, right : pd.Timestamp
        The values to compare.
    compare_nat_equal : bool, optional
        Whether to consider `NaT` values equal.  Defaults to True.
    msg : str, optional
        A message to forward to `pd.util.testing.assert_equal`.
    """
    if compare_nat_equal and left is pd.NaT and right is pd.NaT:
        return
    return pd.util.testing.assert_equal(left, right, msg=msg)


def powerset(values):
    """
    Return the power set (i.e., the set of all subsets) of entries in `values`.
    """
    return concat(combinations(values, i) for i in range(len(values) + 1))


def to_series(knowledge_dates, earning_dates):
    """
    Helper for converting a dict of strings to a Series of datetimes.

    This is just for making the test cases more readable.
    """
    return pd.Series(
        index=pd.to_datetime(knowledge_dates),
        data=pd.to_datetime(earning_dates),
    )


def gen_calendars(start, stop, critical_dates):
    """
    Generate calendars to use as inputs.
    """
    all_dates = pd.date_range(start, stop, tz='utc')
    for to_drop in map(list, powerset(critical_dates)):
        # Have to yield tuples.
        yield (all_dates.drop(to_drop),)

    # Also test with the trading calendar.
    trading_days = get_calendar("NYSE").all_days
    yield (trading_days[trading_days.slice_indexer(start, stop)],)


@contextmanager
def temp_pipeline_engine(calendar, sids, random_seed, symbols=None):
    """
    A contextManager that yields a SimplePipelineEngine holding a reference to
    an AssetFinder generated via tmp_asset_finder.

    Parameters
    ----------
    calendar : pd.DatetimeIndex
        Calendar to pass to the constructed PipelineEngine.
    sids : iterable[int]
        Sids to use for the temp asset finder.
    random_seed : int
        Integer used to seed instances of SeededRandomLoader.
    symbols : iterable[str], optional
        Symbols for constructed assets. Forwarded to make_simple_equity_info.
    """
    equity_info = make_simple_equity_info(
        sids=sids,
        start_date=calendar[0],
        end_date=calendar[-1],
        symbols=symbols,
    )

    loader = make_seeded_random_loader(random_seed, calendar, sids)

    def get_loader(column):
        return loader

    with tmp_asset_finder(equities=equity_info) as finder:
        yield SimplePipelineEngine(get_loader, calendar, finder)


def parameter_space(__fail_fast=False, **params):
    """
    Wrapper around subtest that allows passing keywords mapping names to
    iterables of values.

    The decorated test function will be called with the cross-product of all
    possible inputs

    Usage
    -----
    >>> from unittest import TestCase
    >>> class SomeTestCase(TestCase):
    ...     @parameter_space(x=[1, 2], y=[2, 3])
    ...     def test_some_func(self, x, y):
    ...         # Will be called with every possible combination of x and y.
    ...         self.assertEqual(somefunc(x, y), expected_result(x, y))

    See Also
    --------
    zipline.testing.subtest
    """
    def decorator(f):

        argspec = getargspec(f)
        if argspec.varargs:
            raise AssertionError("parameter_space() doesn't support *args")
        if argspec.keywords:
            raise AssertionError("parameter_space() doesn't support **kwargs")
        if argspec.defaults:
            raise AssertionError("parameter_space() doesn't support defaults.")

        # Skip over implicit self.
        argnames = argspec.args
        if argnames[0] == 'self':
            argnames = argnames[1:]

        extra = set(params) - set(argnames)
        if extra:
            raise AssertionError(
                "Keywords %s supplied to parameter_space() are "
                "not in function signature." % extra
            )

        unspecified = set(argnames) - set(params)
        if unspecified:
            raise AssertionError(
                "Function arguments %s were not "
                "supplied to parameter_space()." % extra
            )

        def make_param_sets():
            return product(*(params[name] for name in argnames))

        if __fail_fast:
            @wraps(f)
            def wrapped(self):
                for args in make_param_sets():
                    f(self, *args)
            return wrapped
        else:
            @wraps(f)
            def wrapped(*args, **kwargs):
                subtest(make_param_sets(), *argnames)(f)(*args, **kwargs)

        return wrapped

    return decorator


def create_empty_dividends_frame():
    return pd.DataFrame(
        np.array(
            [],
            dtype=[
                ('ex_date', 'datetime64[ns]'),
                ('pay_date', 'datetime64[ns]'),
                ('record_date', 'datetime64[ns]'),
                ('declared_date', 'datetime64[ns]'),
                ('amount', 'float64'),
                ('sid', 'int32'),
            ],
        ),
        index=pd.DatetimeIndex([], tz='UTC'),
    )


def create_empty_splits_mergers_frame():
    return pd.DataFrame(
        np.array(
            [],
            dtype=[
                ('effective_date', 'int64'),
                ('ratio', 'float64'),
                ('sid', 'int64'),
            ],
        ),
        index=pd.DatetimeIndex([]),
    )


def make_alternating_boolean_array(shape, first_value=True):
    """
    Create a 2D numpy array with the given shape containing alternating values
    of False, True, False, True,... along each row and each column.

    Examples
    --------
    >>> make_alternating_boolean_array((4,4))
    array([[ True, False,  True, False],
           [False,  True, False,  True],
           [ True, False,  True, False],
           [False,  True, False,  True]], dtype=bool)
    >>> make_alternating_boolean_array((4,3), first_value=False)
    array([[False,  True, False],
           [ True, False,  True],
           [False,  True, False],
           [ True, False,  True]], dtype=bool)
    """
    if len(shape) != 2:
        raise ValueError(
            'Shape must be 2-dimensional. Given shape was {}'.format(shape)
        )
    alternating = np.empty(shape, dtype=np.bool)
    for row in alternating:
        row[::2] = first_value
        row[1::2] = not(first_value)
        first_value = not(first_value)
    return alternating


def make_cascading_boolean_array(shape, first_value=True):
    """
    Create a numpy array with the given shape containing cascading boolean
    values, with `first_value` being the top-left value.

    Examples
    --------
    >>> make_cascading_boolean_array((4,4))
    array([[ True,  True,  True, False],
           [ True,  True, False, False],
           [ True, False, False, False],
           [False, False, False, False]], dtype=bool)
    >>> make_cascading_boolean_array((4,2))
    array([[ True, False],
           [False, False],
           [False, False],
           [False, False]], dtype=bool)
    >>> make_cascading_boolean_array((2,4))
    array([[ True,  True,  True, False],
           [ True,  True, False, False]], dtype=bool)
    """
    if len(shape) != 2:
        raise ValueError(
            'Shape must be 2-dimensional. Given shape was {}'.format(shape)
        )
    cascading = np.full(shape, not(first_value), dtype=np.bool)
    ending_col = shape[1] - 1
    for row in cascading:
        if ending_col > 0:
            row[:ending_col] = first_value
            ending_col -= 1
        else:
            break
    return cascading


@expect_dimensions(array=2)
def permute_rows(seed, array):
    """
    Shuffle each row in ``array`` based on permutations generated by ``seed``.

    Parameters
    ----------
    seed : int
        Seed for numpy.RandomState
    array : np.ndarray[ndim=2]
        Array over which to apply permutations.
    """
    rand = np.random.RandomState(seed)
    return np.apply_along_axis(rand.permutation, 1, array)


@nottest
def make_test_handler(testcase, *args, **kwargs):
    """
    Returns a TestHandler which will be used by the given testcase. This
    handler can be used to test log messages.

    Parameters
    ----------
    testcase: unittest.TestCase
        The test class in which the log handler will be used.
    *args, **kwargs
        Forwarded to the new TestHandler object.

    Returns
    -------
    handler: logbook.TestHandler
        The handler to use for the test case.
    """
    handler = TestHandler(*args, **kwargs)
    testcase.addCleanup(handler.close)
    return handler


def write_compressed(path, content):
    """
    Write a compressed (gzipped) file to `path`.
    """
    with gzip.open(path, 'wb') as f:
        f.write(content)


def read_compressed(path):
    """
    Write a compressed (gzipped) file from `path`.
    """
    with gzip.open(path, 'rb') as f:
        return f.read()


zipline_git_root = abspath(
    join(realpath(dirname(__file__)), '..', '..'),
)


@nottest
def test_resource_path(*path_parts):
    return os.path.join(zipline_git_root, 'tests', 'resources', *path_parts)


@contextmanager
def patch_os_environment(remove=None, **values):
    """
    Context manager for patching the operating system environment.
    """
    old_values = {}
    remove = remove or []
    for key in remove:
        old_values[key] = os.environ.pop(key)
    for key, value in values.iteritems():
        old_values[key] = os.getenv(key)
        os.environ[key] = value
    try:
        yield
    finally:
        for old_key, old_value in old_values.iteritems():
            if old_value is None:
                # Value was not present when we entered, so del it out if it's
                # still present.
                try:
                    del os.environ[key]
                except KeyError:
                    pass
            else:
                # Restore the old value.
                os.environ[old_key] = old_value


class tmp_dir(TempDirectory, object):
    """New style class that wrapper for TempDirectory in python 2.
    """
    pass


class _TmpBarReader(with_metaclass(ABCMeta, tmp_dir)):
    """A helper for tmp_bcolz_equity_minute_bar_reader and
    tmp_bcolz_equity_daily_bar_reader.

    Parameters
    ----------
    env : TradingEnvironment
        The trading env.
    days : pd.DatetimeIndex
        The days to write for.
    data : dict[int -> pd.DataFrame]
        The data to write.
    path : str, optional
        The path to the directory to write the data into. If not given, this
        will be a unique name.
    """
    @abstractproperty
    def _reader_cls(self):
        raise NotImplementedError('_reader')

    @abstractmethod
    def _write(self, env, days, path, data):
        raise NotImplementedError('_write')

    def __init__(self, env, days, data, path=None):
        super(_TmpBarReader, self).__init__(path=path)
        self._env = env
        self._days = days
        self._data = data

    def __enter__(self):
        tmpdir = super(_TmpBarReader, self).__enter__()
        env = self._env
        try:
            self._write(
                env,
                self._days,
                tmpdir.path,
                self._data,
            )
            return self._reader_cls(tmpdir.path)
        except:
            self.__exit__(None, None, None)
            raise


class tmp_bcolz_equity_minute_bar_reader(_TmpBarReader):
    """A temporary BcolzMinuteBarReader object.

    Parameters
    ----------
    env : TradingEnvironment
        The trading env.
    days : pd.DatetimeIndex
        The days to write for.
    data : iterable[(int, pd.DataFrame)]
        The data to write.
    path : str, optional
        The path to the directory to write the data into. If not given, this
        will be a unique name.

    See Also
    --------
    tmp_bcolz_equity_daily_bar_reader
    """
    _reader_cls = BcolzMinuteBarReader
    _write = staticmethod(write_bcolz_minute_data)


class tmp_bcolz_equity_daily_bar_reader(_TmpBarReader):
    """A temporary BcolzDailyBarReader object.

    Parameters
    ----------
    env : TradingEnvironment
        The trading env.
    days : pd.DatetimeIndex
        The days to write for.
    data : dict[int -> pd.DataFrame]
        The data to write.
    path : str, optional
        The path to the directory to write the data into. If not given, this
        will be a unique name.

    See Also
    --------
    tmp_bcolz_equity_daily_bar_reader
    """
    _reader_cls = BcolzDailyBarReader

    @staticmethod
    def _write(env, days, path, data):
        BcolzDailyBarWriter(path, days).write(data)


@contextmanager
def patch_read_csv(url_map, module=pd, strict=False):
    """Patch pandas.read_csv to map lookups from url to another.

    Parameters
    ----------
    url_map : mapping[str or file-like object -> str or file-like object]
        The mapping to use to redirect read_csv calls.
    module : module, optional
        The module to patch ``read_csv`` on. By default this is ``pandas``.
        This should be set to another module if ``read_csv`` is early-bound
        like ``from pandas import read_csv`` instead of late-bound like:
        ``import pandas as pd; pd.read_csv``.
    strict : bool, optional
        If true, then this will assert that ``read_csv`` is only called with
        elements in the ``url_map``.
    """
    read_csv = pd.read_csv

    def patched_read_csv(filepath_or_buffer, *args, **kwargs):
        if filepath_or_buffer in url_map:
            return read_csv(url_map[filepath_or_buffer], *args, **kwargs)
        elif not strict:
            return read_csv(filepath_or_buffer, *args, **kwargs)
        else:
            raise AssertionError(
                'attempted to call read_csv on  %r which not in the url map' %
                filepath_or_buffer,
            )

    with patch.object(module, 'read_csv', patched_read_csv):
        yield


@curry
def ensure_doctest(f, name=None):
    """Ensure that an object gets doctested. This is useful for instances
    of objects like curry or partial which are not discovered by default.

    Parameters
    ----------
    f : any
        The thing to doctest.
    name : str, optional
        The name to use in the doctest function mapping. If this is None,
        Then ``f.__name__`` will be used.

    Returns
    -------
    f : any
       ``f`` unchanged.
    """
    _getframe(2).f_globals.setdefault('__test__', {})[
        f.__name__ if name is None else name
    ] = f
    return f


class RecordBatchBlotter(Blotter):
    """Blotter that tracks how its batch_order method was called.
    """
    def __init__(self, data_frequency):
        super(RecordBatchBlotter, self).__init__(data_frequency)
        self.order_batch_called = []

    def batch_order(self, *args, **kwargs):
        self.order_batch_called.append((args, kwargs))
        return super(RecordBatchBlotter, self).batch_order(*args, **kwargs)


####################################
# Shared factors for pipeline tests.
####################################

class AssetID(CustomFactor):
    """
    CustomFactor that returns the AssetID of each asset.

    Useful for providing a Factor that produces a different value for each
    asset.
    """
    window_length = 1
    inputs = ()

    def compute(self, today, assets, out):
        out[:] = assets


class AssetIDPlusDay(CustomFactor):
    window_length = 1
    inputs = ()

    def compute(self, today, assets, out):
        out[:] = assets + today.day


class OpenPrice(CustomFactor):
    window_length = 1
    inputs = [USEquityPricing.open]

    def compute(self, today, assets, out, open):
        out[:] = open
