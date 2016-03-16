from contextlib import contextmanager
from functools import wraps
from inspect import getargspec
from itertools import (
    combinations,
    count,
    product,
)
from nose.tools import nottest
import operator
import os
import shutil
from string import ascii_uppercase
import tempfile
from bcolz import ctable

from logbook import FileHandler, TestHandler
from mock import patch
from numpy.testing import assert_allclose, assert_array_equal
import pandas as pd
from pandas.tseries.offsets import MonthBegin
from six import iteritems, itervalues
from six.moves import filter, map
from sqlalchemy import create_engine
from toolz import concat

from zipline.assets import AssetFinder
from zipline.assets.asset_writer import AssetDBWriterFromDataFrame
from zipline.assets.futures import CME_CODE_TO_MONTH
from zipline.data.data_portal import DataPortal
from zipline.data.minute_bars import (
    BcolzMinuteBarReader,
    BcolzMinuteBarWriter,
    US_EQUITIES_MINUTES_PER_DAY
)
from zipline.data.us_equity_pricing import SQLiteAdjustmentWriter, OHLC, \
    UINT32_MAX, BcolzDailyBarWriter, BcolzDailyBarReader
from zipline.finance.order import ORDER_STATUS
from zipline.pipeline.engine import SimplePipelineEngine
from zipline.pipeline.loaders.testing import make_seeded_random_loader
from zipline.finance.trading import TradingEnvironment
from zipline.utils import security_list
from zipline.utils.tradingcalendar import trading_days
import numpy as np
from numpy import (
    float64,
    uint32
)


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


def make_rotating_equity_info(num_assets,
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
            'symbol': [chr(ord('A') + i) for i in range(num_assets)],
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
        },
        index=range(num_assets),
    )


def make_simple_equity_info(sids, start_date, end_date, symbols=None):
    """
    Create a DataFrame representing assets that exist for the full duration
    between `start_date` and `end_date`.

    Parameters
    ----------
    sids : array-like of int
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
    num_assets = len(sids)
    if symbols is None:
        symbols = list(ascii_uppercase[:num_assets])
    return pd.DataFrame(
        {
            'symbol': symbols,
            'start_date': [start_date] * num_assets,
            'end_date': [end_date] * num_assets,
            'exchange': 'TEST',
        },
        index=sids,
    )


def make_jagged_equity_info(num_assets,
                            start_date,
                            first_end,
                            frequency,
                            periods_between_ends,
                            auto_close_delta):
    """
    Create a dictionary representing assets that all begin at the same start
    date, but have cascading end dates.
    Parameters
    ----------
    num_assets : int
        How many assets to create.
    start_date : pd.Timestamp
        The start date for all the assets.
    first_end : pd.Timestamp
        The date at which the first equity will end.
    frequency : str or pd.tseries.offsets.Offset (e.g. trading_day)
        Frequency used to interpret the next argument.
    periods_between_ends : int
        Starting after the first end date, end each asset every
        `frequency` * `periods_between_ends`.
    auto_close_delta : int
        The constant delta that cascades the auto_close_date
    Returns
    -------
    info : pd.DataFrame
        DataFrame representing newly-created assets.
    """
    equity_info = {}

    for sid in range(num_assets):
        equity_info[sid] = {
            'symbol': chr(ord('A') + sid),
            'start_date': start_date,
            'end_date': first_end + sid * periods_between_ends * frequency,
            'exchange': 'TEST',
            'auto_close_date':
                auto_close_delta + first_end + sid * periods_between_ends *
                frequency
        }

    return equity_info


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
    Convert the asset info dict to a dataframe of trade data for each sid, and
    write to the writer if provided. Write NaNs for locations where assets did
    not exist. Return a dict of the dataframes, keyed by sid.
    """
    trade_data = {}
    sids = asset_info.keys()
    date_field = 'day' if frequency == 'daily' else 'dt'

    price_sid_deltas = np.arange(len(sids), dtype=float) * price_step_by_sid
    price_date_deltas = np.arange(len(dates), dtype=float) * price_step_by_date
    prices = (price_sid_deltas + price_date_deltas[:, None]) + price_start

    volume_sid_deltas = np.arange(len(sids)) * volume_step_by_sid
    volume_date_deltas = np.arange(len(dates)) * volume_step_by_date
    volumes = (volume_sid_deltas + volume_date_deltas[:, None]) + volume_start

    for j, sid in enumerate(sids):
        start_date = asset_info[sid]['start_date']
        end_date = asset_info[sid]['end_date']
        # Normalize here so the we still generate non-NaN values on the minutes
        # for an asset's last trading day.
        for i, date in enumerate(dates.normalize()):
            if not (start_date <= date <= end_date):
                prices[i, j] = np.nan
                volumes[i, j] = 0

        if frequency == 'daily':
            dates_arr = [date.value for date in dates]
        else:
            dates_arr = dates

        df = pd.DataFrame({
            "open": prices[:, j],
            "high": prices[:, j],
            "low": prices[:, j],
            "close": prices[:, j],
            "volume": volumes[:, j],
            date_field: dates_arr
        })

        if frequency == 'minute':
            df = df.set_index(date_field)

        if writer:
            writer.write(sid, df)

        trade_data[sid] = df

    return trade_data


def make_future_info(first_sid,
                     root_symbols,
                     years,
                     notice_date_func,
                     expiration_date_func,
                     start_date_func,
                     month_codes=None):
    """
    Create a DataFrame representing futures for `root_symbols` during `year`.

    Generates a contract per triple of (symbol, year, month) supplied to
    `root_symbols`, `years`, and `month_codes`.

    Parameters
    ----------
    first_sid : int
        The first sid to use for assigning sids to the created contracts.
    root_symbols : list[str]
        A list of root symbols for which to create futures.
    years : list[int or str]
        Years (e.g. 2014), for which to produce individual contracts.
    notice_date_func : (Timestamp) -> Timestamp
        Function to generate notice dates from first of the month associated
        with asset month code.  Return NaT to simulate futures with no notice
        date.
    expiration_date_func : (Timestamp) -> Timestamp
        Function to generate expiration dates from first of the month
        associated with asset month code.
    start_date_func : (Timestamp) -> Timestamp, optional
        Function to generate start dates from first of the month associated
        with each asset month code.  Defaults to a start_date one year prior
        to the month_code date.
    month_codes : dict[str -> [1..12]], optional
        Dictionary of month codes for which to create contracts.  Entries
        should be strings mapped to values from 1 (January) to 12 (December).
        Default is zipline.futures.CME_CODE_TO_MONTH

    Returns
    -------
    futures_info : pd.DataFrame
        DataFrame of futures data suitable for passing to an
        AssetDBWriterFromDataFrame.
    """
    if month_codes is None:
        month_codes = CME_CODE_TO_MONTH

    year_strs = list(map(str, years))
    years = [pd.Timestamp(s, tz='UTC') for s in year_strs]

    # Pairs of string/date like ('K06', 2006-05-01)
    contract_suffix_to_beginning_of_month = tuple(
        (month_code + year_str[-2:], year + MonthBegin(month_num))
        for ((year, year_str), (month_code, month_num))
        in product(
            zip(years, year_strs),
            iteritems(month_codes),
        )
    )

    contracts = []
    parts = product(root_symbols, contract_suffix_to_beginning_of_month)
    for sid, (root_sym, (suffix, month_begin)) in enumerate(parts, first_sid):
        contracts.append({
            'sid': sid,
            'root_symbol': root_sym,
            'symbol': root_sym + suffix,
            'start_date': start_date_func(month_begin),
            'notice_date': notice_date_func(month_begin),
            'expiration_date': notice_date_func(month_begin),
            'multiplier': 500,
        })
    return pd.DataFrame.from_records(contracts, index='sid').convert_objects()


def make_commodity_future_info(first_sid,
                               root_symbols,
                               years,
                               month_codes=None):
    """
    Make futures testing data that simulates the notice/expiration date
    behavior of physical commodities like oil.

    Parameters
    ----------
    first_sid : int
    root_symbols : list[str]
    years : list[int]
    month_codes : dict[str -> int]

    Expiration dates are on the 20th of the month prior to the month code.
    Notice dates are are on the 20th two months prior to the month code.
    Start dates are one year before the contract month.

    See Also
    --------
    make_future_info
    """
    nineteen_days = pd.Timedelta(days=19)
    one_year = pd.Timedelta(days=365)
    return make_future_info(
        first_sid=first_sid,
        root_symbols=root_symbols,
        years=years,
        notice_date_func=lambda dt: dt - MonthBegin(2) + nineteen_days,
        expiration_date_func=lambda dt: dt - MonthBegin(1) + nineteen_days,
        start_date_func=lambda dt: dt - one_year,
        month_codes=month_codes,
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


def write_minute_data(env, tempdir, minutes, sids):
    assets = {}

    length = len(minutes)

    for sid_idx, sid in enumerate(sids):
        assets[sid] = pd.DataFrame({
            "open": (np.array(range(10, 10 + length)) + sid_idx),
            "high": (np.array(range(15, 15 + length)) + sid_idx),
            "low": (np.array(range(8, 8 + length)) + sid_idx),
            "close": (np.array(range(10, 10 + length)) + sid_idx),
            "volume": np.array(range(100, 100 + length)) + sid_idx,
            "dt": minutes
        }).set_index("dt")

    write_bcolz_minute_data(
        env,
        env.days_in_range(minutes[0], minutes[-1]),
        tempdir.path,
        assets
    )

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


def create_data_portal(env, tempdir, sim_params, sids, adjustment_reader=None):
    if sim_params.data_frequency == "daily":
        daily_path = write_daily_data(tempdir, sim_params, sids)

        equity_daily_reader = BcolzDailyBarReader(daily_path)

        return DataPortal(
            env,
            equity_daily_reader=equity_daily_reader,
            adjustment_reader=adjustment_reader
        )
    else:
        minutes = env.minutes_for_days_in_range(
            sim_params.first_open,
            sim_params.last_close
        )

        minute_path = write_minute_data(env, tempdir, minutes, sids)

        equity_minute_reader = BcolzMinuteBarReader(minute_path)

        return DataPortal(
            env,
            equity_minute_reader=equity_minute_reader,
            adjustment_reader=adjustment_reader
        )


def write_bcolz_minute_data(env, days, path, df_dict):
    market_opens = env.open_and_closes.market_open.loc[days]

    writer = BcolzMinuteBarWriter(
        days[0],
        path,
        market_opens,
        US_EQUITIES_MINUTES_PER_DAY
    )

    for sid, df in iteritems(df_dict):
        writer.write(sid, df)


def write_minute_data_for_asset(env, writer, start_dt, end_dt, sid,
                                interval=1, start_val=1):

    asset_minutes = env.minutes_for_days_in_range(start_dt, end_dt)
    minutes_count = len(asset_minutes)
    minutes_arr = np.array(range(start_val, start_val + minutes_count))

    df = pd.DataFrame({
        "open": minutes_arr + 1,
        "high": minutes_arr + 2,
        "low": minutes_arr - 1,
        "close": minutes_arr,
        "volume": 100 * minutes_arr,
        "dt": asset_minutes
    }).set_index("dt")

    if interval > 1:
        counter = 0
        while counter < len(minutes_arr):
            df[counter:(counter + interval - 1)] = 0
            counter += interval

    writer.write(sid, df)


def create_daily_df_for_asset(env, start_day, end_day, interval=1):
    days = env.days_in_range(start_day, end_day)
    days_count = len(days)
    days_arr = np.array(range(2, days_count + 2))

    df = pd.DataFrame({
        "open": days_arr + 1,
        "high": days_arr + 2,
        "low": days_arr - 1,
        "close": days_arr,
        "volume": days_arr * 100,
        "day": [day.value for day in days]
    })

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

        equity_daily_reader = BcolzDailyBarReader(path)

        return DataPortal(
            env,
            equity_daily_reader=equity_daily_reader,
        )
    else:
        minutes = env.minutes_for_days_in_range(
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
            env,
            env.days_in_range(
                sim_params.first_open,
                sim_params.last_close
            ),
            tempdir.path,
            assets
        )

        equity_minute_reader = BcolzMinuteBarReader(tempdir.path)

        return DataPortal(
            env,
            equity_minute_reader=equity_minute_reader,
        )


class FakeDataPortal(DataPortal):

    def __init__(self, env=None):
        if env is None:
            env = TradingEnvironment()

        super(FakeDataPortal, self).__init__(env)

    def get_spot_value(self, asset, field, dt, data_frequency):
        return 1.0


class FetcherDataPortal(DataPortal):
    """
    Mock dataportal that returns fake data for history and non-fetcher
    spot value.
    """
    def __init__(self, env):
        super(FetcherDataPortal, self).__init__(env)

    def get_spot_value(self, asset, field, dt, data_frequency):
        # if this is a fetcher field, exercise the regular code path
        if self._check_extra_sources(asset, field, (dt or self.current_dt)):
            return super(FetcherDataPortal, self).get_spot_value(
                asset, field, dt, data_frequency)

        # otherwise just return a fixed value
        return int(asset)

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
    data : pd.DataFrame, optional
        The data to feed to the writer. By default this maps:
        ('A', 'B', 'C') -> map(ord, 'ABC')
    """
    def __init__(self, **frames):
        self._eng = None
        if not frames:
            frames = {
                'equities': make_simple_equity_info(
                    list(map(ord, 'ABC')),
                    pd.Timestamp(0),
                    pd.Timestamp('2015'),
                )
            }
        self._data = AssetDBWriterFromDataFrame(**frames)

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
    def __init__(self, finder_cls=AssetFinder, **frames):
        self._finder_cls = finder_cls
        super(tmp_asset_finder, self).__init__(**frames)

    def __enter__(self):
        return self._finder_cls(super(tmp_asset_finder, self).__enter__())


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
    def spot_price(self, col, sid, dt):
        return 100


def create_mock_adjustments(tempdir, days, splits=None, dividends=None,
                            mergers=None):
    path = tempdir.getpath("test_adjustments.db")

    # create a split for the last day
    writer = SQLiteAdjustmentWriter(path, days, MockDailyBarReader())
    if splits is None:
        splits = create_empty_splits_mergers_frame()
    else:
        splits = pd.DataFrame(splits)

    if mergers is None:
        mergers = create_empty_splits_mergers_frame()
    else:
        mergers = pd.DataFrame(mergers)

    if dividends is None:
        data = {
            # Hackery to make the dtypes correct on an empty frame.
            'ex_date': np.array([], dtype='datetime64[ns]'),
            'pay_date': np.array([], dtype='datetime64[ns]'),
            'record_date': np.array([], dtype='datetime64[ns]'),
            'declared_date': np.array([], dtype='datetime64[ns]'),
            'amount': np.array([], dtype=float),
            'sid': np.array([], dtype=int),
        }
        dividends = pd.DataFrame(
            data,
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


def num_days_in_range(dates, start, end):
    """
    Return the number of days in `dates` between start and end, inclusive.
    """
    start_idx, stop_idx = dates.slice_locs(start, end)
    return stop_idx - start_idx


def gen_calendars(start, stop, critical_dates):
    """
    Generate calendars to use as inputs.
    """
    all_dates = pd.date_range(start, stop, tz='utc')
    for to_drop in map(list, powerset(critical_dates)):
        # Have to yield tuples.
        yield (all_dates.drop(to_drop),)

    # Also test with the trading calendar.
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
    get_loader = lambda column: loader

    with tmp_asset_finder(equities=equity_info) as finder:
        yield SimplePipelineEngine(get_loader, calendar, finder)


def parameter_space(**params):
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

        param_sets = product(*(params[name] for name in argnames))
        return subtest(param_sets, *argnames)(f)
    return decorator


def create_empty_splits_mergers_frame():
    return pd.DataFrame(
        {
            # Hackery to make the dtypes correct on an empty frame.
            'effective_date': np.array([], dtype=int),
            'ratio': np.array([], dtype=float),
            'sid': np.array([], dtype=int),
        },
        index=pd.DatetimeIndex([]),
        columns=['effective_date', 'ratio', 'sid'],
    )


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
