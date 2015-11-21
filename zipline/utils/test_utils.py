from contextlib import contextmanager
from functools import wraps
from itertools import (
    count,
    product,
)
import operator
import os
import shutil
from string import ascii_uppercase
import tempfile

from logbook import FileHandler
from mock import patch
from numpy.testing import assert_allclose, assert_array_equal
import pandas as pd
from pandas.tseries.offsets import MonthBegin
from six import iteritems, itervalues
from six.moves import filter
from sqlalchemy import create_engine

from zipline.assets import AssetFinder
from zipline.assets.asset_writer import AssetDBWriterFromDataFrame
from zipline.assets.futures import CME_CODE_TO_MONTH
from zipline.finance.blotter import ORDER_STATUS
from zipline.utils import security_list


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
            'contract_multiplier': 500,
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
    """Construct a subtest in a unittest.

    This works by decorating a function as a subtest. The test will be run
    by iterating over the ``iterator`` and *unpacking the values into the
    function. If any of the runs fail, the result will be put into a set and
    the rest of the tests will be run. Finally, if any failed, all of the
    results will be dumped as one failure.

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
