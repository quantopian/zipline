from contextlib import contextmanager
from functools import partial
from itertools import (
    ifilter,
    product,
)
from logbook import FileHandler
from mock import patch
import operator
from zipline.finance.blotter import ORDER_STATUS
from zipline.utils import security_list

from six import (
    itervalues,
)

import os
import pandas as pd
import shutil
import tempfile


def to_utc(time_str):
    return pd.Timestamp(time_str, tz='US/Eastern').tz_convert('UTC')


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
def nullctx():
    """
    Null context manager.  Useful for conditionally adding a contextmanager in
    a single line, e.g.:

    with SomeContextManager() if some_expr else nullctx():
        do_stuff()
    """
    yield


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
    return ifilter(lambda pair: pred(*pair), product(values, repeat=2))


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
