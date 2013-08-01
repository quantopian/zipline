from datetime import datetime
import blist
from zipline.utils.date_utils import EPOCH
from itertools import izip_longest
from logbook import FileHandler
from zipline.finance.blotter import ORDER_STATUS


def setup_logger(test, path='test.log'):
    test.log_handler = FileHandler(path)
    test.log_handler.push_application()


def teardown_logger(test):
    test.log_handler.pop_application()
    test.log_handler.close()


def check_list(test, a, b, label):
    test.assertIsInstance(a, (list, blist.blist), "not list at: " + label)
    test.assertIsInstance(b, (list, blist.blist), "not list at: " + label)
    for i, (a_val, b_val) in enumerate(izip_longest(a, b)):
        check(test, a_val, b_val, label + "[" + str(i) + "]")


def check_dict(test, a, b, label):
    test.assertIsInstance(a, dict, "not dict at: " + label)
    test.assertIsInstance(b, dict, "not dict at: " + label)
    test.assertEqual(sorted(a), sorted(b), "different keys at: " + label)
    for key in a:
        a_val = a[key]
        b_val = b[key]
        check(test, a_val, b_val, label + "." + key)


def check_datetime(test, a, b, label):
    test.assertIsInstance(a, datetime, "not datetime at: " + label)
    test.assertIsInstance(b, datetime, "not datetime at: " + label)
    test.assertEqual(EPOCH(a), EPOCH(b), "mismatched dates " + label)


def check(test, a, b, label=None):
    """
    Check equality for arbitrarily nested dicts and lists that terminate
    in types that allow direct comparisons (string, ints, floats, datetimes)
    """
    if not label:
        label = '<root>'
    if isinstance(a, dict):
        check_dict(test, a, b, label)
    elif isinstance(a, (list, blist.blist)):
        check_list(test, a, b, label)
    elif isinstance(a, datetime):
        check_datetime(test, a, b, label)
    else:
        test.assertEqual(a, b, "mismatch on path: " + label)


def drain_zipline(test, zipline):
    output = []
    transaction_count = 0
    msg_counter = 0
    # start the simulation
    for update in zipline:
        msg_counter += 1
        output.append(update)
        if isinstance(update, dict):
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
        if isinstance(update, dict):
            if 'orders' in update['daily_perf']:
                for order in update['daily_perf']['orders']:
                    orders_by_id[order['id']] = order

    for order in orders_by_id.itervalues():
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


class ExceptionTransform(object):

    def __init__(self):
        self.window_length = 1
        pass

    def get_hash(self):
        return "ExceptionTransform"

    def update(self, event):
        assert False, "An assertion message"
