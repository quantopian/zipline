import multiprocessing
import zmq
import time
import zipline.protocol as zp
from datetime import datetime
import blist
from zipline.utils.date_utils import EPOCH
from itertools import izip
from logbook import FileHandler


def setup_logger(test, path='/var/log/zipline/zipline.log'):
    test.log_handler = FileHandler(path)
    test.log_handler.push_application()


def teardown_logger(test):
    test.log_handler.pop_application()
    test.log_handler.close()


def check_list(test, a, b, label):
    test.assertTrue(isinstance(a, (list, blist.blist)))
    test.assertTrue(isinstance(b, (list, blist.blist)))
    i = 0
    for a_val, b_val in izip(a, b):
        check(test, a_val, b_val, label + "[" + str(i) + "]")


def check_dict(test, a, b, label):
    test.assertTrue(isinstance(a, dict))
    test.assertTrue(isinstance(b, dict))
    for key in a.keys():
        # ignore the extra fields used by dictshield
        if key in ['progress']:
            continue

        test.assertTrue(key in a, "missing key at: " + label + "." + key)
        test.assertTrue(key in b, "missing key at: " + label + "." + key)
        a_val = a[key]
        b_val = b[key]
        check(test, a_val, b_val, label + "." + key)


def check_datetime(test, a, b, label):
    test.assertTrue(isinstance(a, datetime))
    test.assertTrue(isinstance(b, datetime))
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


def drain_zipline(test, zipline, p_blocking=False):
    assert test.ctx, "method expects a valid zmq context"
    assert test.zipline_test_config, "method expects a valid test config"
    assert isinstance(test.zipline_test_config, dict)
    assert test.zipline_test_config['results_socket_uri'], \
            "need to specify a socket address for logs/perf/risk"
    test.receiver = create_receiver(
        test.zipline_test_config['results_socket_uri'],
        test.ctx
    )
    # Bind and connect are asynch, so allow time for bind before
    # starting the zipline (TSC connects internally).
    time.sleep(1)

    # start the simulation
    zipline.simulate(blocking=p_blocking)
    output, transaction_count = drain_receiver(test.receiver)
    # some processes will exit after the message stream is
    # finished. We block here to avoid collisions with subsequent
    # ziplines.
    zipline.join()

    return output, transaction_count


def create_receiver(socket_addr, ctx):
    receiver = ctx.socket(zmq.PULL)
    receiver.bind(socket_addr)

    return receiver


def drain_receiver(receiver, count=None):
    output = []
    transaction_count = 0
    msg_counter = 0
    while True:
        msg = receiver.recv()
        msg_counter += 1
        update = zp.BT_UPDATE_UNFRAME(msg)
        output.append(update)
        if update['prefix'] == 'PERF':
            transaction_count += \
                len(update['payload']['daily_perf']['transactions'])
        elif update['prefix'] == 'EXCEPTION':
            break
        elif update['prefix'] == 'DONE':
            break

        if count and msg_counter >= count:
            break

    receiver.close()
    del receiver

    return output, transaction_count


def assert_single_position(test, zipline, blocking=False):
    output, transaction_count = drain_zipline(test,
                                              zipline,
                                              p_blocking=blocking)
    test.assertEqual(output[-1]['prefix'], 'DONE')

    test.assertEqual(
        test.zipline_test_config['order_count'],
        transaction_count
    )

    # the final message is the risk report, the second to
    # last is the final day's results. Positions is a list of
    # dicts.
    perfs = [x for x in output if x['prefix'] == 'PERF']
    closing_positions = perfs[-2]['payload']['daily_perf']['positions']

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


def launch_component(component):
    proc = multiprocessing.Process(target=component.run)
    proc.start()
    return proc


def launch_monitor(monitor):
    proc = multiprocessing.Process(target=monitor.run)
    proc.start()
    return proc


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
        pass

    def get_hash(self):
        return "ExceptionTransform"

    def update(self, event):
        assert False, "An assertion message"
