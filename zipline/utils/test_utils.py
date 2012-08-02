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
        test.assertTrue(a.has_key(key), "missing key at: " + label + "." + key)
        test.assertTrue(b.has_key(key), "missing key at: " + label + "." + key)
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


def drain_zipline(test, zipline):
    assert test.ctx, "method expects a valid zmq context"
    assert test.zipline_test_config, "method expects a valid test config"
    assert isinstance(test.zipline_test_config, dict)
    assert test.zipline_test_config['results_socket'], \
            "need to specify a socket address for logs/perf/risk"
    test.receiver = create_receiver(
        test.zipline_test_config['results_socket'],
        test.ctx
    )
    # Bind and connect are asynch, so allow time for bind before
    # starting the zipline (TSC connects internally).
    time.sleep(1)

    # start the simulation
    zipline.simulate(blocking=False)
    output, transaction_count = drain_receiver(test.receiver)
    # some processes will exit after the message stream is
    # finished. We block here to avoid collisions with subsequent
    # ziplines.
    for process in zipline.sim.subprocesses:
        process.join()

    return output, transaction_count

def create_receiver(socket_addr, ctx):
    receiver = ctx.socket(zmq.PULL)
    receiver.bind(socket_addr)

    return receiver

def drain_receiver(receiver):
    output = []
    transaction_count  = 0
    while True:
        msg = receiver.recv()
        if msg == str(zp.CONTROL_PROTOCOL.DONE):
            break
        else:
            update = zp.BT_UPDATE_UNFRAME(msg)
            output.append(update)
            if update['prefix'] == 'PERF':
                transaction_count += \
                    len(update['payload']['daily_perf']['transactions'])
            elif update['prefix'] == 'EXCEPTION':
                break

    receiver.close()
    del receiver

    return output, transaction_count


def assert_single_position(test, zipline):
    output, transaction_count = drain_zipline(test, zipline)

    test.assertTrue(zipline.sim.ready())
    test.assertFalse(zipline.sim.exception)

    test.assertEqual(
        test.zipline_test_config['order_count'],
        transaction_count
    )

    # the final message is the risk report, the second to
    # last is the final day's results. Positions is a list of
    # dicts.
    closing_positions = output[-2]['payload']['daily_perf']['positions']

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


def launch_component(self, component):
    proc = multiprocessing.Process(target=component.run)
    proc.start()
    self.subprocesses.append(proc)

    self.mapping[proc.pid] = component.get_id
    return proc

def gen_from_socket(socket_uri, context, unframe):
    """
    A generator that takes a socket_uri,  and yields
    messages from the poller until it gets a zp.CONTROL_PROTOCOL.DONE.
    """
    pull_socket = context.socket(zmq.PULL)
    pull_socket.connect(socket_uri)
    poller = zmq.Poller()
    poller.register(pull_socket, zmq.POLLIN)

    while True:
        socks = dict(poller.poll(1000))

        if socks.get(pull_socket) == zmq.POLLIN:
            message = pull_socket.recv()

            if message.type == zp.CONTROL_PROTOCOL.DONE:
                break
            else:
                yield unframe(message)
