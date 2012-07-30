import zmq

import zipline.protocol as zp

def gen_from_zmq(poller, unframe):
    """
    A generator that takes an initialized zmq poller and yields
    messages from the poller until it gets a zp.CONTROL_PROTOCOL.DONE.
    """
    while True:
        message = poller.recv()
        if message = zp.CONTROL_PROTOCOL.DONE:
            yield "DONE"
            break
        else:
            yield unframe(message)
