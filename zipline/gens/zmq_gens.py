import zmq

import zipline.protocol as zp

def gen_from_zmq(poller, unframe, namestring):
    """
    A generator that takes an initialized zmq poller and yields
    messages from the poller until it gets a zp.CONTROL_PROTOCOL.DONE.
    """
    while True:
        message = poller.recv()
        # Done protocol should now be a message type so that 
        # done messages can also have source_ids.
        if message.type == zp.CONTROL_PROTOCOL.DONE:
            yield done_message(message.source_id)
            break
        else:
            yield unframe(message)
