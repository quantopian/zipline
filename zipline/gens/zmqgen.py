import zmq
import zipline.protocol as zp

def gen_from_pull_socket(socket_uri, context, unframe):
    """
    A generator that takes a socket_uri,  and yields
    messages from the poller until it gets a zp.CONTROL_PROTOCOL.DONE.
    """
    pull_socket = context.socket(zmq.PULL)
    pull_socket.connect(socket_uri)
    poller = zmq.Poller()
    poller.register(pull_socket, zmq.POLLIN)

    return gen_from_poller(poller, pull_socket, unframe)


# this generator needs to know about the source_ids coming in via
# the poller, and need to yield DONE messages for each
# source_id.
