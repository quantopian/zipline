import gevent
from gevent_zeromq import zmq

def ZmqConsole(sock_typ, socket_addr, sock_conn=None, context=None):

    context = context or zmq.Context.instance()
    socket = context.socket(zmq.PULL)
    socket.connect('tcp://127.0.0.1:3141')

    def console():
        while True:
            msg = socket.recv()
            print msg
            import pdb; pdb.set_trace()

    return gevent.spawn(console)
