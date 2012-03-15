import gevent
from gevent_zeromq import zmq

def ZmqConsole(sock_typ, socket_addr, sock_conn=None, context=None):

    context = context or zmq.Context.instance()
    socket = context.socket(zmq.PULL)
    socket.bind(socket_addr)

    def console():
        while True:
            msg = socket.recv_pyobj()
            print msg
            import pdb; pdb.set_trace()

    return gevent.spawn(console)
