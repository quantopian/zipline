"""
Format serializer for Zipline.

Because I'm opinionated about how you should send things over
ZeroMQ. :)
"""

import zlib
import hmac
import base64
#import blosc

import cPickle as pickle

# Pickle does the equivelant of builtin ``eval``. Be afraid, be
# very afraid.

def send_zipped_pickle(socket, obj, flags=0, protocol=-1):
    """
    Pickle an object, and zip the pickle before sending it.
    """
    p = pickle.dumps(obj, protocol)
    z = zlib.compress(p)
    return socket.send(z, flags=flags)

def recv_zipped_pickle(socket, flags=0, protocol=-1):
    """
    Unpickle and uncompress a received object.
    """
    z = socket.recv(flags)
    p = zlib.uncompress(z)
    return pickle.loads(p, protocol=protocol)

# Cryptographically secure wire protocol for ZeroMQ Using HMAC.

# Compare byte strings, backported from Python 3.
def byte_eq(a, b):
    return not sum(0 if x==y else 1 for x, y in zip(a, b)) and len(a) == len(b)

def send_secure(socket, data, key, flags=0):
    msg = base64.b64encode(data)
    sig = base64.b64encode(hmac.new(key, msg).digest())
    return socket.send(bytes('!') + sig + bytes('?') + msg, flags=flags)

def recv_secure(socket, data, key, flags):
    data = socket.recv(flags=flags)

    try:
        sig, msg = data.split(bytes('?'), 1)
    except ValueError:
        raise Exception('Invalid signature/message pair.')

    if byte_eq(sig[1:], base64.b64encode(hmac.new(key, msg).digest())):
        return base64.b64decode(msg)
    else:
        raise Exception('Cryptographically invalid message received')
