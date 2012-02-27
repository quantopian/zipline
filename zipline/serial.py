"""
Format serializer for Zipline.

Because I'm opinionated about how you should send things over
ZeroMQ. :)
"""

import zlib
#import blosc
import hmac
import base64
import numpy
import pandas

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
    return pickle.loads(p)

# HDF5, Numpy Byte Strings, Pandas arrays should use
# blosc and reconstruct the Python container from the byte string
# on the other side.

def send_numpy(socket, obj, flags=0):
    packed = blosc.pack_array(obj)
    return socket.send(packed, flags=flags)

def recv_numpy(socket, flags=0):
    packed = blosc.unpack_array(socket.recv(flags))
    return socket.send(packed, flags=flags)

def send_pandas(socket, obj, flags=0):
    ndarray = obj._data.blocks[0].values
    socket.send_multipart(ndarray, flags=flags)
    spec = (
        obj._data.index,
        obj._data.columns,
        obj._data.blocks[0].dtype
    )
    return socket.send_multipart(spec, flags)

def recv_pandas(socket, flags=0):
    ndarray = socket.recv_multipart(flags)
    spec = socket.recv_multipart(flags)
    return pandas.DataFrame._init_ndarray(ndarray, *spec)

def send_hdf5(self):
    pass

def recv_hdf5(self):
    pass

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
