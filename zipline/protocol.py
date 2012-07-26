"""
The messaging protocol for Zipline.

Asserts are in place because any protocol error corresponds to a
programmer error so we want it to fail fast and in an obvious way
so it doesn't happen again. ZeroMQ follows the same philosophy.

Notes
=====

Msgpack
-------
Msgpack is the fastest serialization protocol in Python at the
moment. Its 100% C is typically orders of magnitude faster than
json and pickle making it awesome for ZeroMQ.

You can only serialize Python structural primitives: strings,
numeric types, dicts, tuples and lists. Any any recursive
combinations of these.

Basically every basestring in Python corresponds to valid
msgpack message since the protocol is highly error tolerant.
Just keep in mind that if you ever unpack a raw msgpack string
make sure it looks like what you intend and/or catch ValueError
and TypeError exceptions.

It also has the nice benefit of never invoking ``eval`` ( unlike
json and pickle) which is a major security boon since it is
impossible to arbitrary code for evaluation through messages.

UltraJSON
---------
For anything going to the browser UltraJSON is the fastest
serializer, its mostly C as well.

The same domain of serialization as msgpack applies: Python
structural primitives. It also has the additional constraint
that anything outside of UTF8 can cause serious problems, so if
you have a strong desire to JSON encode ancient Sanskrit
( admit it, we all do ), just say no.

Data Structures
===============

Enum
----

Classic C style enumeration::

    opts = Enum('FOO', 'BAR')

    opts.FOO # 0
    opts.BAR # 1
    opts.FOO = opts.BAR # False

Oh, and if you do this::

    protocol.Enum([1,2,3])

Your interpreter will segfault, think of this like an extreme assert.

Namedict
--------

Namedicts are dict like objects that have fields accessible by attribute lookup
as well as being indexable and iterable::

    HEARTBEAT_PROTOCOL = ndict({
        'REQ' : b'\x01',
        'REP' : b'\x02',
    })

    HEARTBEAT_PROTOCOL.REQ # syntactic sugar
    HEARTBEAT_PROTOCOL.REP # oh suga suga

    HEARTBEAT_PROTOCOL['REQ'] # classic dictionary index

Namedtuple
----------

From the standard library, namedtuples are great for specifying
containers for spec'ing data container objects::

    from collections import namedtuple

    Person = namedtuple('Person', 'name age gender')
    bob = Person(name='Bob', age=30, gender='male')

    bob.name   # 'Bob'
    bob.age    # 30
    bob.gender # male

    # The slots on the tuple are also finite and read-only. This
    # is a good thing, keeps us honest!

    bob.hobby = 'underwater archery'
    # Will raise:
    # AttributeError: 'Person' object has no attribute 'hobby'

    bob.name = 'joe'
    # Will raise:
    # AttributeError: can't set attribute

    # Namedtuples are normally read-only, but you can change the
    # internals using a private operation.
    bob._replace(gender='female')

    # You can also dump out to dictionary form:
    OrderedDict([('name', 'Bob'), ('age', 30), ('gender', 'male')])

    # Or JSON.
    json.dumps(bob._asdict())
    '{"gender":"male","age":30,"name":"Bob"}'

"""

import msgpack
import numbers
import datetime
import pytz
import traceback

from collections import namedtuple

from utils.protocol_utils import Enum, FrameExceptionFactory, ndict, namelookup
from utils.date_utils import EPOCH, UN_EPOCH

# -----------------------
# Control Protocol
# -----------------------

INVALID_CONTROL_FRAME = FrameExceptionFactory('CONTROL')

CONTROL_STATES = Enum(
    'INIT',
    'SOURCES_READY',
    'RUNNING',
    'TERMINATE',
)

CONTROL_PROTOCOL = Enum(
    'HEARTBEAT' , # 0 - req
    'SHUTDOWN'  , # 1 - req
    'KILL'      , # 2 - req
    'GO'        , #   - req

    'OK'        , # 3 - rep
    'DONE'      , # 4 - rep
    'EXCEPTION' , # 5 - rep
    'READY'     , # 6 - rep
)

def CONTROL_FRAME(event, payload):
    assert isinstance(event, int,)
    assert isinstance(payload, basestring)

    return msgpack.dumps(tuple([event, payload]))

def CONTROL_UNFRAME(msg):
    """
    A status code and a message.
    """
    assert isinstance(msg, basestring)

    try:
        event, payload = msgpack.loads(msg)
        assert isinstance(event, int)
        assert isinstance(payload, basestring)

        return event, payload
    except TypeError:
        raise INVALID_CONTROL_FRAME(msg)
    except ValueError:
        raise INVALID_CONTROL_FRAME(msg)

# -----------------------
# Component State
# -----------------------

COMPONENT_TYPE = Enum(
    'SOURCE'  , # 0
    'CONDUIT' , # 1
    'SINK'    , # 2
)

COMPONENT_STATE = Enum(
    'OK'        , # 0
    'DONE'      , # 1
    'EXCEPTION' , # 2
)

# NOFAILURE  - Component is either not running or has not failed
# ALGOEXCEPT - Exception thrown in the given algorithm
# HOSTEXCEPT - Exception thrown on our end.
# INTERRUPT  - Manually interuptted by user

COMPONENT_FAILURE = Enum(
    'NOFAILURE'  ,
    'ALGOEXCEPT' ,
    'HOSTEXCEPT' ,
    'INTERRUPT'  ,
)

BACKTEST_STATE = Enum(
    'IDLE'       ,
    'QUEUED'     ,
    'INPROGRESS' ,
    'CANCELLED'  , # cancelled ( before natural completion )
    'EXCEPTION'  , # failure ( due to unnatural causes )
    'DONE'       , # done ( naturally completed )
)

# -----------------------
# Datasource Protocol
# -----------------------

INVALID_DATASOURCE_FRAME = FrameExceptionFactory('DATASOURCE')

def DATASOURCE_FRAME(event):
    """
    Wraps any datasource payload with id and type, so that unpacking may choose
    the write UNFRAME for the payload.

    :param event: ndict with following properties

    - *ds_id* an identifier that is unique to the datasource in the context of a component host (e.g. Simulator)
    - *ds_type* a string denoting the datasource type. Must be on of:

        - TRADE
        - (others to follow soon)

    - *payload* a msgpack string carrying the payload for the frame
    """
    assert isinstance(event.source_id, basestring)
    assert isinstance(event.type, int), 'Unexpected type %s' % (event.type)

    #datasources will send sometimes send empty msgs to feel gaps
    if len(event.keys()) == 2:
        return msgpack.dumps(tuple([
            event.type,
            event.source_id,
            DATASOURCE_TYPE.EMPTY
        ]))

    if(event.type == DATASOURCE_TYPE.TRADE):
        return msgpack.dumps(tuple([
            event.type,
            event.source_id,
            TRADE_FRAME(event)
        ]))
    else:
        raise INVALID_DATASOURCE_FRAME(str(event))

def DATASOURCE_UNFRAME(msg):
    """

    Extracts payload, and calls correct UNFRAME method based on the
    datasource type passed along.

    Returns a dict containing at least:

    - source_id
    - type

    other properties are added based on the datasource type:

    - TRADE

        - sid - int security identifier
        - price - float
        - volume - int
        - dt - a datetime object

    """
    try:
        ds_type, source_id, payload = msgpack.loads(msg)
        assert isinstance(ds_type, int)

        rval = ndict({'source_id':source_id})

        if payload == DATASOURCE_TYPE.EMPTY:
            child_value = ndict({'dt':None})
        elif(ds_type == DATASOURCE_TYPE.TRADE):
            child_value = TRADE_UNFRAME(payload)
        else:
            raise INVALID_DATASOURCE_FRAME(msg)

        rval.merge(child_value)
        return rval

    except TypeError:
        raise INVALID_DATASOURCE_FRAME(msg)
    except ValueError:
        raise INVALID_DATASOURCE_FRAME(msg)

# -----------------------
# Feed Protocol
# -----------------------

INVALID_FEED_FRAME = FrameExceptionFactory('FEED')

def FEED_FRAME(event):
    """
    :param event: a ndict with at least

        - source_id
        - type
    """
    assert isinstance(event, ndict), 'unknown type %s' % str(event)
    source_id = event.source_id
    ds_type = event.type
    PACK_DATE(event)
    payload = event.as_dict()
    return msgpack.dumps(payload)

def FEED_UNFRAME(msg):
    try:
        payload = msgpack.loads(msg)
        #TODO: anything we can do to assert more about the content of the dict?
        assert isinstance(payload, dict)
        rval = ndict(payload)
        UNPACK_DATE(rval)
        return rval
    except TypeError:
        raise INVALID_FEED_FRAME(msg)
    except ValueError:
        raise INVALID_FEED_FRAME(msg)

# -----------------------
# Transform Protocol
# -----------------------

INVALID_TRANSFORM_FRAME = FrameExceptionFactory('TRANSFORM')

def TRANSFORM_FRAME(name, value):
    assert isinstance(name, basestring)
    if value == None:
        return msgpack.dumps(tuple([name, TRANSFORM_TYPE.EMPTY]))
    return msgpack.dumps(tuple([name, value]))

def TRANSFORM_UNFRAME(msg):
    """
    :rtype: ndict with <transform_name>:<transform_value>
    """
    try:

        name, value = msgpack.loads(msg)
        if(value == TRANSFORM_TYPE.EMPTY):
            return ndict({name : None})
        #TODO: anything we can do to assert more about the content of the dict?
        assert isinstance(name, basestring)
        if(name == TRANSFORM_TYPE.PASSTHROUGH):
            value = FEED_UNFRAME(value)

        return ndict({name : value})
    except TypeError:
        raise INVALID_TRANSFORM_FRAME(msg)
    except ValueError:
        raise INVALID_TRANSFORM_FRAME(msg)

# -----------------------
# Merge Protocol
# -----------------------
INVALID_MERGE_FRAME = FrameExceptionFactory('MERGE')

def MERGE_FRAME(event):
    """
    :param event: a nameddict with at least:

        - source_id
        - type
    """
    assert isinstance(event, ndict)
    PACK_DATE(event)
    payload = event.as_dict()
    return msgpack.dumps(payload)

def MERGE_UNFRAME(msg):
    try:
        payload = msgpack.loads(msg)
        #TODO: anything we can do to assert more about the content of the dict?
        assert isinstance(payload, dict)
        payload = ndict(payload)
        UNPACK_DATE(payload)
        return payload
    except TypeError:
        raise INVALID_MERGE_FRAME(msg)
    except ValueError:
        raise INVALID_MERGE_FRAME(msg)


# -----------------------
# Trades
# -----------------------
#
# - Should only be called from inside DATASOURCE_ (UN)FRAME.

def TRADE_FRAME(event):
    """
    :param event: should be a ndict with:

    - ds_id     -- the datasource id sending this trade out
    - sid       -- the security id
    - price     -- float of the price printed for the trade
    - volume    -- int for shares in the trade
    - dt        -- datetime for the trade

    """
    assert isinstance(event, ndict)
    assert event.type == DATASOURCE_TYPE.TRADE
    assert isinstance(event.sid, int)
    assert isinstance(event.price, numbers.Real)
    assert isinstance(event.volume, numbers.Integral)
    PACK_DATE(event)
    return msgpack.dumps(tuple([
        event.sid,
        event.price,
        event.volume,
        event.dt,
        event.type,
    ]))

def TRADE_UNFRAME(msg):
    try:
        packed = msgpack.loads(msg)
        sid, price, volume, dt, source_type = packed

        assert isinstance(sid, int)
        assert isinstance(price, numbers.Real)
        assert isinstance(volume, numbers.Integral)
        rval = ndict({
            'sid'       : sid,
            'price'     : price,
            'volume'    : volume,
            'dt'        : dt,
            'type'      : source_type
        })
        UNPACK_DATE(rval)
        return rval
    except TypeError:
        raise INVALID_TRADE_FRAME(msg)
    except ValueError:
        raise INVALID_TRADE_FRAME(msg)

# -----------------------
# Performance and Risk
# -----------------------

def PERF_FRAME(perf):
    """
    Frame the performance update created at the end of each simulated trading
    day. The msgpack is a tuple with the first element statically set to 'PERF'.
    Like RISK_FRAME, this method calls BT_UPDATE_FRAME internally, so that
    clients can call BT_UPDATE_UNFRAME for all messages from the backtest.

    :param perf: the dictionary created by zipline.trade_client.perf
    :rvalue: a msgpack string
    """

    #TODO: add asserts...

    assert isinstance(perf['started_at'], datetime.datetime)
    assert isinstance(perf['period_start'], datetime.datetime)
    assert isinstance(perf['period_end'], datetime.datetime)

    assert isinstance(perf['daily_perf'], dict)
    assert isinstance(perf['cumulative_perf'], dict)

    tp   = perf['daily_perf']
    cp   = perf['cumulative_perf']

    assert isinstance(tp['transactions'], list)
    # we never want to send transactions for the cumulative period.
    # performance.py should never send them, but just to be safe:
    assert not cp.has_key('transactions')
    assert isinstance(tp['positions'], list)
    assert isinstance(cp['positions'], list)
    assert isinstance(tp['period_close'], datetime.datetime)
    assert isinstance(tp['period_open'], datetime.datetime)
    assert isinstance(cp['period_close'], datetime.datetime)
    assert isinstance(cp['period_open'], datetime.datetime)

    perf['started_at']   = EPOCH(perf['started_at'])
    perf['period_start'] = EPOCH(perf['period_start'])
    perf['period_end']   = EPOCH(perf['period_end'])
    tp['period_close']   = EPOCH(tp['period_close'])
    tp['period_open']    = EPOCH(tp['period_open'])
    cp['period_close']   = EPOCH(cp['period_close'])
    cp['period_open']    = EPOCH(cp['period_open'])

    tp['transactions']   = convert_transactions(tp['transactions'])

    return BT_UPDATE_FRAME('PERF', perf)

def convert_transactions(transactions):
    results = []
    for txn in transactions:
        txn['date'] = EPOCH(txn['dt'])
        del(txn['dt'])
        results.append(txn)
    return results

def RISK_FRAME(risk):
    return BT_UPDATE_FRAME('RISK', risk)

def EXCEPTION_FRAME(exception_tb):
    stack_list = traceback.extract_tb(exception_tb)
    rlist = []
    for stack in stack_list:
        rstack = {
            'file'      : stack[0],
            'lineno'    : stack[1],
            'method'    : stack[2],
            'line'      : stack[3]
        }
        rlist.append(rstack)

    return BT_UPDATE_FRAME('EXCEPTION', rlist)

def BT_UPDATE_FRAME(prefix, payload):
    """
    Frames prepared by RISK_FRAME and PERF_FRAME methods are sent via the same
    socket. This method provides a prefix to allow for muxing the messages
    onto a single socket.
    """
    return msgpack.dumps(tuple([prefix, payload]))

def BT_UPDATE_UNFRAME(msg):
    """
    Risk, Perf, and LOG framing methods prefix the payload with
    a shorthand for their type. That way, all messages received from the socket
    can be PERF_FRAMED(), whether they are risk, perf, or log.
    """
    prefix, payload = msgpack.loads(msg, use_list=True)
    return dict(prefix=prefix, payload=payload)

# -----------------------
# Date Helpers
# -----------------------

def PACK_DATE(event):
    """
    Packs the datetime property of event into msgpack'able longs.
    This function should be called purely for its side effects.
    The event's 'dt' property is replaced by a tuple of integers

        - year, month, day, hour, minute, second, microsecond

    PACK_DATE and UNPACK_DATE are inverse operations.

    :param event: event must a ndict with a property named 'dt' that is a datetime.
    :rtype: None
    """
    assert isinstance(event.dt, datetime.datetime)
    # utc only please
    assert event.dt.tzinfo == pytz.utc
    event['dt'] = date_to_tuple(event['dt'])

def date_to_tuple(dt):
    year, month, day, hour, minute, second =  dt.timetuple()[0:6]
    micros = dt.microsecond
    return tuple([year, month, day, hour, minute, second, micros])

def UNPACK_DATE(event):
    """
    Unpacks the datetime property of event from msgpack'able longs.
    This function should be called purely for its side effects.
    The event's 'dt' property is converted to a datetime by reading and then
    combining a tuple of integers.

    UNPACK_DATE and PACK_DATE are inverse operations.

    :param tuple event: event must a ndict with:

    - a property named 'dt_tuple' that is a tuple of integers \
    representing the date and time in UTC.
    - dt_tuple must have year, month, day, hour, minute, second, and microsecond

    :rtype: None
    """
    assert isinstance(event.dt, tuple)
    assert len(event.dt) == 7
    for item in event.dt:
        assert isinstance(item, numbers.Integral)
    event.dt = tuple_to_date(event.dt)

def tuple_to_date(date_tuple):
    year, month, day, hour, minute, second, micros = date_tuple
    dt = datetime.datetime(year, month, day, hour, minute, second)
    dt = dt.replace(microsecond = micros, tzinfo = pytz.utc)
    return dt

DATASOURCE_TYPE = Enum(
    'TRADE',
    'EMPTY',
)


#Transform type needs to be a ndict to facilitate merging.
TRANSFORM_TYPE = ndict({
    'PASSTHROUGH' : 'PASSTHROUGH',
    'EMPTY'       : ''
})


FINANCE_COMPONENT = namelookup({
    'TRADING_CLIENT'   : 'TRADING_CLIENT',
    'PORTFOLIO_CLIENT' : 'PORTFOLIO_CLIENT',
})


# the simulation style enumerates the available transaction simulation
# strategies.
SIMULATION_STYLE  = Enum(
    'PARTIAL_VOLUME',
    'BUY_ALL',
    'FIXED_SLIPPAGE',
    'NOOP'
)

#Global variables for the fields we extract out of a standard logbook record.
LOG_FIELDS = set(['func_name', 'lineno', 'time', 'msg',\
                      'level', 'channel', ])
LOG_EXTRA_FIELDS = set(['algo_dt',])
LOG_DONE = "DONE"

def LOG_FRAME(payload):
    """
    Expects a dictionary of the form:
      {
       'algo_dt'   : 1199223000, #Algo simulation date.
       'time'      : 1199223001, #Realtime date of log creation.
       'func_name' : 'foo',
       'lineno'    : 46,
       'msg'   : 'Successfully disintegrated llama #3',
       'level'     :  4, #Logbook enum
       'channel'   : 'MyLogger'
      }

    Frame checks that we have all expected fields and exports an
    event/payload dict as JSON.
           """

    assert isinstance(payload, dict), \
        "LOG_FRAME expected a dict"

    assert payload.has_key('algo_dt'), \
        "LOG_FRAME with no algo_dt"
    assert payload.has_key('time'), \
        "LOG_FRAME with no time"
    assert payload.has_key('channel'),\
        "LOG_FRAME with no channel"
    assert payload.has_key('level'),\
        "LOG_FRAME with no level"
    assert payload.has_key('msg'),\
        "LOG_FRAME with no message"


    return BT_UPDATE_FRAME('LOG', payload)

def LOG_UNFRAME(msg):
    """
    msg should be a tuple of ('LOG',dict)
    """
    record = msgpack.loads(msg)
    assert isinstance(record, tuple)
    assert len(record) == 2
    assert record[0] == 'LOG'
    payload = record[1]

    return payload
