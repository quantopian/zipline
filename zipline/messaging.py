"""
Commonly used messaging components.
"""

import datetime

from collections import Counter

import zipline.util as qutil
from zipline.component import Component
import zipline.protocol as zp
from zipline.protocol import CONTROL_PROTOCOL, COMPONENT_TYPE, \
    COMPONENT_STATE, CONTROL_FRAME, CONTROL_UNFRAME

class ComponentHost(Component):
    """
    Components that can launch multiple sub-components, synchronize their
    start, and then wait for all components to be finished.
    """

    def __init__(self, addresses):
        Component.__init__(self)
        self.addresses     = addresses
        self.running       = False

        self.init()

    def init(self):
        assert hasattr(self, 'zmq_flavor'), \
        """ You must specify a flavor of ZeroMQ for all
        ComponentHost subclasses. """

        # Component Registry, keyed by get_id
        # ----------------------
        self.components     = {}
        # ----------------------
        # Internal Registry, keyed by guid
        self._components     = {}
        # ----------------------

        self.sync_register  = {}
        self.timeout        = datetime.timedelta(seconds=60)

        self.feed           = Feed()
        self.merge          = Merge()
        self.passthrough    = PassthroughTransform()
        self.controller     = None

        #register the feed and the merge
        self.register_components([self.feed, self.merge, self.passthrough])

    def register_controller(self, controller):
        """
        Add the given components to the registry. Establish
        communication with them.
        """
        if self.controller != None:
            raise Exception("There can be only one!")

        self.controller = controller
        self.controller.zmq_flavor = self.zmq_flavor

        # Propogate the controller to all the subcomponents
        for component in self.components.itervalues():
            component.controller = controller

    def register_components(self, components):
        """
        Add the given components to the registry. Establish
        communication with them.
        """
        assert isinstance(components, list)
        for component in components:

            component.addresses     = self.addresses
            component.controller    = self.controller

            # Hosts share their zmq flavor with hosted components
            component.zmq_flavor    = self.zmq_flavor

            self._components[component.guid] = component
            self.components[component.get_id] = component
            self.sync_register[component.get_id] = datetime.datetime.utcnow()

            if isinstance(component, DataSource):
                self.feed.add_source(component.get_id, component.is_blocking)
                if not component.is_blocking:
                    self.feed.ds_finished_counter +=1 
            if isinstance(component, BaseTransform):
                self.merge.add_source(component.get_id, component.is_blocking)
                if not component.is_blocking:
                    self.feed.ds_finished_counter +=1
                        
    def unregister_component(self, component_id):
        del self.components[component_id]
        del self.sync_register[component_id]

    def setup_sync(self):
        """
        Setup the sync socket and poller. ( Bind )
        """
        qutil.LOGGER.debug("Connecting sync server.")

        self.sync_socket = self.context.socket(self.zmq.REP)
        self.sync_socket.bind(self.addresses['sync_address'])

        self.sync_poller = self.zmq_poller()
        self.sync_poller.register(self.sync_socket, self.zmq.POLLIN)

        self.sockets.append(self.sync_socket)

    def open(self):
        for component in self.components.values():
            self.launch_component(component)
        self.launch_controller()

    def is_timed_out(self):
        """
        DEPRECATED, left in for compatability for now.
        """

        cur_time = datetime.datetime.utcnow()

        if len(self.components) == 0:
            qutil.LOGGER.info("Component register is empty.")
            return True

        for source, last_dt in self.sync_register.iteritems():
            if (cur_time - last_dt) > self.timeout:
                qutil.LOGGER.info(
                    "Time out for {source}. Current component registery: {reg}".
                    format(source=source, reg=self.components)
                )
                return True

        return False

    def loop(self, lockstep=True):

        while not self.is_timed_out():
            # wait for synchronization request
            socks = dict(self.sync_poller.poll(self.heartbeat_timeout)) #timeout after 2 seconds.

            if self.sync_socket in socks and socks[self.sync_socket] == self.zmq.POLLIN:
                msg = self.sync_socket.recv()

                try:
                    parts = msg.split(':')
                    sync_id, status = parts
                except ValueError as exc:
                    self.signal_exception(exc)

                if status == str(CONTROL_PROTOCOL.DONE): # TODO: other way around
                    #qutil.LOGGER.debug("{id} is DONE".format(id=sync_id))
                    self.unregister_component(sync_id)
                    self.state_flag = COMPONENT_STATE.DONE
                else:
                    self.sync_register[sync_id] = datetime.datetime.utcnow()

                #qutil.LOGGER.info("confirmed {id}".format(id=msg))
                # send synchronization reply
                self.sync_socket.send('ack', self.zmq.NOBLOCK)

    # ------------------
    # Simulation Control
    # ------------------

    def launch_controller(self, controller):
        raise NotImplementedError

    def launch_component(self, component):
        raise NotImplementedError

    def teardown_component(self, component):
        raise NotImplementedError


class Feed(Component):
    """
    Connects to N PULL sockets, publishing all messages received to a PUB
    socket.  Published messages are guaranteed to be in chronological order
    based on message property dt.  Expects to be instantiated in one execution
    context (thread, process, etc) and run in another.
    """

    def __init__(self):
        Component.__init__(self)

        self.sent_count             = 0
        self.received_count         = 0
        self.draining               = False
        self.ds_finished_counter    = 0

        # Depending on the size of this, might want to use a data
        # structure with better asymptotics.
        self.data_buffer            = {}
        
        # source_id -> integer count
        self.sent_counters          = Counter()
        self.recv_counters          = Counter()
        
        # source_id -> boolean. True is for blocking
        self.is_blocking_map = {}

    def init(self):
        pass

    @property
    def get_id(self):
        return "FEED"

    @property
    def get_type(self):
        return COMPONENT_TYPE.CONDUIT

    # -------------
    # Core Methods
    # -------------

    def open(self):
        self.pull_socket = self.bind_data()
        self.feed_socket = self.bind_feed()

    def do_work(self):
        # wait for synchronization reply from the host
        socks = dict(self.poll.poll(self.heartbeat_timeout)) 

        # TODO: Abstract this out, maybe on base component
        if self.control_in in socks and socks[self.control_in] == self.zmq.POLLIN:
            msg = self.control_in.recv()
            event, payload = CONTROL_UNFRAME(msg)

            # -- Heartbeat --
            if event == CONTROL_PROTOCOL.HEARTBEAT:
                # Heart outgoing
                heartbeat_frame = CONTROL_FRAME(
                    CONTROL_PROTOCOL.OK,
                    payload
                )
                self.control_out.send(heartbeat_frame)

            # -- Soft Kill --
            elif event == CONTROL_PROTOCOL.SHUTDOWN:
                self.done()
                self.shutdown()

            # -- Hard Kill --
            elif event == CONTROL_PROTOCOL.KILL:
                self.kill()


        if self.pull_socket in socks and socks[self.pull_socket] == self.zmq.POLLIN:
            message = self.pull_socket.recv()

            if message == str(CONTROL_PROTOCOL.DONE):
                self.ds_finished_counter += 1

                if len(self.data_buffer) == self.ds_finished_counter:
                    #drain any remaining messages in the buffer
                    qutil.LOGGER.debug("draining feed")
                    self.drain()
                    self.signal_done()
            else:
                try:
                    event = self.unframe(message)
                # deserialization error
                except zp.INVALID_DATASOURCE_FRAME as exc:
                    return self.signal_exception(exc)

                try:
                    self.append(event)
                    self.send_next()

                # Invalid message
                except zp.INVALID_DATASOURCE_FRAME as exc:
                    return self.signal_exception(exc)

    def unframe(self, msg):
        return zp.DATASOURCE_UNFRAME(msg)

    def frame(self, event):
        return zp.FEED_FRAME(event)

    # -------------
    # Flow Control
    # -------------

    def drain(self):
        """
        Send all messages in the buffer.
        """
        self.draining = True
        while self.pending_messages() > 0:
            self.send_next()

    def send_next(self):
        """
        Send the (chronologically) next message in the buffer.
        """
        if not (self.is_full() or self.draining):
            return

        event = self.next()
        if(event != None):
            self.feed_socket.send(self.frame(event), self.zmq.NOBLOCK)
            self.sent_counters[event.source_id] += 1
            self.sent_count += 1

    def append(self, event):
        """
        Add an event to the buffer for the source specified by
        source_id.
        """
        self.data_buffer[event.source_id].append(event)
        self.recv_counters[event.source_id] += 1
        self.received_count += 1

    def next(self):
        """
        Get the next message in chronological order.
        """
        if not(self.is_full() or self.draining):
            return

        cur_source = None
        earliest_source = None
        earliest_event = None
        #iterate over the queues of events from all sources 
        #(1 queue per datasource)
        for events in self.data_buffer.values():
            if len(events) == 0:
                continue
            cur_source = events
            first_in_list = events[0]
            if first_in_list.dt == None:
                #this is a filler event, discard
                events.pop(0)
                continue
                
            if (earliest_event == None) or (first_in_list.dt <= earliest_event.dt):
                earliest_event = first_in_list
                earliest_source = cur_source

        if earliest_event != None:
            return earliest_source.pop(0)

    def is_full(self):
        """
        Indicates whether the buffer has messages in buffer for
        all un-DONE, blocking sources.
        """
        for source_id, events in self.data_buffer.iteritems():
            if not self.is_blocking_map[source_id]:
                continue
                
            if len(events) == 0:
                return False
        return True

    def pending_messages(self):
        """
        Returns the count of all events from all sources in the
        buffer.
        """
        total = 0
        for events in self.data_buffer.values():
            total += len(events)
        return total

    def add_source(self, source_id, is_blocking=True):
        """
        Add a data source to the buffer.
        """
        self.data_buffer[source_id] = []
        self.is_blocking_map[source_id] = is_blocking

    def __len__(self):
        """
        Buffer's length is same as internal map holding separate
        sorted arrays of events keyed by source id.
        """
        return len(self.data_buffer)


class Merge(Feed):
    """
    Merges multiple streams of events into single messages.
    """

    def __init__(self):
        Feed.__init__(self)

        self.init()

    def init(self):
        pass

    @property
    def get_id(self):
        return "MERGE"

    @property
    def get_type(self):
        return COMPONENT_TYPE.CONDUIT

    def open(self):
        self.pull_socket = self.bind_merge()
        self.feed_socket = self.bind_result()

    def next(self):
        """Get the next merged message from the feed buffer."""
        if not (self.is_full() or self.draining):
            return
        
        if self.pending_messages() == 0:
            return

        #
        #get the raw event from the passthrough transform.
        result = self.data_buffer[zp.TRANSFORM_TYPE.PASSTHROUGH].pop(0).PASSTHROUGH
        for source, events in self.data_buffer.iteritems():
            if source == zp.TRANSFORM_TYPE.PASSTHROUGH:
                continue
            if len(events) > 0:
                cur = events.pop(0)
                result.merge(cur)
        return result
        
    def unframe(self, msg):
        return zp.TRANSFORM_UNFRAME(msg)   
    
    def frame(self, event):
        return zp.MERGE_FRAME(event)

    def append(self, event):
        """
        :param event: a namedict with one entry. key is the name of the 
        transform, value is the transformed value.
        Add an event to the buffer for the source specified by
        source_id.
        """

        self.data_buffer[event.keys()[0]].append(event)
        self.received_count += 1


class BaseTransform(Component):
    """
    Top level execution entry point for the transform

    - connects to the feed socket to subscribe to events
    - connects to the result socket (most oftened bound by a TransformsMerge) to PUSH transforms
    - processes all messages received from feed, until DONE message received
    - pushes all transforms
    - sends DONE to result socket, closes all sockets and context

    Parent class for feed transforms. Subclass and override transform
    method to create a new derived value from the combined feed.
    """

    def __init__(self, name):
        Component.__init__(self)

        self.state = {
            'name': name
        }

        self.init()

    def init(self):
        pass

    @property
    def get_id(self):
        return self.state['name']

    @property
    def get_type(self):
        return COMPONENT_TYPE.CONDUIT

    @property
    def is_blocking(self):
        return True

    def open(self):
        """
        Establishes zmq connections.
        """
        #create the feed.
        self.feed_socket = self.connect_feed()
        #create the result PUSH
        self.result_socket = self.connect_merge()

    def do_work(self):
        """
        Loops until feed's DONE message is received:

        - receive an event from the data feed
        - call transform (subclass' method) on event
        - send the transformed event

        """
        socks = dict(self.poll.poll(self.heartbeat_timeout))

        # TODO: Abstract this out, maybe on base component
        if self.control_in in socks and socks[self.control_in] == self.zmq.POLLIN:
            msg = self.control_in.recv()
            event, payload = CONTROL_UNFRAME(msg)

            # -- Heartbeat --
            if event == CONTROL_PROTOCOL.HEARTBEAT:
                # Heart outgoing
                heartbeat_frame = CONTROL_FRAME(
                    CONTROL_PROTOCOL.OK,
                    payload
                )
                self.control_out.send(heartbeat_frame)

            # -- Soft Kill --
            elif event == CONTROL_PROTOCOL.SHUTDOWN:
                self.done()
                self.shutdown()

            # -- Hard Kill --
            elif event == CONTROL_PROTOCOL.KILL:
                self.kill()

        if self.feed_socket in socks and socks[self.feed_socket] == self.zmq.POLLIN:
            message = self.feed_socket.recv()

            if message == str(CONTROL_PROTOCOL.DONE):
                self.signal_done()
                return

            try:
                event = self.unframe(message)
            except zp.INVALID_FEED_FRAME as exc:
                return self.signal_exception(exc)

            try:
                cur_state = self.transform(event)

            # This is overloaded, so it can fail in all sorts of
            # unknown ways. Its best to catch it in the
            # Transformer itself.
            except Exception as exc:
                return self.signal_exception(exc)

            try:
                transform_frame = self.frame(cur_state)
            except zp.INVALID_TRANSFORM_FRAME as exc:
                return self.signal_exception(exc)

            self.result_socket.send(transform_frame, self.zmq.NOBLOCK)
            
    def frame(self, cur_state):
        return zp.TRANSFORM_FRAME(cur_state['name'], cur_state['value'])
        
    def unframe(self, msg):
        return zp.FEED_UNFRAME(msg)
        
    def transform(self, event):
        """
        Must return the transformed value as a map with::

            {name:"name of new transform", value: "value of new field"}

        Transforms run in parallel and results are merged into a single map, so
        transform names must be unique.  Best practice is to use the self.state
        object initialized from the transform configuration, and only set the
        transformed value::

            self.state['value'] = transformed_value
        """
        raise NotImplementedError


class PassthroughTransform(BaseTransform):
    """
    A bypass transform which is also an identity transform::

            +-------+
        +---|   f   |--->
            +-------+
        +------id------->

    """

    def __init__(self):
        BaseTransform.__init__(self, "PASSTHROUGH")
        self.init()

    def init(self):
        pass

    @property
    def get_type(self):
        return COMPONENT_TYPE.CONDUIT

    #TODO, could save some cycles by skipping the _UNFRAME call and just setting value to original msg string.
    def transform(self, event):
        return {'name':zp.TRANSFORM_TYPE.PASSTHROUGH, 'value': zp.FEED_FRAME(event) }


class DataSource(Component):
    """
    Baseclass for data sources. Subclass and implement send_all - usually this
    means looping through all records in a store, converting to a dict, and
    calling send(map).
    
    Every datasource has a dict property to hold filters::
        - key -- name of the filter, e.g. SID
        - value -- a primitive representing the filter. e.g. a list of ints.
        
    Modify the datasource's filters via the set_filter(name, value)
    """
    def __init__(self, source_id):
        Component.__init__(self)

        self.id = source_id
        self.init()
        self.filter = {}

    def init(self):
        self.cur_event = None

    def set_filter(self, name, value):
        self.filter[name] = value

    @property
    def get_id(self):
        return self.id
        
    @property
    def is_blocking(self):
        return True

    @property
    def get_type(self):
        return COMPONENT_TYPE.SOURCE

    def open(self):
        self.data_socket = self.connect_data()

    def send(self, event):
        """
        Emit data.
        """
        assert isinstance(event, zp.namedict)

        event['source_id'] = self.get_id
        event['type'] = self.get_type

        try:
            ds_frame = self.frame(event)
        except zp.INVALID_DATASOURCE_FRAME as exc:
            return self.signal_exception(exc)

        self.data_socket.send(ds_frame)

    def frame(self, event):
        return zp.DATASOURCE_FRAME(event)
