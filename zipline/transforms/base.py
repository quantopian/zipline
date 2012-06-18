import logging
from zipline.core.component import Component

import zipline.protocol as zp
from zipline.protocol import CONTROL_PROTOCOL, COMPONENT_TYPE, \
    CONTROL_FRAME, CONTROL_UNFRAME

LOGGER = logging.getLogger('ZiplineLogger')

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
    def init(self):
        pass

    @property
    def get_id(self):
        return self.props['name']

    @property
    def get_type(self):
        return COMPONENT_TYPE.CONDUIT

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
                self.signal_done()
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

        Transforms run in parallel and results are merged into a
        single map, so transform names must be unique. Best practice
        is to use the self.props object initialized from the transform
        configuration, and only set the transformed value::

            self.props['value'] = transformed_value
        """
        raise NotImplementedError
