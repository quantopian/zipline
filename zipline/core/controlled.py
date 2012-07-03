"""
Poller logic for a component which is controlled by the monitor, this is
largely universal and thus we break it out into a seperate module and
splice it into the dispatch loops for each component instance.

Example usage::

    def do_work():
        socks = self.poll.poll()

        # Handle control events
        do_handle_control_events()

        # Handle other events
        if socks.get(socket) == zmq.POLLIN:
            ...
"""

import zmq
from zipline.core.component import Component
from zipline.protocol import CONTROL_PROTOCOL, CONTROL_FRAME, CONTROL_UNFRAME

def do_handle_control_events(cls, poller):
    assert isinstance(cls, Component)
    assert cls.control_in, 'Component does not have a control_in socket'

    # If we're in devel mode drop out because the controller
    # isn't guaranteed to be around anymore
    if cls.devel:
        import logbook
        logbook.info("Dropping out")
        return

    if poller.get(cls.control_in) == zmq.POLLIN:
        msg = cls.control_in.recv()
        event, payload = CONTROL_UNFRAME(msg)

        # ===========
        #  Heartbeat
        # ===========

        # The controller will send out a single number packed in
        # a CONTROL_FRAME with ``heartbeat`` event every
        # (n)-seconds. The component then has n seconds to
        # respond to it. If not then it will be considered as
        # malfunctioning or maybe CPU bound.

        if event == CONTROL_PROTOCOL.HEARTBEAT:
            # Heart outgoing
            heartbeat_frame = CONTROL_FRAME(
                CONTROL_PROTOCOL.OK,
                payload
            )
            # Echo back the heartbeat identifier to tell the
            # controller that this component is still alive and
            # doing work
            cls.control_out.send(heartbeat_frame)

        # =========
        # Soft Kill
        # =========

        # Try and clean up properly and send out any reports or
        # data that are done during a clean shutdown. Inform the
        # controller that we're done.
        elif event == CONTROL_PROTOCOL.SHUTDOWN:
            cls.signal_done()
            cls.shutdown()

        # =========
        # Hard Kill
        # =========

        # Just exit.
        elif event == CONTROL_PROTOCOL.KILL:
            cls.kill()
