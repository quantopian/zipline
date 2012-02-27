import zmq

class Controller(object):
    """
    A N to N messaging system for inter component communication.
    Ostensibly a broker of sorts. Putting messages to the broker
    is durable, if the broker goes down messages will queue up
    until the HWM and then go out when the new broker comes up.

    The other end is not durable, it is simply PUB/SUB which has
    the benefit of of allowing more fluid time evolution of the
    whole system since the messaging passing topology will not
    alter itself as a result of more nodes listening.

    The actual brokerin' is either a Python loop ( slow ) or a
    zmq.FORWARDER device ( fast ).

    :param pull_socket: Socket to subscribe to for republication of
                        published messages. The endpoint for
                        :func message_sender:.
    :param pub_socket: Socket to publish messages, the starting
                       point of :func message_listener:.
    :param logging: Logging interface for tracking broker state
        Defaults to None

    Usage::

        controller = Controller(
            'tcp://127.0.0.1:5000',
            'tcp://127.0.0.1:5001',
        )

        # typically you'd want to run this async to your main
        # program since it blocks indefinetely.
        controller.run()


        sub = self.controller.message_listener()
        push = self.controller.message_sender()

        push.send('DIE')
        sub.recv()

    """

    polling = False
    debug = False

    def __init__(self, pull_socket, pub_socket, logging = None):

        self._ctx = None

        self.associated = []

        self.pull_socket = pull_socket
        self.push_socket = pull_socket # same port
        self.pub_socket = pub_socket
        self.sub_socket = pub_socket # same port

        if logging:
            self.logging = logging
            self.dologging = True
        else:
            self.logging = False
            self.dologging = False

        self.success = 0
        self.failed = 0

    def run(self, debug=False, context=None):
        """
        Run's the loop for the broker.
        """
        self.polling = True

        if not context:
            self._ctx = zmq.Context()
        else:
            self._ctx = context

        if debug:
            return self._poll_fast() # the c loop
        else:
            return self._poll() # use a python loop

    def _poll_fast(self):
        """
        C version of the polling forwarder.
        """
        self.pull = self._ctx.socket(zmq.PULL)
        self.pub = self._ctx.socket(zmq.PUB)

        zmq.device(zmq.FORWARDER, self.pull, self.pub)

    def _poll(self):
        """
        Python version of the polling forwarder. With logging,
        mostly used for debugging.
        """

        self.pull = self._ctx.socket(zmq.PULL)
        self.pub = self._ctx.socket(zmq.PUB)

        self.pull.bind(self.pull_socket)
        self.pub.bind(self.pub_socket)

        self.associated.extend([self.pull, self.pub])

        while self.polling:
            try:
                msg = self.pull.recv()
                print msg
                self.pub.send(msg)
            except KeyboardInterrupt:
                self.polling = False
                break
            except zmq.ZMQError:
                self.polling = False
                break
            except Exception as e:
                # Its common to wrap these in wildcard exceptions so
                # that we don't loose messages, ever
                if self.logging:
                    self.logging.error(str(e))
                self.failed += 1
                continue

    # -------------------
    # Hooks for Endpoints
    # -------------------

    def message_sender(self, context=None):
        """
        Spin off a socket used for sending messages to this
        controller.
        """

        if not context:
            context = zmq.Context()

        s = context.socket(zmq.PUSH)
        s.connect(self.push_socket)
        self.associated.append(s)
        return s

    def message_listener(self, context = None, filters=None):
        """
        Spin off a socket used for receiving messages from this
        controller.
        """

        if not context:
            context = zmq.Context()

        s = context.socket(zmq.SUB)
        s.connect(self.sub_socket)
        s.setsockopt(zmq.SUBSCRIBE, filters or '')
        self.associated.append(s)
        return s

    def destroy(self):
        """
        Manual cleanup.
        """
        self.polling = False

        for asoc in self.associated:
            asoc.close()

        #if self._ctx:
            #self._ctx.destroy()

    def __del__(self):
        self.destroy()

    def qos(self):
        if not self.debug:
            return
        return float(self.success) / (self.success + self.failed)

if __name__ == '__main__':
    print 'Running on ',\
        'tcp://127.0.0.1:5000', \
        'tcp://127.0.0.1:5001',

    controller = Controller(
        'tcp://127.0.0.1:5000',
        'tcp://127.0.0.1:5001',
    )
    controller.run()
