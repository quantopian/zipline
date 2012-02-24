import zmq

class Controller(object):
    """
    A broker of sorts.
    """

    polling = False
    debug = False

    def __init__(self, pull_socket, pub_socket, context=None, logging = None):


        if not context:
            self._ctx = zmq.Context()
        else:
            self._ctx = context

        self.pull_socket = pull_socket
        self.pub_socket = pub_socket

        self.pull = self._ctx.socket(zmq.PULL)
        self.pub = self._ctx.socket(zmq.PUB)

        self.associated = [self.pull, self.pub]

        if logging:
            self.logging = logging
            self.dologging = True
        else:
            self.logging = False
            self.dologging = False

        self.success = 0
        self.failed = 0

        try:
            self.pull.bind(pull_socket)
        except zmq.ZMQError:
            raise Exception('Cannot not bind on %s' % pull_socket)

        try:
            self.pub.bind(pub_socket)
        except zmq.ZMQError:
            raise Exception('Cannot not bind on %s' % pub_socket)

    def run(self, debug=False):
        self.polling = True

        if debug:
            return self._poll(False)
        else:
            return self._poll_fast()

    def _poll_fast(self):
        """
        C version of the polling forwarder.
        """
        zmq.device(zmq.FORWARDER, self.pull, self.pub)

    def _poll(self):
        """
        Python version of the polling forwarder. With logging,
        mostly used for debugging.
        """

        while self.polling:
            try:
                self.logging.info('msg')
                self.pub.send(self.pull.recv())
                #self.pub.send(self.pull.recv(copy=False))
            except KeyboardInterrupt:
                self.polling = False
                break
            except zmq.ZMQError:
                self.polling = False
                break
            except Exception as e:
                # Its common to wrap these in wildcard exceptions so
                # that we don't loose messages, ever
                self.logging.error(str(e))
                self.failed += 1
                continue

    def message_sender(self):
        """
        Spin off a socket used for sending messages to this
        controller.
        """
        s = self._ctx.socket(zmq.PUSH)
        s.connect(self.pull_socket)
        s.setsockopt(zmq.LINGER, -1)
        self.associated.append(s)
        return s

    def message_listener(self):
        """
        Spin off a socket used for receiving messages from this
        controller.
        """
        s = self._ctx.socket(zmq.SUB)
        s.connect(self.pub_socket)
        s.setsockopt(zmq.SUBSCRIBE, '')
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

