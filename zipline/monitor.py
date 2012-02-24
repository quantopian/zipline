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

    def run(self, debug_step=False, stats=True):
        self.polling = True

        if self.debug or debug_step:
            return self._poll_verbose(True, stats)
        else:
            return self._poll(False, stats)

    def _poll(self, debug_step, stats):
        while self.polling:
            try:
                self.logging.info('msg')
                self.pub.send(self.pull.recv())
                #self.pub.send(self.pull.recv(copy=False))
            except KeyboardInterrupt:
                self.polling = False
                break
            except Exception as e:
                # Its common to wrap these in wildcard exceptions so
                # that we don't loose messages, ever
                self.logging.error(str(e))
                self.failed += 1
                continue

    def _poll_verbose(self, debug_step, stats):
        while self.polling:
            try:
                if debug_step:
                    msg = self.pull.recv(copy=False)
                    if self.dologging:
                        self.logging.info(msg)
                    self.pub.send(msg)
                    self.success += 1
            except KeyboardInterrupt:
                self.polling = False
                break
            except Exception as e:
                # Its common to wrap these in wildcard exceptions so
                # that we don't loose messages, ever
                self.logging.error(str(e))
                self.failed += 1
                continue

    def qos(self):
        return float(self.success) / (self.success + self.failed)

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

