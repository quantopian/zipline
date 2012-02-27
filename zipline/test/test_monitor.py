"""
Dummy simulator backported from Qexec for development on Zipline.
"""

import logging
from multiprocessing import Process
from unittest2 import TestCase

from zipline.monitor import Controller

import gevent
from gevent_zeromq import zmq

ctx = zmq.Context()

class TestControlProtocol(TestCase):

    def setUpController(self):
        self.controller.run()

    def setUp(self):
        self.controller = Controller(
            'tcp://127.0.0.1:5000',
            'tcp://127.0.0.1:5001',
        )
        self.control_proc = Process(target=self.setUpController)
        self.control_proc.start()

    def tearDown(self):
        self.control_proc.terminate()
        ctx.destroy()

    def asyncMessage(self, socket):
        return socket.recv()

    def send_and_receive(self, push, sub, message='\x01'):
        msg = gevent.spawn(sub.recv)
        push.send(message)
        gevent.sleep(0) # explicit gevent yield
        msg.join()
        self.assertEqual(msg.value, message)

    def test_control_message(self):

        sub = self.controller.message_listener(context=ctx)
        message = gevent.spawn(self.asyncMessage, sub)

        push = self.controller.message_sender(context=ctx)

        # Don't like introducing time constants but because of
        # the way gevent scheduler works zmq will often send all
        # the messages off before the other thread even gets to
        # listening.
        self.send_and_receive(push, sub)
        sub.close()
        push.close()

    def test_control_delivery(self):
        # Assert that the number of messages sent on the wire is
        # the number of messages received, ie we don't drop any.
        # This is of course depenendent on the topology of the
        # listeners being fixed. Which normally it isn't.

        sub = self.controller.message_listener(context=ctx)
        message = gevent.spawn(self.asyncMessage, sub)

        push = self.controller.message_sender(context=ctx)

        # Don't like introducing time constants but because of
        # the way gevent scheduler works zmq will often send all
        # the messages off before the other thread even gets to
        # listening.
        for i in xrange(25):
            self.send_and_receive(push, sub)

        sub.close()
        push.close()
