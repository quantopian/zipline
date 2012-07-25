import logbook
import datetime

import zmq

import zipline.protocol as zp
import zipline.finance.performance as perf

from zipline.core.component import Component
from zipline.finance.trading import TransactionSimulator
from zipline.utils.protocol_utils import  ndict

from zipline.utils.log_utils import ZeroMQLogHandler, stdout_only_pipe

from logbook import Logger, NestedSetup, Processor, queues


log = logbook.Logger('TradeSimulation')


class TradeSimulationClient(Component):

    def init(self, trading_environment, sim_style, results_socket):
        self.received_count         = 0
        self.prev_dt                = None
        self.event_queue            = None
        self.txn_count              = 0
        self.order_count            = 0
        self.trading_environment    = trading_environment
        self.current_dt             = trading_environment.period_start
        self.last_iteration_dur     = datetime.timedelta(seconds=0)
        self.algorithm              = None
        self.max_wait               = datetime.timedelta(seconds=60)
        self.last_msg_dt            = datetime.datetime.utcnow()
        self.txn_sim                = TransactionSimulator(sim_style)

        self.event_data = ndict()
        self.perf = perf.PerformanceTracker(self.trading_environment)
        self.zmq_out = None
        self.results_socket = results_socket

    @property
    def get_id(self):
        return str(zp.FINANCE_COMPONENT.TRADING_CLIENT)

    def set_algorithm(self, algorithm):
        """
        :param algorithm: must implement the algorithm protocol. See
        :py:mod:`zipline.test.algorithm`
        """
        self.algorithm = algorithm
        # register the client's order method with the algorithm
        self.algorithm.set_order(self.order)
        # we need to provide the performance tracker with the
        # sids referenced in the algorithm, so portfolio can
        # initialize with all possible sids.
        self.perf.set_sids(self.algorithm.get_sid_filter())

        # N.B. Initialize is now called from open, because we
        # need to have a socket open for logging.

    def open(self):
        self.result_feed = self.connect_result()
        if self.results_socket:
            sock = self.context.socket(zmq.PUSH)
            sock.connect(self.results_socket)
            self.results_socket = sock
            self.sockets.append(sock)
            self.out_socket = sock

            self.setup_logging(sock)
            self.perf.publish_to(sock)
        self.initialize_algo()

    def initialize_algo(self):
        """ Setup loggers for algorithm and run algorithm's own
        initialize method.
        """
        self.logger = Logger("Print")
        self.algo_log = Logger("AlgoLog")
        self.algorithm.set_logger(self.algo_log)

        self.do_op(self.algorithm.initialize)

    def setup_logging(self, socket = None):
        sock = socket or self.results_socket

        self.zmq_out = ZeroMQLogHandler(
            socket = sock,
        )


        # This is a class, which is instantiated later
        # in run_algorithm. The class provides a generator.
        self.stdout_capture = stdout_only_pipe

    def do_work(self):
        # see if the poller has results for the result_feed
        if self.socks.get(self.result_feed) == self.zmq.POLLIN:

            self.last_msg_dt = datetime.datetime.utcnow()

            # get the next message from the result feed
            msg = self.result_feed.recv()

            # if the feed is done, shut 'er down
            if msg == str(zp.CONTROL_PROTOCOL.DONE):
                self.finish_simulation()
                return

            # result_feed is a merge component, so unframe accordingly
            event = zp.MERGE_UNFRAME(msg)
            self.received_count += 1
            # update performance and relay the event to the algorithm

            self.process_event(event)
            if self.perf.exceeded_max_loss:
                self.finish_simulation()

    def finish_simulation(self):
        log.info("TradeSimulation is Done")
        # signal the performance tracker that the simulation has
        # ended. Perf will internally calculate the full risk report.
        self.perf.handle_simulation_end()

        # signal Simulator, our ComponentHost, that this component is
        # done and Simulator needn't block exit on this component.
        self.signal_done()


    def process_event(self, event):
        # generate transactions, if applicable
        txn = self.txn_sim.apply_trade_to_open_orders(event)
        if txn:
            event.TRANSACTION = txn
            # track the number of transactions, for testing purposes.
            self.txn_count += 1
        else:
            event.TRANSACTION = None

        # the performance class needs to process each event, without
        # skipping. Algorithm should wait until the performance has been
        # updated, so that down stream components can safely assume that
        # performance is up to date. Note that this is done before we
        # mark the time for the algorithm's processing, thereby not
        # running the algo's clock for performance book keeping.
        self.perf.process_event(event)

        # mark the start time for client's processing of this event.
        event_start = datetime.datetime.utcnow()


        # queue the event.
        self.queue_event(event)

        # if the event is later than our current time, run the algo
        # otherwise, the algorithm has fallen behind the feed
        # and processing per event is longer than time between events.

        if event.dt >= self.current_dt:
            # compress time by moving the current_time up to the event
            # time.
            self.current_dt = event.dt
            self.run_algorithm()

        # tally the time spent on this iteration
        self.last_iteration_dur = datetime.datetime.utcnow() - event_start
        # move the algorithm's clock forward to include iteration time
        self.current_dt = self.current_dt  + self.last_iteration_dur

    def run_algorithm(self):
        """
        As per the algorithm protocol:

        - Set the current portfolio for the algorithm as per protocol.
        - Construct data based on backlog of events, send to algorithm.
        """
        data = self.get_data()

        if len(data) > 0:
            data.portfolio = self.perf.get_portfolio()

            # data injection pipeline for log rerouting
            # any fields injected here should be added to
            # LOG_EXTRA_FIELDS in zipline/protocol.py
            self.do_op(self.algorithm.handle_data, data)

    def exception_callback(self, trace):
        if self.results_socket:
            self.out_socket.send("EXCEPTION")

    def do_op(self, callable_op, *args, **kwargs):
        """ Wrap a callable operation with the zmq logbook
        handler if it exits. Also wrap the invocation of callable
        op in a try/except, and relay """
        if self.zmq_out:

            def inject_event_data(record):
                # Record the simulation time.
                record.extra['algo_dt'] = self.current_dt

            data_injector = Processor(inject_event_data)
            log_pipeline = NestedSetup([self.zmq_out,data_injector])
            with log_pipeline.threadbound(), self.stdout_capture(self.logger, ''):
                callable_op(*args, **kwargs)
            # if no log socket, just run the algo normally
        else:
            callable_op(*args, **kwargs)


    #Testing utility for log capture.
    # TODO: remove test code from here.
    def test_run_algorithm(self):
        # since open is never called from some tests we need to
        # set the logger explicitly
        self.algorithm.set_logger(self.algo_log)

        def inject_event_data(record):
            # Mock an event.dt
            record.extra['algo_dt'] = datetime.datetime.utcnow()

        data_injector = Processor(inject_event_data)
        log_pipeline = NestedSetup([self.zmq_out,
                                    #e.g. FileHandler(...)
                                    data_injector])
        with log_pipeline.threadbound(), self.stdout_capture(self.logger, ''):
            self.algorithm.handle_data('data')

    def connect_order(self):
        return self.connect_push_socket(self.addresses['order_address'])

    def order(self, sid, amount):
        order = zp.ndict({
            'dt':self.current_dt,
            'sid':sid,
            'amount':amount
        })
        self.order_count += 1
        self.perf.log_order(order)
        self.txn_sim.add_open_order(order)

    def queue_event(self, event):
        if self.event_queue == None:
            self.event_queue = []
        self.event_queue.append(event)

    def get_data(self):
        for event in self.event_queue:
            #alias the dt as datetime
            event.datetime = event.dt
            self.event_data[event['sid']] = event

        self.event_queue = []
        return self.event_data
