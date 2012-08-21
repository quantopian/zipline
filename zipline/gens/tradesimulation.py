import signal
from logbook import Logger, Processor

from datetime import datetime, timedelta
from numbers import Integral
from itertools import groupby

from zipline import ndict
from zipline.utils.timeout import timeout, heartbeat, Timeout

from zipline.gens.transform import StatefulTransform
from zipline.finance.trading import TransactionSimulator
from zipline.finance.performance import PerformanceTracker
from zipline.utils.log_utils import stdout_only_pipe
from zipline.gens.utils import hash_args

log = Logger('Trade Simulation')

# TODO: make these arguments rather than global constants
INIT_TIMEOUT = 5
HEARTBEAT_INTERVAL = 1 # seconds
MAX_HEARTBEAT_INTERVALS = 15 #count

class TradeSimulationClient(object):
    """
    Generator-style class that takes the expected output of a merge, a
    user algorithm, a trading environment, and a simulator style as
    arguments.  Pipes the merge stream through a TransactionSimulator
    and a PerformanceTracker, which keep track of the current state of
    our algorithm's simulated universe. Results are fed to the user's
    algorithm, which directly inserts transactions into the
    TransactionSimulator's order book.

    TransactionSimulator maintains a dictionary from sids to the
    as-yet unfilled orders placed by the user's algorithm.  As trade
    events arrive, if the algorithm has open orders against the
    trade's sid, the simulator will fill orders up to 25% of market
    cap.  Applied transactions are added to a txn field on the event
    and forwarded to PerformanceTracker. The txn field is set to None
    on non-trade events and events that do not match any open orders.

    PerformanceTracker receives the updated event messages from
    TransactionSimulator, maintaining a set of daily and cumulative
    performance metrics for the algorithm.  The tracker removes the
    txn field from each event it receives, replacing it with a
    portfolio field to be fed into the user algo. At the end of each
    trading day, the PerformanceTracker also generates a daily
    performance report, which is appended to event's perf_report
    field.

    Fully processed events are fed to AlgorithmSimulator, which
    batches together events with the same dt field into a single
    snapshot to be fed to the algo. The portfolio object is repeatedly
    overwritten so that only the most recent snapshot of the universe
    is sent to the algo.
    """

    def __init__(self, algo, environment, sim_style):

        self.algo = algo
        self.sids = algo.get_sid_filter()
        self.environment = environment
        self.style = sim_style
        self.algo_sim = None

        self.warmup_start = self.environment.prior_day_open
        self.algo_start = self.environment.first_open

    def get_hash(self):
        """
        There should only ever be one TSC in the system, so
        we don't bother passing args into the hash.
        """
        return self.__class__.__name__ + hash_args()

    def simulate(self, stream_in):
        """
        Main generator work loop.
        """

        # Simulate filling any open orders made by the previous run of
        # the user's algorithm.  Sets the txn field to true on any
        # event that results in a filled order.
        ordering_client = StatefulTransform(
            TransactionSimulator,
            self.sids,
            style = self.style
        )
        with_filled_orders = ordering_client.transform(stream_in)

        # Pipe the events with transactions to perf. This will remove
        # the txn field added by TransactionSimulator and replace it
        # with a portfolio object to be passed to the user's
        # algorithm. Also adds a perf_message field which is usually
        # none, but contains an update message once per day.
        perf_tracker = StatefulTransform(
            PerformanceTracker,
            self.environment,
            self.sids
        )
        with_portfolio = perf_tracker.transform(with_filled_orders)

        # Pass the messages from perf along with the trading client's
        # state into the algorithm for simulation. We provide a
        # pointer to the ordering client's internal state so that the
        # algorithm can place new orders into the client's order book.
        self.algo_sim = AlgorithmSimulator(
            with_portfolio,
            ordering_client.state,
            self.algo,
            self.algo_start
        )

        # The algorithm will yield a daily_results message (as
        # calculated by the performance tracker) at the end of each
        # day.  It will also yield a risk report at the end of the
        # simulation.
        
        for message in self.algo_sim:
            yield message

class AlgorithmSimulator(object):

    def __init__(self,
                 stream_in,
                 order_book,
                 algo,
                 algo_start):

        self.stream_in = stream_in

        # ==========
        # Algo Setup
        # ==========

        # We extract the order book from the txn client so that
        # the algo can place new orders.
        self.order_book = order_book

        self.algo = algo
        self.sids = algo.get_sid_filter()
        self.algo_start = algo_start

        # Monkey patch the user algorithm to place orders in the
        # TransactionSimulator's order book and use our logger.
        self.algo.set_order(self.order)
        self.algolog = Logger("AlgoLog")
        self.algo.set_logger(self.algolog)

        # Handler for heartbeats during calls to handle_data.
        def log_heartbeats(beat_count, stackframe):
            t = beat_count * HEARTBEAT_INTERVAL
            warning = "handle_data has been processing for %i seconds" %t
            self.algolog.warn(warning)

        # Context manager that calls log_heartbeats every HEARTBEAT_INTERVAL
        # seconds, raising an exception after MAX_HEARTBEATS
        self.heartbeat_monitor = heartbeat(
            HEARTBEAT_INTERVAL,
            MAX_HEARTBEAT_INTERVALS,
            frame_handler=log_heartbeats,
            timeout_message="Too much time spent in handle_data call"
        )

        # ==============
        # Snapshot Setup
        # ==============

        # The algorithm's universe as of our most recent event.
        self.universe = ndict()

        for sid in self.sids:
            self.universe[sid] = ndict()
        self.universe.portfolio = None

        # We don't have a datetime for the current snapshot until we
        # receive a message.
        self.simulation_dt = None
        self.snapshot_dt = None

        # =============
        # Logging Setup
        # =============

        # Processor function for injecting the algo_dt into
        # user prints/logs.
        def inject_algo_dt(record):
            record.extra['algo_dt'] = self.snapshot_dt
        self.processor = Processor(inject_algo_dt)

        # This is a class, which is instantiated later
        # in run_algorithm. The class provides a generator.
        self.stdout_capture = stdout_only_pipe

        self.__generator = None

    def __iter__(self):
        return self

    def next(self):
        if self.__generator:
            return self.__generator.next()
        else:
            self.__generator = self._gen()
            return self.__generator.next()

    def order(self, sid, amount):
        """
        Closure to pass into the user's algo to allow placing orders
        into the txn_sim's dict of open orders.
        """
        assert sid in self.sids, "Order on invalid sid: %i" % sid
        order = ndict({
            'dt'     : self.simulation_dt,
            'sid'    : sid,
            'amount' : int(amount),
            'filled' : 0
        })

        # Tell the user if they try to buy 0 shares of something.
        if order.amount == 0:
            zero_message = "Requested to trade zero shares of {sid}".format(
                sid=order.sid
            )
            log.debug(zero_message)
            # Don't bother placing orders for 0 shares.
            return

        # Add non-zero orders to the order book.
        # !!!IMPORTANT SIDE-EFFECT!!!
        # This modifies the internal state of the transaction
        # simulator so that it can fill the placed order when it
        # receives its next message.
        self.order_book.place_order(order)

    def _gen(self):
        """
        Internal generator work loop.
        """
        # Capture any output of this generator to stdout and pipe it
        # to a logbook interface.  Also inject the current algo
        # snapshot time to any log record generated.
        with self.processor.threadbound(), self.stdout_capture(Logger('Print'),''):

            # Call user's initialize method with a timeout.
            with timeout(INIT_TIMEOUT, message="Call to initialize timed out"):
                self.algo.initialize()

            # Group together events with the same dt field. This depends on the
            # events already being sorted.
            for date, snapshot in groupby(self.stream_in, lambda e: e.dt):

                # Set the simulation date to be the first event we see.
                # This should only occur once, at the start of the test.
                if self.simulation_dt == None:
                    self.simulation_dt = date

                # Done message has the risk report, so we yield before exiting.
                if date == 'DONE':
                    for event in snapshot:
                        yield event.perf_message
                    break

                # We're still in the warmup period.  Use the event to
                # update our universe, but don't yield any perf messages,
                # and don't send a snapshot to handle_data.
                elif date < self.algo_start:
                    for event in snapshot:
                        del event['perf_message']
                        self.update_universe(event)

                # The algo has taken so long to process events that
                # its simulated time is later than the event time.
                # Update the universe and yield any perf messages
                # encountered, but don't call handle_data.
                elif date < self.simulation_dt:
                    for event in snapshot:
                        # Only yield if we have something interesting to say.
                        if event.perf_message != None:
                            yield event.perf_message
                        # Delete the message before updating so we don't send it
                        # to the user.
                        del event['perf_message']
                        self.update_universe(event)

                # Regular snapshot.  Update the universe and send a snapshot
                # to handle data.
                else:
                    for event in snapshot:
                        # Only yield if we have something interesting to say.
                        if event.perf_message != None:
                            yield event.perf_message
                        del event['perf_message']

                        self.update_universe(event)
                        
                    # Send the current state of the universe to the user's algo.
                    self.simulate_snapshot(date)

    def update_universe(self, event):
        """
        Update the universe with new event information.
        """
        # Update our portfolio.
        self.universe.portfolio = event.portfolio

        # Update our knowledge of this event's sid
        for field in event.keys():
            self.universe[event.sid][field] = event[field]

    def simulate_snapshot(self, date):
        """
        Run the user's algo against our current snapshot and update
        the algo's simulated time.
        """
        # Needs to be set so that we inject the proper date into algo
        # log/print lines.
        self.snapshot_dt = date
        
        start_tic = datetime.now()
        with self.heartbeat_monitor:
            self.algo.handle_data(self.universe)
        stop_tic = datetime.now()

        # How long did you take?
        delta = stop_tic - start_tic

        # Update the simulation time.
        self.simulation_dt = date + delta
