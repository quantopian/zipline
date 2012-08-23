import signal
from logbook import Logger, Processor

from datetime import datetime, timedelta
from numbers import Integral
from itertools import groupby

from zipline import ndict
from zipline.utils.timeout import Heartbeat, Timeout

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

        self.ordering_client = TransactionSimulator(self.sids, style=self.style)
        self.perf_tracker = PerformanceTracker(self.environment, self.sids)

        self.algo_start = self.environment.first_open
        self.algo_sim = AlgorithmSimulator(
            self.ordering_client,
            self.algo,
            self.algo_start
        )

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
        # the user's algorithm.  Fills the Transaction field on any
        # event that results in a filled order.
        with_filled_orders = self.ordering_client.transform(stream_in)

        # Pipe the events with transactions to perf. This will remove
        # the TRANSACTION field added by TransactionSimulator and replace it
        # with a portfolio field to be passed to the user's
        # algorithm. Also adds a perf_message field which is usually
        # none, but contains an update message once per day.
        with_portfolio = self.perf_tracker.transform(with_filled_orders)

        # Pass the messages from perf to the user's algorithm for simulation.
        # Events are batched by dt so that the algo handles all events for a
        # given timestamp at one one go.
        performance_messages = self.algo_sim.transform(with_portfolio)

        # The algorithm will yield a daily_results message (as
        # calculated by the performance tracker) at the end of each
        # day.  It will also yield a risk report at the end of the
        # simulation.
        for message in performance_messages:
            yield message

class AlgorithmSimulator(object):
    
    def __init__(self,
                 order_book,
                 algo,
                 algo_start):
        
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
        self.heartbeat_monitor = Heartbeat(
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

        # Single_use generator that uses the @contextmanager decorator
        # to monkey patch sys.stdout with a logbook interface.
        self.stdout_capture = stdout_only_pipe

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
        
    def transform(self, stream_in):
        """
        Main generator work loop.
        """
        # Capture any output of this generator to stdout and pipe it
        # to a logbook interface.  Also inject the current algo
        # snapshot time to any log record generated.
        with self.processor.threadbound(), self.stdout_capture(Logger('Print'),''):

            # Call user's initialize method with a timeout.
            with Timeout(INIT_TIMEOUT, message="Call to initialize timed out"):
                self.algo.initialize()

            # Group together events with the same dt field. This depends on the
            # events already being sorted.
            for date, snapshot in groupby(stream_in, lambda e: e.dt):

                # Set the simulation date to be the first event we see.
                # This should only occur once, at the start of the test.
                if self.simulation_dt == None:
                    self.simulation_dt = date

                # Done message has the risk report, so we yield before exiting.
                if date == 'DONE':
                    for event in snapshot:
                        yield event.perf_message
                    raise StopIteration()

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
