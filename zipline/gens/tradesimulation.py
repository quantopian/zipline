from logbook import Logger, Processor

from datetime import datetime, timedelta
from numbers import Integral

from zipline import ndict

from zipline.gens.transform import StatefulTransform
from zipline.finance.trading import TransactionSimulator
from zipline.finance.performance import PerformanceTracker
from zipline.utils.log_utils import stdout_only_pipe
from zipline.gens.utils import hash_args

log = Logger('Trade Simulation')

class TradeSimulationClient(object):
    """
    Generator that takes the expected output of a merge, a user
    algorithm, a trading environment, and a simulator style as
    arguments.  Pipes the merge stream through a TransactionSimulator
    and a PerformanceTracker, which keep track of the current state of
    our algorithm's simulated universe. Results are fed to the user's
    algorithm, which directly inserts transactions into the
    TransactionSimulator's order book.

    TransactionSimulator maintains a dictionary from sids to the
    unfulfilled orders placed by the user's algorithm.  As trade
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

    Fully processed events are run through a batcher generator, which
    batches together events with the same dt field into a single event
    to be fed to the algo. The portfolio object is repeatedly
    overwritten so that only the most recent snapshot of the universe
    is sent to the algo.
    """

    def __init__(self, algo, environment, sim_style):

        self.algo = algo
        self.sids = algo.get_sid_filter()
        self.environment = environment
        self.style = sim_style
        self.algo_sim = None

    def get_hash(self):
        """
        There should only ever be one TSC in the system.
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
        # state into the algorithm for simulation. We provide the
        # trading client so that the algorithm can place new orders
        # into the client's order book.
        self.algo_sim = AlgorithmSimulator(
            with_portfolio,
            ordering_client.state,
            self.algo,
        )

        # The algorithm will yield a daily_results message (as
        # calculated by the performance tracker) at the end of each
        # day.  It will also yield a risk report at the end of the
        # simulation.
        for message in self.algo_sim:
            yield message

class AlgorithmSimulator(object):

    def __init__(self, stream_in, order_book, algo):

        self.stream_in = stream_in

        # ==========
        # Algo Setup
        # ==========

        # We extract the order book from the txn client so that
        # the algo can place new orders.
        self.order_book = order_book

        self.algo = algo
        self.sids = algo.get_sid_filter()

        # Monkey patch the user algorithm to place orders in the
        # TransactionSimulator's order book.
        self.algo.set_order(self.order)
        self.algo.set_logger(Logger("AlgoLog"))


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
        self.this_snapshot_dt = None

        # =============
        # Logging Setup
        # =============

        # Processor function for injecting the algo_dt into
        # user prints/logs.
        def inject_algo_dt(record):
            record.extra['algo_dt'] = self.this_snapshot_dt
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
            # Call the user's initialize method.
            self.algo.initialize()

            for event in self.stream_in:
                # Yield any perf messages received to be relayed back to
                # the browser.

                if event.perf_message:
                    yield event.perf_message
                    del event['perf_message']

                if event.dt == "DONE":
                    if self.this_snapshot_dt:
                        # StopIteration happened mid-snapshot, so we
                        # have a universe snapshot that is not yet
                        # processed by the algorithm.
                        self.simulate_current_snapshot()

                    # Break out of the loop, causing us to raise
                    # StopIteration This needs to be outside the check
                    # on self.this_snapshot_dt or else getting a DONE
                    # immediately after a snapshot finishes will cause
                    # type errors.
                    break

                # This should only happen for the first event we run.
                if self.simulation_dt == None:
                    self.simulation_dt = event.dt

                # ======================
                # Time Compression Logic
                # ======================

                if self.this_snapshot_dt != None:
                    self.update_current_snapshot(event)

                # The algorithm has been missing events because it took
                # too long processing.  Update the universe with data from
                # this event, then check if enough time has passed that we
                # can start a new snapshot.
                else:
                    self.update_universe(event)
                    if event.dt >= self.simulation_dt:
                        self.this_snapshot_dt = event.dt



    def update_current_snapshot(self, event):
        """
        Update our current snapshot of the universe. Call handle_data if
        """
        # The new event matches our snapshot dt. Just update the
        # universe and move on.
        if event.dt == self.this_snapshot_dt:
            self.update_universe(event)

        # The new event does not match our snapshot.
        else:
            self.simulate_current_snapshot()

            # Once we've finished simulating the old snapshot,
            # we can update the universe with the new event.
            self.update_universe(event)

            # The current event is later than the simulation time,
            # which means the algorithm finished quickly enough to
            # receive the new event.  Start a new snapshot with this
            # event's dt.
            if event.dt >= self.simulation_dt:
                self.this_snapshot_dt = event.dt

            # The algorithm spent enough time processing that it
            # missed the new event. Wait to start a new snapshot until
            # the events catch up to the algo's simulated dt.
            else:
                self.this_snapshot_dt = None

    def simulate_current_snapshot(self):
        """
        Run the user's algo against our current snapshot and update the algo's
        simulated time.
        """
        start_tic = datetime.now()
        self.algo.handle_data(self.universe)
        stop_tic = datetime.now()

        # How long did you take?
        delta = stop_tic - start_tic

        # Update the simulation time.
        self.simulation_dt = self.this_snapshot_dt + delta

    def update_universe(self, event):
        """
        Update the universe with new event information.
        """
        # Update our portfolio.
        self.universe.portfolio = event.portfolio

        # Update our knowledge of this event's sid
        for field in event.keys():
            self.universe[event.sid][field] = event[field]
