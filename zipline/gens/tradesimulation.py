import logbook

from datetime import datetime, timedelta
from numbers import Integral

from zipline import ndict

from zipline.gens.transform import stateful_transform
from zipline.finance.trading import TransactionSimulator
from zipline.finance.performance import PerformanceTracker

def trade_simulation_client(stream_in, algo, environment, sim_style):
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

    #============
    # Algo Setup
    #============

    # Initialize txn_sim's dictionary of orders here so that we can
    # reference it from within the user's algorithm.
    
    sids = algo.get_sid_filter()
    open_orders = {}

    for sid in sids:
        open_orders[sid] = []
    
    # Pipe the in stream into the transaction simulator.
    # Creates a txn field on the event containing transaction
    # information if we filled any pending orders on the event's sid.
    # TRANSACTION is None if we didn't fill any orders.
    with_txns = stateful_transform(
        stream_in,
        TransactionSimulator,
        open_orders,
        style = sim_style
    )


    # Pipe the events with transactions to perf. This will remove the
    # txn field added by TransactionSimulator and replace it with
    # a portfolio object to be passed to the user's algorithm. Also adds
    # a PERF_MESSAGE field which is usually none, but contains an update
    # message once per day.
    with_portfolio_and_perf_msg = stateful_transform(
        with_txns,
        PerformanceTracker,
        environment,
        sids
    )

    # Batch the event stream by dt to be processed by the user's algo.
    # Yields perf messages whenever it encounters them.
    perf_messages = algo_simulator(with_portfolio_and_perf_msg, sids, algo, open_orders)

    for message in perf_messages:
        yield message


def algo_simulator(stream_in, sids, algo, order_book):
    
    simulation_dt = None

    # Closure to pass into the user's algo to allow placing orders
    # into the txn_sim's dict of open orders.
    def order(sid, amount):
        assert sid in sids, "Order on invalid sid: %i" % sid
        order = ndict({
            'dt'     : simulation_dt,
            'sid'    : sid,
            'amount' : int(amount),
            'filled' : 0
        })

        # Tell the user if they try to buy 0 shares of something.
        if order.amount == 0:
            log = "requested to trade zero shares of {sid}".format(
                sid=event.sid
            )
            log.debug(log)
            return
        
        order_book[sid].append(order)
    
    # Set the algo's order method.
    algo.set_order(order)

    # Provide a logbook logging interface to user code.
    algo.set_logger(logbook.Logger("Algolog"))

    # Call user-defined initialize method before we process any
    # events.
    algo.initialize()
    
    this_snapshot_dt = None
    
    universe = ndict()
    
    for sid in sids:
        universe[sid] = ndict()
    universe.portfolio = None

    for event in stream_in:
        # Yield any perf messages received to be relayed back to the browser.
        if event.perf_message:
            yield event.perf_message

        # This should only happen for the first event we run.
        if simulation_dt == None:
            simulation_dt = event.dt
            
        # If we are currently creating a new message and this update
        # matches the message dt, update the state of the universe.

        if this_snapshot_dt != None:

            if event.dt == this_snapshot_dt:
                update_universe(event, universe)

            # If we are constructing a snapshot and we hit a new dt, call
            # handle_data and record how long it takes.
            else:
                
                start_tic = datetime.now()
                algo.handle_data(universe)
                stop_tic = datetime.now()

                # How long did you take?
                delta = stop_tic - start_tic

                # Update the simulation time.
                simulation_dt = this_snapshot_dt + delta
                
                # Update the universe with the new event.
                update_universe(event, universe)

                # If the current event is later than the simulation
                # time, update the universe and start constructing
                # another snapshot.
                if event.dt >= simulation_dt:
                    this_snapshot_dt = event.dt
                else:
                    this_snapshot_dt = None
        # We have been fastforwarding.  Update the universe
        # and check if we can start a new snapshot.
        else:
            update_universe(event, universe)
            if event.dt >= simulation_dt:
                this_snapshot_dt = event.dt

                

        
def update_universe(event, universe):

    universe.portfolio = event.portfolio
    del event['portfolio']
    
    event_sid = event.sid
    del event['sid']
        
    for field in event.keys():
        universe[event_sid][field] = event[field]
            
    
    
    
