"""
Ziplines are composed of multiple components connected by asynchronous
messaging. All ziplines follow a general topology of parallel sources,
datetimestamp serialization, parallel transformations, and finally sinks.
Furthermore, many ziplines have common needs. For example, all trade
simulations require a
:py:class:`~zipline.finance.trading.TradeSimulationClient`.

To establish best practices and minimize code replication, the lines module
provides complete zipline topologies. You can extend any zipline without
the need to extend the class. Simply instantiate any additional components
that you would like included in the zipline, and add them to the zipline
before invoking simulate.


        Here is a diagram of the SimulatedTrading zipline:


              +----------------------+  +------------------------+
              |    Trade History     |  |    (DataSource added   |
              |                      |  |     via add_source)    |
              |                      |  |                        |
              +--------------------+-+  +-+----------------------+
                                   |      |
                                   |      |
                                   v      v
                                  +---------+
                                  |   Feed  |  (ensures events are serialized
                                  +-+------++   in chronological order)
                                    |      |
                                    |      |
                                    v      v
               +----------------------+   +----------------------+
               | (Transforms added    |   |  (Transforms added   |
               |  via add_transform)  |   |   via add_transform) |
               +-------------------+--+   +-+--------------------+
                                   |        |
                                   |        |
                                   v        v
                                 +------------+
                                 |    Merge   | (combines original event and
                                 +------+-----+  transforms into one vector)
                                        |
                                        |
                                        V
    +---------------+     +--------------------------------+
    | Risk and Perf |     |                                |
    | Tracker       |     |     TradingSimulationClient    |
    +---------------+     |     tracks performance and     |
       ^  Trades and      |     provides API to algorithm. |
       |  simulated       |                                |
       |  transactions    +--+------------------+----------+
       |                     |      ^           |
       +---------------------+      | orders    |  frames
                                    |           |
                                    |           v
                          +---------------------------------+
                          |      Algorithm added via        |
                          |      __init__.                  |
                          +---------------------------------+
"""
import sys
import zmq
import multiprocessing

from zipline.test_algorithms import TestAlgorithm
from zipline.finance.trading import SIMULATION_STYLE
from zipline.utils.log_utils import ZeroMQLogHandler, stdout_only_pipe
from zipline.utils import factory

from zipline.test_algorithms import TestAlgorithm

from zipline.gens.composites import  \
    date_sorted_sources, merged_transforms
from zipline.gens.transform import Passthrough, StatefulTransform
from zipline.gens.tradesimulation import TradeSimulationClient as tsc
from logbook import Logger, NestedSetup, Processor

import zipline.protocol as zp


log = Logger('Lines')

class CancelSignal(Exception):
    def __init__(self):
        pass

class SimulatedTrading(object):

    def __init__(self,
            sources,
            transforms,
            algorithm,
            environment,
            style,
            results_socket_uri,
            context,
            sim_id):

        self.date_sorted = date_sorted_sources(*sources)
        self.transforms = transforms
        self.transforms.append(StatefulTransform(Passthrough))
        self.merged = merged_transforms(self.date_sorted, *self.transforms)
        self.trading_client = tsc(algorithm, environment, style)
        self.gen = self.trading_client.simulate(self.merged)
        self.results_uri = results_socket_uri
        self.results_socket = None
        self.context = context
        self.sim_id = sim_id

        # optional process if we fork simulate into an
        # independent process.
        self.proc = None
        self.logger = Logger(sim_id)


    def simulate(self, blocking=True):

        # for non-blocking,
        if blocking:
            self.run_gen()
        else:
            return self.fork_and_sim()

    def fork_and_sim(self):
        self.proc = multiprocessing.Process(target=self.run_gen)
        self.proc.start()
        return self.proc

    def run_gen(self):

        self.open()
        if self.zmq_out:

            def inject_event_data(record):
                # Record the simulation time.
                #record.extra['algo_dt'] = self.current_dt
                pass

            data_injector = Processor(inject_event_data)
            log_pipeline = NestedSetup([self.zmq_out,data_injector])
            with log_pipeline.threadbound(), self.stdout_capture(self.logger, ''):
                self.stream_results()
            # if no log socket, just run the algo normally
        else:
            self.stream_results()

    def stream_results(self):
        assert self.results_socket, \
                "Results socket must exist to stream results"
        try:
            for event in self.gen:
                if event.has_key('daily_perf'):
                    msg = zp.PERF_FRAME(event)
                else:
                    msg = zp.RISK_FRAME(event)
                self.results_socket.send(msg)

            self.signal_done()
        except Exception as exc:
            self.handle_exception(exc)
        finally:
            self.close()

    def signal_done(self):
        # notify monitor we're done
        done_frame = zp.DONE_FRAME('succes')
        self.results_socket.send(done_frame)

    def close(self):
        log.info("Closing Simulation: {id}".format(id=self.sim_id))

    def cancel(self):
        if self.proc and self.proc.is_alive():
            self.proc.terminate()
        else:
            self.gen.throw(CancelSignal())

    def handle_exception(self, exc):
        if isinstance(exc, CancelSignal):
            # signal from monitor of an orderly shutdown,
            # do nothing.
            pass
        else:
            self.signal_exception(exc)

    def signal_exception(self, exc=None):
        """
        All exceptions inside any component should boil back to
        this handler.

        Will inform the system that the component has failed and how it
        has failed.
        """
        exc_type, exc_value, exc_traceback = sys.exc_info()

        try:
            log.exception('{id} sending exception to result stream.'\
                .format(id=self.sim_id))
            msg = zp.EXCEPTION_FRAME(
                    exc_traceback,
                    exc_type.__name__,
                    exc_value.message
                )

            self.results_socket.send(msg)

        except:
            log.exception("Exception while reporting simulation exception.")


    def open(self):
        if not self.context:
            self.context = zmq.Context()
        if self.results_uri:
            sock = self.context.socket(zmq.PUSH)
            sock.connect(self.results_uri)
            self.results_socket = sock
            self.setup_logging()

    def setup_logging(self, socket = None):
        sock = socket or self.results_socket

        self.zmq_out = ZeroMQLogHandler(
            socket = sock,
        )


        # This is a class, which is instantiated later
        # in run_algorithm. The class provides a generator.
        self.stdout_capture = stdout_only_pipe

    def join(self):
        if self.proc:
            self.proc.join()

    @staticmethod
    def create_test_zipline(**config):
        """
        :param config: A configuration object that is a dict with:

            - environment - a \
              :py:class:`zipline.finance.trading.TradingEnvironment`
            - sid - an integer, which will be used as the security ID.
            - order_count - the number of orders the test algo will place,
              defaults to 100
            - order_amount - the number of shares per order, defaults to 100
            - trade_count - the number of trades to simulate, defaults to 101
              to ensure all orders are processed.
            - algorithm - optional parameter providing an algorithm. defaults
              to :py:class:`zipline.test.algorithms.TestAlgorithm`
            - trade_source - optional parameter to specify trades, if present.
              If not present :py:class:`zipline.sources.SpecificEquityTrades`
              is the source, with daily frequency in trades.
            - simulation_style: optional parameter that configures the
              :py:class:`zipline.finance.trading.TransactionSimulator`. Expects
              a SIMULATION_STYLE as defined in :py:mod:`zipline.finance.trading`
            - transforms: optional parameter that provides a list
              of StatefulTransform objects.
        """
        assert isinstance(config, dict)

        sid = config['sid']

        #--------------------
        # Trading Environment
        #--------------------
        if config.has_key('environment'):
            trading_environment = config['environment']
        else:
            trading_environment = factory.create_trading_environment()

        if config.has_key('order_count'):
            order_count = config['order_count']
        else:
            order_count = 100

        if config.has_key('order_amount'):
            order_amount = config['order_amount']
        else:
            order_amount = 100

        if config.has_key('trade_count'):
            trade_count = config['trade_count']
        else:
            # to ensure all orders are filled, we provide one more
            # trade than order
            trade_count = 101

        simulation_style = config.get('simulation_style')
        if not simulation_style:
            simulation_style = SIMULATION_STYLE.FIXED_SLIPPAGE

        zmq_context         = config.get('zmq_context', None)
        simulation_id       = config.get('simumlation_id', 'test_simulation')
        results_socket_uri  = config.get('results_socket_uri', None)

        #-------------------
        # Trade Source
        #-------------------
        sids = [sid]
        #-------------------
        if config.has_key('trade_source'):
            trade_source = config['trade_source']
        else:
            trade_source = factory.create_daily_trade_source(
                sids,
                trade_count,
                trading_environment
            )

        #-------------------
        # Transforms
        #-------------------
        transforms = config.get('transforms', [])

        #-------------------
        # Create the Algo
        #-------------------
        if config.has_key('algorithm'):
            test_algo = config['algorithm']
        else:
            test_algo = TestAlgorithm(
                sid,
                order_amount,
                order_count
            )

        #-------------------
        # Simulation
        #-------------------

        sim = SimulatedTrading(
                [trade_source],
                transforms,
                test_algo,
                trading_environment,
                simulation_style,
                results_socket_uri,
                zmq_context,
                simulation_id)
        #-------------------

        return sim
