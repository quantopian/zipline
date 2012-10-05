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

from zipline.gens.composites import (
    date_sorted_sources,
    sequential_transforms
)
from zipline.gens.tradesimulation import TradeSimulationClient as tsc

from logbook import Logger

log = Logger('Lines')


class SimulatedTrading(object):

    def __init__(self,
            sources,
            transforms,
            algorithm,
            environment
            ):
        """
        @sources - an iterable of iterables
        These iterables must yield ndicts that contain:
        - type :: a ziplines.protocol.DATASOURCE_TYPE
        - dt :: a milliseconds since epoch timestamp in UTC

        @transforms - An iterable of instances of StatefulTransform.

        @algorithm - An object that implements:
        `def initialize(self)`
        `def handle_data(self, data)`
        `def get_sid_filter(self)`
        `def set_logger(self, logger)`
        `def set_order(self, order_callable)`

        @environment - An instance of finance.trading.TradingEnvironment

        @slippage - an object with a simulate method that takes a
        trade event and returns a transaction
        """

        self.date_sorted = date_sorted_sources(*sources)
        self.transforms = transforms
        # Formerly merged_transforms.
        self.with_tnfms = sequential_transforms(self.date_sorted,
                                                *self.transforms)
        self.trading_client = tsc(algorithm, environment)

        # give the algorithm access to the simulator to control
        # state such as universe, commissions, and slippage. With
        # great power comes great responsibility.
        algorithm.simulator = self

        self.gen = self.trading_client.simulate(self.with_tnfms)

    def __iter__(self):
        return self

    def next(self):
        return self.gen.next()
