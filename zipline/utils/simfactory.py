import zipline.utils.factory as factory

from zipline.test_algorithms import TestAlgorithm
from zipline.lines import SimulatedTrading
from zipline.finance.slippage import FixedSlippage, transact_partial
from zipline.finance.commission import PerShare

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
           - slippage: optional parameter that configures the
             :py:class:`zipline.gens.tradingsimulation.TransactionSimulator`. Expects
             an object with a simulate mehod, such as
             :py:class:`zipline.gens.tradingsimulation.FixedSlippage`.
             :py:mod:`zipline.finance.trading`
           - transforms: optional parameter that provides a list
             of StatefulTransform objects.
       """
       assert isinstance(config, dict)
       sid_list = config.get('sid_list')
       if not sid_list:
           sid = config.get('sid')
           sid_list = [sid]

       concurrent_trades = config.get('concurrent_trades', False)

       #--------------------
       # Trading Environment
       #--------------------
       if 'environment' in config:
           trading_environment = config['environment']
       else:
           trading_environment = factory.create_trading_environment()

       if 'order_count' in config:
           order_count = config['order_count']
       else:
           order_count = 100

       if 'order_amount' in config:
           order_amount = config['order_amount']
       else:
           order_amount = 100

       if 'trade_count' in config:
           trade_count = config['trade_count']
       else:
           # to ensure all orders are filled, we provide one more
           # trade than order
           trade_count = 101

       slippage = config.get('slippage', FixedSlippage())
       commission = PerShare()
       transact_method = transact_partial(slippage, commission)

       #-------------------
       # Trade Source
       #-------------------
       if 'trade_source' in config:
           trade_source = config['trade_source']
       else:
           trade_source = factory.create_daily_trade_source(
               sid_list,
               trade_count,
               trading_environment,
               concurrent=concurrent_trades
           )

       #-------------------
       # Transforms
       #-------------------
       transforms = config.get('transforms', [])

       #-------------------
       # Create the Algo
       #-------------------
       if 'algorithm' in config:
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
               transact_method
               )
       #-------------------

       return sim
