import zipline.utils.factory as factory
from zipline.testing.core import create_data_portal_from_trade_history

from zipline.test_algorithms import TestAlgorithm


def create_test_zipline(**config):
    """
       :param config: A configuration object that is a dict with:

           - sid - an integer, which will be used as the asset ID.
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
             :py:class:`zipline.gens.tradingsimulation.TransactionSimulator`.
             Expects an object with a simulate mehod, such as
             :py:class:`zipline.gens.tradingsimulation.FixedSlippage`.
             :py:mod:`zipline.finance.trading`
       """
    assert isinstance(config, dict)

    try:
        sid_list = config['sid_list']
    except KeyError:
        try:
            sid_list = [config['sid']]
        except KeyError:
            raise Exception("simfactory create_test_zipline() requires "
                            "argument 'sid_list' or 'sid'")

    concurrent_trades = config.get('concurrent_trades', False)

    if 'order_count' in config:
        order_count = config['order_count']
    else:
        order_count = 100

    if 'order_amount' in config:
        order_amount = config['order_amount']
    else:
        order_amount = 100

    # -------------------
    # Create the Algo
    # -------------------
    if 'algorithm' in config:
        test_algo = config['algorithm']
    else:
        test_algo = TestAlgorithm(
            sid_list[0],
            order_amount,
            order_count,
            sim_params=config.get('sim_params',
                                  factory.create_simulation_parameters()),
            slippage=config.get('slippage'),
            identifiers=sid_list
        )

    # -------------------
    # Trade Source
    # -------------------
    if 'skip_data' not in config:
        if 'trade_source' in config:
            trade_source = config['trade_source']
        else:
            trade_source = factory.create_daily_trade_source(
                sid_list,
                test_algo.sim_params,
                test_algo.trading_environment,
                concurrent=concurrent_trades,
            )

        trades_by_sid = {}
        for trade in trade_source:
            if trade.sid not in trades_by_sid:
                trades_by_sid[trade.sid] = []

            trades_by_sid[trade.sid].append(trade)

        data_portal = create_data_portal_from_trade_history(
            config['env'],
            config['tempdir'],
            config['sim_params'],
            trades_by_sid
        )

        test_algo.data_portal = data_portal

    # -------------------
    # Benchmark source
    # -------------------

    test_algo.benchmark_return_source = config.get('benchmark_source', None)

    # ------------------
    # generator/simulator
    sim = test_algo.get_generator()

    return sim
