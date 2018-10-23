import numpy as np
import pandas as pd
from trading_calendars import get_calendar
from zipline.algorithm import TradingAlgorithm
from zipline.api import order, record, symbol
from zipline.data import bundles
from zipline.data.data_portal import DataPortal
from zipline.data.loader import load_market_data
from zipline.extensions import load
from zipline.finance import commission, slippage, metrics
from zipline.finance.blotter import Blotter
from zipline.finance.trading import SimulationParameters

def initialize(context):
    context.asset = symbol('AAPL')
    context.set_commission(commission.PerShare(cost=.0075, min_trade_cost=1.0))
    context.set_slippage(slippage.VolumeShareSlippage())

def handle_data(context, data):
    order(context.asset, 10)
    record(AAPL=data.current(context.asset, 'price'))

def analyze(context=None, results=None):
    import matplotlib.pyplot as plt
    ax1 = plt.subplot(211)
    results.portfolio_value.plot(ax=ax1)
    ax1.set_ylabel('Portfolio value (USD)')
    ax2 = plt.subplot(212, sharex=ax1)
    results.AAPL.plot(ax=ax2)
    ax2.set_ylabel('AAPL price (USD)')
    plt.gcf().set_size_inches(18, 8)
    plt.show()

benchmark_returns, _ = load_market_data()
trading_calendar = get_calendar('NYSE')
bundle_data = bundles.load('quandl')
first_trading_day = bundle_data.equity_minute_bar_reader.first_trading_day
data_portal = DataPortal(
    asset_finder=bundle_data.asset_finder,
    trading_calendar=trading_calendar,
    first_trading_day=first_trading_day,
    equity_minute_reader=bundle_data.equity_minute_bar_reader,
    equity_daily_reader=bundle_data.equity_daily_bar_reader,
    adjustment_reader=bundle_data.adjustment_reader
)
metrics_set = metrics.load('default')
blotter = load(Blotter, 'default')
perf = TradingAlgorithm(
    namespace={},
    data_portal=data_portal,
    trading_calendar=trading_calendar,
    sim_params=SimulationParameters(
        start_session=pd.Timestamp(2018, 1, 1).tz_localize('UTC'),
        end_session=pd.Timestamp(2018, 10, 1).tz_localize('UTC'),
        trading_calendar=trading_calendar,
        capital_base=1000000.0,
        data_frequency='daily'
    ),
    metrics_set=metrics_set,
    blotter=blotter,
    benchmark_returns=benchmark_returns,
    initialize=initialize,
    handle_data=handle_data,
    analyze=analyze
).run()
print(perf)