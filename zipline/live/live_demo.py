from __future__ import print_function

"""
This code is used to demonstrate how to run a Zipline backtest from within code.  (not using the command line tool)
The reason for doing is using Python to drive the running of multiple backtests.

A good reference is the zipline code that actually runs the backtest:
   _run() in zipline/utils/run_algo.py
"""
from zipline import TradingAlgorithm
from zipline.api import attach_pipeline, order_target_percent, pipeline_output, schedule_function
from zipline.api import symbol, order  # used in handle_data
from zipline.data.data_portal import DataPortal
from zipline.finance.trading import TradingEnvironment
from zipline.pipeline import Pipeline
from zipline.utils.events import date_rules
from zipline.utils.factory import create_simulation_parameters
from zipline.utils.calendars import get_calendar
from zipline.pipeline.loaders import USEquityPricingLoader
from zipline.pipeline.data.equity_pricing import USEquityPricing
from zipline.data.bundles.core import load
from zipline.pipeline.factors import RSI
from six import viewkeys

import os
import re
from time import time
import pandas as pd

CAPITAL_BASE = 1.0e6
ONE_THIRD = 1.0 / 3.0


def makeTS(date_str):
    """creates a Pandas DT object from a string"""
    return pd.Timestamp(date_str, tz='utc')


def parse_sqlite_connstr(db_URL):
    """parses out the db connection string (needed to make a TradingEnvironment"""
    _, connstr = re.split(r'sqlite:///', str(db_URL), maxsplit=1,)
    return connstr


def make_choose_loader(pl_loader):
    def cl(column):
        if column in USEquityPricing.columns:
            return pipeline_loader
        raise ValueError("No PipelineLoader registered for column %s." % column)
    return cl


if __name__ == '__main__':

    # load the bundle
    bundle_data = load('quantopian-quandl', os.environ, None)
    cal = bundle_data.equity_daily_bar_reader.trading_calendar.all_sessions
    pipeline_loader = USEquityPricingLoader(bundle_data.equity_daily_bar_reader, bundle_data.adjustment_reader)
    choose_loader = make_choose_loader(pipeline_loader)

    env = TradingEnvironment(asset_db_path=parse_sqlite_connstr(bundle_data.asset_finder.engine.url))

    data = DataPortal(
        env.asset_finder, get_calendar("NYSE"),
        first_trading_day=bundle_data.equity_minute_bar_reader.first_trading_day,
        equity_minute_reader=bundle_data.equity_minute_bar_reader,
        equity_daily_reader=bundle_data.equity_daily_bar_reader,
        adjustment_reader=bundle_data.adjustment_reader,
    )

    start = makeTS("2014-11-01"); end = makeTS("2015-11-01")  # this can go anywhere before the TradingAlgorithm

    def make_pipeline():
        rsi = RSI()
        return Pipeline(
            columns={
                'longs': rsi.top(3),
                'shorts': rsi.bottom(3),
            },)


    def rebalance(context, data):

        # Pipeline data will be a dataframe with boolean columns named 'longs' and
        # 'shorts'.
        pipeline_data = context.pipeline_data
        all_assets = pipeline_data.index

        longs = all_assets[pipeline_data.longs]
        shorts = all_assets[pipeline_data.shorts]

        # Build a 2x-leveraged, equal-weight, long-short portfolio.
        for asset in longs:
            order_target_percent(asset, ONE_THIRD)

        for asset in shorts:
            order_target_percent(asset, -ONE_THIRD)

        # Remove any assets that should no longer be in our portfolio.
        portfolio_assets = longs | shorts
        positions = context.portfolio.positions
        for asset in viewkeys(positions) - set(portfolio_assets):
            # This will fail if the asset was removed from our portfolio because it
            # was delisted.
            if data.can_trade(asset):
                order_target_percent(asset, 0)

    def initialize(context):
        attach_pipeline(make_pipeline(), 'my_pipeline')
        schedule_function(rebalance, date_rules.week_start())

    def before_trading_start(context, data):
        context.pipeline_data = pipeline_output('my_pipeline')

    def handle_data(context, data):
        order(symbol('AAPL'), 10)

    # the actual running of the backtest happens in the TradingAlgorithm object
    bt_start = time()
    perf = TradingAlgorithm(
        env=env,
        get_pipeline_loader=choose_loader,
        sim_params=create_simulation_parameters(
            start=start,
            end=end,
            capital_base=CAPITAL_BASE,
            data_frequency='daily',
        ),
        **{
            'initialize': initialize,
            'handle_data': handle_data,
            'before_trading_start': before_trading_start,
            'analyze': None,
        }
    ).run(data, overwrite_sim_params=False,)
    bt_end = time()

    print(perf.columns)
    print(perf['portfolio_value'])

    print("The backtest took %0.2f seconds to run." % (bt_end - bt_start))
    print("all done boss")