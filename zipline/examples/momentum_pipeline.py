"""
A simple Pipeline algorithm that longs the top 3 stocks by RSI and shorts
the bottom 3 each day.
"""
from six import viewkeys
from zipline.api import (
    attach_pipeline,
    date_rules,
    order_target_percent,
    pipeline_output,
    record,
    schedule_function,
)
from zipline.pipeline import Pipeline
from zipline.pipeline.factors import RSI


def make_pipeline():
    rsi = RSI()
    return Pipeline(
        columns={
            'longs': rsi.top(3),
            'shorts': rsi.bottom(3),
        },
    )


def rebalance(context, data):

    # Pipeline data will be a dataframe with boolean columns named 'longs' and
    # 'shorts'.
    pipeline_data = context.pipeline_data
    all_assets = pipeline_data.index

    longs = all_assets[pipeline_data.longs]
    shorts = all_assets[pipeline_data.shorts]

    record(universe_size=len(all_assets))

    # Build a 2x-leveraged, equal-weight, long-short portfolio.
    one_third = 1.0 / 3.0
    for asset in longs:
        order_target_percent(asset, one_third)

    for asset in shorts:
        order_target_percent(asset, -one_third)

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

    # Rebalance each day.  In daily mode, this is equivalent to putting
    # `rebalance` in our handle_data, but in minute mode, it's equivalent to
    # running at the start of the day each day.
    schedule_function(rebalance, date_rules.every_day())


def before_trading_start(context, data):
    context.pipeline_data = pipeline_output('my_pipeline')


def _test_args():
    """
    Extra arguments to use when zipline's automated tests run this example.

    Notes for testers:

    Gross leverage should be roughly 2.0 on every day except the first.
    Net leverage should be roughly 2.0 on every day except the first.

    Longs Count should always be 3 after the first day.
    Shorts Count should be 3 after the first day, except on 2013-10-30, when it
    dips to 2 for a day because DELL is delisted.
    """
    import pandas as pd

    return {
        # We run through october of 2013 because DELL is in the test data and
        # it went private on 2013-10-29.
        'start': pd.Timestamp('2013-10-07', tz='utc'),
        'end': pd.Timestamp('2013-11-30', tz='utc'),
        'capital_base': 100000,
    }
